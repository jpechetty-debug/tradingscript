"""Swing research: LP + PEAD trades and 20-slot portfolio. Usage:
  PYTHONPATH=. python scripts/swing_dl.py hist.pkl
  python scripts/swing_dl_earn.py hist.pkl earn.pkl
  python scripts/swing_research2.py hist.pkl earn.pkl [members.csv]
Caveats: current constituents (survivorship), yfinance EPS estimates may be backfilled."""
import pickle, sys, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
H = pickle.load(open(sys.argv[1], "rb")); E = pickle.load(open(sys.argv[2], "rb"))
COST, SPLIT, RISK, CAP, SLOTS = 0.004, pd.Timestamp("2025-01-01"), 0.005, 0.10, 20
nifty = H.pop("^NSEI")["Close"]; mkt_ok = nifty > nifty.rolling(200).mean()
C = pd.DataFrame({k: v["Close"] for k, v in H.items()})
M = pd.DataFrame(True, index=C.index, columns=C.columns)
if len(sys.argv) > 3:   # point-in-time membership CSV: ticker,start,end
    mem = pd.read_csv(sys.argv[3], parse_dates=["start", "end"]); M[:] = False
    for r in mem.itertuples():
        t = r.ticker + ".NS"
        if t in M: M.loc[(M.index >= r.start) & (pd.isna(r.end) | (M.index < r.end)), t] = True
    print("PIT membership on; avg members with prices per day:", int(M.sum(axis=1)[M.index >= "2021-01-01"].mean()))
rs_pct = C.pct_change(126).where(M).rank(axis=1, pct=True)
def rsi(s, n):
    d = s.diff(); u = d.clip(lower=0).ewm(alpha=1/n).mean(); l = (-d.clip(upper=0)).ewm(alpha=1/n).mean(); return 100 - 100/(1+u/l)

trades = []   # (setup, ticker, signal_date, entry_date, exit_date, entry, exit, stop, priority)
for k, df in H.items():
    df = df.dropna(); c = df.Close; idx = df.index; n = len(df)
    if n < 250: continue
    tr = pd.concat([df.High-df.Low, (df.High-c.shift()).abs(), (df.Low-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().values; sma5 = c.rolling(5).mean().values
    liq = (((c*df.Volume).rolling(20).mean() > 1e8) & M[k].reindex(idx).fillna(False)).values; mk = mkt_ok.reindex(idx).fillna(False).values
    o, h, l, cv = df.Open.values, df.High.values, df.Low.values, c.values
    def run(setup, sigs, tmax, sma_exit, prio):
        busy = -1
        for i in sigs:
            if i <= busy or i >= n-1 or np.isnan(atr[i]): continue
            e = o[i+1]; st = e - 3*atr[i]; x = None
            for j in range(i+1, min(i+1+tmax, n)):
                if l[j] <= st: x = min(st, o[j]); break
                if sma_exit and cv[j] > sma5[j]: x = cv[j]; break
            else: j = min(i+tmax, n-1); x = cv[j]
            trades.append((setup, k, idx[i], idx[i+1], idx[j], e, x, st, prio[i] if isinstance(prio, dict) else prio)); busy = j
    up = ((c > c.rolling(50).mean()) & (c.rolling(50).mean() > c.rolling(200).mean())).values
    r2 = rsi(c, 2).values; rs = rs_pct[k].reindex(idx).values
    lp = np.where(mk & liq & up & (rs >= .8) & (r2 < 10))[0]
    run("LP", lp, 10, True, {i: -r2[i] for i in lp})
    ev = E.get(k)
    if ev is None: continue
    for variant, need_mkt in (("PEAD", False), ("PEAD_MKT", True)):
        sigs, pr = [], {}
        for ts, row in ev.iterrows():
            a = pd.Timestamp(ts).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
            pos = idx.searchsorted(a, side="right")          # first session strictly after announcement date
            pre = idx.searchsorted(a, side="left") - 1         # last session before announcement date
            if pos >= n or pre < 0 or not np.isfinite(row["Surprise(%)"]): continue
            if row["Surprise(%)"] >= 10 and cv[pos] > cv[pre] and liq[pos] and (mk[pos] or not need_mkt):
                sigs.append(pos); pr[pos] = row["Surprise(%)"]
        run(variant, sorted(set(sigs)), 20, False, pr)

T = pd.DataFrame(trades, columns=["setup","tk","sig","ent","ext","e","x","st","prio"])
T["ret"] = T.x/T.e - 1 - COST; T["R"] = (T.x - T.e)/(T.e - T.st)
rng = np.random.default_rng(20261005)
def ci(d):
    b = [g.ret.values for _, g in d.groupby(d.sig.dt.to_period("M"))]
    m = [np.concatenate([b[i] for i in rng.integers(0, len(b), len(b))]).mean() for _ in range(2000)]
    return np.percentile(m, [2.5, 97.5])*100
print("== per-trade (after 0.40% costs)")
for s, g in T.groupby("setup"):
    for nm, d in (("DEV", g[g.sig < SPLIT]), ("HOLD", g[g.sig >= SPLIT])):
        if len(d) < 10: print(f"{s:9s}{nm:5s}{len(d):5d} too few"); continue
        lo, hi = ci(d); pf = d.ret[d.ret>0].sum()/-d.ret[d.ret<0].sum()
        print(f"{s:9s}{nm:5s}{len(d):5d} avg {d.ret.mean()*100:+.2f}% win {(d.ret>0).mean():.0%} PF {pf:.2f} CI [{lo:+.2f},{hi:+.2f}]")
mr = T.assign(m=T.sig.dt.to_period("M")).pivot_table(index="m", columns="setup", values="ret", aggfunc="mean")
print("== monthly-return correlation\n", mr.corr().round(2))

def portfolio(setups):
    t = T[T.setup.isin(setups)].sort_values(["ent", "prio"], ascending=[True, False])
    days = nifty.index[nifty.index >= t.ent.min()]
    by_ent = {d: g for d, g in t.groupby("ent")}
    nav, cash, open_, curve = 1.0, 1.0, [], []
    for d in days:
        still = []
        for p in open_:
            if p["ext"] == d: cash += p["q"]*p["x"]*(1-COST/2)
            else: still.append(p)
        open_ = still
        mtm = cash + sum(p["q"]*H[p["tk"]].Close.get(d, p["e"]) for p in open_)
        for _, r in by_ent.get(d, pd.DataFrame()).iterrows():
            if len(open_) >= SLOTS or any(p["tk"] == r.tk for p in open_): continue
            q = min(RISK*mtm/(r.e-r.st), CAP*mtm/r.e, cash/(r.e*(1+COST/2)))
            if q <= 0: continue
            cash -= q*r.e*(1+COST/2); open_.append(dict(tk=r.tk, q=q, e=r.e, x=r.x, ext=r.ext))
        curve.append(cash + sum(p["q"]*H[p["tk"]].Close.get(d, p["e"]) for p in open_))
    return pd.Series(curve, index=days)
print("== portfolio (20 slots, 0.5% risk/trade, 10% cap)")
for combo in (["LP"], ["PEAD"], ["PEAD_MKT"], ["LP","PEAD"], ["LP","PEAD_MKT"]):
    eq = portfolio(combo); out = []
    for nm, s in (("DEV", eq[eq.index < SPLIT]), ("HOLD", eq[eq.index >= SPLIT])):
        r = s.pct_change().dropna(); yrs = len(r)/252
        cagr = (s.iloc[-1]/s.iloc[0])**(1/yrs)-1; dd = (s/s.cummax()-1).min(); sh = r.mean()/r.std()*np.sqrt(252)
        out.append(f"{nm} CAGR {cagr*100:+5.1f}% maxDD {dd*100:5.1f}% Sharpe {sh:4.2f}")
    print(f"{'+'.join(combo):12s}", " | ".join(out))
b = nifty[nifty.index >= T.ent.min()]
for nm, s in (("DEV", b[b.index < SPLIT]), ("HOLD", b[b.index >= SPLIT])):
    r = s.pct_change().dropna(); print(f"NIFTY {nm} CAGR {((s.iloc[-1]/s.iloc[0])**(252/len(r))-1)*100:+.1f}% maxDD {(s/s.cummax()-1).min()*100:.1f}% Sharpe {r.mean()/r.std()*np.sqrt(252):.2f}")

def hedge(eq, win=60, trade_cost=0.0005, roll_cost=0.0005, carry=0.0):
    """Short Nifty futures sized to trailing beta (lagged 1 day, clipped 0..1.5)."""
    r = eq.pct_change().fillna(0); m = nifty.reindex(eq.index).pct_change().fillna(0)
    beta = (r.rolling(win).cov(m) / m.rolling(win).var()).shift(1).clip(0, 1.5).fillna(0)
    cost = beta.diff().abs().fillna(beta) * trade_cost + beta * roll_cost / 21
    return (1 + r - beta * m - cost + beta * carry / 252).cumprod(), beta
print("== beta-hedged with short Nifty futures (60d beta, costs incl.)")
for CARRY, combo in ((0.0, ["LP"]), (0.055, ["LP"]), (0.055, ["LP", "PEAD_MKT"])):
    eq = portfolio(combo); h, beta = hedge(eq, carry=CARRY); out = []
    for nm, s in (("DEV", h[h.index < SPLIT]), ("HOLD", h[h.index >= SPLIT])):
        r = s.pct_change().dropna(); m = nifty.reindex(s.index).pct_change().dropna()
        cagr = (s.iloc[-1]/s.iloc[0])**(252/len(r))-1; dd = (s/s.cummax()-1).min()
        out.append(f"{nm} CAGR {cagr*100:+5.1f}% maxDD {dd*100:5.1f}% Sharpe {r.mean()/r.std()*np.sqrt(252):4.2f} corrNifty {r.corr(m):+.2f}")
    print(f"{"+".join(combo)} carry {CARRY:.1%}  avg beta {beta[beta>0].mean():.2f} |", " | ".join(out))
