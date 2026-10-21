"""Standalone LP/PEAD research using trusted local pickle snapshots.

python -m scripts.swing_research2 prices.pkl earnings.pkl [members.csv] --out PATH
python -m scripts.swing_research2 prices.pkl --lp-only --out PATH

The original signals and scheduled exit rules are retained. Portfolio replay now
uses explicit exit phases and prior-close sizing. Current constituents, stale
membership snapshots and possibly backfilled EPS estimates remain limitations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.scheduled_replay import replay_scheduled_trades  # noqa: E402

COST, SPLIT, RISK, CAP, SLOTS = 0.004, pd.Timestamp("2025-01-01"), 0.005, 0.10, 20
TRADE_COLUMNS = ["setup", "tk", "sig", "ent", "ext", "e", "x", "st", "prio", "exit_phase"]


def rsi(series: pd.Series, periods: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / periods).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / periods).mean()
    return 100 - 100 / (1 + up / down)


def generate_trades(history: dict[str, pd.DataFrame], earnings: dict[str, pd.DataFrame],
                    benchmark: pd.Series, membership: pd.DataFrame | None = None) -> pd.DataFrame:
    """Preserve the original standalone setup/overlap rules; annotate exit timing."""
    closes = pd.DataFrame({ticker: frame.Close for ticker, frame in history.items()})
    members = pd.DataFrame(True, index=closes.index, columns=closes.columns)
    if membership is not None:
        members[:] = False
        for row in membership.itertuples():
            ticker = row.ticker if row.ticker.endswith(".NS") else row.ticker + ".NS"
            if ticker in members:
                mask = (members.index >= row.start) & (pd.isna(row.end) | (members.index < row.end))
                members.loc[mask, ticker] = True
    # Explicit forward fill preserves the old pct_change default, using past data only.
    rs_percentile = closes.ffill().pct_change(126, fill_method=None).where(members).rank(axis=1, pct=True)
    market_ok = benchmark > benchmark.rolling(200).mean()
    rows: list[tuple[Any, ...]] = []
    for ticker, raw in history.items():
        frame = raw.dropna()
        close, index = frame.Close, frame.index
        count = len(frame)
        if count < 250:
            continue
        tr = pd.concat([frame.High - frame.Low, (frame.High - close.shift()).abs(),
                        (frame.Low - close.shift()).abs()], axis=1).max(axis=1)
        atr, sma5 = tr.rolling(14).mean().to_numpy(), close.rolling(5).mean().to_numpy()
        liquid = (((close * frame.Volume).rolling(20).mean() > 1e8)
                  & members[ticker].reindex(index, fill_value=False)).to_numpy()
        market = market_ok.reindex(index, fill_value=False).to_numpy()
        opening, low, values = frame.Open.to_numpy(), frame.Low.to_numpy(), close.to_numpy()

        def schedule(setup: str, signals: Sequence[int], holding: int, sma_exit: bool,
                     priorities: dict[int, float]) -> None:
            busy = -1
            for signal in signals:
                if signal <= busy or signal >= count - 1 or not np.isfinite(atr[signal]):
                    continue
                entry = opening[signal + 1]
                stop = entry - 3 * atr[signal]
                if not entry > stop > 0:
                    continue
                exit_price, phase = None, "CLOSE"
                for session in range(signal + 1, min(signal + 1 + holding, count)):
                    if low[session] <= stop:
                        exit_price = min(stop, opening[session])
                        phase = "OPEN" if opening[session] <= stop else "INTRADAY"
                        break
                    if sma_exit and values[session] > sma5[session]:
                        exit_price = values[session]
                        break
                else:
                    session = min(signal + holding, count - 1)
                    exit_price = values[session]
                rows.append((setup, ticker, index[signal], index[signal + 1], index[session],
                             entry, exit_price, stop, priorities[signal], phase))
                busy = session

        uptrend = ((close > close.rolling(50).mean()) & (close.rolling(50).mean() > close.rolling(200).mean())).to_numpy()
        rsi2 = rsi(close, 2).to_numpy()
        ranks = rs_percentile[ticker].reindex(index).to_numpy()
        pullbacks = np.where(market & liquid & uptrend & (ranks >= .8) & (rsi2 < 10))[0].tolist()
        schedule("LP", pullbacks, 10, True, {i: -float(rsi2[i]) for i in pullbacks})
        events = earnings.get(ticker)
        if events is None:
            continue
        for variant, needs_market in (("PEAD", False), ("PEAD_MKT", True)):
            signals, priorities = [], {}
            for timestamp, event in events.iterrows():
                announcement = pd.Timestamp(timestamp).tz_convert("Asia/Kolkata").tz_localize(None).normalize()
                after = int(index.searchsorted(announcement, side="right"))
                before = int(index.searchsorted(announcement, side="left")) - 1
                surprise = float(event["Surprise(%)"])
                if after >= count or before < 0 or not np.isfinite(surprise):
                    continue
                if surprise >= 10 and values[after] > values[before] and liquid[after] and (market[after] or not needs_market):
                    signals.append(after)
                    priorities[after] = surprise
            schedule(variant, sorted(set(signals)), 20, False, priorities)
    trades = pd.DataFrame(rows, columns=TRADE_COLUMNS)
    gross = trades.x - trades.e
    fees = (trades.e + trades.x) * COST / 2
    trades["ret"] = (gross - fees) / trades.e
    trades["gross_R"] = gross / (trades.e - trades.st)
    trades["R"] = (gross - fees) / (trades.e - trades.st)
    return trades


def monthly_interval(trades: pd.DataFrame) -> list[float] | None:
    if len(trades) < 10:
        return None
    blocks = [group.ret.to_numpy() for _, group in trades.groupby(trades.sig.dt.to_period("M"))]
    rng = np.random.default_rng(20261005)
    means = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]).mean() for _ in range(2000)]
    return [float(value) for value in np.percentile(means, [2.5, 97.5])]


def statistics(nav: pd.Series) -> dict[str, dict[str, Any]]:
    """Include first-session P&L and the capital at each segment's start."""
    returns = nav / nav.shift(1, fill_value=1.) - 1
    previous = nav.shift(1, fill_value=1.)
    output: dict[str, dict[str, Any]] = {}
    for label, mask in (("DEV", nav.index < SPLIT), ("HOLD", nav.index >= SPLIT)):
        segment, daily = nav.loc[mask], returns.loc[mask]
        if segment.empty:
            output[label] = {"status": "UNAVAILABLE", "sessions": 0}
            continue
        start = float(previous.loc[mask].iloc[0])
        wealth = np.r_[start, segment.to_numpy()]
        deviation = float(daily.std())
        output[label] = {
            "status": "AVAILABLE", "sessions": len(segment), "initial_nav": start,
            "final_nav": float(segment.iloc[-1]), "total_return": float(segment.iloc[-1] / start - 1),
            "cagr": float((segment.iloc[-1] / start) ** (252 / len(segment)) - 1),
            "max_drawdown": float((wealth / np.maximum.accumulate(wealth) - 1).min()),
            "sharpe": float(daily.mean() / deviation * np.sqrt(252)) if np.isfinite(deviation) and deviation > 0 else None,
        }
    return output


def hedge(equity: pd.Series, benchmark: pd.Series, *, carry: float = 0.,
          window: int = 60, trade_cost: float = 0.0005, roll_cost: float = 0.0005) -> pd.Series:
    returns = equity / equity.shift(1, fill_value=1.) - 1
    market = benchmark.pct_change(fill_method=None).reindex(equity.index).fillna(0.)
    beta = (returns.rolling(window).cov(market) / market.rolling(window).var()).shift(1).clip(0, 1.5).fillna(0.)
    fees = beta.diff().abs().fillna(beta) * trade_cost + beta * roll_cost / 21
    return (1 + returns - beta * market - fees + beta * carry / 252).cumprod()


def load_snapshot(path: Path) -> dict[str, pd.DataFrame]:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, pd.DataFrame) for k, v in value.items()):
        raise ValueError(f"Expected a ticker-to-DataFrame snapshot: {path}")
    return dict(value)


def observation_calendar(history: dict[str, pd.DataFrame], benchmark: pd.Series,
                         trades: pd.DataFrame) -> pd.DatetimeIndex:
    """Preserve observed stock sessions even when the benchmark has a data gap."""
    dates = benchmark.index
    for frame in history.values():
        dates = dates.union(frame.index)
    return dates[dates >= trades.ent.min()] if len(trades) else dates


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prices", type=Path)
    parser.add_argument("earnings", type=Path, nargs="?")
    parser.add_argument("members", type=Path, nargs="?", help="Legacy positional membership CSV")
    parser.add_argument("--membership", type=Path, help="Membership CSV, also usable with --lp-only")
    parser.add_argument("--lp-only", action="store_true", help="Explicitly omit unavailable PEAD evidence")
    parser.add_argument("--out", type=Path, default=Path("artifacts/standalone_research/causal-v1"))
    args = parser.parse_args(argv)
    if args.earnings is None and not args.lp_only:
        parser.error("Supply an earnings snapshot or explicitly select --lp-only")
    if args.lp_only and args.earnings is not None:
        parser.error("--lp-only cannot be combined with an earnings snapshot")
    if args.members is not None and args.membership is not None:
        parser.error("Supply membership once")
    history = load_snapshot(args.prices)
    benchmark = history.pop("^NSEI").Close
    earnings = load_snapshot(args.earnings) if args.earnings else {}
    membership_path = args.membership or args.members
    membership = pd.read_csv(membership_path, parse_dates=["start", "end"]) if membership_path else None
    trades = generate_trades(history, earnings, benchmark, membership)
    calendar = observation_calendar(history, benchmark, trades)
    combinations = [["LP"], ["PEAD"], ["PEAD_MKT"], ["LP", "PEAD"], ["LP", "PEAD_MKT"]] if earnings else [["LP"]]
    sources = {"prices": args.prices, "earnings": args.earnings, "membership": membership_path}
    report: dict[str, Any] = {
        "schema_version": 2, "replay_version": "SCHEDULED_CAUSAL_V1", "status": "RESEARCH_ONLY_EDGE_UNPROVEN",
        "inputs": {name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                   if path else None for name, path in sources.items()},
        "earnings_status": "SUPPLIED_NOT_ASOF_CERTIFIED" if earnings else "UNAVAILABLE_PEAD_NOT_EVALUATED",
        "settings": {"round_trip_cost": COST, "risk_fraction": RISK, "position_cap": CAP, "slots": SLOTS},
        "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                  (Path(__file__), Path(__file__).resolve().parents[1] / "core/scheduled_replay.py")},
        "calendar": {"basis": "OBSERVED_STOCK_AND_BENCHMARK_SESSIONS_NOT_EXCHANGE_CERTIFIED",
                     "missing_benchmark_sessions": [date.isoformat() for date in calendar.difference(benchmark.index)]},
        "execution": {"sizing": "PREVIOUS_CLOSE_NAV", "order": ["OPEN_EXITS", "OPEN_ENTRIES", "INTRADAY_EXITS", "CLOSE_EXITS"],
                      "fractional_quantities": True, "initial_nav": 1.0},
        "limitations": ["Constituent and price availability may retain survivorship/revision bias.",
                        "Membership CSV may be partial/stale; a filter does not certify complete historical coverage.",
                        "EPS estimates may be backfilled; publication/ingestion evidence is not established.",
                        "Daily-bar exits, including the modeled SMA close exit, are not executable broker quotes.",
                        "Per-setup opportunities retain the original overlap prefilter even if the portfolio rejects an earlier trade.",
                        "Opening exits can fund opening entries in the model; auction ordering/settlement is not established.",
                        "Missing benchmark sessions carry its last observed mark; hedging treats their benchmark return as zero.",
                        "No positive edge or live readiness is established."],
        "per_trade": {}, "portfolios": {}, "hedged": {},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.out / "opportunities.csv", index=False)
    for setup, group in trades.groupby("setup"):
        report["per_trade"][setup] = {}
        for label, mask in (("DEV", group.sig < SPLIT), ("HOLD", group.sig >= SPLIT)):
            segment = group.loc[mask]
            losses = -float(segment.loc[segment.ret < 0, "ret"].sum())
            report["per_trade"][setup][label] = {
                "trades": len(segment), "mean_net_return": float(segment.ret.mean()) if len(segment) else None,
                "win_rate": float((segment.ret > 0).mean()) if len(segment) else None,
                "mean_net_r": float(segment.R.mean()) if len(segment) else None,
                "profit_factor": float(segment.loc[segment.ret > 0, "ret"].sum()) / losses if losses else None,
                "monthly_ci95": monthly_interval(segment),
            }
    for setups in combinations:
        name = "+".join(setups)
        replay = replay_scheduled_trades(trades, history, calendar, setups)
        destination = args.out / name
        destination.mkdir(exist_ok=True)
        replay.equity.to_csv(destination / "equity.csv")
        replay.fills.to_csv(destination / "fills.csv", index=False)
        replay.decisions.to_csv(destination / "decisions.csv", index=False)
        buys = replay.fills.loc[replay.fills.action == "BUY"]
        report["portfolios"][name] = {
            "metrics": statistics(replay.equity.nav), "entries": len(buys),
            "exits": int((replay.fills.action == "SELL").sum()), "rejected": len(replay.decisions),
            "minimum_cash": float(replay.equity.cash.min()) if len(replay.equity) else None,
            "open_at_end": int(replay.equity.positions.iloc[-1]) if len(replay.equity) else 0,
        }
        print(name, json.dumps(report["portfolios"][name]))
        if setups in (["LP"], ["LP", "PEAD_MKT"]):
            for carry in ([0., .055] if setups == ["LP"] else [.055]):
                hedged = hedge(replay.equity.nav, benchmark, carry=carry)
                hedged.to_csv(destination / f"hedged_{carry:.3f}.csv", header=["nav"])
                report["hedged"][f"{name}_carry_{carry:.3f}"] = statistics(hedged)
    if len(calendar):
        first = int(benchmark.index.searchsorted(calendar[0]))
        base = float(benchmark.iloc[max(0, first - 1)])
        report["benchmark"] = statistics(benchmark.reindex(calendar).ffill() / base)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report: {args.out / 'report.json'}; {report['earnings_status']}")


if __name__ == "__main__":
    main()
