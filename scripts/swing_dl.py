# Swing research (2026-10-05). Usage: see README in docstring of research_swing_portfolio.py
import pickle, sys, yfinance as yf
from core.universe import ALL_TICKERS
t = list(dict.fromkeys(ALL_TICKERS + ["^NSEI"]))
df = yf.download(t, period="6y", interval="1d", auto_adjust=True, group_by="ticker", threads=True, progress=False)
out = {}
for k in t:
    try:
        d = df[k].dropna(how="all")
        if len(d) > 300: out[k] = d
    except KeyError: pass
pickle.dump(out, open(sys.argv[1], "wb"))
print(len(out), "tickers saved")
