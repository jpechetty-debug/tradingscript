# Swing research (2026-10-05). Usage: see README in docstring of research_swing_portfolio.py
import pickle, sys, time, yfinance as yf, pandas as pd
from concurrent.futures import ThreadPoolExecutor
H = pickle.load(open(sys.argv[1], "rb")); tick = [t for t in H if t != "^NSEI"]
def get(t):
    for _ in range(3):
        try:
            e = yf.Ticker(t).get_earnings_dates(limit=40)
            return t, None if e is None else e.dropna(subset=["Reported EPS"])
        except Exception: time.sleep(2)
    return t, None
with ThreadPoolExecutor(8) as ex: out = {t: e for t, e in ex.map(get, tick) if e is not None and len(e)}
pickle.dump(out, open(sys.argv[2], "wb")); print(len(out), "tickers with earnings")
