"""LEADER_PULLBACK_V1: paper-only swing setup, rules frozen from the 2026-10-05 research run.

Buy next open when, on a completed daily bar:
  Nifty close > SMA200; 20d avg turnover > 10 cr; 126d return in top 20% of the universe;
  close > SMA50 > SMA200; RSI(2) < 10.
Exit at first of: close > SMA5, stop = entry - 3 * ATR14, 10 sessions.
Research after 0.40% costs, point-in-time Nifty 500 membership (artifacts/universe/nifty500_members.csv):
dev 2021-24 +0.22%/trade, holdout 2025-26 +0.11%/trade; 95% intervals include zero.
(Survivorship-biased run had shown +0.44% / +0.22%.) Edge unproven — paper trade only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

STRATEGY_ID = "LEADER_PULLBACK_V1"
EXIT_RULE = "close > SMA5, or stop, or 10 sessions"
MIN_TURNOVER_INR = 1e8
RS_LOOKBACK, RS_MIN_PCT = 126, 0.80
RSI_LEN, RSI_MAX = 2, 10.0
STOP_ATR, TIME_STOP = 3.0, 10
RISK_FRACTION, MAX_POSITION_FRACTION, MAX_POSITIONS = 0.005, 0.10, 20   # portfolio-level sizing


@dataclass(frozen=True)
class PullbackSignal:
    ticker: str
    signal_date: str
    close: float
    atr: float
    stop_offset: float
    rs_percentile: float
    rsi2: float
    shares: int = 0
    strategy_id: str = STRATEGY_ID
    time_stop_bars: int = TIME_STOP
    exit_rule: str = EXIT_RULE


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n).mean()
    return 100 - 100 / (1 + up / dn)


def position_size(capital: float, close: float, stop_offset: float) -> int:
    """Equal risk per trade (0.5% of capital to the stop), capped at 10% of capital per position."""
    if capital <= 0 or close <= 0 or stop_offset <= 0:
        return 0
    return int(min(RISK_FRACTION * capital / stop_offset, MAX_POSITION_FRACTION * capital / close))


def load_members(path: str | Path) -> pd.DataFrame:
    """Point-in-time index membership CSV: ticker,start,end (end blank while still a member)."""
    m = pd.read_csv(path, parse_dates=["start", "end"])
    m["ticker"] = m["ticker"].map(lambda t: t if t.endswith(".NS") else f"{t}.NS")
    return m


def members_on(members: pd.DataFrame, date: pd.Timestamp) -> set[str]:
    live = (members.start <= date) & (members.end.isna() | (members.end > date))
    return set(members.loc[live, "ticker"])


def scan_leader_pullback(frames: dict[str, pd.DataFrame], bench_close: pd.Series,
                         date: pd.Timestamp, capital: float = 0.0,
                         members: Optional[set[str]] = None) -> list[PullbackSignal]:
    """Signals on ``date`` using only bars up to ``date``. Frames need OHLCV columns.

    ``members`` restricts the ranking universe to index constituents on ``date`` (no survivorship).
    """
    if members is not None:
        frames = {t: f for t, f in frames.items() if t in members}
    bench = bench_close.loc[:date].dropna()
    if len(bench) < 200 or bench.index[-1] != date or bench.iloc[-1] <= bench.tail(200).mean():
        return []
    closes = {t: f["Close"].loc[:date] for t, f in frames.items()}
    mom = {t: c.iloc[-1] / c.iloc[-RS_LOOKBACK - 1] - 1 for t, c in closes.items()
           if len(c) > RS_LOOKBACK and c.index[-1] == date and c.iloc[-RS_LOOKBACK - 1] > 0}
    if not mom:
        return []
    rs_pct = pd.Series(mom, dtype=float).rank(pct=True)
    out: list[PullbackSignal] = []
    for ticker, pct in rs_pct[rs_pct >= RS_MIN_PCT].items():
        f = frames[ticker].loc[:date]
        c = f["Close"]
        if len(c) < 200 or (c * f["Volume"]).tail(20).mean() <= MIN_TURNOVER_INR:
            continue
        sma50, sma200 = c.tail(50).mean(), c.tail(200).mean()
        rsi2 = float(_rsi(c, RSI_LEN).iloc[-1])
        if not (c.iloc[-1] > sma50 > sma200 and rsi2 < RSI_MAX):
            continue
        tr = pd.concat([f.High - f.Low, (f.High - c.shift()).abs(), (f.Low - c.shift()).abs()], axis=1).max(axis=1)
        atr = float(tr.tail(14).mean())
        if not np.isfinite(atr) or atr <= 0:
            continue
        out.append(PullbackSignal(ticker=ticker.replace(".NS", ""), signal_date=date.date().isoformat(),
                                  close=round(float(c.iloc[-1]), 2), atr=round(atr, 2),
                                  stop_offset=round(STOP_ATR * atr, 2), rs_percentile=round(float(pct) * 100, 1),
                                  rsi2=round(rsi2, 1),
                                  shares=position_size(capital, float(c.iloc[-1]), STOP_ATR * atr)))
    return sorted(out, key=lambda s: s.rsi2)[:MAX_POSITIONS]
