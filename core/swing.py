"""Versioned swing hypotheses with causal setup detection and bounded entry plans."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Optional

import numpy as np
import pandas as pd

from .config import IST, SystemConfig, parse_market_time


def completed_daily_bars(df: pd.DataFrame, config: SystemConfig,
                         now: Optional[datetime] = None) -> pd.DataFrame:
    """Exclude the active daily candle and future dates in exchange local time."""
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_unique or not df.index.is_monotonic_increasing:
        return df.iloc[:0]
    current = pd.Timestamp(now or datetime.now(IST))
    current = current.tz_localize(IST) if current.tzinfo is None else current.tz_convert(IST)
    dates = df.index.tz_localize(IST) if df.index.tz is None else df.index.tz_convert(IST)
    close_time = parse_market_time(config.MARKET_CLOSE_TIME)
    completed = (dates.date < current.date()) | ((dates.date == current.date()) & (current.time() >= close_time))
    return df.loc[completed]


@dataclass(frozen=True)
class SwingPlan:
    strategy_id: str
    signal_time: str
    entry: float
    entry_min: float
    entry_max: float
    stop: float
    target: float
    target2: float
    time_stop_bars: int
    atr: float


def detect_swing_setup(df: pd.DataFrame, benchmark: pd.Series, config: SystemConfig) -> Optional[SwingPlan]:
    """Preserve V1 eligibility; V2 selects its alpha independently of this wrapper."""
    return detect_swing_trigger(df, config, benchmark=benchmark)


def detect_swing_trigger(df: pd.DataFrame, config: SystemConfig, *,
                         benchmark: Optional[pd.Series] = None) -> Optional[SwingPlan]:
    """Completed-bar timing/risk plan; benchmark supplied only for V1 eligibility."""
    required = ["Open", "High", "Low", "Close", "ATR", "EMA_20", "EMA_50", "EMA_200", "RVol_20"]
    if (len(df) < config.SWING_MIN_BARS or not isinstance(df.index, pd.DatetimeIndex)
            or not df.index.is_monotonic_increasing or not df.index.is_unique
            or any(c not in df for c in required)):
        return None
    window = df.iloc[-64:]
    if not np.isfinite(window[required].to_numpy(dtype=float)).all():
        return None
    row = df.iloc[-1]
    close, atr = float(row.Close), float(row.ATR)
    if close <= 0 or atr <= 0:
        return None
    if benchmark is not None:
        if not (close > row.EMA_20 > row.EMA_50 > row.EMA_200):
            return None
        if row.EMA_50 <= df.EMA_50.iloc[-11] or (close - row.EMA_20) / atr > 3:
            return None
        bench = benchmark.loc[benchmark.index <= df.index[-1]].dropna()
        if len(bench) < 64 or bench.index[-1] != df.index[-1] or float(bench.iloc[-64]) <= 0:
            return None
        if bench.iloc[-1] <= bench.ewm(span=50, adjust=False).mean().iloc[-1]:
            return None
        if close / float(df.Close.iloc[-64]) < float(bench.iloc[-1] / bench.iloc[-64]):
            return None
    candle_range = float(row.High - row.Low)
    if candle_range <= 0 or candle_range > 2.5 * atr or (close - row.Low) / candle_range < .6:
        return None
    prior = df.iloc[:-1]
    resistance = float(prior.High.iloc[-config.SWING_BREAKOUT_LOOKBACK:].max())
    recent = prior.iloc[-5:]
    breakout = close > resistance and close - resistance <= .75 * atr and row.RVol_20 >= config.SWING_BREAKOUT_RVOL
    pullback = (bool((recent.Low <= recent.EMA_20 + .25 * atr).any())
                and bool((recent.Close >= recent.EMA_50).all())
                and close > float(prior.High.iloc[-1]) and close > float(row.Open))
    if not breakout and not pullback:
        return None
    stop = round(float(min(recent.Low.min(), row.Low)) - config.SWING_STOP_BUFFER_ATR * atr, 2)
    if config.SWING_STOP_MODE == "ATR":
        stop = round(close - config.SWING_STOP_ATR_MULT * atr, 2)
    elif config.SWING_STOP_MODE != "STRUCTURE":
        raise ValueError("Unknown swing stop mode")
    distance = close - stop
    min_distance = .25 if config.SWING_STOP_MODE == "ATR" else .75
    if stop <= 0 or not (min_distance * atr <= distance <= 3 * atr):
        return None
    overhead = float(prior.High.iloc[-63:].max())
    target = close + 2 * distance
    if overhead > close + .2 * atr:
        target = min(target, overhead)
    target = round(target, 2)
    if (target - close) / distance < config.SWING_MIN_RR:
        return None
    entry_max = min(close + config.SWING_MAX_GAP_ATR * atr,
                    (target + config.SWING_MIN_RR * stop) / (1 + config.SWING_MIN_RR))
    return SwingPlan(
        strategy_id=("SWING_BREAKOUT_V1" if breakout else "SWING_PULLBACK_V1") + config.SWING_RESEARCH_TAG,
        signal_time=df.index[-1].isoformat(), entry=close,
        entry_min=round(max(close - .5 * atr, stop + .5 * atr), 2),
        entry_max=math.floor(entry_max * 100) / 100,
        stop=stop, target=target, target2=round(close + 3 * distance, 2),
        time_stop_bars=config.SWING_BREAKOUT_TIME_BARS if breakout else config.SWING_PULLBACK_TIME_BARS, atr=atr,
    )


def swing_fill_size(plan: SwingPlan, price: float, config: SystemConfig,
                    capital_fraction: float = 1.) -> tuple[int, float]:
    """Reject gaps outside the plan; fixed fractional risk has no forced minimum size."""
    if not math.isfinite(price) or not plan.entry_min <= price <= plan.entry_max or price <= plan.stop:
        return 0, 0.
    if (plan.target - price) / (price - plan.stop) < config.SWING_MIN_RR:
        return 0, 0.
    capital = max(0., config.CAPITAL_INR * max(0., min(1., capital_fraction)))
    budget = min(config.RISK_PER_TRADE_INR, capital * config.SWING_RISK_FRACTION)
    quantity = max(0, min(int(budget / (price - plan.stop)),
                          int(capital * config.SWING_MAX_EXPOSURE_FRACTION / price)))
    return quantity, round(quantity * (price - plan.stop), 2)
