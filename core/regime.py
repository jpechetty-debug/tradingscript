"""
core/regime.py
==============
Market regime classification with 5-state taxonomy.

v14.5 hardening changes:
  Fix 1 — PANIC hysteresis: asymmetric entry/exit (0.25 in, 0.35 out)
  Fix 2 — TREND deadband:   0.55/0.45 replaces the 0.50 cliff in both branches
  Fix 3 — EXPANSION confidence: now scales with ADX and ATR strength
  Fix 4 — breadth_delta:    informational field on MarketRegime
  Fix 5 — RegimeTracker:    deque(maxlen=N) replaces list.pop(0)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MarketRegimeType


# ─────────────────────────────────────────────────────────────────────────────
# REGIME TRACKER  (Fix 5: deque, Fix 1: last_regime for hysteresis)
# ─────────────────────────────────────────────────────────────────────────────

class RegimeTracker:
    def __init__(self, max_history: int = 10):
        self._history: deque[MarketRegimeType] = deque(maxlen=max_history)
        self._last_breadth: float = 0.5

    def push(self, regime: MarketRegimeType, breadth: float = 0.5) -> None:
        self._history.append(regime)
        self._last_breadth = breadth

    def last_regime(self) -> MarketRegimeType | None:
        return self._history[-1] if self._history else None

    def last_breadth(self) -> float:
        return self._last_breadth

    def is_confirmed(self, regime: MarketRegimeType, confirm_bars: int) -> bool:
        if regime == MarketRegimeType.PANIC:
            return True
        tail = list(self._history)[-confirm_bars:]
        return len(tail) == confirm_bars and all(r == regime for r in tail)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET REGIME DATACLASS  (Fix 4: breadth_delta field)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MarketRegime:
    regime: MarketRegimeType
    breadth: float
    breadth_delta: float      # Fix 4: current - previous breadth
    adx_median: float
    atr_ratio: float
    confidence: float
    confirmed: bool

    def allows_long(self) -> bool:
        return self.regime in (MarketRegimeType.TREND_UP, MarketRegimeType.EXPANSION) and self.confirmed

    def allows_short(self) -> bool:
        return self.regime in (MarketRegimeType.TREND_DOWN, MarketRegimeType.EXPANSION) and self.confirmed

    def is_tradeable(self) -> bool:
        return self.regime != MarketRegimeType.PANIC

    def strategy_hint(self) -> str:
        s = {
            MarketRegimeType.TREND_UP: "BREAKOUT / MOMENTUM (confirmed)",
            MarketRegimeType.TREND_DOWN: "SHORT MOMENTUM (confirmed)",
            MarketRegimeType.RANGE: "MEAN REVERSION — fade edges",
            MarketRegimeType.EXPANSION: "VOLATILITY BREAKOUT — both sides",
            MarketRegimeType.PANIC: "NO TRADE — protect capital",
        }[self.regime]
        if not self.confirmed:
            s += " [UNCONFIRMED]"
        return s


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORING  (Fix 3: EXPANSION now scales)
# ─────────────────────────────────────────────────────────────────────────────

def _regime_confidence(
    regime: MarketRegimeType,
    adx_med: float,
    breadth: float,
    atr_rat: float,
    config,
) -> float:
    """Computes a strength-weighted confidence value [0-1]."""
    if regime == MarketRegimeType.PANIC:
        return float(np.clip(0.70 + (config.REGIME_BREADTH_PANIC - breadth) * 2, 0.70, 1.0))

    if regime in (MarketRegimeType.TREND_UP, MarketRegimeType.TREND_DOWN):
        adx_strength = float(np.clip((adx_med - config.REGIME_ADX_TREND) / 15.0, 0.0, 1.0))
        breadth_str = float(np.clip(abs(breadth - 0.5) * 2.0, 0.0, 1.0))
        return round(0.50 + adx_strength * 0.30 + breadth_str * 0.20, 3)

    # Fix 3: EXPANSION confidence scales with ADX and ATR strength
    if regime == MarketRegimeType.EXPANSION:
        adx_str = float(np.clip((adx_med - config.REGIME_ADX_TREND) / 15.0, 0.0, 1.0))
        atr_str = float(np.clip((atr_rat - config.REGIME_ATR_EXPANSION) / 0.7, 0.0, 1.0))
        return round(0.55 + adx_str * 0.25 + atr_str * 0.20, 3)

    return 0.55  # RANGE — keep neutral/baseline


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION  (Fix 1: PANIC hysteresis, Fix 2: TREND deadband)
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime(
    processed: dict[str, pd.DataFrame],
    breadth: float,
    tracker: RegimeTracker,
    config,
) -> MarketRegime:
    adx_vals, atr_ratios = [], []
    for ticker, df in processed.items():
        if ticker == config.BENCHMARK or df.empty:
            continue
        if "ADX" in df.columns:
            adx_vals.append(float(df["ADX"].iloc[-1]))
        if "ATR" in df.columns and "ATR_50_mean" in df.columns:
            m = float(df["ATR_50_mean"].iloc[-1])
            if m > 0:
                atr_ratios.append(float(df["ATR"].iloc[-1]) / m)

    adx_med = float(np.median(adx_vals)) if adx_vals else 20.0
    atr_rat = float(np.median(atr_ratios)) if atr_ratios else 1.0

    # Fix 4: breadth momentum
    breadth_delta = breadth - tracker.last_breadth()

    # ── Classification tree ──────────────────────────────────────────────
    # Fix 1: PANIC with asymmetric hysteresis
    if breadth < config.REGIME_BREADTH_PANIC:
        regime = MarketRegimeType.PANIC
    elif (tracker.last_regime() == MarketRegimeType.PANIC
          and breadth < config.REGIME_BREADTH_PANIC_EXIT):
        regime = MarketRegimeType.PANIC  # stay in PANIC until exit threshold

    elif atr_rat >= config.REGIME_ATR_EXPANSION and adx_med >= config.REGIME_ADX_TREND:
        regime = MarketRegimeType.EXPANSION

    elif adx_med >= config.REGIME_ADX_TREND:
        # Fix 2: deadband replaces cliff at 0.50
        if breadth >= config.REGIME_BREADTH_TREND_UP:
            regime = MarketRegimeType.TREND_UP
        elif breadth < config.REGIME_BREADTH_TREND_DN:
            regime = MarketRegimeType.TREND_DOWN
        else:
            regime = MarketRegimeType.RANGE  # ambiguous zone

    elif adx_med < config.REGIME_ADX_RANGE:
        regime = MarketRegimeType.RANGE

    else:
        # Mid-ADX zone — same deadband thresholds for consistency
        if breadth >= config.REGIME_BREADTH_TREND_UP:
            regime = MarketRegimeType.TREND_UP
        elif breadth < config.REGIME_BREADTH_TREND_DN:
            regime = MarketRegimeType.TREND_DOWN
        else:
            regime = MarketRegimeType.RANGE

    tracker.push(regime, breadth)
    confirmed = tracker.is_confirmed(regime, config.REGIME_CONFIRM_BARS)

    conf = _regime_confidence(regime, adx_med, breadth, atr_rat, config)
    return MarketRegime(regime, breadth, breadth_delta, adx_med, atr_rat, conf, confirmed)


# ─────────────────────────────────────────────────────────────────────────────
# RELATIVE STRENGTH
# ─────────────────────────────────────────────────────────────────────────────

def compute_rs(
    stock: pd.Series,
    bench: pd.Series,
    lookback: int | None = None,
    config=None,
) -> float:
    """
    Log-return relative-strength of *stock* vs *bench* over *lookback* bars.

    Parameters
    ----------
    stock:
        Close price series for the individual ticker.
    bench:
        Close price series for the benchmark (e.g. Nifty50).
    lookback:
        Number of bars to look back.  **Prefer passing this explicitly.**
        If omitted, ``config.RS_LOOKBACK`` is used when *config* is
        supplied; otherwise the module-level CONFIG singleton is the
        last resort.  Unit tests should always pass either *lookback*
        or *config* directly so they can control the parameter without
        patching the singleton.
    config:
        Optional ``SystemConfig`` instance.  Ignored when *lookback* is
        provided.
    """
    if lookback is None:
        if config is not None:
            lookback = config.RS_LOOKBACK
        else:
            from .config import CONFIG  # last-resort singleton — avoid in tests
            lookback = CONFIG.RS_LOOKBACK

    m = stock.rename("s").to_frame().join(bench.rename("b"), how="inner").dropna()
    if len(m) < lookback + 1:
        return 0.0
    s = np.log(m["s"].iloc[-1] / m["s"].iloc[-lookback - 1])
    b = np.log(m["b"].iloc[-1] / m["b"].iloc[-lookback - 1])
    return round((s - b) * 100, 3)


# ─────────────────────────────────────────────────────────────────────────────
# BREADTH & SECTOR RS
# ─────────────────────────────────────────────────────────────────────────────

def compute_breadth(processed: dict[str, pd.DataFrame], config) -> float:
    total = above = 0
    for ticker, df in processed.items():
        if ticker == config.BENCHMARK or df.empty:
            continue
        if "EMA_50" in df.columns:
            total += 1
            if float(df["Close"].iloc[-1]) > float(df["EMA_50"].iloc[-1]):
                above += 1
    return above / total if total else 0.5


def compute_sector_rs(processed: dict[str, pd.DataFrame], bench: pd.Series, config) -> dict[str, float]:
    from collections import defaultdict
    from .universe import TICKER_TO_SECTOR
    scores: dict[str, list[float]] = defaultdict(list)
    for ticker, df in processed.items():
        if ticker == config.BENCHMARK or df.empty:
            continue
        sector = TICKER_TO_SECTOR.get(ticker)
        if sector:
            scores[sector].append(compute_rs(df["Close"], bench, lookback=config.RS_LOOKBACK))
    return {s: round(float(np.median(v)), 3) if v else 0.0 for s, v in scores.items()}
