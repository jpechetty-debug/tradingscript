import numpy as np
import pandas as pd
from dataclasses import dataclass
from .config import MarketRegimeType

class RegimeTracker:
    def __init__(self, max_history: int = 10):
        self._history: list[MarketRegimeType] = []
        self._max = max_history

    def push(self, regime: MarketRegimeType) -> None:
        self._history.append(regime)
        if len(self._history) > self._max:
            self._history.pop(0)

    def is_confirmed(self, regime: MarketRegimeType, confirm_bars: int) -> bool:
        if regime == MarketRegimeType.PANIC: return True
        return (len(self._history) >= confirm_bars and all(r == regime for r in self._history[-confirm_bars:]))

@dataclass
class MarketRegime:
    regime: MarketRegimeType
    breadth: float
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
        if not self.confirmed: s += " [UNCONFIRMED]"
        return s

def _regime_confidence(regime: MarketRegimeType, adx_med: float, breadth: float, atr_rat: float, config) -> float:
    """Computes a strength-weighted confidence value [0-1]."""
    if regime == MarketRegimeType.PANIC:
        return float(np.clip(0.70 + (config.REGIME_BREADTH_PANIC - breadth) * 2, 0.70, 1.0))
    
    if regime in (MarketRegimeType.TREND_UP, MarketRegimeType.TREND_DOWN):
        # Scale ADX strength: 20-35 range adds up to 0.30 confidence
        adx_strength = float(np.clip((adx_med - config.REGIME_ADX_TREND) / 15.0, 0.0, 1.0))
        # Scale Breadth extreme: 0.5 center, distance adds up to 0.20 confidence
        breadth_str  = float(np.clip(abs(breadth - 0.5) * 2.0, 0.0, 1.0))
        return round(0.50 + adx_strength * 0.30 + breadth_str * 0.20, 3)
    
    return 0.55  # RANGE / EXPANSION - keep neutral/baseline


def classify_regime(processed: dict[str, pd.DataFrame], breadth: float, tracker: RegimeTracker, config) -> MarketRegime:
    adx_vals, atr_ratios = [], []
    for ticker, df in processed.items():
        if ticker == config.BENCHMARK or df.empty: continue
        if "ADX" in df.columns: adx_vals.append(float(df["ADX"].iloc[-1]))
        if "ATR" in df.columns and "ATR_50_mean" in df.columns:
            m = float(df["ATR_50_mean"].iloc[-1])
            if m > 0: atr_ratios.append(float(df["ATR"].iloc[-1]) / m)

    adx_med = float(np.median(adx_vals)) if adx_vals else 20.0
    atr_rat = float(np.median(atr_ratios)) if atr_ratios else 1.0

    if breadth < config.REGIME_BREADTH_PANIC:
        regime = MarketRegimeType.PANIC
    elif atr_rat >= config.REGIME_ATR_EXPANSION and adx_med >= config.REGIME_ADX_TREND:
        regime = MarketRegimeType.EXPANSION
    elif adx_med >= config.REGIME_ADX_TREND:
        regime = MarketRegimeType.TREND_UP if breadth >= 0.5 else MarketRegimeType.TREND_DOWN
    elif adx_med < config.REGIME_ADX_RANGE:
        regime = MarketRegimeType.RANGE
    else:
        regime = MarketRegimeType.TREND_UP if breadth >= 0.55 else (MarketRegimeType.TREND_DOWN if breadth < 0.45 else MarketRegimeType.RANGE)

    tracker.push(regime)
    confirmed = tracker.is_confirmed(regime, config.REGIME_CONFIRM_BARS)
    
    conf = _regime_confidence(regime, adx_med, breadth, atr_rat, config)
    return MarketRegime(regime, breadth, adx_med, atr_rat, conf, confirmed)


def compute_rs(stock: pd.Series, bench: pd.Series, lookback: int | None = None) -> float:
    from .config import CONFIG
    lb = lookback or CONFIG.RS_LOOKBACK
    m  = stock.rename("s").to_frame().join(bench.rename("b"), how="inner").dropna()
    if len(m) < lb + 1: return 0.0
    s = np.log(m["s"].iloc[-1] / m["s"].iloc[-lb-1])
    b = np.log(m["b"].iloc[-1] / m["b"].iloc[-lb-1])
    return round((s - b) * 100, 3)


def compute_breadth(processed: dict[str, pd.DataFrame], config) -> float:
    total = above = 0
    for ticker, df in processed.items():
        if ticker == config.BENCHMARK or df.empty: continue
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
        if ticker == config.BENCHMARK or df.empty: continue
        sector = TICKER_TO_SECTOR.get(ticker)
        if sector: scores[sector].append(compute_rs(df["Close"], bench, lookback=config.RS_LOOKBACK))
    return {s: round(float(np.median(v)), 3) if v else 0.0 for s, v in scores.items()}
