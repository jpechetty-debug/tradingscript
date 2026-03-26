"""
core — Sovereign Intraday Engine v14
=====================================
Public surface so consumers can do:

    from core import SystemConfig, CONFIG, add_indicators
    from core import compute_factors, FactorScores
    from core import score_ticker, TickerResult
"""

from .config import SystemConfig, MarketRegimeType, CONFIG, IST
from .indicators import add_indicators
from .factors import FactorScores, compute_factors, calibrate_ic_weights
from .scorer import TickerResult, score_ticker

__all__ = [
    "SystemConfig", "MarketRegimeType", "CONFIG", "IST",
    "add_indicators",
    "FactorScores", "compute_factors", "calibrate_ic_weights",
    "TickerResult", "score_ticker",
]
