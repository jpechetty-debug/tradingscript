"""
core/config.py
==============
Centralised, immutable system parameters for Sovereign Engine v14.

All runtime state (ICIR history, Platt calibration flag, etc.) has been
moved OUT of this dataclass and into ScanState in screener_v14_modular.py.
Config is now a true value object — safe to share across threads.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pytz

IST = pytz.timezone("Asia/Kolkata")


class MarketRegimeType(Enum):
    TREND_UP   = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE      = "RANGE"
    EXPANSION  = "EXPANSION"
    PANIC      = "PANIC"


@dataclass
class SystemConfig:
    # ── Probability & expectancy gates ───────────────────────────────────────
    MIN_PROB_WIN:      float = 0.52
    MIN_EXPECTANCY_R:  float = 0.15

    # ── Kelly position sizing ─────────────────────────────────────────────────
    KELLY_FRACTION:          float = 0.25
    KELLY_MIN_SHARES:        int   = 1
    KELLY_MAX_MULT:          float = 3.0
    KELLY_KURTOSIS_FALLBACK: float = 4.0
    KELLY_KURTOSIS_WINDOW:   int   = 252
    KELLY_KURTOSIS_MIN_OBS:  int   = 60

    # ── Platt calibration (loaded from file at startup) ───────────────────────
    PLATT_A: float = -4.0
    PLATT_B: float =  2.0

    # ── IC/ICIR weighting ────────────────────────────────────────────────────
    IC_LOOKBACK_DAYS:  int   = 60
    IC_FORWARD_BARS:   int   = 1
    ICIR_MIN_OBS:      int   = 20
    IC_CALIB_OFFSET:   int   = 60   # held-out window offset (Fix D)

    FACTOR_WEIGHTS: dict = field(default_factory=lambda: {
        "trend":      1 / 7,
        "momentum":   1 / 7,
        "volume":     1 / 7,
        "volatility": 1 / 7,
        "rs":         1 / 7,
        "breakout":   1 / 7,
        "quality":    1 / 7,
    })

    # ── Covariance matrix ────────────────────────────────────────────────────
    COV_EWMA_LAMBDA: float = 0.94
    COV_EWMA_BLEND:  float = 0.30
    COV_LOOKBACK:    int   = 60

    # ── Liquidity floors ──────────────────────────────────────────────────────
    ADV_SHARE_FLOOR:   int = 750_000
    ADV_TURNOVER_FLOOR: int = 35_000_000

    # ── Indicators ───────────────────────────────────────────────────────────
    SUPER_PERIOD: int   = 10
    SUPER_MULT:   float = 3.0
    ADX_PERIOD:   int   = 14
    ADX_STRONG:   int   = 25

    # ── Trade targets ─────────────────────────────────────────────────────────
    STOP_ATR_MULT:    float = 1.5
    TARGET1_ATR_MULT: float = 3.8
    TARGET2_ATR_MULT: float = 6.0
    RISK_PER_TRADE_INR: float = 10_000.0

    # ── Market regime ─────────────────────────────────────────────────────────
    BREADTH_VETO_BELOW:    float = 0.35
    RS_LOOKBACK:           int   = 20
    NEAR_52W_MAX_DIST_PCT: float = 8.0
    VOL_CONTRACT_RATIO:    float = 0.85

    REGIME_ADX_TREND:    int   = 25
    REGIME_ADX_RANGE:    int   = 18
    REGIME_ATR_EXPANSION: float = 1.3
    REGIME_BREADTH_PANIC: float = 0.25
    REGIME_CONFIRM_BARS:  int   = 2

    # ── EMA / structural filters ──────────────────────────────────────────────
    USE_EMA200_FILTER: bool  = True
    MAX_CORR:          float = 0.70

    # ── Portfolio construction ────────────────────────────────────────────────
    MAX_SECTOR_PICKS: int = 2
    PORTFOLIO_SIZE:   int = 6

    # ── Execution costs ───────────────────────────────────────────────────────
    SLIPPAGE_BPS:   int = 8
    COMMISSION_INR: int = 20

    # ── Volume profile ────────────────────────────────────────────────────────
    VPROFILE_LOOKBACK: int  = 30
    VPROFILE_BINS:     int  = 100
    USE_VALUE_AREA_RR: bool = True
    VA_MIN_RR:         float = 1.5

    # ── Data fetch ────────────────────────────────────────────────────────────
    DAILY_PERIOD: str = "1y"
    BENCHMARK:    str = "^NSEI"
    MAX_WORKERS:  int = 20

    # ── Session times (IST) ───────────────────────────────────────────────────
    SESSION_OPEN_END:   str = "10:15"
    SESSION_MIDDAY_END: str = "13:30"

    # ── Telegram alerts ───────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN:    str   = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID:      str   = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_ALERT_MIN_PROB: float = 0.60
    TELEGRAM_ALERT_TOP_N:  int   = 3
    TELEGRAM_DEDUP_HOURS:  int   = 4

    # ── Backtest ──────────────────────────────────────────────────────────────
    BACKTEST_DAYS:    int   = 90
    BACKTEST_MIN_PROB: float = 0.52

    # ── Fyers API ─────────────────────────────────────────────────────────────
    FYERS_CLIENT_ID:    str  = os.getenv("FYERS_CLIENT_ID", "")
    FYERS_SECRET_KEY:   str  = os.getenv("FYERS_SECRET_KEY", "")
    FYERS_REDIRECT_URI: str  = os.getenv("FYERS_REDIRECT_URI", "")
    FYERS_ACCESS_TOKEN: str  = os.getenv("FYERS_ACCESS_TOKEN", "")
    USE_FYERS:          bool = True

    def session_from_time(self, now: datetime | None = None) -> str:
        """Return session label for the given IST datetime (defaults to now)."""
        import datetime as dt
        now = now or datetime.now(IST)
        t = now.time() if hasattr(now, "time") else now
        t1 = dt.datetime.strptime(self.SESSION_OPEN_END,   "%H:%M").time()
        t2 = dt.datetime.strptime(self.SESSION_MIDDAY_END, "%H:%M").time()
        if t < t1:  return "OPENING_RANGE"
        if t < t2:  return "MIDDAY_CHOP"
        return "CLOSING_TREND"


CONFIG = SystemConfig()
