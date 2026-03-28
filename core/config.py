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
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta
from enum import Enum
from zoneinfo import ZoneInfo


class SecretStr(str):
    """
    A ``str`` subclass whose ``__repr__`` never exposes the secret value.

    Use for any field that holds an API key, token, or password so that
    accidentally logging or printing the config object does not leak
    credentials::

        token = SecretStr("bot123:ABC-def")
        repr(token)   # → "SecretStr('bot1...ef')"
        str(token)    # → "bot123:ABC-def"   (full value for API calls)

    The masked repr shows the first 4 and last 4 characters when the
    secret is at least 12 chars long; otherwise it returns ``'****'``.
    """

    _SECRET_FIELDS = frozenset({
        "TELEGRAM_BOT_TOKEN",
        "FYERS_CLIENT_ID",
        "FYERS_SECRET_KEY",
        "FYERS_ACCESS_TOKEN",
    })

    def __repr__(self) -> str:
        if len(self) >= 12:
            return f"SecretStr('{self[:4]}...{self[-4:]}')"
        return "SecretStr('****')"


IST = ZoneInfo("Asia/Kolkata")

# Fields whose values must never appear in repr / logs.
_SECRET_FIELD_NAMES: frozenset[str] = frozenset({
    "TELEGRAM_BOT_TOKEN",
    "FYERS_CLIENT_ID",
    "FYERS_SECRET_KEY",
    "FYERS_ACCESS_TOKEN",
})


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

    REGIME_ADX_TREND:          int   = 25
    REGIME_ADX_RANGE:          int   = 18
    REGIME_ATR_EXPANSION:      float = 1.3
    REGIME_BREADTH_PANIC:      float = 0.25  # entry threshold
    REGIME_BREADTH_PANIC_EXIT: float = 0.35  # Fix 1: asymmetric exit (higher than entry)
    REGIME_CONFIRM_BARS:       int   = 2

    # ── Opening-range regime lock (Fix 6) ─────────────────────────────────────
    # Suppress regime *changes* for this many minutes after NSE open (09:15 IST)
    # to avoid whipsaw reclassification during price-discovery noise.
    MARKET_OPEN_TIME:    str = "09:15"
    REGIME_LOCK_MINUTES: int = 20

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
    MAX_WORKERS:  int = min(20, (os.cpu_count() or 4) * 2)

    # ── Session times (IST) ───────────────────────────────────────────────────
    SESSION_OPEN_END:   str = "10:15"
    SESSION_MIDDAY_END: str = "13:30"

    # ── Telegram alerts ───────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN:    SecretStr = field(
        default_factory=lambda: SecretStr(os.getenv("TELEGRAM_BOT_TOKEN", ""))
    )
    TELEGRAM_CHAT_ID:      str   = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_ALERT_MIN_PROB: float = 0.60
    TELEGRAM_ALERT_TOP_N:  int   = 3
    TELEGRAM_DEDUP_HOURS:  int   = 4

    # ── Backtest ──────────────────────────────────────────────────────────────
    BACKTEST_DAYS:    int   = 90
    BACKTEST_MIN_PROB: float = 0.52

    # ── Fyers API ─────────────────────────────────────────────────────────────
    FYERS_CLIENT_ID:    SecretStr = field(
        default_factory=lambda: SecretStr(os.getenv("FYERS_CLIENT_ID", ""))
    )
    FYERS_SECRET_KEY:   SecretStr = field(
        default_factory=lambda: SecretStr(os.getenv("FYERS_SECRET_KEY", ""))
    )
    FYERS_REDIRECT_URI: str  = os.getenv("FYERS_REDIRECT_URI", "")
    FYERS_ACCESS_TOKEN: SecretStr = field(
        default_factory=lambda: SecretStr(os.getenv("FYERS_ACCESS_TOKEN", ""))
    )
    USE_FYERS:          bool = True

    def __repr__(self) -> str:
        """
        Safe repr that masks all ``SecretStr`` fields.

        Without this override, ``dataclasses.dataclass`` generates a repr
        that calls ``repr()`` on every field — which would print the full
        token/key values for plain ``str`` fields and, before SecretStr was
        introduced, would leak credentials in any log that records the
        config object.
        """
        parts = []
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name in _SECRET_FIELD_NAMES:
                # Use SecretStr's own masked repr
                parts.append(f"{f.name}={SecretStr(str(val))!r}")
            else:
                parts.append(f"{f.name}={val!r}")
        return f"SystemConfig({', '.join(parts)})"

    def session_from_time(self, now: datetime | None = None) -> str:
        """Return session label for the given IST datetime (defaults to now)."""
        current_time = (now.astimezone(IST) if now is not None else datetime.now(IST)).time()
        t1 = datetime.strptime(self.SESSION_OPEN_END, "%H:%M").time()
        t2 = datetime.strptime(self.SESSION_MIDDAY_END, "%H:%M").time()
        if current_time < t1:
            return "OPENING_RANGE"
        if current_time < t2:
            return "MIDDAY_CHOP"
        return "CLOSING_TREND"

    def is_regime_locked(self, now: datetime | None = None) -> bool:
        """
        Fix 6 — Return True during the opening noise window.

        The first ``REGIME_LOCK_MINUTES`` minutes after ``MARKET_OPEN_TIME``
        (IST) are structurally noisy: gap opens and price-discovery cause
        breadth / ADX readings that do not represent the true session regime.
        While the lock is active ``classify_regime`` skips ``tracker.push()``
        so no noisy bar can reclassify the confirmed regime.

        Outside market hours (pre-open / post-close) this always returns
        ``False`` so off-hours backtests and unit tests are unaffected.
        """
        current_time = (now.astimezone(IST) if now is not None else datetime.now(IST)).time()
        open_t = datetime.strptime(self.MARKET_OPEN_TIME, "%H:%M").time()
        lock_end = (
            datetime.combine(date.today(), open_t)
            + timedelta(minutes=self.REGIME_LOCK_MINUTES)
        ).time()
        return open_t <= current_time < lock_end


CONFIG = SystemConfig()
