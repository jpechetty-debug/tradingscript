"""
core/config.py
==============
Centralised system parameters and typed domain-specific settings for Sovereign Engine v14.

Provides strongly-typed frozen setting slices (MarketDataSettings, RegimeSettings,
SignalSettings, PortfolioSettings, ExecutionCostSettings, AlertSettings, BacktestSettings)
aggregated into AppSettings composition root, plus ScoringRuntime for dynamic parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Mapping, Optional
from zoneinfo import ZoneInfo


class SecretStr(str):
    """
    A ``str`` subclass whose ``__repr__`` never exposes the secret value.
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


class MarketPhase(str, Enum):
    PRE_MARKET          = "PRE_MARKET"           # Before 09:15
    OPENING_NOISE       = "OPENING_NOISE"       # 09:15 - 09:30: Gap auction & Initial Range formation
    PRIME_MORNING       = "PRIME_MORNING"       # 09:30 - 10:15: Prime ORB / Trend Discovery (Golden Window #1)
    MIDDAY_CHOP         = "MIDDAY_CHOP"         # 10:15 - 13:30: Low volume consolidation (Mean-Rev / Strict Hurdle)
    AFTERNOON_EXPANSION = "AFTERNOON_EXPANSION" # 13:30 - 14:30: European open trend momentum (Golden Window #2)
    INTRADAY_FREEZE     = "INTRADAY_FREEZE"     # 14:30 - 15:15: Hard freeze on new MIS entries; broker cutoff approaches
    SWING_CLOSING       = "SWING_CLOSING"       # 14:30 - 15:30: Daily candle confirmation (Prime SWING / BTST)
    POST_MARKET         = "POST_MARKET"         # After 15:30


# ── Domain-specific Frozen Settings Dataclasses ─────────────────────────────

@dataclass(frozen=True)
class MarketDataSettings:
    daily_period: str = "1y"
    benchmark: str = "^NSEI"
    max_workers: int = field(default_factory=lambda: min(20, (os.cpu_count() or 4) * 2))
    use_fyers: bool = True
    fyers_client_id: SecretStr = field(default_factory=lambda: SecretStr(os.getenv("FYERS_CLIENT_ID", "")))
    fyers_secret_key: SecretStr = field(default_factory=lambda: SecretStr(os.getenv("FYERS_SECRET_KEY", "")))
    fyers_redirect_uri: str = field(default_factory=lambda: os.getenv("FYERS_REDIRECT_URI", ""))
    fyers_access_token: SecretStr = field(default_factory=lambda: SecretStr(os.getenv("FYERS_ACCESS_TOKEN", "")))


@dataclass(frozen=True)
class RegimeSettings:
    breadth_veto_below: float = 0.35
    rs_lookback: int = 20
    near_52w_max_dist_pct: float = 8.0
    vol_contract_ratio: float = 0.85
    regime_adx_trend: int = 25
    regime_adx_range: int = 18
    regime_atr_expansion: float = 1.3
    regime_breadth_panic: float = 0.25
    regime_breadth_panic_exit: float = 0.35
    regime_confirm_bars: int = 2
    market_open_time: str = "09:15"
    regime_lock_minutes: int = 20
    session_open_end: str = "10:15"
    session_midday_end: str = "13:30"
    intraday_entry_start: str = "09:30"
    intraday_entry_cutoff: str = "14:30"
    swing_scan_start: str = "14:30"
    mis_squareoff_time: str = "15:15"
    market_close_time: str = "15:30"
    midday_breakout_min_prob: float = 0.55
    intraday_enabled: bool = False  # Disabled until real 1/5-min data pipeline is connected

    def session_from_time(self, now: datetime | None = None) -> str:
        current_time = (now.astimezone(IST) if now is not None else datetime.now(IST)).time()
        t1 = datetime.strptime(self.session_open_end, "%H:%M").time()
        t2 = datetime.strptime(self.session_midday_end, "%H:%M").time()
        if current_time < t1:
            return "OPENING_RANGE"
        if current_time < t2:
            return "MIDDAY_CHOP"
        return "CLOSING_TREND"

    def get_market_phase(self, now: datetime | None = None) -> MarketPhase:
        current_time = (now.astimezone(IST) if now is not None else datetime.now(IST)).time()
        t_open = datetime.strptime(self.market_open_time, "%H:%M").time()
        t_intra_start = datetime.strptime(self.intraday_entry_start, "%H:%M").time()
        t_morning_end = datetime.strptime(self.session_open_end, "%H:%M").time()
        t_midday_end = datetime.strptime(self.session_midday_end, "%H:%M").time()
        t_cutoff = datetime.strptime(self.intraday_entry_cutoff, "%H:%M").time()
        t_close = datetime.strptime(self.market_close_time, "%H:%M").time()

        if current_time < t_open:
            return MarketPhase.PRE_MARKET
        if current_time < t_intra_start:
            return MarketPhase.OPENING_NOISE
        if current_time < t_morning_end:
            return MarketPhase.PRIME_MORNING
        if current_time < t_midday_end:
            return MarketPhase.MIDDAY_CHOP
        if current_time < t_cutoff:
            return MarketPhase.AFTERNOON_EXPANSION
        if current_time < t_close:
            return MarketPhase.INTRADAY_FREEZE
        return MarketPhase.POST_MARKET

    def minutes_to_squareoff(self, now: datetime | None = None) -> int:
        dt = now.astimezone(IST) if now is not None else datetime.now(IST)
        t_sq = datetime.strptime(self.mis_squareoff_time, "%H:%M").time()
        sq_dt = dt.replace(hour=t_sq.hour, minute=t_sq.minute, second=0, microsecond=0)
        diff = (sq_dt - dt).total_seconds() / 60.0
        return max(0, int(diff))

    def is_regime_locked(self, now: datetime | None = None) -> bool:
        current_time = (now.astimezone(IST) if now is not None else datetime.now(IST)).time()
        open_t = datetime.strptime(self.market_open_time, "%H:%M").time()
        lock_end = (
            datetime.combine(date.today(), open_t)
            + timedelta(minutes=self.regime_lock_minutes)
        ).time()
        return open_t <= current_time < lock_end


@dataclass(frozen=True)
class SignalSettings:
    super_period: int = 10
    super_mult: float = 3.0
    adx_period: int = 14
    adx_strong: int = 25
    stop_atr_mult: float = 1.5
    target1_atr_mult: float = 3.8
    target2_atr_mult: float = 6.0
    risk_per_trade_inr: float = 10_000.0
    use_ema200_filter: bool = True
    vprofile_lookback: int = 30
    vprofile_bins: int = 100
    use_value_area_rr: bool = True
    va_min_rr: float = 1.5
    adv_share_floor: int = 750_000
    adv_turnover_floor: int = 35_000_000
    ic_lookback_days: int = 60
    ic_forward_bars: int = 1
    icir_min_obs: int = 20
    ic_calib_offset: int = 60
    watchlist_min_prob: float = 0.45
    prob_hold_floor: float = 0.47
    enable_watchlist: bool = True
    short_is_intraday_only: bool = True
    cohort_min_obs: int = 10
    cohort_rank_weight: float = 0.40
    min_factor_weight: float = 0.03  # Floor per factor to prevent starvation


@dataclass(frozen=True)
class PortfolioSettings:
    kelly_fraction: float = 0.25
    kelly_min_shares: int = 1
    kelly_max_mult: float = 3.0
    kelly_kurtosis_fallback: float = 4.0
    kelly_kurtosis_window: int = 252
    kelly_kurtosis_min_obs: int = 60
    cov_ewma_lambda: float = 0.94
    cov_ewma_blend: float = 0.30
    cov_lookback: int = 60
    max_corr: float = 0.70
    max_sector_picks: int = 2
    portfolio_size: int = 6
    candidates_max: int = 20


@dataclass(frozen=True)
class ExecutionCostSettings:
    slippage_bps: int = 8
    commission_inr: int = 20


@dataclass(frozen=True)
class AlertSettings:
    telegram_bot_token: SecretStr = field(
        default_factory=lambda: SecretStr(os.getenv("TELEGRAM_BOT_TOKEN", ""))
    )
    telegram_chat_id: str = field(default_factory=lambda: os.getenv("TELEGRAM_CHAT_ID", ""))
    telegram_alert_min_prob: float = 0.60
    telegram_alert_top_n: int = 3
    telegram_dedup_hours: int = 4


@dataclass(frozen=True)
class BacktestSettings:
    backtest_days: int = 90
    backtest_min_prob: float = 0.52


@dataclass(frozen=True)
class AppSettings:
    market_data: MarketDataSettings = field(default_factory=MarketDataSettings)
    regime: RegimeSettings = field(default_factory=RegimeSettings)
    signal: SignalSettings = field(default_factory=SignalSettings)
    portfolio: PortfolioSettings = field(default_factory=PortfolioSettings)
    execution_cost: ExecutionCostSettings = field(default_factory=ExecutionCostSettings)
    alerts: AlertSettings = field(default_factory=AlertSettings)
    backtest: BacktestSettings = field(default_factory=BacktestSettings)


@dataclass(frozen=True)
class ScoringRuntime:
    platt_a: float = -4.0
    platt_b: float = 2.0
    factor_weights: Mapping[str, float] = field(default_factory=lambda: {
        "trend":      1 / 7,
        "momentum":   1 / 7,
        "volume":     1 / 7,
        "volatility": 1 / 7,
        "rs":         1 / 7,
        "breakout":   1 / 7,
        "quality":    1 / 7,
    })
    min_prob_win: float = 0.52
    prob_hold_floor: float = 0.47
    min_prob_watchlist: float = 0.45
    enable_watchlist: bool = True
    min_expectancy_r: float = 0.15
    use_ema200_filter: bool = True
    cohort_min_obs: int = 10
    cohort_rank_weight: float = 0.40


# ── Monolithic Legacy Config (Compatibility Adapter) ─────────────────────────

@dataclass
class SystemConfig:
    # ── Probability & expectancy gates ───────────────────────────────────────
    MIN_PROB_WIN:       float = field(default_factory=lambda: float(os.getenv("MIN_PROB_WIN", "0.52")))
    PROB_HOLD_FLOOR:    float = field(default_factory=lambda: float(os.getenv("PROB_HOLD_FLOOR", "0.47")))
    WATCHLIST_MIN_PROB: float = field(default_factory=lambda: float(os.getenv("WATCHLIST_MIN_PROB", "0.45")))
    ENABLE_WATCHLIST:   bool  = field(default_factory=lambda: os.getenv("ENABLE_WATCHLIST", "true").lower() in ("true", "1", "yes"))
    MIN_EXPECTANCY_R:   float = field(default_factory=lambda: float(os.getenv("MIN_EXPECTANCY_R", "0.15")))
    COHORT_MIN_OBS:     int   = field(default_factory=lambda: int(os.getenv("COHORT_MIN_OBS", "10")))
    COHORT_RANK_WEIGHT: float = field(default_factory=lambda: float(os.getenv("COHORT_RANK_WEIGHT", "0.40")))

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

    # ── Trade targets (SWING defaults) ────────────────────────────────────────
    STOP_ATR_MULT:    float = 1.5
    TARGET1_ATR_MULT: float = 3.8
    TARGET2_ATR_MULT: float = 6.0
    RISK_PER_TRADE_INR: float = field(default_factory=lambda: float(os.getenv("RISK_PER_TRADE_INR", "10000.0")))

    # ── Intraday trade targets (tighter for 6-hour sessions) ─────────────────
    INTRADAY_STOP_ATR_MULT:    float = 0.50
    INTRADAY_TARGET1_ATR_MULT: float = 1.20
    INTRADAY_TARGET2_ATR_MULT: float = 1.80
    SHORT_IS_INTRADAY_ONLY:    bool  = True
    INTRADAY_ENABLED:          bool  = False  # Disabled until real 1/5-min data pipeline
    MIN_FACTOR_WEIGHT:         float = 0.03   # Floor per factor to prevent starvation

    # ── Market regime ─────────────────────────────────────────────────────────
    BREADTH_VETO_BELOW:    float = 0.35
    RS_LOOKBACK:           int   = 20
    NEAR_52W_MAX_DIST_PCT: float = 8.0
    VOL_CONTRACT_RATIO:    float = 0.85

    REGIME_ADX_TREND:          int   = 25
    REGIME_ADX_RANGE:          int   = 18
    REGIME_ATR_EXPANSION:      float = 1.3
    REGIME_BREADTH_PANIC:      float = 0.25
    REGIME_BREADTH_PANIC_EXIT: float = 0.35
    REGIME_CONFIRM_BARS:       int   = 2

    # ── Opening-range regime lock ─────────────────────────────────────────────
    MARKET_OPEN_TIME:    str = "09:15"
    REGIME_LOCK_MINUTES: int = 20

    # ── EMA / structural filters ──────────────────────────────────────────────
    USE_EMA200_FILTER: bool  = field(default_factory=lambda: os.getenv("USE_EMA200_FILTER", "true").lower() in ("true", "1", "yes"))
    MAX_CORR:          float = 0.70

    # ── Portfolio construction ────────────────────────────────────────────────
    MAX_SECTOR_PICKS: int = 2
    PORTFOLIO_SIZE:   int = field(default_factory=lambda: int(os.getenv("PORTFOLIO_SIZE", "6")))
    CANDIDATES_MAX:   int = field(default_factory=lambda: int(os.getenv("CANDIDATES_MAX", "20")))

    # ── Execution costs ───────────────────────────────────────────────────────
    SLIPPAGE_BPS:   int = field(default_factory=lambda: int(os.getenv("SLIPPAGE_BPS", "8")))
    COMMISSION_INR: int = field(default_factory=lambda: int(os.getenv("COMMISSION_INR", "20")))

    # ── Volume profile ────────────────────────────────────────────────────────
    VPROFILE_LOOKBACK: int  = 30
    VPROFILE_BINS:     int  = 100
    USE_VALUE_AREA_RR: bool = True
    VA_MIN_RR:         float = 1.5

    # ── Data fetch ────────────────────────────────────────────────────────────
    DAILY_PERIOD: str = "1y"
    BENCHMARK:    str = "^NSEI"
    MAX_WORKERS:  int = field(default_factory=lambda: min(20, (os.cpu_count() or 4) * 2))

    # ── Session times (IST) ───────────────────────────────────────────────────
    SESSION_OPEN_END:   str = "10:15"
    SESSION_MIDDAY_END: str = "13:30"
    INTRADAY_ENTRY_START:     str = "09:30"
    INTRADAY_ENTRY_CUTOFF:    str = "14:30"
    SWING_SCAN_START:         str = "14:30"
    MIS_SQUAREOFF_TIME:       str = "15:15"
    MARKET_CLOSE_TIME:        str = "15:30"
    MIDDAY_BREAKOUT_MIN_PROB: float = 0.55

    # ── Telegram alerts ───────────────────────────────────────────────────────
    TELEGRAM_BOT_TOKEN:    SecretStr = field(
        default_factory=lambda: SecretStr(os.getenv("TELEGRAM_BOT_TOKEN", ""))
    )
    TELEGRAM_CHAT_ID:      str   = os.getenv("TELEGRAM_CHAT_ID", "")
    TELEGRAM_ALERT_MIN_PROB: float = 0.60
    TELEGRAM_ALERT_TOP_N:  int   = 3
    TELEGRAM_DEDUP_HOURS:  int   = 4

    # ── Duplicate alert suppression ───────────────────────────────────────────
    DUPLICATE_LOOKBACK_SCANS: int   = 5
    DUPLICATE_COMPOSITE_DELTA: float = 0.05

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
        parts = []
        for f in fields(self):
            val = getattr(self, f.name)
            if f.name in _SECRET_FIELD_NAMES:
                parts.append(f"{f.name}={SecretStr(str(val))!r}")
            else:
                parts.append(f"{f.name}={val!r}")
        return f"SystemConfig({', '.join(parts)})"

    def session_from_time(self, now: datetime | None = None) -> str:
        return self.as_regime().session_from_time(now)

    def is_regime_locked(self, now: datetime | None = None) -> bool:
        return self.as_regime().is_regime_locked(now)

    # ── Slice Adapters ────────────────────────────────────────────────────────
    def as_market_data(self) -> MarketDataSettings:
        return MarketDataSettings(
            daily_period=self.DAILY_PERIOD,
            benchmark=self.BENCHMARK,
            max_workers=self.MAX_WORKERS,
            use_fyers=self.USE_FYERS,
            fyers_client_id=self.FYERS_CLIENT_ID,
            fyers_secret_key=self.FYERS_SECRET_KEY,
            fyers_redirect_uri=self.FYERS_REDIRECT_URI,
            fyers_access_token=self.FYERS_ACCESS_TOKEN,
        )

    def as_regime(self) -> RegimeSettings:
        return RegimeSettings(
            breadth_veto_below=self.BREADTH_VETO_BELOW,
            rs_lookback=self.RS_LOOKBACK,
            near_52w_max_dist_pct=self.NEAR_52W_MAX_DIST_PCT,
            vol_contract_ratio=self.VOL_CONTRACT_RATIO,
            regime_adx_trend=self.REGIME_ADX_TREND,
            regime_adx_range=self.REGIME_ADX_RANGE,
            regime_atr_expansion=self.REGIME_ATR_EXPANSION,
            regime_breadth_panic=self.REGIME_BREADTH_PANIC,
            regime_breadth_panic_exit=self.REGIME_BREADTH_PANIC_EXIT,
            regime_confirm_bars=self.REGIME_CONFIRM_BARS,
            market_open_time=self.MARKET_OPEN_TIME,
            regime_lock_minutes=self.REGIME_LOCK_MINUTES,
            session_open_end=self.SESSION_OPEN_END,
            session_midday_end=self.SESSION_MIDDAY_END,
            intraday_entry_start=self.INTRADAY_ENTRY_START,
            intraday_entry_cutoff=self.INTRADAY_ENTRY_CUTOFF,
            swing_scan_start=self.SWING_SCAN_START,
            mis_squareoff_time=self.MIS_SQUAREOFF_TIME,
            market_close_time=self.MARKET_CLOSE_TIME,
            midday_breakout_min_prob=self.MIDDAY_BREAKOUT_MIN_PROB,
        )

    def get_market_phase(self, now: datetime | None = None) -> MarketPhase:
        return self.as_regime().get_market_phase(now)

    def minutes_to_squareoff(self, now: datetime | None = None) -> int:
        return self.as_regime().minutes_to_squareoff(now)

    def as_signal(self) -> SignalSettings:
        return SignalSettings(
            super_period=self.SUPER_PERIOD,
            super_mult=self.SUPER_MULT,
            adx_period=self.ADX_PERIOD,
            adx_strong=self.ADX_STRONG,
            stop_atr_mult=self.STOP_ATR_MULT,
            target1_atr_mult=self.TARGET1_ATR_MULT,
            target2_atr_mult=self.TARGET2_ATR_MULT,
            risk_per_trade_inr=self.RISK_PER_TRADE_INR,
            use_ema200_filter=self.USE_EMA200_FILTER,
            vprofile_lookback=self.VPROFILE_LOOKBACK,
            vprofile_bins=self.VPROFILE_BINS,
            use_value_area_rr=self.USE_VALUE_AREA_RR,
            va_min_rr=self.VA_MIN_RR,
            adv_share_floor=self.ADV_SHARE_FLOOR,
            adv_turnover_floor=self.ADV_TURNOVER_FLOOR,
            ic_lookback_days=self.IC_LOOKBACK_DAYS,
            ic_forward_bars=self.IC_FORWARD_BARS,
            icir_min_obs=self.ICIR_MIN_OBS,
            ic_calib_offset=self.IC_CALIB_OFFSET,
            watchlist_min_prob=self.WATCHLIST_MIN_PROB,
            prob_hold_floor=self.PROB_HOLD_FLOOR,
            enable_watchlist=self.ENABLE_WATCHLIST,
            cohort_min_obs=self.COHORT_MIN_OBS,
            cohort_rank_weight=self.COHORT_RANK_WEIGHT,
        )

    def as_portfolio(self) -> PortfolioSettings:
        return PortfolioSettings(
            kelly_fraction=self.KELLY_FRACTION,
            kelly_min_shares=self.KELLY_MIN_SHARES,
            kelly_max_mult=self.KELLY_MAX_MULT,
            kelly_kurtosis_fallback=self.KELLY_KURTOSIS_FALLBACK,
            kelly_kurtosis_window=self.KELLY_KURTOSIS_WINDOW,
            kelly_kurtosis_min_obs=self.KELLY_KURTOSIS_MIN_OBS,
            cov_ewma_lambda=self.COV_EWMA_LAMBDA,
            cov_ewma_blend=self.COV_EWMA_BLEND,
            cov_lookback=self.COV_LOOKBACK,
            max_corr=self.MAX_CORR,
            max_sector_picks=self.MAX_SECTOR_PICKS,
            portfolio_size=self.PORTFOLIO_SIZE,
            candidates_max=self.CANDIDATES_MAX,
        )

    def as_execution_cost(self) -> ExecutionCostSettings:
        return ExecutionCostSettings(
            slippage_bps=self.SLIPPAGE_BPS,
            commission_inr=self.COMMISSION_INR,
        )

    def as_alerts(self) -> AlertSettings:
        return AlertSettings(
            telegram_bot_token=self.TELEGRAM_BOT_TOKEN,
            telegram_chat_id=self.TELEGRAM_CHAT_ID,
            telegram_alert_min_prob=self.TELEGRAM_ALERT_MIN_PROB,
            telegram_alert_top_n=self.TELEGRAM_ALERT_TOP_N,
            telegram_dedup_hours=self.TELEGRAM_DEDUP_HOURS,
        )

    def as_backtest(self) -> BacktestSettings:
        return BacktestSettings(
            backtest_days=self.BACKTEST_DAYS,
            backtest_min_prob=self.BACKTEST_MIN_PROB,
        )

    def as_scoring_runtime(self) -> ScoringRuntime:
        return ScoringRuntime(
            platt_a=self.PLATT_A,
            platt_b=self.PLATT_B,
            factor_weights=self.FACTOR_WEIGHTS,
            min_prob_win=self.MIN_PROB_WIN,
            prob_hold_floor=self.PROB_HOLD_FLOOR,
            min_prob_watchlist=self.WATCHLIST_MIN_PROB,
            enable_watchlist=self.ENABLE_WATCHLIST,
            min_expectancy_r=self.MIN_EXPECTANCY_R,
            use_ema200_filter=self.USE_EMA200_FILTER,
            cohort_min_obs=self.COHORT_MIN_OBS,
            cohort_rank_weight=self.COHORT_RANK_WEIGHT,
        )

    def as_app_settings(self) -> AppSettings:
        return AppSettings(
            market_data=self.as_market_data(),
            regime=self.as_regime(),
            signal=self.as_signal(),
            portfolio=self.as_portfolio(),
            execution_cost=self.as_execution_cost(),
            alerts=self.as_alerts(),
            backtest=self.as_backtest(),
        )


def load_system_config(env_file: Optional[str | Path] = None) -> SystemConfig:
    """
    Load environment variables via dotenv (if installed) and produce a typed SystemConfig.
    """
    try:
        from dotenv import load_dotenv
        if env_file:
            load_dotenv(env_file, override=True)
        else:
            load_dotenv()
    except Exception:
        pass
    return SystemConfig()


CONFIG = load_system_config()
