"""
tests/test_regime.py
====================
Unit tests for core/regime.py — v14 modular regime classification.

Covers:
  - RegimeTracker      — push, confirm, PANIC fast-path, history cap, isolation
  - MarketRegime       — allows_long/short, is_tradeable, strategy_hint
  - classify_regime()  — all 5 regime states, boundary conditions, confirmation
  - _regime_confidence — confidence ranges per regime type
  - compute_rs()       — log-return relative strength
  - compute_breadth()  — market breadth (% above EMA50)
  - compute_sector_rs() — sector aggregation

Run:
    pytest tests/test_regime.py -v
    pytest tests/test_regime.py --cov=core/regime --cov-report=term-missing
"""

from __future__ import annotations

import sys
import os
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ── Stub optional heavy dependencies ──────────────────────────────────────────
def _stub_modules() -> None:
    stubs: dict[str, types.ModuleType] = {
        "fyers_apiv3":            types.ModuleType("fyers_apiv3"),
        "fyers_apiv3.fyersModel": types.ModuleType("fyers_apiv3.fyersModel"),
    }
    stubs["fyers_apiv3.fyersModel"].FyersModel = MagicMock
    stubs["fyers_apiv3"].fyersModel = stubs["fyers_apiv3.fyersModel"]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

_stub_modules()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import SystemConfig, MarketRegimeType
from core.regime import (
    RegimeTracker,
    MarketRegime,
    classify_regime,
    compute_sector_concentration,
    _regime_confidence,
    compute_rs,
    compute_breadth,
    compute_sector_rs,
)


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_config(**overrides) -> SystemConfig:
    return SystemConfig(**overrides)


def _make_daily_df(
    n: int = 200,
    seed: int = 42,
    adx: float = 28.0,
    close_above_ema: bool = True,
    atr_ratio: float = 1.0,
) -> pd.DataFrame:
    np.random.seed(seed)
    close = 1000 * np.cumprod(1 + np.random.normal(0.001, 0.01, n))
    df = pd.DataFrame({
        "Open": close * 0.999,
        "High": close * 1.005,
        "Low": close * 0.995,
        "Close": close,
        "Volume": np.full(n, 1_000_000.0),
    })
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["ADX"] = adx
    atr_val = (df["High"] - df["Low"]).rolling(14).mean().iloc[-1]
    df["ATR"] = atr_val * atr_ratio
    df["ATR_50_mean"] = atr_val
    if close_above_ema:
        df["Close"] = df["EMA_50"] * 1.05
    else:
        df["Close"] = df["EMA_50"] * 0.95
    return df


def _make_processed(
    n_tickers: int = 10,
    adx: float = 28.0,
    breadth_frac: float = 0.7,
    atr_ratio: float = 1.0,
    config: SystemConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or _make_config()
    out = {}
    above_count = int(n_tickers * breadth_frac)
    for i in range(n_tickers):
        df = _make_daily_df(
            seed=i + 100,
            adx=adx,
            close_above_ema=(i < above_count),
            atr_ratio=atr_ratio,
        )
        out[f"TICKER{i}.NS"] = df
    # benchmark
    out[config.BENCHMARK] = _make_daily_df(seed=999)
    return out


def _make_price_series(n: int = 200, seed: int = 0) -> pd.Series:
    np.random.seed(seed)
    return pd.Series(
        1000 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n)),
        name="Close",
    )


# ─────────────────────────────────────────────────────────────────────────────
# RegimeTracker
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeTracker:

    def test_fresh_tracker_not_confirmed(self):
        t = RegimeTracker()
        t.push(MarketRegimeType.TREND_UP)
        assert t.is_confirmed(MarketRegimeType.TREND_UP, confirm_bars=2) is False

    def test_confirmed_after_enough_bars(self):
        t = RegimeTracker()
        t.push(MarketRegimeType.TREND_UP)
        t.push(MarketRegimeType.TREND_UP)
        assert t.is_confirmed(MarketRegimeType.TREND_UP, confirm_bars=2) is True

    def test_mixed_history_not_confirmed(self):
        t = RegimeTracker()
        t.push(MarketRegimeType.TREND_UP)
        t.push(MarketRegimeType.RANGE)
        assert t.is_confirmed(MarketRegimeType.TREND_UP, confirm_bars=2) is False

    def test_panic_always_confirmed(self):
        """PANIC should be confirmed immediately — capital protection."""
        t = RegimeTracker()
        t.push(MarketRegimeType.PANIC)
        assert t.is_confirmed(MarketRegimeType.PANIC, confirm_bars=10) is True

    def test_history_capped_at_max(self):
        t = RegimeTracker(max_history=3)
        for _ in range(10):
            t.push(MarketRegimeType.TREND_UP)
        assert len(t._history) == 3

    def test_two_instances_isolated(self):
        """Watch-mode regression — two trackers must not share state."""
        t1 = RegimeTracker()
        t2 = RegimeTracker()
        t1.push(MarketRegimeType.TREND_UP)
        t1.push(MarketRegimeType.TREND_UP)
        assert t2.is_confirmed(MarketRegimeType.TREND_UP, confirm_bars=2) is False

    def test_confirm_bars_one_passes_after_single_push(self):
        t = RegimeTracker()
        t.push(MarketRegimeType.EXPANSION)
        assert t.is_confirmed(MarketRegimeType.EXPANSION, confirm_bars=1) is True


# ─────────────────────────────────────────────────────────────────────────────
# MarketRegime Dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestMarketRegime:

    def _make(self, regime: MarketRegimeType, confirmed: bool = True) -> MarketRegime:
        return MarketRegime(
            regime=regime,
            breadth=0.50,
            adx_median=25.0,
            atr_ratio=1.0,
            confidence=0.60,
            confirmed=confirmed,
        )

    def test_trend_up_allows_long(self):
        r = self._make(MarketRegimeType.TREND_UP)
        assert r.allows_long() is True
        assert r.allows_short() is False

    def test_trend_down_allows_short(self):
        r = self._make(MarketRegimeType.TREND_DOWN)
        assert r.allows_long() is False
        assert r.allows_short() is True

    def test_expansion_allows_both(self):
        r = self._make(MarketRegimeType.EXPANSION)
        assert r.allows_long() is True
        assert r.allows_short() is True

    def test_range_blocks_both(self):
        """RANGE regime should not allow directional trades."""
        r = self._make(MarketRegimeType.RANGE)
        assert r.allows_long() is False
        assert r.allows_short() is False

    def test_panic_blocks_both(self):
        r = self._make(MarketRegimeType.PANIC)
        assert r.allows_long() is False
        assert r.allows_short() is False

    def test_panic_not_tradeable(self):
        r = self._make(MarketRegimeType.PANIC)
        assert r.is_tradeable() is False

    def test_all_non_panic_tradeable(self):
        for rt in (MarketRegimeType.TREND_UP, MarketRegimeType.TREND_DOWN,
                   MarketRegimeType.RANGE, MarketRegimeType.EXPANSION):
            r = self._make(rt)
            assert r.is_tradeable() is True

    def test_unconfirmed_blocks_long(self):
        r = self._make(MarketRegimeType.TREND_UP, confirmed=False)
        assert r.allows_long() is False

    def test_unconfirmed_blocks_short(self):
        r = self._make(MarketRegimeType.TREND_DOWN, confirmed=False)
        assert r.allows_short() is False


# ─────────────────────────────────────────────────────────────────────────────
# classify_regime — all 5 paths
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyRegime:

    def _run(self, adx=28.0, breadth=0.60, atr_ratio=1.0, confirm_bars=1):
        config = _make_config(REGIME_CONFIRM_BARS=confirm_bars)
        processed = _make_processed(
            n_tickers=10, adx=adx, breadth_frac=breadth,
            atr_ratio=atr_ratio, config=config,
        )
        tracker = RegimeTracker()
        return classify_regime(processed, breadth, tracker, config)

    def test_panic_when_breadth_very_low(self):
        r = self._run(adx=30, breadth=0.15)
        assert r.regime == MarketRegimeType.PANIC

    def test_trend_up_high_adx_high_breadth(self):
        r = self._run(adx=30, breadth=0.70)
        assert r.regime == MarketRegimeType.TREND_UP

    def test_trend_down_high_adx_low_breadth(self):
        r = self._run(adx=30, breadth=0.30)
        assert r.regime == MarketRegimeType.TREND_DOWN

    def test_range_when_adx_very_low(self):
        r = self._run(adx=12, breadth=0.50)
        assert r.regime == MarketRegimeType.RANGE

    def test_expansion_high_adx_high_atr(self):
        r = self._run(adx=30, breadth=0.60, atr_ratio=1.5)
        assert r.regime == MarketRegimeType.EXPANSION

    def test_boundary_adx_at_trend_threshold(self):
        """ADX exactly at REGIME_ADX_TREND threshold should classify as TREND."""
        config = _make_config()
        r = self._run(adx=config.REGIME_ADX_TREND, breadth=0.55)
        assert r.regime in (MarketRegimeType.TREND_UP, MarketRegimeType.TREND_DOWN,
                            MarketRegimeType.EXPANSION)

    def test_else_branch_mid_adx_high_breadth(self):
        """ADX between ADX_RANGE and ADX_TREND, breadth >= 0.55 → TREND_UP."""
        config = _make_config()
        mid_adx = (config.REGIME_ADX_RANGE + config.REGIME_ADX_TREND) / 2
        r = self._run(adx=mid_adx, breadth=0.60)
        assert r.regime == MarketRegimeType.TREND_UP

    def test_else_branch_mid_adx_low_breadth(self):
        """ADX between ADX_RANGE and ADX_TREND, breadth < 0.45 → TREND_DOWN."""
        config = _make_config()
        mid_adx = (config.REGIME_ADX_RANGE + config.REGIME_ADX_TREND) / 2
        r = self._run(adx=mid_adx, breadth=0.40)
        assert r.regime == MarketRegimeType.TREND_DOWN

    def test_else_branch_mid_adx_mid_breadth(self):
        """ADX mid, breadth ∈ [0.45, 0.55) → RANGE."""
        config = _make_config()
        mid_adx = (config.REGIME_ADX_RANGE + config.REGIME_ADX_TREND) / 2
        r = self._run(adx=mid_adx, breadth=0.50)
        assert r.regime == MarketRegimeType.RANGE

    def test_confirmed_after_multiple_pushes(self):
        config = _make_config(REGIME_CONFIRM_BARS=3)
        processed = _make_processed(adx=30, breadth_frac=0.70, config=config)
        tracker = RegimeTracker()
        for _ in range(3):
            r = classify_regime(processed, 0.70, tracker, config)
        assert r.confirmed is True

    def test_unconfirmed_after_single_push(self):
        r = self._run(adx=30, breadth=0.70, confirm_bars=3)
        assert r.confirmed is False

    def test_empty_processed_no_crash(self):
        config = _make_config()
        processed = {config.BENCHMARK: _make_daily_df(seed=999)}
        tracker = RegimeTracker()
        r = classify_regime(processed, 0.5, tracker, config)
        assert isinstance(r.regime, MarketRegimeType)


# ─────────────────────────────────────────────────────────────────────────────
# _regime_confidence
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeConfidence:

    def test_panic_confidence_at_least_070(self):
        config = _make_config()
        conf = _regime_confidence(MarketRegimeType.PANIC, 20, 0.10, 1.0, config)
        assert conf >= 0.70

    def test_panic_confidence_capped_at_one(self):
        config = _make_config()
        conf = _regime_confidence(MarketRegimeType.PANIC, 20, 0.0, 1.0, config)
        assert conf <= 1.0

    def test_trend_confidence_scales_with_adx(self):
        config = _make_config()
        low = _regime_confidence(MarketRegimeType.TREND_UP, 25, 0.60, 1.0, config)
        high = _regime_confidence(MarketRegimeType.TREND_UP, 40, 0.60, 1.0, config)
        assert high >= low

    def test_range_confidence_is_baseline(self):
        config = _make_config()
        conf = _regime_confidence(MarketRegimeType.RANGE, 15, 0.50, 1.0, config)
        assert conf == 0.55

    def test_expansion_confidence_scales_above_baseline(self):
        """Fix 3: EXPANSION confidence now scales with ADX and ATR — no longer flat."""
        config = _make_config()
        # ADX=30 (above threshold), ATR ratio=1.5 (above expansion threshold)
        conf_strong = _regime_confidence(MarketRegimeType.EXPANSION, 30, 0.60, 1.5, config)
        # ADX at threshold, ATR at threshold — minimum EXPANSION inputs
        conf_min = _regime_confidence(
            MarketRegimeType.EXPANSION,
            config.REGIME_ADX_TREND,      # ADX exactly at trend threshold
            0.50,
            config.REGIME_ATR_EXPANSION,  # ATR exactly at expansion threshold
            config,
        )
        assert conf_strong > 0.55, "strong EXPANSION should score above the old flat baseline"
        assert conf_min == pytest.approx(0.55), "minimum-threshold EXPANSION equals old baseline"
        assert conf_strong <= 1.0

    def test_expansion_confidence_higher_for_stronger_signals(self):
        """ADX=45 + ATR=2.0 should beat ADX=26 + ATR=1.4 for EXPANSION."""
        config = _make_config()
        high = _regime_confidence(MarketRegimeType.EXPANSION, 45, 0.60, 2.0, config)
        low  = _regime_confidence(MarketRegimeType.EXPANSION, 26, 0.60, 1.4, config)
        assert high > low


# ─────────────────────────────────────────────────────────────────────────────
# compute_rs
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRS:

    def test_outperforming_positive(self):
        bench = _make_price_series(200, seed=0)
        stock = bench.copy()
        stock.iloc[-1] *= 1.05
        rs = compute_rs(stock, bench, lookback=20)
        assert rs > 0

    def test_underperforming_negative(self):
        bench = _make_price_series(200, seed=0)
        stock = bench.copy()
        stock.iloc[-1] *= 0.95
        rs = compute_rs(stock, bench, lookback=20)
        assert rs < 0

    def test_same_series_zero(self):
        bench = _make_price_series(200, seed=0)
        rs = compute_rs(bench, bench, lookback=20)
        assert abs(rs) < 1e-6

    def test_short_series_returns_zero(self):
        short = _make_price_series(5, seed=0)
        bench = _make_price_series(200, seed=1)
        rs = compute_rs(short, bench, lookback=20)
        assert rs == 0.0

    def test_custom_lookback(self):
        bench = _make_price_series(200, seed=0)
        stock = bench * np.linspace(1.0, 1.05, 200)
        rs5 = compute_rs(stock, bench, lookback=5)
        rs50 = compute_rs(stock, bench, lookback=50)
        assert rs5 != rs50  # different windows → different results


# ─────────────────────────────────────────────────────────────────────────────
# compute_breadth
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBreadth:

    def test_all_above_gives_one(self):
        config = _make_config()
        processed = _make_processed(n_tickers=10, breadth_frac=1.0, config=config)
        b = compute_breadth(processed, config)
        assert b == pytest.approx(1.0, abs=0.01)

    def test_all_below_gives_zero(self):
        config = _make_config()
        processed = _make_processed(n_tickers=10, breadth_frac=0.0, config=config)
        b = compute_breadth(processed, config)
        assert b == pytest.approx(0.0, abs=0.01)

    def test_half_above_gives_half(self):
        config = _make_config()
        processed = _make_processed(n_tickers=10, breadth_frac=0.5, config=config)
        b = compute_breadth(processed, config)
        assert 0.4 < b < 0.6

    def test_benchmark_excluded(self):
        config = _make_config()
        processed = _make_processed(n_tickers=5, breadth_frac=1.0, config=config)
        # Force benchmark below EMA
        processed[config.BENCHMARK]["Close"] = processed[config.BENCHMARK]["EMA_50"] * 0.5
        b = compute_breadth(processed, config)
        assert b == pytest.approx(1.0, abs=0.01)

    def test_empty_returns_fallback(self):
        config = _make_config()
        processed = {config.BENCHMARK: _make_daily_df(seed=999)}
        b = compute_breadth(processed, config)
        assert b == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# compute_sector_rs
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSectorRS:

    @pytest.fixture(autouse=True)
    def _require_universe(self):
        """Skip tests in this class when core.universe is not importable."""
        pytest.importorskip("core.universe")

    def test_returns_dict(self):
        config = _make_config()
        processed = _make_processed(config=config)
        bench = processed[config.BENCHMARK]["Close"]
        result = compute_sector_rs(processed, bench, config)
        assert isinstance(result, dict)

    def test_benchmark_excluded(self):
        config = _make_config()
        processed = _make_processed(config=config)
        bench = processed[config.BENCHMARK]["Close"]
        result = compute_sector_rs(processed, bench, config)
        # benchmark key should not appear as a sector
        assert config.BENCHMARK not in result


# ─────────────────────────────────────────────────────────────────────────────
# strategy_hint
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyHint:

    def test_trend_up_hint(self):
        r = MarketRegime(MarketRegimeType.TREND_UP, 0.6, 28, 1.0, 0.7, True)
        assert "BREAKOUT" in r.strategy_hint()

    def test_trend_down_hint(self):
        r = MarketRegime(MarketRegimeType.TREND_DOWN, 0.3, 28, 1.0, 0.7, True)
        assert "SHORT" in r.strategy_hint()

    def test_range_hint(self):
        r = MarketRegime(MarketRegimeType.RANGE, 0.5, 15, 1.0, 0.55, True)
        assert "MEAN REVERSION" in r.strategy_hint()

    def test_expansion_hint(self):
        r = MarketRegime(MarketRegimeType.EXPANSION, 0.6, 30, 1.5, 0.55, True)
        assert "VOLATILITY" in r.strategy_hint()

    def test_panic_hint(self):
        r = MarketRegime(MarketRegimeType.PANIC, 0.1, 20, 1.0, 0.8, True)
        assert "NO TRADE" in r.strategy_hint()

    def test_unconfirmed_suffix(self):
        r = MarketRegime(MarketRegimeType.TREND_UP, 0.6, 28, 1.0, 0.7, False)
        hint = r.strategy_hint()
        assert "UNCONFIRMED" in hint


# ─────────────────────────────────────────────────────────────────────────────
# Fix 6 — RegimeLock (opening noise window)
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeLock:
    """
    Verify that classify_regime(locked=True) suppresses tracker.push() so a
    single noisy opening bar cannot reclassify the confirmed regime.
    """

    def _make_processed_adx(self, adx: float = 30.0, breadth_frac: float = 0.70) -> dict:
        config = _make_config()
        return _make_processed(n_tickers=10, adx=adx, breadth_frac=breadth_frac, config=config)

    def test_locked_skips_tracker_push(self):
        """tracker._history must be unchanged after a locked call."""
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = self._make_processed_adx()
        tracker = RegimeTracker()
        # Pre-load one confirmed TREND_UP
        tracker.push(MarketRegimeType.TREND_UP, 0.70)
        assert len(tracker._history) == 1

        classify_regime(processed, 0.70, tracker, config, locked=True)

        assert len(tracker._history) == 1, "locked call must not grow tracker history"

    def test_locked_returns_last_known_regime(self):
        """When locked, the returned regime matches the tracker's last push."""
        config = _make_config(REGIME_CONFIRM_BARS=1)
        # Processed data would naturally produce TREND_DOWN (low breadth)
        processed = self._make_processed_adx(adx=30.0, breadth_frac=0.20)
        tracker = RegimeTracker()
        tracker.push(MarketRegimeType.TREND_UP, 0.70)  # prior confirmed state

        r = classify_regime(processed, 0.20, tracker, config, locked=True)

        assert r.regime == MarketRegimeType.TREND_UP, (
            "locked call should return last known regime, not freshly computed one"
        )

    def test_locked_falls_back_to_raw_when_history_empty(self):
        """If the tracker has no history yet, locked must still return something sensible."""
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = self._make_processed_adx(adx=30.0, breadth_frac=0.70)
        tracker = RegimeTracker()  # empty history

        r = classify_regime(processed, 0.70, tracker, config, locked=True)

        assert isinstance(r.regime, MarketRegimeType)

    def test_locked_sets_regime_locked_field_true(self):
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = self._make_processed_adx()
        tracker = RegimeTracker()
        tracker.push(MarketRegimeType.TREND_UP, 0.70)

        r = classify_regime(processed, 0.70, tracker, config, locked=True)

        assert r.regime_locked is True

    def test_unlocked_sets_regime_locked_field_false(self):
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = self._make_processed_adx()
        tracker = RegimeTracker()

        r = classify_regime(processed, 0.70, tracker, config, locked=False)

        assert r.regime_locked is False

    def test_unlocked_default_updates_tracker(self):
        """Default (locked omitted) must update the tracker normally."""
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = self._make_processed_adx()
        tracker = RegimeTracker()
        assert len(tracker._history) == 0

        classify_regime(processed, 0.70, tracker, config)

        assert len(tracker._history) == 1

    def test_locked_does_not_affect_breadth_delta(self):
        """breadth_delta is computed from tracker.last_breadth(), not affected by lock."""
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = self._make_processed_adx()
        tracker = RegimeTracker()
        tracker.push(MarketRegimeType.TREND_UP, 0.60)  # last breadth = 0.60

        r = classify_regime(processed, 0.70, tracker, config, locked=True)

        assert abs(r.breadth_delta - 0.10) < 1e-4, (
            f"breadth_delta should be 0.70-0.60=0.10, got {r.breadth_delta}"
        )

    def test_locked_strategy_hint_contains_locked(self):
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = self._make_processed_adx()
        tracker = RegimeTracker()
        tracker.push(MarketRegimeType.TREND_UP, 0.70)

        r = classify_regime(processed, 0.70, tracker, config, locked=True)

        assert "LOCKED" in r.strategy_hint()

    def test_consecutive_locked_calls_history_unchanged(self):
        """Multiple locked calls in the same watch cycle must not accumulate history."""
        config = _make_config(REGIME_CONFIRM_BARS=2)
        processed = self._make_processed_adx()
        tracker = RegimeTracker()
        tracker.push(MarketRegimeType.TREND_UP, 0.70)
        initial_len = len(tracker._history)

        for _ in range(5):
            classify_regime(processed, 0.70, tracker, config, locked=True)

        assert len(tracker._history) == initial_len


# ─────────────────────────────────────────────────────────────────────────────
# Fix 6 — SystemConfig.is_regime_locked()
# ─────────────────────────────────────────────────────────────────────────────

class TestIsRegimeLocked:
    """Verify the config helper correctly identifies the opening noise window."""

    import datetime as _dt

    def _make_ist_time(self, hour: int, minute: int):
        from core.config import IST
        import datetime as dt
        return IST.localize(dt.datetime(2024, 1, 15, hour, minute, 0))

    def test_locked_at_open(self):
        config = _make_config(MARKET_OPEN_TIME="09:15", REGIME_LOCK_MINUTES=20)
        assert config.is_regime_locked(self._make_ist_time(9, 15)) is True

    def test_locked_during_window(self):
        config = _make_config(MARKET_OPEN_TIME="09:15", REGIME_LOCK_MINUTES=20)
        assert config.is_regime_locked(self._make_ist_time(9, 20)) is True

    def test_locked_at_last_minute_of_window(self):
        config = _make_config(MARKET_OPEN_TIME="09:15", REGIME_LOCK_MINUTES=20)
        assert config.is_regime_locked(self._make_ist_time(9, 34)) is True

    def test_unlocked_at_exact_window_end(self):
        config = _make_config(MARKET_OPEN_TIME="09:15", REGIME_LOCK_MINUTES=20)
        # 09:15 + 20 min = 09:35 — the boundary itself is NOT locked
        assert config.is_regime_locked(self._make_ist_time(9, 35)) is False

    def test_unlocked_after_window(self):
        config = _make_config(MARKET_OPEN_TIME="09:15", REGIME_LOCK_MINUTES=20)
        assert config.is_regime_locked(self._make_ist_time(10, 0)) is False

    def test_unlocked_before_market_open(self):
        config = _make_config(MARKET_OPEN_TIME="09:15", REGIME_LOCK_MINUTES=20)
        assert config.is_regime_locked(self._make_ist_time(8, 0)) is False

    def test_zero_lock_minutes_never_locked(self):
        """Setting REGIME_LOCK_MINUTES=0 disables the feature entirely."""
        config = _make_config(MARKET_OPEN_TIME="09:15", REGIME_LOCK_MINUTES=0)
        assert config.is_regime_locked(self._make_ist_time(9, 15)) is False

    def test_custom_lock_window(self):
        """Configurable window — should work for any REGIME_LOCK_MINUTES value."""
        config = _make_config(MARKET_OPEN_TIME="09:15", REGIME_LOCK_MINUTES=5)
        assert config.is_regime_locked(self._make_ist_time(9, 19)) is True
        assert config.is_regime_locked(self._make_ist_time(9, 20)) is False


# ─────────────────────────────────────────────────────────────────────────────
# Fix 7 — compute_sector_concentration()
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorConcentration:
    """
    Unit tests for the pure compute_sector_concentration() function.
    """

    def test_all_positive_rs_trend_up_gives_one(self):
        sector_rs = {"IT": 2.1, "BANKING": 1.5, "AUTO": 0.8, "PHARMA": 0.3}
        result = compute_sector_concentration(sector_rs, MarketRegimeType.TREND_UP)
        assert result == pytest.approx(1.0)

    def test_all_negative_rs_trend_down_gives_one(self):
        sector_rs = {"IT": -2.1, "BANKING": -1.5, "AUTO": -0.8}
        result = compute_sector_concentration(sector_rs, MarketRegimeType.TREND_DOWN)
        assert result == pytest.approx(1.0)

    def test_all_negative_rs_trend_up_gives_zero(self):
        """Everything down but regime says UP → no alignment → 0.0."""
        sector_rs = {"IT": -2.1, "BANKING": -1.5, "AUTO": -0.8}
        result = compute_sector_concentration(sector_rs, MarketRegimeType.TREND_UP)
        assert result == pytest.approx(0.0)

    def test_mixed_rs_trend_up_gives_fraction(self):
        """4 out of 8 sectors positive → 0.5."""
        sector_rs = {
            "IT": 2.0, "BANKING": 1.5, "AUTO": 0.5, "PHARMA": 0.1,
            "FMCG": -0.3, "METALS": -1.0, "ENERGY": -0.8, "REALTY": -0.2,
        }
        result = compute_sector_concentration(sector_rs, MarketRegimeType.TREND_UP)
        assert result == pytest.approx(0.5)

    def test_mixed_rs_trend_down_gives_fraction(self):
        """6 out of 8 sectors negative → 0.75."""
        sector_rs = {
            "IT": 2.0, "BANKING": 1.5,
            "AUTO": -0.5, "PHARMA": -0.1, "FMCG": -0.3,
            "METALS": -1.0, "ENERGY": -0.8, "REALTY": -0.2,
        }
        result = compute_sector_concentration(sector_rs, MarketRegimeType.TREND_DOWN)
        assert result == pytest.approx(0.75)

    def test_non_directional_regime_returns_neutral(self):
        sector_rs = {"IT": 2.0, "BANKING": -1.5, "AUTO": 0.5}
        for regime in (MarketRegimeType.RANGE, MarketRegimeType.EXPANSION,
                       MarketRegimeType.PANIC):
            result = compute_sector_concentration(sector_rs, regime)
            assert result == pytest.approx(0.5), f"failed for {regime}"

    def test_empty_sector_rs_returns_neutral(self):
        result = compute_sector_concentration({}, MarketRegimeType.TREND_UP)
        assert result == pytest.approx(0.5)

    def test_single_sector_aligned(self):
        result = compute_sector_concentration(
            {"IT": 3.0}, MarketRegimeType.TREND_UP
        )
        assert result == pytest.approx(1.0)

    def test_single_sector_not_aligned(self):
        result = compute_sector_concentration(
            {"IT": -1.0}, MarketRegimeType.TREND_UP
        )
        assert result == pytest.approx(0.0)

    def test_result_is_rounded_to_three_decimals(self):
        """Output should not carry floating-point noise."""
        sector_rs = {f"S{i}": (1.0 if i < 7 else -1.0) for i in range(10)}
        result = compute_sector_concentration(sector_rs, MarketRegimeType.TREND_UP)
        assert result == round(result, 3)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 7 — sector concentration feeds confidence
# ─────────────────────────────────────────────────────────────────────────────

class TestSectorConcentrationConfidence:
    """
    Verify that a broadly participating TREND regime earns higher confidence
    than a narrow one, and that RANGE / EXPANSION are unaffected.
    """

    def _make_sector_rs(self, n_positive: int, n_total: int) -> dict[str, float]:
        rs = {}
        for i in range(n_total):
            rs[f"S{i}"] = 1.0 if i < n_positive else -1.0
        return rs

    def test_broad_trend_up_higher_confidence_than_narrow(self):
        config = _make_config()
        adx, breadth, atr = 30.0, 0.65, 1.0
        broad  = self._make_sector_rs(10, 10)   # 100% aligned
        narrow = self._make_sector_rs(5, 10)    # 50% aligned
        c_broad  = _regime_confidence(MarketRegimeType.TREND_UP, adx, breadth, atr, config,
                                      sector_conc=compute_sector_concentration(broad,  MarketRegimeType.TREND_UP))
        c_narrow = _regime_confidence(MarketRegimeType.TREND_UP, adx, breadth, atr, config,
                                      sector_conc=compute_sector_concentration(narrow, MarketRegimeType.TREND_UP))
        assert c_broad > c_narrow, (
            f"broad confidence {c_broad} should exceed narrow {c_narrow}"
        )

    def test_broad_trend_down_higher_confidence_than_narrow(self):
        config = _make_config()
        adx, breadth, atr = 30.0, 0.35, 1.0
        broad  = self._make_sector_rs(0, 10)    # 100% negative = fully aligned DOWN
        narrow = self._make_sector_rs(5, 10)    # 50% aligned
        c_broad  = _regime_confidence(MarketRegimeType.TREND_DOWN, adx, breadth, atr, config,
                                      sector_conc=compute_sector_concentration(broad,  MarketRegimeType.TREND_DOWN))
        c_narrow = _regime_confidence(MarketRegimeType.TREND_DOWN, adx, breadth, atr, config,
                                      sector_conc=compute_sector_concentration(narrow, MarketRegimeType.TREND_DOWN))
        assert c_broad > c_narrow

    def test_max_sector_bonus_does_not_exceed_one(self):
        config = _make_config()
        full_rs = {f"S{i}": 2.0 for i in range(10)}  # 100% aligned
        conf = _regime_confidence(
            MarketRegimeType.TREND_UP, 60.0, 0.90, 1.0, config,
            sector_conc=compute_sector_concentration(full_rs, MarketRegimeType.TREND_UP),
        )
        assert conf <= 1.0

    def test_range_confidence_unaffected_by_sector_rs(self):
        config = _make_config()
        # All sectors positive — but RANGE confidence should stay flat 0.55
        full_rs = {f"S{i}": 2.0 for i in range(10)}
        conf = _regime_confidence(
            MarketRegimeType.RANGE, 15.0, 0.50, 1.0, config,
            sector_conc=compute_sector_concentration(full_rs, MarketRegimeType.RANGE),
        )
        assert conf == pytest.approx(0.55)

    def test_classify_regime_populates_sector_concentration(self):
        """End-to-end: sector_concentration field is set when sector_rs is passed."""
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = _make_processed(n_tickers=10, adx=30.0, breadth_frac=0.70,
                                    config=config)
        tracker = RegimeTracker()
        sector_rs = {"IT": 2.0, "BANKING": 1.5, "AUTO": 0.8}  # all positive

        r = classify_regime(processed, 0.70, tracker, config, sector_rs=sector_rs)

        assert r.sector_concentration > 0.5, (
            "all-positive sector RS with TREND_UP should give concentration > 0.5"
        )

    def test_classify_regime_without_sector_rs_defaults_to_neutral(self):
        """sector_rs=None must leave sector_concentration at 0.5 (backward compat)."""
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = _make_processed(n_tickers=10, adx=30.0, breadth_frac=0.70,
                                    config=config)
        tracker = RegimeTracker()

        r = classify_regime(processed, 0.70, tracker, config, sector_rs=None)

        assert r.sector_concentration == pytest.approx(0.5)

    def test_classify_regime_with_empty_sector_rs_defaults_to_neutral(self):
        config = _make_config(REGIME_CONFIRM_BARS=1)
        processed = _make_processed(n_tickers=10, adx=30.0, breadth_frac=0.70,
                                    config=config)
        tracker = RegimeTracker()

        r = classify_regime(processed, 0.70, tracker, config, sector_rs={})

        assert r.sector_concentration == pytest.approx(0.5)
