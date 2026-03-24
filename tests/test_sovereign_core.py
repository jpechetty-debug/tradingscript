"""
tests/test_sovereign_core.py
============================
Pytest test suite for Sovereign Engine v13.0 core math.

Covers:
  - kelly_size()           — fat-tail Kelly position sizing
  - compute_dynamic_targets() — ATR-based stop/target/RR
  - classify_regime()      — all 5 regime states
  - calibrate_platt()      — round-trip Platt A/B fitting
  - composite_to_prob()    — sigmoid probability conversion
  - passes_liquidity()     — ADV + turnover gate
  - compute_rs()           — relative strength calculation
  - compute_breadth()      — market breadth (% above EMA50)
  - RegimeTracker          — history isolation & confirm logic
  - optimise_portfolio()   — sector cap + correlation filter

Run:
    pytest tests/ -v
    pytest tests/ --cov=. --cov-report=term-missing
"""

from __future__ import annotations

import sys
import os
import json
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ─────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP: stub heavy optional imports so tests run without live API keys
# or a full fyers / telegram setup.
# ─────────────────────────────────────────────────────────────────────────────

def _stub_modules():
    """Inject lightweight stubs for modules that require credentials or GPU."""
    stubs = {
        "fyers_apiv3":              types.ModuleType("fyers_apiv3"),
        "fyers_apiv3.fyersModel":   types.ModuleType("fyers_apiv3.fyersModel"),
    }
    stubs["fyers_apiv3.fyersModel"].FyersModel = MagicMock
    stubs["fyers_apiv3"].fyersModel = stubs["fyers_apiv3.fyersModel"]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

_stub_modules()

# Add project root to path so we can import screener directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub capital_scaler before screener import (sovereign_improvements may be absent)
_mock_scaler = MagicMock()
_mock_scaler.capital_fraction.return_value = 1.0
_mock_gate = MagicMock()
_mock_gate.passes.return_value = True

with patch.dict("sys.modules", {}):
    try:
        import sovereign_improvements  # noqa: F401 — let it load if present
    except Exception:
        pass

import screener as eng  # noqa: E402  — our target module

# Patch the capital_scaler the engine uses so Kelly tests are deterministic
eng.capital_scaler = _mock_scaler


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_price_series(n: int = 300, seed: int = 42) -> pd.Series:
    """Synthetic log-normal close price series."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, n)
    prices = 1000.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, name="Close")


def _make_daily_df(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Minimal OHLCV DataFrame with the columns Kelly sizing needs."""
    close = _make_price_series(n, seed)
    df = pd.DataFrame({
        "Open":   close * 0.999,
        "High":   close * 1.005,
        "Low":    close * 0.995,
        "Close":  close,
        "Volume": np.full(n, 1_000_000),
    })
    df["Vol_Avg_20"]     = df["Volume"].rolling(20).mean()
    df["Turnover_Avg_20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["EMA_50"]         = df["Close"].ewm(span=50, adjust=False).mean()
    df["ATR"]            = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR_50_mean"]    = df["ATR"].rolling(50).mean()
    df["ADX"]            = pd.Series(np.full(n, 28.0))  # strong trend by default
    return df.dropna().reset_index(drop=True)


def _make_processed(
    n_tickers: int = 5,
    adx: float = 28.0,
    breadth_above: float = 0.8,
) -> dict[str, pd.DataFrame]:
    """Synthetic processed dict for regime classification tests."""
    out = {}
    benchmark = eng.CONFIG["BENCHMARK"]
    for i in range(n_tickers):
        df = _make_daily_df(seed=i)
        # Force ADX to requested value
        df["ADX"] = adx
        # Force breadth: set Close above/below EMA_50 based on desired fraction
        threshold = int(n_tickers * breadth_above)
        if i < threshold:
            df["Close"] = df["EMA_50"] * 1.05
        else:
            df["Close"] = df["EMA_50"] * 0.95
        out[f"TICKER{i}.NS"] = df
    # Benchmark entry (must exist but be ignored in breadth calc)
    out[benchmark] = _make_daily_df(seed=99)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. KELLY SIZING
# ─────────────────────────────────────────────────────────────────────────────

class TestKellySize:
    """kelly_size(entry, stop, prob_win, rr, daily_df, regime)"""

    def setup_method(self):
        self.df = _make_daily_df()

    def test_returns_four_values(self):
        shares, risk, f, kurt = eng.kelly_size(1000, 950, 0.55, 2.5, self.df)
        assert isinstance(shares, int)
        assert isinstance(risk, float)
        assert isinstance(f, float)
        assert isinstance(kurt, float)

    def test_zero_rps_returns_zeros(self):
        """entry == stop → no risk per share → zero position."""
        shares, risk, f, kurt = eng.kelly_size(1000, 1000, 0.55, 2.5, self.df)
        assert shares == 0
        assert risk == 0.0

    def test_minimum_one_share(self):
        """Even a tiny Kelly fraction should return at least 1 share."""
        shares, *_ = eng.kelly_size(1000, 990, 0.53, 1.5, self.df)
        assert shares >= eng.CONFIG["KELLY_MIN_SHARES"]

    def test_higher_prob_win_yields_more_shares(self):
        s_low,  *_ = eng.kelly_size(1000, 950, 0.52, 2.0, self.df)
        s_high, *_ = eng.kelly_size(1000, 950, 0.70, 2.0, self.df)
        assert s_high >= s_low

    def test_wider_stop_yields_fewer_shares(self):
        """Wider stop → higher risk per share → fewer shares for same ₹ risk."""
        s_tight, *_ = eng.kelly_size(1000, 990, 0.60, 2.5, self.df)
        s_wide,  *_ = eng.kelly_size(1000, 950, 0.60, 2.5, self.df)
        assert s_tight >= s_wide

    def test_kurt_correction_is_in_valid_range(self):
        *_, kurt = eng.kelly_size(1000, 950, 0.60, 2.5, self.df)
        assert 0.0 < kurt <= 1.0

    def test_short_history_uses_fallback_kurtosis(self):
        tiny_df = _make_daily_df(n=50)
        shares, *_ = eng.kelly_size(500, 480, 0.60, 2.5, tiny_df)
        assert shares >= 0  # should not raise

    def test_capital_fraction_zero_gives_zero_shares(self):
        """PANIC regime → capital_fraction = 0 → no new positions."""
        _mock_scaler.capital_fraction.return_value = 0.0
        try:
            shares, *_ = eng.kelly_size(1000, 950, 0.60, 2.5, self.df, regime="PANIC")
            assert shares == 0
        finally:
            _mock_scaler.capital_fraction.return_value = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE DYNAMIC TARGETS
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeDynamicTargets:
    """compute_dynamic_targets(direction, close, atr, val, vah)"""

    def test_long_stop_is_below_entry(self):
        stop, t1, t2, rr = eng.compute_dynamic_targets("LONG", 1000, 20, 980, 1060)
        assert stop < 1000

    def test_short_stop_is_above_entry(self):
        stop, t1, t2, rr = eng.compute_dynamic_targets("SHORT", 1000, 20, 940, 1020)
        assert stop > 1000

    def test_long_targets_are_above_entry(self):
        stop, t1, t2, rr = eng.compute_dynamic_targets("LONG", 1000, 20, 980, 1060)
        assert t1 > 1000
        assert t2 > 1000

    def test_short_targets_are_below_entry(self):
        stop, t1, t2, rr = eng.compute_dynamic_targets("SHORT", 1000, 20, 940, 1020)
        assert t1 < 1000
        assert t2 < 1000

    def test_t2_is_further_than_t1_long(self):
        stop, t1, t2, rr = eng.compute_dynamic_targets("LONG", 1000, 20, 980, 1040)
        assert t2 >= t1

    def test_t2_is_further_than_t1_short(self):
        stop, t1, t2, rr = eng.compute_dynamic_targets("SHORT", 1000, 20, 940, 1020)
        assert t2 <= t1

    def test_rr_is_positive(self):
        stop, t1, t2, rr = eng.compute_dynamic_targets("LONG", 1000, 20, 980, 1060)
        assert rr > 0

    def test_rr_math_long(self):
        """RR = (T1 - entry) / (entry - stop). Verify within 0.05 tolerance."""
        atr = 20
        close = 1000
        val, vah = 960, 1080  # vah far enough to be used as T1
        stop, t1, t2, rr = eng.compute_dynamic_targets("LONG", close, atr, val, vah)
        sl_dist = close - stop
        expected_rr = (t1 - close) / sl_dist if sl_dist > 0 else 0
        assert abs(rr - expected_rr) < 0.05

    def test_zero_atr_does_not_raise(self):
        """Edge case: zero ATR should not crash (may produce rr=0)."""
        try:
            stop, t1, t2, rr = eng.compute_dynamic_targets("LONG", 1000, 0, 1000, 1000)
            assert rr == 0.0
        except ZeroDivisionError:
            pytest.fail("compute_dynamic_targets raised ZeroDivisionError on zero ATR")


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGIME CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyRegime:
    """classify_regime(processed, breadth, tracker) → MarketRegime"""

    def _run(self, adx, breadth, atr_ratio=1.0, confirm_bars=1):
        processed = _make_processed(n_tickers=10, adx=adx, breadth_above=breadth)
        # Override ATR_50_mean to force specific atr_ratio
        bench = eng.CONFIG["BENCHMARK"]
        for t, df in processed.items():
            if t == bench:
                continue
            df["ATR_50_mean"] = df["ATR"] / atr_ratio
        tracker = eng.RegimeTracker()
        old_bars = eng.CONFIG["REGIME_CONFIRM_BARS"]
        eng.CONFIG["REGIME_CONFIRM_BARS"] = confirm_bars
        try:
            regime = eng.classify_regime(processed, breadth, tracker)
        finally:
            eng.CONFIG["REGIME_CONFIRM_BARS"] = old_bars
        return regime

    def test_panic_when_breadth_very_low(self):
        r = self._run(adx=30, breadth=0.15)
        assert r.regime == "PANIC"

    def test_trend_up_high_adx_high_breadth(self):
        r = self._run(adx=32, breadth=0.75)
        assert r.regime == "TREND_UP"

    def test_trend_down_high_adx_low_breadth(self):
        r = self._run(adx=32, breadth=0.30)
        assert r.regime == "TREND_DOWN"

    def test_range_when_adx_very_low(self):
        r = self._run(adx=12, breadth=0.50)
        assert r.regime == "RANGE"

    def test_expansion_high_adx_high_atr_ratio(self):
        r = self._run(adx=32, breadth=0.65, atr_ratio=1.5)
        assert r.regime == "EXPANSION"

    def test_regime_has_confidence_between_zero_and_one(self):
        r = self._run(adx=28, breadth=0.60)
        assert 0.0 < r.confidence <= 1.0

    def test_confirmed_after_enough_bars(self):
        """Push same regime confirm_bars times → confirmed == True."""
        processed = _make_processed(n_tickers=10, adx=32, breadth_above=0.75)
        tracker = eng.RegimeTracker()
        confirm = eng.CONFIG["REGIME_CONFIRM_BARS"]
        for _ in range(confirm):
            r = eng.classify_regime(processed, 0.75, tracker)
        assert r.confirmed is True

    def test_unconfirmed_after_fewer_than_confirm_bars(self):
        processed = _make_processed(n_tickers=10, adx=32, breadth_above=0.75)
        tracker = eng.RegimeTracker()
        old = eng.CONFIG["REGIME_CONFIRM_BARS"]
        eng.CONFIG["REGIME_CONFIRM_BARS"] = 5
        try:
            r = eng.classify_regime(processed, 0.75, tracker)  # only 1 push
            assert r.confirmed is False
        finally:
            eng.CONFIG["REGIME_CONFIRM_BARS"] = old

    def test_empty_processed_does_not_raise(self):
        """Only benchmark present — should fall back gracefully."""
        bench = eng.CONFIG["BENCHMARK"]
        processed = {bench: _make_daily_df()}
        tracker = eng.RegimeTracker()
        r = eng.classify_regime(processed, 0.5, tracker)
        assert r.regime in {"PANIC", "TREND_UP", "TREND_DOWN", "RANGE", "EXPANSION"}


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLATT CALIBRATION ROUND-TRIP
# ─────────────────────────────────────────────────────────────────────────────

class TestPlattCalibration:
    """calibrate_platt(composites, outcomes) and composite_to_prob()"""

    def test_returns_two_floats(self):
        comps = np.linspace(-1, 1, 50)
        outcomes = (comps > 0).astype(float)
        a, b = eng.calibrate_platt(comps, outcomes)
        assert isinstance(a, float)
        assert isinstance(b, float)

    def test_fewer_than_10_samples_returns_defaults(self):
        comps = np.array([0.1, 0.2, 0.3])
        outcomes = np.array([1, 0, 1])
        a, b = eng.calibrate_platt(comps, outcomes)
        assert a == eng.CONFIG["PLATT_A"]
        assert b == eng.CONFIG["PLATT_B"]

    def test_calibrated_ab_produce_valid_probabilities(self):
        rng = np.random.default_rng(7)
        comps = rng.uniform(-1, 1, 100)
        outcomes = (comps + rng.normal(0, 0.3, 100) > 0).astype(float)
        a, b = eng.calibrate_platt(comps, outcomes)

        old_a, old_b = eng.CONFIG["PLATT_A"], eng.CONFIG["PLATT_B"]
        eng.CONFIG["PLATT_A"], eng.CONFIG["PLATT_B"] = a, b
        try:
            for c in np.linspace(-1, 1, 21):
                p = eng.composite_to_prob(c)
                assert 0.0 < p < 1.0, f"Probability {p} out of range for composite={c}"
        finally:
            eng.CONFIG["PLATT_A"], eng.CONFIG["PLATT_B"] = old_a, old_b

    def test_higher_composite_gives_higher_probability(self):
        """Monotonicity check for default Platt params."""
        probs = [eng.composite_to_prob(c) for c in [-1.0, -0.5, 0.0, 0.5, 1.0]]
        assert probs == sorted(probs), f"Probabilities not monotone: {probs}"

    def test_composite_zero_gives_probability_near_half(self):
        """With default params (A=-4, B=2), composite=0.5 should be ≈ sigmoid(0)=0.5."""
        # composite=B/(-A) makes A*c+B=0 → prob=0.5
        mid = eng.CONFIG["PLATT_B"] / (-eng.CONFIG["PLATT_A"])
        p = eng.composite_to_prob(mid)
        assert abs(p - 0.5) < 0.01, f"Expected ~0.5, got {p}"

    def test_round_trip_a_b_values_are_reasonable(self):
        """Fitted A should be negative (higher composite → higher prob)."""
        rng = np.random.default_rng(13)
        comps = rng.uniform(-1, 1, 200)
        # Clean signal: outcomes mostly follow composite sign
        outcomes = ((comps + rng.normal(0, 0.2, 200)) > 0).astype(float)
        a, b = eng.calibrate_platt(comps, outcomes)
        assert a < 0, f"Fitted A={a} should be negative for positive IC"


# ─────────────────────────────────────────────────────────────────────────────
# 5. LIQUIDITY GATE
# ─────────────────────────────────────────────────────────────────────────────

class TestPassesLiquidity:
    """passes_liquidity(row) → (bool, reason_str)"""

    def _row(self, vol=1_000_000, turnover=50_000_000):
        return pd.Series({"Vol_Avg_20": vol, "Turnover_Avg_20": turnover})

    def test_passes_when_both_above_floor(self):
        ok, reason = eng.passes_liquidity(self._row())
        assert ok is True
        assert reason == ""

    def test_fails_when_volume_below_floor(self):
        ok, reason = eng.passes_liquidity(self._row(vol=100_000))
        assert ok is False
        assert "Vol" in reason

    def test_fails_when_turnover_below_floor(self):
        ok, reason = eng.passes_liquidity(self._row(turnover=10_000_000))
        assert ok is False
        assert "Turnover" in reason

    def test_fails_volume_takes_priority_over_turnover(self):
        """Both below floor — volume checked first in the function."""
        ok, reason = eng.passes_liquidity(self._row(vol=100, turnover=100))
        assert ok is False
        assert "Vol" in reason

    def test_exact_floor_volume_passes(self):
        ok, _ = eng.passes_liquidity(self._row(vol=eng.CONFIG["ADV_SHARE_FLOOR"]))
        # The check is strict <, so exactly at the floor should pass
        assert ok is True

    def test_exact_floor_turnover_passes(self):
        ok, _ = eng.passes_liquidity(self._row(turnover=eng.CONFIG["ADV_TURNOVER_FLOOR"]))
        assert ok is True


# ─────────────────────────────────────────────────────────────────────────────
# 6. RELATIVE STRENGTH
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeRS:
    """compute_rs(stock, bench, lookback) → float"""

    def setup_method(self):
        n = 150
        self.bench = _make_price_series(n, seed=0)

    def test_outperforming_stock_has_positive_rs(self):
        stock = self.bench.copy()
        stock.iloc[-1] *= 1.05  # force recent jump
        rs = eng.compute_rs(stock, self.bench, lookback=20)
        assert rs > 0

    def test_underperforming_stock_has_negative_rs(self):
        stock = self.bench.copy()
        stock.iloc[-1] *= 0.95  # force recent drop
        rs = eng.compute_rs(stock, self.bench, lookback=20)
        assert rs < 0

    def test_same_series_gives_zero_rs(self):
        rs = eng.compute_rs(self.bench, self.bench)
        assert abs(rs) < 1e-6

    def test_too_short_series_returns_zero(self):
        short = _make_price_series(5)
        rs = eng.compute_rs(short, self.bench, lookback=20)
        assert rs == 0.0

    def test_custom_lookback_respected(self):
        stock = self.bench.copy()
        # Progressive outperformance
        for i in range(len(stock)):
            stock.iloc[i] *= (1.0001 ** i)
        rs_short = eng.compute_rs(stock, self.bench, lookback=5)
        rs_long  = eng.compute_rs(stock, self.bench, lookback=40)
        assert rs_short != rs_long


# ─────────────────────────────────────────────────────────────────────────────
# 7. MARKET BREADTH
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBreadth:
    """compute_breadth(processed) → float in [0, 1]"""

    def test_all_above_ema_gives_one(self):
        processed = _make_processed(n_tickers=10, breadth_above=1.0)
        b = eng.compute_breadth(processed)
        assert b == pytest.approx(1.0, abs=0.01)

    def test_all_below_ema_gives_zero(self):
        processed = _make_processed(n_tickers=10, breadth_above=0.0)
        b = eng.compute_breadth(processed)
        assert b == pytest.approx(0.0, abs=0.01)

    def test_half_above_gives_half(self):
        processed = _make_processed(n_tickers=10, breadth_above=0.5)
        b = eng.compute_breadth(processed)
        assert 0.4 < b < 0.6

    def test_benchmark_excluded(self):
        """Benchmark ticker should not count toward breadth."""
        processed = _make_processed(n_tickers=4, breadth_above=1.0)
        bench = eng.CONFIG["BENCHMARK"]
        # Force benchmark below EMA
        processed[bench]["Close"] = processed[bench]["EMA_50"] * 0.5
        b = eng.compute_breadth(processed)
        assert b == pytest.approx(1.0, abs=0.01)

    def test_empty_processed_returns_fallback(self):
        bench = eng.CONFIG["BENCHMARK"]
        b = eng.compute_breadth({bench: _make_daily_df()})
        assert b == 0.5  # fallback when no non-benchmark tickers


# ─────────────────────────────────────────────────────────────────────────────
# 8. REGIME TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeTracker:
    """RegimeTracker — history isolation, confirm logic, PANIC fast-path."""

    def test_fresh_tracker_not_confirmed(self):
        t = eng.RegimeTracker()
        t.push("TREND_UP")
        assert t.is_confirmed("TREND_UP", confirm_bars=2) is False

    def test_confirmed_after_enough_bars(self):
        t = eng.RegimeTracker()
        t.push("TREND_UP")
        t.push("TREND_UP")
        assert t.is_confirmed("TREND_UP", confirm_bars=2) is True

    def test_mixed_history_not_confirmed(self):
        t = eng.RegimeTracker()
        t.push("TREND_UP")
        t.push("RANGE")
        assert t.is_confirmed("TREND_UP", confirm_bars=2) is False

    def test_panic_always_confirmed(self):
        """PANIC regime should be confirmed immediately (capital protection)."""
        t = eng.RegimeTracker()
        t.push("PANIC")
        assert t.is_confirmed("PANIC", confirm_bars=5) is True

    def test_history_capped_at_max(self):
        t = eng.RegimeTracker(max_history=3)
        for _ in range(10):
            t.push("TREND_UP")
        assert len(t._history) == 3

    def test_two_instances_are_isolated(self):
        """Watch-mode regression test — two trackers must not share history."""
        t1 = eng.RegimeTracker()
        t2 = eng.RegimeTracker()
        t1.push("TREND_UP")
        t1.push("TREND_UP")
        assert t2.is_confirmed("TREND_UP", confirm_bars=2) is False

    def test_last_returns_none_when_empty(self):
        t = eng.RegimeTracker()
        assert t.last() is None

    def test_last_returns_most_recent(self):
        t = eng.RegimeTracker()
        t.push("RANGE")
        t.push("TREND_UP")
        assert t.last() == "TREND_UP"


# ─────────────────────────────────────────────────────────────────────────────
# 9. PORTFOLIO OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimisePortfolio:
    """optimise_portfolio(candidates, corr_matrix)"""

    def setup_method(self):
        eng.CONFIG["PORTFOLIO_SIZE"] = 5
        eng.CONFIG["MAX_SECTOR_PICKS"] = 2
        eng.CONFIG["MAX_CORR"] = 0.7

    def _make_result(self, ticker, sector, sharpe=1.0):
        class MockResult:
            def __init__(self, t, s, sr):
                self.ticker = t
                self.sector = s
                self.sharpe_rank = sr
        return MockResult(ticker, sector, sharpe)

    def test_respects_portfolio_size_cap(self):
        eng.CONFIG["PORTFOLIO_SIZE"] = 5
        eng.CONFIG["MAX_SECTOR_PICKS"] = 10 
        n = eng.CONFIG["PORTFOLIO_SIZE"] + 5
        candidates = [self._make_result(f"T{i}", "BANKS") for i in range(n)]
        selected = eng.optimise_portfolio(candidates, pd.DataFrame())
        assert len(selected) <= 5

    def test_respects_sector_cap(self):
        eng.CONFIG["MAX_SECTOR_PICKS"] = 2
        candidates = [self._make_result(f"T{i}", "BANKS", sharpe=10 - i)
                      for i in range(10)]
        selected = eng.optimise_portfolio(candidates, pd.DataFrame())
        assert len(selected) == 2

    def test_empty_candidates_returns_empty(self):
        assert eng.optimise_portfolio([], pd.DataFrame()) == []

    def test_higher_sharpe_ranked_first(self):
        candidates = [
            self._make_result("LOW",  "IT",    sharpe=0.5),
            self._make_result("HIGH", "BANKS", sharpe=2.0),
        ]
        selected = eng.optimise_portfolio(candidates, pd.DataFrame())
        assert selected[0].ticker == "HIGH"

    def test_correlated_ticker_excluded(self):
        """A pair with abs(corr) > MAX_CORR — second should be dropped."""
        max_corr = eng.CONFIG["MAX_CORR"]
        t1 = self._make_result("RELIANCE", "ENERGY", sharpe=2.0)
        t2 = self._make_result("ONGC",     "ENERGY", sharpe=1.5)

        # Build a correlation matrix where RELIANCE.NS and ONGC.NS are highly correlated
        tickers = ["RELIANCE.NS", "ONGC.NS"]
        corr_val = max_corr + 0.10
        corr_matrix = pd.DataFrame(
            [[1.0, corr_val], [corr_val, 1.0]],
            index=tickers, columns=tickers,
        )
        selected = eng.optimise_portfolio([t1, t2], corr_matrix)
        selected_tickers = [s.ticker for s in selected]
        assert "RELIANCE" in selected_tickers
        assert "ONGC" not in selected_tickers

    def test_uncorrelated_tickers_both_included(self):
        t1 = self._make_result("INFY", "IT",    sharpe=2.0)
        t2 = self._make_result("SBIN", "BANKS", sharpe=1.8)

        tickers = ["INFY.NS", "SBIN.NS"]
        corr_matrix = pd.DataFrame(
            [[1.0, 0.10], [0.10, 1.0]],
            index=tickers, columns=tickers,
        )
        selected = eng.optimise_portfolio([t1, t2], corr_matrix)
        assert len(selected) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 10. INTEGRATION SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationSmoke:
    """Light end-to-end path: regime → kelly → targets, no network calls."""

    def test_full_single_ticker_pipeline(self):
        """Ensure no exceptions through the core math path for one ticker."""
        df = _make_daily_df(n=300)
        bench = _make_price_series(300)

        # Breadth & regime
        processed = {
            "SBIN.NS": df,
            eng.CONFIG["BENCHMARK"]: df.copy(),
        }
        breadth = eng.compute_breadth(processed)
        tracker = eng.RegimeTracker()
        regime = eng.classify_regime(processed, breadth, tracker)

        # Targets
        close = float(df["Close"].iloc[-1])
        atr   = float(df["ATR"].iloc[-1])
        stop, t1, t2, rr = eng.compute_dynamic_targets(
            "LONG", close, atr, close * 0.97, close * 1.05
        )

        # Kelly
        prob = eng.composite_to_prob(0.65)
        shares, risk, f, kurt = eng.kelly_size(close, stop, prob, rr, df, regime.regime)

        assert regime.regime in {"PANIC", "TREND_UP", "TREND_DOWN", "RANGE", "EXPANSION"}
        assert shares >= 0
        assert rr >= 0
        assert 0 < prob < 1
