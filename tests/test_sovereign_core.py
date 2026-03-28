"""
tests/test_sovereign_core.py
============================
Pytest test suite for Sovereign Engine v14 core math — ported from v13.

Covers:
  - calculate_kelly_size()   — fat-tail Kelly position sizing
  - compute_targets()        — ATR-based stop/target/RR
  - classify_regime()        — all 5 regime states
  - calibrate_platt()        — round-trip Platt A/B fitting
  - composite_to_prob()      — sigmoid probability conversion
  - passes_liquidity()       — ADV + turnover gate
  - compute_rs()             — relative strength calculation
  - compute_breadth()        — market breadth (% above EMA50)
  - RegimeTracker            — history isolation & confirm logic
  - optimize_portfolio()     — sector cap + correlation filter

Run:
    pytest tests/ -v
    pytest tests/ --cov=core --cov-report=term-missing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import CONFIG, SystemConfig
from core.portfolio import (
    calculate_kelly_size,
    compute_targets,
    optimize_portfolio,
)
from core.regime import (
    RegimeTracker,
    classify_regime,
    compute_breadth,
    compute_rs,
)
from core.scorer import (
    calibrate_platt,
    composite_to_prob,
    passes_liquidity,
)


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
    df["Vol_Avg_20"]      = df["Volume"].rolling(20).mean()
    df["Turnover_Avg_20"] = (df["Close"] * df["Volume"]).rolling(20).mean()
    df["EMA_50"]          = df["Close"].ewm(span=50, adjust=False).mean()
    df["ATR"]             = (df["High"] - df["Low"]).rolling(14).mean()
    df["ATR_50_mean"]     = df["ATR"].rolling(50).mean()
    df["ADX"]             = pd.Series(np.full(n, 28.0))  # strong trend by default
    return df.dropna().reset_index(drop=True)


def _make_processed(
    n_tickers: int = 5,
    adx: float = 28.0,
    breadth_above: float = 0.8,
) -> dict[str, pd.DataFrame]:
    """Synthetic processed dict for regime classification tests."""
    out = {}
    benchmark = CONFIG.BENCHMARK
    for i in range(n_tickers):
        df = _make_daily_df(seed=i)
        df["ADX"] = adx
        threshold = int(n_tickers * breadth_above)
        if i < threshold:
            df["Close"] = df["EMA_50"] * 1.05
        else:
            df["Close"] = df["EMA_50"] * 0.95
        out[f"TICKER{i}.NS"] = df
    out[benchmark] = _make_daily_df(seed=99)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. KELLY SIZING
# ─────────────────────────────────────────────────────────────────────────────

class TestKellySize:
    """calculate_kelly_size(entry, stop, prob_win, rr, daily_df, config)"""

    def setup_method(self):
        self.df = _make_daily_df()

    def test_returns_four_values(self):
        shares, risk, f, kurt = calculate_kelly_size(1000, 950, 0.55, 2.5, self.df, CONFIG)
        assert isinstance(shares, int)
        assert isinstance(risk, (int, float))  # v14 returns rounded float or int
        assert isinstance(f, float)
        assert isinstance(kurt, float)

    def test_zero_rps_returns_zeros(self):
        """entry == stop → no risk per share → zero position."""
        shares, risk, f, kurt = calculate_kelly_size(1000, 1000, 0.55, 2.5, self.df, CONFIG)
        assert shares == 0
        assert risk == 0.0

    def test_minimum_one_share(self):
        """Even a tiny Kelly fraction should return at least MIN_SHARES."""
        shares, *_ = calculate_kelly_size(1000, 990, 0.53, 1.5, self.df, CONFIG)
        assert shares >= CONFIG.KELLY_MIN_SHARES

    def test_higher_prob_win_yields_more_shares(self):
        s_low,  *_ = calculate_kelly_size(1000, 950, 0.52, 2.0, self.df, CONFIG)
        s_high, *_ = calculate_kelly_size(1000, 950, 0.70, 2.0, self.df, CONFIG)
        assert s_high >= s_low

    def test_wider_stop_yields_fewer_shares(self):
        """Wider stop → higher risk per share → fewer shares for same risk."""
        s_tight, *_ = calculate_kelly_size(1000, 990, 0.60, 2.5, self.df, CONFIG)
        s_wide,  *_ = calculate_kelly_size(1000, 950, 0.60, 2.5, self.df, CONFIG)
        assert s_tight >= s_wide

    def test_kurt_correction_is_in_valid_range(self):
        *_, kurt = calculate_kelly_size(1000, 950, 0.60, 2.5, self.df, CONFIG)
        assert 0.0 < kurt <= 1.0

    def test_short_history_uses_fallback_kurtosis(self):
        tiny_df = _make_daily_df(n=50)
        shares, *_ = calculate_kelly_size(500, 480, 0.60, 2.5, tiny_df, CONFIG)
        assert shares >= 0

    def test_capital_fraction_zero_gives_zero_shares(self):
        """PANIC regime → capital_fraction = 0 → no new positions."""
        shares, *_ = calculate_kelly_size(
            1000, 950, 0.60, 2.5, self.df, CONFIG,
            regime="PANIC", capital_fraction=0.0,
        )
        assert shares == 0


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE TARGETS
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTargets:
    """compute_targets(direction, close, atr, config) → TradeTargets"""

    def test_long_stop_is_below_entry(self):
        t = compute_targets("LONG", 1000, 20, CONFIG)
        assert t.stop < 1000

    def test_short_stop_is_above_entry(self):
        t = compute_targets("SHORT", 1000, 20, CONFIG)
        assert t.stop > 1000

    def test_long_targets_are_above_entry(self):
        t = compute_targets("LONG", 1000, 20, CONFIG)
        assert t.t1 > 1000
        assert t.t2 > 1000

    def test_short_targets_are_below_entry(self):
        t = compute_targets("SHORT", 1000, 20, CONFIG)
        assert t.t1 < 1000
        assert t.t2 < 1000

    def test_t2_is_further_than_t1_long(self):
        t = compute_targets("LONG", 1000, 20, CONFIG)
        assert t.t2 >= t.t1

    def test_t2_is_further_than_t1_short(self):
        t = compute_targets("SHORT", 1000, 20, CONFIG)
        assert t.t2 <= t.t1

    def test_rr_is_positive(self):
        t = compute_targets("LONG", 1000, 20, CONFIG)
        assert t.rr > 0

    def test_rr_math_long(self):
        """RR = (T1 - entry) / (entry - stop). Verify within 0.05 tolerance."""
        t = compute_targets("LONG", 1000, 20, CONFIG)
        sl_dist = 1000 - t.stop
        expected_rr = (t.t1 - 1000) / sl_dist if sl_dist > 0 else 0
        assert abs(t.rr - expected_rr) < 0.05

    def test_zero_atr_does_not_raise(self):
        """Edge case: zero ATR should not crash (may produce rr=0)."""
        try:
            t = compute_targets("LONG", 1000, 0, CONFIG)
            assert t.rr == 0.0
        except ZeroDivisionError:
            pytest.fail("compute_targets raised ZeroDivisionError on zero ATR")


# ─────────────────────────────────────────────────────────────────────────────
# 3. REGIME CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyRegime:
    """classify_regime(processed, breadth, tracker, config) → MarketRegime"""

    # v14: regime is a MarketRegimeType enum — compare via .value
    VALID_REGIMES = {"PANIC", "TREND_UP", "TREND_DOWN", "RANGE", "EXPANSION"}

    def _run(self, adx, breadth, atr_ratio=1.0, confirm_bars=1):
        processed = _make_processed(n_tickers=10, adx=adx, breadth_above=breadth)
        bench = CONFIG.BENCHMARK
        for t, df in processed.items():
            if t == bench:
                continue
            df["ATR_50_mean"] = df["ATR"] / atr_ratio

        config = SystemConfig(**{
            **CONFIG.__dict__,
            "REGIME_CONFIRM_BARS": confirm_bars,
        })
        tracker = RegimeTracker()
        regime = classify_regime(processed, breadth, tracker, config)
        return regime

    def test_panic_when_breadth_very_low(self):
        r = self._run(adx=30, breadth=0.15)
        assert r.regime.value == "PANIC"

    def test_trend_up_high_adx_high_breadth(self):
        r = self._run(adx=32, breadth=0.75)
        assert r.regime.value == "TREND_UP"

    def test_trend_down_high_adx_low_breadth(self):
        r = self._run(adx=32, breadth=0.30)
        assert r.regime.value == "TREND_DOWN"

    def test_range_when_adx_very_low(self):
        r = self._run(adx=12, breadth=0.50)
        assert r.regime.value == "RANGE"

    def test_expansion_high_adx_high_atr_ratio(self):
        r = self._run(adx=32, breadth=0.65, atr_ratio=1.5)
        assert r.regime.value == "EXPANSION"

    def test_regime_has_confidence_between_zero_and_one(self):
        r = self._run(adx=28, breadth=0.60)
        assert 0.0 < r.confidence <= 1.0

    def test_confirmed_after_enough_bars(self):
        """Push same regime confirm_bars times → confirmed == True."""
        processed = _make_processed(n_tickers=10, adx=32, breadth_above=0.75)
        tracker = RegimeTracker()
        confirm = CONFIG.REGIME_CONFIRM_BARS
        for _ in range(confirm):
            r = classify_regime(processed, 0.75, tracker, CONFIG)
        assert r.confirmed is True

    def test_unconfirmed_after_fewer_than_confirm_bars(self):
        processed = _make_processed(n_tickers=10, adx=32, breadth_above=0.75)
        tracker = RegimeTracker()
        config = SystemConfig(**{**CONFIG.__dict__, "REGIME_CONFIRM_BARS": 5})
        r = classify_regime(processed, 0.75, tracker, config)
        assert r.confirmed is False

    def test_empty_processed_does_not_raise(self):
        """Only benchmark present — should fall back gracefully."""
        bench = CONFIG.BENCHMARK
        processed = {bench: _make_daily_df()}
        tracker = RegimeTracker()
        r = classify_regime(processed, 0.5, tracker, CONFIG)
        assert r.regime.value in self.VALID_REGIMES


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLATT CALIBRATION ROUND-TRIP
# ─────────────────────────────────────────────────────────────────────────────

class TestPlattCalibration:
    """calibrate_platt(composites, outcomes) and composite_to_prob()"""

    def test_returns_two_floats(self):
        # v14 requires ≥80 samples for validation holdout
        rng = np.random.default_rng(42)
        comps = rng.uniform(-1, 1, 200).tolist()
        outcomes = [1 if c > 0 else 0 for c in comps]
        a, b = calibrate_platt(comps, outcomes)
        assert isinstance(a, float)
        assert isinstance(b, float)

    def test_fewer_than_min_samples_raises(self):
        """v14 raises ValueError when samples < calib_offset + 20."""
        comps = [0.1, 0.2, 0.3]
        outcomes = [1, 0, 1]
        with pytest.raises(ValueError, match="need >= 80 samples"):
            calibrate_platt(comps, outcomes)

    def test_calibrated_ab_produce_valid_probabilities(self):
        rng = np.random.default_rng(7)
        comps = rng.uniform(-1, 1, 200).tolist()
        outcomes = [1 if (c + rng.normal(0, 0.3)) > 0 else 0 for c in comps]
        a, b = calibrate_platt(comps, outcomes)

        for c in np.linspace(-1, 1, 21):
            p = composite_to_prob(float(c), a, b)
            assert 0.0 < p < 1.0, f"Probability {p} out of range for composite={c}"

    def test_monotonicity_across_composites(self):
        """v14 sigmoid convention: prob is monotone across composites."""
        probs = [composite_to_prob(c, CONFIG.PLATT_A, CONFIG.PLATT_B)
                 for c in [-1.0, -0.5, 0.0, 0.5, 1.0]]
        # With default A=-4, B=2 the sigmoid is decreasing
        assert probs == sorted(probs, reverse=True), f"Probabilities not monotone-decreasing: {probs}"

    def test_composite_zero_gives_probability_near_half(self):
        """With default params (A=-4, B=2), composite=0.5 → sigmoid(0)=0.5."""
        mid = CONFIG.PLATT_B / (-CONFIG.PLATT_A)
        p = composite_to_prob(mid, CONFIG.PLATT_A, CONFIG.PLATT_B)
        assert abs(p - 0.5) < 0.01, f"Expected ~0.5, got {p}"

    def test_round_trip_a_b_values_are_reasonable(self):
        """Fitted A should be non-zero (sigmoid has discriminative power)."""
        rng = np.random.default_rng(13)
        comps = rng.uniform(-1, 1, 200).tolist()
        outcomes = [1 if (c + rng.normal(0, 0.2)) > 0 else 0 for c in comps]
        a, b = calibrate_platt(comps, outcomes)
        assert a != 0, f"Fitted A={a} should be non-zero for discriminative sigmoid"


# ─────────────────────────────────────────────────────────────────────────────
# 5. LIQUIDITY GATE
# ─────────────────────────────────────────────────────────────────────────────

class TestPassesLiquidity:
    """passes_liquidity(row, config) → (bool, reason_str)"""

    def _row(self, vol=1_000_000, turnover=50_000_000):
        return pd.Series({"Vol_Avg_20": vol, "Turnover_Avg_20": turnover})

    def test_passes_when_both_above_floor(self):
        ok, reason = passes_liquidity(self._row(), CONFIG)
        assert ok is True
        assert reason == ""

    def test_fails_when_volume_below_floor(self):
        ok, reason = passes_liquidity(self._row(vol=100_000), CONFIG)
        assert ok is False
        assert "Vol" in reason

    def test_fails_when_turnover_below_floor(self):
        ok, reason = passes_liquidity(self._row(turnover=10_000_000), CONFIG)
        assert ok is False
        assert "Turnover" in reason

    def test_fails_volume_takes_priority_over_turnover(self):
        """Both below floor — volume checked first in the function."""
        ok, reason = passes_liquidity(self._row(vol=100, turnover=100), CONFIG)
        assert ok is False
        assert "Vol" in reason

    def test_exact_floor_volume_passes(self):
        ok, _ = passes_liquidity(self._row(vol=CONFIG.ADV_SHARE_FLOOR), CONFIG)
        assert ok is True

    def test_exact_floor_turnover_passes(self):
        ok, _ = passes_liquidity(self._row(turnover=CONFIG.ADV_TURNOVER_FLOOR), CONFIG)
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
        stock.iloc[-1] *= 1.05
        rs = compute_rs(stock, self.bench, lookback=20)
        assert rs > 0

    def test_underperforming_stock_has_negative_rs(self):
        stock = self.bench.copy()
        stock.iloc[-1] *= 0.95
        rs = compute_rs(stock, self.bench, lookback=20)
        assert rs < 0

    def test_same_series_gives_zero_rs(self):
        rs = compute_rs(self.bench, self.bench)
        assert abs(rs) < 1e-6

    def test_too_short_series_returns_zero(self):
        short = _make_price_series(5)
        rs = compute_rs(short, self.bench, lookback=20)
        assert rs == 0.0

    def test_custom_lookback_respected(self):
        stock = self.bench.copy()
        for i in range(len(stock)):
            stock.iloc[i] *= (1.0001 ** i)
        rs_short = compute_rs(stock, self.bench, lookback=5)
        rs_long  = compute_rs(stock, self.bench, lookback=40)
        assert rs_short != rs_long


# ─────────────────────────────────────────────────────────────────────────────
# 7. MARKET BREADTH
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeBreadth:
    """compute_breadth(processed, config) → float in [0, 1]"""

    def test_all_above_ema_gives_one(self):
        processed = _make_processed(n_tickers=10, breadth_above=1.0)
        b = compute_breadth(processed, CONFIG)
        assert b == pytest.approx(1.0, abs=0.01)

    def test_all_below_ema_gives_zero(self):
        processed = _make_processed(n_tickers=10, breadth_above=0.0)
        b = compute_breadth(processed, CONFIG)
        assert b == pytest.approx(0.0, abs=0.01)

    def test_half_above_gives_half(self):
        processed = _make_processed(n_tickers=10, breadth_above=0.5)
        b = compute_breadth(processed, CONFIG)
        assert 0.4 < b < 0.6

    def test_benchmark_excluded(self):
        """Benchmark ticker should not count toward breadth."""
        processed = _make_processed(n_tickers=4, breadth_above=1.0)
        bench = CONFIG.BENCHMARK
        processed[bench]["Close"] = processed[bench]["EMA_50"] * 0.5
        b = compute_breadth(processed, CONFIG)
        assert b == pytest.approx(1.0, abs=0.01)

    def test_empty_processed_returns_fallback(self):
        bench = CONFIG.BENCHMARK
        b = compute_breadth({bench: _make_daily_df()}, CONFIG)
        assert b == 0.5  # fallback when no non-benchmark tickers


# ─────────────────────────────────────────────────────────────────────────────
# 8. REGIME TRACKER
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeTracker:
    """RegimeTracker — history isolation, confirm logic."""

    def test_fresh_tracker_not_confirmed(self):
        t = RegimeTracker()
        t.push("TREND_UP")
        assert t.is_confirmed("TREND_UP", confirm_bars=2) is False

    def test_confirmed_after_enough_bars(self):
        t = RegimeTracker()
        t.push("TREND_UP")
        t.push("TREND_UP")
        assert t.is_confirmed("TREND_UP", confirm_bars=2) is True

    def test_mixed_history_not_confirmed(self):
        t = RegimeTracker()
        t.push("TREND_UP")
        t.push("RANGE")
        assert t.is_confirmed("TREND_UP", confirm_bars=2) is False

    def test_panic_confirmed_after_one_bar(self):
        """PANIC should confirm after 1 push with confirm_bars=1."""
        t = RegimeTracker()
        t.push("PANIC")
        assert t.is_confirmed("PANIC", confirm_bars=1) is True

    def test_history_capped_at_max(self):
        t = RegimeTracker(max_history=3)
        for _ in range(10):
            t.push("TREND_UP")
        assert len(t._history) == 3

    def test_two_instances_are_isolated(self):
        """Watch-mode regression test — two trackers must not share history."""
        t1 = RegimeTracker()
        t2 = RegimeTracker()
        t1.push("TREND_UP")
        t1.push("TREND_UP")
        assert t2.is_confirmed("TREND_UP", confirm_bars=2) is False


# ─────────────────────────────────────────────────────────────────────────────
# 9. PORTFOLIO OPTIMISATION
# ─────────────────────────────────────────────────────────────────────────────

class TestOptimisePortfolio:
    """optimize_portfolio(candidates, config, corr_matrix)"""

    def _make_result(self, ticker, sector, sharpe=1.0):
        class MockResult:
            def __init__(self, t, s, sr):
                self.ticker = t
                self.sector = s
                self.sharpe_rank = sr
        return MockResult(ticker, sector, sharpe)

    def _config(self, **overrides) -> SystemConfig:
        defaults = {
            **CONFIG.__dict__,
            "PORTFOLIO_SIZE": 5,
            "MAX_SECTOR_PICKS": 2,
            "MAX_CORR": 0.7,
        }
        defaults.update(overrides)
        return SystemConfig(**defaults)

    def test_respects_portfolio_size_cap(self):
        cfg = self._config(PORTFOLIO_SIZE=5, MAX_SECTOR_PICKS=10)
        n = cfg.PORTFOLIO_SIZE + 5
        candidates = [self._make_result(f"T{i}", "BANKS") for i in range(n)]
        selected = optimize_portfolio(candidates, cfg, pd.DataFrame())
        assert len(selected) <= 5

    def test_respects_sector_cap(self):
        cfg = self._config(MAX_SECTOR_PICKS=2)
        candidates = [self._make_result(f"T{i}", "BANKS", sharpe=10 - i)
                      for i in range(10)]
        selected = optimize_portfolio(candidates, cfg, pd.DataFrame())
        assert len(selected) == 2

    def test_empty_candidates_returns_empty(self):
        cfg = self._config()
        assert optimize_portfolio([], cfg, pd.DataFrame()) == []

    def test_higher_sharpe_ranked_first(self):
        cfg = self._config()
        candidates = [
            self._make_result("LOW",  "IT",    sharpe=0.5),
            self._make_result("HIGH", "BANKS", sharpe=2.0),
        ]
        selected = optimize_portfolio(candidates, cfg, pd.DataFrame())
        assert selected[0].ticker == "HIGH"

    def test_correlated_ticker_excluded(self):
        """A pair with abs(corr) > MAX_CORR — second should be dropped."""
        cfg = self._config()
        t1 = self._make_result("RELIANCE", "ENERGY", sharpe=2.0)
        t2 = self._make_result("ONGC",     "ENERGY", sharpe=1.5)

        tickers = ["RELIANCE.NS", "ONGC.NS"]
        corr_val = cfg.MAX_CORR + 0.10
        corr_matrix = pd.DataFrame(
            [[1.0, corr_val], [corr_val, 1.0]],
            index=tickers, columns=tickers,
        )
        selected = optimize_portfolio([t1, t2], cfg, corr_matrix)
        selected_tickers = [s.ticker for s in selected]
        assert "RELIANCE" in selected_tickers
        assert "ONGC" not in selected_tickers

    def test_uncorrelated_tickers_both_included(self):
        cfg = self._config()
        t1 = self._make_result("INFY", "IT",    sharpe=2.0)
        t2 = self._make_result("SBIN", "BANKS", sharpe=1.8)

        tickers = ["INFY.NS", "SBIN.NS"]
        corr_matrix = pd.DataFrame(
            [[1.0, 0.10], [0.10, 1.0]],
            index=tickers, columns=tickers,
        )
        selected = optimize_portfolio([t1, t2], cfg, corr_matrix)
        assert len(selected) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 10. INTEGRATION SMOKE TEST
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationSmoke:
    """Light end-to-end path: regime → kelly → targets, no network calls."""

    def test_full_single_ticker_pipeline(self):
        """Ensure no exceptions through the core math path for one ticker."""
        df = _make_daily_df(n=300)

        processed = {
            "SBIN.NS": df,
            CONFIG.BENCHMARK: df.copy(),
        }
        breadth = compute_breadth(processed, CONFIG)
        tracker = RegimeTracker()
        regime = classify_regime(processed, breadth, tracker, CONFIG)

        close = float(df["Close"].iloc[-1])
        atr   = float(df["ATR"].iloc[-1])
        tt = compute_targets("LONG", close, atr, CONFIG)

        prob = composite_to_prob(0.65, CONFIG.PLATT_A, CONFIG.PLATT_B)
        shares, risk, f, kurt = calculate_kelly_size(
            close, tt.stop, prob, tt.rr, df, CONFIG, regime.regime.value,
        )

        assert regime.regime.value in {"PANIC", "TREND_UP", "TREND_DOWN", "RANGE", "EXPANSION"}
        assert shares >= 0
        assert tt.rr >= 0
        assert 0 < prob < 1
