"""
tests/test_indicators.py
=========================
Comprehensive unit tests for core/indicators.py.

Covers: EMA stack, MACD, RSI, StochRSI, ATR + percentile, Supertrend,
        ADX, Bollinger Bands, volume metrics, candle helpers, NaN cleanup.

Run:
    pytest tests/test_indicators.py -v
    pytest tests/test_indicators.py --cov=core/indicators --cov-report=term-missing
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
    stubs["fyers_apiv3.fyersModel"].FyersModel = MagicMock  # type: ignore[attr-defined]
    stubs["fyers_apiv3"].fyersModel = stubs["fyers_apiv3.fyersModel"]  # type: ignore[attr-defined]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

_stub_modules()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.indicators import add_indicators, DROPNA_COLS


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _config():
    cfg = MagicMock()
    cfg.SUPER_PERIOD = 10
    cfg.SUPER_MULT = 3.0
    cfg.ADX_PERIOD = 14
    return cfg


def _make_raw_ohlcv(
    n: int = 300,
    seed: int = 7,
    base: float = 500.0,
    drift: float = 0.001,
    vol_range: tuple[int, int] = (1_000_000, 3_000_000),
) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = base * np.cumprod(1 + drift + np.random.normal(0, 0.012, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.004, n))
    vol = np.random.randint(*vol_range, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )


def _make_trending_up(n: int = 300) -> pd.DataFrame:
    return _make_raw_ohlcv(n=n, seed=10, drift=0.004)


def _make_trending_down(n: int = 300) -> pd.DataFrame:
    return _make_raw_ohlcv(n=n, seed=20, drift=-0.004)


def _make_flat(n: int = 300) -> pd.DataFrame:
    return _make_raw_ohlcv(n=n, seed=30, drift=0.0)


@pytest.fixture
def cfg():
    return _config()


@pytest.fixture
def df_up(cfg):
    return add_indicators(_make_trending_up(), cfg)


@pytest.fixture
def df_down(cfg):
    return add_indicators(_make_trending_down(), cfg)


@pytest.fixture
def df_flat(cfg):
    return add_indicators(_make_flat(), cfg)


@pytest.fixture
def df_std(cfg):
    return add_indicators(_make_raw_ohlcv(), cfg)


# ─────────────────────────────────────────────────────────────────────────────
# EMA Stack
# ─────────────────────────────────────────────────────────────────────────────

class TestEMAStack:

    def test_ema_columns_present(self, df_std):
        for col in ("EMA_20", "EMA_50", "EMA_200"):
            assert col in df_std.columns

    def test_ema_ordering_uptrend(self, df_up):
        """In a strong uptrend, EMA20 > EMA50 > EMA200 on the last bar."""
        row = df_up.iloc[-1]
        assert row["EMA_20"] > row["EMA_50"] > row["EMA_200"]

    def test_ema_ordering_downtrend(self, df_down):
        """In a strong downtrend, EMA20 < EMA50 < EMA200 on the last bar."""
        row = df_down.iloc[-1]
        assert row["EMA_20"] < row["EMA_50"] < row["EMA_200"]

    def test_ema_convergence_flat(self, df_flat):
        """On flat data, EMAs should converge to roughly the same value."""
        row = df_flat.iloc[-1]
        spread = abs(row["EMA_20"] - row["EMA_200"]) / row["EMA_200"]
        assert spread < 0.05, f"EMA spread {spread:.4f} too wide for flat data"

    def test_ema20_responsive(self, cfg):
        """EMA_20 should react faster to a price jump than EMA_200."""
        df = _make_flat(200)
        # Spike the last 10 bars
        df.iloc[-10:, df.columns.get_loc("Close")] *= 1.10
        df.iloc[-10:, df.columns.get_loc("High")] *= 1.10
        out = add_indicators(df, cfg)
        row = out.iloc[-1]
        ema20_pct = (row["EMA_20"] - row["EMA_200"]) / row["EMA_200"]
        assert ema20_pct > 0.01, "EMA_20 should react more to recent spike"


# ─────────────────────────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────────────────────────

class TestMACD:

    def test_macd_hist_column_exists(self, df_std):
        assert "MACD_Hist" in df_std.columns

    def test_macd_positive_in_uptrend(self, df_up):
        """MACD histogram should be predominantly positive in an uptrend."""
        positive_pct = (df_up["MACD_Hist"].tail(50) > 0).mean()
        assert positive_pct > 0.35

    def test_macd_negative_in_downtrend(self, df_down):
        """MACD histogram should be predominantly negative in a downtrend."""
        negative_pct = (df_down["MACD_Hist"].tail(50) < 0).mean()
        assert negative_pct > 0.35

    def test_macd_hist_not_all_nan(self, df_std):
        assert df_std["MACD_Hist"].notna().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────────────────────────

class TestRSI:

    def test_rsi_bounded(self, df_std):
        rsi = df_std["RSI"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_rsi_high_in_uptrend(self, df_up):
        """RSI should be elevated in a strong uptrend."""
        assert df_up["RSI"].iloc[-1] > 50

    def test_rsi_low_in_downtrend(self, df_down):
        """RSI should be depressed in a strong downtrend."""
        assert df_down["RSI"].iloc[-1] < 50

    def test_constant_up_rsi_near_100(self, cfg):
        """A series of constant gains should push RSI toward 100."""
        n = 200
        np.random.seed(99)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 * np.cumprod(np.ones(n) * 1.01)  # 1% daily gain
        df = pd.DataFrame({
            "Open": close * 0.999, "High": close * 1.005,
            "Low": close * 0.998, "Close": close,
            "Volume": np.full(n, 1_500_000.0),
        }, index=dates)
        out = add_indicators(df, cfg)
        assert out["RSI"].iloc[-1] > 90


# ─────────────────────────────────────────────────────────────────────────────
# StochRSI
# ─────────────────────────────────────────────────────────────────────────────

class TestStochRSI:

    def test_stochrsi_columns_exist(self, df_std):
        assert "StochRSI_K" in df_std.columns
        assert "StochRSI_D" in df_std.columns

    def test_stochrsi_k_bounded(self, df_std):
        valid = df_std["StochRSI_K"].dropna()
        assert valid.between(0, 100).all(), f"StochRSI_K out of range: min={valid.min()}, max={valid.max()}"

    def test_stochrsi_d_is_smoothed_k(self, df_std):
        """StochRSI_D is a moving average of K, so D should be smoother."""
        k_std = df_std["StochRSI_K"].dropna().std()
        d_std = df_std["StochRSI_D"].dropna().std()
        assert d_std <= k_std * 1.1  # D should be equal or smoother


# ─────────────────────────────────────────────────────────────────────────────
# ATR + Percentile
# ─────────────────────────────────────────────────────────────────────────────

class TestATR:

    def test_atr_positive(self, df_std):
        assert (df_std["ATR"].dropna() > 0).all()

    def test_atr_50_mean_is_rolling_average(self, df_std):
        """ATR_50_mean should be close to the mean of ATR over last 50 bars."""
        atr_tail = df_std["ATR"].tail(50).mean()
        atr_50_last = df_std["ATR_50_mean"].iloc[-1]
        assert abs(atr_tail - atr_50_last) / atr_50_last < 0.15

    def test_atr_pctile_bounded(self, df_std):
        valid = df_std["ATR_Pctile"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_atr_higher_in_volatile_data(self, cfg):
        """Higher volatility data should produce higher ATR."""
        calm = _make_raw_ohlcv(n=300, seed=1)
        # Make a volatile version
        wild = calm.copy()
        wild["High"] = wild["Close"] * 1.03
        wild["Low"] = wild["Close"] * 0.97
        out_calm = add_indicators(calm, cfg)
        out_wild = add_indicators(wild, cfg)
        assert out_wild["ATR"].iloc[-1] > out_calm["ATR"].iloc[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Supertrend
# ─────────────────────────────────────────────────────────────────────────────

class TestSupertrend:

    def test_supertrend_columns_exist(self, df_std):
        assert "Supertrend" in df_std.columns
        assert "Super_Up" in df_std.columns

    def test_super_up_is_boolean(self, df_std):
        values = df_std["Super_Up"].dropna().unique()
        assert set(values).issubset({True, False, 1, 0, np.True_, np.False_})

    def test_super_up_true_in_uptrend(self, df_up):
        """In a strong uptrend, the last bar should have Super_Up = True."""
        assert df_up["Super_Up"].iloc[-1]

    def test_super_up_false_in_downtrend(self, df_down):
        """In a strong downtrend, the last bar should have Super_Up = False."""
        assert not df_down["Super_Up"].iloc[-1]

    def test_supertrend_no_nan_in_later_bars(self, df_std):
        """After warm-up, Supertrend should have no NaN values."""
        later = df_std["Supertrend"].iloc[50:]
        assert later.notna().all()

    def test_supertrend_flip_detection(self, cfg):
        """An up-then-down series should show a Supertrend flip."""
        n = 300
        np.random.seed(55)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        # First 150 bars up, last 150 bars down
        up = 100 * np.cumprod(1 + 0.005 + np.random.normal(0, 0.008, 150))
        dn = up[-1] * np.cumprod(1 - 0.005 + np.random.normal(0, 0.008, 150))
        close = np.concatenate([up, dn])
        df = pd.DataFrame({
            "Open": close * 0.999, "High": close * 1.004,
            "Low": close * 0.996, "Close": close,
            "Volume": np.full(n, 2_000_000.0),
        }, index=dates)
        out = add_indicators(df, cfg)
        # Should have both True and False in Super_Up
        su = out["Super_Up"].dropna()
        assert True in su.values and False in su.values


# ─────────────────────────────────────────────────────────────────────────────
# ADX
# ─────────────────────────────────────────────────────────────────────────────

class TestADX:

    def test_adx_column_exists(self, df_std):
        assert "ADX" in df_std.columns

    def test_adx_bounded(self, df_std):
        valid = df_std["ADX"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_adx_high_in_strong_trend(self, df_up):
        """Strong uptrend should produce elevated ADX."""
        assert df_up["ADX"].iloc[-1] > 15

    def test_adx_lower_in_flat_market(self, df_flat, df_up):
        """Flat market should produce lower ADX than trending market."""
        assert df_flat["ADX"].iloc[-1] < df_up["ADX"].iloc[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

class TestBollingerBands:

    def test_bb_columns_exist(self, df_std):
        for col in ("BB_Upper", "BB_Lower", "BB_Width", "BB_Squeeze"):
            assert col in df_std.columns

    def test_bb_upper_above_lower(self, df_std):
        valid = df_std[["BB_Upper", "BB_Lower"]].dropna()
        assert (valid["BB_Upper"] >= valid["BB_Lower"]).all()

    def test_bb_width_positive(self, df_std):
        valid = df_std["BB_Width"].dropna()
        assert (valid > 0).all()

    def test_bb_squeeze_is_boolean(self, df_std):
        values = df_std["BB_Squeeze"].dropna().unique()
        assert set(values).issubset({True, False, 1, 0, np.True_, np.False_})

    def test_bb_squeeze_triggered_on_narrow_bands(self, cfg):
        """Narrow bands (low vol) followed by normal bands should trigger squeeze."""
        n = 300
        np.random.seed(44)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        # Very calm period (narrow BB) for last 30 bars
        close = 100 * np.cumprod(1 + np.random.normal(0, 0.015, n))
        close[-30:] = close[-31] * np.cumprod(1 + np.random.normal(0, 0.002, 30))
        df = pd.DataFrame({
            "Open": close * 0.999, "High": close * 1.003,
            "Low": close * 0.997, "Close": close,
            "Volume": np.full(n, 1_500_000.0),
        }, index=dates)
        out = add_indicators(df, cfg)
        # At least some squeeze bars should exist in the calm period
        assert out["BB_Squeeze"].tail(25).any()


# ─────────────────────────────────────────────────────────────────────────────
# Volume Metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestVolumeMetrics:

    def test_vol_avg_20_column_exists(self, df_std):
        assert "Vol_Avg_20" in df_std.columns

    def test_vol_avg_20_near_actual_mean(self, df_std):
        """Vol_Avg_20 should approximate the rolling 20-bar volume mean."""
        last_20_vol = df_std["Volume"].tail(20).mean()
        vol_avg = df_std["Vol_Avg_20"].iloc[-1]
        assert abs(last_20_vol - vol_avg) / vol_avg < 0.15

    def test_turnover_avg_20_exists(self, df_std):
        assert "Turnover_Avg_20" in df_std.columns
        assert (df_std["Turnover_Avg_20"].dropna() > 0).all()

    def test_rvol_20_near_one_for_steady_volume(self, cfg):
        """Constant volume should produce RVol_20 ≈ 1.0."""
        n = 100
        np.random.seed(88)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        close = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))
        df = pd.DataFrame({
            "Open": close * 0.999, "High": close * 1.003,
            "Low": close * 0.997, "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        }, index=dates)
        out = add_indicators(df, cfg)
        median_rvol = out["RVol_20"].dropna().median()
        assert 0.8 < median_rvol < 1.2, f"RVol_20 median={median_rvol:.4f} should be ~1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Candle Helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestCandleHelpers:

    def test_up_day_dn_day_exist(self, df_std):
        assert "Up_Day" in df_std.columns
        assert "Dn_Day" in df_std.columns

    def test_up_day_is_binary(self, df_std):
        assert set(df_std["Up_Day"].unique()).issubset({0, 1})

    def test_dn_day_is_binary(self, df_std):
        assert set(df_std["Dn_Day"].unique()).issubset({0, 1})

    def test_mutual_exclusivity(self, df_std):
        """A bar cannot be both an up day and a down day."""
        both = (df_std["Up_Day"] == 1) & (df_std["Dn_Day"] == 1)
        assert not both.any()

    def test_uptrend_has_more_up_days(self, df_up):
        """Strong uptrend should have more up days than down days."""
        up_pct = df_up["Up_Day"].tail(100).mean()
        assert up_pct > 0.45


# ─────────────────────────────────────────────────────────────────────────────
# NaN Cleanup
# ─────────────────────────────────────────────────────────────────────────────

class TestDropNaN:

    def test_no_nan_in_core_columns(self, df_std):
        """After add_indicators(), DROPNA_COLS should contain no NaNs."""
        for col in DROPNA_COLS:
            assert col in df_std.columns, f"Missing column: {col}"
            assert df_std[col].isna().sum() == 0, f"NaN found in {col}"

    def test_output_not_empty(self, df_std):
        """Output should have retained most rows after NaN cleanup."""
        assert len(df_std) > 100

    def test_short_input_does_not_crash(self, cfg):
        """Even a very short OHLCV input should not crash, though output may be empty."""
        df = _make_raw_ohlcv(n=30, seed=77)
        out = add_indicators(df, cfg)
        # May be empty due to warm-up requirements — that's OK
        assert isinstance(out, pd.DataFrame)
