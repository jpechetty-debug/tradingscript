"""
tests/test_factors.py
=====================
Unit tests for core/factors.py factor functions.

Target: ≥ 80% branch coverage of all factor_* functions and
        compute_factors / true_volume_profile.

Run:
    pytest tests/test_factors.py -v
    pytest tests/test_factors.py --cov=core/factors --cov-report=term-missing
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

from core.factors import (
    true_volume_profile,
    factor_trend,
    factor_momentum,
    factor_volume,
    factor_volatility,
    factor_relative_strength,
    factor_breakout,
    factor_quality,
    compute_factors,
    FactorScores,
)


# ── Shared fixtures ────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 120, base: float = 100.0, trend: float = 0.002) -> pd.DataFrame:
    """
    Synthetic OHLCV with a mild upward drift and realistic column set needed
    by the factor functions.
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = base * np.cumprod(1 + trend + np.random.normal(0, 0.01, n))
    high  = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low   = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.004, n))
    vol   = np.random.randint(800_000, 2_000_000, n).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    # Pre-compute indicator columns that factor functions consume
    df["EMA_20"]  = close
    df["EMA_50"]  = close * 0.99
    df["EMA_200"] = close * 0.95
    df["RSI"]     = 58.0
    df["ADX"]     = 28.0
    df["ATR"]     = close * 0.012
    df["ATR_50_mean"] = close * 0.014
    df["ATR_Pctile"]  = 40.0
    df["MACD_Hist"]   = 0.5
    df["BB_Width"]    = 0.04
    df["BB_Squeeze"]  = False
    df["Vol_Avg_20"]  = vol.mean()
    df["Turnover_Avg_20"] = close.mean() * vol.mean()
    df["Up_Day"]  = (close > open_).astype(int)
    df["Dn_Day"]  = (close < open_).astype(int)
    df["Super_Up"] = True
    df["Supertrend"] = close * 0.97
    df["StochRSI_K"] = 55.0
    df["StochRSI_D"] = 52.0
    return df


def _make_row(df: pd.DataFrame) -> pd.Series:
    return df.iloc[-1]


def _make_bench(df: pd.DataFrame) -> pd.Series:
    """Benchmark that slightly underperforms the stock (positive RS)."""
    return pd.Series(
        df["Close"].values * np.linspace(1.0, 0.97, len(df)),
        index=df.index,
    )


# ─────────────────────────────────────────────────────────────────────────────
# true_volume_profile
# ─────────────────────────────────────────────────────────────────────────────

class TestTrueVolumeProfile:

    def test_returns_three_floats(self):
        df  = _make_ohlcv()
        poc, val, vah = true_volume_profile(df)
        assert isinstance(poc, float)
        assert isinstance(val, float)
        assert isinstance(vah, float)

    def test_ordering_val_le_poc_le_vah(self):
        df = _make_ohlcv()
        poc, val, vah = true_volume_profile(df)
        assert val <= poc <= vah, f"Ordering violated: val={val} poc={poc} vah={vah}"

    def test_short_df_fallback(self):
        """Fewer than 5 rows triggers the close-based fallback."""
        df = _make_ohlcv(n=3)
        poc, val, vah = true_volume_profile(df, lookback=30)
        c = float(df["Close"].iloc[-1])
        assert abs(poc - c) < 1.0

    def test_flat_price_no_crash(self):
        """hi_p == lo_p edge case (flat price bar)."""
        df = _make_ohlcv(n=10)
        df["High"] = df["Low"] = df["Close"] = 100.0
        poc, val, vah = true_volume_profile(df, lookback=10)
        assert poc == pytest.approx(100.0, abs=1.0)

    def test_custom_lookback_bins(self):
        df = _make_ohlcv(n=80)
        poc, val, vah = true_volume_profile(df, lookback=20, bins=50)
        assert val <= poc <= vah


# ─────────────────────────────────────────────────────────────────────────────
# factor_trend
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorTrend:

    def _row(self, close, ema20, ema50, ema200, super_up=True) -> pd.Series:
        return pd.Series({
            "EMA_20": ema20, "EMA_50": ema50, "EMA_200": ema200,
            "Super_Up": super_up,
        })

    def test_perfect_long_score_is_one(self):
        row = self._row(110, 105, 100, 90, super_up=True)
        score = factor_trend(row, close=110, direction="LONG")
        assert score == pytest.approx(1.0)

    def test_perfect_short_score_is_one(self):
        row = self._row(90, 95, 100, 110, super_up=False)
        score = factor_trend(row, close=90, direction="SHORT")
        assert score == pytest.approx(1.0)

    def test_zero_long_below_all_emas(self):
        row = self._row(80, 90, 100, 110, super_up=False)
        score = factor_trend(row, close=80, direction="LONG")
        assert score == pytest.approx(0.0)

    def test_zero_short_above_all_emas(self):
        row = self._row(120, 110, 100, 90, super_up=True)
        score = factor_trend(row, close=120, direction="SHORT")
        assert score == pytest.approx(0.0)

    def test_mtf_alignment_bonus_applied(self):
        row = self._row(110, 105, 100, 90, super_up=True)
        with_mtf    = factor_trend(row, close=110, direction="LONG", mtf_60m_trend_aligned=True)
        without_mtf = factor_trend(row, close=110, direction="LONG", mtf_60m_trend_aligned=False)
        assert with_mtf >= without_mtf

    def test_output_clipped_to_unit_interval(self):
        row = self._row(110, 105, 100, 90, super_up=True)
        score = factor_trend(row, close=110, direction="LONG", mtf_60m_trend_aligned=True)
        assert 0.0 <= score <= 1.0

    def test_ema_stack_bonus_long(self):
        """ema20 > ema50 > ema200 yields extra +0.05 bonus."""
        row = self._row(110, 105, 100, 90, super_up=True)
        score = factor_trend(row, close=110, direction="LONG")
        # Full score without stack bonus would be 0.90; with bonus capped to 1.0
        assert score == pytest.approx(1.0)

    def test_ema_stack_bonus_short(self):
        row = self._row(85, 90, 95, 100, super_up=False)
        score = factor_trend(row, close=85, direction="SHORT")
        assert score == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# factor_momentum
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorMomentum:

    def _df_and_row(self, rsi=58, adx=28, macd_hist=0.5, sk=55, direction="LONG"):
        df  = _make_ohlcv()
        df["RSI"]       = rsi
        df["ADX"]       = adx
        df["MACD_Hist"] = macd_hist
        df["StochRSI_K"] = sk
        df["Up_Day"]    = 1 if direction == "LONG" else 0
        df["Dn_Day"]    = 0 if direction == "LONG" else 1
        return df, _make_row(df)

    def test_ideal_long_conditions(self):
        df, row = self._df_and_row(rsi=60, adx=30, macd_hist=0.8, sk=55, direction="LONG")
        score = factor_momentum(row=row, daily_df=df, direction="LONG")
        assert score > 0.7

    def test_ideal_short_conditions(self):
        df, row = self._df_and_row(rsi=38, adx=30, macd_hist=-0.8, sk=45, direction="SHORT")
        score = factor_momentum(row=row, daily_df=df, direction="SHORT")
        assert score > 0.7

    def test_rsi_overbought_penalty_long(self):
        df_good, row_good = self._df_and_row(rsi=60)
        df_ob, row_ob     = self._df_and_row(rsi=82)
        assert factor_momentum(row_good, df_good, "LONG") > factor_momentum(row_ob, df_ob, "LONG")

    def test_macd_negative_reduces_long_score(self):
        df_pos, row_pos = self._df_and_row(macd_hist=0.5)
        df_neg, row_neg = self._df_and_row(macd_hist=-0.5)
        assert factor_momentum(row_pos, df_pos, "LONG") > factor_momentum(row_neg, df_neg, "LONG")

    def test_stochrsi_extreme_penalty_long(self):
        df_ok, row_ok   = self._df_and_row(sk=55)
        df_ext, row_ext = self._df_and_row(sk=95)
        assert factor_momentum(row_ok, df_ok, "LONG") > factor_momentum(row_ext, df_ext, "LONG")

    def test_streak_increases_score(self):
        df = _make_ohlcv()
        df["RSI"] = 58
        df["ADX"] = 28
        df["MACD_Hist"] = 0.5
        df["StochRSI_K"] = 55
        df["Up_Day"] = 0
        row_no_streak = _make_row(df)
        score_no = factor_momentum(row_no_streak, df, "LONG")

        df["Up_Day"] = 1
        row_streak = _make_row(df)
        score_yes = factor_momentum(row_streak, df, "LONG")
        assert score_yes >= score_no

    def test_output_clipped(self):
        df, row = self._df_and_row()
        score = factor_momentum(row=row, daily_df=df, direction="LONG")
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# factor_volume
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorVolume:

    def _df_row(self, rvol_mult=2.0, turnover=80_000_000):
        df  = _make_ohlcv()
        avg = float(df["Volume"].mean())
        row = _make_row(df).copy()
        row["Vol_Avg_20"]      = avg
        row["Turnover_Avg_20"] = turnover
        return df, row, avg, int(avg * rvol_mult)

    def test_high_rvol_scores_high(self):
        df, row, avg, vol_today = self._df_row(rvol_mult=2.5)
        score = factor_volume(
            row=row, daily_df=df, direction="LONG",
            close=float(df["Close"].iloc[-1]),
            intraday_vol_today=vol_today,
        )
        assert score > 0.5

    def test_below_average_volume_scores_low(self):
        df, row, avg, _ = self._df_row()
        score = factor_volume(
            row=row, daily_df=df, direction="LONG",
            close=float(df["Close"].iloc[-1]),
            intraday_vol_today=int(avg * 0.5),  # half of average
        )
        assert score < 0.45

    def test_no_intraday_vol_uses_avg(self):
        df, row, avg, _ = self._df_row(rvol_mult=1.0)
        score = factor_volume(
            row=row, daily_df=df, direction="LONG",
            close=float(df["Close"].iloc[-1]),
            intraday_vol_today=None,
        )
        assert 0.0 <= score <= 1.0

    def test_zero_vol_avg_no_crash(self):
        df, row, _, _ = self._df_row()
        row["Vol_Avg_20"] = 0
        score = factor_volume(
            row=row, daily_df=df, direction="LONG",
            close=float(df["Close"].iloc[-1]),
        )
        assert 0.0 <= score <= 1.0

    def test_output_clipped(self):
        df, row, avg, vol_today = self._df_row(rvol_mult=10.0, turnover=500_000_000)
        score = factor_volume(
            row=row, daily_df=df, direction="LONG",
            close=float(df["Close"].iloc[-1]),
            intraday_vol_today=vol_today,
        )
        assert score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# factor_volatility
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorVolatility:

    def _row(self, atr, atr50m, atr_pct=40, bbs=False) -> pd.Series:
        return pd.Series({
            "ATR": atr, "ATR_50_mean": atr50m,
            "ATR_Pctile": atr_pct, "BB_Squeeze": bbs,
        })

    def test_contracting_atr_scores_high(self):
        row   = self._row(atr=1.0, atr50m=1.5)   # atr < 0.85 * atr50m → contracting
        score = factor_volatility(row, vol_contract_ratio=0.85)
        assert score > 0.6

    def test_expanding_atr_scores_low(self):
        row   = self._row(atr=2.5, atr50m=1.5)   # atr >> atr50m → expanding
        score = factor_volatility(row, vol_contract_ratio=0.85)
        assert score < 0.5

    def test_bb_squeeze_bonus_applied(self):
        row_no  = self._row(atr=1.0, atr50m=1.5, bbs=False)
        row_yes = self._row(atr=1.0, atr50m=1.5, bbs=True)
        assert factor_volatility(row_yes) > factor_volatility(row_no)

    def test_bb_squeeze_capped_at_one(self):
        row = self._row(atr=0.5, atr50m=2.0, atr_pct=5, bbs=True)
        assert factor_volatility(row) <= 1.0

    def test_zero_atr50m_no_crash(self):
        row   = self._row(atr=1.0, atr50m=0.0)
        score = factor_volatility(row)
        assert 0.0 <= score <= 1.0

    def test_output_clipped(self):
        for atr, atr50m in [(0.1, 5.0), (5.0, 0.1)]:
            row   = self._row(atr=atr, atr50m=atr50m)
            score = factor_volatility(row)
            assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# factor_relative_strength
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorRelativeStrength:

    def _call(self, sec_rank=1, n_sectors=10, direction="LONG", rs_outperform=True):
        df    = _make_ohlcv()
        bench = _make_bench(df)
        if not rs_outperform:
            # Make stock underperform
            bench = pd.Series(df["Close"].values * 1.05, index=df.index)
        return factor_relative_strength(
            ticker="TEST.NS",
            daily_df=df,
            bench=bench,
            sector="Technology",
            sector_ranks={"Technology": sec_rank},
            direction=direction,
            n_sectors=n_sectors,
        )

    def test_best_sector_long_scores_high(self):
        score = self._call(sec_rank=1, n_sectors=10, direction="LONG")
        assert score > 0.5

    def test_worst_sector_long_scores_low(self):
        score = self._call(sec_rank=10, n_sectors=10, direction="LONG")
        assert score < 0.5

    def test_best_sector_short_scores_high(self):
        score = self._call(sec_rank=10, n_sectors=10, direction="SHORT")
        assert score > 0.3

    def test_unknown_sector_defaults_to_worst_rank(self):
        df    = _make_ohlcv()
        bench = _make_bench(df)
        score = factor_relative_strength(
            ticker="UNKNOWN.NS", daily_df=df, bench=bench,
            sector="Obscure", sector_ranks={}, direction="LONG", n_sectors=10,
        )
        assert 0.0 <= score <= 1.0

    def test_output_clipped(self):
        assert 0.0 <= self._call() <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# factor_breakout
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorBreakout:

    def _call(self, close_pct_of_52w=0.99, bw=0.03, bwavg=0.06, direction="LONG"):
        df   = _make_ohlcv()
        h52  = float(df["High"].max())
        close = h52 * close_pct_of_52w
        df["BB_Width"] = bwavg  # fill all rows so rolling mean works
        row = _make_row(df).copy()
        row["BB_Width"] = bw
        return factor_breakout(
            row=row, daily_df=df, close=close, direction=direction,
        )

    def test_near_52w_high_long_scores_high(self):
        score = self._call(close_pct_of_52w=0.99)
        assert score > 0.5

    def test_far_from_52w_high_long_scores_low(self):
        score = self._call(close_pct_of_52w=0.85)
        assert score < 0.5

    def test_short_proximity_gives_fixed_contribution(self):
        score = self._call(direction="SHORT", close_pct_of_52w=0.99)
        # SHORT: ds = 0.5 always; narrow BB adds 0.35 → ~0.85 max
        assert 0.0 <= score <= 1.0

    def test_narrow_bb_boosts_score(self):
        # Wide BB
        score_wide   = self._call(bw=0.08, bwavg=0.04)   # bw > bwavg * 0.85 → not narrow
        # Narrow BB
        score_narrow = self._call(bw=0.02, bwavg=0.04)   # bw < bwavg * 0.85 → narrow
        assert score_narrow >= score_wide

    def test_output_clipped(self):
        assert 0.0 <= self._call() <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# factor_quality
# ─────────────────────────────────────────────────────────────────────────────

class TestFactorQuality:

    def test_strong_uptrend_scores_high(self):
        df = _make_ohlcv(n=120, trend=0.004)
        df["Up_Day"]      = 1
        df["ATR"]         = df["Close"] * 0.012
        df["ATR_50_mean"] = df["Close"] * 0.010   # atr > atr50m → expansion
        row = _make_row(df)
        score = factor_quality(daily_df=df, row=row, close=float(df["Close"].iloc[-1]))
        assert score > 0.55

    def test_short_df_uses_fallback_mom(self):
        df = _make_ohlcv(n=30)
        df["Up_Day"]      = 1
        df["ATR"]         = 1.2
        df["ATR_50_mean"] = 1.4
        row = _make_row(df)
        score = factor_quality(daily_df=df, row=row, close=float(df["Close"].iloc[-1]))
        # mom_score falls back to 0.5 — just check it's valid
        assert 0.0 <= score <= 1.0

    def test_missing_up_day_column_no_crash(self):
        df = _make_ohlcv(n=80)
        df = df.drop(columns=["Up_Day"])
        df["ATR"] = 1.0
        df["ATR_50_mean"] = 1.0
        row = _make_row(df)
        score = factor_quality(daily_df=df, row=row, close=float(df["Close"].iloc[-1]))
        assert 0.0 <= score <= 1.0

    def test_output_clipped(self):
        df = _make_ohlcv(n=120)
        df["Up_Day"] = 1
        df["ATR"] = 0.0
        df["ATR_50_mean"] = 0.0
        row = _make_row(df)
        score = factor_quality(daily_df=df, row=row, close=float(df["Close"].iloc[-1]))
        assert 0.0 <= score <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_factors — integration / smoke
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFactors:

    def _call(self, direction="LONG", weights=None):
        df    = _make_ohlcv(n=120)
        bench = _make_bench(df)
        row   = _make_row(df)
        close = float(df["Close"].iloc[-1])
        return compute_factors(
            ticker="RELIANCE.NS",
            daily_df=df,
            bench=bench,
            sector="Energy",
            sector_ranks={"Energy": 2},
            direction=direction,
            close=close,
            row=row,
            n_sectors=10,
            weights=weights,
        )

    def test_returns_factor_scores_dataclass(self):
        result = self._call()
        assert isinstance(result, FactorScores)

    def test_all_sub_scores_in_unit_interval(self):
        result = self._call()
        for field in ("trend", "momentum", "volume", "volatility", "rs", "breakout", "quality"):
            val = getattr(result, field)
            assert 0.0 <= val <= 1.0, f"{field}={val} out of [0,1]"

    def test_composite_in_unit_interval(self):
        result = self._call()
        assert 0.0 <= result.composite <= 1.0

    def test_ic_weights_sum_to_one(self):
        result = self._call()
        total = sum(result.ic_weights.values())
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_as_dict_has_all_factors(self):
        result = self._call()
        d = result.as_dict()
        for key in ("trend", "momentum", "volume", "volatility", "rs", "breakout", "quality", "composite"):
            assert key in d

    def test_short_direction_runs_cleanly(self):
        result = self._call(direction="SHORT")
        assert isinstance(result, FactorScores)
        assert 0.0 <= result.composite <= 1.0

    def test_custom_weights_applied(self):
        custom = {"trend": 0.5, "momentum": 0.5, "volume": 0.0,
                  "volatility": 0.0, "rs": 0.0, "breakout": 0.0, "quality": 0.0}
        result = self._call(weights=custom)
        assert 0.0 <= result.composite <= 1.0

    def test_with_intraday_and_mtf(self):
        df    = _make_ohlcv(n=120)
        bench = _make_bench(df)
        row   = _make_row(df)
        close = float(df["Close"].iloc[-1])
        result = compute_factors(
            ticker="TEST.NS",
            daily_df=df,
            bench=bench,
            sector="IT",
            sector_ranks={"IT": 1},
            direction="LONG",
            close=close,
            row=row,
            n_sectors=10,
            intraday={"vol_today": 2_000_000},
            mtf_60m={"trend_aligned": True},
        )
        assert isinstance(result, FactorScores)


# ─────────────────────────────────────────────────────────────────────────────
# add_indicators (from core.indicators)
# ─────────────────────────────────────────────────────────────────────────────

class TestAddIndicators:
    """Smoke tests that indicators.py produces the expected columns."""

    def _config(self):
        cfg = MagicMock()
        cfg.SUPER_PERIOD = 10
        cfg.SUPER_MULT   = 3.0
        cfg.ADX_PERIOD   = 14
        return cfg

    def _make_raw_ohlcv(self, n=300) -> pd.DataFrame:
        np.random.seed(7)
        dates = pd.date_range("2023-01-01", periods=n, freq="B")
        c = 500 * np.cumprod(1 + np.random.normal(0.001, 0.012, n))
        h = c * (1 + np.abs(np.random.normal(0, 0.005, n)))
        low = c * (1 - np.abs(np.random.normal(0, 0.005, n)))
        o = c * (1 + np.random.normal(0, 0.004, n))
        v = np.random.randint(1_000_000, 3_000_000, n).astype(float)
        return pd.DataFrame(
            {"Open": o, "High": h, "Low": low, "Close": c, "Volume": v},
            index=dates,
        )

    def test_required_columns_present(self):
        from core.indicators import add_indicators
        df  = self._make_raw_ohlcv()
        out = add_indicators(df, self._config())
        for col in ("EMA_20", "EMA_50", "EMA_200", "RSI", "StochRSI_K", "StochRSI_D",
                    "ATR", "ATR_50_mean", "ATR_Pctile", "ADX", "MACD_Hist",
                    "BB_Upper", "BB_Lower", "BB_Width", "BB_Squeeze",
                    "Vol_Avg_20", "Turnover_Avg_20", "RVol_20",
                    "Up_Day", "Dn_Day", "Supertrend", "Super_Up"):
            assert col in out.columns, f"Missing column: {col}"

    def test_rvol_20_is_ratio_not_volatility(self):
        """RVol_20 should be ~1.0 for flat volume, not a tiny std-dev."""
        from core.indicators import add_indicators
        df  = self._make_raw_ohlcv()
        out = add_indicators(df, self._config())
        median_rvol = out["RVol_20"].median()
        # For random volume around a constant mean the ratio hovers near 1.0
        assert 0.5 < median_rvol < 2.5, (
            f"RVol_20 median={median_rvol:.4f} looks wrong — "
            "check it's vol/rolling_mean not pct_change().std()"
        )

    def test_no_nan_in_core_columns(self):
        from core.indicators import add_indicators
        df  = self._make_raw_ohlcv()
        out = add_indicators(df, self._config())
        core = ["EMA_20", "EMA_50", "EMA_200", "RSI", "ATR", "ADX", "Vol_Avg_20", "MACD_Hist"]
        assert out[core].isna().sum().sum() == 0

    def test_stochrsi_in_range(self):
        from core.indicators import add_indicators
        df = self._make_raw_ohlcv()
        out = add_indicators(df, self._config())
        valid = out["StochRSI_K"].dropna()
        assert valid.between(0, 100).all()


# ─────────────────────────────────────────────────────────────────────────────
# data_provider exception handling
# ─────────────────────────────────────────────────────────────────────────────

class TestDataProviderExceptions:

    def test_fyers_to_df_raises_on_bad_status(self):
        from core.data_provider import _fyers_to_df
        with pytest.raises(ValueError, match="status="):
            _fyers_to_df({"s": "error", "message": "auth failed"})

    def test_fyers_to_df_raises_on_missing_candles(self):
        from core.data_provider import _fyers_to_df
        with pytest.raises(ValueError, match="candles"):
            _fyers_to_df({"s": "ok"})

    def test_fyers_to_df_ok_path(self):
        from core.data_provider import _fyers_to_df
        candles = [[1_700_000_000 + i * 86400, 100+i, 105+i, 98+i, 102+i, 1_000_000]
                   for i in range(5)]
        df = _fyers_to_df({"s": "ok", "candles": candles})
        assert len(df) == 5
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_fetch_single_ticker_returns_none_on_no_client(self):
        from core.data_provider import fetch_single_ticker, FyersSessionManager
        FyersSessionManager._instance = None
        cfg = MagicMock()
        cfg.FYERS_CLIENT_ID    = ""
        cfg.FYERS_ACCESS_TOKEN = ""
        ticker, df = fetch_single_ticker("TEST.NS", "NSE:TEST-EQ", cfg)
        assert ticker == "TEST.NS"
        assert df is None

def test_nan_atr_rejected_at_gate():
    from core.scorer import passes_data_quality
    row = pd.Series({"ATR": float("nan"), "ATR_50_mean": 1.0, "ATR_Pctile": 40.0,
                     "EMA_20": 100.0, "EMA_50": 98.0, "EMA_200": 90.0,
                     "RSI": 55.0, "ADX": 25.0, "MACD_Hist": 0.1, "Vol_Avg_20": 1e6})
    ok, msg = passes_data_quality(row, "TEST")
    assert not ok
    assert "ATR" in msg

def test_pd_na_rejected_at_gate():
    from core.scorer import passes_data_quality
    row = pd.Series({"ATR": 1.0, "ATR_50_mean": pd.NA, "ATR_Pctile": 40.0,
                     "EMA_20": 100.0, "EMA_50": 98.0, "EMA_200": 90.0,
                     "RSI": 55.0, "ADX": 25.0, "MACD_Hist": 0.1, "Vol_Avg_20": 1e6})
    ok, msg = passes_data_quality(row, "TEST")
    assert not ok
