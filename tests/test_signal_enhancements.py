"""
tests/test_signal_enhancements.py
=================================
Unit tests for Dual-Engine Signal Enhancements (Intraday vs Swing, Short Parity,
VCP Volume, RSI Pullback, and Regime-Conditional Weights).
"""

from __future__ import annotations

import sys
import os
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Stub heavy/optional dependencies
def _stub_modules() -> None:
    stubs: dict[str, types.ModuleType] = {
        "fyers_apiv3": types.ModuleType("fyers_apiv3"),
        "fyers_apiv3.fyersModel": types.ModuleType("fyers_apiv3.fyersModel"),
    }
    stubs["fyers_apiv3.fyersModel"].FyersModel = MagicMock  # type: ignore[attr-defined]
    stubs["fyers_apiv3"].fyersModel = stubs["fyers_apiv3.fyersModel"]  # type: ignore[attr-defined]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

_stub_modules()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import SystemConfig
from core.factors import (
    factor_breakout,
    factor_quality,
    factor_momentum,
    factor_volume,
    get_regime_factor_weights,
)
from core.portfolio import compute_targets
from core.scorer import score_ticker, TickerResult
from core.regime import MarketRegime, MarketRegimeType


def _make_df(n: int = 100, base: float = 100.0, trend: float = 0.0) -> pd.DataFrame:
    """Helper to generate synthetic OHLCV data."""
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = base * np.cumprod(1 + trend + np.zeros(n))
    high = close * 1.01
    low = close * 0.99
    open_ = close * 1.00
    vol = np.full(n, 1_000_000.0)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    df["EMA_20"] = close
    df["EMA_50"] = close * 0.98
    df["EMA_200"] = close * 0.92
    df["RSI"] = 55.0
    df["ADX"] = 28.0
    df["ATR"] = 2.0
    df["ATR_50_mean"] = 2.0
    df["ATR_Pctile"] = 50.0
    df["MACD_Hist"] = 0.5
    df["BB_Width"] = 0.05
    df["BB_Squeeze"] = False
    df["Vol_Avg_20"] = 1_000_000.0
    df["Turnover_Avg_20"] = 100_000_000.0
    df["Up_Day"] = 1
    df["Dn_Day"] = 0
    df["Super_Up"] = True
    df["Supertrend"] = close * 0.95
    df["StochRSI_K"] = 50.0
    df["StochRSI_D"] = 50.0
    return df


class TestShortFactorParity:
    """Validate short signals achieve parity with long signals."""

    def test_short_factor_parity_breakout(self):
        df = _make_df(100, base=100.0)
        # Price is breaking down right at the 52w and 20d low
        df.loc[df.index[-1], "Close"] = 90.0
        df.loc[df.index[-1], "Low"] = 90.0
        row = df.iloc[-1]

        score_short = factor_breakout(
            row=row,
            daily_df=df,
            close=90.0,
            direction="SHORT",
            near_52w_max_dist_pct=8.0,
        )
        assert score_short >= 0.65, f"Expected high breakdown score for short, got {score_short}"

    def test_short_factor_parity_quality(self):
        # Stock declining over 64 bars (negative 63d momentum, high down days)
        df = _make_df(100, base=100.0, trend=-0.005)
        df["Dn_Day"] = 1
        df["Up_Day"] = 0
        df["ATR"] = 2.0
        df["ATR_50_mean"] = 2.0
        row = df.iloc[-1]
        close = float(row["Close"])

        score_short = factor_quality(df, row, close, direction="SHORT")
        score_long = factor_quality(df, row, close, direction="LONG")

        assert score_short > 0.60, f"Expected high quality for strong downtrend short, got {score_short}"
        assert score_short > score_long, "Short quality should be significantly higher than Long for downtrend"


class TestRSIPullbackZone:
    """Validate RSI 42-48 bull pullback zone scores properly."""

    def test_rsi_bull_pullback(self):
        df = _make_df(100, base=100.0)
        df["MACD_Hist"] = 0.5
        df.loc[df.index[-2], "MACD_Hist"] = 0.4  # accelerating
        row = df.iloc[-1].copy()
        row["RSI"] = 44.0  # Bull pullback dip zone (42 <= RSI < 48)

        score_pullback = factor_momentum(row, df, direction="LONG")

        # In bull pullback zone, rsi_score is 0.85 -> momentum score should be strong (> 0.55)
        assert score_pullback >= 0.55, f"Expected high momentum score in pullback dip, got {score_pullback}"

        # Compare with dead zone (e.g. RSI = 40.5 which falls into 0.0 score)
        row_dead = row.copy()
        row_dead["RSI"] = 40.5
        score_dead = factor_momentum(row_dead, df, direction="LONG")
        assert score_pullback > score_dead, "Pullback zone (42-48) should score higher than dead zone (<42)"


class TestVCPVolumeRecognition:
    """Validate low volume during tight volatility contraction (VCP) is rewarded."""

    def test_vcp_volume_recognition(self):
        df = _make_df(100, base=100.0)
        row = df.iloc[-1].copy()
        row["ATR"] = 1.0
        row["ATR_50_mean"] = 2.0  # ATR is 50% of 50d mean -> tight contraction
        row["Vol_Avg_20"] = 1_000_000.0

        # RVOL = 0.50 (low volume coiling)
        intraday_vol = 500_000

        score = factor_volume(
            row=row,
            daily_df=df,
            direction="LONG",
            close=100.0,
            intraday_vol_today=intraday_vol,
        )
        assert score >= 0.40, f"Expected VCP consolidation score to be >= 0.40, got {score}"


class TestRegimeConditionalWeights:
    """Validate dynamic factor weights per market regime."""

    def test_regime_weights_sum_and_priorities(self):
        regimes = ["TREND_UP", "RANGE", "TREND_DOWN", "EXPANSION", "PANIC"]
        for r in regimes:
            w = get_regime_factor_weights(r)
            assert isinstance(w, dict)
            assert len(w) == 7
            total = sum(w.values())
            assert abs(total - 1.0) < 1e-4, f"Weights for {r} must sum to 1.0, got {total}"

        # TREND_UP prioritizes trend, momentum, rs
        w_up = get_regime_factor_weights("TREND_UP")
        assert w_up["trend"] >= 0.20
        assert w_up["momentum"] >= 0.20

        # RANGE prioritizes volatility and volume
        w_range = get_regime_factor_weights("RANGE")
        assert w_range["volatility"] >= 0.20
        assert w_range["volume"] >= 0.20


class TestHorizonATRTargets:
    """Validate horizon-specific ATR targets for Intraday vs Swing."""

    def test_intraday_targets(self):
        config = SystemConfig()
        targets = compute_targets(
            direction="LONG",
            close=100.0,
            atr=2.0,
            config=config,
            trade_horizon="INTRADAY",
        )
        # INTRADAY_STOP_ATR_MULT = 0.50 -> stop = 100 - 1.0 = 99.0
        # INTRADAY_TARGET1_ATR_MULT = 1.20 -> t1 = 100 + 2.4 = 102.4
        # INTRADAY_TARGET2_ATR_MULT = 1.80 -> t2 = 100 + 3.6 = 103.6
        assert targets.stop == 99.0
        assert targets.t1 == 102.4
        assert targets.t2 == 103.6
        assert targets.rr == 2.4  # (102.4 - 100) / (100 - 99) = 2.4

    def test_swing_targets(self):
        config = SystemConfig()
        targets = compute_targets(
            direction="LONG",
            close=100.0,
            atr=2.0,
            config=config,
            trade_horizon="SWING",
        )
        # STOP_ATR_MULT = 1.50 -> stop = 100 - 3.0 = 97.0
        # TARGET1_ATR_MULT = 3.80 -> t1 = 100 + 7.6 = 107.6
        # TARGET2_ATR_MULT = 6.00 -> t2 = 100 + 12.0 = 112.0
        assert targets.stop == 97.0
        assert targets.t1 == 107.6
        assert targets.t2 == 112.0
        assert targets.rr == round((107.6 - 100.0) / (100.0 - 97.0), 2)  # 2.53

    def test_short_targets_horizon(self):
        config = SystemConfig()
        targets = compute_targets(
            direction="SHORT",
            close=100.0,
            atr=2.0,
            config=config,
            trade_horizon="INTRADAY",
        )
        assert targets.stop == 101.0
        assert targets.t1 == 97.6
        assert targets.t2 == 96.4
        assert targets.rr == 2.4


class TestScorerIntegration:
    """Validate that score_ticker populates trade_horizon and action."""

    def test_scorer_outputs_action_and_horizon(self):
        config = SystemConfig()
        config.MIN_PROB_WIN = 0.30
        config.MIN_EXPECTANCY_R = 0.0

        df = _make_df(120, base=100.0, trend=0.002)
        bench = df["Close"] * 0.98

        regime = MarketRegime(
            regime=MarketRegimeType.TREND_UP,
            breadth=0.75,
            adx_median=30.0,
            atr_ratio=0.8,
            confidence=0.85,
            confirmed=True,
        )

        res = score_ticker(
            ticker="RELIANCE.NS",
            daily_df=df,
            bench=bench,
            sector_ranks={"Energy": 1},
            sector_rs={"Energy": 2.0},
            session="OPENING_RANGE",
            regime=regime,
            config=config,
            force_score=True,
        )

        assert res is not None
        assert res.action in ("BUY", "SELL")
        assert res.trade_horizon == "INTRADAY"  # OPENING_RANGE maps to INTRADAY

    def test_short_defaults_to_intraday_targets(self):
        config = SystemConfig()
        config.MIN_PROB_WIN = 0.30
        config.MIDDAY_BREAKOUT_MIN_PROB = 0.30
        config.MIN_EXPECTANCY_R = 0.0
        config.SHORT_IS_INTRADAY_ONLY = True

        df = _make_df(120, base=100.0, trend=-0.002)
        df["EMA_20"] = df["Close"] * 1.02
        df["EMA_50"] = df["Close"] * 1.04
        df["EMA_200"] = df["Close"] * 1.10
        df["Super_Up"] = False
        df["Supertrend"] = df["Close"] * 1.05
        df["RSI"] = 38.0
        df["ATR"] = 2.0
        df["ATR_50_mean"] = 2.0
        df["Dn_Day"] = 1
        df["Up_Day"] = 0
        df["MACD_Hist"] = -0.5
        bench = df["Close"] * 1.05

        regime = MarketRegime(
            regime=MarketRegimeType.TREND_DOWN,
            breadth=0.35,
            adx_median=25.0,
            atr_ratio=1.0,
            confidence=0.85,
            confirmed=True,
        )

        res = score_ticker(
            ticker="MUTHOOTFIN.NS",
            daily_df=df,
            bench=bench,
            sector_ranks={"Financial Services": 1},
            sector_rs={"Financial Services": -2.0},
            session="MIDDAY_CHOP",  # Even in midday, short is strictly INTRADAY
            regime=regime,
            config=config,
            force_score=False,
        )

        assert res is not None
        assert res.action == "SELL"
        assert res.trade_horizon == "INTRADAY"
        assert res.rr_t1 == 2.4
        close = float(df["Close"].iloc[-1])
        # 0.50x ATR stop, 1.20x ATR target 1
        assert res.stop == round(close + 0.50 * 2.0, 2)
        assert res.t1 == round(close - 1.20 * 2.0, 2)
        assert res.t2 == round(close - 1.80 * 2.0, 2)

