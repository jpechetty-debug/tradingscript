"""
tests/test_timing_enhancements.py
=================================
Unit tests for Temporal Optimization and Phase-Gated Intraday Timing.
"""

from __future__ import annotations

import sys
import os
import types
from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

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

from core.config import SystemConfig, MarketPhase, IST
from core.regime import MarketRegime, MarketRegimeType
from core.scorer import score_ticker


def _make_df(n: int = 120, base: float = 100.0, trend: float = 0.002) -> pd.DataFrame:
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
    df["EMA_20"] = close * 0.99
    df["EMA_50"] = close * 0.97
    df["EMA_200"] = close * 0.90
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


class TestMarketPhaseTransitions:
    """Validate IST market phases and time calculations."""

    def test_market_phase_sequence(self):
        config = SystemConfig()

        t_pre = datetime(2026, 9, 10, 8, 45, tzinfo=IST)
        assert config.get_market_phase(t_pre) == MarketPhase.PRE_MARKET

        t_noise = datetime(2026, 9, 10, 9, 20, tzinfo=IST)
        assert config.get_market_phase(t_noise) == MarketPhase.OPENING_NOISE

        t_morning = datetime(2026, 9, 10, 9, 45, tzinfo=IST)
        assert config.get_market_phase(t_morning) == MarketPhase.PRIME_MORNING

        t_midday = datetime(2026, 9, 10, 11, 30, tzinfo=IST)
        assert config.get_market_phase(t_midday) == MarketPhase.MIDDAY_CHOP

        t_afternoon = datetime(2026, 9, 10, 13, 45, tzinfo=IST)
        assert config.get_market_phase(t_afternoon) == MarketPhase.AFTERNOON_EXPANSION

        t_freeze = datetime(2026, 9, 10, 14, 45, tzinfo=IST)
        assert config.get_market_phase(t_freeze) == MarketPhase.INTRADAY_FREEZE

        t_post = datetime(2026, 9, 10, 15, 45, tzinfo=IST)
        assert config.get_market_phase(t_post) == MarketPhase.POST_MARKET

    def test_minutes_to_squareoff(self):
        config = SystemConfig()
        t1 = datetime(2026, 9, 10, 14, 15, tzinfo=IST)
        assert config.minutes_to_squareoff(t1) == 60

        t2 = datetime(2026, 9, 10, 14, 45, tzinfo=IST)
        assert config.minutes_to_squareoff(t2) == 30

        t3 = datetime(2026, 9, 10, 15, 20, tzinfo=IST)
        assert config.minutes_to_squareoff(t3) == 0


class TestIntradayCutoffGate:
    """Validate that new Intraday entries are vetoed after 14:30 cutoff."""

    def test_intraday_cutoff_blocks_late_short(self):
        config = SystemConfig()
        config.INTRADAY_ENABLED = True
        config.MIN_PROB_WIN = 0.25
        config.MIN_EXPECTANCY_R = -0.10
        config.SHORT_IS_INTRADAY_ONLY = True
        config.INTRADAY_ENTRY_CUTOFF = "14:30"

        df = _make_df(120, base=100.0, trend=-0.002)
        df["EMA_20"] = df["Close"] * 1.02
        df["EMA_200"] = df["Close"] * 1.10
        df["Super_Up"] = False
        df["Dn_Day"] = 1
        df["Up_Day"] = 0
        bench = df["Close"] * 1.05

        regime = MarketRegime(
            regime=MarketRegimeType.TREND_DOWN,
            breadth=0.35,
            adx_median=25.0,
            atr_ratio=1.0,
            confidence=0.85,
            confirmed=True,
        )

        # 14:45 IST: past 14:30 cutoff
        t_late = datetime(2026, 9, 10, 14, 45, tzinfo=IST)
        res_late = score_ticker(
            ticker="MUTHOOTFIN.NS",
            daily_df=df,
            bench=bench,
            sector_ranks={"Financial Services": 1},
            sector_rs={"Financial Services": -2.0},
            session="CLOSING_TREND",
            regime=regime,
            config=config,
            force_score=False,
            now=t_late,
        )
        assert res_late is None, "Expected intraday short to be vetoed past 14:30 cutoff"

        # 13:45 IST: before 14:30 cutoff -> allowed
        t_valid = datetime(2026, 9, 10, 13, 45, tzinfo=IST)
        res_valid = score_ticker(
            ticker="MUTHOOTFIN.NS",
            daily_df=df,
            bench=bench,
            sector_ranks={"Financial Services": 1},
            sector_rs={"Financial Services": -2.0},
            session="CLOSING_TREND",
            regime=regime,
            config=config,
            force_score=False,
            now=t_valid,
        )
        assert res_valid is not None, "Expected intraday short before 14:30 cutoff to be accepted"
        assert res_valid.trade_horizon == "INTRADAY"
