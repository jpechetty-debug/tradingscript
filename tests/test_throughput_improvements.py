"""
tests/test_throughput_improvements.py
======================================
Unit tests for suggestion throughput improvements (Phase 1):
1. RANGE regime allows mean reversion setups (long & short)
2. Direction gate 2-of-3 intraday logic
3. Watchlist tier identification and flag
4. Dual EMA-200 removal from passes_static_filters
5. Portfolio vs all_results candidates separation
"""

import numpy as np
import pandas as pd
import pytest

from core.config import SystemConfig, MarketRegimeType
from core.regime import MarketRegime
from core.scorer import score_ticker, TickerResult
from core.services import passes_static_filters


def _make_ohlcv(n: int = 200, trend: float = 0.001, base: float = 500.0) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = base * np.cumprod(1 + trend + np.random.normal(0, 0.01, n))
    high = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_ = close * (1 + np.random.normal(0, 0.004, n))
    vol = np.random.randint(1_000_000, 3_000_000, n).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["EMA_200"] = df["Close"].ewm(span=200, adjust=False).mean()
    df["Super_Up"] = True
    df["RSI"] = 48.0
    df["StochRSI_K"] = 35.0
    df["ADX"] = 22.0
    df["MACD_Hist"] = 0.05
    df["ATR"] = 8.0
    df["ATR_Pctile"] = 40.0
    df["ATR_50_mean"] = 7.5
    df["Vol_Contract"] = False
    df["RVol_20"] = 1.2
    df["Dist_52w_High"] = 5.0
    df["Vol_Avg_20"] = 1_500_000.0
    df["Turnover_Avg_20"] = 750_000_000.0
    df["Up_Day"] = 1
    df["Dn_Day"] = 0
    return df


def _regime(rt=MarketRegimeType.RANGE, confirmed=True) -> MarketRegime:
    return MarketRegime(
        regime=rt,
        breadth=0.50,
        adx_median=20.0,
        atr_ratio=1.0,
        confidence=0.70,
        confirmed=confirmed,
    )


def _config(**overrides) -> SystemConfig:
    cfg = SystemConfig()
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


class TestRangeMeanReversion:
    def test_allows_mean_reversion_true_only_in_confirmed_range(self):
        r_range_conf = _regime(MarketRegimeType.RANGE, confirmed=True)
        r_range_unconf = _regime(MarketRegimeType.RANGE, confirmed=False)
        r_trend = _regime(MarketRegimeType.TREND_UP, confirmed=True)

        assert r_range_conf.allows_mean_reversion() is True
        assert r_range_unconf.allows_mean_reversion() is False
        assert r_trend.allows_mean_reversion() is False

    def test_score_ticker_allows_long_in_confirmed_range_regime(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["Close"] = 510.0
        df["EMA_20"] = 500.0
        df["EMA_200"] = 450.0
        df["RSI"] = 45.0  # Oversold / pullback setup

        cfg = _config(
            USE_EMA200_FILTER=True,
            MIN_PROB_WIN=0.0,
            MIN_EXPECTANCY_R=-99.0,
            USE_VALUE_AREA_RR=False,
        )
        regime = _regime(MarketRegimeType.RANGE, confirmed=True)

        res = score_ticker(
            ticker="TICKER.NS",
            daily_df=df,
            bench=df["Close"],
            sector_ranks={"IT": 1},
            sector_rs={"IT": 1.0},
            session="OPENING_RANGE",
            regime=regime,
            config=cfg,
        )

        assert res is not None
        assert res.direction == "LONG"
        assert "MeanRev✅" in res.reasons

    def test_score_ticker_blocks_range_when_unconfirmed(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["Close"] = 510.0
        df["EMA_20"] = 500.0
        df["EMA_200"] = 450.0
        df["RSI"] = 45.0

        cfg = _config(MIN_PROB_WIN=0.0, MIN_EXPECTANCY_R=-99.0)
        regime = _regime(MarketRegimeType.RANGE, confirmed=False)

        res = score_ticker(
            ticker="TICKER.NS",
            daily_df=df,
            bench=df["Close"],
            sector_ranks={"IT": 1},
            sector_rs={"IT": 1.0},
            session="OPENING_RANGE",
            regime=regime,
            config=cfg,
        )
        assert res is None


class TestDirectionGateRelaxation:
    def test_intraday_2_of_3_allows_dip_below_vwap(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["Close"] = 510.0
        df["EMA_20"] = 500.0
        df["EMA_200"] = 450.0

        cfg = _config(MIN_PROB_WIN=0.0, MIN_EXPECTANCY_R=-99.0, USE_VALUE_AREA_RR=False)
        regime = _regime(MarketRegimeType.TREND_UP, confirmed=True)

        # 2 of 3: Super_Up=True, Close > EMA_20, but intraday price temporarily below VWAP
        intraday = {"live_price": 510.0, "above_vwap": False}

        res = score_ticker(
            ticker="TICKER.NS",
            daily_df=df,
            bench=df["Close"],
            sector_ranks={"IT": 1},
            sector_rs={"IT": 1.0},
            session="OPENING_RANGE",
            regime=regime,
            config=cfg,
            intraday=intraday,
        )
        assert res is not None
        assert res.direction == "LONG"


class TestWatchlistTier:
    def test_watchlist_tier_returned_when_allow_watchlist_true(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["Close"] = 510.0
        df["EMA_20"] = 500.0
        df["EMA_200"] = 450.0

        # Set MIN_PROB_WIN to high threshold (0.80), but WATCHLIST_MIN_PROB to 0.40
        cfg = _config(
            MIN_PROB_WIN=0.80,
            WATCHLIST_MIN_PROB=0.40,
            MIN_EXPECTANCY_R=0.01,
            USE_VALUE_AREA_RR=False,
        )
        regime = _regime(MarketRegimeType.TREND_UP, confirmed=True)

        # Without allow_watchlist: rejected
        res_default = score_ticker(
            ticker="TICKER.NS",
            daily_df=df,
            bench=df["Close"],
            sector_ranks={"IT": 1},
            sector_rs={"IT": 1.0},
            session="OPENING_RANGE",
            regime=regime,
            config=cfg,
            allow_watchlist=False,
        )
        assert res_default is None

        # With allow_watchlist: returned with is_watchlist=True
        res_watchlist = score_ticker(
            ticker="TICKER.NS",
            daily_df=df,
            bench=df["Close"],
            sector_ranks={"IT": 1},
            sector_rs={"IT": 1.0},
            session="OPENING_RANGE",
            regime=regime,
            config=cfg,
            allow_watchlist=True,
        )
        assert res_watchlist is not None
        assert res_watchlist.is_watchlist is True
        assert "Watchlist" in res_watchlist.reasons


class TestStaticFiltersEma200Removal:
    def test_passes_static_filters_allows_below_ema200(self):
        df = _make_ohlcv(n=200)
        # Price is below EMA 200, but ADV turnover is ample
        df["Close"] = 100.0
        df["Volume"] = 1_000_000.0  # Turnover = 100M >= 35M floor

        cfg = _config(USE_EMA200_FILTER=True, ADV_TURNOVER_FLOOR=35_000_000)
        # Before fix, this returned False; now it returns True so scorer can evaluate SHORTs
        assert passes_static_filters(df, cfg) is True
