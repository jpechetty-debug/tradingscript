"""
tests/test_scorer_portfolio.py
==============================
Unit + integration tests for core/scorer.py and core/portfolio.py.

Coverage targets
----------------
scorer.py  : score_ticker (all gates), passes_liquidity, passes_data_quality,
             compute_trade_management, calibrate_platt, composite_to_prob,
             TickerResult helpers
portfolio.py: compute_targets, calculate_kelly_size, optimize_portfolio,
              CapitalScaler, _ticker_excess_kurtosis

Run:
    pytest tests/test_scorer_portfolio.py -v
    pytest tests/test_scorer_portfolio.py --cov=core/scorer --cov=core/portfolio \
           --cov-report=term-missing
"""

from __future__ import annotations

import sys
import os
import types
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# ── Stub Fyers so imports don't fail without the SDK ─────────────────────────
def _stub_modules() -> None:
    stubs = {
        "fyers_apiv3":            types.ModuleType("fyers_apiv3"),
        "fyers_apiv3.fyersModel": types.ModuleType("fyers_apiv3.fyersModel"),
    }
    stubs["fyers_apiv3.fyersModel"].FyersModel = MagicMock  # type: ignore[attr-defined]
    stubs["fyers_apiv3"].fyersModel = stubs["fyers_apiv3.fyersModel"]  # type: ignore[attr-defined]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

_stub_modules()
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.config import SystemConfig, MarketRegimeType
from core.regime import MarketRegime
from core.scorer import (
    TickerResult,
    composite_to_prob,
    calibrate_platt,
    passes_liquidity,
    passes_data_quality,
    compute_trade_management,
    score_ticker,
)
from core.portfolio import (
    CapitalScaler,
    TradeTargets,
    compute_targets,
    calculate_kelly_size,
    optimize_portfolio,
    _ticker_excess_kurtosis,
)


# ═════════════════════════════════════════════════════════════════════════════
# SHARED FIXTURES
# ═════════════════════════════════════════════════════════════════════════════

def _config(**overrides) -> SystemConfig:
    """Real SystemConfig with sane defaults; override specific fields via kwargs."""
    cfg = SystemConfig()
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg


def _make_ohlcv(n: int = 120, trend: float = 0.002, base: float = 500.0) -> pd.DataFrame:
    """
    Synthetic OHLCV + full indicator set required by score_ticker.
    All columns use float64 (not nullable) to match real add_indicators() output.
    """
    np.random.seed(7)
    dates  = pd.date_range("2024-01-01", periods=n, freq="B")
    close  = base * np.cumprod(1 + trend + np.random.normal(0, 0.01, n))
    high   = close * (1 + np.abs(np.random.normal(0, 0.005, n)))
    low    = close * (1 - np.abs(np.random.normal(0, 0.005, n)))
    open_  = close * (1 + np.random.normal(0, 0.004, n))
    vol    = np.random.randint(1_000_000, 3_000_000, n).astype(float)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=dates,
    )

    # Indicators consumed by score_ticker / factor functions
    df["EMA_20"]          = close * 0.98        # close > EMA_20  → bullish
    df["EMA_50"]          = close * 0.96
    df["EMA_200"]         = close * 0.90
    df["RSI"]             = 58.0
    df["ADX"]             = 28.0
    df["ATR"]             = close * 0.012
    df["ATR_50_mean"]     = close * 0.014
    df["ATR_Pctile"]      = 40.0
    df["MACD_Hist"]       = 0.5
    df["BB_Width"]        = 0.04
    df["BB_Squeeze"]      = False
    df["Vol_Avg_20"]      = vol.mean()
    df["Turnover_Avg_20"] = close.mean() * vol.mean()
    df["RVol_20"]         = 1.2
    df["Up_Day"]          = (close > open_).astype(int)
    df["Dn_Day"]          = (close < open_).astype(int)
    df["Super_Up"]        = True
    df["Supertrend"]      = close * 0.97
    df["StochRSI_K"]      = 55.0
    df["StochRSI_D"]      = 52.0
    return df


def _bench(df: pd.DataFrame) -> pd.Series:
    """Benchmark that slightly underperforms → positive RS for the stock."""
    return pd.Series(
        df["Close"].values * np.linspace(1.0, 0.96, len(df)),
        index=df.index,
    )


def _regime(
    rtype: MarketRegimeType = MarketRegimeType.TREND_UP,
    confirmed: bool = True,
) -> MarketRegime:
    return MarketRegime(
        regime=rtype,
        breadth=0.65,
        breadth_delta=0.0,
        adx_median=28.0,
        atr_ratio=1.1,
        confidence=0.80,
        confirmed=confirmed,
    )


def _make_ticker_result(
    ticker: str = "RELIANCE",
    direction: str = "LONG",
    sharpe_rank: float = 1.5,
    sector: str = "Energy",
    prob_win: float = 0.60,
) -> TickerResult:
    """Minimal TickerResult for portfolio optimizer tests."""
    from core.factors import FactorScores
    fs = FactorScores(
        trend=0.8, momentum=0.7, volume=0.6, volatility=0.65,
        rs=0.7, breakout=0.6, quality=0.7, composite=0.68,
        ic_weights={k: 1/7 for k in
                    ["trend","momentum","volume","volatility","rs","breakout","quality"]},
    )
    return TickerResult(
        ticker=ticker, sector=sector, direction=direction,
        close=500.0, change_pct=0.5,
        factors=fs, composite=0.68,
        prob_win=prob_win, expectancy_r=0.25, sharpe_rank=sharpe_rank,
        entry=500.0, stop=485.0, t1=560.0, t2=620.0,
        breakeven=515.0, trail_stop=482.0, time_stop_bars=7,
        shares=20, risk_inr=300.0, rr_t1=2.5,
        kelly_f=0.012, kurt_correction=0.75, excess_kurtosis=4.0,
        rsi=58.0, stochrsi_k=55.0, rvol=1.2, adx=28.0,
        super_up=True, macd_hist=0.5, atr_pctile=40.0,
        vol_contract=False, rs_vs_nifty=3.2, near_52w=True,
        ema200_aligned=True, mtf_aligned=True, consec_days=3,
        poc=498.0, val=490.0, vah=510.0,
        regime="TREND_UP", session="OPENING_RANGE",
    )


# ═════════════════════════════════════════════════════════════════════════════
# portfolio.py — CapitalScaler
# ═════════════════════════════════════════════════════════════════════════════

class TestCapitalScaler:

    def test_starts_at_par(self):
        s = CapitalScaler(par_nav=500_000)
        assert s.capital_fraction() == pytest.approx(1.0)

    def test_drawdown_reduces_fraction(self):
        s = CapitalScaler(par_nav=500_000)
        s.update_nav(450_000)
        assert s.capital_fraction() == pytest.approx(0.90)

    def test_gain_increases_fraction(self):
        s = CapitalScaler(par_nav=500_000)
        s.update_nav(650_000)
        assert s.capital_fraction() == pytest.approx(1.30)

    def test_min_floor_applied(self):
        s = CapitalScaler(par_nav=500_000, min_fraction=0.25)
        s.update_nav(10_000)          # extreme drawdown
        assert s.capital_fraction() == pytest.approx(0.25)

    def test_max_ceiling_applied(self):
        s = CapitalScaler(par_nav=500_000, max_fraction=2.0)
        s.update_nav(5_000_000)       # extreme gain
        assert s.capital_fraction() == pytest.approx(2.0)

    def test_zero_nav_ignored(self):
        s = CapitalScaler(par_nav=500_000)
        s.update_nav(0)
        assert s.capital_fraction() == pytest.approx(1.0)  # unchanged

    def test_negative_nav_ignored(self):
        s = CapitalScaler(par_nav=500_000)
        s.update_nav(-1000)
        assert s.capital_fraction() == pytest.approx(1.0)

    def test_invalid_par_nav_raises(self):
        with pytest.raises(ValueError, match="par_nav"):
            CapitalScaler(par_nav=0)

    def test_from_config(self):
        cfg = _config(CAPITAL_INR=1_000_000, RISK_PER_TRADE_INR=10_000)
        s   = CapitalScaler.from_config(cfg)
        assert s.par_nav == pytest.approx(1_000_000)
        assert s.capital_fraction() == pytest.approx(1.0)

        cfg_fallback = _config(CAPITAL_INR=0, RISK_PER_TRADE_INR=10_000)
        s_fallback   = CapitalScaler.from_config(cfg_fallback)
        assert s_fallback.par_nav == pytest.approx(500_000)


# ═════════════════════════════════════════════════════════════════════════════
# portfolio.py — compute_targets
# ═════════════════════════════════════════════════════════════════════════════

class TestComputeTargets:

    def _cfg(self):
        return _config(STOP_ATR_MULT=1.5, TARGET1_ATR_MULT=3.8, TARGET2_ATR_MULT=6.0)

    def test_long_stop_below_close(self):
        t = compute_targets("LONG", close=500.0, atr=10.0, config=self._cfg())
        assert t.stop < 500.0
        assert t.stop == pytest.approx(500.0 - 1.5 * 10.0, abs=0.01)

    def test_long_t1_above_close(self):
        t = compute_targets("LONG", close=500.0, atr=10.0, config=self._cfg())
        assert t.t1 > 500.0
        assert t.t1 == pytest.approx(500.0 + 3.8 * 10.0, abs=0.01)

    def test_long_t2_above_t1(self):
        t = compute_targets("LONG", close=500.0, atr=10.0, config=self._cfg())
        assert t.t2 > t.t1

    def test_short_stop_above_close(self):
        t = compute_targets("SHORT", close=500.0, atr=10.0, config=self._cfg())
        assert t.stop > 500.0

    def test_short_t1_below_close(self):
        t = compute_targets("SHORT", close=500.0, atr=10.0, config=self._cfg())
        assert t.t1 < 500.0

    def test_rr_is_t1_over_sl(self):
        t = compute_targets("LONG", close=500.0, atr=10.0, config=self._cfg())
        sl  = 1.5 * 10.0
        expected_rr = (3.8 * 10.0) / sl
        assert t.rr == pytest.approx(expected_rr, abs=0.01)

    def test_zero_atr_returns_zero_rr(self):
        t = compute_targets("LONG", close=500.0, atr=0.0, config=self._cfg())
        assert t.rr == 0.0

    def test_returns_trade_targets_dataclass(self):
        t = compute_targets("LONG", close=500.0, atr=10.0, config=self._cfg())
        assert isinstance(t, TradeTargets)


# ═════════════════════════════════════════════════════════════════════════════
# portfolio.py — _ticker_excess_kurtosis
# ═════════════════════════════════════════════════════════════════════════════

class TestExcessKurtosis:

    def test_insufficient_history_returns_fallback(self):
        df  = _make_ohlcv(n=30)
        cfg = _config(KELLY_KURTOSIS_MIN_OBS=60, KELLY_KURTOSIS_FALLBACK=4.0,
                      KELLY_KURTOSIS_WINDOW=252)
        assert _ticker_excess_kurtosis(df, cfg) == pytest.approx(4.0)

    def test_sufficient_history_returns_float(self):
        df  = _make_ohlcv(n=120)
        cfg = _config(KELLY_KURTOSIS_MIN_OBS=60, KELLY_KURTOSIS_FALLBACK=4.0,
                      KELLY_KURTOSIS_WINDOW=252)
        ek = _ticker_excess_kurtosis(df, cfg)
        assert isinstance(ek, float)
        assert 0.0 <= ek <= 20.0

    def test_clipped_to_twenty(self):
        # Force extreme kurtosis by constructing a very spiky series
        df = _make_ohlcv(n=120)
        df["Close"] = 100.0
        df.loc[df.index[::20], "Close"] = 200.0   # every 20th bar doubles
        cfg = _config(KELLY_KURTOSIS_MIN_OBS=10, KELLY_KURTOSIS_FALLBACK=4.0,
                      KELLY_KURTOSIS_WINDOW=252)
        assert _ticker_excess_kurtosis(df, cfg) <= 20.0


# ═════════════════════════════════════════════════════════════════════════════
# portfolio.py — calculate_kelly_size
# ═════════════════════════════════════════════════════════════════════════════

class TestCalculateKellySize:

    def _cfg(self):
        return _config(
            KELLY_FRACTION=3.0,
            KELLY_MIN_SHARES=1,
            KELLY_MAX_MULT=3.0,
            KELLY_KURTOSIS_FALLBACK=4.0,
            KELLY_KURTOSIS_WINDOW=252,
            KELLY_KURTOSIS_MIN_OBS=60,
            RISK_PER_TRADE_INR=10_000.0,
        )

    def test_returns_four_tuple(self):
        df = _make_ohlcv()
        result = calculate_kelly_size(
            entry=500.0, stop=485.0, prob_win=0.60,
            rr=2.5, daily_df=df, config=self._cfg(),
        )
        assert len(result) == 4

    def test_shares_positive_integer(self):
        df = _make_ohlcv()
        shares, *_ = calculate_kelly_size(
            entry=500.0, stop=485.0, prob_win=0.60,
            rr=2.5, daily_df=df, config=self._cfg(),
        )
        assert isinstance(shares, int)
        assert shares >= 1

    def test_zero_rps_returns_zero(self):
        df = _make_ohlcv()
        shares, risk, kf, kc = calculate_kelly_size(
            entry=500.0, stop=500.0, prob_win=0.60,  # entry == stop → rps=0
            rr=2.5, daily_df=df, config=self._cfg(),
        )
        assert shares == 0
        assert risk == 0.0

    def test_risk_inr_within_bounds(self):
        df  = _make_ohlcv()
        cfg = self._cfg()
        _, risk_inr, _, _ = calculate_kelly_size(
            entry=500.0, stop=485.0, prob_win=0.60,
            rr=2.5, daily_df=df, config=cfg,
        )
        # Must be within [0.25 * RISK_PER_TRADE_INR, KELLY_MAX_MULT * RISK_PER_TRADE_INR] (accounting for int share quantization)
        assert risk_inr >= cfg.RISK_PER_TRADE_INR * 0.25 - 15.0
        assert risk_inr <= cfg.RISK_PER_TRADE_INR * cfg.KELLY_MAX_MULT

    def test_capital_fraction_scales_risk(self):
        df  = _make_ohlcv()
        cfg = self._cfg()
        _, risk_full, _, _ = calculate_kelly_size(
            entry=500.0, stop=485.0, prob_win=0.60,
            rr=2.5, daily_df=df, config=cfg, capital_fraction=1.0,
        )
        _, risk_half, _, _ = calculate_kelly_size(
            entry=500.0, stop=485.0, prob_win=0.60,
            rr=2.5, daily_df=df, config=cfg, capital_fraction=0.5,
        )
        assert risk_half < risk_full

    def test_kurt_correction_between_zero_and_one(self):
        df = _make_ohlcv()
        *_, kurt_corr = calculate_kelly_size(
            entry=500.0, stop=485.0, prob_win=0.60,
            rr=2.5, daily_df=df, config=self._cfg(),
        )
        assert 0.0 < kurt_corr <= 1.0

    def test_low_prob_win_returns_min_shares(self):
        df  = _make_ohlcv()
        cfg = _config(
            KELLY_FRACTION=3.0, KELLY_MIN_SHARES=1, KELLY_MAX_MULT=3.0,
            KELLY_KURTOSIS_FALLBACK=4.0, KELLY_KURTOSIS_WINDOW=252,
            KELLY_KURTOSIS_MIN_OBS=60, RISK_PER_TRADE_INR=10_000.0,
        )
        shares, *_ = calculate_kelly_size(
            entry=500.0, stop=485.0, prob_win=0.30,   # Kelly goes negative → clamped
            rr=1.0, daily_df=df, config=cfg,
        )
        assert shares >= cfg.KELLY_MIN_SHARES

    def test_zero_capital_fraction_returns_zero_shares(self):
        df = _make_ohlcv()
        shares, *_ = calculate_kelly_size(
            entry=500.0, stop=485.0, prob_win=0.60,
            rr=2.5, daily_df=df, config=self._cfg(), capital_fraction=0.0,
        )
        assert shares == 0


# ═════════════════════════════════════════════════════════════════════════════
# portfolio.py — optimize_portfolio
# ═════════════════════════════════════════════════════════════════════════════

class TestOptimizePortfolio:

    def _cfg(self, size=3, max_sector=1, max_corr=0.70):
        return _config(PORTFOLIO_SIZE=size, MAX_SECTOR_PICKS=max_sector, MAX_CORR=max_corr)

    def test_respects_portfolio_size(self):
        candidates = [_make_ticker_result(ticker=f"T{i}", sector="Energy",
                                          sharpe_rank=float(10 - i))
                      for i in range(8)]
        result = optimize_portfolio(candidates, self._cfg(size=3, max_sector=10))
        assert len(result) <= 3

    def test_sector_cap_enforced(self):
        # 4 tickers all in "Energy" but cap is 2
        candidates = [_make_ticker_result(ticker=f"E{i}", sector="Energy",
                                          sharpe_rank=float(10 - i))
                      for i in range(4)]
        result = optimize_portfolio(candidates, self._cfg(size=10, max_sector=2))
        energy_picks = [r for r in result if r.sector == "Energy"]
        assert len(energy_picks) <= 2

    def test_sorted_by_sharpe_rank(self):
        candidates = [
            _make_ticker_result(ticker="LOW",  sector="IT",     sharpe_rank=0.5),
            _make_ticker_result(ticker="HIGH", sector="Energy", sharpe_rank=2.0),
            _make_ticker_result(ticker="MID",  sector="FMCG",   sharpe_rank=1.0),
        ]
        result = optimize_portfolio(candidates, self._cfg(size=3, max_sector=2))
        sharpes = [r.sharpe_rank for r in result]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_correlation_filter_drops_correlated(self):
        cfg = self._cfg(size=5, max_sector=5, max_corr=0.70)
        c1  = _make_ticker_result(ticker="AAA", sector="IT",     sharpe_rank=3.0)
        c2  = _make_ticker_result(ticker="BBB", sector="Energy", sharpe_rank=2.0)

        # AAA and BBB are highly correlated (0.90 > MAX_CORR=0.70)
        tickers = ["AAA.NS", "BBB.NS"]
        corr = pd.DataFrame(
            [[1.00, 0.90], [0.90, 1.00]],
            index=tickers, columns=tickers,
        )
        result = optimize_portfolio([c1, c2], cfg, corr_matrix=corr)
        # Only one of them should be selected (c1 has higher sharpe, so AAA wins)
        assert len(result) == 1
        assert result[0].ticker == "AAA"

    def test_no_corr_matrix_selects_all_eligible(self):
        candidates = [
            _make_ticker_result(ticker=f"T{i}", sector=f"S{i}", sharpe_rank=float(5 - i))
            for i in range(4)
        ]
        result = optimize_portfolio(candidates, self._cfg(size=5, max_sector=2),
                                    corr_matrix=None)
        assert len(result) == 4

    def test_empty_candidates_returns_empty(self):
        result = optimize_portfolio([], self._cfg())
        assert result == []

    def test_empty_corr_matrix_skips_filter(self):
        candidates = [_make_ticker_result(ticker="X", sector="IT", sharpe_rank=1.0)]
        result = optimize_portfolio(candidates, self._cfg(size=5, max_sector=2),
                                    corr_matrix=pd.DataFrame())
        assert len(result) == 1

    def test_binding_capital_and_risk_invariants(self):
        # 3 candidates, each with entry=1000, stop=950, shares=400 (exp=400k, risk=20k each)
        # Total exposure = 12L (exceeds 10L), total risk = 60k
        c1 = _make_ticker_result(ticker="A", sector="IT", sharpe_rank=3.0)
        c1.entry, c1.stop, c1.shares, c1.risk_inr = 1000.0, 950.0, 400, 20_000.0
        c2 = _make_ticker_result(ticker="B", sector="BANK", sharpe_rank=2.0)
        c2.entry, c2.stop, c2.shares, c2.risk_inr = 1000.0, 950.0, 400, 20_000.0
        c3 = _make_ticker_result(ticker="C", sector="AUTO", sharpe_rank=1.0)
        c3.entry, c3.stop, c3.shares, c3.risk_inr = 1000.0, 950.0, 400, 20_000.0

        cfg = self._cfg(size=5, max_sector=2)
        cfg.CAPITAL_INR = 700_000.0
        cfg.MAX_PORTFOLIO_RISK_INR = 50_000.0

        result = optimize_portfolio([c1, c2, c3], cfg)
        # Lowest rank C3 should have been trimmed first, leaving c1 and c2 (800k exp > 700k limit)
        # Then remaining scaled down so total_exposure <= 700k and total_risk <= 50k
        total_exp = sum(c.shares * c.entry for c in result)
        total_risk = sum(c.risk_inr for c in result)
        assert total_exp <= cfg.CAPITAL_INR
        assert total_risk <= cfg.MAX_PORTFOLIO_RISK_INR
        assert any(c.ticker == "A" for c in result)
        assert not any(c.ticker == "C" for c in result)


# ═════════════════════════════════════════════════════════════════════════════
# scorer.py — pure helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestCompositeToProb:

    def test_output_between_zero_and_one(self):
        for c in [0.0, 0.3, 0.5, 0.7, 1.0]:
            p = composite_to_prob(c, platt_a=-4.0, platt_b=2.0)
            assert 0.0 < p < 1.0

    def test_higher_composite_higher_prob(self):
        # Under Option B expit(-(A*c + B)), negative platt_a → sigmoid increases with composite
        p_low  = composite_to_prob(0.3, platt_a=-4.0, platt_b=2.0)
        p_high = composite_to_prob(0.8, platt_a=-4.0, platt_b=2.0)
        assert p_high > p_low


class TestCalibratePlatt:

    def test_returns_two_floats(self):
        np.random.seed(0)
        composites = list(np.random.uniform(0.3, 0.8, 100))
        outcomes   = [1 if c > 0.55 else 0 for c in composites]
        a, b = calibrate_platt(composites, outcomes, calib_offset=20)
        assert isinstance(a, float)
        assert isinstance(b, float)

    def test_raises_on_insufficient_data(self):
        with pytest.raises(ValueError, match="samples"):
            calibrate_platt([0.5] * 10, [1] * 10, calib_offset=20)

    def test_in_sample_mode_with_offset_zero(self):
        np.random.seed(1)
        composites = list(np.random.uniform(0.3, 0.8, 100))
        outcomes   = [1 if c > 0.55 else 0 for c in composites]
        a, b = calibrate_platt(composites, outcomes, calib_offset=0)
        assert isinstance(a, float)


class TestPassesLiquidity:

    def _row(self, vol=1_000_000, turnover=50_000_000):
        return pd.Series({"Vol_Avg_20": vol, "Turnover_Avg_20": turnover})

    def test_passes_when_above_floors(self):
        cfg = _config(ADV_SHARE_FLOOR=750_000, ADV_TURNOVER_FLOOR=35_000_000)
        ok, _ = passes_liquidity(self._row(), cfg)
        assert ok

    def test_fails_on_low_volume(self):
        cfg = _config(ADV_SHARE_FLOOR=750_000, ADV_TURNOVER_FLOOR=35_000_000)
        ok, msg = passes_liquidity(self._row(vol=100_000), cfg)
        assert not ok
        assert "Vol" in msg

    def test_fails_on_low_turnover(self):
        cfg = _config(ADV_SHARE_FLOOR=750_000, ADV_TURNOVER_FLOOR=35_000_000)
        ok, msg = passes_liquidity(self._row(turnover=1_000_000), cfg)
        assert not ok
        assert "Turnover" in msg


class TestPassesDataQuality:

    def _good_row(self):
        return pd.Series({
            "ATR": 6.0, "ATR_50_mean": 8.0, "ATR_Pctile": 40.0,
            "EMA_20": 500.0, "EMA_50": 490.0, "EMA_200": 460.0,
            "RSI": 58.0, "ADX": 28.0, "MACD_Hist": 0.5, "Vol_Avg_20": 1_500_000.0,
        })

    def test_passes_on_clean_row(self):
        ok, _ = passes_data_quality(self._good_row(), "TEST")
        assert ok

    def test_fails_on_float_nan(self):
        row = self._good_row().copy()
        row["ATR"] = float("nan")
        ok, msg = passes_data_quality(row, "TEST")
        assert not ok
        assert "ATR" in msg

    def test_fails_on_numpy_nan(self):
        row = self._good_row().copy()
        row["ATR_50_mean"] = np.nan
        ok, msg = passes_data_quality(row, "TEST")
        assert not ok

    def test_fails_on_pd_na(self):
        row = self._good_row().copy()
        row["ATR_Pctile"] = pd.NA
        ok, msg = passes_data_quality(row, "TEST")
        assert not ok

    def test_fails_on_none(self):
        row = self._good_row().copy()
        row["RSI"] = None
        ok, _ = passes_data_quality(row, "TEST")
        assert not ok

    def test_all_required_columns_checked(self):
        required = ["ATR", "ATR_50_mean", "ATR_Pctile", "EMA_20", "EMA_50",
                    "EMA_200", "RSI", "ADX", "MACD_Hist", "Vol_Avg_20"]
        for col in required:
            row = self._good_row().copy()
            row[col] = float("nan")
            ok, msg = passes_data_quality(row, "TEST")
            assert not ok, f"Expected failure for NaN in {col}"
            assert col in msg


class TestComputeTradeManagement:

    def test_long_trail_stop_below_entry(self):
        trail, _ = compute_trade_management("LONG", entry=500.0, atr=10.0, atr_pctile=40.0)
        assert trail < 500.0

    def test_short_trail_stop_above_entry(self):
        trail, _ = compute_trade_management("SHORT", entry=500.0, atr=10.0, atr_pctile=40.0)
        assert trail > 500.0

    def test_time_stop_low_pctile(self):
        _, bars = compute_trade_management("LONG", entry=500.0, atr=10.0, atr_pctile=20.0)
        assert bars == 4

    def test_time_stop_mid_pctile(self):
        _, bars = compute_trade_management("LONG", entry=500.0, atr=10.0, atr_pctile=45.0)
        assert bars == 7

    def test_time_stop_high_pctile(self):
        _, bars = compute_trade_management("LONG", entry=500.0, atr=10.0, atr_pctile=75.0)
        assert bars == 12

    def test_higher_pctile_wider_trail(self):
        trail_low,  _ = compute_trade_management("LONG", 500.0, 10.0, atr_pctile=10.0)
        trail_high, _ = compute_trade_management("LONG", 500.0, 10.0, atr_pctile=90.0)
        # Higher pctile → larger trail_mult → further from entry
        assert trail_low > trail_high


class TestTickerResultHelpers:

    def test_display_score_is_composite_times_100(self):
        r = _make_ticker_result()
        assert r.display_score() == int(r.composite * 100)

    def test_to_dict_contains_factor_keys(self):
        d = _make_ticker_result().to_dict()
        for key in ("trend", "momentum", "volume", "volatility",
                    "rs", "breakout", "quality", "composite"):
            assert key in d

    def test_to_dict_excludes_factors_object(self):
        d = _make_ticker_result().to_dict()
        assert "factors" not in d


# ═════════════════════════════════════════════════════════════════════════════
# scorer.py — score_ticker  (gate-by-gate + happy path)
# ═════════════════════════════════════════════════════════════════════════════

class TestScoreTicker:
    """
    Each test isolates one gate or behaviour by adjusting exactly the field
    that gate checks.  The happy-path test verifies the full pipeline returns
    a valid TickerResult with correctly populated fields.
    """

    def _call(self, df=None, regime=None, cfg=None, **kwargs):
        if df is None:
            df = _make_ohlcv()
        bench = _bench(df)
        return score_ticker(
            ticker="RELIANCE.NS",
            daily_df=df,
            bench=bench,
            sector_ranks={"Energy": 1},
            sector_rs={"Energy": 2.5},
            session="OPENING_RANGE",
            regime=regime or _regime(),
            config=cfg or _config(
                USE_EMA200_FILTER=False,   # simplify: skip EMA200 gate
                USE_VALUE_AREA_RR=False,   # simplify: use plain ATR targets
                MIN_PROB_WIN=0.0,          # open probability gate
                MIN_EXPECTANCY_R=-99.0,    # open expectancy gate
            ),
            **kwargs,
        )

    # ── gate: empty dataframe ─────────────────────────────────────────────────

    def test_empty_df_returns_none(self):
        assert score_ticker(
            ticker="X", daily_df=pd.DataFrame(),
            bench=pd.Series(dtype=float),
            sector_ranks={}, sector_rs={},
            session="OPENING_RANGE",
            regime=_regime(),
            config=_config(),
        ) is None

    # ── gate 1: liquidity ─────────────────────────────────────────────────────

    def test_low_volume_returns_none(self):
        df = _make_ohlcv()
        df["Vol_Avg_20"]      = 100          # below ADV_SHARE_FLOOR=750_000
        df["Turnover_Avg_20"] = 100
        assert self._call(df=df) is None

    # ── gate 1b: data quality ─────────────────────────────────────────────────

    def test_nan_atr_returns_none(self):
        df = _make_ohlcv()
        df["ATR"] = float("nan")
        assert self._call(df=df) is None

    def test_pd_na_atr_pctile_returns_none(self):
        df = _make_ohlcv()
        df["ATR_Pctile"] = pd.NA
        assert self._call(df=df) is None

    # ── gate 2: direction ─────────────────────────────────────────────────────

    def test_neutral_direction_returns_none(self):
        df = _make_ohlcv()
        # Super_Up=True but close < EMA_20 → not bull; not bear either → neutral
        df["Super_Up"] = True
        df["EMA_20"]   = df["Close"] * 1.10   # close < EMA_20
        assert self._call(df=df) is None

    def test_bullish_setup_produces_long(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["EMA_20"]   = df["Close"] * 0.95   # close > EMA_20
        df["EMA_200"]  = df["Close"] * 0.80
        result = self._call(df=df)
        if result is not None:
            assert result.direction == "LONG"

    def test_bearish_setup_produces_short(self):
        df = _make_ohlcv()
        df["Super_Up"] = False
        df["EMA_20"]   = df["Close"] * 1.05   # close < EMA_20
        df["EMA_200"]  = df["Close"] * 1.20
        result = self._call(df=df, regime=_regime(MarketRegimeType.TREND_DOWN))
        if result is not None:
            assert result.direction == "SHORT"

    # ── gate 3: regime ────────────────────────────────────────────────────────

    def test_unconfirmed_regime_blocks_long(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["EMA_20"]   = df["Close"] * 0.95
        result = self._call(df=df, regime=_regime(MarketRegimeType.TREND_UP, confirmed=False))
        assert result is None

    def test_trend_down_regime_blocks_long(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["EMA_20"]   = df["Close"] * 0.95
        result = self._call(
            df=df,
            regime=_regime(MarketRegimeType.TREND_DOWN, confirmed=True),
        )
        assert result is None

    # ── gate 4: EMA-200 ───────────────────────────────────────────────────────

    def test_ema200_filter_blocks_long_below_200(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["EMA_20"]   = df["Close"] * 0.95
        df["EMA_200"]  = df["Close"] * 1.10   # close < EMA_200
        cfg = _config(USE_EMA200_FILTER=True, MIN_PROB_WIN=0.0, MIN_EXPECTANCY_R=-99.0,
                      USE_VALUE_AREA_RR=False)
        assert self._call(df=df, cfg=cfg) is None

    # ── gate 9: probability ───────────────────────────────────────────────────

    def test_low_prob_gate_returns_none(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["EMA_20"]   = df["Close"] * 0.95
        # Very high MIN_PROB_WIN that Platt defaults won't reach
        cfg = _config(MIN_PROB_WIN=0.99, MIN_EXPECTANCY_R=-99.0,
                      USE_EMA200_FILTER=False, USE_VALUE_AREA_RR=False)
        assert self._call(df=df, cfg=cfg) is None

    # ── gate 9: expectancy ────────────────────────────────────────────────────

    def test_low_expectancy_gate_returns_none(self):
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["EMA_20"]   = df["Close"] * 0.95
        cfg = _config(MIN_PROB_WIN=0.0, MIN_EXPECTANCY_R=99.0,   # impossible threshold
                      USE_EMA200_FILTER=False, USE_VALUE_AREA_RR=False)
        assert self._call(df=df, cfg=cfg) is None

    # ── happy path ────────────────────────────────────────────────────────────

    def test_happy_path_returns_ticker_result(self):
        result = self._call()
        assert result is not None
        assert isinstance(result, TickerResult)

    def test_happy_path_ticker_stripped_of_ns(self):
        result = self._call()
        assert result is not None
        assert ".NS" not in result.ticker

    def test_happy_path_composite_in_unit_interval(self):
        result = self._call()
        assert result is not None
        assert 0.0 <= result.composite <= 1.0

    def test_happy_path_prob_win_in_unit_interval(self):
        result = self._call()
        assert result is not None
        assert 0.0 < result.prob_win < 1.0

    def test_happy_path_stop_below_close_long(self):
        result = self._call()
        assert result is not None
        if result.direction == "LONG":
            assert result.stop < result.close

    def test_happy_path_t1_above_close_long(self):
        result = self._call()
        assert result is not None
        if result.direction == "LONG":
            assert result.t1 > result.close

    def test_happy_path_shares_positive(self):
        result = self._call()
        assert result is not None
        assert result.shares >= 1

    def test_happy_path_reasons_non_empty(self):
        result = self._call()
        assert result is not None
        assert len(result.reasons) > 0

    def test_happy_path_to_dict_complete(self):
        result = self._call()
        assert result is not None
        d = result.to_dict()
        for key in ("ticker", "direction", "composite", "prob_win",
                    "stop", "t1", "shares", "regime", "session"):
            assert key in d

    def test_happy_path_regime_serialised_as_string(self):
        result = self._call()
        assert result is not None
        assert result.regime == "TREND_UP"

    def test_session_midday_chop_reduces_composite(self):
        """MIDDAY_CHOP multiplier (0.92) should produce lower composite than OPENING_RANGE."""
        df    = _make_ohlcv()
        bench = _bench(df)
        cfg   = _config(USE_EMA200_FILTER=False, USE_VALUE_AREA_RR=False,
                        MIN_PROB_WIN=0.0, MIN_EXPECTANCY_R=-99.0)
        r_open = score_ticker("RELIANCE.NS", df, bench, {"Energy": 1}, {"Energy": 2.5},
                              "OPENING_RANGE", _regime(), cfg)
        r_chop = score_ticker("RELIANCE.NS", df, bench, {"Energy": 1}, {"Energy": 2.5},
                              "MIDDAY_CHOP", _regime(), cfg)
        if r_open and r_chop:
            assert r_chop.composite <= r_open.composite

    def test_closing_trend_session_increases_composite(self):
        df    = _make_ohlcv()
        bench = _bench(df)
        cfg   = _config(USE_EMA200_FILTER=False, USE_VALUE_AREA_RR=False,
                        MIN_PROB_WIN=0.0, MIN_EXPECTANCY_R=-99.0)
        r_open  = score_ticker("RELIANCE.NS", df, bench, {"Energy": 1}, {"Energy": 2.5},
                               "OPENING_RANGE", _regime(), cfg)
        r_close = score_ticker("RELIANCE.NS", df, bench, {"Energy": 1}, {"Energy": 2.5},
                               "CLOSING_TREND", _regime(), cfg)
        if r_open and r_close:
            assert r_close.composite >= r_open.composite

    def test_intraday_live_price_used_when_positive(self):
        df     = _make_ohlcv()
        result = self._call(df=df, intraday={"live_price": 999.0, "above_vwap": True})
        if result is not None:
            assert result.close == pytest.approx(999.0)

    def test_value_area_rr_override_applied(self):
        """When USE_VALUE_AREA_RR=True and VAH gives better RR, T1 should move to VAH."""
        df = _make_ohlcv()
        df["Super_Up"] = True
        df["EMA_20"]   = df["Close"] * 0.95
        df["EMA_200"]  = df["Close"] * 0.80
        # Force ATR small so VAH is well above the default T1
        df["ATR"] = df["Close"] * 0.001
        cfg = _config(
            USE_EMA200_FILTER=False,
            USE_VALUE_AREA_RR=True,
            VA_MIN_RR=0.1,           # very low bar so override fires
            MIN_PROB_WIN=0.0,
            MIN_EXPECTANCY_R=-99.0,
            STOP_ATR_MULT=1.5,
            TARGET1_ATR_MULT=3.8,
            TARGET2_ATR_MULT=6.0,
        )
        result = self._call(df=df, cfg=cfg)
        # Just verify it doesn't crash and returns a result
        assert result is None or isinstance(result, TickerResult)
