import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.factors import calibrate_ic_weights, FactorScores
from core.scorer import TickerResult
from core.config import SystemConfig

# ── Helpers ──────────────────────────────────────────────────────────────────

def _config(**overrides) -> SystemConfig:
    cfg = SystemConfig()
    for k, v in overrides.items():
        object.__setattr__(cfg, k, v)
    return cfg

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

def _bench(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        df["Close"].values * np.linspace(1.0, 0.97, len(df)),
        index=df.index,
    )

def _make_ticker_result(ticker: str = "RELIANCE") -> TickerResult:
    fs = FactorScores(
        trend=0.8, momentum=0.7, volume=0.6, volatility=0.65,
        rs=0.7, breakout=0.6, quality=0.7, composite=0.68,
        ic_weights={k: 1/7 for k in ["trend","momentum","volume","volatility","rs","breakout","quality"]},
    )
    return TickerResult(
        ticker=ticker, sector="Energy", direction="LONG",
        close=500.0, change_pct=0.5, factors=fs, composite=0.68,
        prob_win=0.6, expectancy_r=0.25, sharpe_rank=1.5,
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

# ── Tests ────────────────────────────────────────────────────────────────────

def test_calibrate_ic_weights_returns_valid_weights():
    results = [_make_ticker_result(ticker="RELIANCE.NS")]
    processed = {"RELIANCE.NS": _make_ohlcv(n=120)}
    bench = _bench(processed["RELIANCE.NS"])
    sector_ranks = {"Energy": 1}
    cfg = _config()
    weights = calibrate_ic_weights(
        results=results, processed=processed, bench=bench,
        sector_ranks=sector_ranks, n_sectors=10, config=cfg,
        lookback=20, calib_offset=2, fwd_bars=3,
    )
    assert set(weights.keys()) == {"trend","momentum","volume","volatility","rs","breakout","quality"}
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(v >= 0.0 for v in weights.values())

def test_calibrate_ic_weights_skips_ticker_without_super_up():
    df = _make_ohlcv(n=80).drop(columns=["Super_Up"])
    results = [_make_ticker_result(ticker="RELIANCE.NS")]
    processed = {"RELIANCE.NS": df}
    bench = _bench(df)
    weights = calibrate_ic_weights(
        results=results, processed=processed, bench=bench,
        sector_ranks={}, n_sectors=10, config=_config(),
        lookback=10, calib_offset=2, fwd_bars=3,
    )
    # Falls back to equal weights when all bars are skipped
    assert all(abs(v - 1/7) < 0.01 for v in weights.values())

def test_calibrate_ic_weights_insufficient_history_returns_equal():
    df = _make_ohlcv(n=15)  # too short for offset+fwd_bars+5
    results = [_make_ticker_result(ticker="RELIANCE.NS")]
    processed = {"RELIANCE.NS": df}
    weights = calibrate_ic_weights(
        results=results, processed=processed,
        bench=_bench(_make_ohlcv(n=15)),
        sector_ranks={}, n_sectors=10, config=_config(),
        lookback=20, calib_offset=5, fwd_bars=5,
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert all(abs(v - 1/7) < 0.01 for v in weights.values())


def test_calibrate_ic_weights_regime_fallback():
    from core.factors import get_regime_factor_weights
    df = _make_ohlcv(n=15)
    results = [_make_ticker_result(ticker="RELIANCE.NS")]
    processed = {"RELIANCE.NS": df}
    weights = calibrate_ic_weights(
        results=results, processed=processed,
        bench=_bench(_make_ohlcv(n=15)),
        sector_ranks={}, n_sectors=10, config=_config(),
        lookback=20, calib_offset=5, fwd_bars=5,
        regime_label="RANGE",
    )
    expected_prior = get_regime_factor_weights("RANGE")
    for f, expected_val in expected_prior.items():
        assert abs(weights[f] - expected_val) < 1e-6


def test_calibrate_ic_weights_bayesian_shrinkage_and_floor():
    # 12 tickers to clear the min-10 ticker hurdle in cross-sectional IC calculation
    results = [_make_ticker_result(ticker=f"TICK{i}.NS") for i in range(12)]
    processed = {
        f"TICK{i}.NS": _make_ohlcv(n=140, base=100.0 + i * 15, trend=0.002 * (i - 5))
        for i in range(12)
    }
    bench = _bench(processed["TICK0.NS"])
    weights = calibrate_ic_weights(
        results=results,
        processed=processed,
        bench=bench,
        sector_ranks={},
        n_sectors=10,
        config=_config(),
        lookback=60,
        calib_offset=5,
        fwd_bars=5,
        regime_label="TREND_UP",
    )
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    # 5% floor rule verification
    for f, w in weights.items():
        assert w >= 0.0499, f"Factor {f} weight {w} violates 5% minimum floor"

