"""
core/indicators.py
==================
Technical indicator computation for the Sovereign Engine.

Fixes vs original
-----------------
* ``ema12`` / ``ema26`` defined BEFORE the MACD line (were used before
  assignment — NameError at runtime).
* ``RVol_20`` computed correctly as ``volume / prior_20bar_mean``, not as
  ``close.pct_change().rolling(20).std()`` (which is price volatility, not
  relative volume).
* All magic numeric literals extracted to named module-level constants so
  they are easy to audit and tune.
* Full type annotations on every public symbol.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

from .config import SystemConfig

# ── Indicator parameter constants ─────────────────────────────────────────────
EMA_FAST_SPAN: int      = 12
EMA_SLOW_SPAN: int      = 26
MACD_SIGNAL_SPAN: int   = 9
RSI_PERIOD: int         = 14
STOCHRSI_PERIOD: int    = 14
STOCHRSI_K_SMOOTH: int  = 3
STOCHRSI_D_SMOOTH: int  = 3
ATR_PERIOD: int         = 14
ATR_LONG_PERIOD: int    = 50
ATR_ANNUAL_BARS: int    = 252
ATR_MIN_PERIODS: int    = 50
BB_PERIOD: int          = 20
BB_STD_MULT: float      = 2.0
BB_SQUEEZE_LOOKBACK: int  = 50
BB_SQUEEZE_RATIO: float   = 0.85
VOL_AVG_PERIOD: int     = 20
RVOL_PERIOD: int        = 20

# Columns that must be non-NaN after indicator computation.
DROPNA_COLS: list[str] = [
    "EMA_20", "EMA_50", "EMA_200",
    "RSI", "ATR", "ADX", "Vol_Avg_20", "MACD_Hist",
]


# ── Private helpers ────────────────────────────────────────────────────────────

def _true_range(df: pd.DataFrame) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - df["Close"].shift(1)).abs()
    lpc = (df["Low"]  - df["Close"].shift(1)).abs()
    return pd.concat([hl, hpc, lpc], axis=1).max(axis=1)


def _wilder(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False


def _supertrend_inner_loop_py(
    close: np.ndarray,
    final_upper: np.ndarray,
    final_lower: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure-Python implementation of recursive Supertrend band tracking."""
    for i in range(1, n):
        if np.isnan(final_upper[i - 1]) or np.isnan(final_lower[i - 1]):
            continue
        final_upper[i] = (
            final_upper[i] if close[i - 1] > final_upper[i - 1]
            else min(final_upper[i], final_upper[i - 1])
        )
        final_lower[i] = (
            final_lower[i] if close[i - 1] < final_lower[i - 1]
            else max(final_lower[i], final_lower[i - 1])
        )

    trend_up = np.empty(n, dtype=np.bool_)
    st = np.full(n, np.nan)

    seed = -1
    for i in range(n):
        if not (np.isnan(final_upper[i]) or np.isnan(final_lower[i])):
            seed = i
            break

    if seed == -1:
        return st, trend_up

    trend_up[seed] = close[seed] >= final_lower[seed]
    st[seed] = final_lower[seed] if trend_up[seed] else final_upper[seed]

    for i in range(seed + 1, n):
        if st[i - 1] == final_lower[i - 1]:
            trend_up[i] = close[i] >= final_lower[i]
        else:
            trend_up[i] = close[i] > final_upper[i]
        st[i] = final_lower[i] if trend_up[i] else final_upper[i]

    return st, trend_up


if _NUMBA_AVAILABLE:
    _supertrend_inner_loop = njit(fastmath=True, cache=True)(_supertrend_inner_loop_py)
else:
    _supertrend_inner_loop = _supertrend_inner_loop_py


def _supertrend_vectorised(
    df: pd.DataFrame,
    period: int,
    mult: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised Supertrend — returns (supertrend_line, trend_up_mask).
    Accelerated via Numba JIT when available with pure-Python fallback.
    """
    tr = _true_range(df)
    atr = _wilder(tr, period)
    hl2 = (df["High"] + df["Low"]) / 2
    close = np.ascontiguousarray(df["Close"].values, dtype=np.float64)
    n = len(close)

    final_upper = np.ascontiguousarray((hl2 + mult * atr).values, dtype=np.float64)
    final_lower = np.ascontiguousarray((hl2 - mult * atr).values, dtype=np.float64)

    return _supertrend_inner_loop(close, final_upper, final_lower, n)  # type: ignore[no-any-return]


# ── Public API ────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame, config: SystemConfig) -> pd.DataFrame:
    """
    Compute all technical indicators and append them as new columns.

    Parameters
    ----------
    df:
        OHLCV DataFrame with columns Open / High / Low / Close / Volume.
    config:
        ``SystemConfig`` instance providing SUPER_PERIOD, SUPER_MULT,
        ADX_PERIOD.

    Returns
    -------
    A copy of *df* with indicator columns added; rows with NaN in the
    core indicator set are dropped in-place before returning.
    """
    df = df.copy()
    c  = df["Close"]

    # ── EMA stack ─────────────────────────────────────────────────────────────
    df["EMA_20"]  = c.ewm(span=20,  adjust=False).mean()
    df["EMA_50"]  = c.ewm(span=50,  adjust=False).mean()
    df["EMA_200"] = c.ewm(span=200, adjust=False).mean()

    # ── MACD — ema12/ema26 must be defined before the difference line ─────────
    ema12     = c.ewm(span=EMA_FAST_SPAN,  adjust=False).mean()
    ema26     = c.ewm(span=EMA_SLOW_SPAN,  adjust=False).mean()
    macd_line = ema12 - ema26
    df["MACD_Hist"] = macd_line - macd_line.ewm(span=MACD_SIGNAL_SPAN, adjust=False).mean()

    # ── RSI (Wilder smoothing) ─────────────────────────────────────────────────
    delta    = c.diff()
    avg_gain = _wilder(delta.clip(lower=0), RSI_PERIOD)
    avg_loss = _wilder((-delta.clip(upper=0)), RSI_PERIOD)
    rs_s     = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
    df["RSI"] = 100 - (100 / (1 + rs_s))

    # ── StochRSI ───────────────────────────────────────────────────────────────
    rsi_s     = df["RSI"]
    rsi_lo    = rsi_s.rolling(STOCHRSI_PERIOD).min()
    rsi_hi    = rsi_s.rolling(STOCHRSI_PERIOD).max()
    stoch_rsi = (rsi_s - rsi_lo) / (rsi_hi - rsi_lo + 1e-9)
    df["StochRSI_K"] = stoch_rsi.rolling(STOCHRSI_K_SMOOTH).mean() * 100
    df["StochRSI_D"] = df["StochRSI_K"].rolling(STOCHRSI_D_SMOOTH).mean()

    # ── ATR + percentile ───────────────────────────────────────────────────────
    tr = _true_range(df)
    df["ATR"]         = _wilder(tr, ATR_PERIOD)
    df["ATR_50_mean"] = df["ATR"].rolling(ATR_LONG_PERIOD).mean()
    df["ATR_Pctile"]  = (
        df["ATR"]
        .rolling(ATR_ANNUAL_BARS, min_periods=ATR_MIN_PERIODS)
        .rank(pct=True) * 100
    )

    # ── Supertrend ─────────────────────────────────────────────────────────────
    st_vals, trend_up = _supertrend_vectorised(df, config.SUPER_PERIOD, config.SUPER_MULT)
    df["Supertrend"] = st_vals
    df["Super_Up"]   = trend_up

    # ── ADX ────────────────────────────────────────────────────────────────────
    adx_p = config.ADX_PERIOD
    h, l_ = df["High"].values, df["Low"].values
    pdm = np.where(
        (h[1:] - h[:-1] > l_[:-1] - l_[1:]) & (h[1:] - h[:-1] > 0),
        h[1:] - h[:-1], 0.0,
    )
    ndm = np.where(
        (l_[:-1] - l_[1:] > h[1:] - h[:-1]) & (l_[:-1] - l_[1:] > 0),
        l_[:-1] - l_[1:], 0.0,
    )
    pdm   = pd.Series(np.insert(pdm, 0, 0.0), index=df.index)
    ndm   = pd.Series(np.insert(ndm, 0, 0.0), index=df.index)
    tr_s  = pd.Series(tr.values, index=df.index)
    pdm_s = _wilder(pdm, adx_p)
    ndm_s = _wilder(ndm, adx_p)
    tr_sm = _wilder(tr_s, adx_p).replace(0, np.nan)
    pdi   = 100 * pdm_s / tr_sm
    ndi   = 100 * ndm_s / tr_sm
    dx    = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    df["ADX"] = _wilder(dx, adx_p)

    # ── Bollinger Bands ────────────────────────────────────────────────────────
    bb_mid = c.rolling(BB_PERIOD).mean()
    bb_sd  = c.rolling(BB_PERIOD).std(ddof=0)
    df["BB_Upper"]   = bb_mid + BB_STD_MULT * bb_sd
    df["BB_Lower"]   = bb_mid - BB_STD_MULT * bb_sd
    df["BB_Width"]   = (df["BB_Upper"] - df["BB_Lower"]) / bb_mid
    bw_avg           = df["BB_Width"].rolling(BB_SQUEEZE_LOOKBACK).mean()
    df["BB_Squeeze"] = df["BB_Width"] < (bw_avg * BB_SQUEEZE_RATIO)

    # ── Volume metrics ─────────────────────────────────────────────────────────
    df["Vol_Avg_20"]      = df["Volume"].rolling(VOL_AVG_PERIOD).mean()
    df["Turnover_Avg_20"] = (c * df["Volume"]).rolling(VOL_AVG_PERIOD).mean()

    # RVol_20: today's volume / prior 20-bar average (shift avoids look-ahead).
    # This is relative volume — dimensionless ratio, not price volatility.
    vol_rolling_mean = df["Volume"].shift(1).rolling(RVOL_PERIOD).mean()
    df["RVol_20"]    = df["Volume"] / vol_rolling_mean.replace(0, np.nan)

    # ── Candle direction helpers ───────────────────────────────────────────────
    df["Up_Day"] = (c > df["Open"]).astype(int)
    df["Dn_Day"] = (c < df["Open"]).astype(int)

    df.dropna(subset=DROPNA_COLS, inplace=True)
    return df
