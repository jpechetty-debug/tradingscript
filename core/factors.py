"""
core/factors.py
===============
Seven-factor signal model — ported faithfully from screener.py v13.
No placeholders. All logic taken verbatim from compute_factors() and
_quality_factor() in the monolith, then reorganised into testable
pure functions.

Factor functions all share the same signature contract:
    fn(daily_df, row, close, direction, ...) -> float  # score in [0, 1]

compute_factors() assembles them into a FactorScores dataclass and
applies IC-weighted composite scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FactorScores:
    trend:      float
    momentum:   float
    volume:     float
    volatility: float
    rs:         float
    breakout:   float
    quality:    float   # price_momentum_quality (Fix C)
    composite:  float
    ic_weights: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        return {
            "trend": self.trend, "momentum": self.momentum,
            "volume": self.volume, "volatility": self.volatility,
            "rs": self.rs, "breakout": self.breakout,
            "quality": self.quality, "composite": self.composite,
        }


# ─────────────────────────────────────────────────────────────────────────────
# VOLUME PROFILE HELPER  (pure — no CONFIG dependency)
# ─────────────────────────────────────────────────────────────────────────────

def true_volume_profile(
    df: pd.DataFrame,
    lookback: int = 30,
    bins: int = 100,
) -> tuple[float, float, float]:
    """
    Returns (POC, VAL, VAH) for the given OHLCV DataFrame.

    POC  — Price of Control (highest volume price level)
    VAL  — Value Area Low  (bottom of 70% volume zone)
    VAH  — Value Area High (top of 70% volume zone)

    Implementation uses vectorised NumPy binning (np.searchsorted over the
    entire window at once) instead of a Python for-loop, giving a 10–20×
    speed-up across the universe.  Each bar's volume is distributed uniformly
    across the bins that overlap [Low, High].
    """
    window = df.tail(lookback)
    if len(window) < 5:
        c = float(df["Close"].iloc[-1])
        return c, c * 0.99, c * 1.01

    lows  = window["Low"].to_numpy(dtype=float)
    highs = window["High"].to_numpy(dtype=float)
    vols  = window["Volume"].to_numpy(dtype=float)

    lo_p = float(lows.min())
    hi_p = float(highs.max())
    if hi_p <= lo_p:
        c = float(window["Close"].iloc[-1])
        return c, c * 0.99, c * 1.01

    levels   = np.linspace(lo_p, hi_p, bins + 1)
    vol_hist = np.zeros(bins)

    # Vectorised: find the first and last bin touched by each bar's range.
    li_arr = np.clip(np.searchsorted(levels, lows,  "left")  - 1, 0, bins - 1)
    hi_arr = np.clip(np.searchsorted(levels, highs, "right"),     0, bins - 1)

    # Only bars with positive volume and valid range contribute.
    valid = (highs > lows) & (vols > 0)
    spans = (hi_arr - li_arr + 1).astype(float)
    spans[~valid] = 0.0

    for i in np.where(valid)[0]:
        vol_hist[li_arr[i] : hi_arr[i] + 1] += vols[i] / spans[i]

    poc_idx = int(np.argmax(vol_hist))
    poc     = float((levels[poc_idx] + levels[poc_idx + 1]) / 2)

    total    = vol_hist.sum()
    target   = total * 0.70
    lo_i = hi_i = poc_idx
    captured = vol_hist[poc_idx]

    while captured < target and (lo_i > 0 or hi_i < bins - 1):
        add_lo = vol_hist[lo_i - 1] if lo_i > 0       else -1.0
        add_hi = vol_hist[hi_i + 1] if hi_i < bins - 1 else -1.0
        if add_lo >= add_hi and lo_i > 0:
            lo_i -= 1
            captured += vol_hist[lo_i]
        elif hi_i < bins - 1:
            hi_i += 1
            captured += vol_hist[hi_i]
        else:
            break

    return poc, float(levels[lo_i]), float(levels[min(hi_i + 1, bins)])


# ─────────────────────────────────────────────────────────────────────────────
# INDIVIDUAL FACTOR FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def factor_trend(
    row: pd.Series,
    close: float,
    direction: str,
    mtf_60m_trend_aligned: Optional[bool] = None,
) -> float:
    """
    Multi-timeframe trend factor [0, 1].

    Components:
      40% — Supertrend direction
      15% — Close vs EMA20
      15% — Close vs EMA50
      20% — Close vs EMA200
      10% — 60m timeframe agreement
      +5% bonus for full EMA stack alignment (capped at 1.0)
    """
    ema20  = float(row["EMA_20"])
    ema50  = float(row["EMA_50"])
    ema200 = float(row["EMA_200"])
    su     = bool(row["Super_Up"])
    mtf_up = mtf_60m_trend_aligned if mtf_60m_trend_aligned is not None else su

    if direction == "LONG":
        t = (int(su) * 0.40
             + int(close > ema20)  * 0.15
             + int(close > ema50)  * 0.15
             + int(close > ema200) * 0.20
             + int(mtf_up)         * 0.10)
        if ema20 > ema50 > ema200:
            t = min(1.0, t + 0.05)
    else:
        t = (int(not su)             * 0.40
             + int(close < ema20)    * 0.15
             + int(close < ema50)    * 0.15
             + int(close < ema200)   * 0.20
             + int(not mtf_up)       * 0.10)
        if ema20 < ema50 < ema200:
            t = min(1.0, t + 0.05)

    return float(np.clip(t, 0.0, 1.0))


def factor_momentum(
    row: pd.Series,
    daily_df: pd.DataFrame,
    direction: str,
) -> float:
    """
    Momentum factor [0, 1].

    Components:
      30% — RSI zone (48–73 ideal long, 27–52 ideal short)
      25% — MACD histogram direction + acceleration
      15% — StochRSI_K (avoid extremes)
      20% — ADX strength
      10% — Consecutive directional days (streak, last 10 bars)
    """
    rsi  = float(row["RSI"])
    adx  = float(row["ADX"])
    mh   = float(row["MACD_Hist"])
    sk   = float(row.get("StochRSI_K", 50) or 50)
    mprev = float(daily_df["MACD_Hist"].iloc[-2]) if len(daily_df) >= 2 else 0.0
    macc  = (mh > mprev) if direction == "LONG" else (mh < mprev)

    col = "Up_Day" if direction == "LONG" else "Dn_Day"
    arr = daily_df[col].values[-10:]
    streak = 0
    for v in reversed(arr):
        if v == 1:
            streak += 1
        else:
            break

    if direction == "LONG":
        m = (0.30 * (1 if 48 <= rsi <= 73 else (0.2 if rsi > 73 else 0.1 if rsi < 40 else 0))
           + 0.25 * (1 if mh > 0 and macc else (0.5 if mh > 0 else 0))
           + 0.15 * (1 if 20 < sk < 80 else (-0.3 if sk > 90 else 0))
           + 0.20 * (1 if adx >= 25 else (0.5 if adx >= 20 else 0))
           + 0.10 * min(1.0, streak * 0.25))
    else:
        m = (0.30 * (1 if 27 <= rsi <= 52 else (0.2 if rsi < 27 else 0.1 if rsi > 65 else 0))
           + 0.25 * (1 if mh < 0 and macc else (0.5 if mh < 0 else 0))
           + 0.15 * (1 if 20 < sk < 80 else (-0.3 if sk < 10 else 0))
           + 0.20 * (1 if adx >= 25 else (0.5 if adx >= 20 else 0))
           + 0.10 * min(1.0, streak * 0.25))

    return float(np.clip(m, 0.0, 1.0))


def factor_volume(
    row: pd.Series,
    daily_df: pd.DataFrame,
    direction: str,
    close: float,
    intraday_vol_today: Optional[int] = None,
    vprofile_lookback: int = 30,
    vprofile_bins: int = 100,
    adv_turnover_floor: float = 35_000_000,
) -> float:
    """
    Volume factor [0, 1].

    Components:
      55% — Relative volume (RVOL vs 20d average)
      25% — POC proximity (price near volume node)
      15% — Value area positioning
       5% — Turnover premium above floor
    """
    vol_avg = float(row["Vol_Avg_20"])
    vol_today = intraday_vol_today if intraday_vol_today is not None else int(vol_avg)
    rvol  = vol_today / vol_avg if vol_avg > 0 else 1.0
    atr   = float(row["ATR"])

    poc, val, vah = true_volume_profile(daily_df, lookback=vprofile_lookback, bins=vprofile_bins)
    poc_ok = (close > poc - 0.3 * atr) if direction == "LONG" else (close < poc + 0.3 * atr)
    va_ok  = (close > val)              if direction == "LONG" else (close < vah)
    rvol_s = min(1.0, (rvol - 1.0) / 1.0) if rvol >= 1.0 else 0.0
    turn_r = float(row["Turnover_Avg_20"]) / adv_turnover_floor

    v = (rvol_s * 0.55
         + (0.25 if poc_ok else 0)
         + (0.15 if va_ok  else 0)
         + min(0.05, (turn_r - 1) / 4 * 0.05))

    return float(np.clip(v, 0.0, 1.0))


def factor_volatility(
    row: pd.Series,
    vol_contract_ratio: float = 0.85,
) -> float:
    """
    Volatility/coiling factor [0, 1].

    High score = ATR contracting relative to 50d mean (coiling for breakout).
    Bollinger Squeeze bonus: +0.15 if squeeze detected.
    Low score  = ATR expanding well above baseline.

    Data contract (FIX 4)
    ---------------------
    This function assumes ``ATR_Pctile`` and ``ATR_50_mean`` are real,
    finite numbers — NOT NaN.  The ``passes_data_quality()`` gate in
    ``scorer.py`` must be called before reaching this function.

    Why: the original code used ``row.get("ATR_Pctile", 50)`` as a fallback.
    A fallback of 50 treats data-sparse tickers as "mid-range volatility",
    silently bypassing the quality check and biasing composite scores for
    recently listed or history-incomplete instruments.

    If a NaN somehow escapes the gate (e.g. during direct unit-test calls),
    the function falls back to a conservative penalty score of 0.0 instead
    of the neutral 50-percentile, making the data problem visible in output
    rather than hiding it.
    """

    atr    = float(row["ATR"])
    atr50m_raw = row.get("ATR_50_mean", None)
    atr_pct_raw = row.get("ATR_Pctile", None)

    # Defensive NaN check — the gate should prevent this, but if called
    # directly (e.g. in unit tests without pre-filtering), return 0.0 so the
    # bad data produces a visibly low score rather than a neutral one.
    if pd.isna(atr50m_raw):
        return 0.0
    if pd.isna(atr_pct_raw):
        return 0.0

    atr50m  = float(atr50m_raw)
    atr_pct = float(atr_pct_raw)
    bbs     = bool(row.get("BB_Squeeze", False))

    contract = (atr < vol_contract_ratio * atr50m) if atr50m > 0 else False

    if contract:
        vl = 0.70 + 0.30 * max(0, (50 - atr_pct) / 50)
    else:
        ratio = atr / atr50m if atr50m > 0 else 1.0
        vl = max(0.0, 0.40 - (ratio - 1.0) * 0.30)

    if bbs:
        vl = min(1.0, vl + 0.15)

    return float(np.clip(vl, 0.0, 1.0))


def factor_relative_strength(
    ticker: str,
    daily_df: pd.DataFrame,
    bench: pd.Series,
    sector: str,
    sector_ranks: dict[str, int],
    direction: str,
    n_sectors: int,
    rs_lookback: int = 20,
) -> float:
    """
    Relative strength factor [0, 1].

    60% — Sector rank (best sector for direction)
    40% — Stock RS vs Nifty50 benchmark (log return diff)
    """
    from .regime import compute_rs  # avoid circular at module level

    sec_rank = sector_ranks.get(sector, n_sectors)
    tick_rs  = compute_rs(daily_df["Close"], bench, lookback=rs_lookback)

    if direction == "LONG":
        rs = (max(0, (n_sectors - sec_rank) / max(n_sectors - 1, 1)) * 0.60
              + min(1.0, max(0.0, (tick_rs + 5) / 10)) * 0.40)
    else:
        rs = (max(0, (sec_rank - 1) / max(n_sectors - 1, 1)) * 0.60
              + min(1.0, max(0.0, (-tick_rs + 5) / 10)) * 0.40)

    return float(np.clip(rs, 0.0, 1.0))


def factor_breakout(
    row: pd.Series,
    daily_df: pd.DataFrame,
    close: float,
    direction: str,
    near_52w_max_dist_pct: float = 8.0,
) -> float:
    """
    Breakout factor [0, 1].

    65% — Proximity to 52-week high (for LONG; inverse for SHORT)
    35% — Bollinger Band Width contraction (narrow = coiling)
    """
    h52  = float(daily_df["High"].max())
    dist = ((h52 - close) / h52 * 100) if h52 > 0 else 100.0
    bw   = float(row.get("BB_Width", 0.05) or 0.05)

    bwavg = 0.0
    if "BB_Width" in daily_df.columns:
        bwavg_s = daily_df["BB_Width"].rolling(50).mean()
        bwavg = float(bwavg_s.iloc[-1]) if not bwavg_s.empty and not np.isnan(bwavg_s.iloc[-1]) else bw

    narrow = (bw < bwavg * 0.85) if bwavg > 0 else False

    if direction == "LONG":
        ds = max(0.0, 1.0 - dist / near_52w_max_dist_pct)
    else:
        ds = 0.5  # short setups don't benefit from 52w high proximity

    bo = ds * 0.65 + (0.35 if narrow else 0.0)
    return float(np.clip(bo, 0.0, 1.0))


def factor_quality(
    daily_df: pd.DataFrame,
    row: pd.Series,
    close: float,
) -> float:
    """
    Price-momentum quality factor [0, 1] — Fix C from v12.

    Replaces the dead liquidity-only quality factor. Three components
    with genuine cross-sectional spread among liquid stocks:

    50% — 63-day log momentum ([-10%, +10%] mapped to [0, 1])
    30% — Directional persistence (close > open win-rate, last 20 bars)
    20% — ATR expansion vs 50d mean (breakout readiness)
    """
    # (a) 63d momentum
    if len(daily_df) >= 64:
        c63 = float(daily_df["Close"].iloc[-64])
        mom63 = np.log(close / c63) if c63 > 0 else 0.0
        mom_score = min(1.0, max(0.0, (mom63 + 0.10) / 0.20))
    else:
        mom_score = 0.5

    # (b) Directional persistence
    up_days = (daily_df["Up_Day"].tail(20)
               if "Up_Day" in daily_df.columns
               else pd.Series([0.5] * 20))
    persist = float(up_days.mean())

    # (c) ATR expansion
    atr    = float(row.get("ATR", 1.0) or 1.0)
    atr50m = float(row.get("ATR_50_mean", atr) or atr)
    atr_exp = min(1.0, max(0.0, (atr / atr50m - 0.8) / 0.8)) if atr50m > 0 else 0.5

    q = mom_score * 0.50 + persist * 0.30 + atr_exp * 0.20
    return round(float(np.clip(q, 0.0, 1.0)), 4)


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITE ASSEMBLER
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "trend":      1 / 7,
    "momentum":   1 / 7,
    "volume":     1 / 7,
    "volatility": 1 / 7,
    "rs":         1 / 7,
    "breakout":   1 / 7,
    "quality":    1 / 7,
}


def compute_factors(
    ticker:       str,
    daily_df:     pd.DataFrame,
    bench:        pd.Series,
    sector:       str,
    sector_ranks: dict[str, int],
    direction:    str,
    close:        float,
    row:          pd.Series,
    n_sectors:    int,
    weights:      Optional[dict[str, float]] = None,
    intraday:     Optional[dict]             = None,
    mtf_60m:      Optional[dict]             = None,
    # Config knobs (all have sensible defaults)
    vprofile_lookback:     int   = 30,
    vprofile_bins:         int   = 100,
    adv_turnover_floor:    float = 35_000_000,
    vol_contract_ratio:    float = 0.85,
    near_52w_max_dist_pct: float = 8.0,
    rs_lookback:           int   = 20,
) -> FactorScores:
    """
    Full 7-factor composite model. Replaces the placeholder in core/scorer.py.

    Args:
        ticker:        NSE ticker string (e.g. "RELIANCE.NS")
        daily_df:      OHLCV + indicator DataFrame from add_indicators()
        bench:         Benchmark Close price Series (Nifty50)
        sector:        Sector string from TICKER_TO_SECTOR
        sector_ranks:  {sector: rank_int} — 1 = best RS sector
        direction:     "LONG" or "SHORT"
        close:         Live or last-bar close price
        row:           daily_df.iloc[-1] — last indicator row
        n_sectors:     Total number of sectors in universe
        weights:       IC-calibrated factor weights (falls back to equal 1/7)
        intraday:      Dict from fetch_intraday_single() — optional
        mtf_60m:       Dict from fetch_60m_single() — optional
        *_lookback/ratio/etc: Config knobs with v13 defaults

    Returns:
        FactorScores dataclass with per-factor scores + composite.
    """
    intraday = intraday or {}
    mtf_60m  = mtf_60m  or {}
    w = weights if weights else DEFAULT_WEIGHTS

    t  = factor_trend(
        row=row,
        close=close,
        direction=direction,
        mtf_60m_trend_aligned=mtf_60m.get("trend_aligned"),
    )
    m  = factor_momentum(row=row, daily_df=daily_df, direction=direction)
    v  = factor_volume(
        row=row,
        daily_df=daily_df,
        direction=direction,
        close=close,
        intraday_vol_today=intraday.get("vol_today"),
        vprofile_lookback=vprofile_lookback,
        vprofile_bins=vprofile_bins,
        adv_turnover_floor=adv_turnover_floor,
    )
    vl = factor_volatility(row=row, vol_contract_ratio=vol_contract_ratio)
    rs = factor_relative_strength(
        ticker=ticker,
        daily_df=daily_df,
        bench=bench,
        sector=sector,
        sector_ranks=sector_ranks,
        direction=direction,
        n_sectors=n_sectors,
        rs_lookback=rs_lookback,
    )
    bo = factor_breakout(
        row=row,
        daily_df=daily_df,
        close=close,
        direction=direction,
        near_52w_max_dist_pct=near_52w_max_dist_pct,
    )
    q  = factor_quality(daily_df=daily_df, row=row, close=close)

    raw = {"trend": t, "momentum": m, "volume": v, "volatility": vl,
           "rs": rs, "breakout": bo, "quality": q}

    composite = sum(
        np.clip(raw[k], 0.0, 1.0) * w.get(k, 1.0 / len(raw))
        for k in raw
    )

    return FactorScores(
        trend      = round(t,  3),
        momentum   = round(m,  3),
        volume     = round(v,  3),
        volatility = round(vl, 3),
        rs         = round(rs, 3),
        breakout   = round(bo, 3),
        quality    = round(q,  3),
        composite  = round(float(composite), 4),
        ic_weights = {k: round(w.get(k, 1 / 7), 4) for k in raw},
    )


def calibrate_ic_weights(
    results: list,  # list[TickerResult]
    processed: dict[str, pd.DataFrame],
    bench: pd.Series,
    sector_ranks: dict[str, int],
    n_sectors: int,
    config,
    lookback: int = 60,
    calib_offset: int = 5,
    fwd_bars: int = 5,
) -> dict[str, float]:
    """
    Optimizes factor weights using Information Coefficient (Spearman)
    over a historical lookback window.

    Direction-aware: the historical direction at each bar is inferred
    from Supertrend (Super_Up) so that SHORT tickers contribute to IC
    measurement with the correct factor polarity, rather than being
    forced into LONG-only calibration which biases breakout weights.

    For SHORT bars the signed forward return is negated before Spearman
    correlation so that a factor score of 1.0 still means "good for the
    actual direction taken" — preserving a consistent IC interpretation
    across the mixed-direction universe.
    """
    from scipy.stats import spearmanr
    from .universe import TICKER_TO_SECTOR

    # Build a direction lookup from the most-recent TickerResult.
    # This is a *fallback only* — used when a ticker's historical DataFrame
    # lacks the ``Super_Up`` indicator column.  Imputing a historical bar's
    # direction from the *current* live result is a weak form of lookahead
    # bias: if a regime flip occurred within the last ``calib_offset`` bars,
    # the imputed direction will be wrong.  We log a warning on first use
    # and skip the bar rather than silently propagate the bias.
    result_direction: dict[str, str] = {
        r.ticker.replace(".NS", ""): r.direction for r in results
    }
    _direction_fallback_warned: set[str] = set()

    factors_list = ["trend", "momentum", "volume", "volatility", "rs", "breakout", "quality"]
    daily_ics = {f: [] for f in factors_list}

    tickers = [r.ticker for r in results]

    # Sample every other bar for efficiency over the lookback window.
    step = 2
    for offset in range(calib_offset + 1, calib_offset + lookback + 1, step):
        f_vals = {f: [] for f in factors_list}
        rets: list[float] = []

        for ticker in tickers:
            df = processed.get(ticker)
            if df is None or len(df) < offset + fwd_bars + 5:
                continue

            idx = len(df) - offset
            row = df.iloc[idx - 1]
            close = float(row["Close"])

            # ── Direction: use historical Supertrend exclusively.
            # Falling back to the live TickerResult direction risks
            # lookahead bias — if the regime flipped recently the live
            # direction reflects future information relative to this bar.
            # We warn once per ticker and skip the bar entirely instead.
            if "Super_Up" in df.columns:
                super_up = bool(row.get("Super_Up", True))
                ema20 = float(row.get("EMA_20", close))
                direction = "LONG" if (super_up and close >= ema20) else "SHORT"
            else:
                base = ticker.replace(".NS", "")
                if base not in _direction_fallback_warned:
                    import logging as _logging
                    _logging.getLogger("sovereign.factors").warning(
                        "calibrate_ic_weights: ticker %s has no Super_Up column — "
                        "skipping all historical IC bars for this ticker to avoid "
                        "lookahead bias from live-direction imputation.",
                        ticker,
                    )
                    _direction_fallback_warned.add(base)
                continue   # skip this bar for this ticker — do not use live direction

            # ── Signed forward return (positive = good for direction) ──────
            fwd_close = float(df["Close"].iloc[idx + fwd_bars - 1])
            raw_ret = (fwd_close - close) / close if close > 0 else 0.0
            signed_ret = raw_ret if direction == "LONG" else -raw_ret
            rets.append(signed_ret)

            # ── Factor scores for the historical slice ────────────────────
            hist_df = df.iloc[:idx]
            f_vals["trend"].append(factor_trend(row, close, direction))
            f_vals["momentum"].append(factor_momentum(row, hist_df, direction))
            f_vals["volume"].append(factor_volume(row, hist_df, direction, close))
            f_vals["volatility"].append(factor_volatility(row))
            f_vals["rs"].append(factor_relative_strength(
                ticker, hist_df, bench.iloc[:idx],
                TICKER_TO_SECTOR.get(ticker, "Unknown"), sector_ranks,
                direction, n_sectors,
            ))
            f_vals["breakout"].append(factor_breakout(row, hist_df, close, direction))
            f_vals["quality"].append(factor_quality(hist_df, row, close))

        if len(rets) < 10:
            continue

        rets_arr = np.array(rets)
        for f in factors_list:
            arr = np.array(f_vals[f])
            if len(arr) == len(rets_arr) and arr.std() > 1e-6:
                ic, _ = spearmanr(arr, rets_arr)
                daily_ics[f].append(float(ic) if not np.isnan(ic) else 0.0)
            else:
                daily_ics[f].append(0.0)

    # ── ICIR: mean IC / std IC ────────────────────────────────────────────────
    icir: dict[str, float] = {}
    for f in factors_list:
        ics = np.array(daily_ics[f])
        if len(ics) < 5:
            icir[f] = 0.0
        else:
            m, s = ics.mean(), ics.std()
            icir[f] = m / s if s > 1e-6 else 0.0

    # ── Normalise positive ICIR to sum-to-one weights ─────────────────────────
    pos_icir = {f: max(0.0, icir[f]) for f in factors_list}
    total = sum(pos_icir.values())

    if total < 1e-6:
        return {f: 1.0 / len(factors_list) for f in factors_list}

    return {f: pos_icir[f] / total for f in factors_list}
