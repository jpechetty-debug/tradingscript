"""
core/scorer.py
==============
Individual ticker evaluation — v14 modular rewrite.

Replaces the placeholder (composite=0.6, prob_win=0.55) with the full
7-factor signal model from screener.py v13. Depends only on other
core/ modules; zero imports from the monolith screener.py.

Public API:
    score_ticker(...) -> Optional[TickerResult]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import expit as _sigmoid

from .config import SystemConfig
from .factors import FactorScores, compute_factors, true_volume_profile
from .portfolio import compute_targets, calculate_kelly_size
from .regime import MarketRegime, compute_rs
from .universe import TICKER_TO_SECTOR, N_SECTORS

log = logging.getLogger("sovereign.scorer")


# ─────────────────────────────────────────────────────────────────────────────
# RESULT DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TickerResult:
    # Identity
    ticker:    str
    sector:    str
    direction: str

    # Price & market context
    close:      float
    change_pct: float

    # Factor model
    factors:   FactorScores
    composite: float

    # Probability & expectancy
    prob_win:    float
    expectancy_r: float
    sharpe_rank:  float

    # Trade levels
    entry:     float
    stop:      float
    t1:        float
    t2:        float
    breakeven: float
    trail_stop:    float
    time_stop_bars: int

    # Position sizing
    shares:    int
    risk_inr:  float
    rr_t1:     float
    kelly_f:   float
    kurt_correction: float
    excess_kurtosis: float

    # Indicator snapshot (for display / Telegram)
    rsi:         float
    stochrsi_k:  float
    rvol:        float
    adx:         float
    super_up:    bool
    macd_hist:   float
    atr_pctile:  float
    vol_contract: bool
    rs_vs_nifty: float
    near_52w:    bool
    ema200_aligned: bool
    mtf_aligned:    bool
    consec_days:    int

    # Volume profile
    poc: float
    val: float
    vah: float

    # Regime / session context
    regime:  str
    session: str

    # Human-readable signal reasons
    reasons: list[str] = field(default_factory=list)

    def display_score(self) -> int:
        return int(self.composite * 100)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "factors"}
        d.update(self.factors.as_dict())
        return d


# ─────────────────────────────────────────────────────────────────────────────
# PLATT PROBABILITY CONVERSION
# ─────────────────────────────────────────────────────────────────────────────

def composite_to_prob(composite: float, platt_a: float, platt_b: float) -> float:
    """Sigmoid probability from composite score using Platt A/B params."""
    return float(_sigmoid(platt_a * composite + platt_b))


def calibrate_platt(composites: list[float], outcomes: list[int]) -> tuple[float, float]:
    """
    Fits Platt A and B parameters using MLE.
    Optimises: P(y=1|x) = 1 / (1 + exp(A*x + B))
    """
    from scipy.optimize import minimize
    def nll(ab):
        p = _sigmoid(ab[0] * np.array(composites) + ab[1])
        p = np.clip(p, 1e-7, 1-1e-7)
        return -np.mean(np.array(outcomes)*np.log(p) + (1-np.array(outcomes))*np.log(1-p))
    
    # Init from a reasonable starting point (negative slope)
    res = minimize(nll, [-4.0, 2.0], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


# ─────────────────────────────────────────────────────────────────────────────
# LIQUIDITY GATE
# ─────────────────────────────────────────────────────────────────────────────

def passes_liquidity(row: pd.Series, config: SystemConfig) -> tuple[bool, str]:
    if row["Vol_Avg_20"] < config.ADV_SHARE_FLOOR:
        return False, f"Vol {int(row['Vol_Avg_20']):,} < {config.ADV_SHARE_FLOOR:,}"
    if row["Turnover_Avg_20"] < config.ADV_TURNOVER_FLOOR:
        return False, f"Turnover < ₹{config.ADV_TURNOVER_FLOOR / 1e7:.0f}cr"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# TRADE MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def compute_trade_management(
    direction: str,
    entry: float,
    atr: float,
    atr_pctile: float,
) -> tuple[float, int]:
    """Returns (trail_stop, time_stop_bars)."""
    trail_mult     = 1.0 + (atr_pctile / 100) * 1.0
    trail_stop     = (round(entry - trail_mult * atr, 2)
                      if direction == "LONG"
                      else round(entry + trail_mult * atr, 2))
    time_stop_bars = 4 if atr_pctile < 30 else (7 if atr_pctile < 60 else 12)
    return trail_stop, time_stop_bars


# ─────────────────────────────────────────────────────────────────────────────
# SESSION MULTIPLIER
# ─────────────────────────────────────────────────────────────────────────────

_SESSION_MULT = {
    "CLOSING_TREND": 1.05,
    "MIDDAY_CHOP":   0.92,
    "OPENING_RANGE": 1.00,
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORER
# ─────────────────────────────────────────────────────────────────────────────

def score_ticker(
    ticker:       str,
    daily_df:     pd.DataFrame,
    bench:        pd.Series,
    sector_ranks: dict[str, int],
    sector_rs:    dict[str, float],
    session:      str,
    regime:       MarketRegime,
    config:       SystemConfig,
    intraday:     Optional[dict] = None,
    mtf_60m:      Optional[dict] = None,
    factor_weights: Optional[dict[str, float]] = None,
    debug:        bool = False,
) -> Optional[TickerResult]:
    """
    Full ticker evaluation pipeline. Returns None if the ticker does not
    pass any gate (liquidity, direction, regime, EMA200, probability, E(R)).

    Args:
        ticker:         NSE ticker (e.g. "RELIANCE.NS")
        daily_df:       OHLCV + indicators from add_indicators()
        bench:          Benchmark Close series (Nifty50)
        sector_ranks:   {sector: rank_int} best=1
        sector_rs:      {sector: rs_float}
        session:        "OPENING_RANGE" | "MIDDAY_CHOP" | "CLOSING_TREND"
        regime:         MarketRegime dataclass from classify_regime()
        config:         SystemConfig instance
        intraday:       Optional dict from fetch_intraday_single()
        mtf_60m:        Optional dict from fetch_60m_single()
        factor_weights: IC-calibrated weights — falls back to equal 1/7
        debug:          Print rejection reason to stdout

    Returns:
        TickerResult or None
    """
    if daily_df.empty:
        return None

    intraday = intraday or {}
    mtf_60m  = mtf_60m  or {}
    row = daily_df.iloc[-1]

    # ── 1. Liquidity gate ────────────────────────────────────────────────────
    ok, msg = passes_liquidity(row, config)
    if not ok:
        if debug:
            log.debug("%s: LIQUIDITY — %s", ticker, msg)
        return None

    # ── 2. Direction ─────────────────────────────────────────────────────────
    live_price  = intraday.get("live_price", 0.0)
    close       = live_price if live_price > 0 else float(row["Close"])
    above_vwap  = intraday.get("above_vwap", close > float(row["EMA_20"]))

    super_up = bool(row["Super_Up"])
    ema20    = float(row["EMA_20"])
    ema200   = float(row["EMA_200"])

    is_bull = super_up and close > ema20 and above_vwap
    is_bear = (not super_up) and close < ema20 and (not above_vwap)

    if not is_bull and not is_bear:
        if debug:
            log.debug("%s: NEUTRAL — no directional bias", ticker)
        return None

    direction = "LONG" if is_bull else "SHORT"

    # ── 3. Regime gate ───────────────────────────────────────────────────────
    if direction == "LONG"  and not regime.allows_long():
        if debug: log.debug("%s: regime blocks LONG (%s)", ticker, regime.regime)
        return None
    if direction == "SHORT" and not regime.allows_short():
        if debug: log.debug("%s: regime blocks SHORT (%s)", ticker, regime.regime)
        return None

    # ── 4. EMA-200 structural filter ─────────────────────────────────────────
    if config.USE_EMA200_FILTER:
        above200 = close > ema200
        if direction == "LONG"  and not above200:
            if debug: log.debug("%s: EMA-200 VETO (LONG below 200)", ticker)
            return None
        if direction == "SHORT" and above200:
            if debug: log.debug("%s: EMA-200 VETO (SHORT above 200)", ticker)
            return None

    # ── 5. Factor model ──────────────────────────────────────────────────────
    sector = TICKER_TO_SECTOR.get(ticker, "")
    factors = compute_factors(
        ticker=ticker,
        daily_df=daily_df,
        bench=bench,
        sector=sector,
        sector_ranks=sector_ranks,
        direction=direction,
        close=close,
        row=row,
        n_sectors=N_SECTORS,
        weights=factor_weights,
        intraday=intraday,
        mtf_60m=mtf_60m,
        vprofile_lookback=config.VPROFILE_LOOKBACK,
        vprofile_bins=config.VPROFILE_BINS,
        adv_turnover_floor=config.ADV_TURNOVER_FLOOR,
        vol_contract_ratio=config.VOL_CONTRACT_RATIO,
        near_52w_max_dist_pct=config.NEAR_52W_MAX_DIST_PCT,
        rs_lookback=config.RS_LOOKBACK,
    )

    # ── 6. Session + regime composite adjustment ──────────────────────────────
    sess_mult = _SESSION_MULT.get(session, 1.0)
    if regime.regime == "RANGE":
        sess_mult *= 0.88
    adj_composite = float(np.clip(factors.composite * sess_mult, 0.0, 1.0))

    # ── 7. Platt probability ─────────────────────────────────────────────────
    prob_win = composite_to_prob(adj_composite, config.PLATT_A, config.PLATT_B)

    # ── 8. Targets & expectancy ───────────────────────────────────────────────
    atr = float(row["ATR"])
    poc, val, vah = true_volume_profile(
        daily_df,
        lookback=config.VPROFILE_LOOKBACK,
        bins=config.VPROFILE_BINS,
    )

    targets = compute_targets(direction, close, atr, config)

    # Use value-area T1 if RR qualifies
    if config.USE_VALUE_AREA_RR:
        sl_dist = config.STOP_ATR_MULT * atr
        if direction == "LONG":
            rr_va = (vah - close) / sl_dist if sl_dist > 0 else 0
            if rr_va >= config.VA_MIN_RR:
                targets = targets.__class__(
                    stop=targets.stop,
                    t1=round(vah, 2),
                    t2=targets.t2,
                    rr=round(rr_va, 2),
                )
        else:
            rr_va = (close - val) / sl_dist if sl_dist > 0 else 0
            if rr_va >= config.VA_MIN_RR:
                targets = targets.__class__(
                    stop=targets.stop,
                    t1=round(val, 2),
                    t2=targets.t2,
                    rr=round(rr_va, 2),
                )

    exp_r = round(prob_win * targets.rr - (1 - prob_win) * 1.0, 3)

    # ── 9. Probability & expectancy gates ────────────────────────────────────
    if prob_win < config.MIN_PROB_WIN:
        if debug: log.debug("%s: prob %.2f < gate %.2f", ticker, prob_win, config.MIN_PROB_WIN)
        return None
    if exp_r < config.MIN_EXPECTANCY_R:
        if debug: log.debug("%s: E(R) %.3f < gate %.3f", ticker, exp_r, config.MIN_EXPECTANCY_R)
        return None

    # ── 10. Kelly position sizing ─────────────────────────────────────────────
    shares, risk_inr, kelly_f, kurt_corr = calculate_kelly_size(
        entry=close,
        stop=targets.stop,
        prob_win=prob_win,
        rr=targets.rr,
        daily_df=daily_df,
        config=config,
    )
    from .portfolio import _ticker_excess_kurtosis
    excess_kurt = _ticker_excess_kurtosis(daily_df, config)

    # ── 11. Auxiliary metrics ─────────────────────────────────────────────────
    rvol_20      = float(row.get("RVol_20", 0.20) or 0.20)
    sharpe_rank  = exp_r / rvol_20 if rvol_20 > 0 else exp_r
    sl_dist      = abs(close - targets.stop)
    breakeven    = (round(close + sl_dist, 2)
                    if direction == "LONG"
                    else round(close - sl_dist, 2))

    atr_pctile   = float(row.get("ATR_Pctile", 50) or 50)
    trail_stop, time_stop = compute_trade_management(direction, close, atr, atr_pctile)

    rsi    = float(row["RSI"])
    adx    = float(row["ADX"])
    mh     = float(row["MACD_Hist"])
    sk     = float(row.get("StochRSI_K", 50) or 50)
    vol_avg = float(row["Vol_Avg_20"])
    vol_today = intraday.get("vol_today", int(vol_avg))
    rvol   = round(vol_today / vol_avg, 2) if vol_avg > 0 else 1.0
    atr50m = float(row.get("ATR_50_mean", atr) or atr)
    vol_c  = (atr < config.VOL_CONTRACT_RATIO * atr50m) if atr50m > 0 else False

    h52    = float(daily_df["High"].max())
    dist52 = ((h52 - close) / h52 * 100) if h52 > 0 else 100.0
    ema50  = float(row["EMA_50"])
    mtf_full = (ema20 > ema50 > ema200) if direction == "LONG" else (ema20 < ema50 < ema200)
    ema200_al = (close > ema200) if direction == "LONG" else (close < ema200)

    col = "Up_Day" if direction == "LONG" else "Dn_Day"
    streak = 0
    for v in reversed(daily_df[col].values[-10:]):
        if v == 1: streak += 1
        else: break

    tick_rs  = compute_rs(daily_df["Close"], bench, lookback=config.RS_LOOKBACK)
    sec_rank = sector_ranks.get(sector, N_SECTORS)
    sec_rs   = sector_rs.get(sector, 0.0)

    change_pct = ((close - float(daily_df["Open"].iloc[-1]))
                  / float(daily_df["Open"].iloc[-1])) * 100

    # ── 12. Signal reasons (human-readable) ───────────────────────────────────
    reasons: list[str] = []
    if factors.trend      > 0.7: reasons.append("Trend✅")
    if factors.momentum   > 0.6: reasons.append(f"Mom✅RSI{rsi:.0f}")
    if factors.volume     > 0.6: reasons.append(f"Vol✅×{rvol:.1f}")
    if factors.volatility > 0.6: reasons.append("Coiled")
    if factors.rs         > 0.6: reasons.append(f"RS✅#{sec_rank}")
    if factors.quality    > 0.6: reasons.append("Qual✅")
    if adx >= 25:                 reasons.append(f"ADX{adx:.0f}")
    if mtf_full:                  reasons.append("MTF✅")
    reasons.append(f"Regime:{regime.regime}")
    reasons.append(f"RR:{targets.rr:.1f}x")
    reasons.append(f"Kurt:k={excess_kurt:.1f}->{kurt_corr:.0%}Kelly")

    return TickerResult(
        ticker=ticker.replace(".NS", ""),
        sector=sector,
        direction=direction,
        close=round(close, 2),
        change_pct=round(change_pct, 2),
        factors=FactorScores(
            trend=factors.trend, momentum=factors.momentum,
            volume=factors.volume, volatility=factors.volatility,
            rs=factors.rs, breakout=factors.breakout,
            quality=factors.quality,
            composite=round(adj_composite, 4),
            ic_weights=factors.ic_weights,
        ),
        composite=round(adj_composite, 4),
        prob_win=round(prob_win, 3),
        expectancy_r=round(exp_r, 3),
        sharpe_rank=round(sharpe_rank, 4),
        entry=close,
        stop=targets.stop,
        t1=targets.t1,
        t2=targets.t2,
        breakeven=breakeven,
        trail_stop=trail_stop,
        time_stop_bars=time_stop,
        shares=shares,
        risk_inr=risk_inr,
        rr_t1=targets.rr,
        kelly_f=round(kelly_f, 5),
        kurt_correction=round(kurt_corr, 4),
        excess_kurtosis=round(excess_kurt, 2),
        rsi=round(rsi, 1),
        stochrsi_k=round(sk, 1),
        rvol=rvol,
        adx=round(adx, 1),
        super_up=super_up,
        macd_hist=round(mh, 4),
        atr_pctile=round(atr_pctile, 1),
        vol_contract=vol_c,
        rs_vs_nifty=tick_rs,
        near_52w=(dist52 <= config.NEAR_52W_MAX_DIST_PCT),
        ema200_aligned=ema200_al,
        mtf_aligned=mtf_full,
        consec_days=streak,
        poc=round(poc, 2),
        val=round(val, 2),
        vah=round(vah, 2),
        regime=regime.regime,
        session=session,
        reasons=reasons,
    )
