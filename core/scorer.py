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

from datetime import datetime
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.special import expit as _sigmoid
from scipy.stats import rankdata

from .config import SystemConfig, IST
from .factors import FactorScores, compute_factors, true_volume_profile, get_regime_factor_weights, DEFAULT_WEIGHTS
from .portfolio import compute_targets, calculate_kelly_size
from .regime import MarketRegime, MarketRegimeType, compute_rs
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

    # Trade horizon & action (v14.2)
    trade_horizon: str = "SWING"   # "INTRADAY" | "SWING"
    action:        str = ""        # "BUY" | "SELL"

    # Human-readable signal reasons
    reasons: list[str] = field(default_factory=list)
    is_watchlist: bool = False
    is_held: bool = False

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
    return float(_sigmoid(-(platt_a * composite + platt_b)))


def calibrate_platt(
    composites: list[float],
    outcomes:   list[int],
    calib_offset: int = 60,
) -> tuple[float, float]:
    """
    Fit Platt A and B parameters on a **held-out** validation window.

    FIX 3 — out-of-sample calibration
    -----------------------------------
    The original implementation trained on the *same* composites that the
    live system uses to generate signals.  This is in-sample calibration:
    the sigmoid is tuned to the training residuals, producing
    over-confident probabilities (P(win) consistently > actual win-rate)
    and inflating position sizes.

    Fix: split the data at ``calib_offset`` bars from the end.

    * Training window  : composites[:-calib_offset]  / outcomes[:-calib_offset]
      Used ONLY to fit A and B via MLE.
    * Validation window: composites[-calib_offset:]  / outcomes[-calib_offset:]
      Never seen during fitting; used to log calibration quality (Brier
      score and mean predicted prob vs actual win-rate) so you can monitor
      drift over time.

    The calib_offset default (60 bars) mirrors ``IC_CALIB_OFFSET`` in
    factors.py — keep them in sync if you change either.

    Parameters
    ----------
    composites
        List of composite factor scores (floats in [0, 1]) in chronological
        order, oldest first.
    outcomes
        List of binary trade outcomes (1 = win, 0 = loss), same order.
    calib_offset
        Number of most-recent samples to hold out.  Must be < len(composites).
        Set to 0 to replicate the old in-sample behaviour (not recommended).

    Returns
    -------
    (A, B) — Platt parameters for ``composite_to_prob()``.

    Raises
    ------
    ValueError
        If there are fewer than ``calib_offset + 20`` samples — not enough
        data to calibrate reliably.
    """
    from scipy.optimize import minimize

    n = len(composites)
    min_required = calib_offset + 20
    if n < min_required:
        raise ValueError(
            f"calibrate_platt: need >= {min_required} samples to hold out "
            f"{calib_offset} for validation, got {n}.  "
            "Collect more trade history or reduce calib_offset."
        )

    # ── Split ─────────────────────────────────────────────────────────────────
    if calib_offset > 0:
        train_x = np.array(composites[:-calib_offset])
        train_y = np.array(outcomes[:-calib_offset])
        val_x   = np.array(composites[-calib_offset:])
        val_y   = np.array(outcomes[-calib_offset:])
    else:
        # calib_offset=0 → in-sample (legacy); caller's explicit choice.
        train_x = np.array(composites)
        train_y = np.array(outcomes)
        val_x   = train_x
        val_y   = train_y

    # ── MLE fit on training window ────────────────────────────────────────────
    def nll(ab: np.ndarray) -> float:
        p = _sigmoid(-(ab[0] * train_x + ab[1]))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -float(np.mean(train_y * np.log(p) + (1 - train_y) * np.log(1 - p)))

    bounds = [(-15.0, 15.0), (-10.0, 10.0)]
    res = minimize(nll, [-4.0, 2.0], method="L-BFGS-B", bounds=bounds)
    a, b = float(res.x[0]), float(res.x[1])

    # ── Validation diagnostics (logged, not used for fitting) ─────────────────
    val_p       = _sigmoid(-(a * val_x + b))
    brier       = float(np.mean((val_p - val_y) ** 2))
    mean_pred   = float(val_p.mean())
    actual_wr   = float(val_y.mean())
    cal_err     = mean_pred - actual_wr          # positive → over-confident

    log.info(
        "Platt calibration (train=%d, val=%d): A=%.4f B=%.4f | "
        "val Brier=%.4f  pred_prob=%.3f  actual_wr=%.3f  cal_err=%+.3f%s",
        len(train_x), len(val_x), a, b,
        brier, mean_pred, actual_wr, cal_err,
        "  [OVER-CONFIDENT]" if cal_err > 0.05 else
        "  [UNDER-CONFIDENT]" if cal_err < -0.05 else "",
    )

    return a, b


# ─────────────────────────────────────────────────────────────────────────────
# LIQUIDITY GATE
# ─────────────────────────────────────────────────────────────────────────────

def passes_liquidity(row: pd.Series, config: SystemConfig) -> tuple[bool, str]:
    if row["Vol_Avg_20"] < config.ADV_SHARE_FLOOR:
        return False, f"Vol {int(row['Vol_Avg_20']):,} < {config.ADV_SHARE_FLOOR:,}"
    if row["Turnover_Avg_20"] < config.ADV_TURNOVER_FLOOR:
        return False, f"Turnover < ₹{config.ADV_TURNOVER_FLOOR / 1e7:.0f}cr"
    return True, ""


def passes_data_quality(row: pd.Series, ticker: str) -> tuple[bool, str]:
    """
    Reject tickers whose indicator columns contain NaN/NA.
    These indicate insufficient history or a pipeline failure —
    scoring them produces misleading composites.
    """
    required = ["ATR", "ATR_50_mean", "ATR_Pctile", "EMA_20", "EMA_50",
                "EMA_200", "RSI", "ADX", "MACD_Hist", "Vol_Avg_20"]
    for col in required:
        val = row.get(col)
        if val is None or pd.isna(val):
            return False, f"{col} is NaN — insufficient history"
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

FACTOR_NAMES = ("trend", "momentum", "volume", "volatility", "rs", "breakout", "quality")


@dataclass
class CandidateContext:
    ticker: str
    sector: str
    direction: str
    close: float
    daily_df: pd.DataFrame
    bench: pd.Series
    sector_ranks: dict[str, int]
    sector_rs: dict[str, float]
    session: str
    regime: MarketRegime
    trade_horizon: str
    factors: FactorScores
    capital_fraction: float
    intraday: dict
    row: pd.Series


def apply_cohort_factor_ranking(
    candidates: list[Any],
    cohort_rank_weight: float = 0.40,
    cohort_min_obs: int = 10,
) -> list[Any]:
    """
    Apply cross-sectional cohort factor ranking to directional cohorts.
    For each directional cohort (LONG and SHORT) with >= cohort_min_obs candidates,
    computes average percentile rank for each of the 7 factors using:
        rank_pct = (rankdata(raw_vals, method='average') - 1.0) / (N - 1.0)
    Blends:
        factor_blended = (1.0 - cohort_rank_weight) * factor_raw + cohort_rank_weight * rank_pct
    Recomputes composite from blended factors and updates each candidate.
    """
    if not candidates:
        return candidates

    cohort_rank_weight = float(np.clip(cohort_rank_weight, 0.0, 1.0))
    if cohort_rank_weight <= 0.0:
        return candidates

    def _get_dir(c: Any) -> str:
        if isinstance(c, dict):
            return str(c.get("direction", "LONG")).upper()
        return str(getattr(c, "direction", "LONG")).upper()

    def _get_factors(c: Any) -> Optional[FactorScores]:
        if isinstance(c, dict):
            return c.get("factors")
        return getattr(c, "factors", None)

    cohort_long = [c for c in candidates if _get_dir(c) == "LONG"]
    cohort_short = [c for c in candidates if _get_dir(c) == "SHORT"]

    for cohort in (cohort_long, cohort_short):
        n = len(cohort)
        if n < cohort_min_obs or n <= 1:
            continue

        factor_vals: dict[str, list[float]] = {f: [] for f in FACTOR_NAMES}
        for c in cohort:
            fs = _get_factors(c)
            for f in FACTOR_NAMES:
                val = float(getattr(fs, f, 0.5) if fs is not None else 0.5)
                factor_vals[f].append(val)

        factor_rank_pcts: dict[str, np.ndarray] = {}
        for f in FACTOR_NAMES:
            raw_arr = np.array(factor_vals[f], dtype=float)
            ranks = rankdata(raw_arr, method="average")
            rank_pcts = (ranks - 1.0) / (n - 1.0)
            factor_rank_pcts[f] = rank_pcts

        for idx, c in enumerate(cohort):
            fs = _get_factors(c)
            weights = fs.ic_weights if (fs is not None and fs.ic_weights) else dict(DEFAULT_WEIGHTS)
            w_sum = sum(weights.get(f, 1 / 7) for f in FACTOR_NAMES)
            norm_weights = {f: weights.get(f, 1 / 7) / w_sum for f in FACTOR_NAMES}

            blended_scores: dict[str, float] = {}
            for f in FACTOR_NAMES:
                raw_v = factor_vals[f][idx]
                rank_p = float(factor_rank_pcts[f][idx])
                b_val = (1.0 - cohort_rank_weight) * raw_v + cohort_rank_weight * rank_p
                blended_scores[f] = float(np.clip(b_val, 0.0, 1.0))

            new_composite = sum(norm_weights[f] * blended_scores[f] for f in FACTOR_NAMES)

            new_fs = FactorScores(
                trend=round(blended_scores["trend"], 4),
                momentum=round(blended_scores["momentum"], 4),
                volume=round(blended_scores["volume"], 4),
                volatility=round(blended_scores["volatility"], 4),
                rs=round(blended_scores["rs"], 4),
                breakout=round(blended_scores["breakout"], 4),
                quality=round(blended_scores["quality"], 4),
                composite=round(new_composite, 4),
                ic_weights=weights,
            )

            if isinstance(c, dict):
                c["factors"] = new_fs
                if "composite" in c:
                    c["composite"] = round(new_composite, 4)
            else:
                c.factors = new_fs
                if hasattr(c, "composite"):
                    c.composite = round(new_composite, 4)

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# TRADE HORIZON CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _classify_trade_horizon(
    *,
    config: SystemConfig,
    direction: str,
    session: str,
    adx: float,
    intraday: dict,
) -> str:
    """
    Classify trade horizon.  Returns ``'SWING'`` when the intraday data
    pipeline is not connected (``INTRADAY_ENABLED=False``).

    When intraday *is* enabled, classification uses session, ADX, and
    direction to decide between INTRADAY and SWING.
    """
    if not getattr(config, "INTRADAY_ENABLED", False):
        return "SWING"

    # Future: require intraday["has_vwap"], intraday["has_60m_trend"], etc.
    short_intraday = getattr(config, "SHORT_IS_INTRADAY_ONLY", True)
    if direction == "SHORT" and short_intraday:
        return "INTRADAY"
    if session == "OPENING_RANGE" or (adx < 20 and session != "CLOSING_TREND"):
        return "INTRADAY"
    return "SWING"


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORER
# ─────────────────────────────────────────────────────────────────────────────

def score_candidate_pass1(
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
    capital_fraction: float = 1.0,
    debug:        bool = False,
    no_intraday:  bool = False,
    force_score:  bool = False,
    now:          Optional[datetime] = None,
    is_open_position: bool = False,
    held_direction: Optional[str] = None,
) -> Optional[CandidateContext]:
    """
    Pass 1 of ticker scoring pipeline:
    Evaluates liquidity, data quality, directional bias, regime veto, EMA200 filter,
    and calculates raw FactorScores. Returns CandidateContext or None.
    """
    if daily_df.empty:
        return None

    intraday = intraday or {}
    mtf_60m  = mtf_60m  or {}
    row = daily_df.iloc[-1]

    # ── 1. Liquidity gate ────────────────────────────────────────────────────
    if not is_open_position:
        ok, msg = passes_liquidity(row, config)
        if not ok:
            if debug:
                log.debug("%s: LIQUIDITY — %s", ticker, msg)
            return None

    # ── 1b. Data-quality gate (FIX 4) ────────────────────────────────────────
    # Reject tickers with NaN ATR_Pctile or ATR_50_mean.  These arise when
    # a ticker has fewer than 50 bars of history.  Without this gate,
    # factor_volatility silently treats them as mid-range (fallback=50),
    # which hides data-sparse tickers and biases the composite score.
    ok, msg = passes_data_quality(row, ticker)
    if not ok:
        if debug:
            log.debug("%s: DATA_QUALITY — %s", ticker, msg)
        return None

    # ── 2. Direction ─────────────────────────────────────────────────────────
    live_price  = intraday.get("live_price", 0.0)
    close       = live_price if live_price > 0 else float(row["Close"])
    has_intraday_vwap = "above_vwap" in intraday
    above_vwap  = intraday.get("above_vwap", close > float(row["EMA_20"]))

    super_up = bool(row["Super_Up"])
    ema20    = float(row["EMA_20"])
    ema200   = float(row["EMA_200"])

    if has_intraday_vwap:
        bull_signals = (1 if super_up else 0) + (1 if close > ema20 else 0) + (1 if above_vwap else 0)
        bear_signals = (1 if not super_up else 0) + (1 if close < ema20 else 0) + (1 if not above_vwap else 0)
        is_bull = bull_signals >= 2
        is_bear = bear_signals >= 2
    else:
        is_bull = super_up and close > ema20
        is_bear = (not super_up) and close < ema20

    if force_score:
        direction = "LONG"
    elif is_open_position:
        # For held positions, retain the established direction unless directional bias is strong
        if held_direction:
            direction = held_direction
        elif is_bull:
            direction = "LONG"
        elif is_bear:
            direction = "SHORT"
        else:
            direction = "LONG"
    elif not is_bull and not is_bear:
        if debug:
            log.debug("%s: NEUTRAL — no directional bias", ticker)
        return None
    else:
        direction = "LONG" if is_bull else "SHORT"

    # ── 3. Regime gate ───────────────────────────────────────────────────────
    if not force_score:
        if direction == "LONG":
            if not regime.allows_long():
                # In confirmed RANGE, allow mean-reversion LONG if setup is not overbought
                rsi_val = float(row.get("RSI", 50))
                if regime.allows_mean_reversion() and rsi_val <= 55:
                    pass
                else:
                    if is_open_position:
                        log.info("%s: Held position exited via regime structural veto (%s blocks LONG)", ticker, regime.regime)
                    elif debug:
                        log.debug("%s: regime blocks LONG (%s)", ticker, regime.regime)
                    return None
        if direction == "SHORT":
            if not regime.allows_short():
                # In confirmed RANGE, allow mean-reversion SHORT if setup is not oversold
                rsi_val = float(row.get("RSI", 50))
                if regime.allows_mean_reversion() and rsi_val >= 45:
                    pass
                else:
                    if is_open_position:
                        log.info("%s: Held position exited via regime structural veto (%s blocks SHORT)", ticker, regime.regime)
                    elif debug:
                        log.debug("%s: regime blocks SHORT (%s)", ticker, regime.regime)
                    return None

    # ── 4. EMA-200 structural filter ─────────────────────────────────────────
    if config.USE_EMA200_FILTER:
        above200 = close > ema200
        if direction == "LONG"  and not above200:
            if is_open_position:
                log.info("%s: Held position exited via EMA-200 structural breakdown (LONG below 200)", ticker)
            elif debug:
                log.debug("%s: EMA-200 VETO (LONG below 200)", ticker)
            return None
        if direction == "SHORT" and above200:
            if is_open_position:
                log.info("%s: Held position exited via EMA-200 structural breakdown (SHORT above 200)", ticker)
            elif debug:
                log.debug("%s: EMA-200 VETO (SHORT above 200)", ticker)
            return None

    # ── 5a. Trade horizon classification & Cutoff Gate ───────────────────────
    adx_now = float(row.get("ADX", 0) or 0)
    trade_horizon = _classify_trade_horizon(
        config=config,
        direction=direction,
        session=session,
        adx=adx_now,
        intraday=intraday,
    )

    # Suppress intraday candidates when --no-intraday is passed or when
    # INTRADAY_ENABLED is False (which already forces SWING above, but
    # this is a safety net for any future code path).
    if no_intraday and trade_horizon == "INTRADAY":
        return None

    # Intraday Hard Entry Freeze: Veto new MIS entries past INTRADAY_ENTRY_CUTOFF (14:30)
    # because broker auto-square-off occurs at 15:15 (less than 45 min runway).
    # Existing held positions are exempt since they are already active.
    if trade_horizon == "INTRADAY" and not force_score and not is_open_position:
        current_dt = now.astimezone(IST) if now is not None else datetime.now(IST)
        current_time = current_dt.time()
        cutoff_time_str = getattr(config, "INTRADAY_ENTRY_CUTOFF", "14:30")
        t_cutoff = datetime.strptime(cutoff_time_str, "%H:%M").time()
        if session == "CLOSING_TREND" and current_time >= t_cutoff:
            if debug:
                log.debug("%s: INTRADAY_CUTOFF veto (time %s >= cutoff %s)", ticker, current_time, t_cutoff)
            return None

    # ── 5b. Regime-conditional weights (fallback when IC not calibrated) ──────
    effective_weights = factor_weights
    if not effective_weights:
        effective_weights = get_regime_factor_weights(regime.label)

    # ── 5c. Factor model ─────────────────────────────────────────────────────
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
        weights=effective_weights,
        intraday=intraday,
        mtf_60m=mtf_60m,
        vprofile_lookback=config.VPROFILE_LOOKBACK,
        vprofile_bins=config.VPROFILE_BINS,
        adv_turnover_floor=config.ADV_TURNOVER_FLOOR,
        vol_contract_ratio=config.VOL_CONTRACT_RATIO,
        near_52w_max_dist_pct=config.NEAR_52W_MAX_DIST_PCT,
        rs_lookback=config.RS_LOOKBACK,
    )

    return CandidateContext(
        ticker=ticker,
        sector=sector,
        direction=direction,
        close=close,
        daily_df=daily_df,
        bench=bench,
        sector_ranks=sector_ranks,
        sector_rs=sector_rs,
        session=session,
        regime=regime,
        trade_horizon=trade_horizon,
        factors=factors,
        capital_fraction=capital_fraction,
        intraday=intraday,
        row=row,
    )


def score_candidate_pass2(
    candidate: CandidateContext,
    config: SystemConfig,
    is_open_position: bool = False,
    allow_watchlist: bool = False,
    debug: bool = False,
    no_intraday: bool = False,
    now: Optional[datetime] = None,
) -> Optional[TickerResult]:
    """
    Pass 2 of ticker scoring pipeline:
    Applies session multiplier, Platt probability scaling, targets & expectancy,
    probability gating (with is_open_position hysteresis), and Kelly position sizing.
    """
    ticker = candidate.ticker
    sector = candidate.sector
    direction = candidate.direction
    close = candidate.close
    daily_df = candidate.daily_df
    bench = candidate.bench
    sector_ranks = candidate.sector_ranks
    sector_rs = candidate.sector_rs
    session = candidate.session
    regime = candidate.regime
    trade_horizon = candidate.trade_horizon
    factors = candidate.factors
    capital_fraction = candidate.capital_fraction
    intraday = candidate.intraday
    row = candidate.row

    # ── 6. Session + regime composite adjustment ──────────────────────────────
    sess_mult = _SESSION_MULT.get(session, 1.0)
    if regime.regime == MarketRegimeType.RANGE:
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

    targets = compute_targets(direction, close, atr, config, trade_horizon=trade_horizon)

    # Use value-area T1 if RR qualifies (only for SWING trades; INTRADAY preserves tight ATR targets)
    if config.USE_VALUE_AREA_RR and trade_horizon != "INTRADAY":
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
    min_prob = config.PROB_HOLD_FLOOR if is_open_position else config.MIN_PROB_WIN

    # Midday Chop Gate (10:30–13:30): Require higher hurdle (0.55) for directional breakouts
    # to protect against false breakouts, while allowing mean-reversion pullbacks
    if session == "MIDDAY_CHOP" and not regime.allows_mean_reversion() and not is_open_position:
        midday_hurdle = getattr(config, "MIDDAY_BREAKOUT_MIN_PROB", 0.55)
        min_prob = max(min_prob, midday_hurdle)

    watchlist_floor = getattr(config, "WATCHLIST_MIN_PROB", 0.45)
    is_watchlist = False

    if prob_win < min_prob:
        if allow_watchlist and prob_win >= watchlist_floor and exp_r >= 0.0:
            is_watchlist = True
        else:
            if debug:
                log.debug("%s: prob %.2f < gate %.2f (open_pos=%s)", ticker, prob_win, min_prob, is_open_position)
            return None
    if exp_r < config.MIN_EXPECTANCY_R and not is_watchlist:
        if debug:
            log.debug("%s: E(R) %.3f < gate %.3f", ticker, exp_r, config.MIN_EXPECTANCY_R)
        return None

    # ── 10. Kelly position sizing ─────────────────────────────────────────────
    shares, risk_inr, kelly_f, kurt_corr = calculate_kelly_size(
        entry=close,
        stop=targets.stop,
        prob_win=prob_win,
        rr=targets.rr,
        daily_df=daily_df,
        config=config,
        capital_fraction=capital_fraction,
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
    super_up = bool(row["Super_Up"])
    ema20  = float(row["EMA_20"])
    ema50  = float(row["EMA_50"])
    ema200 = float(row["EMA_200"])
    mtf_full = (ema20 > ema50 > ema200) if direction == "LONG" else (ema20 < ema50 < ema200)
    ema200_al = (close > ema200) if direction == "LONG" else (close < ema200)

    col = "Up_Day" if direction == "LONG" else "Dn_Day"
    streak = 0
    for v in reversed(daily_df[col].values[-10:]):
        if v == 1:
            streak += 1
        else:
            break

    tick_rs  = compute_rs(daily_df["Close"], bench, lookback=config.RS_LOOKBACK)
    sec_rank = sector_ranks.get(sector, N_SECTORS)
    sec_rs = sector_rs.get(sector, 0.0)

    change_pct = ((close - float(daily_df["Open"].iloc[-1]))
                  / float(daily_df["Open"].iloc[-1])) * 100

    # ── 12. Signal reasons (human-readable) ───────────────────────────────────
    reasons: list[str] = []
    if factors.trend > 0.7:
        reasons.append("Trend✅")
    if factors.momentum > 0.6:
        reasons.append(f"Mom✅RSI{rsi:.0f}")
    if factors.volume > 0.6:
        reasons.append(f"Vol✅×{rvol:.1f}")
    if factors.volatility > 0.6:
        reasons.append("Coiled")
    if factors.rs > 0.6:
        reasons.append(f"RS✅#{sec_rank}/{sec_rs:+.1f}")
    if factors.quality > 0.6:
        reasons.append("Qual✅")
    if adx >= 25:
        reasons.append(f"ADX{adx:.0f}")
    if mtf_full:
        reasons.append("MTF✅")
    if regime.allows_mean_reversion():
        reasons.append("MeanRev✅")
    if is_watchlist:
        reasons.append("Watchlist")
    if is_open_position:
        reasons.append("HeldPos")
    reasons.append(f"Regime:{regime.label}")
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
        regime=regime.label,
        session=session,
        trade_horizon=trade_horizon,
        action="BUY" if direction == "LONG" else "SELL",
        reasons=reasons,
        is_watchlist=is_watchlist,
        is_held=is_open_position,
    )


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
    capital_fraction: float = 1.0,
    debug:        bool = False,
    no_intraday:  bool = False,
    force_score:  bool = False,
    allow_watchlist: bool = False,
    now:          Optional[datetime] = None,
    is_open_position: bool = False,
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
        factor_weights: IC-calibrated weights - falls back to equal 1/7
        debug:          Print rejection reason to stdout
        force_score:    Bypass direction and regime gates
        allow_watchlist:Return watchlist tier candidates below MIN_PROB_WIN
        now:            Optional evaluation datetime
        is_open_position:If True, apply PROB_HOLD_FLOOR hysteresis gate

    Returns:
        TickerResult or None
    """
    cand = score_candidate_pass1(
        ticker=ticker,
        daily_df=daily_df,
        bench=bench,
        sector_ranks=sector_ranks,
        sector_rs=sector_rs,
        session=session,
        regime=regime,
        config=config,
        intraday=intraday,
        mtf_60m=mtf_60m,
        factor_weights=factor_weights,
        capital_fraction=capital_fraction,
        debug=debug,
        no_intraday=no_intraday,
        force_score=force_score,
        now=now,
        is_open_position=is_open_position,
    )
    if cand is None:
        return None
    return score_candidate_pass2(
        candidate=cand,
        config=config,
        is_open_position=is_open_position,
        allow_watchlist=allow_watchlist,
        debug=debug,
        now=now,
    )
