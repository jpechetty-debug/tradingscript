"""
core/portfolio.py
=================
Position sizing, trade targets, and portfolio optimisation.

v14.1 changes — FIX 2: NAV-aware Kelly sizing
----------------------------------------------
The original ``calculate_kelly_size`` clamped risk to a fixed
``RISK_PER_TRADE_INR`` constant with no connection to actual portfolio NAV.
Rs.10,000 risk per trade is reasonable at Rs.5L capital (2%) but catastrophic
at Rs.50,000 capital (20% per trade — ruin in five losers).

Root cause: ``capital_fraction`` defaulted to 1.0 and was never wired to a
live NAV source.  The ``CapitalScaler`` introduced here closes that gap.

How it works
~~~~~~~~~~~~
``CapitalScaler`` holds:
  * ``par_nav``  -- the "100%" reference NAV set once at session start.
  * ``live_nav`` -- updated each scan cycle from the broker / portfolio tracker.

``capital_fraction()`` returns ``live_nav / par_nav``, clipped to
[min_fraction, max_fraction] (defaults 0.25-2.0).  Pass this value into
``calculate_kelly_size(capital_fraction=...)`` so risk scales with NAV:

    * fraction > 1  winning run, size scales up proportionally.
    * fraction < 1  drawdown, size scales down automatically.
    * fraction = 1.0 (default)  identical to v14 behaviour; backtests safe.

In the modular runtime, the live NAV source is read per scan from
``state/portfolio_state.json`` (override via ``PORTFOLIO_STATE_PATH``).
External broker syncs can update that file between scans without changing
application code.

v14 changes (carried forward)
------------------------------
  - All CONFIG dict references replaced with SystemConfig injection.
  - _ticker_excess_kurtosis() made public (needed by scorer.py).
  - optimise_portfolio() wires MAX_CORR correlation filter (was dead code).
  - No module-level globals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis

from .config import SystemConfig

log = logging.getLogger("sovereign.portfolio")


# -----------------------------------------------------------------------------
# CAPITAL SCALER  (FIX 2)
# -----------------------------------------------------------------------------

@dataclass
class CapitalScaler:
    """
    NAV-aware scaling factor for Kelly position sizing.

    Attributes
    ----------
    par_nav : float
        Reference NAV (Rs.) corresponding to capital_fraction = 1.0.
        Set to actual portfolio value at session start, or use
        ``CapitalScaler.from_config(config)`` for a sensible default.
    live_nav : float
        Most recent broker portfolio value. Updated via ``update_nav()``.
    min_fraction : float
        Hard floor (default 0.25) -- never risk less than 25% of normal
        even in deep drawdown, so positions stay meaningful.
    max_fraction : float
        Hard ceiling (default 2.0) -- never more than 2x par sizing even
        on a hot streak, preventing runaway over-leverage.

    Example
    -------
    >>> scaler = CapitalScaler(par_nav=500_000)
    >>> scaler.update_nav(450_000)    # 10% drawdown
    >>> scaler.capital_fraction()     # -> 0.90
    >>> scaler.update_nav(650_000)    # 30% gain
    >>> scaler.capital_fraction()     # -> 1.30
    """

    par_nav:      float
    min_fraction: float = 0.25
    max_fraction: float = 2.0
    live_nav:     float = field(init=False)

    def __post_init__(self) -> None:
        if self.par_nav <= 0:
            raise ValueError(
                f"CapitalScaler: par_nav must be positive, got {self.par_nav}"
            )
        self.live_nav = self.par_nav    # start at par -- fraction = 1.0

    def update_nav(self, nav: float) -> None:
        """Record the current live portfolio value (Rs.)."""
        if nav <= 0:
            log.warning(
                "CapitalScaler.update_nav: ignoring non-positive nav=%.2f", nav
            )
            return
        prev = self.live_nav
        self.live_nav = nav
        log.debug(
            "CapitalScaler: nav Rs.%.0f -> Rs.%.0f  fraction=%.3f",
            prev, nav, self.capital_fraction(),
        )

    def capital_fraction(self) -> float:
        """
        Return live_nav / par_nav, clipped to [min_fraction, max_fraction].

        A value of 1.0 means full par sizing.  Above 1 scales up;
        below 1 scales down.
        """
        raw = self.live_nav / self.par_nav
        return float(np.clip(raw, self.min_fraction, self.max_fraction))

    @classmethod
    def from_config(cls, config: SystemConfig) -> "CapitalScaler":
        """
        Construct a CapitalScaler whose par NAV is implied by the config.

        Uses ``RISK_PER_TRADE_INR * 50`` as the reference capital, i.e. the
        configured per-trade risk already implies ~2% risk at that NAV.
        Call ``update_nav()`` with the actual portfolio value to override.
        """
        par = config.RISK_PER_TRADE_INR * 50
        log.info(
            "CapitalScaler: par_nav=Rs.%.0f (RISK_PER_TRADE_INR=Rs.%.0f x 50). "
            "Call update_nav() with the actual live portfolio value.",
            par, config.RISK_PER_TRADE_INR,
        )
        return cls(par_nav=par)


# -----------------------------------------------------------------------------
# TRADE TARGETS
# -----------------------------------------------------------------------------

@dataclass
class TradeTargets:
    stop: float
    t1:   float
    t2:   float
    rr:   float


def compute_targets(
    direction: str,
    close: float,
    atr: float,
    config: SystemConfig,
) -> TradeTargets:
    """
    ATR-based stop, T1, T2 and reward:risk ratio.

    Uses config.STOP_ATR_MULT, TARGET1_ATR_MULT, TARGET2_ATR_MULT.
    Value-area override (USE_VALUE_AREA_RR) is handled in scorer.py
    after the volume profile is computed.
    """
    sl_dist = config.STOP_ATR_MULT * atr

    if direction == "LONG":
        stop = round(close - sl_dist, 2)
        t1   = round(close + config.TARGET1_ATR_MULT * atr, 2)
        t2   = round(close + config.TARGET2_ATR_MULT * atr, 2)
    else:
        stop = round(close + sl_dist, 2)
        t1   = round(close - config.TARGET1_ATR_MULT * atr, 2)
        t2   = round(close - config.TARGET2_ATR_MULT * atr, 2)

    rr = round(abs(t1 - close) / sl_dist, 2) if sl_dist > 0 else 0.0
    return TradeTargets(stop=stop, t1=t1, t2=t2, rr=rr)


# -----------------------------------------------------------------------------
# FAT-TAIL KURTOSIS CORRECTION
# -----------------------------------------------------------------------------

def _ticker_excess_kurtosis(daily_df: pd.DataFrame, config: SystemConfig) -> float:
    """
    Excess kurtosis of daily returns over the configured lookback window.
    Falls back to KELLY_KURTOSIS_FALLBACK (4.0) if insufficient history.
    Clipped to [0, 20] -- extreme values destabilise the correction.
    """
    rets = daily_df["Close"].pct_change().dropna()
    if len(rets) < config.KELLY_KURTOSIS_MIN_OBS:
        return config.KELLY_KURTOSIS_FALLBACK
    rets_arr = rets.tail(config.KELLY_KURTOSIS_WINDOW).values
    try:
        ek = float(_kurtosis(rets_arr, fisher=True))
        return float(np.clip(ek, 0.0, 20.0))
    except Exception:
        log.debug("Kurtosis calc failed, using fallback.", exc_info=True)
        return config.KELLY_KURTOSIS_FALLBACK


# -----------------------------------------------------------------------------
# KELLY POSITION SIZING  (FIX 2 -- capital_fraction is now NAV-aware)
# -----------------------------------------------------------------------------

def calculate_kelly_size(
    entry:    float,
    stop:     float,
    prob_win: float,
    rr:       float,
    daily_df: pd.DataFrame,
    config:   SystemConfig,
    regime:   str = "EXPANSION",
    capital_fraction: float = 1.0,
) -> tuple[int, float, float, float]:
    """
    Fat-tail Kelly position sizing, NAV-scaled via ``capital_fraction``.

    Returns
    -------
    (shares, risk_inr, kelly_f, kurt_correction)

    Parameters
    ----------
    entry / stop
        Price levels -- rps = |entry - stop| (Rs. at risk per share).
    prob_win
        Win probability from the Platt-calibrated model.
    rr
        Reward:Risk ratio to T1.
    daily_df
        Daily OHLCV with Close -- used for excess-kurtosis calculation.
    config
        SystemConfig instance.
    regime
        Market regime string -- reserved for future regime-aware scaling.
    capital_fraction : float
        Ratio of live NAV to par NAV, from ``CapitalScaler.capital_fraction()``.
        Defaults to 1.0 (par sizing) so backtests without a live NAV feed
        are identical to v14 behaviour.

        The fraction scales both the target risk AND the absolute clamp
        bounds, so the min/max ratio stays constant regardless of NAV.
        A 50% drawdown halves both the target and the cap automatically.

    Sizing arithmetic
    -----------------
    1. Kelly fraction  f* = (p*(rr+1) - 1) / rr
    2. Fat-tail corr   kurt_corr = 3 / (3 + excess_kurtosis)
    3. Fractional K    f = f* x KELLY_FRACTION x kurt_corr
    4. Raw risk (Rs.)  risk = RISK_PER_TRADE_INR x max(f, 0.01) x 100 x cf
    5. Clamp           risk in [RISK_PER_TRADE_INR x 0.25 x cf,
                                RISK_PER_TRADE_INR x KELLY_MAX_MULT x cf]
    6. Shares          floor(risk / rps), min KELLY_MIN_SHARES.
    """
    rps = abs(entry - stop)
    if rps <= 0:
        return 0, 0.0, 0.0, 1.0

    # 1. Kelly fraction
    f_star = (prob_win * (rr + 1) - 1) / rr if rr > 0 else 0.0
    f_star = max(0.0, f_star)

    # 2. Fat-tail kurtosis correction
    excess_kurt = _ticker_excess_kurtosis(daily_df, config)
    kurt_corr   = 3.0 / (3.0 + excess_kurt)

    f = f_star * config.KELLY_FRACTION * kurt_corr

    # 3. NAV-scaled risk in Rs.
    risk_inr = config.RISK_PER_TRADE_INR * max(f, 0.01) * 100 * capital_fraction
    risk_inr = min(
        risk_inr,
        config.RISK_PER_TRADE_INR * config.KELLY_MAX_MULT * capital_fraction,
    )
    risk_inr = max(
        risk_inr,
        config.RISK_PER_TRADE_INR * 0.25 * capital_fraction,
    )

    shares = (
        max(config.KELLY_MIN_SHARES, int(risk_inr / rps))
        if capital_fraction > 0 else 0
    )
    return shares, round(shares * rps, 2), round(f, 5), round(kurt_corr, 4)


# -----------------------------------------------------------------------------
# PORTFOLIO OPTIMISATION  (with MAX_CORR filter -- previously dead code)
# -----------------------------------------------------------------------------

def optimize_portfolio(
    candidates:  list,
    config:      SystemConfig,
    corr_matrix: Optional[pd.DataFrame] = None,
) -> list:
    """
    Select up to config.PORTFOLIO_SIZE tickers from candidates, subject to:
      1. Sector cap  -- max config.MAX_SECTOR_PICKS per sector.
      2. Correlation -- skip any ticker with |corr| > config.MAX_CORR
                        vs an already-selected ticker.

    Args
    ----
    candidates
        List of TickerResult -- sorted by sharpe_rank desc.
    config
        SystemConfig instance.
    corr_matrix
        Pairwise correlation matrix of daily returns.
        Index/columns in "{ticker}.NS" format.
        Pass pd.DataFrame() or None to skip the correlation filter.

    Returns
    -------
    List of selected TickerResult (length <= PORTFOLIO_SIZE).
    """
    sorted_candidates = sorted(candidates, key=lambda r: r.sharpe_rank, reverse=True)
    selected: list          = []
    sector_counts: dict[str, int] = {}

    for c in sorted_candidates:
        if len(selected) >= config.PORTFOLIO_SIZE:
            break

        # Sector cap
        if sector_counts.get(c.sector, 0) >= config.MAX_SECTOR_PICKS:
            log.debug("Portfolio: %s sector cap hit (%s)", c.ticker, c.sector)
            continue

        # Correlation filter
        if corr_matrix is not None and not corr_matrix.empty:
            c_key    = f"{c.ticker}.NS"
            too_corr = False
            if c_key in corr_matrix.columns:
                for s in selected:
                    s_key = f"{s.ticker}.NS"
                    if s_key in corr_matrix.index and c_key in corr_matrix.index:
                        corr_val = abs(float(corr_matrix.loc[c_key, s_key]))
                        if corr_val > config.MAX_CORR:
                            log.debug(
                                "Portfolio: %s dropped -- corr(%.2f) > MAX_CORR(%.2f) with %s",
                                c.ticker, corr_val, config.MAX_CORR, s.ticker,
                            )
                            too_corr = True
                            break
            if too_corr:
                continue

        selected.append(c)
        sector_counts[c.sector] = sector_counts.get(c.sector, 0) + 1

    return selected
