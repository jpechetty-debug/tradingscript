"""
core/regime.py
==============
Market regime detection for Sovereign Engine v14.

Hardening history
-----------------
Fix 1 — PANIC hysteresis      : asymmetric exit threshold (REGIME_BREADTH_PANIC_EXIT)
Fix 2 — TREND deadband        : [0.45, 0.55] in both ADX branches eliminates cliff edge
Fix 3 — EXPANSION confidence  : scales with ADX (+0.25) and ATR (+0.20)
Fix 4 — breadth_delta         : direction-of-breadth field on MarketRegime
Fix 5 — deque migration       : O(1) eviction; last_regime() / last_breadth() helpers
Fix 6 — RegimeLock            : locked=True during opening noise window suppresses
                                 tracker.push() so a single noisy bar cannot flip regime
Fix 7 — Sector RS feedback    : sector_concentration field; broad sector participation
                                 raises TREND confidence by up to +0.10
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MarketRegimeType, SystemConfig


# ── RegimeTracker ─────────────────────────────────────────────────────────────

class RegimeTracker:
    """Ring buffer of recent regime classifications.

    Uses ``deque(maxlen)`` for O(1) eviction (Fix 5).  Breadth is stored
    alongside each push so callers can retrieve the last observed breadth
    for hysteresis decisions without a separate data structure.
    """

    def __init__(self, max_history: int = 10) -> None:
        self._history: deque[MarketRegimeType] = deque(maxlen=max_history)
        self._breadths: deque[float] = deque(maxlen=max_history)

    def push(self, regime: MarketRegimeType, breadth: float = 0.0) -> None:
        self._history.append(regime)
        self._breadths.append(breadth)

    def last_regime(self) -> MarketRegimeType | None:
        """Most recent regime, or ``None`` if history is empty."""
        return self._history[-1] if self._history else None

    def last_breadth(self) -> float:
        """Breadth recorded at the most recent push, or 0.0 if empty."""
        return self._breadths[-1] if self._breadths else 0.0

    def is_confirmed(self, regime: MarketRegimeType, confirm_bars: int) -> bool:
        if regime == MarketRegimeType.PANIC:
            return True  # PANIC is always confirmed — capital protection
        tail = list(self._history)[-confirm_bars:]
        return len(tail) == confirm_bars and all(r == regime for r in tail)


# ── MarketRegime dataclass ────────────────────────────────────────────────────

@dataclass
class MarketRegime:
    regime: MarketRegimeType
    breadth: float
    adx_median: float
    atr_ratio: float
    confidence: float
    confirmed: bool
    breadth_delta: float = 0.0          # Fix 4: change vs previous bar; >0 = improving
    sector_concentration: float = 0.5   # Fix 7: fraction of sectors aligned with regime
    regime_locked: bool = False          # Fix 6: True during opening noise window

    @property
    def label(self) -> str:
        """String label for logging / serialization boundaries."""
        return self.regime.value

    def allows_long(self) -> bool:
        return (
            self.regime in (MarketRegimeType.TREND_UP, MarketRegimeType.EXPANSION)
            and self.confirmed
        )

    def allows_short(self) -> bool:
        return (
            self.regime in (MarketRegimeType.TREND_DOWN, MarketRegimeType.EXPANSION)
            and self.confirmed
        )

    def is_tradeable(self) -> bool:
        return self.regime != MarketRegimeType.PANIC

    def strategy_hint(self) -> str:
        s = {
            MarketRegimeType.TREND_UP:   "BREAKOUT / MOMENTUM (confirmed)",
            MarketRegimeType.TREND_DOWN: "SHORT MOMENTUM (confirmed)",
            MarketRegimeType.RANGE:      "MEAN REVERSION ONLY — directional trades blocked",
            MarketRegimeType.EXPANSION:  "VOLATILITY BREAKOUT — both sides",
            MarketRegimeType.PANIC:      "NO TRADE — protect capital",
        }[self.regime]
        if not self.confirmed:
            s += " [UNCONFIRMED]"
        if self.regime_locked:
            s += " [LOCKED — opening noise window]"
        return s


# ── Confidence ────────────────────────────────────────────────────────────────

def _regime_confidence(
    regime: MarketRegimeType,
    adx_med: float,
    breadth: float,
    atr_rat: float,
    config: SystemConfig,
    sector_conc: float = 0.5,
) -> float:
    """Strength-weighted confidence score in [0, 1].

    sector_conc (Fix 7):
        Fraction of sectors aligned with current regime direction.
        At 0.5 the bonus is zero; at 1.0 it adds +0.10 to TREND confidence.
    """
    if regime == MarketRegimeType.PANIC:
        return float(np.clip(
            0.70 + (config.REGIME_BREADTH_PANIC - breadth) * 2,
            0.70, 1.0,
        ))

    if regime in (MarketRegimeType.TREND_UP, MarketRegimeType.TREND_DOWN):
        adx_strength = float(np.clip((adx_med - config.REGIME_ADX_TREND) / 15.0, 0.0, 1.0))
        breadth_str  = float(np.clip(abs(breadth - 0.5) * 2.0, 0.0, 1.0))
        # Fix 7: broad sector participation lifts confidence by up to +0.10
        conc_bonus   = float(np.clip((sector_conc - 0.5) * 2.0, 0.0, 1.0)) * 0.10
        base = 0.50 + adx_strength * 0.30 + breadth_str * 0.20
        return round(float(np.clip(base + conc_bonus, 0.0, 1.0)), 3)

    if regime == MarketRegimeType.EXPANSION:
        # Fix 3: scale with ADX and ATR rather than flat 0.55
        adx_str = float(np.clip((adx_med - config.REGIME_ADX_TREND) / 15.0, 0.0, 1.0))
        atr_str = float(np.clip((atr_rat - config.REGIME_ATR_EXPANSION) / 0.7, 0.0, 1.0))
        return round(float(np.clip(0.55 + adx_str * 0.25 + atr_str * 0.20, 0.0, 1.0)), 3)

    return 0.55  # RANGE — neutral baseline


def confidence_position_scale(confidence: float) -> float:
    """
    Convert regime confidence into a gentle position-size multiplier.

    The regime itself already gates directional eligibility. This helper
    makes confidence operational without letting a middling confidence
    reading collapse sizing too aggressively:

    - 0.00 confidence -> 0.80x size
    - 0.50 confidence -> 0.90x size
    - 1.00 confidence -> 1.00x size
    """
    bounded = float(np.clip(confidence, 0.0, 1.0))
    return round(0.80 + 0.20 * bounded, 3)


# ── Sector concentration (Fix 7) ─────────────────────────────────────────────

def compute_sector_concentration(
    sector_rs: dict[str, float],
    regime: MarketRegimeType,
) -> float:
    """
    Fraction of sectors whose RS is aligned with the current regime direction.

    - ``TREND_UP``  : fraction with RS > 0  (broad participation → closer to 1.0)
    - ``TREND_DOWN``: fraction with RS < 0  (broad selling → closer to 1.0)
    - All others    : 0.5 (neutral; concentration is not meaningful)

    Returns 0.5 when ``sector_rs`` is empty or regime is non-directional so
    existing confidence formulas are unaffected in the default no-RS path.
    """
    if not sector_rs or regime not in (
        MarketRegimeType.TREND_UP, MarketRegimeType.TREND_DOWN
    ):
        return 0.5
    if regime == MarketRegimeType.TREND_UP:
        aligned = sum(1 for v in sector_rs.values() if v > 0)
    else:
        aligned = sum(1 for v in sector_rs.values() if v < 0)
    return round(aligned / len(sector_rs), 3)


# ── Core classification ───────────────────────────────────────────────────────

def classify_regime(
    processed: dict[str, pd.DataFrame],
    breadth: float,
    tracker: RegimeTracker,
    config: SystemConfig,
    *,
    locked: bool = False,
    sector_rs: dict[str, float] | None = None,
) -> MarketRegime:
    """
    Classify the current market regime from cross-sectional indicator data.

    Parameters
    ----------
    processed:
        Dict of ``{ticker: DataFrame}`` with indicator columns pre-computed.
    breadth:
        Fraction of non-benchmark tickers trading above their EMA-50.
    tracker:
        Persistent ``RegimeTracker`` that maintains bar history across calls.
    config:
        ``SystemConfig`` with all threshold parameters.
    locked:
        Fix 6 — when ``True`` (opening noise window), metrics are computed
        for logging but ``tracker.push()`` is **skipped**.  The last
        confirmed regime is returned unchanged, preventing whipsaw
        reclassification during the first ``REGIME_LOCK_MINUTES`` minutes
        after market open.  Defaults to ``False`` so existing call-sites
        and tests that do not pass the argument are unaffected.
    sector_rs:
        Fix 7 — optional per-sector RS dict from ``compute_sector_rs()``.
        Drives ``sector_concentration`` on the returned ``MarketRegime`` and
        lifts TREND confidence for broadly participating moves.
        ``None`` → defaults to 0.5 (no bonus, backward-compatible).
    """
    adx_vals: list[float] = []
    atr_ratios: list[float] = []
    for ticker, df in processed.items():
        if ticker == config.BENCHMARK or df.empty:
            continue
        if "ADX" in df.columns:
            adx_vals.append(float(df["ADX"].iloc[-1]))
        if "ATR" in df.columns and "ATR_50_mean" in df.columns:
            m = float(df["ATR_50_mean"].iloc[-1])
            if m > 0:
                atr_ratios.append(float(df["ATR"].iloc[-1]) / m)

    adx_med = float(np.median(adx_vals)) if adx_vals else 20.0
    atr_rat = float(np.median(atr_ratios)) if atr_ratios else 1.0

    # Fix 4: breadth delta — captured before any push
    prev_breadth  = tracker.last_breadth()
    breadth_delta = round(breadth - prev_breadth, 4)

    # Fix 1: asymmetric PANIC exit — require higher breadth to *leave* PANIC
    in_panic        = tracker.last_regime() == MarketRegimeType.PANIC
    panic_threshold = (
        config.REGIME_BREADTH_PANIC_EXIT if in_panic else config.REGIME_BREADTH_PANIC
    )

    # ── Classification tree ────────────────────────────────────────────────────
    if breadth < panic_threshold:
        raw_regime = MarketRegimeType.PANIC
    elif atr_rat >= config.REGIME_ATR_EXPANSION and adx_med >= config.REGIME_ADX_TREND:
        raw_regime = MarketRegimeType.EXPANSION
    elif adx_med >= config.REGIME_ADX_TREND:
        # Fix 2: [0.45, 0.55] deadband in primary high-ADX branch (was cliff at 0.50)
        if breadth >= 0.55:
            raw_regime = MarketRegimeType.TREND_UP
        elif breadth < 0.45:
            raw_regime = MarketRegimeType.TREND_DOWN
        else:
            raw_regime = MarketRegimeType.RANGE
    elif adx_med < config.REGIME_ADX_RANGE:
        raw_regime = MarketRegimeType.RANGE
    else:
        # Mid-ADX zone — [0.45, 0.55] deadband unchanged from hardening
        if breadth >= 0.55:
            raw_regime = MarketRegimeType.TREND_UP
        elif breadth < 0.45:
            raw_regime = MarketRegimeType.TREND_DOWN
        else:
            raw_regime = MarketRegimeType.RANGE

    # Fix 6: during the opening lock window, freeze tracker state
    if locked:
        # Do NOT update the tracker so the lock cannot alter confirmed history.
        # Fall back to computed regime only if tracker has no prior history
        # (e.g., the very first bar of the session).
        regime = tracker.last_regime() or raw_regime
    else:
        tracker.push(raw_regime, breadth)
        regime = raw_regime

    confirmed = tracker.is_confirmed(regime, config.REGIME_CONFIRM_BARS)

    # Fix 7: sector concentration → confidence bonus
    sector_conc = compute_sector_concentration(sector_rs or {}, regime)
    conf = _regime_confidence(regime, adx_med, breadth, atr_rat, config, sector_conc)

    return MarketRegime(
        regime=regime,
        breadth=breadth,
        adx_median=adx_med,
        atr_ratio=atr_rat,
        confidence=conf,
        confirmed=confirmed,
        breadth_delta=breadth_delta,
        sector_concentration=sector_conc,
        regime_locked=locked,
    )


# ── RS helpers (unchanged) ────────────────────────────────────────────────────

def compute_rs(
    stock: pd.Series,
    bench: pd.Series,
    lookback: int | None = None,
    config: SystemConfig | None = None,
) -> float:
    """
    Log-return relative-strength of *stock* vs *bench* over *lookback* bars.

    Parameters
    ----------
    stock:
        Close price series for the individual ticker.
    bench:
        Close price series for the benchmark (e.g. Nifty50).
    lookback:
        Number of bars to look back.  **Prefer passing this explicitly.**
        If omitted, ``config.RS_LOOKBACK`` is used when *config* is
        supplied; otherwise the module-level CONFIG singleton is the
        last resort.  Unit tests should always pass either *lookback*
        or *config* directly so they can control the parameter without
        patching the singleton.
    config:
        Optional ``SystemConfig`` instance.  Ignored when *lookback* is
        provided.
    """
    if lookback is None:
        if config is not None:
            lookback = config.RS_LOOKBACK
        else:
            from .config import CONFIG  # last-resort singleton — avoid in tests
            lookback = CONFIG.RS_LOOKBACK

    m = stock.rename("s").to_frame().join(bench.rename("b"), how="inner").dropna()
    if len(m) < lookback + 1:
        return 0.0
    s = np.log(m["s"].iloc[-1] / m["s"].iloc[-lookback - 1])
    b = np.log(m["b"].iloc[-1] / m["b"].iloc[-lookback - 1])
    return round(float((s - b) * 100), 3)


def compute_breadth(processed: dict[str, pd.DataFrame], config: SystemConfig) -> float:
    total = above = 0
    for ticker, df in processed.items():
        if ticker == config.BENCHMARK or df.empty:
            continue
        if "EMA_50" in df.columns:
            total += 1
            if float(df["Close"].iloc[-1]) > float(df["EMA_50"].iloc[-1]):
                above += 1
    return float(above / total) if total else 0.5


def compute_sector_rs(
    processed: dict[str, pd.DataFrame],
    bench: pd.Series,
    config: SystemConfig,
) -> dict[str, float]:
    from collections import defaultdict
    from .universe import TICKER_TO_SECTOR
    scores: dict[str, list[float]] = defaultdict(list)
    for ticker, df in processed.items():
        if ticker == config.BENCHMARK or df.empty:
            continue
        sector = TICKER_TO_SECTOR.get(ticker)
        if sector:
            scores[sector].append(
                compute_rs(df["Close"], bench, lookback=config.RS_LOOKBACK)
            )
    return {
        s: round(float(np.median(v)), 3) if v else 0.0
        for s, v in scores.items()
    }
