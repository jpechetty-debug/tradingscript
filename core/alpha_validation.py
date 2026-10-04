"""Calendar validation and reproducible trade-level stress tests; no edge claims."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ValidationFold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    partial: bool


def calendar_folds(sessions: pd.DatetimeIndex, train_years: int = 5,
                   test_years: int = 1) -> list[ValidationFold]:
    """Rolling calendar-year windows, disjoint tests, observed exchange sessions only."""
    if train_years < 1 or test_years < 1 or sessions.empty or sessions.has_duplicates or not sessions.is_monotonic_increasing:
        raise ValueError("Need ordered unique sessions and positive window lengths")
    if sessions.tz is not None:
        raise ValueError("Sessions must be timezone-free")
    folds = []
    for year in range(sessions[0].year + train_years, sessions[-1].year + 1, test_years):
        boundary = pd.Timestamp(year, 1, 1)
        next_boundary = pd.Timestamp(year + test_years, 1, 1)
        train = sessions[(sessions >= pd.Timestamp(year - train_years, 1, 1)) & (sessions < boundary)]
        test = sessions[(sessions >= boundary) & (sessions < next_boundary)]
        if train.empty or test.empty:
            continue
        folds.append(ValidationFold(train[0], train[-1], test[0], test[-1],
                                    sessions[-1] < next_boundary - pd.Timedelta(days=7)))
    return folds


@dataclass(frozen=True)
class StressSettings:
    simulations: int = 10_000
    seed: int = 20261004
    risk_fraction: float = .01
    slippage_r_mean: float = .05
    gap_probability: float = .03
    gap_loss_r: float = 1.
    missed_fill_probability: float = .05
    ruin_equity_fraction: float = .5
    block_size: int = 5

    def __post_init__(self) -> None:
        if (not all(np.isfinite(v) for v in asdict(self).values()) or self.simulations < 1
                or self.simulations > 200_000 or self.block_size < 1 or self.seed < 0
                or not 0 < self.risk_fraction <= 1 or self.slippage_r_mean < 0 or self.gap_loss_r < 0
                or not 0 <= self.gap_probability <= 1 or not 0 <= self.missed_fill_probability <= 1
                or not 0 < self.ruin_equity_fraction < 1):
            raise ValueError("Invalid stress parameters")


def monte_carlo(trades: Sequence[Any], settings: StressSettings = StressSettings()) -> dict[str, Any]:
    r_values = np.array([float(t.r_multiple) for t in trades])
    if not np.isfinite(r_values).all():
        raise ValueError("Stress testing requires finite net R outcomes")
    base = {"settings": asdict(settings), "trades": len(trades),
            "method": "CIRCULAR_BLOCK_TRADE_BOOTSTRAP_WITH_ADDITIONAL_FRICTION",
            "limitations": ["Trade-level approximation; concurrent exposure, cash constraints and daily correlations are not replayed.",
                            "Shock distributions and ruin threshold are assumptions, not estimated market probabilities."]}
    if not len(trades):
        return {**base, "status": "UNAVAILABLE_NO_CLOSED_TRADES"}
    rng = np.random.default_rng(settings.seed)
    final = np.empty(settings.simulations)
    max_dd = np.empty(settings.simulations)
    ruined = np.empty(settings.simulations, dtype=bool)
    n = len(trades)
    block = min(settings.block_size, n)
    blocks = int(np.ceil(n / block))
    for start in range(0, settings.simulations, 256):
        count = min(256, settings.simulations - start)
        indices = (rng.integers(0, n, (count, blocks, 1)) + np.arange(block)) % n
        outcomes = r_values[indices.reshape(count, -1)[:, :n]].copy()
        outcomes -= rng.exponential(settings.slippage_r_mean, outcomes.shape)
        outcomes -= (rng.random(outcomes.shape) < settings.gap_probability) * settings.gap_loss_r
        outcomes[rng.random(outcomes.shape) < settings.missed_fill_probability] = 0.
        factors = np.maximum(0., 1. + settings.risk_fraction * outcomes)
        with np.errstate(over="ignore"):
            wealth = np.column_stack([np.ones(count), np.cumprod(factors, axis=1)])
        if not np.isfinite(wealth).all():
            raise ValueError("Stress assumptions overflowed simulated wealth")
        peaks = np.maximum.accumulate(wealth, axis=1)
        max_dd[start:start + count] = np.max(1. - wealth / peaks, axis=1)
        final[start:start + count] = wealth[:, -1]
        ruined[start:start + count] = np.min(wealth, axis=1) <= settings.ruin_equity_fraction
    loss = 1. - final
    var = float(np.quantile(loss, .95))
    return {**base, "status": "SIMULATED_ASSUMPTIONS_ONLY", "worst_drawdown": float(max_dd.max()),
            "drawdown_p95": float(np.quantile(max_dd, .95)), "loss_var95": max(0., var),
            "loss_expected_shortfall95": max(0., float(loss[loss >= var].mean())),
            "risk_of_ruin": float(ruined.mean()), "terminal_equity_p05": float(np.quantile(final, .05)),
            "terminal_equity_median": float(np.median(final))}
