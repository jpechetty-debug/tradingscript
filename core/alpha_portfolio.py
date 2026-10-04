"""Held-book allocation limits for the four-phase alpha research pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PortfolioLimits:
    max_heat: float = .06
    max_sector_exposure: float = .25
    max_correlation: float = .75
    min_correlation_observations: int = 30

    def __post_init__(self) -> None:
        if (not all(np.isfinite(v) and 0 < v <= 1 for v in
                    (self.max_heat, self.max_sector_exposure, self.max_correlation))
                or self.min_correlation_observations < 2):
            raise ValueError("Invalid portfolio limits")


def bounded_quantity(quantity: int, price: float, stop: float, nav: float,
                     held_risk: float, sector_value: float, absolute_risk_limit: float,
                     limits: PortfolioLimits, risk_scale: float = 1.) -> tuple[int, str]:
    """Additional entries only; never resize or liquidate the already-held book."""
    numbers = (price, stop, nav, held_risk, sector_value, absolute_risk_limit, risk_scale)
    if (not all(np.isfinite(v) for v in numbers) or quantity < 0 or price <= stop or stop <= 0
            or nav <= 0 or min(held_risk, sector_value, absolute_risk_limit) < 0 or not 0 <= risk_scale <= 1):
        return 0, "invalid_allocation_inputs"
    if risk_scale == 0:
        return 0, "breadth_defensive"
    scaled = int(quantity * risk_scale)
    available_heat = max(0., min(absolute_risk_limit, nav * limits.max_heat) - held_risk)
    heat_quantity = int(available_heat / (price - stop))
    sector_quantity = int(max(0., nav * limits.max_sector_exposure - sector_value) / price)
    result = min(scaled, heat_quantity, sector_quantity)
    reason = "heat_cap" if heat_quantity <= 0 else "sector_exposure_cap" if sector_quantity <= 0 else "allocated"
    return result, reason


def allocate_candidates(candidates: Sequence[Any], config: Any, returns: pd.DataFrame,
                        held: Sequence[dict[str, Any]] = (), limits: PortfolioLimits = PortfolioLimits()) -> list[dict[str, Any]]:
    """Greedy alpha-rank allocation with cash, heat, sector and measured correlation limits.

    held rows: ticker, sector, shares, price (current mark), risk_inr.
    Returns new allocations and rejection reasons without mutating candidates.
    """
    nav = float(config.CAPITAL_INR)
    if not np.isfinite(nav) or nav <= 0:
        raise ValueError("Allocation requires positive NAV")
    book = [dict(row) for row in held]
    identifiers = [str(p.get("ticker", "")).removesuffix(".NS") for p in book]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Duplicate held ticker")
    for row in book:
        if (not row.get("ticker") or not row.get("sector") or row["shares"] <= 0 or row["price"] <= 0
                or row["risk_inr"] < 0 or not all(np.isfinite(row[k]) for k in ("shares", "price", "risk_inr"))):
            raise ValueError("Invalid held position")
    # Accept both plain security symbols and price-series identifiers.
    matrix = returns.copy()
    matrix.columns = pd.Index([str(t).removesuffix(".NS") for t in matrix.columns])
    if matrix.columns.duplicated().any():
        raise ValueError("Duplicate return series")
    matrix = matrix.replace([np.inf, -np.inf], np.nan)
    corr = matrix.corr(min_periods=limits.min_correlation_observations)
    cash = max(0., nav - sum(p["shares"] * p["price"] for p in book))
    allocations: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=lambda c: (-c.composite, c.ticker)):
        ticker = candidate.ticker.removesuffix(".NS")
        reason = ""
        if ticker in {p["ticker"].removesuffix(".NS") for p in book}:
            reason = "already_held"
        elif len(book) >= config.PORTFOLIO_SIZE:
            reason = "position_cap"
        elif sum(p["sector"] == candidate.sector for p in book) >= config.MAX_SECTOR_PICKS:
            reason = "sector_cap"
        else:
            for position in book:
                key = position["ticker"].removesuffix(".NS")
                value = corr.loc[ticker, key] if ticker in corr.index and key in corr else np.nan
                if not np.isfinite(value):
                    reason = "correlation_unknown"
                    break
                if abs(value) > config.MAX_CORR or abs(value) >= limits.max_correlation:
                    reason = "correlation_cap"
                    break
        quantity = 0
        if not reason:
            from .swing import SwingPlan, swing_fill_size
            plan = SwingPlan(candidate.strategy_id, candidate.signal_time, candidate.entry,
                             candidate.entry_min, candidate.entry_max, candidate.stop, candidate.t1,
                             candidate.t2, candidate.time_stop_bars, 0.)
            quantity, _ = swing_fill_size(plan, candidate.entry, config)
            quantity, reason = bounded_quantity(
                quantity, candidate.entry, candidate.stop, nav, sum(p["risk_inr"] for p in book),
                sum(p["shares"] * p["price"] for p in book if p["sector"] == candidate.sector),
                config.MAX_PORTFOLIO_RISK_INR, limits, candidate.research_context.get("breadth_risk_scale", 1.))
            quantity = min(quantity, int(cash / candidate.entry))
            if quantity <= 0 and reason == "allocated":
                reason = "cash_or_size_cap"
        allocations.append({"ticker": ticker, "sector": candidate.sector, "shares": quantity,
                            "reason": reason, "price": candidate.entry,
                            "risk_inr": quantity * max(0., candidate.entry - candidate.stop)})
        if quantity:
            book.append(allocations[-1])
            cash -= quantity * candidate.entry
    return allocations
