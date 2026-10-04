"""Calibration boundary separated from scan/data/portfolio orchestration."""
from __future__ import annotations

import logging
from typing import Any, Callable, Protocol

from .database import SqliteDatabase
from .factors import DEFAULT_WEIGHTS

log = logging.getLogger("sovereign.services")


class CalibrationPersistence(Protocol):
    db: SqliteDatabase

    def load_trade_log(self) -> list[dict[str, Any]]: ...
    def save_platt(self, a: float, b: float) -> None: ...


class CalibrationService:
    def __init__(self, persistence: CalibrationPersistence,
                 platt_fit: Callable[..., tuple[float, float]]) -> None:
        self.persistence = persistence
        self.platt_fit = platt_fit

    def run_platt(self, calib_offset: int = 60) -> None:
        # Prefer executed_trades (proper lifecycle) over legacy trade_log
        composites, outcomes = self.persistence.db.fetch_calibration_trades(min_samples=80)
        if composites and outcomes:
            log.info(
                "Calibrating from %d closed executed trades.",
                len(composites),
            )
            a, b = self.platt_fit(composites, outcomes, calib_offset=calib_offset)
            self.persistence.save_platt(a, b)
            return

        # Fallback: legacy trade_log (but warn that it's not validated)
        trades = self.persistence.load_trade_log()
        if not trades:
            log.warning(
                "No executed trades and no legacy trade log found. "
                "Using config-default Platt coefficients (A=%.1f, B=%.1f). "
                "Record real trades with outcomes to enable calibration.",
                -4.0, 2.0,
            )
            return

        composites_legacy: list[float] = []
        outcomes_legacy: list[int] = []
        for trade in trades:
            if trade.get("source", "EXECUTED") != "EXECUTED":
                continue
            factors = trade.get("factors")
            pnl = trade.get("pnl")
            if not isinstance(factors, dict) or pnl is None:
                continue

            composite = trade.get("composite")
            if composite is None:
                composite = sum(
                    float(factors.get(key, 0.0)) * DEFAULT_WEIGHTS.get(key, 0.0)
                    for key in DEFAULT_WEIGHTS
                )
            composites_legacy.append(float(composite))
            outcomes_legacy.append(1 if float(pnl) > 0 else 0)

        if len(composites_legacy) < 5:
            log.warning("Insufficient data for calibration (%d samples).", len(composites_legacy))
            return

        a, b = self.platt_fit(composites_legacy, outcomes_legacy, calib_offset=calib_offset)
        self.persistence.save_platt(a, b)
