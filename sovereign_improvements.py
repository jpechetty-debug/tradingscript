"""
Legacy compatibility layer for runtime collaborators that now live in ``core/``.

The modular runtime no longer executes this file directly. It remains available
for older tooling that imports ``sovereign_improvements`` and expects the gate,
scaler, calibrator, alerter, and ``apply()`` helper to exist.
"""

from __future__ import annotations

import warnings
from typing import Any

import yfinance as yf

from core.data_provider import fetch_daily_batch
from core.runtime_components import (
    RECALIBRATION_INTERVAL,
    RECALIBRATION_WINDOW,
    RegimeAwareTelegramAlerter,
    RegimeProbabilityGate,
    RollingFactorCalibrator,
    RuntimeComponents,
    TieredCapitalScaler,
    create_runtime_components,
)

__all__ = [
    "RECALIBRATION_INTERVAL",
    "RECALIBRATION_WINDOW",
    "RegimeAwareTelegramAlerter",
    "RegimeProbabilityGate",
    "ResilientDataProvider",
    "RollingFactorCalibrator",
    "RuntimeComponents",
    "SovereignComponents",
    "TieredCapitalScaler",
    "apply",
    "create_components",
]

SovereignComponents = RuntimeComponents


class ResilientDataProvider:
    """
    Compatibility adapter around the modular market-data fetch path.

    The old patch exposed a ``data_provider`` object with a ``_yf`` attribute
    and a fetch-like method. That behavior is preserved just enough for legacy
    scripts while keeping the implementation in ``core.data_provider``.
    """

    def __init__(self, fyers_client: Any = None) -> None:
        self._fyers = fyers_client
        self._yf = yf

    def fetch(self, symbols: list[str], config: Any) -> Any:
        return fetch_daily_batch(symbols, config)


def create_components(
    *,
    portfolio_peak: float = 1_000_000.0,
    current_nav: float | None = None,
    fyers_client: Any = None,
    recal_window: int = RECALIBRATION_WINDOW,
    recal_interval: int = RECALIBRATION_INTERVAL,
    announce: bool = False,
) -> RuntimeComponents:
    _ = current_nav, fyers_client
    warnings.warn(
        "sovereign_improvements.create_components() is deprecated; use "
        "core.runtime_components.create_runtime_components() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_runtime_components(
        portfolio_peak=portfolio_peak,
        recal_window=recal_window,
        recal_interval=recal_interval,
        announce=announce,
    )


def _safe_attach(target: Any, attr: str, value: Any) -> None:
    if target is None:
        return
    if isinstance(target, dict):
        target[attr] = value
        return
    setattr(target, attr, value)


def apply(
    engine: Any,
    *,
    portfolio_peak: float = 1_000_000.0,
    current_nav: float | None = None,
    fyers_client: Any = None,
    recal_window: int = RECALIBRATION_WINDOW,
    recal_interval: int = RECALIBRATION_INTERVAL,
    announce: bool = False,
) -> dict[str, Any]:
    warnings.warn(
        "sovereign_improvements.apply() is deprecated; the live runtime now "
        "builds collaborators from core.runtime_components directly.",
        DeprecationWarning,
        stacklevel=2,
    )
    components = create_components(
        portfolio_peak=portfolio_peak,
        current_nav=current_nav,
        fyers_client=fyers_client,
        recal_window=recal_window,
        recal_interval=recal_interval,
        announce=announce,
    )
    data_provider = ResilientDataProvider(fyers_client=fyers_client)

    _safe_attach(engine, "capital_scaler", components.scaler)
    _safe_attach(engine, "prob_gate", components.gate)
    _safe_attach(engine, "calibrator", components.calibrator)
    _safe_attach(engine, "alerter", components.alerter)
    _safe_attach(engine, "data_provider", data_provider)

    return {
        "scaler": components.scaler,
        "gate": components.gate,
        "calibrator": components.calibrator,
        "alerter": components.alerter,
        "data": data_provider,
        "capital_scaler": components.scaler,
        "prob_gate": components.gate,
        "data_provider": data_provider,
    }
