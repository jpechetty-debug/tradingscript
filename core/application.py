"""Shared application composition root for CLI and API entry points."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .runtime_components import RuntimeComponents, create_runtime_components
from .services import PersistenceService, ServiceBundle, configure_services


@dataclass(frozen=True)
class ApplicationRuntime:
    services: ServiceBundle
    components: RuntimeComponents

    def close(self) -> None:
        self.components.stop()


def build_application_runtime(
    *,
    version: str,
    persistence: PersistenceService | None = None,
    component_factory: Callable[..., RuntimeComponents] = create_runtime_components,
    service_factory: Callable[..., ServiceBundle] = configure_services,
) -> ApplicationRuntime:
    """Build the same live collaborators for every application surface."""
    resolved_persistence = persistence or PersistenceService()
    snapshot = resolved_persistence.load_portfolio_state()
    portfolio_peak = (
        snapshot.peak_nav
        if snapshot is not None and snapshot.peak_nav is not None
        else snapshot.current_nav
        if snapshot is not None
        else 1_000_000.0
    )
    components = component_factory(portfolio_peak=portfolio_peak)

    def current_nav() -> float | None:
        current = resolved_persistence.load_portfolio_state()
        return current.current_nav if current is not None else None

    services = service_factory(
        persistence=resolved_persistence,
        probability_gate=components.gate,
        capital_scaler=components.scaler,
        factor_calibrator=components.calibrator,
        alerter=components.alerter,
        current_nav_provider=current_nav,
        version=version,
    )
    return ApplicationRuntime(services=services, components=components)
