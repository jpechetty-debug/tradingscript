"""
screener_v14_modular.py
=======================
Backward-compatible library entry point for the Sovereign Engine.

The runtime orchestration now lives in explicit services under ``core/``:

- ``core/services.py`` for scan, data, alert, and persistence boundaries
- ``core/runtime_paths.py`` for state / artifacts / logs directories

This module intentionally stays thin so existing imports continue to work.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable, Optional

from core.backtest import WalkForwardResult
from core.config import CONFIG, SystemConfig
from core.regime import MarketRegime, RegimeTracker
from core.runtime_paths import RUNTIME_PATHS, ensure_runtime_dirs
from core.scorer import TickerResult
from core.services import AlertService, PersistenceService, ScanService
from core.telemetry import setup_logging


VERSION = "14.6-Modular"

ensure_runtime_dirs()
setup_logging(level="INFO", json_log_file=str(RUNTIME_PATHS.telemetry_log_file))
log = logging.getLogger("sovereign")


class _LegacyPatchShim:
    def create_components(self, *args: Any, **kwargs: Any) -> Any:
        from sovereign_improvements import create_components

        return create_components(*args, **kwargs)

    def apply(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from sovereign_improvements import apply

        return apply(*args, **kwargs)

    def __getattr__(self, name: str) -> Callable[..., None]:
        def _noop(*args: object, **kwargs: object) -> None:
            return None

        return _noop


SE_PATCH: Any = _LegacyPatchShim()


@dataclass(frozen=True)
class ServiceBundle:
    persistence: PersistenceService
    scan_service: ScanService
    alert_service: AlertService

    def __iter__(self):
        yield self.scan_service
        yield self.alert_service


def _build_services(
    *,
    data_service: Optional[Any] = None,
    persistence: Optional[PersistenceService] = None,
    probability_gate: Optional[Any] = None,
    capital_scaler: Optional[Any] = None,
    factor_calibrator: Optional[Any] = None,
    alerter: Optional[Any] = None,
    current_nav_provider: Optional[Any] = None,
    alert_service: Optional[AlertService] = None,
) -> ServiceBundle:
    resolved_persistence = persistence or PersistenceService()
    scan_service = ScanService(
        version=VERSION,
        data_service=data_service,
        persistence=resolved_persistence,
        probability_gate=probability_gate,
        capital_scaler=capital_scaler,
        factor_calibrator=factor_calibrator,
        alerter=alerter,
        current_nav_provider=current_nav_provider,
    )
    resolved_alert_service = alert_service or scan_service.create_alert_service()
    return ServiceBundle(
        persistence=resolved_persistence,
        scan_service=scan_service,
        alert_service=resolved_alert_service,
    )


DEFAULT_SERVICES = _build_services()


def _resolve_services(services: Optional[ServiceBundle]) -> ServiceBundle:
    return services or DEFAULT_SERVICES


def __getattr__(name: str) -> Any:
    if name == "PERSISTENCE":
        return DEFAULT_SERVICES.persistence
    if name == "SCAN_SERVICE":
        return DEFAULT_SERVICES.scan_service
    if name == "ALERT_SERVICE":
        return DEFAULT_SERVICES.alert_service
    raise AttributeError(name)


def configure_services(
    *,
    data_service: Optional[Any] = None,
    persistence: Optional[PersistenceService] = None,
    probability_gate: Optional[Any] = None,
    capital_scaler: Optional[Any] = None,
    factor_calibrator: Optional[Any] = None,
    alerter: Optional[Any] = None,
    current_nav_provider: Optional[Any] = None,
    alert_service: Optional[AlertService] = None,
) -> ServiceBundle:
    return _build_services(
        data_service=data_service,
        persistence=persistence,
        probability_gate=probability_gate,
        capital_scaler=capital_scaler,
        factor_calibrator=factor_calibrator,
        alerter=alerter,
        current_nav_provider=current_nav_provider,
        alert_service=alert_service,
    )


def run_scan(
    config: SystemConfig = CONFIG,
    debug: bool = False,
    no_intraday: bool = False,
    regime_tracker: Optional[RegimeTracker] = None,
    regime_override: Optional[str] = None,
    no_ema_filter: bool = False,
    force_score: bool = False,
    services: Optional[ServiceBundle] = None,
) -> tuple[list[TickerResult], list[TickerResult], Optional[MarketRegime]]:
    """
    Compatibility wrapper over ``ScanService.scan``.

    ``no_intraday`` is preserved for the legacy CLI surface even though the
    modular runtime no longer branches on it directly.
    """
    _ = no_intraday
    bundle = _resolve_services(services)
    return bundle.scan_service.scan(
        config=config,
        debug=debug,
        regime_tracker=regime_tracker,
        regime_override=regime_override,
        no_ema_filter=no_ema_filter,
        force_score=force_score,
    )


def _send_alert(
    portfolio: list[TickerResult],
    regime: MarketRegime,
    config: SystemConfig,
    services: Optional[ServiceBundle] = None,
) -> None:
    bundle = _resolve_services(services)
    bundle.alert_service.send_portfolio_summary(portfolio, regime, config)


def run_calibration(*, services: Optional[ServiceBundle] = None) -> None:
    bundle = _resolve_services(services)
    log.info("Starting Platt calibration from %s...", bundle.persistence.paths.trade_log_file)
    bundle.scan_service.run_calibration()


def run_backtest(
    config: SystemConfig = CONFIG,
    train_days: int = 120,
    test_days: int = 20,
    step_days: int = 10,
    out_csv: str = "backtest_results.csv",
    direction: str = "LONG",
    debug: bool = False,
    services: Optional[ServiceBundle] = None,
) -> WalkForwardResult:
    bundle = _resolve_services(services)
    log.info(
        "Starting walk-forward backtest | train=%d test=%d step=%d direction=%s",
        train_days,
        test_days,
        step_days,
        direction,
    )

    results = bundle.scan_service.run_backtest(
        config=config,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        out_csv=out_csv,
        direction=direction,
        debug=debug,
    )

    overall = results.overall
    if overall is None:
        return results

    print("\n" + "=" * 60)
    print(f"  Walk-Forward Backtest Summary  (v{VERSION})")
    print("=" * 60)
    print(f"  Folds          : {overall.n_folds}")
    print(f"  Total trades   : {overall.n_trades}")
    print(f"  Hit rate       : {overall.hit_rate:.1%}")
    print(f"  Mean R         : {overall.mean_r:+.3f}")
    print(f"  Total R        : {overall.total_r:+.2f}")
    print(f"  Sharpe (ann.)  : {overall.sharpe:+.2f}")
    print(f"  Max drawdown   : {overall.max_dd:.2f} R")
    print(f"  Profit factor  : {overall.profit_factor:.2f}")
    print(f"  Expectancy R   : {overall.expectancy_r:+.4f}")
    print("=" * 60)

    if overall.fold_stats:
        print("\nPer-fold breakdown:")
        print(
            f"  {'Fold':>4}  {'Start':>12}  {'End':>12}  "
            f"{'Trades':>6}  {'Hit%':>5}  {'MeanR':>6}  {'TotalR':>7}"
        )
        for fold_stats in overall.fold_stats:
            print(
                f"  {fold_stats.fold:>4}  {str(fold_stats.start_date.date()):>12}  "
                f"{str(fold_stats.end_date.date()):>12}  {fold_stats.n_trades:>6}  "
                f"{fold_stats.hit_rate:>4.0%}  {fold_stats.mean_r:>+6.3f}  "
                f"{fold_stats.total_r:>+7.2f}"
            )

    if results.trades:
        print(f"\nTrade log saved -> {bundle.persistence.artifact_path(out_csv)}")
    else:
        log.warning("No trades generated - check min_prob threshold and data quality.")

    return results


def main() -> None:
    """Backward compatibility wrapper that delegates to the canonical CLI."""
    import sys

    sys.modules.setdefault("screener_v14_modular", sys.modules[__name__])
    from run import main as run_main

    run_main(prog="run.py")


if __name__ == "__main__":
    main()
