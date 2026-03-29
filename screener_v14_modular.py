"""
screener_v14_modular.py
=======================
Backward-compatible library entry point for the Sovereign Engine.

All runtime orchestration lives in ``core/``:

- ``core/services.py``       — :class:`ScanService`, :class:`AlertService`,
                               :class:`PersistenceService`, :class:`ServiceBundle`,
                               :func:`configure_services`
- ``core/runtime_paths.py``  — state / artifacts / logs directory layout
- ``core/backtest.py``       — walk-forward engine + :class:`TransactionCostModel`

This module re-exports the most-used symbols and provides thin CLI-facing
wrappers (``run_scan``, ``run_backtest``, ``run_calibration``) so that
``run.py`` and external tooling that did ``from screener_v14_modular import …``
continue to work without modification.

* Module-level ``ensure_runtime_dirs()`` / ``setup_logging()`` side-effects —
  these ran on every import, including in tests.  They are now called once
  inside ``main()``.
* Module-level ``_DEFAULT_SERVICES`` lazy singleton — constructed on first
  access via ``__getattr__``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from core.backtest import WalkForwardResult
from core.config import CONFIG, SystemConfig
from core.regime import MarketRegime, RegimeTracker
from core.runtime_paths import RUNTIME_PATHS, ensure_runtime_dirs
from core.scorer import TickerResult
from core.services import (
    ServiceBundle,         # re-exported for backward compat
    configure_services,    # re-exported for backward compat
)
from core.telemetry import setup_logging

VERSION = "14.6-Modular"

log = logging.getLogger("sovereign")

# ---------------------------------------------------------------------------
# Lazy default service bundle — constructed on first access, not at import.
# ---------------------------------------------------------------------------

_DEFAULT_SERVICES: Optional[ServiceBundle] = None


def _get_default_services() -> ServiceBundle:
    global _DEFAULT_SERVICES
    if _DEFAULT_SERVICES is None:
        _DEFAULT_SERVICES = configure_services(version=VERSION)
    return _DEFAULT_SERVICES


def __getattr__(name: str) -> Any:
    if name == "PERSISTENCE":
        return _get_default_services().persistence
    if name == "SCAN_SERVICE":
        return _get_default_services().scan_service
    if name == "ALERT_SERVICE":
        return _get_default_services().alert_service
    raise AttributeError(name)


def _resolve_services(services: Optional[ServiceBundle]) -> ServiceBundle:
    return services if services is not None else _get_default_services()


# ---------------------------------------------------------------------------
# Public compatibility wrappers
# ---------------------------------------------------------------------------

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
    Compatibility wrapper over :meth:`ScanService.scan`.

    ``no_intraday`` is accepted but ignored — the modular runtime no longer
    branches on it directly.
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
    log.info("Starting Platt calibration from %s…", bundle.persistence.paths.trade_log_file)
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
        train_days, test_days, step_days, direction,
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

    # Compute gross total R for comparison (costs were deducted per-trade)
    gross_total = sum(t.gross_r_multiple for t in results.trades) if results.trades else 0.0
    total_friction = sum(t.friction_r_applied for t in results.trades) if results.trades else 0.0

    print("\n" + "=" * 64)
    print(f"  Walk-Forward Backtest Summary  (v{VERSION})")
    print("=" * 64)
    print(f"  Folds          : {overall.n_folds}")
    print(f"  Total trades   : {overall.n_trades}")
    print(f"  Hit rate       : {overall.hit_rate:.1%}")
    print(f"  Mean R (net)   : {overall.mean_r:+.3f}")
    print(f"  Total R (net)  : {overall.total_r:+.2f}  "
          f"(gross {gross_total:+.2f}, cost drag {total_friction:.2f} R)")
    print(f"  Sharpe (ann.)  : {overall.sharpe:+.2f}")
    print(f"  Max drawdown   : {overall.max_dd:.2f} R")
    print(f"  Profit factor  : {overall.profit_factor:.2f}")
    print(f"  Expectancy R   : {overall.expectancy_r:+.4f}")
    print("=" * 64)

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
        log.warning("No trades generated — check min_prob threshold and data quality.")

    return results


def main() -> None:
    """Backward-compatibility wrapper that delegates to the canonical CLI."""
    import sys

    # Side effects that used to run at module level now happen here, once.
    ensure_runtime_dirs()
    setup_logging(level="INFO", json_log_file=str(RUNTIME_PATHS.telemetry_log_file))

    sys.modules.setdefault("screener_v14_modular", sys.modules[__name__])
    from run import main as run_main

    run_main(prog="run.py")


if __name__ == "__main__":
    main()
