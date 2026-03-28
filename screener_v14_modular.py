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


try:
    import sovereign_improvements as _SE_PATCH_MODULE

    SE_PATCH: Any = _SE_PATCH_MODULE
    _SE_PATCH_AVAILABLE = True
except Exception as _se_err:

    class _NoOpPatch:
        def apply(self, *args: object, **kwargs: object) -> dict[str, Any]:
            return {}

        def __getattr__(self, name: str) -> Callable[..., None]:
            def _noop(*args: object, **kwargs: object) -> None:
                return None

            return _noop

    SE_PATCH = _NoOpPatch()
    _SE_PATCH_AVAILABLE = False
    _SE_PATCH_LOAD_ERROR = _se_err


if not _SE_PATCH_AVAILABLE:
    log.warning(
        "sovereign_improvements could not be loaded (%s: %s). Running without patch components.",
        type(_SE_PATCH_LOAD_ERROR).__name__,
        _SE_PATCH_LOAD_ERROR,
    )


PERSISTENCE = PersistenceService()
SCAN_SERVICE = ScanService(version=VERSION, persistence=PERSISTENCE)
ALERT_SERVICE = AlertService(version=VERSION)


def run_scan(
    config: SystemConfig = CONFIG,
    debug: bool = False,
    no_intraday: bool = False,
    patch: Optional[dict[str, Any]] = None,
    regime_tracker: Optional[RegimeTracker] = None,
    regime_override: Optional[str] = None,
    no_ema_filter: bool = False,
    force_score: bool = False,
) -> tuple[list[TickerResult], list[TickerResult], Optional[MarketRegime]]:
    """
    Compatibility wrapper over ``ScanService.scan``.

    ``no_intraday`` is preserved for the legacy CLI surface even though the
    modular runtime no longer branches on it directly.
    """
    _ = no_intraday
    return SCAN_SERVICE.scan(
        config=config,
        debug=debug,
        patch=patch,
        regime_tracker=regime_tracker,
        regime_override=regime_override,
        no_ema_filter=no_ema_filter,
        force_score=force_score,
    )


def _send_alert(
    portfolio: list[TickerResult],
    regime: MarketRegime,
    config: SystemConfig,
    patch: Optional[dict[str, Any]] = None,
) -> None:
    ALERT_SERVICE.send_portfolio_summary(portfolio, regime, config, patch=patch)


def run_calibration() -> None:
    log.info("Starting Platt calibration from %s...", PERSISTENCE.paths.trade_log_file)
    SCAN_SERVICE.run_calibration()


def run_backtest(
    config: SystemConfig = CONFIG,
    train_days: int = 120,
    test_days: int = 20,
    step_days: int = 10,
    out_csv: str = "backtest_results.csv",
    direction: str = "LONG",
    debug: bool = False,
) -> WalkForwardResult:
    log.info(
        "Starting walk-forward backtest | train=%d test=%d step=%d direction=%s",
        train_days,
        test_days,
        step_days,
        direction,
    )

    results = SCAN_SERVICE.run_backtest(
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
        print(f"\nTrade log saved -> {PERSISTENCE.artifact_path(out_csv)}")
    else:
        log.warning("No trades generated - check min_prob threshold and data quality.")

    return results


def main() -> None:
    """Backward compatibility wrapper - calls run.main()."""
    from run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
