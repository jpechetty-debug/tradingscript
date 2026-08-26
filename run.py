"""
run.py
======
Canonical CLI entry point for the Sovereign Engine.

All argparse handling and main() orchestration lives here.
screener_v14_modular.py is the *library* — it exposes run_scan(),
_send_alert(), run_calibration(), and run_backtest() as importable
functions.  This module owns the CLI surface.

Usage:
    python run.py
    python run.py --watch 15
    python run.py --debug
    python run.py --version
    python run.py --no-telegram
    python run.py --backtest
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Callable

from core.regime import RegimeTracker
from core.runtime_components import create_runtime_components
from core.services import PersistenceService
from screener_v14_modular import (
    CONFIG,
    VERSION,
    _send_alert,
    configure_services,
    run_backtest,
    run_calibration,
    run_scan,
)

log = logging.getLogger("sovereign")


def _build_current_nav_provider(persistence: PersistenceService) -> Callable[[], float | None]:
    def _current_nav() -> float | None:
        snapshot = persistence.load_portfolio_state()
        return snapshot.current_nav if snapshot is not None else None

    return _current_nav


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    from core.telemetry import setup_logging
    from core.runtime_paths import RUNTIME_PATHS

    # Ensure logging is wired up even if run.py is invoked directly
    setup_logging(level="INFO", json_log_file=str(RUNTIME_PATHS.telemetry_log_file))

    parser = argparse.ArgumentParser(prog=prog, description=f"Sovereign Engine v{VERSION}")
    parser.add_argument("--watch",       type=int,  default=None, metavar="MINUTES")
    parser.add_argument("--version",     action="store_true")
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--no-intraday", action="store_true")
    parser.add_argument("--calibrate",   action="store_true", help="Re-fit Platt A/B from the runtime trade log")
    parser.add_argument("--backtest",      action="store_true",  help="Run walk-forward backtest and exit")
    parser.add_argument("--bt-train",      type=int, default=120, metavar="DAYS", help="Backtest train window (default 120)")
    parser.add_argument("--bt-test",       type=int, default=20,  metavar="DAYS", help="Backtest test window (default 20)")
    parser.add_argument("--bt-step",       type=int, default=10,  metavar="DAYS", help="Backtest step between folds (default 10)")
    parser.add_argument("--bt-out",        type=str, default="backtest_results.csv", metavar="FILE", help="CSV output path (relative paths go under artifacts/)")
    parser.add_argument("--bt-direction",  type=str, default="LONG", choices=["LONG","SHORT","BOTH"], help="Trade direction")
    parser.add_argument("--regime-override", type=str, default=None, help="Force a specific regime (TREND_UP, RANGE, etc.)")
    parser.add_argument("--no-ema-filter", action="store_true", help="Disable EMA200 structural filter for debugging")
    parser.add_argument("--force-score", action="store_true", help="Force scoring of all tickers (bypass BULL/BEAR directional gates)")
    args = parser.parse_args(args=argv)

    if args.version:
        print(f"Sovereign Engine v{VERSION}")
        return

    config = CONFIG
    persistence = PersistenceService()
    portfolio_state = persistence.load_portfolio_state()
    portfolio_peak = (
        portfolio_state.peak_nav
        if portfolio_state is not None and portfolio_state.peak_nav is not None
        else portfolio_state.current_nav
        if portfolio_state is not None
        else 1_000_000.0
    )
    components = create_runtime_components(portfolio_peak=portfolio_peak)
    services = configure_services(
        persistence=persistence,
        probability_gate=components.gate,
        capital_scaler=components.scaler,
        factor_calibrator=components.calibrator,
        alerter=components.alerter,
        current_nav_provider=_build_current_nav_provider(persistence),
    )
    regime_tracker = RegimeTracker()

    def _step() -> None:
        results, portfolio, regime = run_scan(
            config=config,
            debug=args.debug,
            no_intraday=args.no_intraday,
            regime_tracker=regime_tracker,
            regime_override=args.regime_override,
            no_ema_filter=args.no_ema_filter,
            force_score=args.force_score,
            services=services,
        )
        if not args.no_telegram and portfolio and regime:
            _send_alert(portfolio, regime, config, services=services)

        # Feed completed trades into the calibrator buffer for live updates.
        # In a real live environment, record actual closed trades here.
        if components.calibrator:
            pass  # placeholder — wire in broker PnL events here

    if args.backtest:
        run_backtest(
            config=config,
            train_days=args.bt_train,
            test_days=args.bt_test,
            step_days=args.bt_step,
            out_csv=args.bt_out,
            direction=args.bt_direction,
            debug=args.debug,
            services=services,
        )
        return
    elif args.calibrate:
        run_calibration(services=services)
    elif args.watch:
        log.info("Watch mode — scanning every %d minutes", args.watch)
        while True:
            try:
                _step()
                time.sleep(args.watch * 60)
            except KeyboardInterrupt:
                log.info("Watch mode stopped")
                sys.exit(0)
            except Exception as e:
                log.error("Scan error: %s", e, exc_info=args.debug)
                time.sleep(args.watch * 60)
    else:
        _step()


if __name__ == "__main__":
    main()
