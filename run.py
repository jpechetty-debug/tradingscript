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

from core.regime import RegimeTracker
from screener_v14_modular import (
    CONFIG,
    SE_PATCH,
    VERSION,
    _send_alert,
    run_backtest,
    run_calibration,
    run_scan,
)

log = logging.getLogger("sovereign")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Sovereign Engine v{VERSION}")
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
    args = parser.parse_args()

    if args.version:
        print(f"Sovereign Engine v{VERSION}")
        return

    config = CONFIG

    # Initialise patch components once at startup; pass them into each
    # run_scan() call rather than storing them in a module-level global.
    patch = SE_PATCH.apply(None, portfolio_peak=1_000_000)
    regime_tracker = RegimeTracker()

    def _step() -> None:
        results, portfolio, regime = run_scan(
            config=config,
            debug=args.debug,
            no_intraday=args.no_intraday,
            patch=patch,
            regime_tracker=regime_tracker,
            regime_override=args.regime_override,
            no_ema_filter=args.no_ema_filter,
            force_score=args.force_score,
        )
        if not args.no_telegram and portfolio and regime:
            _send_alert(portfolio, regime, config, patch=patch)

        # Feed completed trades into the calibrator buffer for live updates.
        # In a real live environment, record actual closed trades here.
        if patch.get("calibrator"):
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
        )
        return
    elif args.calibrate:
        run_calibration()
    elif args.watch:
        log.info("Watch mode — scanning every %d minutes", args.watch)
        while True:
            try:
                _step()
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
