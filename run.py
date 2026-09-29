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
import atexit
import logging
import sys
import time
from core.application import build_application_runtime
from core.regime import RegimeTracker
from core.runtime_components import create_runtime_components
from core.scorer import TickerResult
from core.snapshots import build_scan_snapshot, write_json_atomic
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


def main(argv: list[str] | None = None, prog: str | None = None) -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

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
    parser.add_argument("--bt-step",       type=int, default=20,  metavar="DAYS", help="Backtest step between folds (default 20)")
    parser.add_argument("--bt-out",        type=str, default="backtest_results.csv", metavar="FILE", help="CSV output path (relative paths go under artifacts/)")
    parser.add_argument("--bt-direction",  type=str, default="LONG", choices=["LONG","SHORT","BOTH"], help="Trade direction")
    parser.add_argument("--bt-horizon",    type=str, default="SWING", choices=["SWING","INTRADAY","BOTH"], help="Backtest horizon filter (default SWING)")
    parser.add_argument("--regime-override", type=str, default=None, help="Force a specific regime (TREND_UP, RANGE, etc.)")
    parser.add_argument("--no-ema-filter", action="store_true", help="Disable EMA200 structural filter for debugging")
    parser.add_argument("--force-score", action="store_true", help="Force scoring of all tickers (bypass BULL/BEAR directional gates)")
    args = parser.parse_args(args=argv)

    if args.version:
        print(f"Sovereign Engine v{VERSION}")
        return

    config = CONFIG
    runtime = build_application_runtime(
        version=VERSION,
        component_factory=create_runtime_components,
        service_factory=configure_services,
    )
    atexit.register(runtime.close)
    components = runtime.components
    services = runtime.services
    regime_tracker = RegimeTracker()
    last_portfolio: list[TickerResult] = []
    last_trade_count: int = 0

    def _step() -> None:
        nonlocal last_portfolio, last_trade_count
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

        # Persist latest scan for FastAPI UI server through the shared serializer.
        try:
            from datetime import datetime, timezone
            target = services.persistence.paths.state_dir / "latest_scan.json"
            payload = build_scan_snapshot(
                scan_time=datetime.now(timezone.utc).isoformat(),
                candidates=results,
                portfolio=portfolio,
                sector_rs=services.scan_service.get_last_sector_rs(),
                regime_info={
                    "regime": str(regime.regime.value if hasattr(regime.regime, "value") else regime.regime),
                    "label": regime.label,
                    "confidence": regime.confidence,
                    "confirmed": regime.confirmed,
                } if regime else None,
                config=config,
            )
            write_json_atomic(target, payload)
        except Exception as exc:
            log.debug("Failed to persist latest_scan.json in run.py: %s", exc)


        # Feed completed trades into the calibrator buffer for live updates.
        # Sourcing outcomes from the persistence layer's most recently closed trades.
        if components.calibrator:
            trade_log = services.persistence.load_trade_log()
            if len(trade_log) > last_trade_count:
                for trade in trade_log[last_trade_count:]:
                    factors = trade.get("factors")
                    pnl = trade.get("pnl")
                    if isinstance(factors, dict) and pnl is not None:
                        components.calibrator.record_trade(factors, float(pnl))
                last_trade_count = len(trade_log)

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
            horizon_filter=args.bt_horizon,
        )
        return
    elif args.calibrate:
        run_calibration(services=services)
    elif args.watch:
        log.info("Watch mode — scanning every %d minutes", args.watch)
        try:
            while True:
                try:
                    _step()
                except Exception as e:
                    log.error("Scan error: %s", e, exc_info=args.debug)

                time.sleep(args.watch * 60)
        except KeyboardInterrupt:
            log.info("Watch mode stopped")
            sys.exit(0)
    else:
        _step()


if __name__ == "__main__":
    main()
