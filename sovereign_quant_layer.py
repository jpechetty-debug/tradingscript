"""
Compatibility CLI for the quantitative utilities now backed by the modular runtime.

Historically this file carried a separate analytics stack. The live engine now
stores calibration and portfolio state under ``core/`` services, so this script
acts as a thin wrapper around those stateful runtime components instead of
maintaining a second implementation tree.
"""

from __future__ import annotations

import argparse

from core.runtime_components import TieredCapitalScaler
from core.services import PersistenceService, ScanService

VERSION = "14.6-Modular"


def _resolved_nav(
    persistence: PersistenceService,
    *,
    current_nav: float | None,
    peak_nav: float | None,
) -> tuple[float, float]:
    snapshot = persistence.load_portfolio_state()
    resolved_current = (
        current_nav
        if current_nav is not None and current_nav > 0
        else snapshot.current_nav
        if snapshot is not None
        else 1_000_000.0
    )
    resolved_peak = (
        peak_nav
        if peak_nav is not None and peak_nav > 0
        else snapshot.peak_nav
        if snapshot is not None and snapshot.peak_nav is not None
        else resolved_current
    )
    return resolved_current, resolved_peak


def _run_calibration(persistence: PersistenceService, trade_log: str) -> None:
    if trade_log != str(persistence.paths.trade_log_file):
        print(
            "Note: custom --trade-log is deprecated here; calibration now uses "
            f"the modular runtime trade log at {persistence.paths.trade_log_file}."
        )
    ScanService(version=VERSION, persistence=persistence).run_calibration()
    print(f"Platt calibration updated -> {persistence.paths.platt_calibration_file}")
    print(f"Factor weights path      -> {persistence.paths.factor_weights_file}")


def _print_capital_plan(
    *,
    persistence: PersistenceService,
    regime: str,
    current_nav: float | None,
    peak_nav: float | None,
) -> None:
    resolved_current, resolved_peak = _resolved_nav(
        persistence,
        current_nav=current_nav,
        peak_nav=peak_nav,
    )
    scaler = TieredCapitalScaler(portfolio_peak=resolved_peak)
    fraction = scaler.capital_fraction(resolved_current, regime)
    drawdown = scaler.current_drawdown(resolved_current)

    print(f"Regime        : {regime}")
    print(f"Current NAV   : INR {resolved_current:,.0f}")
    print(f"Peak NAV      : INR {resolved_peak:,.0f}")
    print(f"Drawdown      : {drawdown:.1%}")
    print(f"Capital usage : {fraction:.0%}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Legacy quantitative wrapper over the modular Sovereign runtime."
    )
    parser.add_argument("--trade-log", default="state/trade_log.json")
    parser.add_argument("--full", action="store_true", help="Run modular calibration.")
    parser.add_argument(
        "--capital-plan",
        action="store_true",
        help="Print live capital scaling for the given regime.",
    )
    parser.add_argument("--regime", default="TREND_UP")
    parser.add_argument("--current-nav", type=float, default=None)
    parser.add_argument("--peak-nav", type=float, default=None)
    args = parser.parse_args(args=argv)

    if not args.full and not args.capital_plan:
        parser.error("choose --full or --capital-plan")

    persistence = PersistenceService()

    if args.full:
        _run_calibration(persistence, args.trade_log)

    if args.capital_plan:
        _print_capital_plan(
            persistence=persistence,
            regime=args.regime,
            current_nav=args.current_nav,
            peak_nav=args.peak_nav,
        )


if __name__ == "__main__":
    main()
