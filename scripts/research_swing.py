"""Retrospective expanding-history validation and a fixed, unselected sensitivity grid."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.backtest import SWING_COST_MODEL  # noqa: E402
from core.config import CONFIG, SystemConfig  # noqa: E402
from core.indicators import add_indicators  # noqa: E402
from core.swing_analytics import evidence, research_parameters  # noqa: E402
from core.swing_replay import replay_swing  # noqa: E402


def validation_windows(dates: pd.DatetimeIndex, train_start: pd.Timestamp,
                       validate_from: pd.Timestamp) -> list[dict[str, Any]]:
    windows = []
    available = dates[dates >= validate_from]
    for quarter in available.to_period("Q").unique():
        validation = available[available.to_period("Q") == quarter]
        training = dates[(dates >= train_start) & (dates < validation[0])]
        if len(training) < 252:
            continue
        windows.append(dict(label=str(quarter), train_start=training[0], train_end=training[-1],
                            validation_start=validation[0], validation_end=validation[-1],
                            training_sessions=len(training), validation_sessions=len(validation),
                            partial_quarter=validation[-1].normalize() < quarter.end_time.normalize() - pd.Timedelta(days=4)))
    return windows


def sensitivity_grid(config: SystemConfig) -> list[tuple[str, SystemConfig]]:
    specs: list[tuple[str, dict[str, Any]]] = [("fixed_default", {})]
    for value in (15, 25):
        specs.append((f"breakout_lookback_{value}", {"SWING_BREAKOUT_LOOKBACK": value}))
    for volume in (1.1, 1.3):
        specs.append((f"breakout_rvol_{volume}", {"SWING_BREAKOUT_RVOL": volume}))
    for value in (10, 15, 20):
        specs.append((f"both_time_exits_{value}", {"SWING_BREAKOUT_TIME_BARS": value, "SWING_PULLBACK_TIME_BARS": value}))
    for width in (.5, 1., 1.5):
        specs.append((f"atr_stop_{width}", {"SWING_STOP_MODE": "ATR", "SWING_STOP_ATR_MULT": width}))
    return [(name, replace(config, **params, SWING_RESEARCH_TAG="" if not params else f"_RESEARCH_{name}"))
            for name, params in specs]


def _number(value: Any, percent: bool = False) -> str:
    if value is None:
        return "—"
    return f"{value:.2%}" if percent else f"{value:.4f}"


def write_report(report: dict[str, Any], out: Path) -> None:
    lines = ["# Swing research evidence", "", report["status"], "", report["protocol"], "",
             "## Quarterly validation", "", "| Window | Policy | Trades | Mean R | Win % | NAV return | Drawdown | Exposure | Daily NAV Sharpe |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for window in report["walk_forward"]:
        for policy, stat in window["policies"].items():
            lines.append(f"| {window['label']} | {policy} | {stat['trades']} | {_number(stat['mean_r'])} | {_number(stat['win_rate'], True)} | {_number(stat['total_return'], True)} | {_number(stat['max_drawdown'], True)} | {_number(stat['average_exposure'], True)} | {_number(stat['daily_nav_sharpe'])} |")
    lines += ["", "Training and validation timestamps, partial-quarter flags and full diagnostics are in report.json. Quarter books start flat and liquidate at their boundary; the continuous run below preserves positions across quarters.", "",
              "## Continuous validation portfolio", "", "| Policy | CAGR | Exposure | CAGR / exposure¹ | Return / exposure¹ | Benchmark CAGR | Beta | Annual alpha² | Information ratio | Exposure-matched excess NAV return³ |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for policy, stat in report["continuous"].items():
        keys = ("cagr", "average_exposure", "exposure_adjusted_cagr", "return_per_exposure", "benchmark_cagr", "beta", "annualized_alpha", "information_ratio", "exposure_matched_active_return")
        lines.append(f"| {policy} | " + " | ".join(_number(stat[k], k not in ("beta", "information_ratio")) for k in keys) + " |")
    rules = report["continuous"]["rules"]
    lines += ["", "¹ Descriptive ratios, not attainable fully invested returns. ² Daily OLS intercept ×252 with cash yield/risk-free rate set to zero; no statistical significance claim. ³ Benchmark overnight return uses prior-close exposure and its daytime return uses post-allocation opening exposure. Intrabar exit time is unknown, so this is an exposure proxy.", "",
              "## Setup and regime attribution (continuous rules)"]
    for field, groups in rules["attribution"].items():
        lines += ["", f"### {field}", "", "| Group | Trades | Mean R | Win % | Average hold | Net INR P&L | Stop frequency | Mean excess trade return |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label, stat in groups.items():
            lines.append(f"| {label} | {stat['trades']} | {_number(stat['mean_r'])} | {_number(stat['win_rate'], True)} | {_number(stat['average_hold'])} | {_number(stat['net_pnl'])} | {_number(stat['stop_out_frequency'], True)} | {_number(stat.get('mean_excess_return'), True)} |")
    lines += ["", "## Return distribution and benchmark-relative trades", "", "```json",
              json.dumps({k: rules.get(k) for k in ("percentiles", "skewness", "excess_kurtosis", "largest_winner", "largest_loser", "mean_excess_return", "median_excess_return", "fraction_outperforming_benchmark", "monthly_mean_r_ci95")}, indent=2), "```", "",
              "Stock returns are net of modeled costs. Holding-period benchmark returns use entry open to exit close, except opening-gap exits use benchmark open. An intraday stop/target exit therefore uses a closing-price benchmark proxy.", "",
              "## Signal decisions", "", "```json", json.dumps(rules["diagnostics"], indent=2), "```", "",
              "Counts refer to scorer-qualified daily opportunities. Terminal reasons are exclusive; resizing counters describe fills reduced rather than rejected. Per-signal decisions are saved as CSV.", "",
              "## Prespecified sensitivity grid (development only)", "", "| Variant | Trades | Mean R | NAV return | Drawdown | Exposure | Stop frequency |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, stat in report["sensitivity"].items():
        lines.append(f"| {name} | {stat['trades']} | {_number(stat['mean_r'])} | {_number(stat['total_return'], True)} | {_number(stat['max_drawdown'], True)} | {_number(stat['average_exposure'], True)} | {_number(stat['stop_out_frequency'], True)} |")
    lines += ["", "Defaults remain fixed. The grid is one-factor-at-a-time, not a search or a ranking. Wider stops also change target distances and eligible setups; they are policy variants, not isolated causal estimates of stop width.", "", "## Limitations", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/performance_validation/historical.pkl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/swing_research"))
    parser.add_argument("--train-start", default="2022-01-01")
    parser.add_argument("--validate-from", default="2024-01-01")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    args.out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_pickle(args.snapshot)
    config = replace(CONFIG, SWING_SETUP_ENABLED=True, INTRADAY_ENABLED=False, MAX_WORKERS=4,
                     SWING_BREAKOUT_LOOKBACK=20, SWING_BREAKOUT_RVOL=1.2, SWING_BREAKOUT_TIME_BARS=15,
                     SWING_PULLBACK_TIME_BARS=10, SWING_STOP_MODE="STRUCTURE", SWING_STOP_BUFFER_ATR=.1,
                     SWING_RESEARCH_TAG="", PLATT_A=-4., PLATT_B=2.)
    prepared = {t: add_indicators(frame, config) for t, frame in raw.items()}
    dates = raw[config.BENCHMARK].index
    windows = validation_windows(dates, pd.Timestamp(args.train_start), pd.Timestamp(args.validate_from))
    if not windows:
        parser.error("No validation quarters with at least 252 prior training sessions")
    model = replace(SWING_COST_MODEL, slippage_pct=config.SLIPPAGE_BPS / 10000.)

    def run(name: str, policy: SystemConfig, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
        print(f"Running {name}: {start.date()} to {end.date()}", flush=True)
        replay = replay_swing(raw, policy, start, end, model, prepared=prepared, warm_history=True)
        stats = evidence(replay)
        stats["parameters"] = research_parameters(policy)
        replay.result.to_csv(str(args.out / f"{name}_trades.csv"))
        replay.equity.to_csv(args.out / f"{name}_equity.csv")
        replay.diagnostics.to_csv(args.out / f"{name}_decisions.csv", index=False)
        print(f"{name}: {stats['trades']} trades; NAV {stats['total_return']:.2%}", flush=True)
        return stats

    report: dict[str, Any] = {
        "status": "RETROSPECTIVE, PREVIOUSLY VIEWED DATA — NOT PROSPECTIVE EVIDENCE",
        "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
        "protocol": "Expanding prior-history / nonoverlapping quarterly validation; fixed rules, no refitting or parameter selection. Prior data warms causal indicators, regime confirmation and trailing volatility context. Prior-close signals may fill at the first validation open. No outcome labels are fitted, so no label purge is needed. Adjacent market periods are not assumed statistically independent.",
        "walk_forward": [], "continuous": {}, "sensitivity": {},
        "limitations": ["All historical dates were already available/inspected; re-splitting cannot create a fresh holdout.",
                        "Current constituents and the 60-stock subset retain survivorship/selection bias; point-in-time membership is unavailable.",
                        "Daily adjusted bars, index-price benchmark and modeled fills/fees are proxies, not actual executable or dividend-inclusive total returns.",
                        "Quarter-boundary liquidation differs from a continuous trading book; small/partial windows have limited evidence.",
                        "The sensitivity grid diagnoses instability without selecting a winner or changing deployed defaults.",
                        "Regime groups describe entry conditions; missing groups are absence of evidence and do not justify automatic filter tightening.",
                        "Prospective paper evidence requires future observed signals and linked fill/exit events; historical events are never relabeled prospective."]}
    for window in windows:
        window["policies"] = {policy: run(f"{window['label']}_{policy}", replace(config, SWING_SETUP_ENABLED=enabled), window["validation_start"], window["validation_end"])
                              for policy, enabled in (("baseline", False), ("rules", True))}
        report["walk_forward"].append(window)
    report["continuous"] = {policy: run(f"continuous_{policy}", replace(config, SWING_SETUP_ENABLED=enabled), windows[0]["validation_start"], dates[-1])
                            for policy, enabled in (("baseline", False), ("rules", True))}
    development_end = dates[dates < windows[0]["validation_start"]][-1]
    development_start = dates[dates >= pd.Timestamp(args.train_start)][0]
    if not args.skip_sensitivity:
        report["sensitivity"] = {name: run(f"sensitivity_{name}", policy, development_start, development_end)
                                 for name, policy in sensitivity_grid(config)}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str, allow_nan=False), encoding="utf-8")
    write_report(report, args.out)


if __name__ == "__main__":
    main()
