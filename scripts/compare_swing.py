"""One fixed exploratory comparison; previously inspected periods are not fresh holdouts."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.backtest import SWING_COST_MODEL  # noqa: E402
from core.config import CONFIG  # noqa: E402
from core.swing_replay import replay_swing  # noqa: E402
from scripts.validate_performance import summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/performance_validation/historical.pkl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/swing_comparison"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.ERROR)
    raw = pd.read_pickle(args.snapshot)
    dates = raw[CONFIG.BENCHMARK].index
    config = replace(CONFIG, INTRADAY_ENABLED=False, PLATT_A=-4., PLATT_B=2., MAX_WORKERS=4)
    records: dict[str, Any] = {}
    specs = [("baseline_development", False, dates[252], dates[-253], 1),
             ("rules_development", True, dates[252], dates[-253], 1),
             ("baseline_seen_validation", False, dates[-252], dates[-1], 1),
             ("rules_seen_validation", True, dates[-252], dates[-1], 1),
             ("rules_seen_validation_2x_slippage", True, dates[-252], dates[-1], 2)]
    for name, enabled, start, end, multiplier in specs:
        print(f"Running {name}: {start.date()} to {end.date()}", flush=True)
        model = replace(SWING_COST_MODEL, slippage_pct=config.SLIPPAGE_BPS * multiplier / 10000.)
        replay = replay_swing(raw, replace(config, SWING_SETUP_ENABLED=enabled), start, end, model)
        stats = summarize(replay.result, np.random.default_rng(20261002))
        stats.update(replay.metrics)
        stats["by_strategy"] = {strategy: {
            "trades": len(group), "mean_net_r": float(np.mean([t.r_multiple for t in group]))
        } for strategy in sorted({t.strategy_id for t in replay.result.trades})
            if (group := [t for t in replay.result.trades if t.strategy_id == strategy])}
        records[name] = stats
        replay.result.to_csv(str(args.out / f"{name}_trades.csv"))
        replay.equity.to_csv(args.out / f"{name}_equity.csv")
        print(f"{name}: {stats['n_trades']} trades, mean R {stats['mean_r']}, NAV return {stats['total_return']:.2%}", flush=True)
    report: dict[str, Any] = {"snapshot": str(args.snapshot), "data_start": str(dates[0]), "data_end": str(dates[-1]),
              "protocol": "Fixed rules, no parameter search. Daily close signals, next-session opening fills, cash funding, initial-risk budget, sector and correlation limits. Identical fixed fractional sizing for both policies. No live calibration. End-of-period liquidation assumed.",
              "limitations": ["The validation period was viewed before this change; it is exploratory, not untouched evidence",
                              "Current constituents and selected 60-stock subset introduce selection/survivorship bias",
                              "Adjusted OHLC data cannot establish intrabar order or actual broker fills; same-bar stop wins conservatively",
                              "Fixed heuristic probabilities and modeled expectancy are uncalibrated",
                              "No earnings/event feed or point-in-time membership data is available",
                              "Slippage and transaction fees are research assumptions; prospective paper validation remains required"],
              "runs": records}
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines: list[str] = ["# Swing policy comparison", "", report["protocol"], "",
             "| Policy / period | Trades | Mean net R | NAV return | NAV max drawdown | Mean-R 95% interval |",
             "|---|---:|---:|---:|---:|---|"]
    for name, stats in records.items():
        lines.append(f"| {name} | {stats['n_trades']} | {stats['mean_r']} | {stats['total_return']:.2%} | {stats['max_drawdown']:.2%} | {stats['monthly_block_mean_r_ci95']} |")
    lines += ["", "Limitations:", ""] + [f"- {item}" for item in report["limitations"]]
    lines += ["", "The rules policy is research-only. This comparison cannot establish that it is the best strategy or authorize live deployment."]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
