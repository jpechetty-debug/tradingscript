"""Reproducible chronological swing evaluation with a final holdout and cost stress."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.backtest import SWING_COST_MODEL, WalkForwardResult, walk_forward  # noqa: E402
from core.config import CONFIG  # noqa: E402
from core.universe import SECTORS  # noqa: E402


def summarize(result: WalkForwardResult, rng: np.random.Generator) -> dict[str, Any]:
    stats = asdict(result.overall)
    stats.pop("fold_stats", None)
    records = result.to_dataframe()
    if result.trades:
        grouped: dict[str, list[float]] = {}
        for trade in result.trades:
            grouped.setdefault(str(trade.entry_date.to_period("M")), []).append(trade.r_multiple)
        blocks = list(grouped.values())
        means = []
        for _ in range(2000):
            sample = [blocks[int(i)] for i in rng.integers(0, len(blocks), len(blocks))]
            means.append(sum(sum(b) for b in sample) / sum(len(b) for b in sample))
        stats["monthly_block_mean_r_ci95"] = np.quantile(means, [.025, .975]).tolist()
        stats["independent_month_blocks"] = len(blocks)
    else:
        stats["monthly_block_mean_r_ci95"] = None
        stats["independent_month_blocks"] = 0
    stats["record_count"] = len(records)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", type=int, default=60)
    parser.add_argument("--period", default="5y")
    parser.add_argument("--out", type=Path, default=Path("artifacts/performance_validation"))
    parser.add_argument("--offline", action="store_true", help="Reuse the saved historical snapshot")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out / "historical.pkl"
    logging.basicConfig(level=logging.ERROR)
    pools = [sorted(tickers) for _, tickers in sorted(SECTORS.items())]
    symbols: list[str] = []
    for index in range(max(map(len, pools))):
        for pool in pools:
            if index < len(pool) and pool[index] not in symbols:
                symbols.append(pool[index])
    symbols = symbols[:args.symbols]
    if args.offline:
        raw: dict[str, pd.DataFrame] = pd.read_pickle(snapshot)
    else:
        download = yf.download(symbols + [CONFIG.BENCHMARK], period=args.period,
                               auto_adjust=True, progress=False, threads=4)
        raw = {}
        for symbol in symbols + [CONFIG.BENCHMARK]:
            try:
                frame = download.xs(symbol, axis=1, level=1).dropna(subset=["Close"])
            except KeyError:
                continue
            if not frame.empty:
                raw[symbol] = frame
        pd.to_pickle(raw, snapshot)
    if CONFIG.BENCHMARK not in raw or len(raw[CONFIG.BENCHMARK]) < 620:
        raise RuntimeError("At least 620 benchmark sessions are needed; no performance claim can be made")
    dates = raw[CONFIG.BENCHMARK].index
    holdout_start = dates[-252]
    holdout_input_start = dates[-372]
    config = replace(CONFIG, PLATT_A=-4., PLATT_B=2., INTRADAY_ENABLED=False, MAX_WORKERS=4)
    runs: dict[str, Any] = {}
    rng = np.random.default_rng(20261002)
    for name, holdout, multiplier in [("development", False, 1), ("holdout", True, 1),
                                       ("holdout_2x_slippage", True, 2), ("holdout_3x_slippage", True, 3)]:
        data = {s: f.loc[f.index >= holdout_input_start] if holdout
                else f.loc[f.index < holdout_start] for s, f in raw.items()}
        model = replace(SWING_COST_MODEL, slippage_pct=config.SLIPPAGE_BPS * multiplier / 10000.)
        print(f"Running {name}: {len(data)} symbols", flush=True)
        result = walk_forward(data, config, train_days=120, test_days=20, step_days=20,
                              direction="LONG", horizon_filter="SWING", cost_model=model)
        result.to_csv(str(args.out / f"{name}.csv"))
        runs[name] = summarize(result, rng)
        print(f"{name}: {result.overall.n_trades} trades; mean R={result.overall.mean_r}", flush=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "period": args.period, "requested_symbols": symbols,
        "fetched_symbols": sorted(raw), "missing_symbols": sorted(set(symbols) - set(raw)),
        "data_start": str(dates[0]), "data_end": str(dates[-1]), "holdout_start": str(holdout_start),
        "protocol": "Fixed default heuristic calibration; 120-session training, 20-session nonoverlapping test windows; last 252 sessions held out; LONG SWING; monthly block bootstrap, 2000 resamples, seed 20261002",
        "limitations": ["Current constituents introduce survivorship bias", "Subset selected by sector rotation, not point-in-time membership",
                        "Adjusted research prices, not broker executable quotes", "Simulator selects at fold boundaries, not a daily live portfolio replay",
                        "R statistics are not INR NAV returns; no proven live trading edge", "Final holdout is a retrospective protocol, not a prospectively registered experiment"],
        "runs": runs,
    }
    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    lines = ["# Trading performance evaluation", "", report["protocol"], "",
             f"Data: {report['data_start']} to {report['data_end']}; holdout from {holdout_start}.", "",
             "| Run | Trades | Mean net R | Profit factor | Monthly block 95% CI |",
             "|---|---:|---:|---:|---|"]
    for name, stats in runs.items():
        lines.append(f"| {name} | {stats['n_trades']} | {stats['mean_r']} | {stats['profit_factor']} | {stats['monthly_block_mean_r_ci95']} |")
    lines += ["", "Limitations:", ""] + [f"- {item}" for item in report["limitations"]]
    baseline = runs["holdout"]
    ci = baseline["monthly_block_mean_r_ci95"]
    verdict = "Insufficient evidence of positive net expectancy."
    if ci is not None and ci[0] > 0:
        verdict = "Positive net expectancy in this retrospective sample; limitations prevent a live profitability claim."
    lines += ["", f"Conclusion: {verdict}"]
    (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out / "report.md", flush=True)


if __name__ == "__main__":
    main()
