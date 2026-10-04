"""Exploratory leadership/MAE/MFE and prespecified exit comparisons on saved data."""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.backtest import SWING_COST_MODEL, TradeRecord  # noqa: E402
from core.config import CONFIG, IST  # noqa: E402
from core.indicators import add_indicators  # noqa: E402
from core.research_feeds import EarningsCalendar, MembershipHistory  # noqa: E402
from core.swing_analytics import evidence, monthly_mean_r_interval, trade_statistics  # noqa: E402
from core.swing_exits import ExitPolicy  # noqa: E402
from core.swing_replay import replay_swing  # noqa: E402
from core.swing_research import leadership_snapshot, percentile_bucket, trade_excursions  # noqa: E402


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    names = {field.name for field in fields(TradeRecord)} - {"daily_pnl", "research_context", "exit_fills"}
    trades = []
    for row in frame.to_dict("records"):
        values = {key: value for key, value in row.items() if key in names and not pd.isna(value)}
        values["entry_date"] = pd.Timestamp(values["entry_date"])
        values["exit_date"] = pd.Timestamp(values["exit_date"])
        trades.append(TradeRecord(**values))
    stat = trade_statistics(trades)
    stat["monthly_mean_r_ci95"] = monthly_mean_r_interval(trades)
    if len(frame):
        cumulative = np.r_[0., frame.sort_values("exit_date").r_multiple.to_numpy()].cumsum()
        stat["trade_r_curve_drawdown"] = float((cumulative - np.maximum.accumulate(cumulative)).min())
    else:
        stat["trade_r_curve_drawdown"] = None
    return stat


def buckets(frame: pd.DataFrame) -> dict[str, Any]:
    fields = {"rs63_decile": [f"{i:02d}-{i+10:03d}" for i in range(0, 100, 10)],
              "rs252_decile": [f"{i:02d}-{i+10:03d}" for i in range(0, 100, 10)],
              "sector_alignment": ["POSITIVE_PEER_RS", "NONPOSITIVE_PEER_RS"],
              "leadership_distance": ["0-10%", "10-20%", "20-40%", "40%+"],
              "breadth50_bucket": [f"{i:02d}-{i+20:03d}" for i in range(0, 100, 20)],
              "breadth200_bucket": [f"{i:02d}-{i+20:03d}" for i in range(0, 100, 20)],
              "atr20_bucket": [f"{i:02d}-{i+20:03d}" for i in range(0, 100, 20)],
              "atr_expanding": ["True", "False"], "bandwidth_expanding": ["True", "False"],
              "benchmark_above_ema200": ["True", "False"], "earnings_status": []}
    result = {}
    for field, labels in fields.items():
        observed = frame[field].fillna("UNKNOWN").astype(str)
        result[field] = {label: summarize(frame.loc[observed == label]) for label in sorted(set(labels) | set(observed))}
    return result


def excursion_summary(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"trades": len(frame), "exit_day_order_unknown": int(frame.exit_day_order_unknown.sum())}
    for name, subset, field, threshold in (
            ("losers_reached_1r", frame.loc[frame.r_multiple < 0], "mfe", 1.),
            ("winners_suffered_half_r", frame.loc[frame.r_multiple > 0], "mae", .5),
            ("all_reached_2r", frame, "mfe", 2.)):
        result[name] = {"denominator": len(subset), "definite": int((subset[f"{field}_r_lower"] >= threshold).sum()),
                        "possible_including_unknown_exit_day_order": int((subset[f"{field}_r_upper"] >= threshold).sum())}
    for key in ("mfe_r_lower", "mfe_r_upper", "mae_r_lower", "mae_r_upper"):
        result[key] = {str(p): float(frame[key].quantile(p / 100)) for p in (25, 50, 75, 95)}
    return result


def annotate(trades: pd.DataFrame, prepared: dict[str, pd.DataFrame], raw: dict[str, pd.DataFrame],
             config: Any, calendar: EarningsCalendar | None, membership: MembershipHistory | None) -> tuple[pd.DataFrame, dict[str, Any]]:
    sessions = raw[config.BENCHMARK].index
    snapshots: dict[pd.Timestamp, dict[str, dict[str, Any]]] = {}
    rows = []
    missing_members: set[str] = set()
    for record in trades.to_dict("records"):
        entry_date = pd.Timestamp(record["entry_date"])
        entry_i = sessions.get_indexer([entry_date])[0]
        if entry_i <= 0:
            raise ValueError("Entry needs a known prior signal session")
        signal_date = sessions[entry_i - 1]
        if signal_date not in snapshots:
            snapshots[signal_date] = leadership_snapshot(prepared, config, signal_date)
        ticker = record["ticker"]
        context = {"earnings_status": "UNKNOWN_NO_ASOF_EVENT_FEED", "bandwidth_expanding": None,
                   "benchmark_above_ema200": None, **snapshots[signal_date].get(ticker, {})}
        observation = pd.Timestamp(signal_date).tz_localize(IST) + pd.Timedelta(hours=15, minutes=30)
        if calendar:
            context = {**context, "earnings_status": calendar.status(ticker, observation, entry_date, sessions)}
        members = membership.members(observation, entry_date) if membership else None
        if members is not None and membership is not None:
            missing_members.update(membership.missing_prices(members, {t for t, f in raw.items() if entry_date in f.index}))
        trade = SimpleNamespace(**{**record, "entry_date": entry_date, "exit_date": pd.Timestamp(record["exit_date"])})
        row = {**record, **context, **trade_excursions(trade, raw[ticker]), "signal_session": signal_date,
               "pit_member_at_entry": ticker in members if members is not None else None}
        row["rs63_decile"] = percentile_bucket(context.get("rs_63_percentile"))
        row["rs252_decile"] = percentile_bucket(context.get("rs_252_percentile"))
        sector = context.get("sector_rs_63")
        row["sector_alignment"] = "UNKNOWN" if sector is None else "POSITIVE_PEER_RS" if sector > 0 else "NONPOSITIVE_PEER_RS"
        distance = context.get("distance_ema200")
        row["leadership_distance"] = "UNKNOWN" if distance is None else "0-10%" if distance < .1 else "10-20%" if distance < .2 else "20-40%" if distance < .4 else "40%+"
        for span in (50, 200):
            value = context.get(f"breadth_ema{span}")
            row[f"breadth{span}_bucket"] = percentile_bucket(value * 100 if value is not None else None, 20)
        row["atr20_bucket"] = percentile_bucket(context.get("atr20_percentile"), 20)
        expansion = context.get("atr20_expansion")
        row["atr_expanding"] = expansion > 1 if expansion is not None else None
        rows.append(row)
    return pd.DataFrame(rows), {"earnings": "ASOF_SUPPLIED_COVERAGE_UNVERIFIED" if calendar else "NOT_EVALUATED_NO_EVENT_FEED",
                               "membership": "ASOF_SUPPLIED_AUDIT_ONLY" if membership else "NOT_EVALUATED_NO_MEMBERSHIP_HISTORY",
                               "missing_member_price_symbols": sorted(missing_members)}


def write_report(report: dict[str, Any], out: Path) -> None:
    lines = ["# Structural swing research", "", report["status"], "", "## MAE/MFE bounds", "",
             "```json", json.dumps(report["excursions"], indent=2), "```", "",
             "Exit-session high/low may occur after a stop or target. Definite and possible counts are bounds, not reconstructed intraday paths. Opening exits exclude subsequent extremes. Close/time exits include the full session. Existing exits censor excursion paths; this analysis alone cannot establish a better exit.", ""]
    for field, groups in report["cohorts"].items():
        lines += [f"## {field}", "", "| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |",
                  "|---|---:|---:|---:|---:|---:|"]
        for label, stat in groups.items():
            def fmt(value: Any) -> str:
                return "—" if value is None else f"{value:.4f}"
            lines.append(f"| {label} | {stat['trades']} | {fmt(stat['mean_r'])} | {fmt(stat['win_rate'])} | {fmt(stat['trade_r_curve_drawdown'])} | {fmt(stat.get('mean_excess_return'))} |")
        lines.append("")
    lines += ["¹ Cumulative net R in exit-date order, not a deployable subgroup NAV/drawdown. Win rates and excess returns are fractions. Small/empty groups and confidence intervals are retained in JSON; these post-hoc cohorts are not significance-tested discoveries.", "",
              "## Research-only exit portfolios", "", "| Policy | Trades | Mean R | NAV return | Drawdown | Exposure | Benchmark-matched excess NAV |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, stat in report["exit_policies"].items():
        lines.append(f"| {name} | {stat['trades']} | {stat['mean_r']:.4f} | {stat['total_return']:.2%} | {stat['max_drawdown']:.2%} | {stat['average_exposure']:.2%} | {stat['exposure_matched_active_return']:.2%} |")
    lines += ["", "Exit variants retain original entry eligibility, initial stop and time limit. ATR trail uses highest held close minus 2 ATR; Chandelier uses highest held high minus 3 ATR. Both replace the fixed target, update after the close and apply next session. Stops never loosen. Scale-out sells floor(50% of initial shares) at 1R and the rest at 3R, with original stop/time exit. Extra legs incur their own fees. Unknown intraday stop/target order is stop-first; opening gaps fill at the opening price. Changed holding times alter later capital, admission and fill quantities, so these are complete portfolio variants, not identical-entry causal estimates.", "",
              "## Data dependencies", "", "```json", json.dumps(report["feed_audit"], indent=2), "```", "",
              "Earnings absence is unknown, never an all-clear. Supplied events must be known at the historical signal close; supplied membership is audited as-of, including missing/delisted prices. Without complete membership/price/event histories this run cannot claim Nifty-500 point-in-time or earnings-avoidance evidence.", "",
              "## Limits and next evaluation", ""]
    lines += [f"- {item}" for item in report["limitations"]]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=Path("artifacts/performance_validation/historical.pkl"))
    parser.add_argument("--base-report", type=Path, default=Path("artifacts/swing_research"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/swing_structural"))
    parser.add_argument("--earnings", type=Path)
    parser.add_argument("--membership", type=Path)
    parser.add_argument("--skip-exits", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    raw = pickle.loads(args.snapshot.read_bytes())
    config = replace(CONFIG, SWING_SETUP_ENABLED=True)
    prepared = {ticker: add_indicators(frame, config) for ticker, frame in raw.items()}
    original = pd.read_csv(args.base_report / "continuous_rules_trades.csv")
    original["entry_date"] = pd.to_datetime(original.entry_date)
    original["exit_date"] = pd.to_datetime(original.exit_date)
    annotated, audit = annotate(original, prepared, raw, config,
                                EarningsCalendar(pd.read_csv(args.earnings)) if args.earnings else None,
                                MembershipHistory(pd.read_csv(args.membership)) if args.membership else None)
    annotated.to_csv(args.out / "annotated_trades.csv", index=False)
    protocol_path = Path(__file__).resolve().parents[1] / "docs" / "swing-hypotheses-v1.json"
    report: dict[str, Any] = {"status": "EXPLORATORY ON PREVIOUSLY VIEWED DATA; NO WINNER SELECTED OR DEPLOYED",
                              "snapshot_sha256": hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
                              "research_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol_path.exists() else None,
                              "excursions": excursion_summary(annotated), "cohorts": buckets(annotated),
                              "feed_audit": audit, "exit_policies": {},
                              "limitations": ["The available liquid 60-stock subset/current mappings retain survivorship, selection and sector-proxy bias.",
                                              "Rank denominators are stored; these are ranks in the available liquid universe, not Nifty-500 population deciles.",
                                              "Sector alignment is leave-one-out median peer relative strength with at least two peers, not an official sector-index EMA200 test.",
                                              "ATR20 is a Wilder-style smoothed true range; its percentile compares strictly prior observations. Volatility expansion compares the current ATR20 with its prior 20-session average.",
                                              "Prior signal session is inferred from next-session entry for old records; new replay records retain actual signal_time.",
                                              "Original factor/liquidity/setup gates restrict trade cohorts. Filled-trade attribution does not estimate the effect of replacing the whole candidate-generation pipeline.",
                                              "Exit attribution and favorable excursion do not establish that entries are good; daily bars do not resolve threshold ordering.",
                                              "All exit variants are research-only and leave application trade management unchanged.",
                                              "Future protocol is docs/swing-hypotheses-v1.json; no new live entry filters or automatic deployment are introduced."]}
    if not args.skip_exits:
        model = replace(SWING_COST_MODEL, slippage_pct=config.SLIPPAGE_BPS / 10000.)
        start, end = pd.Timestamp("2024-01-01"), raw[config.BENCHMARK].index[-1]
        for policy in (ExitPolicy(), ExitPolicy("ATR_TRAIL", 2.), ExitPolicy("CHANDELIER", 3.), ExitPolicy("SCALE_OUT")):
            print(f"Running {policy.mode}: {start.date()} to {end.date()}", flush=True)
            replay = replay_swing(raw, config, start, end, model, prepared=prepared, warm_history=True, exit_policy=policy)
            stat = evidence(replay)
            report["exit_policies"][policy.mode] = stat
            replay.result.to_csv(str(args.out / f"{policy.mode}_trades.csv"))
            replay.equity.to_csv(args.out / f"{policy.mode}_equity.csv")
            replay.diagnostics.to_csv(args.out / f"{policy.mode}_decisions.csv", index=False)
            print(f"{policy.mode}: {stat['trades']} trades; mean {stat['mean_r']:.4f}R; NAV {stat['total_return']:.2%}", flush=True)
            if policy.mode == "FIXED":
                previous = json.loads((args.base_report / "report.json").read_text(encoding="utf-8"))
                if previous["snapshot_sha256"] == report["snapshot_sha256"]:
                    old = previous["continuous"]["rules"]
                    if stat["trades"] != old["trades"] or abs(stat["total_return"] - old["total_return"]) > 1e-8:
                        raise RuntimeError("Fixed-policy replay differs from frozen evidence; investigate before comparing variants")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    write_report(report, args.out)
    print("Saved structural research; no deployed defaults changed.", flush=True)


if __name__ == "__main__":
    main()
