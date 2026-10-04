"""Run all four alpha-roadmap phases on trusted local snapshots and dated feeds."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.alpha_engine import AlphaSettings, SwingAlpha  # noqa: E402
from core.alpha_feeds import AlphaFeed, AlphaFeeds, SCHEMAS  # noqa: E402
from core.alpha_meta import MetaModel  # noqa: E402
from core.alpha_portfolio import PortfolioLimits, allocate_candidates  # noqa: E402
from core.alpha_research import ResearchAgent, openai_generator  # noqa: E402
from core.alpha_validation import StressSettings, calendar_folds, monte_carlo  # noqa: E402
from core.backtest import SWING_COST_MODEL  # noqa: E402
from core.config import CONFIG, IST  # noqa: E402
from core.indicators import add_indicators  # noqa: E402
from core.swing import completed_daily_bars  # noqa: E402
from core.swing_analytics import evidence  # noqa: E402
from core.swing_replay import replay_swing  # noqa: E402


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=lambda v: v.isoformat()), encoding="utf-8")


def load_feeds(directory: Path | None) -> tuple[AlphaFeeds, str, dict[str, str]]:
    if directory is None:
        return AlphaFeeds(), CONFIG.BENCHMARK, {}
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feeds = AlphaFeeds(tuple(manifest["sector_universe"]), benchmark_index=manifest["benchmark_index"],
                       vix_index=manifest["vix_index"])
    hashes = {"manifest.json": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    for kind in SCHEMAS:
        path = directory / f"{kind}.csv"
        if path.exists():
            feeds.tables[kind] = AlphaFeed(kind, pd.read_csv(path))
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return feeds, manifest["benchmark_ticker"], hashes


def save_replay(out: Path, replay: Any, engine: SwingAlpha, stress: StressSettings) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    replay.result.to_dataframe().to_csv(out / "trades.csv", index=False)
    write_json(out / "closed_trades.json", [asdict(t) for t in replay.result.trades])
    replay.equity.to_csv(out / "equity.csv")
    replay.diagnostics.to_csv(out / "execution_decisions.csv", index=False)
    (out / "decisions.jsonl").write_text("".join(json.dumps(d, allow_nan=False) + "\n" for d in engine.decisions), encoding="utf-8")
    result = {"evidence": evidence(replay), "monte_carlo": monte_carlo(replay.result.trades, stress),
              "decision_count": len(engine.decisions),
              "rejection_counts": dict(Counter(r for d in engine.decisions for r in d["reasons"]))}
    write_json(out / "validation.json", result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    feeds, benchmark, hashes = load_feeds(args.feeds)
    data = args.snapshot.read_bytes()
    raw = pickle.loads(data)  # Only trusted local snapshots, as in the existing research CLI.
    config = replace(CONFIG, BENCHMARK=benchmark, SWING_SETUP_ENABLED=True)
    now = pd.Timestamp.now(IST)
    raw = {t: completed_daily_bars(f, config, now.to_pydatetime()) for t, f in raw.items()}
    if benchmark not in raw or raw[benchmark].empty:
        raise ValueError("Snapshot must contain the declared benchmark")
    end = pd.Timestamp(args.end) if args.end else raw[benchmark].index[-1]
    raw = {t: f.loc[:end] for t, f in raw.items()}
    if raw[benchmark].empty:
        raise ValueError("End precedes available history")
    date = raw[benchmark].index[-1]
    if any(f.index.has_duplicates or not f.index.is_monotonic_increasing for f in raw.values()):
        raise ValueError("Snapshot sessions must be ordered and unique")
    if "indices" in feeds.tables:
        official = feeds.index_history(feeds.benchmark_index, now, date)
        common = raw[benchmark].index.intersection(official.index)
        if len(common) and ((raw[benchmark].Close.loc[common] / official.loc[common] - 1).abs() > .001).any():
            raise ValueError("Declared benchmark differs from official index feed")
    settings = AlphaSettings(phase=args.phase)
    limits = PortfolioLimits()
    stress = StressSettings(simulations=args.simulations, seed=args.seed, risk_fraction=config.SWING_RISK_FRACTION)
    prepared = {t: add_indicators(f, config) for t, f in raw.items() if not f.empty}
    protocol_path = ROOT / "docs/alpha-roadmap-protocol.json"
    provenance = {"snapshot_sha256": hashlib.sha256(data).hexdigest(), "feed_sha256": hashes,
                  "benchmark_ticker": benchmark, "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    report: dict[str, Any] = {"status": "RESEARCH_ONLY_EDGE_UNPROVEN", "mode": args.mode,
                              "settings": asdict(settings), "portfolio_limits": asdict(limits),
                              "cost_model": asdict(SWING_COST_MODEL), "provenance": provenance,
                              "meta_enabled": args.meta, "ai_enabled": args.ai_model is not None,
                              "limitations": ["Snapshot constituents and adjusted price revisions retain survivorship/revision bias.",
                                              "Local publication/ingestion timestamps require external provider audit.",
                                              "Daily OHLC does not establish exchange/broker fill feasibility.",
                                              "Scores and thresholds are hypotheses; no validated performance improvement is claimed."]}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "scan":
        model = None
        if args.meta:
            model = MetaModel()
            if args.training_trades:
                payload = args.training_trades.read_bytes()
                trades = [SimpleNamespace(**row) for row in json.loads(payload)]
                prior = raw[benchmark].index[raw[benchmark].index < date]
                if len(prior) <= args.embargo_sessions:
                    raise ValueError("Insufficient prior sessions for model embargo")
                cutoff = prior[-args.embargo_sessions - 1].tz_localize(IST) + pd.Timedelta(hours=18)
                model.fit(trades, cutoff)
                provenance["training_trades_sha256"] = hashlib.sha256(payload).hexdigest()
            report["meta_training"] = model.report
        engine = SwingAlpha(feeds, settings, model)
        candidates = engine(prepared, date, config)
        returns = pd.DataFrame({t: f.Close.reindex(raw[benchmark].index).pct_change(fill_method=None).iloc[-60:]
                                for t, f in prepared.items() if t != benchmark})
        held = json.loads(args.held_book.read_text(encoding="utf-8")) if args.held_book else []
        report["allocations"] = allocate_candidates(candidates, config, returns, held, limits)
        if args.held_book:
            provenance["held_book_sha256"] = hashlib.sha256(args.held_book.read_bytes()).hexdigest()
        documents = json.loads(args.documents.read_text(encoding="utf-8")) if args.documents else []
        if args.documents:
            provenance["documents_sha256"] = hashlib.sha256(args.documents.read_bytes()).hexdigest()
        agent = ResearchAgent(openai_generator(args.ai_model) if args.ai_model else None)
        write_json(args.out / "research.json", [agent.run(c, documents) for c in candidates])
        write_json(args.out / "candidates.json", [asdict(c) for c in candidates])
        report["qualified"] = len(candidates)
        report["ai_model"] = args.ai_model
        report["rejection_counts"] = dict(Counter(r for d in engine.decisions for r in d["reasons"]))
        (args.out / "decisions.jsonl").write_text("".join(json.dumps(d, allow_nan=False) + "\n" for d in engine.decisions), encoding="utf-8")
    elif args.mode == "replay":
        engine = SwingAlpha(feeds, settings)
        replay = replay_swing(raw, config, pd.Timestamp(args.start), date, prepared=prepared,
                              warm_history=True, signal_provider=engine, entry_guard=engine.entry_guard,
                              portfolio_limits=limits)
        report.update(save_replay(args.out, replay, engine, stress))
    else:
        sessions = raw[benchmark].loc[pd.Timestamp(args.start):date].index
        folds = calendar_folds(sessions, args.train_years)
        report["folds"] = []
        all_trades = []
        test_returns = []
        for i, fold in enumerate(folds):
            model = None
            training: dict[str, Any] = {"status": "DISABLED"}
            if args.meta:
                model = MetaModel()
                train_sessions = raw[benchmark].loc[fold.train_start:fold.train_end].index
                if len(train_sessions) <= args.embargo_sessions:
                    raise ValueError("Training window too short for embargo")
                cutoff_date = train_sessions[-args.embargo_sessions - 1]
                train_raw = {t: f.loc[:cutoff_date] for t, f in raw.items()}
                train_prepared = {t: f.loc[:cutoff_date] for t, f in prepared.items()}
                train_engine = SwingAlpha(feeds, replace(settings, phase=min(3, settings.phase)))
                train_replay = replay_swing(train_raw, config, fold.train_start, cutoff_date,
                                            prepared=train_prepared, warm_history=True, signal_provider=train_engine,
                                            entry_guard=train_engine.entry_guard, portfolio_limits=limits)
                training = model.fit([t for t in train_replay.result.trades if t.exit_reason != "DATA_END"],
                                     cutoff_date.tz_localize(IST) + pd.Timedelta(hours=18))
            engine = SwingAlpha(feeds, settings, model)
            test_raw = {t: f.loc[:fold.test_end] for t, f in raw.items()}
            test_prepared = {t: f.loc[:fold.test_end] for t, f in prepared.items()}
            replay = replay_swing(test_raw, config, fold.test_start, fold.test_end,
                                  prepared=test_prepared, warm_history=True, signal_provider=engine,
                                  entry_guard=engine.entry_guard, portfolio_limits=limits)
            fold_dir = args.out / f"fold_{i + 1}"
            validation = save_replay(fold_dir, replay, engine, stress)
            all_trades.extend(replay.result.trades)
            test_returns.extend(replay.equity.daily_return.tolist())
            report["folds"].append({"window": asdict(fold), "meta_training": training, **validation})
        report["validation_status"] = "OUT_OF_SAMPLE_FOLDS" if folds else "UNAVAILABLE_INSUFFICIENT_CALENDAR_HISTORY"
        report["aggregate"] = {"closed_trades": len(all_trades), "test_sessions": len(test_returns),
                               "stitched_flat_boundary_return": float(pd.Series(test_returns, dtype=float).add(1).prod() - 1),
                               "monte_carlo": monte_carlo(all_trades, stress)}
    write_json(args.out / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "artifacts/performance_validation/historical.pkl")
    parser.add_argument("--feeds", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/alpha_roadmap")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4], default=4)
    parser.add_argument("--mode", choices=["scan", "replay", "walk-forward"], default="scan")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end")
    parser.add_argument("--train-years", type=int, default=5)
    parser.add_argument("--embargo-sessions", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261004)
    parser.add_argument("--meta", action="store_true", help="Enable the Phase 4 gate; insufficient training evidence rejects candidates")
    parser.add_argument("--training-trades", type=Path, help="JSON closed trade records for scan-mode meta training")
    parser.add_argument("--held-book", type=Path, help="JSON actual held positions for scan allocation")
    parser.add_argument("--documents", type=Path, help="JSON dated source documents for research packets")
    parser.add_argument("--ai-model", help="Explicit model; opt in to sending supplied research packets to the Responses API")
    args = parser.parse_args()
    if args.embargo_sessions < 0:
        parser.error("Embargo must be nonnegative")
    if args.meta and (args.phase != 4 or args.mode == "replay"):
        parser.error("Meta model requires Phase 4 scan or walk-forward mode")
    if args.training_trades and (not args.meta or args.mode != "scan"):
        parser.error("Training-trades are for scan-mode meta training")
    if (args.ai_model or args.documents or args.held_book) and args.mode != "scan":
        parser.error("AI research, source documents and held-book allocation are scan-mode features")
    try:
        report = run(args)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(args.out / "report.json"), "status": report["status"],
                      "qualified": report.get("qualified"), "validation_status": report.get("validation_status")}))


if __name__ == "__main__":
    main()
