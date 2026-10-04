"""Local V2 scans/replay. No broker connection and no fabricated feed substitutes."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import pickle
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.alpha_feeds import AlphaFeed, AlphaFeeds, SCHEMAS  # noqa: E402
from core.backtest import SWING_COST_MODEL  # noqa: E402
from core.config import CONFIG, IST  # noqa: E402
from core.database import SqliteDatabase  # noqa: E402
from core.indicators import add_indicators  # noqa: E402
from core.paper_ledger import PaperLedger  # noqa: E402
from core.swing import completed_daily_bars  # noqa: E402
from core.swing_analytics import evidence, research_parameters  # noqa: E402
from core.swing_exits import ExitPolicy  # noqa: E402
from core.swing_replay import replay_swing  # noqa: E402
from core.swing_v2 import SwingV2, V2Settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "artifacts/performance_validation/historical.pkl")
    parser.add_argument("--feeds", type=Path, help="Directory containing manifest.json and optional typed CSV feeds")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/swing_v2")
    parser.add_argument("--model", choices=["SECTOR", "PEAD", "COMBINED"], default="SECTOR")
    parser.add_argument("--mode", choices=["scan", "replay"], default="scan")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", help="Last completed session by default")
    parser.add_argument("--gate", action="append", choices=["market", "breadth", "vix", "delivery"], default=[])
    parser.add_argument("--exit", choices=["FIXED", "ATR_TRAIL", "CHANDELIER"], default="FIXED")
    parser.add_argument("--journal-db", type=Path, help="Record today's completed scan to an append-only research DB")
    args = parser.parse_args()
    if args.journal_db and args.mode != "scan":
        parser.error("Historical replay cannot be recorded as a prospective paper scan")
    if args.mode == "scan" and args.exit != "FIXED":
        parser.error("Alternative exit policies are replay experiments; prospective scans use the frozen fixed policy")
    settings = V2Settings(model=args.model, **{f"{gate}_gate": True for gate in args.gate})
    feeds = AlphaFeeds()
    hashes = {}
    benchmark_ticker = CONFIG.BENCHMARK
    if args.feeds:
        manifest_path = args.feeds / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes["manifest.json"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        feeds = AlphaFeeds(sector_universe=tuple(manifest["sector_universe"]),
                           benchmark_index=manifest["benchmark_index"], vix_index=manifest["vix_index"])
        benchmark_ticker = manifest["benchmark_ticker"]
        for kind in SCHEMAS:
            path = args.feeds / f"{kind}.csv"
            if path.exists():
                feeds.tables[kind] = AlphaFeed(kind, pd.read_csv(path))
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot_bytes = args.snapshot.read_bytes()
    raw = pickle.loads(snapshot_bytes)  # Local trusted research snapshots only.
    config = replace(CONFIG, BENCHMARK=benchmark_ticker, SWING_SETUP_ENABLED=True)
    now = pd.Timestamp.now(IST)
    raw = {t: completed_daily_bars(f, config, now.to_pydatetime()) for t, f in raw.items()}
    if benchmark_ticker not in raw or raw[benchmark_ticker].empty:
        parser.error("Snapshot must contain the declared benchmark price series")
    end = pd.Timestamp(args.end) if args.end else raw[benchmark_ticker].index[-1]
    raw = {t: f.loc[:end] for t, f in raw.items()}
    if raw[benchmark_ticker].empty:
        parser.error("End date precedes the available benchmark history")
    date = raw[benchmark_ticker].index[-1]
    if args.journal_db and (date.date() != now.date() or now.hour < settings.observation_hour):
        parser.error("Journal requires today's completed bar and an observation time after 18:00 IST")
    if args.feeds and "indices" in feeds.tables:
        official = feeds.index_history(feeds.benchmark_index, now, date)
        common = raw[benchmark_ticker].Close.index.intersection(official.index)
        if len(common) and ((raw[benchmark_ticker].Close.loc[common] / official.loc[common] - 1).abs() > .001).any():
            parser.error("Declared benchmark prices disagree with the index feed; provide matching Nifty 500 data")
    prepared = {t: add_indicators(f, config) for t, f in raw.items() if not f.empty}
    engine = SwingV2(feeds, settings)
    protocol_path = ROOT / "docs/swing-v2-protocol.json"
    protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    provenance = {"snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(), "feed_sha256": hashes,
                  "protocol_version": "SWING_ALPHA_V2", "protocol_sha256": protocol_hash,
                  "benchmark_ticker": benchmark_ticker}
    report = {"status": "RESEARCH_ONLY_EDGE_UNPROVEN", "mode": args.mode,
              "settings": asdict(settings), "provenance": provenance,
              "limitations": ["Current-snapshot membership and adjusted OHLC retain survivorship and revision bias.",
                              "Locally supplied timestamps/coverage require provider audit; no exchange feed is downloaded.",
                              "Daily OHLC models market fills; price-band/auction feasibility is not established.",
                              "V2 has no calibrated win probabilities. Delivery is an accumulation proxy."]}
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "replay":
        policy = ExitPolicy(mode=args.exit, atr_multiple=2. if args.exit == "ATR_TRAIL" else 3.)
        replay = replay_swing(raw, config, pd.Timestamp(args.start), end, prepared=prepared,
                              warm_history=True, exit_policy=policy, signal_provider=engine, entry_guard=engine.entry_guard)
        report["evidence"] = evidence(replay)
        report["cost_model"] = asdict(SWING_COST_MODEL)
        replay.result.to_dataframe().to_csv(args.out / "trades.csv", index=False)
        replay.equity.to_csv(args.out / "equity.csv")
        replay.diagnostics.to_csv(args.out / "execution_decisions.csv", index=False)
    else:
        candidates = engine(prepared, date, config)
        report["qualified"] = len(candidates)
        (args.out / "candidates.json").write_text(json.dumps([asdict(c) for c in candidates], indent=2, allow_nan=False), encoding="utf-8")
        if args.journal_db:
            params = {"v2": asdict(settings), "execution": research_parameters(config), "protocol_sha256": protocol_hash}
            ledger = PaperLedger(SqliteDatabase(args.journal_db))
            ledger.record_research_scan(candidates, date.isoformat(), params, provenance)
            ledger.export_jsonl(args.journal_db.with_suffix(".jsonl"))
    reasons = Counter(reason for d in engine.decisions for reason in d["reasons"])
    report["decision_count"] = len(engine.decisions)
    report["rejection_counts"] = dict(reasons)
    (args.out / "decisions.jsonl").write_text("".join(json.dumps(d, allow_nan=False) + "\n" for d in engine.decisions), encoding="utf-8")
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"report": str(args.out / "report.json"), "decisions": len(engine.decisions), "reasons": dict(reasons)}))


if __name__ == "__main__":
    main()
