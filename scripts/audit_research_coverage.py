"""Audit trusted local research inputs without downloading data or changing policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from core.alpha_feeds import SCHEMAS  # noqa: E402
from core.coverage_audit import FeedAudit, MembershipAudit, audit_coverage, session_date  # noqa: E402


def provenance(path: Path | None) -> dict[str, Any]:
    return {"path": str(path.resolve()) if path else None,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path and path.is_file() else None,
            "status": "PRESENT" if path and path.is_file() else "MISSING"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.sector_lookback < 1:
        raise ValueError("Sector lookback must be positive")
    with args.snapshot.open("rb") as stream:
        history = pickle.load(stream)
    if not isinstance(history, dict) or not all(isinstance(k, str) and isinstance(v, pd.DataFrame) for k, v in history.items()):
        raise ValueError("Expected a trusted ticker-to-DataFrame price snapshot")
    if any(not isinstance(frame.index, pd.DatetimeIndex) for frame in history.values()):
        raise ValueError("Price sessions must use a DatetimeIndex")
    inventory = {"prices": provenance(args.snapshot), "membership": provenance(args.membership),
                 "membership_boundary_source": provenance(args.boundary_source)}
    manifest: dict[str, Any] = {}
    manifest_error = ""
    manifest_path = args.feeds / "manifest.json" if args.feeds else None
    inventory["manifest"] = provenance(manifest_path)
    if manifest_path and manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (not isinstance(manifest, dict) or not isinstance(manifest.get("sector_universe"), list)
                    or not manifest["sector_universe"] or not all(isinstance(v, str) and v.strip() for v in manifest["sector_universe"])
                    or len(set(manifest["sector_universe"])) != len(manifest["sector_universe"])
                    or not isinstance(manifest.get("benchmark_index"), str) or not manifest["benchmark_index"].strip()
                    or not isinstance(manifest.get("benchmark_ticker"), str) or not manifest["benchmark_ticker"].strip()):
                raise ValueError("Manifest requires a unique nonempty sector universe and benchmark index/ticker")
        except (ValueError, TypeError, OSError) as exc:
            manifest, manifest_error = {}, str(exc)
    benchmark = args.benchmark or manifest.get("benchmark_ticker", "^NSEI")
    if args.sessions:
        calendar = pd.read_csv(args.sessions)
        if "session" not in calendar:
            raise ValueError("Session CSV requires a session column")
        sessions = pd.DatetimeIndex([session_date(v) for v in calendar.session])
        calendar_basis = "SUPPLIED_SESSIONS_NOT_PROVIDER_VERIFIED"
        inventory["sessions"] = provenance(args.sessions)
    else:
        sessions = pd.DatetimeIndex([])
        for frame in history.values():
            sessions = sessions.union(frame.index)
        calendar_basis = "UNION_OF_OBSERVED_PRICE_DATES_NOT_EXCHANGE_CERTIFIED"
    context_sessions = sessions.copy()
    if args.start:
        sessions = sessions[sessions >= session_date(args.start)]
    if args.end:
        sessions = sessions[sessions <= session_date(args.end)]
    supported = session_date(args.membership_supported_through) if args.membership_supported_through else None
    membership = MembershipAudit(None, supported)
    if args.membership and args.membership.is_file():
        try:
            membership = MembershipAudit(pd.read_csv(args.membership), supported)
        except (ValueError, OSError) as exc:
            membership.mode, membership.error = "INVALID", str(exc)
    feeds: dict[str, FeedAudit] = {}
    for kind in SCHEMAS:
        path = args.feeds / f"{kind}.csv" if args.feeds else None
        inventory[kind] = provenance(path)
        feeds[kind] = FeedAudit(kind, None)
        if path and path.is_file():
            try:
                feeds[kind] = FeedAudit(kind, pd.read_csv(path))
            except (ValueError, OSError) as exc:
                feeds[kind].error = str(exc)
    windows = {name: args.sector_lookback for name in manifest.get("sector_universe", [])}
    if manifest:
        windows[manifest["benchmark_index"]] = args.price_lookback
        if isinstance(manifest.get("vix_index"), str) and manifest["vix_index"].strip():
            windows[manifest["vix_index"]] = 127
    result = audit_coverage(history, sessions, membership, feeds, benchmark=benchmark, index_windows=windows,
                            price_lookback=args.price_lookback, observation_hour=args.observation_hour,
                            context_sessions=context_sessions)
    report = {**result.summary, "schema_version": 1, "calendar_basis": calendar_basis,
              "manifest_error": manifest_error, "benchmark_ticker": benchmark, "observation_hour_ist": args.observation_hour,
              "index_lookback_sessions": windows, "inventory": inventory, "saved_runs": [],
              "implementation_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in
                                        (Path(__file__), ROOT / "core/coverage_audit.py", ROOT / "core/alpha_feeds.py")}}
    for path in args.saved_run:
        item = provenance(path)
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            source = saved.get("provenance", {})
            item.update(snapshot_hash_matches=source.get("snapshot_sha256") == inventory["prices"]["sha256"],
                        recorded_feed_hashes=source.get("feed_sha256"), recorded_qualified=saved.get("qualified"),
                        recorded_mode=saved.get("mode"), recorded_trades=saved.get("evidence", {}).get("trades"))
        except (OSError, ValueError, AttributeError) as exc:
            item["error"] = str(exc)
        report["saved_runs"].append(item)
    args.out.mkdir(parents=True, exist_ok=True)
    daily = result.daily.copy()
    daily["overlapping_members"] = daily.overlapping_members.map(json.dumps)
    daily.to_csv(args.out / "daily_coverage.csv", index=False)
    feed_daily = result.feed_daily.copy()
    feed_daily["reason_counts"] = feed_daily.reason_counts.map(lambda v: json.dumps(v, sort_keys=True))
    feed_daily.to_csv(args.out / "feed_coverage.csv", index=False)
    result.ingestion.to_csv(args.out / "ingestion_audit.csv", index=False)
    result.unavailable.to_csv(args.out / "unavailable_intervals.csv", index=False)
    (args.out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    markdown = ["# Universe and dated-feed coverage audit", "", f"Status: **{report['status']}**",
                f"Audited {report['sessions']} sessions, {report['start']} through {report['end']}.", "",
                "## Membership status", "", "| Status | Sessions |", "|---|---:|"]
    markdown += [f"| {status} | {count} |" for status, count in report["membership_status_sessions"].items()]
    markdown += ["", "## Feed inventory", "", "| Feed | Status | Rows |", "|---|---|---:|"]
    markdown += [f"| {kind} | {item['status']} | {item['rows']} |" for kind, item in report["feeds"].items()]
    markdown += ["", "Coverage ratios use the declared/known row set, whose completeness is unverified.",
                 "The supported-through date is a declared upper bound, not evidence of ingestion or continuity.",
                 "Prices have no observation timestamps. File modification times are never substituted.",
                 "See daily_coverage.csv, feed_coverage.csv, ingestion_audit.csv and unavailable_intervals.csv.", ""]
    (args.out / "report.md").write_text("\n".join(markdown), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=ROOT / "artifacts/performance_validation/historical.pkl")
    parser.add_argument("--membership", type=Path, default=ROOT / "artifacts/universe/nifty500_members.csv")
    parser.add_argument("--membership-supported-through", help="Declared upper bound; does not certify earlier coverage")
    parser.add_argument("--boundary-source", type=Path, help="Local source establishing the declared upper bound; hash retained")
    parser.add_argument("--feeds", type=Path)
    parser.add_argument("--sessions", type=Path)
    parser.add_argument("--benchmark")
    parser.add_argument("--price-lookback", type=int, default=200)
    parser.add_argument("--sector-lookback", type=int, default=64)
    parser.add_argument("--observation-hour", type=int, default=18)
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--saved-run", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/coverage_audit")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"report": str(args.out / "report.json"), "status": report["status"],
                      "sessions": report["sessions"], "membership_status_sessions": report["membership_status_sessions"]}))


if __name__ == "__main__":
    main()
