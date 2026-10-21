"""Build daily read-only reconciliation from the configured append-only paper journal."""
from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import IST  # noqa: E402
from core.paper_reconciliation import reconcile_paper_day, write_daily_report  # noqa: E402
from core.runtime_paths import RuntimePaths  # noqa: E402


def main() -> None:
    paths = RuntimePaths.discover()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=paths.state_db_file)
    parser.add_argument("--date", type=date.fromisoformat, default=datetime.now(IST).date(), help="IST observation day (YYYY-MM-DD); default today")
    parser.add_argument("--as-of", type=lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")),
                        help="Aware observation cutoff, capped at report day-end")
    parser.add_argument("--max-recording-delay-seconds", type=float, default=900,
                        help="Flag submissions later than this delay (default 900 seconds)")
    parser.add_argument("--out", type=Path, help="Default: configured artifacts/paper_reconciliation/YYYY-MM-DD")
    args = parser.parse_args()
    try:
        report = reconcile_paper_day(args.db, args.date, as_of=args.as_of,
                                     max_recording_delay_seconds=args.max_recording_delay_seconds)
        directory = args.out or paths.artifacts_dir / "paper_reconciliation" / args.date.isoformat()
        write_daily_report(report, directory)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.error(str(exc))
    print(f"{report['status']}: {report['summary']['new_signals']} new signals; report: {directory / 'report.md'}")


if __name__ == "__main__":
    main()
