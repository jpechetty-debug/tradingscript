"""Append new durable paper events to JSONL without rewriting existing history."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.database import SqliteDatabase  # noqa: E402
from core.paper_ledger import PaperLedger  # noqa: E402
from core.runtime_paths import RUNTIME_PATHS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=RUNTIME_PATHS.state_db_file)
    parser.add_argument("--out", type=Path, default=RUNTIME_PATHS.state_dir / "paper_ledger.jsonl")
    args = parser.parse_args()
    count = PaperLedger(SqliteDatabase(args.db)).export_jsonl(args.out)
    print(f"Appended {count} events to {args.out}")


if __name__ == "__main__":
    main()
