"""Sanitized historical audit summaries for the dashboard; never trading gates."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .alpha_feeds import SCHEMAS


def _session_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(count, int) or isinstance(count, bool) or count < 0
        for key, count in value.items()
    ):
        raise ValueError("Invalid session counts")
    return value


def load_dashboard_evidence(artifacts_dir: Path) -> dict[str, Any]:
    directory = artifacts_dir / "coverage_audit"
    base: dict[str, Any] = {"scope": "HISTORICAL_RESEARCH_INPUTS", "status": "UNAVAILABLE_NO_AUDIT"}
    try:
        paths = ([directory / "report.json"] if (directory / "report.json").is_file() else [])
        paths += sorted(directory.glob("*/report.json"))
    except OSError:
        return {**base, "status": "UNAVAILABLE_UNREADABLE_AUDIT"}
    reports: list[tuple[str, str, dict[str, Any]]] = []
    errors = 0
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (not isinstance(payload, dict) or payload.get("schema_version") != 1
                    or not isinstance(payload.get("feeds"), dict)):
                raise ValueError("Unsupported audit schema")
            start = date.fromisoformat(payload["start"])
            end = date.fromisoformat(payload["end"])
            if start > end or not isinstance(payload.get("status"), str):
                raise ValueError("Invalid audit interval")
            payload["start"], payload["end"] = start.isoformat(), end.isoformat()
            for key in ("membership_status_sessions", "membership_integrity_sessions"):
                payload[key] = _session_counts(payload.get(key, {}))
            missing = payload.get("missing_benchmark_sessions", [])
            if not isinstance(missing, list):
                raise ValueError("Invalid missing sessions")
            payload["missing_benchmark_sessions"] = [date.fromisoformat(day).isoformat() for day in missing]
            reports.append((end.isoformat(), str(path.relative_to(directory)), payload))
        except (OSError, ValueError, KeyError, TypeError):
            errors += 1
    if not reports:
        return {**base, "status": "UNAVAILABLE_INVALID_AUDIT" if errors else base["status"], "unreadable_reports": errors}
    _, _, latest = max(reports, key=lambda item: (item[0], item[1]))
    feeds = {}
    for kind in SCHEMAS:
        item = latest["feeds"].get(kind)
        status = item.get("status") if isinstance(item, dict) else None
        feeds[kind] = status if status in (
            "VALID_RECORDED_TIMESTAMPS_NOT_PROVIDER_VERIFIED", "MISSING", "INVALID"
        ) else "UNAVAILABLE"
    return {
        **base, "status": latest["status"], "start": latest["start"], "end": latest["end"],
        "feeds": feeds, "unreadable_reports": errors,
        "membership_status_sessions": latest.get("membership_status_sessions", {}),
        "membership_integrity_sessions": latest.get("membership_integrity_sessions", {}),
        "missing_benchmark_sessions": latest.get("missing_benchmark_sessions", []),
        "price_asof_status": "UNAVAILABLE_PRICE_OBSERVATION_TIMESTAMPS",
    }
