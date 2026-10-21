import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.dashboard_evidence import load_dashboard_evidence


def save_report(root, name="2026-10-20", **changes):
    payload = {
        "schema_version": 1, "start": "2021-10-04", "end": "2026-10-01",
        "status": "UNAVAILABLE_POINT_IN_TIME_COMPLETENESS",
        "feeds": {"earnings": {"status": "MISSING", "path": "private/path"}},
        "membership_status_sessions": {"UNAVAILABLE_STALE_MEMBERSHIP": 1117},
        "membership_integrity_sessions": {"OVERLAPPING_INTERVALS": 1222},
        "missing_benchmark_sessions": ["2026-05-01"],
        "private_extra": "never returned",
    }
    payload.update(changes)
    path = root / "coverage_audit" / name / "report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_missing_audit_is_explicit_and_does_not_create_directories(tmp_path):
    result = load_dashboard_evidence(tmp_path)
    assert result["status"] == "UNAVAILABLE_NO_AUDIT"
    assert result["scope"] == "HISTORICAL_RESEARCH_INPUTS"
    assert not list(tmp_path.iterdir())


def test_summary_preserves_gaps_but_exposes_no_raw_paths_or_extra_fields(tmp_path):
    path = save_report(tmp_path)
    before = path.read_bytes()
    result = load_dashboard_evidence(tmp_path)
    assert result["feeds"]["earnings"] == "MISSING"
    assert result["feeds"]["delivery"] == "UNAVAILABLE"
    assert result["membership_status_sessions"] == {"UNAVAILABLE_STALE_MEMBERSHIP": 1117}
    assert result["membership_integrity_sessions"] == {"OVERLAPPING_INTERVALS": 1222}
    assert result["missing_benchmark_sessions"] == ["2026-05-01"]
    assert result["price_asof_status"].startswith("UNAVAILABLE")
    assert "private" not in json.dumps(result)
    assert path.read_bytes() == before


def test_latest_recorded_end_wins_over_mtime_or_directory_name(tmp_path):
    old = save_report(tmp_path, "9999-new-looking-folder", end="2025-01-01")
    latest = save_report(tmp_path, "0000-old-looking-folder", end="2026-10-01")
    os.utime(latest, (1000, 1000))
    os.utime(old, (2000, 2000))
    assert load_dashboard_evidence(tmp_path)["end"] == "2026-10-01"


def test_invalid_reports_are_counted_while_valid_report_remains_usable(tmp_path):
    save_report(tmp_path, "valid")
    bad = save_report(tmp_path, "invalid")
    bad.write_text("not json", encoding="utf-8")
    result = load_dashboard_evidence(tmp_path)
    assert result["end"] == "2026-10-01"
    assert result["unreadable_reports"] == 1
    bad.unlink()
    (tmp_path / "coverage_audit" / "report.json").write_text("[]", encoding="utf-8")
    assert load_dashboard_evidence(tmp_path)["unreadable_reports"] == 1


@pytest.mark.parametrize("changes", [
    {"schema_version": 2}, {"feeds": []}, {"start": "invalid"}, {"end": None},
    {"start": "2027-01-01"}, {"status": None}, {"membership_status_sessions": []},
    {"membership_status_sessions": {"UNAVAILABLE": -1}},
    {"membership_integrity_sessions": {"OVERLAPPING_INTERVALS": True}},
    {"missing_benchmark_sessions": "2026-05-01"}, {"missing_benchmark_sessions": [None]},
])
def test_malformed_audits_never_certify_coverage(tmp_path, changes):
    save_report(tmp_path, **changes)
    assert load_dashboard_evidence(tmp_path)["status"] == "UNAVAILABLE_INVALID_AUDIT"


def test_unknown_feed_status_is_unavailable_not_trusted(tmp_path):
    save_report(tmp_path, feeds={
        "earnings": {"status": "VALID_MADE_UP"}, "delivery": {"status": []},
        "indices": {"status": "VALID_RECORDED_TIMESTAMPS_NOT_PROVIDER_VERIFIED"},
        "fundamentals": {"status": "INVALID"}, "flows": None,
    })
    feeds = load_dashboard_evidence(tmp_path)["feeds"]
    assert feeds["earnings"] == feeds["delivery"] == feeds["flows"] == "UNAVAILABLE"
    assert feeds["indices"] == "VALID_RECORDED_TIMESTAMPS_NOT_PROVIDER_VERIFIED"
    assert feeds["fundamentals"] == "INVALID"


def test_filesystem_errors_are_reported_not_raised(tmp_path, monkeypatch):
    save_report(tmp_path)

    def unreadable(*args, **kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(Path, "read_text", unreadable)
    assert load_dashboard_evidence(tmp_path)["status"] == "UNAVAILABLE_INVALID_AUDIT"
    monkeypatch.setattr(Path, "glob", unreadable)
    assert load_dashboard_evidence(tmp_path)["status"] == "UNAVAILABLE_UNREADABLE_AUDIT"


def test_endpoint_uses_application_paths_and_declares_auth_dependency(tmp_path, monkeypatch):
    import server

    save_report(tmp_path)
    app = object()
    persistence = SimpleNamespace(paths=SimpleNamespace(artifacts_dir=tmp_path))

    def configured_persistence(actual_app):
        assert actual_app is app
        return persistence

    monkeypatch.setattr(server, "_persistence", configured_persistence)
    result = server.get_dashboard_evidence(SimpleNamespace(app=app))
    assert result["feeds"]["earnings"] == "MISSING"
    route = next(route for route in server.router.routes if route.path == "/api/evidence")
    assert any(dependency.call is server.verify_api_key for dependency in route.dependant.dependencies)
