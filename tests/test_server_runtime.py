import asyncio
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest

import server
from core.config import CONFIG, IST, MarketPhase, parse_market_time
from core import data_provider
from core.runtime_paths import RuntimePaths
from core.services import PersistenceService


@pytest.mark.parametrize("error", [sqlite3.OperationalError("database unavailable"), OSError("read failed")])
def test_failed_killswitch_read_preserves_cause_and_blocks_scans(tmp_path, error, caplog):
    persistence = PersistenceService(RuntimePaths.discover(root=tmp_path))
    persistence.get_killswitch = Mock(side_effect=error)
    state = server.EngineState()
    with pytest.raises(RuntimeError, match="scans blocked") as raised:
        state.load_persisted_state(persistence)
    assert raised.value.__cause__ is error
    assert state.is_killed and str(error) in caplog.text


@pytest.mark.parametrize("payload", ["{", "[]", '{"candidates":[1]}',
                                     '{"scan_time":"new","candidates":[{"entry":"invalid"}]}'])
def test_corrupt_snapshot_does_not_partially_restore_or_clear_killswitch(tmp_path, payload, caplog):
    persistence = PersistenceService(RuntimePaths.discover(root=tmp_path))
    persistence.set_killswitch(True)
    (persistence.paths.state_dir / "latest_scan.json").write_text(payload, encoding="utf-8")
    state = server.EngineState()
    state.last_scan_time = "old"
    state.last_candidates = [{"ticker": "OLD"}]
    assert not state.load_persisted_state(persistence)
    assert state.is_killed and state.last_scan_time == "old"
    assert state.last_candidates == [{"ticker": "OLD"}]
    assert "Failed to re-hydrate" in caplog.text


def test_import_and_factory_do_not_open_database():
    code = "import sqlite3; sqlite3.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('database opened')); import server; assert server.app.state.persistence is None; assert server.create_app().state.persistence is None"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[1], timeout=30)
    assert result.returncode == 0, result.stderr


def test_scan_cannot_start_without_application_lifespan(monkeypatch):
    monkeypatch.setenv("API_KEY", "startup-test-key")
    application = server.create_app(auto_scan=False)
    response = TestClient(application).post("/api/scan/trigger", headers={"X-API-Key": "startup-test-key"})
    assert response.status_code == 503
    assert application.state.persistence is None and not application.state.engine.is_scanning


def test_app_instances_isolate_state_database_scans_and_events(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "isolated-test-key")
    first = server.create_app(paths=RuntimePaths.discover(root=tmp_path / "first"), auto_scan=False)
    second = server.create_app(paths=RuntimePaths.discover(root=tmp_path / "second"), auto_scan=False)
    captured = []

    def scan(**kwargs):
        captured.append((kwargs["services"].persistence.db.db_path, kwargs["regime_override"]))
        return [], [], None

    monkeypatch.setattr(server.svm, "run_scan", scan)
    headers = {"X-API-Key": "isolated-test-key"}
    with TestClient(first) as a, TestClient(second) as b:
        assert first.state.engine._lock is not second.state.engine._lock
        assert first.state.broadcaster is not second.state.broadcaster
        assert first.state.limiter is not second.state.limiter
        a.post("/api/regime/override", json={"regime": "PANIC"}, headers=headers)
        assert b.get("/api/status", headers=headers).json()["regime_override"] is None
        asyncio.run(server._run_scan_task_async(second))
        assert captured == [(second.state.persistence.db.db_path, None)]
        assert first.state.engine.last_scan_time is None
        assert second.state.engine.last_scan_time is not None
        assert not (first.state.persistence.paths.state_dir / "latest_scan.json").exists()
        assert (second.state.persistence.paths.state_dir / "latest_scan.json").exists()
        assert a.post("/api/killswitch", headers=headers).status_code == 200
        assert first.state.scan_cancel_event.is_set() and not second.state.scan_cancel_event.is_set()
        assert not b.get("/api/killswitch/status", headers=headers).json()["killswitch_active"]
        with a.stream("GET", "/api/events?limit=1", headers=headers) as events:
            payload = json.loads(next(line[6:] for line in events.iter_lines() if line.startswith("data: ")))
            assert payload["data"]["killswitch_active"]
        assert first.state.persistence.get_killswitch() and not second.state.persistence.get_killswitch()
    assert first.state.runtime is None and second.state.runtime is None


def test_failed_killswitch_reset_keeps_memory_blocked(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "failure-test-key")
    app = server.create_app(paths=RuntimePaths.discover(root=tmp_path), auto_scan=False)
    headers = {"X-API-Key": "failure-test-key"}
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.post("/api/killswitch", headers=headers).status_code == 200
        monkeypatch.setattr(app.state.persistence, "set_killswitch", Mock(side_effect=sqlite3.OperationalError("write failed")))
        assert client.post("/api/killswitch/reset", headers=headers).status_code == 500
        assert app.state.engine.is_killed and app.state.scan_cancel_event.is_set()
        assert client.post("/api/scan/trigger", headers=headers).status_code == 403


def test_failed_killswitch_activation_still_cancels_inflight_scan(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "failure-test-key")
    app = server.create_app(paths=RuntimePaths.discover(root=tmp_path), auto_scan=False)
    with TestClient(app, raise_server_exceptions=False) as client:
        monkeypatch.setattr(app.state.persistence, "set_killswitch", Mock(side_effect=sqlite3.OperationalError("write failed")))
        assert client.post("/api/killswitch", headers={"X-API-Key": "failure-test-key"}).status_code == 500
        assert app.state.engine.is_killed and app.state.scan_cancel_event.is_set()


def test_time_cache_survives_settings_reconstruction_and_config_replace():
    parse_market_time.cache_clear()
    morning = datetime(2026, 10, 2, 10, 0, tzinfo=IST)
    assert CONFIG.as_regime().session_from_time(morning) == "OPENING_RANGE"
    misses = parse_market_time.cache_info().misses
    assert CONFIG.as_regime().session_from_time(morning) == "OPENING_RANGE"
    assert parse_market_time.cache_info().misses == misses
    changed = replace(CONFIG, SESSION_OPEN_END="09:45")
    assert changed.session_from_time(morning) == "MIDDAY_CHOP"
    assert changed.as_regime().get_market_phase(morning) == MarketPhase.MIDDAY_CHOP


def test_shutdown_waits_for_scan_triggered_after_startup(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "shutdown-test-key")
    application = server.create_app(paths=RuntimePaths.discover(root=tmp_path), auto_scan=False)
    started = threading.Event()
    order = []
    runtime = SimpleNamespace(services=object(), close=lambda: order.append("runtime_closed"))
    monkeypatch.setattr(server, "build_application_runtime", lambda **kwargs: runtime)

    def scan(**kwargs):
        started.set()
        assert application.state.scan_cancel_event.wait(timeout=2.)
        assert kwargs["cancel_requested"]()
        order.append("worker_stopped")
        return [], [], None

    monkeypatch.setattr(server.svm, "run_scan", scan)

    async def run():
        async with server.lifespan(application):
            task = server._start_scan_task(application)
            assert await asyncio.to_thread(started.wait, 2.)
        assert task.done() and not application.state.engine.is_scanning

    asyncio.run(run())
    assert order == ["worker_stopped", "runtime_closed"]


def test_cache_failure_is_optional_but_memory_failure_propagates(monkeypatch):
    monkeypatch.setattr(data_provider, "_is_fresh", Mock(side_effect=OSError("cache unreadable")))
    assert data_provider._load_cached_daily("TEST.NS", "1y", 24) is None
    monkeypatch.setattr(data_provider, "_is_fresh", Mock(return_value=True))
    monkeypatch.setattr(data_provider.pd, "read_pickle", Mock(side_effect=MemoryError("out of memory")))
    with pytest.raises(MemoryError, match="out of memory"):
        data_provider._load_cached_daily("TEST.NS", "1y", 24)
