"""
tests/test_server.py
====================
Unit tests for FastAPI REST API endpoints in server.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import server
from server import app, STATE, API_KEY


@pytest.fixture(autouse=True)
def reset_server_state():
    with STATE._lock:
        STATE.is_killed = False
        STATE.is_scanning = False
        STATE.scan_error = None
    server.PERSISTENCE.set_killswitch(False)
    yield
    with STATE._lock:
        STATE.is_killed = False
        STATE.is_scanning = False
        STATE.scan_error = None
    server.PERSISTENCE.set_killswitch(False)


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_server_startup_fails_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY", raising=False)
    with patch.object(server, "API_KEY", None):
        with pytest.raises(RuntimeError, match="CRITICAL SECURITY CONFIGURATION ERROR"):
            import asyncio
            async def run_startup():
                async with server.lifespan(server.app):
                    pass
            asyncio.run(run_startup())


def test_read_root(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200
    assert "html" in res.headers.get("content-type", "").lower()


def test_get_status(client: TestClient) -> None:
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "session" in data
    assert "regime_locked" in data


def test_get_scan_results_empty_initially(client: TestClient) -> None:
    res = client.get("/api/scan")
    assert res.status_code == 200
    data = res.json()
    assert "is_scanning" in data
    assert "portfolio" in data
    assert isinstance(data["portfolio"], list)


def test_set_regime_override(client: TestClient) -> None:
    # Set PANIC
    res = client.post("/api/regime/override", json={"regime": "PANIC"}, headers={"X-API-Key": API_KEY or ""})
    assert res.status_code == 200
    assert res.json()["regime_override"] == "PANIC"
    assert STATE.regime_override == "PANIC"

    # Clear override
    res_clear = client.post("/api/regime/override", json={"regime": "CLEAR"}, headers={"X-API-Key": API_KEY or ""})
    assert res_clear.status_code == 200
    assert res_clear.json()["regime_override"] is None
    assert STATE.regime_override is None


def test_set_invalid_regime_override(client: TestClient) -> None:
    res = client.post("/api/regime/override", json={"regime": "INVALID_REGIME"}, headers={"X-API-Key": API_KEY or ""})
    assert res.status_code == 400


def test_get_sectors(client: TestClient) -> None:
    res = client.get("/api/sectors")
    assert res.status_code == 200
    data = res.json()
    assert "total_sectors" in data
    assert data["total_sectors"] > 0
    assert "sectors" in data


def test_get_config(client: TestClient) -> None:
    res = client.get("/api/config")
    assert res.status_code == 200
    data = res.json()
    assert "benchmark" in data
    assert "risk_per_trade_inr" in data
    assert "regime_settings" in data


def test_trigger_scan_unauthorized(client: TestClient) -> None:
    # No header
    res = client.post("/api/scan/trigger")
    assert res.status_code == 401

    # Wrong header
    res_wrong = client.post("/api/scan/trigger", headers={"X-API-Key": "invalid_key"})
    assert res_wrong.status_code == 401


def test_trigger_scan_already_running(client: TestClient) -> None:
    with STATE._lock:
        STATE.is_scanning = True

    try:
        res = client.post("/api/scan/trigger", headers={"X-API-Key": API_KEY or ""})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "already_running"
    finally:
        with STATE._lock:
            STATE.is_scanning = False


def test_trigger_scan_authorized(client: TestClient) -> None:
    with STATE._lock:
        STATE.is_scanning = False

    with patch("server.svm.run_scan", return_value=([], [], None)):
        res = client.post("/api/scan/trigger", headers={"X-API-Key": API_KEY or ""})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "triggered"


def test_get_trades(client: TestClient) -> None:
    res = client.get("/api/trades")
    assert res.status_code == 200
    data = res.json()
    assert "count" in data
    assert "trades" in data
    assert isinstance(data["trades"], list)

    # limit=0 is rejected by ge=1
    res_zero = client.get("/api/trades?limit=0")
    assert res_zero.status_code == 422

    # valid limit
    res_valid = client.get("/api/trades?limit=50")
    assert res_valid.status_code == 200


def test_format_ticker_result_preserves_engine_swing_horizon_with_tight_stop() -> None:
    from unittest.mock import MagicMock
    from server import _format_ticker_result
    from core.scorer import TickerResult

    res = MagicMock(spec=TickerResult)
    res.direction = "LONG"
    res.entry = 1000.0
    res.stop = 985.0  # 1.5% stop (< 2.5%)
    res.t1 = 1050.0
    res.reasons = []
    res.trade_horizon = "SWING"
    res.__dict__ = {
        "direction": "LONG",
        "entry": 1000.0,
        "stop": 985.0,
        "t1": 1050.0,
        "trade_horizon": "SWING",
    }
    formatted = _format_ticker_result(res)
    assert formatted["trade_horizon"] == "SWING"
    assert formatted["horizon_label"] == "SWING (CNC)"


def test_enrich_preserves_persisted_swing_horizon_with_tight_stop(tmp_path: pytest.TempPathFactory) -> None:
    import json
    from pathlib import Path
    from unittest.mock import MagicMock

    target_dir = Path(str(tmp_path)) / "state"
    target_dir.mkdir(parents=True, exist_ok=True)
    scan_file = target_dir / "latest_scan.json"
    scan_data = {
        "scan_time": "2026-09-13T00:00:00Z",
        "candidates": [{
            "direction": "LONG",
            "entry": 1000.0,
            "stop": 985.0,  # 1.5% stop (< 2.5%)
            "t1": 1050.0,
            "reasons": [],
            "trade_horizon": "SWING",
        }],
        "portfolio": [],
    }
    scan_file.write_text(json.dumps(scan_data), encoding="utf-8")

    mock_persistence = MagicMock()
    mock_persistence.paths.state_dir = target_dir

    STATE.load_persisted_state(mock_persistence)
    assert len(STATE.last_candidates) == 1
    assert STATE.last_candidates[0]["trade_horizon"] == "SWING"
    assert STATE.last_candidates[0]["horizon_label"] == "SWING (CNC)"


