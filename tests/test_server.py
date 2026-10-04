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
from server import app

TEST_API_KEY = "test-secret-key-12345"


@pytest.fixture(autouse=True)
def ensure_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    monkeypatch.setattr(server, "API_KEY", TEST_API_KEY)
    yield TEST_API_KEY


@pytest.fixture(autouse=True)
def reset_server_state():
    with server.app.state.engine._lock:
        server.app.state.engine.is_killed = False
        server.app.state.engine.is_scanning = False
        server.app.state.engine.scan_error = None
    server.app.state.persistence.set_killswitch(False)
    yield
    with server.app.state.engine._lock:
        server.app.state.engine.is_killed = False
        server.app.state.engine.is_scanning = False
        server.app.state.engine.scan_error = None
    server.app.state.persistence.set_killswitch(False)


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


def test_server_startup_fails_with_template_placeholder_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "your_secure_api_key_here")
    with patch.object(server, "API_KEY", "your_secure_api_key_here"):
        with pytest.raises(RuntimeError, match="unedited template placeholder"):
            import asyncio
            async def run_startup():
                async with server.lifespan(server.app):
                    pass
            asyncio.run(run_startup())


def test_server_startup_fails_with_insecure_key_on_public_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "sovereign-dev-secret-key")
    monkeypatch.setenv("HOST", "0.0.0.0")
    with patch.object(server, "API_KEY", "sovereign-dev-secret-key"):
        with pytest.raises(RuntimeError, match="insecure default placeholder"):
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
    # Unauthenticated request must return 401
    res_unauth = client.get("/api/status")
    assert res_unauth.status_code == 401

    # Authenticated request must succeed
    res = client.get("/api/status", headers={"X-API-Key": TEST_API_KEY})
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "version" in data
    assert "session" in data
    assert "regime_locked" in data
    assert "platt_calibration" in data
    assert "a" in data["platt_calibration"]
    assert "b" in data["platt_calibration"]
    assert "is_calibrated" in data["platt_calibration"]


def test_get_scan_results_empty_initially(client: TestClient) -> None:
    # Unauthenticated request must return 401
    res_unauth = client.get("/api/scan")
    assert res_unauth.status_code == 401

    # Authenticated request must succeed
    res = client.get("/api/scan", headers={"X-API-Key": TEST_API_KEY})
    assert res.status_code == 200
    data = res.json()
    assert "is_scanning" in data
    assert "portfolio" in data
    assert isinstance(data["portfolio"], list)


def test_set_regime_override(client: TestClient) -> None:
    # Set PANIC
    res = client.post("/api/regime/override", json={"regime": "PANIC"}, headers={"X-API-Key": TEST_API_KEY})
    assert res.status_code == 200
    assert res.json()["regime_override"] == "PANIC"
    assert server.app.state.engine.regime_override == "PANIC"

    # Clear override
    res_clear = client.post("/api/regime/override", json={"regime": "CLEAR"}, headers={"X-API-Key": TEST_API_KEY})
    assert res_clear.status_code == 200
    assert res_clear.json()["regime_override"] is None
    assert server.app.state.engine.regime_override is None


def test_set_invalid_regime_override(client: TestClient) -> None:
    res = client.post("/api/regime/override", json={"regime": "INVALID_REGIME"}, headers={"X-API-Key": TEST_API_KEY})
    assert res.status_code == 400


def test_get_sectors(client: TestClient) -> None:
    assert client.get("/api/sectors").status_code == 401
    res = client.get("/api/sectors", headers={"X-API-Key": TEST_API_KEY})
    assert res.status_code == 200
    data = res.json()
    assert "total_sectors" in data
    assert data["total_sectors"] > 0
    assert "sectors" in data


def test_get_config(client: TestClient) -> None:
    assert client.get("/api/config").status_code == 401
    res = client.get("/api/config", headers={"X-API-Key": TEST_API_KEY})
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
    with server.app.state.engine._lock:
        server.app.state.engine.is_scanning = True

    try:
        res = client.post("/api/scan/trigger", headers={"X-API-Key": TEST_API_KEY})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "already_running"
    finally:
        with server.app.state.engine._lock:
            server.app.state.engine.is_scanning = False


def test_trigger_scan_authorized(client: TestClient) -> None:
    with server.app.state.engine._lock:
        server.app.state.engine.is_scanning = False

    with patch("server.svm.run_scan", return_value=([], [], None)):
        res = client.post("/api/scan/trigger", headers={"X-API-Key": TEST_API_KEY})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "triggered"


def test_get_trades_requires_auth(client: TestClient) -> None:
    # Missing API key
    res_no_key = client.get("/api/trades")
    assert res_no_key.status_code == 401

    # Invalid API key
    res_bad_key = client.get("/api/trades", headers={"X-API-Key": "wrong-key"})
    assert res_bad_key.status_code == 401


def test_get_trades(client: TestClient) -> None:
    headers = {"X-API-Key": TEST_API_KEY}
    res = client.get("/api/trades", headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert "count" in data
    assert "trades" in data
    assert isinstance(data["trades"], list)

    # limit=0 is rejected by ge=1
    res_zero = client.get("/api/trades?limit=0", headers=headers)
    assert res_zero.status_code == 422

    # valid limit
    res_valid = client.get("/api/trades?limit=50", headers=headers)
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

    server.app.state.engine.load_persisted_state(mock_persistence)
    assert len(server.app.state.engine.last_candidates) == 1
    assert server.app.state.engine.last_candidates[0]["trade_horizon"] == "SWING"
    assert server.app.state.engine.last_candidates[0]["horizon_label"] == "SWING (CNC)"


def test_format_ticker_result_preserves_engine_swing_horizon_for_short() -> None:
    from unittest.mock import MagicMock
    from server import _format_ticker_result
    from core.scorer import TickerResult

    res = MagicMock(spec=TickerResult)
    res.direction = "SHORT"
    res.entry = 1000.0
    res.stop = 1015.0  # 1.5% stop
    res.t1 = 950.0
    res.reasons = []
    res.trade_horizon = "SWING"
    res.__dict__ = {
        "direction": "SHORT",
        "entry": 1000.0,
        "stop": 1015.0,
        "t1": 950.0,
        "trade_horizon": "SWING",
    }
    formatted = _format_ticker_result(res)
    assert formatted["action"] == "SELL"
    assert formatted["trade_horizon"] == "SWING"
    assert formatted["horizon_label"] == "SWING (CNC)"


def test_enrich_preserves_persisted_swing_horizon_for_short(tmp_path: pytest.TempPathFactory) -> None:
    import json
    from pathlib import Path
    from unittest.mock import MagicMock

    target_dir = Path(str(tmp_path)) / "state"
    target_dir.mkdir(parents=True, exist_ok=True)
    scan_file = target_dir / "latest_scan.json"
    scan_data = {
        "scan_time": "2026-09-13T00:00:00Z",
        "candidates": [{
            "direction": "SHORT",
            "entry": 1000.0,
            "stop": 1015.0,
            "t1": 950.0,
            "reasons": [],
            "trade_horizon": "SWING",
        }],
        "portfolio": [],
    }
    scan_file.write_text(json.dumps(scan_data), encoding="utf-8")

    mock_persistence = MagicMock()
    mock_persistence.paths.state_dir = target_dir

    server.app.state.engine.load_persisted_state(mock_persistence)
    assert len(server.app.state.engine.last_candidates) == 1
    assert server.app.state.engine.last_candidates[0]["action"] == "SELL"
    assert server.app.state.engine.last_candidates[0]["trade_horizon"] == "SWING"
    assert server.app.state.engine.last_candidates[0]["horizon_label"] == "SWING (CNC)"


def test_get_client_ip_trusted_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    from unittest.mock import MagicMock
    from server import get_client_ip

    mock_request = MagicMock()
    mock_request.client.host = "10.0.0.1"
    mock_request.headers = {
        "x-forwarded-for": "203.0.113.195, 10.0.0.1",
        "x-real-ip": "203.0.113.195",
    }

    # When proxy trust is disabled, returns direct client host
    monkeypatch.delenv("TRUSTED_PROXIES", raising=False)
    monkeypatch.delenv("TRUST_PROXIES", raising=False)
    assert get_client_ip(mock_request) == "10.0.0.1"

    # When proxy trust is enabled, extracts leftmost client IP
    monkeypatch.setenv("TRUSTED_PROXIES", "10.0.0.1/32")
    assert get_client_ip(mock_request) == "203.0.113.195"

    monkeypatch.setenv("TRUSTED_PROXIES", "192.0.2.0/24")
    assert get_client_ip(mock_request) == "10.0.0.1"

    # Falls back to x-real-ip if x-forwarded-for is missing
    monkeypatch.setenv("TRUSTED_PROXIES", "10.0.0.1/32")
    mock_request.headers = {"x-real-ip": "198.51.100.22"}
    assert get_client_ip(mock_request) == "198.51.100.22"


def test_sse_subscriber_limit_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio
    BROADCASTER = server.app.state.broadcaster

    # Temporarily set cap to small number for testing
    orig_cap = BROADCASTER.max_subscribers
    BROADCASTER.max_subscribers = 2
    try:
        # Subscribe 2 test queues directly
        q1 = asyncio.run(BROADCASTER.subscribe())
        q2 = asyncio.run(BROADCASTER.subscribe())

        # Third attempt should be rejected with 503
        res = client.get("/api/events", headers={"X-API-Key": TEST_API_KEY})
        assert res.status_code == 503
        assert "Max SSE subscribers reached" in res.json()["detail"]

        # Clean up
        asyncio.run(BROADCASTER.unsubscribe(q1))
        asyncio.run(BROADCASTER.unsubscribe(q2))
    finally:
        BROADCASTER.max_subscribers = orig_cap


