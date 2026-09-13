"""
tests/test_server.py
====================
Unit tests for FastAPI REST API endpoints in server.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from server import app, STATE, API_KEY


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


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

