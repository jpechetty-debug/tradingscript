"""
tests/test_server.py
====================
Unit tests for FastAPI REST API endpoints in server.py.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server import app, STATE


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
    res = client.post("/api/regime/override", json={"regime": "PANIC"})
    assert res.status_code == 200
    assert res.json()["regime_override"] == "PANIC"
    assert STATE.regime_override == "PANIC"

    # Clear override
    res_clear = client.post("/api/regime/override", json={"regime": "CLEAR"})
    assert res_clear.status_code == 200
    assert res_clear.json()["regime_override"] is None
    assert STATE.regime_override is None


def test_set_invalid_regime_override(client: TestClient) -> None:
    res = client.post("/api/regime/override", json={"regime": "INVALID_REGIME"})
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
