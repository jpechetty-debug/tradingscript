import warnings

from fastapi.testclient import TestClient
import numpy as np
import pandas as pd
import pytest

from core.config import CONFIG
from core.cache import _build_corr_matrix
from core.portfolio import _ticker_excess_kurtosis
from core.runtime_paths import RuntimePaths
from core.services import PersistenceService
import server
from server import app, EngineState, STATE, API_KEY_NAME

client = TestClient(app)
AUTH_HEADERS = {API_KEY_NAME: server.API_KEY or "test-api-key"}


@pytest.fixture(autouse=True)
def reset_killswitch_state():
    """Ensure clean killswitch state for every test."""
    with STATE._lock:
        STATE.is_killed = False
        STATE.is_scanning = False
        STATE.scan_error = None
    yield
    with STATE._lock:
        STATE.is_killed = False
        STATE.is_scanning = False
        STATE.scan_error = None


def test_killswitch_status_public():
    response = client.get("/api/killswitch/status")
    assert response.status_code == 200
    data = response.json()
    assert "killswitch_active" in data
    assert data["killswitch_active"] is False


def test_killswitch_activation_unauthorized():
    response = client.post("/api/killswitch")
    assert response.status_code == 401


def test_killswitch_lifecycle():
    # 1. Activate killswitch
    res = client.post("/api/killswitch", headers=AUTH_HEADERS)
    assert res.status_code == 200
    assert res.json()["killswitch_active"] is True

    # 2. Check status reflects killed
    status_res = client.get("/api/killswitch/status")
    assert status_res.json()["killswitch_active"] is True

    # Check /api/status also reflects killswitch
    engine_status = client.get("/api/status").json()
    assert engine_status["killswitch_active"] is True

    # 3. Triggering scan must be blocked with 403 Forbidden
    trigger_res = client.post("/api/scan/trigger", headers=AUTH_HEADERS)
    assert trigger_res.status_code == 403
    assert "Killswitch is active" in trigger_res.json()["detail"]

    # 4. Reset killswitch
    reset_res = client.post("/api/killswitch/reset", headers=AUTH_HEADERS)
    assert reset_res.status_code == 200
    assert reset_res.json()["killswitch_active"] is False

    # 5. Status is now clean
    assert client.get("/api/killswitch/status").json()["killswitch_active"] is False


def test_engine_state_persistence_and_rehydration(tmp_path):
    paths = RuntimePaths.discover(root=tmp_path)
    persistence = PersistenceService(paths=paths)

    state = EngineState()
    state.last_scan_time = "2026-09-09T09:00:00Z"
    state.last_candidates = [{"ticker": "RELIANCE.NS", "score": 0.88}]
    state.last_portfolio = [{"ticker": "TCS.NS", "score": 0.92}]
    state.last_sector_rs = {"IT": 1.25, "ENERGY": 0.95}
    state.last_regime_info = {"regime": "TREND_UP", "breadth": 0.65}

    # Persist
    state.persist_state(persistence)
    target_file = paths.state_dir / "latest_scan.json"
    assert target_file.exists()

    # Re-hydrate into a fresh EngineState
    new_state = EngineState()
    assert new_state.last_scan_time is None
    assert new_state.last_candidates == []

    loaded = new_state.load_persisted_state(persistence)
    assert loaded is True
    assert new_state.last_scan_time == "2026-09-09T09:00:00Z"
    assert len(new_state.last_candidates) == 1
    assert new_state.last_candidates[0]["ticker"] == "RELIANCE.NS"
    assert len(new_state.last_portfolio) == 1
    assert new_state.last_sector_rs["IT"] == 1.25
    assert new_state.last_regime_info["regime"] == "TREND_UP"


def test_pct_change_warning_eliminated():
    dates = pd.date_range("2026-01-01", periods=100)
    df1 = pd.DataFrame({"Close": np.linspace(100, 200, 100)}, index=dates)
    df2 = pd.DataFrame({"Close": np.linspace(50, 150, 100)}, index=dates)
    processed = {"INFY.NS": df1, "WIPRO.NS": df2}

    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always")
        corr = _build_corr_matrix(processed, config=CONFIG)
        kurt = _ticker_excess_kurtosis(df1, config=CONFIG)

        # Ensure no FutureWarnings about fill_method are raised
        future_warnings = [
            w for w in recorded_warnings
            if issubclass(w.category, FutureWarning) and "fill_method" in str(w.message)
        ]
        assert len(future_warnings) == 0, f"Unexpected FutureWarning: {future_warnings}"
        assert not corr.empty
        assert isinstance(kurt, float)
