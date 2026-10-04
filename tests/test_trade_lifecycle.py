from types import SimpleNamespace
from dataclasses import replace
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import server
from core.config import CONFIG
from core.runtime_paths import RuntimePaths
from core.services import AlertService, PersistenceService, ScanService


@pytest.fixture
def client(tmp_path, monkeypatch):
    persistence = PersistenceService(RuntimePaths.discover(root=tmp_path))
    monkeypatch.setattr(server.app.state, "persistence", persistence)
    monkeypatch.setenv("API_KEY", "lifecycle-test-key")
    return TestClient(server.app), persistence


def fill(**overrides):
    return dict(trade_id="fill-1", ticker="TEST.NS", direction="LONG", entry_price=100.,
                entry_ts="2024-01-02T10:00:00+05:30", stop_loss=90., target=120., shares=10,
                composite=.7, prob_win=.6, factors={"trend": .7}, **overrides)


HEADERS = {"X-API-Key": "lifecycle-test-key"}
EXIT = dict(exit_price=110., exit_ts="2024-01-04T10:00:00+05:30", costs=20., exit_reason="MANUAL")


def test_api_fill_close_is_idempotent_and_cost_adjusted(client):
    http, persistence = client
    assert http.post("/api/trades/fills", json=fill()).status_code == 401
    for _ in range(2):
        assert http.post("/api/trades/fills", json=fill(), headers=HEADERS).status_code == 200
    assert persistence.db.count_executed_trades() == 1
    assert persistence.load_open_positions()["TEST"]["shares"] == 10
    service = ScanService(version="test", persistence=persistence)
    frame = pd.DataFrame({"Open": [100.], "High": [130.], "Low": [80.], "Close": [100.]},
                         index=pd.to_datetime(["2024-01-04"]))
    service._monitor_open_position_stops({"TEST.NS": frame}, persistence.create_scan_state(CONFIG))
    assert "TEST" in persistence.load_open_positions()
    for _ in range(2):
        result = http.post("/api/trades/fill-1/close", json=EXIT, headers=HEADERS)
        assert result.status_code == 200
        assert result.json()["net_pnl"] == 80.
        assert result.json()["realised_r"] == .8
    assert persistence.load_open_positions() == {}
    assert persistence.db.count_trades() == 1
    assert persistence.db.fetch_calibration_trades(min_samples=1) == ([.7], [1])
    # Retried entry events after closure must not reopen a position.
    http.post("/api/trades/fills", json=fill(), headers=HEADERS)
    assert persistence.load_open_positions() == {}


def test_fill_validation_conflicts_and_atomic_rollback(client, monkeypatch):
    http, persistence = client
    invalid = fill()
    invalid["stop_loss"] = 101.
    assert http.post("/api/trades/fills", json=invalid, headers=HEADERS).status_code == 422
    original = persistence.db.upsert_open_position
    monkeypatch.setattr(persistence.db, "upsert_open_position", Mock(side_effect=RuntimeError("disk error")))
    with pytest.raises(RuntimeError):
        http.post("/api/trades/fills", json=fill(), headers=HEADERS)
    assert persistence.db.count_executed_trades() == 0
    monkeypatch.setattr(persistence.db, "upsert_open_position", original)
    http.post("/api/trades/fills", json=fill(), headers=HEADERS)
    changed = fill()
    changed["shares"] = 11
    assert http.post("/api/trades/fills", json=changed, headers=HEADERS).status_code == 409
    assert http.post("/api/trades/missing/close", json=EXIT, headers=HEADERS).status_code == 404
    earlier = {**EXIT, "exit_ts": "2024-01-01T10:00:00+05:30"}
    assert http.post("/api/trades/fill-1/close", json=earlier, headers=HEADERS).status_code == 409
    assert persistence.db.count_trades() == 0
    assert "TEST" in persistence.load_open_positions()


def test_paper_exit_updates_both_books_but_excludes_real_calibration(client):
    http, persistence = client
    http.post("/api/trades/fills", json=fill(source="SIMULATED"), headers=HEADERS)
    frame = pd.DataFrame({"Open": [100.], "High": [110.], "Low": [80.], "Close": [95.]},
                         index=pd.to_datetime(["2024-01-04"]))
    ScanService(version="test", persistence=persistence)._monitor_open_position_stops(
        {"TEST.NS": frame}, persistence.create_scan_state(CONFIG))
    assert persistence.load_open_positions() == {}
    assert persistence.db.fetch_executed_trades()[0]["outcome"] == "LOSS"
    assert persistence.db.fetch_calibration_trades(min_samples=1) == ([], [])


@pytest.mark.parametrize("custom", [False, True])
def test_alert_retries_failed_send_and_only_deduplicates_sent_picks(custom):
    sender = Mock(side_effect=[False, True, True])
    service = AlertService(version="test", messenger=sender,
                           alerter=SimpleNamespace(send_daily_summary=sender) if custom else None)
    config = replace(CONFIG, TELEGRAM_BOT_TOKEN="token", TELEGRAM_CHAT_ID="chat",
                     TELEGRAM_ALERT_MIN_PROB=.5, TELEGRAM_ALERT_TOP_N=1)
    picks = [SimpleNamespace(ticker=t, prob_win=.7, expectancy_r=.5, entry=100., stop=90.,
                             t1=120., shares=10, direction="LONG", trade_horizon="SWING") for t in ["A", "B"]]
    regime = SimpleNamespace(label="TREND_UP")
    service.send_portfolio_summary(picks, regime, config)
    assert service._last_alerted == {}
    service.send_portfolio_summary(picks, regime, config)
    assert set(service._last_alerted) == {"A:SWING"}
    service.send_portfolio_summary(picks, regime, config)
    assert set(service._last_alerted) == {"A:SWING", "B:SWING"}
    service.send_portfolio_summary(picks, regime, config)
    assert sender.call_count == 3
