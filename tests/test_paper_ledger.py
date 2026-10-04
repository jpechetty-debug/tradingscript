from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
import json
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pandas as pd
import pytest

import server
from core.config import CONFIG
from core.database import SqliteDatabase
from core.factors import FactorScores
from core.paper_ledger import PaperLedger
from core.runtime_paths import RuntimePaths
from core.regime import MarketRegime
from core.config import MarketRegimeType
from core.services import PersistenceService, PreparedScanData, ScanService
from core.trades import TradeExit, TradeFill, TradeLifecycle


def signal():
    return SimpleNamespace(ticker="TEST", strategy_id="SWING_BREAKOUT_V1", signal_time="2026-10-01T00:00:00",
                           entry=100., entry_min=99., entry_max=101., stop=98., t1=104., time_stop_bars=15,
                           shares=10, probability_status="HEURISTIC_UNVALIDATED", regime="TREND_UP",
                           composite=.7, prob_win=.6, factors=FactorScores(*([.7] * 8), ic_weights={}),
                           is_watchlist=False, is_held=False)


def fill(**overrides):
    payload = dict(trade_id="paper-1", ticker="TEST", direction="LONG", source="SIMULATED",
                   entry_price=100., entry_ts="2024-01-02T10:00:00+05:30", stop_loss=98., target=104.,
                   shares=10, composite=.7, prob_win=.6, factors={"trend": .7})
    return TradeFill(**(payload | overrides))


def test_journal_is_immutable_idempotent_and_concurrent(tmp_path):
    db = SqliteDatabase(tmp_path / "state.db")
    with ThreadPoolExecutor(max_workers=4) as pool:
        events = list(pool.map(lambda _: db.append_paper_event("one", "SIGNAL", {"ticker": "A"}), range(4)))
    assert all(event == events[0] for event in events)
    assert len(db.fetch_paper_events()) == 1
    with pytest.raises(ValueError, match="different"):
        db.append_paper_event("one", "SIGNAL", {"ticker": "B"})
    for statement in ("UPDATE paper_events SET payload = '{}'", "DELETE FROM paper_events",
                      "INSERT OR REPLACE INTO paper_events SELECT * FROM paper_events"):
        with pytest.raises(sqlite3.IntegrityError):
            with db.get_connection() as conn:
                conn.execute(statement)
    assert db.fetch_paper_events()[0] == events[0]


def test_jsonl_export_appends_and_refuses_altered_history(tmp_path):
    db = SqliteDatabase(tmp_path / "state.db")
    ledger = PaperLedger(db)
    path = tmp_path / "daily.jsonl"
    db.append_paper_event("a", "SIGNAL", {"x": 1})
    assert ledger.export_jsonl(path) == 1
    prefix = path.read_bytes()
    db.append_paper_event("b", "REJECTION", {"x": 2})
    assert ledger.export_jsonl(path) == 1 and path.read_bytes().startswith(prefix)
    assert ledger.export_jsonl(path) == 0
    assert [json.loads(line)["sequence"] for line in path.read_text().splitlines()] == [1, 2]
    path.write_text("changed history\n")
    with pytest.raises(ValueError, match="refusing"):
        ledger.export_jsonl(path)
    assert path.read_text() == "changed history\n"
    assert not path.with_suffix(".jsonl.lock").exists()


def test_first_observed_signal_is_preserved_and_variants_are_distinct(tmp_path):
    db = SqliteDatabase(tmp_path / "state.db")
    ledger = PaperLedger(db)
    pick = signal()
    pick.research_context = {"rs_63_percentile": 90., "breadth_ema200": .6}
    first = ledger.record_signals([pick], [pick], CONFIG)[0]
    pick.shares = 5
    pick.research_context["rs_63_percentile"] = 100.
    assert ledger.record_signals([pick], [], CONFIG) == [first]
    assert first["payload"]["selected"] and first["payload"]["shares"] == 10
    assert first["payload"]["research_context"]["rs_63_percentile"] == 90.
    ledger.record_signals([pick], [], replace(CONFIG, SWING_BREAKOUT_LOOKBACK=15))
    assert len(db.fetch_paper_events()) == 2
    pick.is_held = True
    assert ledger.record_signals([pick], [], CONFIG) == []


def test_linked_fill_exit_events_are_atomic_and_paginated(tmp_path, monkeypatch):
    persistence = PersistenceService(RuntimePaths.discover(root=tmp_path))
    db = persistence.db
    event = PaperLedger(db).record_signals([signal()], [signal()], CONFIG)[0]
    monkeypatch.setattr(server.app.state, "persistence", persistence)
    monkeypatch.setenv("API_KEY", "paper-test-key")
    http = TestClient(server.app)
    headers = {"X-API-Key": "paper-test-key"}
    payload = fill(strategy_id="SWING_BREAKOUT_V1", signal_event_id=event["event_id"],
                   entry_ts=datetime.now(timezone.utc), fees_inr=.5, slippage_inr=.5, cost_basis="MODELED").model_dump(mode="json")
    assert http.get("/api/paper/events").status_code == 401
    for _ in range(2):
        assert http.post("/api/trades/fills", json=payload, headers=headers).status_code == 200
    exit_payload = dict(exit_price=103., exit_ts=datetime.now(timezone.utc).isoformat(), costs=2.,
                        exit_reason="TIME", fees_inr=1., slippage_inr=1., cost_basis="MODELED")
    for _ in range(2):
        result = http.post("/api/trades/paper-1/close", json=exit_payload, headers=headers)
        assert result.status_code == 200 and result.json()["net_pnl"] == 28.
    events = db.fetch_paper_events()
    assert [e["event_type"] for e in events] == ["SIGNAL", "PAPER_FILL", "PAPER_EXIT"]
    assert events[1]["payload"]["signal_precedes_entry"] and not events[1]["payload"]["future_dated_event"]
    assert events[1]["payload"]["within_entry_bounds"]
    assert events[2]["payload"]["fees"] == 1. and events[2]["payload"]["cost_basis"] == "MODELED"
    page = http.get("/api/paper/events?limit=1", headers=headers).json()
    assert page["next_after"] == 1
    assert len(http.get("/api/paper/events?after=1", headers=headers).json()["events"]) == 2
    changed = exit_payload | {"exit_reason": "CHANGED"}
    assert http.post("/api/trades/paper-1/close", json=changed, headers=headers).status_code == 409
    assert len(db.fetch_paper_events()) == 3 and persistence.load_open_positions() == {}


def test_backdated_fill_is_not_prospective_and_failed_journal_rolls_back(tmp_path, monkeypatch):
    db = SqliteDatabase(tmp_path / "state.db")
    lifecycle = TradeLifecycle(db)
    event = PaperLedger(db).record_signals([signal()], [signal()], CONFIG)[0]
    paper_fill = fill(strategy_id="SWING_BREAKOUT_V1", signal_event_id=event["event_id"])
    original = db.append_paper_event
    monkeypatch.setattr(db, "append_paper_event", Mock(side_effect=OSError("disk error")))
    with pytest.raises(OSError):
        lifecycle.register(paper_fill)
    assert db.count_executed_trades() == 0 and db.fetch_open_positions() == {}
    monkeypatch.setattr(db, "append_paper_event", original)
    lifecycle.register(paper_fill)
    recorded = db.fetch_paper_events()[1]
    assert not recorded["payload"]["signal_precedes_entry"]
    assert recorded["payload"]["entry_recording_delay_seconds"] > 86400
    monkeypatch.setattr(db, "delete_open_position", Mock(side_effect=OSError("disk error")))
    with pytest.raises(OSError):
        lifecycle.close("paper-1", TradeExit(exit_price=103., exit_ts="2024-01-03T10:00:00+05:30", costs=2., exit_reason="TIME"))
    assert db.fetch_executed_trades()[0]["outcome"] == "OPEN" and db.fetch_open_positions()
    assert len(db.fetch_paper_events()) == 2 and db.count_trades() == 0


def test_invalid_signal_link_cannot_create_position(tmp_path):
    db = SqliteDatabase(tmp_path / "state.db")
    with pytest.raises(ValueError, match="Signal link"):
        TradeLifecycle(db).register(fill(signal_event_id="missing"))
    assert db.count_executed_trades() == 0 and db.fetch_paper_events() == []


def test_swing_scan_records_signals_without_creating_fills(tmp_path, monkeypatch):
    persistence = PersistenceService(RuntimePaths.discover(root=tmp_path))
    pick = signal()
    pick.sector, pick.direction = "IT", "LONG"
    pick.sharpe_rank, pick.risk_inr = .5, 20.
    pick.prob_win, pick.expectancy_r, pick.rr_t1 = .7, .5, 2.
    frame = pd.DataFrame({"Close": [99., 100.], "EMA_50": [98., 99.]}, index=pd.bdate_range("2026-09-30", periods=2))
    prepared = PreparedScanData({}, {CONFIG.BENCHMARK: frame, "TEST.NS": frame}, frame.Close, {}, {})
    service = ScanService(version="test", persistence=persistence,
                          data_service=SimpleNamespace(prepare_scan_data=lambda *a, **kw: prepared))
    monkeypatch.setattr(service, "_score_candidates", lambda **kw: ([pick], 0.))
    monkeypatch.setattr(service, "_maybe_recalibrate_ic_weights", lambda **kw: None)
    monkeypatch.setattr("core.services.classify_regime", lambda *a, **kw: MarketRegime(MarketRegimeType.TREND_UP, .8, 25., 1., .9, True))
    service.scan(replace(CONFIG, SWING_SETUP_ENABLED=True))
    events = persistence.db.fetch_paper_events()
    assert [event["event_type"] for event in events] == ["SIGNAL", "SCAN"]
    assert persistence.load_open_positions() == {} and persistence.db.count_executed_trades() == 0


def test_swing_regime_confirmation_counts_dates_not_repeated_scans(tmp_path, monkeypatch):
    from core.regime import RegimeTracker
    persistence = PersistenceService(RuntimePaths.discover(root=tmp_path))
    frame = pd.DataFrame({"Close": [99., 101.], "EMA_50": [100., 100.], "ADX": 25.,
                          "ATR": 1., "ATR_50_mean": 1.}, index=pd.bdate_range("2026-09-29", periods=2))
    prepared = PreparedScanData({}, {CONFIG.BENCHMARK: frame, "TEST.NS": frame}, frame.Close, {}, {})
    data = SimpleNamespace(prepare_scan_data=lambda *a, **kw: prepared)
    service = ScanService(version="test", persistence=persistence, data_service=data)
    confirmed = []

    def score(**kw):
        confirmed.append(kw["regime"].confirmed)
        return [], 0.

    monkeypatch.setattr(service, "_score_candidates", score)
    monkeypatch.setattr(service, "_maybe_recalibrate_ic_weights", lambda **kw: None)
    tracker = RegimeTracker()
    for _ in range(2):
        service.scan(replace(CONFIG, SWING_SETUP_ENABLED=True), regime_tracker=tracker)
    assert confirmed == [False, False]
    assert len(persistence.db.fetch_paper_events()) == 2
    extra = frame.iloc[[-1]].copy()
    extra.index = pd.DatetimeIndex([frame.index[-1] + pd.offsets.BDay()])
    frame = pd.concat([frame, extra])
    prepared = PreparedScanData({}, {CONFIG.BENCHMARK: frame, "TEST.NS": frame}, frame.Close, {}, {})
    service.scan(replace(CONFIG, SWING_SETUP_ENABLED=True), regime_tracker=tracker)
    assert confirmed[-1]
