"""
tests/test_database.py
======================
Unit tests for raw SQL SQLite database layer (core/database.py)
and PersistenceService SQLite integration.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.database import SqliteDatabase
from core.runtime_paths import RuntimePaths
from core.services import PersistenceService
from core.config import CONFIG


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "test_state.db"


@pytest.fixture
def db(db_path: Path) -> SqliteDatabase:
    return SqliteDatabase(db_path)


def test_sqlite_wal_mode_and_tables(db: SqliteDatabase) -> None:
    with db.get_connection() as conn:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.upper() == "WAL"

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "portfolio_state" in table_names
        assert "platt_calibration" in table_names
        assert "trade_log" in table_names
        assert "factor_weights" in table_names


def test_portfolio_state_crud(db: SqliteDatabase) -> None:
    assert db.fetch_latest_portfolio_state() is None

    db.upsert_portfolio_state(current_nav=1_000_000.0, peak_nav=1_050_000.0)
    state = db.fetch_latest_portfolio_state()
    assert state is not None
    assert state["current_nav"] == 1_000_000.0
    assert state["peak_nav"] == 1_050_000.0

    # Upsert updates existing singleton row
    db.upsert_portfolio_state(current_nav=1_100_000.0, peak_nav=1_100_000.0)
    updated = db.fetch_latest_portfolio_state()
    assert updated is not None
    assert updated["current_nav"] == 1_100_000.0


def test_platt_calibration_crud(db: SqliteDatabase) -> None:
    assert db.fetch_latest_platt() is None

    db.upsert_platt(a=-1.85, b=0.42)
    platt = db.fetch_latest_platt()
    assert platt is not None
    a, b, fitted_at, version = platt
    assert pytest.approx(a, 0.001) == -1.85
    assert pytest.approx(b, 0.001) == 0.42
    assert len(fitted_at) > 0
    assert version == 2


def test_trade_log_insert_and_fetch(db: SqliteDatabase) -> None:
    assert db.count_trades() == 0

    trade1 = {
        "ticker": "RELIANCE.NS",
        "direction": "LONG",
        "pnl": 1250.0,
        "composite": 0.75,
        "factors": {"trend": 1.2, "vol": 0.8},
        "ts": "2026-09-01T10:00:00Z",
    }
    trade_id = db.insert_trade(trade1)
    assert trade_id > 0
    assert db.count_trades() == 1

    trade2 = {
        "ticker": "TCS.NS",
        "direction": "LONG",
        "pnl": -500.0,
        "composite": 0.45,
        "factors": {"trend": 0.6},
        "ts": "2026-09-02T10:00:00Z",
    }
    db.insert_trades([trade2])
    assert db.count_trades() == 2

    all_trades = db.fetch_trades()
    assert len(all_trades) == 2
    assert all_trades[0]["ticker"] == "RELIANCE.NS"
    assert all_trades[1]["ticker"] == "TCS.NS"

    reliance_only = db.fetch_trades(ticker="RELIANCE.NS")
    assert len(reliance_only) == 1
    assert reliance_only[0]["ticker"] == "RELIANCE.NS"


def test_factor_weights_crud(db: SqliteDatabase) -> None:
    assert db.fetch_factor_weights() is None

    weights = {"f_trend": 0.35, "f_vol": 0.25, "f_mom": 0.40}
    db.upsert_factor_weights(weights)
    fetched = db.fetch_factor_weights()
    assert fetched is not None
    assert fetched["f_trend"] == 0.35
    assert fetched["f_mom"] == 0.40


def test_persistence_service_sqlite_migration(tmp_path: Path) -> None:
    """Test that existing JSON files are migrated into SQLite transparently."""
    paths = RuntimePaths.discover(root=tmp_path)
    paths.state_dir.mkdir(parents=True, exist_ok=True)

    # Pre-populate JSON files
    paths.portfolio_state_file.write_text(
        json.dumps({"current_nav": 950000.0, "peak_nav": 1000000.0}), encoding="utf-8"
    )
    paths.platt_calibration_file.write_text(
        json.dumps({"A": -2.1, "B": 0.35, "fitted_at": "2026-09-01", "version": 2}), encoding="utf-8"
    )
    paths.trade_log_file.write_text(
        json.dumps([{"ticker": "INFY.NS", "pnl": 500.0}]), encoding="utf-8"
    )

    service = PersistenceService(paths)

    # 1. Load portfolio state (reads JSON & migrates to SQLite)
    state = service.load_portfolio_state()
    assert state is not None
    assert state.current_nav == 950000.0

    # Verify state was saved into SQLite
    db_state = service.db.fetch_latest_portfolio_state()
    assert db_state is not None
    assert db_state["current_nav"] == 950000.0

    # 2. Load platt (reads JSON & migrates to SQLite)
    a, b, from_file = service.load_platt(CONFIG)
    assert from_file is True
    assert pytest.approx(a, 0.01) == -2.1

    db_platt = service.db.fetch_latest_platt()
    assert db_platt is not None
    assert pytest.approx(db_platt[0], 0.01) == -2.1

    # 3. Load trade log (reads JSON & migrates to SQLite)
    trades = service.load_trade_log()
    assert len(trades) == 1
    assert trades[0]["ticker"] == "INFY.NS"
    assert service.db.count_trades() == 1


def test_trade_log_duplicate_prevention(db: SqliteDatabase) -> None:
    trade = {
        "trade_id": "T1001",
        "ticker": "RELIANCE.NS",
        "direction": "LONG",
        "pnl": 1500.0,
        "timestamp": "2026-09-05T10:00:00Z",
    }
    db.insert_trade(trade)
    assert db.count_trades() == 1

    # Insert identical trade with same trade_id
    db.insert_trade(trade)
    assert db.count_trades() == 1

    # Batch insert with duplicates
    db.insert_trades([trade, trade, {**trade, "pnl": 2000.0}])
    assert db.count_trades() == 1
    fetched = db.fetch_trades()
    assert len(fetched) == 1
    assert fetched[0]["pnl"] == 2000.0

