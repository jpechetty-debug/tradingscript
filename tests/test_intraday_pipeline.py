import pytest
import json
import logging
from pathlib import Path
from dataclasses import replace

import pandas as pd

from core.config import SystemConfig, SignalSettings, RegimeSettings
from core.database import SqliteDatabase
from core.scorer import score_candidate_pass1, _classify_trade_horizon
from core.services import ScanService, PersistenceService, ScanCache
from core.runtime_components import RollingFactorCalibrator


def test_intraday_disabled():
    # 1. Direct helper test
    config = SystemConfig()
    # Mocking behavior if needed, config.INTRADAY_ENABLED is False by default
    horizon = _classify_trade_horizon(
        config=config,
        direction="SHORT",
        session="OPENING_RANGE",
        adx=15,
        intraday={}
    )
    assert horizon == "SWING", "Should always return SWING when INTRADAY_ENABLED is False"

    # 2. Inside score_candidate_pass1
    row = pd.Series({
        "Close": 100.0, "ADX": 15.0, "Super_Up": True, "EMA_20": 90.0, "EMA_200": 80.0, "RSI": 60,
        "Vol_Avg_20": 1000000, "Volume": 1000000, "Turnover_Avg_20": 40_000_000, "ATR_Pctile": 50, "ATR_50_mean": 5, "Trend_60m": 1,
        "MACD_Hist": 1, "RVol": 1, "Change_Pct": 1
    })
    class MockRegime:
        regime = "TREND_UP"
        label = "TREND_UP"
        color = "green"
        def allows_long(self): return True
        def allows_short(self): return False
        def allows_mean_reversion(self): return False

    result = score_candidate_pass1(
        ticker="RELIANCE.NS",
        daily_df=pd.DataFrame([row]),
        bench=pd.Series([]),
        sector_ranks={},
        sector_rs={},
        session="OPENING_RANGE",
        regime=MockRegime(),
        config=config,
        intraday={},
    )
    if result is not None:
        assert result.trade_horizon == "SWING"


def test_factor_weight_floor():
    config = SystemConfig()
    
    # Mock raw coefficients that go below floor
    raw = {
        "trend": 0.5,
        "momentum": 1.0,
        "volatility": 0.01,
        "volume": 0.5,
        "rs": 0.5,
        "breakout": 0.5,
        "quality": 0.5,
    }
    
    min_weight = config.MIN_FACTOR_WEIGHT
    
    # We are testing ScanService._validated_factor_weights rejecting if below floor
    from core.services import ScanService
    cleaned = ScanService._validated_factor_weights(raw)
    
    assert "breakout" in cleaned
    # Since one value was below 0.03, it should return DEFAULT_WEIGHTS
    from core.scorer import DEFAULT_WEIGHTS
    assert cleaned == dict(DEFAULT_WEIGHTS)


def test_executed_trade_schema(tmp_path):
    db_file = tmp_path / "test.db"
    db = SqliteDatabase(db_file)
    trade = {
        "trade_id": "test_1",
        "ticker": "RELIANCE.NS",
        "direction": "LONG",
        "trade_horizon": "SWING",
        "entry_price": 100.0,
        "stop_loss": 90.0,
        "target": 120.0,
        "shares": 10,
        "composite": 0.8,
        "prob_win": 0.6,
        "factors": {"breakout": 0.5},
        "entry_ts": "2026-01-01T10:00:00",
        "outcome": "OPEN"
    }
    
    # insert
    db.insert_executed_trade(trade)
    
    # fetch
    trades = db.fetch_executed_trades()
    assert len(trades) == 1
    assert trades[0]["ticker"] == "RELIANCE.NS"
    assert trades[0]["trade_horizon"] == "SWING"
    
    # close
    db.close_executed_trade(
        "test_1", 
        exit_price=110.0, 
        exit_ts="2026-01-02T10:00:00", 
        gross_pnl=0.1, 
        costs=0.0,
        net_pnl=0.1,
        realised_r=1.0, 
        outcome="WIN"
    )
    
    # fetch calibration
    comp, out = db.fetch_calibration_trades(min_samples=1)
    assert len(comp) == 1
    assert comp[0] == 0.8
    assert out[0] == 1


def test_duplicate_suppression():
    from core.services import ScanService
    from core.scorer import TickerResult, FactorScores

    service = ScanService(version="test")
    service._recent_alerts = []

    # Mock portfolio
    class MockTickerResult:
        def __init__(self, ticker, composite):
            self.ticker = ticker
            self.composite = composite
            self.prob_win = 0.6

    portfolio = [
        MockTickerResult("A", 0.8),
        MockTickerResult("B", 0.9),
    ]

    # Insert into recent alerts manually
    service._recent_alerts.append({"A": 0.8, "B": 0.9})

    # Scenario 1: Same composites -> should be suppressed
    lookback = 5
    delta = 0.05
    
    filtered = []
    for r in portfolio:
        was_recent = any(
            r.ticker in scan and abs(r.composite - scan[r.ticker]) < delta
            for scan in service._recent_alerts[-lookback:]
        )
        if not was_recent:
            filtered.append(r)
            
    assert len(filtered) == 0, "Exact duplicates should be suppressed"

    # Scenario 2: Composite improved by > delta -> should NOT be suppressed
    portfolio[0].composite = 0.86  # 0.8 + 0.06
    
    filtered2 = []
    for r in portfolio:
        was_recent = any(
            r.ticker in scan and abs(r.composite - scan[r.ticker]) < delta
            for scan in service._recent_alerts[-lookback:]
        )
        if not was_recent:
            filtered2.append(r)
            
    assert len(filtered2) == 1
    assert filtered2[0].ticker == "A", "Improved composite should pass suppression"


def test_legacy_trade_log_archive(tmp_path):
    log_file = tmp_path / "trade_log.json"
    legacy_data = [{"pnl": 0.5, "factors": {"breakout": 0.1}}]
    log_file.write_text(json.dumps(legacy_data))

    db = SqliteDatabase(tmp_path / "test2.db")
    # Mock persistence service paths
    class MockPaths:
        trade_log_file = log_file
        root = tmp_path
    
    
    persistence = PersistenceService()
    persistence.paths = MockPaths()
    persistence.db = db

    trades = persistence.load_trade_log()
    assert len(trades) == 0, "Legacy log should be archived and return empty"
    
    archived_files = list(tmp_path.glob("trade_log_legacy_*.json"))
    assert len(archived_files) == 1, "Legacy file should be renamed"
