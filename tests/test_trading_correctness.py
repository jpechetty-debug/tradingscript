from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

import core.services as services
from core.backtest import ZERO_COST_MODEL, walk_forward
from core.config import CONFIG
from core.runtime_paths import RuntimePaths
from core.services import PersistenceService, ScanService
from tests.test_hysteresis import _make_dummy_candidate
from core.scorer import score_candidate_pass2


@pytest.mark.parametrize("ticker", ["RELIANCE", "RELIANCE.NS"])
def test_backtest_accepts_real_scorer_symbol_and_provider_symbol(monkeypatch, ticker):
    cfg = replace(CONFIG, PLATT_A=-4.0, PLATT_B=2.0)
    dates = pd.date_range("2023-01-01", periods=160, freq="B")
    close = pd.Series([100 + i * .5 for i in range(160)], index=dates)
    frame = pd.DataFrame({"Open": close, "High": close + 2,
                          "Low": close - 1, "Close": close + .5, "Volume": 500000})
    candidate = SimpleNamespace(
        ticker=ticker, sector="ENERGY", direction="LONG", composite=.75,
        prob_win=.65, is_watchlist=False, trade_horizon="SWING", sharpe_rank=1,
        reasons=[], entry=150, stop=145, shares=100, risk_inr=1500,
        time_stop_bars=4, atr_pctile=50,
    )
    monkeypatch.setattr(services, "score_universe", lambda **kw: [candidate])
    monkeypatch.setattr(PersistenceService, "load_platt",
                        lambda *a: pytest.fail("Historical run read live calibration"))
    result = walk_forward({cfg.BENCHMARK: frame, "RELIANCE.NS": frame}, cfg,
                          train_days=120, test_days=20, step_days=20,
                          cost_model=ZERO_COST_MODEL)
    assert len(result.trades) == 2


@pytest.mark.parametrize("bar_date,closed", [
    ("2024-01-02", False), ("2024-01-03", False), ("2024-01-04", True),
])
def test_stop_monitor_uses_only_post_entry_candles(tmp_path, bar_date, closed):
    persistence = PersistenceService(RuntimePaths.discover(root=tmp_path))
    persistence.save_open_positions([{
        "ticker": "TEST", "direction": "LONG", "entry": 105., "shares": 10,
        "stop": 95., "t1": 120., "opened_at": "2024-01-03T10:00:00+05:30",
    }])
    service = ScanService(version="test", persistence=persistence)
    frame = pd.DataFrame({"Open": [100.], "High": [108.], "Low": [90.],
                          "Close": [105.]}, index=pd.to_datetime([bar_date]))
    service._monitor_open_position_stops({"TEST.NS": frame},
                                         persistence.create_scan_state(CONFIG))
    assert ("TEST" not in persistence.load_open_positions()) is closed
    assert persistence.db.count_trades() == int(closed)


def test_scan_does_not_convert_signals_to_fills_or_rewrite_book(tmp_path, monkeypatch):
    persistence = PersistenceService(RuntimePaths.discover(root=tmp_path))
    persistence.save_open_positions([{
        "ticker": "HELD", "direction": "LONG", "entry": 100., "shares": 10,
        "stop": 90., "t1": 120., "opened_at": "2024-01-03T10:00:00+05:30",
    }])
    before = persistence.load_open_positions()
    frame = pd.DataFrame({"Close": [110.]}, index=pd.to_datetime(["2024-01-02"]))
    prepared = services.PreparedScanData({}, {"HELD.NS": frame}, frame.Close, {}, {})
    market_data = MagicMock()
    market_data.prepare_scan_data.return_value = prepared
    regime = MagicMock()
    regime.is_tradeable.return_value = True
    new = SimpleNamespace(ticker="NEW", direction="LONG", composite=.8,
                          prob_win=.7, expectancy_r=.5, rr_t1=2., shares=10,
                          risk_inr=100., is_watchlist=False)
    service = ScanService(version="test", persistence=persistence, data_service=market_data)
    service._recent_alerts = [{"NEW": .8}]
    monkeypatch.setattr(services, "compute_breadth", lambda *a: .7)
    monkeypatch.setattr(services, "classify_regime", lambda *a, **kw: regime)
    monkeypatch.setattr(service, "_log_regime", lambda *a: None)
    monkeypatch.setattr(service, "_score_candidates", lambda **kw: ([new], 0.))
    monkeypatch.setattr(service, "_maybe_recalibrate_ic_weights", lambda **kw: None)
    monkeypatch.setattr(services, "optimize_portfolio", lambda *a: [new])
    output = service.scan()
    assert output[1] == [new]
    assert persistence.load_open_positions() == before


def test_service_backtest_never_loads_live_calibration(tmp_path, monkeypatch):
    persistence = PersistenceService(RuntimePaths.discover(root=tmp_path))
    monkeypatch.setattr(persistence, "load_platt",
                        lambda *a: pytest.fail("Historical run read live calibration"))
    data = MagicMock()
    data.fetch_universe.return_value = {}
    result = ScanService(version="test", persistence=persistence,
                         data_service=data).run_backtest()
    assert not result.trades


def test_live_net_expectancy_decreases_with_slippage():
    candidate = _make_dummy_candidate(composite=.8)
    low = score_candidate_pass2(candidate, replace(CONFIG, SLIPPAGE_BPS=0))
    high = score_candidate_pass2(candidate, replace(CONFIG, SLIPPAGE_BPS=20))
    assert low is not None and high is not None
    assert high.expectancy_r < low.expectancy_r
