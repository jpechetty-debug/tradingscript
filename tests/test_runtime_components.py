import os
import json
from pathlib import Path


from core.runtime_components import (
    TieredCapitalScaler,
    RegimeProbabilityGate,
    RollingFactorCalibrator,
    RegimeAwareTelegramAlerter,
    create_runtime_components,
)

class DummyRegime:
    def __init__(self, value: str):
        self.value = value


def test_tiered_capital_scaler():
    scaler = TieredCapitalScaler(portfolio_peak=100000.0)

    # Base cases
    assert scaler.capital_fraction(100000.0, "TREND_UP") == 1.0
    assert scaler.capital_fraction(100000.0, "EXPANSION") == 1.0
    assert scaler.capital_fraction(100000.0, "RANGE") == 0.75

    # Peak tracking
    assert scaler.capital_fraction(110000.0, "TREND_UP") == 1.0
    assert scaler.peak == 110000.0

    # Drawdown logic - slight drawdown (e.g., 3%) reduces base
    # 110000 -> 106700 is 3% DD. Base = 1.0 * max(0.5, 1 - 0.03*3) = 0.91
    assert scaler.capital_fraction(106700.0, "TREND_UP") == 0.91

    # Heavy drawdown in trend up
    assert scaler.capital_fraction(90000.0, "TREND_UP") == 0.50  # Cap at 0.5

    # Panic tiers
    # 110000 -> 105000 is ~4.5% DD. PANIC cap should be 0.50 (for DD 2-5%)
    assert scaler.capital_fraction(105000.0, "PANIC") == 0.50
    assert scaler.capital_fraction(105000.0, "TREND_DOWN") == 0.50

    # DD > 10%
    assert scaler.capital_fraction(95000.0, "PANIC") == 0.00

    # Enum regime support
    assert scaler.capital_fraction(95000.0, DummyRegime("PANIC")) == 0.00

def test_tiered_capital_scaler_zero_peak():
    scaler = TieredCapitalScaler(portfolio_peak=0.0)
    assert scaler.current_drawdown(1000.0) == 0.0


def test_regime_probability_gate():
    gate = RegimeProbabilityGate(overrides={"TEST_REGIME": 0.6})
    assert gate.threshold("TREND_UP") == 0.50
    assert gate.threshold("TEST_REGIME") == 0.60
    assert gate.threshold("UNKNOWN") == 0.52  # default fallback

    assert gate.passes(0.55, "TREND_UP") is True
    assert gate.passes(0.45, "TREND_UP") is False
    assert gate.margin(0.55, "TREND_UP") == 0.05
    assert gate.passes(0.99, DummyRegime("PANIC")) is True


def test_rolling_factor_calibrator_basics(tmp_path):
    wpath = tmp_path / "weights.json"
    tpath = tmp_path / "trade_log.json"

    calibrator = RollingFactorCalibrator(
        window=50,
        interval_sec=1,
        weights_path=wpath,
        trade_log_path=tpath
    )

    assert calibrator.current_weights()["trend"] == 0.28

    calibrator.record_trade({"trend": 0.5, "momentum": 0.5}, 0.05)
    assert tpath.exists()

    # Thread controls
    calibrator.start()
    assert calibrator._running is True
    calibrator.start() # idempotent
    calibrator.stop()
    assert calibrator._running is False


def test_rolling_factor_calibrator_load_weights_valid(tmp_path):
    wpath = tmp_path / "weights.json"
    wpath.write_text(json.dumps({"trend": 0.99}))

    calibrator = RollingFactorCalibrator(weights_path=wpath, trade_log_path=tmp_path / "log.json")
    assert calibrator.current_weights()["trend"] == 0.99

def test_rolling_factor_calibrator_load_weights_invalid(tmp_path):
    wpath = tmp_path / "weights.json"
    wpath.write_text("INVALID JSON")

    calibrator = RollingFactorCalibrator(weights_path=wpath, trade_log_path=tmp_path / "log.json")
    assert calibrator.current_weights()["trend"] == 0.28


def test_rolling_factor_calibrator_loop_and_calibrate(tmp_path, monkeypatch):
    wpath = tmp_path / "weights.json"
    tpath = tmp_path / "trade_log.json"

    calibrator = RollingFactorCalibrator(
        window=10,
        interval_sec=1,
        weights_path=wpath,
        trade_log_path=tpath
    )

    # Feed enough trades
    for i in range(5):
        calibrator.record_trade({"f1": 1.0, "f2": 0.0}, 0.10)
        calibrator.record_trade({"f1": 0.0, "f2": 1.0}, -0.10)

    weights = calibrator._calibrate()
    assert "f1" in weights
    assert "f2" in weights
    assert weights["f1"] > 0
    # f2 produced negative returns, weight should be clipped to 0
    assert weights["f2"] == 0.0

def test_rolling_factor_calibrator_calibrate_exception(tmp_path, monkeypatch):
    calibrator = RollingFactorCalibrator(weights_path=tmp_path/"w.json", trade_log_path=tmp_path/"t.json")
    for i in range(5):
        calibrator.record_trade({"f1": 1.0}, 0.10)

    def fail_fit(*args, **kwargs):
        raise ValueError("Simulated failure")

    from sklearn.linear_model import Ridge
    monkeypatch.setattr(Ridge, "fit", fail_fit)

    weights = calibrator._calibrate()
    assert weights == calibrator._weights # Returns old weights

def test_rolling_factor_calibrator_too_few_records(tmp_path):
    calibrator = RollingFactorCalibrator(weights_path=tmp_path/"w.json", trade_log_path=tmp_path/"t.json")
    # Feed 4 trades (< 5)
    for i in range(4):
        calibrator.record_trade({"f1": 1.0}, 0.10)

    # Force call _calibrate directly
    weights = calibrator._calibrate()
    assert weights == calibrator._weights


def test_rolling_factor_calibrator_persist_trade_existing_json(tmp_path):
    tpath = tmp_path / "trade_log.json"
    tpath.write_text(json.dumps([{"ts": "old"}]))
    calibrator = RollingFactorCalibrator(weights_path=tmp_path/"w.json", trade_log_path=tpath)
    calibrator.record_trade({"f": 1}, 0)

    loaded = json.loads(tpath.read_text())
    assert len(loaded) == 2

def test_rolling_factor_calibrator_save_and_persist_oserror(tmp_path, monkeypatch):
    wpath = tmp_path / "w.json"
    tpath = tmp_path / "t.json"

    # Make directory read-only to trigger OSError
    def raise_oserror(*args, **kwargs):
        raise OSError("Simulated OSError")

    # Patch write_text in pathlib.Path
    monkeypatch.setattr(Path, "write_text", raise_oserror)

    calibrator = RollingFactorCalibrator(weights_path=wpath, trade_log_path=tpath)
    calibrator.record_trade({"f": 1}, 0)
    # Should not raise

    calibrator._save_weights({"f": 1})
    # Should not raise


def test_regime_aware_telegram_alerter(monkeypatch):
    gate = RegimeProbabilityGate()
    scaler = TieredCapitalScaler(portfolio_peak=100000.0)
    alerter = RegimeAwareTelegramAlerter(gate=gate, scaler=scaler)

    called_with = ""
    def mock_send(text, **kwargs):
        nonlocal called_with
        called_with = text
        return True

    monkeypatch.setattr("core.runtime_components._send_telegram", mock_send)

    # With picks
    picks = [{"ticker": "ABC", "prob_win": 0.60}]
    alerter.send_daily_summary("TREND_UP", picks, current_nav=100000.0)

    assert "Regime: TREND_UP" in called_with
    assert "Capital: 100% capital" in called_with
    assert "Drawdown: 0.0%" in called_with
    assert "Active capital: 100%" in called_with
    assert "OK ABC" in called_with

    # Without picks
    alerter.send_daily_summary("PANIC", [], current_nav=90000.0)
    assert "No qualifying setups" in called_with
    assert "Regime: PANIC" in called_with

    # Missing credentials in _send_telegram
    monkeypatch.undo()
    monkeypatch.setattr(os, "getenv", lambda k, d=None: "")
    import core.runtime_components
    assert core.runtime_components._send_telegram("test") is False

    # With credentials in _send_telegram
    monkeypatch.setattr(os, "getenv", lambda k, d=None: "dummy")
    # Patch the inner send_telegram
    import utils.messaging
    monkeypatch.setattr(utils.messaging, "send_telegram", lambda text, token, chat_id: True)
    assert core.runtime_components._send_telegram("test") is True


def test_create_runtime_components(monkeypatch):
    # Prevent starting thread
    monkeypatch.setattr(RollingFactorCalibrator, "start", lambda self: None)

    with create_runtime_components(announce=True) as comps:
        assert isinstance(comps.gate, RegimeProbabilityGate)
        assert isinstance(comps.scaler, TieredCapitalScaler)
        assert isinstance(comps.calibrator, RollingFactorCalibrator)
        assert isinstance(comps.alerter, RegimeAwareTelegramAlerter)
