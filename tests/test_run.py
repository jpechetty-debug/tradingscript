from __future__ import annotations

from unittest.mock import MagicMock

import run
from core import runtime_components, services

def test_watch_mode_pnl_wiring(monkeypatch, tmp_path):
    """
    Test that _step() feeds new closed trades from the persistence trade log
    into the calibrator buffer over multiple iterations in watch mode.
    """
    # 1. Mock persistence and calibrator
    persistence = MagicMock(spec=services.PersistenceService)
    persistence.load_trade_log.return_value = []

    calibrator = MagicMock()

    # We patch configure_services to return our mocked persistence
    original_configure = run.configure_services
    def fake_configure_services(**kwargs):
        kwargs["persistence"] = persistence
        kwargs["factor_calibrator"] = calibrator
        return original_configure(**kwargs)

    monkeypatch.setattr(run, "configure_services", fake_configure_services)

    # We mock RollingFactorCalibrator to return our mock so the real one doesn't start a thread
    monkeypatch.setattr(runtime_components, "RollingFactorCalibrator", lambda **kwargs: calibrator)

    # We also mock run_scan to avoid actual network/scanning
    monkeypatch.setattr(run, "run_scan", lambda **kwargs: ([], [], None))

    # 3. We use time.sleep to advance the loop and eventually break out of it
    sleep_calls = 0
    def fake_sleep(secs):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            # Between step 1 and step 2, write a synthetic trade
            synthetic_trade = {
                "factors": {"trend": 0.8, "momentum": 0.5},
                "pnl": 0.15
            }
            persistence.load_trade_log.return_value = [synthetic_trade]
        elif sleep_calls == 2:
            # After step 2, break the loop
            # Raising KeyboardInterrupt inside the main loop instead of thread
            raise KeyboardInterrupt()

    monkeypatch.setattr(run.time, "sleep", fake_sleep)
    monkeypatch.setattr(run.sys, "exit", lambda x: None)

    # 4. Run main in watch mode
    run.main(["--watch", "1"])

    # 5. Assertions
    # Calibrator should have been called exactly once with the synthetic trade
    assert calibrator.record_trade.call_count == 1
    calibrator.record_trade.assert_called_with({"trend": 0.8, "momentum": 0.5}, 0.15)
