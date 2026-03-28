import os
import sys

sys.path.append(os.getcwd())

import screener_v14_modular as screener
import sovereign_improvements as se_patch


def test_integration():
    print("--- Testing Sovereign compatibility layer ---")

    attached = se_patch.apply(screener, announce=False)

    if not hasattr(screener, "capital_scaler"):
        print("FAIL: capital_scaler not attached to screener module")
        return

    print("PASS: Compatibility components attached to screener module")

    fraction = screener.capital_scaler.capital_fraction(current_nav=900_000, regime="PANIC")
    print(f"Capital Fraction (PANIC at 10% DD): {fraction}")

    passed = screener.prob_gate.passes(p_win=0.54, regime="RANGE")
    print(f"Probability Gate (0.54 in RANGE): {passed}")

    initial_weights = screener.calibrator.current_weights()
    print(f"Initial Weights: {initial_weights}")

    dummy_factors = {
        "trend": 0.8,
        "momentum": 0.9,
        "volume": 0.7,
        "volatility": 0.2,
        "rs": 0.9,
        "breakout": 0.5,
        "quality": 0.9,
    }

    print("Recording 15 winning trades to trigger recalibration...")
    for _ in range(15):
        screener.calibrator.record_trade(dummy_factors, pnl_pct=0.05)

    new_weights = screener.calibrator._calibrate()
    print(f"New Weights after training: {new_weights}")
    print("Data provider exposes yfinance handle:", attached["data_provider"]._yf is not None)


if __name__ == "__main__":
    test_integration()
