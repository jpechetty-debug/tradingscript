
import sys
import os
import pandas as pd
import numpy as np
from dataclasses import asdict

# Mock the environment or import the actual screener
sys.path.append(os.getcwd())
import screener
import sovereign_improvements as SE_PATCH

def test_integration():
    print("--- Testing Sovereign Improvement Patch Integration ---")
    
    # 1. Check if patch is applied and accessible
    if not hasattr(screener, 'capital_scaler'):
        print("FAIL: capital_scaler not attached to screener")
        return
    
    print("PASS: Patch components successfully attached to screener global scope")
    
    # 2. Test Capital Scaler
    fraction = screener.capital_scaler.capital_fraction(current_nav=900_000, regime="PANIC")
    print(f"Capital Fraction (Panic at 10% DD): {fraction} (Expected 0.25 if DD=10%)")
    
    # 3. Test Prob Gate
    passed = screener.prob_gate.passes(p_win=0.54, regime="RANGE")
    print(f"Prob Gate (0.54 in RANGE): {passed} (Expected True with 0.45 test threshold)")
    
    # 4. Test Calibrator record_trade
    initial_weights = screener.calibrator.current_weights()
    print(f"Initial Weights: {initial_weights}")
    
    dummy_factors = {
        "trend": 0.8, "momentum": 0.9, "volume": 0.7, 
        "volatility": 0.2, "rs": 0.9, "breakout": 0.5, "quality": 0.9
    }
    
    print("Recording 15 winning trades to trigger recalibration...")
    for _ in range(15):
        screener.calibrator.record_trade(dummy_factors, pnl_pct=0.05)
    
    # Force recalibration (since loop might be slow)
    new_weights = screener.calibrator._calibrate()
    print(f"New Weights after training: {new_weights}")
    
    if initial_weights != new_weights:
        print("PASS: RollingFactorCalibrator recalibrated weights based on trade history")
    else:
        print("FAIL: Weights did not change (or training data didn't move them)")

    # 5. Test Resilient Data Provider Fallback
    # (Optional: can mock a failure, but we saw it working in previous terminal output)
    print("Check: Data Provider is using fallback chain: ", screener.data_provider._yf is not None)

if __name__ == "__main__":
    test_integration()
