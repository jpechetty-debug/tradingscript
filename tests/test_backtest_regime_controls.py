import pandas as pd
from core.regime import MarketRegimeType, RegimeTracker, classify_regime
from core.config import SystemConfig

def create_mock_processed(n_tickers=5, adx=30.0, atr_ratio=1.0, breadth_hit=True):
    processed = {}
    for i in range(n_tickers):
        ticker = f"T{i}"
        df = pd.DataFrame({
            "Close": [100.0, 101.0 if breadth_hit else 99.0],
            "EMA_50": [100.0, 100.0],
            "ADX": [adx, adx],
            "ATR": [1.0, 1.0],
            "ATR_50_mean": [1.0 / atr_ratio, 1.0 / atr_ratio]
        })
        processed[ticker] = df
    return processed

def test_regime_tracker_initial_state():
    tracker = RegimeTracker()
    assert tracker.last_regime() is None
    assert tracker.last_breadth() == 0.0

def test_regime_tracker_push_and_history():
    tracker = RegimeTracker(max_history=3)
    tracker.push(MarketRegimeType.RANGE, 0.5)
    assert tracker.last_regime() == MarketRegimeType.RANGE
    assert tracker.last_breadth() == 0.5

    tracker.push(MarketRegimeType.TREND_UP, 0.7)
    tracker.push(MarketRegimeType.TREND_DOWN, 0.3)
    tracker.push(MarketRegimeType.PANIC, 0.1) # Evicts RANGE

    assert tracker.last_regime() == MarketRegimeType.PANIC
    assert len(tracker._history) == 3

def test_regime_tracker_confirmation_logic():
    tracker = RegimeTracker()
    # 2 bars of UP
    tracker.push(MarketRegimeType.TREND_UP, 0.6)
    tracker.push(MarketRegimeType.TREND_UP, 0.6)

    # Needs 3 for confirmation (default in SystemConfig is 3 usually)
    assert tracker.is_confirmed(MarketRegimeType.TREND_UP, 3) is False

    tracker.push(MarketRegimeType.TREND_UP, 0.6)
    assert tracker.is_confirmed(MarketRegimeType.TREND_UP, 3) is True

    # PANIC is always confirmed
    assert tracker.is_confirmed(MarketRegimeType.PANIC, 3) is True

def test_classify_regime_panic_threshold():
    config = SystemConfig(REGIME_BREADTH_PANIC=0.25)
    tracker = RegimeTracker()
    processed = create_mock_processed()

    # Breadth 0.1 < 0.25 -> PANIC
    regime = classify_regime(processed, 0.1, tracker, config)
    assert regime.regime == MarketRegimeType.PANIC
    assert tracker.last_regime() == MarketRegimeType.PANIC

def test_classify_regime_panic_hysteresis():
    config = SystemConfig(REGIME_BREADTH_PANIC=0.25, REGIME_BREADTH_PANIC_EXIT=0.35)
    tracker = RegimeTracker()
    tracker.push(MarketRegimeType.PANIC, 0.1)

    processed = create_mock_processed()
    # Breadth 0.3 is above ENTER (0.25) but below EXIT (0.35) -> stay in PANIC
    regime = classify_regime(processed, 0.3, tracker, config)
    assert regime.regime == MarketRegimeType.PANIC

    # Breadth 0.4 -> exit PANIC
    regime = classify_regime(processed, 0.4, tracker, config)
    assert regime.regime != MarketRegimeType.PANIC

def test_classify_regime_trend_deadband():
    config = SystemConfig() # Default deadband is around 0.5
    tracker = RegimeTracker()
    processed = create_mock_processed(adx=35.0) # High ADX

    # Breadth 0.52 is in the deadband [0.45, 0.55] -> RANGE
    regime = classify_regime(processed, 0.52, tracker, config)
    assert regime.regime == MarketRegimeType.RANGE

    # Breadth 0.6 -> TREND_UP
    regime = classify_regime(processed, 0.6, tracker, config)
    assert regime.regime == MarketRegimeType.TREND_UP

def test_regime_lock_suppresses_updates():
    config = SystemConfig()
    tracker = RegimeTracker()
    tracker.push(MarketRegimeType.TREND_UP, 0.7)

    processed = create_mock_processed()
    # Locked scan with poor breadth: should NOT push to tracker, should return last_regime
    regime = classify_regime(processed, 0.1, tracker, config, locked=True)

    assert regime.regime == MarketRegimeType.TREND_UP
    assert regime.regime_locked is True
    assert tracker.last_regime() == MarketRegimeType.TREND_UP
    assert tracker.last_breadth() == 0.7 # No change

def test_sector_concentration_bonus():
    config = SystemConfig()
    tracker = RegimeTracker()
    processed = create_mock_processed(adx=40.0)

    # No sector RS
    regime_no_rs = classify_regime(processed, 0.8, tracker, config)

    # High sector RS alignment
    sector_rs = {"IT": 0.5, "FIN": 0.4, "AUTO": 0.6} # All positive
    regime_with_rs = classify_regime(processed, 0.8, tracker, config, sector_rs=sector_rs)

    assert regime_with_rs.confidence > regime_no_rs.confidence
    assert regime_with_rs.sector_concentration == 1.0
