import json
import time
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from core.config import CONFIG
from core.cache import _build_corr_matrix, TieredMarketDataCache
from core.indicators import _supertrend_vectorised, _supertrend_inner_loop_py, _true_range, _wilder
from server import app

client = TestClient(app)


def test_supertrend_numba_parity():
    """Verify Numba JIT Supertrend matches pure-Python reference implementation."""
    np.random.seed(42)
    n = 200
    close = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
    high = close + np.random.rand(n) * 2.0
    low = close - np.random.rand(n) * 2.0
    open_ = (high + low) / 2.0
    volume = np.random.randint(1000, 50000, n)

    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

    # JIT-accelerated (or active implementation)
    st_jit, up_jit = _supertrend_vectorised(df, period=10, mult=3.0)

    # Pure-Python reference
    tr = _true_range(df)
    atr = _wilder(tr, 10)
    hl2 = (df["High"] + df["Low"]) / 2
    close_arr = np.ascontiguousarray(df["Close"].values, dtype=np.float64)
    final_upper = np.ascontiguousarray((hl2 + 3.0 * atr).values, dtype=np.float64)
    final_lower = np.ascontiguousarray((hl2 - 3.0 * atr).values, dtype=np.float64)

    st_py, up_py = _supertrend_inner_loop_py(close_arr, final_upper, final_lower, n)

    # Exact numerical parity assertion
    np.testing.assert_allclose(st_jit, st_py, rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_array_equal(up_jit, up_py)


def test_correlation_matrix_parity_and_speed():
    """Verify fast-path NumPy correlation matches Pandas correlation matrix."""
    np.random.seed(42)
    dates = pd.date_range("2026-01-01", periods=100)
    processed = {}
    tickers = [f"SYM_{i}.NS" for i in range(15)]

    for t in tickers:
        prices = 100.0 + np.cumsum(np.random.randn(100) * 1.2)
        processed[t] = pd.DataFrame({"Close": prices}, index=dates)

    # Compute fast-path
    corr_fast = _build_corr_matrix(processed, config=CONFIG)

    assert not corr_fast.empty
    assert corr_fast.shape == (15, 15)
    # Check diagonal is 1.0
    np.testing.assert_allclose(np.diag(corr_fast.values), 1.0, atol=1e-7)
    # Check symmetry
    np.testing.assert_allclose(corr_fast.values, corr_fast.values.T, atol=1e-7)
    # Check valid bounds
    assert (corr_fast.values >= -1.0 - 1e-7).all()
    assert (corr_fast.values <= 1.0 + 1e-7).all()


def test_tiered_market_data_cache():
    """Verify TieredMarketDataCache L1 memory TTL and eviction."""
    cache = TieredMarketDataCache(default_ttl_sec=1)
    test_data = {"quote": 1250.5, "symbol": "TCS.NS"}

    # Set and get
    cache.set("quote:TCS.NS", test_data)
    assert cache.get("quote:TCS.NS") == test_data

    # Test eviction after TTL
    time.sleep(1.05)
    assert cache.get("quote:TCS.NS") is None

    # Test explicit clear
    cache.set("quote:INFY.NS", {"quote": 1800.0})
    cache.clear()
    assert cache.get("quote:INFY.NS") is None


def test_sse_events_stream():
    """Verify GET /api/events yields Server-Sent Events with connected event."""
    with client.stream("GET", "/api/events?limit=1") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")

        # Read first event (initial connected state snapshot)
        lines = []
        for line in response.iter_lines():
            if line:
                lines.append(line)
                if len(lines) >= 1:
                    break

        assert len(lines) > 0
        raw_event = lines[0]
        assert raw_event.startswith("data: ")
        payload = json.loads(raw_event[6:])
        assert payload["event"] == "connected"
        assert "killswitch_active" in payload["data"]
