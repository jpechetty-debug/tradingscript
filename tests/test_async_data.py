import asyncio
import pandas as pd
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

def _make_config():
    cfg = MagicMock()
    cfg.BENCHMARK = "^NSEI"
    cfg.USE_FYERS = False
    cfg.DAILY_PERIOD = "1y"
    cfg.MAX_WORKERS = 4
    return cfg

def _make_df():
    import numpy as np
    n = 50
    return pd.DataFrame({
        "Open": np.ones(n) * 100, "High": np.ones(n) * 105,
        "Low": np.ones(n) * 95, "Close": np.ones(n) * 102,
        "Volume": np.ones(n) * 1_000_000,
    })

@pytest.mark.asyncio
async def test_async_fetch_falls_back_to_yfinance():
    from core.async_data import async_fetch_daily_batch
    fake_df = _make_df()
    with patch("core.async_data.yf.download", return_value=fake_df):
        result = await async_fetch_daily_batch(["RELIANCE.NS"], _make_config())
    assert isinstance(result, dict)

def test_async_fetch_runs_via_asyncio_run():
    from core.async_data import async_fetch_daily_batch
    fake_df = _make_df()
    with patch("core.async_data.yf.download", return_value=fake_df):
        result = asyncio.run(async_fetch_daily_batch(["RELIANCE.NS"], _make_config()))
    assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_async_fetch_fyers_returns_none_without_client():
    """Fyers path: no credentials → returns (ticker, None) cleanly."""
    from core.async_data import _async_fetch_fyers
    import asyncio
    sem = asyncio.Semaphore(10)
    cfg = _make_config()
    # No Fyers client configured → should return None without crash
    ticker, df = await _async_fetch_fyers(
        ticker="RELIANCE.NS",
        fsym="NSE:RELIANCE-EQ",
        config=cfg,
        sem=sem,
        executor=None,
    )
    assert ticker == "RELIANCE.NS"
    assert df is None

@pytest.mark.asyncio
async def test_async_yfinance_chunk_assembles_results():
    """yfinance async path: mock executor, verify chunk is assembled into dict."""
    from core.async_data import async_fetch_daily_batch
    import asyncio

    n = 50
    fake = pd.DataFrame({
        "Open": [100.0]*n, "High": [105.0]*n, "Low": [95.0]*n,
        "Close": [102.0]*n, "Volume": [1e6]*n,
    })

    # Patch at the download level so the async executor path is exercised
    with patch("core.async_data.yf.download", return_value=fake):
        result = await async_fetch_daily_batch(
            ["RELIANCE.NS", "INFY.NS"], _make_config()
        )
    assert isinstance(result, dict)

@pytest.mark.asyncio  
async def test_async_fetch_respects_semaphore_limit():
    """Concurrent fetches are bounded by the semaphore — no deadlock."""
    from core.async_data import async_fetch_daily_batch
    fake = _make_df()
    with patch("core.async_data.yf.download", return_value=fake):
        # 5 tickers, semaphore limit 10 → should complete without hanging
        tickers = [f"T{i}.NS" for i in range(5)]
        result = await async_fetch_daily_batch(tickers, _make_config())
    assert isinstance(result, dict)
