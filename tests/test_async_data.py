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
