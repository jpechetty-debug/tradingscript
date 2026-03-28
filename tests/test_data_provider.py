import types
import sys
import os
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def _stub_fyers():
    stubs = {
        "fyers_apiv3": types.ModuleType("fyers_apiv3"),
        "fyers_apiv3.fyersModel": types.ModuleType("fyers_apiv3.fyersModel"),
    }
    stubs["fyers_apiv3.fyersModel"].FyersModel = MagicMock
    stubs["fyers_apiv3"].fyersModel = stubs["fyers_apiv3.fyersModel"]
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)

_stub_fyers()

from core.data_provider import fetch_daily_batch
from core.config import SystemConfig

def _cfg(**kw):
    cfg = SystemConfig()
    object.__setattr__(cfg, "USE_FYERS", False)
    object.__setattr__(cfg, "DAILY_PERIOD", "1y")
    object.__setattr__(cfg, "MAX_WORKERS", 4)
    object.__setattr__(cfg, "BENCHMARK", "^NSEI")
    for k, v in kw.items():
        object.__setattr__(cfg, k, v)
    return cfg

def _fake_multi_df(tickers):
    """MultiIndex DataFrame matching yfinance group_by='ticker' output."""
    n = 50
    arrays = []
    for t in tickers:
        for col in ["Open","High","Low","Close","Volume"]:
            arrays.append((t, col))
    idx = pd.MultiIndex.from_tuples(arrays)
    data = np.ones((n, len(arrays))) * 100.0
    df = pd.DataFrame(data, columns=idx)
    df.index = pd.date_range("2024-01-01", periods=n, freq="B")
    return df

class TestFetchDailyBatch:

    def test_yfinance_multiindex_parsed_correctly(self):
        # fetch_daily_batch adds benchmark. Deduping might result in 2 or 3 tickers.
        # We'll use 3 to force MultiIndex.
        fake = _fake_multi_df(["RELIANCE.NS", "INFY.NS", "^NSEI"])
        with patch("core.data_provider._yf_download_chunk", return_value=fake):
            # Pass INFY.NS as well to ensure it's a multi-ticker request
            result = fetch_daily_batch(["RELIANCE.NS", "INFY.NS"], _cfg())
        assert "RELIANCE.NS" in result
        assert "INFY.NS" in result
        assert isinstance(result["RELIANCE.NS"], pd.DataFrame)

    def test_yfinance_single_ticker_flat_columns(self):
        """Single-ticker download returns flat columns, not MultiIndex."""
        n = 50
        fake = pd.DataFrame({
            "open": np.ones(n)*100, "high": np.ones(n)*105,
            "low": np.ones(n)*95, "close": np.ones(n)*102,
            "volume": np.ones(n)*1e6,
        }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
        # Set benchmark to same ticker to force single-ticker results
        with patch("core.data_provider._yf_download_chunk", return_value=fake):
            result = fetch_daily_batch(["RELIANCE.NS"], _cfg(BENCHMARK="RELIANCE.NS"))
        assert isinstance(result, dict)
        assert "RELIANCE.NS" in result

    def test_yfinance_single_ticker_multiindex_parsed_correctly(self):
        """
        yfinance 1.x can return a ticker-first MultiIndex even for one symbol.
        """
        fake = _fake_multi_df(["RELIANCE.NS"])
        with patch("core.data_provider._yf_download_chunk", return_value=fake):
            result = fetch_daily_batch(["RELIANCE.NS"], _cfg(BENCHMARK="RELIANCE.NS"))
        assert isinstance(result, dict)
        assert "RELIANCE.NS" in result
        assert list(result["RELIANCE.NS"].columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_empty_chunk_result_skipped(self):
        with patch("core.data_provider._yf_download_chunk",
                   return_value=pd.DataFrame()):
            result = fetch_daily_batch(["RELIANCE.NS"], _cfg())
        assert result == {}

    def test_download_exception_skips_chunk(self):
        with patch("core.data_provider._yf_download_chunk",
                   side_effect=Exception("network error")):
            result = fetch_daily_batch(["RELIANCE.NS"], _cfg())
        assert isinstance(result, dict)

    def test_benchmark_included_in_fetch(self):
        tickers = ["RELIANCE.NS"]
        captured = []
        def fake_download(symbols, period, interval):
            captured.extend(symbols)
            return pd.DataFrame()
        with patch("core.data_provider._yf_download_chunk", side_effect=fake_download):
            fetch_daily_batch(tickers, _cfg())
        assert "^NSEI" in captured

    def test_columns_title_cased(self):
        tickers = ["RELIANCE.NS", "^NSEI"]
        fake = _fake_multi_df(tickers)
        with patch("core.data_provider._yf_download_chunk", return_value=fake):
            result = fetch_daily_batch(["RELIANCE.NS"], _cfg())
        if "RELIANCE.NS" in result:
            cols = list(result["RELIANCE.NS"].columns)
            assert all(c[0].isupper() for c in cols)
        else:
            pytest.fail("RELIANCE.NS missing from result")

    def test_fyers_session_manager_skips_stale_dotenv_token(self, tmp_path, monkeypatch):
        from core.data_provider import FyersSessionManager

        env_path = tmp_path / ".env"
        env_path.write_text("FYERS_ACCESS_TOKEN=stale-token\n", encoding="utf-8")
        stale_timestamp = 1_700_000_000
        os.utime(env_path, (stale_timestamp, stale_timestamp))
        monkeypatch.chdir(tmp_path)

        FyersSessionManager._instance = None
        cfg = _cfg(USE_FYERS=True, FYERS_CLIENT_ID="client", FYERS_ACCESS_TOKEN="stale-token")

        with patch("fyers_apiv3.fyersModel.FyersModel") as model:
            client = FyersSessionManager.get_client(cfg)

        assert client is None
        model.assert_not_called()
        FyersSessionManager._instance = None

    def test_fyers_session_manager_allows_shell_overrides_when_dotenv_is_stale(self, tmp_path, monkeypatch):
        from core.data_provider import FyersSessionManager

        env_path = tmp_path / ".env"
        env_path.write_text("FYERS_ACCESS_TOKEN=old-file-token\n", encoding="utf-8")
        stale_timestamp = 1_700_000_000
        os.utime(env_path, (stale_timestamp, stale_timestamp))
        monkeypatch.chdir(tmp_path)

        FyersSessionManager._instance = None
        cfg = _cfg(USE_FYERS=True, FYERS_CLIENT_ID="client", FYERS_ACCESS_TOKEN="fresh-shell-token")
        sentinel = object()

        with patch("fyers_apiv3.fyersModel.FyersModel", return_value=sentinel) as model:
            client = FyersSessionManager.get_client(cfg)

        assert client is sentinel
        model.assert_called_once()
        FyersSessionManager._instance = None
