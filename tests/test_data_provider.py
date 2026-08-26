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

    def test_fyers_session_manager_env_not_exists(self, monkeypatch):
        from core.data_provider import FyersSessionManager
        with patch("pathlib.Path.exists", return_value=False):
            assert FyersSessionManager._warn_if_token_stale(_cfg()) is False

    def test_fyers_session_manager_env_token_oserror(self):
        from core.data_provider import FyersSessionManager
        with patch("pathlib.Path.read_text", side_effect=OSError):
            assert FyersSessionManager._env_token(MagicMock()) is None

    def test_fyers_session_manager_stat_oserror(self, tmp_path, monkeypatch):
        from core.data_provider import FyersSessionManager
        env_path = tmp_path / ".env"
        env_path.write_text("FYERS_ACCESS_TOKEN=token\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", side_effect=OSError):
                assert FyersSessionManager._warn_if_token_stale(_cfg(FYERS_ACCESS_TOKEN="token")) is False

    def test_fyers_session_manager_cached_instance(self):
        from core.data_provider import FyersSessionManager
        FyersSessionManager._instance = "cached"
        assert FyersSessionManager.get_client(_cfg()) == "cached"
        FyersSessionManager._instance = None

    def test_fyers_session_manager_import_error(self, monkeypatch):
        from core.data_provider import FyersSessionManager
        cfg = _cfg(USE_FYERS=True, FYERS_CLIENT_ID="client", FYERS_ACCESS_TOKEN="token")
        import builtins
        real_import = builtins.__import__
        def fake_import(name, *args, **kwargs):
            if name == "fyers_apiv3":
                raise ImportError("mock import error")
            return real_import(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=fake_import):
            with patch("pathlib.Path.exists", return_value=False):
                assert FyersSessionManager.get_client(cfg) is None
        FyersSessionManager._instance = None

    def test_fyers_session_manager_generic_exception(self):
        from core.data_provider import FyersSessionManager
        cfg = _cfg(USE_FYERS=True, FYERS_CLIENT_ID="client", FYERS_ACCESS_TOKEN="token")
        with patch("fyers_apiv3.fyersModel.FyersModel", side_effect=Exception("error")):
            with patch("pathlib.Path.exists", return_value=False):
                assert FyersSessionManager.get_client(cfg) is None
        FyersSessionManager._instance = None

    def test_fetch_single_ticker_success(self):
        from core.data_provider import fetch_single_ticker
        cfg = _cfg(USE_FYERS=True, FYERS_CLIENT_ID="client", FYERS_ACCESS_TOKEN="token")
        mock_client = MagicMock()
        mock_client.history.return_value = {
            "s": "ok",
            "candles": [[1700000000, 100, 105, 95, 102, 1000]]
        }
        with patch("core.data_provider.FyersSessionManager.get_client", return_value=mock_client):
            ticker, df = fetch_single_ticker("RELIANCE.NS", "NSE:RELIANCE-EQ", cfg)
            assert ticker == "RELIANCE.NS"
            assert df is not None
            assert not df.empty

    def test_fetch_single_ticker_value_error(self):
        from core.data_provider import fetch_single_ticker
        cfg = _cfg(USE_FYERS=True)
        mock_client = MagicMock()
        mock_client.history.return_value = {"s": "error", "message": "bad"}
        with patch("core.data_provider.FyersSessionManager.get_client", return_value=mock_client):
            ticker, df = fetch_single_ticker("RELIANCE.NS", "NSE:RELIANCE-EQ", cfg)
            assert df is None

    def test_fetch_single_ticker_exception(self):
        from core.data_provider import fetch_single_ticker
        cfg = _cfg(USE_FYERS=True)
        mock_client = MagicMock()
        mock_client.history.side_effect = Exception("network")
        with patch("core.data_provider.FyersSessionManager.get_client", return_value=mock_client):
            ticker, df = fetch_single_ticker("RELIANCE.NS", "NSE:RELIANCE-EQ", cfg)
            assert df is None

    def test_fetch_daily_batch_fyers_path(self):
        cfg = _cfg(USE_FYERS=True)
        def fake_fetch(t, fsym, c):
            if t == "RELIANCE.NS":
                return t, pd.DataFrame({"Close": [100]})
            return t, None
        with patch("core.data_provider.fetch_single_ticker", side_effect=fake_fetch):
            res = fetch_daily_batch(["RELIANCE.NS", "INFY.NS"], cfg)
            assert "RELIANCE.NS" in res
            assert "INFY.NS" not in res

    def test_fetch_daily_batch_fyers_thread_crash(self):
        cfg = _cfg(USE_FYERS=True, MAX_WORKERS=1)
        def fake_fetch(t, fsym, c):
            raise Exception("thread crash")
        with patch("core.data_provider.fetch_single_ticker", side_effect=fake_fetch):
            with patch("core.data_provider._yf_download_chunk", return_value=pd.DataFrame()):
                res = fetch_daily_batch(["RELIANCE.NS"], cfg)
                assert not res

    def test_yf_chunk_multiindex_missing_ticker(self):
        fake = _fake_multi_df(["RELIANCE.NS"])
        # We ask for INFY.NS, but fake only has RELIANCE.NS
        with patch("core.data_provider._yf_download_chunk", return_value=fake):
            result = fetch_daily_batch(["INFY.NS"], _cfg())
            assert "INFY.NS" not in result

    def test_yf_chunk_flat_multi_ticker_warning(self):
        n = 10
        fake = pd.DataFrame({"open": np.ones(n)}, index=pd.date_range("2024-01-01", periods=n))
        with patch("core.data_provider._yf_download_chunk", return_value=fake):
            # 2 tickers => expects multiindex, gets flat
            result = fetch_daily_batch(["RELIANCE.NS", "INFY.NS"], _cfg())
            assert not result

    def test_yf_chunk_empty_after_dropna(self):
        n = 10
        fake = pd.DataFrame({"open": [np.nan]*n}, index=pd.date_range("2024-01-01", periods=n))
        with patch("core.data_provider._yf_download_chunk", return_value=fake):
            result = fetch_daily_batch(["RELIANCE.NS"], _cfg(BENCHMARK="RELIANCE.NS"))
            assert not result

    def test_yf_chunk_generic_exception(self):
        fake = MagicMock()
        fake.empty = False
        fake.shape = (10, 5)
        fake.columns = pd.MultiIndex.from_tuples([("RELIANCE.NS", "Open")])
        fake.xs.side_effect = Exception("xs failed")
        with patch("core.data_provider._yf_download_chunk", return_value=fake):
            result = fetch_daily_batch(["RELIANCE.NS"], _cfg())
            assert not result
