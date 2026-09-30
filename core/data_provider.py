"""
core/data_provider.py
=====================
Market data retrieval for the Sovereign Engine.

Fixes vs original
-----------------
* All bare ``except Exception as e: log.error(...)`` blocks that swallowed
  tracebacks are replaced with ``exc_info=True`` so the full stack is
  available at DEBUG/ERROR level.
* ``_fyers_to_df`` raises ``ValueError`` on bad API status instead of
  silently returning an empty DataFrame — surfaces configuration problems
  early rather than masking them.
* Full type annotations on all public symbols.
* Magic literals (chunk size, lookback days, resolution strings) extracted
  to named module-level constants.
* ``FyersSessionManager.get_client`` guards against missing credentials and
  unavailable package cleanly, with appropriate log levels per case.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional, Protocol

import pandas as pd
import yfinance as yf

from .config import IST, MarketDataSettings, SystemConfig, get_secret_value
from .runtime_paths import RUNTIME_PATHS, ensure_runtime_dirs
from .retry import FYERS_BREAKER, YFINANCE_BREAKER, guarded_call

log = logging.getLogger("sovereign.data")
ensure_runtime_dirs()

MarketDataConfig = MarketDataSettings | SystemConfig


def _setting(config: MarketDataConfig, typed_name: str, legacy_name: str) -> Any:
    return getattr(config, typed_name) if hasattr(config, typed_name) else getattr(config, legacy_name)


class _HistoryClient(Protocol):
    def history(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        ...

# ── Constants ─────────────────────────────────────────────────────────────────
FYERS_RESOLUTION: str    = "D"
FYERS_DATE_FORMAT: str   = "1"
FYERS_CONT_FLAG: str     = "1"
FYERS_LOOKBACK_DAYS: int = 365
YFINANCE_CHUNK_SIZE: int = 40
YFINANCE_INTERVAL: str   = "1d"
_YFINANCE_CACHE_DIR = RUNTIME_PATHS.state_dir / "yfinance"
_YFINANCE_FETCH_LOCK = threading.Lock()
_YFINANCE_RATE_LOCK = threading.Lock()
_YFINANCE_LAST_REQUEST = 0.0


def _cache_path(symbol: str, period: str, *, negative: bool = False) -> Path:
    """Return a filesystem-safe cache path for one Yahoo response."""
    digest = hashlib.sha256(f"{symbol}|{period}|{YFINANCE_INTERVAL}".encode()).hexdigest()
    suffix = ".missing" if negative else ".pkl"
    return _YFINANCE_CACHE_DIR / f"{digest}{suffix}"


def _is_fresh(path: Path, ttl_hours: float) -> bool:
    return ttl_hours > 0 and path.exists() and (time.time() - path.stat().st_mtime) < ttl_hours * 3600


def _load_cached_daily(symbol: str, period: str, ttl_hours: float) -> Optional[pd.DataFrame]:
    path = _cache_path(symbol, period)
    if not _is_fresh(path, ttl_hours):
        return None
    try:
        frame = pd.read_pickle(path)
        return frame if isinstance(frame, pd.DataFrame) and not frame.empty else None
    except Exception:
        log.warning("Ignoring unreadable yfinance cache for %s.", symbol, exc_info=True)
        return None


def _store_cached_daily(symbol: str, period: str, frame: pd.DataFrame) -> None:
    _YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = _cache_path(symbol, period)
    tmp = target.with_suffix(".tmp")
    frame.to_pickle(tmp)
    tmp.replace(target)


def _is_negative_cached(symbol: str, period: str, ttl_hours: float) -> bool:
    return _is_fresh(_cache_path(symbol, period, negative=True), ttl_hours)


def _store_negative_cache(symbol: str, period: str) -> None:
    _YFINANCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(symbol, period, negative=True).touch()


def _rate_limited_yf_download(symbols: list[str], period: str, interval: str, min_interval: float) -> pd.DataFrame:
    """Serialize Yahoo requests across overlapping scans and enforce a safe pace."""
    global _YFINANCE_LAST_REQUEST
    with _YFINANCE_RATE_LOCK:
        delay = min_interval - (time.monotonic() - _YFINANCE_LAST_REQUEST)
        if delay > 0:
            time.sleep(delay)
        try:
            return _yf_download_chunk(symbols, period, interval)
        finally:
            _YFINANCE_LAST_REQUEST = time.monotonic()


def _yf_download_chunk(
    symbols: list[str],
    period: str,
    interval: str,
) -> pd.DataFrame:
    """Perform one yfinance request; the caller owns retry policy."""
    return yf.download(
        symbols,
        period=period,
        interval=interval,
        group_by="ticker",
        progress=False,
        auto_adjust=True,
    )


# ── Fyers session singleton ───────────────────────────────────────────────────

class FyersSessionManager:
    """Lazy singleton for the Fyers API client."""

    _instance: Optional[_HistoryClient] = None

    # Fyers access tokens are issued daily and expire at midnight IST.
    # Keeping a stale token in .env causes every history() call to return
    # {"s": "error", "message": "Invalid token"} which _fyers_to_df raises
    # as ValueError.  fetch_single_ticker catches that and falls back to
    # yfinance silently — the operator only notices when scan times triple.
    # We detect staleness at client construction time so the warning appears
    # at startup rather than buried in per-ticker DEBUG logs.
    _TOKEN_MAX_AGE_HOURS: int = 20   # warn if .env was last written > 20h ago

    @classmethod
    def _env_token(cls, env_path: Path) -> Optional[str]:
        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("FYERS_ACCESS_TOKEN="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            return None
        return None

    @classmethod
    def _warn_if_token_stale(cls, config: MarketDataConfig) -> bool:
        """
        Emit a WARNING if the loaded access token is likely stale.

        Strategy: check the mtime of the .env file that was loaded by
        python-dotenv.  If it was last written more than ``_TOKEN_MAX_AGE_HOURS``
        ago the token is probably yesterday's and will fail.  Falls back
        gracefully if no .env file is found (token may have been injected
        via the environment directly, in which case we cannot check age).
        """
        from datetime import timezone

        env_path = Path(".env")
        if not env_path.exists():
            # Token came from the shell environment — we cannot check age.
            return False
        env_token = cls._env_token(env_path)
        token = _setting(config, "fyers_access_token", "FYERS_ACCESS_TOKEN")
        if not env_token or env_token != get_secret_value(token):
            return False
        try:
            mtime = env_path.stat().st_mtime
            age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600
            if age_hours > cls._TOKEN_MAX_AGE_HOURS:
                log.warning(
                    "FYERS_ACCESS_TOKEN may be stale — .env was last written "
                    "%.1f hours ago (limit: %d h).  Fyers tokens expire daily at "
                    "midnight IST.  Run fyers_setup.py to refresh, or the engine "
                    "will silently fall back to yfinance for all tickers.",
                    age_hours,
                    cls._TOKEN_MAX_AGE_HOURS,
                )
                return True
        except OSError:
            pass   # stat failed — ignore, not worth crashing over
        return False

    @classmethod
    def get_client(cls, config: MarketDataConfig) -> Optional[_HistoryClient]:
        """
        Return a cached Fyers client, constructing it on first call.

        Returns ``None`` (with an appropriate log message) if credentials
        are absent, the package is not installed, or initialisation fails.
        """
        if cls._instance is not None:
            return cls._instance

        client_id = _setting(config, "fyers_client_id", "FYERS_CLIENT_ID")
        access_token = _setting(config, "fyers_access_token", "FYERS_ACCESS_TOKEN")
        if not client_id or not access_token:
            log.warning(
                "Fyers credentials not set (FYERS_CLIENT_ID / FYERS_ACCESS_TOKEN); "
                "falling back to yfinance."
            )
            return None

        # Check token age before attempting to construct the client.
        if cls._warn_if_token_stale(config):
            log.warning(
                "Skipping Fyers initialisation because the loaded token appears stale; "
                "using yfinance until scripts/fyers_setup.py refreshes .env."
            )
            return None

        try:
            from fyers_apiv3 import fyersModel

            cls._instance = fyersModel.FyersModel(
                client_id=get_secret_value(client_id),
                token=get_secret_value(access_token),
                log_path=str(RUNTIME_PATHS.logs_dir),
            )
            log.info("Fyers client initialised (client_id=%s).", client_id)
            return cls._instance

        except ImportError:
            log.warning("fyers_apiv3 not installed — falling back to yfinance.")
            return None

        except Exception:
            # Auth failure, network error, SDK bug — full traceback at ERROR.
            log.error(
                "Failed to initialise Fyers client; will use yfinance.",
                exc_info=True,
            )
            return None


# ── Fyers response parser ─────────────────────────────────────────────────────

def _fyers_to_df(data: dict) -> pd.DataFrame:
    """
    Parse a Fyers ``history`` API response into an OHLCV DataFrame.

    Raises
    ------
    ValueError
        If the response status is not ``"ok"`` or the ``candles`` key is
        missing — makes configuration errors visible immediately.
    """
    status = data.get("s")
    if status != "ok":
        raise ValueError(
            f"Fyers API returned status={status!r}: {data.get('message', '<no message>')}"
        )
    if "candles" not in data:
        raise ValueError("Fyers API response is missing the 'candles' key.")

    df = pd.DataFrame(
        data["candles"],
        columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )
    df["Timestamp"] = (
        pd.to_datetime(df["Timestamp"], unit="s")
        .dt.tz_localize("UTC")
        .dt.tz_convert(IST)
    )
    df.set_index("Timestamp", inplace=True)
    return df


# ── Single-ticker Fyers fetch ─────────────────────────────────────────────────

def fetch_single_ticker(
    ticker: str,
    fsym: str,
    config: MarketDataConfig,
) -> tuple[str, Optional[pd.DataFrame]]:
    """
    Fetch one year of daily OHLCV for *ticker* from Fyers.

    Returns ``(ticker, DataFrame)`` on success, ``(ticker, None)`` on any
    failure. Never raises — all exceptions are caught and logged.
    """
    fyers = FyersSessionManager.get_client(config)
    if fyers is None:
        return ticker, None

    request = {
        "symbol":      fsym,
        "resolution":  FYERS_RESOLUTION,
        "date_format": FYERS_DATE_FORMAT,
        "range_from":  (datetime.now(IST) - timedelta(days=FYERS_LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
        "range_to":    datetime.now(IST).strftime("%Y-%m-%d"),
        "cont_flag":   FYERS_CONT_FLAG,
    }

    try:
        res = guarded_call(
            lambda: fyers.history(data=request),
            breaker=FYERS_BREAKER,
            max_attempts=2,
            base_delay=0.25,
            max_delay=2.0,
            label=f"fyers history[{ticker}]",
        )
        df  = _fyers_to_df(res)
        if not df.empty:
            log.debug("Fyers OK: %s (%d bars).", ticker, len(df))
            return ticker, df
        log.warning("Fyers returned empty candles for %s.", ticker)
        return ticker, None

    except ValueError as exc:
        # API-level problem (bad status, missing key) — WARNING because it
        # is actionable (check credentials / symbol mapping).
        log.warning("Fyers API error for %s: %s", ticker, exc)
        return ticker, None

    except Exception:
        # Network timeout, SDK bug, etc. — DEBUG to avoid flooding logs
        # during market-close periods when some symbols are unavailable.
        log.debug("Unexpected error fetching %s from Fyers.", ticker, exc_info=True)
        return ticker, None


# ── Batch fetch (Fyers → yfinance fallback) ───────────────────────────────────

def fetch_daily_batch(
    tickers: list[str],
    config: MarketDataConfig,
) -> dict[str, pd.DataFrame]:
    """
    Download daily OHLCV for *tickers* plus the benchmark symbol.

    Strategy
    --------
    1. Attempt Fyers in parallel (up to ``config.MAX_WORKERS`` threads).
    2. If Fyers is disabled, unconfigured, or returns nothing, fall back
       to yfinance in chunks of ``YFINANCE_CHUNK_SIZE`` symbols.

    Returns
    -------
    ``{ticker: OHLCV_DataFrame}`` for every symbol successfully fetched.
    """
    out: dict[str, pd.DataFrame] = {}
    benchmark = str(_setting(config, "benchmark", "BENCHMARK"))
    max_workers = int(_setting(config, "max_workers", "MAX_WORKERS"))
    use_fyers = bool(_setting(config, "use_fyers", "USE_FYERS"))
    daily_period = str(_setting(config, "daily_period", "DAILY_PERIOD"))
    chunk_size = max(1, int(_setting(config, "yfinance_chunk_size", "YFINANCE_CHUNK_SIZE")))
    min_interval = max(0.0, float(_setting(config, "yfinance_min_chunk_interval", "YFINANCE_MIN_CHUNK_INTERVAL")))
    cache_ttl = max(0.0, float(_setting(config, "yfinance_cache_ttl_hours", "YFINANCE_CACHE_TTL_HOURS")))
    negative_ttl = max(0.0, float(_setting(config, "yfinance_negative_cache_ttl_hours", "YFINANCE_NEGATIVE_CACHE_TTL_HOURS")))
    all_symbols: list[str] = sorted(list(set(tickers + [benchmark])))

    fyers_map: dict[str, str] = {
        t: f"NSE:{t.replace('.NS', '')}-EQ" for t in all_symbols
    }
    fyers_map[benchmark] = "NSE:NIFTY50-INDEX"

    # ── 1. Fyers path ─────────────────────────────────────────────────────────
    if use_fyers:
        log.info(
            "📡 Fyers download: %d symbols (parallel, workers=%d)…",
            len(all_symbols), max_workers,
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: list[Future] = [
                executor.submit(fetch_single_ticker, t, fyers_map[t], config)
                for t in all_symbols
            ]
            for future in futures:
                try:
                    ticker, df = future.result()
                    if df is not None:
                        out[ticker] = df
                except Exception:
                    # future.result() re-raises if the thread itself crashed —
                    # should not happen (fetch_single_ticker never raises), but
                    # guard anyway.
                    log.error(
                        "Worker thread raised unexpectedly in fetch_daily_batch.",
                        exc_info=True,
                    )

        if len(out) == len(all_symbols):
            log.info("Fyers: all %d symbols fetched.", len(all_symbols))
            return out

        missing_symbols = [symbol for symbol in all_symbols if symbol not in out]
        log.warning(
            "Fyers: %d / %d symbols fetched; filling %d missing symbols via yfinance.",
            len(out), len(all_symbols), len(missing_symbols),
        )
    else:
        missing_symbols = all_symbols

    # ── 2. yfinance fallback ──────────────────────────────────────────────────
    # This lock is deliberately broader than one request: a second overlapping
    # scan re-checks the cache after the first scan completes instead of issuing
    # the same Yahoo requests again.
    with _YFINANCE_FETCH_LOCK:
        uncached_symbols: list[str] = []
        for symbol in missing_symbols:
            cached = _load_cached_daily(symbol, daily_period, cache_ttl)
            if cached is not None:
                out[symbol] = cached
            elif _is_negative_cached(symbol, daily_period, negative_ttl):
                log.debug("yfinance: skipping negative-cached symbol %s.", symbol)
            else:
                uncached_symbols.append(symbol)

        log.info(
            "yfinance fallback: %d symbols (%d cache hits, %d network candidates).",
            len(missing_symbols), len(out), len(uncached_symbols),
        )

        for i in range(0, len(uncached_symbols), chunk_size):
            chunk = uncached_symbols[i : i + chunk_size]

            try:
                raw: pd.DataFrame = guarded_call(
                    lambda: _rate_limited_yf_download(
                        chunk, daily_period, YFINANCE_INTERVAL, min_interval
                    ),
                    breaker=YFINANCE_BREAKER,
                    max_attempts=3,
                    base_delay=1.0,
                    max_delay=30.0,
                    label=f"yfinance chunk[{chunk[0]}]",
                )
            except Exception:
                log.error(
                    "yfinance: download failed for chunk starting at %s.",
                    chunk[0],
                    exc_info=True,
                )
                continue

            if raw.empty:
                log.debug("yfinance: empty result for chunk %s.", chunk)
                continue

            log.info("yfinance chunk %s: shape=%s.", chunk, raw.shape)

            for t in chunk:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if t not in raw.columns.get_level_values(0):
                            _store_negative_cache(t, daily_period)
                            log.debug("yfinance: %s absent from MultiIndex result.", t)
                            continue
                        df = raw.xs(t, axis=1, level=0).copy()
                    else:
                        if len(chunk) == 1:
                            df = raw.copy()
                        else:
                            log.warning(
                                "Expected MultiIndex for multi-ticker chunk but got flat "
                                "columns — skipping %s.", t
                            )
                            continue

                    df.dropna(how="all", inplace=True)
                    df.columns = [c.title() for c in df.columns]

                    if df.empty:
                        _store_negative_cache(t, daily_period)
                        log.debug("yfinance: %s — empty after dropna.", t)
                        continue

                    out[t] = df
                    _store_cached_daily(t, daily_period, df)
                    log.debug("yfinance OK: %s (%d bars).", t, len(df))

                except KeyError:
                    log.debug("yfinance: %s not found in response.", t, exc_info=True)
                except Exception:
                    log.error(
                        "yfinance: unexpected error extracting %s.", t, exc_info=True
                    )

    log.info("yfinance: %d / %d symbols fetched.", len(out), len(all_symbols))
    return out
