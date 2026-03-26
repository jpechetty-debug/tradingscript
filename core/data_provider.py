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

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from .config import IST, SystemConfig
from utils.retry import retry_with_backoff, YFINANCE_BREAKER, guarded_call

log = logging.getLogger("sovereign.data")

# ── Constants ─────────────────────────────────────────────────────────────────
FYERS_RESOLUTION: str    = "D"
FYERS_DATE_FORMAT: str   = "1"
FYERS_CONT_FLAG: str     = "1"
FYERS_LOOKBACK_DAYS: int = 365
YFINANCE_CHUNK_SIZE: int = 40
YFINANCE_INTERVAL: str   = "1d"


@retry_with_backoff(retries=4, base_delay=1.0, max_delay=30.0)
def _yf_download_chunk(
    symbols: list[str],
    period: str,
    interval: str,
) -> pd.DataFrame:
    """yfinance batch download with exponential-backoff retry."""
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

    _instance: Optional[object] = None

    @classmethod
    def get_client(cls, config: SystemConfig) -> Optional[object]:
        """
        Return a cached Fyers client, constructing it on first call.

        Returns ``None`` (with an appropriate log message) if credentials
        are absent, the package is not installed, or initialisation fails.
        """
        if cls._instance is not None:
            return cls._instance

        if not config.FYERS_CLIENT_ID or not config.FYERS_ACCESS_TOKEN:
            log.warning(
                "Fyers credentials not set (FYERS_CLIENT_ID / FYERS_ACCESS_TOKEN); "
                "falling back to yfinance."
            )
            return None

        try:
            from fyers_apiv3 import fyersModel  # type: ignore[import]

            cls._instance = fyersModel.FyersModel(
                client_id=config.FYERS_CLIENT_ID,
                token=config.FYERS_ACCESS_TOKEN,
                log_path=os.getcwd(),
            )
            log.info("Fyers client initialised (client_id=%s).", config.FYERS_CLIENT_ID)
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
    config: SystemConfig,
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
        "range_from":  (datetime.now() - timedelta(days=FYERS_LOOKBACK_DAYS)).strftime("%Y-%m-%d"),
        "range_to":    datetime.now().strftime("%Y-%m-%d"),
        "cont_flag":   FYERS_CONT_FLAG,
    }

    try:
        res = fyers.history(data=request)
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
    config: SystemConfig,
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
    all_symbols: list[str] = tickers + [config.BENCHMARK]

    fyers_map: dict[str, str] = {
        t: f"NSE:{t.replace('.NS', '')}-EQ" for t in all_symbols
    }
    fyers_map[config.BENCHMARK] = "NSE:NIFTY50-INDEX"

    # ── 1. Fyers path ─────────────────────────────────────────────────────────
    if config.USE_FYERS:
        log.info(
            "📡 Fyers download: %d symbols (parallel, workers=%d)…",
            len(all_symbols), config.MAX_WORKERS,
        )
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
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

        if out:
            log.info("Fyers: %d / %d symbols fetched.", len(out), len(all_symbols))
            return out

        log.warning("Fyers returned no data — falling back to yfinance.")

    # ── 2. yfinance fallback ──────────────────────────────────────────────────
    log.warning("📡 yfinance fallback: %d symbols (chunked)…", len(all_symbols))

    for i in range(0, len(all_symbols), YFINANCE_CHUNK_SIZE):
        chunk = all_symbols[i : i + YFINANCE_CHUNK_SIZE]

        try:
            raw: pd.DataFrame = guarded_call(
                lambda: _yf_download_chunk(chunk, config.DAILY_PERIOD, YFINANCE_INTERVAL),
                breaker=YFINANCE_BREAKER,
                max_attempts=4,
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
                    log.debug("yfinance: %s — empty after dropna.", t)
                    continue

                out[t] = df
                log.debug("yfinance OK: %s (%d bars).", t, len(df))

            except KeyError:
                log.debug("yfinance: %s not found in response.", t, exc_info=True)
            except Exception:
                log.error(
                    "yfinance: unexpected error extracting %s.", t, exc_info=True
                )

    log.info("yfinance: %d / %d symbols fetched.", len(out), len(all_symbols))
    return out
