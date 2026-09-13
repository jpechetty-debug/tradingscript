"""
core/async_data.py
==================
Async data fetching layer for the Sovereign Engine.

Why async?
----------
``fetch_daily_batch`` uses ``ThreadPoolExecutor`` which is effective but
creates one OS thread per in-flight request. For large universes (200+
tickers) this can exhaust the thread pool and introduce latency spikes.

This module provides ``async_fetch_daily_batch`` — a drop-in async
replacement that uses ``asyncio`` + ``aiohttp`` for HTTP-level
concurrency with configurable semaphore limits, meaning we can issue
hundreds of requests with a much smaller resource footprint.

The yfinance library is synchronous, so we run each yfinance call in
the default ``ThreadPoolExecutor`` via ``asyncio.get_event_loop().run_in_executor``.
This gives us both concurrency *and* the battle-tested yfinance parsing
logic.

Usage
-----
::

    import asyncio
    from core.async_data import async_fetch_daily_batch

    data = asyncio.run(async_fetch_daily_batch(tickers, config))

Or from an existing event loop::

    data = await async_fetch_daily_batch(tickers, config)

The return type is identical to ``fetch_daily_batch`` so callers need
no changes beyond the ``await``.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Optional

import pandas as pd
import yfinance as yf

from .config import SystemConfig
from .data_provider import (
    FyersSessionManager,
    _fyers_to_df,
    FYERS_LOOKBACK_DAYS,
    FYERS_RESOLUTION,
    FYERS_DATE_FORMAT,
    FYERS_CONT_FLAG,
    YFINANCE_CHUNK_SIZE,
    YFINANCE_INTERVAL,
)
from .retry import guarded_call, FYERS_BREAKER, YFINANCE_BREAKER

from datetime import datetime, timedelta

log = logging.getLogger("sovereign.async_data")

# Maximum concurrent requests (avoids overwhelming brokers / rate limits)
_FYERS_SEMAPHORE_LIMIT:    int = 20
_YFINANCE_SEMAPHORE_LIMIT: int = 10


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE TICKER — async Fyers
# ─────────────────────────────────────────────────────────────────────────────

async def _async_fetch_fyers(
    ticker:   str,
    fsym:     str,
    config:   SystemConfig,
    sem:      asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> tuple[str, Optional[pd.DataFrame]]:
    """
    Fetch one ticker from Fyers inside an asyncio semaphore.

    The Fyers SDK is synchronous, so we run it in *executor*.
    """
    async with sem:
        loop = asyncio.get_running_loop()
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

        def _sync_call() -> Optional[pd.DataFrame]:
            try:
                res = guarded_call(
                    lambda: fyers.history(data=request),
                    breaker=FYERS_BREAKER,
                    max_attempts=3,
                    base_delay=1.0,
                    label=f"fyers:{ticker}",
                )
                df = _fyers_to_df(res)
                return df if not df.empty else None
            except Exception:
                log.debug("Fyers async fetch failed for %s.", ticker, exc_info=True)
                return None

        df = await loop.run_in_executor(executor, _sync_call)
        return ticker, df


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK — async yfinance
# ─────────────────────────────────────────────────────────────────────────────

async def _async_fetch_yfinance_chunk(
    chunk:    list[str],
    config:   SystemConfig,
    sem:      asyncio.Semaphore,
    executor: ThreadPoolExecutor,
) -> dict[str, pd.DataFrame]:
    """
    Fetch one yfinance chunk inside an asyncio semaphore.

    Returns a partial ``{ticker: df}`` dict for the tickers in this chunk.
    """
    async with sem:
        loop = asyncio.get_running_loop()

        def _sync_download() -> dict[str, pd.DataFrame]:
            out: dict[str, pd.DataFrame] = {}
            try:
                raw = guarded_call(
                    partial(
                        yf.download,
                        chunk,
                        period=config.DAILY_PERIOD,
                        interval=YFINANCE_INTERVAL,
                        group_by="ticker",
                        progress=False,
                        auto_adjust=True,
                    ),
                    breaker=YFINANCE_BREAKER,
                    max_attempts=3,
                    base_delay=2.0,
                    label=f"yfinance_chunk:{chunk[0]}",
                )
            except Exception:
                log.error("yfinance async chunk %s failed.", chunk[0], exc_info=True)
                return out

            if raw is None or (hasattr(raw, "empty") and raw.empty):
                return out

            for t in chunk:
                try:
                    if isinstance(raw.columns, pd.MultiIndex):
                        if t not in raw.columns.get_level_values(0):
                            continue
                        df = raw.xs(t, axis=1, level=0).copy()
                    else:
                        df = raw.copy() if len(chunk) == 1 else None
                        if df is None:
                            continue

                    df.dropna(how="all", inplace=True)
                    df.columns = [c.title() for c in df.columns]
                    if not df.empty:
                        out[t] = df
                except Exception:
                    log.debug("yfinance async extraction failed for %s.", t, exc_info=True)

            return out

        return await loop.run_in_executor(executor, _sync_download)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ASYNC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

async def async_fetch_daily_batch(
    tickers: list[str],
    config:  SystemConfig,
) -> dict[str, pd.DataFrame]:
    """
    Async version of ``fetch_daily_batch``.

    Uses ``asyncio`` + bounded semaphores for concurrency, runs
    synchronous SDK calls in a ``ThreadPoolExecutor``.

    Parameters
    ----------
    tickers: NSE ticker list (e.g. ``["RELIANCE.NS", ...]``).
    config:  ``SystemConfig`` instance.

    Returns
    -------
    ``{ticker: OHLCV_DataFrame}`` — same contract as ``fetch_daily_batch``.
    """
    all_symbols = tickers + [config.BENCHMARK]
    fyers_map   = {t: f"NSE:{t.replace('.NS', '')}-EQ" for t in all_symbols}
    fyers_map[config.BENCHMARK] = "NSE:NIFTY50-INDEX"

    out: dict[str, pd.DataFrame] = {}

    max_workers = max(config.MAX_WORKERS, 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        # ── 1. Try Fyers (async) ──────────────────────────────────────────────
        if config.USE_FYERS and FyersSessionManager.get_client(config) is not None:
            fyers_sem = asyncio.Semaphore(_FYERS_SEMAPHORE_LIMIT)
            log.info(
                "📡 Async Fyers: %d symbols (sem=%d, workers=%d)…",
                len(all_symbols), _FYERS_SEMAPHORE_LIMIT, max_workers,
            )

            tasks = [
                _async_fetch_fyers(t, fyers_map[t], config, fyers_sem, executor)
                for t in all_symbols
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for fyers_item in results:
                if isinstance(fyers_item, BaseException):
                    log.error("Async Fyers task raised: %s", fyers_item)
                    continue
                ticker, df = fyers_item
                if df is not None:
                    out[ticker] = df

            if out:
                log.info("Async Fyers: %d / %d symbols fetched.", len(out), len(all_symbols))
                return out

            log.warning("Async Fyers returned nothing — falling back to yfinance.")

        # ── 2. yfinance async fallback ────────────────────────────────────────
        yf_sem = asyncio.Semaphore(_YFINANCE_SEMAPHORE_LIMIT)
        chunks = [
            all_symbols[i : i + YFINANCE_CHUNK_SIZE]
            for i in range(0, len(all_symbols), YFINANCE_CHUNK_SIZE)
        ]
        log.warning(
            "📡 Async yfinance: %d chunks (sem=%d, workers=%d)…",
            len(chunks), _YFINANCE_SEMAPHORE_LIMIT, max_workers,
        )

        chunk_tasks = [
            _async_fetch_yfinance_chunk(chunk, config, yf_sem, executor)
            for chunk in chunks
        ]
        chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)

        for chunk_item in chunk_results:
            if isinstance(chunk_item, BaseException):
                log.error("Async yfinance chunk raised: %s", chunk_item)
                continue
            out.update(chunk_item)

    log.info("Async yfinance: %d / %d symbols fetched.", len(out), len(all_symbols))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SYNC BRIDGE  (for callers that cannot use await)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_daily_batch_async(
    tickers: list[str],
    config:  SystemConfig,
) -> dict[str, pd.DataFrame]:
    """
    Synchronous wrapper around ``async_fetch_daily_batch``.

    Suitable for use in non-async callers (e.g. CLI, scripts, tests).
    Creates a new event loop if one is not already running.

    ::

        from core.async_data import fetch_daily_batch_async
        data = fetch_daily_batch_async(tickers, config)
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, async_fetch_daily_batch(tickers, config)).result()
    return asyncio.run(async_fetch_daily_batch(tickers, config))
