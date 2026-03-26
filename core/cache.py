"""
core/cache.py
=============
LRU caching for computationally expensive, frequently repeated
calculations in the Sovereign Engine.

Two things dominate hot-path CPU time:
1. ``true_volume_profile``  — O(n·bins) loop per ticker per scan tick
2. ``_build_corr_matrix``   — O(tickers²) pairwise correlation per scan

Both are pure functions of their inputs and can safely be memoised
for the duration of a single scan cycle.

Usage
-----
::

    from core.cache import ScanCache

    cache = ScanCache()

    # Volume profile — cached by (ticker, last_bar_date, lookback, bins)
    poc, val, vah = cache.volume_profile(df, ticker="RELIANCE.NS",
                                          lookback=30, bins=100)

    # Correlation matrix — cached per scan (replace on new data)
    corr = cache.corr_matrix(processed, config)

    cache.clear()          # between scan cycles
    cache.stats()          # {"hits": 120, "misses": 40, ...}
"""

from __future__ import annotations

import hashlib
import logging
import time
from functools import lru_cache
from typing import Any, Optional

import numpy as np
import pandas as pd

from .factors import true_volume_profile
from .config import SystemConfig

log = logging.getLogger("sovereign.cache")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _df_cache_key(df: pd.DataFrame, extra: str = "") -> str:
    """
    Stable, cheap cache key for a DataFrame.

    Uses the last bar date + row count + last Close value — fast to
    compute and captures the cases where the frame has grown or the
    last price has moved.
    """
    last_date = str(df.index[-1]) if len(df) > 0 else "empty"
    last_close = f"{df['Close'].iloc[-1]:.4f}" if "Close" in df.columns and len(df) > 0 else "0"
    return f"{last_date}|{len(df)}|{last_close}|{extra}"


# ─────────────────────────────────────────────────────────────────────────────
# SCAN-SCOPED CACHE
# ─────────────────────────────────────────────────────────────────────────────

class ScanCache:
    """
    In-memory cache scoped to one scan cycle.

    Designed to be created fresh at the start of ``run_scan`` and
    discarded (or ``.clear()``-ed) afterwards.  Thread-safe for reads
    once populated; writes should happen from a single thread (the
    scan loop).

    Attributes
    ----------
    max_vprofile_entries : int
        Maximum number of volume profile entries to keep in memory.
        Each entry is ~3 floats; 500 entries ≈ negligible RAM.
    """

    def __init__(self, max_vprofile_entries: int = 500) -> None:
        self._vprofile:  dict[str, tuple[float, float, float]] = {}
        self._corr:      Optional[pd.DataFrame] = None
        self._corr_key:  str = ""
        self._max        = max_vprofile_entries

        # Telemetry
        self._hits:   int = 0
        self._misses: int = 0
        self._vp_time_saved: float = 0.0

    # ── Volume profile ────────────────────────────────────────────────────────

    def volume_profile(
        self,
        df:       pd.DataFrame,
        ticker:   str = "",
        lookback: int = 30,
        bins:     int = 100,
    ) -> tuple[float, float, float]:
        """
        Return ``(POC, VAL, VAH)`` for *df*, using cache when possible.

        Cache key: ``{ticker}|{last_bar_date}|{nrows}|{last_close}|{lookback}|{bins}``

        The key is invalidated whenever new price data arrives for the
        ticker (last bar date or close changes) or when parameters change.
        """
        key = f"{ticker}|" + _df_cache_key(df, f"{lookback}|{bins}")

        if key in self._vprofile:
            self._hits += 1
            return self._vprofile[key]

        self._misses += 1
        t0  = time.monotonic()
        val = true_volume_profile(df, lookback=lookback, bins=bins)
        elapsed = time.monotonic() - t0
        self._vp_time_saved += elapsed   # will subtract from missed entries

        if len(self._vprofile) >= self._max:
            # Evict oldest quarter of entries (simple FIFO approximation)
            evict_n  = max(1, self._max // 4)
            for k in list(self._vprofile.keys())[:evict_n]:
                del self._vprofile[k]
            log.debug("VProfile cache: evicted %d entries (size cap=%d).", evict_n, self._max)

        self._vprofile[key] = val
        return val

    # ── Correlation matrix ────────────────────────────────────────────────────

    def corr_matrix(
        self,
        processed: dict[str, pd.DataFrame],
        config:    SystemConfig,
    ) -> pd.DataFrame:
        """
        Build (or return cached) pairwise return correlation matrix.

        The matrix is invalidated when the set of tickers changes or
        the last bar date of the benchmark changes — both situations
        indicate new data has arrived.
        """
        bench_key = config.BENCHMARK
        bench_df  = processed.get(bench_key)
        last_date = str(bench_df.index[-1]) if bench_df is not None and len(bench_df) else "?"
        n_tickers = len(processed)
        new_key   = f"{last_date}|{n_tickers}"

        if self._corr is not None and new_key == self._corr_key:
            self._hits += 1
            return self._corr

        self._misses += 1
        t0 = time.monotonic()
        matrix = _build_corr_matrix(processed, config)
        log.debug(
            "Corr matrix computed in %.3fs (shape=%s).",
            time.monotonic() - t0,
            matrix.shape,
        )
        self._corr     = matrix
        self._corr_key = new_key
        return matrix

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Drop all cached values — call between scan cycles if reusing."""
        self._vprofile.clear()
        self._corr     = None
        self._corr_key = ""
        log.debug("ScanCache cleared.")

    def stats(self) -> dict[str, Any]:
        """Return a snapshot of cache effectiveness."""
        total = self._hits + self._misses
        return {
            "hits":         self._hits,
            "misses":       self._misses,
            "hit_rate":     round(self._hits / total, 4) if total else 0.0,
            "vprofile_size": len(self._vprofile),
            "corr_cached":  self._corr is not None,
        }

    def log_stats(self) -> None:
        """Emit cache stats at DEBUG level."""
        s = self.stats()
        log.debug(
            "ScanCache stats: hits=%d misses=%d hit_rate=%.1f%% "
            "vprofile_entries=%d corr_cached=%s",
            s["hits"], s["misses"], s["hit_rate"] * 100,
            s["vprofile_size"], s["corr_cached"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE CORR MATRIX BUILDER  (extracted from screener_v14_modular.py)
# ─────────────────────────────────────────────────────────────────────────────

def _build_corr_matrix(
    processed: dict[str, pd.DataFrame],
    config:    SystemConfig,
) -> pd.DataFrame:
    """
    Compute pairwise daily-return correlation from *processed* data.

    Same logic as the inline version in screener_v14_modular.py but
    centralised here so both the screener and the backtest use the
    same implementation.
    """
    bench = config.BENCHMARK
    returns: dict[str, pd.Series] = {}

    for ticker, df in processed.items():
        if ticker == bench or df.empty:
            continue
        rets = df["Close"].pct_change().dropna().tail(config.COV_LOOKBACK)
        if len(rets) >= 20:
            returns[ticker] = rets

    if len(returns) < 2:
        return pd.DataFrame()

    ret_df = pd.DataFrame(returns).dropna(how="all")
    return ret_df.corr()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON  (opt-in convenience)
# ─────────────────────────────────────────────────────────────────────────────

#: Module-level cache instance.  Use this in screener_v14_modular.py by
#: calling ``SCAN_CACHE.clear()`` at the start of each run_scan() call and
#: passing ``SCAN_CACHE`` to score_ticker / _build_corr_matrix wrappers.
SCAN_CACHE = ScanCache()
