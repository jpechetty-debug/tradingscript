"""
tests/test_phase34.py
=====================
Tests for Phase 3 & 4 deliverables:
  - core/telemetry.py  (structured logging + ScanMetrics)
  - core/retry.py      (retry + circuit breaker)
  - core/cache.py      (ScanCache for volume profile + corr matrix)
  - core/backtest.py   (walk-forward trade simulation helpers)
"""

from __future__ import annotations

import asyncio
import logging
import sys
import os
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub fyers
def _stub():
    m = types.ModuleType("fyers_apiv3")
    m.fyersModel = types.ModuleType("fyers_apiv3.fyersModel")
    m.fyersModel.FyersModel = MagicMock
    sys.modules.setdefault("fyers_apiv3", m)
    sys.modules.setdefault("fyers_apiv3.fyersModel", m.fyersModel)
_stub()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_ohlcv(n=150, base=100.0, seed=42) -> pd.DataFrame:
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    c = base * np.cumprod(1 + np.random.normal(0.001, 0.012, n))
    h = c * (1 + np.abs(np.random.normal(0, 0.005, n)))
    l = c * (1 - np.abs(np.random.normal(0, 0.005, n)))
    o = c * (1 + np.random.normal(0, 0.004, n))
    v = np.random.randint(1_000_000, 3_000_000, n).astype(float)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=dates)


def _make_config():
    from core.config import SystemConfig
    return SystemConfig()


# ═════════════════════════════════════════════════════════════════════════════
# TELEMETRY
# ═════════════════════════════════════════════════════════════════════════════

class TestTelemetry:

    def test_setup_logging_idempotent(self):
        from core.telemetry import setup_logging
        setup_logging(level="DEBUG")
        setup_logging(level="DEBUG")  # second call is no-op
        log = logging.getLogger("sovereign")
        assert len(log.handlers) > 0

    def test_emit_does_not_raise(self):
        from core.telemetry import setup_logging, emit
        setup_logging()
        emit("test_event", value=42, tag="unit")

    def test_scan_metrics_record_fetch(self):
        from core.telemetry import ScanMetrics
        m = ScanMetrics()
        m.record_fetch(n_ok=100, n_fail=3, elapsed_s=2.5)
        assert m.fetch_n_ok == 100
        assert m.fetch_n_fail == 3
        assert m.fetch_elapsed == pytest.approx(2.5, abs=0.01)

    def test_scan_metrics_record_indicators(self):
        from core.telemetry import ScanMetrics
        m = ScanMetrics()
        m.record_indicators(n_ok=98, n_fail=2, elapsed_s=1.0)
        assert m.ind_n_ok == 98

    def test_scan_metrics_record_score(self):
        from core.telemetry import ScanMetrics
        m = ScanMetrics()
        m.record_score(n_passed=15, n_total=98, elapsed_s=0.8)
        assert m.score_n_passed == 15

    def test_scan_metrics_record_regime(self):
        from core.telemetry import ScanMetrics
        m = ScanMetrics()
        regime = MagicMock()
        regime.regime = "TREND_UP"
        regime.confidence = 0.75
        regime.breadth = 0.62
        m.record_regime(regime)
        assert m.regime == "TREND_UP"
        assert m.regime_conf == pytest.approx(0.75)

    def test_scan_metrics_record_portfolio(self):
        from core.telemetry import ScanMetrics
        m = ScanMetrics()
        r1, r2 = MagicMock(), MagicMock()
        r1.ticker = "RELIANCE"; r1.prob_win = 0.62; r1.expectancy_r = 0.3
        r2.ticker = "INFY";     r2.prob_win = 0.58; r2.expectancy_r = 0.2
        m.record_portfolio([r1, r2])
        assert m.portfolio_size == 2
        assert "RELIANCE" in m.portfolio_tickers

    def test_scan_metrics_emit_summary_no_crash(self):
        from core.telemetry import ScanMetrics, setup_logging
        setup_logging()
        m = ScanMetrics()
        m.record_fetch(100, 0, 1.0)
        m.record_score(10, 100, 0.5)
        m.record_portfolio([])
        m.emit_summary()  # should not raise


# ═════════════════════════════════════════════════════════════════════════════
# RETRY
# ═════════════════════════════════════════════════════════════════════════════

class TestRetryWithBackoff:

    def test_succeeds_on_first_try(self):
        from core.retry import retry_with_backoff
        result = retry_with_backoff(lambda: 42, max_attempts=3, base_delay=0.0)
        assert result == 42

    def test_retries_and_eventually_succeeds(self):
        from core.retry import retry_with_backoff
        calls = {"n": 0}
        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "ok"
        result = retry_with_backoff(fn, max_attempts=5, base_delay=0.0)
        assert result == "ok"
        assert calls["n"] == 3

    def test_raises_max_retries_exceeded(self):
        from core.retry import retry_with_backoff, MaxRetriesExceeded
        with pytest.raises(MaxRetriesExceeded):
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(IOError("always fails")),
                max_attempts=3, base_delay=0.0,
            )

    def test_non_retryable_propagates_immediately(self):
        from core.retry import retry_with_backoff
        calls = {"n": 0}
        def fn():
            calls["n"] += 1
            raise ValueError("not retryable")
        with pytest.raises(ValueError):
            retry_with_backoff(fn, max_attempts=5, base_delay=0.0,
                               retryable=(IOError,))
        assert calls["n"] == 1   # stopped immediately

    def test_respects_max_attempts(self):
        from core.retry import retry_with_backoff, MaxRetriesExceeded
        calls = {"n": 0}
        def fn():
            calls["n"] += 1
            raise IOError("fail")
        with pytest.raises(MaxRetriesExceeded):
            retry_with_backoff(fn, max_attempts=3, base_delay=0.0)
        assert calls["n"] == 3


class TestCircuitBreaker:

    def _breaker(self, threshold=3, timeout=0.05):
        from core.retry import CircuitBreaker
        return CircuitBreaker("test", failure_threshold=threshold,
                              reset_timeout=timeout)

    def test_starts_closed(self):
        from core.retry import BreakerState
        b = self._breaker()
        assert b.state == BreakerState.CLOSED

    def test_trips_after_threshold_failures(self):
        from core.retry import BreakerState
        b = self._breaker(threshold=3)
        for _ in range(3):
            try:
                b.call(lambda: (_ for _ in ()).throw(IOError()))
            except IOError:
                pass
        assert b.state == BreakerState.OPEN

    def test_open_breaker_raises_circuit_open(self):
        from core.retry import CircuitBreakerOpen, BreakerState
        b = self._breaker(threshold=1)
        try:
            b.call(lambda: (_ for _ in ()).throw(IOError()))
        except IOError:
            pass
        assert b.state == BreakerState.OPEN
        with pytest.raises(CircuitBreakerOpen):
            b.call(lambda: 42)

    def test_half_open_after_timeout(self):
        from core.retry import BreakerState
        b = self._breaker(threshold=1, timeout=0.05)
        try:
            b.call(lambda: (_ for _ in ()).throw(IOError()))
        except IOError:
            pass
        time.sleep(0.1)
        assert b.state == BreakerState.HALF_OPEN

    def test_closes_after_successful_probe(self):
        from core.retry import BreakerState
        b = self._breaker(threshold=1, timeout=0.05)
        try:
            b.call(lambda: (_ for _ in ()).throw(IOError()))
        except IOError:
            pass
        time.sleep(0.1)
        b.call(lambda: "ok")
        assert b.state == BreakerState.CLOSED

    def test_reset_force_closes(self):
        from core.retry import BreakerState
        b = self._breaker(threshold=1)
        try:
            b.call(lambda: (_ for _ in ()).throw(IOError()))
        except IOError:
            pass
        b.reset()
        assert b.state == BreakerState.CLOSED

    def test_successful_call_passes_through(self):
        b = self._breaker()
        result = b.call(lambda: 99)
        assert result == 99


class TestGuardedCall:

    def test_succeeds_no_breaker(self):
        from core.retry import guarded_call
        assert guarded_call(lambda: "hi", max_attempts=1) == "hi"

    def test_succeeds_with_breaker(self):
        from core.retry import guarded_call, CircuitBreaker
        b = CircuitBreaker("g", failure_threshold=5)
        result = guarded_call(lambda: 7, breaker=b, max_attempts=1)
        assert result == 7

    def test_breaker_trips_on_persistent_failure(self):
        from core.retry import guarded_call, CircuitBreaker, BreakerState, MaxRetriesExceeded
        b = CircuitBreaker("g2", failure_threshold=1)
        with pytest.raises(MaxRetriesExceeded):
            guarded_call(
                lambda: (_ for _ in ()).throw(IOError("fail")),
                breaker=b, max_attempts=1, base_delay=0.0,
            )
        assert b.state == BreakerState.OPEN


# ═════════════════════════════════════════════════════════════════════════════
# CACHE
# ═════════════════════════════════════════════════════════════════════════════

class TestScanCache:

    def _processed(self):
        """Minimal processed dict for corr matrix tests."""
        cfg = _make_config()
        dfs = {}
        for name in ["A.NS", "B.NS", "C.NS", cfg.BENCHMARK]:
            df = _make_ohlcv(seed=hash(name) % 1000)
            dfs[name] = df
        return dfs, cfg

    def test_volume_profile_returns_valid_tuple(self):
        from core.cache import ScanCache
        cache = ScanCache()
        df    = _make_ohlcv()
        poc, val, vah = cache.volume_profile(df, ticker="TEST")
        assert val <= poc <= vah

    def test_volume_profile_second_call_is_cache_hit(self):
        from core.cache import ScanCache
        cache = ScanCache()
        df    = _make_ohlcv()
        cache.volume_profile(df, ticker="TEST")
        assert cache.stats()["misses"] == 1
        cache.volume_profile(df, ticker="TEST")
        assert cache.stats()["hits"] == 1

    def test_volume_profile_invalidates_on_new_data(self):
        from core.cache import ScanCache
        cache = ScanCache()
        df1   = _make_ohlcv(n=100)
        df2   = _make_ohlcv(n=110)   # different last date
        cache.volume_profile(df1, ticker="T")
        cache.volume_profile(df2, ticker="T")
        assert cache.stats()["misses"] == 2

    def test_corr_matrix_returns_dataframe(self):
        from core.cache import ScanCache
        cache = ScanCache()
        processed, cfg = self._processed()
        corr = cache.corr_matrix(processed, cfg)
        assert isinstance(corr, pd.DataFrame)

    def test_corr_matrix_second_call_is_cache_hit(self):
        from core.cache import ScanCache
        cache = ScanCache()
        processed, cfg = self._processed()
        cache.corr_matrix(processed, cfg)
        m1 = cache.stats()["misses"]
        cache.corr_matrix(processed, cfg)
        assert cache.stats()["hits"] >= 1
        assert cache.stats()["misses"] == m1

    def test_clear_resets_everything(self):
        from core.cache import ScanCache
        cache = ScanCache()
        df    = _make_ohlcv()
        cache.volume_profile(df, ticker="X")
        cache.clear()
        s = cache.stats()
        assert s["vprofile_size"] == 0
        assert not s["corr_cached"]

    def test_stats_hit_rate_sensible(self):
        from core.cache import ScanCache
        cache = ScanCache()
        df    = _make_ohlcv()
        cache.volume_profile(df, ticker="Z")   # miss
        cache.volume_profile(df, ticker="Z")   # hit
        assert cache.stats()["hit_rate"] == pytest.approx(0.5, abs=0.01)

    def test_eviction_when_full(self):
        from core.cache import ScanCache
        cache = ScanCache(max_vprofile_entries=5)
        for i in range(8):
            df = _make_ohlcv(n=60 + i, seed=i)
            cache.volume_profile(df, ticker=f"T{i}")
        # Should not raise; size capped
        assert cache.stats()["vprofile_size"] <= 5


# ═════════════════════════════════════════════════════════════════════════════
# BACKTEST — helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestBacktestHelpers:

    def _fwd_bars(self, n=10, start=100.0, trend=0.0):
        dates = pd.date_range("2024-06-01", periods=n, freq="B")
        c = start * np.cumprod(1 + trend + np.random.normal(0, 0.005, n))
        h = c * 1.005
        l = c * 0.995
        return pd.DataFrame({"High": h, "Low": l, "Close": c}, index=dates)

    def test_realised_r_long_stop_hit(self):
        from core.backtest import _realised_r
        # Make a bar where Low < stop
        bars = self._fwd_bars(n=5, start=100.0)
        bars.iloc[1, bars.columns.get_loc("Low")] = 97.0   # hits stop
        r, hit, bars_held, _ = _realised_r("LONG", entry=100.0, stop=98.0,
                                            t1=110.0, fwd_bars=bars, time_stop=20)
        assert r == pytest.approx(-1.0)
        assert not hit
        assert bars_held == 2

    def test_realised_r_long_target_hit(self):
        from core.backtest import _realised_r
        bars = self._fwd_bars(n=5, start=100.0)
        bars.iloc[2, bars.columns.get_loc("High")] = 115.0   # hits T1
        r, hit, bars_held, _ = _realised_r("LONG", entry=100.0, stop=97.0,
                                            t1=110.0, fwd_bars=bars, time_stop=20)
        assert hit
        assert r > 0
        assert bars_held == 3

    def test_realised_r_time_stop(self):
        from core.backtest import _realised_r
        np.random.seed(1)
        bars = self._fwd_bars(n=10, start=100.0, trend=0.0)
        r, hit, bars_held, _ = _realised_r("LONG", entry=100.0, stop=80.0,
                                            t1=150.0, fwd_bars=bars, time_stop=5)
        assert bars_held <= 6   # capped by time stop (exit on bar i >= time_stop)
        assert not hit

    def test_realised_r_empty_bars(self):
        from core.backtest import _realised_r
        empty = pd.DataFrame(columns=["High", "Low", "Close"])
        r, hit, bars, _ = _realised_r("LONG", 100, 95, 110, empty, 10)
        assert r == 0.0
        assert bars == 0

    def test_fold_stats_empty_trades(self):
        from core.backtest import _fold_stats
        ts = pd.Timestamp("2024-01-01")
        fs = _fold_stats(0, [], (ts, ts))
        assert fs.n_trades == 0
        assert fs.hit_rate == 0.0

    def test_fold_stats_with_trades(self):
        from core.backtest import _fold_stats, TradeRecord
        ts = pd.Timestamp("2024-01-01")
        trades = [
            TradeRecord(0, "A", "LONG", ts, ts, 100, 97, 110, 0.7, 0.6, 2.5, True, 5),
            TradeRecord(0, "B", "LONG", ts, ts, 200, 195, 220, 0.6, 0.55, -1.0, False, 3),
        ]
        fs = _fold_stats(0, trades, (ts, ts))
        assert fs.n_trades == 2
        assert fs.hit_rate == pytest.approx(0.5)
        assert fs.total_r == pytest.approx(1.5)

    def test_overall_stats_profit_factor(self):
        from core.backtest import _overall_stats, _fold_stats, TradeRecord
        ts = pd.Timestamp("2024-01-01")
        trades = [
            TradeRecord(0, "A", "LONG", ts, ts, 100, 97, 110, 0.7, 0.6, 3.0, True, 5),
            TradeRecord(0, "B", "LONG", ts, ts, 200, 195, 220, 0.6, 0.55, -1.0, False, 3),
        ]
        overall = _overall_stats(trades, [_fold_stats(0, trades, (ts, ts))])
        assert overall.profit_factor == pytest.approx(3.0, abs=0.1)
        assert overall.n_trades == 2

    def test_walk_forward_no_crash_small_universe(self):
        """Smoke test: walk_forward runs on a tiny synthetic universe."""
        from core.backtest import walk_forward
        cfg = _make_config()
        cfg2 = type(cfg)(**{**cfg.__dict__,
                            "BACKTEST_DAYS": 30,
                            "BACKTEST_MIN_PROB": 0.0,   # pass all trades
                            "MIN_PROB_WIN": 0.0,
                            "MIN_EXPECTANCY_R": -9.0,
                            "BENCHMARK": "BENCH"})

        raw = {
            "AAA.NS": _make_ohlcv(n=250, seed=1),
            "BBB.NS": _make_ohlcv(n=250, seed=2),
            "BENCH":  _make_ohlcv(n=250, seed=99),
        }

        result = walk_forward(
            raw_data=raw,
            config=cfg2,
            train_days=80,
            test_days=10,
            step_days=20,
            max_trades_per_fold=5,
        )
        # Should complete without raising; may have 0 trades if no signals
        assert result.overall.n_folds >= 0
        assert isinstance(result.to_dataframe(), pd.DataFrame)

    def test_walk_forward_missing_benchmark(self):
        from core.backtest import walk_forward
        cfg = _make_config()
        result = walk_forward({"AAA.NS": _make_ohlcv()}, cfg)
        assert result.overall.n_trades == 0

    def test_walk_forward_insufficient_history(self):
        from core.backtest import walk_forward
        cfg = _make_config()
        cfg2 = type(cfg)(**{**cfg.__dict__, "BENCHMARK": "BENCH"})
        raw = {"AAA.NS": _make_ohlcv(n=20), "BENCH": _make_ohlcv(n=20)}
        result = walk_forward(raw, cfg2, train_days=100, test_days=20)
        assert result.overall.n_trades == 0
