"""
tests/test_worker.py
====================
Unit tests for core/worker.py (Scan worker and dispatching abstraction).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from core.worker import (
    ScanJob,
    ScanResultPayload,
    LocalThreadDispatcher,
    execute_scan_task,
)
from core.scorer import TickerResult


def test_scan_job_and_payload_defaults() -> None:
    job = ScanJob(tickers=["RELIANCE.NS"])
    assert len(job.job_id) > 0
    assert job.tickers == ["RELIANCE.NS"]
    assert job.created_at is not None

    payload = ScanResultPayload(job_id=job.job_id, status="completed")
    assert payload.portfolio == []
    assert payload.candidates == []
    assert payload.error is None


def test_execute_scan_task_success() -> None:
    dummy_res = MagicMock(spec=TickerResult)
    dummy_res.ticker = "TCS.NS"
    dummy_res.shares = 100
    dummy_res.__dict__ = {
        "ticker": "TCS.NS",
        "prob_win": 0.65,
        "expectancy_r": 0.8,
        "rr_t1": 1.5,
        "entry": 3500.0,
        "stop": 3450.0,
        "t1": 3575.0,
        "direction": "LONG",
        "composite": 1.2,
        "shares": 100,
        "risk_inr": 5000.0,
    }

    with patch("screener_v14_modular.run_scan", return_value=([dummy_res], [dummy_res], None)):
        job = ScanJob(tickers=["TCS.NS"])
        result = execute_scan_task(job)

        assert result.status == "completed"
        assert len(result.portfolio) == 1
        assert result.portfolio[0]["ticker"] == "TCS.NS"
        assert result.portfolio[0]["shares"] == 100


def test_execute_scan_task_failure_handles_cleanly() -> None:
    with patch("screener_v14_modular.run_scan", side_effect=RuntimeError("Data stream offline")):
        job = ScanJob()
        result = execute_scan_task(job)

        assert result.status == "failed"
        assert "Data stream offline" in (result.error or "")


def test_local_thread_dispatcher_submit() -> None:
    dispatcher = LocalThreadDispatcher(max_workers=1)
    try:
        dummy_res = MagicMock(spec=TickerResult)
        dummy_res.ticker = "INFY.NS"
        dummy_res.__dict__ = {
            "ticker": "INFY.NS",
            "prob_win": 0.7,
            "expectancy_r": 1.0,
            "rr_t1": 2.0,
            "entry": 1600.0,
            "stop": 1550.0,
            "t1": 1700.0,
            "direction": "LONG",
            "composite": 0.9,
            "shares": 50,
            "risk_inr": 2500.0,
        }

        with patch("screener_v14_modular.run_scan", return_value=([], [dummy_res], None)):
            job = ScanJob(tickers=["INFY.NS"])
            future = dispatcher.submit(job)
            res = future.result(timeout=5)

            assert res.status == "completed"
            assert len(res.portfolio) == 1
            assert res.portfolio[0]["ticker"] == "INFY.NS"
            assert len(res.candidates) == 0
    finally:
        dispatcher.shutdown(wait=True)


def test_execute_scan_task_distinct_candidates_and_portfolio_regression() -> None:
    candidate_res = MagicMock(spec=TickerResult)
    candidate_res.ticker = "CANDIDATE.NS"
    candidate_res.__dict__ = {
        "ticker": "CANDIDATE.NS",
        "shares": 10,
        "prob_win": 0.55,
    }

    portfolio_res = MagicMock(spec=TickerResult)
    portfolio_res.ticker = "PORTFOLIO.NS"
    portfolio_res.__dict__ = {
        "ticker": "PORTFOLIO.NS",
        "shares": 50,
        "prob_win": 0.75,
    }

    # run_scan returns (candidates, portfolio, regime)
    with patch("screener_v14_modular.run_scan", return_value=([candidate_res], [portfolio_res], None)):
        job = ScanJob()
        result = execute_scan_task(job)

        assert result.status == "completed"
        assert len(result.candidates) == 1
        assert result.candidates[0]["ticker"] == "CANDIDATE.NS"
        assert len(result.portfolio) == 1
        assert result.portfolio[0]["ticker"] == "PORTFOLIO.NS"
