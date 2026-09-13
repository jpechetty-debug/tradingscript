"""
core/worker.py
==============
Scan worker and dispatching abstraction for distributed and asynchronous execution.

Provides decoupled execution options:
- LocalThreadDispatcher: In-process multi-threaded execution (default)
- ProcessPoolDispatcher: Multi-core process pool execution for CPU-heavy scans
- QueueDispatcher / Worker readiness interface for Redis/Celery/RQ
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
import uuid

from .scorer import TickerResult

log = logging.getLogger("sovereign.worker")


@dataclass
class ScanJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tickers: Optional[List[str]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScanResultPayload:
    job_id: str
    status: str  # "completed" | "failed"
    portfolio: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    regime_info: Optional[Dict[str, Any]] = None
    sector_rs: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _format_ticker(res: TickerResult) -> Dict[str, Any]:
    d = res.__dict__.copy()
    if hasattr(res, "factors") and res.factors is not None:
        d["factors"] = res.factors.__dict__.copy()
    return d


def execute_scan_task(job: ScanJob) -> ScanResultPayload:
    """Standalone worker task entry point capable of running in separate process/worker."""
    import screener_v14_modular as svm

    try:
        log.info(f"Worker executing ScanJob {job.job_id}...")
        candidates, portfolio, regime = svm.run_scan()
        regime_info = regime.__dict__.copy() if regime else None
        sector_rs: Dict[str, float] = {}

        if hasattr(svm, "_DEFAULT_SERVICES") and svm._DEFAULT_SERVICES is not None:
            srv = svm._DEFAULT_SERVICES.scan_service
            last_info = getattr(srv, "last_regime_info", None)
            if last_info and not regime_info:
                regime_info = last_info.__dict__.copy()
            sector_rs = getattr(srv, "last_sector_rs", {})

        return ScanResultPayload(
            job_id=job.job_id,
            status="completed",
            portfolio=[_format_ticker(p) for p in portfolio],
            candidates=[_format_ticker(c) for c in candidates],
            regime_info=regime_info,
            sector_rs=sector_rs,
        )
    except Exception as exc:
        log.exception(f"ScanJob {job.job_id} failed: {exc}")
        return ScanResultPayload(
            job_id=job.job_id,
            status="failed",
            error=str(exc),
        )


class BaseScanDispatcher(ABC):
    """Abstract dispatcher contract for submitting scan jobs."""

    @abstractmethod
    def submit(self, job: ScanJob) -> concurrent.futures.Future[ScanResultPayload]:
        """Submit job for execution, returning a Future."""
        pass

    @abstractmethod
    def shutdown(self, wait: bool = True) -> None:
        """Cleanly shutdown worker pools."""
        pass


class LocalThreadDispatcher(BaseScanDispatcher):
    """Thread pool dispatcher for lightweight asynchronous in-process execution."""

    def __init__(self, max_workers: int = 2) -> None:
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="scan_worker"
        )

    def submit(self, job: ScanJob) -> concurrent.futures.Future[ScanResultPayload]:
        return self.executor.submit(execute_scan_task, job)

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)


class ProcessPoolDispatcher(BaseScanDispatcher):
    """Process pool dispatcher for multi-core scale-out."""

    def __init__(self, max_workers: int = 2) -> None:
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        )

    def submit(self, job: ScanJob) -> concurrent.futures.Future[ScanResultPayload]:
        return self.executor.submit(execute_scan_task, job)

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)
