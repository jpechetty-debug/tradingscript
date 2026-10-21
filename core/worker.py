"""
core/worker.py
==============
Versioned scan jobs for local threads and process workers.
Execution inputs travel with the job; metadata is descriptive provenance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import concurrent.futures
from copy import deepcopy
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, cast
import uuid

from .application import build_application_runtime
from .config import CONFIG, MarketRegimeType, SystemConfig
from .regime import RegimeTracker
from .runtime_paths import RuntimePaths
from .services import CancellationProbe, PersistenceService, ScanCancelled, ServiceBundle
from .snapshots import build_scan_snapshot
from .universe import resolve_scan_tickers

log = logging.getLogger("sovereign.worker")


@dataclass
class ScanOptions:
    debug: bool = False
    no_intraday: bool = False
    regime_override: str | None = None
    no_ema_filter: bool = False
    force_score: bool = False

    def validate(self) -> None:
        for name in ("debug", "no_intraday", "no_ema_filter", "force_score"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.regime_override is not None and self.regime_override not in MarketRegimeType.__members__:
            raise ValueError("Unknown regime override")


def _copy_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(metadata, dict) or any(not isinstance(key, str) for key in metadata):
        raise ValueError("metadata must be a JSON object with string keys")
    encoded = json.dumps(metadata, allow_nan=False)
    pending: list[Any] = [metadata]
    while pending:
        item = pending.pop()
        if isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("metadata object keys must be strings")
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
        elif item is not None and not isinstance(item, (str, bool, int, float)):
            raise ValueError("metadata must contain only JSON values")
    return cast(Dict[str, Any], json.loads(encoded))


@dataclass
class ScanJob:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tickers: Optional[List[str]] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1
    config: SystemConfig = field(default_factory=lambda: deepcopy(CONFIG), repr=False)
    paths: RuntimePaths = field(default_factory=RuntimePaths.discover)
    options: ScanOptions = field(default_factory=ScanOptions)
    regime_tracker: RegimeTracker = field(default_factory=RegimeTracker, repr=False)

    def __post_init__(self) -> None:
        self.config = deepcopy(self.config)
        self.tickers = resolve_scan_tickers(deepcopy(self.tickers), self.config.BENCHMARK)
        self.options = deepcopy(self.options)
        self.regime_tracker = deepcopy(self.regime_tracker)
        self.metadata = _copy_metadata(self.metadata)
        # Resolve at submission-side construction, never against a worker's cwd.
        self.paths = RuntimePaths(**{
            item.name: Path(getattr(self.paths, item.name)).resolve() for item in fields(RuntimePaths)
        })


@dataclass
class ScanResultPayload:
    job_id: str
    status: str  # "completed" | "failed" | "cancelled"
    portfolio: List[Dict[str, Any]] = field(default_factory=list)
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    regime_info: Optional[Dict[str, Any]] = None
    sector_rs: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    completed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    regime_tracker: RegimeTracker | None = field(default=None, repr=False)


def _execution_context(job: ScanJob, tickers: list[str]) -> Dict[str, Any]:
    # Credentials are neither serialized into results nor used as strategy identity.
    config = {key: value for key, value in asdict(job.config).items()
              if key != "settings_payload" and not key.startswith(("TELEGRAM_", "FYERS_"))}
    reference = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return {
        "schema_version": job.schema_version, "created_at": job.created_at,
        "config_reference": f"sha256:{reference}", "tickers": list(tickers),
        "paths": {item.name: str(getattr(job.paths, item.name)) for item in fields(RuntimePaths)},
        "options": asdict(job.options),
    }


def execute_scan_task(
    job: ScanJob, *, services: ServiceBundle | None = None,
    regime_tracker: RegimeTracker | None = None,
    cancel_requested: CancellationProbe | None = None,
) -> ScanResultPayload:
    """Execute a job with an owned runtime, or borrow an explicitly supplied local runtime."""
    import screener_v14_modular as svm

    metadata: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    tracker: RegimeTracker | None = None
    try:
        job = deepcopy(job)
        metadata = _copy_metadata(job.metadata)
        if not isinstance(job.schema_version, int) or isinstance(job.schema_version, bool) or job.schema_version != 1:
            raise ValueError("Unsupported ScanJob schema version")
        job.options.validate()
        universe = resolve_scan_tickers(job.tickers, job.config.BENCHMARK)
        if any(not getattr(job.paths, item.name).is_absolute() for item in fields(RuntimePaths)):
            raise ValueError("ScanJob runtime paths must be absolute")
        context = _execution_context(job, universe)
        context["engine_version"] = svm.VERSION
        tracker = regime_tracker if regime_tracker is not None else job.regime_tracker
        if cancel_requested is not None and cancel_requested():
            raise ScanCancelled("ScanJob cancelled before execution")
        if services is not None and services.persistence.paths != job.paths:
            raise ValueError("ScanJob paths do not match the supplied runtime")
        owned_runtime = None if services is not None else build_application_runtime(
            version=svm.VERSION, persistence=PersistenceService(paths=job.paths),
        )
        try:
            if services is not None:
                bundle = services
            else:
                assert owned_runtime is not None
                bundle = owned_runtime.services
            log.info("Worker executing ScanJob %s", job.job_id)
            output = svm.run_scan(
                config=job.config, tickers=universe, services=bundle,
                regime_tracker=tracker, cancel_requested=cancel_requested, **asdict(job.options),
            )
            if cancel_requested is not None and cancel_requested():
                raise ScanCancelled("ScanJob cancelled during execution")
            candidates, portfolio, regime = output
            # ScanOutput carries per-call sector data, including an explicitly empty result.
            sector_rs = getattr(output, "sector_rs", {})
            snapshot = build_scan_snapshot(
                scan_time=datetime.now(timezone.utc).isoformat(), candidates=candidates,
                portfolio=portfolio, sector_rs=sector_rs,
                regime_info=regime.__dict__.copy() if regime else None, config=job.config,
            )
            context["snapshot_schema_version"] = snapshot["schema_version"]
            payload = ScanResultPayload(
                job_id=job.job_id, status="completed", portfolio=snapshot["portfolio"],
                candidates=snapshot["candidates"], regime_info=snapshot["regime_info"],
                sector_rs=snapshot["sector_rs"], metadata=metadata, context=context,
                regime_tracker=deepcopy(tracker),
            )
        finally:
            if owned_runtime is not None:
                owned_runtime.close()
        return payload
    except Exception as exc:
        log.exception("ScanJob %s failed", job.job_id)
        return ScanResultPayload(
            job_id=job.job_id,
            status="cancelled" if isinstance(exc, ScanCancelled) else "failed",
            error=str(exc), metadata=metadata, context=context, regime_tracker=deepcopy(tracker),
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

    def __init__(
        self, max_workers: int = 2, *, services: ServiceBundle | None = None,
        regime_tracker: RegimeTracker | None = None,
        cancel_requested: CancellationProbe | None = None,
    ) -> None:
        self._services = services
        self._regime_tracker = regime_tracker
        self._cancel_requested = cancel_requested
        self._runtime_lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="scan_worker"
        )

    def submit(self, job: ScanJob) -> concurrent.futures.Future[ScanResultPayload]:
        return self.executor.submit(self._execute, deepcopy(job))

    def _execute(self, job: ScanJob) -> ScanResultPayload:
        # A borrowed scan service/tracker has mutable state and cannot run overlapping jobs.
        with self._runtime_lock:
            return execute_scan_task(
                job, services=self._services, regime_tracker=self._regime_tracker,
                cancel_requested=self._cancel_requested,
            )

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)


class ProcessPoolDispatcher(BaseScanDispatcher):
    """Process pool dispatcher for multi-core scale-out."""

    def __init__(self, max_workers: int = 2) -> None:
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        )

    def submit(self, job: ScanJob) -> concurrent.futures.Future[ScanResultPayload]:
        return self.executor.submit(execute_scan_task, deepcopy(job))

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait=wait)
