"""
server.py
=========
Institutional FastAPI web server & real-time REST API for Sovereign Engine v14.

Endpoints:
    GET  /                     -> Serve dynamic dashboard.html
    GET  /api/status           -> Engine status, market session, current regime & lock state
    GET  /api/scan             -> Cached/latest scan results (portfolio picks & candidates)
    POST /api/scan/trigger     -> Asynchronously trigger a fresh market scan
    POST /api/regime/override  -> Set or clear manual market regime override
    GET  /api/sectors          -> Sector relative strength & concentration breakdown
    GET  /api/config           -> System configuration settings
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import ipaddress
import json
import logging
import os
import sqlite3
from pathlib import Path
import threading
from typing import Any, AsyncGenerator, AsyncIterator, Dict, List, Optional

import secrets
from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import uvicorn

import screener_v14_modular as svm
from core.application import ApplicationRuntime, build_application_runtime
from core.config import CONFIG, MarketRegimeType
from core.regime import RegimeTracker
from core.runtime_paths import RuntimePaths
from core.scorer import TickerResult
from core.services import PersistenceService, ScanCancelled
from core.snapshots import build_scan_snapshot, format_ticker, write_json_atomic
from core.universe import SECTORS, TICKER_TO_SECTOR
from core.trades import TradeExit, TradeFill, TradeLifecycle

load_dotenv()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

INSECURE_PLACEHOLDER_KEYS = frozenset({
    "your_secure_api_key_here",
    "your_api_key_here",
    "sovereign-dev-secret-key",
    "change-me",
    "secret",
    "password",
    "123456",
})

API_KEY = os.environ.get("API_KEY")

def verify_api_key(api_key: str = Security(api_key_header)) -> None:
    current_key = os.environ.get("API_KEY") or API_KEY
    if not current_key or not api_key or not secrets.compare_digest(api_key, current_key):
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )


def get_client_ip(request: Request) -> str:
    """
    Extract client IP for rate limiting, with support for trusted reverse proxies.

    Forwarded headers are accepted only when the direct peer belongs to a
    network explicitly listed in the comma-separated TRUSTED_PROXIES setting.
    """
    trusted = os.environ.get("TRUSTED_PROXIES", "").strip()
    peer = get_remote_address(request)
    networks = []
    for value in (part.strip() for part in trusted.split(",")):
        if not value:
            continue
        try:
            networks.append(ipaddress.ip_network(value, strict=False))
        except ValueError:
            log.warning("Ignoring invalid TRUSTED_PROXIES entry: %r", value)
    try:
        peer_is_trusted = any(ipaddress.ip_address(peer) in network for network in networks)
    except ValueError:
        peer_is_trusted = False

    if peer_is_trusted:
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
            if parts:
                return parts[0]
        real_ip = request.headers.get("x-real-ip")
        if real_ip and real_ip.strip():
            return real_ip.strip()

    return peer


log = logging.getLogger("sovereign.server")



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Validate server security config and trigger initial non-blocking market scan."""
    app.state.broadcaster.set_loop(asyncio.get_running_loop())
    current_key = os.environ.get("API_KEY") or API_KEY
    if not current_key:
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: 'API_KEY' environment variable must be set. "
            "Please configure API_KEY in your .env or environment variables."
        )

    key_clean = current_key.lower().strip()
    if key_clean == "your_secure_api_key_here":
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: 'API_KEY' is set to the unedited template placeholder "
            "('your_secure_api_key_here') from .env.example. Generate a unique, secure API_KEY in .env."
        )

    host = os.environ.get("HOST", "127.0.0.1").lower().strip()
    if host not in ("127.0.0.1", "localhost") and key_clean in INSECURE_PLACEHOLDER_KEYS:
        raise RuntimeError(
            f"CRITICAL SECURITY CONFIGURATION ERROR: 'API_KEY' is set to an insecure default placeholder "
            f"({current_key!r}) while bound to non-loopback host {host!r}. Server startup refused."
        )

    if key_clean in INSECURE_PLACEHOLDER_KEYS:
        log.warning(
            "SECURITY WARNING: 'API_KEY' is using a development placeholder (%r). "
            "Permitted on loopback only. Set a strong key before exposing to any network.",
            current_key,
        )
    if app.state.persistence is None:
        app.state.persistence = PersistenceService(paths=app.state.paths or RuntimePaths.discover())
    # Re-hydrate state from disk and verify persistent killswitch flag
    app.state.engine.load_persisted_state(_persistence(app))
    runtime = build_application_runtime(version=svm.VERSION, persistence=_persistence(app))
    app.state.runtime = runtime

    # Trigger initial scan in non-blocking background task unless killswitch is active
    scan_task = None
    if app.state.auto_scan and not app.state.engine.is_killed:
        scan_task = _start_scan_task(app)
    try:
        yield
    finally:
        app.state.scan_cancel_event.set()
        active_task = app.state.scan_task or scan_task
        if active_task and not active_task.done():
            try:
                await asyncio.wait_for(active_task, timeout=10.0)
            except (asyncio.TimeoutError, ScanCancelled):
                log.warning("Scan did not stop before server shutdown timeout.")
        runtime.close()
        app.state.runtime = None


router = APIRouter()

class AsyncEngineLock(asyncio.Lock):
    """Asyncio lock that also supports synchronous context manager protocol in tests."""
    def __enter__(self) -> "AsyncEngineLock":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        pass


# State owned by each application instance
class EngineState:
    def __init__(self) -> None:
        self.regime_tracker = RegimeTracker()
        self.regime_override: Optional[str] = None
        self.last_scan_time: Optional[str] = None
        self.last_portfolio: List[Dict[str, Any]] = []
        self.last_candidates: List[Dict[str, Any]] = []
        self.last_regime_info: Optional[Dict[str, Any]] = None
        self.last_sector_rs: Dict[str, float] = {}
        self.is_scanning: bool = False
        self.scan_error: Optional[str] = None
        self.is_killed: bool = False
        self._lock = AsyncEngineLock()

    def persist_state(self, persistence: PersistenceService) -> None:
        """Persist current scan results to disk (state/latest_scan.json)."""
        try:
            target = persistence.paths.state_dir / "latest_scan.json"
            payload = build_scan_snapshot(
                scan_time=self.last_scan_time,
                candidates=self.last_candidates,
                portfolio=self.last_portfolio,
                sector_rs=self.last_sector_rs,
                regime_info=self.last_regime_info,
                config=CONFIG,
            )
            write_json_atomic(target, payload)
            log.info("Persisted latest scan snapshot to %s", target)
        except (OSError, TypeError, ValueError) as exc:
            log.warning("Could not persist latest scan state: %s", exc)

    def load_persisted_state(self, persistence: PersistenceService) -> bool:
        """Re-hydrate state from state/latest_scan.json if it exists."""
        target = persistence.paths.state_dir / "latest_scan.json"
        try:
            self.is_killed = persistence.get_killswitch()
        except (sqlite3.Error, OSError) as exc:
            self.is_killed = True
            self.scan_error = "Could not verify persisted killswitch; scans blocked"
            log.error("Could not restore killswitch: %s", exc)
            raise RuntimeError(self.scan_error) from exc
        try:
            if not target.exists():
                return False
            data = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Scan snapshot must be an object")
            raw_cands = data.get("candidates") or []
            raw_port = data.get("portfolio") or []
            if not isinstance(raw_cands, list) or not isinstance(raw_port, list) or any(
                    not isinstance(row, dict) for row in [*raw_cands, *raw_port]):
                raise ValueError("Scan candidates and portfolio must contain objects")
            candidates = [format_ticker(c, CONFIG) for c in raw_cands]
            portfolio = [format_ticker(p, CONFIG) for p in raw_port]
            self.last_scan_time = data.get("scan_time")
            self.last_candidates = candidates
            self.last_portfolio = portfolio
            self.last_sector_rs = data.get("sector_rs") or {}
            self.last_regime_info = data.get("regime_info")
            log.info("Re-hydrated EngineState from %s (%d candidates, %d portfolio)",
                     target, len(self.last_candidates), len(self.last_portfolio))
            return True
        except (OSError, ValueError, TypeError, KeyError) as exc:
            log.warning("Failed to re-hydrate state from %s: %s", target, exc)
            return False




MAX_SSE_SUBSCRIBERS = int(os.environ.get("MAX_SSE_SUBSCRIBERS", "50"))


class SSEBroadcaster:
    """Thread-safe event broadcaster for Server-Sent Events (SSE)."""

    def __init__(self, max_subscribers: int = MAX_SSE_SUBSCRIBERS) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self.max_subscribers = max_subscribers

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def subscribe(self) -> asyncio.Queue:
        if self._loop is None:
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
        async with self._lock:
            if len(self._subscribers) >= self.max_subscribers:
                raise HTTPException(
                    status_code=503,
                    detail=f"Max SSE subscribers reached ({self.max_subscribers}). Please retry later.",
                )
            queue: asyncio.Queue = asyncio.Queue(maxsize=100)
            self._subscribers.add(queue)
            return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    def _push_to_queues(self, message: dict[str, Any]) -> None:
        for q in list(self._subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(message)
                except asyncio.QueueEmpty:
                    pass

    async def broadcast(self, event: str, data: dict[str, Any]) -> None:
        message = {
            "event": event,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        current_loop = None
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        if (
            self._loop is not None
            and not self._loop.is_closed()
            and current_loop is not None
            and current_loop != self._loop
        ):
            self._loop.call_soon_threadsafe(self._push_to_queues, message)
        else:
            if current_loop is not None and (self._loop is None or self._loop.is_closed()):
                self._loop = current_loop
            self._push_to_queues(message)



BASE_DIR = Path(__file__).parent.resolve()
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"


def _format_ticker_result(res: TickerResult) -> Dict[str, Any]:
    return format_ticker(res, CONFIG)



async def _run_scan_task_async(application: FastAPI | None = None, *, claimed: bool = False) -> None:
    application = application or app
    state: EngineState = application.state.engine
    persistence = _persistence(application)
    broadcaster: SSEBroadcaster = application.state.broadcaster
    cancel_event: threading.Event = application.state.scan_cancel_event
    runtime: ApplicationRuntime | None = application.state.runtime
    if runtime is None:
        raise RuntimeError("Engine runtime has not started")
    if not claimed:
        async with state._lock:
            if state.is_killed:
                log.warning("Scan aborted: Emergency Killswitch is active.")
                return
            if state.is_scanning:
                return
            cancel_event.clear()
            state.is_scanning = True
            state.scan_error = None

    try:
        await broadcaster.broadcast("scan_started", {"timestamp": datetime.now(timezone.utc).isoformat()})
        scan_output = await asyncio.to_thread(
            svm.run_scan,
            config=CONFIG,
            regime_tracker=state.regime_tracker,
            regime_override=state.regime_override,
            services=runtime.services,
            cancel_requested=cancel_event.is_set,
        )
        if cancel_event.is_set() or state.is_killed:
            raise ScanCancelled("scan cancelled by emergency killswitch")
        candidates, portfolio, regime = scan_output[0], scan_output[1], scan_output[2]

        async with state._lock:
            state.last_scan_time = datetime.now(timezone.utc).isoformat()
            state.last_candidates = [_format_ticker_result(c) for c in candidates]
            state.last_portfolio = [_format_ticker_result(p) for p in portfolio]
            state.last_sector_rs = getattr(scan_output, "sector_rs", None) or svm.get_last_sector_rs(services=runtime.services)
            if regime:
                state.last_regime_info = {
                    "regime": regime.regime.value if isinstance(regime.regime, MarketRegimeType) else str(regime.regime),
                    "breadth": regime.breadth,
                    "adx_median": regime.adx_median,
                    "atr_ratio": regime.atr_ratio,
                    "confidence": regime.confidence,
                    "confirmed": regime.confirmed,
                    "breadth_delta": regime.breadth_delta,
                    "regime_locked": regime.regime_locked,
                    "sector_concentration": regime.sector_concentration,
                    "label": regime.label,
                    "strategy_hint": regime.strategy_hint(),
                    "is_tradeable": regime.is_tradeable(),
                }
            state.is_scanning = False

        # Route disk I/O off the event loop without holding the state lock
        await asyncio.to_thread(state.persist_state, persistence)

        await broadcaster.broadcast("scan_completed", {
            "candidates_count": len(state.last_candidates),
            "portfolio_count": len(state.last_portfolio),
            "regime": state.last_regime_info,
        })
    except ScanCancelled as exc:
        log.warning("Scan cancelled: %s", exc)
        async with state._lock:
            state.is_scanning = False
            state.scan_error = str(exc)
        await broadcaster.broadcast("scan_cancelled", {"error": str(exc)})
    except Exception as exc:
        log.exception("Error executing scan task")
        async with state._lock:
            state.is_scanning = False
            state.scan_error = str(exc)
        await broadcaster.broadcast("scan_failed", {"error": str(exc)})
    finally:
        if getattr(application.state, "scan_task", None) is asyncio.current_task():
            application.state.scan_task = None


def _start_scan_task(application: FastAPI, *, claimed: bool = False) -> asyncio.Task[None]:
    task = asyncio.create_task(_run_scan_task_async(application, claimed=claimed))
    application.state.scan_task = task
    return task


def _run_scan_task() -> None:
    """Synchronous bridge for running scan task in workers or tests."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, _run_scan_task_async()).result()
    else:
        asyncio.run(_run_scan_task_async())


# --- Request/Response Models ---

class OverrideRequest(BaseModel):
    regime: Optional[str] = Field(
        None, description="Regime type (PANIC, TREND_UP, TREND_DOWN, RANGE, EXPANSION) or null/CLEAR"
    )


# --- API Routes ---

@router.post("/api/trades/fills", dependencies=[Depends(verify_api_key)])
def register_trade_fill(request: Request, req: TradeFill) -> Dict[str, Any]:
    """Record a confirmed or paper fill; never submit a broker order."""
    try:
        return TradeLifecycle(_persistence(request.app).db).register(req)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/trades/{trade_id}/close", dependencies=[Depends(verify_api_key)])
def close_trade_fill(request: Request, trade_id: str, req: TradeExit) -> Dict[str, Any]:
    try:
        return TradeLifecycle(_persistence(request.app).db).close(trade_id, req)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/api/trades/executed", dependencies=[Depends(verify_api_key)])
def get_executed_trades(request: Request, limit: int = Query(default=100, ge=1, le=1000)) -> Dict[str, Any]:
    return {"trades": _persistence(request.app).db.fetch_executed_trades(limit=limit)}


@router.get("/api/positions", dependencies=[Depends(verify_api_key)])
def get_open_positions(request: Request) -> Dict[str, Any]:
    return {"positions": _persistence(request.app).load_open_positions()}


@router.get("/api/paper/events", dependencies=[Depends(verify_api_key)])
def get_paper_events(request: Request, after: int = Query(default=0, ge=0), limit: int = Query(default=1000, ge=1, le=1000)) -> Dict[str, Any]:
    events = _persistence(request.app).db.fetch_paper_events(after=after, limit=limit)
    return {"events": events, "next_after": events[-1]["sequence"] if events else after}

@router.get("/")
def read_root() -> FileResponse:
    dist_index = FRONTEND_DIST_DIR / "index.html"
    if dist_index.exists():
        return FileResponse(str(dist_index), media_type="text/html")
    dashboard_path = BASE_DIR / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(str(dashboard_path), media_type="text/html")


@router.get("/api/status", dependencies=[Depends(verify_api_key)])
async def get_status(request: Request) -> Dict[str, Any]:
    rg_settings = CONFIG.as_regime()
    session = rg_settings.session_from_time()
    locked = rg_settings.is_regime_locked()
    phase = rg_settings.get_market_phase()
    mins_to_sq = rg_settings.minutes_to_squareoff()
    phase_val = phase.value if hasattr(phase, "value") else str(phase)
    cutoff_active = phase_val in ("INTRADAY_FREEZE", "SWING_CLOSING", "POST_MARKET")
    platt_a, platt_b, platt_calibrated = _persistence(request.app).load_platt(CONFIG)
    async with request.app.state.engine._lock:
        return {
            "status": "online",
            "version": svm.VERSION,
            "session": session,
            "market_phase": phase_val,
            "minutes_to_squareoff": mins_to_sq,
            "intraday_cutoff_active": cutoff_active,
            "regime_locked": locked,
            "regime_override": request.app.state.engine.regime_override,
            "is_scanning": request.app.state.engine.is_scanning,
            "last_scan_time": request.app.state.engine.last_scan_time,
            "scan_error": request.app.state.engine.scan_error,
            "last_known_regime": request.app.state.engine.last_regime_info,
            "killswitch_active": request.app.state.engine.is_killed,
            "platt_calibration": {
                "a": platt_a,
                "b": platt_b,
                "is_calibrated": platt_calibrated,
                "min_trades_required": 80,
            },
        }


@router.get("/api/scan", dependencies=[Depends(verify_api_key)])
async def get_scan_results(request: Request) -> Dict[str, Any]:
    rg_settings = CONFIG.as_regime()
    phase = rg_settings.get_market_phase()
    mins_to_sq = rg_settings.minutes_to_squareoff()
    phase_val = phase.value if hasattr(phase, "value") else str(phase)
    cutoff_active = phase_val in ("INTRADAY_FREEZE", "SWING_CLOSING", "POST_MARKET")
    async with request.app.state.engine._lock:
        return {
            "last_scan_time": request.app.state.engine.last_scan_time,
            "is_scanning": request.app.state.engine.is_scanning,
            "scan_error": request.app.state.engine.scan_error,
            "market_phase": phase_val,
            "minutes_to_squareoff": mins_to_sq,
            "intraday_cutoff_active": cutoff_active,
            "regime": request.app.state.engine.last_regime_info,
            "portfolio": request.app.state.engine.last_portfolio,
            "candidates_count": len(request.app.state.engine.last_candidates),
            "candidates": request.app.state.engine.last_candidates,
        }


@router.post("/api/scan/trigger", dependencies=[Depends(verify_api_key)])
async def trigger_scan(request: Request) -> Dict[str, Any]:
    if request.app.state.runtime is None:
        raise HTTPException(status_code=503, detail="Engine startup has not completed")
    async with request.app.state.engine._lock:
        if request.app.state.engine.is_killed:
            raise HTTPException(
                status_code=403,
                detail="Emergency Killswitch is active. Clear killswitch before initiating scans.",
            )
        if request.app.state.engine.is_scanning:
            return {"status": "already_running", "message": "Scan execution already in progress"}
        request.app.state.scan_cancel_event.clear()
        request.app.state.engine.is_scanning = True
        request.app.state.engine.scan_error = None

    _start_scan_task(request.app, claimed=True)
    return {"status": "triggered", "message": "Market scan started in background worker"}


@router.post("/api/regime/override", dependencies=[Depends(verify_api_key)])
async def set_regime_override(request: Request, req: OverrideRequest) -> Dict[str, Any]:
    valid_regimes = {"PANIC", "TREND_UP", "TREND_DOWN", "RANGE", "EXPANSION"}
    async with request.app.state.engine._lock:
        if req.regime is None or req.regime.upper() in ("CLEAR", "NONE", "AUTO"):
            request.app.state.engine.regime_override = None
            msg = "Market regime override cleared (Auto Mode)"
        elif req.regime.upper() in valid_regimes:
            request.app.state.engine.regime_override = req.regime.upper()
            msg = f"Market regime override set to {request.app.state.engine.regime_override}"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid regime. Must be one of: {sorted(valid_regimes)} or null/CLEAR",
            )
        current_override = request.app.state.engine.regime_override

    return {
        "status": "ok",
        "message": msg,
        "regime_override": current_override,
    }


# --- Emergency Kill-Switch Endpoints ---

@router.post("/api/killswitch", dependencies=[Depends(verify_api_key)])
async def activate_killswitch(request: Request) -> Dict[str, Any]:
    """Emergency Kill-Switch: aborts active scans and halts execution."""
    async with request.app.state.engine._lock:
        request.app.state.engine.is_killed = True
        request.app.state.engine.scan_error = "Emergency Killswitch Activated; cancellation requested"
        request.app.state.scan_cancel_event.set()
        _persistence(request.app).set_killswitch(True)

    log.critical("EMERGENCY KILLSWITCH ACTIVATED: Active scans cancelled.")
    await request.app.state.broadcaster.broadcast("killswitch_engaged", {"status": "killed"})
    return {
        "status": "killed",
        "message": "Emergency killswitch engaged. In-flight scan cancellation requested.",
        "killswitch_active": True,
    }


@router.post("/api/killswitch/reset", dependencies=[Depends(verify_api_key)])
async def reset_killswitch(request: Request) -> Dict[str, Any]:
    """Reset Emergency Kill-Switch to resume normal operations."""
    async with request.app.state.engine._lock:
        if request.app.state.engine.is_scanning:
            raise HTTPException(
                status_code=409,
                detail="Wait for the cancelled scan to stop before resetting the killswitch.",
            )
        _persistence(request.app).set_killswitch(False)
        request.app.state.engine.is_killed = False
        request.app.state.engine.scan_error = None
        request.app.state.scan_cancel_event.clear()

    log.info("Emergency killswitch cleared. Normal operations resumed.")
    await request.app.state.broadcaster.broadcast("killswitch_reset", {"status": "reset"})
    return {
        "status": "reset",
        "message": "Emergency killswitch reset. Normal operations resumed.",
        "killswitch_active": False,
    }


@router.get("/api/killswitch/status", dependencies=[Depends(verify_api_key)])
async def get_killswitch_status(request: Request) -> Dict[str, Any]:
    """Check current emergency kill-switch status."""
    async with request.app.state.engine._lock:
        return {
            "killswitch_active": request.app.state.engine.is_killed,
            "is_scanning": request.app.state.engine.is_scanning,
            "scan_error": request.app.state.engine.scan_error,
        }


# --- Server-Sent Events (SSE) Endpoint ---

@router.get("/api/events", dependencies=[Depends(verify_api_key)])
async def sse_events(
    request: Request,
    limit: Optional[int] = Query(default=None, ge=1, description="Optional max events to receive before closing"),
) -> StreamingResponse:
    """
    Real-time Server-Sent Events (SSE) stream for market scan progress, regime shifts,
    and emergency killswitch state transitions. Eliminates client-side polling.
    """
    queue = await request.app.state.broadcaster.subscribe()

    async def event_generator() -> AsyncGenerator[str, None]:
        yielded_count = 0
        # Emit initial state snapshot upon connection
        initial_payload = {
            "event": "connected",
            "data": {
                "version": svm.VERSION,
                "killswitch_active": request.app.state.engine.is_killed,
                "is_scanning": request.app.state.engine.is_scanning,
                "last_scan_time": request.app.state.engine.last_scan_time,
                "candidates_count": len(request.app.state.engine.last_candidates),
                "portfolio_count": len(request.app.state.engine.last_portfolio),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        yield f"data: {json.dumps(initial_payload)}\n\n"
        yielded_count += 1
        if limit is not None and yielded_count >= limit:
            await request.app.state.broadcaster.unsubscribe(queue)
            return

        try:
            while True:
                if await request.is_disconnected():
                    break

                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(msg)}\n\n"
                    yielded_count += 1
                    if limit is not None and yielded_count >= limit:
                        break
                except asyncio.TimeoutError:
                    heartbeat = {
                        "event": "ping",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    yield f"data: {json.dumps(heartbeat)}\n\n"
                    yielded_count += 1
                    if limit is not None and yielded_count >= limit:
                        break
        finally:
            await request.app.state.broadcaster.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/sectors", dependencies=[Depends(verify_api_key)])
async def get_sectors(request: Request) -> Dict[str, Any]:
    sector_summary: Dict[str, List[str]] = {
        sec: tickers for sec, tickers in SECTORS.items()
    }
    async with request.app.state.engine._lock:
        rs = dict(request.app.state.engine.last_sector_rs)
    return {
        "total_sectors": len(SECTORS),
        "total_tickers": len(TICKER_TO_SECTOR),
        "sectors": sector_summary,
        "last_sector_rs": rs,
    }


@router.get("/api/trades", dependencies=[Depends(verify_api_key)])
def get_trades(
    request: Request,
    ticker: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000, description="Max number of trades to return (1-1000)"),
) -> Dict[str, Any]:
    """Fetch trade execution logs from SQLite persistence."""
    trades = _persistence(request.app).load_trade_log()
    if ticker:
        trades = [t for t in trades if t.get("ticker") == ticker]
    sliced = trades[-limit:] if limit > 0 else []
    return {
        "count": len(sliced),
        "trades": sliced,
    }


@router.get("/api/config", dependencies=[Depends(verify_api_key)])
def get_config() -> Dict[str, Any]:
    rg = CONFIG.as_regime()
    return {
        "version": svm.VERSION,
        "benchmark": CONFIG.BENCHMARK,
        "risk_per_trade_inr": CONFIG.RISK_PER_TRADE_INR,
        "portfolio_size": CONFIG.PORTFOLIO_SIZE,
        "max_sector_picks": CONFIG.MAX_SECTOR_PICKS,
        "max_corr": CONFIG.MAX_CORR,
        "use_ema200_filter": CONFIG.USE_EMA200_FILTER,
        "regime_settings": {
            "breadth_veto_below": rg.breadth_veto_below,
            "regime_adx_trend": rg.regime_adx_trend,
            "regime_adx_range": rg.regime_adx_range,
            "regime_atr_expansion": rg.regime_atr_expansion,
            "regime_breadth_panic": rg.regime_breadth_panic,
        },
    }


def _persistence(application: FastAPI) -> PersistenceService:
    persistence: PersistenceService | None = application.state.persistence
    if persistence is None:
        raise HTTPException(status_code=503, detail="Engine startup has not completed")
    return persistence


def create_app(*, paths: RuntimePaths | None = None,
               persistence: PersistenceService | None = None, auto_scan: bool = True) -> FastAPI:
    """Independent engine, journal, cancellation and SSE state; no database opened here."""
    application = FastAPI(title="Sovereign Engine API Server",
                          description="Real-time quantitative scanner and market regime API server",
                          version=svm.VERSION, lifespan=lifespan)
    application.state.engine = EngineState()
    application.state.broadcaster = SSEBroadcaster()
    application.state.scan_cancel_event = threading.Event()
    application.state.persistence = persistence
    application.state.paths = paths
    application.state.auto_scan = auto_scan
    application.state.scan_task = None
    application.state.runtime = None
    limiter = Limiter(key_func=get_client_ip)
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    origins = ["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8000", "http://127.0.0.1:8000"]
    origins += [value.strip() for value in os.environ.get("CORS_ORIGINS", "").split(",") if value.strip()]
    application.add_middleware(CORSMiddleware, allow_origins=list(dict.fromkeys(origins)),
                               allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    limits = {"trigger_scan": "5/minute", "set_regime_override": "10/minute"}
    for route in router.routes:
        if isinstance(route, APIRoute):
            endpoint = limiter.limit(limits[route.name])(route.endpoint) if route.name in limits else route.endpoint
            application.add_api_route(route.path, endpoint, methods=list(route.methods),
                                      dependencies=route.dependencies, response_model=route.response_model,
                                      name=route.name)
    assets = FRONTEND_DIST_DIR / "assets"
    if assets.exists():
        application.mount("/assets", StaticFiles(directory=str(assets)), name="assets")
    return application


app = create_app()


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Sovereign Engine v{svm.VERSION} Server starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
