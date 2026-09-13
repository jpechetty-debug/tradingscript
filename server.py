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
import json
import logging
import os
from pathlib import Path
import sys
import threading
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
import uvicorn

import screener_v14_modular as svm
from core.config import CONFIG, MarketRegimeType
from core.regime import RegimeTracker
from core.scorer import TickerResult
from core.services import PersistenceService
from core.universe import SECTORS, TICKER_TO_SECTOR

load_dotenv()

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    if os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        API_KEY = "test-api-key"
    else:
        API_KEY = None

def verify_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY or not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Could not validate credentials"
        )


log = logging.getLogger("sovereign.server")

limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Validate server security config and trigger initial non-blocking market scan."""
    is_testing = bool(os.environ.get("PYTEST_CURRENT_TEST") or "pytest" in sys.modules)
    if not API_KEY and not is_testing:
        raise RuntimeError(
            "CRITICAL SECURITY CONFIGURATION ERROR: 'API_KEY' environment variable must be set. "
            "Please configure API_KEY in your .env or environment variables."
        )
    # Re-hydrate state from disk so server immediately has previous scan results ready
    STATE.load_persisted_state(PERSISTENCE)

    # Trigger initial scan in non-blocking background task unless killswitch is active
    scan_task = None
    if not STATE.is_killed:
        scan_task = asyncio.create_task(_run_scan_task_async())
        app.state.scan_task = scan_task
    try:
        yield
    finally:
        if scan_task and not scan_task.done():
            scan_task.cancel()
            try:
                await scan_task
            except asyncio.CancelledError:
                pass


app = FastAPI(
    title="Sovereign Engine API Server",
    description="Real-time quantitative scanner and market regime API server",
    version=svm.VERSION,
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Explicit CORS allowlist
cors_env = os.environ.get("CORS_ORIGINS", "")
custom_origins = [o.strip() for o in cors_env.split(",") if o.strip()]
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
ALLOWED_ORIGINS = list(dict.fromkeys(DEFAULT_ORIGINS + custom_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Engine State
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
        self._lock = threading.Lock()

    def persist_state(self, persistence: PersistenceService) -> None:
        """Persist current scan results to disk (state/latest_scan.json)."""
        try:
            target = persistence.paths.state_dir / "latest_scan.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "scan_time": self.last_scan_time,
                "candidates": self.last_candidates,
                "portfolio": self.last_portfolio,
                "sector_rs": self.last_sector_rs,
                "regime_info": self.last_regime_info,
            }
            target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            log.info("Persisted latest scan snapshot to %s", target)
        except Exception as exc:
            log.warning("Could not persist latest scan state: %s", exc)

    def load_persisted_state(self, persistence: PersistenceService) -> bool:
        """Re-hydrate state from state/latest_scan.json if it exists."""
        try:
            target = persistence.paths.state_dir / "latest_scan.json"
            if not target.exists():
                return False
            data = json.loads(target.read_text(encoding="utf-8"))
            with self._lock:
                self.last_scan_time = data.get("scan_time")
                raw_cands = data.get("candidates") or []
                raw_port = data.get("portfolio") or []

                def _enrich(item: dict) -> dict:
                    d = item.copy()
                    dirn = d.get("direction", "LONG")
                    d["action"] = d.get("action") or ("BUY" if dirn == "LONG" else "SELL")
                    entry = float(d.get("entry") or 0.0)
                    stop = float(d.get("stop") or 0.0)
                    t1 = float(d.get("t1") or 0.0)
                    sl_dist = abs(entry - stop) if (entry and stop) else 0.0
                    sl_pct = (sl_dist / entry * 100) if entry > 0 else 5.0
                    t1_dist = abs(t1 - entry) if (entry and t1) else 0.0
                    t1_pct = (t1_dist / entry * 100) if entry > 0 else 10.0
                    is_mean_rev = any("MeanRev" in str(r) for r in d.get("reasons", []))

                    # Calibrate cash equity shorts to true intraday targets (tight ~1.2% stop, ~2.8% target)
                    if dirn == "SHORT" and sl_pct > 2.0:
                        atr_est = sl_dist / 1.50
                        new_stop = round(entry + 0.50 * atr_est, 2)
                        new_t1 = round(entry - 1.20 * atr_est, 2)
                        new_t2 = round(entry - 1.80 * atr_est, 2)
                        new_sl_dist = abs(entry - new_stop)
                        d["stop"] = new_stop
                        d["t1"] = new_t1
                        d["t2"] = new_t2
                        d["rr_t1"] = 2.4
                        alloc_risk = float(d.get("risk_inr") or 0.0)
                        if alloc_risk > 0 and new_sl_dist > 0:
                            d["shares"] = int(alloc_risk / new_sl_dist)
                        sl_dist = new_sl_dist
                        sl_pct = (sl_dist / entry * 100) if entry > 0 else 1.2
                        t1_dist = abs(new_t1 - entry)
                        t1_pct = (t1_dist / entry * 100) if entry > 0 else 2.8

                    if dirn == "SHORT" or sl_pct < 2.5 or (is_mean_rev and sl_pct < 3.0):
                        d["trade_horizon"] = "INTRADAY"
                        d["horizon_label"] = "INTRADAY (MIS)"
                    else:
                        d["trade_horizon"] = "SWING"
                        d["horizon_label"] = "SWING (CNC)"
                    d["stop_pct"] = round(sl_pct, 2)
                    d["target_pct"] = round(t1_pct, 2)
                    return d

                self.last_candidates = [_enrich(c) for c in raw_cands]
                self.last_portfolio = [_enrich(p) for p in raw_port]
                self.last_sector_rs = data.get("sector_rs") or {}
                self.last_regime_info = data.get("regime_info")
            log.info("Re-hydrated EngineState from %s (%d candidates, %d portfolio)",
                     target, len(self.last_candidates), len(self.last_portfolio))
            return True
        except Exception as exc:
            log.warning("Failed to re-hydrate state from %s: %s", target, exc)
            return False

STATE = EngineState()
PERSISTENCE = PersistenceService()
STATE.load_persisted_state(PERSISTENCE)



class SSEBroadcaster:
    """Thread-safe event broadcaster for Server-Sent Events (SSE)."""

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        async with self._lock:
            self._subscribers.add(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue) -> None:
        async with self._lock:
            self._subscribers.discard(queue)

    async def broadcast(self, event: str, data: dict[str, Any]) -> None:
        message = {
            "event": event,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        async with self._lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(message)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                        q.put_nowait(message)
                    except Exception:
                        pass


BROADCASTER = SSEBroadcaster()

BASE_DIR = Path(__file__).parent.resolve()
FRONTEND_DIST_DIR = BASE_DIR / "frontend" / "dist"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"
if FRONTEND_ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_ASSETS_DIR)), name="frontend_assets")


def _format_ticker_result(res: TickerResult) -> Dict[str, Any]:
    d = res.__dict__.copy()
    if hasattr(res, "factors") and res.factors is not None:
        d["factors"] = res.factors.__dict__.copy()

    # Explicit Action (BUY vs SELL)
    d["action"] = getattr(res, "action", None) or ("BUY" if res.direction == "LONG" else "SELL")

    # Trade Horizon: In Indian equity cash market, SHORT is strictly INTRADAY (SEBI square-off by 15:15)
    # Prefer engine-classified trade_horizon if available
    engine_horizon = getattr(res, "trade_horizon", None)

    sl_dist = abs(res.entry - res.stop) if (res.entry and res.stop) else 0.0
    sl_pct = (sl_dist / res.entry * 100) if res.entry > 0 else 5.0
    t1_dist = abs(res.t1 - res.entry) if (res.entry and res.t1) else 0.0
    t1_pct = (t1_dist / res.entry * 100) if res.entry > 0 else 10.0

    is_mean_rev = any("MeanRev" in str(r) for r in getattr(res, "reasons", []))

    if res.direction == "SHORT":
        horizon = "INTRADAY"
    elif engine_horizon in ("INTRADAY", "SWING"):
        horizon = engine_horizon
    elif sl_pct < 2.5 or (is_mean_rev and sl_pct < 3.0):
        horizon = "INTRADAY"
    else:
        horizon = "SWING"

    d["trade_horizon"] = horizon
    d["horizon_label"] = "INTRADAY (MIS)" if horizon == "INTRADAY" else "SWING (CNC)"

    d["stop_pct"] = round(sl_pct, 2)
    d["target_pct"] = round(t1_pct, 2)
    return d



async def _run_scan_task_async() -> None:
    with STATE._lock:
        if STATE.is_killed:
            log.warning("Scan aborted: Emergency Killswitch is active.")
            return
        if STATE.is_scanning:
            return
        STATE.is_scanning = True
        STATE.scan_error = None

    try:
        await BROADCASTER.broadcast("scan_started", {"timestamp": datetime.now(timezone.utc).isoformat()})
        scan_output = await asyncio.to_thread(
            svm.run_scan,
            config=CONFIG,
            regime_tracker=STATE.regime_tracker,
            regime_override=STATE.regime_override,
        )
        candidates, portfolio, regime = scan_output[0], scan_output[1], scan_output[2]

        with STATE._lock:
            STATE.last_scan_time = datetime.now(timezone.utc).isoformat()
            STATE.last_candidates = [_format_ticker_result(c) for c in candidates]
            STATE.last_portfolio = [_format_ticker_result(p) for p in portfolio]
            STATE.last_sector_rs = getattr(scan_output, "sector_rs", None) or svm.get_last_sector_rs()
            if regime:
                STATE.last_regime_info = {
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
            STATE.is_scanning = False
            STATE.persist_state(PERSISTENCE)

        await BROADCASTER.broadcast("scan_completed", {
            "candidates_count": len(STATE.last_candidates),
            "portfolio_count": len(STATE.last_portfolio),
            "regime": STATE.last_regime_info,
        })
    except Exception as exc:
        log.exception("Error executing scan task")
        with STATE._lock:
            STATE.is_scanning = False
            STATE.scan_error = str(exc)
        await BROADCASTER.broadcast("scan_failed", {"error": str(exc)})


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

@app.get("/")
def read_root() -> FileResponse:
    dist_index = FRONTEND_DIST_DIR / "index.html"
    if dist_index.exists():
        return FileResponse(str(dist_index), media_type="text/html")
    dashboard_path = BASE_DIR / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return FileResponse(str(dashboard_path), media_type="text/html")


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    rg_settings = CONFIG.as_regime()
    session = rg_settings.session_from_time()
    locked = rg_settings.is_regime_locked()
    phase = rg_settings.get_market_phase()
    mins_to_sq = rg_settings.minutes_to_squareoff()
    phase_val = phase.value if hasattr(phase, "value") else str(phase)
    cutoff_active = phase_val in ("INTRADAY_FREEZE", "SWING_CLOSING", "POST_MARKET")
    with STATE._lock:
        return {
            "status": "online",
            "version": svm.VERSION,
            "session": session,
            "market_phase": phase_val,
            "minutes_to_squareoff": mins_to_sq,
            "intraday_cutoff_active": cutoff_active,
            "regime_locked": locked,
            "regime_override": STATE.regime_override,
            "is_scanning": STATE.is_scanning,
            "last_scan_time": STATE.last_scan_time,
            "scan_error": STATE.scan_error,
            "last_known_regime": STATE.last_regime_info,
            "killswitch_active": STATE.is_killed,
        }


@app.get("/api/scan")
def get_scan_results() -> Dict[str, Any]:
    rg_settings = CONFIG.as_regime()
    phase = rg_settings.get_market_phase()
    mins_to_sq = rg_settings.minutes_to_squareoff()
    phase_val = phase.value if hasattr(phase, "value") else str(phase)
    cutoff_active = phase_val in ("INTRADAY_FREEZE", "SWING_CLOSING", "POST_MARKET")
    with STATE._lock:
        return {
            "last_scan_time": STATE.last_scan_time,
            "is_scanning": STATE.is_scanning,
            "scan_error": STATE.scan_error,
            "market_phase": phase_val,
            "minutes_to_squareoff": mins_to_sq,
            "intraday_cutoff_active": cutoff_active,
            "regime": STATE.last_regime_info,
            "portfolio": STATE.last_portfolio,
            "candidates_count": len(STATE.last_candidates),
            "candidates": STATE.last_candidates,
        }


@app.post("/api/scan/trigger", dependencies=[Depends(verify_api_key)])
@limiter.limit("5/minute")
async def trigger_scan(request: Request, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    if STATE.is_killed:
        raise HTTPException(
            status_code=403,
            detail="Emergency Killswitch is active. Clear killswitch before initiating scans.",
        )
    if STATE.is_scanning:
        return {"status": "already_running", "message": "Scan execution already in progress"}

    background_tasks.add_task(_run_scan_task_async)
    return {"status": "triggered", "message": "Market scan started in background worker"}


@app.post("/api/regime/override", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def set_regime_override(request: Request, req: OverrideRequest) -> Dict[str, Any]:
    valid_regimes = {"PANIC", "TREND_UP", "TREND_DOWN", "RANGE", "EXPANSION"}
    if req.regime is None or req.regime.upper() in ("CLEAR", "NONE", "AUTO"):
        STATE.regime_override = None
        msg = "Market regime override cleared (Auto Mode)"
    elif req.regime.upper() in valid_regimes:
        STATE.regime_override = req.regime.upper()
        msg = f"Market regime override set to {STATE.regime_override}"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid regime. Must be one of: {sorted(valid_regimes)} or null/CLEAR",
        )

    return {
        "status": "ok",
        "message": msg,
        "regime_override": STATE.regime_override,
    }


# --- Emergency Kill-Switch Endpoints ---

@app.post("/api/killswitch", dependencies=[Depends(verify_api_key)])
async def activate_killswitch() -> Dict[str, Any]:
    """Emergency Kill-Switch: aborts active scans and halts execution."""
    with STATE._lock:
        STATE.is_killed = True
        STATE.is_scanning = False
        STATE.scan_error = "Emergency Killswitch Activated"

    scan_task = getattr(app.state, "scan_task", None)
    if scan_task and not scan_task.done():
        scan_task.cancel()

    log.critical("EMERGENCY KILLSWITCH ACTIVATED: Active scans cancelled.")
    await BROADCASTER.broadcast("killswitch_engaged", {"status": "killed"})
    return {
        "status": "killed",
        "message": "Emergency killswitch engaged. In-flight scans cancelled.",
        "killswitch_active": True,
    }


@app.post("/api/killswitch/reset", dependencies=[Depends(verify_api_key)])
async def reset_killswitch() -> Dict[str, Any]:
    """Reset Emergency Kill-Switch to resume normal operations."""
    with STATE._lock:
        STATE.is_killed = False
        STATE.scan_error = None

    log.info("Emergency killswitch cleared. Normal operations resumed.")
    await BROADCASTER.broadcast("killswitch_reset", {"status": "reset"})
    return {
        "status": "reset",
        "message": "Emergency killswitch reset. Normal operations resumed.",
        "killswitch_active": False,
    }


@app.get("/api/killswitch/status")
def get_killswitch_status() -> Dict[str, Any]:
    """Check current emergency kill-switch status."""
    with STATE._lock:
        return {
            "killswitch_active": STATE.is_killed,
            "is_scanning": STATE.is_scanning,
            "scan_error": STATE.scan_error,
        }


# --- Server-Sent Events (SSE) Endpoint ---

@app.get("/api/events")
async def sse_events(
    request: Request,
    limit: Optional[int] = Query(default=None, ge=1, description="Optional max events to receive before closing"),
) -> StreamingResponse:
    """
    Real-time Server-Sent Events (SSE) stream for market scan progress, regime shifts,
    and emergency killswitch state transitions. Eliminates client-side polling.
    """
    queue = await BROADCASTER.subscribe()

    async def event_generator():
        yielded_count = 0
        # Emit initial state snapshot upon connection
        initial_payload = {
            "event": "connected",
            "data": {
                "version": svm.VERSION,
                "killswitch_active": STATE.is_killed,
                "is_scanning": STATE.is_scanning,
                "last_scan_time": STATE.last_scan_time,
                "candidates_count": len(STATE.last_candidates),
                "portfolio_count": len(STATE.last_portfolio),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        yield f"data: {json.dumps(initial_payload)}\n\n"
        yielded_count += 1
        if limit is not None and yielded_count >= limit:
            await BROADCASTER.unsubscribe(queue)
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
            await BROADCASTER.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sectors")
def get_sectors() -> Dict[str, Any]:
    sector_summary: Dict[str, List[str]] = {
        sec: tickers for sec, tickers in SECTORS.items()
    }
    return {
        "total_sectors": len(SECTORS),
        "total_tickers": len(TICKER_TO_SECTOR),
        "sectors": sector_summary,
        "last_sector_rs": STATE.last_sector_rs,
    }


@app.get("/api/trades")
def get_trades(
    ticker: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000, description="Max number of trades to return (1-1000)"),
) -> Dict[str, Any]:
    """Fetch trade execution logs from SQLite persistence."""
    trades = PERSISTENCE.load_trade_log()
    if ticker:
        trades = [t for t in trades if t.get("ticker") == ticker]
    sliced = trades[-limit:] if limit > 0 else []
    return {
        "count": len(sliced),
        "trades": sliced,
    }


@app.get("/api/config")
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


def main() -> None:
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print(f"Sovereign Engine v{svm.VERSION} Server starting on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
