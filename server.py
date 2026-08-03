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

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import screener_v14_modular as svm
from core.config import CONFIG, MarketRegimeType
from core.regime import RegimeTracker
from core.scorer import TickerResult
from core.universe import SECTORS, TICKER_TO_SECTOR

from contextlib import asynccontextmanager

log = logging.getLogger("sovereign.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Trigger initial market scan in background worker on server startup."""
    t = threading.Thread(target=_run_scan_task, daemon=True)
    t.start()
    yield


app = FastAPI(
    title="Sovereign Engine API Server",
    description="Real-time quantitative scanner and market regime API server",
    version=svm.VERSION,
    lifespan=lifespan,
)

# CORS middleware for local web dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
        self._lock = threading.Lock()

STATE = EngineState()
BASE_DIR = Path(__file__).parent.resolve()


def _format_ticker_result(res: TickerResult) -> Dict[str, Any]:
    d = res.__dict__.copy()
    if hasattr(res, "factors") and res.factors is not None:
        d["factors"] = res.factors.__dict__.copy()
    return d


def _run_scan_task() -> None:
    with STATE._lock:
        STATE.is_scanning = True
        STATE.scan_error = None

    try:
        candidates, portfolio, regime = svm.run_scan(
            config=CONFIG,
            regime_tracker=STATE.regime_tracker,
            regime_override=STATE.regime_override,
        )

        with STATE._lock:
            STATE.last_scan_time = datetime.now(timezone.utc).isoformat()
            STATE.last_candidates = [_format_ticker_result(c) for c in candidates]
            STATE.last_portfolio = [_format_ticker_result(p) for p in portfolio]
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
    except Exception as exc:
        log.exception("Error executing scan task")
        with STATE._lock:
            STATE.is_scanning = False
            STATE.scan_error = str(exc)


# --- Request/Response Models ---

class OverrideRequest(BaseModel):
    regime: Optional[str] = Field(
        None, description="Regime type (PANIC, TREND_UP, TREND_DOWN, RANGE, EXPANSION) or null/CLEAR"
    )


# --- API Routes ---

@app.get("/")
def read_root() -> FileResponse:
    dashboard_path = BASE_DIR / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")
    return FileResponse(str(dashboard_path), media_type="text/html")


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    rg_settings = CONFIG.as_regime()
    session = rg_settings.session_from_time()
    locked = rg_settings.is_regime_locked()
    return {
        "status": "online",
        "version": svm.VERSION,
        "session": session,
        "regime_locked": locked,
        "regime_override": STATE.regime_override,
        "is_scanning": STATE.is_scanning,
        "last_scan_time": STATE.last_scan_time,
        "scan_error": STATE.scan_error,
        "last_known_regime": STATE.last_regime_info,
    }


@app.get("/api/scan")
def get_scan_results() -> Dict[str, Any]:
    return {
        "last_scan_time": STATE.last_scan_time,
        "is_scanning": STATE.is_scanning,
        "scan_error": STATE.scan_error,
        "regime": STATE.last_regime_info,
        "portfolio": STATE.last_portfolio,
        "candidates_count": len(STATE.last_candidates),
        "candidates": STATE.last_candidates,
    }


@app.post("/api/scan/trigger")
def trigger_scan(background_tasks: BackgroundTasks) -> Dict[str, Any]:
    if STATE.is_scanning:
        return {"status": "already_running", "message": "Scan execution already in progress"}

    background_tasks.add_task(_run_scan_task)
    return {"status": "triggered", "message": "Market scan started in background worker"}


@app.post("/api/regime/override")
def set_regime_override(req: OverrideRequest) -> Dict[str, Any]:
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
    print(f"Sovereign Engine v{svm.VERSION} Server starting on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
