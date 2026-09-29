"""Stable, atomic scan-snapshot serialization shared by CLI and API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .config import SystemConfig

SNAPSHOT_VERSION = 1


def format_ticker(value: Any, config: SystemConfig) -> dict[str, Any]:
    data = dict(value) if isinstance(value, dict) else dict(value.__dict__)
    factors = data.get("factors")
    if factors is not None and not isinstance(factors, dict):
        data["factors"] = dict(getattr(factors, "__dict__", {}))
    direction = str(data.get("direction", "LONG"))
    data["action"] = data.get("action") or ("BUY" if direction == "LONG" else "SELL")
    entry = float(data.get("entry") or 0.0)
    stop = float(data.get("stop") or 0.0)
    target = float(data.get("t1") or 0.0)
    stop_pct = abs(entry - stop) / entry * 100 if entry > 0 else 5.0
    target_pct = abs(target - entry) / entry * 100 if entry > 0 else 10.0
    horizon = data.get("trade_horizon")
    is_mean_reversion = any("MeanRev" in str(reason) for reason in data.get("reasons", []))
    if horizon not in ("INTRADAY", "SWING"):
        if not config.INTRADAY_ENABLED:
            horizon = "SWING"
        elif direction == "SHORT" and config.SHORT_IS_INTRADAY_ONLY:
            horizon = "INTRADAY"
        elif stop_pct < 2.5 or (is_mean_reversion and stop_pct < 3.0):
            horizon = "INTRADAY"
        else:
            horizon = "SWING"
    data["trade_horizon"] = horizon
    data["horizon_label"] = "INTRADAY (MIS)" if horizon == "INTRADAY" else "SWING (CNC)"
    data["stop_pct"] = round(stop_pct, 2)
    data["target_pct"] = round(target_pct, 2)
    return data


def build_scan_snapshot(
    *,
    scan_time: str | None,
    candidates: Iterable[Any],
    portfolio: Iterable[Any],
    sector_rs: dict[str, float],
    regime_info: dict[str, Any] | None,
    config: SystemConfig,
) -> dict[str, Any]:
    return {
        "schema_version": SNAPSHOT_VERSION,
        "scan_time": scan_time,
        "candidates": [format_ticker(item, config) for item in candidates],
        "portfolio": [format_ticker(item, config) for item in portfolio],
        "sector_rs": dict(sector_rs),
        "regime_info": regime_info,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=str)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
