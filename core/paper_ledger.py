"""First-observed signals and append-only export of the authoritative paper journal."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from typing import Any, TYPE_CHECKING, Sequence
from uuid import uuid4

from .config import SystemConfig
from .database import SqliteDatabase
from .scorer import TickerResult

if TYPE_CHECKING:
    from .swing_replay import ReplaySignal


class PaperLedger:
    def __init__(self, db: SqliteDatabase) -> None:
        self.db = db

    def record_scan(self, market_date: str, regime: str, signals: int, selected: int) -> dict[str, Any]:
        """Record zero-signal days too; repeated scans remain separate observations."""
        return self.db.append_paper_event(f"scan:{uuid4().hex}", "SCAN", {
            "market_date": market_date, "regime": regime, "qualified_signals": signals,
            "selected": selected, "mode": "SWING_RESEARCH", "fill": None,
        })

    def record_signals(self, results: list[TickerResult], portfolio: list[TickerResult], config: SystemConfig) -> list[dict[str, Any]]:
        selected = {r.ticker for r in portfolio if not r.is_held}
        parameters = {k: v for k, v in asdict(config).items() if k.startswith("SWING_") or k in {
            "RISK_PER_TRADE_INR", "MAX_PORTFOLIO_RISK_INR", "MAX_CORR", "MAX_SECTOR_PICKS", "CAPITAL_INR",
            "MIN_PROB_WIN", "MIN_EXPECTANCY_R", "COHORT_RANK_WEIGHT", "PORTFOLIO_SIZE",
            "ADV_TURNOVER_FLOOR", "ADV_SHARE_FLOOR", "SLIPPAGE_BPS"}}
        protocol_path = Path(__file__).resolve().parents[1] / "docs" / "swing-hypotheses-v1.json"
        protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest() if protocol_path.exists() else None
        parameters["RESEARCH_PROTOCOL_SHA256"] = protocol_hash
        fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()
        events = []
        with self.db.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            for result in results:
                if result.is_held or not result.strategy_id.startswith("SWING_"):
                    continue
                identity = f"{result.ticker}|{result.strategy_id}|{result.signal_time}|{fingerprint}"
                event_id = "signal:" + hashlib.sha256(identity.encode()).hexdigest()
                existing = self.db.get_paper_event(event_id)
                if existing is not None:
                    events.append(existing)
                    continue
                payload = dict(ticker=result.ticker, strategy_version=result.strategy_id, signal_time=result.signal_time,
                               entry_bound=[result.entry_min, result.entry_max], entry_reference=result.entry,
                               stop=result.stop, target=result.t1, time_stop_bars=result.time_stop_bars,
                               shares=result.shares, probability_status=result.probability_status,
                               composite=result.composite, prob_win=result.prob_win,
                               factors=result.factors.as_dict(), factor_weights=result.factors.ic_weights,
                               research_context=getattr(result, "research_context", {}),
                               research_protocol={"version": "SWING_STRUCTURAL_V1", "sha256": protocol_hash},
                               regime=result.regime, selected=result.ticker in selected, watchlist=result.is_watchlist,
                               fill=None, exit_reason=None, fees=None, slippage=None,
                               parameters=parameters, parameter_fingerprint=fingerprint)
                events.append(self.db.append_paper_event(event_id, "SIGNAL", payload))
        return events

    def record_pullback_signals(self, signals: Sequence[Any]) -> list[dict[str, Any]]:
        """Idempotent per ticker/strategy/date; entry is next open, so no fill is recorded here."""
        events = []
        for s in signals:
            event_id = "signal:" + hashlib.sha256(f"{s.ticker}|{s.strategy_id}|{s.signal_date}".encode()).hexdigest()
            existing = self.db.get_paper_event(event_id)
            events.append(existing if existing is not None else self.db.append_paper_event(
                event_id, "SIGNAL", {**asdict(s), "mode": "PAPER_ONLY", "entry": "next_open", "fill": None}))
        return events

    def record_research_scan(self, results: Sequence[ReplaySignal], market_date: str,
                             parameters: dict[str, Any], provenance: dict[str, Any]) -> list[dict[str, Any]]:
        """Freeze V2 observations with their own protocol, without V1 calibration or selection."""
        fingerprint = hashlib.sha256(json.dumps(parameters, sort_keys=True, allow_nan=False).encode()).hexdigest()
        events = []
        with self.db.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self.db.append_paper_event(f"scan:{uuid4().hex}", "SCAN", {
                "market_date": market_date, "mode": "SWING_V2_RESEARCH", "qualified_signals": len(results),
                "selected": 0, "fill": None, "parameters": parameters, "provenance": provenance})
            for result in results:
                identity = f"{result.ticker}|{result.strategy_id}|{result.signal_time}|{fingerprint}"
                event_id = "signal:" + hashlib.sha256(identity.encode()).hexdigest()
                existing = self.db.get_paper_event(event_id)
                if existing is not None:
                    events.append(existing)
                    continue
                payload = dict(ticker=result.ticker, strategy_version=result.strategy_id,
                               signal_time=result.signal_time, entry_reference=result.entry,
                               entry_bound=[result.entry_min, result.entry_max], stop=result.stop,
                               target=result.t1, time_stop_bars=result.time_stop_bars,
                               probability_status="UNAVAILABLE", prob_win=None, expectancy_r=None,
                               composite=result.composite, selected=False, fill=None,
                               research_context=result.research_context, parameters=parameters,
                               parameter_fingerprint=fingerprint, provenance=provenance)
                events.append(self.db.append_paper_event(event_id, "SIGNAL", payload))
        return events

    def export_jsonl(self, path: Path) -> int:
        """Validate the existing prefix, then append only. Never repair by overwriting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.with_suffix(path.suffix + ".lock")
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(descriptor)
        try:
            events: list[dict[str, Any]] = []
            cursor = 0
            while batch := self.db.fetch_paper_events(after=cursor):
                events.extend(batch)
                cursor = batch[-1]["sequence"]
            expected = [json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) for event in events]
            previous = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
            if previous != expected[:len(previous)]:
                raise ValueError("Export history differs from the paper journal; refusing to rewrite it")
            # A crash-truncated final line must not be concatenated into a new event.
            if path.exists() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
                raise ValueError("Export has an incomplete final line; refusing to alter history")
            new = expected[len(previous):]
            with path.open("ab") as stream:
                if new:
                    stream.write(("\n".join(new) + "\n").encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            return len(new)
        finally:
            lock.unlink()
