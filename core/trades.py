"""Explicit fill lifecycle; recommendations never enter the execution book."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from .database import SqliteDatabase


class TradeFill(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, str_strip_whitespace=True)
    trade_id: str = Field(min_length=1, max_length=128)
    ticker: str = Field(min_length=1, max_length=32, pattern=r"^[A-Z0-9&._-]+$")
    direction: Literal["LONG", "SHORT"]
    trade_horizon: Literal["SWING", "INTRADAY"] = "SWING"
    source: Literal["EXECUTED", "SIMULATED"] = "EXECUTED"
    entry_price: float = Field(gt=0)
    entry_ts: AwareDatetime
    stop_loss: float = Field(gt=0)
    target: float = Field(gt=0)
    shares: int = Field(gt=0, strict=True)
    composite: float = Field(ge=0, le=1)
    prob_win: float = Field(ge=0, le=1)
    factors: dict[str, float]
    strategy_id: str = Field(default="LEGACY", min_length=1, max_length=128)
    signal_event_id: str | None = None
    signal_time: str | None = None
    entry_min: float | None = Field(default=None, gt=0)
    entry_max: float | None = Field(default=None, gt=0)
    fees_inr: float | None = Field(default=None, ge=0)
    slippage_inr: float | None = Field(default=None, ge=0)
    cost_basis: Literal["OBSERVED", "MODELED", "UNKNOWN"] = "UNKNOWN"

    @model_validator(mode="after")
    def validate_levels(self) -> "TradeFill":
        valid = (self.stop_loss < self.entry_price < self.target if self.direction == "LONG"
                 else self.target < self.entry_price < self.stop_loss)
        if not valid:
            raise ValueError("Stop and target must straddle the fill in its trade direction")
        if (self.entry_min is None) != (self.entry_max is None) or (
            self.entry_min is not None and self.entry_max is not None and self.entry_min > self.entry_max
        ):
            raise ValueError("Entry bounds must be supplied together and ordered")
        self.ticker = self.ticker.removesuffix(".NS")
        if not self.ticker:
            raise ValueError("Ticker must identify an instrument")
        return self


class TradeExit(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    exit_price: float = Field(gt=0)
    exit_ts: AwareDatetime
    costs: float = Field(ge=0, description="Total round-trip costs in INR")
    exit_reason: str = Field(min_length=1, max_length=128)
    fees_inr: float | None = Field(default=None, ge=0)
    slippage_inr: float | None = Field(default=None, ge=0)
    cost_basis: Literal["OBSERVED", "MODELED", "UNKNOWN"] = "UNKNOWN"


class TradeLifecycle:
    def __init__(self, db: SqliteDatabase) -> None:
        self.db = db

    def register(self, fill: TradeFill) -> dict[str, Any]:
        payload = fill.model_dump(mode="json")
        metadata = {key: payload[key] for key in ("signal_event_id", "signal_time", "entry_min", "entry_max",
                                                 "fees_inr", "slippage_inr", "cost_basis")}
        with self.db.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM executed_trades WHERE trade_id = ?",
                                    (fill.trade_id,)).fetchone()
            if existing is not None:
                keys = ("ticker", "direction", "trade_horizon", "entry_price", "entry_ts",
                        "stop_loss", "target", "shares", "composite", "prob_win", "source", "strategy_id")
                old_metadata = json.loads(existing["execution_metadata"])
                if (any(existing[key] != payload[key] for key in keys) or json.loads(existing["factors"]) != fill.factors
                        or any(old_metadata.get(key, "UNKNOWN" if key == "cost_basis" else None) != value for key, value in metadata.items())):
                    raise ValueError("Trade ID already belongs to a different fill")
                return dict(existing)
            signal = self.db.get_paper_event(fill.signal_event_id) if fill.signal_event_id else None
            if fill.signal_event_id and (signal is None or signal["event_type"] != "SIGNAL"
                                          or signal["payload"]["ticker"] != fill.ticker
                                          or signal["payload"]["strategy_version"] != fill.strategy_id):
                raise ValueError("Signal link must identify the same instrument and strategy")
            if conn.execute("SELECT 1 FROM open_positions WHERE ticker IN (?, ?)",
                            (fill.ticker, f"{fill.ticker}.NS")).fetchone():
                raise ValueError("Instrument already has an open position; close it before recording another fill")
            self.db.insert_executed_trade(payload)
            conn.execute("UPDATE executed_trades SET source = ?, strategy_id = ?, execution_metadata = ? WHERE trade_id = ?",
                         (fill.source, fill.strategy_id, json.dumps(metadata), fill.trade_id))
            self.db.upsert_open_position(
                fill.ticker, fill.direction, fill.entry_price, fill.shares, fill.stop_loss,
                fill.target, fill.prob_win, fill.composite, opened_at=payload["entry_ts"],
            )
            payload["opened_at"] = payload["entry_ts"]
            conn.execute("UPDATE open_positions SET raw_payload = ? WHERE ticker = ?",
                         (json.dumps(payload), fill.ticker))
            if fill.source == "SIMULATED":
                recorded = datetime.now(timezone.utc)
                bounds = signal["payload"]["entry_bound"] if signal else [fill.entry_min, fill.entry_max]
                delay = (recorded - fill.entry_ts).total_seconds()
                before_entry = bool(signal and datetime.fromisoformat(signal["recorded_at"]) <= fill.entry_ts)
                self.db.append_paper_event(f"fill:{fill.trade_id}", "PAPER_FILL", {
                    **payload, "strategy_version": fill.strategy_id, "fill": fill.entry_price,
                    "entry_bound": bounds, "stop": fill.stop_loss, "fees": fill.fees_inr,
                    "slippage": fill.slippage_inr, "exit_reason": None,
                    "signal_precedes_entry": before_entry, "entry_recording_delay_seconds": delay,
                    "future_dated_event": delay < 0,
                    "within_entry_bounds": bounds[0] <= fill.entry_price <= bounds[1] if all(b is not None for b in bounds) else None,
                })
            row = conn.execute("SELECT * FROM executed_trades WHERE trade_id = ?", (fill.trade_id,)).fetchone()
            return dict(row)

    def close(self, trade_id: str, exit_fill: TradeExit) -> dict[str, Any]:
        with self.db.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM executed_trades WHERE trade_id = ?", (trade_id,)).fetchone()
            if row is None:
                raise LookupError("Unknown trade ID")
            trade = dict(row)
            metadata = json.loads(trade["execution_metadata"])
            if exit_fill.exit_ts < datetime.fromisoformat(trade["entry_ts"]):
                raise ValueError("Exit precedes entry")
            exit_ts = exit_fill.exit_ts.isoformat()
            if trade["outcome"] != "OPEN":
                if (trade["exit_price"], trade["exit_ts"], trade["costs"]) != (
                    exit_fill.exit_price, exit_ts, exit_fill.costs
                ):
                    raise ValueError("Trade already closed with a different exit")
                if metadata.get("exit_event") not in (None, exit_fill.model_dump(mode="json")):
                    raise ValueError("Trade already closed with different exit metadata")
                return trade
            quantity = int(trade["shares"])
            entry = float(trade["entry_price"])
            gross = (exit_fill.exit_price - entry) * quantity * (1 if trade["direction"] == "LONG" else -1)
            net = gross - exit_fill.costs
            risk = abs(entry - float(trade["stop_loss"])) * quantity
            outcome = "WIN" if net > 0 else "LOSS" if net < 0 else "BREAKEVEN"
            self.db.close_executed_trade(trade_id, exit_fill.exit_price, exit_ts, gross,
                                         exit_fill.costs, net, net / risk, outcome)
            metadata["exit_event"] = exit_fill.model_dump(mode="json")
            conn.execute("UPDATE executed_trades SET execution_metadata = ? WHERE trade_id = ?",
                         (json.dumps(metadata), trade_id))
            if trade["source"] == "SIMULATED":
                self.db.append_paper_event(f"exit:{trade_id}", "PAPER_EXIT", {
                    "trade_id": trade_id, "ticker": trade["ticker"], "strategy_version": trade["strategy_id"],
                    "signal_event_id": metadata.get("signal_event_id"), **exit_fill.model_dump(mode="json"),
                    "fill": exit_fill.exit_price, "stop": trade["stop_loss"], "target": trade["target"],
                    "fees": exit_fill.fees_inr, "slippage": exit_fill.slippage_inr,
                    "net_pnl": net, "realised_r": net / risk,
                    "exit_recording_delay_seconds": (datetime.now(timezone.utc) - exit_fill.exit_ts).total_seconds(),
                })
            self.db.insert_trade({
                **trade, "factors": json.loads(trade["factors"]), "entry": entry,
                "exit_price": exit_fill.exit_price, "exit_reason": exit_fill.exit_reason,
                "timestamp": exit_ts, "gross_pnl": gross, "costs": exit_fill.costs,
                "net_pnl": net, "realised_r": net / risk, "pnl": net / (entry * quantity),
                "outcome": outcome,
            })
            self.db.delete_open_position(str(trade["ticker"]))
            closed = conn.execute("SELECT * FROM executed_trades WHERE trade_id = ?", (trade_id,)).fetchone()
            return dict(closed)
