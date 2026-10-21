"""Read-only, observation-dated reconciliation of the append-only paper journal."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import closing
import csv
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

from .config import IST


def _aware(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
        if isinstance(parsed, datetime) and parsed.tzinfo is not None and parsed.utcoffset() is not None:
            return parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        pass
    return None


def _number(value: Any, minimum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) and (minimum is None or number >= minimum) else None
    except OverflowError:
        return None


def _label_date(value: Any) -> str | None:
    """Candle dates are labels, never first-observed timestamps."""
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date().isoformat() if isinstance(value, str) else None
    except ValueError:
        return None


def _symbol(value: Any) -> str:
    return str(value or "").strip().upper().removesuffix(".NS")


def _strategy(payload: dict[str, Any]) -> str | None:
    value = payload.get("strategy_version") or payload.get("strategy_id")
    return value if isinstance(value, str) and value else None


def read_paper_journal(path: Path) -> list[dict[str, Any]]:
    """One SQLite read transaction, no constructor migrations, table writes or row limit."""
    path = path.resolve(strict=True)
    with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=10)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only = ON")
        conn.execute("BEGIN")
        cursor = conn.execute("SELECT sequence, event_id, event_type, recorded_at, payload FROM paper_events ORDER BY sequence")
        return [dict(row) for row in cursor]


class _DayReport:
    def __init__(self, day: date, cutoff: datetime, delay_seconds: float) -> None:
        self.day, self.cutoff, self.delay_seconds = day, cutoff, delay_seconds
        self.issues: list[dict[str, Any]] = []

    def issue(self, event: dict[str, Any], code: str, category: str, detail: str, severity: str = "WARNING") -> None:
        self.issues.append({"sequence": event.get("sequence"), "event_id": event.get("event_id"),
                            "trade_id": event.get("payload", {}).get("trade_id"),
                            "observed_date": event.get("observed_date"), "category": category,
                            "code": code, "severity": severity, "detail": detail})

    def cost_fields(self, event: dict[str, Any], prefix: str) -> dict[str, Any]:
        payload = event["payload"]
        result: dict[str, Any] = {}
        for name in ("fees", "slippage"):
            raw = payload.get(f"{name}_inr")
            if raw is None:
                raw = payload.get(name)
            result[f"{prefix}_{name}_inr"] = _number(raw, 0)
            if raw is None:
                self.issue(event, f"MISSING_{prefix.upper()}_{name.upper()}", "COST", "Omitted amounts are unknown, not zero.")
            elif result[f"{prefix}_{name}_inr"] is None:
                self.issue(event, f"INVALID_{prefix.upper()}_{name.upper()}", "COST", "Expected a finite nonnegative INR amount.", "ERROR")
        basis = payload.get("cost_basis", "UNKNOWN")
        result[f"{prefix}_cost_basis"] = basis if basis in ("OBSERVED", "MODELED", "UNKNOWN") else "UNKNOWN"
        if basis not in ("OBSERVED", "MODELED"):
            self.issue(event, f"UNKNOWN_{prefix.upper()}_COST_BASIS", "COST", "Observed versus modeled costs are not established.")
        return result

    def event_time(self, event: dict[str, Any], field: str, prefix: str) -> tuple[datetime | None, float | None]:
        stamp = _aware(event["payload"].get(field))
        if stamp is None:
            self.issue(event, f"INVALID_{prefix.upper()}_TIME", "TIMING", f"{field} is missing, invalid or lacks a timezone.", "ERROR")
            return None, None
        delay = (event["recorded"] - stamp).total_seconds()
        if delay < 0:
            self.issue(event, f"FUTURE_DATED_{prefix.upper()}", "TIMING", "Supplied event time is later than the journal observation.", "ERROR")
        elif delay > self.delay_seconds:
            self.issue(event, f"DELAYED_{prefix.upper()}_RECORDING", "TIMING", f"Recorded {delay:.3f} seconds after the supplied event time.")
        return stamp, delay


def reconcile_paper_events(
    events: list[dict[str, Any]], day: date, *, as_of: datetime | None = None,
    max_recording_delay_seconds: float = 900,
) -> dict[str, Any]:
    """Reconstruct known journal links by day-end; daily counts use observation time in IST."""
    clock = _aware(as_of if as_of is not None else datetime.now(timezone.utc))
    if clock is None:
        raise ValueError("as_of must be a timezone-aware datetime")
    if day > clock.astimezone(IST).date():
        raise ValueError("Report date is later than as_of in IST")
    if _number(max_recording_delay_seconds, 0) is None:
        raise ValueError("Recording-delay threshold must be finite and nonnegative")
    end = datetime.combine(day + timedelta(days=1), time.min, IST) - timedelta(microseconds=1)
    cutoff = min(clock, end.astimezone(timezone.utc))
    audit = _DayReport(day, cutoff, max_recording_delay_seconds)
    known: list[dict[str, Any]] = []
    digest_rows = []
    excluded: Counter[str] = Counter()
    ids: set[str] = set()
    for raw in events:
        event = dict(raw)
        stamp = _aware(event.get("recorded_at"))
        event["payload"] = {}
        if stamp is None:
            audit.issue(event, "UNASSIGNED_JOURNAL_TIME", "INTEGRITY", "Cannot place this row before the cutoff or in a reporting day.", "ERROR")
            continue
        if stamp > cutoff:
            continue
        digest_rows.append(raw)
        event["recorded"], event["observed_date"] = stamp, stamp.astimezone(IST).date().isoformat()
        try:
            payload = raw.get("payload")
            payload = json.loads(payload) if isinstance(payload, str) else payload
            if not isinstance(payload, dict):
                raise ValueError("Expected an object")
            # Reject NaN/Infinity even if a legacy JSON decoder accepted them.
            json.dumps(payload, allow_nan=False)
            event["payload"] = payload
        except (ValueError, TypeError):
            audit.issue(event, "INVALID_EVENT_PAYLOAD", "INTEGRITY", "Payload is not a finite JSON object.", "ERROR")
            continue
        if not isinstance(event.get("sequence"), int) or isinstance(event["sequence"], bool):
            audit.issue(event, "INVALID_SEQUENCE", "INTEGRITY", "Sequence is not an integer.", "ERROR")
            continue
        identity = event.get("event_id")
        if not isinstance(identity, str) or not identity or identity in ids:
            audit.issue(event, "DUPLICATE_OR_INVALID_EVENT_ID", "INTEGRITY", "Stable journal identity is missing or duplicated.", "ERROR")
            continue
        ids.add(identity)
        mode, source = str(payload.get("mode", "")).upper(), str(payload.get("source", "")).upper()
        if source == "EXECUTED" or mode in ("REPLAY", "HISTORICAL_REPLAY") or source == "REPLAY" or str(event["event_type"]).upper() == "REPLAY":
            excluded["EXECUTED" if source == "EXECUTED" else "REPLAY"] += 1
            audit.issue(event, "NON_PAPER_EVENT_EXCLUDED", "SCOPE", "Executed/replay records are excluded from paper reconciliation.", "INFO")
            continue
        known.append(event)
    known.sort(key=lambda item: item["sequence"])
    signal_events = {e["event_id"]: e for e in known if e["event_type"] == "SIGNAL"}
    signals: dict[str, dict[str, Any]] = {}
    scans: list[dict[str, Any]] = []
    fill_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    exit_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in known:
        payload, kind = event["payload"], event["event_type"]
        if kind == "SIGNAL":
            provenance = payload.get("provenance")
            protocol = payload.get("research_protocol")
            provenance = provenance if isinstance(provenance, dict) else {}
            protocol = protocol if isinstance(protocol, dict) else {}
            row = {"event_id": event["event_id"], "sequence": event["sequence"],
                   "first_observed_at": event["recorded_at"], "observed_date": event["observed_date"],
                   "signal_label": payload.get("signal_time", payload.get("signal_date")),
                   "ticker": _symbol(payload.get("ticker")), "strategy_id": _strategy(payload),
                   "mode": payload.get("mode"), "selected": payload.get("selected"), "watchlist": payload.get("watchlist"),
                   "parameter_fingerprint": payload.get("parameter_fingerprint"),
                   "protocol_sha256": protocol.get("sha256") or provenance.get("protocol_sha256"),
                   "probability_status": payload.get("probability_status"), "linked_trade_ids": []}
            row["market_date"] = _label_date(row["signal_label"])
            row["observation_lag_calendar_days"] = ((date.fromisoformat(row["observed_date"]) - date.fromisoformat(row["market_date"])).days
                                                    if row["market_date"] else None)
            lag = row["observation_lag_calendar_days"]
            if lag is None:
                audit.issue(event, "SIGNAL_MARKET_DATE_UNAVAILABLE", "TIMING", "Signal candle/date label is invalid or absent.")
            elif lag < 0:
                audit.issue(event, "FUTURE_DATED_SIGNAL_LABEL", "TIMING", "Signal's date label is later than its observation date.", "ERROR")
            elif lag > 0:
                audit.issue(event, "SIGNAL_OBSERVED_AFTER_MARKET_DATE", "TIMING", f"Signal observed {lag} calendar day(s) after its date label; session timing is not inferred.")
            if not row["ticker"] or not row["strategy_id"]:
                audit.issue(event, "SIGNAL_IDENTITY_UNAVAILABLE", "EVIDENCE", "Instrument or strategy identity is unavailable.")
            if row["parameter_fingerprint"] is None:
                audit.issue(event, "MISSING_PARAMETER_FINGERPRINT", "EVIDENCE", "No original configuration fingerprint was journaled.")
            if row["protocol_sha256"] is None:
                audit.issue(event, "MISSING_PROTOCOL_HASH", "EVIDENCE", "No original protocol hash was journaled.")
            signals[event["event_id"]] = row
        elif kind == "SCAN":
            count, selected = payload.get("qualified_signals"), payload.get("selected")
            valid = isinstance(count, int) and not isinstance(count, bool) and count >= 0
            if not valid:
                audit.issue(event, "INVALID_SCAN_COUNT", "SCAN", "Qualified-signal count is unavailable or invalid.", "ERROR")
            if (not isinstance(selected, int) or isinstance(selected, bool) or selected < 0
                    or (valid and selected > count)):
                audit.issue(event, "INVALID_SCAN_SELECTION_COUNT", "SCAN", "Selection count is unavailable, negative or exceeds the qualified count.")
            market_day = _label_date(payload.get("market_date"))
            if market_day is None:
                audit.issue(event, "INVALID_SCAN_MARKET_DATE", "SCAN", "Completed market-date label is unavailable.")
            elif market_day > event["observed_date"]:
                audit.issue(event, "FUTURE_DATED_SCAN_MARKET_DATE", "TIMING", "Market-date label is later than the scan observation date.", "ERROR")
            scans.append({"event_id": event["event_id"], "sequence": event["sequence"],
                          "recorded_at": event["recorded_at"], "observed_date": event["observed_date"],
                          "market_date": market_day, "mode": payload.get("mode"), "regime": payload.get("regime"),
                          "qualified_signals": count if valid else None, "selected": selected,
                          "zero_signal_observation": valid and count == 0})
        elif kind in ("PAPER_FILL", "PAPER_EXIT"):
            trade_id = payload.get("trade_id")
            if not isinstance(trade_id, str) or not trade_id:
                audit.issue(event, "MISSING_TRADE_ID", "LINK", "A paper execution event has no trade ID.", "ERROR")
                stage = "entry" if kind == "PAPER_FILL" else "exit"
                audit.cost_fields(event, stage)
                audit.event_time(event, "entry_ts" if stage == "entry" else "exit_ts", stage)
            else:
                (fill_groups if kind == "PAPER_FILL" else exit_groups)[trade_id].append(event)
        else:
            audit.issue(event, "UNSUPPORTED_EVENT_TYPE", "SCOPE", f"Event type {kind!r} is retained only in the source digest.", "INFO")

    trades = []
    for trade_id in sorted(set(fill_groups) | set(exit_groups)):
        fills, exits = fill_groups[trade_id], exit_groups[trade_id]
        row = {"trade_id": trade_id, "fill_event_ids": [e["event_id"] for e in fills],
               "exit_event_ids": [e["event_id"] for e in exits], "status": "OPEN" if not exits else "CLOSED",
               "fill_count": len(fills), "exit_count": len(exits), "linked_signal_id": None,
               "valid_signal_link": False, "signal_precedes_entry": None,
               "computed_gross_pnl_inr": None, "computed_net_pnl_inr": None, "computed_realised_r": None}
        for group, label in ((fills, "FILL"), (exits, "EXIT")):
            if len(group) > 1:
                row["status"] = "AMBIGUOUS"
                for item in group:
                    audit.issue(item, f"MULTIPLE_{label}S_FOR_TRADE", "LINK", "Full-fill/full-exit lifecycle is ambiguous; no event is chosen.", "ERROR")
                    stage = "entry" if label == "FILL" else "exit"
                    audit.cost_fields(item, stage)
                    audit.event_time(item, "entry_ts" if stage == "entry" else "exit_ts", stage)
        fill = fills[0] if len(fills) == 1 else None
        exit_event = exits[0] if len(exits) == 1 else None
        entry_time = None
        signal = None
        if fill:
            payload = fill["payload"]
            row.update(ticker=_symbol(payload.get("ticker")), strategy_id=_strategy(payload),
                       fill_recorded_at=fill["recorded_at"], fill_observed_date=fill["observed_date"], entry_ts=payload.get("entry_ts"),
                       entry_price=payload.get("entry_price"), shares=payload.get("shares"), direction=payload.get("direction"))
            row.update(audit.cost_fields(fill, "entry"))
            entry_time, delay = audit.event_time(fill, "entry_ts", "entry")
            row["entry_recording_delay_seconds"] = delay
            signal_id = payload.get("signal_event_id")
            row["linked_signal_id"] = signal_id
            signal = signal_events.get(signal_id) if isinstance(signal_id, str) else None
            if not signal:
                audit.issue(fill, "SIGNAL_LINK_NOT_FOUND" if signal_id else "UNLINKED_FILL", "LINK", "No explicit link to an included signal; ticker/date similarity is not a link.", "ERROR")
            else:
                valid = (bool(row["ticker"] and row["strategy_id"]) and row["ticker"] == signals[signal_id]["ticker"]
                         and row["strategy_id"] == signals[signal_id]["strategy_id"])
                row["valid_signal_link"] = valid
                if not valid:
                    audit.issue(fill, "SIGNAL_IDENTITY_MISMATCH", "LINK", "Signal and fill instrument/strategy do not match.", "ERROR")
                else:
                    signals[signal_id]["linked_trade_ids"].append(trade_id)
                if signal["sequence"] >= fill["sequence"]:
                    audit.issue(fill, "SIGNAL_RECORDED_AFTER_FILL", "TIMING", "Signal was not earlier in journal order.", "ERROR")
                if entry_time is not None:
                    row["signal_precedes_entry"] = signal["recorded"] <= entry_time
                    if not row["signal_precedes_entry"]:
                        audit.issue(fill, "SIGNAL_OBSERVED_AFTER_ENTRY", "TIMING", "Signal first-observed time is after the supplied entry time.", "ERROR")
                if signal["payload"].get("selected") is False:
                    audit.issue(fill, "FILL_FROM_UNSELECTED_SIGNAL", "EVIDENCE", "The first-observed signal was not selected.")
                if signal["payload"].get("watchlist") is True:
                    audit.issue(fill, "FILL_FROM_WATCHLIST_SIGNAL", "EVIDENCE", "The first-observed signal was a watchlist observation.")
            bounds = signal["payload"].get("entry_bound") if signal else payload.get("entry_bound")
            row["entry_bounds_source"] = "SIGNAL" if signal else "FILL_DECLARATION"
            price = _number(payload.get("entry_price"), 0)
            if (isinstance(bounds, list) and len(bounds) == 2 and all(_number(b, 0) is not None for b in bounds)
                    and bounds[0] <= bounds[1] and price is not None):
                row["within_entry_bounds"] = bounds[0] <= price <= bounds[1]
                if not row["within_entry_bounds"]:
                    audit.issue(fill, "ENTRY_OUTSIDE_BOUNDS", "EVIDENCE", "Fill price is outside the recorded entry bounds.", "ERROR")
            else:
                row["within_entry_bounds"] = None
                audit.issue(fill, "ENTRY_BOUNDS_UNAVAILABLE", "EVIDENCE", "Original entry-bound compliance cannot be established.")
        if exit_event:
            payload = exit_event["payload"]
            row.update(exit_recorded_at=exit_event["recorded_at"], exit_observed_date=exit_event["observed_date"], exit_ts=payload.get("exit_ts"),
                       exit_price=payload.get("exit_price"), exit_reason=payload.get("exit_reason"),
                       exit_signal_event_id=payload.get("signal_event_id"),
                       recorded_net_pnl_inr=payload.get("net_pnl"), recorded_realised_r=payload.get("realised_r"))
            row.setdefault("ticker", _symbol(payload.get("ticker")))
            row.setdefault("strategy_id", _strategy(payload))
            row.update(audit.cost_fields(exit_event, "exit"))
            exit_time, delay = audit.event_time(exit_event, "exit_ts", "exit")
            row["exit_recording_delay_seconds"] = delay
            costs = _number(payload.get("costs"), 0)
            row["round_trip_costs_inr"] = costs
            if costs is None:
                audit.issue(exit_event, "MISSING_OR_INVALID_ROUND_TRIP_COSTS", "COST", "Net P&L is unavailable without a finite nonnegative total.", "ERROR")
            if all(row.get(f"exit_{key}_inr") is not None for key in ("fees", "slippage")) and costs is not None:
                if row["exit_fees_inr"] + row["exit_slippage_inr"] > costs + .01:
                    audit.issue(exit_event, "COST_COMPONENTS_EXCEED_TOTAL", "COST", "Exit fee/slippage annotations exceed the round-trip total.", "ERROR")
            if not fill:
                row["status"] = "ORPHAN_EXIT" if not fills else "AMBIGUOUS"
                audit.issue(exit_event, "EXIT_WITHOUT_UNIQUE_FILL", "LINK", "Exit cannot be reconciled to exactly one known fill.", "ERROR")
            else:
                if (_symbol(payload.get("ticker")) != row["ticker"] or _strategy(payload) != row["strategy_id"]
                        or payload.get("signal_event_id") != row["linked_signal_id"]):
                    audit.issue(exit_event, "EXIT_LINK_MISMATCH", "LINK", "Exit instrument, strategy or signal link differs from the fill.", "ERROR")
                if exit_event["sequence"] <= fill["sequence"]:
                    audit.issue(exit_event, "EXIT_RECORDED_BEFORE_FILL", "TIMING", "Exit was journaled before its fill.", "ERROR")
                if exit_time is not None and entry_time is not None and exit_time < entry_time:
                    audit.issue(exit_event, "EXIT_BEFORE_ENTRY", "TIMING", "Supplied exit time precedes entry.", "ERROR")
                entry, exit_price = _number(row["entry_price"], 0), _number(payload.get("exit_price"), 0)
                shares = row["shares"]
                if (entry is not None and entry > 0 and exit_price is not None and exit_price > 0
                        and isinstance(shares, int) and not isinstance(shares, bool) and shares > 0
                        and _number(shares, 1) is not None
                        and row["direction"] in ("LONG", "SHORT")):
                    gross = (exit_price - entry) * shares * (1 if row["direction"] == "LONG" else -1)
                    row["computed_gross_pnl_inr"] = gross if math.isfinite(gross) else None
                    if not math.isfinite(gross):
                        audit.issue(exit_event, "NONFINITE_PNL", "INTEGRITY", "Fill arithmetic exceeds finite numeric range.", "ERROR")
                    if costs is not None and math.isfinite(gross - costs):
                        net = gross - costs  # Fees/slippage annotate this total; do not subtract again.
                        row["computed_net_pnl_inr"] = net
                        recorded_net = _number(payload.get("net_pnl"))
                        if recorded_net is None or not math.isclose(net, recorded_net, abs_tol=.01, rel_tol=1e-9):
                            audit.issue(exit_event, "NET_PNL_MISMATCH_OR_UNAVAILABLE", "INTEGRITY", "Stored net P&L does not reconcile to gross less total costs.", "ERROR")
                        stop = _number(fill["payload"].get("stop_loss"), 0)
                        risk = abs(entry - stop) * shares if stop is not None else 0
                        if risk > 0 and math.isfinite(risk):
                            row["computed_realised_r"] = net / risk
                            stored_r = _number(payload.get("realised_r"))
                            if stored_r is None or not math.isclose(net / risk, stored_r, abs_tol=1e-6, rel_tol=1e-9):
                                audit.issue(exit_event, "REALISED_R_MISMATCH_OR_UNAVAILABLE", "INTEGRITY", "Stored net risk-unit return does not reconcile.", "ERROR")
                        else:
                            audit.issue(exit_event, "INVALID_INITIAL_RISK", "INTEGRITY", "A finite positive initial risk is unavailable.", "ERROR")
                else:
                    audit.issue(exit_event, "INVALID_FILL_FOR_PNL", "INTEGRITY", "Price, quantity or direction cannot support P&L reconciliation.", "ERROR")
        trades.append(row)

    for identity, signal in signals.items():
        signal["fill_status"] = "LINKED_FILL" if signal["linked_trade_ids"] else "NO_LINKED_FILL"
        if len(signal["linked_trade_ids"]) > 1:
            audit.issue(signal_events[identity], "SIGNAL_LINKED_TO_MULTIPLE_TRADES", "LINK", "Review repeated use of one signal; partial-fill accounting is not supported.")
    observed = [e for e in known if e["observed_date"] == day.isoformat()]
    daily_counts = Counter(e["event_type"] for e in observed)
    scan_days: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scan in scans:
        if scan["market_date"]:
            scan_days[scan["market_date"]].append(scan)
    market_dates = []
    for market_day, rows in sorted(scan_days.items()):
        counts = [r["qualified_signals"] for r in rows]
        state = ("COUNT_UNAVAILABLE" if None in counts else "ZERO_SIGNALS_IN_ALL_RECORDED_SCANS" if all(c == 0 for c in counts)
                 else "MIXED_ZERO_AND_NONZERO_SCANS" if 0 in counts else "SIGNALS_REPORTED")
        market_dates.append({"market_date": market_day, "scan_observations": len(rows), "status": state})
    today_signals = [s for s in signals.values() if s["observed_date"] == day.isoformat()]
    has_review = any(e["severity"] != "INFO" for e in audit.issues)
    return {
        "schema_version": 1, "report_date": day.isoformat(), "timezone": "Asia/Kolkata", "as_of": cutoff.isoformat(),
        "status": "NEEDS_REVIEW" if has_review else "NO_NEW_OBSERVATIONS" if not observed else "JOURNAL_LINKS_RECONCILED",
        "recording_delay_threshold_seconds": max_recording_delay_seconds,
        "source": {"included_sequence_max": max((e["sequence"] for e in digest_rows if isinstance(e.get("sequence"), int)), default=0),
                   "included_raw_rows": len(digest_rows), "unassigned_time_rows": sum(e["code"] == "UNASSIGNED_JOURNAL_TIME" for e in audit.issues),
                   "snapshot_sha256": hashlib.sha256(json.dumps(digest_rows, sort_keys=True, default=str).encode()).hexdigest()},
        "summary": {"daily_event_counts": dict(daily_counts), "new_signals": len(today_signals),
                    "new_paper_fills": daily_counts["PAPER_FILL"], "new_paper_exits": daily_counts["PAPER_EXIT"],
                    "daily_scan_observations": daily_counts["SCAN"],
                    "daily_zero_signal_scan_observations": sum(e["observed_date"] == day.isoformat() and e["zero_signal_observation"] for e in scans),
                    "signals_as_of": len(signals), "signals_without_linked_fill_as_of": sum(not s["linked_trade_ids"] for s in signals.values()),
                    "open_trades_as_of": sum(t["status"] == "OPEN" for t in trades),
                    "closed_trades_as_of": sum(t["status"] == "CLOSED" for t in trades),
                    "exception_counts": dict(Counter(e["code"] for e in audit.issues)),
                    "daily_exception_counts": dict(Counter(e["code"] for e in audit.issues if e["observed_date"] == day.isoformat())),
                    "excluded_nonpaper_events_as_of": dict(excluded)},
        "scans": scans, "market_dates": market_dates, "signals": list(signals.values()), "trades": trades,
        "exceptions": audit.issues,
        "limitations": [
            "Daily counts use first-observed journal dates in IST, not supplied signal/fill dates. Earlier links and outstanding items are reconstructed through the cutoff.",
            "No-scan days are not certified exchange no-trade days. Scan observations have no scan-to-signal ID; repeated scan counts are not unique signal counts.",
            "Signals are observations, never assumed fills. Timing checks and stored hashes do not prove independent prospective performance.",
            "This reconciles journal relationships, not broker statements or historical mutable execution/open-position tables. No replay or executed outcomes are pooled.",
            "Net P&L uses the reported round-trip total once, even when its components or cost basis remain incomplete; it is not a verified performance aggregate.",
        ],
    }


def reconcile_paper_day(path: Path, day: date, *, as_of: datetime | None = None,
                        max_recording_delay_seconds: float = 900) -> dict[str, Any]:
    report = reconcile_paper_events(read_paper_journal(path), day, as_of=as_of,
                                    max_recording_delay_seconds=max_recording_delay_seconds)
    report["source"]["database"] = str(path.resolve())
    return report


def write_daily_report(report: dict[str, Any], directory: Path) -> None:
    """Derived exports only; refuse any target resolving to the source database or WAL files."""
    filenames = ["report.json", "report.md", "scans.csv", "signals.csv", "trades.csv", "exceptions.csv", "market_dates.csv"]
    source = report["source"].get("database")
    if source:
        protected = {Path(str(source) + suffix).resolve() for suffix in ("", "-wal", "-shm", "-journal")}
        if any((directory / name).resolve() in protected for name in filenames):
            raise ValueError("Report output would overwrite source database files")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for key in ("scans", "signals", "trades", "exceptions", "market_dates"):
        rows = report[key]
        columns = list(dict.fromkeys(name for row in rows for name in row)) or ["event_id"]
        with (directory / f"{key}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
                                 for key, value in row.items()})
    summary = report["summary"]
    lines = [f"# Daily paper reconciliation — {report['report_date']} (IST)", "",
             f"Status: **{report['status']}**. Observation cutoff: {report['as_of']}.", "",
             f"New observed signals: **{summary['new_signals']}**. Scan observations: **{summary['daily_scan_observations']}** "
             f"({summary['daily_zero_signal_scan_observations']} zero-signal observations).", "",
             f"Journal events observed this day: {json.dumps(summary['daily_event_counts'], sort_keys=True)}.", "",
             f"As of cutoff: {summary['open_trades_as_of']} open paper trades; {summary['closed_trades_as_of']} closed; "
             f"{summary['signals_without_linked_fill_as_of']} signals without an explicit linked fill.", "",
             "## Exceptions", "", "| Code | Count |", "|---|---:|"]
    lines += [f"| {code} | {count} |" for code, count in sorted(summary["exception_counts"].items())] or ["| None | 0 |"]
    def table(columns: list[tuple[str, str]], rows: list[dict[str, Any]]) -> list[str]:
        def cell(value: Any) -> str:
            if value is None:
                return "Unavailable"
            if isinstance(value, list):
                value = ", ".join(str(item) for item in value) or "None"
            return html.escape(str(value)).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
        return (["| " + " | ".join(label for _, label in columns) + " |",
                 "|" + "---|" * len(columns)] +
                ["| " + " | ".join(cell(row.get(key)) for key, _ in columns) + " |" for row in rows]) if rows else ["None observed in this section."]

    lines += ["", "## Signals first observed this day", ""] + table(
        [("ticker", "Ticker"), ("strategy_id", "Strategy"), ("first_observed_at", "First observed"),
         ("selected", "Selected"), ("linked_trade_ids", "Linked trades")],
        [row for row in report["signals"] if row["observed_date"] == report["report_date"]])
    lines += ["", "## Paper lifecycles observed this day or still open", ""] + table(
        [("trade_id", "Trade"), ("ticker", "Ticker"), ("linked_signal_id", "Signal event"), ("status", "Lifecycle"),
         ("round_trip_costs_inr", "Reported total cost INR"), ("computed_net_pnl_inr", "Recomputed net INR")],
        [row for row in report["trades"] if row["status"] == "OPEN" or row.get("fill_observed_date") == report["report_date"]
         or row.get("exit_observed_date") == report["report_date"]])
    lines += ["", "## Exception details observed this day", ""] + table(
        [("event_id", "Event"), ("code", "Exception"), ("detail", "Detail")],
        [row for row in report["exceptions"] if row["observed_date"] == report["report_date"]])
    lines += ["", "Exception counts above and CSV files include all unresolved/history checks through the cutoff. "
              "Earlier closed lifecycles and unassigned-time rows remain in report.json and the CSV exports. "
              "Those exports also preserve original strategy/configuration hashes and linked event IDs.",
              "", "## Interpretation", ""] + [f"- {note}" for note in report["limitations"]]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
