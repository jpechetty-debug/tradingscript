from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest

from core.database import SqliteDatabase
from core.config import IST
from core.paper_ledger import PaperLedger
from core.paper_reconciliation import read_paper_journal, reconcile_paper_day, reconcile_paper_events, write_daily_report
from core.trades import TradeExit, TradeFill, TradeLifecycle


DAY = date(2026, 10, 2)
ASOF = datetime(2026, 10, 3, tzinfo=timezone.utc)


def event(sequence, kind, payload, stamp="2026-10-02T15:31:00+05:30", identity=None):
    return {"sequence": sequence, "event_id": identity or f"{kind.lower()}:{sequence}",
            "event_type": kind, "recorded_at": stamp, "payload": payload}


def signal(sequence=1, **changes):
    payload = {"ticker": "TEST", "strategy_version": "SWING_BREAKOUT_V1", "signal_time": "2026-10-01T00:00:00",
               "entry_bound": [99., 101.], "stop": 90., "target": 120., "selected": True, "watchlist": False,
               "parameter_fingerprint": "original-config", "research_protocol": {"sha256": "original-protocol"},
               "probability_status": "HEURISTIC_UNVALIDATED"}
    payload.update(changes)
    return event(sequence, "SIGNAL", payload, "2026-10-01T18:00:00+05:30", f"signal:{sequence}")


def fill(sequence=3, **changes):
    payload = {"trade_id": "trade-1", "ticker": "TEST.NS", "strategy_id": "SWING_BREAKOUT_V1",
               "signal_event_id": "signal:1", "source": "SIMULATED", "entry_price": 100., "stop_loss": 90.,
               "shares": 10, "direction": "LONG", "entry_ts": "2026-10-02T09:15:00+05:30",
               "fees_inr": 2., "slippage_inr": 3., "cost_basis": "MODELED"}
    payload.update(changes)
    return event(sequence, "PAPER_FILL", payload, "2026-10-02T09:16:00+05:30")


def exit_event(sequence=4, **changes):
    payload = {"trade_id": "trade-1", "ticker": "TEST", "strategy_version": "SWING_BREAKOUT_V1", "signal_event_id": "signal:1",
               "exit_ts": "2026-10-02T15:30:00+05:30", "exit_price": 110., "exit_reason": "MANUAL",
               "costs": 5., "fees_inr": 3., "slippage_inr": 2., "cost_basis": "MODELED", "net_pnl": 95., "realised_r": .95}
    payload.update(changes)
    return event(sequence, "PAPER_EXIT", payload)


def chain():
    return [signal(), event(2, "SCAN", {"market_date": "2026-10-01", "qualified_signals": 1, "selected": 1},
                            "2026-10-01T18:01:00+05:30"), fill(), exit_event()]


def report(events, **kwargs):
    return reconcile_paper_events(events, DAY, as_of=ASOF, **kwargs)


def codes(result):
    return result["summary"]["exception_counts"]


def test_linked_lifecycle_uses_first_observed_signal_and_deducts_costs_once():
    result = report(chain())
    assert result["status"] == "JOURNAL_LINKS_RECONCILED"
    assert result["summary"]["daily_event_counts"] == {"PAPER_FILL": 1, "PAPER_EXIT": 1}
    assert result["summary"]["new_signals"] == 0
    assert result["summary"]["closed_trades_as_of"] == 1
    row = result["trades"][0]
    assert row["valid_signal_link"] and row["signal_precedes_entry"] and row["within_entry_bounds"]
    assert row["computed_gross_pnl_inr"] == 100.
    assert row["computed_net_pnl_inr"] == 95.  # Not 85 after re-subtracting the cost annotations.
    assert row["computed_realised_r"] == .95
    assert result["signals"][0]["parameter_fingerprint"] == "original-config"
    assert result["signals"][0]["protocol_sha256"] == "original-protocol"
    assert result["signals"][0]["first_observed_at"] == "2026-10-01T18:00:00+05:30"
    assert result["signals"][0]["linked_trade_ids"] == ["trade-1"]


def test_historical_cutoff_excludes_later_fills_and_never_assumes_selection_was_filled():
    before = reconcile_paper_events(chain(), date(2026, 10, 1), as_of=ASOF)
    assert before["trades"] == []
    assert before["signals"][0]["fill_status"] == "NO_LINKED_FILL"
    assert before["summary"]["signals_without_linked_fill_as_of"] == 1
    assert before["source"]["included_sequence_max"] == 2
    assert before == reconcile_paper_events(chain()[:2], date(2026, 10, 1), as_of=ASOF)


def test_reporting_day_is_ist_observation_day_not_market_label():
    item = signal()
    item["recorded_at"] = "2026-10-01T20:00:00Z"  # 02 October, 01:30 IST.
    result = report([item])
    assert result["summary"]["new_signals"] == 1
    assert result["signals"][0]["market_date"] == "2026-10-01"
    assert result["signals"][0]["observed_date"] == "2026-10-02"
    assert result["signals"][0]["observation_lag_calendar_days"] == 1
    assert "SIGNAL_OBSERVED_AFTER_MARKET_DATE" in codes(result)


def test_late_exit_is_reported_on_observation_day_and_does_not_rewrite_prior_day():
    events = chain()
    events[-1]["recorded_at"] = "2026-10-03T12:00:00+05:30"
    assert report(events)["trades"][0]["status"] == "OPEN"
    result = reconcile_paper_events(events, date(2026, 10, 3), as_of=datetime(2026, 10, 4, tzinfo=timezone.utc))
    assert result["summary"]["daily_event_counts"] == {"PAPER_EXIT": 1}
    assert "DELAYED_EXIT_RECORDING" in codes(result)
    assert result["trades"][0]["exit_ts"].startswith("2026-10-02")


@pytest.mark.parametrize("field,value,expected", [
    ("entry_ts", "2026-10-02T08:00:00+05:30", "DELAYED_ENTRY_RECORDING"),
    ("entry_ts", "2026-10-02T10:00:00+05:30", "FUTURE_DATED_ENTRY"),
    ("entry_ts", "2026-10-02T09:15:00", "INVALID_ENTRY_TIME"),
    ("entry_ts", "invalid", "INVALID_ENTRY_TIME"),
])
def test_entry_timing_exceptions_are_recomputed_from_journal_time(field, value, expected):
    item = fill(**{field: value})
    item["payload"]["future_dated_event"] = False
    assert expected in codes(report([signal(), item]))


def test_signal_observed_after_supplied_entry_and_wrong_journal_order_are_flagged():
    item = signal(sequence=5)
    item["recorded_at"] = "2026-10-02T11:00:00+05:30"
    result = report([fill(signal_event_id="signal:5"), item])
    assert "SIGNAL_RECORDED_AFTER_FILL" in codes(result)
    assert "SIGNAL_OBSERVED_AFTER_ENTRY" in codes(result)
    assert result["trades"][0]["signal_precedes_entry"] is False


@pytest.mark.parametrize("changes,expected", [
    ({"exit_ts": "2026-10-02T16:00:00+05:30"}, "FUTURE_DATED_EXIT"),
    ({"exit_ts": "2026-10-01T16:00:00+05:30"}, "EXIT_BEFORE_ENTRY"),
    ({"exit_ts": None}, "INVALID_EXIT_TIME"),
    ({"ticker": "OTHER"}, "EXIT_LINK_MISMATCH"),
    ({"signal_event_id": "another"}, "EXIT_LINK_MISMATCH"),
    ({"net_pnl": 90.}, "NET_PNL_MISMATCH_OR_UNAVAILABLE"),
    ({"realised_r": 10.}, "REALISED_R_MISMATCH_OR_UNAVAILABLE"),
])
def test_exit_timing_identity_and_arithmetic_exceptions(changes, expected):
    assert expected in codes(report([signal(), fill(), exit_event(**changes)]))


def test_missing_costs_stay_unknown_explicit_zero_is_preserved():
    result = report([signal(), fill(fees_inr=None, slippage_inr=0., cost_basis="UNKNOWN"),
                     exit_event(costs=None, fees_inr=None, slippage_inr=None, cost_basis="UNKNOWN")])
    row = result["trades"][0]
    assert row["entry_fees_inr"] is None and row["entry_slippage_inr"] == 0.
    assert row["computed_net_pnl_inr"] is None
    assert {"MISSING_ENTRY_FEES", "UNKNOWN_ENTRY_COST_BASIS", "MISSING_EXIT_SLIPPAGE",
            "MISSING_OR_INVALID_ROUND_TRIP_COSTS"} <= codes(result).keys()


def test_reported_total_can_reconcile_with_incomplete_components_but_remains_flagged():
    result = report([signal(), fill(), exit_event(fees_inr=None, slippage_inr=None, cost_basis="UNKNOWN")])
    assert result["trades"][0]["computed_net_pnl_inr"] == 95.
    assert "MISSING_EXIT_FEES" in codes(result)
    assert result["status"] == "NEEDS_REVIEW"
    assert "COST_COMPONENTS_EXCEED_TOTAL" in codes(report([signal(), fill(), exit_event(fees_inr=20.)]))


def test_short_pnl_and_legacy_cost_aliases():
    short_fill = fill(direction="SHORT", stop_loss=110., fees_inr=None, slippage_inr=None)
    short_fill["payload"].update(fees=0., slippage=0.)
    result = report([signal(), short_fill, exit_event(exit_price=90.)])
    assert result["trades"][0]["computed_net_pnl_inr"] == 95.
    assert "MISSING_ENTRY_FEES" not in codes(result)


@pytest.mark.parametrize("kwargs,expected", [
    ({"signal_event_id": None}, "UNLINKED_FILL"),
    ({"signal_event_id": "missing"}, "SIGNAL_LINK_NOT_FOUND"),
    ({"ticker": "WRONG"}, "SIGNAL_IDENTITY_MISMATCH"),
    ({"strategy_id": "WRONG"}, "SIGNAL_IDENTITY_MISMATCH"),
])
def test_linking_uses_only_explicit_ids_and_matching_identity(kwargs, expected):
    result = report([signal(), fill(**kwargs)])
    assert expected in codes(result)
    assert result["signals"][0]["linked_trade_ids"] == []


def test_orphan_exit_duplicates_and_reused_signal_are_not_silently_matched():
    orphan = report([exit_event()])
    assert orphan["trades"][0]["status"] == "ORPHAN_EXIT"
    assert "EXIT_WITHOUT_UNIQUE_FILL" in codes(orphan)
    duplicate = report([signal(), fill(), fill(5), exit_event()])
    assert duplicate["trades"][0]["status"] == "AMBIGUOUS"
    assert duplicate["trades"][0]["computed_net_pnl_inr"] is None
    assert codes(duplicate)["MULTIPLE_FILLS_FOR_TRADE"] == 2
    duplicate_exit = report([signal(), fill(), exit_event(), exit_event(5)])
    assert duplicate_exit["trades"][0]["computed_net_pnl_inr"] is None
    reused = report([signal(), fill(), fill(5, trade_id="trade-2")])
    assert "SIGNAL_LINKED_TO_MULTIPLE_TRADES" in codes(reused)


def test_entry_plan_and_selection_exceptions():
    result = report([signal(selected=False, watchlist=True), fill(entry_price=105.)])
    assert {"ENTRY_OUTSIDE_BOUNDS", "FILL_FROM_UNSELECTED_SIGNAL", "FILL_FROM_WATCHLIST_SIGNAL"} <= codes(result).keys()
    assert report([signal(entry_bound=None), fill()])["trades"][0]["within_entry_bounds"] is None


def test_missing_trade_id_and_malformed_numbers_are_reported_without_crashing():
    assert "MISSING_TRADE_ID" in codes(report([fill(trade_id=None)]))
    for quantity in (True, -1, "10", 10**999):
        result = report([signal(), fill(shares=quantity), exit_event()])
        assert "INVALID_FILL_FOR_PNL" in codes(result)
    assert "INVALID_INITIAL_RISK" in codes(report([signal(), fill(stop_loss=100.), exit_event()]))
    assert "INVALID_ENTRY_FEES" in codes(report([signal(), fill(fees_inr=-1.)]))


def test_repeated_scan_observations_are_not_unique_signals_or_false_no_trade_days():
    events = [event(i, "SCAN", {"market_date": "2026-10-02", "qualified_signals": n, "selected": 0})
              for i, n in enumerate([0, 0, 1], 1)]
    result = report(events)
    assert result["summary"]["daily_scan_observations"] == 3
    assert result["summary"]["daily_zero_signal_scan_observations"] == 2
    assert result["summary"]["new_signals"] == 0
    assert result["market_dates"][0]["status"] == "MIXED_ZERO_AND_NONZERO_SCANS"
    assert report(events[:2])["market_dates"][0]["status"] == "ZERO_SIGNALS_IN_ALL_RECORDED_SCANS"
    assert "INVALID_SCAN_COUNT" in codes(report([event(1, "SCAN", {"qualified_signals": -1})]))


def test_pullback_and_v2_signal_shapes_retain_original_evidence_gaps():
    pullback = event(1, "SIGNAL", {"ticker": "TEST", "strategy_id": "LEADER_PULLBACK_V1", "signal_date": "2026-10-02",
                                   "mode": "PAPER_ONLY", "entry": "next_open"})
    v2 = event(2, "SIGNAL", {"ticker": "V2", "strategy_version": "SWING_ALPHA_V2", "signal_time": "2026-10-02T00:00:00",
                             "parameter_fingerprint": "v2-original", "provenance": {"protocol_sha256": "v2-protocol"}})
    result = report([pullback, v2])
    assert [s["strategy_id"] for s in result["signals"]] == ["LEADER_PULLBACK_V1", "SWING_ALPHA_V2"]
    assert result["signals"][0]["parameter_fingerprint"] is None
    assert result["signals"][1]["protocol_sha256"] == "v2-protocol"
    assert codes(result)["MISSING_PARAMETER_FINGERPRINT"] == 1
    assert all(s["fill_status"] == "NO_LINKED_FILL" for s in result["signals"])


def test_replay_executed_and_unknown_types_are_not_pooled_with_paper_outcomes():
    result = report([signal(source="executed"), event(2, "SIGNAL", {"mode": "replay"}),
                     event(3, "REJECTION", {"reason": "none"})])
    assert result["signals"] == result["trades"] == []
    assert result["summary"]["excluded_nonpaper_events_as_of"] == {"EXECUTED": 1, "REPLAY": 1}
    assert codes(result)["UNSUPPORTED_EVENT_TYPE"] == 1


def test_corrupt_payload_and_unknown_journal_time_remain_explicit():
    events = [event(1, "SIGNAL", "{bad"), event(2, "SIGNAL", []), event(3, "SIGNAL", {"x": float("nan")}),
              event(4, "SIGNAL", {}, "naive or missing"), event(5, "SIGNAL", {}, "2026-10-02T12:00:00")]
    result = report(events)
    assert result["signals"] == []
    assert codes(result)["INVALID_EVENT_PAYLOAD"] == 3
    assert codes(result)["UNASSIGNED_JOURNAL_TIME"] == 2
    assert result["status"] == "NEEDS_REVIEW"


def test_duplicate_identity_invalid_sequence_and_future_signal_date():
    item = signal()
    duplicate = deepcopy(item)
    duplicate["sequence"] = 2
    invalid_sequence = signal(3)
    invalid_sequence["sequence"] = "3"
    result = report([item, duplicate, invalid_sequence, signal(4, signal_time="2026-10-09")])
    assert {"DUPLICATE_OR_INVALID_EVENT_ID", "INVALID_SEQUENCE", "FUTURE_DATED_SIGNAL_LABEL"} <= codes(result).keys()


def test_future_scan_dates_and_impossible_selection_counts_require_review():
    result = report([event(1, "SCAN", {"market_date": "2026-10-03", "qualified_signals": 1, "selected": 2})])
    assert {"FUTURE_DATED_SCAN_MARKET_DATE", "INVALID_SCAN_SELECTION_COUNT"} <= codes(result).keys()


@pytest.mark.parametrize("kwargs", [{"as_of": datetime(2026, 10, 3)}, {"max_recording_delay_seconds": -1},
                                    {"max_recording_delay_seconds": float("inf")}])
def test_invalid_report_controls_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        reconcile_paper_events([], DAY, **kwargs)
    with pytest.raises(ValueError):
        reconcile_paper_events([], date(2027, 1, 1), as_of=ASOF)


def test_database_reader_is_unbounded_read_only_and_never_creates_missing_source(tmp_path):
    path = tmp_path / "journal.db"
    db = SqliteDatabase(path)
    with db.get_connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        for i in range(1003):
            db.append_paper_event(f"scan:{i}", "SCAN", {"market_date": "2026-10-01", "qualified_signals": 0, "selected": 0})
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = read_paper_journal(path)
    assert len(rows) == 1003 and rows[-1]["sequence"] == 1003
    today = datetime.now(timezone.utc).astimezone(IST).date()
    result = reconcile_paper_day(path, today)
    assert result["summary"]["daily_scan_observations"] == 1003
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    missing = tmp_path / "not-created.db"
    with pytest.raises(FileNotFoundError):
        read_paper_journal(missing)
    assert not missing.exists()


def test_real_lifecycle_events_are_read_without_mutating_the_journal(tmp_path):
    path = tmp_path / "real.db"
    db = SqliteDatabase(path)
    ledger = PaperLedger(db)
    ledger.record_scan("2026-10-01", "TREND_UP", 0, 0)
    linked = db.append_paper_event("original", "SIGNAL", signal()["payload"])
    now = datetime.now(timezone.utc)
    lifecycle = TradeLifecycle(db)
    fill_data = TradeFill(trade_id="real-paper", ticker="TEST", direction="LONG", source="SIMULATED", entry_price=100.,
                          entry_ts=now, stop_loss=90., target=120., shares=10, composite=.7, prob_win=.6, factors={},
                          strategy_id="SWING_BREAKOUT_V1", signal_event_id=linked["event_id"],
                          fees_inr=0., slippage_inr=0., cost_basis="OBSERVED")
    lifecycle.register(fill_data)
    lifecycle.close("real-paper", TradeExit(exit_price=110., exit_ts=now, costs=5., exit_reason="MANUAL",
                                           fees_inr=3., slippage_inr=2., cost_basis="OBSERVED"))
    before = db.fetch_paper_events()
    result = reconcile_paper_day(path, now.astimezone(IST).date())
    assert result["trades"][0]["computed_net_pnl_inr"] == 95.
    assert db.fetch_paper_events() == before
    assert db.count_executed_trades() == 1 and db.fetch_open_positions() == {}


def test_exports_empty_tables_and_refuses_source_overwrite(tmp_path):
    empty = report([])
    write_daily_report(empty, tmp_path / "out")
    assert json.loads((tmp_path / "out/report.json").read_text())["status"] == "NO_NEW_OBSERVATIONS"
    assert (tmp_path / "out/signals.csv").read_text().strip() == "event_id"
    assert "not certified exchange no-trade days" in (tmp_path / "out/report.md").read_text()
    empty["source"]["database"] = str(tmp_path / "out/report.json")
    with pytest.raises(ValueError, match="overwrite"):
        write_daily_report(empty, tmp_path / "out")


def test_populated_exports_preserve_links_null_costs_and_escape_markdown(tmp_path):
    new_signal = signal(sequence=5, ticker="<script>|TEST", signal_time="2026-10-02")
    new_signal["recorded_at"] = "2026-10-02T16:00:00+05:30"
    result = report([*chain()[:-1], exit_event(fees_inr=None), new_signal])
    write_daily_report(result, tmp_path)
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "trade-1" in markdown and "signal:1" in markdown
    assert "MISSING_EXIT_FEES" in markdown
    assert "&lt;SCRIPT&gt;\\|TEST" in markdown and "<SCRIPT>" not in markdown
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["trades"][0]["exit_fees_inr"] is None
    assert 'trade-1' in (tmp_path / "signals.csv").read_text(encoding="utf-8")


def test_cli_reads_existing_database_and_outputs_reports_without_rewrites(tmp_path):
    path = tmp_path / "paper.db"
    SqliteDatabase(path)
    before = path.read_bytes()
    out = tmp_path / "report"
    command = [sys.executable, "scripts/reconcile_paper_trading.py", "--db", str(path), "--date", "2026-10-02",
               "--as-of", "2026-10-03T00:00:00Z", "--out", str(out)]
    run = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stderr
    assert "NO_NEW_OBSERVATIONS" in run.stdout
    assert path.read_bytes() == before
    assert {p.name for p in out.iterdir()} == {"report.json", "report.md", "scans.csv", "signals.csv", "trades.csv", "exceptions.csv", "market_dates.csv"}
    command[command.index("--db")+1] = str(tmp_path / "missing.db")
    failed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
    assert failed.returncode != 0
    assert not (tmp_path / "missing.db").exists()


def test_legacy_database_without_journal_table_is_not_reported_as_empty(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE old_table (id INTEGER)")
    with pytest.raises(sqlite3.OperationalError, match="paper_events"):
        read_paper_journal(path)
