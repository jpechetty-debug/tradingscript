import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pandas as pd
import pytest

from core.coverage_audit import FeedAudit, MembershipAudit, audit_coverage, timing_rows
from scripts import audit_research_coverage as cli


DATES = pd.bdate_range("2022-03-29", periods=6)


def prices(dates=DATES):
    return pd.DataFrame({"Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 1000.}, index=dates)


def observation(date):
    return pd.Timestamp(date).tz_localize("Asia/Kolkata") + pd.Timedelta(hours=18)


def legacy_members():
    return pd.DataFrame({"ticker": ["A", "MISSING"], "start": ["2021-01-01"] * 2, "end": [None, None]})


def audit(membership=None, history=None, **kwargs):
    return audit_coverage(history or {"A.NS": prices(), "^NSEI": prices()}, DATES,
                          membership or MembershipAudit(legacy_members(), DATES[2]), {},
                          benchmark="^NSEI", price_lookback=2, **kwargs)


def test_legacy_frozen_membership_never_certifies_point_in_time_coverage():
    result = audit()
    assert result.summary["membership_status_sessions"] == {
        "UNAVAILABLE_LEGACY_OBSERVATION_TIMING": 3, "UNAVAILABLE_STALE_MEMBERSHIP": 3}
    assert result.daily.declared_members.tolist() == [2] * 6
    assert result.daily.members_with_valid_bar.tolist() == [1] * 6
    assert result.daily.declared_member_price_coverage.tolist() == [.5] * 6
    assert result.daily.known_at_member_count.isna().all()
    assert result.daily.price_asof_status.str.startswith("UNAVAILABLE").all()
    missing = result.unavailable.loc[(result.unavailable.component == "prices") & (result.unavailable.entity == "MISSING")]
    assert missing.iloc[0].audited_sessions == 6
    assert missing.iloc[0].reason == "UNAVAILABLE_NO_PRICE_SERIES"


def test_membership_end_is_exclusive():
    membership = MembershipAudit(pd.DataFrame({"ticker": ["A.NS"], "start": ["2022-03-29"], "end": ["2022-03-31"]}))
    assert membership.members(DATES[1], observation(DATES[1]))[0] == {"A"}
    assert membership.members(DATES[2], observation(DATES[2]))[0] == set()


def test_overlap_preserves_diagnostic_denominator_and_separate_integrity_intervals():
    frame = pd.DataFrame({"ticker": ["A", "A"], "start": ["2021-01-01", "2022-04-01"], "end": [None, None]})
    result = audit(MembershipAudit(frame, DATES[2]))
    assert result.daily.declared_members.tolist() == [1] * 6
    assert result.daily.membership_integrity.tolist() == ["NO_ACTIVE_OVERLAP_DETECTED"] * 3 + ["OVERLAPPING_INTERVALS"] * 3
    assert result.daily.membership_status.iloc[-1] == "UNAVAILABLE_STALE_MEMBERSHIP"
    interval = result.unavailable.loc[result.unavailable.component == "membership_integrity"].iloc[0]
    assert interval.entity == "A" and interval.start == "2022-04-01" and interval.end == "2022-04-05"
    assert interval.audited_sessions == 3


@pytest.mark.parametrize("frame", [
    pd.DataFrame({"ticker": ["A"], "start": ["bad"], "end": [None]}),
    pd.DataFrame({"ticker": ["A"], "start": ["2022-04-01"], "end": ["2022-03-29"]}),
    pd.concat([legacy_members(), legacy_members()], ignore_index=True),
])
def test_malformed_membership_is_unavailable(frame):
    result = audit(MembershipAudit(frame))
    assert result.summary["membership_mode"] == "INVALID"
    assert result.daily.membership_status.eq("UNAVAILABLE_INVALID_MEMBERSHIP").all()
    assert result.daily.declared_member_price_coverage.isna().all()


def dated_members():
    return pd.DataFrame([{"ticker": "A", "effective_date": "2022-03-29", "action": "ADD",
                          "published_at": "2022-03-28T12:00:00+05:30", "ingested_at": "2022-03-31T17:00:00+05:30", "source": "fixture"}])


def test_membership_availability_uses_ingestion_not_effective_or_publication_date():
    result = audit(MembershipAudit(dated_members()))
    assert result.daily.declared_members.tolist() == [0, 0, 1, 1, 1, 1]
    assert result.daily.membership_status.iloc[2] == "UNVERIFIED_BASELINE_AND_CONTINUITY"
    assert result.summary["status"] == "UNAVAILABLE_POINT_IN_TIME_COMPLETENESS"


def test_future_membership_revision_cannot_change_earlier_row_set():
    original = dated_members()
    future = original.iloc[0].to_dict()
    future.update(ticker="B", ingested_at="2022-04-05T17:00:00+05:30")
    first = audit(MembershipAudit(original))
    second = audit(MembershipAudit(pd.concat([original, pd.DataFrame([future])], ignore_index=True)))
    pd.testing.assert_frame_equal(first.daily.iloc[:-1], second.daily.iloc[:-1])
    assert second.daily.declared_members.iloc[-1] == 2


def test_new_expired_membership_revision_does_not_resurrect_old_interval():
    frame = pd.DataFrame([
        {"ticker": "A", "start": "2022-03-29", "end": "2022-04-10", "published_at": "2022-03-28T12:00:00Z",
         "ingested_at": "2022-03-28T12:01:00Z", "source": "initial"},
        {"ticker": "A", "start": "2022-03-29", "end": "2022-03-31", "published_at": "2022-04-01T12:00:00Z",
         "ingested_at": "2022-04-01T12:01:00Z", "source": "revision"},
    ])
    result = audit(MembershipAudit(frame))
    assert result.daily.declared_members.tolist() == [1, 1, 1, 0, 0, 0]


def test_prices_are_not_forward_filled_and_invalid_ohlc_breaks_lookback():
    frame = prices().drop(DATES[3])
    frame.loc[DATES[4], "High"] = 90.
    result = audit(history={"A": frame, "^NSEI": prices()})
    assert result.daily.members_with_valid_bar.tolist() == [1, 1, 1, 0, 0, 1]
    assert result.daily.members_with_price_lookback.tolist() == [0, 1, 1, 0, 0, 0]


def test_start_filter_preserves_prior_warmup_without_future_leakage():
    history = {"A": prices(), "^NSEI": prices()}
    result = audit_coverage(history, DATES[-2:], MembershipAudit(legacy_members()), {}, benchmark="^NSEI", price_lookback=3)
    assert result.daily.members_with_price_lookback.tolist() == [1, 1]
    history["A"].loc[DATES[-1], "Close"] = 0.
    changed = audit_coverage(history, DATES[-2:], MembershipAudit(legacy_members()), {}, benchmark="^NSEI", price_lookback=3)
    pd.testing.assert_frame_equal(result.daily.iloc[:1], changed.daily.iloc[:1])


def test_missing_benchmark_session_is_explicit():
    result = audit(history={"A": prices(), "^NSEI": prices().drop(DATES[3])})
    assert result.summary["missing_benchmark_sessions"] == ["2022-04-01"]
    assert result.unavailable.component.eq("benchmark").any()


@pytest.mark.parametrize("published,ingested,source", [
    ("2022-03-28T12:00:00", "2022-03-28T13:00:00Z", "fixture"),
    ("2022-03-28T12:00:00Z", "2022-03-28T11:00:00Z", "fixture"),
    ("2022-03-28T12:00:00Z", "2022-03-28T13:00:00Z", ""),
])
def test_bad_ingestion_claims_are_audited_not_replaced(published, ingested, source):
    row = timing_rows("test", pd.DataFrame([{"published_at": published, "ingested_at": ingested, "source": source}]))[0]
    assert row["status"] == "INVALID_OR_MISSING_TIMING"
    assert row["error"] and row["csv_row"] == 2


def test_missing_feeds_never_become_clear_or_zero_observations():
    result = audit()
    assert result.feed_daily.status.eq("UNAVAILABLE_OR_PARTIAL").all()
    assert all(v == {"UNAVAILABLE_MISSING_FEED": 1} for v in result.feed_daily.reason_counts)
    assert result.ingestion.loc[result.ingestion.component == "prices", "ingested_at"].isna().all()


def surveillance(ingested="2022-03-29T17:00:00+05:30", state="ASM"):
    return {"ticker": "A", "effective_date": "2022-03-29", "valid_until": "2022-04-10", "state": state,
            "published_at": "2022-03-29T16:00:00+05:30", "ingested_at": ingested, "source": "fixture"}


def test_late_ingestion_cannot_clear_an_earlier_decision():
    feed = FeedAudit("surveillance", pd.DataFrame([surveillance("2022-03-29T19:00:00+05:30", "CLEAR")]))
    assert feed.states(DATES[0], observation(DATES[0]), DATES, {"A"}, {})["A"] == "UNAVAILABLE_NO_KNOWN_OBSERVATION"
    assert feed.states(DATES[1], observation(DATES[1]), DATES, {"A"}, {})["A"] == "OBSERVED_CLEAR"
    assert feed.timing[0]["ingestion_delay_seconds"] == 10800.


def test_known_restriction_is_observed_data_and_unknown_is_unavailable():
    feed = FeedAudit("surveillance", pd.DataFrame([surveillance()]))
    assert feed.states(DATES[0], observation(DATES[0]), DATES, {"A"}, {})["A"] == "OBSERVED_ASM"
    feed = FeedAudit("surveillance", pd.DataFrame([surveillance(state="UNKNOWN")]))
    assert feed.states(DATES[0], observation(DATES[0]), DATES, {"A"}, {})["A"] == "UNAVAILABLE_UNKNOWN_STATE"


def test_expired_feed_revision_does_not_resurrect_prior_clearance():
    old = surveillance()
    new = surveillance("2022-03-31T17:00:00+05:30")
    new.update(valid_until="2022-03-30", published_at="2022-03-31T16:00:00+05:30")
    feed = FeedAudit("surveillance", pd.DataFrame([old, new]))
    assert feed.states(DATES[2], observation(DATES[2]), DATES, {"A"}, {})["A"] == "UNAVAILABLE_EXPIRED"


def test_one_invalid_feed_row_quarantines_file_and_identifies_record():
    good, bad = surveillance(), surveillance()
    bad.update(ticker="B", ingested_at="2022-03-29T15:00:00+05:30")
    feed = FeedAudit("surveillance", pd.DataFrame([good, bad]))
    assert feed.error and feed.table is None
    assert feed.timing[1]["csv_row"] == 3 and feed.timing[1]["error"]
    assert feed.timing[0]["feed_validation"] == "INVALID_FILE"
    assert feed.states(DATES[0], observation(DATES[0]), DATES, {"A"}, {}) == {"*": "UNAVAILABLE_INVALID_FEED"}


@pytest.mark.parametrize("kind,count", [("flows", 20), ("delivery", 25), ("indices", 2)])
def test_session_feed_window_needs_every_asof_observation(kind, count):
    sessions = pd.bdate_range("2022-01-03", periods=count)
    rows = []
    for date in sessions:
        row = {"session": date.date().isoformat(), "published_at": (date.tz_localize("UTC") + pd.Timedelta(hours=10)).isoformat(),
               "ingested_at": (date.tz_localize("UTC") + pd.Timedelta(hours=11)).isoformat(), "source": "fixture"}
        if kind == "indices":
            row.update(index="IDX", close=100.)
        elif kind == "flows":
            row.update(ticker="A", fii_net_inr=1., dii_net_inr=1.)
        else:
            row.update(ticker="A", delivery_qty=1., total_qty=2., turnover=100.)
        rows.append(row)
    frame = pd.DataFrame(rows)
    feed = FeedAudit(kind, frame)
    key, windows = ("IDX", {"IDX": count}) if kind == "indices" else ("A", {})
    assert feed.states(sessions[-1], observation(sessions[-1]), sessions, {"A"}, windows)[key] == "OBSERVED_WINDOW"
    frame.loc[len(frame) - 1, "ingested_at"] = (sessions[-1].tz_localize("Asia/Kolkata") + pd.Timedelta(hours=19)).isoformat()
    late = FeedAudit(kind, frame)
    assert late.states(sessions[-1], observation(sessions[-1]), sessions, {"A"}, windows)[key] == "UNAVAILABLE_SESSION_WINDOW"


def test_index_feed_requires_declared_index_universe():
    frame = pd.DataFrame([{"index": "IDX", "session": "2022-03-29", "close": 100.,
                           "published_at": "2022-03-29T16:00:00+05:30", "ingested_at": "2022-03-29T17:00:00+05:30", "source": "fixture"}])
    feed = FeedAudit("indices", frame)
    assert feed.states(DATES[0], observation(DATES[0]), DATES, {"A"}, {}) == {"*": "UNAVAILABLE_INDEX_UNIVERSE_MANIFEST"}


def test_fundamentals_staleness_is_not_forward_filled_as_current():
    frame = pd.DataFrame([{"ticker": "A", "period_end": "2021-06-30", "eps_growth": .2, "revenue_growth": .2,
                           "guidance_raised": True, "published_at": "2021-07-01T16:00:00Z", "ingested_at": "2021-07-01T17:00:00Z", "source": "fixture"}])
    feed = FeedAudit("fundamentals", frame)
    assert feed.states(DATES[0], observation(DATES[0]), DATES, {"A"}, {})["A"] == "UNAVAILABLE_STALE_PERIOD"


def test_duplicate_same_time_revisions_are_invalid():
    feed = FeedAudit("surveillance", pd.DataFrame([surveillance(), surveillance()]))
    assert feed.table is None and "revisions" in feed.error


@pytest.mark.parametrize("kind,expected,state", [("CONSENSUS", 1., "OBSERVED_CONSENSUS"),
                                                 ("CONSENSUS", 0., "UNAVAILABLE_CONSENSUS_OR_NONZERO_EXPECTATION"),
                                                 ("SEASONAL", 1., "UNAVAILABLE_CONSENSUS_OR_NONZERO_EXPECTATION")])
def test_earnings_require_usable_consensus_and_expire(kind, expected, state):
    row = {"ticker": "A", "period_end": "2021-12-31", "announced_at": "2022-03-29T16:00:00+05:30",
           "expected_at": "2022-03-28T12:00:00+05:30", "expectation_ingested_at": "2022-03-28T13:00:00+05:30",
           "actual_eps": 2., "expected_eps": expected, "expectation_kind": kind,
           "published_at": "2022-03-29T16:30:00+05:30", "ingested_at": "2022-03-29T17:00:00+05:30", "source": "fixture"}
    sessions = pd.bdate_range("2022-03-29", periods=23)
    feed = FeedAudit("earnings", pd.DataFrame([row]))
    assert feed.states(sessions[0], observation(sessions[0]), sessions, {"A"}, {})["A"] == state
    assert feed.states(sessions[-1], observation(sessions[-1]), sessions, {"A"}, {})["A"] == "UNAVAILABLE_STALE_EARNINGS"


def test_future_feed_revision_does_not_change_earlier_known_state():
    original = surveillance()
    future = surveillance("2022-04-05T17:00:00+05:30", "CLEAR")
    future["published_at"] = "2022-04-05T16:00:00+05:30"
    first = FeedAudit("surveillance", pd.DataFrame([original]))
    second = FeedAudit("surveillance", pd.DataFrame([original, future]))
    for date in DATES[:-1]:
        assert first.states(date, observation(date), DATES, {"A"}, {}) == second.states(date, observation(date), DATES, {"A"}, {})
    assert second.states(DATES[-1], observation(DATES[-1]), DATES, {"A"}, {})["A"] == "OBSERVED_CLEAR"


def test_audit_exports_are_deterministic_and_mtime_is_not_knowledge(tmp_path):
    snapshot = tmp_path / "prices.pkl"
    with snapshot.open("wb") as stream:
        pickle.dump({"A.NS": prices(), "^NSEI": prices()}, stream)
    membership = tmp_path / "members.csv"
    legacy_members().to_csv(membership, index=False)
    saved_run = tmp_path / "saved.json"
    saved_run.write_text(json.dumps({"provenance": {"snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(), "feed_sha256": {}}, "qualified": 0}))
    output = tmp_path / "audit"
    arguments = ["--snapshot", str(snapshot), "--membership", str(membership), "--membership-supported-through", "2022-03-31",
                 "--start", "2022-03-31", "--saved-run", str(saved_run), "--out", str(output), "--price-lookback", "2"]
    cli.main(arguments)
    first = (output / "report.json").read_bytes()
    os.utime(snapshot, (946684800., 946684800.))
    cli.main(arguments)
    assert first == (output / "report.json").read_bytes()
    report = json.loads(first)
    assert report["saved_runs"][0]["snapshot_hash_matches"]
    assert report["saved_runs"][0]["recorded_feed_hashes"] == {}
    daily = pd.read_csv(output / "daily_coverage.csv")
    assert daily.members_with_price_lookback.iloc[0] == 1
    for name in ["feed_coverage.csv", "ingestion_audit.csv", "unavailable_intervals.csv"]:
        assert not pd.read_csv(output / name).empty


def test_direct_cli_entry_point_works_outside_repository(tmp_path):
    result = subprocess.run([sys.executable, str(Path(cli.__file__).resolve()), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--membership-supported-through" in result.stdout
