"""Read-only research coverage diagnostics; no data acquisition or strategy gates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .alpha_feeds import AlphaFeed, SCHEMAS, aware

TIMING = ("published_at", "ingested_at", "source")


def symbol(value: Any) -> str:
    result = str(value).strip().upper().removesuffix(".NS")
    if not result or result == "NAN":
        raise ValueError("Missing ticker")
    return result


def session_date(value: Any) -> pd.Timestamp:
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError("Session dates must be timezone-free calendar dates")
    return result


def timing_rows(kind: str, frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Retain each row's recording claims, never substitute file modification time."""
    result = []
    for number, row in enumerate(frame.to_dict("records"), start=2):
        error, lag = "", None
        published, ingested = None, None
        try:
            if not str(row.get("source", "")).strip() or pd.isna(row.get("source")):
                raise ValueError("Missing source")
            published, ingested = aware(row.get("published_at")), aware(row.get("ingested_at"))
            lag = float((ingested - published).total_seconds())
            if lag < 0:
                raise ValueError("Ingestion precedes publication")
        except (ValueError, TypeError, OverflowError) as exc:
            error = str(exc)
        result.append({"component": kind, "csv_row": number, "entity": str(row.get("ticker", row.get("index", "*"))),
                       "published_at_raw": row.get("published_at"), "ingested_at_raw": row.get("ingested_at"),
                       "published_at": published.isoformat() if published is not None else None,
                       "ingested_at": ingested.isoformat() if ingested is not None else None,
                       "source": row.get("source"), "ingestion_delay_seconds": lag,
                       "status": "INVALID_OR_MISSING_TIMING" if error else "TIMESTAMPS_VALID_NOT_PROVIDER_VERIFIED",
                       "error": error})
    return result


@dataclass
class Intervals:
    """Compress adjacent audited sessions, not assumed consecutive calendar days."""
    active: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)

    def observe(self, date: pd.Timestamp, gaps: set[tuple[str, str, str]]) -> None:
        for key in sorted(set(self.active) - gaps):
            self.rows.append(self.active.pop(key))
        for component, entity, reason in sorted(gaps):
            key = component, entity, reason
            if key not in self.active:
                self.active[key] = {"component": component, "entity": entity, "reason": reason,
                                    "start": date.date().isoformat(), "end": date.date().isoformat(), "audited_sessions": 0}
            self.active[key]["end"] = date.date().isoformat()
            self.active[key]["audited_sessions"] += 1

    def finish(self) -> pd.DataFrame:
        rows = self.rows + list(self.active.values())
        return pd.DataFrame(rows, columns=["component", "entity", "reason", "start", "end", "audited_sessions"])


class MembershipAudit:
    """Legacy [start,end) intervals or dated ADD/REMOVE observations.

    A supported-through declaration is only an upper bound. Neither it nor valid
    row timestamps establish a complete initial baseline/change history.
    """

    def __init__(self, frame: pd.DataFrame | None, supported_through: pd.Timestamp | None = None):
        self.frame = pd.DataFrame() if frame is None else frame.copy()
        self.supported_through = session_date(supported_through) if supported_through is not None else None
        self.error = ""
        self.mode = "MISSING" if frame is None else "INVALID"
        self.timed = False
        self.timing = timing_rows("membership", self.frame)
        if frame is None:
            return
        try:
            required = {"ticker", "start", "end"} if "start" in frame else {"ticker", "effective_date", "action"}
            if not required.issubset(frame):
                raise ValueError(f"Membership requires {sorted(required)}")
            data = frame.copy()
            data["ticker"] = [symbol(t) for t in data.ticker]
            self.mode = "INTERVALS" if "start" in data else "EVENTS"
            for col in (["start", "end"] if self.mode == "INTERVALS" else ["effective_date"]):
                data[col] = pd.to_datetime([pd.NaT if col == "end" and pd.isna(v) else session_date(v) for v in data[col]])
            if self.mode == "INTERVALS":
                if (data.end.notna() & (data.end <= data.start)).any():
                    raise ValueError("Membership end must be exclusive and later than start")
            else:
                data["action"] = data.action.astype(str).str.upper()
                if not data.action.isin(["ADD", "REMOVE"]).all():
                    raise ValueError("Membership action must be ADD or REMOVE")
            supplied = [c in data for c in TIMING]
            if any(supplied) and not all(supplied):
                raise ValueError("Membership timing requires published_at, ingested_at and source together")
            self.timed = all(supplied)
            keys = ["ticker", "start" if self.mode == "INTERVALS" else "effective_date"]
            if self.timed:
                if any(row["error"] for row in self.timing):
                    raise ValueError("Invalid membership observation timing; see ingestion audit")
                data["ingested_at"] = pd.to_datetime([aware(v) for v in data.ingested_at], utc=True)
                keys.append("ingested_at")
            if data.duplicated(keys).any():
                raise ValueError("Ambiguous duplicate membership revision")
            if self.mode == "INTERVALS":
                # Distinct intervals may not overlap. Revisions of the same start
                # are checked when selecting the observations known at a date.
                try:
                    self._check_overlap(data.drop_duplicates(["ticker", "start"], keep="last"))
                except ValueError as exc:
                    self.error = str(exc)
            self.frame = data
        except (ValueError, TypeError, OverflowError) as exc:
            self.error, self.mode = str(exc), "INVALID"

    @staticmethod
    def _check_overlap(data: pd.DataFrame) -> None:
        for _, group in data.sort_values(["ticker", "start"]).groupby("ticker"):
            rows = list(group.itertuples())
            if any(pd.isna(left.end) or left.end > right.start for left, right in zip(rows, rows[1:])):
                raise ValueError("Overlapping membership intervals")

    def members(self, date: pd.Timestamp, asof: pd.Timestamp) -> tuple[set[str], str, set[str]]:
        if self.mode in {"MISSING", "INVALID"}:
            return set(), f"UNAVAILABLE_{self.mode}_MEMBERSHIP", set()
        data = self.frame
        if self.timed:
            data = data.loc[data.ingested_at <= asof].sort_values("ingested_at")
        key = "start" if self.mode == "INTERVALS" else "effective_date"
        data = data.drop_duplicates(["ticker", key], keep="last")
        overlaps: set[str] = set()
        if self.mode == "INTERVALS":
            active = data.loc[(data.start <= date) & (data.end.isna() | (data.end > date))]
            members = set(active.ticker)
            overlaps = set(active.loc[active.ticker.duplicated(keep=False), "ticker"])
        else:
            known = data.loc[data.effective_date <= date].sort_values("effective_date").groupby("ticker").tail(1)
            members = set(known.loc[known.action == "ADD", "ticker"])
        if self.supported_through is not None and date > self.supported_through:
            return members, "UNAVAILABLE_STALE_MEMBERSHIP", overlaps
        if overlaps:
            return members, "UNAVAILABLE_OVERLAPPING_MEMBERSHIP", overlaps
        if not self.timed:
            return members, "UNAVAILABLE_LEGACY_OBSERVATION_TIMING", overlaps
        return members, "UNVERIFIED_BASELINE_AND_CONTINUITY" if members else "UNAVAILABLE_NO_KNOWN_MEMBERS", overlaps


@dataclass
class FeedAudit:
    kind: str
    frame: pd.DataFrame | None
    table: AlphaFeed | None = field(init=False, default=None)
    error: str = field(init=False, default="")
    timing: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.frame is None:
            return
        self.timing = timing_rows(self.kind, self.frame)
        try:
            self.table = AlphaFeed(self.kind, self.frame)
        except (ValueError, TypeError, OverflowError) as exc:
            self.error = str(exc)
        for row in self.timing:
            row["feed_validation"] = "INVALID_FILE" if self.error else "VALID_NOT_PROVIDER_VERIFIED"

    def states(self, date: pd.Timestamp, asof: pd.Timestamp, sessions: pd.DatetimeIndex,
               targets: set[str], index_windows: dict[str, int]) -> dict[str, str]:
        if self.table is None:
            return {"*": "UNAVAILABLE_INVALID_FEED" if self.error else "UNAVAILABLE_MISSING_FEED"}
        if self.kind == "indices" and not index_windows:
            return {"*": "UNAVAILABLE_INDEX_UNIVERSE_MANIFEST"}
        if self.kind != "indices" and not targets:
            return {"*": "UNAVAILABLE_MEMBERSHIP_DENOMINATOR"}
        rows = self.table.latest(asof, date)
        result: dict[str, str] = {}
        window = sessions[sessions <= date]
        if self.kind in {"indices", "flows", "delivery"}:
            entity = "index" if self.kind == "indices" else "ticker"
            for target in sorted(index_windows if self.kind == "indices" else targets):
                count = index_windows[target] if self.kind == "indices" else 20 if self.kind == "flows" else 25
                seen = set(rows.loc[rows[entity] == target, "session"])
                result[target] = "OBSERVED_WINDOW" if len(window) >= count and set(window[-count:]).issubset(seen) else "UNAVAILABLE_SESSION_WINDOW"
            return result
        if self.kind in {"sectors", "surveillance"}:
            latest = rows.sort_values(["effective_date", "known_at"]).drop_duplicates("ticker", keep="last").set_index("ticker")
        else:
            latest = rows.sort_values("period_end").drop_duplicates("ticker", keep="last").set_index("ticker")
        for target in sorted(targets):
            state = "UNAVAILABLE_NO_KNOWN_OBSERVATION"
            if target in latest.index:
                row = latest.loc[target]
                if self.kind in {"sectors", "surveillance"}:
                    state = "UNAVAILABLE_EXPIRED" if row.valid_until < date else "OBSERVED_INTERVAL"
                    if self.kind == "surveillance" and state.startswith("OBSERVED"):
                        state = "UNAVAILABLE_UNKNOWN_STATE" if row.state == "UNKNOWN" else f"OBSERVED_{row.state}"
                elif self.kind == "fundamentals":
                    state = "OBSERVED_PERIOD" if (date - row.period_end).days <= 150 else "UNAVAILABLE_STALE_PERIOD"
                else:
                    closes = window.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)
                    age = int((closes >= row.announced_at).sum())
                    state = "OBSERVED_CONSENSUS"
                    if age > 20:
                        state = "UNAVAILABLE_STALE_EARNINGS"
                    elif row.expectation_kind != "CONSENSUS" or abs(row.expected_eps) < 1e-9:
                        state = "UNAVAILABLE_CONSENSUS_OR_NONZERO_EXPECTATION"
            result[target] = state
        return result


@dataclass(frozen=True)
class CoverageResult:
    daily: pd.DataFrame
    feed_daily: pd.DataFrame
    ingestion: pd.DataFrame
    unavailable: pd.DataFrame
    summary: dict[str, Any]


def audit_coverage(history: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex,
                   membership: MembershipAudit, feeds: dict[str, FeedAudit], *, benchmark: str,
                   index_windows: dict[str, int] | None = None, price_lookback: int = 200,
                   observation_hour: int = 18, context_sessions: pd.DatetimeIndex | None = None) -> CoverageResult:
    if not len(sessions) or not sessions.is_unique or not sessions.is_monotonic_increasing or sessions.tz is not None:
        raise ValueError("Audit sessions must be nonempty, ordered, unique and timezone-free")
    if price_lookback < 1 or not 0 <= observation_hour <= 23 or any(v < 1 for v in (index_windows or {}).values()):
        raise ValueError("Invalid audit lookback or observation hour")
    context = context_sessions if context_sessions is not None else sessions
    if any(not isinstance(frame.index, pd.DatetimeIndex) for frame in history.values()):
        raise ValueError("Price sessions must use a DatetimeIndex")
    if context_sessions is None:
        for frame in history.values():
            context = context.union(frame.index)
    if (not context.is_unique or not context.is_monotonic_increasing or context.tz is not None
            or not sessions.isin(context).all()):
        raise ValueError("Invalid warm-up session calendar")
    valid: dict[str, pd.Series] = {}
    present: dict[str, set[pd.Timestamp]] = {}
    for ticker, frame in history.items():
        if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is not None or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError(f"Invalid price session index for {ticker}")
        if not frame.index.equals(frame.index.normalize()):
            raise ValueError(f"Non-daily price timestamps for {ticker}")
        columns = ["Open", "High", "Low", "Close", "Volume"]
        if not set(columns).issubset(frame):
            raise ValueError(f"Missing OHLCV columns for {ticker}")
        data = frame[columns].reindex(context).apply(pd.to_numeric, errors="coerce")
        good = pd.Series(np.isfinite(data.to_numpy(dtype=float)).all(axis=1), index=context)
        good &= (data[["Open", "High", "Low", "Close"]] > 0).all(axis=1) & (data.Volume >= 0)
        good &= (data.High >= data[["Open", "Close", "Low"]].max(axis=1)) & (data.Low <= data[["Open", "Close"]].min(axis=1))
        key = symbol(ticker)
        if key in valid:
            raise ValueError(f"Ambiguous normalized price ticker {key}")
        valid[key] = good
        present[key] = set(frame.index)
    lookbacks = {t: s.rolling(price_lookback, min_periods=price_lookback).sum().eq(price_lookback) for t, s in valid.items()}
    gaps = Intervals()
    daily, feed_daily = [], []
    for date in sessions:
        asof = date.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=observation_hour)
        targets, membership_status, overlaps = membership.members(date, asof)
        unavailable = {("membership", "*", membership_status), ("prices_asof", "*", "UNAVAILABLE_PRICE_OBSERVATION_TIMESTAMPS")}
        unavailable.update(("membership_integrity", ticker, "UNAVAILABLE_OVERLAPPING_INTERVALS") for ticker in overlaps)
        current, warmed, no_series, no_bar, invalid_bar = 0, 0, 0, 0, 0
        for ticker in sorted(targets):
            reason = ""
            if ticker not in valid:
                reason = "UNAVAILABLE_NO_PRICE_SERIES"
                no_series += 1
            elif date not in present[ticker]:
                reason = "UNAVAILABLE_NO_PRICE_BAR"
                no_bar += 1
            elif not bool(valid[ticker].loc[date]):
                reason = "UNAVAILABLE_INVALID_OHLCV"
                invalid_bar += 1
            else:
                current += 1
                if bool(lookbacks[ticker].loc[date]):
                    warmed += 1
                else:
                    reason = "UNAVAILABLE_PRICE_LOOKBACK"
            if reason:
                unavailable.add(("prices", ticker, reason))
        benchmark_valid = symbol(benchmark) in valid and bool(valid[symbol(benchmark)].loc[date])
        if not benchmark_valid:
            unavailable.add(("benchmark", benchmark, "UNAVAILABLE_MISSING_OR_INVALID_OHLCV"))
        daily.append({"session": date.date().isoformat(), "decision_at": asof.isoformat(), "membership_status": membership_status,
                      "membership_integrity": "OVERLAPPING_INTERVALS" if overlaps else "NO_ACTIVE_OVERLAP_DETECTED",
                      "overlapping_members": sorted(overlaps),
                      "denominator_basis": "KNOWN_ROW_SET_COMPLETENESS_UNVERIFIED" if membership.timed else "LEGACY_DECLARED_INTERVAL_SET_UNVERIFIED",
                      "declared_members": len(targets), "members_with_valid_bar": current, "members_missing_valid_bar": len(targets) - current,
                      "members_without_price_series": no_series, "members_without_session_bar": no_bar,
                      "members_with_invalid_ohlcv": invalid_bar,
                      "members_with_price_lookback": warmed, "declared_member_price_coverage": current / len(targets) if targets else None,
                      "known_at_member_count": len(targets) if membership.timed and membership.mode != "INVALID" and not overlaps else None,
                      "benchmark_bar_valid": benchmark_valid, "price_asof_status": "UNAVAILABLE_PRICE_OBSERVATION_TIMESTAMPS"})
        for kind in SCHEMAS:
            feed = feeds.get(kind, FeedAudit(kind, None))
            states = feed.states(date, asof, context, targets, index_windows or {})
            covered = sum(state.startswith("OBSERVED") for state in states.values())
            denominator = len(index_windows or {}) if kind == "indices" else len(targets)
            unavailable.update((kind, entity, reason) for entity, reason in states.items() if not reason.startswith("OBSERVED"))
            feed_daily.append({"session": date.date().isoformat(), "decision_at": asof.isoformat(), "feed": kind,
                               "expected_entities": denominator, "covered_entities": covered,
                               "status": "RECORDED_COVERAGE_NOT_PROVIDER_VERIFIED" if denominator and covered == denominator else "UNAVAILABLE_OR_PARTIAL",
                               "reason_counts": {str(k): int(v) for k, v in pd.Series(list(states.values())).value_counts().items()}})
        gaps.observe(date, unavailable)
    ingestion = membership.timing or [{"component": "membership", "status": "UNAVAILABLE_NO_ROWS"}]
    ingestion = ingestion + [{"component": "prices", "status": "UNAVAILABLE_PRICE_OBSERVATION_TIMESTAMPS"}]
    for kind in SCHEMAS:
        feed = feeds.get(kind, FeedAudit(kind, None))
        ingestion += feed.timing or [{"component": kind, "status": "INVALID_FILE" if feed.error else "UNAVAILABLE_NO_ROWS"}]
    daily_frame = pd.DataFrame(daily)
    summary = {
        "status": "UNAVAILABLE_POINT_IN_TIME_COMPLETENESS", "sessions": len(sessions),
        "start": sessions[0].date().isoformat(), "end": sessions[-1].date().isoformat(),
        "membership_mode": membership.mode, "membership_error": membership.error,
        "membership_timestamps_supplied": membership.timed,
        "membership_supported_through_declared": membership.supported_through.date().isoformat() if membership.supported_through is not None else None,
        "supported_through_basis": "DECLARED_UPPER_BOUND_NOT_OBSERVATION_OR_COMPLETENESS_PROOF",
        "membership_status_sessions": {str(k): int(v) for k, v in daily_frame.membership_status.value_counts().items()},
        "membership_integrity_sessions": {str(k): int(v) for k, v in daily_frame.membership_integrity.value_counts().items()},
        "snapshot_price_series": len(history), "price_lookback_sessions": price_lookback,
        "declared_members_min": int(daily_frame.declared_members.min()), "declared_members_max": int(daily_frame.declared_members.max()),
        "minimum_declared_member_price_coverage": float(daily_frame.declared_member_price_coverage.min()) if daily_frame.declared_member_price_coverage.notna().any() else None,
        "maximum_declared_member_price_coverage": float(daily_frame.declared_member_price_coverage.max()) if daily_frame.declared_member_price_coverage.notna().any() else None,
        "missing_benchmark_sessions": daily_frame.loc[~daily_frame.benchmark_bar_valid, "session"].tolist(),
        "feeds": {kind: {"status": "INVALID" if feed.error else "MISSING" if feed.table is None else "VALID_RECORDED_TIMESTAMPS_NOT_PROVIDER_VERIFIED",
                         "error": feed.error, "rows": 0 if feed.frame is None else len(feed.frame)} for kind, feed in feeds.items()},
        "limitations": ["Membership row sets do not establish a complete baseline, continuity, rename map or delisted-price coverage.",
                        "Legacy interval dates are retrospective declarations, not proof they were known then.",
                        "Valid local publication/ingestion claims are not independently verified provider evidence.",
                        "The price snapshot lacks per-observation ingestion timestamps and revision provenance.",
                        "OHLCV coverage is retrospective; prices are never forward-filled for coverage.",
                        "Feed coverage does not imply eligibility, matched earnings periods, profitability or trading permission.",
                        "Unavailable intervals cover adjacent audited sessions; exchange-calendar completeness requires separate evidence."],
    }
    ingestion_columns = ["component", "csv_row", "entity", "published_at_raw", "ingested_at_raw", "published_at", "ingested_at", "source",
                         "ingestion_delay_seconds", "status", "error", "feed_validation"]
    return CoverageResult(daily_frame, pd.DataFrame(feed_daily), pd.DataFrame(ingestion, columns=ingestion_columns), gaps.finish(), summary)
