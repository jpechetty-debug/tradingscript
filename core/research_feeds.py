"""As-of adapters for supplied event and membership data, never current-list substitutes."""
from __future__ import annotations

import pandas as pd


def _validate(frame: pd.DataFrame, required: set[str]) -> pd.DataFrame:
    if not required.issubset(frame.columns) or frame[list(required)].isna().any().any():
        raise ValueError(f"Feed requires nonmissing columns {sorted(required)}")
    result = frame.copy()
    if any(pd.Timestamp(value).tzinfo is None for value in result.known_at):
        raise ValueError("Feed known_at timestamps must be timezone-aware")
    result["known_at"] = pd.to_datetime(result.known_at, utc=True, errors="raise")
    result["ticker"] = result.ticker.astype(str).str.strip().str.upper()
    if result.ticker.str.strip().eq("").any():
        raise ValueError("Blank ticker in research feed")
    return result


class EarningsCalendar:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = _validate(frame, {"ticker", "event_date", "known_at"})
        self.frame["event_date"] = pd.to_datetime(self.frame.event_date, errors="raise").dt.normalize()

    def status(self, ticker: str, asof: pd.Timestamp, entry_session: pd.Timestamp,
               sessions: pd.DatetimeIndex, buffer: int = 5) -> str:
        """Known upcoming event within five exchange sessions; no row is NOT an all-clear.

        `known_at` is a timezone-aware observation time. The session index must
        include the event date; incomplete future calendars cannot certify safety.
        """
        if asof.tzinfo is None:
            raise ValueError("Event observation time must be timezone-aware")
        events = self.frame.loc[(self.frame.ticker == ticker.upper()) & (self.frame.known_at <= asof)]
        entry_position = sessions.get_indexer([entry_session])[0]
        if entry_position < 0:
            raise ValueError("Entry date missing from exchange sessions")
        for date in events.event_date:
            event_position = sessions.get_indexer([date])[0]
            if event_position >= 0 and 0 <= event_position - entry_position <= buffer:
                return "KNOWN_EARNINGS_WITHIN_5_SESSIONS"
        return "NO_KNOWN_EVENT_COVERAGE_UNVERIFIED"


class MembershipHistory:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = _validate(frame, {"ticker", "effective_date", "action", "known_at"})
        self.frame["effective_date"] = pd.to_datetime(self.frame.effective_date, errors="raise").dt.normalize()
        self.frame["action"] = self.frame.action.astype(str).str.upper()
        if not self.frame.action.isin(["ADD", "REMOVE"]).all():
            raise ValueError("Membership action must be ADD or REMOVE")
        identity = ["ticker", "effective_date", "known_at"]
        if self.frame.duplicated(identity).any():
            raise ValueError("Ambiguous duplicate membership event")

    def members(self, asof: pd.Timestamp, session: pd.Timestamp) -> set[str]:
        if asof.tzinfo is None:
            raise ValueError("Membership observation time must be timezone-aware")
        known = self.frame.loc[(self.frame.known_at <= asof) & (self.frame.effective_date <= session)]
        latest = known.sort_values(["effective_date", "known_at"]).groupby("ticker").tail(1)
        return set(latest.loc[latest.action == "ADD", "ticker"])

    def missing_prices(self, members: set[str], available: set[str]) -> set[str]:
        return members - available
