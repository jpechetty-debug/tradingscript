"""Validated local V2 feeds. Missing observations never imply regulatory clearance."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


SCHEMAS = {
    "indices": ("index", "session", "close"),
    "sectors": ("ticker", "effective_date", "valid_until", "sector_index"),
    "surveillance": ("ticker", "effective_date", "valid_until", "state"),
    "earnings": ("ticker", "period_end", "announced_at", "expected_at", "expectation_ingested_at",
                 "actual_eps", "expected_eps", "expectation_kind"),
    "delivery": ("ticker", "session", "delivery_qty", "total_qty", "turnover"),
}


def aware(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Feed observation timestamps must include a timezone")
    return stamp.tz_convert("UTC")


class AlphaFeed:
    def __init__(self, kind: str, frame: pd.DataFrame):
        if kind not in SCHEMAS:
            raise ValueError(f"Unknown feed: {kind}")
        self.kind = kind
        required = [*SCHEMAS[kind], "published_at", "ingested_at", "source"]
        if any(c not in frame for c in required) or frame[required].isna().any().any():
            raise ValueError(f"{kind}: missing required values/columns: {required}")
        data = frame.copy()
        for col in ("published_at", "ingested_at", "announced_at", "expected_at", "expectation_ingested_at"):
            if col in data:
                data[col] = pd.to_datetime([aware(v) for v in data[col]], utc=True)
        if (data.ingested_at < data.published_at).any():
            raise ValueError("Ingestion cannot precede publication")
        data["known_at"] = data.ingested_at
        for col in ("session", "effective_date", "valid_until", "period_end"):
            if col in data:
                data[col] = pd.to_datetime(data[col], errors="raise")
                if data[col].dt.tz is not None or (data[col] != data[col].dt.normalize()).any():
                    raise ValueError("Session/effective dates must be timezone-free calendar dates")
        for col in ("ticker", "index", "sector_index", "source"):
            if col in data:
                data[col] = data[col].astype(str).str.strip()
                if (data[col] == "").any():
                    raise ValueError(f"Empty {col}")
        if "ticker" in data:
            data["ticker"] = data.ticker.str.upper().str.removesuffix(".NS")
        numeric = [c for c in ("close", "actual_eps", "expected_eps", "delivery_qty", "total_qty", "turnover") if c in data]
        for col in numeric:
            data[col] = pd.to_numeric(data[col], errors="raise")
        if numeric and not np.isfinite(data[numeric].to_numpy(dtype=float)).all():
            raise ValueError("Feed numbers must be finite")
        if "close" in data and (data.close <= 0).any():
            raise ValueError("Index closes must be positive")
        if "valid_until" in data and (data.valid_until < data.effective_date).any():
            raise ValueError("Invalid validity interval")
        if kind == "surveillance" and not data.state.isin(["CLEAR", "ASM", "GSM", "BOTH", "UNKNOWN"]).all():
            raise ValueError("Invalid surveillance state")
        if kind == "delivery" and ((data.total_qty <= 0) | (data.delivery_qty < 0)
                                   | (data.delivery_qty > data.total_qty) | (data.turnover < 0)).any():
            raise ValueError("Invalid delivery quantities/turnover")
        if kind == "earnings":
            if not data.expectation_kind.isin(["CONSENSUS", "SEASONAL"]).all():
                raise ValueError("Expectation kind must be CONSENSUS or SEASONAL")
            if ((data.expected_at >= data.announced_at) | (data.expectation_ingested_at < data.expected_at)
                    | (data.expectation_ingested_at >= data.announced_at)
                    | (data.published_at < data.announced_at) | (data.period_end > data.announced_at.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None).dt.normalize())).any():
                raise ValueError("Earnings expectations must have been observed before the announcement")
        keys = {"indices": ["index", "session"], "delivery": ["ticker", "session"],
                "earnings": ["ticker", "period_end"], "sectors": ["ticker", "effective_date"],
                "surveillance": ["ticker", "effective_date"]}
        self.keys = keys[kind]
        if data.duplicated([*self.keys, "known_at"]).any():
            raise ValueError("Conflicting same-time feed revisions")
        self.frame = data

    def latest(self, asof: pd.Timestamp, session: pd.Timestamp) -> pd.DataFrame:
        known = self.frame.loc[self.frame.known_at <= aware(asof)]
        date_col = "session" if "session" in known else "effective_date" if "effective_date" in known else "period_end"
        known = known.loc[known[date_col] <= session]
        # Select revisions before checking validity, so an expired revision cannot resurrect an old row.
        return known.sort_values("known_at").drop_duplicates(self.keys, keep="last")

    def interval(self, ticker: str, asof: pd.Timestamp, session: pd.Timestamp) -> dict[str, Any] | None:
        known = self.latest(asof, session)
        known = known.loc[known.ticker == ticker.removesuffix(".NS")].sort_values(["effective_date", "known_at"])
        if known.empty or known.iloc[-1].valid_until < session:
            return None
        return dict(known.iloc[-1])


@dataclass
class AlphaFeeds:
    sector_universe: tuple[str, ...] = ()
    tables: dict[str, AlphaFeed] = field(default_factory=dict)
    benchmark_index: str = "NIFTY500"
    vix_index: str = "INDIA_VIX"

    def index_history(self, name: str, asof: pd.Timestamp, session: pd.Timestamp) -> pd.Series:
        if "indices" not in self.tables:
            return pd.Series(dtype=float)
        data = self.tables["indices"].latest(asof, session)
        return data.loc[data["index"] == name].sort_values("session").set_index("session").close

    def interval(self, kind: str, ticker: str, asof: pd.Timestamp, session: pd.Timestamp) -> dict[str, Any] | None:
        return self.tables[kind].interval(ticker, asof, session) if kind in self.tables else None

    def earnings(self, ticker: str, asof: pd.Timestamp, session: pd.Timestamp,
                 sessions: pd.DatetimeIndex, max_age: int = 20) -> dict[str, Any]:
        unknown = {"earnings_status": "UNAVAILABLE", "earnings_surprise": None}
        if "earnings" not in self.tables:
            return unknown
        rows = self.tables["earnings"].latest(asof, session)
        rows = rows.loc[rows.ticker == ticker.removesuffix(".NS")].sort_values("period_end")
        if rows.empty:
            return unknown
        row = rows.iloc[-1]
        # Sessions whose close preceded publication cannot be counted as post-announcement sessions.
        closes = sessions.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)
        age = int(((closes >= row.announced_at) & (sessions <= session)).sum())
        if age > max_age:
            return {**unknown, "earnings_status": "STALE"}
        if abs(row.expected_eps) < 1e-9:
            return {**unknown, "earnings_status": "ZERO_EXPECTATION_UNSCORABLE"}
        return {"earnings_status": row.expectation_kind,
                "earnings_surprise": float((row.actual_eps - row.expected_eps) / abs(row.expected_eps)),
                "earnings_age_sessions": age, "earnings_known_at": row.known_at.isoformat(),
                "earnings_source": row.source, "earnings_period_end": row.period_end.isoformat()}

    def delivery(self, ticker: str, asof: pd.Timestamp, session: pd.Timestamp,
                 sessions: pd.DatetimeIndex) -> dict[str, Any]:
        unknown = {"delivery_status": "UNAVAILABLE", "accumulation_proxy": None}
        if "delivery" not in self.tables:
            return unknown
        data = self.tables["delivery"].latest(asof, session)
        data = data.loc[data.ticker == ticker.removesuffix(".NS")].set_index("session").reindex(sessions[-25:])
        if len(data) != 25 or data.delivery_qty.isna().any():
            return unknown
        recent, prior = data.iloc[-5:], data.iloc[:-5]
        pct_recent = float(recent.delivery_qty.sum() / recent.total_qty.sum())
        pct_prior = float(prior.delivery_qty.sum() / prior.total_qty.sum())
        qty_base, turnover_base = float(prior.delivery_qty.mean()), float(prior.turnover.mean())
        if qty_base <= 0 or turnover_base <= 0:
            return unknown
        qty_ratio = float(recent.delivery_qty.mean() / qty_base)
        turnover_ratio = float(recent.turnover.mean() / turnover_base)
        return {"delivery_status": "OBSERVED_PROXY", "delivery_fraction_5": pct_recent,
                "delivery_fraction_prior20": pct_prior, "delivery_quantity_ratio": qty_ratio,
                "delivery_turnover_ratio": turnover_ratio,
                "accumulation_proxy": pct_recent > pct_prior and qty_ratio > 1 and turnover_ratio > 1}
