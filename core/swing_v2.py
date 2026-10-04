"""Separate sector/PEAD hypotheses, independent of the V1 factor scorer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .alpha_feeds import AlphaFeeds
from .config import IST, SystemConfig
from .swing import SwingPlan, detect_swing_trigger
from .swing_replay import ReplaySignal


@dataclass(frozen=True)
class V2Settings:
    model: str = "SECTOR"
    market_gate: bool = False
    breadth_gate: bool = False
    vix_gate: bool = False
    delivery_gate: bool = False
    sector_lookback: int = 63
    sector_fraction: float = .25
    earnings_max_age: int = 20
    breadth_min: float = .5
    breadth_coverage: float = .9
    observation_hour: int = 18
    pead_hold_sessions: int = 40

    def __post_init__(self) -> None:
        if self.model not in {"SECTOR", "PEAD", "COMBINED"}:
            raise ValueError("Unknown V2 model")
        if (self.sector_lookback < 2 or self.earnings_max_age < 1 or self.pead_hold_sessions < 1
                or not 0 < self.sector_fraction <= 1 or not 0 <= self.breadth_min <= 1
                or not 0 < self.breadth_coverage <= 1 or not 16 <= self.observation_hour <= 23):
            raise ValueError("Invalid V2 parameters")

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


@dataclass
class V2Candidate:
    ticker: str
    sector: str
    strategy_id: str
    signal_time: str
    entry: float
    entry_min: float
    entry_max: float
    stop: float
    t1: float
    t2: float
    time_stop_bars: int
    composite: float
    research_context: dict[str, Any]
    # Replay's legacy numeric column is unused for V2 selection/sizing. Status discloses the sentinel.
    prob_win: float = 0.
    expectancy_r: float = 0.
    probability_status: str = "UNAVAILABLE_UNUSED_SENTINEL"


class SwingV2:
    def __init__(self, feeds: AlphaFeeds, settings: V2Settings = V2Settings()):
        self.feeds = feeds
        self.settings = settings
        self.decisions: list[dict[str, Any]] = []

    def market_context(self, frames: dict[str, pd.DataFrame], date: pd.Timestamp,
                       asof: pd.Timestamp, config: SystemConfig) -> dict[str, Any]:
        sessions = frames[config.BENCHMARK].index
        benchmark = self.feeds.index_history(self.feeds.benchmark_index, asof, date).reindex(sessions)
        aligned = len(benchmark) >= 200 and benchmark.iloc[-200:].notna().all()
        above = bool(benchmark.iloc[-1] > benchmark.iloc[-200:].mean()) if aligned else None
        stocks = [f for t, f in frames.items() if t != config.BENCHMARK and t.endswith(".NS")]
        valid = [f for f in stocks if len(f) >= 200 and date in f.index
                 and np.isfinite(f.Close.iloc[-200:]).all() and (f.Close.iloc[-200:] > 0).all()]
        coverage = len(valid) / len(stocks) if stocks else 0.
        breadth = float(np.mean([f.Close.iloc[-1] > f.Close.iloc[-200:].mean() for f in valid])) if valid else None
        vix = self.feeds.index_history(self.feeds.vix_index, asof, date).reindex(sessions)
        low_vix = bool(vix.iloc[-1] < vix.iloc[-127:-1].median()) if len(vix) >= 127 and vix.iloc[-127:].notna().all() else None
        ranks: dict[str, float] = {}
        values: dict[str, float] = {}
        lookback = self.settings.sector_lookback
        if (len(sessions) >= lookback + 1 and benchmark.iloc[-lookback-1:].notna().all()
                and len(set(self.feeds.sector_universe)) == len(self.feeds.sector_universe)
                and len(self.feeds.sector_universe) >= 4):
            for sector in self.feeds.sector_universe:
                prices = self.feeds.index_history(sector, asof, date).reindex(sessions[-lookback-1:])
                if prices.notna().all():
                    values[sector] = float((prices.iloc[-1] / prices.iloc[0]) /
                                           (benchmark.iloc[-1] / benchmark.iloc[-lookback-1]) - 1)
            if len(values) == len(self.feeds.sector_universe):
                # Conservative ties: tied leaders receive the worst occupied rank.
                order = pd.Series(values).rank(ascending=False, method="max")
                ranks = {str(k): float(v / len(values)) for k, v in order.items()}
        return {"benchmark_above_sma200": above, "breadth_sma200": breadth,
                "breadth_denominator": len(valid), "breadth_available_universe": len(stocks),
                "breadth_coverage": coverage, "breadth_universe_status": "AVAILABLE_SNAPSHOT_NOT_POINT_IN_TIME",
                "vix_below_prior126_median": low_vix, "sector_rank_fractions": ranks,
                "sector_rs": values, "sector_universe_size": len(self.feeds.sector_universe)}

    def __call__(self, frames: dict[str, pd.DataFrame], date: pd.Timestamp,
                 config: SystemConfig) -> list[V2Candidate]:
        date = pd.Timestamp(date).tz_localize(None).normalize()
        frames = {t: f.loc[f.index <= date] for t, f in frames.items() if date in f.index}
        if config.BENCHMARK not in frames:
            raise ValueError("V2 requires a benchmark bar for the signal session")
        asof = date.tz_localize(IST) + pd.Timedelta(hours=self.settings.observation_hour)
        market = self.market_context(frames, date, asof, config)
        settings = self.settings
        candidates: list[V2Candidate] = []
        for ticker, frame in sorted(frames.items()):
            if ticker == config.BENCHMARK or not ticker.endswith(".NS"):
                continue
            symbol = ticker.removesuffix(".NS")
            mapping = self.feeds.interval("sectors", symbol, asof, date)
            surveillance = self.feeds.interval("surveillance", symbol, asof, date)
            earnings = self.feeds.earnings(symbol, asof, date, frames[config.BENCHMARK].index, settings.earnings_max_age)
            delivery = self.feeds.delivery(symbol, asof, date, frames[config.BENCHMARK].index)
            sector = str(mapping["sector_index"]) if mapping else "UNKNOWN"
            fraction = market["sector_rank_fractions"].get(sector)
            context = {**market, **earnings, **delivery, "model": settings.model,
                       "settings_sha256": settings.fingerprint(), "observed_at": asof.isoformat(),
                       "sector_index": sector, "sector_rank_fraction": fraction,
                       "surveillance_state": surveillance["state"] if surveillance else "UNAVAILABLE",
                       "surveillance_source": surveillance["source"] if surveillance else None,
                       "probability_status": "UNAVAILABLE_UNUSED_SENTINEL"}
            reasons: list[str] = []
            if mapping is None:
                reasons.append("sector_mapping_unavailable")
            if surveillance is None or surveillance["state"] != "CLEAR":
                reasons.append("surveillance_unavailable" if surveillance is None else "surveillance_" + surveillance["state"].lower())
            if settings.model in {"SECTOR", "COMBINED"}:
                if fraction is None:
                    reasons.append("sector_history_unavailable")
                elif fraction > settings.sector_fraction:
                    reasons.append("sector_outside_top_quartile")
            if settings.model in {"PEAD", "COMBINED"}:
                if earnings["earnings_status"] != "CONSENSUS":
                    reasons.append("consensus_surprise_unavailable")
                elif earnings["earnings_surprise"] <= 0:
                    reasons.append("nonpositive_earnings_surprise")
            if settings.market_gate and market["benchmark_above_sma200"] is not True:
                reasons.append("market_gate_unknown_or_weak")
            if settings.breadth_gate and (market["breadth_sma200"] is None or market["breadth_coverage"] < settings.breadth_coverage
                                          or market["breadth_sma200"] <= settings.breadth_min):
                reasons.append("breadth_gate_unknown_or_weak")
            if settings.vix_gate and market["vix_below_prior126_median"] is not True:
                reasons.append("vix_gate_unknown_or_high")
            if settings.delivery_gate and delivery["accumulation_proxy"] is not True:
                reasons.append("delivery_gate_unknown_or_weak")
            volume = frame.Volume.iloc[-20:] if "Volume" in frame else pd.Series(dtype=float)
            if (len(volume) != 20 or not np.isfinite(volume).all() or volume.mean() < config.ADV_SHARE_FLOOR
                    or (volume * frame.Close.iloc[-20:]).mean() < config.ADV_TURNOVER_FLOOR):
                reasons.append("liquidity_insufficient")
            # V2's alpha comes from external evidence; RVOL is recorded rather than an extra entry gate.
            timing_config = replace(config, SWING_BREAKOUT_RVOL=0.)
            plan = detect_swing_trigger(frame, timing_config)
            if plan is None:
                reasons.append("no_timing_trigger")
            self.decisions.append({"date": date.isoformat(), "ticker": ticker,
                                   "status": "REJECTED" if reasons else "QUALIFIED",
                                   "reasons": reasons, "context": context})
            if reasons or plan is None:
                continue
            strength = (1 - fraction) if settings.model == "SECTOR" else float(earnings["earnings_surprise"])
            candidates.append(self.candidate(symbol, sector, plan, float(strength), context))
        return list(sorted(candidates, key=lambda c: (-c.composite, c.ticker)))

    def candidate(self, ticker: str, sector: str, plan: SwingPlan, strength: float,
                  context: dict[str, Any]) -> V2Candidate:
        trigger = "BREAKOUT" if "BREAKOUT" in plan.strategy_id else "PULLBACK"
        return V2Candidate(ticker=ticker, sector=sector,
                           strategy_id=f"SWING_{self.settings.model}_V2_{trigger}", signal_time=plan.signal_time,
                           entry=plan.entry, entry_min=plan.entry_min, entry_max=plan.entry_max,
                           stop=plan.stop, t1=plan.target, t2=plan.target2,
                           time_stop_bars=self.settings.pead_hold_sessions if self.settings.model != "SECTOR" else plan.time_stop_bars,
                           composite=strength, research_context=context)

    def entry_guard(self, candidate: ReplaySignal, date: pd.Timestamp) -> str | None:
        session = pd.Timestamp(date).tz_localize(None).normalize()
        opening = session.tz_localize(IST) + pd.Timedelta(hours=9, minutes=15)
        row = self.feeds.interval("surveillance", candidate.ticker, opening, session)
        if row is None or row["state"] != "CLEAR":
            return "entry_surveillance_unavailable" if row is None else "entry_surveillance_" + row["state"].lower()
        sector = self.feeds.interval("sectors", candidate.ticker, opening, session)
        if sector is None or sector["sector_index"] != candidate.sector:
            return "entry_sector_mapping_changed_or_unavailable"
        return None
