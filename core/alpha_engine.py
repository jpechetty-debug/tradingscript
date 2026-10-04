"""Four-phase swing alpha hypotheses over observable, point-in-time evidence."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from .alpha_feeds import AlphaFeeds
from .config import IST, SystemConfig
from .swing_v2 import SwingV2, V2Candidate, V2Settings


WEIGHTS = {"relative_strength": .30, "earnings": .25, "flow": .20, "sector": .15, "structure": .10}


@dataclass(frozen=True)
class AlphaSettings:
    phase: int = 4
    rs_lookback: int = 63
    rs_min_percentile: float = 80.
    top_sectors: int = 3
    min_alpha_score: float = 85.
    breadth_coverage: float = .9
    full_risk_breadth: float = .6
    half_risk_breadth: float = .4
    eps_growth_min: float = .20
    revenue_growth_min: float = .15
    fundamentals_max_age_days: int = 150
    flow_sessions: int = 20
    meta_min_probability: float = .55

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Alpha parameters must be finite")
        if (self.phase not in (1, 2, 3, 4) or self.rs_lookback < 2 or self.top_sectors < 1
                or not 0 <= self.rs_min_percentile <= 100 or not 0 <= self.min_alpha_score <= 100
                or not 0 < self.breadth_coverage <= 1
                or not 0 <= self.half_risk_breadth < self.full_risk_breadth <= 1
                or self.eps_growth_min < 0 or self.revenue_growth_min < 0
                or self.fundamentals_max_age_days < 1 or self.flow_sessions < 2
                or not 0 < self.meta_min_probability < 1):
            raise ValueError("Invalid alpha parameters")

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def breadth_context(frames: dict[str, pd.DataFrame], benchmark: str, settings: AlphaSettings) -> dict[str, Any]:
    stocks = [f for t, f in frames.items() if t != benchmark and t.endswith(".NS")]
    valid = [f for f in stocks if len(f) >= 50 and np.isfinite(f.Close.iloc[-50:]).all()
             and (f.Close.iloc[-50:] > 0).all()]
    coverage = len(valid) / len(stocks) if stocks else 0.
    breadth = float(np.mean([f.Close.iloc[-1] > f.Close.iloc[-50:].mean() for f in valid])) if valid else None
    scale = (1. if breadth is not None and breadth > settings.full_risk_breadth else
             .5 if breadth is not None and breadth >= settings.half_risk_breadth else 0.)
    if coverage < settings.breadth_coverage:
        scale = 0.
    return {"breadth_sma50": breadth, "breadth_valid": len(valid), "breadth_total": len(stocks),
            "breadth_sma50_coverage": coverage, "breadth_risk_scale": scale}


def relative_strength_ranks(frames: dict[str, pd.DataFrame], bench: pd.Series,
                            config: SystemConfig, lookback: int) -> dict[str, float]:
    scores = {}
    for ticker, frame in frames.items():
        if ticker == config.BENCHMARK or not ticker.endswith(".NS") or len(frame) < max(lookback + 1, 20):
            continue
        liquid = frame[["Close", "Volume"]].iloc[-20:]
        if (not np.isfinite(liquid.to_numpy()).all() or (liquid.Close <= 0).any() or (liquid.Volume < 0).any()
                or liquid.Volume.mean() < config.ADV_SHARE_FLOOR
                or (liquid.Volume * liquid.Close).mean() < config.ADV_TURNOVER_FLOOR):
            continue
        rank_sessions = bench.index[-lookback - 1:]
        joined = pd.DataFrame({"stock": frame.Close.reindex(rank_sessions), "benchmark": bench.reindex(rank_sessions)})
        if (len(joined) != lookback + 1 or not np.isfinite(joined.to_numpy()).all()
                or (joined <= 0).any().any() or joined.index[-1] != bench.index[-1]):
            continue
        # Return differences remain defined when the benchmark return is zero or negative.
        scores[ticker.removesuffix(".NS")] = float(np.log(joined.stock.iloc[-1] / joined.stock.iloc[0])
                                                   - np.log(joined.benchmark.iloc[-1] / joined.benchmark.iloc[0]))
    return {str(t): float(v) for t, v in pd.Series(scores, dtype=float).rank(method="average", pct=True).mul(100).items()}


class SwingAlpha:
    def __init__(self, feeds: AlphaFeeds, settings: AlphaSettings = AlphaSettings(), meta_model: Any = None):
        self.feeds = feeds
        self.settings = settings
        self.meta_model = meta_model
        self.base = SwingV2(feeds, V2Settings(model="SECTOR", sector_fraction=1., sector_lookback=settings.rs_lookback))
        self.decisions: list[dict[str, Any]] = []

    def earnings_context(self, ticker: str, date: pd.Timestamp, asof: pd.Timestamp,
                         sessions: pd.DatetimeIndex) -> dict[str, Any]:
        result = self.feeds.earnings(ticker, asof, date, sessions)
        result.update(fundamentals_status="UNAVAILABLE", earnings_score=None)
        table = self.feeds.tables.get("fundamentals")
        if table is None:
            return result
        rows = table.latest(asof, date)
        rows = rows.loc[rows.ticker == ticker].sort_values("period_end")
        if rows.empty:
            return result
        row = rows.iloc[-1]
        age = int((date - row.period_end).days)
        result.update(eps_growth=float(row.eps_growth), revenue_growth=float(row.revenue_growth),
                      guidance_raised=bool(row.guidance_raised), fundamentals_source=row.source,
                      fundamentals_known_at=row.known_at.isoformat(), fundamentals_age_days=age,
                      fundamentals_status="OBSERVED" if age <= self.settings.fundamentals_max_age_days else "STALE")
        if result.get("earnings_period_end") is not None and pd.Timestamp(result["earnings_period_end"]) != row.period_end:
            result["fundamentals_status"] = "PERIOD_MISMATCH"
        if result["fundamentals_status"] == "OBSERVED" and result["earnings_status"] == "CONSENSUS":
            result["earnings_score"] = float(np.mean([
                np.clip(row.eps_growth / max(self.settings.eps_growth_min * 2, .01), 0, 1),
                np.clip(row.revenue_growth / max(self.settings.revenue_growth_min * 2, .01), 0, 1),
                float(row.guidance_raised), np.clip(result["earnings_surprise"] / .2, 0, 1)]))
        return result

    def flow_context(self, ticker: str, date: pd.Timestamp, asof: pd.Timestamp,
                     sessions: pd.DatetimeIndex, frame: pd.DataFrame) -> dict[str, Any]:
        result = self.feeds.delivery(ticker, asof, date, sessions)
        result.update(flow_status="UNAVAILABLE", flow_score=None)
        table = self.feeds.tables.get("flows")
        if table is None:
            return result
        rows = table.latest(asof, date)
        rows = rows.loc[rows.ticker == ticker].set_index("session").reindex(sessions[-self.settings.flow_sessions:])
        if (len(rows) != self.settings.flow_sessions or rows.fii_net_inr.isna().any()
                or rows.dii_net_inr.isna().any() or result["delivery_status"] != "OBSERVED_PROXY"):
            return result
        fii, dii = float(rows.fii_net_inr.sum()), float(rows.dii_net_inr.sum())
        volume = frame.Volume.iloc[-25:]
        if len(volume) != 25 or not np.isfinite(volume).all() or volume.iloc[:-5].mean() <= 0:
            return result
        volume_ratio = float(volume.iloc[-5:].mean() / volume.iloc[:-5].mean())
        score = float(np.mean([float(fii > 0), float(dii > 0),
                               float(result["accumulation_proxy"]), np.clip(volume_ratio - 1., 0, 1)]))
        return {**result, "flow_status": "OBSERVED_INSTRUMENT_FLOW", "flow_score": score,
                "fii_net_inr": fii, "dii_net_inr": dii, "volume_expansion": volume_ratio,
                "flow_sources": sorted(set(rows.source)), "flow_known_at": rows.known_at.max().isoformat()}

    def __call__(self, frames: dict[str, pd.DataFrame], date: pd.Timestamp,
                 config: SystemConfig) -> list[V2Candidate]:
        date = pd.Timestamp(date).tz_localize(None).normalize()
        frames = {t: f.loc[:date] for t, f in frames.items() if date in f.index}
        start = len(self.base.decisions)
        candidates = {c.ticker: c for c in self.base(frames, date, config)}
        bench = frames[config.BENCHMARK].Close
        sessions = bench.index
        asof = date.tz_localize(IST) + pd.Timedelta(hours=18)
        ranks = relative_strength_ranks(frames, bench, config, self.settings.rs_lookback)
        breadth = breadth_context(frames, config.BENCHMARK, self.settings)
        adx = [float(f.ADX.iloc[-1]) for t, f in frames.items() if t != config.BENCHMARK
               and "ADX" in f and np.isfinite(f.ADX.iloc[-1])]
        bframe = frames[config.BENCHMARK]
        high_vol = ("ATR" in bframe and len(bframe) >= 51 and bframe.ATR.iloc[-51:-1].mean() > 0
                    and bframe.ATR.iloc[-1] / bframe.ATR.iloc[-51:-1].mean() >= config.REGIME_ATR_EXPANSION)
        bull = len(bench) >= 200 and bench.iloc[-1] > bench.iloc[-200:].mean()
        regime = "UNKNOWN" if not adx else "HIGH_VOL" if high_vol else "RANGE" if np.median(adx) < config.REGIME_ADX_RANGE else "BULL" if bull else "BEAR"
        threshold = {"BULL": 20., "BEAR": 30., "HIGH_VOL": 35., "RANGE": 0., "UNKNOWN": None}[regime]
        output = []
        for decision in self.base.decisions[start:]:
            ticker = decision["ticker"].removesuffix(".NS")
            frame = frames[decision["ticker"]]
            reasons = list(decision["reasons"])
            context = {**decision["context"], **breadth, "model": "ALPHA_ROADMAP",
                       "settings_sha256": self.settings.fingerprint(), "alpha_phase": self.settings.phase,
                       "rs_percentile": ranks.get(ticker), "rs_rank_universe": len(ranks),
                       "adaptive_regime": regime, "adaptive_adx_min": threshold,
                       "meta_status": "DISABLED" if self.meta_model is None else "PENDING"}
            rank = ranks.get(ticker)
            if rank is None or rank <= self.settings.rs_min_percentile:
                reasons.append("rs_rank_unknown_or_below_threshold")
            fraction = context.get("sector_rank_fraction")
            if fraction is None or fraction * context["sector_universe_size"] > self.settings.top_sectors + 1e-9:
                reasons.append("sector_outside_top_n")
            if breadth["breadth_risk_scale"] == 0:
                reasons.append("breadth_unknown_or_defensive")
            row = frame.iloc[-1]
            candle_range = float(row.High - row.Low)
            structure = float(np.clip((row.Close - row.Low) / candle_range, 0., 1.)) if candle_range > 0 else 0.
            factors = {"relative_strength": rank / 100 if rank is not None else None,
                       "sector": 1. - fraction + 1. / context["sector_universe_size"] if fraction is not None else None,
                       "structure": structure}
            if self.settings.phase >= 2:
                earnings = self.earnings_context(ticker, date, asof, sessions)
                flow = self.flow_context(ticker, date, asof, sessions, frame)
                context.update(earnings, **flow)
                factors.update(earnings=earnings["earnings_score"], flow=flow["flow_score"])
                if (earnings["earnings_status"] != "CONSENSUS" or earnings.get("earnings_surprise", 0) is None
                        or (earnings.get("earnings_surprise") or 0) <= 0):
                    reasons.append("pead_consensus_surprise_unknown_or_nonpositive")
                if (earnings["fundamentals_status"] != "OBSERVED"
                        or earnings.get("eps_growth", 0) <= self.settings.eps_growth_min
                        or earnings.get("revenue_growth", 0) <= self.settings.revenue_growth_min
                        or earnings.get("guidance_raised") is not True):
                    reasons.append("earnings_growth_unknown_or_weak")
                if len(frame) < 200 or row.Close <= frame.Close.iloc[-200:].mean():
                    reasons.append("price_below_sma200_or_unknown")
                if flow["flow_score"] is None:
                    reasons.append("institutional_flow_unavailable")
            candidate = candidates.get(ticker)
            if self.settings.phase >= 3:
                if threshold is None or "ADX" not in frame or not np.isfinite(row.ADX) or row.ADX < threshold:
                    reasons.append("adaptive_adx_unknown_or_weak")
                if regime == "RANGE" and candidate is not None and "BREAKOUT" in candidate.strategy_id:
                    reasons.append("breakout_disabled_in_range")
            weight_sum = sum(WEIGHTS[k] for k in factors)
            score = (100 * sum(WEIGHTS[k] * float(v) for k, v in factors.items()) / weight_sum
                     if all(v is not None for v in factors.values()) else None)
            context.update(alpha_factors=factors, alpha_weights={k: WEIGHTS[k] / weight_sum for k in factors},
                           alpha_score=score)
            if score is None or score <= self.settings.min_alpha_score:
                reasons.append("alpha_score_unknown_or_below_threshold")
            if self.settings.phase >= 4 and self.meta_model is not None and candidate is not None and not reasons:
                probability = self.meta_model.predict(context, asof)
                context.update(meta_probability=probability, meta_status="TRAINING_ONLY_ESTIMATE" if probability is not None else "UNAVAILABLE")
                if probability is None or probability < self.settings.meta_min_probability:
                    reasons.append("meta_probability_unknown_or_weak")
            self.decisions.append({"date": date.isoformat(), "ticker": ticker,
                                   "status": "REJECTED" if reasons else "QUALIFIED", "reasons": reasons, "context": context})
            if reasons or candidate is None or score is None:
                continue
            output.append(replace(candidate, strategy_id=candidate.strategy_id.replace("SECTOR_V2", "ALPHA_V3"),
                                  composite=score / 100, research_context=context))
        return sorted(output, key=lambda c: (-c.composite, c.ticker))

    def entry_guard(self, candidate: Any, date: pd.Timestamp) -> str | None:
        return self.base.entry_guard(candidate, date)
