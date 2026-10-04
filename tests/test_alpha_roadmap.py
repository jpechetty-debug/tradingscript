from dataclasses import replace
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from core.alpha_engine import AlphaSettings, SwingAlpha, breadth_context, relative_strength_ranks
from core.alpha_feeds import AlphaFeed
from core.alpha_meta import MetaModel
from core.alpha_portfolio import PortfolioLimits, allocate_candidates, bounded_quantity
from core.alpha_research import ResearchAgent, openai_generator, research_packet, validate_brief
from core.alpha_validation import StressSettings, calendar_folds, monte_carlo
from core.backtest import ZERO_COST_MODEL
from core.config import CONFIG
from core.swing_replay import replay_swing
from tests.test_swing_v2 import fixture, earning
from scripts.research_alpha import run


def full_fixture():
    frames, feeds, date = fixture()
    for i in range(4):
        weak = frames["TEST.NS"].copy()
        weak.Close = np.linspace(80, 85, len(weak))
        weak.Open = weak.Close - .2
        weak.High = weak.Close + .4
        weak.Low = weak.Close - .8
        frames[f"WEAK{i}.NS"] = weak
    for frame in frames.values():
        frame["ADX"] = 26.
    frames["TEST.NS"].loc[frames["TEST.NS"].index[-5:], "Volume"] *= 2
    stamp = (date.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=16)).isoformat()
    feeds.tables["earnings"] = AlphaFeed("earnings", pd.DataFrame([earning(date, actual_eps=15.)]))
    feeds.tables["fundamentals"] = AlphaFeed("fundamentals", pd.DataFrame([
        dict(ticker="TEST", period_end=date - pd.Timedelta(days=30), eps_growth=.5, revenue_growth=.4,
             guidance_raised=True, published_at=stamp, ingested_at=stamp, source="synthetic-results")]))
    deliveries, flows = [], []
    for i, session in enumerate(frames["TEST.NS"].index[-25:]):
        ts = (session.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=16)).isoformat()
        common = dict(ticker="TEST", session=session, published_at=ts, ingested_at=ts, source="synthetic-observation")
        deliveries.append(dict(**common, total_qty=1000., delivery_qty=500. if i < 20 else 750.,
                               turnover=1e8 if i < 20 else 2e8))
        flows.append(dict(**common, fii_net_inr=100., dii_net_inr=50.))
    feeds.tables["delivery"] = AlphaFeed("delivery", pd.DataFrame(deliveries))
    feeds.tables["flows"] = AlphaFeed("flows", pd.DataFrame(flows))
    return frames, feeds, date


def candidate(ticker="NEW", sector="BANK", scale=1.):
    return SimpleNamespace(ticker=ticker, sector=sector, composite=.95, strategy_id="SWING_ALPHA_V3_BREAKOUT",
                           signal_time="2023-01-01", entry=100., entry_min=99., entry_max=101., stop=95.,
                           t1=110., t2=115., time_stop_bars=10, prob_win=0., expectancy_r=0.,
                           research_context={"breadth_risk_scale": scale})


@pytest.mark.parametrize("phase", [1, 2, 3, 4])
def test_all_phases_qualify_with_complete_observed_evidence(phase):
    frames, feeds, date = full_fixture()
    engine = SwingAlpha(feeds, AlphaSettings(phase=phase))
    picks = engine(frames, date, CONFIG)
    assert len(picks) == 1
    assert picks[0].ticker == "TEST" and "ALPHA_V3" in picks[0].strategy_id
    assert picks[0].composite > .85
    assert picks[0].research_context["rs_percentile"] == 100.
    assert picks[0].research_context["breadth_risk_scale"] == 1.
    if phase >= 2:
        assert picks[0].research_context["flow_status"] == "OBSERVED_INSTRUMENT_FLOW"


def test_future_prices_and_feed_revisions_cannot_change_historical_alpha():
    frames, feeds, date = full_fixture()
    first = SwingAlpha(feeds)(frames, date, CONFIG)
    future_date = date + pd.offsets.BDay()
    future = frames["TEST.NS"].iloc[[-1]].copy()
    future.index = pd.DatetimeIndex([future_date])
    future.Close = .01
    frames["TEST.NS"] = pd.concat([frames["TEST.NS"], future])
    old = feeds.tables["fundamentals"].frame.drop(columns="known_at")
    revision = old.iloc[[0]].copy()
    revision.eps_growth = -1.
    revision.published_at = future_date.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=16)
    revision.ingested_at = revision.published_at
    feeds.tables["fundamentals"] = AlphaFeed("fundamentals", pd.concat([old, revision]))
    assert SwingAlpha(feeds)(frames, date, CONFIG) == first


@pytest.mark.parametrize("missing,reason", [("flows", "institutional_flow_unavailable"),
                                           ("fundamentals", "earnings_growth_unknown_or_weak"),
                                           ("earnings", "pead_consensus_surprise_unknown_or_nonpositive")])
def test_missing_phase2_evidence_rejects_without_renormalization(missing, reason):
    frames, feeds, date = full_fixture()
    del feeds.tables[missing]
    engine = SwingAlpha(feeds)
    assert engine(frames, date, CONFIG) == []
    decision = next(d for d in engine.decisions if d["ticker"] == "TEST")
    assert reason in decision["reasons"]
    assert decision["context"]["alpha_score"] is None
    assert len(SwingAlpha(feeds, AlphaSettings(phase=1))(frames, date, CONFIG)) == 1


def test_adaptive_models_disable_range_breakout_and_raise_high_vol_adx():
    frames, feeds, date = full_fixture()
    for frame in frames.values():
        frame.ADX = 15.
    engine = SwingAlpha(feeds)
    assert engine(frames, date, CONFIG) == []
    assert "breakout_disabled_in_range" in next(d for d in engine.decisions if d["ticker"] == "TEST")["reasons"]
    frames, feeds, date = full_fixture()
    frames[CONFIG.BENCHMARK].loc[date, "ATR"] = 3.
    engine = SwingAlpha(feeds)
    assert engine(frames, date, CONFIG) == []
    decision = next(d for d in engine.decisions if d["ticker"] == "TEST")
    assert decision["context"]["adaptive_adx_min"] == 35.
    assert "adaptive_adx_unknown_or_weak" in decision["reasons"]


def test_rs_ranking_handles_zero_benchmark_return_and_excludes_illiquid_stocks():
    frames, _, _ = full_fixture()
    bench = frames[CONFIG.BENCHMARK].Close * 0 + 100.
    frames["WEAK0.NS"].Volume = 0
    ranks = relative_strength_ranks(frames, bench, CONFIG, 63)
    assert ranks["TEST"] == 100.
    assert "WEAK0" not in ranks
    assert len(ranks) == 4


def test_rs_ranks_require_identical_benchmark_sessions():
    frames, _, _ = full_fixture()
    frames["TEST.NS"] = frames["TEST.NS"].drop(frames["TEST.NS"].index[-10])
    ranks = relative_strength_ranks(frames, frames[CONFIG.BENCHMARK].Close, CONFIG, 63)
    assert "TEST" not in ranks


@pytest.mark.parametrize("fraction,scale", [(.8, 1.), (.6, .5), (.4, .5), (.2, 0.)])
def test_sma50_breadth_boundary_bands(fraction, scale):
    frames = {}
    for i in range(5):
        frames[f"S{i}.NS"] = pd.DataFrame({"Close": [100.] * 49 + [101. if i < int(fraction * 5) else 99.]})
    context = breadth_context(frames, "BENCH", AlphaSettings())
    assert context["breadth_risk_scale"] == scale
    frames["MISSING.NS"] = pd.DataFrame({"Close": [100.]})
    assert breadth_context(frames, "BENCH", AlphaSettings())["breadth_risk_scale"] == 0.


def test_supplied_feeds_reject_nan_unknown_guidance_and_backdated_publication():
    _, feeds, _ = full_fixture()
    data = feeds.tables["fundamentals"].frame.drop(columns="known_at")
    for col, value in [("eps_growth", np.inf), ("guidance_raised", "unknown")]:
        invalid = data.copy()
        invalid[col] = value
        with pytest.raises(ValueError):
            AlphaFeed("fundamentals", invalid)
    invalid = data.copy()
    invalid.published_at = invalid.period_end.dt.tz_localize("UTC") - pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="before its session"):
        AlphaFeed("fundamentals", invalid)


def test_earnings_and_growth_must_describe_the_same_reporting_period():
    frames, feeds, date = full_fixture()
    feeds.tables["fundamentals"].frame["period_end"] -= pd.Timedelta(days=90)
    engine = SwingAlpha(feeds)
    assert engine(frames, date, CONFIG) == []
    assert next(d for d in engine.decisions if d["ticker"] == "TEST")["context"]["fundamentals_status"] == "PERIOD_MISMATCH"


def test_allocation_uses_remaining_sector_notional_heat_and_preserves_held_book():
    config = replace(CONFIG, CAPITAL_INR=100_000., RISK_PER_TRADE_INR=10_000., SWING_RISK_FRACTION=.1,
                     SWING_MAX_EXPOSURE_FRACTION=1., MAX_PORTFOLIO_RISK_INR=100_000.)
    rng = np.random.default_rng(1)
    returns = pd.DataFrame({"OLD.NS": rng.normal(size=60), "NEW.NS": rng.normal(size=60)})
    held = [dict(ticker="OLD", sector="BANK", shares=200, price=100., risk_inr=1000.)]
    original = [dict(row) for row in held]
    allocations = allocate_candidates([candidate()], config, returns, held)
    assert allocations[0]["shares"] == 50  # 5,000 INR remaining sector capacity.
    assert held == original
    held[0]["risk_inr"] = 6000.
    assert allocate_candidates([candidate()], config, returns, held)[0]["reason"] == "heat_cap"


def test_unknown_correlation_blocks_new_allocation_and_held_duplicates_are_invalid():
    held = [dict(ticker="OLD", sector="IT", shares=10, price=100., risk_inr=50.)]
    out = allocate_candidates([candidate()], CONFIG, pd.DataFrame({"OLD": range(5), "NEW": range(5)}), held)
    assert out[0]["shares"] == 0 and out[0]["reason"] == "correlation_unknown"
    with pytest.raises(ValueError, match="Duplicate held"):
        allocate_candidates([], CONFIG, pd.DataFrame(), held + held)


def test_correlated_candidates_cannot_share_the_book():
    returns = pd.DataFrame({"OLD": np.linspace(-.01, .01, 60), "NEW": np.linspace(-.01, .01, 60)})
    held = [dict(ticker="OLD", sector="IT", shares=10, price=100., risk_inr=50.)]
    decision = allocate_candidates([candidate()], CONFIG, returns, held)[0]
    assert decision["shares"] == 0 and decision["reason"] == "correlation_cap"


def test_sizing_honors_half_risk_and_existing_absolute_risk_cap():
    assert bounded_quantity(100, 100, 95, 100_000, 0, 0, 50_000, PortfolioLimits(), .5)[0] == 50
    assert bounded_quantity(100, 100, 95, 100_000, 490, 0, 500, PortfolioLimits())[0] == 2
    assert bounded_quantity(100, 100, 95, 100_000, 0, 0, 500, PortfolioLimits(), 0)[0] == 0


def test_replay_applies_new_entry_percentage_limits_and_half_breadth():
    frames, _, date = full_fixture()
    for i in (1, 2):
        day = date + pd.offsets.BDay(i)
        for frame in frames.values():
            frame.loc[day] = frame.loc[date]
        frames["TEST.NS"].loc[day, ["Open", "High", "Low", "Close"]] = [100., 100.5, 99.9, 100.2]
    pick = candidate("TEST", scale=.5)
    config = replace(CONFIG, CAPITAL_INR=100_000., RISK_PER_TRADE_INR=10_000., SWING_RISK_FRACTION=.1,
                     SWING_MAX_EXPOSURE_FRACTION=1., MAX_PORTFOLIO_RISK_INR=100_000.)
    def provider(f, d, c):
        return [pick] if d == date else []
    replay = replay_swing(frames, config, date + pd.offsets.BDay(), date + pd.offsets.BDay(2), ZERO_COST_MODEL,
                          prepared=frames, warm_history=True, signal_provider=provider, portfolio_limits=PortfolioLimits())
    assert replay.result.trades[0].shares == 250
    assert replay.equity.portfolio_heat.max() <= .06
    # Marks may drift above the entry cap; entry was exactly 25%.
    assert replay.diagnostics.loc[replay.diagnostics.reason == "filled", "shares"].iloc[0] == 250


def test_calendar_windows_are_disjoint_rolling_and_partial_history_is_explicit():
    sessions = pd.bdate_range("2015-01-01", "2023-06-30")
    folds = calendar_folds(sessions)
    assert folds[0].train_start.year == 2015 and folds[0].test_start.year == 2020
    assert all(f.train_end < f.test_start for f in folds)
    assert all(a.test_end < b.test_start for a, b in zip(folds, folds[1:]))
    assert folds[-1].partial and not folds[0].partial
    assert calendar_folds(pd.bdate_range("2025-01-01", "2026-10-01")) == []


def test_monte_carlo_is_repeatable_includes_initial_equity_and_reports_empty_data():
    settings = StressSettings(simulations=100, risk_fraction=.1, slippage_r_mean=0,
                              gap_probability=0, missed_fill_probability=0, block_size=1)
    losses = [SimpleNamespace(r_multiple=-1.) for _ in range(10)]
    first = monte_carlo(losses, settings)
    assert first == monte_carlo(losses, settings)
    assert first["worst_drawdown"] == pytest.approx(1 - .9 ** 10)
    assert first["risk_of_ruin"] == 1.
    assert first["loss_var95"] == pytest.approx(1 - .9 ** 10)
    assert monte_carlo([])["status"] == "UNAVAILABLE_NO_CLOSED_TRADES"
    with pytest.raises(ValueError, match="finite"):
        monte_carlo([SimpleNamespace(r_multiple=np.nan)])


def training_trades(n=100):
    rows = []
    dates = pd.bdate_range("2021-01-01", periods=n + 25)
    for i, date in enumerate(dates[:n]):
        strength = .3 if i % 2 else .9
        context = dict(observed_at=(date.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=18)).isoformat(),
                       alpha_factors={key: strength for key in ("relative_strength", "earnings", "flow", "sector", "structure")},
                       breadth_sma50=.8, adaptive_regime="BULL")
        rows.append(SimpleNamespace(research_context=context, exit_date=dates[i + 10], r_multiple=-1. if i % 2 else 2.))
    return rows, dates[-1].tz_localize("Asia/Kolkata") + pd.Timedelta(hours=18)


def test_meta_model_purges_overlapping_labels_and_excludes_future_outcomes():
    trades, cutoff = training_trades()
    model = MetaModel()
    report = model.fit(trades, cutoff)
    assert report["status"] == "FITTED_REQUIRES_OUT_OF_SAMPLE_VALIDATION"
    assert report["models"]["GLOBAL"]["purged_samples"] > 0
    prediction = model.predict(trades[0].research_context, cutoff)
    assert 0 < prediction < 1
    future = SimpleNamespace(research_context=trades[-1].research_context, exit_date=cutoff + pd.Timedelta(days=1), r_multiple=100.)
    other = MetaModel()
    assert other.fit(trades + [future], cutoff) == report
    forced_boundary = SimpleNamespace(**vars(trades[0]), exit_reason="DATA_END")
    assert MetaModel().fit(trades + [forced_boundary], cutoff) == report
    assert other.predict(trades[0].research_context, cutoff) == prediction
    with pytest.raises(ValueError, match="before its training"):
        model.predict(trades[0].research_context, cutoff - pd.Timedelta(days=1))
    assert MetaModel().fit(trades[:3], cutoff)["status"] == "INSUFFICIENT_TRAINING_EVIDENCE"


def test_enabled_but_unfitted_meta_gate_rejects_candidates():
    frames, feeds, date = full_fixture()
    engine = SwingAlpha(feeds, meta_model=MetaModel())
    assert engine(frames, date, CONFIG) == []
    assert "meta_probability_unknown_or_weak" in next(d for d in engine.decisions if d["ticker"] == "TEST")["reasons"]


def test_research_filters_future_documents_and_rejects_fabricated_citations():
    frames, feeds, date = full_fixture()
    pick = SwingAlpha(feeds)(frames, date, CONFIG)[0]
    documents = [dict(ticker="TEST", topic="quarterly_results", source="original", text="Revenue rose 40 percent.",
                      published_at=pick.research_context["observed_at"], ingested_at=pick.research_context["observed_at"])]
    documents.append({**documents[0], "ingested_at": (date + pd.Timedelta(days=1)).tz_localize("UTC").isoformat()})
    packet = research_packet(pick, documents)
    assert len(packet["sources"]) == 1
    assert "promoter_holding" in packet["missing_topics"]
    brief = {key: [] for key in ("bull_thesis", "bear_thesis", "risk_factors", "catalysts", "unknowns")}
    brief["bull_thesis"] = [dict(text="Revenue growth supports the thesis.", source_id="S1", quote="Revenue rose 40 percent.")]
    assert ResearchAgent(lambda p: brief).run(pick, documents)["status"] == "AI_DRAFT_REQUIRES_REVIEW"
    brief["bull_thesis"][0]["quote"] = "Invented promoter ownership."
    with pytest.raises(ValueError, match="fabricated"):
        validate_brief(brief, packet)
    assert ResearchAgent().run(pick)["status"] == "EVIDENCE_PACKET_ONLY"


def test_responses_adapter_requires_completed_structured_output(monkeypatch):
    seen = []
    brief = {key: [] for key in ("bull_thesis", "bear_thesis", "risk_factors", "catalysts", "unknowns")}
    def post(url, **kwargs):
        seen.append((url, kwargs))
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            "status": "completed", "output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(brief)}]}]})
    monkeypatch.setattr("core.alpha_research.requests.post", post)
    assert openai_generator("explicit-model", "dummy-test-key")({"sources": []}) == brief
    assert seen[0][1]["json"]["store"] is False
    assert seen[0][1]["json"]["text"]["format"]["strict"] is True
    monkeypatch.setattr("core.alpha_research.requests.post", lambda *a, **k: SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: {"status": "incomplete"}))
    with pytest.raises(ValueError, match="did not complete"):
        openai_generator("explicit-model", "dummy-test-key")({})


@pytest.mark.parametrize("mode", ["scan", "replay", "walk-forward"])
def test_cli_workflows_export_provenance_and_honest_unavailable_results(tmp_path, mode):
    frames, _, date = full_fixture()
    snapshot = tmp_path / "prices.pkl"
    snapshot.write_bytes(pickle.dumps(frames))
    out = tmp_path / mode
    args = SimpleNamespace(feeds=None, snapshot=snapshot, end=None, phase=4, simulations=100, seed=7,
                           mode=mode, out=out, meta=False, ai_model=None, training_trades=None,
                           held_book=None, documents=None, start=(date - pd.Timedelta(days=5)).isoformat(),
                           train_years=5, embargo_sessions=5)
    report = run(args)
    assert len(report["provenance"]["snapshot_sha256"]) == 64
    assert json.loads((out / "report.json").read_text())["status"] == "RESEARCH_ONLY_EDGE_UNPROVEN"
    if mode == "scan":
        assert report["qualified"] == 0 and (out / "research.json").exists()
        assert report["rejection_counts"]["institutional_flow_unavailable"] > 0
    elif mode == "replay":
        assert report["monte_carlo"]["status"] == "UNAVAILABLE_NO_CLOSED_TRADES"
        assert json.loads((out / "closed_trades.json").read_text()) == []
    else:
        assert report["validation_status"] == "UNAVAILABLE_INSUFFICIENT_CALENDAR_HISTORY"
