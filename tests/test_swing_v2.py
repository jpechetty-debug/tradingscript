from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.alpha_feeds import AlphaFeed, AlphaFeeds
from core.backtest import ZERO_COST_MODEL
from core.config import CONFIG
from core.database import SqliteDatabase
from core.paper_ledger import PaperLedger
from core.swing import detect_swing_setup, detect_swing_trigger
from core.swing_replay import replay_swing
from core.swing_v2 import SwingV2, V2Settings


def fixture():
    dates = pd.bdate_range("2023-01-02", periods=200)
    close = np.linspace(80, 100, 200)
    stock = pd.DataFrame({"Open": close - .2, "Close": close, "High": close + .4, "Low": close - .8,
                          "ATR": 1., "EMA_20": close - .5, "EMA_50": close - 2, "EMA_200": close - 5,
                          "RVol_20": 1.3, "Volume": 10_000_000.}, index=dates)
    stock.loc[dates[-1], ["Open", "Close", "High", "Low"]] = [100., 100.6, 100.8, 99.8]
    frames = {CONFIG.BENCHMARK: stock.copy(), "TEST.NS": stock}
    rows = []
    for name, growth in [("NIFTY500", .1), ("BANK", .6), ("IT", .4), ("AUTO", .2), ("PHARMA", .05)]:
        for i, date in enumerate(dates):
            timestamp = (date.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=16)).isoformat()
            rows.append(dict(index=name, session=date, close=100 * (1 + growth * i / 199),
                             published_at=timestamp, ingested_at=timestamp, source="synthetic"))
    early = (dates[0].tz_localize("Asia/Kolkata") + pd.Timedelta(hours=16)).isoformat()
    base = dict(ticker="TEST", effective_date=dates[0], valid_until=dates[-1] + pd.Timedelta(days=10),
                published_at=early, ingested_at=early, source="synthetic")
    feeds = AlphaFeeds(sector_universe=("BANK", "IT", "AUTO", "PHARMA"), tables={
        "indices": AlphaFeed("indices", pd.DataFrame(rows)),
        "sectors": AlphaFeed("sectors", pd.DataFrame([{**base, "sector_index": "BANK"}])),
        "surveillance": AlphaFeed("surveillance", pd.DataFrame([{**base, "state": "CLEAR"}]))})
    return frames, feeds, dates[-1]


def earning(date, **changes):
    announced = date.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=16)
    return dict(ticker="TEST", period_end=date - pd.Timedelta(days=30), announced_at=announced.isoformat(),
                expected_at=(announced - pd.Timedelta(days=2)).isoformat(),
                expectation_ingested_at=(announced - pd.Timedelta(days=1)).isoformat(),
                published_at=announced.isoformat(), ingested_at=announced.isoformat(), source="synthetic",
                actual_eps=12., expected_eps=10., expectation_kind="CONSENSUS") | changes


def test_v2_is_independent_of_v1_filters_and_future_prices():
    frames, feeds, date = fixture()
    frames["TEST.NS"].loc[date, "EMA_200"] = 200.
    assert detect_swing_setup(frames["TEST.NS"], frames[CONFIG.BENCHMARK].Close, CONFIG) is None
    assert detect_swing_trigger(frames["TEST.NS"], CONFIG) is not None
    engine = SwingV2(feeds)
    picks = engine(frames, date, CONFIG)
    assert len(picks) == 1 and picks[0].strategy_id == "SWING_SECTOR_V2_BREAKOUT"
    future = frames["TEST.NS"].iloc[[-1]].copy()
    future.index = pd.DatetimeIndex([date + pd.offsets.BDay()])
    future.Close = 1.
    frames["TEST.NS"] = pd.concat([frames["TEST.NS"], future])
    assert engine(frames, date, CONFIG) == picks
    assert picks[0].research_context["sector_rank_fraction"] == .25


def test_complete_sector_universe_ties_and_optional_gates():
    frames, feeds, date = fixture()
    engine = SwingV2(feeds)
    feeds.tables["indices"].frame = feeds.tables["indices"].frame.loc[feeds.tables["indices"].frame["index"] != "PHARMA"]
    assert engine(frames, date, CONFIG) == []
    assert "sector_history_unavailable" in engine.decisions[-1]["reasons"]
    frames, feeds, date = fixture()
    table = feeds.tables["indices"].frame
    table.loc[table["index"] == "IT", "close"] = table.loc[table["index"] == "BANK", "close"].to_numpy()
    assert SwingV2(feeds)(frames, date, CONFIG) == []  # Two tied leaders occupy rank 2/4.
    frames, feeds, date = fixture()
    assert len(SwingV2(feeds)(frames, date, CONFIG)) == 1
    assert SwingV2(feeds, V2Settings(vix_gate=True))(frames, date, CONFIG) == []
    assert SwingV2(feeds, V2Settings(delivery_gate=True))(frames, date, CONFIG) == []
    assert len(SwingV2(feeds, V2Settings(market_gate=True, breadth_gate=True))(frames, date, CONFIG)) == 1


def test_surveillance_expiry_and_opening_recheck():
    frames, feeds, date = fixture()
    engine = SwingV2(feeds)
    candidate = engine(frames, date, CONFIG)[0]
    next_date = date + pd.offsets.BDay()
    assert engine.entry_guard(candidate, next_date) is None
    data = feeds.tables["surveillance"].frame.drop(columns="known_at").copy()
    restriction = dict(data.iloc[0])
    restriction.update(effective_date=next_date, state="ASM",
                       published_at=(next_date.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9)).isoformat(),
                       ingested_at=(next_date.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9, minutes=14)).isoformat())
    feeds.tables["surveillance"] = AlphaFeed("surveillance", pd.concat([data, pd.DataFrame([restriction])]))
    assert engine.entry_guard(candidate, next_date) == "entry_surveillance_asm"
    feeds.tables["surveillance"].frame.loc[:, "valid_until"] = date - pd.Timedelta(days=1)
    assert engine(frames, date, CONFIG) == []
    assert "surveillance_unavailable" in engine.decisions[-1]["reasons"]


def test_pead_expectations_revisions_and_independent_sector_selection():
    frames, feeds, date = fixture()
    feeds.tables["earnings"] = AlphaFeed("earnings", pd.DataFrame([earning(date)]))
    pead = SwingV2(feeds, V2Settings(model="PEAD"))
    picks = pead(frames, date, CONFIG)
    assert len(picks) == 1 and picks[0].time_stop_bars == 40
    assert picks[0].research_context["earnings_surprise"] == pytest.approx(.2)
    assert picks[0].research_context["earnings_age_sessions"] == 0  # After-close announcement.
    later = (date.tz_localize("Asia/Kolkata") + pd.Timedelta(days=1)).isoformat()
    revised = earning(date, actual_eps=1., published_at=later, ingested_at=later)
    feeds.tables["earnings"] = AlphaFeed("earnings", pd.DataFrame([earning(date), revised]))
    assert pead(frames, date, CONFIG) == picks
    feeds.tables["indices"].frame = feeds.tables["indices"].frame.iloc[:0]
    assert len(pead(frames, date, CONFIG)) == 1
    assert SwingV2(feeds, V2Settings(model="COMBINED"))(frames, date, CONFIG) == []
    feeds.tables["earnings"] = AlphaFeed("earnings", pd.DataFrame([earning(date, expectation_kind="SEASONAL")]))
    assert pead(frames, date, CONFIG) == []
    feeds.tables["earnings"] = AlphaFeed("earnings", pd.DataFrame([earning(date, expected_eps=0.)]))
    assert pead(frames, date, CONFIG) == []


@pytest.mark.parametrize("change", [
    {"expected_at": "2024-01-01T16:00:00+05:30"},
    {"ingested_at": "2023-01-01T00:00:00+05:30"},
    {"published_at": "2023-01-01T00:00:00"},
    {"actual_eps": np.inf},
])
def test_invalid_earnings_provenance_rejected(change):
    _, _, date = fixture()
    with pytest.raises(ValueError):
        AlphaFeed("earnings", pd.DataFrame([earning(date, **change)]))


def test_delivery_proxy_needs_observed_quantity_and_turnover():
    frames, feeds, date = fixture()
    dates = frames[CONFIG.BENCHMARK].index[-25:]
    data = []
    for i, day in enumerate(dates):
        stamp = (day.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=17)).isoformat()
        data.append(dict(ticker="TEST", session=day, delivery_qty=30 if i < 20 else 60,
                         total_qty=100, turnover=1000 if i < 20 else 2000,
                         published_at=stamp, ingested_at=stamp, source="synthetic"))
    feeds.tables["delivery"] = AlphaFeed("delivery", pd.DataFrame(data))
    engine = SwingV2(feeds, V2Settings(delivery_gate=True))
    picks = engine(frames, date, CONFIG)
    assert len(picks) == 1 and picks[0].research_context["accumulation_proxy"] is True
    feeds.tables["delivery"].frame.loc[:, "known_at"] += pd.Timedelta(days=365)
    assert engine(frames, date, CONFIG) == []


def test_v2_cash_replay_bypasses_legacy_scorer_and_guard_blocks_entries(monkeypatch):
    frames, feeds, date = fixture()
    next_date = date + pd.offsets.BDay()
    for ticker, frame in list(frames.items()):
        next_bar = frame.iloc[[-1]].copy()
        next_bar.index = pd.DatetimeIndex([next_date])
        next_bar.loc[:, ["Open", "High", "Low", "Close"]] = [100.6, 101., 100., 100.8]
        frames[ticker] = pd.concat([frame, next_bar])
    monkeypatch.setattr("core.swing_replay.score_universe", lambda **kw: pytest.fail("V2 must bypass V1"))
    config = replace(CONFIG, MAX_CORR=1.)
    engine = SwingV2(feeds)
    result = replay_swing(frames, config, next_date, next_date, ZERO_COST_MODEL,
                          prepared=frames, warm_history=True, signal_provider=engine, entry_guard=engine.entry_guard)
    assert len(result.result.trades) == 1
    trade = result.result.trades[0]
    assert trade.entry_date == next_date and trade.strategy_id.startswith("SWING_SECTOR_V2")
    assert trade.research_context["model"] == "SECTOR" and result.metrics["minimum_cash"] >= 0
    assert result.metrics["final_nav"] - config.CAPITAL_INR == pytest.approx(trade.r_multiple * trade.risk_inr)
    blocked = replay_swing(frames, config, next_date, next_date, ZERO_COST_MODEL,
                           prepared=frames, warm_history=True, signal_provider=engine,
                           entry_guard=lambda *a: "entry_surveillance_gsm")
    assert blocked.result.trades == []
    assert blocked.metrics["diagnostics"]["entry_surveillance_gsm"] == 1


def test_v2_paper_first_observation_and_protocol_identity(tmp_path):
    frames, feeds, date = fixture()
    candidate = SwingV2(feeds)(frames, date, CONFIG)[0]
    db = SqliteDatabase(tmp_path / "v2.db")
    ledger = PaperLedger(db)
    parameters = {"v2": "SECTOR", "protocol_sha256": "original"}
    first = ledger.record_research_scan([candidate], date.isoformat(), parameters, {"snapshot": "one"})
    candidate.research_context["sector_rank_fraction"] = .1
    assert ledger.record_research_scan([candidate], date.isoformat(), parameters, {"snapshot": "two"}) == first
    assert first[0]["payload"]["prob_win"] is None
    ledger.record_research_scan([candidate], date.isoformat(), parameters | {"protocol_sha256": "new"}, {})
    ledger.record_research_scan([], date.isoformat(), parameters, {})
    events = db.fetch_paper_events()
    assert [e["event_type"] for e in events].count("SIGNAL") == 2
    assert events[-1]["payload"]["qualified_signals"] == 0
    assert db.count_executed_trades() == 0
