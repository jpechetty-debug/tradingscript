from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import core.swing_replay as replay
import core.swing_research as research
from core.backtest import ZERO_COST_MODEL
from core.config import CONFIG
from core.research_feeds import EarningsCalendar, MembershipHistory
from core.swing_exits import ExitPolicy
from core.swing_analytics import evidence


def test_leadership_ranks_use_liquid_universe_and_only_known_data(monkeypatch):
    dates = pd.bdate_range("2023-01-02", periods=300)
    frames = {}
    for ticker, slope in ((CONFIG.BENCHMARK, .05), ("A.NS", .3), ("B.NS", .1), ("C.NS", .15), ("ILLIQUID.NS", .5)):
        price = 100 + slope * np.arange(300)
        frames[ticker] = pd.DataFrame({"Close": price, "High": price + 1, "Low": price - 1,
                                      "Volume": 0. if ticker == "ILLIQUID.NS" else 1e6,
                                      "EMA_50": pd.Series(price).ewm(span=50).mean().values,
                                      "EMA_200": pd.Series(price).ewm(span=200).mean().values}, index=dates)
    monkeypatch.setattr(research, "TICKER_TO_SECTOR", {t: "IT" for t in frames})
    config = replace(CONFIG, ADV_TURNOVER_FLOOR=1., ADV_SHARE_FLOOR=1.)
    first = research.leadership_snapshot(frames, config, dates[270])
    assert set(first) == {"A.NS", "B.NS", "C.NS"}
    assert first["A.NS"]["rs_63_percentile"] == 100
    assert first["A.NS"]["rs_252_rank_count"] == 3
    assert first["A.NS"]["sector_peer_count"] == 2
    assert first["A.NS"]["sector_rs_63"] < first["A.NS"]["rs_63"]
    for frame in frames.values():
        frame.loc[dates[271]:, ["Close", "High", "Low"]] *= 100
    assert research.leadership_snapshot(frames, config, dates[270]) == first
    assert research.percentile_bucket(100.) == "90-100"
    assert research.percentile_bucket(None) == "UNKNOWN"


def test_excursions_do_not_count_exit_day_high_as_known_pre_stop_profit():
    dates = pd.bdate_range("2024-01-01", periods=2)
    frame = pd.DataFrame({"Open": [100., 100.], "High": [104., 112.], "Low": [99., 88.], "Close": [102., 90.]}, index=dates)
    trade = SimpleNamespace(entry=100., stop=90., direction="LONG", entry_date=dates[0], exit_date=dates[1], exit_price=90., exit_reason="STOP")
    stats = research.trade_excursions(trade, frame)
    assert stats["mfe_r_lower"] == .4 and stats["mfe_r_upper"] == 1.2
    assert stats["mae_r_lower"] == 1. and stats["mae_r_upper"] == 1.2
    assert stats["exit_day_order_unknown"]
    trade.exit_reason = "TIME"
    assert research.trade_excursions(trade, frame)["mfe_r_lower"] == 1.2
    trade.exit_reason, trade.exit_price = "STOP", 80.
    frame.loc[dates[1], "Open"] = 80.
    gap = research.trade_excursions(trade, frame)
    assert gap["mae_r_upper"] == 2. and gap["mfe_r_upper"] == .4


def test_asof_feeds_ignore_later_knowledge_and_never_certify_missing_events():
    sessions = pd.bdate_range("2024-01-01", periods=10)
    calendar = EarningsCalendar(pd.DataFrame([{"ticker": "A.NS", "event_date": "2024-01-08", "known_at": "2024-01-03T18:00:00+05:30"}]))
    assert calendar.status("A.NS", pd.Timestamp("2024-01-02T18:00:00+05:30"), sessions[2], sessions).startswith("NO_KNOWN")
    assert calendar.status("A.NS", pd.Timestamp("2024-01-03T19:00:00+05:30"), sessions[3], sessions).startswith("KNOWN_EARNINGS")
    membership = MembershipHistory(pd.DataFrame([
        {"ticker": "A.NS", "effective_date": "2024-01-01", "action": "ADD", "known_at": "2023-12-20T00:00:00Z"},
        {"ticker": "A.NS", "effective_date": "2024-01-04", "action": "REMOVE", "known_at": "2024-01-03T00:00:00Z"}]))
    assert membership.members(pd.Timestamp("2024-01-02T00:00:00Z"), sessions[4]) == {"A.NS"}
    assert membership.members(pd.Timestamp("2024-01-03T00:00:00Z"), sessions[4]) == set()
    assert membership.missing_prices({"DELISTED.NS"}, {"A.NS"}) == {"DELISTED.NS"}
    with pytest.raises(ValueError, match="timezone"):
        EarningsCalendar(pd.DataFrame([{"ticker": "A.NS", "event_date": "2024-01-08", "known_at": "2024-01-03"}]))


def _exit_replay(monkeypatch, mode, *, stop_first=False, gap=False, commission=0.):
    dates = pd.bdate_range("2024-01-01", periods=4)
    trail = mode in ("ATR_TRAIL", "CHANDELIER")
    frame = pd.DataFrame({"Open": [100., 100., 105. if trail else 140. if gap else 110., 120.],
                          "High": [101., 115. if trail else 111., 145. if gap else 131., 121.],
                          "Low": [99., 89. if stop_first else 95., 104. if trail else 105., 119.],
                          "Close": [100., 110., 129., 120.], "ATR": 2.}, index=dates)
    monkeypatch.setattr(replay, "add_indicators", lambda f, c: f)
    monkeypatch.setattr(replay, "compute_sector_rs", lambda *a: {})
    monkeypatch.setattr(replay, "compute_breadth", lambda *a: .7)
    monkeypatch.setattr(replay, "classify_regime", lambda *a, **k: SimpleNamespace(label="TREND_UP"))
    candidate = SimpleNamespace(ticker="A", sector="IT", strategy_id="SWING_BREAKOUT_V1", signal_time=dates[0].isoformat(),
                                entry=100., entry_min=99., entry_max=101., stop=90., t1=120., t2=130., time_stop_bars=4,
                                expectancy_r=.5, composite=.7, prob_win=.6, is_watchlist=False)
    monkeypatch.setattr(replay, "score_universe", lambda **kw: [candidate] if kw["bench"].index[-1] == dates[0] else [])
    config = replace(CONFIG, CAPITAL_INR=10000., SWING_SETUP_ENABLED=True, SWING_RISK_FRACTION=.01, SWING_MAX_EXPOSURE_FRACTION=.5)
    return replay.replay_swing({CONFIG.BENCHMARK: frame.copy(), "A.NS": frame}, config, dates[0], dates[-1],
                              replace(ZERO_COST_MODEL, commission_inr=commission), exit_policy=ExitPolicy(mode, 2. if mode == "ATR_TRAIL" else 3.))


def test_scale_out_reconciles_cash_risk_costs_and_original_quantity(monkeypatch):
    result = _exit_replay(monkeypatch, "SCALE_OUT", commission=2.)
    trade, = result.result.trades
    assert trade.shares == 10 and trade.risk_inr == 100.
    assert [(leg["shares"], leg["price"]) for leg in trade.exit_fills] == [(5, 110.), (5, 130.)]
    assert trade.r_multiple == pytest.approx(1.94)
    assert result.metrics["final_nav"] == pytest.approx(10194.)
    assert sum(t.r_multiple * t.risk_inr for t in result.result.trades) == pytest.approx(result.metrics["final_nav"] - 10000.)


@pytest.mark.parametrize("mode", ["ATR_TRAIL", "CHANDELIER"])
def test_close_updated_trails_apply_only_next_session_and_gap_fills_use_open(monkeypatch, mode):
    result = _exit_replay(monkeypatch, mode)
    trade, = result.result.trades
    assert trade.bars_held == 1 and trade.exit_reason == "TRAIL_STOP"
    assert trade.exit_price == 105. and trade.stop == 90.
    assert trade.r_multiple == .5
    assert evidence(result)["stop_out_frequency"] == 1.


def test_scale_out_stop_first_and_opening_gap_are_conservative(monkeypatch):
    stopped = _exit_replay(monkeypatch, "SCALE_OUT", stop_first=True)
    assert stopped.result.trades[0].r_multiple == -1.
    assert len(stopped.result.trades[0].exit_fills) == 1
    assert stopped.metrics["diagnostics"]["ambiguous_stop_and_target_bars"] == 1
    gap = _exit_replay(monkeypatch, "SCALE_OUT", gap=True)
    assert gap.result.trades[0].r_multiple == 2.5
    assert gap.result.trades[0].exit_fills[-1]["at_open"]
