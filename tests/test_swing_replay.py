from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

import core.swing_replay as replay
from core.backtest import ZERO_COST_MODEL, walk_forward
from core.config import CONFIG


@pytest.mark.parametrize("gap", [103., 98.])
def test_daily_replay_rejects_gaps_respects_cash_and_uses_only_past_signals(monkeypatch, gap):
    dates = pd.bdate_range("2024-01-01", periods=4)
    frame = pd.DataFrame({"Open": [100., 100., gap, 100.], "High": [101., 104., 104., 104.],
                          "Low": [99., 99., 98., 99.], "Close": [100., 103., 100., 103.],
                          "ATR": [1.] * 4}, index=dates)
    raw = {CONFIG.BENCHMARK: frame.copy(), "TEST.NS": frame}
    monkeypatch.setattr(replay, "add_indicators", lambda f, c: f)
    monkeypatch.setattr(replay, "compute_sector_rs", lambda *a: {})
    monkeypatch.setattr(replay, "compute_breadth", lambda *a: .7)
    monkeypatch.setattr(replay, "classify_regime", lambda *a, **k: SimpleNamespace(label="TREND_UP"))
    calls = []

    def score(**kw):
        last = kw["bench"].index[-1]
        assert all(f.index[-1] <= last for f in kw["processed"].values())
        calls.append(last)
        return [SimpleNamespace(ticker="TEST", sector="IT", direction="LONG", strategy_id="SWING_BREAKOUT_V1",
                                signal_time=last.isoformat(), entry=100., entry_min=99., entry_max=101.,
                                stop=98., t1=104., t2=106., time_stop_bars=3,
                                expectancy_r=.5, composite=.7, prob_win=.6, is_watchlist=False)] * 2

    monkeypatch.setattr(replay, "score_universe", score)
    config = replace(CONFIG, CAPITAL_INR=10000., PORTFOLIO_SIZE=1, SWING_SETUP_ENABLED=True)
    result = replay.replay_swing(raw, config, dates[0], dates[-1], ZERO_COST_MODEL)
    assert calls == list(dates)
    assert len(result.result.trades) == 2
    assert result.result.trades[0].entry_date == dates[1]
    assert result.metrics["minimum_cash"] >= 0
    assert result.metrics["rejected"]["gap_or_size"] >= 1
    assert all(t.entry == 100. and t.stop == 98. and t.t1 == 104. for t in result.result.trades)
    assert result.metrics["total_return"] > 0


@pytest.mark.parametrize("limit", ["positions", "sector", "correlation", "risk", "cash"])
def test_replay_limits_apply_to_held_book(monkeypatch, limit):
    dates = pd.bdate_range("2024-01-01", periods=4)
    frame = pd.DataFrame({"Open": 100., "High": 103., "Low": 99.,
                          "Close": [100., 100.2, 100.4, 100.6], "ATR": 1.}, index=dates)
    raw = {CONFIG.BENCHMARK: frame.copy(), "A.NS": frame.copy(), "B.NS": frame.copy()}
    monkeypatch.setattr(replay, "add_indicators", lambda f, c: f)
    monkeypatch.setattr(replay, "compute_sector_rs", lambda *a: {})
    monkeypatch.setattr(replay, "compute_breadth", lambda *a: .7)
    monkeypatch.setattr(replay, "classify_regime", lambda *a, **k: SimpleNamespace(label="TREND_UP"))

    def score(**kw):
        return [SimpleNamespace(ticker=ticker, sector="IT", direction="LONG", strategy_id="SWING_BREAKOUT_V1",
                                signal_time=kw["bench"].index[-1].isoformat(), entry=100., entry_min=99., entry_max=101.,
                                stop=98., t1=104., t2=106., time_stop_bars=10,
                                expectancy_r=.5, composite=.7, prob_win=.6, is_watchlist=False)
                for ticker in ("A", "B")]

    monkeypatch.setattr(replay, "score_universe", score)
    options = {"positions": {"PORTFOLIO_SIZE": 1}, "sector": {"MAX_SECTOR_PICKS": 1},
               "correlation": {"MAX_CORR": .5}, "risk": {"MAX_PORTFOLIO_RISK_INR": 35.},
               "cash": {"SWING_RISK_FRACTION": .5, "SWING_MAX_EXPOSURE_FRACTION": 1.}}
    config = replace(CONFIG, **({"CAPITAL_INR": 10000., "SWING_SETUP_ENABLED": True, "MAX_CORR": 1.} | options[limit]))
    result = replay.replay_swing(raw, config, dates[0], dates[-1], ZERO_COST_MODEL)
    trades = result.result.trades
    assert result.metrics["minimum_cash"] >= 0
    if limit == "risk":
        assert len(trades) == 2 and sum(t.risk_inr for t in trades) <= 35.
        assert trades[1].shares < trades[0].shares
    else:
        assert len(trades) == 1 and result.metrics["peak_positions"] == 1


@pytest.mark.parametrize("opening", [100., 103., 98.])
def test_walk_forward_preserves_swing_plan_at_actual_fill(monkeypatch, opening):
    dates = pd.bdate_range("2023-01-02", periods=200)
    frame = pd.DataFrame({"Open": opening, "High": max(103., opening + 1.),
                          "Low": min(99., opening - 1.), "Close": [100. + i * .01 for i in range(200)],
                          "Volume": 10_000_000.}, index=dates)
    candidate = SimpleNamespace(ticker="RELIANCE", sector="ENERGY", direction="LONG",
                                strategy_id="SWING_BREAKOUT_V1", signal_time=dates[119].isoformat(),
                                entry=100., entry_min=99., entry_max=101., stop=98., t1=104., t2=106.,
                                time_stop_bars=3, expectancy_r=.5, composite=.7, prob_win=.6,
                                is_watchlist=False, is_held=False, trade_horizon="SWING",
                                sharpe_rank=.5, atr_pctile=50., shares=50, risk_inr=100., reasons=[])
    monkeypatch.setattr("core.services.score_universe", lambda **kw: [candidate])
    monkeypatch.setattr("core.backtest.add_indicators", lambda f, c: f.assign(ATR=1., ATR_Pctile=50.))
    result = walk_forward({CONFIG.BENCHMARK: frame.copy(), "RELIANCE.NS": frame},
                          replace(CONFIG, SWING_SETUP_ENABLED=True), cost_model=ZERO_COST_MODEL)
    if opening != 100.:
        assert not result.trades
    else:
        assert result.trades
        for trade in result.trades:
            assert trade.stop == 98. and trade.t1 == 104. and trade.bars_held == 3
            assert trade.shares > 0 and trade.risk_inr == trade.shares * 2.
            assert trade.strategy_id == candidate.strategy_id


def test_future_price_changes_cannot_change_validation_inputs(monkeypatch):
    import numpy as np
    dates = pd.bdate_range("2023-01-02", periods=300)
    close = 100. + np.arange(300) * .05 + np.sin(np.arange(300)) * .5
    frame = pd.DataFrame({"Open": close - .2, "High": close + .6, "Low": close - .6,
                          "Close": close, "Volume": 1e6 + np.sin(np.arange(300)) * 3e5}, index=dates)
    captured = []

    def score(**kw):
        captured.append({ticker: f.copy() for ticker, f in kw["processed"].items()})
        assert all(f.index[-1] <= pd.Timestamp(kw["now"]) for f in kw["processed"].values())
        return []

    monkeypatch.setattr(replay, "score_universe", score)
    config = replace(CONFIG, SWING_SETUP_ENABLED=True)
    raw = {CONFIG.BENCHMARK: frame.copy(), "TEST.NS": frame.copy()}
    first = replay.replay_swing(raw, config, dates[250], dates[260], warm_history=True)
    before = list(captured)
    captured.clear()
    changed = {t: f.copy() for t, f in raw.items()}
    for f in changed.values():
        f.loc[dates[261]:, ["Open", "High", "Low", "Close"]] *= .01
        f.loc[dates[261]:, "Volume"] *= 1000
    second = replay.replay_swing(changed, config, dates[250], dates[260], warm_history=True)
    assert len(before) == len(captured) > 0
    for old, new in zip(before, captured):
        for ticker in old:
            pd.testing.assert_frame_equal(old[ticker], new[ticker])
    pd.testing.assert_frame_equal(first.equity, second.equity)
