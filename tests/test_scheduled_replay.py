import hashlib
import json
import pickle
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from core.scheduled_replay import replay_scheduled_trades
from scripts import swing_research2 as research


DATES = pd.bdate_range("2025-01-01", periods=4)


def opportunity(ticker="A", entry=0, exit=2, *, phase="CLOSE", price=100., stop=90., target=100., priority=1.):
    return {"setup": "LP", "tk": ticker, "sig": DATES[entry] - pd.Timedelta(days=1),
            "ent": DATES[entry], "ext": DATES[exit], "e": price, "st": stop, "x": target,
            "prio": priority, "exit_phase": phase}


def replay(rows, *, history=None, **kwargs):
    frames = history or {row["tk"]: pd.DataFrame({"Close": 100.}, index=DATES) for row in rows}
    return replay_scheduled_trades(pd.DataFrame(rows), frames, DATES, ["LP"], cost=0., **kwargs)


@pytest.mark.parametrize("phase", ["INTRADAY", "CLOSE"])
def test_entry_day_exit_realizes_loss_and_releases_position(phase):
    result = replay([opportunity(exit=0, target=90., phase=phase)])
    assert result.equity.nav.tolist() == pytest.approx([.995] * 4)
    assert result.equity.positions.tolist() == [0] * 4
    assert result.fills.action.tolist() == ["BUY", "SELL"]
    assert result.fills.phase.tolist() == ["OPEN", phase]


def test_later_close_cannot_change_opening_quantity():
    rows = [opportunity(), opportunity("B", entry=1, exit=3)]
    frames = {ticker: pd.DataFrame({"Close": [100., 100., 200. if ticker == "B" else 100., 100.]}, index=DATES)
              for ticker in ("A", "B")}
    original = replay(rows, history=frames)
    frames["A"].loc[DATES[1], "Close"] = 1000.
    changed = replay(rows, history=frames)
    before = original.fills.loc[original.fills.action == "BUY"]
    after = changed.fills.loc[changed.fills.action == "BUY"]
    pd.testing.assert_frame_equal(before, after)
    assert before.loc[before.tk == "B", "sizing_nav"].iloc[0] == 1.
    assert original.equity.nav.iloc[2] == changed.equity.nav.iloc[2] == pytest.approx(1.05)


@pytest.mark.parametrize("phase,second_filled", [("OPEN", True), ("INTRADAY", False), ("CLOSE", False)])
def test_only_open_exit_proceeds_can_fund_opening_entries(phase, second_filled):
    result = replay([opportunity(exit=1, phase=phase, stop=99.),
                     opportunity("B", entry=1, exit=3, stop=99.)], cap=1., risk=1.)
    buys = result.fills.loc[result.fills.action == "BUY"]
    assert ("B" in buys.tk.tolist()) == second_filled
    assert result.equity.cash.min() >= 0
    if not second_filled:
        assert result.decisions.reason.tolist() == ["cash_or_nav"]


def test_closing_exit_does_not_release_position_slot_at_open():
    result = replay([opportunity(exit=1), opportunity("B", entry=1, exit=3)], slots=1)
    assert result.fills.loc[result.fills.action == "BUY", "tk"].tolist() == ["A"]
    assert result.decisions.reason.tolist() == ["position_cap"]


def test_same_day_stop_cannot_fund_another_order_in_the_opening_batch():
    result = replay([opportunity(exit=0, phase="INTRADAY", stop=99., priority=2.),
                     opportunity("B", exit=3, stop=99.)], cap=1., risk=1.)
    assert result.fills.action.tolist() == ["BUY", "SELL"]
    assert result.fills.tk.tolist() == ["A", "A"]
    assert result.decisions.reason.tolist() == ["cash_or_nav"]


def test_future_exit_price_and_date_cannot_change_earlier_book():
    original = replay([opportunity(exit=2, target=90.)])
    changed = replay([opportunity(exit=3, target=10000.)])
    pd.testing.assert_frame_equal(original.equity.iloc[:2], changed.equity.iloc[:2])
    pd.testing.assert_frame_equal(original.fills.iloc[:1], changed.fills.iloc[:1])


def test_fees_and_cash_match_realized_same_day_pnl():
    row = opportunity(exit=0, phase="INTRADAY", target=90.)
    history = {"A": pd.DataFrame({"Close": 100.}, index=DATES)}
    result = replay_scheduled_trades(pd.DataFrame([row]), history, DATES, ["LP"], cost=.004)
    quantity = .005 / 10
    fees = quantity * (100 + 90) * .002
    assert result.fills.fees.sum() == pytest.approx(fees)
    assert result.equity.nav.iloc[-1] == pytest.approx(1 - quantity * 10 - fees)
    assert result.equity.cash.iloc[-1] == result.equity.nav.iloc[-1]


def test_missing_close_carries_previous_observation_not_entry_or_future_close():
    frames = {"A": pd.DataFrame({"Close": [110., np.nan, 900., 100.]}, index=DATES)}
    result = replay([opportunity(exit=3)], history=frames)
    assert result.equity.nav.iloc[1] == result.equity.nav.iloc[0] == pytest.approx(1.005)


def test_empty_opportunities_preserve_initial_cash():
    empty = pd.DataFrame(columns=research.TRADE_COLUMNS)
    result = replay_scheduled_trades(empty, {}, DATES, ["LP"])
    assert result.equity.nav.tolist() == [1.] * len(DATES)
    assert result.fills.empty and result.decisions.empty


def test_entry_without_a_close_keeps_fill_mark_until_a_new_close():
    dates = pd.bdate_range("2024-12-31", periods=5)
    frame = pd.DataFrame({"Close": [80., np.nan, np.nan, 100., 100.]}, index=dates)
    result = replay([opportunity(exit=3)], history={"A": frame})
    assert result.equity.nav.iloc[:2].tolist() == [1., 1.]


@pytest.mark.parametrize("change", [{"exit_phase": "UNKNOWN"}, {"st": 100.}, {"x": np.nan},
                                     {"sig": DATES[0]}, {"ent": DATES[2], "ext": DATES[0]},
                                     {"ext": DATES[0], "exit_phase": "OPEN"}])
def test_invalid_schedules_are_rejected(change):
    row = opportunity()
    row.update(change)
    with pytest.raises(ValueError):
        replay([row])


def price_history():
    dates = pd.bdate_range("2024-01-01", periods=260)
    close = np.linspace(100., 150., len(dates))
    return pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1,
                         "Close": close, "Volume": 10_000_000.}, index=dates)


@pytest.mark.parametrize("phase", ["OPEN", "INTRADAY", "CLOSE"])
def test_signal_generator_annotates_gap_stop_touch_and_close_exits(monkeypatch, phase):
    frame = price_history()
    signal, entry = 230, 231
    values = pd.Series(50., index=frame.index)
    values.iloc[signal] = 5.
    monkeypatch.setattr(research, "rsi", lambda *args: values)
    opening = frame.Open.iloc[entry]
    if phase == "INTRADAY":
        frame.loc[frame.index[entry], "Low"] = opening - 7
    elif phase == "OPEN":
        frame.loc[frame.index[entry], ["Close", "Low"]] = [opening - 2, opening - 3]
        frame.loc[frame.index[entry + 1], ["Open", "Low"]] = [opening - 8, opening - 9]
    trades = research.generate_trades({"A": frame}, {}, price_history().Close)
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row.exit_phase == phase
    assert row.ext == frame.index[entry + (phase == "OPEN")]
    assert row.R < row.gross_R


def test_statistics_include_entry_day_loss_and_split_boundary_capital():
    nav = pd.Series([.99, .98], index=pd.to_datetime(["2024-12-31", "2025-01-01"]))
    stats = research.statistics(nav)
    assert stats["DEV"]["total_return"] == pytest.approx(-.01)
    assert stats["DEV"]["max_drawdown"] == pytest.approx(-.01)
    assert stats["HOLD"]["initial_nav"] == .99
    assert stats["HOLD"]["total_return"] == pytest.approx(.98 / .99 - 1)


def test_observed_stock_exit_is_processed_when_benchmark_session_is_missing():
    benchmark = pd.Series(100., index=DATES.delete(2))
    history = {"A": pd.DataFrame({"Close": 100.}, index=DATES)}
    trades = pd.DataFrame([opportunity(target=90.)])
    calendar = research.observation_calendar(history, benchmark, trades)
    assert DATES[2] in calendar
    result = replay_scheduled_trades(trades, history, calendar, ["LP"], cost=0.)
    assert result.equity.positions.iloc[-1] == 0
    assert result.equity.nav.iloc[-1] == pytest.approx(.995)


@pytest.mark.parametrize("with_earnings", [False, True])
def test_cli_exports_replay_ledgers_and_explicit_evidence_status(tmp_path, with_earnings):
    frame = price_history()
    prices = tmp_path / "prices.pkl"
    with prices.open("wb") as stream:
        pickle.dump({"^NSEI": frame, "A": frame.copy()}, stream)
    args = [str(prices)]
    if with_earnings:
        events = pd.DataFrame({"Surprise(%)": [20.]}, index=[frame.index[230].tz_localize("UTC")])
        earnings = tmp_path / "earnings.pkl"
        with earnings.open("wb") as stream:
            pickle.dump({"A": events}, stream)
        args.append(str(earnings))
    else:
        args.append("--lp-only")
    destination = tmp_path / "report"
    research.main(args + ["--out", str(destination)])
    report = json.loads((destination / "report.json").read_text())
    assert report["inputs"]["prices"]["sha256"] == hashlib.sha256(prices.read_bytes()).hexdigest()
    assert report["execution"]["sizing"] == "PREVIOUS_CLOSE_NAV"
    assert (destination / "LP" / "equity.csv").exists()
    assert (destination / "LP" / "fills.csv").exists()
    if with_earnings:
        assert "PEAD" in report["portfolios"]
        assert report["portfolios"]["PEAD"]["entries"] > 0
    else:
        assert report["earnings_status"] == "UNAVAILABLE_PEAD_NOT_EVALUATED"
        assert "PEAD" not in report["portfolios"]


def test_earnings_omission_requires_explicit_lp_only(tmp_path):
    with pytest.raises(SystemExit):
        research.main([str(tmp_path / "prices.pkl")])


def test_direct_script_entry_point_works_outside_the_repository(tmp_path):
    script = Path(research.__file__).resolve()
    result = subprocess.run([sys.executable, str(script), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--lp-only" in result.stdout
