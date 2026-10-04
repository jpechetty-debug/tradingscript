from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from core.backtest import TradeRecord
from core.config import CONFIG
from core.swing_analytics import monthly_mean_r_interval, portfolio_statistics, trade_statistics
from core.swing import detect_swing_setup
from scripts.research_swing import sensitivity_grid, validation_windows
from tests.test_swing_setups import breakout


def curve(exposure=.2, alpha=0.):
    market = np.array([-.01, .02, .03, -.005])
    daily = exposure * market + alpha
    return pd.DataFrame(dict(daily_return=daily, benchmark_return=market,
                             exposure_matched_benchmark_return=exposure * market,
                             exposure_open=exposure, exposure_close=exposure))


def test_exposure_and_alpha_have_known_reference_values():
    stats = portfolio_statistics(curve())
    assert stats["beta"] == pytest.approx(.2)
    assert stats["annualized_alpha"] == pytest.approx(0., abs=1e-12)
    assert stats["exposure_matched_active_return"] == pytest.approx(0.)
    assert stats["cagr"] == pytest.approx(np.prod(1 + .2 * np.array([-.01, .02, .03, -.005])) ** 63 - 1)
    assert stats["exposure_adjusted_cagr"] == pytest.approx(stats["cagr"] / .2)
    assert portfolio_statistics(curve(alpha=.001))["annualized_alpha"] == pytest.approx(.252)


def test_zero_exposure_and_constant_returns_are_unavailable_not_infinity():
    data = curve(exposure=0.)
    data["benchmark_return"] = 0.
    stats = portfolio_statistics(data)
    assert stats["exposure_adjusted_cagr"] is None and stats["beta"] is None
    assert stats["daily_nav_sharpe"] is None and stats["information_ratio"] is None
    data.loc[0, "daily_return"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        portfolio_statistics(data)


def test_distribution_and_trade_excess_returns():
    trades = [TradeRecord(0, "A", "LONG", pd.Timestamp(f"2024-0{1 + i // 2}-02"), None,
                          100., 98., 104., .8, .7, value, value > 0, i + 1,
                          risk_inr=100., net_return=value * .02, benchmark_return=.01,
                          excess_return=value * .02 - .01, exit_reason="STOP" if value < 0 else "TIME")
              for i, value in enumerate([-1., -1., .5, 2.])]
    stats = trade_statistics(trades)
    assert stats["mean_r"] == .125 and stats["win_rate"] == .5
    assert stats["average_hold"] == 2.5 and stats["stop_out_frequency"] == .5
    assert stats["percentiles"]["50"] == -.25
    assert stats["largest_winner"] == 2. and stats["largest_loser"] == -1.
    assert stats["mean_excess_return"] == pytest.approx(-.0075)
    assert stats["fraction_outperforming_benchmark"] == .25
    assert monthly_mean_r_interval(trades) == monthly_mean_r_interval(trades)
    assert trade_statistics([])["mean_r"] is None and monthly_mean_r_interval([]) is None


def test_quarters_use_only_prior_training_and_do_not_overlap():
    dates = pd.bdate_range("2022-01-03", "2026-10-01")
    windows = validation_windows(dates, pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01"))
    assert len(windows) == 12 and windows[-1]["partial_quarter"]
    for i, window in enumerate(windows):
        assert window["train_end"] < window["validation_start"]
        assert window["training_sessions"] >= 252
        if i:
            assert windows[i - 1]["validation_end"] < window["validation_start"]
    assert validation_windows(dates[:50], dates[0], dates[30]) == []


@pytest.mark.parametrize("width", [.5, 1., 1.5])
def test_atr_sensitivity_changes_stop_without_changing_default(width):
    df, bench = breakout()
    plan = detect_swing_setup(df, bench, replace(CONFIG, SWING_STOP_MODE="ATR", SWING_STOP_ATR_MULT=width,
                                                SWING_RESEARCH_TAG=f"_RESEARCH_atr_{width}"))
    assert plan is not None
    assert plan.stop == pytest.approx(plan.entry - width)
    assert "RESEARCH" in plan.strategy_id
    assert CONFIG.SWING_STOP_MODE == "STRUCTURE" and CONFIG.SWING_BREAKOUT_LOOKBACK == 20
    grid = sensitivity_grid(CONFIG)
    assert len(grid) == 11 and grid[0][0] == "fixed_default"
    assert all(config.SWING_RESEARCH_TAG for _, config in grid[1:])
