"""
tests/test_backtest_coverage.py
================================
Targeted unit tests for core/backtest.py to ensure thorough coverage:
- TransactionCostModel.from_config
- SHORT execution and boundary branches in _realised_r
- _decompose_trade_pnl_daily edge cases (single-day, short, fallback without shares)
- _daily_portfolio_sharpe edge cases (empty trades, zero std, linear fallback)
- WalkForwardResult.to_dataframe and to_csv
- OverallStats formatting and edge cases
- compute_trade_management_wrapper
- walk_forward auto-adjustment for step_days < test_days and simulated trade execution
"""

import math
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd

from core.backtest import (
    ZERO_COST_MODEL,
    OverallStats,
    TradeRecord,
    TransactionCostModel,
    WalkForwardResult,
    _compute_trade_daily_pnl,
    _daily_portfolio_sharpe,
    _realised_r,
    compute_trade_management_wrapper,
    walk_forward,
)
from core.config import SystemConfig


def test_transaction_cost_model_from_config() -> None:
    cfg = MagicMock()
    cfg.SLIPPAGE_BPS = 10
    cfg.COMMISSION_INR = 25
    model = TransactionCostModel.from_config(cfg)
    assert math.isclose(model.slippage_pct, 0.0010)
    assert math.isclose(model.commission_inr, 25.0)


def test_realised_r_short_exits() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="B")

    # 1. SHORT gap stop: Open >= stop
    df_gap_stop = pd.DataFrame({
        "Open": [106.0, 107.0],
        "High": [108.0, 108.0],
        "Low": [105.0, 104.0],
        "Close": [107.0, 105.0],
    }, index=idx[:2])
    net_r, hit_t1, bars, exit_ts, gross_r, fric = _realised_r(
        direction="SHORT", entry=100.0, stop=105.0, t1=90.0,
        fwd_bars=df_gap_stop, time_stop=10, cost_model=ZERO_COST_MODEL
    )
    assert not hit_t1
    assert gross_r == -1.2  # (100 - 106) / 5 = -1.2

    # 2. SHORT normal stop: High >= stop
    df_stop = pd.DataFrame({
        "Open": [102.0, 103.0],
        "High": [106.0, 104.0],
        "Low": [101.0, 100.0],
        "Close": [103.0, 102.0],
    }, index=idx[:2])
    net_r, hit_t1, bars, exit_ts, gross_r, fric = _realised_r(
        direction="SHORT", entry=100.0, stop=105.0, t1=90.0,
        fwd_bars=df_stop, time_stop=10, cost_model=ZERO_COST_MODEL
    )
    assert not hit_t1
    assert gross_r == -1.0

    # 3. SHORT gap target: Open <= t1
    df_gap_t1 = pd.DataFrame({
        "Open": [88.0, 87.0],
        "High": [89.0, 88.0],
        "Low": [85.0, 84.0],
        "Close": [87.0, 86.0],
    }, index=idx[:2])
    net_r, hit_t1, bars, exit_ts, gross_r, fric = _realised_r(
        direction="SHORT", entry=100.0, stop=105.0, t1=90.0,
        fwd_bars=df_gap_t1, time_stop=10, cost_model=ZERO_COST_MODEL
    )
    assert hit_t1
    assert gross_r == 2.4  # (100 - 88) / 5 = 2.4

    # 4. SHORT normal target: Low <= t1
    df_t1 = pd.DataFrame({
        "Open": [98.0, 95.0],
        "High": [99.0, 96.0],
        "Low": [97.0, 89.0],
        "Close": [97.5, 91.0],
    }, index=idx[:2])
    net_r, hit_t1, bars, exit_ts, gross_r, fric = _realised_r(
        direction="SHORT", entry=100.0, stop=105.0, t1=90.0,
        fwd_bars=df_t1, time_stop=10, cost_model=ZERO_COST_MODEL
    )
    assert hit_t1
    assert gross_r == 2.0  # (100 - 90) / 5 = 2.0
    assert bars == 2

    # 5. LONG gap stop: Open <= stop
    df_long_gap_stop = pd.DataFrame({
        "Open": [93.0], "High": [94.0], "Low": [92.0], "Close": [93.5],
    }, index=idx[:1])
    net_r, hit_t1, bars, exit_ts, gross_r, fric = _realised_r(
        direction="LONG", entry=100.0, stop=95.0, t1=110.0,
        fwd_bars=df_long_gap_stop, time_stop=10, cost_model=ZERO_COST_MODEL
    )
    assert not hit_t1
    assert gross_r == -1.4  # (93 - 100) / 5 = -1.4

    # 6. LONG gap target: Open >= t1
    df_long_gap_t1 = pd.DataFrame({
        "Open": [112.0], "High": [114.0], "Low": [111.0], "Close": [113.0],
    }, index=idx[:1])
    net_r, hit_t1, bars, exit_ts, gross_r, fric = _realised_r(
        direction="LONG", entry=100.0, stop=95.0, t1=110.0,
        fwd_bars=df_long_gap_t1, time_stop=10, cost_model=ZERO_COST_MODEL
    )
    assert hit_t1
    assert gross_r == 2.4  # (112 - 100) / 5 = 2.4

    # 7. sl_dist <= 0 guard
    res_zero = _realised_r(
        direction="LONG", entry=100.0, stop=100.0, t1=110.0,
        fwd_bars=df_t1, time_stop=10, cost_model=ZERO_COST_MODEL
    )
    assert res_zero == (0.0, False, 0, None, 0.0, 0.0)


def test_decompose_trade_pnl_daily_variants() -> None:
    idx = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame({
        "Open": [100.0, 102.0, 104.0],
        "High": [103.0, 105.0, 106.0],
        "Low": [99.0, 101.0, 103.0],
        "Close": [102.0, 104.0, 105.0],
    }, index=idx)

    # Empty cases
    assert _compute_trade_daily_pnl(df, "LONG", 100.0, 10, 0, 50.0) == {}
    assert _compute_trade_daily_pnl(pd.DataFrame(), "LONG", 100.0, 10, 2, 50.0) == {}

    # Single day
    pnl_1d = _compute_trade_daily_pnl(df, "LONG", 100.0, 10, 1, 50.0)
    assert len(pnl_1d) == 1
    assert pnl_1d[idx[0].date()] == 50.0

    # Short direction with shares
    pnl_short = _compute_trade_daily_pnl(
        fwd_bars=df, direction="SHORT", entry_price=105.0,
        shares=10, bars_held=3, net_pnl=-50.0,
    )
    assert len(pnl_short) == 3
    assert math.isclose(sum(pnl_short.values()), -50.0, abs_tol=0.01)

    # No shares, risk_inr fallback
    pnl_rsk = _compute_trade_daily_pnl(
        fwd_bars=df, direction="LONG", entry_price=100.0,
        shares=0, bars_held=3, net_pnl=100.0, sl_dist=5.0, risk_inr=1000.0,
    )
    assert len(pnl_rsk) == 3
    assert math.isclose(sum(pnl_rsk.values()), 100.0, abs_tol=0.01)

    # No shares, no risk_inr fallback (equal split)
    pnl_eq = _compute_trade_daily_pnl(
        fwd_bars=df, direction="LONG", entry_price=100.0,
        shares=0, bars_held=3, net_pnl=90.0, sl_dist=0.0, risk_inr=0.0,
    )
    assert len(pnl_eq) == 3
    assert math.isclose(sum(pnl_eq.values()), 90.0, abs_tol=0.01)


def test_daily_portfolio_sharpe_edge_cases() -> None:
    # 1. Empty trades
    assert _daily_portfolio_sharpe([]) == 0.0

    # 2. No valid exit dates
    t_no_exit = TradeRecord(
        fold=0, ticker="TEST", direction="LONG",
        entry_date=pd.Timestamp("2024-01-01"), exit_date=None,
        entry=100.0, stop=95.0, t1=110.0, composite=0.8, prob_win=0.6,
        r_multiple=1.0, hit_t1=True, bars_held=2,
    )
    assert _daily_portfolio_sharpe([t_no_exit]) == 0.0

    # 3. Single date (b_days < 2)
    t_1d = TradeRecord(
        fold=0, ticker="TEST", direction="LONG",
        entry_date=pd.Timestamp("2024-01-01"), exit_date=pd.Timestamp("2024-01-01"),
        entry=100.0, stop=95.0, t1=110.0, composite=0.8, prob_win=0.6,
        r_multiple=1.0, hit_t1=True, bars_held=1,
    )
    assert _daily_portfolio_sharpe([t_1d]) == 0.0

    # 4. Fallback when trade has no daily_pnl (synthetic/legacy linear accrual)
    t_legacy = TradeRecord(
        fold=0, ticker="TEST", direction="LONG",
        entry_date=pd.Timestamp("2024-01-01"), exit_date=pd.Timestamp("2024-01-05"),
        entry=100.0, stop=95.0, t1=110.0, composite=0.8, prob_win=0.6,
        r_multiple=1.0, hit_t1=True, bars_held=5, risk_inr=5000.0,
        daily_pnl={},
    )
    sharpe = _daily_portfolio_sharpe([t_legacy], capital=100_000.0)
    assert isinstance(sharpe, float)

    # 5. Zero variance returns -> 0.0
    t_flat = TradeRecord(
        fold=0, ticker="TEST", direction="LONG",
        entry_date=pd.Timestamp("2024-01-01"), exit_date=pd.Timestamp("2024-01-03"),
        entry=100.0, stop=95.0, t1=110.0, composite=0.8, prob_win=0.6,
        r_multiple=0.0, hit_t1=False, bars_held=3, risk_inr=5000.0,
        daily_pnl={pd.Timestamp("2024-01-01"): 0.0, pd.Timestamp("2024-01-02"): 0.0},
    )
    assert _daily_portfolio_sharpe([t_flat]) == 0.0


def test_walk_forward_result_to_csv_and_dataframe(tmp_path: Path) -> None:
    t = TradeRecord(
        fold=0, ticker="INFY", direction="LONG",
        entry_date=pd.Timestamp("2024-01-01"), exit_date=pd.Timestamp("2024-01-05"),
        entry=100.0, stop=95.0, t1=110.0, composite=0.8, prob_win=0.6,
        r_multiple=1.0, hit_t1=True, bars_held=5,
    )
    overall = OverallStats(
        n_folds=1, n_trades=1, hit_rate=1.0, mean_r=1.0,
        sharpe=1.5, max_dd=0.0, total_r=1.0, profit_factor=2.0,
        expectancy_r=0.6,
    )
    wfr = WalkForwardResult(trades=[t], fold_stats=[], overall=overall)
    df = wfr.to_dataframe()
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "INFY"

    out_file = tmp_path / "wf_out.csv"
    wfr.to_csv(str(out_file))
    assert out_file.exists()
    assert "INFY" in out_file.read_text()


def test_compute_trade_management_wrapper() -> None:
    row = pd.Series({"ATR_Pctile": 75.0})
    trail, time_stop = compute_trade_management_wrapper("LONG", 100.0, 2.0, row)
    assert trail > 0.0
    assert time_stop > 0


def test_walk_forward_simulates_trades_with_adjusted_step_days(monkeypatch) -> None:
    cfg = SystemConfig()
    bench = cfg.BENCHMARK
    n = 200
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    c = pd.Series([100.0 + i * 0.5 for i in range(n)], index=idx, dtype=float)
    df_mock = pd.DataFrame({
        "Open": c, "High": c + 2.0, "Low": c - 1.0, "Close": c + 0.5,
        "Volume": pd.Series(500_000, index=idx),
    })
    raw_data = {bench: df_mock, "RELIANCE.NS": df_mock}

    # Mock score_universe in core.services to return a high conviction candidate
    from core.scorer import TickerResult
    mock_res = MagicMock(spec=TickerResult)
    mock_res.ticker = "RELIANCE.NS"
    mock_res.sector = "ENERGY"
    mock_res.direction = "LONG"
    mock_res.composite = 0.75
    mock_res.prob_win = 0.65
    mock_res.is_watchlist = False
    mock_res.trade_horizon = "SWING"
    mock_res.sharpe_rank = 1.0
    mock_res.reasons = []
    mock_res.entry = 150.0
    mock_res.stop = 145.0
    mock_res.t1 = 160.0
    mock_res.shares = 100
    mock_res.risk_inr = 5000.0

    monkeypatch.setattr("core.services.score_universe", lambda **kwargs: [mock_res])

    # Pass step_days=10 < test_days=20 to trigger auto-adjustment warning branch
    result = walk_forward(
        raw_data,
        cfg,
        train_days=120,
        test_days=20,
        step_days=10,  # < test_days -> triggers auto-adjust to 20
        min_prob=0.52,
    )
    assert len(result.trades) > 0
    assert result.trades[0].ticker == "RELIANCE.NS"
    assert result.trades[0].entry > 0
    assert result.trades[0].r_multiple != 0.0
    assert result.overall.n_trades > 0
