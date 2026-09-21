"""
tests/test_sharpe_budget_cache.py
=================================
Validates:
1. Daily portfolio Sharpe accrues mark-to-market returns across holding days,
   preventing degenerate constant Sharpe on clustered exits.
2. Coherent capital (Rs 10L) and risk (Rs 30k portfolio / Rs 5k per-trade) budgets
   with responsive Kelly sizing and reachable 1.5x cap.
3. Thread-safe ScanCache under concurrent worker access and factor_volume ticker wiring.
"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import pytest

from core.backtest import (
    TradeRecord,
    _compute_trade_daily_pnl,
    _daily_portfolio_sharpe,
    _fold_stats,
    _overall_stats,
)
from core.cache import ScanCache, SCAN_CACHE
from core.config import SystemConfig
from core.factors import factor_volume
from core.portfolio import calculate_kelly_size


def _make_sample_df(n_bars: int = 150, base_price: float = 1000.0) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=n_bars)
    np.random.seed(42)
    closes = [base_price]
    for _ in range(n_bars - 1):
        ret = np.random.normal(0.0005, 0.015)
        closes.append(closes[-1] * (1.0 + ret))
    closes_arr = np.array(closes)
    highs = closes_arr * (1.0 + np.random.uniform(0.002, 0.015, n_bars))
    lows = closes_arr * (1.0 - np.random.uniform(0.002, 0.015, n_bars))
    opens = (highs + lows) / 2.0
    vols = np.random.randint(500_000, 2_000_000, n_bars)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes_arr,
            "Volume": vols,
            "ATR": closes_arr * 0.015,
            "ATR_50_mean": closes_arr * 0.015,
            "Vol_Avg_20": vols.astype(float),
            "Turnover_Avg_20": (closes_arr * vols),
            "Super_Up": True,
            "EMA_20": closes_arr * 0.98,
            "EMA_50": closes_arr * 0.95,
            "EMA_200": closes_arr * 0.90,
            "ADX": 26.0,
            "RSI": 55.0,
            "MACD_Hist": 0.5,
            "MACD_Acc": True,
            "Stoch_K": 50.0,
            "Streak": 2,
            "Up_Day": True,
        },
        index=dates,
    )
    return df


class TestDailyPortfolioSharpe:
    def test_clustered_exits_produce_different_sharpe_based_on_returns(self):
        """
        When all trades exit on the same day, mark-to-market accrual must
        distinguish high-return vs low-return books instead of collapsing
        algebraically to a constant Sharpe (which happened under point-mass exits).
        """
        df = _make_sample_df(10)
        entry_price = float(df.iloc[0]["Open"])
        entry_date = df.index[0]
        exit_date = df.index[-1]

        # Book A: high-return trades (+1.00 R = Rs 5,000) with MTM accrual
        trades_high = []
        for i in range(10):
            mtm_h = _compute_trade_daily_pnl(
                fwd_bars=df,
                direction="LONG",
                entry_price=entry_price,
                shares=50,
                bars_held=10,
                net_pnl=5_000.0,
            )
            trades_high.append(
                TradeRecord(
                    fold=0,
                    ticker=f"STK_{i}",
                    direction="LONG",
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry=entry_price,
                    stop=entry_price * 0.97,
                    t1=entry_price * 1.10,
                    composite=0.7,
                    prob_win=0.6,
                    r_multiple=1.00,
                    hit_t1=True,
                    bars_held=10,
                    risk_inr=5_000.0,
                    daily_pnl=mtm_h,
                )
            )

        # Book B: low-return trades (+0.07 R = Rs 350) with MTM accrual
        trades_low = []
        for i in range(10):
            mtm_l = _compute_trade_daily_pnl(
                fwd_bars=df,
                direction="LONG",
                entry_price=entry_price,
                shares=50,
                bars_held=10,
                net_pnl=350.0,
            )
            trades_low.append(
                TradeRecord(
                    fold=0,
                    ticker=f"STK_{i}",
                    direction="LONG",
                    entry_date=entry_date,
                    exit_date=exit_date,
                    entry=entry_price,
                    stop=entry_price * 0.97,
                    t1=entry_price * 1.10,
                    composite=0.7,
                    prob_win=0.6,
                    r_multiple=0.07,
                    hit_t1=False,
                    bars_held=10,
                    risk_inr=5_000.0,
                    daily_pnl=mtm_l,
                )
            )

        sharpe_high = _daily_portfolio_sharpe(trades_high, entry_date, exit_date)
        sharpe_low = _daily_portfolio_sharpe(trades_low, entry_date, exit_date)

        # Under old spike train, both were identically 3.31.
        # Now mark-to-market Sharpe captures edge: high-return book has substantially higher Sharpe
        assert sharpe_high > sharpe_low
        assert sharpe_high > 5.0
        assert sharpe_low < 3.0

    def test_mark_to_market_daily_pnl_exact_sum(self):
        """Daily mark-to-market values must sum exactly to net_pnl to the penny."""
        df = _make_sample_df(10)
        entry_price = float(df.iloc[0]["Open"])
        shares = 100
        bars_held = 6
        net_pnl = 4523.50  # Arbitrary net PnL after costs

        mtm = _compute_trade_daily_pnl(
            fwd_bars=df,
            direction="LONG",
            entry_price=entry_price,
            shares=shares,
            bars_held=bars_held,
            net_pnl=net_pnl,
        )

        assert len(mtm) == bars_held
        total_mtm = sum(mtm.values())
        assert total_mtm == pytest.approx(net_pnl, abs=0.01)

    def test_unified_window_fold_and_overall_stats(self):
        """_fold_stats and _overall_stats must use active trading window for Sharpe."""
        df = _make_sample_df(30)
        d0 = df.index[5]
        d1 = df.index[15]
        calendar_start = df.index[0]
        calendar_end = df.index[-1]

        trade = TradeRecord(
            fold=0,
            ticker="RELIANCE.NS",
            direction="LONG",
            entry_date=d0,
            exit_date=d1,
            entry=1000.0,
            stop=970.0,
            t1=1100.0,
            composite=0.7,
            prob_win=0.6,
            r_multiple=1.5,
            hit_t1=True,
            bars_held=8,
            risk_inr=5_000.0,
        )

        f_stats = _fold_stats(0, [trade], (calendar_start, calendar_end))
        o_stats = _overall_stats([trade], [f_stats])

        # Both compute Sharpe across the active trade span
        assert f_stats.sharpe == pytest.approx(o_stats.sharpe, rel=1e-3)
        assert f_stats.sharpe > 0.0


class TestCapitalAndRiskBudgets:
    def test_config_budget_coherence(self):
        cfg = SystemConfig()
        assert cfg.CAPITAL_INR == 1_000_000.0
        assert cfg.MAX_PORTFOLIO_RISK_INR == 30_000.0
        assert cfg.RISK_PER_TRADE_INR == 5_000.0
        assert cfg.KELLY_FRACTION == 4.5

    def test_kelly_sizing_responsive_and_capped(self):
        # Under standard stock fat-tail conditions (excess kurtosis ~ 4.0)
        df = _make_sample_df(40)  # < 60 bars uses KELLY_KURTOSIS_FALLBACK = 4.0
        cfg = SystemConfig()

        # Gate level (p=0.52) -> ~Rs. 3,180
        _, risk_gate, _, _ = calculate_kelly_size(
            entry=1000.0, stop=970.0, prob_win=0.52, rr=2.53, daily_df=df, config=cfg
        )
        assert 2500.0 <= risk_gate <= 4000.0

        # Median level (p=0.55) -> ~Rs. 3,570
        _, risk_med, _, _ = calculate_kelly_size(
            entry=1000.0, stop=970.0, prob_win=0.55, rr=2.53, daily_df=df, config=cfg
        )
        assert 3000.0 <= risk_med <= 4500.0

        # High conviction (p=0.85) -> reaches and binds at KELLY_MAX_MULT = 1.5 (Rs 7,500)
        _, risk_high, _, _ = calculate_kelly_size(
            entry=1000.0, stop=970.0, prob_win=0.85, rr=2.53, daily_df=df, config=cfg
        )
        assert risk_high == pytest.approx(7_500.0, abs=50.0)

        # Floor test: even on marginal edge, floor is 0.25 * 5000 = Rs 1,250
        _, risk_floor, _, _ = calculate_kelly_size(
            entry=1000.0, stop=970.0, prob_win=0.48, rr=2.53, daily_df=df, config=cfg
        )
        assert risk_floor >= 1250.0 - 30.0  # quantization margin


class TestScanCacheThreadSafetyAndWiring:
    def test_scan_cache_concurrent_access_no_race(self):
        cache = ScanCache(max_vprofile_entries=20)
        df = _make_sample_df(40)

        def worker(idx: int):
            # Interleave reads, writes, and evictions
            t_name = f"TICKER_{idx % 50}"
            cache.volume_profile(df, ticker=t_name, lookback=20, bins=50)
            if idx % 5 == 0:
                cache.stats()

        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(worker, i) for i in range(100)]
            for fut in futures:
                fut.result()

        stats = cache.stats()
        assert stats["vprofile_size"] <= 20
        assert stats["hits"] + stats["misses"] == 100

    def test_factor_volume_uses_scan_cache_with_ticker(self):
        SCAN_CACHE.clear()
        df = _make_sample_df(50)
        row = df.iloc[-1]

        # First call with ticker should record a miss and cache the result
        v1 = factor_volume(
            row=row,
            daily_df=df,
            direction="LONG",
            close=float(row["Close"]),
            ticker="RELIANCE.NS",
        )
        assert SCAN_CACHE.stats()["misses"] >= 1

        # Second call with same ticker and data should record a cache hit
        v2 = factor_volume(
            row=row,
            daily_df=df,
            direction="LONG",
            close=float(row["Close"]),
            ticker="RELIANCE.NS",
        )
        assert SCAN_CACHE.stats()["hits"] >= 1
        assert v1 == pytest.approx(v2)
