"""Descriptive swing evidence; exposure ratios are not investable returns."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from .backtest import TradeRecord
from .swing_replay import ReplayResult


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if np.isfinite(denominator) and abs(denominator) > 1e-12 else None


def trade_statistics(trades: list[TradeRecord]) -> dict[str, Any]:
    values = np.array([t.r_multiple for t in trades], dtype=float)
    returns = np.array([t.excess_return for t in trades if t.excess_return is not None], dtype=float)
    if not len(values):
        return {"trades": 0, "mean_r": None, "win_rate": None, "expectancy_r": None,
                "average_hold": None, "net_pnl": 0., "stop_out_frequency": None}
    winners, losers = values[values > 0], values[values < 0]
    mean = float(values.mean())
    return dict(trades=len(values), mean_r=mean, expectancy_r=mean, win_rate=float((values > 0).mean()),
                average_hold=float(np.mean([t.bars_held for t in trades])),
                net_pnl=float(sum(t.r_multiple * t.risk_inr for t in trades)),
                profit_factor=_ratio(float(winners.sum()), abs(float(losers.sum()))),
                stop_out_frequency=float(np.mean([t.exit_reason in {"STOP", "TRAIL_STOP"} for t in trades])),
                percentiles={str(p): float(np.percentile(values, p)) for p in (5, 25, 50, 75, 95)},
                skewness=float(skew(values, bias=False)) if len(values) >= 3 and values.std() > 0 else None,
                excess_kurtosis=float(kurtosis(values, fisher=True, bias=False)) if len(values) >= 4 and values.std() > 0 else None,
                largest_winner=float(values.max()), largest_loser=float(values.min()),
                benchmark_matched_trades=len(returns), mean_excess_return=float(returns.mean()) if len(returns) else None,
                median_excess_return=float(np.median(returns)) if len(returns) else None,
                fraction_outperforming_benchmark=float((returns > 0).mean()) if len(returns) else None)


def portfolio_statistics(curve: pd.DataFrame) -> dict[str, Any]:
    """252-session CAGR; zero cash yield; daily OLS intercept with no significance claim."""
    if curve.empty:
        raise ValueError("A nonempty daily equity curve is required")
    daily = curve.daily_return.to_numpy(dtype=float)
    market = curve.benchmark_return.to_numpy(dtype=float)
    matched = curve.exposure_matched_benchmark_return.to_numpy(dtype=float)
    if not np.isfinite(np.column_stack([daily, market, matched])).all():
        raise ValueError("Daily returns must be finite")
    total = float(np.prod(1 + daily) - 1)
    market_total = float(np.prod(1 + market) - 1)
    cagr = float((1 + total) ** (252 / len(daily)) - 1) if total > -1 else -1.
    market_cagr = float((1 + market_total) ** (252 / len(market)) - 1) if market_total > -1 else -1.
    average_exposure = float(curve.exposure_open.mean())
    active, matched_active = daily - market, daily - matched
    beta = _ratio(float(np.mean((daily - daily.mean()) * (market - market.mean()))), float(market.var()))
    alpha = float((daily.mean() - beta * market.mean()) * 252) if beta is not None else None
    nav = np.cumprod(1 + daily)
    peaks = np.maximum.accumulate(np.r_[1., nav])[1:]
    return dict(total_return=total, cagr=cagr, max_drawdown=float((nav / peaks - 1).min()),
                average_exposure=average_exposure, average_close_exposure=float(curve.exposure_close.mean()),
                invested_session_fraction=float((curve.exposure_open > 0).mean()),
                exposure_adjusted_cagr=_ratio(cagr, average_exposure), return_per_exposure=_ratio(total, average_exposure),
                benchmark_return=market_total, benchmark_cagr=market_cagr, beta=beta, annualized_alpha=alpha,
                daily_nav_sharpe=_ratio(float(daily.mean() * np.sqrt(252)), float(daily.std(ddof=1))) if len(daily) > 1 else None,
                information_ratio=_ratio(float(active.mean() * np.sqrt(252)), float(active.std(ddof=1))) if len(daily) > 1 else None,
                exposure_matched_benchmark_return=float(np.prod(1 + matched) - 1),
                exposure_matched_active_return=total - float(np.prod(1 + matched) - 1),
                exposure_matched_information_ratio=_ratio(float(matched_active.mean() * np.sqrt(252)), float(matched_active.std(ddof=1))) if len(daily) > 1 else None)


def monthly_mean_r_interval(trades: list[TradeRecord], draws: int = 2000, seed: int = 20261002) -> list[float] | None:
    """Resample entry-month clusters; adjacent windows are not claimed independent."""
    groups: dict[str, list[float]] = {}
    for trade in trades:
        groups.setdefault(str(pd.Timestamp(trade.entry_date).to_period("M")), []).append(trade.r_multiple)
    if len(groups) < 2:
        return None
    blocks = list(groups.values())
    totals = np.array([sum(block) for block in blocks])
    counts = np.array([len(block) for block in blocks])
    indexes = np.random.default_rng(seed).integers(0, len(blocks), size=(draws, len(blocks)))
    means = totals[indexes].sum(axis=1) / counts[indexes].sum(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def evidence(replay: ReplayResult) -> dict[str, Any]:
    trades = replay.result.trades
    record: dict[str, Any] = {**trade_statistics(trades), **replay.metrics, **portfolio_statistics(replay.equity)}
    record["monthly_mean_r_ci95"] = monthly_mean_r_interval(trades)
    fields = ("strategy_id", "entry_regime", "benchmark_trend", "volatility_regime", "breadth_regime", "exit_reason")
    record["attribution"] = {field: {
        label: trade_statistics([t for t in trades if getattr(t, field) == label])
        for label in sorted({str(getattr(t, field)) for t in trades})
    } for field in fields}
    # An unobserved regime is an absence of evidence, not a zero-return strategy.
    for field, labels in {"benchmark_trend": ("ABOVE_EMA200", "BELOW_EMA200"),
                          "volatility_regime": ("HIGH", "LOW"),
                          "breadth_regime": ("STRONG", "WEAK", "NEUTRAL")}.items():
        for label in labels:
            record["attribution"][field].setdefault(label, trade_statistics([]))
    return record


def research_parameters(config: Any) -> dict[str, Any]:
    keys = {"CAPITAL_INR", "RISK_PER_TRADE_INR", "MAX_PORTFOLIO_RISK_INR", "PORTFOLIO_SIZE", "MAX_SECTOR_PICKS",
            "MAX_CORR", "MIN_PROB_WIN", "MIN_EXPECTANCY_R", "SLIPPAGE_BPS", "COHORT_RANK_WEIGHT"}
    return {key: value for key, value in asdict(config).items() if key.startswith("SWING_") or key in keys}
