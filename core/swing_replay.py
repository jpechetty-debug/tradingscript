"""Daily, cash-funded swing replay using the live scorer and next-session fills."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import pandas as pd

from .backtest import SWING_COST_MODEL, TradeRecord, TransactionCostModel, WalkForwardResult, _overall_stats
from .config import SystemConfig
from .factors import get_regime_factor_weights
from .indicators import add_indicators
from .portfolio import compute_targets
from .swing_exits import ExitPolicy
from .regime import RegimeTracker, classify_regime, compute_breadth, compute_sector_rs
from .services import score_universe
from .swing import SwingPlan, swing_fill_size
from .alpha_portfolio import PortfolioLimits, bounded_quantity


class ReplaySignal(Protocol):
    ticker: str
    sector: str
    strategy_id: str
    signal_time: str
    entry: float
    entry_min: float
    entry_max: float
    stop: float
    t1: float
    t2: float
    time_stop_bars: int
    composite: float
    prob_win: float
    expectancy_r: float
    research_context: dict[str, Any]


@dataclass
class ReplayResult:
    result: WalkForwardResult
    equity: pd.DataFrame
    metrics: dict[str, Any]
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)


def replay_swing(raw: dict[str, pd.DataFrame], config: SystemConfig,
                 start: pd.Timestamp, end: pd.Timestamp,
                 cost_model: TransactionCostModel = SWING_COST_MODEL, *,
                 prepared: dict[str, pd.DataFrame] | None = None,
                 warm_history: bool = False, exit_policy: ExitPolicy | None = None,
                 signal_provider: Callable[[dict[str, pd.DataFrame], pd.Timestamp, SystemConfig], Sequence[ReplaySignal]] | None = None,
                 entry_guard: Callable[[ReplaySignal, pd.Timestamp], str | None] | None = None,
                 portfolio_limits: PortfolioLimits | None = None) -> ReplayResult:
    """No live state or calibration; limits apply to the actual simulated held book."""
    processed = prepared if prepared is not None else {ticker: add_indicators(frame, config) for ticker, frame in raw.items()}
    dates = raw[config.BENCHMARK].loc[start:end].index
    if dates.empty or config.CAPITAL_INR <= 0:
        raise ValueError("Replay requires a nonempty date range and positive starting capital")
    cash = initial = float(config.CAPITAL_INR)
    book: dict[str, dict[str, Any]] = {}
    pending: list[ReplaySignal] = []
    trades: list[TradeRecord] = []
    equity: list[dict[str, Any]] = []
    tracker = RegimeTracker()
    rejected = {"gap_or_size": 0, "portfolio_limit": 0, "missing_bar": 0}
    correlation = pd.DataFrame()
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    context: dict[str, dict[str, Any]] = {}
    policy = exit_policy or ExitPolicy()

    def decision(candidate: ReplaySignal, date: pd.Timestamp, reason: str, shares: int = 0) -> None:
        counts[reason] += 1
        decisions.append(dict(date=date, signal_time=candidate.signal_time, ticker=candidate.ticker,
                              strategy_id=candidate.strategy_id, reason=reason, shares=shares))

    def exit_position(ticker: str, price: float, date: pd.Timestamp, reason: str, at_open: bool = False,
                      quantity: int | None = None) -> None:
        nonlocal cash
        pos = book[ticker]
        quantity = pos["shares"] if quantity is None else quantity
        if not 0 < quantity <= pos["shares"]:
            raise ValueError("Invalid simulated exit quantity")
        exit_cost = price * quantity * cost_model.exit_cost_pct() + cost_model.commission_inr
        cash += price * quantity - exit_cost
        pos["realized_gross"] += (price - pos["entry"]) * quantity
        pos["exit_costs"] += exit_cost
        pos["exit_value"] += price * quantity
        pos["shares"] -= quantity
        pos["risk"] = pos["initial_risk"] * pos["shares"] / pos["initial_shares"]
        candidate = pos["candidate"]
        exit_benchmark = float(raw[config.BENCHMARK].loc[date, "Open" if at_open else "Close"])
        pos["benchmark_weighted_return"] += (exit_benchmark / pos["benchmark_entry"] - 1) * quantity
        pos["exit_fills"].append(dict(date=date.isoformat(), price=price, shares=quantity, reason=reason,
                                     at_open=at_open, modeled_cost=exit_cost))
        if pos["shares"]:
            counts["partial_exits"] += 1
            return
        book.pop(ticker)
        gross = pos["realized_gross"]
        costs = pos["entry_cost"] + pos["exit_costs"]
        risk = pos["initial_risk"]
        initial_quantity = pos["initial_shares"]
        benchmark_return = pos["benchmark_weighted_return"] / initial_quantity
        net_return = (gross - costs) / (pos["entry"] * initial_quantity)
        trades.append(TradeRecord(
            fold=0, ticker=ticker, direction="LONG", entry_date=pos["date"], exit_date=date,
            entry=pos["entry"], stop=pos["initial_stop"], t1=pos["target"], composite=candidate.composite,
            prob_win=candidate.prob_win, r_multiple=(gross - costs) / risk,
            hit_t1=reason == "TARGET", bars_held=pos["bars"], gross_r_multiple=gross / risk,
            friction_r_applied=costs / risk, shares=initial_quantity, risk_inr=risk,
            strategy_id=candidate.strategy_id,
            signal_time=candidate.signal_time,
            exit_price=pos["exit_value"] / initial_quantity, exit_reason=reason, net_return=net_return,
            exit_fills=pos["exit_fills"],
            benchmark_return=benchmark_return, excess_return=net_return - benchmark_return,
            **pos["context"],
        ))

    def signals(date: pd.Timestamp, nav: float) -> list[ReplaySignal]:
        nonlocal correlation, context
        frames = {t: f.loc[f.index <= date].iloc[-252:] for t, f in processed.items()
                  if date in f.index}
        if config.BENCHMARK not in frames:
            return []
        bench = frames[config.BENCHMARK].Close
        sectors = compute_sector_rs(frames, bench, config)
        breadth = compute_breadth(frames, config)
        regime = classify_regime(frames, breadth, tracker, config, sector_rs=sectors)
        close_config = replace(config, CAPITAL_INR=max(0., nav), PLATT_A=-4., PLATT_B=2.)
        if signal_provider is None:
            results = score_universe(processed=frames, bench=bench, sector_rs=sectors, regime=regime,
                                     config=close_config, weights=get_regime_factor_weights(regime.label),
                                     session="CLOSING_TREND", direction="LONG", allow_watchlist=False,
                                     now=date.to_pydatetime().replace(hour=23, minute=59))
            result: list[ReplaySignal] = list(sorted((r for r in results if not r.is_watchlist),
                                                    key=lambda r: (-r.expectancy_r, -r.composite, r.ticker)))
        else:
            result = list(signal_provider(frames, date, close_config))
        counts["signals_generated"] += len(result)
        bf = processed[config.BENCHMARK].loc[:date]
        ema200 = float(bf.iloc[-1].get("EMA_200", np.nan))
        volatility = bf.ATR / bf.Close
        threshold = volatility.iloc[-253:-1].median()
        labels = dict(entry_regime=regime.label,
                      benchmark_trend=("ABOVE_EMA200" if bench.iloc[-1] >= ema200 else "BELOW_EMA200") if np.isfinite(ema200) else "UNKNOWN",
                      volatility_regime=("HIGH" if volatility.iloc[-1] > threshold else "LOW") if np.isfinite(threshold) else "UNKNOWN",
                      breadth_regime="STRONG" if breadth >= .55 else "WEAK" if breadth < .45 else "NEUTRAL")
        context = {r.ticker: dict(labels, research_context=getattr(r, "research_context", {})) for r in result}
        returns = {t: (f.Close.reindex(bench.index).pct_change(fill_method=None).iloc[-60:]
                       if portfolio_limits else f.Close.pct_change().iloc[-60:]) for t, f in frames.items()}
        correlation = pd.DataFrame(returns).corr(
            min_periods=portfolio_limits.min_correlation_observations if portfolio_limits else 1)
        return result

    if warm_history:
        prior_dates = raw[config.BENCHMARK].index[raw[config.BENCHMARK].index < dates[0]]
        # The final prior close is classified by signals(); never count it twice.
        for prior_date in prior_dates[-max(2, config.REGIME_CONFIRM_BARS) - 1:-1]:
            frames = {t: f.loc[:prior_date].iloc[-252:] for t, f in processed.items() if prior_date in f.index}
            sectors = compute_sector_rs(frames, frames[config.BENCHMARK].Close, config)
            classify_regime(frames, compute_breadth(frames, config), tracker, config, sector_rs=sectors)
        if len(prior_dates):
            pending = signals(prior_dates[-1], initial)

    for date in dates:
        bars = {t: frame.loc[date] for t, frame in raw.items() if date in frame.index}
        previous_nav = equity[-1]["nav"] if equity else initial
        previous_exposure = equity[-1]["exposure_close"] if equity else 0.
        # Only opening information is available before submitting pending entries.
        for ticker, pos in list(book.items()):
            if ticker not in bars:
                continue
            opening = float(bars[ticker].Open)
            if opening <= pos["stop"]:
                exit_position(ticker, opening, date, "TRAIL_STOP" if pos["stop"] > pos["initial_stop"] else "STOP", at_open=True)
            elif policy.mode == "SCALE_OUT" and not pos["scaled"] and opening >= pos["partial_target"]:
                pos["scaled"] = True
                half = pos["initial_shares"] // 2
                if half:
                    exit_position(ticker, opening, date, "SCALE_1R", at_open=True, quantity=half)
                if ticker in book and opening >= pos["target"]:
                    exit_position(ticker, opening, date, "TARGET", at_open=True)
            elif policy.mode in {"FIXED", "SCALE_OUT"} and opening >= pos["target"]:
                exit_position(ticker, opening, date, "TARGET", at_open=True)
        nav_open = cash + sum(p["shares"] * float(bars[t].Open if t in bars else p["mark"])
                              for t, p in book.items())
        sizing = replace(config, CAPITAL_INR=max(0., nav_open))
        for candidate in pending:
            counts["entry_attempts"] += 1
            guard_reason = entry_guard(candidate, date) if entry_guard else None
            if guard_reason:
                decision(candidate, date, guard_reason)
                continue
            ticker = f"{candidate.ticker}.NS"
            if ticker not in bars:
                rejected["missing_bar"] += 1
                decision(candidate, date, "missing_bar")
                continue
            reason = ""
            if ticker in book:
                reason = "already_held"
            elif len(book) >= config.PORTFOLIO_SIZE:
                reason = "position_cap"
            elif sum(p["candidate"].sector == candidate.sector for p in book.values()) >= config.MAX_SECTOR_PICKS:
                reason = "sector_cap"
            elif config.MAX_CORR < 1 or portfolio_limits is not None:
                for held in book:
                    if held not in correlation or ticker not in correlation.index or not np.isfinite(float(correlation.loc[ticker, held])):
                        reason = "correlation_unknown"
                        break
                    corr_value = abs(float(correlation.loc[ticker, held]))
                    if (corr_value > config.MAX_CORR or
                            (portfolio_limits is not None and corr_value >= portfolio_limits.max_correlation)):
                        reason = "correlation_cap"
                        break
            if reason:
                rejected["portfolio_limit"] += 1
                decision(candidate, date, reason)
                continue
            price = float(bars[ticker].Open)
            previous = processed[ticker].loc[processed[ticker].index < date]
            if price <= 0 or previous.empty:
                rejected["gap_or_size"] += 1
                decision(candidate, date, "invalid_open_or_history")
                continue
            if candidate.strategy_id.startswith("SWING_"):
                plan = SwingPlan(candidate.strategy_id, candidate.signal_time, candidate.entry,
                                 candidate.entry_min, candidate.entry_max, candidate.stop, candidate.t1,
                                 candidate.t2, candidate.time_stop_bars, float(previous.ATR.iloc[-1]))
                quantity, risk = swing_fill_size(plan, price, sizing)
                stop, target = plan.stop, plan.target
            else:
                levels = compute_targets("LONG", price, float(previous.ATR.iloc[-1]), config)
                stop, target = levels.stop, levels.t1
                budget = min(config.RISK_PER_TRADE_INR, nav_open * config.SWING_RISK_FRACTION)
                quantity = max(0, min(int(budget / (price - stop)),
                                      int(nav_open * config.SWING_MAX_EXPOSURE_FRACTION / price)))
                risk = quantity * (price - stop)
            if quantity <= 0 or price <= stop or stop <= 0 or target <= price:
                rejected["gap_or_size"] += 1
                decision(candidate, date, "entry_bounds_or_rr" if candidate.strategy_id.startswith("SWING_") else "invalid_levels_or_size")
                continue
            if portfolio_limits is not None:
                sector_value = sum(p["shares"] * float(bars[t].Open if t in bars else p["mark"])
                                   for t, p in book.items() if p["candidate"].sector == candidate.sector)
                quantity, limit_reason = bounded_quantity(
                    quantity, price, stop, nav_open, sum(p["risk"] for p in book.values()), sector_value,
                    config.MAX_PORTFOLIO_RISK_INR, portfolio_limits,
                    getattr(candidate, "research_context", {}).get("breadth_risk_scale", 1.))
                if quantity <= 0:
                    decision(candidate, date, limit_reason)
                    continue
            available_risk = max(0., config.MAX_PORTFOLIO_RISK_INR - sum(p["risk"] for p in book.values()))
            risk_quantity = int(available_risk / (price - stop))
            cash_quantity = max(0, int((cash - cost_model.commission_inr) / (price * (1 + cost_model.entry_cost_pct()))))
            requested_quantity = quantity
            quantity = min(quantity, risk_quantity, cash_quantity)
            risk = quantity * (price - stop)
            if quantity <= 0:
                rejected["gap_or_size"] += 1
                decision(candidate, date, "risk_cap" if risk_quantity <= 0 else "cash_cap")
                continue
            if risk_quantity < requested_quantity:
                counts["resized_by_risk"] += 1
            if cash_quantity < min(requested_quantity, risk_quantity):
                counts["resized_by_cash"] += 1
            entry_cost = price * quantity * cost_model.entry_cost_pct() + cost_model.commission_inr
            cash -= price * quantity + entry_cost
            book[ticker] = dict(entry=price, shares=quantity, stop=stop, target=target, date=date,
                                bars=0, mark=price, risk=risk, entry_cost=entry_cost, candidate=candidate,
                                benchmark_entry=float(bars[config.BENCHMARK].Open), context=context[candidate.ticker])
            book[ticker].update(initial_shares=quantity, initial_risk=risk, initial_stop=stop,
                                realized_gross=0., exit_costs=0., exit_value=0., benchmark_weighted_return=0.,
                                exit_fills=[], scaled=False, partial_target=price + (price - stop),
                                highest_close=price, highest_high=price)
            if policy.mode == "SCALE_OUT":
                book[ticker]["target"] = price + 3 * (price - stop)
            decision(candidate, date, "filled", quantity)
        gross_open = sum(p["shares"] * float(bars[t].Open if t in bars else p["mark"]) for t, p in book.items())
        exposure_open = gross_open / (cash + gross_open) if cash + gross_open > 0 else 0.
        # Intrabar exits happen after entry allocation; their cash cannot fund earlier orders.
        for ticker, pos in list(book.items()):
            if ticker not in bars:
                continue
            bar = bars[ticker]
            pos["bars"] += 1
            pos["mark"] = float(bar.Close)
            if bar.Low <= pos["stop"]:
                threshold = pos["partial_target"] if policy.mode == "SCALE_OUT" and not pos["scaled"] else pos["target"]
                if bar.High >= threshold:
                    counts["ambiguous_stop_and_target_bars"] += 1
                exit_position(ticker, pos["stop"], date, "TRAIL_STOP" if pos["stop"] > pos["initial_stop"] else "STOP")
                continue
            if policy.mode == "SCALE_OUT" and not pos["scaled"] and bar.High >= pos["partial_target"]:
                pos["scaled"] = True
                half = pos["initial_shares"] // 2
                if half:
                    exit_position(ticker, pos["partial_target"], date, "SCALE_1R", quantity=half)
            if policy.mode in {"FIXED", "SCALE_OUT"} and bar.High >= pos["target"]:
                exit_position(ticker, pos["target"], date, "TARGET")
                continue
            if pos["bars"] >= pos["candidate"].time_stop_bars:
                exit_position(ticker, float(bar.Close), date, "TIME")
                continue
            if policy.mode in {"ATR_TRAIL", "CHANDELIER"}:
                # The just-completed close/high/ATR can only change tomorrow's stop.
                pos["highest_close"] = max(pos["highest_close"], float(bar.Close))
                pos["highest_high"] = max(pos["highest_high"], float(bar.High))
                atr = float(processed[ticker].loc[date, "ATR"])
                if np.isfinite(atr) and atr > 0:
                    anchor = pos["highest_close"] if policy.mode == "ATR_TRAIL" else pos["highest_high"]
                    pos["stop"] = max(pos["stop"], anchor - policy.atr_multiple * atr)
        if date == dates[-1]:
            for ticker, pos in list(book.items()):
                exit_position(ticker, pos["mark"], date, "DATA_END")
        nav = cash + sum(p["shares"] * p["mark"] for p in book.values())
        bench_open, bench_close = float(bars[config.BENCHMARK].Open), float(bars[config.BENCHMARK].Close)
        previous_bench_close = equity[-1]["benchmark_close"] if equity else bench_open
        overnight = bench_open / previous_bench_close - 1
        intraday = bench_close / bench_open - 1
        sector_values: dict[str, float] = {}
        for position in book.values():
            sector = position["candidate"].sector
            sector_values[sector] = sector_values.get(sector, 0.) + position["shares"] * position["mark"]
        equity.append(dict(date=date, nav=nav, cash=cash, positions=len(book), daily_return=nav / previous_nav - 1,
                           portfolio_heat=sum(p["risk"] for p in book.values()) / nav if nav > 0 else 0.,
                           max_sector_exposure=max(sector_values.values(), default=0.) / nav if nav > 0 else 0.,
                           exposure_open=exposure_open, exposure_close=(nav - cash) / nav if nav > 0 else 0.,
                           benchmark_close=bench_close, benchmark_return=bench_close / previous_bench_close - 1,
                           exposure_matched_benchmark_return=previous_exposure * overnight + exposure_open * intraday))
        pending = signals(date, nav)
        if date == dates[-1]:
            for candidate in pending:
                decision(candidate, date, "expired_at_data_end")
    curve = pd.DataFrame(equity).set_index("date")
    daily_returns = curve.nav.pct_change().fillna(0.)
    daily_returns.iloc[0] = curve.nav.iloc[0] / initial - 1
    drawdown = curve.nav / curve.nav.cummax().clip(lower=initial) - 1
    metrics = dict(total_return=float(curve.nav.iloc[-1] / initial - 1),
                   max_drawdown=float(drawdown.min()),
                   daily_nav_sharpe=float(np.sqrt(252) * daily_returns.mean() / daily_returns.std()) if daily_returns.std() > 0 else 0.,
                   final_nav=float(curve.nav.iloc[-1]), minimum_cash=float(curve.cash.min()),
                   peak_positions=int(curve.positions.max()), rejected=rejected)
    metrics["diagnostics"] = dict(counts)
    metrics["exit_policy"] = dict(mode=policy.mode, atr_multiple=policy.atr_multiple)
    return ReplayResult(WalkForwardResult(trades, [], _overall_stats(trades, [])), curve, metrics, pd.DataFrame(decisions))
