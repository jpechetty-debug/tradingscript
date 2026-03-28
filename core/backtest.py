"""
core/backtest.py
================
Walk-forward backtesting framework for the Sovereign Engine.

Architecture
------------
A walk-forward test splits the full history into *anchored* windows:

    Train window  →  Test window  →  (advance by step)
    [─────────────────────]
             [─────────────────────]
                      [─────────────────────]

For each fold:
1. Run the scoring pipeline on the *train* slice of every ticker's daily
   data to produce ``TickerResult`` objects with composite scores.
2. Evaluate actual forward returns over the *test* slice.
3. Record per-fold metrics (Sharpe, hit-rate, mean R-multiple, max DD).

After all folds, aggregate into an ``OverallStats`` summary.

Key public API
--------------
::

    results = walk_forward(
        processed,          # {ticker: full_df_with_indicators}
        bench_series,       # Nifty50 Close series
        config,             # SystemConfig
        train_days=120,
        test_days=20,
        step_days=10,
    )
    print(results.overall)
    results.to_csv("wf_results.csv")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .config import SystemConfig
from .factors import compute_factors, DEFAULT_WEIGHTS
from .indicators import add_indicators
from .portfolio import compute_targets
from .regime import (
    RegimeTracker,
    classify_regime,
    compute_breadth,
    compute_sector_rs,
)
from .scorer import composite_to_prob, passes_liquidity, compute_trade_management
from .universe import TICKER_TO_SECTOR, N_SECTORS

log = logging.getLogger("sovereign.backtest")


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    """Single simulated trade from the backtest."""
    fold:       int
    ticker:     str
    direction:  str
    entry_date: pd.Timestamp
    exit_date:  Optional[pd.Timestamp]
    entry:      float
    stop:       float
    t1:         float
    composite:  float
    prob_win:   float
    r_multiple: float    # realised R-multiple (+ = profit, - = loss)
    hit_t1:     bool
    bars_held:  int


@dataclass
class FoldStats:
    """Per-fold statistics."""
    fold:        int
    start_date:  pd.Timestamp
    end_date:    pd.Timestamp
    n_trades:    int
    hit_rate:    float    # fraction of trades that hit T1
    mean_r:      float    # mean R-multiple
    sharpe:      float    # annualised Sharpe of daily R-multiples
    max_dd:      float    # maximum drawdown of cumulative R-curve
    total_r:     float    # sum of R-multiples


@dataclass
class OverallStats:
    """Aggregated statistics across all folds."""
    n_folds:         int
    n_trades:        int
    hit_rate:        float
    mean_r:          float
    sharpe:          float    # annualised, across all trades
    max_dd:          float
    total_r:         float
    profit_factor:   float    # gross wins / gross losses
    expectancy_r:    float    # mean_r weighted by prob_win (model vs actual)
    fold_stats:      list[FoldStats] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Container returned by ``walk_forward``."""
    trades:      list[TradeRecord]
    fold_stats:  list[FoldStats]
    overall:     OverallStats

    def to_dataframe(self) -> pd.DataFrame:
        """Convert trade records to a tidy DataFrame for analysis."""
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def to_csv(self, path: str) -> None:
        """Write trade records to CSV."""
        self.to_dataframe().to_csv(path, index=False)
        log.info("Walk-forward results written to %s", path)


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _realised_r(
    direction: str,
    entry:     float,
    stop:      float,
    t1:        float,
    fwd_bars:  pd.DataFrame,
    time_stop: int,
) -> tuple[float, bool, int, Optional[pd.Timestamp]]:
    """
    Simulate a single trade against forward OHLCV bars.

    Exit logic (in priority order):
    1. Stop hit (High/Low touches stop level) → -1.0 R
    2. T1 hit (High/Low touches target)       → +RR R
    3. Time-stop (max bars held)              → exit at close of last bar
    4. End of test window                     → exit at last available close

    Returns
    -------
    (r_multiple, hit_t1, bars_held, exit_date)
    """
    sl_dist = abs(entry - stop)
    if sl_dist <= 0:
        return 0.0, False, 0, None

    rr = abs(t1 - entry) / sl_dist

    for i, (ts, bar) in enumerate(fwd_bars.iterrows()):
        if i >= time_stop:
            # Time stop: exit at close
            close = float(bar["Close"])
            r = (close - entry) / sl_dist if direction == "LONG" else (entry - close) / sl_dist
            return round(r, 4), False, i + 1, ts

        high  = float(bar["High"])
        low   = float(bar["Low"])

        if direction == "LONG":
            if low <= stop:
                return -1.0, False, i + 1, ts
            if high >= t1:
                return round(rr, 4), True, i + 1, ts
        else:
            if high >= stop:
                return -1.0, False, i + 1, ts
            if low <= t1:
                return round(rr, 4), True, i + 1, ts

    # Ran out of bars: exit at last close
    if len(fwd_bars) == 0:
        return 0.0, False, 0, None

    last_bar  = fwd_bars.iloc[-1]
    close     = float(last_bar["Close"])
    exit_date = fwd_bars.index[-1]
    r = (close - entry) / sl_dist if direction == "LONG" else (entry - close) / sl_dist
    return round(r, 4), False, len(fwd_bars), exit_date


def _fold_stats(fold: int, trades: list[TradeRecord], dates: tuple) -> FoldStats:
    """Compute statistics for one fold from its trade list."""
    start, end = dates
    if not trades:
        return FoldStats(fold=fold, start_date=start, end_date=end,
                         n_trades=0, hit_rate=0.0, mean_r=0.0,
                         sharpe=0.0, max_dd=0.0, total_r=0.0)

    rs    = [t.r_multiple for t in trades]
    hits  = [t.hit_t1 for t in trades]
    total = sum(rs)
    mean  = total / len(rs)
    hit_r = sum(hits) / len(hits)

    # Annualised Sharpe of trade R-multiples (assume ~252 trades/year avg rate)
    std = float(np.std(rs)) if len(rs) > 1 else 0.0
    sharpe = round((mean / std) * np.sqrt(252), 3) if std > 0 else 0.0

    # Max drawdown of cumulative R curve
    cum  = np.cumsum([0.0] + rs)
    peak = np.maximum.accumulate(cum)
    dd   = float(np.min(cum - peak))

    return FoldStats(
        fold=fold, start_date=start, end_date=end,
        n_trades=len(trades), hit_rate=round(hit_r, 4),
        mean_r=round(mean, 4), sharpe=sharpe,
        max_dd=round(dd, 4), total_r=round(total, 4),
    )


def _overall_stats(all_trades: list[TradeRecord], fold_stats: list[FoldStats]) -> OverallStats:
    """Aggregate statistics across all folds."""
    if not all_trades:
        return OverallStats(n_folds=len(fold_stats), n_trades=0,
                            hit_rate=0.0, mean_r=0.0, sharpe=0.0,
                            max_dd=0.0, total_r=0.0, profit_factor=0.0,
                            expectancy_r=0.0, fold_stats=fold_stats)

    rs    = [t.r_multiple for t in all_trades]
    total = sum(rs)
    mean  = total / len(rs)
    hits  = sum(1 for t in all_trades if t.hit_t1)

    std = float(np.std(rs)) if len(rs) > 1 else 0.0
    sharpe = round((mean / std) * np.sqrt(252), 3) if std > 0 else 0.0

    cum  = np.cumsum([0.0] + rs)
    peak = np.maximum.accumulate(cum)
    dd   = float(np.min(cum - peak))

    wins   = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    pf     = (sum(wins) / abs(sum(losses))) if losses else float("inf")

    exp_r  = round(
        sum(t.r_multiple * t.prob_win for t in all_trades) / len(all_trades), 4
    )

    return OverallStats(
        n_folds=len(fold_stats),
        n_trades=len(all_trades),
        hit_rate=round(hits / len(all_trades), 4),
        mean_r=round(mean, 4),
        sharpe=sharpe,
        max_dd=round(dd, 4),
        total_r=round(total, 4),
        profit_factor=round(pf, 3),
        expectancy_r=exp_r,
        fold_stats=fold_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(
    raw_data:       dict[str, pd.DataFrame],
    config:         SystemConfig,
    train_days:     int = 120,
    test_days:      int = 20,
    step_days:      int = 10,
    min_prob:       float | None = None,
    direction:      str = "LONG",
    max_trades_per_fold: int = 10,
) -> WalkForwardResult:
    """
    Run a walk-forward backtest over *raw_data*.

    Parameters
    ----------
    raw_data:
        ``{ticker: OHLCV_DataFrame}`` — *without* indicators applied.
        The function slices each ticker's data per fold and re-applies
        ``add_indicators`` on the train window.
    config:
        ``SystemConfig`` instance.
    train_days:
        Bars in the train (signal-generation) window.
    test_days:
        Bars in the forward test window.
    step_days:
        Step between fold start dates.
    min_prob:
        Minimum ``prob_win`` to include a trade.  Defaults to
        ``config.BACKTEST_MIN_PROB``.
    direction:
        ``"LONG"`` or ``"SHORT"`` (or ``"BOTH"``).
    max_trades_per_fold:
        Cap on trades per fold (ranked by composite score) to avoid
        over-trading in volatile regimes.

    Returns
    -------
    ``WalkForwardResult`` with per-trade records and summary statistics.
    """
    min_prob  = min_prob if min_prob is not None else config.BACKTEST_MIN_PROB
    all_trades: list[TradeRecord] = []
    all_folds:  list[FoldStats]   = []

    # Work only with tickers that have data
    bench_key = config.BENCHMARK
    tickers   = [t for t in raw_data if t != bench_key and not raw_data[t].empty]
    bench_raw = raw_data.get(bench_key, pd.DataFrame())

    if bench_raw.empty:
        log.error("Benchmark %s missing from raw_data — cannot run backtest.", bench_key)
        return WalkForwardResult(trades=[], fold_stats=[], overall=_overall_stats([], []))

    # Determine usable date range from the benchmark
    all_dates = bench_raw.index
    total_bars = len(all_dates)
    required   = train_days + test_days

    if total_bars < required:
        log.error(
            "Benchmark has only %d bars; need at least %d (train=%d + test=%d).",
            total_bars, required, train_days, test_days,
        )
        return WalkForwardResult(trades=[], fold_stats=[], overall=_overall_stats([], []))

    fold_starts = range(0, total_bars - required + 1, step_days)
    n_folds     = len(list(fold_starts))
    log.info(
        "Walk-forward: %d folds | train=%d test=%d step=%d | tickers=%d",
        n_folds, train_days, test_days, step_days, len(tickers),
    )
    regime_tracker = RegimeTracker()

    for fold_idx, start_i in enumerate(range(0, total_bars - required + 1, step_days)):
        train_end_i = start_i + train_days
        test_end_i  = train_end_i + test_days

        train_dates = all_dates[start_i:train_end_i]
        test_dates  = all_dates[train_end_i:test_end_i]

        if len(train_dates) < train_days or len(test_dates) < 1:
            continue

        log.debug(
            "Fold %d: train %s→%s | test %s→%s",
            fold_idx,
            train_dates[0].date(), train_dates[-1].date(),
            test_dates[0].date(), test_dates[-1].date(),
        )

        # ── Build processed train data ────────────────────────────────────────
        processed: dict[str, pd.DataFrame] = {}
        for ticker in tickers + [bench_key]:
            df = raw_data[ticker]
            # Slice to train window
            train_df = df.loc[df.index.isin(train_dates)].copy()
            if len(train_df) < 50:   # need enough bars for indicators
                continue
            try:
                processed[ticker] = add_indicators(train_df, config)
            except Exception:
                log.debug("Indicator error for %s in fold %d.", ticker, fold_idx, exc_info=True)

        if bench_key not in processed:
            log.debug("Benchmark missing from processed data in fold %d — skipping.", fold_idx)
            continue

        bench_series = processed[bench_key]["Close"]

        # ── Regime (on train window) ──────────────────────────────────────────
        breadth = compute_breadth(processed, config)
        sector_rs = compute_sector_rs(processed, bench_series, config)
        regime = classify_regime(
            processed,
            breadth,
            regime_tracker,
            config,
            sector_rs=sector_rs,
        )

        directions = ["LONG", "SHORT"] if direction == "BOTH" else [direction]

        # ── Score each ticker on last bar of train window ─────────────────────
        candidates: list[tuple[float, float, str, str, pd.Series, pd.DataFrame]] = []
        # (composite, prob_win, ticker, direction, last_row, full_ticker_df)

        sector_ranks = {s: i+1 for i, (s, _) in enumerate(
            sorted(sector_rs.items(), key=lambda x: x[1], reverse=True)
        )}

        for ticker in tickers:
            df = processed.get(ticker)
            if df is None or df.empty:
                continue

            row   = df.iloc[-1]
            close = float(row["Close"])

            liq_ok, _ = passes_liquidity(row, config)
            if not liq_ok:
                continue

            sector = TICKER_TO_SECTOR.get(ticker, "")

            for d in directions:
                if d == "LONG" and not regime.allows_long():
                    continue
                if d == "SHORT" and not regime.allows_short():
                    continue
                try:
                    factors = compute_factors(
                        ticker=ticker,
                        daily_df=df,
                        bench=bench_series,
                        sector=sector,
                        sector_ranks=sector_ranks,
                        direction=d,
                        close=close,
                        row=row,
                        n_sectors=N_SECTORS,
                        weights=DEFAULT_WEIGHTS,
                    )
                    prob = composite_to_prob(
                        factors.composite, config.PLATT_A, config.PLATT_B
                    )
                    if prob >= min_prob:
                        candidates.append((factors.composite, prob, ticker, d, row, df))
                except Exception:
                    log.debug("Scoring error %s/%s fold %d.", ticker, d, fold_idx, exc_info=True)

        # Sort by composite descending and cap per fold
        candidates.sort(key=lambda x: x[0], reverse=True)
        candidates = candidates[:max_trades_per_fold]

        # ── Simulate each trade in the test window ────────────────────────────
        fold_trades: list[TradeRecord] = []

        for composite, prob, ticker, d, row, train_df in candidates:
            close = float(row["Close"])
            atr   = float(row.get("ATR", close * 0.015))
            targets = compute_targets(d, close, atr, config)

            _, time_stop = compute_trade_management_wrapper(d, close, atr, row)

            # Get forward (test-window) bars for this ticker
            full_df   = raw_data[ticker]
            fwd_bars  = full_df.loc[full_df.index.isin(test_dates)].copy()

            if fwd_bars.empty:
                continue

            r, hit, bars, exit_date = _realised_r(
                direction=d,
                entry=close,
                stop=targets.stop,
                t1=targets.t1,
                fwd_bars=fwd_bars,
                time_stop=time_stop,
            )

            fold_trades.append(TradeRecord(
                fold=fold_idx,
                ticker=ticker,
                direction=d,
                entry_date=train_df.index[-1],
                exit_date=exit_date,
                entry=round(close, 2),
                stop=targets.stop,
                t1=targets.t1,
                composite=round(composite, 4),
                prob_win=round(prob, 4),
                r_multiple=r,
                hit_t1=hit,
                bars_held=bars,
            ))

        fold_dates = (train_dates[-1], test_dates[-1])
        all_trades.extend(fold_trades)
        all_folds.append(_fold_stats(fold_idx, fold_trades, fold_dates))

        log.info(
            "Fold %d: %d trades | hit=%.1f%% | mean_R=%.3f | total_R=%.2f",
            fold_idx,
            len(fold_trades),
            all_folds[-1].hit_rate * 100,
            all_folds[-1].mean_r,
            all_folds[-1].total_r,
        )

    overall = _overall_stats(all_trades, all_folds)

    log.info(
        "Walk-forward complete: %d folds | %d trades | hit=%.1f%% | "
        "Sharpe=%.2f | MaxDD=%.2f | total_R=%.2f | PF=%.2f",
        overall.n_folds, overall.n_trades,
        overall.hit_rate * 100, overall.sharpe,
        overall.max_dd, overall.total_r, overall.profit_factor,
    )

    return WalkForwardResult(trades=all_trades, fold_stats=all_folds, overall=overall)


def compute_trade_management_wrapper(
    direction: str,
    close: float,
    atr: float,
    row: pd.Series,
) -> tuple[float, int]:
    """Thin shim to call scorer.compute_trade_management cleanly."""
    atr_pctile = float(row.get("ATR_Pctile", 50) or 50)
    return compute_trade_management(direction, close, atr, atr_pctile)
