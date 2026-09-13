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
# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION COST MODEL  (NSE intraday defaults)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TransactionCostModel:
    """
    Round-trip friction for NSE intraday trades, expressed as percentages of
    the *entry price* per side (buy or sell).

    Default values reflect Zerodha/Fyers retail costs on NSE CashMIS (intraday):

    =========================================  ===============================
    Component                                  Default
    =========================================  ===============================
    Slippage (half-spread + mkt impact)        0.05 % per side
    Brokerage                                  0.03 % per side
    STT (Securities Transaction Tax)           0.025 % on SELL side only
    NSE exchange transaction charge            0.00345 % per side
    SEBI regulatory charge                     0.0001 % per side
    Stamp duty                                 0.015 % on BUY side only
    GST (on brokerage + exchange + SEBI)       18 %  (multiplicative)
    =========================================  ===============================

    Approximate round-trip drag: ~0.21 % of entry price, which on a typical
    1 ATR stop (~1.5 % of price) costs roughly **0.14 R** per trade.

    Use ``ZERO_COST_MODEL`` for frictionless back-tests (legacy / unit-test
    behaviour).  Pass a custom instance to ``walk_forward`` to model different
    broker tiers or larger slippage assumptions.
    """
    # Per-side
    slippage_pct:        float = 0.0005       # 0.05 %
    brokerage_pct:       float = 0.0003       # 0.03 %
    exchange_pct:        float = 0.0000345    # NSE exchange charge
    sebi_pct:            float = 0.000001     # SEBI charge
    # Asymmetric
    stt_sell_pct:        float = 0.00025      # STT on sell side only (intraday)
    stamp_buy_pct:       float = 0.00015      # Stamp duty on buy side only
    # GST applies on brokerage + exchange + SEBI
    gst_rate:            float = 0.18

    # ── derived helpers ───────────────────────────────────────────────────────

    def _gst_mult(self) -> float:
        return 1.0 + self.gst_rate

    def entry_cost_pct(self) -> float:
        """Total buy-side friction as a fraction of entry price."""
        taxable = (self.brokerage_pct + self.exchange_pct + self.sebi_pct) * self._gst_mult()
        return self.slippage_pct + taxable + self.stamp_buy_pct

    def exit_cost_pct(self) -> float:
        """Total sell-side friction as a fraction of exit price (approx entry price)."""
        taxable = (self.brokerage_pct + self.exchange_pct + self.sebi_pct) * self._gst_mult()
        return self.slippage_pct + self.stt_sell_pct + taxable

    def friction_r(self, entry: float, sl_dist: float) -> float:
        """
        Round-trip cost drag expressed in R-multiples.

        Parameters
        ----------
        entry:
            Entry price (INR).
        sl_dist:
            Absolute distance from entry to stop-loss (must be > 0).

        Returns
        -------
        A positive float — subtract this from the gross R-multiple to get the
        net (after-cost) R-multiple.  Returns 0.0 if sl_dist ≤ 0.
        """
        if sl_dist <= 0:
            return 0.0
        total_pct = self.entry_cost_pct() + self.exit_cost_pct()
        return round(entry * total_pct / sl_dist, 5)


# Pre-built instances — reference by name instead of constructing inline.
DEFAULT_COST_MODEL = TransactionCostModel()   # NSE intraday retail defaults
ZERO_COST_MODEL    = TransactionCostModel(    # frictionless (legacy / tests)
    slippage_pct=0.0, brokerage_pct=0.0, exchange_pct=0.0,
    sebi_pct=0.0, stt_sell_pct=0.0, stamp_buy_pct=0.0, gst_rate=0.0,
)



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
    r_multiple: float    # net R-multiple after transaction costs
    hit_t1:     bool
    bars_held:  int
    # Cost fields — default 0.0 so positional construction in existing code
    # and tests remains valid.
    gross_r_multiple:   float = field(default=0.0)  # R before cost deduction
    friction_r_applied: float = field(default=0.0)  # cost drag in R units


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
    cost_model: TransactionCostModel = ZERO_COST_MODEL,
) -> tuple[float, bool, int, Optional[pd.Timestamp], float, float]:
    """
    Simulate a single trade against forward OHLCV bars.

    Exit logic (in priority order):
    1. Stop hit (High/Low touches stop level) → -1.0 R gross
    2. T1 hit (High/Low touches target)       → +RR R gross
    3. Time-stop (max bars held)              → exit at close of last bar
    4. End of test window                     → exit at last available close

    Transaction costs are applied via *cost_model*.  Pass ``ZERO_COST_MODEL``
    for frictionless behaviour (unit tests, legacy callers).

    Returns
    -------
    (net_r, hit_t1, bars_held, exit_date, gross_r, friction_r_applied)
    """
    sl_dist = abs(entry - stop)
    if sl_dist <= 0:
        return 0.0, False, 0, None, 0.0, 0.0

    rr      = abs(t1 - entry) / sl_dist
    friction = cost_model.friction_r(entry, sl_dist)

    for i, (ts, bar) in enumerate(fwd_bars.iterrows()):
        if i >= time_stop:
            close   = float(bar["Close"])
            gross_r = (close - entry) / sl_dist if direction == "LONG" else (entry - close) / sl_dist
            gross_r = round(gross_r, 4)
            net_r   = round(gross_r - friction, 4)
            return net_r, False, i + 1, ts, gross_r, round(friction, 5)

        open_p = float(bar["Open"])
        high = float(bar["High"])
        low  = float(bar["Low"])

        if direction == "LONG":
            if open_p <= stop:
                gross_r = (open_p - entry) / sl_dist
                return round(gross_r - friction, 4), False, i + 1, ts, round(gross_r, 4), round(friction, 5)
            if low <= stop:
                gross_r = -1.0
                return round(gross_r - friction, 4), False, i + 1, ts, gross_r, round(friction, 5)
            if open_p >= t1:
                gross_r = (open_p - entry) / sl_dist
                return round(gross_r - friction, 4), True, i + 1, ts, round(gross_r, 4), round(friction, 5)
            if high >= t1:
                gross_r = round(rr, 4)
                return round(gross_r - friction, 4), True, i + 1, ts, gross_r, round(friction, 5)
        else:
            if open_p >= stop:
                gross_r = (entry - open_p) / sl_dist
                return round(gross_r - friction, 4), False, i + 1, ts, round(gross_r, 4), round(friction, 5)
            if high >= stop:
                gross_r = -1.0
                return round(gross_r - friction, 4), False, i + 1, ts, gross_r, round(friction, 5)
            if open_p <= t1:
                gross_r = (entry - open_p) / sl_dist
                return round(gross_r - friction, 4), True, i + 1, ts, round(gross_r, 4), round(friction, 5)
            if low <= t1:
                gross_r = round(rr, 4)
                return round(gross_r - friction, 4), True, i + 1, ts, gross_r, round(friction, 5)

    # Ran out of bars: exit at last close
    if len(fwd_bars) == 0:
        return 0.0, False, 0, None, 0.0, 0.0

    last_bar  = fwd_bars.iloc[-1]
    close     = float(last_bar["Close"])
    exit_date = fwd_bars.index[-1]
    gross_r   = round(
        (close - entry) / sl_dist if direction == "LONG" else (entry - close) / sl_dist, 4
    )
    net_r = round(gross_r - friction, 4)
    return net_r, False, len(fwd_bars), exit_date, gross_r, round(friction, 5)


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
    cost_model:     TransactionCostModel = DEFAULT_COST_MODEL,
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
    cost_model:
        Transaction cost model to apply to each simulated trade.
        Defaults to ``DEFAULT_COST_MODEL`` (NSE intraday retail costs).
        Pass ``ZERO_COST_MODEL`` for frictionless results.

    Returns
    -------
    ``WalkForwardResult`` with per-trade records and summary statistics.
    Net R-multiples in ``TradeRecord.r_multiple`` already reflect costs;
    the gross figure is preserved in ``TradeRecord.gross_r_multiple``.

    Notes
    -----
    **Next-Day Open Assumption**: The simulation enters trades at the exact
    opening price of the day following the signal. This removes the optimistic
    assumption of filling at the exact MOC (Market on Close) price of the signal day.
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

    if step_days < test_days:
        log.warning(
            "Overlapping walk-forward test periods detected: step_days (%d) < test_days (%d). "
            "Auto-adjusting step_days to %d to ensure statistically independent out-of-sample test windows.",
            step_days, test_days, test_days,
        )
        step_days = test_days

    fold_starts = range(0, total_bars - required + 1, step_days)
    n_folds     = len(list(fold_starts))
    log.info(
        "Walk-forward: %d folds | train=%d test=%d step=%d | tickers=%d | "
        "cost=%.3f%% round-trip (entry=%.4f%% + exit=%.4f%%)",
        n_folds, train_days, test_days, step_days, len(tickers),
        (cost_model.entry_cost_pct() + cost_model.exit_cost_pct()) * 100,
        cost_model.entry_cost_pct() * 100,
        cost_model.exit_cost_pct() * 100,
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

            _, time_stop = compute_trade_management_wrapper(d, close, atr, row)

            # Get forward (test-window) bars for this ticker
            full_df   = raw_data[ticker]
            fwd_bars  = full_df.loc[full_df.index.isin(test_dates)].copy()

            if fwd_bars.empty:
                continue

            entry_price = float(fwd_bars.iloc[0]["Open"])
            targets = compute_targets(d, entry_price, atr, config)

            r, hit, bars, exit_date, gross_r, friction = _realised_r(
                direction=d,
                entry=entry_price,
                stop=targets.stop,
                t1=targets.t1,
                fwd_bars=fwd_bars,
                time_stop=time_stop,
                cost_model=cost_model,
            )

            fold_trades.append(TradeRecord(
                fold=fold_idx,
                ticker=ticker,
                direction=d,
                entry_date=fwd_bars.index[0],
                exit_date=exit_date,
                entry=round(entry_price, 2),
                stop=targets.stop,
                t1=targets.t1,
                composite=round(composite, 4),
                prob_win=round(prob, 4),
                r_multiple=r,
                hit_t1=hit,
                bars_held=bars,
                gross_r_multiple=gross_r,
                friction_r_applied=friction,
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
