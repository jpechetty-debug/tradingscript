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
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import SystemConfig
from .factors import DEFAULT_WEIGHTS
from .indicators import add_indicators
from .portfolio import compute_targets
from .regime import (
    RegimeTracker,
    classify_regime,
    compute_breadth,
    compute_sector_rs,
)
from .scorer import compute_trade_management
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
    slippage_pct:        float = 0.0008       # 8 bps (from SLIPPAGE_BPS=8)
    brokerage_pct:       float = 0.0003       # 0.03 %
    exchange_pct:        float = 0.0000345    # NSE exchange charge
    sebi_pct:            float = 0.000001     # SEBI charge
    # Asymmetric
    stt_buy_pct:         float = 0.0          # STT on buy side (0.0 for intraday, 0.001 for delivery)
    stt_sell_pct:        float = 0.00025      # STT on sell side (0.025% intraday, 0.1% delivery)
    stamp_buy_pct:       float = 0.00015      # Stamp duty on buy side only (0.015%)
    # GST applies on brokerage + exchange + SEBI
    gst_rate:            float = 0.18
    commission_inr:      float = 20.0         # Flat commission in INR per trade (from COMMISSION_INR=20)

    # ── derived helpers ───────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: Any) -> "TransactionCostModel":
        """Construct a TransactionCostModel wired directly to system config."""
        slippage_bps = getattr(config, "SLIPPAGE_BPS", 8)
        commission_inr = getattr(config, "COMMISSION_INR", 20)
        return cls(
            slippage_pct=float(slippage_bps) / 10_000.0,
            commission_inr=float(commission_inr),
        )

    def entry_cost_pct(self) -> float:
        """Total buy-side friction as a fraction of entry price."""
        taxable = (self.brokerage_pct + self.exchange_pct + self.sebi_pct) * (1.0 + self.gst_rate)
        return self.slippage_pct + self.stt_buy_pct + taxable + self.stamp_buy_pct

    def exit_cost_pct(self) -> float:
        """Total sell-side friction as a fraction of exit price (approx entry price)."""
        taxable = (self.brokerage_pct + self.exchange_pct + self.sebi_pct) * (1.0 + self.gst_rate)
        return self.slippage_pct + self.stt_sell_pct + taxable

    def friction_r(self, entry: float, sl_dist: float, shares: Optional[int] = None) -> float:
        """
        Round-trip cost drag expressed in R-multiples.

        Parameters
        ----------
        entry:
            Entry price (INR).
        sl_dist:
            Absolute distance from entry to stop-loss (must be > 0).
        shares:
            Optional number of shares. When provided along with non-zero commission_inr,
            incorporates flat commission drag alongside percentage costs.

        Returns
        -------
        A positive float — subtract this from the gross R-multiple to get the
        net (after-cost) R-multiple.  Returns 0.0 if sl_dist ≤ 0.
        """
        if sl_dist <= 0:
            return 0.0
        total_pct = self.entry_cost_pct() + self.exit_cost_pct()
        pct_friction = entry * total_pct / sl_dist
        flat_friction = (self.commission_inr / (sl_dist * shares)) if (shares is not None and shares > 0 and self.commission_inr > 0) else 0.0
        return round(pct_friction + flat_friction, 5)


# Pre-built instances — reference by name instead of constructing inline.
DEFAULT_COST_MODEL = TransactionCostModel()   # NSE intraday retail defaults
SWING_COST_MODEL   = TransactionCostModel(    # NSE delivery defaults (0.1% STT on both legs, 0.015% stamp, zero brokerage)
    brokerage_pct=0.0,
    stt_buy_pct=0.001,
    stt_sell_pct=0.001,
    stamp_buy_pct=0.00015,
    commission_inr=0.0,
)
ZERO_COST_MODEL    = TransactionCostModel(    # frictionless (legacy / tests)
    slippage_pct=0.0, brokerage_pct=0.0, exchange_pct=0.0,
    sebi_pct=0.0, stt_buy_pct=0.0, stt_sell_pct=0.0, stamp_buy_pct=0.0, gst_rate=0.0,
    commission_inr=0.0,
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
    trade_horizon:      str = field(default="SWING")  # "SWING" | "INTRADAY"
    shares:             int = field(default=0)
    risk_inr:           float = field(default=0.0)
    daily_pnl:          dict[Any, float] = field(default_factory=dict)


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
    shares:    Optional[int] = None,
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
    friction = cost_model.friction_r(entry, sl_dist, shares=shares)

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


def _compute_trade_daily_pnl(
    fwd_bars: pd.DataFrame,
    direction: str,
    entry_price: float,
    shares: int,
    bars_held: int,
    net_pnl: float,
    sl_dist: float = 0.0,
    risk_inr: float = 0.0,
) -> dict[Any, float]:
    """
    Mark-to-market daily PnL distribution across held bars.

    Ensures the sum of daily MTM PnLs exactly equals the trade's total net PnL
    (including transaction friction).
    """
    if bars_held <= 0 or fwd_bars.empty:
        return {}

    held_bars = fwd_bars.iloc[:bars_held]
    n = len(held_bars)
    if n == 0:
        return {}

    if n == 1:
        d = held_bars.index[0].date() if hasattr(held_bars.index[0], "date") else held_bars.index[0]
        return {d: round(net_pnl, 2)}

    res: dict[Any, float] = {}
    accumulated = 0.0

    # Bar 0 (entry day: from entry open to bar 0 close)
    d0 = held_bars.index[0].date() if hasattr(held_bars.index[0], "date") else held_bars.index[0]
    c0 = float(held_bars.iloc[0]["Close"])
    if shares > 0:
        pnl_0 = (c0 - entry_price) * shares if direction == "LONG" else (entry_price - c0) * shares
    elif sl_dist > 0 and risk_inr > 0:
        r0 = (c0 - entry_price) / sl_dist if direction == "LONG" else (entry_price - c0) / sl_dist
        pnl_0 = r0 * risk_inr
    else:
        pnl_0 = net_pnl / n

    pnl_0 = round(pnl_0, 2)
    res[d0] = pnl_0
    accumulated += pnl_0

    # Intermediate bars (close-to-close)
    for i in range(1, n - 1):
        di = held_bars.index[i].date() if hasattr(held_bars.index[i], "date") else held_bars.index[i]
        ci = float(held_bars.iloc[i]["Close"])
        c_prev = float(held_bars.iloc[i - 1]["Close"])
        if shares > 0:
            pnl_i = (ci - c_prev) * shares if direction == "LONG" else (c_prev - ci) * shares
        elif sl_dist > 0 and risk_inr > 0:
            ri = (ci - c_prev) / sl_dist if direction == "LONG" else (c_prev - ci) / sl_dist
            pnl_i = ri * risk_inr
        else:
            pnl_i = net_pnl / n

        pnl_i = round(pnl_i, 2)
        res[di] = pnl_i
        accumulated += pnl_i

    # Final bar (remainder to guarantee exact penny sum to net_pnl)
    dk = held_bars.index[-1].date() if hasattr(held_bars.index[-1], "date") else held_bars.index[-1]
    res[dk] = round(net_pnl - accumulated, 2)

    return res


def _daily_portfolio_sharpe(
    trades: list[TradeRecord],
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    capital: float = 1_000_000.0,
) -> float:
    """
    Annualised portfolio Sharpe computed from a daily mark-to-market portfolio return curve.

    Accrues concurrent position PnL across active holding bars.
    Captures cross-sectional correlation and concurrent risk rather than
    treating concurrent trades as independent sequential bets or booking
    entire PnL as a single spike on exit date.
    """
    if not trades:
        return 0.0

    valid_trades = [t for t in trades if t.exit_date is not None]
    if not valid_trades:
        return 0.0

    s_date = start_date or min(t.entry_date for t in valid_trades)
    e_date = end_date or max(t.exit_date for t in valid_trades if t.exit_date is not None)

    try:
        b_days = pd.bdate_range(s_date, e_date)
    except Exception:
        b_days = pd.DatetimeIndex([t.exit_date for t in valid_trades if t.exit_date is not None])

    if len(b_days) < 2:
        return 0.0

    daily_pnl: dict[Any, float] = {d.date(): 0.0 for d in b_days}
    for t in valid_trades:
        if t.exit_date is None:
            continue
        rsk = t.risk_inr if t.risk_inr > 0 else 5_000.0
        total_pnl = t.r_multiple * rsk

        if getattr(t, "daily_pnl", None):
            for d_raw, pnl_val in t.daily_pnl.items():
                d = d_raw.date() if hasattr(d_raw, "date") else d_raw
                if d in daily_pnl:
                    daily_pnl[d] += pnl_val
                else:
                    daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl_val
        else:
            # Fallback for synthetic/legacy records: accrue linearly over bars_held
            bars = max(1, t.bars_held)
            pnl_per_bar = total_pnl / bars
            try:
                t_exit = pd.Timestamp(t.exit_date)
                trade_bdays = pd.bdate_range(end=t_exit, periods=bars)
                for bd in trade_bdays:
                    d = bd.date()
                    if d in daily_pnl:
                        daily_pnl[d] += pnl_per_bar
                    else:
                        daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl_per_bar
            except Exception:
                d = t.exit_date.date() if hasattr(t.exit_date, "date") else t.exit_date
                if d in daily_pnl:
                    daily_pnl[d] += total_pnl
                else:
                    daily_pnl[d] = daily_pnl.get(d, 0.0) + total_pnl

    daily_rets = np.array([pnl / capital for pnl in daily_pnl.values()])
    std = float(np.std(daily_rets))
    if std <= 0:
        return 0.0
    mean = float(np.mean(daily_rets))
    return float(round((mean / std) * float(np.sqrt(252.0)), 3))


def _fold_stats(fold: int, trades: list[TradeRecord], dates: tuple) -> FoldStats:
    """Compute statistics for one fold from its trade list."""
    start, end = dates
    if not trades:
        return FoldStats(fold=fold, start_date=start, end_date=end,
                         n_trades=0, hit_rate=0.0, mean_r=0.0,
                         sharpe=0.0, max_dd=0.0, total_r=0.0)

    sorted_trades = sorted(trades, key=lambda t: t.entry_date)
    rs    = [t.r_multiple for t in sorted_trades]
    hits  = [t.hit_t1 for t in sorted_trades]
    total = sum(rs)
    mean  = total / len(rs)
    hit_r = sum(hits) / len(hits)

    valid_exits = [t.exit_date for t in sorted_trades if t.exit_date is not None]
    start_d = sorted_trades[0].entry_date if sorted_trades else None
    end_d = max(valid_exits) if valid_exits else (sorted_trades[-1].entry_date if sorted_trades else None)

    # Annualised portfolio Sharpe across active trading window (matching _overall_stats)
    sharpe = _daily_portfolio_sharpe(sorted_trades, start_d, end_d)

    # Max drawdown of cumulative R curve (in chronological order)
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

    sorted_trades = sorted(all_trades, key=lambda t: t.entry_date)
    rs    = [t.r_multiple for t in sorted_trades]
    total = sum(rs)
    mean  = total / len(rs)
    hits  = sum(1 for t in sorted_trades if t.hit_t1)

    valid_exits = [t.exit_date for t in sorted_trades if t.exit_date is not None]
    start_d = sorted_trades[0].entry_date if sorted_trades else None
    end_d = max(valid_exits) if valid_exits else (sorted_trades[-1].entry_date if sorted_trades else None)
    sharpe = _daily_portfolio_sharpe(sorted_trades, start_d, end_d)

    cum  = np.cumsum([0.0] + rs)
    peak = np.maximum.accumulate(cum)
    dd   = float(np.min(cum - peak))

    wins   = [r for r in rs if r > 0]
    losses = [r for r in rs if r < 0]
    pf     = (sum(wins) / abs(sum(losses))) if losses else float("inf")

    # Expectancy is mean_r
    exp_r  = round(mean, 4)

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
    horizon_filter: str = "SWING",
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

    **Survivorship Bias Warning**: The universe is evaluated on the current list of
    ALL_TICKERS. Tickers that were delisted, halted, or dropped out of the index during
    historical periods are not represented, which introduces potential survivorship bias.
    """
    min_prob  = min_prob if min_prob is not None else config.BACKTEST_MIN_PROB
    if cost_model is DEFAULT_COST_MODEL and horizon_filter.upper() == "SWING":
        cost_model = SWING_COST_MODEL
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

        # ── Score each ticker on last bar of train window via shared score_universe ─
        from .services import score_universe
        from .factors import get_regime_factor_weights
        from .portfolio import optimize_portfolio
        from .cache import ScanCache

        fold_weights = get_regime_factor_weights(regime.label) if hasattr(regime, "label") else DEFAULT_WEIGHTS
        results = score_universe(
            processed=processed,
            bench=bench_series,
            sector_rs=sector_rs,
            regime=regime,
            config=config,
            weights=fold_weights,
            session="CLOSING_TREND",
            allow_watchlist=False,
            direction=direction,
            min_bars=min(50, train_days),
        )

        candidates = [
            r for r in results
            if r.prob_win >= min_prob
            and (horizon_filter.upper() == "BOTH" or getattr(r, "trade_horizon", "SWING") == horizon_filter.upper())
            and not getattr(r, "is_watchlist", False)
        ]
        # Optimise portfolio with sector caps and correlation filter
        corr_matrix = ScanCache().corr_matrix(processed, config)
        portfolio_candidates = optimize_portfolio(candidates, config, corr_matrix)
        candidates = portfolio_candidates[:max_trades_per_fold]

        # ── Simulate each trade in the test window ────────────────────────────
        fold_trades: list[TradeRecord] = []

        for cand in candidates:
            d = cand.direction
            ticker = cand.ticker
            full_df = raw_data.get(ticker)
            if full_df is None or full_df.empty:
                continue

            fwd_bars = full_df.loc[full_df.index.isin(test_dates)].copy()
            if fwd_bars.empty:
                continue

            train_df = processed[ticker]
            row = train_df.iloc[-1]
            close = float(row["Close"])
            atr = float(row.get("ATR", close * 0.015))

            time_stop = getattr(cand, "time_stop_bars", None)
            if time_stop is None:
                _, time_stop = compute_trade_management_wrapper(d, close, atr, row)

            entry_price = float(fwd_bars.iloc[0]["Open"])
            targets = compute_targets(
                d,
                entry_price,
                atr,
                config,
                trade_horizon=getattr(cand, "trade_horizon", "SWING"),
            )

            r, hit, bars, exit_date, gross_r, friction = _realised_r(
                direction=d,
                entry=entry_price,
                stop=targets.stop,
                t1=targets.t1,
                fwd_bars=fwd_bars,
                time_stop=time_stop,
                cost_model=cost_model,
                shares=getattr(cand, "shares", None),
            )

            sl_dist = abs(entry_price - targets.stop)
            shares_held = getattr(cand, "shares", 0)
            trade_risk = getattr(cand, "risk_inr", 0.0)
            if trade_risk <= 0:
                trade_risk = getattr(config, "RISK_PER_TRADE_INR", 5_000.0)
            net_trade_pnl = r * trade_risk

            trade_daily_pnl = _compute_trade_daily_pnl(
                fwd_bars=fwd_bars,
                direction=d,
                entry_price=entry_price,
                shares=shares_held,
                bars_held=bars,
                net_pnl=net_trade_pnl,
                sl_dist=sl_dist,
                risk_inr=trade_risk,
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
                composite=round(cand.composite, 4),
                prob_win=round(cand.prob_win, 4),
                r_multiple=r,
                hit_t1=hit,
                bars_held=bars,
                gross_r_multiple=gross_r,
                friction_r_applied=friction,
                trade_horizon=getattr(cand, "trade_horizon", "SWING"),
                shares=shares_held,
                risk_inr=trade_risk,
                daily_pnl=trade_daily_pnl,
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
