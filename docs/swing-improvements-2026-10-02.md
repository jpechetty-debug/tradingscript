# Swing improvements and fixed comparison — 2 October 2026

The swing policy now has explicit setups, bounded entry plans, fixed fractional sizing and a daily portfolio replay. The observed performance does not establish a profitable edge. It remains opt-in for research and paper evaluation.

## Policy

`SWING_BREAKOUT_V1` requires a close above the preceding 20-session high, relative volume of at least 1.2, limited extension and a strong closing candle. `SWING_PULLBACK_V1` requires a recent EMA20 touch, closes holding EMA50 and a bullish reclaim above the previous session's high. Both require close > EMA20 > EMA50 > EMA200, rising EMA50, positive benchmark trend and 63-session relative strength at least matching the benchmark. Existing liquidity, data-quality and regime gates also apply.

Only completed daily candles enter setup, indicator, breadth and sector calculations. New swing signals ignore intraday prices and session multipliers. Version identifiers, signal timestamps, entry bounds and `HEURISTIC_UNVALIDATED` probability status accompany recommendations. Fixed regime factor weights and heuristic sigmoid coefficients are used; legacy fitted weights and probabilities are not assumed valid for these new setups.

Stops sit below recent five-session structure with an ATR buffer. Targets use 2R with an overhead-resistance cap; minimum proposed reward/risk is 1.5. Entry bounds prevent buying an adverse next-session opening gap or moving the stop to accommodate it. Default initial risk is capped at 0.25% of sizing capital and position notional at 10%, with additional existing INR risk limits. This replaces Kelly sizing for the new setups; zero-share trades are rejected. Breakouts use a 15-session time exit and pullbacks 10 sessions.

## Replay protocol

The fixed comparison reuses the saved five-year, 60-stock-plus-benchmark adjusted daily snapshot. No parameter search was performed. Development covers 10 October 2022–23 September 2025. The recent comparison covers 24 September 2025–1 October 2026. That recent period had already been inspected during the previous evaluation, so it is exploratory evidence rather than an untouched holdout.

Signals form after a daily close and may fill at the next session's open only. Pending signals expire after that session. Stops and targets remain tied to the original setup; quantity is recalculated for actual proposed fill risk. The simulation tracks cash, held positions, portfolio slots, sector limits, aggregate initial risk and trailing 60-session return correlation. Unknown correlation blocks additional positions when a correlation cap is active. Opening exit proceeds are available for opening entries; later intraday exit proceeds cannot fund earlier entries. Same-bar stop/target ambiguity resolves to the stop. Entry and exit friction uses the existing swing cost assumptions; surviving positions liquidate at the end of each evaluated period.

The baseline uses the existing daily scoring policy with the same fixed fractional risk and notional caps as the rules policy. This isolates setup and exit-policy differences; it does not reproduce the old Kelly-sized smoke test. Portfolio returns and drawdowns derive from the daily cash and marked position NAV. Mean-R intervals use a monthly block bootstrap with 2,000 draws and fixed seed 20261002.

## Results

| Policy / period | Trades | Mean net R | NAV return | NAV max drawdown | Mean-R 95% interval |
|---|---:|---:|---:|---:|---|
| Baseline development | 449 | +0.0345 | +3.60% | -7.70% | [-0.1258, +0.1902] |
| Rules development | 83 | +0.1877 | +3.80% | -1.80% | [-0.0547, +0.4616] |
| Baseline recent period | 163 | -0.0716 | -2.97% | -5.71% | [-0.3684, +0.2621] |
| Rules recent period | 30 | -0.1829 | -1.30% | -2.27% | [-0.5548, +0.2130] |
| Rules recent period, 2× slippage | 30 | -0.2099 | -1.49% | -2.39% | [-0.5848, +0.1895] |

Fewer entries and lower observed portfolio drawdown are encouraging risk-control results. The recent sample still loses money and its mean net R is worse than the baseline. Every interval includes losses and gains. These results justify retaining a clear research hypothesis; they do not demonstrate that this is the best swing strategy or support live deployment.

## Limits and next evidence

Current constituents and the selected 60-stock subset introduce survivorship and selection bias. Adjusted daily OHLC cannot establish intrabar ordering, executable liquidity or actual broker fills. Fees and slippage are research assumptions. There is no earnings/event feed, so event avoidance is not implemented. Stop orders cannot guarantee a maximum loss through a market gap. There is no broker order submission in this change.

Further evidence requires point-in-time membership and event data, a newly reserved chronological period, and prospective paper tracking of signal times, rejected entries, fills, fees and exits. Evaluate net performance and drawdown against a benchmark and exposure before considering live operation. Strategy changes require a new version and fresh validation, rather than tuning against this already viewed period.

## Use and reproduction

```powershell
# Research scan; does not place an order
.venv\Scripts\python.exe run.py --swing-rules --no-telegram

# Reuse the existing snapshot for the fixed daily comparison
.venv\Scripts\python.exe scripts/compare_swing.py --snapshot artifacts/performance_validation/historical.pkl

# Regression and coverage checks
.venv\Scripts\python.exe -m pytest tests/ -q --cov=core --cov-fail-under=80
```

`SWING_SETUP_ENABLED=true` also selects this policy for API scans. The default is false. Trades, daily NAV curves and JSON/Markdown reports are saved under `artifacts/swing_comparison/`. Setup, scorer, market-preparation, bounded-fill and held-book limit regressions cover the new behavior.
