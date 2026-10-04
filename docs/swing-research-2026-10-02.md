# Swing research validation — 2026-10-02

All ten requested research capabilities are implemented. The evidence does not establish positive swing expectancy, and strategy defaults remain unchanged. Enable the fixed rules explicitly with `--swing-rules`; no sensitivity winner has been adopted.

## Interpretation

Continuous validation from 2024-01-01 through 2026-10-01 produced **80 rules trades, −0.0800R net expectancy, 45% wins and −1.55% NAV return**, versus −5.73% NAV for the baseline. Lower losses than the baseline do not establish an edge. Rules drawdown was −2.48%, average opening exposure 4.29%, annualized regression alpha −0.62%, and exposure-matched benchmark underperformance −1.91 percentage points. The entry-month clustered 95% Mean-R interval was [−0.2922, +0.1483]; it does not exclude losses.

Breakouts averaged −0.0943R across 45 trades; pullbacks −0.0616R across 35. Both were negative. Entry below benchmark EMA200 averaged −0.5017R across only 12 trades; above it averaged −0.0056R across 68. Weak breadth had only two trades. These descriptive subdivisions do not justify choosing new filters after seeing outcomes.

The development-only sensitivity grid covers 2022-01-03 through 2023-12-29 and has 11 variants. Lookback changes produced identical filled trades in this sample; this shows no observed difference here, rather than proving general robustness. RVOL/time variants remained positive in development, while ATR variants changed eligibility substantially: 22/73/62 trades at 0.5/1.0/1.5 ATR and +2.82%/+2.51%/+0.14% NAV. Development gains did not persist in the fixed-policy validation. No variant is selected.

Mean R is net expectancy per trade expressed in initial-risk units. Results include modeled fees/slippage, fixed fractional sizing, next-session entries and a cash-constrained portfolio. Exposure-normalized ratios are descriptive, and all benchmark proxies/limitations remain visible below.

## Reproduction and prospective collection

Run `.venv\Scripts\python.exe scripts/research_swing.py --snapshot artifacts/performance_validation/historical.pkl`. Canonical generated evidence is in `artifacts/swing_research/`: `report.json`, `report.md`, and each run's trades, daily equity and signal-decision CSVs. Snapshot SHA-256: `698d4232d6f6e164deeb2ed87e094e373ec53204c26b28a64168bc0a6ecfcf23`.

The expanding history begins on 2022-01-03. Parameters are fixed, with causal prior-data warming and no outcome fitting. Repartitioning previously inspected dates cannot restore an untouched holdout. Quarter books start flat and liquidate at each boundary; the continuous run retains positions across quarters. The final quarter has one observed session.

The [prospective paper workflow](paper-trading-ledger.md) records first-observed signals, zero-signal scan days and explicit linked paper fills/exits. Its append-only export is initialized empty. Future observations are required; no historical backtest was inserted into this journal.

Verification: **703 tests passed, 93.09% core coverage**; final paper-ledger checks passed after adding frozen factor snapshots. Ruff passed and Mypy reported no errors across 98 sources under configured rules. Tests cover causal replay, benchmark calculations, distinct-date regime confirmation, atomic ledger rollback/idempotency and refusal to rewrite historical rows.

## Validation schedule

| Window | Prior history ends | Validation starts | Validation ends | Sessions | Partial |
|---|---|---|---|---:|---|
| 2024Q1 | 2023-12-29 | 2024-01-01 | 2024-03-28 | 60 | no |
| 2024Q2 | 2024-03-28 | 2024-04-01 | 2024-06-28 | 60 | no |
| 2024Q3 | 2024-06-28 | 2024-07-01 | 2024-09-30 | 64 | no |
| 2024Q4 | 2024-09-30 | 2024-10-01 | 2024-12-31 | 62 | no |
| 2025Q1 | 2024-12-31 | 2025-01-01 | 2025-03-28 | 62 | no |
| 2025Q2 | 2025-03-28 | 2025-04-01 | 2025-06-30 | 61 | no |
| 2025Q3 | 2025-06-30 | 2025-07-01 | 2025-09-30 | 64 | no |
| 2025Q4 | 2025-09-30 | 2025-10-01 | 2025-12-31 | 62 | no |
| 2026Q1 | 2025-12-31 | 2026-01-01 | 2026-03-30 | 59 | no |
| 2026Q2 | 2026-03-30 | 2026-04-01 | 2026-06-30 | 60 | no |
| 2026Q3 | 2026-06-30 | 2026-07-01 | 2026-09-30 | 65 | no |
| 2026Q4 | 2026-09-30 | 2026-10-01 | 2026-10-01 | 1 | yes |

## Complete opportunity counts (continuous rules)

| Decision | Count |
|---|---:|
| Scorer-qualified daily opportunities | 124 |
| Filled | 80 |
| Already held, no additional entry | 25 |
| Entry bounds or reward-to-risk rejected | 13 |
| Sector cap | 2 |
| Position cap | 2 |
| Correlation cap | 2 |
| Risk cap | 0 |
| Cash cap | 0 |
| Unfilled at data end | 0 |

Terminal decisions are exclusive and sum to 124. The 25 held observations are not new rejected entries; 19 other opportunities did not fill. Repeated daily opportunities need not represent distinct independent setups. Unobserved counters are zero; they are omitted from the compact JSON diagnostic object below. These counts show no large correlation bottleneck in this particular sample.

RETROSPECTIVE, PREVIOUSLY VIEWED DATA — NOT PROSPECTIVE EVIDENCE

Expanding prior-history / nonoverlapping quarterly validation; fixed rules, no refitting or parameter selection. Prior data warms causal indicators, regime confirmation and trailing volatility context. Prior-close signals may fill at the first validation open. No outcome labels are fitted, so no label purge is needed. Adjacent market periods are not assumed statistically independent.

## Quarterly validation

| Window | Policy | Trades | Mean R | Win % | NAV return | Drawdown | Exposure | Daily NAV Sharpe |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2024Q1 | baseline | 35 | 0.0143 | 40.00% | 0.09% | -2.36% | 25.51% | 0.0994 |
| 2024Q1 | rules | 11 | 0.0063 | 54.55% | 0.02% | -0.71% | 6.72% | 0.0535 |
| 2024Q2 | baseline | 43 | 0.0067 | 39.53% | 0.07% | -3.18% | 23.82% | 0.0766 |
| 2024Q2 | rules | 11 | -0.1191 | 36.36% | -0.32% | -1.33% | 4.39% | -0.6247 |
| 2024Q3 | baseline | 49 | 0.2443 | 53.06% | 3.01% | -2.41% | 29.50% | 2.3391 |
| 2024Q3 | rules | 15 | -0.0033 | 66.67% | -0.03% | -0.70% | 10.62% | -0.0509 |
| 2024Q4 | baseline | 32 | -0.2659 | 37.50% | -2.09% | -3.53% | 18.36% | -1.9917 |
| 2024Q4 | rules | 1 | -1.0542 | 0.00% | -0.26% | -0.29% | 0.28% | -2.9915 |
| 2025Q1 | baseline | 8 | -0.9150 | 12.50% | -1.81% | -1.97% | 2.88% | -4.2016 |
| 2025Q1 | rules | 0 | — | — | 0.00% | 0.00% | 0.00% | — |
| 2025Q2 | baseline | 45 | 0.0036 | 53.33% | -0.05% | -3.77% | 22.50% | 0.0072 |
| 2025Q2 | rules | 8 | 0.3449 | 62.50% | 0.67% | -0.60% | 3.31% | 1.7456 |
| 2025Q3 | baseline | 48 | -0.3545 | 25.00% | -4.16% | -4.64% | 25.26% | -3.4289 |
| 2025Q3 | rules | 5 | -0.0583 | 40.00% | -0.09% | -0.49% | 1.87% | -0.4885 |
| 2025Q4 | baseline | 50 | 0.1427 | 48.00% | 1.78% | -1.99% | 32.31% | 1.4181 |
| 2025Q4 | rules | 13 | 0.2259 | 53.85% | 0.71% | -0.46% | 7.53% | 1.6520 |
| 2026Q1 | baseline | 23 | -0.6163 | 17.39% | -3.48% | -3.96% | 15.33% | -3.3785 |
| 2026Q1 | rules | 2 | -0.8046 | 0.00% | -0.40% | -0.43% | 1.62% | -3.5025 |
| 2026Q2 | baseline | 42 | 0.1285 | 50.00% | 1.29% | -2.69% | 23.30% | 0.9463 |
| 2026Q2 | rules | 5 | -0.2889 | 20.00% | -0.35% | -0.54% | 2.22% | -1.8651 |
| 2026Q3 | baseline | 52 | -0.1191 | 38.46% | -1.52% | -2.76% | 24.13% | -1.1883 |
| 2026Q3 | rules | 12 | -0.5308 | 0.00% | -1.50% | -1.54% | 8.23% | -2.6153 |
| 2026Q4 | baseline | 3 | -0.7354 | 0.00% | -0.53% | -0.53% | 14.66% | — |
| 2026Q4 | rules | 0 | — | — | 0.00% | 0.00% | 0.00% | — |

Training and validation timestamps, partial-quarter flags and full diagnostics are in report.json. Quarter books start flat and liquidate at their boundary; the continuous run below preserves positions across quarters.

## Continuous validation portfolio

| Policy | CAGR | Exposure | CAGR / exposure¹ | Return / exposure¹ | Benchmark CAGR | Beta | Annual alpha² | Information ratio | Exposure-matched excess NAV return³ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | -2.16% | 22.03% | -9.81% | -25.99% | 1.17% | 0.1784 | -2.42% | -0.3437 | -3.08% |
| rules | -0.58% | 4.29% | -13.43% | -36.07% | 1.17% | 0.0246 | -0.62% | -0.1999 | -1.91% |

¹ Descriptive ratios, not attainable fully invested returns. ² Daily OLS intercept ×252 with cash yield/risk-free rate set to zero; no statistical significance claim. ³ Benchmark overnight return uses prior-close exposure and its daytime return uses post-allocation opening exposure. Intrabar exit time is unknown, so this is an exposure proxy.

## Setup and regime attribution (continuous rules)

### strategy_id

| Group | Trades | Mean R | Win % | Average hold | Net INR P&L | Stop frequency | Mean excess trade return |
|---|---:|---:|---:|---:|---:|---:|---:|
| SWING_BREAKOUT_V1 | 45 | -0.0943 | 46.67% | 12.2222 | -10540.6982 | 24.44% | -0.40% |
| SWING_PULLBACK_V1 | 35 | -0.0616 | 42.86% | 7.4000 | -4920.7046 | 34.29% | -1.05% |

### entry_regime

| Group | Trades | Mean R | Win % | Average hold | Net INR P&L | Stop frequency | Mean excess trade return |
|---|---:|---:|---:|---:|---:|---:|---:|
| TREND_DOWN | 2 | -0.2713 | 50.00% | 8.0000 | -1359.0911 | 50.00% | -0.16% |
| TREND_UP | 78 | -0.0751 | 44.87% | 10.1667 | -14102.3117 | 28.21% | -0.70% |

### benchmark_trend

| Group | Trades | Mean R | Win % | Average hold | Net INR P&L | Stop frequency | Mean excess trade return |
|---|---:|---:|---:|---:|---:|---:|---:|
| ABOVE_EMA200 | 68 | -0.0056 | 50.00% | 9.8382 | -614.7133 | 29.41% | -0.41% |
| BELOW_EMA200 | 12 | -0.5017 | 16.67% | 11.6667 | -14846.6894 | 25.00% | -2.24% |

### volatility_regime

| Group | Trades | Mean R | Win % | Average hold | Net INR P&L | Stop frequency | Mean excess trade return |
|---|---:|---:|---:|---:|---:|---:|---:|
| HIGH | 46 | -0.0739 | 41.30% | 10.3696 | -8437.0520 | 23.91% | -0.23% |
| LOW | 34 | -0.0883 | 50.00% | 9.7647 | -7024.3508 | 35.29% | -1.30% |

### breadth_regime

| Group | Trades | Mean R | Win % | Average hold | Net INR P&L | Stop frequency | Mean excess trade return |
|---|---:|---:|---:|---:|---:|---:|---:|
| STRONG | 78 | -0.0751 | 44.87% | 10.1667 | -14102.3117 | 28.21% | -0.70% |
| WEAK | 2 | -0.2713 | 50.00% | 8.0000 | -1359.0911 | 50.00% | -0.16% |
| NEUTRAL | 0 | — | — | — | 0.0000 | — | — |

### exit_reason

| Group | Trades | Mean R | Win % | Average hold | Net INR P&L | Stop frequency | Mean excess trade return |
|---|---:|---:|---:|---:|---:|---:|---:|
| STOP | 23 | -1.0852 | 0.00% | 4.7826 | -60930.0193 | 100.00% | -5.92% |
| TARGET | 9 | 1.6616 | 100.00% | 7.1111 | 36665.7599 | 0.00% | 9.15% |
| TIME | 48 | 0.0751 | 56.25% | 13.2292 | 8802.8567 | 0.00% | -0.02% |

## Return distribution and benchmark-relative trades

```json
{
  "percentiles": {
    "5": -1.0897218968404967,
    "25": -1.037943049793892,
    "50": -0.11512181251462254,
    "75": 0.438821400832786,
    "95": 1.6235313681153098
  },
  "skewness": 0.5561364097486647,
  "excess_kurtosis": -0.550901057127601,
  "largest_winner": 1.9505678509985283,
  "largest_loser": -1.3282282706369264,
  "mean_excess_return": -0.006852981208274013,
  "median_excess_return": -0.006041135962016463,
  "fraction_outperforming_benchmark": 0.4625,
  "monthly_mean_r_ci95": [
    -0.2922007771101404,
    0.14830867973343184
  ]
}
```

Stock returns are net of modeled costs. Holding-period benchmark returns use entry open to exit close, except opening-gap exits use benchmark open. An intraday stop/target exit therefore uses a closing-price benchmark proxy.

## Signal decisions

```json
{
  "signals_generated": 124,
  "entry_attempts": 124,
  "filled": 80,
  "already_held": 25,
  "entry_bounds_or_rr": 13,
  "sector_cap": 2,
  "position_cap": 2,
  "correlation_cap": 2
}
```

Counts refer to scorer-qualified daily opportunities. Terminal reasons are exclusive; resizing counters describe fills reduced rather than rejected. Per-signal decisions are saved as CSV.

## Prespecified sensitivity grid (development only)

| Variant | Trades | Mean R | NAV return | Drawdown | Exposure | Stop frequency |
|---|---:|---:|---:|---:|---:|---:|
| fixed_default | 37 | 0.3372 | 3.03% | -0.92% | 3.75% | 21.62% |
| breakout_lookback_15 | 37 | 0.3372 | 3.03% | -0.92% | 3.75% | 21.62% |
| breakout_lookback_25 | 37 | 0.3372 | 3.03% | -0.92% | 3.75% | 21.62% |
| breakout_rvol_1.1 | 36 | 0.3797 | 3.33% | -0.91% | 3.74% | 22.22% |
| breakout_rvol_1.3 | 36 | 0.3314 | 2.89% | -0.92% | 3.63% | 22.22% |
| both_time_exits_10 | 40 | 0.3599 | 3.51% | -0.82% | 3.11% | 12.50% |
| both_time_exits_15 | 36 | 0.3616 | 3.16% | -1.09% | 3.81% | 22.22% |
| both_time_exits_20 | 34 | 0.4510 | 3.77% | -0.96% | 4.12% | 23.53% |
| atr_stop_0.5 | 22 | 0.6600 | 2.82% | -0.33% | 0.98% | 31.82% |
| atr_stop_1.0 | 73 | 0.1839 | 2.51% | -2.07% | 5.19% | 49.32% |
| atr_stop_1.5 | 62 | 0.0197 | 0.14% | -2.26% | 5.37% | 45.16% |

Defaults remain fixed. The grid is one-factor-at-a-time, not a search or a ranking. Wider stops also change target distances and eligible setups; they are policy variants, not isolated causal estimates of stop width.

## Limitations

- All historical dates were already available/inspected; re-splitting cannot create a fresh holdout.
- Current constituents and the 60-stock subset retain survivorship/selection bias; point-in-time membership is unavailable.
- Daily adjusted bars, index-price benchmark and modeled fills/fees are proxies, not actual executable or dividend-inclusive total returns.
- Quarter-boundary liquidation differs from a continuous trading book; small/partial windows have limited evidence.
- The sensitivity grid diagnoses instability without selecting a winner or changing deployed defaults.
- Regime groups describe entry conditions; missing groups are absence of evidence and do not justify automatic filter tightening.
- Prospective paper evidence requires future observed signals and linked fill/exit events; historical events are never relabeled prospective.
