# Structural swing experiments — 2026-10-02

The proposed research directions are useful hypotheses. This run improves measurement and future paper evidence, but does not establish a profitable structural edge or justify changing active entry/exit rules. All observations below come from previously viewed data with current-constituent and 60-stock subset bias.

## What the tests found

- **Top decile leadership did not help here.** The 63-session RS top decile had 16 trades at −0.1278R; the 252-session top decile had 15 at −0.4297R. A positive middle bucket or the eight trades in the 252-session 80–90 bucket cannot be selected as a new rule after inspecting many groups. These are liquid-subset ranks, not Nifty-500 ranks.
- **Sector-peer strength merits a future test, not deployment.** Positive leave-one-out peer RS had 49 trades at +0.0660R, versus six at −0.4248R with nonpositive peer RS; 25 lacked enough peers. The positive group's monthly-cluster interval was [−0.2202, +0.3681]. Mean benchmark excess return was only +0.0274% per trade. The proxy is not an official sector index.
- **Most losses were not missed +1R winners.** Only 1 of 44 losing trades definitely reached +1R before exit; the possible count including unknown exit-day ordering is also one. That provides little support for the suggested explanation that scale-out would rescue many losers.
- **Do not tighten from 1R to 0.5R on this evidence.** Fifteen of 36 winners definitely experienced at least 0.5R adverse excursion while held. This does not establish that widening stops would improve performance.
- **MFE is censored by the existing exits.** Only one trade definitely reached +2R while held; six possibly did when including exit-day extremes. Twenty-nine exits have unknown intraday ordering. Stop/target-day highs cannot be treated as pre-exit opportunities, and these paths cannot establish how far a trade would run after its original exit.
- **Time exits were frequent, but targets contributed more positive P&L.** The 48 time exits averaged +0.0751R and contributed about INR 8,803; nine targets averaged +1.6616R and contributed INR 36,666. The 23 stops lost about INR 60,930. Thus exit attribution alone does not show that entries are good or that time exits drive most positive P&L.
- **Volatility expansion is exploratory.** ATR20 expansion had 49 trades at −0.0048R versus 31 at −0.1989R without expansion; both uncertainty intervals include losses. Bollinger-bandwidth expansion also remained negative. These measurements are now recorded for future signals; no expansion gate was adopted.

## Exit comparison

The fixed policy reproduced the frozen 80-trade result. ATR trailing returned −3.23%, Chandelier −2.83%, and 50%-at-1R / remainder-at-3R scale-out −2.02%, against −1.55% fixed. All also trailed their exposure-matched benchmark. Each variant's cash book reconciles net trade P&L to final NAV; cash never goes negative. Three alternatives failing here does not prove every exit design fails, but it gives no reason to deploy these recipes.

## Implemented improvements and future test

`core/swing_research.py` provides causal MAE/MFE bounds and pre-selection liquid-universe leadership/breadth/volatility observations. `core/swing_exits.py` and the replay support isolated research exit policies, partial-leg cash/cost accounting, conservative ambiguous-bar handling and close-updated stops effective next session. Application fill/exit APIs retain their full-fill/full-close contract.

Future paper signals preserve these observations and the SHA-256 of [the fixed hypothesis protocol](swing-hypotheses-v1.json). The primary future hypothesis is the benchmark-EMA200 expectancy difference; leadership, sector, breadth and volatility groups are secondary descriptive tests. Observation begins 2026-10-05 and the evaluation ends 2027-10-02 independent of results. Small groups remain inconclusive. This is a local preregistration, and no future trading evidence has been manufactured.

Earnings avoidance and a true point-in-time Nifty 500 universe remain **not evaluated**, because the current snapshot lacks as-of event schedules, dated complete membership and delisted prices. Validated feed adapters and [input schemas](research-input-formats.md) are supplied; missing events never count as an all-clear. Supplied membership is an audit, not an automatic replacement for the missing price history.

## Reproduction and checks

Run `.venv\Scripts\python.exe scripts/research_structural_edge.py`. Full CSV and JSON evidence is in `artifacts/swing_structural/`; `annotated_trades.csv` contains each original validation trade's features and excursion bounds, with each policy's trade/leg, decision and daily-equity files alongside it. `--skip-exits` generates feature diagnostics only. Optional feed arguments are documented in the input schemas.

Snapshot SHA-256: `698d4232d6f6e164deeb2ed87e094e373ec53204c26b28a64168bc0a6ecfcf23`. Future protocol SHA-256: `f6e907e7bdb75fdc90590987bd6902c1f37bfb68487d9dd4f8899581a70993ee`.

The full regression suite passed **710 tests**. Final targeted checks after protocol hashing and trailing-stop attribution passed; aggregate coverage is **93.21%**, above the required 80%. Ruff and Mypy passed (103 sources). Tests cover future-price invariance, leave-one-out ranks, unknown exit-day ordering, as-of knowledge, partial quantities/fees, opening gaps, next-session trailing stops, frozen signal observations and atomic journal integrity. A separate final driver smoke run reproduced the feature/cohort results without repeating completed portfolio simulations.

EXPLORATORY ON PREVIOUSLY VIEWED DATA; NO WINNER SELECTED OR DEPLOYED

## MAE/MFE bounds

```json
{
  "trades": 80,
  "exit_day_order_unknown": 29,
  "losers_reached_1r": {
    "denominator": 44,
    "definite": 1,
    "possible_including_unknown_exit_day_order": 1
  },
  "winners_suffered_half_r": {
    "denominator": 36,
    "definite": 15,
    "possible_including_unknown_exit_day_order": 15
  },
  "all_reached_2r": {
    "denominator": 80,
    "definite": 1,
    "possible_including_unknown_exit_day_order": 6
  },
  "mfe_r_lower": {
    "25": 0.18638531897141492,
    "50": 0.655728809025512,
    "75": 0.9545598769570353,
    "95": 1.7062575751235614
  },
  "mfe_r_upper": {
    "25": 0.18638531897141492,
    "50": 0.655728809025512,
    "75": 0.9545598769570353,
    "95": 2.0231035277649747
  },
  "mae_r_lower": {
    "25": 0.4200924354484943,
    "50": 0.6539461217444784,
    "75": 1.0,
    "95": 1.0
  },
  "mae_r_upper": {
    "25": 0.4200924354484943,
    "50": 0.6539461217444784,
    "75": 1.029711107084701,
    "95": 1.6592903328046906
  }
}
```

Exit-session high/low may occur after a stop or target. Definite and possible counts are bounds, not reconstructed intraday paths. Opening exits exclude subsequent extremes. Close/time exits include the full session. Existing exits censor excursion paths; this analysis alone cannot establish a better exit.

## rs63_decile

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| 00-010 | 0 | — | — | — | — |
| 10-020 | 0 | — | — | — | — |
| 20-030 | 0 | — | — | — | — |
| 30-040 | 2 | 0.2962 | 0.5000 | -1.3282 | 0.0282 |
| 40-050 | 9 | 0.2516 | 0.4444 | -1.3267 | 0.0019 |
| 50-060 | 8 | -0.0807 | 0.5000 | -1.4603 | -0.0098 |
| 60-070 | 14 | -0.2466 | 0.4286 | -3.4522 | -0.0008 |
| 70-080 | 16 | -0.2299 | 0.3750 | -3.8180 | -0.0105 |
| 80-090 | 15 | 0.0376 | 0.6000 | -2.0867 | -0.0103 |
| 90-100 | 16 | -0.1278 | 0.3750 | -3.5592 | -0.0131 |

## rs252_decile

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| 00-010 | 1 | 0.2924 | 1.0000 | 0.0000 | 0.0031 |
| 10-020 | 3 | -0.1606 | 0.6667 | -1.0823 | -0.0304 |
| 20-030 | 3 | -0.0862 | 0.3333 | -2.1793 | -0.0015 |
| 30-040 | 5 | -0.4314 | 0.2000 | -2.1569 | -0.0254 |
| 40-050 | 12 | 0.3737 | 0.6667 | -0.4372 | 0.0084 |
| 50-060 | 9 | -0.3330 | 0.3333 | -2.9973 | -0.0253 |
| 60-070 | 8 | -0.2443 | 0.2500 | -2.7643 | -0.0043 |
| 70-080 | 14 | -0.1894 | 0.3571 | -4.2999 | -0.0067 |
| 80-090 | 8 | 0.7523 | 0.8750 | -0.1986 | 0.0525 |
| 90-100 | 15 | -0.4297 | 0.3333 | -6.7111 | -0.0304 |
| UNKNOWN | 2 | -0.1245 | 0.5000 | -1.0412 | -0.0188 |

## sector_alignment

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| NONPOSITIVE_PEER_RS | 6 | -0.4248 | 0.3333 | -3.4376 | -0.0399 |
| POSITIVE_PEER_RS | 49 | 0.0660 | 0.5102 | -6.0530 | 0.0003 |
| UNKNOWN | 25 | -0.2834 | 0.3600 | -7.7195 | -0.0129 |

## leadership_distance

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| 0-10% | 9 | 0.0948 | 0.5556 | -2.2582 | 0.0051 |
| 10-20% | 19 | -0.0106 | 0.4737 | -2.1455 | -0.0043 |
| 20-40% | 37 | -0.1247 | 0.4054 | -6.9983 | -0.0058 |
| 40%+ | 15 | -0.1626 | 0.4667 | -4.0794 | -0.0198 |

## breadth50_bucket

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| 00-020 | 0 | — | — | — | — |
| 20-040 | 0 | — | — | — | — |
| 40-060 | 6 | -0.2834 | 0.5000 | -3.1471 | -0.0387 |
| 60-080 | 48 | 0.0114 | 0.4792 | -7.0339 | -0.0011 |
| 80-100 | 26 | -0.2018 | 0.3846 | -5.2464 | -0.0102 |

## breadth200_bucket

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| 00-020 | 0 | — | — | — | — |
| 20-040 | 1 | 1.6513 | 1.0000 | 0.0000 | 0.1388 |
| 40-060 | 5 | -0.2734 | 0.4000 | -1.4091 | -0.0267 |
| 60-080 | 38 | -0.0103 | 0.4474 | -5.2935 | -0.0036 |
| 80-100 | 36 | -0.1748 | 0.4444 | -6.2927 | -0.0116 |

## atr20_bucket

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| 00-020 | 5 | 0.2269 | 0.8000 | -1.3282 | 0.0175 |
| 20-040 | 4 | 0.5686 | 0.5000 | -0.2499 | 0.0208 |
| 40-060 | 9 | -0.2088 | 0.3333 | -3.3467 | -0.0078 |
| 60-080 | 11 | 0.0003 | 0.6364 | -3.2501 | -0.0037 |
| 80-100 | 51 | -0.1555 | 0.3922 | -11.2010 | -0.0119 |

## atr_expanding

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| False | 31 | -0.1989 | 0.3871 | -8.4017 | -0.0169 |
| True | 49 | -0.0048 | 0.4898 | -5.1078 | -0.0005 |

## bandwidth_expanding

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| False | 23 | -0.1617 | 0.4348 | -5.9381 | -0.0195 |
| True | 57 | -0.0470 | 0.4561 | -8.9660 | -0.0017 |

## benchmark_above_ema200

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| False | 12 | -0.5017 | 0.1667 | -6.0203 | -0.0224 |
| True | 68 | -0.0056 | 0.5000 | -6.5354 | -0.0041 |

## earnings_status

| Group | Trades | Mean R | Win % | Trade-R curve drawdown¹ | Mean trade excess return |
|---|---:|---:|---:|---:|---:|
| UNKNOWN_NO_ASOF_EVENT_FEED | 80 | -0.0800 | 0.4500 | -9.6679 | -0.0069 |

¹ Cumulative net R in exit-date order, not a deployable subgroup NAV/drawdown. Win rates and excess returns are fractions. Small/empty groups and confidence intervals are retained in JSON; these post-hoc cohorts are not significance-tested discoveries.

## Research-only exit portfolios

| Policy | Trades | Mean R | NAV return | Drawdown | Exposure | Benchmark-matched excess NAV |
|---|---:|---:|---:|---:|---:|---:|
| FIXED | 80 | -0.0800 | -1.55% | -2.48% | 4.29% | -1.91% |
| ATR_TRAIL | 83 | -0.1635 | -3.23% | -3.29% | 4.01% | -3.11% |
| CHANDELIER | 81 | -0.1472 | -2.83% | -2.90% | 4.42% | -2.87% |
| SCALE_OUT | 80 | -0.1048 | -2.02% | -2.36% | 4.19% | -2.23% |

Exit variants retain original entry eligibility, initial stop and time limit. ATR trail uses highest held close minus 2 ATR; Chandelier uses highest held high minus 3 ATR. Both replace the fixed target, update after the close and apply next session. Stops never loosen. Scale-out sells floor(50% of initial shares) at 1R and the rest at 3R, with original stop/time exit. Extra legs incur their own fees. Unknown intraday stop/target order is stop-first; opening gaps fill at the opening price. Changed holding times alter later capital, admission and fill quantities, so these are complete portfolio variants, not identical-entry causal estimates.

## Data dependencies

```json
{
  "earnings": "NOT_EVALUATED_NO_EVENT_FEED",
  "membership": "NOT_EVALUATED_NO_MEMBERSHIP_HISTORY",
  "missing_member_price_symbols": []
}
```

Earnings absence is unknown, never an all-clear. Supplied events must be known at the historical signal close; supplied membership is audited as-of, including missing/delisted prices. Without complete membership/price/event histories this run cannot claim Nifty-500 point-in-time or earnings-avoidance evidence.

## Limits and next evaluation

- The available liquid 60-stock subset/current mappings retain survivorship, selection and sector-proxy bias.
- Rank denominators are stored; these are ranks in the available liquid universe, not Nifty-500 population deciles.
- Sector alignment is leave-one-out median peer relative strength with at least two peers, not an official sector-index EMA200 test.
- ATR20 is a Wilder-style smoothed true range; its percentile compares strictly prior observations. Volatility expansion compares the current ATR20 with its prior 20-session average.
- Prior signal session is inferred from next-session entry for old records; new replay records retain actual signal_time.
- Original factor/liquidity/setup gates restrict trade cohorts. Filled-trade attribution does not estimate the effect of replacing the whole candidate-generation pipeline.
- Exit attribution and favorable excursion do not establish that entries are good; daily bars do not resolve threshold ordering.
- All exit variants are research-only and leave application trade management unchanged.
- Future protocol is docs/swing-hypotheses-v1.json; no new live entry filters or automatic deployment are introduced.
