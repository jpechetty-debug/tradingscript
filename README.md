# Sovereign Engine v14.6-Modular 🏛️

[![Security Scan](https://img.shields.io/badge/Security-Verified-success?style=flat-square)](#)
[![Lint Compliance](https://img.shields.io/badge/Lint-Ruff%20%7C%20Mypy-blue?style=flat-square)](#)
[![Test Suite](https://img.shields.io/badge/Tests-735%20Passed-brightgreen?style=flat-square)](#)
[![Version](https://img.shields.io/badge/Version-14.6--Modular-indigo?style=flat-square)](#)

**Quantitative research, paper trading and portfolio screening for NSE India.**

Sovereign Engine is a modular, high-performance quantitative screener and trading engine designed for systematic Indian cash equity markets. Built with mathematical rigor, the platform incorporates continuous $C^0/C^1$ factor scoring curves, Bayesian-shrunk IC weight calibration, a Two-Pass Cohort Scan pipeline with cross-sectional percentile ranking, and signal position hysteresis backed by SQLite persistence.

The execution ledger accepts explicit fill events; it does not submit broker orders. Current performance evidence does not establish a profitable live strategy.

---

## 🚀 Key Architectural Pillars

The diagram below describes the legacy scoring policy. The opt-in swing policy uses explicit daily setups, structural stops and fixed fractional sizing, described next.

```mermaid
graph TD
    A[Market Data Fetcher<br/>Fyers API v3 + yfinance fallback] --> B[Static & Liquidity Filters<br/>ADV, Turnover, Data Quality]
    B --> C[Market Regime Detection<br/>RegimeTracker: TREND, RANGE, PANIC]
    C --> D[Pass 1: Indicator Engine<br/>Compute Raw 7-Factor Model]
    D --> E{Directional Cohort<br/>N >= 10 Candidates?}
    E -- Yes --> F[Cross-Sectional Factor Ranking<br/>0.60 Raw + 0.40 Cohort Percentile]
    E -- No --> G[Raw Factor Pass-Through<br/>100% Unadjusted Scores]
    F --> H[Pass 2: Platt Calibration & Gating<br/>Entry: P>=0.52 | Open Pos: P>=0.47]
    G --> H
    H --> I[True Volume Profile & Kelly Sizing<br/>POC/VAL/VAH + Kurtosis Correction]
    I --> J[Portfolio Construction<br/>Covariance & Sector Constraints]
    J --> K[SQLite State Sync & Notifications<br/>open_positions, Telemetry, Telegram]
```

1. **Source-State-Log (SSL) Boundary Isolation**: Zero mutable state inside source files. All mutable state (`state.db`, `portfolio_state.json`, `factor_weights.json`) resides under `state/`, artifacts in `artifacts/`, and structured events in `logs/sovereign.jsonl`.
2. **Two-Pass Cohort Scan Pipeline**: Separates raw candidate extraction from cross-sectional relative ranking and position hysteresis gating.
3. **Continuous $C^0/C^1$ Factor Curves**: Step functions and threshold cliffs are eliminated in favor of continuous piecewise-linear and Hermite cubic spline smoothstep curves, preserving core domain knowledge plateaus without noise flips.
4. **Bayesian Shrinkage IC Calibration**: Eliminates overlapping-sample bias via independent 5-bar steps, shrinking noisy empirical ICIR weights toward the macro regime prior with a 5% floor simplex projection.
5. **Position Hysteresis & SQLite Persistence**: Signal edge churn is mitigated through asymmetric entry/holding thresholds ($P \ge 0.52$ for new entries vs. $P \ge 0.47$ for open positions), synchronized to SQLite `open_positions`.
6. **Dual-Provider Resilience**: Thread-safe async fetching prioritizing Fyers API v3 with automatic, circuit-broken fallback to yfinance.

---

## Swing rules — opt-in research policy

Run `python run.py --swing-rules --no-telegram` for the versioned long-only swing policy. `SWING_SETUP_ENABLED=true` also enables it for API scans. It remains disabled by default because the observed validation period does not establish positive net expectancy.

- **Completed daily bars:** unfinished candles are excluded before indicators, breadth, sector strength and setup detection. Live intraday prices, session score multipliers and legacy fitted calibration do not alter new swing entries.
- **Two explicit setups:** `SWING_BREAKOUT_V1` requires a 20-session breakout with relative volume confirmation; `SWING_PULLBACK_V1` requires a bullish reclaim following an EMA20 pullback. Both require a rising EMA stack, strength versus the benchmark and a benchmark above its EMA50.
- **Bounded entry plans:** preserve the structural stop and resistance-aware target. Reject next-session opening gaps outside the entry bounds or with reward/risk below 1.5. Replay signals expire after that session; recommendations do not submit an order.
- **Fixed risk:** default maximum initial risk is 0.25% of sizing capital, further limited by `RISK_PER_TRADE_INR`; maximum position notional is 10%. Zero-size positions are rejected. Probabilities carry `HEURISTIC_UNVALIDATED` status and do not drive Kelly sizing.
- **Daily replay:** next-session fills, cash funding, held-position risk, sector and correlation caps, conservative same-bar stops, transaction costs and 10/15-session time exits. Missing correlation evidence blocks an additional position when the correlation cap is active.

Reproduce the fixed comparison with `python scripts/compare_swing.py --snapshot artifacts/performance_validation/historical.pkl`. This uses the snapshot produced by `scripts/validate_performance.py`. Results are written under `artifacts/swing_comparison/`; see the [swing report](docs/swing-improvements-2026-10-02.md).

The rules had fewer trades and lower portfolio drawdown in this snapshot, but the previously viewed recent period lost money. This is an improvement to strategy definition and validation controls, with profitability still unproven. Earnings/event exclusions, point-in-time constituents and prospective paper evidence remain pending.

### Quarterly validation, attribution and sensitivity

Run `python scripts/research_swing.py` to generate expanding-history / nonoverlapping quarterly validation, plus a continuous portfolio across those quarters. Parameters remain fixed: prior data supplies causal indicator and regime context, with no fitting or parameter selection. Already viewed historical dates remain retrospective. The final partial quarter is explicitly flagged.

Reports include setup and entry-regime attribution, win rate, holding time, net-R distribution, stop frequency, exposure, CAGR, benchmark returns, beta, descriptive daily-regression alpha and information ratios. Exposure-normalized return ratios are descriptive and are not investable fully deployed returns. Holding-period trade excess returns and the portfolio exposure-matched benchmark both use documented daily-bar proxies. Portfolio decision CSVs separate sector, correlation, position, risk, cash and expiry constraints.

The prespecified development-only grid covers breakout lookbacks 15/20/25, relative volume 1.1/1.2/1.3, time exits 10/15/20, and separate 0.5/1.0/1.5 ATR stop variants alongside the structural default. All variants are reported, with no winner selected. See [the research report](docs/swing-research-2026-10-02.md).

### API runtime isolation

`server.create_app(paths=..., auto_scan=...)` creates an independent engine, persistence service, SSE broadcaster, cancellation event and rate limiter. SQLite opens during validated startup; API tests must enter the application lifespan. Killswitch read/reset failures keep scans blocked. Configuration and eligibility refactors preserve the trading rules. See [architecture fixes and instance setup](docs/architecture-fixes-2026-10-02.md).

### Independent V2 alpha experiments

`python scripts/research_v2.py --model SECTOR` runs the separate sector-leadership model; `--model PEAD` evaluates earnings surprises against consensus observed before the announcement. `--mode replay` uses the existing cash/risk book with next-session surveillance checks. Missing index, mapping, surveillance or earnings feeds produce explicit rejections. Delivery and regime gates are optional individual experiments; no new policy is activated in application scans. See [V2 commands and timestamped feed schemas](docs/swing-v2.md) and [the separate V2 protocol](docs/swing-v2-protocol.json). The current snapshot cannot establish V2 profitability without the required genuine data.

### Append-only prospective paper journal

Structural research is reproducible with `python scripts/research_structural_edge.py`: MAE/MFE bounds, liquid-universe RS deciles, leave-one-out sector strength, breadth percentages, volatility expansion and separate exit portfolios. The tested ATR trail, Chandelier and scale-out alternatives all lost more than the fixed policy in this snapshot; no new entry filter or exit is activated. See [structural research](docs/swing-structural-research-2026-10-02.md) and [as-of input formats / future protocol](docs/research-input-formats.md).

Swing scans record operational observations, including zero-signal days, and preserve each signal's first observed timestamp, parameter fingerprint, factor snapshot and entry plan. Confirmed paper fill/exit events append to the same SQLite journal atomically with the trade book. Historical rows cannot be updated, deleted or replaced through normal SQL operations.

Read with authenticated `GET /api/paper/events?after=0&limit=1000`; append new events to `state/paper_ledger.jsonl` with `python scripts/export_paper_ledger.py`. Export verifies the existing prefix and refuses to rewrite altered history. Use `source=SIMULATED`, the strategy ID and `signal_event_id` when recording paper fills. Supplied event timestamps remain separate from recording timestamps; backdated and future-dated entries are flagged. Fees/slippage and whether costs are observed or modeled are retained. The new journal starts empty and contains no fabricated historical evidence. [Paper workflow and event fields](docs/paper-trading-ledger.md).

---

## 📈 The Seven-Factor Composite Signal Model

Each candidate is evaluated across seven distinct factor dimensions (`core/factors.py`), smoothed using continuous curves to prevent boundary instability:

| Factor | Base Weight | Core Indicators | Smoothing & Mathematical Formulation |
| :--- | :--- | :--- | :--- |
| **Trend** | 22% | Supertrend, EMA(20, 50, 200), ADX | Multi-timeframe trend alignment. Continuous ADX ramp: $\le 18 \to 0.0$, $[18, 20] \to [0.0, 0.5]$, $[20, 25] \to [0.5, 1.0]$, $\ge 25 \to 1.0$. |
| **Momentum** | 22% | RSI, MACD Histogram, StochRSI | **LONG RSI**: $[48, 73] \to 1.00$ plateau; $[42, 45] \to 0.85$ pullback bonus; continuous ramps $[38, 42] \to [0.10, 0.85]$ and $[73, 78] \to [1.00, 0.20]$. Safe StochRSI band $[20, 80] \to 1.00$. |
| **Volume** | 10% | RVOL, POC, Value Area Low/High | Relative volume vs 20d mean. Hermite $C^1$ smoothstep transition across $[0.95, 1.10]$ from $0.00$ to $0.10$, linear scaling above $1.10$. |
| **Volatility** | 5% | ATR Ratio, BB Squeeze | Volatility compression identification (ATR $< 85\%$ of 50d mean) and Bollinger Band Squeeze bonus ($+0.15$). |
| **Relative Strength** | 22% | Sector Rank, Nifty50 RS | 60% sector RS rank within universe + 40% stock log-return differential vs. Nifty50 benchmark over 20-day lookback. |
| **Breakout** | 14% | 52w High (LONG) / 20d Low (SHORT), BB Width | Distance to 52-week high for LONG (continuation structure) vs. 20-day breakdown low for SHORT (swing breakdown structure), plus Bollinger Band Width contraction relative to 50d rolling mean. |
| **Quality** | 5% | 63d Momentum, Persistence | 3-month direction-aware log momentum, 20-day directional day persistence, and ATR expansion readiness. |

---

## 🔄 Two-Pass Cohort Scan & Cross-Sectional Ranking

Scoring occurs via a coordinated two-pass architecture (`core/services.py` & `core/scorer.py`):

### Pass 1: Independent Raw Scoring (`score_candidate_pass1`)
- Concurrently validates ADV share and turnover liquidity floors.
- Enforces strict data-quality gates (rejecting incomplete/NaN indicators).
- Determines directional bias (LONG / SHORT) via Supertrend, EMA-20, and intraday VWAP.
- Evaluates macro regime veto and structural EMA-200 alignment.
- Computes raw `FactorScores` across all 7 dimensions and returns intermediate `CandidateContext`.

### Cross-Sectional Cohort Percentile Ranking (`apply_cohort_factor_ranking`)
- Groups Pass 1 candidates into directional cohorts (`LONG` and `SHORT`).
- For cohorts meeting the sample threshold ($N \ge 10$), computes cross-sectional percentile ranks for each factor using average rank ties:
  $$\text{rank\_pct}(f_i) = \frac{\text{rank}(f_i) - 1.0}{N - 1.0} \in [0, 1]$$
- Blends raw factor scores with cohort ranks:
  $$f_{\text{blended}} = (1 - w_{\text{cohort}}) \times f_{\text{raw}} + w_{\text{cohort}} \times \text{rank\_pct}(f_i) \quad (w_{\text{cohort}} = 0.40)$$
- Recomputes composite score from the blended factor values.
- *Graceful Fallback*: Cohorts with $N < 10$ preserve 100% of raw factor scores without distortion.

### Pass 2: Probability Conversion, Hysteresis & Sizing (`score_candidate_pass2`)
- Applies intraday session multipliers and regime adjustments.
- Converts composite scores to win probabilities via Platt calibration.
- Enforces **Position Hysteresis**:
  - **New Candidate Hurdle**: $P(\text{win}) \ge 0.52$ (`MIN_PROB_WIN`).
  - **Open Position Hold Floor**: $P(\text{win}) \ge 0.47$ (`PROB_HOLD_FLOOR`) with `"HeldPos"` audit tag.
- Calculates True Volume Profile targets (POC, VAL, VAH) and fat-tail corrected Kelly sizing.

---

## 🛡️ Statistical IC Calibration & Bayesian Shrinkage

Factor weights dynamically adapt to forward market performance (`core/factors.py::calibrate_ic_weights`):

1. **Non-Overlapping Forward Horizon**: To eliminate serial correlation and artificially deflated standard error, sampling steps match the forward return horizon (`step = max(fwd_bars, 1) = 5`).
2. **Extended Estimation Window**: Lookback is extended to 120 bars, yielding $\approx 24$ independent cross-sectional IC observation periods.
3. **Bayesian Shrinkage toward Macro Prior**:
   $$w_{\text{shrunk}} = 0.70 \times w_{\text{empirical}} + 0.30 \times w_{\text{prior}}(\text{regime})$$
4. **Iterative Simplex Water-Filling with 5% Floor**: Exact iterative simplex projection guarantees every factor retains $w_f \ge 0.05$ while strictly enforcing $\sum w_f = 1.000$.

---

## 🛡️ Hardened Market Regime Detection

The **RegimeTracker** (`core/regime.py`) classifies macro market conditions into five states:

- 🟢 **TREND_UP**: Breadth $\ge 55\%$, high ADX. Priority on momentum and breakout setups.
- 🟡 **RANGE**: Mid-range breadth, low ADX. Priority on mean-reversion pullbacks.
- 🟠 **TREND_DOWN**: Breadth $< 45\%$, high ADX. Capital preservation or intraday short momentum.
- 🔵 **EXPANSION**: High ATR ratio, high ADX. Volatility expansion trades on both sides.
- 🔴 **PANIC**: Breadth $< 25\%$. All new long entries blocked; existing positions managed defensively.

### Regime Hardening Mechanisms:
- **Asymmetric PANIC Hysteresis**: Requires $35\%$ breadth to exit PANIC, but $25\%$ to enter.
- **Deadband Buffer**: $[45\%, 55\%]$ deadband between TREND and RANGE eliminates boundary whipsaws.
- **Regime Lock**: Suppresses regime tracker state changes during the opening 20-minute noise window.
- **Confidence Scaling**: Position size scales with $f(\text{ADX}, \text{ATR\_Ratio}, \text{Sector\_RS})$.

---

## 📐 Position Sizing & Risk Management

### Fat-Tail Corrected Kelly Sizing
Positions are sized via NAV-aware, kurtosis-corrected fractional Kelly criterion (`core/portfolio.py`):
- **Base Kelly Fraction**: $f^* = \frac{p(r + 1) - 1}{r}$
- **Kurtosis Penalty**: $\text{kurt\_corr} = \frac{3.0}{3.0 + \max(0, \text{excess\_kurtosis})}$
- **Dynamic Allocation**: $\text{Risk (INR)} = \text{Target Risk} \times f^* \times \text{kurt\_corr} \times \text{capital\_fraction}$

### Out-of-Sample Platt Probability Calibration
Composite factor scores are mapped to win probabilities using a logistic sigmoid (Option B convention):
$$P(\text{win}) = \frac{1}{1 + \exp(A \cdot \text{score} + B)}$$

- **Cold-Start Heuristic ($N < 80$ trades)**: Engine defaults to $A=-4.0, B=2.0$ ($\text{sigmoid}(4 \cdot \text{score} - 2)$), where $P \ge 0.52$ gates scores $\ge 0.505$.
- **Empirical Calibration ($N \ge 80$ trades)**: Once $\ge 80$ trade records accumulate in SQLite (`state/state.db`), maximum-likelihood estimation (MLE) fits parameters $A$ and $B$ on an out-of-sample validation slice (`IC_CALIB_OFFSET = 60`). The calibrated coefficients are saved to `state/platt_calibration.json` for live scans. Historical backtests use explicitly supplied coefficients and never load the latest live calibration, preventing future-outcome leakage. Paper outcomes are excluded from executed-trade calibration.

### Realistic Transaction Cost Friction
The **TransactionCostModel** (`core/backtest.py`) incorporates real-world execution drag into all backtests:
- Slippage: 8 bps per side (`SLIPPAGE_BPS = 8`)
- Brokerage: ₹20 flat per trade (`COMMISSION_INR = 20`)
- Regulatory STT, GST, and exchange fees
Net realized return $R_{\text{net}}$ accounts for total round-trip friction, preventing over-trading illusions.

---

## 📊 Walk-Forward Verification Smoke Test (Sample Validation)

Walk-forward backtest executed across the 504 NSE cash equity universe across rolling out-of-sample folds to verify pipeline integration, execution friction, and order simulation:

| Metric | Pre-Overhaul Baseline | Overhauled Engine (Phases 1–5) | Statistical Note ($n=10$) |
| :--- | :--- | :--- | :--- |
| **Out-of-Sample Trades** | 10 | 10 | Sample fold validation run |
| **Hit Rate (Target 1 Reached)** | 0.0% | **10.0%** | Reached T1 prior to time-stop or SL |
| **Win Rate (Net $R > 0$)** | 10.0% | **30.0%** | Includes profitable time-stop exits |
| **Mean Net Realized R** | -0.595 R | **+0.064 R** | 95% t-CI: [-0.45 R, +0.58 R] |
| **Total Realized Return** | -5.95 R | **+0.64 R** | Friction-adjusted net positive |
| **Trade-Series Sharpe** | -17.65 | **+0.91** | Discrete trade R return dispersion |
| **Profit Factor** | 0.07 | **1.16** | Sum of gains / sum of losses |
| **Max Drawdown** | -5.95 R | **-1.87 R** | Net drawdown over sample fold |
| **Edge Churn Protection** | None (whipsaws) | **Hysteresis Protected ($P \ge 0.47$)** | Eliminates marginal noise exits |

> ⚠️ **Smoke Test Disclaimer & Statistical Power ($n=10$)**:
> This table represents a mechanical smoke test verifying that time-stop exits, stop orders, slippage (8 bps), and commissions (₹20) execute as designed. **It is not an institutional statistical proof of trading edge.** With $n=10$, statistical power is limited (mean net $R = +0.064$, $95\%$ CI: $[-0.45\text{ R}, +0.58\text{ R}]$). Hit rate strictly tracks Target 1 hits; profitable time-stop exits contribute to realized return and Profit Factor without triggering Target 1. Full empirical edge validation requires larger longitudinal backtest datasets ($\ge 100+$ trades).
>
> ⚠️ **Survivorship Bias Disclosure**:
> Historical walk-forward tests evaluate historical bars for current `ALL_TICKERS` constituents. Companies that were delisted, merged, or removed from the index during historical periods are not captured, introducing survivorship bias. Institutional deployment requires point-in-time index constituent history.

---

## 🖥️ Command Center & Unified API Server

### Fill registration and closure

Recommendations never create or resize positions. Record confirmed fills with authenticated `POST /api/trades/fills`; provide a stable `trade_id`, ticker, direction, horizon, actual entry price/time, shares, stop, target, composite, probability and factor snapshot. Use `source=EXECUTED` for confirmed fills or `source=SIMULATED` for paper fills. This endpoint records an event and does not send a broker order. One active position per instrument is supported; partial fills and scaling require aggregation before registration.

Record a full exit with `POST /api/trades/{trade_id}/close`, supplying `exit_price`, timezone-aware `exit_ts`, total round-trip `costs` in INR, and `exit_reason`. Registration and closure atomically update the execution ledger and open-position book; identical retries are idempotent, and conflicting events are rejected. Confirmed fills require explicit exits; only paper fills are closed automatically from post-entry candles. Net P&L and realized R include recorded costs. Paper outcomes are excluded from empirical calibration.

`GET /api/positions` returns the held book; `GET /api/trades/executed` returns the execution ledger. All trade endpoints require `X-API-Key`. Alert deduplication records only successfully sent picks, so failed sends can be retried.

The engine includes an asynchronous FastAPI server (`server.py`) and dynamic monitoring dashboard (`dashboard.html`):

### Starting the Services
```bash
# Start the API server & web interface
python server.py

# Run a live or dry-run market scan
python run.py --no-telegram --force-score

# Run walk-forward backtest
python run.py --backtest --bt-train 120 --bt-test 20 --bt-step 20
```

### Key API Endpoints
- `GET /` — Interactive web dashboard (`dashboard.html`).
- `GET /api/status` — Market phase, active session, regime state, and lock status.
- `GET /api/scan` — Latest scan results, portfolio selections, and watchlist.
- `POST /api/scan/trigger` — Trigger fresh background market scan (`X-API-Key` required).
- `POST /api/regime/override` — Set/clear manual macro regime override (`X-API-Key` required).
- `GET /api/sectors` — Sector relative strength and concentration metrics.

---

## ⚙️ Configuration & Environment Reference

All parameters can be set in `.env` or passed via system environment variables:

| Variable | Default | Component | Description |
| :--- | :--- | :--- | :--- |
| `MIN_PROB_WIN` | `0.52` | Gating | Minimum win probability required for new position entry. |
| `PROB_HOLD_FLOOR` | `0.47` | Hysteresis | Minimum win probability floor required to retain existing open positions. |
| `COHORT_RANK_WEIGHT` | `0.40` | Scoring | Cross-sectional percentile weight in cohort ranking ($0.40 = 40\%$). |
| `COHORT_MIN_OBS` | `10` | Scoring | Minimum candidates in directional cohort to trigger cross-sectional ranking. |
| `IC_LOOKBACK_DAYS` | `120` | Calibration | Historical lookback bars used for factor IC estimation. |
| `IC_FORWARD_BARS` | `5` | Calibration | Forward return horizon for IC calculation. |
| `IC_CALIB_OFFSET` | `60` | Calibration | Held-out validation offset bars for out-of-sample Platt calibration. |
| `RISK_PER_TRADE_INR` | `5000.0` | Risk | Target risk allocation in INR per trade. |
| `PORTFOLIO_SIZE` | `6` | Portfolio | Maximum number of concurrent positions in optimized portfolio. |
| `MAX_SECTOR_PICKS` | `2` | Portfolio | Maximum ticker concentration allowed within a single sector. |
| `MAX_CORR` | `0.70` | Portfolio | Maximum pairwise correlation permitted between portfolio holdings. |
| `SLIPPAGE_BPS` | `8` | Execution | Slippage friction in basis points per side for backtesting. |
| `COMMISSION_INR` | `20` | Execution | Brokerage fee in INR per order execution. |
| `SWING_SETUP_ENABLED` | `false` | Swing | Opt-in versioned long-only daily research policy. |
| `SWING_MIN_RR` | `1.5` | Swing | Minimum reward/risk at the actual proposed fill. |
| `SWING_MAX_GAP_ATR` | `0.5` | Swing | Maximum entry extension above signal close, in ATR units. |
| `SWING_RISK_FRACTION` | `0.0025` | Swing | Maximum initial risk as a fraction of sizing capital. |
| `SWING_MAX_EXPOSURE_FRACTION` | `0.10` | Swing | Maximum per-position notional as a fraction of sizing capital. |

---

## 🛡️ Repository Verification & Quality Gates

### Larger chronological evaluation — 2 October 2026

The [performance report](docs/performance-validation-2026-10-02.md) evaluates five years of adjusted daily data for 60 stocks selected by sector rotation, plus the benchmark. Default heuristic calibration is fixed; training uses 120 sessions and test windows use 20 sessions without overlap. The last 252 sessions are held out, with monthly block bootstrap confidence intervals and 2×/3× slippage stress.

| Sample | Trades | Mean net R | Profit factor | 95% mean-R interval |
|---|---:|---:|---:|---|
| Development | 116 | -0.2319 | 0.633 | [-0.4585, +0.0043] |
| Final holdout | 34 | +0.1830 | 1.405 | [-0.3401, +0.6858] |
| Holdout, 2× slippage | 34 | +0.1493 | 1.320 | [-0.4054, +0.6453] |
| Holdout, 3× slippage | 34 | +0.1156 | 1.240 | [-0.4283, +0.6251] |

These results do **not** establish positive net expectancy: every interval includes losses, and the development sample has negative mean R. Current constituents introduce survivorship bias; fold-boundary entries do not reproduce daily portfolio execution. Point-in-time membership and prospective paper evidence remain necessary before a live profitability claim.

Reproduce with `python scripts/validate_performance.py`; use `--offline` to reuse its saved snapshot. CSVs, the historical snapshot and JSON metrics are saved under `artifacts/performance_validation/`.

The repository includes automated lint, type and regression checks:

```bash
# Install the pinned runtime and development dependencies in the project virtualenv
python -m pip install -r requirements-dev.txt

# Run the complete test suite (735 unit & integration tests)
python -m pytest tests/ -v

# Run configured code quality & lint verification
python -m ruff check core/ tests/ run.py screener_v14_modular.py server.py scripts/ tools/ utils/

# Run static type checking
python -m mypy core/ run.py screener_v14_modular.py server.py scripts/ tools/ utils/ --ignore-missing-imports --disallow-untyped-defs --warn-return-any --warn-unused-ignores
```

- **Static Analysis**: Ruff and Mypy pass with the repository's configured rules; Mypy strict mode is not enabled.
- **Regression Safety**: 735/735 tests passing with zero failures; 93.23% core coverage.

---

*Built for Quantitative Precision — Sovereign Engine v14.6-Modular*
