# Sovereign Engine v14.6-Modular 🏛️

[![Security Scan](https://img.shields.io/badge/Security-Verified-success?style=flat-square)](#)
[![Lint Compliance](https://img.shields.io/badge/Lint-Strict.Ruff-blue?style=flat-square)](#)
[![Test Suite](https://img.shields.io/badge/Tests-459%20Passed-success?style=flat-square)](#)
[![Version](https://img.shields.io/badge/Version-14.6--Modular-indigo?style=flat-square)](#)

**Institutional-Grade Quantitative Trading Intelligence for NSE India.**

Sovereign Engine is a modular, high-performance quantitative runtime designed for systematic market analysis, regime-aware scoring, and automated risk management. Built for professional traders and quantitative analysts, it provides a robust pipeline from raw market data to optimized portfolio candidate selection.

---

## 🛡️ Core Architecture & Resilience

The engine is built on a "Source-State-Log" (SSL) boundary model to ensure operational safety and clean version control.

- **Dual-Provider Fallback**: A resilient data chain that prioritizes **Fyers API (v3)** for low-latency intraday data and falls back to **yfinance** for global coverage and historical backfills.
- **Circuit Breakers & Backoff**: Implements exponential backoff with full jitter for API resilience. Consecutive failures trigger a circuit-breaker pause to prevent account throttling.
- **Structured Telemetry**: Every internal event is serialized to `logs/sovereign.jsonl` in structured JSON, facilitating audit trails and post-trade performance review.

---

## 📈 The Seven-Factor Signal Model

The heart of the engine is a composite scoring model (`core/factors.py`) that evaluates each candidate across seven non-correlated dimensions:

| Factor | Weight | Components | Logic Description |
| :--- | :--- | :--- | :--- |
| **Trend** | 28% | Supertrend, EMA(20, 50, 200) | Multi-timeframe trend alignment with MTF-60m confirmation. |
| **Momentum** | 20% | RSI, MACD, StochRSI, ADX | RSI zone tracking (48-73 ideal) with MACD acceleration. |
| **Volume** | 18% | RVOL, POC, Value Area | Relative volume vs 20d mean and price proximity to the **POC**. |
| **Volatility** | 12% | ATR, BB Squeeze | Identifies coiling phases (ATR < 85% of mean) and squeeze events. |
| **Relative Strength** | 12% | Sector Rank, Nifty50 RS | Log-return differential scoring vs the benchmark and sector peers. |
| **Breakout** | 6% | 52w High, BB Width | Proximity to 52-week horizontal levels and narrow-band coiling. |
| **Quality** | 4% | 63d Momentum, Persistence | 3-month momentum stability and close-over-open directional win-rate. |

---

## 🛡️ Hardened Market Regime Detection

The **RegimeTracker** (`core/regime.py`) classifies the broad market into one of five states:
- 🟢 **TREND_UP**: Breadth ≥ 55%, high ADX. Momentum/Breakout strategies priority.
- 🟡 **RANGE**: Mid-range breadth, low ADX. Mean-reversion priority.
- 🟠 **TREND_DOWN**: Breadth < 45%, high ADX. Capital preservation or Short momentum.
- 🔵 **EXPANSION**: High ATR, high ADX. Volatility breakout both sides.
- 🔴 **PANIC**: Breadth < `REGIME_BREADTH_PANIC` (0.25). All new longs blocked.

### Hardening Fixes (v14):
1.  **Hysteresis**: Require 35% breadth to leave PANIC, but only 25% to enter.
2.  **Deadband**: A 10% breadth "no-man's land" between TREND and RANGE to eliminate whipsaw.
3.  **Regime Lock**: Suppresses regime changes during the opening 15-minute noise window.
4.  **Confidence Scaling**: Confidence = `f(ADX, ATR_Ratio, Sector_RS)`.

---

## 📐 The Quantitative Math

### Professional Position Sizing (Kelly Criterion)
The engine uses a NAV-aware, fat-tail corrected Kelly sizing formula:
- **Base Kelly**: `f* = (p * (rr + 1) - 1) / rr`
- **Fat-Tail Correction**: `kurt_corr = 3.0 / (3.0 + excess_kurtosis)`
- **Dynamic Sizing**: `Risk = par_risk * f* * kurt_corr * capital_fraction`

### Platt Scaling & Calibration
Composite scores in `[0, 1]` are mapped to win-probabilities via the **Platt Sigmoid**:
- `P(win) = 1 / (1 + exp(A * score + B))`
- Parameters **A** and **B** are calibrated via maximum-likelihood estimation (MLE) on an **out-of-sample** validation window (default: 60 bars). This separates the fitting noise from predicted probabilities, preventing over-confident sizing in regimes with high kurtosis.

### IC-Weighted Factor Recalibration
Factor weights are not static. The engine periodically re-calculates the **Information Coefficient (Spearman)** to optimize weight distributions:
- `IC = SpearmanCorrelation(Factor_Scores, Signed_Forward_Returns)`
- `Weight = max(0, ICIR) / Σ(pos_ICIR)`

### Transaction Cost Modeling
The **TransactionCostModel** (`core/backtest.py`) enables testing strategies under realistic market friction conditions. It supports configurable slippage, brokerage fees, and tax implications, generating a net realized risk-reward profile (friction-adjusted Return vs. Gross Return). This parameter is seamlessly injected into the `walk_forward` execution context, ensuring institutional-grade resilience against idealized backtesting illusions.

### Modular Runtime Collaborators
Sovereign Engine v14.6 centralizes critical runtime decision-makers in `core/runtime_components.py` to ensure state consistency across asynchronous processes:
- **RegimeProbabilityGate**: Enforces trade blocking during PANIC or low-confidence regimes.
- **TieredCapitalScaler**: Dynamic position scaling based on real-time NAV and drawdown thresholds.
- **RollingFactorCalibrator**: Manages periodic weight updates and Platt parameter fitting.
- **RegimeAwareTelegramAlerter**: Context-sensitive notifications that adapt to market state.

### Modular Service Architecture & Orchestration
The engine employs a lazily-initialized `ServiceBundle` (`core/services.py`) to orchestrate high-level workloads:
- **ScanService**: The primary entry point for market scanning and signal generation.
- **DataService**: High-performance async chain with Fyers/yfinance fallbacks and cache management.
- **PersistenceService**: Scoped artifact and state management across `state/` and `artifacts/`.
- **AlertService**: Decoupled, multi-provider notification handlers.

> [!TIP]
> The `ServiceBundle` allows components to be gracefully swapped or monkey-patched during testing. The entry point `run.py` leverages this architecture to decouple logic from the execution environment.

---

## 🛡️ Repo Stability & Master Validation

To ensure institutional-grade code quality and repository stability, every PR/commit is subjected to a tiered validation protocol via `checklist.py`.

### The P0/P1 Check Hierarchy:
1.  **P0: Security Scan** - Automated vulnerability and secret detection.
2.  **P0: Lint & Type Compliance** - Strict **Ruff** (E402/F821) and **Mypy** verification.
3.  **P1: Test Suite Compliance** - **459 PASSED** unit and integration tests (100% pass requirement).
4.  **P1: UX & SEO Optimization** - Accessibility and Meta-tag validation for reporting artifacts.

---

## 🖥️ Unified Command Center & API Server

The **Sovereign Engine API Server** (`server.py`) and **NSE Unified Scanner** (`dashboard.html`) provide a localized, professional-grade interface for monitoring market conditions and trade execution plans.

### Starting the API Server
```bash
python server.py
```
This launches a FastAPI server on `http://127.0.0.1:8000` with background scanning capabilities and serves the dynamic web dashboard.

> [!IMPORTANT]
> The API Server now requires authentication. You must provide an `X-API-Key` header with the value defined in `API_KEY` (default: `gr_sovereign_local_secret`) for all state-mutating endpoints (`POST`).

### Available REST API Endpoints
- **`GET /`** - Serve dynamic `dashboard.html`.
- **`GET /api/status`** - Engine status, market session, current regime & lock state.
- **`GET /api/scan`** - Cached/latest scan results (portfolio picks & candidates).
- **`POST /api/scan/trigger`** (Requires Auth) - Asynchronously trigger a fresh market scan.
- **`POST /api/regime/override`** (Requires Auth) - Set or clear manual market regime override (`PANIC`, `TREND_UP`, etc.).
- **`GET /api/sectors`** - Sector relative strength & concentration breakdown.
- **`GET /api/config`** - System configuration settings.

### Dashboard Features
- **33 EMA Signal Intelligence**: Real-time signal validation using a 9-minute spot chart timeframe with MTF-60m confirmation.
- **8-Param Momentum Screener**: A binary filter-set evaluating 4 Technical and 4 Fundamental parameters with zero-tolerance pass logic.
- **Trade Execution Planner**: Quantitative risk-reward calculator with ATR-based volatility scaling and volatility-matched bias detection.
- **Institutional SEO**: Fully optimized for internal reporting with high-fidelity `og:meta` headers.


---

## 📁 Environment Variables Guide

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | — | Token for the alert delivery layer. |
| `FYERS_ACCESS_TOKEN` | — | Daily token for high-priority market data, refreshed by `tools/scripts/fyers_setup.py`. |
| `RISK_PER_TRADE_INR` | `10000` | Target risk in Rs. per trade. |
| `PORTFOLIO_SIZE` | `5` | Maximum candidates to select in optimized portfolio. |
| `MAX_SECTOR_PICKS` | `2` | Maximum tickers from the same sector. |
| `MIN_PROB_WIN` | `0.52` | Minimum Platt-scaled probability to qualify. |
| `REGIME_ADX_TREND` | `25.0` | ADX threshold to define a trending regime. |
| `PORTFOLIO_STATE_PATH` | `state/portfolio_state.json` | JSON path for live NAV and peak tracking. |

---

## ⌨️ Advanced Operations

### Walk-Forward Backtesting
Execute a walk-forward analysis with a training-test split:
```bash
python run.py --backtest --bt-train 120 --bt-test 20 --bt-step 10
```

### Institutional Watch Mode
Run high-frequency scans with exponential backoff and Telegram state updates:
```bash
python run.py --watch 15
```

### Live Capital Scaling
To enable dynamic position sizing that responds to portfolio performance between scans, update the `state/portfolio_state.json` file. The engine reads this file at the start of every scan cycle:

```json
{
  "current_nav": 875000,
  "peak_nav": 1000000
}
```
- **current_nav**: Current portfolio value in INR. Scales the Kelly fraction proportionally.
- **peak_nav**: Historical peak value. Used for **Tiered PANIC** drawdown calculation.

### Diagnostic & Validation Scripts
- `python .agent/scripts/checklist.py .`: The definitive master validation source (Security, Lint, Tests, SEO).
- `pytest tests/test_services.py`: Comprehensive service-layer verification (100% pass required).
- `tools/scripts/fyers_setup.py`: Daily token refreshment, account verification, and `.env` synchronization.
- `tools/scripts/test_yf_diagnostic.py`: Integrity check for yfinance connectivity and data ingestion health.
- `tools/scripts/test_icir.py`: Real-time audit of cumulative factor Information Coefficients.

---

- **v14.6-Modularized Runtime**: Migrated all collaborators into a first-class `core/` module hierarchy.
- **Out-of-Sample Platt Calibration**: Integrated 60-bar validation window for institutional probability stability.
- **Service-Level Test Suite**: 100% pass confirmed for all decoupled services via `tests/test_services.py`.
- **Security Hardening**: Implemented X-API-Key auth, disabled CORS credentials, protected state with thread-locks, and sanitized XSS in `dashboard.html`.
- **Repository Hygiene**: Separated dev dependencies into `requirements-dev.txt` and migrated legacy utility scripts into the `tools/` directory.

> [!IMPORTANT]
> The Modular runtime architecture is now the primary path. Ensure any custom factor implementations utilize the `ServiceBundle` for state persistence and regime-aware logic.

*Built for Quantitative Precision — Sovereign Engine v14.6-Modular*

