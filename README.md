# Sovereign Engine v14.4-Modular

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
- Parameters **A** and **B** are calibrated via maximum-likelihood estimation (MLE) on out-of-sample trade history to prevent over-confidence.

### IC-Weighted Factor Recalibration
Factor weights are not static. The engine periodically re-calculates the **Information Coefficient (Spearman)** to optimize weight distributions:
- `IC = SpearmanCorrelation(Factor_Scores, Signed_Forward_Returns)`
- `Weight = max(0, ICIR) / Σ(pos_ICIR)`

---

## 📁 Environment Variables Guide

| Variable | Default | Description |
| :--- | :--- | :--- |
| `TELEGRAM_BOT_TOKEN` | — | Token for the alert delivery layer. |
| `FYERS_ACCESS_TOKEN` | — | Daily token for high-priority market data. |
| `RISK_PER_TRADE_INR` | `10000` | Target risk in Rs. per trade. |
| `PORTFOLIO_SIZE` | `5` | Maximum candidates to select in optimized portfolio. |
| `MAX_SECTOR_PICKS` | `2` | Maximum tickers from the same sector. |
| `MIN_PROB_WIN` | `0.52` | Minimum Platt-scaled probability to qualify. |
| `REGIME_ADX_TREND` | `25.0` | ADX threshold to define a trending regime. |

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

### Diagnostic Scripts
- `scripts/fyers_setup.py`: Daily token refreshment and account verification.
- `scripts/test_yf_diagnostic.py`: Verify yfinance connectivity and data health.
- `scripts/test_icir.py`: Audit current factor Information Coefficients.

---
*Built for Quantitative Precision — Sovereign Engine v14.4-Modular*
