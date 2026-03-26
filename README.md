# 🦅 Sovereign Engine v14.4-Modular: Institutional State Isolation

Sovereign Engine is a professional-grade quantitative trading architecture designed for high-fidelity scanning, probabilistic setup evaluation, and automated portfolio optimization. 

> [!IMPORTANT]
> **v14.4 Architectural Refinement**: This version introduces **ScanState-scoped isolation** (eliminating state bleed), **Direction-Aware IC Calibration** (removing lookahead bias), and **Institutional CI/CD** (80% coverage gate).

---

## 🌌 System Architecture

The project features a high-performance modular architecture where each component is an isolated logic bucket, optimized for concurrency and resilience:

```mermaid
graph TD
    A[Market Data Providers <br/> Fyers v3 / yfinance] --> B[Modular Entry <br/> screener_v14_modular.py]
    B --> C[Core Infrastructure <br/> /core]
    
    subgraph "Core Modules"
        C --> D[Data & Config <br/> config, data_provider, universe]
        C --> E[Alpha Engine <br/> indicators, regime, factors]
        C --> F[Optimisation <br/> scorer, portfolio, cache]
    end
    
    subgraph "Institutional Grade Pillars"
        C --> H[Testing & QA <br/> 100+ tests, 80% coverage]
        C --> I[Telemetry & Metrics <br/> structured JSON logging]
        C --> J[Resilience Layer <br/> circuit breakers & retries]
    end
    
    subgraph "CI/CD & Automation"
        K[GitHub Actions] --> L[Lint & Test Gate]
        L --> M[Main Branch Protection]
    end
    
    B --> G[Persistence <br/> platt_calibration.json]
    G --> B
```

---

## 🧠 Core Methodology

### 1. Unified 7-Factor Model (v14.0)
The engine evaluates the Nifty 200 universe using 7 orthogonal factors, fully implemented in `core/factors.py` with zero placeholders:
- **Trend (28%)**: Multi-timeframe EMA alignment + vectorised Supertrend.
- **Momentum (20%)**: RSI-Wilder, MACD Histogram acceleration, and directional streaks.
- **Volume (18%)**: RVOL, POC proximity, and Value Area (VAH/VAL) positioning.
- **Volatility (12%)**: ATR coiling (relative to 50d mean) + Bollinger Band Squeeze.
- **Relative Strength (12%)**: Sector-relative performance vs Nifty 50 Benchmark.
- **Breakout (6%)**: 52-week high proximity and BB-Width consolidation.
- **Quality (4%)**: 63-day log momentum, directional persistence, and ATR expansion.

### 2. Probabilistic Framework (Platt Scaling)
Every setup is transformed into a **Win Probability P(Win)** using calibrated **Platt scaling**. The v14.0 system automatically loads and saves `platt_calibration.json` to ensure consistent execution across restarts and backtests.

### 3. Institutional Risk Management
- **CapitalScaler (FIX 2)**: NAV-aware position sizing. Use `par_nav` to scale risk proportional to live portfolio value.
- **Fat-Tail Kelly Sizing**: Corrected for excess kurtosis to prevent over-leverage in volatile names.
- **Correlation Gate**: Optimized portfolio selection using a `MAX_CORR` filter (0.70 default).
- **Regime Breadth Veto**: Automatic trading suspension (PANIC mode) when market breadth falls below thresholds.

---

## 🏛️ Audit & Quality — **Score: 8.2 / 10**

The system underwent a professional audit in March 2026, achieving a "Production-Grade" rating.

| Category | Score | Highlights |
|:---|:---:|:---|
| Architecture | **9.0** | Clean ScanState isolation, zero monolith reliance. |
| Resilience | **9.0** | Multi-service Circuit Breakers (Fyers/yfinance/Telegram). |
| Methodology | **8.5** | Directional ICIR weights, out-of-sample Platt fitting. |
| Testing | **9.0** | **100+ unit tests** with automated 80% coverage gate. |

---

## 📦 Institutional Modules (`/core`)

| Module | Description |
| :--- | :--- |
| `backtest.py` | **Walk-Forward Engine**: Multi-fold simulation with Sharpe, MaxDD, and Hit-Rate metrics. |
| `telemetry.py` | **Structured Logging**: Emits JSON-line metrics for log aggregators (ELK/Loki compatible). |
| `retry.py` | **Resilience Layer**: Circuit breakers and exponential backoff for APIs. |
| `indicators.py` | **Vectorised Math**: Low-latency technical indicators (Supertrend, ADX, StochRSI). |
| `regime.py` | **Regime Tracker**: 5-state classification with confirmation-lag protection. |
| `portfolio.py` | **Optimizer**: Correlation-aware selection and **NAV-aware scaling**. |

---

## 🖥️ Operations

### Quick Start (Windows)
Double-click **`run_watch.bat`** to launch the engine in 15-minute Watch Mode.

### Command Line Interface
- **Production Scan**: `python screener_v14_modular.py`
- **Watch Mode**: `python screener_v14_modular.py --watch 15`
- **Backtest**: `python screener_v14_modular.py --backtest --days 180`
- **Calibration**: `python screener_v14_modular.py --calibrate` 
- **Tests**: `pytest tests/ -v --cov=core --cov-report=term-missing`

---

## ⚙️ CI/CD & Testing
The engine uses **GitHub Actions** to enforce institutional quality:
- **Automated Tests**: Every push/PR triggers 100+ tests on Python 3.10-3.12.
- **Quality Gate**: Build fails if code coverage drops below **80%**.
- **Linting**: Enforces **Ruff** standards for clean, performant code.

**Test Suites**:
- `tests/test_indicators.py`: 47 tests for technical math.
- `tests/test_regime.py`: 51 tests for market state logic.
- `tests/test_factors.py`: Alpha factor validation.
- `tests/test_sovereign_core.py`: Portfolio and risk engine.

---

**Disclaimer**: *Sovereign Engine is a high-performance quantitative tool. All estimates are probabilistic. Trade responsibly.*
