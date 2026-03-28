# Sovereign Engine

Sovereign Engine is a modular quantitative trading runtime for market scans,
regime-aware scoring, walk-forward backtests, and alert delivery.

The project is organized so source code, mutable runtime state, generated
artifacts, and logs live in separate places. That keeps the repo easier to
operate, cleaner to review, and safer to automate.

## What It Does

- Fetches market data through the provider layer in `core/data_provider.py`
- Computes indicators, factor scores, and market regime state
- Builds ranked candidates and optimized portfolios
- Supports walk-forward backtests and Platt calibration
- Sends alert summaries through the messaging layer

## Repo Layout

```text
core/         domain logic, service orchestration, scoring, regime, portfolio
tests/        maintained automated test suite
scripts/      diagnostics and operator helper scripts
state/        mutable runtime state
artifacts/    generated outputs such as backtests and exports
logs/         structured logs and provider diagnostics
run.py        canonical CLI entry point
```

## Runtime Boundaries

The current runtime is split around explicit services:

- `core/services.py` contains `PersistenceService`, `MarketDataService`,
  `AlertService`, and `ScanService`.
- `core/runtime_paths.py` centralizes `state/`, `artifacts/`, and `logs/`
  discovery.
- `screener_v14_modular.py` is a thin compatibility wrapper over the service
  layer.
- `run.py` is the supported CLI surface for scans, watch mode, calibration,
  and backtests.

## Quick Start

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and fill in the credentials you use.

3. Run a single scan.

```bash
python run.py
```

## Environment Variables

The engine supports a small runtime-path contract in addition to provider and
Telegram credentials.

```env
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

FYERS_CLIENT_ID=your_client_id_here
FYERS_SECRET_KEY=your_secret_key_here
FYERS_ACCESS_TOKEN=your_access_token_here
FYERS_REDIRECT_URI=your_redirect_uri_here

STATE_DIR=state
ARTIFACTS_DIR=artifacts
LOG_DIR=logs

WEIGHTS_PATH=state/factor_weights.json
TRADE_LOG_PATH=state/trade_log.json
PLATT_CALIB_PATH=state/platt_calibration.json
TELEMETRY_LOG_PATH=logs/sovereign.jsonl
DEGRADATION_LOG_PATH=logs/data_provider_degradation.log
```

If you leave the path variables unset, those defaults are used automatically.

## Common Commands

Single scan:

```bash
python run.py
```

Watch mode:

```bash
python run.py --watch 15
```

Regime override:

```bash
python run.py --regime-override TREND_UP
```

Calibration from the runtime trade log:

```bash
python run.py --calibrate
```

Walk-forward backtest:

```bash
python run.py --backtest --bt-train 120 --bt-test 20 --bt-step 10 --bt-out backtest_results.csv
```

Notes:

- Relative `--bt-out` paths are written under `artifacts/`.
- Runtime state is loaded from `state/` and still falls back to legacy root
  files when present.
- Structured telemetry is written to `logs/sovereign.jsonl`.

## Quality Gates

Run the maintained validation surface with:

```bash
pytest tests/ -v --cov=core --cov-report=term-missing
ruff check core/ tests/ run.py screener_v14_modular.py sovereign_improvements.py
mypy core/ run.py screener_v14_modular.py sovereign_improvements.py \
  --ignore-missing-imports \
  --disallow-untyped-defs \
  --warn-return-any \
  --warn-unused-ignores
```

The current maintained suite is green with `420` passing tests.

## Diagnostics

Operator and provider diagnostics live in `scripts/`. These are intentionally
kept outside `tests/` so CI only runs the maintained automated surface.

Examples:

- `scripts/fyers_setup.py`
- `scripts/test_alert.py`
- `scripts/test_icir.py`
- `scripts/test_yf_diagnostic.py`

## Outputs and Persistence

Typical runtime files now land in these locations:

- `state/platt_calibration.json`
- `state/trade_log.json`
- `state/factor_weights.json`
- `artifacts/backtest_results.csv`
- `logs/sovereign.jsonl`
- `logs/data_provider_degradation.log`

This separation is intentional: source stays versioned, state stays mutable,
and generated outputs are easy to inspect or clean up without touching code.
