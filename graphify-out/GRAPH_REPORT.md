# Graph Report - D:\Tradeidesa\grclaudescript  (2026-08-03)

## Corpus Check
- Corpus is ~49,177 words - fits in a single context window. You may not need a graph.

## Summary
- 1273 nodes · 3315 edges · 84 communities (78 shown, 6 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 416 edges (avg confidence: 0.51)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Async Data
- Runtime Components
- Portfolio
- Regime
- Backtest
- Services
- Regime
- Regime
- Scorer Portfolio Tests
- Backtest
- Regime
- Portfolio
- Cache
- Regime
- Backtest
- Config
- Factors
- Scorer
- Factors
- Retry
- Retry
- Factors Tests
- Portfolio
- Portfolio
- Regime
- Regime
- Regime
- Regime Tests
- Data Provider
- Telemetry
- Indicators Tests
- Backtest
- Indicators
- Factors
- Factors
- Fyers Setup
- Transaction Costs Tests
- Backtest
- Telemetry
- Phase34 Tests
- Regime Tests
- Factors
- Scorer
- Indicators Tests
- Backtest
- Factors
- Scorer
- Indicators Tests
- Factors
- Factors Tests
- Phase34 Tests
- Backtest
- Retry
- Indicators Tests
- Indicators Tests
- Config
- Indicators Tests
- Indicators Tests
- Indicators Tests
- Indicators Tests
- Indicators Tests
- Indicators Tests
- Retry
- Config
- Retry
- Dashboard
- Backtest
- Cache
- Indicators Tests
- Sovereign Core Tests
- Ci
- Pyproject

## God Nodes (most connected - your core abstractions)
1. `SystemConfig` - 150 edges
2. `RegimeTracker` - 110 edges
3. `MarketRegime` - 81 edges
4. `MarketRegimeType` - 54 edges
5. `_make_config()` - 48 edges
6. `TickerResult` - 46 edges
7. `ScanCache` - 44 edges
8. `classify_regime()` - 44 edges
9. `PersistenceService` - 44 edges
10. `FactorScores` - 40 edges

## Surprising Connections (you probably didn't know these)
- `Eight-Parameter Screener` --semantically_similar_to--> `Seven-Factor Signal Model`  [INFERRED] [semantically similar]
  dashboard.html → README.md
- `TestRealisedRWithCosts` --uses--> `TransactionCostModel`  [INFERRED]
  tests/test_transaction_costs.py → core/backtest.py
- `TestTransactionCostModel` --uses--> `TransactionCostModel`  [INFERRED]
  tests/test_transaction_costs.py → core/backtest.py
- `TestBacktestHelpers` --uses--> `TradeRecord`  [INFERRED]
  tests/test_phase34.py → core/backtest.py
- `TestCircuitBreaker` --uses--> `TradeRecord`  [INFERRED]
  tests/test_phase34.py → core/backtest.py

## Import Cycles
- None detected.

## Communities (84 total, 6 thin omitted)

### Community 0 - "Async Data"
Cohesion: 0.06
Nodes (47): asyncio, async_fetch_daily_batch(), _async_fetch_fyers(), _async_fetch_yfinance_chunk(), fetch_daily_batch_async(), DataFrame, core/async_data.py ================== Async data fetching layer for the…, Fetch one yfinance chunk inside an asyncio semaphore. Returns a partial… (+39 more)

### Community 1 - "Runtime Components"
Cohesion: 0.06
Nodes (33): create_runtime_components(), Any, Path, core/runtime_components.py ========================== Core-native runtime…, Immutable container of live runtime collaborators. Supports the context-manager…, Stop the background calibrator thread., _regime_str(), RegimeAwareTelegramAlerter (+25 more)

### Community 2 - "Portfolio"
Cohesion: 0.08
Nodes (18): compute_targets(), optimize_portfolio(), ATR-based stop, T1, T2 and reward:risk ratio. Uses config.STOP_ATR_MULT,…, Select up to config.PORTFOLIO_SIZE tickers from candidates, subject to: 1.…, MockResult, _make_ticker_result(), Minimal TickerResult for portfolio optimizer tests., TestComputeTargets (+10 more)

### Community 3 - "Regime"
Cohesion: 0.07
Nodes (24): compute_rs(), Log-return relative-strength of *stock* vs *bench* over *lookback* bars.…, passes_liquidity(), _make_price_series(), Series, TestComputeRS, _make_daily_df(), _make_price_series() (+16 more)

### Community 4 - "Backtest"
Cohesion: 0.09
Nodes (30): core/backtest.py ================ Walk-forward backtesting framework for the…, Enum, core/config.py ============== Centralised, immutable system parameters for…, core/portfolio.py ================= Position sizing, trade targets, and…, core/regime.py ============== Market regime detection for Sovereign Engine v14.…, core/scorer.py ============== Individual ticker evaluation — v14 modular…, TickerResult, configure_services() (+22 more)

### Community 5 - "Services"
Cohesion: 0.08
Nodes (35): PersistenceService, Path, _FakePersistence, _make_trade_log(), _paths(), DataFrame, save_platt writes valid JSON; a fresh service loads it back exactly., When no calibration file exists, config defaults are returned. (+27 more)

### Community 6 - "Regime"
Cohesion: 0.11
Nodes (14): Strength-weighted confidence score in [0, 1]. sector_conc (Fix 7): Fraction of…, _regime_confidence(), _make_config(), Fix 3: EXPANSION confidence now scales with ADX and ATR — no longer flat., ADX=45 + ATR=2.0 should beat ADX=26 + ATR=1.4 for EXPANSION., Verify the config helper correctly identifies the opening noise window., Setting REGIME_LOCK_MINUTES=0 disables the feature entirely., Configurable window — should work for any REGIME_LOCK_MINUTES value. (+6 more)

### Community 7 - "Regime"
Cohesion: 0.10
Nodes (18): Ring buffer of recent regime classifications. Uses ``deque(maxlen)`` for O(1)…, RegimeTracker, create_mock_processed(), test_classify_regime_panic_hysteresis(), test_classify_regime_panic_threshold(), test_classify_regime_trend_deadband(), test_regime_lock_suppresses_updates(), test_regime_tracker_confirmation_logic() (+10 more)

### Community 8 - "Scorer Portfolio Tests"
Cohesion: 0.15
Nodes (6): _make_ohlcv(), Each test isolates one gate or behaviour by adjusting exactly the field that…, Synthetic OHLCV + full indicator set required by score_ticker. All columns use…, When USE_VALUE_AREA_RR=True and VAH gives better RR, T1 should move to VAH., _regime(), TestScoreTicker

### Community 9 - "Backtest"
Cohesion: 0.21
Nodes (21): OverallStats, Aggregated statistics across all folds., Container returned by ``walk_forward``., Round-trip friction for NSE intraday trades, expressed as percentages of the…, TransactionCostModel, WalkForwardResult, MarketRegimeType, CapitalFractionScaler (+13 more)

### Community 10 - "Regime"
Cohesion: 0.12
Nodes (14): classify_regime(), Classify the current market regime from cross-sectional indicator data.…, Most recent regime, or ``None`` if history is empty., Breadth recorded at the most recent push, or 0.0 if empty., Verify that classify_regime(locked=True) suppresses tracker.push() so a single…, tracker._history must be unchanged after a locked call., When locked, the returned regime matches the tracker's last push., If the tracker has no history yet, locked must still return something sensible. (+6 more)

### Community 11 - "Portfolio"
Cohesion: 0.12
Nodes (9): calculate_kelly_size(), Fat-tail Kelly position sizing, NAV-scaled via ``capital_fraction``. Returns…, TestCalculateKellySize, calculate_kelly_size(entry, stop, prob_win, rr, daily_df, config), entry == stop → no risk per share → zero position., Even a tiny Kelly fraction should return at least MIN_SHARES., Wider stop → higher risk per share → fewer shares for same risk., PANIC regime → capital_fraction = 0 → no new positions. (+1 more)

### Community 12 - "Cache"
Cohesion: 0.14
Nodes (13): _build_corr_matrix(), _df_cache_key(), DataFrame, core/cache.py ============= LRU caching for computationally expensive,…, Return ``(POC, VAL, VAH)`` for *df*, using cache when possible. Cache key:…, Build (or return cached) pairwise return correlation matrix. The matrix is…, Drop all cached values — call between scan cycles if reusing., Compute pairwise daily-return correlation from *processed* data. Same logic as… (+5 more)

### Community 13 - "Regime"
Cohesion: 0.14
Nodes (12): compute_breadth(), compute_sector_rs(), DataFrame, Series, _make_daily_df(), _make_processed(), DataFrame, fixture (+4 more)

### Community 14 - "Backtest"
Cohesion: 0.13
Nodes (17): Single simulated trade from the backtest., TradeRecord, parametrize, _bars(), _make_ohlcv(), DataFrame, tests/test_transaction_costs.py ================================ Unit tests for…, With ZERO_COST_MODEL the net_r == gross_r. (+9 more)

### Community 15 - "Config"
Cohesion: 0.17
Nodes (8): SystemConfig, passes_static_filters(), Any, DataFrame, Check ADV and structural filters before spending scorer time., Mutable scan-session state. This is intentionally scoped to one scan cycle. The…, ScanService, ScanState

### Community 16 - "Factors"
Cohesion: 0.19
Nodes (7): factor_trend(), factor_volatility(), Multi-timeframe trend factor [0, 1]. Components: 40% — Supertrend direction 15%…, Volatility/coiling factor [0, 1]. High score = ATR contracting relative to 50d…, ema20 > ema50 > ema200 yields extra +0.05 bonus., TestFactorTrend, TestFactorVolatility

### Community 17 - "Scorer"
Cohesion: 0.12
Nodes (11): calibrate_platt(), composite_to_prob(), Sigmoid probability from composite score using Platt A/B params., Fit Platt A and B parameters on a **held-out** validation window. FIX 3 — out-…, TestCalibratePlatt, TestCompositeToProb, v14 raises ValueError when samples < calib_offset + 20., v14 sigmoid convention: prob is monotone across composites. (+3 more)

### Community 18 - "Factors"
Cohesion: 0.19
Nodes (13): factor_quality(), Price-momentum quality factor [0, 1] — Fix C from v12. Replaces the dead…, _make_bench(), _make_ohlcv(), _make_row(), DataFrame, Series, tests/test_factors.py ===================== Unit tests for core/factors.py… (+5 more)

### Community 19 - "Retry"
Cohesion: 0.18
Nodes (13): BreakerState, CircuitBreakerOpen, guarded_call(), MaxRetriesExceeded, Enum, core/retry.py ============= Retry logic with exponential back-off and a…, Raised when a call is rejected because the breaker is OPEN., Execute *fn* with retry + circuit breaker protection. The retry loop runs… (+5 more)

### Community 20 - "Retry"
Cohesion: 0.15
Nodes (5): CircuitBreaker, Per-resource circuit breaker with CLOSED → OPEN → HALF-OPEN state machine.…, Manually record a failure (e.g. from an async path)., Force-reset to CLOSED (useful in tests or after maintenance)., TestCircuitBreaker

### Community 21 - "Factors Tests"
Cohesion: 0.16
Nodes (3): TestComputeFactors, TestFactorBreakout, TestFactorRelativeStrength

### Community 22 - "Portfolio"
Cohesion: 0.16
Nodes (6): CapitalScaler, Record the current live portfolio value (Rs.)., Return live_nav / par_nav, clipped to [min_fraction, max_fraction]. A value of…, Construct a CapitalScaler whose par NAV is implied by the config. Uses…, NAV-aware scaling factor for Kelly position sizing. Attributes ----------…, TestCapitalScaler

### Community 23 - "Portfolio"
Cohesion: 0.19
Nodes (9): DataFrame, Excess kurtosis of daily returns over the configured lookback window. Falls…, _ticker_excess_kurtosis(), TradeTargets, _config(), tests/test_scorer_portfolio.py ============================== Unit +…, Real SystemConfig with sane defaults; override specific fields via kwargs., TestExcessKurtosis (+1 more)

### Community 24 - "Regime"
Cohesion: 0.15
Nodes (6): MarketRegime, String label for logging / serialization boundaries., AlertService, TestStrategyHint, test_alert_service_filters_by_threshold_and_uses_messenger(), test_alert_service_uses_injected_alerter()

### Community 25 - "Regime"
Cohesion: 0.17
Nodes (8): compute_sector_concentration(), Fraction of sectors whose RS is aligned with the current regime direction. -…, Unit tests for the pure compute_sector_concentration() function., Everything down but regime says UP → no alignment → 0.0., 4 out of 8 sectors positive → 0.5., 6 out of 8 sectors negative → 0.75., Output should not carry floating-point noise., TestSectorConcentration

### Community 26 - "Regime"
Cohesion: 0.27
Nodes (16): confidence_position_scale(), Convert regime confidence into a gentle position-size multiplier. The regime…, Compatibility wrapper over :meth:`ScanService.scan`. ``no_intraday`` is…, run_scan(), _FakePersistence, _install_scan_service(), _make_regime_df(), DataFrame (+8 more)

### Community 27 - "Regime Tests"
Cohesion: 0.21
Nodes (5): ADX exactly at REGIME_ADX_TREND threshold should classify as TREND., ADX between ADX_RANGE and ADX_TREND, breadth >= 0.55 → TREND_UP., ADX between ADX_RANGE and ADX_TREND, breadth < 0.45 → TREND_DOWN., ADX mid, breadth ∈ [0.45, 0.55) → RANGE., TestClassifyRegime

### Community 28 - "Data Provider"
Cohesion: 0.23
Nodes (9): core/data_provider.py ===================== Market data retrieval for the…, ensure_parent(), ensure_runtime_dirs(), Path, core/runtime_paths.py ===================== Central runtime paths for mutable…, resolve_artifact_path(), _resolve_repo_path(), RuntimePaths (+1 more)

### Community 29 - "Telemetry"
Cohesion: 0.15
Nodes (6): _JsonFormatter, Emit each log record as a single JSON object on stdout/file. Fields always…, Configure the ``sovereign`` logger hierarchy. Parameters ---------- level: Root…, setup_logging(), LogRecord, TestTelemetry

### Community 30 - "Indicators Tests"
Cohesion: 0.28
Nodes (13): cfg(), _config(), df_down(), df_flat(), df_std(), df_up(), _make_flat(), _make_raw_ohlcv() (+5 more)

### Community 31 - "Backtest"
Cohesion: 0.19
Nodes (7): Total buy-side friction as a fraction of entry price., Total sell-side friction as a fraction of exit price (approx entry price)., Round-trip cost drag expressed in R-multiples. Parameters ---------- entry:…, Run a walk-forward backtest over *raw_data*. Parameters ---------- raw_data:…, walk_forward(), _make_config(), Smoke test: walk_forward runs on a tiny synthetic universe.

### Community 32 - "Indicators"
Cohesion: 0.26
Nodes (11): add_indicators(), DataFrame, Series, core/indicators.py ================== Technical indicator computation for the…, Compute all technical indicators and append them as new columns. Parameters…, Vectorised Supertrend — returns (supertrend_line, trend_up_mask). Bar-0 fix…, _supertrend_vectorised(), _true_range() (+3 more)

### Community 33 - "Factors"
Cohesion: 0.28
Nodes (12): calibrate_ic_weights(), compute_factors(), factor_breakout(), factor_relative_strength(), Any, DataFrame, Series, core/factors.py =============== Seven-factor signal model — ported faithfully… (+4 more)

### Community 34 - "Factors"
Cohesion: 0.33
Nodes (11): FactorScores, _bench(), _config(), _make_ohlcv(), _make_ticker_result(), DataFrame, Series, Synthetic OHLCV with a mild upward drift and realistic column set needed by the… (+3 more)

### Community 35 - "Fyers Setup"
Cohesion: 0.28
Nodes (10): generate_access_token(), persist_access_token(), Path, _repo_env_path(), _upsert_env_var(), test_persist_access_token_appends_missing_value(), test_persist_access_token_replaces_existing_value(), mask_token() (+2 more)

### Community 36 - "Transaction Costs Tests"
Cohesion: 0.15
Nodes (5): Doubling the entry price should double friction_r (sl_dist held constant)., Doubling sl_dist should halve friction_r (entry held constant)., A model with only slippage rounds to expected value., At entry=100, ATR stop = 1.5 (1.5% of price) the round-trip drag should be in…, TestTransactionCostModel

### Community 37 - "Backtest"
Cohesion: 0.24
Nodes (6): compute_trade_management_wrapper(), Series, Thin shim to call scorer.compute_trade_management cleanly., compute_trade_management(), Returns (trail_stop, time_stop_bars)., TestComputeTradeManagement

### Community 38 - "Telemetry"
Cohesion: 0.18
Nodes (6): emit(), Any, Emit one structured metric event. Parameters ---------- event: Short,…, Accept a ``MarketRegime`` dataclass instance., Accept a list of ``TickerResult`` instances., Emit the full scan summary as one JSON event.

### Community 39 - "Phase34 Tests"
Cohesion: 0.29
Nodes (3): _make_ohlcv(), Minimal processed dict for corr matrix tests., TestScanCache

### Community 41 - "Factors"
Cohesion: 0.40
Nodes (3): factor_momentum(), Momentum factor [0, 1]. Components: 30% — RSI zone (48–73 ideal long, 27–52…, TestFactorMomentum

### Community 42 - "Scorer"
Cohesion: 0.22
Nodes (9): DataFrame, Series, Full ticker evaluation pipeline. Returns None if the ticker does not pass any…, score_ticker(), _bench(), DataFrame, Series, Benchmark that slightly underperforms → positive RS for the stock. (+1 more)

### Community 43 - "Indicators Tests"
Cohesion: 0.18
Nodes (5): In a strong uptrend, the last bar should have Super_Up = True., In a strong downtrend, the last bar should have Super_Up = False., After warm-up, Supertrend should have no NaN values., An up-then-down series should show a Supertrend flip., TestSupertrend

### Community 44 - "Backtest"
Cohesion: 0.36
Nodes (5): Simulate a single trade against forward OHLCV bars. Exit logic (in priority…, _realised_r(), DataFrame, TestBacktestHelpers, Timestamp

### Community 45 - "Factors"
Cohesion: 0.29
Nodes (5): Returns (POC, VAL, VAH) for the given OHLCV DataFrame. POC — Price of Control…, true_volume_profile(), Fewer than 5 rows triggers the close-based fallback., hi_p == lo_p edge case (flat price bar)., TestTrueVolumeProfile

### Community 46 - "Scorer"
Cohesion: 0.44
Nodes (3): passes_data_quality(), Reject tickers whose indicator columns contain NaN/NA. These indicate…, TestPassesDataQuality

### Community 47 - "Indicators Tests"
Cohesion: 0.20
Nodes (5): In a strong uptrend, EMA20 > EMA50 > EMA200 on the last bar., In a strong downtrend, EMA20 < EMA50 < EMA200 on the last bar., On flat data, EMAs should converge to roughly the same value., EMA_20 should react faster to a price jump than EMA_200., TestEMAStack

### Community 48 - "Factors"
Cohesion: 0.47
Nodes (3): factor_volume(), Volume factor [0, 1]. Components: 55% — Relative volume (RVOL vs 20d average)…, TestFactorVolume

### Community 49 - "Factors Tests"
Cohesion: 0.44
Nodes (3): Smoke tests that indicators.py produces the expected columns., RVol_20 should be ~1.0 for flat volume, not a tiny std-dev., TestAddIndicators

### Community 51 - "Backtest"
Cohesion: 0.29
Nodes (5): _fold_stats(), FoldStats, _overall_stats(), Compute statistics for one fold from its trade list., Aggregate statistics across all folds.

### Community 52 - "Retry"
Cohesion: 0.39
Nodes (3): Call *fn* up to *max_attempts* times, sleeping between failures. Parameters…, retry_with_backoff(), TestRetryWithBackoff

### Community 53 - "Indicators Tests"
Cohesion: 0.25
Nodes (4): RSI should be elevated in a strong uptrend., RSI should be depressed in a strong downtrend., A series of constant gains should push RSI toward 100., TestRSI

### Community 54 - "Indicators Tests"
Cohesion: 0.25
Nodes (3): A bar cannot be both an up day and a down day., Strong uptrend should have more up days than down days., TestCandleHelpers

### Community 55 - "Config"
Cohesion: 0.29
Nodes (3): Return session label for the given IST datetime (defaults to now)., Fix 6 — Return True during the opening noise window. The first…, datetime

### Community 56 - "Indicators Tests"
Cohesion: 0.29
Nodes (3): MACD histogram should be predominantly positive in an uptrend., MACD histogram should be predominantly negative in a downtrend., TestMACD

### Community 57 - "Indicators Tests"
Cohesion: 0.29
Nodes (3): ATR_50_mean should be close to the mean of ATR over last 50 bars., Higher volatility data should produce higher ATR., TestATR

### Community 58 - "Indicators Tests"
Cohesion: 0.29
Nodes (3): Strong uptrend should produce elevated ADX., Flat market should produce lower ADX than trending market., TestADX

### Community 60 - "Indicators Tests"
Cohesion: 0.29
Nodes (3): Vol_Avg_20 should approximate the rolling 20-bar volume mean., Constant volume should produce RVol_20 ≈ 1.0., TestVolumeMetrics

### Community 61 - "Indicators Tests"
Cohesion: 0.29
Nodes (4): After add_indicators(), DROPNA_COLS should contain no NaNs., Output should have retained most rows after NaN cleanup., Even a very short OHLCV input should not crash, though output may be empty., TestDropNaN

### Community 62 - "Retry"
Cohesion: 0.33
Nodes (6): BaseException, F, _default_retryable(), Return True for exceptions that are worth retrying., Decorator factory: wraps *fn* with exponential-backoff retry logic. Parameters…, retry_with_backoff()

### Community 63 - "Config"
Cohesion: 0.33
Nodes (4): Safe repr that masks all ``SecretStr`` fields. Without this override,…, A ``str`` subclass whose ``__repr__`` never exposes the secret value. Use for…, SecretStr, str

### Community 64 - "Retry"
Cohesion: 0.33
Nodes (3): Any, Execute *fn* through the breaker. Raises ------ CircuitBreakerOpen If the…, Manually record a success (e.g. from an async path).

### Community 65 - "Dashboard"
Cohesion: 0.33
Nodes (6): Eight-Parameter Screener, NSE Unified Scanner, RegimeTracker, ServiceBundle, Seven-Factor Signal Model, Sovereign Engine v14.6-Modular

### Community 66 - "Backtest"
Cohesion: 0.40
Nodes (3): DataFrame, Convert trade records to a tidy DataFrame for analysis., Write trade records to CSV.

### Community 67 - "Cache"
Cohesion: 0.40
Nodes (3): Any, Return a snapshot of cache effectiveness., Emit cache stats at DEBUG level.

### Community 70 - "Sovereign Core Tests"
Cohesion: 0.50
Nodes (3): Light end-to-end path: regime → kelly → targets, no network calls., Ensure no exceptions through the core math path for one ticker., TestIntegrationSmoke

## Knowledge Gaps
- **5 isolated node(s):** `sovereign-engine`, `RegimeTracker`, `ServiceBundle`, `NSE Unified Scanner`, `CI Workflow`
  These have ≤1 connection - possible missing edges or undocumented components.
- **6 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `SystemConfig` connect `Config` to `Async Data`, `Portfolio`, `Regime`, `Backtest`, `Services`, `Regime`, `Regime`, `Scorer Portfolio Tests`, `Backtest`, `Regime`, `Portfolio`, `Cache`, `Regime`, `Backtest`, `Scorer`, `Retry`, `Portfolio`, `Portfolio`, `Regime`, `Regime`, `Regime`, `Regime Tests`, `Data Provider`, `Telemetry`, `Backtest`, `Indicators`, `Factors`, `Factors`, `Transaction Costs Tests`, `Backtest`, `Phase34 Tests`, `Regime Tests`, `Scorer`, `Backtest`, `Scorer`, `Phase34 Tests`, `Backtest`, `Retry`, `Config`, `Config`, `Sovereign Core Tests`?**
  _High betweenness centrality (0.466) - this node is a cross-community bridge._
- **Why does `add_indicators()` connect `Indicators` to `Backtest`, `Backtest`, `Indicators Tests`, `Config`, `Indicators Tests`, `Factors Tests`, `Factors`, `Indicators Tests`, `Indicators Tests`, `Indicators Tests`, `Indicators Tests`, `Indicators Tests`, `Indicators Tests`, `Backtest`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Why does `RegimeTracker` connect `Regime` to `Portfolio`, `Regime`, `Backtest`, `Services`, `Regime`, `Backtest`, `Regime`, `Portfolio`, `Regime`, `Backtest`, `Config`, `Scorer`, `Regime`, `Regime`, `Regime`, `Regime Tests`, `Backtest`, `Regime Tests`, `Backtest`, `Sovereign Core Tests`?**
  _High betweenness centrality (0.090) - this node is a cross-community bridge._
- **Are the 69 inferred relationships involving `SystemConfig` (e.g. with `FoldStats` and `OverallStats`) actually correct?**
  _`SystemConfig` has 69 INFERRED edges - model-reasoned connections that need verification._
- **Are the 43 inferred relationships involving `RegimeTracker` (e.g. with `FoldStats` and `OverallStats`) actually correct?**
  _`RegimeTracker` has 43 INFERRED edges - model-reasoned connections that need verification._
- **Are the 41 inferred relationships involving `MarketRegime` (e.g. with `MarketRegimeType` and `SystemConfig`) actually correct?**
  _`MarketRegime` has 41 INFERRED edges - model-reasoned connections that need verification._
- **Are the 39 inferred relationships involving `MarketRegimeType` (e.g. with `MarketRegime` and `RegimeTracker`) actually correct?**
  _`MarketRegimeType` has 39 INFERRED edges - model-reasoned connections that need verification._