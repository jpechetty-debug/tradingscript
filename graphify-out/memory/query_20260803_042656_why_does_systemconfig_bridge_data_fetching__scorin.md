---
type: "query"
date: "2026-08-03T04:26:56.797125+00:00"
question: "Why does SystemConfig bridge data fetching, scoring, portfolio construction, regimes, backtesting, and services?"
contributor: "graphify"
source_nodes: ["SystemConfig", "ScanService", "fetch_daily_batch()", "classify_regime()", "score_ticker()", "optimize_portfolio()", "walk_forward()"]
---

# Q: Why does SystemConfig bridge data fetching, scoring, portfolio construction, regimes, backtesting, and services?

## Answer

SystemConfig is passed directly from the CLI through ScanService to market-data preparation, regime classification, scoring, portfolio optimization, and walk-forward backtesting. It also holds dynamic calibration and probability thresholds, so static policy and per-scan state share one object. Split it into typed domain settings, pass narrow settings to leaf modules, and move calibrated Platt parameters and regime-adjusted thresholds into scan state.

## Source Nodes

- SystemConfig
- ScanService
- fetch_daily_batch()
- classify_regime()
- score_ticker()
- optimize_portfolio()
- walk_forward()