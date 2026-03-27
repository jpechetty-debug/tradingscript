"""
screener_v14_modular.py
=======================
Sovereign Engine v14 — modular entry point.

What changed from v13 (screener_v13_modular.py):
  ✅ core/scorer.py   — full 7-factor model, zero placeholders
  ✅ core/factors.py  — new module: all factor functions, individually testable
  ✅ core/universe.py — new module: SECTORS / TICKER_TO_SECTOR extracted from monolith
  ✅ core/portfolio.py — MAX_CORR correlation gate now actually wired
  ✅ core/config.py   — session_from_time() added; runtime state removed
  ✅ ScanState        — encapsulates all mutable scan context (no global dict mutation)
  ✅ No imports from screener.py monolith

Usage:
    python screener_v14_modular.py
    python screener_v14_modular.py --watch 15
    python screener_v14_modular.py --debug
    python screener_v14_modular.py --version
    python screener_v14_modular.py --no-telegram
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.config import CONFIG, SystemConfig, IST, MarketRegimeType
from core.data_provider import fetch_daily_batch
from core.factors import DEFAULT_WEIGHTS, calibrate_ic_weights
from core.indicators import add_indicators
from core.portfolio import optimize_portfolio, CapitalScaler
from core.regime import (
    RegimeTracker,
    MarketRegime,
    classify_regime,
    compute_breadth,
    compute_sector_rs,
    compute_sector_concentration,
)
from core.scorer import TickerResult, score_ticker
from core.universe import ALL_TICKERS, SECTORS, TICKER_TO_SECTOR, N_SECTORS
from utils.messaging import send_telegram
from core.telemetry import setup_logging, ScanMetrics, emit
from core.cache import _build_corr_matrix, ScanCache
from core.backtest import walk_forward, WalkForwardResult
from core.retry import guarded_call, FYERS_BREAKER, YFINANCE_BREAKER, TELEGRAM_BREAKER

# sovereign_improvements is an optional monkey-patch layer.  Guard the import
# so a missing file, a broken dependency (e.g. sklearn not installed), or a
# syntax error in the patch module never crashes the entry point.  When the
# import fails we substitute a minimal no-op stub so the rest of the file
# can call SE_PATCH.apply() unconditionally without branching everywhere.
try:
    import sovereign_improvements as SE_PATCH
    _SE_PATCH_AVAILABLE = True
except Exception as _se_err:                         # ImportError, SyntaxError, …
    import types as _types

    class _NoOpPatch:
        """
        Minimal stand-in returned when sovereign_improvements cannot be loaded.

        Every attribute access returns either a no-op callable or a safe
        default so callers that do ``SE_PATCH.apply(...)`` or
        ``patch.get("calibrator")`` work without any additional guards.
        """
        def apply(self, *a, **kw) -> dict:
            return {}

        def __getattr__(self, name: str):
            return lambda *a, **kw: None

    SE_PATCH = _NoOpPatch()  # type: ignore[assignment]
    _SE_PATCH_AVAILABLE = False
    # Defer logging until after setup_logging() runs below.
    _SE_PATCH_LOAD_ERROR = _se_err

VERSION = "14.4-Modular"
PLATT_CALIB_FILE = Path("platt_calibration.json")

setup_logging(level="INFO", json_log_file="logs/sovereign.jsonl")
log = logging.getLogger("sovereign")

if not _SE_PATCH_AVAILABLE:
    log.warning(
        "sovereign_improvements could not be loaded (%s: %s). "
        "Running without patch components — tiered capital scaling, "
        "regime-dependent prob gates, and regime-aware Telegram alerts "
        "are disabled. Install missing dependencies or restore the file "
        "to re-enable.",
        type(_SE_PATCH_LOAD_ERROR).__name__,
        _SE_PATCH_LOAD_ERROR,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCAN STATE  (replaces mutable CONFIG dict entries)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanState:
    """
    All mutable context for a single scan session.
    Created fresh per run_scan() call — no watch-mode bleed.

    Patch components (gate, scaler, calibrator, alerter) live here rather
    than in a module-level PATCH dict so that each scan cycle is fully
    self-contained and watch-mode runs cannot share stale state.
    """
    platt_a: float = CONFIG.PLATT_A
    platt_b: float = CONFIG.PLATT_B
    platt_from_file: bool = False
    weights_calibrated: bool = False
    factor_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    ic_history: dict = field(default_factory=dict)
    capital_scaler: Optional[CapitalScaler] = None

    # Patch components — populated by main() via SE_PATCH.apply().
    # Keeping them on ScanState rather than in a module global means
    # each run_scan() call is fully self-contained.
    patch_gate:       Optional[object] = None
    patch_scaler:     Optional[object] = None
    patch_calibrator: Optional[object] = None
    patch_alerter:    Optional[object] = None

    # Fix 6: opening-range regime lock — set once per run_scan() call
    regime_locked: bool = False

    # Per-scan cache — created fresh so different scan cycles never
    # share cached volume profiles or correlation matrices.
    cache: "ScanCache" = field(default_factory=lambda: ScanCache())  # ScanCache imported below

    def load_platt(self) -> None:
        """Load Platt A/B from file if present; fall back to config defaults."""
        if PLATT_CALIB_FILE.exists():
            try:
                d = json.loads(PLATT_CALIB_FILE.read_text())
                self.platt_a = float(d["A"])
                self.platt_b = float(d["B"])
                self.platt_from_file = True
                log.info("Platt params loaded from file: A=%.4f B=%.4f", self.platt_a, self.platt_b)
            except Exception as e:
                log.warning("Could not load %s: %s — using defaults", PLATT_CALIB_FILE, e)

    def save_platt(self, a: float, b: float) -> None:
        try:
            PLATT_CALIB_FILE.write_text(
                json.dumps({"A": a, "B": b, "fitted_at": datetime.now().isoformat()}, indent=2)
            )
            self.platt_a, self.platt_b = a, b
            log.info("Platt params saved: A=%.4f B=%.4f", a, b)
        except Exception as e:
            log.warning("Could not save Platt params: %s", e)

    def calibrate(self, composites: list[float], outcomes: list[int]) -> None:
        """Fit new Platt A/B and persist."""
        from core.scorer import calibrate_platt
        if not composites or len(composites) != len(outcomes):
            log.error("Invalid calibration data")
            return
        a, b = calibrate_platt(composites, outcomes)
        self.save_platt(a, b)


# ─────────────────────────────────────────────────────────────────────────────
# COVARIANCE (for portfolio correlation filter)
# ─────────────────────────────────────────────────────────────────────────────

# _build_corr_matrix moved to core/cache.py and imported above.


# ─────────────────────────────────────────────────────────────────────────────
# SECTOR RANKING
# ─────────────────────────────────────────────────────────────────────────────

def _rank_sectors(sector_rs: dict[str, float]) -> dict[str, int]:
    """Rank sectors by RS score; rank 1 = strongest."""
    sorted_sectors = sorted(sector_rs.items(), key=lambda x: x[1], reverse=True)
    return {s: i + 1 for i, (s, _) in enumerate(sorted_sectors)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCAN
# ─────────────────────────────────────────────────────────────────────────────

def passes_static_filters(df: pd.DataFrame, config: SystemConfig) -> bool:
    """Check ADV and structural filters (EMA200)."""
    if len(df) < 200: return False
    
    # ADV floor (Fix: use tail(20) for ADV20)
    close = df['Close']
    volume = df['Volume']
    turnover = (close * volume).tail(20).mean()
    
    if turnover < config.ADV_TURNOVER_FLOOR:
        return False
    
    # EMA200 filter
    if config.USE_EMA200_FILTER:
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        if close.iloc[-1] < ema200:
            return False
            
    return True


def run_scan(
    config: SystemConfig = CONFIG,
    debug: bool = False,
    no_intraday: bool = False,
    patch: Optional[dict] = None,
    regime_override: Optional[str] = None,
    no_ema_filter: bool = False,
    force_score: bool = False,
) -> tuple[list[TickerResult], list[TickerResult], Optional[MarketRegime]]:
    """
    Full scan pipeline. Returns (all_results, portfolio, regime).

    Parameters
    ----------
    patch
        Dict of patch components returned by SE_PATCH.apply() — keys
        ``gate``, ``scaler``, ``calibrator``, ``alerter``.  Pass None
        (default) for a bare scan with no patch components.

    Steps:
        1. Fetch & process daily data
        2. Classify market regime
        3. Score each ticker (parallel)
        4. Optimise portfolio
    """
    patch = patch or {}

    # Fresh per-scan state and cache — no watch-mode bleed, no cross-test
    # contamination when run_scan is called from tests.
    state = ScanState()
    state.patch_gate       = patch.get("gate")
    state.patch_scaler     = patch.get("scaler")
    state.patch_calibrator = patch.get("calibrator")
    state.patch_alerter    = patch.get("alerter")

    state.cache.clear()   # defensive: ensure the new cache starts clean
    _metrics = ScanMetrics()

    state.load_platt()

    # 14.4: Use dynamic factor weights from the background calibrator if available
    if state.patch_calibrator:
        state.factor_weights = state.patch_calibrator.current_weights()
        log.info("Using dynamic factor weights: %s", {k: round(v, 3) for k, v in state.factor_weights.items()})

    # Inject calibrated Platt into config for this scan
    config = SystemConfig(
        **{**config.__dict__, "PLATT_A": state.platt_a, "PLATT_B": state.platt_b}
    )

    log.info("Sovereign Engine v%s | %s", VERSION, datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"))

    # ── 1. Data ───────────────────────────────────────────────────────────────
    _t0 = time.monotonic()
    raw_data = fetch_daily_batch(ALL_TICKERS, config)
    _fetch_elapsed = time.monotonic() - _t0
    if not raw_data:
        log.error("No data fetched — aborting scan")
        return [], [], None
    _metrics.record_fetch(
        n_ok=len(raw_data),
        n_fail=len(ALL_TICKERS) + 1 - len(raw_data),
        elapsed_s=_fetch_elapsed,
    )

    processed: dict[str, pd.DataFrame] = {}
    _ind_fail = 0
    _t0 = time.monotonic()
    for ticker, df in raw_data.items():
        try:
            processed[ticker] = add_indicators(df, config)
        except Exception:
            _ind_fail += 1
            log.debug("Indicator error for %s.", ticker, exc_info=True)
    _metrics.record_indicators(
        n_ok=len(processed),
        n_fail=_ind_fail,
        elapsed_s=time.monotonic() - _t0,
    )

    bench_key = config.BENCHMARK
    if bench_key not in processed:
        log.error("Benchmark %s not in processed data", bench_key)
        return [], [], None

    bench_series = processed[bench_key]["Close"]

    # ── 2. Regime ─────────────────────────────────────────────────────────────
    breadth = compute_breadth(processed, config)

    # Fix 7: compute sector RS *before* classify_regime so concentration
    # feeds into confidence scoring on the same bar it's measured.
    sector_rs    = compute_sector_rs(processed, bench_series, config)
    sector_ranks = _rank_sectors(sector_rs)

    # Fix 6: lock regime changes during the opening noise window
    state.regime_locked = config.is_regime_locked()
    if state.regime_locked:
        log.info("🔒 Regime lock active — opening noise window (%d min). "
                 "Tracker will not update this bar.", config.REGIME_LOCK_MINUTES)

    tracker = RegimeTracker()
    regime  = classify_regime(
        processed, breadth, tracker, config,
        locked=state.regime_locked,     # Fix 6
        sector_rs=sector_rs,            # Fix 7
    )

    if regime_override:
        try:
            o_type = MarketRegimeType(regime_override.upper())
            log.warning("⚠️ REGIME OVERRIDE ACTIVE: %s (was %s)", o_type, regime.regime)
            regime = MarketRegime(
                regime=o_type,
                breadth=regime.breadth,
                breadth_delta=regime.breadth_delta,
                adx_median=regime.adx_median,
                atr_ratio=regime.atr_ratio,
                confidence=1.0,  # Max confidence for override
                confirmed=True,
                sector_concentration=regime.sector_concentration,
                regime_locked=False,
            )
        except ValueError:
            log.error("Invalid override regime: %s — ignoring", regime_override)

    session = config.session_from_time()

    _metrics.record_regime(regime)
    log.info(
        "📊 Breadth: %.0f%% (Δ%.2f)  |  Regime: %s (%s)  |  "
        "Conf: %.2f  |  ADX: %.1f  |  ATR ratio: %.2f  |  "
        "Sector conc: %.0f%%  |  %s",
        regime.breadth * 100,
        regime.breadth_delta,
        regime.regime,
        "CONFIRMED" if regime.confirmed else f"{config.REGIME_CONFIRM_BARS} bar CONFIRM REQ",
        regime.confidence,
        regime.adx_median,
        regime.atr_ratio,
        regime.sector_concentration * 100,
        "LOCKED" if regime.regime_locked else "live",
    )

    if not regime.is_tradeable():
        log.warning("PANIC regime — no new positions")
        return [], [], regime

    # ── 4. Filtering & Scoring ────────────────────────────────────────────────
    all_results: list[TickerResult] = []
    
    # Temporary config override for filter toggle
    effective_config = config
    if no_ema_filter:
        import copy
        effective_config = copy.copy(config)
        effective_config.USE_EMA200_FILTER = False

    _t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {}
        for ticker, df in processed.items():
            if ticker == config.BENCHMARK:
                continue

            # Liquidity & Structural filters
            if not passes_static_filters(df, effective_config):
                continue
            
            # Submit to thread pool for scoring
            capital_f = 1.0
            if state.patch_scaler:
                try:
                    capital_f = state.patch_scaler.capital_fraction(1_000_000, regime.regime)
                except Exception:
                    pass

            futures[executor.submit(
                score_ticker,
                ticker=ticker,
                daily_df=df,
                bench=bench_series,
                sector_ranks=sector_ranks,
                sector_rs=sector_rs,
                session=session,
                regime=regime,
                config=config,
                factor_weights=state.factor_weights,
                capital_fraction=capital_f,
                debug=debug,
                force_score=force_score,
            )] = ticker

        for future in as_completed(futures):
            t = futures[future]
            try:
                result = future.result()
                if result is not None:
                    all_results.append(result)
            except Exception as e:
                log.error("score_ticker error for %s: %s", t, e, exc_info=True)

    _metrics.record_score(
        n_passed=len(all_results),
        n_total=len(processed) - 1,
        elapsed_s=time.monotonic() - _t0,
    )
    log.info("%d tickers passed all gates", len(all_results))

    # ── 5. ICIR Factor Weight Calibration ─────────────────────────────────────
    if len(all_results) >= config.ICIR_MIN_OBS:
        try:
            new_weights = calibrate_ic_weights(
                results=all_results,
                processed=processed,
                bench=bench_series,
                sector_ranks=sector_ranks,
                n_sectors=N_SECTORS,
                config=config,
            )
            state.factor_weights = new_weights
            state.weights_calibrated = True
            log.info("🎯 IC weights optimized: %s", {k: round(v, 4) for k, v in new_weights.items()})
        except Exception as e:
            log.warning("IC calibration failed: %s", e)

    # ── 6. Portfolio optimisation ─────────────────────────────────────────────
    corr_matrix = state.cache.corr_matrix(processed, config)
    portfolio   = optimize_portfolio(all_results, config, corr_matrix)

    _metrics.record_portfolio(portfolio)
    _metrics.emit_summary()
    state.cache.log_stats()
    log.info("Portfolio: %d positions selected", len(portfolio))
    for r in portfolio:
        log.info(
            "  %s | %s | P=%.2f | E(R)=%.3f | RR=%.1fx | %d shares | ₹%.0f risk",
            r.ticker, r.direction, r.prob_win, r.expectancy_r, r.rr_t1, r.shares, r.risk_inr,
        )

    return all_results, portfolio, regime


# ─────────────────────────────────────────────────────────────────────────────
# TELEGRAM ALERT
# ─────────────────────────────────────────────────────────────────────────────

def _send_alert(portfolio: list[TickerResult], regime: MarketRegime, config: SystemConfig, patch: Optional[dict] = None) -> None:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        log.debug("Telegram credentials missing in config")
        return

    top = [r for r in portfolio if r.prob_win >= config.TELEGRAM_ALERT_MIN_PROB]
    log.debug("Telegram candidates: %d/%d (threshold %.2f)", len(top), len(portfolio), config.TELEGRAM_ALERT_MIN_PROB)

    if not top:
        return

    patch = patch or {}
    if patch.get("alerter"):
        patch["alerter"].send_daily_summary(regime.regime, [r.__dict__ for r in top])
    else:
        lines = [f"<b>Sovereign v{VERSION}</b> | {regime.regime} | {datetime.now(IST).strftime('%H:%M IST')}"]
        for r in top[:config.TELEGRAM_ALERT_TOP_N]:
            lines.append(
                f"<b>{r.ticker}</b> {r.direction} | P={r.prob_win:.0%} | E(R)={r.expectancy_r:.2f} | "
                f"Entry {r.entry} | SL {r.stop} | T1 {r.t1} | {r.shares} shares"
            )
        send_telegram("\n".join(lines), config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID)


def run_calibration() -> None:
    """Load trade_log.json, re-calculate composites, and re-fit Platt A/B."""
    log.info("Starting Platt calibration from trade_log.json...")
    trade_log_path = Path(os.getenv("TRADE_LOG_PATH", "trade_log.json"))
    if not trade_log_path.exists():
        log.error("Trade log not found at %s", trade_log_path)
        return

    try:
        trades = json.loads(trade_log_path.read_text())
    except Exception as e:
        log.error("Could not load trade log: %s", e)
        return

    composites, outcomes = [], []
    for t in trades:
        f = t.get("factors")
        pnl = t.get("pnl")
        if f and pnl is not None:
            # Reconstruct composite using default weights if not present
            comp = t.get("composite")
            if comp is None:
                comp = sum(f.get(k, 0) * DEFAULT_WEIGHTS.get(k, 0) for k in DEFAULT_WEIGHTS)
            
            composites.append(float(comp))
            outcomes.append(1 if pnl > 0 else 0)

    if len(composites) < 5:
        log.warning("Insufficient data for calibration (%d samples)", len(composites))
        return

    log.info("Calibrating on %d samples...", len(composites))
    state = ScanState()
    state.calibrate(composites, outcomes)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────



# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    config:      SystemConfig = CONFIG,
    train_days:  int = 120,
    test_days:   int = 20,
    step_days:   int = 10,
    out_csv:     str = "backtest_results.csv",
    direction:   str = "LONG",
    debug:       bool = False,
) -> WalkForwardResult:
    """
    Fetch data for the full universe, run walk-forward backtest, print summary.

    Called by: ``python screener_v14_modular.py --backtest``

    Parameters
    ----------
    config:       SystemConfig instance.
    train_days:   Bars in each train window (default 120 ~ 6 months).
    test_days:    Bars in each forward test window (default 20 ~ 1 month).
    step_days:    Step between fold start dates (default 10 ~ 2 weeks).
    out_csv:      Path to write per-trade CSV output.
    direction:    "LONG", "SHORT", or "BOTH".
    debug:        Enable DEBUG logging.

    Returns
    -------
    WalkForwardResult with .overall, .fold_stats, .trades.
    """
    if debug:
        logging.getLogger("sovereign").setLevel(logging.DEBUG)

    log.info(
        "Starting walk-forward backtest | train=%d test=%d step=%d direction=%s",
        train_days, test_days, step_days, direction,
    )

    # Fetch raw data (no indicators — backtest adds them per fold)
    t0 = time.monotonic()
    raw_data = fetch_daily_batch(ALL_TICKERS, config)
    elapsed = time.monotonic() - t0
    log.info("Data fetched: %d symbols in %.1fs", len(raw_data), elapsed)

    if not raw_data:
        log.error("No data fetched — aborting backtest.")
        return WalkForwardResult(trades=[], fold_stats=[], overall=None)  # type: ignore

    # Run walk-forward
    results = walk_forward(
        raw_data=raw_data,
        config=config,
        train_days=train_days,
        test_days=test_days,
        step_days=step_days,
        direction=direction,
    )

    # Print summary table
    o = results.overall
    print("\n" + "=" * 60)
    print(f"  Walk-Forward Backtest Summary  (v{VERSION})")
    print("=" * 60)
    print(f"  Folds          : {o.n_folds}")
    print(f"  Total trades   : {o.n_trades}")
    print(f"  Hit rate       : {o.hit_rate:.1%}")
    print(f"  Mean R         : {o.mean_r:+.3f}")
    print(f"  Total R        : {o.total_r:+.2f}")
    print(f"  Sharpe (ann.)  : {o.sharpe:+.2f}")
    print(f"  Max drawdown   : {o.max_dd:.2f} R")
    print(f"  Profit factor  : {o.profit_factor:.2f}")
    print(f"  Expectancy R   : {o.expectancy_r:+.4f}")
    print("=" * 60)

    if o.fold_stats:
        print("\nPer-fold breakdown:")
        print(f"  {'Fold':>4}  {'Start':>12}  {'End':>12}  "
              f"{'Trades':>6}  {'Hit%':>5}  {'MeanR':>6}  {'TotalR':>7}")
        for fs in o.fold_stats:
            print(f"  {fs.fold:>4}  {str(fs.start_date.date()):>12}  "
                  f"{str(fs.end_date.date()):>12}  "
                  f"{fs.n_trades:>6}  {fs.hit_rate:>4.0%}  "
                  f"{fs.mean_r:>+6.3f}  {fs.total_r:>+7.2f}")

    # Save CSV
    if results.trades:
        results.to_csv(out_csv)
        print(f"\nTrade log saved → {out_csv}")
    else:
        log.warning("No trades generated — check min_prob threshold and data quality.")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Sovereign Engine v{VERSION}")
    parser.add_argument("--watch",       type=int,  default=None, metavar="MINUTES")
    parser.add_argument("--version",     action="store_true")
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--no-intraday", action="store_true")
    parser.add_argument("--calibrate",   action="store_true", help="Re-fit Platt A/B from trade_log.json")
    parser.add_argument("--backtest",      action="store_true",  help="Run walk-forward backtest and exit")
    parser.add_argument("--bt-train",      type=int, default=120, metavar="DAYS", help="Backtest train window (default 120)")
    parser.add_argument("--bt-test",       type=int, default=20,  metavar="DAYS", help="Backtest test window (default 20)")
    parser.add_argument("--bt-step",       type=int, default=10,  metavar="DAYS", help="Backtest step between folds (default 10)")
    parser.add_argument("--bt-out",        type=str, default="backtest_results.csv", metavar="FILE", help="CSV output path")
    parser.add_argument("--bt-direction",  type=str, default="LONG", choices=["LONG","SHORT","BOTH"], help="Trade direction")
    parser.add_argument("--regime-override", type=str, default=None, help="Force a specific regime (TREND_UP, RANGE, etc.)")
    parser.add_argument("--no-ema-filter", action="store_true", help="Disable EMA200 structural filter for debugging")
    parser.add_argument("--force-score", action="store_true", help="Force scoring of all tickers (bypass BULL/BEAR directional gates)")
    args = parser.parse_args()

    if args.version:
        print(f"Sovereign Engine v{VERSION}")
        return

    config = CONFIG

    # Initialise patch components once at startup; pass them into each
    # run_scan() call rather than storing them in a module-level global.
    patch = SE_PATCH.apply(None, portfolio_peak=1_000_000)

    def _step() -> None:
        results, portfolio, regime = run_scan(
            config=config,
            debug=args.debug,
            no_intraday=args.no_intraday,
            patch=patch,
            regime_override=args.regime_override,
            no_ema_filter=args.no_ema_filter,
            force_score=args.force_score,
        )
        if not args.no_telegram and portfolio and regime:
            _send_alert(portfolio, regime, config, patch=patch)

        # Feed completed trades into the calibrator buffer for live updates.
        # In a real live environment, record actual closed trades here.
        if patch.get("calibrator"):
            pass  # placeholder — wire in broker PnL events here

    if args.backtest:
        run_backtest(
            config=config,
            train_days=args.bt_train,
            test_days=args.bt_test,
            step_days=args.bt_step,
            out_csv=args.bt_out,
            direction=args.bt_direction,
            debug=args.debug,
        )
        return
    elif args.calibrate:
        run_calibration()
    elif args.watch:
        log.info("Watch mode — scanning every %d minutes", args.watch)
        while True:
            try:
                _step()
            except KeyboardInterrupt:
                log.info("Watch mode stopped")
                sys.exit(0)
            except Exception as e:
                log.error("Scan error: %s", e, exc_info=args.debug)
            time.sleep(args.watch * 60)
    else:
        _step()


if __name__ == "__main__":
    main()
