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

from core.config import CONFIG, SystemConfig, IST
from core.data_provider import fetch_daily_batch
from core.factors import DEFAULT_WEIGHTS
from core.indicators import add_indicators
from core.portfolio import optimize_portfolio
from core.regime import (
    RegimeTracker,
    MarketRegime,
    classify_regime,
    compute_breadth,
    compute_sector_rs,
)
from core.scorer import TickerResult, score_ticker
from core.universe import ALL_TICKERS, SECTORS, TICKER_TO_SECTOR, N_SECTORS
from utils.messaging import send_telegram

VERSION = "14.0-Modular"
PLATT_CALIB_FILE = Path("platt_calibration.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sovereign")


# ─────────────────────────────────────────────────────────────────────────────
# SCAN STATE  (replaces mutable CONFIG dict entries)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ScanState:
    """
    All mutable context for a single scan session.
    Created fresh per run_scan() call — no watch-mode bleed.
    """
    platt_a: float = CONFIG.PLATT_A
    platt_b: float = CONFIG.PLATT_B
    platt_from_file: bool = False
    weights_calibrated: bool = False
    factor_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    ic_history: dict = field(default_factory=dict)

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

def _build_corr_matrix(
    processed: dict[str, pd.DataFrame],
    config: SystemConfig,
) -> pd.DataFrame:
    """Build pairwise return correlation matrix from daily close prices."""
    bench = config.BENCHMARK
    returns = {}
    for ticker, df in processed.items():
        if ticker == bench or df.empty:
            continue
        rets = df["Close"].pct_change().dropna().tail(config.COV_LOOKBACK)
        if len(rets) >= 20:
            returns[ticker] = rets

    if len(returns) < 2:
        return pd.DataFrame()

    ret_df = pd.DataFrame(returns).dropna(how="all")
    return ret_df.corr()


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

def run_scan(
    config: SystemConfig = CONFIG,
    debug: bool = False,
    no_intraday: bool = False,
) -> tuple[list[TickerResult], list[TickerResult], Optional[MarketRegime]]:
    """
    Full scan pipeline. Returns (all_results, portfolio, regime).

    Steps:
        1. Fetch & process daily data
        2. Classify market regime
        3. Score each ticker (parallel)
        4. Optimise portfolio
    """
    state = ScanState()
    state.load_platt()

    # Inject calibrated Platt into config for this scan
    config = SystemConfig(
        **{**config.__dict__, "PLATT_A": state.platt_a, "PLATT_B": state.platt_b}
    )

    log.info("Sovereign Engine v%s | %s", VERSION, datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"))

    # ── 1. Data ───────────────────────────────────────────────────────────────
    raw_data = fetch_daily_batch(ALL_TICKERS, config)
    if not raw_data:
        log.error("No data fetched — aborting scan")
        return [], [], None

    processed: dict[str, pd.DataFrame] = {}
    for ticker, df in raw_data.items():
        try:
            processed[ticker] = add_indicators(df, config)
        except Exception as e:
            log.debug("Indicator error for %s: %s", ticker, e)

    bench_key = config.BENCHMARK
    if bench_key not in processed:
        log.error("Benchmark %s not in processed data", bench_key)
        return [], [], None

    bench_series = processed[bench_key]["Close"]

    # ── 2. Regime ─────────────────────────────────────────────────────────────
    breadth   = compute_breadth(processed, config)
    tracker   = RegimeTracker()
    regime    = classify_regime(processed, breadth, tracker, config)
    session   = config.session_from_time()

    log.info("📊 Breadth: %.0f%% above EMA-50  |  Regime: %s (%s)  |  Conf: %.2f  |  ADX: %.1f  |  ATR ratio: %.2f",
             regime.breadth*100, regime.regime, "CONFIRMED" if regime.confirmed else f"{config.REGIME_CONFIRM_BARS} bar CONFIRM REQ",
             regime.confidence, regime.adx_median, regime.atr_ratio)

    if not regime.is_tradeable():
        log.warning("PANIC regime — no new positions")
        return [], [], regime

    # ── 3. Sector RS & ranks ──────────────────────────────────────────────────
    sector_rs    = compute_sector_rs(processed, bench_series, config)
    sector_ranks = _rank_sectors(sector_rs)

    # ── 4. Score tickers (parallel) ───────────────────────────────────────────
    all_results: list[TickerResult] = []

    def _score_one(ticker: str) -> Optional[TickerResult]:
        df = processed.get(ticker)
        if df is None or df.empty:
            return None
        try:
            return score_ticker(
                ticker=ticker,
                daily_df=df,
                bench=bench_series,
                sector_ranks=sector_ranks,
                sector_rs=sector_rs,
                session=session,
                regime=regime,
                config=config,
                intraday={} if no_intraday else {},  # hook for live intraday data
                mtf_60m={},
                factor_weights=state.factor_weights,
                debug=debug,
            )
        except Exception as e:
            log.debug("score_ticker error for %s: %s", ticker, e)
            return None

    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
        futures = {executor.submit(_score_one, t): t for t in ALL_TICKERS}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_results.append(result)

    log.info("%d tickers passed all gates", len(all_results))

    # ── 5. Portfolio optimisation ─────────────────────────────────────────────
    corr_matrix = _build_corr_matrix(processed, config)
    portfolio   = optimize_portfolio(all_results, config, corr_matrix)

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

def _send_alert(portfolio: list[TickerResult], regime: MarketRegime, config: SystemConfig) -> None:
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    top = [r for r in portfolio if r.prob_win >= config.TELEGRAM_ALERT_MIN_PROB]
    if not top:
        return
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

def main() -> None:
    parser = argparse.ArgumentParser(description=f"Sovereign Engine v{VERSION}")
    parser.add_argument("--watch",       type=int,  default=None, metavar="MINUTES")
    parser.add_argument("--version",     action="store_true")
    parser.add_argument("--debug",       action="store_true")
    parser.add_argument("--no-telegram", action="store_true")
    parser.add_argument("--no-intraday", action="store_true")
    parser.add_argument("--calibrate",   action="store_true", help="Re-fit Platt A/B from trade_log.json")
    args = parser.parse_args()

    if args.version:
        print(f"Sovereign Engine v{VERSION}")
        return

    config = CONFIG

    def _step() -> None:
        results, portfolio, regime = run_scan(
            config=config,
            debug=args.debug,
            no_intraday=args.no_intraday,
        )
        if not args.no_telegram and portfolio and regime:
            _send_alert(portfolio, regime, config)

    if args.calibrate:
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
