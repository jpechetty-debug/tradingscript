"""
core/services.py
================
Service boundaries for the Sovereign Engine runtime.

This module turns the main runtime concerns into explicit collaborators:

- ``PersistenceService`` for stateful files
- ``MarketDataService`` for fetch + indicator preparation
- ``AlertService`` for outbound notifications
- ``ScanService`` for the end-to-end scan orchestration
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import pandas as pd

from .backtest import OverallStats, WalkForwardResult, walk_forward
from .cache import ScanCache
from .config import CONFIG, IST, MarketRegimeType, SystemConfig
from .data_provider import fetch_daily_batch
from .factors import DEFAULT_WEIGHTS, calibrate_ic_weights
from .indicators import add_indicators
from .portfolio import optimize_portfolio
from .regime import (
    MarketRegime,
    RegimeTracker,
    classify_regime,
    confidence_position_scale,
    compute_breadth,
    compute_sector_rs,
)
from .runtime_paths import RUNTIME_PATHS, RuntimePaths, ensure_parent, ensure_runtime_dirs, resolve_artifact_path
from .scorer import TickerResult, calibrate_platt, score_ticker
from .telemetry import ScanMetrics
from .universe import ALL_TICKERS, N_SECTORS
from utils.messaging import send_telegram


log = logging.getLogger("sovereign.services")


class ProbabilityGate(Protocol):
    def threshold(self, regime: Any) -> float: ...


class CapitalFractionScaler(Protocol):
    def capital_fraction(self, current_nav: float, regime: Any) -> float: ...


class FactorWeightProvider(Protocol):
    def current_weights(self) -> dict[str, float]: ...


class SummaryAlerter(Protocol):
    def send_daily_summary(
        self,
        regime: str,
        top_picks: list[dict[str, Any]],
        current_nav: Optional[float] = None,
    ) -> Any: ...


CurrentNavProvider = Callable[[], Optional[float]]


@dataclass
class ScanState:
    """
    Mutable scan-session state.

    This is intentionally scoped to one scan cycle. The one stateful object that
    may outlive a cycle is ``RegimeTracker``, which callers inject separately.
    """

    platt_a: float
    platt_b: float
    platt_from_file: bool = False
    weights_calibrated: bool = False
    factor_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    ic_history: dict[str, Any] = field(default_factory=dict)
    regime_locked: bool = False
    cache: ScanCache = field(default_factory=ScanCache)


@dataclass(frozen=True)
class PreparedScanData:
    raw_data: dict[str, pd.DataFrame]
    processed: dict[str, pd.DataFrame]
    bench_series: pd.Series
    sector_rs: dict[str, float]
    sector_ranks: dict[str, int]


@dataclass(frozen=True)
class PortfolioStateSnapshot:
    current_nav: float
    peak_nav: Optional[float] = None


def _rank_sectors(sector_rs: dict[str, float]) -> dict[str, int]:
    sorted_sectors = sorted(sector_rs.items(), key=lambda item: item[1], reverse=True)
    return {sector: rank + 1 for rank, (sector, _) in enumerate(sorted_sectors)}


def passes_static_filters(df: pd.DataFrame, config: SystemConfig) -> bool:
    """Check ADV and structural filters before spending scorer time."""
    if len(df) < 200:
        return False

    close = df["Close"]
    volume = df["Volume"]
    turnover = (close * volume).tail(20).mean()
    if turnover < config.ADV_TURNOVER_FLOOR:
        return False

    if config.USE_EMA200_FILTER:
        ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        if close.iloc[-1] < ema200:
            return False

    return True


class PersistenceService:
    def __init__(self, paths: RuntimePaths = RUNTIME_PATHS) -> None:
        self.paths = paths
        ensure_runtime_dirs(self.paths)

    def create_scan_state(
        self,
        config: SystemConfig,
    ) -> ScanState:
        platt_a, platt_b, from_file = self.load_platt(config)
        state = ScanState(platt_a=platt_a, platt_b=platt_b, platt_from_file=from_file)
        state.cache.clear()
        return state

    def load_platt(self, config: SystemConfig) -> tuple[float, float, bool]:
        for candidate in self._candidate_paths(self.paths.platt_calibration_file, "platt_calibration.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                return float(payload["A"]), float(payload["B"]), True
            except Exception as exc:
                log.warning("Could not load %s: %s; using config defaults.", candidate, exc)
        return config.PLATT_A, config.PLATT_B, False

    def save_platt(self, a: float, b: float) -> None:
        payload = {
            "A": a,
            "B": b,
            "fitted_at": datetime.now(IST).isoformat(),
        }
        target = ensure_parent(self.paths.platt_calibration_file)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Platt params saved to %s: A=%.4f B=%.4f", target, a, b)

    def load_trade_log(self) -> list[dict[str, Any]]:
        for candidate in self._candidate_paths(self.paths.trade_log_file, "trade_log.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    return [entry for entry in payload if isinstance(entry, dict)]
            except Exception as exc:
                log.error("Could not load trade log %s: %s", candidate, exc)
                return []
        return []

    def load_portfolio_state(self) -> Optional[PortfolioStateSnapshot]:
        for candidate in self._candidate_paths(self.paths.portfolio_state_file, "portfolio_state.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    continue
                current_nav = float(payload["current_nav"])
                peak_raw = payload.get("peak_nav")
                peak_nav = float(peak_raw) if peak_raw is not None else None
                if current_nav <= 0:
                    raise ValueError("current_nav must be positive")
                if peak_nav is not None and peak_nav <= 0:
                    raise ValueError("peak_nav must be positive when provided")
                return PortfolioStateSnapshot(current_nav=current_nav, peak_nav=peak_nav)
            except Exception as exc:
                log.warning("Could not load portfolio state %s: %s", candidate, exc)
                return None
        return None

    def save_portfolio_state(self, current_nav: float, peak_nav: Optional[float] = None) -> None:
        payload: dict[str, Any] = {
            "current_nav": float(current_nav),
            "updated_at": datetime.now(IST).isoformat(),
        }
        if peak_nav is not None:
            payload["peak_nav"] = float(peak_nav)
        target = ensure_parent(self.paths.portfolio_state_file)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Portfolio state saved to %s", target)

    def artifact_path(self, output_path: str | Path) -> Path:
        return ensure_parent(resolve_artifact_path(output_path, self.paths))

    def _candidate_paths(self, preferred: Path, legacy_name: str) -> list[Path]:
        legacy = self.paths.root / legacy_name
        if legacy == preferred:
            return [preferred]
        return [preferred, legacy]


class MarketDataService:
    def __init__(
        self,
        fetcher: Callable[[list[str], SystemConfig], dict[str, pd.DataFrame]] = fetch_daily_batch,
    ) -> None:
        self._fetcher = fetcher

    def fetch_universe(
        self,
        tickers: list[str],
        config: SystemConfig,
        metrics: Optional[ScanMetrics] = None,
    ) -> dict[str, pd.DataFrame]:
        started = time.monotonic()
        raw_data = self._fetcher(tickers, config)
        if metrics is not None:
            metrics.record_fetch(
                n_ok=len(raw_data),
                n_fail=len(tickers) + 1 - len(raw_data),
                elapsed_s=time.monotonic() - started,
            )
        return raw_data

    def prepare_scan_data(
        self,
        tickers: list[str],
        config: SystemConfig,
        metrics: Optional[ScanMetrics] = None,
    ) -> PreparedScanData:
        raw_data = self.fetch_universe(tickers, config, metrics=metrics)
        if not raw_data:
            raise ValueError("No data fetched.")

        processed: dict[str, pd.DataFrame] = {}
        ind_fail = 0
        started = time.monotonic()
        for ticker, df in raw_data.items():
            try:
                processed[ticker] = add_indicators(df, config)
            except Exception:
                ind_fail += 1
                log.debug("Indicator error for %s.", ticker, exc_info=True)

        if metrics is not None:
            metrics.record_indicators(
                n_ok=len(processed),
                n_fail=ind_fail,
                elapsed_s=time.monotonic() - started,
            )

        bench_key = config.BENCHMARK
        if bench_key not in processed:
            raise ValueError(f"Benchmark {bench_key} not in processed data.")

        bench_series = processed[bench_key]["Close"]
        sector_rs = compute_sector_rs(processed, bench_series, config)
        sector_ranks = _rank_sectors(sector_rs)

        return PreparedScanData(
            raw_data=raw_data,
            processed=processed,
            bench_series=bench_series,
            sector_rs=sector_rs,
            sector_ranks=sector_ranks,
        )


class AlertService:
    def __init__(
        self,
        *,
        version: str,
        messenger: Callable[[str, str, str], bool] = send_telegram,
        alerter: Optional[SummaryAlerter] = None,
        current_nav_provider: Optional[CurrentNavProvider] = None,
    ) -> None:
        self._version = version
        self._messenger = messenger
        self._alerter = alerter
        self._current_nav_provider = current_nav_provider

    def send_portfolio_summary(
        self,
        portfolio: list[TickerResult],
        regime: MarketRegime,
        config: SystemConfig,
    ) -> None:
        if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
            log.debug("Telegram credentials missing in config.")
            return

        top = [result for result in portfolio if result.prob_win >= config.TELEGRAM_ALERT_MIN_PROB]
        if not top:
            return

        if self._alerter is not None:
            self._alerter.send_daily_summary(
                regime.label,
                [result.__dict__ for result in top],
                current_nav=self._resolve_current_nav(),
            )
            return

        lines = [
            f"<b>Sovereign v{self._version}</b> | {regime.label} | {datetime.now(IST).strftime('%H:%M IST')}"
        ]
        for result in top[: config.TELEGRAM_ALERT_TOP_N]:
            lines.append(
                f"<b>{result.ticker}</b> {result.direction} | P={result.prob_win:.0%} | "
                f"E(R)={result.expectancy_r:.2f} | Entry {result.entry} | SL {result.stop} | "
                f"T1 {result.t1} | {result.shares} shares"
            )
        self._messenger("\n".join(lines), str(config.TELEGRAM_BOT_TOKEN), config.TELEGRAM_CHAT_ID)

    def _resolve_current_nav(self) -> Optional[float]:
        if self._current_nav_provider is None:
            return None
        try:
            current_nav = self._current_nav_provider()
            return float(current_nav) if current_nav is not None and current_nav > 0 else None
        except Exception:
            log.debug("Current NAV provider failed while building alert summary.", exc_info=True)
            return None


class ScanService:
    def __init__(
        self,
        *,
        version: str,
        data_service: Optional[MarketDataService] = None,
        persistence: Optional[PersistenceService] = None,
        probability_gate: Optional[ProbabilityGate] = None,
        capital_scaler: Optional[CapitalFractionScaler] = None,
        factor_calibrator: Optional[FactorWeightProvider] = None,
        alerter: Optional[SummaryAlerter] = None,
        current_nav_provider: Optional[CurrentNavProvider] = None,
        default_current_nav: float = 1_000_000.0,
    ) -> None:
        self._version = version
        self._data_service = data_service or MarketDataService()
        self._persistence = persistence or PersistenceService()
        self._probability_gate = probability_gate
        self._capital_scaler = capital_scaler
        self._factor_calibrator = factor_calibrator
        self._alerter = alerter
        self._current_nav_provider = current_nav_provider
        self._default_current_nav = default_current_nav

    def create_alert_service(
        self,
        messenger: Callable[[str, str, str], bool] = send_telegram,
    ) -> AlertService:
        return AlertService(
            version=self._version,
            messenger=messenger,
            alerter=self._alerter,
            current_nav_provider=self._current_nav_provider,
        )

    def scan(
        self,
        config: SystemConfig = CONFIG,
        *,
        debug: bool = False,
        regime_tracker: Optional[RegimeTracker] = None,
        regime_override: Optional[str] = None,
        no_ema_filter: bool = False,
        force_score: bool = False,
    ) -> tuple[list[TickerResult], list[TickerResult], Optional[MarketRegime]]:
        state = self._persistence.create_scan_state(config)
        metrics = ScanMetrics()

        if self._factor_calibrator is not None:
            state.factor_weights = self._factor_calibrator.current_weights()
            log.info(
                "Using dynamic factor weights: %s",
                {key: round(value, 3) for key, value in state.factor_weights.items()},
            )

        config = replace(config, PLATT_A=state.platt_a, PLATT_B=state.platt_b)
        log.info(
            "Sovereign Engine v%s | %s",
            self._version,
            datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"),
        )

        try:
            prepared = self._data_service.prepare_scan_data(ALL_TICKERS, config, metrics=metrics)
        except ValueError as exc:
            log.error("%s", exc)
            return [], [], None

        breadth = compute_breadth(prepared.processed, config)
        state.regime_locked = config.is_regime_locked()
        if state.regime_locked:
            log.info(
                "Regime lock active during opening noise window (%d min).",
                config.REGIME_LOCK_MINUTES,
            )

        tracker = regime_tracker if regime_tracker is not None else RegimeTracker()
        regime = classify_regime(
            prepared.processed,
            breadth,
            tracker,
            config,
            locked=state.regime_locked,
            sector_rs=prepared.sector_rs,
        )
        regime = self._apply_regime_override(regime, regime_override)

        session = config.session_from_time()
        scoring_config = self._apply_regime_probability_gate(config, regime, debug=debug)

        metrics.record_regime(regime)
        self._log_regime(regime, config)

        if not regime.is_tradeable():
            log.warning("PANIC regime - no new positions.")
            return [], [], regime

        effective_config = scoring_config
        if no_ema_filter:
            effective_config = replace(effective_config, USE_EMA200_FILTER=False)

        all_results, score_elapsed = self._score_candidates(
            prepared=prepared,
            regime=regime,
            config=effective_config,
            state=state,
            session=session,
            debug=debug,
            force_score=force_score,
        )

        metrics.record_score(
            n_passed=len(all_results),
            n_total=max(len(prepared.processed) - 1, 0),
            elapsed_s=score_elapsed,
        )
        log.info("%d tickers passed all gates.", len(all_results))

        self._maybe_recalibrate_ic_weights(
            all_results=all_results,
            prepared=prepared,
            config=config,
            state=state,
        )

        corr_matrix = state.cache.corr_matrix(prepared.processed, config)
        portfolio = optimize_portfolio(all_results, config, corr_matrix)

        metrics.record_portfolio(portfolio)
        metrics.emit_summary()
        state.cache.log_stats()
        log.info("Portfolio: %d positions selected.", len(portfolio))
        for result in portfolio:
            log.info(
                "  %s | %s | P=%.2f | E(R)=%.3f | RR=%.1fx | %d shares | %.0f INR risk",
                result.ticker,
                result.direction,
                result.prob_win,
                result.expectancy_r,
                result.rr_t1,
                result.shares,
                result.risk_inr,
            )

        return all_results, portfolio, regime

    def run_calibration(self) -> None:
        trades = self._persistence.load_trade_log()
        if not trades:
            log.error("Trade log not found or empty at %s", self._persistence.paths.trade_log_file)
            return

        composites: list[float] = []
        outcomes: list[int] = []
        for trade in trades:
            factors = trade.get("factors")
            pnl = trade.get("pnl")
            if not isinstance(factors, dict) or pnl is None:
                continue

            composite = trade.get("composite")
            if composite is None:
                composite = sum(
                    float(factors.get(key, 0.0)) * DEFAULT_WEIGHTS.get(key, 0.0)
                    for key in DEFAULT_WEIGHTS
                )
            composites.append(float(composite))
            outcomes.append(1 if float(pnl) > 0 else 0)

        if len(composites) < 5:
            log.warning("Insufficient data for calibration (%d samples).", len(composites))
            return

        a, b = calibrate_platt(composites, outcomes)
        self._persistence.save_platt(a, b)

    def run_backtest(
        self,
        *,
        config: SystemConfig = CONFIG,
        train_days: int = 120,
        test_days: int = 20,
        step_days: int = 10,
        out_csv: str = "backtest_results.csv",
        direction: str = "LONG",
        debug: bool = False,
    ) -> WalkForwardResult:
        if debug:
            logging.getLogger("sovereign").setLevel(logging.DEBUG)

        started = time.monotonic()
        raw_data = self._data_service.fetch_universe(ALL_TICKERS, config)
        elapsed = time.monotonic() - started
        log.info("Data fetched: %d symbols in %.1fs", len(raw_data), elapsed)

        if not raw_data:
            log.error("No data fetched - aborting backtest.")
            empty = WalkForwardResult(
                trades=[],
                fold_stats=[],
                overall=OverallStats(
                    n_folds=0,
                    n_trades=0,
                    hit_rate=0.0,
                    mean_r=0.0,
                    sharpe=0.0,
                    max_dd=0.0,
                    total_r=0.0,
                    profit_factor=0.0,
                    expectancy_r=0.0,
                    fold_stats=[],
                ),
            )
            return empty

        results = walk_forward(
            raw_data=raw_data,
            config=config,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            direction=direction,
        )

        output_path = self._persistence.artifact_path(out_csv)
        if results.trades:
            results.to_csv(str(output_path))
        else:
            log.warning("No trades generated - check thresholds and data quality.")
        return results

    def _score_candidates(
        self,
        *,
        prepared: PreparedScanData,
        regime: MarketRegime,
        config: SystemConfig,
        state: ScanState,
        session: str,
        debug: bool,
        force_score: bool,
    ) -> tuple[list[TickerResult], float]:
        started = time.monotonic()
        all_results: list[TickerResult] = []
        current_nav = self._resolve_current_nav()
        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures: dict[Any, str] = {}
            for ticker, df in prepared.processed.items():
                if ticker == config.BENCHMARK:
                    continue
                if not passes_static_filters(df, config):
                    continue

                capital_fraction = confidence_position_scale(regime.confidence)
                if self._capital_scaler is not None:
                    try:
                        capital_fraction *= float(
                            self._capital_scaler.capital_fraction(current_nav, regime.regime)
                        )
                    except Exception:
                        log.debug("Injected capital scaler failed for %s.", ticker, exc_info=debug)

                future = executor.submit(
                    score_ticker,
                    ticker=ticker,
                    daily_df=df,
                    bench=prepared.bench_series,
                    sector_ranks=prepared.sector_ranks,
                    sector_rs=prepared.sector_rs,
                    session=session,
                    regime=regime,
                    config=config,
                    factor_weights=state.factor_weights,
                    capital_fraction=capital_fraction,
                    debug=debug,
                    force_score=force_score,
                )
                futures[future] = ticker

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                except Exception as exc:
                    log.error("score_ticker error for %s: %s", ticker, exc, exc_info=True)

        elapsed = time.monotonic() - started
        log.debug("Scoring completed in %.3fs.", elapsed)
        return all_results, elapsed

    def _resolve_current_nav(self) -> float:
        if self._current_nav_provider is not None:
            try:
                current_nav = self._current_nav_provider()
                if current_nav is not None and current_nav > 0:
                    return float(current_nav)
            except Exception:
                log.warning("Current NAV provider failed; using default NAV %.0f.", self._default_current_nav)

        loader = getattr(self._persistence, "load_portfolio_state", None)
        if callable(loader):
            snapshot = loader()
            if snapshot is not None:
                return snapshot.current_nav

        return self._default_current_nav

    def _maybe_recalibrate_ic_weights(
        self,
        *,
        all_results: list[TickerResult],
        prepared: PreparedScanData,
        config: SystemConfig,
        state: ScanState,
    ) -> None:
        if len(all_results) < config.ICIR_MIN_OBS:
            return

        try:
            new_weights = calibrate_ic_weights(
                results=all_results,
                processed=prepared.processed,
                bench=prepared.bench_series,
                sector_ranks=prepared.sector_ranks,
                n_sectors=N_SECTORS,
                config=config,
            )
            state.factor_weights = new_weights
            state.weights_calibrated = True
            log.info("IC weights optimized: %s", {key: round(value, 4) for key, value in new_weights.items()})
        except Exception as exc:
            log.warning("IC calibration failed: %s", exc)

    def _apply_regime_probability_gate(
        self,
        config: SystemConfig,
        regime: MarketRegime,
        *,
        debug: bool,
    ) -> SystemConfig:
        if self._probability_gate is None:
            return config

        try:
            threshold = float(self._probability_gate.threshold(regime.regime))
            log.info("Regime probability gate active: %s -> P(win) >= %.2f", regime.label, threshold)
            return replace(config, MIN_PROB_WIN=threshold)
        except Exception:
            log.warning("Regime probability gate failed; using static threshold.", exc_info=debug)
            return config

    def _apply_regime_override(
        self,
        regime: MarketRegime,
        regime_override: Optional[str],
    ) -> MarketRegime:
        if not regime_override:
            return regime

        try:
            override_type = MarketRegimeType(regime_override.upper())
        except ValueError:
            log.error("Invalid override regime: %s - ignoring.", regime_override)
            return regime

        log.warning("REGIME OVERRIDE ACTIVE: %s (was %s)", override_type.value, regime.label)
        return MarketRegime(
            regime=override_type,
            breadth=regime.breadth,
            adx_median=regime.adx_median,
            atr_ratio=regime.atr_ratio,
            confidence=1.0,
            confirmed=True,
            breadth_delta=regime.breadth_delta,
            sector_concentration=regime.sector_concentration,
            regime_locked=False,
        )

    def _log_regime(self, regime: MarketRegime, config: SystemConfig) -> None:
        log.info(
            "Breadth: %.0f%% (d=%.2f) | Regime: %s (%s) | Conf: %.2f | ADX: %.1f | "
            "ATR ratio: %.2f | Sector conc: %.0f%% | %s",
            regime.breadth * 100,
            regime.breadth_delta,
            regime.label,
            "CONFIRMED" if regime.confirmed else f"{config.REGIME_CONFIRM_BARS} bar CONFIRM REQ",
            regime.confidence,
            regime.adx_median,
            regime.atr_ratio,
            regime.sector_concentration * 100,
            "LOCKED" if regime.regime_locked else "live",
        )
