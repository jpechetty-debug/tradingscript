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

import inspect
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .backtest import TransactionCostModel

import pandas as pd

from .backtest import OverallStats, WalkForwardResult, walk_forward
from .cache import ScanCache
from .config import CONFIG, IST, MarketRegimeType, SystemConfig
from .database import SqliteDatabase
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
from .scorer import (
    CandidateContext,
    TickerResult,
    apply_cohort_factor_ranking,
    calibrate_platt,
    score_candidate_pass1,
    score_candidate_pass2,
    score_ticker,
)
from .telemetry import ScanMetrics
from .universe import ALL_TICKERS, N_SECTORS
from utils.messaging import send_telegram

_DEFAULT_SCORE_TICKER = score_ticker


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
    open_positions: set[str] = field(default_factory=set)


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


def passes_static_filters(df: pd.DataFrame, config: SystemConfig, min_bars: int = 50) -> bool:
    """Check ADV and minimum length before spending scorer time."""
    if len(df) < min_bars:
        return False

    close = df["Close"]
    volume = df["Volume"]
    turnover = (close * volume).tail(20).mean()
    if turnover < config.ADV_TURNOVER_FLOOR:
        return False

    return True


class PersistenceService:
    def __init__(self, paths: RuntimePaths = RUNTIME_PATHS) -> None:
        self.paths = paths
        ensure_runtime_dirs(self.paths)
        self.db = SqliteDatabase(self.paths.state_db_file)

    def create_scan_state(
        self,
        config: SystemConfig,
    ) -> ScanState:
        platt_a, platt_b, from_file = self.load_platt(config)
        open_pos_map = self.load_open_positions()
        state = ScanState(
            platt_a=platt_a,
            platt_b=platt_b,
            platt_from_file=from_file,
            open_positions=set(open_pos_map.keys()),
        )
        state.cache.clear()
        return state

    def load_platt(self, config: SystemConfig) -> tuple[float, float, bool]:
        latest = self.db.fetch_latest_platt()
        if latest is not None:
            a, b, _, version = latest
            if version >= 2:
                return a, b, True
            log.warning("Discarding legacy Platt calibration (version %s < 2); falling back to config defaults.", version)

        for candidate in self._candidate_paths(self.paths.platt_calibration_file, "platt_calibration.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                version = int(payload.get("version", 1))
                if version < 2:
                    log.warning("Discarding legacy Platt calibration JSON (version %s < 2): %s", version, candidate)
                    continue
                a, b = float(payload["A"]), float(payload["B"])
                ts = str(payload.get("fitted_at") or datetime.now(IST).isoformat())
                self.db.upsert_platt(a, b, fitted_at=ts, version=version)
                return a, b, True
            except Exception as exc:
                log.warning("Could not load %s: %s; using config defaults.", candidate, exc)
        return config.PLATT_A, config.PLATT_B, False

    def save_platt(self, a: float, b: float) -> None:
        ts = datetime.now(IST).isoformat()
        self.db.upsert_platt(a, b, fitted_at=ts, version=2)
        payload = {
            "A": a,
            "B": b,
            "fitted_at": ts,
            "version": 2,
        }
        target = ensure_parent(self.paths.platt_calibration_file)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Platt params saved to DB and %s: A=%.4f B=%.4f (v2)", target, a, b)

    def load_trade_log(self) -> list[dict[str, Any]]:
        trades = self.db.fetch_trades()
        if trades:
            return trades

        for candidate in self._candidate_paths(self.paths.trade_log_file, "trade_log.json"):
            if not candidate.exists():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    entries = [entry for entry in payload if isinstance(entry, dict)]
                    is_root_legacy = (
                        hasattr(self.paths, "root")
                        and candidate.resolve() == (self.paths.root / "trade_log.json").resolve()
                    )
                    if is_root_legacy and entries and "ticker" not in entries[-1] and "pnl" in entries[-1]:
                        ts = datetime.now(IST).strftime("%Y%m%d_%H%M%S")
                        archive_path = candidate.parent / f"trade_log_legacy_{ts}.json"
                        candidate.rename(archive_path)
                        log.info("Archived legacy trade log (missing ticker) to %s. Starting fresh.", archive_path)
                        return []

                    if entries:
                        self.db.insert_trades(entries)
                    return entries
            except Exception as exc:
                log.error("Could not load trade log %s: %s", candidate, exc)
                return []
        return []

    def save_trade_log(self, trades: list[dict[str, Any]]) -> None:
        target = ensure_parent(self.paths.trade_log_file)
        target.write_text(json.dumps(trades, indent=2, default=str), encoding="utf-8")
        self.db.insert_trades(trades)

    def append_trade(self, trade: dict[str, Any]) -> None:
        self.db.insert_trade(trade)
        existing = self.load_trade_log()
        target = ensure_parent(self.paths.trade_log_file)
        target.write_text(json.dumps(existing[-2000:], indent=2, default=str), encoding="utf-8")

    def set_killswitch(self, is_killed: bool) -> None:
        self.db.set_killswitch(is_killed)

    def get_killswitch(self) -> bool:
        return self.db.get_killswitch()

    def load_portfolio_state(self) -> Optional[PortfolioStateSnapshot]:
        latest = self.db.fetch_latest_portfolio_state()
        if latest is not None and latest.get("current_nav", 0) > 0:
            return PortfolioStateSnapshot(
                current_nav=float(latest["current_nav"]),
                peak_nav=float(latest["peak_nav"]) if latest.get("peak_nav") is not None else None,
            )

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
                self.db.upsert_portfolio_state(current_nav, peak_nav, updated_at=payload.get("updated_at"))
                return PortfolioStateSnapshot(current_nav=current_nav, peak_nav=peak_nav)
            except Exception as exc:
                log.warning("Could not load portfolio state %s: %s", candidate, exc)
                return None
        return None

    def save_portfolio_state(self, current_nav: float, peak_nav: Optional[float] = None) -> None:
        ts = datetime.now(IST).isoformat()
        self.db.upsert_portfolio_state(current_nav, peak_nav, updated_at=ts)
        payload: dict[str, Any] = {
            "current_nav": float(current_nav),
            "updated_at": ts,
        }
        if peak_nav is not None:
            payload["peak_nav"] = float(peak_nav)
        target = ensure_parent(self.paths.portfolio_state_file)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Portfolio state saved to DB and %s", target)

    def load_open_positions(self) -> dict[str, dict[str, Any]]:
        return self.db.fetch_open_positions()

    def save_open_positions(self, positions: list[dict[str, Any]]) -> None:
        self.db.sync_open_positions(positions)

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
        self._last_alerted: dict[str, datetime] = {}

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
        now = datetime.now(IST)
        cutoff = getattr(config, "TELEGRAM_DEDUP_HOURS", 4) * 3600
        top = [r for r in top if (now - self._last_alerted.get(
            f"{r.ticker}:{getattr(r, 'trade_horizon', 'SWING')}", now.replace(year=2000)
        )).total_seconds() >= cutoff]
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
        for result in top:
            self._last_alerted[f"{result.ticker}:{getattr(result, 'trade_horizon', 'SWING')}"] = now

    def _resolve_current_nav(self) -> Optional[float]:
        if self._current_nav_provider is None:
            return None
        try:
            current_nav = self._current_nav_provider()
            return float(current_nav) if current_nav is not None and current_nav > 0 else None
        except Exception:
            log.debug("Current NAV provider failed while building alert summary.", exc_info=True)
            return None


class ScanOutput(tuple):
    """Structured scan output backward-compatible with 3-tuple (candidates, portfolio, regime)."""

    candidates: list[TickerResult]
    portfolio: list[TickerResult]
    regime: Optional[MarketRegime]
    sector_rs: dict[str, float]

    def __new__(
        cls,
        candidates: list[TickerResult],
        portfolio: list[TickerResult],
        regime: Optional[MarketRegime],
        sector_rs: Optional[dict[str, float]] = None,
    ) -> ScanOutput:
        instance = super().__new__(cls, (candidates, portfolio, regime))
        instance.candidates = candidates
        instance.portfolio = portfolio
        instance.regime = regime
        instance.sector_rs = dict(sector_rs) if sector_rs is not None else {}
        return instance


def score_universe(
    processed: dict[str, pd.DataFrame],
    bench: pd.Series,
    sector_rs: dict[str, float],
    regime: MarketRegime,
    config: SystemConfig,
    weights: Optional[dict[str, float]] = None,
    *,
    open_positions: Optional[set[str]] = None,
    open_pos_map: Optional[dict[str, dict[str, Any]]] = None,
    session: str = "CLOSING_TREND",
    capital_fraction: float = 1.0,
    debug: bool = False,
    no_intraday: bool = False,
    force_score: bool = False,
    allow_watchlist: bool = True,
    capital_scaler: Any = None,
    current_nav: float = 1_000_000.0,
    sector_ranks: Optional[dict[str, int]] = None,
    direction: str = "BOTH",
    min_bars: int = 50,
) -> list[TickerResult]:
    """
    Pure, shared candidate scoring pipeline across live scan and walk-forward backtest.

    Applies:
      1. Static filters (ADV turnover and minimum bar count)
      2. Pass 1: Multi-factor scoring, trend/regime veto, ATR target calculation
      3. Cross-sectional cohort factor ranking
      4. Pass 2: Probability gating with hysteresis and Kelly position sizing
    """
    if sector_ranks is None:
        sector_ranks = _rank_sectors(sector_rs)

    open_pos: set[str] = set(open_positions or ())
    pos_map: dict[str, dict[str, Any]] = open_pos_map or {}
    factor_weights = weights if weights is not None else DEFAULT_WEIGHTS

    from .cache import SCAN_CACHE
    SCAN_CACHE.clear()

    base_cap_frac = confidence_position_scale(regime.confidence) * capital_fraction
    if capital_scaler is not None:
        try:
            base_cap_frac *= float(capital_scaler.capital_fraction(current_nav, regime.regime))
        except Exception:
            log.debug("Injected capital scaler failed.", exc_info=debug)

    pass1_candidates: list[CandidateContext] = []

    candidate_items: list[tuple[str, pd.DataFrame]] = []
    for ticker, df in processed.items():
        if ticker == config.BENCHMARK:
            continue
        clean_ticker = ticker.replace(".NS", "")
        is_open = (ticker in open_pos) or (clean_ticker in open_pos)
        if not is_open and not passes_static_filters(df, config, min_bars=min_bars):
            continue
        candidate_items.append((ticker, df))

    max_workers = config.MAX_WORKERS if len(candidate_items) > 2 else 1
    with ThreadPoolExecutor(max_workers=max_workers) as pass1_executor:
        pass1_futures: dict[Any, str] = {}
        for ticker, df in candidate_items:
            clean_ticker = ticker.replace(".NS", "")
            is_open = (ticker in open_pos) or (clean_ticker in open_pos)
            pos_info = pos_map.get(ticker) or pos_map.get(clean_ticker) or {}
            held_dir = pos_info.get("direction")

            fut = pass1_executor.submit(
                score_candidate_pass1,
                ticker=ticker,
                daily_df=df,
                bench=bench,
                sector_ranks=sector_ranks,
                sector_rs=sector_rs,
                session=session,
                regime=regime,
                config=config,
                factor_weights=factor_weights,
                capital_fraction=base_cap_frac,
                debug=debug,
                no_intraday=no_intraday,
                force_score=force_score,
                is_open_position=is_open,
                held_direction=held_dir,
            )
            pass1_futures[fut] = ticker

        for fut in as_completed(pass1_futures):
            ticker = pass1_futures[fut]
            try:
                cand = fut.result()
                if cand is not None:
                    if direction == "BOTH" or cand.direction == direction:
                        pass1_candidates.append(cand)
            except Exception as exc:
                log.error("score_candidate_pass1 error for %s: %s", ticker, exc, exc_info=True)

    # Cross-sectional cohort factor ranking
    cohort_rank_weight = getattr(config, "COHORT_RANK_WEIGHT", 0.40)
    cohort_min_obs = getattr(config, "COHORT_MIN_OBS", 10)
    cohort_full_obs = getattr(config, "COHORT_FULL_OBS", 30)
    ranked_candidates: list[CandidateContext] = apply_cohort_factor_ranking(
        pass1_candidates,
        cohort_rank_weight=cohort_rank_weight,
        cohort_min_obs=cohort_min_obs,
        cohort_full_obs=cohort_full_obs,
    )

    # Pass 2: Probability gating with hysteresis and Kelly position sizing
    all_results: list[TickerResult] = []
    for cand in ranked_candidates:
        clean_ticker = cand.ticker.replace(".NS", "")
        is_open = (cand.ticker in open_pos) or (clean_ticker in open_pos)
        try:
            res = score_candidate_pass2(
                candidate=cand,
                config=config,
                is_open_position=is_open,
                allow_watchlist=allow_watchlist,
                debug=debug,
            )
            if res is not None:
                if direction == "BOTH" or res.direction == direction:
                    all_results.append(res)
        except Exception as exc:
            log.error("score_candidate_pass2 error for %s: %s", cand.ticker, exc, exc_info=True)

    return all_results


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
        self.last_sector_rs: dict[str, float] = {}
        self.last_regime_info: Optional[MarketRegime] = None
        self._recent_alerts: list[dict[str, float]] = []  # [{ticker: composite}, ...]

    @staticmethod
    def _validated_factor_weights(weights: dict[str, float]) -> dict[str, float]:
        """Reject unsafe runtime calibration instead of starving signal factors."""
        expected = set(DEFAULT_WEIGHTS)
        try:
            cleaned = {name: float(weights[name]) for name in expected}
        except (KeyError, TypeError, ValueError):
            # Test/injected providers may expose a deliberately partial map.
            return dict(weights)
        total = sum(cleaned.values())
        min_w = 0.03  # matches MIN_FACTOR_WEIGHT config default
        if total <= 0 or any(value < min_w or value > 0.40 for value in cleaned.values()):
            log.warning("Discarding unsafe dynamic factor weights; using balanced defaults.")
            return dict(DEFAULT_WEIGHTS)
        return {name: value / total for name, value in cleaned.items()}

    def get_last_sector_rs(self) -> dict[str, float]:
        return dict(self.last_sector_rs)

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
        no_intraday: bool = False,
        force_score: bool = False,
    ) -> tuple[list[TickerResult], list[TickerResult], Optional[MarketRegime]]:
        state = self._persistence.create_scan_state(config)
        metrics = ScanMetrics()

        if self._factor_calibrator is not None:
            state.factor_weights = self._validated_factor_weights(self._factor_calibrator.current_weights())
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
            return ScanOutput([], [], None, sector_rs={})

        self.last_sector_rs = dict(prepared.sector_rs)
        self._monitor_open_position_stops(prepared.processed, state)

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
        self.last_regime_info = regime

        session = config.session_from_time()
        scoring_config = self._apply_regime_probability_gate(config, regime, debug=debug)

        metrics.record_regime(regime)
        self._log_regime(regime, config)

        if not regime.is_tradeable():
            log.warning("PANIC regime - no new positions.")
            return ScanOutput([], [], regime, sector_rs=self.last_sector_rs)

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
            no_intraday=no_intraday,
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
            regime_label=regime.label if regime else None,
        )

        corr_matrix = state.cache.corr_matrix(prepared.processed, config)
        portfolio_candidates = [r for r in all_results if not getattr(r, "is_watchlist", False)]
        portfolio = optimize_portfolio(portfolio_candidates, config, corr_matrix)

        # ── Duplicate alert suppression ───────────────────────────────────────
        lookback = getattr(config, "DUPLICATE_LOOKBACK_SCANS", 5)
        delta = getattr(config, "DUPLICATE_COMPOSITE_DELTA", 0.05)
        if self._recent_alerts and portfolio:
            filtered: list[TickerResult] = []
            for r in portfolio:
                ticker = getattr(r, "ticker", "")
                composite = getattr(r, "composite", 0.0)
                was_recent = any(
                    ticker in scan and abs(composite - scan[ticker]) < delta
                    for scan in self._recent_alerts[-lookback:]
                )
                if was_recent and not getattr(r, "is_held", False):
                    log.debug(
                        "%s: suppressed duplicate alert (composite %.3f, delta < %.3f)",
                        ticker, composite, delta,
                    )
                else:
                    filtered.append(r)
            portfolio = filtered

        # Record this scan's portfolio for future dedup checks
        current_scan_map: dict[str, float] = {
            getattr(r, "ticker", ""): getattr(r, "composite", 0.0)
            for r in portfolio
        }
        self._recent_alerts.append(current_scan_map)
        if len(self._recent_alerts) > lookback + 1:
            self._recent_alerts = self._recent_alerts[-(lookback + 1):]

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

        # Synchronize open_positions with current portfolio selections
        # Retain all selected portfolio items + any surviving held positions from all_results
        retained_tickers = {getattr(r, "ticker", "") for r in portfolio}
        for r in all_results:
            if getattr(r, "is_held", False) or "HeldPos" in getattr(r, "reasons", []):
                retained_tickers.add(getattr(r, "ticker", ""))

        open_pos_payload: list[dict[str, Any]] = [
            {
                "ticker": getattr(result, "ticker", ""),
                "direction": getattr(result, "direction", "LONG"),
                "entry": getattr(result, "entry", getattr(result, "close", 0.0)),
                "shares": getattr(result, "shares", 0),
                "stop": getattr(result, "stop", 0.0),
                "t1": getattr(result, "t1", 0.0),
                "prob_win": getattr(result, "prob_win", 0.0),
                "composite": getattr(result, "composite", 0.0),
                "trade_horizon": getattr(result, "trade_horizon", "SWING"),
                "factors": getattr(getattr(result, "factors", None), "as_dict", lambda: {})(),
            }
            for result in all_results
            if getattr(result, "ticker", "") in retained_tickers
        ]

        seen_t: set[str] = set()
        deduped_payload: list[dict[str, Any]] = []
        for p in open_pos_payload:
            t = str(p.get("ticker", ""))
            if t and t not in seen_t:
                seen_t.add(t)
                seen_t.add(t.replace(".NS", ""))
                deduped_payload.append(p)

        # Distinguish between evaluated held positions and those not evaluated due to data/fetch issues
        evaluated_tickers = {t.replace(".NS", "") for t in prepared.processed}.union(set(prepared.processed.keys()))
        loader = getattr(self._persistence, "load_open_positions", None)
        open_pos_map: dict[str, dict[str, Any]] = loader() if callable(loader) else {}

        for old_t in list(state.open_positions):
            clean_old_t = old_t.replace(".NS", "")
            if clean_old_t not in seen_t and old_t not in seen_t:
                was_evaluated = (old_t in evaluated_tickers) or (clean_old_t in evaluated_tickers)
                if not was_evaluated:
                    prior_record = open_pos_map.get(old_t) or open_pos_map.get(clean_old_t)
                    if prior_record:
                        deduped_payload.append(prior_record)
                        seen_t.add(old_t)
                        seen_t.add(clean_old_t)
                        log.warning(
                            "Held position %s was NOT evaluated this scan (data unavailable/fetch error); preserving open position.",
                            old_t,
                        )
                else:
                    log.info("Held position %s EXITED: Evaluated and fell below holding threshold or structural criteria.", old_t)

        saver = getattr(self._persistence, "save_open_positions", None)
        if callable(saver):
            try:
                saver(deduped_payload)
            except Exception as exc:
                log.warning("Could not sync open positions: %s", exc)

        return ScanOutput(all_results, portfolio, regime, sector_rs=self.last_sector_rs)

    def run_calibration(self, calib_offset: int = 60) -> None:
        # Prefer executed_trades (proper lifecycle) over legacy trade_log
        composites, outcomes = self._persistence.db.fetch_calibration_trades(min_samples=80)
        if composites and outcomes:
            log.info(
                "Calibrating from %d closed executed trades.",
                len(composites),
            )
            a, b = calibrate_platt(composites, outcomes, calib_offset=calib_offset)
            self._persistence.save_platt(a, b)
            return

        # Fallback: legacy trade_log (but warn that it's not validated)
        trades = self._persistence.load_trade_log()
        if not trades:
            log.warning(
                "No executed trades and no legacy trade log found. "
                "Using config-default Platt coefficients (A=%.1f, B=%.1f). "
                "Record real trades with outcomes to enable calibration.",
                -4.0, 2.0,
            )
            return

        composites_legacy: list[float] = []
        outcomes_legacy: list[int] = []
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
            composites_legacy.append(float(composite))
            outcomes_legacy.append(1 if float(pnl) > 0 else 0)

        if len(composites_legacy) < 5:
            log.warning("Insufficient data for calibration (%d samples).", len(composites_legacy))
            return

        a, b = calibrate_platt(composites_legacy, outcomes_legacy, calib_offset=calib_offset)
        self._persistence.save_platt(a, b)

    def run_backtest(
        self,
        *,
        config: SystemConfig = CONFIG,
        train_days: int = 120,
        test_days: int = 20,
        step_days: int = 20,
        out_csv: str = "backtest_results.csv",
        direction: str = "LONG",
        debug: bool = False,
        cost_model: "TransactionCostModel | None" = None,
        horizon_filter: str = "SWING",
    ) -> WalkForwardResult:
        from .backtest import DEFAULT_COST_MODEL

        resolved_cost = cost_model if cost_model is not None else DEFAULT_COST_MODEL

        # Load persisted Platt calibration to match live scan behavior
        platt_a, platt_b, from_file = self._persistence.load_platt(config)
        if from_file:
            log.info("Backtest using persisted Platt parameters: A=%.4f B=%.4f", platt_a, platt_b)
            config = replace(config, PLATT_A=platt_a, PLATT_B=platt_b)

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
            cost_model=resolved_cost,
            horizon_filter=horizon_filter,
        )

        output_path = self._persistence.artifact_path(out_csv)
        if results.trades:
            results.to_csv(str(output_path))
        else:
            log.warning("No trades generated - check thresholds and data quality.")
        return results
    def _monitor_open_position_stops(
        self,
        processed: dict[str, pd.DataFrame],
        state: ScanState,
    ) -> None:
        """
        Deterministic price-vs-level check for active open positions.
        If current bar breaches stop_loss or reaches target, execute trade exit,
        remove from DB and state.open_positions.
        """
        loader = getattr(self._persistence, "load_open_positions", None)
        if not callable(loader):
            return
        try:
            open_pos_map = loader()
        except Exception as exc:
            log.warning("Could not load open positions for stop monitor: %s", exc)
            return
        if not open_pos_map:
            return

        db = getattr(self._persistence, "db", None)
        deleter = getattr(db, "delete_open_position", None)

        for ticker, pos in open_pos_map.items():
            df = processed.get(ticker)
            if df is None:
                df = processed.get(f"{ticker}.NS")
            if df is None or df.empty:
                continue

            row = df.iloc[-1]
            open_p = float(row.get("Open", 0.0))
            close_p = float(row.get("Close", 0.0))
            low_p = float(row.get("Low", close_p))
            high_p = float(row.get("High", close_p))
            if open_p <= 0.0:
                open_p = close_p

            direction = str(pos.get("direction", "LONG")).upper()
            stop_loss = float(pos.get("stop_loss", 0.0))
            target = float(pos.get("target", 0.0))

            stopped_out = False
            target_hit = False

            if direction == "LONG":
                if stop_loss > 0 and (low_p <= stop_loss or close_p <= stop_loss):
                    stopped_out = True
                elif target > 0 and (high_p >= target or close_p >= target):
                    target_hit = True
            elif direction == "SHORT":
                if stop_loss > 0 and (high_p >= stop_loss or close_p >= stop_loss):
                    stopped_out = True
                elif target > 0 and (low_p <= target or close_p <= target):
                    target_hit = True

            if stopped_out:
                exit_fill = min(open_p, stop_loss) if direction == "LONG" else max(open_p, stop_loss)
                log.info(
                    "Held position %s EXITED: Stop-loss triggered (Fill=%.2f Low=%.2f Close=%.2f <= Stop=%.2f)",
                    ticker, exit_fill, low_p, close_p, stop_loss,
                )
                if callable(deleter):
                    deleter(ticker)
                self._record_closed_trade(pos, ticker, exit_fill, "STOP", source="SIMULATED")
                state.open_positions.discard(ticker)
                state.open_positions.discard(ticker.replace(".NS", ""))
            elif target_hit:
                exit_fill = max(open_p, target) if direction == "LONG" else min(open_p, target)
                log.info(
                    "Held position %s EXITED: Profit target triggered (Fill=%.2f High=%.2f Close=%.2f >= Target=%.2f)",
                    ticker, exit_fill, high_p, close_p, target,
                )
                if callable(deleter):
                    deleter(ticker)
                self._record_closed_trade(pos, ticker, exit_fill, "TARGET", source="SIMULATED")
                state.open_positions.discard(ticker)
                state.open_positions.discard(ticker.replace(".NS", ""))

    def _record_closed_trade(
        self,
        position: dict[str, Any],
        ticker: str,
        exit_price: float,
        exit_reason: str,
        source: str = "SIMULATED",
    ) -> None:
        entry = float(position.get("entry_price", position.get("entry", 0.0)))
        direction = str(position.get("direction", "LONG")).upper()
        if entry <= 0 or exit_price <= 0:
            return
        pnl = (exit_price - entry) / entry if direction == "LONG" else (entry - exit_price) / entry
        self._persistence.append_trade({
            **position, "ticker": ticker, "direction": direction, "entry": entry,
            "exit_price": exit_price, "exit_reason": exit_reason, "pnl": round(pnl, 6),
            "timestamp": datetime.now(IST).isoformat(),
            "source": source,
        })

    def _score_candidates_legacy_mock(
        self,
        *,
        prepared: PreparedScanData,
        regime: MarketRegime,
        config: SystemConfig,
        state: ScanState,
        session: str,
        debug: bool,
        no_intraday: bool = False,
        force_score: bool,
        current_nav: float,
    ) -> list[TickerResult]:
        mock_results: list[TickerResult] = []
        open_pos: set[str] = getattr(state, "open_positions", set())
        loader = getattr(self._persistence, "load_open_positions", None)
        pos_map: dict[str, dict[str, Any]] = loader() if callable(loader) else {}

        try:
            sig = inspect.signature(passes_static_filters)
            accepts_min_bars = "min_bars" in sig.parameters
        except Exception:
            accepts_min_bars = False

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures: dict[Any, str] = {}
            for ticker, df in prepared.processed.items():
                clean_ticker = ticker.replace(".NS", "")
                is_open = (ticker in open_pos) or (clean_ticker in open_pos)
                passes_static = (
                    passes_static_filters(df, config, min_bars=50)
                    if accepts_min_bars
                    else passes_static_filters(df, config)
                )

                if ticker == config.BENCHMARK or (not is_open and not passes_static):
                    continue

                pos_info = pos_map.get(ticker) or pos_map.get(clean_ticker) or {}
                held_dir = pos_info.get("direction")

                capital_fraction = confidence_position_scale(regime.confidence)
                if self._capital_scaler is not None:
                    try:
                        capital_fraction *= float(
                            self._capital_scaler.capital_fraction(current_nav, regime.regime)
                        )
                    except Exception:
                        log.debug("Injected capital scaler failed for %s.", ticker, exc_info=debug)

                call_kwargs: dict[str, Any] = {
                    "ticker": ticker,
                    "daily_df": df,
                    "bench": prepared.bench_series,
                    "sector_ranks": prepared.sector_ranks,
                    "sector_rs": prepared.sector_rs,
                    "session": session,
                    "regime": regime,
                    "config": config,
                    "factor_weights": state.factor_weights,
                    "allow_watchlist": getattr(config, "ENABLE_WATCHLIST", True),
                    "capital_fraction": capital_fraction,
                    "debug": debug,
                    "no_intraday": no_intraday,
                    "force_score": force_score,
                    "is_open_position": is_open,
                    "held_direction": held_dir,
                }

                future = executor.submit(score_ticker, **call_kwargs)
                futures[future] = ticker

            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        mock_results.append(res)
                except Exception as exc:
                    log.error("score_ticker mock error for %s: %s", ticker, exc, exc_info=True)
        return mock_results

    def _score_candidates(
        self,
        *,
        prepared: PreparedScanData,
        regime: MarketRegime,
        config: SystemConfig,
        state: ScanState,
        session: str,
        debug: bool,
        no_intraday: bool = False,
        force_score: bool,
    ) -> tuple[list[TickerResult], float]:
        started = time.monotonic()
        current_nav = self._resolve_current_nav()

        # Backward compatibility for tests that monkeypatch services.score_ticker
        if score_ticker is not _DEFAULT_SCORE_TICKER:
            mock_results = self._score_candidates_legacy_mock(
                prepared=prepared,
                regime=regime,
                config=config,
                state=state,
                session=session,
                debug=debug,
                no_intraday=no_intraday,
                force_score=force_score,
                current_nav=current_nav,
            )
            return mock_results, time.monotonic() - started

        all_results = score_universe(
            processed=prepared.processed,
            bench=prepared.bench_series,
            sector_rs=prepared.sector_rs,
            regime=regime,
            config=config,
            weights=state.factor_weights,
            open_positions=state.open_positions,
            open_pos_map=self._persistence.load_open_positions(),
            session=session,
            debug=debug,
            no_intraday=no_intraday,
            force_score=force_score,
            allow_watchlist=getattr(config, "ENABLE_WATCHLIST", True),
            capital_scaler=self._capital_scaler,
            current_nav=current_nav,
            sector_ranks=prepared.sector_ranks,
            min_bars=50,
        )
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
                return float(snapshot.current_nav)

        return self._default_current_nav

    def _maybe_recalibrate_ic_weights(
        self,
        *,
        all_results: list[TickerResult],
        prepared: PreparedScanData,
        config: SystemConfig,
        state: ScanState,
        regime_label: Optional[str] = None,
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
                regime_label=regime_label,
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


# ─────────────────────────────────────────────────────────────────────────────
# SERVICE BUNDLE  (moved here from screener_v14_modular.py)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ServiceBundle:
    """
    Immutable container that groups the three top-level service objects.

    Returned by ``configure_services`` and accepted by every public wrapper
    function in ``screener_v14_modular`` for dependency injection.
    Iterating a bundle yields ``(scan_service, alert_service)`` so legacy
    tuple-unpacking still works.
    """
    persistence:   PersistenceService
    scan_service:  ScanService
    alert_service: AlertService

    def __iter__(self) -> Iterator[Any]:
        yield self.scan_service
        yield self.alert_service


def configure_services(
    *,
    data_service:          Optional[Any] = None,
    persistence:           Optional[PersistenceService] = None,
    probability_gate:      Optional[Any] = None,
    capital_scaler:        Optional[Any] = None,
    factor_calibrator:     Optional[Any] = None,
    alerter:               Optional[Any] = None,
    current_nav_provider:  Optional[Any] = None,
    alert_service:         Optional[AlertService] = None,
    version:               str = "14.6-Modular",
) -> "ServiceBundle":
    """
    Assemble and return a fully wired :class:`ServiceBundle`.

    All parameters are optional — omit any collaborator to use the
    corresponding default (no-op gate, no alerter, etc.).  The returned
    bundle is frozen and thread-safe to share across scan loops.

    Parameters
    ----------
    data_service:
        Optional custom :class:`MarketDataService`.
    persistence:
        Persistence layer; a new :class:`PersistenceService` is created if
        not supplied.
    probability_gate:
        Object implementing :class:`ProbabilityGate` (regime-aware threshold).
    capital_scaler:
        Object implementing :class:`CapitalFractionScaler`.
    factor_calibrator:
        Object implementing :class:`FactorWeightProvider`.
    alerter:
        Object implementing :class:`SummaryAlerter`.
    current_nav_provider:
        Zero-arg callable returning the live portfolio NAV as ``float | None``.
    alert_service:
        Pre-built :class:`AlertService`; constructed via
        ``ScanService.create_alert_service()`` if not supplied.
    version:
        Version string embedded in outbound alerts.
    """
    resolved_persistence = persistence or PersistenceService()
    scan_svc = ScanService(
        version=version,
        data_service=data_service,
        persistence=resolved_persistence,
        probability_gate=probability_gate,
        capital_scaler=capital_scaler,
        factor_calibrator=factor_calibrator,
        alerter=alerter,
        current_nav_provider=current_nav_provider,
    )
    resolved_alert = alert_service or scan_svc.create_alert_service()
    return ServiceBundle(
        persistence=resolved_persistence,
        scan_service=scan_svc,
        alert_service=resolved_alert,
    )
