"""
core/services.py
================
Service boundaries for the Sovereign Engine runtime.

This module turns the main runtime concerns into explicit collaborators:

- ``PersistenceService`` for stateful files
- ``MarketDataService`` for fetch + indicator preparation
- ``PositionMonitorService`` for open-position stop/target lifecycle
- ``RegimeService`` for breadth and regime classification
- ``ScoringPipelineService`` for multi-factor candidate scoring and calibration
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
import html
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from .backtest import TransactionCostModel

import pandas as pd

from .backtest import OverallStats, WalkForwardResult, walk_forward
from .cache import ScanCache
from .calibration import CalibrationService
from .config import CONFIG, IST, MarketDataSettings, MarketRegimeType, SystemConfig, get_secret_value
from .database import SqliteDatabase
from .data_provider import fetch_daily_batch
from .factors import DEFAULT_WEIGHTS, calibrate_ic_weights
from .indicators import add_indicators
from .portfolio import optimize_portfolio
from .swing import completed_daily_bars
from .regime import (
    MarketRegime,
    RegimeTracker,
    classify_regime,
    confidence_position_scale,
    compute_breadth,
    compute_sector_rs,
)
from .runtime_paths import RUNTIME_PATHS, RuntimePaths, ensure_parent, ensure_runtime_dirs, resolve_artifact_path
from .snapshots import write_json_atomic
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
CancellationProbe = Callable[[], bool]


class ScanCancelled(RuntimeError):
    """Raised when an in-flight scan receives a cooperative stop request."""


def _check_cancelled(cancel_requested: Optional[CancellationProbe]) -> None:
    if cancel_requested is not None and cancel_requested():
        raise ScanCancelled("scan cancelled by emergency killswitch")


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
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
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
        write_json_atomic(target, payload)
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
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
                log.error("Could not load trade log %s: %s", candidate, exc)
                return []
        return []

    def save_trade_log(self, trades: list[dict[str, Any]]) -> None:
        target = ensure_parent(self.paths.trade_log_file)
        write_json_atomic(target, trades)
        self.db.insert_trades(trades)

    def append_trade(self, trade: dict[str, Any]) -> None:
        self.db.insert_trade(trade)
        existing = self.load_trade_log()
        target = ensure_parent(self.paths.trade_log_file)
        write_json_atomic(target, existing[-2000:])

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
            except (OSError, json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
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
        write_json_atomic(target, payload)
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
        fetcher: Callable[[list[str], MarketDataSettings | SystemConfig], dict[str, pd.DataFrame]] = fetch_daily_batch,
    ) -> None:
        self._fetcher = fetcher

    def fetch_universe(
        self,
        tickers: list[str],
        config: SystemConfig,
        metrics: Optional[ScanMetrics] = None,
    ) -> dict[str, pd.DataFrame]:
        started = time.monotonic()
        provider_config: MarketDataSettings | SystemConfig = (
            config.as_market_data() if self._fetcher is fetch_daily_batch else config
        )
        raw_data = self._fetcher(tickers, provider_config)
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
                if config.SWING_SETUP_ENABLED:
                    df = completed_daily_bars(df, config)
                    if df.empty:
                        ind_fail += 1
                        continue
                processed[ticker] = add_indicators(df, config)
            except (KeyError, ValueError, TypeError, IndexError):
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
        if config.SWING_SETUP_ENABLED:
            processed = {t: frame for t, frame in processed.items()
                         if frame.index[-1] == bench_series.index[-1]}
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

        top = top[: min(config.TELEGRAM_ALERT_TOP_N, 5) if self._alerter is not None
                  else config.TELEGRAM_ALERT_TOP_N]
        if not top:
            return
        if self._alerter is not None:
            sent = self._alerter.send_daily_summary(
                regime.label,
                [result.__dict__ for result in top],
                current_nav=self._resolve_current_nav(),
            )
            if sent is True:
                for result in top:
                    self._last_alerted[f"{result.ticker}:{getattr(result, 'trade_horizon', 'SWING')}"] = now
            return

        version_esc = html.escape(str(self._version))
        regime_esc = html.escape(str(regime.label))
        lines = [
            f"<b>Sovereign v{version_esc}</b> | {regime_esc} | {datetime.now(IST).strftime('%H:%M IST')}"
        ]
        for result in top[: config.TELEGRAM_ALERT_TOP_N]:
            ticker_esc = html.escape(str(result.ticker))
            direction_esc = html.escape(str(result.direction))
            lines.append(
                f"<b>{ticker_esc}</b> {direction_esc} | P={result.prob_win:.0%} | "
                f"E(R)={result.expectancy_r:.2f} | Entry {result.entry} | SL {result.stop} | "
                f"T1 {result.t1} | {result.shares} shares"
            )
        sent = self._messenger(
            "\n".join(lines),
            get_secret_value(config.TELEGRAM_BOT_TOKEN),
            config.TELEGRAM_CHAT_ID,
        )
        if sent:
            for result in top:
                self._last_alerted[f"{result.ticker}:{getattr(result, 'trade_horizon', 'SWING')}"] = now

    def _resolve_current_nav(self) -> Optional[float]:
        if self._current_nav_provider is None:
            return None
        try:
            current_nav = self._current_nav_provider()
            return float(current_nav) if current_nav is not None and current_nav > 0 else None
        except (TypeError, ValueError, RuntimeError, AttributeError):
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
    cancel_requested: Optional[CancellationProbe] = None,
    now: Optional[datetime] = None,
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
        except (TypeError, ValueError, AttributeError, RuntimeError):
            log.debug("Injected capital scaler failed.", exc_info=debug)

    pass1_candidates: list[CandidateContext] = []

    candidate_items: list[tuple[str, pd.DataFrame]] = []
    for ticker, df in processed.items():
        _check_cancelled(cancel_requested)
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
            _check_cancelled(cancel_requested)
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
                now=now,
            )
            pass1_futures[fut] = ticker

        for fut in as_completed(pass1_futures):
            if cancel_requested is not None and cancel_requested():
                for pending in pass1_futures:
                    pending.cancel()
                raise ScanCancelled("scan cancelled during candidate scoring")
            ticker = pass1_futures[fut]
            try:
                cand = fut.result()
                if cand is not None:
                    if direction == "BOTH" or cand.direction == direction:
                        pass1_candidates.append(cand)
            except Exception as exc:
                log.error("score_candidate_pass1 error for %s: %s", ticker, exc, exc_info=True)

    pass1_candidates.sort(key=lambda c: c.ticker)

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
        _check_cancelled(cancel_requested)
        clean_ticker = cand.ticker.replace(".NS", "")
        is_open = (cand.ticker in open_pos) or (clean_ticker in open_pos)
        try:
            res = score_candidate_pass2(
                candidate=cand,
                config=config,
                is_open_position=is_open,
                allow_watchlist=allow_watchlist,
                debug=debug,
                now=now,
            )
            if res is not None:
                if is_open:
                    held = pos_map.get(cand.ticker) or pos_map.get(clean_ticker) or {}
                    # Rescoring updates signal metadata, never the original fill.
                    res.entry = float(held.get("entry_price", held.get("entry", res.entry)))
                    res.shares = int(held.get("shares", res.shares))
                    res.stop = float(held.get("stop_loss", held.get("stop", res.stop)))
                    res.t1 = float(held.get("target", held.get("t1", res.t1)))
                    res.trade_horizon = held.get("trade_horizon", res.trade_horizon)
                    res.risk_inr = round(res.shares * abs(res.entry - res.stop), 2)
                if direction == "BOTH" or res.direction == direction:
                    all_results.append(res)
        except Exception as exc:
            log.error("score_candidate_pass2 error for %s: %s", cand.ticker, exc, exc_info=True)

    return all_results


class PositionMonitorService:
    """
    Deterministic price-vs-level monitoring and trade closures for open positions.
    """

    def __init__(self, persistence: Optional[PersistenceService] = None) -> None:
        self._persistence = persistence or PersistenceService()

    def monitor_open_position_stops(
        self,
        processed: dict[str, pd.DataFrame],
        state: ScanState,
        record_callback: Optional[Callable[..., None]] = None,
    ) -> None:
        loader = getattr(self._persistence, "load_open_positions", None)
        if not callable(loader):
            return
        try:
            open_pos_map = loader()
        except (OSError, RuntimeError, ValueError, TypeError) as exc:
            log.warning("Could not load open positions for stop monitor: %s", exc)
            return
        if not open_pos_map:
            return

        db = getattr(self._persistence, "db", None)
        deleter = getattr(db, "delete_open_position", None)
        recorder = record_callback or self.record_closed_trade

        for ticker, pos in open_pos_map.items():
            # Confirmed broker positions require an explicit exit fill event.
            if pos.get("source") == "EXECUTED":
                continue
            df = processed.get(ticker)
            if df is None:
                df = processed.get(f"{ticker}.NS")
            if df is None or df.empty:
                continue

            if not isinstance(df.index, pd.DatetimeIndex):
                log.warning("Skipping stop monitor for %s: missing candle timestamps.", ticker)
                continue
            try:
                bar_start = pd.Timestamp(df.index[-1])
                opened_at = pd.Timestamp(pos.get("opened_at"))
                if pd.isna(bar_start) or pd.isna(opened_at):
                    continue
                if bar_start.tzinfo is None:
                    bar_start = bar_start.tz_localize(IST)
                if opened_at.tzinfo is None:
                    opened_at = opened_at.tz_localize(IST)
                # A candle containing the entry has unknowable pre-fill extrema.
                if bar_start <= opened_at:
                    continue
            except (TypeError, ValueError):
                log.warning("Skipping stop monitor for %s: invalid entry/candle time.", ticker)
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
                recorder(pos, ticker, exit_fill, "STOP", source="SIMULATED")
                if callable(deleter) and not pos.get("trade_id"):
                    deleter(ticker)
                state.open_positions.discard(ticker)
                state.open_positions.discard(ticker.replace(".NS", ""))
            elif target_hit:
                exit_fill = max(open_p, target) if direction == "LONG" else min(open_p, target)
                log.info(
                    "Held position %s EXITED: Profit target triggered (Fill=%.2f High=%.2f Close=%.2f >= Target=%.2f)",
                    ticker, exit_fill, high_p, close_p, target,
                )
                recorder(pos, ticker, exit_fill, "TARGET", source="SIMULATED")
                if callable(deleter) and not pos.get("trade_id"):
                    deleter(ticker)
                state.open_positions.discard(ticker)
                state.open_positions.discard(ticker.replace(".NS", ""))

    def record_closed_trade(
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
        if position.get("trade_id"):
            from .backtest import DEFAULT_COST_MODEL, SWING_COST_MODEL
            from .trades import TradeExit, TradeLifecycle
            model = SWING_COST_MODEL if position.get("trade_horizon") == "SWING" else DEFAULT_COST_MODEL
            stop = float(position.get("stop_loss", position.get("stop", entry)))
            shares = int(position.get("shares", 0))
            costs = model.friction_r(entry, abs(entry - stop), shares) * abs(entry - stop) * shares
            TradeLifecycle(self._persistence.db).close(str(position["trade_id"]), TradeExit(
                exit_price=exit_price, exit_ts=datetime.now(IST), costs=costs, exit_reason=exit_reason, cost_basis="MODELED",
            ))
            return
        pnl = (exit_price - entry) / entry if direction == "LONG" else (entry - exit_price) / entry
        self._persistence.append_trade({
            **position, "ticker": ticker, "direction": direction, "entry": entry,
            "exit_price": exit_price, "exit_reason": exit_reason, "pnl": round(pnl, 6),
            "timestamp": datetime.now(IST).isoformat(),
            "source": source,
        })


class RegimeService:
    """
    Evaluates market breadth, regime classification, and override enforcement.
    """

    @staticmethod
    def apply_regime_override(
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

    @staticmethod
    def log_regime(regime: MarketRegime, config: SystemConfig) -> None:
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

    def determine_regime(
        self,
        prepared: PreparedScanData,
        config: SystemConfig,
        state: ScanState,
        regime_tracker: Optional[RegimeTracker] = None,
        regime_override: Optional[str] = None,
    ) -> MarketRegime:
        breadth = compute_breadth(prepared.processed, config)
        state.regime_locked = False if config.SWING_SETUP_ENABLED else config.is_regime_locked()
        if state.regime_locked:
            log.info(
                "Regime lock active during opening noise window (%d min).",
                config.REGIME_LOCK_MINUTES,
            )

        tracker = regime_tracker if regime_tracker is not None else RegimeTracker()
        if config.SWING_SETUP_ENABLED:
            # Confirmation counts distinct completed dates, never scan invocations.
            tracker = RegimeTracker()
            for prior_date in prepared.bench_series.index[:-1][-max(2, config.REGIME_CONFIRM_BARS):]:
                frames = {ticker: frame.loc[:prior_date] for ticker, frame in prepared.processed.items()
                          if prior_date in frame.index}
                prior_sectors = compute_sector_rs(frames, frames[config.BENCHMARK].Close, config)
                classify_regime(frames, compute_breadth(frames, config), tracker, config, sector_rs=prior_sectors)
        regime = classify_regime(
            prepared.processed,
            breadth,
            tracker,
            config,
            locked=state.regime_locked,
            sector_rs=prepared.sector_rs,
        )
        return self.apply_regime_override(regime, regime_override)


class ScoringPipelineService:
    """
    Candidate scoring, legacy test mock scoring, dynamic NAV resolution, and IC recalibration.
    """

    def __init__(
        self,
        persistence: Optional[PersistenceService] = None,
        probability_gate: Optional[ProbabilityGate] = None,
        capital_scaler: Optional[CapitalFractionScaler] = None,
        factor_calibrator: Optional[FactorWeightProvider] = None,
        current_nav_provider: Optional[CurrentNavProvider] = None,
        default_current_nav: float = 1_000_000.0,
    ) -> None:
        self._persistence = persistence or PersistenceService()
        self._probability_gate = probability_gate
        self._capital_scaler = capital_scaler
        self._factor_calibrator = factor_calibrator
        self._current_nav_provider = current_nav_provider
        self._default_current_nav = default_current_nav

    @staticmethod
    def validated_factor_weights(weights: dict[str, float]) -> dict[str, float]:
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

    def resolve_current_nav(self) -> float:
        if self._current_nav_provider is not None:
            try:
                current_nav = self._current_nav_provider()
                if current_nav is not None and current_nav > 0:
                    return float(current_nav)
            except (TypeError, ValueError, AttributeError, RuntimeError):
                log.warning("Current NAV provider failed; using default NAV %.0f.", self._default_current_nav)

        loader = getattr(self._persistence, "load_portfolio_state", None)
        if callable(loader):
            snapshot = loader()
            if snapshot is not None:
                return float(snapshot.current_nav)

        return self._default_current_nav

    def apply_regime_probability_gate(
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
        except (TypeError, ValueError, AttributeError, RuntimeError):
            log.warning("Regime probability gate failed; using static threshold.", exc_info=debug)
            return config

    def maybe_recalibrate_ic_weights(
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
            state.factor_weights = self.validated_factor_weights(new_weights)
            state.weights_calibrated = True
            log.info("IC weights optimized: %s", {key: round(value, 4) for key, value in new_weights.items()})
        except (ValueError, TypeError, KeyError, ZeroDivisionError, RuntimeError) as exc:
            log.warning("IC calibration failed: %s", exc)

    def score_candidates_legacy_mock(
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
        cancel_requested: Optional[CancellationProbe] = None,
    ) -> list[TickerResult]:
        mock_results: list[TickerResult] = []
        open_pos: set[str] = getattr(state, "open_positions", set())
        loader = getattr(self._persistence, "load_open_positions", None)
        pos_map: dict[str, dict[str, Any]] = loader() if callable(loader) else {}

        try:
            sig = inspect.signature(passes_static_filters)
            accepts_min_bars = "min_bars" in sig.parameters
        except (TypeError, ValueError):
            accepts_min_bars = False

        with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            futures: dict[Any, str] = {}
            for ticker, df in prepared.processed.items():
                _check_cancelled(cancel_requested)
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
                    except (TypeError, ValueError, AttributeError, RuntimeError):
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
                _check_cancelled(cancel_requested)
                ticker = futures[future]
                try:
                    res = future.result()
                    if res is not None:
                        mock_results.append(res)
                except Exception as exc:
                    log.error("score_ticker mock error for %s: %s", ticker, exc, exc_info=True)
        return mock_results

    def score_candidates(
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
        cancel_requested: Optional[CancellationProbe] = None,
    ) -> tuple[list[TickerResult], float]:
        started = time.monotonic()
        current_nav = self.resolve_current_nav()

        # Backward compatibility for tests that monkeypatch services.score_ticker
        if score_ticker is not _DEFAULT_SCORE_TICKER:
            mock_results = self.score_candidates_legacy_mock(
                prepared=prepared,
                regime=regime,
                config=config,
                state=state,
                session=session,
                debug=debug,
                no_intraday=no_intraday,
                force_score=force_score,
                current_nav=current_nav,
                cancel_requested=cancel_requested,
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
            cancel_requested=cancel_requested,
        )
        elapsed = time.monotonic() - started
        log.debug("Scoring completed in %.3fs.", elapsed)
        return all_results, elapsed


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
        position_monitor: Optional[PositionMonitorService] = None,
        regime_service: Optional[RegimeService] = None,
        scoring_pipeline: Optional[ScoringPipelineService] = None,
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

        self._position_monitor = position_monitor or PositionMonitorService(self._persistence)
        self._regime_service = regime_service or RegimeService()
        self._scoring_pipeline = scoring_pipeline or ScoringPipelineService(
            persistence=self._persistence,
            probability_gate=self._probability_gate,
            capital_scaler=self._capital_scaler,
            factor_calibrator=self._factor_calibrator,
            current_nav_provider=self._current_nav_provider,
            default_current_nav=self._default_current_nav,
        )

        self.last_sector_rs: dict[str, float] = {}
        self.last_regime_info: Optional[MarketRegime] = None
        self._recent_alerts: list[dict[str, float]] = []  # [{ticker: composite}, ...]

    @staticmethod
    def _validated_factor_weights(weights: dict[str, float]) -> dict[str, float]:
        return ScoringPipelineService.validated_factor_weights(weights)

    def _sync_subservices(self) -> None:
        self._position_monitor._persistence = self._persistence
        self._scoring_pipeline._persistence = self._persistence
        self._scoring_pipeline._probability_gate = self._probability_gate
        self._scoring_pipeline._capital_scaler = self._capital_scaler
        self._scoring_pipeline._factor_calibrator = self._factor_calibrator
        self._scoring_pipeline._current_nav_provider = self._current_nav_provider
        self._scoring_pipeline._default_current_nav = self._default_current_nav

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
        cancel_requested: Optional[CancellationProbe] = None,
    ) -> tuple[list[TickerResult], list[TickerResult], Optional[MarketRegime]]:
        _check_cancelled(cancel_requested)
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

        _check_cancelled(cancel_requested)
        self.last_sector_rs = dict(prepared.sector_rs)
        self._monitor_open_position_stops(prepared.processed, state)

        regime = self._determine_regime(
            prepared=prepared,
            config=config,
            state=state,
            regime_tracker=regime_tracker,
            regime_override=regime_override,
        )
        self.last_regime_info = regime
        _check_cancelled(cancel_requested)

        session = config.session_from_time()
        scoring_config = config if config.SWING_SETUP_ENABLED else self._apply_regime_probability_gate(config, regime, debug=debug)

        metrics.record_regime(regime)
        self._log_regime(regime, config)

        if not regime.is_tradeable():
            log.warning("PANIC regime - no new positions.")
            if config.SWING_SETUP_ENABLED:
                from .paper_ledger import PaperLedger
                PaperLedger(self._persistence.db).record_scan(str(prepared.bench_series.index[-1]), regime.label, 0, 0)
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
            cancel_requested=cancel_requested,
        )

        _check_cancelled(cancel_requested)

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

        _check_cancelled(cancel_requested)
        corr_matrix = state.cache.corr_matrix(prepared.processed, config)
        portfolio_candidates = [r for r in all_results if not getattr(r, "is_watchlist", False)]
        portfolio = optimize_portfolio(portfolio_candidates, config, corr_matrix)

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

        if config.SWING_SETUP_ENABLED:
            from .paper_ledger import PaperLedger
            from .swing_research import leadership_snapshot
            observations = leadership_snapshot(prepared.processed, config, prepared.bench_series.index[-1])
            for result in all_results:
                result.research_context = observations.get(f"{result.ticker}.NS", {})
            ledger = PaperLedger(self._persistence.db)
            ledger.record_signals(all_results, portfolio, config)
            ledger.record_scan(str(prepared.bench_series.index[-1]), regime.label, len(all_results), len(portfolio))

        self._record_leader_pullback(prepared, config)
        return ScanOutput(all_results, portfolio, regime, sector_rs=self.last_sector_rs)

    def _record_leader_pullback(self, prepared: Any, config: SystemConfig) -> None:
        """Paper-only research setup; never affects the live portfolio and never fails the scan."""
        if not config.LEADER_PULLBACK_ENABLED:
            return
        try:
            from .leader_pullback import scan_leader_pullback
            from .paper_ledger import PaperLedger
            from .swing import completed_daily_bars
            bench = completed_daily_bars(prepared.processed[config.BENCHMARK], config)
            frames = {t: completed_daily_bars(df, config) for t, df in prepared.processed.items() if t != config.BENCHMARK}
            signals = scan_leader_pullback(frames, bench["Close"], bench.index[-1], capital=config.CAPITAL_INR)
            self.last_leader_pullback = signals
            PaperLedger(self._persistence.db).record_pullback_signals(signals)
            log.info("LEADER_PULLBACK_V1 (paper only): %d signal(s)", len(signals))
            for s in signals:
                log.info("  %s | close %.2f | stop -%.2f | %d sh | RS %.0f%% | RSI2 %.1f | %s",
                         s.ticker, s.close, s.stop_offset, s.shares, s.rs_percentile, s.rsi2, s.exit_rule)
        except Exception:
            log.warning("LEADER_PULLBACK_V1 scan failed", exc_info=True)

    def run_calibration(self, calib_offset: int = 60) -> None:
        CalibrationService(self._persistence, calibrate_platt).run_platt(calib_offset)

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
        self._sync_subservices()
        self._position_monitor.monitor_open_position_stops(
            processed,
            state,
            record_callback=self._record_closed_trade,
        )

    def _record_closed_trade(
        self,
        position: dict[str, Any],
        ticker: str,
        exit_price: float,
        exit_reason: str,
        source: str = "SIMULATED",
    ) -> None:
        self._sync_subservices()
        self._position_monitor.record_closed_trade(
            position,
            ticker,
            exit_price,
            exit_reason,
            source=source,
        )

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
        cancel_requested: Optional[CancellationProbe] = None,
    ) -> list[TickerResult]:
        self._sync_subservices()
        return self._scoring_pipeline.score_candidates_legacy_mock(
            prepared=prepared,
            regime=regime,
            config=config,
            state=state,
            session=session,
            debug=debug,
            no_intraday=no_intraday,
            force_score=force_score,
            current_nav=current_nav,
            cancel_requested=cancel_requested,
        )

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
        cancel_requested: Optional[CancellationProbe] = None,
    ) -> tuple[list[TickerResult], float]:
        self._sync_subservices()
        return self._scoring_pipeline.score_candidates(
            prepared=prepared,
            regime=regime,
            config=config,
            state=state,
            session=session,
            debug=debug,
            no_intraday=no_intraday,
            force_score=force_score,
            cancel_requested=cancel_requested,
        )

    def _resolve_current_nav(self) -> float:
        self._sync_subservices()
        return self._scoring_pipeline.resolve_current_nav()

    def _maybe_recalibrate_ic_weights(
        self,
        *,
        all_results: list[TickerResult],
        prepared: PreparedScanData,
        config: SystemConfig,
        state: ScanState,
        regime_label: Optional[str] = None,
    ) -> None:
        self._sync_subservices()
        self._scoring_pipeline.maybe_recalibrate_ic_weights(
            all_results=all_results,
            prepared=prepared,
            config=config,
            state=state,
            regime_label=regime_label,
        )

    def _apply_regime_probability_gate(
        self,
        config: SystemConfig,
        regime: MarketRegime,
        *,
        debug: bool,
    ) -> SystemConfig:
        self._sync_subservices()
        return self._scoring_pipeline.apply_regime_probability_gate(
            config,
            regime,
            debug=debug,
        )

    def _apply_regime_override(
        self,
        regime: MarketRegime,
        regime_override: Optional[str],
    ) -> MarketRegime:
        return self._regime_service.apply_regime_override(
            regime,
            regime_override,
        )

    def _log_regime(self, regime: MarketRegime, config: SystemConfig) -> None:
        self._regime_service.log_regime(regime, config)

    def _determine_regime(
        self,
        prepared: PreparedScanData,
        config: SystemConfig,
        state: ScanState,
        regime_tracker: Optional[RegimeTracker] = None,
        regime_override: Optional[str] = None,
    ) -> MarketRegime:
        return self._regime_service.determine_regime(
            prepared=prepared,
            config=config,
            state=state,
            regime_tracker=regime_tracker,
            regime_override=regime_override,
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
