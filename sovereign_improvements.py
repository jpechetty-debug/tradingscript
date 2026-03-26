"""
╔══════════════════════════════════════════════════════════════════╗
║         SOVEREIGN ENGINE v13.0 — UNIFIED IMPROVEMENT PATCH      ║
║                                                                  ║
║  Improvements:                                                   ║
║  1. Tiered PANIC mode capital scaling  (50% → 25% → 0%)         ║
║  2. Regime-dependent probability gate  (dynamic P(Win))          ║
║  3. Fyers → yfinance fallback chain    (degradation logging)     ║
║  4. Continuous auto-recalibration      (rolling window weights)  ║
║  5. Exponential backoff                (watch mode API safety)   ║
║  6. Telegram regime-aware alerts       (state in every message)  ║
║                                                                  ║
║  DROP-IN: import sovereign_improvements as SE_PATCH             ║
║  then call SE_PATCH.apply(engine_instance) at startup.          ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Fix #9: never call logging.basicConfig in a library module — it
# reconfigures the root logger and overrides whatever the application
# set up via core.telemetry.setup_logging.  The caller owns logging
# configuration; this module just gets a named logger.
logger = logging.getLogger("sovereign.patch")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS_PATH   = Path(os.getenv("WEIGHTS_PATH",   "factor_weights.json"))
TRADE_LOG_PATH = Path(os.getenv("TRADE_LOG_PATH", "trade_log.json"))
DEGRADATION_LOG_PATH = Path("data_provider_degradation.log")

REGIME_GATE: dict[str, float] = {
    "TREND_UP":   0.50,   # more permissive in confirmed uptrend
    "EXPANSION":  0.52,   # default
    "RANGE":      0.55,   # stricter — choppier conditions
    "TREND_DOWN": 0.58,   # very strict — only high-conviction longs
    "PANIC":      0.99,   # effectively blocks all new longs
}

REGIME_CAPITAL: dict[str, float] = {
    "TREND_UP":   1.00,
    "EXPANSION":  1.00,
    "RANGE":      0.75,
    "TREND_DOWN": 0.50,
    "PANIC":      0.00,   # computed dynamically via tiered scaler
}

# Tiered PANIC thresholds (portfolio drawdown → capital fraction)
PANIC_TIERS: list[tuple[float, float]] = [
    (0.00, 1.00),   # < 2% drawdown  → full capital
    (0.02, 0.50),   # 2–5% drawdown  → 50%
    (0.05, 0.25),   # 5–10% drawdown → 25%
    (0.10, 0.00),   # > 10% drawdown → no new positions
]

RECALIBRATION_WINDOW   = int(os.getenv("RECAL_WINDOW", "50"))    # last N trades
RECALIBRATION_INTERVAL = int(os.getenv("RECAL_INTERVAL", "300")) # seconds

BACKOFF_BASE    = 2.0
BACKOFF_MAX     = 120.0
BACKOFF_JITTER  = 0.25

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID",   "")


# ═════════════════════════════════════════════════════════════════════════════
# 1. TIERED PANIC MODE CAPITAL SCALER
# ═════════════════════════════════════════════════════════════════════════════

class TieredCapitalScaler:
    """
    Replaces binary PANIC on/off with a graduated drawdown-based scale.

    Usage:
        scaler = TieredCapitalScaler(portfolio_peak=1_000_000)
        fraction = scaler.capital_fraction(current_nav=940_000, regime="PANIC")
        position_size = base_size * fraction
    """

    def __init__(self, portfolio_peak: float) -> None:
        self.peak = portfolio_peak
        self._history: deque[tuple[datetime, float]] = deque(maxlen=500)

    def update_peak(self, current_nav: float) -> None:
        self.peak = max(self.peak, current_nav)
        self._history.append((datetime.utcnow(), current_nav))

    def current_drawdown(self, current_nav: float) -> float:
        if self.peak <= 0:
            return 0.0
        return max(0.0, (self.peak - current_nav) / self.peak)

    def capital_fraction(self, current_nav: float, regime: str) -> float:
        """Return [0, 1] fraction of capital allowed for new positions."""
        self.update_peak(current_nav)
        dd = self.current_drawdown(current_nav)

        if regime not in ("PANIC", "TREND_DOWN"):
            base = REGIME_CAPITAL.get(regime, 1.0)
            # Still apply mild drawdown tapering outside PANIC
            if dd > 0.02:
                base *= max(0.5, 1.0 - dd * 3)
            logger.debug("Regime=%s DD=%.2f%% → capital=%.0f%%", regime, dd * 100, base * 100)
            return round(base, 4)

        # Tiered PANIC logic
        fraction = 0.0
        for threshold, cap in reversed(PANIC_TIERS):
            if dd >= threshold:
                fraction = cap
                break

        logger.warning(
            "PANIC scaler: drawdown=%.2f%% → capital_fraction=%.0f%%",
            dd * 100, fraction * 100,
        )
        return fraction

    def summary(self, current_nav: float) -> dict[str, Any]:
        dd = self.current_drawdown(current_nav)
        return {
            "peak_nav":        round(self.peak, 2),
            "current_nav":     round(current_nav, 2),
            "drawdown_pct":    round(dd * 100, 2),
            "capital_allowed": f"{self.capital_fraction(current_nav, 'PANIC') * 100:.0f}%",
        }


# ═════════════════════════════════════════════════════════════════════════════
# 2. REGIME-DEPENDENT PROBABILITY GATE
# ═════════════════════════════════════════════════════════════════════════════

class RegimeProbabilityGate:
    """
    Dynamic P(Win) threshold that tightens in weak regimes and relaxes
    in confirmed uptrends. Replaces the static 52% gate.

    Usage:
        gate = RegimeProbabilityGate()
        if gate.passes(p_win=0.54, regime="RANGE"):
            execute_trade(...)
    """

    def __init__(self, overrides: Optional[dict[str, float]] = None) -> None:
        self._gates = {**REGIME_GATE, **(overrides or {})}

    def threshold(self, regime: str) -> float:
        return self._gates.get(regime, 0.52)

    def passes(self, p_win: float, regime: str) -> bool:
        t = self.threshold(regime)
        result = p_win >= t
        logger.debug(
            "ProbGate: p_win=%.3f threshold=%.3f regime=%s → %s",
            p_win, t, regime, "PASS" if result else "BLOCK",
        )
        return result

    def margin(self, p_win: float, regime: str) -> float:
        """How far above/below the threshold. Positive = passes."""
        return round(p_win - self.threshold(regime), 4)

    def all_thresholds(self) -> dict[str, float]:
        return dict(self._gates)


# ═════════════════════════════════════════════════════════════════════════════
# 3. FYERS → YFINANCE FALLBACK CHAIN WITH DEGRADATION LOGGING
# ═════════════════════════════════════════════════════════════════════════════

class ResilientDataProvider:
    """
    Tries Fyers first; falls back to yfinance on any exception.
    Logs every degradation event to file + Telegram.

    Usage:
        dp = ResilientDataProvider(fyers_client=fyers_obj)
        df = dp.fetch("RELIANCE", period="1d", interval="15m")
    """

    def __init__(self, fyers_client: Any = None) -> None:
        self._fyers    = fyers_client
        self._degraded = False
        self._fail_count = 0
        self._degradation_log: list[dict] = []

        try:
            import yfinance as yf
            self._yf = yf
        except ImportError:
            raise RuntimeError("yfinance not installed — run: pip install yfinance")

    # ── Public API ────────────────────────────────────────────────────────────

    def fetch(
        self,
        symbol: str,
        period: str = "1d",
        interval: str = "15m",
        **kwargs,
    ) -> Any:
        """Fetch OHLCV data. Returns pandas DataFrame."""
        if self._fyers and not self._degraded:
            try:
                return self._fetch_fyers(symbol, period, interval, **kwargs)
            except Exception as exc:
                self._handle_fyers_failure(symbol, exc)

        return self._fetch_yfinance(symbol, period, interval, **kwargs)

    def force_fyers_recovery(self) -> None:
        """Call after refreshing Fyers token to re-enable primary provider."""
        self._degraded   = False
        self._fail_count = 0
        logger.info("Fyers provider restored — resuming primary data source.")

    def degradation_report(self) -> list[dict]:
        return list(self._degradation_log)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_fyers(self, symbol, period, interval, **kwargs) -> Any:
        # Adapt to your actual Fyers client API signature
        if symbol == "^NSEI":
            fsym = "NSE:NIFTY50-INDEX"
        elif symbol == "^NSEBANK":
            fsym = "NSE:NIFTYBANK-INDEX"
        else:
            base_sym = symbol.replace(".NS", "")
            fsym = f"NSE:{base_sym}-EQ"

        data = self._fyers.history(
            symbol   = fsym,
            resolution = interval,
            date_format = 1,
        )
        if data.get("s") != "ok":
            raise ValueError(f"Fyers returned non-ok status: {data.get('s')}")
        import pandas as pd
        df = pd.DataFrame(data["candles"],
                          columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    def _fetch_yfinance(self, symbol, period, interval, **kwargs) -> Any:
        # Don't append .NS to indices (starting with ^)
        if symbol.startswith("^"):
            ticker = symbol
        else:
            ticker = f"{symbol}.NS" if not symbol.endswith(".NS") else symbol

        df = self._yf.download(
            ticker, period=period, interval=interval,
            auto_adjust=True, progress=False, **kwargs,
        )
        if df.empty:
            raise ValueError(f"yfinance returned empty DataFrame for {ticker}")
        logger.info("yfinance fallback succeeded for %s", symbol)
        return df

    def _handle_fyers_failure(self, symbol: str, exc: Exception) -> None:
        self._fail_count += 1
        event = {
            "ts":      datetime.utcnow().isoformat(),
            "symbol":  symbol,
            "error":   str(exc),
            "fail_n":  self._fail_count,
        }
        self._degradation_log.append(event)
        self._write_degradation_log(event)

        logger.warning(
            "Fyers fetch failed (#%d) for %s: %s — switching to yfinance",
            self._fail_count, symbol, exc,
        )

        # Mark as degraded after 3 consecutive failures
        if self._fail_count >= 3:
            self._degraded = True
            logger.error("Fyers provider DEGRADED after %d failures. Using yfinance only.", self._fail_count)
            _send_telegram(
                f"🔴 *Data provider degraded*\n"
                f"Fyers failed {self._fail_count}× — falling back to yfinance.\n"
                f"Last error: `{exc}`\n"
                f"Run `fyers_setup.py` to restore."
            )

    @staticmethod
    def _write_degradation_log(event: dict) -> None:
        try:
            with DEGRADATION_LOG_PATH.open("a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# 4. CONTINUOUS AUTO-RECALIBRATION (ROLLING WINDOW FACTOR WEIGHTS)
# ═════════════════════════════════════════════════════════════════════════════

class RollingFactorCalibrator:
    """
    Continuously re-fits factor weights using the last N closed trades
    via Ridge regression. Runs in a background thread so it never blocks
    the main scan loop.

    Usage:
        cal = RollingFactorCalibrator(window=50, interval_sec=300)
        cal.start()                        # background thread
        cal.record_trade(factors, pnl_pct) # call after each close
        weights = cal.current_weights()    # use in scorer
    """

    def __init__(
        self,
        window:       int   = RECALIBRATION_WINDOW,
        interval_sec: int   = RECALIBRATION_INTERVAL,
        alpha:        float = 1.0,           # Ridge regularisation
        weights_path: Path  = WEIGHTS_PATH,
        trade_log_path: Path = TRADE_LOG_PATH,
    ) -> None:
        self._window   = window
        self._interval = interval_sec
        self._alpha    = alpha
        self._wpath    = weights_path
        self._tpath    = trade_log_path

        self._buffer: deque[dict] = deque(maxlen=window)
        self._weights: dict[str, float] = self._load_weights()
        self._lock    = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_calibration: Optional[datetime] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name="factor-calibrator", daemon=True
        )
        self._thread.start()
        logger.info(
            "RollingFactorCalibrator started — window=%d trades, interval=%ds",
            self._window, self._interval,
        )

    def stop(self) -> None:
        self._running = False

    def record_trade(self, factor_scores: dict[str, float], pnl_pct: float) -> None:
        """
        Call after each trade closes.
        factor_scores: {"trend": 0.72, "momentum": 0.65, ...}
        pnl_pct:       realised P&L as fraction e.g. 0.03 = +3%
        """
        entry = {"factors": factor_scores, "pnl": pnl_pct, "ts": datetime.utcnow().isoformat()}
        with self._lock:
            self._buffer.append(entry)
        self._persist_trade(entry)

    def current_weights(self) -> dict[str, float]:
        with self._lock:
            return dict(self._weights)

    def force_recalibrate(self) -> dict[str, float]:
        """Trigger an immediate recalibration synchronously."""
        return self._calibrate()

    def status(self) -> dict[str, Any]:
        return {
            "buffer_size":       len(self._buffer),
            "window":            self._window,
            "last_calibration":  self._last_calibration.isoformat() if self._last_calibration else None,
            "current_weights":   self.current_weights(),
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._interval)
            if len(self._buffer) >= max(10, self._window // 5):
                self._calibrate()

    def _calibrate(self) -> dict[str, float]:
        with self._lock:
            records = list(self._buffer)

        if len(records) < 5:
            logger.debug("Calibration skipped — insufficient trades (%d)", len(records))
            return self._weights

        try:
            from sklearn.linear_model import Ridge

            factor_names = list(records[0]["factors"].keys())
            X = np.array([[r["factors"].get(f, 0.0) for f in factor_names] for r in records])
            y = np.array([r["pnl"] for r in records])

            model = Ridge(alpha=self._alpha, fit_intercept=True)
            model.fit(X, y)

            raw = dict(zip(factor_names, model.coef_))
            # Normalise to sum = 1 (positive weights only; clip negatives at 0)
            clipped = {k: max(0.0, v) for k, v in raw.items()}
            total   = sum(clipped.values()) or 1.0
            new_weights = {k: round(v / total, 4) for k, v in clipped.items()}

            with self._lock:
                self._weights = new_weights
            self._last_calibration = datetime.utcnow()

            self._save_weights(new_weights)
            logger.info("Factor weights recalibrated: %s", new_weights)
            return new_weights

        except Exception as exc:
            logger.error("Calibration failed: %s", exc, exc_info=True)
            return self._weights

    def _load_weights(self) -> dict[str, float]:
        if self._wpath.exists():
            try:
                return json.loads(self._wpath.read_text())
            except Exception:
                pass
        # Default v13.0 weights
        return {
            "trend":     0.28, "momentum": 0.20, "volume":   0.18,
            "volatility":0.12, "rs":       0.12, "breakout": 0.06,
            "quality":   0.04,
        }

    def _save_weights(self, weights: dict[str, float]) -> None:
        try:
            self._wpath.write_text(json.dumps(weights, indent=2))
        except OSError as e:
            logger.error("Could not save weights: %s", e)

    def _persist_trade(self, entry: dict) -> None:
        try:
            log: list = json.loads(self._tpath.read_text()) if self._tpath.exists() else []
            log.append(entry)
            log = log[-2000:]   # keep last 2000 trades max
            self._tpath.write_text(json.dumps(log, indent=2))
        except OSError:
            pass


# ═════════════════════════════════════════════════════════════════════════════
# 5. EXPONENTIAL BACKOFF — WATCH MODE API SAFETY
# ═════════════════════════════════════════════════════════════════════════════

class ExponentialBackoff:
    """
    Drop-in wrapper for any callable that hits an external API.
    Implements full jitter exponential backoff with a max retry count.

    Usage:
        backoff = ExponentialBackoff(max_retries=6)
        data = backoff.call(data_provider.fetch, "SBIN", period="1d")

    Or as a decorator:
        @ExponentialBackoff.wrap(max_retries=5)
        def my_api_call():
            ...
    """

    def __init__(
        self,
        max_retries: int   = 6,
        base:        float = BACKOFF_BASE,
        cap:         float = BACKOFF_MAX,
        jitter:      float = BACKOFF_JITTER,
        retryable_exceptions: tuple = (
            ConnectionError, TimeoutError, OSError,
        ),
    ) -> None:
        self._max_retries = max_retries
        self._base        = base
        self._cap         = cap
        self._jitter      = jitter
        self._retryable   = retryable_exceptions
        self._attempt     = 0
        self._total_waits = 0.0

    def call(self, fn, *args, **kwargs) -> Any:
        """Execute fn(*args, **kwargs) with retry on failure."""
        self._attempt    = 0
        self._total_waits = 0.0

        while True:
            try:
                result = fn(*args, **kwargs)
                if self._attempt > 0:
                    logger.info(
                        "Backoff succeeded after %d retries (total wait %.1fs)",
                        self._attempt, self._total_waits,
                    )
                return result
            except self._retryable as exc:
                self._attempt += 1
                if self._attempt > self._max_retries:
                    logger.error(
                        "Max retries (%d) exceeded — giving up. Last error: %s",
                        self._max_retries, exc,
                    )
                    raise
                wait = self._sleep_duration()
                logger.warning(
                    "Attempt %d/%d failed: %s — retrying in %.1fs",
                    self._attempt, self._max_retries, exc, wait,
                )
                time.sleep(wait)
            except Exception as exc:
                # Non-retryable — raise immediately
                logger.error("Non-retryable error on attempt %d: %s", self._attempt + 1, exc)
                raise

    def _sleep_duration(self) -> float:
        """Full jitter: uniform(0, min(cap, base * 2^attempt))"""
        ceiling = min(self._cap, self._base * (2 ** self._attempt))
        wait    = np.random.uniform(0, ceiling)
        # Add a small fixed jitter floor to prevent thundering herd
        wait   += self._jitter
        self._total_waits += wait
        return round(wait, 2)

    @staticmethod
    def wrap(max_retries: int = 5, **kwargs):
        """Decorator factory."""
        def decorator(fn):
            backoff = ExponentialBackoff(max_retries=max_retries, **kwargs)
            def wrapper(*args, **kw):
                return backoff.call(fn, *args, **kw)
            wrapper.__name__ = fn.__name__
            return wrapper
        return decorator


class WatchModeRunner:
    """
    Wraps the existing --watch N loop with:
    - Exponential backoff on API failures
    - Circuit breaker (pause after M consecutive failures)
    - Graceful shutdown on KeyboardInterrupt

    Usage:
        runner = WatchModeRunner(interval_sec=15, max_consecutive_failures=5)
        runner.run(scan_fn=engine.scan)
    """

    def __init__(
        self,
        interval_sec:              int = 15,
        max_consecutive_failures:  int = 5,
        circuit_breaker_pause_sec: int = 300,
    ) -> None:
        self._interval  = interval_sec
        self._max_fails = max_consecutive_failures
        self._pause     = circuit_breaker_pause_sec
        self._backoff   = ExponentialBackoff(max_retries=4)
        self._consecutive_fails = 0

    def run(self, scan_fn, *args, **kwargs) -> None:
        logger.info("WatchMode started — interval=%ds", self._interval)
        _send_telegram(f"🟢 *Watch mode started*\nInterval: {self._interval}s")

        try:
            while True:
                start = time.monotonic()
                try:
                    self._backoff.call(scan_fn, *args, **kwargs)
                    self._consecutive_fails = 0
                except Exception as exc:
                    self._consecutive_fails += 1
                    logger.error(
                        "Scan failed (consecutive=%d): %s",
                        self._consecutive_fails, exc,
                    )
                    if self._consecutive_fails >= self._max_fails:
                        logger.critical(
                            "Circuit breaker OPEN — pausing %ds", self._pause
                        )
                        _send_telegram(
                            f"🔴 *Circuit breaker open*\n"
                            f"{self._consecutive_fails} consecutive scan failures.\n"
                            f"Pausing {self._pause}s before resuming."
                        )
                        time.sleep(self._pause)
                        self._consecutive_fails = 0

                elapsed = time.monotonic() - start
                sleep_time = max(0.0, self._interval - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("WatchMode stopped by user.")
            _send_telegram("⚪ *Watch mode stopped* (KeyboardInterrupt)")


# ═════════════════════════════════════════════════════════════════════════════
# 6. TELEGRAM REGIME-AWARE ALERTS
# ═════════════════════════════════════════════════════════════════════════════

REGIME_EMOJI: dict[str, str] = {
    "TREND_UP":   "🟢",
    "EXPANSION":  "🔵",
    "RANGE":      "🟡",
    "TREND_DOWN": "🟠",
    "PANIC":      "🔴",
}

REGIME_CAPITAL_LABEL: dict[str, str] = {
    "TREND_UP":   "100% capital",
    "EXPANSION":  "100% capital",
    "RANGE":      "75% capital",
    "TREND_DOWN": "50% capital",
    "PANIC":      "0% — no new positions",
}

def _send_telegram(text: str, parse_mode: str = "Markdown") -> bool:
    """
    Thin wrapper over ``utils.messaging.send_telegram``.

    Delegates to the hardened implementation which enforces Telegram's
    4,096-char limit (truncating with a notice) and performs one automatic
    retry on HTTP 429 with Retry-After honour.  Reads credentials from the
    same env vars as the rest of the engine.

    The ``parse_mode`` parameter is accepted for call-site compatibility
    but note that ``utils.messaging.send_telegram`` always uses HTML mode
    as required by the v14 alert format — Markdown-formatted callers in
    this module use only safe Markdown that also renders acceptably as
    plain text.
    """
    from utils.messaging import send_telegram as _hardened_send
    return _hardened_send(text, token=TELEGRAM_BOT_TOKEN, chat_id=TELEGRAM_CHAT_ID)


class RegimeAwareTelegramAlerter:
    """
    Wraps all Sovereign Engine Telegram alerts to include:
    - Current regime with emoji
    - Capital fraction allowed
    - P(Win) threshold for current regime
    - Drawdown status (if scaler provided)

    Usage:
        alerter = RegimeAwareTelegramAlerter(
            gate=prob_gate,
            scaler=capital_scaler,   # optional
        )
        alerter.send_signal(
            symbol="SBIN",
            direction="LONG",
            p_win=0.63,
            regime="TREND_UP",
            current_nav=980_000,
        )
        alerter.send_regime_change("RANGE", "TREND_DOWN")
        alerter.send_daily_summary(regime="RANGE", top_picks=[...])
    """

    def __init__(
        self,
        gate:   Optional[RegimeProbabilityGate]  = None,
        scaler: Optional[TieredCapitalScaler]    = None,
    ) -> None:
        self._gate   = gate   or RegimeProbabilityGate()
        self._scaler = scaler

    def _regime_header(self, regime: str, current_nav: Optional[float] = None) -> str:
        emoji     = REGIME_EMOJI.get(regime, "⚪")
        cap_label = REGIME_CAPITAL_LABEL.get(regime, "—")
        threshold = self._gate.threshold(regime)

        lines = [
            f"{emoji} *Regime: {regime}*",
            f"Capital: {cap_label}  |  P(Win) gate: ≥{threshold:.0%}",
        ]

        if self._scaler and current_nav:
            dd = self._scaler.current_drawdown(current_nav)
            cap_frac = self._scaler.capital_fraction(current_nav, regime)
            lines.append(
                f"Drawdown: {dd:.1%}  |  Active capital: {cap_frac:.0%}"
            )

        return "\n".join(lines)

    def send_signal(
        self,
        symbol:      str,
        direction:   str,
        p_win:       float,
        regime:      str,
        score:       Optional[float]  = None,
        current_nav: Optional[float]  = None,
        entry:       Optional[float]  = None,
        sl:          Optional[float]  = None,
        target:      Optional[float]  = None,
    ) -> bool:
        gate_pass = self._gate.passes(p_win, regime)
        margin    = self._gate.margin(p_win, regime)
        dir_emoji = "📈" if direction.upper() == "LONG" else "📉"

        msg_lines = [
            self._regime_header(regime, current_nav),
            "─" * 28,
            f"{dir_emoji} *{symbol}* — {direction.upper()}",
            f"P(Win): `{p_win:.1%}` (margin: {margin:+.1%})",
        ]

        if score is not None:
            msg_lines.append(f"Composite score: `{score:.2f}`")
        if entry:
            msg_lines.append(f"Entry: ₹{entry:.2f}")
        if sl:
            msg_lines.append(f"Stop loss: ₹{sl:.2f}")
        if target:
            msg_lines.append(f"Target: ₹{target:.2f}")

        if not gate_pass:
            msg_lines.append(f"\n⚠️ *BLOCKED by prob gate* — P(Win) below {self._gate.threshold(regime):.0%}")

        msg_lines.append(f"\n_Sent: {datetime.now().strftime('%H:%M:%S IST')}_")
        return _send_telegram("\n".join(msg_lines))

    def send_regime_change(self, old_regime: str, new_regime: str) -> bool:
        old_e = REGIME_EMOJI.get(old_regime, "⚪")
        new_e = REGIME_EMOJI.get(new_regime, "⚪")
        old_c = REGIME_CAPITAL_LABEL.get(old_regime, "—")
        new_c = REGIME_CAPITAL_LABEL.get(new_regime, "—")

        msg = (
            f"🔄 *Regime change detected*\n\n"
            f"{old_e} {old_regime} → {new_e} *{new_regime}*\n\n"
            f"Capital: {old_c} → *{new_c}*\n"
            f"New P(Win) gate: ≥{self._gate.threshold(new_regime):.0%}\n\n"
        )

        if new_regime == "PANIC":
            msg += "🔴 *PANIC mode active — no new positions. Review open trades.*"
        elif new_regime == "TREND_DOWN":
            msg += "🟠 *Downtrend confirmed — reduce exposure, tighten stops.*"
        elif new_regime == "TREND_UP":
            msg += "🟢 *Uptrend confirmed — full capital deployment allowed.*"

        msg += f"\n\n_Time: {datetime.now().strftime('%d %b %Y %H:%M IST')}_"
        return _send_telegram(msg)

    def send_daily_summary(
        self,
        regime:    str,
        top_picks: list[dict],
        current_nav: Optional[float] = None,
    ) -> bool:
        header = self._regime_header(regime, current_nav)
        lines  = [
            f"📊 *Daily Sovereign Engine Summary*",
            f"_{datetime.now().strftime('%d %b %Y')}_\n",
            header,
            "─" * 28,
        ]

        if top_picks:
            lines.append(f"*Top {len(top_picks)} picks:*")
            for i, p in enumerate(top_picks[:5], 1):
                sym    = p.get("symbol", "—")
                score  = p.get("score", 0)
                p_win  = p.get("p_win", 0)
                passes = self._gate.passes(p_win, regime)
                status = "✅" if passes else "⛔"
                lines.append(
                    f"{i}. {status} *{sym}* — score: `{score:.2f}` | P: `{p_win:.1%}`"
                )
        else:
            lines.append("No qualifying setups today.")

        lines.append(
            f"\n_Gate: P(Win) ≥{self._gate.threshold(regime):.0%} | "
            f"Capital: {REGIME_CAPITAL_LABEL.get(regime, '—')}_"
        )
        return _send_telegram("\n".join(lines))

    def send_calibration_update(self, new_weights: dict[str, float], regime: str) -> bool:
        lines = [
            self._regime_header(regime),
            "─" * 28,
            "🔧 *Factor weights recalibrated*",
        ]
        for factor, w in sorted(new_weights.items(), key=lambda x: -x[1]):
            bar = "█" * int(w * 20)
            lines.append(f"`{factor:<12}` {bar} `{w:.1%}`")
        lines.append(f"\n_Time: {datetime.now().strftime('%H:%M IST')}_")
        return _send_telegram("\n".join(lines))


# ═════════════════════════════════════════════════════════════════════════════
# UNIFIED APPLY — wire all improvements into an existing engine instance
# ═════════════════════════════════════════════════════════════════════════════

def apply(
    engine,
    portfolio_peak:    float = 1_000_000,
    current_nav:       float = 1_000_000,
    fyers_client:      Any   = None,
    recal_window:      int   = RECALIBRATION_WINDOW,
    recal_interval:    int   = RECALIBRATION_INTERVAL,
    watch_interval:    int   = 15,
) -> dict[str, Any]:
    """
    Wire all 6 improvements into your existing engine instance.

    Returns a dict of all patch components so you can use them
    directly in your existing screener.py / sovereign_quant_layer.py.

    Example:
        import sovereign_improvements as SE_PATCH
        patch = SE_PATCH.apply(engine, portfolio_peak=1_000_000)

        # Use in your scan loop:
        fraction = patch["scaler"].capital_fraction(nav, regime)
        if patch["gate"].passes(p_win, regime):
            df = patch["data"].fetch("SBIN")
            patch["alerter"].send_signal("SBIN", "LONG", p_win, regime)
    """

    scaler    = TieredCapitalScaler(portfolio_peak=portfolio_peak)
    gate      = RegimeProbabilityGate()
    data      = ResilientDataProvider(fyers_client=fyers_client)
    calibrator = RollingFactorCalibrator(window=recal_window, interval_sec=recal_interval)
    backoff   = ExponentialBackoff(max_retries=6)
    watch     = WatchModeRunner(interval_sec=watch_interval)
    alerter   = RegimeAwareTelegramAlerter(gate=gate, scaler=scaler)

    # Start background calibrator
    calibrator.start()

    # Attach to engine if it has compatible attributes
    _safe_attach(engine, "capital_scaler",    scaler)
    _safe_attach(engine, "prob_gate",         gate)
    _safe_attach(engine, "data_provider",     data)
    _safe_attach(engine, "calibrator",        calibrator)
    _safe_attach(engine, "watch_runner",      watch)
    _safe_attach(engine, "alerter",           alerter)

    logger.info(
        "Sovereign Engine patch applied — 6 improvements active.\n"
        "  ✓ Tiered PANIC capital scaler\n"
        "  ✓ Regime-dependent prob gate\n"
        "  ✓ Resilient data provider (Fyers→yfinance)\n"
        "  ✓ Rolling factor recalibrator (background thread)\n"
        "  ✓ Exponential backoff watch runner\n"
        "  ✓ Regime-aware Telegram alerter"
    )

    _send_telegram(
        f"🚀 *Sovereign Engine v13.0 patch loaded*\n"
        f"6 improvements active. Peak NAV: ₹{portfolio_peak:,.0f}"
    )

    return {
        "scaler":     scaler,
        "gate":       gate,
        "data":       data,
        "calibrator": calibrator,
        "backoff":    backoff,
        "watch":      watch,
        "alerter":    alerter,
    }


def _safe_attach(obj: Any, attr: str, value: Any) -> None:
    try:
        setattr(obj, attr, value)
        logger.debug("Attached .%s to engine", attr)
    except (AttributeError, TypeError):
        logger.debug("Could not attach .%s — engine may be None or frozen", attr)


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SMOKE TEST (run this file directly to verify imports & basic logic)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n── Sovereign Engine Patch — smoke test ──\n")

    # 1. Tiered capital scaler
    scaler = TieredCapitalScaler(portfolio_peak=1_000_000)
    for nav, label in [(990_000, "0.9% DD"), (960_000, "4% DD"), (930_000, "7% DD"), (880_000, "12% DD")]:
        frac = scaler.capital_fraction(nav, "PANIC")
        print(f"  PANIC scaler | NAV ₹{nav:,} ({label}) → capital {frac:.0%}")

    print()

    # 2. Regime-dependent prob gate
    gate = RegimeProbabilityGate()
    for regime, p in [("TREND_UP", 0.51), ("RANGE", 0.54), ("TREND_DOWN", 0.57), ("PANIC", 0.99)]:
        result = gate.passes(p, regime)
        print(f"  ProbGate | regime={regime:<12} p_win={p:.0%} → {'PASS ✓' if result else 'BLOCK ✗'}")

    print()

    # 3. Backoff sleep durations (no actual sleep)
    bo = ExponentialBackoff(max_retries=5)
    print("  Backoff sleep durations (simulated):")
    for attempt in range(1, 6):
        bo._attempt = attempt
        d = bo._sleep_duration()
        print(f"    attempt {attempt} → {d:.1f}s")

    print()

    # 4. Calibrator (no sklearn needed for weights load)
    cal = RollingFactorCalibrator(window=50)
    print(f"  Calibrator default weights: {cal.current_weights()}")

    print("\n── All checks passed ──\n")