"""
core/runtime_components.py
==========================
Core-native runtime collaborators for the modular Sovereign Engine.

This module absorbs the live runtime pieces that were previously built out of
``sovereign_improvements.py`` so the services layer can depend on first-class
core modules rather than a legacy patch bundle.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .runtime_paths import RUNTIME_PATHS, ensure_parent, ensure_runtime_dirs

logger = logging.getLogger("sovereign.runtime")

REGIME_GATE: dict[str, float] = {
    "TREND_UP": 0.50,
    "EXPANSION": 0.52,
    "RANGE": 0.55,
    "TREND_DOWN": 0.58,
    "PANIC": 0.99,
}

REGIME_CAPITAL: dict[str, float] = {
    "TREND_UP": 1.00,
    "EXPANSION": 1.00,
    "RANGE": 0.75,
    "TREND_DOWN": 0.50,
    "PANIC": 0.00,
}

REGIME_CAPITAL_LABEL: dict[str, str] = {
    "TREND_UP": "100% capital",
    "EXPANSION": "100% capital",
    "RANGE": "75% capital",
    "TREND_DOWN": "50% capital",
    "PANIC": "0% - no new positions",
}

PANIC_TIERS: list[tuple[float, float]] = [
    (0.00, 1.00),
    (0.02, 0.50),
    (0.05, 0.25),
    (0.10, 0.00),
]

RECALIBRATION_WINDOW = int(os.getenv("RECAL_WINDOW", "50"))
RECALIBRATION_INTERVAL = int(os.getenv("RECAL_INTERVAL", "300"))

WEIGHTS_PATH = RUNTIME_PATHS.factor_weights_file
TRADE_LOG_PATH = RUNTIME_PATHS.trade_log_file


def _regime_str(regime: Any) -> str:
    value = getattr(regime, "value", None)
    return str(value if value is not None else regime)


def _send_telegram(text: str) -> bool:
    from utils.messaging import send_telegram

    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        logger.debug("Telegram credentials missing; skipping runtime component alert.")
        return False
    return send_telegram(text, token=token, chat_id=chat_id)


class TieredCapitalScaler:
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
        regime_name = _regime_str(regime)
        self.update_peak(current_nav)
        dd = self.current_drawdown(current_nav)

        if regime_name not in ("PANIC", "TREND_DOWN"):
            base = REGIME_CAPITAL.get(regime_name, 1.0)
            if dd > 0.02:
                base *= max(0.5, 1.0 - dd * 3)
            return round(base, 4)

        fraction = 0.0
        for threshold, cap in reversed(PANIC_TIERS):
            if dd >= threshold:
                fraction = cap
                break

        logger.warning(
            "PANIC scaler: drawdown=%.2f%% -> capital_fraction=%.0f%%",
            dd * 100,
            fraction * 100,
        )
        return fraction


class RegimeProbabilityGate:
    def __init__(self, overrides: Optional[dict[str, float]] = None) -> None:
        self._gates = {**REGIME_GATE, **(overrides or {})}

    def threshold(self, regime: str) -> float:
        return self._gates.get(_regime_str(regime), 0.52)

    def passes(self, p_win: float, regime: str) -> bool:
        return p_win >= self.threshold(regime)

    def margin(self, p_win: float, regime: str) -> float:
        return round(p_win - self.threshold(regime), 4)


class RollingFactorCalibrator:
    def __init__(
        self,
        window: int = RECALIBRATION_WINDOW,
        interval_sec: int = RECALIBRATION_INTERVAL,
        alpha: float = 1.0,
        weights_path: Path = WEIGHTS_PATH,
        trade_log_path: Path = TRADE_LOG_PATH,
    ) -> None:
        self._window = window
        self._interval = interval_sec
        self._alpha = alpha
        self._wpath = weights_path
        self._tpath = trade_log_path

        self._buffer: deque[dict[str, Any]] = deque(maxlen=window)
        self._weights: dict[str, float] = self._load_weights()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_calibration: Optional[datetime] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="factor-calibrator", daemon=True)
        self._thread.start()
        logger.info(
            "RollingFactorCalibrator started - window=%d trades, interval=%ds",
            self._window,
            self._interval,
        )

    def stop(self) -> None:
        self._running = False

    def record_trade(self, factor_scores: dict[str, float], pnl_pct: float) -> None:
        entry = {"factors": factor_scores, "pnl": pnl_pct, "ts": datetime.utcnow().isoformat()}
        with self._lock:
            self._buffer.append(entry)
        self._persist_trade(entry)

    def current_weights(self) -> dict[str, float]:
        with self._lock:
            return dict(self._weights)

    def _loop(self) -> None:
        while self._running:
            time.sleep(self._interval)
            if len(self._buffer) >= max(10, self._window // 5):
                self._calibrate()

    def _calibrate(self) -> dict[str, float]:
        with self._lock:
            records = list(self._buffer)

        if len(records) < 5:
            return self._weights

        try:
            from sklearn.linear_model import Ridge

            factor_names = list(records[0]["factors"].keys())
            x_values = np.array([[record["factors"].get(name, 0.0) for name in factor_names] for record in records])
            y_values = np.array([record["pnl"] for record in records])

            model = Ridge(alpha=self._alpha, fit_intercept=True)
            model.fit(x_values, y_values)

            raw = dict(zip(factor_names, model.coef_))
            clipped = {key: max(0.0, value) for key, value in raw.items()}
            total = sum(clipped.values()) or 1.0
            new_weights = {key: round(value / total, 4) for key, value in clipped.items()}

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
                loaded = json.loads(self._wpath.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return {str(key): float(value) for key, value in loaded.items()}
            except Exception:
                pass
        return {
            "trend": 0.28,
            "momentum": 0.20,
            "volume": 0.18,
            "volatility": 0.12,
            "rs": 0.12,
            "breakout": 0.06,
            "quality": 0.04,
        }

    def _save_weights(self, weights: dict[str, float]) -> None:
        try:
            ensure_parent(self._wpath).write_text(json.dumps(weights, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.error("Could not save weights: %s", exc)

    def _persist_trade(self, entry: dict[str, Any]) -> None:
        try:
            if self._tpath.exists():
                loaded = json.loads(self._tpath.read_text(encoding="utf-8"))
                log_entries = [item for item in loaded if isinstance(item, dict)] if isinstance(loaded, list) else []
            else:
                log_entries = []
            log_entries.append(entry)
            ensure_parent(self._tpath).write_text(json.dumps(log_entries[-2000:], indent=2), encoding="utf-8")
        except OSError:
            pass


class RegimeAwareTelegramAlerter:
    def __init__(
        self,
        gate: Optional[RegimeProbabilityGate] = None,
        scaler: Optional[TieredCapitalScaler] = None,
    ) -> None:
        self._gate = gate or RegimeProbabilityGate()
        self._scaler = scaler

    def send_daily_summary(
        self,
        regime: str,
        top_picks: list[dict[str, Any]],
        current_nav: Optional[float] = None,
    ) -> bool:
        regime_name = _regime_str(regime)
        cap_label = REGIME_CAPITAL_LABEL.get(regime_name, "-")
        lines = [
            f"Regime: {regime_name}",
            f"Capital: {cap_label} | P(Win) gate: >={self._gate.threshold(regime_name):.0%}",
        ]

        if self._scaler is not None and current_nav is not None:
            dd = self._scaler.current_drawdown(current_nav)
            cap_frac = self._scaler.capital_fraction(current_nav, regime_name)
            lines.append(f"Drawdown: {dd:.1%} | Active capital: {cap_frac:.0%}")

        if top_picks:
            lines.append("")
            lines.append(f"Top {len(top_picks)} picks:")
            for index, pick in enumerate(top_picks[:5], 1):
                symbol = pick.get("ticker", "-")
                p_win = float(pick.get("prob_win", 0.0))
                passes = self._gate.passes(p_win, regime_name)
                status = "OK" if passes else "BLOCKED"
                lines.append(f"{index}. {status} {symbol} | P={p_win:.1%}")
        else:
            lines.append("")
            lines.append("No qualifying setups today.")

        return _send_telegram("\n".join(lines))


@dataclass(frozen=True)
class RuntimeComponents:
    """Immutable container of live runtime collaborators.

    Supports the context-manager protocol so tests and short-lived
    scripts can reliably stop the background calibrator thread::

        with create_runtime_components() as components:
            ...
        # calibrator thread stopped automatically on exit
    """
    gate: RegimeProbabilityGate
    scaler: TieredCapitalScaler
    calibrator: RollingFactorCalibrator
    alerter: RegimeAwareTelegramAlerter

    def stop(self) -> None:
        """Stop the background calibrator thread."""
        self.calibrator.stop()

    def __enter__(self) -> "RuntimeComponents":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def create_runtime_components(
    *,
    portfolio_peak: float = 1_000_000.0,
    recal_window: int = RECALIBRATION_WINDOW,
    recal_interval: int = RECALIBRATION_INTERVAL,
    announce: bool = False,
) -> RuntimeComponents:
    # Ensure state/artifacts/logs dirs exist exactly once, at the point
    # where runtime components are intentionally constructed — not at import.
    ensure_runtime_dirs()
    scaler = TieredCapitalScaler(portfolio_peak=portfolio_peak)
    gate = RegimeProbabilityGate()
    calibrator = RollingFactorCalibrator(window=recal_window, interval_sec=recal_interval)
    alerter = RegimeAwareTelegramAlerter(gate=gate, scaler=scaler)

    calibrator.start()

    if announce:
        logger.info(
            "Runtime components initialised: gate, tiered capital scaler, calibrator, alerter."
        )

    return RuntimeComponents(
        gate=gate,
        scaler=scaler,
        calibrator=calibrator,
        alerter=alerter,
    )
