"""
core/telemetry.py
=================
Structured logging and scan-metrics collection for the Sovereign Engine.

Why structured logging?
------------------------
Plain-text log lines are hard to parse, aggregate, and alert on. This
module emits every significant event as a JSON line on a dedicated
``sovereign.metrics`` logger, making it trivially importable by any log
aggregator (Loki, CloudWatch, Datadog, even ``grep | jq``).

A human-readable handler is kept on the root ``sovereign`` logger so
existing ``logging.basicConfig`` output is unchanged.

Usage
-----
    from core.telemetry import setup_logging, ScanMetrics, emit

    setup_logging(level="INFO", json_log_file="logs/sovereign.jsonl")

    metrics = ScanMetrics()
    metrics.record_fetch(n_ok=120, n_fail=3, elapsed_s=4.2)
    metrics.record_score(n_passed=18, n_total=120, elapsed_s=2.1)
    metrics.record_portfolio(portfolio)
    metrics.emit_summary()          # → JSON line to sovereign.metrics logger
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


# ── Logger names ──────────────────────────────────────────────────────────────
ROOT_LOGGER   = "sovereign"
METRIC_LOGGER = "sovereign.metrics"

# ── JSON line formatter ────────────────────────────────────────────────────────

class _JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single JSON object on stdout/file.

    Fields always present
    ---------------------
    ts      ISO-8601 UTC timestamp
    level   DEBUG / INFO / WARNING / ERROR / CRITICAL
    logger  dotted logger name
    msg     human-readable message
    """

    def format(self, record: logging.LogRecord) -> str:
        doc: dict[str, Any] = {
            "ts":     datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        # Attach any extra keys passed via extra={} or record.__dict__
        for key, val in record.__dict__.items():
            if key.startswith("_") or key in (
                "msg", "args", "levelname", "levelno", "pathname", "filename",
                "module", "exc_info", "exc_text", "stack_info", "lineno",
                "funcName", "created", "msecs", "relativeCreated", "thread",
                "threadName", "processName", "process", "name", "taskName",
            ):
                continue
            doc[key] = val

        if record.exc_info:
            doc["exc"] = self.formatException(record.exc_info)

        return json.dumps(doc, default=str)


# ── Public setup ──────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    json_log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,   # 10 MB
    backup_count: int = 5,
) -> None:
    """
    Configure the ``sovereign`` logger hierarchy.

    Parameters
    ----------
    level:          Root log level (DEBUG / INFO / WARNING / ERROR).
    json_log_file:  Path to a rotating JSON-lines file.  ``None`` → no
                    file handler (metrics still go to stderr via JSON).
    max_bytes:      Rotate the JSON log after this many bytes.
    backup_count:   Keep this many rotated files.

    Call once at application start-up, before any other imports from
    ``core``.  Calling a second time is a no-op (handlers are only added
    once).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger(ROOT_LOGGER)
    if root.handlers:
        return   # already configured — skip

    root.setLevel(numeric_level)

    # Human-readable handler → stderr
    human_handler = logging.StreamHandler(sys.stderr)
    human_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    )
    root.addHandler(human_handler)

    # JSON handler for metrics logger
    metric_log = logging.getLogger(METRIC_LOGGER)
    metric_log.propagate = False   # don't duplicate to human handler

    if json_log_file:
        log_path = Path(json_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        fh.setFormatter(_JsonFormatter())
        metric_log.addHandler(fh)
    else:
        # Fallback: JSON to stderr so metrics are never silently dropped
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(_JsonFormatter())
        metric_log.addHandler(sh)

    metric_log.setLevel(logging.DEBUG)


def emit(event: str, **kwargs: Any) -> None:
    """
    Emit one structured metric event.

    Parameters
    ----------
    event:   Short, snake_case event name (e.g. ``"scan_complete"``).
    kwargs:  Arbitrary key-value payload attached to the JSON record.

    Example
    -------
    ::

        emit("ticker_scored", ticker="RELIANCE", prob=0.63, composite=0.71)
    """
    log = logging.getLogger(METRIC_LOGGER)
    # Build a flat dict so the JSON formatter picks it up via extra fields
    extra = {"event": event, **kwargs}
    log.info(event, extra=extra)


# ── Scan metrics accumulator ──────────────────────────────────────────────────

@dataclass
class ScanMetrics:
    """
    Accumulates per-scan statistics and emits a summary JSON event.

    Typical lifecycle
    -----------------
    ::

        m = ScanMetrics()
        m.record_fetch(n_ok=120, n_fail=3, elapsed_s=4.2)
        m.record_indicators(n_ok=118, n_fail=2, elapsed_s=0.9)
        m.record_score(n_passed=18, n_total=118, elapsed_s=2.1)
        m.record_regime(regime)
        m.record_portfolio(portfolio)
        m.emit_summary()
    """

    scan_id:       str   = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    start_time:    float = field(default_factory=time.monotonic)

    # Fetch stage
    fetch_n_ok:    int   = 0
    fetch_n_fail:  int   = 0
    fetch_elapsed: float = 0.0

    # Indicator stage
    ind_n_ok:      int   = 0
    ind_n_fail:    int   = 0
    ind_elapsed:   float = 0.0

    # Scoring stage
    score_n_passed: int  = 0
    score_n_total:  int  = 0
    score_elapsed:  float = 0.0

    # Regime
    regime:        str   = ""
    regime_conf:   float = 0.0
    breadth:       float = 0.0

    # Portfolio
    portfolio_size: int  = 0
    portfolio_tickers: list[str] = field(default_factory=list)
    mean_prob_win:  float = 0.0
    mean_exp_r:     float = 0.0

    def record_fetch(self, n_ok: int, n_fail: int, elapsed_s: float) -> None:
        self.fetch_n_ok    = n_ok
        self.fetch_n_fail  = n_fail
        self.fetch_elapsed = round(elapsed_s, 3)
        emit("fetch_complete", scan_id=self.scan_id,
             n_ok=n_ok, n_fail=n_fail, elapsed_s=elapsed_s)

    def record_indicators(self, n_ok: int, n_fail: int, elapsed_s: float) -> None:
        self.ind_n_ok    = n_ok
        self.ind_n_fail  = n_fail
        self.ind_elapsed = round(elapsed_s, 3)
        emit("indicators_complete", scan_id=self.scan_id,
             n_ok=n_ok, n_fail=n_fail, elapsed_s=elapsed_s)

    def record_score(self, n_passed: int, n_total: int, elapsed_s: float) -> None:
        self.score_n_passed = n_passed
        self.score_n_total  = n_total
        self.score_elapsed  = round(elapsed_s, 3)
        emit("score_complete", scan_id=self.scan_id,
             n_passed=n_passed, n_total=n_total,
             pass_rate=round(n_passed / max(n_total, 1), 4),
             elapsed_s=elapsed_s)

    def record_regime(self, regime: Any) -> None:
        """Accept a ``MarketRegime`` dataclass instance."""
        self.regime      = str(getattr(regime, "regime", regime))
        self.regime_conf = float(getattr(regime, "confidence", 0.0))
        self.breadth     = float(getattr(regime, "breadth", 0.0))
        emit("regime_classified", scan_id=self.scan_id,
             regime=self.regime, confidence=self.regime_conf, breadth=self.breadth)

    def record_portfolio(self, portfolio: list) -> None:
        """Accept a list of ``TickerResult`` instances."""
        self.portfolio_size    = len(portfolio)
        self.portfolio_tickers = [r.ticker for r in portfolio]
        self.mean_prob_win     = round(
            sum(r.prob_win for r in portfolio) / max(len(portfolio), 1), 4
        )
        self.mean_exp_r = round(
            sum(r.expectancy_r for r in portfolio) / max(len(portfolio), 1), 4
        )
        emit("portfolio_built", scan_id=self.scan_id,
             size=self.portfolio_size, tickers=self.portfolio_tickers,
             mean_prob_win=self.mean_prob_win, mean_exp_r=self.mean_exp_r)

    def emit_summary(self) -> None:
        """Emit the full scan summary as one JSON event."""
        total_elapsed = round(time.monotonic() - self.start_time, 3)
        emit(
            "scan_complete",
            **{k: v for k, v in asdict(self).items() if k != "start_time"},
            total_elapsed_s=total_elapsed,
        )
        logging.getLogger(ROOT_LOGGER).info(
            "Scan %s complete in %.1fs | fetch=%d ok/%d fail | "
            "scored=%d/%d | portfolio=%d | regime=%s (conf=%.2f)",
            self.scan_id, total_elapsed,
            self.fetch_n_ok, self.fetch_n_fail,
            self.score_n_passed, self.score_n_total,
            self.portfolio_size, self.regime, self.regime_conf,
        )
