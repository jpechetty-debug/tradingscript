"""
core/retry.py
=============
Retry logic with exponential back-off and a circuit-breaker for all
external API calls (Fyers, yfinance, Telegram).

Design
------
* ``retry_with_backoff`` — decorator / context-free function that retries
  a callable on transient errors with jittered exponential back-off.
* ``CircuitBreaker`` — per-resource breaker with CLOSED → OPEN → HALF-OPEN
  state machine.  Prevents cascading failures when a downstream API goes
  down for an extended period.
* ``guarded_call`` — thin wrapper that composes both: retries inside the
  breaker so the breaker only opens on *persistent* failures, not transient
  ones that self-heal within the retry budget.

Usage
-----
::

    from core.retry import guarded_call, CircuitBreaker

    fyers_breaker = CircuitBreaker("fyers", failure_threshold=5, reset_timeout=60)

    def _fetch():
        return fyers.history(data=request)

    result = guarded_call(_fetch, breaker=fyers_breaker,
                          max_attempts=3, base_delay=1.0)
"""

from __future__ import annotations

import logging
import random
import time
from enum import Enum, auto
from typing import Any, Callable, Optional, Type

log = logging.getLogger("sovereign.retry")


# ─────────────────────────────────────────────────────────────────────────────
# RETRY WITH EXPONENTIAL BACK-OFF
# ─────────────────────────────────────────────────────────────────────────────

class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts are exhausted."""


def retry_with_backoff(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base_delay:   float = 1.0,
    max_delay:    float = 30.0,
    backoff_mult: float = 2.0,
    jitter:       float = 0.3,
    retryable:    tuple[Type[Exception], ...] = (Exception,),
    label:        str = "",
) -> Any:
    """
    Call *fn* up to *max_attempts* times, sleeping between failures.

    Parameters
    ----------
    fn:           Zero-argument callable to execute.
    max_attempts: Total attempts (1 = no retry).
    base_delay:   Initial sleep before second attempt (seconds).
    max_delay:    Cap on sleep time.
    backoff_mult: Multiplier applied to delay after each failure.
    jitter:       Fraction of delay to randomise (avoids thundering herd).
    retryable:    Exception types that trigger a retry.  Non-matching
                  exceptions propagate immediately.
    label:        Human-readable name for log messages.

    Returns
    -------
    Return value of *fn* on the first successful call.

    Raises
    ------
    MaxRetriesExceeded
        If all attempts fail with a retryable exception.
    Any non-retryable exception
        Propagated immediately from the first occurrence.
    """
    delay = base_delay
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except retryable as exc:
            last_exc = exc
            if attempt == max_attempts:
                break
            sleep_time = min(delay * (1 + jitter * (2 * random.random() - 1)), max_delay)
            log.warning(
                "%s attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                label or fn.__name__,
                attempt, max_attempts,
                type(exc).__name__, exc,
                sleep_time,
            )
            time.sleep(sleep_time)
            delay = min(delay * backoff_mult, max_delay)

    raise MaxRetriesExceeded(
        f"{label or fn.__name__} failed after {max_attempts} attempts"
    ) from last_exc


# ─────────────────────────────────────────────────────────────────────────────
# CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────

class BreakerState(Enum):
    CLOSED    = auto()   # Normal operation — requests pass through
    OPEN      = auto()   # Tripped — requests fail immediately
    HALF_OPEN = auto()   # Probe — one request allowed to test recovery


class CircuitBreakerOpen(Exception):
    """Raised when a call is rejected because the breaker is OPEN."""


class CircuitBreaker:
    """
    Per-resource circuit breaker with CLOSED → OPEN → HALF-OPEN state machine.

    Transition rules
    ----------------
    CLOSED  → OPEN      after *failure_threshold* consecutive failures.
    OPEN    → HALF-OPEN after *reset_timeout* seconds.
    HALF-OPEN → CLOSED  on one successful probe.
    HALF-OPEN → OPEN    on one failed probe.

    Parameters
    ----------
    name:              Human-readable resource name (e.g. ``"fyers"``).
    failure_threshold: Consecutive failures to trip the breaker.
    reset_timeout:     Seconds to wait before trying HALF-OPEN probe.
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
    ) -> None:
        self.name              = name
        self.failure_threshold = failure_threshold
        self.reset_timeout     = reset_timeout

        self._state:      BreakerState    = BreakerState.CLOSED
        self._failures:   int             = 0
        self._opened_at:  Optional[float] = None

    # ── Public interface ──────────────────────────────────────────────────────

    @property
    def state(self) -> BreakerState:
        self._maybe_transition_half_open()
        return self._state

    def call(self, fn: Callable[[], Any]) -> Any:
        """
        Execute *fn* through the breaker.

        Raises
        ------
        CircuitBreakerOpen
            If the breaker is OPEN and the reset timeout has not elapsed.
        Any exception from *fn*
            Recorded as a failure and re-raised.
        """
        current = self.state   # may transition OPEN → HALF-OPEN

        if current == BreakerState.OPEN:
            raise CircuitBreakerOpen(
                f"Circuit breaker '{self.name}' is OPEN — calls blocked"
            )

        try:
            result = fn()
            self._on_success()
            return result
        except Exception as exc:
            self._on_failure()
            raise exc

    def record_success(self) -> None:
        """Manually record a success (e.g. from an async path)."""
        self._on_success()

    def record_failure(self) -> None:
        """Manually record a failure (e.g. from an async path)."""
        self._on_failure()

    def reset(self) -> None:
        """Force-reset to CLOSED (useful in tests or after maintenance)."""
        self._state    = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = None
        log.info("CircuitBreaker '%s' manually reset to CLOSED.", self.name)

    # ── Internal state transitions ────────────────────────────────────────────

    def _on_success(self) -> None:
        if self._state == BreakerState.HALF_OPEN:
            log.info(
                "CircuitBreaker '%s': probe succeeded — returning to CLOSED.", self.name
            )
        self._state    = BreakerState.CLOSED
        self._failures = 0
        self._opened_at = None

    def _on_failure(self) -> None:
        self._failures += 1
        if self._state == BreakerState.HALF_OPEN:
            log.warning(
                "CircuitBreaker '%s': probe failed — re-opening.", self.name
            )
            self._trip()
            return

        if self._failures >= self.failure_threshold:
            log.error(
                "CircuitBreaker '%s': %d consecutive failures — OPENING (timeout=%.0fs).",
                self.name, self._failures, self.reset_timeout,
            )
            self._trip()

    def _trip(self) -> None:
        self._state     = BreakerState.OPEN
        self._opened_at = time.monotonic()

    def _maybe_transition_half_open(self) -> None:
        if (
            self._state == BreakerState.OPEN
            and self._opened_at is not None
            and (time.monotonic() - self._opened_at) >= self.reset_timeout
        ):
            log.info(
                "CircuitBreaker '%s': reset timeout elapsed — entering HALF-OPEN.", self.name
            )
            self._state = BreakerState.HALF_OPEN

    # ── String representation ─────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"CircuitBreaker(name={self.name!r}, state={self._state.name}, "
            f"failures={self._failures}, threshold={self.failure_threshold})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSED: RETRY + CIRCUIT BREAKER
# ─────────────────────────────────────────────────────────────────────────────

def guarded_call(
    fn: Callable[[], Any],
    *,
    breaker:      Optional[CircuitBreaker] = None,
    max_attempts: int = 3,
    base_delay:   float = 1.0,
    max_delay:    float = 30.0,
    backoff_mult: float = 2.0,
    jitter:       float = 0.3,
    retryable:    tuple[Type[Exception], ...] = (Exception,),
    label:        str = "",
) -> Any:
    """
    Execute *fn* with retry + circuit breaker protection.

    The retry loop runs *inside* the breaker: transient errors that
    succeed within the retry budget never count as breaker failures.
    Only when all retry attempts are exhausted does the breaker
    record a failure.

    Parameters
    ----------
    fn, max_attempts, base_delay, max_delay, backoff_mult, jitter, retryable, label:
        Forwarded verbatim to ``retry_with_backoff``.
    breaker:
        Optional ``CircuitBreaker`` instance.  Pass ``None`` to skip
        circuit-breaker logic (pure retry).

    Returns
    -------
    Return value of *fn* on success.

    Raises
    ------
    CircuitBreakerOpen
        If the breaker is OPEN before the first attempt.
    MaxRetriesExceeded
        If all retry attempts fail and the breaker trips.
    """
    # Fail fast if breaker is already OPEN
    if breaker is not None and breaker.state == BreakerState.OPEN:
        raise CircuitBreakerOpen(
            f"Circuit breaker '{breaker.name}' is OPEN — call blocked before retry"
        )

    try:
        result = retry_with_backoff(
            fn,
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            backoff_mult=backoff_mult,
            jitter=jitter,
            retryable=retryable,
            label=label,
        )
        if breaker is not None:
            breaker.record_success()
        return result

    except MaxRetriesExceeded:
        if breaker is not None:
            breaker.record_failure()
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PRE-BUILT BREAKERS (shared across the application)
# ─────────────────────────────────────────────────────────────────────────────

#: Breaker for Fyers API calls.  Trips after 5 consecutive failures;
#: probes again after 60 s.
FYERS_BREAKER = CircuitBreaker("fyers", failure_threshold=5, reset_timeout=60.0)

#: Breaker for yfinance calls.  More lenient — yfinance rate-limits often
#: self-heal after a short pause.
YFINANCE_BREAKER = CircuitBreaker("yfinance", failure_threshold=8, reset_timeout=30.0)

#: Breaker for Telegram alerts.  Low threshold — alerts are best-effort.
TELEGRAM_BREAKER = CircuitBreaker("telegram", failure_threshold=3, reset_timeout=120.0)
