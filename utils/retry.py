"""
utils/retry.py
==============
Exponential backoff with full jitter for network operations.

Referenced in the README but absent from the repository.  The primary
consumer is ``core/data_provider.fetch_daily_batch`` which fires up to
``config.MAX_WORKERS`` (default 20) concurrent yfinance requests and can
trigger HTTP 429 (Too Many Requests) throttling under load.

Usage
-----
Decorate any function that may raise a retryable exception::

    from utils.retry import retry_with_backoff

    @retry_with_backoff(retries=4, base_delay=1.0, max_delay=30.0)
    def _download_chunk(symbols):
        return yf.download(symbols, ...)

Or call it imperatively::

    result = retry_with_backoff(retries=3)(my_fn)(arg1, arg2)

Design
------
* **Full jitter** (sleep = random(0, cap)) per the AWS "Exponential
  Backoff And Jitter" post.  Full jitter outperforms equal-jitter and
  decorrelated jitter at high concurrency because it spreads retries
  uniformly in time, reducing thundering-herd after a 429 burst.
* A **Retry-After header** value (seconds) surfaced by some API wrappers
  is respected when the exception carries it as ``exc.retry_after``.
* The ``retryable`` predicate is a caller-supplied hook so that permanent
  errors (e.g. bad ticker symbol → 404) are not retried needlessly.
"""

from __future__ import annotations

import functools
import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

log = logging.getLogger("sovereign.retry")

F = TypeVar("F", bound=Callable[..., Any])

# Exceptions that yfinance and requests commonly raise on throttle / transient
# network failures.  ImportError-safe: the set is populated lazily.
_RETRYABLE_TYPES: tuple[type[BaseException], ...] = (
    OSError,          # socket-level timeouts
    TimeoutError,
    ConnectionError,
)

try:
    import requests
    _RETRYABLE_TYPES += (requests.exceptions.RequestException,)
except ImportError:
    pass


def _default_retryable(exc: BaseException) -> bool:
    """Return True for exceptions that are worth retrying."""
    if isinstance(exc, _RETRYABLE_TYPES):
        return True
    # yfinance raises generic Exception for HTTP 429 / 5xx; check the message.
    msg = str(exc).lower()
    return any(kw in msg for kw in ("429", "too many requests", "rate limit",
                                    "timed out", "connection", "503", "502"))


def retry_with_backoff(
    retries: int = 4,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    multiplier: float = 2.0,
    retryable: Callable[[BaseException], bool] = _default_retryable,
) -> Callable[[F], F]:
    """
    Decorator factory: wraps *fn* with exponential-backoff retry logic.

    Parameters
    ----------
    retries:
        Maximum number of **retry** attempts (not counting the first call).
        Total calls = retries + 1.
    base_delay:
        Initial ceiling for the random sleep (seconds).  On attempt ``k``
        the ceiling is ``min(base_delay * multiplier ** k, max_delay)``.
    max_delay:
        Hard upper bound on sleep time (seconds).
    multiplier:
        Exponential growth factor per attempt (default 2 → 1 s, 2 s, 4 s …).
    retryable:
        Predicate that receives the exception and returns True if the call
        should be retried.  Defaults to matching common transient network
        errors and HTTP 429 / 5xx strings.

    Returns
    -------
    A decorator that wraps a callable with the retry logic.
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: BaseException | None = None
            for attempt in range(retries + 1):
                try:
                    return fn(*args, **kwargs)
                except BaseException as exc:
                    if attempt == retries or not retryable(exc):
                        raise

                    last_exc = exc

                    # Respect an explicit Retry-After if the exception carries one.
                    retry_after: float | None = getattr(exc, "retry_after", None)
                    if retry_after and isinstance(retry_after, (int, float)):
                        sleep_s = float(retry_after)
                    else:
                        cap = min(base_delay * (multiplier ** attempt), max_delay)
                        sleep_s = random.uniform(0, cap)   # full jitter

                    log.warning(
                        "%s attempt %d/%d failed (%s); retrying in %.2fs.",
                        fn.__qualname__,
                        attempt + 1,
                        retries + 1,
                        type(exc).__name__,
                        sleep_s,
                    )
                    time.sleep(sleep_s)

            # Should never reach here, but satisfy type checkers.
            raise RuntimeError("Retry loop exited unexpectedly") from last_exc  # noqa: EM101

        return wrapper  # type: ignore[return-value]
    return decorator

from core.retry import (  # noqa: E402, F401 — re-export for unified import surface
    CircuitBreaker,
    CircuitBreakerOpen,
    guarded_call,
    FYERS_BREAKER,
    YFINANCE_BREAKER,
    TELEGRAM_BREAKER,
)
