"""
utils/retry.py
==============
Exponential backoff with full jitter for network operations.

Referenced in the README but absent from the repository.  The primary
consumer is ``core/data_provider.fetch_daily_batch`` which fires up to
``config.MAX_WORKERS`` (default 20) concurrent yfinance requests and can
trigger HTTP 429 (Too Many Requests) throttling under load.

NOTE: The implementation has moved to `core.retry`. This file is preserved
as a re-export shim to avoid breaking any external imports.
"""

from __future__ import annotations

from core.retry import (  # noqa: E402, F401 — re-export for unified import surface
    CircuitBreaker,
    CircuitBreakerOpen,
    guarded_call,
    retry_with_backoff,
    FYERS_BREAKER,
    YFINANCE_BREAKER,
    TELEGRAM_BREAKER,
)
