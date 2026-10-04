"""
utils/retry.py
==============
Backward-compatible exports of the network retry implementation in ``core.retry``.
Internal consumers use ``core.retry`` directly; these names remain available
for existing external callers importing ``utils.retry``.
"""

from __future__ import annotations

import warnings

from core.retry import (
    CircuitBreaker,
    CircuitBreakerOpen,
    guarded_call,
    retry_with_backoff,
    FYERS_BREAKER,
    YFINANCE_BREAKER,
    TELEGRAM_BREAKER,
)

__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "guarded_call",
    "retry_with_backoff",
    "FYERS_BREAKER",
    "YFINANCE_BREAKER",
    "TELEGRAM_BREAKER",
]

warnings.warn(
    "utils.retry is deprecated and will be removed in a future version. "
    "Please import from core.retry directly.",
    DeprecationWarning,
    stacklevel=2,
)
