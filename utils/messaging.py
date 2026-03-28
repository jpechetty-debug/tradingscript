"""
utils/messaging.py
==================
Telegram alert delivery with:
  - 4,096-character truncation (Telegram hard limit)
  - One automatic retry on HTTP 429 (Too Many Requests), honouring the
    Retry-After header when present
  - token masking helper for safe logging
"""

import logging
import time

import requests  # type: ignore[import-untyped]

log = logging.getLogger("sovereign.utils")

_TG_MAX_LEN = 4_096
_TG_TRUNCATION_NOTICE = "\n… [truncated]"


def send_telegram(message: str, token: str, chat_id: str) -> bool:
    """
    Send *message* via the Telegram Bot API.

    Returns True on success, False on permanent failure (after retries).

    Telegram constraints enforced here
    ------------------------------------
    * **4,096-char limit** — messages longer than this are silently rejected
      by the API (HTTP 400).  We truncate to 4,096 chars and append a
      ``… [truncated]`` notice so the recipient knows the message was cut.
    * **30 msg/sec rate limit** — a single 429 response triggers one retry
      after honouring the ``Retry-After`` header (or a 2-second default).
      Callers that need higher throughput should implement a token-bucket
      queue above this function.
    """
    if not token or not chat_id:
        return False

    # ── Truncate to Telegram's hard limit ────────────────────────────────────
    if len(message) > _TG_MAX_LEN:
        keep = _TG_MAX_LEN - len(_TG_TRUNCATION_NOTICE)
        message = message[:keep] + _TG_TRUNCATION_NOTICE
        log.warning("Telegram message truncated to %d chars.", _TG_MAX_LEN)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}

    for attempt in range(2):          # attempt 0 = first try, attempt 1 = retry
        try:
            res = requests.post(url, json=payload, timeout=10)

            if res.ok:
                return True

            if res.status_code == 429 and attempt == 0:
                # Honour Retry-After if the server provides it; fall back to 2 s.
                retry_after = 2.0
                ra_header = res.headers.get("Retry-After")
                if ra_header:
                    try:
                        retry_after = float(ra_header)
                    except ValueError:
                        pass
                else:
                    try:
                        retry_after = float(
                            (res.json().get("parameters") or {}).get("retry_after", 2)
                        )
                    except (ValueError, AttributeError):
                        pass

                log.warning(
                    "Telegram 429 – rate-limited. Retrying in %.1f s.", retry_after
                )
                time.sleep(retry_after)
                continue            # go to attempt 1

            log.error(
                "Telegram API error (attempt %d): %d – %s",
                attempt + 1, res.status_code, res.text[:200],
            )
            return False

        except requests.exceptions.RequestException as exc:
            log.error("Telegram network error (attempt %d): %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(2)
                continue
            return False

    return False


def mask_token(token: str) -> str:
    """Return a partially-masked token safe for log output."""
    if not token or len(token) < 12:
        return "****"
    return f"{token[:6]}...{token[-6:]}"
