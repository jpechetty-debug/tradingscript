import requests
import logging

log = logging.getLogger("sovereign.utils")

def send_telegram(message, token, chat_id):
    if not token or not chat_id: return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"}, timeout=10)
    except Exception as e:
        log.error(f"Telegram error: {e}")

def mask_token(token: str) -> str:
    if not token or len(token) < 12: return "****"
    return f"{token[:6]}...{token[-6:]}"
