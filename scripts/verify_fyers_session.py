#!/usr/bin/env python3
"""
scripts/verify_fyers_session.py
===============================
Pre-market validation script for Sovereign Engine.
Verifies that Fyers v3 API credentials and daily access tokens are active
before the Indian equity market opens (09:15 IST).

Exit Codes:
    0: Session verified and active.
    1: Token missing, invalid, or expired (Action required: run tools/scripts/fyers_setup.py).
    2: Network or unexpected broker API error.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = REPO_ROOT / ".env"

load_dotenv(dotenv_path=ENV_FILE)


def check_token_age(env_path: Path, max_age_hours: int = 18) -> tuple[bool, float]:
    """Check if .env was modified more than max_age_hours ago."""
    if not env_path.exists():
        return False, 0.0
    try:
        mtime = env_path.stat().st_mtime
        age_hours = (datetime.now(timezone.utc).timestamp() - mtime) / 3600.0
        return (age_hours <= max_age_hours), age_hours
    except Exception:
        return False, 0.0


def verify_fyers_session(json_output: bool = False) -> int:
    client_id = os.getenv("FYERS_CLIENT_ID")
    access_token = os.getenv("FYERS_ACCESS_TOKEN")

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "client_id_configured": bool(client_id),
        "token_configured": bool(access_token),
        "token_fresh": False,
        "token_age_hours": None,
        "api_connectivity": "NOT_CHECKED",
        "status": "FAILED",
        "action": None,
    }

    if not client_id or not access_token:
        results["action"] = "Set FYERS_CLIENT_ID and FYERS_ACCESS_TOKEN in .env"
        if json_output:
            print(json.dumps(results, indent=2))
        else:
            print("❌ FYERS SESSION ERROR: Missing client_id or access_token in .env")
            print("   Action required: Run `python tools/scripts/fyers_setup.py`")
        return 1

    is_fresh, age_hours = check_token_age(ENV_FILE)
    results["token_fresh"] = is_fresh
    results["token_age_hours"] = round(age_hours, 2)

    try:
        from fyers_apiv3 import fyersModel
    except ImportError:
        results["status"] = "FALLBACK_YFINANCE"
        results["action"] = "fyers_apiv3 not installed; engine will fall back to yfinance."
        if json_output:
            print(json.dumps(results, indent=2))
        else:
            print("⚠️  fyers_apiv3 library not installed.")
            print("   Engine will operate in yfinance fallback mode.")
        return 0

    try:
        fyers = fyersModel.FyersModel(
            client_id=client_id,
            token=access_token,
            is_async=False,
            log_path=str(REPO_ROOT / "logs"),
        )
        profile = fyers.get_profile()
        if profile and profile.get("s") == "ok":
            results["api_connectivity"] = "OK"
            results["status"] = "VERIFIED"
            results["profile_name"] = profile.get("data", {}).get("name")
            if json_output:
                print(json.dumps(results, indent=2))
            else:
                print("✅ FYERS SESSION VERIFIED: Active & Ready for Live Scanning.")
                print(f"   Account: {results.get('profile_name', 'Fyers Trader')}")
                print(f"   Token Age: {results['token_age_hours']} hours")
            return 0
        else:
            results["api_connectivity"] = "AUTH_FAILED"
            results["status"] = "EXPIRED"
            results["error_message"] = profile.get("message", "Authentication rejected")
            results["action"] = "Token has expired. Re-authenticate via tools/scripts/fyers_setup.py"
            if json_output:
                print(json.dumps(results, indent=2))
            else:
                print("❌ FYERS SESSION REJECTED: Token is expired or revoked.")
                print(f"   Broker response: {results['error_message']}")
                print("   Action required: Run `python tools/scripts/fyers_setup.py`")
            return 1
    except Exception as exc:
        results["api_connectivity"] = "ERROR"
        results["status"] = "ERROR"
        results["error_message"] = str(exc)
        if json_output:
            print(json.dumps(results, indent=2))
        else:
            print(f"❌ FYERS CONNECTIVITY ERROR: {exc}")
        return 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-market Fyers API session verifier.")
    parser.add_argument("--json", action="store_true", help="Output results in structured JSON format.")
    args = parser.parse_args()
    sys.exit(verify_fyers_session(json_output=args.json))


if __name__ == "__main__":
    main()
