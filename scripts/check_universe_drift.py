import csv
import logging
import requests
import sys
from pathlib import Path

# Add project root to sys.path so we can import from core
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.universe import ALL_TICKERS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("check_universe_drift")

NIFTY_500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"


def check_universe_drift():
    """
    Downloads the current Nifty 500 constituents and compares them against
    the curated ALL_TICKERS list in core/universe.py.
    """
    log.info(f"Downloading latest Nifty 500 list from {NIFTY_500_URL}...")
    try:
        response = requests.get(
            NIFTY_500_URL,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=10
        )
        response.raise_for_status()
    except Exception as e:
        log.error(f"Failed to fetch Nifty 500 list: {e}")
        sys.exit(1)

    lines = response.text.strip().split("\n")
    reader = csv.DictReader(lines)

    nifty_500_tickers = set()
    for row in reader:
        symbol = row.get("Symbol")
        if symbol:
            nifty_500_tickers.add(f"{symbol}.NS")

    if not nifty_500_tickers:
        log.error("Failed to parse any symbols from the Nifty 500 list.")
        sys.exit(1)

    our_tickers = set(ALL_TICKERS)

    # Check for tickers in our universe that are no longer in Nifty 500
    missing_from_index = our_tickers - nifty_500_tickers

    log.info("=== Universe Drift Report ===")
    log.info(f"Our Universe Size: {len(our_tickers)}")
    log.info(f"Nifty 500 Size: {len(nifty_500_tickers)}")
    log.info("-" * 30)

    if missing_from_index:
        log.warning(f"Found {len(missing_from_index)} tickers in our universe that are NOT in Nifty 500:")
        for t in sorted(missing_from_index):
            log.warning(f"  - {t}")
    else:
        log.info("All our tickers are present in the Nifty 500 index. No removals needed.")

    # In a full implementation, we might save the state to track exact additions/removals
    # to the index over time. For now, we report the set differences.
    log.info("-" * 30)
    log.info("Run complete.")


if __name__ == "__main__":
    check_universe_drift()
