import argparse
import logging
import time
from datetime import datetime
from core.config import CONFIG, MarketRegimeType
from core.data_provider import fetch_daily_batch
from core.indicators import add_indicators
from core.regime import RegimeTracker, classify_regime, compute_breadth # Added breadh import
from core.scorer import score_ticker
from core.portfolio import optimize_portfolio
from utils.messaging import send_telegram

# Initialize logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sovereign")

VERSION = "13.1-Modular"

# Note: In a full refactor, compute_breadth would also be moved to core.regime or similar.
# I will add it here for now or ensure it's in core.regime as planned.

def run_scan(config, debug=False):
    tracker = RegimeTracker() # Simplified for refactor
    log.info(f"🦅 SOVEREIGN ENGINE v{VERSION} | Launching Scan")
    
    # 1. Fetch data
    tickers = ["SBIN.NS", "RELIANCE.NS"] # Shortened for test
    raw_data = fetch_daily_batch(tickers, config)
    
    # 2. Add indicators
    processed = {t: add_indicators(df, config) for t, df in raw_data.items()}
    
    # 3. Classify Regime
    # breadth = compute_breadth(processed) # Import from core.regime
    # regime = classify_regime(processed, breadth, tracker, config)
    
    log.info("Scan completed (Logic simplified for refactor demonstration)")
    return [], [], None

def main():
    parser = argparse.ArgumentParser(description=f"Sovereign Engine v{VERSION}")
    parser.add_argument("--watch", type=int, default=None)
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()

    if args.version:
        print(f"Sovereign Engine v{VERSION}")
        return

    if args.watch:
        print(f"👁 WATCH MODE — every {args.watch}min.")
        while True:
            run_scan(CONFIG)
            time.sleep(args.watch * 60)
    else:
        run_scan(CONFIG)

if __name__ == "__main__":
    main()
