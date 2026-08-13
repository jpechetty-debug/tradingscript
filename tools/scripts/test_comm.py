
import requests
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_comm")

def test_raw_get():
    # Use a standard browser User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    url = "https://query1.finance.yahoo.com/v7/finance/download/RELIANCE.NS?period1=1700000000&period2=1710000000&interval=1d&events=history"
    logger.info(f"Testing raw GET to {url}")
    try:
        r = requests.get(url, headers=headers, timeout=10)
        logger.info(f"Response status: {r.status_code}")
        if r.status_code == 200:
            logger.info("Content received!")
            print(r.text[:200])
        else:
            logger.warning(f"Response failed: {r.text[:200]}")
    except Exception as e:
        logger.error(f"Raw GET failed with exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_raw_get()
