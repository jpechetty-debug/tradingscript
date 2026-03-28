
import yfinance as yf
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_yf_session")

def test_fetch_with_session():
    # Use a standard browser User-Agent
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)

    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    logger.info(f"Testing yfinance fetch with session for {tickers}")

    try:
        # yfinance 0.2.x supports passing a session
        data = yf.download(tickers, period="1mo", interval="1d", progress=False, session=session)
        if data.empty:
            logger.warning("Data is still empty!")
        else:
            logger.info(f"Data fetched successfully! Shape: {data.shape}")
            print(data.head())
    except Exception as e:
        logger.error(f"Fetch failed with exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_fetch_with_session()
