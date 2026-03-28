
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_yf")

def test_fetch():
    tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    logger.info(f"Testing yfinance fetch for {tickers}")
    try:
        data = yf.download(tickers, period="1mo", interval="1d", progress=False)
        if data.empty:
            logger.warning("Data is empty!")
        else:
            logger.info(f"Data fetched successfully. Shape: {data.shape}")
            print(data.head())
    except Exception as e:
        logger.error(f"Fetch failed with exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_fetch()
