
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_yf_single")

def test_fetch_single() -> None:
    ticker_symbol = "RELIANCE.NS"
    logger.info(f"Testing yfinance Ticker().history() for {ticker_symbol}")
    try:
        tk = yf.Ticker(ticker_symbol)
        df = tk.history(period="1mo", interval="1d")
        if df.empty:
            logger.warning(f"Data for {ticker_symbol} is empty!")
        else:
            logger.info(f"Data for {ticker_symbol} fetched successfully! Shape: {df.shape}")
            print(df.head())
    except Exception as e:
        logger.error(f"Fetch failed with exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_fetch_single()
