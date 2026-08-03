import yfinance as yf

ticker = "RELIANCE.NS"
print(f"Fetching {ticker}...")
df = yf.download(ticker, period="1mo", interval="1d")
print(f"Result shape: {df.shape}")
print(df.tail())
