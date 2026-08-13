import yfinance as yf

tickers = ["RELIANCE.NS", "INFY.NS", "TCS.NS"]
print(f"Fetching {tickers} with group_by='ticker'...")
data = yf.download(
    tickers,
    period="1y",
    interval="1d",
    group_by="ticker",
    auto_adjust=True,
    progress=False,
    threads=True,
)
print(f"Result columns: {data.columns}")
print(f"Result shape: {data.shape}")
if not data.empty:
    for t in tickers:
        if t in data.columns.levels[0]:
            print(f"{t} tail:\n{data[t].tail(2)}")
        else:
            print(f"{t} NOT FOUND in columns")
