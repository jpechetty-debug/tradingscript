import yfinance as yf

def check_multiindex():
    tickers = ["SBIN.NS", "RELIANCE.NS"]
    df = yf.download(tickers, period="5d", interval="1d", group_by="ticker", progress=False)
    print(f"Columns: {df.columns}")
    print(f"Level 0: {df.columns.get_level_values(0).unique()}")
    print(f"Level 1: {df.columns.get_level_values(1).unique()}")

if __name__ == "__main__":
    check_multiindex()
