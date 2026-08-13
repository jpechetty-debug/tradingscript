import yfinance as yf
import pandas as pd

def test_yf_layout():
    # Attempt a small download to inspect column structure
    tickers = ["AAPL", "MSFT"]
    data = yf.download(tickers, period="5d", interval="1d", group_by="ticker", progress=False)
    print("Columns Type:", type(data.columns))
    print("Columns MultiIndex Levels:", data.columns.nlevels if isinstance(data.columns, pd.MultiIndex) else "Flat")
    print("Columns Index:", data.columns)
    
    if isinstance(data.columns, pd.MultiIndex):
        level_0 = data.columns.get_level_values(0).unique()
        print("Level 0:", level_0)
        level_1 = data.columns.get_level_values(1).unique()
        print("Level 1:", level_1)

if __name__ == "__main__":
    test_yf_layout()
