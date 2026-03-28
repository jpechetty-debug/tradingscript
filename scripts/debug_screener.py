import pandas as pd
import yfinance as yf
import screener

def debug_data():
    ticker = "SBIN.NS"
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    if df.empty:
        print("Data is empty!")
        return

    print(f"Columns: {df.columns}")
    df.columns = [c.title() if isinstance(c, str) else c for c in (df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns)]
    print(f"Cleaned columns: {df.columns}")

    # Try indicator calculation
    from screener import _true_range, _wilder
    tr = _true_range(df)
    atr = _wilder(tr, 14)
    print(f"Latest ATR: {atr.iloc[-1]}")

    # Try scoring
    bench = yf.download("^NSEI", period="1y", interval="1d", progress=False)
    bench.columns = [c.title() if isinstance(c, str) else c for c in (bench.columns.get_level_values(0) if isinstance(bench.columns, pd.MultiIndex) else bench.columns)]

    # Mock some inputs
    sector_ranks = {"BANKING": 1}
    sector_rs = {"BANKING": 1.5}

    # Calculate indicators using screener's function
    df = screener.add_indicators(df)

    # TR and ATR for ST
    tr = screener._true_range(df)
    atr = screener._wilder(tr, 14)
    df["TR"] = tr
    df["ATR"] = atr
    df["ATR_50_mean"] = atr.rolling(50).mean()

    # Indicators are already on df from add_indicators()

    print("Scoring...")
    regime = screener.MarketRegime("TREND_UP", 0.6, 25, 1.0, 0.8, True)
    res = screener.score_ticker(ticker, df, bench["Close"], sector_ranks, sector_rs, {}, {}, "CLOSING_TREND", regime, debug=True)

    if res:
        print(f"Success! Score: {res.composite}, Prob: {res.prob_win}")
    else:
        print("Scoring returned None")

if __name__ == "__main__":
    debug_data()
