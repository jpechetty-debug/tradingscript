import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import pytz

IST = pytz.timezone("Asia/Kolkata")

def check_dates():
    ticker = "^NSEI"
    df = yf.download(ticker, period="1y", interval="1d", progress=False)
    index = df.index
    print(f"Index type: {type(index)}")
    print(f"First element: {index[0]}, type: {type(index[0])}")
    
    t_date = datetime.now(IST).date()
    c_start = t_date - timedelta(days=90)
    
    print(f"Today (IST): {t_date}")
    print(f"Cutoff (IST): {c_start}")
    
    count = 0
    for x in index:
        # Convert to date
        if hasattr(x, 'date'):
            d = x.date()
        else:
            d = pd.to_datetime(x).date()
        
        if c_start <= d < t_date:
            count += 1
            if count <= 2:
                print(f"Match found: {d}")
    
    print(f"Total matches: {count}")

if __name__ == "__main__":
    check_dates()
