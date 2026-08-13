import yfinance as yf
import time
from core.universe import ALL_TICKERS

def download_chunk(tickers):
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(
        tickers,
        period="1y",
        interval="1d",
        group_by="ticker",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return data

chunk_size = 40
all_data: dict = {}

for i in range(0, len(ALL_TICKERS), chunk_size):
    chunk = ALL_TICKERS[i : i + chunk_size]
    start = time.time()
    try:
        data = download_chunk(chunk)
        elapsed = time.time() - start
        print(f"Chunk {i//chunk_size + 1} done in {elapsed:.2f}s. Shape: {data.shape}")
    except Exception as e:
        print(f"Chunk {i//chunk_size + 1} FAILED: {e}")

print("All chunks processed.")
