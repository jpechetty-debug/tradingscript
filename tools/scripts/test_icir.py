import pandas as pd
from _bootstrap import ensure_repo_root

ensure_repo_root()

from core.factors import calibrate_ic_weights  # noqa: E402
from core.config import CONFIG  # noqa: E402
from core.data_provider import fetch_daily_batch  # noqa: E402
from core.indicators import add_indicators  # noqa: E402
from core.universe import ALL_TICKERS  # noqa: E402

class MockResult:
    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

# 1. Fetch some data
tickers = ALL_TICKERS[:15] # 15 tickers
raw = fetch_daily_batch(tickers, CONFIG)
if not raw:
    print("No data fetched")
    exit(1)

processed = {t: add_indicators(df, CONFIG) for t, df in raw.items() if t != CONFIG.BENCHMARK}
bench = raw.get(CONFIG.BENCHMARK, pd.DataFrame()).get("Close", pd.Series())

# 2. Mock results
results = [MockResult(t) for t in processed.keys()]

# 3. Calibrate
print(f"Starting calibration on {len(results)} tickers...")
new_weights = calibrate_ic_weights(
    results=results,
    processed=processed,
    bench=bench,
    sector_ranks={},
    n_sectors=12,
    config=CONFIG,
    lookback=30, # smaller for speed
    calib_offset=10
)

print("Optimized Weights:")
for k, v in new_weights.items():
    print(f"  {k}: {v:.4f}")

assert abs(sum(new_weights.values()) - 1.0) < 1e-6
print("Verification SUCCESS")
