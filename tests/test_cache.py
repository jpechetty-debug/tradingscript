import numpy as np
import pandas as pd
import pytest
from core.cache import ScanCache

def _make_df(n=40):
    np.random.seed(0)
    c = np.random.uniform(100, 200, n)
    return pd.DataFrame({
        "Open": c, "High": c * 1.01, "Low": c * 0.99,
        "Close": c, "Volume": np.ones(n) * 1e6,
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))

class TestScanCache:

    def test_volume_profile_cache_hit(self):
        cache = ScanCache()
        df = _make_df()
        r1 = cache.volume_profile(df, ticker="TEST", lookback=20, bins=50)
        r2 = cache.volume_profile(df, ticker="TEST", lookback=20, bins=50)
        assert r1 == r2
        assert cache.stats()["hits"] == 1
        assert cache.stats()["misses"] == 1

    def test_cache_miss_on_different_params(self):
        cache = ScanCache()
        df = _make_df()
        cache.volume_profile(df, ticker="TEST", lookback=20, bins=50)
        cache.volume_profile(df, ticker="TEST", lookback=30, bins=50)
        assert cache.stats()["misses"] == 2

    def test_clear_resets_state(self):
        cache = ScanCache()
        df = _make_df()
        cache.volume_profile(df, ticker="TEST")
        cache.clear()
        assert cache.stats()["vprofile_size"] == 0

    def test_eviction_at_capacity(self):
        cache = ScanCache(max_vprofile_entries=3)
        for i in range(5):
            df = _make_df(n=40 + i)  # slightly different df each time
            cache.volume_profile(df, ticker=f"T{i}")
        assert cache.stats()["vprofile_size"] <= 4  # evicted at least one quarter

    def test_hit_rate_reported(self):
        cache = ScanCache()
        df = _make_df()
        cache.volume_profile(df, ticker="X")
        cache.volume_profile(df, ticker="X")
        s = cache.stats()
        assert s["hit_rate"] == pytest.approx(0.5)
