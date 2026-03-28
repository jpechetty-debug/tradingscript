from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

import core.backtest as bt
from core.regime import MarketRegime, MarketRegimeType


def _make_ohlcv(n: int = 250, base: float = 100.0, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = base * np.cumprod(1 + rng.normal(0.001, 0.012, n))
    high = close * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, n)))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    volume = rng.integers(1_000_000, 3_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def test_walk_forward_reuses_regime_tracker_and_passes_sector_rs(monkeypatch) -> None:
    config = replace(
        bt.SystemConfig(),
        BENCHMARK="BENCH",
        BACKTEST_MIN_PROB=0.0,
        MIN_PROB_WIN=0.0,
        MIN_EXPECTANCY_R=-99.0,
        REGIME_CONFIRM_BARS=2,
    )
    raw = {
        "AAA.NS": _make_ohlcv(seed=1),
        "BBB.NS": _make_ohlcv(seed=2),
        "BENCH": _make_ohlcv(seed=99),
    }
    tracker_ids: list[int] = []
    sector_rs_payloads: list[dict[str, float] | None] = []
    fake_sector_rs = {"IT": 1.2, "BANKING": 0.8}

    def _fake_compute_sector_rs(processed, bench_series, cfg):
        return fake_sector_rs

    def _fake_classify_regime(processed, breadth, tracker, cfg, *, locked=False, sector_rs=None):
        tracker_ids.append(id(tracker))
        sector_rs_payloads.append(sector_rs)
        return MarketRegime(
            regime=MarketRegimeType.TREND_UP,
            breadth=breadth,
            adx_median=30.0,
            atr_ratio=1.1,
            confidence=0.8,
            confirmed=True,
            sector_concentration=1.0,
            regime_locked=False,
        )

    monkeypatch.setattr(bt, "compute_sector_rs", _fake_compute_sector_rs)
    monkeypatch.setattr(bt, "classify_regime", _fake_classify_regime)

    result = bt.walk_forward(
        raw_data=raw,
        config=config,
        train_days=80,
        test_days=10,
        step_days=20,
        max_trades_per_fold=2,
    )

    assert result.overall.n_folds > 1
    assert len(set(tracker_ids)) == 1
    assert tracker_ids
    assert sector_rs_payloads
    assert all(payload == fake_sector_rs for payload in sector_rs_payloads)
