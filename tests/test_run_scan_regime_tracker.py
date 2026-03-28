from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import screener_v14_modular as svm
import core.services as services
from core.regime import (
    MarketRegimeType,
    RegimeTracker,
    confidence_position_scale,
)


class _FakePersistence:
    def create_scan_state(self, config):
        return services.ScanState(platt_a=config.PLATT_A, platt_b=config.PLATT_B)


def _install_scan_service(monkeypatch, *, gate=None, scaler=None):
    persistence = _FakePersistence()
    scan_service = services.ScanService(
        version=svm.VERSION,
        persistence=persistence,
        probability_gate=gate,
        capital_scaler=scaler,
    )
    monkeypatch.setattr(
        svm,
        "DEFAULT_SERVICES",
        svm.ServiceBundle(
            persistence=persistence,
            scan_service=scan_service,
            alert_service=scan_service.create_alert_service(),
        ),
    )
    return scan_service


def _make_regime_df(
    *,
    close_mult: float,
    adx: float = 30.0,
    atr: float = 2.0,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=80, freq="B")
    base = pd.Series(np.linspace(100.0, 130.0, len(index)), index=index)
    ema_50 = base.ewm(span=50, adjust=False).mean()
    close = ema_50 * close_mult

    return pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(index), 1_000_000.0) + rng.integers(0, 10_000, len(index)),
            "EMA_50": ema_50,
            "ADX": np.full(len(index), adx),
            "ATR": np.full(len(index), atr),
            "ATR_50_mean": np.full(len(index), atr),
        },
        index=index,
    )


def test_run_scan_preserves_regime_confirmation_across_calls(monkeypatch) -> None:
    config = replace(
        svm.CONFIG,
        REGIME_CONFIRM_BARS=2,
        REGIME_LOCK_MINUTES=0,
        MAX_WORKERS=1,
    )

    raw_data = {
        "RELIANCE.NS": _make_regime_df(close_mult=1.05, seed=1),
        config.BENCHMARK: _make_regime_df(close_mult=1.01, seed=2),
    }

    scan_service = _install_scan_service(monkeypatch)

    monkeypatch.setattr(scan_service._data_service, "_fetcher", lambda tickers, cfg: raw_data)
    monkeypatch.setattr(services, "add_indicators", lambda df, cfg: df)
    monkeypatch.setattr(services, "passes_static_filters", lambda df, cfg: True)
    monkeypatch.setattr(services, "score_ticker", lambda **kwargs: None)
    monkeypatch.setattr(services, "optimize_portfolio", lambda results, cfg, corr: [])

    tracker = RegimeTracker()

    _, _, first_regime = svm.run_scan(config=config, regime_tracker=tracker)
    _, _, second_regime = svm.run_scan(config=config, regime_tracker=tracker)

    assert first_regime is not None
    assert second_regime is not None
    assert first_regime.regime == MarketRegimeType.TREND_UP
    assert second_regime.regime == MarketRegimeType.TREND_UP
    assert first_regime.confirmed is False
    assert second_regime.confirmed is True


def test_run_scan_applies_regime_gate_and_confidence_sizing(monkeypatch) -> None:
    config = replace(
        svm.CONFIG,
        REGIME_CONFIRM_BARS=1,
        REGIME_LOCK_MINUTES=0,
        MAX_WORKERS=1,
    )

    raw_data = {
        "RELIANCE.NS": _make_regime_df(close_mult=1.05, seed=3),
        config.BENCHMARK: _make_regime_df(close_mult=1.01, seed=4),
    }
    captured: dict = {}

    class _FakeGate:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def threshold(self, regime: object) -> float:
            self.calls.append(regime)
            return 0.61

    class _FakeScaler:
        def capital_fraction(self, current_nav: float, regime: object) -> float:
            return 0.50

    def _fake_score_ticker(**kwargs):
        captured.update(kwargs)
        return None

    gate = _FakeGate()
    scaler = _FakeScaler()
    scan_service = _install_scan_service(monkeypatch, gate=gate, scaler=scaler)

    monkeypatch.setattr(scan_service._data_service, "_fetcher", lambda tickers, cfg: raw_data)
    monkeypatch.setattr(services, "add_indicators", lambda df, cfg: df)
    monkeypatch.setattr(services, "passes_static_filters", lambda df, cfg: True)
    monkeypatch.setattr(services, "score_ticker", _fake_score_ticker)
    monkeypatch.setattr(services, "optimize_portfolio", lambda results, cfg, corr: [])

    _, _, regime = svm.run_scan(
        config=config,
        regime_tracker=RegimeTracker(),
    )

    assert regime is not None
    assert captured["config"].MIN_PROB_WIN == pytest.approx(0.61)
    assert captured["capital_fraction"] == pytest.approx(
        0.50 * confidence_position_scale(regime.confidence)
    )
    assert gate.calls == [regime.regime]
