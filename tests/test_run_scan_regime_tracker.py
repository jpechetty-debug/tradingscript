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


def _install_scan_service(monkeypatch, *, gate=None, scaler=None, alert_service=None):
    persistence = _FakePersistence()
    scan_service = services.ScanService(
        version=svm.VERSION,
        persistence=persistence,
        probability_gate=gate,
        capital_scaler=scaler,
    )
    resolved_alert = alert_service or scan_service.create_alert_service()
    monkeypatch.setattr(
        svm,
        "_DEFAULT_SERVICES",
        svm.ServiceBundle(
            persistence=persistence,
            scan_service=scan_service,
            alert_service=resolved_alert,
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


def test_run_scan_regime_override(monkeypatch) -> None:
    config = replace(svm.CONFIG, MAX_WORKERS=1)
    scan_service = _install_scan_service(monkeypatch)
    
    # Mock data
    raw_data = {
        "RELIANCE.NS": _make_regime_df(close_mult=1.05),
        config.BENCHMARK: _make_regime_df(close_mult=1.01),
    }
    monkeypatch.setattr(scan_service._data_service, "_fetcher", lambda t, c: raw_data)
    monkeypatch.setattr(services, "add_indicators", lambda d, c: d)
    monkeypatch.setattr(services, "passes_static_filters", lambda d, c: True)
    monkeypatch.setattr(services, "score_ticker", lambda **k: None)
    monkeypatch.setattr(services, "optimize_portfolio", lambda r, c, m: [])

    # Override to PANIC
    _, _, regime = svm.run_scan(config=config, regime_override="PANIC")
    assert regime.regime == MarketRegimeType.PANIC
    assert "PANIC" in regime.label


def test_run_scan_respects_no_ema_filter(monkeypatch) -> None:
    config = replace(svm.CONFIG, MAX_WORKERS=1, USE_EMA200_FILTER=True)
    scan_service = _install_scan_service(monkeypatch)
    captured = {}
    
    def _fake_score_ticker(**kwargs):
        captured["config"] = kwargs["config"]
        return None

    monkeypatch.setattr(scan_service._data_service, "_fetcher", lambda t, c: {
        "T1": _make_regime_df(close_mult=1.0), 
        config.BENCHMARK: _make_regime_df(close_mult=1.0)
    })
    monkeypatch.setattr(services, "add_indicators", lambda d, c: d)
    monkeypatch.setattr(services, "passes_static_filters", lambda d, c: True)
    monkeypatch.setattr(services, "score_ticker", _fake_score_ticker)
    monkeypatch.setattr(services, "optimize_portfolio", lambda r, c, m: [])

    svm.run_scan(config=config, no_ema_filter=True, regime_override="TREND_UP")
    assert captured["config"].USE_EMA200_FILTER is False


def test_run_scan_respects_force_score(monkeypatch) -> None:
    config = replace(svm.CONFIG, MAX_WORKERS=1)
    scan_service = _install_scan_service(monkeypatch)
    captured = {}
    
    def _fake_score_ticker(**kwargs):
        captured["force_score"] = kwargs.get("force_score")
        return None

    monkeypatch.setattr(scan_service._data_service, "_fetcher", lambda t, c: {
        "T1": _make_regime_df(close_mult=1.0), 
        config.BENCHMARK: _make_regime_df(close_mult=1.0)
    })
    monkeypatch.setattr(services, "add_indicators", lambda d, c: d)
    monkeypatch.setattr(services, "passes_static_filters", lambda d, c: True)
    monkeypatch.setattr(services, "score_ticker", _fake_score_ticker)
    monkeypatch.setattr(services, "optimize_portfolio", lambda r, c, m: [])

    svm.run_scan(config=config, force_score=True, regime_override="TREND_UP")
    assert captured["force_score"] is True


def test_run_scan_session_lock(monkeypatch) -> None:
    config = replace(svm.CONFIG, MAX_WORKERS=1)
    # Mock the class method in the module where it is used (services.py)
    monkeypatch.setattr(services.SystemConfig, "is_regime_locked", lambda self, now=None: True)
    
    scan_service = _install_scan_service(monkeypatch)
    tracker = RegimeTracker()
    tracker.push(MarketRegimeType.TREND_UP, 0.7)
    
    # Data that would normally trigger RANGE or DOWN
    raw_data = {
        "T1": _make_regime_df(close_mult=0.9), 
        config.BENCHMARK: _make_regime_df(close_mult=1.0)
    }
    monkeypatch.setattr(scan_service._data_service, "_fetcher", lambda t, c: raw_data)
    monkeypatch.setattr(services, "add_indicators", lambda d, c: d)
    monkeypatch.setattr(services, "passes_static_filters", lambda d, c: True)
    monkeypatch.setattr(services, "score_ticker", lambda **k: None)
    monkeypatch.setattr(services, "optimize_portfolio", lambda r, c, m: [])

    _, _, regime = svm.run_scan(config=config, regime_tracker=tracker)
    # Should still be TREND_UP because it's locked
    assert regime.regime == MarketRegimeType.TREND_UP
    assert regime.regime_locked is True


def test_run_scan_sector_rs_recording(monkeypatch) -> None:
    config = replace(svm.CONFIG, MAX_WORKERS=1)
    scan_service = _install_scan_service(monkeypatch)
    
    raw_data = {
        "IT_T": _make_regime_df(close_mult=1.1), 
        config.BENCHMARK: _make_regime_df(close_mult=1.0)
    }
    monkeypatch.setattr(scan_service._data_service, "_fetcher", lambda t, c: raw_data)
    monkeypatch.setattr(services, "add_indicators", lambda d, c: d)
    monkeypatch.setattr(services, "compute_sector_rs", lambda p, b, c: {"IT": 0.08, "FIN": -0.02})
    monkeypatch.setattr(services, "passes_static_filters", lambda d, c: True)
    monkeypatch.setattr(services, "score_ticker", lambda **k: None)
    monkeypatch.setattr(services, "optimize_portfolio", lambda r, c, m: [])

    _, _, regime = svm.run_scan(config=config)
    assert regime.sector_concentration >= 0.0


def test_run_scan_alert_integration(monkeypatch) -> None:
    config = replace(svm.CONFIG, MAX_WORKERS=1, TELEGRAM_BOT_TOKEN="123", TELEGRAM_CHAT_ID="456")
    class _FakeAlerter:
        def __init__(self): self.called = False
        def send_portfolio_summary(self, portfolio, regime, config):
            self.called = True
            self.regime_label = regime.label

    alerter = _FakeAlerter()
    scan_service = _install_scan_service(monkeypatch, alert_service=alerter)
    
    raw_data = {
        "T1": _make_regime_df(close_mult=1.05), 
        config.BENCHMARK: _make_regime_df(close_mult=1.01)
    }
    monkeypatch.setattr(scan_service._data_service, "_fetcher", lambda t, c: raw_data)
    monkeypatch.setattr(services, "add_indicators", lambda d, c: d)
    monkeypatch.setattr(services, "passes_static_filters", lambda d, c: True)
    
    # Create a valid TickerResult-like object or mock it carefully
    from core.factors import FactorScores
    dummy_factors = FactorScores(
        trend=0.8, momentum=0.7, volume=0.6, volatility=0.5, 
        rs=0.8, breakout=0.4, quality=0.3, composite=0.7, ic_weights={}
    )
    
    mock_result = services.TickerResult(
        ticker="T1", sector="IT", direction="LONG",
        close=100.0, change_pct=1.5,
        factors=dummy_factors,
        composite=0.7, prob_win=0.8, expectancy_r=1.2, sharpe_rank=0.5,
        entry=100.0, stop=90.0, t1=110.0, t2=120.0, breakeven=105.0,
        trail_stop=88.0, time_stop_bars=5,
        shares=10, risk_inr=1000.0, rr_t1=2.0, kelly_f=0.05,
        kurt_correction=1.0, excess_kurtosis=0.0,
        rsi=60.0, stochrsi_k=50.0, rvol=1.5, adx=30.0,
        super_up=True, macd_hist=0.5, atr_pctile=50.0,
        vol_contract=False, rs_vs_nifty=0.05, near_52w=True,
        ema200_aligned=True, mtf_aligned=True, consec_days=3,
        poc=100.0, val=95.0, vah=105.0,
        regime="TREND_UP", session="OPENING_RANGE"
    )

    monkeypatch.setattr(services, "score_ticker", lambda **k: mock_result)
    monkeypatch.setattr(services, "optimize_portfolio", lambda r, c, m: [mock_result])

    from core.regime import MarketRegime
    regime_obj = MarketRegime(
        regime=MarketRegimeType.TREND_UP,
        breadth=30.0,
        adx_median=25.0,
        atr_ratio=1.1,
        confidence=0.8,
        confirmed=True,
    )
    svm._send_alert([mock_result], regime_obj, config)
    
    assert alerter.called is True
    assert "UP" in alerter.regime_label


