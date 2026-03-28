from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

import core.services as services
from core.config import CONFIG, MarketRegimeType
from core.regime import MarketRegime, RegimeTracker
from core.runtime_paths import RuntimePaths
from core.services import AlertService, MarketDataService, PersistenceService, ScanService


def _paths(tmp_path) -> RuntimePaths:
    return RuntimePaths.discover(root=tmp_path)


def _scan_df() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=5, freq="B")
    close = pd.Series([100.0, 101.0, 102.5, 103.0, 104.0], index=index)
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": [1_000_000] * len(index),
        },
        index=index,
    )


class _FakePersistence:
    def __init__(self, *, platt_a: float = -1.5, platt_b: float = 0.25) -> None:
        self._platt_a = platt_a
        self._platt_b = platt_b

    def create_scan_state(self, config):
        _ = config
        return services.ScanState(platt_a=self._platt_a, platt_b=self._platt_b)


def test_persistence_service_prefers_state_trade_log_over_legacy(tmp_path):
    paths = _paths(tmp_path)
    legacy = tmp_path / "trade_log.json"
    legacy.write_text('[{"source": "legacy"}]', encoding="utf-8")

    state_file = paths.state_dir / "trade_log.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text('[{"source": "state"}]', encoding="utf-8")

    service = PersistenceService(paths)

    assert service.load_trade_log() == [{"source": "state"}]


def test_persistence_service_scopes_relative_artifacts(tmp_path):
    service = PersistenceService(_paths(tmp_path))

    target = service.artifact_path("backtest_results.csv")

    assert target == tmp_path / "artifacts" / "backtest_results.csv"
    assert target.parent.exists()


def test_persistence_service_reads_portfolio_state_from_state_dir(tmp_path):
    paths = _paths(tmp_path)
    portfolio_state = paths.portfolio_state_file
    portfolio_state.parent.mkdir(parents=True, exist_ok=True)
    portfolio_state.write_text(
        '{"current_nav": 875000, "peak_nav": 1000000}',
        encoding="utf-8",
    )

    service = PersistenceService(paths)
    snapshot = service.load_portfolio_state()

    assert snapshot is not None
    assert snapshot.current_nav == pytest.approx(875000)
    assert snapshot.peak_nav == pytest.approx(1000000)


def test_alert_service_filters_by_threshold_and_uses_messenger():
    sent: list[tuple[str, str, str]] = []

    def messenger(message: str, token: str, chat_id: str) -> bool:
        sent.append((message, token, chat_id))
        return True

    service = AlertService(version="test", messenger=messenger)
    config = replace(
        CONFIG,
        TELEGRAM_BOT_TOKEN="token",
        TELEGRAM_CHAT_ID="chat",
        TELEGRAM_ALERT_MIN_PROB=0.6,
        TELEGRAM_ALERT_TOP_N=5,
    )
    regime = MarketRegime(
        regime=MarketRegimeType.TREND_UP,
        breadth=0.7,
        adx_median=25.0,
        atr_ratio=1.0,
        confidence=0.8,
        confirmed=True,
    )
    portfolio = [
        SimpleNamespace(
            ticker="AAA",
            direction="LONG",
            prob_win=0.72,
            expectancy_r=1.1,
            entry=100.0,
            stop=96.0,
            t1=108.0,
            shares=10,
        ),
        SimpleNamespace(
            ticker="BBB",
            direction="LONG",
            prob_win=0.55,
            expectancy_r=0.8,
            entry=50.0,
            stop=48.0,
            t1=54.0,
            shares=20,
        ),
    ]

    service.send_portfolio_summary(portfolio, regime, config)

    assert len(sent) == 1
    message, token, chat_id = sent[0]
    assert "AAA" in message
    assert "BBB" not in message
    assert token == "token"
    assert chat_id == "chat"


def test_alert_service_uses_injected_alerter():
    captured: dict[str, object] = {}

    class FakeAlerter:
        def send_daily_summary(
            self,
            regime: str,
            top_picks: list[dict[str, object]],
            current_nav: float | None = None,
        ) -> None:
            captured["regime"] = regime
            captured["tickers"] = [pick["ticker"] for pick in top_picks]
            captured["current_nav"] = current_nav

    service = AlertService(version="test", alerter=FakeAlerter())
    config = replace(
        CONFIG,
        TELEGRAM_BOT_TOKEN="token",
        TELEGRAM_CHAT_ID="chat",
        TELEGRAM_ALERT_MIN_PROB=0.6,
    )
    regime = MarketRegime(
        regime=MarketRegimeType.TREND_UP,
        breadth=0.7,
        adx_median=25.0,
        atr_ratio=1.0,
        confidence=0.8,
        confirmed=True,
    )
    portfolio = [
        SimpleNamespace(
            ticker="AAA",
            direction="LONG",
            prob_win=0.72,
            expectancy_r=1.1,
            entry=100.0,
            stop=96.0,
            t1=108.0,
            shares=10,
        )
    ]

    service.send_portfolio_summary(portfolio, regime, config)

    assert captured == {"regime": "TREND_UP", "tickers": ["AAA"], "current_nav": None}


def test_scan_service_scan_uses_injected_fetcher_and_dependencies(monkeypatch):
    raw_data = {
        "AAA.NS": _scan_df(),
        CONFIG.BENCHMARK: _scan_df(),
    }
    fetch_calls: list[tuple[list[str], object]] = []
    score_calls: list[dict[str, object]] = []

    def fetcher(tickers, config):
        fetch_calls.append((list(tickers), config))
        return raw_data

    class FakeGate:
        def __init__(self) -> None:
            self.calls: list[object] = []

        def threshold(self, regime: object) -> float:
            self.calls.append(regime)
            return 0.61

    class FakeScaler:
        def __init__(self) -> None:
            self.calls: list[tuple[float, object]] = []

        def capital_fraction(self, current_nav: float, regime: object) -> float:
            self.calls.append((current_nav, regime))
            return 0.50

    class FakeCalibrator:
        def __init__(self) -> None:
            self.calls = 0

        def current_weights(self) -> dict[str, float]:
            self.calls += 1
            return {"trend": 0.7, "momentum": 0.3}

    regime = MarketRegime(
        regime=MarketRegimeType.TREND_UP,
        breadth=0.68,
        adx_median=29.0,
        atr_ratio=1.1,
        confidence=0.8,
        confirmed=True,
    )
    result = SimpleNamespace(
        ticker="AAA.NS",
        direction="LONG",
        prob_win=0.72,
        expectancy_r=1.2,
        rr_t1=2.4,
        shares=10,
        risk_inr=1_000.0,
    )

    monkeypatch.setattr(services, "add_indicators", lambda df, cfg: df)
    monkeypatch.setattr(services, "compute_sector_rs", lambda processed, bench, cfg: {"LEAD": 1.0})
    monkeypatch.setattr(services, "compute_breadth", lambda processed, cfg: 0.68)
    monkeypatch.setattr(services, "classify_regime", lambda *args, **kwargs: regime)
    monkeypatch.setattr(services, "confidence_position_scale", lambda confidence: 0.8)
    monkeypatch.setattr(services, "passes_static_filters", lambda df, cfg: True)
    monkeypatch.setattr(services, "optimize_portfolio", lambda results, cfg, corr: list(results))

    def fake_score_ticker(**kwargs):
        score_calls.append(kwargs)
        return result

    monkeypatch.setattr(services, "score_ticker", fake_score_ticker)

    config = replace(CONFIG, MAX_WORKERS=1, REGIME_LOCK_MINUTES=0, ICIR_MIN_OBS=99)
    gate = FakeGate()
    scaler = FakeScaler()
    calibrator = FakeCalibrator()
    service = ScanService(
        version="test",
        data_service=MarketDataService(fetcher=fetcher),
        persistence=_FakePersistence(),
        probability_gate=gate,
        capital_scaler=scaler,
        factor_calibrator=calibrator,
        current_nav_provider=lambda: 875_000.0,
    )

    all_results, portfolio, observed_regime = service.scan(config=config, regime_tracker=RegimeTracker())

    assert fetch_calls
    assert score_calls
    assert all_results == [result]
    assert portfolio == [result]
    assert observed_regime == regime
    assert gate.calls == [regime.regime]
    assert scaler.calls == [(875_000.0, regime.regime)]
    assert calibrator.calls == 1
    assert score_calls[0]["config"].MIN_PROB_WIN == 0.61
    assert score_calls[0]["config"].PLATT_A == -1.5
    assert score_calls[0]["config"].PLATT_B == 0.25
    assert score_calls[0]["factor_weights"] == {"trend": 0.7, "momentum": 0.3}
    assert score_calls[0]["capital_fraction"] == pytest.approx(0.4)


def test_scan_service_scan_returns_early_in_panic_regime(monkeypatch):
    raw_data = {
        "AAA.NS": _scan_df(),
        CONFIG.BENCHMARK: _scan_df(),
    }
    scored: list[str] = []
    panic = MarketRegime(
        regime=MarketRegimeType.PANIC,
        breadth=0.12,
        adx_median=35.0,
        atr_ratio=1.4,
        confidence=0.9,
        confirmed=True,
    )

    monkeypatch.setattr(services, "add_indicators", lambda df, cfg: df)
    monkeypatch.setattr(services, "compute_sector_rs", lambda processed, bench, cfg: {"LEAD": 1.0})
    monkeypatch.setattr(services, "compute_breadth", lambda processed, cfg: 0.12)
    monkeypatch.setattr(services, "classify_regime", lambda *args, **kwargs: panic)
    monkeypatch.setattr(services, "score_ticker", lambda **kwargs: scored.append("called"))

    service = ScanService(
        version="test",
        data_service=MarketDataService(fetcher=lambda tickers, cfg: raw_data),
        persistence=_FakePersistence(),
    )
    config = replace(CONFIG, MAX_WORKERS=1, REGIME_LOCK_MINUTES=0)

    all_results, portfolio, observed_regime = service.scan(config=config, regime_tracker=RegimeTracker())

    assert all_results == []
    assert portfolio == []
    assert observed_regime == panic
    assert scored == []


def test_scan_service_backtest_writes_relative_output_under_artifacts(tmp_path, monkeypatch):
    class FakeResults:
        def __init__(self) -> None:
            self.trades = [object()]
            self.overall = SimpleNamespace()
            self.saved_path: str | None = None

        def to_csv(self, path: str) -> None:
            self.saved_path = path

    bench_df = pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
    data_service = MarketDataService(fetcher=lambda tickers, config: {config.BENCHMARK: bench_df})
    persistence = PersistenceService(_paths(tmp_path))
    service = ScanService(version="test", data_service=data_service, persistence=persistence)
    fake_results = FakeResults()

    monkeypatch.setattr("core.services.walk_forward", lambda **kwargs: fake_results)

    result = service.run_backtest(out_csv="walk_forward.csv")

    assert result is fake_results
    assert fake_results.saved_path == str(tmp_path / "artifacts" / "walk_forward.csv")
