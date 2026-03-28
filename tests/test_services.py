from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd

from core.config import CONFIG, MarketRegimeType
from core.regime import MarketRegime
from core.runtime_paths import RuntimePaths
from core.services import AlertService, MarketDataService, PersistenceService, ScanService


def _paths(tmp_path) -> RuntimePaths:
    return RuntimePaths.discover(root=tmp_path)


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
