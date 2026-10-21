"""Execution-level regressions for explicit worker inputs and runtime ownership."""
from copy import deepcopy
from dataclasses import replace
import json
import pickle
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

import screener_v14_modular as svm
from core import services, worker
from core.application import build_application_runtime
from core.config import CONFIG, MarketRegimeType, SystemConfig
from core.regime import MarketRegime, RegimeTracker
from core.runtime_paths import RuntimePaths
from core.services import MarketDataService, PersistenceService, ScanCancelled, ScanOutput, ScanService, ServiceBundle
from core.worker import LocalThreadDispatcher, ProcessPoolDispatcher, ScanJob, ScanOptions, execute_scan_task


@pytest.fixture
def paths(tmp_path, monkeypatch):
    for key in ("STATE_DIR", "ARTIFACTS_DIR", "LOG_DIR", "PORTFOLIO_STATE_PATH", "PLATT_CALIB_PATH",
                "TRADE_LOG_PATH", "WEIGHTS_PATH", "DEGRADATION_LOG_PATH", "TELEMETRY_LOG_PATH", "STATE_DB_PATH"):
        monkeypatch.delenv(key, raising=False)
    return RuntimePaths.discover(root=tmp_path)


def bundle_for(paths):
    persistence = PersistenceService(paths)
    scan = ScanService(version="test", persistence=persistence)
    return ServiceBundle(persistence=persistence, scan_service=scan, alert_service=scan.create_alert_service())


def test_execution_preserves_explicit_flags_config_paths_metadata_and_tracker(paths, tmp_path, monkeypatch):
    bundle = bundle_for(paths)
    tracker = RegimeTracker()
    tracker.push(MarketRegimeType.TREND_UP, .7)
    options = ScanOptions(debug=True, no_intraday=True, regime_override="TREND_UP", no_ema_filter=True, force_score=True)
    metadata = {"request_id": "r-42", "provenance": {"source": "paper"}, "no_intraday": False}
    config = replace(CONFIG, CAPITAL_INR=123456, INTRADAY_ENABLED=False)
    job = ScanJob(tickers=[" tcs.ns ", "TCS.NS", "^NSEI"], config=config, paths=paths, options=options, metadata=metadata)
    # Mutating caller-owned inputs after job creation cannot alter its execution.
    config.CAPITAL_INR = 1
    options.debug = False
    metadata["provenance"]["source"] = "changed"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    row = SimpleNamespace(ticker="TCS", direction="SHORT", entry=100., stop=101., t1=98.,
                          trade_horizon="SWING", factors=SimpleNamespace(trend=.7),
                          strategy_id="SWING_PULLBACK_V1", probability_status="HEURISTIC_UNVALIDATED")
    regime = MarketRegime(MarketRegimeType.TREND_UP, .7, 30., 1., .9, True)
    run = MagicMock(return_value=ScanOutput([row], [row], regime, {"IT": .1}))
    monkeypatch.setattr(svm, "run_scan", run)
    monkeypatch.setattr(svm, "_DEFAULT_SERVICES", MagicMock())
    def probe():
        return False
    result = execute_scan_task(job, services=bundle, regime_tracker=tracker, cancel_requested=probe)
    assert result.status == "completed"
    kwargs = run.call_args.kwargs
    assert kwargs["tickers"] == ["TCS.NS"]
    assert kwargs["services"] is bundle
    assert kwargs["regime_tracker"] is tracker
    assert kwargs["cancel_requested"] is probe
    assert kwargs["config"].CAPITAL_INR == 123456
    assert all(kwargs[key] is True for key in ("debug", "no_intraday", "no_ema_filter", "force_score"))
    assert kwargs["regime_override"] == "TREND_UP"
    assert result.metadata == {"request_id": "r-42", "provenance": {"source": "paper"}, "no_intraday": False}
    assert result.context["paths"]["state_db_file"] == str(paths.state_db_file)
    assert result.context["options"]["no_intraday"] is True
    assert result.context["config_reference"].startswith("sha256:")
    assert result.context["engine_version"] == svm.VERSION
    assert result.context["snapshot_schema_version"] == 1
    assert result.portfolio[0]["action"] == "SELL"
    assert result.portfolio[0]["horizon_label"] == "SWING (CNC)"
    assert result.portfolio[0]["stop_pct"] == 1.
    assert result.portfolio[0]["factors"] == {"trend": .7}
    assert result.portfolio[0]["probability_status"] == "HEURISTIC_UNVALIDATED"
    result.metadata["provenance"]["source"] = "result mutation"
    assert job.metadata["provenance"]["source"] == "paper"


def test_real_worker_scan_scores_only_requested_symbols_and_scopes_owned_runtime(paths, monkeypatch):
    """Only the provider is synthetic: real indicators, regime, scoring and optimizer run."""
    bars = 260
    dates = pd.bdate_range("2024-01-01", periods=bars)
    close = np.linspace(100., 150., bars) + np.sin(np.arange(bars) / 3.)
    frame = pd.DataFrame({"Open": close-.2, "High": close+1., "Low": close-1.,
                          "Close": close, "Volume": np.full(bars, 2_000_000.)}, index=dates)
    raw = {symbol: frame.copy() for symbol in ("TCS.NS", "INFY.NS", "OUTSIDE.NS", CONFIG.BENCHMARK)}
    # Outside bars must be dropped before indicator work as well as before scoring.
    raw["OUTSIDE.NS"] = pd.DataFrame({"not": ["OHLCV"]})
    fetched, evaluated, built = [], [], []

    def fetcher(tickers, config):
        fetched.append(list(tickers))
        return raw

    original_pass1 = services.score_candidate_pass1

    def observed_pass1(**kwargs):
        evaluated.append(kwargs["ticker"])
        return original_pass1(**kwargs)

    def factory(**kwargs):
        def configure(**service_kwargs):
            return services.configure_services(data_service=MarketDataService(fetcher=fetcher), **service_kwargs)
        runtime = build_application_runtime(service_factory=configure, **kwargs)
        built.append(runtime)
        return runtime

    monkeypatch.setattr(worker, "build_application_runtime", factory)
    monkeypatch.setattr(services, "score_candidate_pass1", observed_pass1)
    default_services = MagicMock()
    monkeypatch.setattr(svm, "_DEFAULT_SERVICES", default_services)
    config = replace(CONFIG, MAX_WORKERS=1, REGIME_LOCK_MINUTES=0, USE_EMA200_FILTER=False,
                     ADV_SHARE_FLOOR=0, ADV_TURNOVER_FLOOR=0, ICIR_MIN_OBS=999,
                     SWING_SETUP_ENABLED=False, LEADER_PULLBACK_ENABLED=False)
    tracker = RegimeTracker()
    tracker.push(MarketRegimeType.TREND_UP, 1.)
    tracker.push(MarketRegimeType.TREND_UP, 1.)
    job = ScanJob(tickers=["TCS.NS", "INFY.NS"], config=config, paths=paths,
                  options=ScanOptions(force_score=True, no_intraday=True, regime_override="TREND_UP"),
                  regime_tracker=tracker, metadata={"source": "offline contract test"})
    result = execute_scan_task(job)
    assert result.status == "completed", result.error
    assert fetched == [["TCS.NS", "INFY.NS"]]
    assert sorted(evaluated) == ["INFY.NS", "TCS.NS"]
    assert all(row["ticker"] in {"TCS", "INFY"} for row in result.candidates + result.portfolio)
    assert result.metadata == job.metadata
    assert result.context["tickers"] == ["TCS.NS", "INFY.NS"]
    assert result.regime_tracker.is_confirmed(MarketRegimeType.TREND_UP, 2)
    assert built[0].services.persistence.paths == paths
    calibrator = built[0].components.calibrator
    assert calibrator._wpath == paths.factor_weights_file
    assert calibrator._tpath == paths.trade_log_file
    assert not calibrator._thread.is_alive()
    default_services.scan_service.scan.assert_not_called()


@pytest.mark.parametrize("tickers", [[], [""], ["  "], [None], "TCS.NS", ["TCS NS"], [CONFIG.BENCHMARK]])
def test_invalid_or_empty_requests_never_expand_to_default_universe(paths, tickers):
    with pytest.raises(ValueError):
        ScanJob(tickers=tickers, paths=paths)
    bundle = bundle_for(paths)
    fetcher = MagicMock()
    bundle.scan_service._data_service = MarketDataService(fetcher=fetcher)
    with pytest.raises(ValueError):
        svm.run_scan(tickers=tickers, services=bundle)
    fetcher.assert_not_called()


def test_default_universe_is_snapshotted_at_job_creation(paths, monkeypatch):
    from core import universe
    monkeypatch.setattr(universe, "ALL_TICKERS", ["TCS.NS"])
    job = ScanJob(paths=paths)
    monkeypatch.setattr(universe, "ALL_TICKERS", ["OTHER.NS"])
    assert job.tickers == ["TCS.NS"]


@pytest.mark.parametrize("status", ["success", "failure", "cancel"])
def test_owned_runtime_is_closed_on_every_execution_outcome(paths, monkeypatch, status):
    runtime = MagicMock()
    monkeypatch.setattr(worker, "build_application_runtime", lambda **kwargs: runtime)
    effect = RuntimeError("offline") if status == "failure" else ScanCancelled("stop") if status == "cancel" else None
    monkeypatch.setattr(svm, "run_scan", MagicMock(side_effect=effect, return_value=ScanOutput([], [], None)))
    result = execute_scan_task(ScanJob(tickers=["TCS.NS"], paths=paths, metadata={"trace": "123"}))
    assert result.status == {"success": "completed", "failure": "failed", "cancel": "cancelled"}[status]
    assert result.metadata == {"trace": "123"}
    assert result.context["tickers"] == ["TCS.NS"]
    runtime.close.assert_called_once()


def test_preexecution_cancellation_and_runtime_path_mismatch_do_no_work(paths, tmp_path, monkeypatch):
    builder, run = MagicMock(), MagicMock()
    monkeypatch.setattr(worker, "build_application_runtime", builder)
    monkeypatch.setattr(svm, "run_scan", run)
    job = ScanJob(tickers=["TCS.NS"], paths=paths, metadata={"request": 1})
    result = execute_scan_task(job, cancel_requested=lambda: True)
    assert result.status == "cancelled"
    assert result.metadata == job.metadata
    foreign = bundle_for(RuntimePaths.discover(root=tmp_path / "foreign"))
    result = execute_scan_task(job, services=foreign)
    assert result.status == "failed"
    assert "paths do not match" in result.error
    builder.assert_not_called()
    run.assert_not_called()


def test_empty_call_results_never_reuse_previous_service_regime_or_sector_data(paths, monkeypatch):
    bundle = bundle_for(paths)
    bundle.scan_service.last_sector_rs = {"STALE": 99.}
    bundle.scan_service.last_regime_info = MarketRegime(MarketRegimeType.TREND_UP, .7, 30., 1., .9, True)
    monkeypatch.setattr(svm, "run_scan", MagicMock(side_effect=[ScanOutput([], [], None, {"IT": .2}), ScanOutput([], [], None)]))
    job = ScanJob(tickers=["TCS.NS"], paths=paths)
    assert execute_scan_task(job, services=bundle).sector_rs == {"IT": .2}
    result = execute_scan_task(job, services=bundle)
    assert result.sector_rs == {}
    assert result.regime_info is None


def test_job_and_result_roundtrip_with_configuration_and_tracker(paths, monkeypatch):
    config = SystemConfig(settings_payload=CONFIG.as_app_settings(), CAPITAL_INR=555000)
    tracker = RegimeTracker()
    tracker.push(MarketRegimeType.RANGE, .5)
    job = ScanJob(tickers=["TCS.NS"], config=config, paths=paths, regime_tracker=tracker,
                  metadata={"nested": {"values": [1, "two", None]}})
    restored = pickle.loads(pickle.dumps(job))
    assert restored.paths == job.paths
    assert restored.config.CAPITAL_INR == job.config.CAPITAL_INR
    assert restored.regime_tracker.last_regime() == MarketRegimeType.RANGE
    monkeypatch.setattr(svm, "run_scan", lambda **kwargs: ScanOutput([], [], None))
    result = execute_scan_task(restored, services=bundle_for(paths))
    assert result.status == "completed", result.error
    restored_result = pickle.loads(pickle.dumps(result))
    assert restored_result.metadata == job.metadata
    assert restored_result.context == result.context
    assert restored_result.regime_tracker.last_regime() == MarketRegimeType.RANGE
    json.dumps(result.context, allow_nan=False)


def test_configuration_reference_tracks_strategy_settings_and_excludes_credentials(paths, monkeypatch):
    bundle = bundle_for(paths)
    monkeypatch.setattr(svm, "run_scan", lambda **kwargs: ScanOutput([], [], None))
    base = ScanJob(tickers=["TCS.NS"], paths=paths)
    first = execute_scan_task(base, services=bundle)
    changed = deepcopy(base)
    changed.config.CAPITAL_INR += 100
    second = execute_scan_task(changed, services=bundle)
    assert first.context["config_reference"] != second.context["config_reference"]
    changed = deepcopy(base)
    changed.config.FYERS_ACCESS_TOKEN = "private-token"
    changed.config.TELEGRAM_BOT_TOKEN = "private-bot"
    third = execute_scan_task(changed, services=bundle)
    assert first.context["config_reference"] == third.context["config_reference"]
    assert "private-" not in json.dumps(third.context)


def test_thread_dispatch_snapshots_queued_job_and_serializes_borrowed_runtime(paths, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    seen = []
    tracker = RegimeTracker()
    bundle = bundle_for(paths)

    def scan(**kwargs):
        seen.append((list(kwargs["tickers"]), kwargs["config"].CAPITAL_INR))
        kwargs["regime_tracker"].push(MarketRegimeType.TREND_UP, .7)
        if len(seen) == 1:
            entered.set()
            assert release.wait(5)
        return ScanOutput([], [], None)

    monkeypatch.setattr(svm, "run_scan", scan)
    dispatcher = LocalThreadDispatcher(max_workers=2, services=bundle, regime_tracker=tracker)
    try:
        first = dispatcher.submit(ScanJob(tickers=["TCS.NS"], paths=paths))
        assert entered.wait(5)
        queued = ScanJob(tickers=["INFY.NS"], paths=paths, metadata={"nested": [1]})
        second = dispatcher.submit(queued)
        queued.tickers.append("UNREQUESTED.NS")
        queued.config.CAPITAL_INR = 1
        queued.metadata["nested"].append(2)
        assert len(seen) == 1
        release.set()
        first_result, second_result = first.result(timeout=10), second.result(timeout=10)
        assert first_result.status == second_result.status == "completed"
        assert seen == [(["TCS.NS"], CONFIG.CAPITAL_INR), (["INFY.NS"], CONFIG.CAPITAL_INR)]
        assert second_result.metadata == {"nested": [1]}
        assert second_result.regime_tracker.is_confirmed(MarketRegimeType.TREND_UP, 2)
        assert tracker.is_confirmed(MarketRegimeType.TREND_UP, 2)
    finally:
        release.set()
        dispatcher.shutdown()


@pytest.mark.parametrize("option", [ScanOptions(debug="yes"), ScanOptions(regime_override="INVALID")])
def test_invalid_execution_options_fail_before_runtime_creation(paths, monkeypatch, option):
    builder = MagicMock()
    monkeypatch.setattr(worker, "build_application_runtime", builder)
    result = execute_scan_task(ScanJob(tickers=["TCS.NS"], paths=paths, options=option, metadata={"trace": 1}))
    assert result.status == "failed"
    assert result.metadata == {"trace": 1}
    builder.assert_not_called()


@pytest.mark.parametrize("metadata", [{"nan": float("nan")}, {"object": object()}, ["not an object"],
                                      {1: "key"}, {"nested": {1: "key"}}, {"tuple": (1, 2)}])
def test_nonserializable_metadata_is_rejected(paths, metadata):
    with pytest.raises((ValueError, TypeError)):
        ScanJob(tickers=["TCS.NS"], paths=paths, metadata=metadata)


def test_cyclic_metadata_is_rejected_before_traversal(paths):
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="Circular reference"):
        ScanJob(tickers=["TCS.NS"], paths=paths, metadata=cyclic)


def test_process_dispatcher_roundtrips_job_and_failed_payload_without_fetching(paths):
    dispatcher = ProcessPoolDispatcher(max_workers=1)
    try:
        job = ScanJob(tickers=["TCS.NS"], paths=paths, schema_version=2, metadata={"trace": ["process"]})
        result = dispatcher.submit(job).result(timeout=30)
        assert result.job_id == job.job_id
        assert result.status == "failed"
        assert result.metadata == job.metadata
        assert "Unsupported ScanJob schema version" in result.error
    finally:
        dispatcher.shutdown()


def test_application_factory_stops_components_if_service_construction_fails(paths):
    components = MagicMock()
    persistence = PersistenceService(paths)
    factory = MagicMock(return_value=components)
    with pytest.raises(RuntimeError, match="construction failed"):
        build_application_runtime(version="test", persistence=persistence, component_factory=factory,
                                  service_factory=MagicMock(side_effect=RuntimeError("construction failed")))
    assert factory.call_args.kwargs["paths"] == paths
    components.stop.assert_called_once()


def test_standalone_jobs_read_their_own_nav_calibration_and_factor_weights(paths, tmp_path, monkeypatch):
    other = RuntimePaths.discover(root=tmp_path / "other")
    contexts = []
    for root, nav, platt, trend in [(paths, 600000., -2., .6), (other, 800000., -3., .3)]:
        persistence = PersistenceService(root)
        persistence.save_portfolio_state(nav, peak_nav=900000.)
        persistence.save_platt(platt, 1.)
        root.factor_weights_file.write_text(json.dumps({"trend": trend, "momentum": 1.-trend}), encoding="utf-8")

    def inspect_scan(**kwargs):
        bundle = kwargs["services"]
        scan = bundle.scan_service
        contexts.append((bundle.persistence.paths, scan._resolve_current_nav(),
                         bundle.persistence.load_platt(kwargs["config"])[0],
                         scan._factor_calibrator.current_weights()["trend"]))
        return ScanOutput([], [], None)

    monkeypatch.setattr(svm, "run_scan", inspect_scan)
    first = execute_scan_task(ScanJob(tickers=["TCS.NS"], paths=paths))
    second = execute_scan_task(ScanJob(tickers=["INFY.NS"], paths=other))
    assert first.status == second.status == "completed"
    assert contexts == [(paths, 600000., -2., .6), (other, 800000., -3., .3)]


def test_cancellation_received_during_scan_is_not_reported_as_completion(paths, monkeypatch):
    cancelled = threading.Event()
    runtime = MagicMock()
    monkeypatch.setattr(worker, "build_application_runtime", lambda **kwargs: runtime)

    def scan(**kwargs):
        cancelled.set()
        return ScanOutput([], [], None)

    monkeypatch.setattr(svm, "run_scan", scan)
    result = execute_scan_task(ScanJob(tickers=["TCS.NS"], paths=paths), cancel_requested=cancelled.is_set)
    assert result.status == "cancelled"
    assert result.candidates == result.portfolio == []
    runtime.close.assert_called_once()
