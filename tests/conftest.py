import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def setup_test_environment():
    """Ensure standard test environment variables are populated."""
    os.environ.setdefault("API_KEY", "test-api-key")
    os.environ.setdefault("SECRET_KEY", "test-secret-key-32-chars-long--")


@pytest.fixture(autouse=True)
def isolated_server_runtime(request, tmp_path, monkeypatch):
    """API tests enter startup/shutdown and never mutate the project's real trade journal."""
    from pathlib import Path
    api_tests = {"test_server", "test_server_killswitch_persistence", "test_performance_and_sse",
                 "test_trade_lifecycle", "test_paper_ledger"}
    if Path(str(request.fspath)).stem not in api_tests:
        yield
        return
    from fastapi.testclient import TestClient
    import server
    from core.runtime_paths import RuntimePaths
    from core.services import PersistenceService
    monkeypatch.setenv("API_KEY", server.API_KEY or "test-api-key")
    monkeypatch.setattr(server.app.state, "engine", server.EngineState())
    monkeypatch.setattr(server.app.state, "broadcaster", server.SSEBroadcaster())
    monkeypatch.setattr(server.app.state, "persistence", PersistenceService(RuntimePaths.discover(root=tmp_path)))
    monkeypatch.setattr(server.app.state, "auto_scan", False)
    with TestClient(server.app):
        yield
