# Architecture fixes — 2026-10-02

Implemented against the current CodeGraph source. Validation: **735 tests passed**, **93.23% core coverage**, all configured Ruff/Mypy CI checks passed.

- **Server lifecycle:** `create_app(paths=..., persistence=..., auto_scan=...)` owns its engine/lock, database service, broadcaster, cancellation event, scan task, runtime and rate limiter. Importing `server` or calling the factory does not open SQLite. Startup validates the API key before opening persistence. Requests and background scans resolve collaborators from their owning app. Startup-incomplete scan requests return 503. Shutdown cancels/waits for the current scan before closing its runtime, including scans triggered after startup.
- **Killswitch failures:** restoration keeps the original database/I/O exception as the cause and blocks startup/scans. Logging cannot reference an unassigned snapshot path. Activation sets memory/cancellation before persistence; reset persists successfully before clearing either. A failed write remains visible as an error; it cannot establish durable persistence across a restart.
- **Snapshot recovery:** expected file/JSON/value errors are logged; malformed candidates cannot partially replace existing state. Killswitch verification is independent of optional scan-snapshot restoration.
- **Configuration:** typed slices share one uppercase mapping convention. Uppercase constructors, `dataclasses.replace`, flat serialization and research fingerprints remain compatible. HH:MM parsing is cached by string, so reconstructed settings reuse parsed values.
- **Boundaries:** Platt calibration moved into `core/calibration.py`. Regime and EMA eligibility became separate scorer gates; existing RANGE/countertrend/held-position rules and calibration selection remain unchanged.
- **Exception handling:** optional cache reads catch expected filesystem/pickle compatibility failures, including freshness-check failures. Resource exhaustion propagates. Transaction rollback/re-raise and scan-worker error boundaries remain intentional.
- **Compatibility/hygiene:** retain `utils.retry` exports; correct its outdated description. Root text/log outputs are already ignored; no historical evidence was deleted.

## Instance creation

```python
from pathlib import Path
from core.runtime_paths import RuntimePaths
from server import create_app

staging = create_app(paths=RuntimePaths.discover(root=Path("D:/engine-staging")), auto_scan=False)
production = create_app(paths=RuntimePaths.discover(root=Path("D:/engine-production")))
```

Supply distinct runtime paths; explicit environment path overrides take precedence. API tests/embedders must enter the lifespan, e.g. `with TestClient(application) as client:`. Replace old `server.STATE`/`server.PERSISTENCE` references with `application.state.engine`/`application.state.persistence` after startup. The standard `server:app` entry point remains available; `uvicorn server:create_app --factory` is also supported.

Immutable module configuration, environment-based API credentials, market-data caches and the Fyers client still belong to the process. Different broker accounts/configurations require separate processes; this factory isolates application engine/persistence resources.

Further decomposition of the scorer, scan orchestration and replay book is incremental maintenance work. The uploaded plan's blanket config rewrite, retry removal, deletion of evidence and indiscriminate exception narrowing were not applied. These changes do not establish positive trading expectancy.
