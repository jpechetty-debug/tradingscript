# Worker job contract — 21 October 2026

Priority 5 is implemented. A job's candidate universe now reaches `run_scan()` and `ScanService.scan()`. Provider rows outside that universe are removed before indicator preparation, regime calculations, scoring, portfolio construction and paper setup evaluation. The configured benchmark remains available as context and is not scored as a candidate.

`None` selects the default universe. `ScanJob` captures that universe at construction so a queued job cannot silently expand when the worker's defaults change. Explicit symbols are trimmed, uppercased and deduplicated; ticker suffixes are not invented. Empty, malformed and benchmark-only lists are rejected before fetching. Calls to the existing `run_scan()` wrapper without a ticker argument retain their default behavior.

## Job inputs and results

`ScanJob` schema version 1 carries:

- `tickers`: the captured candidate universe.
- `config`: a detached copy of `SystemConfig`.
- `paths`: absolute `RuntimePaths`, resolved when the job is constructed.
- `options`: `ScanOptions(debug, no_intraday, regime_override, no_ema_filter, force_score)`.
- `regime_tracker`: a detached confirmation-history snapshot for standalone execution.
- `metadata`: caller provenance, such as request/source IDs; a finite JSON object with string keys. It is preserved in successful, failed and cancelled results. Execution flags belong in the typed fields above; metadata cannot override them.
- `job_id`, `created_at` and `schema_version`.

Inputs are copied at construction and again at dispatcher submission. Changing a caller's configuration, nested metadata or a queued job cannot alter the submitted work. Unsupported schema versions and invalid execution options fail before runtime construction.

Results preserve job identity and metadata, distinguish `completed`, `failed` and `cancelled`, and include an execution context containing the resolved universe, paths, options, engine version, snapshot schema version and a SHA-256 reference to the requested noncredential configuration. The reference excludes the redundant settings payload and Telegram/Fyers fields. It identifies requested settings, not a frozen copy of market data or all mutable calibration state.

Ticker rows use `core.snapshots.build_scan_snapshot()` / `format_ticker()`, preserving strategy/probability status, explicit horizon and factor data while adding the same action and price-distance labels as the other application surfaces. Sector and regime data come from the current call; empty results never inherit old values from a service singleton.

## Runtime ownership

Standalone jobs construct an application runtime using the job's persistence paths, execute with its explicit services and configuration, and close the runtime on success, failure or cancellation. They never resolve `screener_v14_modular._DEFAULT_SERVICES`.

The application builder now passes its owning persistence paths to `create_runtime_components()`. The calibrator therefore loads/writes that runtime's factor-weight and trade-log files. Previously those component paths came from module defaults even when the application's database used a different root. If service construction fails after components start, the builder stops them before propagating the error.

Local callers may supply `services`, `regime_tracker` and a cooperative `cancel_requested` callback to `execute_scan_task()` or `LocalThreadDispatcher`. The supplied service paths must match the job. Borrowed services remain caller-owned; the worker does not close them. A local dispatcher serializes its scans to protect mutable service/tracker state, even when its executor has multiple worker threads. Independent dispatchers do not provide a shared lock.

`ProcessPoolDispatcher` passes a detached job to the standalone entry point. Jobs and results use Python's process-pool pickle transport, including configuration and confirmation history; this is not a new JSON queue protocol. The caller can reuse a result's `regime_tracker` when constructing the next job. A process job owns its newly constructed runtime; live services and callback closures are local-only inputs. Standard `Future.cancel()` applies before a queued job starts. The local cooperative callback is checked before work and during/after scanning. Process-pool callers on Windows must create the pool under their normal `if __name__ == "__main__"` guard.

## Example

```python
from core.config import CONFIG
from core.runtime_paths import RuntimePaths
from core.worker import LocalThreadDispatcher, ScanJob, ScanOptions

job = ScanJob(
    tickers=["TCS.NS", "INFY.NS"],
    config=CONFIG,
    paths=RuntimePaths.discover(),
    options=ScanOptions(no_intraday=True),
    metadata={"request_id": "research-42", "source": "paper-screen"},
)
dispatcher = LocalThreadDispatcher(max_workers=1)
try:
    result = dispatcher.submit(job).result()
finally:
    dispatcher.shutdown()
```

## Completion evidence

**97 focused tests passed** across worker contracts, legacy worker behavior, services, scan-wrapper/regime behavior, runtime components, CLI, trading correctness and architecture. Worker statement coverage was **98.06%**. The three uncovered statements are the abstract dispatcher methods and a defensive rejection of runtime paths mutated to relative paths after construction.

The new tests include:

- A real scan using synthetic provider data, real indicators/regime/scoring/optimization, and a provider that over-returns an unrelated symbol. Only `TCS.NS` and `INFY.NS` reach scoring; the benchmark remains contextual.
- Two standalone jobs reading different persisted NAV, Platt calibration and factor weights from separate roots.
- Configuration/options propagation, metadata preservation, a different worker working directory, canonical row serialization and current-call sector data.
- Rejection of empty/invalid universes, malformed/cyclic metadata, invalid options and unsupported job schemas.
- Runtime cleanup and cancellation before/during execution; queued-input mutation protection and continued local confirmation history.
- Pickle round trips and an actual Windows process-pool round trip of a rejected job and its failed payload. A successful scan inside a spawned process was not integration-tested; the real successful scan runs in the test process through the same standalone entry point.

The existing CI Ruff and strict Mypy commands pass with the pinned tools (Ruff 0.4.7; Mypy 1.10.0, **76 source files**). No scopes, suppressions or test exclusions were added. This focused run does not claim the full cross-platform CI matrix or API HTTP integration suite passed. No live market download was needed.

Local logs and machine-readable evidence are in `artifacts/worker_contract/2026-10-21/` (ignored by Git). The existing Graphify graph was queried for orientation; current CodeGraph source supplied implementation details. The older Graphify graph was not rebuilt as part of this change.
