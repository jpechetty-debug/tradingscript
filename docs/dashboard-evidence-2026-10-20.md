# Dashboard strategy and evidence status — 20 October 2026

Priority 4 is implemented. The portfolio and candidate tables and the factor modal now share the same evidence display:

- `HEURISTIC_UNVALIDATED` displays a decimal score, such as **0.63 — Heuristic · unvalidated**.
- Missing or unknown probability status displays **Unverified model output**. A global calibration flag cannot upgrade an individual row.
- Only explicit `CALIBRATED` / `CALIBRATED_OUT_OF_SAMPLE` status displays a percentage with **Calibrated estimate**.
- Missing, nonnumeric, out-of-range, unused and unavailable probabilities display **Unavailable** rather than an invented zero.
- Strategy ID and supplied version appear beside each row and in the modal. Missing identity is **Not reported**; no version is invented.

The candidate filter now uses a model-output scale instead of claiming win probabilities. Tier descriptions identify screening policy, and portfolio copy no longer claims all allocations use calibrated Kelly sizing.

## Persistent evidence panel

The panel remains visible across dashboard tabs. It distinguishes scan-snapshot age, model evidence, and historical research inputs. Age is computed from the scan timestamp, independently of polling success or event-stream connectivity, and updates every 30 seconds. A snapshot older than 15 minutes is marked stale. Missing, invalid, timezone-ambiguous and future timestamps remain explicit. This is scan freshness, not proof of market-bar freshness or a market-session-aware expiry rule.

HTTP/auth/network refresh failures remain visible while previously received data is retained. API-key changes clear the previous data and abort outstanding resource requests; superseded responses cannot restore the previous snapshot. Empty tables distinguish waiting for data, failed refreshes, scan errors, stale snapshots and completed scans returning no selection. An empty selection does not prove that required feeds were available.

`GET /api/evidence` uses the existing API-key dependency and the application's configured artifacts directory. It reads `coverage_audit/report.json` and `coverage_audit/*/report.json`, selects the valid report with the latest recorded period end (deterministic path tie-break), and returns a narrow summary. File modification time never establishes observation timing. Malformed/unreadable reports and missing audits remain unavailable; unreadable alternatives are counted. Unknown feed status is unavailable. Reports are not modified and raw paths/feed rows are not returned.

The historical panel identifies its audit period, missing/invalid feeds, stale membership, overlapping intervals, missing benchmark sessions and unavailable historical price-ingestion timing. These findings describe the saved research inputs; they do not assert which feeds the live scan requires. A new audit must be generated to update this evidence. The dashboard does not manufacture provider observations or change strategy gates.

## Verification

- `cd frontend; npm test`: **15 passed**, including **8 component-rendering tests**, 5 evidence/fetch utility tests and the 2 existing SSE parser tests. The production JSX components are compiled with Vite and rendered with React DOM Server; this is not mounted-browser interaction coverage.
- `cd frontend; npm run build`: passed.
- **80 focused Python tests passed** across dashboard evidence, coverage audit, scheduled replay and architecture; the new evidence loader has **100% statement coverage** in that run.
- Existing CI Ruff and strict Mypy command scopes pass using the repository's pinned tools (Ruff 0.4.7, Mypy 1.10.0; **76 source files**). No exclusions or suppressions were added. An initial invocation of the unrelated global Mypy installation encountered missing third-party stubs; the configured environment passes.
- The endpoint was tested directly with application paths and its declared authentication dependency inspected. This does not claim HTTP integration or full cross-platform CI coverage.
- Browser preview was attempted against a local fixture serving the production bundle with synthetic rows and the saved audit. Local socket access restrictions prevented loading it, so visual/browser-interaction verification remains unavailable. No market scan was triggered.

Local evidence and command outputs are under `artifacts/dashboard_evidence/2026-10-20/`. Generated artifacts are ignored by Git.
