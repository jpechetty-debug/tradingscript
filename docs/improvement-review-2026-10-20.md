# Improvement review — 20 October 2026

Scope: recommendations based on the existing graphify graph, current CodeGraph source, `core/`, `scripts/`, `frontend/`, `tests/`, `docs/`, saved research artifacts, and CI configuration. Application code and strategy settings were not changed.

## Assessment

The highest-value work is to correct the newer research simulator, establish trustworthy data coverage, and make the dashboard explain evidence quality. More indicators, thresholds, or AI features will not resolve these limitations.

Several foundations already exist: application-owned runtime/persistence, cancellation and killswitch controls, shared scan serialization, causal research feeds, cash-constrained replay, and an append-only paper journal. Extend and test these rather than repeating the earlier architecture rewrite.

## What graphify revealed

The existing graph has **1,659 nodes, 4,324 edges and 113 communities**. Its report is dated 5 September 2026, built at commit `67fc115a`; the reviewed checkout is `d26d541`.

The query was expanded against the graph's own vocabulary to: `services scorer portfolio backtest calibration provider persistence worker runtime test security dashboard`. BFS located 878 nodes; a separate NetworkX inspection preserved edge relation and confidence attributes.

The largest degree hubs include `SystemConfig` (165), `RegimeTracker` (119), `MarketRegime` (85), `services.py` (68), and `TickerResult` (56). Degree measures connectivity, not defects. Extracted edges connect `score_ticker()` to factor computation, targets, Kelly sizing and regime/config types; `ScanService` also has inferred dependency edges, which are navigation hints rather than verified call paths.

**The graph needs refreshing before further architectural decisions.** Nineteen current core files have no source representation in it, including the alpha modules, swing replay/research modules, application runtime, calibration, snapshots, trades, paper journal and leader-pullback setup. Twenty-two of its 108 distinct nonempty source paths no longer resolve. The CLI also reports the old node-ID scheme, with possible same-name-file collisions. `.graphify_python` points to an earlier checkout under `D:\Tradeidesa`, not this checkout.

Current CodeGraph source supplied the evidence below. Old graph locations and earlier review documents were not treated as current defects automatically.

## Prioritized changes

| Priority | Change | Why it matters | Completion evidence |
|---|---|---|---|
| 1 | Correct or replace the standalone research portfolio loop | It uses current-close information for opening allocations and mishandles same-day exits | Causality, cash timing and same-day stop regression checks; reproduce saved reports |
| 2 | Audit universe coverage and dated feeds | Membership is frozen after March 2022 in the new builder; saved alpha runs lack required feeds | Date-by-date membership/price coverage and ingestion audit; explicit unavailable intervals |
| 3 | Restore the configured quality checks | Current Ruff and strict Mypy commands fail | Existing CI commands pass without excluding the new scripts |
| 4 | Surface strategy/evidence status in the dashboard | Heuristic probabilities appear as percentages without their status | Component tests for unvalidated scores, missing feeds and stale data |
| 5 | Finish the worker job contract | Requested tickers and metadata do not affect execution | A restricted-universe job evaluates only that universe and preserves runtime context |
| 6 | Turn the paper journal into a daily evidence report | Infrastructure does not establish prospective performance by itself | Reconciled scans, first-observed signals, linked fills/exits and cost/timing exceptions |
| 7 | Refresh graphify and maintain module boundaries incrementally | Stale graphs omit much of today's research and runtime | Correct interpreter/root, current commit metadata and scoped source coverage |

### 1. Correct the research portfolio loop first

**Implemented after this review:** the chronological book, regression checks and matched local LP reruns are documented in [the replay correction](standalone-replay-fix-2026-10-20.md). The original findings below describe the reviewed baseline. PEAD reproduction remains unavailable without its earnings snapshot.

Source: `scripts/swing_research2.py:74–92`.

Three issues are visible in this separate simulator:

- Lines 81–84 credit all exits for a session before allocating its new entries, including exits that occur later that session.
- Line 85 values held positions at the current session's close; line 88 uses that value to size entries at the session's open.
- Lines 81–84 process exits before new positions are appended at line 90. A position that opens and closes on the same date misses that exit pass; the equality check will never close it on a later date.

Two isolated synthetic checks executed only the actual `portfolio()` function extracted from its AST, with no download or full research run:

1. A trade opens at 100 and stops at 90 on the same day. At 0.5% account risk and zero costs, expected final NAV is **0.995**. The function returns **1.0**, retaining the position instead of realizing the stop.
2. Change only an existing holding's day-two close from 100 to 1,000, after another instrument's day-two opening entry. Keep that holding's later exit price unchanged. Day-three NAV changes from **1.05** to **1.0725**, demonstrating that information unavailable at the opening entry affected its size.

Prefer adapting the existing chronological replay book in `core/swing_replay.py` after inspecting its extension points. If a separate engine remains necessary, use explicit opening/stop/closing events, prior-close sizing information, and cash released only when its exit occurs. Keep realized and net risk-unit returns consistent: this script subtracts costs in `ret` at line 59 but its `R` column is gross.

Rerun the affected LP/PEAD portfolio comparisons after correction. These findings concern this standalone script; they do not establish the same defects in the separate core replay implementation.

### 2. Make data availability a measured deliverable

**Implemented after this review:** the offline [universe and dated-feed audit](coverage-audit-2026-10-20.md) exports daily membership/price coverage, ingestion diagnostics and explicit unavailable intervals. It also identifies overlapping EQUITASBNK intervals. Missing genuine feeds, historical timing evidence and complete membership remain unavailable; no substitute observations were manufactured.

Source: `scripts/swing_build_members.py:1–5,20–27`, `core/leader_pullback.py:58–87`, `core/services.py:1333–1340`, and `artifacts/alpha_roadmap_validation/scan/report.json`.

The membership builder uses four snapshots from October 2020 through March 2022 and explicitly freezes later membership. Its header records manual symbol mapping. This is partial historical membership, not complete membership coverage for the 2025–26 evaluation period.

The core leader-pullback scanner accepts an optional `members` set, but the application call does not pass it. Its percentile ranks therefore describe the configured scan universe. Historical membership support alone does not make application ranks full Nifty-500 ranks.

Add a coverage report recording membership effective date, publication/ingestion evidence, stale intervals, symbol changes, available/delisted prices, ranking denominator and missing instruments. Rebuild symbol mapping deterministically with recorded source hashes. Label partial coverage explicitly, and define a research policy for stale membership before evaluating results.

The saved alpha scan has **zero qualified candidates**, with missing sector mapping/history, surveillance, earnings/fundamentals and instrument flow evidence. Its saved replay has **zero trades**. These are saved run results, not a newly executed market evaluation. Prioritize genuine dated feeds and a coverage/rejection funnel before expanding Phase 4 model complexity. Do not turn unknown evidence into a pass just to generate trades.

### 3. Restore quality checks and reproducible research entry points

**Configured checks restored after this review:** the existing full-scope Ruff and strict Mypy commands now pass with no new exclusions or rule changes. See [quality-check restoration and verification](quality-checks-2026-10-20.md). The error counts below describe the original reviewed baseline.

Current checks on this checkout found **73 Ruff errors** and **8 strict Mypy errors**. Ruff findings are concentrated in the new download/membership/research scripts and `tests/test_leader_pullback.py`. Mypy reports six missing function annotations in the new research/download scripts and two `dataclasses.fields()` argument-type errors at `core/config.py:649,683`.

Refactor executable top-level script bodies into typed `main()` functions with CLI arguments, context-managed files, validated input schemas and explicit output paths. Declare the additional `pypdf` dependency used by the membership builder in an appropriate research dependency group. Preserve original source archives and their hashes outside mutable working outputs.

CI already includes Python 3.10/3.11/3.12 on Linux, Windows Python 3.11, lint, typing, frontend build/tests and 80% core coverage. Keep those checks. Replace the frontend `npm ci || npm install` fallback with a strict lockfile install so dependency drift is visible. The declared Python 3.10+ support should also cover research scripts, not just imported core modules.

### 4. Make the dashboard reflect evidence quality

**Implemented after this review:** [dashboard evidence status](dashboard-evidence-2026-10-20.md) adds shared row/modal labels, strategy identity, a persistent snapshot/research-input panel and explicit refresh/empty states. The 15 frontend tests include 8 component-rendering checks for heuristic scores, missing feeds and stale snapshots. Browser interaction verification remains unavailable because local socket access was restricted. The original findings below describe the reviewed baseline.

Source: `core/scorer.py:109–113,1289–1293`, `frontend/src/components/PortfolioTable.jsx:153–157`, `frontend/src/components/CandidatesTable.jsx:196–200`, and `frontend/src/App.jsx:52–127`.

The backend carries `strategy_id` and `probability_status`, including `HEURISTIC_UNVALIDATED`. Both tables format `prob_win` as a percentage, while the frontend source has no references to those status fields.

Show strategy/version, heuristic or calibrated status, data timestamp, entry bounds and paper/research status beside a candidate. Distinguish a valid scan with no qualifying signals from unavailable required data. Show failed refreshes and stale data explicitly; several fetch paths currently log failures or ignore non-success responses while retaining prior state.

The frontend test command currently executes only **two SSE parser tests**. Add component/integration coverage for credential changes, scan failure, reconnection, stale snapshots, heuristic score labels, and killswitch activation/reset. Keep the current SSE parser checks.

### 5. Finish the worker boundary before scaling it

**Implemented after this review:** the [worker job contract](worker-contract-2026-10-21.md) carries an explicit universe, detached configuration, runtime paths, typed scan options and metadata. The worker uses application-owned services and canonical snapshots; restricted-universe and separate-runtime regressions pass. Runtime component paths now follow the owning application. The original findings below describe the reviewed baseline.

Source: `core/worker.py:27–34,51–81`, `core/services.py:1228–1229`, and `server.py:359–366`.

`ScanJob` exposes `tickers` and `metadata`, but `execute_scan_task()` calls `svm.run_scan()` without either. The scan service itself selects `ALL_TICKERS`. The worker also resolves the lazy default service bundle and uses its own `__dict__` serializer instead of `core/snapshots.py:format_ticker`.

Either implement the advertised job inputs through the service boundary or reject unsupported inputs. Include a versioned configuration reference, runtime/state location, cancellation semantics and a canonical payload schema. Ensure concurrent jobs do not accidentally share mutable default runtime context.

The current API scan path already supplies its owning application's runtime services and cancellation probe directly. Do not describe the worker limitation as an established cross-instance bug in that API path. Fix the worker contract before adding Redis/Celery or multiple process workers.

### 6. Collect and report prospective evidence

**Implemented after this review:** [daily paper reconciliation](paper-reconciliation-2026-10-21.md) reads the existing journal without mutation, reports first-observed signals and explicit fills/exits, and flags costs, timing and link exceptions. The current journal is empty, so its generated report makes no performance claim. Collection of genuine future observations remains necessary. The original recommendations below describe the reviewed baseline.

Source: `docs/paper-trading-ledger.md`, `docs/swing-research-2026-10-02.md`, and `docs/alpha-roadmap.md`.

The saved continuous swing-policy evaluation reports **80 trades, −0.0800R net expectancy and −1.55% NAV return**. Its mean-R interval includes zero. The leader-pullback module also labels its edge unproven. These are strategy-specific historical observations; they should not be pooled into an overall project performance claim.

Build a read-only daily evidence report over the existing append-only journal: completed scan dates, no-signal dates, original strategy/config hashes, first-observed signal times, linked fills/exits, fees/slippage completeness, delayed/future-dated submissions, and reconciliation exceptions. Keep observed paper events distinct from replay and actual executed outcomes.

Use frozen protocols and independent future observations to evaluate strategies. The existing annual walk-forward, attribution, stress and paper-ledger components already provide much of the machinery. Adding another ranking factor does not supply the missing outcome evidence.

### 7. Keep architecture changes narrow

The graph's hubs identify central contracts that deserve stable schemas and explicit ownership. Much of the earlier server/config/calibration separation has already been completed.

If maintenance warrants it, move existing persistence, data preparation, regime, scoring and position-monitor classes out of `core/services.py` into dedicated modules while preserving `ServiceBundle` and public entry points. Review dependency direction with the refreshed graph and protect behavior with integration checks. A wholesale config or service rewrite is lower priority than the verified simulator, data and worker issues above.

Refresh graphify using the installed version's migration workflow, a corrected interpreter/root and exclusions for environments, generated artifacts, backups and scratch copies. Preserve the old graph as an audit artifact if rebuilding. Check source coverage and current-commit metadata before relying on the rebuilt graph.

## Verification performed

- Queried the existing graphify graph and inspected connectivity/freshness with NetworkX; used current CodeGraph source for implementation findings.
- Reproduced same-day exit and opening-allocation causality defects with the isolated standalone portfolio function.
- Ran the repository's Ruff and strict Mypy commands: 73 and 8 errors respectively.
- Ran `npm test` in `frontend/`: 2 passed.
- Focused Python checks passed: **39 tests**, covering worker behavior, leader-pullback signals/journaling and alpha research; 3 alpha CLI workflow cases were deselected. API/journal integration selections did not complete locally and were interrupted after stalling. The first attempt also encountered a pytest temporary-directory permission error; subsequent runs used unique workspace-local temporary directories. No current full-suite or coverage claim is made.
- No market download, paid API call, broker execution or strategy change was performed.

## Suggested implementation order

First correct the standalone replay and bring the new research scripts through existing CI. Next audit dated data coverage and update dashboard evidence labels. Then complete the worker contract and daily paper reconciliation. Refresh the graph before any wider module refactor.
