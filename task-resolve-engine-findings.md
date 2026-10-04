# Task Plan: Resolve Engine Findings (Sovereign Engine v14.6.0)

## Objective
Resolve the remaining 5 findings identified in the Sovereign Engine codebase analysis, following the user-selected phased rollout and non-breaking facade strategy:
1. Finding #6: Scorer C901 Complexity (`core/scorer.py`)
2. Finding #5: Narrow 34 Blind Excepts (`BLE001`) project-wide
3. Finding #8: Cleanup `utils/retry.py` shim
4. Finding #2: Decompose `ScanService` (`core/services.py`) preserving facade
5. Finding #3: SystemConfig Composition (`core/config.py`) via property delegation

---

## Phase 1: Complexity, Exception Safety & Shim Cleanup
- [x] **Step 1.1**: Audit C901 complexity and extract helper gates/functions in `core/scorer.py`:
  - `score_candidate_pass1`
  - `score_candidate_pass2`
  - `apply_cohort_factor_ranking`
  - `_regime_gate`
- [x] **Step 1.2**: Audit BLE001 blind excepts using ruff and narrow exception types across target files.
- [x] **Step 1.3**: Audit and clean up `utils/retry.py` shim.
- [x] **Checkpoint 1**: Run full test suite (`pytest`) and verify all 735 tests pass with zero regressions.

---

## Phase 2: Architectural Modularization (ScanService & SystemConfig)
- [x] **Step 2.1**: Decompose `ScanService` (`core/services.py`):
  - Extract `PositionMonitorService`, `RegimeService`, and `ScoringPipelineService`
  - Retain `ScanService` as orchestrator facade for all 29 callers
- [x] **Step 2.2**: Refactor `SystemConfig` (`core/config.py`):
  - Compose logical sub-configs (`.regime`, `.signal`, `.portfolio`, `.market_data`, `.execution_cost`, `.alerts`, `.backtest`, `.settings`, `.app_settings`)
  - Provide backwards-compatible property delegation for all callers and `from_app_settings` factory
- [x] **Checkpoint 2**: Run full test suite, lint check, and verify zero regressions (735 passed, 0 ruff errors).

