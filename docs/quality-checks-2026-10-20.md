# Configured quality checks restored — 20 October 2026

Priority 3 is complete: the existing CI Ruff and strict Mypy commands both pass locally across their full configured paths, including all new research/audit scripts. No check scope, exclusion, lint rule, type-ignore override, dependency pin or CI workflow was changed.

## Changes

- Reformatted imports and compound statements in `scripts/swing_build_members.py`, `scripts/swing_dl.py`, `scripts/swing_dl_earn.py` and `tests/test_leader_pullback.py`, resolving the remaining 23 Ruff findings.
- Annotated the earnings helper as `get(t: str) -> tuple[str, pd.DataFrame | None]`.
- Added an explicitly typed union of the seven settings dataclasses and a shared ordered `_settings_slices()` helper. `SystemConfig.from_app_settings()` and payload construction now give `dataclasses.fields()` concrete dataclass types instead of Mypy's inferred `object`. Field precedence, order and runtime behavior remain the same.

The script and test formatting diffs were checked against their Git-baseline executable ASTs, normalizing split import statements and removing annotations for comparison. Executable structure is unchanged. Existing configuration characterization tests exercise both `from_app_settings()` and `settings_payload` construction.

## Completion evidence

The tools match the repository pins: **Ruff 0.4.7**, **Mypy 1.10.0**. Validation used Windows / Python 3.12.10; remote CI and its full platform matrix were not run.

The existing quality-job commands pass, with exit code 0:

```text
ruff check core/ tests/ run.py screener_v14_modular.py server.py scripts/ tools/ utils/ --output-format=github
mypy core/ run.py screener_v14_modular.py server.py scripts/ tools/ utils/ --ignore-missing-imports --disallow-untyped-defs --warn-return-any --warn-unused-ignores
```

Mypy reports **Success: no issues found in 75 source files**. Ruff produces no findings. `.github/workflows/ci.yml` and `pyproject.toml` were verified unchanged against Git HEAD.

**72 focused regression tests passed** across architecture/configuration adapters, secret handling, leader-pullback signals/journaling, coverage audit and standalone replay. Tests used an isolated workspace-local temporary directory. This is not a full-test-suite or full-core-coverage claim. No network downloader or missing-source membership builder was executed to validate formatting.

Machine-readable command arguments, versions, exit codes and output are saved in `artifacts/quality_checks/2026-10-20/report.json`, with `ruff.txt` and `mypy.txt`. These local generated artifacts are ignored by Git. Previous research snapshots and audit evidence were preserved.
