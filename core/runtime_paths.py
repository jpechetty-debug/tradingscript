"""
core/runtime_paths.py
=====================
Central runtime paths for mutable state, generated artifacts, and logs.

The source tree should stay focused on code. Runtime files such as Platt
calibration, trade logs, backtest exports, and structured logs live under
explicit directories so operators know what can be deleted, persisted, or
archived without touching application code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(raw_path: str | os.PathLike[str], *, root: Path = REPO_ROOT) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    state_dir: Path
    artifacts_dir: Path
    logs_dir: Path
    portfolio_state_file: Path
    platt_calibration_file: Path
    trade_log_file: Path
    factor_weights_file: Path
    degradation_log_file: Path
    telemetry_log_file: Path

    @classmethod
    def discover(cls, root: Path | None = None) -> "RuntimePaths":
        repo_root = root or REPO_ROOT
        state_dir = _resolve_repo_path(os.getenv("STATE_DIR", "state"), root=repo_root)
        artifacts_dir = _resolve_repo_path(os.getenv("ARTIFACTS_DIR", "artifacts"), root=repo_root)
        logs_dir = _resolve_repo_path(os.getenv("LOG_DIR", "logs"), root=repo_root)

        return cls(
            root=repo_root,
            state_dir=state_dir,
            artifacts_dir=artifacts_dir,
            logs_dir=logs_dir,
            portfolio_state_file=_resolve_repo_path(
                os.getenv("PORTFOLIO_STATE_PATH", state_dir / "portfolio_state.json"),
                root=repo_root,
            ),
            platt_calibration_file=_resolve_repo_path(
                os.getenv("PLATT_CALIB_PATH", state_dir / "platt_calibration.json"),
                root=repo_root,
            ),
            trade_log_file=_resolve_repo_path(
                os.getenv("TRADE_LOG_PATH", state_dir / "trade_log.json"),
                root=repo_root,
            ),
            factor_weights_file=_resolve_repo_path(
                os.getenv("WEIGHTS_PATH", state_dir / "factor_weights.json"),
                root=repo_root,
            ),
            degradation_log_file=_resolve_repo_path(
                os.getenv("DEGRADATION_LOG_PATH", logs_dir / "data_provider_degradation.log"),
                root=repo_root,
            ),
            telemetry_log_file=_resolve_repo_path(
                os.getenv("TELEMETRY_LOG_PATH", logs_dir / "sovereign.jsonl"),
                root=repo_root,
            ),
        )


RUNTIME_PATHS = RuntimePaths.discover()


def ensure_runtime_dirs(paths: RuntimePaths = RUNTIME_PATHS) -> None:
    for directory in (paths.state_dir, paths.artifacts_dir, paths.logs_dir):
        directory.mkdir(parents=True, exist_ok=True)


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_artifact_path(path: str | os.PathLike[str], paths: RuntimePaths = RUNTIME_PATHS) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else paths.artifacts_dir / candidate
