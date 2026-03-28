Diagnostic and operator helper scripts live here.

These files are intentionally kept out of `tests/` so CI only exercises the
maintained automated suite.

Run them from the repository root unless a script states otherwise.
*** Add File: d:\Tradeidesa\grclaudescript\scripts\_bootstrap.py
from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root() -> Path:
    root = Path(__file__).resolve().parents[1]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
