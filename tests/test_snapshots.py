from __future__ import annotations

import json

from core.snapshots import write_json_atomic


def test_write_json_atomic_replaces_complete_document(tmp_path):
    target = tmp_path / "state" / "latest_scan.json"
    write_json_atomic(target, {"value": 1})
    write_json_atomic(target, {"value": 2, "complete": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 2, "complete": True}
    assert list(target.parent.glob("*.tmp")) == []
