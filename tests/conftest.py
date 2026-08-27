"""Pytest fixtures — isolate release-control state from live state/ tree."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolated_release_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    active = state_dir / "active_releases.json"
    candidates = state_dir / "release_candidates.json"
    empty = {"schema_version": 1, "updated_at": None, "datasets": {}}
    active.write_text(json.dumps(empty) + "\n", encoding="utf-8")
    candidates.write_text(json.dumps(empty) + "\n", encoding="utf-8")
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(active))
    yield tmp_path
