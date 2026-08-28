"""Atomic SNF ownership pair promotion tests (no live activation)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from active_release_registry import load_registry, registry_path
from ownership_pairing import ENROLLMENTS, OWNERS, promote_ownership_pair
from release_control_plane import ReleaseState, load_candidates, promote_active_pair


def _write_pair_fixtures(
    tmp_path: Path,
    *,
    owners_csv: Path,
    enroll_csv: Path,
    candidate_state: str = "VALIDATED",
) -> None:
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    active = {
        "schema_version": 1,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "datasets": {
            OWNERS: {
                "dataset_id": OWNERS,
                "active_release_id": "2026-07-17",
                "status": "ACTIVE",
                "hash": "old",
                "source_uri": owners_csv.as_uri(),
            },
            ENROLLMENTS: {
                "dataset_id": ENROLLMENTS,
                "active_release_id": "2026-07-17",
                "status": "ACTIVE",
                "hash": "old",
                "source_uri": enroll_csv.as_uri(),
            },
        },
    }
    owners_hash = __import__("hashlib").sha256(owners_csv.read_bytes()).hexdigest()
    enroll_hash = __import__("hashlib").sha256(enroll_csv.read_bytes()).hexdigest()
    candidates = {
        "schema_version": 1,
        "updated_at": "2026-01-01T00:00:00+00:00",
        "datasets": {
            OWNERS: {
                "dataset_id": OWNERS,
                "release_id": "2026-07-31",
                "state": candidate_state,
                "source_uri": owners_csv.as_uri(),
                "hash": owners_hash,
                "validation": {"status": "PASS", "validated_at": "2026-08-28T00:00:00+00:00"},
            },
            ENROLLMENTS: {
                "dataset_id": ENROLLMENTS,
                "release_id": "2026-07-31",
                "state": candidate_state,
                "source_uri": enroll_csv.as_uri(),
                "hash": enroll_hash,
                "validation": {"status": "PASS", "validated_at": "2026-08-28T00:00:00+00:00"},
            },
        },
    }
    (state / "active_releases.json").write_text(json.dumps(active), encoding="utf-8")
    (state / "release_candidates.json").write_text(json.dumps(candidates), encoding="utf-8")


def test_promote_active_pair_both_members_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    owners = tmp_path / "owners.csv"
    enroll = tmp_path / "enroll.csv"
    owners.write_text("ORG_ID,CCN\n1,015009\n", encoding="utf-8")
    enroll.write_text("CCN,ORG_ID\n015009,1\n", encoding="utf-8")
    _write_pair_fixtures(tmp_path, owners_csv=owners, enroll_csv=enroll)

    import active_release_registry as arr
    import release_control_plane as rcp

    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "candidates_path", lambda _root=None: tmp_path / "state" / "release_candidates.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda: tmp_path)

    result = promote_active_pair((OWNERS, ENROLLMENTS), root=tmp_path)
    assert result["release_id"] == "2026-07-31"

    active = load_registry(registry_path(tmp_path))["datasets"]
    candidates = load_candidates(tmp_path)["datasets"]
    for dataset_id in (OWNERS, ENROLLMENTS):
        assert active[dataset_id]["active_release_id"] == "2026-07-31"
        assert active[dataset_id]["status"] == "ACTIVE"
        assert candidates[dataset_id]["state"] == ReleaseState.ACTIVE.value
        assert candidates[dataset_id]["superseded_release_id"] == "2026-07-17"


def test_promote_active_pair_rolls_back_active_on_candidate_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owners = tmp_path / "owners.csv"
    enroll = tmp_path / "enroll.csv"
    owners.write_text("ORG_ID,CCN\n1,015009\n", encoding="utf-8")
    enroll.write_text("CCN,ORG_ID\n015009,1\n", encoding="utf-8")
    _write_pair_fixtures(tmp_path, owners_csv=owners, enroll_csv=enroll)

    import active_release_registry as arr
    import release_control_plane as rcp

    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "candidates_path", lambda _root=None: tmp_path / "state" / "release_candidates.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda: tmp_path)

    before_active = (tmp_path / "state" / "active_releases.json").read_text(encoding="utf-8")
    before_candidates = (tmp_path / "state" / "release_candidates.json").read_text(encoding="utf-8")
    real_atomic = rcp._atomic_json
    calls = {"n": 0}

    def _flaky_atomic(path: Path, payload: dict) -> None:
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("simulated candidates write failure")
        real_atomic(path, payload)

    monkeypatch.setattr(rcp, "_atomic_json", _flaky_atomic)

    with pytest.raises(OSError, match="simulated candidates write failure"):
        promote_active_pair((OWNERS, ENROLLMENTS), root=tmp_path)

    assert json.loads((tmp_path / "state" / "active_releases.json").read_text(encoding="utf-8")) == json.loads(
        before_active
    )
    assert (tmp_path / "state" / "release_candidates.json").read_text(encoding="utf-8") == before_candidates


def test_promote_ownership_pair_delegates_to_atomic_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    owners = tmp_path / "owners.csv"
    enroll = tmp_path / "enroll.csv"
    owners.write_text("ORG_ID,CCN\n1,015009\n", encoding="utf-8")
    enroll.write_text("CCN,ORG_ID\n015009,1\n", encoding="utf-8")
    _write_pair_fixtures(tmp_path, owners_csv=owners, enroll_csv=enroll)

    import active_release_registry as arr
    import ownership_pairing as op
    import release_control_plane as rcp

    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "candidates_path", lambda _root=None: tmp_path / "state" / "release_candidates.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda: tmp_path)
    monkeypatch.setattr(op, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")

    result = promote_ownership_pair(root=tmp_path)
    assert result["release_id"] == "2026-07-31"
    assert result["next_operator_action"] == "Rebuild downstream"
    active = load_registry(registry_path(tmp_path))["datasets"]
    assert active[OWNERS]["active_release_id"] == "2026-07-31"
    assert active[ENROLLMENTS]["active_release_id"] == "2026-07-31"
