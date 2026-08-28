"""Ownership downstream stale audit and attention queue wiring."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PBJ = ROOT.parent / "PBJapp"
sys.path.insert(0, str(ROOT))


def _write_min_policy(pbj_root: Path, *, active: str = "2026-07-17") -> None:
    policy_path = pbj_root / "ownership" / "ownership_release_policy.json"
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(
        json.dumps(
            {
                "active_release_date": active,
                "releases": {
                    active: {
                        "status": "active",
                        "bridge_lookup_filename": f"release_{active}_lookup.json",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_registry(tmp_path: Path, *, release_id: str = "2026-07-31") -> None:
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    active = {
        "schema_version": 1,
        "datasets": {
            "cms.snf_all_owners": {
                "dataset_id": "cms.snf_all_owners",
                "active_release_id": release_id,
                "status": "ACTIVE",
                "hash": "ownershash",
                "source_filename": "SNF_All_Owners_2026.07.31.csv",
                "source_uri": "file:///tmp/owners.csv",
            },
            "cms.snf_enrollments": {
                "dataset_id": "cms.snf_enrollments",
                "active_release_id": release_id,
                "status": "ACTIVE",
                "hash": "enrollhash",
                "source_filename": "SNF_Enrollments_2026.07.31.csv",
                "source_uri": "file:///tmp/enroll.csv",
            },
        },
    }
    (state / "active_releases.json").write_text(json.dumps(active), encoding="utf-8")


def test_audit_marks_stale_when_policy_lags_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ownership_downstream_rebuild import audit_ownership_downstream_stale

    pbj_root = tmp_path / "PBJapp"
    _write_min_policy(pbj_root, active="2026-07-17")
    _write_registry(tmp_path, release_id="2026-07-31")
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbj_root))

    audit = audit_ownership_downstream_stale(root=tmp_path, pbj_root=pbj_root)
    assert audit["is_stale"] is True
    assert audit["release_id"] == "2026-07-31"
    assert "facility.snf_owners" in audit["stale_capabilities"]
    assert "ownership.enrollment_ccn_bridge" in audit["stale_capabilities"]


def test_audit_current_when_policy_and_bridge_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ownership_downstream_rebuild import audit_ownership_downstream_stale

    pbj_root = tmp_path / "PBJapp"
    release_id = "2026-07-31"
    _write_min_policy(pbj_root, active=release_id)
    _write_registry(tmp_path, release_id=release_id)
    bridge_dir = pbj_root / "ownership" / "_derived" / "cms_snf_ownership_ccn_bridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    (bridge_dir / f"release_{release_id}_lookup.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbj_root))

    audit = audit_ownership_downstream_stale(root=tmp_path, pbj_root=pbj_root)
    assert audit["is_stale"] is False
    assert audit["stale_capabilities"] == []


def test_needs_attention_includes_downstream_item_when_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cms_data_ops import build_needs_attention_queue
    from ownership_downstream_rebuild import OWNERSHIP_DOWNSTREAM_SOURCE_ID

    pbj_root = tmp_path / "PBJapp"
    _write_min_policy(pbj_root, active="2026-07-17")
    _write_registry(tmp_path, release_id="2026-07-31")
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbj_root))

    control = {
        "datasets": [
            {
                "dataset_id": "cms.snf_all_owners",
                "active": {"active_release_id": "2026-07-31", "status": "ACTIVE"},
                "pending": None,
                "health": "CURRENT",
            },
            {
                "dataset_id": "cms.snf_enrollments",
                "active": {"active_release_id": "2026-07-31", "status": "ACTIVE"},
                "pending": None,
                "health": "CURRENT",
            },
        ]
    }
    queue = build_needs_attention_queue(
        control=control,
        check_by_dataset={},
        snapshots=[],
        root=tmp_path,
    )
    ids = [item.get("source_id") for item in queue]
    assert OWNERSHIP_DOWNSTREAM_SOURCE_ID in ids
    assert "cms.snf_all_owners" not in ids
    assert "cms.snf_enrollments" not in ids
    downstream = next(item for item in queue if item.get("source_id") == OWNERSHIP_DOWNSTREAM_SOURCE_ID)
    assert downstream["next_action"]["label"] == "Rebuild downstream"
    assert downstream["next_action"]["endpoint"] == "action_ownership_downstream_rebuild"
