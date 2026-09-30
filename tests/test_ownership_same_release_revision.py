from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from active_release_registry import load_registry, registry_path
from data_ops_app import create_app
from ownership_downstream_rebuild import audit_ownership_downstream_stale
from ownership_pairing import (
    ENROLLMENTS,
    OWNERS,
    pairing_status,
    promote_ownership_pair,
    validate_ownership_pair,
)
from release_control_plane import load_candidates


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_revision_state(tmp_path: Path) -> tuple[Path, Path, Path]:
    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    old_owners = tmp_path / "SNF_All_Owners_2026.07.31.csv"
    revised_owners = tmp_path / "SNF_All_Owners_2026.07.31_update.csv"
    enrollments = tmp_path / "SNF_Enrollments_2026.07.31.csv"
    old_owners.write_text("ENROLLMENT ID,ASSOCIATE ID\n1,old\n", encoding="utf-8")
    revised_owners.write_text("ENROLLMENT ID,ASSOCIATE ID\n1,new\n2,newer\n", encoding="utf-8")
    enrollments.write_text("ENROLLMENT ID,CCN\n1,015009\n2,015010\n", encoding="utf-8")
    active = {
        "schema_version": 1,
        "updated_at": "2026-09-29T00:00:00+00:00",
        "datasets": {
            OWNERS: {
                "dataset_id": OWNERS,
                "active_release_id": "2026-07-31",
                "status": "ACTIVE",
                "hash": _hash(old_owners),
                "source_filename": old_owners.name,
                "source_uri": old_owners.as_uri(),
                "metadata": {"cms_publisher_url": "https://cms/owners-old.csv"},
            },
            ENROLLMENTS: {
                "dataset_id": ENROLLMENTS,
                "active_release_id": "2026-07-31",
                "status": "ACTIVE",
                "hash": _hash(enrollments),
                "source_filename": enrollments.name,
                "source_uri": enrollments.as_uri(),
                "metadata": {"cms_publisher_url": "https://cms/enrollments.csv"},
            },
        },
    }
    candidates = {
        "schema_version": 1,
        "updated_at": "2026-09-30T00:00:00+00:00",
        "datasets": {
            OWNERS: {
                "dataset_id": OWNERS,
                "release_id": "2026-07-31",
                "state": "ACQUIRED",
                "source_uri": revised_owners.as_uri(),
                "hash": _hash(revised_owners),
                "validation": {"status": "PASS", "row_count": 2, "hash": _hash(revised_owners)},
                "metadata": {
                    "change_kind": "REVISED",
                    "publisher_revision_changed": True,
                    "cms_publisher_url": "https://cms/owners-update.csv",
                    "cms_file_uuid": "owners-file-update",
                    "cms_dataset_version_id": "owners-version-update",
                },
            }
        },
    }
    (state / "active_releases.json").write_text(json.dumps(active), encoding="utf-8")
    (state / "release_candidates.json").write_text(json.dumps(candidates), encoding="utf-8")
    return old_owners, revised_owners, enrollments


def _publisher(source_id: str, **_kwargs) -> dict:
    if source_id == OWNERS:
        return {
            "source_id": OWNERS,
            "release_id": "2026-07-31",
            "filename": "SNF_All_Owners_2026.07.31_update.csv",
            "url": "https://cms/owners-update.csv",
            "file_uuid": "owners-file-update",
            "dataset_version_id": "owners-version-update",
        }
    return {
        "source_id": ENROLLMENTS,
        "release_id": "2026-07-31",
        "filename": "SNF_Enrollments_2026.07.31.csv",
        "url": "https://cms/enrollments.csv",
    }


def test_one_sided_revision_validates_without_fake_enrollments_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_revision_state(tmp_path)
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(tmp_path / "state" / "active_releases.json"))
    monkeypatch.setattr("ownership_pairing.detect_cms_publisher_release", _publisher)

    before_enroll = load_registry(registry_path(tmp_path))["datasets"][ENROLLMENTS]
    status = pairing_status(tmp_path)
    assert status["mode"] == "ONE_SIDED_REVISION"
    assert status["pending"]["enrollment_role"] == "UNCHANGED_ACTIVE"
    assert status["review_state"] == "READY FOR REVIEW"

    result = validate_ownership_pair(root=tmp_path)
    assert result["mode"] == "ONE_SIDED_REVISION"
    assert result["linkage"]["missing_owners_enrollment_ids"] == 0
    assert result["linkage"]["owners_row_count"] == 2
    pending = load_candidates(tmp_path)["datasets"]
    assert pending[OWNERS]["state"] == "VALIDATED"
    assert ENROLLMENTS not in pending
    assert load_registry(registry_path(tmp_path))["datasets"][ENROLLMENTS] == before_enroll


def test_one_sided_revision_promotes_only_revised_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _old, revised, _enroll = _write_revision_state(tmp_path)
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(tmp_path / "state" / "active_releases.json"))
    monkeypatch.setattr("ownership_pairing.detect_cms_publisher_release", _publisher)
    validate_ownership_pair(root=tmp_path)
    enroll_before = load_registry(registry_path(tmp_path))["datasets"][ENROLLMENTS]

    result = promote_ownership_pair(root=tmp_path)
    active = load_registry(registry_path(tmp_path))["datasets"]
    assert result["unchanged_active_partners"] == [ENROLLMENTS]
    assert active[OWNERS]["active_release_id"] == "2026-07-31"
    assert active[OWNERS]["hash"] == _hash(revised)
    assert active[ENROLLMENTS] == enroll_before


def test_same_release_hash_change_marks_ownership_policy_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old, revised, enroll = _write_revision_state(tmp_path)
    registry = json.loads((tmp_path / "state" / "active_releases.json").read_text(encoding="utf-8"))
    registry["datasets"][OWNERS]["hash"] = _hash(revised)
    registry["datasets"][OWNERS]["source_uri"] = revised.as_uri()
    (tmp_path / "state" / "active_releases.json").write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(tmp_path / "state" / "active_releases.json"))

    pbj = tmp_path / "pbj"
    policy = pbj / "ownership" / "ownership_release_policy.json"
    policy.parent.mkdir(parents=True)
    policy.write_text(
        json.dumps(
            {
                "active_release_date": "2026-07-31",
                "releases": {
                    "2026-07-31": {
                        "ownership_source_sha256": _hash(old),
                        "enrollment_source_sha256": _hash(enroll),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    bridge = pbj / "ownership" / "_derived" / "cms_snf_ownership_ccn_bridge" / "release_2026-07-31_lookup.json"
    bridge.parent.mkdir(parents=True)
    bridge.write_text("{}", encoding="utf-8")

    audit = audit_ownership_downstream_stale(root=tmp_path, pbj_root=pbj)
    assert audit["is_stale"] is True
    assert "facility.snf_owners" in audit["stale_capabilities"]
    assert "ownership.enrollment_ccn_bridge" in audit["stale_capabilities"]
    assert any("cms.snf_all_owners hash" in reason for reason in audit["blocking_reasons"])


def test_revision_review_page_shows_evidence_preview_and_separate_activation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_revision_state(tmp_path)
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(tmp_path / "state" / "active_releases.json"))
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-password")
    monkeypatch.setattr("ownership_pairing.detect_cms_publisher_release", _publisher)
    validate_ownership_pair(root=tmp_path)

    app = create_app()
    client = app.test_client()
    with client.session_transaction() as session:
        session["data_ops_authenticated"] = True
    response = client.get("/sources/cms.snf_ownership_pair", query_string={"check_cms": "0"})
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "SNF Ownership Pair Review" in html
    assert "REVISED" in html
    assert "SNF All Owners (PECOS)" in html
    assert "UNCHANGED ACTIVE" in html
    assert "owners-version-update" in html
    assert '<details class="do-collapse-panel" open>' in html
    assert "SNF All Owners (PECOS) preview" in html
    assert "Make SNF All Owners ACTIVE" in html
    assert "Rebuild downstream, packaging, and deployment remain separate" in html
    assert "/actions/ownership-pair/activate" in html
