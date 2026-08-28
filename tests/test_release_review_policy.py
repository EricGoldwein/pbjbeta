"""Release Review policy and operator transition tests."""
from __future__ import annotations

import json
from pathlib import Path

import cms_data_ops as ops
import cms_data_paths
import pytest


def _write_control_plane_state(
    tmp_path: Path,
    *,
    active: dict | None = None,
    pending: dict | None = None,
) -> None:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    active_datasets = {}
    for source_id, row in (active or {}).items():
        active_datasets[source_id] = {
            "active_release_id": row["release_id"],
            "status": row.get("status", "ACTIVE"),
            "hash": row.get("hash", "abc"),
            "metadata": row.get("metadata", {}),
        }
        if row.get("zweli_status"):
            active_datasets[source_id]["metadata"]["zweli_status"] = row["zweli_status"]
    pending_datasets = {}
    for source_id, row in (pending or {}).items():
        pending_datasets[source_id] = {
            "release_id": row["release_id"],
            "state": row.get("state", "ACQUIRED"),
            "validation": row.get("validation") or {"status": row.get("validation_status", "PASS")},
            "zweli_status": row.get("zweli_status"),
        }
    (state / "active_releases.json").write_text(
        json.dumps({"schema_version": 1, "datasets": active_datasets}),
        encoding="utf-8",
    )
    (state / "release_candidates.json").write_text(
        json.dumps({"schema_version": 1, "datasets": pending_datasets}),
        encoding="utf-8",
    )


def test_release_review_excludes_acquired_only_candidates(tmp_path: Path):
    _write_control_plane_state(
        tmp_path,
        active={
            "cms.provider_info": {"release_id": "2026-08", "status": "ACTIVE"},
            "cms.pbj_nurse_staffing": {"release_id": "CY2025Q4", "status": "ACTIVE"},
        },
        pending={
            "cms.provider_info": {"release_id": "2026-09", "state": "ACQUIRED"},
            "cms.pbj_nurse_staffing": {"release_id": "CY2026Q1", "state": "ACQUIRED"},
            "cms.snf_all_owners": {"release_id": "2026-07-31", "state": "ACQUIRED"},
            "cms.snf_enrollments": {"release_id": "2026-07-31", "state": "ACQUIRED"},
        },
    )
    import active_release_registry as arr
    import release_control_plane as rcp

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)
    try:
        control = rcp.control_panel_payload(tmp_path)
        items = ops.release_review_items(check_cms=False, root=tmp_path, control=control)
        source_ids = {item["source_id"] for item in items}
        assert "cms.provider_info" not in source_ids
        assert "cms.pbj_nurse_staffing" not in source_ids
        assert "cms.snf_all_owners" not in source_ids
        assert "cms.snf_enrollments" not in source_ids
        assert "cms.snf_ownership_pair" in source_ids
    finally:
        monkeypatch.undo()


def test_release_review_health_citations_validated_approvable(tmp_path: Path):
    _write_control_plane_state(
        tmp_path,
        active={
            "cms.health_citations": {"release_id": "2026-07", "status": "ACTIVE"},
        },
        pending={
            "cms.health_citations": {
                "release_id": "2026-08",
                "state": "VALIDATED",
                "validation_status": "PASS",
            },
        },
    )
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    items = ops.release_review_items(check_cms=False, root=tmp_path, control=control)
    hc = next(i for i in items if i["source_id"] == "cms.health_citations")
    assert hc["human_state"] == "Ready to activate"
    assert hc["approvable"] is True
    assert hc["primary_action_label"] == "Activate Aug 2026"
    assert hc.get("zweli_status") is None


def test_release_review_focus_filter(tmp_path: Path):
    _write_control_plane_state(
        tmp_path,
        active={"cms.health_citations": {"release_id": "2026-07", "status": "ACTIVE"}},
        pending={
            "cms.health_citations": {"release_id": "2026-08", "state": "VALIDATED"},
            "cms.provider_info": {"release_id": "2026-09", "state": "VALIDATED"},
        },
    )
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    items = ops.release_review_items(
        check_cms=False,
        root=tmp_path,
        control=control,
        focus_source_id="cms.health_citations",
        focus_release_id="2026-08",
    )
    assert len(items) == 1
    assert items[0]["source_id"] == "cms.health_citations"
    assert items[0]["release_id"] == "2026-08"


def test_post_activation_operator_target_stale_downstream(tmp_path: Path, monkeypatch):
    _write_control_plane_state(
        tmp_path,
        active={
            "cms.health_citations": {
                "release_id": "2026-08",
                "status": "ACTIVE",
                "metadata": {"upstream_releases": {"cms.provider_info": "2026-08"}},
            },
            "cms.provider_info": {"release_id": "2026-08", "status": "ACTIVE"},
        },
        pending={},
    )
    import active_release_registry as arr
    import release_control_plane as rcp
    from release_review_policy import post_activation_operator_target

    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)
    control = rcp.control_panel_payload(tmp_path)
    target = post_activation_operator_target(
        "cms.provider_info",
        release_id="2026-08",
        root=tmp_path,
        control=control,
    )
    assert target["redirect_endpoint"] in {"sources", "source_detail"}
    assert "ACTIVE" in target["flash"]


def test_next_action_validated_health_citations_links_focused_review(tmp_path: Path):
    control_row = {
        "active": {"active_release_id": "2026-07", "status": "ACTIVE"},
        "pending": {"release_id": "2026-08", "state": "VALIDATED"},
    }
    wf = ops.build_source_operator_workflow(
        "cms.health_citations",
        record={"human_name": "Health Citations"},
        snapshot=None,
        control_row=control_row,
        root=tmp_path,
    )
    na = wf["next_action"]
    assert na["endpoint"] == "release_review"
    assert na["endpoint_args"]["source_id"] == "cms.health_citations"
    assert na["endpoint_args"]["release_id"] == "2026-08"
    assert "Activate" in na["label"]


def _write_hc_candidate_state(
    tmp_path: Path,
    *,
    validation_status: str = "PASS",
    candidate_state: str = "VALIDATED",
    candidate_hash: str = "aug-candidate-hash",
) -> None:
    import release_control_plane as rcp

    cit = tmp_path / "Citations"
    cit.mkdir(parents=True)
    aug = cit / "NH_HealthCitations_Aug2026.csv"
    jul = cit / "NH_HealthCitations_Jul2026.csv"
    body = "CMS Certification Number (CCN),Survey Date\n" + "\n".join(
        f"015009,01/15/2024" for _ in range(120)
    )
    aug.write_text(body, encoding="utf-8")
    jul.write_text(body, encoding="utf-8")
    state_dir = rcp._state_dir(tmp_path)
    state_dir.mkdir(parents=True, exist_ok=True)
    active = {
        "schema_version": 1,
        "datasets": {
            "cms.health_citations": {
                "active_release_id": "2026-07",
                "status": "ACTIVE",
                "hash": "jul-active-hash",
                "source_uri": jul.as_uri(),
            }
        },
    }
    pending = {
        "schema_version": 1,
        "datasets": {
            "cms.health_citations": {
                "dataset_id": "cms.health_citations",
                "release_id": "2026-08",
                "state": candidate_state,
                "hash": candidate_hash,
                "source_uri": aug.as_uri(),
                "metadata": {"structural_status": validation_status},
                "validation": {
                    "status": validation_status,
                    "validated_at": "2026-08-27T00:00:00+00:00",
                },
            }
        },
    }
    (state_dir / "active_releases.json").write_text(json.dumps(active), encoding="utf-8")
    (state_dir / "release_candidates.json").write_text(json.dumps(pending), encoding="utf-8")


def test_hc_validated_pass_is_promotion_eligible(tmp_path: Path):
    from data_ops_approval import ApprovalError
    from release_review_policy import assert_promotion_eligible, structural_status_from_candidate

    _write_hc_candidate_state(tmp_path)
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    pending = next(
        row["pending"]
        for row in control["datasets"]
        if row["dataset_id"] == "cms.health_citations"
    )
    assert structural_status_from_candidate(pending) == "PASS"
    assert_promotion_eligible("cms.health_citations", "2026-08", root=tmp_path, control=control)


def test_hc_structural_fail_blocked(tmp_path: Path):
    from data_ops_approval import ApprovalError
    from release_review_policy import assert_promotion_eligible

    _write_hc_candidate_state(tmp_path, validation_status="FAIL", candidate_state="VALIDATED")
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    with pytest.raises(ApprovalError, match="Structural validation"):
        assert_promotion_eligible(
            "cms.health_citations", "2026-08", root=tmp_path, control=control
        )


def test_hc_not_validated_blocked(tmp_path: Path):
    from data_ops_approval import ApprovalError
    from release_review_policy import assert_promotion_eligible

    _write_hc_candidate_state(tmp_path, candidate_state="ACQUIRED", validation_status="")
    import release_control_plane as rcp

    control = rcp.control_panel_payload(tmp_path)
    with pytest.raises(ApprovalError):
        assert_promotion_eligible(
            "cms.health_citations", "2026-08", root=tmp_path, control=control
        )


def test_hc_review_provenance_uses_candidate_not_active(tmp_path: Path):
    from release_review_policy import build_candidate_review_provenance

    _write_hc_candidate_state(tmp_path, candidate_hash="aug-candidate-hash")
    review = build_candidate_review_provenance(
        "cms.health_citations", "2026-08", root=tmp_path
    )
    assert "NH_HealthCitations_Aug2026.csv" in str(review["candidate_artifact_path"])
    assert review["candidate_sha256"] == "aug-candidate-hash"
    assert review["active_release_id"] == "2026-07"
    assert "NH_HealthCitations_Jul2026.csv" in str(review["active_artifact_path"])
    assert review["active_sha256"] == "jul-active-hash"


def test_hc_failed_eligibility_writes_no_audit(tmp_path: Path, monkeypatch):
    import cms_data_ops as ops_mod
    from data_ops_approval import ApprovalError, read_audit

    _write_hc_candidate_state(tmp_path, validation_status="FAIL", candidate_state="VALIDATED")
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(cms_data_paths, "repo_root", lambda: tmp_path)
    with pytest.raises(ApprovalError):
        ops_mod.approve_release_authoritative(
            "cms.health_citations",
            "2026-08",
            root=tmp_path,
            audit_path=audit,
        )
    assert read_audit(audit) == []
