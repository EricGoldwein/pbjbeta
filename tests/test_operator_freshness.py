"""Operator freshness layers and nurse candidate audit tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_audit_nurse_same_quarter_reacquisition_live_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(ROOT / "state" / "active_releases.json"))
    from operator_freshness import audit_nurse_staffing_candidate_state
    from release_control_plane import control_panel_payload

    control = control_panel_payload()
    row = next(
        r for r in control.get("datasets") or [] if r.get("dataset_id") == "cms.pbj_nurse_staffing"
    )
    if not row.get("pending"):
        pytest.skip("nurse pending candidate cleared — re-acquisition audit not applicable")
    audit = audit_nurse_staffing_candidate_state(control_row=row)
    assert audit["same_quarter"] is True
    assert audit["kind"] == "reacquisition_same_quarter"
    assert audit["is_redundant_reacquisition"] is True
    assert audit["active_hash"] != audit["pending_hash"]
    assert "standardized_PBJ" in audit["active_uri"]
    assert "PBJcsv" in audit["pending_uri"]


def test_discard_candidate_removes_pending(tmp_path: Path) -> None:
    from release_control_plane import ReleaseState, discard_candidate, load_candidates, record_candidate
    import release_control_plane as rcp

    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    (state / "release_candidates.json").write_text(
        json.dumps({"schema_version": 1, "datasets": {}, "updated_at": None}),
        encoding="utf-8",
    )
    raw = tmp_path / "nurse.csv"
    raw.write_text("a,b\n1,2\n", encoding="utf-8")

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(rcp, "candidates_path", lambda _root=None: state / "release_candidates.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)
    record_candidate(
        "cms.pbj_nurse_staffing",
        "CY2026Q1",
        ReleaseState.ACQUIRED,
        source_path=raw,
        root=tmp_path,
    )
    removed = discard_candidate("cms.pbj_nurse_staffing", reason="test", root=tmp_path)
    assert removed is not None
    assert "cms.pbj_nurse_staffing" not in load_candidates(tmp_path).get("datasets", {})
    monkeypatch.undo()


def test_derived_inventory_fields_use_upstream() -> None:
    import cms_data_ops as ops

    availability = ops.build_release_availability_context(
        "pbj.benchmarks.national",
        control_row={
            "active": {"active_release_id": "sha256:abc", "status": "ACTIVE"},
            "pending": None,
        },
        check_row={
            "upstream_active": {"cms.pbj_nurse_staffing": "CY2026Q1"},
            "new_release_available": False,
        },
        record={"human_name": "National benchmarks", "source_id": "pbj.benchmarks.national"},
    )
    assert availability["inventory_axis"] == "upstream"
    assert availability["inventory_label"] == "Built from"
    assert "CY2026Q1" in availability["inventory_value"]


def test_derived_benchmark_unknown_when_provenance_missing(tmp_path: Path, monkeypatch) -> None:
    import active_release_registry as arr
    import cms_data_ops as ops

    registry = tmp_path / "active_releases.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "pbj.benchmarks.national": {
                        "dataset_id": "pbj.benchmarks.national",
                        "active_release_id": "sha256:abc",
                        "status": "ACTIVE",
                        "metadata": {},
                    },
                    "cms.pbj_nurse_staffing": {
                        "dataset_id": "cms.pbj_nurse_staffing",
                        "active_release_id": "CY2026Q1",
                        "status": "ACTIVE",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: registry)
    monkeypatch.setattr("release_check.registry_path", lambda _root=None: registry)

    from release_check import derived_state

    derived = derived_state("pbj.benchmarks.national", ("cms.pbj_nurse_staffing",), root=tmp_path)
    assert derived["status"] == "UNKNOWN"
    assert derived["provenance_missing"] is True
    assert derived["new_release_available"] is False

    availability = ops.build_release_availability_context(
        "pbj.benchmarks.national",
        control_row={
            "active": {"active_release_id": "sha256:abc", "status": "ACTIVE"},
            "pending": None,
        },
        check_row=derived,
        record={"human_name": "National benchmarks", "source_id": "pbj.benchmarks.national"},
        root=tmp_path,
    )
    assert availability["inventory_status"] == "UNKNOWN"
    assert availability.get("new_release_available") is False


def test_derived_benchmark_stale_when_upstream_mismatch(tmp_path: Path, monkeypatch) -> None:
    import active_release_registry as arr
    from release_check import derived_state

    registry = tmp_path / "active_releases.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "pbj.benchmarks.national": {
                        "dataset_id": "pbj.benchmarks.national",
                        "active_release_id": "sha256:abc",
                        "hash": "derived-hash",
                        "status": "ACTIVE",
                        "metadata": {
                            "output_artifact_hash": "derived-hash",
                            "upstream_provenance": {
                                "cms.pbj_nurse_staffing": {
                                    "release_id": "CY2025Q4",
                                    "source_hash": "old-hash",
                                }
                            },
                        },
                    },
                    "cms.pbj_nurse_staffing": {
                        "dataset_id": "cms.pbj_nurse_staffing",
                        "active_release_id": "CY2026Q1",
                        "hash": "new-hash",
                        "status": "ACTIVE",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: registry)
    monkeypatch.setattr("release_check.registry_path", lambda _root=None: registry)
    derived = derived_state("pbj.benchmarks.national", ("cms.pbj_nurse_staffing",), root=tmp_path)
    assert derived["status"] == "STALE"
    assert derived["new_release_available"] is True
    assert derived["provenance_missing"] is False


def test_health_citations_freshness_layers_three_tiers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import active_release_registry as arr
    import cms_data_ops as ops
    import release_control_plane as rcp

    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    registry = state / "active_releases.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    "cms.health_citations": {
                        "dataset_id": "cms.health_citations",
                        "active_release_id": "2026-08",
                        "status": "ACTIVE",
                        "hash": "hc",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: registry)
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)
    monkeypatch.setattr(
        "provenance_freshness.downstream_stale_capabilities_for_source",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(
        "operator_freshness.count_stale_citation_packages",
        lambda **kwargs: (0, 0),
    )

    wf = ops.build_source_operator_workflow(
        "cms.health_citations",
        record={"human_name": "Health Citations"},
        snapshot={"release_id": "2026-08"},
        control_row={
            "active": {"active_release_id": "2026-08", "status": "ACTIVE"},
            "pending": None,
            "health": "PASS",
        },
        release_availability={
            "active_release_label": "Aug 2026",
            "publisher_latest_label": "Aug 2026",
            "new_release_available": False,
            "unchanged_in_latest_publication": True,
        },
        root=tmp_path,
    )
    keys = [layer["key"] for layer in wf.get("freshness_layers") or []]
    assert keys[0] == "cms_source"
    assert "canonical" in keys
