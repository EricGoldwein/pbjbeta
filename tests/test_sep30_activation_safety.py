from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_local_module(name: str):
    """Avoid cross-repo sys.path pollution from legacy full-suite tests."""
    spec = importlib.util.spec_from_file_location(f"_sep30_{name}", ROOT / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    from active_release_registry import sha256_file

    return sha256_file(path)


def _active_row(dataset_id: str, release_id: str, source: Path, **metadata) -> dict:
    return {
        "dataset_id": dataset_id,
        "active_release_id": release_id,
        "source_filename": source.name,
        "source_uri": source.resolve().as_uri(),
        "validated_at": "2026-09-29T00:00:00+00:00",
        "hash": _sha(source),
        "status": "ACTIVE",
        "schema_version": 1,
        "metadata": metadata,
    }


def _write_registry(root: Path, datasets: dict[str, dict]) -> Path:
    target = root / "state" / "active_releases.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({"schema_version": 1, "updated_at": None, "datasets": datasets}),
        encoding="utf-8",
    )
    return target


@pytest.mark.parametrize("source_id", ["cms.provider_info", "cms.pbj_nurse_staffing"])
def test_new_upstream_can_progress_detected_acquired_validated(tmp_path: Path, source_id: str) -> None:
    from release_control_plane import ReleaseState, load_candidates, record_candidate

    artifact = tmp_path / f"{source_id}.csv"
    artifact.write_text("a,b\n1,2\n", encoding="utf-8")
    record_candidate(source_id, "new-release", ReleaseState.DETECTED, root=tmp_path)
    assert load_candidates(tmp_path)["datasets"][source_id]["state"] == "DETECTED"
    record_candidate(source_id, "new-release", ReleaseState.ACQUIRED, source_path=artifact, root=tmp_path)
    assert load_candidates(tmp_path)["datasets"][source_id]["state"] == "ACQUIRED"
    record_candidate(
        source_id,
        "new-release",
        ReleaseState.VALIDATED,
        source_path=artifact,
        validation={"status": "PASS"},
        root=tmp_path,
    )
    assert load_candidates(tmp_path)["datasets"][source_id]["state"] == "VALIDATED"


@pytest.mark.parametrize(
    ("source_id", "missing_derivative"),
    [
        ("cms.provider_info", "pbj.benchmarks.geo_cmi"),
        ("cms.pbj_nurse_staffing", "pbj.benchmarks.region"),
    ],
)
def test_make_active_fails_closed_and_preserves_active_and_candidate(
    tmp_path: Path, source_id: str, missing_derivative: str
) -> None:
    import cms_data_ops
    from data_ops_approval import ApprovalError
    from release_control_plane import ReleaseState, load_candidates, record_candidate

    old = tmp_path / "old.csv"
    old.write_text("old,value\n1,old\n", encoding="utf-8")
    candidate = tmp_path / "candidate.csv"
    candidate.write_text("new,value\n1,new\n", encoding="utf-8")
    registry = _write_registry(tmp_path, {source_id: _active_row(source_id, "old-release", old)})
    record_candidate(
        source_id,
        "new-release",
        ReleaseState.VALIDATED,
        source_path=candidate,
        validation={"status": "PASS"},
        root=tmp_path,
    )
    before = registry.read_bytes()
    audit = tmp_path / "approval.jsonl"

    with pytest.raises(ApprovalError, match=missing_derivative):
        cms_data_ops.approve_release_authoritative(
            source_id, "new-release", root=tmp_path, audit_path=audit
        )

    assert registry.read_bytes() == before
    assert load_candidates(tmp_path)["datasets"][source_id]["state"] == "VALIDATED"
    assert not audit.exists() or audit.read_text(encoding="utf-8") == ""


def test_unrelated_source_families_have_no_derivative_activation_gate() -> None:
    from derived_provenance import assert_activation_derivatives_ready

    for source_id in (
        "cms.pbj_non_nurse_staffing",
        "cms.snf_all_owners",
        "cms.snf_enrollments",
    ):
        assert_activation_derivatives_ready(source_id)


@pytest.mark.parametrize(
    ("dataset_id", "builder", "filename"),
    [
        ("pbj.benchmarks.state", "generate_metrics.py", "state_quarterly_metrics.csv"),
        ("pbj.benchmarks.national", "generate_metrics.py", "national_quarterly_metrics.csv"),
        ("pbj.peer_distribution", "lite_report.py", "facility_lite_metrics.csv"),
    ],
)
def test_derived_candidate_build_does_not_overwrite_served_active(
    tmp_path: Path, dataset_id: str, builder: str, filename: str
) -> None:
    from derived_provenance import record_validated_derived_candidate

    upstream = tmp_path / "nurse.csv"
    upstream.write_text("nurse\nold\n", encoding="utf-8")
    served = tmp_path / "served" / filename
    served.parent.mkdir()
    served.write_bytes(b"served-active-bytes\n")
    staged = tmp_path / "state" / "derived_candidates" / "new" / filename
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"candidate-bytes\n")
    _write_registry(
        tmp_path,
        {
            "cms.pbj_nurse_staffing": _active_row(
                "cms.pbj_nurse_staffing", "CY2026Q1", upstream
            ),
            dataset_id: _active_row(dataset_id, "old-derived", served),
        },
    )
    before = served.read_bytes()
    record = record_validated_derived_candidate(
        dataset_id, staged, builder=builder, root=tmp_path
    )

    assert served.read_bytes() == before
    assert record["state"] == "VALIDATED"
    assert record["source_uri"] == staged.resolve().as_uri()
    assert record["metadata"]["served_target_uri"] == served.resolve().as_uri()


def test_explicit_derived_promotion_copies_candidate_to_served_path(tmp_path: Path) -> None:
    from derived_provenance import record_validated_derived_candidate
    from release_control_plane import promote_candidate

    upstream = tmp_path / "nurse.csv"
    upstream.write_text("nurse\nold\n", encoding="utf-8")
    served = tmp_path / "state_quarterly_metrics.csv"
    served.write_bytes(b"old-served\n")
    staged = tmp_path / "state" / "derived_candidates" / "state.csv"
    staged.parent.mkdir(parents=True)
    staged.write_bytes(b"new-candidate\n")
    _write_registry(
        tmp_path,
        {
            "cms.pbj_nurse_staffing": _active_row(
                "cms.pbj_nurse_staffing", "CY2026Q1", upstream
            ),
            "pbj.benchmarks.state": _active_row(
                "pbj.benchmarks.state", "old-derived", served
            ),
        },
    )
    record_validated_derived_candidate(
        "pbj.benchmarks.state", staged, builder="generate_metrics.py", root=tmp_path
    )
    assert served.read_bytes() == b"old-served\n"

    promoted = promote_candidate("pbj.benchmarks.state", root=tmp_path)

    assert served.read_bytes() == b"new-candidate\n"
    assert promoted["source_uri"] == served.resolve().as_uri()
    assert promoted["hash"] == _sha(served)


def test_actual_state_national_peer_builders_leave_served_bytes_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("duckdb")
    from release_control_plane import ReleaseState, record_candidate

    generate_metrics = _load_local_module("generate_metrics").generate_metrics
    generate_lite_metrics = _load_local_module("lite_report").generate_lite_metrics

    inputs = tmp_path / "standardized_PBJ"
    inputs.mkdir()
    (inputs / "PBJ.csv").write_text(
        "PROVNUM,PROVNAME,STATE,COUNTY_NAME,WorkDate,CY_Qtr,MDScensus,"
        "Hrs_RNDON,Hrs_RNadmin,Hrs_RN,Hrs_LPNadmin,Hrs_LPN,Hrs_CNA,Hrs_NAtrn,Hrs_MedAide,"
        "Hrs_RNDON_ctr,Hrs_RNadmin_ctr,Hrs_RN_ctr,Hrs_LPNadmin_ctr,Hrs_LPN_ctr,Hrs_CNA_ctr,Hrs_NAtrn_ctr,Hrs_MedAide_ctr\n"
        "015009,Example,AL,Autauga,2026-01-01,2026Q1,10,1,1,2,1,2,3,0,0,0,0,0,0,0,0,0,0\n",
        encoding="utf-8",
    )
    nurse = tmp_path / "nurse-candidate.csv"
    nurse.write_text("candidate\n", encoding="utf-8")
    served_paths = {
        "pbj.benchmarks.state": tmp_path / "served" / "state_quarterly_metrics.csv",
        "pbj.benchmarks.national": tmp_path / "served" / "national_quarterly_metrics.csv",
        "pbj.peer_distribution": tmp_path / "served" / "pbj_lite" / "facility_lite_metrics.csv",
    }
    for path in served_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"served-before-build\n")
    _write_registry(
        tmp_path,
        {
            "cms.pbj_nurse_staffing": _active_row(
                "cms.pbj_nurse_staffing", "CY2025Q4", nurse
            ),
            **{
                dataset_id: _active_row(dataset_id, "old-derived", path)
                for dataset_id, path in served_paths.items()
            },
        },
    )
    record_candidate(
        "cms.pbj_nurse_staffing",
        "CY2026Q1",
        ReleaseState.VALIDATED,
        source_path=nurse,
        validation={"status": "PASS"},
        root=tmp_path,
    )
    before = {dataset_id: path.read_bytes() for dataset_id, path in served_paths.items()}
    stage = tmp_path / "state" / "derived_candidates" / "cms.pbj_nurse_staffing" / "CY2026Q1"
    monkeypatch.chdir(tmp_path)

    generate_metrics(output_dir=stage, control_root=tmp_path)
    generate_lite_metrics(input_dir=stage, output_dir=stage, control_root=tmp_path)

    assert all(path.read_bytes() == before[dataset_id] for dataset_id, path in served_paths.items())
    assert (stage / "state_quarterly_metrics.csv").is_file()
    assert (stage / "national_quarterly_metrics.csv").is_file()
    assert (stage / "pbj_lite" / "facility_lite_metrics.csv").is_file()


@pytest.mark.parametrize("provenance", [None, {"cms.pbj_nurse_staffing": {"release_id": "CY2025Q4", "source_hash": "old-hash"}}])
def test_premium_preflight_rejects_unknown_or_stale_derived_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provenance: dict | None
) -> None:
    local_client = _load_local_module("active_release_client")
    local_contract = _load_local_module("premium_source_contract")
    load_active_release = local_client.load_active_release
    ReleaseRegistryError = local_contract.ReleaseRegistryError
    validate_derived_upstreams = local_contract.validate_derived_upstreams

    nurse = tmp_path / "nurse.csv"
    nurse.write_text("nurse\ncurrent\n", encoding="utf-8")
    derived = tmp_path / "state.csv"
    derived.write_text("state,value\nNY,1\n", encoding="utf-8")
    metadata = {} if provenance is None else {"upstream_provenance": provenance}
    registry = _write_registry(
        tmp_path,
        {
            "cms.pbj_nurse_staffing": _active_row(
                "cms.pbj_nurse_staffing", "CY2026Q1", nurse
            ),
            "pbj.benchmarks.state": _active_row(
                "pbj.benchmarks.state", "derived", derived, **metadata
            ),
        },
    )
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(registry))
    release = load_active_release("pbj.benchmarks.state")
    with pytest.raises(ReleaseRegistryError, match="UNKNOWN|STALE"):
        validate_derived_upstreams(release)
