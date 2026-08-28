"""Tests for provenance / freshness contract builder."""
from __future__ import annotations

import json
from pathlib import Path

import cms_data_ops as ops
import cms_data_paths
from active_release_registry import sha256_file
from cms_theme_publication import resolve_theme_publication
from provenance_freshness import build_source_provenance_freshness


FIXTURES = Path(__file__).parent / "fixtures"


def _write_registry(tmp_path: Path) -> None:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    payload = {
        "schema_version": 1,
        "datasets": {
            "cms.provider_info": {
                "active_release_id": "2026-08",
                "status": "ACTIVE",
                "hash": "pi-hash",
                "promoted_at": "2026-08-10T12:00:00+00:00",
                "validated_at": "2026-08-10T11:00:00+00:00",
                "downloaded_at": "2026-08-10T10:00:00+00:00",
            },
            "cms.health_citations": {
                "active_release_id": "2026-07",
                "status": "ACTIVE",
                "hash": "hc-hash",
                "promoted_at": "2026-07-10T12:00:00+00:00",
            },
        },
    }
    (state / "active_releases.json").write_text(json.dumps(payload), encoding="utf-8")
    (state / "release_candidates.json").write_text(
        json.dumps({"schema_version": 1, "datasets": {}}),
        encoding="utf-8",
    )


def _theme_publication_aug26():
    manifest = json.loads((FIXTURES / "cms_theme_manifest_2026-08-26.json").read_text(encoding="utf-8"))
    return resolve_theme_publication(
        archive_index=[
            {
                "type": "theme",
                "date": "2026-08-26",
                "id": "nh-aug26",
                "url": "/provider-data/dataset-archives/theme/nursing-homes/nursing-homes_2026-08-26.zip",
                "name": "nursing-homes_2026-08-26",
                "theme": "nursing-homes",
                "size": 1,
            }
        ],
        manifest=manifest,
    )


def test_build_source_provenance_freshness_contract_fields(tmp_path: Path, monkeypatch):
    import active_release_registry as arr
    import release_control_plane as rcp

    _write_registry(tmp_path)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)

    publication = _theme_publication_aug26()
    contract = build_source_provenance_freshness(
        "cms.health_citations",
        root=tmp_path,
        theme_publication=publication,
    )
    assert contract["publisher"]
    assert contract["cms_publication_id"] == "nh-aug26"
    assert contract["cms_publication_date"] == "2026-08-26"
    assert contract["processing_modified_date"] == "2026-08-01"
    assert contract["product_release_id"] == "2026-07"
    assert contract["cms_dataset_id"] == "r5ix-sfxw"
    assert contract["sha256"] == "hc-hash"
    assert contract["activated_at"]
    assert isinstance(contract["downstream_artifacts"], list)


def _setup_hc_bundle(tmp_path: Path, release_id: str = "2026-08") -> None:
    cit_dir = cms_data_paths.citations_dir(tmp_path)
    man_dir = cms_data_paths.provider_info_dir(tmp_path) / "_manifests" / release_id
    cit_dir.mkdir(parents=True)
    man_dir.mkdir(parents=True)
    basename = "NH_HealthCitations_Aug2026.csv"
    rows = ["CMS Certification Number (CCN),Survey Date"]
    rows.extend(f"015009,01/15/2024" for _ in range(120))
    csv_path = cit_dir / basename
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    digest = sha256_file(csv_path)
    manifest = {
        "release_key": release_id,
        "source_members": [{"basename": basename, "source_sha256": digest}],
    }
    (man_dir / "release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_health_citations_validate_action_when_bundle_provenance_ok(tmp_path: Path, monkeypatch):
    import active_release_registry as arr
    import release_control_plane as rcp

    _write_registry(tmp_path)
    _setup_hc_bundle(tmp_path)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)

    publication = _theme_publication_aug26()
    control_row = {
        "active": {
            "active_release_id": "2026-07",
            "status": "ACTIVE",
        },
        "pending": None,
        "health": "PASS",
        "impact": {"would_mark_stale": ["facility.citations"]},
    }
    record = {"human_name": "Health Citations"}
    availability = ops.build_release_availability_context(
        "cms.health_citations",
        control_row=control_row,
        record=record,
        root=tmp_path,
        theme_publication=publication,
    )
    assert availability["publisher_latest_label"] == "Aug 2026"
    assert availability["bundle_provenance_ok"] is True
    assert availability["local_artifact_ready"] is True

    wf = ops.build_source_operator_workflow(
        "cms.health_citations",
        record=record,
        snapshot={"release_id": "2026-08", "local_raw_present": True},
        control_row=control_row,
        release_availability=availability,
        theme_publication=publication,
        root=tmp_path,
    )
    assert wf["next_action"]["label"] == "Validate Aug 2026"
    assert wf["next_action"]["endpoint"] == "action_citations_validate"
    assert wf["next_action"]["wired"] is True
    assert wf["provenance_freshness"]["new_release_available"] is True


def test_health_citations_downstream_stale_when_facility_slice_older(tmp_path: Path, monkeypatch):
    import active_release_registry as arr
    import release_control_plane as rcp
    from provenance_freshness import downstream_stale_capabilities_for_source

    _write_registry(tmp_path)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)

    pbj_root = tmp_path / "pbjapp"
    cit_dir = pbj_root / "Citations"
    cit_dir.mkdir(parents=True)
    national = cit_dir / "NH_HealthCitations_Aug2026.csv"
    national.write_text(
        "CMS Certification Number (CCN),Survey Date\n"
        + "\n".join(f"015009,01/15/2024" for _ in range(120)),
        encoding="utf-8",
    )
    from active_release_registry import sha256_file

    digest = sha256_file(national)
    active_path = tmp_path / "state" / "active_releases.json"
    active_payload = json.loads(active_path.read_text(encoding="utf-8"))
    active_payload["datasets"]["cms.health_citations"].update(
        {
            "source_filename": national.name,
            "source_uri": national.as_uri(),
            "hash": digest,
            "validated_at": "2026-08-01T00:00:00+00:00",
        }
    )
    active_path.write_text(json.dumps(active_payload), encoding="utf-8")

    dep = pbj_root / "deployments" / "pbj320-335581"
    dep.mkdir(parents=True)
    facility_slice = dep / "facility_335581_citations.csv"
    facility_slice.write_text("CMS Certification Number (CCN),Survey Date\n", encoding="utf-8")

    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbj_root))
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(active_path))

    stale = downstream_stale_capabilities_for_source("cms.health_citations", root=tmp_path)
    assert "facility.citations" in stale

    contract = build_source_provenance_freshness(
        "cms.health_citations",
        root=tmp_path,
        theme_publication=_theme_publication_aug26(),
    )
    downstream = {row["consumer"]: row["freshness"] for row in contract["downstream_artifacts"]}
    assert downstream.get("facility.citations") == "STALE" or any(
        row.get("freshness") == "STALE" for row in contract["downstream_artifacts"]
    )


def test_health_citations_post_activation_next_action_rebuild_when_stale(tmp_path: Path, monkeypatch):
    import active_release_registry as arr
    import release_control_plane as rcp

    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    active = {
        "schema_version": 1,
        "datasets": {
            "cms.health_citations": {
                "active_release_id": "2026-08",
                "status": "ACTIVE",
                "hash": "hc-hash",
                "metadata": {"upstream_releases": {"cms.provider_info": "2026-08"}},
            },
        },
    }
    (state / "active_releases.json").write_text(json.dumps(active), encoding="utf-8")
    (state / "release_candidates.json").write_text(
        json.dumps({"schema_version": 1, "datasets": {}}), encoding="utf-8"
    )
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: state / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)

    pbj_root = tmp_path / "pbjapp"
    cit_dir = pbj_root / "Citations"
    cit_dir.mkdir(parents=True)
    national = cit_dir / "NH_HealthCitations_Aug2026.csv"
    national.write_text(
        "CMS Certification Number (CCN),Survey Date\n"
        + "\n".join(f"015009,01/15/2024" for _ in range(120)),
        encoding="utf-8",
    )
    from active_release_registry import sha256_file

    digest = sha256_file(national)
    active["datasets"]["cms.health_citations"].update(
        {
            "source_filename": national.name,
            "source_uri": national.as_uri(),
            "hash": digest,
            "validated_at": "2026-08-01T00:00:00+00:00",
        }
    )
    (state / "active_releases.json").write_text(json.dumps(active), encoding="utf-8")

    dep = pbj_root / "deployments" / "pbj320-335581"
    dep.mkdir(parents=True)
    (dep / "facility_335581_citations.csv").write_text(
        "CMS Certification Number (CCN),Survey Date\n", encoding="utf-8"
    )
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbj_root))
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(state / "active_releases.json"))

    publication = _theme_publication_aug26()
    control_row = {"active": active["datasets"]["cms.health_citations"], "pending": None, "health": "PASS"}
    record = {"human_name": "Health Citations"}
    availability = ops.build_release_availability_context(
        "cms.health_citations",
        control_row=control_row,
        record=record,
        root=tmp_path,
        theme_publication=publication,
    )
    assert availability["new_release_available"] is False
    assert availability["publisher_latest_label"] == "Aug 2026"

    wf = ops.build_source_operator_workflow(
        "cms.health_citations",
        record=record,
        snapshot={"release_id": "2026-08", "status": "CURRENT"},
        control_row=control_row,
        release_availability=availability,
        theme_publication=publication,
        root=tmp_path,
    )
    assert wf["operator_reference"]["status_label"] == "NEEDS ATTENTION"
    assert wf["next_action"]["label"] == "Rebuild facility packages"
    assert wf["next_action"]["wired"] is True
    assert wf["next_action"]["endpoint"] == "action_citation_packages_rebuild"
    assert "no deploy" in wf["next_action"]["detail"].lower()
