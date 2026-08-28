"""Citation package rebuild audit and orchestration tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write_hc_registry(tmp_path: Path, *, national_path: Path) -> None:
    from active_release_registry import sha256_file

    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    digest = sha256_file(national_path)
    active = {
        "schema_version": 1,
        "datasets": {
            "cms.health_citations": {
                "dataset_id": "cms.health_citations",
                "active_release_id": "2026-08",
                "status": "ACTIVE",
                "source_filename": national_path.name,
                "source_uri": national_path.as_uri(),
                "validated_at": "2026-08-01T00:00:00+00:00",
                "hash": digest,
                "metadata": {},
            }
        },
    }
    (state / "active_releases.json").write_text(json.dumps(active), encoding="utf-8")
    (state / "release_candidates.json").write_text(
        json.dumps({"schema_version": 1, "datasets": {}}), encoding="utf-8"
    )


def test_health_citations_catalog_is_external_recurring() -> None:
    from release_source_catalog import BY_ID, UpdateMechanism

    catalog = BY_ID["cms.health_citations"]
    assert catalog.mechanism == UpdateMechanism.EXTERNAL_RECURRING
    assert catalog.upstream == ()


def test_audit_and_rebuild_citation_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import active_release_registry as arr
    import release_control_plane as rcp

    pbj_root = tmp_path / "pbjapp"
    cit_dir = pbj_root / "Citations"
    cit_dir.mkdir(parents=True)
    national = cit_dir / "NH_HealthCitations_Aug2026.csv"
    national.write_text(
        "CMS Certification Number (CCN),Survey Date\n"
        + "\n".join("335581,01/15/2024" for _ in range(120)),
        encoding="utf-8",
    )
    _write_hc_registry(tmp_path, national_path=national)

    dep = pbj_root / "deployments" / "pbj320-335581"
    dep.mkdir(parents=True)
    stale_slice = dep / "facility_335581_citations.csv"
    stale_slice.write_text("CMS Certification Number (CCN),Survey Date\n", encoding="utf-8")

    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbj_root))
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(tmp_path / "state" / "active_releases.json"))

    from citation_packages_rebuild import audit_stale_citation_packages, rebuild_citation_packages

    pre = audit_stale_citation_packages(root=tmp_path, pbj_root=pbj_root)
    assert pre["is_stale"] is True
    assert pre["stale_count"] == 1
    assert pre["stale_packages"][0]["ccn"] == "335581"

    dry = rebuild_citation_packages(root=tmp_path, pbj_root=pbj_root, dry_run=True)
    assert dry["would_rebuild"] == ["335581"]

    result = rebuild_citation_packages(root=tmp_path, pbj_root=pbj_root)
    assert len(result["rebuilt"]) == 1
    assert result["rebuilt"][0]["rows"] >= 1
    assert result["stale_remaining"] == 0
    assert stale_slice.with_suffix(stale_slice.suffix + ".source.json").is_file()

    post = audit_stale_citation_packages(root=tmp_path, pbj_root=pbj_root)
    assert post["is_stale"] is False


def test_rebuild_targets_single_ccn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import active_release_registry as arr
    import release_control_plane as rcp

    pbj_root = tmp_path / "pbjapp"
    cit_dir = pbj_root / "Citations"
    cit_dir.mkdir(parents=True)
    national = cit_dir / "NH_HealthCitations_Aug2026.csv"
    national.write_text(
        "CMS Certification Number (CCN),Survey Date\n"
        + "\n".join(f"{ccn},01/15/2024" for ccn in ("335581", "335513") for _ in range(60)),
        encoding="utf-8",
    )
    _write_hc_registry(tmp_path, national_path=national)

    for ccn in ("335581", "335513"):
        dep = pbj_root / "deployments" / f"pbj320-{ccn}"
        dep.mkdir(parents=True)
        (dep / f"facility_{ccn}_citations.csv").write_text(
            "CMS Certification Number (CCN),Survey Date\n", encoding="utf-8"
        )

    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbj_root))
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(tmp_path / "state" / "active_releases.json"))

    from citation_packages_rebuild import rebuild_citation_packages

    result = rebuild_citation_packages(ccns=["335581"], root=tmp_path, pbj_root=pbj_root)
    assert len(result["rebuilt"]) == 1
    assert result["rebuilt"][0]["ccn"] == "335581"
    assert result["stale_remaining"] == 1


def test_rebuild_zero_row_ccn_writes_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import active_release_registry as arr
    import release_control_plane as rcp

    pbj_root = tmp_path / "pbjapp"
    cit_dir = pbj_root / "Citations"
    cit_dir.mkdir(parents=True)
    national = cit_dir / "NH_HealthCitations_Aug2026.csv"
    national.write_text("CMS Certification Number (CCN),Survey Date\n335581,01/15/2024\n", encoding="utf-8")
    _write_hc_registry(tmp_path, national_path=national)

    ccn = "225783"
    dep = pbj_root / "deployments" / f"pbj320-{ccn}"
    dep.mkdir(parents=True)
    stale_slice = dep / f"facility_{ccn}_citations.csv"
    stale_slice.write_text("CMS Certification Number (CCN),Survey Date\n", encoding="utf-8")

    monkeypatch.setattr(arr, "registry_path", lambda _root=None: tmp_path / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: tmp_path)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbj_root))
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(tmp_path / "state" / "active_releases.json"))

    from citation_packages_rebuild import audit_stale_citation_packages, rebuild_citation_packages

    pre = audit_stale_citation_packages(root=tmp_path, pbj_root=pbj_root)
    assert pre["stale_count"] == 1

    result = rebuild_citation_packages(ccns=[ccn], root=tmp_path, pbj_root=pbj_root)
    assert result["failed"] == []
    assert len(result["rebuilt"]) == 1
    assert result["rebuilt"][0]["rows"] == 0
    assert stale_slice.read_text(encoding="utf-8").strip().endswith("Survey Date")
    assert stale_slice.with_suffix(stale_slice.suffix + ".source.json").is_file()
    assert audit_stale_citation_packages(root=tmp_path, pbj_root=pbj_root)["stale_count"] == 0
