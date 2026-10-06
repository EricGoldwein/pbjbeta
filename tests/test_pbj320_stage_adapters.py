"""Focused tests for non-PI PBJ320 Stage adapters."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_pbj320_publish_provider_info import _init_git_with_remote  # noqa: E402


def _mock_baseline(monkeypatch: pytest.MonkeyPatch, base_sha: str) -> None:
    import pbj320_publication as pub

    monkeypatch.setattr(pub, "fetch_publish_base", lambda *_a, **_k: base_sha)
    monkeypatch.setattr(pub, "resolve_publish_branch", lambda *_a: ("master", "origin"))

    def _fake_prepare(dev: Path, *, remote: str, branch: str, base_sha: str, worktree_path: Path) -> Path:
        if worktree_path.exists():
            import shutil

            shutil.rmtree(worktree_path, ignore_errors=True)
        worktree_path.mkdir(parents=True, exist_ok=True)
        (worktree_path / "README.md").write_text("seed\n", encoding="utf-8")
        (worktree_path / "ownership").mkdir(exist_ok=True)
        (worktree_path / "ownership" / "_derived" / "cms_snf_ownership_ccn_bridge").mkdir(parents=True, exist_ok=True)
        (worktree_path / "ownership" / "_sources" / "cms_snf_enrollments" / "raw" / "downloaded").mkdir(
            parents=True, exist_ok=True
        )
        (worktree_path / "ownership" / "_sources" / "cms_snf_all_owners" / "raw" / "downloaded").mkdir(
            parents=True, exist_ok=True
        )
        (worktree_path / "scripts").mkdir(exist_ok=True)
        (worktree_path / "data" / "derived" / "sff" / "tables").mkdir(parents=True, exist_ok=True)
        (worktree_path / "data_sources" / "cms" / "sff" / "raw").mkdir(parents=True, exist_ok=True)
        (worktree_path / "pbj-wrapped" / "public").mkdir(parents=True, exist_ok=True)
        (worktree_path / "provider_info").mkdir(exist_ok=True)
        (worktree_path / "provider_info" / "ProviderInfoNorm_2026_08.csv").write_text(
            "PROVNUM,PROVNAME\n015009,Test\n", encoding="utf-8"
        )
        return worktree_path

    monkeypatch.setattr(pub, "prepare_baseline_worktree", _fake_prepare)

    def _fake_fp(dev: Path, base: str, rel: str) -> dict:
        return {"path": rel, "mode": "PUBLISHED_BASELINE", "present_at_base": False, "sha256": None}

    monkeypatch.setattr(pub, "fingerprint_baseline_path", _fake_fp)


def _write_ownership_registry(control: Path, tmp_path: Path, *, release_id: str = "2026-07-31") -> tuple[Path, Path]:
    owners_csv = tmp_path / "owners.csv"
    enroll_csv = tmp_path / "enroll.csv"
    owners_csv.write_text("ENROLLMENT ID,ORGANIZATION NAME\nO1,Owner\n", encoding="utf-8")
    enroll_csv.write_text("ENROLLMENT ID,CCN\nO1,335513\n", encoding="utf-8")
    from active_release_registry import sha256_file

    active = {
        "schema_version": 1,
        "datasets": {
            "cms.snf_all_owners": {
                "dataset_id": "cms.snf_all_owners",
                "active_release_id": release_id,
                "status": "ACTIVE",
                "hash": sha256_file(owners_csv),
                "source_filename": "SNF_All_Owners_2026.07.31.csv",
                "source_uri": owners_csv.as_uri(),
            },
            "cms.snf_enrollments": {
                "dataset_id": "cms.snf_enrollments",
                "active_release_id": release_id,
                "status": "ACTIVE",
                "hash": sha256_file(enroll_csv),
                "source_filename": "SNF_Enrollments_2026.07.31.csv",
                "source_uri": enroll_csv.as_uri(),
            },
        },
    }
    reg = control / "state" / "active_releases.json"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps(active), encoding="utf-8")
    (reg.parent / 'release_checks.json').write_text(json.dumps({'datasets': [
        {'dataset_id': key, 'status': 'CURRENT', 'publisher_sha256': member['hash'],
         'publisher_checked_at': 'fixture', 'cms_release_vintage': '2026-08',
         'snapshot_date': release_id, 'cms_dataset_version_id': key+'-version',
         'publisher_file_uuid': key+'-file', 'publisher_url': 'https://data.cms.gov/'+key}
        for key, member in active['datasets'].items()
    ]}), encoding='utf-8')
    return owners_csv, enroll_csv


def _write_bridge_builder(worktree: Path) -> None:
    script = worktree / "ownership" / "_derived" / "cms_snf_ownership_ccn_bridge" / "build_release_lookup.py"
    script.write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "release = sys.argv[sys.argv.index('--ownership-release') + 1]\n"
        "root = Path(__file__).resolve().parents[3]\n"
        "out = root / 'ownership' / '_derived' / 'cms_snf_ownership_ccn_bridge' / f'release_{release}_lookup.json'\n"
        "out.parent.mkdir(parents=True, exist_ok=True)\n"
        "out.write_text(json.dumps({'release': release, 'ccn_by_enrollment': {'335513': 'O1'}}), encoding='utf-8')\n"
        "print(json.dumps({'lookup': str(out)}))\n",
        encoding="utf-8",
    )


def test_ownership_stage_produces_staged_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import active_release_registry as arr
    import pbj320_stage_ownership as own_mod
    import release_control_plane as rcp

    dev, _, base_sha = _init_git_with_remote(tmp_path)
    control = tmp_path / "control"
    _write_ownership_registry(control, tmp_path)
    _mock_baseline(monkeypatch, base_sha)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr("ownership_pairing.registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr("ownership_downstream_rebuild.registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.delenv("PBJ_ACTIVE_RELEASE_REGISTRY", raising=False)
    monkeypatch.setenv("PBJ_ROOT", str(dev))
    monkeypatch.setenv("PBJ_PUBLISH_BRANCH", "master")

    def _fake_apply(wt: Path, *, release_id: str, owners_active: dict, enroll_active: dict) -> dict[str, str]:
        rels = own_mod._ownership_paths(
            release_id,
            owners_name="SNF_All_Owners_2026.07.31.csv",
            enroll_name="SNF_Enrollments_2026.07.31.csv",
        )
        policy_path = wt / rels["ownership_release_policy"].replace("/", os.sep)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        policy_path.write_text(json.dumps({"active_release_date": release_id, "releases": {}}), encoding="utf-8")
        lookup = wt / rels["ownership_bridge_lookup"].replace("/", os.sep)
        lookup.parent.mkdir(parents=True, exist_ok=True)
        lookup.write_text(json.dumps({"ccn_by_enrollment": {"335513": "O1"}}), encoding="utf-8")
        enroll = wt / rels["enrollment_release_artifact"].replace("/", os.sep)
        enroll.parent.mkdir(parents=True, exist_ok=True)
        enroll.write_text("ENROLLMENT ID,CCN\nO1,335513\n", encoding="utf-8")
        return rels

    monkeypatch.setattr(own_mod, "_apply_ownership_overlay", _fake_apply)

    result = own_mod.stage_ownership_pair_for_pbj320(release_id="2026-07-31", root=control)
    manifest = result["manifest"]
    assert result["status"] == "STAGED"
    assert manifest["publication_base_sha"] == base_sha
    assert manifest["verification_contract"]["release_id"] == "2026-07-31"
    for artifact in manifest['artifacts']:
        for member in artifact['inputs']:
            if member['source_id'] in ('cms.snf_all_owners', 'cms.snf_enrollments'):
                assert member['cms_release_vintage'] == '2026-08'
                assert member['snapshot_date'] == '2026-07-31'
                assert member['cms_source_sha256'] == member['sha256']
    roles = {row["destination_id"] for row in manifest["artifacts"]}
    assert "ownership_bridge_lookup" in roles
    cache = Path(manifest["stage_artifact_cache"])
    assert (cache / "ownership" / "ownership_release_policy.json").is_file()


def test_sff_stage_rebuilds_search_index_from_baseline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import active_release_registry as arr
    import pbj320_stage_sff as sff_mod
    import release_control_plane as rcp

    dev, _, base_sha = _init_git_with_remote(tmp_path)
    control = tmp_path / "control"
    sff_dir = tmp_path / "sff" / "2026-08"
    sff_dir.mkdir(parents=True)
    pdf = sff_dir / "cms_sff_posting_2026-08.pdf"
    pdf.write_bytes(b"%PDF-1.4 sff")
    tables = {}
    for letter, cat in zip("abcd", ("SFF", "Graduate", "Terminated", "Candidate")):
        path = sff_dir / f"sff_table_{letter}.csv"
        path.write_text(
            "provider_number,facility_name,category\n015009,Test," + cat + "\n",
            encoding="utf-8",
        )
        from active_release_registry import sha256_file

        tables[f"sff_table_{letter}.csv"] = sha256_file(path)
    from active_release_registry import sha256_file

    handoff = [
        {
            "filename": name,
            "hash": digest,
            "role": name,
            "source_uri": (sff_dir / name).as_uri(),
        }
        for name, digest in tables.items()
    ]
    active = {
        "schema_version": 1,
        "datasets": {
            "cms.sff_pdf_list": {
                "dataset_id": "cms.sff_pdf_list",
                "active_release_id": "2026-08",
                "status": "ACTIVE",
                "hash": "abc",
                "metadata": {
                    "source_pdf_uri": pdf.as_uri(),
                    "source_pdf_hash": sha256_file(pdf),
                    "pbj_handoff": handoff,
                    "posting_period": "2026-08",
                },
            }
        },
    }
    reg = control / "state" / "active_releases.json"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps(active), encoding="utf-8")

    _mock_baseline(monkeypatch, base_sha)
    monkeypatch.setattr("pbj320_stage_sff.registry_path", lambda _root=None: reg)
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.delenv("PBJ_ACTIVE_RELEASE_REGISTRY", raising=False)
    monkeypatch.setenv("PBJ_ROOT", str(dev))
    monkeypatch.setenv("PBJ_PUBLISH_BRANCH", "master")

    search_runs: list[str] = []

    def _fake_build(_wt: Path) -> dict:
        out = _wt / "data" / "derived" / "sff" / "sff_facilities.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"facilities": [{"provider_number": "015009", "category": "SFF"}]}), encoding="utf-8")
        return {"command": "fake-build", "exit_code": 0, "passed": True, "summary": "ok"}

    def _fake_search(wt: Path) -> tuple[str, dict]:
        out = wt / "search_index.json"
        out.write_text(json.dumps({"f": [{"c": "015009", "n": "Test", "s": "S"}]}), encoding="utf-8")
        from active_release_registry import sha256_file

        search_runs.append(sha256_file(out))
        return sha256_file(out), {"command": "search", "exit_code": 0, "passed": True, "summary": "ok"}

    monkeypatch.setattr(sff_mod, "_run_build_sff_dataset", _fake_build)
    monkeypatch.setattr(sff_mod, "_run_search_index", _fake_search)

    dirty = dev / "data" / "derived" / "sff" / "sff_facilities.json"
    dirty.parent.mkdir(parents=True, exist_ok=True)
    dirty.write_text(json.dumps({"facilities": [{"provider_number": "999999", "category": "SFF"}]}), encoding="utf-8")

    result = sff_mod.stage_sff_for_pbj320(release_id="2026-08", root=control)
    assert result["status"] == "STAGED"
    search_row = next(row for row in result["manifest"]["artifacts"] if row["destination_id"] == "search_index")
    assert search_row["publication_class"] == "shared_derived"
    assert search_runs
    assert search_runs[0] not in {None, sha256_file(dirty)}


def test_pbj_nurse_stage_classifies_git_vs_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import active_release_registry as arr
    import pbj320_stage_pbj_nurse as nurse_mod
    import release_control_plane as rcp

    dev, _, base_sha = _init_git_with_remote(tmp_path)
    for name in (
        "facility_quarterly_metrics.csv",
        "state_quarterly_metrics.csv",
        "national_quarterly_metrics.csv",
        "latest_quarter_data.json",
    ):
        if name.endswith(".json"):
            (dev / name).write_text(json.dumps({"quarter": "CY2026Q1"}), encoding="utf-8")
        else:
            (dev / name).write_text("PROVNUM,CY_Qtr\n335513,CY2026Q1\n", encoding="utf-8")

    control = tmp_path / "control"
    active = {
        "schema_version": 1,
        "datasets": {
            "cms.pbj_nurse_staffing": {
                "dataset_id": "cms.pbj_nurse_staffing",
                "active_release_id": "CY2026Q1",
                "status": "ACTIVE",
                "hash": "nursehash",
            }
        },
    }
    reg = control / "state" / "active_releases.json"
    reg.parent.mkdir(parents=True, exist_ok=True)
    reg.write_text(json.dumps(active), encoding="utf-8")

    _mock_baseline(monkeypatch, base_sha)
    monkeypatch.setattr("pbj320_stage_pbj_nurse.registry_path", lambda _root=None: reg)
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.delenv("PBJ_ACTIVE_RELEASE_REGISTRY", raising=False)
    monkeypatch.setenv("PBJ_ROOT", str(dev))
    monkeypatch.setenv("PBJ_PUBLISH_BRANCH", "master")

    def _fake_agg(wt: Path) -> tuple[str, dict]:
        out = wt / "data" / "state_page_aggregates.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"agg")
        from active_release_registry import sha256_file

        return sha256_file(out), {"command": "agg", "exit_code": 0, "passed": True, "summary": "ok"}

    monkeypatch.setattr(nurse_mod, "_run_state_aggregates", _fake_agg)

    result = nurse_mod.stage_pbj_nurse_for_pbj320(release_id="CY2026Q1", root=control)
    assert result["status"] == "STAGED"
    by_role = {row["destination_id"]: row for row in result["manifest"]["artifacts"] if row.get("path")}
    assert by_role["facility_quarterly_metrics"]["publication_class"] == "commit_destination"
    assert by_role["state_page_aggregates"]["publication_class"] == "shared_derived"
    deploy = next(row for row in result["manifest"]["artifacts"] if row.get("destination_id") == "provider_indexes")
    assert deploy["publication_class"] == "deploy_generated"


def test_macpac_reference_current_not_stale() -> None:
    from pbj320_stage_macpac import macpac_reference_contract

    contract = macpac_reference_contract()
    if contract.get("reference_paths"):
        assert contract["status"] == "REFERENCE_CURRENT"
        assert contract["publishable"] is False
        assert "March 2022" in contract["reference_vintage"]
        assert contract["freshness_semantics"]["macpac_reference"].startswith("REFERENCE_CURRENT")


def test_stage_adapters_never_fake_staged_on_hard_failure(tmp_path: Path) -> None:
    from pbj320_stage_adapters import NOT_READY_STATUS, stage_sff_for_pbj320

    result = stage_sff_for_pbj320(release_id="2099-01", root=tmp_path / "missing")
    assert result["status"] == NOT_READY_STATUS
    assert result["artifacts"] == []
    assert result["publishable"] is False


def test_premium_only_no_publish_spec() -> None:
    from pbj320_source_adapters import is_premium_only_source, stage_publish_spec

    assert stage_publish_spec("cms.health_citations") is None
    assert is_premium_only_source("cms.pbj_non_nurse_staffing") is True
