"""Tests for PBJ320 Provider Information staging (local pbj-root handoff)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _mock_baseline_stage_ops(monkeypatch: pytest.MonkeyPatch, base_sha: str = "a" * 64) -> None:
    import pbj320_publication as pub

    monkeypatch.setattr(pub, "fetch_publish_base", lambda *_a, **_k: base_sha)
    monkeypatch.setattr(pub, "resolve_publish_branch", lambda *_a: ("master", "origin"))

    def _fake_prepare(dev: Path, *, remote: str, branch: str, base_sha: str, worktree_path: Path) -> Path:
        if worktree_path.exists():
            import shutil

            shutil.rmtree(worktree_path, ignore_errors=True)
        worktree_path.mkdir(parents=True, exist_ok=True)
        (worktree_path / "README.md").write_text("seed\n", encoding="utf-8")
        return worktree_path

    monkeypatch.setattr(pub, "prepare_baseline_worktree", _fake_prepare)
    monkeypatch.setattr(
        pub,
        "fingerprint_baseline_path",
        lambda _dev, _base, rel: {
            "path": rel,
            "mode": "PUBLISHED_BASELINE",
            "present_at_base": False,
            "sha256": None,
        },
    )


@pytest.fixture(autouse=True)
def _autouse_baseline_stage_ops(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_baseline_stage_ops(monkeypatch)


def _write_pi_release(
    tmp_path: Path,
    *,
    release_id: str = "2026-08",
    norm_content: str | None = None,
    nh_content: str | None = None,
) -> tuple[Path, Path, Path]:
    from active_release_registry import sha256_file

    pbjapp = tmp_path / "pbjapp"
    norm_dir = pbjapp / "provider_info_normalized"
    pi_dir = pbjapp / "provider_info"
    norm_dir.mkdir(parents=True, exist_ok=True)
    pi_dir.mkdir(parents=True, exist_ok=True)
    year, month = release_id.split("-")
    norm_name = f"ProviderInfoNorm_{year}_{month}.csv"
    nh_name = f"NH_ProviderInfo_Aug{year}.csv" if month == "08" else f"NH_ProviderInfo_{month}{year}.csv"
    if month == "08":
        nh_name = f"NH_ProviderInfo_Aug{year}.csv"

    header = "ccn,processing_date,PROVNAME,STATE\n"
    norm_body = norm_content or (header + "015009,2026-08-01,TEST FACILITY,AL\n")
    nh_body = nh_content or norm_body
    norm_path = norm_dir / norm_name
    nh_path = pi_dir / nh_name
    norm_path.write_text(norm_body, encoding="utf-8")
    nh_path.write_text(nh_body, encoding="utf-8")

    manifest_dir = pi_dir / "_manifests" / release_id
    manifest_dir.mkdir(parents=True, exist_ok=True)
    handoff = {
        "release_key": release_id,
        "pbj_root_sync": {
            "destination_file": f"provider_info/{norm_name}",
            "sha256": sha256_file(norm_path),
            "row_count": 1,
        },
        "pbj_root_nh_snapshot_sync": {
            "destination_file": f"provider_info/{nh_name}",
            "sha256": sha256_file(nh_path),
            "row_count": 1,
        },
        "provider_promotion": {"ready_for_pbj_commit": True},
    }
    (manifest_dir / "pbj_root_handoff.json").write_text(json.dumps(handoff), encoding="utf-8")

    state = tmp_path / "state"
    state.mkdir(parents=True, exist_ok=True)
    active = {
        "schema_version": 1,
        "datasets": {
            "cms.provider_info": {
                "dataset_id": "cms.provider_info",
                "active_release_id": release_id,
                "status": "ACTIVE",
                "source_filename": norm_name,
                "source_uri": norm_path.as_uri(),
                "hash": sha256_file(norm_path),
                "validated_at": "2026-08-28T00:00:00+00:00",
                "metadata": {},
            }
        },
    }
    (state / "active_releases.json").write_text(json.dumps(active), encoding="utf-8")
    (state / "release_candidates.json").write_text(
        json.dumps({"schema_version": 1, "datasets": {}}), encoding="utf-8"
    )
    return pbjapp, tmp_path / "pbj-root", tmp_path


def test_audit_before_stage_not_staged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import active_release_registry as arr
    import release_control_plane as rcp

    pbjapp, pbj_root, control = _write_pi_release(tmp_path)
    pbj_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))

    from pbj320_stage_provider_info import audit_provider_info_pbj320_destination

    audit = audit_provider_info_pbj320_destination(root=control, pbj_root=pbj_root, pbjapp_root=pbjapp)
    assert audit["canonical_current"] is True
    assert audit["destination_staged"] is False
    assert audit["production_deployed"] == "UNKNOWN"


def test_stage_writes_manifest_and_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp

    pbjapp, pbj_root, control = _write_pi_release(tmp_path)
    pbj_root.mkdir(parents=True, exist_ok=True)
    norm_rel = pbj_root / "provider_info" / "ProviderInfoNorm_2026_08.csv"
    assert not norm_rel.is_file()
    decoy = pbj_root / "decoy_unrelated.txt"
    decoy.write_text("untouched", encoding="utf-8")
    decoy_mtime = decoy.stat().st_mtime

    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))

    def _fake_gates(_pbj_root: Path, *, release_id: str) -> list[dict]:
        return [
            {"command": "fake-gate", "exit_code": 0, "passed": True, "summary": "ok"},
        ]

    def _fake_agg(_pbj_root: Path) -> tuple[str, dict]:
        out = _pbj_root / "data" / "state_page_aggregates.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"aggregates")
        from active_release_registry import sha256_file

        return sha256_file(out), {
            "command": "python scripts/build_state_page_aggregates.py",
            "exit_code": 0,
            "passed": True,
            "summary": "ok",
        }

    def _fake_search(_pbj_root: Path) -> tuple[str, dict]:
        out = _pbj_root / "search_index.json"
        out.write_text('{"f":[{"c":"015009","n":"TEST","s":"AL"}]}', encoding="utf-8")
        from active_release_registry import sha256_file

        return sha256_file(out), {
            "command": "python generate_search_index.py",
            "exit_code": 0,
            "passed": True,
            "summary": "ok",
        }

    def _fake_combined(*, pbjapp_root: Path, pbj_root: Path, release_id: str, combined_rel: str, **_: object) -> str:
        out = pbj_root / combined_rel
        out.write_text("ccn,processing_date,PROVNAME,STATE\n015009,2026-08-01,TEST,AL\n", encoding="utf-8")
        from active_release_registry import sha256_file

        return sha256_file(out)

    monkeypatch.setattr(stage_mod, "_run_pbj_root_gates", _fake_gates)
    monkeypatch.setattr(stage_mod, "_run_state_aggregates", _fake_agg)
    monkeypatch.setattr(stage_mod, "_run_search_index", _fake_search)
    monkeypatch.setattr(stage_mod, "_rebuild_combined_latest", _fake_combined)

    result = stage_mod.stage_provider_info_for_pbj320(
        root=control, pbj_root=pbj_root, pbjapp_root=pbjapp
    )
    assert result["status"] == "STAGED"
    assert str(result["manifest"]["pbj_root"]) == str(pbj_root.resolve())
    manifest = result["manifest"]
    assert manifest["destination_layers"]["pbj320_destination"] == "STAGED"
    assert manifest["destination_layers"]["committed"] == "NO"
    assert len(manifest["artifacts"]) >= 4
    commit_paths = [
        row["path"]
        for row in manifest["artifacts"]
        if row.get("publication_class") in {"commit_destination", "shared_derived"}
    ]
    assert "search_index.json" in commit_paths

    norm_row = next(row for row in manifest["artifacts"] if row["destination_id"] == "provider_norm")
    assert norm_row["existed_on_disk_before"] is False
    assert norm_row["present_before"] is False
    assert norm_row["material_change"] is True
    assert norm_row["publication_action"] == "add"
    assert norm_row["publication_class"] == "commit_destination"
    assert norm_row["old_sha256"] is None
    assert "provider_info/ProviderInfoNorm_2026_08.csv" in manifest["files_added"]
    nh_row = next((row for row in manifest["artifacts"] if row["destination_id"] == "nh_snapshot_parity"), None)
    if nh_row:
        assert nh_row["publication_class"] == "validation_parity"
        assert nh_row["publication_action"] == "validation-only"
        assert "provider_info/NH_ProviderInfo_Aug2026.csv" not in manifest["files_added"]
        assert "provider_info/NH_ProviderInfo_Aug2026.csv" in manifest.get("validation_artifacts", [])

    norm_dest = pbj_root / "provider_info" / "ProviderInfoNorm_2026_08.csv"
    combined_dest = pbj_root / "provider_info_combined_latest.csv"
    assert norm_dest.is_file()
    assert combined_dest.is_file()

    for row in manifest["artifacts"]:
        on_disk = pbj_root / str(row["path"]).replace("/", "\\" if sys.platform == "win32" else "/")
        if on_disk.is_file() and row.get("proposed_sha256"):
            from active_release_registry import sha256_file

            assert sha256_file(on_disk) == row["proposed_sha256"]

    assert decoy.read_text(encoding="utf-8") == "untouched"
    assert decoy.stat().st_mtime == decoy_mtime

    second = stage_mod.stage_provider_info_for_pbj320(
        root=control, pbj_root=pbj_root, pbjapp_root=pbjapp
    )
    assert second["status"] == "NO_MATERIAL_DIFF"


def test_stage_gate_failure_restores_and_not_staged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp

    pbjapp, pbj_root, control = _write_pi_release(tmp_path)
    pbj_root.mkdir(parents=True, exist_ok=True)
    old_norm = pbj_root / "provider_info" / "ProviderInfoNorm_2026_07.csv"
    old_norm.parent.mkdir(parents=True)
    old_norm.write_text("ccn,processing_date\n015009,2026-07-01\n", encoding="utf-8")
    old_sha = __import__("hashlib").sha256(old_norm.read_bytes()).hexdigest()

    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))

    def _fail_gates(_pbj_root: Path, *, release_id: str) -> list[dict]:
        return [
            {"command": "fake-gate", "exit_code": 1, "passed": False, "summary": "fail"},
        ]

    monkeypatch.setattr(stage_mod, "_run_pbj_root_gates", _fail_gates)

    with pytest.raises(stage_mod.ProviderInfoStageError):
        stage_mod.stage_provider_info_for_pbj320(
            root=control, pbj_root=pbj_root, pbjapp_root=pbjapp
        )

    assert old_norm.is_file()
    from active_release_registry import sha256_file

    assert sha256_file(old_norm) == old_sha
    failed_manifest = json.loads(
        (control / "state" / "pbj320_stages" / "cms.provider_info" / "2026-08.json").read_text(encoding="utf-8")
    )
    assert failed_manifest["status"] == "FAILED"


def test_health_citations_not_in_pi_stage_manifest(tmp_path: Path) -> None:
    from release_source_catalog import BY_ID

    assert BY_ID["cms.health_citations"].upstream == ()


def test_rebuild_combined_latest_pins_pbj_repo_root_for_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subprocess must inherit PBJ_REPO_ROOT=pbjapp even when parent env points at control plane."""
    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp

    pbjapp, pbj_root, control = _write_pi_release(tmp_path)
    pbj_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(control))
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)

    build_script = pbjapp / "scripts" / "build_provider_info_combined.py"
    build_script.parent.mkdir(parents=True)
    build_script.write_text(
        "import os, sys\n"
        "from pathlib import Path\n"
        "root = Path(os.environ['PBJ_REPO_ROOT'])\n"
        "norm_dir = Path(os.environ['PBJ_PROVIDER_INFO_NORMALIZED'])\n"
        "norm = norm_dir / 'ProviderInfoNorm_2026_08.csv'\n"
        "if not norm.is_file():\n"
        "    raise SystemExit(f'missing norm at {norm}')\n"
        "out = Path(sys.argv[sys.argv.index('--output') + 1])\n"
        "out.write_text(norm.read_text(encoding='utf-8'), encoding='utf-8')\n",
        encoding="utf-8",
    )

    captured: dict[str, object] = {}
    import subprocess as subprocess_mod

    real_run = subprocess_mod.run

    def _capture_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["env"] = kwargs.get("env")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(stage_mod.subprocess, "run", _capture_run)

    combined_sha = stage_mod._rebuild_combined_latest(
        pbjapp_root=pbjapp,
        pbj_root=pbj_root,
        release_id="2026-08",
        combined_rel="provider_info_combined_latest.csv",
        canonical_norm_dir=pbjapp / "provider_info_normalized",
    )
    env = captured.get("env") or {}
    assert env.get("PBJ_REPO_ROOT") == str(pbjapp.resolve())
    assert env.get("PBJ_PROVIDER_INFO_NORMALIZED") == str((pbjapp / "provider_info_normalized").resolve())
    combined_dest = pbj_root / "provider_info_combined_latest.csv"
    assert combined_dest.is_file()
    from active_release_registry import sha256_file

    assert sha256_file(combined_dest) == combined_sha


def test_stage_combined_failure_rolls_back_norm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp

    pbjapp, pbj_root, control = _write_pi_release(tmp_path)
    pbj_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(control))

    def _fail_combined(**_kwargs: object) -> str:
        raise stage_mod.ProviderInfoStageError(
            "build_provider_info_combined failed: required normalized month missing"
        )

    monkeypatch.setattr(stage_mod, "_rebuild_combined_latest", _fail_combined)

    with pytest.raises(stage_mod.ProviderInfoStageError):
        stage_mod.stage_provider_info_for_pbj320(
            root=control, pbj_root=pbj_root, pbjapp_root=pbjapp
        )

    assert not (pbj_root / "provider_info" / "ProviderInfoNorm_2026_08.csv").is_file()
    failed_manifest = json.loads(
        (control / "state" / "pbj320_stages" / "cms.provider_info" / "2026-08.json").read_text(
            encoding="utf-8"
        )
    )
    assert failed_manifest["status"] == "FAILED"
    assert failed_manifest["destination_layers"]["pbj320_destination"] == "NOT_STAGED"


def test_artifact_row_does_not_reread_destination_after_add(tmp_path: Path) -> None:
    from pbj320_stage_provider_info import _artifact_row, _destination_pre_state

    pbj_root = tmp_path / "pbj-root"
    rel = "provider_info/ProviderInfoNorm_2026_08.csv"
    pre = _destination_pre_state(pbj_root, rel)
    assert pre["existed_on_disk_before"] is False
    dest = pbj_root / "provider_info" / "ProviderInfoNorm_2026_08.csv"
    dest.parent.mkdir(parents=True)
    dest.write_text("ccn,processing_date\n015009,2026-08-01\n", encoding="utf-8")
    from active_release_registry import sha256_file

    proposed = sha256_file(dest)
    row = _artifact_row(
        rel_path=rel,
        pre_state=pre,
        proposed_sha=proposed,
        expected_release_id="2026-08",
        transformation="test",
        role="provider_norm",
        publication_class="commit_destination",
    )
    assert row["existed_on_disk_before"] is False
    assert row["present_before"] is False
    assert row["old_sha256"] is None
    assert row["material_change"] is True
    assert row["publication_action"] == "add"


def test_stage_manifest_git_untracked_norm_with_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp
    from active_release_registry import sha256_file

    pbjapp, pbj_root, control = _write_pi_release(tmp_path)
    pbj_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(pbj_root), check=True, capture_output=True)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))
    monkeypatch.setattr(stage_mod, "_run_pbj_root_gates", lambda *_a, **_k: [{"command": "x", "passed": True, "summary": "ok"}])

    def _fake_agg(pbj: Path) -> tuple[str, dict]:
        out = pbj / "data" / "state_page_aggregates.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"agg")
        return sha256_file(out), {"command": "agg", "passed": True, "summary": "ok"}

    def _fake_combined(**kwargs: object) -> str:
        pbj = kwargs["pbj_root"]
        combined_rel = kwargs["combined_rel"]
        out = pbj / str(combined_rel)
        out.write_text("ccn,processing_date\n015009,2026-08-01\n", encoding="utf-8")
        return sha256_file(out)

    def _fake_search(pbj: Path) -> tuple[str, dict]:
        out = pbj / "search_index.json"
        out.write_text('{"f":[]}', encoding="utf-8")
        return sha256_file(out), {"command": "search", "passed": True, "summary": "ok"}

    monkeypatch.setattr(stage_mod, "_run_state_aggregates", _fake_agg)
    monkeypatch.setattr(stage_mod, "_run_search_index", _fake_search)
    monkeypatch.setattr(stage_mod, "_rebuild_combined_latest", _fake_combined)

    result = stage_mod.stage_provider_info_for_pbj320(root=control, pbj_root=pbj_root, pbjapp_root=pbjapp)
    norm_row = next(row for row in result["manifest"]["artifacts"] if row["destination_id"] == "provider_norm")
    assert norm_row["git_state_before"] == "absent"
    assert norm_row["publication_action"] == "add"


def test_refresh_stage_manifest_rewrites_without_mutating_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp
    from active_release_registry import sha256_file

    pbjapp, pbj_root, control = _write_pi_release(tmp_path)
    pbj_root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=str(pbj_root), check=True, capture_output=True)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))
    monkeypatch.setattr(stage_mod, "_run_pbj_root_gates", lambda *_a, **_k: [{"command": "x", "passed": True, "summary": "ok"}])

    def _fake_agg(pbj: Path) -> tuple[str, dict]:
        out = pbj / "data" / "state_page_aggregates.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"agg")
        return sha256_file(out), {"command": "agg", "passed": True, "summary": "ok"}

    def _fake_combined(**kwargs: object) -> str:
        pbj = kwargs["pbj_root"]
        combined_rel = kwargs["combined_rel"]
        out = pbj / str(combined_rel)
        out.write_text("ccn,processing_date\n015009,2026-08-01\n", encoding="utf-8")
        return sha256_file(out)

    def _fake_search(pbj: Path) -> tuple[str, dict]:
        out = pbj / "search_index.json"
        out.write_text('{"f":[]}', encoding="utf-8")
        return sha256_file(out), {"command": "search", "passed": True, "summary": "ok"}

    monkeypatch.setattr(stage_mod, "_run_state_aggregates", _fake_agg)
    monkeypatch.setattr(stage_mod, "_run_search_index", _fake_search)
    monkeypatch.setattr(stage_mod, "_rebuild_combined_latest", _fake_combined)

    stage_mod.stage_provider_info_for_pbj320(root=control, pbj_root=pbj_root, pbjapp_root=pbjapp)
    norm_path = pbj_root / "provider_info" / "ProviderInfoNorm_2026_08.csv"
    before_bytes = norm_path.read_bytes()
    before_mtime = norm_path.stat().st_mtime

    refreshed = stage_mod.refresh_provider_info_stage_manifest(root=control, pbj_root=pbj_root, pbjapp_root=pbjapp)
    assert refreshed["status"] == "MANIFEST_REFRESHED"
    norm_row = next(row for row in refreshed["manifest"]["artifacts"] if row["destination_id"] == "provider_norm")
    assert norm_row["git_state_before"] == "absent"
    assert norm_row["publication_action"] == "add"
    assert norm_row["existed_on_disk_before"] is False
    assert norm_row.get("publication_base_sha")
    assert norm_path.read_bytes() == before_bytes
    assert norm_path.stat().st_mtime == before_mtime
    assert "provider_info/ProviderInfoNorm_2026_08.csv" in refreshed["manifest"]["files_added"]
    assert "provider_info/NH_ProviderInfo_Aug2026.csv" in refreshed["manifest"]["validation_artifacts"]


def test_refresh_preserves_original_stage_timestamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp
    from active_release_registry import sha256_file

    pbjapp, pbj_root, control = _write_pi_release(tmp_path)
    pbj_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))
    monkeypatch.setattr(stage_mod, "_run_pbj_root_gates", lambda *_a, **_k: [{"command": "x", "passed": True, "summary": "ok"}])

    gate_ran_at = "2026-08-28T20:42:30+00:00"

    def _fake_agg(pbj: Path) -> tuple[str, dict]:
        out = pbj / "data" / "state_page_aggregates.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"agg")
        return sha256_file(out), {
            "command": "agg",
            "passed": True,
            "summary": "ok",
            "executed_at": gate_ran_at,
        }

    monkeypatch.setattr(
        stage_mod,
        "_run_state_aggregates",
        _fake_agg,
    )
    monkeypatch.setattr(
        stage_mod,
        "_rebuild_combined_latest",
        lambda **kwargs: (
            kwargs["pbj_root"].joinpath(str(kwargs["combined_rel"])).write_text(
                "ccn,processing_date\n015009,2026-08-01\n", encoding="utf-8"
            )
            or sha256_file(kwargs["pbj_root"] / str(kwargs["combined_rel"]))
        ),
    )
    monkeypatch.setattr(
        stage_mod,
        "_run_search_index",
        lambda pbj: (
            (pbj / "search_index.json").write_text('{"f":[]}', encoding="utf-8")
            or sha256_file(pbj / "search_index.json"),
            {"command": "search", "passed": True, "summary": "ok"},
        ),
    )

    staged = stage_mod.stage_provider_info_for_pbj320(root=control, pbj_root=pbj_root, pbjapp_root=pbjapp)
    original_stage_ts = staged["manifest"]["stage_timestamp"]
    original_gates = list(staged["manifest"]["validation_gates"])

    refreshed = stage_mod.refresh_provider_info_stage_manifest(root=control, pbj_root=pbj_root, pbjapp_root=pbjapp)
    manifest = refreshed["manifest"]
    assert manifest["stage_timestamp"] == original_stage_ts
    assert manifest["manifest_refreshed"] is True
    assert manifest.get("manifest_refreshed_at")
    assert manifest["validation_gates"] == original_gates
    agg_gate = next(g for g in manifest["validation_gates"] if g.get("command") == "agg")
    assert agg_gate.get("executed_at") == gate_ran_at
