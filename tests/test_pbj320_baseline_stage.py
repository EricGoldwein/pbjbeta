"""Baseline+overlay Stage isolation tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_pbj320_publish_provider_info import _init_git_with_remote, _stage_pi_fixture  # noqa: E402


def _mock_baseline_stage(monkeypatch: pytest.MonkeyPatch, base_sha: str) -> None:
    import pbj320_publication as pub

    monkeypatch.setattr(pub, "fetch_publish_base", lambda *_a, **_k: base_sha)
    monkeypatch.setattr(pub, "resolve_publish_branch", lambda *_a: ("master", "origin"))

    def _fake_prepare(dev: Path, *, remote: str, branch: str, base_sha: str, worktree_path: Path) -> Path:
        if worktree_path.exists():
            import shutil

            shutil.rmtree(worktree_path, ignore_errors=True)
        worktree_path.mkdir(parents=True, exist_ok=True)
        (worktree_path / "README.md").write_text("seed\n", encoding="utf-8")
        sff_dir = worktree_path / "data" / "derived" / "sff"
        sff_dir.mkdir(parents=True, exist_ok=True)
        (sff_dir / "sff_facilities.json").write_text(
            json.dumps({"facilities": [{"provider_number": "015009", "category": "SFF"}]}),
            encoding="utf-8",
        )
        return worktree_path

    monkeypatch.setattr(pub, "prepare_baseline_worktree", _fake_prepare)

    def _fake_fp(dev: Path, base: str, rel: str) -> dict:
        wt_sff = dev / "data" / "derived" / "sff" / "sff_facilities.json"
        if rel == "data/derived/sff/sff_facilities.json" and wt_sff.is_file():
            from active_release_registry import sha256_file

            return {
                "path": rel,
                "mode": "PUBLISHED_BASELINE",
                "present_at_base": True,
                "sha256": sha256_file(wt_sff),
            }
        return {"path": rel, "mode": "PUBLISHED_BASELINE", "present_at_base": False, "sha256": None}

    monkeypatch.setattr(pub, "fingerprint_baseline_path", _fake_fp)


def test_stage_records_publication_base_sha_and_input_fingerprints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pbjapp, dev_root, control, manifest, base_sha = _stage_pi_fixture(tmp_path, monkeypatch)
    assert manifest.get("publication_base_sha") == base_sha
    search_row = next(row for row in manifest["artifacts"] if row["destination_id"] == "search_index")
    assert search_row["publication_class"] == "shared_derived"
    assert search_row.get("inputs")
    assert search_row.get("baseline_sha256") is not None or search_row.get("baseline_existed") is False
    agg_row = next(row for row in manifest["artifacts"] if row["destination_id"] == "state_page_aggregates")
    assert agg_row["publication_class"] == "shared_derived"
    nurse_inputs = [i for i in agg_row["inputs"] if i.get("source_id") == "cms.pbj_nurse_staffing"]
    assert nurse_inputs


def test_dirty_dev_sff_does_not_change_staged_search_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp
    from test_pbj320_stage_provider_info import _write_pi_release

    base = tmp_path
    dev, _, base_sha = _init_git_with_remote(base)
    _mock_baseline_stage(monkeypatch, base_sha)

    pbjapp, _, control = _write_pi_release(tmp_path)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))
    monkeypatch.setenv("PBJ_PUBLISH_BRANCH", "master")

    search_runs: list[str] = []

    def _fake_gates(_pbj_root: Path, *, release_id: str) -> list[dict]:
        return [{"command": "fake-gate", "exit_code": 0, "passed": True, "summary": "ok"}]

    def _fake_agg(_pbj_root: Path) -> tuple[str, dict]:
        out = _pbj_root / "data" / "state_page_aggregates.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"aggregates-v1")
        from active_release_registry import sha256_file

        return sha256_file(out), {"command": "agg", "exit_code": 0, "passed": True, "summary": "ok"}

    def _fake_search(pbj_root: Path) -> tuple[str, dict]:
        search_runs.append(str(pbj_root.resolve()))
        sff = pbj_root / "data" / "derived" / "sff" / "sff_facilities.json"
        tag = "dirty" if "DIRTY_SFF_MARKER" in sff.read_text(encoding="utf-8") else "baseline"
        out = pbj_root / "search_index.json"
        out.write_text(json.dumps({"f": [], "sff_tag": tag}), encoding="utf-8")
        from active_release_registry import sha256_file

        return sha256_file(out), {"command": "search", "exit_code": 0, "passed": True, "summary": tag}

    def _fake_combined(*, pbjapp_root: Path, pbj_root: Path, release_id: str, combined_rel: str, **_: object) -> str:
        out = pbj_root / combined_rel
        out.write_text("ccn,processing_date,PROVNAME,STATE\n015009,2026-08-01,TEST,AL\n", encoding="utf-8")
        from active_release_registry import sha256_file

        return sha256_file(out)

    monkeypatch.setattr(stage_mod, "_run_pbj_root_gates", _fake_gates)
    monkeypatch.setattr(stage_mod, "_run_state_aggregates", _fake_agg)
    monkeypatch.setattr(stage_mod, "_run_search_index", _fake_search)
    monkeypatch.setattr(stage_mod, "_rebuild_combined_latest", _fake_combined)

    result1 = stage_mod.stage_provider_info_for_pbj320(root=control, pbj_root=dev, pbjapp_root=pbjapp)
    cache1 = Path(result1["manifest"]["stage_artifact_cache"]) / "search_index.json"
    tag1 = json.loads(cache1.read_text(encoding="utf-8"))["sff_tag"]

    dirty_sff = dev / "data" / "derived" / "sff" / "sff_facilities.json"
    dirty_sff.parent.mkdir(parents=True, exist_ok=True)
    dirty_sff.write_text('{"facilities":[],"DIRTY_SFF_MARKER":true}', encoding="utf-8")

    result2 = stage_mod.stage_provider_info_for_pbj320(
        root=control, pbj_root=dev, pbjapp_root=pbjapp, force=True
    )
    cache2 = Path(result2["manifest"]["stage_artifact_cache"]) / "search_index.json"
    tag2 = json.loads(cache2.read_text(encoding="utf-8"))["sff_tag"]

    assert tag1 == "baseline"
    assert tag2 == "baseline"
    assert all("stage_baselines" in p or "pbj320_stage_baselines" in p for p in search_runs)


def test_dirty_dev_staffing_does_not_change_staged_aggregates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp
    from test_pbj320_stage_provider_info import _write_pi_release

    base = tmp_path
    dev, _, base_sha = _init_git_with_remote(base)
    _mock_baseline_stage(monkeypatch, base_sha)

    pbjapp, _, control = _write_pi_release(tmp_path)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))

    agg_runs: list[str] = []

    def _fake_gates(_pbj_root: Path, *, release_id: str) -> list[dict]:
        return [{"command": "fake-gate", "exit_code": 0, "passed": True, "summary": "ok"}]

    def _fake_agg(pbj_root: Path) -> tuple[str, dict]:
        agg_runs.append(str(pbj_root.resolve()))
        fq = pbj_root / "facility_quarterly_metrics.csv"
        tag = "dirty" if fq.is_file() and b"DIRTY_STAFF" in fq.read_bytes() else "baseline"
        out = pbj_root / "data" / "state_page_aggregates.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(f"aggregates-{tag}".encode())
        from active_release_registry import sha256_file

        return sha256_file(out), {"command": "agg", "exit_code": 0, "passed": True, "summary": tag}

    def _fake_search(pbj_root: Path) -> tuple[str, dict]:
        out = pbj_root / "search_index.json"
        out.write_text('{"f":[]}', encoding="utf-8")
        from active_release_registry import sha256_file

        return sha256_file(out), {"command": "search", "exit_code": 0, "passed": True, "summary": "ok"}

    def _fake_combined(*, pbjapp_root: Path, pbj_root: Path, release_id: str, combined_rel: str, **_: object) -> str:
        out = pbj_root / combined_rel
        out.write_text("ccn,processing_date\n015009,2026-08-01\n", encoding="utf-8")
        from active_release_registry import sha256_file

        return sha256_file(out)

    monkeypatch.setattr(stage_mod, "_run_pbj_root_gates", _fake_gates)
    monkeypatch.setattr(stage_mod, "_run_state_aggregates", _fake_agg)
    monkeypatch.setattr(stage_mod, "_run_search_index", _fake_search)
    monkeypatch.setattr(stage_mod, "_rebuild_combined_latest", _fake_combined)

    r1 = stage_mod.stage_provider_info_for_pbj320(root=control, pbj_root=dev, pbjapp_root=pbjapp)
    t1 = (Path(r1["manifest"]["stage_artifact_cache"]) / "data/state_page_aggregates.json.gz").read_bytes()

    (dev / "facility_quarterly_metrics.csv").write_bytes(b"DIRTY_STAFF")
    r2 = stage_mod.stage_provider_info_for_pbj320(root=control, pbj_root=dev, pbjapp_root=pbjapp, force=True)
    t2 = (Path(r2["manifest"]["stage_artifact_cache"]) / "data/state_page_aggregates.json.gz").read_bytes()

    assert b"baseline" in t1
    assert b"baseline" in t2
    assert all("stage_baselines" in p or "pbj320_stage_baselines" in p for p in agg_runs)


def test_publish_blocks_when_origin_advanced_since_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, _manifest, base_sha = _stage_pi_fixture(tmp_path, monkeypatch)

    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "remote advance"],
        cwd=dev_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "push", "origin", "master"], cwd=dev_root, check=True, capture_output=True)

    with pytest.raises(pub_mod.ProviderInfoPublishError, match="advanced since Stage"):
        pub_mod.validate_publish_ready(root=control, pbj_root=dev_root, release_id="2026-08")
