"""Tests for PBJ320 Provider Information Publish (isolated worktree, selective commit)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test_pbj320_stage_provider_info import _write_pi_release  # noqa: E402


def _init_git_with_remote(tmp_path: Path) -> tuple[Path, Path, str]:
    """Dev pbj-root + bare origin remote; returns dev_root, bare_path, base_sha."""
    bare = tmp_path / "origin.git"
    dev = tmp_path / "dev-pbj-root"
    dev.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "init"], cwd=dev, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=dev, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=dev, check=True, capture_output=True)
    (dev / "README.md").write_text("seed\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=dev, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=dev, check=True, capture_output=True)
    subprocess.run(["git", "branch", "-M", "master"], cwd=dev, check=True, capture_output=True)
    subprocess.run(["git", "remote", "add", "origin", str(bare)], cwd=dev, check=True, capture_output=True)
    subprocess.run(["git", "push", "-u", "origin", "master"], cwd=dev, check=True, capture_output=True)
    base_sha = subprocess.run(
        ["git", "rev-parse", "origin/master"], cwd=dev, capture_output=True, text=True, check=True
    ).stdout.strip()
    return dev, bare, base_sha


def _stage_pi_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, dict[str, Any], str]:
    import pbj320_publication as pub_mod

    import active_release_registry as arr
    import pbj320_stage_provider_info as stage_mod
    import release_control_plane as rcp

    pbjapp, _pbj_root_unused, control = _write_pi_release(tmp_path)
    dev_root, _bare, base_sha = _init_git_with_remote(tmp_path)
    monkeypatch.setattr(pub_mod, "fetch_publish_base", lambda *_a, **_k: base_sha)
    monkeypatch.setattr(pub_mod, "resolve_publish_branch", lambda *_a: ("master", "origin"))

    def _fake_prepare(dev: Path, *, remote: str, branch: str, base_sha: str, worktree_path: Path) -> Path:
        if worktree_path.exists():
            import shutil

            shutil.rmtree(worktree_path, ignore_errors=True)
        worktree_path.mkdir(parents=True, exist_ok=True)
        (worktree_path / "README.md").write_text("seed\n", encoding="utf-8")
        return worktree_path

    monkeypatch.setattr(pub_mod, "prepare_baseline_worktree", _fake_prepare)

    def _fake_fp(dev: Path, base: str, rel: str) -> dict:
        return {"path": rel, "mode": "PUBLISHED_BASELINE", "present_at_base": False, "sha256": None}

    monkeypatch.setattr(pub_mod, "fingerprint_baseline_path", _fake_fp)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)
    monkeypatch.setenv("PBJ_REPO_ROOT", str(pbjapp))
    monkeypatch.setenv("PBJ_PUBLISH_BRANCH", "master")

    def _fake_gates(_pbj_root: Path, *, release_id: str) -> list[dict]:
        return [{"command": "fake-gate", "exit_code": 0, "passed": True, "summary": "ok"}]

    def _fake_agg(_pbj_root: Path) -> tuple[str, dict]:
        out = _pbj_root / "data" / "state_page_aggregates.json.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"aggregates")
        from active_release_registry import sha256_file

        return sha256_file(out), {"command": "agg", "exit_code": 0, "passed": True, "summary": "ok"}

    def _fake_search(_pbj_root: Path) -> tuple[str, dict]:
        out = _pbj_root / "search_index.json"
        out.write_text('{"f":[{"c":"015009","n":"TEST","s":"AL"}]}', encoding="utf-8")
        from active_release_registry import sha256_file

        return sha256_file(out), {"command": "search", "exit_code": 0, "passed": True, "summary": "ok"}

    def _fake_combined(*, pbjapp_root: Path, pbj_root: Path, release_id: str, combined_rel: str, **_: object) -> str:
        out = pbj_root / combined_rel
        out.write_text("ccn,processing_date,PROVNAME,STATE\n015009,2026-08-01,TEST,AL\n", encoding="utf-8")
        from active_release_registry import sha256_file

        return sha256_file(out)

    monkeypatch.setattr(stage_mod, "_run_pbj_root_gates", _fake_gates)
    monkeypatch.setattr(stage_mod, "_run_state_aggregates", _fake_agg)
    monkeypatch.setattr(stage_mod, "_run_search_index", _fake_search)
    monkeypatch.setattr(stage_mod, "_rebuild_combined_latest", _fake_combined)

    result = stage_mod.stage_provider_info_for_pbj320(root=control, pbj_root=dev_root, pbjapp_root=pbjapp)
    assert result["status"] == "STAGED"
    return pbjapp, dev_root, control, result["manifest"], base_sha


def _pass_validations(monkeypatch: pytest.MonkeyPatch) -> None:
    import pbj320_publication as pub_core

    monkeypatch.setattr(
        pub_core,
        "run_validation_commands",
        lambda _wt, commands: [
            {"command": c, "exit_code": 0, "passed": True, "summary": "ok"} for c in commands
        ],
    )


def test_preflight_includes_search_index_as_staged() -> None:
    from pbj320_publish_provider_info import preflight_pi_publish_completeness

    result = preflight_pi_publish_completeness()
    assert result["pass"] is True
    search = next(r for r in result["artifacts"] if r["artifact"] == "search_index.json")
    assert search["classification"] == "already_staged"


def test_cannot_publish_non_staged_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import active_release_registry as arr
    import pbj320_publish_provider_info as pub_mod
    import release_control_plane as rcp

    pbjapp, pbj_root, control = _write_pi_release(tmp_path)
    dev_root, _, _ = _init_git_with_remote(tmp_path)
    monkeypatch.setattr(arr, "registry_path", lambda _root=None: control / "state" / "active_releases.json")
    monkeypatch.setattr(rcp, "control_plane_root", lambda _root=None: control)

    manifest_path = control / "state" / "pbj320_stages" / "cms.provider_info" / "2026-08.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"status": "FAILED", "active_release_id": "2026-08"}), encoding="utf-8")

    with pytest.raises(pub_mod.ProviderInfoPublishError, match="STAGED"):
        pub_mod.validate_publish_ready(root=control, pbj_root=dev_root, release_id="2026-08")


def test_cannot_publish_if_sha_drifted_after_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pbj320_publish_provider_info as pub_mod
    from pbj320_publication import stage_artifact_cache_path

    _pbjapp, dev_root, control, manifest, _base = _stage_pi_fixture(tmp_path, monkeypatch)
    cache = stage_artifact_cache_path(pub_mod.SOURCE_ID, "2026-08", root=control)
    norm = cache / "provider_info" / "ProviderInfoNorm_2026_08.csv"
    norm.write_text(norm.read_text(encoding="utf-8") + "# drift\n", encoding="utf-8")

    with pytest.raises(pub_mod.ProviderInfoPublishError, match="drift"):
        pub_mod.validate_publish_ready(root=control, pbj_root=dev_root, release_id="2026-08")


def test_cannot_publish_with_unrelated_staged_index_in_dev_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, _manifest, base_sha = _stage_pi_fixture(tmp_path, monkeypatch)
    _pass_validations(monkeypatch)
    decoy = dev_root / "decoy_unrelated.txt"
    decoy.write_text("staged early", encoding="utf-8")
    subprocess.run(["git", "add", "decoy_unrelated.txt"], cwd=dev_root, check=True, capture_output=True)

    with pytest.raises(pub_mod.ProviderInfoPublishError, match="primary pbj-root index has staged"):
        pub_mod.publish_provider_info_for_pbj320(
            root=control,
            pbj_root=dev_root,
            release_id="2026-08",
            confirm=True,
            push=False,
            publish_base_sha=base_sha,
        )


def test_origin_master_advance_blocks_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, manifest, base_sha = _stage_pi_fixture(tmp_path, monkeypatch)
    _pass_validations(monkeypatch)

    pub_mod.validate_publish_ready(root=control, pbj_root=dev_root, release_id="2026-08")

    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "remote advance"],
        cwd=dev_root,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "push", "origin", "master"], cwd=dev_root, check=True, capture_output=True)

    with pytest.raises(pub_mod.ProviderInfoPublishError, match="advanced since Stage"):
        pub_mod.validate_publish_ready(root=control, pbj_root=dev_root, release_id="2026-08")

    with pytest.raises(pub_mod.ProviderInfoPublishError, match="advanced since Stage"):
        pub_mod.publish_provider_info_for_pbj320(
            root=control,
            pbj_root=dev_root,
            release_id="2026-08",
            confirm=True,
            push=False,
            publish_base_sha=base_sha,
        )


def test_failed_validation_does_not_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pbj320_publication as pub_core
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, _manifest, base_sha = _stage_pi_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pub_core,
        "run_validation_commands",
        lambda _wt, commands: [
            {"command": commands[0], "exit_code": 1, "passed": False, "summary": "fail"},
        ],
    )

    with pytest.raises(pub_mod.ProviderInfoPublishError, match="pre-publish validation failed"):
        pub_mod.publish_provider_info_for_pbj320(
            root=control,
            pbj_root=dev_root,
            release_id="2026-08",
            confirm=True,
            push=False,
            publish_base_sha=base_sha,
        )

    pub = pub_mod.load_publication_record("2026-08", root=control)
    assert pub is not None
    assert pub.get("committed") is False
    assert pub.get("pushed") is False


def test_successful_publish_dry_run_no_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, manifest, base_sha = _stage_pi_fixture(tmp_path, monkeypatch)
    remote_head_before = subprocess.run(
        ["git", "rev-parse", "origin/master"], cwd=dev_root, capture_output=True, text=True, check=True
    ).stdout.strip()

    result = pub_mod.publish_provider_info_for_pbj320(
        root=control,
        pbj_root=dev_root,
        release_id="2026-08",
        dry_run=True,
        push=True,
        publish_base_sha=base_sha,
    )
    assert result["status"] == "DRY_RUN"
    assert len(result["commit_paths"]) == 4
    assert "search_index.json" in result["commit_paths"]
    assert "provider_info/NH_ProviderInfo_Aug2026.csv" not in result["commit_paths"]

    remote_head_after = subprocess.run(
        ["git", "rev-parse", "origin/master"], cwd=dev_root, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert remote_head_before == remote_head_after


def test_successful_publish_worktree_commit_and_push(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, manifest, base_sha = _stage_pi_fixture(tmp_path, monkeypatch)
    _pass_validations(monkeypatch)
    decoy = dev_root / "decoy_unrelated.txt"
    decoy.write_text("untouched", encoding="utf-8")
    dev_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=dev_root, capture_output=True, text=True, check=True
    ).stdout.strip()

    result = pub_mod.publish_provider_info_for_pbj320(
        root=control,
        pbj_root=dev_root,
        release_id="2026-08",
        confirm=True,
        push=True,
        publish_base_sha=base_sha,
    )
    assert result["push_succeeded"] is True
    assert result["commit_sha"]
    assert len(result["committed_paths"]) == 4
    assert decoy.read_text(encoding="utf-8") == "untouched"
    assert (
        subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=dev_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        == dev_branch
    )

    pub = pub_mod.load_publication_record("2026-08", root=control)
    assert pub is not None
    assert pub["commit_sha"] == result["commit_sha"]
    assert pub["push_succeeded"] is True
    assert pub.get("publish_base_sha") == base_sha
    assert pub.get("publication_worktree")


def test_failed_push_records_committed_not_pushed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pbj320_publication as pub_core
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, _manifest, base_sha = _stage_pi_fixture(tmp_path, monkeypatch)
    _pass_validations(monkeypatch)

    real_run = subprocess.run

    def _mock_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        if cmd and cmd[0] == "git" and "push" in cmd:
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="push rejected")
        return real_run(cmd, **kwargs)

    monkeypatch.setattr(subprocess, "run", _mock_run)

    with pytest.raises(pub_mod.ProviderInfoPublishError, match="git push failed"):
        pub_mod.publish_provider_info_for_pbj320(
            root=control,
            pbj_root=dev_root,
            release_id="2026-08",
            confirm=True,
            push=True,
            publish_base_sha=base_sha,
        )

    pub = pub_mod.load_publication_record("2026-08", root=control)
    assert pub is not None
    assert pub.get("commit_sha")
    assert pub["push_succeeded"] is False
    assert pub["destination_layers"]["committed"] == "YES"
    assert pub["destination_layers"]["pushed"] == "NO"


def test_publish_requires_explicit_confirm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, _manifest, _base = _stage_pi_fixture(tmp_path, monkeypatch)

    with pytest.raises(pub_mod.ProviderInfoPublishError, match="explicit confirmation"):
        pub_mod.publish_provider_info_for_pbj320(
            root=control, pbj_root=dev_root, release_id="2026-08", confirm=False, push=False
        )


def test_git_index_allowlist_exact_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pbj320_publication as pub_core
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, manifest, base_sha = _stage_pi_fixture(tmp_path, monkeypatch)
    _pass_validations(monkeypatch)

    calls: list[tuple[str, ...]] = []
    original_git_run = pub_core._git_run

    def _spy_git_run(repo: Path, *args: str, check: bool = True):  # type: ignore[no-untyped-def]
        if "worktree" not in args and args and args[0] == "add":
            calls.append(args)
        return original_git_run(repo, *args, check=check)

    monkeypatch.setattr(pub_core, "_git_run", _spy_git_run)

    pub_mod.publish_provider_info_for_pbj320(
        root=control,
        pbj_root=dev_root,
        release_id="2026-08",
        confirm=True,
        push=True,
        publish_base_sha=base_sha,
    )

    add_calls = [c for c in calls if c[0] == "add"]
    assert len(add_calls) == 4
    staged_paths = {c[2] for c in add_calls}
    expected = set(pub_mod.commit_destination_paths(manifest))
    assert staged_paths == expected
    assert "provider_info/NH_ProviderInfo_Aug2026.csv" not in staged_paths


def test_build_publish_review_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import pbj320_publish_provider_info as pub_mod

    _pbjapp, dev_root, control, _manifest, _base = _stage_pi_fixture(tmp_path, monkeypatch)
    ctx = pub_mod.build_publish_review_context(root=control, pbj_root=dev_root, release_id="2026-08")
    assert ctx["release_id"] == "2026-08"
    assert ctx["review"]["commit_file_count"] == 4
    assert ctx["review"]["publish_base_sha"]
