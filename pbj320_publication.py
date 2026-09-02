"""Generic PBJ320 publication: isolated origin/master worktree, selective commit, fast-forward push."""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cms_data_paths
from active_release_registry import sha256_file


class PublicationError(RuntimeError):
    pass


@dataclass
class PublicationAdapter:
    """Source-specific inputs for generic publish."""

    source_id: str
    release_id: str
    manifest: dict[str, Any]
    dev_pbj_root: Path
    commit_paths: list[str]
    path_hashes: dict[str, str]
    validation_only_paths: set[str]
    validation_commands: list[str]
    commit_message: str
    stage_manifest_sha256: str
    publication_record_path: Path
    write_publication_record: Callable[[dict[str, Any]], None]
    load_publication_record: Callable[[], dict[str, Any] | None]
    already_published: bool = False


def resolve_publish_branch(dev_pbj_root: Path) -> tuple[str, str]:
    remote = (os.environ.get("PBJ_PUBLISH_REMOTE") or "origin").strip() or "origin"
    env_branch = (os.environ.get("PBJ_PUBLISH_BRANCH") or "").strip()
    if env_branch:
        return env_branch, remote
    proc = subprocess.run(
        ["git", "-C", str(dev_pbj_root), "symbolic-ref", "--short", f"refs/remotes/{remote}/HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        ref = (proc.stdout or "").strip()
        if ref.startswith(f"{remote}/"):
            return ref.split("/", 1)[1], remote
    raise PublicationError(
        f"cannot determine publish branch from {remote}/HEAD (set PBJ_PUBLISH_BRANCH)"
    )


def publication_worktree_path(source_id: str, release_id: str, *, root: Path | None = None) -> Path:
    from release_control_plane import control_plane_root

    control = control_plane_root(root)
    safe_source = source_id.replace("/", "_")
    return control / "state" / "pbj320_publication_worktrees" / safe_source / release_id


def _git_run(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise PublicationError(tail[-1] if tail else f"git {' '.join(args)} failed")
    return proc


def fetch_publish_base(dev_pbj_root: Path, *, remote: str, branch: str) -> str:
    _git_run(dev_pbj_root, "fetch", remote, branch)
    return _git_run(dev_pbj_root, "rev-parse", f"{remote}/{branch}").stdout.strip()


def _git_index_paths(repo: Path) -> set[str]:
    proc = _git_run(repo, "diff", "--cached", "--name-only", check=False)
    if proc.returncode != 0:
        raise PublicationError("git diff --cached failed")
    return {line.strip().replace("\\", "/") for line in (proc.stdout or "").splitlines() if line.strip()}


def stage_baseline_worktree_path(source_id: str, release_id: str, *, root: Path | None = None) -> Path:
    from release_control_plane import control_plane_root

    control = control_plane_root(root)
    safe_source = source_id.replace("/", "_")
    return control / "state" / "pbj320_stage_baselines" / safe_source / release_id


def stage_artifact_cache_path(source_id: str, release_id: str, *, root: Path | None = None) -> Path:
    from release_control_plane import control_plane_root

    control = control_plane_root(root)
    safe_source = source_id.replace("/", "_")
    return control / "state" / "pbj320_stage_artifacts" / safe_source / release_id


def _git_show_bytes(dev_pbj_root: Path, ref: str, rel_path: str) -> bytes | None:
    rel_posix = rel_path.replace("\\", "/")
    proc = subprocess.run(
        ["git", "-C", str(dev_pbj_root), "show", f"{ref}:{rel_posix}"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def fingerprint_baseline_path(
    dev_pbj_root: Path,
    publication_base_sha: str,
    rel_path: str,
) -> dict[str, Any]:
    """SHA256 of a tracked path at publication_base_sha (or absent)."""
    rel_posix = rel_path.replace("\\", "/")
    blob = _git_show_bytes(dev_pbj_root, publication_base_sha, rel_posix)
    present = blob is not None
    sha: str | None = None
    if blob is not None:
        import hashlib

        sha = hashlib.sha256(blob).hexdigest()
    return {
        "path": rel_posix,
        "mode": "PUBLISHED_BASELINE",
        "present_at_base": present,
        "sha256": sha,
    }


def resolve_sff_facilities_path(repo_root: Path) -> str | None:
    """Mirror generate_search_index.load_sff_ccns path precedence."""
    candidates = [
        "data/derived/sff/sff_facilities.json",
        "pbj-wrapped/public/sff-facilities.json",
        "sff-facilities.json",
    ]
    for rel in candidates:
        if (repo_root / rel.replace("/", os.sep)).is_file():
            return rel.replace("\\", "/")
    return None


def resolve_chain_performance_path(repo_root: Path) -> str | None:
    """Mirror generate_search_index.load_chain_performance_facility_count selection."""
    ownership_dir = repo_root / "ownership"
    chain_glob = str(ownership_dir / "Nursing_Home_Chain_Performance_Measures_*.csv")
    chain_paths = sorted(glob.glob(chain_glob), key=os.path.getmtime, reverse=True)
    canonical = repo_root / "2025-11" / "Chain_Performance_20260218.csv"
    candidates = [Path(p) for p in chain_paths] + [canonical, repo_root / "chain_performance.csv"]
    for path in candidates:
        if path.is_file():
            try:
                return str(path.relative_to(repo_root)).replace("\\", "/")
            except ValueError:
                return str(path)
    return None


def prepare_baseline_worktree(
    dev_pbj_root: Path,
    *,
    remote: str,
    branch: str,
    base_sha: str,
    worktree_path: Path,
) -> Path:
    """Alias for stage/publish baseline worktrees at publication_base_sha."""
    return prepare_publication_worktree(
        dev_pbj_root,
        remote=remote,
        branch=branch,
        base_sha=base_sha,
        worktree_path=worktree_path,
    )


def sync_stage_artifacts_to_cache(
    *,
    baseline_wt: Path,
    cache_dir: Path,
    rel_paths: list[str],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for rel in rel_paths:
        rel_posix = rel.replace("\\", "/")
        src = baseline_wt / rel_posix.replace("/", os.sep)
        if not src.is_file():
            continue
        dst = cache_dir / rel_posix.replace("/", os.sep)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def prepare_publication_worktree(
    dev_pbj_root: Path,
    *,
    remote: str,
    branch: str,
    base_sha: str,
    worktree_path: Path,
) -> Path:
    """Detached worktree at base_sha; never changes dev_pbj_root checkout."""
    if worktree_path.exists():
        _git_run(dev_pbj_root, "worktree", "remove", "--force", str(worktree_path), check=False)
        if worktree_path.exists():
            shutil.rmtree(worktree_path, ignore_errors=True)
    worktree_path.parent.mkdir(parents=True, exist_ok=True)
    _git_run(dev_pbj_root, "worktree", "add", "--detach", str(worktree_path), base_sha)
    return worktree_path


def copy_manifest_destinations(
    *,
    source_root: Path,
    worktree_root: Path,
    path_hashes: dict[str, str],
) -> list[str]:
    copied: list[str] = []
    for rel_posix, expected_sha in sorted(path_hashes.items()):
        rel_posix = rel_posix.replace("\\", "/")
        src = source_root / rel_posix.replace("/", os.sep)
        if not src.is_file():
            raise PublicationError(f"staged source missing: {rel_posix}")
        actual = sha256_file(src)
        if actual != expected_sha:
            raise PublicationError(f"sha mismatch before copy: {rel_posix}")
        dst = worktree_root / rel_posix.replace("/", os.sep)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256_file(dst) != expected_sha:
            raise PublicationError(f"copy verification failed: {rel_posix}")
        copied.append(rel_posix)
    return copied


def run_validation_commands(worktree_root: Path, commands: list[str]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(worktree_root),
            capture_output=True,
            text=True,
            check=False,
        )
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        results.append(
            {
                "command": cmd,
                "exit_code": proc.returncode,
                "passed": proc.returncode == 0,
                "summary": tail[-1] if tail else f"exit {proc.returncode}",
            }
        )
        if proc.returncode != 0:
            break
    return results


def publish_from_adapter(
    adapter: PublicationAdapter,
    *,
    confirm: bool = False,
    dry_run: bool = False,
    push: bool = True,
    publish_base_sha: str | None = None,
    artifact_source_root: Path | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Generic publish: worktree at origin/master, copy manifest bytes, commit, FF push."""
    if not confirm and not dry_run:
        raise PublicationError("Publish requires explicit confirmation")

    if adapter.already_published:
        raise PublicationError("release already pushed; refusing duplicate publish")

    dev_root = adapter.dev_pbj_root
    copy_root = artifact_source_root or dev_root
    branch, remote = resolve_publish_branch(dev_root)
    allowlist = set(adapter.commit_paths)

    if publish_base_sha:
        current_base = fetch_publish_base(dev_root, remote=remote, branch=branch)
        if current_base != publish_base_sha:
            raise PublicationError(
                f"{remote}/{branch} advanced since review ({publish_base_sha[:12]}… → {current_base[:12]}…); "
                "refresh Publish review before continuing"
            )
        base_sha = current_base
    else:
        base_sha = fetch_publish_base(dev_root, remote=remote, branch=branch)

    wt_path = publication_worktree_path(adapter.source_id, adapter.release_id, root=root)

    review = {
        "source_id": adapter.source_id,
        "release_id": adapter.release_id,
        "commit_paths": list(adapter.commit_paths),
        "commit_file_count": len(adapter.commit_paths),
        "publish_branch": branch,
        "publish_remote": remote,
        "publish_base_sha": base_sha,
        "publication_worktree": str(wt_path),
        "stage_manifest_sha256": adapter.stage_manifest_sha256,
    }

    if dry_run:
        return {
            "status": "DRY_RUN",
            "release_id": adapter.release_id,
            "commit_paths": adapter.commit_paths,
            "validation_only_excluded": sorted(adapter.validation_only_paths),
            "review": review,
            "would_push": push,
            "publication_worktree": str(wt_path),
            "publish_base_sha": base_sha,
        }

    pre_index = _git_index_paths(dev_root)
    if pre_index:
        raise PublicationError(
            f"primary pbj-root index has staged paths (unrelated to Publish): {sorted(pre_index)}"
        )

    prepare_publication_worktree(
        dev_root, remote=remote, branch=branch, base_sha=base_sha, worktree_path=wt_path
    )

    try:
        copy_manifest_destinations(
            source_root=copy_root,
            worktree_root=wt_path,
            path_hashes=adapter.path_hashes,
        )

        validation_results = run_validation_commands(wt_path, adapter.validation_commands)
        if not all(r.get("passed") for r in validation_results):
            failed = next(r for r in validation_results if not r.get("passed"))
            adapter.write_publication_record(
                {
                    "status": "FAILED",
                    "release_id": adapter.release_id,
                    "validation_results": validation_results,
                    "committed": False,
                    "pushed": False,
                    "publish_base_sha": base_sha,
                    "publication_worktree": str(wt_path),
                }
            )
            raise PublicationError(f"pre-publish validation failed: {failed.get('command')}")

        wt_pre_index = _git_index_paths(wt_path)
        if wt_pre_index:
            raise PublicationError(
                f"publication worktree index not clean before staging: {sorted(wt_pre_index)}"
            )

        for rel in adapter.commit_paths:
            rel_posix = rel.replace("\\", "/")
            if rel_posix in adapter.validation_only_paths:
                raise PublicationError(f"refusing to stage validation-only path: {rel_posix}")
            _git_run(wt_path, "add", "--", rel_posix)

        post_index = _git_index_paths(wt_path)
        if post_index != allowlist:
            _git_run(wt_path, "reset", check=False)
            raise PublicationError(
                f"git index allowlist mismatch: expected {sorted(allowlist)} got {sorted(post_index)}"
            )

        commit_proc = _git_run(wt_path, "commit", "-m", adapter.commit_message, check=False)
        if commit_proc.returncode != 0:
            _git_run(wt_path, "reset", check=False)
            adapter.write_publication_record(
                {
                    "status": "FAILED",
                    "release_id": adapter.release_id,
                    "validation_results": validation_results,
                    "committed": False,
                    "pushed": False,
                    "publish_base_sha": base_sha,
                    "publication_worktree": str(wt_path),
                    "error": (commit_proc.stderr or commit_proc.stdout or "").strip(),
                }
            )
            raise PublicationError("git commit failed in publication worktree")

        commit_sha = _git_run(wt_path, "rev-parse", "HEAD").stdout.strip()
        push_succeeded = False
        push_error: str | None = None
        push_timestamp: str | None = None

        if push:
            push_proc = subprocess.run(
                ["git", "-C", str(wt_path), "push", remote, f"HEAD:{branch}"],
                capture_output=True,
                text=True,
                check=False,
            )
            push_timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            if push_proc.returncode != 0:
                push_error = (push_proc.stderr or push_proc.stdout or "").strip()
            else:
                push_succeeded = True

        record = {
            "schema_version": 1,
            "source_id": adapter.source_id,
            "release_id": adapter.release_id,
            "status": "PUSHED" if push_succeeded else "COMMITTED",
            "stage_manifest_sha256": adapter.stage_manifest_sha256,
            "publish_base_sha": base_sha,
            "commit_sha": commit_sha,
            "commit_message": adapter.commit_message,
            "committed_paths": list(adapter.commit_paths),
            "committed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "push_succeeded": push_succeeded,
            "push_remote": remote,
            "push_branch": branch,
            "push_timestamp": push_timestamp,
            "push_error": push_error,
            "publication_worktree": str(wt_path),
            "dev_pbj_root": str(dev_root),
            "validation_results": validation_results,
            "destination_layers": {
                "canonical": "CURRENT",
                "pbj320_destination": "STAGED",
                "committed": "YES",
                "pushed": "YES" if push_succeeded else "NO",
                "deployed": "UNKNOWN",
                "production_verified": "NO",
            },
        }
        adapter.write_publication_record(record)

        if push and not push_succeeded:
            raise PublicationError(f"git push failed (non-fast-forward or remote rejected): {push_error}")

        return {
            "status": "PUSHED" if push_succeeded else "COMMITTED",
            "release_id": adapter.release_id,
            "commit_sha": commit_sha,
            "committed_paths": adapter.commit_paths,
            "push_succeeded": push_succeeded,
            "publication_record": record,
            "review": review,
            "publish_base_sha": base_sha,
            "publication_worktree": str(wt_path),
        }
    finally:
        pass  # worktree retained for audit; removed on next publish
