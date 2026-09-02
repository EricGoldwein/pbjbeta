"""Shared baseline+overlay Stage helpers for non-PI PBJ320 source adapters."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cms_data_paths
from active_release_registry import sha256_file
from pbj320_publication_contract import PUBLICATION_CONTRACT_VERSION, STAGE_STATUS_STAGED


class StageError(RuntimeError):
    pass


SEARCH_INDEX_REL = "search_index.json"


def control_root(root: Path | None = None) -> Path:
    from release_control_plane import control_plane_root

    return control_plane_root(root)


def stage_manifest_dir(source_id: str, *, root: Path | None = None) -> Path:
    return control_root(root) / "state" / "pbj320_stages" / source_id


def stage_manifest_path(source_id: str, release_id: str, *, root: Path | None = None) -> Path:
    return stage_manifest_dir(source_id, root=root) / f"{release_id}.json"


def load_stage_manifest(source_id: str, release_id: str, *, root: Path | None = None) -> dict[str, Any] | None:
    path = stage_manifest_path(source_id, release_id, root=root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def resolve_pbj_root(explicit: Path | str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        env = (os.environ.get("PBJ_ROOT") or "").strip()
        if env:
            path = Path(env).expanduser().resolve()
        else:
            path = (cms_data_paths.repo_root().parent / "pbj-root").resolve()
    if not path.is_dir():
        raise StageError(f"pbj-root not found: {path} (set PBJ_ROOT)")
    return path


def resolve_pbjapp_root(explicit: Path | None = None) -> Path:
    if explicit:
        return explicit.resolve()
    env = (os.environ.get("PBJ_REPO_ROOT") or "").strip()
    if env:
        return Path(env).resolve()
    return (cms_data_paths.repo_root().parent / "PBJapp").resolve()


def file_uri_to_path(uri: str | None) -> Path:
    if not uri or not str(uri).startswith("file:"):
        raise StageError(f"expected local file URI, got {uri!r}")
    parsed = urlparse(str(uri))
    raw = unquote(parsed.path)
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return Path(raw)


def sha256_or_none(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    return sha256_file(path)


def destination_pre_state_at_base(
    dev_pbj_root: Path,
    publication_base_sha: str,
    rel_path: str,
) -> dict[str, Any]:
    from pbj320_publication import fingerprint_baseline_path

    fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, rel_path)
    return {
        "rel_path": rel_path.replace("\\", "/"),
        "existed_on_disk_before": bool(fp.get("present_at_base")),
        "sha256": fp.get("sha256"),
        "git_state_before": "tracked" if fp.get("present_at_base") else "absent",
    }


def publication_action(
    *,
    publication_class: str,
    existed_before: bool,
    material_change: bool,
    git_state_before: str = "unknown",
) -> str:
    if publication_class in {"validation_parity", "deploy_generated"}:
        return "validation-only"
    if git_state_before == "untracked":
        return "add"
    if not existed_before:
        return "add"
    if material_change:
        return "modify"
    return "none"


def artifact_row(
    *,
    rel_path: str,
    pre_state: dict[str, Any],
    proposed_sha: str | None,
    expected_release_id: str,
    transformation: str,
    role: str,
    publication_class: str,
    publication_base_sha: str | None = None,
    inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    dest_suffix = Path(rel_path.replace("\\", "/")).suffix.lstrip(".") or "unknown"
    prior_sha = pre_state.get("sha256")
    existed_before = bool(pre_state.get("existed_on_disk_before"))
    git_before = str(pre_state.get("git_state_before") or "unknown")
    material_change = bool(proposed_sha) and (not existed_before or prior_sha is None or prior_sha != proposed_sha)
    row = {
        "destination_id": role,
        "publication_class": publication_class,
        "publication_action": publication_action(
            publication_class=publication_class,
            existed_before=existed_before,
            material_change=material_change,
            git_state_before=git_before,
        ),
        "repo": "pbj-root",
        "path": rel_path.replace("\\", "/"),
        "artifact_type": dest_suffix,
        "existed_on_disk_before": existed_before,
        "git_state_before": git_before,
        "old_sha256": prior_sha,
        "proposed_sha256": proposed_sha,
        "expected_source_release_id": expected_release_id,
        "transformation": transformation,
        "present_before": existed_before,
        "material_change": material_change,
    }
    if publication_base_sha:
        row["publication_base_sha"] = publication_base_sha
        row["baseline_sha256"] = prior_sha
        row["baseline_existed"] = existed_before
    if inputs:
        row["inputs"] = inputs
    return row


def overlay_input(
    *,
    source_id: str,
    release_id: str,
    rel_path: str,
    sha256: str,
    role: str,
) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "release_id": release_id,
        "mode": "GOVERNED_OVERLAY",
        "path": rel_path.replace("\\", "/"),
        "sha256": sha256,
        "role": role,
    }


def baseline_input_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": row.get("source_id"),
        "release_id": row.get("release_id"),
        "mode": "PUBLISHED_BASELINE",
        "path": row.get("path"),
        "sha256": row.get("sha256"),
        "present_at_base": row.get("present_at_base"),
        "role": row.get("role"),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def run_python_script(
    script: Path,
    *,
    cwd: Path,
    args: list[str] | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    cmd = [sys.executable, str(script), *(args or [])]
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    return {
        "command": label or " ".join(cmd),
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "summary": tail[-1] if tail else f"exit {proc.returncode}",
    }


def copy_file_verified(src: Path, dest: Path, *, expected_sha: str | None = None) -> str:
    if not src.is_file():
        raise StageError(f"missing governed source: {src}")
    digest = sha256_file(src)
    if expected_sha and digest != expected_sha:
        raise StageError(f"sha256 mismatch for {src.name}: {digest} != {expected_sha}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        dest.write_bytes(src.read_bytes())
    return digest


def write_fail_manifest(
    *,
    source_id: str,
    release_id: str,
    error: str,
    root: Path | None = None,
    publication_base_sha: str | None = None,
) -> None:
    payload = {
        "schema_version": 2,
        "publication_contract_version": PUBLICATION_CONTRACT_VERSION,
        "source_id": source_id,
        "active_release_id": release_id,
        "status": "FAILED",
        "stage_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "publication_base_sha": publication_base_sha,
        "error": error,
        "destination_layers": {
            "canonical": "CURRENT",
            "pbj320_destination": "FAILED",
            "committed": "NO",
            "pushed": "NO",
            "deployed": "UNKNOWN",
            "production_verified": "NO",
        },
    }
    atomic_write_json(stage_manifest_path(source_id, release_id, root=root), payload)


def finalize_stage_manifest(
    *,
    source_id: str,
    release_id: str,
    publication_base_sha: str,
    publish_remote: str,
    publish_branch: str,
    baseline_wt: Path,
    artifact_cache: Path,
    dev_pbj_root: Path,
    artifacts: list[dict[str, Any]],
    gate_results: list[dict[str, Any]],
    files_added: list[str],
    files_modified: list[str],
    validation_artifacts: list[str],
    verification_contract: dict[str, Any],
    export_commands: list[str],
    graph: str,
    root: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    material = any(row.get("material_change") for row in artifacts)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "publication_contract_version": PUBLICATION_CONTRACT_VERSION,
        "source_id": source_id,
        "active_release_id": release_id,
        "status": STAGE_STATUS_STAGED,
        "stage_timestamp": now,
        "publication_base_sha": publication_base_sha,
        "publication_base_remote": publish_remote,
        "publication_base_branch": publish_branch,
        "stage_baseline_worktree": str(baseline_wt),
        "stage_artifact_cache": str(artifact_cache),
        "dev_pbj_root": str(dev_pbj_root),
        "idempotent_no_material_diff": not material,
        "destination_layers": {
            "canonical": "CURRENT",
            "pbj320_destination": STAGE_STATUS_STAGED,
            "committed": "NO",
            "pushed": "NO",
            "deployed": "UNKNOWN",
            "production_verified": "NO",
        },
        "artifacts": artifacts,
        "files_added": sorted(set(files_added)),
        "files_modified": sorted(set(files_modified)),
        "validation_artifacts": sorted(set(validation_artifacts)),
        "files_removed": [],
        "validation_gates": gate_results,
        "export_commands": export_commands,
        "verification_contract": verification_contract,
        "source_destination_graph": graph,
        "material_change": material,
        "next_human_step": (
            "Review production-relative stage manifest (diff vs publication_base_sha), then Publish "
            "via isolated worktree. Dev pbj-root checkout unchanged."
        ),
    }
    if extra:
        manifest.update(extra)
    out_path = stage_manifest_path(source_id, release_id, root=root)
    atomic_write_json(out_path, manifest)
    return {
        "status": STAGE_STATUS_STAGED,
        "material_change": material,
        "active_release_id": release_id,
        "publication_base_sha": publication_base_sha,
        "manifest_path": str(out_path),
        "manifest": manifest,
        "artifacts_changed": sum(1 for row in artifacts if row.get("material_change")),
    }
