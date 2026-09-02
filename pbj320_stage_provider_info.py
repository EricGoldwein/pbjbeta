"""Stage cms.provider_info artifacts into local pbj-root for PBJ320 (no commit/push/deploy)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cms_data_paths
from active_release_registry import get_active_release, registry_path, sha256_file
from pbj320_publication_contract import (
    PUBLICATION_CONTRACT_VERSION,
    evaluate_stage_publish_eligibility,
)
from pbj320_source_adapters import PI_STAGE_PUBLISH_SPEC
SOURCE_ID = "cms.provider_info"
STAGE_STATUS_STAGED = "STAGED"
STAGE_STATUS_FAILED = "FAILED"

# Verified from pbj-root/render.yaml buildCommand, docs/DATA_DEPLOY.md, app.py loaders.
PROVEN_CONSUMERS = (
    "public provider pages (/provider/*)",
    "state page aggregates (/state/*)",
    "MCP get_facility",
    "ownership legal-name crosswalk (provider_info_combined_latest.csv)",
    "public facility search (/search_index.json; public-search.js)",
)

SEARCH_INDEX_REL = "search_index.json"
SEARCH_INDEX_CACHE_NOTE = (
    "public-search.js caches /search_index.json in sessionStorage per browser tab; "
    "production verification should fetch /search_index.json directly."
)

GATE_COMMANDS = (
    "python scripts/backfill_provider_norm_urban.py",
    "python scripts/validate_provider_norm_snapshot.py",
    "python scripts/simulate_render_deploy_gates.py",
)


class ProviderInfoStageError(RuntimeError):
    pass


def _pbjapp_root() -> Path:
    configured = (os.environ.get("PBJ_REPO_ROOT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    sibling = (Path(__file__).resolve().parent.parent / "PBJapp").resolve()
    if sibling.is_dir():
        return sibling
    return cms_data_paths.repo_root()


def _file_uri_to_path(uri: str) -> Path:
    raw = unquote(urlparse(str(uri or "")).path)
    if os.name == "nt":
        if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]
        elif raw.startswith("//") and len(raw) > 3 and raw[3] == ":":
            raw = raw[2:]
    return Path(raw)


def _control_root(root: Path | None = None) -> Path:
    from release_control_plane import control_plane_root

    return control_plane_root(root)


def _stage_manifest_dir(root: Path | None = None) -> Path:
    return _control_root(root) / "state" / "pbj320_stages" / SOURCE_ID


def stage_manifest_path(release_id: str, *, root: Path | None = None) -> Path:
    return _stage_manifest_dir(root) / f"{release_id}.json"


def _resolve_pbj_root(explicit: Path | str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
    else:
        env = (os.environ.get("PBJ_ROOT") or "").strip()
        if env:
            path = Path(env).expanduser().resolve()
        else:
            path = (cms_data_paths.repo_root().parent / "pbj-root").resolve()
    if not path.is_dir():
        raise ProviderInfoStageError(f"pbj-root not found: {path} (set PBJ_ROOT)")
    return path


def _parse_release_id(release_id: str) -> tuple[int, int, str]:
    parts = (release_id or "").strip().split("-", 1)
    if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
        raise ProviderInfoStageError(f"invalid release_id: {release_id!r}")
    year, month = int(parts[0]), int(parts[1])
    if month < 1 or month > 12:
        raise ProviderInfoStageError(f"invalid release month in {release_id!r}")
    abbr = (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )[month - 1]
    return year, month, abbr


def _load_handoff(release_id: str, pbjapp_root: Path) -> dict[str, Any]:
    path = (
        cms_data_paths.provider_release_manifest_dir(release_id, pbjapp_root)
        / "pbj_root_handoff.json"
    )
    if not path.is_file():
        path = pbjapp_root / "provider_info" / "_manifests" / release_id / "pbj_root_handoff.json"
    if not path.is_file():
        raise ProviderInfoStageError(
            f"missing pbj_root_handoff.json for {release_id} under PBJapp provider_info/_manifests/"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _handoff_paths(release_id: str, handoff: dict[str, Any]) -> dict[str, str]:
    year, month, abbr = _parse_release_id(release_id)
    sync = handoff.get("pbj_root_sync") or {}
    nh = handoff.get("pbj_root_nh_snapshot_sync") or {}
    norm_rel = str(sync.get("destination_file") or f"provider_info/ProviderInfoNorm_{year}_{month:02d}.csv")
    nh_rel = str(nh.get("destination_file") or f"provider_info/NH_ProviderInfo_{abbr}{year}.csv")
    return {
        "norm": norm_rel.replace("\\", "/"),
        "nh": nh_rel.replace("\\", "/"),
        "combined_latest": "provider_info_combined_latest.csv",
        "state_aggregates": "data/state_page_aggregates.json.gz",
        "search_index": SEARCH_INDEX_REL,
    }


def _sha256_or_none(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size <= 0:
        return None
    return sha256_file(path)


def _git_disposition(pbj_root: Path, rel_path: str) -> str:
    """Git working-tree disposition for a pbj-root relative path before Stage writes."""
    rel_posix = rel_path.replace("\\", "/")
    dest = pbj_root / rel_posix.replace("/", os.sep)
    try:
        ignored = subprocess.run(
            ["git", "-C", str(pbj_root), "check-ignore", "-q", "--", rel_posix],
            capture_output=True,
            check=False,
        )
        if ignored.returncode == 0:
            return "ignored"
        tracked = subprocess.run(
            ["git", "-C", str(pbj_root), "ls-files", "--error-unmatch", "--", rel_posix],
            capture_output=True,
            check=False,
        )
        if tracked.returncode == 0:
            return "tracked" if dest.is_file() else "tracked_absent"
        if dest.is_file():
            return "untracked"
        return "absent"
    except (FileNotFoundError, OSError):
        return "unknown"


def _git_head_sha256(pbj_root: Path, rel_path: str) -> str | None:
    """SHA256 of the path at HEAD, or None when not tracked in the last commit."""
    rel_posix = rel_path.replace("\\", "/")
    try:
        proc = subprocess.run(
            ["git", "-C", str(pbj_root), "show", f"HEAD:{rel_posix}"],
            capture_output=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout:
            return None
        import hashlib

        return hashlib.sha256(proc.stdout).hexdigest()
    except (FileNotFoundError, OSError):
        return None


def _destination_pre_state(pbj_root: Path, rel_path: str) -> dict[str, Any]:
    dest = pbj_root / rel_path.replace("/", os.sep)
    existed = dest.is_file() and dest.stat().st_size > 0
    return {
        "rel_path": rel_path.replace("\\", "/"),
        "existed_on_disk_before": existed,
        "sha256": _sha256_or_none(dest) if existed else None,
        "git_state_before": _git_disposition(pbj_root, rel_path),
    }


def _destination_pre_state_for_manifest_refresh(pbj_root: Path, rel_path: str) -> dict[str, Any]:
    """Pre-stage baseline from git HEAD vs current working tree (manifest refresh only)."""
    dest = pbj_root / rel_path.replace("/", os.sep)
    on_disk = dest.is_file() and dest.stat().st_size > 0
    git_before = _git_disposition(pbj_root, rel_path)
    head_sha = _git_head_sha256(pbj_root, rel_path)
    existed_in_repo_before = head_sha is not None or git_before == "tracked_absent"
    if git_before == "untracked":
        existed_in_repo_before = False
    return {
        "rel_path": rel_path.replace("\\", "/"),
        "existed_on_disk_before": existed_in_repo_before,
        "sha256": head_sha,
        "git_state_before": git_before,
        "on_disk_sha256": _sha256_or_none(dest) if on_disk else None,
    }


def _publication_action(
    *,
    publication_class: str,
    existed_before: bool,
    material_change: bool,
    git_state_before: str = "unknown",
) -> str:
    if publication_class == "validation_parity":
        return "validation-only"
    if git_state_before == "untracked":
        return "add"
    if not existed_before:
        return "add"
    if material_change:
        return "modify"
    return "none"


def _destination_pre_state_at_base(
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


def _artifact_row(
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
    material_change = (
        bool(proposed_sha)
        and (
            not existed_before
            or prior_sha is None
            or prior_sha != proposed_sha
        )
    )
    publication_action = _publication_action(
        publication_class=publication_class,
        existed_before=existed_before,
        material_change=material_change,
        git_state_before=git_before,
    )
    row = {
        "destination_id": role,
        "publication_class": publication_class,
        "publication_action": publication_action,
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


def load_stage_manifest(release_id: str, *, root: Path | None = None) -> dict[str, Any] | None:
    path = stage_manifest_path(release_id, root=root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def evaluate_pi_stage_publish_eligibility(
    manifest: dict[str, Any] | None,
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    release_id = str((manifest or {}).get("active_release_id") or "")
    return evaluate_stage_publish_eligibility(
        manifest,
        PI_STAGE_PUBLISH_SPEC,
        root=root,
        release_id=release_id or None,
    )


def pi_stage_manifest_publishable(
    manifest: dict[str, Any],
    *,
    root: Path | None = None,
) -> bool:
    """True when manifest satisfies baseline+overlay Publish requirements."""
    return bool(evaluate_pi_stage_publish_eligibility(manifest, root=root).get("publishable"))


def pi_stage_manifest_stale_detail(
    manifest: dict[str, Any],
    *,
    root: Path | None = None,
) -> str | None:
    """Human-readable reason a STAGED manifest cannot be published as-is."""
    evaluation = evaluate_pi_stage_publish_eligibility(manifest, root=root)
    if not evaluation.get("stale"):
        return None
    return str(evaluation.get("detail") or "manifest not publishable")


def audit_provider_info_pbj320_destination(
    *,
    root: Path | None = None,
    pbj_root: Path | None = None,
    pbjapp_root: Path | None = None,
) -> dict[str, Any]:
    """Compare governed ACTIVE vs local pbj-root working-tree destinations (not production)."""
    root = root or cms_data_paths.repo_root()
    pbjapp_root = pbjapp_root or _pbjapp_root()
    pbj_root = pbj_root or _resolve_pbj_root()

    active = get_active_release(SOURCE_ID, registry_path(root)) or {}
    release_id = str(active.get("active_release_id") or "")
    if not release_id:
        return {
            "source_id": SOURCE_ID,
            "canonical_current": False,
            "destination_staged": False,
            "stage_detail": "No ACTIVE Provider Information release in registry.",
        }

    canonical_path = _file_uri_to_path(str(active.get("source_uri") or ""))
    canonical_sha = str(active.get("hash") or "")
    canonical_current = (
        canonical_path.is_file()
        and canonical_sha
        and sha256_file(canonical_path) == canonical_sha
    )

    try:
        handoff = _load_handoff(release_id, pbjapp_root)
        rel_paths = _handoff_paths(release_id, handoff)
    except ProviderInfoStageError as exc:
        return {
            "source_id": SOURCE_ID,
            "active_release_id": release_id,
            "canonical_current": canonical_current,
            "destination_staged": False,
            "stage_detail": str(exc),
        }

    expected_norm_sha = str((handoff.get("pbj_root_sync") or {}).get("sha256") or canonical_sha)
    norm_dest = pbj_root / rel_paths["norm"].replace("/", os.sep)
    norm_staged = norm_dest.is_file() and _sha256_or_none(norm_dest) == expected_norm_sha

    combined_dest = pbj_root / rel_paths["combined_latest"]
    combined_staged = False
    combined_detail = "combined_latest missing"
    if combined_dest.is_file():
        try:
            import pandas as pd

            sample = pd.read_csv(combined_dest, usecols=["processing_date"], nrows=5000, dtype=str)
            max_date = str(sample["processing_date"].max() or "")
            year, month, _ = _parse_release_id(release_id)
            combined_staged = max_date.startswith(f"{year}-{month:02d}")
            combined_detail = f"combined_latest max processing_date={max_date or '—'}"
        except Exception as exc:  # noqa: BLE001
            combined_detail = f"combined_latest unreadable: {exc}"

    manifest = load_stage_manifest(release_id, root=root)
    manifest_staged = (
        manifest is not None
        and manifest.get("status") == STAGE_STATUS_STAGED
        and manifest.get("active_release_id") == release_id
    )
    artifact_root = pbj_root
    cache_path = str(manifest.get("stage_artifact_cache") or "") if manifest else ""
    if cache_path:
        cache_dir = Path(cache_path)
        if cache_dir.is_dir():
            artifact_root = cache_dir
    if manifest_staged and manifest.get("artifacts"):
        on_disk_match = all(
            _sha256_or_none(artifact_root / str(row.get("path") or "").replace("/", os.sep))
            == str(row.get("proposed_sha256") or "")
            for row in manifest["artifacts"]
            if row.get("proposed_sha256")
            and row.get("path")
            and row.get("publication_class") in {"commit_destination", "shared_derived"}
        )
        manifest_staged = manifest_staged and on_disk_match

    destination_staged = norm_staged and combined_staged and manifest_staged

    return {
        "source_id": SOURCE_ID,
        "active_release_id": release_id,
        "canonical_artifact_path": str(canonical_path),
        "canonical_sha256": canonical_sha,
        "canonical_current": canonical_current,
        "destination_staged": destination_staged,
        "destination_norm_staged": norm_staged,
        "destination_combined_staged": combined_staged,
        "stage_manifest_present": manifest is not None,
        "publication_base_sha": str(manifest.get("publication_base_sha") or "") if manifest else "",
        "stage_artifact_cache": cache_path or None,
        "stage_detail": (
            f"PBJ320 staged for {release_id} (artifact cache)"
            if destination_staged and cache_path
            else (
                f"PBJ320 working tree staged for {release_id}"
                if destination_staged
                else f"Norm {'ok' if norm_staged else 'stale'} · {combined_detail}"
            )
        ),
        "pbj_root": str(pbj_root),
        "production_deployed": "UNKNOWN",
        "production_verified": False,
    }


def _backup_file(path: Path) -> dict[str, Any]:
    if path.is_file():
        return {"path": str(path), "existed": True, "bytes": path.read_bytes()}
    return {"path": str(path), "existed": False, "bytes": None}


def _restore_backup(backup: dict[str, Any]) -> None:
    path = Path(str(backup["path"]))
    if backup.get("existed"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(backup["bytes"])
    elif path.is_file():
        path.unlink()


def _copy_canonical(src: Path, dst: Path, *, expected_sha: str) -> str:
    if not src.is_file():
        raise ProviderInfoStageError(f"canonical source missing: {src}")
    digest = sha256_file(src)
    if expected_sha and digest != expected_sha:
        raise ProviderInfoStageError(
            f"canonical sha256 mismatch: disk={digest} expected={expected_sha}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return digest


def _rebuild_combined_latest(
    *,
    pbjapp_root: Path,
    pbj_root: Path,
    release_id: str,
    combined_rel: str,
    canonical_norm_dir: Path | None = None,
) -> str:
    """Rebuild provider_info_combined_latest.csv from PBJapp Norm snapshots (latest month rows)."""
    build_script = pbjapp_root / "scripts" / "build_provider_info_combined.py"
    if not build_script.is_file():
        raise ProviderInfoStageError(f"missing builder: {build_script}")

    with tempfile.TemporaryDirectory(prefix="pi-combined-") as tmp:
        tmp_combined = Path(tmp) / "provider_info_combined.csv"
        # PBJapp builder resolves Norm via cms_data_paths (PBJ_REPO_ROOT + env/json overrides).
        # Data Ops may set PBJ_REPO_ROOT to the control-plane root, and PBJapp data_paths.local.json
        # may redirect Norm to bulk storage that lags the governed ACTIVE artifact — pin both.
        sub_env = os.environ.copy()
        sub_env["PBJ_REPO_ROOT"] = str(pbjapp_root.resolve())
        norm_dir = (canonical_norm_dir or (pbjapp_root / "provider_info_normalized")).resolve()
        sub_env["PBJ_PROVIDER_INFO_NORMALIZED"] = str(norm_dir)
        proc = subprocess.run(
            [
                sys.executable,
                str(build_script),
                "--output",
                str(tmp_combined),
                "--through",
                release_id,
                "--require-month",
                release_id,
            ],
            cwd=str(pbjapp_root),
            env=sub_env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise ProviderInfoStageError(
                f"build_provider_info_combined failed: {tail[-1] if tail else proc.returncode}"
            )
        if not tmp_combined.is_file():
            raise ProviderInfoStageError("build_provider_info_combined produced no output")

        import pandas as pd

        df = pd.read_csv(tmp_combined, dtype=str, low_memory=False)
        if "processing_date" not in df.columns:
            raise ProviderInfoStageError("combined rebuild missing processing_date column")
        max_date = df["processing_date"].max()
        latest = df[df["processing_date"] == max_date].copy()
        year, month, _ = _parse_release_id(release_id)
        if not str(max_date).startswith(f"{year}-{month:02d}"):
            raise ProviderInfoStageError(
                f"combined_latest month mismatch: max processing_date={max_date} expected {release_id}"
            )
        out = pbj_root / combined_rel.replace("/", os.sep)
        out.parent.mkdir(parents=True, exist_ok=True)
        latest.to_csv(out, index=False)
        return sha256_file(out)


def _run_pbj_root_gates(pbj_root: Path, *, release_id: str) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    commands = list(GATE_COMMANDS) + [
        f"python scripts/verify_provider_release_handoff.py --release-key {release_id}",
    ]
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=str(pbj_root),
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


def _run_state_aggregates(pbj_root: Path) -> tuple[str | None, dict[str, Any]]:
    script = pbj_root / "scripts" / "build_state_page_aggregates.py"
    if not script.is_file():
        return None, {
            "command": str(script),
            "exit_code": -1,
            "passed": False,
            "summary": "build_state_page_aggregates.py missing",
        }
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(pbj_root),
        capture_output=True,
        text=True,
        check=False,
    )
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    rel = "data/state_page_aggregates.json.gz"
    digest = _sha256_or_none(pbj_root / rel.replace("/", os.sep))
    return digest, {
        "command": f"python scripts/build_state_page_aggregates.py",
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0,
        "summary": tail[-1] if tail else f"exit {proc.returncode}",
    }


def _run_search_index(pbj_root: Path) -> tuple[str | None, dict[str, Any]]:
    """Rebuild search_index.json from staged Provider Info (+ existing SFF/chain inputs)."""
    script = pbj_root / "generate_search_index.py"
    if not script.is_file():
        return None, {
            "command": str(script),
            "exit_code": -1,
            "passed": False,
            "summary": "generate_search_index.py missing",
        }
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(pbj_root),
        capture_output=True,
        text=True,
        check=False,
    )
    tail = (proc.stdout or proc.stderr or "").strip().splitlines()
    rel = SEARCH_INDEX_REL
    digest = _sha256_or_none(pbj_root / rel)
    return digest, {
        "command": "python generate_search_index.py",
        "exit_code": proc.returncode,
        "passed": proc.returncode == 0 and digest is not None,
        "summary": tail[-1] if tail else f"exit {proc.returncode}",
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
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


def stage_provider_info_for_pbj320(
    *,
    root: Path | None = None,
    pbj_root: Path | str | None = None,
    pbjapp_root: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build PI publication candidate in isolated origin/master baseline worktree."""
    from pbj320_pi_stage_inputs import (
        collect_pi_overlay_fingerprints,
        collect_shared_baseline_inputs,
        inputs_for_combined_latest,
        inputs_for_provider_norm,
        inputs_for_search_index,
        inputs_for_state_aggregates,
        unresolved_inputs,
    )
    from pbj320_publication import (
        fetch_publish_base,
        fingerprint_baseline_path,
        prepare_baseline_worktree,
        resolve_publish_branch,
        stage_artifact_cache_path,
        stage_baseline_worktree_path,
        sync_stage_artifacts_to_cache,
    )

    root = root or cms_data_paths.repo_root()
    pbjapp_root = pbjapp_root or _pbjapp_root()
    dev_pbj_root = _resolve_pbj_root(pbj_root)

    active = get_active_release(SOURCE_ID, registry_path(root)) or {}
    release_id = str(active.get("active_release_id") or "")
    if not release_id:
        raise ProviderInfoStageError("cms.provider_info has no ACTIVE release")

    canonical_uri = str(active.get("source_uri") or "")
    canonical_path = _file_uri_to_path(canonical_uri)
    canonical_sha = str(active.get("hash") or "")

    handoff = _load_handoff(release_id, pbjapp_root)
    rel_paths = _handoff_paths(release_id, handoff)
    expected_norm_sha = str((handoff.get("pbj_root_sync") or {}).get("sha256") or canonical_sha)

    nh_sync = handoff.get("pbj_root_nh_snapshot_sync") or {}
    nh_src = pbjapp_root / str(nh_sync.get("source_file") or "").replace("\\", "/")
    if not nh_src.is_file():
        nh_year, nh_month, nh_abbr = _parse_release_id(release_id)
        nh_src = cms_data_paths.provider_info_dir(pbjapp_root) / f"NH_ProviderInfo_{nh_abbr}{nh_year}.csv"
    nh_expected_sha = str(nh_sync.get("sha256") or "")

    publish_branch, publish_remote = resolve_publish_branch(dev_pbj_root)
    publication_base_sha = fetch_publish_base(dev_pbj_root, remote=publish_remote, branch=publish_branch)
    baseline_wt = stage_baseline_worktree_path(SOURCE_ID, release_id, root=root)
    artifact_cache = stage_artifact_cache_path(SOURCE_ID, release_id, root=root)

    commit_rels = [
        rel_paths["norm"],
        rel_paths["combined_latest"],
        rel_paths["state_aggregates"],
        rel_paths["search_index"],
    ]

    pre_manifest = load_stage_manifest(release_id, root=root)
    if (
        pre_manifest
        and pre_manifest.get("status") == STAGE_STATUS_STAGED
        and not force
        and str(pre_manifest.get("publication_base_sha") or "") == publication_base_sha
    ):
        on_disk_ok = True
        for row in pre_manifest.get("artifacts") or []:
            rel = str(row.get("path") or "")
            expected = str(row.get("proposed_sha256") or "")
            if not rel or not expected:
                continue
            if row.get("publication_class") == "validation_parity":
                continue
            actual = _sha256_or_none(artifact_cache / rel.replace("/", os.sep))
            if actual != expected:
                on_disk_ok = False
                break
        if on_disk_ok:
            return {
                "status": "NO_MATERIAL_DIFF",
                "active_release_id": release_id,
                "manifest_path": str(stage_manifest_path(release_id, root=root)),
                "manifest": pre_manifest,
                "message": "PBJ320 destination already matches staged manifest — no material diff.",
                "audit": audit_provider_info_pbj320_destination(
                    root=root, pbj_root=dev_pbj_root, pbjapp_root=pbjapp_root
                ),
            }

    baseline_inputs = collect_shared_baseline_inputs(dev_pbj_root, baseline_wt, publication_base_sha)
    if unresolved_inputs(baseline_inputs):
        raise ProviderInfoStageError(
            "UNRESOLVED baseline inputs: " + ", ".join(unresolved_inputs(baseline_inputs))
        )

    baseline_pre_states = {
        rel: _destination_pre_state_at_base(dev_pbj_root, publication_base_sha, rel)
        for rel in commit_rels
    }
    nh_baseline_pre = _destination_pre_state_at_base(dev_pbj_root, publication_base_sha, rel_paths["nh"])

    artifacts: list[dict[str, Any]] = []
    gate_results: list[dict[str, Any]] = []
    files_added: list[str] = []
    files_modified: list[str] = []
    validation_artifacts: list[str] = []

    def _record_publication_lists(row: dict[str, Any]) -> None:
        rel = str(row.get("path") or "")
        action = str(row.get("publication_action") or "")
        if row.get("publication_class") == "validation_parity":
            if rel:
                validation_artifacts.append(rel)
            return
        if action == "add" and rel:
            files_added.append(rel)
        elif action == "modify" and rel:
            files_modified.append(rel)

    try:
        prepare_baseline_worktree(
            dev_pbj_root,
            remote=publish_remote,
            branch=publish_branch,
            base_sha=publication_base_sha,
            worktree_path=baseline_wt,
        )

        norm_dest = baseline_wt / rel_paths["norm"].replace("/", os.sep)
        norm_sha = _copy_canonical(canonical_path, norm_dest, expected_sha=expected_norm_sha)
        overlay_fps: dict[str, dict[str, Any]] = {"norm": {}}

        nh_sha: str | None = None
        nh_overlay: dict[str, Any] | None = None
        if nh_src.is_file():
            nh_dest = baseline_wt / rel_paths["nh"].replace("/", os.sep)
            nh_sha = _copy_canonical(
                nh_src,
                nh_dest,
                expected_sha=nh_expected_sha or sha256_file(nh_src),
            )
            if nh_expected_sha and nh_sha != nh_expected_sha:
                raise ProviderInfoStageError(f"NH snapshot sha256 mismatch vs handoff: {nh_sha}")
            nh_row = _artifact_row(
                rel_path=rel_paths["nh"],
                pre_state=nh_baseline_pre,
                proposed_sha=nh_sha,
                expected_release_id=release_id,
                transformation="sync_to_pbj_root NH snapshot copy (parity gates; gitignored in pbj-root)",
                role="nh_snapshot_parity",
                publication_class="validation_parity",
                publication_base_sha=publication_base_sha,
            )
            artifacts.append(nh_row)
            _record_publication_lists(nh_row)

        combined_sha = _rebuild_combined_latest(
            pbjapp_root=pbjapp_root,
            pbj_root=baseline_wt,
            release_id=release_id,
            combined_rel=rel_paths["combined_latest"],
            canonical_norm_dir=canonical_path.parent,
        )

        overlay_fps = collect_pi_overlay_fingerprints(
            baseline_wt,
            release_id=release_id,
            rel_paths=rel_paths,
            norm_sha=norm_sha,
            nh_sha=nh_sha,
            combined_sha=combined_sha,
        )
        if nh_sha:
            nh_overlay = overlay_fps.get("nh_snapshot")

        norm_row = _artifact_row(
            rel_path=rel_paths["norm"],
            pre_state=baseline_pre_states[rel_paths["norm"]],
            proposed_sha=norm_sha,
            expected_release_id=release_id,
            transformation="PBJapp/scripts/sync_to_pbj_root.py provider-norm (copy ProviderInfoNorm)",
            role="provider_norm",
            publication_class="commit_destination",
            publication_base_sha=publication_base_sha,
            inputs=inputs_for_provider_norm(overlay_fps["norm"]),
        )
        artifacts.append(norm_row)
        _record_publication_lists(norm_row)

        combined_row = _artifact_row(
            rel_path=rel_paths["combined_latest"],
            pre_state=baseline_pre_states[rel_paths["combined_latest"]],
            proposed_sha=combined_sha,
            expected_release_id=release_id,
            transformation="PBJapp/scripts/build_provider_info_combined.py → latest-month slice",
            role="provider_combined_latest",
            publication_class="commit_destination",
            publication_base_sha=publication_base_sha,
            inputs=inputs_for_combined_latest(overlay_fps["combined_latest"], overlay_fps["norm"]),
        )
        artifacts.append(combined_row)
        _record_publication_lists(combined_row)

        gate_results = _run_pbj_root_gates(baseline_wt, release_id=release_id)
        if not all(row.get("passed") for row in gate_results):
            failed = next(row for row in gate_results if not row.get("passed"))
            raise ProviderInfoStageError(f"pre-publication gate failed: {failed.get('command')}")

        agg_sha, agg_result = _run_state_aggregates(baseline_wt)
        gate_results.append(agg_result)
        if not agg_result.get("passed"):
            raise ProviderInfoStageError("build_state_page_aggregates failed")
        if agg_sha:
            agg_row = _artifact_row(
                rel_path=rel_paths["state_aggregates"],
                pre_state=baseline_pre_states[rel_paths["state_aggregates"]],
                proposed_sha=agg_sha,
                expected_release_id=release_id,
                transformation="pbj-root/scripts/build_state_page_aggregates.py",
                role="state_page_aggregates",
                publication_class="shared_derived",
                publication_base_sha=publication_base_sha,
                inputs=inputs_for_state_aggregates(overlay=overlay_fps, baseline=baseline_inputs),
            )
            artifacts.append(agg_row)
            _record_publication_lists(agg_row)

        search_sha, search_result = _run_search_index(baseline_wt)
        gate_results.append(search_result)
        if not search_result.get("passed"):
            raise ProviderInfoStageError("generate_search_index failed")
        if search_sha:
            search_row = _artifact_row(
                rel_path=rel_paths["search_index"],
                pre_state=baseline_pre_states[rel_paths["search_index"]],
                proposed_sha=search_sha,
                expected_release_id=release_id,
                transformation="pbj-root/generate_search_index.py (Provider Info + baseline SFF/chain)",
                role="search_index",
                publication_class="shared_derived",
                publication_base_sha=publication_base_sha,
                inputs=inputs_for_search_index(
                    overlay=overlay_fps,
                    baseline=baseline_inputs,
                    nh_overlay=nh_overlay,
                ),
            )
            artifacts.append(search_row)
            _record_publication_lists(search_row)

        sync_stage_artifacts_to_cache(
            baseline_wt=baseline_wt,
            cache_dir=artifact_cache,
            rel_paths=commit_rels + ([rel_paths["nh"]] if nh_sha else []),
        )
        sync_stage_artifacts_to_cache(
            baseline_wt=baseline_wt,
            cache_dir=dev_pbj_root,
            rel_paths=commit_rels + ([rel_paths["nh"]] if nh_sha else []),
        )

        material = any(row.get("material_change") for row in artifacts)
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        manifest = {
            "schema_version": 2,
            "publication_contract_version": PUBLICATION_CONTRACT_VERSION,
            "source_id": SOURCE_ID,
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
            "canonical": {
                "path": str(canonical_path),
                "sha256": canonical_sha,
                "state": "CURRENT",
            },
            "pbj_root": str(dev_pbj_root),
            "destination_layers": {
                "canonical": "CURRENT",
                "pbj320_destination": "STAGED",
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
            "affected_consumers": list(PROVEN_CONSUMERS),
            "validation_gates": gate_results,
            "export_commands": [
                f"PBJapp/scripts/sync_to_pbj_root.py provider-norm --release-key {release_id} --force --run-gates",
                f"PBJapp/scripts/build_provider_info_combined.py --through {release_id}",
                "pbj-root/scripts/build_state_page_aggregates.py",
                "pbj-root/generate_search_index.py",
            ],
            "search_index_cache_note": SEARCH_INDEX_CACHE_NOTE,
            "handoff_manifest": str(
                pbjapp_root / "provider_info" / "_manifests" / release_id / "pbj_root_handoff.json"
            ),
            "material_change": material,
            "next_human_step": (
                "Review production-relative stage manifest (diff vs publication_base_sha), then Publish "
                "via isolated worktree. Dev pbj-root checkout unchanged."
            ),
        }
        out_path = stage_manifest_path(release_id, root=root)
        _atomic_write_json(out_path, manifest)

        return {
            "status": STAGE_STATUS_STAGED,
            "material_change": material,
            "active_release_id": release_id,
            "publication_base_sha": publication_base_sha,
            "manifest_path": str(out_path),
            "manifest": manifest,
            "artifacts_changed": sum(1 for row in artifacts if row.get("material_change")),
            "audit": audit_provider_info_pbj320_destination(
                root=root, pbj_root=dev_pbj_root, pbjapp_root=pbjapp_root
            ),
        }
    except Exception:
        fail_path = stage_manifest_path(release_id, root=root)
        fail_payload = {
            "schema_version": 2,
            "source_id": SOURCE_ID,
            "active_release_id": release_id,
            "status": STAGE_STATUS_FAILED,
            "stage_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "publication_base_sha": publication_base_sha,
            "validation_gates": gate_results,
            "destination_layers": {
                "pbj320_destination": "NOT_STAGED",
            },
        }
        _atomic_write_json(fail_path, fail_payload)
        raise


def refresh_provider_info_stage_manifest(
    *,
    root: Path | None = None,
    pbj_root: Path | str | None = None,
    pbjapp_root: Path | None = None,
) -> dict[str, Any]:
    """Rewrite STAGED manifest from stage artifact cache; no destination mutation."""
    root = root or cms_data_paths.repo_root()
    pbjapp_root = pbjapp_root or _pbjapp_root()
    dev_pbj_root = _resolve_pbj_root(pbj_root)

    active = get_active_release(SOURCE_ID, registry_path(root)) or {}
    release_id = str(active.get("active_release_id") or "")
    if not release_id:
        raise ProviderInfoStageError("cms.provider_info has no ACTIVE release")

    prev = load_stage_manifest(release_id, root=root)
    if not prev or prev.get("status") != STAGE_STATUS_STAGED:
        raise ProviderInfoStageError(f"No STAGED manifest for {release_id} to refresh")

    publication_base_sha = str(prev.get("publication_base_sha") or "")
    if not publication_base_sha:
        raise ProviderInfoStageError(
            f"STAGED manifest for {release_id} missing publication_base_sha; re-Stage against production baseline"
        )

    from pbj320_publication import stage_artifact_cache_path

    artifact_cache = stage_artifact_cache_path(SOURCE_ID, release_id, root=root)
    if not artifact_cache.is_dir():
        raise ProviderInfoStageError(f"stage artifact cache missing: {artifact_cache}")

    canonical_uri = str(active.get("source_uri") or "")
    canonical_path = _file_uri_to_path(canonical_uri)
    canonical_sha = str(active.get("hash") or "")

    handoff = _load_handoff(release_id, pbjapp_root)
    rel_paths = _handoff_paths(release_id, handoff)
    expected_norm_sha = str((handoff.get("pbj_root_sync") or {}).get("sha256") or canonical_sha)
    nh_rel = rel_paths["nh"]

    refresh_specs: list[tuple[str, str, str, str]] = [
        (rel_paths["norm"], "provider_norm", "commit_destination", expected_norm_sha),
        (rel_paths["combined_latest"], "provider_combined_latest", "commit_destination", ""),
        (rel_paths["state_aggregates"], "state_page_aggregates", "shared_derived", ""),
        (rel_paths["search_index"], "search_index", "shared_derived", ""),
    ]

    prev_inputs_by_role = {
        str(row.get("destination_id") or ""): list(row.get("inputs") or [])
        for row in prev.get("artifacts") or []
    }

    pre_states = {
        rel: _destination_pre_state_at_base(dev_pbj_root, publication_base_sha, rel)
        for rel, *_rest in refresh_specs
    }
    pre_states[nh_rel] = _destination_pre_state_at_base(dev_pbj_root, publication_base_sha, nh_rel)

    norm_cache = artifact_cache / rel_paths["norm"].replace("/", os.sep)
    norm_on_disk = _sha256_or_none(norm_cache)
    if not norm_on_disk:
        raise ProviderInfoStageError(f"staged destination missing in cache: {rel_paths['norm']}")
    if expected_norm_sha and norm_on_disk != expected_norm_sha:
        raise ProviderInfoStageError(
            f"staged norm sha256 mismatch: cache={norm_on_disk} expected={expected_norm_sha}"
        )

    artifacts: list[dict[str, Any]] = []
    files_added: list[str] = []
    files_modified: list[str] = []
    validation_artifacts: list[str] = []

    def _record_publication_lists(row: dict[str, Any]) -> None:
        rel = str(row.get("path") or "")
        action = str(row.get("publication_action") or "")
        if row.get("publication_class") == "validation_parity":
            if rel:
                validation_artifacts.append(rel)
            return
        if action == "add" and rel:
            files_added.append(rel)
        elif action == "modify" and rel:
            files_modified.append(rel)

    transformations = {
        "provider_norm": "PBJapp/scripts/sync_to_pbj_root.py provider-norm (copy ProviderInfoNorm)",
        "provider_combined_latest": "PBJapp/scripts/build_provider_info_combined.py → latest-month slice",
        "state_page_aggregates": "pbj-root/scripts/build_state_page_aggregates.py",
        "search_index": "pbj-root/generate_search_index.py",
        "nh_snapshot_parity": "sync_to_pbj_root NH snapshot copy (parity gates; gitignored in pbj-root)",
    }

    for rel, role, pub_class, _expected in refresh_specs:
        pre = pre_states[rel]
        cache_path = artifact_cache / rel.replace("/", os.sep)
        on_disk = _sha256_or_none(cache_path)
        if not on_disk:
            raise ProviderInfoStageError(f"staged destination missing in cache: {rel}")
        row = _artifact_row(
            rel_path=rel,
            pre_state=pre,
            proposed_sha=on_disk,
            expected_release_id=release_id,
            transformation=transformations[role],
            role=role,
            publication_class=pub_class,
            publication_base_sha=publication_base_sha,
            inputs=prev_inputs_by_role.get(role) or None,
        )
        artifacts.append(row)
        _record_publication_lists(row)

    nh_cache = artifact_cache / nh_rel.replace("/", os.sep)
    nh_on_disk = _sha256_or_none(nh_cache)
    if nh_on_disk:
        nh_row = _artifact_row(
            rel_path=nh_rel,
            pre_state=pre_states[nh_rel],
            proposed_sha=nh_on_disk,
            expected_release_id=release_id,
            transformation=transformations["nh_snapshot_parity"],
            role="nh_snapshot_parity",
            publication_class="validation_parity",
            publication_base_sha=publication_base_sha,
        )
        artifacts.append(nh_row)
        _record_publication_lists(nh_row)

    material = any(row.get("material_change") for row in artifacts)
    original_stage_timestamp = str(prev.get("stage_timestamp") or "").strip()
    if not original_stage_timestamp:
        raise ProviderInfoStageError(
            f"STAGED manifest for {release_id} missing stage_timestamp; cannot refresh provenance"
        )
    manifest_refreshed_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    manifest = {
        "schema_version": 2,
        "source_id": SOURCE_ID,
        "active_release_id": release_id,
        "status": STAGE_STATUS_STAGED,
        "stage_timestamp": original_stage_timestamp,
        "publication_base_sha": publication_base_sha,
        "publication_base_remote": prev.get("publication_base_remote"),
        "publication_base_branch": prev.get("publication_base_branch"),
        "stage_baseline_worktree": prev.get("stage_baseline_worktree"),
        "stage_artifact_cache": str(artifact_cache),
        "dev_pbj_root": str(dev_pbj_root),
        "manifest_refreshed": True,
        "manifest_refreshed_at": manifest_refreshed_at,
        "manifest_refresh_note": (
            "Manifest regenerated from stage artifact cache vs publication_base_sha baseline; "
            "no Stage writes or gate re-runs."
        ),
        "idempotent_no_material_diff": not material,
        "canonical": {
            "path": str(canonical_path),
            "sha256": canonical_sha,
            "state": "CURRENT",
        },
        "pbj_root": str(dev_pbj_root),
        "destination_layers": dict(prev.get("destination_layers") or {}),
        "artifacts": artifacts,
        "files_added": sorted(set(files_added)),
        "files_modified": sorted(set(files_modified)),
        "validation_artifacts": sorted(set(validation_artifacts)),
        "files_removed": [],
        "affected_consumers": list(PROVEN_CONSUMERS),
        "validation_gates": list(prev.get("validation_gates") or []),
        "export_commands": list(prev.get("export_commands") or []),
        "handoff_manifest": str(
            pbjapp_root / "provider_info" / "_manifests" / release_id / "pbj_root_handoff.json"
        ),
        "material_change": material,
        "next_human_step": prev.get("next_human_step")
        or (
            "Review production-relative stage manifest (diff vs publication_base_sha), then Publish "
            "from isolated worktree when ready."
        ),
    }
    manifest.setdefault("destination_layers", {})
    manifest["destination_layers"]["pbj320_destination"] = "STAGED"

    out_path = stage_manifest_path(release_id, root=root)
    _atomic_write_json(out_path, manifest)

    return {
        "status": "MANIFEST_REFRESHED",
        "active_release_id": release_id,
        "manifest_path": str(out_path),
        "manifest": manifest,
        "audit": audit_provider_info_pbj320_destination(
            root=root, pbj_root=dev_pbj_root, pbjapp_root=pbjapp_root
        ),
    }
