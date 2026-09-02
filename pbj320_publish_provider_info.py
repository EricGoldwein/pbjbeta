"""Publish cms.provider_info from STAGED manifest via isolated publication worktree."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cms_data_paths
from active_release_registry import get_active_release, registry_path, sha256_file

from pbj320_publication import (
    PublicationAdapter,
    PublicationError,
    fetch_publish_base,
    publish_from_adapter,
    resolve_publish_branch,
)
from pbj320_stage_provider_info import (
    SOURCE_ID,
    STAGE_STATUS_STAGED,
    _resolve_pbj_root,
    _sha256_or_none,
    load_stage_manifest,
    stage_manifest_path,
)

ProviderInfoPublishError = PublicationError

PUBLISH_STATUS_COMMITTED = "COMMITTED"
PUBLISH_STATUS_PUSHED = "PUSHED"
PUBLISH_STATUS_FAILED = "FAILED"

SEARCH_INDEX_CACHE_NOTE = (
    "public-search.js caches /search_index.json in sessionStorage for the browser tab. "
    "Production verification should fetch /search_index.json directly (hard refresh or new tab)."
)


def _control_root(root: Path | None = None) -> Path:
    from release_control_plane import control_plane_root

    return control_plane_root(root)


def publication_record_path(release_id: str, *, root: Path | None = None) -> Path:
    return _control_root(root) / "state" / "pbj320_publications" / SOURCE_ID / f"{release_id}.json"


def load_publication_record(release_id: str, *, root: Path | None = None) -> dict[str, Any] | None:
    path = publication_record_path(release_id, root=root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _write_publication_record(root: Path, release_id: str, payload: dict[str, Any]) -> None:
    path = publication_record_path(release_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _manifest_sha256(manifest_path: Path) -> str:
    return sha256_file(manifest_path)


def commit_destination_paths(manifest: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for row in manifest.get("artifacts") or []:
        if row.get("publication_class") in {"commit_destination", "shared_derived"} and row.get("path"):
            paths.append(str(row["path"]).replace("\\", "/"))
    if not paths:
        paths.extend(str(p).replace("\\", "/") for p in (manifest.get("files_added") or []))
        paths.extend(str(p).replace("\\", "/") for p in (manifest.get("files_modified") or []))
    return sorted(set(paths))


def validation_only_paths(manifest: dict[str, Any]) -> set[str]:
    out: set[str] = set()
    for row in manifest.get("artifacts") or []:
        if row.get("publication_class") == "validation_parity" and row.get("path"):
            out.add(str(row["path"]).replace("\\", "/"))
    out.update(str(p).replace("\\", "/") for p in (manifest.get("validation_artifacts") or []))
    return out


def preflight_pi_publish_completeness() -> dict[str, Any]:
    """Trace PI-dependent production paths (read-only). Does not mutate."""
    rows = [
        {
            "artifact": "provider_info/ProviderInfoNorm_*.csv",
            "consumer": "public provider pages (/provider/*), MCP get_facility PI fields",
            "classification": "already_staged",
        },
        {
            "artifact": "provider_info_combined_latest.csv",
            "consumer": "ownership legal-name crosswalk, combined fallbacks",
            "classification": "already_staged",
        },
        {
            "artifact": "data/state_page_aggregates.json.gz",
            "consumer": "state pages (/state/*)",
            "classification": "already_staged",
            "note": "Also rebuilt on Render via build_state_page_aggregates.py",
        },
        {
            "artifact": "search_index.json",
            "consumer": "public facility search (public-search.js → /search_index.json)",
            "classification": "already_staged",
            "note": (
                "generate_search_index.py reads latest Provider Info snapshot/combined_latest; "
                "not in render.yaml buildCommand; must be committed with PI release."
            ),
        },
        {
            "artifact": "data/provider_indexes/*",
            "consumer": "provider cold-path HPRD percentiles",
            "classification": "deterministically_generated_by_render",
            "note": "Built from facility_quarterly_metrics.csv (PBJ nurse), not Provider Info Stage",
        },
        {
            "artifact": "provider_info/NH_ProviderInfo_*.csv",
            "consumer": "local parity gates; generate_search_index input when present",
            "classification": "validation_only_never_commit",
        },
    ]
    blocking = [
        r
        for r in rows
        if r["classification"]
        not in {
            "already_staged",
            "deterministically_generated_by_render",
            "runtime_loaded_from_staged_inputs",
            "validation_only_never_commit",
        }
    ]
    return {
        "pass": len(blocking) == 0,
        "artifacts": rows,
        "search_index_cache_note": SEARCH_INDEX_CACHE_NOTE,
    }


def validate_publish_ready(
    *,
    root: Path | None = None,
    pbj_root: Path | str | None = None,
    release_id: str | None = None,
) -> dict[str, Any]:
    root = root or cms_data_paths.repo_root()
    dev_pbj_root = _resolve_pbj_root(pbj_root)
    preflight = preflight_pi_publish_completeness()
    if not preflight.get("pass"):
        raise ProviderInfoPublishError("preflight completeness check failed")

    active = get_active_release(SOURCE_ID, registry_path(root)) or {}
    active_release_id = str(active.get("active_release_id") or "")
    release_id = release_id or active_release_id
    if not release_id:
        raise ProviderInfoPublishError("no ACTIVE Provider Information release")

    manifest = load_stage_manifest(release_id, root=root)
    if not manifest or manifest.get("status") != STAGE_STATUS_STAGED:
        raise ProviderInfoPublishError(
            f"manifest status must be STAGED (got {manifest and manifest.get('status')})"
        )
    manifest_base_sha = str(manifest.get("publication_base_sha") or "")
    if not manifest_base_sha:
        raise ProviderInfoPublishError(
            "manifest missing publication_base_sha; refresh Stage against production baseline"
        )
    if str(manifest.get("active_release_id") or "") != release_id:
        raise ProviderInfoPublishError("manifest active_release_id mismatch")

    canonical_sha = str(active.get("hash") or "")
    if str(manifest.get("canonical", {}).get("sha256") or "") != canonical_sha:
        raise ProviderInfoPublishError("canonical hash drift vs ACTIVE registry")
    canonical_path = str(active.get("source_uri") or "")
    if canonical_path:
        from pbj320_stage_provider_info import _file_uri_to_path

        path = _file_uri_to_path(canonical_path)
        if not path.is_file() or sha256_file(path) != canonical_sha:
            raise ProviderInfoPublishError("canonical artifact no longer CURRENT on disk")

    gates = manifest.get("validation_gates") or []
    if not gates or not all(g.get("passed") for g in gates):
        raise ProviderInfoPublishError("Stage validation gates missing or not all PASS")

    commit_paths = commit_destination_paths(manifest)
    if not commit_paths:
        raise ProviderInfoPublishError("no commit destinations in manifest")

    validation_only = validation_only_paths(manifest)
    drift: list[str] = []
    path_rows: list[dict[str, Any]] = []
    path_hashes: dict[str, str] = {}

    pub = load_publication_record(release_id, root=root)
    branch, remote = resolve_publish_branch(dev_pbj_root)
    current_base_sha = fetch_publish_base(dev_pbj_root, remote=remote, branch=branch)
    if current_base_sha != manifest_base_sha:
        raise ProviderInfoPublishError(
            f"{remote}/{branch} advanced since Stage ({manifest_base_sha[:12]}… → {current_base_sha[:12]}…); "
            "refresh Stage against latest production before Publish"
        )
    publish_base_sha = current_base_sha

    from pbj320_publication import stage_artifact_cache_path

    artifact_cache = stage_artifact_cache_path(SOURCE_ID, release_id, root=root)
    if not artifact_cache.is_dir():
        raise ProviderInfoPublishError(f"stage artifact cache missing: {artifact_cache}")

    for row in manifest.get("artifacts") or []:
        rel = str(row.get("path") or "").replace("\\", "/")
        if row.get("publication_class") not in {"commit_destination", "shared_derived"}:
            continue
        expected = str(row.get("proposed_sha256") or "")
        actual = _sha256_or_none(artifact_cache / rel.replace("/", os.sep))
        if not actual:
            drift.append(f"missing: {rel}")
        elif expected and actual != expected:
            drift.append(f"sha drift: {rel}")
        if expected:
            path_hashes[rel] = expected
        path_rows.append({"path": rel, "expected_sha256": expected, "actual_sha256": actual})

    if drift:
        raise ProviderInfoPublishError("destination drift since Stage: " + "; ".join(drift))

    return {
        "source_id": SOURCE_ID,
        "release_id": release_id,
        "manifest_path": str(stage_manifest_path(release_id, root=root)),
        "manifest_sha256": _manifest_sha256(stage_manifest_path(release_id, root=root)),
        "manifest_publication_base_sha": manifest_base_sha,
        "commit_paths": commit_paths,
        "validation_only_paths": sorted(validation_only),
        "commit_file_count": len(commit_paths),
        "validation_gates_pass": True,
        "destination_layers": {
            "canonical": "CURRENT",
            "pbj320_destination": "STAGED",
            "committed": "YES" if pub and pub.get("commit_sha") else "NO",
            "pushed": "YES" if pub and pub.get("push_succeeded") else "NO",
            "deployed": "UNKNOWN",
            "production_verified": "NO",
        },
        "publish_branch": branch,
        "publish_remote": remote,
        "publish_base_sha": publish_base_sha,
        "dev_pbj_root": str(dev_pbj_root),
        "stage_artifact_cache": str(artifact_cache),
        "paths": path_rows,
        "path_hashes": path_hashes,
        "preflight": preflight,
        "already_published": bool(pub and pub.get("push_succeeded")),
        "search_index_cache_note": SEARCH_INDEX_CACHE_NOTE,
    }


def _pi_validation_commands(release_id: str) -> list[str]:
    return [
        "python scripts/validate_release.py",
        f"python scripts/verify_provider_release_handoff.py --release-key {release_id}",
    ]


def _production_verification_targets() -> list[dict[str, str]]:
    return [
        {
            "check": "provider_norm_month",
            "description": "Known Aug 2026 ProviderInfoNorm field visible on /provider/<ccn>",
        },
        {
            "check": "search_index_direct",
            "description": (
                "GET /search_index.json shows Aug 2026 facility metadata "
                "(bypass sessionStorage cache)"
            ),
        },
        {
            "check": "state_page_aggregates",
            "description": "State page reflects Aug 2026 processing_date",
        },
        {
            "check": "combined_latest_crosswalk",
            "description": "Ownership legal-name lookup uses Aug 2026 combined_latest",
        },
    ]


def publish_provider_info_for_pbj320(
    *,
    root: Path | None = None,
    pbj_root: Path | str | None = None,
    release_id: str | None = None,
    confirm: bool = False,
    dry_run: bool = False,
    push: bool = True,
    publish_base_sha: str | None = None,
) -> dict[str, Any]:
    root = root or cms_data_paths.repo_root()
    dev_pbj_root = _resolve_pbj_root(pbj_root)
    review = validate_publish_ready(root=root, pbj_root=dev_pbj_root, release_id=release_id)
    release_id = str(review["release_id"])
    manifest = load_stage_manifest(release_id, root=root) or {}

    def _write(payload: dict[str, Any]) -> None:
        payload.setdefault("production_verification_targets", _production_verification_targets())
        payload.setdefault("search_index_cache_note", SEARCH_INDEX_CACHE_NOTE)
        _write_publication_record(root, release_id, payload)

    adapter = PublicationAdapter(
        source_id=SOURCE_ID,
        release_id=release_id,
        manifest=manifest,
        dev_pbj_root=dev_pbj_root,
        commit_paths=list(review["commit_paths"]),
        path_hashes=dict(review["path_hashes"]),
        validation_only_paths=set(review["validation_only_paths"]),
        validation_commands=_pi_validation_commands(release_id),
        commit_message=f"Publish Provider Information {release_id}",
        stage_manifest_sha256=str(review.get("manifest_sha256") or ""),
        publication_record_path=publication_record_path(release_id, root=root),
        write_publication_record=_write,
        load_publication_record=lambda: load_publication_record(release_id, root=root),
        already_published=bool(review.get("already_published")),
    )

    from pbj320_publication import stage_artifact_cache_path as _cache_path

    artifact_cache = Path(str(review.get("stage_artifact_cache") or _cache_path(SOURCE_ID, release_id, root=root)))

    result = publish_from_adapter(
        adapter,
        confirm=confirm,
        dry_run=dry_run,
        push=push,
        publish_base_sha=publish_base_sha or str(review.get("publish_base_sha") or ""),
        artifact_source_root=artifact_cache,
        root=root,
    )
    result["review"] = review
    return result


def build_publish_review_context(
    *,
    root: Path | None = None,
    pbj_root: Path | str | None = None,
    release_id: str | None = None,
) -> dict[str, Any]:
    review = validate_publish_ready(root=root, pbj_root=pbj_root, release_id=release_id)
    manifest = load_stage_manifest(str(review["release_id"]), root=root) or {}
    return {
        "review": review,
        "manifest": manifest,
        "release_id": review["release_id"],
        "preflight": review.get("preflight"),
        "search_index_cache_note": SEARCH_INDEX_CACHE_NOTE,
    }
