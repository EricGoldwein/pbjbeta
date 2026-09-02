"""Read-only production verification for PBJ320 publications."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cms_data_paths
from active_release_registry import sha256_file

import pbj320_verify_core as verify_core
from pbj320_publication import stage_artifact_cache_path
from pbj320_publish_provider_info import (
    load_publication_record,
    publication_record_path,
)
from pbj320_stage_provider_info import SOURCE_ID, load_stage_manifest

ProviderInfoVerifyError = RuntimeError

DEFAULT_PRODUCTION_ORIGIN = "https://www.pbj320.com"


def _production_origin() -> str:
    origin = (os.environ.get("PBJ320_PRODUCTION_ORIGIN") or DEFAULT_PRODUCTION_ORIGIN).strip()
    return origin.rstrip("/")


def verify_provider_info_production(
    release_id: str,
    *,
    root: Path | None = None,
    pbj_root: Path | str | None = None,
    production_origin: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Read-only verification for a pushed Provider Information publication."""
    root = root or cms_data_paths.repo_root()
    pub = load_publication_record(release_id, root=root)
    if not pub or not pub.get("push_succeeded"):
        raise ProviderInfoVerifyError("publication record missing or not pushed")

    manifest = load_stage_manifest(release_id, root=root)
    if not manifest:
        raise ProviderInfoVerifyError("stage manifest missing")

    dev_pbj_root = verify_core.resolve_dev_pbj_root_for_verify(pbj_root, pub)
    origin = (production_origin or _production_origin()).rstrip("/")

    commit_sha = str(pub.get("commit_sha") or "")
    remote = str(pub.get("push_remote") or "origin")
    branch = str(pub.get("push_branch") or "master")
    checks: list[dict[str, Any]] = []

    on_branch = verify_core.git_commit_on_branch(dev_pbj_root, commit_sha, remote=remote, branch=branch)
    checks.append(
        verify_core.check_row(
            check_id="commit_on_production_branch",
            target=f"{remote}/{branch}",
            expected=commit_sha,
            actual="present" if on_branch else "missing",
            passed=on_branch,
        )
    )

    cache = verify_core.resolve_stage_artifact_cache_dir(manifest)
    if cache is None:
        cache = stage_artifact_cache_path(SOURCE_ID, release_id, root=root)
    expected_search_path = cache / "search_index.json"
    expected_search_sha = sha256_file(expected_search_path) if expected_search_path.is_file() else None
    expected_index: dict[str, Any] = {}
    if expected_search_path.is_file():
        try:
            expected_index = json.loads(expected_search_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            expected_index = {}

    search_url = f"{origin}/search_index.json"
    status, body = verify_core.http_get_bytes(search_url)
    live_search_sha = verify_core.search_index_sha(body)
    search_sha_ok = bool(expected_search_sha and live_search_sha == expected_search_sha and status == 200)
    checks.append(
        verify_core.check_row(
            check_id="search_index_sha",
            target=search_url,
            expected=expected_search_sha or "—",
            actual=live_search_sha or f"HTTP {status}",
            passed=search_sha_ok,
            artifact_sha=live_search_sha or None,
        )
    )

    live_index = verify_core.load_search_index_json(body)

    processing_prefix = verify_core.release_processing_prefix(release_id) or ""
    baseline_sha = str(manifest.get("publication_base_sha") or pub.get("publish_base_sha") or "")

    candidates, norm_diff_provenance = verify_core.prepare_norm_release_diff_candidates(
        manifest=manifest,
        release_id=release_id,
        dev_pbj_root=dev_pbj_root,
        publication_record=pub,
    )
    baseline_norm_rows = norm_diff_provenance["baseline"].get("row_count", 0)
    expected_norm_rows = norm_diff_provenance["expected"].get("row_count", 0)

    if not candidates and expected_index:
        baseline_index_bytes = verify_core.git_show_bytes(dev_pbj_root, baseline_sha, "search_index.json") if baseline_sha else None
        baseline_index = verify_core.load_search_index_json(baseline_index_bytes or b"")
        candidates = verify_core.derive_search_index_diff_candidates(
            expected_index=expected_index,
            baseline_index=baseline_index,
        )

    data_ok, evidence_ccn, evidence_field, evidence_surfaces = verify_core.verify_data_level_release_diff(
        origin=origin,
        candidates=candidates,
        live_index=live_index,
    )

    if not data_ok and search_sha_ok and expected_index:
        baseline_index_bytes = verify_core.git_show_bytes(dev_pbj_root, baseline_sha, "search_index.json") if baseline_sha else None
        baseline_index = verify_core.load_search_index_json(baseline_index_bytes or b"")
        fallback = verify_core.derive_search_index_diff_candidates(
            expected_index=expected_index,
            baseline_index=baseline_index,
        )
        data_ok, evidence_ccn, evidence_field, evidence_surfaces = verify_core.verify_data_level_release_diff(
            origin=origin,
            candidates=fallback,
            live_index=live_index,
        )

    checks.append(
        verify_core.check_row(
            check_id="data_level_release_diff_visible",
            target=f"{origin}/search_index.json; {origin}/api/public/provider/<ccn>.json",
            expected=f"≥1 release diff visible ({processing_prefix})",
            actual=(
                f"ccn={evidence_ccn} field={evidence_field} via {evidence_surfaces[0]}"
                if data_ok
                else (evidence_surfaces[0] if evidence_surfaces else "no candidates")
            ),
            passed=data_ok,
        )
    )

    expected_agg_path = cache / "data" / "state_page_aggregates.json.gz"
    expected_agg_sha = sha256_file(expected_agg_path) if expected_agg_path.is_file() else None
    if expected_agg_sha:
        checks.append(
            verify_core.check_row(
                check_id="shared_derived_staged_aggregates_present",
                target=str(expected_agg_path),
                expected=expected_agg_sha,
                actual="present",
                passed=True,
                artifact_sha=expected_agg_sha,
            )
        )

    all_pass = all(c.get("result") == "PASS" for c in checks)
    verified_at = checks[0]["verification_timestamp"] if checks else None

    result = {
        "source_id": SOURCE_ID,
        "release_id": release_id,
        "production_origin": origin,
        "commit_sha": commit_sha,
        "checks": checks,
        "all_pass": all_pass,
        "production_verified": all_pass,
        "verification_timestamp": verified_at,
        "dry_run": dry_run,
        "candidate_count": len(candidates),
        "baseline_norm_rows": baseline_norm_rows,
        "expected_norm_rows": expected_norm_rows,
        "norm_diff_provenance": norm_diff_provenance,
    }

    def _write_record(record: dict[str, Any]) -> None:
        path = publication_record_path(release_id, root=root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    verify_core.persist_publication_verification(pub, checks=checks, dry_run=dry_run, write_record=_write_record)

    return result
