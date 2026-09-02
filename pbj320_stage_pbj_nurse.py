"""PBJ nurse staffing quarterly Stage for PBJ320 (baseline+overlay Git metrics)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cms_data_paths
from active_release_registry import get_active_release, registry_path, sha256_file

from pbj320_stage_common import (
    StageError,
    artifact_row,
    baseline_input_row,
    destination_pre_state_at_base,
    finalize_stage_manifest,
    load_stage_manifest,
    overlay_input,
    resolve_pbj_root,
    resolve_pbjapp_root,
    run_python_script,
    sha256_or_none,
    stage_manifest_path,
    write_fail_manifest,
)

SOURCE_ID = "cms.pbj_nurse_staffing"

GRAPH = (
    "ACTIVE quarter (cms.pbj_nurse_staffing) → facility/state/national quarterly CSVs + latest_quarter_data.json (Git) → "
    "Render: build_facility_provider_indexes + compliance bundle (deploy_generated) → "
    "shared: state_page_aggregates"
)

GIT_METRICS = (
    ("facility_quarterly_metrics", "facility_quarterly_metrics.csv"),
    ("state_quarterly_metrics", "state_quarterly_metrics.csv"),
    ("national_quarterly_metrics", "national_quarterly_metrics.csv"),
    ("latest_quarter_data", "latest_quarter_data.json"),
)

RENDER_BUILD_COMMANDS = (
    "scripts/build_facility_provider_indexes.py",
    "scripts/ensure_staffing_compliance_bundle.py",
    "scripts/build_staffing_compliance_runtime_index.py",
)


def _resolve_overlay_sources(dev_pbj_root: Path, pbjapp_root: Path, release_id: str) -> dict[str, Path]:
    """Resolve governed overlay files — prefer pbj-root dev checkout (post-sync), not dirty partials."""
    out: dict[str, Path] = {}
    for _role, rel in GIT_METRICS:
        dev_path = dev_pbj_root / rel.replace("/", os.sep)
        if dev_path.is_file():
            out[rel] = dev_path
            continue
        app_path = pbjapp_root / rel.replace("/", os.sep)
        if app_path.is_file():
            out[rel] = app_path
    if release_id and "latest_quarter_data.json" not in out:
        stub = dev_pbj_root / "latest_quarter_data.json"
        if not stub.is_file():
            stub.parent.mkdir(parents=True, exist_ok=True)
            stub.write_text(
                json.dumps({"quarter": release_id, "source": "pbj320_stage_pbj_nurse"}, indent=2) + "\n",
                encoding="utf-8",
            )
        out["latest_quarter_data.json"] = stub
    missing = [rel for _role, rel in GIT_METRICS if rel not in out]
    if missing:
        raise StageError(f"missing governed nurse metrics inputs: {', '.join(missing)}")
    return out


def _collect_baseline_inputs(dev_pbj_root: Path, publication_base_sha: str) -> list[dict[str, Any]]:
    from pbj320_publication import fingerprint_baseline_path

    rows: list[dict[str, Any]] = []
    macpac_rel = "macpac_state_standards_clean.csv"
    fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, macpac_rel)
    if fp.get("present_at_base"):
        fp["source_id"] = "cms.macpac_state_staffing"
        fp["role"] = "macpac_reference"
        rows.append(baseline_input_row(fp))

    proc = __import__("subprocess").run(
        ["git", "-C", str(dev_pbj_root), "ls-tree", "-r", "--name-only", publication_base_sha, "provider_info/"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        norms = sorted(line.strip() for line in (proc.stdout or "").splitlines() if "ProviderInfoNorm_" in line)
        if norms:
            fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, norms[-1])
            fp["source_id"] = "cms.provider_info"
            fp["role"] = "provider_norm"
            rows.append(baseline_input_row(fp))
    return rows


def _run_state_aggregates(baseline_wt: Path) -> tuple[str | None, dict[str, Any]]:
    script = baseline_wt / "scripts" / "build_state_page_aggregates.py"
    gate = run_python_script(script, cwd=baseline_wt, label="python scripts/build_state_page_aggregates.py")
    rel = "data/state_page_aggregates.json.gz"
    digest = sha256_or_none(baseline_wt / rel.replace("/", os.sep))
    gate["passed"] = bool(gate.get("passed")) and digest is not None
    return digest, gate


def _verification_contract(release_id: str) -> dict[str, Any]:
    return {
        "release_id": release_id,
        "checks": [
            {
                "kind": "latest_quarter",
                "path": "latest_quarter_data.json",
                "field": "quarter",
                "expected": release_id,
            },
            {
                "kind": "facility_metrics_row",
                "ccn": "335513",
                "path": "facility_quarterly_metrics.csv",
                "quarter_column": "CY_Qtr",
                "expected_quarter": release_id,
            },
            {
                "kind": "state_page_aggregates",
                "surface": "/data/state_page_aggregates.json.gz",
                "note": "Shared derived includes nurse quarter overlay",
            },
        ],
    }


def stage_pbj_nurse_for_pbj320(
    *,
    release_id: str | None = None,
    root: Path | None = None,
    pbj_root: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    from pbj320_publication import (
        fetch_publish_base,
        prepare_baseline_worktree,
        resolve_publish_branch,
        stage_artifact_cache_path,
        stage_baseline_worktree_path,
        sync_stage_artifacts_to_cache,
    )

    root = root or cms_data_paths.repo_root()
    dev_pbj_root = resolve_pbj_root(pbj_root)
    pbjapp_root = resolve_pbjapp_root()

    active = get_active_release(SOURCE_ID, registry_path(root)) or {}
    active_release_id = str(active.get("active_release_id") or "")
    if not active_release_id:
        raise StageError("cms.pbj_nurse_staffing has no ACTIVE release")
    if release_id and release_id != active_release_id:
        raise StageError(f"requested release_id {release_id} != ACTIVE {active_release_id}")

    commit_rels = [rel for _role, rel in GIT_METRICS] + ["data/state_page_aggregates.json.gz"]

    publish_branch, publish_remote = resolve_publish_branch(dev_pbj_root)
    publication_base_sha = fetch_publish_base(dev_pbj_root, remote=publish_remote, branch=publish_branch)
    baseline_wt = stage_baseline_worktree_path(SOURCE_ID, active_release_id, root=root)
    artifact_cache = stage_artifact_cache_path(SOURCE_ID, active_release_id, root=root)

    pre_manifest = load_stage_manifest(SOURCE_ID, active_release_id, root=root)
    if (
        pre_manifest
        and pre_manifest.get("status") == "STAGED"
        and not force
        and str(pre_manifest.get("publication_base_sha") or "") == publication_base_sha
    ):
        return {
            "status": "NO_MATERIAL_DIFF",
            "active_release_id": active_release_id,
            "manifest_path": str(stage_manifest_path(SOURCE_ID, active_release_id, root=root)),
            "manifest": pre_manifest,
        }

    overlay_sources = _resolve_overlay_sources(dev_pbj_root, pbjapp_root, active_release_id)
    baseline_pre = {
        rel: destination_pre_state_at_base(dev_pbj_root, publication_base_sha, rel) for rel in commit_rels
    }
    baseline_inputs = _collect_baseline_inputs(dev_pbj_root, publication_base_sha)

    artifacts: list[dict[str, Any]] = []
    gate_results: list[dict[str, Any]] = []
    files_added: list[str] = []
    files_modified: list[str] = []
    validation_artifacts: list[str] = []

    def _record(row: dict[str, Any]) -> None:
        rel = str(row.get("path") or "")
        action = str(row.get("publication_action") or "")
        pub_class = str(row.get("publication_class") or "")
        if pub_class == "deploy_generated":
            if rel:
                validation_artifacts.append(rel)
            return
        if action == "add" and rel:
            files_added.append(rel)
        elif action == "modify" and rel:
            files_modified.append(rel)

    nurse_overlay = overlay_input(
        source_id=SOURCE_ID,
        release_id=active_release_id,
        rel_path="facility_quarterly_metrics.csv",
        sha256=str(active.get("hash") or sha256_file(overlay_sources["facility_quarterly_metrics.csv"])),
        role="facility_quarterly_metrics",
    )

    try:
        prepare_baseline_worktree(
            dev_pbj_root,
            remote=publish_remote,
            branch=publish_branch,
            base_sha=publication_base_sha,
            worktree_path=baseline_wt,
        )

        overlay_digests: dict[str, str] = {}
        for role, rel in GIT_METRICS:
            src = overlay_sources[rel]
            dest = baseline_wt / rel.replace("/", os.sep)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())
            overlay_digests[rel] = sha256_file(dest)

        agg_sha, agg_gate = _run_state_aggregates(baseline_wt)
        gate_results.append(agg_gate)
        if not agg_gate.get("passed"):
            raise StageError("build_state_page_aggregates failed")

        for role, rel in GIT_METRICS:
            row = artifact_row(
                rel_path=rel,
                pre_state=baseline_pre[rel],
                proposed_sha=overlay_digests.get(rel),
                expected_release_id=active_release_id,
                transformation="governed nurse quarterly overlay (Git commit destination)",
                role=role,
                publication_class="commit_destination",
                publication_base_sha=publication_base_sha,
                inputs=[nurse_overlay],
            )
            artifacts.append(row)
            _record(row)

        agg_rel = "data/state_page_aggregates.json.gz"
        agg_inputs = [nurse_overlay] + baseline_inputs
        agg_row = artifact_row(
            rel_path=agg_rel,
            pre_state=baseline_pre[agg_rel],
            proposed_sha=agg_sha,
            expected_release_id=active_release_id,
            transformation="scripts/build_state_page_aggregates.py",
            role="state_page_aggregates",
            publication_class="shared_derived",
            publication_base_sha=publication_base_sha,
            inputs=agg_inputs,
        )
        artifacts.append(agg_row)
        _record(agg_row)

        for cmd in RENDER_BUILD_COMMANDS:
            validation_artifacts.append(cmd)
        artifacts.append(
            {
                "destination_id": "provider_indexes",
                "publication_class": "deploy_generated",
                "publication_action": "validation-only",
                "path": "data/provider_indexes/ (Render buildCommand)",
                "transformation": "; ".join(RENDER_BUILD_COMMANDS),
                "inputs": [nurse_overlay],
            }
        )

        sync_stage_artifacts_to_cache(
            baseline_wt=baseline_wt,
            cache_dir=artifact_cache,
            rel_paths=[rel for _role, rel in GIT_METRICS] + [agg_rel],
        )

        return finalize_stage_manifest(
            source_id=SOURCE_ID,
            release_id=active_release_id,
            publication_base_sha=publication_base_sha,
            publish_remote=publish_remote,
            publish_branch=publish_branch,
            baseline_wt=baseline_wt,
            artifact_cache=artifact_cache,
            dev_pbj_root=dev_pbj_root,
            artifacts=artifacts,
            gate_results=gate_results,
            files_added=files_added,
            files_modified=files_modified,
            validation_artifacts=validation_artifacts,
            verification_contract=_verification_contract(active_release_id),
            export_commands=[
                "PBJapp/scripts/sync_to_pbj_root.py pbj-metrics",
                "python scripts/build_state_page_aggregates.py",
                "Render: " + " && ".join(RENDER_BUILD_COMMANDS),
            ],
            graph=GRAPH,
            root=root,
            extra={
                "render_build_audit": {
                    "git_committed": [rel for _role, rel in GIT_METRICS],
                    "render_generated": list(RENDER_BUILD_COMMANDS),
                }
            },
        )
    except Exception as exc:
        write_fail_manifest(
            source_id=SOURCE_ID,
            release_id=active_release_id,
            error=str(exc),
            root=root,
            publication_base_sha=publication_base_sha,
        )
        raise
