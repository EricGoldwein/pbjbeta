"""Owners+Enrollments aligned-pair Stage for PBJ320 (baseline+overlay, no partial publish)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cms_data_paths
from active_release_registry import sha256_file
from ownership_pairing import ENROLLMENTS, OWNERS, PAIR_SOURCE_ID, pairing_status

from ownership_downstream_rebuild import (
    BRIDGE_SUBDIR,
    ENROLL_DOWNLOAD_REL,
    OWNERS_DOWNLOAD_REL,
    PAIRING_MANIFEST_REL,
    OwnershipRebuildError,
    _active_pair_release,
    _copy_active_source,
    _ensure_pairing_manifest_row,
    _pbjapp_root,
    _run_build_release_lookup,
    _update_ownership_release_policy,
)
from pbj320_stage_common import (
    StageError,
    artifact_row,
    baseline_input_row,
    destination_pre_state_at_base,
    file_uri_to_path,
    finalize_stage_manifest,
    load_stage_manifest,
    overlay_input,
    resolve_pbj_root,
    run_python_script,
    sha256_or_none,
    stage_manifest_path,
    write_fail_manifest,
)

SOURCE_ID = PAIR_SOURCE_ID

GRAPH = (
    "Owners ACTIVE + Enrollments ACTIVE (aligned pair) → canonical CSV pair → "
    "ownership_release_policy.json + bridge lookup + enrollment raw artifact (Git) → "
    "Render: build_snf_owners_index / build_owners_database (deploy_generated) → "
    "entity/search consumers"
)

PAIRING_MANIFEST_FIELDS = [
    "ownership_source_filename",
    "ownership_release_date",
    "enrollment_source_filename",
    "enrollment_release_date",
    "enrollment_raw_relative_path",
    "pairing_status",
    "pairing_rationale",
]

VERIFICATION_CCNS = ("335513", "335581")


def ownership_input_provenance(source_id: str, active: dict[str, Any], *, root: Path) -> dict[str, Any]:
    """Publisher observation bound to exact ACTIVE bytes, separate from pair key."""
    from release_check import load_check_state
    check = next((row for row in load_check_state(root).get('datasets', [])
                  if row.get('dataset_id') == source_id), {})
    if not (check.get('cms_release_vintage') and check.get('snapshot_date') and
            check.get('cms_dataset_version_id') and check.get('publisher_file_uuid') and check.get('publisher_url') and
            check.get('status') == 'CURRENT' and check.get('publisher_checked_at') and
            check.get('publisher_sha256') == active.get('hash')):
        raise StageError(f'{source_id}: Check CMS must verify current raw bytes before ownership staging')
    return {
        'cms_release_vintage': check.get('cms_release_vintage'),
        'snapshot_date': check.get('snapshot_date'),
        'cms_dataset_version_id': check.get('cms_dataset_version_id'),
        'cms_dataset_version_label': check.get('cms_dataset_version_label'),
        'cms_dataset_version_modified': check.get('cms_dataset_version_modified'),
        'cms_file_uuid': check.get('publisher_file_uuid'),
        'cms_publisher_url': check.get('publisher_url'),
        'cms_source_sha256': check.get('publisher_sha256'),
        'publisher_checked_at': check.get('publisher_checked_at'),
        'acquired_at': active.get('downloaded_at'),
    }


def _ownership_paths(release_id: str, *, owners_name: str, enroll_name: str) -> dict[str, str]:
    bridge_rel = f"ownership/_derived/cms_snf_ownership_ccn_bridge/release_{release_id}_lookup.json"
    return {
        "ownership_release_policy": "ownership/ownership_release_policy.json",
        "ownership_bridge_lookup": bridge_rel,
        "enrollment_release_artifact": str(ENROLL_DOWNLOAD_REL / enroll_name).replace("\\", "/"),
        "owners_download": str(OWNERS_DOWNLOAD_REL / owners_name).replace("\\", "/"),
        "owners_staged": f"ownership/{owners_name}",
        "pairing_manifest": str(PAIRING_MANIFEST_REL).replace("\\", "/"),
    }


def _ensure_pairing_manifest_template(worktree: Path) -> None:
    manifest_path = worktree / PAIRING_MANIFEST_REL
    if manifest_path.is_file():
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIRING_MANIFEST_FIELDS)
        writer.writeheader()


def _apply_ownership_overlay(
    baseline_wt: Path,
    *,
    release_id: str,
    owners_active: dict[str, Any],
    enroll_active: dict[str, Any],
) -> dict[str, str]:
    owners_name = Path(str(owners_active.get("source_filename") or "")).name
    enroll_name = Path(str(enroll_active.get("source_filename") or "")).name
    if not owners_name or not enroll_name:
        raise StageError("ACTIVE ownership records missing source_filename")

    rels = _ownership_paths(release_id, owners_name=owners_name, enroll_name=enroll_name)
    _copy_active_source(str(owners_active.get("source_uri")), baseline_wt / rels["owners_download"].replace("/", os.sep))
    _copy_active_source(str(enroll_active.get("source_uri")), baseline_wt / rels["enrollment_release_artifact"].replace("/", os.sep))
    _copy_active_source(str(owners_active.get("source_uri")), baseline_wt / rels["owners_staged"].replace("/", os.sep))

    _ensure_pairing_manifest_template(baseline_wt)
    _ensure_pairing_manifest_row(baseline_wt, release_id, owners=owners_active, enroll=enroll_active)

    # The canonical ownership bridge builder is tracked in PBJapp, not pbj-root.
    # Feed that builder the governed ACTIVE pair, then copy only its generated
    # publication artifact into the isolated pbj-root baseline candidate.
    builder_root = _pbjapp_root()
    builder_owners_download = builder_root / OWNERS_DOWNLOAD_REL / owners_name
    builder_enroll_download = builder_root / ENROLL_DOWNLOAD_REL / enroll_name
    builder_owners_staged = builder_root / "ownership" / owners_name

    _copy_active_source(str(owners_active.get("source_uri")), builder_owners_download)
    _copy_active_source(str(enroll_active.get("source_uri")), builder_enroll_download)
    _copy_active_source(str(owners_active.get("source_uri")), builder_owners_staged)
    _ensure_pairing_manifest_row(
        builder_root,
        release_id,
        owners=owners_active,
        enroll=enroll_active,
    )
    _run_build_release_lookup(builder_root, release_id)

    built_lookup = builder_root / BRIDGE_SUBDIR / f"release_{release_id}_lookup.json"
    if not built_lookup.is_file():
        raise OwnershipRebuildError(
            f"canonical bridge builder did not produce expected lookup: {built_lookup}"
        )

    candidate_lookup = baseline_wt / rels["ownership_bridge_lookup"].replace("/", os.sep)
    candidate_lookup.parent.mkdir(parents=True, exist_ok=True)
    candidate_lookup.write_bytes(built_lookup.read_bytes())

    _update_ownership_release_policy(
        baseline_wt,
        release_id,
        owners=owners_active,
        enroll=enroll_active,
    )
    return rels


def _collect_baseline_pi_input(dev_pbj_root: Path, baseline_wt: Path, publication_base_sha: str) -> dict[str, Any]:
    from pbj320_publication import fingerprint_baseline_path, resolve_chain_performance_path

    rows: list[dict[str, Any]] = []
    norm_glob = "provider_info/ProviderInfoNorm_*.csv"
    proc = __import__("subprocess").run(
        ["git", "-C", str(dev_pbj_root), "ls-tree", "-r", "--name-only", publication_base_sha, "provider_info/"],
        capture_output=True,
        text=True,
        check=False,
    )
    norm_rel = None
    if proc.returncode == 0:
        norms = sorted(line.strip() for line in (proc.stdout or "").splitlines() if "ProviderInfoNorm_" in line)
        norm_rel = norms[-1] if norms else None
    if norm_rel:
        fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, norm_rel)
        fp["source_id"] = "cms.provider_info"
        fp["role"] = "provider_norm"
        rows.append(baseline_input_row(fp))

    chain_rel = resolve_chain_performance_path(baseline_wt)
    if chain_rel:
        fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, chain_rel)
        fp["source_id"] = "chain_performance"
        fp["role"] = "chain_performance"
        rows.append(baseline_input_row(fp))
    return {"inputs": rows}


def _verification_contract(release_id: str, *, lookup_path: Path) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "release_id": release_id,
        "checks": [
            {
                "kind": "owner_facility_bridge",
                "ccns": list(VERIFICATION_CCNS),
                "surface": "/api/public/provider/{ccn}.json",
                "field": "ownership.enrollment_id",
            },
            {
                "kind": "policy_active_release",
                "path": "ownership/ownership_release_policy.json",
                "field": "active_release_date",
                "expected": release_id,
            },
        ],
        "bridge_lookup_path": str(lookup_path),
    }
    if lookup_path.is_file():
        try:
            payload = json.loads(lookup_path.read_text(encoding="utf-8"))
            by_ccn = payload.get("ccn_by_enrollment") or payload.get("facility_by_enrollment") or {}
            contract["sample_enrollment_ids"] = {
                ccn: by_ccn.get(ccn) for ccn in VERIFICATION_CCNS if isinstance(by_ccn, dict)
            }
        except json.JSONDecodeError:
            pass
    return contract


def stage_ownership_pair_for_pbj320(
    *,
    release_id: str | None = None,
    root: Path | None = None,
    pbj_root: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build ownership-pair publication candidate in isolated baseline worktree."""
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

    pair = pairing_status(root)
    if not (pair.get("active") or {}).get("aligned"):
        raise StageError(
            "Owners+Enrollments pair not aligned: "
            + "; ".join(pair.get("blocking_reasons") or ["ACTIVE releases differ"])
        )

    try:
        active_release_id, owners_active, enroll_active = _active_pair_release(root)
    except OwnershipRebuildError as exc:
        raise StageError(str(exc)) from exc

    if release_id and release_id != active_release_id:
        raise StageError(f"requested release_id {release_id} != aligned ACTIVE {active_release_id}")

    owners_provenance = ownership_input_provenance(OWNERS, owners_active, root=root)
    enroll_provenance = ownership_input_provenance(ENROLLMENTS, enroll_active, root=root)
    if owners_provenance['cms_release_vintage'] != enroll_provenance['cms_release_vintage']:
        raise StageError('Owners and Enrollments current CMS publication vintages differ; review the pair')

    publish_branch, publish_remote = resolve_publish_branch(dev_pbj_root)
    publication_base_sha = fetch_publish_base(dev_pbj_root, remote=publish_remote, branch=publish_branch)
    baseline_wt = stage_baseline_worktree_path(SOURCE_ID, active_release_id, root=root)
    artifact_cache = stage_artifact_cache_path(SOURCE_ID, active_release_id, root=root)

    owners_name = Path(str(owners_active.get("source_filename") or "")).name
    enroll_name = Path(str(enroll_active.get("source_filename") or "")).name
    rel_paths = _ownership_paths(active_release_id, owners_name=owners_name, enroll_name=enroll_name)
    commit_rels = [
        rel_paths["ownership_release_policy"],
        rel_paths["ownership_bridge_lookup"],
        rel_paths["enrollment_release_artifact"],
        rel_paths["owners_download"],
        rel_paths["owners_staged"],
        rel_paths["pairing_manifest"],
    ]

    pre_manifest = load_stage_manifest(SOURCE_ID, active_release_id, root=root)
    if (
        pre_manifest
        and pre_manifest.get("status") == "STAGED"
        and not force
        and str(pre_manifest.get("publication_base_sha") or "") == publication_base_sha
        and all(any(i.get('source_id') == source_id and all(i.get(key) == provenance.get(key)
                    for key in ('cms_release_vintage', 'snapshot_date', 'cms_dataset_version_id', 'cms_file_uuid', 'cms_publisher_url'))
                    for artifact in pre_manifest.get('artifacts', []) for i in artifact.get('inputs', []))
                for source_id, provenance in ((OWNERS, owners_provenance), (ENROLLMENTS, enroll_provenance)))
    ):
        cache_ok = all(
            (artifact_cache / rel.replace("/", os.sep)).is_file()
            for rel in commit_rels
            if (pre_manifest.get("artifacts") or [])
        )
        if cache_ok:
            return {
                "status": "NO_MATERIAL_DIFF",
                "active_release_id": active_release_id,
                "manifest_path": str(stage_manifest_path(SOURCE_ID, active_release_id, root=root)),
                "manifest": pre_manifest,
            }

    baseline_pre = {
        rel: destination_pre_state_at_base(dev_pbj_root, publication_base_sha, rel) for rel in commit_rels
    }
    artifacts: list[dict[str, Any]] = []
    gate_results: list[dict[str, Any]] = []
    files_added: list[str] = []
    files_modified: list[str] = []
    validation_artifacts: list[str] = []

    def _record(row: dict[str, Any]) -> None:
        rel = str(row.get("path") or "")
        action = str(row.get("publication_action") or "")
        pub_class = str(row.get("publication_class") or "")
        if pub_class in {"validation_parity", "deploy_generated"}:
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
        _apply_ownership_overlay(
            baseline_wt,
            release_id=active_release_id,
            owners_active=owners_active,
            enroll_active=enroll_active,
        )

        owners_overlay = overlay_input(
            source_id=OWNERS,
            release_id=active_release_id,
            rel_path=rel_paths["owners_download"],
            sha256=str(owners_active.get("hash") or sha256_file(file_uri_to_path(str(owners_active.get("source_uri"))))),
            role="owners_source_csv",
        )
        enroll_overlay = overlay_input(
            source_id=ENROLLMENTS,
            release_id=active_release_id,
            rel_path=rel_paths["enrollment_release_artifact"],
            sha256=str(enroll_active.get("hash") or ""),
            role="enrollment_release_artifact",
        )
        owners_overlay.update(owners_provenance)
        enroll_overlay.update(enroll_provenance)
        baseline_pi = _collect_baseline_pi_input(dev_pbj_root, baseline_wt, publication_base_sha)

        for role_key, pub_class in (
            ("ownership_release_policy", "commit_destination"),
            ("ownership_bridge_lookup", "commit_destination"),
            ("enrollment_release_artifact", "commit_destination"),
        ):
            rel = rel_paths[role_key]
            dest = baseline_wt / rel.replace("/", os.sep)
            digest = sha256_or_none(dest)
            inputs = [owners_overlay, enroll_overlay]
            if role_key == "ownership_bridge_lookup":
                inputs = [owners_overlay, enroll_overlay]
            row = artifact_row(
                rel_path=rel,
                pre_state=baseline_pre[rel],
                proposed_sha=digest,
                expected_release_id=active_release_id,
                transformation="ownership_downstream_rebuild (policy + bridge lookup)",
                role=role_key,
                publication_class=pub_class,
                publication_base_sha=publication_base_sha,
                inputs=inputs,
            )
            artifacts.append(row)
            _record(row)

        validate_script = baseline_wt / "scripts" / "validate_ownership_linkage.py"
        if validate_script.is_file():
            gate = run_python_script(validate_script, cwd=baseline_wt, label="python scripts/validate_ownership_linkage.py")
            gate_results.append(gate)
            if not gate.get("passed"):
                raise StageError(f"validate_ownership_linkage failed: {gate.get('summary')}")
        else:
            gate_results.append(
                {
                    "command": str(validate_script),
                    "exit_code": 0,
                    "passed": True,
                    "summary": "skipped (script absent at publication_base_sha)",
                }
            )

        deploy_note = {
            "destination_id": "owner_profile_index",
            "publication_class": "deploy_generated",
            "publication_action": "validation-only",
            "path": "data/state_owner_index.json.gz (Render build_snf_owners_index.py)",
            "transformation": "Render buildCommand — not Git-committed",
            "inputs": baseline_pi.get("inputs") or [],
        }
        artifacts.append(deploy_note)
        validation_artifacts.append(deploy_note["path"])

        sync_stage_artifacts_to_cache(
            baseline_wt=baseline_wt,
            cache_dir=artifact_cache,
            rel_paths=commit_rels,
        )

        lookup_path = baseline_wt / rel_paths["ownership_bridge_lookup"].replace("/", os.sep)
        result = finalize_stage_manifest(
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
            verification_contract=_verification_contract(active_release_id, lookup_path=lookup_path),
            export_commands=[
                "ownership_downstream_rebuild.rebuild_ownership_downstream (local policy + bridge)",
                "Render: build_snf_owners_index.py + build_owners_database.py",
            ],
            graph=GRAPH,
            root=root,
            extra={
                "pair_alignment": {
                    "owners_release": active_release_id,
                    "enrollment_release": active_release_id,
                    "owners_hash": owners_active.get("hash"),
                    "enrollments_hash": enroll_active.get("hash"),
                },
            },
        )
        return result
    except Exception as exc:
        write_fail_manifest(
            source_id=SOURCE_ID,
            release_id=active_release_id,
            error=str(exc),
            root=root,
            publication_base_sha=publication_base_sha,
        )
        raise
