"""CMS SFF posting Stage for PBJ320 (baseline PI/chain + SFF overlay → search_index)."""

from __future__ import annotations

import calendar
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cms_data_paths
from active_release_registry import get_active_release, registry_path, sha256_file

from pbj320_stage_common import (
    SEARCH_INDEX_REL,
    StageError,
    artifact_row,
    baseline_input_row,
    destination_pre_state_at_base,
    finalize_stage_manifest,
    load_stage_manifest,
    overlay_input,
    resolve_pbj_root,
    run_python_script,
    sha256_or_none,
    stage_manifest_path,
    write_fail_manifest,
)

SOURCE_ID = "cms.sff_pdf_list"

GRAPH = (
    "SFF ACTIVE (cms.sff_pdf_list) → governed table CSVs + PDF → "
    "data/derived/sff/sff_facilities.json + pbj-wrapped/public/sff-facilities.json → "
    "generate_search_index.py (retains baseline Provider Info + chain) → search consumers"
)

SFF_TABLES = ("sff_table_a.csv", "sff_table_b.csv", "sff_table_c.csv", "sff_table_d.csv")


def _local_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise StageError(f"SFF ACTIVE artifact must be a local file URI, got {uri!r}")
    raw = unquote(parsed.path)
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return Path(raw)


def _sff_paths() -> dict[str, str]:
    return {
        "sff_facilities_json": "data/derived/sff/sff_facilities.json",
        "sff_public_json": "pbj-wrapped/public/sff-facilities.json",
        "current_release": "data_sources/cms/sff/current_release.json",
        "search_index": SEARCH_INDEX_REL,
    }


def _import_active_sff_overlay(baseline_wt: Path, *, root: Path, release_id: str, active: dict[str, Any]) -> str:
    metadata = active.get("metadata") or {}
    pdf = _local_uri(str(metadata.get("source_pdf_uri") or ""))
    pdf_hash = str(metadata.get("source_pdf_hash") or "")
    if not pdf.is_file():
        raise StageError(f"ACTIVE SFF PDF missing: {pdf}")
    if pdf_hash and sha256_file(pdf) != pdf_hash:
        raise StageError("ACTIVE SFF PDF hash mismatch vs registry")

    handoff = metadata.get("pbj_handoff") or []
    expected_roles = set(SFF_TABLES)
    roles = {str(item.get("role") or "") for item in handoff}
    if roles != expected_roles:
        raise StageError(f"ACTIVE SFF handoff incomplete: {sorted(roles)}")

    year, month = release_id.split("-", 1)
    pdf_name = f"sff-posting-with-candidate-list-{calendar.month_name[int(month)].lower()}-{year}.pdf"
    raw_dir = baseline_wt / "data_sources" / "cms" / "sff" / "raw" / release_id
    table_dir = baseline_wt / "data" / "derived" / "sff" / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, raw_dir / pdf_name)

    for item in handoff:
        src = _local_uri(str(item.get("source_uri") or ""))
        role = str(item.get("role") or "")
        item_hash = str(item.get("hash") or "")
        if not src.is_file():
            raise StageError(f"SFF handoff source missing: {role}")
        if item_hash and sha256_file(src) != item_hash:
            raise StageError(f"SFF handoff hash mismatch: {role}")
        shutil.copy2(src, table_dir / role)

    manifest = {
        "dataset_id": SOURCE_ID,
        "release_id": release_id,
        "original_filename": pdf_name,
        "sha256": sha256_file(pdf),
        "active_registry": str(registry_path(root)),
    }
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return pdf_name


def _write_current_release(baseline_wt: Path, *, release_id: str, active: dict[str, Any], pdf_name: str) -> None:
    rel_path = baseline_wt / "data_sources" / "cms" / "sff" / "current_release.json"
    rel_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_release": release_id,
        "source_filename": pdf_name,
        "posting_period": (active.get("metadata") or {}).get("posting_period") or release_id,
        "source_url": (active.get("metadata") or {}).get("source_url"),
        "promoted_at": active.get("promoted_at"),
    }
    rel_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_build_sff_dataset(baseline_wt: Path) -> dict[str, Any]:
    script = baseline_wt / "scripts" / "sff" / "build_sff_dataset.py"
    if not script.is_file():
        return {
            "command": str(script),
            "exit_code": -1,
            "passed": False,
            "summary": "build_sff_dataset.py missing at publication_base_sha",
        }
    return run_python_script(script, cwd=baseline_wt, label="python scripts/sff/build_sff_dataset.py")


def _run_search_index(baseline_wt: Path) -> tuple[str | None, dict[str, Any]]:
    script = baseline_wt / "generate_search_index.py"
    gate = run_python_script(script, cwd=baseline_wt, label="python generate_search_index.py")
    digest = sha256_or_none(baseline_wt / SEARCH_INDEX_REL)
    gate["passed"] = bool(gate.get("passed")) and digest is not None
    return digest, gate


def _collect_baseline_inputs(dev_pbj_root: Path, baseline_wt: Path, publication_base_sha: str) -> list[dict[str, Any]]:
    from pbj320_publication import fingerprint_baseline_path, resolve_chain_performance_path, resolve_sff_facilities_path

    rows: list[dict[str, Any]] = []
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

    chain_rel = resolve_chain_performance_path(dev_pbj_root)
    if chain_rel:
        fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, chain_rel)
        fp["source_id"] = "chain_performance"
        fp["role"] = "chain_performance"
        rows.append(baseline_input_row(fp))

    baseline_sff = resolve_sff_facilities_path(baseline_wt)
    if baseline_sff:
        fp = fingerprint_baseline_path(dev_pbj_root, publication_base_sha, baseline_sff)
        fp["source_id"] = SOURCE_ID
        fp["role"] = "sff_facilities_baseline"
        rows.append(baseline_input_row(fp))
    return rows


def _copy_public_sff_json(baseline_wt: Path) -> str:
    src = baseline_wt / "data" / "derived" / "sff" / "sff_facilities.json"
    dest = baseline_wt / "pbj-wrapped" / "public" / "sff-facilities.json"
    if not src.is_file():
        raise StageError("sff_facilities.json missing after build_sff_dataset")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(src.read_bytes())
    return sha256_file(dest)


def _verification_contract(release_id: str, *, dataset_path: Path) -> dict[str, Any]:
    """Facility with SFF/Candidate status change vs prior baseline."""
    contract: dict[str, Any] = {
        "release_id": release_id,
        "checks": [
            {
                "kind": "search_index_sff_field",
                "ccn": "015009",
                "surface": "/search_index.json",
                "field": "s",
                "note": "Verify SFF letter code reflects overlay release",
            },
            {
                "kind": "sff_public_json",
                "surface": "/pbj-wrapped/public/sff-facilities.json",
                "field": "document_date.source_release",
                "expected": release_id,
            },
        ],
    }
    if dataset_path.is_file():
        try:
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            facilities = payload.get("facilities") or []
            candidate = next((f for f in facilities if str(f.get("category") or "") == "Candidate"), None)
            sff = next((f for f in facilities if str(f.get("category") or "") == "SFF"), None)
            contract["sample_facilities"] = {
                "candidate_ccn": (candidate or {}).get("provider_number"),
                "sff_ccn": (sff or {}).get("provider_number"),
            }
        except json.JSONDecodeError:
            pass
    return contract


def stage_sff_for_pbj320(
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

    active = get_active_release(SOURCE_ID, registry_path(root)) or {}
    active_release_id = str(active.get("active_release_id") or "")
    if not active_release_id:
        raise StageError("cms.sff_pdf_list has no ACTIVE release")
    if release_id and release_id != active_release_id:
        raise StageError(f"requested release_id {release_id} != ACTIVE {active_release_id}")

    rel_paths = _sff_paths()
    commit_rels = [
        rel_paths["sff_facilities_json"],
        rel_paths["sff_public_json"],
        rel_paths["current_release"],
        rel_paths["search_index"],
    ]

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

    baseline_pre = {
        rel: destination_pre_state_at_base(dev_pbj_root, publication_base_sha, rel) for rel in commit_rels
    }
    baseline_inputs = _collect_baseline_inputs(dev_pbj_root, baseline_wt, publication_base_sha)
    sff_overlay = overlay_input(
        source_id=SOURCE_ID,
        release_id=active_release_id,
        rel_path=rel_paths["sff_facilities_json"],
        sha256=str(active.get("hash") or ""),
        role="sff_facilities_json",
    )

    artifacts: list[dict[str, Any]] = []
    gate_results: list[dict[str, Any]] = []
    files_added: list[str] = []
    files_modified: list[str] = []
    validation_artifacts: list[str] = []

    def _record(row: dict[str, Any]) -> None:
        rel = str(row.get("path") or "")
        action = str(row.get("publication_action") or "")
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
        pdf_name = _import_active_sff_overlay(baseline_wt, root=root, release_id=active_release_id, active=active)
        _write_current_release(baseline_wt, release_id=active_release_id, active=active, pdf_name=pdf_name)

        build_gate = _run_build_sff_dataset(baseline_wt)
        gate_results.append(build_gate)
        if not build_gate.get("passed"):
            raise StageError(f"build_sff_dataset failed: {build_gate.get('summary')}")

        public_sha = _copy_public_sff_json(baseline_wt)
        derived_sha = sha256_or_none(baseline_wt / rel_paths["sff_facilities_json"].replace("/", os.sep))
        current_sha = sha256_or_none(baseline_wt / rel_paths["current_release"].replace("/", os.sep))

        validate_script = baseline_wt / "scripts" / "sff" / "validate_sff_dataset.py"
        if validate_script.is_file():
            val_gate = run_python_script(validate_script, cwd=baseline_wt, label="python scripts/sff/validate_sff_dataset.py")
            gate_results.append(val_gate)
            if not val_gate.get("passed"):
                raise StageError(f"validate_sff_dataset failed: {val_gate.get('summary')}")

        search_sha, search_gate = _run_search_index(baseline_wt)
        gate_results.append(search_gate)
        if not search_gate.get("passed"):
            raise StageError("generate_search_index failed")

        search_inputs = [sff_overlay] + [row for row in baseline_inputs if row.get("source_id") != SOURCE_ID]

        for role, rel, digest, pub_class, transform in (
            ("sff_facilities_json", rel_paths["sff_facilities_json"], derived_sha, "commit_destination", "scripts/sff/build_sff_dataset.py"),
            ("sff_public_json", rel_paths["sff_public_json"], public_sha, "commit_destination", "mirror derived → pbj-wrapped/public"),
            ("current_release", rel_paths["current_release"], current_sha, "commit_destination", "data_sources/cms/sff/current_release.json"),
        ):
            row = artifact_row(
                rel_path=rel,
                pre_state=baseline_pre[rel],
                proposed_sha=digest,
                expected_release_id=active_release_id,
                transformation=transform,
                role=role if role != "current_release" else "sff_current_release",
                publication_class=pub_class,
                publication_base_sha=publication_base_sha,
                inputs=[sff_overlay],
            )
            artifacts.append(row)
            _record(row)

        search_row = artifact_row(
            rel_path=rel_paths["search_index"],
            pre_state=baseline_pre[rel_paths["search_index"]],
            proposed_sha=search_sha,
            expected_release_id=active_release_id,
            transformation="generate_search_index.py (baseline PI/chain + SFF overlay)",
            role="search_index",
            publication_class="shared_derived",
            publication_base_sha=publication_base_sha,
            inputs=search_inputs,
        )
        artifacts.append(search_row)
        _record(search_row)

        sync_stage_artifacts_to_cache(baseline_wt=baseline_wt, cache_dir=artifact_cache, rel_paths=commit_rels)

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
            verification_contract=_verification_contract(
                active_release_id,
                dataset_path=baseline_wt / rel_paths["sff_facilities_json"].replace("/", os.sep),
            ),
            export_commands=[
                "scripts/sff/import_active_release.py (governed ACTIVE only)",
                "python scripts/sff/build_sff_dataset.py",
                "python generate_search_index.py",
            ],
            graph=GRAPH,
            root=root,
            extra={"sff_overlay_source": "governed ACTIVE registry (not dev-tree artifacts)"},
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
