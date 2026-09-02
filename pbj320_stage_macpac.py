"""MACPAC state staffing standards — reference vintage contract (no standalone Publish)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cms_data_paths
from active_release_registry import get_active_release, registry_path, sha256_file

from pbj320_publication_contract import PUBLICATION_CONTRACT_VERSION
from pbj320_stage_common import atomic_write_json, resolve_pbj_root, stage_manifest_dir

SOURCE_ID = "cms.macpac_state_staffing"
REFERENCE_VINTAGE = "March 2022 compendium"
REFERENCE_RELEASE_ID = "2022-03"

GRAPH = (
    "MACPAC March 2022 compendium (reference vintage) → macpac_state_standards_clean.csv → "
    "state page min-staffing context + chart footnotes (consumers); "
    "PBJ nurse/compliance freshness follows nurse quarter upstream — MACPAC age is not staleness"
)

REFERENCE_PATHS = (
    "macpac_state_standards_clean.csv",
    "data/state_standards/macpac_xlsx_extract.json",
)

CONSUMERS = (
    "app.get_macpac_hprd_for_state",
    "app.get_macpac_chart_info",
    "state page staffing footnotes",
    "scripts/build_state_page_aggregates.py (baseline context only when present)",
)


def _reference_paths(pbj_root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for rel in REFERENCE_PATHS:
        path = pbj_root / rel.replace("\\", "/").replace("/", os.sep)
        if path.is_file():
            rows.append((rel.replace("\\", "/"), path))
    return rows


def macpac_reference_contract(*, root: Path | None = None, pbj_root: Path | None = None) -> dict[str, Any]:
    root = root or cms_data_paths.repo_root()
    dev_pbj_root = resolve_pbj_root(pbj_root)

    active = get_active_release("macpac.state_staffing_standards", registry_path(root)) or {}
    active_hash = str(active.get("hash") or "")
    active_release_id = str(active.get("active_release_id") or REFERENCE_RELEASE_ID)

    present = _reference_paths(dev_pbj_root)
    primary_rel, primary_path = present[0] if present else ("macpac_state_standards_clean.csv", dev_pbj_root / "macpac_state_standards_clean.csv")
    fingerprint = sha256_file(primary_path) if primary_path.is_file() else None

    return {
        "status": "REFERENCE_CURRENT",
        "implementation_status": "REFERENCE_ONLY",
        "schema_version": 2,
        "publication_contract_version": PUBLICATION_CONTRACT_VERSION,
        "source_id": SOURCE_ID,
        "active_release_id": active_release_id,
        "reference_vintage": REFERENCE_VINTAGE,
        "reference_fingerprint_sha256": fingerprint or active_hash,
        "governed_registry_hash": active_hash,
        "reference_paths": [{"path": rel, "sha256": sha256_file(path)} for rel, path in present],
        "destinations": [
            {
                "role": "macpac_state_staffing_csv",
                "path": primary_rel,
                "publication_class": "reference_static",
                "git_committed": primary_path.is_file(),
            }
        ],
        "consumers": list(CONSUMERS),
        "freshness_semantics": {
            "macpac_reference": "REFERENCE_CURRENT — March 2022 compendium; age alone is not staleness",
            "downstream_pbj_nurse": "follows cms.pbj_nurse_staffing ACTIVE quarter when consumed with nurse metrics",
            "benchmarks_peer_distributions": "DERIVED — freshness follows actual upstreams",
        },
        "destination_layers": {
            "canonical": "REFERENCE",
            "pbj320_destination": "REFERENCE",
            "committed": "YES" if primary_path.is_file() else "UNKNOWN",
            "pushed": "N/A",
            "deployed": "N/A",
            "production_verified": "N/A",
        },
        "source_destination_graph": GRAPH,
        "publishable": False,
        "next_human_step": "No standalone MACPAC Publish — reference is versioned static data in pbj-root Git.",
    }


def stage_macpac_for_pbj320(*, release_id: str | None = None, root: Path | None = None) -> dict[str, Any]:
    """Persist reference contract manifest; no Stage candidate artifacts."""
    root = root or cms_data_paths.repo_root()
    contract = macpac_reference_contract(root=root)
    rel_id = release_id or contract.get("active_release_id") or REFERENCE_RELEASE_ID
    out_dir = stage_manifest_dir(SOURCE_ID, root=root)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{rel_id}.json"
    contract["active_release_id"] = rel_id
    atomic_write_json(out_path, contract)
    contract["manifest_path"] = str(out_path)
    return contract
