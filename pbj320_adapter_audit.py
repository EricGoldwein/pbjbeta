"""Source-specific publication graph audits (blockers + destination links)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cms_data_paths


def _resolve_pbj_root(explicit: Path | str | None = None) -> Path:
    from pbj320_stage_common import resolve_pbj_root

    return resolve_pbj_root(explicit)


def audit_ownership_pair_graph(*, release_id: str, root: Path | None = None) -> dict[str, Any]:
    root = root or cms_data_paths.repo_root()
    pbj_root = _resolve_pbj_root(None)
    from ownership_pairing import PAIR_SOURCE_ID, pairing_status

    pair = pairing_status(root)
    blockers: list[str] = list(pair.get("blocking_reasons") or [])
    active = pair.get("active") or {}
    if not active.get("aligned"):
        blockers.append("ACTIVE owners/enrollment releases not aligned")
    policy_path = pbj_root / "ownership" / "ownership_release_policy.json"
    if not policy_path.is_file():
        blockers.append(f"missing {policy_path.relative_to(pbj_root)} on pbj-root")
    graph = (
        "Owners ACTIVE + Enrollments ACTIVE → aligned canonical pair → "
        "ownership_release_policy + bridge lookup → Git ownership/ artifacts → "
        "Render owner indexes (state_owner_index, owner profiles) → entity/search consumers"
    )
    first_blocker = blockers[0] if blockers else ""
    return {
        "source_id": PAIR_SOURCE_ID,
        "release_id": release_id,
        "graph": graph,
        "blockers": blockers,
        "first_unproven_link": first_blocker or None,
        "pair_status": pair,
        "operational": not blockers,
    }


def audit_pbj_nurse_graph(*, release_id: str, root: Path | None = None) -> dict[str, Any]:
    _ = root
    pbj_root = _resolve_pbj_root(None)
    blockers: list[str] = []
    for rel in (
        "facility_quarterly_metrics.csv",
        "national_quarterly_metrics.csv",
        "latest_quarter_data.json",
    ):
        if not (pbj_root / rel).is_file():
            blockers.append(f"missing Git input {rel}")
    graph = (
        "ACTIVE quarter → canonical metrics CSV/JSON → Git deploy inputs → "
        "compliance/evidence → provider/state JSON indexes → benchmarks/shared derived"
    )
    return {
        "source_id": "cms.pbj_nurse_staffing",
        "release_id": release_id,
        "graph": graph,
        "blockers": blockers,
        "first_unproven_link": blockers[0] if blockers else None,
        "operational": not blockers,
    }


def audit_sff_graph(*, release_id: str, root: Path | None = None) -> dict[str, Any]:
    root = root or cms_data_paths.repo_root()
    from active_release_registry import get_active_release, registry_path

    blockers: list[str] = []
    active = get_active_release("cms.sff_pdf_list", registry_path(root)) or {}
    if not active.get("active_release_id"):
        blockers.append("cms.sff_pdf_list has no ACTIVE release")
    metadata = active.get("metadata") or {}
    if not metadata.get("pbj_handoff"):
        blockers.append("ACTIVE SFF metadata missing pbj_handoff table CSVs")
    graph = (
        "SFF ACTIVE → parsed SFF artifacts → public SFF JSON/CSV → "
        "search_index rebuild (retain baseline PI/chain) → provider/entity consumers"
    )
    return {
        "source_id": "cms.sff_pdf_list",
        "release_id": release_id,
        "graph": graph,
        "blockers": blockers,
        "first_unproven_link": blockers[0] if blockers else None,
        "operational": not blockers,
    }


def audit_macpac_graph(*, release_id: str = "2022-03", root: Path | None = None) -> dict[str, Any]:
    _ = root
    pbj_root = _resolve_pbj_root(None)
    ref_path = pbj_root / "macpac_state_standards_clean.csv"
    blockers: list[str] = []
    if not ref_path.is_file():
        blockers.append(f"missing {ref_path.relative_to(pbj_root)}")
    graph = (
        "MACPAC March 2022 compendium (reference) → macpac_state_standards_clean.csv → "
        "state policy context consumers; not independently published"
    )
    return {
        "source_id": "cms.macpac_state_staffing",
        "release_id": release_id,
        "graph": graph,
        "blockers": blockers,
        "first_unproven_link": blockers[0] if blockers else None,
        "operational": bool(ref_path.is_file()),
        "reference_only": True,
    }
