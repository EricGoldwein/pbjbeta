"""Stage adapter dispatch for remaining public PBJ320 source families."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pbj320_adapter_audit import (
    audit_macpac_graph,
    audit_ownership_pair_graph,
    audit_pbj_nurse_graph,
    audit_sff_graph,
)
from pbj320_publication_contract import PUBLICATION_CONTRACT_VERSION
from pbj320_source_adapters import is_premium_only_source

NOT_READY_STATUS = "NOT_READY"
NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class StageAdapterNotImplementedError(RuntimeError):
    pass


def _not_ready_manifest(
    *,
    source_id: str,
    release_id: str,
    blocked_link: str,
    notes: str,
    blockers: list[str] | None = None,
    first_unproven_link: str | None = None,
) -> dict[str, Any]:
    return {
        "status": NOT_READY_STATUS,
        "implementation_status": NOT_IMPLEMENTED,
        "dry_run": True,
        "schema_version": 2,
        "publication_contract_version": PUBLICATION_CONTRACT_VERSION,
        "source_id": source_id,
        "active_release_id": release_id,
        "artifacts": [],
        "validation_gates": [],
        "destination_layers": {
            "canonical": "CURRENT",
            "pbj320_destination": NOT_READY_STATUS,
            "committed": "NO",
            "pushed": "NO",
            "deployed": "UNKNOWN",
            "production_verified": "NO",
        },
        "blocked_link": blocked_link,
        "blockers": blockers or [],
        "first_unproven_link": first_unproven_link,
        "next_human_step": notes,
        "publishable": False,
    }


def _not_ready_from_audit(audit: dict[str, Any], *, notes: str) -> dict[str, Any]:
    return _not_ready_manifest(
        source_id=str(audit.get("source_id") or ""),
        release_id=str(audit.get("release_id") or ""),
        blocked_link=str(audit.get("graph") or ""),
        notes=notes,
        blockers=list(audit.get("blockers") or []),
        first_unproven_link=audit.get("first_unproven_link"),
    )


def stage_ownership_pair_for_pbj320(*, release_id: str, root: Path | None = None) -> dict[str, Any]:
    from pbj320_stage_ownership import stage_ownership_pair_for_pbj320 as _stage

    try:
        return _stage(release_id=release_id, root=root)
    except Exception as exc:
        audit = audit_ownership_pair_graph(release_id=release_id, root=root)
        blockers = list(audit.get("blockers") or [])
        blockers.append(str(exc))
        return _not_ready_manifest(
            source_id=str(audit.get("source_id") or "cms.snf_ownership_pair"),
            release_id=release_id,
            blocked_link=str(audit.get("graph") or ""),
            notes=f"Owners+Enrollments NOT_READY: {exc}",
            blockers=blockers,
            first_unproven_link=str(exc),
        )


def stage_pbj_nurse_for_pbj320(*, release_id: str, root: Path | None = None) -> dict[str, Any]:
    from pbj320_stage_pbj_nurse import stage_pbj_nurse_for_pbj320 as _stage

    try:
        return _stage(release_id=release_id, root=root)
    except Exception as exc:
        audit = audit_pbj_nurse_graph(release_id=release_id, root=root)
        return _not_ready_manifest(
            source_id="cms.pbj_nurse_staffing",
            release_id=release_id,
            blocked_link=str(audit.get("graph") or ""),
            notes=f"PBJ nurse NOT_READY: {exc}",
            blockers=[str(exc)],
            first_unproven_link=str(exc),
        )


def stage_sff_for_pbj320(*, release_id: str, root: Path | None = None) -> dict[str, Any]:
    from pbj320_stage_sff import stage_sff_for_pbj320 as _stage

    try:
        return _stage(release_id=release_id, root=root)
    except Exception as exc:
        audit = audit_sff_graph(release_id=release_id, root=root)
        return _not_ready_manifest(
            source_id="cms.sff_pdf_list",
            release_id=release_id,
            blocked_link=str(audit.get("graph") or ""),
            notes=f"SFF NOT_READY: {exc}",
            blockers=[str(exc)],
            first_unproven_link=str(exc),
        )


def stage_macpac_for_pbj320(*, release_id: str, root: Path | None = None) -> dict[str, Any]:
    from pbj320_stage_macpac import stage_macpac_for_pbj320 as _stage

    return _stage(release_id=release_id, root=root)


def assert_not_premium_only(source_id: str) -> None:
    if is_premium_only_source(source_id):
        raise StageAdapterNotImplementedError(f"{source_id} is PREMIUM_ONLY / NO_PUBLIC_DESTINATION")
