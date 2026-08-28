"""Operator-facing freshness layers and candidate audits (presentation + safe cleanup)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cms_data_paths
from active_release_registry import get_active_release, registry_path
from cms_source_registry import get_source
from release_source_catalog import UpdateMechanism


def audit_nurse_staffing_candidate_state(
    *,
    control_row: dict[str, Any] | None,
    root: Path | None = None,
) -> dict[str, Any]:
    """Explain ACTIVE + pending when both share CY2026Q1 (or any same quarter label)."""
    root = root or cms_data_paths.repo_root()
    active = (control_row or {}).get("active") or {}
    pending = (control_row or {}).get("pending") or {}
    active_id = str(active.get("active_release_id") or "")
    pending_id = str(pending.get("release_id") or "")
    pending_state = str(pending.get("state") or "").upper()

    if not active_id or not pending_id or active_id != pending_id:
        return {
            "same_quarter": False,
            "is_redundant_reacquisition": False,
            "summary": None,
        }

    active_uri = str(active.get("source_uri") or "")
    pending_uri = str(pending.get("source_uri") or "")
    active_hash = str(active.get("hash") or "")
    pending_hash = str(pending.get("hash") or "")

    acquisition_path = root / "PBJcsv" / "_manifests" / active_id / "acquisition.json"
    acquisition: dict[str, Any] = {}
    if acquisition_path.is_file():
        try:
            acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            acquisition = {}

    same_bytes = bool(active_hash and pending_hash and active_hash == pending_hash)
    active_is_standardized = "standardized_PBJ" in active_uri.replace("\\", "/")
    pending_is_raw = "PBJcsv" in pending_uri.replace("\\", "/") and "standardized" not in pending_uri.replace("\\", "/")

    is_redundant = (
        pending_state == "ACQUIRED"
        and str(active.get("status") or "").upper() == "ACTIVE"
        and not same_bytes
        and active_is_standardized
        and pending_is_raw
    )

    if same_bytes:
        kind = "duplicate_candidate"
        summary = (
            f"Pending {pending_id} matches ACTIVE bytes — redundant candidate, not a new CMS quarter."
        )
    elif is_redundant:
        kind = "reacquisition_same_quarter"
        summary = (
            f"Re-acquired {pending_id} raw CSV ({pending_hash[:12]}…) while ACTIVE already governs "
            f"standardized {active_id} ({active_hash[:12]}…). CMS did not publish a new quarter — "
            f"this is a repeat download awaiting validation, not a newer vintage."
        )
    else:
        kind = "same_label_distinct_artifact"
        summary = (
            f"ACTIVE and pending both labeled {active_id} but artifacts differ "
            f"(ACTIVE {active_hash[:12]}… vs pending {pending_hash[:12]}…)."
        )

    return {
        "same_quarter": True,
        "same_bytes": same_bytes,
        "is_redundant_reacquisition": is_redundant or same_bytes,
        "kind": kind,
        "summary": summary,
        "active_hash": active_hash,
        "pending_hash": pending_hash,
        "active_uri": active_uri,
        "pending_uri": pending_uri,
        "cms_dataset_id": acquisition.get("stable_cms_dataset_id"),
        "cms_source_url": acquisition.get("source_url"),
        "acquired_at": pending.get("detected_at") or acquisition.get("acquired_at"),
        "promoted_at": active.get("promoted_at"),
    }


def count_stale_citation_packages(*, root: Path | None = None) -> tuple[int, int]:
    """Return (stale_count, checked_count) for facility citation slices vs ACTIVE registry."""
    from citation_packages_rebuild import count_stale_citation_packages as _count

    return _count(root=root)


def _pbjapp_root() -> Path:
    configured = (os.environ.get("PBJ_REPO_ROOT") or "").strip()
    if configured:
        return Path(configured).resolve()
    return cms_data_paths.repo_root()


def inventory_fields_for_source(
    source_id: str,
    *,
    availability: dict[str, Any],
    check_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Table column semantics: CMS latest vs built-from upstream for derived artifacts."""
    from release_source_catalog import BY_ID, UpdateMechanism

    catalog = BY_ID.get(source_id)
    mechanism = catalog.mechanism if catalog else None

    if mechanism == UpdateMechanism.DERIVED:
        upstream = catalog.upstream if catalog else ()
        upstream_active = (check_row or {}).get("upstream_active") or availability.get("upstream_active") or {}
        labels: list[str] = []
        for upstream_id in upstream:
            release_id = upstream_active.get(upstream_id)
            if release_id:
                labels.append(f"{upstream_id.split('.')[-1]} {release_id}")
        built_from = ", ".join(labels) if labels else "—"
        derived_status = (check_row or {}).get("status") or availability.get("derived_upstream_status")
        provenance_missing = bool(
            (check_row or {}).get("provenance_missing") or availability.get("provenance_missing")
        )
        if derived_status == "UNKNOWN" or provenance_missing:
            return {
                "inventory_axis": "upstream",
                "inventory_label": "Built from",
                "inventory_value": built_from,
                "inventory_status": "UNKNOWN",
                "inventory_detail": "Upstream build provenance not recorded",
            }
        stale = bool((check_row or {}).get("new_release_available") or availability.get("new_release_available"))
        return {
            "inventory_axis": "upstream",
            "inventory_label": "Built from",
            "inventory_value": built_from,
            "inventory_status": "STALE" if stale else "CURRENT",
            "inventory_detail": "Derived artifact — freshness follows upstream ACTIVE releases",
        }

    if mechanism == UpdateMechanism.STATIC_CONFIGURATION:
        return {
            "inventory_axis": "none",
            "inventory_label": "Version",
            "inventory_value": availability.get("active_release_id") or "—",
            "inventory_status": None,
            "inventory_detail": "Static configuration",
        }

    if mechanism == UpdateMechanism.MANUAL_VERSIONED:
        return {
            "inventory_axis": "none",
            "inventory_label": "Version",
            "inventory_value": availability.get("active_release_label") or availability.get("active_release_id") or "—",
            "inventory_status": None,
            "inventory_detail": "Manually versioned reference",
        }

    pub_label = availability.get("publisher_latest_label") or availability.get("publisher_latest_release_id")
    return {
        "inventory_axis": "cms",
        "inventory_label": "CMS latest",
        "inventory_value": pub_label or "—",
        "inventory_status": None,
        "inventory_detail": None,
    }


def build_freshness_layers(
    source_id: str,
    *,
    workflow: dict[str, Any],
    root: Path | None = None,
) -> list[dict[str, Any]]:
    """Three-layer operator view: CMS source → canonical build → consumer packages."""
    root = root or cms_data_paths.repo_root()
    release_availability = workflow.get("release_availability") or {}
    provenance = workflow.get("provenance_freshness") or {}
    pending_state = str(workflow.get("pending_state") or "").upper()
    layers: list[dict[str, Any]] = []

    cms_current = (
        not release_availability.get("new_release_available")
        and pending_state not in {"ACQUIRED", "VALIDATED", "DETECTED"}
    )
    active_label = release_availability.get("active_release_label") or workflow.get("active_release_id") or "—"
    cms_latest = release_availability.get("publisher_latest_label") or "—"
    cms_detail = f"{active_label} active"
    if release_availability.get("unchanged_in_latest_publication"):
        cms_detail += f" · CMS latest {cms_latest} · unchanged"
    elif release_availability.get("new_release_available"):
        cms_detail = f"CMS latest {cms_latest} available · ACTIVE {active_label}"
        cms_current = False
    elif pending_state == "ACQUIRED" and workflow.get("pending_release_id"):
        cms_detail = f"ACTIVE {active_label} · pending candidate {workflow.get('pending_release_id')}"
        cms_current = False

    layers.append(
        {
            "key": "cms_source",
            "label": "CMS source",
            "status": "current" if cms_current else "attention",
            "status_label": "Current" if cms_current else "Needs attention",
            "detail": cms_detail,
        }
    )

    if source_id == "cms.health_citations":
        active_id = workflow.get("active_release_id")
        canonical_current = True
        canonical_detail = "National Health Citations file matches ACTIVE registry"
        if active_id:
            from health_citations_acquire import citations_artifact_path

            path = citations_artifact_path(active_id, root=_pbjapp_root())
            if not path.is_file():
                canonical_current = False
                canonical_detail = f"Missing canonical artifact for ACTIVE {active_id}"
            else:
                canonical_detail = path.name
        layers.append(
            {
                "key": "canonical",
                "label": "Canonical data",
                "status": "current" if canonical_current else "attention",
                "status_label": "Current" if canonical_current else "Missing",
                "detail": canonical_detail,
            }
        )
        stale_count, checked = count_stale_citation_packages(root=root)
        if checked:
            consumer_current = stale_count == 0
            layers.append(
                {
                    "key": "consumers",
                    "label": "Facility packages",
                    "status": "current" if consumer_current else "attention",
                    "status_label": "Current" if consumer_current else f"{stale_count} stale",
                    "detail": (
                        f"{checked} facility citation tables checked"
                        if consumer_current
                        else f"{stale_count} of {checked} facility citation tables lag ACTIVE national file"
                    ),
                    "stale_count": stale_count,
                    "checked_count": checked,
                }
            )
    elif source_id in {"cms.snf_all_owners", "cms.snf_enrollments"}:
        from ownership_downstream_rebuild import audit_ownership_downstream_stale

        audit = audit_ownership_downstream_stale(root=root)
        canonical_current = not audit.get("is_stale")
        layers.append(
            {
                "key": "canonical",
                "label": "Ownership bridge & policy",
                "status": "current" if canonical_current else "attention",
                "status_label": "Current" if canonical_current else "Stale",
                "detail": (
                    f"Bridge lookup and policy at {audit.get('release_label') or audit.get('release_id')}"
                    if canonical_current
                    else "; ".join(audit.get("blocking_reasons") or [])[:200]
                ),
            }
        )
        layers.append(
            {
                "key": "consumers",
                "label": "Packaged bundles",
                "status": "attention",
                "status_label": "May lag",
                "detail": "Facility PACKAGE_MANIFEST ownership provenance may trail canonical rebuild until repackage",
            }
        )
    else:
        downstream_rows = provenance.get("downstream_artifacts") or []
        stale_rows = [row for row in downstream_rows if row.get("freshness") == "STALE"]
        if downstream_rows:
            layers.append(
                {
                    "key": "consumers",
                    "label": "Downstream outputs",
                    "status": "current" if not stale_rows else "attention",
                    "status_label": "Current" if not stale_rows else f"{len(stale_rows)} stale",
                    "detail": ", ".join(
                        row.get("operator_label") or row.get("consumer") or row.get("artifact") or "output"
                        for row in (stale_rows or downstream_rows[:3])
                    ),
                }
            )

    return layers
