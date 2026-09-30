"""Minimal provenance / freshness contract for governed sources and downstream artifacts.

Reuses active_releases.json, release_candidates.json, theme publication discovery,
and existing derived-consumer maps — no new provenance database.

Contract fields (every governed source / downstream artifact row):
  publisher                  — cms_source_registry publisher
  cms_publication_id         — PDC theme archive publication id
  cms_publication_date       — theme archive drop date (publication axis)
  processing_modified_date   — CMS dataset modified_date from theme manifest
  product_release_id         — YYYY-MM product vintage (ACTIVE / pending / manifest)
  cms_dataset_id             — PDC dataset uuid
  canonical_artifact_path    — local governed file path or active source_uri
  sha256                     — byte hash from registry or on-disk artifact
  acquired_at                — download / adopt timestamp
  validated_at               — structural validation timestamp
  activated_at               — ACTIVE promotion timestamp (promoted_at)
  downstream_artifacts[]     — dependent pipelines with freshness CURRENT|STALE
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cms_data_paths
from active_release_registry import get_active_release, registry_path, sha256_file
from cms_source_registry import get_source
from cms_theme_publication import ThemePublication, publication_availability_for_source
from release_control_plane import (
    DERIVED_ARTIFACT_PIPELINES,
    DEPENDENCY_GRAPH,
    load_candidates,
    stale_derived_consumers,
)


def _publisher_for_source(source_id: str) -> str:
    record = get_source(source_id)
    if record is None:
        return "unknown"
    return record.publisher.value


def _canonical_artifact_path(active: dict[str, Any] | None, *, source_id: str, root: Path) -> str | None:
    if active and active.get("source_uri"):
        return str(active["source_uri"])
    if source_id == "cms.health_citations":
        from health_citations_acquire import citations_artifact_path

        active_id = (active or {}).get("active_release_id")
        if active_id:
            path = citations_artifact_path(active_id, root=root)
            return str(path) if path.is_file() else None
    if source_id == "cms.provider_info":
        pi = cms_data_paths.provider_info_dir(root)
        if pi.is_dir():
            candidates = sorted(pi.glob("NH_ProviderInfo_*.csv"), key=lambda p: p.name)
            if candidates:
                return str(candidates[-1])
    return None


def _pbjapp_root() -> Path:
    configured = (os.environ.get("PBJ_REPO_ROOT") or "").strip()
    if configured:
        return Path(configured).resolve()
    return cms_data_paths.repo_root()


def _health_citations_slice_stale_capabilities(root: Path) -> list[str]:
    """True when facility citation slices lag the ACTIVE national Health Citations file."""
    pbj_root = _pbjapp_root()
    try:
        from citation_packages_rebuild import citation_slice_needs_rebuild
    except ImportError:
        return []

    deployments = pbj_root / "deployments"
    if not deployments.is_dir():
        return []

    checked = False
    for dep in sorted(deployments.glob("pbj320-*")):
        ccn = dep.name.removeprefix("pbj320-")
        cit_out = dep / f"facility_{ccn}_citations.csv"
        if not cit_out.is_file():
            continue
        checked = True
        needs_rebuild, _reason = citation_slice_needs_rebuild(pbj_root, str(cit_out))
        if needs_rebuild:
            return ["facility.citations"]

    active = get_active_release("cms.health_citations", registry_path(root))
    active_id = (active or {}).get("active_release_id")
    if active_id:
        from health_citations_acquire import citations_artifact_path

        if citations_artifact_path(active_id, root=pbj_root).is_file() and not checked:
            return ["facility.citations"]
    return []


def downstream_stale_capabilities_for_source(source_id: str, *, root: Path | None = None) -> list[str]:
    """Capabilities stale for one governed source (cross-source + artifact gate checks)."""
    root = root or cms_data_paths.repo_root()
    stale = list(stale_derived_consumers(root).get(source_id, []))
    if source_id == "cms.health_citations":
        for cap in _health_citations_slice_stale_capabilities(root):
            if cap not in stale:
                stale.append(cap)
    if source_id in {"cms.snf_all_owners", "cms.snf_enrollments"}:
        try:
            from ownership_downstream_rebuild import audit_ownership_downstream_stale

            audit = audit_ownership_downstream_stale(root=root)
            for cap in audit.get("stale_capabilities") or []:
                if cap not in stale:
                    stale.append(cap)
        except Exception:
            pass
    return stale


def ownership_downstream_stale_capabilities(*, root: Path | None = None) -> list[str]:
    """Stale ownership-derived capabilities for the consolidated operator row."""
    try:
        from ownership_downstream_rebuild import audit_ownership_downstream_stale

        return list(audit_ownership_downstream_stale(root=root).get("stale_capabilities") or [])
    except Exception:
        return []


def _downstream_artifact_rows(source_id: str, *, stale_capabilities: list[str]) -> list[dict[str, Any]]:
    from release_control_plane import CONSUMER_OPERATOR_LABELS

    stale_set = set(stale_capabilities)
    rows: list[dict[str, Any]] = []
    for pipe in DERIVED_ARTIFACT_PIPELINES.get(source_id, ()):
        consumer = pipe.get("consumer") or ""
        status = "STALE" if stale_set and any(part in consumer for part in stale_set) else "CURRENT"
        if stale_set:
            for cap in stale_set:
                if cap in consumer or cap in str(pipe.get("artifact") or ""):
                    status = "STALE"
                    break
        rows.append(
            {
                "artifact": pipe.get("artifact"),
                "rebuild": pipe.get("rebuild"),
                "consumer": consumer,
                "operator_label": CONSUMER_OPERATOR_LABELS.get(consumer, consumer),
                "layer": "consumer",
                "freshness": status,
            }
        )
    seen_consumers = {row.get("consumer") for row in rows}
    for capability in DEPENDENCY_GRAPH.get(source_id, ()):
        if capability in seen_consumers:
            continue
        rows.append(
            {
                "artifact": capability,
                "rebuild": None,
                "consumer": capability,
                "operator_label": CONSUMER_OPERATOR_LABELS.get(
                    capability,
                    capability.replace("facility.", "Facility ").replace("_", " ").title(),
                ),
                "layer": "capability",
                "freshness": "STALE" if capability in stale_set else "CURRENT",
            }
        )
    return rows


def build_source_provenance_freshness(
    source_id: str,
    *,
    root: Path | None = None,
    theme_publication: ThemePublication | None = None,
) -> dict[str, Any]:
    """Answer the minimal provenance/freshness contract for one governed source."""
    root = root or cms_data_paths.repo_root()
    active = get_active_release(source_id, registry_path(root))
    candidates = load_candidates(root).get("datasets", {})
    pending = candidates.get(source_id) if isinstance(candidates.get(source_id), dict) else None

    record = get_source(source_id)
    cms_dataset_id = record.cms_dataset_id if record else None

    theme_fields = publication_availability_for_source(
        source_id,
        active_release_id=(active or {}).get("active_release_id"),
        publication=theme_publication,
    ) or {}

    canonical_path = _canonical_artifact_path(active, source_id=source_id, root=root)
    sha256 = (active or {}).get("hash")
    if not sha256 and canonical_path:
        path = Path(canonical_path.replace("file:///", "").replace("file://", ""))
        if path.is_file():
            sha256 = sha256_file(path)

    stale_capabilities = downstream_stale_capabilities_for_source(source_id, root=root)
    downstream = _downstream_artifact_rows(
        source_id,
        stale_capabilities=stale_capabilities,
    )
    if source_id == "cms.provider_info":
        from pbj320_stage_provider_info import ProviderInfoStageError, audit_provider_info_pbj320_destination

        try:
            dest_audit = audit_provider_info_pbj320_destination(root=root)
        except ProviderInfoStageError as exc:
            dest_audit = {"destination_staged": False, "error": str(exc)}
        destination_staged = bool(dest_audit.get("destination_staged"))
        for row in downstream:
            if row.get("consumer") == "public PBJ320.com Provider surfaces":
                row["operator_label"] = "Public Provider pages (pbj-root destination)"
                row["freshness"] = "UNKNOWN" if dest_audit.get("error") else ("STAGED" if destination_staged else "NOT_STAGED")
                if dest_audit.get("error"):
                    row["detail"] = dest_audit["error"]
                break

    product_release_id = (
        (active or {}).get("active_release_id")
        or (pending or {}).get("release_id")
        or theme_fields.get("product_release_id")
    )

    return {
        "source_id": source_id,
        "publisher": _publisher_for_source(source_id),
        "cms_publication_id": theme_fields.get("cms_publication_id"),
        "cms_publication_date": theme_fields.get("cms_publication_date"),
        "processing_modified_date": theme_fields.get("processing_modified_date"),
        "product_release_id": product_release_id,
        "cms_dataset_id": cms_dataset_id or theme_fields.get("cms_dataset_id"),
        "canonical_artifact_path": canonical_path,
        "sha256": sha256,
        "acquired_at": (active or {}).get("downloaded_at") or (pending or {}).get("acquired_at"),
        "validated_at": (active or {}).get("validated_at")
        or ((pending or {}).get("validation") or {}).get("validated_at"),
        "activated_at": (active or {}).get("promoted_at"),
        "pending_release_id": (pending or {}).get("release_id"),
        "pending_state": (pending or {}).get("state"),
        "in_latest_publication": theme_fields.get("in_latest_publication"),
        "unchanged_in_latest_publication": theme_fields.get("unchanged_in_latest_publication"),
        "new_release_available": theme_fields.get("new_release_available"),
        "downstream_artifacts": downstream,
    }
