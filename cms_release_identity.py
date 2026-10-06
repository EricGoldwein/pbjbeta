"""Read-only CMS raw-byte assessment shared by release adapters.

Dates describe periods; they never establish equality. Adapters must supply a
raw record bound to ACTIVE when its primary artifact has been transformed.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from active_release_registry import sha256_file
from release_control_plane import candidate_local_path

RAW_PRIMARY_SOURCES = {"cms.chain_performance", "cms.snf_all_owners", "cms.snf_enrollments", "cms.health_citations", "cms.nh_ownership"}


def active_raw_record(source_id: str, active: dict[str, Any]) -> dict[str, Any] | None:
    """Resolve existing evidence, without inventing historical provenance."""
    metadata = active.get("metadata") or {}
    if source_id in RAW_PRIMARY_SOURCES:
        return {"source_uri": active.get("source_uri"), "hash": active.get("hash")}
    if source_id == "cms.sff_pdf_list":
        return {"source_uri": metadata.get("source_pdf_uri"), "hash": metadata.get("source_pdf_hash")}
    raw = metadata.get("immutable_raw_source")
    if isinstance(raw, dict) and raw.get("normalized_sha256") == active.get("hash"):
        return raw
    if source_id == "cms.provider_info" and metadata.get("validation_evidence"):
        try:
            manifest = json.loads(Path(metadata["validation_evidence"]).read_text(encoding="utf-8"))
            active_path = candidate_local_path(active.get("source_uri"))
            for member in manifest.get("source_members") or []:
                outputs = member.get("normalized_outputs") or []
                bound = any(Path(o.get("path") or "").resolve() == active_path.resolve() and
                            o.get("sha256") == active.get("hash") for o in outputs) if active_path else False
                if not bound:
                    continue
                for output in outputs:
                    if output.get("sha256") == member.get("source_sha256"):
                        return {"source_uri": Path(output["path"]).resolve().as_uri(),
                                "hash": member["source_sha256"]}
        except (OSError, ValueError, KeyError, TypeError):
            return None
    return None


def assess_raw_identity(source_id: str, *, active: dict[str, Any], current: dict[str, Any],
                        fetch_bytes=None, remote_sha256: str | None = None,
                        raw_record: dict[str, Any] | None = None) -> dict[str, Any]:
    """Stream official bytes, never acquire/save/promote an artifact."""
    from generic_cms_csv import _remote_sha256, compare_publisher_artifact_identity
    common = {**current, "active_hash": active.get("hash"),
              "identity_check_version": 1, "publisher_checked_at": datetime.now(timezone.utc).isoformat()}
    resource = re.search(r"/resources/([^/]+)_([0-9]+)/", str(current.get("publisher_url") or ""))
    if resource:
        common.update(publisher_resource_id=resource.group(1), publisher_resource_version=resource.group(2))
    try:
        remote = remote_sha256 or _remote_sha256(current["publisher_url"], fetch_bytes)
        common["publisher_sha256"] = remote
        active_id = active.get("active_release_id")
        if not active_id:
            return {**common, "status": "NEWER", "new_release_available": True}
        metadata = active.get("metadata") or {}
        vintage = str(metadata.get("cms_release_vintage") or "")
        if not vintage and source_id in {"cms.snf_all_owners", "cms.snf_enrollments"}:
            vintage = str(metadata.get("cms_dataset_version_label") or "")[:7]
        newer = bool(vintage and current.get("cms_release_vintage") and current["cms_release_vintage"] > vintage)
        newer = newer or str(current.get("release_id") or "") > str(active_id)
        active_path = candidate_local_path(active.get("source_uri"))
        if not active_path or not active_path.is_file() or sha256_file(active_path) != active.get("hash"):
            return {**common, "status": "NEWER" if newer else "UNKNOWN", "new_release_available": True if newer else None,
                    "detail": "ACTIVE artifact is missing or differs from its governed SHA; currentness cannot be established"}
        raw = raw_record or active_raw_record(source_id, active)
        raw_path = candidate_local_path((raw or {}).get("source_uri"))
        local_sha = (raw or {}).get("hash")
        if not raw_path or not raw_path.is_file() or not local_sha or sha256_file(raw_path) != local_sha:
            if newer:
                return {**common, "status": "NEWER", "new_release_available": True,
                        "detail": "Authoritative CMS period advanced; ACTIVE raw-byte equivalence remains unverified"}
            return {**common, "status": "UNKNOWN", "new_release_available": None,
                    "detail": "Immutable raw source is not proven to correspond to ACTIVE; dates cannot establish currentness"}
        common["active_raw_sha256"] = local_sha
        if remote == local_sha:
            return {**common, "status": "CURRENT", "new_release_available": False,
                    "detail": "Current CMS distribution matches authoritative raw bytes bound to ACTIVE"}
        changes, _matches = compare_publisher_artifact_identity(metadata, current)
        return {**common, "status": "NEWER" if newer else "REVISED", "new_release_available": True,
                "publisher_revision_changed": True, "revision_identity_changes": list(dict.fromkeys(changes + ["source_sha256"])),
                "detail": "CMS distribution bytes differ from the immutable raw source bound to ACTIVE"}
    except Exception as exc:
        return {**common, "status": "ERROR", "new_release_available": None,
                "detail": f"CMS byte verification failed: {exc}"}
