"""Local PBJ release control-plane services (no Flask dependency)."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from active_release_registry import (
    ActiveReleaseError,
    get_active_release,
    load_registry,
    promote_release,
    registry_path,
    sha256_file,
)


class ReleaseState(str, Enum):
    DETECTED = "DETECTED"
    ACQUIRED = "ACQUIRED"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    SUPERSEDED = "SUPERSEDED"
    STALE = "STALE"
    FAILED = "FAILED"


# Derived from PBJapp premium_source_contract.py and the canonical builders.
DEPENDENCY_GRAPH: dict[str, tuple[str, ...]] = {
    "cms.pbj_nurse_staffing": (
        "facility.staffing",
        "benchmarks.state_county",
        "benchmarks.peer_distribution",
    ),
    "cms.pbj_non_nurse_staffing": ("facility.non_nurse_staffing",),
    "cms.provider_info": (
        "facility.provider_info",
        "facility.nh_ownership",
        "facility.citations",
        "benchmarks.state_county",
        "benchmarks.peer_distribution",
    ),
    "cms.health_citations": ("facility.citations",),
    "cms.snf_all_owners": ("facility.snf_owners", "ownership.enrollment_ccn_bridge"),
    "cms.snf_enrollments": ("ownership.enrollment_ccn_bridge",),
    "cms.sff_pdf_list": ("facility.sff_status",),
    "pbj.benchmarks.state": ("facility.benchmarks",),
    "pbj.benchmarks.national": ("facility.benchmarks",),
    "pbj.benchmarks.region": ("facility.benchmarks",),
    "pbj.benchmarks.region_mapping": ("facility.benchmarks",),
    "pbj.benchmarks.geo_cmi": ("facility.benchmarks",),
    "pbj.peer_distribution": ("facility.peer_distribution",),
    "macpac.state_staffing_standards": ("facility.macpac",),
}

CAPABILITY_LABELS = {
    "facility.staffing": "Staffing",
    "facility.non_nurse_staffing": "Non-nurse staffing",
    "facility.provider_info": "Provider Info",
    "facility.nh_ownership": "NH Ownership",
    "facility.citations": "Citations",
    "facility.snf_owners": "SNF All Owners",
    "ownership.enrollment_ccn_bridge": "Ownership bridge",
    "benchmarks.state_county": "County/state benchmarks",
    "benchmarks.peer_distribution": "Peer distribution",
    "facility.benchmarks": "Benchmarks",
    "facility.peer_distribution": "Peer distribution",
    "facility.macpac": "MACPAC",
    "facility.sff_status": "SFF status",
}


def _state_dir(root: Path | None = None) -> Path:
    return control_plane_root(root) / "state"


def control_plane_root(data_root: Path | None = None) -> Path:
    """Repository root whose state/ subtree holds release registries."""
    configured = (os.environ.get("PBJ_ACTIVE_RELEASE_REGISTRY") or "").strip()
    if configured:
        return Path(configured).resolve().parent.parent
    if data_root is not None:
        return data_root.resolve()
    return Path(__file__).resolve().parent


def candidates_path(root: Path | None = None) -> Path:
    return _state_dir(root) / "release_candidates.json"


def facility_index_path(root: Path | None = None) -> Path:
    return _state_dir(root) / "facility_release_status.json"


def source_health_path(root: Path | None = None) -> Path:
    return _state_dir(root) / "source_health.json"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_candidates(root: Path | None = None) -> dict[str, Any]:
    path = candidates_path(root)
    if not path.is_file():
        return {"schema_version": 1, "updated_at": None, "datasets": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("datasets"), dict):
        raise ActiveReleaseError(f"invalid candidate registry: {path}")
    return payload


def record_candidate(
    dataset_id: str,
    release_id: str,
    state: ReleaseState | str,
    *,
    source_path: str | Path | None = None,
    validation: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    state = ReleaseState(state)
    if state in {ReleaseState.ACTIVE, ReleaseState.SUPERSEDED, ReleaseState.STALE}:
        raise ActiveReleaseError("candidate records cannot directly declare ACTIVE/SUPERSEDED/STALE")
    source = Path(source_path).expanduser().resolve() if source_path else None
    if state in {ReleaseState.ACQUIRED, ReleaseState.VALIDATED} and (source is None or not source.is_file()):
        raise ActiveReleaseError(f"{state.value} candidate requires an existing source file")
    validation = dict(validation or {})
    if state == ReleaseState.VALIDATED and validation.get("status") != "PASS":
        raise ActiveReleaseError("VALIDATED candidate requires validation.status=PASS")
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "dataset_id": dataset_id,
        "release_id": release_id,
        "state": state.value,
        "detected_at": now,
        "source_uri": source.as_uri() if source else None,
        "hash": sha256_file(source) if source else None,
        "validation": validation,
        "metadata": metadata or {},
    }
    payload = load_candidates(root)
    payload["datasets"][dataset_id] = record
    payload["updated_at"] = now
    _atomic_json(candidates_path(root), payload)
    return record


def promote_candidate(dataset_id: str, *, root: Path | None = None) -> dict[str, Any]:
    candidates = load_candidates(root)
    candidate = candidates.get("datasets", {}).get(dataset_id)
    if not isinstance(candidate, dict) or candidate.get("state") != ReleaseState.VALIDATED.value:
        raise ActiveReleaseError(f"{dataset_id} has no VALIDATED candidate to promote")
    uri = str(candidate.get("source_uri") or "")
    if not uri.startswith("file:///"):
        raise ActiveReleaseError("local promotion currently requires a file URI")
    from urllib.parse import unquote, urlparse

    parsed = urlparse(uri)
    raw = unquote(parsed.path)
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    previous = get_active_release(dataset_id, registry_path(root))
    record = promote_release(
        dataset_id,
        str(candidate["release_id"]),
        Path(raw),
        validated_at=str((candidate.get("validation") or {}).get("validated_at") or datetime.now(timezone.utc).isoformat()),
        metadata=dict(candidate.get("metadata") or {}),
        path=registry_path(root),
    )
    candidate["state"] = ReleaseState.ACTIVE.value
    candidate["promoted_at"] = record["promoted_at"]
    candidate["superseded_release_id"] = previous.get("active_release_id") if previous else None
    candidates["updated_at"] = record["promoted_at"]
    _atomic_json(candidates_path(root), candidates)
    return record


def what_would_change(dataset_id: str) -> dict[str, Any]:
    affected = list(DEPENDENCY_GRAPH.get(dataset_id, ()))
    all_capabilities = {item for values in DEPENDENCY_GRAPH.values() for item in values}
    return {
        "dataset_id": dataset_id,
        "would_mark_stale": affected,
        "would_remain_current": sorted(all_capabilities - set(affected)),
    }


def source_health(record: dict[str, Any]) -> tuple[str, str]:
    if record.get("status") != "ACTIVE":
        return "INVALID", "record is not ACTIVE"
    uri = str(record.get("source_uri") or "")
    if not uri.startswith("file:///"):
        return "REMOTE", "hash verification delegated to storage adapter"
    from urllib.parse import unquote, urlparse

    raw = unquote(urlparse(uri).path)
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    path = Path(raw)
    if not path.is_file():
        return "INVALID", "source file missing"
    if sha256_file(path) != record.get("hash"):
        return "INVALID", "source hash mismatch"
    for member in (record.get("metadata") or {}).get("source_set") or []:
        member_uri = str(member.get("source_uri") or "")
        if not member_uri.startswith("file:///"):
            return "REMOTE", "source-set hash verification delegated to storage adapter"
        member_raw = unquote(urlparse(member_uri).path)
        if os.name == "nt" and member_raw.startswith("/") and len(member_raw) > 2 and member_raw[2] == ":":
            member_raw = member_raw[1:]
        member_path = Path(member_raw)
        if not member_path.is_file():
            return "INVALID", f"source-set member missing: {member.get('role') or 'unknown'}"
        if sha256_file(member_path) != member.get("hash"):
            return "INVALID", f"source-set hash mismatch: {member.get('role') or 'unknown'}"
    return "PASS", "source hash matches"


def refresh_source_health(root: Path | None = None) -> dict[str, Any]:
    active = load_registry(registry_path(root)).get("datasets", {})
    checked_at = datetime.now(timezone.utc).isoformat()
    datasets = {}
    for dataset_id, record in active.items():
        status, detail = source_health(record)
        datasets[dataset_id] = {"status": status, "detail": detail, "checked_at": checked_at}
    payload = {"schema_version": 1, "checked_at": checked_at, "datasets": datasets}
    _atomic_json(source_health_path(root), payload)
    return payload


def load_source_health(root: Path | None = None) -> dict[str, Any]:
    path = source_health_path(root)
    if not path.is_file():
        return {"schema_version": 1, "checked_at": None, "datasets": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def refresh_facility_index(
    pbjapp_root: Path,
    *,
    root: Path | None = None,
    ccns: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    active = load_registry(registry_path(root)).get("datasets", {})
    targets = ccns or tuple(
        p.name.removeprefix("pbj320-")
        for p in (pbjapp_root / "deployments").glob("pbj320-[0-9][0-9][0-9][0-9][0-9][0-9]")
    )
    facilities: dict[str, Any] = {}
    for ccn in targets:
        manifest_path = pbjapp_root / "deployments" / f"pbj320-{ccn}" / "PACKAGE_MANIFEST.json"
        if not manifest_path.is_file():
            facilities[ccn] = {"status": "UNKNOWN", "capabilities": {}, "reason": "package manifest missing"}
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        capabilities: dict[str, str] = {}
        for item in manifest.get("source_release_provenance") or []:
            dataset_id = str(item.get("source_dataset") or "")
            current = active.get(dataset_id)
            status = "CURRENT" if current and item.get("source_release") == current.get("active_release_id") and item.get("source_hash") == current.get("hash") else "STALE"
            for capability in DEPENDENCY_GRAPH.get(dataset_id, (dataset_id,)):
                if capabilities.get(capability) != "STALE":
                    capabilities[capability] = status
        facilities[ccn] = {
            "status": "STALE" if "STALE" in capabilities.values() else ("CURRENT" if capabilities else "UNKNOWN"),
            "capabilities": capabilities,
            "manifest": str(manifest_path),
        }
    payload = {"schema_version": 1, "refreshed_at": datetime.now(timezone.utc).isoformat(), "facilities": facilities}
    _atomic_json(facility_index_path(root), payload)
    return payload


def load_facility_index(root: Path | None = None) -> dict[str, Any]:
    path = facility_index_path(root)
    if not path.is_file():
        return {"schema_version": 1, "refreshed_at": None, "facilities": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def control_panel_payload(root: Path | None = None) -> dict[str, Any]:
    active = load_registry(registry_path(root)).get("datasets", {})
    pending = load_candidates(root).get("datasets", {})
    health_cache = load_source_health(root)
    rows = []
    for dataset_id in sorted(set(active) | set(pending) | set(DEPENDENCY_GRAPH)):
        current = active.get(dataset_id)
        candidate = pending.get(dataset_id)
        cached_health = (health_cache.get("datasets") or {}).get(dataset_id) or {}
        health = str(cached_health.get("status") or ("MISSING" if not current else "UNKNOWN"))
        health_detail = str(cached_health.get("detail") or ("no ACTIVE release" if not current else "health check pending"))
        rows.append({
            "dataset_id": dataset_id,
            "active": current,
            "pending": candidate if candidate and candidate.get("state") != "ACTIVE" else None,
            "health": health,
            "health_detail": health_detail,
            "impact": what_would_change(dataset_id),
        })
    return {
        "datasets": rows,
        "facilities": load_facility_index(root),
        "source_health_checked_at": health_cache.get("checked_at"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
