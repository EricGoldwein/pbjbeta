"""Local PBJ release control-plane services (no Flask dependency)."""

from __future__ import annotations

import json
import os
import shutil
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

CONSUMER_OPERATOR_LABELS: dict[str, str] = {
    "premium facility bundles": "Facility provider bundles",
    "premium facility citation tables": "Facility citation tables",
    "PBJapp national combined; facility provider slices": "National + facility provider slices",
    "public PBJ320.com Provider surfaces": "Public Provider pages (pbj-root)",
    "benchmark pods; public state pages": "Benchmark pods & state pages",
    "public /owners/* pages": "Public SNF owner pages",
    "facility.citations": "Facility citation capability",
    "facility.snf_owners": "SNF owners capability",
    "ownership.enrollment_ccn_bridge": "Ownership CCN bridge",
}

# Derived artifact rebuild paths (national / publication consumers — not CMS sources).
DERIVED_ARTIFACT_PIPELINES: dict[str, tuple[dict[str, str], ...]] = {
    "cms.provider_info": (
        {
            "artifact": "provider_info_combined.csv",
            "rebuild": "scripts/build_provider_info_combined.py",
            "consumer": "PBJapp national combined; facility provider slices",
        },
        {
            "artifact": "facility_*_provider_info_data.csv",
            "rebuild": "facility packaging (registry-gated slice refresh)",
            "consumer": "premium facility bundles",
        },
        {
            "artifact": "pbj-root search / state aggregates",
            "rebuild": "scripts/build_state_page_aggregates.py; generate_search_index.py (pbj-root)",
            "consumer": "public PBJ320.com Provider surfaces",
        },
    ),
    "cms.health_citations": (
        {
            "artifact": "facility_*_citations.csv",
            "rebuild": "facility packaging (ACTIVE cms.health_citations gate)",
            "consumer": "premium facility citation tables",
        },
    ),
    "cms.pbj_nurse_staffing": (
        {
            "artifact": "state_quarterly_metrics.csv / national_quarterly_metrics.csv",
            "rebuild": "generate_metrics.py / benchmark builders",
            "consumer": "benchmark pods; public state pages",
        },
    ),
    "cms.snf_all_owners": (
        {
            "artifact": "ownership indexes / SNF owner pages",
            "rebuild": "scripts/build_snf_owners_index.py (pbj-root)",
            "consumer": "public /owners/* pages",
        },
    ),
}


def stale_derived_consumers(root: Path | None = None) -> dict[str, list[str]]:
    """Capabilities stale because a derived ACTIVE release lags its upstream CMS ACTIVE."""
    active = load_registry(registry_path(root)).get("datasets", {})
    stale: dict[str, list[str]] = {}
    upstream_pairs = (("cms.provider_info", "cms.nh_ownership"),)
    for upstream_id, derived_id in upstream_pairs:
        upstream_release = (active.get(upstream_id) or {}).get("active_release_id")
        derived_release = (active.get(derived_id) or {}).get("active_release_id")
        if not upstream_release or not derived_release or upstream_release == derived_release:
            continue
        recorded = ((active.get(derived_id) or {}).get("metadata") or {}).get("upstream_releases") or {}
        if recorded.get(upstream_id) == upstream_release:
            continue
        for capability in DEPENDENCY_GRAPH.get(derived_id, ()):
            stale.setdefault(upstream_id, [])
            if capability not in stale[upstream_id]:
                stale[upstream_id].append(capability)
    return stale


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


def candidate_local_path(source_uri: str | None) -> Path | None:
    """Resolve a file:// candidate URI to a local path (Windows-safe)."""
    if not source_uri:
        return None
    raw = str(source_uri).strip()
    if raw.startswith("file:///"):
        raw = raw[8:]
    elif raw.startswith("file://"):
        raw = raw[7:]
    from urllib.parse import unquote

    decoded = unquote(raw)
    if os.name == "nt" and decoded.startswith("/") and len(decoded) > 2 and decoded[2] == ":":
        decoded = decoded[1:]
    path = Path(decoded)
    return path if path.is_file() else path


def record_validated_pair(
    records: dict[str, dict[str, Any]],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Atomically write one or more pair-validated candidate records. Does not promote ACTIVE."""
    if not records:
        raise ActiveReleaseError("validated pair requires candidate records")
    now = datetime.now(timezone.utc).isoformat()
    payload = load_candidates(root)
    for dataset_id, record in records.items():
        if str(record.get("state") or "") != ReleaseState.VALIDATED.value:
            raise ActiveReleaseError(f"{dataset_id} is not VALIDATED")
        validation = record.get("validation") if isinstance(record.get("validation"), dict) else {}
        if validation.get("status") != "PASS":
            raise ActiveReleaseError(f"{dataset_id} VALIDATED record requires validation.status=PASS")
        source = candidate_local_path(record.get("source_uri"))
        if source is None or not source.is_file():
            raise ActiveReleaseError(f"{dataset_id} VALIDATED candidate requires an existing source file")
        payload["datasets"][dataset_id] = dict(record)
    payload["updated_at"] = now
    _atomic_json(candidates_path(root), payload)
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


def discard_candidate(
    dataset_id: str,
    *,
    reason: str | None = None,
    root: Path | None = None,
) -> dict[str, Any] | None:
    """Remove a pending candidate record (does not touch ACTIVE registry)."""
    payload = load_candidates(root)
    removed = payload.get("datasets", {}).pop(dataset_id, None)
    if removed is None:
        return None
    if reason:
        meta = removed.get("metadata") if isinstance(removed.get("metadata"), dict) else {}
        meta = dict(meta)
        meta["discarded_reason"] = reason
        meta["discarded_at"] = datetime.now(timezone.utc).isoformat()
        removed["metadata"] = meta
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(candidates_path(root), payload)
    return removed


def _active_record_from_candidate(
    dataset_id: str,
    candidate: dict[str, Any],
    *,
    promoted_at: str,
) -> dict[str, Any]:
    """Build an ACTIVE registry row from a VALIDATED candidate (no I/O)."""
    source = candidate_local_path(str(candidate.get("source_uri") or ""))
    if source is None or not source.is_file():
        raise ActiveReleaseError(f"{dataset_id} VALIDATED candidate requires a local source file")
    validation = candidate.get("validation") if isinstance(candidate.get("validation"), dict) else {}
    if validation.get("status") != "PASS":
        raise ActiveReleaseError(f"{dataset_id} VALIDATED candidate requires validation.status=PASS")
    validated_at = str(validation.get("validated_at") or promoted_at)
    actual_hash = sha256_file(source)
    if candidate.get("hash") and str(candidate.get("hash")).lower() != actual_hash:
        raise ActiveReleaseError(f"{dataset_id} candidate hash does not match source file")
    return {
        "dataset_id": dataset_id,
        "active_release_id": str(candidate["release_id"]),
        "source_filename": source.name,
        "source_uri": source.as_uri(),
        "release_date": None,
        "downloaded_at": candidate.get("downloaded_at"),
        "validated_at": validated_at,
        "hash": actual_hash,
        "status": "ACTIVE",
        "schema_version": 1,
        "metadata": dict(candidate.get("metadata") or {}),
        "promoted_at": promoted_at,
    }


def promote_active_pair(
    dataset_ids: tuple[str, str],
    *,
    root: Path | None = None,
) -> dict[str, Any]:
    """Atomically promote two VALIDATED candidates to ACTIVE (one write per registry file).

    Restores the prior active_releases snapshot if the candidates write fails.
    """
    if len(dataset_ids) != 2 or len(set(dataset_ids)) != 2:
        raise ActiveReleaseError("promote_active_pair requires two distinct dataset ids")
    root = root or control_plane_root()
    active_path = registry_path(root)
    candidates_path_file = candidates_path(root)
    candidates_payload = load_candidates(root)
    active_payload = load_registry(active_path)
    now = datetime.now(timezone.utc).isoformat()
    release_ids: set[str] = set()
    promoted_active: dict[str, dict[str, Any]] = {}

    for dataset_id in dataset_ids:
        candidate = (candidates_payload.get("datasets") or {}).get(dataset_id)
        if not isinstance(candidate, dict) or candidate.get("state") != ReleaseState.VALIDATED.value:
            raise ActiveReleaseError(f"{dataset_id} has no VALIDATED candidate to promote")
        release_ids.add(str(candidate.get("release_id") or ""))
        promoted_active[dataset_id] = _active_record_from_candidate(dataset_id, candidate, promoted_at=now)
    if len(release_ids) != 1 or not release_ids.pop():
        raise ActiveReleaseError("pair candidates must share one release_id")

    for dataset_id in dataset_ids:
        active_payload.setdefault("datasets", {})[dataset_id] = promoted_active[dataset_id]
        cand = dict((candidates_payload.get("datasets") or {})[dataset_id])
        prior_active = get_active_release(dataset_id, active_path)
        cand["state"] = ReleaseState.ACTIVE.value
        cand["promoted_at"] = now
        cand["superseded_release_id"] = (prior_active or {}).get("active_release_id")
        candidates_payload.setdefault("datasets", {})[dataset_id] = cand

    active_payload["updated_at"] = now
    candidates_payload["updated_at"] = now

    active_backup = active_path.read_text(encoding="utf-8") if active_path.is_file() else None
    try:
        _atomic_json(active_path, active_payload)
        _atomic_json(candidates_path_file, candidates_payload)
    except Exception:
        if active_backup is not None:
            active_path.write_text(active_backup, encoding="utf-8")
        raise

    return {
        "release_id": promoted_active[dataset_ids[0]]["active_release_id"],
        "promoted_at": now,
        "datasets": promoted_active,
    }


def promote_candidate(dataset_id: str, *, root: Path | None = None) -> dict[str, Any]:
    from cms_source_registry import is_review_only_source

    if is_review_only_source(dataset_id):
        raise ActiveReleaseError(f"{dataset_id} is review-only; no activation is permitted")
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
    source = Path(raw)
    if sha256_file(source) != str(candidate.get("hash") or ""):
        raise ActiveReleaseError(f"{dataset_id} candidate hash does not match staged artifact")

    previous = get_active_release(dataset_id, registry_path(root))
    promotion_source = source
    backup: Path | None = None
    staged_copy: Path | None = None
    served_target_path: Path | None = None
    active_registry_backup = (
        registry_path(root).read_text(encoding="utf-8")
        if registry_path(root).is_file()
        else None
    )
    try:
        from derived_provenance import GOVERNED_DERIVATIVES

        if dataset_id in GOVERNED_DERIVATIVES:
            served_uri = str((candidate.get("metadata") or {}).get("served_target_uri") or "")
            served_target = candidate_local_path(served_uri)
            if served_target is None or not served_target.is_file():
                raise ActiveReleaseError(
                    f"{dataset_id} promotion requires an existing local ACTIVE consumption path"
                )
            if served_target.resolve() == source.resolve():
                raise ActiveReleaseError(
                    f"{dataset_id} candidate must be staged separately from the served ACTIVE artifact"
                )
            served_target_path = served_target
            backup_fd, backup_name = tempfile.mkstemp(
                prefix=f".{served_target.name}.", suffix=".backup", dir=served_target.parent
            )
            os.close(backup_fd)
            backup = Path(backup_name)
            shutil.copy2(served_target, backup)
            copy_fd, copy_name = tempfile.mkstemp(
                prefix=f".{served_target.name}.", suffix=".candidate", dir=served_target.parent
            )
            os.close(copy_fd)
            staged_copy = Path(copy_name)
            shutil.copy2(source, staged_copy)
            if sha256_file(staged_copy) != str(candidate["hash"]):
                raise ActiveReleaseError(f"{dataset_id} staged promotion copy failed hash verification")
            os.replace(staged_copy, served_target)
            promotion_source = served_target

        record = promote_release(
            dataset_id,
            str(candidate["release_id"]),
            promotion_source,
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
    except Exception:
        if backup is not None and served_target_path is not None:
            if backup.is_file():
                os.replace(backup, served_target_path)
        if active_registry_backup is not None:
            _atomic_json(registry_path(root), json.loads(active_registry_backup))
        raise
    finally:
        for temporary in (backup, staged_copy):
            if temporary is not None and temporary.exists():
                temporary.unlink()


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
