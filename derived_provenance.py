"""Governed staging, provenance, and activation gates for derived artifacts.

Builders write to :func:`pending_build_directory`, then record a VALIDATED
candidate. Nothing here changes a served artifact or ACTIVE state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

from active_release_registry import get_active_release, load_registry, registry_path, sha256_file
from release_control_plane import ReleaseState, control_plane_root, load_candidates, record_candidate
from release_source_catalog import BY_ID, UpdateMechanism


GOVERNED_DERIVATIVES = frozenset(
    {
        "pbj.benchmarks.state",
        "pbj.benchmarks.national",
        "pbj.benchmarks.region",
        "pbj.benchmarks.geo_cmi",
        "pbj.peer_distribution",
    }
)

# The only builders proven current in this operational tree. Region and
# geographic CMI deliberately remain absent because their historical builders
# do not match the current governed output contract.
AUTHORITATIVE_BUILDERS: dict[str, str] = {
    "pbj.benchmarks.state": "generate_metrics.py",
    "pbj.benchmarks.national": "generate_metrics.py",
    "pbj.peer_distribution": "lite_report.py",
}


class DerivativeActivationBlocked(RuntimeError):
    """Raised before approval/audit/ACTIVE mutation when derivatives are unsafe."""

    def __init__(self, upstream_id: str, blockers: list[dict[str, str]]) -> None:
        self.upstream_id = upstream_id
        self.blockers = blockers
        detail = "; ".join(
            f"{item['dataset_id']}: {item['reason']}" for item in blockers
        )
        super().__init__(
            f"Activation BLOCKED for {upstream_id}. Required derivative(s): {detail}. "
            "Keep the source VALIDATED for review; ACTIVE is unchanged."
        )


def _file_uri_path(uri: str | None) -> Path | None:
    raw = str(uri or "")
    if not raw.startswith("file:"):
        return None
    parsed = urlparse(raw)
    value = unquote(parsed.path)
    if parsed.netloc:
        value = f"//{parsed.netloc}{value}"
    if value.startswith("/") and len(value) > 2 and value[2] == ":":
        value = value[1:]
    return Path(value)


def dependent_derivatives(upstream_id: str) -> tuple[str, ...]:
    """Return every catalog-declared derivative for an upstream source."""
    return tuple(
        source.dataset_id
        for source in BY_ID.values()
        if source.mechanism == UpdateMechanism.DERIVED and upstream_id in source.upstream
    )


def _bundled_ownership_blocker(pending: Mapping[str, Any]) -> str | None:
    """Validate the Provider candidate's co-versioned NH Ownership member."""
    source_set = ((pending.get("metadata") or {}).get("source_set") or [])
    if not isinstance(source_set, list):
        return "Provider candidate source_set is missing or invalid"
    member = next(
        (
            item
            for item in source_set
            if isinstance(item, dict) and item.get("role") == "nh_ownership"
        ),
        None,
    )
    if not member:
        return "Provider candidate does not include the co-versioned nh_ownership member"
    path = Path(str(member.get("source_path") or "")).expanduser()
    if not path.is_file() or path.stat().st_size == 0:
        return f"co-versioned nh_ownership artifact is missing or empty: {path}"
    claimed_hash = str(member.get("hash") or "")
    if claimed_hash and claimed_hash.lower() != sha256_file(path):
        return "co-versioned nh_ownership artifact hash does not match candidate metadata"
    return None


def pending_upstream_provenance(
    upstream_id: str,
    *,
    root: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Return an override for the pending VALIDATED upstream candidate."""
    pending = (load_candidates(root).get("datasets") or {}).get(upstream_id) or {}
    if pending.get("state") != ReleaseState.VALIDATED.value:
        raise RuntimeError(f"{upstream_id} has no VALIDATED candidate to build from")
    release_id = str(pending.get("release_id") or "")
    source_hash = str(pending.get("hash") or "")
    if not release_id or not source_hash:
        raise RuntimeError(f"{upstream_id} VALIDATED candidate is missing release identity or hash")
    return {upstream_id: {"release_id": release_id, "source_hash": source_hash}}


def pending_build_directory(
    upstream_id: str,
    *,
    root: Path | None = None,
) -> Path:
    """Versioned candidate directory for the current pending upstream release."""
    provenance = pending_upstream_provenance(upstream_id, root=root)[upstream_id]
    safe_release = provenance["release_id"].replace("/", "_").replace("\\", "_")
    return control_plane_root(root) / "state" / "derived_candidates" / upstream_id / safe_release


def capture_upstream_provenance(
    dataset_id: str,
    *,
    root: Path | None = None,
    upstream_overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, dict[str, str]]:
    """Snapshot required upstream releases/hashes, allowing a pending override."""
    source = BY_ID.get(dataset_id)
    if source is None or source.mechanism != UpdateMechanism.DERIVED:
        raise ValueError(f"{dataset_id} is not a governed derived dataset")
    if not source.upstream:
        raise RuntimeError(f"{dataset_id} has no declared upstream dependencies")

    overrides = dict(upstream_overrides or {})
    unexpected = sorted(set(overrides) - set(source.upstream))
    if unexpected:
        raise RuntimeError(
            f"{dataset_id} received undeclared upstream override(s): {', '.join(unexpected)}"
        )
    active = load_registry(registry_path(root)).get("datasets", {})
    provenance: dict[str, dict[str, str]] = {}
    for upstream_id in source.upstream:
        upstream = overrides.get(upstream_id) or active.get(upstream_id) or {}
        release_id = str(upstream.get("release_id") or upstream.get("active_release_id") or "")
        source_hash = str(upstream.get("source_hash") or upstream.get("hash") or "")
        if not release_id or not source_hash:
            raise RuntimeError(
                f"{dataset_id} cannot record provenance: {upstream_id} "
                "is missing release identity or hash"
            )
        provenance[upstream_id] = {"release_id": release_id, "source_hash": source_hash}
    return provenance


def _served_target_uri(dataset_id: str, root: Path | None) -> str:
    active = get_active_release(dataset_id, registry_path(root)) or {}
    uri = str(active.get("source_uri") or "")
    target = _file_uri_path(uri)
    if target is None or not target.is_file():
        # Candidate creation remains safe and reviewable; explicit promotion
        # will fail closed until an ACTIVE consumption target is configured.
        return ""
    return target.resolve().as_uri()


def record_validated_derived_candidate(
    dataset_id: str,
    artifact: str | Path,
    *,
    builder: str,
    root: Path | None = None,
    upstream_overrides: Mapping[str, Mapping[str, str]] | None = None,
    served_target: str | Path | None = None,
) -> dict[str, Any]:
    """Record a staged artifact as VALIDATED with exact upstream provenance."""
    if dataset_id not in AUTHORITATIVE_BUILDERS:
        raise RuntimeError(f"{dataset_id} has no proven authoritative builder in this tree")
    if not str(builder).strip():
        raise RuntimeError(f"{dataset_id} requires builder provenance")
    target = Path(artifact).resolve()
    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError(f"derived artifact missing or empty: {target}")

    artifact_hash = sha256_file(target)
    upstream_provenance = capture_upstream_provenance(
        dataset_id, root=root, upstream_overrides=upstream_overrides
    )
    served_target_uri = (
        Path(served_target).resolve().as_uri()
        if served_target is not None
        else _served_target_uri(dataset_id, root)
    )
    built_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "builder": builder,
        "built_at": built_at,
        "output_artifact_hash": artifact_hash,
        "served_target_uri": served_target_uri,
        "upstream_provenance": upstream_provenance,
        "upstream_releases": {
            source_id: item["release_id"] for source_id, item in upstream_provenance.items()
        },
    }
    validation = {
        "status": "PASS",
        "validated_at": built_at,
        "artifact_hash": artifact_hash,
        "upstream_provenance": upstream_provenance,
    }
    return record_candidate(
        dataset_id,
        f"sha256:{artifact_hash[:16]}",
        ReleaseState.VALIDATED,
        source_path=target,
        validation=validation,
        metadata=metadata,
        root=root,
    )


def activation_derivative_blockers(
    upstream_id: str,
    *,
    root: Path | None = None,
) -> list[dict[str, str]]:
    """Explain why an upstream candidate cannot yet be made ACTIVE."""
    if upstream_id not in {"cms.provider_info", "cms.pbj_nurse_staffing"}:
        return []
    pending = (load_candidates(root).get("datasets") or {}).get(upstream_id) or {}
    release_id = str(pending.get("release_id") or "")
    source_hash = str(pending.get("hash") or "")
    blockers: list[dict[str, str]] = []
    candidates = load_candidates(root).get("datasets") or {}
    active = load_registry(registry_path(root)).get("datasets", {})

    for derivative_id in dependent_derivatives(upstream_id):
        if derivative_id == "cms.nh_ownership":
            reason = _bundled_ownership_blocker(pending)
            if reason:
                blockers.append({"dataset_id": derivative_id, "reason": reason})
            continue
        if derivative_id not in AUTHORITATIVE_BUILDERS:
            blockers.append(
                {
                    "dataset_id": derivative_id,
                    "reason": "no proven authoritative builder is available in the operational tree",
                }
            )
            continue
        candidate = candidates.get(derivative_id) or {}
        if candidate.get("state") != ReleaseState.VALIDATED.value:
            blockers.append(
                {
                    "dataset_id": derivative_id,
                    "reason": f"run {AUTHORITATIVE_BUILDERS[derivative_id]} to create a VALIDATED candidate",
                }
            )
            continue
        provenance = ((candidate.get("metadata") or {}).get("upstream_provenance") or {})
        recorded = provenance.get(upstream_id) if isinstance(provenance, dict) else None
        if not isinstance(recorded, dict) or (
            recorded.get("release_id") != release_id
            or recorded.get("source_hash") != source_hash
        ):
            blockers.append(
                {
                    "dataset_id": derivative_id,
                    "reason": f"candidate provenance does not match pending {upstream_id} {release_id}",
                }
            )
            continue
        for other_upstream in BY_ID[derivative_id].upstream:
            if other_upstream == upstream_id:
                continue
            current = active.get(other_upstream) or {}
            other_recorded = provenance.get(other_upstream) or {}
            if (
                not current.get("active_release_id")
                or not current.get("hash")
                or other_recorded.get("release_id") != current.get("active_release_id")
                or other_recorded.get("source_hash") != current.get("hash")
            ):
                blockers.append(
                    {
                        "dataset_id": derivative_id,
                        "reason": f"candidate provenance does not match ACTIVE {other_upstream}",
                    }
                )
                break
    return blockers


def assert_activation_derivatives_ready(
    upstream_id: str,
    *,
    root: Path | None = None,
) -> None:
    blockers = activation_derivative_blockers(upstream_id, root=root)
    if blockers:
        raise DerivativeActivationBlocked(upstream_id, blockers)
