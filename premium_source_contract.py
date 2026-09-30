"""Fail-closed ACTIVE-release and upstream provenance contract for bundles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from active_release_client import (
    ActiveRelease,
    ReleaseRegistryError,
    artifact_release_state,
    load_active_release,
    validate_active_source,
)


@dataclass(frozen=True)
class SourceArtifactRequirement:
    capability: str
    dataset_id: str
    relative_artifact: str
    sidecar_suffix: str = ".source.json"


DERIVED_UPSTREAMS = {
    "pbj.benchmarks.state": ("cms.pbj_nurse_staffing",),
    "pbj.benchmarks.national": ("cms.pbj_nurse_staffing",),
    "pbj.benchmarks.region": ("cms.pbj_nurse_staffing", "cms.provider_info"),
    "pbj.benchmarks.geo_cmi": ("cms.provider_info",),
    "pbj.peer_distribution": ("cms.pbj_nurse_staffing",),
}


def requirements(ccn: str) -> tuple[SourceArtifactRequirement, ...]:
    ccn = str(ccn).strip().zfill(6)
    bridge = "ownership/_derived/cms_snf_ownership_ccn_bridge/packaged_bridge_manifest.json"
    return (
        SourceArtifactRequirement("Staffing", "cms.pbj_nurse_staffing", f"facility_{ccn}_complete_data.csv"),
        SourceArtifactRequirement("Staffing", "cms.pbj_non_nurse_staffing", f"facility_{ccn}_nonnurse_daily.csv"),
        SourceArtifactRequirement("Provider Info", "cms.provider_info", f"facility_{ccn}_provider_info_data.csv"),
        SourceArtifactRequirement("Citations", "cms.health_citations", f"facility_{ccn}_citations.csv"),
        SourceArtifactRequirement("Ownership", "cms.provider_info", f"ownership/NH_Ownership_facility_{ccn}.csv"),
        SourceArtifactRequirement("Ownership", "cms.snf_all_owners", f"ownership/SNF_All_Owners_facility_{ccn}.csv"),
        SourceArtifactRequirement("Ownership", "cms.snf_all_owners", bridge, ".cms_snf_all_owners.source.json"),
        SourceArtifactRequirement("Ownership", "cms.snf_enrollments", bridge, ".cms_snf_enrollments.source.json"),
        SourceArtifactRequirement("Benchmarks", "pbj.benchmarks.state", "state_quarterly_metrics.csv"),
        SourceArtifactRequirement("Benchmarks", "pbj.benchmarks.national", "national_quarterly_metrics.csv"),
        SourceArtifactRequirement("Benchmarks", "pbj.benchmarks.region", "cms_region_quarterly_metrics.csv"),
        SourceArtifactRequirement("Benchmarks", "pbj.benchmarks.region_mapping", "cms_region_state_mapping.csv"),
        SourceArtifactRequirement("Benchmarks", "pbj.benchmarks.geo_cmi", "geo_nursing_cmi_quarterly.csv"),
        SourceArtifactRequirement("Benchmarks", "pbj.peer_distribution", "pbj_lite/facility_lite_metrics.csv"),
        SourceArtifactRequirement("MACPAC", "macpac.state_staffing_standards", "macpac_state_standards_clean.csv"),
    )


def validate_derived_upstreams(release: ActiveRelease) -> None:
    """Require every governed derivative to match CURRENT ACTIVE upstreams."""
    required = DERIVED_UPSTREAMS.get(release.dataset_id)
    if not required:
        return
    provenance = release.metadata.get("upstream_provenance")
    if not isinstance(provenance, dict):
        raise ReleaseRegistryError(
            f"{release.dataset_id} upstream provenance is UNKNOWN; refusing current"
        )
    for upstream_id in required:
        recorded = provenance.get(upstream_id)
        if not isinstance(recorded, dict):
            raise ReleaseRegistryError(
                f"{release.dataset_id} upstream provenance is UNKNOWN for {upstream_id}"
            )
        active = load_active_release(upstream_id)
        if (
            recorded.get("release_id") != active.active_release_id
            or recorded.get("source_hash") != active.hash
        ):
            raise ReleaseRegistryError(
                f"{release.dataset_id} is STALE: recorded {upstream_id} "
                f"{recorded.get('release_id') or 'UNKNOWN'}/{recorded.get('source_hash') or 'UNKNOWN'} "
                f"but ACTIVE is {active.active_release_id}/{active.hash}"
            )


def _semantic_ok(path: Path) -> tuple[bool, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return False, "missing or empty artifact"
    if path.suffix.lower() == ".json":
        try:
            if not isinstance(json.loads(path.read_text(encoding="utf-8")), dict):
                return False, "JSON artifact root is not an object"
        except (OSError, json.JSONDecodeError):
            return False, "invalid JSON artifact"
    elif path.suffix.lower() == ".csv":
        try:
            header = path.open("r", encoding="utf-8-sig", errors="replace").readline().strip()
        except OSError:
            return False, "unreadable CSV artifact"
        if not header or "," not in header:
            return False, "CSV header is missing or invalid"
    return True, "semantic validation passed"


def validate_bundle_sources(deploy_dir: Path, ccn: str) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []
    for req in requirements(ccn):
        try:
            release = load_active_release(req.dataset_id)
            validate_active_source(release)
            validate_derived_upstreams(release)
            artifact = deploy_dir / req.relative_artifact
            semantic_ok, semantic_reason = _semantic_ok(artifact)
            sidecar = Path(f"{artifact}{req.sidecar_suffix}")
            if req.sidecar_suffix == ".source.json":
                state, reason = artifact_release_state(artifact, release)
            elif not sidecar.is_file():
                state, reason = "BUILD", "artifact provenance missing"
            else:
                raw = json.loads(sidecar.read_text(encoding="utf-8"))
                state = "REUSE" if (
                    raw.get("source_dataset") == req.dataset_id
                    and raw.get("source_release") == release.active_release_id
                    and raw.get("source_hash") == release.hash
                ) else "BUILD"
                reason = "validated" if state == "REUSE" else "artifact source release is STALE"
            results.append((req.capability, semantic_ok and state == "REUSE", reason if semantic_ok else semantic_reason))
        except Exception as exc:
            results.append((req.capability, False, str(exc)))
    return results
