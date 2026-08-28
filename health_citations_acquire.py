"""Governed Health Citations acquire/adopt path (CMS r5ix-sfxw + Provider bundle provenance)."""

from __future__ import annotations

import csv
import json
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cms_data_paths
from active_release_registry import sha256_file
from cms_source_registry import CMS_ID_HEALTH_CITATIONS

METASTORE_URL = (
    "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/"
    f"{CMS_ID_HEALTH_CITATIONS}?show-reference-ids=true"
)

MONTH_ABBR = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

REQUIRED_COLUMNS = (
    "CMS Certification Number (CCN)",
    "Survey Date",
)

FetchJson = Callable[[str], Any]


@dataclass(frozen=True)
class CmsHealthCitationsRelease:
    dataset_id: str
    data_vintage_label: str
    release_id: str
    distribution_filename: str
    distribution_url: str
    modified: str | None


class HealthCitationsAcquireError(RuntimeError):
    pass


def _default_fetch_json(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "PBJ-data-ops-health-citations/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _release_id_from_label(label: str) -> str | None:
    match = re.fullmatch(r"([A-Za-z]+)\s+(20\d{2})", label.strip())
    if not match:
        return None
    month = MONTH_ABBR.get(match.group(1).lower()[:3])
    if not month:
        return None
    return f"{match.group(2)}-{month:02d}"


def _basename_for_release(release_id: str) -> str:
    year_s, month_s = release_id.split("-", 1)
    year, month = int(year_s), int(month_s)
    abbr = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")[month - 1]
    return f"NH_HealthCitations_{abbr}{year}.csv"


def resolve_cms_health_citations_release(
    *,
    fetch_json: FetchJson | None = None,
) -> CmsHealthCitationsRelease:
    payload = (fetch_json or _default_fetch_json)(METASTORE_URL)
    distributions = payload.get("distribution") or []
    primary = next(
        (item for item in distributions if isinstance(item, dict) and item.get("data", {}).get("downloadURL")),
        None,
    )
    if primary is None:
        raise HealthCitationsAcquireError("CMS metastore returned no downloadable Health Citations distribution")
    data = primary.get("data") or {}
    filename = str(data.get("filename") or primary.get("title") or "")
    url = str(data.get("downloadURL") or "")
    modified = str(data.get("modified") or payload.get("modified") or "") or None
    label = ""
    title_sources = (payload, primary, data)
    for source in title_sources:
        if not isinstance(source, dict):
            continue
        for key in ("title", "identifier", "name", "description"):
            value = str(source.get(key) or "")
            month_match = re.search(r"([A-Za-z]{3,9})\s+(20\d{2})", value)
            if month_match:
                label = f"{month_match.group(1)[:3].title()} {month_match.group(2)}"
                break
        if label:
            break
    if not label:
        file_match = re.search(r"NH_HealthCitations_([A-Za-z]{3})(\d{4})", filename, re.I)
        if file_match:
            label = f"{file_match.group(1).title()} {file_match.group(2)}"
    if not label and modified:
        mod_match = re.match(r"(\d{4})-(\d{2})", modified)
        if mod_match:
            year, month = int(mod_match.group(1)), int(mod_match.group(2))
            if 1 <= month <= 12:
                abbr = (
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
                )[month - 1]
                label = f"{abbr} {year}"
    if not label:
        raise HealthCitationsAcquireError("could not derive Health Citations vintage label from CMS metastore")
    release_id = _release_id_from_label(label)
    if not release_id:
        raise HealthCitationsAcquireError(f"unsupported CMS vintage label: {label}")
    return CmsHealthCitationsRelease(
        dataset_id=CMS_ID_HEALTH_CITATIONS,
        data_vintage_label=label,
        release_id=release_id,
        distribution_filename=filename or _basename_for_release(release_id),
        distribution_url=url,
        modified=modified,
    )


def citations_artifact_path(release_id: str, *, root: Path | None = None) -> Path:
    return cms_data_paths.citations_dir(root) / _basename_for_release(release_id)


def provider_bundle_manifest_path(release_id: str, *, root: Path | None = None) -> Path:
    return cms_data_paths.provider_info_dir(root) / "_manifests" / release_id / "release_manifest.json"


def _manifest_member(manifest: dict[str, Any], basename: str) -> dict[str, Any]:
    for item in manifest.get("source_members") or []:
        if item.get("basename") == basename:
            return item
    raise HealthCitationsAcquireError(f"provider manifest missing {basename}")


def verify_bundle_provenance(release_id: str, *, root: Path | None = None) -> dict[str, Any]:
    """Confirm local citations CSV matches Provider Info bundle manifest (CMS-provenanced)."""
    artifact = citations_artifact_path(release_id, root=root)
    if not artifact.is_file():
        raise HealthCitationsAcquireError(f"local citations artifact missing: {artifact}")
    manifest_path = provider_bundle_manifest_path(release_id, root=root)
    if not manifest_path.is_file():
        raise HealthCitationsAcquireError(f"provider bundle manifest missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    member = _manifest_member(manifest, artifact.name)
    expected = str(member.get("source_sha256") or "")
    outputs = member.get("normalized_outputs") or []
    if not expected and outputs:
        expected = str(outputs[0].get("sha256") or "")
    actual = sha256_file(artifact)
    if not expected or actual != expected:
        raise HealthCitationsAcquireError(
            f"bundle provenance mismatch for {artifact.name}: manifest={expected[:12]}… local={actual[:12]}…"
        )
    return {
        "status": "PASS",
        "release_id": release_id,
        "artifact_path": str(artifact),
        "artifact_hash": actual,
        "manifest_path": str(manifest_path),
        "manifest_member": member.get("basename"),
        "row_count": member.get("row_count"),
        "provenance": "cms.provider_info_bundle",
    }


def validate_health_citations_csv(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 1000:
        raise HealthCitationsAcquireError(f"citations CSV missing or too small: {path}")
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
    normalized = {str(col).strip().upper() for col in header}
    missing = [col for col in REQUIRED_COLUMNS if col.upper() not in normalized]
    if missing:
        raise HealthCitationsAcquireError(f"citations schema missing columns: {missing}")
    return {
        "status": "PASS",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "columns": len(header),
        "path": str(path),
    }


def adopt_health_citations_candidate(
    release_id: str,
    *,
    root: Path | None = None,
    control_root: Path | None = None,
) -> dict[str, Any]:
    from release_control_plane import ReleaseState, control_plane_root, record_candidate

    provenance = verify_bundle_provenance(release_id, root=root)
    artifact = Path(provenance["artifact_path"])
    candidate = record_candidate(
        "cms.health_citations",
        release_id,
        ReleaseState.ACQUIRED,
        source_path=artifact,
        metadata={
            "provenance": provenance["provenance"],
            "validation_evidence": provenance["manifest_path"],
            "upstream_releases": {"cms.provider_info": release_id},
        },
        root=control_root or control_plane_root(),
    )
    return {"candidate": candidate, "provenance": provenance}


def check_health_citations_cms(
    *,
    fetch_json: FetchJson | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    cms = resolve_cms_health_citations_release(fetch_json=fetch_json)
    pbj_root = root or cms_data_paths.repo_root()
    from active_release_registry import get_active_release, registry_path

    active_id = (get_active_release("cms.health_citations", registry_path()) or {}).get("active_release_id")
    local_ready = citations_artifact_path(cms.release_id, root=pbj_root).is_file()
    bundle_ok = False
    bundle_detail = ""
    if local_ready:
        try:
            verify_bundle_provenance(cms.release_id, root=pbj_root)
            bundle_ok = True
            bundle_detail = "local artifact matches Provider Info bundle manifest"
        except HealthCitationsAcquireError as exc:
            bundle_detail = str(exc)
    cms_is_newer = bool(active_id and cms.release_id != active_id) or (not active_id)
    return {
        "cms": {
            "data_vintage_label": cms.data_vintage_label,
            "release_id": cms.release_id,
            "distribution_filename": cms.distribution_filename,
        },
        "cms_is_newer": cms_is_newer,
        "active_release_id": active_id,
        "local_artifact_ready": local_ready,
        "bundle_provenance_ok": bundle_ok,
        "bundle_detail": bundle_detail,
    }


def prepare_health_citations_validated_candidate(
    release_id: str,
    *,
    root: Path | None = None,
    control_root: Path | None = None,
) -> dict[str, Any]:
    from release_control_plane import ReleaseState, control_plane_root, record_candidate, what_would_change

    pbj_root = root or cms_data_paths.repo_root()
    provenance = verify_bundle_provenance(release_id, root=pbj_root)
    artifact = Path(provenance["artifact_path"])
    validation = validate_health_citations_csv(artifact)
    candidate = record_candidate(
        "cms.health_citations",
        release_id,
        ReleaseState.VALIDATED,
        source_path=artifact,
        validation=validation,
        metadata={
            "provenance": provenance["provenance"],
            "validation_evidence": provenance["manifest_path"],
            "upstream_releases": {"cms.provider_info": release_id},
            "structural_status": "PASS",
        },
        root=control_root or control_plane_root(),
    )
    return {
        "candidate": candidate,
        "validation": validation,
        "provenance": provenance,
        "downstream_impact": what_would_change("cms.health_citations"),
    }
