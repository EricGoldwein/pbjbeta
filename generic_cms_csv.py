"""Small fail-closed CMS CSV adapter for sources without a bespoke pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import date
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from active_release_registry import load_registry, registry_path
from release_control_plane import ReleaseState, load_candidates, record_candidate


@dataclass(frozen=True)
class CsvFeed:
    dataset_id: str
    cms_dataset_id: str
    filename_pattern: str
    destination: Path
    required_column_groups: tuple[tuple[str, ...], ...]
    automatic_validation: bool
    cms_product_path: str | None = None
    cms_product_name: str | None = None
    version_date_basis: str = "release"


def _fetch(url: str) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "PBJ-data-ops-release-check/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _release_id(row: dict[str, Any], filename: str) -> str:
    quarter = re.search(r"CY(\d{4})Q([1-4])", filename, re.I)
    if quarter:
        return f"CY{quarter.group(1)}Q{quarter.group(2)}"
    date = re.search(r"(20\d{2})[._-]?(\d{2})[._-]?(\d{2})", filename)
    if date:
        return "-".join(date.groups())
    month = re.search(r"([A-Za-z]{3,9})[_-]?(20\d{2})", filename)
    if month:
        return f"{month.group(2)}-{month.group(1)[:3].lower()}"
    return str(row.get("last_updated") or row.get("modified") or row.get("file_uuid") or filename)


def _version_sort_key(row: dict[str, Any]) -> tuple[date, tuple[int, int, str], str, str]:
    attributes = row.get("attributes") if isinstance(row.get("attributes"), dict) else {}
    raw_version = str(attributes.get("field_dataset_version") or "")
    try:
        version = date.fromisoformat(raw_version)
    except ValueError as exc:
        raise RuntimeError(f"CMS dataset version has an invalid date: {raw_version!r}") from exc
    raw_revision = str(attributes.get("field_re_release_version") or "")
    revision = (1, int(raw_revision), "") if raw_revision.isdigit() else (bool(raw_revision), 0, raw_revision)
    return (
        version,
        revision,
        str(attributes.get("field_last_updated_date") or ""),
        str(row.get("id") or ""),
    )


def _resolve_current_version(
    feed: CsvFeed,
    fetch_json: Callable[[str], Any],
) -> tuple[str, dict[str, Any]]:
    """Resolve a stable CMS product page to its newest published dataset node."""
    if not feed.cms_product_path and not feed.cms_product_name:
        return feed.cms_dataset_id, {}
    if not feed.cms_product_path or not feed.cms_product_name:
        raise RuntimeError(f"{feed.dataset_id} CMS product discovery is incompletely configured")

    slug_url = (
        "https://data.cms.gov/data-api/v1/slug?path="
        + quote(feed.cms_product_path, safe="")
    )
    slug_payload = fetch_json(slug_url)
    product = slug_payload.get("data") if isinstance(slug_payload, dict) else None
    if not isinstance(product, dict):
        raise RuntimeError(f"CMS slug lookup returned no product for {feed.dataset_id}")
    product_id = str(product.get("uuid") or "")
    if product_id != feed.cms_dataset_id:
        raise RuntimeError(
            f"CMS slug product identity changed for {feed.dataset_id}: "
            f"expected {feed.cms_dataset_id}, got {product_id or 'missing'}"
        )

    versions_url = (
        "https://data.cms.gov/jsonapi/node/dataset?"
        "fields%5Bnode--dataset%5D=field_dataset_version,field_last_updated_date,"
        "field_re_release_select,field_re_release_version&"
        "filter%5Bfield_dataset_type.name%5D="
        + quote(feed.cms_product_name, safe="")
        + "&sort=-field_dataset_version,-field_re_release_version"
    )
    versions_payload = fetch_json(versions_url)
    rows = versions_payload.get("data") if isinstance(versions_payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"CMS version lookup returned no versions for {feed.dataset_id}")
    valid_rows = [row for row in rows if isinstance(row, dict) and row.get("id")]
    if not valid_rows:
        raise RuntimeError(f"CMS version lookup returned no usable versions for {feed.dataset_id}")
    newest = max(valid_rows, key=_version_sort_key)
    version_id = str(newest["id"])

    current = product.get("current_dataset")
    current_id = str(current.get("uuid") or "") if isinstance(current, dict) else ""
    if not current_id:
        raise RuntimeError(f"CMS product has no current dataset version for {feed.dataset_id}")
    if current_id != version_id:
        raise RuntimeError(
            f"CMS discovery sources disagree for {feed.dataset_id}: "
            f"slug current={current_id}, newest version={version_id}"
        )
    attributes = newest.get("attributes") if isinstance(newest.get("attributes"), dict) else {}
    return version_id, {
        "product_id": product_id,
        "product_path": feed.cms_product_path,
        "product_name": feed.cms_product_name,
        "slug_url": slug_url,
        "versions_url": versions_url,
        "dataset_version_id": version_id,
        "dataset_version_label": attributes.get("field_dataset_version"),
        "dataset_version_modified": attributes.get("field_last_updated_date"),
    }


def detect(feed: CsvFeed, *, fetch_json: Callable[[str], Any] | None = None) -> dict[str, Any]:
    fetch = fetch_json or _fetch
    version_id, discovery = _resolve_current_version(feed, fetch)
    resources_url = f"https://data.cms.gov/data-api/v1/dataset/{version_id}/resources"
    payload = fetch(resources_url)
    rows = payload.get("data") if isinstance(payload, dict) else payload
    matches = [row for row in (rows or []) if isinstance(row, dict) and re.search(feed.filename_pattern, str(row.get("file_name") or ""), re.I)]
    if not matches:
        raise RuntimeError(f"CMS resources contained no expected CSV for {feed.dataset_id}")
    primary = [item for item in matches if item.get("type") == "Primary" or item.get("media_bundle") == "primary_dataset_file"]
    candidates = primary or matches
    row = max(candidates, key=lambda item: (_release_id(item, str(item.get("file_name") or "")), str(item.get("file_uuid") or "")))
    filename = str(row.get("file_name") or "")
    url = str(row.get("file_url") or "")
    if not url.startswith("http"):
        raise RuntimeError("CMS resource is missing a public file URL")
    return {
        "release_id": _release_id(row, filename),
        # Keep the governed pair key stable: it is the file/snapshot period,
        # not the CMS publication vintage or the acquisition date.
        "snapshot_date": _release_id(row, filename) if re.search(r"CY\d{4}Q[1-4]|20\d{2}[._-]?\d{2}[._-]?\d{2}|[A-Za-z]{3,9}[_-]?20\d{2}", filename, re.I) else None,
        "cms_release_vintage": (str(discovery.get("dataset_version_label") or "")[:7] or None) if feed.version_date_basis == "release" else None,
        "publisher_period_basis": "CMS reporting quarter" if feed.version_date_basis == "reporting_period" else "CMS dataset release label",
        "filename": filename,
        "url": url,
        "file_uuid": row.get("file_uuid"),
        "resources_url": resources_url,
        "row": row,
        **discovery,
    }


def assess_feed(
    feed: CsvFeed,
    *,
    root: Path,
    fetch_json: Callable[[str], Any] | None = None,
    fetch_bytes: Callable[[str], bytes] | None = None,
) -> dict[str, Any]:
    """Read-only authoritative comparison of CMS latest against ACTIVE."""
    try:
        found = detect(feed, fetch_json=fetch_json)
    except Exception as exc:
        return {
            "status": "ERROR",
            "new_release_available": None,
            "detail": str(exc),
        }
    return _assess_found(feed, root=root, found=found, fetch_bytes=fetch_bytes)


def _remote_sha256(url: str, fetch_bytes=None) -> str:
    if fetch_bytes:
        return hashlib.sha256(fetch_bytes(url)).hexdigest()
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "PBJ-data-ops-release-check/1.0"})
    with urllib.request.urlopen(request, timeout=600) as response:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assess_found(feed: CsvFeed, *, root: Path, found: dict[str, Any], fetch_bytes=None) -> dict[str, Any]:
    active = ((load_registry(registry_path(root)).get("datasets") or {}).get(feed.dataset_id) or {})
    common = {
        "release_id": found["release_id"],
        "publisher_latest_release_id": found.get("cms_release_vintage") or found["release_id"],
        "publisher_filename": found["filename"],
        "publisher_url": found["url"],
        "publisher_file_uuid": found.get("file_uuid"),
        "cms_product_id": found.get("product_id") or feed.cms_dataset_id,
        "cms_dataset_version_id": found.get("dataset_version_id") or feed.cms_dataset_id,
        "cms_dataset_version_label": found.get("dataset_version_label"),
        "cms_dataset_version_modified": found.get("dataset_version_modified"),
        "cms_resources_url": found.get("resources_url"),
        "cms_release_vintage": found.get("cms_release_vintage"),
        "snapshot_date": found.get("snapshot_date"),
        "publisher_period_basis": found.get("publisher_period_basis"),
    }
    from cms_release_identity import assess_raw_identity
    return assess_raw_identity(feed.dataset_id, active=active, current=common, fetch_bytes=fetch_bytes)



def validate_local_csv(path: Path, groups: tuple[tuple[str, ...], ...]) -> dict[str, Any]:
    """Public schema/hash check used by pair validation and acquire."""
    return _validate(path, groups)


def _validate(path: Path, groups: tuple[tuple[str, ...], ...]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("downloaded CSV is missing or empty")
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        row_count = sum(1 for row in reader if any(str(value).strip() for value in row))
    normalized = {re.sub(r"[^a-z0-9]", "", col.lower()) for col in header}
    missing = [group[0] for group in groups if not any(re.sub(r"[^a-z0-9]", "", alias.lower()) in normalized for alias in group)]
    if missing:
        raise RuntimeError(f"schema validation failed; missing {missing}")
    digest_builder = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    return {"status": "PASS", "columns": len(header), "row_count": row_count, "hash": digest}


def compare_publisher_artifact_identity(
    left: dict[str, Any] | None,
    right: dict[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Return differing and matching authoritative artifact identity fields.

    Only fields present on both sides are compared.  URL is the primary legacy
    identity; file UUID and dataset-version UUID strengthen newer records.
    """
    left = left or {}
    right = right or {}
    aliases = {
        "source_sha256": ("cms_source_sha256", "publisher_sha256"),
        "publisher_url": ("cms_publisher_url", "download_url", "publisher_url", "url"),
        "file_uuid": ("cms_file_uuid", "cms_publisher_file_uuid", "publisher_file_uuid", "file_uuid"),
        "version_uuid": ("cms_dataset_version_id", "dataset_version_id", "version_uuid"),
    }

    def _first(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
        return next((str(payload.get(key)).strip() for key in keys if payload.get(key)), "")

    changed: list[str] = []
    matched: list[str] = []
    for label, keys in aliases.items():
        lhs, rhs = _first(left, keys), _first(right, keys)
        if not lhs or not rhs:
            continue
        (matched if lhs == rhs else changed).append(label)
    return changed, matched


def run_feed(feed: CsvFeed, acquire: bool, *, root: Path, fetch_json=None, fetch_bytes=None) -> dict[str, Any]:
    try:
        found = detect(feed, fetch_json=fetch_json)
    except Exception as exc:
        return {"status": "ERROR", "new_release_available": None, "detail": str(exc)}
    assessment = _assess_found(feed, root=root, found=found, fetch_bytes=fetch_bytes)
    if assessment["status"] in {"ERROR"}:
        return assessment
    if assessment["status"] in {"CURRENT", "UNKNOWN"}:
        return assessment
    pending = ((load_candidates(root).get("datasets") or {}).get(feed.dataset_id) or {})
    pending_meta = pending.get("metadata") if isinstance(pending.get("metadata"), dict) else {}
    pending_changes, pending_matches = compare_publisher_artifact_identity(
        pending_meta,
        {
            "cms_publisher_url": found.get("url"),
            "cms_file_uuid": found.get("file_uuid"),
            "cms_dataset_version_id": found.get("dataset_version_id"),
            "cms_source_sha256": assessment.get("publisher_sha256"),
        },
    )
    pending_hash_matches = not assessment.get("publisher_sha256") or (
        pending.get("hash") == assessment["publisher_sha256"] or
        pending_meta.get("cms_source_sha256") == assessment["publisher_sha256"])
    if pending.get("release_id") == found["release_id"] and pending_hash_matches and not pending_changes and pending_matches and (
        pending.get("state") in {"ACQUIRED", "VALIDATED"}
        or (pending.get("state") == "DETECTED" and not acquire)
    ):
        return {**assessment, "status": pending["state"], "new_release_available": True, "detail": "existing pending candidate"}
    publisher_metadata = {
        "download_url": found["url"],
        "filename": found["filename"],
        "cms_publisher_url": found["url"],
        "cms_publisher_filename": found["filename"],
        "cms_publisher_release_id": found.get("cms_release_vintage") or found["release_id"],
        "cms_product_id": found.get("product_id") or feed.cms_dataset_id,
        "cms_dataset_version_id": found.get("dataset_version_id") or feed.cms_dataset_id,
        "cms_dataset_version_label": found.get("dataset_version_label"),
        "cms_dataset_version_modified": found.get("dataset_version_modified"),
        "cms_release_vintage": found.get("cms_release_vintage"),
        "snapshot_date": found.get("snapshot_date"),
        "cms_source_sha256": assessment.get("publisher_sha256"),
        "publisher_checked_at": assessment.get("publisher_checked_at"),
        "cms_file_uuid": found.get("file_uuid"),
        "publisher_revision_changed": assessment.get("publisher_revision_changed", False),
        "change_kind": "REVISED" if assessment.get("status") == "REVISED" else "NEWER",
        "revision_of_active_hash": assessment.get("active_hash"),
        "revision_identity_changes": assessment.get("revision_identity_changes") or [],
    }
    record_candidate(feed.dataset_id, found["release_id"], ReleaseState.DETECTED, metadata=publisher_metadata, root=root)
    if not acquire:
        return assessment
    feed.destination.mkdir(parents=True, exist_ok=True)
    final = feed.destination / found["filename"]
    fd, temp_name = tempfile.mkstemp(prefix=f".{final.name}.", suffix=".partial", dir=feed.destination)
    os.close(fd)
    temp = Path(temp_name)
    try:
        if fetch_bytes:
            temp.write_bytes(fetch_bytes(found["url"]))
        else:
            request = urllib.request.Request(found["url"], headers={"User-Agent": "PBJ-data-ops-release-check/1.0"})
            with urllib.request.urlopen(request, timeout=600) as response, temp.open("wb") as output:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
        validation = _validate(temp, feed.required_column_groups)
        if assessment.get("publisher_sha256") and validation["hash"] != assessment["publisher_sha256"]:
            raise RuntimeError("CMS bytes changed between detection and acquisition; check again")
        existing_hash = None
        if final.exists():
            digest_builder = hashlib.sha256()
            with final.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_builder.update(chunk)
            existing_hash = digest_builder.hexdigest()
        if final.exists() and existing_hash != validation["hash"]:
            if feed.dataset_id not in {"cms.snf_all_owners", "cms.snf_enrollments"}:
                raise RuntimeError(f"refusing to overwrite differing {final.name}")
            # Immutable revision path; the currently ACTIVE file is never replaced.
            final = feed.destination / f"{final.stem}__sha256_{validation['hash']}{final.suffix}"
            if final.exists() and _validate(final, feed.required_column_groups)["hash"] != validation["hash"]:
                raise RuntimeError("revision artifact hash collision")
        if not final.exists():
            os.replace(temp, final)
        else:
            temp.unlink()
        state = ReleaseState.VALIDATED if feed.automatic_validation else ReleaseState.ACQUIRED
        publisher_metadata["acquired_at"] = datetime.now(timezone.utc).isoformat()
        record_candidate(feed.dataset_id, found["release_id"], state, source_path=final, validation=validation, metadata=publisher_metadata, root=root)
        return {"status": state.value, "new_release_available": True, "release_id": found["release_id"]}
    except Exception:
        temp.unlink(missing_ok=True)
        raise
