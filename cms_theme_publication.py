"""CMS nursing-home publication compatibility view over catalog/API metadata.

Verified from PDC archive aggregate API and Aug 2026 theme zips (_scratch audit).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urljoin

from cms_source_registry import (
    CMS_ORIGIN,
    NH_THEME_ARCHIVE_INDEX,
    NH_THEME_DATASET_SOURCE_MAP,
    THEME_PUBLICATION_SOURCES,
)

FetchJson = Callable[[str], Any]
FetchBytes = Callable[[str], bytes]

_CACHE: dict[str, Any] | None = None


@dataclass(frozen=True)
class ThemeManifestMember:
    dataset_id: str
    source_id: str
    name: str
    modified_date: str
    product_release_id: str
    filename: str
    filesize: int
    mime_type: str


@dataclass(frozen=True)
class ThemePublication:
    publication_id: str
    publication_date: str
    theme: str
    archive_name: str
    download_url: str
    archive_size_bytes: int
    members_by_source: dict[str, ThemeManifestMember] = field(default_factory=dict)
    members_by_dataset: dict[str, ThemeManifestMember] = field(default_factory=dict)

    def member_for_source(self, source_id: str) -> ThemeManifestMember | None:
        return self.members_by_source.get(source_id)

    def changed_source_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.members_by_source))


def _default_fetch_json(url: str) -> Any:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "PBJ-data-ops-theme-publication/1.0"})
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _default_fetch_bytes(url: str) -> bytes:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": "PBJ-data-ops-theme-publication/1.0"})
    with urllib.request.urlopen(req, timeout=300) as response:
        return response.read()


def absolute_cms_url(path_or_url: str) -> str:
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        return path_or_url
    if path_or_url.startswith("/"):
        return urljoin(CMS_ORIGIN, path_or_url)
    return urljoin(CMS_ORIGIN + "/", path_or_url)


def modified_date_to_product_release_id(modified_date: str) -> str:
    """Processing vintage date → product release identity YYYY-MM."""
    return str(modified_date)[:7]


def parse_theme_manifest(manifest: list[dict[str, Any]]) -> dict[str, ThemeManifestMember]:
    members: dict[str, ThemeManifestMember] = {}
    for entry in manifest:
        if not isinstance(entry, dict):
            continue
        dataset_id = str(entry.get("dataset_id") or "").strip()
        source_id = NH_THEME_DATASET_SOURCE_MAP.get(dataset_id)
        if not source_id:
            continue
        resources = entry.get("resources") or []
        resource = resources[0] if resources else {}
        modified_date = str(entry.get("modified_date") or "")
        members[source_id] = ThemeManifestMember(
            dataset_id=dataset_id,
            source_id=source_id,
            name=str(entry.get("name") or ""),
            modified_date=modified_date,
            product_release_id=modified_date_to_product_release_id(modified_date),
            filename=str(resource.get("filename") or ""),
            filesize=int(resource.get("filesize") or 0),
            mime_type=str(resource.get("mime_type") or ""),
        )
    return members


def pick_latest_theme_publication_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Latest nursing-home theme drop (excludes snapshots and annual bundles)."""
    candidates = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("type") == "theme"
        and "Snapshot" not in str(row.get("name") or "")
    ]
    if not candidates:
        raise ValueError("no theme publication rows in archive index")
    return max(candidates, key=lambda row: str(row.get("date") or ""))


def read_manifest_from_zip_bytes(data: bytes) -> list[dict[str, Any]]:
    with zipfile.ZipFile(io.BytesIO(data), "r") as zf:
        if "manifest.json" not in zf.namelist():
            raise ValueError("manifest.json missing from theme archive")
        payload = json.loads(zf.read("manifest.json").decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("theme manifest.json must be a JSON array")
    return payload


def resolve_theme_publication(
    *,
    archive_index: list[dict[str, Any]] | None = None,
    manifest: list[dict[str, Any]] | None = None,
    fetch_json: FetchJson | None = None,
    fetch_bytes: FetchBytes | None = None,
) -> ThemePublication:
    """Resolve latest NH theme publication and parse manifest members."""
    explicit_bytes = fetch_bytes is not None
    custom_json = fetch_json is not None
    fetch_json = fetch_json or _default_fetch_json

    if archive_index is None:
        payload = fetch_json(NH_THEME_ARCHIVE_INDEX)
        archive_index = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(archive_index, list):
        raise ValueError("archive index must be a list")

    row = pick_latest_theme_publication_row(archive_index)
    download_url = absolute_cms_url(str(row.get("url") or ""))
    if manifest is None and explicit_bytes:
        # Explicit fixture/audit path only. Routine discovery never fetches a ZIP.
        manifest = read_manifest_from_zip_bytes(fetch_bytes(download_url))
    elif manifest is None:
        from cms_nh_catalog import discover_theme, load_catalog

        snapshot = load_catalog() if not custom_json else {}
        observed = snapshot.get("datasets") or []
        if snapshot.get("status") == "OK" and observed:
            manifest = [
                {"dataset_id": item["stable_id"], "name": item["title"],
                 "modified_date": item.get("modified") or "",
                 "resources": [{"filename": resource["filename"], "mime_type": resource.get("media_type"), "filesize": 0}
                               for resource in item.get("resources") or []]}
                for item in observed if item.get("status") != "REMOVED_OR_ARCHIVED"
            ]
        else:
            datasets = discover_theme(fetch_json)
            manifest = []
            for item in datasets:
                resources = []
                for wrapper in item.get("distribution") or []:
                    dist = wrapper.get("data", wrapper)
                    url = str(dist.get("downloadURL") or "")
                    if url:
                        resources.append({"filename": url.rsplit("/", 1)[-1], "mime_type": dist.get("mediaType"), "filesize": 0})
                manifest.append({"dataset_id": item["identifier"], "name": item.get("title"),
                                 "modified_date": item.get("modified"), "resources": resources})

    members_by_source = parse_theme_manifest(manifest)
    members_by_dataset = {member.dataset_id: member for member in members_by_source.values()}
    return ThemePublication(
        publication_id=str(row.get("id") or ""),
        publication_date=str(row.get("date") or ""),
        theme=str(row.get("theme") or "nursing-homes"),
        archive_name=str(row.get("name") or ""),
        download_url=download_url,
        archive_size_bytes=int(row.get("size") or 0),
        members_by_source=members_by_source,
        members_by_dataset=members_by_dataset,
    )


def get_latest_nh_theme_publication(
    *,
    fetch_json: FetchJson | None = None,
    fetch_bytes: FetchBytes | None = None,
    force_refresh: bool = False,
) -> ThemePublication | None:
    global _CACHE
    if _CACHE is not None and not force_refresh:
        return _CACHE.get("publication")
    try:
        publication = resolve_theme_publication(fetch_json=fetch_json, fetch_bytes=fetch_bytes)
    except Exception:
        return None
    _CACHE = {"publication": publication}
    return publication


def clear_theme_publication_cache() -> None:
    global _CACHE
    _CACHE = None


def publication_availability_for_source(
    source_id: str,
    *,
    active_release_id: str | None,
    publication: ThemePublication | None,
) -> dict[str, Any] | None:
    """Map theme manifest member to availability fields (distinct date axes)."""
    if publication is None or source_id not in THEME_PUBLICATION_SOURCES:
        return None
    member = publication.member_for_source(source_id)
    if member is None:
        return {
            "source_id": source_id,
            "in_latest_publication": False,
            "cms_publication_id": publication.publication_id,
            "cms_publication_date": publication.publication_date,
            "availability_source": "theme_publication",
            "new_release_available": False,
            "unchanged_in_latest_publication": True,
        }
    product_release_id = member.product_release_id
    new_available = bool(
        product_release_id and (not active_release_id or product_release_id != active_release_id)
    )
    return {
        "source_id": source_id,
        "in_latest_publication": True,
        "cms_publication_id": publication.publication_id,
        "cms_publication_date": publication.publication_date,
        "processing_modified_date": member.modified_date,
        "product_release_id": product_release_id,
        "publisher_latest_release_id": product_release_id,
        "cms_dataset_id": member.dataset_id,
        "manifest_filename": member.filename,
        "manifest_filesize": member.filesize,
        "availability_source": "theme_publication",
        "new_release_available": new_available,
        "unchanged_in_latest_publication": not new_available,
    }
