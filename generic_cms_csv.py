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
from pathlib import Path
from typing import Any, Callable

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


def detect(feed: CsvFeed, *, fetch_json: Callable[[str], Any] | None = None) -> dict[str, Any]:
    payload = (fetch_json or _fetch)(f"https://data.cms.gov/data-api/v1/dataset/{feed.cms_dataset_id}/resources")
    rows = payload.get("data") if isinstance(payload, dict) else payload
    matches = [row for row in (rows or []) if isinstance(row, dict) and re.search(feed.filename_pattern, str(row.get("file_name") or ""), re.I)]
    if not matches:
        raise RuntimeError(f"CMS resources contained no expected CSV for {feed.dataset_id}")
    row = next((item for item in matches if item.get("type") == "Primary" or item.get("media_bundle") == "primary_dataset_file"), matches[0])
    filename = str(row.get("file_name") or "")
    url = str(row.get("file_url") or "")
    if not url.startswith("http"):
        raise RuntimeError("CMS resource is missing a public file URL")
    return {"release_id": _release_id(row, filename), "filename": filename, "url": url, "row": row}


def _validate(path: Path, groups: tuple[tuple[str, ...], ...]) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("downloaded CSV is missing or empty")
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = next(csv.reader(handle), [])
    normalized = {re.sub(r"[^a-z0-9]", "", col.lower()) for col in header}
    missing = [group[0] for group in groups if not any(re.sub(r"[^a-z0-9]", "", alias.lower()) in normalized for alias in group)]
    if missing:
        raise RuntimeError(f"schema validation failed; missing {missing}")
    digest_builder = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest_builder.update(chunk)
    digest = digest_builder.hexdigest()
    return {"status": "PASS", "columns": len(header), "hash": digest}


def run_feed(feed: CsvFeed, acquire: bool, *, root: Path, fetch_json=None, fetch_bytes=None) -> dict[str, Any]:
    found = detect(feed, fetch_json=fetch_json)
    active_id = ((load_registry(registry_path(root)).get("datasets") or {}).get(feed.dataset_id) or {}).get("active_release_id")
    pending = ((load_candidates(root).get("datasets") or {}).get(feed.dataset_id) or {})
    if found["release_id"] == active_id:
        return {"status": "CURRENT", "new_release_available": False, "release_id": active_id}
    if pending.get("release_id") == found["release_id"] and (
        pending.get("state") in {"ACQUIRED", "VALIDATED"}
        or (pending.get("state") == "DETECTED" and not acquire)
    ):
        return {"status": pending["state"], "new_release_available": True, "release_id": found["release_id"], "detail": "existing pending candidate"}
    record_candidate(feed.dataset_id, found["release_id"], ReleaseState.DETECTED, metadata={"download_url": found["url"], "filename": found["filename"]}, root=root)
    if not acquire:
        return {"status": "DETECTED", "new_release_available": True, "release_id": found["release_id"]}
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
        existing_hash = None
        if final.exists():
            digest_builder = hashlib.sha256()
            with final.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest_builder.update(chunk)
            existing_hash = digest_builder.hexdigest()
        if final.exists() and existing_hash != validation["hash"]:
            raise RuntimeError(f"refusing to overwrite differing {final.name}")
        if not final.exists():
            os.replace(temp, final)
        else:
            temp.unlink()
        state = ReleaseState.VALIDATED if feed.automatic_validation else ReleaseState.ACQUIRED
        record_candidate(feed.dataset_id, found["release_id"], state, source_path=final, validation=validation, metadata={"download_url": found["url"]}, root=root)
        return {"status": state.value, "new_release_available": True, "release_id": found["release_id"]}
    except Exception:
        temp.unlink(missing_ok=True)
        raise
