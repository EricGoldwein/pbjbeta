"""Read-only CMS Provider Data theme discovery in the existing control-plane state.

Catalog observations never acquire artifacts, record candidates, or change ACTIVE.
Resource IDs/versions are publisher metadata tokens, not content hashes.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urljoin, urlparse
from zoneinfo import ZoneInfo

from release_control_plane import _atomic_json, control_plane_root

BASE = "https://data.cms.gov/provider-data"
THEME = "Nursing homes including rehab services"
THEME_SLUG = "nursing-homes"
SEARCH_URL = BASE + "/api/1/search?theme=" + quote(THEME, safe="") + "&page-size=100"
METASTORE = BASE + "/api/1/metastore/schemas/dataset/items/"
ARCHIVES_URL = BASE + "/api/1/archive/aggregate/theme/nursing-homes/relative"
CURRENT_ARCHIVES_URL = BASE + "/api/1/archive/aggregate/current/theme/all/relative"
CURRENT_ZIP_URL = BASE + "/sites/default/files/dataset-archives/current/theme/theme_nursing-homes_current.zip"
SCHEMA_VERSION = 1


def _request(url: str, *, method: str = "GET"):
    if urlparse(url).hostname != "data.cms.gov" or not url.startswith("https://"):
        raise ValueError("Catalog discovery requires an official HTTPS CMS URL")
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "PBJ-Data-Ops-NH-Catalog/1.0"})
    return urllib.request.urlopen(req, timeout=30)


def fetch_json(url: str) -> Any:
    with _request(url) as response:
        return json.loads(response.read(8 * 1024 * 1024).decode("utf-8"))


def fetch_head(url: str) -> dict[str, str]:
    with _request(url, method="HEAD") as response:
        return {key.lower(): value for key, value in response.headers.items()}


def snapshot_path(root: Path | None = None) -> Path:
    return control_plane_root(root) / "state" / "cms_nh_catalog.json"


def load_catalog(root: Path | None = None) -> dict[str, Any]:
    path = snapshot_path(root)
    if not path.is_file():
        return {"schema_version": SCHEMA_VERSION, "status": "NOT_CHECKED", "datasets": [], "summary": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION or not isinstance(data.get("datasets"), list):
            raise ValueError("unsupported catalog snapshot")
        for row in data["datasets"]:
            row["coverage"] = coverage_for(row["stable_id"])
        return data
    except (ValueError, OSError) as exc:
        return {"schema_version": SCHEMA_VERSION, "status": "ERROR", "error": str(exc), "datasets": [], "summary": {"errors": 1}}


def _unwrap(value: Any) -> Any:
    return value.get("data", value) if isinstance(value, dict) else value


def _theme_names(dataset: dict[str, Any]) -> list[str]:
    return [str(_unwrap(item)) for item in dataset.get("theme") or []]


def discover_theme(fetch: Callable[[str], Any]) -> list[dict[str, Any]]:
    """Require complete authoritative membership, including future additions."""
    found: dict[str, dict[str, Any]] = {}
    total: int | None = None
    for page in range(1, 101):
        payload = fetch(SEARCH_URL + f"&page={page}")
        if not isinstance(payload, dict) or "results" not in payload or "total" not in payload:
            raise ValueError("CMS search response has no authoritative results/total")
        page_total = int(payload["total"])
        if total is not None and total != page_total:
            raise ValueError("CMS theme membership changed during pagination; retry discovery")
        total = page_total
        results = payload["results"]
        rows = list(results.values()) if isinstance(results, dict) else results
        if not isinstance(rows, list):
            raise ValueError("CMS search results are not a collection")
        before = len(found)
        for row in rows:
            if not isinstance(row, dict) or not row.get("identifier") or THEME not in _theme_names(row):
                raise ValueError("CMS theme search returned an invalid or foreign dataset")
            identifier = str(row["identifier"])
            if identifier in found:
                raise ValueError("CMS theme search returned duplicate dataset IDs")
            found[identifier] = row
        if len(found) == total:
            if total == 0:
                raise ValueError("CMS nursing-home catalog unexpectedly empty; removals not inferred")
            return list(found.values())
        if len(found) > total or len(found) == before:
            raise ValueError("CMS theme listing incomplete; removals not inferred")
    raise ValueError("CMS theme pagination limit exceeded")


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def normalize_dataset(dataset: dict[str, Any], expected_id: str) -> dict[str, Any]:
    if dataset.get("identifier") != expected_id or THEME not in _theme_names(dataset):
        raise ValueError("CMS metastore identity/theme does not match the listed dataset")
    resources = []
    for wrapper in dataset.get("distribution") or []:
        dist = _unwrap(wrapper)
        if not isinstance(dist, dict):
            continue
        url = str(dist.get("downloadURL") or "")
        if not url:
            continue
        if urlparse(url).hostname != "data.cms.gov" or not url.startswith("https://"):
            raise ValueError("CMS resource URL is not an official HTTPS CMS URL")
        refs = []
        for ref in dist.get("%Ref:downloadURL") or []:
            data = _unwrap(ref)
            if isinstance(data, dict):
                refs.append({key: data.get(key) for key in ("identifier", "version", "perspective", "checksum")})
        refs.sort(key=lambda item: json.dumps(item, sort_keys=True))
        # The URL itself carries ID/version when reference expansion is absent.
        match = re.search(r"/resources/([^/]+)_([0-9]+)/", url)
        resources.append({
            "distribution_id": wrapper.get("identifier") if "data" in wrapper else None,
            "resource_id": match.group(1) if match else next((r["identifier"] for r in refs if r.get("identifier")), None),
            "version": match.group(2) if match else next((r["version"] for r in refs if r.get("version")), None),
            "url": url, "filename": unquote(urlparse(url).path.rsplit("/", 1)[-1]),
            "media_type": dist.get("mediaType"), "references": refs,
            "dictionary_url": dist.get("describedBy"),
        })
    if not resources:
        raise ValueError("CMS dataset has no current downloadable resource")
    resources.sort(key=lambda item: item["url"])
    modified = str(dataset.get("modified") or "")
    released = str(dataset.get("released") or "")
    if not modified or not released:
        raise ValueError("CMS dataset has no logical modified/released metadata")
    artifact_identity = _fingerprint([{key: r.get(key) for key in ("distribution_id", "resource_id", "version", "url", "references")} for r in resources])
    logical_release = modified[:7] if re.match(r"^\d{4}-\d{2}-\d{2}", modified) else modified
    return {
        "stable_id": expected_id, "title": str(dataset.get("title") or expected_id),
        "description": str(dataset.get("description") or "")[:4000],
        "modified": modified, "released": released, "planned_update": dataset.get("nextUpdateDate"),
        "metadata_modified": dataset.get("%modified"), "logical_release": logical_release,
        "resources": resources, "artifact_identity": artifact_identity,
        "publisher_identity": _fingerprint({"id": expected_id, "resources": artifact_identity, "modified": modified, "released": released}),
        "metadata_url": METASTORE + expected_id + "?show-reference-ids=true",
        "overview_url": BASE + "/dataset/" + expected_id,
    }


def coverage_for(stable_id: str) -> dict[str, Any]:
    """Audited stable-ID mappings; never infer a lifecycle from a similar title."""
    mapping = {
        "4pq5-n9py": ("A", "Governed", "cms.provider_info", "ProviderInfo adapter; normalized bundle + Zweli; explicit ACTIVE", "Provider charts and public/provider facility slices", "Required region/CMI builders block activation until proven"),
        "r5ix-sfxw": ("A", "Governed", "cms.health_citations", "health_citations_acquire; schema/provenance validation; explicit ACTIVE", "Facility citations and public citation surfaces", "Acquisition and review remain source-specific"),
        "y2hd-n93e": ("B", "Governed via Provider bundle", "cms.nh_ownership", "Provider extraction/promotion bundle member; member hash + ownership schema", "Care Compare ownership in facility bundles", "Co-versioned Provider member; distinct from PECOS Owners"),
        "qmdc-9999": ("B", "Governed via Provider bundle", "cms.provider_info", "Provider extraction manifest; interval CSV member hashes", "Provider processing-month/quarter mapping", "No independent ACTIVE dataset"),
        "tagd-9999": ("B", "Governed via Provider bundle", "cms.provider_info", "Provider extraction manifest; citation description member hashes", "Citation code descriptions", "No independent ACTIVE dataset"),
        "ifjz-ge4w": ("B", "Governed via Provider bundle", "cms.provider_info", "Provider retained member inventory/extraction; hashes only", "Retained source archive; no dedicated downstream consumer", "No standalone semantic validator or ACTIVE lifecycle"),
        "svdt-c123": ("B", "Governed via Provider bundle", "cms.provider_info", "Provider retained survey member inventory/extraction; hashes only", "Retained source archive", "No standalone semantic validator or ACTIVE lifecycle"),
        "tbry-pc2d": ("C", "Candidate workflow available", "cms.survey_summary", "survey_summary.prepare_candidate; immutable raw + provenance + validation receipt; explicit review", "No downstream consumer or asserted foreign keys", "Candidate availability is not an ACTIVE release; inspect control state"),
        "djen-97ju": ("C", "Built, not integrated", None, "mds_quality_measures_lifecycle.prepare_mds_quality_measures_candidate", "Normalized MDS measure candidates in claims worktree", "cms.quality_measures_mds lifecycle at f09c7cb; no operational integration"),
        "ijh5-nb2v": ("C", "Built, not integrated", None, "claims_quality_measures_lifecycle.prepare_claims_quality_measures_candidate", "Normalized claims measure candidates in claims worktree", "cms.quality_measures_claims lifecycle at f09c7cb; no operational integration"),
        "fykj-qjee": ("C", "Built, not integrated", None, "snf_qrp_provider_lifecycle.prepare_snf_qrp_provider_candidate", "Normalized SNF QRP candidates in claims worktree", "cms.snf_qrp_provider lifecycle at f09c7cb; no operational integration"),
        "hicp-9999": ("E", "Out of scope", None, "Provider inventory policy: documentation_reference", "Reference archive", "Documentation reference; no operational source lifecycle"),
    }
    values = mapping.get(stable_id, ("D", "Detected only", None, "CMS catalog discovery only", "None governed operationally", "No governed lifecycle; some files retained as adjacent CMS program members"))
    category, label, source_id, lifecycle, consumer, gap = values
    external_id = {"djen-97ju": "cms.quality_measures_mds", "ijh5-nb2v": "cms.quality_measures_claims", "fykj-qjee": "cms.snf_qrp_provider"}.get(stable_id)
    workflow_source = "cms.provider_info" if stable_id == "y2hd-n93e" else source_id
    result = {"class": category, "label": label, "source_id": source_id, "workflow_source_id": workflow_source, "dataset_id": external_id or source_id,
            "detector": "cms_nh_catalog / CMS search + metastore", "lifecycle": lifecycle, "consumer": consumer, "gap": gap,
            "active_lifecycle": "Existing explicit source approval" if category == "A" else ("Parent Provider bundle; no standalone promotion" if category == "B" else "Not integrated"),
            "acquirer": "Existing source workflow" if source_id else "No catalog acquisition action",
            "validator": lifecycle, "action": "Open source" if source_id else None}
    if stable_id == "tbry-pc2d":
        result["active_lifecycle"] = "Explicit approval required; ACTIVE determined by release registry"
        result["acquirer"] = "survey_summary.prepare_candidate"
    return result


def compare_observation(current: dict[str, Any], previous: dict[str, Any] | None, today: str) -> tuple[str, str]:
    if previous is None:
        return "NEW_DATASET", "First successful observation; no earlier catalog baseline to establish publication timing"
    if current["artifact_identity"] == previous["artifact_identity"]:
        if str(current.get("planned_update") or "")[:10] == today:
            return "PLANNED_TODAY", "CMS planned an update today; current resource identity has not advanced since the previous successful observation"
        return "CURRENT", "Authoritative lookup succeeded; current resource identity matches the previous successful observation"
    if current["logical_release"] > previous["logical_release"]:
        return "NEWER", "A changed publisher artifact has a newer logical modified month/version"
    if current["logical_release"] < previous["logical_release"]:
        return "ERROR", "Publisher logical release regressed; inspect evidence before treating it as a new release"
    return "REVISED", "Publisher resource identity changed under the same logical month/version"


def _archive_observation(fetch: Callable[[str], Any], head: Callable[[str], dict[str, str]]) -> dict[str, Any]:
    current_payload = fetch(CURRENT_ARCHIVES_URL)
    rows = current_payload.get("data") if isinstance(current_payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("CMS current archive response missing data collection")
    current = next((row for row in rows if row.get("theme") == THEME_SLUG and row.get("type") == "current"), None)
    if current is None:
        raise ValueError("CMS current nursing-home archive missing")
    url = urljoin("https://data.cms.gov", str(current.get("url") or ""))
    if url != CURRENT_ZIP_URL:
        raise ValueError("CMS current theme archive URL differs from the audited official route")
    headers = {key.lower(): value for key, value in head(url).items()}
    if "zip" not in headers.get("content-type", "").lower():
        raise ValueError("CMS current archive HEAD is not a ZIP response")
    history_payload = fetch(ARCHIVES_URL)
    history = history_payload.get("data") if isinstance(history_payload, dict) else None
    if not isinstance(history, list):
        raise ValueError("CMS archive history missing")
    dated = [row for row in history if row.get("type") == "theme" and "Snapshot" not in str(row.get("name") or "")]
    latest = max(dated, key=lambda row: str(row.get("date") or "")) if dated else {}
    metadata = {key: current.get(key) for key in ("id", "name", "date", "size", "theme")}
    metadata.update({"url": url, "etag": headers.get("etag"), "last_modified": headers.get("last-modified"),
                     "content_length": headers.get("content-length"), "content_type": headers.get("content-type"),
                     "checksum": None, "head_checked": True})
    identity = _fingerprint(metadata)
    return {**metadata, "identity": identity, "latest_publication": {key: latest.get(key) for key in ("id", "name", "date", "url", "size")},
            "history_count": len(history), "history_url": ARCHIVES_URL,
            "note": "Archive row ID is not a per-dataset release ID; HEAD has no guaranteed content checksum or atomicity contract"}


def catalog_summary(rows: list[dict[str, Any]], archive: dict[str, Any]) -> dict[str, int]:
    counts = Counter(row.get("status") for row in rows)
    return {"datasets": sum(row.get("status") != "REMOVED_OR_ARCHIVED" for row in rows),
            "published": counts["NEWER"], "revised": counts["REVISED"], "new_datasets": counts["NEW_DATASET"],
            "unchanged": counts["CURRENT"], "planned_today": counts["PLANNED_TODAY"],
            "removed": counts["REMOVED_OR_ARCHIVED"], "errors": counts["ERROR"],
            "archive_errors": int(archive.get("status") == "ERROR")}


def refresh_catalog(*, root: Path | None = None, fetch: Callable[[str], Any] | None = None,
                    head: Callable[[str], dict[str, str]] | None = None, now: datetime | None = None) -> dict[str, Any]:
    """Refresh bounded discovery metadata; preserve last success on any failure."""
    fetch = fetch or fetch_json
    head = head or fetch_head
    now = now or datetime.now(ZoneInfo("America/New_York"))
    now = now.astimezone(ZoneInfo("America/New_York"))
    checked_at, today = now.isoformat(), now.date().isoformat()
    old = load_catalog(root)
    previous = {row["stable_id"]: row for row in old.get("datasets") or []}
    listing_error = None
    try:
        listed = discover_theme(fetch)
    except Exception as exc:
        listing_error = str(exc)
        listed = []
    def observe(row: dict[str, Any]) -> dict[str, Any]:
        identifier = str(row["identifier"])
        prior = previous.get(identifier) or {}
        success = prior.get("last_successful")
        try:
            metadata = fetch(METASTORE + quote(identifier, safe="") + "?show-reference-ids=true")
            if not isinstance(metadata, dict):
                raise ValueError("CMS metastore response is not a dataset object")
            current = normalize_dataset(metadata, identifier)
            status, reason = compare_observation(current, success, today)
            if status == "ERROR":
                raise ValueError(reason)
            if prior.get("status") == "REMOVED_OR_ARCHIVED":
                status, reason = "NEW_DATASET", "Dataset re-entered the authoritative theme listing"
            return {**current, "checked_at": checked_at, "last_successful_at": checked_at,
                    "first_observed_at": prior.get("first_observed_at") or checked_at,
                    "last_successful": current, "previous_successful": success,
                    "status": status, "reason": reason, "error": None, "coverage": coverage_for(identifier)}
        except Exception as exc:
            return {**prior, **(success or {}), "stable_id": identifier, "title": (success or {}).get("title") or row.get("title") or identifier,
                    "checked_at": checked_at, "last_successful": success, "coverage": coverage_for(identifier),
                    "status": "ERROR", "reason": "Authoritative dataset lookup failed; last successful metadata preserved", "error": str(exc)}
    if listing_error:
        datasets = [{**row, "checked_at": checked_at, "status": "ERROR", "error": listing_error,
                     "reason": "Authoritative theme listing failed; membership and last successful observations preserved"} for row in previous.values()]
    else:
        with ThreadPoolExecutor(max_workers=6) as pool:
            datasets = list(pool.map(observe, listed))
        ids = {row["stable_id"] for row in datasets}
        datasets.extend({**row, "checked_at": checked_at, "status": "REMOVED_OR_ARCHIVED", "error": None,
                         "reason": "Absent from a complete successful theme listing; archival is not independently proven"}
                        for identifier, row in previous.items() if identifier not in ids)
    datasets.sort(key=lambda row: str(row.get("title") or row["stable_id"]).lower())
    prior_archive = old.get("theme_archive") or {}
    try:
        archive_success = _archive_observation(fetch, head)
        archive_previous = prior_archive.get("last_successful")
        archive = {**archive_success, "checked_at": checked_at, "last_successful": archive_success,
                   "previous_successful": archive_previous, "error": None,
                   "status": "NEW_OBSERVATION" if not archive_previous else ("CHANGED" if archive_success["identity"] != archive_previous["identity"] else "CURRENT")}
    except Exception as exc:
        archive = {**prior_archive, "checked_at": checked_at, "status": "ERROR", "error": str(exc)}
    summary = catalog_summary(datasets, archive)
    if listing_error and not datasets:
        summary["errors"] += 1
    payload = {"schema_version": SCHEMA_VERSION, "checked_at": checked_at, "theme": THEME, "theme_slug": THEME_SLUG,
               "theme_identity": "fd81c516-c2be-5fc3-81f2-8232d8b2b07b", "listing_url": SEARCH_URL,
               "status": "ERROR" if listing_error or summary["errors"] or summary["archive_errors"] else "OK",
               "error": listing_error, "membership_successful_at": old.get("membership_successful_at") if listing_error else checked_at,
               "datasets": datasets, "theme_archive": archive, "summary": summary}
    _atomic_json(snapshot_path(root), payload)
    return payload
