"""Bounded Survey Summary candidate adapter; never activates a release.

Retains official source bytes and metadata by content hash. Existing retained
Provider members can supply the same bytes; no archive extraction pipeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from cms_nh_catalog import METASTORE, _request, fetch_json, normalize_dataset
from release_control_plane import ReleaseState, control_plane_root, record_candidate

SOURCE_ID = "cms.survey_summary"
DATASET_ID = "tbry-pc2d"
CCN = "CMS Certification Number (CCN)"
KEY = (CCN, "Inspection Cycle")
DATES = ("Health Survey Date", "Fire Safety Survey Date", "Processing Date")
TOTALS = ("Total Number of Health Deficiencies", "Total Number of Fire Safety Deficiencies")
SCHEMA = json.loads((Path(__file__).parent / "schemas" / "cms_survey_summary.json").read_text())


def validate_csv(raw: bytes, *, modified: str) -> dict:
    errors = Counter()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    columns = reader.fieldnames or []
    required = {CCN, "Inspection Cycle", "Provider Name", *DATES, *TOTALS}
    counts = [c for c in columns if c.startswith("Count of ")]
    fire_columns = counts[counts.index("Count of Emergency Preparedness Deficiencies"):] if "Count of Emergency Preparedness Deficiencies" in counts else []
    if columns != SCHEMA["columns"] or not required <= set(columns):
        errors["schema"] += 1
    keys, dated_keys, providers = Counter(), Counter(), set()
    rows = 0
    for row in reader:
        rows += 1
        if None in row or any(v is None for v in row.values()):
            errors["row_width"] += 1
        ccn = row.get(CCN, "") or ""
        if not re.fullmatch(r"[0-9]{2}[0-9A-Z]{4}", ccn):
            errors["ccn"] += 1
        cycle = row.get("Inspection Cycle")
        if cycle not in {"1", "2", "3"}:
            errors["cycle"] += 1
        keys[(ccn, cycle)] += 1
        dated_keys[(ccn, row.get("Health Survey Date"))] += 1
        providers.add(ccn)
        for column in DATES:
            value = row.get(column) or ""
            if not value and column != "Processing Date":
                continue  # CMS can report an inspection cycle without a survey date.
            try:
                parsed = date.fromisoformat(value)
                if parsed > date.fromisoformat(modified) or parsed.year < 1900:
                    errors["date_range"] += 1
            except ValueError:
                errors["date"] += 1
        if row.get("Processing Date") != modified:
            errors["processing_date"] += 1
        for column in (*TOTALS, *counts):
            value = row.get(column) or ""
            if not value and not row.get("Fire Safety Survey Date"):
                if column == TOTALS[1] or column in fire_columns:
                    continue  # Preserve CMS missing values; never normalize to zero.
            if not re.fullmatch(r"[0-9]+", value):
                errors["count"] += 1
    duplicate_keys = sum(n - 1 for n in keys.values() if n > 1)
    if duplicate_keys:
        errors["duplicate_key"] = duplicate_keys
    if not rows:
        errors["empty"] += 1
    return {"status": "PASS" if not errors else "FAIL", "errors": dict(errors),
            "row_count": rows, "provider_count": len(providers), "columns": columns,
            "schema_sha256": hashlib.sha256(json.dumps(columns).encode()).hexdigest(),
            "grain": "provider inspection cycle", "key": list(KEY), "duplicate_key_rows": duplicate_keys,
            "ccn_health_survey_date_duplicate_rows": sum(n - 1 for n in dated_keys.values() if n > 1),
            "missing_survey_dates_allowed": True, "foreign_keys": []}


def _retain(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError(f"Immutable retained artifact differs: {path}")


def probe_snapshot(snap):
    """Project the control-plane candidate into the existing Sources UI."""
    from release_control_plane import candidate_local_path, load_candidates
    candidate = load_candidates().get("datasets", {}).get(SOURCE_ID) or {}
    source = candidate_local_path(candidate.get("source_uri"))
    validation = candidate.get("validation") or {}
    snap.local_raw_present = bool(source and source.is_file())
    snap.raw_available = candidate.get("release_id") or "—"
    snap.release_id = candidate.get("release_id")
    snap.canonical_source_path = str(source) if source else None
    snap.structural_status = validation.get("status") or "NOT_RUN"
    snap.status = "READY_FOR_HANDOFF" if candidate.get("state") == "VALIDATED" and validation.get("status") == "PASS" else "PROCESSING_REQUIRED" if candidate else "UNKNOWN"
    snap.detail = "Survey Summary candidate only; CCN + Inspection Cycle; explicit review required, no automatic ACTIVE"
    return snap


def prepare_candidate(*, root: Path | None = None, fetch=fetch_json, download=None,
                      retained_source: Path | None = None) -> dict:
    root = control_plane_root(root)
    metadata = fetch(METASTORE + DATASET_ID + "?show-reference-ids=true")
    current = normalize_dataset(metadata, DATASET_ID)
    resources = [r for r in current["resources"] if r["filename"].startswith("NH_SurveySummary_") and r["filename"].endswith(".csv")]
    if len(resources) != 1:
        raise ValueError("Expected exactly one official Survey Summary CSV")
    resource = resources[0]
    release = current["logical_release"] + "-" + current["artifact_identity"][:12]
    record_candidate(SOURCE_ID, release, ReleaseState.DETECTED, metadata=current, root=root)
    if download is None:
        def download(url):
            with _request(url) as response:
                raw = response.read(32 * 1024 * 1024 + 1)
                if len(raw) > 32 * 1024 * 1024:
                    raise ValueError("Survey Summary exceeds bounded download size")
                return raw
    official = download(resource["url"])
    reused = retained_source is not None and retained_source.read_bytes() == official
    raw = retained_source.read_bytes() if reused else official
    sha = hashlib.sha256(raw).hexdigest()
    release += "-" + sha[:12]
    directory = root / "state" / "source_artifacts" / SOURCE_ID / sha
    source = directory / "source.csv"
    _retain(source, raw)
    metadata_bytes = json.dumps(metadata, sort_keys=True, indent=2).encode()
    meta_sha = hashlib.sha256(metadata_bytes).hexdigest()
    metadata_path = directory / f"metadata-{meta_sha}.json"
    _retain(metadata_path, metadata_bytes)
    provenance = {**current, "sha256": sha, "publisher_sha256": sha,
                  "acquired_at": datetime.now(timezone.utc).isoformat(),
                  "byte_count": len(raw), "metadata_sha256": meta_sha,
                  "metadata_path": str(metadata_path), "resource": resource,
                  "retained_source_reused": str(retained_source) if reused else None}
    record_candidate(SOURCE_ID, release, ReleaseState.ACQUIRED, source_path=source, metadata=provenance, root=root)
    try:
        validation = validate_csv(raw, modified=current["modified"])
    except (UnicodeError, csv.Error, ValueError) as exc:
        validation = {"status": "FAIL", "errors": {"source_parse": str(exc)}, "row_count": None}
    receipt = {**validation, "dataset_id": SOURCE_ID, "release_id": release, "sha256": sha,
               "validated_at": datetime.now(timezone.utc).isoformat(), "provenance": provenance}
    receipt_bytes = json.dumps(receipt, sort_keys=True, indent=2).encode()
    receipt_path = directory / ("validation-" + hashlib.sha256(receipt_bytes).hexdigest() + ".json")
    _retain(receipt_path, receipt_bytes)
    validation["receipt_path"] = str(receipt_path)
    return record_candidate(SOURCE_ID, release, ReleaseState.VALIDATED if validation["status"] == "PASS" else ReleaseState.FAILED,
                            source_path=source, validation=validation, metadata={**provenance, "review_ready": validation["status"] == "PASS"}, root=root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Survey Summary for review; does not promote ACTIVE")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--retained-source", type=Path)
    args = parser.parse_args()
    candidate = prepare_candidate(root=args.root, retained_source=args.retained_source)
    print(json.dumps({k: candidate[k] for k in ("dataset_id", "release_id", "state", "hash", "validation")}, indent=2))
    raise SystemExit(0 if candidate["state"] == "VALIDATED" else 1)
