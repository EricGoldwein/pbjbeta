"""Validate explicitly trusted local PBJ sources and publish the initial ACTIVE registry.

This bootstrap is intentionally path-explicit. It never discovers ACTIVE by mtime.
"""

from __future__ import annotations

import csv
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from active_release_registry import promote_release, registry_path, sha256_file

ROOT = Path(r"C:\Users\egold\PycharmProjects\PBJapp")
LATEST_STAFFING = "CY2026Q1"
PROVIDER_RELEASE = "2026-07"
OWNERSHIP_RELEASE = "2026-07-17"


class InitialValidationError(RuntimeError):
    pass


def _csv_header(path: Path, required: tuple[str, ...] = ()) -> list[str]:
    if not path.is_file() or path.stat().st_size <= 20:
        raise InitialValidationError(f"missing/empty trusted source: {path}")
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header = next(csv.reader(handle), [])
    normalized = {str(value).strip().upper() for value in header}
    missing = [name for name in required if name.upper() not in normalized]
    if not header or missing:
        raise InitialValidationError(f"invalid schema {path.name}; missing {missing or 'header'}")
    return header


def _staffing_files(directory: Path, prefix: str) -> list[Path]:
    files = [p for p in directory.glob(f"{prefix}_CY*.csv") if p.is_file()]
    def key(path: Path) -> tuple[int, int]:
        match = re.search(r"CY(\d{4})Q([1-4])", path.name, re.I)
        return (int(match.group(1)), int(match.group(2))) if match else (0, 0)
    files.sort(key=key)
    if not files or LATEST_STAFFING not in files[-1].name.upper():
        raise InitialValidationError(f"trusted staffing series does not end at {LATEST_STAFFING}: {directory}")
    return files


def _manifest_member(manifest: dict[str, Any], basename: str) -> dict[str, Any]:
    for item in manifest.get("source_members") or []:
        if item.get("basename") == basename:
            return item
    raise InitialValidationError(f"provider manifest missing {basename}")


def _verify_manifest_output(manifest: dict[str, Any], basename: str, path: Path) -> None:
    member = _manifest_member(manifest, basename)
    expected = ""
    for output in member.get("normalized_outputs") or []:
        if Path(str(output.get("path") or "")).name == path.name or output.get("artifact") == "ProviderInfoNorm":
            expected = str(output.get("sha256") or "")
            break
    if not expected or sha256_file(path) != expected:
        raise InitialValidationError(f"manifest hash mismatch: {path}")


def _promote(dataset_id: str, release_id: str, source: Path, validated_at: str, *, metadata=None) -> dict[str, Any]:
    return promote_release(
        dataset_id,
        release_id,
        source,
        validated_at=validated_at,
        release_date=release_id,
        metadata=metadata or {},
        root=Path(__file__).resolve().parent,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-time migration bootstrap for an empty ACTIVE release registry."
    )
    parser.add_argument(
        "--acknowledge-initial-migration",
        action="store_true",
        help="Required acknowledgement that this is the one-time initial migration only.",
    )
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Read-only verification of an already-published registry; never promotes.",
    )
    args = parser.parse_args()
    target_registry = registry_path(Path(__file__).resolve().parent)
    if args.verify_existing:
        if not target_registry.is_file():
            raise InitialValidationError(f"ACTIVE registry does not exist: {target_registry}")
        payload = json.loads(target_registry.read_text(encoding="utf-8"))
        datasets = payload.get("datasets") or {}
        if len(datasets) != 13:
            raise InitialValidationError("existing initial registry does not contain 13 datasets")
        for dataset_id, record in datasets.items():
            uri = str(record.get("source_uri") or "")
            if not uri.startswith("file:///"):
                raise InitialValidationError(f"{dataset_id}: bootstrap verification requires local file URI")
            from urllib.parse import unquote, urlparse

            raw = unquote(urlparse(uri).path)
            if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
                raw = raw[1:]
            source = Path(raw)
            if not source.is_file() or sha256_file(source) != record.get("hash"):
                raise InitialValidationError(f"{dataset_id}: ACTIVE source missing or hash mismatch")
            for member in (record.get("metadata") or {}).get("source_set") or []:
                member_uri = str(member.get("source_uri") or "")
                member_raw = unquote(urlparse(member_uri).path)
                if member_raw.startswith("/") and len(member_raw) > 2 and member_raw[2] == ":":
                    member_raw = member_raw[1:]
                member_path = Path(member_raw)
                if not member_path.is_file() or sha256_file(member_path) != member.get("hash"):
                    raise InitialValidationError(f"{dataset_id}: source-set member missing or hash mismatch")
        print("PASS: all 13 existing ACTIVE releases and source hashes verified (read-only)")
        return 0
    if not args.acknowledge_initial_migration:
        parser.error("--acknowledge-initial-migration is required")
    if target_registry.exists():
        raise InitialValidationError(
            "initial bootstrap is migration-only and refuses to overwrite an existing ACTIVE registry"
        )
    validated_at = datetime.now(timezone.utc).isoformat()
    evidence: dict[str, Any] = {"validated_at": validated_at, "status": "PASS", "datasets": {}}

    provider_manifest_path = ROOT / "provider_info" / "_manifests" / PROVIDER_RELEASE / "release_manifest.json"
    provider_manifest = json.loads(provider_manifest_path.read_text(encoding="utf-8"))
    provider = ROOT / "provider_info_normalized" / "ProviderInfoNorm_2026_07.csv"
    nh_ownership = ROOT / "ownership" / "NH_Ownership_Jul2026.csv"
    citations = ROOT / "Citations" / "NH_HealthCitations_Jul2026.csv"
    _csv_header(provider, ("ccn", "provider_name", "state"))
    _csv_header(nh_ownership, ("CMS Certification Number (CCN)", "Owner Name"))
    _csv_header(citations, ("CMS Certification Number (CCN)", "Survey Date"))
    _verify_manifest_output(provider_manifest, "NH_ProviderInfo_Jul2026.csv", provider)
    _verify_manifest_output(provider_manifest, "NH_Ownership_Jul2026.csv", nh_ownership)
    _verify_manifest_output(provider_manifest, "NH_HealthCitations_Jul2026.csv", citations)
    _promote(
        "cms.provider_info", PROVIDER_RELEASE, provider, validated_at,
        metadata={"validation_evidence": str(provider_manifest_path), "source_set": [
            {"role": "provider_info", "source_path": str(provider)},
            {"role": "nh_ownership", "source_path": str(nh_ownership)},
        ]},
    )
    _promote("cms.health_citations", PROVIDER_RELEASE, citations, validated_at, metadata={"validation_evidence": str(provider_manifest_path)})

    nurse_files = _staffing_files(ROOT / "standardized_PBJ", "PBJ_dailynursestaffing")
    nonnurse_files = _staffing_files(ROOT / "standardized_NonNurse", "PBJ_dailynonnursestaffing")
    _csv_header(nurse_files[-1], ("PROVNUM", "WorkDate", "CY_Qtr"))
    _csv_header(nonnurse_files[-1], ("PROVNUM", "WorkDate", "CY_Qtr"))
    _promote(
        "cms.pbj_nurse_staffing", LATEST_STAFFING, nurse_files[-1], validated_at,
        metadata={"source_set": [{"role": re.search(r"CY\d{4}Q[1-4]", p.name, re.I).group(0).upper(), "source_path": str(p)} for p in nurse_files]},
    )
    _promote(
        "cms.pbj_non_nurse_staffing", LATEST_STAFFING, nonnurse_files[-1], validated_at,
        metadata={"source_set": [{"role": re.search(r"CY\d{4}Q[1-4]", p.name, re.I).group(0).upper(), "source_path": str(p)} for p in nonnurse_files]},
    )

    policy_path = ROOT / "ownership" / "ownership_release_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if policy.get("active_release_date") != OWNERSHIP_RELEASE:
        raise InitialValidationError("ownership policy active release mismatch")
    entry = policy["releases"][OWNERSHIP_RELEASE]
    owners = ROOT / "ownership" / entry["ownership_source_filename"]
    enrollments = ROOT / "ownership" / "_sources" / "cms_snf_enrollments" / "raw" / "downloaded" / entry["enrollment_source_filename"]
    _csv_header(owners, ("ENROLLMENT ID",))
    _csv_header(enrollments, ("ENROLLMENT ID", "CCN"))
    if sha256_file(owners) != entry["ownership_source_sha256"] or sha256_file(enrollments) != entry["enrollment_source_sha256"]:
        raise InitialValidationError("ownership/enrollment policy hash mismatch")
    _promote("cms.snf_all_owners", OWNERSHIP_RELEASE, owners, validated_at, metadata={"validation_evidence": str(policy_path)})
    _promote("cms.snf_enrollments", OWNERSHIP_RELEASE, enrollments, validated_at, metadata={"validation_evidence": str(policy_path)})

    derived = {
        "pbj.benchmarks.state": ROOT / "state_quarterly_metrics.csv",
        "pbj.benchmarks.national": ROOT / "national_quarterly_metrics.csv",
        "pbj.benchmarks.region": ROOT / "deployments" / "pbj320-335581" / "cms_region_quarterly_metrics.csv",
        "pbj.benchmarks.region_mapping": ROOT / "cms_region_state_mapping.csv",
        "pbj.benchmarks.geo_cmi": ROOT / "pbj_lite" / "geo_nursing_cmi_quarterly.csv",
        "pbj.peer_distribution": ROOT / "pbj_lite" / "facility_lite_metrics.csv",
        "macpac.state_staffing_standards": ROOT / "pbj_lite" / "macpac_state_standards_clean.csv",
    }
    for dataset_id, source in derived.items():
        _csv_header(source)
        release_id = f"sha256:{sha256_file(source)[:16]}"
        _promote(dataset_id, release_id, source, validated_at, metadata={"validation": "nonempty CSV schema/header"})

    registry = json.loads(registry_path(Path(__file__).resolve().parent).read_text(encoding="utf-8"))
    evidence["datasets"] = {
        key: {"active_release_id": value["active_release_id"], "hash": value["hash"], "source_uri": value["source_uri"]}
        for key, value in registry["datasets"].items()
    }
    evidence_path = Path(__file__).resolve().parent / "state" / "initial_active_release_validation.json"
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS: published {len(evidence['datasets'])} ACTIVE releases")
    print(f"Registry: {registry_path(Path(__file__).resolve().parent)}")
    print(f"Evidence: {evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
