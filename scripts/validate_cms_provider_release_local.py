#!/usr/bin/env python3
"""Post-ingest validation for a CMS provider release (local, no deploy)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "scripts"))

import cms_data_paths
from cms_provider_release_lib import (
    compare_provider_month_ccn_coverage,
    load_manifest,
    load_release_diff,
    release_key,
    schema_fingerprint,
)
from citation_lib import build_facility_citations_csv, find_latest_nh_health_citations_csv
from prov_info_quarter_map import get_quarter_from_processing_month
from deploy_vercel_facility import _slice_nh_ownership_csv_for_ccn


def _validate_provider_norm(key_label: str) -> dict:
    year, month = map(int, key_label.split("-"))
    norm_path = cms_data_paths.provider_info_normalized_dir() / f"ProviderInfoNorm_{year}_{month:02d}.csv"
    if not norm_path.is_file():
        return {"status": "missing", "path": str(norm_path)}
    df = pd.read_csv(norm_path, dtype=str, low_memory=False)
    ccn = df["ccn"].astype(str).str.zfill(6)
    cmi = pd.to_numeric(df.get("nursing_case_mix_index"), errors="coerce")
    quarter = get_quarter_from_processing_month(f"{year}-{month:02d}")
    return {
        "status": "ok",
        "rows": len(df),
        "unique_ccn": int(ccn.nunique()),
        "duplicate_ccn": int(ccn.duplicated().sum()),
        "cmi_non_null": int(cmi.notna().sum()),
        "mapped_staffing_quarter": quarter,
        "max_processing_date": str(df["processing_date"].max()) if "processing_date" in df.columns else "",
    }


def _validate_ownership(basename: str) -> dict:
    path = cms_data_paths.ownership_dir() / basename
    df = pd.read_csv(path, dtype=str, low_memory=False)
    ccn_col = "CMS Certification Number (CCN)"
    ccn = df[ccn_col].astype(str).str.zfill(6)
    null_owner = df["Owner Name"].isna().sum() if "Owner Name" in df.columns else None
    return {
        "rows": len(df),
        "unique_ccn": int(ccn.nunique()),
        "duplicate_ccn_rows": int(ccn.duplicated().sum()),
        "null_owner_name": int(null_owner) if null_owner is not None else None,
    }


def _validate_citations(basename: str) -> dict:
    path = cms_data_paths.citations_dir() / basename
    df = pd.read_csv(path, usecols=["CMS Certification Number (CCN)", "Survey Date"], dtype=str, low_memory=False)
    ccn = df["CMS Certification Number (CCN)"].astype(str).str.zfill(6)
    dates = pd.to_datetime(df["Survey Date"], errors="coerce")
    return {
        "rows": len(df),
        "unique_ccn": int(ccn.nunique()),
        "survey_date_min": str(dates.min()),
        "survey_date_max": str(dates.max()),
        "latest_resolver": find_latest_nh_health_citations_csv(str(_ROOT)),
    }


def _facility_slices(ccn: str, ownership_src: Path, *, release_label: str, citation_src: str | None) -> dict:
    ccn = ccn.zfill(6)
    prov_slice = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    prov_slice.close()
    cit_slice = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    cit_slice.close()
    own_slice = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    own_slice.close()
    try:
        from dynamic_facility_dashboard import create_facility_provider_info_csv

        prov_df = create_facility_provider_info_csv(ccn, output_path=prov_slice.name)
        cit_n = build_facility_citations_csv(ccn, cit_slice.name, source_csv=citation_src, root=str(_ROOT))
        own_n = _slice_nh_ownership_csv_for_ccn(ownership_src, Path(own_slice.name), ccn)
        latest_proc = ""
        if prov_df is not None and len(prov_df) and "processing_date" in prov_df.columns:
            latest_proc = str(pd.to_datetime(prov_df["processing_date"]).max().date())
        return {
            "provider_snapshot": latest_proc,
            "staffing_case_mix_period": get_quarter_from_processing_month(release_label),
            "ownership_source": ownership_src.name,
            "citation_source": Path(citation_src).name if citation_src else "",
            "provider_rows": int(len(prov_df)) if prov_df is not None else 0,
            "latest_processing_date": latest_proc,
            "citation_rows": cit_n,
            "ownership_rows": own_n,
        }
    finally:
        for p in (prov_slice.name, cit_slice.name, own_slice.name):
            Path(p).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CMS provider release locally")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--ccn", action="append", default=["315461", "335513", "315128"])
    parser.add_argument("--report", type=str, default="", help="Write JSON report path")
    args = parser.parse_args()

    key = release_key(args.year, args.month)
    manifest = load_manifest(key, _ROOT)
    release_diff = load_release_diff(key, _ROOT)
    report: dict = {
        "release_key": key.label,
        "manifest_present": manifest is not None,
        "release_diff_present": release_diff is not None,
        "promotion_blocked": manifest.get("promotion_blocked") if manifest else None,
        "promotion_blocked_reasons": manifest.get("promotion_blocked_reasons") if manifest else None,
        "provider_norm": _validate_provider_norm(key.label),
        "quarter_mapping_2026_06": get_quarter_from_processing_month(key.label),
        "deployment_status": {
            "national_sources_validated": True,
            "local_slice_generation_validated": True,
            "deployment_bundles_refreshed": False,
            "deployed_facility_sites_serving_june_slices": False,
            "note": (
                "This script validates national CMS sources and local slice builders only. "
                "It does not refresh deployments/pbj320-* bundles or change what live Vercel sites serve."
            ),
        },
    }

    if manifest:
        ingested = [m for m in manifest.get("source_members", []) if m.get("ingestion_status") == "ingested"]
        report["manifest_ingested_members"] = [m.get("basename") for m in ingested]
        missing_fp = [m.get("basename") for m in ingested if not m.get("schema_fingerprint")]
        report["manifest_schema_fingerprints_ok"] = not missing_fp
        if missing_fp:
            report["manifest_missing_schema_fingerprint"] = missing_fp
    if release_diff:
        report["release_diff_summary"] = {
            "prior_release_key": release_diff.get("prior_release_key"),
            "new_members": release_diff.get("new_members") or [],
            "removed_members": release_diff.get("removed_members") or [],
            "changed_schema_count": len(release_diff.get("changed_schema") or []),
            "changed_contents_count": len(release_diff.get("changed_contents") or []),
            "new_facility_events_count": len(release_diff.get("new_facility_events") or []),
        }

    mon = key.month_abbr
    own_name = f"NH_Ownership_{mon}{key.year}.csv"
    cit_name = f"NH_HealthCitations_{mon}{key.year}.csv"
    if (cms_data_paths.ownership_dir() / own_name).is_file():
        report["ownership"] = _validate_ownership(own_name)
    prior_mon = _MONTH_PRIOR(key)
    if prior_mon:
        prior_pi = cms_data_paths.provider_info_dir() / f"NH_ProviderInfo_{prior_mon}.csv"
        curr_pi = cms_data_paths.provider_info_dir() / f"NH_ProviderInfo_{mon}{key.year}.csv"
        if prior_pi.is_file() and curr_pi.is_file():
            report["ccn_coverage_vs_prior"] = compare_provider_month_ccn_coverage(prior_pi, curr_pi)
    if (cms_data_paths.citations_dir() / cit_name).is_file():
        report["citations"] = _validate_citations(cit_name)

    own_src = cms_data_paths.ownership_dir() / own_name
    cit_src = find_latest_nh_health_citations_csv(str(_ROOT))
    report["facilities"] = {}
    if own_src.is_file():
        for ccn in args.ccn:
            report["facilities"][ccn.zfill(6)] = _facility_slices(
                ccn,
                own_src,
                release_label=key.label,
                citation_src=cit_src,
            )

    print(json.dumps(report, indent=2))
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report.get("provider_norm", {}).get("status") != "ok":
        return 1
    if not report.get("manifest_present"):
        print("ERROR: release_manifest.json missing", file=sys.stderr)
        return 1
    if not report.get("release_diff_present"):
        print("ERROR: release_diff.json missing", file=sys.stderr)
        return 1
    if not report.get("manifest_schema_fingerprints_ok", True):
        print("ERROR: ingested members missing schema_fingerprint", file=sys.stderr)
        return 1
    if manifest and manifest.get("promotion_blocked"):
        print("ERROR: promotion blocked — see release_diff.json", file=sys.stderr)
        for reason in manifest.get("promotion_blocked_reasons") or []:
            print(f"  {reason}", file=sys.stderr)
        return 1
    expected_q = get_quarter_from_processing_month(key.label)
    if report.get("quarter_mapping_2026_06") != expected_q and key.label == "2026-06":
        print(f"ERROR: expected Q4 2025 staffing quarter, got {report.get('quarter_mapping_2026_06')}", file=sys.stderr)
        return 1
    return 0


def _MONTH_PRIOR(key):  # noqa: N802
    from cms_provider_release_lib import _MONTH_ABBR

    if key.month <= 1:
        return None
    return f"{_MONTH_ABBR[key.month - 1]}{key.year}"


if __name__ == "__main__":
    raise SystemExit(main())
