"""Tests for CMS provider monthly release ingest (scripts/cms_provider_release_lib.py)."""
from __future__ import annotations
import io
import json
import sys
import zipfile
from pathlib import Path
import pytest
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import cms_data_paths
from cms_provider_release_lib import (
    build_release_diff,
    classification_policy,
    classify_member_basename,
    compute_promotion_blocked,
    expected_active_csv_names,
    extract_active_csvs,
    _ingestion_status_for_member,
    load_manifest,
    load_release_diff,
    release_key,
    schema_fingerprint,
)
from prov_info_quarter_map import get_quarter_from_processing_month
def _write_minimal_release_zips(root: Path, key, *, extra_files: dict[str, str] | None = None) -> None:
    pi = root / "provider_info"
    pi.mkdir(parents=True, exist_ok=True)
    (root / "ownership").mkdir(exist_ok=True)
    (root / "Citations").mkdir(exist_ok=True)
    inner_files = {
        f"NH_ProviderInfo_{key.month_abbr}{key.year}.csv": "CMS Certification Number (CCN),Provider Name\n015009,Test\n",
        f"NH_DataCollectionIntervals_{key.month_abbr}{key.year}.csv": (
            "Measure Code,Measure Description,Data Collection Period From Date,"
            "Data Collection Period Through Date,Measure Date Range,Processing Date\n"
            "STAFFING_LEVELS,Staffing,10/01/2025,12/31/2025,,20260601\n"
        ),
        f"NH_Ownership_{key.month_abbr}{key.year}.csv": "CMS Certification Number (CCN),Owner Name\n015009,Owner A\n",
        f"NH_HealthCitations_{key.month_abbr}{key.year}.csv": (
            "CMS Certification Number (CCN),Survey Date\n015009,01/15/2024\n"
        ),
        f"NH_CitationDescriptions_{key.month_abbr}{key.year}.csv": "Deficiency Tag Number,Description\n1,Test\n",
        "readme.txt": "cms\n",
    }
    if extra_files:
        inner_files.update(extra_files)
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as z_inner:
        for name, body in inner_files.items():
            z_inner.writestr(name, body)
    inner_name = f"nursing_homes_including_rehab_services_{key.month:02d}_{key.year}.zip"
    outer = pi / f"nursing_homes_including_rehab_services_{key.year}.zip"
    inner_payload = inner_buf.getvalue()
    if outer.is_file():
        with zipfile.ZipFile(outer, "r") as z_read:
            existing = {info.filename: z_read.read(info.filename) for info in z_read.infolist()}
        existing[inner_name] = inner_payload
        with zipfile.ZipFile(outer, "w") as z_outer:
            for name, body in existing.items():
                z_outer.writestr(name, body)
    else:
        with zipfile.ZipFile(outer, "w") as z_outer:
            z_outer.writestr(inner_name, inner_payload)
def test_expected_active_csv_names_june() -> None:
    key = release_key(2026, 6)
    names = expected_active_csv_names(key)
    assert "NH_ProviderInfo_Jun2026.csv" in names
    assert names["NH_Ownership_Jun2026.csv"] == "ownership"
    assert names["NH_HealthCitations_Jun2026.csv"] == "citations"
def test_schema_fingerprint_sorted_normalized_columns() -> None:
    fp1 = schema_fingerprint(["CCN", "Provider Name"])
    fp2 = schema_fingerprint(["provider name", "  CCN  "])
    assert fp1 == fp2
    fp3 = schema_fingerprint(["CCN", "Provider Name", "State"])
    assert fp1 != fp3
def test_classify_member_basename() -> None:
    family, dtype = classify_member_basename("NH_ProviderInfo_Jun2026.csv")
    assert family == "provider_info"
    assert dtype == "provider_info"
    family2, _ = classify_member_basename("NH_FireSafetyCitations_Jun2026.csv")
    assert family2 == "citations"
    family3, _ = classify_member_basename("totally_new_file.csv")
    assert family3 is None
def test_extract_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PBJ_REPO_ROOT", str(tmp_path))
    key = release_key(2026, 6)
    _write_minimal_release_zips(tmp_path, key)
    m1 = extract_active_csvs(key, root=tmp_path)
    m2 = extract_active_csvs(key, root=tmp_path)
    assert len(m1["extracted_active_files"]) == 5
    assert m2["skipped_identical"] == [
        "NH_ProviderInfo_Jun2026.csv",
        "NH_DataCollectionIntervals_Jun2026.csv",
        "NH_Ownership_Jun2026.csv",
        "NH_HealthCitations_Jun2026.csv",
        "NH_CitationDescriptions_Jun2026.csv",
    ]
    manifest = load_manifest(key, tmp_path)
    assert manifest is not None
    assert manifest["release_key"] == "2026-06"
    assert manifest["inner_archive"]["member_name"].endswith("06_2026.zip")
    ingested = [m for m in manifest["source_members"] if m["ingestion_status"] == "ingested"]
    assert len(ingested) == 5
    assert all(m.get("schema_fingerprint") for m in ingested)
    assert load_release_diff(key, tmp_path) is not None
def test_release_diff_vs_prior_month(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PBJ_REPO_ROOT", str(tmp_path))
    may = release_key(2026, 5)
    june = release_key(2026, 6)
    _write_minimal_release_zips(tmp_path, may)
    extract_active_csvs(may, root=tmp_path)
    _write_minimal_release_zips(
        tmp_path,
        june,
        extra_files={"NH_SurpriseNewDrop_Jun2026.csv": "col_a,col_b\n1,2\n"},
    )
    manifest = extract_active_csvs(june, root=tmp_path)
    diff = manifest["release_diff"]
    assert diff["prior_release_key"] == "2026-05"
    assert "NH_SurpriseNewDrop_Jun2026.csv" in diff["new_members"]
def test_promotion_blocked_on_unmapped_new_source() -> None:
    manifest = {
        "source_members": [
            {
                "basename": "NH_Mystery_Jun2026.csv",
                "ingestion_status": "unmapped_new_source",
            }
        ]
    }
    diff = {"new_members": ["NH_Mystery_Jun2026.csv"], "changed_schema": [], "removed_members": []}
    blocked, reasons = compute_promotion_blocked(manifest, diff, prior_manifest={"source_members": []})
    assert blocked
    assert any("unmapped new source" in r for r in reasons)
def test_promotion_blocked_on_ingested_schema_change() -> None:
    manifest = {
        "source_members": [
            {
                "basename": "NH_ProviderInfo_Jun2026.csv",
                "ingestion_status": "ingested",
                "schema_fingerprint": "bbb",
                "source_sha256": "bbb",
            }
        ]
    }
    prior = {
        "source_members": [
            {
                "basename": "NH_ProviderInfo_Jun2026.csv",
                "ingestion_status": "ingested",
                "schema_fingerprint": "aaa",
                "source_sha256": "aaa",
            }
        ]
    }
    diff = {
        "new_members": [],
        "removed_members": [],
        "changed_schema": [
            {
                "basename": "NH_ProviderInfo_Jun2026.csv",
                "ingestion_status": "ingested",
            }
        ],
        "changed_contents": [],
    }
    blocked, reasons = compute_promotion_blocked(manifest, diff, prior_manifest=prior)
    assert blocked
    assert any("schema change" in r for r in reasons)
def test_refuse_overwrite_different_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PBJ_REPO_ROOT", str(tmp_path))
    key = release_key(2026, 6)
    _write_minimal_release_zips(tmp_path, key)
    extract_active_csvs(key, root=tmp_path)
    path = cms_data_paths.provider_info_dir(tmp_path) / "NH_ProviderInfo_Jun2026.csv"
    path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        extract_active_csvs(key, root=tmp_path)
def test_june_2026_former_blockers_classified_retained() -> None:
    """Eight June 2026 members previously unmapped must be retained_unmodeled, not blocking."""
    key = release_key(2026, 6)
    blockers = [
        "NH_Data_Dictionary.pdf",
        "NH_HlthInspecCutpointsState_Jun2026.csv",
        "NH_QualityMsr_Claims_Jun2026.csv",
        "NH_QualityMsr_MDS_Jun2026.csv",
        "NH_StateUSAverages_Jun2026.csv",
        "readme.txt",
        "Skilled_Nursing_Facility_Quality_Reporting_Program_National_Data_Apr2026.csv",
        "Skilled_Nursing_Facility_Quality_Reporting_Program_Provider_Data_Jun2026.csv",
    ]
    members = []
    for basename in blockers:
        family, _ = classify_member_basename(basename)
        status = _ingestion_status_for_member(key, basename, family, prior_member=None)
        policy = classification_policy(family, basename, status)
        members.append(
            {
                "basename": basename,
                "ingestion_status": status,
                "classification_policy": policy,
            }
        )
        assert status == "retained_unmodeled", basename
        assert policy in {
            "documentation_reference",
            "adjacent_cms_program",
        }, basename
    manifest = {"source_members": members}
    diff = {"new_members": blockers, "changed_schema": [], "removed_members": []}
    blocked, reasons = compute_promotion_blocked(manifest, diff, prior_manifest={"source_members": []})
    assert not blocked, reasons
    assert not reasons
def test_interval_mapping_json_has_june() -> None:
    path = Path(__file__).resolve().parents[1] / "static" / "data" / "interval_quarter_mapping.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["processing_month"]: r for r in data["rows"]}
    assert "06-2026" in rows
    assert rows["06-2026"]["interval_staffing_level_quarter"] == "Q4 2025"
    assert rows["06-2026"]["interval_csv_name"] == "NH_DataCollectionIntervals_Jun2026.csv"
    assert "08-2026" in rows
    assert rows["08-2026"]["interval_staffing_level_quarter"] == "Q1 2026"
