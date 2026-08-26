"""Canonical CMS source registry for PBJ Data Ops (v0).

Populate only fields verified from repo evidence or live CMS metastore calls.
Do not invent dataset IDs, cadences, or schedules.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class SourceFormat(str, Enum):
    CSV = "CSV"
    ZIP = "ZIP"
    ZIP_MEMBER = "ZIP-member"
    PDF = "PDF"
    OTHER = "other"
    MIXED = "mixed"  # e.g. CSV column + optional PDF list


class SourceCadence(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    IRREGULAR = "irregular"
    FOLLOWS_PROVIDER_INFO = "follows_provider_info"
    UNKNOWN = "unknown"


class AutomationLevel(str, Enum):
    """Highest verified automation maturity for the family on main."""

    DETECTION_EXISTS = "detection_exists"
    ACQUISITION_EXISTS = "acquisition_exists"
    NORMALIZATION_EXISTS = "normalization_exists"
    VALIDATION_EXISTS = "validation_exists"
    FULLY_MANUAL = "fully_manual"
    BROKEN_LEGACY = "broken_legacy"


class OpsStatus(str, Enum):
    CURRENT = "CURRENT"
    CMS_NEWER = "CMS_NEWER"
    LOCAL_RAW_ONLY = "LOCAL_RAW_ONLY"
    PROCESSING_REQUIRED = "PROCESSING_REQUIRED"
    READY_FOR_HANDOFF = "READY_FOR_HANDOFF"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class SourceFamily(str, Enum):
    PROVIDER_INFO = "provider_info"
    PBJ_NURSE = "pbj_nurse_staffing"
    PBJ_NON_NURSE = "pbj_non_nurse_staffing"
    PBJ_EIN = "pbj_employee_ein_detail"
    SNF_ALL_OWNERS = "snf_all_owners"
    SNF_ENROLLMENTS = "snf_enrollments"
    SNF_CHOW = "snf_chow"
    CHAIN_PERFORMANCE = "chain_performance"
    SFF = "sff"


@dataclass(frozen=True)
class CmsSourceRecord:
    """Static registry row — verified facts only; runtime status is separate."""

    source_id: str
    human_name: str
    source_family: SourceFamily
    formats: tuple[SourceFormat, ...]
    cadence: SourceCadence
    release_version_strategy: str
    cms_dataset_id: Optional[str]
    cms_metadata_endpoint: Optional[str]
    cms_landing_url: Optional[str]
    local_raw_location: str
    normalized_derived_location: str
    validator: Optional[str]
    downstream_consumers: tuple[str, ...]
    automation_level: AutomationLevel
    automation_notes: str
    actions_enabled: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source_family"] = self.source_family.value
        d["formats"] = [f.value for f in self.formats]
        d["cadence"] = self.cadence.value
        d["automation_level"] = self.automation_level.value
        return d


# Verified Provider Info metastore (PR #63).
_PI_DATASET = "4pq5-n9py"
_PI_METASTORE = (
    "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/"
    f"{_PI_DATASET}?show-reference-ids=true"
)


CMS_SOURCE_REGISTRY: tuple[CmsSourceRecord, ...] = (
    CmsSourceRecord(
        source_id="cms.provider_info",
        human_name="Provider Information",
        source_family=SourceFamily.PROVIDER_INFO,
        formats=(SourceFormat.CSV, SourceFormat.ZIP, SourceFormat.ZIP_MEMBER),
        cadence=SourceCadence.MONTHLY,
        release_version_strategy=(
            "Filename NH_ProviderInfo_{Mon}{YYYY}.csv; release key YYYY-MM; "
            "SHA256 acquisition record under provider_info/_manifests/"
        ),
        cms_dataset_id=_PI_DATASET,
        cms_metadata_endpoint=_PI_METASTORE,
        cms_landing_url=f"https://data.cms.gov/provider-data/dataset/{_PI_DATASET}",
        local_raw_location="provider_info/NH_ProviderInfo_*.csv (+ yearly zip archive path)",
        normalized_derived_location=(
            "provider_info_normalized/ProviderInfoNorm_{YYYY}_{MM}.csv; "
            "provider_info/_manifests/{YYYY-MM}/"
        ),
        validator="scripts/cms_provider_info_acquire.validate_raw_provider_info_csv",
        downstream_consumers=(
            "normalize_provider_info.py",
            "PBJ_Dashboard.py",
            "dynamic_facility_dashboard.py",
            "prov_info.py",
            "pbj-root handoff (contract only)",
        ),
        automation_level=AutomationLevel.VALIDATION_EXISTS,
        automation_notes=(
            "Detection + CMS metastore acquire + normalize + validation + handoff "
            "manifest (PR #63). No pbj-root write."
        ),
        actions_enabled=("check_cms", "acquire_process"),
        evidence=(
            "scripts/cms_provider_info_acquire.py",
            "docs/CMS_PROVIDER_RELEASE_WORKFLOW.md",
            "provider_info/_manifests/",
        ),
    ),
    CmsSourceRecord(
        source_id="cms.pbj_nurse_staffing",
        human_name="PBJ nurse staffing",
        source_family=SourceFamily.PBJ_NURSE,
        formats=(SourceFormat.CSV,),
        cadence=SourceCadence.QUARTERLY,
        release_version_strategy="Filename PBJ_dailynursestaffing_CY{YYYY}Q{n}.csv",
        cms_dataset_id=None,
        cms_metadata_endpoint=None,
        cms_landing_url=(
            "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing"
        ),
        local_raw_location="PBJcsv/",
        normalized_derived_location="standardized_PBJ/",
        validator="standardize_pbj_files.py (structure warnings); packaging gates",
        downstream_consumers=(
            "standardize_pbj_files.py",
            "generate_metrics.py",
            "run_pipeline_update.py",
            "PBJ_Dashboard.py",
        ),
        automation_level=AutomationLevel.NORMALIZATION_EXISTS,
        automation_notes=(
            "Detection + standardize exist; no CMS download/acquire on main."
        ),
        evidence=(
            "run_pipeline_update.py",
            "cms_data_paths.nurse_raw_dir",
            "pbj_identifiers/urls.py",
            "docs/data_source_map.md",
        ),
    ),
    CmsSourceRecord(
        source_id="cms.pbj_non_nurse_staffing",
        human_name="PBJ non-nurse staffing",
        source_family=SourceFamily.PBJ_NON_NURSE,
        formats=(SourceFormat.CSV, SourceFormat.ZIP),
        cadence=SourceCadence.QUARTERLY,
        release_version_strategy=(
            "Canonical local CSV PBJ_dailynonnursestaffing_CY*; "
            "manage CLI expects ZIP ingest then standardize"
        ),
        cms_dataset_id=None,
        cms_metadata_endpoint=None,
        cms_landing_url=(
            "https://data.cms.gov/quality-of-care/"
            "payroll-based-journal-daily-non-nurse-staffing"
        ),
        local_raw_location="NonNursecsv/",
        normalized_derived_location="standardized_NonNurse/",
        validator="standardize_nonnursepbj_files.py; packaging gates",
        downstream_consumers=(
            "standardize_nonnursepbj_files.py",
            "generate_non_nurse_*.py",
            "run_pipeline_update.py",
        ),
        automation_level=AutomationLevel.BROKEN_LEGACY,
        automation_notes=(
            "Detection + normalize exist; manage_cms_sources nonnurse ingest "
            "calls ingest_cms_nonnurse_quarter.py which is ABSENT on main."
        ),
        evidence=(
            "scripts/manage_cms_sources.py",
            "run_pipeline_update.py",
            "cms_data_paths.nonnurse_raw_dir",
        ),
    ),
    CmsSourceRecord(
        source_id="cms.pbj_employee_ein_detail",
        human_name="PBJ employee / EIN detail",
        source_family=SourceFamily.PBJ_EIN,
        formats=(SourceFormat.ZIP, SourceFormat.ZIP_MEMBER, SourceFormat.CSV),
        cadence=SourceCadence.QUARTERLY,
        release_version_strategy=(
            "Monolithic PUF zip + quarter zips under EIN/quarters/; "
            "members PBJ_employeedetail_CY* / CYyyyyQn"
        ),
        cms_dataset_id=None,
        cms_metadata_endpoint=None,
        cms_landing_url=(
            "https://data.cms.gov/quality-of-care/"
            "payroll-based-journal-employee-detail-nursing-home-staffing"
        ),
        local_raw_location="EIN/monolithic/; EIN/quarters/",
        normalized_derived_location="EIN/extracted/; facility EIN slices",
        validator=(
            "Intended validate_ein_supplemental_zips.py (ABSENT on main); "
            "gate_ein_slice readiness"
        ),
        downstream_consumers=(
            "facility_ein_lib.py",
            "facility_ein_employee_analytics.py",
            "scripts/manage_cms_sources.py",
        ),
        automation_level=AutomationLevel.BROKEN_LEGACY,
        automation_notes=(
            "Path/layout + status detection exist; install/organize/validate "
            "scripts referenced by CLI are missing on main."
        ),
        evidence=(
            "cms_data_paths.py",
            "facility_ein_lib.py",
            "scripts/manage_cms_sources.py",
            "docs/data_source_map.md",
        ),
    ),
    CmsSourceRecord(
        source_id="cms.snf_all_owners",
        human_name="SNF All Owners",
        source_family=SourceFamily.SNF_ALL_OWNERS,
        formats=(SourceFormat.CSV,),
        cadence=SourceCadence.UNKNOWN,
        release_version_strategy="Glob ownership/SNF_All_Owners*.csv (when present)",
        cms_dataset_id=None,
        cms_metadata_endpoint=None,
        cms_landing_url=(
            "https://data.cms.gov/provider-characteristics/"
            "hospitals-and-other-facilities/skilled-nursing-facility-all-owners"
        ),
        local_raw_location="ownership/SNF_All_Owners*.csv",
        normalized_derived_location=(
            "Intended indexes via build_snf_owners_index.py (ABSENT on main)"
        ),
        validator="Intended validate_ownership_linkage.py (ABSENT on main)",
        downstream_consumers=(
            "dynamic_facility_dashboard._latest_snf_all_owners_csv_path",
            "ownership UI / owners deep links",
        ),
        automation_level=AutomationLevel.FULLY_MANUAL,
        automation_notes="Consumer glob only; no acquire/normalize scripts on main.",
        evidence=(
            "docs/CMS_PROVIDER_RELEASE_WORKFLOW.md",
            "scripts/cms_provider_release_lib.py",
            "templates/superdynamic_dashboard_v2.html",
        ),
        notes="Distinct from monthly NH_Ownership_* extracted from Provider Info zip.",
    ),
    CmsSourceRecord(
        source_id="cms.snf_enrollments",
        human_name="SNF Enrollments",
        source_family=SourceFamily.SNF_ENROLLMENTS,
        formats=(SourceFormat.OTHER,),
        cadence=SourceCadence.UNKNOWN,
        release_version_strategy=(
            "No standalone drop verified on main; enrollment IDs appear as "
            "fields on All Owners / CHOW consumers"
        ),
        cms_dataset_id=None,
        cms_metadata_endpoint=None,
        cms_landing_url=None,
        local_raw_location="(none verified on main)",
        normalized_derived_location="(none verified on main)",
        validator=None,
        downstream_consumers=(
            "UI enrollment-id filters against SNF All Owners explorer",
        ),
        automation_level=AutomationLevel.FULLY_MANUAL,
        automation_notes=(
            "No separate source family paths/scripts on main; registry placeholder "
            "for inventory completeness."
        ),
        evidence=("docs/CMS_PROVIDER_RELEASE_WORKFLOW.md (all-owners enrollment wording)",),
        notes="Do not invent a CMS dataset ID until verified.",
    ),
    CmsSourceRecord(
        source_id="cms.snf_chow",
        human_name="SNF CHOW",
        source_family=SourceFamily.SNF_CHOW,
        formats=(SourceFormat.OTHER, SourceFormat.CSV),
        cadence=SourceCadence.UNKNOWN,
        release_version_strategy=(
            "Documented raw under ownership/_sources/cms_chow/ (ABSENT on main); "
            "runtime chow_index.json in facility bundles"
        ),
        cms_dataset_id=None,
        cms_metadata_endpoint=None,
        cms_landing_url=None,
        local_raw_location="ownership/_sources/cms_chow/ (documented; missing on main)",
        normalized_derived_location="deployments/.../ownership/chow_index.json",
        validator=None,
        downstream_consumers=(
            "V2 CHOW modal / facility events",
            "chow_index.json facility API",
        ),
        automation_level=AutomationLevel.FULLY_MANUAL,
        automation_notes=(
            "Manual / cross-repo; documented source tree and some modules absent on main."
        ),
        evidence=(
            "docs/CMS_PROVIDER_RELEASE_WORKFLOW.md",
            "templates/superdynamic_dashboard_v2.html",
        ),
    ),
    CmsSourceRecord(
        source_id="cms.chain_performance",
        human_name="Chain Performance",
        source_family=SourceFamily.CHAIN_PERFORMANCE,
        formats=(SourceFormat.CSV,),
        cadence=SourceCadence.IRREGULAR,
        release_version_strategy=(
            "Nursing_Home_Chain_Performance_Measures_{Mon}_{YYYY}.csv "
            "(legacy Affiliated_Entity_* names also scanned)"
        ),
        cms_dataset_id=None,
        cms_metadata_endpoint=None,
        cms_landing_url=(
            "https://data.cms.gov/quality-of-care/nursing-home-chain-performance-measures"
        ),
        local_raw_location="ownership/Nursing_Home_Chain_Performance_Measures_*.csv",
        normalized_derived_location="facility entity_lookup / longitudinal slices",
        validator="Detection only (utils/file_finder.find_latest_affiliated_entity)",
        downstream_consumers=(
            "utils/file_finder.py",
            "PBJ_Dashboard.py",
            "scripts/Ownership.py",
        ),
        automation_level=AutomationLevel.DETECTION_EXISTS,
        automation_notes="Manual CSV drop; detection of latest file exists; no download script.",
        evidence=(
            "utils/file_finder.py",
            "ownership/*.csv",
            "docs/data_source_map.md",
            "generate_report.py (landing URL)",
        ),
        notes=(
            "Cadence marked irregular: repo has dated snapshots (e.g. Jul/Nov 2025) "
            "without a verified fixed schedule on main."
        ),
    ),
    CmsSourceRecord(
        source_id="cms.sff",
        human_name="SFF (Special Focus Facility)",
        source_family=SourceFamily.SFF,
        formats=(SourceFormat.CSV, SourceFormat.PDF),
        cadence=SourceCadence.FOLLOWS_PROVIDER_INFO,
        release_version_strategy=(
            "Primary PBJapp path: Special Focus Status column on Provider Info CSV "
            "(mapped sff_status). Schema also allows PDF because CMS historically "
            "publishes SFF program lists as PDF; no separate SFF acquire path or "
            "dataset ID is verified on main."
        ),
        cms_dataset_id=None,
        cms_metadata_endpoint=None,
        cms_landing_url=f"https://data.cms.gov/provider-data/dataset/{_PI_DATASET}",
        local_raw_location="via provider_info/NH_ProviderInfo_*.csv (column)",
        normalized_derived_location="ProviderInfoNorm_* sff_status; facility risk UI",
        validator=None,
        downstream_consumers=(
            "PBJ_Dashboard.py",
            "dynamic_facility_dashboard.py",
            "nh_provider_column_map.py",
        ),
        automation_level=AutomationLevel.NORMALIZATION_EXISTS,
        automation_notes=(
            "Inherited from Provider Info pipeline; not an independent acquire family."
        ),
        evidence=(
            "nh_provider_column_map.py",
            "scripts/cms_provider_info_acquire.py",
            "PBJ_Dashboard.py",
        ),
        notes=(
            "Do not confuse UI typo dataset r5ix-sffd (citations) with SFF. "
            "PDF container supported in schema; PDF list not ingested on main."
        ),
    ),
)


def get_registry() -> tuple[CmsSourceRecord, ...]:
    return CMS_SOURCE_REGISTRY


def get_source(source_id: str) -> Optional[CmsSourceRecord]:
    for row in CMS_SOURCE_REGISTRY:
        if row.source_id == source_id:
            return row
    return None


def registry_as_dicts() -> list[dict[str, Any]]:
    return [r.to_dict() for r in CMS_SOURCE_REGISTRY]
