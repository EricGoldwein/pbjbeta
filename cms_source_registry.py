"""Canonical CMS source registry for PBJ Data Ops (control plane).

PBJapp is the data factory. This module is static inventory + architecture
hooks only — no ETL. Runtime status lives in cms_data_ops / data_ops_*.

Stable CMS dataset IDs below were verified via the sibling pbj-root CMS
release watcher and encoded here so PBJapp does not need a runtime
dependency on pbj-root. Do not invent additional IDs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class Publisher(str, Enum):
    CMS = "CMS"


class SourceContainer(str, Enum):
    CSV = "CSV"
    ZIP = "ZIP"
    NESTED_ZIP_MEMBER = "nested_ZIP_member"
    PDF = "PDF"
    API = "API"
    OTHER = "other"


# Backward-compatible aliases used by earlier #64 tests/callers.
class SourceFormat(str, Enum):
    CSV = "CSV"
    ZIP = "ZIP"
    ZIP_MEMBER = "ZIP-member"
    PDF = "PDF"
    OTHER = "other"
    MIXED = "mixed"
    API = "API"


class SourceCadence(str, Enum):
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    IRREGULAR = "irregular"
    UNKNOWN = "unknown"


class CatalogMechanism(str, Enum):
    PROVIDER_DATA_METASTORE = "provider_data_metastore"
    DATA_CMS_GOV_DATASET_UUID = "data_cms_gov_dataset_uuid"
    LANDING_PAGE_ONLY = "landing_page_only"
    NONE_VERIFIED = "none_verified"


class AutomationMaturity(str, Enum):
    AUTOMATED = "automated"
    PARTIALLY_AUTOMATED = "partially_automated"
    MANUAL = "manual"
    BROKEN_LEGACY = "broken_legacy"
    UNMODELED = "unmodeled"


# Legacy alias for callers/tests from initial #64.
class AutomationLevel(str, Enum):
    DETECTION_EXISTS = "detection_exists"
    ACQUISITION_EXISTS = "acquisition_exists"
    NORMALIZATION_EXISTS = "normalization_exists"
    VALIDATION_EXISTS = "validation_exists"
    FULLY_MANUAL = "fully_manual"
    BROKEN_LEGACY = "broken_legacy"
    UNMODELED = "unmodeled"


class OpsStatus(str, Enum):
    CURRENT = "CURRENT"
    CMS_NEWER = "CMS_NEWER"
    LOCAL_RAW_ONLY = "LOCAL_RAW_ONLY"
    PROCESSING_REQUIRED = "PROCESSING_REQUIRED"
    READY_FOR_HANDOFF = "READY_FOR_HANDOFF"
    NOT_AVAILABLE_IN_THIS_RUNTIME = "NOT_AVAILABLE_IN_THIS_RUNTIME"
    UNKNOWN = "UNKNOWN"
    ERROR = "ERROR"


class ReleaseLifecycle(str, Enum):
    DETECTED = "DETECTED"
    ACQUIRED = "ACQUIRED"
    STRUCTURAL_PASS = "STRUCTURAL_PASS"
    STRUCTURAL_FAIL = "STRUCTURAL_FAIL"
    PROCESSED = "PROCESSED"
    ZWELI_PASS = "ZWELI_PASS"
    ZWELI_REQUIRES_REVIEW = "ZWELI_REQUIRES_REVIEW"
    ZWELI_BLOCKED = "ZWELI_BLOCKED"
    READY = "READY"
    APPROVED = "APPROVED"


class AccessMode(str, Enum):
    LOCAL_FILESYSTEM = "local_filesystem"
    CMS_HTTP = "cms_http"
    REMOTE_STORE = "remote_store"  # reserved; not implemented in V0
    API = "api"  # reserved
    MCP = "mcp"  # reserved
    UNAVAILABLE = "unavailable"


class SourceFamily(str, Enum):
    PROVIDER_INFO = "provider_info"
    PBJ_NURSE = "pbj_nurse_staffing"
    PBJ_NON_NURSE = "pbj_non_nurse_staffing"
    PBJ_EIN = "pbj_employee_ein_detail"
    SNF_ALL_OWNERS = "snf_all_owners"
    SNF_ENROLLMENTS = "snf_enrollments"
    SNF_CHOW = "snf_chow"
    CHAIN_PERFORMANCE = "chain_performance"
    HEALTH_CITATIONS = "health_citations"
    SFF_PDF_LIST = "sff_pdf_list"


# ---------------------------------------------------------------------------
# Verified CMS dataset IDs
# Provenance: sibling EricGoldwein/pbj-root CMS release watcher inventory
# (encoded into PBJapp so Data Ops has no runtime dependency on pbj-root).
# Provider Information / Health Citations also appear in PBJapp code paths.
# ---------------------------------------------------------------------------
CMS_ID_PROVIDER_INFO = "4pq5-n9py"
CMS_ID_PBJ_NURSE = "7e0d53ba-8f02-4c66-98a5-14a1c997c50d"
CMS_ID_PBJ_NON_NURSE = "b497431a-5b57-42c0-9016-90105b51841e"
CMS_ID_PBJ_EMPLOYEE = "d65b8be0-946e-410b-ab06-01829628d5a1"
CMS_ID_SNF_ALL_OWNERS = "afe44b85-cc6d-40d7-b5df-00ae8910d1d2"
CMS_ID_SNF_ENROLLMENTS = "5f2c306f-3b1c-42cd-b037-187b2ce22126"
CMS_ID_SNF_CHOW = "f557a6ed-95b3-4a22-8433-4175db2dec1c"
CMS_ID_CHAIN_PERFORMANCE = "97ecfad1-d3f1-4d42-b774-d74661d830bc"
CMS_ID_HEALTH_CITATIONS = "r5ix-sfxw"

_ID_PROVENANCE = (
    "Verified via sibling pbj-root CMS release watcher; encoded in PBJapp "
    "registry (no runtime dependency on pbj-root)."
)

_PI_METASTORE = (
    "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/"
    f"{CMS_ID_PROVIDER_INFO}?show-reference-ids=true"
)
_CITATIONS_METASTORE = (
    "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/"
    f"{CMS_ID_HEALTH_CITATIONS}?show-reference-ids=true"
)


@dataclass(frozen=True)
class CmsSourceRecord:
    """Static registry row for a heterogeneous CMS source."""

    source_id: str
    human_name: str
    publisher: Publisher
    source_family: SourceFamily
    containers: tuple[SourceContainer, ...]
    cadence: SourceCadence
    release_identity_strategy: str
    fingerprint_strategy: str
    cms_dataset_id: Optional[str]
    cms_dataset_id_provenance: Optional[str]
    landing_url: Optional[str]
    catalog_mechanism: CatalogMechanism
    metadata_endpoint: Optional[str]
    raw_artifact_resolver: str
    normalized_artifact_resolver: str
    acquisition_implementation: Optional[str]
    structural_validator: Optional[str]
    zweli_quality_profile: Optional[str]
    downstream_consumers: tuple[str, ...]
    automation_maturity: AutomationMaturity
    automation_notes: str
    actions_enabled: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    notes: str = ""
    # Legacy #64 field aliases (populated in to_dict)
    formats: tuple[SourceFormat, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["publisher"] = self.publisher.value
        d["source_family"] = self.source_family.value
        d["containers"] = [c.value for c in self.containers]
        d["cadence"] = self.cadence.value
        d["catalog_mechanism"] = self.catalog_mechanism.value
        d["automation_maturity"] = self.automation_maturity.value
        d["formats"] = [f.value for f in (self.formats or _containers_to_formats(self.containers))]
        # Compat aliases used by initial UI/tests
        d["cms_metadata_endpoint"] = self.metadata_endpoint
        d["cms_landing_url"] = self.landing_url
        d["local_raw_location"] = self.raw_artifact_resolver
        d["normalized_derived_location"] = self.normalized_artifact_resolver
        d["validator"] = self.structural_validator
        d["automation_level"] = self.automation_maturity.value
        d["release_version_strategy"] = self.release_identity_strategy
        return d


def _containers_to_formats(containers: tuple[SourceContainer, ...]) -> tuple[SourceFormat, ...]:
    mapping = {
        SourceContainer.CSV: SourceFormat.CSV,
        SourceContainer.ZIP: SourceFormat.ZIP,
        SourceContainer.NESTED_ZIP_MEMBER: SourceFormat.ZIP_MEMBER,
        SourceContainer.PDF: SourceFormat.PDF,
        SourceContainer.API: SourceFormat.API,
        SourceContainer.OTHER: SourceFormat.OTHER,
    }
    return tuple(mapping[c] for c in containers)


@dataclass(frozen=True)
class DerivedSignalRecord:
    """Product signal that may depend on one or more CMS sources/artifacts."""

    signal_id: str
    human_name: str
    upstream_source_ids: tuple[str, ...]
    derivation: str
    consumers: tuple[str, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rec(**kwargs: Any) -> CmsSourceRecord:
    containers = kwargs["containers"]
    if "formats" not in kwargs or not kwargs.get("formats"):
        kwargs["formats"] = _containers_to_formats(containers)
    return CmsSourceRecord(**kwargs)


CMS_SOURCE_REGISTRY: tuple[CmsSourceRecord, ...] = (
    _rec(
        source_id="cms.provider_info",
        human_name="Provider Information",
        publisher=Publisher.CMS,
        source_family=SourceFamily.PROVIDER_INFO,
        containers=(SourceContainer.CSV, SourceContainer.ZIP, SourceContainer.NESTED_ZIP_MEMBER),
        cadence=SourceCadence.MONTHLY,
        release_identity_strategy="NH_ProviderInfo_{Mon}{YYYY}.csv → release key YYYY-MM",
        fingerprint_strategy="SHA256 of raw CSV; acquisition.json raw_fingerprint",
        cms_dataset_id=CMS_ID_PROVIDER_INFO,
        cms_dataset_id_provenance=(
            f"{_ID_PROVENANCE} Also used by scripts/cms_provider_info_acquire.py (PR #63)."
        ),
        landing_url=f"https://data.cms.gov/provider-data/dataset/{CMS_ID_PROVIDER_INFO}",
        catalog_mechanism=CatalogMechanism.PROVIDER_DATA_METASTORE,
        metadata_endpoint=_PI_METASTORE,
        raw_artifact_resolver="cms_data_paths.provider_info_dir / NH_ProviderInfo_*.csv",
        normalized_artifact_resolver=(
            "cms_data_paths.provider_info_normalized_dir / ProviderInfoNorm_{YYYY}_{MM}.csv"
        ),
        acquisition_implementation="scripts/cms_provider_info_acquire.acquire_and_process",
        structural_validator="scripts/cms_provider_info_acquire.validate_raw_provider_info_csv",
        zweli_quality_profile="provider_info_v0",
        downstream_consumers=(
            "normalize_provider_info.py",
            "PBJ_Dashboard.py",
            "dynamic_facility_dashboard.py",
            "sff_status derived signal",
            "pbj-root handoff contract",
        ),
        automation_maturity=AutomationMaturity.AUTOMATED,
        automation_notes="PR #63 detect→acquire→validate→normalize→handoff (no pbj-root write).",
        actions_enabled=("check_cms", "acquire_process"),
        evidence=(
            "scripts/cms_provider_info_acquire.py",
            "docs/CMS_PROVIDER_RELEASE_WORKFLOW.md",
        ),
    ),
    _rec(
        source_id="cms.pbj_nurse_staffing",
        human_name="PBJ nurse staffing",
        publisher=Publisher.CMS,
        source_family=SourceFamily.PBJ_NURSE,
        containers=(SourceContainer.CSV, SourceContainer.ZIP),
        cadence=SourceCadence.QUARTERLY,
        release_identity_strategy="PBJ_dailynursestaffing_CY{YYYY}Q{n}",
        fingerprint_strategy="Filename quarter label; optional SHA256 when acquired",
        cms_dataset_id=CMS_ID_PBJ_NURSE,
        cms_dataset_id_provenance=_ID_PROVENANCE,
        landing_url=(
            "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing"
        ),
        catalog_mechanism=CatalogMechanism.DATA_CMS_GOV_DATASET_UUID,
        metadata_endpoint=None,
        raw_artifact_resolver="cms_data_paths.nurse_raw_dir (PBJcsv/)",
        normalized_artifact_resolver="cms_data_paths.standardized_nurse_dir (standardized_PBJ/)",
        acquisition_implementation=None,
        structural_validator="standardize_pbj_files.py (structure warnings)",
        zweli_quality_profile=None,
        downstream_consumers=("generate_metrics.py", "PBJ_Dashboard.py", "facility slices"),
        automation_maturity=AutomationMaturity.PARTIALLY_AUTOMATED,
        automation_notes="Detection + standardize exist; CMS download not automated on main.",
        evidence=("run_pipeline_update.py", "cms_data_paths.py", "pbj_identifiers/urls.py"),
    ),
    _rec(
        source_id="cms.pbj_non_nurse_staffing",
        human_name="PBJ non-nurse staffing",
        publisher=Publisher.CMS,
        source_family=SourceFamily.PBJ_NON_NURSE,
        containers=(SourceContainer.CSV, SourceContainer.ZIP),
        cadence=SourceCadence.QUARTERLY,
        release_identity_strategy="PBJ_dailynonnursestaffing_CY{YYYY}Q{n}",
        fingerprint_strategy="Filename quarter label",
        cms_dataset_id=CMS_ID_PBJ_NON_NURSE,
        cms_dataset_id_provenance=_ID_PROVENANCE,
        landing_url=(
            "https://data.cms.gov/quality-of-care/"
            "payroll-based-journal-daily-non-nurse-staffing"
        ),
        catalog_mechanism=CatalogMechanism.DATA_CMS_GOV_DATASET_UUID,
        metadata_endpoint=None,
        raw_artifact_resolver="cms_data_paths.nonnurse_raw_dir (NonNursecsv/)",
        normalized_artifact_resolver="cms_data_paths.standardized_nonnurse_dir",
        acquisition_implementation=(
            "manage_cms_sources nonnurse ingest → ingest_cms_nonnurse_quarter.py (ABSENT)"
        ),
        structural_validator="standardize_nonnursepbj_files.py",
        zweli_quality_profile=None,
        downstream_consumers=("generate_non_nurse_*.py", "facility nonnurse slices"),
        automation_maturity=AutomationMaturity.BROKEN_LEGACY,
        automation_notes="Normalize exists; acquire CLI references missing script — do not restore in V0.",
        evidence=("scripts/manage_cms_sources.py", "run_pipeline_update.py"),
    ),
    _rec(
        source_id="cms.pbj_employee_ein_detail",
        human_name="PBJ employee / EIN detail",
        publisher=Publisher.CMS,
        source_family=SourceFamily.PBJ_EIN,
        containers=(SourceContainer.ZIP, SourceContainer.NESTED_ZIP_MEMBER, SourceContainer.CSV),
        cadence=SourceCadence.QUARTERLY,
        release_identity_strategy="Monolithic PUF + EIN/quarters CYyyyyQn zips",
        fingerprint_strategy="Zip member names / quarter labels",
        cms_dataset_id=CMS_ID_PBJ_EMPLOYEE,
        cms_dataset_id_provenance=_ID_PROVENANCE,
        landing_url=(
            "https://data.cms.gov/quality-of-care/"
            "payroll-based-journal-employee-detail-nursing-home-staffing"
        ),
        catalog_mechanism=CatalogMechanism.DATA_CMS_GOV_DATASET_UUID,
        metadata_endpoint=None,
        raw_artifact_resolver="cms_data_paths.ein_monolithic_dir / ein_quarters_dir",
        normalized_artifact_resolver="cms_data_paths.ein_extracted_dir; facility EIN slices",
        acquisition_implementation="manage_cms_sources ein* (install scripts ABSENT on main)",
        structural_validator="validate_ein_supplemental_zips.py (ABSENT)",
        zweli_quality_profile=None,
        downstream_consumers=("facility_ein_lib.py", "facility dashboards"),
        automation_maturity=AutomationMaturity.BROKEN_LEGACY,
        automation_notes="Layout/status exist; install/validate scripts missing — leave broken-legacy.",
        evidence=("facility_ein_lib.py", "cms_data_paths.py"),
    ),
    _rec(
        source_id="cms.snf_all_owners",
        human_name="SNF All Owners",
        publisher=Publisher.CMS,
        source_family=SourceFamily.SNF_ALL_OWNERS,
        containers=(SourceContainer.CSV,),
        cadence=SourceCadence.MONTHLY,
        release_identity_strategy="SNF_All_Owners*.csv filename / CMS dataset release",
        fingerprint_strategy="SHA256 when present",
        cms_dataset_id=CMS_ID_SNF_ALL_OWNERS,
        cms_dataset_id_provenance=_ID_PROVENANCE,
        landing_url=(
            "https://data.cms.gov/provider-characteristics/"
            "hospitals-and-other-facilities/skilled-nursing-facility-all-owners"
        ),
        catalog_mechanism=CatalogMechanism.DATA_CMS_GOV_DATASET_UUID,
        metadata_endpoint=None,
        raw_artifact_resolver="ownership/SNF_All_Owners*.csv",
        normalized_artifact_resolver="build_snf_owners_index.py (ABSENT on main)",
        acquisition_implementation=None,
        structural_validator=None,
        zweli_quality_profile=None,
        downstream_consumers=("dynamic_facility_dashboard owners UI",),
        automation_maturity=AutomationMaturity.MANUAL,
        automation_notes="Distinct from SNF Enrollments and from NH_Ownership_* co-extract.",
        evidence=("docs/CMS_PROVIDER_RELEASE_WORKFLOW.md",),
        notes="Cadence monthly per CMS enrollment-style drops (watcher); local schedule not scripted.",
    ),
    _rec(
        source_id="cms.snf_enrollments",
        human_name="SNF Enrollments",
        publisher=Publisher.CMS,
        source_family=SourceFamily.SNF_ENROLLMENTS,
        containers=(SourceContainer.CSV,),
        cadence=SourceCadence.MONTHLY,
        release_identity_strategy="CMS SNF Enrollments dataset release (separate from All Owners)",
        fingerprint_strategy="SHA256 when present",
        cms_dataset_id=CMS_ID_SNF_ENROLLMENTS,
        cms_dataset_id_provenance=_ID_PROVENANCE,
        landing_url=None,
        catalog_mechanism=CatalogMechanism.DATA_CMS_GOV_DATASET_UUID,
        metadata_endpoint=None,
        raw_artifact_resolver="(no PBJapp path verified on main)",
        normalized_artifact_resolver="(none on main)",
        acquisition_implementation=None,
        structural_validator=None,
        zweli_quality_profile=None,
        downstream_consumers=("enrollment ID fields in ownership UI narratives",),
        automation_maturity=AutomationMaturity.UNMODELED,
        automation_notes=(
            "Separate source from SNF All Owners. No acquire/normalize path on main."
        ),
        evidence=("pbj-root watcher ID encoding",),
    ),
    _rec(
        source_id="cms.snf_chow",
        human_name="SNF CHOW",
        publisher=Publisher.CMS,
        source_family=SourceFamily.SNF_CHOW,
        containers=(SourceContainer.CSV, SourceContainer.ZIP, SourceContainer.OTHER),
        cadence=SourceCadence.UNKNOWN,
        release_identity_strategy="CMS CHOW public release identity (dataset UUID)",
        fingerprint_strategy="unknown on main",
        cms_dataset_id=CMS_ID_SNF_CHOW,
        cms_dataset_id_provenance=_ID_PROVENANCE,
        landing_url=None,
        catalog_mechanism=CatalogMechanism.DATA_CMS_GOV_DATASET_UUID,
        metadata_endpoint=None,
        raw_artifact_resolver="ownership/_sources/cms_chow/ (documented; often absent)",
        normalized_artifact_resolver="deployments/.../ownership/chow_index.json",
        acquisition_implementation=None,
        structural_validator=None,
        zweli_quality_profile=None,
        downstream_consumers=("V2 CHOW modal", "facility events"),
        automation_maturity=AutomationMaturity.MANUAL,
        automation_notes="Manual / cross-repo index; no automated acquire on main.",
        evidence=("docs/CMS_PROVIDER_RELEASE_WORKFLOW.md",),
    ),
    _rec(
        source_id="cms.chain_performance",
        human_name="Chain Performance",
        publisher=Publisher.CMS,
        source_family=SourceFamily.CHAIN_PERFORMANCE,
        containers=(SourceContainer.CSV,),
        cadence=SourceCadence.IRREGULAR,
        release_identity_strategy="Nursing_Home_Chain_Performance_Measures_{Mon}_{YYYY}.csv",
        fingerprint_strategy="Filename vintage; optional SHA256",
        cms_dataset_id=CMS_ID_CHAIN_PERFORMANCE,
        cms_dataset_id_provenance=_ID_PROVENANCE,
        landing_url=(
            "https://data.cms.gov/quality-of-care/nursing-home-chain-performance-measures"
        ),
        catalog_mechanism=CatalogMechanism.DATA_CMS_GOV_DATASET_UUID,
        metadata_endpoint=None,
        raw_artifact_resolver="ownership/Nursing_Home_Chain_Performance_Measures_*.csv",
        normalized_artifact_resolver="facility entity_lookup / longitudinal slices",
        acquisition_implementation=None,
        structural_validator=None,
        zweli_quality_profile=None,
        downstream_consumers=("utils/file_finder.py", "PBJ_Dashboard.py"),
        automation_maturity=AutomationMaturity.PARTIALLY_AUTOMATED,
        automation_notes="Latest-file detection exists; acquire is manual drop.",
        evidence=("utils/file_finder.py", "ownership/*.csv"),
    ),
    _rec(
        source_id="cms.health_citations",
        human_name="Health Citations",
        publisher=Publisher.CMS,
        source_family=SourceFamily.HEALTH_CITATIONS,
        containers=(SourceContainer.CSV,),
        cadence=SourceCadence.MONTHLY,
        release_identity_strategy=(
            "Standalone CMS dataset r5ix-sfxw (distinct from NH_HealthCitations_* "
            "co-extracted from Provider Info yearly zip)"
        ),
        fingerprint_strategy="SHA256 when acquired",
        cms_dataset_id=CMS_ID_HEALTH_CITATIONS,
        cms_dataset_id_provenance=(
            f"{_ID_PROVENANCE} Also CMS_NH_HEALTH_CITATIONS_DATASET_ID in "
            "pbj_identifiers/urls.py."
        ),
        landing_url=(
            f"https://data.cms.gov/provider-data/dataset/{CMS_ID_HEALTH_CITATIONS}"
        ),
        catalog_mechanism=CatalogMechanism.PROVIDER_DATA_METASTORE,
        metadata_endpoint=_CITATIONS_METASTORE,
        raw_artifact_resolver="cms_data_paths.citations_dir / standalone CMS citations CSV",
        normalized_artifact_resolver="citation_lib / facility citation slices",
        acquisition_implementation=None,
        structural_validator=None,
        zweli_quality_profile=None,
        downstream_consumers=("citation_lib.py", "facility dashboards", "pbj_identifiers.urls"),
        automation_maturity=AutomationMaturity.PARTIALLY_AUTOMATED,
        automation_notes=(
            "Consumer URL builders exist. Co-extracted NH_HealthCitations_* from PI zip "
            "are a different artifact path — do not conflate with this dataset."
        ),
        evidence=("pbj_identifiers/urls.py", "cms_data_paths.citations_dir"),
    ),
    _rec(
        source_id="cms.sff_pdf_list",
        human_name="SFF / Candidate PDF list",
        publisher=Publisher.CMS,
        source_family=SourceFamily.SFF_PDF_LIST,
        containers=(SourceContainer.PDF, SourceContainer.OTHER),
        cadence=SourceCadence.IRREGULAR,
        release_identity_strategy=(
            "CMS Special Focus Facility / Candidate publication (PDF/list + archives)"
        ),
        fingerprint_strategy="PDF/list checksum when acquired (not implemented V0)",
        cms_dataset_id=None,
        cms_dataset_id_provenance=(
            "No stable open-data dataset ID verified for the PDF/list publication "
            "in PBJapp; distinct from Provider Info sff_status column."
        ),
        landing_url=None,
        catalog_mechanism=CatalogMechanism.NONE_VERIFIED,
        metadata_endpoint=None,
        raw_artifact_resolver="(unmodeled — not acquired in PBJapp V0)",
        normalized_artifact_resolver="(none — parser MANUAL/UNMODELED)",
        acquisition_implementation=None,
        structural_validator=None,
        zweli_quality_profile=None,
        downstream_consumers=("future SFF history / cross-check vs Provider Info signal",),
        automation_maturity=AutomationMaturity.UNMODELED,
        automation_notes=(
            "Do not parse PDF in V0 merely for registry completeness. "
            "Product SFF UI today uses Provider Info sff_status (derived signal)."
        ),
        evidence=("registry architecture requirement",),
    ),
)


DERIVED_SIGNALS: tuple[DerivedSignalRecord, ...] = (
    DerivedSignalRecord(
        signal_id="signal.sff_status",
        human_name="Special Focus Status (Provider Info column)",
        upstream_source_ids=("cms.provider_info",),
        derivation=(
            "source cms.provider_info → release → NH_ProviderInfo / Norm artifact → "
            "column Special Focus Status (mapped sff_status) → consumers"
        ),
        consumers=(
            "PBJ_Dashboard.py",
            "dynamic_facility_dashboard.py",
            "nh_provider_column_map.py",
        ),
        notes=(
            "Signal A. Distinct from source cms.sff_pdf_list (Source B). "
            "A product concept may depend on both."
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


def get_derived_signals() -> tuple[DerivedSignalRecord, ...]:
    return DERIVED_SIGNALS


def registry_as_dicts() -> list[dict[str, Any]]:
    return [r.to_dict() for r in CMS_SOURCE_REGISTRY]


def verified_cms_dataset_ids() -> dict[str, str]:
    """source_id → cms_dataset_id for sources with verified IDs."""
    return {
        r.source_id: r.cms_dataset_id
        for r in CMS_SOURCE_REGISTRY
        if r.cms_dataset_id
    }
