"""Tests for canonical CMS source registry (Data Ops control plane)."""

from __future__ import annotations

from cms_source_registry import (
    CMS_ID_CHAIN_PERFORMANCE,
    CMS_ID_HEALTH_CITATIONS,
    CMS_ID_PBJ_EMPLOYEE,
    CMS_ID_PBJ_NON_NURSE,
    CMS_ID_PBJ_NURSE,
    CMS_ID_PROVIDER_INFO,
    CMS_ID_SNF_ALL_OWNERS,
    CMS_ID_SNF_CHOW,
    CMS_ID_SNF_ENROLLMENTS,
    CMS_SOURCE_REGISTRY,
    AutomationMaturity,
    SourceCadence,
    SourceContainer,
    SourceFamily,
    get_derived_signals,
    get_source,
    registry_as_dicts,
    verified_cms_dataset_ids,
)


REQUIRED_SOURCE_IDS = {
    "cms.provider_info",
    "cms.pbj_nurse_staffing",
    "cms.pbj_non_nurse_staffing",
    "cms.pbj_employee_ein_detail",
    "cms.snf_all_owners",
    "cms.snf_enrollments",
    "cms.snf_chow",
    "cms.chain_performance",
    "cms.health_citations",
    "cms.sff_pdf_list",
}


def test_registry_covers_required_inventory():
    ids = {r.source_id for r in CMS_SOURCE_REGISTRY}
    assert REQUIRED_SOURCE_IDS <= ids
    assert len(CMS_SOURCE_REGISTRY) == len(ids)


def test_all_verified_stable_cms_ids():
    ids = verified_cms_dataset_ids()
    assert ids["cms.provider_info"] == CMS_ID_PROVIDER_INFO == "4pq5-n9py"
    assert ids["cms.pbj_nurse_staffing"] == CMS_ID_PBJ_NURSE
    assert ids["cms.pbj_non_nurse_staffing"] == CMS_ID_PBJ_NON_NURSE
    assert ids["cms.pbj_employee_ein_detail"] == CMS_ID_PBJ_EMPLOYEE
    assert ids["cms.snf_all_owners"] == CMS_ID_SNF_ALL_OWNERS
    assert ids["cms.snf_enrollments"] == CMS_ID_SNF_ENROLLMENTS
    assert ids["cms.snf_chow"] == CMS_ID_SNF_CHOW
    assert ids["cms.chain_performance"] == CMS_ID_CHAIN_PERFORMANCE
    assert ids["cms.health_citations"] == CMS_ID_HEALTH_CITATIONS
    assert "cms.sff_pdf_list" not in ids  # no verified dataset ID


def test_snf_enrollments_distinct_from_all_owners():
    owners = get_source("cms.snf_all_owners")
    enroll = get_source("cms.snf_enrollments")
    assert owners and enroll
    assert owners.cms_dataset_id != enroll.cms_dataset_id
    assert owners.source_family != enroll.source_family


def test_sff_pdf_distinct_from_provider_info_signal():
    sff = get_source("cms.sff_pdf_list")
    assert sff is not None
    assert SourceContainer.PDF in sff.containers
    assert sff.automation_maturity == AutomationMaturity.UNMODELED
    signals = get_derived_signals()
    assert any(s.signal_id == "signal.sff_status" for s in signals)
    sig = next(s for s in signals if s.signal_id == "signal.sff_status")
    assert "cms.provider_info" in sig.upstream_source_ids
    assert "cms.sff_pdf_list" not in sig.upstream_source_ids


def test_health_citations_distinct():
    hc = get_source("cms.health_citations")
    assert hc is not None
    assert hc.cms_dataset_id == "r5ix-sfxw"
    assert hc.source_family == SourceFamily.HEALTH_CITATIONS


def test_cadences_represented():
    cadences = {r.cadence for r in CMS_SOURCE_REGISTRY}
    assert SourceCadence.MONTHLY in cadences
    assert SourceCadence.QUARTERLY in cadences
    assert SourceCadence.IRREGULAR in cadences
    assert SourceCadence.UNKNOWN in cadences


def test_containers_represented():
    containers = {c for r in CMS_SOURCE_REGISTRY for c in r.containers}
    assert SourceContainer.CSV in containers
    assert SourceContainer.ZIP in containers
    assert SourceContainer.NESTED_ZIP_MEMBER in containers
    assert SourceContainer.PDF in containers


def test_provider_info_actions_others_readonly():
    pi = get_source("cms.provider_info")
    assert pi is not None
    assert "check_cms" in pi.actions_enabled
    for r in CMS_SOURCE_REGISTRY:
        if r.source_id != "cms.provider_info":
            assert r.actions_enabled == ()


def test_registry_as_dicts_serializable():
    rows = registry_as_dicts()
    assert isinstance(rows[0]["formats"], list)
    assert rows[0]["cms_dataset_id_provenance"]
