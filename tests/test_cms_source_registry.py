"""Tests for canonical CMS source registry (Data Ops v0)."""

from __future__ import annotations

from cms_source_registry import (
    AutomationLevel,
    CMS_SOURCE_REGISTRY,
    OpsStatus,
    SourceFormat,
    get_source,
    registry_as_dicts,
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
    "cms.sff",
}


def test_registry_covers_required_inventory():
    ids = {r.source_id for r in CMS_SOURCE_REGISTRY}
    assert REQUIRED_SOURCE_IDS <= ids
    assert len(CMS_SOURCE_REGISTRY) == len(ids)  # unique


def test_only_provider_info_has_verified_dataset_id():
    with_id = [r for r in CMS_SOURCE_REGISTRY if r.cms_dataset_id]
    assert len(with_id) == 1
    assert with_id[0].source_id == "cms.provider_info"
    assert with_id[0].cms_dataset_id == "4pq5-n9py"
    assert with_id[0].cms_metadata_endpoint and "4pq5-n9py" in with_id[0].cms_metadata_endpoint


def test_sff_allows_pdf_and_csv():
    sff = get_source("cms.sff")
    assert sff is not None
    assert SourceFormat.PDF in sff.formats
    assert SourceFormat.CSV in sff.formats


def test_provider_info_actions_enabled_others_readonly():
    pi = get_source("cms.provider_info")
    assert pi is not None
    assert "check_cms" in pi.actions_enabled
    assert "acquire_process" in pi.actions_enabled
    for r in CMS_SOURCE_REGISTRY:
        if r.source_id != "cms.provider_info":
            assert r.actions_enabled == ()


def test_ops_status_enum_complete():
    expected = {
        "CURRENT",
        "CMS_NEWER",
        "LOCAL_RAW_ONLY",
        "PROCESSING_REQUIRED",
        "READY_FOR_HANDOFF",
        "UNKNOWN",
        "ERROR",
    }
    assert {s.value for s in OpsStatus} == expected


def test_nonnurse_and_ein_marked_broken_legacy():
    assert get_source("cms.pbj_non_nurse_staffing").automation_level == AutomationLevel.BROKEN_LEGACY
    assert get_source("cms.pbj_employee_ein_detail").automation_level == AutomationLevel.BROKEN_LEGACY


def test_registry_as_dicts_serializable():
    rows = registry_as_dicts()
    assert isinstance(rows[0]["formats"], list)
    assert isinstance(rows[0]["source_family"], str)
