"""Source-specific Stage/Publish adapter registry for currently-public PBJ320 families."""

from __future__ import annotations

from pbj320_publication_contract import StagePublishSpec

PI_STAGE_PUBLISH_SPEC = StagePublishSpec(
    source_id="cms.provider_info",
    required_destination_roles=(
        "provider_norm",
        "provider_combined_latest",
        "state_page_aggregates",
        "search_index",
    ),
    shared_derived_roles=frozenset({"state_page_aggregates", "search_index"}),
)

OWNERSHIP_PAIR_STAGE_PUBLISH_SPEC = StagePublishSpec(
    source_id="cms.snf_ownership_pair",
    required_destination_roles=(
        "ownership_release_policy",
        "ownership_bridge_lookup",
        "enrollment_release_artifact",
    ),
    shared_derived_roles=frozenset({"search_index", "owner_profile_index"}),
    require_artifact_cache=True,
)

PBJ_NURSE_STAGE_PUBLISH_SPEC = StagePublishSpec(
    source_id="cms.pbj_nurse_staffing",
    required_destination_roles=(
        "facility_quarterly_metrics",
        "national_quarterly_metrics",
        "state_quarterly_metrics",
        "latest_quarter_data",
    ),
    shared_derived_roles=frozenset({"provider_indexes", "state_page_aggregates"}),
    require_artifact_cache=True,
)

SFF_STAGE_PUBLISH_SPEC = StagePublishSpec(
    source_id="cms.sff_pdf_list",
    required_destination_roles=(
        "sff_facilities_json",
        "sff_public_json",
    ),
    shared_derived_roles=frozenset({"search_index"}),
    require_artifact_cache=True,
)

MACPAC_STAGE_PUBLISH_SPEC = StagePublishSpec(
    source_id="cms.macpac_state_staffing",
    required_destination_roles=("macpac_state_staffing_json",),
    shared_derived_roles=frozenset(),
    require_artifact_cache=False,
)

PREMIUM_ONLY_SOURCES = frozenset(
    {
        "cms.health_citations",
        "cms.pbj_non_nurse_staffing",
    }
)

STAGE_PUBLISH_SPECS: dict[str, StagePublishSpec] = {
    "cms.provider_info": PI_STAGE_PUBLISH_SPEC,
    "cms.snf_ownership_pair": OWNERSHIP_PAIR_STAGE_PUBLISH_SPEC,
    "cms.pbj_nurse_staffing": PBJ_NURSE_STAGE_PUBLISH_SPEC,
    "cms.sff_pdf_list": SFF_STAGE_PUBLISH_SPEC,
    "cms.macpac_state_staffing": MACPAC_STAGE_PUBLISH_SPEC,
}


def stage_publish_spec(source_id: str) -> StagePublishSpec | None:
    return STAGE_PUBLISH_SPECS.get(source_id)


def is_premium_only_source(source_id: str) -> bool:
    return source_id in PREMIUM_ONLY_SOURCES
