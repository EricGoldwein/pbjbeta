"""Authoritative classification of sources feeding the ACTIVE release registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum

from release_control_plane import DEPENDENCY_GRAPH


class UpdateMechanism(str, Enum):
    EXTERNAL_RECURRING = "external recurring release"
    DERIVED = "derived from ACTIVE release"
    MANUAL_VERSIONED = "manually versioned reference"
    STATIC_CONFIGURATION = "static configuration"


@dataclass(frozen=True)
class ReleaseSource:
    dataset_id: str
    label: str
    mechanism: UpdateMechanism
    upstream: tuple[str, ...] = ()
    detector: str | None = None
    acquirer: str | None = None
    validator: str = "semantic/schema validation required"
    approval: str = "explicit promotion"

    def as_dict(self) -> dict:
        out = asdict(self)
        out["mechanism"] = self.mechanism.value
        out["downstream"] = list(DEPENDENCY_GRAPH.get(self.dataset_id, ()))
        return out


SOURCES = (
    ReleaseSource("cms.pbj_nurse_staffing", "PBJ nurse", UpdateMechanism.EXTERNAL_RECURRING, detector="CMS data-api resources", acquirer="cms_pbj_nurse_acquire", validator="quarter identity, schema, minimum rows/bytes; automatic"),
    ReleaseSource("cms.pbj_non_nurse_staffing", "PBJ non-nurse", UpdateMechanism.EXTERNAL_RECURRING, detector="CMS data-api resources", acquirer="generic CMS CSV adapter", validator="raw schema automatic; normalization and complete-series validation required before VALIDATED"),
    ReleaseSource("cms.provider_info", "Provider Info", UpdateMechanism.EXTERNAL_RECURRING, detector="CMS Provider Data metastore", acquirer="cms_provider_info_acquire", validator="schema, release manifest, Zweli; review when required"),
    ReleaseSource(
        "cms.health_citations",
        "Health Citations",
        UpdateMechanism.EXTERNAL_RECURRING,
        detector="CMS theme publication / r5ix-sfxw metastore",
        acquirer="health_citations_acquire",
        validator="citation schema; bundle provenance when co-extracted from Provider Info",
    ),
    ReleaseSource("cms.nh_ownership", "Nursing Home Ownership (Provider Data)", UpdateMechanism.DERIVED, ("cms.provider_info",), validator="Provider bundle member hash + ownership schema"),
    ReleaseSource("cms.snf_all_owners", "SNF All Owners (PECOS)", UpdateMechanism.EXTERNAL_RECURRING, detector="CMS data-api resources", acquirer="generic CMS CSV adapter", validator="release identity, enrollment-id schema; human approval"),
    ReleaseSource("cms.snf_enrollments", "SNF Enrollment", UpdateMechanism.EXTERNAL_RECURRING, detector="CMS data-api resources", acquirer="generic CMS CSV adapter", validator="release identity, enrollment-id/CCN schema; human approval"),
    ReleaseSource("cms.sff_pdf_list", "SFF Posting", UpdateMechanism.MANUAL_VERSIONED, detector="monitored CMS publication; no stable index verified", acquirer="restricted CMS PDF staging", validator="PDF hash, four status tables, CCN/category/state semantics; explicit promotion"),
    ReleaseSource("pbj.benchmarks.state", "State benchmarks", UpdateMechanism.DERIVED, ("cms.pbj_nurse_staffing",), validator="builder semantic checks + upstream provenance"),
    ReleaseSource("pbj.benchmarks.national", "National benchmarks", UpdateMechanism.DERIVED, ("cms.pbj_nurse_staffing",), validator="builder semantic checks + upstream provenance"),
    ReleaseSource("pbj.benchmarks.region", "Region benchmarks", UpdateMechanism.DERIVED, ("cms.pbj_nurse_staffing", "cms.provider_info"), validator="builder semantic checks + upstream provenance"),
    ReleaseSource("pbj.benchmarks.region_mapping", "Region mapping", UpdateMechanism.STATIC_CONFIGURATION, validator="mapping coverage and schema; explicit versioning"),
    ReleaseSource("pbj.benchmarks.geo_cmi", "Geographic CMI", UpdateMechanism.DERIVED, ("cms.provider_info",), validator="builder semantic checks + upstream provenance"),
    ReleaseSource("pbj.peer_distribution", "Peer distribution", UpdateMechanism.DERIVED, ("cms.pbj_nurse_staffing",), validator="builder semantic checks + upstream provenance"),
    ReleaseSource("macpac.state_staffing_standards", "MACPAC", UpdateMechanism.MANUAL_VERSIONED, validator="publication/version review, state coverage, numeric constraints; human approval"),
)

BY_ID = {item.dataset_id: item for item in SOURCES}
