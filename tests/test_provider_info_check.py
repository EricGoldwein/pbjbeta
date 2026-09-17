import inspect

from sff_history.derive import derive_changes, derive_graduation_events, derive_intervals
from sff_history.observations import Observation
from sff_history.provider_info_check import reconcile

ERA = "era3b_ccn_2023_03_plus"
PARSER = "sff_history.pdf_parser:v1"


def _obs(ccn: str, category: str, *, table: str) -> Observation:
    return Observation(
        observation_id=f"2026-08:{table}:{ccn}:0",
        publication_id="2026-08",
        ccn=ccn,
        raw_facility_name="Example Facility",
        raw_table=table,
        normalized_category=category,
        source_page="4",
        address="1 Main St",
        city="Anytown",
        state="AL",
        zip="35004",
        phone="205-555-0100",
        most_recent_inspection="",
        met_survey_criteria="",
        months_in_status="1",
        months_field_label="Months as an SFF",
        explicit_status_date="",
        explicit_status_date_kind=None,
        era_id=ERA,
        parser_version=PARSER,
    )


def test_provider_info_lag_disagreement_is_flagged_not_agreed():
    # Reproduces the audit's confirmed August 2026 case: PDF says GRADUATED,
    # Provider Info's sff_status still says "SFF" (stale/lagging).
    observations = [_obs("045421", "GRADUATED", table="Table B")]
    provider_info = {"045421": "SFF"}
    rows = reconcile("2026-08", observations, provider_info)
    assert len(rows) == 1
    assert rows[0]["reconciliation_note"] == "DISAGREE_VALUE"
    assert rows[0]["provider_info_sff_status_raw"] == "SFF"


def test_provider_info_blank_is_distinct_from_disagree_value():
    # Reproduces CCN 175334: PDF says CURRENT_SFF, Provider Info is blank
    # (not a stale value) — audit S4 treats this as its own distinct case.
    observations = [_obs("175334", "CURRENT_SFF", table="Table A")]
    provider_info = {"175334": ""}
    rows = reconcile("2026-08", observations, provider_info)
    assert rows[0]["reconciliation_note"] == "PROVIDER_INFO_BLANK_PDF_NONBLANK"


def test_ccn_absent_from_provider_info_is_its_own_category():
    # Reproduces CCN 676355: present in the PDF, entirely absent from the
    # Provider Info extract — a roster-membership gap, not a value mismatch.
    observations = [_obs("676355", "SFF_CANDIDATE", table="Table D")]
    rows = reconcile("2026-08", observations, {})
    assert rows[0]["reconciliation_note"] == "CCN_ABSENT_FROM_PROVIDER_INFO"


def test_agreement_case():
    observations = [_obs("015009", "SFF_CANDIDATE", table="Table D")]
    rows = reconcile("2026-08", observations, {"015009": "SFF Candidate"})
    assert rows[0]["reconciliation_note"] == "AGREE"


def test_provider_info_disagreement_cannot_feed_derived_layer():
    """Structural guarantee: the Layer-3 derivation functions only ever take
    publications/observations (Layer 1-2). Provider Info reconciliation data
    has no parameter path into them, so a PDF-vs-Provider-Info disagreement
    can never be converted into a derived transition — it is architecturally
    impossible, not just avoided by convention.
    """
    for fn in (derive_changes, derive_graduation_events, derive_intervals):
        params = set(inspect.signature(fn).parameters)
        assert not params & {"provider_info", "provider_ccns", "provider_info_status", "reconciliation"}
