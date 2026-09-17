from sff_history.observations import build_observations, validate_observations
from sff_history.schema import PARSER_VERSION


def _row(**overrides):
    base = {
        "ccn": "015009",
        "facility_name": "Example Facility",
        "address": "1 Main St",
        "city": "Anytown",
        "state": "AL",
        "zip": "35004",
        "phone": "205-555-0100",
        "category": "CURRENT_SFF",
        "source_table": "Table A",
        "source_page": "4",
        "status_date": "01/15/2026",
        "survey_criteria": "Not Met",
        "months_in_status": "3",
    }
    base.update(overrides)
    return base


def test_multi_table_membership_same_publication_produces_two_observations():
    """The audit found 8 CCNs cross-listed as both CURRENT_SFF and
    SFF_CANDIDATE within one August 2026 posting (SFF_CURRENT_RECONCILIATION.md
    S3) — a facility can hold simultaneous memberships in one publication, and
    that must produce two Layer-2 rows, never one row with two categories.
    """
    ccn = "045143"
    rows = [
        _row(ccn=ccn, category="CURRENT_SFF", source_table="Table A", months_in_status="1", status_date="", survey_criteria=""),
        _row(ccn=ccn, category="SFF_CANDIDATE", source_table="Table D", months_in_status="6", status_date="", survey_criteria=""),
    ]
    observations = build_observations("2026-08", rows, era_id="era3b_ccn_2023_03_plus", parser_version=PARSER_VERSION)
    assert len(observations) == 2
    categories = {o.normalized_category for o in observations}
    assert categories == {"CURRENT_SFF", "SFF_CANDIDATE"}
    assert all(o.ccn == ccn for o in observations)
    # Distinct observation IDs, both traceable back to the same publication+CCN.
    assert len({o.observation_id for o in observations}) == 2


def test_alphanumeric_ccn_is_never_modified():
    ccn = "15E064"
    rows = [_row(ccn=ccn, category="SFF_CANDIDATE", source_table="Table D", months_in_status="2", status_date="", survey_criteria="")]
    observations = build_observations("2026-07", rows, era_id="era3b_ccn_2023_03_plus", parser_version=PARSER_VERSION)
    assert observations[0].ccn == "15E064"
    assert "observation_id" in observations[0].to_row()
    assert observations[0].to_row()["ccn"] == "15E064"


def test_graduation_row_carries_explicit_dated_event_fields():
    rows = [
        _row(
            ccn="045421",
            category="GRADUATED",
            source_table="Table B",
            status_date="07/16/2026",
            survey_criteria="",
            months_in_status="18",
        )
    ]
    observations = build_observations("2026-08", rows, era_id="era3b_ccn_2023_03_plus", parser_version=PARSER_VERSION)
    obs = observations[0]
    assert obs.explicit_status_date == "07/16/2026"
    assert obs.explicit_status_date_kind == "graduation"


def test_candidate_row_has_no_explicit_status_date():
    rows = [_row(ccn="175334", category="SFF_CANDIDATE", source_table="Table D", status_date="", survey_criteria="", months_in_status="4")]
    observations = build_observations("2026-07", rows, era_id="era3b_ccn_2023_03_plus", parser_version=PARSER_VERSION)
    obs = observations[0]
    assert obs.explicit_status_date == ""
    assert obs.explicit_status_date_kind is None


def test_duplicate_ccn_within_category_is_a_warning_not_a_fatal_error():
    # Confirmed real: April 2023's own PDF text literally repeats CCN 535025
    # in Table A twice, back to back, with different months values.
    rows = [
        _row(ccn="535025", months_in_status="18"),
        _row(ccn="535025", months_in_status="17"),
    ]
    observations = build_observations("2023-04", rows, era_id="era3b_ccn_2023_03_plus", parser_version=PARSER_VERSION)
    result = validate_observations(observations)
    assert result["status"] == "PASS"
    assert any("duplicate CCN" in w for w in result["warnings"])


def test_implausible_months_value_is_a_warning_not_a_fatal_error():
    # Confirmed real: CCN 135116 in October 2024's Table A has months_in_status
    # "-2160" in CMS's own published PDF text, with blank inspection/criteria.
    rows = [_row(ccn="135116", months_in_status="-2160", status_date="", survey_criteria="")]
    observations = build_observations("2024-10", rows, era_id="era3b_ccn_2023_03_plus", parser_version=PARSER_VERSION)
    result = validate_observations(observations)
    assert result["status"] == "PASS"
    assert any("implausible months-in-status" in w for w in result["warnings"])


def test_invalid_ccn_shape_is_a_fatal_error():
    rows = [_row(ccn="BADCCN!")]
    observations = build_observations("2023-03", rows, era_id="era3b_ccn_2023_03_plus", parser_version=PARSER_VERSION)
    result = validate_observations(observations)
    assert result["status"] == "FAIL"
    assert any("invalid CCNs" in e for e in result["errors"])
