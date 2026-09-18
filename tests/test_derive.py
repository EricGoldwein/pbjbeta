from sff_history.derive import derive_changes, derive_graduation_events, derive_intervals, derive_survey_events
from sff_history.observations import Observation
from sff_history.publications import Publication

ERA = "era3b_ccn_2023_03_plus"
PARSER = "sff_history.pdf_parser:v1"


def _pub(publication_id: str, *, status: str = "PASS") -> Publication:
    return Publication(
        publication_id=publication_id,
        publication_period=publication_id,
        updated_date=None,
        updated_date_precision="unknown",
        updated_label_raw="",
        updated_date_period_mismatch=False,
        source_filename=f"{publication_id}.pdf",
        source_kind="archive",
        sha256="0" * 64,
        era_id=ERA,
        parser_version=PARSER,
        page_count=10,
        row_count=1,
        validation_status=status,
    )


def _obs(publication_id: str, ccn: str, category: str, *, table: str, seq: int = 0, months: str = "1", status_date: str = "", status_kind=None, inspection: str = "", criteria: str = "") -> Observation:
    return Observation(
        observation_id=f"{publication_id}:{table}:{ccn}:{seq}",
        publication_id=publication_id,
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
        most_recent_inspection=inspection,
        met_survey_criteria=criteria,
        months_in_status=months,
        months_field_label="Months as an SFF",
        explicit_status_date=status_date,
        explicit_status_date_kind=status_kind,
        era_id=ERA,
        parser_version=PARSER,
    )


def test_explicit_graduation_event_only_when_cms_publishes_a_date():
    # Dated case: CMS's own row carries "Date of Graduation".
    dated_pub = _pub("2026-08")
    dated_obs = {
        "2026-08": [
            _obs("2026-08", "045421", "GRADUATED", table="Table B", status_date="07/16/2026", status_kind="graduation")
        ]
    }
    events = derive_graduation_events(dated_obs)
    assert len(events) == 1
    assert events[0]["event_date"] == "2026-07-16"
    assert events[0]["ccn"] == "045421"

    # Snapshot-only case: a CCN's category set differs between two adjacent
    # publications, but neither row carries a CMS-published date (e.g. a
    # clean SFF_CANDIDATE -> CURRENT_SFF move). This must show up only as an
    # OBSERVED_CHANGE with no asserted_date, never as a dated event.
    prev_pub, next_pub = _pub("2026-07"), _pub("2026-08")
    obs_by_pub = {
        "2026-07": [_obs("2026-07", "175334", "SFF_CANDIDATE", table="Table D")],
        "2026-08": [_obs("2026-08", "175334", "CURRENT_SFF", table="Table A")],
    }
    assert derive_graduation_events(obs_by_pub) == []  # no dates published anywhere
    changes = derive_changes([prev_pub, next_pub], obs_by_pub)
    assert len(changes) == 1
    change = changes[0]
    assert change["change_kind"] == "CATEGORY_SET_CHANGED"
    assert change["asserted_date"] == ""  # never inferred


def test_full_graduation_scenario_july_to_august():
    """Reproduces the acceptance-criterion shape: raw July CURRENT_SFF
    observation, raw August GRADUATED observation (with CMS's own date),
    an OBSERVED_CHANGE linking them, and a separately-tracked dated event —
    never conflating "when we first saw it" with "the date CMS published".
    """
    july, august = _pub("2026-07"), _pub("2026-08")
    obs_by_pub = {
        "2026-07": [_obs("2026-07", "045421", "CURRENT_SFF", table="Table A", months="18")],
        "2026-08": [
            _obs("2026-08", "045421", "GRADUATED", table="Table B", status_date="07/16/2026", status_kind="graduation")
        ],
    }
    changes = derive_changes([july, august], obs_by_pub)
    assert len(changes) == 1
    assert changes[0]["from_categories"] == "CURRENT_SFF"
    assert changes[0]["to_categories"] == "GRADUATED"
    assert changes[0]["asserted_date"] == ""

    events = derive_graduation_events(obs_by_pub)
    assert len(events) == 1
    assert events[0]["event_date"] == "2026-07-16"
    assert events[0]["first_observed_publication_id"] == "2026-08"  # observed in the August posting
    # The event's date is independent of which snapshot first surfaced it.
    assert events[0]["event_date"] != "2026-08"


def test_missing_month_gap_prevents_change_derivation_and_interval_continuity():
    # 2023-12 is a known missing publication in this archive (SFF_ARCHIVE_AUDIT.md S4).
    nov = _pub("2023-11")
    jan = _pub("2024-01")
    obs_by_pub = {
        "2023-11": [_obs("2023-11", "015009", "CURRENT_SFF", table="Table A", months="5")],
        "2024-01": [_obs("2024-01", "015009", "CURRENT_SFF", table="Table A", months="7")],
    }
    # Same category held in both snapshots, but the gap must prevent this
    # from being chained into one continuous derived interval at all — it
    # must not even be evaluated as a candidate pair, gapless or not.
    changes = derive_changes([nov, jan], obs_by_pub)
    assert changes == []  # no adjacency across the gap, so nothing to compare
    intervals = derive_intervals([nov, jan], obs_by_pub)
    assert intervals == []  # never bridges the gap


def test_gapless_adjacent_pair_does_derive_interval_and_change_free_when_stable():
    nov, dec = _pub("2023-11"), _pub("2023-12")
    obs_by_pub = {
        "2023-11": [_obs("2023-11", "015009", "CURRENT_SFF", table="Table A", months="5")],
        "2023-12": [_obs("2023-12", "015009", "CURRENT_SFF", table="Table A", months="6")],
    }
    assert derive_changes([nov, dec], obs_by_pub) == []  # stable membership, no change
    intervals = derive_intervals([nov, dec], obs_by_pub)
    assert len(intervals) == 1
    interval = intervals[0]
    assert interval["start_publication_id"] == "2023-11"
    assert interval["end_publication_id"] == "2023-12"
    assert interval["months_counter_consistency"] == "consistent"


def test_repeated_graduation_observations_collapse_to_one_canonical_event():
    # Reproduces the confirmed real pattern: CMS's Table B/C re-lists a
    # graduated/terminated facility, with the identical stated date, across
    # many consecutive monthly postings. One raw row per publication must
    # collapse to one canonical event, not one timeline event per posting.
    ccn = "035166"
    obs_by_pub = {
        pub_id: [_obs(pub_id, ccn, "GRADUATED", table="Table B", status_date="11/30/2021", status_kind="graduation")]
        for pub_id in ("2023-03", "2023-04", "2023-05", "2023-06")
    }
    events = derive_graduation_events(obs_by_pub)
    assert len(events) == 1
    event = events[0]
    assert event["event_date"] == "2021-11-30"
    assert event["observation_count"] == 4
    assert event["first_observed_publication_id"] == "2023-03"
    assert event["last_observed_publication_id"] == "2023-06"
    assert event["supporting_publication_ids"] == "2023-03;2023-04;2023-05;2023-06"
    assert len(event["supporting_observation_ids"].split(";")) == 4


def test_different_ccns_or_dates_remain_distinct_canonical_events():
    obs_by_pub = {
        "2026-07": [_obs("2026-07", "045421", "GRADUATED", table="Table B", status_date="07/16/2026", status_kind="graduation")],
        "2026-08": [
            _obs("2026-08", "045421", "GRADUATED", table="Table B", status_date="07/16/2026", status_kind="graduation"),
            _obs("2026-08", "265258", "GRADUATED", table="Table B", status_date="07/21/2026", status_kind="graduation"),
        ],
    }
    events = derive_graduation_events(obs_by_pub)
    assert len(events) == 2  # distinct CCNs, not merged
    dates = {e["ccn"]: e["event_date"] for e in events}
    assert dates == {"045421": "2026-07-16", "265258": "2026-07-21"}


def test_genuine_date_correction_produces_two_canonical_events_not_silently_merged():
    # Confirmed real: CCN 675799's graduation date is stated as 10/12/2023 in
    # publications through 2024-07, then permanently as 11/15/2023 from
    # 2024-08 onward -- a probable CMS date correction. Canonical identity is
    # (ccn, event_kind, event_date) exactly as specified, so this correctly
    # yields two distinct events rather than one merged/guessed-at event.
    ccn = "675799"
    obs_by_pub = {
        "2024-01": [_obs("2024-01", ccn, "GRADUATED", table="Table B", status_date="10/12/2023", status_kind="graduation")],
        "2024-07": [_obs("2024-07", ccn, "GRADUATED", table="Table B", status_date="10/12/2023", status_kind="graduation")],
        "2024-08": [_obs("2024-08", ccn, "GRADUATED", table="Table B", status_date="11/15/2023", status_kind="graduation")],
        "2024-09": [_obs("2024-09", ccn, "GRADUATED", table="Table B", status_date="11/15/2023", status_kind="graduation")],
    }
    events = derive_graduation_events(obs_by_pub)
    ccn_events = [e for e in events if e["ccn"] == ccn]
    assert len(ccn_events) == 2
    dates = sorted(e["event_date"] for e in ccn_events)
    assert dates == ["2023-10-12", "2023-11-15"]


def test_fail_status_publication_excluded_from_derivation():
    good, bad = _pub("2026-07"), _pub("2026-08", status="FAIL")
    obs_by_pub = {
        "2026-07": [_obs("2026-07", "015009", "CURRENT_SFF", table="Table A")],
        "2026-08": [_obs("2026-08", "015009", "GRADUATED", table="Table B", status_date="07/16/2026", status_kind="graduation")],
    }
    # derive_changes/derive_intervals only look at PASS publications via gapless_runs.
    assert derive_changes([good, bad], obs_by_pub) == []
    assert derive_intervals([good, bad], obs_by_pub) == []
    # Graduation events are read straight off observations regardless of the
    # publication's own status — the row's own date is still what CMS said —
    # but observations for a FAIL publication are never built by the real
    # pipeline in the first place (see build.py), so this function is only
    # ever called with PASS-sourced observations in practice.


def test_survey_new_awaiting_first_survey_blank_fields():
    ccn = "015009"
    obs_by_pub = {
        pub_id: [_obs(pub_id, ccn, "CURRENT_SFF", table="Table A", inspection="", criteria="")]
        for pub_id in ("2024-04", "2024-05", "2024-06")
    }
    events = derive_survey_events(obs_by_pub)
    assert len(events) == 1
    event = events[0]
    assert event["survey_outcome"] == "NEW_AWAITING_FIRST_SURVEY"
    assert event["most_recent_inspection_raw"] == ""
    assert event["met_survey_criteria_raw"] == ""
    assert event["observation_count"] == 3  # restated 3 times, collapsed to one canonical fact


def test_survey_met_and_not_met_use_raw_cms_values():
    ccn = "015009"
    obs_by_pub = {
        "2024-04": [_obs("2024-04", ccn, "CURRENT_SFF", table="Table A", inspection="03/15/2024", criteria="Met")],
        "2024-07": [_obs("2024-07", ccn, "CURRENT_SFF", table="Table A", inspection="06/20/2024", criteria="Not Met")],
    }
    events = derive_survey_events(obs_by_pub)
    assert len(events) == 2
    by_outcome = {e["survey_outcome"]: e for e in events}
    assert by_outcome["MET_LATEST_SURVEY"]["most_recent_inspection_raw"] == "03/15/2024"
    assert by_outcome["MET_LATEST_SURVEY"]["met_survey_criteria_raw"] == "Met"
    assert by_outcome["NOT_MET_LATEST_SURVEY"]["most_recent_inspection_raw"] == "06/20/2024"
    assert by_outcome["NOT_MET_LATEST_SURVEY"]["met_survey_criteria_raw"] == "Not Met"


def test_survey_repeated_monthly_restatement_of_same_result_collapses_to_one_event():
    # A facility can remain CURRENT_SFF for many months with the same
    # "Most Recent Inspection" date until its next actual survey -- this
    # must not turn into one timeline event per restating publication.
    ccn = "015009"
    obs_by_pub = {
        pub_id: [_obs(pub_id, ccn, "CURRENT_SFF", table="Table A", inspection="03/15/2024", criteria="Met")]
        for pub_id in ("2024-04", "2024-05", "2024-06", "2024-07", "2024-08")
    }
    events = derive_survey_events(obs_by_pub)
    assert len(events) == 1
    event = events[0]
    assert event["observation_count"] == 5
    assert event["first_observed_publication_id"] == "2024-04"
    assert event["last_observed_publication_id"] == "2024-08"


def test_survey_distinct_inspection_dates_are_not_merged_into_one_survey():
    # A genuinely later, distinct survey (a different CMS-published
    # inspection date) must never be inferred as "the same survey restated"
    # -- each distinct date is its own canonical event.
    ccn = "015009"
    obs_by_pub = {
        "2024-04": [_obs("2024-04", ccn, "CURRENT_SFF", table="Table A", inspection="03/15/2024", criteria="Not Met")],
        "2024-05": [_obs("2024-05", ccn, "CURRENT_SFF", table="Table A", inspection="03/15/2024", criteria="Not Met")],
        "2024-09": [_obs("2024-09", ccn, "CURRENT_SFF", table="Table A", inspection="08/10/2024", criteria="Met")],
    }
    events = derive_survey_events(obs_by_pub)
    assert len(events) == 2
    dates = sorted(e["most_recent_inspection_raw"] for e in events)
    assert dates == ["03/15/2024", "08/10/2024"]


def test_survey_events_only_derived_from_current_sff_table_a_rows():
    ccn = "015009"
    obs_by_pub = {
        "2024-04": [_obs("2024-04", ccn, "SFF_CANDIDATE", table="Table D", months="2")],
    }
    assert derive_survey_events(obs_by_pub) == []


def test_survey_events_never_calculate_or_assert_two_of_two_progress():
    # Structural guarantee: derive_survey_events's output rows carry no
    # progress-count field at all -- there is nothing for a caller to
    # (mis)read as "1 of 2" / "2 of 2".
    ccn = "015009"
    obs_by_pub = {
        "2024-04": [_obs("2024-04", ccn, "CURRENT_SFF", table="Table A", inspection="03/15/2024", criteria="Met")],
        "2024-09": [_obs("2024-09", ccn, "CURRENT_SFF", table="Table A", inspection="08/10/2024", criteria="Met")],
    }
    events = derive_survey_events(obs_by_pub)
    assert len(events) == 2
    for event in events:
        assert "progress" not in {k.lower() for k in event}
        assert not any("of 2" in str(v) for v in event.values())
