from sff_history.observations import Observation
from sff_history.timeline import (
    PROMINENCE_CONTEXT,
    PROMINENCE_CONTINUITY_WARNING,
    PROMINENCE_PRIMARY_EVENT,
    build_facility_timeline,
    build_timeline_events,
    classify_change,
)

ERA = "era3b_ccn_2023_03_plus"
PARSER = "sff_history.pdf_parser:v1"


def _obs(publication_id, ccn, category, *, table, months="1", inspection="", criteria=""):
    return Observation(
        observation_id=f"{publication_id}:{table}:{ccn}:0",
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
        explicit_status_date="",
        explicit_status_date_kind=None,
        era_id=ERA,
        parser_version=PARSER,
    )


def _change(ccn, from_cats, to_cats, *, from_pub, to_pub, kind="CATEGORY_SET_CHANGED"):
    return {
        "change_id": f"change:{from_pub}:{to_pub}:{ccn}",
        "ccn": ccn,
        "from_publication_id": from_pub,
        "to_publication_id": to_pub,
        "change_kind": kind,
        "from_categories": ";".join(sorted(from_cats)) if from_cats else "",
        "to_categories": ";".join(sorted(to_cats)) if to_cats else "",
        "gapless": "Y",
        "derived_from_observation_ids": f"{from_pub}:x:{ccn}:0;{to_pub}:y:{ccn}:0",
        "asserted_date": "",
        "method": "adjacent_snapshot_comparison,same_parser,no_gap",
    }


def _build_timeline(ccn, *, observations_by_pub, all_publication_ids, changes=None, graduation_events=None, survey_events=None, gap_months=None):
    return build_facility_timeline(
        ccn,
        observations_by_pub=observations_by_pub,
        changes=changes or [],
        graduation_events=graduation_events or [],
        survey_events=survey_events or [],
        gap_months=gap_months or set(),
        all_publication_ids=all_publication_ids,
    )


class TestClassifyChange:
    def test_candidate_only_churn_is_routine(self):
        row = _change("015009", set(), {"SFF_CANDIDATE"}, from_pub="2023-03", to_pub="2023-04", kind="NEWLY_OBSERVED")
        assert classify_change(row) is None

    def test_candidate_dropping_off_is_routine(self):
        row = _change("015009", {"SFF_CANDIDATE"}, set(), from_pub="2023-03", to_pub="2023-04", kind="NO_LONGER_OBSERVED")
        assert classify_change(row) is None

    def test_promotion_to_current_sff_is_meaningful(self):
        row = _change("045143", {"SFF_CANDIDATE"}, {"SFF_CANDIDATE", "CURRENT_SFF"}, from_pub="2026-07", to_pub="2026-08")
        assert classify_change(row) == "PROMOTED_TO_CURRENT_SFF"

    def test_graduation_is_meaningful(self):
        row = _change("045421", {"CURRENT_SFF"}, {"GRADUATED"}, from_pub="2026-07", to_pub="2026-08")
        assert classify_change(row) == "GRADUATED_FROM_CURRENT_SFF"

    def test_no_longer_participating_is_meaningful(self):
        row = _change("015009", {"CURRENT_SFF"}, {"NO_LONGER_PARTICIPATING"}, from_pub="2023-03", to_pub="2023-04")
        assert classify_change(row) == "NO_LONGER_PARTICIPATING"

    def test_current_sff_disappearing_with_no_stated_outcome_is_flagged(self):
        row = _change("015009", {"CURRENT_SFF"}, set(), from_pub="2023-03", to_pub="2023-04", kind="NO_LONGER_OBSERVED")
        assert classify_change(row) == "LEFT_CURRENT_SFF_NO_STATED_OUTCOME"


class TestFacilityTimeline:
    def test_multi_transition_facility(self):
        """A facility observed as Candidate, then promoted to Current SFF,
        then explicitly graduated with a CMS-published date, then later
        re-observed as a Candidate again -- with a real intervening PASS
        publication (2024-01, CCN confirmed absent) providing affirmative
        evidence the facility had actually left active status, so this
        correctly remains a proven REENTRY even though a coincidental
        publication gap (2023-12) also falls within the same hiatus.
        """
        ccn = "999001"
        pubs = ["2023-03", "2023-04", "2023-05", "2024-01", "2024-02"]
        obs_by_pub = {
            "2023-03": [_obs("2023-03", ccn, "SFF_CANDIDATE", table="Table D", months="1")],
            "2023-04": [_obs("2023-04", ccn, "CURRENT_SFF", table="Table A", months="1")],
            "2023-05": [_obs("2023-05", ccn, "GRADUATED", table="Table B", months="1")],
            # gap: 2023-12 missing (not a key in obs_by_pub at all)
            "2024-01": [],  # ccn not observed this month -- real, published, confirmed absent
            "2024-02": [_obs("2024-02", ccn, "SFF_CANDIDATE", table="Table D", months="1")],
        }
        graduation_events = [
            {
                "event_id": f"event:{ccn}:graduation:05-20-2023",
                "ccn": ccn,
                "event_kind": "GRADUATION",
                "event_date": "2023-05-20",
                "event_date_raw": "05/20/2023",
                "first_observed_publication_id": "2023-05",
                "last_observed_publication_id": "2023-05",
                "observation_count": 1,
                "supporting_publication_ids": "2023-05",
                "supporting_observation_ids": "2023-05:Table B:999001:0",
            }
        ]
        changes = [
            _change(ccn, {"SFF_CANDIDATE"}, {"CURRENT_SFF"}, from_pub="2023-03", to_pub="2023-04"),
            _change(ccn, {"CURRENT_SFF"}, {"GRADUATED"}, from_pub="2023-04", to_pub="2023-05"),
        ]
        gap_months = {"2023-12"}

        timeline = _build_timeline(
            ccn,
            observations_by_pub=obs_by_pub,
            all_publication_ids=pubs,
            changes=changes,
            graduation_events=graduation_events,
            gap_months=gap_months,
        )
        event_types = [e["event_type"] for e in timeline]

        assert "FIRST_OBSERVED_SFF_CANDIDATE" in event_types
        assert "PROMOTED_TO_CURRENT_SFF" in event_types
        assert "GRADUATED_FROM_CURRENT_SFF" in event_types
        assert "EXPLICIT_GRADUATION" in event_types
        assert "REENTRY_SFF_CANDIDATE" in event_types  # real evidence exists (2023-05, 2024-01)
        assert "PUBLICATION_GAP" in event_types
        assert "CONTINUITY_UNKNOWN_AFTER_GAP_SFF_CANDIDATE" not in event_types

        by_type = {e["event_type"]: e for e in timeline}
        assert by_type["FIRST_OBSERVED_SFF_CANDIDATE"]["prominence"] == PROMINENCE_PRIMARY_EVENT
        assert by_type["EXPLICIT_GRADUATION"]["prominence"] == PROMINENCE_PRIMARY_EVENT
        assert by_type["REENTRY_SFF_CANDIDATE"]["prominence"] == PROMINENCE_PRIMARY_EVENT
        assert by_type["PUBLICATION_GAP"]["prominence"] == PROMINENCE_CONTEXT

        grad_explicit = by_type["EXPLICIT_GRADUATION"]
        assert grad_explicit["event_date"] == "2023-05-20"
        assert grad_explicit["event_date_precision"] == "explicit_cms_date"

        reentry = by_type["REENTRY_SFF_CANDIDATE"]
        assert reentry["as_of_publication_id"] == "2024-02"
        assert "Confirmed by an intervening published snapshot" in reentry["summary"]

        gap_event = by_type["PUBLICATION_GAP"]
        assert gap_event["as_of_publication_id"] == "2023-12"

        first_idx = event_types.index("FIRST_OBSERVED_SFF_CANDIDATE")
        grad_idx = event_types.index("GRADUATED_FROM_CURRENT_SFF")
        reentry_idx = event_types.index("REENTRY_SFF_CANDIDATE")
        assert first_idx < grad_idx < reentry_idx

    def test_reentry_not_flagged_when_span_is_truly_continuous(self):
        ccn = "999002"
        pubs = ["2023-03", "2023-04", "2023-05"]
        obs_by_pub = {
            "2023-03": [_obs("2023-03", ccn, "CURRENT_SFF", table="Table A")],
            "2023-04": [_obs("2023-04", ccn, "CURRENT_SFF", table="Table A")],
            "2023-05": [_obs("2023-05", ccn, "CURRENT_SFF", table="Table A")],
        }
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs)
        assert not any(e["event_type"].startswith("REENTRY_") for e in timeline)
        assert not any(e["event_type"].startswith("CONTINUITY_UNKNOWN_AFTER_GAP_") for e in timeline)

    def test_reentry_requires_a_real_intervening_absence_not_just_ccn_filtered_adjacency(self):
        """A real, published intervening month (2023-04) where the CCN is
        confirmed absent from every table is affirmative evidence -- this is
        a proven REENTRY, not a continuity-unknown case.
        """
        ccn = "999003"
        pubs = ["2023-03", "2023-04", "2023-05"]
        obs_by_pub = {
            "2023-03": [_obs("2023-03", ccn, "CURRENT_SFF", table="Table A")],
            "2023-04": [],  # 2023-04 is a real, published month; ccn just isn't in it
            "2023-05": [_obs("2023-05", ccn, "CURRENT_SFF", table="Table A")],
        }
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs)
        reentries = [e for e in timeline if e["event_type"].startswith("REENTRY_")]
        assert len(reentries) == 1
        assert reentries[0]["event_type"] == "REENTRY_CURRENT_SFF"
        assert reentries[0]["prominence"] == PROMINENCE_PRIMARY_EVENT
        assert reentries[0]["as_of_publication_id"] == "2023-05"
        assert "Confirmed by an intervening published snapshot" in reentries[0]["summary"]
        assert not any(e["event_type"].startswith("CONTINUITY_UNKNOWN_AFTER_GAP_") for e in timeline)

    def test_gap_with_no_intervening_publication_is_continuity_unknown_not_reentry(self):
        """The exact scenario the user flagged: active before a missing
        publication, active again after it, with NO real publication in
        between at all. This must NOT be proven as a re-entry -- there is no
        affirmative evidence the facility ever left. Confirmed real: CCN
        105234 is CURRENT_SFF in both 2024-11 and 2025-01 with no
        publication for 2024-12 in this archive.
        """
        ccn = "105234"
        pubs = ["2024-11", "2025-01"]  # 2024-12 has no PASS publication at all
        obs_by_pub = {
            "2024-11": [_obs("2024-11", ccn, "CURRENT_SFF", table="Table A")],
            "2025-01": [_obs("2025-01", ccn, "CURRENT_SFF", table="Table A")],
        }
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs, gap_months={"2024-12"})

        assert not any(e["event_type"].startswith("REENTRY_") for e in timeline)
        continuity = [e for e in timeline if e["event_type"].startswith("CONTINUITY_UNKNOWN_AFTER_GAP_")]
        assert len(continuity) == 1
        event = continuity[0]
        assert event["event_type"] == "CONTINUITY_UNKNOWN_AFTER_GAP_CURRENT_SFF"
        assert event["prominence"] == PROMINENCE_CONTINUITY_WARNING
        assert event["as_of_publication_id"] == "2025-01"
        assert "has NOT been shown to be a re-entry" in event["summary"]

    def test_continuity_unknown_applies_to_candidate_status_too(self):
        ccn = "105235"
        pubs = ["2024-11", "2025-01"]
        obs_by_pub = {
            "2024-11": [_obs("2024-11", ccn, "SFF_CANDIDATE", table="Table D")],
            "2025-01": [_obs("2025-01", ccn, "SFF_CANDIDATE", table="Table D")],
        }
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs, gap_months={"2024-12"})
        continuity = [e for e in timeline if e["event_type"].startswith("CONTINUITY_UNKNOWN_AFTER_GAP_")]
        assert len(continuity) == 1
        assert continuity[0]["event_type"] == "CONTINUITY_UNKNOWN_AFTER_GAP_SFF_CANDIDATE"
        assert continuity[0]["prominence"] == PROMINENCE_CONTINUITY_WARNING

    def test_routine_candidate_churn_excluded_from_compact_timeline(self):
        ccn = "999004"
        pubs = ["2023-03", "2023-04", "2023-05"]
        obs_by_pub = {
            "2023-03": [_obs("2023-03", ccn, "SFF_CANDIDATE", table="Table D")],
            "2023-04": [],
            "2023-05": [_obs("2023-05", ccn, "SFF_CANDIDATE", table="Table D")],
        }
        changes = [
            _change(ccn, {"SFF_CANDIDATE"}, set(), from_pub="2023-03", to_pub="2023-04", kind="NO_LONGER_OBSERVED"),
            _change(ccn, set(), {"SFF_CANDIDATE"}, from_pub="2023-04", to_pub="2023-05", kind="NEWLY_OBSERVED"),
        ]
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs, changes=changes)
        assert not any(e["timeline_event_id"].startswith("timeline:change:") for e in timeline)
        assert any(e["event_type"] == "FIRST_OBSERVED_SFF_CANDIDATE" for e in timeline)

    def test_gap_marker_is_context_prominence(self):
        ccn = "999006"
        # 2023-04 has no PASS publication at all (not in all_publication_ids,
        # not a key in obs_by_pub) and is a known coverage gap within this
        # CCN's observed span [2023-03, 2023-05].
        pubs = ["2023-03", "2023-05"]
        obs_by_pub = {
            "2023-03": [_obs("2023-03", ccn, "CURRENT_SFF", table="Table A")],
            "2023-05": [_obs("2023-05", ccn, "CURRENT_SFF", table="Table A")],
        }
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs, gap_months={"2023-04"})
        gap_events = [e for e in timeline if e["event_type"] == "PUBLICATION_GAP"]
        assert gap_events and all(e["prominence"] == PROMINENCE_CONTEXT for e in gap_events)
        # the same gap also drives a CONTINUITY_WARNING for the facility's own span
        continuity_events = [e for e in timeline if e["prominence"] == PROMINENCE_CONTINUITY_WARNING]
        assert continuity_events and all(e["event_type"].startswith("CONTINUITY_UNKNOWN_AFTER_GAP_") for e in continuity_events)


class TestSurveyEvents:
    def _survey_event(self, ccn, outcome, *, inspection="", criteria="", first_pub="2024-04", last_pub="2024-04", count=1):
        return {
            "survey_event_id": f"survey:{ccn}:{outcome}:{(inspection or 'none').replace('/', '-')}",
            "ccn": ccn,
            "survey_outcome": outcome,
            "most_recent_inspection_raw": inspection,
            "met_survey_criteria_raw": criteria,
            "first_observed_publication_id": first_pub,
            "last_observed_publication_id": last_pub,
            "observation_count": count,
            "supporting_publication_ids": first_pub if first_pub == last_pub else f"{first_pub};{last_pub}",
            "supporting_observation_ids": f"{first_pub}:Table A:{ccn}:0",
        }

    def test_new_awaiting_first_survey(self):
        ccn = "888001"
        pubs = ["2024-04"]
        obs_by_pub = {"2024-04": [_obs("2024-04", ccn, "CURRENT_SFF", table="Table A")]}
        survey_events = [self._survey_event(ccn, "NEW_AWAITING_FIRST_SURVEY")]
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs, survey_events=survey_events)
        event = next(e for e in timeline if e["event_type"] == "NEW_AWAITING_FIRST_SURVEY")
        assert event["prominence"] == PROMINENCE_PRIMARY_EVENT
        assert event["event_date"] == ""
        assert event["event_date_precision"] == "publication_period"

    def test_met_latest_survey_has_explicit_date(self):
        ccn = "888002"
        pubs = ["2024-04"]
        obs_by_pub = {"2024-04": [_obs("2024-04", ccn, "CURRENT_SFF", table="Table A", inspection="03/15/2024", criteria="Met")]}
        survey_events = [self._survey_event(ccn, "MET_LATEST_SURVEY", inspection="03/15/2024", criteria="Met")]
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs, survey_events=survey_events)
        event = next(e for e in timeline if e["event_type"] == "MET_LATEST_SURVEY")
        assert event["event_date"] == "2024-03-15"
        assert event["event_date_precision"] == "explicit_cms_date"
        assert "2 of 2" not in event["summary"]
        assert "1 of 2" not in event["summary"]

    def test_not_met_latest_survey(self):
        ccn = "888003"
        pubs = ["2024-04"]
        obs_by_pub = {"2024-04": [_obs("2024-04", ccn, "CURRENT_SFF", table="Table A", inspection="03/15/2024", criteria="Not Met")]}
        survey_events = [self._survey_event(ccn, "NOT_MET_LATEST_SURVEY", inspection="03/15/2024", criteria="Not Met")]
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs, survey_events=survey_events)
        event = next(e for e in timeline if e["event_type"] == "NOT_MET_LATEST_SURVEY")
        assert event["event_date"] == "2024-03-15"
        assert event["prominence"] == PROMINENCE_PRIMARY_EVENT

    def test_distinct_survey_dates_are_two_separate_events(self):
        ccn = "888004"
        pubs = ["2024-04", "2024-05"]
        obs_by_pub = {
            "2024-04": [_obs("2024-04", ccn, "CURRENT_SFF", table="Table A", inspection="03/15/2024", criteria="Not Met")],
            "2024-05": [_obs("2024-05", ccn, "CURRENT_SFF", table="Table A", inspection="04/20/2024", criteria="Met")],
        }
        survey_events = [
            self._survey_event(ccn, "NOT_MET_LATEST_SURVEY", inspection="03/15/2024", criteria="Not Met", first_pub="2024-04", last_pub="2024-04"),
            self._survey_event(ccn, "MET_LATEST_SURVEY", inspection="04/20/2024", criteria="Met", first_pub="2024-05", last_pub="2024-05"),
        ]
        timeline = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs, survey_events=survey_events)
        survey_type_events = [e for e in timeline if e["event_type"] in {"MET_LATEST_SURVEY", "NOT_MET_LATEST_SURVEY"}]
        assert len(survey_type_events) == 2  # genuinely distinct dates -> two separate events, not merged


def test_build_timeline_events_bulk_matches_per_facility_output():
    ccn = "999005"
    pubs = ["2023-03", "2023-04"]
    obs_by_pub = {
        "2023-03": [_obs("2023-03", ccn, "SFF_CANDIDATE", table="Table D")],
        "2023-04": [_obs("2023-04", ccn, "CURRENT_SFF", table="Table A")],
    }
    changes = [_change(ccn, {"SFF_CANDIDATE"}, {"CURRENT_SFF"}, from_pub="2023-03", to_pub="2023-04")]
    coverage_rows = [
        {"year_month": "2023-03", "present": "Y"},
        {"year_month": "2023-04", "present": "Y"},
    ]
    bulk = build_timeline_events(
        publications=[],
        observations_by_pub=obs_by_pub,
        changes=changes,
        graduation_events=[],
        survey_events=[],
        coverage_rows=coverage_rows,
    )
    individual = _build_timeline(ccn, observations_by_pub=obs_by_pub, all_publication_ids=pubs, changes=changes)
    assert {e["timeline_event_id"] for e in bulk} == {e["timeline_event_id"] for e in individual}
