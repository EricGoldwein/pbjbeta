from sff_history.observations import Observation
from sff_history.timeline import build_facility_timeline, build_timeline_events, classify_change

ERA = "era3b_ccn_2023_03_plus"
PARSER = "sff_history.pdf_parser:v1"


def _obs(publication_id, ccn, category, *, table, months="1"):
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
        most_recent_inspection="",
        met_survey_criteria="",
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
        re-observed as a Candidate again (re-entry) after a hiatus that
        spans a known missing-publication month.
        """
        ccn = "999001"
        pubs = ["2023-03", "2023-04", "2023-05", "2024-01", "2024-02"]
        obs_by_pub = {
            "2023-03": [_obs("2023-03", ccn, "SFF_CANDIDATE", table="Table D", months="1")],
            "2023-04": [_obs("2023-04", ccn, "CURRENT_SFF", table="Table A", months="1")],
            "2023-05": [_obs("2023-05", ccn, "GRADUATED", table="Table B", months="1")],
            # gap: 2023-12 missing (not a key in obs_by_pub at all)
            "2024-01": [],  # ccn not observed this month (absent from every table)
            "2024-02": [_obs("2024-02", ccn, "SFF_CANDIDATE", table="Table D", months="1")],
        }
        # explicit graduation event, deduplicated form as derive_graduation_events would emit it
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
        gap_months = {"2023-12"}  # the only known gap between 2023-05 and 2024-02 in this scenario

        timeline = build_facility_timeline(
            ccn,
            observations_by_pub=obs_by_pub,
            changes=changes,
            graduation_events=graduation_events,
            gap_months=gap_months,
            all_publication_ids=pubs,
        )
        event_types = [e["event_type"] for e in timeline]

        assert "FIRST_OBSERVED_SFF_CANDIDATE" in event_types
        assert "PROMOTED_TO_CURRENT_SFF" in event_types
        assert "GRADUATED_FROM_CURRENT_SFF" in event_types
        assert "EXPLICIT_GRADUATION" in event_types
        assert "REENTRY_SFF_CANDIDATE" in event_types
        assert "PUBLICATION_GAP" in event_types

        grad_explicit = next(e for e in timeline if e["event_type"] == "EXPLICIT_GRADUATION")
        assert grad_explicit["event_date"] == "2023-05-20"
        assert grad_explicit["event_date_precision"] == "explicit_cms_date"

        reentry = next(e for e in timeline if e["event_type"] == "REENTRY_SFF_CANDIDATE")
        assert reentry["as_of_publication_id"] == "2024-02"
        assert "gap falls within the hiatus" in reentry["summary"]

        gap_event = next(e for e in timeline if e["event_type"] == "PUBLICATION_GAP")
        assert gap_event["as_of_publication_id"] == "2023-12"

        # chronological ordering: first-observed comes before the graduation,
        # which comes before the re-entry.
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
        timeline = build_facility_timeline(
            ccn,
            observations_by_pub=obs_by_pub,
            changes=[],
            graduation_events=[],
            gap_months=set(),
            all_publication_ids=pubs,
        )
        assert not any(e["event_type"].startswith("REENTRY_") for e in timeline)

    def test_reentry_requires_a_real_intervening_absence_not_just_ccn_filtered_adjacency(self):
        """Regression test for a real bug caught during review: span
        adjacency must be computed against every PASS publication in the
        dataset, not just the publications where this CCN happens to have a
        row. A naive CCN-filtered adjacency check would wrongly treat
        03 -> 05 (skipping 04, where the CCN was genuinely absent from every
        table despite 04 being a real, fully-published month) as one
        continuous span and miss the re-entry entirely.
        """
        ccn = "999003"
        pubs = ["2023-03", "2023-04", "2023-05"]
        obs_by_pub = {
            "2023-03": [_obs("2023-03", ccn, "CURRENT_SFF", table="Table A")],
            "2023-04": [],  # 2023-04 is a real, published month; ccn just isn't in it
            "2023-05": [_obs("2023-05", ccn, "CURRENT_SFF", table="Table A")],
        }
        timeline = build_facility_timeline(
            ccn,
            observations_by_pub=obs_by_pub,
            changes=[],
            graduation_events=[],
            gap_months=set(),  # no known gap -- 2023-04 was genuinely published, ccn just absent
            all_publication_ids=pubs,
        )
        reentries = [e for e in timeline if e["event_type"].startswith("REENTRY_")]
        assert len(reentries) == 1
        assert reentries[0]["as_of_publication_id"] == "2023-05"
        assert "directly evidenced" in reentries[0]["summary"]

    def test_reentry_flagged_across_a_missing_calendar_month_even_with_no_intervening_publication(self):
        """Regression test for a second real bug caught during review:
        confirmed real case, CCN 105234 is CURRENT_SFF in both 2024-11 and
        2025-01 with no publication at all for 2024-12 in this archive (a
        known gap). List-position adjacency alone would treat 2024-11 and
        2025-01 as list-adjacent (2024-12 simply isn't a list entry) and
        silently claim one continuous active span across the gap -- exactly
        the kind of bridging derive_changes/derive_intervals are built never
        to do. This must instead surface as a re-entry with
        continuity_uncertain=True, not silence.
        """
        ccn = "105234"
        pubs = ["2024-11", "2025-01"]  # 2024-12 has no PASS publication at all
        obs_by_pub = {
            "2024-11": [_obs("2024-11", ccn, "CURRENT_SFF", table="Table A")],
            "2025-01": [_obs("2025-01", ccn, "CURRENT_SFF", table="Table A")],
        }
        timeline = build_facility_timeline(
            ccn,
            observations_by_pub=obs_by_pub,
            changes=[],
            graduation_events=[],
            gap_months={"2024-12"},
            all_publication_ids=pubs,
        )
        reentries = [e for e in timeline if e["event_type"].startswith("REENTRY_")]
        assert len(reentries) == 1
        assert reentries[0]["event_type"] == "REENTRY_CURRENT_SFF"
        assert reentries[0]["as_of_publication_id"] == "2025-01"
        assert "gap falls within the hiatus" in reentries[0]["summary"]

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
        timeline = build_facility_timeline(
            ccn,
            observations_by_pub=obs_by_pub,
            changes=changes,
            graduation_events=[],
            gap_months=set(),
            all_publication_ids=pubs,
        )
        # No routine-churn change should appear as its own timeline entry.
        assert not any(e["timeline_event_id"].startswith("timeline:change:") for e in timeline)
        # The CCN does still get a FIRST_OBSERVED marker and a re-entry marker
        # (candidate presence itself is meaningful when it resumes after a
        # confirmed absence) -- only the underlying routine change rows are
        # excluded, not the facility's own presence facts.
        assert any(e["event_type"] == "FIRST_OBSERVED_SFF_CANDIDATE" for e in timeline)


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
    bulk = build_timeline_events(publications=[], observations_by_pub=obs_by_pub, changes=changes, graduation_events=[], coverage_rows=coverage_rows)
    individual = build_facility_timeline(
        ccn, observations_by_pub=obs_by_pub, changes=changes, graduation_events=[], gap_months=set(), all_publication_ids=pubs
    )
    assert {e["timeline_event_id"] for e in bulk} == {e["timeline_event_id"] for e in individual}
