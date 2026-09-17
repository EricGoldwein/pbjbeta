"""Layer 4: compact, human-facing facility timeline contract.

This is a read-only *presentation* view computed from Layers 1-3 — it never
modifies ``observations`` (Layer 2) or ``derived_changes`` /
``derived_graduation_events`` / ``derived_intervals`` (Layer 3), and it adds
no new source-of-truth facts of its own. Its only job is to answer one
question for a future UI: "what is worth showing a person about this
facility's SFF history, in order, without the UI re-deriving anything
itself?"

Timeline-event contract
------------------------
Each row is one ``TimelineEvent`` with these fields (``TIMELINE_EVENT_FIELDS``):

- ``timeline_event_id`` — deterministic, stable across rebuilds.
- ``ccn`` — opaque string, never modified.
- ``event_type`` — one of the constants below. This is the whole contract:
  a UI switches on this value and never needs to know how it was computed.
- ``as_of_publication_id`` — the publication this event is anchored to for
  chronological ordering (``PUBLICATION_GAP`` uses the missing YYYY-MM
  instead).
- ``event_date`` — ISO date, populated **only** when CMS published the date
  directly (graduation/termination events). Every other event type leaves
  this blank and is ordered/dated only to publication-period granularity —
  never invent day-level precision from a monthly snapshot.
- ``event_date_precision`` — ``"explicit_cms_date"`` | ``"publication_period"``
  | ``"gap"``, so a UI can render date confidence correctly instead of
  treating every date the same way.
- ``summary`` — a short, human-readable sentence. Convenience only; a UI is
  free to re-render from the structured fields instead.
- ``derived_from`` — ``;``-joined provenance IDs (observation_id / change_id
  / event_id / coverage year_month) pointing back to the exact Layer 1-3
  evidence this event was built from.

Event types
-----------
``FIRST_OBSERVED_CANDIDATE`` / ``FIRST_OBSERVED_CURRENT_SFF`` /
``FIRST_OBSERVED_GRADUATED`` / ``FIRST_OBSERVED_NO_LONGER_PARTICIPATING``
    The earliest publication in this dataset's window in which the CCN
    appears at all, labeled by its most substantive category that month.
    This is explicitly a left-censored fact ("earliest we have on record"),
    never a claim about when the facility actually entered the program.

``PROMOTED_TO_CURRENT_SFF``
    An OBSERVED_CHANGE (Layer 3) where CURRENT_SFF newly appears in the
    category set between two adjacent, gapless publications (typically a
    Candidate being selected onto the SFF list, sometimes while still
    cross-listed as a Candidate the same month).

``GRADUATED_FROM_CURRENT_SFF``
    An OBSERVED_CHANGE where GRADUATED newly appears in the category set
    (in every case observed so far in this archive, CURRENT_SFF was held
    immediately before, which the label reflects; the classifier does not
    require it). Distinct from, and shown alongside, ``EXPLICIT_GRADUATION``
    — the change is "we detected this transition between two snapshots";
    the explicit event is "CMS published this exact date." They usually
    co-occur but are never merged into one fact.

``NO_LONGER_PARTICIPATING``
    An OBSERVED_CHANGE where NO_LONGER_PARTICIPATING newly appears.

``LEFT_CURRENT_SFF_NO_STATED_OUTCOME``
    An OBSERVED_CHANGE where CURRENT_SFF is lost and the CCN is not
    observed in any category the following gapless publication (no
    Graduated/Terminated/Candidate destination was stated). Flagged, not
    silently dropped — CMS gave no stated outcome, so none is invented.

``EXPLICIT_GRADUATION`` / ``EXPLICIT_TERMINATION``
    A deduplicated canonical event from ``derive_graduation_events`` — CMS's
    own stated "Date of Graduation" / "Date of Termination", collapsed
    across every publication that re-states the identical
    (ccn, event_kind, event_date), per the dedup fix in ``derive.py``.

``REENTRY_CANDIDATE`` / ``REENTRY_CURRENT_SFF``
    The CCN is observed active (CURRENT_SFF or SFF_CANDIDATE) again after a
    prior active span had already ended (graduated, terminated, or simply
    dropped from every table). Only ever asserts that both observed
    instants are real (each is its own OBSERVED_SNAPSHOT); never asserts
    when in between the facility actually left or returned. Carries
    ``continuity_uncertain`` in its summary text and derived_from when a
    known publication gap falls in the hiatus (see ``PUBLICATION_GAP``).

``PUBLICATION_GAP``
    A known missing-publication month (from ``publication_coverage``) that
    falls within this CCN's observed span. Not a fact about the facility —
    a fact about what CMS did not publish — surfaced so a UI never silently
    draws a continuous line across a month with no evidence.

Deliberately excluded from this compact timeline (still fully available,
unfiltered, in ``derived_changes.csv``): routine month-to-month SFF
Candidate list churn — a CCN entering or leaving Table D alone, with no
CURRENT_SFF/GRADUATED/NO_LONGER_PARTICIPATING involvement. CMS recomputes
the candidate pool every month by methodology (min 5/max 30 per state); by
itself that churn is expected and not informative about any one facility.
"""

from __future__ import annotations

from typing import Any

from .observations import Observation
from .publications import Publication
from .schema import NORMALIZED_CATEGORIES

TIMELINE_EVENT_FIELDS = [
    "timeline_event_id",
    "ccn",
    "event_type",
    "as_of_publication_id",
    "event_date",
    "event_date_precision",
    "summary",
    "derived_from",
]

# Category-substantiveness order used only to pick the label when a CCN's
# very first observation already spans multiple categories in one posting.
_FIRST_OBSERVED_PRIORITY = ("CURRENT_SFF", "GRADUATED", "NO_LONGER_PARTICIPATING", "SFF_CANDIDATE")
_ACTIVE_CATEGORIES = frozenset({"CURRENT_SFF", "SFF_CANDIDATE"})
_CANDIDATE_ONLY = frozenset({"SFF_CANDIDATE"})


def classify_change(change_row: dict[str, Any]) -> str | None:
    """Classify one ``derived_changes`` row for timeline prominence.

    Returns a meaningful ``event_type`` name, or ``None`` if the change is
    routine Candidate-list churn that should not appear on a compact,
    human-facing timeline (it remains fully visible in derived_changes.csv
    either way — this function never deletes or alters that row).
    """
    from_set = set(change_row["from_categories"].split(";")) if change_row["from_categories"] else set()
    to_set = set(change_row["to_categories"].split(";")) if change_row["to_categories"] else set()
    gained = to_set - from_set
    lost = from_set - to_set

    if "CURRENT_SFF" in gained:
        return "PROMOTED_TO_CURRENT_SFF"
    if "GRADUATED" in gained:
        return "GRADUATED_FROM_CURRENT_SFF"
    if "NO_LONGER_PARTICIPATING" in gained:
        return "NO_LONGER_PARTICIPATING"
    if "CURRENT_SFF" in lost and not to_set:
        return "LEFT_CURRENT_SFF_NO_STATED_OUTCOME"
    if gained <= _CANDIDATE_ONLY and lost <= _CANDIDATE_ONLY:
        return None  # routine Candidate-list churn only
    return "OTHER_CATEGORY_CHANGE"  # conservative catch-all; never silently dropped


def _first_observed_events(ccn: str, sorted_obs_by_pub: list[tuple[str, list[Observation]]]) -> list[dict[str, Any]]:
    for publication_id, observations in sorted_obs_by_pub:
        categories = {o.normalized_category for o in observations if o.ccn == ccn}
        if not categories:
            continue
        label_category = next((c for c in _FIRST_OBSERVED_PRIORITY if c in categories), sorted(categories)[0])
        observation_ids = sorted(o.observation_id for o in observations if o.ccn == ccn)
        return [
            {
                "timeline_event_id": f"timeline:{ccn}:FIRST_OBSERVED:{publication_id}",
                "ccn": ccn,
                "event_type": f"FIRST_OBSERVED_{label_category}",
                "as_of_publication_id": publication_id,
                "event_date": "",
                "event_date_precision": "publication_period",
                "summary": (
                    f"First observed in this dataset's window as {label_category.replace('_', ' ').title()} "
                    f"in {publication_id} (earliest available publication; not a claim about actual program entry)."
                ),
                "derived_from": ";".join(observation_ids),
            }
        ]
    return []


def _reentry_events(
    ccn: str,
    all_publication_ids: list[str],
    observations_by_pub: dict[str, list[Observation]],
    gap_months: set[str],
) -> list[dict[str, Any]]:
    """Span detection walks *every* PASS publication in the whole dataset, in
    order — not just the publications where this CCN happens to appear.
    Using only the CCN-filtered publication list would wrongly treat "CCN
    absent from a real, published month" the same as "that month simply
    doesn't exist in the dataset," silently skipping over a genuine
    disappearance and missing a real re-entry.

    A span only continues across two active publications that are both
    list-adjacent *and* calendar-adjacent. List-adjacency alone is not
    enough: if a whole calendar month has no PASS publication at all (a
    known archive gap), it is simply missing from ``all_publication_ids`` —
    so the active publication immediately before it and the one immediately
    after it would be list-adjacent despite a real, unevidenced month
    sitting between them. Silently treating that as one continuous span
    would assert continuity across exactly the kind of gap this layer is
    built never to bridge (see ``coverage.gapless_runs`` for the equivalent
    discipline in ``derive_changes``/``derive_intervals``). Confirmed real:
    CCN 105234 is CURRENT_SFF in both 2024-11 and 2025-01 with no
    publication for 2024-12 in this archive.
    """
    active_by_pub: dict[str, set[str]] = {
        pub_id: {o.normalized_category for o in observations_by_pub.get(pub_id, []) if o.ccn == ccn}
        for pub_id in all_publication_ids
    }

    active_spans: list[list[str]] = []  # each span: calendar-adjacent publication_ids where ccn is active
    prev_active_index: int | None = None
    for index, publication_id in enumerate(all_publication_ids):
        if not (active_by_pub[publication_id] & _ACTIVE_CATEGORIES):
            continue
        if (
            active_spans
            and prev_active_index == index - 1
            and _calendar_adjacent(all_publication_ids[index - 1], publication_id)
        ):
            active_spans[-1].append(publication_id)
        else:
            active_spans.append([publication_id])
        prev_active_index = index

    if len(active_spans) < 2:
        return []

    events: list[dict[str, Any]] = []
    for previous_span, next_span in zip(active_spans, active_spans[1:]):
        prev_end, next_start = previous_span[-1], next_span[0]
        continuity_uncertain = _hiatus_spans_a_gap(prev_end, next_start, gap_months)
        reentry_category = "CURRENT_SFF" if "CURRENT_SFF" in active_by_pub[next_start] else "SFF_CANDIDATE"
        observation_ids = sorted(
            o.observation_id for o in observations_by_pub.get(next_start, []) if o.ccn == ccn
        )
        uncertainty_note = (
            " A known missing-publication gap falls within the hiatus, so continuity across it cannot be claimed."
            if continuity_uncertain
            else " No known publication gap falls within the hiatus; the absence is directly evidenced by published data."
        )
        events.append(
            {
                "timeline_event_id": f"timeline:{ccn}:REENTRY:{next_start}",
                "ccn": ccn,
                "event_type": f"REENTRY_{reentry_category}",
                "as_of_publication_id": next_start,
                "event_date": "",
                "event_date_precision": "publication_period",
                "summary": (
                    f"Observed active again as {reentry_category.replace('_', ' ').title()} in {next_start}, "
                    f"after last being observed active in {prev_end}." + uncertainty_note
                ),
                "derived_from": ";".join(observation_ids),
            }
        )
    return events


def _calendar_adjacent(prev_pub_id: str, next_pub_id: str) -> bool:
    py, pm = (int(x) for x in prev_pub_id.split("-"))
    py, pm = (py, pm + 1) if pm < 12 else (py + 1, 1)
    return f"{py}-{pm:02d}" == next_pub_id


def _hiatus_spans_a_gap(prev_pub_id: str, next_pub_id: str, gap_months: set[str]) -> bool:
    py, pm = (int(x) for x in prev_pub_id.split("-"))
    ny, nm = (int(x) for x in next_pub_id.split("-"))
    year, month = py, pm
    while (year, month) != (ny, nm):
        month += 1
        if month > 12:
            month, year = 1, year + 1
        if f"{year}-{month:02d}" in gap_months and (year, month) != (ny, nm):
            return True
    return False


def _gap_events(ccn: str, first_pub: str, last_pub: str, gap_months: set[str]) -> list[dict[str, Any]]:
    events = []
    for year_month in sorted(m for m in gap_months if first_pub <= m <= last_pub):
        events.append(
            {
                "timeline_event_id": f"timeline:{ccn}:GAP:{year_month}",
                "ccn": ccn,
                "event_type": "PUBLICATION_GAP",
                "as_of_publication_id": year_month,
                "event_date": "",
                "event_date_precision": "gap",
                "summary": f"CMS did not publish an SFF posting for {year_month} (or none is staged in this archive); no continuity claim can be made across this month.",
                "derived_from": f"coverage:{year_month}",
            }
        )
    return events


def build_facility_timeline(
    ccn: str,
    *,
    observations_by_pub: dict[str, list[Observation]],
    changes: list[dict[str, Any]],
    graduation_events: list[dict[str, Any]],
    gap_months: set[str],
    all_publication_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build one CCN's compact timeline from Layers 1-3. Pure function; reads
    only, never mutates its inputs.

    ``all_publication_ids`` should be every PASS publication_id in the whole
    dataset, sorted — required for correct re-entry/hiatus detection (see
    ``_reentry_events``). Defaults to the CCN's own observed publications if
    omitted, which is only correct when that CCN happens to appear in every
    PASS publication in the dataset (true for none of them in practice);
    callers processing the full dataset should always pass the real list —
    see ``build_timeline_events``.
    """
    sorted_obs_by_pub = sorted(
        ((pub_id, obs_list) for pub_id, obs_list in observations_by_pub.items() if any(o.ccn == ccn for o in obs_list)),
        key=lambda pair: pair[0],
    )
    if not sorted_obs_by_pub:
        return []

    global_publication_ids = all_publication_ids if all_publication_ids is not None else sorted(observations_by_pub)

    events: list[dict[str, Any]] = []
    events.extend(_first_observed_events(ccn, sorted_obs_by_pub))
    events.extend(_reentry_events(ccn, global_publication_ids, observations_by_pub, gap_months))

    for change in changes:
        if change["ccn"] != ccn:
            continue
        event_type = classify_change(change)
        if event_type is None:
            continue
        events.append(
            {
                "timeline_event_id": f"timeline:{change['change_id']}",
                "ccn": ccn,
                "event_type": event_type,
                "as_of_publication_id": change["to_publication_id"],
                "event_date": "",
                "event_date_precision": "publication_period",
                "summary": (
                    f"{event_type.replace('_', ' ').title()}: category set changed from "
                    f"[{change['from_categories'] or 'none observed'}] to [{change['to_categories'] or 'none observed'}] "
                    f"between {change['from_publication_id']} and {change['to_publication_id']}."
                ),
                "derived_from": change["derived_from_observation_ids"],
            }
        )

    for event in graduation_events:
        if event["ccn"] != ccn:
            continue
        kind = "EXPLICIT_GRADUATION" if event["event_kind"] == "GRADUATION" else "EXPLICIT_TERMINATION"
        events.append(
            {
                "timeline_event_id": f"timeline:{event['event_id']}",
                "ccn": ccn,
                "event_type": kind,
                "as_of_publication_id": event["first_observed_publication_id"],
                "event_date": event["event_date"],
                "event_date_precision": "explicit_cms_date",
                "summary": (
                    f"CMS published {kind.replace('EXPLICIT_', '').title()} date {event['event_date']} "
                    f"(first appeared stating this in {event['first_observed_publication_id']}; "
                    f"restated in {event['observation_count']} publication(s) through {event['last_observed_publication_id']})."
                ),
                "derived_from": event["event_id"],
            }
        )

    first_pub, last_pub = sorted_obs_by_pub[0][0], sorted_obs_by_pub[-1][0]
    events.extend(_gap_events(ccn, first_pub, last_pub, gap_months))

    events.sort(key=lambda e: (e["event_date"] or e["as_of_publication_id"], e["as_of_publication_id"], e["event_type"]))
    return events


def build_timeline_events(
    publications: list[Publication],
    observations_by_pub: dict[str, list[Observation]],
    changes: list[dict[str, Any]],
    graduation_events: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Bulk-build the compact timeline for every CCN observed anywhere in the
    dataset. This is the artifact a UI actually queries (filter by ``ccn``)
    rather than re-deriving history itself.
    """
    gap_months = {row["year_month"] for row in coverage_rows if row["present"] == "N"}
    all_publication_ids = sorted(observations_by_pub)
    all_ccns: set[str] = set()
    for observations in observations_by_pub.values():
        all_ccns.update(o.ccn for o in observations)

    changes_by_ccn: dict[str, list[dict[str, Any]]] = {}
    for change in changes:
        changes_by_ccn.setdefault(change["ccn"], []).append(change)
    events_by_ccn: dict[str, list[dict[str, Any]]] = {}
    for event in graduation_events:
        events_by_ccn.setdefault(event["ccn"], []).append(event)

    out: list[dict[str, Any]] = []
    for ccn in sorted(all_ccns):
        out.extend(
            build_facility_timeline(
                ccn,
                all_publication_ids=all_publication_ids,
                observations_by_pub=observations_by_pub,
                changes=changes_by_ccn.get(ccn, []),
                graduation_events=events_by_ccn.get(ccn, []),
                gap_months=gap_months,
            )
        )
    return out
