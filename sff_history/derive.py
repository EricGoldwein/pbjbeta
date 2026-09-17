"""Layer 3: derived facts.

Reads Layer 1 (publications) and Layer 2 (observations) only. Never writes
back to either — a bug or re-derivation here must be recoverable by
recomputation alone. Three fact tables, matching the corrected canonical
model in SFF_ARCHIVE_AUDIT.md S12:

- ``derived_changes``     — OBSERVED_CHANGE: a category-membership difference
  between two adjacent, gapless publications. Never carries a date beyond
  what CMS itself published (there usually is none).
- ``derived_graduation_events`` — the EXPLICIT_SOURCE_EVENT case: CMS's own
  stated "Date of Graduation" / "Date of Termination" on a single row. Dated
  only because CMS supplied the date directly, independent of which snapshot
  first surfaced the change.
- ``derived_intervals``   — DERIVED_INTERVAL: a contiguous run (>=2
  publications) of adjacent, gapless snapshots showing one CCN holding the
  same normalized category throughout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .coverage import gapless_runs
from .observations import Observation
from .publications import Publication
from .schema import NORMALIZED_CATEGORIES

_EXPLICIT_DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")

CHANGE_FIELDS = [
    "change_id",
    "ccn",
    "from_publication_id",
    "to_publication_id",
    "change_kind",
    "from_categories",
    "to_categories",
    "gapless",
    "derived_from_observation_ids",
    "asserted_date",
    "method",
]

GRADUATION_EVENT_FIELDS = [
    "event_id",
    "ccn",
    "event_kind",
    "event_date",
    "event_date_raw",
    "first_observed_publication_id",
    "last_observed_publication_id",
    "observation_count",
    "supporting_publication_ids",
    "supporting_observation_ids",
]

INTERVAL_FIELDS = [
    "interval_id",
    "ccn",
    "normalized_category",
    "start_publication_id",
    "end_publication_id",
    "publication_ids",
    "months_counter_consistency",
    "derived_from_observation_ids",
]


def _explicit_iso_date(raw: str) -> str | None:
    match = _EXPLICIT_DATE_RE.match(raw)
    if not match:
        return None
    mm, dd, yyyy = match.groups()
    return f"{yyyy}-{mm}-{dd}"


def derive_graduation_events(observations_by_pub: dict[str, list[Observation]]) -> list[dict[str, Any]]:
    """EXPLICIT_SOURCE_EVENT rows: only when CMS's own row carries a parseable
    graduation/termination date. A category change detected merely by
    comparing two snapshots gets no date here — see derive_changes.

    CMS's Graduated (Table B) and No-Longer-Participating (Table C) tables
    are not "recently graduated" rolling lists — a facility can be re-listed,
    with the identical stated date, across dozens of consecutive monthly
    postings (confirmed: one CCN carries the same graduation date across 26
    consecutive publications spanning nearly two years). One raw row per
    publication is therefore evidence *supporting* one real-world event, not
    a separate event. Canonical event identity is ``(ccn, event_kind,
    event_date)`` — every raw row sharing that key collapses into one output
    row, with the full set of corroborating publications/observations kept
    as provenance rather than discarded.

    A CCN can legitimately carry two *different* dates for the same
    event_kind (a handful of confirmed cases in this archive: CMS revised a
    previously-stated date once, then held the corrected date stable in
    every later posting). Because the date differs, that correctly produces
    two canonical events here rather than being silently merged — this
    module does not attempt to infer which of two differing CMS-published
    dates was "correct."
    """
    groups: dict[tuple[str, str, str], list[tuple[str, Observation]]] = {}
    for publication_id, observations in observations_by_pub.items():
        for obs in observations:
            if not obs.explicit_status_date_kind or not obs.explicit_status_date:
                continue
            key = (obs.ccn, obs.explicit_status_date_kind, obs.explicit_status_date)
            groups.setdefault(key, []).append((publication_id, obs))

    events: list[dict[str, Any]] = []
    for (ccn, kind, raw_date), items in groups.items():
        items.sort(key=lambda pair: pair[0])
        supporting_publication_ids = sorted({pub_id for pub_id, _ in items})
        supporting_observation_ids = sorted({obs.observation_id for _, obs in items})
        iso = _explicit_iso_date(raw_date)
        events.append(
            {
                "event_id": f"event:{ccn}:{kind}:{raw_date.replace('/', '-')}",
                "ccn": ccn,
                "event_kind": kind.upper(),
                "event_date": iso or "",
                "event_date_raw": raw_date,
                "first_observed_publication_id": supporting_publication_ids[0],
                "last_observed_publication_id": supporting_publication_ids[-1],
                "observation_count": len(items),
                "supporting_publication_ids": ";".join(supporting_publication_ids),
                "supporting_observation_ids": ";".join(supporting_observation_ids),
            }
        )
    return events


def _category_map(observations: list[Observation]) -> dict[str, dict[str, str]]:
    """ccn -> {normalized_category: observation_id} for one publication.

    Only the first observation_id per (ccn, category) is kept as the
    provenance pointer; duplicates are a validation concern (see
    observations.validate_observations), not a derivation concern.
    """
    out: dict[str, dict[str, str]] = {}
    for obs in observations:
        out.setdefault(obs.ccn, {})
        out[obs.ccn].setdefault(obs.normalized_category, obs.observation_id)
    return out


def derive_changes(publications: list[Publication], observations_by_pub: dict[str, list[Observation]]) -> list[dict[str, Any]]:
    """OBSERVED_CHANGE rows, computed only across gapless-adjacent PASS
    publications (see coverage.gapless_runs) — a category difference across a
    missing month is not derived at all, per the audit's governing principle.
    """
    changes: list[dict[str, Any]] = []
    for run in gapless_runs(publications):
        for prev_pub, next_pub in zip(run, run[1:]):
            prev_map = _category_map(observations_by_pub.get(prev_pub.publication_id, []))
            next_map = _category_map(observations_by_pub.get(next_pub.publication_id, []))
            all_ccns = sorted(set(prev_map) | set(next_map))
            for ccn in all_ccns:
                prev_cats = prev_map.get(ccn, {})
                next_cats = next_map.get(ccn, {})
                if set(prev_cats) == set(next_cats):
                    continue
                if not prev_cats:
                    kind = "NEWLY_OBSERVED"
                elif not next_cats:
                    kind = "NO_LONGER_OBSERVED"
                else:
                    kind = "CATEGORY_SET_CHANGED"
                derived_from = sorted(set(prev_cats.values()) | set(next_cats.values()))
                changes.append(
                    {
                        "change_id": f"change:{prev_pub.publication_id}:{next_pub.publication_id}:{ccn}",
                        "ccn": ccn,
                        "from_publication_id": prev_pub.publication_id,
                        "to_publication_id": next_pub.publication_id,
                        "change_kind": kind,
                        "from_categories": ";".join(sorted(prev_cats)),
                        "to_categories": ";".join(sorted(next_cats)),
                        "gapless": "Y",
                        "derived_from_observation_ids": ";".join(derived_from),
                        "asserted_date": "",  # never inferred — only CMS-published dates get one (see derive_graduation_events)
                        "method": "adjacent_snapshot_comparison,same_parser,no_gap",
                    }
                )
    return changes


def _months_value(observations: list[Observation], ccn: str, category: str) -> int | None:
    for obs in observations:
        if obs.ccn == ccn and obs.normalized_category == category and obs.months_in_status.isdigit():
            return int(obs.months_in_status)
    return None


def derive_intervals(publications: list[Publication], observations_by_pub: dict[str, list[Observation]]) -> list[dict[str, Any]]:
    """DERIVED_INTERVAL rows: a maximal run (>=2 publications) of
    calendar-adjacent, gapless snapshots where one CCN holds the same
    normalized category in every snapshot of the run. Never spans a run
    boundary (i.e. never spans a known missing-publication gap).
    """
    intervals: list[dict[str, Any]] = []
    for run in gapless_runs(publications):
        membership = [_category_map(observations_by_pub.get(pub.publication_id, [])) for pub in run]
        all_ccns: set[str] = set()
        for cat_map in membership:
            all_ccns.update(cat_map.keys())

        for ccn in sorted(all_ccns):
            for category in sorted(NORMALIZED_CATEGORIES):
                i, n = 0, len(run)
                while i < n:
                    if category not in membership[i].get(ccn, {}):
                        i += 1
                        continue
                    j = i
                    while j + 1 < n and category in membership[j + 1].get(ccn, {}):
                        j += 1
                    if j > i:  # a single-snapshot "interval" is just the observation itself
                        pub_ids = [run[k].publication_id for k in range(i, j + 1)]
                        derived_from = [membership[k][ccn][category] for k in range(i, j + 1)]
                        consistency = _months_counter_consistency(
                            [observations_by_pub.get(run[k].publication_id, []) for k in range(i, j + 1)],
                            ccn,
                            category,
                        )
                        intervals.append(
                            {
                                "interval_id": f"interval:{run[i].publication_id}:{run[j].publication_id}:{ccn}:{category}",
                                "ccn": ccn,
                                "normalized_category": category,
                                "start_publication_id": run[i].publication_id,
                                "end_publication_id": run[j].publication_id,
                                "publication_ids": ";".join(pub_ids),
                                "months_counter_consistency": consistency,
                                "derived_from_observation_ids": ";".join(derived_from),
                            }
                        )
                    i = j + 1
    return intervals


def _months_counter_consistency(obs_sequence: list[list[Observation]], ccn: str, category: str) -> str:
    values = [_months_value(obs, ccn, category) for obs in obs_sequence]
    pairs_checked = 0
    for prev_val, next_val in zip(values, values[1:]):
        if prev_val is None or next_val is None:
            continue
        pairs_checked += 1
        if next_val != prev_val + 1:
            return "inconsistent"
    return "consistent" if pairs_checked else "not_checked"
