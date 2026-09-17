"""Shared constants and small value types for the SFF history layer.

CCNs are always opaque strings. Never digit-strip or numeric-normalize them —
the archive audit found real CMS CCNs that end in a letter (e.g. ``15E064``)
and confirmed that stripping non-digit characters silently corrupts them.
"""

from __future__ import annotations

import re

# Era in scope for this phase. Every table in every Era-3b publication carries
# a leading "Provider Number" (CCN) column; nothing before March 2023 does
# (see SFF_LAYOUT_ANALYSIS.md, Era 3b). Phase 1 deliberately covers only this
# era — the pre-2023 name+address parser is out of scope.
ERA_3B = "era3b_ccn_2023_03_plus"

# CCN is a 6-character alphanumeric token (first 2 chars = USPS state code in
# practice, but that is not relied on for validation). Matches
# pbj-data-ops/sff_release.py's CCN_RE exactly.
CCN_RE = re.compile(r"^[0-9A-Z]{6}$")

USPS = set(
    "AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN "
    "MS MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA "
    "WV WI WY PR VI GU MP AS".split()
)

# Raw CMS table caption -> normalized category. Mirrors
# pbj-data-ops/sff_release.py's CATEGORIES mapping for Era 3a/3b (the 4-table
# model). Do not silently equate this with the Era 1/2 5/6-table vocabulary
# (SFF_ARCHIVE_AUDIT.md S6) — that mapping is out of scope for this phase.
CATEGORIES: dict[str, str] = {
    "Table A": "CURRENT_SFF",
    "Table B": "GRADUATED",
    "Table C": "NO_LONGER_PARTICIPATING",
    "Table D": "SFF_CANDIDATE",
}

NORMALIZED_CATEGORIES = frozenset(CATEGORIES.values())

# Months-in-status column label depends on which table the row came from.
MONTHS_FIELD_LABEL: dict[str, str] = {
    "Table A": "Months as an SFF",
    "Table B": "Months as an SFF",
    "Table C": "Months as an SFF",
    "Table D": "Months as an SFF Candidate",
}

# Explicit CMS-published per-row date, when the table publishes one. Table A's
# "Most Recent Inspection" is not treated as an explicit status/event date —
# it is a survey date, not a transition date (kept separately as
# most_recent_inspection on the observation).
STATUS_DATE_KIND: dict[str, str | None] = {
    "Table A": None,
    "Table B": "graduation",
    "Table C": "termination",
    "Table D": None,
}

PARSER_VERSION = "sff_history.pdf_parser:v1"


def is_valid_ccn(value: str) -> bool:
    return bool(CCN_RE.fullmatch(value))
