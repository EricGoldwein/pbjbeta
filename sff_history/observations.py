"""Layer 2: observation / membership records.

Immutable raw evidence. One row per facility x table/category membership
within one publication — many rows per facility per publication when CMS
cross-lists a CCN (the audit found 8 concrete CURRENT_SFF + SFF_CANDIDATE
cross-listings in the August 2026 governed release). Never collapse multiple
memberships into one row, and never modify a row once written; derivation
(Layer 3) only ever reads these.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .schema import CCN_RE, MONTHS_FIELD_LABEL, STATUS_DATE_KIND, USPS

OBSERVATION_FIELDS = [
    "observation_id",
    "publication_id",
    "ccn",
    "raw_facility_name",
    "raw_table",
    "normalized_category",
    "source_page",
    "address",
    "city",
    "state",
    "zip",
    "phone",
    "most_recent_inspection",
    "met_survey_criteria",
    "months_in_status",
    "months_field_label",
    "explicit_status_date",
    "explicit_status_date_kind",
    "era_id",
    "parser_version",
]

_DATE_RE = re.compile(r"\d{2}/\d{2}/\d{4}")


@dataclass
class Observation:
    observation_id: str
    publication_id: str
    ccn: str
    raw_facility_name: str
    raw_table: str
    normalized_category: str
    source_page: str
    address: str
    city: str
    state: str
    zip: str
    phone: str
    most_recent_inspection: str
    met_survey_criteria: str
    months_in_status: str
    months_field_label: str
    explicit_status_date: str
    explicit_status_date_kind: str | None
    era_id: str
    parser_version: str

    def to_row(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "publication_id": self.publication_id,
            "ccn": self.ccn,
            "raw_facility_name": self.raw_facility_name,
            "raw_table": self.raw_table,
            "normalized_category": self.normalized_category,
            "source_page": self.source_page,
            "address": self.address,
            "city": self.city,
            "state": self.state,
            "zip": self.zip,
            "phone": self.phone,
            "most_recent_inspection": self.most_recent_inspection,
            "met_survey_criteria": self.met_survey_criteria,
            "months_in_status": self.months_in_status,
            "months_field_label": self.months_field_label,
            "explicit_status_date": self.explicit_status_date,
            "explicit_status_date_kind": self.explicit_status_date_kind or "",
            "era_id": self.era_id,
            "parser_version": self.parser_version,
        }


def build_observations(
    publication_id: str, raw_rows: list[dict[str, str]], *, era_id: str, parser_version: str
) -> list[Observation]:
    """Convert one publication's raw parsed rows into Layer-2 observations.

    ``observation_id`` is deterministic (publication_id + table + ccn + a
    per-(table,ccn) sequence index), so a re-run over the same PDF produces
    identical IDs — required for idempotent regeneration.
    """
    seen: Counter[tuple[str, str]] = Counter()
    observations: list[Observation] = []
    for row in raw_rows:
        table = row["source_table"]
        ccn = row["ccn"]  # opaque string; never digit-strip or numeric-normalize
        key = (table, ccn)
        seq = seen[key]
        seen[key] += 1
        observation_id = f"{publication_id}:{table}:{ccn}:{seq}"

        most_recent_inspection = row["status_date"] if table == "Table A" else ""
        met_survey_criteria = row["survey_criteria"] if table == "Table A" else ""
        status_date_kind = STATUS_DATE_KIND[table]
        explicit_status_date = row["status_date"] if table in {"Table B", "Table C"} else ""

        observations.append(
            Observation(
                observation_id=observation_id,
                publication_id=publication_id,
                ccn=ccn,
                raw_facility_name=row["facility_name"],
                raw_table=table,
                normalized_category=row["category"],
                source_page=row["source_page"],
                address=row["address"],
                city=row["city"],
                state=row["state"],
                zip=row["zip"],
                phone=row["phone"],
                most_recent_inspection=most_recent_inspection,
                met_survey_criteria=met_survey_criteria,
                months_in_status=row["months_in_status"],
                months_field_label=MONTHS_FIELD_LABEL[table],
                explicit_status_date=explicit_status_date,
                explicit_status_date_kind=status_date_kind if explicit_status_date else None,
                era_id=era_id,
                parser_version=parser_version,
            )
        )
    return observations


def validate_observations(
    observations: list[Observation], *, provider_ccns: set[str] | None = None
) -> dict[str, Any]:
    """Field-level validation, mirroring pbj-data-ops/sff_release.py's
    ``validate_rows`` checks (CCN shape, USPS state codes, required fields,
    date shape, months plausibility, survey-criteria vocabulary), plus an
    optional cross-check against a contemporaneous Provider Info CCN roster.

    Split into ``errors`` (structural problems that should block a
    publication from entering derivation — e.g. a CCN that isn't
    CCN-shaped, which likely means the geometry parser mis-anchored a row)
    and ``warnings`` (values that are directly, verifiably present in CMS's
    own source PDF text and are not parser artifacts, but are still
    implausible — e.g. a duplicate CCN row within one category, or a
    negative "months in status" counter, both confirmed in this archive by
    direct inspection of the underlying PDF text). Treating a warning as
    fatal would silently manufacture a false publication gap out of one
    anomalous source row and break interval continuity for hundreds of
    unrelated, clean rows in the same publication — worse than preserving
    the anomaly as-is.
    """
    errors: list[str] = []
    warnings: list[str] = []
    invalid_ccns = sorted({o.ccn for o in observations if not CCN_RE.fullmatch(o.ccn)})
    invalid_states = sorted({o.state for o in observations if o.state not in USPS})
    duplicate_pairs = [
        key
        for key, count in Counter((o.normalized_category, o.ccn) for o in observations).items()
        if count > 1
    ]
    if invalid_ccns:
        errors.append(f"invalid CCNs: {invalid_ccns[:5]}")
    if invalid_states:
        errors.append(f"invalid states: {invalid_states}")
    if duplicate_pairs:
        warnings.append(f"duplicate CCN within category (verified present in source PDF text): {duplicate_pairs[:5]}")

    required_blank = sorted(
        {
            "ccn" if not o.ccn else "facility_name" if not o.raw_facility_name else ""
            for o in observations
            if not o.ccn or not o.raw_facility_name
        }
        - {""}
    )
    if required_blank:
        errors.append(f"blank required fields: {required_blank}")

    bad_dates = [
        o.observation_id
        for o in observations
        if o.explicit_status_date and not _DATE_RE.fullmatch(o.explicit_status_date)
    ]
    if bad_dates:
        errors.append(f"invalid explicit status dates: {bad_dates[:5]}")

    bad_months = [
        o.observation_id
        for o in observations
        if not o.months_in_status.lstrip("-").isdigit() or not 0 <= int(o.months_in_status) <= 200
    ]
    if bad_months:
        warnings.append(f"implausible months-in-status value in source PDF: {bad_months[:5]}")

    bad_criteria = [
        o.observation_id
        for o in observations
        if o.normalized_category == "CURRENT_SFF" and o.met_survey_criteria not in {"", "Met", "Not Met"}
    ]
    if bad_criteria:
        errors.append(f"invalid survey criteria: {bad_criteria[:5]}")

    known = provider_ccns or set()
    unique_ccns = {o.ccn for o in observations}
    unmatched = unique_ccns - known if known else set()

    return {
        "status": "PASS" if not errors else "FAIL",
        "observation_count": len(observations),
        "unique_ccn_count": len(unique_ccns),
        "category_counts": dict(sorted(Counter(o.normalized_category for o in observations).items())),
        "cross_category_ccn_count": len(observations) - len(unique_ccns),
        "warnings": warnings,
        "provider_info_mapping": {
            "reference_available": bool(known),
            "matched": len(unique_ccns & known),
            "unmatched": len(unmatched),
            "unmatched_ccns": sorted(unmatched),
        },
        "errors": errors,
    }
