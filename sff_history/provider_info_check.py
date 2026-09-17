"""Cross-source reconciliation evidence: PDF observations vs. Provider Info.

Provider Information's ``Special Focus Status`` (``sff_status``) is a
distinct current CMS signal, not the same source as the SFF posting PDF. The
archive audit found the two disagree in a same-nominal-month comparison,
mostly consistent with a processing-date-skew hypothesis (Provider Info
lagging the PDF by one prior-month label), and warned explicitly against
ever converting that disagreement into an inferred transition
(SFF_CURRENT_RECONCILIATION.md S4-6). This module only ever produces
reconciliation *evidence* rows (agree / disagree-value / absent) — it never
writes to the Layer-3 derived tables and never asserts a transition.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .observations import Observation

# Provider Info's own normalized vocabulary (pbj-data-ops's
# normalize_provider_info.py output), not the raw CMS column text.
_PROVIDER_INFO_TO_CATEGORY = {
    "SFF": "CURRENT_SFF",
    "SFF Candidate": "SFF_CANDIDATE",
}

RECONCILIATION_FIELDS = [
    "publication_id",
    "ccn",
    "pdf_categories",
    "provider_info_sff_status_raw",
    "provider_info_category",
    "ccn_in_provider_info",
    "reconciliation_note",
]


def load_provider_info_ccn_status(csv_path: Path) -> dict[str, str]:
    """ccn -> raw sff_status string, read verbatim (no normalization) from a
    pbj-data-ops ProviderInfoNorm_<YYYY>_<MM>.csv. CCN is kept as an opaque
    string, matching PDF-side handling.
    """
    out: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ccn = str(row.get("ccn") or "").strip()
            if ccn:
                out[ccn] = str(row.get("sff_status") or "").strip()
    return out


def reconcile(
    publication_id: str,
    observations: list[Observation],
    provider_info_status: dict[str, str],
) -> list[dict[str, Any]]:
    """Build one reconciliation row per CCN touched by either source. Never
    computes or labels a "transition" — only ever an agreement/disagreement
    classification for the same nominal publication period.
    """
    pdf_categories: dict[str, set[str]] = {}
    for obs in observations:
        pdf_categories.setdefault(obs.ccn, set()).add(obs.normalized_category)

    all_ccns = sorted(set(pdf_categories) | set(provider_info_status))
    rows: list[dict[str, Any]] = []
    for ccn in all_ccns:
        pdf_cats = pdf_categories.get(ccn, set())
        in_provider_info = ccn in provider_info_status
        raw_status = provider_info_status.get(ccn, "")
        provider_category = _PROVIDER_INFO_TO_CATEGORY.get(raw_status)

        if not in_provider_info:
            note = "CCN_ABSENT_FROM_PROVIDER_INFO"
        elif provider_category is None and not raw_status:
            note = "AGREE" if not pdf_cats else "PROVIDER_INFO_BLANK_PDF_NONBLANK"
        elif provider_category in pdf_cats:
            note = "AGREE"
        else:
            note = "DISAGREE_VALUE"

        rows.append(
            {
                "publication_id": publication_id,
                "ccn": ccn,
                "pdf_categories": ";".join(sorted(pdf_cats)),
                "provider_info_sff_status_raw": raw_status,
                "provider_info_category": provider_category or "",
                "ccn_in_provider_info": "Y" if in_provider_info else "N",
                "reconciliation_note": note,
            }
        )
    return rows
