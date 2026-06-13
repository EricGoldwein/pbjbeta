"""
Date-confidence for citation/PDF signals vs PBJ daily staffing.

PBJ is daily; do not claim exact incident-day linkage when 2567 narratives use [DATE].
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Any, Optional

# ``exact_incident_date_available`` is reserved for a future verified incident-date source
# (not inferred from redacted 2567 narratives or ambiguous calendar mentions).
_INCIDENT_DATE_CONFIDENCE = (
    "exact_incident_date_available",
    "redacted_or_missing_incident_date",
    "ambiguous_date_available",
)

_REDACTION_TOKENS = ("[DATE]", "[date]", "XXXX", "XXX")
_DATE_IN_TEXT_RE = re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b")


def _parse_iso(s: Optional[str]) -> Optional[date]:
    if not s or len(str(s).strip()) != 10:
        return None
    try:
        y, m, d = str(s).strip().split("-")
        return date(int(y), int(m), int(d))
    except (ValueError, TypeError):
        return None


def _iso(d: date) -> str:
    return d.isoformat()


def _narrative_blob(citation: dict[str, Any]) -> str:
    pdf = citation.get("pdf_enrichment") or citation.get("pdf_extract") or {}
    block = pdf.get("tag_block") or {}
    parts = [
        str(block.get("narrative_excerpt") or ""),
        str(citation.get("deficiency_description") or ""),
    ]
    return " ".join(parts)


def _has_redaction_markers(text: str) -> bool:
    return any(tok in text for tok in _REDACTION_TOKENS)


def build_date_context(
    citation: dict[str, Any],
    *,
    pdf_redaction_heavy: Optional[bool] = None,
) -> dict[str, Any]:
    """
    Build date-confidence fields for one normalized citation (optionally PDF-enriched).

    Survey date from CMS citation row is always the primary anchor when incident date is unknown.
    """
    survey_iso = str(citation.get("survey_date") or "").strip() or None
    survey_d = _parse_iso(survey_iso)

    pdf = citation.get("pdf_enrichment") or citation.get("pdf_extract") or {}
    redaction_heavy = pdf_redaction_heavy
    if redaction_heavy is None:
        redaction_heavy = bool(pdf.get("redaction_heavy") or pdf.get("narrative_dates_redacted"))

    narrative = _narrative_blob(citation)
    narrative_redacted = _has_redaction_markers(narrative)
    dates_in_narrative = sorted(set(_DATE_IN_TEXT_RE.findall(narrative)))

    times = list(pdf.get("times_mentioned") or [])
    if not times:
        block = pdf.get("tag_block") or {}
        times = list(block.get("time_mentions") or [])
    shifts = list(pdf.get("shifts_mentioned") or [])
    if not shifts:
        block = pdf.get("tag_block") or {}
        shifts = list(block.get("shift_mentions") or [])
    time_hints_available = bool(times or shifts)

    incident_date: Optional[str] = None
    confidence: str
    anchor_reason: str

    if redaction_heavy or narrative_redacted:
        confidence = "redacted_or_missing_incident_date"
        anchor_reason = (
            "Narrative dates redacted in CMS 2567 PDF; survey date used for staffing context only."
        )
    elif dates_in_narrative:
        confidence = "ambiguous_date_available"
        anchor_reason = (
            "PDF or description contains calendar dates that may be assessment, care-plan, "
            "or interview dates—not classified as incident dates."
        )
    else:
        confidence = "redacted_or_missing_incident_date"
        anchor_reason = "No reliable incident date in citation or PDF; survey date used as anchor."

    can_exact = confidence == "exact_incident_date_available" and incident_date is not None

    anchor_iso = survey_iso
    if not anchor_iso and survey_d:
        anchor_iso = _iso(survey_d)

    return {
        "incident_date_confidence": confidence,
        "incident_date": incident_date,
        "survey_date": survey_iso,
        "anchor_date_used": anchor_iso,
        "anchor_date_reason": anchor_reason,
        "default_pbj_window_type": "survey_minus_7_plus_7",
        "can_support_exact_daily_pbj_match": can_exact,
        "time_hints_available": time_hints_available,
        "times_mentioned": times[:12],
        "shifts_mentioned": shifts[:6],
        "narrative_date_mentions": dates_in_narrative[:20],
        "redaction_heavy": bool(redaction_heavy or narrative_redacted),
    }


def pbj_window_ranges(
    anchor_date_iso: Optional[str],
    window_types: list[str],
) -> list[dict[str, Any]]:
    """Compute labeled PBJ context windows from an anchor survey date."""
    anchor = _parse_iso(anchor_date_iso)
    if not anchor:
        return []

    out: list[dict[str, Any]] = []
    for wtype in window_types:
        wt = str(wtype).strip()
        if wt == "survey_minus_7_plus_7":
            out.append(
                {
                    "type": wt,
                    "start": _iso(anchor - timedelta(days=7)),
                    "end": _iso(anchor + timedelta(days=7)),
                }
            )
        elif wt == "survey_minus_30_plus_30":
            out.append(
                {
                    "type": wt,
                    "start": _iso(anchor - timedelta(days=30)),
                    "end": _iso(anchor + timedelta(days=30)),
                }
            )
        elif wt == "survey_quarter":
            q_start_month = ((anchor.month - 1) // 3) * 3 + 1
            start = date(anchor.year, q_start_month, 1)
            if q_start_month == 10:
                end = date(anchor.year, 12, 31)
            elif q_start_month == 7:
                end = date(anchor.year, 9, 30)
            elif q_start_month == 4:
                end = date(anchor.year, 6, 30)
            else:
                end = date(anchor.year, 3, 31)
            out.append({"type": wt, "start": _iso(start), "end": _iso(end)})
    return out
