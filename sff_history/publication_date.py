"""Extract CMS's own in-PDF publication/update date from an SFF posting.

Two precisions are possible from this archive:

- Day precision: the page body text carries a footer like
  ``Updated July 29, 2026`` on every table page (confirmed by direct
  inspection across multiple Era-3b files; see SFF_LAYOUT_ANALYSIS.md).
- Month precision only: pbj-data-ops/sff_release.py's existing regex matches
  ``Updated <Month> <Year>`` against the raw PDF byte stream, which also
  matches the PDF's own ``/Title`` metadata string (e.g.
  ``... Updated July 2026``) even when the body-text day is unavailable.

Day precision is preferred when extractable; month-only is a documented
fallback, never silently promoted to day precision.
"""

from __future__ import annotations

import re
from pathlib import Path

_MONTH_SLUGS = (
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
)
_MONTH_NAME_TO_NUM = {name: index for index, name in enumerate(_MONTH_SLUGS, start=1)}

# Body-text pattern: "Updated July 29, 2026" (day precision).
_UPDATED_DAY_RE = re.compile(r"Updated\s+([A-Za-z]+)\s+(\d{1,2}),\s*(20\d{2})", re.I)

# Byte-stream pattern: "Updated July 2026" (month precision only). Matches
# pbj-data-ops/sff_release.py's SFF_POSTING_UPDATED_RE exactly.
_UPDATED_MONTH_RE = re.compile(r"Updated\s+([A-Za-z]+)\s+(20\d{2})", re.I)


class PublicationDate:
    __slots__ = ("release_id", "iso_date", "precision", "raw_label")

    def __init__(self, release_id: str, iso_date: str | None, precision: str, raw_label: str):
        self.release_id = release_id  # "YYYY-MM"
        self.iso_date = iso_date  # "YYYY-MM-DD" or None
        self.precision = precision  # "day" | "month" | "unknown"
        self.raw_label = raw_label  # verbatim matched text

    def to_dict(self) -> dict[str, str | None]:
        return {
            "release_id": self.release_id,
            "updated_date": self.iso_date,
            "updated_date_precision": self.precision,
            "updated_label_raw": self.raw_label,
        }


def extract_publication_date(pdf_path: Path, *, fallback_release_id: str | None = None) -> PublicationDate:
    """Best-effort extraction of CMS's own stated publication date.

    Tries day precision from page body text first (the footer repeats on
    every table page, but the first table page varies by file — cover/intro
    pages come first — so every page is scanned until a match is found),
    then falls back to month-only from the raw PDF bytes, then falls back to
    ``fallback_release_id`` (e.g. a filename-derived YYYY-MM) with precision
    "unknown" if neither pattern is found anywhere in the file.
    """
    day_match = _find_day_precision(pdf_path)
    if day_match:
        month_name, day, year = day_match
        month_num = _MONTH_NAME_TO_NUM.get(month_name.lower())
        if month_num:
            release_id = f"{year}-{month_num:02d}"
            iso_date = f"{year}-{month_num:02d}-{int(day):02d}"
            return PublicationDate(
                release_id=release_id,
                iso_date=iso_date,
                precision="day",
                raw_label=f"Updated {month_name.title()} {day}, {year}",
            )

    month_match = _find_month_precision(pdf_path)
    if month_match:
        month_name, year = month_match
        month_num = _MONTH_NAME_TO_NUM.get(month_name.lower())
        if month_num:
            release_id = f"{year}-{month_num:02d}"
            return PublicationDate(
                release_id=release_id,
                iso_date=None,
                precision="month",
                raw_label=f"Updated {month_name.title()} {year}",
            )

    if fallback_release_id:
        return PublicationDate(
            release_id=fallback_release_id,
            iso_date=None,
            precision="unknown",
            raw_label="",
        )
    raise ValueError(f"{pdf_path.name}: no 'Updated ...' date found and no fallback release_id given")


def _find_day_precision(pdf_path: Path) -> tuple[str, str, str] | None:
    try:
        import pdfplumber
    except ImportError:
        return None
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            match = _UPDATED_DAY_RE.search(text)
            if match:
                return match.group(1), match.group(2), match.group(3)
    return None


def _find_month_precision(pdf_path: Path) -> tuple[str, str] | None:
    payload = pdf_path.read_bytes()
    match = _UPDATED_MONTH_RE.search(payload.decode("latin-1", errors="ignore"))
    if match:
        return match.group(1), match.group(2)
    return None
