"""Era-3b (March 2023+) CMS SFF posting PDF row extraction.

This is a from-scratch re-implementation of the CCN-anchor word-geometry
algorithm in ``pbj-data-ops/sff_release.py:_rows_from_pdf`` — not an import.
pbj-data-ops is a separate, governed repository that this worktree must not
modify or depend on at runtime, so the algorithm is deliberately reproduced
here rather than reused as a library call. The row shape and column mapping
match that module's output exactly so a diff against the governed release is
meaningful.

CMS supplies vertical column rules in these PDFs but no horizontal row rules.
Provider numbers (CCNs) are the only stable per-row anchor: this parser finds
every CCN-shaped word on a table page, then buckets every other word on that
same horizontal band into the table's real columns using the table's column
left-edges. Line-based text extraction (``pdftotext -layout``) was proven
unreliable for this archive (SFF_LAYOUT_ANALYSIS.md methodology note): a
physically 2-3-row stacked header can split a value across rows, and naive
row-order text can mispair a name with the wrong row's address.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .schema import CATEGORIES, CCN_RE, PARSER_VERSION


class SffPdfParseError(RuntimeError):
    """Raised when a CMS SFF posting PDF does not match the expected Era-3b geometry."""


def rows_from_pdf(pdf_path: Path) -> list[dict[str, str]]:
    """Extract one row per CCN x table-membership from a single Era-3b PDF.

    Raises SffPdfParseError if no governed table is found on any page, or if
    a table's column geometry does not match the expected Era-3b shape (>=8
    columns). Both are treated as hard failures at the publication level —
    silently dropping rows on a geometry change would be worse than failing
    loudly.
    """
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover - environment issue, not a code path
        raise RuntimeError("pdfplumber is required to parse CMS SFF postings") from exc

    rows: list[dict[str, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            tables = page.find_tables()
            if not tables:
                continue
            table = tables[0]
            extracted = table.extract()
            title = str((extracted[0] or [""])[0] or "")
            table_key = next((key for key in CATEGORIES if title.startswith(key)), None)
            if not table_key:
                continue
            # The title row spans the first logical column, so pdfplumber
            # reports that column as the whole table; the remaining column
            # left edges are accurate. Prepend the table's own left edge to
            # recover the provider-number column boundary.
            bounds = [table.bbox[0]] + [col.bbox[0] for col in table.columns[1:]] + [table.bbox[2]]
            words = page.extract_words(x_tolerance=1, y_tolerance=1)
            anchors = [
                word
                for word in words
                if CCN_RE.fullmatch(word["text"].upper())
                and any(ch.isdigit() for ch in word["text"])
                and bounds[0] - 2 <= word["x0"] < bounds[1]
                and word["top"] > 75
            ]
            for anchor in anchors:
                cells: list[list[str]] = [[] for _ in range(len(bounds) - 1)]
                center_y = (anchor["top"] + anchor["bottom"]) / 2
                for word in words:
                    word_y = (word["top"] + word["bottom"]) / 2
                    if abs(word_y - center_y) > 2.2:
                        continue
                    word_x = (word["x0"] + word["x1"]) / 2
                    for index in range(len(bounds) - 1):
                        if bounds[index] <= word_x < bounds[index + 1]:
                            cells[index].append(word["text"])
                            break
                values = [" ".join(parts).strip() for parts in cells]
                if len(values) < 8:
                    raise SffPdfParseError(
                        f"{pdf_path.name}: SFF table geometry changed on page {page_number} "
                        f"({table_key}, {len(values)} columns found, expected >= 8)"
                    )
                row = {
                    "ccn": values[0].upper(),
                    "facility_name": values[1],
                    "address": values[2],
                    "city": values[3],
                    "state": values[4].upper(),
                    "zip": values[5],
                    "phone": values[6],
                    "category": CATEGORIES[table_key],
                    "source_table": table_key,
                    "source_page": str(page_number),
                    "status_date": "",
                    "survey_criteria": "",
                    "months_in_status": "",
                }
                if table_key == "Table A":
                    row["status_date"], row["survey_criteria"], row["months_in_status"] = (
                        values[7],
                        values[8],
                        values[9],
                    )
                elif table_key in {"Table B", "Table C"}:
                    row["status_date"], row["months_in_status"] = values[7], values[8]
                else:
                    row["months_in_status"] = values[7]
                rows.append(row)
    if not rows:
        raise SffPdfParseError(f"{pdf_path.name}: no governed SFF tables were parsed")
    return rows


def parser_version() -> str:
    return PARSER_VERSION


def _self_test_shape(rows: list[dict[str, Any]]) -> None:
    """Sanity check used by tests/build: every row has the expected keys."""
    expected_keys = {
        "ccn",
        "facility_name",
        "address",
        "city",
        "state",
        "zip",
        "phone",
        "category",
        "source_table",
        "source_page",
        "status_date",
        "survey_criteria",
        "months_in_status",
    }
    for row in rows:
        missing = expected_keys - row.keys()
        if missing:
            raise SffPdfParseError(f"row missing expected fields: {sorted(missing)}")
