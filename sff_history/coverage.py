"""Publication coverage / gap metadata.

CMS did not publish every month in this archive (ten known filename-derived
gaps across the full 2008-2026 span; five of them fall inside the Era-3b
window this phase covers: 2023-12, 2024-12, 2025-08, 2025-10, 2025-12 — see
SFF_ARCHIVE_AUDIT.md S4). A missing publication is not a continuous
observation, so downstream derivation must never silently bridge one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .publications import Publication


def _parse_release_id(release_id: str) -> tuple[int, int]:
    year, month = release_id.split("-", 1)
    return int(year), int(month)


def _next_month(year: int, month: int) -> tuple[int, int]:
    return (year + 1, 1) if month == 12 else (year, month + 1)


def _release_id(year: int, month: int) -> str:
    return f"{year}-{month:02d}"


@dataclass(frozen=True)
class CoverageRow:
    year_month: str
    present: bool
    publication_id: str | None
    validation_status: str | None
    gap_before: bool  # True if the immediately preceding calendar month has no PASS publication


def build_coverage(publications: list[Publication], *, start: tuple[int, int], end: tuple[int, int]) -> list[CoverageRow]:
    """One row per calendar month from ``start`` to ``end`` inclusive."""
    by_month = {p.publication_id: p for p in publications}
    rows: list[CoverageRow] = []
    year, month = start
    prev_present = False
    while (year, month) <= end:
        rid = _release_id(year, month)
        pub = by_month.get(rid)
        present = pub is not None
        rows.append(
            CoverageRow(
                year_month=rid,
                present=present,
                publication_id=rid if present else None,
                validation_status=pub.validation_status if pub else None,
                gap_before=not prev_present and bool(rows),  # first row never counts as a gap
            )
        )
        prev_present = present and pub.validation_status == "PASS"
        year, month = _next_month(year, month)
    return rows


def gapless_runs(publications: list[Publication]) -> list[list[Publication]]:
    """Split PASS-status publications into maximal runs of calendar-adjacent
    months. Only publications inside the same run may be compared for
    OBSERVED_CHANGE or chained into a DERIVED_INTERVAL.
    """
    passed = sorted((p for p in publications if p.validation_status == "PASS"), key=lambda p: p.publication_id)
    runs: list[list[Publication]] = []
    for pub in passed:
        year, month = _parse_release_id(pub.publication_id)
        if runs:
            last_year, last_month = _parse_release_id(runs[-1][-1].publication_id)
            if (year, month) == _next_month(last_year, last_month):
                runs[-1].append(pub)
                continue
        runs.append([pub])
    return runs


def coverage_to_rows(rows: list[CoverageRow]) -> list[dict[str, Any]]:
    return [
        {
            "year_month": row.year_month,
            "present": "Y" if row.present else "N",
            "publication_id": row.publication_id or "",
            "validation_status": row.validation_status or "",
            "gap_before": "Y" if row.gap_before else "N",
        }
        for row in rows
    ]
