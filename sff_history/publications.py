"""Layer 1: publication records.

One immutable row per CMS SFF posting. Never edited after creation — this is
the raw-evidence anchor everything else (observations, derived facts)
references by ``publication_id``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import PublicationSource
from .pdf_parser import SffPdfParseError, rows_from_pdf
from .publication_date import extract_publication_date
from .schema import ERA_3B, PARSER_VERSION

PUBLICATION_FIELDS = [
    "publication_id",
    "publication_period",
    "updated_date",
    "updated_date_precision",
    "updated_label_raw",
    "updated_date_period_mismatch",
    "source_filename",
    "source_kind",
    "sha256",
    "era_id",
    "parser_version",
    "page_count",
    "row_count",
    "validation_status",
    "validation_errors",
    "validation_warnings",
    "built_at",
]


@dataclass
class Publication:
    publication_id: str
    publication_period: str
    updated_date: str | None
    updated_date_precision: str
    updated_label_raw: str
    updated_date_period_mismatch: bool
    source_filename: str
    source_kind: str
    sha256: str
    era_id: str
    parser_version: str
    page_count: int
    row_count: int
    validation_status: str  # PASS | FAIL
    validation_errors: list[str] = field(default_factory=list)
    validation_warnings: list[str] = field(default_factory=list)
    built_at: str = ""

    def to_row(self) -> dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "publication_period": self.publication_period,
            "updated_date": self.updated_date or "",
            "updated_date_precision": self.updated_date_precision,
            "updated_label_raw": self.updated_label_raw,
            "updated_date_period_mismatch": "Y" if self.updated_date_period_mismatch else "N",
            "source_filename": self.source_filename,
            "source_kind": self.source_kind,
            "sha256": self.sha256,
            "era_id": self.era_id,
            "parser_version": self.parser_version,
            "page_count": self.page_count,
            "row_count": self.row_count,
            "validation_status": self.validation_status,
            "validation_errors": ";".join(self.validation_errors),
            "validation_warnings": ";".join(self.validation_warnings),
            "built_at": self.built_at,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _page_count(pdf_path: Path) -> int:
    import pdfplumber

    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


def build_publication(source: PublicationSource) -> tuple[Publication, list[dict[str, str]]]:
    """Parse one PDF and build its Layer-1 publication record plus its raw
    parsed rows (Layer-2 material; not yet observation records).

    ``publication_id``/``publication_period`` are always the filename- (or
    governed-release-) derived YYYY-MM from ``source.release_id`` — never the
    in-PDF "Updated" date. Those two are related but genuinely distinct: one
    verified example in this archive (the file named/titled "November 2023")
    carries an internal body footer reading "Updated December 6, 2023" —
    CMS's own processing date for that posting, not its nominal period. Only
    ``updated_date`` reflects that extracted value; conflating the two would
    silently relabel a publication under the wrong period.

    Never raises on a parse failure — a bad/unexpected-geometry PDF becomes a
    FAIL-status publication with zero rows and an explicit error, so one bad
    file cannot block ingest of the rest of the archive. The caller decides
    whether to build observations for a FAIL publication (it should not).
    """
    built_at = datetime.now(timezone.utc).isoformat()
    sha = _sha256_file(source.pdf_path)
    pub_date = extract_publication_date(source.pdf_path, fallback_release_id=source.release_id)
    date_mismatch = pub_date.precision != "unknown" and pub_date.release_id != source.release_id
    try:
        page_count = _page_count(source.pdf_path)
    except Exception as exc:  # pragma: no cover - defensive; pdfplumber open already succeeded once
        page_count = 0
        rows: list[dict[str, str]] = []
        publication = Publication(
            publication_id=source.release_id,
            publication_period=source.release_id,
            updated_date=pub_date.iso_date,
            updated_date_precision=pub_date.precision,
            updated_label_raw=pub_date.raw_label,
            updated_date_period_mismatch=date_mismatch,
            source_filename=source.pdf_path.name,
            source_kind=source.source_kind,
            sha256=sha,
            era_id=ERA_3B,
            parser_version=PARSER_VERSION,
            page_count=page_count,
            row_count=0,
            validation_status="FAIL",
            validation_errors=[f"could not open PDF: {exc}"],
            built_at=built_at,
        )
        return publication, rows

    try:
        rows = rows_from_pdf(source.pdf_path)
        errors: list[str] = []
        status = "PASS"
    except SffPdfParseError as exc:
        rows = []
        errors = [str(exc)]
        status = "FAIL"

    publication = Publication(
        publication_id=source.release_id,
        publication_period=source.release_id,
        updated_date=pub_date.iso_date,
        updated_date_precision=pub_date.precision,
        updated_label_raw=pub_date.raw_label,
        updated_date_period_mismatch=date_mismatch,
        source_filename=source.pdf_path.name,
        source_kind=source.source_kind,
        sha256=sha,
        era_id=ERA_3B,
        parser_version=PARSER_VERSION,
        page_count=page_count,
        row_count=len(rows),
        validation_status=status,
        validation_errors=errors,
        built_at=built_at,
    )
    return publication, rows
