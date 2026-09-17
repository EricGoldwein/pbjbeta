"""Source and output locations for the SFF history build.

Source PDFs are never copied into this worktree — they are read in place
from the staged archive-audit extraction directory and (for the single
newest publication) from pbj-data-ops's own governed ACTIVE release, which
is read-only reference material for this worktree. All overridable via env
vars so tests can point at a small fixture directory instead.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

_MONTH_NAME_TO_NUM = {
    name: index
    for index, name in enumerate(
        (
            "january", "february", "march", "april", "may", "june", "july",
            "august", "september", "october", "november", "december",
        ),
        start=1,
    )
}

# The first Era-3b publication (first month every table carries a "Provider
# Number" / CCN column). Phase 1 scope starts here; nothing earlier is
# CCN-identifiable (SFF_LAYOUT_ANALYSIS.md Era 3b).
ERA_3B_START = (2023, 3)

_ARCHIVE_FILENAME_RE = re.compile(
    r"^SFF Posting with Candidate List\s*-\s*([A-Za-z]+)\s+(\d{4})\.pdf$", re.I
)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def archive_extracted_dir() -> Path:
    configured = os.environ.get("SFF_HISTORY_ARCHIVE_DIR")
    if configured:
        return Path(configured)
    return Path(r"D:\PBJapp-data\cms\sff\archive-audit\extracted")


def pbj_data_ops_root() -> Path:
    configured = os.environ.get("SFF_HISTORY_PBJ_DATA_OPS_ROOT")
    if configured:
        return Path(configured)
    return Path(r"C:\Users\egold\PycharmProjects\pbj-data-ops")


def output_dir() -> Path:
    configured = os.environ.get("SFF_HISTORY_OUTPUT_DIR")
    if configured:
        return Path(configured)
    return repo_root() / "data" / "sff_history"


def provider_info_normalized_csv(release_id: str) -> Path | None:
    """Best-effort path to pbj-data-ops's normalized Provider Info CSV for a
    YYYY-MM release id, used only for read-only cross-source reconciliation.
    Returns None if not staged locally (reconciliation is then skipped, not
    faked).
    """
    year, month = release_id.split("-", 1)
    candidate = pbj_data_ops_root() / "provider_info_normalized" / f"ProviderInfoNorm_{year}_{month}.csv"
    return candidate if candidate.is_file() else None


@dataclass(frozen=True)
class PublicationSource:
    release_id: str  # YYYY-MM, filename-derived (a fallback identity; the
    # authoritative publication_period comes from the in-PDF "Updated" date)
    pdf_path: Path
    source_kind: str  # "archive" | "governed_active"


def _archive_sources() -> dict[str, PublicationSource]:
    out: dict[str, PublicationSource] = {}
    directory = archive_extracted_dir()
    if not directory.is_dir():
        return out
    for path in directory.glob("*.pdf"):
        match = _ARCHIVE_FILENAME_RE.match(path.name)
        if not match:
            continue
        month_name, year = match.group(1).lower(), int(match.group(2))
        month_num = _MONTH_NAME_TO_NUM.get(month_name)
        if not month_num:
            continue
        if (year, month_num) < ERA_3B_START:
            continue  # Era 1/2/3a: no CCN, out of scope for this phase.
        release_id = f"{year}-{month_num:02d}"
        out[release_id] = PublicationSource(release_id=release_id, pdf_path=path, source_kind="archive")
    return out


def _local_file_uri(uri: str) -> Path:
    parsed = urlparse(uri)
    raw = unquote(parsed.path)
    if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return Path(raw)


def _governed_active_source() -> PublicationSource | None:
    """The current governed ACTIVE cms.sff_pdf_list release from pbj-data-ops,
    read strictly read-only via its own active-release registry file. This is
    what makes the newest in-scope publication authoritative even when it
    postdates the archive-audit's own PDF snapshot (e.g. archive tops out at
    July 2026; the governed release may already be August 2026).
    """
    registry_file = pbj_data_ops_root() / "state" / "active_releases.json"
    if not registry_file.is_file():
        return None
    try:
        payload = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    record = (payload.get("datasets") or {}).get("cms.sff_pdf_list")
    if not isinstance(record, dict) or record.get("status") != "ACTIVE":
        return None
    release_id = str(record.get("active_release_id") or "")
    metadata = record.get("metadata") or {}
    pdf_uri = str(metadata.get("source_pdf_uri") or "")
    if not release_id or not pdf_uri:
        return None
    pdf_path = _local_file_uri(pdf_uri)
    if not pdf_path.is_file():
        return None
    return PublicationSource(release_id=release_id, pdf_path=pdf_path, source_kind="governed_active")


def discover_publication_sources() -> list[PublicationSource]:
    """All Era-3b publication PDFs in scope, newest-governed-release wins on
    a release_id collision with the archive copy (should be byte-identical
    when both exist, per the audit's provenance check).
    """
    sources = _archive_sources()
    governed = _governed_active_source()
    if governed is not None:
        sources[governed.release_id] = governed
    return sorted(sources.values(), key=lambda s: s.release_id)
