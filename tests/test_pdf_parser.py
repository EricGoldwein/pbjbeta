"""Tests against real archive PDFs (read-only; never copied into this
worktree, per the task's explicit constraint — referenced in place under
D:\\PBJapp-data\\cms\\sff\\archive-audit\\extracted).
"""

from pathlib import Path

import pytest

from sff_history.pdf_parser import rows_from_pdf
from sff_history.schema import CCN_RE

ARCHIVE_DIR = Path(r"D:\PBJapp-data\cms\sff\archive-audit\extracted")
JULY_2026 = ARCHIVE_DIR / "SFF Posting with Candidate List -  July 2026.pdf"

pytestmark = pytest.mark.skipif(not JULY_2026.is_file(), reason="archive PDF not staged locally")


def test_all_ccns_match_expected_shape():
    rows = rows_from_pdf(JULY_2026)
    assert rows
    assert all(CCN_RE.fullmatch(row["ccn"]) for row in rows)


def test_alphanumeric_ccn_present_and_unmodified():
    rows = rows_from_pdf(JULY_2026)
    alpha_ccns = {row["ccn"] for row in rows if any(ch.isalpha() for ch in row["ccn"])}
    # Confirmed present in this file by the archive audit (SFF_CURRENT_RECONCILIATION.md S2).
    assert "15E064" in alpha_ccns


def test_current_sff_and_candidate_are_distinct_categories():
    rows = rows_from_pdf(JULY_2026)
    categories = {row["category"] for row in rows}
    assert {"CURRENT_SFF", "SFF_CANDIDATE"} <= categories
    assert all(row["category"] != "CURRENT_SFF" for row in rows if row["source_table"] == "Table D")


def test_matches_governed_parser_row_count_one_month_back():
    # The archive audit independently re-verified this file against
    # pbj-data-ops's governed parser: 648 rows, 637 unique CCNs
    # (SFF_CURRENT_RECONCILIATION.md S1). This re-parse (a fresh
    # implementation of the same algorithm) must reproduce it exactly.
    rows = rows_from_pdf(JULY_2026)
    assert len(rows) == 648
    assert len({row["ccn"] for row in rows}) == 637
