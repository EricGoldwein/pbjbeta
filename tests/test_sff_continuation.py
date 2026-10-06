import csv
from pathlib import Path
from collections import Counter

import pdfplumber
import pytest

import sff_release as sff

ROOT = Path(__file__).resolve().parents[1]
SEPTEMBER = ROOT / "sff/releases/2026-09/cms_sff_posting_2026-09.pdf"
COUNTS = {"CURRENT_SFF": 87, "GRADUATED": 125, "NO_LONGER_PARTICIPATING": 2, "SFF_CANDIDATE": 440}
ALPHA = {"15E064", "17E210", "04E262", "17E531", "24E185", "24E507", "34A002", "43A138", "51E148"}


def hide_title(monkeypatch, page_number, *, unrelated=False):
    original = pdfplumber.open

    class Table:
        def __init__(self, table): self.table = table
        def __getattr__(self, name): return getattr(self.table, name)
        def extract(self):
            cells = self.table.extract()
            cells[0][0] = "Continuation" if not unrelated else "Budget summary"
            if unrelated: cells[1][0] = "Account number"
            return cells

    class Page:
        def __init__(self, page): self.page = page
        def __getattr__(self, name): return getattr(self.page, name)
        def find_tables(self): return [Table(table) for table in self.page.find_tables()]

    class PDF:
        def __init__(self, path):
            self.pdf = original(path)
            self.pages = list(self.pdf.pages)
            self.pages[page_number - 1] = Page(self.pages[page_number - 1])
        def __enter__(self): return self
        def __exit__(self, *args): self.pdf.close()

    monkeypatch.setattr(pdfplumber, "open", PDF)


@pytest.mark.parametrize("page,count,first,last", [(5, 38, "315125", "535022"), (11, 50, "075228", "165580")])
def test_september_untitled_continuation_preserves_page_rows(monkeypatch, page, count, first, last):
    hide_title(monkeypatch, page)
    rows = sff._rows_from_pdf(SEPTEMBER)
    assert len(rows) == 654
    assert Counter(row["category"] for row in rows) == COUNTS
    page_rows = [row for row in rows if row["source_page"] == str(page)]
    assert len(page_rows) == count
    assert (page_rows[0]["ccn"], page_rows[-1]["ccn"]) == (first, last)
    assert {row["ccn"] for row in rows if any(c.isalpha() for c in row["ccn"])} == ALPHA


def test_unrelated_table_after_sff_context_fails_explicitly(monkeypatch):
    hide_title(monkeypatch, 5, unrelated=True)
    with pytest.raises(RuntimeError, match="Ambiguous SFF table schema/geometry on page 5"):
        sff._rows_from_pdf(SEPTEMBER)


def test_real_september_matches_existing_candidate():
    rows = sff._rows_from_pdf(SEPTEMBER)
    with SEPTEMBER.with_suffix(".csv").open(encoding="utf-8-sig", newline="") as handle:
        assert rows == list(csv.DictReader(handle))
    assert sff.validate_rows(rows)["category_counts"] == COUNTS
    assert len({row["ccn"] for row in rows}) == 639


def test_older_active_cannot_complete_pending_sff_review_or_promotion(monkeypatch):
    from cms_data_ops import build_sff_lifecycle_steps, build_source_operator_workflow
    control = {"active": {"active_release_id": "2026-08", "validation": {"status": "PASS"}},
               "pending": {"release_id": "2026-09", "state": "VALIDATED", "validation": {"status": "PASS"}}}
    steps = {step["id"]: step for step in build_sff_lifecycle_steps(control_row=control)}
    assert steps["review"]["state"] == "current"
    assert steps["make_active"]["state"] == "upcoming"
    assert all(steps[key]["state"] == "completed" for key in ("check_cms", "acquire_pdf", "validate"))
    workflow = build_source_operator_workflow("cms.sff_pdf_list", control_row=control, record=None, snapshot=None)
    assert workflow["next_action"]["endpoint"] == "release_review"
    from data_ops_app import create_app
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-password")
    app = create_app()
    with app.test_request_context():
        html = app.jinja_env.get_template("data_ops/partials/source_detail_panel.html").render(
            record={"source_id": "cms.sff_pdf_list"}, workflow=workflow, control_row=None, snapshot=None)
    review = html.split('do-lifecycle-current')[1].split('</li>')[0]
    assert "Review" in review and "READY / NEXT" in review and "completed" not in review
    promote = html.split('do-lifecycle-upcoming')[1].split('</li>')[0]
    assert "Make ACTIVE" in promote and "NOT YET" in promote and "completed" not in promote
