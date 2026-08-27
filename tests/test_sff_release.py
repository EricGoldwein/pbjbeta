import csv
import io
import json
from pathlib import Path

import pytest

from manual_staging import MACPAC, stage_upload
from release_control_plane import DEPENDENCY_GRAPH, load_candidates, what_would_change
from sff_release import _rows_from_pdf, validate_rows


ROOT = Path(__file__).resolve().parents[1]
ACTUAL_PDF = ROOT / "sff" / "releases" / "2026-08" / "cms_sff_posting_2026-08.pdf"


def test_actual_august_structure_and_categories():
    rows = _rows_from_pdf(ACTUAL_PDF)
    result = validate_rows(rows, provider_ccns={row["ccn"] for row in rows})
    assert result["status"] == "PASS"
    assert result["category_counts"] == {"CURRENT_SFF": 87, "GRADUATED": 126, "NO_LONGER_PARTICIPATING": 1, "SFF_CANDIDATE": 441}
    assert any("E" in row["ccn"] for row in rows)
    assert all(row["source_page"].isdigit() for row in rows)


def test_candidate_and_current_sff_are_distinct():
    rows = _rows_from_pdf(ACTUAL_PDF)
    assert {row["category"] for row in rows} >= {"CURRENT_SFF", "SFF_CANDIDATE"}
    assert all(row["category"] != "CURRENT_SFF" for row in rows if row["source_table"] == "Table D")


def test_sff_staleness_scope_isolated():
    impact = what_would_change("cms.sff_pdf_list")
    assert impact["would_mark_stale"] == ["facility.sff_status"]
    assert "facility.staffing" in impact["would_remain_current"]
    assert "facility.snf_owners" in impact["would_remain_current"]


def test_manual_upload_validates_but_never_promotes(tmp_path):
    (tmp_path / "state").mkdir()
    payload = "State,Total_Estimated_Staffing_Requirements,Min_Staffing,Max_Staffing,Value_Type,Is_Federal_Minimum,Display_Text\nNY,3.5,3,4,hprd,N,Example\n"
    result = stage_upload(MACPAC, "test-v1", "standards.csv", io.BytesIO(payload.encode()), root=tmp_path)
    assert result["validation"]["status"] == "PASS"
    assert load_candidates(tmp_path)["datasets"][MACPAC]["state"] == "VALIDATED"
    assert not (tmp_path / "state" / "active_releases.json").exists()


def test_upload_rejects_unapproved_dataset(tmp_path):
    with pytest.raises(ValueError, match="not approved"):
        stage_upload("arbitrary.dataset", "v1", "data.csv", io.BytesIO(b"x\n1\n"), root=tmp_path)
