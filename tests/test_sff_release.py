import csv
import io
import json
from pathlib import Path

import pytest

from active_release_registry import load_registry
from manual_staging import MACPAC, stage_upload
from release_control_plane import DEPENDENCY_GRAPH, load_candidates, what_would_change
from sff_release import (
    DATASET_ID,
    OFFICIAL_AUGUST_2026_URL,
    _rows_from_pdf,
    check_sff_cms,
    discover_latest_cms_sff_posting,
    parse_sff_posting_updated_label,
    stage_detected_candidate,
    validate_rows,
)


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


def test_parse_sff_posting_updated_label_from_local_pdf():
    parsed = parse_sff_posting_updated_label(ACTUAL_PDF.read_bytes())
    assert parsed is not None
    assert parsed[0] == "2026-08"


def test_discover_latest_cms_sff_posting_uses_updated_label(monkeypatch):
    payload = ACTUAL_PDF.read_bytes()

    def fake_fetch(url: str) -> bytes:
        if url == OFFICIAL_AUGUST_2026_URL:
            return payload
        return b""

    discovered = discover_latest_cms_sff_posting(
        fetch_bytes=fake_fetch,
        months_back=3,
        anchor=__import__("datetime").datetime(2026, 8, 15, tzinfo=__import__("datetime").timezone.utc),
    )
    assert discovered["release_id"] == "2026-08"
    assert discovered["source_url"] == OFFICIAL_AUGUST_2026_URL


def test_check_sff_cms_current_when_active_matches(tmp_path, monkeypatch):
    from active_release_registry import load_registry

    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    active = {
        "schema_version": 1,
        "datasets": {
            "cms.sff_pdf_list": {
                "dataset_id": "cms.sff_pdf_list",
                "active_release_id": "2026-08",
                "status": "ACTIVE",
            }
        },
    }
    (state / "active_releases.json").write_text(__import__("json").dumps(active), encoding="utf-8")
    (state / "release_candidates.json").write_text('{"schema_version":1,"datasets":{}}', encoding="utf-8")
    monkeypatch.setattr("sff_release.registry_path", lambda _root=None: state / "active_releases.json")
    payload = ACTUAL_PDF.read_bytes()

    def fake_fetch(url: str) -> bytes:
        if url == OFFICIAL_AUGUST_2026_URL:
            return payload
        return b""

    result = check_sff_cms(fetch_bytes=fake_fetch, root=tmp_path)
    assert result["cms_is_newer"] is False
    assert result["cms"]["release_id"] == "2026-08"


def test_check_sff_cms_records_detected_candidate_with_trusted_url(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    (state / "active_releases.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    DATASET_ID: {
                        "dataset_id": DATASET_ID,
                        "active_release_id": "2026-08",
                        "status": "ACTIVE",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (state / "release_candidates.json").write_text(
        '{"schema_version":1,"datasets":{}}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "sff_release.registry_path", lambda _root=None: state / "active_releases.json"
    )
    september_url = OFFICIAL_AUGUST_2026_URL.replace("august", "september")
    monkeypatch.setattr(
        "sff_release.discover_latest_cms_sff_posting",
        lambda **_kwargs: {
            "release_id": "2026-09",
            "posting_label": "September 2026",
            "source_url": september_url,
            "url_release_id": "2026-09",
            "candidates": [{"release_id": "2026-09"}],
        },
    )

    result = check_sff_cms(root=tmp_path)

    assert result["cms_is_newer"] is True
    candidate = load_candidates(tmp_path)["datasets"][DATASET_ID]
    assert candidate["release_id"] == "2026-09"
    assert candidate["state"] == "DETECTED"
    assert candidate["metadata"]["source_url"] == september_url


def test_sff_diagnostic_probe_does_not_create_candidate(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    (state / "active_releases.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "datasets": {
                    DATASET_ID: {
                        "dataset_id": DATASET_ID,
                        "active_release_id": "2026-08",
                        "status": "ACTIVE",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (state / "release_candidates.json").write_text(
        '{"schema_version":1,"datasets":{}}', encoding="utf-8"
    )
    monkeypatch.setattr(
        "sff_release.registry_path", lambda _root=None: state / "active_releases.json"
    )
    monkeypatch.setattr(
        "sff_release.discover_latest_cms_sff_posting",
        lambda **_kwargs: {
            "release_id": "2026-09",
            "posting_label": "September 2026",
            "source_url": OFFICIAL_AUGUST_2026_URL.replace("august", "september"),
            "url_release_id": "2026-09",
            "candidates": [{"release_id": "2026-09"}],
        },
    )

    result = check_sff_cms(root=tmp_path, record_detection=False)

    assert result["cms_is_newer"] is True
    assert load_candidates(tmp_path)["datasets"] == {}


def test_stage_detected_sff_uses_recorded_cms_url_and_validates(tmp_path):
    from release_control_plane import ReleaseState, record_candidate

    record_candidate(
        DATASET_ID,
        "2026-08",
        ReleaseState.DETECTED,
        metadata={"source_url": OFFICIAL_AUGUST_2026_URL},
        root=tmp_path,
    )
    payload = ACTUAL_PDF.read_bytes()
    result = stage_detected_candidate(root=tmp_path, fetch_bytes=lambda _url: payload)

    assert result["release_id"] == "2026-08"
    assert result["validation"]["status"] == "PASS"
    assert len(result["pbj_handoff"]) == 4
    candidate = load_candidates(tmp_path)["datasets"][DATASET_ID]
    assert candidate["state"] == "VALIDATED"
    assert candidate["metadata"]["source_url"] == OFFICIAL_AUGUST_2026_URL


def test_stage_detected_sff_rejects_embedded_month_mismatch(tmp_path):
    from release_control_plane import ReleaseState, record_candidate

    september_url = OFFICIAL_AUGUST_2026_URL.replace("august", "september")
    record_candidate(
        DATASET_ID,
        "2026-09",
        ReleaseState.DETECTED,
        metadata={"source_url": september_url},
        root=tmp_path,
    )
    with pytest.raises(RuntimeError, match="identity 2026-08 does not match candidate 2026-09"):
        stage_detected_candidate(
            root=tmp_path,
            fetch_bytes=lambda _url: ACTUAL_PDF.read_bytes(),
        )
    assert load_candidates(tmp_path)["datasets"][DATASET_ID]["state"] == "DETECTED"


def test_manual_upload_validates_but_never_promotes(tmp_path):
    payload = "State,Total_Estimated_Staffing_Requirements,Min_Staffing,Max_Staffing,Value_Type,Is_Federal_Minimum,Display_Text\nNY,3.5,3,4,hprd,N,Example\n"
    result = stage_upload(MACPAC, "test-v1", "standards.csv", io.BytesIO(payload.encode()), root=tmp_path)
    assert result["validation"]["status"] == "PASS"
    candidate = load_candidates(tmp_path)["datasets"][MACPAC]
    assert candidate["state"] == "VALIDATED"
    assert candidate["state"] != "ACTIVE"
    assert MACPAC not in load_registry(root=tmp_path).get("datasets", {})


def test_upload_rejects_unapproved_dataset(tmp_path):
    with pytest.raises(ValueError, match="not approved"):
        stage_upload("arbitrary.dataset", "v1", "data.csv", io.BytesIO(b"x\n1\n"), root=tmp_path)
