from __future__ import annotations

from pathlib import Path

from active_release_registry import load_registry, promote_release
from generic_cms_csv import CsvFeed, run_feed
from release_control_plane import promote_candidate


def test_detect_validate_and_explicitly_promote(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "state" / "active_releases.json"
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(registry))
    old = tmp_path / "old.csv"
    old.write_text("PROVNUM,WorkDate\n1,2026-01-01\n", encoding="utf-8")
    promote_release("cms.pbj_non_nurse_staffing", "CY2026Q1", old, validated_at="now", path=registry)
    feed = CsvFeed("cms.pbj_non_nurse_staffing", "fixture", r"PBJ_dailynonnursestaffing_CY\d{4}Q[1-4]\.csv$", tmp_path / "downloads", (("PROVNUM",), ("WorkDate",)), True)
    meta = {"data": [{"type": "Primary", "file_name": "PBJ_dailynonnursestaffing_CY2026Q2.csv", "file_url": "https://fixture/new.csv"}]}
    result = run_feed(feed, True, root=tmp_path, fetch_json=lambda _: meta, fetch_bytes=lambda _: b"PROVNUM,WorkDate\n1,2026-04-01\n")
    assert result["status"] == "VALIDATED"
    assert load_registry(registry)["datasets"][feed.dataset_id]["active_release_id"] == "CY2026Q1"
    promote_candidate(feed.dataset_id, root=tmp_path)
    assert load_registry(registry)["datasets"][feed.dataset_id]["active_release_id"] == "CY2026Q2"


def test_nurse_current_check_includes_publisher_release_id(monkeypatch) -> None:
    from release_check import production_handlers

    monkeypatch.setattr(
        "cms_data_ops.check_nurse_cms",
        lambda **_: {
            "cms": {"quarter_label": "CY2026Q1"},
            "cms_is_newer": False,
        },
    )
    monkeypatch.setattr(
        "active_release_registry.load_registry",
        lambda _path=None: {"datasets": {"cms.pbj_nurse_staffing": {"active_release_id": "CY2026Q1"}}},
    )
    result = production_handlers()["cms.pbj_nurse_staffing"](False)
    assert result["status"] == "CURRENT"
    assert result["release_id"] == "CY2026Q1"
    assert result["publisher_latest_release_id"] == "CY2026Q1"


def test_schema_failure_preserves_active(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "state" / "active_releases.json"
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(registry))
    old = tmp_path / "old.csv"
    old.write_text("PROVNUM,WorkDate\n1,2026-01-01\n", encoding="utf-8")
    promote_release("cms.pbj_non_nurse_staffing", "CY2026Q1", old, validated_at="now", path=registry)
    feed = CsvFeed("cms.pbj_non_nurse_staffing", "fixture", r"new\.csv$", tmp_path / "downloads", (("PROVNUM",),), True)
    meta = {"data": [{"type": "Primary", "file_name": "new.csv", "file_url": "https://fixture/new.csv", "file_uuid": "release-2"}]}
    import pytest
    with pytest.raises(RuntimeError, match="schema validation failed"):
        run_feed(feed, True, root=tmp_path, fetch_json=lambda _: meta, fetch_bytes=lambda _: b"wrong\n1\n")
    assert load_registry(registry)["datasets"][feed.dataset_id]["active_release_id"] == "CY2026Q1"
