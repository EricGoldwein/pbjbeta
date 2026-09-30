from __future__ import annotations

from pathlib import Path
from typing import Callable

from active_release_registry import load_registry, promote_release
from generic_cms_csv import CsvFeed, assess_feed, detect, run_feed
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


def _versioned_feed(tmp_path: Path, dataset_id: str = "cms.snf_all_owners") -> CsvFeed:
    return CsvFeed(
        dataset_id,
        "stable-product-id",
        r"SNF.*\.csv$",
        tmp_path / "downloads",
        (("ENROLLMENT ID",),),
        False,
        cms_product_path="/stable-product",
        cms_product_name="Stable Product",
    )


def _versioned_fetch(version_rows: list[dict], resources: dict[str, dict]) -> Callable[[str], dict]:
    newest_id = max(
        version_rows,
        key=lambda row: row["attributes"]["field_dataset_version"],
    )["id"]

    def fetch(url: str):
        if "/data-api/v1/slug" in url:
            return {
                "data": {
                    "uuid": "stable-product-id",
                    "current_dataset": {"uuid": newest_id},
                }
            }
        if "/jsonapi/node/dataset?" in url:
            return {"data": version_rows}
        for version_id, payload in resources.items():
            if f"/dataset/{version_id}/resources" in url:
                return payload
        raise AssertionError(f"unexpected URL: {url}")

    return fetch


def _version(version_id: str, label: str, modified: str | None = None) -> dict:
    return {
        "id": version_id,
        "attributes": {
            "field_dataset_version": label,
            "field_last_updated_date": modified or label,
            "field_re_release_version": None,
        },
    }


def _resource(filename: str, url: str, file_uuid: str = "file-1") -> dict:
    return {
        "data": [
            {
                "type": "Primary",
                "media_bundle": "primary_dataset_file",
                "file_name": filename,
                "file_url": url,
                "file_uuid": file_uuid,
            }
        ]
    }


def test_versioned_detection_selects_newest_version_not_first(tmp_path: Path) -> None:
    feed = _versioned_feed(tmp_path)
    rows = [_version("old-version", "2026-07-01"), _version("new-version", "2026-08-01")]
    fetch = _versioned_fetch(
        rows,
        {
            "old-version": _resource("SNF_Owners_2026.06.30.csv", "https://cms/old.csv"),
            "new-version": _resource("SNF_Owners_2026.07.31.csv", "https://cms/new.csv"),
        },
    )
    found = detect(feed, fetch_json=fetch)
    assert found["dataset_version_id"] == "new-version"
    assert found["url"] == "https://cms/new.csv"


def test_changed_version_uuid_is_discovered_dynamically(tmp_path: Path) -> None:
    feed = _versioned_feed(tmp_path)
    rows = [_version("future-version-uuid", "2026-09-01")]
    found = detect(
        feed,
        fetch_json=_versioned_fetch(
            rows,
            {"future-version-uuid": _resource("SNF_Owners_2026.08.31.csv", "https://cms/future.csv")},
        ),
    )
    assert found["product_id"] == "stable-product-id"
    assert found["dataset_version_id"] == "future-version-uuid"


def test_active_july_publisher_august_is_detected_not_current(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "state" / "active_releases.json"
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(registry))
    active = tmp_path / "active.csv"
    active.write_text("ENROLLMENT ID\n1\n", encoding="utf-8")
    promote_release(
        "cms.snf_all_owners",
        "2026-07-31",
        active,
        validated_at="now",
        metadata={"cms_publisher_url": "https://cms/july.csv"},
        path=registry,
    )
    feed = _versioned_feed(tmp_path)
    rows = [_version("august-version", "2026-09-01")]
    result = assess_feed(
        feed,
        root=tmp_path,
        fetch_json=_versioned_fetch(
            rows,
            {"august-version": _resource("SNF_Owners_2026.08.31.csv", "https://cms/august.csv")},
        ),
    )
    assert result["status"] == "DETECTED"
    assert result["new_release_available"] is True
    assert result["release_id"] == "2026-08-31"


def test_lookup_or_parse_failure_is_error_never_current(tmp_path: Path) -> None:
    result = assess_feed(
        _versioned_feed(tmp_path),
        root=tmp_path,
        fetch_json=lambda _url: (_ for _ in ()).throw(TimeoutError("CMS unavailable")),
    )
    assert result["status"] == "ERROR"
    assert result["new_release_available"] is None


def test_missing_active_publisher_provenance_is_unknown(tmp_path: Path, monkeypatch) -> None:
    registry = tmp_path / "state" / "active_releases.json"
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(registry))
    active = tmp_path / "active.csv"
    active.write_text("ENROLLMENT ID\n1\n", encoding="utf-8")
    promote_release(
        "cms.snf_all_owners",
        "2026-07-31",
        active,
        validated_at="now",
        path=registry,
    )
    feed = _versioned_feed(tmp_path)
    rows = [_version("current-version", "2026-08-01")]
    result = assess_feed(
        feed,
        root=tmp_path,
        fetch_json=_versioned_fetch(
            rows,
            {"current-version": _resource("SNF_Owners_2026.07.31.csv", "https://cms/current.csv")},
        ),
    )
    assert result["status"] == "UNKNOWN"
    assert result["new_release_available"] is None


def test_next_month_metadata_requires_no_code_change(tmp_path: Path) -> None:
    feed = _versioned_feed(tmp_path)
    for version_id, label, filename in (
        ("september-uuid", "2026-09-01", "SNF_Owners_2026.08.31.csv"),
        ("october-uuid", "2026-10-01", "SNF_Owners_2026.09.30.csv"),
    ):
        found = detect(
            feed,
            fetch_json=_versioned_fetch(
                [_version(version_id, label)],
                {version_id: _resource(filename, f"https://cms/{filename}")},
            ),
        )
        assert found["dataset_version_id"] == version_id
        assert found["filename"] == filename


def test_both_ownership_sources_use_dynamic_product_resolution(tmp_path: Path) -> None:
    from release_check import ownership_csv_feeds

    feeds = ownership_csv_feeds(tmp_path)
    assert set(feeds) == {"cms.snf_all_owners", "cms.snf_enrollments"}
    assert all(feed.cms_product_path for feed in feeds.values())
    assert all(feed.cms_product_name for feed in feeds.values())


def test_same_release_changed_artifact_is_revised_and_acquired_without_overwrite(
    tmp_path: Path, monkeypatch
) -> None:
    registry = tmp_path / "state" / "active_releases.json"
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(registry))
    download_dir = tmp_path / "downloads"
    download_dir.mkdir()
    old = download_dir / "SNF_All_Owners_2026.07.31.csv"
    old.write_text("ENROLLMENT ID\nold\n", encoding="utf-8")
    promote_release(
        "cms.snf_all_owners",
        "2026-07-31",
        old,
        validated_at="now",
        metadata={
            "cms_publisher_url": "https://cms/owners-old.csv",
            "cms_file_uuid": "old-file",
            "cms_dataset_version_id": "old-version",
        },
        path=registry,
    )
    feed = _versioned_feed(tmp_path)
    rows = [_version("revised-version", "2026-08-02", "2026-09-29")]
    fetch = _versioned_fetch(
        rows,
        {
            "revised-version": _resource(
                "SNF_All_Owners_2026.07.31_update.csv",
                "https://cms/owners-update.csv",
                "revised-file",
            )
        },
    )
    assessment = assess_feed(feed, root=tmp_path, fetch_json=fetch)
    assert assessment["status"] == "REVISED"
    assert set(assessment["revision_identity_changes"]) == {
        "publisher_url",
        "file_uuid",
        "version_uuid",
    }

    result = run_feed(
        feed,
        True,
        root=tmp_path,
        fetch_json=fetch,
        fetch_bytes=lambda _url: b"ENROLLMENT ID\nnew\nnewer\n",
    )
    assert result["status"] == "ACQUIRED"
    assert old.read_text(encoding="utf-8") == "ENROLLMENT ID\nold\n"
    revised = download_dir / "SNF_All_Owners_2026.07.31_update.csv"
    assert revised.is_file()
    candidate = __import__("release_control_plane").load_candidates(tmp_path)["datasets"][feed.dataset_id]
    assert candidate["release_id"] == "2026-07-31"
    assert candidate["metadata"]["change_kind"] == "REVISED"
    assert candidate["validation"]["row_count"] == 2
    assert load_registry(registry)["datasets"][feed.dataset_id]["hash"] != candidate["hash"]


def test_individual_acquire_allows_only_proven_same_release_revision(tmp_path: Path, monkeypatch) -> None:
    from release_check import acquire_detected_source
    from release_control_plane import ReleaseState, record_candidate

    registry = tmp_path / "state" / "active_releases.json"
    monkeypatch.setenv("PBJ_ACTIVE_RELEASE_REGISTRY", str(registry))
    old = tmp_path / "old.csv"
    old.write_text("ENROLLMENT ID\nold\n", encoding="utf-8")
    promote_release(
        "cms.snf_all_owners",
        "2026-07-31",
        old,
        validated_at="now",
        metadata={"cms_publisher_url": "https://cms/old.csv"},
        path=registry,
    )
    record_candidate(
        "cms.snf_all_owners",
        "2026-07-31",
        ReleaseState.DETECTED,
        metadata={
            "publisher_revision_changed": True,
            "cms_publisher_url": "https://cms/update.csv",
        },
        root=tmp_path,
    )
    monkeypatch.setattr(
        "release_check.production_handlers",
        lambda: {"cms.snf_all_owners": lambda acquire: {"status": "ACQUIRED", "called": acquire}},
    )
    result = acquire_detected_source("cms.snf_all_owners", root=tmp_path)
    assert result["called"] is True
