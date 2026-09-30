from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

import cms_nh_catalog as catalog

NOW = datetime(2026, 9, 30, 11, 0, tzinfo=ZoneInfo("America/New_York"))


def dataset(identifier="y2hd-n93e", *, month="08", version="100", planned="2026-09-30", title="Ownership"):
    url = catalog.BASE + f"/sites/default/files/resources/resource-{identifier}_{version}/NH_Example_{month}2026.csv"
    return {"identifier": identifier, "title": title, "description": "CMS evidence", "theme": [catalog.THEME],
            "modified": f"2026-{month}-01", "released": f"2026-{month}-26", "nextUpdateDate": planned,
            "distribution": [{"identifier": "dist-" + identifier, "data": {"downloadURL": url, "mediaType": "text/csv",
                "%Ref:downloadURL": [{"data": {"identifier": "resource-" + identifier, "version": version, "perspective": "source", "checksum": None}}]}}]}


class CMS:
    def __init__(self, rows=None):
        self.rows = rows or [dataset()]
        self.fail = set()
        self.archive_date = "2026-08-26"
        self.urls = []

    def fetch(self, url):
        self.urls.append(url)
        if url in self.fail or "all" in self.fail:
            raise OSError("CMS unavailable")
        if url.startswith(catalog.SEARCH_URL):
            return {"total": len(self.rows), "results": {row["identifier"]: copy.deepcopy(row) for row in self.rows}}
        if url.startswith(catalog.METASTORE):
            identifier = url[len(catalog.METASTORE):].split("?")[0]
            return copy.deepcopy(next(row for row in self.rows if row["identifier"] == identifier))
        if url == catalog.CURRENT_ARCHIVES_URL:
            return {"data": [{"id": "6281", "type": "current", "theme": "nursing-homes", "date": self.archive_date,
                             "size": "39000", "url": catalog.CURRENT_ZIP_URL}]}
        if url == catalog.ARCHIVES_URL:
            return {"data": [{"id": "9266", "type": "theme", "date": self.archive_date, "url": "archive.zip"}]}
        raise AssertionError("Unexpected acquisition/network request: " + url)

    def head(self, url):
        assert url == catalog.CURRENT_ZIP_URL
        return {"content-type": "application/zip", "content-length": "39000", "last-modified": self.archive_date}

    def refresh(self, root):
        return catalog.refresh_catalog(root=root, fetch=self.fetch, head=self.head, now=NOW)


def test_complete_catalog_discovery_and_compact_identity(tmp_path):
    cms = CMS([dataset(str(i), title=f"Dataset {i}") for i in range(18)])
    result = cms.refresh(tmp_path)
    assert result["summary"]["datasets"] == 18
    assert result["summary"]["new_datasets"] == 18
    assert result["status"] == "OK"
    assert all(row["resources"][0]["version"] == "100" for row in result["datasets"])
    assert all(row["publisher_identity"] for row in result["datasets"])
    assert all("baseline" in row["reason"] for row in result["datasets"])
    assert catalog.snapshot_path(tmp_path).stat().st_size < 100_000


def test_next_month_version_discovered_without_code_changes(tmp_path):
    cms = CMS(); cms.refresh(tmp_path)
    cms.rows = [dataset(month="09", version="200")]
    row = cms.refresh(tmp_path)["datasets"][0]
    assert row["status"] == "NEWER"
    assert row["previous_successful"]["logical_release"] == "2026-08"
    assert row["logical_release"] == "2026-09"


def test_same_month_resource_replacement_is_revised(tmp_path):
    cms = CMS(); cms.refresh(tmp_path)
    cms.rows = [dataset(version="101")]
    result = cms.refresh(tmp_path)
    assert result["datasets"][0]["status"] == "REVISED"
    assert result["summary"]["revised"] == 1
    assert result["summary"]["published"] == 0


def test_planned_today_with_unchanged_resource_is_not_newer(tmp_path):
    cms = CMS(); cms.refresh(tmp_path)
    result = cms.refresh(tmp_path)
    assert result["datasets"][0]["status"] == "PLANNED_TODAY"
    assert result["summary"]["published"] == 0


def test_not_due_and_metadata_only_change_remain_current(tmp_path):
    cms = CMS([dataset(planned="2026-10-28")]); cms.refresh(tmp_path)
    cms.rows[0]["description"] = "Changed description"
    cms.rows[0]["%modified"] = "2026-09-30"
    assert cms.refresh(tmp_path)["datasets"][0]["status"] == "CURRENT"


def test_lookup_failure_preserves_success_and_recovery_compares_it(tmp_path):
    cms = CMS(); first = cms.refresh(tmp_path)["datasets"][0]
    cms.fail.add(catalog.METASTORE + "y2hd-n93e?show-reference-ids=true")
    failed = cms.refresh(tmp_path)["datasets"][0]
    assert failed["status"] == "ERROR"
    assert failed["last_successful"] == first["last_successful"]
    assert failed["publisher_identity"] == first["publisher_identity"]
    assert failed["last_successful_at"] == first["last_successful_at"]
    cms.fail.clear(); cms.rows = [dataset(version="102")]
    assert cms.refresh(tmp_path)["datasets"][0]["status"] == "REVISED"


def test_new_dataset_and_removed_membership(tmp_path):
    cms = CMS(); cms.refresh(tmp_path)
    cms.rows.append(dataset("new-id", title="New CMS data"))
    result = cms.refresh(tmp_path)
    assert next(row for row in result["datasets"] if row["stable_id"] == "new-id")["status"] == "NEW_DATASET"
    cms.rows = [cms.rows[-1]]
    result = cms.refresh(tmp_path)
    assert next(row for row in result["datasets"] if row["stable_id"] == "y2hd-n93e")["status"] == "REMOVED_OR_ARCHIVED"


def test_listing_failure_preserves_previous_success_and_never_infers_removal(tmp_path):
    cms = CMS(); cms.refresh(tmp_path)
    before = catalog.load_catalog(tmp_path)
    cms.fail.add("all")
    failed = cms.refresh(tmp_path)
    assert failed["status"] == "ERROR"
    assert failed["datasets"][0]["status"] == "ERROR"
    assert failed["datasets"][0]["last_successful"] == before["datasets"][0]["last_successful"]
    assert failed["theme_archive"]["last_successful"] == before["theme_archive"]["last_successful"]


def test_archive_change_cannot_mark_dataset_newer(tmp_path):
    cms = CMS(); cms.refresh(tmp_path)
    cms.archive_date = "2026-09-30"
    result = cms.refresh(tmp_path)
    assert result["theme_archive"]["status"] == "CHANGED"
    assert result["datasets"][0]["status"] == "PLANNED_TODAY"


def test_coverage_keeps_source_families_separate():
    assert catalog.coverage_for("y2hd-n93e")["source_id"] == "cms.nh_ownership"
    assert catalog.coverage_for("xcdc-v8bm")["source_id"] is None
    assert catalog.coverage_for("djen-97ju")["class"] == "C"
    assert catalog.coverage_for("fykj-qjee")["dataset_id"] == "cms.snf_qrp_provider"
    assert catalog.coverage_for("unknown-stable-id")["action"] is None


def test_incomplete_listing_fails_closed():
    with pytest.raises(ValueError, match="duplicate|incomplete"):
        catalog.discover_theme(lambda _: {"total": 18, "results": {"one": dataset()}})


def test_regressing_logical_release_does_not_replace_baseline(tmp_path):
    cms = CMS([dataset(month="09", version="200")]); before = cms.refresh(tmp_path)["datasets"][0]
    cms.rows = [dataset()]
    row = cms.refresh(tmp_path)["datasets"][0]
    assert row["status"] == "ERROR"
    assert row["last_successful"] == before["last_successful"]


def test_refresh_does_not_touch_active_candidates_or_acquire(tmp_path):
    state = tmp_path / "state"
    before = {name: (state / name).read_bytes() for name in ("active_releases.json", "release_candidates.json")}
    CMS().refresh(tmp_path)
    assert all((state / name).read_bytes() == data for name, data in before.items())
    assert not (tmp_path / "downloads").exists()


def test_check_releases_includes_detect_only_catalog_summary(tmp_path):
    from release_check import check_releases
    cms = CMS()
    result = check_releases(acquire=False, root=tmp_path, catalog_fetch=cms.fetch, catalog_head=cms.head)
    assert result["cms_nh_catalog"]["summary"]["datasets"] == 1
    assert catalog.load_catalog(tmp_path)["status"] == "OK"


def test_sources_catalog_inspection_and_no_fake_acquire(tmp_path, monkeypatch):
    from data_ops_app import create_app
    import data_ops_app
    cms = CMS([dataset(), dataset("unknown", title="Unsupported CMS source")])
    cms.refresh(tmp_path); cms.refresh(tmp_path)
    monkeypatch.setattr(catalog, "load_catalog", lambda root=None: json.loads(catalog.snapshot_path(tmp_path).read_text()))
    monkeypatch.setenv("PBJ_DATA_OPS_PASSWORD", "test-password")
    monkeypatch.setattr(data_ops_app, "snapshots_with_control_plane", lambda **kwargs: [])
    monkeypatch.setattr(data_ops_app, "build_needs_attention_queue", lambda **kwargs: [])
    monkeypatch.setattr(data_ops_app, "control_panel_payload", lambda: {"datasets": [], "facilities": {"facilities": {}, "counts": {}}})
    client = create_app().test_client()
    with client.session_transaction() as session:
        session["data_ops_authenticated"] = True
    response = client.get("/sources")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    section = html.split('id="cms-nh-catalog"')[1].split('</section>')[0]
    assert "CMS Nursing Home Provider Data" in section
    assert "Previous identity" in section and "Current identity" in section
    assert "PLANNED_TODAY" in section
    assert "Detected only — no governed lifecycle" in section
    assert "/sources/cms.provider_info?check_cms=0" in section
    assert catalog.coverage_for("y2hd-n93e")["dataset_id"] == "cms.nh_ownership"
    assert "<form" not in section and "Acquire" not in section
    assert "Download current theme ZIP from CMS" in section


def test_theme_compatibility_discovery_does_not_download_zip(tmp_path):
    from cms_theme_publication import resolve_theme_publication
    cms = CMS()
    result = resolve_theme_publication(fetch_json=cms.fetch)
    assert result.member_for_source("cms.nh_ownership").filename.endswith("082026.csv")
    assert not any(url.endswith(".zip") for url in cms.urls)
