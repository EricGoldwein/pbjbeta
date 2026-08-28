"""Tests for CMS nursing-home theme publication discovery."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from cms_theme_publication import (
    modified_date_to_product_release_id,
    parse_theme_manifest,
    pick_latest_theme_publication_row,
    publication_availability_for_source,
    resolve_theme_publication,
)


FIXTURES = Path(__file__).parent / "fixtures"


def _archive_row(date: str, *, pub_id: str = "theme-pub-1") -> dict:
    return {
        "type": "theme",
        "date": date,
        "id": pub_id,
        "url": "/provider-data/dataset-archives/theme/nursing-homes/nursing-homes_2026-08-26.zip",
        "name": f"nursing-homes_{date}",
        "theme": "nursing-homes",
        "size": 34000000,
    }


def test_modified_date_to_product_release_id():
    assert modified_date_to_product_release_id("2026-08-01") == "2026-08"
    assert modified_date_to_product_release_id("2026-07-29") == "2026-07"


def test_pick_latest_theme_publication_row_excludes_snapshots():
    rows = [
        {"type": "theme", "date": "2026-07-01", "name": "nursing-homes_2026-07-01"},
        {"type": "theme", "date": "2026-08-26", "name": "nursing-homes_2026-08-26"},
        {"type": "theme", "date": "2025-12-01", "name": "Annual Snapshot 2025"},
    ]
    latest = pick_latest_theme_publication_row(rows)
    assert latest["date"] == "2026-08-26"


def test_parse_theme_manifest_maps_governed_sources():
    manifest = json.loads((FIXTURES / "cms_theme_manifest_2026-08-26.json").read_text(encoding="utf-8"))
    members = parse_theme_manifest(manifest)
    assert set(members) == {"cms.health_citations", "cms.provider_info", "cms.nh_ownership"}
    hc = members["cms.health_citations"]
    assert hc.dataset_id == "r5ix-sfxw"
    assert hc.product_release_id == "2026-08"
    assert hc.filename.endswith("NH_HealthCitations_Aug2026.csv")


def test_resolve_theme_publication_from_fixture_manifest():
    manifest = json.loads((FIXTURES / "cms_theme_manifest_2026-08-06.json").read_text(encoding="utf-8"))
    publication = resolve_theme_publication(
        archive_index=[_archive_row("2026-08-06", pub_id="aug6")],
        manifest=manifest,
    )
    assert publication.publication_date == "2026-08-06"
    assert publication.member_for_source("cms.provider_info") is not None
    assert publication.member_for_source("cms.health_citations") is None


def test_publication_availability_new_release_when_active_lags():
    manifest = json.loads((FIXTURES / "cms_theme_manifest_2026-08-26.json").read_text(encoding="utf-8"))
    publication = resolve_theme_publication(
        archive_index=[_archive_row("2026-08-26")],
        manifest=manifest,
    )
    avail = publication_availability_for_source(
        "cms.health_citations",
        active_release_id="2026-07",
        publication=publication,
    )
    assert avail is not None
    assert avail["new_release_available"] is True
    assert avail["publisher_latest_release_id"] == "2026-08"
    assert avail["cms_publication_date"] == "2026-08-26"
    assert avail["processing_modified_date"] == "2026-08-01"


def test_publication_availability_unchanged_when_not_in_manifest():
    manifest = json.loads((FIXTURES / "cms_theme_manifest_2026-08-06.json").read_text(encoding="utf-8"))
    publication = resolve_theme_publication(
        archive_index=[_archive_row("2026-08-06")],
        manifest=manifest,
    )
    avail = publication_availability_for_source(
        "cms.health_citations",
        active_release_id="2026-07",
        publication=publication,
    )
    assert avail is not None
    assert avail["unchanged_in_latest_publication"] is True
    assert avail["new_release_available"] is False
