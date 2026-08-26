"""Tests for PBJ Data Ops status probes and Provider Info action wrappers."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

import cms_data_ops as ops
import cms_provider_info_acquire as acq


def _nh_csv_bytes(n: int = 1200, month_label: str = "2026-08-01", prefix: str = "000") -> bytes:
    cols = list(acq.REQUIRED_NH_COLUMNS) + ["City/Town", "Overall Rating"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for i in range(n):
        ccn = f"{prefix}{i:03d}"[-6:].zfill(6)
        w.writerow(
            {
                cols[0]: ccn,
                cols[1]: f"Facility {ccn}",
                cols[2]: "CT",
                cols[3]: month_label,
                "City/Town": "Hartford",
                "Overall Rating": "3",
            }
        )
    return buf.getvalue().encode("utf-8")


def _metastore(filename: str = "NH_ProviderInfo_Aug2026.csv") -> dict:
    return {
        "title": "Provider Information",
        "identifier": "4pq5-n9py",
        "released": "2026-08-26",
        "modified": "2026-08-01",
        "nextUpdateDate": "2026-09-30",
        "distribution": [
            {
                "data": {
                    "@type": "dcat:Distribution",
                    "downloadURL": (
                        "https://data.cms.gov/provider-data/sites/default/files/resources/"
                        f"abc/{filename}"
                    ),
                    "mediaType": "text/csv",
                }
            }
        ],
    }


def test_probe_all_sources_local_only(tmp_path: Path):
    snaps = ops.probe_all_sources(check_cms=False, root=tmp_path)
    assert len(snaps) == 9
    by_id = {s.source_id: s for s in snaps}
    assert by_id["cms.snf_enrollments"].status == "UNKNOWN"
    assert by_id["cms.provider_info"].status == "UNKNOWN"
    assert by_id["cms.provider_info"].actions_enabled == ["check_cms", "acquire_process"]
    assert by_id["cms.pbj_nurse_staffing"].actions_enabled == []


def test_probe_provider_info_cms_newer(tmp_path: Path):
    pi = tmp_path / "provider_info"
    pi.mkdir()
    (pi / "NH_ProviderInfo_Jul2026.csv").write_bytes(_nh_csv_bytes(1100, "2026-07-01", "100"))
    snap = ops.probe_source(
        "cms.provider_info",
        check_cms=True,
        fetch_json=lambda url: _metastore(),
        root=tmp_path,
    )
    assert snap.status == "CMS_NEWER"
    assert snap.cms_latest == "Aug 2026"
    assert "Jul" in (snap.pbjapp_latest or "")


def test_probe_provider_info_ready_for_handoff(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    pi = tmp_path / "provider_info"
    norm = tmp_path / "provider_info_normalized"
    pi.mkdir()
    norm.mkdir()
    (pi / "NH_ProviderInfo_Aug2026.csv").write_bytes(_nh_csv_bytes(1200, "2026-08-01", "200"))
    (norm / "ProviderInfoNorm_2026_08.csv").write_bytes(_nh_csv_bytes(1200, "2026-08-01", "200"))
    man = pi / "_manifests" / "2026-08"
    man.mkdir(parents=True)
    (man / "acquisition.json").write_text(
        json.dumps({"acquired_at": "2026-08-26T17:00:00+00:00", "sha256": "abc"}),
        encoding="utf-8",
    )
    (man / "pbj_root_handoff.json").write_text(
        json.dumps(
            {
                "pbj_root_sync": {"sha256": "abc", "row_count": 1200},
                "provider_promotion": {"ready_for_pbj_commit": True},
            }
        ),
        encoding="utf-8",
    )

    # Point path helpers at tmp_path
    monkeypatch.setenv("PBJ_REPO_ROOT", str(tmp_path))
    snap = ops.probe_source(
        "cms.provider_info",
        check_cms=True,
        fetch_json=lambda url: _metastore(),
        root=tmp_path,
    )
    assert snap.status == "READY_FOR_HANDOFF"
    assert snap.last_successful_local_processing is not None


def test_probe_chain_local_raw(tmp_path: Path):
    own = tmp_path / "ownership"
    own.mkdir()
    (own / "Nursing_Home_Chain_Performance_Measures_Jul_2025.csv").write_text(
        "a,b\n1,2\n", encoding="utf-8"
    )
    (own / "Nursing_Home_Chain_Performance_Measures_Nov_2025.csv").write_text(
        "a,b\n1,2\n", encoding="utf-8"
    )
    snap = ops.probe_source("cms.chain_performance", check_cms=False, root=tmp_path)
    assert snap.status == "LOCAL_RAW_ONLY"
    assert snap.pbjapp_latest == "Nov 2025"


def test_check_provider_info_cms_wrapper(tmp_path: Path):
    pi = tmp_path / "provider_info"
    pi.mkdir()
    (pi / "NH_ProviderInfo_Aug2026.csv").write_bytes(_nh_csv_bytes(1200, "2026-08-01", "300"))
    result = ops.check_provider_info_cms(
        fetch_json=lambda url: _metastore(),
        root=tmp_path,
    )
    assert result["action"] == "check_cms"
    assert result["cms"]["dataset_id"] == "4pq5-n9py"
    assert result["cms_is_newer"] is False
    assert result["dry_run"]["status"] == "CURRENT"


def test_acquire_provider_info_dry_run(tmp_path: Path):
    pi = tmp_path / "provider_info"
    pi.mkdir()
    (pi / "NH_ProviderInfo_Aug2026.csv").write_bytes(_nh_csv_bytes(1200, "2026-08-01", "300"))
    result = ops.acquire_provider_info(
        dry_run=True,
        fetch_json=lambda url: _metastore(),
        root=tmp_path,
    )
    assert result["action"] == "acquire_process"
    assert result["acquire_report"]["status"] == "CURRENT"


def test_recommended_next_is_nurse():
    nxt = ops.recommended_next_automation()
    assert nxt["source_id"] == "cms.pbj_nurse_staffing"
