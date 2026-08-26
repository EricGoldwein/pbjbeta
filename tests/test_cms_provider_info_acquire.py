"""Tests for CMS Provider Information acquire (4pq5-n9py) → normalize handoff."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

import sys
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import cms_provider_info_acquire as acq


def _nh_csv_bytes(n: int = 1200, month_label: str = "2026-08-01", prefix: str = "000") -> bytes:
    """Build a minimal NH_ProviderInfo-like CSV with required columns."""
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


def test_resolve_cms_provider_info_release_from_metastore():
    cms = acq.resolve_cms_provider_info_release(fetch_json=lambda url: _metastore())
    assert cms.dataset_id == "4pq5-n9py"
    assert cms.distribution_filename == "NH_ProviderInfo_Aug2026.csv"
    assert cms.year == 2026 and cms.month == 8
    assert cms.data_vintage_label == "Aug 2026"
    assert cms.released == "2026-08-26"


def test_already_current_is_noop(tmp_path: Path):
    root = tmp_path
    pi = root / "provider_info"
    pi.mkdir()
    # Seed July as local latest
    jul = pi / "NH_ProviderInfo_Jul2026.csv"
    jul.write_bytes(_nh_csv_bytes(1100, "2026-07-01", "100"))
    # And Aug already present → CURRENT
    aug = pi / "NH_ProviderInfo_Aug2026.csv"
    aug.write_bytes(_nh_csv_bytes(1100, "2026-08-01", "100"))

    report = acq.acquire_and_process(
        root=root,
        fetch_json=lambda url: _metastore(),
        fetch_bytes=lambda url: (_ for _ in ()).throw(AssertionError("should not download")),
        dry_run=False,
    )
    assert report["status"] == "CURRENT"
    assert report.get("cross_repo_write") is not True


def test_newer_release_downloads_and_normalizes(tmp_path: Path, monkeypatch):
    root = tmp_path
    pi = root / "provider_info"
    pi.mkdir()
    jul = pi / "NH_ProviderInfo_Jul2026.csv"
    jul.write_bytes(_nh_csv_bytes(1100, "2026-07-01", "100"))

    # Point cms_data_paths at tmp root
    monkeypatch.setenv("PBJ_REPO_ROOT", str(root))
    # cms_data_paths may not read PBJ_REPO_ROOT — patch helpers
    import cms_data_paths

    monkeypatch.setattr(cms_data_paths, "repo_root", lambda: root)
    monkeypatch.setattr(cms_data_paths, "provider_info_dir", lambda r=None: (r or root) / "provider_info")
    monkeypatch.setattr(
        cms_data_paths,
        "provider_info_normalized_dir",
        lambda r=None: (r or root) / "provider_info_normalized",
    )

    # Normalize in-process (avoid depending on subprocess cwd)
    def fake_normalize(nh_basename: str, *, force: bool = False) -> int:
        import normalize_provider_info as npi

        nh_path = pi / nh_basename
        out = root / "provider_info_normalized" / "ProviderInfoNorm_2026_08.csv"
        npi.normalize_nh_file(nh_path, out, npi._template_columns(out.parent))
        return 0

    monkeypatch.setattr(acq, "run_normalize_for_file", fake_normalize)
    monkeypatch.setattr(acq, "_ROOT", root)

    payload = _nh_csv_bytes(1150, "2026-08-01", "200")
    report = acq.acquire_and_process(
        root=root,
        fetch_json=lambda url: _metastore(),
        fetch_bytes=lambda url: payload,
    )
    assert report["status"] == "READY_FOR_PUBLIC_HANDOFF"
    assert (pi / "NH_ProviderInfo_Aug2026.csv").is_file()
    assert (root / "provider_info_normalized" / "ProviderInfoNorm_2026_08.csv").is_file()
    assert Path(report["handoff_path"]).is_file()
    handoff = json.loads(Path(report["handoff_path"]).read_text(encoding="utf-8"))
    assert handoff["pbj_root_sync"]["sha256"]
    assert handoff["pbj_root_sync"]["destination_file"].endswith("ProviderInfoNorm_2026_08.csv")
    assert report["delta"]["providers_added"] >= 0
    assert report["cross_repo_write"] is False
    assert report["deployed"] is False


def test_corrupt_empty_download_fails_closed(tmp_path: Path, monkeypatch):
    root = tmp_path
    (root / "provider_info").mkdir()
    monkeypatch.setattr(acq.cms_data_paths, "repo_root", lambda: root)
    monkeypatch.setattr(acq.cms_data_paths, "provider_info_dir", lambda r=None: (r or root) / "provider_info")

    with pytest.raises(acq.AcquireError, match="too small|empty|does not look"):
        acq.acquire_and_process(
            root=root,
            fetch_json=lambda url: _metastore(),
            fetch_bytes=lambda url: b"",
        )


def test_schema_failure_fails_closed(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(acq.AcquireError, match="missing required columns"):
        acq.validate_raw_provider_info_csv(path, min_rows=1)


def test_no_cross_repo_write_in_acquire_module():
    text = Path(acq.__file__).read_text(encoding="utf-8")
    assert "pbj-root" not in text.lower() or "Does not write to pbj-root" in text
    assert "git push" not in text
    assert "sync_to_pbj_root" not in text


@pytest.mark.live
def test_live_cms_resolves_aug_2026():
    import os

    if os.environ.get("RUN_LIVE_CMS") != "1":
        pytest.skip("Set RUN_LIVE_CMS=1 for live CMS metastore check")
    cms = acq.resolve_cms_provider_info_release()
    assert cms.dataset_id == "4pq5-n9py"
    assert "2026" in cms.distribution_filename
    assert cms.month >= 7  # Jul or Aug 2026 depending on CMS clock
