"""Regression: Git LFS pointer stubs must never become ProviderInfoNorm."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import pytest

import normalize_provider_info as npi
from run_pipeline_update import detect_new_provider_info_files


LFS_POINTER = (
    b"version https://git-lfs.github.com/spec/v1\n"
    b"oid sha256:742ae596a3bb5e1474dcab25b057128eaf8cd73961897441334819ad82789698\n"
    b"size 9233835\n"
)


def _real_nh_csv(n: int = 5) -> bytes:
    cols = [
        "CMS Certification Number (CCN)",
        "Provider Name",
        "State",
        "Processing Date",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for i in range(n):
        w.writerow(
            {
                cols[0]: f"{100000 + i}",
                cols[1]: f"Facility {i}",
                cols[2]: "CT",
                cols[3]: "2026-08-01",
            }
        )
    return buf.getvalue().encode("utf-8")


def test_detect_skips_lfs_pointer(tmp_path: Path, monkeypatch):
    pi = tmp_path / "provider_info"
    pi.mkdir()
    (tmp_path / "provider_info_normalized").mkdir()
    stub = pi / "NH_ProviderInfo_Jan2026.csv"
    stub.write_bytes(LFS_POINTER)
    real = pi / "NH_ProviderInfo_Aug2026.csv"
    real.write_bytes(_real_nh_csv())

    monkeypatch.chdir(tmp_path)
    found = detect_new_provider_info_files(force=False)
    names = {p.name for p in found}
    assert "NH_ProviderInfo_Jan2026.csv" not in names
    assert "NH_ProviderInfo_Aug2026.csv" in names


def test_normalize_lfs_pointer_fails_closed(tmp_path: Path):
    stub = tmp_path / "NH_ProviderInfo_Mar2026.csv"
    stub.write_bytes(LFS_POINTER)
    out = tmp_path / "ProviderInfoNorm_2026_03.csv"
    with pytest.raises(npi.LfsPointerError):
        npi.normalize_nh_file(stub, out, list(npi.NH_TO_NORM.keys()) + ["quarter"])
    assert not out.exists()


def test_normalize_main_rejects_lfs_pointer(tmp_path: Path, monkeypatch):
    pi = tmp_path / "provider_info"
    pi.mkdir()
    norm = tmp_path / "provider_info_normalized"
    norm.mkdir()
    stub = pi / "NH_ProviderInfo_Jun2026.csv"
    stub.write_bytes(LFS_POINTER)

    monkeypatch.setattr(npi.cms_data_paths, "provider_info_dir", lambda root=None: pi)
    monkeypatch.setattr(npi.cms_data_paths, "provider_info_normalized_dir", lambda root=None: norm)
    rc = npi.main(["--file", "NH_ProviderInfo_Jun2026.csv", "--force"])
    assert rc == 1
    assert not list(norm.glob("ProviderInfoNorm_*"))


def test_normalize_real_csv_still_works(tmp_path: Path):
    nh = tmp_path / "NH_ProviderInfo_Aug2026.csv"
    nh.write_bytes(_real_nh_csv(8))
    out = tmp_path / "ProviderInfoNorm_2026_08.csv"
    n = npi.normalize_nh_file(nh, out, list(npi.NH_TO_NORM.keys()) + ["quarter"])
    assert n == 8
    assert out.is_file()
    assert out.stat().st_size > 0
    # Must not look like an LFS pointer
    assert not out.read_bytes().startswith(b"version https://git-lfs.github.com/spec/v1")
