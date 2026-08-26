"""Tests for CMS PBJ nurse staffing acquire (data-api resources → standardize)."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from unittest import mock

import pytest

import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import cms_pbj_nurse_acquire as nurse  # noqa: E402


def _nurse_csv_bytes(n: int = 1200, quarter: str = "2026Q1") -> bytes:
    cols = ["PROVNUM", "CY_Qtr", "WorkDate", "MDScensus", "Hrs_RN", "PROVNAME", "STATE"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for i in range(n):
        ccn = f"{i:06d}"
        w.writerow(
            {
                "PROVNUM": ccn,
                "CY_Qtr": quarter,
                "WorkDate": "20260115",
                "MDScensus": "100",
                "Hrs_RN": "8.0",
                "PROVNAME": f"Facility {ccn}",
                "STATE": "CT",
            }
        )
    return buf.getvalue().encode("utf-8")


def _resources_payload(
    filename: str = "PBJ_dailynursestaffing_CY2026Q1.csv",
    *,
    file_size: int | None = None,
    file_uuid: str = "abc-uuid",
) -> dict:
    return {
        "meta": {"success": True},
        "data": [
            {
                "type": "Primary",
                "title": "PBJ Daily Nurse Staffing Q1 2026",
                "media_bundle": "primary_dataset_file",
                "file_uuid": file_uuid,
                "file_name": filename,
                "file_mime": "text/csv",
                "file_size": file_size,
                "file_url": f"https://data.cms.gov/sites/default/files/fake/{filename}",
            }
        ],
    }


def _write_manifest(
    root: Path,
    data: bytes,
    *,
    filename: str = "PBJ_dailynursestaffing_CY2026Q1.csv",
    quarter: str = "CY2026Q1",
    file_size: int | None = None,
    file_uuid: str = "abc-uuid",
    sha256: str | None = None,
) -> Path:
    sha = sha256 or hashlib.sha256(data).hexdigest()
    size = file_size if file_size is not None else len(data)
    record = {
        "stable_cms_dataset_id": nurse.NURSE_DATASET_ID,
        "quarter_label": quarter,
        "source_filename": filename,
        "source_url": f"https://data.cms.gov/sites/default/files/fake/{filename}",
        "cms_file_uuid": file_uuid,
        "cms_reported_file_size": size,
        "byte_size": len(data),
        "sha256": sha,
    }
    d = root / "PBJcsv" / "_manifests" / quarter
    d.mkdir(parents=True, exist_ok=True)
    path = d / "acquisition.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return path


def test_resolve_cms_nurse_release_from_resources():
    data = _nurse_csv_bytes()
    cms = nurse.resolve_cms_nurse_release(
        fetch_json=lambda url: _resources_payload(file_size=len(data))
    )
    assert cms.dataset_id == nurse.NURSE_DATASET_ID
    assert cms.quarter_label == "CY2026Q1"
    assert cms.year == 2026 and cms.quarter == 1
    assert cms.distribution_filename.endswith("CY2026Q1.csv")


def test_dry_run_does_not_download(tmp_path: Path):
    calls = {"bytes": 0}
    data = _nurse_csv_bytes()

    def fetch_bytes(url: str) -> bytes:
        calls["bytes"] += 1
        return data

    report = nurse.acquire_and_process(
        root=tmp_path,
        fetch_json=lambda url: _resources_payload(file_size=len(data)),
        fetch_bytes=fetch_bytes,
        dry_run=True,
    )
    assert report["status"] == "WOULD_ACQUIRE"
    assert calls["bytes"] == 0
    assert not list((tmp_path / "PBJcsv").glob("*.csv")) if (tmp_path / "PBJcsv").exists() else True


def test_successful_acquisition_and_provenance(tmp_path: Path):
    data = _nurse_csv_bytes(1500)
    standardize_calls = {"n": 0}

    def fake_standardize(*, root=None):
        standardize_calls["n"] += 1
        std_dir = tmp_path / "standardized_PBJ"
        std_dir.mkdir(exist_ok=True)
        (std_dir / "PBJ_dailynursestaffing_CY2026Q1.csv").write_bytes(data)
        return 0

    with mock.patch.object(nurse, "run_standardize_nurse", side_effect=fake_standardize):
        report = nurse.acquire_and_process(
            root=tmp_path,
            fetch_json=lambda url: _resources_payload(file_size=len(data)),
            fetch_bytes=lambda url: data,
            min_rows=1000,
            min_bytes=100,
        )
    assert report["status"] == "PROCESSED"
    assert report["lifecycle"] == "PROCESSED"
    assert standardize_calls["n"] == 1
    dest = tmp_path / "PBJcsv" / "PBJ_dailynursestaffing_CY2026Q1.csv"
    assert dest.is_file()
    acq = tmp_path / "PBJcsv" / "_manifests" / "CY2026Q1" / "acquisition.json"
    assert acq.is_file()
    text = acq.read_text(encoding="utf-8")
    assert "stable_cms_dataset_id" in text
    assert "sha256" in text
    assert "source_url" in text


def test_same_filename_wrong_bytes_not_current(tmp_path: Path):
    good = _nurse_csv_bytes(1200)
    bad = _nurse_csv_bytes(1300)
    raw = tmp_path / "PBJcsv"
    raw.mkdir()
    dest = raw / "PBJ_dailynursestaffing_CY2026Q1.csv"
    dest.write_bytes(bad)
    _write_manifest(tmp_path, good)  # manifest claims good; disk is bad
    std = tmp_path / "standardized_PBJ"
    std.mkdir()
    (std / dest.name).write_bytes(bad)

    with pytest.raises(nurse.AcquireError, match="does not match acquisition"):
        nurse.acquire_and_process(
            root=tmp_path,
            fetch_json=lambda url: _resources_payload(file_size=len(good)),
            fetch_bytes=lambda url: (_ for _ in ()).throw(AssertionError("no download")),
            min_rows=1000,
            min_bytes=100,
        )


def test_manifest_sha_match_is_current_noop(tmp_path: Path):
    data = _nurse_csv_bytes(1200)
    raw = tmp_path / "PBJcsv"
    raw.mkdir()
    dest = raw / "PBJ_dailynursestaffing_CY2026Q1.csv"
    dest.write_bytes(data)
    _write_manifest(tmp_path, data)
    std = tmp_path / "standardized_PBJ"
    std.mkdir()
    (std / dest.name).write_bytes(data)

    report = nurse.acquire_and_process(
        root=tmp_path,
        fetch_json=lambda url: _resources_payload(file_size=len(data)),
        fetch_bytes=lambda url: (_ for _ in ()).throw(AssertionError("should not download")),
        min_rows=1000,
        min_bytes=100,
    )
    assert report["status"] == "CURRENT"
    assert report["identity"]["cryptographically_identical"] is True
    assert report["identity"]["verdict"] == nurse.IDENTITY_IDENTICAL


def test_manifest_sha_mismatch_fails(tmp_path: Path):
    data = _nurse_csv_bytes(1200)
    raw = tmp_path / "PBJcsv"
    raw.mkdir()
    dest = raw / "PBJ_dailynursestaffing_CY2026Q1.csv"
    dest.write_bytes(data)
    _write_manifest(tmp_path, data, sha256="0" * 64)

    with pytest.raises(nurse.AcquireError, match="does not match acquisition"):
        nurse.acquire_and_process(
            root=tmp_path,
            fetch_json=lambda url: _resources_payload(file_size=len(data)),
            min_rows=1000,
            min_bytes=100,
        )
    identity = nurse.assess_local_release_identity(
        nurse.resolve_cms_nurse_release(
            fetch_json=lambda url: _resources_payload(file_size=len(data))
        ),
        root=tmp_path,
        min_rows=1000,
        min_bytes=100,
    )
    assert identity.verdict == nurse.IDENTITY_MANIFEST_MISMATCH
    assert identity.cryptographically_identical is False


def test_unmanifested_historical_raw_conservative(tmp_path: Path):
    data = _nurse_csv_bytes(1200)
    raw = tmp_path / "PBJcsv"
    raw.mkdir()
    dest = raw / "PBJ_dailynursestaffing_CY2026Q1.csv"
    dest.write_bytes(data)
    std = tmp_path / "standardized_PBJ"
    std.mkdir()
    (std / dest.name).write_bytes(data)

    report = nurse.acquire_and_process(
        root=tmp_path,
        fetch_json=lambda url: _resources_payload(file_size=len(data)),
        fetch_bytes=lambda url: (_ for _ in ()).throw(AssertionError("should not download")),
        min_rows=1000,
        min_bytes=100,
    )
    assert report["status"] == "LOCAL_UNMANIFESTED"
    assert report["status"] != "CURRENT"
    assert report["identity"]["cryptographically_identical"] is False
    assert report["identity"]["verdict"] == nurse.IDENTITY_UNMANIFESTED_OK


def test_duplicate_identical_release_noop(tmp_path: Path):
    """Back-compat name: manifested SHA match → CURRENT without download."""
    test_manifest_sha_match_is_current_noop(tmp_path)


def test_refuses_overwrite_different_checksum(tmp_path: Path):
    raw = tmp_path / "PBJcsv"
    raw.mkdir()
    dest = raw / "PBJ_dailynursestaffing_CY2026Q1.csv"
    existing = _nurse_csv_bytes(1200, "2026Q1")
    dest.write_bytes(existing)
    other = _nurse_csv_bytes(1300, "2026Q1")  # different content
    with pytest.raises(nurse.AcquireError, match="refusing overwrite"):
        nurse.download_nurse_csv(
            nurse.resolve_cms_nurse_release(
                fetch_json=lambda u: _resources_payload(file_size=len(other))
            ),
            dest,
            fetch_bytes=lambda u: other,
            min_bytes=100,
            validate=False,
        )
    # existing untouched
    assert dest.read_bytes() == existing


def test_stream_download_writes_chunks_and_sha(tmp_path: Path):
    data = _nurse_csv_bytes(1100)
    chunks_seen: list[int] = []

    def fake_stream(url: str, dest: Path) -> str:
        h = hashlib.sha256()
        with dest.open("wb") as out:
            view = memoryview(data)
            step = 4096
            for i in range(0, len(view), step):
                chunk = bytes(view[i : i + step])
                chunks_seen.append(len(chunk))
                h.update(chunk)
                out.write(chunk)
        return h.hexdigest()

    cms = nurse.resolve_cms_nurse_release(
        fetch_json=lambda u: _resources_payload(file_size=len(data))
    )
    dest = tmp_path / "PBJcsv" / cms.distribution_filename
    result = nurse.download_nurse_csv(
        cms,
        dest,
        stream_download=fake_stream,
        min_bytes=100,
        min_rows=1000,
        validate=True,
    )
    assert result["action"] == "written"
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert dest.read_bytes() == data
    assert len(chunks_seen) > 1


def test_failed_stream_removes_partial_leaves_dest(tmp_path: Path):
    existing = _nurse_csv_bytes(1200)
    raw = tmp_path / "PBJcsv"
    raw.mkdir()
    dest = raw / "PBJ_dailynursestaffing_CY2026Q1.csv"
    dest.write_bytes(existing)

    def boom(url: str, path: Path) -> str:
        path.write_bytes(b"partial-junk")
        raise nurse.AcquireError("simulated stream failure")

    cms = nurse.resolve_cms_nurse_release(
        fetch_json=lambda u: _resources_payload(file_size=len(existing))
    )
    with pytest.raises(nurse.AcquireError, match="simulated stream failure"):
        nurse.download_nurse_csv(
            cms,
            dest,
            stream_download=boom,
            min_bytes=100,
            validate=False,
        )
    assert dest.read_bytes() == existing
    leftovers = list(raw.glob(".*.partial")) + list(raw.glob(".staging_*"))
    assert leftovers == []


def test_malformed_missing_columns_blocked(tmp_path: Path):
    bad = b"foo,bar\n1,2\n" + b"1,2\n" * 1200
    path = tmp_path / "PBJ_dailynursestaffing_CY2026Q1.csv"
    path.write_bytes(bad)
    with pytest.raises(nurse.AcquireError, match="missing required columns"):
        nurse.validate_raw_nurse_csv(path, min_rows=1000, min_bytes=10)


def test_wrong_dataset_blocked(tmp_path: Path):
    cols = ["CMS Certification Number (CCN)", "Special Focus Status", "Provider Name"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for i in range(1200):
        w.writerow({cols[0]: f"{i:06d}", cols[1]: "", cols[2]: "X"})
    path = tmp_path / "PBJ_dailynursestaffing_CY2026Q1.csv"
    path.write_bytes(buf.getvalue().encode("utf-8"))
    with pytest.raises(nurse.AcquireError, match="Provider Info"):
        nurse.validate_raw_nurse_csv(path, min_rows=1000, min_bytes=10)


def test_standardizer_only_after_structural_pass(tmp_path: Path):
    bad = b"not,a,nurse,file\n" + b"1,2,3,4\n" * 50
    standardize_calls = {"n": 0}

    def boom(*, root=None):
        standardize_calls["n"] += 1
        return 0

    with mock.patch.object(nurse, "run_standardize_nurse", side_effect=boom):
        with pytest.raises(nurse.AcquireError):
            nurse.acquire_and_process(
                root=tmp_path,
                fetch_json=lambda url: _resources_payload(file_size=len(bad)),
                fetch_bytes=lambda url: bad,
                min_rows=10,
                min_bytes=10,
            )
    assert standardize_calls["n"] == 0
    # failed validate should not leave final dest (download rolls back)
    dest = tmp_path / "PBJcsv" / "PBJ_dailynursestaffing_CY2026Q1.csv"
    assert not dest.exists()


def test_cms_newer_status_via_ops(tmp_path: Path):
    import cms_data_ops as ops

    snap = ops.probe_source(
        "cms.pbj_nurse_staffing",
        check_cms=True,
        fetch_json=lambda url: _resources_payload(file_size=1000),
        root=tmp_path,
        run_zweli=False,
    )
    assert snap.publisher_latest == "CY2026Q1"
    assert snap.status in {"CMS_NEWER", "NOT_AVAILABLE_IN_THIS_RUNTIME"} or "CMS" in (
        snap.detail or ""
    )


def test_probe_unmanifested_not_current(tmp_path: Path):
    import cms_data_ops as ops

    data = _nurse_csv_bytes(1200)
    raw = tmp_path / "PBJcsv"
    raw.mkdir()
    (raw / "PBJ_dailynursestaffing_CY2026Q1.csv").write_bytes(data)
    std = tmp_path / "standardized_PBJ"
    std.mkdir()
    (std / "PBJ_dailynursestaffing_CY2026Q1.csv").write_bytes(data)

    snap = ops.probe_source(
        "cms.pbj_nurse_staffing",
        check_cms=True,
        fetch_json=lambda url: _resources_payload(file_size=len(data)),
        root=tmp_path,
        run_zweli=False,
    )
    assert snap.status != "CURRENT"
    assert "provenance" in (snap.detail or "").lower() or "acquisition" in (
        snap.detail or ""
    ).lower() or "not cryptographically" in (snap.detail or "").lower()


@pytest.mark.live_cms
def test_live_cms_nurse_resources_opt_in():
    cms = nurse.resolve_cms_nurse_release()
    assert cms.dataset_id == nurse.NURSE_DATASET_ID
    assert nurse.FILENAME_RE.match(cms.distribution_filename)
    assert cms.distribution_url.startswith("https://")
