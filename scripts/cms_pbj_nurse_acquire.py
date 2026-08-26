#!/usr/bin/env python3
"""Acquire current CMS PBJ daily nurse staffing into PBJapp raw storage.

Dataset UUID (registry): 7e0d53ba-8f02-4c66-98a5-14a1c997c50d

Uses data.cms.gov data-api resources (not provider-data metastore — 404 for this UUID):

  GET https://data.cms.gov/data-api/v1/dataset/{uuid}/resources

Flow: resolve → download (if needed) → structural validate → existing
standardize_pbj_files.py. Does not write pbj-root, deploy, or promote metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import cms_data_paths  # noqa: E402
from cms_source_registry import CMS_ID_PBJ_NURSE  # noqa: E402

NURSE_DATASET_ID = CMS_ID_PBJ_NURSE
RESOURCES_URL = f"https://data.cms.gov/data-api/v1/dataset/{NURSE_DATASET_ID}/resources"

FILENAME_RE = re.compile(
    r"^PBJ_dailynursestaffing_CY(\d{4})Q([1-4])\.csv$",
    re.IGNORECASE,
)

# Core columns expected in raw CMS nurse daily (case-insensitive match).
REQUIRED_COLUMN_ALIASES: tuple[tuple[str, ...], ...] = (
    ("PROVNUM", "provnum"),
    ("CY_Qtr", "cy_qtr", "CY_QTR"),
    ("WorkDate", "workdate", "WORKDATE"),
    ("MDScensus", "mdscensus", "MDSCensus"),
    ("Hrs_RN", "hrs_rn", "HRS_RN"),
)

FetchJson = Callable[[str], Any]
# Test/fixture inject: returns full bytes (small fixtures only). Production downloads stream.
FetchBytes = Callable[[str], bytes]
# Production/stream inject: write URL body to path, return SHA-256 hex.
StreamDownload = Callable[[str, Path], str]

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


class AcquireError(Exception):
    """Fail-closed nurse acquisition / validation error."""


@dataclass(frozen=True)
class CmsNurseRelease:
    dataset_id: str
    quarter_label: str  # CY2026Q1
    year: int
    quarter: int
    distribution_filename: str
    distribution_url: str
    title: Optional[str]
    file_size: Optional[int]
    file_uuid: Optional[str]
    raw_fingerprint: str  # hash of identity fields (not file body)


@dataclass(frozen=True)
class LocalIdentityResult:
    """Provenance-aware assessment of a local raw nurse artifact vs CMS release."""

    verdict: str
    cryptographically_identical: bool
    path: Optional[Path]
    local_sha256: Optional[str]
    manifest_sha256: Optional[str]
    detail: str


# Verdicts for LocalIdentityResult.verdict
IDENTITY_MISSING = "MISSING"
IDENTITY_IDENTICAL = "IDENTICAL"
IDENTITY_MANIFEST_MISMATCH = "MANIFEST_MISMATCH"
IDENTITY_CMS_DIVERGENCE = "CMS_DIVERGENCE"
IDENTITY_UNMANIFESTED_OK = "UNMANIFESTED_OK"
IDENTITY_UNMANIFESTED_FAIL = "UNMANIFESTED_FAIL"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_fetch_json(url: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PBJapp-pbj-nurse-acquire/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _default_stream_download(
    url: str,
    dest: Path,
    *,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> str:
    """Stream CMS → file in chunks while hashing. Bounded memory."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PBJapp-pbj-nurse-acquire/1.0"},
    )
    h = hashlib.sha256()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, dest.open("wb") as out:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
                out.write(chunk)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        dest.unlink(missing_ok=True)
        raise AcquireError(f"stream download failed: {exc}") from exc
    if not dest.is_file() or dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise AcquireError(f"empty download from {url}")
    return h.hexdigest()


def _write_bytes_chunked(data: bytes, dest: Path, *, chunk_size: int = DOWNLOAD_CHUNK_SIZE) -> str:
    """Write injectable fixture bytes in chunks (tests); still hashes streaming-style."""
    if not data:
        raise AcquireError("empty download (inject)")
    h = hashlib.sha256()
    with dest.open("wb") as out:
        view = memoryview(data)
        for i in range(0, len(view), chunk_size):
            chunk = view[i : i + chunk_size]
            h.update(chunk)
            out.write(chunk)
    return h.hexdigest()


def stream_url_to_path(
    url: str,
    dest: Path,
    *,
    fetch_bytes: FetchBytes | None = None,
    stream_download: StreamDownload | None = None,
    chunk_size: int = DOWNLOAD_CHUNK_SIZE,
) -> str:
    """Download URL to ``dest`` with bounded memory; return SHA-256 hex."""
    if stream_download is not None:
        return stream_download(url, dest)
    if fetch_bytes is not None:
        return _write_bytes_chunked(fetch_bytes(url), dest, chunk_size=chunk_size)
    return _default_stream_download(url, dest, chunk_size=chunk_size)

def parse_nurse_filename(filename: str) -> tuple[int, int, str]:
    m = FILENAME_RE.match(Path(filename).name)
    if not m:
        raise AcquireError(
            f"unexpected nurse filename {filename!r}; "
            f"expected PBJ_dailynursestaffing_CY{{YYYY}}Q{{n}}.csv"
        )
    year, q = int(m.group(1)), int(m.group(2))
    label = f"CY{year}Q{q}"
    return year, q, label


def nurse_manifest_dir(quarter_label: str, root: Path | None = None) -> Path:
    return cms_data_paths.nurse_raw_dir(root) / "_manifests" / quarter_label


def resolve_cms_nurse_release(fetch_json: FetchJson | None = None) -> CmsNurseRelease:
    """Resolve current Primary CSV distribution from data-api /resources."""
    fetch = fetch_json or _default_fetch_json
    try:
        payload = fetch(RESOURCES_URL)
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
        raise AcquireError(f"CMS nurse resources fetch failed: {exc}") from exc

    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise AcquireError("CMS nurse resources response has no data list")

    primary = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("type") == "Primary" or row.get("media_bundle") == "primary_dataset_file":
            name = str(row.get("file_name") or "")
            if FILENAME_RE.match(name):
                primary = row
                break
    if primary is None:
        # Fallback: any CSV matching nurse filename pattern
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("file_name") or "")
            if FILENAME_RE.match(name) and str(row.get("file_url") or "").startswith("http"):
                primary = row
                break
    if primary is None:
        raise AcquireError("CMS nurse resources has no Primary PBJ_dailynursestaffing CSV")

    filename = str(primary["file_name"])
    url = str(primary.get("file_url") or "").strip()
    if not url:
        raise AcquireError("Primary nurse distribution missing file_url")
    year, quarter, label = parse_nurse_filename(filename)
    identity = f"{NURSE_DATASET_ID}|{filename}|{url}|{primary.get('file_uuid') or ''}"
    return CmsNurseRelease(
        dataset_id=NURSE_DATASET_ID,
        quarter_label=label,
        year=year,
        quarter=quarter,
        distribution_filename=filename,
        distribution_url=url,
        title=str(primary.get("title") or "") or None,
        file_size=int(primary["file_size"]) if primary.get("file_size") is not None else None,
        file_uuid=str(primary["file_uuid"]) if primary.get("file_uuid") else None,
        raw_fingerprint=_sha256_bytes(identity.encode("utf-8")),
    )


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:120]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def list_local_nurse_snapshots(root: Path | None = None) -> list[tuple[int, int, Path]]:
    d = cms_data_paths.nurse_raw_dir(root)
    out: list[tuple[int, int, Path]] = []
    if not d.is_dir():
        return out
    for path in sorted(d.glob("PBJ_dailynursestaffing_CY*.csv")):
        if _looks_like_lfs_pointer(path):
            continue
        try:
            year, q, _ = parse_nurse_filename(path.name)
        except AcquireError:
            continue
        out.append((year, q, path))
    return out


def latest_local_nurse(root: Path | None = None) -> Optional[tuple[int, int, Path]]:
    snaps = list_local_nurse_snapshots(root)
    if not snaps:
        return None
    return max(snaps, key=lambda t: (t[0], t[1]))


def load_acquisition_manifest(
    quarter_label: str,
    *,
    root: Path | None = None,
) -> Optional[dict[str, Any]]:
    path = nurse_manifest_dir(quarter_label, root) / "acquisition.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    sha = data.get("sha256")
    if not isinstance(sha, str) or len(sha) != 64:
        return None
    return data


def assess_local_release_identity(
    cms: CmsNurseRelease,
    *,
    root: Path | None = None,
    min_rows: int = 1000,
    min_bytes: int = 10_000,
) -> LocalIdentityResult:
    """Prove local raw identity against CMS + acquisition provenance when present.

    Cryptographic ``IDENTICAL`` requires a trustworthy ``acquisition.json`` whose
    SHA-256 matches the on-disk file, plus filename/quarter (and CMS size/uuid
    when available). Same filename alone never yields IDENTICAL / CURRENT.
    """
    dest = cms_data_paths.nurse_raw_dir(root) / cms.distribution_filename
    if not dest.is_file() or _looks_like_lfs_pointer(dest) or dest.stat().st_size == 0:
        return LocalIdentityResult(
            verdict=IDENTITY_MISSING,
            cryptographically_identical=False,
            path=None,
            local_sha256=None,
            manifest_sha256=None,
            detail="No usable local raw for CMS Primary filename",
        )

    local_size = dest.stat().st_size
    local_sha = _sha256_file(dest)
    manifest = load_acquisition_manifest(cms.quarter_label, root=root)

    if manifest is not None:
        m_sha = str(manifest["sha256"]).lower()
        m_name = str(manifest.get("source_filename") or "")
        m_quarter = str(manifest.get("quarter_label") or "")
        if m_name and m_name != cms.distribution_filename:
            return LocalIdentityResult(
                verdict=IDENTITY_CMS_DIVERGENCE,
                cryptographically_identical=False,
                path=dest,
                local_sha256=local_sha,
                manifest_sha256=m_sha,
                detail=(
                    f"Manifest filename {m_name!r} != CMS Primary "
                    f"{cms.distribution_filename!r}"
                ),
            )
        if m_quarter and m_quarter.upper() != cms.quarter_label.upper():
            return LocalIdentityResult(
                verdict=IDENTITY_CMS_DIVERGENCE,
                cryptographically_identical=False,
                path=dest,
                local_sha256=local_sha,
                manifest_sha256=m_sha,
                detail=f"Manifest quarter {m_quarter} != CMS {cms.quarter_label}",
            )
        if local_sha.lower() != m_sha:
            return LocalIdentityResult(
                verdict=IDENTITY_MANIFEST_MISMATCH,
                cryptographically_identical=False,
                path=dest,
                local_sha256=local_sha,
                manifest_sha256=m_sha,
                detail=(
                    "Local raw SHA-256 does not match acquisition.json; "
                    "refusing to treat as CURRENT"
                ),
            )
        if cms.file_size is not None and local_size != int(cms.file_size):
            return LocalIdentityResult(
                verdict=IDENTITY_CMS_DIVERGENCE,
                cryptographically_identical=False,
                path=dest,
                local_sha256=local_sha,
                manifest_sha256=m_sha,
                detail=(
                    f"Local size {local_size} != CMS-reported file_size {cms.file_size}"
                ),
            )
        m_uuid = manifest.get("cms_file_uuid")
        if cms.file_uuid and m_uuid and str(m_uuid) != str(cms.file_uuid):
            return LocalIdentityResult(
                verdict=IDENTITY_CMS_DIVERGENCE,
                cryptographically_identical=False,
                path=dest,
                local_sha256=local_sha,
                manifest_sha256=m_sha,
                detail="Manifest cms_file_uuid differs from current CMS file_uuid",
            )
        return LocalIdentityResult(
            verdict=IDENTITY_IDENTICAL,
            cryptographically_identical=True,
            path=dest,
            local_sha256=local_sha,
            manifest_sha256=m_sha,
            detail="Manifest SHA-256 matches local raw; CMS filename/quarter aligned",
        )

    # No trustworthy provenance: structural gate only — never cryptographic identical.
    try:
        validate_raw_nurse_csv(
            dest,
            expected_quarter=cms.quarter_label,
            min_rows=min_rows,
            min_bytes=min_bytes,
        )
    except AcquireError as exc:
        return LocalIdentityResult(
            verdict=IDENTITY_UNMANIFESTED_FAIL,
            cryptographically_identical=False,
            path=dest,
            local_sha256=local_sha,
            manifest_sha256=None,
            detail=f"Unmanifested raw failed structural validation: {exc}",
        )
    return LocalIdentityResult(
        verdict=IDENTITY_UNMANIFESTED_OK,
        cryptographically_identical=False,
        path=dest,
        local_sha256=local_sha,
        manifest_sha256=None,
        detail=(
            "Local raw structurally OK but no acquisition.json provenance; "
            "not cryptographically identical to CMS"
        ),
    )


def local_has_identical_release(
    cms: CmsNurseRelease,
    *,
    root: Path | None = None,
    expected_sha256: str | None = None,
    min_rows: int = 1000,
    min_bytes: int = 10_000,
) -> bool:
    """True only when local raw is cryptographically identical to the CMS release."""
    if expected_sha256:
        dest = cms_data_paths.nurse_raw_dir(root) / cms.distribution_filename
        if not dest.is_file() or _looks_like_lfs_pointer(dest) or dest.stat().st_size == 0:
            return False
        return _sha256_file(dest).lower() == expected_sha256.lower()
    return assess_local_release_identity(
        cms, root=root, min_rows=min_rows, min_bytes=min_bytes
    ).cryptographically_identical


def cms_is_newer_than_local(
    cms: CmsNurseRelease,
    root: Path | None = None,
    *,
    min_rows: int = 1000,
    min_bytes: int = 10_000,
) -> bool:
    """True when CMS Primary is not cryptographically present locally.

    Unmanifested structural-OK local files are not treated as possessing the
    release for CURRENT, but also do not count as a strictly newer CMS quarter
    when the same filename already exists (see probe for conservative status).
    """
    identity = assess_local_release_identity(
        cms, root=root, min_rows=min_rows, min_bytes=min_bytes
    )
    if identity.cryptographically_identical:
        return False
    if identity.verdict == IDENTITY_UNMANIFESTED_OK:
        return False
    if identity.verdict in {
        IDENTITY_MANIFEST_MISMATCH,
        IDENTITY_CMS_DIVERGENCE,
        IDENTITY_UNMANIFESTED_FAIL,
    }:
        # Local artifact for this name is wrong/untrusted — treat as needing acquire.
        return True
    local = latest_local_nurse(root)
    if local is None:
        return True
    ly, lq, _ = local
    return (cms.year, cms.quarter) > (ly, lq)


def _normalize_header(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.strip().lower())


def validate_raw_nurse_csv(
    path: Path,
    *,
    expected_quarter: str | None = None,
    min_rows: int = 1000,
    min_bytes: int = 10_000,
) -> dict[str, Any]:
    """Structural validation before standardization (fail closed)."""
    if not path.is_file() or path.stat().st_size == 0:
        raise AcquireError(f"raw nurse CSV missing or empty: {path}")
    if _looks_like_lfs_pointer(path):
        raise AcquireError(f"raw CSV is a Git LFS pointer, not data: {path}")
    size = path.stat().st_size
    if size < min_bytes:
        raise AcquireError(f"raw nurse CSV too small ({size} bytes)")

    year, quarter, label = parse_nurse_filename(path.name)
    if expected_quarter and label.upper() != expected_quarter.upper():
        raise AcquireError(
            f"filename quarter {label} does not match expected {expected_quarter}"
        )

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise AcquireError("CSV has no header") from exc
        if not header:
            raise AcquireError("CSV header empty")
        # Wrong dataset heuristics
        joined = ",".join(header).lower()
        if "provider information" in joined or "special focus status" in joined:
            raise AcquireError("artifact looks like Provider Info, not PBJ nurse staffing")
        if "nonnurse" in joined or "non_nurse" in joined:
            raise AcquireError("artifact looks like non-nurse staffing")

        norm_map = {_normalize_header(h): h for h in header}
        missing_groups = []
        resolved = {}
        for group in REQUIRED_COLUMN_ALIASES:
            found = None
            for alias in group:
                key = _normalize_header(alias)
                if key in norm_map:
                    found = norm_map[key]
                    break
            if not found:
                missing_groups.append(group[0])
            else:
                resolved[group[0]] = found
        if missing_groups:
            raise AcquireError(f"missing required columns: {missing_groups}")

        rows = 0
        for _ in reader:
            rows += 1
            if rows > min_rows + 5:
                # Enough to satisfy min; avoid full scan of 200MB+ in validate when huge
                # Still count remaining for report if file is modest
                if size > 5_000_000:
                    # Estimate remaining not needed — mark as at_least
                    break
        if size > 5_000_000 and rows <= min_rows + 5:
            # Continue counting with cheap line iteration
            for _ in reader:
                rows += 1

    if rows < min_rows:
        raise AcquireError(f"row count too low ({rows} < {min_rows})")

    return {
        "quarter_label": label,
        "year": year,
        "quarter": quarter,
        "row_count": rows,
        "row_count_complete": size <= 5_000_000,
        "columns": header,
        "resolved_columns": resolved,
        "sha256": _sha256_file(path),
        "byte_size": size,
    }


def download_nurse_csv(
    cms: CmsNurseRelease,
    dest: Path,
    *,
    fetch_bytes: FetchBytes | None = None,
    stream_download: StreamDownload | None = None,
    min_bytes: int = 10_000,
    validate: bool = True,
    min_rows: int = 1000,
) -> dict[str, Any]:
    """Stream CMS → temp file (chunked SHA-256), validate, then install atomically.

    If ``dest`` already exists with a different checksum, refuse overwrite and
    leave the existing file untouched. Partial failures delete temp artifacts only.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    staged: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(dest.parent),
            prefix=f".{dest.name}.",
            suffix=".partial",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)

        try:
            digest = stream_url_to_path(
                cms.distribution_url,
                tmp_path,
                fetch_bytes=fetch_bytes,
                stream_download=stream_download,
            )
        except AcquireError:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
            raise AcquireError(f"download failed: {exc}") from exc

        size = tmp_path.stat().st_size
        if size < min_bytes:
            tmp_path.unlink(missing_ok=True)
            raise AcquireError(f"download too small ({size} bytes)")

        with tmp_path.open("rb") as f:
            head = f.read(300).lstrip(b"\xef\xbb\xbf")
        if b"," not in head:
            tmp_path.unlink(missing_ok=True)
            raise AcquireError("download does not look like CSV")
        head_l = head.lower()
        if b"special focus status" in head_l:
            tmp_path.unlink(missing_ok=True)
            raise AcquireError("download looks like Provider Info CSV")

        if dest.is_file() and not _looks_like_lfs_pointer(dest):
            existing = _sha256_file(dest)
            if existing == digest:
                tmp_path.unlink(missing_ok=True)
                return {
                    "action": "skipped_identical",
                    "path": str(dest),
                    "sha256": existing,
                    "byte_size": dest.stat().st_size,
                }
            tmp_path.unlink(missing_ok=True)
            raise AcquireError(
                f"refusing overwrite of differing file {dest.name} "
                f"(existing={existing[:12]}… new={digest[:12]}…)"
            )

        staged = dest.parent / f".staging_{dest.name}"
        staged.unlink(missing_ok=True)
        tmp_path.replace(staged)
        tmp_path = None

        validation: dict[str, Any] | None = None
        if validate:
            if dest.exists():
                staged.unlink(missing_ok=True)
                raise AcquireError(f"unexpected existing dest during staging: {dest}")
            staged.replace(dest)
            staged = None
            try:
                validation = validate_raw_nurse_csv(
                    dest,
                    expected_quarter=cms.quarter_label,
                    min_rows=min_rows,
                    min_bytes=min_bytes,
                )
            except AcquireError:
                dest.unlink(missing_ok=True)
                raise
        else:
            staged.replace(dest)
            staged = None

        out: dict[str, Any] = {
            "action": "written",
            "path": str(dest),
            "sha256": digest,
            "byte_size": dest.stat().st_size,
            "acquired_at": datetime.now(timezone.utc).isoformat(),
        }
        if validation:
            out["validation"] = validation
        return out
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if staged is not None:
            staged.unlink(missing_ok=True)
        raise


def write_acquisition_record(
    quarter_label: str,
    record: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    d = nurse_manifest_dir(quarter_label, root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "acquisition.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def run_standardize_nurse(*, root: Path | None = None) -> int:
    """Delegate to existing standardize_pbj_files.py (no duplication)."""
    root = root or cms_data_paths.repo_root()
    script = root / "standardize_pbj_files.py"
    if not script.is_file():
        raise AcquireError(f"missing standardize_pbj_files.py at {script}")
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise AcquireError(
            f"standardize_pbj_files.py failed ({proc.returncode}): "
            f"{(proc.stderr or proc.stdout)[-2000:]}"
        )
    return int(proc.returncode)


def acquire_and_process(
    *,
    root: Path | None = None,
    fetch_json: FetchJson | None = None,
    fetch_bytes: FetchBytes | None = None,
    stream_download: StreamDownload | None = None,
    dry_run: bool = False,
    force: bool = False,
    skip_standardize: bool = False,
    min_rows: int = 1000,
    min_bytes: int = 10_000,
) -> dict[str, Any]:
    """CMS resources → raw PBJcsv → structural validate → standardize_pbj_files."""
    root = root or cms_data_paths.repo_root()
    cms = resolve_cms_nurse_release(fetch_json=fetch_json)
    dest = cms_data_paths.nurse_raw_dir(root) / cms.distribution_filename
    std = cms_data_paths.standardized_nurse_dir(root) / cms.distribution_filename
    report: dict[str, Any] = {
        "status": "UNKNOWN",
        "lifecycle": "DETECTED",
        "cms": asdict(cms),
        "quarter_label": cms.quarter_label,
        "dest": str(dest),
        "standardized": str(std),
        "cross_repo_write": False,
        "deployed": False,
        "metrics_promoted": False,
    }

    identity = assess_local_release_identity(
        cms, root=root, min_rows=min_rows, min_bytes=min_bytes
    )
    report["identity"] = {
        "verdict": identity.verdict,
        "cryptographically_identical": identity.cryptographically_identical,
        "detail": identity.detail,
        "local_sha256": identity.local_sha256,
        "manifest_sha256": identity.manifest_sha256,
    }

    if identity.verdict == IDENTITY_MANIFEST_MISMATCH and not force:
        report["status"] = "PROVENANCE_MISMATCH"
        report["lifecycle"] = "STRUCTURAL_FAIL"
        report["error"] = identity.detail
        raise AcquireError(identity.detail)

    if identity.cryptographically_identical and not force:
        report["lifecycle"] = "ACQUIRED"
        if std.is_file() and std.stat().st_size > 0:
            report["status"] = "CURRENT"
            report["lifecycle"] = "PROCESSED"
            report["note"] = (
                "PBJapp already possesses this CMS nurse quarter "
                "(manifest SHA + raw + standardized)"
            )
            return report
        if dry_run:
            report["status"] = "WOULD_STANDARDIZE"
            return report
        try:
            validation = validate_raw_nurse_csv(
                dest, expected_quarter=cms.quarter_label, min_rows=min_rows, min_bytes=min_bytes
            )
            report["validation"] = validation
            report["lifecycle"] = "STRUCTURAL_PASS"
        except AcquireError as exc:
            report["status"] = "STRUCTURAL_FAIL"
            report["error"] = str(exc)
            raise
        if not skip_standardize:
            run_standardize_nurse(root=root)
            report["lifecycle"] = "PROCESSED"
            report["status"] = "PROCESSED"
        else:
            report["status"] = "ACQUIRED_RAW_ONLY"
        return report

    if identity.verdict == IDENTITY_UNMANIFESTED_OK and not force:
        report["lifecycle"] = "STRUCTURAL_PASS"
        report["status"] = "LOCAL_UNMANIFESTED"
        report["note"] = identity.detail
        if dry_run:
            return report
        if not std.is_file() or std.stat().st_size == 0:
            if not skip_standardize:
                run_standardize_nurse(root=root)
                report["lifecycle"] = "PROCESSED"
                report["status"] = "LOCAL_UNMANIFESTED"
                report["note"] = (
                    identity.detail + "; standardized without claiming CMS identity"
                )
        return report

    if identity.verdict == IDENTITY_UNMANIFESTED_FAIL and not force:
        report["status"] = "STRUCTURAL_FAIL"
        report["lifecycle"] = "STRUCTURAL_FAIL"
        report["error"] = identity.detail
        raise AcquireError(identity.detail)

    if dry_run:
        report["status"] = "WOULD_ACQUIRE"
        report["lifecycle"] = "DETECTED"
        return report

    # Preserve existing on failure: stream validates before finalizing new dest;
    # differing existing dest is never overwritten.
    try:
        dl = download_nurse_csv(
            cms,
            dest,
            fetch_bytes=fetch_bytes,
            stream_download=stream_download,
            min_bytes=min_bytes,
            validate=True,
            min_rows=min_rows,
        )
        report["download"] = dl
        report["lifecycle"] = "ACQUIRED"
        validation = dl.get("validation")
        if not validation:
            validation = validate_raw_nurse_csv(
                dest,
                expected_quarter=cms.quarter_label,
                min_rows=min_rows,
                min_bytes=min_bytes,
            )
        report["validation"] = validation
        report["lifecycle"] = "STRUCTURAL_PASS"

        acquisition = {
            "stable_cms_dataset_id": NURSE_DATASET_ID,
            "quarter_label": cms.quarter_label,
            "source_filename": cms.distribution_filename,
            "source_url": cms.distribution_url,
            "cms_title": cms.title,
            "cms_file_uuid": cms.file_uuid,
            "cms_reported_file_size": cms.file_size,
            "byte_size": validation["byte_size"],
            "sha256": validation["sha256"],
            "acquired_at": datetime.now(timezone.utc).isoformat(),
            "raw_fingerprint": cms.raw_fingerprint,
            "validation": validation,
            "resources_url": RESOURCES_URL,
        }
        acq_path = write_acquisition_record(cms.quarter_label, acquisition, root=root)
        report["acquisition_path"] = str(acq_path)

        if not skip_standardize:
            run_standardize_nurse(root=root)
            if not std.is_file():
                report["status"] = "PROCESSING_FAILURE"
                report["lifecycle"] = "ACQUIRED"
                report["note"] = "standardize ran but standardized output missing"
                return report
            report["lifecycle"] = "PROCESSED"
            report["status"] = "PROCESSED"
            report["standardized_sha256"] = _sha256_file(std)
        else:
            report["status"] = "ACQUIRED_RAW_ONLY"
    except AcquireError:
        report["status"] = "ERROR"
        raise

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire current CMS PBJ nurse staffing CSV into PBJcsv/, validate, "
            "then run standardize_pbj_files.py. No pbj-root write / deploy."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-standardize", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--min-rows", type=int, default=1000)
    args = parser.parse_args(argv)

    try:
        report = acquire_and_process(
            dry_run=args.dry_run,
            force=args.force,
            skip_standardize=args.skip_standardize,
            min_rows=args.min_rows,
        )
    except AcquireError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status: {report.get('status')}")
        print(f"lifecycle: {report.get('lifecycle')}")
        cms = report.get("cms") or {}
        print(
            f"CMS: {cms.get('quarter_label')} file={cms.get('distribution_filename')} "
            f"id={cms.get('dataset_id')}"
        )
        if report.get("download"):
            print(f"download: {report['download']}")
        if report.get("acquisition_path"):
            print(f"acquisition: {report['acquisition_path']}")
    ok = report.get("status") in {
        "CURRENT",
        "PROCESSED",
        "WOULD_ACQUIRE",
        "WOULD_STANDARDIZE",
        "ACQUIRED_RAW_ONLY",
        "LOCAL_UNMANIFESTED",
    }
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
