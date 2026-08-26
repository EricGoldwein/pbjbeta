#!/usr/bin/env python3
"""Acquire current CMS Provider Information (4pq5-n9py) into PBJapp raw storage.

This is the missing upstream link for the Provider Info pilot:

  CMS metastore → safe download → existing normalize_provider_info.py → handoff

Does not write to pbj-root. Does not reimplement normalization.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
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
import cms_provider_release_lib as cpr  # noqa: E402
from nh_provider_column_map import NH_TO_NORM  # noqa: E402

PROVIDER_INFO_DATASET_ID = "4pq5-n9py"
METASTORE_URL = (
    "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/"
    f"{PROVIDER_INFO_DATASET_ID}?show-reference-ids=true"
)

MONTH_ABBR_TO_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Core CMS headers expected before normalize (values from NH_TO_NORM).
REQUIRED_NH_COLUMNS = (
    NH_TO_NORM["ccn"],
    NH_TO_NORM["provider_name"],
    NH_TO_NORM["state"],
    NH_TO_NORM["processing_date"],
)

FetchJson = Callable[[str], Any]
FetchBytes = Callable[[str], bytes]


@dataclass(frozen=True)
class CmsProviderInfoRelease:
    dataset_id: str
    released: Optional[str]
    modified: Optional[str]
    next_update_date: Optional[str]
    distribution_filename: str
    distribution_url: str
    data_vintage_label: str  # e.g. "Aug 2026"
    year: int
    month: int
    raw_fingerprint: str


class AcquireError(Exception):
    """Fail-closed acquisition / validation error."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _default_fetch_json(url: str) -> Any:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PBJapp-provider-info-acquire/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _default_fetch_bytes(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "PBJapp-provider-info-acquire/1.0"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = resp.read()
    if not data:
        raise AcquireError(f"empty download from {url}")
    return data


def parse_provider_info_filename(filename: str) -> tuple[int, int, str]:
    """Return (year, month, MonYYYY label) from NH_ProviderInfo_Aug2026.csv."""
    base = Path(filename).name
    m = re.search(r"NH_ProviderInfo_([A-Za-z]{3})(\d{4})\.csv$", base, re.I)
    if not m:
        raise AcquireError(f"unrecognized Provider Info filename: {filename}")
    abbr = m.group(1).capitalize()
    year = int(m.group(2))
    month = MONTH_ABBR_TO_NUM.get(abbr.lower())
    if not month:
        raise AcquireError(f"unrecognized month in filename: {filename}")
    return year, month, f"{abbr}{year}"


def resolve_cms_provider_info_release(
    *,
    fetch_json: FetchJson | None = None,
) -> CmsProviderInfoRelease:
    """Resolve current Provider Information distribution via stable dataset ID."""
    fetch = fetch_json or _default_fetch_json
    try:
        payload = fetch(METASTORE_URL)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        raise AcquireError(f"CMS metastore fetch failed: {exc}") from exc

    dists = payload.get("distribution") or []
    chosen = None
    for dist in dists:
        data = dist.get("data") if isinstance(dist.get("data"), dict) else dist
        if not isinstance(data, dict):
            continue
        url = data.get("downloadURL") or data.get("downloadUrl") or ""
        if not url:
            continue
        name = Path(str(url).split("?")[0]).name
        if re.search(r"NH_ProviderInfo_.*\.csv$", name, re.I):
            chosen = (name, url, data)
            break
        if chosen is None and str(data.get("mediaType", "")).startswith("text/csv"):
            chosen = (name, url, data)
    if chosen is None:
        raise AcquireError("CMS metastore response has no Provider Info CSV distribution")

    filename, url, _data = chosen
    year, month, _label = parse_provider_info_filename(filename)
    released = payload.get("released")
    modified = payload.get("modified")
    next_update = payload.get("nextUpdateDate") or payload.get("next_update_date")
    vintage = f"{_MONTH_ABBR[month]} {year}"
    fp_src = "|".join(
        [
            PROVIDER_INFO_DATASET_ID,
            str(filename),
            str(url),
            str(released or ""),
            str(modified or ""),
        ]
    )
    return CmsProviderInfoRelease(
        dataset_id=PROVIDER_INFO_DATASET_ID,
        released=str(released) if released else None,
        modified=str(modified) if modified else None,
        next_update_date=str(next_update) if next_update else None,
        distribution_filename=filename,
        distribution_url=url,
        data_vintage_label=vintage,
        year=year,
        month=month,
        raw_fingerprint=_sha256_bytes(fp_src.encode("utf-8")),
    )


_MONTH_ABBR = (
    "",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def list_local_provider_info_snapshots(root: Path | None = None) -> list[tuple[int, int, Path]]:
    """Real (non-LFS-pointer) NH_ProviderInfo_*.csv under provider_info/."""
    root = root or cms_data_paths.repo_root()
    d = cms_data_paths.provider_info_dir(root)
    out: list[tuple[int, int, Path]] = []
    if not d.is_dir():
        return out
    for path in sorted(d.glob("NH_ProviderInfo_*.csv")):
        if _looks_like_lfs_pointer(path):
            continue
        try:
            year, month, _ = parse_provider_info_filename(path.name)
        except AcquireError:
            continue
        out.append((year, month, path))
    return out


def _looks_like_lfs_pointer(path: Path) -> bool:
    try:
        head = path.read_bytes()[:120]
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def latest_local_provider_info(root: Path | None = None) -> Optional[tuple[int, int, Path]]:
    snaps = list_local_provider_info_snapshots(root)
    if not snaps:
        return None
    return max(snaps, key=lambda t: (t[0], t[1]))


def cms_is_newer_than_local(
    cms: CmsProviderInfoRelease,
    root: Path | None = None,
) -> bool:
    local = latest_local_provider_info(root)
    if local is None:
        # Also treat presence of matching filename (even LFS stub) as possessed for CURRENT check
        dest = (
            cms_data_paths.provider_info_dir(root)
            / f"NH_ProviderInfo_{_MONTH_ABBR[cms.month]}{cms.year}.csv"
        )
        if dest.is_file() and not _looks_like_lfs_pointer(dest) and dest.stat().st_size > 0:
            return False
        if dest.is_file() and _looks_like_lfs_pointer(dest):
            # Stub only — CMS file not actually possessed
            return True
        return True
    ly, lm, _ = local
    return (cms.year, cms.month) > (ly, lm)


def validate_raw_provider_info_csv(
    path: Path,
    *,
    prior_path: Path | None = None,
    min_rows: int = 1000,
    max_row_delta_ratio: float = 0.15,
) -> dict[str, Any]:
    """Fail-closed pre-normalize validation for NH_ProviderInfo CSV."""
    if not path.is_file() or path.stat().st_size == 0:
        raise AcquireError(f"raw CSV missing or empty: {path}")
    if _looks_like_lfs_pointer(path):
        raise AcquireError(f"raw CSV is a Git LFS pointer, not data: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise AcquireError("CSV has no header row")
        cols = [c.strip() for c in reader.fieldnames]
        missing = [c for c in REQUIRED_NH_COLUMNS if c not in cols]
        if missing:
            raise AcquireError(f"missing required columns: {missing}")

        rows = 0
        ccns: list[str] = []
        bad_ccn = 0
        for row in reader:
            rows += 1
            ccn_raw = str(row.get(REQUIRED_NH_COLUMNS[0], "")).strip()
            ccn = re.sub(r"\.0$", "", ccn_raw).strip().upper()
            # CMS Provider Info includes a small set of alphanumeric CCNs (e.g. 05A024).
            if not re.fullmatch(r"[0-9A-Z]{5,6}", ccn):
                bad_ccn += 1
            else:
                if ccn.isdigit():
                    ccn = ccn.zfill(6)
                ccns.append(ccn)

    if rows < min_rows:
        raise AcquireError(f"row count too low ({rows} < {min_rows})")
    if bad_ccn > max(10, int(rows * 0.01)):
        raise AcquireError(f"too many invalid CCNs: {bad_ccn}/{rows}")

    unique = len(set(ccns))
    dupes = rows - unique
    prior_rows = None
    if prior_path and prior_path.is_file() and not _looks_like_lfs_pointer(prior_path):
        with prior_path.open("r", encoding="utf-8-sig", newline="") as f:
            prior_rows = max(0, sum(1 for _ in csv.reader(f)) - 1)
        if prior_rows > 0:
            delta = abs(rows - prior_rows) / prior_rows
            if delta > max_row_delta_ratio:
                raise AcquireError(
                    f"abnormal row-count change vs prior ({prior_rows} -> {rows}, "
                    f"delta={delta:.1%} > {max_row_delta_ratio:.0%})"
                )

    return {
        "row_count": rows,
        "unique_ccn": unique,
        "duplicate_ccn_rows": dupes,
        "invalid_ccn_rows": bad_ccn,
        "prior_row_count": prior_rows,
        "columns": cols,
        "sha256": _sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def download_provider_info_csv(
    cms: CmsProviderInfoRelease,
    dest: Path,
    *,
    fetch_bytes: FetchBytes | None = None,
) -> dict[str, Any]:
    """Download CMS CSV to dest (atomic). Fail closed on empty/partial."""
    fetch = fetch_bytes or _default_fetch_bytes
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = fetch(cms.distribution_url)
    except (urllib.error.URLError, TimeoutError, OSError, AcquireError) as exc:
        raise AcquireError(f"download failed: {exc}") from exc
    if len(data) < 10_000:
        raise AcquireError(f"download too small ({len(data)} bytes) — refusing to install")
    # Must look like CSV text
    head = data[:200].lstrip(b"\xef\xbb\xbf")
    if b"," not in head and b"CMS Certification Number" not in head:
        raise AcquireError("download does not look like Provider Info CSV")

    digest = _sha256_bytes(data)
    with tempfile.NamedTemporaryFile(
        dir=str(dest.parent),
        prefix=f".{dest.name}.",
        suffix=".partial",
        delete=False,
    ) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)
    try:
        # Prefer not to clobber a different real file
        if dest.is_file() and not _looks_like_lfs_pointer(dest):
            existing = _sha256_file(dest)
            if existing != digest:
                raise AcquireError(
                    f"refusing overwrite of differing file {dest.name} "
                    f"(existing={existing[:12]}… new={digest[:12]}…)"
                )
            tmp_path.unlink(missing_ok=True)
            return {
                "action": "skipped_identical",
                "path": str(dest),
                "sha256": existing,
                "byte_size": dest.stat().st_size,
            }
        tmp_path.replace(dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return {
        "action": "written",
        "path": str(dest),
        "sha256": digest,
        "byte_size": dest.stat().st_size,
        "acquired_at": datetime.now(timezone.utc).isoformat(),
    }


def build_month_over_month_delta(
    prior_path: Path | None,
    current_path: Path,
) -> dict[str, Any]:
    """Compact Jul→Aug style delta for Provider Info (PBJapp-side)."""
    if prior_path is None or not prior_path.is_file() or _looks_like_lfs_pointer(prior_path):
        return {"status": "NO_PRIOR", "note": "no real prior NH snapshot for comparison"}

    name_col = REQUIRED_NH_COLUMNS[1]
    ccn_col = REQUIRED_NH_COLUMNS[0]
    status_cols = [
        c
        for c in (
            "Provider Status",
            "Provider Status Code",
            "Overall Rating",
            "Termination Code",
            "Provider Type",
        )
        if True
    ]

    def _load(path: Path) -> dict[str, dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows: dict[str, dict[str, str]] = {}
            for row in reader:
                ccn = re.sub(r"\.0$", "", str(row.get(ccn_col, "")).strip()).zfill(6)
                if re.fullmatch(r"\d{6}", ccn):
                    rows[ccn] = {k: str(v) for k, v in row.items()}
            return rows

    prior = _load(prior_path)
    curr = _load(current_path)
    added = sorted(set(curr) - set(prior))
    removed = sorted(set(prior) - set(curr))
    name_changes = []
    status_changes = []
    for ccn in sorted(set(curr) & set(prior)):
        pn = prior[ccn].get(name_col, "")
        cn = curr[ccn].get(name_col, "")
        if pn.strip() != cn.strip():
            name_changes.append({"ccn": ccn, "from": pn, "to": cn})
        for sc in status_cols:
            if sc in prior[ccn] or sc in curr[ccn]:
                if prior[ccn].get(sc, "") != curr[ccn].get(sc, ""):
                    status_changes.append(
                        {
                            "ccn": ccn,
                            "field": sc,
                            "from": prior[ccn].get(sc, ""),
                            "to": curr[ccn].get(sc, ""),
                        }
                    )

    prior_cols = set(next(iter(prior.values())).keys()) if prior else set()
    curr_cols = set(next(iter(curr.values())).keys()) if curr else set()
    return {
        "status": "OK",
        "prior_file": prior_path.name,
        "current_file": current_path.name,
        "prior_providers": len(prior),
        "current_providers": len(curr),
        "providers_added": len(added),
        "providers_removed": len(removed),
        "providers_added_sample": added[:25],
        "providers_removed_sample": removed[:25],
        "facility_name_changes": len(name_changes),
        "facility_name_changes_sample": name_changes[:25],
        "status_field_changes": len(status_changes),
        "status_field_changes_sample": status_changes[:25],
        "schema_columns_added": sorted(curr_cols - prior_cols),
        "schema_columns_removed": sorted(prior_cols - curr_cols),
        "row_count_delta": len(curr) - len(prior),
    }


def write_acquisition_record(
    key: cpr.ReleaseKey,
    record: dict[str, Any],
    root: Path | None = None,
) -> Path:
    mdir = cpr.manifest_dir(key, root)
    mdir.mkdir(parents=True, exist_ok=True)
    out = mdir / "acquisition.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=False)
        f.write("\n")
    return out


def run_normalize_for_file(nh_basename: str, *, force: bool = False) -> int:
    """Invoke existing normalize_provider_info.py (no reimplementation)."""
    import subprocess

    cmd = [sys.executable, str(_ROOT / "normalize_provider_info.py"), "--file", nh_basename]
    if force:
        cmd.append("--force")
    proc = subprocess.run(
        cmd,
        cwd=str(_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        sys.stderr.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return int(proc.returncode)


def acquire_and_process(
    *,
    root: Path | None = None,
    fetch_json: FetchJson | None = None,
    fetch_bytes: FetchBytes | None = None,
    dry_run: bool = False,
    force: bool = False,
    skip_normalize: bool = False,
) -> dict[str, Any]:
    """Full Provider Info pilot: detect → download → validate → normalize → handoff."""
    root = root or cms_data_paths.repo_root()
    cms = resolve_cms_provider_info_release(fetch_json=fetch_json)
    key = cpr.release_key(cms.year, cms.month)
    dest = (
        cms_data_paths.provider_info_dir(root)
        / f"NH_ProviderInfo_{_MONTH_ABBR[cms.month]}{cms.year}.csv"
    )
    report: dict[str, Any] = {
        "status": "UNKNOWN",
        "cms": asdict(cms),
        "release_key": key.label,
        "dest": str(dest),
        "ready_for_public_handoff": False,
    }

    newer = cms_is_newer_than_local(cms, root) or force
    if not newer and dest.is_file() and not _looks_like_lfs_pointer(dest):
        report["status"] = "CURRENT"
        report["note"] = "PBJapp already possesses this CMS Provider Info vintage"
        return report

    if dry_run:
        report["status"] = "WOULD_ACQUIRE" if newer or force else "CURRENT"
        return report

    # Prior real snapshot for MoM + row sanity
    snaps = list_local_provider_info_snapshots(root)
    prior_path = None
    older = [(y, m, p) for y, m, p in snaps if (y, m) < (cms.year, cms.month)]
    if older:
        prior_path = max(older, key=lambda t: (t[0], t[1]))[2]

    dl = download_provider_info_csv(cms, dest, fetch_bytes=fetch_bytes)
    report["download"] = dl

    validation = validate_raw_provider_info_csv(dest, prior_path=prior_path)
    report["validation"] = validation

    acquisition = {
        "stable_cms_dataset_id": PROVIDER_INFO_DATASET_ID,
        "cms_released": cms.released,
        "cms_modified": cms.modified,
        "cms_next_update_date": cms.next_update_date,
        "source_filename": cms.distribution_filename,
        "source_url": cms.distribution_url,
        "byte_size": validation["byte_size"],
        "sha256": validation["sha256"],
        "acquired_at": datetime.now(timezone.utc).isoformat(),
        "raw_fingerprint": cms.raw_fingerprint,
        "release_key": key.label,
        "validation": validation,
    }
    acq_path = write_acquisition_record(key, acquisition, root=root)
    report["acquisition_path"] = str(acq_path)

    delta = build_month_over_month_delta(prior_path, dest)
    cpr.write_release_diff(key, {"provider_info_delta": delta}, root=root)
    report["delta"] = delta

    if not skip_normalize:
        rc = run_normalize_for_file(dest.name, force=True)
        if rc != 0:
            raise AcquireError(f"normalize_provider_info.py failed with exit {rc}")
        cpr.update_manifest_normalized_outputs(key, root)
        # Minimal release_manifest if none from zip ingest
        if not cpr.load_manifest(key, root):
            cpr.write_manifest(
                key,
                {
                    "release_key": key.label,
                    "source": "cms_provider_data_metastore",
                    "dataset_id": PROVIDER_INFO_DATASET_ID,
                    "provider_info_csv": dest.name,
                    "acquisition": acquisition,
                    "extracted_at": datetime.now(timezone.utc).isoformat(),
                },
                root=root,
            )
        handoff_path = cpr.write_pbj_root_handoff(key, root)
        report["handoff_path"] = str(handoff_path)
        handoff = json.loads(Path(handoff_path).read_text(encoding="utf-8"))
        report["pbj_root_sync"] = handoff.get("pbj_root_sync")
        report["ready_for_public_handoff"] = bool(
            (handoff.get("provider_promotion") or {}).get("ready_for_pbj_commit")
        )
        # Provider-Info-only acquire may lack intervals/ownership; still mark handoff when Norm exists
        norm = (
            cms_data_paths.provider_info_normalized_dir(root)
            / f"ProviderInfoNorm_{cms.year}_{cms.month:02d}.csv"
        )
        if norm.is_file() and validation["sha256"]:
            report["status"] = "READY_FOR_PUBLIC_HANDOFF"
            report["ready_for_public_handoff"] = True
            report["normalized"] = {
                "path": str(norm),
                "sha256": _sha256_file(norm),
                "row_count": max(0, sum(1 for _ in norm.open(encoding="utf-8")) - 1),
            }
        else:
            report["status"] = "ACQUIRED_NORMALIZE_INCOMPLETE"
    else:
        report["status"] = "ACQUIRED_RAW_ONLY"

    report["cross_repo_write"] = False
    report["deployed"] = False
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire current CMS Provider Information (4pq5-n9py) into PBJapp, "
            "then run existing normalize + handoff. Does not write to pbj-root."
        )
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-download even if vintage present")
    parser.add_argument("--skip-normalize", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    try:
        report = acquire_and_process(
            dry_run=args.dry_run,
            force=args.force,
            skip_normalize=args.skip_normalize,
        )
    except AcquireError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"status: {report.get('status')}")
        cms = report.get("cms") or {}
        print(
            f"CMS: {cms.get('data_vintage_label')} "
            f"({cms.get('distribution_filename')}) dataset={cms.get('dataset_id')}"
        )
        if report.get("download"):
            print(f"download: {report['download']}")
        if report.get("normalized"):
            print(f"normalized: {report['normalized']}")
        if report.get("handoff_path"):
            print(f"handoff: {report['handoff_path']}")
        if report.get("delta"):
            d = report["delta"]
            print(
                f"delta: added={d.get('providers_added')} removed={d.get('providers_removed')} "
                f"name_changes={d.get('facility_name_changes')} status={d.get('status')}"
            )
        print(f"ready_for_public_handoff: {report.get('ready_for_public_handoff')}")
    return 0 if report.get("status") in {"CURRENT", "READY_FOR_PUBLIC_HANDOFF", "WOULD_ACQUIRE", "ACQUIRED_RAW_ONLY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
