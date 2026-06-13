"""
Lightweight "should we run this expensive Step-1 job?" checks for Vercel packaging.

General rule: compare **what is on disk** (national PBJ quarters, Citations file ages,
provider_info_combined mtime) to **what the facility slice already contains**. If there is
nothing new to ingest, skip calling the heavy builder so packaging stays a small refresh
when only app/template changed.

Callers pass ``smart_refresh=False`` (e.g. ``--force-refresh-data``) to disable all gates
and always run the underlying functions (safe, slower).

Quarter coverage for nurse/non-nurse slices can be read from a small sidecar
(``<facility.csv>.cy_qtr.json``) or ``<facility.csv>.meta.json`` (``cy_qtr_keys``) when
``slice_mtime`` matches the CSV. Stale or missing metadata triggers a full CY_Qtr column
scan and refreshes the sidecar (conservative).
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Optional, cast

import pandas as pd

import cms_data_paths

GateResult = tuple[bool, str]  # (needs_refresh, human_reason)

_SIDECAR_SCHEMA_VERSION = 1


def _normalize_cy_qtr(val: object) -> Optional[str]:
    """Normalize values like 2025Q3 / CY2025Q3 to CYyyyyQn."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper().replace("\ufeff", "")
    m = re.search(r"(?:CY)?(\d{4})Q([1-4])", s)
    if m:
        return f"CY{m.group(1)}Q{m.group(2)}"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        try:
            i = int(val)
        except (TypeError, ValueError):
            return None
        if 20101 <= i <= 20304 and (i % 10) in (1, 2, 3, 4):
            y, q = str(i // 10), str(i % 10)
            return f"CY{y}Q{q}"
    return None


def _quarter_from_nurse_filename(path: str) -> Optional[str]:
    m = re.search(r"CY(\d{4})Q(\d)", os.path.basename(path), re.IGNORECASE)
    return f"CY{m.group(1)}Q{m.group(2)}" if m else None


def _facility_csv_mtime(facility_csv: str) -> Optional[float]:
    try:
        return os.path.getmtime(facility_csv)
    except OSError:
        return None


def _facility_cy_qtr_sidecar_path(facility_csv: str) -> str:
    return f"{facility_csv}.cy_qtr.json"


def _read_cy_qtr_keys_from_json_meta(
    meta_path: str,
    facility_csv: str,
) -> Optional[set[str]]:
    """Read ``cy_qtr_keys`` when ``slice_mtime`` matches the facility CSV."""
    if not os.path.isfile(meta_path):
        return None
    csv_mtime = _facility_csv_mtime(facility_csv)
    if csv_mtime is None:
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None
    if not isinstance(meta, dict):
        return None
    try:
        recorded = float(meta.get("slice_mtime", -1))
    except (TypeError, ValueError):
        return None
    if abs(recorded - csv_mtime) > 1e-6:
        return None
    raw = meta.get("cy_qtr_keys")
    if not isinstance(raw, list):
        return None
    keys: set[str] = set()
    for item in raw:
        nq = _normalize_cy_qtr(item)
        if nq:
            keys.add(nq)
    return keys if keys else None


def write_facility_csv_cy_qtr_sidecar(facility_csv: str, keys: set[str]) -> None:
    """Persist quarter keys for fast packaging gates (best-effort; failures are ignored)."""
    csv_mtime = _facility_csv_mtime(facility_csv)
    if csv_mtime is None:
        return
    payload = {
        "schema_version": _SIDECAR_SCHEMA_VERSION,
        "slice_mtime": csv_mtime,
        "cy_qtr_keys": sorted(keys),
    }
    path = _facility_cy_qtr_sidecar_path(facility_csv)
    try:
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except OSError:
        pass


def _scan_cy_qtr_keys_from_csv(facility_csv: str) -> Optional[set[str]]:
    """Full CY_Qtr column scan (fallback when sidecar / non-nurse meta is missing or stale)."""
    try:
        peek = pd.read_csv(facility_csv, nrows=0, low_memory=False)
        cy_actual = next(
            (c for c in peek.columns if str(c).strip().replace("\ufeff", "").upper() == "CY_QTR"),
            None,
        )
        if not cy_actual:
            return None

        def _usecols(name: str) -> bool:
            return name == cy_actual

        frame = pd.read_csv(
            facility_csv,
            usecols=_usecols,
            low_memory=False,
            dtype=str,
        )
        series = cast(pd.Series, frame[cy_actual])
        keys: set[str] = set()
        for v in series.dropna().unique():
            nq = _normalize_cy_qtr(v)
            if nq:
                keys.add(nq)
        return keys
    except Exception:
        return None


def facility_csv_cy_qtr_keys(facility_csv: str) -> Optional[set[str]]:
    """Unique normalized CY quarters in a facility nurse or non-nurse CSV; None if unreadable."""
    if not os.path.isfile(facility_csv):
        return None

    cached = _read_cy_qtr_keys_from_json_meta(
        _facility_cy_qtr_sidecar_path(facility_csv),
        facility_csv,
    )
    if cached is not None:
        return cached

    cached = _read_cy_qtr_keys_from_json_meta(f"{facility_csv}.meta.json", facility_csv)
    if cached is not None:
        return cached

    keys = _scan_cy_qtr_keys_from_csv(facility_csv)
    if keys is not None:
        write_facility_csv_cy_qtr_sidecar(facility_csv, keys)
    return keys


def disk_nurse_quarters(repo_root: str) -> set[str]:
    """Quarter keys from ``standardized_PBJ/PBJ_dailynursestaffing_*.csv``."""
    std_dir = cms_data_paths.standardized_nurse_dir(cms_data_paths.optional_repo_root(repo_root))
    out: set[str] = set()
    for fp in glob.glob(str(std_dir / "PBJ_dailynursestaffing_*.csv")):
        q = _quarter_from_nurse_filename(fp)
        if q:
            out.add(q)
    return out


def disk_nonnurse_quarters(repo_root: str) -> set[str]:
    """Quarter keys from standardized + legacy NonNursecsv paths (same dedupe idea as nonnurse_staffing_lib)."""
    from nonnurse_staffing_lib import discover_standardized_nonnurse_csv_paths, quarter_from_pbj_staffing_filename

    nurse_files = list(discover_standardized_nonnurse_csv_paths(repo_root))
    legacy_dir = cms_data_paths.nonnurse_raw_dir(cms_data_paths.optional_repo_root(repo_root))
    if legacy_dir.is_dir():
        nurse_files.extend(str(p) for p in legacy_dir.glob("PBJ_dailynonnurse*.csv") if p.is_file())
    by_quarter: dict[str, str] = {}
    for fp in sorted(set(nurse_files)):
        q = quarter_from_pbj_staffing_filename(fp) or os.path.basename(fp).lower()
        prev = by_quarter.get(q)
        if prev is None:
            by_quarter[q] = fp
            continue
        try:
            if os.path.getsize(fp) >= os.path.getsize(prev):
                by_quarter[q] = fp
        except OSError:
            by_quarter[q] = fp
    out: set[str] = set()
    for fp in by_quarter.values():
        q = quarter_from_pbj_staffing_filename(fp)
        if q:
            out.add(q)
    return out


def gate_nurse_pbj_slice(repo_root: str, facility_csv: str) -> GateResult:
    if not os.path.isfile(facility_csv):
        return True, "facility nurse CSV missing"
    disk = disk_nurse_quarters(repo_root)
    if not disk:
        return False, "no nurse PBJ quarter files under standardized_PBJ/; leaving existing slice"
    got = facility_csv_cy_qtr_keys(facility_csv)
    if got is None:
        return True, "could not read CY_Qtr from existing nurse CSV; refreshing to be safe"
    pending = disk - got
    if not pending:
        return False, "slice already covers every nurse quarter file on disk"
    head = ", ".join(sorted(pending)[:8])
    tail = "…" if len(pending) > 8 else ""
    return True, f"disk has quarter file(s) not in slice: {head}{tail}"


def gate_nonnurse_pbj_slice(repo_root: str, facility_csv: str) -> GateResult:
    if not os.path.isfile(facility_csv):
        return True, "facility non-nurse CSV missing"
    disk = disk_nonnurse_quarters(repo_root)
    if not disk:
        return False, "no non-nurse PBJ quarter files found; leaving existing slice"
    got = facility_csv_cy_qtr_keys(facility_csv)
    if got is None:
        return True, "could not read CY_Qtr from existing non-nurse CSV; refreshing to be safe"
    pending = disk - got
    if pending:
        meta_path = f"{facility_csv}.meta.json"
        try:
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
                known_empty = meta.get("known_empty_quarters", {}) if isinstance(meta, dict) else {}
                if isinstance(known_empty, dict):
                    pending = {q for q in pending if q not in known_empty}
        except Exception:
            # Ignore metadata errors; gate should remain conservative.
            pass
    if not pending:
        return False, "slice covers disk quarters (remaining gaps are known-empty for this CCN)"
    head = ", ".join(sorted(pending)[:8])
    tail = "…" if len(pending) > 8 else ""
    return True, f"disk has quarter file(s) not in slice: {head}{tail}"


def gate_citations_slice(repo_root: str, citations_out: str) -> GateResult:
    from citation_lib import find_latest_nh_health_citations_csv

    src = find_latest_nh_health_citations_csv(repo_root)
    if not src or not os.path.isfile(src):
        return False, "no national NH_HealthCitations*.csv; skip citations rebuild"
    if not os.path.isfile(citations_out):
        return True, "facility citations slice missing"
    try:
        if os.path.getmtime(citations_out) >= os.path.getmtime(src):
            return False, "citations slice is up to date vs latest national Citations file (mtime)"
    except OSError:
        return True, "could not compare mtimes; rebuilding citations"
    return True, "national Citations file is newer than facility slice"


def gate_provider_info_slice(repo_root: str, provider_out: str) -> GateResult:
    combined = os.path.join(repo_root, "provider_info_combined.csv")
    if os.path.isfile(combined):
        if not os.path.isfile(provider_out):
            return True, "provider slice missing"
        try:
            if os.path.getmtime(provider_out) >= os.path.getmtime(combined):
                return False, "provider slice is up to date vs provider_info_combined.csv (mtime)"
        except OSError:
            return True, "could not compare mtimes; refreshing provider slice"
        return True, "provider_info_combined.csv is newer than facility slice"
    norm_dir = cms_data_paths.provider_info_normalized_dir(cms_data_paths.optional_repo_root(repo_root))
    if not norm_dir.is_dir():
        return False, "no provider_info_combined.csv or provider_info_normalized/; skip gate"
    norms = glob.glob(str(norm_dir / "ProviderInfoNorm_*.csv"))
    if not norms:
        return False, "no ProviderInfoNorm_*.csv files; skip gate"
    try:
        latest_m = max(os.path.getmtime(p) for p in norms)
    except OSError:
        return True, "could not stat normalized provider files; refreshing"
    if os.path.isfile(provider_out) and os.path.getmtime(provider_out) >= latest_m:
        return False, "provider slice is up to date vs newest ProviderInfoNorm_*.csv (mtime)"
    return True, "normalized provider file(s) newer than facility slice"


def disk_ein_extractable_quarters(repo_root: str) -> set[str]:
    """
    CY quarters available from national EIN sources (monolithic PUF + ``EIN/quarters/``).
    """
    from facility_ein_lib import (
        collect_ein_detail_jobs,
        discover_ein_quarter_range_on_disk,
        ein_cy_quarter_from_ein_source_name,
        resolve_ein_primary_zip,
    )

    root = str(Path(repo_root).resolve())
    puf = resolve_ein_primary_zip(root) or ""
    lo, hi = discover_ein_quarter_range_on_disk(root, puf)
    if lo is None or hi is None:
        return set()
    out: set[str] = set()
    for zp, member in collect_ein_detail_jobs(root, puf, lo, hi):
        cy = ein_cy_quarter_from_ein_source_name(member) or ein_cy_quarter_from_ein_source_name(
            os.path.basename(zp)
        )
        if cy:
            nq = _normalize_cy_qtr(cy)
            if nq:
                out.add(nq)
    return out


def facility_ein_job_quarter_keys(repo_root: str, provnum: str) -> Optional[set[str]]:
    """CY quarters present in facility EIN job_quarterly artifact (parquet or csv)."""
    from file_path_utils import find_facility_ein_table_base
    from facility_ein_lib import normalize_cy_qtr_ein, read_facility_ein_parquet_or_csv

    prov = str(provnum).strip().zfill(6)
    base = find_facility_ein_table_base(prov, "job_quarterly")
    if not base:
        deploy_base = str(
            cms_data_paths.facility_deploy_dir(prov, cms_data_paths.optional_repo_root(repo_root))
            / f"facility_{prov}_ein_job_quarterly"
        )
        if Path(f"{deploy_base}.parquet").is_file() or Path(f"{deploy_base}.csv").is_file():
            base = deploy_base
        else:
            return None
    df = read_facility_ein_parquet_or_csv(base)
    if df is None or df.empty:
        return set()
    qcol = next((c for c in df.columns if str(c).strip().upper() in ("CY_QTR", "QUARTER")), None)
    if not qcol:
        return None
    keys: set[str] = set()
    for raw in df[qcol].dropna().unique():
        nq = normalize_cy_qtr_ein(raw) or _normalize_cy_qtr(raw)
        if nq:
            keys.add(nq)
    return keys


def gate_ein_slice(repo_root: str, provnum: str) -> GateResult:
    """True when national EIN has quarters not yet in the facility deploy bundle."""
    disk = disk_ein_extractable_quarters(repo_root)
    if not disk:
        return False, "no national EIN sources on disk; leaving existing EIN bundle"
    got = facility_ein_job_quarter_keys(repo_root, provnum)
    if got is None:
        return True, "facility EIN job_quarterly missing; extract from national sources"
    pending = disk - got
    if not pending:
        return False, "EIN slice covers every extractable national quarter"
    head = ", ".join(sorted(pending)[:8])
    tail = "…" if len(pending) > 8 else ""
    return True, f"national EIN has quarter(s) not in deploy slice: {head}{tail}"


def refresh_ein_slice(
    repo_root: str,
    provnum: str,
    *,
    force: bool = False,
    merge: bool = True,
) -> tuple[bool, str]:
    """
    Run ``scripts/extract_facility_ein_from_zip.py`` when gate says pending quarters exist.

    Returns (ran_extract, message).
    """
    import subprocess
    import sys

    prov = str(provnum).strip().zfill(6)
    need, why = (True, "forced refresh") if force else gate_ein_slice(repo_root, prov)
    if not need:
        return False, why

    disk = disk_ein_extractable_quarters(repo_root)
    got = facility_ein_job_quarter_keys(repo_root, provnum) or set()
    pending = sorted(disk - got) if got else sorted(disk)
    if not pending and not force:
        return False, why

    script = Path(repo_root) / "scripts" / "extract_facility_ein_from_zip.py"
    if not script.is_file():
        return False, f"EIN refresh needed ({why}) but missing {script}"

    cmd = [sys.executable, str(script), prov, "--skip-supplemental-validation"]
    if merge and got:
        cmd.append("--merge-with-existing")
        cmd.extend(["--min-quarter", pending[0], "--max-quarter", pending[-1]])
    print(f"  [EIN] Running extract for {prov}: {why}", flush=True)
    rc = subprocess.call(cmd, cwd=str(repo_root))
    if rc != 0:
        return False, f"EIN extract failed (exit {rc}); {why}"
    return True, why
