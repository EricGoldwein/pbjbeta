"""
PBJ Employee Detail (CMS PUF) helpers.

ZIP layout: ``EIN/Payroll Based Journal Employee Detail Nursing Home Staffing.zip`` with
folders ``YYYY-Qn/PBJ_employeedetail_CY*.csv``. Additional quarter releases may be placed as
flat zips under ``EIN/supplemental/*.zip`` (e.g. ``PBJ_Employee_Detail_*_Q3_2025.zip``).

**Ongoing workflow:** add new supplemental zips under ``EIN/supplemental/`` (keep the CMS filename).
To refresh one facility without re-scanning all history, run
``scripts/extract_facility_ein_from_zip.py <CCN> --min-quarter CYyyyyQn --max-quarter CYyyyyQn
--merge-with-existing`` (writes merged detail + aggregates under ``deployments/pbj320-<CCN>/``).
For a full rebuild of all quarters visible on disk, run the same script **without** narrow
min/max (slower, no merge needed).

**Finding the monolithic PUF** (no manual rename required if you use env or a single zip in ``EIN/``):

- ``PBJ_EIN_DETAIL_ZIP``, ``CMS_EIN_DETAIL_ZIP``, or ``EIN_DETAIL_ZIP`` — absolute path to the zip.
- ``PBJ_EIN_ARCHIVE_DIR`` — directory to scan for ``*.zip`` (same heuristics as ``EIN/``).
- Any ``*.zip`` directly under ``EIN/`` that contains the CMS tree, or **exactly one** ``*.zip``
  in ``EIN/`` (treated as the PUF even if renamed).

Field definitions align with C:\\Users\\egold\\PycharmProjects\\EIN\\data_dictionary.md
(PROVNUM, CY_Qtr, WorkDate, SYS_EMPLEE_ID, EMPLEE_JOB_CD_ID, EMP_CTR, WORK_HRS_NUM, ...).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
import glob
import io
import json
import os
import re
import urllib.parse
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Iterator, cast

import numpy as np
import pandas as pd

# (title, category) per EMPLEE_JOB_CD_ID — from data_dictionary.md job code table
JOB_CODE_INFO: dict[int, tuple[str, str]] = {
    1: ("Administrator", "Administration"),
    2: ("Medical Director", "Medical Leadership"),
    3: ("Other Physician", "Medical"),
    4: ("Physician Assistant", "Medical"),
    5: ("RN Director of Nursing", "Nursing Leadership"),
    6: ("RN with Administrative Duties", "Nursing Leadership"),
    7: ("Registered Nurse", "Nursing"),
    8: ("LPN with Administrative Duties", "Nursing Leadership"),
    9: ("Licensed Practical/Vocational Nurse", "Nursing"),
    10: ("Certified Nurse Aide", "Direct Care"),
    11: ("Nurse Aide in Training", "Direct Care"),
    12: ("Medication Aide/Technician", "Direct Care"),
    13: ("Nurse Practitioner", "Medical"),
    14: ("Clinical Nurse Specialist", "Medical"),
    15: ("Pharmacist", "Support Services"),
    16: ("Dietitian", "Support Services"),
    17: ("Food Service Worker", "Support Services"),
    18: ("Occupational Therapist", "Therapy"),
    19: ("Occupational Therapy Assistant", "Therapy"),
    20: ("Occupational Therapy Aide", "Therapy"),
    21: ("Physical Therapist", "Therapy"),
    22: ("Physical Therapy Assistant", "Therapy"),
    23: ("Physical Therapy Aide", "Therapy"),
    24: ("Respiratory Therapist", "Therapy"),
    25: ("Respiratory Therapy Technician", "Therapy"),
    26: ("Speech/Language Pathologist", "Therapy"),
    27: ("Therapeutic Recreation Specialist", "Activities"),
    28: ("Qualified Activities Professional", "Activities"),
    29: ("Other Activities Staff", "Activities"),
    30: ("Qualified Social Worker", "Social Services"),
    31: ("Other Social Worker", "Social Services"),
    34: ("Mental Health Service Worker", "Social Services"),
}


# Short labels for dense UI (nursing-focused abbreviations where common).
JOB_TITLE_SHORT: dict[int, str] = {
    1: "Admin",
    2: "Med Dir",
    3: "Phys (oth)",
    4: "PA",
    5: "RN DON",
    6: "RN Admin",
    7: "RN",
    8: "LPN Admin",
    9: "LPN",
    10: "CNA",
    11: "Aide Trn",
    12: "Med Aide",
    13: "NP",
    14: "CNS",
    15: "Pharm",
    16: "Diet",
    17: "Food Svc",
    18: "OT",
    19: "OTA",
    20: "OT Aide",
    21: "PT",
    22: "PTA",
    23: "PT Aide",
    24: "RT",
    25: "RT Tech",
    26: "SLP",
    27: "Rec Ther",
    28: "Act Prof",
    29: "Act Oth",
    30: "SW",
    31: "SW Oth",
    34: "MH Svc",
}


def job_title(code: int | float) -> str:
    """Return job title for EMPLEE_JOB_CD_ID, or 'Unknown (n)'."""
    try:
        c = int(float(code))
    except (TypeError, ValueError):
        return "Unknown"
    if c in JOB_CODE_INFO:
        return JOB_CODE_INFO[c][0]
    return f"Unknown ({c})"


def job_title_short(code: int | float) -> str:
    """Abbreviated role label for tables and filters."""
    try:
        c = int(float(code))
    except (TypeError, ValueError):
        return "?"
    if c in JOB_TITLE_SHORT:
        return JOB_TITLE_SHORT[c]
    if c in JOB_CODE_INFO:
        t = JOB_CODE_INFO[c][0]
        return t if len(t) <= 14 else (t[:12] + "…")
    return f"?({c})"


def job_category(code: int | float) -> str:
    """Return rollup category for EMPLEE_JOB_CD_ID."""
    try:
        c = int(float(code))
    except (TypeError, ValueError):
        return "Other"
    if c in JOB_CODE_INFO:
        return JOB_CODE_INFO[c][1]
    return "Other"


# Single-day / daily PBJ metric keys → Employee Detail job codes (same CMS dictionary).
# Daily "Hrs_RN" is direct RN only → EIN 7; total RN adds DON (5) + RN admin (6).
PBJ_METRIC_TO_EIN_JOB_CODES: dict[str, tuple[int, ...]] = {
    "rn_hours": (7,),
    "total_rn_hours": (5, 6, 7),
    "rn_admin_hours": (6,),
    "rn_don_hours": (5,),
    "lpn_hours": (9,),
    "total_lpn_hours": (8, 9),
    "lpn_admin_hours": (8,),
    "cna_hours": (10, 11, 12),
    "total_nurse_aide_hours": (10, 11, 12),
}

def format_ein_job_codes_for_ui(codes: tuple[int, ...] | list[int]) -> str:
    """Human-readable EIN code list for UI, e.g. ``7`` or ``5–7`` or ``10, 11, 12``."""
    u = sorted({int(x) for x in codes})
    if not u:
        return ""
    if len(u) == 1:
        return str(u[0])
    if u == list(range(u[0], u[-1] + 1)):
        return f"{u[0]}–{u[-1]}"
    return ", ".join(str(x) for x in u)


# Short labels for PBJ→EIN bridge UI (daily / single-day clickable cells).
PBJ_METRIC_DISPLAY_NAME: dict[str, str] = {
    "rn_hours": "Direct RN",
    "total_rn_hours": "Total RN",
    "rn_admin_hours": "RN admin",
    "rn_don_hours": "RN DON",
    "lpn_hours": "Direct LPN",
    "total_lpn_hours": "Total LPN",
    "lpn_admin_hours": "LPN admin",
    "cna_hours": "Direct CNA / aide line",
    "total_nurse_aide_hours": "Total nurse aide",
}


def pbj_metric_display_name(metric_key: str) -> str:
    """Display name for a PBJ metric key used in the EIN bridge API."""
    k = str(metric_key or "").strip()
    if k in PBJ_METRIC_DISPLAY_NAME:
        return PBJ_METRIC_DISPLAY_NAME[k]
    return k.replace("_", " ").title() if k else ""


# PBJ ``complete_data`` columns summed per bridge metric (same definitions as dashboard daily cells).
PBJ_METRIC_TO_DAILY_COLUMNS: dict[str, tuple[str, ...]] = {
    "rn_hours": ("Hrs_RN",),
    "total_rn_hours": ("Hrs_RN", "Hrs_RNadmin", "Hrs_RNDON"),
    "rn_admin_hours": ("Hrs_RNadmin",),
    "rn_don_hours": ("Hrs_RNDON",),
    "lpn_hours": ("Hrs_LPN",),
    "total_lpn_hours": ("Hrs_LPN", "Hrs_LPNadmin"),
    "lpn_admin_hours": ("Hrs_LPNadmin",),
    "cna_hours": ("Hrs_CNA",),
    "total_nurse_aide_hours": ("Hrs_CNA", "Hrs_NAtrn", "Hrs_MedAide"),
}


def get_pbj_complete_data_row_for_date(df: pd.DataFrame | None, date_iso: str) -> pd.Series | None:
    """Return the first row for calendar ``date_iso`` (``YYYY-MM-DD``) from PBJ daily data."""
    if df is None or df.empty or not str(date_iso).strip():
        return None
    try:
        day = pd.to_datetime(str(date_iso).strip()).normalize()
    except Exception:
        return None
    if "WorkDate" not in df.columns:
        return None
    wd = pd.to_datetime(df["WorkDate"], errors="coerce").dt.normalize()
    sub = df[wd == day]
    if sub.empty:
        return None
    return sub.iloc[0]


def sum_pbj_hours_for_bridge_metric(row: Any, metric: str) -> float | None:
    """Sum PBJ hour columns for one facility-day row (Series-like) and bridge ``metric``."""
    cols = PBJ_METRIC_TO_DAILY_COLUMNS.get(metric)
    if cols is None or row is None:
        return None
    tot = 0.0
    for c in cols:
        try:
            if hasattr(row, "index") and c not in row.index:
                return None
            v = row[c]
        except Exception:
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(x):
            return None
        tot += x
    return round(float(tot), 2)


def compare_pbj_vs_ein_hours(
    pbj_sum: float | None,
    ein_sum: float,
    *,
    abs_tol_hours: float = 2.5,
    rel_tol: float = 0.03,
) -> dict[str, Any]:
    """
    Compare PBJ daily bucket hours to Employee Detail rollup for the same day.

    Uses max(absolute tolerance, relative tolerance × max side) to allow rounding noise
    while flagging material gaps (extract errors, quarter mismatch, definitional drift).
    """
    ein_r = round(float(ein_sum), 2)
    if pbj_sum is None:
        return {
            "hours_compare_available": False,
            "hours_pbj": None,
            "hours_ein": ein_r,
            "delta_ein_minus_pbj": None,
            "mismatch_flag": False,
            "compare_note": "No PBJ daily row for this date in the loaded facility file—cross-check skipped.",
        }
    pbj_r = round(float(pbj_sum), 2)
    delta = round(ein_r - pbj_r, 2)
    base = max(abs(pbj_r), abs(ein_r), 1.0)
    thresh = max(abs_tol_hours, rel_tol * base)
    flagged = abs(delta) > thresh
    note: str | None = None
    if flagged:
        note = (
            f"Employee Detail sum ({ein_r} hrs) vs PBJ ({pbj_r} hrs): Δ {delta:+.2f}. "
            f"Exceeds sanity band (~±{thresh:.1f} hr); check extract coverage, CY_Qtr, and job-code filters."
        )
    return {
        "hours_compare_available": True,
        "hours_pbj": pbj_r,
        "hours_ein": ein_r,
        "delta_ein_minus_pbj": delta,
        "mismatch_flag": flagged,
        "compare_tolerance_applied": round(thresh, 2),
        "compare_note": note,
    }


# When PBJ columns do not cover the full EIN job-code set, explain expected disagreement.
PBJ_BRIDGE_COMPARE_CAVEATS: dict[str, str] = {
    "cna_hours": (
        "PBJ column is Hrs_CNA only; Employee Detail rolls up job codes 10–12 (CNA, aide trainee, med aide)—"
        "totals often differ by design."
    ),
}


PBJ_METRIC_BRIDGE_NOTES: dict[str, str] = {
    "rn_hours": "PBJ Hrs_RN (direct RN) lines up with EIN job 7; excludes DON (5) and RN admin (6).",
    "total_rn_hours": "PBJ total RN hours ≈ EIN jobs 5 (DON) + 6 (RN admin) + 7 (RN).",
    "rn_admin_hours": "PBJ Hrs_RNadmin ↔ EIN job 6.",
    "rn_don_hours": "PBJ Hrs_RNDON ↔ EIN job 5.",
    "lpn_hours": "PBJ Hrs_LPN (direct) ↔ EIN job 9; excludes LPN admin (8).",
    "total_lpn_hours": "PBJ total LPN ↔ EIN 8 (LPN admin) + 9 (LPN/LVN).",
    "lpn_admin_hours": "PBJ Hrs_LPNadmin ↔ EIN job 8.",
    "cna_hours": "PBJ Hrs_CNA ↔ EIN 10–12 (CNA, aide in training, med aide).",
    "total_nurse_aide_hours": "PBJ total nurse aide ↔ EIN direct-care aide codes 10–12.",
}


def ein_quarter_in_puf_archive(year: int, quarter: int) -> bool:
    """
    True if (year, quarter) is a plausible CMS Employee Detail release.

    Uses a rolling ceiling (current calendar year + 1) so new quarterly drops (e.g. 2026Q1)
    are picked up without editing this file. Earliest supported release is 2020Q3.
    """
    y, q = int(year), int(quarter)
    if q < 1 or q > 4:
        return False
    if y < 2020:
        return False
    if y == 2020 and q < 3:
        return False
    return y <= datetime.now().year + 1


def ein_quarter_sort_key_from_year_q(year: int, quarter: int) -> int:
    """Monotonic integer for (year, quarter) comparisons (Q1..Q4)."""
    return int(year) * 4 + (int(quarter) - 1)


def ein_quarter_sort_key_to_label(sort_key: int) -> str:
    """Inverse of ``ein_quarter_sort_key_from_year_q`` → ``CYyyyyQn``."""
    q = (int(sort_key) % 4) + 1
    y = int(sort_key) // 4
    return f"CY{y}Q{q}"


def puf_listing_ceiling_sort_key() -> int:
    """Upper bound for scanning ZIPs: last quarter of (current year + 1)."""
    return ein_quarter_sort_key_from_year_q(datetime.now().year + 1, 4)


# Sort keys for the PUF archive span (for --full-puf-range fallback if discovery finds nothing)
PUF_ARCHIVE_MIN_SORT_KEY = ein_quarter_sort_key_from_year_q(2020, 3)
PUF_ARCHIVE_MAX_SORT_KEY = puf_listing_ceiling_sort_key()


def parse_ein_quarter_bound(label: str | None) -> int | None:
    """
    Parse a bound like 'CY2024Q3', '2024Q3', or '2024-Q3' to a sort key.
    Returns None if label is empty/invalid.
    """
    if label is None:
        return None
    s = str(label).strip()
    if not s:
        return None
    cy = normalize_cy_qtr_ein(s)
    if not cy:
        return None
    m = re.match(r"^CY(\d{4})Q([1-4])$", cy)
    if not m:
        return None
    return ein_quarter_sort_key_from_year_q(int(m.group(1)), int(m.group(2)))


def normalize_cy_qtr_ein(val) -> str | None:
    """Normalize employee-detail CY_Qtr (e.g. 2025Q2) to CYyyyyQn."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip().upper().replace("\ufeff", "")
    m = re.search(r"(?:CY)?(\d{4})Q([1-4])", s)
    if not m:
        return None
    return f"CY{m.group(1)}Q{m.group(2)}"


def ein_cy_quarter_from_ein_source_name(name: str) -> str | None:
    """
    Infer CYyyyyQn from a monolithic ZIP member path, supplemental ZIP filename, or CSV basename.

    Handles ``.../2025-Q3/...`` and CMS flat names like ``..._Q3_2025.csv``.
    """
    m = re.search(r"(\d{4})-Q([1-4])", name)
    if m:
        y, q = int(m.group(1)), int(m.group(2))
        if ein_quarter_in_puf_archive(y, q):
            return f"CY{y}Q{q}"
        return None
    m2 = re.search(r"_Q([1-4])_(\d{4})", name, re.I)
    if m2:
        q, y = int(m2.group(1)), int(m2.group(2))
        if ein_quarter_in_puf_archive(y, q):
            return f"CY{y}Q{q}"
        return None
    return None


def _ein_all_row_level_members_monolithic(z: zipfile.ZipFile) -> list[str]:
    """List every row-level Employee Detail CSV path under the monolithic PUF tree."""
    prefix = "Payroll Based Journal Employee Detail Nursing Home Staffing/"
    out: list[str] = []
    for name in z.namelist():
        if not name.lower().endswith(".csv"):
            continue
        if not _ein_row_level_csv_basename_ok(os.path.basename(name)):
            continue
        rest = name[len(prefix) :] if name.startswith(prefix) else ""
        m = re.match(r"(\d{4})-Q([1-4])/", rest)
        if not m:
            continue
        y, q = int(m.group(1)), int(m.group(2))
        if not ein_quarter_in_puf_archive(y, q):
            continue
        out.append(name)
    return out


def discover_ein_quarter_range_on_disk(
    app_root: str,
    primary_zip_path: str,
) -> tuple[int | None, int | None]:
    """
    Min/max sort keys for Employee Detail quarters actually present on disk (monolithic + supplemental).

    Used to default extraction windows without hardcoding the latest CMS release.
    """
    keys: list[int] = []

    def consider(zp: str, member: str) -> None:
        cy = ein_cy_quarter_from_ein_source_name(member) or ein_cy_quarter_from_ein_source_name(
            os.path.basename(zp)
        )
        if not cy:
            return
        sk = parse_ein_quarter_bound(cy)
        if sk is not None:
            keys.append(sk)

    if primary_zip_path and os.path.isfile(primary_zip_path):
        try:
            with zipfile.ZipFile(primary_zip_path, "r") as z:
                for m in _ein_all_row_level_members_monolithic(z):
                    consider(primary_zip_path, m)
        except zipfile.BadZipFile:
            pass

    sup_dir = os.path.join(app_root, "EIN", "supplemental")
    if os.path.isdir(sup_dir):
        for zp in sorted(glob.glob(os.path.join(sup_dir, "*.zip"))):
            if not os.path.isfile(zp):
                continue
            try:
                with zipfile.ZipFile(zp, "r") as z:
                    for name in z.namelist():
                        if not name.lower().endswith(".csv"):
                            continue
                        base = os.path.basename(name)
                        if not _ein_row_level_csv_basename_ok(base):
                            continue
                        consider(zp, name)
            except zipfile.BadZipFile:
                continue

    if not keys:
        return None, None
    return min(keys), max(keys)


def _ein_row_level_csv_basename_ok(basename: str) -> bool:
    """True for national row-level Employee Detail CSVs (monolithic or supplemental naming)."""
    b = basename.lower()
    if not b.endswith(".csv"):
        return False
    compact = b.replace("_", "")
    if "employeedetail" in compact:
        return True
    return "employee" in compact and "detail" in compact


def collect_ein_detail_jobs(
    app_root: str,
    primary_zip_path: str,
    min_sort_key: int | None,
    max_sort_key: int | None,
) -> list[tuple[str, str]]:
    """
    All (zip_path, member_path) jobs for the requested quarter window.

    Includes the monolithic PUF (if present) plus flat quarterly zips under
    ``EIN/supplemental/``. Same calendar quarter is only included once (monolithic wins).
    """
    lo = PUF_ARCHIVE_MIN_SORT_KEY if min_sort_key is None else int(min_sort_key)
    hi = puf_listing_ceiling_sort_key() if max_sort_key is None else int(max_sort_key)
    if lo > hi:
        lo, hi = hi, lo

    seen_cy: set[str] = set()
    jobs: list[tuple[str, str]] = []

    def add_if_in_range(zp: str, member: str) -> None:
        cy = ein_cy_quarter_from_ein_source_name(member) or ein_cy_quarter_from_ein_source_name(
            os.path.basename(zp)
        )
        if not cy:
            return
        sk = parse_ein_quarter_bound(cy)
        if sk is None or sk < lo or sk > hi:
            return
        if cy in seen_cy:
            return
        seen_cy.add(cy)
        jobs.append((zp, member))

    if primary_zip_path and os.path.isfile(primary_zip_path):
        with zipfile.ZipFile(primary_zip_path, "r") as z:
            for m in _ein_detail_csv_members_in_zip(
                z, min_sort_key=min_sort_key, max_sort_key=max_sort_key
            ):
                add_if_in_range(primary_zip_path, m)

    sup_dir = os.path.join(app_root, "EIN", "supplemental")
    if os.path.isdir(sup_dir):
        for zp in sorted(glob.glob(os.path.join(sup_dir, "*.zip"))):
            if not os.path.isfile(zp):
                continue
            try:
                with zipfile.ZipFile(zp, "r") as z:
                    for name in z.namelist():
                        if not name.lower().endswith(".csv"):
                            continue
                        base = os.path.basename(name)
                        if not _ein_row_level_csv_basename_ok(base):
                            continue
                        add_if_in_range(zp, name)
            except zipfile.BadZipFile:
                continue

    return jobs


def _ein_detail_csv_members_in_zip(
    z: zipfile.ZipFile,
    *,
    min_sort_key: int | None = None,
    max_sort_key: int | None = None,
) -> list[str]:
    """
    Member paths for employeedetail CSVs inside the PUF archive, filtered to a quarter range.

    When min/max sort keys are None, uses PUF_ARCHIVE_MIN_SORT_KEY .. listing ceiling (year+1 Q4).
    """
    lo = PUF_ARCHIVE_MIN_SORT_KEY if min_sort_key is None else int(min_sort_key)
    hi = puf_listing_ceiling_sort_key() if max_sort_key is None else int(max_sort_key)
    if lo > hi:
        lo, hi = hi, lo
    out: list[str] = []
    for name in _ein_all_row_level_members_monolithic(z):
        m = re.search(r"(\d{4})-Q([1-4])", name)
        if not m:
            continue
        sk = ein_quarter_sort_key_from_year_q(int(m.group(1)), int(m.group(2)))
        if sk < lo or sk > hi:
            continue
        out.append(name)
    return out


def iter_ein_detail_csv_members(zip_path: str) -> Iterator[tuple[str, str]]:
    """
    Yield (zip_member_path, canonical_quarter) for employeedetail CSVs in the supported range.
    """
    if not os.path.isfile(zip_path):
        return
    z = zipfile.ZipFile(zip_path, "r")
    try:
        for name in _ein_detail_csv_members_in_zip(z):
            m = re.search(r"(\d{4})-Q([1-4])", name)
            if m:
                yield name, f"CY{int(m.group(1))}Q{int(m.group(2))}"
    finally:
        z.close()


# Columns needed for aggregation (see EIN/data_dictionary.md); omit unused to speed ZIP scans.
_EIN_DETAIL_USECOLS = (
    "PROVNUM",
    "STATE",
    "CY_Qtr",
    "WorkDate",
    "SYS_EMPLEE_ID",
    "EMPLEE_JOB_CD_ID",
    "EMP_CTR",
    "WORK_HRS_NUM",
)


def _normalize_ein_provnum_cell(val) -> str:
    """Match pandas normalization used previously (strip, drop trailing .0, zfill)."""
    s = str(val).strip()
    s = re.sub(r"\.0$", "", s)
    return s.zfill(6) if s.isdigit() else s


def _stream_ein_detail_member_from_zip(z: zipfile.ZipFile, member: str, prov: str) -> pd.DataFrame:
    """
    One national employeedetail CSV inside an already-open ZIP: csv.reader, keep only PROVNUM rows.

    Avoids pandas chunked reads over the full national file (orders of magnitude faster for one facility).
    """
    prov = str(prov).strip().zfill(6)
    rows: list[list[str]] = []
    col_names: list[str] = []
    try:
        with z.open(member, "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
            reader = csv.reader(text)
            try:
                header = next(reader)
            except StopIteration:
                return pd.DataFrame()
            header = [h.strip().lstrip("\ufeff") for h in header]
            idx_map = {name: i for i, name in enumerate(header)}
            if "PROVNUM" not in idx_map:
                return pd.DataFrame()
            prov_i = idx_map["PROVNUM"]
            use_names = [c for c in _EIN_DETAIL_USECOLS if c in idx_map]
            if "PROVNUM" not in use_names:
                return pd.DataFrame()
            col_indices = [idx_map[c] for c in use_names]
            col_names = use_names
            for row in reader:
                if prov_i >= len(row):
                    continue
                if _normalize_ein_provnum_cell(row[prov_i]) != prov:
                    continue
                rows.append(
                    [row[i] if i < len(row) else "" for i in col_indices]
                )
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=pd.Index(col_names))


def _stream_ein_detail_member_for_prov(zip_path: str, member: str, prov: str) -> pd.DataFrame:
    """Open ZIP, stream one member, close (for multiprocessing workers)."""
    z = zipfile.ZipFile(zip_path, "r")
    try:
        return _stream_ein_detail_member_from_zip(z, member, prov)
    finally:
        z.close()


def ein_member_cy_quarter(member_path: str) -> str | None:
    """Canonical label ``CYyyyyQn`` from a ZIP member path (e.g. .../2025-Q1/...)."""
    return ein_cy_quarter_from_ein_source_name(member_path)


def _ein_quarter_cache_file(quarter_cache_parent: str, prov: str, zip_path: str, member: str) -> str:
    """Path to per-quarter Parquet checkpoint under ``quarter_cache_parent / prov /``."""
    cy = ein_cy_quarter_from_ein_source_name(member) or ein_cy_quarter_from_ein_source_name(
        os.path.basename(zip_path)
    )
    safe = cy if cy else re.sub(r"[^\w.\-]+", "_", os.path.basename(member))[:120]
    return os.path.join(quarter_cache_parent, str(prov).strip().zfill(6), f"{safe}.parquet")


def extract_facility_ein_one_member_cached(
    zip_path: str,
    member: str,
    prov: str,
    *,
    cache_path: str | None,
    resume: bool,
    z: zipfile.ZipFile | None = None,
) -> tuple[str, int, str, pd.DataFrame]:
    """
    One quarter file: load from checkpoint Parquet if ``resume`` and file exists, else stream ZIP.

    Returns ``(cy_quarter, row_count, source, dataframe)`` where ``source`` is ``cached`` or ``scanned``.
    Empty quarters are not written to cache (so a later run will rescan).
    """
    cy = (
        ein_cy_quarter_from_ein_source_name(member)
        or ein_cy_quarter_from_ein_source_name(os.path.basename(zip_path))
        or "UNKNOWN"
    )
    prov = str(prov).strip().zfill(6)
    if resume and cache_path and os.path.isfile(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return (cy, len(df), "cached", df)
        except Exception:
            pass
    if z is not None:
        df = _stream_ein_detail_member_from_zip(z, member, prov)
    else:
        df = _stream_ein_detail_member_for_prov(zip_path, member, prov)
    n = len(df)
    if cache_path and n > 0:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            typed = normalize_ein_detail_dtypes_for_disk(df)
            write_facility_ein_parquet(typed, cache_path)
        except Exception:
            pass
    return (cy, n, "scanned", df)


def _sort_key_cy_quarter(label: str) -> int:
    k = parse_ein_quarter_bound(label)
    return k if k is not None else -1


def _concat_quarter_chunks(chunks: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    if not chunks:
        return pd.DataFrame()
    chunks.sort(key=lambda t: _sort_key_cy_quarter(t[0]))
    return pd.concat([t[1] for t in chunks], ignore_index=True)


def extract_facility_ein_detail(
    zip_path: str,
    provnum: str,
    chunksize: int = 400_000,
    min_sort_key: int | None = None,
    max_sort_key: int | None = None,
    *,
    quarter_cache_parent: str | None = None,
    resume: bool = True,
    verbose: bool = False,
    app_root: str | None = None,
) -> pd.DataFrame:
    """
    Read employee-detail rows for PROVNUM from the monolithic PUF and/or ``EIN/supplemental/*.zip``.

    Quarter folders are filtered by sort keys (see parse_ein_quarter_bound). Defaults
    when both None: entire supported span on disk (2020Q3 through listing ceiling).

    ``chunksize`` is retained for API compatibility; extraction uses line streaming (not pandas chunks).

    If ``quarter_cache_parent`` is set (e.g. ``out_dir/.ein_quarter_cache``), each quarter's rows are
    saved under ``.../{prov}/CYyyyyQn.parquet`` so interrupted runs can resume when ``resume`` is True.

    ``app_root`` is the repo root (parent of ``EIN/``). If omitted, inferred from ``zip_path`` when
    that file exists; otherwise the current working directory is used (pass ``app_root`` when the main
    PUF is missing and only supplemental zips exist).
    """
    del chunksize  # streaming path; ignore legacy chunk size
    prov = str(provnum).strip().zfill(6)
    ar = app_root
    if ar is None:
        if zip_path and os.path.isfile(zip_path):
            ar = os.path.dirname(os.path.dirname(os.path.abspath(zip_path)))
        else:
            ar = os.getcwd()
    primary = zip_path if (zip_path and os.path.isfile(zip_path)) else ""
    jobs = collect_ein_detail_jobs(ar, primary, min_sort_key, max_sort_key)
    if not jobs:
        return pd.DataFrame()

    collected: list[tuple[str, pd.DataFrame]] = []
    by_zip: dict[str, list[str]] = defaultdict(list)
    for zp, m in jobs:
        by_zip[zp].append(m)

    for zp, members in by_zip.items():
        z = zipfile.ZipFile(zp, "r")
        try:
            for member in members:
                cpath = (
                    _ein_quarter_cache_file(quarter_cache_parent, prov, zp, member)
                    if quarter_cache_parent
                    else None
                )
                cy, n, src, part = extract_facility_ein_one_member_cached(
                    zp,
                    member,
                    prov,
                    cache_path=cpath,
                    resume=resume,
                    z=z,
                )
                if verbose:
                    tag = "checkpoint" if src == "cached" else "scanned"
                    print(f"  OK {cy}  {n:,} rows  ({tag})")
                if not part.empty:
                    collected.append((cy, part))
        finally:
            z.close()
    return _concat_quarter_chunks(collected)


def _extract_facility_ein_worker_cached(
    args: tuple[str, str, str, str | None, bool],
) -> tuple[str, int, str, pd.DataFrame]:
    """Picklable worker: one ZIP member with optional quarter checkpoint."""
    zip_path, member, prov, cache_path, resume = args
    resume_b = bool(resume)
    cp = cache_path if cache_path else None
    return extract_facility_ein_one_member_cached(
        zip_path, member, prov, cache_path=cp, resume=resume_b, z=None
    )


def extract_facility_ein_detail_parallel(
    zip_path: str,
    provnum: str,
    chunksize: int = 400_000,
    max_workers: int = 8,
    min_sort_key: int | None = None,
    max_sort_key: int | None = None,
    *,
    quarter_cache_parent: str | None = None,
    resume: bool = True,
    verbose: bool = False,
    app_root: str | None = None,
) -> pd.DataFrame:
    """
    Same as extract_facility_ein_detail but processes each quarter CSV in parallel (Windows-safe).

    ``chunksize`` is ignored; scanning is streaming per file. Supports the same quarter checkpoint
    directory as the sequential path.
    """
    del chunksize
    prov = str(provnum).strip().zfill(6)
    ar = app_root
    if ar is None:
        if zip_path and os.path.isfile(zip_path):
            ar = os.path.dirname(os.path.dirname(os.path.abspath(zip_path)))
        else:
            ar = os.getcwd()
    primary = zip_path if (zip_path and os.path.isfile(zip_path)) else ""
    jobs = collect_ein_detail_jobs(ar, primary, min_sort_key, max_sort_key)
    if not jobs:
        return pd.DataFrame()
    args_list: list[tuple[str, str, str, str | None, bool]] = []
    for zp, m in jobs:
        cpath: str | None = None
        if quarter_cache_parent:
            cpath = _ein_quarter_cache_file(quarter_cache_parent, prov, zp, m)
        args_list.append((zp, m, prov, cpath, resume))
    collected: list[tuple[str, pd.DataFrame]] = []
    workers = max(1, min(int(max_workers), len(jobs)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_extract_facility_ein_worker_cached, a): a for a in args_list}
        for fut in as_completed(futures):
            cy, n, src, part = fut.result()
            if verbose:
                tag = "checkpoint" if src == "cached" else "scanned"
                print(f"  OK {cy}  {n:,} rows  ({tag})")
            if not part.empty:
                collected.append((cy, part))
    return _concat_quarter_chunks(collected)


def build_ein_job_quarterly(detail: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate employee-detail rows to one row per (CY_Qtr, job code) with hours and headcount.
    """
    if detail is None or detail.empty:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "CY_Qtr",
                    "EMPLEE_JOB_CD_ID",
                    "job_title",
                    "job_category",
                    "hours_total",
                    "hours_employee",
                    "hours_contract",
                    "row_count",
                    "unique_employees",
                ]
            )
        )
    d: pd.DataFrame = detail.copy()
    d["CY_Qtr"] = d["CY_Qtr"].apply(normalize_cy_qtr_ein)
    d = cast(pd.DataFrame, d[d["CY_Qtr"].notna()])
    d["EMPLEE_JOB_CD_ID"] = pd.to_numeric(d["EMPLEE_JOB_CD_ID"], errors="coerce")
    d = cast(pd.DataFrame, d[d["EMPLEE_JOB_CD_ID"].notna()])
    d["WORK_HRS_NUM"] = pd.to_numeric(d["WORK_HRS_NUM"], errors="coerce").fillna(0.0)
    d["EMP_CTR"] = pd.to_numeric(d["EMP_CTR"], errors="coerce")
    d["hours_employee"] = np.where(d["EMP_CTR"] == 1, d["WORK_HRS_NUM"], 0.0)
    d["hours_contract"] = np.where(d["EMP_CTR"] == 2, d["WORK_HRS_NUM"], 0.0)
    if "SYS_EMPLEE_ID" not in d.columns:
        d["SYS_EMPLEE_ID"] = np.nan
    g = cast(
        pd.DataFrame,
        d.groupby(["CY_Qtr", "EMPLEE_JOB_CD_ID"], as_index=False).agg(
            hours_total=("WORK_HRS_NUM", "sum"),
            hours_employee=("hours_employee", "sum"),
            hours_contract=("hours_contract", "sum"),
            row_count=("WORK_HRS_NUM", "count"),
            unique_employees=("SYS_EMPLEE_ID", pd.Series.nunique),
        ),
    )
    g["job_title"] = g["EMPLEE_JOB_CD_ID"].apply(job_title)
    g["job_category"] = g["EMPLEE_JOB_CD_ID"].apply(job_category)
    g = g.sort_values(["CY_Qtr", "EMPLEE_JOB_CD_ID"])
    return g


def build_ein_category_quarterly(job_quarterly: pd.DataFrame) -> pd.DataFrame:
    """Roll job_quarterly up by CY_Qtr + job_category."""
    if job_quarterly is None or job_quarterly.empty:
        return pd.DataFrame(
            columns=pd.Index(
                [
                    "CY_Qtr",
                    "job_category",
                    "hours_total",
                    "hours_employee",
                    "hours_contract",
                ]
            )
        )
    return cast(
        pd.DataFrame,
        job_quarterly.groupby(["CY_Qtr", "job_category"], as_index=False)[
            ["hours_total", "hours_employee", "hours_contract"]
        ]
        .sum()
        .sort_values(["CY_Qtr", "job_category"]),
    )


def normalize_ein_detail_dtypes_for_disk(detail: pd.DataFrame) -> pd.DataFrame:
    """
    Compact dtypes before Parquet (smaller files, faster reads). Safe for analytics code that
    still uses pd.to_numeric when loading legacy CSV.
    """
    if detail is None or detail.empty:
        return detail
    out: pd.DataFrame = detail.copy()
    if "PROVNUM" in out.columns:
        s = out["PROVNUM"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
        is_num = s.str.match(r"^\d+$", na=False)
        out["PROVNUM"] = np.where(is_num, s.str.zfill(6), s)
    if "STATE" in out.columns:
        out["STATE"] = out["STATE"].astype(str).str.strip()
    if "CY_Qtr" in out.columns:
        out["CY_Qtr"] = out["CY_Qtr"].astype(str).str.strip()
    if "WorkDate" in out.columns:
        out["WorkDate"] = pd.to_numeric(out["WorkDate"], errors="coerce").astype("Int64")
    if "SYS_EMPLEE_ID" in out.columns:
        out["SYS_EMPLEE_ID"] = pd.to_numeric(out["SYS_EMPLEE_ID"], errors="coerce").astype("Int64")
    if "EMPLEE_JOB_CD_ID" in out.columns:
        out["EMPLEE_JOB_CD_ID"] = pd.to_numeric(out["EMPLEE_JOB_CD_ID"], errors="coerce").astype("Int16")
    if "EMP_CTR" in out.columns:
        out["EMP_CTR"] = pd.to_numeric(out["EMP_CTR"], errors="coerce").astype("Int8")
    if "WORK_HRS_NUM" in out.columns:
        out["WORK_HRS_NUM"] = pd.to_numeric(out["WORK_HRS_NUM"], errors="coerce").astype("float32")
    return out


def read_facility_ein_parquet_or_csv(base_no_ext: str) -> pd.DataFrame | None:
    """
    Load a facility EIN table: prefer ``{base_no_ext}.parquet``, else ``{base_no_ext}.csv``.

    ``base_no_ext`` is the full path without extension, e.g.
    ``os.path.join(root, "facility_335513_ein_employee_detail")``.
    """
    pq = f"{base_no_ext}.parquet"
    csv = f"{base_no_ext}.csv"
    if os.path.isfile(pq):
        try:
            return pd.read_parquet(pq)
        except Exception as exc:
            # Vercel/serverless often has no pyarrow; prefer CSV in bundle when parquet fails
            print(f"[EIN] Parquet load failed for {base_no_ext!r} ({exc}); trying CSV if present.")
    if os.path.isfile(csv):
        try:
            return pd.read_csv(csv, low_memory=False)
        except Exception as exc:
            print(f"[EIN] Could not load CSV {base_no_ext!r}: {exc}")
    return None


def write_facility_ein_parquet(df: pd.DataFrame, path: str) -> None:
    """Write Parquet with zstd (requires pyarrow)."""
    df.to_parquet(path, index=False, compression="zstd", engine="pyarrow")


def default_ein_zip_path(app_root: str) -> str:
    """Default ZIP path next to EIN folder in repo root."""
    return os.path.join(
        app_root,
        "EIN",
        "Payroll Based Journal Employee Detail Nursing Home Staffing.zip",
    )


def _zip_contains_ein_monolithic_structure(zip_path: str) -> bool:
    """True if archive looks like the CMS monolithic Employee Detail PUF (quick scan)."""
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            for i, name in enumerate(z.namelist()):
                if i >= 200_000:
                    break
                if "Payroll Based Journal Employee Detail Nursing Home Staffing" in name:
                    return True
                if re.search(r"\d{4}-Q[1-4]/", name) and name.lower().endswith(".csv"):
                    if _ein_row_level_csv_basename_ok(os.path.basename(name)):
                        return True
    except (OSError, zipfile.BadZipFile, RuntimeError):
        return False
    return False


def _ein_zips_in_dir(dir_path: str) -> list[str]:
    if not dir_path or not os.path.isdir(dir_path):
        return []
    out: list[str] = []
    try:
        for fn in os.listdir(dir_path):
            if not fn.lower().endswith(".zip"):
                continue
            p = os.path.join(dir_path, fn)
            if os.path.isfile(p):
                out.append(p)
    except OSError:
        return []
    return out


def _pick_primary_ein_zip_from_candidates(candidates: list[str]) -> str:
    """Prefer archives that match CMS structure; else a single zip in the pool."""
    if not candidates:
        return ""
    with_struct = [p for p in candidates if _zip_contains_ein_monolithic_structure(p)]
    if with_struct:
        return max(with_struct, key=os.path.getsize)
    if len(candidates) == 1:
        return candidates[0]
    return ""


def resolve_ein_primary_zip(app_root: str) -> str:
    """
    Locate the monolithic Employee Detail PUF zip, or return ``""``.

    Resolution order:

    1. Environment: ``PBJ_EIN_DETAIL_ZIP``, ``CMS_EIN_DETAIL_ZIP``, ``EIN_DETAIL_ZIP``
       (absolute path to a file).
    2. Canonical filename under ``<app_root>/EIN/`` (see ``default_ein_zip_path``).
    3. Any ``*.zip`` in ``<app_root>/EIN/`` that contains the CMS monolithic tree; if several,
       the largest matching file.
    4. Exactly one ``*.zip`` in ``<app_root>/EIN/`` (renamed PUF — trusted).
    5. Same as 3–4 for ``PBJ_EIN_ARCHIVE_DIR`` if set.

    Supplemental-only setups (no monolithic) leave this empty; use ``EIN/supplemental/*.zip``.
    """
    app_root = os.path.abspath(app_root)

    for env_key in ("PBJ_EIN_DETAIL_ZIP", "CMS_EIN_DETAIL_ZIP", "EIN_DETAIL_ZIP"):
        raw = os.environ.get(env_key, "").strip().strip('"')
        if raw and os.path.isfile(raw):
            return os.path.abspath(raw)

    canonical = default_ein_zip_path(app_root)
    if os.path.isfile(canonical):
        return canonical

    ein_dir = os.path.join(app_root, "EIN")
    ein_zips = _ein_zips_in_dir(ein_dir)
    picked = _pick_primary_ein_zip_from_candidates(ein_zips)
    if picked:
        return picked

    extra = os.environ.get("PBJ_EIN_ARCHIVE_DIR", "").strip().strip('"')
    if extra:
        picked2 = _pick_primary_ein_zip_from_candidates(_ein_zips_in_dir(extra))
        if picked2:
            return picked2

    return ""


def ein_employee_detail_sources_available(app_root: str) -> bool:
    """True if a monolithic PUF and/or supplemental quarter ZIPs are discoverable on disk."""
    if bool(resolve_ein_primary_zip(app_root)):
        return True
    sup = os.path.join(app_root, "EIN", "supplemental")
    if os.path.isdir(sup) and glob.glob(os.path.join(sup, "*.zip")):
        return True
    return False


# CMS interactive explorer (same pattern as daily PBJ); quarter-specific path segment, e.g. q2-2025
CMS_EIN_DETAIL_LANDING_URL = (
    "https://data.cms.gov/quality-of-care/payroll-based-journal-employee-detail-nursing-home-staffing"
)


def ein_cy_quarter_to_cms_data_path_slug(cy_quarter: str) -> str | None:
    """Map CY2025Q2 or 2025Q2 to explorer slug ``q2-2025`` (lowercase)."""
    s = str(cy_quarter).strip().upper().replace("\ufeff", "")
    m = re.match(r"^(?:CY)?(\d{4})Q([1-4])$", s)
    if not m:
        return None
    return f"q{m.group(2)}-{m.group(1)}".lower()


def iso_date_to_cy_quarter(iso: str | None) -> str | None:
    """Map ``YYYY-MM-DD`` to ``CYyyyyQn`` so CMS explorer URLs use the dataset for that work date."""
    if not iso or not str(iso).strip():
        return None
    parts = str(iso).strip()[:10].split("-")
    if len(parts) != 3:
        return None
    try:
        y, mo, _ = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if not (1 <= mo <= 12):
        return None
    q = (mo - 1) // 3 + 1
    return f"CY{y}Q{q}"


def _ein_cms_work_date_param(work_date: str | int | None) -> str | None:
    """YYYYMMDD for CMS WorkDate filter (accepts YYYYMMDD or YYYY-MM-DD)."""
    if work_date is None:
        return None
    s = str(work_date).strip().replace("-", "")
    if len(s) == 8 and s.isdigit():
        return s
    return None


def build_cms_ein_detail_explorer_url(
    provnum: str,
    cy_quarter: str,
    *,
    work_date: str | int | None = None,
    job_code: int | None = None,
    job_code_between: tuple[int, int] | None = None,
    sys_employee_id: int | None = None,
    result_limit: int = 25,
) -> str | None:
    """
    Build a data.cms.gov Employee Detail explorer URL (filtered query), e.g.:
    ``.../data/q2-2025?query=...``

    When ``sys_employee_id`` is set, filters use ``SYS_EMPLEE_ID`` (and optional ``WorkDate`` only)
    so the view targets one employee, not everyone with the same job code at a facility.

    Otherwise uses ``PROVNUM``, optional ``WorkDate``, and optional ``EMPLEE_JOB_CD_ID`` (single code
    uses BETWEEN (n, n) to match the CMS UI).
    """
    slug = ein_cy_quarter_to_cms_data_path_slug(cy_quarter)
    if not slug:
        return None
    wd = _ein_cms_work_date_param(work_date)
    conditions: list[dict] = []

    if sys_employee_id is not None:
        conditions.append(
            {
                "column": {"value": "SYS_EMPLEE_ID"},
                "comparator": {"value": "="},
                "filterValue": [str(int(sys_employee_id))],
            }
        )
        if wd:
            conditions.append(
                {"column": {"value": "WorkDate"}, "comparator": {"value": "="}, "filterValue": [wd]}
            )
    else:
        p = str(provnum).strip()
        if p.isdigit():
            p = p.zfill(6)
        conditions.append(
            {"column": {"value": "PROVNUM"}, "comparator": {"value": "="}, "filterValue": [p]}
        )
        if wd:
            conditions.append(
                {"column": {"value": "WorkDate"}, "comparator": {"value": "="}, "filterValue": [wd]}
            )
        if job_code_between is not None:
            lo, hi = int(job_code_between[0]), int(job_code_between[1])
            if lo > hi:
                lo, hi = hi, lo
            conditions.append(
                {
                    "column": {"value": "EMPLEE_JOB_CD_ID"},
                    "comparator": {"value": "BETWEEN"},
                    "filterValue": [str(lo), str(hi)],
                }
            )
        elif job_code is not None:
            jc = int(job_code)
            conditions.append(
                {
                    "column": {"value": "EMPLEE_JOB_CD_ID"},
                    "comparator": {"value": "BETWEEN"},
                    "filterValue": [str(jc), str(jc)],
                }
            )
    query_obj = {
        "filters": {"list": [{"conditions": conditions}], "rootConjunction": {"value": "AND"}},
        "keywords": "",
        "offset": 0,
        "limit": max(1, min(int(result_limit), 500)),
        "sort": {"sortBy": None, "sortOrder": None},
        "columns": [],
    }
    base = f"{CMS_EIN_DETAIL_LANDING_URL}/data/{slug}"
    enc = urllib.parse.quote(json.dumps(query_obj, separators=(",", ":")))
    return f"{base}?query={enc}"
