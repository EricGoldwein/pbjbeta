"""
Employee-level summaries from PBJ Employee Detail (EIN) row-level table (Parquet or CSV).

Canonical module: edit only at repository root. Deployment copies re-export via ``pbj_fea_loader``.

Nursing-related job codes follow CMS PUF dictionary. The dashboard roster also includes job code
1 (facility Administrator) so Employee Detail aligns with Provider Information “administrator” concepts;
licensed and direct-care nursing remain codes 5–12.
"""

from __future__ import annotations

from collections import defaultdict
import re
import statistics
from datetime import date, datetime, timedelta
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd

from facility_ein_lib import (
    ein_quarter_sort_key_to_label,
    job_title,
    job_title_short,
    normalize_cy_qtr_ein,
    parse_ein_quarter_bound,
)

# Employee Detail roster: facility Administrator (1) + licensed / direct nursing (5–12).
NURSING_JOB_CODE_IDS: frozenset[int] = frozenset(
    {
        1,  # Administrator (facility administrator in CMS dictionary; not RN DON / RN admin)
        5,  # RN Director of Nursing
        6,  # RN with Administrative Duties
        7,  # Registered Nurse
        8,  # LPN with Administrative Duties
        9,  # LPN/LVN
        10,  # CNA
        11,  # Nurse Aide in Training
        12,  # Medication Aide/Technician
    }
)

# Roster groupings for UI / quarter summaries (admin = non-nurse bucket for now).
EIN_ADMIN_JOB_CODE: int = 1
EIN_NURSING_JOB_CODES: frozenset[int] = frozenset(range(5, 13))
# Direct care nursing only (excludes RN DON, RN admin, LPN admin).
EIN_DIRECT_NURSING_JOB_CODES: frozenset[int] = frozenset({7, 9, 10, 11, 12})


def ein_job_code_matches_position_group(job_code: Any, group: str | None) -> bool:
    """True if ``job_code`` matches dashboard Role filter ``group``. Mirrors ``jobCodeInGroup`` in the HTML."""
    g = (group or "").strip().lower()
    if not g or g == "all":
        return True
    try:
        c = int(job_code)
    except (TypeError, ValueError):
        return False
    if g == "nurse":
        return 5 <= c <= 12
    if g == "nurse_direct":
        return c in (7, 9, 10, 11, 12)
    if g == "non_nurse" or g == "admin":
        return c == 1
    if g == "rn":
        return 5 <= c <= 7
    if g == "rn_don":
        return c == 5
    if g == "rn_admin":
        return c == 6
    if g == "rn_direct":
        return c == 7
    if g == "lpn":
        return 8 <= c <= 9
    if g == "lpn_admin":
        return c == 8
    if g == "lpn_direct":
        return c == 9
    if g == "aide":
        return 10 <= c <= 12
    if g == "cna":
        return c == 10
    if g == "aide_train":
        return c == 11
    if g == "med_aide":
        return c == 12
    return True


def ein_nonnurse_job_code_matches_position_group(job_code: Any, group: str | None) -> bool:
    """True if ``job_code`` matches non-nurse Role filter (PBJ non-nurse EIN codes 2–4, 13–34, plus admin 1)."""
    from facility_ein_lib import NONNURSE_PBJ_EIN_JOB_CODES

    g = (group or "").strip().lower()
    if not g or g == "all":
        return True
    try:
        c = int(job_code)
    except (TypeError, ValueError):
        return False
    if c not in NONNURSE_PBJ_EIN_JOB_CODES:
        return False
    if g == "admin":
        return c == 1
    if g == "medical_director":
        return c == 2
    if g == "physicians":
        return c in (3, 4, 13)
    if g == "clinical_specialists":
        return c in (14, 15)
    if g == "dietary":
        return c in (16, 17)
    if g == "pt_rehab":
        return c in (21, 22, 23)
    if g == "ot_rehab":
        return c in (18, 19, 20)
    if g == "slp_rehab":
        return c == 26
    if g == "respiratory":
        return c in (24, 25)
    if g == "activities":
        return c in (27, 28, 29)
    if g == "social_behavioral":
        return c in (30, 31, 34)
    return True


# Compact labels for API / UI (day roster, tooltips).
EIN_NURSING_ROSTER_CODES_LABEL = "1, 5–12"
EIN_NURSING_ROSTER_BRIDGE_NOTE = (
    "Facility Administrator (job code 1) plus licensed and direct-care nursing (codes 5–12) "
    "for this work date."
)


def prepare_ein_detail(detail: pd.DataFrame) -> pd.DataFrame:
    """Normalize types and CY_Qtr for analytics."""
    if detail is None or detail.empty:
        return pd.DataFrame()
    d = detail.copy()
    d["WorkDate"] = pd.to_numeric(d["WorkDate"], errors="coerce")
    d["WORK_HRS_NUM"] = pd.to_numeric(d["WORK_HRS_NUM"], errors="coerce").fillna(0.0)
    if "WORK_HRS_FN" in d.columns:
        d["WORK_HRS_FN"] = pd.to_numeric(d["WORK_HRS_FN"], errors="coerce")
    d["SYS_EMPLEE_ID"] = pd.to_numeric(d["SYS_EMPLEE_ID"], errors="coerce")
    d["EMPLEE_JOB_CD_ID"] = pd.to_numeric(d["EMPLEE_JOB_CD_ID"], errors="coerce")
    d["EMP_CTR"] = pd.to_numeric(d["EMP_CTR"], errors="coerce")
    d["CY_Qtr_norm"] = d["CY_Qtr"].apply(normalize_cy_qtr_ein)
    d = d[d["WorkDate"].notna() & d["SYS_EMPLEE_ID"].notna() & d["EMPLEE_JOB_CD_ID"].notna()]
    d = d[d["CY_Qtr_norm"].notna()]
    return pd.DataFrame(d)


CMS_PBJ_STAFF_HOURS_DAILY_CAP = 22.5
"""CMS PBJ v4.10.0 (Feb 2026 clarification): submissions cap **more than** this many hours
per employee system ID per calendar workday across all job titles (linked IDs must still sum under one ID)."""


def ein_employee_day_cms_hours_cap_summary(
    detail: pd.DataFrame,
    *,
    workdate_lo_yyyymmdd: int | None = None,
    workdate_hi_yyyymmdd: int | None = None,
    hours_cap: float = CMS_PBJ_STAFF_HOURS_DAILY_CAP,
) -> dict[str, Any]:
    """Row-level EIN extract: flag (SYS_EMPLEE_ID, WorkDate) totals above the CMS PBJ daily hours cap.

    Uses **all job codes** present in the extract so the sum matches the CMS “across all job titles” rule
    for a single ``SYS_EMPLEE_ID`` (CMS system / employee ID in this file). Legacy **linked** IDs that
    appear as **multiple** ``SYS_EMPLEE_ID`` values are **not** merged here (no ID-link table in-app).

    ``ui_tier`` keeps rare spikes from taking UI space: isolated 1–2 employee-days in a large window
    is ``hidden`` for footnotes (values still returned for APIs / tooltips / exports).
    """
    empty_out: dict[str, Any] = {
        "available": False,
        "hours_cap": float(hours_cap),
        "comparison": "strictly_greater_than",
        "employee_days_in_scope": 0,
        "employee_days_over_cap": 0,
        "share_over_cap": None,
        "share_over_cap_pct_display": None,
        "max_hours_any_employee_day": None,
        "severity": "none",
        "ui_tier": "hidden",
        "tooltip_detail": None,
        "cms_spec_note": "PBJ Data Specifications v4.10.0 (employee-day cap; full programming release March 22, 2026).",
        "caveats": [
            "Per-SYS_EMPLEE_ID day totals only; multiple legacy IDs for one person are not combined without a link table.",
        ],
        "examples": [],
    }
    if detail is None or detail.empty:
        return empty_out

    d = prepare_ein_detail(detail)
    if d.empty:
        return empty_out

    wd_int = d["WorkDate"].astype(int)
    if workdate_lo_yyyymmdd is not None:
        d = d.loc[wd_int >= int(workdate_lo_yyyymmdd)].copy()
    if workdate_hi_yyyymmdd is not None:
        d = d.loc[d["WorkDate"].astype(int) <= int(workdate_hi_yyyymmdd)].copy()
    if d.empty:
        out = dict(empty_out)
        out["available"] = True
        out["caveats"] = list(empty_out["caveats"])
        return out

    daily = (
        d.groupby(["SYS_EMPLEE_ID", "WorkDate"], sort=False)["WORK_HRS_NUM"]
        .sum()
        .reset_index()
    )
    daily["WORK_HRS_NUM"] = pd.to_numeric(daily["WORK_HRS_NUM"], errors="coerce").fillna(0.0)
    n_scope = int(len(daily))
    if n_scope == 0:
        out = dict(empty_out)
        out["available"] = True
        out["caveats"] = list(empty_out["caveats"])
        return out

    mx_all = float(daily["WORK_HRS_NUM"].max())
    over = daily[daily["WORK_HRS_NUM"] > float(hours_cap) + 1e-9]
    n_over = int(len(over))
    share = float(n_over / n_scope) if n_scope else 0.0

    examples: list[dict[str, Any]] = []
    if n_over > 0:
        top = over.sort_values("WORK_HRS_NUM", ascending=False).head(3)
        for _, r in top.iterrows():
            eid = int(r["SYS_EMPLEE_ID"])
            wdi = int(r["WorkDate"])
            slice_ = d[(d["SYS_EMPLEE_ID"] == eid) & (d["WorkDate"] == wdi)]
            by_job = (
                slice_.groupby("EMPLEE_JOB_CD_ID", sort=False)["WORK_HRS_NUM"]
                .sum()
                .reset_index()
                .sort_values("WORK_HRS_NUM", ascending=False)
            )
            jobs = [
                {"job_code": int(row["EMPLEE_JOB_CD_ID"]), "hours": round(float(row["WORK_HRS_NUM"]), 2)}
                for _, row in by_job.iterrows()
            ]
            _eid_s = str(abs(eid))
            _mask = f"…{_eid_s[-4:]}" if len(_eid_s) >= 4 else "…****"
            examples.append(
                {
                    "sys_employee_id": eid,
                    "employee_id_label": _mask,
                    "work_date_iso": workdate_to_iso(wdi),
                    "total_hours": round(float(r["WORK_HRS_NUM"]), 2),
                    "by_job": jobs[:8],
                }
            )

    if n_over == 0:
        severity = "none"
    elif n_over <= 2 and share < 0.001:
        severity = "isolated"
    elif share < 0.02 and n_over < 40:
        severity = "occasional"
    else:
        severity = "frequent"

    # Footnote / banner: hide visually for “needle in haystack” (≤2 hits in a large scope, tiny share).
    if n_over == 0:
        ui_tier = "hidden"
    elif n_over <= 2 and n_scope >= 800 and share < 0.0005:
        ui_tier = "hidden"
    elif n_over <= 4 and share < 0.003:
        ui_tier = "subtle"
    elif severity == "frequent" or share >= 0.01 or n_over >= 25:
        ui_tier = "prominent"
    else:
        ui_tier = "standard"

    if share <= 0:
        share_pct_disp = "0%"
    elif share < 1e-6:
        share_pct_disp = "<0.0001%"
    elif share < 1e-4:
        share_pct_disp = f"{share * 100.0:.4f}%"
    else:
        share_pct_disp = f"{share * 100.0:.3f}%"

    tip_lines: list[str] = []
    for ex in examples[:3]:
        iso = ex.get("work_date_iso") or "?"
        lab = ex.get("employee_id_label") or "…"
        th = ex.get("total_hours")
        tip_lines.append(f"{iso} · {lab}: {th} h (sum over job codes that day)")
    tooltip_detail = " · ".join(tip_lines) if tip_lines else None

    return {
        "available": True,
        "hours_cap": float(hours_cap),
        "comparison": "strictly_greater_than",
        "employee_days_in_scope": n_scope,
        "employee_days_over_cap": n_over,
        "share_over_cap": round(share, 8),
        "share_over_cap_pct_display": share_pct_disp,
        "max_hours_any_employee_day": round(mx_all, 2),
        "severity": severity,
        "ui_tier": ui_tier,
        "tooltip_detail": tooltip_detail,
        "cms_spec_note": empty_out["cms_spec_note"],
        "caveats": list(empty_out["caveats"]),
        "examples": examples,
    }


def workdate_to_iso(wd: float | int) -> str | None:
    """YYYYMMDD int/float to YYYY-MM-DD."""
    try:
        v = int(wd)
    except (TypeError, ValueError):
        return None
    s = str(v).zfill(8)
    if len(s) != 8 or not s.isdigit():
        return None
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"


def _coerce_yyyymmdd_int(val: object) -> int | None:
    """Best-effort CMS WorkDate as YYYYMMDD int; ``None`` if unusable."""
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    if isinstance(val, int):
        n = val
    elif isinstance(val, float):
        if val != val:  # NaN
            return None
        n = int(val)
    elif isinstance(val, str):
        s = val.strip()
        if not s or not s.isdigit():
            return None
        n = int(s)
    else:
        return None
    if n <= 0:
        return None
    z = str(n).zfill(8)
    if len(z) != 8 or not z.isdigit():
        return None
    return n


def ein_span_days_from_workdate_raws(lo_raw: object, hi_raw: object) -> int | None:
    """Days between first/last CMS WorkDate ints on roster rows (full-file career span)."""
    lo = _coerce_yyyymmdd_int(lo_raw)
    hi = _coerce_yyyymmdd_int(hi_raw)
    if lo is None or hi is None:
        return None
    return _ein_span_days_yyyymmdd(lo, hi)


def _ein_span_days_yyyymmdd(lo_wd: int, hi_wd: int) -> int | None:
    """Days between CMS WorkDate ints (from first work day to reference day)."""
    try:
        from datetime import datetime as _dt

        d1 = _dt.strptime(str(int(lo_wd)).zfill(8), "%Y%m%d")
        d2 = _dt.strptime(str(int(hi_wd)).zfill(8), "%Y%m%d")
        days = (d2 - d1).days
        return days if days >= 0 else None
    except Exception:
        return None


def format_ein_tenure_span_days(days: int | None, *, at_least: bool = False) -> str | None:
    """Short label for tenure from first work day to a reference day.

    When ``at_least`` is True, the label uses a trailing ``+`` (e.g. ``5.5+ yr``) because the
    employee's first appearance matches the earliest quarter in the loaded extract — tenure
    in-role may extend before the file window.
    """
    if days is None or days < 0:
        return None
    if days < 14:
        return "<1 mo"
    if days < 365:
        mo = max(1, int(round(days / 30.44)))
        return f"{mo}+ mo" if at_least else f"{mo} mo"
    yr = days / 365.25
    return f"{yr:.1f}+ yr" if at_least else f"{yr:.1f} yr"


def previous_cy_quarter_label(quarter: str | None) -> str | None:
    """Calendar quarter immediately before ``quarter`` (CYyyyyQn), or None if unknown."""
    sk = parse_ein_quarter_bound(quarter)
    if sk is None:
        return None
    return ein_quarter_sort_key_to_label(sk - 1)


# Roster footnote when CMS SYS_EMPLEE_ID carryover drops sharply quarter-over-quarter.
EIN_ID_CONTINUITY_SHARP_DROP = 0.15
EIN_ID_CONTINUITY_BASELINE_MIN = 0.50
EIN_ID_CONTINUITY_MIN_HEADCOUNT = 10


def distinct_employee_ids_in_quarter_pairs(
    pairs_by_q: dict[str, set[tuple[int, int]]],
    quarter: str,
    *,
    job_codes: frozenset[int] | set[int] | None = None,
) -> set[int]:
    """Distinct ``SYS_EMPLEE_ID`` values with any job code in ``quarter`` (optional job filter)."""
    qn = normalize_cy_qtr_ein(quarter)
    if not qn or qn not in pairs_by_q:
        return set()
    allowed = set(job_codes) if job_codes is not None else None
    out: set[int] = set()
    for e, j in pairs_by_q[qn]:
        try:
            jc = int(j)
            eid = int(e)
        except (TypeError, ValueError):
            continue
        if allowed is not None and jc not in allowed:
            continue
        out.add(eid)
    return out


def employee_id_overlap_ratio(current: set[int], prior: set[int]) -> float | None:
    """Share of ``current`` IDs that also appear in ``prior`` (0.0 if ``prior`` is empty)."""
    if not current:
        return None
    if not prior:
        return 0.0
    return float(len(current & prior)) / float(len(current))


def detect_employee_id_continuity_warning(
    pairs_by_q: dict[str, set[tuple[int, int]]],
    quarter: str,
    *,
    job_codes: frozenset[int] | set[int] | None = None,
    min_headcount: int = EIN_ID_CONTINUITY_MIN_HEADCOUNT,
    sharp_drop_threshold: float = EIN_ID_CONTINUITY_SHARP_DROP,
    baseline_min_overlap: float = EIN_ID_CONTINUITY_BASELINE_MIN,
) -> dict[str, Any] | None:
    """
    Flag when ``SYS_EMPLEE_ID`` carryover drops sharply vs the prior quarter while the prior
  transition looked normal (suggests CMS ID re-keying, not true 100% turnover).

    Returns a dict with ``active: True`` for UI footnotes, or None when no warning applies.
    """
    qn = normalize_cy_qtr_ein(quarter)
    if not qn:
        return None
    pq = previous_cy_quarter_label(qn)
    if not pq or pq not in pairs_by_q:
        return None
    ppq = previous_cy_quarter_label(pq)
    if not ppq or ppq not in pairs_by_q:
        return None

    curr_e = distinct_employee_ids_in_quarter_pairs(pairs_by_q, qn, job_codes=job_codes)
    prev_e = distinct_employee_ids_in_quarter_pairs(pairs_by_q, pq, job_codes=job_codes)
    prior_e = distinct_employee_ids_in_quarter_pairs(pairs_by_q, ppq, job_codes=job_codes)
    if len(curr_e) < min_headcount or len(prev_e) < min_headcount:
        return None

    overlap_now = employee_id_overlap_ratio(curr_e, prev_e)
    overlap_baseline = employee_id_overlap_ratio(prev_e, prior_e)
    if overlap_now is None or overlap_baseline is None:
        return None
    if overlap_now > sharp_drop_threshold:
        return None
    if overlap_baseline < baseline_min_overlap:
        return None

    return {
        "active": True,
        "quarter": qn,
        "previous_quarter": pq,
        "baseline_quarter": ppq,
        "overlap_with_prior_pct": round(100.0 * overlap_now, 1),
        "baseline_overlap_pct": round(100.0 * overlap_baseline, 1),
        "distinct_employees_current": len(curr_e),
        "distinct_employees_prior": len(prev_e),
        "shared_ids": len(curr_e & prev_e),
    }


def employee_job_pair_existed_any_prior_quarter_in_extract(
    pairs_by_q: dict[str, set[tuple[int, int]]],
    quarter: str,
    sys_employee_id: int,
    job_code: int,
) -> bool | None:
    """True if (employee, job_code) appears in any **strictly earlier** CY quarter present in ``pairs_by_q``.

    Used for the roster **New** badge: show New only when this pair has no earlier quarter in the
    loaded Employee Detail extract (not merely “missing from last calendar quarter”).
    Returns None if ``quarter`` cannot be ordered.
    """
    qn = normalize_cy_qtr_ein(quarter)
    if not qn:
        return None
    sk_curr = parse_ein_quarter_bound(qn)
    if sk_curr is None:
        return None
    pair = (int(sys_employee_id), int(job_code))
    for qlabel, pset in pairs_by_q.items():
        if pair not in pset:
            continue
        sk = parse_ein_quarter_bound(qlabel)
        if sk is None:
            continue
        if sk < sk_curr:
            return True
    return False


def roster_pairs_by_quarter_from_rows(rows: list[dict[str, Any]]) -> dict[str, set[tuple[int, int]]]:
    """Map normalized quarter → set of (sys_employee_id, job_code) from roster summary dicts."""
    out: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for r in rows:
        qraw = r.get("quarter")
        qn = normalize_cy_qtr_ein(qraw) if qraw is not None else None
        if not qn:
            continue
        try:
            eid = int(r["sys_employee_id"])
            jcid = int(r["job_code"])
        except (KeyError, TypeError, ValueError):
            continue
        out[qn].add((eid, jcid))
    return dict(out)


def roster_pairs_by_quarter_for_job_codes(
    d: pd.DataFrame,
    job_codes: frozenset[int],
) -> dict[str, set[tuple[int, int]]]:
    """Build quarter → (employee, job) sets from ``prepare_ein_detail`` output, scoped to ``job_codes``."""
    if d.empty or not job_codes:
        return {}
    sub = d[d["EMPLEE_JOB_CD_ID"].astype(int).isin(list(job_codes))]
    if sub.empty:
        return {}
    keys = sub[["CY_Qtr_norm", "SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID"]].drop_duplicates()
    out: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for _, row in keys.iterrows():
        qn_raw = row["CY_Qtr_norm"]
        if pd.isna(qn_raw):
            continue
        qns = normalize_cy_qtr_ein(qn_raw)
        if not qns:
            continue
        try:
            eid = int(row["SYS_EMPLEE_ID"])
            jcid = int(row["EMPLEE_JOB_CD_ID"])
        except (TypeError, ValueError):
            continue
        out[qns].add((eid, jcid))
    return dict(out)


def roster_pairs_by_quarter_from_prepared_detail(d: pd.DataFrame) -> dict[str, set[tuple[int, int]]]:
    """Build quarter → (employee, job) sets from ``prepare_ein_detail`` output (nursing roster codes)."""
    return roster_pairs_by_quarter_for_job_codes(d, NURSING_JOB_CODE_IDS)


def roster_pairs_by_quarter_from_detail(detail: pd.DataFrame) -> dict[str, set[tuple[int, int]]]:
    """Same as ``roster_pairs_by_quarter_from_rows`` but from raw row-level detail."""
    return roster_pairs_by_quarter_from_prepared_detail(prepare_ein_detail(detail))


def enrich_nursing_rows_multi_role_flags(rows: list[dict[str, Any]]) -> None:
    """
    Set ``is_multi_role_employee`` and ``multi_role_job_titles`` on roster rows (mutates in place).

    **Dual role** = same ``sys_employee_id`` with more than one distinct nursing job code
    (Administrator 1 or codes 5–12) anywhere in the passed roster rows (typically the full
    loaded extract, not a single-quarter slice).
    """
    if not rows:
        return
    by_emp: dict[int, set[int]] = defaultdict(set)
    for r in rows:
        try:
            eid = int(r["sys_employee_id"])
            jc = int(r["job_code"])
        except (KeyError, TypeError, ValueError):
            continue
        by_emp[eid].add(jc)
    titles_by_emp: dict[int, list[str]] = {}
    for eid, codes in by_emp.items():
        if len(codes) > 1:
            titles_by_emp[eid] = sorted(
                {job_title_short(c) for c in codes},
                key=lambda t: str(t).lower(),
            )
    for r in rows:
        try:
            eid = int(r["sys_employee_id"])
        except (KeyError, TypeError, ValueError):
            r["is_multi_role_employee"] = False
            r["multi_role_job_titles"] = []
            continue
        titles = titles_by_emp.get(eid)
        if titles:
            r["is_multi_role_employee"] = True
            r["multi_role_job_titles"] = titles
        else:
            r["is_multi_role_employee"] = False
            r["multi_role_job_titles"] = []


def enrich_nursing_rows_new_to_quarter_flags(rows: list[dict[str, Any]]) -> None:
    """
    Set ``is_new_to_quarter`` and ``new_to_quarter_known`` on each roster row (mutates in place).

    **New** = first CY quarter in the **loaded** extract where this (employee, job_code) pair
    appears (no strictly earlier quarter in ``pairs_by_q``). If the reference quarter cannot be
    ordered, the badge is suppressed (unknown).
    """
    if not rows:
        return
    pairs_by_q = roster_pairs_by_quarter_from_rows(rows)
    for r in rows:
        qn = normalize_cy_qtr_ein(r.get("quarter"))
        if not qn:
            r["is_new_to_quarter"] = False
            r["new_to_quarter_known"] = False
            continue
        try:
            eid = int(r["sys_employee_id"])
            jcid = int(r["job_code"])
        except (KeyError, TypeError, ValueError):
            r["is_new_to_quarter"] = False
            r["new_to_quarter_known"] = False
            continue
        existed = employee_job_pair_existed_any_prior_quarter_in_extract(pairs_by_q, qn, eid, jcid)
        if existed is None:
            r["is_new_to_quarter"] = False
            r["new_to_quarter_known"] = False
            continue
        r["new_to_quarter_known"] = True
        r["is_new_to_quarter"] = not existed


SUSTAINED_WORK_AGGREGATE_ROLE_GROUPS: frozenset[str] = frozenset({"Aide group", "All nursing"})


def sustained_work_role_group_for_job_code(job_code: int | str | None) -> str | None:
    """Map a nursing job code to the primary sustained-work role group (not aggregate buckets)."""
    try:
        jc = int(job_code)
    except (TypeError, ValueError):
        return None
    for rg_name, codes in SUSTAINED_WORK_ROLE_GROUP_CODES.items():
        if rg_name in SUSTAINED_WORK_AGGREGATE_ROLE_GROUPS:
            continue
        if jc in codes:
            return rg_name
    return None


def _atomic_role_groups_for_job_codes(job_codes: set[int] | frozenset[int] | None) -> frozenset[str]:
    """Distinct atomic role groups (CNA, RN, LPN, …) for a set of job codes — excludes aggregate buckets."""
    out: set[str] = set()
    for jc in job_codes or set():
        rg = sustained_work_role_group_for_job_code(jc)
        if rg:
            out.add(rg)
    return frozenset(out)


def _sustained_work_multi_role_same_day_metrics(
    emp_q_total: pd.DataFrame,
) -> tuple[int, list[str]]:
    """
    Count calendar days with 2+ atomic role groups and return sorted union of those labels.

    Uses job codes on each employee-day row only — never aggregate role buckets (Aide group, All nursing).
    """
    multi_count = 0
    all_labels: set[str] = set()
    if emp_q_total.empty:
        return 0, []
    for row in emp_q_total.itertuples(index=False):
        codes = row.job_codes if isinstance(row.job_codes, set) else set()
        groups = _atomic_role_groups_for_job_codes(codes)
        if len(groups) >= 2:
            multi_count += 1
            all_labels |= groups
    return multi_count, sorted(all_labels)


def _cross_role_narrative_sentence(cross_role_groups: Sequence[str] | None) -> str:
    labels = sorted({str(g).strip() for g in (cross_role_groups or []) if str(g).strip()})
    if labels:
        return (
            "The same reported ID also appears in multiple role groups "
            f"({', '.join(labels)}) on the same day in this quarter."
        )
    return "The same reported ID also appears in multiple role groups on the same day in this quarter."


def enrich_nursing_rows_sustained_work_flags(
    detail: pd.DataFrame,
    rows: list[dict[str, Any]],
    *,
    facility_ccn: str,
    employee_id_continuity_by_period: dict[str, dict[str, Any] | None] | None = None,
    limit_quarters: Sequence[str] | None = None,
) -> None:
    """
    Attach ``sustained_work_flag`` on each roster row when employee-level sustained-work
    review triggers match this employee, quarter, and role group (mutates in place).
    """
    if not rows:
        return
    if detail is None or detail.empty:
        for r in rows:
            r["sustained_work_flag"] = None
        return

    if limit_quarters:
        quarters = sorted(
            {
                qn
                for qn in (normalize_cy_qtr_ein(q) for q in limit_quarters)
                if qn
            }
        )
    else:
        quarters = sorted(
            {
                qn
                for qn in (normalize_cy_qtr_ein(r.get("quarter")) for r in rows)
                if qn
            }
        )
    if not quarters:
        for r in rows:
            r["sustained_work_flag"] = None
        return

    result = ein_sustained_work_pattern_flags(
        detail,
        facility_ccn=str(facility_ccn),
        period_keys=quarters,
        employee_id_continuity_by_period=employee_id_continuity_by_period,
    )
    flag_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    for flag in result.get("sustained_work_pattern_flags") or []:
        if not flag.get("employee_id"):
            continue
        qn = normalize_cy_qtr_ein(flag.get("quarter"))
        rg = str(flag.get("role_group") or "")
        if not qn or not rg or rg in SUSTAINED_WORK_AGGREGATE_ROLE_GROUPS:
            continue
        eid = str(int(flag["employee_id"]))
        key = (eid, qn, rg)
        existing = flag_index.get(key)
        if not existing or SUSTAINED_WORK_SEVERITY_ORDER.get(
            str(flag.get("severity") or ""), 0
        ) > SUSTAINED_WORK_SEVERITY_ORDER.get(str(existing.get("severity") or ""), 0):
            flag_index[key] = flag

    for r in rows:
        try:
            eid = str(int(r["sys_employee_id"]))
            jcid = int(r["job_code"])
            qn = normalize_cy_qtr_ein(r.get("quarter"))
        except (KeyError, TypeError, ValueError):
            r["sustained_work_flag"] = None
            continue
        rg = sustained_work_role_group_for_job_code(jcid)
        if not rg or not qn:
            r["sustained_work_flag"] = None
            continue
        flag = flag_index.get((eid, qn, rg))
        if flag and str(flag.get("severity") or "") not in SUSTAINED_WORK_REVIEW_LEVELS:
            r["sustained_work_flag"] = None
            continue
        if flag:
            attached = dict(flag)
            attached["sys_employee_id"] = int(eid)
            attached["job_code"] = jcid
            r["sustained_work_flag"] = attached
        else:
            r["sustained_work_flag"] = None


def employee_job_pair_new_to_quarter(
    pairs_by_q: dict[str, set[tuple[int, int]]],
    quarter: str,
    sys_employee_id: int,
    job_code: int,
) -> bool | None:
    """True if this is the pair's first quarter in the loaded extract; False if they appear earlier; None if unknown."""
    existed = employee_job_pair_existed_any_prior_quarter_in_extract(
        pairs_by_q, quarter, sys_employee_id, job_code
    )
    if existed is None:
        return None
    return not existed


def compute_ein_quarter_roster_summary(
    quarter: str,
    pairs_by_q: dict[str, set[tuple[int, int]]],
) -> dict[str, Any] | None:
    """
    Headcounts and “new vs prior quarter” stats for one roster quarter.

    Distinct people can appear in both nurse and admin buckets; direct nursing is a subset of
    nurse. ``new_*`` counts are **distinct employees** (by ID) unless noted. Those counts compare
    to the **immediately previous calendar quarter** only; they are unrelated to the roster
    **New** badge (first quarter in the loaded extract for that employee–job pair).
    """
    qn = normalize_cy_qtr_ein(quarter)
    if not qn or qn not in pairs_by_q:
        return None
    curr = pairs_by_q[qn]
    pq = previous_cy_quarter_label(qn)
    prev = pairs_by_q.get(pq, set()) if pq else set()
    prior_ok = bool(pq and pq in pairs_by_q)

    def _eids_for_pairs(pairs: set[tuple[int, int]], jc_allowed: frozenset[int] | set[int]) -> set[int]:
        return {e for (e, j) in pairs if int(j) in jc_allowed}

    def _eids_all(pairs: set[tuple[int, int]]) -> set[int]:
        return {e for (e, _) in pairs}

    total_e = _eids_all(curr)
    nurse_e = _eids_for_pairs(curr, EIN_NURSING_JOB_CODES)
    direct_e = _eids_for_pairs(curr, EIN_DIRECT_NURSING_JOB_CODES)
    admin_e = _eids_for_pairs(curr, {EIN_ADMIN_JOB_CODE})

    prev_total_e = _eids_all(prev)
    prev_nurse_e = _eids_for_pairs(prev, EIN_NURSING_JOB_CODES)
    prev_direct_e = _eids_for_pairs(prev, EIN_DIRECT_NURSING_JOB_CODES)
    prev_admin_pairs = {(e, j) for (e, j) in prev if int(j) == EIN_ADMIN_JOB_CODE}

    new_total_e = len(total_e - prev_total_e) if prior_ok else None
    new_nurse_e = len(nurse_e - prev_nurse_e) if prior_ok else None
    new_direct_e = len(direct_e - prev_direct_e) if prior_ok else None
    new_admin_roles = (
        sum(1 for (e, j) in curr if int(j) == EIN_ADMIN_JOB_CODE and (e, j) not in prev)
        if prior_ok
        else None
    )

    def _ratio(num: int | None, den: int) -> float | None:
        if num is None or den <= 0:
            return None
        return round(100.0 * float(num) / float(den), 1)

    out: dict[str, Any] = {
        "quarter": qn,
        "previous_quarter": pq if prior_ok else None,
        "prior_quarter_in_file": prior_ok,
        "distinct_employees_total": len(total_e),
        "distinct_employees_nurse": len(nurse_e),
        "distinct_employees_direct_nursing": len(direct_e),
        "distinct_employees_admin_job": len(admin_e),
        "new_distinct_employees_total": new_total_e,
        "new_distinct_employees_nurse": new_nurse_e,
        "new_distinct_employees_direct_nursing": new_direct_e,
        "new_admin_job_assignments": new_admin_roles,
        "pct_new_of_total_headcount": _ratio(new_total_e, len(total_e)),
        "pct_new_nurse_of_nurse_headcount": _ratio(new_nurse_e, len(nurse_e)),
        "pct_new_direct_of_direct_headcount": _ratio(new_direct_e, len(direct_e)),
    }
    warn = detect_employee_id_continuity_warning(pairs_by_q, qn, job_codes=NURSING_JOB_CODE_IDS)
    if warn:
        out["employee_id_continuity_warning"] = warn
    return out


def compute_ein_quarter_nonnurse_roster_summary(
    quarter: str,
    pairs_by_q: dict[str, set[tuple[int, int]]],
) -> dict[str, Any] | None:
    """Quarter headcount / new-employee stats for non-nurse PBJ-mapped EIN job codes only."""
    from facility_ein_lib import NONNURSE_PBJ_EIN_JOB_CODES

    qn = normalize_cy_qtr_ein(quarter)
    if not qn or qn not in pairs_by_q:
        return None
    curr = {(e, j) for (e, j) in pairs_by_q[qn] if int(j) in NONNURSE_PBJ_EIN_JOB_CODES}
    pq = previous_cy_quarter_label(qn)
    prev = (
        {(e, j) for (e, j) in pairs_by_q[pq] if int(j) in NONNURSE_PBJ_EIN_JOB_CODES}
        if pq and pq in pairs_by_q
        else set()
    )
    prior_ok = bool(pq and pq in pairs_by_q)

    def _eids(pairs: set[tuple[int, int]]) -> set[int]:
        return {e for (e, _) in pairs}

    nn_e = _eids(curr)
    prev_nn_e = _eids(prev)
    new_nn_e = len(nn_e - prev_nn_e) if prior_ok else None

    def _ratio(num: int | None, den: int) -> float | None:
        if num is None or den <= 0:
            return None
        return round(100.0 * float(num) / float(den), 1)

    out: dict[str, Any] = {
        "quarter": qn,
        "previous_quarter": pq if prior_ok else None,
        "prior_quarter_in_file": prior_ok,
        "distinct_employees_nonnurse": len(nn_e),
        "new_distinct_employees_nonnurse": new_nn_e,
        "pct_new_of_nonnurse_headcount": _ratio(new_nn_e, len(nn_e)),
    }
    warn = detect_employee_id_continuity_warning(
        pairs_by_q, qn, job_codes=NONNURSE_PBJ_EIN_JOB_CODES
    )
    if warn:
        out["employee_id_continuity_warning"] = warn
    return out


def nursing_employee_summaries(
    detail: pd.DataFrame,
    quarter: str | None = None,
) -> list[dict[str, Any]]:
    """
    One summary per (SYS_EMPLEE_ID, EMPLEE_JOB_CD_ID, CY_Qtr_norm) for roster job codes (1 and 5–12).

    Quarter-specific: hours, work days, min/max hours dates, contract share.
    ``first_work_date`` / ``last_work_date`` (and *_raw): earliest and latest work day for that
    employee+job across **all** loaded detail rows, not only the summary quarter.
    """
    d = prepare_ein_detail(detail)
    if d.empty:
        return []
    d = d[d["EMPLEE_JOB_CD_ID"].astype(int).isin(list(NURSING_JOB_CODE_IDS))]
    if d.empty:
        return []

    # Earliest/latest calendar work day per (employee, job) across all quarters in ``detail``
    gspan = d.groupby(["SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID"], sort=False)["WorkDate"].agg(["min", "max"])
    global_first_last: dict[tuple[int, int], tuple[int, int]] = {}
    for (eid_raw, jcid_raw), row in gspan.iterrows():
        global_first_last[(int(eid_raw), int(jcid_raw))] = (int(row["min"]), int(row["max"]))

    if quarter and str(quarter).strip().lower() not in ("", "all"):
        qn = normalize_cy_qtr_ein(quarter)
        if qn:
            d = d[d["CY_Qtr_norm"] == qn]
    if d.empty:
        return []

    # Contract hours share from row-level (before daily rollup)
    def _contract_pct(sub: pd.DataFrame) -> float:
        tot = float(sub["WORK_HRS_NUM"].sum())
        if tot <= 0:
            return 0.0
        c = float(sub.loc[sub["EMP_CTR"] == 2, "WORK_HRS_NUM"].sum())
        return round(100.0 * c / tot, 1)

    keys = ["SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID", "CY_Qtr_norm"]
    out: list[dict[str, Any]] = []
    for (eid_raw, jcid_raw, qnorm), sub in d.groupby(keys, sort=False):
        eid, jcid = int(eid_raw), int(jcid_raw)
        qnorm = str(qnorm)
        daily = sub.groupby("WorkDate", as_index=False)["WORK_HRS_NUM"].sum()
        daily = daily.sort_values("WorkDate")
        hrs = daily["WORK_HRS_NUM"].astype(float)
        work_days = int(len(daily))
        total_h = float(hrs.sum())
        avg_day = round(total_h / work_days, 2) if work_days else 0.0
        mn = round(float(hrs.min()), 2) if work_days else 0.0
        mx = round(float(hrs.max()), 2) if work_days else 0.0
        first_wd = int(daily["WorkDate"].iloc[0])
        last_wd = int(daily["WorkDate"].iloc[-1])
        arr = hrs.to_numpy()
        imn = int(arr.argmin()) if work_days else 0
        imx = int(arr.argmax()) if work_days else 0
        min_hours_work_date = workdate_to_iso(int(daily["WorkDate"].iloc[imn])) if work_days else None
        max_hours_work_date = workdate_to_iso(int(daily["WorkDate"].iloc[imx])) if work_days else None
        g_first, g_last = global_first_last.get((eid, jcid), (first_wd, last_wd))
        out.append(
            {
                "sys_employee_id": eid,
                "job_code": jcid,
                "job_title": job_title(jcid),
                "job_title_short": job_title_short(jcid),
                "quarter": qnorm,
                "work_days": work_days,
                "total_hours": round(total_h, 2),
                "avg_hours_per_work_day": avg_day,
                "min_hours_single_day": mn,
                "max_hours_single_day": mx,
                "first_work_date": workdate_to_iso(g_first),
                "last_work_date": workdate_to_iso(g_last),
                "first_work_date_raw": g_first,
                "last_work_date_raw": g_last,
                "min_hours_work_date": min_hours_work_date,
                "max_hours_work_date": max_hours_work_date,
                "pct_contract_hours": _contract_pct(pd.DataFrame(sub)),
                "display_id": str(eid),
            }
        )
    out.sort(
        key=lambda r: (
            -(parse_ein_quarter_bound(r["quarter"]) or 0),
            -float(r.get("total_hours") or 0),
            int(r.get("first_work_date_raw") or 0) or 10**9,
            int(r["sys_employee_id"]),
            int(r["job_code"]),
        )
    )
    return out


def nonnurse_employee_summaries(
    detail: pd.DataFrame,
    quarter: str | None = None,
) -> list[dict[str, Any]]:
    """One summary per (employee, job, quarter) for PBJ non-nurse mapped EIN codes (see ``NONNURSE_PBJ_EIN_JOB_CODES``)."""
    from facility_ein_lib import NONNURSE_PBJ_EIN_JOB_CODES

    d = prepare_ein_detail(detail)
    if d.empty:
        return []
    d = d[d["EMPLEE_JOB_CD_ID"].astype(int).isin(list(NONNURSE_PBJ_EIN_JOB_CODES))]
    if d.empty:
        return []

    gspan = d.groupby(["SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID"], sort=False)["WorkDate"].agg(["min", "max"])
    global_first_last: dict[tuple[int, int], tuple[int, int]] = {}
    for (eid_raw, jcid_raw), row in gspan.iterrows():
        global_first_last[(int(eid_raw), int(jcid_raw))] = (int(row["min"]), int(row["max"]))

    if quarter and str(quarter).strip().lower() not in ("", "all"):
        qn = normalize_cy_qtr_ein(quarter)
        if qn:
            d = d[d["CY_Qtr_norm"] == qn]
    if d.empty:
        return []

    def _contract_pct(sub: pd.DataFrame) -> float:
        tot = float(sub["WORK_HRS_NUM"].sum())
        if tot <= 0:
            return 0.0
        c = float(sub.loc[sub["EMP_CTR"] == 2, "WORK_HRS_NUM"].sum())
        return round(100.0 * c / tot, 1)

    keys = ["SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID", "CY_Qtr_norm"]
    out: list[dict[str, Any]] = []
    for (eid_raw, jcid_raw, qnorm), sub in d.groupby(keys, sort=False):
        eid, jcid = int(eid_raw), int(jcid_raw)
        qnorm = str(qnorm)
        daily = sub.groupby("WorkDate", as_index=False)["WORK_HRS_NUM"].sum()
        daily = daily.sort_values("WorkDate")
        hrs = daily["WORK_HRS_NUM"].astype(float)
        work_days = int(len(daily))
        total_h = float(hrs.sum())
        avg_day = round(total_h / work_days, 2) if work_days else 0.0
        g_first, g_last = global_first_last.get((eid, jcid), (int(daily["WorkDate"].iloc[0]), int(daily["WorkDate"].iloc[-1])))
        out.append(
            {
                "sys_employee_id": eid,
                "job_code": jcid,
                "job_title": job_title(jcid),
                "job_title_short": job_title_short(jcid),
                "quarter": qnorm,
                "work_days": work_days,
                "total_hours": round(total_h, 2),
                "avg_hours_per_work_day": avg_day,
                "first_work_date": workdate_to_iso(g_first),
                "last_work_date": workdate_to_iso(g_last),
                "first_work_date_raw": g_first,
                "last_work_date_raw": g_last,
                "pct_contract_hours": _contract_pct(pd.DataFrame(sub)),
                "display_id": str(eid),
            }
        )
    out.sort(
        key=lambda r: (
            -(parse_ein_quarter_bound(r["quarter"]) or 0),
            -float(r.get("total_hours") or 0),
            int(r.get("first_work_date_raw") or 0) or 10**9,
            int(r["sys_employee_id"]),
            int(r["job_code"]),
        )
    )
    return out


def _ein_date_to_yyyymmdd(d) -> int:
    return int(d.strftime("%Y%m%d"))


def ein_job_role_index_by_pair_from_scope(scope: pd.DataFrame) -> dict[tuple[int, int], str]:
    """
    Ordinal strings "1", "2", … per job code by earliest WorkDate in ``scope``,
    tie-breaking by ``SYS_EMPLEE_ID`` (same rule as ``enrich_nursing_roster_display_fields``).
    """
    if scope is None or scope.empty:
        return {}
    need = {"SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID", "WorkDate"}
    if not need.issubset(scope.columns):
        return {}
    pairs = (
        scope.groupby(["SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID"], sort=False)["WorkDate"]
        .min()
        .reset_index()
    )
    by_job: dict[int, dict[int, int]] = {}
    for _, r in pairs.iterrows():
        try:
            eid = int(r["SYS_EMPLEE_ID"])
            jc = int(r["EMPLEE_JOB_CD_ID"])
            wd0 = int(r["WorkDate"])
        except (TypeError, ValueError):
            continue
        by_job.setdefault(jc, {})
        by_job[jc][eid] = wd0
    letter_map: dict[tuple[int, int], str] = {}
    for jc, eid_to_first in by_job.items():
        ordered_eids = sorted(
            eid_to_first.keys(),
            key=lambda e: (eid_to_first[e] if eid_to_first[e] > 0 else 10**9, int(e)),
        )
        for idx, eid in enumerate(ordered_eids):
            letter_map[(eid, jc)] = str(idx + 1)
    return letter_map


def enrich_ein_rows_rolling_from_work_date(
    detail: pd.DataFrame,
    rows: list[dict[str, Any]],
    anchor_iso: str,
    job_code_allowlist: frozenset[int],
) -> None:
    """Add hours on anchor day plus L-30 / L-90 / L-365 work-day counts and avg hours (mutates rows)."""
    anchor_iso = str(anchor_iso).strip()
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", anchor_iso)
    if not m:
        return

    try:
        anchor_d = datetime.strptime(anchor_iso, "%Y-%m-%d").date()
    except ValueError:
        return

    lo30_d = anchor_d - timedelta(days=29)
    lo90_d = anchor_d - timedelta(days=89)
    lo365_d = anchor_d - timedelta(days=364)
    lo30_i = _ein_date_to_yyyymmdd(lo30_d)
    lo90_i = _ein_date_to_yyyymmdd(lo90_d)
    lo365_i = _ein_date_to_yyyymmdd(lo365_d)
    anchor_int = _ein_date_to_yyyymmdd(anchor_d)

    def _blank_rolling(row: dict[str, Any]) -> None:
        row["hours_selected_work_day"] = None
        row["work_days_last_30"] = None
        row["avg_hours_per_day_last_30"] = None
        row["work_days_last_90"] = None
        row["avg_hours_per_day_last_90"] = None
        row["work_days_last_365"] = None
        row["avg_hours_per_day_last_365"] = None
        row["day_contract_flag"] = False

    d = prepare_ein_detail(detail)
    d = d[d["EMPLEE_JOB_CD_ID"].astype(int).isin(list(job_code_allowlist))]
    if d.empty:
        for row in rows:
            _blank_rolling(row)
        return

    gg = (
        d.groupby(["SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID", "WorkDate"], sort=False)["WORK_HRS_NUM"]
        .sum()
        .reset_index()
    )
    pair_map: dict[tuple[int, int], dict[int, float]] = {}
    for _, r in gg.iterrows():
        eid = int(r["SYS_EMPLEE_ID"])
        jc = int(r["EMPLEE_JOB_CD_ID"])
        wd = int(r["WorkDate"])
        hrs = float(r["WORK_HRS_NUM"])
        pair_map.setdefault((eid, jc), {})[wd] = hrs

    contract_anchor: dict[tuple[int, int], bool] = {}
    try:
        sub_a = d[d["WorkDate"] == anchor_int]
        for _, r in sub_a.iterrows():
            k = (int(r["SYS_EMPLEE_ID"]), int(r["EMPLEE_JOB_CD_ID"]))
            try:
                is_ctr = int(r["EMP_CTR"]) == 2
            except (TypeError, ValueError):
                is_ctr = False
            contract_anchor[k] = contract_anchor.get(k, False) or is_ctr
    except Exception:
        contract_anchor = {}

    for row in rows:
        sid_raw = row.get("sys_employee_id")
        jcid_raw = row.get("job_code")
        if sid_raw is None or jcid_raw is None:
            _blank_rolling(row)
            continue
        try:
            eid = int(sid_raw)
            jc = int(jcid_raw)
        except (TypeError, ValueError):
            _blank_rolling(row)
            continue
        days = pair_map.get((eid, jc))
        if not days:
            _blank_rolling(row)
            continue

        row["hours_selected_work_day"] = (
            round(float(days[anchor_int]), 2) if anchor_int in days else None
        )
        w30 = {wd: h for wd, h in days.items() if lo30_i <= wd <= anchor_int}
        t30 = sum(w30.values())
        n30 = len(w30)
        row["work_days_last_30"] = int(n30)
        row["avg_hours_per_day_last_30"] = round(t30 / n30, 2) if n30 else None

        w90 = {wd: h for wd, h in days.items() if lo90_i <= wd <= anchor_int}
        t90 = sum(w90.values())
        n90 = len(w90)
        row["work_days_last_90"] = int(n90)
        row["avg_hours_per_day_last_90"] = round(t90 / n90, 2) if n90 else None

        w365 = {wd: h for wd, h in days.items() if lo365_i <= wd <= anchor_int}
        t365 = sum(w365.values())
        n365 = len(w365)
        row["work_days_last_365"] = int(n365)
        row["avg_hours_per_day_last_365"] = round(t365 / n365, 2) if n365 else None
        row["day_contract_flag"] = bool(contract_anchor.get((eid, jc), False))


def enrich_nursing_rows_rolling_from_work_date(
    detail: pd.DataFrame,
    rows: list[dict[str, Any]],
    anchor_iso: str,
) -> None:
    enrich_ein_rows_rolling_from_work_date(detail, rows, anchor_iso, NURSING_JOB_CODE_IDS)


def enrich_nonnurse_rows_rolling_from_work_date(
    detail: pd.DataFrame,
    rows: list[dict[str, Any]],
    anchor_iso: str,
) -> None:
    from facility_ein_lib import NONNURSE_PBJ_EIN_JOB_CODES

    enrich_ein_rows_rolling_from_work_date(detail, rows, anchor_iso, NONNURSE_PBJ_EIN_JOB_CODES)


def _ein_letter_from_index(idx: int) -> str:
    """Map 0 -> A, 25 -> Z, 26 -> AA (per-role roster labels)."""
    if idx < 0:
        return "?"
    n = idx + 1
    letters = ""
    while n > 0:
        n, rem = divmod(n - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def _row_first_work_raw(row: dict[str, Any]) -> int:
    """Earliest work-day int (YYYYMMDD) for letter ordering; 0 if unknown."""
    fr = row.get("first_work_date_raw")
    if fr is not None:
        try:
            v = int(fr)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    iso = row.get("first_work_date")
    if iso:
        wd = work_date_iso_to_yyyymmdd(str(iso))
        return wd or 0
    return 0


def _row_last_work_raw(row: dict[str, Any]) -> int:
    """Latest work-day int (YYYYMMDD) for ordering roles by recency; 0 if unknown."""
    fr = row.get("last_work_date_raw")
    if fr is not None:
        try:
            v = int(fr)
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
    iso = row.get("last_work_date")
    if iso:
        wd = work_date_iso_to_yyyymmdd(str(iso))
        return wd or 0
    return 0


def dedupe_nursing_roster_api_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (sys_employee_id, job_code, quarter); drop Parquet/extract duplicates.

    When duplicates exist, keep the row with the highest ``total_hours``, then ``work_days``.
    """
    if len(rows) < 2:
        return rows

    def stable_key(r: dict[str, Any]) -> tuple[int, int, str]:
        sid_r = r.get("sys_employee_id")
        jc_r = r.get("job_code")
        try:
            sid = int(sid_r) if sid_r is not None else -1
        except (TypeError, ValueError):
            sid = -1
        try:
            jc = int(jc_r) if jc_r is not None else -1
        except (TypeError, ValueError):
            jc = -1
        qn = normalize_cy_qtr_ein(r.get("quarter")) or str(r.get("quarter") or "").strip()
        return (sid, jc, qn)

    def score(r: dict[str, Any]) -> tuple[float, int]:
        try:
            th = float(r.get("total_hours") or 0)
        except (TypeError, ValueError):
            th = 0.0
        try:
            wd = int(r.get("work_days") or 0)
        except (TypeError, ValueError):
            wd = 0
        return (th, wd)

    best: dict[tuple[int, int, str], dict[str, Any]] = {}
    for r in rows:
        k = stable_key(r)
        if k not in best:
            best[k] = r
            continue
        s_new, s_old = score(r), score(best[k])
        if s_new > s_old:
            best[k] = r
        elif s_new == s_old:
            # Deterministic tie: same hours/days → keep row with larger work_date span if present
            span_new = _row_last_work_raw(r) - _row_first_work_raw(r)
            span_old = _row_last_work_raw(best[k]) - _row_first_work_raw(best[k])
            if span_new > span_old:
                best[k] = r
    return list(best.values())


def apply_roster_tenure_quarter_span(
    rows: list[dict[str, Any]],
    *,
    rows_for_global_bounds: list[dict[str, Any]] | None = None,
) -> None:
    """Mutate rows: set ``tenure_quarter_lo`` / ``tenure_quarter_hi`` (``CYyyyyQn``) per employee × job.

    Computed from the **full** ``rows`` list (e.g. all quarters matching filters) so pagination
    does not shrink the displayed quarter span in the UI.

    ``rows_for_global_bounds`` should be the unfiltered facility roster (all loaded quarters) when
    ``rows`` is narrowed to one quarter — otherwise earliest-quarter detection would mark everyone
    in that quarter as censored. When omitted, bounds use ``rows``.
    """
    if not rows:
        return
    bounds = rows_for_global_bounds if rows_for_global_bounds is not None else rows
    global_min_sk: int | None = None
    file_lo_sk: dict[tuple[int, int], int] = {}
    for r in bounds:
        try:
            sid = int(r["sys_employee_id"])
            jc = int(r["job_code"])
        except (KeyError, TypeError, ValueError):
            continue
        qn = normalize_cy_qtr_ein(r.get("quarter"))
        if not qn:
            continue
        sk = parse_ein_quarter_bound(qn)
        if sk is None:
            continue
        key = (sid, jc)
        prev = file_lo_sk.get(key)
        if prev is None or sk < prev:
            file_lo_sk[key] = sk
        if global_min_sk is None or sk < global_min_sk:
            global_min_sk = sk

    span: dict[tuple[int, int], tuple[int, int]] = {}
    for r in rows:
        try:
            sid = int(r["sys_employee_id"])
            jc = int(r["job_code"])
        except (KeyError, TypeError, ValueError):
            continue
        qn = normalize_cy_qtr_ein(r.get("quarter"))
        if not qn:
            continue
        sk = parse_ein_quarter_bound(qn)
        if sk is None:
            continue
        key = (sid, jc)
        if key not in span:
            span[key] = (sk, sk)
        else:
            lo, hi = span[key]
            span[key] = (min(lo, sk), max(hi, sk))
    for r in rows:
        try:
            sid = int(r["sys_employee_id"])
            jc = int(r["job_code"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (sid, jc)
        t = span.get(key)
        if not t:
            r["tenure_quarter_lo"] = None
            r["tenure_quarter_hi"] = None
            r["tenure_censored_at_file_start"] = False
            continue
        lo, hi = t
        r["tenure_quarter_lo"] = ein_quarter_sort_key_to_label(lo)
        r["tenure_quarter_hi"] = ein_quarter_sort_key_to_label(hi)
        flo = file_lo_sk.get(key)
        r["tenure_censored_at_file_start"] = bool(
            flo is not None and global_min_sk is not None and flo == global_min_sk
        )


def enrich_nursing_roster_display_fields(rows: list[dict[str, Any]]) -> None:
    """
    Mutate summary rows: ``job_role_letter`` and ``id_position_display``.

    **Position numbers (#1, #2, …)** are assigned **per CMS job code** (1, 5–12), in order of
    **first work day in the loaded Employee Detail extract** for that employee in that role
    (earliest ``first_work_date`` / ``first_work_date_raw``), then stable tie-break by
    ``sys_employee_id``. The same person can be #2 for one code and #1 for another.
    """
    if not rows:
        return
    by_job: dict[int, dict[int, int]] = {}
    for r in rows:
        try:
            jc = int(r["job_code"])
            eid = int(r["sys_employee_id"])
        except (KeyError, TypeError, ValueError):
            continue
        raw = _row_first_work_raw(r)
        by_job.setdefault(jc, {})
        prev = by_job[jc].get(eid)
        # Keep earliest first-work YYYYMMDD seen for this (employee, job code) across quarters.
        if prev is None or (raw > 0 and (prev == 0 or raw < prev)):
            by_job[jc][eid] = raw
    letter_map: dict[tuple[int, int], str] = {}
    for jc, eid_to_first in by_job.items():
        ordered_eids = sorted(
            eid_to_first.keys(),
            key=lambda e: (
                eid_to_first[e] if eid_to_first[e] > 0 else 10**9,
                int(e),
            ),
        )
        for idx, eid in enumerate(ordered_eids):
            letter_map[(eid, jc)] = str(idx + 1)

    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        try:
            key = (int(r["sys_employee_id"]), str(r.get("quarter") or ""))
        except (TypeError, ValueError):
            continue
        groups[key].append(r)

    for (eid_key, _qtr), grp in groups.items():
        # Most recent role first (last work day in quarter), then stable tie-break by job code
        grp_sorted = sorted(
            grp,
            key=lambda row: (
                -(_row_last_work_raw(row) if _row_last_work_raw(row) > 0 else 0),
                int(row.get("job_code") or 0),
            ),
        )
        parts: list[str] = []
        for row in grp_sorted:
            try:
                jc = int(row["job_code"])
                eid = int(row["sys_employee_id"])
            except (KeyError, TypeError, ValueError):
                continue
            short = str(row.get("job_title_short") or row.get("job_title") or jc)
            ordinal = letter_map.get((eid, jc), "?")
            parts.append(f"{short} #{ordinal}")
        disp_id = ""
        if grp_sorted:
            disp_id = str(grp_sorted[0].get("display_id") or eid_key)
        if not disp_id:
            disp_id = str(eid_key)
        combined = f"{disp_id} ({', '.join(parts)})" if parts else disp_id
        for row in grp:
            row["id_position_display"] = combined
            try:
                jc = int(row["job_code"])
                eid = int(row["sys_employee_id"])
                row["job_role_letter"] = letter_map.get((eid, jc), "")
            except (KeyError, TypeError, ValueError):
                row["job_role_letter"] = ""


def nursing_employee_quarters_for_job(
    detail: pd.DataFrame,
    sys_employee_id: int,
    job_code: int,
) -> list[str]:
    """Distinct CY quarters for one employee + job in the loaded extract (newest first)."""
    d = prepare_ein_detail(detail)
    if d.empty:
        return []
    try:
        sid = int(sys_employee_id)
        jcid = int(job_code)
    except (TypeError, ValueError):
        return []
    sub = d[(d["SYS_EMPLEE_ID"] == sid) & (d["EMPLEE_JOB_CD_ID"] == jcid)]
    if sub.empty:
        return []
    uniq = sub["CY_Qtr_norm"].dropna().astype(str).unique().tolist()
    uniq.sort(key=lambda x: parse_ein_quarter_bound(x) or 0, reverse=True)
    return uniq


def nursing_employee_daily_series(
    detail: pd.DataFrame,
    sys_employee_id: int,
    job_code: int,
    quarter: str,
) -> dict[str, Any] | None:
    """Longitudinal series: one point per calendar work day (hours that day).

    ``first_work_date`` / ``last_work_date`` span the **full** loaded extract for this
    employee+job. ``quarter_first_work_date`` / ``quarter_last_work_date`` bound the
    selected quarter only.
    """
    d = prepare_ein_detail(detail)
    if d.empty:
        return None
    qn = normalize_cy_qtr_ein(quarter)
    if not qn:
        return None
    sub = d[
        (d["SYS_EMPLEE_ID"] == int(sys_employee_id))
        & (d["EMPLEE_JOB_CD_ID"] == int(job_code))
        & (d["CY_Qtr_norm"] == qn)
    ]
    if sub.empty:
        return None
    daily = sub.groupby("WorkDate", as_index=False).agg(
        hours=("WORK_HRS_NUM", "sum"),
        any_contract=("EMP_CTR", lambda s: bool((s == 2).any())),
    )
    daily = daily.sort_values("WorkDate")
    dates = [workdate_to_iso(x) for x in daily["WorkDate"]]
    hours = [round(float(h), 2) for h in daily["hours"]]
    work_days = len(daily)
    total_h = float(daily["hours"].sum())
    avg_day = round(total_h / work_days, 2) if work_days else 0.0
    mn = round(float(daily["hours"].min()), 2) if work_days else 0.0
    mx = round(float(daily["hours"].max()), 2) if work_days else 0.0
    quarter_first_wd = int(daily["WorkDate"].iloc[0])
    quarter_last_wd = int(daily["WorkDate"].iloc[-1])
    sub_all = d[
        (d["SYS_EMPLEE_ID"] == int(sys_employee_id))
        & (d["EMPLEE_JOB_CD_ID"] == int(job_code))
    ]
    if not sub_all.empty:
        career_lo = int(sub_all["WorkDate"].min())
        career_hi = int(sub_all["WorkDate"].max())
    else:
        career_lo = quarter_first_wd
        career_hi = quarter_last_wd
    h_arr = daily["hours"].to_numpy(dtype=float)
    med_day = round(float(statistics.median(h_arr)), 2) if work_days else 0.0
    imn = int(h_arr.argmin()) if work_days else 0
    imx = int(h_arr.argmax()) if work_days else 0
    min_hours_work_date = workdate_to_iso(int(daily["WorkDate"].iloc[imn])) if work_days else None
    max_hours_work_date = workdate_to_iso(int(daily["WorkDate"].iloc[imx])) if work_days else None
    tot_rows = float(sub["WORK_HRS_NUM"].sum())
    ctr_h = float(sub.loc[sub["EMP_CTR"] == 2, "WORK_HRS_NUM"].sum()) if tot_rows > 0 else 0.0
    pct_ctr = round(100.0 * ctr_h / tot_rows, 1) if tot_rows > 0 else 0.0
    pairs_by_q = roster_pairs_by_quarter_from_prepared_detail(d)
    new_flag = employee_job_pair_new_to_quarter(pairs_by_q, qn, int(sys_employee_id), int(job_code))
    return {
        "sys_employee_id": int(sys_employee_id),
        "job_code": int(job_code),
        "job_title": job_title(job_code),
        "job_title_short": job_title_short(job_code),
        "quarter": qn,
        "is_new_to_quarter": new_flag,
        "dates": dates,
        "hours": hours,
        "day_contract_flag": [bool(x) for x in daily["any_contract"]],
        "work_days": work_days,
        "total_hours": round(total_h, 2),
        "avg_hours_per_work_day": avg_day,
        "median_hours_per_work_day": med_day,
        "min_hours_single_day": mn,
        "max_hours_single_day": mx,
        "quarter_first_work_date": workdate_to_iso(quarter_first_wd),
        "quarter_last_work_date": workdate_to_iso(quarter_last_wd),
        "first_work_date": workdate_to_iso(career_lo),
        "last_work_date": workdate_to_iso(career_hi),
        "min_hours_work_date": min_hours_work_date,
        "max_hours_work_date": max_hours_work_date,
        "pct_contract_hours": pct_ctr,
    }


def work_date_iso_to_yyyymmdd(iso: str) -> int | None:
    """Normalize ``YYYY-MM-DD`` (or ``YYYY/MM/DD``) to CMS WorkDate int."""
    s = str(iso).strip().replace("/", "-")
    parts = s.split("-")
    if len(parts) != 3:
        return None
    try:
        y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None
    if not (2000 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31):
        return None
    return y * 10_000 + mo * 100 + d


def cy_quarter_from_yyyymmdd(wd: int) -> str:
    """Calendar quarter label ``CYyyyyQn`` from WorkDate int."""
    s = str(int(wd)).zfill(8)
    y = int(s[:4])
    m = int(s[4:6])
    q = (m - 1) // 3 + 1
    return f"CY{y}Q{q}"


def aggregate_ein_day_by_job_codes(
    detail: pd.DataFrame,
    work_date_iso: str,
    job_codes: frozenset[int],
) -> tuple[list[dict[str, Any]], float, str | None]:
    """
    Sum Employee Detail hours for one calendar day and a set of job codes.

    Returns ``(rows sorted by hours desc, total_hours, CY_Qtr for CMS links)``.
    """
    wd = work_date_iso_to_yyyymmdd(work_date_iso)
    if wd is None:
        return [], 0.0, None
    d = prepare_ein_detail(detail)
    q_fallback = cy_quarter_from_yyyymmdd(wd)
    if d.empty:
        return [], 0.0, q_fallback
    wdi = int(wd)
    wd_series = pd.to_numeric(d["WorkDate"], errors="coerce")
    sub = d.loc[wd_series == wdi]
    if job_codes:
        codes_list = list(job_codes)
        sub = sub[sub["EMPLEE_JOB_CD_ID"].astype(int).isin(codes_list)]
    if sub.empty:
        return [], 0.0, q_fallback
    qn = str(sub["CY_Qtr_norm"].iloc[0])
    pairs_by_q = roster_pairs_by_quarter_for_job_codes(d, job_codes) if job_codes else {}
    if job_codes:
        scope = d.loc[d["EMPLEE_JOB_CD_ID"].astype(int).isin(list(job_codes))].copy()
    else:
        scope = d
    role_idx = ein_job_role_index_by_pair_from_scope(cast(pd.DataFrame, scope))
    pair_first_wd = scope.groupby(["SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID"], sort=False)["WorkDate"].min()
    wd_s = str(int(wd)).zfill(8)
    anchor_day = date(int(wd_s[:4]), int(wd_s[4:6]), int(wd_s[6:8]))
    lo90_i = _ein_date_to_yyyymmdd(anchor_day - timedelta(days=89))
    global_min_q_sk: int | None = None
    for qv in scope["CY_Qtr_norm"].dropna().astype(str).unique():
        qn_g = normalize_cy_qtr_ein(qv)
        if not qn_g:
            continue
        sk_g = parse_ein_quarter_bound(qn_g)
        if sk_g is None:
            continue
        if global_min_q_sk is None or sk_g < global_min_q_sk:
            global_min_q_sk = sk_g
    out: list[dict[str, Any]] = []
    total = 0.0
    for (eid, jcid), g in sub.groupby(["SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID"], sort=False):
        hrs = float(g["WORK_HRS_NUM"].sum())
        total += hrs
        ctr = float(g.loc[g["EMP_CTR"] == 2, "WORK_HRS_NUM"].sum())
        pct = round(100.0 * ctr / hrs, 1) if hrs > 0 else 0.0
        ji, eii = int(jcid), int(eid)
        try:
            lo_raw = int(pair_first_wd.loc[(eii, ji)])
        except (KeyError, TypeError, ValueError):
            lo_raw = wd
        span_days = _ein_span_days_yyyymmdd(lo_raw, wd)
        first_q = normalize_cy_qtr_ein(cy_quarter_from_yyyymmdd(lo_raw))
        first_sk = parse_ein_quarter_bound(first_q) if first_q else None
        censored = bool(
            first_sk is not None and global_min_q_sk is not None and first_sk == global_min_q_sk
        )
        tenure_label = format_ein_tenure_span_days(span_days, at_least=censored)
        nf = employee_job_pair_new_to_quarter(pairs_by_q, qn, eii, ji)
        new_known = nf is not None
        is_new = bool(nf) if new_known else False
        pair_scope = d[(d["SYS_EMPLEE_ID"] == eii) & (d["EMPLEE_JOB_CD_ID"] == ji)]
        wd_series = pair_scope["WorkDate"].dropna().astype(int)
        n90 = (
            int(wd_series[(wd_series >= lo90_i) & (wd_series <= wd)].nunique())
            if not pair_scope.empty and lo90_i > 0
            else 0
        )
        jr = str(role_idx.get((eii, ji), "") or "")
        out.append(
            {
                "sys_employee_id": eii,
                "job_code": ji,
                "job_title": job_title(ji),
                "job_title_short": job_title_short(ji),
                "hours": round(hrs, 2),
                "pct_contract_hours": pct,
                "quarter": qn,
                "display_id": str(eii),
                "first_work_date": workdate_to_iso(lo_raw),
                "tenure_days_at_work_date": span_days,
                "tenure_label": tenure_label,
                "tenure_censored_at_file_start": censored,
                "is_new_to_quarter": is_new,
                "new_to_quarter_known": new_known,
                "job_role_letter": jr,
                "work_days_last_90": int(n90),
            }
        )
    out.sort(key=lambda r: (-float(r["hours"]), str(r["job_title"]), r["sys_employee_id"]))
    return out, total, qn


def ein_pbj_bridge_for_metric(
    detail: pd.DataFrame,
    work_date_iso: str,
    metric_key: str,
) -> tuple[list[dict[str, Any]], float, str | None, str]:
    """PBJ single-day / daily column key → Employee Detail rows for that bucket."""
    from facility_ein_lib import PBJ_METRIC_BRIDGE_NOTES, PBJ_METRIC_TO_EIN_JOB_CODES

    codes = frozenset(PBJ_METRIC_TO_EIN_JOB_CODES.get(metric_key, ()))
    note = PBJ_METRIC_BRIDGE_NOTES.get(metric_key, "")
    if not codes:
        return [], 0.0, None, note or "No Employee Detail mapping for this metric."
    rows, total, qn = aggregate_ein_day_by_job_codes(detail, work_date_iso, codes)
    return rows, total, qn, note


def ein_headcount_buckets_for_work_date(detail: pd.DataFrame, work_date_iso: str) -> dict[str, Any]:
    """Distinct ``SYS_EMPLEE_ID`` counts on one calendar work day (positive hours only).

    Uses **all** CMS job codes present in the row-level extract (not only the nursing roster).
    Bucket counts can overlap (e.g. same ID in nurse and nurse_direct); ``distinct_total`` counts
    each person once across all roles that day. Intended as a secondary QA / context metric
    (e.g. future wage linkage by state), not a substitute for CMS-published PBJ HPRD.
    """
    wd = work_date_iso_to_yyyymmdd(work_date_iso)
    if wd is None:
        return {}
    d = prepare_ein_detail(detail)
    if d.empty:
        return {}
    wdi = int(wd)
    wd_series = pd.to_numeric(d["WorkDate"], errors="coerce")
    day = d.loc[wd_series == wdi].copy()
    if day.empty:
        return {"work_date_int": wdi, "distinct_total": 0}
    hrs = pd.to_numeric(day["WORK_HRS_NUM"], errors="coerce").fillna(0.0)
    day = day.loc[hrs > 0]
    if day.empty:
        return {"work_date_int": wdi, "distinct_total": 0}
    jc = day["EMPLEE_JOB_CD_ID"].astype(int)
    eid = day["SYS_EMPLEE_ID"].astype(int)

    def _uniq(mask: Any) -> int:
        return int(eid.loc[mask].drop_duplicates().shape[0])

    nursing = jc.between(5, 12, inclusive="both")
    nurse_direct = jc.isin((7, 9, 10, 11, 12))
    nurse_lead = jc.isin((5, 6, 8))
    admin = jc.eq(1)
    non_nurse = ~(nursing | admin)

    return {
        "work_date_int": wdi,
        "distinct_total": int(day["SYS_EMPLEE_ID"].dropna().astype(int).nunique()),
        "distinct_admin": _uniq(admin),
        "distinct_nursing_total": _uniq(nursing),
        "distinct_nursing_direct": _uniq(nurse_direct),
        "distinct_nursing_leadership": _uniq(nurse_lead),
        "distinct_non_nurse": _uniq(non_nurse),
        "notes": (
            "Bucket counts use CMS EMPLEE_JOB_CD_ID (see EIN/data_dictionary.md). "
            "nursing_total overlaps nursing_direct and nursing_leadership; non_nurse is all other codes."
        ),
    }


# PBJ bridge metric → CMS job-code family (union) for same-day multi-role / cross-role hints.
_EIN_PBJ_METRIC_JOB_FAMILY: dict[str, frozenset[int]] = {
    "rn_hours": frozenset({5, 6, 7}),
    "rn_admin_hours": frozenset({5, 6, 7}),
    "rn_don_hours": frozenset({5, 6, 7}),
    "total_rn_hours": frozenset({5, 6, 7}),
    "lpn_hours": frozenset({8, 9}),
    "lpn_admin_hours": frozenset({8, 9}),
    "total_lpn_hours": frozenset({8, 9}),
    "cna_hours": frozenset({10, 11, 12}),
    "total_nurse_aide_hours": frozenset({10, 11, 12}),
}

_DEFAULT_DAILY_HEADCOUNT_METRICS: tuple[str, ...] = (
    "rn_hours",
    "rn_admin_hours",
    "rn_don_hours",
    "total_rn_hours",
    "lpn_hours",
    "lpn_admin_hours",
    "total_lpn_hours",
    "cna_hours",
    "total_nurse_aide_hours",
)


def _ein_metric_distinct_and_span_flags(
    job_sets_by_employee: dict[int, frozenset[int]],
    metric_key: str,
    primary_codes: frozenset[int],
) -> tuple[int, bool]:
    """Return (distinct employees touching primary_codes, span flag).

    If primary equals the metric family union, ``span`` is True when any contributing employee
    has two or more distinct job codes from that union on the same day (Employee Detail rows).

    If primary is a strict subset of the family (e.g. direct RN only), ``span`` is True when any
    contributor also has positive hours in another family role the same day (e.g. RN admin).
    """
    if not primary_codes:
        return 0, False
    family = _EIN_PBJ_METRIC_JOB_FAMILY.get(metric_key, primary_codes)
    if not family:
        family = primary_codes

    contrib_js = [js for _eid, js in job_sets_by_employee.items() if js & primary_codes]
    distinct = len(contrib_js)
    if distinct == 0:
        return 0, False

    if primary_codes == family:
        span = any(len(js & family) >= 2 for js in contrib_js)
    else:
        spill = family - primary_codes
        span = any(bool(js & spill) for js in contrib_js) if spill else False

    return distinct, span


def ein_daily_metric_headcounts_range(
    detail: pd.DataFrame,
    date_from_iso: str,
    date_to_iso: str,
    metric_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Batch distinct Employee Detail headcounts aligned to PBJ bridge metrics by work date.

    Positive ``WORK_HRS_NUM`` rows only. One pass over the extract for the date window; suitable
    for a single dashboard fetch per filtered daily range.

    Returns a JSON-serializable dict with ``by_date`` keys ``YYYY-MM-DD`` and per-metric
    ``{"distinct": int, "span_adjacent_roles": bool}``. ``span_adjacent_roles`` marks same-day
    multi-role context (see :func:`_ein_metric_distinct_and_span_flags`).
    """
    from facility_ein_lib import PBJ_METRIC_TO_EIN_JOB_CODES

    d0 = work_date_iso_to_yyyymmdd(date_from_iso)
    d1 = work_date_iso_to_yyyymmdd(date_to_iso)
    if d0 is None or d1 is None:
        return {
            "ok": False,
            "message": "Invalid from/to date (expected YYYY-MM-DD).",
            "by_date": {},
        }
    if d0 > d1:
        d0, d1 = d1, d0

    keys_in = tuple(metric_keys) if metric_keys is not None else _DEFAULT_DAILY_HEADCOUNT_METRICS
    keys: list[str] = []
    for k in keys_in:
        ks = str(k).strip()
        if ks and ks in PBJ_METRIC_TO_EIN_JOB_CODES and ks not in keys:
            keys.append(ks)

    d = prepare_ein_detail(detail)
    if d.empty or not keys:
        return {
            "ok": True,
            "from": workdate_to_iso(int(d0)) or date_from_iso,
            "to": workdate_to_iso(int(d1)) or date_to_iso,
            "metrics": keys,
            "by_date": {},
            "notes": (
                "distinct = unique SYS_EMPLEE_ID with any positive hours in the PBJ-mapped job "
                "codes for that metric; span_adjacent_roles flags same-day multi-role context "
                "within the RN, LPN, or aide family."
            ),
        }

    wd_series = pd.to_numeric(d["WorkDate"], errors="coerce")
    hrs = pd.to_numeric(d["WORK_HRS_NUM"], errors="coerce").fillna(0.0)
    pos = d.loc[(wd_series >= int(d0)) & (wd_series <= int(d1)) & (hrs > 0)].copy()
    if pos.empty:
        return {
            "ok": True,
            "from": workdate_to_iso(int(d0)) or date_from_iso,
            "to": workdate_to_iso(int(d1)) or date_to_iso,
            "metrics": keys,
            "by_date": {},
            "notes": "No Employee Detail rows in range with positive hours.",
        }

    pos["WorkDate_i"] = wd_series.loc[pos.index].astype(int)
    pos["SYS_EMPLEE_ID"] = pos["SYS_EMPLEE_ID"].astype(int)
    pos["EMPLEE_JOB_CD_ID"] = pos["EMPLEE_JOB_CD_ID"].astype(int)

    emp_jobs = pos.groupby(["WorkDate_i", "SYS_EMPLEE_ID"], sort=False)["EMPLEE_JOB_CD_ID"].agg(
        lambda ser: frozenset(int(x) for x in ser.dropna().astype(int).unique())
    )

    by_wd: dict[int, dict[int, frozenset[int]]] = defaultdict(dict)
    for (wd_i, eid), job_set in emp_jobs.items():
        by_wd[int(wd_i)][int(eid)] = job_set

    by_date_out: dict[str, dict[str, dict[str, Any]]] = {}
    for wd_i, emp_map in by_wd.items():
        iso = workdate_to_iso(wd_i)
        if not iso:
            continue
        day_payload: dict[str, dict[str, Any]] = {}
        for mk in keys:
            pc = frozenset(int(x) for x in PBJ_METRIC_TO_EIN_JOB_CODES.get(mk, ()))
            if not pc:
                continue
            distinct, span = _ein_metric_distinct_and_span_flags(emp_map, mk, pc)
            if distinct == 0 and not span:
                continue
            day_payload[mk] = {"distinct": int(distinct), "span_adjacent_roles": bool(span)}
        if day_payload:
            by_date_out[iso] = day_payload

    return {
        "ok": True,
        "from": workdate_to_iso(int(d0)) or date_from_iso,
        "to": workdate_to_iso(int(d1)) or date_to_iso,
        "metrics": keys,
        "by_date": by_date_out,
        "notes": (
            "distinct = unique SYS_EMPLEE_ID with positive hours in PBJ_METRIC_TO_EIN_JOB_CODES; "
            "span_adjacent_roles = same-day multi-role within the RN, LPN, or aide family "
            "(subset metrics) or multiple job codes in the family (total_* metrics)."
        ),
    }


def ein_headcount_quarter_series(
    detail: pd.DataFrame,
    *,
    min_sort_key: int | None = None,
    max_sort_key: int | None = None,
) -> list[dict[str, Any]]:
    """Quarter-level EIN headcount and contract staffing from row-level Employee Detail.

    For each ``CY_Qtr_norm`` present in the extract (positive ``WORK_HRS_NUM`` rows only):

    - ``distinct_*``: unique ``SYS_EMPLEE_ID`` with any positive hours in that quarter in the bucket.
      Job-code masks match ``ein_headcount_buckets_for_work_date`` (buckets can overlap).
    - ``mean_daily_distinct_total``: mean over distinct work dates in the quarter of the daily
      all-role distinct headcount (same spirit as the single-day modal).
    - ``distinct_contract_*``: distinct employees with at least one row in the quarter where
      ``EMP_CTR`` == 2 (CMS contract) and hours > 0, intersected with the same job-code masks.

    ``min_sort_key`` / ``max_sort_key`` filter by ``parse_ein_quarter_bound`` integer sort keys
    (inclusive) when provided.
    """
    d = prepare_ein_detail(detail)
    if d.empty:
        return []
    hrs = pd.to_numeric(d["WORK_HRS_NUM"], errors="coerce").fillna(0.0)
    pos = d.loc[hrs > 0].copy()
    if pos.empty:
        return []

    def _q_sort_key(qv: str) -> int:
        sk = parse_ein_quarter_bound(str(qv))
        return int(sk) if sk is not None else 0

    quarters = sorted(pos["CY_Qtr_norm"].dropna().astype(str).unique(), key=_q_sort_key)
    out: list[dict[str, Any]] = []

    for qn in quarters:
        sk = parse_ein_quarter_bound(str(qn))
        if sk is None:
            continue
        if min_sort_key is not None and int(sk) < int(min_sort_key):
            continue
        if max_sort_key is not None and int(sk) > int(max_sort_key):
            continue

        sub = pos.loc[pos["CY_Qtr_norm"].astype(str) == str(qn)].copy()
        if sub.empty:
            continue

        jc = sub["EMPLEE_JOB_CD_ID"].astype(int)
        eid = sub["SYS_EMPLEE_ID"].astype(int)

        def _uniq(mask: pd.Series) -> int:
            return int(eid.loc[mask].drop_duplicates().shape[0])

        nursing = jc.between(5, 12, inclusive="both")
        nurse_direct = jc.isin((7, 9, 10, 11, 12))
        nurse_lead = jc.isin((5, 6, 8))
        admin = jc.eq(1)
        non_nurse = ~(nursing | admin)

        ctr = pd.to_numeric(sub["EMP_CTR"], errors="coerce").fillna(0).eq(2)
        hpos = pd.to_numeric(sub["WORK_HRS_NUM"], errors="coerce").fillna(0.0) > 0
        ctr_rows = ctr & hpos

        distinct_total = int(eid.dropna().astype(int).nunique())
        distinct_contract_total = int(eid.loc[ctr_rows].dropna().astype(int).nunique())

        daily = sub.groupby("WorkDate", sort=False)["SYS_EMPLEE_ID"].nunique()
        mean_daily_distinct_total = round(float(daily.mean()), 2) if len(daily) else 0.0

        pct_contract = (
            round(100.0 * distinct_contract_total / distinct_total, 1) if distinct_total > 0 else None
        )

        out.append(
            {
                "quarter": str(qn),
                "quarter_sort_key": int(sk),
                "work_days_in_extract": int(len(daily)),
                "distinct_total": distinct_total,
                "distinct_admin": _uniq(admin),
                "distinct_nursing_total": _uniq(nursing),
                "distinct_nursing_direct": _uniq(nurse_direct),
                "distinct_nursing_leadership": _uniq(nurse_lead),
                "distinct_non_nurse": _uniq(non_nurse),
                "distinct_contract_total": distinct_contract_total,
                "distinct_contract_admin": _uniq(admin & ctr_rows),
                "distinct_contract_nursing": _uniq(nursing & ctr_rows),
                "distinct_contract_non_nurse": _uniq(non_nurse & ctr_rows),
                "contract_headcount_share_pct_total": pct_contract,
                "mean_daily_distinct_total": mean_daily_distinct_total,
                "notes": (
                    "Quarter distinct = any positive-hours row in quarter; buckets use EMPLEE_JOB_CD_ID. "
                    "Contract = EMP_CTR==2 with positive hours in quarter."
                ),
            }
        )
    return out


def _parse_iso_date_only(s: str | None) -> date | None:
    if not s:
        return None
    raw = str(s).strip()[:10]
    if len(raw) != 10:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _expand_headcount_display_window(
    start_iso: str,
    end_iso: str,
    *,
    grain: str,
    context: bool,
) -> tuple[str, str]:
    """Widen chart window around audit bounds so before/after context is visible."""
    start_d = _parse_iso_date_only(start_iso)
    end_d = _parse_iso_date_only(end_iso)
    if start_d is None or end_d is None:
        return start_iso, end_iso
    if start_d > end_d:
        start_d, end_d = end_d, start_d
    if not context:
        return start_d.isoformat(), end_d.isoformat()
    g = str(grain or "quarter").strip().lower()
    if g == "day":
        pad = timedelta(days=21)
    elif g == "month":
        pad = timedelta(days=62)
    elif g == "year":
        pad = timedelta(days=370)
    else:
        pad = timedelta(days=95)
    return (start_d - pad).isoformat(), (end_d + pad).isoformat()


def _period_key_date_bounds(pk: str, grain: str) -> tuple[date | None, date | None]:
    g = str(grain or "quarter").strip().lower()
    pk_s = str(pk or "").strip()
    if g == "day" and len(pk_s) >= 10:
        d = _parse_iso_date_only(pk_s)
        return (d, d) if d else (None, None)
    if g == "month" and len(pk_s) >= 7:
        try:
            d0 = datetime.strptime(pk_s[:7] + "-01", "%Y-%m-%d").date()
            if d0.month == 12:
                d1 = date(d0.year, 12, 31)
            else:
                d1 = date(d0.year, d0.month + 1, 1) - timedelta(days=1)
            return d0, d1
        except ValueError:
            return None, None
    if g == "year" and len(pk_s) >= 4 and pk_s[:4].isdigit():
        y = int(pk_s[:4])
        return date(y, 1, 1), date(y, 12, 31)
    sk = parse_ein_quarter_bound(pk_s)
    if sk is None:
        return None, None
    year = sk // 4
    q = (sk % 4) + 1
    start_month = (q - 1) * 3 + 1
    d0 = date(year, start_month, 1)
    if start_month == 10:
        d1 = date(year, 12, 31)
    else:
        d1 = date(year, start_month + 3, 1) - timedelta(days=1)
    return d0, d1


def _period_key_overlaps_iso_range(pk: str, grain: str, start_iso: str, end_iso: str) -> bool:
    lo = _parse_iso_date_only(start_iso)
    hi = _parse_iso_date_only(end_iso)
    if lo is None or hi is None:
        return False
    if lo > hi:
        lo, hi = hi, lo
    p_lo, p_hi = _period_key_date_bounds(pk, grain)
    if p_lo is None or p_hi is None:
        return False
    return p_lo <= hi and p_hi >= lo


EIN_CONTINUITY_LOW_HOUR_THRESHOLD = 16.0


def _ein_continuity_position_group(job_code: int) -> str:
    try:
        c = int(job_code)
    except (TypeError, ValueError):
        return "Unknown"
    if c in (5, 6, 7):
        return "RN"
    if c in (8, 9):
        return "LPN"
    if c in (10, 11, 12):
        return "CNA"
    if 5 <= c <= 12:
        return "Other nursing"
    if c == 1 or (2 <= c <= 4) or c >= 13:
        return "Non-nurse"
    return "Unknown"


def _ein_pct_change(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None:
        return None
    try:
        p = float(prior)
        c = float(current)
    except (TypeError, ValueError):
        return None
    if p == 0:
        return None
    return (c - p) / p


def _ein_in_pct_band(val: float | None, lo: float, hi: float) -> bool:
    if val is None:
        return False
    return float(lo) <= float(val) <= float(hi)


def _ein_quarter_pos_subset(pos_df: pd.DataFrame, quarter_key: str) -> pd.DataFrame:
    sub = pos_df.loc[pos_df["CY_Qtr_norm"].astype(str) == str(quarter_key)].copy()
    if sub.empty:
        return sub
    hrs = pd.to_numeric(sub["WORK_HRS_NUM"], errors="coerce").fillna(0.0)
    sub = sub.loc[hrs > 0].copy()
    sub["WORK_HRS_NUM"] = hrs.loc[sub.index]
    sub["SYS_EMPLEE_ID"] = pd.to_numeric(sub["SYS_EMPLEE_ID"], errors="coerce")
    sub["EMPLEE_JOB_CD_ID"] = pd.to_numeric(sub["EMPLEE_JOB_CD_ID"], errors="coerce")
    sub = sub.loc[sub["SYS_EMPLEE_ID"].notna() & sub["EMPLEE_JOB_CD_ID"].notna()].copy()
    if sub.empty:
        return sub
    sub["SYS_EMPLEE_ID"] = sub["SYS_EMPLEE_ID"].astype(int)
    sub["EMPLEE_JOB_CD_ID"] = sub["EMPLEE_JOB_CD_ID"].astype(int)
    return sub


def _ein_quarter_id_set_from_sub(sub: pd.DataFrame) -> set[int]:
    if sub.empty:
        return set()
    return set(sub["SYS_EMPLEE_ID"].astype(int).unique().tolist())


def _ein_quarter_employee_metrics(
    sub: pd.DataFrame,
    prior_ids: set[int],
) -> dict[str, Any]:
    if sub.empty:
        return {}
    total_hours = float(sub["WORK_HRS_NUM"].sum())
    unique_count = int(sub["SYS_EMPLEE_ID"].nunique())
    new_mask = ~sub["SYS_EMPLEE_ID"].isin(prior_ids)
    new_sub = sub.loc[new_mask]
    new_hours = float(new_sub["WORK_HRS_NUM"].sum()) if not new_sub.empty else 0.0
    emp_hours = sub.groupby("SYS_EMPLEE_ID", sort=False)["WORK_HRS_NUM"].sum()
    workdays = sub.groupby("SYS_EMPLEE_ID", sort=False)["WorkDate"].nunique()
    n_emp = len(emp_hours)
    new_emp_hours = (
        new_sub.groupby("SYS_EMPLEE_ID", sort=False)["WORK_HRS_NUM"].sum()
        if not new_sub.empty
        else pd.Series(dtype=float)
    )
    new_workdays = (
        new_sub.groupby("SYS_EMPLEE_ID", sort=False)["WorkDate"].nunique()
        if not new_sub.empty
        else pd.Series(dtype=float)
    )
    low_hour_share = (
        float((emp_hours < EIN_CONTINUITY_LOW_HOUR_THRESHOLD).sum()) / float(n_emp) if n_emp else None
    )
    return {
        "total_hours_current": total_hours,
        "unique_employee_ids": unique_count,
        "hours_per_unique_id": (total_hours / unique_count) if unique_count > 0 else None,
        "new_id_hours_share": (new_hours / total_hours) if total_hours > 0 else None,
        "median_hours_per_employee": float(emp_hours.median()) if n_emp else None,
        "median_workdays_per_employee": float(workdays.median()) if n_emp else None,
        "median_hours_for_new_ids": float(new_emp_hours.median()) if len(new_emp_hours) else None,
        "median_workdays_for_new_ids": float(new_workdays.median()) if len(new_workdays) else None,
        "one_day_employee_share": float((workdays == 1).sum()) / float(n_emp) if n_emp else None,
        "low_hour_employee_share": low_hour_share,
    }


def _ein_quarter_position_breakdown(
    sub: pd.DataFrame,
    prior_ids: set[int],
) -> dict[str, dict[str, Any]]:
    if sub.empty:
        return {}
    sub = sub.copy()
    sub["pos_group"] = sub["EMPLEE_JOB_CD_ID"].map(_ein_continuity_position_group)
    out: dict[str, dict[str, Any]] = {}
    for grp, gsub in sub.groupby("pos_group", sort=False):
        ids = set(gsub["SYS_EMPLEE_ID"].astype(int).unique().tolist())
        prior_grp_ids = prior_ids  # global prior; retained/new computed vs all prior IDs
        retained = ids & prior_grp_ids
        new_ids = ids - prior_grp_ids
        total_hours = float(gsub["WORK_HRS_NUM"].sum())
        new_hours = float(gsub.loc[~gsub["SYS_EMPLEE_ID"].isin(prior_grp_ids), "WORK_HRS_NUM"].sum())
        unique_n = len(ids)
        out[str(grp)] = {
            "unique_ids": unique_n,
            "total_hours": total_hours,
            "new_ids": len(new_ids),
            "new_id_share": (len(new_ids) / unique_n) if unique_n > 0 else None,
            "retained_ids": len(retained),
            "hours_per_unique_id": (total_hours / unique_n) if unique_n > 0 else None,
            "new_id_hours_share": (new_hours / total_hours) if total_hours > 0 else None,
        }
    return out


def _ein_position_hours_mix_similar(
    current: dict[str, dict[str, Any]],
    prior: dict[str, dict[str, Any]],
    *,
    tolerance: float = 0.10,
) -> bool:
    groups = set(current.keys()) | set(prior.keys())
    cur_total = sum(float(v.get("total_hours") or 0.0) for v in current.values())
    prior_total = sum(float(v.get("total_hours") or 0.0) for v in prior.values())
    if cur_total <= 0 or prior_total <= 0:
        return True
    for g in groups:
        cur_share = float(current.get(g, {}).get("total_hours") or 0.0) / cur_total
        prior_share = float(prior.get(g, {}).get("total_hours") or 0.0) / prior_total
        if abs(cur_share - prior_share) > tolerance:
            return False
    return True


def _ein_position_concentration_note(
    current_pos: dict[str, dict[str, Any]],
    prior_pos: dict[str, dict[str, Any]],
    *,
    total_new_ids: int,
) -> str | None:
    if total_new_ids <= 0:
        return None
    best_grp = None
    best_share = 0.0
    for grp, row in current_pos.items():
        new_n = int(row.get("new_ids") or 0)
        share = new_n / float(total_new_ids)
        if share > best_share:
            best_share = share
            best_grp = grp
    spike_grp = None
    for grp, row in current_pos.items():
        prior_u = int(prior_pos.get(grp, {}).get("unique_ids") or 0)
        cur_u = int(row.get("unique_ids") or 0)
        if prior_u <= 0:
            continue
        id_chg = _ein_pct_change(float(cur_u), float(prior_u))
        prior_h = float(prior_pos.get(grp, {}).get("total_hours") or 0.0)
        cur_h = float(row.get("total_hours") or 0.0)
        hrs_chg = _ein_pct_change(cur_h, prior_h) if prior_h > 0 else None
        if cur_u >= 1.75 * prior_u and _ein_in_pct_band(hrs_chg, -0.15, 0.15):
            spike_grp = grp
            break
    label_grp = None
    if best_share >= 0.50 and best_grp:
        label_grp = best_grp
    elif spike_grp:
        label_grp = spike_grp
    if not label_grp:
        return None
    display = {
        "RN": "RN",
        "LPN": "LPN",
        "CNA": "CNA / nurse aide",
        "Other nursing": "other nursing",
        "Non-nurse": "non-nurse",
        "Unknown": "unknown roles",
    }.get(label_grp, label_grp)
    return f"The discontinuity is concentrated among {display} IDs."


def _ein_format_id_share_pct(share: float | None) -> str:
    if share is None:
        return "—"
    return f"{float(share) * 100:.0f}%"


def _ein_classify_continuity_pattern(
    metrics: dict[str, Any],
    *,
    position_note: str | None,
) -> tuple[str, str, str]:
    """Return (pattern_key, collapsed_label, explanation)."""
    retention = float(metrics.get("retention_rate") or 0.0)
    new_share = float(metrics.get("new_id_share") or 0.0)
    dropped_share = float(metrics.get("dropped_id_share") or 0.0)
    prior_u = int(metrics.get("prior_unique_ids") or 0)
    cur_u = int(metrics.get("current_unique_ids") or 0)
    id_ratio = (cur_u / prior_u) if prior_u > 0 else 0.0
    hours_chg = metrics.get("total_hours_change_pct")
    hprd_chg = metrics.get("hprd_change_pct")
    hpi = metrics.get("hours_per_unique_id")
    hpi_prior = metrics.get("hours_per_unique_id_prior")
    new_hrs_share = metrics.get("new_id_hours_share")
    med_wd_new = metrics.get("median_workdays_for_new_ids")
    severity = str(metrics.get("severity") or "").lower()

    new_hours_meaningful = (
        (new_hrs_share is not None and float(new_hrs_share) >= 0.40)
        or (med_wd_new is not None and float(med_wd_new) >= 2.0)
    )

    workforce_replacement = (
        new_hrs_share is not None
        and float(new_hrs_share) >= 0.40
        and new_share >= 0.50
    )

    remapping = (
        id_ratio >= 2.0
        and retention < 0.40
        and _ein_in_pct_band(hours_chg, -0.15, 0.15)
        and (hprd_chg is None or _ein_in_pct_band(hprd_chg, -0.10, 0.10))
        and bool(metrics.get("position_hours_mix_similar", True))
    )
    fragmentation = (
        id_ratio >= 1.5
        and _ein_in_pct_band(hours_chg, -0.15, 0.15)
        and (hprd_chg is None or _ein_in_pct_band(hprd_chg, -0.15, 0.15))
        and hpi is not None
        and hpi_prior is not None
        and float(hpi) <= 0.75 * float(hpi_prior)
        and not workforce_replacement
    )
    hprd_cur = metrics.get("hprd_current")
    hprd_prior = metrics.get("hprd_prior")
    hrs_cur = metrics.get("total_hours_current")
    hrs_prior = metrics.get("total_hours_prior")
    expansion = id_ratio >= 1.5 and (
        (hrs_prior and hrs_cur and float(hrs_cur) >= 1.25 * float(hrs_prior))
        or (
            hprd_prior
            and hprd_cur
            and float(hprd_cur) >= 1.10 * float(hprd_prior)
        )
    )

    if remapping and not new_hours_meaningful:
        return (
            "id_remapping",
            "Possible ID remapping or reporting reset",
            "Staffing volume appears relatively stable, but reported employee identifiers changed sharply. "
            "This pattern may reflect employee-ID remapping, payroll/vendor changes, ownership or management transition, "
            "PBJ correction, or inconsistent reporting rather than actual workforce change.",
        )
    if workforce_replacement:
        return (
            "workforce_replacement",
            "Possible workforce replacement, agency/temp staffing, or ID remapping",
            f"New IDs accounted for {_ein_format_id_share_pct(new_share)} of current-quarter IDs "
            f"and {_ein_format_id_share_pct(new_hrs_share)} of current-quarter hours. "
            "Because newly reported IDs account for a meaningful share of hours, this may reflect workforce replacement, "
            "high churn, agency/temp staffing, ID remapping, payroll/vendor changes, or reporting inconsistency.",
        )
    if fragmentation:
        return (
            "roster_fragmentation",
            "Possible roster fragmentation",
            "Many more reported employee IDs appeared this quarter, but total staffing hours and HPRD were flat or "
            "changed modestly while hours per unique ID fell materially. "
            "This may indicate short-stint workers, agency/temp use, high churn, employee-ID remapping, or inconsistent reporting.",
        )
    if expansion:
        return (
            "staffing_expansion",
            "Possible staffing expansion",
            "The increase in reported employee IDs coincided with higher reported staffing hours or HPRD. "
            "This may reflect expanded staffing coverage, increased census, hiring, or temporary staffing rather than only data discontinuity.",
        )
    if severity == "orange":
        return (
            "generic",
            "Major employee ID discontinuity",
            "This quarter shows a major discontinuity in reported employee identifiers relative to the prior quarter. "
            "Treat this as a review trigger before interpreting headcount changes as actual workforce growth or turnover.",
        )
    return (
        "generic",
        "Employee ID continuity anomaly",
        "This quarter shows an unusual change in reported employee identifiers relative to the prior quarter. "
        "This may reflect workforce churn, temporary staffing, payroll/vendor changes, employee-ID remapping, "
        "ownership/management changes, or PBJ reporting artifacts.",
    )


def ein_quarter_pbj_context_from_daily(pbj_df: pd.DataFrame | None) -> dict[str, dict[str, float | None]]:
    """Optional pooled HPRD and resident-days by CY quarter from facility daily PBJ rows."""
    if pbj_df is None or pbj_df.empty or "CY_Qtr" not in pbj_df.columns:
        return {}
    out: dict[str, dict[str, float | None]] = {}
    nurse_hour_cols = [
        c
        for c in (
            "Hrs_RN",
            "Hrs_RNadmin",
            "Hrs_RNDON",
            "Hrs_LPN",
            "Hrs_LPNadmin",
            "Hrs_CNA",
            "Hrs_NAtrn",
            "Hrs_MedAide",
        )
        if c in pbj_df.columns
    ]
    for cy_qtr, grp in pbj_df.groupby("CY_Qtr", sort=False):
        cy = normalize_cy_qtr_ein(cy_qtr)
        if not cy:
            continue
        if "MDScensus" not in grp.columns:
            continue
        census = pd.to_numeric(grp["MDScensus"], errors="coerce")
        ok = census.notna() & (census > 0)
        resident_days = float(census.loc[ok].sum()) if bool(ok.any()) else None
        if resident_days is None or resident_days <= 0:
            continue
        sub = grp.loc[ok]
        total_hours = 0.0
        for col in nurse_hour_cols:
            total_hours += float(pd.to_numeric(sub[col], errors="coerce").fillna(0.0).sum())
        hprd = (total_hours / resident_days) if resident_days > 0 else None
        out[str(cy)] = {
            "hprd": round(float(hprd), 4) if hprd is not None else None,
            "resident_days": round(float(resident_days), 2),
            "total_nurse_hours_pbj": round(float(total_hours), 2),
        }
    return out


def ein_employee_id_continuity_flag(
    current_ids: set[int],
    prior_ids: set[int],
    *,
    quarter_metrics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Evaluate quarter-over-quarter SYS_EMPLEE_ID continuity trigger + pattern classification."""
    prior_count = len(prior_ids)
    current_count = len(current_ids)
    if prior_count <= 0 or current_count <= 0:
        return None

    retained_ids = current_ids & prior_ids
    new_ids = current_ids - prior_ids
    dropped_ids = prior_ids - current_ids
    retention_rate = float(len(retained_ids)) / float(prior_count)
    new_id_share = float(len(new_ids)) / float(current_count)
    dropped_id_share = float(len(dropped_ids)) / float(prior_count)

    orange = current_count >= 2.0 * prior_count and new_id_share >= 0.60
    yellow = (
        current_count >= 1.5 * prior_count
        or new_id_share >= 0.50
        or retention_rate < 0.60
    )
    if not orange and not yellow:
        return None

    metrics: dict[str, Any] = {
        "severity": "orange" if orange else "yellow",
        "prior_unique_ids": prior_count,
        "current_unique_ids": current_count,
        "retained_ids": len(retained_ids),
        "new_ids": len(new_ids),
        "dropped_ids": len(dropped_ids),
        "retention_rate": retention_rate,
        "new_id_share": new_id_share,
        "dropped_id_share": dropped_id_share,
    }
    if quarter_metrics:
        metrics.update(quarter_metrics)

    position_note = metrics.pop("position_note", None)
    pattern_key, label, explanation = _ein_classify_continuity_pattern(metrics, position_note=position_note)
    metrics["pattern"] = pattern_key
    metrics["label"] = label
    metrics["explanation"] = explanation
    if position_note:
        metrics["position_note"] = position_note
    return metrics


def _ein_quarter_employee_id_invalid_share(pos_df: pd.DataFrame, quarter_key: str) -> float | None:
    """Share of rows in quarter with missing/invalid SYS_EMPLEE_ID."""
    sub = pos_df.loc[pos_df["CY_Qtr_norm"].astype(str) == str(quarter_key)]
    if sub.empty:
        return None
    ids = pd.to_numeric(sub["SYS_EMPLEE_ID"], errors="coerce")
    return float(ids.isna().sum()) / float(len(sub))


def _ein_quarter_employee_id_sets(pos_df: pd.DataFrame, quarter_key: str) -> set[int]:
    sub = _ein_quarter_pos_subset(pos_df, quarter_key)
    return _ein_quarter_id_set_from_sub(sub)


def ein_employee_id_continuity_by_period(
    pos: pd.DataFrame,
    period_keys: Sequence[str],
    grain: str,
    *,
    quarter_pbj_context: dict[str, dict[str, float | None]] | None = None,
) -> dict[str, dict[str, Any] | None]:
    """Quarter-grain SYS_EMPLEE_ID continuity flags keyed by period_key."""
    g = str(grain or "").strip().lower()
    if g != "quarter" or pos is None or pos.empty:
        return {}

    ctx = quarter_pbj_context or {}
    all_quarters = _ein_all_quarter_keys_sorted(pos)
    if not all_quarters:
        return {}

    q_index = {str(q): i for i, q in enumerate(all_quarters)}
    out: dict[str, dict[str, Any] | None] = {}

    for pk_raw in period_keys:
        pk = str(pk_raw)
        idx = q_index.get(pk)
        if idx is None or idx <= 0:
            out[pk] = None
            continue

        prior_pk = all_quarters[idx - 1]
        invalid_current = _ein_quarter_employee_id_invalid_share(pos, pk)
        invalid_prior = _ein_quarter_employee_id_invalid_share(pos, prior_pk)
        if (
            invalid_current is None
            or invalid_prior is None
            or invalid_current > 0.20
            or invalid_prior > 0.20
        ):
            out[pk] = None
            continue

        cur_sub = _ein_quarter_pos_subset(pos, pk)
        prior_sub = _ein_quarter_pos_subset(pos, prior_pk)
        current_ids = _ein_quarter_id_set_from_sub(cur_sub)
        prior_ids = _ein_quarter_id_set_from_sub(prior_sub)
        if not current_ids or not prior_ids:
            out[pk] = None
            continue

        cur_metrics = _ein_quarter_employee_metrics(cur_sub, prior_ids)
        prior_metrics = _ein_quarter_employee_metrics(prior_sub, set())
        cur_pos = _ein_quarter_position_breakdown(cur_sub, prior_ids)
        prior_pos = _ein_quarter_position_breakdown(prior_sub, set())
        position_note = _ein_position_concentration_note(
            cur_pos, prior_pos, total_new_ids=len(current_ids - prior_ids)
        )

        pbj_cur = ctx.get(str(pk)) or {}
        pbj_prior = ctx.get(str(prior_pk)) or {}
        quarter_metrics: dict[str, Any] = dict(cur_metrics)
        quarter_metrics["total_hours_prior"] = prior_metrics.get("total_hours_current")
        quarter_metrics["hours_per_unique_id_prior"] = prior_metrics.get("hours_per_unique_id")
        quarter_metrics["total_hours_change_pct"] = _ein_pct_change(
            quarter_metrics.get("total_hours_current"),
            quarter_metrics.get("total_hours_prior"),
        )
        quarter_metrics["hours_per_unique_id_change_pct"] = _ein_pct_change(
            quarter_metrics.get("hours_per_unique_id"),
            quarter_metrics.get("hours_per_unique_id_prior"),
        )
        quarter_metrics["hprd_current"] = pbj_cur.get("hprd")
        quarter_metrics["hprd_prior"] = pbj_prior.get("hprd")
        quarter_metrics["hprd_change_pct"] = _ein_pct_change(
            quarter_metrics.get("hprd_current"), quarter_metrics.get("hprd_prior")
        )
        quarter_metrics["resident_days_current"] = pbj_cur.get("resident_days")
        quarter_metrics["resident_days_prior"] = pbj_prior.get("resident_days")
        quarter_metrics["resident_days_change_pct"] = _ein_pct_change(
            quarter_metrics.get("resident_days_current"),
            quarter_metrics.get("resident_days_prior"),
        )
        quarter_metrics["position_hours_mix_similar"] = _ein_position_hours_mix_similar(cur_pos, prior_pos)
        if position_note:
            quarter_metrics["position_note"] = position_note

        out[pk] = ein_employee_id_continuity_flag(
            current_ids, prior_ids, quarter_metrics=quarter_metrics
        )

    return out


def _ein_all_quarter_keys_sorted(pos_df: pd.DataFrame) -> list[str]:
    keys = pos_df["CY_Qtr_norm"].dropna().astype(str).unique().tolist()

    def _sort_key(pk: str) -> tuple:
        sk = parse_ein_quarter_bound(pk)
        return (int(sk) if sk is not None else 0, pk)

    return sorted(keys, key=_sort_key)


# --- Sustained reported work pattern review triggers (PBJ Employee Detail) ---

SUSTAINED_WORK_ROLE_GROUP_CODES: dict[str, frozenset[int]] = {
    "CNA": frozenset({10}),
    "Nurse aide trainee": frozenset({11}),
    "Medication aide": frozenset({12}),
    "Aide group": frozenset({10, 11, 12}),
    "RN": frozenset({5, 6, 7}),
    "LPN": frozenset({8, 9}),
    "All nursing": frozenset(range(5, 13)),
}

SUSTAINED_WORK_PRIMARY_ROLE_GROUPS: frozenset[str] = frozenset({"CNA", "Aide group"})

SUSTAINED_WORK_SEVERITY_ORDER: dict[str, int] = {
    "advisory": 1,
    "review": 2,
    "major": 3,
    "extreme": 4,
}

SUSTAINED_WORK_REVIEW_LEVELS: frozenset[str] = frozenset({"review", "major", "extreme"})


def _mask_sys_employee_id(eid: int) -> str:
    s = str(abs(int(eid)))
    return f"…{s[-4:]}" if len(s) >= 4 else "…****"


def _sustained_work_severity_max(*levels: str | None) -> str | None:
    best: str | None = None
    best_rank = 0
    for lv in levels:
        if not lv:
            continue
        rank = SUSTAINED_WORK_SEVERITY_ORDER.get(str(lv).lower(), 0)
        if rank > best_rank:
            best_rank = rank
            best = str(lv).lower()
    return best


def _yyyymmdd_int_to_date(wd: int) -> date:
    s = str(int(wd)).zfill(8)
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def _max_consecutive_calendar_dates(sorted_wds: list[int]) -> tuple[int, int | None, int | None]:
    """Longest run of consecutive calendar WorkDate ints."""
    if not sorted_wds:
        return 0, None, None
    uniq = sorted({int(w) for w in sorted_wds})
    best_len = 1
    best_start = uniq[0]
    best_end = uniq[0]
    cur_len = 1
    cur_start = uniq[0]
    prev_d = _yyyymmdd_int_to_date(uniq[0])
    prev_wd = uniq[0]
    for wd in uniq[1:]:
        d = _yyyymmdd_int_to_date(wd)
        if (d - prev_d).days == 1:
            cur_len += 1
        else:
            if cur_len > best_len:
                best_len = cur_len
                best_start = cur_start
                best_end = prev_wd
            cur_len = 1
            cur_start = wd
        prev_d = d
        prev_wd = wd
    if cur_len > best_len:
        best_len = cur_len
        best_start = cur_start
        best_end = prev_wd
    return best_len, best_start, best_end


def _max_rolling_90_worked_days_hours(
    day_hours: list[tuple[int, float]],
) -> tuple[int, float]:
    """Max worked days and hours in any inclusive 90-calendar-day window."""
    if not day_hours:
        return 0, 0.0
    rows = sorted(((int(w), float(h)) for w, h in day_hours if int(w) > 0), key=lambda x: x[0])
    if not rows:
        return 0, 0.0
    best_days = 0
    best_hours = 0.0
    for i, (end_wd, _) in enumerate(rows):
        end_d = _yyyymmdd_int_to_date(end_wd)
        start_d = end_d - timedelta(days=89)
        worked = 0
        hrs = 0.0
        for j in range(i, -1, -1):
            wd_j, h_j = rows[j]
            if _yyyymmdd_int_to_date(wd_j) < start_d:
                break
            if h_j > 0:
                worked += 1
            hrs += h_j
        if worked > best_days or (worked == best_days and hrs > best_hours):
            best_days = worked
            best_hours = hrs
    return best_days, round(best_hours, 2)


def _eval_shift_equivalent_streak_severity(max_days: int) -> str | None:
    if max_days >= 75:
        return "extreme"
    if max_days >= 45:
        return "major"
    if max_days >= 30:
        return "review"
    if max_days >= 21:
        return "advisory"
    return None


def _eval_full_shift_streak_severity(max_days: int) -> str | None:
    if max_days >= 30:
        return "major"
    if max_days >= 21:
        return "review"
    if max_days >= 14:
        return "advisory"
    return None


def _eval_rolling_90_severity(worked_days: int, hours: float) -> str | None:
    if hours >= 900:
        return "extreme"
    if worked_days >= 90 or hours >= 720:
        return "major"
    if worked_days >= 84 and hours >= 624:
        return "review"
    if worked_days >= 78 and hours >= 520:
        return "advisory"
    return None


def _eval_daily_hours_severity(
    max_daily: float,
    days_over_16: int,
    days_over_24: int,
) -> str | None:
    if days_over_24 >= 2:
        return "extreme"
    if max_daily > 24:
        return "major"
    if days_over_16 >= 3:
        return "review"
    if max_daily > 16:
        return "advisory"
    return None


def _classify_sustained_work_employee_type(ctr_flags: set[int]) -> str:
    has_emp = 1 in ctr_flags
    has_ctr = 2 in ctr_flags
    if has_emp and has_ctr:
        return "mixed"
    if has_ctr:
        return "contract"
    if has_emp:
        return "employee"
    return "unknown"


def _consolidate_sustained_work_day_rows(detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Build employee-day-role-group rows and employee-day totals from prepared EIN detail.

    Returns ``(role_day_df, employee_day_totals_df, limitations)``.
    """
    limitations: list[str] = []
    d = prepare_ein_detail(detail)
    if d.empty:
        limitations.append("Employee Detail extract is empty or missing required columns.")
        return pd.DataFrame(), pd.DataFrame(), limitations

    if "EMP_CTR" not in d.columns and "EMPLEE_CTR" in d.columns:
        d = d.copy()
        d["EMP_CTR"] = pd.to_numeric(d["EMPLEE_CTR"], errors="coerce")
    d["EMP_CTR"] = pd.to_numeric(d.get("EMP_CTR"), errors="coerce").fillna(0).astype(int)
    d = d[pd.to_numeric(d["EMPLEE_JOB_CD_ID"], errors="coerce").between(5, 12, inclusive="both")].copy()
    if d.empty:
        limitations.append("No nursing job codes (5–12) in Employee Detail extract.")
        return pd.DataFrame(), pd.DataFrame(), limitations

    has_fn = "WORK_HRS_FN" in d.columns
    if not has_fn:
        limitations.append(
            "WORK_HRS_FN was not included in the Employee Detail extract; fractional-hour checks are skipped."
        )

    emp_day_parts: list[pd.DataFrame] = []
    for rg_name, codes in SUSTAINED_WORK_ROLE_GROUP_CODES.items():
        sub = d[d["EMPLEE_JOB_CD_ID"].astype(int).isin(list(codes))].copy()
        if sub.empty:
            continue
        grp_cols = ["SYS_EMPLEE_ID", "WorkDate", "CY_Qtr_norm", "EMP_CTR"]
        agg: dict[str, Any] = {"employee_day_role_hours": ("WORK_HRS_NUM", "sum")}
        if has_fn:
            agg["any_work_hrs_fn"] = ("WORK_HRS_FN", lambda s: bool(s.notna().any()))
        g = sub.groupby(grp_cols, as_index=False).agg(**agg)
        g["role_group"] = rg_name
        emp_day_parts.append(g)

    if not emp_day_parts:
        limitations.append("Could not consolidate employee-day role-group hours.")
        return pd.DataFrame(), pd.DataFrame(), limitations

    role_day = pd.concat(emp_day_parts, ignore_index=True)
    role_day["employee_day_role_hours"] = pd.to_numeric(
        role_day["employee_day_role_hours"], errors="coerce"
    ).fillna(0.0)

    agg2: dict[str, Any] = {
        "employee_day_role_hours": ("employee_day_role_hours", "sum"),
        "emp_ctrs": ("EMP_CTR", lambda s: set(int(x) for x in s.tolist())),
    }
    if has_fn:
        agg2["any_work_hrs_fn"] = ("any_work_hrs_fn", "any")
    role_day_agg = role_day.groupby(
        ["SYS_EMPLEE_ID", "WorkDate", "CY_Qtr_norm", "role_group"], as_index=False
    ).agg(**agg2)
    role_day_agg["shift_equivalent_day"] = role_day_agg["employee_day_role_hours"] >= 4.0
    role_day_agg["full_shift_day"] = role_day_agg["employee_day_role_hours"] >= 7.0
    role_day_agg["any_worked_day"] = role_day_agg["employee_day_role_hours"] > 0.0

    emp_day_total = (
        d.groupby(["SYS_EMPLEE_ID", "WorkDate", "CY_Qtr_norm"], as_index=False)
        .agg(
            employee_day_total_hours=("WORK_HRS_NUM", "sum"),
            emp_ctrs=("EMP_CTR", lambda s: set(int(x) for x in s.tolist())),
            job_codes=("EMPLEE_JOB_CD_ID", lambda s: set(int(x) for x in s.tolist())),
        )
    )
    return role_day_agg, emp_day_total, limitations


def _sustained_work_label_for_flag(
    severity: str,
    role_group: str,
    *,
    max_consecutive_shift_equivalent_days: int,
    daily_hours_trigger: bool,
) -> str:
    if severity == "extreme":
        return "Extreme reported work pattern"
    if max_consecutive_shift_equivalent_days >= 21:
        return "Consecutive-day employee-ID anomaly"
    if role_group == "CNA":
        return "Sustained CNA work pattern"
    return f"Sustained {role_group} work pattern"


SUSTAINED_WORK_ROLLING_WINDOW_DAYS = 90
SUSTAINED_WORK_NARRATIVE_MAX_DAY_MIN = 14.0
SUSTAINED_WORK_NARRATIVE_STREAK_4H_MIN = 21
SUSTAINED_WORK_NARRATIVE_STREAK_7H_MIN = 14
SUSTAINED_WORK_NARRATIVE_SHARE_MIN = 0.02


def _format_sustained_work_metric(value: float | int | None, *, decimals: int = 0) -> str:
    if value is None:
        return "0"
    v = float(value)
    if decimals <= 0:
        return str(int(round(v)))
    text = f"{v:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def build_sustained_work_flag_narrative(
    *,
    role_group: str,
    rolling_90_hours: float | int,
    rolling_90_worked_days: int,
    max_consecutive_shift_equivalent_days: int = 0,
    max_consecutive_full_shift_days: int = 0,
    max_daily_hours: float = 0.0,
    share_of_role_hours: float = 0.0,
    distinct_role_employees_in_period: int = 0,
    appears_across_role_groups: bool = False,
    cross_role_groups: Sequence[str] | None = None,
    appears_across_employee_contract: bool = False,
    window_days: int = SUSTAINED_WORK_ROLLING_WINDOW_DAYS,
) -> str:
    rg = str(role_group or "role group").strip() or "role group"
    hours = float(rolling_90_hours or 0)
    worked_days = int(rolling_90_worked_days or 0)
    wd = int(window_days or SUSTAINED_WORK_ROLLING_WINDOW_DAYS)

    parts: list[str] = [
        (
            f"This reported employee ID accounts for an unusually high volume of {rg} hours: "
            f"{_format_sustained_work_metric(hours)} hours across {worked_days} worked days "
            f"in a {wd}-day window."
        )
    ]

    max_daily = float(max_daily_hours or 0)
    if max_daily >= SUSTAINED_WORK_NARRATIVE_MAX_DAY_MIN:
        parts.append(
            f"The maximum reported day was {_format_sustained_work_metric(max_daily, decimals=1)} hours."
        )

    streak_4 = int(max_consecutive_shift_equivalent_days or 0)
    streak_7 = int(max_consecutive_full_shift_days or 0)
    has_4 = streak_4 >= SUSTAINED_WORK_NARRATIVE_STREAK_4H_MIN
    has_7 = streak_7 >= SUSTAINED_WORK_NARRATIVE_STREAK_7H_MIN
    if has_4 and has_7:
        parts.append(
            f"It also shows sustained multi-day work activity, including {streak_4} consecutive days "
            f"with at least 4 reported hours and {streak_7} consecutive days with at least 7 reported hours."
        )
    elif has_4:
        parts.append(
            f"It also shows sustained multi-day work activity, including {streak_4} consecutive days "
            f"with at least 4 reported hours."
        )
    elif has_7:
        parts.append(
            f"It also shows sustained multi-day work activity, including {streak_7} consecutive days "
            f"with at least 7 reported hours."
        )

    share = float(share_of_role_hours or 0)
    if share >= SUSTAINED_WORK_NARRATIVE_SHARE_MIN:
        share_pct = _format_sustained_work_metric(share * 100.0, decimals=1)
        share_line = f"This ID accounts for {share_pct}% of {rg} hours in the quarter"
        distinct_n = int(distinct_role_employees_in_period or 0)
        if distinct_n > 0:
            role_noun = rg if distinct_n == 1 else f"{rg}s"
            share_line += f" (among {distinct_n} {role_noun} with reported hours)"
        parts.append(share_line + ".")

    if appears_across_role_groups:
        parts.append(_cross_role_narrative_sentence(cross_role_groups))
    if appears_across_employee_contract:
        parts.append("The same reported ID also appears under both employee and contract reporting.")

    parts.append(
        "PBJ data alone cannot determine whether this reflects actual scheduling practices, reporting conventions, "
        "identifier assignment methods, or data quality issues. Review source rows before interpreting this as an "
        "individual worker's schedule."
    )
    return " ".join(parts)


def _timing_vs_id_continuity(flag_quarter: str, continuity_quarters: set[str]) -> tuple[bool, str]:
    if not continuity_quarters:
        return False, "unrelated"
    fq_sk = parse_ein_quarter_bound(flag_quarter)
    if fq_sk is None:
        return False, "unrelated"
    best: tuple[bool, str] | None = None
    for cq in continuity_quarters:
        csk = parse_ein_quarter_bound(cq)
        if csk is None:
            continue
        if csk == fq_sk:
            return True, "during"
        dist = abs(int(csk) - int(fq_sk))
        if dist == 1:
            timing = "before" if fq_sk < csk else "after"
            if best is None or dist < 2:
                best = (True, timing)
    if best:
        return best
    return False, "unrelated"


def _continuity_flag_quarters(
    employee_id_continuity_by_period: dict[str, dict[str, Any] | None],
) -> set[str]:
    out: set[str] = set()
    for pk, flag in (employee_id_continuity_by_period or {}).items():
        if flag and flag.get("severity"):
            qn = normalize_cy_qtr_ein(pk)
            if qn:
                out.add(qn)
    return out


def _employee_sustained_work_flags_for_quarter(
    *,
    quarter: str,
    role_day: pd.DataFrame,
    emp_day_total: pd.DataFrame,
    facility_ccn: str,
    continuity_quarters: set[str],
    limitations: list[str],
) -> list[dict[str, Any]]:
    qn = normalize_cy_qtr_ein(quarter)
    if not qn:
        return []

    q_role = role_day.loc[role_day["CY_Qtr_norm"].astype(str) == str(qn)].copy()
    q_total = emp_day_total.loc[emp_day_total["CY_Qtr_norm"].astype(str) == str(qn)].copy()
    if q_role.empty:
        return []

    cna_hours_by_eid: dict[int, float] = {}
    cna_sub = q_role.loc[q_role["role_group"] == "CNA"]
    if not cna_sub.empty:
        cna_hours_by_eid = (
            cna_sub.groupby("SYS_EMPLEE_ID")["employee_day_role_hours"].sum().astype(float).to_dict()
        )
    total_cna_hours = float(sum(cna_hours_by_eid.values())) if cna_hours_by_eid else 0.0

    role_hours_totals: dict[str, float] = (
        q_role.groupby("role_group")["employee_day_role_hours"].sum().astype(float).to_dict()
    )
    distinct_role_employees: dict[str, int] = (
        q_role.loc[q_role["employee_day_role_hours"] > 0]
        .groupby("role_group")["SYS_EMPLEE_ID"]
        .nunique()
        .astype(int)
        .to_dict()
    )

    flags: list[dict[str, Any]] = []
    role_groups = sorted(q_role["role_group"].dropna().astype(str).unique().tolist())
    for rg in role_groups:
        rg_sub = q_role.loc[q_role["role_group"] == rg]
        if rg_sub.empty:
            continue
        for eid_raw, emp_sub in rg_sub.groupby("SYS_EMPLEE_ID", sort=False):
            try:
                eid = int(eid_raw)
            except (TypeError, ValueError):
                continue

            hist_sub = role_day.loc[
                (role_day["SYS_EMPLEE_ID"] == eid) & (role_day["role_group"] == rg)
            ].copy()
            if hist_sub.empty:
                continue

            shift_days = hist_sub.loc[hist_sub["shift_equivalent_day"], "WorkDate"].astype(int).tolist()
            full_days = hist_sub.loc[hist_sub["full_shift_day"], "WorkDate"].astype(int).tolist()
            max_shift, shift_lo, shift_hi = _max_consecutive_calendar_dates(shift_days)
            max_full, full_lo, full_hi = _max_consecutive_calendar_dates(full_days)

            day_hours = [
                (int(r.WorkDate), float(r.employee_day_role_hours))
                for r in hist_sub.itertuples(index=False)
                if float(r.employee_day_role_hours) > 0
            ]
            roll_days, roll_hours = _max_rolling_90_worked_days_hours(day_hours)

            emp_total_sub = emp_day_total.loc[emp_day_total["SYS_EMPLEE_ID"] == eid].copy()
            emp_q_total = emp_total_sub.loc[emp_total_sub["CY_Qtr_norm"].astype(str) == str(qn)].copy()
            max_daily = 0.0
            days_over_16 = 0
            days_over_24 = 0
            if not emp_q_total.empty:
                hrs_series = pd.to_numeric(emp_q_total["employee_day_total_hours"], errors="coerce").fillna(0.0)
                max_daily = float(hrs_series.max()) if len(hrs_series) else 0.0
                days_over_16 = int((hrs_series > 16.0).sum())
                days_over_24 = int((hrs_series > 24.0).sum())

            sev = _sustained_work_severity_max(
                _eval_shift_equivalent_streak_severity(max_shift),
                _eval_full_shift_streak_severity(max_full),
                _eval_rolling_90_severity(roll_days, roll_hours),
                _eval_daily_hours_severity(max_daily, days_over_16, days_over_24),
            )
            if not sev:
                continue
            if rg not in SUSTAINED_WORK_PRIMARY_ROLE_GROUPS and sev == "advisory":
                continue

            period_end_q = (
                normalize_cy_qtr_ein(cy_quarter_from_yyyymmdd(int(shift_hi)))
                if shift_hi
                else (
                    normalize_cy_qtr_ein(cy_quarter_from_yyyymmdd(int(full_hi)))
                    if full_hi
                    else qn
                )
            )
            streak_triggers = _eval_shift_equivalent_streak_severity(max_shift) or _eval_full_shift_streak_severity(
                max_full
            )
            if streak_triggers and period_end_q != qn:
                continue
            if (
                not streak_triggers
                and _eval_rolling_90_severity(roll_days, roll_hours) is None
                and _eval_daily_hours_severity(max_daily, days_over_16, days_over_24) is None
            ):
                continue

            emp_ctrs: set[int] = set()
            for s in emp_sub.get("emp_ctrs", pd.Series(dtype=object)):
                if isinstance(s, set):
                    emp_ctrs |= s
            employee_type = _classify_sustained_work_employee_type(emp_ctrs)

            multi_role_same_day = 0
            cross_role_groups: list[str] = []
            emp_ctr_same_day = 0
            if not emp_q_total.empty:
                multi_role_same_day, cross_role_groups = _sustained_work_multi_role_same_day_metrics(
                    emp_q_total
                )
                for row in emp_q_total.itertuples(index=False):
                    ctrs = row.emp_ctrs if isinstance(row.emp_ctrs, set) else set()
                    if 1 in ctrs and 2 in ctrs:
                        emp_ctr_same_day += 1

            rg_hours = float(emp_sub["employee_day_role_hours"].sum())
            rg_total = float(role_hours_totals.get(rg, 0.0) or 0.0)
            share = round(rg_hours / rg_total, 4) if rg_total > 0 else 0.0

            worked_days_q = int((emp_sub["employee_day_role_hours"] > 0).sum())
            avg_hpd = round(rg_hours / worked_days_q, 2) if worked_days_q else 0.0

            period_start = workdate_to_iso(shift_lo or full_lo or (int(emp_sub["WorkDate"].min()) if len(emp_sub) else 0))
            period_end = workdate_to_iso(shift_hi or full_hi or (int(emp_sub["WorkDate"].max()) if len(emp_sub) else 0))

            near, timing = _timing_vs_id_continuity(qn, continuity_quarters)

            appears_across_role_groups = multi_role_same_day > 0 and len(cross_role_groups) >= 2
            appears_across_employee_contract = emp_ctr_same_day > 0

            limitations_out = list(limitations)
            if multi_role_same_day > 0 and float(emp_q_total["employee_day_total_hours"].max() if not emp_q_total.empty else 0) > 16:
                limitations_out.append(
                    "Multi-role same-day totals exceed 16 hours on at least one day; may reflect role splitting or duplicate reporting."
                )

            narrative = build_sustained_work_flag_narrative(
                role_group=rg,
                rolling_90_hours=roll_hours,
                rolling_90_worked_days=roll_days,
                max_consecutive_shift_equivalent_days=max_shift,
                max_consecutive_full_shift_days=max_full,
                max_daily_hours=max_daily,
                share_of_role_hours=share,
                distinct_role_employees_in_period=int(distinct_role_employees.get(rg, 0) or 0),
                appears_across_role_groups=appears_across_role_groups,
                cross_role_groups=cross_role_groups if appears_across_role_groups else [],
                appears_across_employee_contract=appears_across_employee_contract,
            )

            flags.append(
                {
                    "severity": sev,
                    "label": _sustained_work_label_for_flag(
                        sev,
                        rg,
                        max_consecutive_shift_equivalent_days=max_shift,
                        daily_hours_trigger=days_over_16 > 0 or days_over_24 > 0,
                    ),
                    "facility_ccn": str(facility_ccn),
                    "quarter": qn,
                    "employee_id": str(eid),
                    "masked_employee_id": _mask_sys_employee_id(eid),
                    "role_group": rg,
                    "employee_type": employee_type,
                    "period_start": period_start,
                    "period_end": period_end,
                    "max_consecutive_shift_equivalent_days": int(max_shift),
                    "max_consecutive_full_shift_days": int(max_full),
                    "rolling_90_worked_days": int(roll_days),
                    "rolling_90_hours": float(roll_hours),
                    "avg_hours_per_worked_day": avg_hpd,
                    "max_daily_hours": round(max_daily, 2),
                    "days_over_16_hours": int(days_over_16),
                    "days_over_24_hours": int(days_over_24),
                    "share_of_role_hours": share,
                    "distinct_role_employees_in_period": int(distinct_role_employees.get(rg, 0) or 0),
                    "multi_role_same_day_count": int(multi_role_same_day),
                    "cross_role_groups": list(cross_role_groups) if appears_across_role_groups else [],
                    "employee_and_contract_same_day_count": int(emp_ctr_same_day),
                    "near_employee_id_continuity_anomaly": bool(near),
                    "timing_vs_id_anomaly": timing,
                    "window_days": SUSTAINED_WORK_ROLLING_WINDOW_DAYS,
                    "appears_across_role_groups": appears_across_role_groups,
                    "appears_across_employee_contract": appears_across_employee_contract,
                    "narrative": narrative,
                    "interpretation": narrative,
                    "source_review": {
                        "pbj_employee_detail": [
                            {
                                "quarter": qn,
                                "filter_provnum": str(facility_ccn),
                            }
                        ]
                    },
                    "limitations": limitations_out,
                    "_cna_hours": float(cna_hours_by_eid.get(eid, 0.0)),
                }
            )

    return flags


def _facility_sustained_work_summary_flags(
    employee_flags: list[dict[str, Any]],
    *,
    quarter: str,
    facility_ccn: str,
    limitations: list[str],
) -> list[dict[str, Any]]:
    qn = normalize_cy_qtr_ein(quarter)
    if not qn:
        return []

    cna_flags = [
        f
        for f in employee_flags
        if f.get("quarter") == qn
        and f.get("role_group") == "CNA"
        and str(f.get("severity") or "") in SUSTAINED_WORK_REVIEW_LEVELS
    ]
    out: list[dict[str, Any]] = []
    if len(cna_flags) >= 3:
        out.append(
            {
                "severity": "review",
                "label": "Multiple sustained CNA work patterns",
                "facility_ccn": str(facility_ccn),
                "quarter": qn,
                "employee_id": None,
                "masked_employee_id": None,
                "role_group": "CNA",
                "employee_type": "unknown",
                "period_start": None,
                "period_end": None,
                "max_consecutive_shift_equivalent_days": max(
                    int(f.get("max_consecutive_shift_equivalent_days") or 0) for f in cna_flags
                ),
                "max_consecutive_full_shift_days": max(
                    int(f.get("max_consecutive_full_shift_days") or 0) for f in cna_flags
                ),
                "rolling_90_worked_days": 0,
                "rolling_90_hours": 0.0,
                "avg_hours_per_worked_day": 0.0,
                "max_daily_hours": 0.0,
                "days_over_16_hours": 0,
                "days_over_24_hours": 0,
                "share_of_role_hours": 0.0,
                "multi_role_same_day_count": 0,
                "employee_and_contract_same_day_count": 0,
                "near_employee_id_continuity_anomaly": any(f.get("near_employee_id_continuity_anomaly") for f in cna_flags),
                "timing_vs_id_anomaly": next(
                    (f.get("timing_vs_id_anomaly") for f in cna_flags if f.get("near_employee_id_continuity_anomaly")),
                    "unrelated",
                ),
                "interpretation": (
                    f"{len(cna_flags)} reported CNA IDs met review-level sustained-work criteria this quarter. "
                    "Review source rows before drawing conclusions about individual workers or compliance."
                ),
                "source_review": {"pbj_employee_detail": [{"quarter": qn, "filter_provnum": str(facility_ccn)}]},
                "limitations": list(limitations),
                "_facility_summary": True,
            }
        )

    total_cna_hours = float(sum(float(f.get("_cna_hours") or 0.0) for f in employee_flags if f.get("quarter") == qn and f.get("role_group") == "CNA"))
    if total_cna_hours > 0:
        flagged_hours = float(
            sum(
                float(f.get("_cna_hours") or 0.0)
                for f in employee_flags
                if f.get("quarter") == qn
                and f.get("role_group") == "CNA"
                and str(f.get("severity") or "") in SUSTAINED_WORK_REVIEW_LEVELS
            )
        )
        share_flagged = flagged_hours / total_cna_hours
        if share_flagged >= 0.10:
            out.append(
                {
                    "severity": "review",
                    "label": "CNA hours concentrated in sustained-work IDs",
                    "facility_ccn": str(facility_ccn),
                    "quarter": qn,
                    "employee_id": None,
                    "masked_employee_id": None,
                    "role_group": "CNA",
                    "employee_type": "unknown",
                    "period_start": None,
                    "period_end": None,
                    "max_consecutive_shift_equivalent_days": 0,
                    "max_consecutive_full_shift_days": 0,
                    "rolling_90_worked_days": 0,
                    "rolling_90_hours": 0.0,
                    "avg_hours_per_worked_day": 0.0,
                    "max_daily_hours": 0.0,
                    "days_over_16_hours": 0,
                    "days_over_24_hours": 0,
                    "share_of_role_hours": round(share_flagged, 4),
                    "multi_role_same_day_count": 0,
                    "employee_and_contract_same_day_count": 0,
                    "near_employee_id_continuity_anomaly": False,
                    "timing_vs_id_anomaly": "unrelated",
                    "interpretation": (
                        f"At least {round(100.0 * share_flagged, 1)}% of reported CNA hours this quarter come from "
                        "employee IDs with review-level sustained-work flags."
                    ),
                    "source_review": {"pbj_employee_detail": [{"quarter": qn, "filter_provnum": str(facility_ccn)}]},
                    "limitations": list(limitations),
                    "_facility_summary": True,
                }
            )

        top = max(
            (
                f
                for f in employee_flags
                if f.get("quarter") == qn
                and f.get("role_group") == "CNA"
                and str(f.get("severity") or "") in SUSTAINED_WORK_REVIEW_LEVELS
            ),
            key=lambda f: float(f.get("_cna_hours") or 0.0),
            default=None,
        )
        if top and float(top.get("_cna_hours") or 0.0) / total_cna_hours >= 0.10:
            out.append(
                {
                    "severity": str(top.get("severity") or "review"),
                    "label": "Single CNA ID share of quarter hours",
                    "facility_ccn": str(facility_ccn),
                    "quarter": qn,
                    "employee_id": top.get("employee_id"),
                    "masked_employee_id": top.get("masked_employee_id"),
                    "role_group": "CNA",
                    "employee_type": top.get("employee_type"),
                    "period_start": top.get("period_start"),
                    "period_end": top.get("period_end"),
                    "max_consecutive_shift_equivalent_days": top.get("max_consecutive_shift_equivalent_days"),
                    "max_consecutive_full_shift_days": top.get("max_consecutive_full_shift_days"),
                    "rolling_90_worked_days": top.get("rolling_90_worked_days"),
                    "rolling_90_hours": top.get("rolling_90_hours"),
                    "avg_hours_per_worked_day": top.get("avg_hours_per_worked_day"),
                    "max_daily_hours": top.get("max_daily_hours"),
                    "days_over_16_hours": top.get("days_over_16_hours"),
                    "days_over_24_hours": top.get("days_over_24_hours"),
                    "share_of_role_hours": round(float(top.get("_cna_hours") or 0.0) / total_cna_hours, 4),
                    "multi_role_same_day_count": top.get("multi_role_same_day_count"),
                    "employee_and_contract_same_day_count": top.get("employee_and_contract_same_day_count"),
                    "near_employee_id_continuity_anomaly": top.get("near_employee_id_continuity_anomaly"),
                    "timing_vs_id_anomaly": top.get("timing_vs_id_anomaly"),
                    "interpretation": (
                        f"One flagged reported CNA ID accounts for about "
                        f"{round(100.0 * float(top.get('_cna_hours') or 0.0) / total_cna_hours, 1)}% of CNA hours this quarter."
                    ),
                    "source_review": top.get("source_review"),
                    "limitations": list(limitations),
                    "_facility_summary": True,
                }
            )

    plaus_major = [
        f
        for f in employee_flags
        if f.get("quarter") == qn
        and str(f.get("severity") or "") in {"major", "extreme"}
        and int(f.get("days_over_24_hours") or 0) >= 1
    ]
    if plaus_major:
        worst = max(plaus_major, key=lambda f: float(f.get("max_daily_hours") or 0.0))
        out.append(
            {
                "severity": str(worst.get("severity") or "major"),
                "label": "Reported daily hours plausibility review",
                "facility_ccn": str(facility_ccn),
                "quarter": qn,
                "employee_id": worst.get("employee_id"),
                "masked_employee_id": worst.get("masked_employee_id"),
                "role_group": worst.get("role_group"),
                "employee_type": worst.get("employee_type"),
                "period_start": worst.get("period_start"),
                "period_end": worst.get("period_end"),
                "max_consecutive_shift_equivalent_days": worst.get("max_consecutive_shift_equivalent_days"),
                "max_consecutive_full_shift_days": worst.get("max_consecutive_full_shift_days"),
                "rolling_90_worked_days": worst.get("rolling_90_worked_days"),
                "rolling_90_hours": worst.get("rolling_90_hours"),
                "avg_hours_per_worked_day": worst.get("avg_hours_per_worked_day"),
                "max_daily_hours": worst.get("max_daily_hours"),
                "days_over_16_hours": worst.get("days_over_16_hours"),
                "days_over_24_hours": worst.get("days_over_24_hours"),
                "share_of_role_hours": worst.get("share_of_role_hours"),
                "multi_role_same_day_count": worst.get("multi_role_same_day_count"),
                "employee_and_contract_same_day_count": worst.get("employee_and_contract_same_day_count"),
                "near_employee_id_continuity_anomaly": worst.get("near_employee_id_continuity_anomaly"),
                "timing_vs_id_anomaly": worst.get("timing_vs_id_anomaly"),
                "interpretation": (
                    "Reported employee-day totals exceed 24 hours for at least one date. "
                    "This is a data-quality review trigger, not proof of individual conduct."
                ),
                "source_review": worst.get("source_review"),
                "limitations": list(limitations),
                "_facility_summary": True,
            }
        )

    return out


def _prepare_sustained_work_flag_for_api(flag: dict[str, Any]) -> dict[str, Any]:
    """Strip internal keys and ensure API/UI flags use metrics-driven narrative."""
    row = {k: v for k, v in flag.items() if not str(k).startswith("_")}
    if row.get("employee_id"):
        narrative = build_sustained_work_flag_narrative(
            role_group=str(row.get("role_group") or ""),
            rolling_90_hours=float(row.get("rolling_90_hours") or 0),
            rolling_90_worked_days=int(row.get("rolling_90_worked_days") or 0),
            max_consecutive_shift_equivalent_days=int(row.get("max_consecutive_shift_equivalent_days") or 0),
            max_consecutive_full_shift_days=int(row.get("max_consecutive_full_shift_days") or 0),
            max_daily_hours=float(row.get("max_daily_hours") or 0),
            share_of_role_hours=float(row.get("share_of_role_hours") or 0),
            distinct_role_employees_in_period=int(row.get("distinct_role_employees_in_period") or 0),
            appears_across_role_groups=bool(row.get("appears_across_role_groups")),
            cross_role_groups=list(row.get("cross_role_groups") or []),
            appears_across_employee_contract=bool(row.get("appears_across_employee_contract")),
            window_days=int(row.get("window_days") or SUSTAINED_WORK_ROLLING_WINDOW_DAYS),
        )
        row["narrative"] = narrative
        row["interpretation"] = narrative
    lims: list[str] = []
    for lim in row.get("limitations") or []:
        s = str(lim)
        if "WORK_HRS_FN" in s or "fractional-hour" in s.lower():
            continue
        if "Same reported ID appears in multiple nursing role groups" in s:
            continue
        lims.append(s)
    row["limitations"] = lims
    return row


def _strip_sustained_work_internal_keys(flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_prepare_sustained_work_flag_for_api(f) for f in flags]


def ein_sustained_work_pattern_flags(
    detail: pd.DataFrame,
    *,
    facility_ccn: str,
    period_keys: Sequence[str],
    employee_id_continuity_by_period: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """
    Detect unusually sustained reported work patterns from consolidated Employee Detail rows.

    Returns ``sustained_work_pattern_flags`` (flat list) and ``sustained_work_pattern_flags_by_period``.
    """
    empty: dict[str, Any] = {
        "sustained_work_pattern_flags": [],
        "sustained_work_pattern_flags_by_period": {},
        "limitations": [],
    }
    role_day, emp_day_total, limitations = _consolidate_sustained_work_day_rows(detail)
    if role_day.empty:
        empty["limitations"] = limitations
        return empty

    continuity_quarters = _continuity_flag_quarters(employee_id_continuity_by_period or {})
    all_employee_flags: list[dict[str, Any]] = []
    by_period: dict[str, list[dict[str, Any]]] = {}

    quarters = [normalize_cy_qtr_ein(pk) for pk in period_keys]
    quarters = [q for q in quarters if q]

    def _q_sort(q: str) -> tuple:
        sk = parse_ein_quarter_bound(q)
        return (int(sk) if sk is not None else 0, q)

    if not quarters:
        quarters = sorted(role_day["CY_Qtr_norm"].dropna().astype(str).unique().tolist(), key=_q_sort)

    for qn in quarters:
        emp_flags = _employee_sustained_work_flags_for_quarter(
            quarter=qn,
            role_day=role_day,
            emp_day_total=emp_day_total,
            facility_ccn=facility_ccn,
            continuity_quarters=continuity_quarters,
            limitations=limitations,
        )
        fac_flags = _facility_sustained_work_summary_flags(
            emp_flags,
            quarter=qn,
            facility_ccn=facility_ccn,
            limitations=limitations,
        )
        combined = emp_flags + fac_flags
        by_period[qn] = _strip_sustained_work_internal_keys(combined)
        all_employee_flags.extend(combined)

    flat = _strip_sustained_work_internal_keys(all_employee_flags)
    sev_rank = SUSTAINED_WORK_SEVERITY_ORDER

    def _flat_sort_key(f: dict[str, Any]) -> tuple:
        return (
            -sev_rank.get(str(f.get("severity") or ""), 0),
            str(f.get("quarter") or ""),
            str(f.get("role_group") or ""),
            str(f.get("employee_id") or ""),
        )

    flat.sort(key=_flat_sort_key)
    return {
        "sustained_work_pattern_flags": flat,
        "sustained_work_pattern_flags_by_period": by_period,
        "limitations": limitations,
    }


def ein_headcount_by_job_longitudinal_series(
    detail: pd.DataFrame,
    *,
    grain: str = "quarter",
    slice_mode: str = "nurse",
    start_date: str | None = None,
    end_date: str | None = None,
    context: bool = True,
    quarter_pbj_context: dict[str, dict[str, float | None]] | None = None,
    facility_ccn: str = "",
    include_forensics: bool = True,
) -> dict[str, Any]:
    """
    Distinct EIN employee headcount by job code over time for stacked charting.

    Returns a payload shaped for ``/api/ein-headcount-by-job``:
    ``periods`` + ``job_codes_ordered`` + ``series``. Distinct counts are based on positive-hour rows.
    Each period includes ``employee_distinct_period`` (unique ``SYS_EMPLEE_ID`` in that slice and bucket,
    deduplicated across job codes) for chart tooltips alongside per-role stacked segments.
    """
    d = prepare_ein_detail(detail)
    if d.empty:
        return {"grain": grain, "slice": slice_mode, "periods": [], "job_codes_ordered": [], "series": [], "meta": {}}

    hrs = pd.to_numeric(d["WORK_HRS_NUM"], errors="coerce").fillna(0.0)
    pos = d.loc[hrs > 0].copy()
    if pos.empty:
        return {"grain": grain, "slice": slice_mode, "periods": [], "job_codes_ordered": [], "series": [], "meta": {}}

    slice_norm = str(slice_mode or "nurse").strip().lower()
    if slice_norm not in {"nurse", "nonnurse", "all"}:
        slice_norm = "nurse"

    jc_all = pd.to_numeric(pos["EMPLEE_JOB_CD_ID"], errors="coerce")
    if slice_norm == "nurse":
        pos = pos.loc[jc_all.between(5, 12, inclusive="both")].copy()
    elif slice_norm == "nonnurse":
        pos = pos.loc[~jc_all.between(5, 12, inclusive="both")].copy()
    if pos.empty:
        return {"grain": grain, "slice": slice_norm, "periods": [], "job_codes_ordered": [], "series": [], "meta": {}}

    g = str(grain or "quarter").strip().lower()
    if g not in {"day", "month", "quarter", "year"}:
        g = "quarter"

    wd_i = pd.to_numeric(pos["WorkDate"], errors="coerce")
    wd_valid = wd_i.notna()
    pos = pos.loc[wd_valid].copy()
    pos["WorkDate_i"] = wd_i.loc[wd_valid].astype(np.int64)
    pos["WorkDate_iso"] = pos["WorkDate_i"].map(workdate_to_iso)
    pos = pos.loc[pos["WorkDate_iso"].astype(str).str.len() == 10].copy()
    if pos.empty:
        return {"grain": g, "slice": slice_norm, "periods": [], "job_codes_ordered": [], "series": [], "meta": {}}

    audit_start = str(start_date or "").strip()[:10] or None
    audit_end = str(end_date or "").strip()[:10] or None
    display_start = audit_start
    display_end = audit_end
    pos_before_window = pos
    if audit_start and audit_end:
        display_start, display_end = _expand_headcount_display_window(
            audit_start, audit_end, grain=g, context=bool(context)
        )
        if g == "quarter":
            # CMS quarter buckets (CY_Qtr) must count all positive-hour rows in that quarter,
            # not only rows whose WorkDate falls inside the padded audit display window.
            # Otherwise context quarters (e.g. CY2024Q3 when audit starts in 2025) show
            # partial headcount (~62) until the user widens the PBJ date filter (~107).
            visible_q: list[str] = []
            for qn in pos_before_window["CY_Qtr_norm"].dropna().astype(str).unique():
                if _period_key_overlaps_iso_range(str(qn), "quarter", display_start, display_end):
                    visible_q.append(str(qn))
            if not visible_q:
                return {
                    "grain": g,
                    "slice": slice_norm,
                    "periods": [],
                    "job_codes_ordered": [],
                    "series": [],
                    "meta": {
                        "audit_start": audit_start,
                        "audit_end": audit_end,
                        "display_start": display_start,
                        "display_end": display_end,
                        "context_applied": bool(context),
                        "focus_period_keys": [],
                    },
                }
            pos = pos_before_window.loc[
                pos_before_window["CY_Qtr_norm"].astype(str).isin(visible_q)
            ].copy()
        else:
            pos = pos_before_window.loc[
                (pos_before_window["WorkDate_iso"].astype(str) >= display_start)
                & (pos_before_window["WorkDate_iso"].astype(str) <= display_end)
            ].copy()
        if pos.empty:
            return {
                "grain": g,
                "slice": slice_norm,
                "periods": [],
                "job_codes_ordered": [],
                "series": [],
                "meta": {
                    "audit_start": audit_start,
                    "audit_end": audit_end,
                    "display_start": display_start,
                    "display_end": display_end,
                    "context_applied": bool(context),
                    "focus_period_keys": [],
                },
            }

    if g == "day":
        pos["period_key"] = pos["WorkDate_iso"].astype(str)
    elif g == "month":
        pos["period_key"] = pos["WorkDate_iso"].astype(str).str.slice(0, 7)
    elif g == "year":
        pos["period_key"] = pos["WorkDate_iso"].astype(str).str.slice(0, 4)
    else:
        pos["period_key"] = pos["CY_Qtr_norm"].astype(str)

    pos["SYS_EMPLEE_ID"] = pd.to_numeric(pos["SYS_EMPLEE_ID"], errors="coerce").astype("Int64")
    pos["EMPLEE_JOB_CD_ID"] = pd.to_numeric(pos["EMPLEE_JOB_CD_ID"], errors="coerce").astype("Int64")
    pos["EMP_CTR"] = pd.to_numeric(pos["EMP_CTR"], errors="coerce").fillna(0).astype(int)
    pos = pos.loc[pos["SYS_EMPLEE_ID"].notna() & pos["EMPLEE_JOB_CD_ID"].notna()].copy()
    if pos.empty:
        return {"grain": g, "slice": slice_norm, "periods": [], "job_codes_ordered": [], "series": [], "meta": {}}

    def _period_sort_key(pk: str) -> tuple:
        if g == "day":
            return (pk,)
        if g == "month":
            return (pk,)
        if g == "year":
            try:
                return (int(pk),)
            except Exception:
                return (pk,)
        sk = parse_ein_quarter_bound(pk)
        return (int(sk) if sk is not None else 0, pk)

    period_keys = sorted(pos["period_key"].dropna().astype(str).unique().tolist(), key=_period_sort_key)
    if not period_keys:
        return {"grain": g, "slice": slice_norm, "periods": [], "job_codes_ordered": [], "series": [], "meta": {}}

    def _period_label(pk: str) -> str:
        if g == "day":
            return pk
        if g == "month":
            try:
                return datetime.strptime(pk + "-01", "%Y-%m-%d").strftime("%b %Y")
            except Exception:
                return pk
        if g == "year":
            return pk
        sk_label = parse_ein_quarter_bound(pk)
        if sk_label is not None:
            return ein_quarter_sort_key_to_label(sk_label) or pk
        return pk

    def _job_category(code: int) -> str:
        if code == 1:
            return "Admin"
        if 5 <= code <= 12:
            return "Nursing"
        if code in (2, 3, 4, 13, 14):
            return "Medical"
        if code in (15, 16, 17):
            return "Support"
        if 18 <= code <= 26:
            return "Therapy"
        if 27 <= code <= 29:
            return "Activities"
        if code in (30, 31, 34):
            return "Social"
        return "Other"

    grp = (
        pos.groupby(["period_key", "EMPLEE_JOB_CD_ID"], as_index=False)
        .agg(
            employee_distinct=("SYS_EMPLEE_ID", "nunique"),
            contract_distinct=("SYS_EMPLEE_ID", lambda s: s.loc[pos.loc[s.index, "EMP_CTR"] == 2].nunique()),
        )
        .sort_values(["period_key", "EMPLEE_JOB_CD_ID"])
    )

    series: list[dict[str, Any]] = []
    for _, row in grp.iterrows():
        jc = int(row["EMPLEE_JOB_CD_ID"])
        pk = str(row["period_key"])
        series.append(
            {
                "period_key": pk,
                "job_code": jc,
                "employee_distinct": int(row["employee_distinct"] or 0),
                "contract_distinct": int(row["contract_distinct"] or 0),
                "job_title_short": job_title_short(jc),
                "job_title": job_title(jc),
                "job_category": _job_category(jc),
            }
        )

    job_codes_ordered = sorted({int(r["job_code"]) for r in series})
    periods: list[dict[str, Any]] = []
    for pk in period_keys:
        sub = pos.loc[pos["period_key"].astype(str) == str(pk)].copy()
        if sub.empty:
            continue
        by_emp = sub.groupby("SYS_EMPLEE_ID")["EMPLEE_JOB_CD_ID"].nunique()
        periods.append(
            {
                "key": str(pk),
                "label": _period_label(str(pk)),
                "employee_distinct_period": int(sub["SYS_EMPLEE_ID"].nunique()),
                "contract_distinct_dedup": int(sub.loc[sub["EMP_CTR"] == 2, "SYS_EMPLEE_ID"].nunique()),
                "multi_role_employees": int((by_emp > 1).sum()),
            }
        )

    meta: dict[str, Any] = {}
    if audit_start and audit_end:
        meta = {
            "audit_start": audit_start,
            "audit_end": audit_end,
            "display_start": display_start,
            "display_end": display_end,
            "context_applied": bool(context),
            "focus_period_keys": [
                pk
                for pk in period_keys
                if _period_key_overlaps_iso_range(pk, g, audit_start, audit_end)
            ],
        }

    continuity_pos = pos_before_window.copy()
    continuity_pos["CY_Qtr_norm"] = continuity_pos["CY_Qtr_norm"].astype(str)
    employee_id_continuity_by_period: dict[str, Any] = {}
    sustained_work: dict[str, Any] = {
        "sustained_work_pattern_flags": [],
        "sustained_work_pattern_flags_by_period": {},
        "limitations": [],
    }
    if include_forensics:
        employee_id_continuity_by_period = ein_employee_id_continuity_by_period(
            continuity_pos, period_keys, g, quarter_pbj_context=quarter_pbj_context
        )
        sustained_work = ein_sustained_work_pattern_flags(
            detail,
            facility_ccn=str(facility_ccn or ""),
            period_keys=period_keys,
            employee_id_continuity_by_period=employee_id_continuity_by_period,
        )

    return {
        "grain": g,
        "slice": slice_norm,
        "periods": periods,
        "job_codes_ordered": job_codes_ordered,
        "series": series,
        "meta": meta,
        "employee_id_continuity_by_period": employee_id_continuity_by_period,
        "sustained_work_pattern_flags": sustained_work.get("sustained_work_pattern_flags") or [],
        "sustained_work_pattern_flags_by_period": sustained_work.get("sustained_work_pattern_flags_by_period") or {},
        "sustained_work_limitations": sustained_work.get("limitations") or [],
    }


def ein_nursing_roster_for_work_date(
    detail: pd.DataFrame,
    work_date_iso: str,
) -> tuple[list[dict[str, Any]], float, str | None]:
    """Administrator (1) plus licensed / direct nursing job codes (5–12) for one work date."""
    return aggregate_ein_day_by_job_codes(
        detail, work_date_iso, frozenset(NURSING_JOB_CODE_IDS)
    )
