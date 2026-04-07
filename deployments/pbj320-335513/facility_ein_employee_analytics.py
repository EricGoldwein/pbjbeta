"""
Employee-level summaries from PBJ Employee Detail (EIN) row-level table (Parquet or CSV).

Nursing-related job codes follow CMS PUF dictionary. The dashboard roster also includes job code
1 (facility Administrator) so Employee Detail aligns with Provider Information “administrator” concepts;
licensed and direct-care nursing remain codes 5–12.
"""

from __future__ import annotations

from collections import defaultdict
import statistics
from typing import Any

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
    d["SYS_EMPLEE_ID"] = pd.to_numeric(d["SYS_EMPLEE_ID"], errors="coerce")
    d["EMPLEE_JOB_CD_ID"] = pd.to_numeric(d["EMPLEE_JOB_CD_ID"], errors="coerce")
    d["EMP_CTR"] = pd.to_numeric(d["EMP_CTR"], errors="coerce")
    d["CY_Qtr_norm"] = d["CY_Qtr"].apply(normalize_cy_qtr_ein)
    d = d[d["WorkDate"].notna() & d["SYS_EMPLEE_ID"].notna() & d["EMPLEE_JOB_CD_ID"].notna()]
    d = d[d["CY_Qtr_norm"].notna()]
    return pd.DataFrame(d)


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


def format_ein_tenure_span_days(days: int | None) -> str | None:
    """Short label for tenure from first work day to a reference day."""
    if days is None or days < 0:
        return None
    if days < 14:
        return "<1 mo"
    if days < 365:
        mo = max(1, int(round(days / 30.44)))
        return f"{mo} mo"
    yr = days / 365.25
    return f"{yr:.1f} yr"


def previous_cy_quarter_label(quarter: str | None) -> str | None:
    """Calendar quarter immediately before ``quarter`` (CYyyyyQn), or None if unknown."""
    sk = parse_ein_quarter_bound(quarter)
    if sk is None:
        return None
    return ein_quarter_sort_key_to_label(sk - 1)


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


def enrich_nursing_rows_new_to_quarter_flags(rows: list[dict[str, Any]]) -> None:
    """
    Set ``is_new_to_quarter`` and ``new_to_quarter_known`` on each roster row (mutates in place).

    New = this (employee, job_code) pair did not appear in the **previous** calendar quarter
    **in the loaded extract**. If the prior quarter is missing from the file, flags are unknown
    (badge suppressed).
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
        pq = previous_cy_quarter_label(qn)
        if not pq or pq not in pairs_by_q:
            r["is_new_to_quarter"] = False
            r["new_to_quarter_known"] = False
            continue
        r["new_to_quarter_known"] = True
        r["is_new_to_quarter"] = (eid, jcid) not in pairs_by_q[pq]


def employee_job_pair_new_to_quarter(
    pairs_by_q: dict[str, set[tuple[int, int]]],
    quarter: str,
    sys_employee_id: int,
    job_code: int,
) -> bool | None:
    """True / False / None (unknown: missing prior quarter in extract)."""
    qn = normalize_cy_qtr_ein(quarter)
    if not qn:
        return None
    pq = previous_cy_quarter_label(qn)
    if not pq or pq not in pairs_by_q:
        return None
    return (int(sys_employee_id), int(job_code)) not in pairs_by_q[pq]


def compute_ein_quarter_roster_summary(
    quarter: str,
    pairs_by_q: dict[str, set[tuple[int, int]]],
) -> dict[str, Any] | None:
    """
    Headcounts and “new vs prior quarter” stats for one roster quarter.

    Distinct people can appear in both nurse and admin buckets; direct nursing is a subset of
    nurse. ``new_*`` counts are **distinct employees** (by ID) unless noted.
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
        try:
            sid = int(r.get("sys_employee_id"))
        except (TypeError, ValueError):
            sid = -1
        try:
            jc = int(r.get("job_code"))
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


def apply_roster_tenure_quarter_span(rows: list[dict[str, Any]]) -> None:
    """Mutate rows: set ``tenure_quarter_lo`` / ``tenure_quarter_hi`` (``CYyyyyQn``) per employee × job.

    Computed from the **full** ``rows`` list (e.g. all quarters matching filters) so pagination
    does not shrink the displayed quarter span in the UI.
    """
    if not rows:
        return
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
        t = span.get((sid, jc))
        if not t:
            r["tenure_quarter_lo"] = None
            r["tenure_quarter_hi"] = None
            continue
        lo, hi = t
        r["tenure_quarter_lo"] = ein_quarter_sort_key_to_label(lo)
        r["tenure_quarter_hi"] = ein_quarter_sort_key_to_label(hi)


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
    sub = d[d["WorkDate"] == wd]
    if job_codes:
        codes_list = list(job_codes)
        sub = sub[sub["EMPLEE_JOB_CD_ID"].astype(int).isin(codes_list)]
    if sub.empty:
        return [], 0.0, q_fallback
    qn = str(sub["CY_Qtr_norm"].iloc[0])
    pairs_by_q = roster_pairs_by_quarter_for_job_codes(d, job_codes) if job_codes else {}
    scope = d[d["EMPLEE_JOB_CD_ID"].astype(int).isin(list(job_codes))] if job_codes else d
    pair_first_wd = scope.groupby(["SYS_EMPLEE_ID", "EMPLEE_JOB_CD_ID"], sort=False)["WorkDate"].min()
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
        tenure_label = format_ein_tenure_span_days(span_days)
        nf = employee_job_pair_new_to_quarter(pairs_by_q, qn, eii, ji)
        new_known = nf is not None
        is_new = bool(nf) if new_known else False
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
                "is_new_to_quarter": is_new,
                "new_to_quarter_known": new_known,
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


def ein_nursing_roster_for_work_date(
    detail: pd.DataFrame,
    work_date_iso: str,
) -> tuple[list[dict[str, Any]], float, str | None]:
    """Administrator (1) plus licensed / direct nursing job codes (5–12) for one work date."""
    return aggregate_ein_day_by_job_codes(
        detail, work_date_iso, frozenset(NURSING_JOB_CODE_IDS)
    )
