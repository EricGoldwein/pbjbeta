#!/usr/bin/env python3
"""Apply deploy-safety patches to 315128 monolith (roster API + summary filters)."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deployments" / "pbj320-315128"
ENTRY = DEPLOY / "facility_315128_superdynamic_dashboard.py"
CANON = ROOT / "dynamic_facility_dashboard.py"

OLD_PRECOMPUTED = """        if use_precomputed:
            df_all = cast(pd.DataFrame, ein_nursing_summaries_df)
            rows_all = dedupe_nursing_roster_api_rows(_json_rows_from_df(df_all))
            enrich_nursing_roster_display_fields(rows_all)
            enrich_nursing_rows_new_to_quarter_flags(rows_all)
            enrich_nursing_rows_multi_role_flags(rows_all)
            if detail_ready:
                enrich_nursing_rows_sustained_work_flags(
                    ein_employee_detail_df,
                    rows_all,
                    facility_ccn=ccn,
                    limit_quarters=[q_filter_norm] if q_filter_norm else None,
                )
            pairs_by_q = roster_pairs_by_quarter_from_rows(rows_all)
            quarter_summary = (
                compute_ein_quarter_roster_summary(q_filter_norm, pairs_by_q) if q_filter_norm else None
            )
            if q_filter_norm and "quarter" in df_all.columns:
                filtered = [
                    r for r in rows_all if normalize_cy_qtr_ein(r.get("quarter")) == q_filter_norm
                ]
            else:
                filtered = rows_all
            if position_group and position_group != "all":
                filtered = [
                    r for r in filtered if ein_job_code_matches_position_group(r.get("job_code"), position_group)
                ]
            apply_roster_tenure_quarter_span(filtered, rows_for_global_bounds=rows_all)
            total = int(len(filtered))
            filtered.sort(
                key=lambda row: (
                    -(parse_ein_quarter_bound(str(row.get("quarter") or "")) or 0),
                    -float(row.get("total_hours") or 0),
                    int(row.get("first_work_date_raw") or 0) or 10**9,
                    int(row.get("sys_employee_id") or 0),
                    int(row.get("job_code") or 0),
                )
            )
            if limit_raw is None:
                rows = filtered[offset:]
            else:
                lim = max(1, int(limit_raw))
                rows = filtered[offset : offset + lim]
            _enrich_nursing_api_rows(rows, ccn)
            if (
                work_date_anchor
                and ein_employee_detail_df is not None
                and not ein_employee_detail_df.empty
            ):
                enrich_nursing_rows_rolling_from_work_date(
                    ein_employee_detail_df, rows, work_date_anchor
                )
            return jsonify(
                {
                    "available": True,
                    "employees": rows,
                    "quarter_filter": quarter,
                    "position_group": position_group,
                    "quarter_summary": quarter_summary,
                    "provnum": ccn,
                    "cms_ein_landing_url": CMS_EIN_DETAIL_LANDING_URL,
                    "total": total,
                    "limit": len(rows),
                    "offset": offset,
                    "truncated": offset + len(rows) < total,
                    "source": "precomputed",
                }
            )"""

NEW_PRECOMPUTED = """        if use_precomputed:
            rows_all = _ein_nursing_summaries_rows_all()
            pairs_by_q = roster_pairs_by_quarter_from_rows(rows_all)
            quarter_summary = (
                compute_ein_quarter_roster_summary(q_filter_norm, pairs_by_q) if q_filter_norm else None
            )
            work_day_filtered = False
            if q_filter_norm:
                filtered = [
                    r for r in rows_all if normalize_cy_qtr_ein(r.get("quarter")) == q_filter_norm
                ]
            else:
                filtered = rows_all
            if position_group and position_group != "all":
                filtered = [
                    r for r in filtered if ein_job_code_matches_position_group(r.get("job_code"), position_group)
                ]
            if (
                work_date_anchor
                and ein_employee_detail_df is not None
                and not ein_employee_detail_df.empty
            ):
                pair_keys = ein_roster_work_day_pair_keys(
                    ein_employee_detail_df, work_date_anchor, NURSING_JOB_CODE_IDS
                )
                filtered = filter_ein_roster_rows_for_work_day(filtered, pair_keys)
                work_day_filtered = True
            apply_roster_tenure_quarter_span(filtered, rows_for_global_bounds=rows_all)
            total = int(len(filtered))
            filtered.sort(
                key=lambda row: (
                    -(parse_ein_quarter_bound(str(row.get("quarter") or "")) or 0),
                    -float(row.get("total_hours") or 0),
                    int(row.get("first_work_date_raw") or 0) or 10**9,
                    int(row.get("sys_employee_id") or 0),
                    int(row.get("job_code") or 0),
                )
            )
            if limit_raw is None:
                rows = filtered[offset:]
            else:
                lim = max(1, int(limit_raw))
                rows = filtered[offset : offset + lim]
            _enrich_nursing_api_rows(rows, ccn)
            if (
                ein_employee_detail_df is not None
                and not ein_employee_detail_df.empty
                and rows
            ):
                enrich_nursing_rows_sustained_work_flags(
                    ein_employee_detail_df,
                    rows,
                    facility_ccn=ccn,
                    limit_quarters=[q_filter_norm] if q_filter_norm else None,
                )
            if (
                work_date_anchor
                and ein_employee_detail_df is not None
                and not ein_employee_detail_df.empty
                and rows
            ):
                enrich_nursing_rows_rolling_from_work_date(
                    ein_employee_detail_df, rows, work_date_anchor
                )
            return jsonify(
                {
                    "available": True,
                    "employees": rows,
                    "quarter_filter": quarter,
                    "position_group": position_group,
                    "quarter_summary": quarter_summary,
                    "provnum": ccn,
                    "cms_ein_landing_url": CMS_EIN_DETAIL_LANDING_URL,
                    "total": total,
                    "limit": len(rows),
                    "offset": offset,
                    "truncated": offset + len(rows) < total,
                    "source": "precomputed",
                    "work_day_filtered": work_day_filtered,
                }
            )"""


def main() -> int:
    if not ENTRY.is_file():
        print(f"missing {ENTRY}", file=sys.stderr)
        return 1

    shutil.copy2(ROOT / "facility_ein_employee_analytics.py", DEPLOY / "facility_ein_employee_analytics.py")

    text = ENTRY.read_text(encoding="utf-8-sig")

    gate = "            and (not work_anchor or not detail_ready)\n"
    if gate in text:
        text = text.replace(gate, "", 1)
        print("removed use_precomputed detail gate")

    if OLD_PRECOMPUTED in text:
        text = text.replace(OLD_PRECOMPUTED, NEW_PRECOMPUTED, 1)
        print("patched precomputed roster block")
    elif "work_day_filtered" in text.split("api_ein_nursing_employees", 1)[-1][:4000]:
        print("roster block already patched")
    else:
        print("WARN: precomputed roster block not found", file=sys.stderr)
        return 1

    norm_fn = re.search(
        r"def _normalize_dashboard_quarter_year_args\(quarter: str, year: str\) -> tuple\[str, str\]:.*?\n    return q_raw, y_raw\n",
        CANON.read_text(encoding="utf-8"),
        re.S,
    )
    if norm_fn and "_normalize_dashboard_quarter_year_args" not in text:
        anchor = "def _filter_facility_daily_for_dashboard("
        text = text.replace(anchor, norm_fn.group(0) + "\n\n" + anchor, 1)
        print("inserted _normalize_dashboard_quarter_year_args")
    if "_normalize_dashboard_quarter_year_args" in text and "quarter, year = _normalize_dashboard_quarter_year_args" not in text:
        text = text.replace(
            '    """Same calendar filters as ``/api/data`` and ``/api/summary`` (inclusive end date).\n\n    Coerces ``WorkDate``',
            '    quarter, year = _normalize_dashboard_quarter_year_args(quarter, year)\n    """Same calendar filters as ``/api/data`` and ``/api/summary`` (inclusive end date).\n\n    Coerces ``WorkDate``',
            1,
        )
        print("wired quarter/year normalization into filter")

    text = text.replace(
        "'total_rn_sub8': int((filtered_df['Total_RN_Hours'] < 8).sum()) if len(filtered_df) > 0 else 0,",
        "'total_rn_sub8': int((filtered_df['Total_RN_Hours'] < 8).sum()) if len(filtered_df) > 0 else None,",
    )
    text = text.replace(
        "'direct_rn_sub8': int((filtered_df['Hrs_RN'] < 8).sum()) if len(filtered_df) > 0 else 0,",
        "'direct_rn_sub8': int((filtered_df['Hrs_RN'] < 8).sum()) if len(filtered_df) > 0 else None,",
    )

    fea_bindings = """ein_roster_work_day_pair_keys = _fea_get("ein_roster_work_day_pair_keys", lambda *a, **k: set())
filter_ein_roster_rows_for_work_day = _fea_get(
    "filter_ein_roster_rows_for_work_day",
    _fea_passthrough_rows,
)
NURSING_JOB_CODE_IDS = _fea_get("NURSING_JOB_CODE_IDS", frozenset())
"""
    anchor = 'workdate_to_iso = _fea_get("workdate_to_iso", lambda *a, **k: None)'
    if "ein_roster_work_day_pair_keys = _fea_get" not in text:
        text = text.replace(anchor, fea_bindings + anchor, 1)
        print("patched _fea_get roster work-day bindings")

    imports = (
        "    enrich_nursing_rows_rolling_from_work_date,\n"
        "    ein_roster_work_day_pair_keys,\n"
        "    filter_ein_roster_rows_for_work_day,\n"
        "    NURSING_JOB_CODE_IDS,\n"
    )
    if "ein_roster_work_day_pair_keys" not in text.split("import facility_ein_employee_analytics", 1)[0][-5000:]:
        pass  # deploy uses _fea_get, not direct imports

    ENTRY.write_text(text, encoding="utf-8")
    print(f"updated {ENTRY.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
