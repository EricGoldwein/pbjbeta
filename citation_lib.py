"""
NH Health Citations: locate CMS files, slice per CCN, normalize for APIs.

PDF and landing URLs use templates from ``pbj_identifiers.urls``; null when inputs invalid.
"""

from __future__ import annotations

import glob
import json
import os
import re
from collections import Counter
from typing import Any, Optional

import pandas as pd

from pbj_identifiers.urls import (
    cms_nh_health_citations_dataset_explorer_url,
    medicare_nursing_home_complaint_inspection_pdf_url,
    medicare_nursing_home_health_inspection_pdf_url,
    medicare_nursing_home_infection_control_inspection_pdf_url,
    medicare_nursing_home_inspection_pdf_url,
    medicare_nursing_home_inspections_landing_url,
)
from pbj_identifiers.validators import normalize_ccn, validate_ccn
from pbj_staffing_normalize import iso_date_or_none, parse_citation_date_series


CCN_COL = "CMS Certification Number (CCN)"


def _repo_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def default_citation_severity_config_path() -> str:
    return os.path.join(_repo_root(), "config", "citation_severity_rank.json")


def load_citation_severity_config(path: Optional[str] = None) -> dict[str, Any]:
    with open(path or default_citation_severity_config_path(), encoding="utf-8") as f:
        return json.load(f)


def _month_year_from_health_citations_filename(name: str) -> Optional[tuple[int, int]]:
    m = re.search(r"NH_HealthCitations_([A-Za-z]{3})(\d{4})\.csv$", name)
    if not m:
        return None
    mon_abbr, year_s = m.group(1).title(), int(m.group(2))
    month_map = {
        "Jan": 1,
        "Feb": 2,
        "Mar": 3,
        "Apr": 4,
        "May": 5,
        "Jun": 6,
        "Jul": 7,
        "Aug": 8,
        "Sep": 9,
        "Oct": 10,
        "Nov": 11,
        "Dec": 12,
    }
    mo = month_map.get(mon_abbr)
    if not mo:
        return None
    return int(year_s), mo


def find_latest_nh_health_citations_csv(root: Optional[str] = None) -> Optional[str]:
    """Pick latest ``Citations/NH_HealthCitations_MonYYYY.csv`` by (year, month)."""
    base = os.path.join(root or _repo_root(), "Citations", "NH_HealthCitations_*.csv")
    files = glob.glob(base)
    best: Optional[tuple[tuple[int, int], str]] = None
    for f in files:
        key = _month_year_from_health_citations_filename(os.path.basename(f))
        if key is None:
            continue
        if best is None or key > best[0]:
            best = (key, f)
    return best[1] if best else None


def _yn_to_bool(val: Any) -> Optional[bool]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper()
    if s == "Y":
        return True
    if s == "N":
        return False
    return None


def _severity_code_raw(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    s = str(val).strip().upper()
    if len(s) == 1 and s.isalpha():
        return s
    m = re.search(r"([A-L])", s)
    return m.group(1) if m else ""


def build_facility_citations_csv(
    ccn: str,
    out_path: str,
    source_csv: Optional[str] = None,
    root: Optional[str] = None,
) -> Optional[int]:
    """
    Write rows for one CCN from national NH_HealthCitations file using usecols + filter.
    Returns row count or None if source missing.
    """
    src = source_csv or find_latest_nh_health_citations_csv(root)
    if not src or not os.path.isfile(src):
        return None
    ccn_canon = normalize_ccn(str(ccn).strip()) if validate_ccn(str(ccn).strip()) else None
    if not ccn_canon:
        return None
    variants = {ccn_canon, ccn_canon.lstrip("0") or "0", ccn_canon.upper()}
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(src, low_memory=False, dtype=str, chunksize=150_000):
        chunk.columns = [str(c).strip().replace("\ufeff", "") for c in chunk.columns]
        if CCN_COL not in chunk.columns:
            continue
        s = chunk[CCN_COL].astype(str).str.strip()
        s = s.apply(lambda x: x.zfill(6) if x.isdigit() else x.upper())
        mask = s.isin(variants)
        hit = chunk.loc[mask]
        if len(hit):
            parts.append(hit)
    if not parts:
        header_df = pd.read_csv(src, nrows=0, low_memory=False, dtype=str)
        header_df.columns = [str(c).strip().replace("\ufeff", "") for c in header_df.columns]
        header_df.to_csv(out_path, index=False)
        return 0
    sub = pd.concat(parts, ignore_index=True)
    sub.to_csv(out_path, index=False)
    return int(len(sub))


def _pdf_reason(ccn_ok: bool, survey_iso: Optional[str]) -> Optional[str]:
    if not ccn_ok:
        return "invalid_ccn"
    if not survey_iso:
        return "missing_date"
    return None


def enrich_citations_dataframe(df: pd.DataFrame, ccn: str, severity_cfg: Optional[dict[str, Any]] = None) -> pd.DataFrame:
    """Add normalized columns and URLs; does not mutate input."""
    cfg = severity_cfg or load_citation_severity_config()
    ranks: dict[str, Any] = dict(cfg.get("harm_rank_by_scope_severity_code", {}))
    high_min = int(cfg.get("high_severity_min_rank", 70))
    dash_min = int(cfg.get("dashboard_citation_flag_min_rank", 60))
    default_rank = cfg.get("default_rank_for_unknown_code")

    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "") for c in out.columns]

    ccn_ok = validate_ccn(str(ccn).strip())
    ccn_norm = normalize_ccn(str(ccn).strip()) if ccn_ok else ""

    survey_ts = parse_citation_date_series(out["Survey Date"]) if "Survey Date" in out.columns else pd.Series(pd.NaT, index=out.index)
    correction_ts = (
        parse_citation_date_series(out["Correction Date"]) if "Correction Date" in out.columns else pd.Series(pd.NaT, index=out.index)
    )
    processing_ts = (
        parse_citation_date_series(out["Processing Date"]) if "Processing Date" in out.columns else pd.Series(pd.NaT, index=out.index)
    )

    out["_survey_ts"] = survey_ts
    out["survey_date_iso"] = survey_ts.map(iso_date_or_none)
    out["correction_date_iso"] = correction_ts.map(iso_date_or_none)
    out["processing_date_iso"] = processing_ts.map(iso_date_or_none)

    if "Scope Severity Code" in out.columns:
        out["scope_severity_code"] = out["Scope Severity Code"].map(_severity_code_raw)
    else:
        out["scope_severity_code"] = ""

    def rank_for_code(code: str) -> Optional[int]:
        if not code:
            return int(default_rank) if default_rank is not None else None
        v = ranks.get(code[0].upper() if code else "")
        return int(v) if v is not None else (int(default_rank) if default_rank is not None else None)

    out["severity_rank"] = out["scope_severity_code"].map(lambda c: rank_for_code(str(c) if c is not None and not (isinstance(c, float) and pd.isna(c)) else ""))

    def _is_high_sev(r: Any) -> bool:
        if r is None or (isinstance(r, float) and pd.isna(r)):
            return False
        try:
            return int(r) >= high_min
        except (TypeError, ValueError):
            return False

    out["is_high_severity"] = out["severity_rank"].map(_is_high_sev)

    def _is_dashboard_flag_sev(r: Any) -> bool:
        if r is None or (isinstance(r, float) and pd.isna(r)):
            return False
        try:
            return int(r) >= dash_min
        except (TypeError, ValueError):
            return False

    out["is_dashboard_citation_flag"] = out["severity_rank"].map(_is_dashboard_flag_sev)

    if "Complaint Deficiency" in out.columns:
        out["complaint_deficiency_bool"] = out["Complaint Deficiency"].map(_yn_to_bool)
    else:
        out["complaint_deficiency_bool"] = pd.NA
    if "Infection Control Inspection Deficiency" in out.columns:
        out["infection_control_deficiency_bool"] = out["Infection Control Inspection Deficiency"].map(_yn_to_bool)
    else:
        out["infection_control_deficiency_bool"] = pd.NA

    cdb = out["complaint_deficiency_bool"].fillna(False).astype(bool) if "complaint_deficiency_bool" in out.columns else pd.Series(
        False, index=out.index
    )
    out["_survey_has_complaint"] = (
        out.assign(_cdb=cdb).groupby("_survey_ts", dropna=False)["_cdb"].transform("any")
    )

    landing = medicare_nursing_home_inspections_landing_url(ccn_norm) if ccn_ok else None

    survey_pdf_by_iso: dict[str, tuple[str, Optional[str]]] = {}
    if ccn_ok and "survey_date_iso" in out.columns:
        for siso, grp in out.groupby("survey_date_iso", dropna=True):
            if not isinstance(siso, str) or not siso:
                continue
            kind = resolve_survey_medicare_inspection_pdf_kind(grp)
            url = medicare_nursing_home_inspection_pdf_url(ccn_norm, siso, kind) if kind else None
            survey_pdf_by_iso[siso] = (kind, url)

    pdf_urls: list[Optional[str]] = []
    pdf_kinds: list[Optional[str]] = []
    pdf_reasons: list[Optional[str]] = []
    for i in range(len(out)):
        siso = out["survey_date_iso"].iloc[i]
        if ccn_ok and isinstance(siso, str) and siso and siso in survey_pdf_by_iso:
            kind, pdf = survey_pdf_by_iso[siso]
            pdf_urls.append(pdf)
            pdf_kinds.append(kind)
            pdf_reasons.append(None if pdf else "invalid_date")
        else:
            pdf_urls.append(None)
            pdf_kinds.append(None)
            pdf_reasons.append(_pdf_reason(ccn_ok, siso if isinstance(siso, str) else None))

    out["care_compare_inspections_landing_url"] = landing
    out["medicare_inspection_pdf_kind"] = pd.Series(pdf_kinds, index=out.index, dtype=object)
    out["pdf_url"] = pd.Series(pdf_urls, index=out.index, dtype=object)
    out["pdf_url_unavailable_reason"] = pd.Series(pdf_reasons, index=out.index, dtype=object)

    def inspection_kind(row: pd.Series) -> str:
        if row.get("infection_control_deficiency_bool") is True:
            return "infection_control"
        st = str(row.get("Survey Type", "") or "").strip().lower()
        if "complaint" in st:
            return "complaint_related"
        if row.get("complaint_deficiency_bool") is True:
            return "complaint_related"
        return "health"

    out["inspection_kind"] = out.apply(inspection_kind, axis=1)
    return out


def _citation_row_for_topic_match(row: Any) -> dict[str, Any]:
    """Minimal citation dict for ``match_citation_topics`` on enriched dataframe rows."""
    prefix = str(row.get("Deficiency Prefix") or "F").strip()
    tag_num = row.get("Deficiency Tag Number")
    return {
        "deficiency_tag": format_deficiency_tag(prefix, tag_num),
        "deficiency_category": row.get("Deficiency Category"),
        "deficiency_description": row.get("Deficiency Description"),
        "scope_severity_code": row.get("scope_severity_code"),
        "is_complaint_deficiency": bool(row.get("complaint_deficiency_bool")),
        "is_infection_control_deficiency": bool(row.get("infection_control_deficiency_bool")),
    }


def citations_to_api_rows(df: pd.DataFrame, ccn: str, max_rows: int = 500) -> list[dict[str, Any]]:
    """Flatten normalized citation rows for JSON (newest first)."""
    if df.empty:
        return []
    d = df if "pdf_url" in df.columns else enrich_citations_dataframe(df, ccn)
    if "_survey_ts" in d.columns:
        d = d.sort_values("_survey_ts", ascending=False, na_position="last")
    d = d.head(max_rows)
    from citation_taxonomy import match_citation_topics

    rows: list[dict[str, Any]] = []
    for _, r in d.iterrows():
        risk_topic_ids = match_citation_topics(_citation_row_for_topic_match(r))
        rows.append(
            {
                "survey_date": r.get("survey_date_iso"),
                "survey_type_raw": r.get("Survey Type"),
                "inspection_kind": r.get("inspection_kind"),
                "deficiency_prefix": r.get("Deficiency Prefix"),
                "deficiency_tag": r.get("Deficiency Tag Number"),
                "deficiency_category": r.get("Deficiency Category"),
                "description": r.get("Deficiency Description"),
                "scope_severity_code": r.get("scope_severity_code") or None,
                "severity_rank": int(r["severity_rank"]) if pd.notna(r.get("severity_rank")) else None,
                "is_high_severity": bool(r.get("is_high_severity")),
                "is_dashboard_citation_flag": bool(r.get("is_dashboard_citation_flag")),
                "complaint_deficiency": r.get("complaint_deficiency_bool"),
                "infection_control_inspection_deficiency": r.get("infection_control_deficiency_bool"),
                "correction_date": r.get("correction_date_iso"),
                "processing_date": r.get("processing_date_iso"),
                "pdf_url": r.get("pdf_url"),
                "medicare_inspection_pdf_kind": r.get("medicare_inspection_pdf_kind"),
                "medicare_inspection_page_url": r.get("pdf_url") or r.get("care_compare_inspections_landing_url"),
                "pdf_url_unavailable_reason": r.get("pdf_url_unavailable_reason"),
                "care_compare_inspections_landing_url": r.get("care_compare_inspections_landing_url"),
                "risk_topic_ids": risk_topic_ids,
            }
        )
    return rows


def citations_summary(
    df: pd.DataFrame,
    ccn: str,
    pbj_min: Optional[str] = None,
    pbj_max: Optional[str] = None,
) -> dict[str, Any]:
    """Counts and timeline seed for UI."""
    sev_cfg = load_citation_severity_config()
    high_rank_default = int(sev_cfg.get("high_severity_min_rank", 70))
    dash_rank_default = int(sev_cfg.get("dashboard_citation_flag_min_rank", 60))
    survey_q_note = (
        "Deficiency rows are grouped by the calendar quarter of the survey (inspection) date. "
        "That quarter label may not match the facility's PBJ work-date quarter or CMS Provider Information "
        "processing quarter for the same period."
    )
    cms_cit = cms_nh_health_citations_dataset_explorer_url(ccn)
    if df.empty:
        return {
            "total": 0,
            "complaint_related_rows": 0,
            "high_severity_rows": 0,
            "high_severity_min_rank": high_rank_default,
            "dashboard_flag_rows": 0,
            "dashboard_citation_flag_min_rank": dash_rank_default,
            "last_survey_date": None,
            "survey_dates_iso": [],
            "max_processing_date": None,
            "pbj_work_date_range": {"min": pbj_min, "max": pbj_max},
            "survey_quarter_note": survey_q_note,
            "nh_health_citations_cms": cms_cit,
        }
    d = enrich_citations_dataframe(df, ccn)
    last = None
    if "_survey_ts" in d.columns and d["_survey_ts"].notna().any():
        last = iso_date_or_none(d["_survey_ts"].max())
    dates = sorted(
        {x for x in d["survey_date_iso"].dropna().unique().tolist() if x},
        reverse=True,
    )
    complaint_n = int(d["complaint_deficiency_bool"].fillna(False).sum()) if "complaint_deficiency_bool" in d.columns else 0
    high_n = int(d["is_high_severity"].fillna(False).sum()) if "is_high_severity" in d.columns else 0
    dash_n = int(d["is_dashboard_citation_flag"].fillna(False).sum()) if "is_dashboard_citation_flag" in d.columns else 0
    proc_dates = [x for x in d["processing_date_iso"].dropna().tolist() if x] if "processing_date_iso" in d.columns else []
    max_proc = max(proc_dates) if proc_dates else None
    return {
        "total": int(len(d)),
        "complaint_related_rows": complaint_n,
        "high_severity_rows": high_n,
        "high_severity_min_rank": high_rank_default,
        "dashboard_flag_rows": dash_n,
        "dashboard_citation_flag_min_rank": dash_rank_default,
        "last_survey_date": last,
        "survey_dates_iso": dates[:120],
        "max_processing_date": max_proc,
        "pbj_work_date_range": {"min": pbj_min, "max": pbj_max},
        "survey_quarter_note": survey_q_note,
        "nh_health_citations_cms": cms_cit,
    }


def citation_ccn_bucket(raw: object) -> str:
    """Normalize CCN from NH Health Citations cells; use as join key to aggregated counts."""
    return _citation_ccn_bucket(raw)


def _citation_ccn_bucket(raw: object) -> str:
    """Normalize CCN cell from NH Health Citations for aggregation keys."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    s = str(raw).strip()
    if not s or s.lower() == "nan":
        return ""
    if validate_ccn(s):
        return normalize_ccn(s)
    digits = re.sub(r"\D", "", s)
    if digits.isdigit() and len(digits) <= 6:
        return digits.zfill(6)
    return ""


def aggregate_dashboard_citation_row_counts_by_ccn(
    source_csv: str,
    severity_cfg: Optional[dict[str, Any]] = None,
) -> tuple[dict[str, int], dict[str, Any]]:
    """
    Scan a national NH_HealthCitations CSV in chunks. For each CCN, count rows whose
    resolved severity rank is >= ``dashboard_citation_flag_min_rank`` (default 60 = G and above).

    Returns ``(ccn -> row_count, meta)``; meta includes ``available``, ``source_path``,
    ``rows_scanned``, ``dashboard_citation_flag_min_rank``.
    """
    path = os.path.abspath(source_csv)
    meta: dict[str, Any] = {
        "available": False,
        "source_path": path,
        "source_basename": os.path.basename(path),
        "rows_scanned": 0,
        "dashboard_citation_flag_min_rank": 60,
        "ccns_with_any_dashboard_flag": 0,
    }
    if not path or not os.path.isfile(path):
        return {}, meta

    cfg = severity_cfg or load_citation_severity_config()
    ranks_raw = cfg.get("harm_rank_by_scope_severity_code", {})
    ranks: dict[str, int] = {str(k).strip().upper(): int(v) for k, v in ranks_raw.items() if str(k).strip()}
    dash_min = int(cfg.get("dashboard_citation_flag_min_rank", 60))
    default_rank = cfg.get("default_rank_for_unknown_code")
    meta["dashboard_citation_flag_min_rank"] = dash_min

    def rank_for_severity_cell(val: object) -> Optional[int]:
        code = _severity_code_raw(val)
        if not code:
            if default_rank is not None:
                try:
                    return int(default_rank)
                except (TypeError, ValueError):
                    return None
            return None
        ch = code[0].upper() if code[0].isalpha() else ""
        v = ranks.get(ch)
        if v is not None:
            return int(v)
        if default_rank is not None:
            try:
                return int(default_rank)
            except (TypeError, ValueError):
                return None
        return None

    cnt: Counter[str] = Counter()
    rows_scanned = 0
    for chunk in pd.read_csv(path, dtype=str, low_memory=False, chunksize=200_000):
        chunk.columns = [str(c).strip().replace("\ufeff", "") for c in chunk.columns]
        if CCN_COL not in chunk.columns or "Scope Severity Code" not in chunk.columns:
            continue
        rows_scanned += len(chunk)
        rnk = chunk["Scope Severity Code"].map(rank_for_severity_cell)
        mask = rnk.notna() & (rnk >= dash_min)
        if not bool(mask.any()):
            continue
        sub = chunk.loc[mask, CCN_COL]
        for raw in sub.tolist():
            k = _citation_ccn_bucket(raw)
            if k:
                cnt[k] += 1

    meta["available"] = True
    meta["rows_scanned"] = int(rows_scanned)
    meta["ccns_with_any_dashboard_flag"] = int(len(cnt))
    return dict(cnt), meta


# --- Citation linking (normalized records, filters, facility summaries) ---

_ABUSE_CATEGORY_PHRASE = "freedom from abuse, neglect, and exploitation"
_ABUSE_TEXT_RE = re.compile(
    r"\b(abuse|neglect|exploitation)\b",
    re.IGNORECASE,
)


def format_survey_date(date: Any) -> tuple[Optional[str], Optional[str]]:
    """
    Return ``(iso YYYY-MM-DD, display MM/DD/YYYY)`` or ``(None, None)`` when invalid.
    """
    if date is None or (isinstance(date, float) and pd.isna(date)):
        return None, None
    if hasattr(date, "strftime") and not isinstance(date, str):
        iso = iso_date_or_none(pd.Timestamp(date))
    else:
        ts = parse_citation_date_series(pd.Series([date]))
        iso = iso_date_or_none(ts.iloc[0]) if len(ts) else None
    if not iso:
        return None, None
    parts = iso.split("-")
    if len(parts) != 3:
        return iso, None
    y, mo, d = parts
    try:
        display = f"{int(mo):02d}/{int(d):02d}/{int(y)}"
    except ValueError:
        display = None
    return iso, display


def format_deficiency_tag(prefix: Any, tag_number: Any) -> str:
    """CMS-style tag label, e.g. ``F-0600``."""
    p = str(prefix or "F").strip().upper().rstrip("-")
    if not p:
        p = "F"
    digits = re.sub(r"\D", "", str(tag_number or ""))
    if not digits:
        return p
    return f"{p}-{digits.zfill(4)}"


def build_cms_filtered_deficiency_url(ccn: str) -> Optional[str]:
    return cms_nh_health_citations_dataset_explorer_url(ccn)


def build_medicare_health_summary_url(ccn: str) -> Optional[str]:
    return medicare_nursing_home_inspections_landing_url(ccn)


def build_complaint_inspection_pdf_url(ccn: str, survey_date: str) -> Optional[str]:
    return medicare_nursing_home_complaint_inspection_pdf_url(ccn, survey_date)


def build_health_inspection_pdf_url(ccn: str, survey_date: str) -> Optional[str]:
    return medicare_nursing_home_health_inspection_pdf_url(ccn, survey_date)


def build_infection_control_inspection_pdf_url(ccn: str, survey_date: str) -> Optional[str]:
    return medicare_nursing_home_infection_control_inspection_pdf_url(ccn, survey_date)


def build_medicare_inspection_pdf_url(ccn: str, survey_date: str, inspection_kind: str) -> Optional[str]:
    return medicare_nursing_home_inspection_pdf_url(ccn, survey_date, inspection_kind)


def resolve_survey_medicare_inspection_pdf_kind(survey_rows: pd.DataFrame) -> Optional[str]:
    """
    Pick the Care Compare PDF path segment for one survey (all rows sharing survey date).

    Priority: infection control > complaint (only when every row on the survey is complaint=Y
    or Survey Type names complaint) > standard health inspection.
    """
    if survey_rows is None or survey_rows.empty:
        return None
    if "Infection Control Inspection Deficiency" in survey_rows.columns:
        ic = survey_rows["Infection Control Inspection Deficiency"].map(_yn_to_bool)
        if bool(ic.eq(True).any()):
            return "infection_control"
    if "infection_control_deficiency_bool" in survey_rows.columns:
        if bool(survey_rows["infection_control_deficiency_bool"].eq(True).any()):
            return "infection_control"
    if "Complaint Deficiency" in survey_rows.columns:
        comp = survey_rows["Complaint Deficiency"].map(_yn_to_bool)
        if bool(comp.eq(True).any()) and bool(comp.eq(True).all()):
            return "complaint"
    if "complaint_deficiency_bool" in survey_rows.columns:
        compb = survey_rows["complaint_deficiency_bool"]
        if bool(compb.eq(True).any()) and bool(compb.fillna(False).eq(True).all()):
            return "complaint"
    if "Survey Type" in survey_rows.columns:
        st = " ".join(str(x or "") for x in survey_rows["Survey Type"].dropna().unique()).lower()
        if "complaint" in st and "health" not in st:
            return "complaint"
    return "health"


def _survey_pdf_lookup(df: pd.DataFrame, ccn: str) -> dict[str, tuple[str, Optional[str]]]:
    """Map survey_date_iso -> (kind, pdf_url) for all surveys in a citations frame."""
    ccn_ok = validate_ccn(str(ccn).strip())
    ccn_norm = normalize_ccn(str(ccn).strip()) if ccn_ok else ""
    if not ccn_ok or df.empty:
        return {}
    work = df if "survey_date_iso" in df.columns else enrich_citations_dataframe(df, ccn)
    out: dict[str, tuple[str, Optional[str]]] = {}
    for siso, grp in work.groupby("survey_date_iso", dropna=True):
        if not isinstance(siso, str) or not siso:
            continue
        kind = resolve_survey_medicare_inspection_pdf_kind(grp)
        url = medicare_nursing_home_inspection_pdf_url(ccn_norm, siso, kind) if kind else None
        out[siso] = (kind, url)
    return out


def _row_cell(row: Any, *names: str) -> Any:
    if isinstance(row, dict):
        for n in names:
            if n in row and row[n] is not None and not (isinstance(row[n], float) and pd.isna(row[n])):
                return row[n]
        return None
    for n in names:
        if hasattr(row, "index") and n in row.index:
            v = row[n]
            if pd.notna(v):
                return v
    return None


def normalize_citation_record(
    row: Any,
    ccn: str,
    *,
    survey_pdf: Optional[tuple[str, Optional[str]]] = None,
) -> dict[str, Any]:
    """Normalized citation object for APIs and UI (one NH Health Deficiency row)."""
    ccn_norm = normalize_ccn(str(ccn).strip()) if validate_ccn(str(ccn).strip()) else str(ccn).strip()
    provider_name = str(_row_cell(row, "Provider Name", "provider_name") or "").strip()
    survey_raw = _row_cell(row, "Survey Date", "survey_date")
    survey_iso, survey_display = format_survey_date(survey_raw)
    correction_raw = _row_cell(row, "Correction Date", "correction_date")
    correction_iso, _ = format_survey_date(correction_raw)

    prefix = str(_row_cell(row, "Deficiency Prefix", "deficiency_prefix") or "F").strip()
    tag_num = _row_cell(row, "Deficiency Tag Number", "deficiency_tag_number")
    tag = format_deficiency_tag(prefix, tag_num)

    scope = _severity_code_raw(_row_cell(row, "Scope Severity Code", "scope_severity_code"))

    cycle_raw = _row_cell(row, "Inspection Cycle", "inspection_cycle")
    inspection_cycle: Optional[int] = None
    if cycle_raw is not None and not (isinstance(cycle_raw, float) and pd.isna(cycle_raw)):
        try:
            inspection_cycle = int(float(str(cycle_raw).strip()))
        except (ValueError, TypeError):
            inspection_cycle = None

    is_complaint = _yn_to_bool(_row_cell(row, "Complaint Deficiency", "complaint_deficiency")) is True
    pdf_kind: Optional[str] = None
    pdf_url: Optional[str] = None
    if survey_pdf:
        pdf_kind, pdf_url = survey_pdf
    elif survey_iso:
        row_kind = "complaint" if is_complaint else "health"
        pdf_kind = row_kind
        pdf_url = build_medicare_inspection_pdf_url(ccn_norm, survey_iso, row_kind)

    health_summary = build_medicare_health_summary_url(ccn_norm)

    return {
        "ccn": ccn_norm,
        "provider_name": provider_name,
        "survey_date": survey_iso,
        "survey_date_display": survey_display,
        "survey_type": str(_row_cell(row, "Survey Type", "survey_type") or "").strip(),
        "deficiency_prefix": prefix.upper().rstrip("-") or "F",
        "deficiency_tag_number": str(tag_num).strip() if tag_num is not None and not (isinstance(tag_num, float) and pd.isna(tag_num)) else "",
        "deficiency_tag": tag,
        "deficiency_category": str(_row_cell(row, "Deficiency Category", "deficiency_category") or "").strip(),
        "deficiency_description": str(_row_cell(row, "Deficiency Description", "deficiency_description") or "").strip(),
        "scope_severity_code": scope or "",
        "deficiency_corrected": str(_row_cell(row, "Deficiency Corrected", "deficiency_corrected") or "").strip(),
        "correction_date": correction_iso,
        "inspection_cycle": inspection_cycle,
        "is_standard_deficiency": _yn_to_bool(_row_cell(row, "Standard Deficiency", "standard_deficiency")) is True,
        "is_complaint_deficiency": is_complaint,
        "is_infection_control_deficiency": _yn_to_bool(
            _row_cell(row, "Infection Control Inspection Deficiency", "infection_control_deficiency")
        )
        is True,
        "is_under_idr": _yn_to_bool(_row_cell(row, "Citation under IDR", "citation_under_idr")) is True,
        "is_under_iidr": _yn_to_bool(_row_cell(row, "Citation under IIDR", "citation_under_iidr")) is True,
        "cms_filtered_dataset_url": build_cms_filtered_deficiency_url(ccn_norm),
        "medicare_health_summary_url": health_summary,
        "medicare_inspection_pdf_kind": pdf_kind,
        "medicare_inspection_pdf_url": pdf_url,
        "medicare_inspection_page_url": pdf_url or health_summary,
    }


def _citations_dataframe_for_ccn(df: pd.DataFrame, ccn: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d.columns = [str(c).strip().replace("\ufeff", "") for c in d.columns]
    if CCN_COL not in d.columns:
        return d
    ccn_canon = normalize_ccn(str(ccn).strip()) if validate_ccn(str(ccn).strip()) else ""
    if not ccn_canon:
        return d.iloc[0:0]
    variants = {ccn_canon, ccn_canon.lstrip("0") or "0", ccn_canon.upper()}
    s = d[CCN_COL].astype(str).str.strip()
    s = s.apply(lambda x: x.zfill(6) if x.isdigit() else x.upper())
    return d.loc[s.isin(variants)].copy()


def get_facility_citations(ccn: str, citations_df: pd.DataFrame) -> list[dict[str, Any]]:
    """All normalized citations for a CCN (newest survey first)."""
    sub = _citations_dataframe_for_ccn(citations_df, ccn)
    if sub.empty:
        return []
    if "survey_date_iso" not in sub.columns:
        sub = enrich_citations_dataframe(sub, ccn)
    survey_pdf = _survey_pdf_lookup(sub, ccn)
    if "_survey_ts" in sub.columns:
        sub = sub.sort_values("_survey_ts", ascending=False, na_position="last")
    rows: list[dict[str, Any]] = []
    for _, r in sub.iterrows():
        siso = r.get("survey_date_iso")
        sp = survey_pdf.get(siso) if isinstance(siso, str) else None
        rows.append(normalize_citation_record(r, ccn, survey_pdf=sp))
    return rows


def get_citations_by_category(ccn: str, category: str, citations_df: pd.DataFrame) -> list[dict[str, Any]]:
    needle = (category or "").strip().lower()
    return [
        c
        for c in get_facility_citations(ccn, citations_df)
        if needle and needle in (c.get("deficiency_category") or "").lower()
    ]


def get_citations_by_tag(ccn: str, tag: str, citations_df: pd.DataFrame) -> list[dict[str, Any]]:
    raw = str(tag or "").strip().upper()
    if raw.startswith("F-"):
        want = raw
    elif raw.startswith("F") and len(raw) > 1:
        want = format_deficiency_tag("F", raw[1:])
    else:
        want = format_deficiency_tag("F", raw)
    return [c for c in get_facility_citations(ccn, citations_df) if (c.get("deficiency_tag") or "").upper() == want]


def _severity_rank_for_code(code: str, cfg: Optional[dict[str, Any]] = None) -> int:
    cfg = cfg or load_citation_severity_config()
    ranks: dict[str, Any] = dict(cfg.get("harm_rank_by_scope_severity_code", {}))
    default_rank = cfg.get("default_rank_for_unknown_code")
    ch = (code or "").strip().upper()[:1]
    if ch and ch in ranks:
        return int(ranks[ch])
    if default_rank is not None:
        return int(default_rank)
    return -1


def get_severe_citations(
    ccn: str,
    citations_df: pd.DataFrame,
    minimum_scope_severity: str = "G",
) -> list[dict[str, Any]]:
    """Citations at or above ``minimum_scope_severity`` letter (A–L scale)."""
    floor = (minimum_scope_severity or "G").strip().upper()[:1]
    if not floor.isalpha():
        floor = "G"
    cfg = load_citation_severity_config()
    floor_rank = _severity_rank_for_code(floor, cfg)
    out: list[dict[str, Any]] = []
    for c in get_facility_citations(ccn, citations_df):
        rk = _severity_rank_for_code(str(c.get("scope_severity_code") or ""), cfg)
        if rk >= floor_rank:
            out.append(c)
    return out


def is_abuse_related_citation_record(rec: dict[str, Any]) -> bool:
    """True when citation matches the config-driven ``abuse_neglect`` topic pack."""
    try:
        from citation_taxonomy import match_citation_topics

        return "abuse_neglect" in match_citation_topics(rec, active_only=True)
    except Exception:
        cat = (rec.get("deficiency_category") or "").lower()
        if _ABUSE_CATEGORY_PHRASE in cat:
            return True
        tag = str(rec.get("deficiency_tag") or "").upper()
        if tag == "F-0689":
            return False
        if re.match(r"^F-?0600", tag):
            return True
        blob = " ".join(
            [
                rec.get("deficiency_category") or "",
                rec.get("deficiency_description") or "",
                rec.get("deficiency_tag") or "",
            ]
        )
        return bool(_ABUSE_TEXT_RE.search(blob))


def get_abuse_related_citations(ccn: str, citations_df: pd.DataFrame) -> list[dict[str, Any]]:
    return [c for c in get_facility_citations(ccn, citations_df) if is_abuse_related_citation_record(c)]


def get_infection_control_citations(ccn: str, citations_df: pd.DataFrame) -> list[dict[str, Any]]:
    return [c for c in get_facility_citations(ccn, citations_df) if c.get("is_infection_control_deficiency")]


def get_complaint_citations(ccn: str, citations_df: pd.DataFrame) -> list[dict[str, Any]]:
    return [c for c in get_facility_citations(ccn, citations_df) if c.get("is_complaint_deficiency")]


def _abuse_citation_sort_key(rec: dict[str, Any]) -> tuple:
    cycle = rec.get("inspection_cycle")
    cycle_pri = 0 if cycle == 1 else 1
    cfg = load_citation_severity_config()
    sev_pri = -_severity_rank_for_code(str(rec.get("scope_severity_code") or ""), cfg)
    complaint_pri = 0 if rec.get("is_complaint_deficiency") else 1
    survey = rec.get("survey_date") or ""
    return (cycle_pri, complaint_pri, sev_pri, survey)


def rank_abuse_related_citations(citations: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(citations, key=_abuse_citation_sort_key)
    return ranked[: max(0, int(limit))]


def _enrich_citations_with_pdf_extract(ccn: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """When a cached 2567 PDF exists, attach parsed tag-block signals to matching citations."""
    if not citations:
        return citations
    try:
        from citation_pdf_extract import load_or_parse_cached_pdf, merge_pdf_extract_into_citation
    except Exception:
        return citations
    extract_cache: dict[tuple[str, str], Any] = {}
    out: list[dict[str, Any]] = []
    for c in citations:
        sd = c.get("survey_date")
        kind = str(c.get("medicare_inspection_pdf_kind") or "complaint")
        if not sd:
            out.append(c)
            continue
        key = (str(sd), kind)
        if key not in extract_cache:
            extract_cache[key] = load_or_parse_cached_pdf(ccn, str(sd), kind)
        ex = extract_cache[key]
        out.append(merge_pdf_extract_into_citation(c, ex) if ex else c)
    return out


def provider_info_abuse_icon_y(provider_info_df: Optional[pd.DataFrame], ccn: str) -> bool:
    if provider_info_df is None or provider_info_df.empty:
        return False
    sub = provider_info_df.copy()
    if "ccn" in sub.columns:
        ccn_col = "ccn"
    elif "PROVNUM" in sub.columns:
        ccn_col = "PROVNUM"
    else:
        ccn_col = None
    if ccn_col:
        ccn_canon = normalize_ccn(str(ccn).strip()) if validate_ccn(str(ccn).strip()) else ""
        if ccn_canon:
            s = sub[ccn_col].astype(str).str.strip().str.zfill(6)
            sub = sub.loc[s == ccn_canon]
    abuse_col = None
    for col in ("abuse_icon", "Abuse Icon", "abuse"):
        if col in sub.columns:
            abuse_col = col
            break
    if not abuse_col:
        return False
    for val in sub[abuse_col].tolist():
        if str(val).strip().upper() in ("Y", "YES", "TRUE", "1"):
            return True
    return False


def build_facility_citation_summary(ccn: str, citations_df: pd.DataFrame) -> dict[str, Any]:
    """Per-facility citation rollup for reports, owner views, and future UI."""
    ccn_norm = normalize_ccn(str(ccn).strip()) if validate_ccn(str(ccn).strip()) else str(ccn).strip()
    all_c = get_facility_citations(ccn_norm, citations_df)
    cfg = load_citation_severity_config()
    dash_min = int(cfg.get("dashboard_citation_flag_min_rank", 60))
    severe = [c for c in all_c if _severity_rank_for_code(str(c.get("scope_severity_code") or ""), cfg) >= dash_min]
    complaint = get_complaint_citations(ccn_norm, citations_df)
    ic = get_infection_control_citations(ccn_norm, citations_df)
    abuse = get_abuse_related_citations(ccn_norm, citations_df)
    dates = [c["survey_date"] for c in all_c if c.get("survey_date")]
    most_recent = max(dates) if dates else None
    sev_codes = [str(c.get("scope_severity_code") or "").upper()[:1] for c in all_c if c.get("scope_severity_code")]
    most_severe = None
    if sev_codes:
        most_severe = max(sev_codes, key=lambda ch: _severity_rank_for_code(ch, cfg))
    by_cat: Counter[str] = Counter()
    by_tag: Counter[str] = Counter()
    for c in all_c:
        cat = (c.get("deficiency_category") or "").strip()
        if cat:
            by_cat[cat] += 1
        tag = (c.get("deficiency_tag") or "").strip()
        if tag:
            by_tag[tag] += 1
    return {
        "ccn": ccn_norm,
        "total_health_citations": len(all_c),
        "severe_citations_count": len(severe),
        "complaint_citations_count": len(complaint),
        "infection_control_citations_count": len(ic),
        "abuse_related_citations_count": len(abuse),
        "most_recent_citation_date": most_recent,
        "most_severe_scope_severity": most_severe,
        "citations_by_category": dict(by_cat),
        "citations_by_tag": dict(by_tag),
        "source_links": {
            "cms_filtered_dataset_url": build_cms_filtered_deficiency_url(ccn_norm),
            "medicare_health_summary_url": build_medicare_health_summary_url(ccn_norm),
        },
    }


def build_abuse_flag_context(
    ccn: str,
    citations_df: Optional[pd.DataFrame],
    provider_info_df: Optional[pd.DataFrame],
    *,
    max_citations: int = 10,
) -> dict[str, Any]:
    """
    Payload for the Abuse icon popover: CMS-attributed copy, top citations, and source links.
    """
    ccn_norm = normalize_ccn(str(ccn).strip()) if validate_ccn(str(ccn).strip()) else str(ccn).strip()
    show = provider_info_abuse_icon_y(provider_info_df, ccn_norm)
    cit_df = citations_df if citations_df is not None else pd.DataFrame()
    top: list[dict[str, Any]] = []
    if not cit_df.empty:
        try:
            from citation_taxonomy import get_facility_citation_slices

            slices = get_facility_citation_slices(
                ccn_norm,
                cit_df,
                topic_ids=["abuse_neglect"],
                limit=max_citations,
                enrich_pdf=True,
            )
            top = slices.get("abuse_neglect") or []
        except Exception:
            abuse_cits = get_abuse_related_citations(ccn_norm, cit_df)
            top = rank_abuse_related_citations(abuse_cits, limit=max_citations)
            top = _enrich_citations_with_pdf_extract(ccn_norm, top)
    cms_url = build_cms_filtered_deficiency_url(ccn_norm)
    medicare_url = build_medicare_health_summary_url(ccn_norm)
    return {
        "show_abuse_flag": show,
        "title": "Abuse flag",
        "body": "CMS has flagged this facility for abuse-related deficiencies.",
        "top_citations": top,
        "citation_details_incomplete": show and not top,
        "citation_fallback_note": (
            "Citation-level details are available in the CMS Health Deficiencies dataset."
            if show and not top
            else None
        ),
        "source_links": {
            "cms_filtered_dataset_url": cms_url,
            "medicare_health_summary_url": medicare_url,
        },
    }


def abuse_flag_context_for_api(
    ccn: str,
    citations_df: Optional[pd.DataFrame],
    provider_info_df: Optional[pd.DataFrame],
) -> Optional[dict[str, Any]]:
    """Return abuse popover payload only when Provider Information Abuse Icon = Y."""
    ctx = build_abuse_flag_context(ccn, citations_df, provider_info_df)
    if not ctx.get("show_abuse_flag"):
        return None
    return ctx


