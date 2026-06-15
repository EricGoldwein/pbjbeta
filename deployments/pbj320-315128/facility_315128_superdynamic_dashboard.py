#!/usr/bin/env python3
"""
Dynamic Facility Dashboard
Uses the complete CSV file for any facility for fast, detailed analysis
"""

import os
import re
import hashlib
import secrets
import sys
import threading
import time

# When this file is run as deployments/pbj320-<CCN>/facility_<CCN>_flask_app.py, Python puts that
# folder first on sys.path and may load stale copies of shared modules. Prefer the repo root.
_this_dir = os.path.dirname(os.path.abspath(__file__))
_path_parts = os.path.normpath(_this_dir).split(os.sep)
if (
    len(_path_parts) >= 2
    and _path_parts[-2].lower() == "deployments"
    and re.match(r"^pbj320-\d{6}$", _path_parts[-1], flags=re.I)
):
    _repo_root = os.path.abspath(os.path.join(_this_dir, "..", ".."))
    _marker = os.path.join(_repo_root, "facility_ein_employee_analytics.py")
    if os.path.isfile(_marker) and _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
    # Keep deployment-local modules (e.g., facility_report_lib.py) highest priority.
    if _this_dir in sys.path:
        sys.path = [p for p in sys.path if p != _this_dir]
    sys.path.insert(0, _this_dir)

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file, redirect, url_for, Response, abort
from flask.typing import ResponseReturnValue
from datetime import datetime, timedelta
import copy
import html
import json
from difflib import SequenceMatcher
from decimal import Decimal, ROUND_HALF_UP
import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast
from urllib.parse import quote

from pbj_identifiers.urls import (
    PBJ_RANKINGS_REPORT_URL,
    cms_nh_health_citations_dataset_explorer_url,
    generate_state_dashboard_url,
)
from pbj_identifiers.validators import validate_state_code
from pbj_case_mix_cmi import (
    coalesce_provider_quarter_snapshots,
    extract_nursing_cmi_from_provider_series,
    narrow_provider_quarter_rows_for_case_mix,
    normalize_provider_info_csv_columns,
)
from pbj_premium_demo_section import pbj_premium_dashboard_offer_context
from pbj_chow_facility import chow_facility_api_payload, format_chow_date
from facility_deploy_meta import read_deployed_date_display
from pbj_ai_config import (
    pbj_ai_skill_zip_facility_enabled,
    pbj_claude_skill_zip_paths,
)

from pbj_facility_display_name import get_facility_name_for_context

import facility_report_lib
from facility_report_lib import (
    calculate_harrington_adjusted_hprd,
    calculate_harrington_residual_lpn_hprd,
)
from geo_distribution_lib import (
    build_geo_distribution_context,
    build_geo_distribution_payload,
    supplement_facility_lite_peer_df,
)
from facility_ein_lib import (
    CMS_EIN_DETAIL_LANDING_URL,
    JOB_CODE_INFO,
    JOB_TITLE_SHORT,
    NONNURSE_PBJ_EIN_JOB_CODES,
    PBJ_BRIDGE_COMPARE_CAVEATS,
    PBJ_METRIC_TO_DAILY_COLUMNS,
    PBJ_METRIC_TO_EIN_JOB_CODES,
    PBJ_NONNURSE_HRS_COLUMN_TO_EIN_JOB_CODES,
    build_cms_ein_detail_explorer_url,
    compare_pbj_vs_ein_hours,
    ein_employee_detail_sources_available,
    format_ein_job_codes_for_ui,
    get_pbj_complete_data_row_for_date,
    iso_date_to_cy_quarter,
    job_title,
    normalize_cy_qtr_ein,
    parse_ein_quarter_bound,
    pbj_metric_display_name,
    read_facility_ein_parquet_or_csv,
    run_cms_mapping_integrity_report,
    sum_pbj_hours_for_bridge_metric,
)
import facility_ein_employee_analytics as _fea


def _fea_get(name: str, fallback):
    return getattr(_fea, name, fallback)


def _fea_empty_rows(*args, **kwargs):
    return []


def _fea_empty_dict(*args, **kwargs):
    return {}


def _fea_passthrough_rows(rows, *args, **kwargs):
    return rows


aggregate_ein_day_by_job_codes = _fea_get("aggregate_ein_day_by_job_codes", lambda *a, **k: ([], 0.0, None))
apply_roster_tenure_quarter_span = _fea_get("apply_roster_tenure_quarter_span", _fea_passthrough_rows)
ein_employee_day_cms_hours_cap_summary = _fea_get("ein_employee_day_cms_hours_cap_summary", _fea_empty_dict)
ein_span_days_from_workdate_raws = _fea_get("ein_span_days_from_workdate_raws", lambda *a, **k: None)
ein_daily_metric_headcounts_range = _fea_get(
    "ein_daily_metric_headcounts_range",
    lambda *a, **k: {"rows": [], "meta": {}},
)
format_ein_tenure_span_days = _fea_get("format_ein_tenure_span_days", lambda *a, **k: "")
nursing_employee_quarters_for_job = _fea_get("nursing_employee_quarters_for_job", _fea_empty_rows)
EIN_NURSING_ROSTER_BRIDGE_NOTE = _fea_get(
    "EIN_NURSING_ROSTER_BRIDGE_NOTE",
    "EIN roster bridge unavailable in this build.",
)
EIN_NURSING_ROSTER_CODES_LABEL = _fea_get("EIN_NURSING_ROSTER_CODES_LABEL", "Codes unavailable")
compute_ein_quarter_roster_summary = _fea_get("compute_ein_quarter_roster_summary", _fea_empty_dict)
compute_ein_quarter_nonnurse_roster_summary = _fea_get(
    "compute_ein_quarter_nonnurse_roster_summary", _fea_empty_dict
)
dedupe_nursing_roster_api_rows = _fea_get("dedupe_nursing_roster_api_rows", _fea_passthrough_rows)
ein_job_code_matches_position_group = _fea_get("ein_job_code_matches_position_group", lambda *a, **k: False)
ein_nonnurse_job_code_matches_position_group = _fea_get(
    "ein_nonnurse_job_code_matches_position_group", lambda *a, **k: False
)
enrich_nonnurse_rows_rolling_from_work_date = _fea_get(
    "enrich_nonnurse_rows_rolling_from_work_date", _fea_passthrough_rows
)
nonnurse_employee_summaries = _fea_get("nonnurse_employee_summaries", _fea_empty_rows)
roster_pairs_by_quarter_for_job_codes = _fea_get("roster_pairs_by_quarter_for_job_codes", _fea_empty_dict)
ein_nursing_roster_for_work_date = _fea_get("ein_nursing_roster_for_work_date", _fea_empty_rows)
ein_headcount_buckets_for_work_date = _fea_get("ein_headcount_buckets_for_work_date", _fea_empty_dict)
ein_headcount_by_job_longitudinal_series = _fea_get(
    "ein_headcount_by_job_longitudinal_series",
    lambda *a, **k: {"rows": [], "meta": {}},
)
ein_headcount_quarter_series = _fea_get("ein_headcount_quarter_series", _fea_empty_rows)
ein_quarter_pbj_context_from_daily = _fea_get("ein_quarter_pbj_context_from_daily", lambda *a, **k: {})
ein_pbj_bridge_for_metric = _fea_get("ein_pbj_bridge_for_metric", _fea_empty_dict)
enrich_nursing_roster_display_fields = _fea_get("enrich_nursing_roster_display_fields", _fea_passthrough_rows)
enrich_nursing_rows_new_to_quarter_flags = _fea_get(
    "enrich_nursing_rows_new_to_quarter_flags",
    _fea_passthrough_rows,
)
enrich_nursing_rows_multi_role_flags = _fea_get(
    "enrich_nursing_rows_multi_role_flags",
    _fea_passthrough_rows,
)
enrich_nursing_rows_sustained_work_flags = _fea_get(
    "enrich_nursing_rows_sustained_work_flags",
    _fea_passthrough_rows,
)
enrich_nursing_rows_rolling_from_work_date = _fea_get(
    "enrich_nursing_rows_rolling_from_work_date",
    _fea_passthrough_rows,
)
nursing_employee_daily_series = _fea_get("nursing_employee_daily_series", _fea_empty_rows)
nursing_employee_summaries = _fea_get("nursing_employee_summaries", _fea_empty_rows)
prepare_ein_detail = _fea_get("prepare_ein_detail", _fea_empty_rows)
roster_pairs_by_quarter_from_rows = _fea_get("roster_pairs_by_quarter_from_rows", _fea_empty_dict)
workdate_to_iso = _fea_get("workdate_to_iso", lambda *a, **k: None)
from pbj_staffing_normalize import (
    _load_or_build_provnum_chunk_index,
    _normalize_provnum_like,
    coerce_provnum_column,
    invalidate_provnum_chunk_index_cache,
    pandas_chunk_read_memory_error,
    provnum_search_variants,
    select_targeted_chunk_ids,
)


def _superdynamic_v3_panes_enabled() -> bool:
    """Multi-pane V3 shell (Overview / Benchmarks / Workforce / Risk). Opt-in via PBJ_SUPERDYNAMIC_V3_PANES."""
    raw = (os.environ.get("PBJ_SUPERDYNAMIC_V3_PANES") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _superdynamic_dashboard_template_name() -> str:
    """Dashboard Jinja template; default v1. Set PBJ_SUPERDYNAMIC_TEMPLATE=v2 for UI scaffold."""
    raw = (os.environ.get("PBJ_SUPERDYNAMIC_TEMPLATE") or "").strip()
    if not raw:
        return "superdynamic_dashboard.html"
    low = raw.lower().removesuffix(".html")
    if low in ("v2", "superdynamic_v2", "superdynamic_dashboard_v2"):
        return "superdynamic_dashboard_v2.html"
    if low.endswith("_v2") or low.endswith("dashboard_v2"):
        return "superdynamic_dashboard_v2.html"
    if raw.endswith(".html"):
        return raw
    return "superdynamic_dashboard.html"


def _resolve_template_folder() -> str:
    """Prefer local deployment templates, then shared repo template directory."""
    env_override = (os.environ.get("PBJ_TEMPLATE_DIR") or "").strip()
    candidates = []
    if env_override:
        candidates.append(env_override)
    candidates.append(os.path.join(_this_dir, "templates"))
    candidates.append(os.path.abspath(os.path.join(_this_dir, "..", "..", "templates")))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    # Safe fallback: Flask will use this path even if directory is created later.
    return candidates[0]


app = Flask(__name__, template_folder=_resolve_template_folder())
_app_root = os.path.dirname(os.path.abspath(__file__))

# Generated facility_*_flask_app.py may inject ``INCLUDE_NONNURSE = False`` before routes
# (see create_vercel_deployment.py) to skip national non-nurse scans and omit non-nurse UI.
INCLUDE_NONNURSE = True


def _pbj_env_str(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _pbj_normalize_public_api_origin(url: str) -> str:
    return (url or "").strip().rstrip("/")


def _pbj_normalize_public_site_base(path: str) -> str:
    p = (path or "").strip().rstrip("/")
    if not p:
        return ""
    return p if p.startswith("/") else "/" + p


def _pbj_https_origin_from_host_value(raw: str) -> str:
    """Turn VERCEL_* or similar host strings into ``https://host`` (no trailing slash)."""
    h = (raw or "").strip()
    if not h:
        return ""
    h = h.split("/")[0].strip()
    low = h.lower()
    if low.startswith("https://"):
        return h.rstrip("/")
    if low.startswith("http://"):
        return ("https://" + h[7:]).rstrip("/")
    return ("https://" + h).rstrip("/")


def _pbj_inferred_public_api_origin() -> str:
    """On Vercel, use the deployment's own URL as API origin when HTML is viewed elsewhere (e.g. www.pbj320.com).

    No per-CCN ``PUBLIC_API_ORIGIN`` env needed unless you override. Order:
    ``VERCEL_PROJECT_PRODUCTION_URL`` (stable ``*.vercel.app``) then ``VERCEL_URL`` (this deployment).
    Local dev: unset → empty → same-origin ``/api`` only.
    """
    prod = _pbj_https_origin_from_host_value(_pbj_env_str("VERCEL_PROJECT_PRODUCTION_URL"))
    if prod:
        return prod.rstrip("/")
    return _pbj_https_origin_from_host_value(_pbj_env_str("VERCEL_URL")).rstrip("/")


_PBJ_GA4_DEFAULT_MEASUREMENT_ID = "G-NDPVY6TWBK"


def _pbj_resolved_ga_measurement_id() -> str:
    """Same GA4 property as main PBJ320 unless overridden via Vercel env (empty = use default)."""
    return (
        _pbj_env_str("NEXT_PUBLIC_GA_MEASUREMENT_ID")
        or _pbj_env_str("VITE_GA_MEASUREMENT_ID")
        or _pbj_env_str("NEXT_PUBLIC_GA_ID")
        or _pbj_env_str("VITE_GA_ID")
        or _PBJ_GA4_DEFAULT_MEASUREMENT_ID
    )


def _pbj_gtag_head_script_snippet(measurement_id: str) -> str:
    """Standard GA4 init; dedupes gtag.js if already present (matches dashboard template)."""
    if not (measurement_id or "").strip():
        return ""
    mid_js = json.dumps((measurement_id or "").strip())
    return (
        "  <!-- Google tag (gtag.js) — same GA4 property as main PBJ320 -->\n"
        "  <script>\n"
        "        (function (measurementId) {\n"
        "            if (!measurementId) return;\n"
        "            if (window.__pbjGtagConfigured === measurementId) return;\n"
        "            window.__pbjGtagConfigured = measurementId;\n"
        "            var hasGtagJs = false;\n"
        "            try {\n"
        "                document.querySelectorAll('script[src]').forEach(function (el) {\n"
        "                    if (el.src.indexOf('googletagmanager.com/gtag/js') !== -1) hasGtagJs = true;\n"
        "                });\n"
        "            } catch (e) { /* ignore */ }\n"
        "            if (!hasGtagJs) {\n"
        "                var s = document.createElement('script');\n"
        "                s.async = true;\n"
        "                s.src = 'https://www.googletagmanager.com/gtag/js?id=' + encodeURIComponent(measurementId);\n"
        "                document.head.appendChild(s);\n"
        "            }\n"
        "            window.dataLayer = window.dataLayer || [];\n"
        "            window.gtag = window.gtag || function () { window.dataLayer.push(arguments); };\n"
        "            gtag('js', new Date());\n"
        "            gtag('config', measurementId);\n"
        f"        }})({mid_js});\n"
        "  </script>\n"
    )


def _ein_employee_detail_workdate_bounds_iso() -> dict[str, Any]:
    """Min/max work dates for roster link gating in the UI (ISO).

    Row-level detail may load on demand when nursing summaries are precomputed; in that case
    bounds come from summaries' ``first_work_date`` / ``last_work_date`` so daily-table Roster
    pills still render on first page load.
    """
    out: dict[str, Any] = {"available": False, "min_iso": None, "max_iso": None}
    global ein_employee_detail_df, ein_nursing_summaries_df

    df = ein_employee_detail_df
    if df is not None and not df.empty and "WorkDate" in df.columns:
        col = df["WorkDate"].dropna()
        if not col.empty:
            try:
                if pd.api.types.is_datetime64_any_dtype(col):
                    lo_ts = cast(Any, col.min())
                    hi_ts = cast(Any, col.max())
                    if not (pd.isna(lo_ts) or pd.isna(hi_ts)):
                        lo = pd.Timestamp(lo_ts).strftime("%Y-%m-%d")
                        hi = pd.Timestamp(hi_ts).strftime("%Y-%m-%d")
                        if lo and hi:
                            out["available"] = True
                            out["min_iso"] = lo
                            out["max_iso"] = hi
                            return out
                else:
                    lo_raw = col.min()
                    hi_raw = col.max()
                    lo = workdate_to_iso(int(float(lo_raw)))
                    hi = workdate_to_iso(int(float(hi_raw)))
                    if lo and hi:
                        out["available"] = True
                        out["min_iso"] = lo
                        out["max_iso"] = hi
                        return out
            except (TypeError, ValueError, OSError):
                pass

    summ = ein_nursing_summaries_df
    if summ is not None and not summ.empty:
        try:
            lo_iso = hi_iso = None
            if "first_work_date" in summ.columns:
                fd = pd.to_datetime(summ["first_work_date"], errors="coerce")
                if fd.notna().any():
                    lo_iso = pd.Timestamp(cast(Any, fd.min())).strftime("%Y-%m-%d")
            if "last_work_date" in summ.columns:
                ld = pd.to_datetime(summ["last_work_date"], errors="coerce")
                if ld.notna().any():
                    hi_iso = pd.Timestamp(cast(Any, ld.max())).strftime("%Y-%m-%d")
            if lo_iso and hi_iso:
                out["available"] = True
                out["min_iso"] = lo_iso
                out["max_iso"] = hi_iso
        except (TypeError, ValueError, OSError):
            return out
    return out


def _pbj_claude_skill_template_context() -> dict[str, Any]:
    """Jinja flags for the AI Toolkit Claude Skill download (facility origin only)."""
    enabled = pbj_ai_skill_zip_facility_enabled(_app_root)
    url = "/downloads/pbj320-staffing-review.zip" if enabled else ""
    return {
        "pbj_claude_skill_zip_enabled": enabled,
        "pbj_claude_skill_zip_url": url,
        "pbj_claude_skill_zip_download_href": _pbj_facility_download_href(url) if url else "",
    }


def _pbj_template_client_config() -> dict[str, str]:
    """Jinja context for `pbjApiUrl` / `pbj_api_href` (see templates/superdynamic_dashboard.html).

    For local Flask, leave ``PUBLIC_API_ORIGIN`` unset so the browser uses same-origin ``/api/...``.
    Setting it to ``http://127.0.0.1:5000`` while running the app on another port causes
    ``ERR_CONNECTION_REFUSED`` for provider info / SFF fetches (the template also corrects
    loopback port mismatches when possible).
    """
    mid = _pbj_resolved_ga_measurement_id()
    explicit_api = _pbj_normalize_public_api_origin(_pbj_env_str("PUBLIC_API_ORIGIN"))
    inferred_api = _pbj_inferred_public_api_origin()
    marketing_origin = _pbj_env_str("PBJ320_MARKETING_ORIGIN").strip().rstrip("/") or "https://www.pbj320.com"
    return {
        "pbj_public_api_origin": explicit_api or inferred_api,
        "pbj_public_site_base_path": _pbj_normalize_public_site_base(_pbj_env_str("PUBLIC_SITE_BASE_PATH")),
        "pbj320_marketing_origin": marketing_origin,
        "pbj_analytics_ga_id": mid,
        "pbj_analytics_plausible_domain": _pbj_env_str("PUBLIC_PLAUSIBLE_DOMAIN")
        or _pbj_env_str("VITE_PLAUSIBLE_DOMAIN"),
    }


def _pbj_facility_download_href(path: str) -> str:
    """Same-app downloads: relative on HTTP/dev; HTTPS origin prefix in production."""
    p = path if path.startswith("/") else "/" + path
    o = _pbj_normalize_public_api_origin(_pbj_template_client_config()["pbj_public_api_origin"])
    if o and o.lower().startswith("https://"):
        return o + p
    return p


def _pbj_server_resolved_api_href(path: str) -> str:
    """Same rules as client `pbjApiUrl` for inline HTML from Flask (e.g. data-matching page)."""
    p = path if path.startswith("/") else "/" + path
    cfg = _pbj_template_client_config()
    o = cfg["pbj_public_api_origin"]
    if o:
        return o + p
    b = cfg["pbj_public_site_base_path"]
    if b:
        return b + p
    return p


def _pbj_server_resolved_site_href(path: str) -> str:
    """Resolve non-API app links with PUBLIC_SITE_BASE_PATH when present."""
    p = path if path.startswith("/") else "/" + path
    cfg = _pbj_template_client_config()
    b = cfg["pbj_public_site_base_path"]
    if b:
        return b + p
    return p


def _pbj_template_site_href(path: str) -> str:
    """Match ``pbj_site_href`` in ``partials/superdynamic_url_macros.html`` (API origin, then site base)."""
    p = path if path.startswith("/") else "/" + path
    cfg = _pbj_template_client_config()
    o = (cfg.get("pbj_public_api_origin") or "").strip()
    if o:
        return o.rstrip("/") + p
    b = (cfg.get("pbj_public_site_base_path") or "").strip()
    if b:
        return b.rstrip("/") + p
    return p


def _pbj_premium_facility_base_url(prov: Optional[str]) -> str:
    """
    Base URL for this facility on the public PBJ320 site (premium embed path).

    Used for methodology / data-matching links so they open on www.pbj320.com/premium/<CCN>/…
    instead of the Vercel deployment host when ``PUBLIC_API_ORIGIN`` points at Vercel.

    Override full base (no trailing slash): ``PBJ320_PREMIUM_PAGE_BASE``.
    Override marketing origin only: ``PBJ320_MARKETING_ORIGIN`` (default ``https://www.pbj320.com``).
    """
    override = _pbj_env_str("PBJ320_PREMIUM_PAGE_BASE").strip().rstrip("/")
    p = str(prov or "").strip().zfill(6)
    if not p.isdigit() or len(p) != 6:
        p = ""
    if override:
        m = re.match(r"^(https?://[^/]+/premium/)(\d{6})$", override, flags=re.I)
        if m and p:
            return m.group(1) + p
        return override
    if not p:
        return ""
    origin = _pbj_env_str("PBJ320_MARKETING_ORIGIN").strip().rstrip("/") or "https://www.pbj320.com"
    return f"{origin}/premium/{p}"


def _pbj_premium_facility_href(prov: Optional[str], path: str) -> str:
    """In-app facility page URL (dashboard, methodology, data-matching, report-builder).

    Same-origin relative on local dev; honors ``PUBLIC_SITE_BASE_PATH`` on marketing embed.
    Does not hardcode www.pbj320.com — use ``_pbj_premium_facility_base_url`` for marketing-only links.
    """
    return _pbj_server_resolved_site_href(path)


def _report_builder_parse_iso_date(raw: Any, field_name: str) -> datetime:
    text = str(raw or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ValueError(f"{field_name} must be YYYY-MM-DD")
    try:
        return datetime.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field_name} must be YYYY-MM-DD") from exc


def _report_builder_default_date_bounds(df: pd.DataFrame) -> tuple[datetime, datetime]:
    if df is None or df.empty or "WorkDate" not in df.columns:
        raise ValueError("Facility PBJ daily rows are not loaded.")
    wd = pd.to_datetime(df["WorkDate"], errors="coerce").dropna()
    if wd.empty:
        raise ValueError("Facility PBJ work dates are unavailable.")
    lo_raw = cast(Any, wd.min())
    hi_raw = cast(Any, wd.max())
    if pd.isna(lo_raw) or pd.isna(hi_raw):
        raise ValueError("Facility PBJ work dates are unavailable.")
    lo = pd.Timestamp(lo_raw).to_pydatetime()
    hi = pd.Timestamp(hi_raw).to_pydatetime()
    return cast(datetime, lo), cast(datetime, hi)


def _report_builder_parse_flexible_key_date_token(token: str) -> Optional[datetime]:
    """Parse a single date token as YYYY-MM-DD or MM-DD-YYYY (flexible)."""
    t = str(token or "").strip()
    if not t:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        try:
            return datetime.strptime(t, "%Y-%m-%d")
        except ValueError:
            return None
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})-(\d{2,4})", t)
    if not m:
        return None
    mm = int(m.group(1))
    dd = int(m.group(2))
    y_raw = m.group(3)
    yyyy = int(y_raw)
    if len(y_raw) == 2:
        yyyy += 2000
    if yyyy < 1900 or yyyy > 2100:
        return None
    try:
        return datetime(year=yyyy, month=mm, day=dd)
    except ValueError:
        return None


def _report_builder_parse_key_dates_and_notes(raw: Any) -> tuple[list[datetime], dict[str, str]]:
    """Parse key_dates from textarea, newline list, or list of strings / dicts with optional notes."""
    if raw is None:
        return [], {}
    entries: list[tuple[str, str]] = []

    def _push_date_note(date_part: str, note_part: str) -> None:
        dp = str(date_part or "").strip()
        np = str(note_part or "").strip()
        if dp:
            entries.append((dp, np))

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                d_raw = str(item.get("date") or item.get("key_date") or "").strip()
                note = str(item.get("note") or item.get("annotation") or item.get("label") or "").strip()
                _push_date_note(d_raw, note)
            else:
                token = str(item or "").strip()
                if "|" in token:
                    left, right = token.split("|", 1)
                    _push_date_note(left, right)
                else:
                    _push_date_note(token, "")
    else:
        parts = re.split(r"[\n,;]+", str(raw))
        for p in parts:
            token = str(p or "").strip()
            if not token:
                continue
            if "|" in token:
                left, right = token.split("|", 1)
                _push_date_note(left, right)
            else:
                _push_date_note(token, "")

    out: list[datetime] = []
    notes_by_iso: dict[str, str] = {}
    seen: set[str] = set()

    for date_token, note in entries:
        dt = _report_builder_parse_flexible_key_date_token(date_token)
        if dt is None:
            continue
        k = dt.strftime("%Y-%m-%d")
        if k in seen:
            if note:
                notes_by_iso[k] = note
            continue
        seen.add(k)
        out.append(dt)
        if note:
            notes_by_iso[k] = note
    return sorted(out), notes_by_iso


def _report_builder_parse_ranges(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        s = str(item.get("start_date") or item.get("start") or "").strip()
        e = str(item.get("end_date") or item.get("end") or "").strip()
        role = str(item.get("role") or "primary").strip().lower()
        if role not in ("primary", "secondary"):
            role = "primary"
        if not s or not e:
            continue
        try:
            sd = _report_builder_parse_iso_date(s, "date_range.start")
            ed = _report_builder_parse_iso_date(e, "date_range.end")
        except ValueError:
            continue
        if ed < sd:
            sd, ed = ed, sd
        out.append({"start": sd, "end": ed, "role": role})
    return out


def _report_builder_generate_html(payload: dict[str, Any]) -> dict[str, Any]:
    global global_df
    if global_df is None or len(global_df) == 0:
        raise ValueError("Facility daily data is not loaded yet.")
    df = cast(pd.DataFrame, global_df.copy())
    bounds_lo, bounds_hi = _report_builder_default_date_bounds(df)

    start_dt = _report_builder_parse_iso_date(payload.get("start_date") or bounds_lo.strftime("%Y-%m-%d"), "start_date")
    end_dt = _report_builder_parse_iso_date(payload.get("end_date") or bounds_hi.strftime("%Y-%m-%d"), "end_date")
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt
    if end_dt < bounds_lo or start_dt > bounds_hi:
        raise ValueError("Selected period is outside loaded PBJ date bounds.")
    start_dt = max(start_dt, bounds_lo)
    end_dt = min(end_dt, bounds_hi)

    key_dates, key_date_notes = _report_builder_parse_key_dates_and_notes(payload.get("key_dates"))
    for _date_key in ("licensee_date", "incident_date"):
        _raw = str(payload.get(_date_key) or "").strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", _raw):
            try:
                _dt = datetime.strptime(_raw, "%Y-%m-%d")
                if all(_dt.date() != ex.date() for ex in key_dates):
                    key_dates.append(_dt)
            except ValueError:
                pass
    key_dates = sorted(key_dates)
    date_ranges_of_interest = _report_builder_parse_ranges(payload.get("date_ranges_of_interest"))
    staffing_emphasis = str(payload.get("staffing_emphasis") or "total_first").strip() or "total_first"
    include_total_staffing = staffing_emphasis != "direct_only"
    direct_first = staffing_emphasis == "direct_first"

    facility_info = facility_report_lib.get_facility_info(df, start_dt, end_dt)
    provnum = str(facility_info.get("provnum") or "").strip().zfill(6)

    all_quarters = sorted(df["CY_Qtr"].astype(str).dropna().unique().tolist()) if "CY_Qtr" in df.columns else []
    start_q = f"{start_dt.year}Q{((start_dt.month - 1) // 3) + 1}"
    end_q = f"{end_dt.year}Q{((end_dt.month - 1) // 3) + 1}"
    quarters_in_range = [q for q in all_quarters if start_q <= q <= end_q]

    quarterly_data: dict[str, Any] = {}
    for quarter in quarters_in_range:
        q_metrics = facility_report_lib.calculate_quarterly_metrics(df, quarter)
        if q_metrics:
            quarterly_data[quarter] = q_metrics

    state_code = str(facility_info.get("state") or "").strip()
    state_comparisons = facility_report_lib.get_state_averages_batch(state_code, quarters_in_range)
    period_metrics = facility_report_lib.calculate_period_metrics(df, start_dt, end_dt) or {}
    macpac_standards = facility_report_lib.get_macpac_state_standards(state_code) or {}
    min_staffing = float(macpac_standards.get("min_staffing", 0.0) or 0.0)
    if min_staffing > 0:
        days_under = facility_report_lib.calculate_days_under_state_minimum(
            df,
            start_dt,
            end_dt,
            min_staffing,
        )
        if isinstance(days_under, dict):
            period_metrics.update(days_under)

    daily_staffing: list[dict[str, Any]] = []
    for d in key_dates:
        day_data = facility_report_lib.get_daily_staffing(df, d)
        if day_data:
            daily_staffing.append(day_data)

    red_flags_history: list[dict[str, Any]] = []
    case_mix_data: list[dict[str, Any]] = []
    previous_names: list[dict[str, str]] | list[str] = facility_info.get("previous_names") or []
    try:
        provider_info_df = facility_report_lib.load_provider_info_data(provnum)
        if provider_info_df is not None and not provider_info_df.empty:
            red_flags_history = facility_report_lib.extract_red_flags_history(provider_info_df, start_dt, end_dt)
            case_mix_data = facility_report_lib.extract_case_mix_data(
                provider_info_df,
                start_dt,
                end_dt,
                quarters_in_range=None,
            )
            previous_names_with_years = facility_report_lib.get_previous_names_with_years(
                provnum,
                provider_info_df,
                str(facility_info.get("name") or ""),
            )
            if previous_names_with_years:
                previous_names = previous_names_with_years
    except Exception:
        provider_info_df = None

    include_sections = payload.get("include_sections")
    if not isinstance(include_sections, dict):
        include_sections = {
            "key_dates": True,
            "date_ranges_of_interest": True,
            "quarterly_staffing": True,
            "case_mix": True,
            "red_flags": True,
            "period_summary": True,
            "state_compliance": True,
            "daily_staffing_table": True,
            "appendix": True,
        }

    report_kwargs: Dict[str, Any] = {
        "provnum": provnum,
        "facility_name": str(facility_info.get("name") or ""),
        "city": str(facility_info.get("city") or ""),
        "state": state_code,
        "start_date": start_dt,
        "end_date": end_dt,
        "key_dates": key_dates,
        "key_date_notes": key_date_notes,
        "quarterly_data": quarterly_data,
        "state_comparisons": state_comparisons,
        "date_ranges_of_interest": date_ranges_of_interest,
        "period_metrics": period_metrics,
        "daily_staffing": daily_staffing,
        "macpac_standards": macpac_standards,
        "red_flags_history": red_flags_history,
        "case_mix_data": case_mix_data,
        "pbj_df": df,
        "include_total_staffing": include_total_staffing,
        "direct_first": direct_first,
        "watermark": False,
        "include_sections": cast(dict[str, bool], include_sections),
        "previous_names": cast(Sequence, previous_names),
    }
    html_report = _generate_attorney_report_compat(report_kwargs)

    safe_name = re.sub(r"[^a-z0-9_]+", "_", str(facility_info.get("name") or provnum).lower()).strip("_")
    file_name = f"staffing_analysis_memo_{provnum}_{safe_name}_{datetime.now().strftime('%Y%m%d')}.html"

    return {
        "html": html_report,
        "file_name": file_name,
        "resolved_start_date": start_dt.strftime("%Y-%m-%d"),
        "resolved_end_date": end_dt.strftime("%Y-%m-%d"),
        "warnings": [],
    }


_REPORT_BUILDER_V3_ALLOWED_CCNS = frozenset({"315128"})


def _pbj_quarter_keys_from_bounds(min_iso: str, max_iso: str) -> list[str]:
    """Calendar quarters (``yyyyQn``) spanning PBJ workdate bounds, ascending."""
    import re
    from datetime import date

    def _parse(iso: str) -> date | None:
        s = str(iso or "").strip()[:10]
        m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
        if not m:
            return None
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None

    def _yq(d: date) -> tuple[int, int]:
        return d.year, (d.month - 1) // 3 + 1

    lo = _parse(min_iso)
    hi = _parse(max_iso)
    if not lo or not hi:
        return []
    if hi < lo:
        lo, hi = hi, lo
    y, q = _yq(lo)
    end_y, end_q = _yq(hi)
    out: list[str] = []
    while (y, q) <= (end_y, end_q):
        out.append(f"{y}Q{q}")
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out


def _pbj_report_builder_v3_route_allowed(provnum: str, *, beta_query: bool = False) -> bool:
    """Hidden beta route: enabled CCNs only; optional env hardening."""
    ccn = str(provnum or "").strip().zfill(6)
    if ccn not in _REPORT_BUILDER_V3_ALLOWED_CCNS:
        return False
    beta_only = _pbj_env_str("PBJ_REPORT_BUILDER_V3_BETA_ONLY").strip().lower() in ("1", "true", "yes")
    if beta_only:
        env_ok = _pbj_env_str("PBJ_REPORT_BUILDER_V3_ENABLED").strip().lower() in ("1", "true", "yes")
        return beta_query or env_ok
    return True


def _generate_attorney_report_compat(report_kwargs: Dict[str, Any]) -> str:
    """Call generate_attorney_report with only kwargs the loaded library accepts."""
    safe = getattr(facility_report_lib, "generate_attorney_report_safe", None)
    if callable(safe):
        return safe(report_kwargs)

    import inspect

    kwargs = dict(report_kwargs or {})
    attempts = 0
    while attempts < 16:
        attempts += 1
        try:
            sig = inspect.signature(facility_report_lib.generate_attorney_report)
            allowed = set(sig.parameters.keys())
            filtered = {k: v for k, v in kwargs.items() if k in allowed}
            return facility_report_lib.generate_attorney_report(**filtered)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" not in msg:
                raise
            match = re.search(r"argument '([^']+)'", msg)
            if not match:
                raise
            bad_key = match.group(1)
            if bad_key not in kwargs:
                raise
            kwargs.pop(bad_key, None)
    raise TypeError("generate_attorney_report_compat: too many incompatible keyword arguments")


def _report_builder_v3_generate_html(payload: dict[str, Any]) -> dict[str, Any]:
    import report_builder_v3 as rb3

    global global_df
    if global_df is None or len(global_df) == 0:
        raise ValueError("Facility daily data is not loaded yet.")
    df = cast(pd.DataFrame, global_df.copy())
    bounds_lo, bounds_hi = _report_builder_default_date_bounds(df)

    start_dt = _report_builder_parse_iso_date(payload.get("start_date") or bounds_lo.strftime("%Y-%m-%d"), "start_date")
    end_dt = _report_builder_parse_iso_date(payload.get("end_date") or bounds_hi.strftime("%Y-%m-%d"), "end_date")
    if end_dt < start_dt:
        start_dt, end_dt = end_dt, start_dt
    if end_dt < bounds_lo or start_dt > bounds_hi:
        raise ValueError("Selected period is outside loaded PBJ date bounds.")
    start_dt = max(start_dt, bounds_lo)
    end_dt = min(end_dt, bounds_hi)

    key_dates, key_date_notes, key_date_types = rb3.parse_key_date_events(payload.get("key_dates"))
    date_ranges_of_interest = _report_builder_parse_ranges(payload.get("date_ranges_of_interest"))
    staffing_emphasis = str(payload.get("staffing_emphasis") or "total_first").strip() or "total_first"
    include_total_staffing = staffing_emphasis != "direct_only"
    direct_first = staffing_emphasis == "direct_first"

    facility_info = facility_report_lib.get_facility_info(df, start_dt, end_dt)
    provnum = str(facility_info.get("provnum") or "").strip().zfill(6)

    all_quarters = sorted(df["CY_Qtr"].astype(str).dropna().unique().tolist()) if "CY_Qtr" in df.columns else []
    start_q = f"{start_dt.year}Q{((start_dt.month - 1) // 3) + 1}"
    end_q = f"{end_dt.year}Q{((end_dt.month - 1) // 3) + 1}"
    quarters_in_range = [q for q in all_quarters if start_q <= q <= end_q]

    quarterly_data: dict[str, Any] = {}
    for quarter in quarters_in_range:
        q_metrics = facility_report_lib.calculate_quarterly_metrics(df, quarter)
        if q_metrics:
            quarterly_data[quarter] = q_metrics

    state_code = str(facility_info.get("state") or "").strip()
    state_comparisons = facility_report_lib.get_state_averages_batch(state_code, quarters_in_range)
    period_metrics = facility_report_lib.calculate_period_metrics(df, start_dt, end_dt) or {}
    macpac_standards = facility_report_lib.get_macpac_state_standards(state_code) or {}
    min_staffing = float(macpac_standards.get("min_staffing", 0.0) or 0.0)
    if min_staffing > 0:
        days_under = facility_report_lib.calculate_days_under_state_minimum(
            df,
            start_dt,
            end_dt,
            min_staffing,
        )
        if isinstance(days_under, dict):
            period_metrics.update(days_under)

    daily_staffing: list[dict[str, Any]] = []
    for d in key_dates:
        day_data = facility_report_lib.get_daily_staffing(df, d)
        if day_data:
            daily_staffing.append(day_data)

    red_flags_history: list[dict[str, Any]] = []
    case_mix_data: list[dict[str, Any]] = []
    previous_names: list[dict[str, str]] | list[str] = facility_info.get("previous_names") or []
    try:
        provider_info_df = facility_report_lib.load_provider_info_data(provnum)
        if provider_info_df is not None and not provider_info_df.empty:
            red_flags_history = facility_report_lib.extract_red_flags_history(provider_info_df, start_dt, end_dt)
            case_mix_data = facility_report_lib.extract_case_mix_data(
                provider_info_df,
                start_dt,
                end_dt,
                quarters_in_range=None,
            )
            previous_names_with_years = facility_report_lib.get_previous_names_with_years(
                provnum,
                provider_info_df,
                str(facility_info.get("name") or ""),
            )
            if previous_names_with_years:
                previous_names = previous_names_with_years
    except Exception:
        provider_info_df = None

    ein_detail = globals().get("ein_employee_detail_df")
    v3_extras = rb3.build_v3_report_extras(
        payload=payload,
        df=df,
        start_dt=start_dt,
        end_dt=end_dt,
        bounds_lo=bounds_lo,
        bounds_hi=bounds_hi,
        key_dates=key_dates,
        key_date_notes=key_date_notes,
        key_date_types=key_date_types,
        include_total_staffing=include_total_staffing,
        min_staffing=min_staffing,
        facility_info=facility_info,
        facility_report_lib=facility_report_lib,
        ein_detail_df=ein_detail,
        state_code=state_code,
        macpac_standards=macpac_standards,
        case_mix_data=case_mix_data,
        quarters_in_range=quarters_in_range,
    )

    report_kwargs: Dict[str, Any] = {
        "provnum": provnum,
        "facility_name": str(facility_info.get("name") or ""),
        "city": str(facility_info.get("city") or ""),
        "state": state_code,
        "start_date": start_dt,
        "end_date": end_dt,
        "key_dates": key_dates,
        "key_date_notes": key_date_notes,
        "quarterly_data": quarterly_data,
        "state_comparisons": state_comparisons,
        "date_ranges_of_interest": date_ranges_of_interest,
        "period_metrics": period_metrics,
        "daily_staffing": daily_staffing,
        "macpac_standards": macpac_standards,
        "red_flags_history": red_flags_history,
        "case_mix_data": case_mix_data,
        "pbj_df": df,
        "include_total_staffing": include_total_staffing,
        "direct_first": direct_first,
        "watermark": False,
        "include_sections": cast(dict[str, bool], v3_extras["include_sections"]),
        "previous_names": cast(Sequence, previous_names),
        "section_order": v3_extras["section_order"],
        "v3_report_summary_html": v3_extras["v3_report_summary_html"],
        "event_windows_section_html": v3_extras["event_windows_section_html"],
        "supporting_context_section_html": v3_extras.get("supporting_context_section_html") or "",
        "key_staffing_findings_section_html": v3_extras.get("key_staffing_findings_section_html") or "",
    }
    html_report = _generate_attorney_report_compat(report_kwargs)

    safe_name = re.sub(r"[^a-z0-9_]+", "_", str(facility_info.get("name") or provnum).lower()).strip("_")
    file_name = f"staffing_analysis_memo_{provnum}_{safe_name}_{datetime.now().strftime('%Y%m%d')}.html"

    return {
        "html": html_report,
        "file_name": file_name,
        "resolved_start_date": start_dt.strftime("%Y-%m-%d"),
        "resolved_end_date": end_dt.strftime("%Y-%m-%d"),
        "warnings": v3_extras.get("warnings") or [],
        "staffing_findings": v3_extras.get("staffing_findings") or [],
        "findings_count": len(v3_extras.get("staffing_findings") or []),
    }


_PBJ_CORS_STATIC_ORIGINS = frozenset(
    (
        "https://www.pbj320.com",
        "https://pbj320.com",
    )
)
_PBJ_CORS_VERCEL_PREMIUM_RE = re.compile(r"^https://pbj320-\d{6}\.vercel\.app$", re.IGNORECASE)


def _pbj_cors_reflect_origin(req_origin: str | None) -> str | None:
    if not req_origin:
        return None
    o = req_origin.strip()
    if o in _PBJ_CORS_STATIC_ORIGINS:
        return o
    if _PBJ_CORS_VERCEL_PREMIUM_RE.match(o):
        return o
    api_o = _pbj_normalize_public_api_origin(_pbj_env_str("PUBLIC_API_ORIGIN"))
    if api_o and o.rstrip("/") == api_o:
        return o
    return None


@app.after_request
def _pbj_cors_after(response: Response):
    if not request.path.startswith("/api"):
        return response
    origin = _pbj_cors_reflect_origin(request.headers.get("Origin"))
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        # Required when the browser uses fetch(..., { credentials: 'include' }) from another site
        # (e.g. www.pbj320.com/premium/<CCN> calling the Vercel API host).
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Vary"] = "Origin"
    return response


@app.before_request
def _pbj_cors_preflight_options():
    if request.method != "OPTIONS" or not request.path.startswith("/api"):
        return None
    r = Response(status=204)
    origin = _pbj_cors_reflect_origin(request.headers.get("Origin"))
    if origin:
        r.headers["Access-Control-Allow-Origin"] = origin
        r.headers["Access-Control-Allow-Credentials"] = "true"
        r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        r.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        r.headers["Access-Control-Max-Age"] = "86400"
        r.headers["Vary"] = "Origin"
    return r


def _plotly_y_nullable(series: Any) -> list:
    """Build Plotly ``y`` arrays with JSON nulls for missing values so lines break across gaps (``connectgaps: False``).

    ``Any``: ``df[col]`` is typed as ``Series | DataFrame`` under pandas stubs; runtime is a 1-D series.
    """
    out: list[Any] = []
    for v in series.tolist():
        try:
            if v is None or pd.isna(v):
                out.append(None)
            elif isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                out.append(None)
            else:
                out.append(float(v))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _plotly_y_nullable_optional(df: pd.DataFrame, col: str) -> list:
    if col not in df.columns:
        return [None] * len(df)
    return _plotly_y_nullable(df[col])


def _pbj_favicon_path() -> Optional[str]:
    """Resolve favicon PNG under app root or ``pbj_images/``."""
    for rel in ("pbj_favicon.png", os.path.join("pbj_images", "pbj_favicon.png")):
        p = os.path.join(_app_root, rel)
        if os.path.isfile(p):
            return p
    return None


def _pbj_favicon_directory_and_name() -> tuple[str, str] | None:
    path = _pbj_favicon_path()
    if not path:
        return None
    return os.path.dirname(path), os.path.basename(path)


def _default_provnum_for_bundle() -> str:
    """Resolve facility CCN for this deployment bundle.

    Order: ``PBJ_FACILITY_CCN``, ``PBJ_PROVNUM``, or ``PROVNUM`` (any string containing digits),
    else the six-digit suffix of a parent directory named ``pbj320-<CCN>`` (matches
    ``deployments/pbj320-<CCN>/`` on Vercel and locally). Returns ``""`` if unknown
    so callers can surface a clear configuration error.
    """
    env_ccn = (
        os.environ.get("PBJ_FACILITY_CCN")
        or os.environ.get("PBJ_PROVNUM")
        or os.environ.get("PROVNUM")
        or ""
    ).strip()
    if env_ccn:
        digits = "".join(ch for ch in env_ccn if ch.isdigit())
        if digits:
            return digits[-6:].zfill(6) if len(digits) >= 6 else digits.zfill(6)
    base = os.path.basename(os.path.normpath(_this_dir))
    m = re.match(r"^pbj320-(\d{6})$", base, flags=re.I)
    if m:
        return m.group(1)
    return ""


# Initialize data lazily (for Vercel deployment). CCN from env or ``pbj320-<CCN>/`` folder name.
PROVNUM = _default_provnum_for_bundle()
EIN_DASHBOARD_MODE = "all"
EIN_SELECTED_QUARTERS = []
# Non-nurse PBJ slice + EIN bridge are enabled for this facility (do not override ``INCLUDE_NONNURSE`` above).
_data_initialized = False

def ensure_data_loaded():
    """Lazy initialization - only load data on first request"""
    global _data_initialized, global_df
    if not (PROVNUM or "").strip():
        print(
            "[ERROR] Facility CCN (PROVNUM) is not set. Set PBJ_FACILITY_CCN to a 6-digit CCN "
            "or run/deploy from ``deployments/pbj320-<CCN>/``.",
            flush=True,
        )
        return
    if global_df is not None and not global_df.empty:
        _data_initialized = True
        try:
            if ein_job_quarterly_df is None:
                _load_ein_position_csvs(PROVNUM)
        except Exception:
            pass
        return
    if _data_initialized:
        return
    try:
        print(f"Initializing facility {PROVNUM} dashboard (lazy load)...")
        create_dynamic_dashboard(PROVNUM)
        if global_df is not None and not global_df.empty:
            try:
                strict_mapping = str(os.environ.get("PBJ_STRICT_CMS_MAPPING", "")).strip().lower() in (
                    "1",
                    "true",
                    "yes",
                    "on",
                )
                mapping_report = run_cms_mapping_integrity_report(strict=strict_mapping)
                if not mapping_report.get("ok", False):
                    print(
                        "[WARNING] CMS mapping integrity check reported issues. "
                        "Run `python scripts/verify_cms_mapping_integrity.py --strict` before deploy."
                    )
            except Exception as map_exc:
                print(f"[WARNING] CMS mapping integrity check failed to run: {map_exc}")
            print(f"[OK] Successfully initialized facility {PROVNUM} dashboard")
            _data_initialized = True
        else:
            print(f"[WARNING] Facility {PROVNUM} init produced no rows; retrying next request.")
    except Exception as e:
        print(f"[WARNING] Error initializing facility {PROVNUM} dashboard: {e}")
        import traceback
        traceback.print_exc()

@app.before_request
def before_request():
    auth_resp = _dashboard_basic_auth_challenge()
    if auth_resp is not None:
        return auth_resp
    ensure_data_loaded()

@app.route("/pbj_favicon.png")
def pbj_favicon_png():
    """Serve ``pbj_favicon.png`` from the application directory (repo root for facility apps)."""
    fav = _pbj_favicon_directory_and_name()
    if not fav:
        return ("", 404)
    directory, filename = fav
    return send_from_directory(directory, filename, mimetype="image/png")


@app.route("/favicon.ico")
def favicon_ico():
    """Browsers request ``/favicon.ico`` by default; reuse the PNG asset when present."""
    fav = _pbj_favicon_directory_and_name()
    if not fav:
        return ("", 204)
    directory, filename = fav
    return send_from_directory(directory, filename, mimetype="image/png")


@app.route("/ai-icons/<path:icon_name>")
def ai_brand_icon(icon_name: str) -> ResponseReturnValue:
    """Brand SVGs (same files as pbj-root ``ai-icons/``; also under ``static/ai-icons/``)."""
    safe = os.path.basename(str(icon_name or "").replace("\\", "/"))
    if safe not in ("claude.svg", "chatgpt.svg", "gemini.svg"):
        abort(404)
    for base in (
        os.path.join(_app_root, "static", "ai-icons"),
        os.path.join(_app_root, "ai-icons"),
    ):
        path = os.path.join(base, safe)
        if os.path.isfile(path):
            return send_file(path, mimetype="image/svg+xml")
    abort(404)


@app.route("/downloads/pbj320-staffing-review.zip")
def download_pbj_claude_skill_zip() -> ResponseReturnValue:
    """Claude Skill install package (facility dashboard origin; not public www downloads)."""
    if not pbj_ai_skill_zip_facility_enabled(_app_root):
        abort(404)
    import zipfile

    skill_dir, zip_path = pbj_claude_skill_zip_paths(_app_root)
    if not os.path.isfile(zip_path) and os.path.isdir(skill_dir):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        prefix = "pbj320-staffing-review"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for dirpath, _dirnames, filenames in os.walk(skill_dir):
                for name in filenames:
                    full = os.path.join(dirpath, name)
                    rel = os.path.relpath(full, skill_dir)
                    arc = os.path.join(prefix, rel).replace("\\", "/")
                    zf.write(full, arc)
    if not os.path.isfile(zip_path):
        abort(404)
    return send_file(
        zip_path,
        mimetype="application/zip",
        as_attachment=True,
        download_name="pbj320-staffing-review.zip",
    )


# Global variables
df = None
global_df = None
nonnurse_df = None
_NONNURSE_CSV_PATH: Optional[str] = None
_NONNURSE_LOAD_PROVNUM: Optional[str] = None
_NONNURSE_LOAD_SCAN_ROOT: Optional[str] = None
_NONNURSE_LOAD_FACILITY_FOLDER: Optional[str] = None
_NONNURSE_LOAD_LOCK = threading.Lock()
citations_df = None
citations_loaded_path: Optional[str] = None
provider_info_df = None
provider_info_loaded_source: Optional[str] = None
macpac_standards_df = None
ein_job_quarterly_df = None
ein_category_quarterly_df = None
ein_employee_detail_df = None
ein_nursing_summaries_df = None
_EIN_NURSING_SUMMARIES_ROWS_CACHE: tuple[Any, ...] | None = None
_EIN_NURSING_SUMMARIES_ROWS: list[dict[str, Any]] | None = None
_EIN_ROSTER_PREWARM_STARTED = False
_EIN_ROSTER_PREWARM_LOCK = threading.Lock()
_PROVIDER_CHARTS_CACHE: dict[str, Any] | None = None
_PROVIDER_CHARTS_CACHE_ID: tuple | None = None
# Bump when provider-quarter aggregation changes so a running process drops stale JSON cache.
_PROVIDER_CHARTS_AGG_VERSION = 3
_NH_OWNERSHIP_CSV_PATH_CACHE: Optional[str] = None
_SNF_ALL_OWNERS_CSV_PATH_CACHE: Optional[str] = None
_OWNERSHIP_CONTACTS_CACHE: dict[str, list[dict[str, Any]]] = {}
_OWNERSHIP_CONTACTS_CACHE_SIG: tuple[Any, ...] | None = None
_SNF_FACILITY_OWNER_ASSOC_CACHE: dict[str, dict[str, str]] = {}
_SNF_FACILITY_OWNER_ASSOC_MTIME: float = 0.0
_FILTERED_DF_CACHE: dict[tuple[Any, ...], tuple[float, pd.DataFrame]] = {}
_FILTERED_DF_CACHE_LOCK = threading.Lock()
_FILTERED_DF_CACHE_TTL_SEC = 90.0
# Production deploy date from config/facility_deployed_at.json (stamped on vercel --prod).
DEPLOYED_DATE = read_deployed_date_display(_app_root)


def _nh_health_citations_dataset_asof_display() -> str:
    """
    Human-readable CMS extract vintage for the NH Health Citations file.

    Prefer the newest ``NH_HealthCitations_MonYYYY.csv`` filename (e.g. "May 2026 CMS extract"),
    not filesystem mtime — bundled copies often keep an old mtime (e.g. 2018) that mislabels the UI.
    """
    month_names = (
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    )
    mm_yyyy = _latest_nh_health_citations_month_year()
    if mm_yyyy and "/" in mm_yyyy:
        try:
            mm_s, yyyy_s = mm_yyyy.split("/", 1)
            mi = int(mm_s)
            yi = int(yyyy_s)
            if 1 <= mi <= 12 and yi >= 2017:
                return f"{month_names[mi]} {yi} CMS extract"
        except (TypeError, ValueError):
            pass
    global citations_loaded_path
    if citations_loaded_path and os.path.isfile(citations_loaded_path):
        base = os.path.basename(citations_loaded_path)
        m = re.match(r"^NH_HealthCitations_([A-Za-z]{3,9})(\d{4})\.csv$", base, flags=re.I)
        if m:
            month_map = {
                "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
            }
            mon_txt = m.group(1)[:3].lower()
            yyyy = int(m.group(2))
            mm = month_map.get(mon_txt)
            if mm and yyyy >= 2017:
                return f"{month_names[mm]} {yyyy} CMS extract"
    try:
        if citations_loaded_path and os.path.isfile(citations_loaded_path):
            mt = os.path.getmtime(citations_loaded_path)
            dt = datetime.fromtimestamp(mt)
            if dt.year >= 2019:
                return dt.strftime("%B %-d, %Y")
    except (ValueError, OSError):
        pass
    return ""


def _latest_nh_health_citations_month_year() -> str:
    """Newest NH_HealthCitations_* file label as MM/YYYY for UI source badges."""
    month_map = {
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    }
    search_dirs = [
        os.path.join(_app_root, "Citations"),
        os.path.join(os.path.abspath(os.path.join(_app_root, "..", "..")), "Citations"),
    ]
    best: tuple[int, int] | None = None
    for base in search_dirs:
        try:
            if not os.path.isdir(base):
                continue
            for fn in os.listdir(base):
                m = re.match(r"^NH_HealthCitations_([A-Za-z]{3,9})(\d{4})\.csv$", str(fn).strip(), flags=re.I)
                if not m:
                    continue
                mon_txt = m.group(1)[:3].lower()
                yyyy = int(m.group(2))
                mm = month_map.get(mon_txt)
                if not mm:
                    continue
                key = (yyyy, mm)
                if best is None or key > best:
                    best = key
        except OSError:
            continue
    if not best:
        return ""
    return f"{best[1]:02d}/{best[0]:04d}"


def _format_workdate_json(value: Any) -> Any:
    """Format a WorkDate cell for JSON (datetime, Timestamp, numpy, int/float YYYYMMDD)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except Exception:
            pass
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")


def _sum_pbj_nurse_staff_hours_excl_admin(pbj_quarter: Any) -> float:
    """Match dashboard direct-hours total when derived column is missing from global_df.

    ``Any``: pandas boolean-index slices are typed as ``DataFrame | Series``; runtime value is always a frame here.
    """
    if pbj_quarter is None or len(pbj_quarter) == 0:
        return 0.0
    if "Nurse_Staff_Hours_Excl_Admin" in pbj_quarter.columns:
        s = cast(
            pd.Series,
            pd.to_numeric(_as_1d_series(pbj_quarter["Nurse_Staff_Hours_Excl_Admin"]), errors="coerce"),
        )
        return float(cast(Any, s.sum(min_count=1))) if s.notna().any() else 0.0
    hour_cols = ["Hrs_RN", "Hrs_LPN", "Hrs_CNA", "Hrs_NAtrn", "Hrs_MedAide"]
    if all(c in pbj_quarter.columns for c in hour_cols):
        m = pbj_quarter[hour_cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
        row_tot = m.sum(axis=1, min_count=len(hour_cols))
        return float(row_tot.sum(min_count=1)) if row_tot.notna().any() else 0.0
    return 0.0


def _pbj_cy_qtr_lookup_key_variants(label: object) -> list[str]:
    """Alias strings so provider ``2018Q1`` matches PBJ ``CY2018Q1`` / ``2018Q1`` in CY_Qtr."""
    if label is None or (isinstance(label, float) and np.isnan(label)):
        return []
    s = str(label).strip()
    if not s:
        return []
    out: set[str] = {s}
    u = s.upper().replace(" ", "")
    m = re.search(r"(?:CY)?(\d{4})Q([1-4])", u)
    if m:
        y, qn = m.group(1), m.group(2)
        out.add(f"{y}Q{qn}")
        out.add(f"CY{y}Q{qn}")
    cq = _provider_quarter_to_canonical(s)
    if cq:
        out.add(cq)
        out.add(cq.replace("CY", ""))
    return list(out)


def _pbj_direct_hprd_lookup_from_global_df(gdf: Optional[pd.DataFrame]) -> dict[str, tuple[float, float, float]]:
    """Single ``groupby(CY_Qtr)`` over PBJ daily rows → map label variants to (direct_hprd, rn_direct_hprd, lpn_direct_hprd).

    Replaces repeated ``global_df[CY_Qtr == q]`` scans (O(rows × quarters)) that stall Flask under parallel requests.
    """
    if gdf is None or len(gdf) == 0 or "CY_Qtr" not in gdf.columns:
        return {}
    lookup: dict[str, tuple[float, float, float]] = {}
    for cy_qtr, grp in gdf.groupby("CY_Qtr", sort=False):
        if "MDScensus" not in grp.columns:
            continue
        mdc = cast(pd.Series, pd.to_numeric(_as_1d_series(grp["MDScensus"]), errors="coerce"))
        ok = mdc.notna() & (mdc > 0)
        total_census = float(cast(Any, mdc[ok].sum()))
        if total_census <= 0:
            continue
        sub = grp.loc[ok]
        direct_hours = _sum_pbj_nurse_staff_hours_excl_admin(sub)
        rn_direct_hours = (
            float(
                cast(
                    Any,
                    cast(
                        pd.Series,
                        pd.to_numeric(_as_1d_series(sub["Hrs_RN"]), errors="coerce"),
                    ).sum(min_count=1),
                )
            )
            if "Hrs_RN" in sub.columns
            else 0.0
        )
        lpn_direct_hours = (
            float(
                cast(
                    Any,
                    cast(
                        pd.Series,
                        pd.to_numeric(_as_1d_series(sub["Hrs_LPN"]), errors="coerce"),
                    ).sum(min_count=1),
                )
            )
            if "Hrs_LPN" in sub.columns
            else 0.0
        )
        pair = (
            direct_hours / total_census,
            rn_direct_hours / total_census,
            lpn_direct_hours / total_census,
        )
        for k in _pbj_cy_qtr_lookup_key_variants(cy_qtr):
            if k and k not in lookup:
                lookup[k] = pair
    return lookup


def _lookup_pbj_direct_hprd(
    lookup: dict[str, tuple[float, float, float]], quarter_normalized: object
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    if not lookup:
        return (None, None, None)
    if quarter_normalized is None or (isinstance(quarter_normalized, float) and np.isnan(quarter_normalized)):
        return (None, None, None)
    qn = str(quarter_normalized).strip()
    if not qn:
        return (None, None, None)
    for k in _pbj_cy_qtr_lookup_key_variants(qn):
        if k in lookup:
            return lookup[k]
    return (None, None, None)


# Raw CMS provider info files for download links. Prefer provider_info_extracted at repo root; fallback to static/data/provider_info
PROVIDER_INFO_EXTRACTED = os.path.join(_app_root, 'provider_info_extracted')
PROVIDER_INFO_DATA_DIR = PROVIDER_INFO_EXTRACTED if os.path.isdir(PROVIDER_INFO_EXTRACTED) else os.path.join(_app_root, 'static', 'data', 'provider_info')

_CMS_PROVIDER_INFO_DATASET_PAGE = "https://data.cms.gov/provider-data/dataset/4pq5-n9py"


def _find_interval_quarter_mapping_json() -> Optional[str]:
    """
    Locate the shared interval→quarter snapshot JSON.

    Tries the deployment directory first (``static/data/`` next to the Flask app), then the
    repo root when the app lives under ``deployments/pbj320-XXXX/``. Same file for every facility.
    """
    candidates: list[str] = []
    candidates.append(os.path.join(_app_root, "static", "data", "interval_quarter_mapping.json"))
    norm_app = os.path.normpath(_app_root)
    parts = norm_app.split(os.sep)
    if (
        len(parts) >= 2
        and parts[-2].lower() == "deployments"
        and re.match(r"^pbj320-\d{6}$", parts[-1], flags=re.I)
    ):
        repo_guess = os.path.abspath(os.path.join(_app_root, "..", ".."))
        candidates.append(os.path.join(repo_guess, "static", "data", "interval_quarter_mapping.json"))
    walk = _app_root
    for _ in range(6):
        candidates.append(os.path.join(walk, "static", "data", "interval_quarter_mapping.json"))
        parent = os.path.dirname(walk)
        if parent == walk:
            break
        walk = parent
    seen: set[str] = set()
    for c in candidates:
        ap = os.path.normpath(os.path.abspath(c))
        if ap in seen:
            continue
        seen.add(ap)
        if os.path.isfile(ap):
            return ap
    return None


def _dec_year_month(y: int, m: int) -> tuple[int, int]:
    if m > 1:
        return y, m - 1
    return y - 1, 12


def _staffing_dates_for_quarter_label(quarter_label: str) -> tuple[str, str]:
    """Map 'Q1 2020' to inclusive CMS-style m/d/YYYY dates for that calendar quarter."""
    if not quarter_label:
        return "", ""
    mat = re.match(r"^Q([1-4])\s+(\d{4})$", str(quarter_label).strip(), flags=re.I)
    if not mat:
        return "", ""
    qi = int(mat.group(1))
    year = int(mat.group(2))
    bounds = {
        1: ((1, 1), (3, 31)),
        2: ((4, 1), (6, 30)),
        3: ((7, 1), (9, 30)),
        4: ((10, 1), (12, 31)),
    }
    (sm, sd), (em, ed) = bounds[qi]
    return f"{sm}/{sd}/{year}", f"{em}/{ed}/{year}"


def _pbj_quarter_url_from_quarter_label(quarter_label: str) -> str:
    if not quarter_label:
        return ""
    m = re.match(r"^Q([1-4])\s+(\d{4})$", str(quarter_label).strip().upper())
    if not m:
        return ""
    return (
        "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data/q"
        + m.group(1).lower()
        + "-"
        + m.group(2)
    )


def _provider_info_manual_interval_table_row(
    mm_yyyy: str, year: int, month: int, used_quarter: str, *, interval_csv_hint: str = ""
) -> dict:
    """One provider-info month→quarter mapping table row when no CMS interval ZIP row exists (not PBJ staffing data)."""
    import calendar
    import urllib.parse

    sf, st = _staffing_dates_for_quarter_label(used_quarter)
    abbr = calendar.month_abbr[month] if 1 <= month <= 12 else ""
    provider_info_csv_name = f"NH_ProviderInfo_{abbr}{year}.csv" if abbr else ""
    provider_info_download_url = ""
    if provider_info_csv_name:
        provider_info_download_url = (
            _cms_provider_info_archive_zip_url(provider_info_csv_name)
            or _pbj_server_resolved_api_href(
                f"/api/provider-info/download?file={urllib.parse.quote(provider_info_csv_name)}"
            )
        )
    interval_disp = (interval_csv_hint or "").strip()
    if interval_disp and not interval_disp.startswith("—"):
        interval_cell = _strip_interval_csv_display_path(interval_disp)
    else:
        interval_cell = "— (manual map in prov_info.py)"
    return {
        "processing_month": mm_yyyy,
        "provider_info_csv_name": provider_info_csv_name,
        "provider_info_download_url": provider_info_download_url,
        "interval_csv_name": interval_cell,
        "staffing_level_from": sf,
        "staffing_level_through": st,
        "interval_staffing_level_quarter": used_quarter,
        "manual_processing_quarter": used_quarter,
        "used_case_mix_quarter": used_quarter,
        "pbj_quarter_url": _pbj_quarter_url_from_quarter_label(used_quarter),
        "turnover_from": "",
        "turnover_through": "",
        "turnover_quarter": "",
    }


def _pbj_only_calendar_quarter_row(quarter_label: str) -> dict:
    """Table row when only a PBJ calendar quarter applies (pre-interval-file era, no monthly interval row)."""
    sf, st = _staffing_dates_for_quarter_label(quarter_label)
    return {
        "processing_month": "—",
        "provider_info_csv_name": "—",
        "provider_info_download_url": "",
        "interval_csv_name": "— (PBJ calendar quarter; no collection-interval row)",
        "staffing_level_from": sf,
        "staffing_level_through": st,
        "interval_staffing_level_quarter": quarter_label,
        "manual_processing_quarter": "—",
        "used_case_mix_quarter": quarter_label,
        "pbj_quarter_url": _pbj_quarter_url_from_quarter_label(quarter_label),
        "turnover_from": "",
        "turnover_through": "",
        "turnover_quarter": "",
    }


def _quarter_label_sort_key(lab: str) -> tuple[int, int]:
    """Parse 'Q2 2020' to (2020, 2); invalid labels -> (-1, -1)."""
    if not lab:
        return (-1, -1)
    mat = re.match(r"^Q([1-4])\s+(\d{4})$", str(lab).strip(), flags=re.I)
    if not mat:
        return (-1, -1)
    return int(mat.group(2)), int(mat.group(1))


def _current_calendar_quarter() -> tuple[int, int]:
    """Return (year, quarter 1–4) for today's date."""
    now = datetime.now()
    qn = (now.month - 1) // 3 + 1
    return now.year, qn


def _calendar_quarter_le(t1: tuple[int, int], t2: tuple[int, int]) -> bool:
    return t1[0] < t2[0] or (t1[0] == t2[0] and t1[1] <= t2[1])


def _min_calendar_quarter(t1: tuple[int, int], t2: tuple[int, int]) -> tuple[int, int]:
    return t1 if _calendar_quarter_le(t1, t2) else t2


def _latest_pbj_calendar_quarter_from_standardized_files() -> tuple[int, int] | None:
    """Newest (year, quarter 1–4) from ``standardized_PBJ`` / ``PBJcsv`` nurse filenames; None if none."""
    files = glob.glob(os.path.join("standardized_PBJ", "PBJ_dailynursestaffing_*.csv"))
    if not files:
        files = glob.glob(os.path.join("PBJcsv", "PBJ_dailynursestaffing_*.csv"))
    best: tuple[int, int] | None = None
    for p in files:
        m = re.search(r"CY(\d{4})Q([1-4])", os.path.basename(p), re.I)
        if not m:
            continue
        y, qn = int(m.group(1)), int(m.group(2))
        cand = (y, qn)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand
    return best


def _latest_pbj_calendar_quarter_from_loaded_global_df() -> tuple[int, int] | None:
    """Newest (year, quarter 1-4) observed in loaded facility PBJ rows (``global_df.CY_Qtr``)."""
    global global_df
    if global_df is None or len(global_df) == 0 or "CY_Qtr" not in global_df.columns:
        return None
    best: tuple[int, int] | None = None
    try:
        for raw in global_df["CY_Qtr"].dropna().astype(str).tolist():
            m = re.search(r"(?:CY)?(\d{4})Q([1-4])", raw.strip().upper())
            if not m:
                continue
            cand = (int(m.group(1)), int(m.group(2)))
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
    except Exception:
        return None
    return best


def _latest_pbj_calendar_quarter_available() -> tuple[int, int] | None:
    """Best PBJ coverage cap for quarter-mapping UI (facility data first, then standardized disk files)."""
    return _latest_pbj_calendar_quarter_from_loaded_global_df() or _latest_pbj_calendar_quarter_from_standardized_files()


def _filter_interval_mapping_rows_to_available_pbj(rows: list[dict]) -> list[dict]:
    """Hide interval rows that map past the latest PBJ quarter currently available to this dashboard."""
    cap = _latest_pbj_calendar_quarter_available()
    if not cap:
        return list(rows or [])
    out: list[dict] = []
    for r in rows or []:
        q = str((r.get("used_case_mix_quarter") or r.get("interval_staffing_level_quarter") or "")).strip()
        ky = _quarter_label_sort_key(q)
        if ky[0] >= 0 and (ky[0] > cap[0] or (ky[0] == cap[0] and ky[1] > cap[1])):
            continue
        out.append(r)
    return out


def _append_all_missing_pbj_calendar_quarters(rows: list[dict]) -> list[dict]:
    """Add PBJ-calendar-only rows for pre-case-mix era quarters missing from the mapping table.

    Only Q1–Q3 2017 are padded here (before CMS Provider Information case-mix staffing began).
    Later calendar-only placeholders (e.g. Q1 2020) are omitted—they clutter the table without
    adding actionable provider-info context.
    """
    have: set[str] = set()
    for r in rows or []:
        u = str(
            r.get("used_case_mix_quarter")
            or r.get("interval_staffing_level_quarter")
            or r.get("manual_processing_quarter")
            or ""
        ).strip()
        if not u or u.startswith("—"):
            continue
        have.add(u)

    out = list(rows or [])
    for q in (1, 2, 3):
        lab = f"Q{q} 2017"
        if lab not in have:
            out.append(_pbj_only_calendar_quarter_row(lab))
            have.add(lab)
    return out


def _data_matching_row_sort_key(r: dict) -> tuple:
    """Newest quarter first; within a quarter, real processing months before placeholder rows."""
    u = str(r.get("used_case_mix_quarter") or "").strip()
    qy, qq = _quarter_label_sort_key(u)
    tier = 2 if qy < 0 else 0
    if qy < 0:
        qy, qq = 0, 0
    pm = str(r.get("processing_month") or "").strip()
    if pm and pm != "—" and pm.count("-") == 1:
        parts = pm.split("-")
        try:
            mm, yy = int(parts[0]), int(parts[1])
        except ValueError:
            mm, yy = 0, 0
        return (tier, -qy, -qq, -yy, -mm)
    return (tier, -qy, -qq, 9999, 0)


def _sort_interval_mapping_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows or [], key=_data_matching_row_sort_key)


def _strip_interval_csv_display_path(name: str) -> str:
    """Drop CMS ZIP folder prefix; show from NH_DataCollectionIntervals... when possible."""
    if not name or not isinstance(name, str):
        return name or ""
    s = name.replace("\\", "/").strip()
    low = s.lower()
    marker = "nh_datacollectionintervals"
    idx = low.find(marker)
    if idx >= 0:
        return s[idx:]
    s2 = re.sub(
        r"^.*?nursing_homes_including_rehab_services[^/]*/",
        "",
        s,
        count=1,
        flags=re.I,
    )
    if s2 != s:
        return s2.lstrip("/")
    return s


def _format_us_date_compact_mdy(mdy: str) -> str:
    """Render m/d/YYYY as m/d/YY when a 4-digit year is present."""
    s = (mdy or "").strip()
    if not s:
        return s
    return re.sub(
        r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b",
        lambda m: f"{m.group(1)}/{m.group(2)}/{m.group(3)[2:]}",
        s,
    )


def _compact_quarter_label_display(text: str) -> str:
    """Q1 2020 -> Q1 20 for tighter table cells."""
    if not text:
        return text
    return re.sub(
        r"Q\s*([1-4])\s+(\d{4})\b",
        lambda m: f"Q{m.group(1)} {m.group(2)[2:]}",
        str(text),
        flags=re.IGNORECASE,
    ).strip()


def _compact_mapping_period_display(sl_from: str, sl_thru: str) -> str:
    sf = _format_us_date_compact_mdy(sl_from)
    st = _format_us_date_compact_mdy(sl_thru)
    if st:
        return sf + (" - " + st if sf else st)
    return sf


def _full_year_mapping_period_display(sl_from: str, sl_thru: str) -> str:
    """Join staffing/turnover date strings without shortening years (data-matching page)."""
    sf = (sl_from or "").strip()
    st = (sl_thru or "").strip()
    if st:
        return sf + (" - " + st if sf else st)
    return sf


def _merge_interval_mapping_json_with_manual_months(json_rows: list[dict]) -> list[dict]:
    """
    Bundled JSON may stop at an arbitrary month (e.g. 05-2021). prov_info keeps older month→quarter
    mappings in get_quarter_from_processing_month. Fill missing calendar months from the newest row
    (or current month) down through 2017-01 (try manual map first, then interval ZIP fallback), then
    sort newest-first. PBJ-calendar-only quarters (e.g. Q1–Q3 2017) are appended in
    ``_load_interval_quarter_mapping_fallback`` after this merge.
    """
    from prov_info_quarter_map import get_manual_quarter_from_processing_month
    import urllib.parse

    by_key: dict[str, dict] = {}
    interval_by_quarter: dict[str, str] = {}
    max_y, max_m = 2018, 4
    for r in json_rows or []:
        k = (r.get("processing_month") or "").strip()
        uq = str(r.get("used_case_mix_quarter") or r.get("interval_staffing_level_quarter") or "").strip()
        icn = str(r.get("interval_csv_name") or "").strip()
        if uq and icn and not icn.startswith("—"):
            interval_by_quarter[uq] = icn
        if not k or k.count("-") != 1:
            continue
        parts = k.split("-")
        try:
            mm, yy = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        by_key[k] = r
        if yy > max_y or (yy == max_y and mm > max_m):
            max_y, max_m = yy, mm

    now = datetime.now()
    if (now.year, now.month) > (max_y, max_m):
        max_y, max_m = now.year, now.month

    y, m = max_y, max_m
    while (y, m) >= (2017, 1):
        key = f"{m:02d}-{y}"
        if key not in by_key:
            proc_iso = f"{y}-{m:02d}"
            mq = get_manual_quarter_from_processing_month(proc_iso)
            if not mq and (y < 2018 or (y == 2018 and m < 4)):
                try:
                    from prov_info import get_quarter_from_processing_month

                    mq = get_quarter_from_processing_month(
                        proc_iso, use_interval_fallback=True
                    )
                except Exception:
                    mq = None
            if mq:
                by_key[key] = _provider_info_manual_interval_table_row(
                    key, y, m, mq, interval_csv_hint=interval_by_quarter.get(mq, "")
                )
        y, m = _dec_year_month(y, m)

    def sort_key(k: str) -> tuple[int, int]:
        mm, yy = k.split("-")
        return int(yy), int(mm)

    ordered_keys = sorted(by_key.keys(), key=sort_key, reverse=True)
    merged_list = [by_key[k] for k in ordered_keys]
    for r in merged_list:
        pm = str(r.get("processing_month") or "").strip()
        if pm.count("-") == 1:
            try:
                mm_s, yy_s = pm.split("-")
                proc_iso = f"{int(yy_s):04d}-{int(mm_s):02d}"
                mq = get_manual_quarter_from_processing_month(proc_iso)
                if mq:
                    r["manual_processing_quarter"] = mq
            except ValueError:
                pass
        src_name = str(r.get("provider_info_csv_name") or "").strip()
        if src_name:
            r["provider_info_download_url"] = (
                _cms_provider_info_archive_zip_url(src_name)
                or _pbj_server_resolved_api_href(
                    f"/api/provider-info/download?file={urllib.parse.quote(src_name)}"
                )
            )
        icn = r.get("interval_csv_name")
        if icn:
            r["interval_csv_name"] = _strip_interval_csv_display_path(str(icn))
        elif str(r.get("manual_processing_quarter") or "").strip():
            mq_hint = str(r.get("manual_processing_quarter") or r.get("used_case_mix_quarter") or "").strip()
            inherited = interval_by_quarter.get(mq_hint, "")
            if inherited:
                r["interval_csv_name"] = _strip_interval_csv_display_path(inherited)
    return _sort_interval_mapping_rows(merged_list)


def _load_interval_quarter_mapping_fallback(limit_i: int) -> list[dict]:
    """Load rows from static JSON; same schema as /api/provider-info/interval-quarter-mapping.

    Use ``limit_i <= 0`` to return the full bundled history (for ``/data-matching`` HTML).
    Positive limits cap how many **newest** rows are returned (JSON is newest-first).

    Rows are merged with prov_info manual month→quarter mappings (and interval ZIP fallback) for missing
    months back to 2017-01, plus PBJ-calendar-only rows for Q1–Q3 2017 when those quarters lack a mapping row.
    """
    path = _find_interval_quarter_mapping_json()
    if not path:
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows") or []
        merged = _merge_interval_mapping_json_with_manual_months(rows)
        if not merged:
            merged = list(rows or [])
        # Always backfill PBJ calendar quarters to Q1 2017, even if prov_info merge failed (e.g. import error).
        merged = _append_all_missing_pbj_calendar_quarters(merged)
        merged = _filter_interval_mapping_rows_to_available_pbj(merged)
        if not merged:
            return []
        if int(limit_i) <= 0:
            return merged
        li = max(1, min(int(limit_i), 500))
        return merged[:li]
    except Exception:
        return []


def _interval_quarter_mapping_table_rows_html(rows: list[dict], limit_i: int = 0) -> str:
    """Build ``<tbody>`` rows for the data-matching page (escaped HTML).

    ``limit_i <= 0`` means use all of ``rows`` (caller passes the already-sliced list).
    """
    if int(limit_i) > 0:
        slice_rows = (rows or [])[: int(limit_i)]
    else:
        slice_rows = rows or []
    if not slice_rows:
        return (
            '<tr><td colspan="6" class="text-muted">No interval mapping rows are bundled. '
            "Add <code>static/data/interval_quarter_mapping.json</code> next to the app "
            "(or regenerate deployments with <code>create_vercel_deployment.py</code>).</td></tr>"
        )

    def td_text(val: str) -> str:
        t = val or ""
        et = html.escape(t)
        return f'<td title="{et}">{et}</td>'

    def td_link(label: str, href: str) -> str:
        lab = label or ""
        if href:
            return (
                '<td><a href="'
                + html.escape(href, quote=True)
                + '" target="_blank" rel="noopener">'
                + html.escape(lab)
                + "</a></td>"
            )
        return td_text(lab)

    parts: list[str] = []
    for r in slice_rows:
        sl_from = str(r.get("staffing_level_from") or "")
        sl_thru = str(r.get("staffing_level_through") or "")
        sl_period = _full_year_mapping_period_display(sl_from, sl_thru)

        used_q = str(r.get("used_case_mix_quarter") or "").strip()
        intv_q = str(r.get("interval_staffing_level_quarter") or "").strip()
        pbj_quarter_label = (used_q or intv_q).strip()
        pbj_url = str(r.get("pbj_quarter_url") or "").strip()
        manual_q = str(r.get("manual_processing_quarter") or "").strip()
        if not manual_q and intv_q:
            # Keep the "PBJ matched" column populated from interval-derived quarter when manual map is blank.
            manual_q = intv_q
        intv_disp = _strip_interval_csv_display_path(str(r.get("interval_csv_name") or ""))

        parts.append("<tr>")
        # 1. PBJ Quarter (linked to CMS PBJ slice when URL is present)
        parts.append(td_link(pbj_quarter_label, pbj_url))
        parts.append(td_link(str(r.get("provider_info_csv_name") or ""), str(r.get("provider_info_download_url") or "")))
        parts.append(td_text(sl_period))
        parts.append(td_text(manual_q))
        parts.append(td_text(str(r.get("processing_month") or "")))
        parts.append(td_text(intv_disp))
        parts.append("</tr>")
    return "".join(parts)


def _cms_provider_info_archive_zip_url(source_filename: str) -> str | None:
    """
    Map a provider info source file name (e.g. 'NH_ProviderInfo_Oct2018.csv') to the CMS archive ZIP URL.
    CMS provider info is published as monthly ZIP archives. This lets attorneys pull the exact snapshot
    referenced in the red-flag table even when we don't host the raw CSV.
    """
    if not source_filename:
        return None
    name = os.path.basename(str(source_filename).strip())
    m = re.search(r'NH_ProviderInfo_([A-Za-z]{3})(\d{4})\.csv$', name)
    if not m:
        return None
    mon_abbr = m.group(1).title()
    year = int(m.group(2))
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    month = month_map.get(mon_abbr)
    if not month:
        return None

    base = "https://data.cms.gov/provider-data/sites/default/files/archive"
    folder = f"Nursing%20homes%20including%20rehab%20services/{year}"
    # Observed CMS naming: 2019 uses nh_archive_MM_YYYY.zip; newer uses nursing_homes_including_rehab_services_MM_YYYY.zip
    if year <= 2019:
        zip_name = f"nh_archive_{month:02d}_{year}.zip"
    else:
        zip_name = f"nursing_homes_including_rehab_services_{month:02d}_{year}.zip"
    return f"{base}/{folder}/{zip_name}"

def _quarter_from_nurse_filename(path):
    """Parse quarter from nurse file name, e.g. PBJ_dailynursestaffing_CY2025Q3.csv -> 'CY2025Q3'."""
    m = re.search(r'CY(\d{4})Q(\d)', os.path.basename(path), re.IGNORECASE)
    return f'CY{m.group(1)}Q{m.group(2)}' if m else None

def _normalize_cy_qtr(val):
    """Normalize CY_Qtr (e.g. 2025Q3 or CY2025Q3) to canonical 'CYyyyyQn' for comparison with nurse filenames."""
    if pd.isna(val):
        return None
    s = str(val).strip().upper().replace('\ufeff', '')
    m = re.search(r'(?:CY)?(\d{4})Q([1-4])', s)
    if m:
        y, q = m.group(1), m.group(2)
        return f'CY{y}Q{q}'
    # Fallback: integer like 20171 -> 2017Q1
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        i = int(val)
        if 20101 <= i <= 20304 and (i % 10) in (1, 2, 3, 4):
            y, q = str(i // 10), str(i % 10)
            return f'CY{y}Q{q}'
    return None

def create_facility_complete_csv(provnum, existing_csv_path=None, output_path=None):
    """Extract all data for any facility into one CSV. If existing_csv_path is provided and exists,
    only process nurse files for quarters not already in that CSV (incremental update)."""
    provnum = str(provnum).strip()
    nurse_files = glob.glob('standardized_PBJ/PBJ_dailynursestaffing_*.csv')
    nurse_files.sort()

    existing_df = None
    quarters_to_skip = set()
    if existing_csv_path:
        existing_csv_abs = os.path.abspath(existing_csv_path)
        if not os.path.exists(existing_csv_abs):
            print(f"[incremental] CSV not found at {existing_csv_abs}; doing full build.")
        else:
            try:
                existing_df = pd.read_csv(existing_csv_abs, low_memory=False)
                # Normalize column names (BOM/whitespace) so we find CY_Qtr
                existing_df.columns = [str(c).strip().replace('\ufeff', '') for c in existing_df.columns]
                cy_qtr_col = next((c for c in existing_df.columns if c == 'CY_Qtr' or c.upper() == 'CY_QTR'), None)
                if not cy_qtr_col:
                    print(f"[incremental] No CY_Qtr column in {existing_csv_abs} (columns: {list(existing_df.columns)[:8]}...); doing full build.")
                else:
                    for q in existing_df[cy_qtr_col].dropna().unique():
                        nq = _normalize_cy_qtr(q)
                        if nq:
                            quarters_to_skip.add(nq)
                    if quarters_to_skip:
                        q_list = sorted(q for q in quarters_to_skip if q)
                        print(f"Incremental update for facility {provnum}: existing quarters {q_list}; adding new quarters only.")
                    else:
                        # File has data but quarter format not recognized: do NOT full rebuild
                        if len(existing_df) > 0:
                            print(f"[incremental] No quarters parsed from CY_Qtr; existing file has {len(existing_df)} rows — treating as up to date (skipping full rebuild).")
                            out = output_path or existing_csv_path or f'facility_{provnum}_complete_data.csv'
                            existing_df.to_csv(out, index=False)
                            return existing_df
                        print(f"[incremental] No quarters parsed from CY_Qtr; doing full build.")
            except Exception as e:
                print(f"Could not read existing CSV for incremental: {e}; doing full build.")
                existing_df = None
                quarters_to_skip = set()

    files_to_process = []
    for path in nurse_files:
        q = _quarter_from_nurse_filename(path)
        if q and q in quarters_to_skip:
            continue
        files_to_process.append(path)

    if existing_df is not None and len(files_to_process) == 0:
        print(f"Facility {provnum} complete data already up to date (no new quarters).")
        out = output_path or existing_csv_path or f'facility_{provnum}_complete_data.csv'
        existing_df.to_csv(out, index=False)
        return existing_df

    if not quarters_to_skip:
        print(f"Creating comprehensive CSV for facility {provnum}...")
    else:
        print(f"Adding {len(files_to_process)} quarter(s) for facility {provnum}...")

    search_variants = list(
        dict.fromkeys(_normalize_provnum_like(v) for v in provnum_search_variants(provnum))
    )

    all_data: list[pd.DataFrame] = [] if existing_df is None else [existing_df]
    total_new = 0

    for file_path in files_to_process:
        chunk_size = 120_000
        min_chunk = 30_000
        processed = False
        last_err: Optional[BaseException] = None
        while chunk_size >= min_chunk and not processed:
            try:
                index_data = _load_or_build_provnum_chunk_index(file_path, chunksize=chunk_size)
                target_ids = select_targeted_chunk_ids(index_data, provnum, neighbor_margin=1)
                file_parts: list[pd.DataFrame] = []
                matched_rows = 0
                did_targeted = bool(target_ids)
                for pass_idx in (1, 2):
                    rerun_full = pass_idx == 2
                    chunk_idx = 0
                    for df_chunk in pd.read_csv(
                        file_path,
                        low_memory=False,
                        dtype={"PROVNUM": str},
                        chunksize=chunk_size,
                    ):
                        chunk_idx += 1
                        if did_targeted and not rerun_full and chunk_idx not in target_ids:
                            continue
                        df_chunk.columns = [str(c).strip().replace('\ufeff', '') for c in df_chunk.columns]
                        df_chunk = coerce_provnum_column(df_chunk)
                        if 'PROVNUM' not in df_chunk.columns:
                            continue
                        df_chunk['PROVNUM'] = df_chunk['PROVNUM'].astype(str).map(_normalize_provnum_like)
                        facility_data = pd.DataFrame(df_chunk[df_chunk['PROVNUM'].isin(search_variants)].copy())
                        if len(facility_data):
                            matched_rows += len(facility_data)
                            file_parts.append(facility_data)
                    if not did_targeted:
                        break
                    if rerun_full or matched_rows > 0:
                        break
                    print(f"  {os.path.basename(file_path)}: targeted window missed; retrying full scan for safety")
                if file_parts:
                    merged = pd.concat(file_parts, ignore_index=True)
                    print(f"  {os.path.basename(file_path)}: {len(merged)} records")
                    all_data.append(merged)
                    total_new += len(merged)
                processed = True
            except Exception as e:
                last_err = e
                if pandas_chunk_read_memory_error(e):
                    invalidate_provnum_chunk_index_cache(file_path)
                    chunk_size //= 2
                    print(
                        f"  {os.path.basename(file_path)}: memory pressure while reading; "
                        f"retrying with chunksize={chunk_size}",
                    )
                    continue
                print(f"Error processing {file_path}: {str(e)}")
                processed = True
        if not processed and last_err is not None:
            print(f"Error processing {file_path} (gave up after smaller chunks): {str(last_err)}")

    if existing_df is None and len(all_data) == 0:
        print(f"No data found for facility {provnum}!")
        return None

    combined_data = pd.concat(all_data, ignore_index=True)
    # Dedupe by WorkDate + PROVNUM (same facility/day can appear in multiple files)
    key_cols = [c for c in ('WorkDate', 'PROVNUM') if c in combined_data.columns]
    if key_cols:
        combined_data = combined_data.drop_duplicates(subset=key_cols, keep='last')
    combined_data = combined_data.sort_values('WorkDate')

    out = output_path or existing_csv_path or os.path.join(os.getcwd(), f'facility_{provnum}_complete_data.csv')
    combined_data.to_csv(out, index=False)

    print(f"\nExtraction Summary:")
    print(f"Total records: {len(combined_data)}")
    print(f"Date range: {combined_data['WorkDate'].min()} to {combined_data['WorkDate'].max()}")
    if 'CY_Qtr' in combined_data.columns:
        print(f"Quarters: {combined_data['CY_Qtr'].nunique()}")
    print(f"File saved as: {out}")
    return combined_data


def _provider_info_combined_csv_path() -> Optional[str]:
    """Resolve ``provider_info_combined.csv`` without relying on cwd (Vercel / PyCharm cwd varies)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(here, "..", "..", "provider_info_combined.csv")),
        os.path.join(here, "provider_info_combined.csv"),
        os.path.join(os.getcwd(), "provider_info_combined.csv"),
        "provider_info_combined.csv",
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    return None


def create_facility_provider_info_csv(provnum, existing_csv_path=None, output_path=None):
    """Extract provider info data for facility.

    When ``provider_info_combined.csv`` is found, the full CCN slice is used so columns such as
    ``nursing_case_mix_index`` backfill on older quarters. Otherwise scans ``provider_info_normalized/``
    and may append only rows newer than the latest ``processing_date`` in an existing CSV.
    """
    provnum = str(provnum).strip()
    search_variants = [provnum.upper()]
    if provnum.isdigit():
        search_variants.extend([provnum.zfill(6), provnum.lstrip('0')])

    existing_df = None
    max_date = None
    if existing_csv_path:
        existing_csv_abs = os.path.abspath(existing_csv_path)
        if os.path.exists(existing_csv_abs):
            try:
                existing_df = pd.read_csv(existing_csv_abs, low_memory=False, dtype={'ccn': str})
                if 'processing_date' in existing_df.columns:
                    existing_df['processing_date'] = pd.to_datetime(existing_df['processing_date'], errors='coerce')
                    max_date = existing_df['processing_date'].max()
                if max_date is not None and not pd.isna(max_date):
                    print(f"Incremental provider info for facility {provnum}: existing up to {max_date}; adding newer only.")
            except Exception as e:
                print(f"Could not read existing provider CSV for incremental: {e}; doing full build.")
                existing_df = None
                max_date = None

    def filter_facility(df_in):
        df_in = df_in.copy()
        df_in['ccn'] = df_in['ccn'].astype(str)
        df_in['ccn'] = df_in['ccn'].apply(lambda x: x.zfill(6) if x.isdigit() else x.upper())
        return df_in[df_in['ccn'].isin(search_variants)]

    facility_from_combined: Optional[pd.DataFrame] = None
    combined_path = _provider_info_combined_csv_path()
    if combined_path:
        try:
            print(f"Loading facility slice from {combined_path} (chunked CCN filter)...")
            t0_pi = time.perf_counter()
            slice_df, rows_scanned = _read_provider_combined_facility_slice(
                combined_path, search_variants
            )
            elapsed_pi_ms = int((time.perf_counter() - t0_pi) * 1000)
            if slice_df is not None and len(slice_df) > 0 and "processing_date" in slice_df.columns:
                slice_df = slice_df.copy()
                slice_df["processing_date"] = pd.to_datetime(slice_df["processing_date"], errors="coerce")
                facility_from_combined = slice_df.sort_values("processing_date")
                print(
                    f"  Loaded {len(facility_from_combined)} row(s) for this CCN from combined "
                    f"(chunked; scanned {rows_scanned} rows in {elapsed_pi_ms}ms)."
                )
            elif slice_df is not None and len(slice_df) == 0:
                print(
                    f"  No combined rows for CCN after chunked scan "
                    f"(scanned {rows_scanned} rows in {elapsed_pi_ms}ms)."
                )
        except Exception as e:
            print(f"Error loading from combined file: {e}")
            facility_from_combined = None

    if facility_from_combined is not None and len(facility_from_combined) > 0:
        combined_provider_df = facility_from_combined
    else:
        facility_data = None
        provider_files = glob.glob("provider_info_normalized/ProviderInfoNorm_*.csv")
        provider_files.sort()
        all_provider_data: list[pd.DataFrame] = []
        for file_path in provider_files:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                chunk = filter_facility(df)
                if len(chunk) > 0:
                    if "processing_date" in chunk.columns:
                        chunk["processing_date"] = pd.to_datetime(chunk["processing_date"], errors="coerce")
                        if max_date is not None:
                            chunk = chunk[chunk["processing_date"] > max_date]
                    if len(chunk) > 0:
                        all_provider_data.append(pd.DataFrame(chunk))
            except Exception:
                continue
        if all_provider_data:
            facility_data = pd.concat(all_provider_data, ignore_index=True)
        else:
            facility_data = None

        if existing_df is not None:
            if facility_data is not None and len(facility_data) > 0:
                combined_provider_df = pd.concat(
                    [pd.DataFrame(existing_df), pd.DataFrame(facility_data)], ignore_index=True
                )
            else:
                print(f"Facility {provnum} provider info already up to date (no newer records).")
                combined_provider_df = existing_df
        else:
            if facility_data is None or len(facility_data) == 0:
                print(f"[ERROR] No provider info data found for facility {provnum}")
                return None
            combined_provider_df = facility_data

    if 'processing_date' in combined_provider_df.columns:
        combined_provider_df['processing_date'] = pd.to_datetime(combined_provider_df['processing_date'], errors='coerce')
        combined_provider_df = combined_provider_df.sort_values('processing_date')
    dedup_cols = [c for c in ('processing_date', 'ccn') if c in combined_provider_df.columns]
    if dedup_cols:
        combined_provider_df = combined_provider_df.drop_duplicates(subset=dedup_cols, keep='last')

    if output_path:
        combined_provider_df.to_csv(output_path, index=False)
        print(f"Provider info saved as: {output_path}")
    return combined_provider_df

def _csv_host_base_url() -> str | None:
    """Base URL for centralized CSV hosting (e.g. Vercel static site).

    If set, facility dashboards can load `facility_<provnum>_*.csv` via HTTPS instead
    of requiring the CSVs to be bundled with each facility deployment.
    """
    value = os.getenv("PBJ_CSV_HOST_BASE_URL", "").strip()
    return value.rstrip("/") if value else None

def _facility_csv_url(base_url: str, filename: str) -> str:
    return f"{base_url}/{filename.lstrip('/')}"


def _lazy_load_nonnurse_enabled() -> bool:
    """When true (default), non-nurse CSV loads on first API/tab use — not at cold start."""
    raw = str(os.environ.get("PBJ_LAZY_LOAD_NONNURSE", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def _reset_nonnurse_load_state() -> None:
    global nonnurse_df, _NONNURSE_CSV_PATH, _NONNURSE_LOAD_PROVNUM, _NONNURSE_LOAD_SCAN_ROOT
    global _NONNURSE_LOAD_FACILITY_FOLDER
    nonnurse_df = None
    _NONNURSE_CSV_PATH = None
    _NONNURSE_LOAD_PROVNUM = None
    _NONNURSE_LOAD_SCAN_ROOT = None
    _NONNURSE_LOAD_FACILITY_FOLDER = None
    try:
        from nonnurse_staffing_lib import clear_nonnurse_prepare_cache

        clear_nonnurse_prepare_cache()
    except Exception:
        pass


def _nonnurse_path_is_usable(path: Optional[str]) -> bool:
    if not path or not str(path).strip():
        return False
    ps = str(path).strip()
    if ps.startswith("http://") or ps.startswith("https://"):
        return True
    return os.path.isfile(ps)


def _ensure_nonnurse_loaded() -> bool:
    """Load non-nurse daily CSV on first use (skipped at cold start when lazy load is enabled)."""
    global nonnurse_df
    if not INCLUDE_NONNURSE:
        return False
    if nonnurse_df is not None and not nonnurse_df.empty:
        return True
    with _NONNURSE_LOAD_LOCK:
        if nonnurse_df is not None and not nonnurse_df.empty:
            return True
        prov = str(_NONNURSE_LOAD_PROVNUM or PROVNUM or "").strip().zfill(6)
        nn_filename = f"facility_{prov}_nonnurse_daily.csv"
        path = _NONNURSE_CSV_PATH
        folder = _NONNURSE_LOAD_FACILITY_FOLDER
        scan_root = _NONNURSE_LOAD_SCAN_ROOT
        try:
            if not _nonnurse_path_is_usable(path):
                if folder:
                    candidate = os.path.join(str(folder), nn_filename)
                    if _nonnurse_path_is_usable(candidate):
                        path = candidate
            if not _nonnurse_path_is_usable(path):
                from nonnurse_staffing_lib import create_facility_nonnurse_csv

                out_path = path or (os.path.join(str(folder), nn_filename) if folder else os.path.join(_app_root, nn_filename))
                create_facility_nonnurse_csv(
                    prov,
                    output_path=out_path,
                    root=scan_root,
                    verbose=False,
                )
                path = out_path
            if not _nonnurse_path_is_usable(path):
                return False
            if str(path).startswith("http://") or str(path).startswith("https://"):
                loaded = pd.read_csv(str(path), low_memory=False, dtype={"PROVNUM": str})
            else:
                loaded = pd.read_csv(str(path), low_memory=False, dtype={"PROVNUM": str})
            if loaded is not None and len(loaded) > 0:
                nonnurse_df = loaded
                from nonnurse_staffing_lib import clear_nonnurse_prepare_cache

                clear_nonnurse_prepare_cache()
                print(f"[OK] Loaded {len(nonnurse_df)} non-nurse daily records (on demand)", flush=True)
                return True
        except Exception as exc:
            print(f"[NON-NURSE] On-demand load failed: {exc}", flush=True)
            nonnurse_df = None
        return False


def create_dynamic_dashboard(provnum):
    """Create and initialize the dynamic dashboard for a specific facility"""
    global global_df, nonnurse_df, citations_df, provider_info_df, provider_info_loaded_source
    global _NONNURSE_CSV_PATH, _NONNURSE_LOAD_PROVNUM, _NONNURSE_LOAD_SCAN_ROOT, _NONNURSE_LOAD_FACILITY_FOLDER
    import shutil
    from file_path_utils import (
        find_facility_complete_data,
        find_facility_citations,
        find_facility_nonnurse_daily,
        find_facility_provider_info,
        get_facility_folder,
    )

    provnum = str(provnum).strip().zfill(6)
    csv_host = _csv_host_base_url()
    csv_filename = f'facility_{provnum}_complete_data.csv'
    provider_filename = f'facility_{provnum}_provider_info_data.csv'
    nonnurse_filename = f"facility_{provnum}_nonnurse_daily.csv"
    citations_filename = f"facility_{provnum}_citations.csv"

    # CSV lives next to this script (Vercel) or in cwd, or use file_path_utils (local project with deployments/)
    _app_dir = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    csv_same_dir = os.path.join(_app_dir, csv_filename)
    provider_same_dir = os.path.join(_app_dir, provider_filename)
    nonnurse_same_dir = os.path.join(_app_dir, nonnurse_filename)
    citations_same_dir = os.path.join(_app_dir, citations_filename)
    csv_cwd = os.path.join(cwd, csv_filename)
    provider_cwd = os.path.join(cwd, provider_filename)
    nonnurse_cwd = os.path.join(cwd, nonnurse_filename)
    citations_cwd = os.path.join(cwd, citations_filename)

    if os.path.exists(csv_same_dir):
        csv_file = csv_same_dir
    elif os.path.exists(csv_cwd):
        csv_file = csv_cwd
    else:
        csv_file = find_facility_complete_data(provnum)
        if (not csv_file or not os.path.exists(csv_file)) and csv_host:
            csv_file = _facility_csv_url(csv_host, csv_filename)
    if os.path.exists(provider_same_dir):
        provider_csv_file = provider_same_dir
    elif os.path.exists(provider_cwd):
        provider_csv_file = provider_cwd
    else:
        provider_csv_file = find_facility_provider_info(provnum)
        if (not provider_csv_file or not os.path.exists(provider_csv_file)) and csv_host:
            provider_csv_file = _facility_csv_url(csv_host, provider_filename)

    nonnurse_csv_file = None
    if INCLUDE_NONNURSE:
        if os.path.exists(nonnurse_same_dir):
            nonnurse_csv_file = nonnurse_same_dir
        elif os.path.exists(nonnurse_cwd):
            nonnurse_csv_file = nonnurse_cwd
        else:
            nonnurse_csv_file = find_facility_nonnurse_daily(provnum)
            if (not nonnurse_csv_file or not os.path.exists(nonnurse_csv_file)) and csv_host:
                nonnurse_csv_file = _facility_csv_url(csv_host, nonnurse_filename)

    if os.path.exists(citations_same_dir):
        citations_csv_file = citations_same_dir
    elif os.path.exists(citations_cwd):
        citations_csv_file = citations_cwd
    else:
        citations_csv_file = find_facility_citations(provnum)
        if (not citations_csv_file or not os.path.exists(citations_csv_file)) and csv_host:
            citations_csv_file = _facility_csv_url(csv_host, citations_filename)

    # Vercel bundles ship CSVs next to the entrypoint; never mkdir under repo deployments/ there.
    from pathlib import Path as _Path

    _bundle_root = _Path(_app_dir)
    _has_bundle_csv = (_bundle_root / csv_filename).is_file()
    _use_repo_deployments_layout = (
        not _has_bundle_csv
        and ("deployments" in _app_dir or os.path.exists(os.path.join(cwd, "deployments")))
    )
    facility_folder = (
        get_facility_folder(provnum) if _use_repo_deployments_layout else _bundle_root
    )

    if not csv_file or (not (isinstance(csv_file, str) and (csv_file.startswith("http://") or csv_file.startswith("https://"))) and not os.path.exists(csv_file)):
        csv_file = str(facility_folder / csv_filename)
        print(f"Creating CSV for facility {provnum}...")
        create_facility_complete_csv(provnum)
        # Check if file was created in root, move it to facility folder
        root_csv = csv_filename
        if os.path.exists(root_csv) and not os.path.exists(csv_file):
            shutil.move(root_csv, csv_file)

    if not provider_csv_file or (not (isinstance(provider_csv_file, str) and (provider_csv_file.startswith("http://") or provider_csv_file.startswith("https://"))) and not os.path.exists(provider_csv_file)):
        provider_csv_file = str(facility_folder / provider_filename)
        print(f"Creating provider info CSV for facility {provnum}...")
        provider_data = create_facility_provider_info_csv(provnum)
        if provider_data is not None:
            provider_data.to_csv(provider_csv_file, index=False)
            # Check if file was created in root, move it to facility folder
            root_provider = provider_filename
            if os.path.exists(root_provider) and not os.path.exists(provider_csv_file):
                shutil.move(root_provider, provider_csv_file)
            print(f"Provider info CSV saved as: {os.path.basename(provider_csv_file)}")
    
    # Load the facility data (pass path so Vercel uses script-dir file)
    global_df = load_facility_data(provnum, csv_file)
    _clear_filtered_df_cache()
    
    # Load MACPAC state standards
    load_macpac_standards()
    
    # Load the provider info data
    global provider_info_df, provider_info_loaded_source
    provider_info_loaded_source = None
    if provider_csv_file and ((isinstance(provider_csv_file, str) and (provider_csv_file.startswith("http://") or provider_csv_file.startswith("https://"))) or os.path.exists(provider_csv_file)):
        try:
            provider_info_df = pd.read_csv(provider_csv_file, low_memory=False, dtype={'ccn': str})
            # Strip BOM/whitespace so row Series labels match ``nursing_case_mix_index`` lookups (CMI extraction).
            normalize_provider_info_csv_columns(provider_info_df)
            # Format CCN to ensure consistency
            provider_info_df['ccn'] = provider_info_df['ccn'].astype(str).str.zfill(6)
            provider_info_df['processing_date'] = pd.to_datetime(provider_info_df['processing_date'], errors='coerce')
            print(f"[OK] Loaded {len(provider_info_df)} provider info records")
            print(f"   CCN values: {provider_info_df['ccn'].unique()[:5]}")
            if 'quarter' in provider_info_df.columns:
                quarter_count = provider_info_df['quarter'].notna().sum()
                print(f"   Records with quarter: {quarter_count}")
                if quarter_count > 0:
                    sample_quarters = provider_info_df['quarter'].dropna().unique()[:5]
                    print(f"   Sample quarters: {list(sample_quarters)}")
            if 'sff_status' in provider_info_df.columns:
                sff_count = provider_info_df['sff_status'].notna().sum()
                print(f"   Records with SFF status: {sff_count}")
            # Check for CMI column
            cmi_columns = ['case_mix_index', 'CMI', 'Case Mix Index', 'case_mix', 'Case-Mix Index', 'Case Mix Index (CMI)', 'nursing_case_mix_index']
            found_cmi = False
            for col in cmi_columns:
                if col in provider_info_df.columns:
                    cmi_count = provider_info_df[col].notna().sum()
                    if cmi_count > 0:
                        print(f"   Found CMI column '{col}' with {cmi_count} non-null values")
                        found_cmi = True
                        break
            if not found_cmi:
                print(f"   [WARN] No CMI column found in provider info data")
            pcs = provider_csv_file
            if isinstance(pcs, str) and (pcs.startswith("http://") or pcs.startswith("https://")):
                provider_info_loaded_source = pcs.split("/")[-1].split("?")[0] or "remote CSV"
            else:
                provider_info_loaded_source = os.path.basename(str(pcs))
        except Exception as e:
            print(f"Error loading provider info data: {e}")
            import traceback
            traceback.print_exc()
            provider_info_df = None
            provider_info_loaded_source = None
    else:
        provider_info_df = None
        provider_info_loaded_source = None
        _pcs = provider_csv_file
        if not _pcs:
            _msg = "no provider_csv_file path resolved"
        else:
            sp = str(_pcs)
            if sp.startswith("http://") or sp.startswith("https://"):
                _msg = f"URL resolved but load branch skipped (unexpected): {sp[:160]}"
            elif not os.path.exists(sp):
                _msg = f"file not found: {sp}"
            else:
                _msg = f"path exists but load condition failed: {sp}"
            print(
            f"[PROVIDER INFO] Not loaded — {_msg}. "
            f"Expected `{provider_filename}` next to the Flask app or under the facility folder."
        )

    _reset_nonnurse_load_state()
    nonnurse_df = None
    if INCLUDE_NONNURSE:
        nn_path = nonnurse_csv_file
        if not nn_path:
            nn_path = str(facility_folder / nonnurse_filename)
        elif isinstance(nn_path, str) and not nn_path.startswith("http") and not os.path.exists(nn_path):
            nn_path = str(facility_folder / nonnurse_filename)
        dep_parent = os.path.dirname(_app_dir)
        nonnurse_scan_root = None
        if os.path.basename(dep_parent) == "deployments":
            candidate = os.path.abspath(os.path.join(dep_parent, ".."))
            nn_sub = os.path.join(candidate, "standardized_NonNurse")
            if os.path.isdir(nn_sub):
                nonnurse_scan_root = candidate
        _NONNURSE_CSV_PATH = str(nn_path) if nn_path else str(facility_folder / nonnurse_filename)
        _NONNURSE_LOAD_PROVNUM = provnum
        _NONNURSE_LOAD_FACILITY_FOLDER = str(facility_folder)
        _NONNURSE_LOAD_SCAN_ROOT = nonnurse_scan_root
        if not _lazy_load_nonnurse_enabled():
            try:
                if _ensure_nonnurse_loaded():
                    pass
                else:
                    print("[NON-NURSE] Eager load enabled but no non-nurse daily rows found.")
            except Exception as e:
                print(f"[NON-NURSE] Not loaded: {e}")
                nonnurse_df = None
        elif _nonnurse_path_is_usable(_NONNURSE_CSV_PATH):
            print(f"[NON-NURSE] Deferred load (lazy): {os.path.basename(str(_NONNURSE_CSV_PATH))}", flush=True)
        else:
            print("[NON-NURSE] Deferred load (lazy): file will be built on first use if source quarters exist.", flush=True)
    else:
        print("[NON-NURSE] Skipped (INCLUDE_NONNURSE=False in this deployment).")

    citations_df = None
    cit_path = citations_csv_file
    if not cit_path:
        cit_path = str(facility_folder / citations_filename)
    elif isinstance(cit_path, str) and not cit_path.startswith("http") and not os.path.exists(cit_path):
        cit_path = str(facility_folder / citations_filename)
    try:
        if isinstance(cit_path, str) and (cit_path.startswith("http://") or cit_path.startswith("https://")):
            citations_df = pd.read_csv(cit_path, low_memory=False, dtype=str)
        else:
            from citation_lib import build_facility_citations_csv

            build_facility_citations_csv(provnum, cit_path)
            if os.path.exists(cit_path):
                citations_df = pd.read_csv(cit_path, low_memory=False, dtype=str)
        if citations_df is not None and len(citations_df) > 0:
            print(f"[OK] Loaded {len(citations_df)} citation record(s)")
    except Exception as e:
        print(f"[CITATIONS] Not loaded: {e}")
        citations_df = None

    globals()["nonnurse_df"] = nonnurse_df
    globals()["citations_df"] = citations_df
    if isinstance(cit_path, str) and cit_path and not (
        cit_path.startswith("http://") or cit_path.startswith("https://")
    ):
        globals()["citations_loaded_path"] = cit_path if os.path.isfile(cit_path) else None
    else:
        globals()["citations_loaded_path"] = None

    if global_df is None:
        print(f"Failed to load data for facility {provnum}")
        return None

    print("[EIN] Loading employee / position extracts (may take a minute)...", flush=True)
    _load_ein_position_csvs(provnum)
    print("[EIN] Done.", flush=True)
    return app

def initialize_data():
    """Initialize the global data variable"""
    global global_df
    # This function is not used in the current implementation
    # Data is loaded in create_dynamic_dashboard()
    return global_df

def round_financial(value, decimals=2):
    """Round using financial rounding (ROUND_HALF_UP)"""
    if pd.isna(value) or value is None:
        return 0.0
    return float(Decimal(str(value)).quantize(Decimal('0.' + '0' * decimals), rounding=ROUND_HALF_UP))


def _as_1d_series(x: Any) -> pd.Series:
    """Coerce ``df[col]`` / arithmetic results to a 1-D Series (avoids DataFrame stubs when columns alias)."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError("Expected a single column for HPRD math")
        return x.iloc[:, 0]
    if isinstance(x, pd.Series):
        return x
    return pd.Series(x)


def _divide_hprd(numer: Any, census: Any) -> pd.Series:
    """Hours per resident day; NaN when census is missing, ≤ 0, or numerator is missing."""
    c = pd.to_numeric(_as_1d_series(census), errors="coerce")
    n = pd.to_numeric(_as_1d_series(numer), errors="coerce")
    out = n / c
    out = out.mask(~(c.notna() & (c > 0) & n.notna()))
    return out.replace([np.inf, -np.inf], np.nan)


def _round_fin_series(s: Any, decimals: int = 2) -> pd.Series:
    """Financial rounding; preserves NaN (unlike ``round_financial``, which maps NaN → 0)."""
    ser = _as_1d_series(s)

    def _one(v: object) -> float:
        if v is None:
            return float("nan")
        try:
            if pd.isna(v):
                return float("nan")
        except (TypeError, ValueError):
            return float("nan")
        return round_financial(cast(Any, v), decimals)

    return pd.Series([_one(v) for v in ser.tolist()], index=ser.index, dtype="float64")


def _contract_pct_series(contract_hrs: Any, direct_hrs: Any) -> pd.Series:
    """Contract % = 100 * contract / direct; NaN when direct is missing or ≤ 0."""
    c = pd.to_numeric(_as_1d_series(contract_hrs), errors="coerce")
    d = pd.to_numeric(_as_1d_series(direct_hrs), errors="coerce")
    out = (c / d * 100.0).where(d.notna() & (d > 0) & c.notna())
    return out.replace([np.inf, -np.inf], np.nan)


def _hprd_quality_block(mdc_series: Any, total_days: int) -> dict[str, Any]:
    _one_col = _as_1d_series(mdc_series)
    m = pd.to_numeric(_one_col, errors="coerce") if len(_one_col) else pd.Series(dtype=float)
    return {
        "days_in_filter": int(total_days),
        "days_hprd_denominator_ok": int((m.notna() & (m > 0)).sum()),
        "days_excluded_missing_census": int(m.isna().sum()),
        "days_excluded_zero_census": int((m.notna() & (m == 0)).sum()),
        "pooled_hprd_note": (
            "Pooled HPRD uses sum(hours)/sum(census) on days with census > 0 and non-missing hours "
            "for that line (missing days do not count as zero hours)."
        ),
    }


def _pbj_filtered_workdate_yyyymmdd_bounds(pbj_df: pd.DataFrame) -> tuple[int | None, int | None]:
    """Min/max WorkDate from a filtered PBJ daily frame as YYYYMMDD ints for EIN alignment."""
    if pbj_df is None or len(pbj_df) == 0 or "WorkDate" not in pbj_df.columns:
        return (None, None)
    ts = pd.to_datetime(pbj_df["WorkDate"], errors="coerce")
    lo = ts.min()
    hi = ts.max()
    if pd.isna(lo) or pd.isna(hi):
        return (None, None)
    return (int(cast(pd.Timestamp, lo).strftime("%Y%m%d")), int(cast(pd.Timestamp, hi).strftime("%Y%m%d")))


def _json_float_list(vals: Sequence[Any]) -> list[Any]:
    """JSON-safe list: NaN/Inf → None for Plotly gaps."""
    out: list[Any] = []
    for v in vals:
        if v is None:
            out.append(None)
            continue
        try:
            if pd.isna(v):
                out.append(None)
            elif isinstance(v, (float, np.floating)) and (np.isnan(float(v)) or np.isinf(float(v))):
                out.append(None)
            else:
                out.append(float(v))
        except (TypeError, ValueError):
            out.append(None)
    return out


def _mean_or_none(series: Any) -> Optional[float]:
    ser = _as_1d_series(series)
    if len(ser) == 0:
        return None
    v = pd.to_numeric(ser, errors="coerce").mean()
    if pd.isna(v):
        return None
    return float(cast(Any, v))


def _weighted_summary_total_contract_pct(filtered_df: pd.DataFrame) -> Optional[float]:
    """Filter-level total contract % = sum(Hrs_RN_ctr+Hrs_LPN_ctr+Hrs_CNA_ctr) / sum(Hrs_RN+Hrs_LPN+Hrs_CNA) × 100.

    Direct-care RN, LPN, and CNA line hours only (excludes admin/DON, NA trainee, med aide).
    Returns None when the denominator sum is ≤ 0 or required columns are missing.
    """
    if len(filtered_df) == 0:
        return None
    ctr_cols = ("Hrs_RN_ctr", "Hrs_LPN_ctr", "Hrs_CNA_ctr")
    dir_cols = ("Hrs_RN", "Hrs_LPN", "Hrs_CNA")
    if not all(c in filtered_df.columns for c in ctr_cols + dir_cols):
        return None
    contract_sum = float(
        cast(Any, pd.to_numeric(filtered_df["Hrs_RN_ctr"], errors="coerce").fillna(0).sum())
        + cast(Any, pd.to_numeric(filtered_df["Hrs_LPN_ctr"], errors="coerce").fillna(0).sum())
        + cast(Any, pd.to_numeric(filtered_df["Hrs_CNA_ctr"], errors="coerce").fillna(0).sum())
    )
    direct_sum = float(
        cast(Any, pd.to_numeric(filtered_df["Hrs_RN"], errors="coerce").fillna(0).sum())
        + cast(Any, pd.to_numeric(filtered_df["Hrs_LPN"], errors="coerce").fillna(0).sum())
        + cast(Any, pd.to_numeric(filtered_df["Hrs_CNA"], errors="coerce").fillna(0).sum())
    )
    if direct_sum <= 0:
        return None
    return float(contract_sum / direct_sum * 100.0)


def _pooled_hprd_hours_ratio(filtered_df: pd.DataFrame, hours_col: str) -> Optional[float]:
    if len(filtered_df) == 0 or "MDScensus" not in filtered_df.columns or hours_col not in filtered_df.columns:
        return None
    mdc = cast(
        pd.Series,
        pd.to_numeric(_as_1d_series(filtered_df["MDScensus"]), errors="coerce"),
    )
    hrs = cast(
        pd.Series,
        pd.to_numeric(_as_1d_series(filtered_df[hours_col]), errors="coerce"),
    )
    ok = mdc.notna() & (mdc > 0) & hrs.notna()
    den = float(cast(Any, mdc[ok].sum()))
    if den <= 0:
        return None
    return float(cast(Any, hrs[ok].sum()) / den)


def _resolve_pbj_lite_csv(filename: str) -> Optional[str]:
    """Find bundled lite/quarterly CSVs under app root, ``pbj_lite/``, or ``data/geo/``."""
    roots = [_app_root, os.path.abspath(os.path.join(_app_root, "..", ".."))]
    rels = (
        filename,
        os.path.join("pbj_lite", filename),
        os.path.join("data", "geo", filename),
    )
    for root in roots:
        for rel in rels:
            p = os.path.join(root, rel)
            if os.path.isfile(p):
                return p
    return None


_PBJ_LITE_CSV_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}


def _read_pbj_lite_csv_cached(filename: str) -> Optional[pd.DataFrame]:
    """Read a bundled lite metrics CSV once per process (invalidates on file mtime change)."""
    path = _resolve_pbj_lite_csv(filename)
    if not path:
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    cached = _PBJ_LITE_CSV_CACHE.get(filename)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] {filename}: {exc}")
        return None
    _PBJ_LITE_CSV_CACHE[filename] = (mtime, df)
    return df


def _cy_qtr_sort_key_lite(q: object) -> tuple[int, int]:
    if pd.isna(q):
        return (0, 0)
    m = re.search(r"(\d{4})Q([1-4])", str(q).upper())
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


def _geo_residual_lpn_hprd(
    total_hprd: Any,
    rn_hprd: Any,
    aide_hprd: Any,
) -> Optional[float]:
    """Published rollup LPN ≈ total nurse − RN − nurse aide (when all three exist)."""
    try:
        if total_hprd is None or rn_hprd is None or aide_hprd is None:
            return None
        if pd.isna(total_hprd) or pd.isna(rn_hprd) or pd.isna(aide_hprd):
            return None
        residual = float(total_hprd) - float(rn_hprd) - float(aide_hprd)
        if residual < 0:
            return None
        return round_financial(residual, 3)
    except (TypeError, ValueError):
        return None


def _geo_residual_lpn_care_hprd(
    nurse_care: Any,
    rn_care: Any,
    aide_hprd: Any,
) -> Optional[float]:
    """Direct-care LPN residual from published care HPRD rollups."""
    try:
        if nurse_care is None or rn_care is None or aide_hprd is None:
            return None
        if pd.isna(nurse_care) or pd.isna(rn_care) or pd.isna(aide_hprd):
            return None
        residual = float(nurse_care) - float(rn_care) - float(aide_hprd)
        if residual < 0:
            return None
        return round_financial(residual, 3)
    except (TypeError, ValueError):
        return None


_GEO_ROLLUP_LPN_PREFIXES: tuple[str, ...] = (
    "state",
    "national",
    "region",
    "county",
    "county_locale",
    "state_rural",
    "state_urban",
    "region_rural",
    "region_urban",
)


def _enrich_geo_rollup_entry_lpn(entry: dict[str, Any]) -> None:
    """Add LPN HPRD fields derived from bundled geography rollups."""
    for prefix in _GEO_ROLLUP_LPN_PREFIXES:
        total_k = f"{prefix}_total_nurse_hprd"
        rn_k = f"{prefix}_rn_hprd"
        aide_k = f"{prefix}_nurse_aide_hprd"
        care_k = f"{prefix}_nurse_care_hprd"
        rn_care_k = f"{prefix}_rn_care_hprd"
        total_v = entry.get(total_k)
        rn_v = entry.get(rn_k)
        if total_v is not None and rn_v is not None:
            aide_v = entry.get(aide_k)
            if aide_v is not None:
                entry[f"{prefix}_lpn_hprd"] = _geo_residual_lpn_hprd(total_v, rn_v, aide_v)
            else:
                try:
                    if not pd.isna(total_v) and not pd.isna(rn_v):
                        resid = float(total_v) - float(rn_v)
                        if resid >= 0:
                            entry[f"{prefix}_lpn_hprd"] = round_financial(resid, 3)
                except (TypeError, ValueError):
                    pass
        entry[f"{prefix}_lpn_care_hprd"] = _geo_residual_lpn_care_hprd(
            entry.get(care_k), entry.get(rn_care_k), entry.get(aide_k)
        )


def _hprd_benchmark_float(row: pd.Series, col: str) -> Optional[float]:
    if col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _hprd_benchmark_int(row: pd.Series, col: str) -> Optional[int]:
    if col not in row.index:
        return None
    v = row[col]
    if pd.isna(v):
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _load_quarterly_hprd_benchmarks(state_abbr: str) -> Optional[dict[str, Any]]:
    """
    Prefer full ``state_quarterly_metrics.csv`` + ``national_quarterly_metrics.csv`` (same CY_Qtr)
    and optional ``cms_region_quarterly_metrics.csv`` for CMS region peers.
    """
    state_abbr = (state_abbr or "").strip().upper()
    if len(state_abbr) != 2:
        return None
    st_path = _resolve_pbj_lite_csv("state_quarterly_metrics.csv")
    nat_path = _resolve_pbj_lite_csv("national_quarterly_metrics.csv")
    if not st_path or not nat_path:
        return None
    try:
        sdf = _read_pbj_lite_csv_cached("state_quarterly_metrics.csv")
        if sdf is None:
            return None
    except Exception as exc:
        print(f"[WARN] state_quarterly_metrics: {exc}")
        return None
    need = ("STATE", "CY_Qtr", "Total_Nurse_HPRD")
    if not all(c in sdf.columns for c in need):
        return None
    sub = sdf[sdf["STATE"].astype(str).str.strip().str.upper() == state_abbr].copy()
    if sub.empty:
        return None
    sub = sub.copy()
    sub["_qk"] = sub["CY_Qtr"].map(_cy_qtr_sort_key_lite)
    sub = sub.sort_values("_qk")
    last = sub.iloc[-1]
    qtr = str(last["CY_Qtr"])
    out: dict[str, Any] = {
        "quarter": qtr,
        "state_total_nurse_hprd": _hprd_benchmark_float(last, "Total_Nurse_HPRD"),
        "national_total_nurse_hprd": None,
        "state_nurse_care_hprd": _hprd_benchmark_float(last, "Nurse_Care_HPRD"),
        "national_nurse_care_hprd": None,
        "region_total_nurse_hprd": None,
        "region_nurse_care_hprd": None,
        "cms_region_number": None,
        "cms_region_name": None,
        "cms_region_full": None,
        "state_facility_count": _hprd_benchmark_int(last, "facility_count")
        or _hprd_benchmark_int(last, "Facility_Count"),
        "national_facility_count": None,
        "region_facility_count": None,
        "data_source": "quarterly",
    }
    if out["state_total_nurse_hprd"] is None:
        return None
    try:
        ndf = _read_pbj_lite_csv_cached("national_quarterly_metrics.csv")
        if ndf is None:
            return None
        if not all(c in ndf.columns for c in need):
            return None
        nsub = ndf[
            (ndf["STATE"].astype(str).str.strip().str.upper() == "NATIONAL")
            & (ndf["CY_Qtr"].astype(str) == qtr)
        ]
        if nsub.empty:
            return None
        nl = nsub.iloc[-1]
        out["national_total_nurse_hprd"] = _hprd_benchmark_float(nl, "Total_Nurse_HPRD")
        out["national_nurse_care_hprd"] = _hprd_benchmark_float(nl, "Nurse_Care_HPRD")
        out["national_facility_count"] = _hprd_benchmark_int(nl, "facility_count")
    except Exception as exc:
        print(f"[WARN] national_quarterly_metrics: {exc}")
        return None
    if out["national_total_nurse_hprd"] is None:
        return None

    map_path = _resolve_pbj_lite_csv("cms_region_state_mapping.csv")
    reg_path = _resolve_pbj_lite_csv("cms_region_quarterly_metrics.csv")
    if map_path and reg_path:
        try:
            mmap = _read_pbj_lite_csv_cached("cms_region_state_mapping.csv")
            if mmap is None or "State_Code" not in mmap.columns:
                return out
            mrow = mmap[mmap["State_Code"].astype(str).str.strip().str.upper() == state_abbr]
            if mrow.empty:
                return out
            rn_raw = mrow["CMS_Region_Number"].iloc[0]
            if pd.isna(rn_raw):
                return out
            rn_int = int(float(rn_raw))
            rname = mrow["CMS_Region_Name"].iloc[0] if "CMS_Region_Name" in mrow.columns else None
            rfull = mrow["CMS_Region_Full"].iloc[0] if "CMS_Region_Full" in mrow.columns else None
            rdf = _read_pbj_lite_csv_cached("cms_region_quarterly_metrics.csv")
            if rdf is None or "CMS_Region_Number" not in rdf.columns or "CY_Qtr" not in rdf.columns:
                return out
            rnum = pd.to_numeric(rdf["CMS_Region_Number"], errors="coerce")
            rsub = rdf[(rnum == rn_int) & (rdf["CY_Qtr"].astype(str) == qtr)]
            if rsub.empty:
                out["cms_region_number"] = rn_int
                out["cms_region_name"] = str(rname).strip() if pd.notna(rname) else None
                out["cms_region_full"] = str(rfull).strip() if pd.notna(rfull) else None
                return out
            rl = rsub.iloc[-1]
            out["cms_region_number"] = rn_int
            out["cms_region_name"] = str(rname).strip() if pd.notna(rname) else None
            out["cms_region_full"] = str(rfull).strip() if pd.notna(rfull) else None
            out["region_total_nurse_hprd"] = _hprd_benchmark_float(rl, "Total_Nurse_HPRD")
            out["region_nurse_care_hprd"] = _hprd_benchmark_float(rl, "Nurse_Care_HPRD")
            out["region_facility_count"] = _hprd_benchmark_int(rl, "facility_count")
        except Exception as exc:
            print(f"[WARN] cms_region_quarterly_metrics: {exc}")
    return out


def lite_hprd_benchmarks_for_state(state_abbr: str) -> dict[str, Any]:
    """
    Latest CY_Qtr: state / national / optional CMS region total nurse HPRD rollups for on-page display.

    Uses full ``state_quarterly_metrics.csv`` and ``national_quarterly_metrics.csv`` when present;
    otherwise falls back to ``state_lite_metrics.csv`` / ``national_lite_metrics.csv``.
    The Summary HPRD chart uses the MACPAC state minimum as its dashed reference when that value is meaningful.
    """
    empty: dict[str, Any] = {
        "quarter": None,
        "state_total_nurse_hprd": None,
        "national_total_nurse_hprd": None,
        "state_nurse_care_hprd": None,
        "national_nurse_care_hprd": None,
        "region_total_nurse_hprd": None,
        "region_nurse_care_hprd": None,
        "cms_region_number": None,
        "cms_region_name": None,
        "cms_region_full": None,
        "state_facility_count": None,
        "national_facility_count": None,
        "region_facility_count": None,
        "data_source": None,
    }
    quarterly = _load_quarterly_hprd_benchmarks(state_abbr)
    if quarterly:
        return quarterly

    out = dict(empty)
    state_abbr = (state_abbr or "").strip().upper()
    nat_path = _resolve_pbj_lite_csv("national_lite_metrics.csv")
    st_path = _resolve_pbj_lite_csv("state_lite_metrics.csv")
    try:
        if nat_path:
            ndf = _read_pbj_lite_csv_cached("national_lite_metrics.csv")
            if ndf is not None and "CY_Qtr" in ndf.columns and "Total_Nurse_HPRD" in ndf.columns:
                ndf = ndf.copy()
                ndf["_qk"] = ndf["CY_Qtr"].map(_cy_qtr_sort_key_lite)
                ndf = ndf.sort_values("_qk")
                last = ndf.iloc[-1]
                out["national_total_nurse_hprd"] = float(last["Total_Nurse_HPRD"])
                out["quarter"] = str(last["CY_Qtr"])
                out["national_nurse_care_hprd"] = _hprd_benchmark_float(last, "Nurse_Care_HPRD")
    except Exception as exc:
        print(f"[WARN] national_lite_metrics: {exc}")
    try:
        if st_path and state_abbr and len(state_abbr) == 2:
            sdf = _read_pbj_lite_csv_cached("state_lite_metrics.csv")
            if (
                sdf is not None
                and "STATE" in sdf.columns
                and "CY_Qtr" in sdf.columns
                and "Total_Nurse_HPRD" in sdf.columns
            ):
                sub = sdf[sdf["STATE"].astype(str).str.strip().str.upper() == state_abbr].copy()
                if not sub.empty:
                    sub = sub.copy()
                    sub["_qk"] = sub["CY_Qtr"].map(_cy_qtr_sort_key_lite)
                    sub = sub.sort_values("_qk")
                    last = sub.iloc[-1]
                    out["state_total_nurse_hprd"] = float(last["Total_Nurse_HPRD"])
                    out["quarter"] = str(last["CY_Qtr"])
                    out["state_nurse_care_hprd"] = _hprd_benchmark_float(last, "Nurse_Care_HPRD")
                    out["state_facility_count"] = _hprd_benchmark_int(last, "Facility_Count")
    except Exception as exc:
        print(f"[WARN] state_lite_metrics: {exc}")
    out["data_source"] = "lite"
    return out


_FACILITY_LITE_RAW_DF: Optional[pd.DataFrame] = None
_FACILITY_LITE_RAW_CACHE: tuple[str, float] | None = None

_FACILITY_LITE_METRICS_USECOLS = (
    "CY_Qtr",
    "PROVNUM",
    "STATE",
    "Total_Nurse_HPRD",
    "Nurse_Care_HPRD",
    "COUNTY_NAME",
    "Total_RN_HPRD",
    "Direct_Care_RN_HPRD",
    "Contract_Percentage",
    "Census",
)


def _load_facility_lite_metrics_raw() -> Optional[pd.DataFrame]:
    """Single read of ``facility_lite_metrics.csv`` for peer rank + geo peer slices."""
    global _FACILITY_LITE_RAW_DF, _FACILITY_LITE_RAW_CACHE
    path = _resolve_pbj_lite_csv("facility_lite_metrics.csv")
    if not path:
        _FACILITY_LITE_RAW_DF = None
        _FACILITY_LITE_RAW_CACHE = None
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if (
        _FACILITY_LITE_RAW_DF is not None
        and _FACILITY_LITE_RAW_CACHE is not None
        and _FACILITY_LITE_RAW_CACHE[0] == path
        and _FACILITY_LITE_RAW_CACHE[1] == mtime
    ):
        return _FACILITY_LITE_RAW_DF
    try:
        raw = pd.read_csv(path, usecols=cast(Any, list(_FACILITY_LITE_METRICS_USECOLS)), low_memory=False)
    except Exception as exc:
        print(f"[WARN] facility_lite_metrics: {exc}")
        return None
    need = {"CY_Qtr", "PROVNUM", "STATE", "Total_Nurse_HPRD", "Nurse_Care_HPRD"}
    if raw.empty or not need.issubset(set(raw.columns)):
        return None
    df = cast(pd.DataFrame, raw.copy())
    df["PROVNUM"] = df["PROVNUM"].astype(str).str.strip()
    df["PROVNUM"] = df["PROVNUM"].apply(lambda x: x.zfill(6) if str(x).isdigit() else str(x).upper())
    df["STATE"] = df["STATE"].astype(str).str.strip().str.upper()
    df["CY_Qtr"] = df["CY_Qtr"].astype(str).str.strip()
    if "COUNTY_NAME" in df.columns:
        df["_county_norm"] = df["COUNTY_NAME"].map(
            lambda n: re.sub(r"\s+", " ", str(n or "").strip()).lower()
        )
    _FACILITY_LITE_RAW_DF = df
    _FACILITY_LITE_RAW_CACHE = (path, mtime)
    return df


def _load_facility_lite_metrics_df() -> Optional[pd.DataFrame]:
    """In-memory cache of ``facility_lite_metrics.csv`` (subset of columns) for peer ranks."""
    return _load_facility_lite_metrics_raw()


def _strip_cy_prefix_quarter_label(q: object) -> str:
    s = str(q or "").strip().upper().replace(" ", "")
    if s.startswith("CY"):
        s = s[2:]
    return s


def _quarter_compact_yyyyqn(q: object) -> Optional[str]:
    """Canonical ``yyyyQn`` for joins (handles ``Q3 2025``, ``CY2025Q3``, ``2025Q3``)."""
    s = str(q or "").strip().upper().replace(" ", "")
    if not s:
        return None
    if s.startswith("CY"):
        s = s[2:]
    m = re.match(r"^(\d{4})Q([1-4])$", s)
    if m:
        return f"{m.group(1)}Q{m.group(2)}"
    m = re.match(r"^Q([1-4])(\d{4})$", s)
    if m:
        return f"{m.group(2)}Q{m.group(1)}"
    return None


_GEO_NURSING_CMI_DF: Optional[pd.DataFrame] = None
_GEO_NURSING_CMI_CACHE: tuple[str, float] | None = None


def _load_geo_nursing_cmi_quarterly_df() -> Optional[pd.DataFrame]:
    """Bundled state / CMS region / national nursing CMI rollups (``geo_nursing_cmi_quarterly.csv``)."""
    global _GEO_NURSING_CMI_DF, _GEO_NURSING_CMI_CACHE
    path = _resolve_pbj_lite_csv("geo_nursing_cmi_quarterly.csv")
    if not path:
        _GEO_NURSING_CMI_DF = None
        _GEO_NURSING_CMI_CACHE = None
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if (
        _GEO_NURSING_CMI_DF is not None
        and _GEO_NURSING_CMI_CACHE is not None
        and _GEO_NURSING_CMI_CACHE[0] == path
        and _GEO_NURSING_CMI_CACHE[1] == mtime
    ):
        return _GEO_NURSING_CMI_DF
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] geo_nursing_cmi_quarterly: {exc}")
        return None
    need = {"CY_Qtr", "scope", "geo_code", "nursing_cmi_weighted_mean", "n_facilities"}
    if not need.issubset(set(df.columns)):
        return None
    df = df.copy()
    df["CY_Qtr"] = df["CY_Qtr"].astype(str).str.strip()
    df["scope"] = df["scope"].astype(str).str.strip().str.upper()
    df["geo_code"] = df["geo_code"].astype(str).str.strip()
    _GEO_NURSING_CMI_DF = df
    _GEO_NURSING_CMI_CACHE = (path, mtime)
    return df


def _case_mix_geo_bundle(state_abbr: str, quarter_compact: Optional[str]) -> dict[str, Any]:
    """
    Published nursing case-mix index by geography for the same CMS quarter key as PBJ/Provider matching
    (``yyyyQn``), from ``geo_nursing_cmi_quarterly.csv`` — not tied to the dashboard date filter.
    """
    out: dict[str, Any] = {
        "available": False,
        "data_source": "geo_nursing_cmi_quarterly.csv",
        "quarter": None,
        "state": {"available": False},
        "cms_region": {"available": False},
        "national": {"available": False},
        "method_note": (
            "Means are weighted by Provider Information ``avg_residents_per_day`` (weight 1.0 when census "
            "is missing or non-positive). Geography uses cms_region_state_mapping.csv. Built from "
            "provider_info_normalized monthly extracts."
        ),
    }
    st = (state_abbr or "").strip().upper()
    qc = _quarter_compact_yyyyqn(quarter_compact) if quarter_compact else None
    if len(st) != 2 or not qc:
        return out
    out["quarter"] = qc
    df = _load_geo_nursing_cmi_quarterly_df()
    if df is None or df.empty:
        out["note"] = "geo_nursing_cmi_quarterly.csv not found or empty; run scripts/build_geo_nursing_cmi_quarterly.py."
        return out
    sub = df[df["CY_Qtr"] == qc]
    if sub.empty:
        out["note"] = f"No geo CMI rows for quarter {qc}."
        return out

    def _one_row(scope: str, geo_code: str) -> dict[str, Any]:
        r = sub[(sub["scope"] == scope) & (sub["geo_code"] == geo_code)]
        if r.empty:
            return {"available": False}
        row = r.iloc[0]
        mean_v = _hprd_benchmark_float(row, "nursing_cmi_weighted_mean")
        if mean_v is None:
            return {"available": False}
        o: dict[str, Any] = {
            "available": True,
            "nursing_cmi_weighted_mean": round(float(mean_v), 4),
            "n_facilities": _hprd_benchmark_int(row, "n_facilities"),
        }
        if "census_weight_sum" in row.index and pd.notna(row["census_weight_sum"]):
            try:
                o["census_weight_sum"] = round(float(row["census_weight_sum"]), 1)
            except (TypeError, ValueError):
                pass
        if scope == "CMS_REGION" and "cms_region_name" in row.index and pd.notna(row["cms_region_name"]):
            o["cms_region_name"] = str(row["cms_region_name"]).strip()
        if scope == "STATE":
            o["state"] = geo_code
        if scope == "CMS_REGION":
            o["cms_region_number"] = int(geo_code) if str(geo_code).isdigit() else geo_code
        return o

    st_blk = _one_row("STATE", st)
    nat_blk = _one_row("NATIONAL", "US")
    rn_int: Optional[int] = None
    map_path = _resolve_pbj_lite_csv("cms_region_state_mapping.csv")
    if map_path:
        try:
            mmap = pd.read_csv(map_path, low_memory=False)
            if "State_Code" in mmap.columns and "CMS_Region_Number" in mmap.columns:
                mrow = mmap[mmap["State_Code"].astype(str).str.strip().str.upper() == st]
                if not mrow.empty:
                    rn_raw = mrow["CMS_Region_Number"].iloc[0]
                    if not pd.isna(rn_raw):
                        rn_int = int(float(rn_raw))
        except Exception as exc:
            print(f"[WARN] case_mix_geo_bundle region map: {exc}")
    reg_blk: dict[str, Any] = {"available": False}
    if rn_int is not None:
        reg_blk = _one_row("CMS_REGION", str(rn_int))

    out["state"] = st_blk
    out["national"] = nat_blk
    out["cms_region"] = reg_blk
    out["available"] = bool(st_blk.get("available") or reg_blk.get("available") or nat_blk.get("available"))
    return out


_PROVIDER_GEO_SLICE_USECOLS: frozenset[str] = frozenset(
    {
        "ccn",
        "quarter",
        "state",
        "county",
        "avg_residents_per_day",
        "case_mix_total_nurse_hrs_per_resident_per_day",
        "case_mix_rn_hrs_per_resident_per_day",
        "case_mix_lpn_hrs_per_resident_per_day",
        "case_mix_na_hrs_per_resident_per_day",
    }
)
_PROVIDER_GEO_QUARTER_SLICE_CACHE: dict[str, tuple[str, float, pd.DataFrame]] = {}
_PROVIDER_COMBINED_CHUNK_ROWS = 100_000


def _provider_geo_slice_usecol(name: object) -> bool:
    return str(name).strip().replace("\ufeff", "") in _PROVIDER_GEO_SLICE_USECOLS


def _read_provider_combined_facility_slice(
    combined_path: str,
    search_variants: Sequence[str],
) -> tuple[Optional[pd.DataFrame], int]:
    """Chunked CCN filter from ``provider_info_combined.csv`` (all columns, facility slice build)."""
    variant_set = {str(v).strip() for v in search_variants if str(v).strip()}
    parts: list[pd.DataFrame] = []
    rows_scanned = 0
    try:
        for chunk in pd.read_csv(
            combined_path,
            low_memory=False,
            dtype={"ccn": str},
            chunksize=_PROVIDER_COMBINED_CHUNK_ROWS,
        ):
            rows_scanned += len(chunk)
            if "ccn" not in chunk.columns:
                continue
            chunk = chunk.copy()
            chunk["ccn"] = chunk["ccn"].astype(str).str.strip()
            chunk["ccn"] = chunk["ccn"].apply(
                lambda x: x.zfill(6) if x.isdigit() else x.upper()
            )
            hit = chunk[chunk["ccn"].isin(variant_set)]
            if not hit.empty:
                parts.append(cast(pd.DataFrame, hit))
    except Exception as exc:
        print(
            f"[WARN] provider_info_combined facility slice path={combined_path} "
            f"rows_scanned={rows_scanned} error={exc}"
        )
        return None, rows_scanned
    if not parts:
        return None, rows_scanned
    return pd.concat(parts, ignore_index=True), rows_scanned


def _provider_info_rows_for_geo_quarter(quarter_compact: str) -> pd.DataFrame:
    """Facilities in Provider combined for one compact quarter — chunked, column-pruned read."""
    qc = _quarter_compact_yyyyqn(quarter_compact) if quarter_compact else None
    if not qc:
        return pd.DataFrame()
    path = _provider_info_combined_csv_path()
    if not path:
        return pd.DataFrame()
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return pd.DataFrame()
    cached = _PROVIDER_GEO_QUARTER_SLICE_CACHE.get(qc)
    if cached and cached[0] == path and cached[1] == mtime:
        return cached[2]
    t0 = time.perf_counter()
    parts: list[pd.DataFrame] = []
    rows_scanned = 0
    try:
        for chunk in pd.read_csv(
            path,
            low_memory=False,
            dtype={"ccn": str},
            chunksize=_PROVIDER_COMBINED_CHUNK_ROWS,
            usecols=_provider_geo_slice_usecol,
        ):
            rows_scanned += len(chunk)
            if "quarter" not in chunk.columns or "state" not in chunk.columns:
                continue
            chunk = chunk.copy()
            chunk["CY_Qtr"] = chunk["quarter"].map(_quarter_compact_yyyyqn)
            hit = chunk[chunk["CY_Qtr"].astype(str) == qc]
            if not hit.empty:
                parts.append(cast(pd.DataFrame, hit))
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        print(
            f"[WARN] provider_info_combined geo slice quarter={qc} "
            f"rows_scanned={rows_scanned} elapsed_ms={elapsed_ms} error={exc}"
        )
        return pd.DataFrame()
    sub = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not sub.empty:
        if "ccn" in sub.columns:
            sub["ccn"] = sub["ccn"].astype(str).str.strip().str.zfill(6)
        if "state" in sub.columns:
            sub["state"] = sub["state"].astype(str).str.strip().str.upper()
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    print(
        f"[INFO] provider_info_combined geo slice quarter={qc} source={path} "
        f"rows_scanned={rows_scanned} rows_kept={len(sub)} elapsed_ms={elapsed_ms}"
    )
    _PROVIDER_GEO_QUARTER_SLICE_CACHE[qc] = (path, mtime, sub)
    return sub


def _weighted_provider_case_mix_hprd_block(grp: pd.DataFrame) -> dict[str, Any]:
    """Census-weighted mean CMS case-mix HPRD fields for a facility slice."""
    if grp is None or grp.empty:
        return {"available": False}
    cen = pd.to_numeric(grp.get("avg_residents_per_day"), errors="coerce")
    wt = cen.where(cen.notna() & (cen > 0), 1.0).astype(float)
    den = float(wt.sum())
    if den <= 0:
        return {"available": False}

    def _wm(col: str) -> Optional[float]:
        if col not in grp.columns:
            return None
        v = pd.to_numeric(grp[col], errors="coerce")
        mask = v.notna() & wt.notna()
        if not mask.any():
            return None
        num = float((v.loc[mask] * wt.loc[mask]).sum())
        d = float(wt.loc[mask].sum())
        if d <= 0:
            return None
        return round_financial(num / d, 3)

    cm_total = _wm("case_mix_total_nurse_hrs_per_resident_per_day")
    cm_rn = _wm("case_mix_rn_hrs_per_resident_per_day")
    cm_lpn = _wm("case_mix_lpn_hrs_per_resident_per_day")
    cm_na = _wm("case_mix_na_hrs_per_resident_per_day")
    cm_direct = None
    if cm_rn is not None and cm_lpn is not None and cm_na is not None:
        cm_direct = round_financial(float(cm_rn) + float(cm_lpn) + float(cm_na), 3)
    elif cm_total is not None:
        cm_direct = cm_total
    if cm_direct is None and cm_total is None:
        return {"available": False}
    return {
        "available": True,
        "case_mix_direct_hprd": cm_direct,
        "case_mix_total_hprd": cm_total,
        "case_mix_rn_hprd": cm_rn,
        "case_mix_lpn_hprd": cm_lpn,
        "case_mix_na_hprd": cm_na,
        "n_facilities": int(grp["ccn"].nunique()) if "ccn" in grp.columns else int(len(grp)),
    }


def _provider_case_mix_hprd_from_row(row: pd.Series) -> dict[str, Any]:
    """CMS case-mix HPRD fields from one Provider Information row."""
    cm_rn = _hprd_benchmark_float(row, "case_mix_rn_hrs_per_resident_per_day")
    cm_lpn = _hprd_benchmark_float(row, "case_mix_lpn_hrs_per_resident_per_day")
    cm_na = _hprd_benchmark_float(row, "case_mix_na_hrs_per_resident_per_day")
    cm_total = _hprd_benchmark_float(row, "case_mix_total_nurse_hrs_per_resident_per_day")
    cm_direct = None
    if cm_rn is not None and cm_lpn is not None and cm_na is not None:
        cm_direct = round_financial(float(cm_rn) + float(cm_lpn) + float(cm_na), 3)
    elif cm_total is not None:
        cm_direct = cm_total
    if cm_direct is None and cm_total is None:
        return {"available": False}
    return {
        "available": True,
        "case_mix_direct_hprd": cm_direct,
        "case_mix_total_hprd": cm_total,
        "case_mix_rn_hprd": cm_rn,
        "case_mix_lpn_hprd": cm_lpn,
        "case_mix_na_hprd": cm_na,
    }


def _facility_case_mix_hprd_for_quarter(
    provnum: str,
    quarter_compact: str,
) -> tuple[dict[str, Any], str]:
    """Facility CMS case-mix HPRD from bundled provider CSV — never reads national combined."""
    out: dict[str, Any] = {"available": False}
    source = provider_info_loaded_source or "facility_provider_info_data.csv"
    prov_df = _scoped_provider_info_df_for_facility()
    if prov_df is None or prov_df.empty:
        return out, source
    canon_q = _provider_quarter_to_canonical(quarter_compact)
    if not canon_q:
        return out, source
    rows = _provider_rows_for_canonical_quarter(prov_df, canon_q)
    if rows.empty:
        return out, source
    snapshot = coalesce_provider_quarter_snapshots(cast(pd.DataFrame, rows))
    block = _provider_case_mix_hprd_from_row(snapshot)
    if block.get("available"):
        block["data_source"] = source
        return block, source
    return out, source


def _enrich_case_mix_hprd_on_geo_bundle(
    bundle: dict[str, Any],
    state_abbr: str,
    quarter_compact: Optional[str],
    county_name: str = "",
    provnum: str = "",
) -> None:
    """Attach census-weighted CMS case-mix HPRD rollups (state / region / national / county / facility).

    Facility values load from the bundled facility provider CSV first. Geo comparisons use a chunked
    quarter slice of ``provider_info_combined.csv`` when present; geo failure does not blank facility.
    """
    qc = _quarter_compact_yyyyqn(quarter_compact) if quarter_compact else None
    bundle["hprd"] = {"available": False, "quarter": qc}
    if not qc:
        return
    st = (state_abbr or "").strip().upper()
    hprd: dict[str, Any] = {"available": False, "quarter": qc}
    fac_block, fac_source = _facility_case_mix_hprd_for_quarter(provnum, qc)
    hprd["facility_source"] = fac_source
    if fac_block.get("available"):
        hprd["facility"] = fac_block
        hprd["available"] = True
        print(
            f"[INFO] case_mix facility hprd quarter={qc} source={fac_source} "
            f"fallback_path=no"
        )
    else:
        print(
            f"[INFO] case_mix facility hprd quarter={qc} source={fac_source} "
            f"fallback_path=no facility_values=unavailable"
        )

    geo_warning: Optional[str] = None
    geo_source = "provider_info_combined.csv (chunked quarter slice)"
    t0_geo = time.perf_counter()
    try:
        sub = _provider_info_rows_for_geo_quarter(qc)
        elapsed_geo_ms = int((time.perf_counter() - t0_geo) * 1000)
        if sub.empty:
            geo_warning = "Geo case-mix HPRD comparison unavailable for this quarter."
            print(
                f"[WARN] case_mix geo hprd quarter={qc} source={geo_source} "
                f"rows_kept=0 elapsed_ms={elapsed_geo_ms} fallback_path=yes"
            )
        else:
            hprd["geo_source"] = geo_source
            if len(st) == 2:
                st_df = sub[sub["state"] == st]
                if not st_df.empty:
                    hprd["state"] = _weighted_provider_case_mix_hprd_block(st_df)
            hprd["national"] = _weighted_provider_case_mix_hprd_block(sub)
            county_norm = _normalize_geo_county_label(county_name)
            if county_norm and "county" in sub.columns:
                sub = sub.copy()
                sub["_county_norm"] = sub["county"].map(_normalize_geo_county_label)
                cdf = sub[sub["_county_norm"] == county_norm]
                if len(st) == 2:
                    cdf = cdf[cdf["state"] == st]
                if not cdf.empty:
                    hprd["county"] = _weighted_provider_case_mix_hprd_block(cdf)
            region_states = _cms_region_state_codes_for(st) if len(st) == 2 else []
            if region_states:
                rdf = sub[sub["state"].isin(region_states)]
                if not rdf.empty:
                    hprd["cms_region"] = _weighted_provider_case_mix_hprd_block(rdf)
            print(
                f"[INFO] case_mix geo hprd quarter={qc} source={geo_source} "
                f"rows_kept={len(sub)} elapsed_ms={elapsed_geo_ms} fallback_path=no"
            )
    except Exception as exc:
        elapsed_geo_ms = int((time.perf_counter() - t0_geo) * 1000)
        geo_warning = "Geo case-mix HPRD comparison unavailable."
        print(
            f"[WARN] case_mix geo hprd quarter={qc} source={geo_source} "
            f"elapsed_ms={elapsed_geo_ms} fallback_path=yes error={exc}"
        )

    if geo_warning:
        hprd["geo_comparison_warning"] = geo_warning
        if not hprd.get("note"):
            hprd["note"] = geo_warning
    hprd["available"] = bool(
        (hprd.get("facility") or {}).get("available")
        or (hprd.get("state") or {}).get("available")
        or (hprd.get("cms_region") or {}).get("available")
        or (hprd.get("national") or {}).get("available")
        or (hprd.get("county") or {}).get("available")
    )
    bundle["hprd"] = hprd


def _state_aide_share_of_direct_care(state_abbr: str, quarter_key: str) -> float:
    """Published nurse-aide share of direct nurse HPRD (state quarterly), for peer LPN/aide splits."""
    st = (state_abbr or "").strip().upper()
    qk = str(quarter_key or "").strip()
    if len(st) != 2 or not qk:
        return 0.62
    try:
        sdf = _read_pbj_lite_csv_cached("state_quarterly_metrics.csv")
        if sdf is None or "STATE" not in sdf.columns or "CY_Qtr" not in sdf.columns:
            return 0.62
        row = sdf[
            (sdf["STATE"].astype(str).str.strip().str.upper() == st)
            & (sdf["CY_Qtr"].astype(str) == qk)
        ]
        if row.empty:
            return 0.62
        r = row.iloc[-1]
        care = _hprd_benchmark_float(r, "Nurse_Care_HPRD")
        aide = _hprd_benchmark_float(r, "Nurse_Assistant_HPRD")
        if care is None or aide is None or float(care) <= 0:
            return 0.62
        share = float(aide) / float(care)
        return min(0.95, max(0.05, share))
    except Exception:
        return 0.62


def _pct_facilities_below(hprd: float, peers: Any) -> Optional[float]:
    """Share of peer facilities with strictly lower total/direct HPRD (0–100)."""
    arr = pd.to_numeric(_as_1d_series(peers), errors="coerce").dropna()
    if len(arr) < 5:
        return None
    a = arr.to_numpy(dtype=float)
    return float(round(100.0 * float((a < float(hprd)).sum()) / float(len(a)), 1))


def _state_peer_hprd_percentiles_bundle(
    state_abbr: str,
    provnum: str,
    bundle_quarter: Optional[str],
) -> dict[str, Any]:
    """
    Cross-sectional rank of this facility vs other facilities in the same state & CMS quarter,
    using bundled ``facility_lite_metrics.csv`` (published rollups — not the dashboard date filter).
    """
    out: dict[str, Any] = {"available": False}
    st = (state_abbr or "").strip().upper()
    pv = str(provnum or "").strip()
    if pv.isdigit():
        pv = pv.zfill(6)
    if len(st) != 2 or not pv or not bundle_quarter:
        return out
    qf = _strip_cy_prefix_quarter_label(bundle_quarter)
    if not re.match(r"^\d{4}Q[1-4]$", qf):
        return out
    df = _load_facility_lite_metrics_df()
    if df is None or df.empty:
        return out
    sub = df[(df["STATE"] == st) & (df["CY_Qtr"] == qf)].copy()
    if sub.empty:
        return out
    fac = sub[sub["PROVNUM"] == pv]
    if fac.empty:
        out["note"] = "Facility not present in facility_lite_metrics for this quarter."
        return out
    t_fac = _hprd_benchmark_float(fac.iloc[0], "Total_Nurse_HPRD")
    d_fac = _hprd_benchmark_float(fac.iloc[0], "Nurse_Care_HPRD")
    if t_fac is None:
        return out
    tv = sub["Total_Nurse_HPRD"]
    dv = sub["Nurse_Care_HPRD"]
    pr_t = _pct_facilities_below(float(t_fac), tv)
    if pr_t is None:
        return out
    out["available"] = True
    out["quarter"] = qf
    out["state"] = st
    out["n_facilities"] = int(pd.to_numeric(tv, errors="coerce").dropna().shape[0])
    out["facility_total_hprd_bundle"] = round(float(t_fac), 3)
    out["pct_below_total"] = pr_t
    pr_d = _pct_facilities_below(float(d_fac), dv) if d_fac is not None else None
    out["facility_direct_hprd_bundle"] = round(float(d_fac), 3) if d_fac is not None else None
    out["pct_below_direct"] = pr_d
    out["state_median_total"] = round(float(pd.to_numeric(tv, errors="coerce").dropna().median()), 3)
    dvn = pd.to_numeric(dv, errors="coerce").dropna()
    out["state_median_direct"] = round(float(dvn.median()), 3) if len(dvn) else None
    out["method_note"] = (
        "Peer % uses CMS quarter rollups for all facilities in this state from facility_lite_metrics.csv; "
        "it is not computed from your custom date filter in this dashboard."
    )
    return out


def _provider_nursing_cmi_for_matched_quarter(
    provnum: str,
    quarter_compact: Optional[str],
) -> dict[str, Any]:
    """Nursing case-mix index from Provider Information for the matched quarter (if present).

    If the matched PBJ quarter has no Provider row with a parsable nursing CMI, walks backward
    through earlier quarters present for this CCN (same spirit as the CMS % Case-Mix anchor).
    """
    global provider_info_df
    out: dict[str, Any] = {"available": False}
    if provider_info_df is None or provider_info_df.empty or not quarter_compact:
        return out
    if "ccn" not in provider_info_df.columns or "quarter" not in provider_info_df.columns:
        return out
    pv = str(provnum or "").strip()
    if pv.isdigit():
        pv = pv.zfill(6)
    fac = provider_info_df[provider_info_df["ccn"].astype(str).str.strip().str.zfill(6) == pv]
    if fac.empty:
        return out
    qn = _quarter_compact_yyyyqn(quarter_compact)
    if not qn:
        return out

    def _qnorm(x: object) -> Optional[str]:
        return _quarter_compact_yyyyqn(x)

    def _row_to_cmi_dict(row: pd.Series, data_quarter: str, anchor_quarter: str) -> Optional[dict[str, Any]]:
        cmi_raw, _cmi_src = extract_nursing_cmi_from_provider_series(row)
        if cmi_raw is None:
            return None
        o: dict[str, Any] = {
            "available": True,
            "quarter": anchor_quarter,
            "nursing_case_mix_index": round(float(cmi_raw), 4),
        }
        cmi_ratio: Optional[float] = None
        for c in row.index:
            if not isinstance(c, str):
                continue
            cl = c.lower().replace("\ufeff", "")
            if "nursing case-mix index ratio" in cl or cl in ("nursing_case_mix_index_ratio", "nursing case-mix index ratio"):
                try:
                    rv = row[c]
                    if pd.notna(rv):
                        cmi_ratio = float(rv)
                        break
                except (TypeError, ValueError):
                    continue
        if cmi_ratio is not None:
            o["nursing_case_mix_index_ratio"] = round(float(cmi_ratio), 4)
        if data_quarter != anchor_quarter:
            o["cmi_source_quarter"] = data_quarter
        return o

    anchor_key = _cy_qtr_sort_key_lite(qn)
    q_keys: list[str] = []
    for raw in fac["quarter"].dropna().unique():
        k = _qnorm(raw)
        if k:
            q_keys.append(k)
    q_keys = sorted(set(q_keys), key=_cy_qtr_sort_key_lite)
    candidates = [k for k in q_keys if _cy_qtr_sort_key_lite(k) <= anchor_key]
    candidates.sort(key=_cy_qtr_sort_key_lite, reverse=True)
    if not candidates:
        return out

    for try_q in candidates:
        fac_q = fac[fac["quarter"].map(_qnorm) == try_q]
        if fac_q.empty:
            continue
        row = fac_q.sort_values("processing_date", na_position="last").iloc[-1]
        parsed = _row_to_cmi_dict(row, try_q, qn)
        if parsed is not None:
            return parsed
    return out


def _cy_qtr_compact_label(val: object) -> Optional[str]:
    """``yyyyQn`` key aligned with ``pbjNormalizeQuarterToCy`` in the facility dashboard JS."""
    n = _normalize_cy_qtr(val)
    if not n:
        return None
    return n[2:] if n.upper().startswith("CY") else n


def _normalize_geo_county_label(name: object) -> str:
    return re.sub(r"\s+", " ", str(name or "").strip()).lower()


_FACILITY_GEO_PEER_DF: Optional[pd.DataFrame] = None
_FACILITY_GEO_PEER_CACHE: Optional[tuple[str, float]] = None
_PROVNUM_LOCALE_LOOKUP: Optional[dict[str, str]] = None
_PROVNUM_LOCALE_CACHE: Optional[tuple[str, float]] = None
_CMS_REGION_STATES_BY_STATE: Optional[dict[str, list[str]]] = None


def _load_facility_geo_peer_df() -> Optional[pd.DataFrame]:
    """Facility-quarter rows for county / locale peer rollups (lite + quarterly supplement)."""
    return supplement_facility_lite_peer_df(_load_facility_lite_metrics_raw())


def _load_provnum_locale_lookup() -> dict[str, str]:
    """CCN → ``rural`` | ``urban`` | ``unknown`` from latest bundled CMS Provider Information file."""
    global _PROVNUM_LOCALE_LOOKUP, _PROVNUM_LOCALE_CACHE
    roots = [
        os.path.join(_app_root, "provider_info"),
        os.path.join(_app_root, "provider_info_extracted"),
        os.path.abspath(os.path.join(_app_root, "..", "..", "provider_info")),
    ]
    candidates: list[str] = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        candidates.extend(glob.glob(os.path.join(root, "NH_ProviderInfo_*.csv")))
    if not candidates:
        _PROVNUM_LOCALE_LOOKUP = {}
        return {}
    latest = max(candidates, key=os.path.getmtime)
    try:
        mtime = os.path.getmtime(latest)
    except OSError:
        return _PROVNUM_LOCALE_LOOKUP or {}
    if (
        _PROVNUM_LOCALE_LOOKUP is not None
        and _PROVNUM_LOCALE_CACHE is not None
        and _PROVNUM_LOCALE_CACHE[0] == latest
        and _PROVNUM_LOCALE_CACHE[1] == mtime
    ):
        return _PROVNUM_LOCALE_LOOKUP
    ccn_col = "CMS Certification Number (CCN)"
    locale_usecols: list[str] = [ccn_col, "Urban"]
    try:
        pdf = pd.read_csv(latest, usecols=cast(Any, locale_usecols), low_memory=False)
    except Exception as exc:
        print(f"[WARN] provnum locale lookup: {exc}")
        _PROVNUM_LOCALE_LOOKUP = {}
        return {}
    out: dict[str, str] = {}
    for _, row in pdf.iterrows():
        ccn_raw = row.get(ccn_col)
        if pd.isna(ccn_raw):
            continue
        ccn = str(ccn_raw).strip()
        if ccn.isdigit():
            ccn = ccn.zfill(6)
        urban_raw = row.get("Urban")
        if pd.isna(urban_raw):
            locale = "unknown"
        else:
            u = str(urban_raw).strip().upper()
            if u in ("Y", "YES", "URBAN", "U"):
                locale = "urban"
            elif u in ("N", "NO", "RURAL", "R"):
                locale = "rural"
            else:
                locale = "unknown"
        out[ccn] = locale
    _PROVNUM_LOCALE_LOOKUP = out
    _PROVNUM_LOCALE_CACHE = (latest, mtime)
    return out


def _cms_region_state_codes_for(state_abbr: str) -> list[str]:
    """Two-letter state codes in the same CMS region as ``state_abbr``."""
    global _CMS_REGION_STATES_BY_STATE
    st = (state_abbr or "").strip().upper()
    if len(st) != 2:
        return []
    if _CMS_REGION_STATES_BY_STATE is None:
        mapping: dict[str, list[str]] = {}
        map_path = _resolve_pbj_lite_csv("cms_region_state_mapping.csv")
        if map_path:
            try:
                mmap = pd.read_csv(map_path, low_memory=False)
                if "State_Code" in mmap.columns and "CMS_Region_Number" in mmap.columns:
                    mmap = mmap.copy()
                    mmap["State_Code"] = mmap["State_Code"].astype(str).str.strip().str.upper()
                    mmap["CMS_Region_Number"] = pd.to_numeric(mmap["CMS_Region_Number"], errors="coerce")
                    for rn, grp in mmap.groupby("CMS_Region_Number"):
                        if pd.isna(rn):
                            continue
                        codes = sorted({str(x) for x in grp["State_Code"].tolist() if x})
                        for code in codes:
                            mapping[code] = codes
            except Exception as exc:
                print(f"[WARN] cms_region_state_codes: {exc}")
        _CMS_REGION_STATES_BY_STATE = mapping
    return list(_CMS_REGION_STATES_BY_STATE.get(st) or [st])


def _as_peer_geo_frame(fr: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(fr, pd.DataFrame):
        return fr
    return pd.DataFrame()


def _peer_geo_metrics_from_facility_frame(
    fr: pd.DataFrame | pd.Series,
    aide_share_of_direct: Optional[float] = None,
) -> dict[str, Any]:
    """Census-weighted HPRD rollups for a facility-quarter slice."""
    frame = _as_peer_geo_frame(fr)
    if frame.empty:
        return {}
    cens = pd.Series(pd.to_numeric(frame["Census"], errors="coerce").fillna(0.0))
    cens_sum = float(cens.sum())
    if cens_sum <= 0:
        return {}
    w = cens
    aide_share = float(aide_share_of_direct) if aide_share_of_direct is not None else 0.62
    aide_share = min(0.95, max(0.05, aide_share))

    def wmean(col: str) -> Optional[float]:
        v = pd.Series(pd.to_numeric(frame[col], errors="coerce"))
        mask = v.notna() & (w > 0)
        if not mask.any():
            return None
        w_sub = w.loc[mask]
        v_sub = v.loc[mask]
        denom = float(w_sub.sum())
        if denom <= 0:
            return None
        return round_financial(float((v_sub * w_sub).sum() / denom), 3)

    def wmean_pairs(pairs: list[tuple[float, float]]) -> Optional[float]:
        if not pairs:
            return None
        num = sum(p[0] * p[1] for p in pairs)
        den = sum(p[1] for p in pairs)
        if den <= 0:
            return None
        return round_financial(num / den, 3)

    total = wmean("Total_Nurse_HPRD")
    care = wmean("Nurse_Care_HPRD")
    rn = wmean("Total_RN_HPRD")
    rn_care = wmean("Direct_Care_RN_HPRD")
    contract = wmean("Contract_Percentage")
    avg_census = round_financial(float(w.sum() / float((w > 0).sum())), 1) if (w > 0).any() else None

    aide_pairs: list[tuple[float, float]] = []
    lpn_care_pairs: list[tuple[float, float]] = []
    lpn_total_pairs: list[tuple[float, float]] = []
    for idx in frame.index:
        w_i = float(w.loc[idx]) if w.loc[idx] > 0 else 0.0
        if w_i <= 0:
            continue
        c_v = pd.to_numeric(frame.at[idx, "Nurse_Care_HPRD"], errors="coerce")
        rc_v = pd.to_numeric(frame.at[idx, "Direct_Care_RN_HPRD"], errors="coerce")
        if pd.isna(c_v) or pd.isna(rc_v):
            continue
        rem = float(c_v) - float(rc_v)
        if rem < 0:
            continue
        aide_i = max(0.0, rem * aide_share)
        lpn_c_i = max(0.0, rem - aide_i)
        aide_pairs.append((aide_i, w_i))
        lpn_care_pairs.append((lpn_c_i, w_i))
        t_v = pd.to_numeric(frame.at[idx, "Total_Nurse_HPRD"], errors="coerce")
        r_v = pd.to_numeric(frame.at[idx, "Total_RN_HPRD"], errors="coerce")
        if not pd.isna(t_v) and not pd.isna(r_v):
            non_rn = float(t_v) - float(r_v)
            if non_rn >= 0:
                lpn_total_pairs.append((max(0.0, non_rn - aide_i), w_i))

    aide = wmean_pairs(aide_pairs)
    lpn_care = wmean_pairs(lpn_care_pairs)
    lpn_total = wmean_pairs(lpn_total_pairs)
    if lpn_total is None and total is not None and rn is not None and aide is not None:
        lpn_total = _geo_residual_lpn_hprd(total, rn, aide)

    return {
        "total_nurse_hprd": total,
        "nurse_care_hprd": care,
        "rn_hprd": rn,
        "rn_care_hprd": rn_care,
        "nurse_aide_hprd": aide,
        "lpn_hprd": lpn_total,
        "lpn_care_hprd": lpn_care,
        "contract_pct": round_financial(contract, 2) if contract is not None else None,
        "avg_census": avg_census,
        "facility_count": int(frame["PROVNUM"].nunique()) if "PROVNUM" in frame.columns else int(len(frame)),
    }


def _merge_peer_geo_metrics(entry: dict[str, Any], prefix: str, metrics: dict[str, Any]) -> None:
    if not metrics:
        return
    entry[f"{prefix}_total_nurse_hprd"] = metrics.get("total_nurse_hprd")
    entry[f"{prefix}_nurse_care_hprd"] = metrics.get("nurse_care_hprd")
    entry[f"{prefix}_rn_hprd"] = metrics.get("rn_hprd")
    entry[f"{prefix}_rn_care_hprd"] = metrics.get("rn_care_hprd")
    entry[f"{prefix}_nurse_aide_hprd"] = metrics.get("nurse_aide_hprd")
    entry[f"{prefix}_lpn_hprd"] = metrics.get("lpn_hprd")
    entry[f"{prefix}_lpn_care_hprd"] = metrics.get("lpn_care_hprd")
    entry[f"{prefix}_contract_pct"] = metrics.get("contract_pct")
    entry[f"{prefix}_avg_census"] = metrics.get("avg_census")
    entry[f"{prefix}_facility_count"] = metrics.get("facility_count")


def _facility_locale_for_provnum(provnum: str, locale_lookup: dict[str, str]) -> str:
    pv = str(provnum or "").strip()
    if pv.isdigit():
        pv = pv.zfill(6)
    return locale_lookup.get(pv, "unknown")


def _enrich_geo_rollup_peer_facility_slices(
    series: dict[str, Any],
    state_abbr: str,
    county_name: str,
    provnum: str,
) -> None:
    """Add county and rural/urban peer slices from ``facility_lite_metrics.csv``."""
    if not _resolve_pbj_lite_csv("facility_lite_metrics.csv"):
        return
    by_quarter = series.get("by_quarter")
    if not isinstance(by_quarter, dict) or not by_quarter:
        return
    st = (state_abbr or "").strip().upper()
    county_norm = _normalize_geo_county_label(county_name)
    df = _load_facility_geo_peer_df()
    if df is None or df.empty or len(st) != 2:
        return
    locale_lookup = _load_provnum_locale_lookup()
    facility_locale = _facility_locale_for_provnum(provnum, locale_lookup)
    series["facility_locale"] = facility_locale
    series["facility_county"] = str(county_name or "").strip() or None
    region_states = _cms_region_state_codes_for(st)
    cms_region_number: Optional[int] = None
    cms_region_name: Optional[str] = None
    cms_region_full: Optional[str] = None
    map_path = _resolve_pbj_lite_csv("cms_region_state_mapping.csv")
    if map_path:
        try:
            mmap = pd.read_csv(map_path, low_memory=False)
            if "State_Code" in mmap.columns and "CMS_Region_Number" in mmap.columns:
                mrow = mmap[mmap["State_Code"].astype(str).str.strip().str.upper() == st]
                if not mrow.empty:
                    rn_raw = mrow["CMS_Region_Number"].iloc[0]
                    if not pd.isna(rn_raw):
                        cms_region_number = int(float(rn_raw))
                    if "CMS_Region_Name" in mrow.columns:
                        cms_region_name = str(mrow["CMS_Region_Name"].iloc[0]).strip() or None
                    if "CMS_Region_Full" in mrow.columns:
                        cms_region_full = str(mrow["CMS_Region_Full"].iloc[0]).strip() or None
        except Exception as exc:
            print(f"[WARN] geo rollup cms_region mapping: {exc}")
    df = df.copy()
    df["_locale"] = df["PROVNUM"].map(lambda p: locale_lookup.get(str(p), "unknown"))

    for qk, entry in by_quarter.items():
        if not isinstance(entry, dict):
            continue
        if cms_region_number is not None and entry.get("cms_region_number") is None:
            entry["cms_region_number"] = cms_region_number
        if cms_region_name and not entry.get("cms_region_name"):
            entry["cms_region_name"] = cms_region_name
        if cms_region_full and not entry.get("cms_region_full"):
            entry["cms_region_full"] = cms_region_full
        qdf = df[df["CY_Qtr"].astype(str) == str(qk)]
        if qdf.empty:
            qdf = df[df["CY_Qtr"].astype(str) == str(entry.get("quarter") or "")]
        if qdf.empty:
            continue
        aide_share = _state_aide_share_of_direct_care(st, str(qk))
        st_df = qdf[qdf["STATE"] == st]
        if county_norm:
            county_df = st_df[st_df["_county_norm"] == county_norm]
            _merge_peer_geo_metrics(
                entry,
                "county",
                _peer_geo_metrics_from_facility_frame(cast(pd.DataFrame, county_df), aide_share),
            )
            entry["county_label"] = str(county_name or "").strip()
            if facility_locale in ("rural", "urban"):
                loc_df = county_df[county_df["_locale"] == facility_locale]
                _merge_peer_geo_metrics(
                    entry,
                    "county_locale",
                    _peer_geo_metrics_from_facility_frame(cast(pd.DataFrame, loc_df), aide_share),
                )
        rural_st = st_df[st_df["_locale"] == "rural"]
        urban_st = st_df[st_df["_locale"] == "urban"]
        _merge_peer_geo_metrics(
            entry, "state_rural", _peer_geo_metrics_from_facility_frame(cast(pd.DataFrame, rural_st), aide_share)
        )
        _merge_peer_geo_metrics(
            entry, "state_urban", _peer_geo_metrics_from_facility_frame(cast(pd.DataFrame, urban_st), aide_share)
        )
        if region_states:
            reg_df = qdf[qdf["STATE"].isin(region_states)]
            reg_all = _peer_geo_metrics_from_facility_frame(cast(pd.DataFrame, reg_df), aide_share)
            if reg_all and entry.get("region_total_nurse_hprd") is None:
                _merge_peer_geo_metrics(entry, "region", reg_all)
            reg_rural = reg_df[reg_df["_locale"] == "rural"]
            reg_urban = reg_df[reg_df["_locale"] == "urban"]
            _merge_peer_geo_metrics(
                entry, "region_rural", _peer_geo_metrics_from_facility_frame(cast(pd.DataFrame, reg_rural), aide_share)
            )
            _merge_peer_geo_metrics(
                entry, "region_urban", _peer_geo_metrics_from_facility_frame(cast(pd.DataFrame, reg_urban), aide_share)
            )


def lite_hprd_rollup_series_by_quarter(
    state_abbr: str,
    county_name: str = "",
    provnum: str = "",
) -> dict[str, Any]:
    """
    Published state / national / CMS region rollup rows for every available quarter,
    keyed by compact ``yyyyQn`` (same as JS quarter labels).
    """
    empty: dict[str, Any] = {"by_quarter": {}, "data_source": None}
    st = (state_abbr or "").strip().upper()
    if len(st) != 2:
        return empty

    st_path_q = _resolve_pbj_lite_csv("state_quarterly_metrics.csv")
    nat_path_q = _resolve_pbj_lite_csv("national_quarterly_metrics.csv")
    if st_path_q and nat_path_q:
        try:
            sdf = _read_pbj_lite_csv_cached("state_quarterly_metrics.csv")
            if sdf is None:
                return empty
        except Exception as exc:
            print(f"[WARN] geo rollup series state_quarterly_metrics: {exc}")
            return empty
        need = ("STATE", "CY_Qtr", "Total_Nurse_HPRD")
        if not all(c in sdf.columns for c in need):
            return empty
        sub = sdf[sdf["STATE"].astype(str).str.strip().str.upper() == st].copy()
        if sub.empty:
            return empty
        sub["_qk"] = sub["CY_Qtr"].map(_cy_qtr_sort_key_lite)
        sub = sub.sort_values("_qk")
        try:
            ndf = _read_pbj_lite_csv_cached("national_quarterly_metrics.csv")
            if ndf is None:
                return empty
        except Exception as exc:
            print(f"[WARN] geo rollup series national_quarterly_metrics: {exc}")
            return empty
        if not all(c in ndf.columns for c in need):
            return empty

        rn_int: Optional[int] = None
        rname: Optional[str] = None
        rfull: Optional[str] = None
        rdf: Optional[pd.DataFrame] = None
        map_path = _resolve_pbj_lite_csv("cms_region_state_mapping.csv")
        reg_path = _resolve_pbj_lite_csv("cms_region_quarterly_metrics.csv")
        if map_path and reg_path:
            try:
                mmap = _read_pbj_lite_csv_cached("cms_region_state_mapping.csv")
                if mmap is not None and "State_Code" in mmap.columns:
                    mrow = mmap[mmap["State_Code"].astype(str).str.strip().str.upper() == st]
                    if not mrow.empty:
                        rn_raw = mrow["CMS_Region_Number"].iloc[0]
                        if not pd.isna(rn_raw):
                            rn_int = int(float(rn_raw))
                            rname = mrow["CMS_Region_Name"].iloc[0] if "CMS_Region_Name" in mrow.columns else None
                            rfull = mrow["CMS_Region_Full"].iloc[0] if "CMS_Region_Full" in mrow.columns else None
                            rdf = _read_pbj_lite_csv_cached("cms_region_quarterly_metrics.csv")
            except Exception as exc:
                print(f"[WARN] geo rollup series cms_region: {exc}")

        quarterly_by_quarter: dict[str, dict[str, Any]] = {}
        for qtr in sub["CY_Qtr"].astype(str).unique():
            last = sub[sub["CY_Qtr"].astype(str) == qtr].sort_values("_qk").iloc[-1]
            st_h = _hprd_benchmark_float(last, "Total_Nurse_HPRD")
            if st_h is None:
                continue
            cq = _cy_qtr_compact_label(qtr)
            if not cq:
                continue
            entry: dict[str, Any] = {
                "quarter": str(qtr),
                "state_total_nurse_hprd": st_h,
                "national_total_nurse_hprd": None,
                "state_nurse_care_hprd": _hprd_benchmark_float(last, "Nurse_Care_HPRD"),
                "national_nurse_care_hprd": None,
                "region_total_nurse_hprd": None,
                "region_nurse_care_hprd": None,
                "cms_region_number": rn_int,
                "cms_region_name": str(rname).strip() if rname is not None and pd.notna(rname) else None,
                "cms_region_full": str(rfull).strip() if rfull is not None and pd.notna(rfull) else None,
                "state_facility_count": _hprd_benchmark_int(last, "facility_count")
                or _hprd_benchmark_int(last, "Facility_Count"),
                "national_facility_count": None,
                "region_facility_count": None,
                "state_rn_hprd": _hprd_benchmark_float(last, "RN_HPRD"),
                "national_rn_hprd": None,
                "region_rn_hprd": None,
                "state_rn_care_hprd": _hprd_benchmark_float(last, "RN_Care_HPRD"),
                "national_rn_care_hprd": None,
                "region_rn_care_hprd": None,
                "state_nurse_aide_hprd": _hprd_benchmark_float(last, "Nurse_Assistant_HPRD"),
                "national_nurse_aide_hprd": None,
                "region_nurse_aide_hprd": None,
                "state_contract_pct": _hprd_benchmark_float(last, "Contract_Percentage"),
                "national_contract_pct": None,
                "region_contract_pct": None,
                "state_avg_census": _hprd_benchmark_float(last, "avg_daily_census"),
                "national_avg_census": None,
                "region_avg_census": None,
                "data_source": "quarterly",
            }
            nsub = ndf[
                (ndf["STATE"].astype(str).str.strip().str.upper() == "NATIONAL")
                & (ndf["CY_Qtr"].astype(str) == str(qtr))
            ]
            if not nsub.empty:
                nl = nsub.iloc[-1]
                entry["national_total_nurse_hprd"] = _hprd_benchmark_float(nl, "Total_Nurse_HPRD")
                entry["national_nurse_care_hprd"] = _hprd_benchmark_float(nl, "Nurse_Care_HPRD")
                entry["national_facility_count"] = _hprd_benchmark_int(nl, "facility_count")
                entry["national_rn_hprd"] = _hprd_benchmark_float(nl, "RN_HPRD")
                entry["national_rn_care_hprd"] = _hprd_benchmark_float(nl, "RN_Care_HPRD")
                entry["national_nurse_aide_hprd"] = _hprd_benchmark_float(nl, "Nurse_Assistant_HPRD")
                entry["national_contract_pct"] = _hprd_benchmark_float(nl, "Contract_Percentage")
                entry["national_avg_census"] = _hprd_benchmark_float(nl, "avg_daily_census")
            if rn_int is not None and rdf is not None and "CMS_Region_Number" in rdf.columns:
                rnum = pd.to_numeric(rdf["CMS_Region_Number"], errors="coerce")
                rsub = rdf[(rnum == rn_int) & (rdf["CY_Qtr"].astype(str) == str(qtr))]
                if not rsub.empty:
                    rl = rsub.iloc[-1]
                    entry["region_total_nurse_hprd"] = _hprd_benchmark_float(rl, "Total_Nurse_HPRD")
                    entry["region_nurse_care_hprd"] = _hprd_benchmark_float(rl, "Nurse_Care_HPRD")
                    entry["region_facility_count"] = _hprd_benchmark_int(rl, "facility_count")
                    entry["region_rn_hprd"] = _hprd_benchmark_float(rl, "RN_HPRD")
                    entry["region_rn_care_hprd"] = _hprd_benchmark_float(rl, "RN_Care_HPRD")
                    entry["region_nurse_aide_hprd"] = _hprd_benchmark_float(rl, "Nurse_Assistant_HPRD")
                    entry["region_contract_pct"] = _hprd_benchmark_float(rl, "Contract_Percentage")
                    entry["region_avg_census"] = _hprd_benchmark_float(rl, "avg_daily_census")
            quarterly_by_quarter[cq] = entry
        out_q = {"by_quarter": quarterly_by_quarter, "data_source": "quarterly"}
        _enrich_geo_rollup_peer_facility_slices(out_q, st, county_name, provnum)
        for entry in quarterly_by_quarter.values():
            if isinstance(entry, dict):
                _enrich_geo_rollup_entry_lpn(entry)
        return out_q

    st_path = _resolve_pbj_lite_csv("state_lite_metrics.csv")
    nat_path = _resolve_pbj_lite_csv("national_lite_metrics.csv")
    if not st_path:
        return empty
    by_quarter: dict[str, dict[str, Any]] = {}
    try:
        sdf = _read_pbj_lite_csv_cached("state_lite_metrics.csv")
        if sdf is None or not (
            "STATE" in sdf.columns
            and "CY_Qtr" in sdf.columns
            and "Total_Nurse_HPRD" in sdf.columns
        ):
            return empty
        sub = sdf[sdf["STATE"].astype(str).str.strip().str.upper() == st].copy()
        if sub.empty:
            return empty
        sub["_qk"] = sub["CY_Qtr"].map(_cy_qtr_sort_key_lite)
        nat_by_raw: dict[str, pd.Series] = {}
        if nat_path and os.path.isfile(nat_path):
            ndf = _read_pbj_lite_csv_cached("national_lite_metrics.csv")
            if ndf is not None and "CY_Qtr" in ndf.columns:
                if "STATE" in ndf.columns:
                    n_nat = ndf[ndf["STATE"].astype(str).str.strip().str.upper() == "NATIONAL"]
                else:
                    n_nat = ndf
                for _, r in n_nat.iterrows():
                    nat_by_raw[str(r["CY_Qtr"])] = r
        for qtr in sub.sort_values("_qk")["CY_Qtr"].astype(str).unique():
            last = sub[sub["CY_Qtr"].astype(str) == qtr].sort_values("_qk").iloc[-1]
            cq = _cy_qtr_compact_label(qtr)
            if not cq:
                continue
            entry = {
                "quarter": str(qtr),
                "state_total_nurse_hprd": float(last["Total_Nurse_HPRD"]),
                "national_total_nurse_hprd": None,
                "state_nurse_care_hprd": _hprd_benchmark_float(last, "Nurse_Care_HPRD"),
                "national_nurse_care_hprd": None,
                "region_total_nurse_hprd": None,
                "region_nurse_care_hprd": None,
                "cms_region_number": None,
                "cms_region_name": None,
                "cms_region_full": None,
                "state_facility_count": _hprd_benchmark_int(last, "Facility_Count"),
                "national_facility_count": None,
                "region_facility_count": None,
                "state_rn_hprd": _hprd_benchmark_float(last, "Total_RN_HPRD"),
                "national_rn_hprd": None,
                "region_rn_hprd": None,
                "state_rn_care_hprd": _hprd_benchmark_float(last, "Direct_Care_RN_HPRD"),
                "national_rn_care_hprd": None,
                "region_rn_care_hprd": None,
                "state_nurse_aide_hprd": None,
                "national_nurse_aide_hprd": None,
                "region_nurse_aide_hprd": None,
                "state_contract_pct": _hprd_benchmark_float(last, "Contract_Percentage"),
                "national_contract_pct": None,
                "region_contract_pct": None,
                "state_avg_census": _hprd_benchmark_float(last, "Census"),
                "national_avg_census": None,
                "region_avg_census": None,
                "data_source": "lite",
            }
            nr = nat_by_raw.get(str(qtr))
            if nr is not None and "Total_Nurse_HPRD" in nr:
                try:
                    entry["national_total_nurse_hprd"] = float(nr["Total_Nurse_HPRD"])
                except (TypeError, ValueError):
                    pass
                entry["national_nurse_care_hprd"] = _hprd_benchmark_float(nr, "Nurse_Care_HPRD")
                entry["national_facility_count"] = _hprd_benchmark_int(nr, "facility_count")
                entry["national_rn_hprd"] = _hprd_benchmark_float(nr, "Total_RN_HPRD")
                entry["national_rn_care_hprd"] = _hprd_benchmark_float(nr, "Direct_Care_RN_HPRD")
                entry["national_contract_pct"] = _hprd_benchmark_float(nr, "Contract_Percentage")
                _nfc = _hprd_benchmark_int(nr, "Facility_Count") or _hprd_benchmark_int(nr, "facility_count")
                _mds = _hprd_benchmark_float(nr, "MDS")
                if _nfc is not None and int(_nfc) > 0 and _mds is not None:
                    entry["national_avg_census"] = float(_mds) / float(int(_nfc))
            by_quarter[cq] = entry
        out_lite = {"by_quarter": by_quarter, "data_source": "lite"}
        _enrich_geo_rollup_peer_facility_slices(out_lite, st, county_name, provnum)
        for entry in by_quarter.values():
            if isinstance(entry, dict):
                _enrich_geo_rollup_entry_lpn(entry)
        return out_lite
    except Exception as exc:
        print(f"[WARN] geo rollup series lite: {exc}")
        return empty


def _facility_geo_from_cms_region_mapping(
    state_abbr: str, region_number: Optional[int]
) -> Tuple[Optional[str], List[str]]:
    """
    CMS Region peer geography from ``cms_region_state_mapping.csv`` (repo / pbj_lite):
    full state name for the facility's state, and sorted state/territory codes in the same CMS region.
    """
    st = (state_abbr or "").strip().upper()
    long_name: Optional[str] = None
    peers: List[str] = []
    if len(st) != 2:
        return long_name, peers
    map_path = _resolve_pbj_lite_csv("cms_region_state_mapping.csv")
    if not map_path:
        return long_name, peers
    try:
        mmap = pd.read_csv(map_path, low_memory=False)
        if "State_Code" not in mmap.columns:
            return long_name, peers
        sc = mmap["State_Code"].astype(str).str.strip().str.upper()
        mrow = mmap[sc == st]
        if not mrow.empty and "State_Name" in mmap.columns:
            sn = mrow["State_Name"].iloc[0]
            if pd.notna(sn):
                long_name = str(sn).strip()
        if region_number is not None and "CMS_Region_Number" in mmap.columns:
            rnum = pd.to_numeric(mmap["CMS_Region_Number"], errors="coerce")
            sub = mmap[rnum == int(region_number)]
            codes = sub["State_Code"].astype(str).str.strip().str.upper().dropna().unique().tolist()
            peers = sorted(x for x in codes if isinstance(x, str) and 1 <= len(x) <= 2)
    except Exception as exc:
        print(f"[WARN] cms_region_state_mapping (geo card): {exc}")
    return long_name, peers


def load_macpac_standards():
    """Load MACPAC state standards data"""
    global macpac_standards_df
    try:
        # Try multiple locations (for deployment flexibility)
        macpac_file = None
        possible_paths = [
            'macpac_state_standards_clean.csv',  # Deployment directory
            'pbj_lite/macpac_state_standards_clean.csv',  # Development
            'macpac/macpac_state_standards.csv'  # Original location
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                macpac_file = path
                break
        
        if macpac_file and os.path.exists(macpac_file):
            macpac_standards_df = pd.read_csv(macpac_file)
            # If using the original file, parse it to create clean structure
            if 'Min_Staffing' not in macpac_standards_df.columns:
                # Parse the HPRD values from the original format
                def parse_hprd(hprd_str):
                    """Parse HPRD string like '3.56 HPRD' or '3.56—4.16 HPRD'"""
                    if pd.isna(hprd_str):
                        return None, None, 'single', False
                    
                    hprd_str = str(hprd_str).replace(' HPRD', '').strip()
                    if '—' in hprd_str or '-' in hprd_str:
                        # Range
                        parts = hprd_str.replace('—', '-').split('-')
                        if len(parts) == 2:
                            try:
                                min_val = float(parts[0].strip())
                                max_val = float(parts[1].strip())
                                is_federal = min_val == 0.3 and max_val == 0.3
                                return min_val, max_val, 'range', is_federal
                            except:
                                return None, None, 'single', False
                    else:
                        # Single value
                        try:
                            val = float(hprd_str)
                            is_federal = val == 0.3
                            return val, val, 'single', is_federal
                        except:
                            return None, None, 'single', False
                
                macpac_standards_df[['Min_Staffing', 'Max_Staffing', 'Value_Type', 'Is_Federal_Minimum']] = \
                    macpac_standards_df['Total_Estimated_Staffing_Requirements'].apply(
                        lambda x: pd.Series(parse_hprd(x))
                    )
            
            print(f"[OK] Loaded MACPAC standards for {len(macpac_standards_df)} states")
            return macpac_standards_df
        else:
            print(f"[WARN] MACPAC standards file not found: {macpac_file}")
            return None
    except Exception as e:
        print(f"Error loading MACPAC standards: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_pbj_source_link(quarter, date, provnum="225500", data_type="nurse"):
    """Generate PBJ source link for a specific quarter and date."""
    # Convert date to YYYYMMDD format if needed
    if isinstance(date, str):
        if len(date) == 10 and '-' in date:  # YYYY-MM-DD format
            date = date.replace('-', '')
        elif len(date) == 8:  # Already YYYYMMDD
            pass
        else:
            return None
    elif hasattr(date, 'strftime'):
        date = date.strftime('%Y%m%d')
    else:
        return None
    
    # Format quarter for URL (e.g., "2021Q3" -> "q3-2021")
    if isinstance(quarter, str):
        if 'Q' in quarter.upper():
            year = quarter[:4]
            q_num = quarter[-1]
            quarter_url = f"q{q_num}-{year}"
        else:
            return None
    else:
        return None
    
    # Determine column names based on quarter
    # Early quarters (2017-2019, Q2-Q3 2020) use lowercase
    early_quarters = [
        "2017Q1", "2017Q2", "2017Q3", "2017Q4",
        "2018Q4", 
        "2019Q1", "2019Q2", "2019Q3", "2019Q4",
        "2020Q2", "2020Q3"
    ]
    
    if quarter in early_quarters:
        provnum_col = "provnum"
        workdate_col = "workdate"
    else:
        provnum_col = "PROVNUM"
        workdate_col = "WorkDate"
    
    # Generate the query parameters
    query_params = {
        "filters": {
            "list": [{
                "conditions": [
                    {
                        "column": {"value": provnum_col},
                        "comparator": {"value": "="},
                        "filterValue": [provnum]
                    },
                    {
                        "column": {"value": workdate_col},
                        "comparator": {"value": "="},
                        "filterValue": [date]
                    }
                ]
            }],
            "rootConjunction": {"value": "AND"}
        },
        "keywords": "",
        "offset": 0,
        "limit": 10,
        "sort": {"sortBy": None, "sortOrder": None},
        "columns": []
    }
    
    # Convert to JSON and URL encode
    import json
    import urllib.parse
    query_json = json.dumps(query_params)
    encoded_query = urllib.parse.quote(query_json)
    
    # Generate the full URL based on data type
    if data_type == "nurse":
        base_url = "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data"
    else:  # nonnurse
        base_url = "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-non-nurse-staffing/data"
    
    url = f"{base_url}/{quarter_url}?query={encoded_query}"
    
    return url


def generate_cms_pbj_facility_link(provnum, data_type="nurse"):
    """
    Reusable: build CMS data.cms.gov link for a facility filtered by provnum (no date).
    Opens main PBJ dataset with provnum pre-filter. Use for 'View raw CMS data' anchor.
    """
    import urllib.parse
    provnum = str(provnum).strip()
    if provnum.isdigit():
        provnum = provnum.zfill(6)
    # CMS uses PROVNUM for current datasets
    query_params = {
        "filters": {
            "list": [{
                "conditions": [{
                    "column": {"value": "PROVNUM"},
                    "comparator": {"value": "="},
                    "filterValue": [provnum]
                }]
            }],
            "rootConjunction": {"value": "AND"}
        },
        "keywords": "",
        "offset": 0,
        "limit": 10,
        "sort": {"sortBy": None, "sortOrder": None},
        "columns": []
    }
    query_json = json.dumps(query_params)
    encoded_query = urllib.parse.quote(query_json)
    if data_type == "nurse":
        base_url = "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data"
    else:
        base_url = "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-non-nurse-staffing/data"
    return f"{base_url}?query={encoded_query}"


def format_pbj_source_link(quarter, date, provnum="225500", data_type="nurse"):
    """Generate formatted PBJ source link with display text."""
    url = generate_pbj_source_link(quarter, date, provnum, data_type)
    if not url:
        return None
    
    # Format date for display (YYYYMMDD -> MM-DD-YYYY)
    if isinstance(date, str) and len(date) == 8:
        display_date = f"{date[4:6]}-{date[6:8]}-{date[:4]}"
    elif hasattr(date, 'strftime') and not isinstance(date, str):
        display_date = date.strftime('%m-%d-%Y')
    else:
        display_date = str(date)
    
    if data_type == "nurse":
        return f'<a href="{url}" target="_blank" style="font-size: 0.9em;">Source: CMS PBJ Nurse: {display_date}</a>'
    else:  # nonnurse
        return f'<a href="{url}" target="_blank" style="font-size: 0.9em;">Source: CMS PBJ NonNurse: {display_date}</a>'

def load_facility_data(provnum, csv_path=None):
    """Load the facility data. Uses csv_path if given, else file next to this script."""
    global global_df
    path = csv_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), f'facility_{provnum}_complete_data.csv')
    if isinstance(path, str) and (path.startswith("http://") or path.startswith("https://")):
        try:
            global_df = pd.read_csv(path)
            print(f"Loaded {len(global_df)} records from {path} (remote)")
        except Exception as e:
            print(f"Error loading remote data: {e}")
            return None
    else:
        if not os.path.exists(path):
            print(f"Error: facility data file not found: {path}")
            return None
        try:
            global_df = pd.read_csv(path)
            print(f"Loaded {len(global_df)} records from {path}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    # Clean up the data - remove any columns with .1 suffix and handle NaN values
    global_df = global_df.drop(columns=[col for col in global_df.columns if '.1' in col])
    
    # PBJ staffing hours + census: preserve missing vs zero (do not fill NaN with 0).
    _staff_no_fillna: set[str] = set()
    for _c in list(global_df.columns):
        if _c.startswith("Hrs_") or _c == "MDScensus":
            _staff_no_fillna.add(_c)
            global_df[_c] = pd.to_numeric(global_df[_c], errors="coerce")
    _numeric_cols = list(global_df.select_dtypes(include=[np.number]).columns)
    _fill_zero_cols = [c for c in _numeric_cols if c not in _staff_no_fillna]
    if _fill_zero_cols:
        global_df[_fill_zero_cols] = global_df[_fill_zero_cols].fillna(0)
    
    # Apply financial rounding to hours columns (NaN stays NaN)
    hours_columns = ['Hrs_RN', 'Hrs_LPN', 'Hrs_CNA', 'Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_LPNadmin', 
                     'Hrs_RN_ctr', 'Hrs_LPN_ctr', 'Hrs_CNA_ctr', 'Hrs_NAtrn', 'Hrs_MedAide']
    for col in hours_columns:
        if col in global_df.columns:
            global_df[col] = pd.to_numeric(global_df[col], errors="coerce").apply(
                lambda x: float("nan") if pd.isna(x) else round_financial(x, 2)
            )
    
    # Structural columns expected by downstream metrics (synthetic zeros only when column absent from file)
    required_cols = ['Hrs_RN', 'Hrs_RNadmin', 'Hrs_RNDON', 'Hrs_LPN', 'Hrs_LPNadmin', 'Hrs_CNA', 'Hrs_MedAide', 'Hrs_NAtrn', 'MDScensus']
    missing_cols = [col for col in required_cols if col not in global_df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        for col in missing_cols:
            global_df[col] = 0
    
    # Convert WorkDate to datetime (naive, no timezone to avoid date shift issues)
    # Handle both string and integer formats
    if global_df['WorkDate'].dtype == 'object':
        global_df['WorkDate'] = pd.to_datetime(global_df['WorkDate'], format='%Y%m%d', errors='coerce', utc=False)
    elif global_df['WorkDate'].dtype in ['int64', 'int32', 'float64', 'float32']:
        # Convert integer dates (YYYYMMDD format) to datetime
        global_df['WorkDate'] = pd.to_datetime(global_df['WorkDate'].astype(str), format='%Y%m%d', errors='coerce', utc=False)
    else:
        global_df['WorkDate'] = pd.to_datetime(global_df['WorkDate'], errors='coerce', utc=False)
    
    # Add day of week
    global_df['DayOfWeek'] = global_df['WorkDate'].dt.day_name()
    global_df['DayOfWeekNum'] = global_df['WorkDate'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Add month and year
    global_df['Month'] = global_df['WorkDate'].dt.month
    global_df['Year'] = global_df['WorkDate'].dt.year
    
    # Calculate HPRD for each position with proper rounding (NaN when census missing or ≤ 0)
    # RN HPRD includes direct care only (not admin/DON)
    global_df["RN_HPRD"] = _round_fin_series(_divide_hprd(global_df["Hrs_RN"], global_df["MDScensus"]), 2)
    # LPN HPRD includes direct care only (not admin)
    global_df["LPN_HPRD"] = _round_fin_series(_divide_hprd(global_df["Hrs_LPN"], global_df["MDScensus"]), 2)
    # CNA HPRD includes direct care only (not medaide/natr)
    global_df["CNA_HPRD"] = _round_fin_series(_divide_hprd(global_df["Hrs_CNA"], global_df["MDScensus"]), 2)
    # Total HPRD includes ALL staff (RN + RNadmin + RNDON + LPN + LPNadmin + CNA + NAtrn + MedAide)
    _tnh_num = (
        global_df["Hrs_RN"]
        + global_df["Hrs_RNadmin"]
        + global_df["Hrs_RNDON"]
        + global_df["Hrs_LPN"]
        + global_df["Hrs_LPNadmin"]
        + global_df["Hrs_CNA"]
        + global_df["Hrs_NAtrn"]
        + global_df["Hrs_MedAide"]
    )
    global_df["Total_Nurse_HPRD"] = _round_fin_series(_divide_hprd(_tnh_num, global_df["MDScensus"]), 2)
    
    global_df["Total_RN_Hours"] = _round_fin_series(
        global_df["Hrs_RN"] + global_df["Hrs_RNadmin"] + global_df["Hrs_RNDON"], 2
    )
    global_df["Total_RN_HPRD"] = _round_fin_series(_divide_hprd(global_df["Total_RN_Hours"], global_df["MDScensus"]), 2)
    global_df["Total_LPN_Hours"] = _round_fin_series(global_df["Hrs_LPN"] + global_df["Hrs_LPNadmin"], 2)
    global_df["Total_LPN_HPRD"] = _round_fin_series(_divide_hprd(global_df["Total_LPN_Hours"], global_df["MDScensus"]), 2)
    global_df["Total_Nurse_Aide_Hours"] = _round_fin_series(
        global_df["Hrs_CNA"] + global_df["Hrs_MedAide"] + global_df["Hrs_NAtrn"], 2
    )
    global_df["Total_Nurse_Aide_HPRD"] = _round_fin_series(
        _divide_hprd(global_df["Total_Nurse_Aide_Hours"], global_df["MDScensus"]), 2
    )
    
    # Nurse Staff Hours (excluding Admin & DON) - includes all direct care staff
    global_df["Nurse_Staff_Hours_Excl_Admin"] = _round_fin_series(
        global_df["Hrs_RN"] + global_df["Hrs_LPN"] + global_df["Hrs_CNA"] + global_df["Hrs_NAtrn"] + global_df["Hrs_MedAide"], 2
    )
    global_df["Nurse_Staff_HPRD_Excl_Admin"] = _round_fin_series(
        _divide_hprd(global_df["Nurse_Staff_Hours_Excl_Admin"], global_df["MDScensus"]), 2
    )
    
    # Total Nurse Hours (All Staff including admin/DON)
    global_df["Total_Nurse_Hours"] = _round_fin_series(
        global_df["Hrs_RN"]
        + global_df["Hrs_RNadmin"]
        + global_df["Hrs_RNDON"]
        + global_df["Hrs_LPN"]
        + global_df["Hrs_LPNadmin"]
        + global_df["Hrs_CNA"]
        + global_df["Hrs_NAtrn"]
        + global_df["Hrs_MedAide"],
        2,
    )
    
    # Total Staff Hours and HPRD
    global_df["Total_Staff_Hours"] = _round_fin_series(
        global_df["Total_RN_Hours"] + global_df["Total_LPN_Hours"] + global_df["Total_Nurse_Aide_Hours"], 2
    )
    global_df["Total_Staff_HPRD"] = _round_fin_series(_divide_hprd(global_df["Total_Staff_Hours"], global_df["MDScensus"]), 2)
    
    # Calculate contract percentages (NaN when direct hours are missing or zero — not "0% contract")
    global_df["RN_Contract_Pct"] = _round_fin_series(_contract_pct_series(global_df["Hrs_RN_ctr"], global_df["Hrs_RN"]), 1)
    global_df["LPN_Contract_Pct"] = _round_fin_series(_contract_pct_series(global_df["Hrs_LPN_ctr"], global_df["Hrs_LPN"]), 1)
    global_df["CNA_Contract_Pct"] = _round_fin_series(_contract_pct_series(global_df["Hrs_CNA_ctr"], global_df["Hrs_CNA"]), 1)
    
    # Calculate more granular contract percentages
    # CNA contract percentage (CNA only)
    global_df["CNA_Only_Contract_Pct"] = _round_fin_series(
        _contract_pct_series(global_df["Hrs_CNA_ctr"], global_df["Hrs_CNA"]), 1
    )
    
    # Nurse Aide contract percentage (CNA + MedAide + NAtrn)
    total_nurse_aide_hours = global_df["Hrs_CNA"] + global_df["Hrs_MedAide"] + global_df["Hrs_NAtrn"]
    _mac = (
        global_df["Hrs_CNA_ctr"]
        + (global_df["Hrs_MedAide_ctr"] if "Hrs_MedAide_ctr" in global_df.columns else 0)
        + (global_df["Hrs_NAtrn_ctr"] if "Hrs_NAtrn_ctr" in global_df.columns else 0)
    )
    global_df["Nurse_Aide_Contract_Pct"] = _round_fin_series(_contract_pct_series(_mac, total_nurse_aide_hours), 1)
    
    # LPN contract percentage (LPN only, excluding admin)
    global_df["LPN_Only_Contract_Pct"] = _round_fin_series(
        _contract_pct_series(global_df["Hrs_LPN_ctr"], global_df["Hrs_LPN"]), 1
    )
    
    # Total LPN contract percentage (LPN + LPN admin)
    total_lpn_hours = global_df["Hrs_LPN"] + global_df["Hrs_LPNadmin"]
    total_lpn_contract_hours = global_df["Hrs_LPN_ctr"] + (
        global_df["Hrs_LPNadmin_ctr"] if "Hrs_LPNadmin_ctr" in global_df.columns else 0
    )
    global_df["Total_LPN_Contract_Pct"] = _round_fin_series(
        _contract_pct_series(total_lpn_contract_hours, total_lpn_hours), 1
    )
    
    # Total Contract Percentage (Direct care contract hours / Direct care total hours)
    # Only use contract hours that actually exist in the data (direct care staff)
    total_contract_hours = global_df["Hrs_RN_ctr"] + global_df["Hrs_LPN_ctr"] + global_df["Hrs_CNA_ctr"]
    total_direct_care_hours = global_df["Hrs_RN"] + global_df["Hrs_LPN"] + global_df["Hrs_CNA"]
    global_df["Total_Contract_Pct"] = _round_fin_series(
        _contract_pct_series(total_contract_hours, total_direct_care_hours), 1
    )
    
    # Direct Care HPRD (excluding admin/DON staff: RN admin, RN DON, LPN admin)
    # Direct care includes: RN (direct care only), LPN (direct care only), CNA, NAtrn, MedAide
    # This is the same as Nurse_Staff_HPRD_Excl_Admin, but we'll keep this column name for clarity
    global_df['Direct_Care_HPRD'] = global_df['Nurse_Staff_HPRD_Excl_Admin']
    
    # Add holiday indicator (comprehensive federal holidays)
    def is_federal_holiday(date):
        """Check if a date is a US federal holiday"""
        year = date.year
        month = date.month
        day = date.day
        
        # Fixed holidays
        fixed_holidays = [
            (1, 1),   # New Year's Day
            (6, 19),  # Juneteenth (since 2021)
            (7, 4),   # Independence Day
            (11, 11), # Veterans Day
            (12, 25), # Christmas Day
        ]
        
        # Check fixed holidays
        if (month, day) in fixed_holidays:
            # Juneteenth only became a federal holiday in 2021
            if month == 6 and day == 19 and year < 2021:
                return False
            return True
        
        # Variable holidays (calculated for each year)
        # Martin Luther King Jr. Day (3rd Monday in January)
        mlk_day = get_third_monday(year, 1)
        if date == mlk_day:
            return True
        
        # Presidents Day (3rd Monday in February)
        presidents_day = get_third_monday(year, 2)
        if date == presidents_day:
            return True
        
        # Memorial Day (last Monday in May)
        memorial_day = get_last_monday(year, 5)
        if date == memorial_day:
            return True
        
        # Labor Day (1st Monday in September)
        labor_day = get_first_monday(year, 9)
        if date == labor_day:
            return True
        
        # Columbus Day (2nd Monday in October)
        columbus_day = get_second_monday(year, 10)
        if date == columbus_day:
            return True
        
        # Thanksgiving (4th Thursday in November)
        thanksgiving = get_fourth_thursday(year, 11)
        if date == thanksgiving:
            return True
        
        return False
    
    def get_first_monday(year, month):
        """Get the first Monday of a given month/year"""
        first_day = pd.Timestamp(year, month, 1)
        days_ahead = 0 - first_day.weekday()  # Monday is 0
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return first_day + pd.Timedelta(days=days_ahead)
    
    def get_second_monday(year, month):
        """Get the second Monday of a given month/year"""
        return get_first_monday(year, month) + pd.Timedelta(days=7)
    
    def get_third_monday(year, month):
        """Get the third Monday of a given month/year"""
        return get_first_monday(year, month) + pd.Timedelta(days=14)
    
    def get_fourth_thursday(year, month):
        """Get the fourth Thursday of a given month/year"""
        first_day = pd.Timestamp(year, month, 1)
        days_ahead = 3 - first_day.weekday()  # Thursday is 3
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return first_day + pd.Timedelta(days=days_ahead + 21)  # 4th Thursday
    
    def get_last_monday(year, month):
        """Get the last Monday of a given month/year"""
        # Get the first day of next month, then go back to find last Monday
        if month == 12:
            next_month = pd.Timestamp(year + 1, 1, 1)
        else:
            next_month = pd.Timestamp(year, month + 1, 1)
        
        # Go back to the last day of current month
        last_day = next_month - pd.Timedelta(days=1)
        
        # Find the last Monday
        days_back = last_day.weekday()  # Monday is 0
        return last_day - pd.Timedelta(days=days_back)

    # Apply holiday detection to all dates
    global_df['IsHoliday'] = global_df['WorkDate'].apply(is_federal_holiday)
    
    # Handle WorkDate - convert to datetime if needed
    min_date = global_df['WorkDate'].min()
    max_date = global_df['WorkDate'].max()
    if not isinstance(min_date, pd.Timestamp):
        min_date = pd.to_datetime(min_date)
    if not isinstance(max_date, pd.Timestamp):
        max_date = pd.to_datetime(max_date)
    print(f"Loaded {len(global_df)} records from {min_date.date()} to {max_date.date()}")
    print(f"Calculated columns: {[col for col in global_df.columns if 'Total' in col or 'HPRD' in col]}")
    
    # Verify critical columns exist
    critical_cols = ['Total_Staff_HPRD', 'Total_Staff_Hours', 'Total_RN_HPRD', 'Total_LPN_HPRD', 'Total_Nurse_Aide_HPRD']
    missing_cols = [col for col in critical_cols if col not in global_df.columns]
    if missing_cols:
        print(f"[ERROR] MISSING CRITICAL COLUMNS: {missing_cols}")
    else:
        print(f"All critical columns present")
    
    return global_df


def _ensure_pbj_dashboard_derived_columns_inplace(df: Optional[pd.DataFrame]) -> None:
    """
    Recompute core derived daily columns when missing (partial load, older CSV, or rare race).

    Mirrors the post-read logic in ``load_facility_data`` so API handlers do not KeyError on
    ``RN_Contract_Pct`` / ``Total_Staff_Hours`` / contract % columns.
    """
    if df is None:
        return
    if len(df) == 0:
        return
    if "Total_RN_Hours" not in df.columns and all(c in df.columns for c in ("Hrs_RN", "Hrs_RNadmin", "Hrs_RNDON")):
        df["Total_RN_Hours"] = _round_fin_series(
            pd.to_numeric(df["Hrs_RN"], errors="coerce")
            + pd.to_numeric(df["Hrs_RNadmin"], errors="coerce")
            + pd.to_numeric(df["Hrs_RNDON"], errors="coerce"),
            2,
        )
    if "Total_LPN_Hours" not in df.columns and all(c in df.columns for c in ("Hrs_LPN", "Hrs_LPNadmin")):
        df["Total_LPN_Hours"] = _round_fin_series(
            pd.to_numeric(df["Hrs_LPN"], errors="coerce") + pd.to_numeric(df["Hrs_LPNadmin"], errors="coerce"),
            2,
        )
    if "Total_Nurse_Aide_Hours" not in df.columns and all(
        c in df.columns for c in ("Hrs_CNA", "Hrs_MedAide", "Hrs_NAtrn")
    ):
        df["Total_Nurse_Aide_Hours"] = _round_fin_series(
            pd.to_numeric(df["Hrs_CNA"], errors="coerce")
            + pd.to_numeric(df["Hrs_MedAide"], errors="coerce")
            + pd.to_numeric(df["Hrs_NAtrn"], errors="coerce"),
            2,
        )
    if "Nurse_Staff_Hours_Excl_Admin" not in df.columns and all(
        c in df.columns for c in ("Hrs_RN", "Hrs_LPN", "Hrs_CNA", "Hrs_NAtrn", "Hrs_MedAide")
    ):
        df["Nurse_Staff_Hours_Excl_Admin"] = _round_fin_series(
            pd.to_numeric(df["Hrs_RN"], errors="coerce")
            + pd.to_numeric(df["Hrs_LPN"], errors="coerce")
            + pd.to_numeric(df["Hrs_CNA"], errors="coerce")
            + pd.to_numeric(df["Hrs_NAtrn"], errors="coerce")
            + pd.to_numeric(df["Hrs_MedAide"], errors="coerce"),
            2,
        )
    if "Total_Staff_Hours" not in df.columns:
        trn = df["Total_RN_Hours"] if "Total_RN_Hours" in df.columns else 0.0
        tln = df["Total_LPN_Hours"] if "Total_LPN_Hours" in df.columns else 0.0
        tna = df["Total_Nurse_Aide_Hours"] if "Total_Nurse_Aide_Hours" in df.columns else 0.0
        df["Total_Staff_Hours"] = _round_fin_series(trn + tln + tna, 2)
    if "RN_Contract_Pct" not in df.columns and "Hrs_RN_ctr" in df.columns and "Hrs_RN" in df.columns:
        df["RN_Contract_Pct"] = _round_fin_series(
            _contract_pct_series(df["Hrs_RN_ctr"], df["Hrs_RN"]),
            1,
        )
    if "LPN_Contract_Pct" not in df.columns and "Hrs_LPN_ctr" in df.columns and "Hrs_LPN" in df.columns:
        df["LPN_Contract_Pct"] = _round_fin_series(
            _contract_pct_series(df["Hrs_LPN_ctr"], df["Hrs_LPN"]),
            1,
        )
    if "CNA_Contract_Pct" not in df.columns and "Hrs_CNA_ctr" in df.columns and "Hrs_CNA" in df.columns:
        df["CNA_Contract_Pct"] = _round_fin_series(
            _contract_pct_series(df["Hrs_CNA_ctr"], df["Hrs_CNA"]),
            1,
        )
    if "IsHoliday" not in df.columns:
        df["IsHoliday"] = False


# Words shown lowercase in chain / entity names (unless first word). Acronyms stay uppercase.
_AFFILIATED_ENTITY_SMALL_WORDS = frozenset(
    {"a", "an", "the", "and", "or", "but", "nor", "for", "of", "in", "on", "at", "to", "from", "vs", "v.", "as", "by"}
)
_AFFILIATED_ENTITY_ACRONYMS = frozenset(
    {
        "LLC",
        "INC",
        "LP",
        "PA",
        "PC",
        "PLLC",
        "LLP",
        "SNF",
        "RN",
        "LPN",
        "CNA",
        "LVN",
        "PACS",
        "CMS",
        "HMO",
        "PPO",
        "IPA",
        "LTAC",
        "IRF",
        "DME",
        "NH",
        "I",
        "II",
        "III",
        "IV",
        "V",
        "EIN",
        "NA",
        "MD",
        "DO",
        "NP",
        "DON",
    }
)


def _latest_affiliated_entity_fields(
    provider_info_df: pd.DataFrame | None,
    facility_provnum: str,
) -> tuple[str | None, str | None]:
    """Most recent provider-info row with a usable affiliated entity (not necessarily latest row)."""
    if provider_info_df is None or provider_info_df.empty:
        return None, None
    if "affiliated_entity_name" not in provider_info_df.columns:
        return None, None
    pi = provider_info_df.copy()
    if "ccn" in pi.columns:
        pi = pi[
            pi["ccn"].astype(str).str.strip().str.zfill(6)
            == str(facility_provnum).strip().zfill(6)
        ]
    if pi.empty:
        return None, None
    invalid = {"", "N", "N/A", "NAN", "NONE", "UNKNOWN"}

    def _valid_name(val) -> bool:
        if pd.isna(val):
            return False
        s = str(val).strip()
        return bool(s) and s.upper() not in invalid

    named = pi[pi["affiliated_entity_name"].apply(_valid_name)]
    if named.empty:
        return None, None
    if "processing_date" in named.columns:
        named = named.copy()
        named["processing_date"] = pd.to_datetime(named["processing_date"], errors="coerce")
        named = named.sort_values("processing_date", ascending=False)
    row = named.iloc[0]
    raw_name = str(row.get("affiliated_entity_name") or "").strip()
    raw_id = row.get("affiliated_entity_id")
    entity_id: str | None = None
    if pd.notna(raw_id):
        rid = str(raw_id).strip()
        if rid and rid.upper() not in invalid:
            try:
                entity_id = str(int(float(rid)))
            except Exception:
                entity_id = rid
    return raw_name or None, entity_id


def format_affiliated_entity_display(name: str | None) -> str | None:
    """
    Title-style formatting for CMS affiliated-entity strings: capitalize words,
    keep small words (and, of, …) lowercase except when first, preserve known acronyms.
    """
    if name is None:
        return None
    raw = str(name).strip()
    if not raw:
        return raw

    parts = re.split(r"(\s+)", raw)
    out: list[str] = []
    word_index = 0
    for segment in parts:
        if not segment or segment.isspace():
            out.append(segment)
            continue
        m = re.match(r"^([\W]*)([\w&'.-]+)([\W]*)$", segment, re.UNICODE)
        if not m:
            out.append(segment)
            continue
        lead, core, trail = m.group(1), m.group(2), m.group(3)
        key = re.sub(r"[^\w]", "", core).upper()
        is_first = word_index == 0
        word_index += 1
        if key in _AFFILIATED_ENTITY_ACRONYMS:
            fixed = core.upper()
        elif not is_first and core.lower() in _AFFILIATED_ENTITY_SMALL_WORDS:
            fixed = core.lower()
        elif len(core) == 1:
            fixed = core.upper()
        else:
            fixed = core[:1].upper() + core[1:].lower()
        out.append(lead + fixed + trail)
    return "".join(out)


_FACILITY_TITLE_MINOR_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "nor",
        "as",
        "at",
        "by",
        "for",
        "if",
        "in",
        "of",
        "on",
        "so",
        "to",
        "up",
        "yet",
        "with",
    }
)


def format_facility_display_name(name: object) -> str:
    """
    Sentence-style facility name for <title> and Open Graph: capitalize words,
    keep minor words (at, of, the, …) lowercase except the first word.
    """
    if name is None or (isinstance(name, float) and pd.isna(name)):
        return "Unknown facility"
    raw = str(name).strip()
    if not raw:
        return "Unknown facility"
    tokens = raw.replace("-", " - ").replace("&", " & ").split()
    out: list[str] = []
    for i, word in enumerate(tokens):
        if word == "-":
            out.append("-")
            continue
        if word == "&":
            out.append("&")
            continue
        lw = word.lower()
        if i > 0 and lw in _FACILITY_TITLE_MINOR_WORDS:
            out.append(lw)
        elif len(word) == 1:
            out.append(word.upper())
        else:
            out.append(word[:1].upper() + word[1:].lower())
    return " ".join(out)


def get_previous_provider_names(provnum, limit: int | None = 3):
    """Get previous provider names that are different from the most recent, with year ranges (e.g. 'Name (2017-23)')."""
    global provider_info_df

    if provider_info_df is None or provider_info_df.empty:
        return []

    # Filter to this facility
    facility_data = provider_info_df[provider_info_df['ccn'] == str(provnum).zfill(6)].copy()
    if facility_data.empty:
        return []

    if 'processing_date' not in facility_data.columns:
        facility_data = facility_data.assign(processing_date=pd.NaT)
    facility_data['processing_date'] = pd.to_datetime(facility_data['processing_date'], errors='coerce')

    # Get the most recent name (by processing_date)
    facility_sorted = facility_data.sort_values('processing_date', ascending=False)
    most_recent_name = None
    if len(facility_sorted) > 0 and 'provider_name' in facility_sorted.columns:
        first = facility_sorted['provider_name'].dropna().iloc[0]
        if pd.notna(first) and str(first).strip():
            most_recent_name = str(first).strip()

    def format_facility_name(name):
        words = name.replace('-', ' - ').replace('&', ' & ').split()
        formatted_words = []
        for word in words:
            if word.lower() in ['at', 'and', 'of', 'the', 'for', 'in', 'on', 'to', 'with']:
                formatted_words.append(word.lower())
            elif word == '-':
                formatted_words.append('-')
            elif word == '&':
                formatted_words.append('&')
            else:
                formatted_words.append(word.capitalize())
        return ' '.join(formatted_words)

    def year_range_str(dates_series):
        """Format a series of dates as 'YYYY-YY' (e.g. 2017-23)."""
        valid = dates_series.dropna()
        if len(valid) == 0:
            return None
        min_d = valid.min()
        max_d = valid.max()
        if pd.isna(min_d) or pd.isna(max_d):
            return None
        try:
            y1, y2 = int(min_d.year), int(max_d.year)
            if y1 == y2:
                return str(y1)
            return f"{y1}-{str(y2)[-2:]}"
        except (ValueError, TypeError):
            return None

    # Group by provider_name and get date range for each
    name_dates = facility_data.groupby(
        facility_data['provider_name'].astype(str).str.strip().str.lower()
    )['processing_date'].apply(lambda s: year_range_str(s)).to_dict()

    # Build list of (name_display, year_str) for names that are not the most recent
    seen_normalized = set()
    previous_with_years = []
    for _, row in facility_sorted.iterrows():
        name = row.get('provider_name')
        if pd.isna(name) or not str(name).strip():
            continue
        name_str = str(name).strip()
        name_lower = name_str.lower()
        if name_lower == (most_recent_name or '').lower():
            continue
        if name_lower in seen_normalized:
            continue
        seen_normalized.add(name_lower)
        year_str = name_dates.get(name_lower)
        display_name = format_facility_name(name_str)
        if year_str:
            previous_with_years.append(f"{display_name} ({year_str})")
        else:
            previous_with_years.append(display_name)
        if limit is not None and len(previous_with_years) >= limit:
            break

    return previous_with_years


def _ein_active_ccn() -> str:
    """CCN for EIN CSV filenames and CMS links (from loaded PBJ data or PROVNUM)."""
    global global_df, PROVNUM
    if global_df is not None and len(global_df) > 0 and "PROVNUM" in global_df.columns:
        return str(global_df["PROVNUM"].iloc[0]).strip().zfill(6)
    p = str(PROVNUM).strip()
    return p.zfill(6) if p.isdigit() else p


def _ein_mode_enabled() -> bool:
    """True when EIN features should be enabled for this app instance."""
    return EIN_DASHBOARD_MODE in {"all", "selected"}


def _filter_ein_by_selected_quarters(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Apply quarter filter only when EIN_DASHBOARD_MODE == 'selected'."""
    if df is None or df.empty:
        return df
    if EIN_DASHBOARD_MODE != "selected":
        return df
    if not EIN_SELECTED_QUARTERS:
        return df.iloc[0:0].copy()
    if "CY_Qtr" not in df.columns:
        return df
    qset = {str(q).strip() for q in EIN_SELECTED_QUARTERS}
    qlist = list(qset)
    mask = df["CY_Qtr"].astype(str).isin(qlist)
    return cast(pd.DataFrame, df.loc[mask].copy())


def _load_ein_position_csvs(provnum: str | None = None) -> None:
    """Load facility EIN tables: prefer ``deployments/pbj320-<CCN>/``, then app root (legacy).

    If ``facility_*_ein_nursing_summaries.parquet`` exists, row-level detail is not loaded until
    an endpoint needs it (series / roster / bridge), which keeps memory low.
    """
    global ein_job_quarterly_df, ein_category_quarterly_df, ein_employee_detail_df
    global ein_nursing_summaries_df
    if not _ein_mode_enabled():
        ein_job_quarterly_df = None
        ein_category_quarterly_df = None
        ein_employee_detail_df = None
        ein_nursing_summaries_df = None
        return
    from file_path_utils import find_facility_ein_table_base

    prov = str(provnum or _ein_active_ccn()).strip().zfill(6)

    def _base(kind: str) -> str:
        found = find_facility_ein_table_base(prov, kind)
        if found:
            return found
        return os.path.join(_app_root, f"facility_{prov}_ein_{kind}")

    job_base = _base("job_quarterly")
    cat_base = _base("category_quarterly")
    detail_base = _base("employee_detail")
    summ_base = _base("nursing_summaries")
    try:
        ein_job_quarterly_df = _filter_ein_by_selected_quarters(read_facility_ein_parquet_or_csv(job_base))
        ein_category_quarterly_df = _filter_ein_by_selected_quarters(read_facility_ein_parquet_or_csv(cat_base))
        summ_df = read_facility_ein_parquet_or_csv(summ_base)
        if summ_df is not None and not summ_df.empty:
            ein_nursing_summaries_df = summ_df
            if EIN_DASHBOARD_MODE == "selected" and EIN_SELECTED_QUARTERS and "quarter" in ein_nursing_summaries_df.columns:
                qset = {str(q).strip() for q in EIN_SELECTED_QUARTERS}
                qlist = list(qset)
                ein_nursing_summaries_df = ein_nursing_summaries_df[
                    ein_nursing_summaries_df["quarter"].astype(str).isin(qlist)
                ].copy()
            ein_employee_detail_df = None
            print(
                f"[EIN] Precomputed nursing summaries ({len(ein_nursing_summaries_df)} rows); "
                "row-level detail loads on demand."
            )
        else:
            ein_nursing_summaries_df = None
            ein_employee_detail_df = _filter_ein_by_selected_quarters(read_facility_ein_parquet_or_csv(detail_base))
        if (
            ein_job_quarterly_df is not None
            or ein_category_quarterly_df is not None
            or ein_employee_detail_df is not None
            or ein_nursing_summaries_df is not None
        ):
            print(f"[EIN] Loaded tables for {prov} (detail={detail_base})")
            _schedule_ein_roster_prewarm(prov)
        else:
            print(
                f"[EIN] No Employee Detail extract found for {prov}. "
                f"Expected `facility_{prov}_ein_job_quarterly` + `_category_quarterly` "
                f"(and `_employee_detail` for row-level IDs) under {_app_root}. "
                f"Run: python scripts/extract_facility_ein_from_zip.py {prov} "
                f"--out-dir deployments/pbj320-{prov}"
            )
    except Exception as exc:
        print(f"[EIN] Could not load EIN tables: {exc}")
        ein_job_quarterly_df = None
        ein_category_quarterly_df = None
        ein_employee_detail_df = None
        ein_nursing_summaries_df = None


def _nonnurse_row_for_work_date_iso(work_date_iso: str) -> Any:
    """Return the first non-nurse PBJ daily row for ``YYYY-MM-DD`` when the slice is loaded."""
    if not _ensure_nonnurse_loaded():
        return None
    global nonnurse_df
    if nonnurse_df is None or nonnurse_df.empty or not str(work_date_iso).strip():
        return None
    try:
        from nonnurse_staffing_lib import prepare_nonnurse_dataframe

        d = prepare_nonnurse_dataframe(nonnurse_df)
        day = pd.Timestamp(str(work_date_iso).strip()[:10]).normalize()
        wd = pd.to_datetime(d["WorkDate"], errors="coerce").dt.normalize()
        sub = d.loc[wd == day]
        if sub.empty:
            return None
        return sub.iloc[0]
    except Exception:
        return None


def _merge_nonnurse_hours_into_target_metrics(target_metrics: dict[str, Any], work_date_iso: str) -> None:
    """Attach raw non-nurse ``Hrs_*`` hours from ``nonnurse_df`` for the single-day staffing modal."""
    if not INCLUDE_NONNURSE:
        return
    row = _nonnurse_row_for_work_date_iso(work_date_iso)
    if row is None:
        return
    for col in PBJ_NONNURSE_HRS_COLUMN_TO_EIN_JOB_CODES:
        if col not in row.index:
            continue
        v = row[col]
        if pd.isna(v):
            target_metrics[col] = None
        else:
            target_metrics[col] = round_financial(float(v), 2)


def _to_float_or_zero(value: Any) -> float:
    """Best-effort numeric coercion for non-nurse day summaries."""
    try:
        if value is None or isinstance(value, bool):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        s = str(value).strip()
        if not s:
            return 0.0
        return float(s.replace(",", ""))
    except Exception:
        return 0.0


def _nonnurse_day_summary_payload(work_date_iso: str) -> dict[str, Any]:
    """Summarize one non-nurse PBJ day with exact CMS PBJ Hrs_* ↔ EIN job-code mapping."""
    row = _nonnurse_row_for_work_date_iso(work_date_iso)
    if row is None:
        return {
            "ok": True,
            "available": False,
            "date": work_date_iso,
            "message": "No non-nurse row found for this date.",
            "roles": [],
        }

    roles: list[dict[str, Any]] = []
    total_hours = 0.0
    total_emp_hours = 0.0
    total_ctr_hours = 0.0
    for col_name, mapped_codes in PBJ_NONNURSE_HRS_COLUMN_TO_EIN_JOB_CODES.items():
        hrs = _to_float_or_zero(row.get(col_name))
        emp = _to_float_or_zero(row.get(f"{col_name}_emp"))
        ctr = _to_float_or_zero(row.get(f"{col_name}_ctr"))
        if hrs <= 0 and emp <= 0 and ctr <= 0:
            continue
        codes = [int(c) for c in mapped_codes]
        code_label = ", ".join(str(c) for c in codes)
        role_label = (
            job_title(codes[0]) if len(codes) == 1 else " / ".join(job_title(c) for c in codes)
        )
        total_hours += hrs
        total_emp_hours += emp
        total_ctr_hours += ctr
        roles.append(
            {
                "metric_key": col_name,
                "role": role_label,
                "ein_job_codes": codes,
                "ein_job_codes_label": code_label,
                "hours_total": round_financial(hrs, 2),
                "hours_employee": round_financial(emp, 2),
                "hours_contract": round_financial(ctr, 2),
                "employee_share_pct": round_financial((emp / hrs) * 100.0, 1) if hrs > 0 else None,
                "contract_share_pct": round_financial((ctr / hrs) * 100.0, 1) if hrs > 0 else None,
            }
        )
    roles.sort(key=lambda r: (str(r.get("role") or ""), str(r.get("metric_key") or "")))

    ccn = _ein_active_ccn()
    prov_cms = str(ccn).strip().zfill(6) if str(ccn).strip().isdigit() else str(ccn).strip()
    q = str(row.get("CY_Qtr") or "").strip()
    cms_url = None
    try:
        from pbj_identifiers.urls import cms_pbj_daily_staffing_explorer_url

        if prov_cms and q and work_date_iso:
            cms_url = cms_pbj_daily_staffing_explorer_url(q, work_date_iso, prov_cms, data_type="nonnurse")
    except Exception:
        cms_url = None

    return {
        "ok": True,
        "available": True,
        "date": work_date_iso,
        "cy_qtr": q,
        "day_census": _to_float_or_zero(row.get("MDScensus")),
        "hours_total": round_financial(total_hours, 2),
        "hours_employee_total": round_financial(total_emp_hours, 2),
        "hours_contract_total": round_financial(total_ctr_hours, 2),
        "roles": roles,
        "cms_nonnurse_daily_url": cms_url,
    }


def _dashboard_auth_cookie_value(expected_password: str) -> str:
    seed = f"pbj320-dashboard-auth::{expected_password}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _dashboard_password_aliases() -> list[str]:
    """Optional extra passwords (comma or pipe separated). Cookie always uses ``PBJ_DASHBOARD_PASSWORD`` only."""
    raw = (os.environ.get("PBJ_DASHBOARD_PASSWORD_ALIASES") or "").strip()
    if not raw:
        return []
    out: list[str] = []
    for part in re.split(r"[|,]", raw):
        p = part.strip()
        if p:
            out.append(p)
    return out


def _dashboard_password_accepted(posted: str, primary: str) -> bool:
    if not primary:
        return False
    if secrets.compare_digest(posted, primary):
        return True
    for alt in _dashboard_password_aliases():
        if secrets.compare_digest(posted, alt):
            return True
    return False


def _dashboard_login_response(expected_password: str, *, error: str = "", status_code: int = 401) -> Response:
    msg = f'<p style="color:#b91c1c;margin:0 0 12px 0;">{error}</p>' if error else ""
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PBJ320 Login</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#f5f7fb; margin:0; }}
    .wrap {{ max-width:420px; margin:8vh auto; background:white; padding:24px; border-radius:12px; box-shadow:0 8px 24px rgba(0,0,0,.08); }}
    h1 {{ margin:0 0 8px 0; font-size:1.3rem; }}
    p {{ color:#4b5563; }}
    input[type=password] {{ width:100%; padding:10px; border:1px solid #d1d5db; border-radius:8px; }}
    .row {{ margin-top:12px; display:flex; gap:8px; align-items:center; }}
    button {{ margin-top:14px; width:100%; padding:10px; border:0; border-radius:8px; background:#1f5fbf; color:#fff; font-weight:600; cursor:pointer; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>PBJ320 Dashboard</h1>
    <p>Enter the dashboard password.</p>
    {msg}
    <form method="get" action="">
      <input name="next" type="hidden" value="{(request.path or '/')}">
      <input type="password" name="password" autocomplete="current-password" placeholder="Password" required>
      <div class="row"><input type="checkbox" id="remember" name="remember" value="1"><label for="remember">Remember this computer</label></div>
      <button type="submit">Continue</button>
    </form>
  </div>
</body>
</html>"""
    return Response(html, status_code, {"Content-Type": "text/html; charset=utf-8"})


def _dashboard_basic_auth_challenge() -> ResponseReturnValue | None:
    """Password gate using an HTTP-only cookie (no username field)."""
    expected = (os.environ.get("PBJ_DASHBOARD_PASSWORD") or "").strip()
    if not expected:
        return None
    if request.method == "OPTIONS":
        return None

    # Premium HTML on www.pbj320.com loads v2 JS from https://pbj320-<CCN>.vercel.app/static/...
    # Cross-origin <script> requests do not send the pbj320.com auth cookie; must not return login HTML.
    p = (request.path or "").strip()
    if p.startswith("/static/") or p in ("/favicon.ico", "/pbj_favicon.png"):
        return None

    # Keep premium embeds/data loaders working from pbj320.com without forcing browser auth dialogs.
    if request.path.startswith("/api/") and _pbj_cors_reflect_origin(request.headers.get("Origin")):
        return None

    cookie_name = "pbj_dashboard_auth"
    expected_cookie = _dashboard_auth_cookie_value(expected)
    cookie_val = (request.cookies.get(cookie_name) or "").strip()
    if cookie_val and secrets.compare_digest(cookie_val, expected_cookie):
        return None

    posted = (request.form.get("password") or request.args.get("password") or "").strip()
    if posted:
        if _dashboard_password_accepted(posted, expected):
            current_path = (request.path or "/").strip() or "/"
            nxt = (request.form.get("next") or request.args.get("next") or current_path).strip()
            if not nxt.startswith("/"):
                nxt = current_path
            if nxt in ("/__pbj_login", "/__pbj_logout"):
                nxt = current_path
            resp = redirect(nxt)
            remember = str(request.form.get("remember") or request.args.get("remember") or "").strip() == "1"
            resp.set_cookie(
                cookie_name,
                expected_cookie,
                max_age=(60 * 60 * 24 * 30) if remember else None,
                httponly=True,
                secure=request.is_secure,
                samesite="Lax",
                path="/",
            )
            return resp
        return _dashboard_login_response(expected, error="Invalid password.")

    if request.path == "/__pbj_login":
        return _dashboard_login_response(expected, status_code=200)
    if request.path == "/__pbj_logout":
        resp = redirect("/__pbj_login")
        resp.delete_cookie(cookie_name, path="/")
        return resp

    # JSON clients expect JSON, not the HTML login page (avoids silent "empty" charts).
    if request.path.startswith("/api"):
        body = json.dumps({"error": "login_required", "login_path": "/__pbj_login"})
        return Response(
            body,
            401,
            {"Content-Type": "application/json; charset=utf-8"},
        )

    return _dashboard_login_response(expected)


@app.route('/')
def index():
    """Main dashboard page"""
    global global_df, provider_info_df, macpac_standards_df
    if global_df is not None and not global_df.empty:
        # Sort by WorkDate to get most recent data first
        if 'WorkDate' in global_df.columns:
            global_df_sorted = global_df.sort_values('WorkDate', ascending=False)
        else:
            global_df_sorted = global_df
        
        # Safety check: ensure sorted dataframe is not empty
        if global_df_sorted.empty or len(global_df_sorted) == 0:
            facility_provnum = "Unknown"
            city = "Unknown"
            state = "Unknown"
            county_name = "Unknown"
        else:
            facility_provnum = str(global_df_sorted['PROVNUM'].iloc[0]).zfill(6) if 'PROVNUM' in global_df_sorted.columns and len(global_df_sorted) > 0 else "Unknown"
            city = global_df_sorted['CITY'].iloc[0] if 'CITY' in global_df_sorted.columns and len(global_df_sorted) > 0 else "Unknown"
            state = global_df_sorted['STATE'].iloc[0] if 'STATE' in global_df_sorted.columns and len(global_df_sorted) > 0 else "Unknown"
            county_name = global_df_sorted['COUNTY_NAME'].iloc[0] if 'COUNTY_NAME' in global_df_sorted.columns and len(global_df_sorted) > 0 else "Unknown"
        
        # Get facility name from provider_info_df (most recent) if available, otherwise use PBJ data
        if global_df_sorted.empty or len(global_df_sorted) == 0:
            facility_name = "Unknown Facility"
        else:
            facility_name = global_df_sorted['PROVNAME'].iloc[0] if 'PROVNAME' in global_df_sorted.columns and len(global_df_sorted) > 0 else "Unknown Facility"
        if provider_info_df is not None and not provider_info_df.empty:
            if 'provider_name' in provider_info_df.columns:
                # Get the most recent provider name (sorted by processing_date)
                if 'processing_date' in provider_info_df.columns:
                    provider_info_sorted = provider_info_df.sort_values('processing_date', ascending=False)
                    latest_name = provider_info_sorted['provider_name'].iloc[0]
                    if pd.notna(latest_name) and str(latest_name).strip():
                        facility_name = str(latest_name).strip()
                else:
                    latest_name = provider_info_df['provider_name'].iloc[-1]  # Last row if no date
                    if pd.notna(latest_name) and str(latest_name).strip():
                        facility_name = str(latest_name).strip()
    else:
        facility_name = "Unknown Facility"
        facility_provnum = "Unknown"
        city = "Unknown"
        state = "Unknown"
        county_name = "Unknown"

    if str(facility_provnum or "").strip() in ("", "Unknown") and str(PROVNUM or "").strip():
        facility_provnum = str(PROVNUM).strip().zfill(6)
    
    # Check if state has a non-federal minimum for State Compliance Review section
    has_state_standard = False
    state_standard_text = ""
    if state and macpac_standards_df is not None and len(macpac_standards_df) > 0:
        # State abbreviation to full name mapping
        state_abbrev_to_name = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
            'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
            'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
            'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
            'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
            'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
            'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
        }
        state_name = state_abbrev_to_name.get(state.upper(), state)
        state_name_full = state_name  # Use full state name for display
        state_standard = macpac_standards_df[macpac_standards_df['State'] == state_name]
        if len(state_standard) == 0:
            state_standard = macpac_standards_df[macpac_standards_df['State'].str.upper() == state_name.upper()]
        
        if len(state_standard) > 0:
            state_standard = state_standard.iloc[0]
            state_code_upper = str(state or '').upper()
            min_staffing_val = float(state_standard['Min_Staffing'])
            max_staffing_val = float(state_standard['Max_Staffing']) if pd.notna(state_standard.get('Max_Staffing')) else None
            # Georgia baseline override for dashboard display/control defaults.
            if state_code_upper == 'GA':
                min_staffing_val = 2.00
                if max_staffing_val is not None:
                    max_staffing_val = 2.00
            # Create state_standard_info for methodology section (include state rule link and thorough legislation text when in CSV)
            state_standard_info = {
                'state_name': state_name_full,
                'display_text': state_standard.get('Display_Text', ''),
                'min_staffing': min_staffing_val,
                'max_staffing': max_staffing_val,
                'value_type': state_standard['Value_Type'],
                'is_federal_minimum': bool(state_standard.get('Is_Federal_Minimum', False)),
                'state_code_url': state_standard.get('State_Code_URL') if pd.notna(state_standard.get('State_Code_URL')) and str(state_standard.get('State_Code_URL', '')).strip() else None,
                'state_code_citation': state_standard.get('State_Code_Citation') if pd.notna(state_standard.get('State_Code_Citation')) and str(state_standard.get('State_Code_Citation', '')).strip() else None,
                'legislation_text': state_standard.get('Legislation_Text') if pd.notna(state_standard.get('Legislation_Text')) and str(state_standard.get('Legislation_Text', '')).strip() else None,
            }
            # Only show if not federal minimum
            if not state_standard.get('Is_Federal_Minimum', False):
                has_state_standard = True
                min_val = state_standard.get('Min_Staffing', 0)
                max_val = state_standard.get('Max_Staffing', 0)
                if state_standard.get('Value_Type', 'single') == 'range':
                    state_standard_text = f"{state} (range)"
                else:
                    state_standard_text = f"{state}"
        else:
            state_standard_info = None
    else:
        state_standard_info = None
    
    # Previous provider names (provider info history); full list for modal, short list for header strip
    previous_provider_history_list = get_previous_provider_names(facility_provnum, limit=None)
    previous_names_header = previous_provider_history_list[:3]
    previous_names_text = ", ".join(previous_names_header) if previous_names_header else "None"
    
    # Entity line: server-render so no N/A flash (provider_info_df already loaded by before_request)
    entity_history = _parse_entity_history(provider_info_df, facility_provnum) if provider_info_df is not None and facility_provnum and facility_provnum != 'Unknown' else None
    affiliated_entity_id = affiliated_entity_name = affiliated_entity_prev_id = affiliated_entity_prev_name = affiliated_entity_prev_years = None
    affiliated_entity_name_display = affiliated_entity_prev_name_display = None
    if entity_history:
        affiliated_entity_id = entity_history.get('current_entity_id')
        affiliated_entity_name = entity_history.get('current_entity_name')
        affiliated_entity_prev_id = entity_history.get('prev_entity_id')
        affiliated_entity_prev_name = entity_history.get('prev_entity_name')
        affiliated_entity_prev_years = entity_history.get('prev_years')
        affiliated_entity_name_display = format_affiliated_entity_display(affiliated_entity_name)
        affiliated_entity_prev_name_display = format_affiliated_entity_display(affiliated_entity_prev_name)
    # Fallback for top header: if parse logic misses current chain but latest provider row has it,
    # still show the current entity name/link when available.
    if (not affiliated_entity_name) and provider_info_df is not None and not provider_info_df.empty:
        try:
            pi = provider_info_df.copy()
            if "ccn" in pi.columns:
                pi = pi[pi["ccn"].astype(str).str.strip().str.zfill(6) == str(facility_provnum).strip().zfill(6)]
            if not pi.empty:
                if "processing_date" in pi.columns:
                    pi["processing_date"] = pd.to_datetime(pi["processing_date"], errors="coerce")
                    pi = pi.sort_values("processing_date", ascending=False)
                latest_row = pi.iloc[0]
                raw_name = str(latest_row.get("affiliated_entity_name") or "").strip()
                if raw_name and raw_name.upper() not in {"N", "N/A", "NAN", "NONE", "UNKNOWN"}:
                    affiliated_entity_name = raw_name
                    affiliated_entity_name_display = format_affiliated_entity_display(raw_name)
                if not affiliated_entity_id:
                    raw_id = latest_row.get("affiliated_entity_id")
                    if pd.notna(raw_id):
                        rid = str(raw_id).strip()
                        if rid and rid.upper() not in {"N", "N/A", "NAN", "NONE", "UNKNOWN"}:
                            try:
                                affiliated_entity_id = str(int(float(rid)))
                            except Exception:
                                affiliated_entity_id = rid
            if not affiliated_entity_name:
                fb_name, fb_id = _latest_affiliated_entity_fields(
                    provider_info_df, facility_provnum
                )
                if fb_name:
                    affiliated_entity_name = fb_name
                    affiliated_entity_name_display = format_affiliated_entity_display(
                        fb_name
                    )
                if fb_id and not affiliated_entity_id:
                    affiliated_entity_id = fb_id
        except Exception:
            pass
    
    # Only show Download CMS Provider File when files exist in static/data/provider_info/
    provider_info_download_available = False
    if os.path.isdir(PROVIDER_INFO_DATA_DIR):
        try:
            provider_info_download_available = any(
                os.path.isfile(os.path.join(PROVIDER_INFO_DATA_DIR, f)) and not f.startswith('.')
                for f in os.listdir(PROVIDER_INFO_DATA_DIR)
            )
        except OSError:
            pass
    
    deployed_date = getattr(sys.modules[__name__], 'DEPLOYED_DATE', '') or ''
    cms_pbj_facility_url = generate_cms_pbj_facility_link(facility_provnum) if facility_provnum and facility_provnum != 'Unknown' else None
    global ein_job_quarterly_df, ein_category_quarterly_df, ein_employee_detail_df, ein_nursing_summaries_df
    show_ein_position_section = bool(
        ein_job_quarterly_df is not None
        and not ein_job_quarterly_df.empty
        and ein_category_quarterly_df is not None
        and not ein_category_quarterly_df.empty
    )
    ein_employee_detail_available = bool(
        (ein_nursing_summaries_df is not None and not ein_nursing_summaries_df.empty)
        or (ein_employee_detail_df is not None and not ein_employee_detail_df.empty)
    )
    ein_employee_detail_bounds = _ein_employee_detail_workdate_bounds_iso()
    cms_ein_landing_url = CMS_EIN_DETAIL_LANDING_URL if show_ein_position_section else None
    favicon_href = url_for("pbj_favicon_png") if _pbj_favicon_path() else None
    pbj_min_work_date = ""
    pbj_max_work_date = ""
    if global_df is not None and len(global_df) > 0 and "WorkDate" in global_df.columns:
        _wd = pd.to_datetime(global_df["WorkDate"], errors="coerce")
        if _wd.notna().any():
            pbj_max_work_date = _wd.max().strftime("%Y-%m-%d")
            pbj_min_work_date = _wd.min().strftime("%Y-%m-%d")
    facility_name_display = format_facility_display_name(facility_name)
    pdf_export_state_ref: dict[str, Any] | None = None
    if state_standard_info:
        pdf_export_state_ref = {
            "min_staffing": float(state_standard_info["min_staffing"]),
            "max_staffing": (
                float(state_standard_info["max_staffing"])
                if state_standard_info.get("max_staffing") is not None
                else None
            ),
            "value_type": str(state_standard_info.get("value_type") or "single"),
            "state_name": str(state_standard_info.get("state_name") or ""),
        }
    pdf_lite_hprd_benchmarks = lite_hprd_benchmarks_for_state(
        state if state and str(state).strip().upper() != "UNKNOWN" else ""
    )
    _state_for_link = str(state).strip() if state else ""
    pbj320_state_dashboard_url: str | None = None
    if _state_for_link and validate_state_code(_state_for_link):
        pbj320_state_dashboard_url = generate_state_dashboard_url(_state_for_link)
    cms_region_peer_context: dict[str, Any] | None = None
    _rn_int: Optional[int] = None
    _rn = pdf_lite_hprd_benchmarks.get("cms_region_number")
    if _rn is not None and not (isinstance(_rn, float) and pd.isna(_rn)):
        try:
            _rn_int = int(float(_rn))
        except (TypeError, ValueError):
            _rn_int = None
        if _rn_int is not None:
            cms_region_peer_context = {
                "number": _rn_int,
                "name": (pdf_lite_hprd_benchmarks.get("cms_region_name") or "") or "",
                "full": (pdf_lite_hprd_benchmarks.get("cms_region_full") or "") or "",
                "rankings_report_url": PBJ_RANKINGS_REPORT_URL,
            }
    facility_state_long_name, _geo_peer_codes = _facility_geo_from_cms_region_mapping(
        _state_for_link, _rn_int
    )
    if cms_region_peer_context is not None:
        cms_region_peer_context = {**cms_region_peer_context, "peer_state_codes": _geo_peer_codes}
    geo_rollup_series = lite_hprd_rollup_series_by_quarter(
        _state_for_link, county_name or "", facility_provnum or ""
    )
    page_title = f"{facility_name_display} | PBJ320"
    og_title = f"PBJ320 — {facility_name_display}"
    og_description = (
        f"PBJ320: CMS payroll-based journal (PBJ) staffing dashboard for {facility_name_display} "
        f"(CCN {facility_provnum}). HPRD, case-mix, provider metrics, and data.cms.gov links."
    )
    og_page_url = request.base_url.rstrip("/")
    og_image_url = "https://www.pbj320.com/pbj.seo.png"
    pbj_entity_base_url = "https://www.pbj320.com/entity"
    nh_health_citations_cms_explorer_url = (
        cms_nh_health_citations_dataset_explorer_url(str(facility_provnum))
        or "https://data.cms.gov/provider-data/dataset/r5ix-sfxw"
    )
    ein_cms_job_codes_reference = [
        {
            "code": code,
            "title": title_cat[0],
            "category": title_cat[1],
            "short": JOB_TITLE_SHORT.get(code, ""),
        }
        for code, title_cat in sorted(JOB_CODE_INFO.items(), key=lambda kv: kv[0])
    ]
    _prem_meth = _pbj_premium_facility_href(facility_provnum, "/methodology")
    _prem_dm = _pbj_premium_facility_href(facility_provnum, "/data-matching")
    ein_headcount_csv_comment_lines = [
        "# Employee Detail (EIN) headcount export — same long-format series as the facility dashboard chart.",
        "# Columns: provnum; slice (nurse|nonnurse|all); grain (day|month|quarter|year); period_key; period_label; "
        "multi_role_employees_period; contract_distinct_dedup_period; job_code; job_title_short; job_category; "
        "employee_distinct; contract_distinct_role.",
        "# employee_distinct: distinct SYS_EMPLEE_ID with positive WORK_HRS_NUM in that period bucket for that "
        "EMPLEE_JOB_CD_ID (one row per job code segment; summing segments over-counts people in multiple roles).",
        "# employee_distinct_period is NOT a column here; the API/chart uses it for the deduped slice headcount "
        "per period (tooltip on the dashboard).",
        "# contract_distinct_role: distinct SYS_EMPLEE_ID with EMP_CTR=2 in that bucket for that job code; "
        "contract_distinct_dedup_period: distinct contract staff across all job codes in the period.",
        "# CMS PAL Employee Detail (EIN-level row source).",
    ]
    if cms_ein_landing_url:
        ein_headcount_csv_comment_lines.append(
            f"# CMS Employee Detail catalog / landing: {cms_ein_landing_url}"
        )
    ein_headcount_csv_comment_lines.append(
        "# CMS PBJ daily nurse staffing (facility-level context only; not row-level in this file): "
        "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing"
    )
    if _prem_meth:
        ein_headcount_csv_comment_lines.append(
            f"# This facility on PBJ320 (methodology, caveats, other panels): {_prem_meth}"
        )
    if _prem_dm:
        ein_headcount_csv_comment_lines.append(
            f"# Same facility — quarter alignment reference (PBJ vs Provider Information): {_prem_dm}"
        )
    _premium_demo_ctx = pbj_premium_dashboard_offer_context(
        app_root=_app_root,
        resolve_static_href=_pbj_template_site_href,
        for_superdynamic=True,
    )
    _dash_tpl = _superdynamic_dashboard_template_name()
    return render_template(
        _dash_tpl,
        superdynamic_v2=_dash_tpl == "superdynamic_dashboard_v2.html",
        superdynamic_v3_panes=_superdynamic_v3_panes_enabled(),
        facility_name=facility_name,
        facility_name_display=facility_name_display,
        page_title=page_title,
        provnum=facility_provnum,
        pbj_entity_base_url=pbj_entity_base_url,
        city=city,
        state=state,
        county_name=county_name,
        has_state_standard=has_state_standard,
        state_standard_text=state_standard_text,
        state_standard_info=state_standard_info,
        previous_names=previous_names_text,
        previous_provider_history_list=previous_provider_history_list,
        deployed_date=deployed_date,
        cms_pbj_facility_url=cms_pbj_facility_url,
        affiliated_entity_id=affiliated_entity_id,
        affiliated_entity_name=affiliated_entity_name,
        affiliated_entity_name_display=affiliated_entity_name_display,
        affiliated_entity_prev_id=affiliated_entity_prev_id,
        affiliated_entity_prev_name=affiliated_entity_prev_name,
        affiliated_entity_prev_name_display=affiliated_entity_prev_name_display,
        affiliated_entity_prev_years=affiliated_entity_prev_years,
        provider_info_download_available=provider_info_download_available,
        show_ein_position_section=show_ein_position_section,
        ein_employee_detail_available=ein_employee_detail_available,
        ein_employee_detail_bounds=ein_employee_detail_bounds,
        cms_ein_landing_url=cms_ein_landing_url,
        favicon_href=favicon_href,
        og_title=og_title,
        og_description=og_description,
        og_page_url=og_page_url,
        og_image_url=og_image_url,
        og_site_name="PBJ320",
        pbj_min_work_date=pbj_min_work_date,
        pbj_max_work_date=pbj_max_work_date,
        pdf_export_state_ref=pdf_export_state_ref,
        pdf_lite_hprd_benchmarks=pdf_lite_hprd_benchmarks,
        pbj320_state_dashboard_url=pbj320_state_dashboard_url,
        cms_region_peer_context=cms_region_peer_context,
        facility_state_long_name=facility_state_long_name,
        geo_rollup_series=geo_rollup_series,
        pbj_include_nonnurse=INCLUDE_NONNURSE,
        nh_health_citations_cms_explorer_url=nh_health_citations_cms_explorer_url,
        ein_cms_job_codes_reference=ein_cms_job_codes_reference,
        ein_headcount_csv_comment_lines=ein_headcount_csv_comment_lines,
        pbj_premium_facility_base_url=_pbj_premium_facility_base_url(str(facility_provnum)),
        **_pbj_template_client_config(),
        **_pbj_claude_skill_template_context(),
        **_premium_demo_ctx,
    )


@app.route("/superdynamic")
def superdynamic_shortcut():
    """Same UI as ``/``; handy if you expect the path name in the URL."""
    return redirect(url_for("index"))


@app.route('/methodology')
def methodology_page():
    """Dedicated methodology page for formulas, caveats, and sources."""
    global global_df, provider_info_df

    facility_name = "Unknown Facility"
    facility_provnum = "Unknown"
    city = "Unknown"
    state = "Unknown"

    if global_df is not None and not global_df.empty:
        df = global_df.sort_values('WorkDate', ascending=False) if 'WorkDate' in global_df.columns else global_df
        if not df.empty:
            if 'PROVNUM' in df.columns:
                facility_provnum = str(df['PROVNUM'].iloc[0]).zfill(6)
            if 'CITY' in df.columns:
                city = str(df['CITY'].iloc[0] or '').strip() or "Unknown"
            if 'STATE' in df.columns:
                state = str(df['STATE'].iloc[0] or '').strip() or "Unknown"
            if 'PROVNAME' in df.columns:
                facility_name = str(df['PROVNAME'].iloc[0] or '').strip() or facility_name

    if provider_info_df is not None and not provider_info_df.empty and 'provider_name' in provider_info_df.columns:
        p_df = provider_info_df.sort_values('processing_date', ascending=False) if 'processing_date' in provider_info_df.columns else provider_info_df
        if not p_df.empty:
            latest_name = p_df['provider_name'].iloc[0]
            if pd.notna(latest_name) and str(latest_name).strip():
                facility_name = str(latest_name).strip()

    facility_name_display = format_facility_display_name(facility_name)
    favicon_href = url_for("pbj_favicon_png") if _pbj_favicon_path() else None

    entity_history = (
        _parse_entity_history(provider_info_df, facility_provnum)
        if provider_info_df is not None and facility_provnum and str(facility_provnum).strip() not in ("", "Unknown")
        else None
    )
    affiliated_entity_id = None
    affiliated_entity_name_display = None
    if entity_history:
        affiliated_entity_id = entity_history.get("current_entity_id")
        affiliated_entity_name_display = format_affiliated_entity_display(
            entity_history.get("current_entity_name")
        )
    pbj_entity_base_url = "https://www.pbj320.com/entity"

    return render_template(
        "superdynamic_methodology.html",
        facility_name_display=facility_name_display,
        provnum=facility_provnum,
        city=city,
        state=state,
        favicon_href=favicon_href,
        page_title=f"Methodology | {facility_name_display} | PBJ320",
        affiliated_entity_id=affiliated_entity_id,
        affiliated_entity_name_display=affiliated_entity_name_display,
        pbj_entity_base_url=pbj_entity_base_url,
        pbj_premium_facility_base_url=_pbj_premium_facility_base_url(str(facility_provnum)),
        **_pbj_template_client_config(),
    )


def _render_report_builder_v3_page():
    """Standalone PBJ Case Builder (v3 UI)."""
    global global_df, provider_info_df

    facility_name = "Unknown Facility"
    facility_provnum = "Unknown"
    city = "Unknown"
    state = "Unknown"
    pbj_min_work_date = ""
    pbj_max_work_date = ""

    if global_df is not None and not global_df.empty:
        df = global_df.sort_values("WorkDate", ascending=False) if "WorkDate" in global_df.columns else global_df
        if not df.empty:
            if "PROVNUM" in df.columns:
                facility_provnum = str(df["PROVNUM"].iloc[0]).zfill(6)
            if "CITY" in df.columns:
                city = str(df["CITY"].iloc[0] or "").strip() or "Unknown"
            if "STATE" in df.columns:
                state = str(df["STATE"].iloc[0] or "").strip() or "Unknown"
            if "PROVNAME" in df.columns:
                facility_name = str(df["PROVNAME"].iloc[0] or "").strip() or facility_name
        if "WorkDate" in global_df.columns:
            _wd = pd.to_datetime(global_df["WorkDate"], errors="coerce").dropna()
            if not _wd.empty:
                pbj_min_work_date = _wd.min().strftime("%Y-%m-%d")
                pbj_max_work_date = _wd.max().strftime("%Y-%m-%d")

    if provider_info_df is not None and not provider_info_df.empty and "provider_name" in provider_info_df.columns:
        p_df = (
            provider_info_df.sort_values("processing_date", ascending=False)
            if "processing_date" in provider_info_df.columns
            else provider_info_df
        )
        if not p_df.empty:
            latest_name = p_df["provider_name"].iloc[0]
            if pd.notna(latest_name) and str(latest_name).strip():
                facility_name = str(latest_name).strip()

    if not _pbj_report_builder_v3_route_allowed(
        facility_provnum,
        beta_query=request.args.get("beta") == "1",
    ):
        abort(404)

    facility_name_display = format_facility_display_name(facility_name)
    facility_name_compact = get_facility_name_for_context(facility_name, "compact") or facility_name_display
    favicon_href = url_for("pbj_favicon_png") if _pbj_favicon_path() else None
    dashboard_href = _pbj_premium_facility_href(facility_provnum, "/") or url_for("index")
    pbj_available_quarters = _pbj_quarter_keys_from_bounds(pbj_min_work_date, pbj_max_work_date)

    return render_template(
        "report_builder_v3_standalone.html",
        facility_name=facility_name,
        facility_name_display=facility_name_display,
        facility_name_compact=facility_name_compact,
        provnum=facility_provnum,
        city=city,
        state=state,
        favicon_href=favicon_href,
        page_title=f"PBJ Case Builder | {facility_name_display} | PBJ320",
        pbj_min_work_date=pbj_min_work_date,
        pbj_max_work_date=pbj_max_work_date,
        pbj_available_quarters=pbj_available_quarters,
        pbj_dashboard_href=dashboard_href,
        superdynamic_v3_panes=True,
        pbj_report_builder_standalone=True,
        pbj_premium_facility_base_url=_pbj_premium_facility_href(facility_provnum, ""),
        **_pbj_template_client_config(),
        **_pbj_claude_skill_template_context(),
    )


@app.route("/case-builder")
def case_builder_page():
    """Canonical user-facing Case Builder URL."""
    return _render_report_builder_v3_page()


@app.route("/report-builder-v3")
def report_builder_v3_legacy_redirect():
    """Legacy alias -> /case-builder."""
    qs = request.query_string.decode("utf-8", errors="replace")
    dest = "/case-builder" + ("?" + qs if qs else "")
    return redirect(dest, code=302)


@app.route('/data-matching')
def data_matching_page():
    """Dedicated page explaining how provider-info quarters are matched to PBJ quarters."""
    global global_df

    facility_provnum = "Unknown"
    state = "Unknown"
    if global_df is not None and len(global_df) > 0:
        # Try to get most recent facility fields
        if 'WorkDate' in global_df.columns:
            global_df_sorted = global_df.sort_values('WorkDate', ascending=False)
        else:
            global_df_sorted = global_df

        if len(global_df_sorted) > 0:
            if 'PROVNUM' in global_df_sorted.columns:
                facility_provnum = str(global_df_sorted['PROVNUM'].iloc[0]).zfill(6)
            if 'STATE' in global_df_sorted.columns:
                state = global_df_sorted['STATE'].iloc[0]

    cms_pbj_facility_url = generate_cms_pbj_facility_link(facility_provnum) if facility_provnum != 'Unknown' else None

    provider_info_dataset_url = _CMS_PROVIDER_INFO_DATASET_PAGE
    dashboard_href = _pbj_premium_facility_href(facility_provnum, "/")
    methodology_href = _pbj_premium_facility_href(facility_provnum, "/methodology")
    data_matching_href = _pbj_premium_facility_href(facility_provnum, "/data-matching")
    brand_href = dashboard_href

    if cms_pbj_facility_url:
        pbj_link_html = (
            '<a href="'
            + html.escape(cms_pbj_facility_url, quote=True)
            + '" target="_blank" rel="noopener">CMS PBJ Data - '
            + html.escape(str(facility_provnum))
            + '</a>'
        )
    else:
        pbj_link_html = (
            '<span class="text-muted">CMS PBJ facility link unavailable (CCN '
            + html.escape(str(facility_provnum))
            + ').</span>'
        )

    mapping_rows = _load_interval_quarter_mapping_fallback(0)
    table_body_html = _interval_quarter_mapping_table_rows_html(mapping_rows, 0)
    dm_row_count = len(mapping_rows)

    fav_head = ""
    brand_mark = (
        '<span class="guided-nav-brand-mark">'
        '<span class="guided-nav-brand-pbj">PBJ</span>'
        '<span class="guided-nav-brand-320">320</span>'
        '</span>'
    )
    brand_inner = brand_mark
    if _pbj_favicon_path():
        _dm_fav = url_for("pbj_favicon_png")
        _dm_fav_e = html.escape(_dm_fav, quote=True)
        fav_head = (
            f'  <link rel="icon" type="image/png" href="{_dm_fav_e}">\n'
            f'  <link rel="apple-touch-icon" href="{_dm_fav_e}">\n'
        )
        brand_inner = (
            f'<img src="{_dm_fav_e}" width="22" height="22" alt="" '
            f'class="me-2 align-text-bottom" decoding="async">'
            f'{brand_mark}'
        )

    dm_gtag_head = _pbj_gtag_head_script_snippet(_pbj_resolved_ga_measurement_id())

    html_out = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Provider info ↔ PBJ quarters</title>
{dm_gtag_head}{fav_head}  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
  <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" rel="stylesheet" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer">
  <style>
    .pbj-dm-page {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }}
    .pbj-dm-page .table-compact th,
    .pbj-dm-page .table-compact td {{ padding: 0.2rem 0.35rem; font-size: 0.72rem; vertical-align: top; line-height: 1.25; }}
    .pbj-dm-page .table-compact thead th {{ white-space: nowrap; font-size: 0.68rem; }}
    .pbj-dm-page h1 {{ font-size: 1.15rem; margin-bottom: 0.35rem; }}
    .pbj-dm-page .dm-lead {{ font-size: 0.8rem; margin-bottom: 0.65rem; }}
    .pbj-dm-page h2.h5 {{ font-size: 0.95rem; margin-top: 0.75rem; margin-bottom: 0.25rem; }}
    .guided-nav-wrap {{
      position: sticky;
      top: 0;
      z-index: 1040;
      margin: 0;
      padding: 0.45rem clamp(0.65rem, 2vw, 1.25rem);
      background: rgba(255, 255, 255, 0.97);
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      border-bottom: 1px solid #dbe7ff;
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
    }}
    .guided-nav {{ max-width: 1320px; margin: 0 auto; }}
    .guided-nav-inner {{ display: flex; align-items: center; gap: 0.75rem; flex-wrap: wrap; }}
    .guided-nav-brand {{
      display: inline-flex; align-items: center; gap: 0.45rem;
      text-decoration: none; color: #0f172a; font-weight: 700; white-space: nowrap;
    }}
    .guided-nav-brand-mark {{ letter-spacing: -0.02em; }}
    .guided-nav-brand-pbj {{ color: #0f172a; }}
    .guided-nav-brand-320 {{ color: #4f46e5; }}
    .guided-nav .nav-link {{
      border-radius: 999px; font-size: 0.92rem; font-weight: 550; color: #334155;
      padding: 0.48rem 0.85rem; border: 1px solid transparent;
    }}
    .guided-nav .nav-link:hover {{ background: #eef4ff; border-color: #cfe0ff; color: #1d4ed8; }}
    .guided-nav .nav-link.active {{
      background: linear-gradient(135deg, #312e81 0%, #1e3a5f 100%);
      color: #fff !important; font-weight: 600;
      border-color: rgba(15, 23, 42, 0.35);
    }}
  </style>
</head>
<body class="bg-light">
  <div class="guided-nav-wrap">
    <nav class="guided-nav w-100" aria-label="Facility navigation">
      <div class="guided-nav-inner py-1">
        <a class="guided-nav-brand" href="{html.escape(brand_href, quote=True)}" title="PBJ320 — Premium home (this facility)" aria-label="PBJ320 — Premium home (this facility)">
          {brand_inner}
        </a>
        <ul class="nav nav-pills gap-1 flex-nowrap mb-0 py-1 ms-md-2">
          <li class="nav-item"><a class="nav-link" href="{html.escape(dashboard_href, quote=True)}">Dashboard</a></li>
          <li class="nav-item"><a class="nav-link" href="{html.escape(methodology_href, quote=True)}">Methodology</a></li>
          <li class="nav-item"><a class="nav-link active" href="{html.escape(data_matching_href, quote=True)}" aria-current="page">Data matching</a></li>
        </ul>
      </div>
    </nav>
  </div>
  <div class="container py-2 pbj-dm-page">
    <h1 class="h3 mb-1">Provider information ↔ PBJ quarters</h1>
    <p class="text-muted dm-lead mb-2">
      CMS publishes Nursing Home Provider Information files monthly, while PBJ staffing data is organized by quarter.
      PBJ320 uses CMS data collection interval files when available to match each Provider Information snapshot to the relevant PBJ staffing quarter.
    </p>
    <p class="text-muted dm-lead mb-2">
      For older months or files without a usable interval record, PBJ320 matches each Provider Information file to the appropriate PBJ quarter by comparing the staffing totals in Provider Info — including total nurse staffing and RN staffing — against the corresponding PBJ staffing data. When those staffing values align, PBJ320 uses that match to connect the related case-mix context to the correct PBJ reporting period.
    </p>
    <p class="text-muted dm-lead mb-2">
      <strong>Note:</strong> Multiple monthly Provider Information files may map to the same PBJ quarter when CMS has not yet released a newer PBJ staffing quarter.
    </p>

    <h2 class="h5">Provider Info → PBJ quarter mapping</h2>
    <p class="small text-muted mb-1">{dm_row_count} rows · bundled <code>interval_quarter_mapping.json</code> merged with <code>prov_info.py</code> for gaps, plus PBJ-calendar-only backfill where noted. Interval CSV names appear only when that processing month shipped an intervals file.</p>

    <div class="table-responsive shadow-sm bg-white rounded border">
      <table class="table table-sm table-striped table-bordered align-middle mb-0 table-compact">
        <thead class="table-light">
          <tr>
            <th class="text-nowrap">PBJ quarter</th>
            <th class="text-nowrap">Provider Info CSV</th>
            <th class="text-nowrap">PBJ period</th>
            <th class="text-nowrap">PBJ matched</th>
            <th class="text-nowrap">Processing month</th>
            <th class="text-nowrap">Interval CSV</th>
          </tr>
        </thead>
        <tbody>
{table_body_html}
        </tbody>
      </table>
    </div>

    <h2 class="h5 mt-3">Links</h2>
    <ul class="small mb-0" style="font-size: 0.78rem;">
      <li>{pbj_link_html}</li>
      <li><a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" rel="noopener">PBJ daily nurse staffing (data.cms.gov)</a></li>
      <li><a href="{html.escape(provider_info_dataset_url, quote=True)}" target="_blank" rel="noopener">Nursing Home Provider Information (data.cms.gov)</a></li>
    </ul>
  </div>
</body>
</html>"""

    return html_out

def _parse_dashboard_iso_date_arg(raw: Optional[str], *, field: str) -> pd.Timestamp:
    """Parse YYYY-MM-DD from query args; reject partial/garbage strings that break pandas (e.g. 0002-06-14)."""
    if raw is None or not str(raw).strip():
        raise ValueError(f"Missing {field} date.")
    s = str(raw).strip()[:10]
    if len(s) != 10 or s[4] != "-" or s[7] != "-":
        raise ValueError(f"Invalid {field} date {raw!r}: use YYYY-MM-DD.")
    try:
        y = int(s[0:4])
        mo = int(s[5:7])
        d = int(s[8:10])
    except ValueError as e:
        raise ValueError(f"Invalid {field} date {raw!r}.") from e
    if y < 1990 or y > 2100:
        raise ValueError(f"Invalid {field} date {raw!r}: year must be between 1990 and 2100.")
    try:
        dt = pd.Timestamp(datetime(y, mo, d))
    except Exception as e:
        raise ValueError(f"Invalid {field} date {raw!r}.") from e
    if pd.isna(dt):
        raise ValueError(f"Invalid {field} date {raw!r}.")
    return cast(pd.Timestamp, dt)


def _filter_facility_daily_for_dashboard(
    df: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    day_of_week: str,
    quarter: str,
    year: str,
    holidays_only: bool,
) -> pd.DataFrame:
    """Same calendar filters as ``/api/data`` and ``/api/summary`` (inclusive end date).

    Coerces ``WorkDate`` to datetime, drops pre-2017 rows, and uses an exclusive
    upper bound on ``WorkDate`` so the last calendar day is included for date-only values.
    """
    filtered_df = df.copy()
    filtered_df["WorkDate"] = pd.to_datetime(filtered_df["WorkDate"], errors="coerce")
    filtered_df = filtered_df[filtered_df["WorkDate"].notna()]
    filtered_df = filtered_df[filtered_df["WorkDate"] >= pd.Timestamp("2017-01-01")]
    if start_date and str(start_date).strip():
        start_dt = _parse_dashboard_iso_date_arg(start_date, field="start")
        filtered_df = filtered_df[filtered_df["WorkDate"] >= start_dt]
    if end_date and str(end_date).strip():
        end_day = _parse_dashboard_iso_date_arg(end_date, field="end")
        filtered_df = filtered_df[filtered_df["WorkDate"] < end_day + pd.Timedelta(days=1)]
    if day_of_week != "all":
        filtered_df = filtered_df[filtered_df["DayOfWeek"] == day_of_week]
    if quarter != "all" and str(quarter).strip():
        quarters = [q.strip() for q in str(quarter).split(",")]
        filtered_df = filtered_df[filtered_df["CY_Qtr"].isin(quarters)]
    if year != "all" and str(year).strip():
        years = [int(y.strip()) for y in str(year).split(",") if str(y).strip().isdigit()]
        if years:
            filtered_df = filtered_df[filtered_df["WorkDate"].dt.year.isin(years)]
    if holidays_only:
        filtered_df = filtered_df[filtered_df["IsHoliday"] == True]
    if not isinstance(filtered_df, pd.DataFrame):
        raise TypeError("_filter_facility_daily_for_dashboard: internal filter must return a DataFrame")
    return filtered_df


def _filtered_df_cache_key(
    *,
    start_date: str | None,
    end_date: str | None,
    day_of_week: str,
    quarter: str,
    year: str,
    holidays_only: bool,
) -> tuple[Any, ...]:
    return (
        str(start_date or "").strip(),
        str(end_date or "").strip(),
        str(day_of_week or "all").strip(),
        str(quarter or "all").strip(),
        str(year or "all").strip(),
        bool(holidays_only),
    )


def _filter_facility_daily_cached(
    df: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
    day_of_week: str,
    quarter: str,
    year: str,
    holidays_only: bool,
) -> pd.DataFrame:
    key = _filtered_df_cache_key(
        start_date=start_date,
        end_date=end_date,
        day_of_week=day_of_week,
        quarter=quarter,
        year=year,
        holidays_only=holidays_only,
    )
    now = time.monotonic()
    with _FILTERED_DF_CACHE_LOCK:
        cached = _FILTERED_DF_CACHE.get(key)
        if cached and (now - cached[0]) <= _FILTERED_DF_CACHE_TTL_SEC:
            return cast(pd.DataFrame, cached[1].copy())

    filtered = _filter_facility_daily_for_dashboard(
        df,
        start_date=start_date,
        end_date=end_date,
        day_of_week=day_of_week,
        quarter=quarter,
        year=year,
        holidays_only=holidays_only,
    )
    with _FILTERED_DF_CACHE_LOCK:
        _FILTERED_DF_CACHE[key] = (now, cast(pd.DataFrame, filtered.copy()))
        if len(_FILTERED_DF_CACHE) > 48:
            oldest_key = min(_FILTERED_DF_CACHE.items(), key=lambda kv: kv[1][0])[0]
            _FILTERED_DF_CACHE.pop(oldest_key, None)
    return filtered


def _clear_filtered_df_cache() -> None:
    with _FILTERED_DF_CACHE_LOCK:
        _FILTERED_DF_CACHE.clear()




@app.route("/api/analysis_bundle")
def api_analysis_bundle():
    """Single response for main dashboard: /api/data + /api/summary + /api/charts (one or two filters)."""
    try:
        data_start = request.args.get("start_date")
        data_end = request.args.get("end_date")
        day_of_week = request.args.get("day_of_week", "all")
        quarter = request.args.get("quarter", "all")
        year = request.args.get("year", "all")
        show_holidays_only = request.args.get("holidays_only", "false") == "true"
        cs_raw = (request.args.get("charts_start_date") or "").strip()
        ce_raw = (request.args.get("charts_end_date") or "").strip()
        charts_start = cs_raw or data_start
        charts_end = ce_raw or data_end
        hprd_view = request.args.get("hprd_view", "daily")
        hours_view = request.args.get("hours_view", "daily")
        census_view = request.args.get("census_view", "daily")
        contract_view = request.args.get("contract_view", "daily")
        composition_view = request.args.get("composition_view", "month")

        global global_df
        if global_df is None or len(global_df) == 0:
            return jsonify(
                {
                    "error": "No data loaded",
                    "data": {"data": [], "total_records": 0, "date_range": {}},
                    "summary": {"error": "No data loaded"},
                    "charts": {"charts": {}, "error": "No data loaded"},
                }
            )
        gdf = cast(pd.DataFrame, global_df)
        _ensure_pbj_dashboard_derived_columns_inplace(gdf)
        try:
            filtered_data = _filter_facility_daily_cached(
                gdf,
                start_date=data_start,
                end_date=data_end,
                day_of_week=day_of_week,
                quarter=quarter,
                year=year,
                holidays_only=show_holidays_only,
            )
        except ValueError as e:
            return (
                jsonify(
                    {
                        "error": str(e),
                        "data": {"error": str(e), "data": [], "date_range": {}},
                        "summary": {"error": str(e)},
                        "charts": {"error": str(e), "charts": {}, "filter_info": str(e)},
                    }
                ),
                400,
            )

        df_for_charts = cast(pd.DataFrame, global_df)
        if "Total_Staff_Hours" not in df_for_charts.columns:
            df_for_charts = df_for_charts.copy()
            trn = df_for_charts["Total_RN_Hours"] if "Total_RN_Hours" in df_for_charts.columns else 0.0
            tln = df_for_charts["Total_LPN_Hours"] if "Total_LPN_Hours" in df_for_charts.columns else 0.0
            tna = df_for_charts["Total_Nurse_Aide_Hours"] if "Total_Nurse_Aide_Hours" in df_for_charts.columns else 0.0
            df_for_charts["Total_Staff_Hours"] = trn + tln + tna
        if "Total_Staff_HPRD" not in df_for_charts.columns and "Total_Staff_Hours" in df_for_charts.columns and "MDScensus" in df_for_charts.columns:
            if df_for_charts is global_df:
                df_for_charts = df_for_charts.copy()
            df_for_charts["Total_Staff_HPRD"] = _round_fin_series(
                _divide_hprd(df_for_charts["Total_Staff_Hours"], df_for_charts["MDScensus"]), 2
            )

        same_charts_slice = (charts_start == data_start) and (charts_end == data_end)
        if same_charts_slice:
            filtered_charts = filtered_data
        else:
            try:
                filtered_charts = _filter_facility_daily_cached(
                    df_for_charts,
                    start_date=charts_start,
                    end_date=charts_end,
                    day_of_week=day_of_week,
                    quarter=quarter,
                    year=year,
                    holidays_only=show_holidays_only,
                )
            except ValueError as e:
                return (
                    jsonify(
                        {
                            "error": str(e),
                            "data": _daily_json_records_from_filtered_df(filtered_data),
                            "summary": _summary_dict_from_filtered_daily_df(filtered_data.copy()),
                            "charts": {"error": str(e), "charts": {}, "filter_info": str(e)},
                        }
                    ),
                    400,
                )

        data_payload = _daily_json_records_from_filtered_df(filtered_data)
        summary_payload = _summary_dict_from_filtered_daily_df(filtered_data.copy())
        charts_payload = _charts_build_payload_dict(
            filtered_charts,
            filter_label_start_date=charts_start,
            filter_label_end_date=charts_end,
            filter_quarter=quarter,
            filter_day_of_week=day_of_week,
            filter_holidays_only=show_holidays_only,
            dow_calendar_year_param=request.args.get("dow_calendar_year"),
            hprd_view=hprd_view,
            hours_view=hours_view,
            census_view=census_view,
            contract_view=contract_view,
            composition_view=composition_view,
        )
        return jsonify({"data": data_payload, "summary": summary_payload, "charts": charts_payload})
    except Exception as e:
        return jsonify({"error": str(e), "data": None, "summary": None, "charts": None})

@app.route('/api/data')
def get_data():
    """Get filtered data"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        day_of_week = request.args.get('day_of_week', 'all')
        quarter = request.args.get('quarter', 'all')
        year = request.args.get('year', 'all')
        show_holidays_only = request.args.get('holidays_only', 'false') == 'true'
        
        # Filter data
        global global_df
        if global_df is None or len(global_df) == 0:
            return jsonify({'error': 'No data loaded', 'data': []})
        gdf_data = cast(pd.DataFrame, global_df)
        try:
            filtered_df = _filter_facility_daily_cached(
                gdf_data,
                start_date=start_date,
                end_date=end_date,
                day_of_week=day_of_week,
                quarter=quarter,
                year=year,
                holidays_only=show_holidays_only,
            )
        except ValueError as e:
            return jsonify({'error': str(e), 'data': [], 'date_range': {}}), 400
        
        payload = _daily_json_records_from_filtered_df(filtered_df)
        return jsonify(payload)
        
    except Exception as e:
        return jsonify({'error': str(e)})


def _provider_match_quarter_param_from_pbj_df(filtered_df: pd.DataFrame) -> Optional[str]:
    """Latest PBJ calendar quarter in ``filtered_df``, as yyyyQn for ``/api/provider_info_summary?quarter=``."""
    if filtered_df is None or len(filtered_df) == 0:
        return None
    if "CY_Qtr" in filtered_df.columns:
        best_t: tuple[int, int] | None = None
        best_raw: str | None = None
        for q in filtered_df["CY_Qtr"].dropna().unique():
            t = _quarter_sort_key_pre_post(q)
            if t == (0, 0):
                continue
            if best_t is None or t > best_t:
                best_t = t
                best_raw = str(q).strip()
        if best_raw:
            m = re.search(r"(?:CY)?(\d{4})Q([1-4])", best_raw.upper().replace(" ", ""))
            if m:
                return f"{m.group(1)}Q{m.group(2)}"
    if "WorkDate" not in filtered_df.columns:
        return None
    wd = pd.to_datetime(filtered_df["WorkDate"], errors="coerce")
    mx = wd.max()
    if pd.isna(mx):
        return None
    ts = cast(pd.Timestamp, pd.Timestamp(mx))
    qn = (int(ts.month) - 1) // 3 + 1
    return f"{int(ts.year)}Q{qn}"


@app.route('/api/summary')
def get_summary():
    """Get summary statistics"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        day_of_week = request.args.get('day_of_week', 'all')
        quarter = request.args.get('quarter', 'all')
        year = request.args.get('year', 'all')
        show_holidays_only = request.args.get('holidays_only', 'false') == 'true'
        
        # Filter data (match /api/data: inclusive end date via exclusive upper bound)
        global global_df
        if global_df is None or len(global_df) == 0:
            return jsonify({"error": "No data loaded"})
        gdf_summary = cast(pd.DataFrame, global_df)
        _ensure_pbj_dashboard_derived_columns_inplace(gdf_summary)
        try:
            filtered_df = _filter_facility_daily_cached(
                gdf_summary,
                start_date=start_date,
                end_date=end_date,
                day_of_week=day_of_week,
                quarter=quarter,
                year=year,
                holidays_only=show_holidays_only,
            )
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        summary = _summary_dict_from_filtered_daily_df(filtered_df)
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/provider_info')
def get_provider_info():
    """Get provider info data for the facility"""
    try:
        global provider_info_df
        
        if provider_info_df is None:
            return jsonify({'error': 'Provider info data not loaded'})
        
        # Convert to JSON-serializable format
        data = provider_info_df.copy()
        
        # Convert datetime to string
        data['processing_date'] = data['processing_date'].dt.strftime('%Y-%m-%d')
        
        # Round numeric values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].apply(lambda x: round_financial(x, 3) if pd.notna(x) else 0)
        
        return jsonify({
            'data': data.to_dict('records'),
            'facility_name': data['provider_name'].iloc[0] if len(data) > 0 else 'Unknown',
            'latest_rating': {
                'overall': _rating_or_none(data['overall_rating'].iloc[-1]) if len(data) > 0 else None,
                'staffing': _rating_or_none(data['staffing_rating'].iloc[-1]) if len(data) > 0 else None,
                'health_inspection': _rating_or_none(data['health_inspection_rating'].iloc[-1]) if len(data) > 0 else None,
                'quality': _rating_or_none(data['qm_rating'].iloc[-1]) if len(data) > 0 and 'qm_rating' in data.columns else None
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def _rating_or_none(val):
    """Return int rating 1-5, or None if missing or 0 (no data)."""
    if pd.isna(val):
        return None
    try:
        r = int(float(val))
        return r if 1 <= r <= 5 else None
    except (ValueError, TypeError):
        return None


def _rating_series_for_chart(series):
    """Convert rating series to list; 0 or missing -> None (no data)."""
    def one_val(x):
        if pd.isna(x):
            return None
        try:
            v = float(x)
            return None if v == 0 else int(v)
        except (ValueError, TypeError):
            return None
    return [one_val(x) for x in series]


def _parse_entity_history(
    provider_info_df,
    provnum,
    as_of_canonical: Optional[str] = None,
):
    """
    Parse facility entity history from provider info: current entity and most recent prior entity.
    Uses quarter/processing_date order; does not imply data completeness before 2017.
    When ``as_of_canonical`` is set (CYyyyyQn), only rows on or before that quarter are used so
    current/prev entity match the matched Provider Information quarter.

    Returns dict: current_entity_id, current_entity_name, prev_entity_id, prev_entity_name, prev_years (e.g. '2018–2022' or None).
    """
    if provider_info_df is None or provider_info_df.empty:
        return None
    id_col = 'affiliated_entity_id'
    name_col = 'affiliated_entity_name'
    if id_col not in provider_info_df.columns or name_col not in provider_info_df.columns:
        return None
    provnum = str(provnum).strip()
    if provnum.isdigit():
        provnum = provnum.zfill(6)
    ccn_col = 'ccn'
    if ccn_col not in provider_info_df.columns:
        return None
    facility = provider_info_df[provider_info_df[ccn_col].astype(str).str.strip().str.zfill(6) == provnum].copy()
    if facility.empty:
        return None
    if as_of_canonical and "quarter" in facility.columns:
        req_i = _provider_quarter_sort_int(as_of_canonical)

        def _row_on_or_before_asof(row: pd.Series) -> bool:
            cq = _provider_quarter_to_canonical(row.get("quarter"))
            if not cq:
                return False
            return _provider_quarter_sort_int(cq) <= req_i

        facility = facility[facility.apply(_row_on_or_before_asof, axis=1)]
        if facility.empty:
            return None

    def _quarter_sort_key(q):
        if pd.isna(q) or not str(q).strip():
            return (0, 0)
        s = str(q).strip().upper()
        m = re.search(r'(\d{4})Q(\d)', s)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (0, 0)

    if 'quarter' in facility.columns:
        facility = facility.dropna(subset=['quarter'])
        facility = facility.sort_values('quarter', key=lambda x: x.map(_quarter_sort_key))
        # One row per quarter (last occurrence per quarter)
        by_q = facility.groupby('quarter', sort=False).agg({
            id_col: 'last',
            name_col: 'last',
        }).reset_index()
        by_q['year'] = by_q['quarter'].apply(lambda q: int(str(q)[:4]) if str(q)[:4].isdigit() else None)
        by_q = by_q.dropna(subset=['year'])
    else:
        if 'processing_date' not in facility.columns:
            return None
        facility = facility.sort_values('processing_date')
        by_q = facility[[id_col, name_col]].copy()
        by_q['year'] = pd.to_datetime(facility['processing_date'], errors='coerce').dt.year
        by_q = by_q.dropna(subset=['year'])
    if by_q.empty:
        return None

    def _norm_entity(eid, ename):
        if pd.isna(eid) or str(eid).strip().upper() in ['', 'N', 'N/A', 'NAN', 'NONE']:
            return None, None
        try:
            eid = str(int(float(eid)))
        except (ValueError, TypeError):
            eid = str(eid).strip()
        if not eid:
            return None, None
        ename = ename if pd.notna(ename) and str(ename).strip() else eid
        return eid, str(ename).strip()

    periods = []
    for _, row in by_q.iterrows():
        eid, ename = _norm_entity(row.get(id_col), row.get(name_col))
        y = int(row['year']) if pd.notna(row.get('year')) else None
        if eid is None:
            continue
        if periods and periods[-1][0] == eid:
            periods[-1][3] = y  # extend end year
        else:
            periods.append([eid, ename, y, y])  # start_year at 2, end_year at 3

    if not periods:
        return None
    current_id, current_name = periods[-1][0], periods[-1][1]
    result = {
        'current_entity_id': current_id,
        'current_entity_name': current_name,
        'prev_entity_id': None,
        'prev_entity_name': None,
        'prev_years': None,
    }
    if len(periods) >= 2 and periods[-2][0] != current_id:
        prev = periods[-2]
        result['prev_entity_id'] = prev[0]
        result['prev_entity_name'] = prev[1]
        start_y, end_y = prev[2], prev[3]
        if start_y and end_y:
            result['prev_years'] = f'{start_y}–{end_y}'
        elif start_y:
            result['prev_years'] = str(start_y)
    return result


def _normalize_provider_org_id(val: object) -> Optional[str]:
    """Normalize affiliated entity / chain id for comparison."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    s = str(val).strip()
    if s.upper() in ("", "N", "N/A", "NAN", "NONE", "NULL"):
        return None
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s if s else None


def _provider_quarter_to_canonical(q: object) -> Optional[str]:
    """Map provider_info quarter label to CYyyyyQn.

    CMS / pipeline rows often use ``Q4 2022`` (from interval mapping); PBJ uses ``2022Q4`` or ``CY2022Q4``.
    All must match ``?quarter=2022Q4`` resolution in ``_select_provider_info_row_for_facility``.
    """
    if pd.isna(q) or not str(q).strip():
        return None
    s = str(q).strip().upper().replace("\ufeff", "")
    m = re.search(r"(?:CY)?(\d{4})Q([1-4])", s)
    if m:
        return f"CY{m.group(1)}Q{m.group(2)}"
    m2 = re.match(r"^Q([1-4])\s+(\d{4})\s*$", s.strip())
    if m2:
        return f"CY{m2.group(2)}Q{m2.group(1)}"
    m3 = re.match(r"^(\d{4})\s+Q([1-4])\s*$", s.strip())
    if m3:
        return f"CY{m3.group(1)}Q{m3.group(2)}"
    return None


def _numeric_cell_to_optional_float(val: object) -> Optional[float]:
    """Coerce one spreadsheet / JSON cell to float (pyright-safe vs ``pd.to_numeric`` unions)."""
    if val is None:
        return None
    arr = np.asarray(pd.to_numeric(val, errors="coerce"), dtype=np.float64)
    if arr.size == 0:
        return None
    out = float(arr.flat[0])
    return None if np.isnan(out) else out


def _quarter_sort_key_pre_post(q: object):
    if pd.isna(q) or not str(q).strip():
        return (0, 0)
    s = str(q).strip().upper()
    m = re.search(r"(?:CY)?(\d{4})Q([1-4])", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    m2 = re.match(r"^Q([1-4])\s+(\d{4})\s*$", s.strip())
    if m2:
        return (int(m2.group(2)), int(m2.group(1)))
    m3 = re.match(r"^(\d{4})\s+Q([1-4])\s*$", s.strip())
    if m3:
        return (int(m3.group(1)), int(m3.group(2)))
    return (0, 0)


def _cy_quarter_sort_key_from_canonical(cyq: object):
    """Sort key for normalized CY_Qtr labels."""
    if cyq is None or (isinstance(cyq, float) and np.isnan(cyq)):
        return (0, 0)
    m = re.search(r"CY(\d{4})Q([1-4])", str(cyq).upper())
    if m:
        return (int(m.group(1)), int(m.group(2)))
    return (0, 0)


def _first_day_of_cy_quarter(canonical_q: Optional[str]) -> Optional[pd.Timestamp]:
    """First calendar day of a CMS quarter from CYyyyyQn."""
    if not canonical_q:
        return None
    m = re.search(r"CY(\d{4})Q([1-4])", str(canonical_q).upper())
    if not m:
        return None
    y, q = int(m.group(1)), int(m.group(2))
    month = {1: 1, 2: 4, 3: 7, 4: 10}[q]
    ts = pd.Timestamp(year=y, month=month, day=1)
    if pd.isna(ts):
        return None
    return cast(pd.Timestamp, ts)


# Tokens that usually rebrand with the same operating identity (legal form, care line, boilerplate).
_GENERIC_FACILITY_NAME_TOKENS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "at",
        "in",
        "for",
        "to",
        "llc",
        "llp",
        "pllc",
        "lp",
        "ltd",
        "inc",
        "corp",
        "corporation",
        "company",
        "co",
        "pc",
        "nh",
        "snf",
        "nursing",
        "home",
        "homes",
        "house",
        "center",
        "centre",
        "campus",
        "facility",
        "facilities",
        "health",
        "healthcare",
        "care",
        "rehab",
        "rehabilitation",
        "hospital",
        "medical",
        "clinic",
        "post",
        "acute",
        "subacute",
        "skilled",
        "living",
        "community",
        "communities",
        "pavilion",
        "suites",
        "suite",
        "senior",
        "assisted",
        "regional",
        "memorial",
        "county",
        "city",
        "saint",
        "st",
        "system",
        "services",
        "service",
        "group",
        "holdings",
        "enterprises",
        "management",
        "operations",
    }
)


def _distinctive_provider_name_tokens(s: str) -> set[str]:
    """Words that likely identify the operating brand (not legal/care-model boilerplate)."""
    toks = [t for t in re.sub(r"[^a-z0-9]+", " ", s.lower()).split() if t]
    out: set[str] = set()
    for t in toks:
        if t in _GENERIC_FACILITY_NAME_TOKENS:
            continue
        if len(t) >= 5:
            out.add(t)
        elif len(t) == 4 and t.isalpha():
            out.add(t)
    return out


def _is_significant_provider_name_change(a: object, b: object) -> bool:
    """True if two facility names differ materially (not minor formatting or a synonym tweak)."""
    if pd.isna(a) or pd.isna(b):
        return False
    sa, sb = str(a).strip(), str(b).strip()
    if not sa or not sb:
        return False
    if sa.lower() == sb.lower():
        return False

    def norm_tokens(s: str) -> list[str]:
        return [t for t in re.sub(r"[^a-z0-9]+", " ", s.lower()).split() if t]

    ta, tb = set(norm_tokens(sa)), set(norm_tokens(sb))
    if not ta or not tb:
        return False
    if ta == tb:
        return False
    union = len(ta | tb)
    jacc = len(ta & tb) / union if union else 0.0
    if jacc >= 0.88:
        return False
    sorted_a = " ".join(sorted(ta))
    sorted_b = " ".join(sorted(tb))
    if SequenceMatcher(None, sorted_a, sorted_b).ratio() >= 0.90:
        return False
    compact_a = re.sub(r"[^a-z0-9]+", "", sa.lower())
    compact_b = re.sub(r"[^a-z0-9]+", "", sb.lower())
    if len(compact_a) >= 10 and len(compact_b) >= 10 and (compact_a in compact_b or compact_b in compact_a):
        return False

    # Same strong brand token (e.g. "ARROWHEAD") but only legal/care-line words differ → not material.
    da, db = _distinctive_provider_name_tokens(sa), _distinctive_provider_name_tokens(sb)
    if da and db and (da & db):
        sym = ta ^ tb
        if not sym or all(
            (t in _GENERIC_FACILITY_NAME_TOKENS) or (len(t) <= 2) for t in sym
        ):
            return False

    return True


def _provider_info_quarter_rows(facility_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """One row per CMS quarter (last row per quarter), chronological."""
    if facility_df is None or facility_df.empty:
        return None
    if "quarter" not in facility_df.columns:
        return None
    fac = facility_df.dropna(subset=["quarter"]).copy()
    if fac.empty:
        return None
    fac["_qsort"] = fac["quarter"].map(_quarter_sort_key_pre_post)
    if "processing_date" in fac.columns:
        fac = fac.sort_values(["_qsort", "processing_date"], na_position="last")
    else:
        fac = fac.sort_values("_qsort")
    agg_cols: dict[str, str] = {}
    for col in (
        "provider_changed_ownership_in_last_12_months",
        "Provider Changed Ownership In Last 12 Months",
        "ownership_change",
        "affiliated_entity_id",
        "affiliated_entity_name",
        "chain_id",
        "chain_name",
        "provider_name",
    ):
        if col in fac.columns:
            agg_cols[col] = "last"
    if not agg_cols:
        return None
    by_q = fac.groupby("quarter", sort=False).agg(agg_cols).reset_index()
    by_q["_qsort"] = by_q["quarter"].map(_quarter_sort_key_pre_post)
    by_q = by_q.sort_values("_qsort")
    return by_q


def _provider_quarter_to_chart_anchor_iso(quarter_label: object) -> Optional[str]:
    """Approximate chart pin for a CMS Provider Information quarter label (15th of first month)."""
    cq = _provider_quarter_to_canonical(quarter_label)
    if not cq:
        return None
    m = re.search(r"(?:CY)?(\d{4})Q([1-4])", str(cq).strip(), flags=re.I)
    if not m:
        return None
    try:
        y = int(m.group(1))
        qn = int(m.group(2))
    except ValueError:
        return None
    if qn < 1 or qn > 4:
        return None
    month = (qn - 1) * 3 + 1
    return f"{y}-{month:02d}-15"


def _name_change_chow_date_near_quarter(provnum: str, event_quarter: Optional[str]) -> Optional[str]:
    """When a CMS CHOW effective date falls in/near the name-change quarter, prefer it for the event pin."""
    if not event_quarter:
        return None
    try:
        from pbj_chow_facility import chow_facility_api_payload
    except ImportError:
        return None
    payload = chow_facility_api_payload(str(provnum).strip().zfill(6), limit=8)
    txs = payload.get("transactions") if isinstance(payload, dict) else None
    if not txs:
        return None
    cq = _provider_quarter_to_canonical(event_quarter)
    if not cq:
        return None
    m = re.search(r"(?:CY)?(\d{4})Q([1-4])", str(cq).strip(), flags=re.I)
    if not m:
        return None
    try:
        y = int(m.group(1))
        qn = int(m.group(2))
    except ValueError:
        return None
    window_start = pd.Timestamp(y, (qn - 1) * 3 + 1, 1)
    window_end = window_start + pd.DateOffset(months=9) - pd.Timedelta(days=1)
    best: Optional[str] = None
    best_ts: Optional[pd.Timestamp] = None
    for tx in txs:
        if not isinstance(tx, dict):
            continue
        raw = str(tx.get("effective_date") or "").strip()
        if not raw:
            continue
        try:
            ts = pd.Timestamp(raw).normalize()
        except (ValueError, TypeError, OSError):
            continue
        if ts < window_start or ts > window_end:
            continue
        if best_ts is None or ts > best_ts:
            best_ts = ts
            best = ts.strftime("%Y-%m-%d")
    return best


def _previous_significant_provider_name_for_quarter(
    facility: pd.DataFrame,
    matched_quarter_label: object,
    current_name: object,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    For the provider-info quarter aligned with the matched row, return:
      - most recent prior quarter's CMS provider_name that differs materially from ``current_name``,
      - that prior name's quarter label (last seen),
      - the matched/current quarter label where the new name is established (event quarter).
    """
    if facility is None or facility.empty or "quarter" not in facility.columns:
        return None, None, None
    by_q = _provider_info_quarter_rows(facility)
    if by_q is None or by_q.empty:
        return None, None, None
    mql = str(matched_quarter_label).strip()
    idx: Optional[int] = None
    for i in range(len(by_q)):
        if str(by_q.iloc[i]["quarter"]).strip() == mql:
            idx = i
            break
    if idx is None:
        mc = _provider_quarter_to_canonical(matched_quarter_label)
        if mc:
            for i in range(len(by_q)):
                if _provider_quarter_to_canonical(by_q.iloc[i]["quarter"]) == mc:
                    idx = i
                    break
    if idx is None:
        idx = len(by_q) - 1
    cur = str(current_name).strip() if current_name is not None and not pd.isna(current_name) else ""
    if not cur and "provider_name" in by_q.columns:
        cur = str(by_q.iloc[idx].get("provider_name") or "").strip()
    if not cur:
        return None, None, None
    if idx <= 0:
        return None, None, None
    for j in range(idx - 1, -1, -1):
        prev = by_q.iloc[j].get("provider_name")
        if _is_significant_provider_name_change(prev, cur):
            prev_s = str(prev).strip() if prev is not None and not pd.isna(prev) else ""
            if prev_s:
                ql = str(by_q.iloc[j].get("quarter") or "").strip()
                # Event quarter = first Provider row where the new name is established, not the matched snapshot quarter.
                transition_q = None
                if j + 1 <= idx:
                    transition_q = str(by_q.iloc[j + 1].get("quarter") or "").strip() or None
                event_q = transition_q or str(by_q.iloc[idx].get("quarter") or "").strip() or None
                return prev_s, ql or None, event_q
    return None, None, None


def _detect_pre_post_inflection_provider_info(
    provider_info_df: pd.DataFrame, provnum: str
) -> Optional[dict]:
    """
    Most recent quarter where CMS Provider Information reports
    "Provider Changed Ownership In Last 12 Months" = Y/Yes/True/1.
    When the prior quarter shows a significant provider name change, attach name_from / name_to for the UI.
    """
    provnum = str(provnum).strip()
    if provnum.isdigit():
        provnum = provnum.zfill(6)
    if "ccn" not in provider_info_df.columns:
        return None
    facility = cast(
        pd.DataFrame,
        provider_info_df[
            provider_info_df["ccn"].astype(str).str.strip().str.zfill(6) == provnum
        ].copy(),
    )
    if facility.empty:
        return None
    by_q = _provider_info_quarter_rows(facility)
    if by_q is None or by_q.empty:
        return None
    rows = by_q.reset_index(drop=True)
    def _is_truthy_ownership_flag(v: object) -> bool:
        s = str(v).strip().upper() if v is not None and not pd.isna(v) else ""
        return s in {"Y", "YES", "TRUE", "1"}

    def _clean_display(v: object) -> str:
        if v is None or pd.isna(v):
            return ""
        s = str(v).strip()
        if not s or s.upper() in {"N", "N/A", "NAN", "NONE"}:
            return ""
        return s

    ownership_cols = [
        c
        for c in (
            "provider_changed_ownership_in_last_12_months",
            "Provider Changed Ownership In Last 12 Months",
            "ownership_change",
        )
        if c in rows.columns
    ]
    if not ownership_cols:
        return None

    candidates: list[tuple[int, dict]] = []
    for i in range(len(rows)):
        curr = rows.iloc[i]
        cq = _provider_quarter_to_canonical(curr["quarter"])
        if not cq:
            continue
        sort_i = _provider_quarter_sort_int(cq)
        if not any(_is_truthy_ownership_flag(curr.get(col)) for col in ownership_cols):
            continue

        name_from_pi: Optional[str] = None
        name_to_pi: Optional[str] = None
        if i >= 1:
            prev = rows.iloc[i - 1]
            prev_nm = _clean_display(prev.get("provider_name")) if "provider_name" in prev.index else ""
            curr_nm = _clean_display(curr.get("provider_name")) if "provider_name" in curr.index else ""
            if prev_nm and curr_nm and _is_significant_provider_name_change(prev_nm, curr_nm):
                name_from_pi = prev_nm
                name_to_pi = curr_nm

        entry: dict = {
            "kind": "ownership_12mo",
            "quarter_label": str(curr["quarter"]),
            "canonical_quarter": cq,
            "note_detail": "",
        }
        if name_from_pi and name_to_pi:
            entry["name_from"] = name_from_pi
            entry["name_to"] = name_to_pi
        candidates.append((sort_i, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def _detect_pre_post_inflection_pbj_names(global_df: pd.DataFrame) -> Optional[dict]:
    """Most recent significant PROVNAME change across PBJ quarters (CY_Qtr)."""
    if global_df is None or global_df.empty:
        return None
    if "CY_Qtr" not in global_df.columns or "PROVNAME" not in global_df.columns:
        return None
    g = global_df[["CY_Qtr", "PROVNAME"]].dropna(subset=["CY_Qtr"])
    if g.empty:
        return None
    g = g.copy()
    g["_norm_q"] = g["CY_Qtr"].map(_normalize_cy_qtr)
    g = g.dropna(subset=["_norm_q"])
    if g.empty:
        return None
    last = g.groupby("_norm_q", sort=False)["PROVNAME"].last()
    qs = sorted(last.index.tolist(), key=_cy_quarter_sort_key_from_canonical)
    if len(qs) < 2:
        return None
    best: Optional[dict] = None
    best_key: tuple[int, int] = (-1, -1)
    for i in range(1, len(qs)):
        q_prev, q_curr = qs[i - 1], qs[i]
        if _is_significant_provider_name_change(last[q_prev], last[q_curr]):
            k = _cy_quarter_sort_key_from_canonical(q_curr)
            if k >= best_key:
                best_key = k
                raw_prev = last[q_prev]
                raw_curr = last[q_curr]
                nm_prev = str(raw_prev).strip() if raw_prev is not None and not pd.isna(raw_prev) else ""
                nm_curr = str(raw_curr).strip() if raw_curr is not None and not pd.isna(raw_curr) else ""
                best = {
                    "kind": "pbj_name",
                    "quarter_label": str(q_curr),
                    "canonical_quarter": str(q_curr),
                    "note_detail": "",
                    "name_from": nm_prev,
                    "name_to": nm_curr,
                }
    return best


def _compute_pre_post_windows_from_inflection(
    inflection: pd.Timestamp,
    min_date: pd.Timestamp,
    max_date: pd.Timestamp,
) -> Optional[dict]:
    """
    Pre: 180 calendar days ending the day before inflection.
    Post: from inflection (inclusive) through the latest PBJ work day in the dataset.
    Clamped to PBJ min/max; returns None if unusable.
    """
    post_start = pd.Timestamp(inflection).normalize()
    if post_start > max_date:
        return None
    post_end = max_date
    pre_end = post_start - pd.Timedelta(days=1)
    pre_start = pre_end - pd.Timedelta(days=179)
    if pre_start < min_date:
        pre_start = min_date
    if pre_end < pre_start or post_end < post_start:
        return None
    return {
        "before_start": pre_start.strftime("%Y-%m-%d"),
        "before_end": pre_end.strftime("%Y-%m-%d"),
        "after_start": post_start.strftime("%Y-%m-%d"),
        "after_end": post_end.strftime("%Y-%m-%d"),
    }


def _normalize_requested_quarter_param(q: object) -> Optional[str]:
    """Return canonical CYyyyyQn for API ?quarter= (e.g. 2025Q3, CY2025Q3)."""
    if q is None:
        return None
    s = str(q).strip().upper().replace(" ", "")
    if not s:
        return None
    m = re.match(r"^(?:CY)?(\d{4})Q([1-4])$", s)
    if m:
        return f"CY{m.group(1)}Q{m.group(2)}"
    return None


def _provider_quarter_sort_int(cq: object) -> int:
    """Single int for chronological compare (year*100 + quarter index)."""
    y, q = _quarter_sort_key_pre_post(cq)
    return y * 100 + q


def _fix_facility_name_casing(name: Optional[str]) -> str:
    """Title-style display for CMS facility names (aligned with dashboard JS fixFacilityNameCasing)."""
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    s_norm = " ".join(s.split())
    lower_words = {
        "and",
        "or",
        "of",
        "at",
        "in",
        "on",
        "for",
        "with",
        "by",
        "the",
        "a",
        "an",
    }
    words = s_norm.lower().split()
    result: list[str] = []
    for i, word in enumerate(words):
        if not word:
            continue
        if i == 0:
            result.append(word[0].upper() + word[1:])
        elif word in lower_words:
            result.append(word)
        else:
            result.append(word[0].upper() + word[1:])
    fixed = " ".join(result)
    fixed = re.sub(r"\briver\b", "River", fixed, flags=re.I)
    fixed = re.sub(r"\b(l)\s+(l)\s+(c)\b", "LLC", fixed, flags=re.I)
    return fixed


def _pre_post_event_quarter_display(canonical_or_label: object) -> str:
    """Display label like ``Q2 2023`` for pre/post event notes (from CYyyyyQn or similar)."""
    if canonical_or_label is None:
        return ""
    s = str(canonical_or_label).strip()
    if not s:
        return ""
    s_compact = re.sub(r"\s+", "", s).upper()
    m = re.match(r"^(?:CY)?(\d{4})Q([1-4])$", s_compact)
    if m:
        return f"Q{m.group(2)} {m.group(1)}"
    m2 = re.match(r"^Q([1-4])\s+(\d{4})$", s, re.I)
    if m2:
        return f"Q{m2.group(1)} {m2.group(2)}"
    return s


def _pre_post_event_note_html(
    kind: Optional[str],
    quarter_display: str,
    name_from: Optional[str] = None,
    name_to: Optional[str] = None,
) -> str:
    """Short HTML blurb with source link; optional title-cased (Old → New) for name context."""
    k = (kind or "").strip()
    q_esc = html.escape((quarter_display or "").strip() or "—")
    href = html.escape(_CMS_PROVIDER_INFO_DATASET_PAGE, quote=True)
    nf = _fix_facility_name_casing((name_from or "").strip() or None)
    nt = _fix_facility_name_casing((name_to or "").strip() or None)
    name_paren = ""
    if nf and nt:
        name_paren = " (" + html.escape(nf) + " → " + html.escape(nt) + ")."
    if k == "chow":
        return f'<strong>CMS CHOW</strong> effective date (<strong>{q_esc}</strong>).'
    if k == "ownership_12mo":
        return (
            f'<strong>Ownership change</strong> in '
            f'<a href="{href}" target="_blank" rel="noopener">CMS Provider Information</a> '
            f"(<strong>{q_esc}</strong>).{name_paren}"
        )
    if k == "pbj_name":
        return f'<strong>Facility name change</strong> in PBJ (<strong>{q_esc}</strong>).{name_paren}'
    if k == "affiliated_entity":
        ph = html.escape("Affiliated entity change")
    elif k == "chain":
        ph = html.escape("Chain change")
    else:
        ph = html.escape("Facility name change")
    return (
        f"<strong>{ph}</strong> in "
        f'<a href="{href}" target="_blank" rel="noopener">CMS data</a> '
        f"(<strong>{q_esc}</strong>).{name_paren}"
    )


def _select_provider_info_row_for_facility(
    facility_df: pd.DataFrame,
    requested_canonical: Optional[str],
) -> tuple[Optional[pd.Series], str, str]:
    """
    Pick one provider_info row for this CCN.
    If requested_canonical is set, prefer that CMS quarter (latest processing_date in-quarter),
    else the chronologically latest quarter in the file.
    Returns (row, matched_quarter_label, match_reason).
    """
    if facility_df is None or facility_df.empty:
        return None, "", "empty"
    fac = facility_df.copy()
    if "processing_date" in fac.columns:
        fac["processing_date"] = pd.to_datetime(fac["processing_date"], errors="coerce")
    else:
        fac["processing_date"] = pd.NaT

    if "quarter" not in fac.columns:
        fac = fac.sort_values("processing_date", ascending=True)
        r = fac.iloc[-1]
        return r, str(r.get("quarter", "")), "no_quarter_column"

    fac["_cq"] = fac["quarter"].map(_provider_quarter_to_canonical)
    fac_valid = fac.dropna(subset=["_cq"])
    if fac_valid.empty:
        fac = fac.sort_values("processing_date", ascending=True)
        r = fac.iloc[-1]
        return r, str(r.get("quarter", "")), "no_canonical_quarter"

    if requested_canonical:
        sub = fac_valid[fac_valid["_cq"] == requested_canonical]
        if not sub.empty:
            sub = sub.sort_values("processing_date", ascending=False, na_position="last")
            r = sub.iloc[0]
            return r, str(r.get("quarter", requested_canonical)), "exact_quarter"

        req_i = _provider_quarter_sort_int(requested_canonical)
        leq = fac_valid[fac_valid["_cq"].map(_provider_quarter_sort_int) <= req_i]
        if not leq.empty:
            leq = leq.sort_values(
                ["_cq", "processing_date"],
                ascending=[True, True],
                na_position="last",
            )
            r = leq.iloc[-1]
            return r, str(r.get("quarter", "")), "fallback_leq"

    fac_valid = fac_valid.sort_values(
        ["_cq", "processing_date"],
        ascending=[True, True],
        na_position="last",
    )
    r = fac_valid.iloc[-1]
    return r, str(r.get("quarter", "")), "latest_quarter"


def _sff_status_from_provider_row(row: pd.Series) -> str:
    """SFF display string from one provider_info row."""
    for col in (
        "sff_status",
        "special_focus_status",
        "Special Focus Status",
        "Special Focus Facility Status",
    ):
        if col not in row.index:
            continue
        v = row.get(col)
        if pd.isna(v):
            continue
        s = str(v).strip()
        if not s or s.upper() in ("N/A", "NAN", "NONE", ""):
            continue
        if s.upper() == "N":
            return "No"
        return s
    return "N/A"


def _provider_row_scalar(val: object) -> object:
    """Single cell from a Series row (avoids Series-typed values for Timestamp/float)."""
    if isinstance(val, pd.Series):
        return val.iloc[0] if len(val) else None
    return val


def _safe_float_provider_field(val: object, default: float = 0.0) -> float:
    """Float from a provider row value; NaN/None → default."""
    v = _provider_row_scalar(val)
    if v is None or pd.isna(v):
        return default
    try:
        return float(cast(Any, v))
    except (TypeError, ValueError):
        return default


def _certified_beds_from_provider_row(latest: pd.Series) -> Optional[float]:
    """Certified bed count from a provider-info row (column name variants). Used for profile census."""
    if latest is None or len(latest) == 0:
        return None
    for col in latest.index:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if key not in (
            "certified_beds",
            "number_of_certified_beds",
            "num_certified_beds",
            "no_of_certified_beds",
        ):
            continue
        v = latest.get(col)
        try:
            if v is None or (isinstance(v, float) and pd.isna(v)):
                continue
            f = float(cast(Any, v))
            if f > 0:
                return f
        except (TypeError, ValueError):
            continue
    return None


def _normalize_request_ccn(raw: str | None) -> str | None:
    """Digits only, last 6, zero-pad — aligns with client ``pbjDashboardFacilityCcn``."""
    if raw is None:
        return None
    digits = re.sub(r"\D+", "", str(raw).strip())
    if not digits:
        return None
    if len(digits) > 6:
        digits = digits[-6:]
    return digits.zfill(6)


def _facility_nh_ownership_csv_path(ccn: str) -> Optional[str]:
    """Per-CCN NH ownership slice (deploy bundles) — avoids reading the national CMS file."""
    c = _normalize_request_ccn(ccn)
    if not c:
        return None
    fname = f"facility_{c}_nh_ownership.csv"
    for d in (
        os.path.join(_app_root, "ownership"),
        os.path.join(os.path.abspath(os.path.join(_app_root, "..", "..")), "ownership"),
    ):
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            return p
    return None


def _read_nh_ownership_slice_for_ccn(ccn: str) -> pd.DataFrame:
    """Rows for one CCN from a facility slice or chunked scan of NH_Ownership_*.csv."""
    c = _normalize_request_ccn(ccn)
    if not c:
        return pd.DataFrame()
    cols: set[str] = {
        "CMS Certification Number (CCN)",
        "Provider Name",
        "Owner Name",
        "Role played by Owner or Manager in Facility",
        "Ownership Percentage",
        "Association Date",
    }
    usecols_filter = lambda col_name: str(col_name) in cols
    ccn_col = "CMS Certification Number (CCN)"

    slice_path = _facility_nh_ownership_csv_path(c)
    if slice_path:
        try:
            df_own = pd.read_csv(slice_path, dtype=str, low_memory=False, usecols=usecols_filter)
            if ccn_col in df_own.columns:
                df_own[ccn_col] = (
                    df_own[ccn_col].astype(str).str.replace(r"\.0+$", "", regex=True).str.zfill(6)
                )
                return cast(pd.DataFrame, df_own[df_own[ccn_col] == c].copy())
        except Exception:
            pass

    p = _latest_nh_ownership_csv_path()
    if not p or not os.path.isfile(p):
        return pd.DataFrame()
    parts: list[pd.DataFrame] = []
    try:
        for chunk in pd.read_csv(p, dtype=str, low_memory=False, usecols=usecols_filter, chunksize=100_000):
            if ccn_col not in chunk.columns:
                continue
            chunk[ccn_col] = chunk[ccn_col].astype(str).str.replace(r"\.0+$", "", regex=True).str.zfill(6)
            sub = chunk[chunk[ccn_col] == c]
            if len(sub) > 0:
                parts.append(cast(pd.DataFrame, sub.copy()))
    except Exception:
        return pd.DataFrame()
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _latest_nh_ownership_csv_path() -> Optional[str]:
    """Newest ownership contacts CSV (NH_Ownership_*.csv), searched in repo + deployment contexts."""
    global _NH_OWNERSHIP_CSV_PATH_CACHE
    if _NH_OWNERSHIP_CSV_PATH_CACHE and os.path.isfile(_NH_OWNERSHIP_CSV_PATH_CACHE):
        return _NH_OWNERSHIP_CSV_PATH_CACHE
    cand_dirs = [
        os.path.join(_app_root, "ownership"),
        os.path.join(os.path.abspath(os.path.join(_app_root, "..", "..")), "ownership"),
    ]
    files: list[str] = []
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        files.extend(glob.glob(os.path.join(d, "NH_Ownership_*.csv")))
    if not files:
        _NH_OWNERSHIP_CSV_PATH_CACHE = None
        return None
    files = sorted(set(files), key=lambda p: os.path.getmtime(p), reverse=True)
    _NH_OWNERSHIP_CSV_PATH_CACHE = files[0]
    return _NH_OWNERSHIP_CSV_PATH_CACHE


def _ownership_dataset_bundled_for_ccn(ccn: str) -> bool:
    """True when a per-facility slice or national NH_Ownership file is available on disk."""
    c = _normalize_request_ccn(ccn)
    if not c:
        return False
    if _facility_nh_ownership_csv_path(c):
        return True
    p = _latest_nh_ownership_csv_path()
    return bool(p and os.path.isfile(p))

def _latest_snf_all_owners_csv_path() -> Optional[str]:
    """Newest CMS SNF All Owners CSV (SNF_All_Owners*.csv), for ASSOCIATE ID - OWNER deep links."""
    global _SNF_ALL_OWNERS_CSV_PATH_CACHE
    if _SNF_ALL_OWNERS_CSV_PATH_CACHE and os.path.isfile(_SNF_ALL_OWNERS_CSV_PATH_CACHE):
        return _SNF_ALL_OWNERS_CSV_PATH_CACHE
    cand_dirs = [
        os.path.join(_app_root, "ownership"),
        os.path.join(os.path.abspath(os.path.join(_app_root, "..", "..")), "ownership"),
    ]
    files: list[str] = []
    for d in cand_dirs:
        if not os.path.isdir(d):
            continue
        files.extend(glob.glob(os.path.join(d, "SNF_All_Owners*.csv")))
    if not files:
        _SNF_ALL_OWNERS_CSV_PATH_CACHE = None
        return None
    files = sorted(set(files), key=lambda p: os.path.getmtime(p), reverse=True)
    _SNF_ALL_OWNERS_CSV_PATH_CACHE = files[0]
    return _SNF_ALL_OWNERS_CSV_PATH_CACHE


def _norm_ownership_match_key(raw: object) -> str:
    s = str(raw or "").strip()
    if not s or s.lower() in ("nan", "none", "n/a"):
        return ""
    t = s.upper()
    t = re.sub(r"[^A-Z0-9]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _snf_row_owner_display_key(row: pd.Series) -> str:
    org_o = str(row.get("ORGANIZATION NAME - OWNER") or "").strip()
    if org_o and org_o.lower() not in ("nan", "none", ""):
        return _norm_ownership_match_key(org_o)
    fn = str(row.get("FIRST NAME - OWNER") or "").strip()
    mi = str(row.get("MIDDLE NAME - OWNER") or "").strip()
    ln = str(row.get("LAST NAME - OWNER") or "").strip()
    first = " ".join(p for p in (fn, mi) if p).strip()
    if ln and first:
        return _norm_ownership_match_key(f"{ln}, {first}")
    if ln:
        return _norm_ownership_match_key(ln)
    if first:
        return _norm_ownership_match_key(first)
    return ""


def _snf_facility_owner_associate_map() -> dict[str, dict[str, str]]:
    """Normalized facility ORGANIZATION NAME -> {normalized owner display -> ASSOCIATE ID - OWNER}."""
    global _SNF_FACILITY_OWNER_ASSOC_CACHE, _SNF_FACILITY_OWNER_ASSOC_MTIME
    p = _latest_snf_all_owners_csv_path()
    if not p or not os.path.isfile(p):
        _SNF_FACILITY_OWNER_ASSOC_CACHE = {}
        _SNF_FACILITY_OWNER_ASSOC_MTIME = 0.0
        return {}
    mt = os.path.getmtime(p)
    if mt == _SNF_FACILITY_OWNER_ASSOC_MTIME and _SNF_FACILITY_OWNER_ASSOC_CACHE:
        return _SNF_FACILITY_OWNER_ASSOC_CACHE
    usecols_set = {
        "ORGANIZATION NAME",
        "ASSOCIATE ID - OWNER",
        "ORGANIZATION NAME - OWNER",
        "FIRST NAME - OWNER",
        "MIDDLE NAME - OWNER",
        "LAST NAME - OWNER",
    }

    def _usecol(c: object) -> bool:
        return str(c).strip().replace("\ufeff", "") in usecols_set

    try:
        df = pd.read_csv(p, dtype=str, low_memory=False, encoding="latin-1", usecols=_usecol)
    except Exception:
        try:
            df = pd.read_csv(p, dtype=str, low_memory=False, usecols=_usecol)
        except Exception:
            _SNF_FACILITY_OWNER_ASSOC_CACHE = {}
            _SNF_FACILITY_OWNER_ASSOC_MTIME = mt
            return {}
    df.columns = [str(c).strip().replace("\ufeff", "") for c in df.columns]
    out: dict[str, dict[str, str]] = {}
    for _, r in df.iterrows():
        fac_k = _norm_ownership_match_key(r.get("ORGANIZATION NAME"))
        if not fac_k:
            continue
        ok = _snf_row_owner_display_key(r)
        if not ok:
            continue
        aid = str(r.get("ASSOCIATE ID - OWNER") or "").strip()
        if not aid.isdigit():
            continue
        inner = out.setdefault(fac_k, {})
        inner.setdefault(ok, aid)
    _SNF_FACILITY_OWNER_ASSOC_CACHE = out
    _SNF_FACILITY_OWNER_ASSOC_MTIME = mt
    return out


def _associate_id_for_nh_contact(
    provider_name: str,
    nh_owner_name: str,
    snf_map: dict[str, dict[str, str]],
) -> Optional[str]:
    """Match NH_Ownership provider + owner strings to SNF All Owners ASSOCIATE ID - OWNER."""
    own_k = _norm_ownership_match_key(nh_owner_name)
    if not own_k:
        return None
    fac_keys = [_norm_ownership_match_key(provider_name)]
    loose = re.sub(r"\b(INC|LLC|LP|L P|LTD|CORP|CORPORATION)\b", "", fac_keys[0], flags=re.I)
    loose_k = _norm_ownership_match_key(loose)
    if loose_k and loose_k != fac_keys[0]:
        fac_keys.append(loose_k)
    for fk in fac_keys:
        if not fk:
            continue
        inner = snf_map.get(fk)
        if not inner:
            continue
        if own_k in inner:
            return inner[own_k]
        for ok, aid in inner.items():
            if own_k == ok or own_k in ok or ok in own_k:
                return aid
    # CCN provider name often differs from SNF ORGANIZATION NAME; match owner across the loaded slice.
    for inner in snf_map.values():
        if own_k in inner:
            return inner[own_k]
        for ok, aid in inner.items():
            if own_k == ok or own_k in ok or ok in own_k:
                return aid
    return None


def _ownership_contacts_cache_reset_if_stale() -> None:
    global _OWNERSHIP_CONTACTS_CACHE_SIG, _OWNERSHIP_CONTACTS_CACHE, _SNF_FACILITY_OWNER_ASSOC_MTIME, _SNF_FACILITY_OWNER_ASSOC_CACHE
    nh = _latest_nh_ownership_csv_path()
    snf = _latest_snf_all_owners_csv_path()
    sig = (
        nh,
        os.path.getmtime(nh) if nh and os.path.isfile(nh) else 0.0,
        snf,
        os.path.getmtime(snf) if snf and os.path.isfile(snf) else 0.0,
    )
    if _OWNERSHIP_CONTACTS_CACHE_SIG != sig:
        _OWNERSHIP_CONTACTS_CACHE.clear()
        _OWNERSHIP_CONTACTS_CACHE_SIG = sig
        _SNF_FACILITY_OWNER_ASSOC_MTIME = 0.0
        _SNF_FACILITY_OWNER_ASSOC_CACHE = {}

# States with live public owner pages on www.pbj320.com (/owners/<id>).
_PBJ_OWNER_PUBLIC_PAGE_STATES = frozenset({"NY", "CT"})


def _pbj_owner_public_pages_live_for_state(state: Optional[str]) -> bool:
    st = str(state or "").strip().upper()
    return st in _PBJ_OWNER_PUBLIC_PAGE_STATES


def _ownership_contacts_for_api(ccn: str, facility_state: Optional[str] = None) -> list[dict[str, Any]]:
    """Ownership rows for API/UI with owner portfolio links when CMS associate id is known."""
    return _ownership_contacts_for_ccn(ccn, facility_state=facility_state)


def _pbj_owner_public_href(associate_id: Optional[str]) -> str:
    """Public PBJ320 owner page: https://www.pbj320.com/owners/<ASSOCIATE ID - OWNER>."""
    oid = str(associate_id or "").strip()
    if not oid.isdigit():
        return ""
    origin = _pbj_env_str("PBJ320_MARKETING_ORIGIN").strip().rstrip("/") or "https://www.pbj320.com"
    return f"{origin}/owners/{oid}"


def _owner_portfolio_href(*, display_name: str = "", associate_id: Optional[str] = None) -> str:
    """Deep-link to owner portfolio when a stable CMS associate id is available."""
    oid = str(associate_id or "").strip()
    # Only link when CMS associate id exists; otherwise keep owner names plain text.
    if not oid.isdigit():
        return ""
    key = oid
    tpl = (os.environ.get("PBJ320_OWNER_PORTFOLIO_URL_TEMPLATE") or "").strip()
    if tpl:
        try:
            return tpl.format(owner=key, query=key)
        except (KeyError, ValueError):
            pass
    base = (os.environ.get("PBJ320_OWNER_PORTFOLIO_BASE_URL") or "").strip()
    if base:
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}owner={quote(key)}"
    return _pbj_owner_public_href(key)


def _ownership_source_month_year() -> str:
    """Best-effort MM/YYYY from newest SNF All Owners file/folder name."""
    p = _latest_snf_all_owners_csv_path() or ""
    if not p:
        return ""
    s = p.replace("\\", "/")
    m = re.search(r"(20\d{2})[.\-](\d{2})(?:[.\-]\d{2})", s)
    if not m:
        m = re.search(r"(20\d{2})[.\-](\d{2})", s)
    if not m:
        return ""
    yyyy = m.group(1)
    mm = m.group(2)
    try:
        m_int = int(mm)
        if m_int < 1 or m_int > 12:
            return ""
    except (TypeError, ValueError):
        return ""
    return f"{mm}/{yyyy}"


def _ownership_pct_to_float(raw: object) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("n/a", "nan", "none", "not applicable"):
        return None
    s = s.replace("%", "").replace(",", "").strip()
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return max(0.0, min(100.0, float(v)))


def _ownership_contacts_for_ccn(
    ccn: str,
    *,
    facility_state: Optional[str] = None,
    resolve_associate_ids: Optional[bool] = None,
) -> list[dict[str, Any]]:
    """Ownership contacts for one CCN from NH_Ownership_*.csv (cached per process)."""
    c = _normalize_request_ccn(ccn)
    if not c:
        return []
    _ownership_contacts_cache_reset_if_stale()
    if c in _OWNERSHIP_CONTACTS_CACHE:
        return _OWNERSHIP_CONTACTS_CACHE[c]
    sub = _read_nh_ownership_slice_for_ccn(c)
    if sub.empty:
        _OWNERSHIP_CONTACTS_CACHE[c] = []
        return []
    own_col = "Owner Name"
    role_col = "Role played by Owner or Manager in Facility"
    pct_col = "Ownership Percentage"
    dt_col = "Association Date"
    need_assoc = resolve_associate_ids if resolve_associate_ids is not None else True
    snf_map = _snf_facility_owner_associate_map() if need_assoc else {}
    rows: list[dict[str, Any]] = []
    prov_col = "Provider Name"
    for _, r in sub.iterrows():
        owner_name = str(r.get(own_col) or "").strip()
        if not owner_name or owner_name.lower() in ("nan", "none", "n/a"):
            continue
        provider_name = str(r.get(prov_col) or "").strip() if prov_col in sub.columns else ""
        role = str(r.get(role_col) or "").strip()
        pct_raw = str(r.get(pct_col) or "").strip()
        assoc = str(r.get(dt_col) or "").strip()
        pct = _ownership_pct_to_float(pct_raw)
        role_low = role.lower()
        role_kind = "direct_owner" if ("ownership" in role_low or "owner" in role_low) else "manager_control"
        owner_associate_id = (
            _associate_id_for_nh_contact(provider_name, owner_name, snf_map) if snf_map else None
        )
        rows.append(
            {
                "owner_name": owner_name,
                "owner_name_display": owner_name.title(),
                "owner_associate_id": owner_associate_id,
                "role_text": role,
                "role_kind": role_kind,
                "ownership_pct_raw": pct_raw,
                "ownership_pct": round(float(pct), 4) if pct is not None else None,
                "association_date": assoc,
                "owner_dashboard_href": _owner_portfolio_href(
                    display_name=owner_name, associate_id=owner_associate_id
                ),
            }
        )
    rows.sort(
        key=lambda x: (
            0 if x.get("role_kind") == "direct_owner" else 1,
            -1 * (float(x["ownership_pct"]) if x.get("ownership_pct") is not None else -1.0),
            str(x.get("owner_name") or "").lower(),
        )
    )
    _OWNERSHIP_CONTACTS_CACHE[c] = rows
    return rows


def _medicare_compare_ownership_href(ccn: str, state: str | None) -> str:
    c = _normalize_request_ccn(ccn) or str(ccn or "").strip()
    st = str(state or "").strip().upper()
    base = f"https://www.medicare.gov/care-compare/details/nursing-home/{c}/view-all"
    if len(st) == 2 and st.isalpha():
        return f"{base}?state={quote(st)}&measure=nursing-home-ownership"
    return f"{base}?measure=nursing-home-ownership"


@app.route("/api/facility-chow")
@app.route("/api/facility-chow/<ccn>")
def facility_chow_api(ccn: str | None = None) -> ResponseReturnValue:
    """CMS CHOW transactions for one facility CCN (pbj-root chow_index.json)."""
    prov = str(ccn or request.args.get("ccn") or request.args.get("provnum") or "").strip()
    prov = prov.zfill(6)[-6:] if prov else ""
    if not prov or not prov.isdigit():
        return jsonify({"error": "Invalid or missing CCN", "error_code": "CHOW_INVALID_CCN"}), 400
    try:
        limit = int(request.args.get("limit", "8"))
    except ValueError:
        limit = 8
    return jsonify(chow_facility_api_payload(prov, limit=max(1, min(limit, 25))))


@app.route('/api/provider_info_summary')
def get_provider_info_summary():
    """Get provider info summary for one CCN, optionally matched to a PBJ filter quarter."""
    try:
        global provider_info_df, provider_info_loaded_source, PROVNUM

        if provider_info_df is None:
            return jsonify(
                {
                    "error": "Provider info data not loaded",
                    "error_code": "PROVIDER_INFO_NOT_LOADED",
                    "hint": "Startup did not load facility_*_provider_info_data.csv (missing next to the app, HTTP fetch failed, or CSV parse error). Check the Flask console for [PROVIDER INFO] lines.",
                    "provider_info_source": provider_info_loaded_source,
                }
            )

        provnum = request.args.get("provnum")
        if not provnum:
            p = str(PROVNUM).strip()
            provnum = p if p else None
        provnum = _normalize_request_ccn(provnum)
        if not provnum:
            return jsonify(
                {
                    "error": "provnum required",
                    "error_code": "MISSING_CCN",
                    "hint": "Pass ?provnum=###### or set PROVNUM for single-facility deployments.",
                }
            )

        facility = cast(
            pd.DataFrame,
            provider_info_df[
                provider_info_df["ccn"].astype(str).str.strip().str.zfill(6) == provnum
            ].copy(),
        )
        if facility.empty:
            ccn_col = provider_info_df["ccn"].astype(str).str.strip().str.zfill(6)
            sample = sorted(ccn_col.unique().tolist())[:15]
            return jsonify(
                {
                    "error": "No provider info rows for this CCN",
                    "error_code": "NO_PROVIDER_ROWS_FOR_CCN",
                    "ccn": provnum,
                    "rows_in_file_total": int(len(provider_info_df)),
                    "distinct_ccns_sample": sample,
                    "hint": "The loaded provider CSV has no rows for this CCN. Regenerate create_vercel_deployment / facility provider CSV, or fix the URL CCN.",
                }
            )

        q_param = request.args.get("quarter")
        req_canon = _normalize_requested_quarter_param(q_param)
        latest, matched_q_label, match_reason = _select_provider_info_row_for_facility(facility, req_canon)
        if latest is None:
            return jsonify(
                {
                    "error": "Could not select provider info row",
                    "error_code": "PROVIDER_ROW_SELECT_FAILED",
                    "ccn": provnum,
                    "facility_provider_rows": int(len(facility)),
                    "requested_quarter_param": q_param,
                    "hint": "Rows exist for this CCN but quarter matching failed. Check quarter / processing_date columns in the provider CSV.",
                }
            )

        entity_history = (
            _parse_entity_history(provider_info_df, provnum, as_of_canonical=req_canon)
            if provnum
            else None
        )

        affiliated_entity_name = None
        affiliated_entity_id = None
        if entity_history:
            affiliated_entity_id = entity_history.get('current_entity_id')
            affiliated_entity_name = entity_history.get('current_entity_name')
        if affiliated_entity_id is None and 'affiliated_entity_name' in latest.index and pd.notna(latest.get('affiliated_entity_name')):
            affiliated_entity_name = str(latest.get('affiliated_entity_name')).strip()
            if affiliated_entity_name and affiliated_entity_name.upper() not in ['N', 'N/A', 'NAN', 'NONE', '']:
                pass
            else:
                affiliated_entity_name = None
        if affiliated_entity_id is None and 'affiliated_entity_id' in latest.index and pd.notna(latest.get('affiliated_entity_id')):
            entity_id_raw = _provider_row_scalar(latest.get('affiliated_entity_id'))
            if entity_id_raw is not None and pd.notna(entity_id_raw):
                try:
                    affiliated_entity_id = str(int(float(str(entity_id_raw))))
                except (ValueError, TypeError):
                    affiliated_entity_id = str(entity_id_raw).strip()
            if affiliated_entity_id and affiliated_entity_id.upper() not in ['N', 'N/A', 'NAN', 'NONE', '']:
                pass
            else:
                affiliated_entity_id = None

        entity_ccn = None
        if 'ccn' in latest.index and pd.notna(latest.get('ccn')):
            entity_ccn = str(latest.get('ccn')).strip()

        latest_proc_date_str = ""
        cms_public_filename = ""
        proc_raw = _provider_row_scalar(latest.get("processing_date"))
        if proc_raw is not None and pd.notna(proc_raw):
            try:
                proc_ts = pd.Timestamp(str(proc_raw))
                latest_proc_date_str = proc_ts.strftime("%Y-%m-%d")
                cms_public_filename = f"NH_ProviderInfo_{proc_ts.strftime('%b%Y')}.csv"
            except (ValueError, TypeError, OSError):
                latest_proc_date_str = ""
                cms_public_filename = ""

        prev_facility_nm, prev_facility_nm_q, prev_facility_nm_event_q = _previous_significant_provider_name_for_quarter(
            facility, matched_q_label, latest.get("provider_name")
        )
        prev_facility_nm_event_date = None
        prev_facility_nm_event_date_note = None
        if prev_facility_nm and prev_facility_nm_event_q:
            chow_date = _name_change_chow_date_near_quarter(provnum, prev_facility_nm_event_q)
            if chow_date:
                prev_facility_nm_event_date = chow_date
                prev_facility_nm_event_date_note = "CMS CHOW effective date"
            else:
                prev_facility_nm_event_date = _provider_quarter_to_chart_anchor_iso(prev_facility_nm_event_q)

        summary = {
            'facility_name': str(latest.get('provider_name', 'Unknown')),
            'city': str(latest.get('city', '')),
            'state': str(latest.get('state', '')),
            'county': str(latest.get('county', '')),
            'ownership_type': str(latest.get('ownership_type', '')),
            'care_compare_ownership_url': _medicare_compare_ownership_href(provnum, str(latest.get('state', '') or '')),
            'latest_processing_date': latest_proc_date_str,
            'latest_quarter': matched_q_label or str(latest.get('quarter', '')),
            'certified_beds': _certified_beds_from_provider_row(latest),
            'latest_census': _safe_float_provider_field(latest.get('avg_residents_per_day'), default=0.0),
            'latest_overall_rating': _rating_or_none(latest.get('overall_rating')),
            'latest_staffing_rating': _rating_or_none(latest.get('staffing_rating')),
            'latest_health_inspection_rating': _rating_or_none(latest.get('health_inspection_rating')),
            'latest_quality_rating': _rating_or_none(latest.get('qm_rating')),
            'latest_reported_total_hprd': _safe_float_provider_field(
                latest.get('reported_total_nurse_hrs_per_resident_per_day'), default=0.0
            ),
            'latest_case_mix_total_hprd': _safe_float_provider_field(
                latest.get('case_mix_total_nurse_hrs_per_resident_per_day'), default=0.0
            ),
            'latest_adjusted_total_hprd': _safe_float_provider_field(
                latest.get('adjusted_total_nurse_hrs_per_resident_per_day'), default=0.0
            ),
            'ownership_change_last_12_months': str(latest.get('provider_changed_ownership_in_last_12_months', 'Unknown')) if pd.notna(latest.get('provider_changed_ownership_in_last_12_months')) else 'Unknown',
            'sff_status': _sff_status_from_provider_row(latest),
            'entity_ccn': entity_ccn,
            'affiliated_entity_name': affiliated_entity_name,
            'affiliated_entity_id': affiliated_entity_id,
            'affiliated_entity_prev_id': entity_history.get('prev_entity_id') if entity_history else None,
            'affiliated_entity_prev_name': entity_history.get('prev_entity_name') if entity_history else None,
            'affiliated_entity_prev_years': entity_history.get('prev_years') if entity_history else None,
            'total_records': len(provider_info_df),
            'facility_provider_rows': len(facility),
            'quarters_covered': facility['quarter'].nunique() if 'quarter' in facility.columns else 0,
            'provider_info_source_file': provider_info_loaded_source or '',
            'provider_info_cms_source_file': cms_public_filename,
            'provider_info_cms_dataset_url': _CMS_PROVIDER_INFO_DATASET_PAGE,
            'provider_info_cms_archive_zip_url': (
                _cms_provider_info_archive_zip_url(cms_public_filename) if cms_public_filename else None
            ),
            'ownership_source_month_year': _ownership_source_month_year(),
            'provider_row_match': match_reason,
            'requested_quarter_param': q_param or None,
            'previous_facility_name_significant': prev_facility_nm,
            'previous_facility_name_quarter': prev_facility_nm_q,
            'previous_facility_name_event_quarter': prev_facility_nm_event_q,
            'previous_facility_name_event_date_iso': prev_facility_nm_event_date,
            'previous_facility_name_event_date_note': prev_facility_nm_event_date_note,
            'previous_facility_name_source_label': _provider_info_source_label(proc_raw),
            'previous_facility_name_cms_file': cms_public_filename or None,
            'previous_facility_name_cms_dataset_url': _CMS_PROVIDER_INFO_DATASET_PAGE,
            'ownership_source_month_year': _ownership_source_month_year(),
            'owner_public_pages_live': True,
        }

        return jsonify(summary)

    except Exception as e:
        return jsonify({"error": str(e), "error_code": "PROVIDER_INFO_SUMMARY_EXCEPTION"})


@app.route('/api/ownership-contacts')
def api_ownership_contacts():
    """CMS ownership contacts for one CCN (lazy-loaded; not bundled in provider_info_summary)."""
    provnum = request.args.get("provnum") or request.args.get("ccn") or str(PROVNUM or "").strip()
    provnum = _normalize_request_ccn(provnum)
    if not provnum:
        return jsonify({"error": "provnum required", "error_code": "MISSING_CCN"}), 400
    state = str(request.args.get("state") or "").strip().upper() or None
    try:
        rows = _ownership_contacts_for_api(provnum, state)
        return jsonify(
            {
                "contacts": rows,
                "ownership_source_month_year": _ownership_source_month_year(),
                "owner_public_pages_live": True,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "error_code": "OWNERSHIP_CONTACTS_EXCEPTION"}), 500


@app.route('/api/provider-info/list')
def provider_info_list():
    """List available raw CMS provider info files in static/data/provider_info/."""
    if not os.path.isdir(PROVIDER_INFO_DATA_DIR):
        return jsonify({'files': []})
    try:
        names = [f for f in os.listdir(PROVIDER_INFO_DATA_DIR)
                 if os.path.isfile(os.path.join(PROVIDER_INFO_DATA_DIR, f)) and not f.startswith('.')]
        names.sort()
        return jsonify({'files': names})
    except Exception as e:
        return jsonify({'error': str(e), 'files': []})


@app.route('/api/provider-info/interval-quarter-mapping')
def provider_info_interval_quarter_mapping():
    """
    Build a small table that derives quarter mapping from NH_DataCollectionIntervals_*.csv.

    This is intended for the dedicated `/data-matching` page (evidence for how case-mix/CMI quarters are assigned).
    """
    import time
    import zipfile

    # Small in-memory cache (interval parsing can require reading nested ZIPs)
    global INTERVAL_QUARTER_MAPPING_CACHE
    if "INTERVAL_QUARTER_MAPPING_CACHE" not in globals():
        INTERVAL_QUARTER_MAPPING_CACHE = {"rows": None, "ts": 0}

    limit = request.args.get("limit", "6").strip()
    try:
        limit_i = int(limit)
    except ValueError:
        limit_i = 6
    # Cap to keep server-side ZIP parsing bounded.
    limit_i = max(1, min(limit_i, 60))

    ttl_seconds = 6 * 60 * 60
    now = time.time()
    cached_rows = INTERVAL_QUARTER_MAPPING_CACHE.get("rows") or []
    if cached_rows and (now - INTERVAL_QUARTER_MAPPING_CACHE.get("ts", 0)) < ttl_seconds and len(cached_rows) >= limit_i:
        rows = cached_rows[:limit_i]
        return jsonify({"rows": rows, "cached": True, "limit": limit_i})

    # Find provider_info/ directory (works both in root and in deployments/)
    def _find_provider_info_dir() -> str | None:
        cur = os.path.abspath(os.path.dirname(__file__))
        while True:
            candidate = os.path.join(cur, "provider_info")
            if os.path.isdir(candidate):
                return candidate
            parent = os.path.dirname(cur)
            if parent == cur:
                return None
            cur = parent

    provider_info_dir = _find_provider_info_dir()

    year_re = re.compile(r"nursing_homes_including_rehab_services_(\d{4})\.zip$")
    outer_years: list[int] = []
    if provider_info_dir:
        for fn in os.listdir(provider_info_dir):
            m = year_re.match(fn)
            if m:
                outer_years.append(int(m.group(1)))
    outer_years.sort(reverse=True)

    import calendar
    import urllib.parse

    prov_import_error: str | None = None
    get_manual_quarter_from_processing_month = None  # type: ignore
    try:
        from prov_info_quarter_map import get_manual_quarter_from_processing_month
    except Exception:
        get_manual_quarter_from_processing_month = None  # type: ignore
    interval_import_error: str | None = None
    try:
        from prov_info import (
            get_interval_reporting_period_mapping_for_processing_month,
        )
    except Exception as e:
        interval_import_error = f"Could not import prov_info interval helper: {e}"
        get_interval_reporting_period_mapping_for_processing_month = None  # type: ignore

    months_to_try: list[tuple[int, int]] = []
    if provider_info_dir and get_interval_reporting_period_mapping_for_processing_month:
        for y in outer_years:
            outer_zip = os.path.join(provider_info_dir, f"nursing_homes_including_rehab_services_{y}.zip")
            if not os.path.isfile(outer_zip):
                continue
            try:
                with zipfile.ZipFile(outer_zip, "r") as z_outer:
                    for inner_name in z_outer.namelist():
                        mm = re.search(
                            r"nursing_homes_including_rehab_services_(\d{2})_" + str(y) + r"\.zip$",
                            inner_name,
                        )
                        if not mm:
                            continue
                        month_i = int(mm.group(1))
                        months_to_try.append((y, month_i))
            except Exception:
                continue

    months_to_try.sort(key=lambda t: (t[0], t[1]), reverse=True)

    max_needed = limit_i
    rows_all: list[dict] = []
    tried: set[tuple[int, int]] = set()
    if get_interval_reporting_period_mapping_for_processing_month:
        for (year, month_i) in months_to_try:
            if (year, month_i) in tried:
                continue
            tried.add((year, month_i))
            interval_map = get_interval_reporting_period_mapping_for_processing_month(year, month_i)
            if not interval_map:
                continue

            proc_month_str = f"{year}-{month_i:02d}"
            manual_quarter = (
                get_manual_quarter_from_processing_month(proc_month_str)
                if get_manual_quarter_from_processing_month
                else None
            )
            used_quarter = manual_quarter or interval_map.get("staffing_level_quarter") or ""

            def _provider_info_csv_name(proc_month: str) -> str:
                try:
                    y = int(proc_month[:4])
                    m = int(proc_month[5:7])
                except Exception:
                    return ""
                abbr = calendar.month_abbr[m] if 1 <= m <= 12 else ""
                if not abbr:
                    return ""
                return f"NH_ProviderInfo_{abbr}{y}.csv"

            def _pbj_quarter_url(quarter_label: str) -> str:
                if not quarter_label:
                    return ""
                m = re.match(r"^Q([1-4])\s+(\d{4})$", str(quarter_label).strip().upper())
                if not m:
                    return ""
                q_num = m.group(1)
                q_year = m.group(2)
                return f"https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data/q{q_num.lower()}-{q_year}"

            provider_info_csv_name = _provider_info_csv_name(proc_month_str)
            provider_info_download_url = ""
            if provider_info_csv_name:
                provider_info_download_url = (
                    _cms_provider_info_archive_zip_url(provider_info_csv_name)
                    or _pbj_server_resolved_api_href(
                        f"/api/provider-info/download?file={urllib.parse.quote(provider_info_csv_name)}"
                    )
                )
            pbj_quarter_url = _pbj_quarter_url(used_quarter)

            row = {
                "processing_month": f"{month_i:02d}-{year}",
                "provider_info_csv_name": provider_info_csv_name,
                "provider_info_download_url": provider_info_download_url,
                "interval_csv_name": _strip_interval_csv_display_path(
                    str(interval_map.get("interval_csv_name", "") or "")
                ),
                "staffing_level_from": interval_map.get("staffing_level_from", ""),
                "staffing_level_through": interval_map.get("staffing_level_through", ""),
                "interval_staffing_level_quarter": interval_map.get("staffing_level_quarter", ""),
                "manual_processing_quarter": manual_quarter or "",
                "used_case_mix_quarter": used_quarter,
                "pbj_quarter_url": pbj_quarter_url,
                "turnover_from": interval_map.get("turnover_from", ""),
                "turnover_through": interval_map.get("turnover_through", ""),
                "turnover_quarter": interval_map.get("turnover_quarter", ""),
            }
            rows_all.append(row)
            if len(rows_all) >= max_needed:
                break

    rows_all = _filter_interval_mapping_rows_to_available_pbj(rows_all)
    source = "zip"
    if not rows_all:
        fb = _load_interval_quarter_mapping_fallback(limit_i)
        if fb:
            rows_all = fb
            source = "static_fallback"

    INTERVAL_QUARTER_MAPPING_CACHE["rows"] = rows_all
    INTERVAL_QUARTER_MAPPING_CACHE["ts"] = now
    err_msg = None
    if not rows_all:
        err_msg = interval_import_error or prov_import_error or (
            "No interval rows: add provider_info ZIPs locally or ship static/data/interval_quarter_mapping.json with the app."
        )
    return jsonify(
        {
            "rows": rows_all[:limit_i],
            "cached": False,
            "limit": limit_i,
            "source": source,
            "error": err_msg,
        }
    )


@app.route('/api/provider-info/download')
def provider_info_download():
    """
    Serve a raw CMS provider info file.
    - If file exists locally in PROVIDER_INFO_DATA_DIR: download it.
    - If missing locally but looks like NH_ProviderInfo_MmmYYYY.csv: redirect to the CMS archive ZIP.
    - If no file param: download latest local file; if none, redirect to CMS dataset page.
    """
    file_param = request.args.get('file')
    if file_param:
        # Restrict to filename only (no path traversal)
        name = os.path.basename(file_param).strip()
        if not name or '..' in name or os.path.sep in name:
            return jsonify({'error': 'Invalid file name'}), 400
        if os.path.isdir(PROVIDER_INFO_DATA_DIR):
            path = os.path.join(PROVIDER_INFO_DATA_DIR, name)
            if os.path.isfile(path):
                return send_from_directory(PROVIDER_INFO_DATA_DIR, name, as_attachment=True)
        cms_zip = _cms_provider_info_archive_zip_url(name)
        if cms_zip:
            return redirect(cms_zip, code=302)
        return jsonify({'error': 'File not found'}), 404
    # No file param: send most recent file by name (e.g. latest extract)
    try:
        if os.path.isdir(PROVIDER_INFO_DATA_DIR):
            names = [f for f in os.listdir(PROVIDER_INFO_DATA_DIR)
                     if os.path.isfile(os.path.join(PROVIDER_INFO_DATA_DIR, f)) and not f.startswith('.')]
            if names:
                names.sort()
                return send_from_directory(PROVIDER_INFO_DATA_DIR, names[-1], as_attachment=True)
        return redirect(_CMS_PROVIDER_INFO_DATASET_PAGE, code=302)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/api/citations/download")
def citations_download():
    """Serve the facility-specific NH health citations CSV when present on disk."""
    global citations_loaded_path, PROVNUM
    try:
        expected = f"facility_{str(PROVNUM).strip().zfill(6)}_citations.csv"
        file_param = (request.args.get("file") or "").strip()
        name = os.path.basename(file_param) if file_param else expected
        if not name or name != expected:
            return jsonify({"error": "Invalid file name"}), 400
        path = citations_loaded_path
        if not path or not os.path.isfile(path):
            return jsonify({"error": "Citations file not found"}), 404
        directory, basename = os.path.split(os.path.abspath(path))
        return send_from_directory(directory, basename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _get_latest_sff_status():
    """Get the most recent SFF status from provider info data."""
    try:
        global provider_info_df
        if provider_info_df is None or len(provider_info_df) == 0:
            return 'N/A'
        
        # Try different possible column names for SFF status
        sff_col = None
        for col in ['sff_status', 'special_focus_status', 'Special Focus Status', 'Special Focus Facility Status']:
            if col in provider_info_df.columns:
                sff_col = col
                break
        
        if not sff_col:
            print("SFF status column not found in provider_info_df")
            print(f"Available columns: {list(provider_info_df.columns)}")
            return 'N/A'
        
        print(f"Using SFF column: {sff_col}")
        
        # Sort by processing date (newest first) and find the most recent record with SFF status
        sorted_df = provider_info_df.sort_values('processing_date', ascending=False)
        print(f"Total records: {len(sorted_df)}")
        print(f"Sample SFF values: {sorted_df[sff_col].value_counts().head(10).to_dict()}")
        
        for _, row in sorted_df.iterrows():
            sff_value = row.get(sff_col)
            if pd.notna(sff_value):
                sff_value = str(sff_value).strip()
                if sff_value and sff_value.upper() not in ['N/A', 'NAN', 'NONE', '']:
                    # If it's "N", return "No" for display, otherwise return the actual value
                    if sff_value.upper() == 'N':
                        print(f"Found SFF status: No (N) from {row.get('processing_date')}")
                        return 'No'
                    print(f"Found SFF status: {sff_value} from {row.get('processing_date')}")
                    return sff_value
        
        print("No SFF status found in any provider info records")
        return 'N/A'
    except Exception as e:
        print(f"Error getting SFF status: {e}")
        import traceback
        traceback.print_exc()
        return 'N/A'


def _quarter_from_processing_date(processing_date):
    """Map processing_date to quarter using prov_info mapping when available; fallback to calendar quarter."""
    if pd.isna(processing_date):
        return None
    try:
        from prov_info import get_quarter_from_processing_month

        proc_month = pd.to_datetime(processing_date).strftime("%Y-%m")
        q = get_quarter_from_processing_month(proc_month)
        if q:
            return q
    except Exception:
        pass
    try:
        dt = pd.to_datetime(processing_date)
        y, m = dt.year, dt.month
        q_num = (m - 1) // 3 + 1
        return f"Q{q_num} {y}"
    except Exception:
        return None


def _quarter_sort_key(quarter_label: str) -> tuple:
    """Chronological sort for 'Q1 2024' labels; unknowns last."""
    if not quarter_label or not isinstance(quarter_label, str):
        return (9999, 9)
    s = quarter_label.strip()
    if s == "Present":
        return (9998, 9)
    try:
        if s.startswith("Q") and " " in s:
            parts = s.split()
            qn = int(parts[0][1])
            yn = int(parts[1])
            return (yn, qn)
    except (ValueError, IndexError):
        pass
    return (9999, 9)


def _survey_calendar_quarter_from_ts(ts: Any) -> Optional[str]:
    """Calendar quarter label (e.g. Q2 2024) from a survey / inspection timestamp."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    if not isinstance(ts, pd.Timestamp):
        ts = pd.to_datetime(ts, errors="coerce")
    if pd.isna(ts):
        return None
    try:
        qn = (int(ts.month) - 1) // 3 + 1
        return f"Q{qn} {int(ts.year)}"
    except (ValueError, TypeError, OSError):
        return None


def _citations_red_flags_basename(provnum: str) -> str:
    """Basename of the loaded facility citations CSV for red-flag table source links."""
    global citations_loaded_path
    if citations_loaded_path and isinstance(citations_loaded_path, str):
        bn = os.path.basename(citations_loaded_path.strip())
        if bn and bn.lower().endswith(".csv"):
            return bn
    return f"facility_{str(provnum).strip().zfill(6)}_citations.csv"


def _provider_info_source_label(proc_date: Any) -> str:
    """User-facing CMS source for Provider Information red-flag rows."""
    if pd.notna(proc_date) and isinstance(proc_date, pd.Timestamp):
        return f"CMS Provider Information · {proc_date.strftime('%b %Y')}"
    return "CMS Provider Information"


def _citations_red_flags_source_label() -> str:
    """User-facing CMS source for NH Health Citations / deficiency red-flag rows."""
    asof = _nh_health_citations_dataset_asof_display()
    if asof:
        pub = str(asof).replace(" CMS extract", "").strip()
        if pub:
            return f"CMS Deficiencies · {pub}"
    return "CMS Deficiencies"


def _survey_iso_to_mmddyyyy(iso: str) -> str:
    """Normalize a survey date string to mm-dd-yy for UI display."""
    s = (iso or "").strip()
    if not s:
        return ""
    parts = s.replace("/", "-").split("-")
    if len(parts) >= 3:
        try:
            if len(parts[0]) == 4:
                y, mo, d = int(parts[0]), int(parts[1]), int(parts[2])
            else:
                mo, d, y = int(parts[0]), int(parts[1]), int(parts[2])
            yy = y % 100 if y >= 100 else y
            return f"{mo:02d}-{d:02d}-{yy:02d}"
        except (ValueError, TypeError, IndexError):
            pass
    return s


def _g_plus_citation_flag_label(n: int) -> str:
    """Compact red-flag badge text for G-or-higher deficiency counts."""
    if n == 1:
        return "G+ citation"
    return f"G+ citations ×{int(n)}"


def _citation_survey_tooltip_and_min_iso(sorted_isos: list[str]) -> tuple[str, Optional[str]]:
    """Hover text for G+ citation badges plus an ISO min date for stable sorting."""
    if not sorted_isos:
        return ("Survey date(s): see Inspections & deficiencies table", None)
    fmt = [_survey_iso_to_mmddyyyy(x) for x in sorted_isos]
    tip = f"Survey date: {fmt[0]}" if len(fmt) == 1 else f"Survey dates: {fmt[0]}–{fmt[-1]}"
    return (tip, sorted_isos[0])


def _merge_dashboard_citation_red_flags_into_history(
    provnum: str,
    history: List[dict[str, Any]],
    latest_red_flags: List[str],
) -> None:
    """
    Append / merge NH health citation rows at **G-or-higher** severity (config
    ``dashboard_citation_flag_min_rank``, default 60 = letter G) into red-flag history,
    bucketed by **survey (inspection) calendar quarter** from ``_survey_ts``.
    """
    global citations_df
    if citations_df is None or citations_df.empty:
        return
    try:
        from citation_lib import enrich_citations_dataframe
    except Exception:
        return
    d = enrich_citations_dataframe(citations_df, provnum)
    if d.empty or "is_dashboard_citation_flag" not in d.columns:
        return
    g = d[d["is_dashboard_citation_flag"].fillna(False)]
    if g.empty:
        return
    by_q: dict[str, dict[str, Any]] = {}
    for _, row in g.iterrows():
        qq = _survey_calendar_quarter_from_ts(row.get("_survey_ts"))
        if not qq:
            continue
        slot = by_q.setdefault(qq, {"n": 0, "survey_dates": []})
        slot["n"] = int(slot["n"]) + 1
        siso = row.get("survey_date_iso")
        if isinstance(siso, str) and siso:
            slot.setdefault("survey_dates", []).append(siso)
    if not by_q:
        return
    cit_name = _citations_red_flags_basename(provnum)
    cit_label = _citations_red_flags_source_label()
    total_n = int(len(g))
    latest_flag = _g_plus_citation_flag_label(total_n)
    if latest_flag not in latest_red_flags:
        latest_red_flags.append(latest_flag)
    hist_by_quarter = {str(h.get("quarter")): h for h in history if h.get("quarter")}
    for qq, info in by_q.items():
        n = int(info["n"])
        flag = _g_plus_citation_flag_label(n)
        dates = sorted({x for x in (info.get("survey_dates") or []) if isinstance(x, str) and x})
        tip, min_iso = _citation_survey_tooltip_and_min_iso(dates)
        cite_date_display = _survey_iso_to_mmddyyyy(min_iso) if min_iso else ""
        if qq in hist_by_quarter:
            h = hist_by_quarter[qq]
            rf = [str(x) for x in (h.get("red_flags") or [])]
            if flag not in rf:
                rf.append(flag)
            rf_sorted = sorted(set(rf))
            h["red_flags"] = rf_sorted
            joined = " | ".join(rf_sorted)
            h["status"] = joined
            h["sff_status"] = joined
            h["citation_survey_dates_tooltip"] = tip
            h["citation_source_file"] = cit_name
            h["citation_source_label"] = cit_label
            if cite_date_display:
                h["citation_processing_date"] = cite_date_display
            if min_iso:
                prev = h.get("citation_survey_min_iso")
                if isinstance(prev, str) and prev:
                    h["citation_survey_min_iso"] = min(prev, min_iso)
                else:
                    h["citation_survey_min_iso"] = min_iso
        else:
            history.append(
                {
                    "status": flag,
                    "sff_status": flag,
                    "processing_date": cite_date_display or min_iso or "",
                    "quarter": qq,
                    "source_file": cit_name,
                    "source_label": cit_label,
                    "citation_source_file": cit_name,
                    "citation_source_label": cit_label,
                    "citation_processing_date": cite_date_display,
                    "red_flags": [flag],
                    "record_count": 1,
                    "records": None,
                    "citation_survey_dates_tooltip": tip,
                    "citation_survey_min_iso": min_iso,
                }
            )
    history.sort(key=lambda x: _quarter_sort_key(str(x.get("quarter") or "")))


@app.route('/api/sff_history')
def get_sff_history():
    """Get Red Flag History for a facility (SFF, 1-star ratings, Abuse, Administrator Turnover, Ownership Change,
    and G-or-higher NH health inspection deficiencies when citation data is loaded).
    Provider info columns used: sff_status, overall_rating, staffing_rating, abuse_icon, administrator_turnover,
    provider_changed_ownership_in_last_12_months; quarter and processing_date for mapping to quarters."""
    try:
        global provider_info_df, PROVNUM, citations_df
        if provider_info_df is None or provider_info_df.empty:
            return jsonify({'history': []})
        
        provnum = request.args.get("provnum")
        if not provnum:
            p = str(PROVNUM).strip()
            provnum = p if p else None
        provnum = _normalize_request_ccn(provnum)
        if not provnum:
            return jsonify(
                {
                    "error": "Missing or invalid provnum parameter",
                    "error_code": "MISSING_OR_INVALID_CCN",
                    "hint": "Use ?provnum=###### or deploy with PROVNUM set.",
                }
            )
        
        # Filter rows for this CCN only (avoid copying the entire provider_info_df — large frames + parallel API calls reset Flask dev connections)
        if "ccn" not in provider_info_df.columns:
            return jsonify({"history": []})
        search_variants = [provnum]
        if provnum.isdigit():
            search_variants.extend([provnum.lstrip("0"), provnum.zfill(6)])
        # Normalize CCN: strip whitespace, trailing ".0" from float-like strings, digits only, zfill 6
        ccn_raw = provider_info_df["ccn"].astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)
        ccn_digits = ccn_raw.str.replace(r"\D", "", regex=True)

        def _ccn6(s: str) -> str:
            d = (s or "").strip()
            if not d:
                return ""
            if len(d) > 6:
                d = d[-6:]
            return d.zfill(6)

        ccn_norm = ccn_digits.map(_ccn6)
        facility_data = provider_info_df.loc[ccn_norm.isin(search_variants)].copy()
        
        if facility_data.empty:
            return jsonify({'history': []})
        
        # Sort by processing date
        facility_data = facility_data.sort_values('processing_date')
        
        # Find column names
        sff_col = None
        for col in ['sff_status', 'special_focus_status', 'Special Focus Status']:
            if col in facility_data.columns:
                sff_col = col
                break
        
        overall_rating_col = None
        for col in ['overall_rating', 'Overall Rating']:
            if col in facility_data.columns:
                overall_rating_col = col
                break
        
        staffing_rating_col = None
        for col in ['staffing_rating', 'Staffing Rating']:
            if col in facility_data.columns:
                staffing_rating_col = col
                break
        
        abuse_col = None
        for col in ['abuse_icon', 'Abuse Icon', 'abuse']:
            if col in facility_data.columns:
                abuse_col = col
                break
        
        admin_turnover_col = None
        for col in ['administrator_turnover', 'Administrator Turnover']:
            if col in facility_data.columns:
                admin_turnover_col = col
                break
        
        # Build red flag history - group by quarter
        history_dict = {}  # key: quarter_str, value: dict with combined info

        for _, row in facility_data.iterrows():
            red_flags = []
            
            # Check SFF status (only show if SFF or SFF Candidate, not "N")
            if sff_col and sff_col in row.index:
                sff_value = str(row[sff_col]).strip() if pd.notna(row[sff_col]) else ''
                if sff_value and sff_value.upper() not in ['N', 'N/A', 'NAN', 'NONE', '']:
                    if 'SFF' in sff_value.upper():
                        # Format SFF status: "Special Focus Facility" -> "SFF", "Special Focus Facility Candidate" -> "SFF Candidate"
                        sff_formatted = sff_value
                        if 'CANDIDATE' in sff_value.upper():
                            sff_formatted = 'SFF Candidate'
                        elif 'SPECIAL FOCUS FACILITY' in sff_value.upper():
                            sff_formatted = 'SFF'
                        red_flags.append(sff_formatted)
            
            # Check 1-star overall rating
            if overall_rating_col and overall_rating_col in row.index:
                overall_rating = row[overall_rating_col]
                if pd.notna(overall_rating):
                    try:
                        rating = float(overall_rating)
                        if rating == 1.0:
                            red_flags.append("1-Star Overall")
                    except (ValueError, TypeError):
                        pass
            
            # Check 1-star staffing rating
            if staffing_rating_col and staffing_rating_col in row.index:
                staffing_rating = row[staffing_rating_col]
                if pd.notna(staffing_rating):
                    try:
                        rating = float(staffing_rating)
                        if rating == 1.0:
                            red_flags.append("1-Star Staffing")
                    except (ValueError, TypeError):
                        pass
            
            # Check Abuse - check multiple possible values
            if abuse_col and abuse_col in row.index:
                abuse_value = str(row[abuse_col]).strip() if pd.notna(row[abuse_col]) else ''
                abuse_upper = abuse_value.upper()
                if abuse_upper in ['Y', 'YES', 'TRUE', '1', 'TRUE', 'Y']:
                    red_flags.append("Abuse Icon")
            
            # Check Administrator Turnover (provider info: administrator_turnover = number of admins who left NH in 12 months; display as integer)
            if admin_turnover_col and admin_turnover_col in row.index:
                at_val = row[admin_turnover_col]
                if pd.notna(at_val) and str(at_val).strip():
                    try:
                        at_float = float(at_val)
                        if at_float > 0:
                            red_flags.append(f"Admin TO: {int(at_float)}")
                    except (ValueError, TypeError):
                        if str(at_val).strip().upper() in ['Y', 'YES', 'TRUE', '1']:
                            red_flags.append("Admin TO")
            
            # Check Ownership Change
            ownership_change = False
            ownership_col = None
            for col in ['provider_changed_ownership_in_last_12_months', 'Provider Changed Ownership In Last 12 Months', 'ownership_change']:
                if col in row.index:
                    ownership_col = col
                    ownership_value = str(row[col]).strip() if pd.notna(row[col]) else ''
                    ownership_upper = ownership_value.upper()
                    if ownership_upper in ['Y', 'YES', 'TRUE', '1']:
                        ownership_change = True
                        break
            
            # Include ownership change even if no other red flags
            if ownership_change:
                red_flags.append("Ownership Change")
            
            # Only process if there are red flags or ownership change
            if red_flags:
                # Get processing date and format it
                proc_date = row.get('processing_date')
                if pd.notna(proc_date):
                    if isinstance(proc_date, str):
                        proc_date = pd.to_datetime(proc_date, errors='coerce')
                    # Skip if date is invalid
                    if pd.isna(proc_date):
                        continue
                    # Allow current CMS file dates (e.g. 2026-01-01 rows reporting Q3 2025); do not cap at 2025.
                    proc_date_str = proc_date.strftime('%Y-%m-%d')
                else:
                    proc_date_str = 'Unknown'
                
                # Get quarter from the dataframe column - use the SAME logic as Ratings Over Time chart
                # The quarter column is already populated with the correct mapping when provider_info_df is loaded
                # This ensures consistency across all features
                quarter = row.get('quarter', '')
                quarter_str = None
                
                if pd.notna(quarter) and str(quarter).strip():
                    quarter_raw = str(quarter).strip()
                    
                    # Normalize quarter format to "Q1 2018" format (same as Ratings Over Time chart)
                    # Handle both "2025Q4" and "Q4 2025" formats
                    if len(quarter_raw) == 6 and quarter_raw[4] == 'Q' and quarter_raw[0:4].isdigit() and quarter_raw[5].isdigit():
                        # "2025Q4" format - convert to "Q4 2025"
                        year = quarter_raw[:4]
                        q_num = quarter_raw[5]
                        quarter_str = f"Q{q_num} {year}"
                    elif quarter_raw.startswith('Q') and ' ' in quarter_raw:
                        # Already in "Q4 2025" format
                        quarter_str = quarter_raw

                if quarter_str is None and pd.notna(proc_date):
                    quarter_str = _quarter_from_processing_date(proc_date)
                
                # Skip this record if we couldn't determine a valid quarter
                if quarter_str is None:
                    continue
                
                # Get source file name (internal download id) and user-facing CMS label
                source_file = f"NH_ProviderInfo_{proc_date.strftime('%b%Y')}.csv" if pd.notna(proc_date) and isinstance(proc_date, pd.Timestamp) else 'Provider Info Data'
                source_label = _provider_info_source_label(proc_date)
                
                # Group by quarter - combine red flags and track multiple dates
                if quarter_str not in history_dict:
                    history_dict[quarter_str] = {
                        'quarter': quarter_str,
                        'red_flags_set': set(),  # Use set to avoid duplicates
                        'dates': [],
                        'source_files': [],
                        'source_labels': [],
                        'records': []  # Store individual records for expansion
                    }
                
                # Add red flags to set (automatically handles duplicates)
                history_dict[quarter_str]['red_flags_set'].update(red_flags)
                history_dict[quarter_str]['dates'].append(proc_date_str)
                history_dict[quarter_str]['source_files'].append(source_file)
                history_dict[quarter_str]['source_labels'].append(source_label)
                history_dict[quarter_str]['records'].append({
                    'processing_date': proc_date_str,
                    'source_file': source_file,
                    'source_label': source_label,
                    'red_flags': red_flags
                })
        
        # Convert to list format, sorted by date
        history = []
        for quarter_str, quarter_data in history_dict.items():
            # Sort records by date
            quarter_data['records'].sort(key=lambda x: x['processing_date'])
            
            # Get earliest and latest dates
            dates = sorted(quarter_data['dates'])
            earliest_date = dates[0] if dates else 'Unknown'
            latest_date = dates[-1] if dates else 'Unknown'
            
            # Combine all unique red flags
            all_red_flags = sorted(list(quarter_data['red_flags_set']))
            status_text = " | ".join(all_red_flags)
            
            # Use earliest date for display, but note if there are multiple
            date_display = earliest_date
            if len(dates) > 1 and earliest_date != latest_date:
                date_display = f"{earliest_date} to {latest_date}"
            
            history.append({
                'status': status_text,
                'sff_status': status_text,  # For compatibility
                'processing_date': date_display,
                'quarter': quarter_str,
                'source_file': quarter_data['source_files'][0] if quarter_data['source_files'] else 'Provider Info Data',
                'source_label': quarter_data['source_labels'][0] if quarter_data.get('source_labels') else _provider_info_source_label(None),
                'red_flags': all_red_flags,
                'record_count': len(quarter_data['records']),
                'records': quarter_data['records'] if len(quarter_data['records']) > 1 else None  # Only include if multiple
            })
        
        history.sort(key=lambda x: _quarter_sort_key(x.get("quarter") or ""))
        
        # Latest provider-info snapshot flags (extended below with G+ citations when present)
        latest_red_flags: list[str] = []
        # Add current status if latest record has red flags
        if not facility_data.empty:
            latest_record = facility_data.iloc[-1]
            
            if sff_col and sff_col in latest_record.index:
                sff_value = str(latest_record[sff_col]).strip() if pd.notna(latest_record[sff_col]) else ''
                if sff_value and sff_value.upper() not in ['N', 'N/A', 'NAN', 'NONE', '']:
                    if 'SFF' in sff_value.upper():
                        sff_formatted = 'SFF Candidate' if 'CANDIDATE' in sff_value.upper() else 'SFF'
                        latest_red_flags.append(sff_formatted)
            
            if overall_rating_col and overall_rating_col in latest_record.index:
                overall_rating = latest_record[overall_rating_col]
                if pd.notna(overall_rating):
                    try:
                        if float(overall_rating) == 1.0:
                            latest_red_flags.append("1-Star Overall Rating")
                    except (ValueError, TypeError):
                        pass
            
            if staffing_rating_col and staffing_rating_col in latest_record.index:
                staffing_rating = latest_record[staffing_rating_col]
                if pd.notna(staffing_rating):
                    try:
                        if float(staffing_rating) == 1.0:
                            latest_red_flags.append("1-Star Staffing Rating")
                    except (ValueError, TypeError):
                        pass
            
            if abuse_col and abuse_col in latest_record.index:
                abuse_value = str(latest_record[abuse_col]).strip() if pd.notna(latest_record[abuse_col]) else ''
                if abuse_value.upper() in ['Y', 'YES', 'TRUE', '1']:
                    latest_red_flags.append("Abuse Icon")
            
            if admin_turnover_col and admin_turnover_col in latest_record.index:
                at_val = latest_record[admin_turnover_col]
                if pd.notna(at_val) and str(at_val).strip():
                    try:
                        at_float = float(at_val)
                        if at_float > 0:
                            latest_red_flags.append(f"Admin TO: {int(at_float)}")
                    except (ValueError, TypeError):
                        if str(at_val).strip().upper() in ['Y', 'YES', 'TRUE', '1']:
                            latest_red_flags.append("Admin TO")
            
            # Check Ownership Change for latest record
            for col in ['provider_changed_ownership_in_last_12_months', 'Provider Changed Ownership In Last 12 Months', 'ownership_change']:
                if col in latest_record.index:
                    ownership_value = str(latest_record[col]).strip() if pd.notna(latest_record[col]) else ''
                    ownership_upper = ownership_value.upper()
                    if ownership_upper in ['Y', 'YES', 'TRUE', '1']:
                        latest_red_flags.append("Ownership Change")
                        break
            
            # If latest record has red flags but not in history, add it
            if latest_red_flags:
                latest_date = latest_record.get('processing_date')
                if pd.notna(latest_date):
                    if isinstance(latest_date, str):
                        latest_date = pd.to_datetime(latest_date, errors='coerce')
                    latest_date_str = latest_date.strftime('%Y-%m-%d') if pd.notna(latest_date) else 'Unknown'
                    
                    # Derive quarter from date
                    quarter = latest_record.get('quarter', '')
                    if pd.notna(quarter) and str(quarter).strip():
                        quarter_str = str(quarter).strip()
                        if len(quarter_str) == 6 and 'Q' in quarter_str:
                            quarter_str = f"Q{quarter_str[-1]} {quarter_str[:4]}"
                    elif pd.notna(latest_date) and isinstance(latest_date, pd.Timestamp):
                        year = latest_date.year
                        month = latest_date.month
                        if month <= 3:
                            q = 1
                        elif month <= 6:
                            q = 2
                        elif month <= 9:
                            q = 3
                        else:
                            q = 4
                        quarter_str = f"Q{q} {year}"
                    else:
                        quarter_str = 'Present'
                    
                    status_text = " | ".join(latest_red_flags)
                    
                    # Check if this quarter is already in history
                    quarter_in_history = any(h['quarter'] == quarter_str for h in history)
                    if not quarter_in_history:
                        history.append({
                            'status': status_text,
                            'sff_status': status_text,
                            'processing_date': latest_date_str,
                            'quarter': quarter_str,
                            'source_file': 'Current Data',
                            'red_flags': latest_red_flags,
                            'record_count': 1,
                            'records': None
                        })
        
        _merge_dashboard_citation_red_flags_into_history(provnum, history, latest_red_flags)
        history.sort(key=lambda x: _quarter_sort_key(x.get("quarter") or ""))

        admin_turnover_reviews: list[dict[str, Any]] = []
        try:
            from admin_turnover_review import build_admin_turnover_reviews_for_api

            _ensure_ein_employee_detail_loaded()
            _ensure_nonnurse_loaded()

            def _pi_quarter_from_row(row: pd.Series) -> str | None:
                quarter = row.get("quarter", "")
                if pd.notna(quarter) and str(quarter).strip():
                    qraw = str(quarter).strip()
                    if len(qraw) == 6 and qraw[4] == "Q" and qraw[0:4].isdigit() and qraw[5].isdigit():
                        return f"Q{qraw[5]} {qraw[:4]}"
                    if qraw.startswith("Q") and " " in qraw:
                        return qraw
                proc_date = row.get("processing_date")
                if pd.notna(proc_date):
                    return _quarter_from_processing_date(proc_date)
                return None

            atr = build_admin_turnover_reviews_for_api(
                facility_data=facility_data,
                history=history,
                ein_df=ein_employee_detail_df,
                nonnurse_df=nonnurse_df,
                nurse_df=global_df,
                provnum=provnum,
                quarter_from_row=_pi_quarter_from_row,
            )
            history = atr.get("history") or history
            admin_turnover_reviews = atr.get("admin_turnover_reviews") or []
        except Exception as _atr_exc:
            print(f"[admin_turnover_review] skipped: {_atr_exc}", flush=True)
            admin_turnover_reviews = []

        abuse_flag_context = None
        risk_signals = None
        risk_signals_summary = None
        try:
            from citation_lib import abuse_flag_context_for_api

            abuse_flag_context = abuse_flag_context_for_api(provnum, citations_df, provider_info_df)
        except Exception:
            abuse_flag_context = None
        try:
            from facility_risk_signals import build_sff_history_risk_payload

            risk_payload = build_sff_history_risk_payload(
                provnum,
                citations_df,
                request.args.get("include_risk_signals"),
            )
            risk_signals = risk_payload.get("risk_signals")
            risk_signals_summary = risk_payload.get("risk_signals_summary")
        except Exception:
            risk_signals = None
            risk_signals_summary = None

        return jsonify(
            {
                "history": history,
                "admin_turnover_reviews": admin_turnover_reviews,
                "abuse_flag_context": abuse_flag_context,
                "risk_signals": risk_signals,
                "risk_signals_summary": risk_signals_summary,
                "citations_quarter_methodology_note": (
                    "G+ citation flags (see config dashboard_citation_flag_min_rank) use the calendar "
                    "quarter of each CMS survey (inspection) date. That quarter may not match PBJ staffing quarters "
                    "or Provider Information processing quarters."
                ),
            }
        )
        
    except Exception as e:
        import traceback
        print(f"Error in get_sff_history: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route('/api/provider_info_charts')
def get_provider_info_charts():
    """Get provider info chart data"""
    try:
        global provider_info_df, global_df, _PROVIDER_CHARTS_CACHE, _PROVIDER_CHARTS_CACHE_ID

        if provider_info_df is None:
            return jsonify(
                {
                    "error": "Provider info data not loaded",
                    "error_code": "PROVIDER_INFO_NOT_LOADED",
                    "hint": "Same as provider summary: provider dataframe never loaded at startup.",
                }
            )

        cache_id = (
            id(provider_info_df),
            id(global_df),
            len(provider_info_df) if provider_info_df is not None else 0,
            len(global_df) if global_df is not None else 0,
            _PROVIDER_CHARTS_AGG_VERSION,
        )
        if _PROVIDER_CHARTS_CACHE_ID == cache_id and _PROVIDER_CHARTS_CACHE is not None:
            _cached = _PROVIDER_CHARTS_CACHE
            _rat = _cached.get("ratings") if isinstance(_cached, dict) else None
            # Reject stale cache from older builds (missing QM series keys or wrong array lengths).
            _nq = len(_rat["quarters"]) if isinstance(_rat, dict) and isinstance(_rat.get("quarters"), list) else 0
            _series_ok = (
                isinstance(_rat, dict)
                and _nq > 0
                and all(
                    k in _rat and isinstance(_rat[k], list) and len(_rat[k]) == _nq
                    for k in ("overall", "staffing", "health_inspection", "quality", "long_stay_qm", "short_stay_qm")
                )
            )
            if _series_ok:
                return jsonify(_cached)
            _PROVIDER_CHARTS_CACHE = None
            _PROVIDER_CHARTS_CACHE_ID = None

        # Apply quarter mapping for rows with null quarter (same mapping as prov_info / normalize_provider_info)
        chart_data = provider_info_df.copy()
        # Normalize rating column names (CMS CSVs vary: "Overall Rating" vs overall_rating).
        _col_lower = {str(c).strip().lower(): c for c in chart_data.columns}

        def _alias_col(std: str, *alts: str) -> None:
            if std in chart_data.columns:
                return
            for a in alts:
                k = a.strip().lower()
                if k in _col_lower:
                    chart_data[std] = chart_data[_col_lower[k]]
                    return

        _alias_col("overall_rating", "overall_rating", "Overall Rating")
        _alias_col("staffing_rating", "staffing_rating", "Staffing Rating")
        _alias_col("health_inspection_rating", "health_inspection_rating", "Health Inspection Rating")
        _alias_col(
            "qm_rating",
            "qm_rating",
            "QM Rating",
            "Quality Measure Rating",
            "Quality Measures Rating",
            "Quality Measure Five-Star Rating",
            "Quality Measures Five-Star Rating",
        )
        _alias_col(
            "long_stay_qm_rating",
            "long_stay_qm_rating",
            "Long Stay QM Rating",
            "Long-Stay QM Rating",
            "Long-Stay Quality Measure Rating",
            "Long Stay Quality Measure Rating",
        )
        _alias_col(
            "short_stay_qm_rating",
            "short_stay_qm_rating",
            "Short Stay QM Rating",
            "Short-Stay QM Rating",
            "Short-Stay Quality Measure Rating",
            "Short Stay Quality Measure Rating",
        )
        if 'quarter' not in chart_data.columns:
            chart_data['quarter'] = None
        if 'processing_date' in chart_data.columns:
            null_quarter = chart_data['quarter'].isna()
            if null_quarter.any():
                chart_data.loc[null_quarter, 'quarter'] = chart_data.loc[null_quarter, 'processing_date'].apply(
                    _quarter_from_processing_date
                )
        
        # Helper function to normalize quarter format
        def normalize_quarter_for_matching(q):
            """Convert quarter to PBJ-style ``yyyyQn`` (accepts CY2018Q1, 2018Q1, Q1 2018)."""
            if q is None:
                return None
            if isinstance(q, float) and pd.isna(q):
                return None
            q_str = str(q).strip()
            if not q_str:
                return None
            u = q_str.upper().replace(" ", "")
            mcy = re.search(r"(?:CY)?(\d{4})Q([1-4])", u)
            if mcy:
                return f"{mcy.group(1)}Q{mcy.group(2)}"
            if q_str.startswith("Q") and " " in q_str:
                parts = q_str.replace("Q", "").split()
                if len(parts) >= 2:
                    quarter_num = parts[0]
                    year = "".join(parts[1:])
                    if quarter_num.isdigit() and year.isdigit():
                        return f"{year}Q{quarter_num}"
            return None
        
        # One logical row per raw quarter label: newest non-null per column (CMS resubmissions).
        chart_data = chart_data.dropna(subset=["quarter"]).copy()

        def _coalesce_provider_chart_group(g: pd.DataFrame) -> pd.Series:
            return coalesce_provider_quarter_snapshots(g)

        chart_data = (
            chart_data.sort_values("processing_date", ascending=True, na_position="first")
            .groupby("quarter", group_keys=False)
            .apply(_coalesce_provider_chart_group)
            .reset_index(drop=True)
        )
        
        # Create normalized quarter column for matching with PBJ data
        chart_data['quarter_normalized'] = chart_data['quarter'].apply(normalize_quarter_for_matching)
        
        # Format quarter labels for x-axis (Q1 2021 instead of 2021Q1)
        # Preserve original format if it's already "Q1 2018", otherwise convert from "2018Q1"
        def format_quarter_label(q):
            if pd.isna(q):
                return None
            q_str = str(q).strip()
            if q_str.startswith("Q") and " " in q_str:
                return q_str
            u = q_str.upper().replace(" ", "")
            mcy = re.search(r"(?:CY)?(\d{4})Q([1-4])", u)
            if mcy:
                return f"Q{mcy.group(2)} {mcy.group(1)}"
            return q_str
        
        chart_data['quarter_label'] = chart_data['quarter'].apply(format_quarter_label)
        # Ensure one row per normalized quarter; coalesce again when two raw labels map to the same key.
        chart_data = (
            chart_data[chart_data["quarter_normalized"].notna()]
            .sort_values("processing_date", ascending=True, na_position="first")
            .groupby("quarter_normalized", group_keys=False)
            .apply(_coalesce_provider_chart_group)
            .reset_index(drop=True)
        )
        chart_data["quarter_label"] = chart_data["quarter_normalized"].apply(
            lambda q: format_quarter_label(q) if pd.notna(q) else None
        )

        pbj_direct_lookup: dict[str, tuple[float, float, float]] = {}
        if global_df is not None and len(global_df) > 0 and "CY_Qtr" in global_df.columns:
            pbj_direct_lookup = _pbj_direct_hprd_lookup_from_global_df(global_df)

        # Add PBJ-calculated direct care values (excludes admin/DON) by matching quarters
        if global_df is not None and len(global_df) > 0:
            pbj_direct_data = []
            for idx, row in chart_data.iterrows():
                quarter_orig = row["quarter"]
                quarter_normalized = row["quarter_normalized"]
                dt, rn, lpn = _lookup_pbj_direct_hprd(pbj_direct_lookup, quarter_normalized)
                if dt is not None:
                    pbj_direct_data.append(
                        {
                            "quarter": quarter_orig,
                            "pbj_direct_total": dt,
                            "pbj_rn_direct": rn or 0.0,
                            "pbj_lpn_direct": lpn if lpn is not None else 0.0,
                        }
                    )
                else:
                    pbj_direct_data.append(
                        {"quarter": quarter_orig, "pbj_direct_total": 0, "pbj_rn_direct": 0, "pbj_lpn_direct": 0}
                    )

            pbj_direct_df = pd.DataFrame(pbj_direct_data)
            chart_data = chart_data.merge(pbj_direct_df, on="quarter", how="left")
        else:
            chart_data["pbj_direct_total"] = 0
            chart_data["pbj_rn_direct"] = 0
            chart_data["pbj_lpn_direct"] = 0
        
        # Sort quarters chronologically using normalized format
        def quarter_sort_key(q_norm):
            """Sort key for ``yyyyQn`` or ``CYyyyyQn``."""
            if pd.isna(q_norm):
                return (9999, 9)
            u = str(q_norm).strip().upper().replace(" ", "")
            mcy = re.search(r"(?:CY)?(\d{4})Q([1-4])", u)
            if mcy:
                try:
                    return (int(mcy.group(1)), int(mcy.group(2)))
                except (ValueError, TypeError):
                    pass
            return (9999, 9)
        
        chart_data['_sort_key'] = chart_data['quarter_normalized'].apply(quarter_sort_key)
        chart_data = chart_data.sort_values('_sort_key').drop(['_sort_key'], axis=1)
        
        # Use case_mix_total for "direct" case-mix in chart too (same as table); no combined RN+LPN+NA
        chart_data['case_mix_direct'] = chart_data['case_mix_total_nurse_hrs_per_resident_per_day']
        
        # Global quarter list: union of all quarters present in any series (PBJ + provider info) so partial data shows all quarters on x-axis with null where a series has no data.
        chart_by_q = chart_data.set_index("quarter_normalized")
        provider_quarters_norm = chart_data["quarter_normalized"].dropna().astype(str).unique().tolist()

        def _canonical_chart_quarter_key(q: object) -> Optional[str]:
            if q is None or (isinstance(q, float) and pd.isna(q)):
                return None
            u = str(q).strip().upper().replace(" ", "")
            mcy = re.search(r"(?:CY)?(\d{4})Q([1-4])", u)
            return f"{mcy.group(1)}Q{mcy.group(2)}" if mcy else None

        if global_df is not None and len(global_df) > 0 and "CY_Qtr" in global_df.columns:
            pbj_keys = {_canonical_chart_quarter_key(x) for x in global_df["CY_Qtr"].dropna().unique().tolist()}
            prov_keys = {_canonical_chart_quarter_key(x) for x in provider_quarters_norm}
            all_quarters_norm = sorted({k for k in (pbj_keys | prov_keys) if k}, key=quarter_sort_key)
        else:
            all_quarters_norm = sorted(
                {k for k in (_canonical_chart_quarter_key(x) for x in provider_quarters_norm) if k},
                key=quarter_sort_key,
            )
        full_quarter_norm = all_quarters_norm
        def quarter_to_iso_start(q_norm):
            """Convert normalized quarter e.g. 2023Q4 to ISO quarter-start date 2023-10-01."""
            if pd.isna(q_norm):
                return None
            q_str = str(q_norm).strip()
            try:
                if len(q_str) == 6 and q_str[4] == 'Q':
                    y, q = int(q_str[:4]), int(q_str[5])
                    month = (q - 1) * 3 + 1
                    return f"{y}-{month:02d}-01"
            except (ValueError, TypeError):
                pass
            return None
        quarter_start_dates = [quarter_to_iso_start(q) for q in full_quarter_norm]
        def _full_quarter_labels():
            out = []
            for q in full_quarter_norm:
                if q not in chart_by_q.index:
                    out.append(format_quarter_label(q))
                    continue
                lab = chart_by_q.loc[q, "quarter_label"]
                if isinstance(lab, pd.Series):
                    lab = lab.iloc[-1]
                out.append(lab if pd.notna(lab) else format_quarter_label(q))
            return out
        def _val_at(q, col, default=None):
            if q not in chart_by_q.index:
                return default
            v = chart_by_q.loc[q, col]
            if isinstance(v, pd.Series):
                v = v.iloc[-1]
            return default if pd.isna(v) else v
        def _series(column, fill_missing=0):
            out = []
            for q in full_quarter_norm:
                if q in chart_by_q.index:
                    v = chart_by_q.loc[q, column]
                    if isinstance(v, pd.Series):
                        v = v.iloc[-1]
                    out.append(fill_missing if (pd.isna(v) or v is None) else v)
                else:
                    out.append(None)
            return out
        def _series_none(column):
            out = []
            for q in full_quarter_norm:
                if q in chart_by_q.index:
                    v = chart_by_q.loc[q, column]
                    if isinstance(v, pd.Series):
                        v = v.iloc[-1]
                    out.append(None if pd.isna(v) else v)
                else:
                    out.append(None)
            return out
        def _series_none_reported(column):
            """Provider-info reported columns (same as _series_none: value or None for missing/NaN)."""
            return _series_none(column)
        def _pbj_direct_at(q):
            """Get (direct_total_hprd, rn_direct_hprd) from PBJ for quarter q when q is in global_df; else (None, None). So quarters that exist only in PBJ still get direct series values."""
            return _lookup_pbj_direct_hprd(pbj_direct_lookup, q)
        def _series_direct_total():
            out = []
            for q in full_quarter_norm:
                if q in chart_by_q.index:
                    v = chart_by_q.loc[q, 'pbj_direct_total']
                    if isinstance(v, pd.Series):
                        v = v.iloc[-1]
                    out.append(None if pd.isna(v) else v)
                else:
                    out.append(_pbj_direct_at(q)[0])
            return out
        def _series_direct_rn():
            out = []
            for q in full_quarter_norm:
                if q in chart_by_q.index:
                    v = chart_by_q.loc[q, 'pbj_rn_direct']
                    if isinstance(v, pd.Series):
                        v = v.iloc[-1]
                    out.append(None if pd.isna(v) else v)
                else:
                    out.append(_pbj_direct_at(q)[1])
            return out
        def _series_direct_lpn():
            out = []
            for q in full_quarter_norm:
                if q in chart_by_q.index:
                    v = chart_by_q.loc[q, 'pbj_lpn_direct']
                    if isinstance(v, pd.Series):
                        v = v.iloc[-1]
                    out.append(None if pd.isna(v) else v)
                else:
                    out.append(_pbj_direct_at(q)[2])
            return out
        provider_quarters = _full_quarter_labels()
        # Ratings: same full quarter list; null for missing (no 0). CMS uses half-stars; preserve 0.5 steps for plotting.
        def _rating_val(v):
            if v is None or pd.isna(v):
                return None
            try:
                f = float(v)
            except (ValueError, TypeError):
                return None
            if 1 <= f <= 5:
                return round(f * 2) / 2.0
            return None

        def _ratings_series_for_col(col: str) -> list[Any]:
            if col not in chart_data.columns:
                return [None] * len(full_quarter_norm)
            return [_rating_val(_val_at(q, col)) for q in full_quarter_norm]

        ratings_overall = _ratings_series_for_col("overall_rating")
        ratings_staffing = _ratings_series_for_col("staffing_rating")
        ratings_health = _ratings_series_for_col("health_inspection_rating")
        ratings_quality = _ratings_series_for_col("qm_rating")
        ratings_long_stay_qm = _ratings_series_for_col("long_stay_qm_rating")
        ratings_short_stay_qm = _ratings_series_for_col("short_stay_qm_rating")
        # Occupancy: census / certified_beds from provider info (same quarter matching as PBJ)
        def _occupancy_series():
            out_pct, out_census, out_beds = [], [], []
            beds_col = next((c for c in chart_data.columns if str(c).strip().lower() == 'certified_beds'), None)
            for q in full_quarter_norm:
                census = _val_at(q, 'avg_residents_per_day')
                beds = _val_at(q, beds_col) if beds_col else None
                if census is not None and pd.notna(census) and beds is not None and pd.notna(beds) and float(beds) > 0:
                    try:
                        c, b = float(census), float(beds)
                        if b > 0 and c >= 0:
                            out_pct.append(round(c / b * 100, 2))
                            out_census.append(round(c, 1))
                            out_beds.append(int(b))
                        else:
                            out_pct.append(None)
                            out_census.append(None)
                            out_beds.append(None)
                    except (TypeError, ValueError):
                        out_pct.append(None)
                        out_census.append(None)
                        out_beds.append(None)
                else:
                    out_pct.append(None)
                    out_census.append(None)
                    out_beds.append(None)
            return out_pct, out_census, out_beds
        _occ_pct, _occ_census, _occ_beds = _occupancy_series()
        charts = {
            'total_staffing': {
                'quarters': provider_quarters,
                'quarter_start_dates': quarter_start_dates,
                'reported_total': _series_none_reported('reported_total_nurse_hrs_per_resident_per_day'),
                'reported_direct': _series_direct_total(),
                'case_mix_total': _series_none('case_mix_total_nurse_hrs_per_resident_per_day'),
                'case_mix_direct': _series_none('case_mix_direct'),
                'adjusted_total': _series_none('adjusted_total_nurse_hrs_per_resident_per_day')
            },
            'rn_staffing': {
                'quarters': provider_quarters,
                'quarter_start_dates': quarter_start_dates,
                'reported_rn': _series_none('reported_rn_hrs_per_resident_per_day'),
                'reported_rn_total': _series_none('reported_rn_hrs_per_resident_per_day'),
                'reported_rn_direct': _series_direct_rn(),
                'case_mix_rn': _series_none('case_mix_rn_hrs_per_resident_per_day'),
                'adjusted_rn': _series_none('adjusted_rn_hrs_per_resident_per_day')
            },
            'lpn_staffing': {
                'quarters': provider_quarters,
                'quarter_start_dates': quarter_start_dates,
                'reported_lpn': _series_none('reported_lpn_hrs_per_resident_per_day'),
                'reported_lpn_direct': _series_direct_lpn(),
                'case_mix_lpn': _series_none('case_mix_lpn_hrs_per_resident_per_day'),
                'adjusted_lpn': _series_none('adjusted_lpn_hrs_per_resident_per_day')
            },
            'cna_staffing': {
                'quarters': provider_quarters,
                'quarter_start_dates': quarter_start_dates,
                'reported_cna': _series_none_reported('reported_na_hrs_per_resident_per_day'),
                'case_mix_cna': _series_none('case_mix_na_hrs_per_resident_per_day'),
                'case_mix_lpn': _series_none('case_mix_lpn_hrs_per_resident_per_day'),
                'adjusted_cna': _series_none('adjusted_na_hrs_per_resident_per_day')
            },
            'census': {
                'quarters': provider_quarters,
                'census': _series('avg_residents_per_day')
            },
            'ratings': {
                'quarters': provider_quarters,
                'overall': ratings_overall,
                'staffing': ratings_staffing,
                'health_inspection': ratings_health,
                'quality': ratings_quality,
                'long_stay_qm': ratings_long_stay_qm,
                'short_stay_qm': ratings_short_stay_qm,
            },
            'occupancy': {
                'quarters': provider_quarters,
                'occupancy_pct': _occ_pct,
                'avg_census': _occ_census,
                'certified_beds': _occ_beds
            }
        }
        
        # Convert any remaining NaN values to None for JSON serialization
        def convert_nan_to_none(obj):
            if isinstance(obj, dict):
                return {key: convert_nan_to_none(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_nan_to_none(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        charts = cast(dict[str, Any], convert_nan_to_none(charts))
        _PROVIDER_CHARTS_CACHE = charts
        _PROVIDER_CHARTS_CACHE_ID = cache_id

        return jsonify(charts)

    except Exception as e:
        return jsonify({"error": str(e), "error_code": "PROVIDER_CHARTS_EXCEPTION"})

def _daily_json_records_from_filtered_df(filtered_df: pd.DataFrame) -> dict[str, Any]:
    """Build /api/data payload dict from an already-filtered daily frame."""
    data = filtered_df.to_dict("records")
    for record in data:
        for key, value in list(record.items()):
            if pd.isna(value):
                record[key] = None
            elif key == "WorkDate" and value is not None and not pd.isna(value):
                record[key] = _format_workdate_json(value)
    wd_coerced = pd.to_datetime(filtered_df["WorkDate"], errors="coerce")
    min_date = wd_coerced.min()
    max_date = wd_coerced.max()
    date_range: dict[str, str] = {}
    if pd.notna(min_date):
        date_range["min"] = min_date.strftime("%Y-%m-%d")
    if pd.notna(max_date):
        date_range["max"] = max_date.strftime("%Y-%m-%d")
    return {"data": data, "total_records": len(data), "date_range": date_range}


def _summary_dict_from_filtered_daily_df(filtered_df: pd.DataFrame) -> dict[str, Any]:
    # Ensure derived contract percentage exists even if lazy/partial loads skipped derived columns.
    if "Total_Contract_Pct" not in filtered_df.columns:
        if all(c in filtered_df.columns for c in ["RN_Contract_Pct", "LPN_Contract_Pct", "CNA_Contract_Pct"]):
            filtered_df["Total_Contract_Pct"] = filtered_df[
                ["RN_Contract_Pct", "LPN_Contract_Pct", "CNA_Contract_Pct"]
            ].mean(axis=1, skipna=True)
        elif all(c in filtered_df.columns for c in ["Hrs_RN_ctr", "Hrs_LPN_ctr", "Hrs_CNA_ctr", "Nurse_Staff_Hours_Excl_Admin"]):
            denom = pd.to_numeric(filtered_df["Nurse_Staff_Hours_Excl_Admin"], errors="coerce")
            denom = denom.where(denom != 0, np.nan)
            num = (
                pd.to_numeric(filtered_df["Hrs_RN_ctr"], errors="coerce")
                + pd.to_numeric(filtered_df["Hrs_LPN_ctr"], errors="coerce")
                + pd.to_numeric(filtered_df["Hrs_CNA_ctr"], errors="coerce")
            )
            filtered_df["Total_Contract_Pct"] = (num / denom * 100).replace([np.inf, -np.inf], np.nan)
        else:
            filtered_df["Total_Contract_Pct"] = np.nan

    # Calculate summary statistics with proper financial rounding
    # Pooled HPRD: sum(hours) / sum(census) on days with census > 0 and non-missing hours (denominator)
    mdc_sum = (
        cast(
            pd.Series,
            pd.to_numeric(_as_1d_series(filtered_df["MDScensus"]), errors="coerce"),
        )
        if len(filtered_df)
        else pd.Series(dtype=float)
    )
    total_rn_hours = float(filtered_df["Hrs_RN"].sum()) if len(filtered_df) > 0 and "Hrs_RN" in filtered_df.columns else 0.0
    total_rn_all_hours = (
        float(filtered_df["Total_RN_Hours"].sum()) if len(filtered_df) > 0 and "Total_RN_Hours" in filtered_df.columns else 0.0
    )
    total_lpn_hours = float(filtered_df["Hrs_LPN"].sum()) if len(filtered_df) > 0 and "Hrs_LPN" in filtered_df.columns else 0.0
    total_cna_hours = float(filtered_df["Hrs_CNA"].sum()) if len(filtered_df) > 0 and "Hrs_CNA" in filtered_df.columns else 0.0
    nurse_staff_hours_excl_admin = (
        _sum_pbj_nurse_staff_hours_excl_admin(filtered_df) if len(filtered_df) > 0 else 0.0
    )
    total_staff_hours = (
        float(filtered_df["Total_Staff_Hours"].sum()) if len(filtered_df) > 0 and "Total_Staff_Hours" in filtered_df.columns else 0.0
    )
    total_nurse_aide_hours = (
        float(filtered_df["Total_Nurse_Aide_Hours"].sum(min_count=1))
        if len(filtered_df) > 0 and "Total_Nurse_Aide_Hours" in filtered_df.columns
        else 0.0
    )
    
    # Calculate indirect staffing hours (RN Admin + RN DON + LPN Admin)
    indirect_staffing_hours = 0.0
    if len(filtered_df) > 0:
        _ind = 0.0
        for _hc in ("Hrs_RNadmin", "Hrs_RNDON", "Hrs_LPNadmin"):
            if _hc in filtered_df.columns:
                _ind += float(
                    cast(
                        Any,
                        cast(
                            pd.Series,
                            pd.to_numeric(_as_1d_series(filtered_df[_hc]), errors="coerce"),
                        ).sum(min_count=1),
                    )
                )
        indirect_staffing_hours = _ind
    
    avg_rn_hprd_weighted = _pooled_hprd_hours_ratio(filtered_df, "Hrs_RN")
    avg_total_rn_hprd_weighted = _pooled_hprd_hours_ratio(filtered_df, "Total_RN_Hours")
    avg_lpn_hprd_weighted = _pooled_hprd_hours_ratio(filtered_df, "Hrs_LPN")
    avg_cna_hprd_weighted = _pooled_hprd_hours_ratio(filtered_df, "Hrs_CNA")
    hour_cols_dir = ["Hrs_RN", "Hrs_LPN", "Hrs_CNA", "Hrs_NAtrn", "Hrs_MedAide"]
    if len(filtered_df) > 0 and all(c in filtered_df.columns for c in hour_cols_dir):
        M = filtered_df[hour_cols_dir].apply(lambda col: pd.to_numeric(col, errors="coerce"))
        row_direct = M.sum(axis=1, min_count=len(hour_cols_dir))
        ok = mdc_sum.notna() & (mdc_sum > 0) & row_direct.notna()
        den_ns = float(cast(Any, mdc_sum[ok].sum()))
        avg_nurse_staff_hprd_excl_admin_weighted = (
            float(cast(Any, row_direct[ok].sum()) / den_ns) if den_ns > 0 else None
        )
    elif len(filtered_df) > 0 and "Nurse_Staff_Hours_Excl_Admin" in filtered_df.columns:
        ns = cast(
            pd.Series,
            pd.to_numeric(_as_1d_series(filtered_df["Nurse_Staff_Hours_Excl_Admin"]), errors="coerce"),
        )
        ok = mdc_sum.notna() & (mdc_sum > 0) & ns.notna()
        den_ns = float(cast(Any, mdc_sum[ok].sum()))
        avg_nurse_staff_hprd_excl_admin_weighted = (
            float(cast(Any, ns[ok].sum()) / den_ns) if den_ns > 0 else None
        )
    else:
        avg_nurse_staff_hprd_excl_admin_weighted = None
    avg_total_hprd_weighted = _pooled_hprd_hours_ratio(filtered_df, "Total_Staff_Hours")
    avg_total_nurse_aide_hprd_weighted = _pooled_hprd_hours_ratio(filtered_df, "Total_Nurse_Aide_Hours")
    if (
        len(filtered_df) > 0
        and "MDScensus" in filtered_df.columns
        and all(c in filtered_df.columns for c in ("Hrs_RNadmin", "Hrs_RNDON", "Hrs_LPNadmin"))
    ):
        ha = cast(
            pd.Series,
            pd.to_numeric(_as_1d_series(filtered_df["Hrs_RNadmin"]), errors="coerce"),
        )
        hd = cast(
            pd.Series,
            pd.to_numeric(_as_1d_series(filtered_df["Hrs_RNDON"]), errors="coerce"),
        )
        la = cast(
            pd.Series,
            pd.to_numeric(_as_1d_series(filtered_df["Hrs_LPNadmin"]), errors="coerce"),
        )
        ind = ha + hd + la
        ok_i = mdc_sum.notna() & (mdc_sum > 0) & ind.notna()
        den_i = float(cast(Any, mdc_sum[ok_i].sum()))
        avg_indirect_staffing_hprd = (
            float(cast(Any, ind[ok_i].sum()) / den_i) if den_i > 0 else None
        )
    else:
        avg_indirect_staffing_hprd = None

    census_days_missing = 0
    census_days_zero = 0
    if len(filtered_df) > 0 and "MDScensus" in filtered_df.columns:
        mdc = cast(
            pd.Series,
            pd.to_numeric(_as_1d_series(filtered_df["MDScensus"]), errors="coerce"),
        )
        census_days_missing = int(mdc.isna().sum())
        census_days_zero = int((mdc.notna() & (mdc == 0)).sum())
    hprd_q = (
        _hprd_quality_block(filtered_df["MDScensus"], len(filtered_df))
        if len(filtered_df)
        else _hprd_quality_block(pd.Series(dtype=float), 0)
    )
    match_quarter_param = _provider_match_quarter_param_from_pbj_df(cast(pd.DataFrame, filtered_df))
    state_peer_state = (
        str(filtered_df["STATE"].iloc[0]).strip().upper()
        if "STATE" in filtered_df.columns and len(filtered_df) > 0
        else ""
    )
    state_peer_bundle = _state_peer_hprd_percentiles_bundle(state_peer_state, _ein_active_ccn(), match_quarter_param)
    case_mix_geo_bundle = _case_mix_geo_bundle(state_peer_state, match_quarter_param)
    _fcb = _provider_nursing_cmi_for_matched_quarter(_ein_active_ccn(), match_quarter_param)
    if _fcb.get("available"):
        case_mix_geo_bundle["facility"] = {
            "available": True,
            "quarter": _fcb.get("quarter"),
            "nursing_case_mix_index": _fcb.get("nursing_case_mix_index"),
            "nursing_case_mix_index_ratio": _fcb.get("nursing_case_mix_index_ratio"),
            **({"cmi_source_quarter": _fcb["cmi_source_quarter"]} if _fcb.get("cmi_source_quarter") else {}),
        }
    else:
        case_mix_geo_bundle["facility"] = {"available": False}
    _geo_county_name = ""
    if "COUNTY_NAME" in filtered_df.columns and len(filtered_df) > 0:
        _geo_county_name = str(filtered_df["COUNTY_NAME"].iloc[0] or "").strip()
    _enrich_case_mix_hprd_on_geo_bundle(
        case_mix_geo_bundle,
        state_peer_state,
        match_quarter_param,
        county_name=_geo_county_name,
        provnum=_ein_active_ccn(),
    )

    summary = {
        'total_days': len(filtered_df),
        'census_quality': {
            'days_missing_census': census_days_missing,
            'days_zero_census': census_days_zero,
        },
        'hprd_quality': hprd_q,
        'avg_census': round_financial(
            float(cast(Any, _as_1d_series(filtered_df["MDScensus"]).mean())) if len(filtered_df) > 0 else 0,
            1,
        ),
        'min_census': round_financial(
            float(cast(Any, _as_1d_series(filtered_df["MDScensus"]).min())) if len(filtered_df) > 0 else 0,
            1,
        ),
        'max_census': round_financial(
            float(cast(Any, _as_1d_series(filtered_df["MDScensus"]).max())) if len(filtered_df) > 0 else 0,
            1,
        ),
        'avg_rn_hprd': None if avg_rn_hprd_weighted is None else round_financial(avg_rn_hprd_weighted, 2),
        'avg_total_rn_hprd': None if avg_total_rn_hprd_weighted is None else round_financial(avg_total_rn_hprd_weighted, 2),
        'avg_lpn_hprd': None if avg_lpn_hprd_weighted is None else round_financial(avg_lpn_hprd_weighted, 2),
        'avg_cna_hprd': None if avg_cna_hprd_weighted is None else round_financial(avg_cna_hprd_weighted, 2),
        'avg_nurse_staff_hprd_excl_admin': None
        if avg_nurse_staff_hprd_excl_admin_weighted is None
        else round_financial(avg_nurse_staff_hprd_excl_admin_weighted, 2),
        'avg_total_hprd': None if avg_total_hprd_weighted is None else round_financial(avg_total_hprd_weighted, 2),
        'avg_total_nurse_aide_hprd': None
        if avg_total_nurse_aide_hprd_weighted is None
        else round_financial(avg_total_nurse_aide_hprd_weighted, 2),
        'avg_indirect_staffing_hprd': None
        if avg_indirect_staffing_hprd is None
        else round_financial(avg_indirect_staffing_hprd, 2),
        'avg_rn_contract_pct': _mean_or_none(filtered_df['RN_Contract_Pct']) if len(filtered_df) > 0 else None,
        'avg_lpn_contract_pct': _mean_or_none(filtered_df['LPN_Contract_Pct']) if len(filtered_df) > 0 else None,
        'avg_cna_contract_pct': _mean_or_none(filtered_df['CNA_Contract_Pct']) if len(filtered_df) > 0 else None,
        'avg_total_contract_pct': _weighted_summary_total_contract_pct(cast(pd.DataFrame, filtered_df)),
        'total_rn_hours': float(filtered_df['Hrs_RN'].sum()) if len(filtered_df) > 0 else 0,
        # RN Sub 8 metrics - days with less than 8 hours of RN staffing
        'total_rn_sub8': int((filtered_df['Total_RN_Hours'] < 8).sum()) if len(filtered_df) > 0 else 0,
        'direct_rn_sub8': int((filtered_df['Hrs_RN'] < 8).sum()) if len(filtered_df) > 0 else 0,
        'total_lpn_hours': float(filtered_df['Hrs_LPN'].sum()) if len(filtered_df) > 0 else 0,
        'total_cna_hours': float(filtered_df['Hrs_CNA'].sum()) if len(filtered_df) > 0 else 0,
        'holiday_days': len(filtered_df[filtered_df['IsHoliday'] == True]),
        # Additional summary statistics
        'avg_rn_admin_hours': float(filtered_df['Hrs_RNadmin'].mean()) if len(filtered_df) > 0 else 0,
        'avg_rn_don_hours': float(filtered_df['Hrs_RNDON'].mean()) if len(filtered_df) > 0 else 0,
        'avg_lpn_admin_hours': float(filtered_df['Hrs_LPNadmin'].mean()) if len(filtered_df) > 0 else 0,
        'avg_na_trainee_hours': float(filtered_df['Hrs_NAtrn'].mean()) if len(filtered_df) > 0 else 0,
        'avg_med_aide_hours': float(filtered_df['Hrs_MedAide'].mean()) if len(filtered_df) > 0 else 0,
        'avg_rn_contract_hours': float(filtered_df['Hrs_RN_ctr'].mean()) if len(filtered_df) > 0 else 0,
        'avg_lpn_contract_hours': float(filtered_df['Hrs_LPN_ctr'].mean()) if len(filtered_df) > 0 else 0,
        'avg_cna_contract_hours': float(filtered_df['Hrs_CNA_ctr'].mean()) if len(filtered_df) > 0 else 0,
        'total_rn_admin_hours': float(filtered_df['Hrs_RNadmin'].sum()) if len(filtered_df) > 0 else 0,
        'total_rn_don_hours': float(filtered_df['Hrs_RNDON'].sum()) if len(filtered_df) > 0 else 0,
        'total_lpn_admin_hours': float(filtered_df['Hrs_LPNadmin'].sum()) if len(filtered_df) > 0 else 0,
        'total_na_trainee_hours': float(filtered_df['Hrs_NAtrn'].sum()) if len(filtered_df) > 0 else 0,
        'total_med_aide_hours': float(filtered_df['Hrs_MedAide'].sum()) if len(filtered_df) > 0 else 0,
        'total_rn_contract_hours': float(filtered_df['Hrs_RN_ctr'].sum()) if len(filtered_df) > 0 else 0,
        'total_lpn_contract_hours': float(filtered_df['Hrs_LPN_ctr'].sum()) if len(filtered_df) > 0 else 0,
        'total_cna_contract_hours': float(filtered_df['Hrs_CNA_ctr'].sum()) if len(filtered_df) > 0 else 0,
        # Align profile CMS ratings with the same filtered PBJ span as this summary (not client table state).
        "provider_info_match_quarter": match_quarter_param,
        "state_peer_bundle": state_peer_bundle,
        "case_mix_geo_bundle": case_mix_geo_bundle,
        'pbj_per_employee_rule_note': (
            "Facility PBJ daily totals in this dashboard are not employee-level XML; the CMS v4.10.0 "
            "rule (no more than 22.5 hours per employee system ID per workday across all job titles) "
            "is evaluated on EIN Employee Detail rows when that extract is loaded."
        ),
    }
    ein_dq: dict[str, Any] = {"available": False}
    try:
        _ensure_ein_employee_detail_loaded()
    except Exception:
        pass
    if ein_employee_detail_df is not None and not ein_employee_detail_df.empty and len(filtered_df) > 0:
        wd_lo, wd_hi = _pbj_filtered_workdate_yyyymmdd_bounds(cast(pd.DataFrame, filtered_df))
        ein_dq = ein_employee_day_cms_hours_cap_summary(
            cast(pd.DataFrame, ein_employee_detail_df),
            workdate_lo_yyyymmdd=wd_lo,
            workdate_hi_yyyymmdd=wd_hi,
        )
    summary["ein_data_quality"] = ein_dq
    

    return summary


def _charts_build_payload_dict(
    filtered_df: pd.DataFrame,
    *,
    filter_label_start_date: str | None,
    filter_label_end_date: str | None,
    filter_quarter: str,
    filter_day_of_week: str,
    filter_holidays_only: bool,
    dow_calendar_year_param: str | None,
    hprd_view: str,
    hours_view: str,
    census_view: str,
    contract_view: str,
    composition_view: str,
) -> dict[str, Any]:
    # Sort by date
    filtered_df = filtered_df.sort_values('WorkDate')
    
    if len(filtered_df) == 0:
        print("[charts] No rows after filters (check quarter/date params).")

    # Helper function to aggregate data by view mode
    def aggregate_by_view_mode(df, view_mode, date_col='WorkDate'):
        if view_mode == 'daily':
            return df
        elif view_mode == 'month':
            # Aggregate by month
            df_copy = df.copy()
            df_copy['year_month'] = df_copy[date_col].dt.to_period('M')
            agg_dict = {
                'Total_Nurse_HPRD': 'mean',
                'Total_RN_HPRD': 'mean', 
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean',
                'Total_Staff_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_Hours': 'sum',
                'Total_LPN_Hours': 'sum',
                'Total_Nurse_Aide_Hours': 'sum',
                'Total_Staff_Hours': 'sum',
                'Nurse_Staff_Hours_Excl_Admin': 'sum',
                'MDScensus': 'mean',
                'RN_Contract_Pct': 'mean',
                'LPN_Contract_Pct': 'mean',
                'CNA_Contract_Pct': 'mean',
                'Total_LPN_Contract_Pct': 'mean',
                'Nurse_Aide_Contract_Pct': 'mean',
                'Total_Contract_Pct': 'mean',
                'IsHoliday': 'any'
            }
            # Add base hours columns if they exist (needed for HPRD calculations)
            if 'Hrs_RN' in df_copy.columns:
                agg_dict['Hrs_RN'] = 'sum'
            if 'Hrs_LPN' in df_copy.columns:
                agg_dict['Hrs_LPN'] = 'sum'
            if 'Hrs_CNA' in df_copy.columns:
                agg_dict['Hrs_CNA'] = 'sum'
            for _role_col in ('Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_LPNadmin', 'Hrs_MedAide', 'Hrs_NAtrn'):
                if _role_col in df_copy.columns:
                    agg_dict[_role_col] = 'sum'
            # Add RN_HPRD, LPN_HPRD, CNA_HPRD if they exist
            if 'RN_HPRD' in df_copy.columns:
                agg_dict['RN_HPRD'] = 'mean'
            if 'LPN_HPRD' in df_copy.columns:
                agg_dict['LPN_HPRD'] = 'mean'
            if 'CNA_HPRD' in df_copy.columns:
                agg_dict['CNA_HPRD'] = 'mean'
            aggregated = df_copy.groupby('year_month').agg(agg_dict).reset_index()
            # Recalculate HPRD from aggregated hours if needed
            if 'RN_HPRD' not in aggregated.columns and 'Hrs_RN' in aggregated.columns and 'MDScensus' in aggregated.columns:
                aggregated["RN_HPRD"] = _round_fin_series(
                    _divide_hprd(aggregated["Hrs_RN"], aggregated["MDScensus"]), 2
                )
            if 'LPN_HPRD' not in aggregated.columns and 'Hrs_LPN' in aggregated.columns and 'MDScensus' in aggregated.columns:
                aggregated["LPN_HPRD"] = _round_fin_series(
                    _divide_hprd(aggregated["Hrs_LPN"], aggregated["MDScensus"]), 2
                )
            if 'CNA_HPRD' not in aggregated.columns and 'Hrs_CNA' in aggregated.columns and 'MDScensus' in aggregated.columns:
                aggregated["CNA_HPRD"] = _round_fin_series(
                    _divide_hprd(aggregated["Hrs_CNA"], aggregated["MDScensus"]), 2
                )
            aggregated[date_col] = aggregated['year_month'].dt.to_timestamp()
            return aggregated
        elif view_mode == 'quarter':
            # Aggregate by quarter
            df_copy = df.copy()
            df_copy['year_quarter'] = df_copy[date_col].dt.to_period('Q')
            agg_dict = {
                'Total_Nurse_HPRD': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean', 
                'Total_Nurse_Aide_HPRD': 'mean',
                'Total_Staff_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_Hours': 'sum',
                'Total_LPN_Hours': 'sum',
                'Total_Nurse_Aide_Hours': 'sum',
                'Total_Staff_Hours': 'sum',
                'Nurse_Staff_Hours_Excl_Admin': 'sum',
                'MDScensus': 'mean',
                'RN_Contract_Pct': 'mean',
                'LPN_Contract_Pct': 'mean',
                'CNA_Contract_Pct': 'mean',
                'Total_LPN_Contract_Pct': 'mean',
                'Nurse_Aide_Contract_Pct': 'mean',
                'Total_Contract_Pct': 'mean',
                'IsHoliday': 'any'
            }
            # Add base hours columns if they exist (needed for HPRD calculations)
            if 'Hrs_RN' in df_copy.columns:
                agg_dict['Hrs_RN'] = 'sum'
            if 'Hrs_LPN' in df_copy.columns:
                agg_dict['Hrs_LPN'] = 'sum'
            if 'Hrs_CNA' in df_copy.columns:
                agg_dict['Hrs_CNA'] = 'sum'
            for _role_col in ('Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_LPNadmin', 'Hrs_MedAide', 'Hrs_NAtrn'):
                if _role_col in df_copy.columns:
                    agg_dict[_role_col] = 'sum'
            # Add RN_HPRD, LPN_HPRD, CNA_HPRD if they exist
            if 'RN_HPRD' in df_copy.columns:
                agg_dict['RN_HPRD'] = 'mean'
            if 'LPN_HPRD' in df_copy.columns:
                agg_dict['LPN_HPRD'] = 'mean'
            if 'CNA_HPRD' in df_copy.columns:
                agg_dict['CNA_HPRD'] = 'mean'
            aggregated = df_copy.groupby('year_quarter').agg(agg_dict).reset_index()
            # Recalculate HPRD from aggregated hours if needed
            if 'RN_HPRD' not in aggregated.columns and 'Hrs_RN' in aggregated.columns and 'MDScensus' in aggregated.columns:
                aggregated["RN_HPRD"] = _round_fin_series(
                    _divide_hprd(aggregated["Hrs_RN"], aggregated["MDScensus"]), 2
                )
            if 'LPN_HPRD' not in aggregated.columns and 'Hrs_LPN' in aggregated.columns and 'MDScensus' in aggregated.columns:
                aggregated["LPN_HPRD"] = _round_fin_series(
                    _divide_hprd(aggregated["Hrs_LPN"], aggregated["MDScensus"]), 2
                )
            if 'CNA_HPRD' not in aggregated.columns and 'Hrs_CNA' in aggregated.columns and 'MDScensus' in aggregated.columns:
                aggregated["CNA_HPRD"] = _round_fin_series(
                    _divide_hprd(aggregated["Hrs_CNA"], aggregated["MDScensus"]), 2
                )
            aggregated[date_col] = aggregated['year_quarter'].dt.to_timestamp()
            return aggregated
        elif view_mode == 'year':
            # Aggregate by year
            df_copy = df.copy()
            df_copy['year'] = df_copy[date_col].dt.year
            agg_dict = {
                'Total_Nurse_HPRD': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean', 
                'Total_Staff_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_Hours': 'sum',
                'Total_LPN_Hours': 'sum',
                'Total_Nurse_Aide_Hours': 'sum',
                'Total_Staff_Hours': 'sum',
                'Nurse_Staff_Hours_Excl_Admin': 'sum',
                'MDScensus': 'mean',
                'RN_Contract_Pct': 'mean',
                'LPN_Contract_Pct': 'mean',
                'CNA_Contract_Pct': 'mean',
                'Total_LPN_Contract_Pct': 'mean',
                'Nurse_Aide_Contract_Pct': 'mean',
                'Total_Contract_Pct': 'mean',
                'IsHoliday': 'any'
            }
            # Add base hours columns if they exist (needed for HPRD calculations)
            if 'Hrs_RN' in df_copy.columns:
                agg_dict['Hrs_RN'] = 'sum'
            if 'Hrs_LPN' in df_copy.columns:
                agg_dict['Hrs_LPN'] = 'sum'
            if 'Hrs_CNA' in df_copy.columns:
                agg_dict['Hrs_CNA'] = 'sum'
            for _role_col in ('Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_LPNadmin', 'Hrs_MedAide', 'Hrs_NAtrn'):
                if _role_col in df_copy.columns:
                    agg_dict[_role_col] = 'sum'
            # Add RN_HPRD, LPN_HPRD, CNA_HPRD if they exist
            if 'RN_HPRD' in df_copy.columns:
                agg_dict['RN_HPRD'] = 'mean'
            if 'LPN_HPRD' in df_copy.columns:
                agg_dict['LPN_HPRD'] = 'mean'
            if 'CNA_HPRD' in df_copy.columns:
                agg_dict['CNA_HPRD'] = 'mean'
            aggregated = df_copy.groupby('year').agg(agg_dict).reset_index()
            # Recalculate HPRD from aggregated hours if needed
            if 'RN_HPRD' not in aggregated.columns and 'Hrs_RN' in aggregated.columns and 'MDScensus' in aggregated.columns:
                aggregated["RN_HPRD"] = _round_fin_series(
                    _divide_hprd(aggregated["Hrs_RN"], aggregated["MDScensus"]), 2
                )
            if 'LPN_HPRD' not in aggregated.columns and 'Hrs_LPN' in aggregated.columns and 'MDScensus' in aggregated.columns:
                aggregated["LPN_HPRD"] = _round_fin_series(
                    _divide_hprd(aggregated["Hrs_LPN"], aggregated["MDScensus"]), 2
                )
            if 'CNA_HPRD' not in aggregated.columns and 'Hrs_CNA' in aggregated.columns and 'MDScensus' in aggregated.columns:
                aggregated["CNA_HPRD"] = _round_fin_series(
                    _divide_hprd(aggregated["Hrs_CNA"], aggregated["MDScensus"]), 2
                )
            aggregated[date_col] = pd.to_datetime(aggregated['year'], format='%Y')
            return aggregated
        else:
            return df

    _COMPOSITION_ROLE_HOUR_COLS = (
        'Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_RN', 'Hrs_LPNadmin', 'Hrs_LPN',
        'Hrs_CNA', 'Hrs_MedAide', 'Hrs_NAtrn',
    )
    _COMPOSITION_CONTRACT_HOUR_COLS = (
        'Hrs_RNDON_ctr', 'Hrs_RNadmin_ctr', 'Hrs_RN_ctr', 'Hrs_LPNadmin_ctr',
        'Hrs_LPN_ctr', 'Hrs_CNA_ctr', 'Hrs_MedAide_ctr', 'Hrs_NAtrn_ctr',
    )
    _COMPOSITION_DIRECT_HOUR_COLS = ('Hrs_RN', 'Hrs_LPN', 'Hrs_CNA', 'Hrs_MedAide', 'Hrs_NAtrn')

    def aggregate_composition_by_view_mode(df, view_mode, date_col='WorkDate'):
        """Average daily hours/HPRD per period — never sum days into one bucket."""
        if view_mode == 'daily' or len(df) == 0:
            return df

        df_copy = df.copy()
        census = (
            pd.to_numeric(df_copy['MDScensus'], errors='coerce')
            if 'MDScensus' in df_copy.columns
            else pd.Series(np.nan, index=df_copy.index, dtype='float64')
        )
        agg_dict: dict[str, str] = {'MDScensus': 'mean'}

        for col in _COMPOSITION_ROLE_HOUR_COLS + _COMPOSITION_CONTRACT_HOUR_COLS:
            if col not in df_copy.columns:
                continue
            hrs = pd.to_numeric(df_copy[col], errors='coerce')
            df_copy[f'__comp_hprd_{col}'] = _divide_hprd(hrs, census)
            agg_dict[col] = 'mean'
            agg_dict[f'__comp_hprd_{col}'] = 'mean'

        direct_hrs = pd.Series(0.0, index=df_copy.index, dtype='float64')
        has_direct = False
        for col in _COMPOSITION_DIRECT_HOUR_COLS:
            if col in df_copy.columns:
                direct_hrs = direct_hrs.add(
                    pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0), fill_value=0.0
                )
                has_direct = True
        if has_direct:
            df_copy['__comp_hours_direct'] = direct_hrs
            df_copy['__comp_hprd_direct'] = _divide_hprd(direct_hrs, census)
            agg_dict['__comp_hours_direct'] = 'mean'
            agg_dict['__comp_hprd_direct'] = 'mean'

        contract_hrs = pd.Series(0.0, index=df_copy.index, dtype='float64')
        has_contract = False
        for col in _COMPOSITION_CONTRACT_HOUR_COLS:
            if col in df_copy.columns:
                contract_hrs = contract_hrs.add(
                    pd.to_numeric(df_copy[col], errors='coerce').fillna(0.0), fill_value=0.0
                )
                has_contract = True
        if has_contract:
            df_copy['__comp_hours_contract'] = contract_hrs
            df_copy['__comp_hprd_contract'] = _divide_hprd(contract_hrs, census)
            agg_dict['__comp_hours_contract'] = 'mean'
            agg_dict['__comp_hprd_contract'] = 'mean'

        if view_mode == 'month':
            df_copy['_period'] = df_copy[date_col].dt.to_period('M')
        elif view_mode == 'quarter':
            df_copy['_period'] = df_copy[date_col].dt.to_period('Q')
        elif view_mode == 'year':
            df_copy['_period'] = df_copy[date_col].dt.year
        else:
            return df

        aggregated = df_copy.groupby('_period', as_index=False).agg(agg_dict)
        if view_mode == 'month':
            aggregated[date_col] = aggregated['_period'].dt.to_timestamp()
        elif view_mode == 'quarter':
            aggregated[date_col] = aggregated['_period'].dt.to_timestamp()
        else:
            aggregated[date_col] = pd.to_datetime(aggregated['_period'], format='%Y')
        return aggregated.drop(columns=['_period'], errors='ignore')
    
    # Check if we have any data after filtering
    if len(filtered_df) == 0:
        return {
            'charts': {},
            'filter_info': 'No data found for the selected filters',
            'error': 'No data available for the selected date range and filters'
        }
    
    # Check for required columns
    required_columns = ['WorkDate', 'Total_RN_HPRD', 'Total_LPN_HPRD', 'Total_Nurse_Aide_HPRD', 
                       'Total_Staff_HPRD', 'Nurse_Staff_HPRD_Excl_Admin', 'Total_RN_Hours', 'Total_LPN_Hours', 'Total_Nurse_Aide_Hours',
                       'Total_Staff_Hours', 'Nurse_Staff_Hours_Excl_Admin', 'MDScensus', 'RN_Contract_Pct', 'LPN_Contract_Pct', 'CNA_Contract_Pct', 'IsHoliday']
    
    missing_columns = [col for col in required_columns if col not in filtered_df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        print(f"Available columns: {list(filtered_df.columns)}")
        print(f"Data shape: {filtered_df.shape}")
        print(f"Columns with 'Total': {[col for col in filtered_df.columns if 'Total' in col]}")
        return {
            'charts': {},
            'filter_info': 'Data structure error',
            'error': f'Missing required columns: {", ".join(missing_columns)}'
        }
    
    charts = {}

    # Optional calendar-year slice for day-of-week charts only (full filtered_df for all other charts).
    dow_calendar_year_raw = (dow_calendar_year_param or "all").strip()
    dow_df = filtered_df
    if dow_calendar_year_raw and dow_calendar_year_raw.lower() != "all":
        try:
            _dow_y = int(dow_calendar_year_raw)
            if 2017 <= _dow_y <= 2100:
                dow_df = filtered_df[filtered_df["WorkDate"].dt.year == _dow_y].copy()
        except (TypeError, ValueError):
            dow_df = filtered_df
    if len(dow_df) == 0:
        dow_df = filtered_df.copy()
    
    # Apply aggregation based on view modes
    hprd_df = aggregate_by_view_mode(filtered_df, hprd_view)
    hours_df = aggregate_by_view_mode(filtered_df, hours_view)
    census_df = aggregate_by_view_mode(filtered_df, census_view)
    contract_df = aggregate_by_view_mode(filtered_df, contract_view)
    composition_df = aggregate_composition_by_view_mode(filtered_df, composition_view)
    # Do not invent direct RN hours: ``hours_trend`` uses ``_plotly_y_nullable_optional`` for Hrs_RN
    # so a missing column yields nulls, not a fabricated fraction of Total_RN_Hours.
    
    # Helper function to format dates based on view mode
    def format_dates_for_view_mode(df, view_mode):
        if view_mode == 'daily':
            return df['WorkDate'].dt.strftime('%m-%d-%Y').tolist()
        elif view_mode == 'month':
            return df['WorkDate'].dt.strftime('%b %Y').tolist()
        elif view_mode == 'quarter':
            # Convert to quarter format like "Q3 2019"
            quarters = []
            for date in df['WorkDate']:
                year = date.year
                month = date.month
                if month <= 3:
                    quarter = "Q1"
                elif month <= 6:
                    quarter = "Q2"
                elif month <= 9:
                    quarter = "Q3"
                else:
                    quarter = "Q4"
                quarters.append(f"{quarter} {year}")
            return quarters
        elif view_mode == 'year':
            return df['WorkDate'].dt.strftime('%Y').tolist()
        else:
            return df['WorkDate'].dt.strftime('%m-%d-%Y').tolist()
    
    # Holiday markers on daily charts: never a legend entry; skip very wide spans to avoid clutter.
    _pbj_chart_holiday_max_days = 120
    try:
        if "WorkDate" in filtered_df.columns:
            _wd_hol = pd.to_datetime(filtered_df["WorkDate"], errors="coerce")
            _span_days_for_chart_holidays = int(_wd_hol.dt.normalize().nunique()) if _wd_hol.notna().any() else 0
        else:
            _span_days_for_chart_holidays = int(len(filtered_df))
    except Exception:
        _span_days_for_chart_holidays = int(len(filtered_df))
    _chart_holidays_ok = 0 < _span_days_for_chart_holidays <= _pbj_chart_holiday_max_days
    _holiday_trace_style: dict[str, Any] = {
        "type": "scatter",
        "mode": "markers",
        "name": "Holidays",
        "marker": {"color": "red", "size": 9, "symbol": "star"},
        "showlegend": False,
        "legendgroup": "holidays",
        "hovertemplate": "<b>%{x}</b><br>%{y:.2f}<extra>Federal holiday</extra>",
    }

    def _holiday_y_coords(h_df: pd.DataFrame, value_col: str) -> list[Any]:
        if h_df.empty:
            return []
        if value_col not in h_df.columns:
            return [None] * len(h_df)
        ser = pd.to_numeric(h_df[value_col], errors="coerce")
        return [None if pd.isna(v) else float(v) for v in ser.tolist()]

    if "IsHoliday" in filtered_df.columns:
        _holiday_df = cast(pd.DataFrame, filtered_df.loc[filtered_df["IsHoliday"].eq(True)].copy())
    else:
        _holiday_df = pd.DataFrame()

    # Nursing staff HPRD trend: do not attach federal-holiday marker trace (null HPRD on
    # holidays plots at y=0 in Plotly and distorts autorange / axis).

    hours_holiday_markers = []
    if hours_view == "daily" and _chart_holidays_ok and not _holiday_df.empty:
        hours_holiday_markers = [
            {
                **_holiday_trace_style,
                "x": format_dates_for_view_mode(_holiday_df, "daily"),
                "y": _holiday_y_coords(_holiday_df, "Total_Staff_Hours"),
            }
        ]

    census_holiday_markers = []
    if census_view == "daily" and _chart_holidays_ok and not _holiday_df.empty:
        census_holiday_markers = [
            {
                **_holiday_trace_style,
                "x": format_dates_for_view_mode(_holiday_df, "daily"),
                "y": _holiday_y_coords(_holiday_df, "MDScensus"),
            }
        ]

    contract_holiday_markers = []
    if contract_view == "daily" and _chart_holidays_ok and not _holiday_df.empty:
        contract_holiday_markers = [
            {
                **_holiday_trace_style,
                "x": format_dates_for_view_mode(_holiday_df, "daily"),
                "y": _holiday_y_coords(_holiday_df, "Total_Contract_Pct"),
            }
        ]
    
    # Get state standard for reference line on Total HPRD chart
    state_standard_lines = []
    try:
        facility_state = filtered_df['STATE'].iloc[0] if 'STATE' in filtered_df.columns and len(filtered_df) > 0 else None
        if facility_state and macpac_standards_df is not None and len(macpac_standards_df) > 0:
            # State abbreviation to full name mapping
            state_abbrev_to_name = {
                'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
                'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
                'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
                'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
                'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
                'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
                'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
                'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
                'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
                'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
                'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
            }
            
            state_name = facility_state
            if facility_state.upper() in state_abbrev_to_name:
                state_name = state_abbrev_to_name[facility_state.upper()]
            
            state_standard = macpac_standards_df[macpac_standards_df['State'] == state_name]
            if len(state_standard) == 0:
                state_standard = macpac_standards_df[macpac_standards_df['State'].str.upper() == state_name.upper()]
            
            if len(state_standard) > 0:
                state_standard = state_standard.iloc[0]
                # Skip federal minimum states
                if not state_standard.get('Is_Federal_Minimum', False):
                    dates = format_dates_for_view_mode(hprd_df, hprd_view)
                    if state_standard['Value_Type'] == 'range':
                        # One line at the upper bound (e.g. 2.31), labeled as range (e.g. "WY min (1.56—2.31)")
                        min_val = state_standard['Min_Staffing']
                        max_val = state_standard['Max_Staffing']
                        _hv = (
                            f"<b>{facility_state} staffing range (MACPAC)</b><br>"
                            f"Range: {min_val}–{max_val} HPRD<br>"
                            f"Upper bound (line): %{{y:.2f}} HPRD<extra></extra>"
                        )
                        state_standard_lines.append({
                            'x': dates,
                            'y': [float(max_val)] * len(dates),
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': f"{facility_state} min. ({min_val}—{max_val})",
                            'line': {'color': '#ffc107', 'width': 2, 'dash': 'dash'},
                            'hovertemplate': _hv,
                        })
                    else:
                        # Single value
                        _ms = float(state_standard['Min_Staffing'])
                        _hv = (
                            f"<b>{facility_state} minimum (MACPAC)</b><br>"
                            f"~{_ms:.2f} HPRD reference<br>"
                            f"Line: %{{y:.2f}} HPRD<extra></extra>"
                        )
                        state_standard_lines.append({
                            'x': dates,
                            'y': [float(state_standard['Min_Staffing'])] * len(dates),
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': f"{facility_state} min. (~{state_standard['Min_Staffing']})",
                            'line': {'color': '#ffc107', 'width': 2, 'dash': 'dash'},
                            'hovertemplate': _hv,
                        })
    except Exception as e:
        print(f"Error adding state standard lines: {e}")
        import traceback
        traceback.print_exc()
    
    charts['hprd_trend'] = {
        'data': [
            {
                'x': format_dates_for_view_mode(hprd_df, hprd_view),
                'y': _plotly_y_nullable(hprd_df['Total_Nurse_HPRD']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total',
                'line': {'color': '#d62728', 'width': 3},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hprd_df, hprd_view),
                'y': _plotly_y_nullable(hprd_df['Nurse_Staff_HPRD_Excl_Admin']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Direct',
                'line': {'color': '#9467bd'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hprd_df, hprd_view),
                'y': _plotly_y_nullable(hprd_df['Total_RN_HPRD']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'RN (Total)',
                'line': {'color': '#1f77b4'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hprd_df, hprd_view),
                'y': _plotly_y_nullable_optional(hprd_df, 'RN_HPRD'),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'RN (excl. Admin / DON)',
                'line': {'color': '#ff7f0e', 'dash': 'dash', 'width': 2},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hprd_df, hprd_view),
                'y': _plotly_y_nullable(hprd_df['Total_LPN_HPRD']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'LPN (Total)',
                'line': {'color': '#ff7f0e'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hprd_df, hprd_view),
                'y': _plotly_y_nullable_optional(hprd_df, 'LPN_HPRD'),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'LPN',
                'line': {'color': '#ffbb78', 'dash': 'dash', 'width': 2},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hprd_df, hprd_view),
                'y': _plotly_y_nullable(hprd_df['Total_Nurse_Aide_HPRD']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Nurse Aide',
                'line': {'color': '#2ca02c'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
            }
        ] + state_standard_lines,
        'layout': {
            'title': {
                'text': 'Daily HPRD Trends',
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'nticks': 10,
                'tickangle': -45
            },
            'yaxis': {'title': 'HPRD'},
            'height': 450,
            'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
        }
    }
    
    # Day of week comparison
    dow_summary = dow_df.groupby('DayOfWeek').agg({
        'Total_RN_HPRD': 'mean',
        'RN_HPRD': 'mean',
        'Total_LPN_HPRD': 'mean',
        'LPN_HPRD': 'mean',
        'Total_Nurse_Aide_HPRD': 'mean',
        'Nurse_Staff_HPRD_Excl_Admin': 'mean',
        'Total_Nurse_HPRD': 'mean'
    }).reset_index()
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_summary['DayOfWeek'] = pd.Categorical(dow_summary['DayOfWeek'], categories=day_order, ordered=True)
    dow_summary = dow_summary.sort_values('DayOfWeek')
    
    dow_data = [
        {
            'x': dow_summary['DayOfWeek'].tolist(),
            'y': _plotly_y_nullable(dow_summary['Total_Nurse_HPRD']),
            'type': 'bar',
            'name': 'Total',
            'marker': {'color': '#d62728'},
            'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
        },
        {
            'x': dow_summary['DayOfWeek'].tolist(),
            'y': _plotly_y_nullable(dow_summary['Nurse_Staff_HPRD_Excl_Admin']),
            'type': 'bar',
            'name': 'Direct',
            'marker': {'color': '#9467bd'},
            'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
        },
        {
            'x': dow_summary['DayOfWeek'].tolist(),
            'y': _plotly_y_nullable(dow_summary['Total_RN_HPRD']),
            'type': 'bar',
            'name': 'RN (Total)',
            'marker': {'color': '#1f77b4'},
            'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
        },
        {
            'x': dow_summary['DayOfWeek'].tolist(),
            'y': _plotly_y_nullable_optional(dow_summary, 'RN_HPRD'),
            'type': 'bar',
            'name': 'RN',
            'marker': {'color': '#8bb8e8'},
            'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
        },
        {
            'x': dow_summary['DayOfWeek'].tolist(),
            'y': _plotly_y_nullable(dow_summary['Total_LPN_HPRD']),
            'type': 'bar',
            'name': 'LPN (Total)',
            'marker': {'color': '#ff7f0e'},
            'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
        },
        {
            'x': dow_summary['DayOfWeek'].tolist(),
            'y': _plotly_y_nullable_optional(dow_summary, 'LPN_HPRD'),
            'type': 'bar',
            'name': 'LPN',
            'marker': {'color': '#ffbb78'},
            'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
        },
        {
            'x': dow_summary['DayOfWeek'].tolist(),
            'y': _plotly_y_nullable(dow_summary['Total_Nurse_Aide_HPRD']),
            'type': 'bar',
            'name': 'Nurse Aide',
            'marker': {'color': '#2ca02c'},
            'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
        }
    ]
    
    # Add state standard line to day of week chart
    try:
        facility_state = filtered_df['STATE'].iloc[0] if 'STATE' in filtered_df.columns and len(filtered_df) > 0 else None
        if facility_state and macpac_standards_df is not None and len(macpac_standards_df) > 0:
            # State abbreviation to full name mapping
            state_abbrev_to_name = {
                'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
                'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
                'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
                'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
                'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
                'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
                'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
                'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
                'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
                'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
                'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
            }
            
            state_name = facility_state
            if facility_state.upper() in state_abbrev_to_name:
                state_name = state_abbrev_to_name[facility_state.upper()]
            
            state_standard = macpac_standards_df[macpac_standards_df['State'] == state_name]
            if len(state_standard) == 0:
                state_standard = macpac_standards_df[macpac_standards_df['State'].str.upper() == state_name.upper()]
            
            if len(state_standard) > 0:
                state_standard = state_standard.iloc[0]
                # Skip federal minimum states
                if not state_standard.get('Is_Federal_Minimum', False):
                    days_list = dow_summary['DayOfWeek'].tolist()
                    if state_standard['Value_Type'] == 'range':
                        # One line at the upper bound, labeled as range (e.g. "WY min (1.56—2.31)")
                        min_val = state_standard['Min_Staffing']
                        max_val = state_standard['Max_Staffing']
                        dow_data.append({
                            'x': days_list,
                            'y': [float(max_val)] * len(days_list),
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': f"{facility_state} min. ({min_val}—{max_val})",
                            'line': {'color': '#ffc107', 'width': 2, 'dash': 'dash'},
                            'hoverinfo': 'name+y'
                        })
                    else:
                        # Single threshold
                        dow_data.append({
                            'x': days_list,
                            'y': [float(state_standard['Min_Staffing'])] * len(days_list),
                            'type': 'scatter',
                            'mode': 'lines',
                            'name': f"{facility_state} min. (~{state_standard['Min_Staffing']})",
                            'line': {'color': '#ffc107', 'width': 2, 'dash': 'dash'},
                            'hoverinfo': 'name+y'
                        })
    except Exception as e:
        print(f"Error adding state standard to day of week chart: {e}")

    # Weekday vs weekend: pool all Mon–Fri rows vs Sat–Sun rows (not the mean of daily means)
    fp_dow = dow_df.copy()
    if "DayOfWeekNum" not in fp_dow.columns:
        fp_dow["DayOfWeekNum"] = pd.to_datetime(fp_dow["WorkDate"], errors="coerce").dt.dayofweek
    fp_dow = fp_dow.dropna(subset=["DayOfWeekNum"])
    fp_dow["WeekPart"] = np.where(fp_dow["DayOfWeekNum"] < 5, "Weekday (Mon–Fri)", "Weekend (Sat–Sun)")
    weekpart_order = ["Weekday (Mon–Fri)", "Weekend (Sat–Sun)"]
    wp_summary = fp_dow.groupby("WeekPart").agg(
        {
            "Total_RN_HPRD": "mean",
            "RN_HPRD": "mean",
            "Total_LPN_HPRD": "mean",
            "LPN_HPRD": "mean",
            "Total_Nurse_Aide_HPRD": "mean",
            "Nurse_Staff_HPRD_Excl_Admin": "mean",
            "Total_Nurse_HPRD": "mean",
        }
    ).reset_index()
    wp_summary["WeekPart"] = pd.Categorical(wp_summary["WeekPart"], categories=weekpart_order, ordered=True)
    wp_summary = wp_summary.sort_values("WeekPart")

    # Weekpart chart: x = each staffing metric; grouped bars = weekday mean vs weekend mean (same metric).
    _wp_lbl_wd = "Weekday (Mon–Fri)"
    _wp_lbl_we = "Weekend (Sat–Sun)"

    def _wp_metric_value(column: str, weekpart_label: str) -> float:
        row = wp_summary[wp_summary["WeekPart"] == weekpart_label]
        if row.empty or column not in row.columns:
            return float("nan")
        v = row.iloc[0][column]
        return float(v) if pd.notna(v) else float("nan")

    _wp_metric_defs = [
        ("Total", "Total_Nurse_HPRD", "#d62728"),
        ("Direct", "Nurse_Staff_HPRD_Excl_Admin", "#9467bd"),
        ("RN (Total)", "Total_RN_HPRD", "#1f77b4"),
        ("RN", "RN_HPRD", "#8bb8e8"),
        ("LPN (Total)", "Total_LPN_HPRD", "#ff7f0e"),
        ("LPN", "LPN_HPRD", "#ffbb78"),
        ("Nurse Aide", "Total_Nurse_Aide_HPRD", "#2ca02c"),
    ]
    _wp_x_labels = [t[0] for t in _wp_metric_defs]
    _y_wd = [_wp_metric_value(col, _wp_lbl_wd) for _lab, col, _c in _wp_metric_defs]
    _y_we = [_wp_metric_value(col, _wp_lbl_we) for _lab, col, _c in _wp_metric_defs]
    _colors_wd = [t[2] for t in _wp_metric_defs]
    # Weekend bars: same hue family, slightly lighter for contrast at a glance
    _colors_we = ["#f87171", "#c084fc", "#60a5fa", "#bfdbfe", "#fb923c", "#fde68a", "#4ade80"]

    wp_data = [
        {
            "x": _wp_x_labels,
            "y": _json_float_list(_y_wd),
            "type": "bar",
            "name": _wp_lbl_wd,
            "marker": {"color": _colors_wd},
            "hovertemplate": "<b>%{x}</b><br>Weekday: %{y:.2f} HPRD<extra></extra>",
        },
        {
            "x": _wp_x_labels,
            "y": _json_float_list(_y_we),
            "type": "bar",
            "name": _wp_lbl_we,
            "marker": {"color": _colors_we},
            "hovertemplate": "<b>%{x}</b><br>Weekend: %{y:.2f} HPRD<extra></extra>",
        },
    ]
    try:
        facility_state = filtered_df["STATE"].iloc[0] if "STATE" in filtered_df.columns and len(filtered_df) > 0 else None
        if facility_state and macpac_standards_df is not None and len(macpac_standards_df) > 0:
            state_abbrev_to_name = {
                "AL": "Alabama",
                "AK": "Alaska",
                "AZ": "Arizona",
                "AR": "Arkansas",
                "CA": "California",
                "CO": "Colorado",
                "CT": "Connecticut",
                "DE": "Delaware",
                "DC": "District of Columbia",
                "FL": "Florida",
                "GA": "Georgia",
                "HI": "Hawaii",
                "ID": "Idaho",
                "IL": "Illinois",
                "IN": "Indiana",
                "IA": "Iowa",
                "KS": "Kansas",
                "KY": "Kentucky",
                "LA": "Louisiana",
                "ME": "Maine",
                "MD": "Maryland",
                "MA": "Massachusetts",
                "MI": "Michigan",
                "MN": "Minnesota",
                "MS": "Mississippi",
                "MO": "Missouri",
                "MT": "Montana",
                "NE": "Nebraska",
                "NV": "Nevada",
                "NH": "New Hampshire",
                "NJ": "New Jersey",
                "NM": "New Mexico",
                "NY": "New York",
                "NC": "North Carolina",
                "ND": "North Dakota",
                "OH": "Ohio",
                "OK": "Oklahoma",
                "OR": "Oregon",
                "PA": "Pennsylvania",
                "RI": "Rhode Island",
                "SC": "South Carolina",
                "SD": "South Dakota",
                "TN": "Tennessee",
                "TX": "Texas",
                "UT": "Utah",
                "VT": "Vermont",
                "VA": "Virginia",
                "WA": "Washington",
                "WV": "West Virginia",
                "WI": "Wisconsin",
                "WY": "Wyoming",
            }
            state_name = facility_state
            if facility_state.upper() in state_abbrev_to_name:
                state_name = state_abbrev_to_name[facility_state.upper()]
            state_standard = macpac_standards_df[macpac_standards_df["State"] == state_name]
            if len(state_standard) == 0:
                state_standard = macpac_standards_df[macpac_standards_df["State"].str.upper() == state_name.upper()]
            if len(state_standard) > 0:
                state_standard = state_standard.iloc[0]
                if not state_standard.get("Is_Federal_Minimum", False):
                    wp_x_list = list(_wp_x_labels)
                    if wp_x_list:
                        if state_standard["Value_Type"] == "range":
                            min_val = state_standard["Min_Staffing"]
                            max_val = state_standard["Max_Staffing"]
                            wp_data.append(
                                {
                                    "x": wp_x_list,
                                    "y": [float(max_val)] * len(wp_x_list),
                                    "type": "scatter",
                                    "mode": "lines",
                                    "name": f"{facility_state} min. ({min_val}—{max_val})",
                                    "line": {"color": "#ffc107", "width": 2, "dash": "dash"},
                                    "hoverinfo": "name+y",
                                }
                            )
                        else:
                            wp_data.append(
                                {
                                    "x": wp_x_list,
                                    "y": [float(state_standard["Min_Staffing"])] * len(wp_x_list),
                                    "type": "scatter",
                                    "mode": "lines",
                                    "name": f"{facility_state} min. (~{state_standard['Min_Staffing']})",
                                    "line": {"color": "#ffc107", "width": 2, "dash": "dash"},
                                    "hoverinfo": "name+y",
                                }
                            )
    except Exception as e:
        print(f"Error adding state standard to weekday/weekend chart: {e}")

    charts["dow_comparison"] = {
        "data": dow_data,
        "layout": {
            "title": {
                "text": "Average HPRD by Day of Week",
                "x": 0.5,
                "xanchor": "center",
            },
            "xaxis": {},
            "yaxis": {"title": "HPRD"},
            "height": 450,
            "margin": {"b": 100, "l": 60, "r": 40, "t": 80},
        },
    }
    charts["dow_comparison_weekpart"] = {
        "data": wp_data,
        "layout": {
            "title": {
                "text": "Average HPRD: Weekday vs. Weekend by metric",
                "x": 0.5,
                "xanchor": "center",
            },
            "barmode": "group",
            "xaxis": {},
            "yaxis": {"title": "HPRD"},
            "height": 450,
            "margin": {"b": 100, "l": 60, "r": 40, "t": 80},
        },
    }
    if len(wp_summary) == 0:
        charts["dow_comparison_weekpart"] = copy.deepcopy(charts["dow_comparison"])
    
    # Hours trend
    charts['hours_trend'] = {
        'data': [
            {
                'x': format_dates_for_view_mode(hours_df, hours_view),
                'y': _plotly_y_nullable(hours_df['Total_Staff_Hours']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total',
                'line': {'color': '#d62728', 'width': 3},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hours_df, hours_view),
                'y': _plotly_y_nullable(hours_df['Nurse_Staff_Hours_Excl_Admin']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Direct',
                'line': {'color': '#9467bd'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hours_df, hours_view),
                'y': _plotly_y_nullable(hours_df['Total_RN_Hours']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'RN (Total)',
                'line': {'color': '#1f77b4'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hours_df, hours_view),
                'y': _plotly_y_nullable_optional(hours_df, 'Hrs_RN'),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'RN',
                'line': {'color': '#8bb8e8'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hours_df, hours_view),
                'y': _plotly_y_nullable(hours_df['Total_LPN_Hours']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'LPN (Total)',
                'line': {'color': '#ff7f0e'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(hours_df, hours_view),
                'y': _plotly_y_nullable(hours_df['Total_Nurse_Aide_Hours']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Nurse Aide',
                'line': {'color': '#2ca02c'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
            }
        ] + hours_holiday_markers,
        'layout': {
            'title': {
                'text': 'Daily Hours Trends',
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'nticks': 10,
                'tickangle': -45
            },
            'yaxis': {'title': 'Hours'},
            'height': 450,
            'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
        }
    }
    
    # Census trend
    charts['census_trend'] = {
        'data': [
            {
                'x': format_dates_for_view_mode(census_df, census_view),
                'y': _plotly_y_nullable(census_df['MDScensus']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Census',
                'line': {'color': '#d62728'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f}<extra></extra>'
            }
        ],
        'layout': {
            'title': {
                'text': 'Daily Census Trend',
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'nticks': 10,
                'tickangle': -45
            },
            'yaxis': {'title': 'Census'},
            'height': 450,
            'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
        }
    }
    
    # Contract percentage trend
    charts['contract_trend'] = {
        'data': [
            {
                'x': format_dates_for_view_mode(contract_df, contract_view),
                'y': _plotly_y_nullable(contract_df['Total_Contract_Pct']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total %',
                'line': {'color': '#d62728', 'width': 3},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(contract_df, contract_view),
                'y': _plotly_y_nullable(contract_df['RN_Contract_Pct']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'RN %',
                'line': {'color': '#1f77b4'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(contract_df, contract_view),
                'y': _plotly_y_nullable(contract_df['Total_LPN_Contract_Pct']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total LPN %',
                'line': {'color': '#ff7f0e'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
            },
            {
                'x': format_dates_for_view_mode(contract_df, contract_view),
                'y': _plotly_y_nullable(contract_df['Nurse_Aide_Contract_Pct']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Nurse Aide %',
                'line': {'color': '#2ca02c'},
                'connectgaps': False,
                'hovertemplate': '<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
            }
        ],
        'layout': {
            'title': {
                'text': 'Contract Percentage Trends',
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'nticks': 10,
                'tickangle': -45
            },
            'yaxis': {'title': '% Contract', 'range': [0, None]},
            'height': 450,
            'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
        }
    }

    def _comp_hours(df_in: pd.DataFrame, col: str) -> pd.Series:
        if col in df_in.columns:
            vals = pd.to_numeric(df_in[col], errors='coerce')
            return cast(pd.Series, vals if isinstance(vals, pd.Series) else pd.Series(vals, index=df_in.index))
        return pd.Series([np.nan] * len(df_in), index=df_in.index, dtype='float64')

    def _comp_hprd(df_in: pd.DataFrame, col: str) -> list:
        hprd_col = f'__comp_hprd_{col}'
        if hprd_col in df_in.columns:
            return _json_float_list(
                _round_fin_series(pd.to_numeric(df_in[hprd_col], errors='coerce'), 3).tolist()
            )
        hours = _comp_hours(df_in, col)
        census = pd.to_numeric(df_in['MDScensus'], errors='coerce') if 'MDScensus' in df_in.columns else pd.Series([np.nan] * len(df_in), index=df_in.index, dtype='float64')
        return _json_float_list(_round_fin_series(_divide_hprd(hours, census), 3).tolist())

    def _comp_overlay_series(
        df_in: pd.DataFrame,
        hours_col: str,
        hprd_col: str,
        *,
        fallback_hour_cols: list[str] | None = None,
    ) -> dict[str, list]:
        if len(df_in) == 0:
            return {'hours': [], 'hprd': []}
        if hours_col in df_in.columns:
            hours = _json_float_list(
                _round_fin_series(pd.to_numeric(df_in[hours_col], errors='coerce'), 1).tolist()
            )
        elif fallback_hour_cols:
            hours = _comp_hours_sum(df_in, fallback_hour_cols)
        else:
            hours = []
        if hprd_col in df_in.columns:
            hprd = _json_float_list(
                _round_fin_series(pd.to_numeric(df_in[hprd_col], errors='coerce'), 3).tolist()
            )
        elif fallback_hour_cols:
            hprd = _comp_hprd_sum(df_in, fallback_hour_cols)
        else:
            hprd = []
        return {'hours': hours, 'hprd': hprd}

    def _comp_hours_list(df_in: pd.DataFrame, col: str) -> list:
        hours = _comp_hours(df_in, col)
        return _json_float_list(_round_fin_series(hours, 1).tolist())

    def _comp_hours_sum(df_in: pd.DataFrame, cols: list[str]) -> list:
        if len(df_in) == 0:
            return []
        total_hours = pd.Series([0.0] * len(df_in), index=df_in.index, dtype='float64')
        has_any = False
        for c in cols:
            if c in df_in.columns:
                total_hours = total_hours.add(pd.to_numeric(df_in[c], errors='coerce').fillna(0.0), fill_value=0.0)
                has_any = True
        if not has_any:
            return [None] * len(df_in)
        return _json_float_list(_round_fin_series(total_hours, 1).tolist())

    def _comp_hprd_sum(df_in: pd.DataFrame, cols: list[str]) -> list:
        if len(df_in) == 0:
            return []
        total_hours = pd.Series([0.0] * len(df_in), index=df_in.index, dtype='float64')
        has_any = False
        for c in cols:
            if c in df_in.columns:
                total_hours = total_hours.add(pd.to_numeric(df_in[c], errors='coerce').fillna(0.0), fill_value=0.0)
                has_any = True
        if not has_any:
            return [None] * len(df_in)
        census = pd.to_numeric(df_in['MDScensus'], errors='coerce') if 'MDScensus' in df_in.columns else pd.Series([np.nan] * len(df_in), index=df_in.index, dtype='float64')
        return _json_float_list(_round_fin_series(_divide_hprd(total_hours, census), 3).tolist())

    composition_dates = format_dates_for_view_mode(composition_df, composition_view) if len(composition_df) else []
    composition_census = (
        _json_float_list(_round_fin_series(pd.to_numeric(composition_df['MDScensus'], errors='coerce'), 1).tolist())
        if len(composition_df) and 'MDScensus' in composition_df.columns
        else []
    )
    charts['composition_trend'] = {
        'view': composition_view,
        'x': composition_dates,
        'census': composition_census,
        'total': {
            'rn_don': _comp_hprd(composition_df, 'Hrs_RNDON'),
            'rn_admin': _comp_hprd(composition_df, 'Hrs_RNadmin'),
            'rn_floor': _comp_hprd(composition_df, 'Hrs_RN'),
            'lpn_admin': _comp_hprd(composition_df, 'Hrs_LPNadmin'),
            'lpn_direct': _comp_hprd(composition_df, 'Hrs_LPN'),
            'cna': _comp_hprd(composition_df, 'Hrs_CNA'),
            'med_aide': _comp_hprd(composition_df, 'Hrs_MedAide'),
            'na_train': _comp_hprd(composition_df, 'Hrs_NAtrn'),
        },
        'total_hours': {
            'rn_don': _comp_hours_list(composition_df, 'Hrs_RNDON'),
            'rn_admin': _comp_hours_list(composition_df, 'Hrs_RNadmin'),
            'rn_floor': _comp_hours_list(composition_df, 'Hrs_RN'),
            'lpn_admin': _comp_hours_list(composition_df, 'Hrs_LPNadmin'),
            'lpn_direct': _comp_hours_list(composition_df, 'Hrs_LPN'),
            'cna': _comp_hours_list(composition_df, 'Hrs_CNA'),
            'med_aide': _comp_hours_list(composition_df, 'Hrs_MedAide'),
            'na_train': _comp_hours_list(composition_df, 'Hrs_NAtrn'),
        },
        'direct': {
            'rn_direct': _comp_hprd(composition_df, 'Hrs_RN'),
            'lpn_direct': _comp_hprd(composition_df, 'Hrs_LPN'),
            'aide_direct': _comp_hprd_sum(composition_df, ['Hrs_CNA', 'Hrs_MedAide', 'Hrs_NAtrn']),
        },
        'direct_hours': {
            'rn_direct': _comp_hours_list(composition_df, 'Hrs_RN'),
            'lpn_direct': _comp_hours_list(composition_df, 'Hrs_LPN'),
            'aide_direct': _comp_hours_sum(composition_df, ['Hrs_CNA', 'Hrs_MedAide', 'Hrs_NAtrn']),
        },
        'overlay': {
            'direct_care': _comp_overlay_series(
                composition_df,
                '__comp_hours_direct',
                '__comp_hprd_direct',
                fallback_hour_cols=list(_COMPOSITION_DIRECT_HOUR_COLS),
            ),
            'contract': _comp_overlay_series(
                composition_df,
                '__comp_hours_contract',
                '__comp_hprd_contract',
                fallback_hour_cols=list(_COMPOSITION_CONTRACT_HOUR_COLS),
            ),
        },
    }
    
    # Get filter information for chart titles
    filter_info = get_filter_description(filter_label_start_date, filter_label_end_date, filter_quarter, filter_day_of_week, filter_holidays_only)
    
    # Update chart titles with filter information
    if filter_info and not filter_info.startswith("All Data ("):
        charts['hprd_trend']['layout']['title']['text'] = f"Daily HPRD Trends<br>{filter_info}"
        charts['dow_comparison']['layout']['title']['text'] = f"Average HPRD by Day of Week<br>{filter_info}"
        charts["dow_comparison_weekpart"]["layout"]["title"]["text"] = f"Average HPRD: Weekday vs. Weekend by metric<br>{filter_info}"
        charts['hours_trend']['layout']['title']['text'] = f"Daily Hours Trends<br>{filter_info}"
        charts['census_trend']['layout']['title']['text'] = f"Daily Census Trend<br>{filter_info}"
        charts['contract_trend']['layout']['title']['text'] = f"Contract Percentage Trends<br>{filter_info}"

    dow_year_range_label = None
    if len(dow_df) > 0 and "WorkDate" in dow_df.columns:
        try:
            _wd_min = pd.to_datetime(dow_df["WorkDate"], errors="coerce").min()
            _wd_max = pd.to_datetime(dow_df["WorkDate"], errors="coerce").max()
            if pd.notna(_wd_min) and pd.notna(_wd_max):
                y0 = int(_wd_min.year)
                y1 = int(_wd_max.year)
                dow_year_range_label = str(y0) if y0 == y1 else f"{y0}–{y1}"
        except Exception:
            dow_year_range_label = None

    census_days_missing = 0
    census_days_zero = 0
    if len(filtered_df) > 0 and "MDScensus" in filtered_df.columns:
        mdc_ch = pd.to_numeric(filtered_df["MDScensus"], errors="coerce")
        census_days_missing = int(mdc_ch.isna().sum())
        census_days_zero = int((mdc_ch.notna() & (mdc_ch == 0)).sum())
    hprd_q_charts = _hprd_quality_block(
        filtered_df["MDScensus"] if len(filtered_df) and "MDScensus" in filtered_df.columns else pd.Series(dtype=float),
        len(filtered_df),
    )
    
    return {
        'charts': charts,
        'filter_info': filter_info,
        'dow_year_range_label': dow_year_range_label,
        'census_quality': {
            'days_missing_census': census_days_missing,
            'days_zero_census': census_days_zero,
        },
        'hprd_quality': hprd_q_charts,
        'provider_info_match_quarter': _provider_match_quarter_param_from_pbj_df(
            cast(pd.DataFrame, filtered_df)
        ),
    }

@app.route('/api/charts')
def get_charts():
    """Get chart data"""
    global global_df
    try:
        # Check if data is loaded
        if global_df is None:
            return jsonify({
                'charts': {},
                'filter_info': 'Data not loaded',
                'error': 'Data not loaded. Please restart the application.'
            })
        _ensure_pbj_dashboard_derived_columns_inplace(global_df)
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        day_of_week = request.args.get('day_of_week', 'all')
        quarter = request.args.get('quarter', 'all')
        year = request.args.get('year', 'all')
        show_holidays_only = request.args.get('holidays_only', 'false') == 'true'
        
        # Get view mode parameters
        hprd_view = request.args.get('hprd_view', 'daily')
        hours_view = request.args.get('hours_view', 'daily')
        census_view = request.args.get('census_view', 'daily')
        contract_view = request.args.get('contract_view', 'daily')
        composition_view = request.args.get('composition_view', 'month')
        
        # Keep source dataframe read-only in request handlers; compute any missing series locally.
        df_for_charts = cast(pd.DataFrame, global_df)
        if "Total_Staff_Hours" not in df_for_charts.columns:
            df_for_charts = df_for_charts.copy()
            trn = df_for_charts["Total_RN_Hours"] if "Total_RN_Hours" in df_for_charts.columns else 0.0
            tln = df_for_charts["Total_LPN_Hours"] if "Total_LPN_Hours" in df_for_charts.columns else 0.0
            tna = df_for_charts["Total_Nurse_Aide_Hours"] if "Total_Nurse_Aide_Hours" in df_for_charts.columns else 0.0
            df_for_charts["Total_Staff_Hours"] = trn + tln + tna
        if "Total_Staff_HPRD" not in df_for_charts.columns and "Total_Staff_Hours" in df_for_charts.columns and "MDScensus" in df_for_charts.columns:
            if df_for_charts is global_df:
                df_for_charts = df_for_charts.copy()
            df_for_charts["Total_Staff_HPRD"] = _round_fin_series(
                _divide_hprd(df_for_charts["Total_Staff_Hours"], df_for_charts["MDScensus"]), 2
            )
        
        try:
            filtered_df = _filter_facility_daily_cached(
                df_for_charts,
                start_date=start_date,
                end_date=end_date,
                day_of_week=day_of_week,
                quarter=quarter,
                year=year,
                holidays_only=show_holidays_only,
            )
        except ValueError as e:
            return jsonify(
                {
                    "error": str(e),
                    "charts": {},
                    "filter_info": str(e),
                }
            ), 400
        
        charts_payload = _charts_build_payload_dict(
            filtered_df,
            filter_label_start_date=start_date,
            filter_label_end_date=end_date,
            filter_quarter=quarter,
            filter_day_of_week=day_of_week,
            filter_holidays_only=show_holidays_only,
            dow_calendar_year_param=request.args.get("dow_calendar_year"),
            hprd_view=hprd_view,
            hours_view=hours_view,
            census_view=census_view,
            contract_view=contract_view,
            composition_view=composition_view,
        )
        return jsonify(charts_payload)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/chart_aggregated')
def get_chart_aggregated():
    """Get chart data aggregated by quarter, month, or year for any chart type"""
    try:
        aggregation_type = request.args.get('type', 'quarter')  # 'quarter', 'month', or 'year'
        chart_type = request.args.get('chart_type', 'hprd')  # 'hprd', 'hours', 'census', 'contract'
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        quarter = request.args.get('quarter', 'all')
        year = request.args.get('year', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        show_holidays_only = request.args.get('show_holidays_only', 'false').lower() == 'true'
        
        global global_df
        if global_df is None or len(global_df) == 0:
            return jsonify({'error': 'No data loaded'})
        _ensure_pbj_dashboard_derived_columns_inplace(global_df)
        try:
            filtered_df = _filter_facility_daily_for_dashboard(
                global_df,
                start_date=start_date,
                end_date=end_date,
                day_of_week=day_of_week,
                quarter=quarter,
                year=year,
                holidays_only=show_holidays_only,
            )
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        if filtered_df.empty:
            return jsonify({'error': 'No data available for the selected filters'})
        
        # Sort by date
        filtered_df = filtered_df.sort_values('WorkDate')
        
        # Aggregate data based on type
        if aggregation_type == 'quarter':
            agg_data = filtered_df.groupby('CY_Qtr').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean',
                'Total_Staff_Hours': 'mean',
                'Nurse_Staff_Hours_Excl_Admin': 'mean',
                'Total_RN_Hours': 'mean',
                'Total_LPN_Hours': 'mean',
                'Total_Nurse_Aide_Hours': 'mean',
                'MDScensus': 'mean',
                'Total_Contract_Pct': 'mean',
                'RN_Contract_Pct': 'mean',
                'LPN_Contract_Pct': 'mean',
                'CNA_Contract_Pct': 'mean',
                'Total_LPN_Contract_Pct': 'mean',
                'Nurse_Aide_Contract_Pct': 'mean'
            }).round(2)
            # Convert 2017Q1 format to Q1 2017 format
            x_values = []
            for quarter in agg_data.index.tolist():
                if 'Q' in quarter:
                    year = quarter.split('Q')[0]
                    quarter_num = quarter.split('Q')[1]
                    x_values.append(f"Q{quarter_num} {year}")
                else:
                    x_values.append(quarter)
            title_suffix = "Quarterly"
        elif aggregation_type == 'year':
            filtered_df['Year'] = filtered_df['WorkDate'].dt.year.astype(str)
            agg_data = filtered_df.groupby('Year').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean',
                'Total_Staff_Hours': 'mean',
                'Nurse_Staff_Hours_Excl_Admin': 'mean',
                'Total_RN_Hours': 'mean',
                'Total_LPN_Hours': 'mean',
                'Total_Nurse_Aide_Hours': 'mean',
                'MDScensus': 'mean',
                'Total_Contract_Pct': 'mean',
                'RN_Contract_Pct': 'mean',
                'LPN_Contract_Pct': 'mean',
                'CNA_Contract_Pct': 'mean',
                'Total_LPN_Contract_Pct': 'mean',
                'Nurse_Aide_Contract_Pct': 'mean'
            }).round(2)
            x_values = agg_data.index.tolist()
            title_suffix = "Annually"
        else:  # month
            filtered_df['YearMonth'] = filtered_df['WorkDate'].dt.to_period('M').astype(str)
            agg_data = filtered_df.groupby('YearMonth').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean',
                'Total_Staff_Hours': 'mean',
                'Nurse_Staff_Hours_Excl_Admin': 'mean',
                'Total_RN_Hours': 'mean',
                'Total_LPN_Hours': 'mean',
                'Total_Nurse_Aide_Hours': 'mean',
                'MDScensus': 'mean',
                'Total_Contract_Pct': 'mean',
                'RN_Contract_Pct': 'mean',
                'LPN_Contract_Pct': 'mean',
                'CNA_Contract_Pct': 'mean',
                'Total_LPN_Contract_Pct': 'mean',
                'Nurse_Aide_Contract_Pct': 'mean'
            }).round(2)
            x_values = agg_data.index.tolist()
            title_suffix = "Monthly"
        
        # Create chart data based on chart type
        if chart_type == 'hprd':
            chart_data = [
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_Nurse_HPRD']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total',
                    'line': {'color': '#d62728', 'width': 3},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Nurse_Staff_HPRD_Excl_Admin']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Direct',
                    'line': {'color': '#9467bd'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_RN_HPRD']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN HPRD',
                    'line': {'color': '#2ca02c'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_LPN_HPRD']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total LPN HPRD',
                    'line': {'color': '#ff7f0e'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_Nurse_Aide_HPRD']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide HPRD',
                    'line': {'color': '#1f77b4'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f} HPRD<extra></extra>'
                }
            ]
            y_title = 'HPRD'
        elif chart_type == 'hours':
            chart_data = [
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_Staff_Hours']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total Staff Hours',
                    'line': {'color': '#d62728', 'width': 3},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Nurse_Staff_Hours_Excl_Admin']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Staff Hours (excl. Admin & DON)',
                    'line': {'color': '#9467bd'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_RN_Hours']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN Hours',
                    'line': {'color': '#2ca02c'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_LPN_Hours']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN Hours',
                    'line': {'color': '#ff7f0e'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_Nurse_Aide_Hours']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide Hours',
                    'line': {'color': '#1f77b4'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:,.2f} Hours<extra></extra>'
                }
            ]
            y_title = 'Hours'
        elif chart_type == 'census':
            chart_data = [
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['MDScensus']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Resident Census',
                    'line': {'color': '#2ca02c', 'width': 3},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f}<extra></extra>'
                }
            ]
            y_title = 'Residents'
        elif chart_type == 'contract':
            chart_data = [
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_Contract_Pct']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total %',
                    'line': {'color': '#d62728', 'width': 3},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['RN_Contract_Pct']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN %',
                    'line': {'color': '#2ca02c'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Total_LPN_Contract_Pct']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total LPN %',
                    'line': {'color': '#ff7f0e'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                },
                {
                    'x': x_values,
                    'y': _plotly_y_nullable(agg_data['Nurse_Aide_Contract_Pct']),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide %',
                    'line': {'color': '#1f77b4'},
                    'connectgaps': False,
                    'hovertemplate': '<b>%{x}</b><br>%{y:.2f}%<extra></extra>'
                }
            ]
            y_title = '% Contract'
        else:
            return jsonify({'error': 'Invalid chart type'})
        
        layout = {
            'title': {
                'text': f'Average {chart_type.title()} {title_suffix}',
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'nticks': 10,
                'tickangle': 45
            },
            'yaxis': {'title': y_title},
            'height': 450,
            'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
        }
        
        return jsonify({
            'data': chart_data,
            'layout': layout,
            'aggregation_type': aggregation_type,
            'chart_type': chart_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/hprd_aggregated')
def get_hprd_aggregated():
    """Get HPRD data aggregated by quarter or month"""
    try:
        aggregation_type = request.args.get('type', 'quarter')  # 'quarter' or 'month'
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        quarter = request.args.get('quarter', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        show_holidays_only = request.args.get('show_holidays_only', 'false').lower() == 'true'
        
        global global_df
        if global_df is None or len(global_df) == 0:
            return jsonify({'error': 'No data loaded'})
        try:
            filtered_df = _filter_facility_daily_for_dashboard(
                global_df,
                start_date=start_date,
                end_date=end_date,
                day_of_week=day_of_week,
                quarter=quarter,
                year="all",
                holidays_only=show_holidays_only,
            )
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        
        if filtered_df.empty:
            return jsonify({'error': 'No data available for the selected filters'})
        
        # Sort by date
        filtered_df = filtered_df.sort_values('WorkDate')
        
        # Aggregate data based on type
        if aggregation_type == 'quarter':
            agg_data = filtered_df.groupby('CY_Qtr').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean'
            }).round(2)
            # Convert 2017Q1 format to Q1 2017 format
            x_values = []
            for quarter in agg_data.index.tolist():
                if 'Q' in quarter:
                    year = quarter.split('Q')[0]
                    quarter_num = quarter.split('Q')[1]
                    x_values.append(f"Q{quarter_num} {year}")
                else:
                    x_values.append(quarter)
            title_suffix = "by Quarter"
        elif aggregation_type == 'year':
            filtered_df['Year'] = filtered_df['WorkDate'].dt.year.astype(str)
            agg_data = filtered_df.groupby('Year').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean'
            }).round(2)
            x_values = agg_data.index.tolist()
            title_suffix = "by Year"
        else:  # month
            filtered_df['YearMonth'] = filtered_df['WorkDate'].dt.to_period('M').astype(str)
            agg_data = filtered_df.groupby('YearMonth').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean'
            }).round(2)
            x_values = agg_data.index.tolist()
            title_suffix = "by Month"
        
        # Create chart data
        chart_data = [
            {
                'x': x_values,
                'y': _plotly_y_nullable(agg_data['Total_Nurse_HPRD']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total HPRD (All Staff)',
                'line': {'color': '#d62728', 'width': 3},
                'connectgaps': False,
            },
            {
                'x': x_values,
                'y': _plotly_y_nullable(agg_data['Nurse_Staff_HPRD_Excl_Admin']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Direct Staff HPRD',
                'line': {'color': '#9467bd'},
                'connectgaps': False,
            },
            {
                'x': x_values,
                'y': _plotly_y_nullable(agg_data['Total_RN_HPRD']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'RN HPRD',
                'line': {'color': '#2ca02c'},
                'connectgaps': False,
            },
            {
                'x': x_values,
                'y': _plotly_y_nullable(agg_data['Total_LPN_HPRD']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total LPN HPRD',
                'line': {'color': '#ff7f0e'},
                'connectgaps': False,
            },
            {
                'x': x_values,
                'y': _plotly_y_nullable(agg_data['Total_Nurse_Aide_HPRD']),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Nurse Aide HPRD',
                'line': {'color': '#1f77b4'},
                'connectgaps': False,
            }
        ]
        
        layout = {
            'title': {
                'text': f'Average HPRD {title_suffix}',
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'nticks': 10,
                'tickangle': 45
            },
            'yaxis': {'title': 'HPRD'},
            'height': 450,
            'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
        }
        
        return jsonify({
            'data': chart_data,
            'layout': layout,
            'aggregation_type': aggregation_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route("/api/nonnurse/meta")
def api_nonnurse_meta():
    global nonnurse_df, global_df
    try:
        if not INCLUDE_NONNURSE:
            return jsonify(
                {
                    "available": False,
                    "included": False,
                    "inventory": None,
                    "reason": None,
                }
            )
        if (nonnurse_df is None or nonnurse_df.empty) and _nonnurse_path_is_usable(_NONNURSE_CSV_PATH):
            return jsonify(
                {
                    "available": True,
                    "included": True,
                    "inventory": None,
                    "deferred": True,
                    "reason": None,
                }
            )
        if not _ensure_nonnurse_loaded():
            prov = ""
            try:
                if global_df is not None and len(global_df) and "PROVNUM" in global_df.columns:
                    prov = str(global_df["PROVNUM"].iloc[0]).zfill(6)
            except Exception:
                prov = ""
            return jsonify(
                {
                    "available": False,
                    "included": True,
                    "inventory": None,
                    "reason": "No non-nurse daily rows are loaded for this deployment.",
                }
            )
        from facility_ein_lib import NONNURSE_PBJ_EIN_JOB_CODES, nonnurse_hrs_columns_to_ein_job_codes
        from nonnurse_staffing_lib import load_nonnurse_groups_config, nonnurse_column_inventory

        inv = nonnurse_column_inventory(nonnurse_df)
        cfg = load_nonnurse_groups_config()
        ein_by_group: dict[str, list[int]] = {}
        group_column_tooltips: dict[str, str] = {}
        for g in cfg.get("groups", []):
            gid = str(g.get("id", ""))
            ein_by_group[gid] = list(nonnurse_hrs_columns_to_ein_job_codes(g.get("columns") or []))
            cols = g.get("columns") or []
            if cols:
                group_column_tooltips[gid] = "PBJ columns summed in this group: " + ", ".join(str(c) for c in cols)
        return jsonify(
            {
                "available": True,
                "included": True,
                "inventory": inv,
                "group_column_tooltips": group_column_tooltips,
                "ein": {
                    "job_codes_by_group": ein_by_group,
                    "job_codes_nonnurse_hrs_mapped": sorted(NONNURSE_PBJ_EIN_JOB_CODES),
                    "note": (
                        "Prepared mapping: PBJ non-nurse Hrs_* columns → Employee Detail EMPLEE_JOB_CD_ID. "
                        "Use for a future non-nurse PBJ→EIN bridge; nursing bridge remains primary."
                    ),
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/nonnurse/summary")
def api_nonnurse_summary():
    global nonnurse_df, global_df
    try:
        if not INCLUDE_NONNURSE:
            return jsonify({"quarters": [], "meta": {"nonnurse_included": False}})
        if not _ensure_nonnurse_loaded():
            return jsonify({"quarters": []})
        from nonnurse_staffing_lib import summarize_nonnurse_by_quarter

        from pbj_identifiers.urls import cms_nh_health_citations_dataset_explorer_url, cms_nonnurse_pbj_data_explorer_url

        rows = summarize_nonnurse_by_quarter(nonnurse_df)
        # Newest payroll quarter first (matches facility quarter pickers and daily segment UX).
        rows = list(reversed(rows))
        prov = (
            str(global_df["PROVNUM"].iloc[0]).zfill(6)
            if global_df is not None and len(global_df) and "PROVNUM" in global_df.columns
            else ""
        )
        latest_q = rows[0].get("cy_qtr") if rows else None
        for row in rows:
            qk = row.get("cy_qtr")
            row["cms_nonnurse_quarter_url"] = cms_nonnurse_pbj_data_explorer_url(prov, str(qk) if qk else None)
        return jsonify(
            {
                "quarters": rows,
                "meta": {
                    "provnum": prov,
                    "methodology": {
                        "hprd_quarter": (
                            "Quarter HPRD = sum(group hours) / sum(MDScensus) within the quarter after "
                            "collapsing duplicate WorkDate rows (same day summed)."
                        ),
                        "days": (
                            "days = distinct WorkDates present in the non-nurse file for that quarter after collapse; "
                            "calendar_days_in_quarter is the full Medicare quarter length; "
                            "work_dates_missing_vs_calendar is the gap (when the file is incomplete)."
                        ),
                        "zero_hour_days": (
                            "days_zero_hours counts days where that group's summed hours are exactly 0 (after day-level sums)."
                        ),
                    },
                    "sources": {
                        "nonnurse_pbj_cms": cms_nonnurse_pbj_data_explorer_url(prov, str(latest_q) if latest_q else None),
                        "nh_health_citations_cms": cms_nh_health_citations_dataset_explorer_url(prov),
                    },
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/nonnurse/daily")
def api_nonnurse_daily():
    global nonnurse_df, global_df
    try:
        if not INCLUDE_NONNURSE:
            return jsonify(
                {"rows": [], "meta": {"row_count": 0, "truncated": False, "nonnurse_included": False}}
            )
        if not _ensure_nonnurse_loaded():
            return jsonify({"rows": [], "meta": {"row_count": 0, "truncated": False}})
        start = request.args.get("start") or None
        end = request.args.get("end") or None
        limit = min(int(request.args.get("limit", 366)), 2000)
        from nonnurse_staffing_lib import nonnurse_daily_for_range

        rows, meta = nonnurse_daily_for_range(nonnurse_df, start, end, limit=limit)
        prov = (
            str(global_df["PROVNUM"].iloc[0]).zfill(6)
            if global_df is not None and len(global_df) and "PROVNUM" in global_df.columns
            else ""
        )
        from pbj_identifiers.urls import cms_pbj_daily_staffing_explorer_url

        for row in rows:
            qk = row.get("cy_qtr")
            wd = row.get("work_date")
            if prov and qk and wd:
                row["cms_nonnurse_daily_url"] = cms_pbj_daily_staffing_explorer_url(
                    str(qk), wd, prov, data_type="nonnurse"
                )
            else:
                row["cms_nonnurse_daily_url"] = None
        return jsonify({"rows": rows, "meta": meta})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/ein-nonnurse-day-summary")
def api_ein_nonnurse_day_summary():
    """Day-level non-nurse summary for the Employee Tracker day view."""
    try:
        if not INCLUDE_NONNURSE:
            return jsonify(
                {
                    "ok": True,
                    "available": False,
                    "message": "Non-nurse mode is disabled for this deployment.",
                    "roles": [],
                }
            )
        if not _ensure_nonnurse_loaded():
            return jsonify(
                {
                    "ok": True,
                    "available": False,
                    "message": "No non-nurse daily rows are loaded for this deployment.",
                    "roles": [],
                }
            )
        date_s = (request.args.get("date") or "").strip()
        if not date_s:
            return jsonify({"ok": True, "available": False, "message": "date is required", "roles": []})
        return jsonify(_nonnurse_day_summary_payload(date_s))
    except Exception as exc:
        return jsonify({"ok": False, "available": False, "message": str(exc), "roles": []})


@app.route("/api/ein-nonnurse-day-roster")
def api_ein_nonnurse_day_roster():
    """Employee-level non-nurse EIN roster for one work date, with non-nurse daily cross-check."""
    global ein_employee_detail_df
    try:
        if not _ein_mode_enabled():
            return jsonify({"ok": False, "available": False, "message": "Employee Detail mode is disabled for this dashboard."})
        _ensure_ein_employee_detail_loaded()
        date_s = (request.args.get("date") or "").strip()
        if not date_s:
            return jsonify({"ok": False, "available": False, "message": "date is required", "employees": []})
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return jsonify(
                {
                    "ok": False,
                    "available": False,
                    "message": "Row-level Employee Detail file not loaded.",
                    "employees": [],
                }
            )
        rows, total, qn = aggregate_ein_day_by_job_codes(
            ein_employee_detail_df, date_s, NONNURSE_PBJ_EIN_JOB_CODES
        )
        ccn = _ein_active_ccn()
        _enrich_nursing_api_rows(rows, ccn)
        summary = _nonnurse_day_summary_payload(date_s)
        daily_total = summary.get("hours_total") if isinstance(summary, dict) else None
        cross = compare_pbj_vs_ein_hours(daily_total, float(total))
        return jsonify(
            {
                "ok": True,
                "available": True,
                "date": date_s,
                "quarter": qn,
                "employees": rows,
                "hours_total_ein": round_financial(float(total), 2),
                "hours_total_nonnurse_daily": daily_total,
                "hours_cross_check": cross,
                "roles": summary.get("roles", []) if isinstance(summary, dict) else [],
                "cms_nonnurse_daily_url": summary.get("cms_nonnurse_daily_url") if isinstance(summary, dict) else None,
                "day_census": summary.get("day_census") if isinstance(summary, dict) else None,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "available": False, "message": str(exc), "employees": []})


@app.route("/api/ein-nonnurse-employees")
def api_ein_nonnurse_employees():
    """Non-nurse PBJ-mapped EIN employee summaries (quarter roster; optional work-day rolling fields)."""
    global ein_employee_detail_df
    try:
        if not _ein_mode_enabled():
            return jsonify(
                {"available": False, "employees": [], "message": "Employee Detail mode is disabled for this dashboard."}
            )
        if not INCLUDE_NONNURSE:
            return jsonify(
                {
                    "available": False,
                    "employees": [],
                    "message": "Non-nurse mode is disabled for this deployment.",
                }
            )
        _ensure_ein_employee_detail_loaded()
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return jsonify(
                {
                    "available": False,
                    "employees": [],
                    "message": "Row-level Employee Detail file not loaded.",
                }
            )
        ccn = _ein_active_ccn()
        quarter = request.args.get("quarter") or "all"
        work_date_anchor = (request.args.get("work_date") or "").strip()
        offset = max(0, int(request.args.get("offset", type=int) or 0))
        limit_raw = request.args.get("limit", type=int)
        q_filter_norm = (
            normalize_cy_qtr_ein(quarter)
            if quarter and str(quarter).strip().lower() not in ("", "all")
            else None
        )
        position_group = (request.args.get("position_group") or "all").strip().lower()

        rows_all = dedupe_nursing_roster_api_rows(
            nonnurse_employee_summaries(ein_employee_detail_df, quarter="all")
        )
        enrich_nursing_roster_display_fields(rows_all)
        enrich_nursing_rows_new_to_quarter_flags(rows_all)
        enrich_nursing_rows_multi_role_flags(rows_all)
        dprep = prepare_ein_detail(ein_employee_detail_df)
        pairs_by_q = roster_pairs_by_quarter_for_job_codes(dprep, NONNURSE_PBJ_EIN_JOB_CODES)
        quarter_summary = (
            compute_ein_quarter_nonnurse_roster_summary(q_filter_norm, pairs_by_q) if q_filter_norm else None
        )
        if q_filter_norm:
            rows_filtered = [
                r for r in rows_all if normalize_cy_qtr_ein(r.get("quarter")) == q_filter_norm
            ]
        else:
            rows_filtered = rows_all
        if position_group and position_group != "all":
            rows_filtered = [
                r
                for r in rows_filtered
                if ein_nonnurse_job_code_matches_position_group(r.get("job_code"), position_group)
            ]
        apply_roster_tenure_quarter_span(rows_filtered, rows_for_global_bounds=rows_all)
        total = len(rows_filtered)
        rows_filtered.sort(
            key=lambda row: (
                -(parse_ein_quarter_bound(str(row.get("quarter") or "")) or 0),
                -float(row.get("total_hours") or 0),
                int(row.get("first_work_date_raw") or 0) or 10**9,
                int(row.get("sys_employee_id") or 0),
                int(row.get("job_code") or 0),
            )
        )
        if limit_raw is None:
            rows = rows_filtered[offset:]
        else:
            lim = max(1, int(limit_raw))
            rows = rows_filtered[offset : offset + lim]
        _enrich_nursing_api_rows(rows, ccn)
        if work_date_anchor and ein_employee_detail_df is not None and not ein_employee_detail_df.empty:
            enrich_nonnurse_rows_rolling_from_work_date(ein_employee_detail_df, rows, work_date_anchor)
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
                "source": "detail",
            }
        )
    except Exception as exc:
        return jsonify({"available": False, "employees": [], "error": str(exc)})


@app.route("/api/citations")
def api_citations():
    global global_df, citations_df
    try:
        if citations_df is None or citations_df.empty:
            return jsonify({"rows": [], "meta": {"total": 0, "returned": 0}})
        prov = str(global_df["PROVNUM"].iloc[0]).zfill(6) if global_df is not None and len(global_df) else ""
        from citation_lib import citations_to_api_rows, enrich_citations_dataframe, _citations_dataframe_for_ccn

        d = _citations_dataframe_for_ccn(citations_df, prov)
        d = enrich_citations_dataframe(d, prov)
        from_ts = request.args.get("from")
        if from_ts:
            d = d[d["_survey_ts"] >= pd.Timestamp(from_ts)]
        to_ts = request.args.get("to")
        if to_ts:
            d = d[d["_survey_ts"] <= pd.Timestamp(to_ts)]
        if request.args.get("complaint_only", "").lower() in ("1", "true", "yes"):
            d = d[d["complaint_deficiency_bool"] == True] if "complaint_deficiency_bool" in d.columns else d.iloc[0:0]
        if request.args.get("infection_only", "").lower() in ("1", "true", "yes"):
            d = (
                d[d["infection_control_deficiency_bool"] == True]
                if "infection_control_deficiency_bool" in d.columns
                else d.iloc[0:0]
            )
        min_sr = request.args.get("min_severity_rank")
        if min_sr is not None and str(min_sr).strip().isdigit():
            d = d[d["severity_rank"].fillna(-1) >= int(min_sr)]
        elif request.args.get("g_plus_only", "").lower() in ("1", "true", "yes"):
            from citation_lib import load_citation_severity_config

            cfg = load_citation_severity_config()
            g_min = int(cfg.get("dashboard_citation_flag_min_rank", 60))
            d = d[d["severity_rank"].fillna(-1) >= g_min]
        elif request.args.get("dashboard_flag_only", "").lower() in ("1", "true", "yes"):
            from citation_lib import load_citation_severity_config

            d_min = int(load_citation_severity_config().get("dashboard_citation_flag_min_rank", 60))
            d = d[d["severity_rank"].fillna(-1) >= d_min]
        lim = min(int(request.args.get("limit", 500)), 2000)
        rows = citations_to_api_rows(cast(pd.DataFrame, d), prov, max_rows=lim)
        return jsonify({"rows": rows, "meta": {"total": int(len(d)), "returned": len(rows)}})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/api/citations/summary")
def api_citations_summary():
    global global_df, citations_df
    try:
        prov = str(global_df["PROVNUM"].iloc[0]).zfill(6) if global_df is not None and len(global_df) else ""
        pbj_min = pbj_max = None
        if global_df is not None and len(global_df) and "WorkDate" in global_df.columns:
            wd = pd.to_datetime(global_df["WorkDate"], errors="coerce")
            if wd.notna().any():
                pbj_min = wd.min().strftime("%Y-%m-%d")
                pbj_max = wd.max().strftime("%Y-%m-%d")
        from citation_lib import citations_summary

        latest_citations_mm_yyyy = _latest_nh_health_citations_month_year()
        citations_as_of = _nh_health_citations_dataset_asof_display()
        if citations_df is None or citations_df.empty:
            out = citations_summary(pd.DataFrame(), prov, pbj_min, pbj_max)
            out["citations_dataset_month_year"] = latest_citations_mm_yyyy
            out["citations_dataset_as_of"] = citations_as_of
            out["citations_gplus_footnote"] = (
                f"G+ citation flags are based on latest deficiencies dataset ({citations_as_of}). "
                "Counts are grouped by inspection survey calendar quarter (e.g. Q3 2025), "
                "not by CMS extract month in the filename."
                if citations_as_of
                else (
                    "G+ citation flags use the loaded NH Health Citations file when present. "
                    "Counts are grouped by inspection survey calendar quarter."
                )
            )
            return jsonify(out)
        from citation_lib import _citations_dataframe_for_ccn

        ccn_citations = _citations_dataframe_for_ccn(citations_df, prov)
        out = citations_summary(ccn_citations, prov, pbj_min, pbj_max)
        out["citations_dataset_month_year"] = latest_citations_mm_yyyy
        out["citations_dataset_as_of"] = citations_as_of
        out["citations_gplus_footnote"] = (
            f"G+ citation flags are based on latest deficiencies dataset ({citations_as_of}). "
            "Counts are grouped by inspection survey calendar quarter (e.g. Q3 2025), "
            "not by CMS extract month in the filename."
            if citations_as_of
            else (
                "G+ citation flags use the loaded NH Health Citations file. "
                "Counts are grouped by inspection survey calendar quarter."
            )
        )
        return jsonify(out)
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/quarters')
def get_quarters():
    """Get available quarters"""
    global global_df
    try:
        if global_df is None or global_df.empty:
            return jsonify([])
        if 'CY_Qtr' not in global_df.columns:
            return jsonify([])
        quarters = sorted(global_df['CY_Qtr'].unique().tolist(), reverse=True)
        return jsonify(quarters)
    except Exception as e:
        print(f"Error getting quarters: {str(e)}")
        return jsonify([])

@app.route('/api/date_range')
def get_date_range():
    """Get available date range (filtered to valid PBJ data from 2017 onwards)"""
    try:
        global global_df
        if global_df is None or len(global_df) == 0:
            return jsonify({"min_date": None, "max_date": None, "data_loaded": False})
        
        # Filter out any data before 2017 (invalid/outlier data)
        # Use explicit date filtering to avoid timezone issues
        from datetime import datetime
        start_2017 = datetime(2017, 1, 1)
        valid_data = global_df[global_df['WorkDate'] >= start_2017]
        
        if len(valid_data) == 0:
            return jsonify(
                {
                    "min_date": None,
                    "max_date": None,
                    "data_loaded": True,
                    "note": "No workdays on or after 2017-01-01 in loaded PBJ daily file.",
                }
            )
        
        # Use .date() to ensure we get just the date part without time/timezone issues
        min_date = valid_data['WorkDate'].min().date()
        max_date = valid_data['WorkDate'].max().date()
        
        return jsonify(
            {
                "min_date": min_date.strftime("%Y-%m-%d"),
                "max_date": max_date.strftime("%Y-%m-%d"),
                "data_loaded": True,
            }
        )
    except Exception as e:
        import traceback
        print(f"Error in get_date_range: {str(e)}")
        traceback.print_exc()
        return jsonify({"min_date": None, "max_date": None, "data_loaded": False, "error": str(e)}), 500


@app.route("/api/pre-post-default-windows")
def pre_post_default_windows():
    """
    Default pre/post windows: affiliated entity / chain / name inflection from Provider Info when
    possible, else significant PROVNAME change in PBJ; otherwise same ~180d / ~90d as latest PBJ day.
    """
    global global_df, provider_info_df
    try:
        ensure_data_loaded()
        if global_df is None or global_df.empty:
            return jsonify({"error": "No data loaded"}), 400
        wd = pd.to_datetime(global_df["WorkDate"], errors="coerce")
        min_d = wd.min()
        max_d = wd.max()
        if pd.isna(min_d) or pd.isna(max_d):
            return jsonify({"error": "Could not parse WorkDate"}), 400
        min_date = pd.Timestamp(min_d).normalize()
        max_date = pd.Timestamp(max_d).normalize()

        chow_payload = chow_facility_api_payload(PROVNUM, limit=1)
        chow_date_str = str(chow_payload.get("latest_effective_date") or "").strip()
        if chow_date_str:
            try:
                chow_day = pd.Timestamp(chow_date_str).normalize()
            except (TypeError, ValueError):
                chow_day = pd.NaT
            if not pd.isna(chow_day) and chow_day <= max_date:
                wins = _compute_pre_post_windows_from_inflection(chow_day, min_date, max_date)
                if wins:
                    chow_disp = format_chow_date(chow_date_str)
                    tx_list = chow_payload.get("transactions") or []
                    note_detail = ""
                    if tx_list and isinstance(tx_list[0], dict):
                        tx0 = tx_list[0]
                        buyer = str(tx0.get("buyer_org_name") or tx0.get("buyer_dba_name") or "").strip()
                        seller = str(tx0.get("seller_org_name") or "").strip()
                        if buyer or seller:
                            note_detail = f"Buyer: {buyer or '—'}; seller: {seller or '—'}."
                    return jsonify(
                        {
                            "mode": "event",
                            "source": "chow",
                            "kind": "chow",
                            "event_date": chow_date_str,
                            "event_quarter_display": chow_disp,
                            "event_note_html": _pre_post_event_note_html("chow", chow_disp),
                            "note": note_detail,
                            "caution": None,
                            **wins,
                        }
                    )

        inf: Optional[dict] = None
        source: Optional[str] = None
        if provider_info_df is not None and not provider_info_df.empty:
            inf = _detect_pre_post_inflection_provider_info(provider_info_df, PROVNUM)
            if inf:
                source = "provider_info"
        if not inf:
            inf = _detect_pre_post_inflection_pbj_names(global_df)
            if inf:
                source = "pbj"

        if inf:
            cq = inf.get("canonical_quarter")
            inf_day = _first_day_of_cy_quarter(cq if isinstance(cq, str) else None)
            if inf_day is not None and inf_day <= max_date:
                wins = _compute_pre_post_windows_from_inflection(inf_day, min_date, max_date)
                if wins:
                    q_disp = _pre_post_event_quarter_display(cq)
                    return jsonify(
                        {
                            "mode": "event",
                            "source": source,
                            "kind": inf.get("kind"),
                            "quarter": inf.get("quarter_label"),
                            "event_quarter_display": q_disp,
                            "event_note_html": _pre_post_event_note_html(
                                cast(Optional[str], inf.get("kind")),
                                q_disp,
                                cast(Optional[str], inf.get("name_from")),
                                cast(Optional[str], inf.get("name_to")),
                            ),
                            "note": inf.get("note_detail", ""),
                            "caution": None,
                            **wins,
                        }
                    )

        # Default comparison buckets (when no ownership/name inflection is found):
        # pre = 2021-01-01 through 2022-12-31, post = 2023-01-01 through latest PBJ date.
        split_start = pd.Timestamp("2023-01-01")
        pre_start = max(min_date, pd.Timestamp("2021-01-01"))
        pre_end = min(max_date, split_start - pd.Timedelta(days=1))
        post_start = max(min_date, split_start)
        post_end = max_date

        if pre_end < pre_start or post_end < post_start:
            # If dataset does not cover both windows, fall back to half-split by available range.
            midpoint = min_date + ((max_date - min_date) / 2)
            pre_start = min_date
            pre_end = pd.Timestamp(midpoint).normalize()
            post_start = pre_end + pd.Timedelta(days=1)
            post_end = max_date

        return jsonify(
            {
                "mode": "uniform",
                "source": None,
                "kind": None,
                "quarter": None,
                "note": "",
                "caution": None,
                "before_start": pre_start.strftime("%Y-%m-%d"),
                "before_end": pre_end.strftime("%Y-%m-%d"),
                "after_start": post_start.strftime("%Y-%m-%d"),
                "after_end": post_end.strftime("%Y-%m-%d"),
            }
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/api/data_completeness')
def get_data_completeness():
    """Analyze data completeness and identify missing quarters/days"""
    try:
        if global_df is None or (hasattr(global_df, 'empty') and global_df.empty):
            return jsonify({
                'issues': [{'type': 'load_error', 'severity': 'high', 'message': 'Data not loaded'}],
                'total_issues': 1,
                'has_issues': True
            })
        completeness_issues = []
        
        # Build expected quarter list dynamically from loaded data bounds.
        expected_quarters = []
        q_series = global_df["CY_Qtr"].dropna().astype(str)
        q_pat = re.compile(r"^(\d{4})Q([1-4])$")
        q_pairs: list[tuple[int, int]] = []
        for qv in q_series.unique():
            m = q_pat.match(qv)
            if m:
                q_pairs.append((int(m.group(1)), int(m.group(2))))
        if q_pairs:
            start_y, start_q = min(q_pairs)
            end_y, end_q = max(q_pairs)
            y, q = start_y, start_q
            while (y, q) <= (end_y, end_q):
                expected_quarters.append(f"{y}Q{q}")
                q += 1
                if q > 4:
                    q = 1
                    y += 1
        
        # Check for missing quarters
        if 'CY_Qtr' not in global_df.columns:
            return jsonify({
                'issues': [{'type': 'load_error', 'severity': 'high', 'message': 'CY_Qtr column missing'}],
                'total_issues': 1,
                'has_issues': True
            })
        actual_quarters = set(global_df['CY_Qtr'].unique())
        missing_quarters = [q for q in expected_quarters if q not in actual_quarters]
        
        if missing_quarters:
            completeness_issues.append({
                'type': 'missing_quarter',
                'severity': 'high',
                'message': f"Missing {len(missing_quarters)} quarter(s): {', '.join(missing_quarters)}"
            })
        
        # Check for incomplete quarters (missing days)
        for quarter in actual_quarters:
            quarter_data = global_df[global_df['CY_Qtr'] == quarter]
            if len(quarter_data) > 0:
                # Calculate expected days in quarter
                year = int(quarter[:4])
                q_num = int(quarter[5])
                
                if q_num == 1:
                    expected_days = 90  # Jan-Mar
                elif q_num == 2:
                    expected_days = 91  # Apr-Jun
                elif q_num == 3:
                    expected_days = 92  # Jul-Sep
                else:  # q_num == 4
                    expected_days = 92  # Oct-Dec
                
                # Adjust for leap years in Q1
                if q_num == 1 and year % 4 == 0:
                    expected_days = 91
                
                actual_days = len(quarter_data)
                missing_days = expected_days - actual_days
                
                if missing_days > 0:
                    severity = 'high' if missing_days > 10 else 'medium' if missing_days > 5 else 'low'
                    completeness_issues.append({
                        'type': 'incomplete_quarter',
                        'severity': severity,
                        'quarter': quarter,
                        'expected_days': expected_days,
                        'actual_days': actual_days,
                        'missing_days': missing_days,
                        'message': f"{quarter}: {missing_days} missing days ({actual_days}/{expected_days})"
                    })
        
        # Check for data gaps (consecutive missing days)
        df_sorted = global_df.sort_values('WorkDate')
        date_gaps = []
        
        for i in range(1, len(df_sorted)):
            prev_date = df_sorted.iloc[i-1]['WorkDate']
            curr_date = df_sorted.iloc[i]['WorkDate']
            days_diff = (curr_date - prev_date).days
            
            if days_diff > 1:  # Gap of more than 1 day
                gap_start = prev_date + timedelta(days=1)
                gap_end = curr_date - timedelta(days=1)
                gap_days = days_diff - 1
                
                severity = 'high' if gap_days > 7 else 'medium' if gap_days > 3 else 'low'
                date_gaps.append({
                    'start_date': gap_start.strftime('%Y-%m-%d'),
                    'end_date': gap_end.strftime('%Y-%m-%d'),
                    'gap_days': gap_days,
                    'severity': severity
                })
        
        if date_gaps:
            # Group consecutive gaps
            gap_summary = {}
            for gap in date_gaps:
                quarter = global_df[global_df['WorkDate'] == gap['start_date']]['CY_Qtr'].iloc[0] if len(global_df[global_df['WorkDate'] == gap['start_date']]) > 0 else 'Unknown'
                if quarter not in gap_summary:
                    gap_summary[quarter] = []
                gap_summary[quarter].append(gap)
            
            for quarter, gaps in gap_summary.items():
                total_gap_days = sum(gap['gap_days'] for gap in gaps)
                max_severity = max(gap['severity'] for gap in gaps)
                # Skip "Unknown" quarters - they're likely edge cases
                if quarter != 'Unknown':
                    completeness_issues.append({
                        'type': 'data_gaps',
                        'severity': max_severity,
                        'quarter': quarter,
                        'gap_count': len(gaps),
                        'total_gap_days': total_gap_days,
                        'message': f"{quarter}: {len(gaps)} gap(s), {total_gap_days} missing days"
                    })
        
        # Check for missing census data (days with 0 census)
        zero_census_data = global_df[global_df['MDScensus'] == 0]
        if len(zero_census_data) > 0:
            # Group by quarter for better reporting
            zero_census_by_quarter = zero_census_data.groupby('CY_Qtr').size()
            for quarter, count in zero_census_by_quarter.items():
                severity = 'high' if count > 10 else 'medium' if count > 5 else 'low'
                completeness_issues.append({
                    'type': 'zero_census',
                    'severity': severity,
                    'quarter': quarter,
                    'count': int(count),
                    'message': f"{quarter}: {count} days with 0 census reported"
                })
        
        # Check for zero staffing hours when census > 0 (facility reported 0 staffing but had residents)
        zero_staffing_with_census = global_df[(global_df['Total_Staff_Hours'] == 0) & (global_df['MDScensus'] > 0)]
        if len(zero_staffing_with_census) > 0:
            # Group by quarter for better reporting
            zero_staffing_by_quarter = zero_staffing_with_census.groupby('CY_Qtr').size()
            for quarter, count in zero_staffing_by_quarter.items():
                severity = 'high' if count > 10 else 'medium' if count > 5 else 'low'
                completeness_issues.append({
                    'type': 'zero_staffing_with_census',
                    'severity': severity,
                    'quarter': quarter,
                    'count': int(count),
                    'message': f"{quarter}: {count} days with 0 staffing hours but census > 0"
                })
        
        return jsonify({
            'issues': completeness_issues,
            'total_issues': len(completeness_issues),
            'has_issues': len(completeness_issues) > 0
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'issues': [{'type': 'error', 'severity': 'high', 'message': str(e)}],
            'total_issues': 1,
            'has_issues': True
        })

@app.route('/api/day_comparison')
def get_day_comparison():
    """Compare a specific day to other similar days"""
    try:
        target_date = request.args.get('target_date')
        comparison_type = request.args.get('comparison_type', 'month')  # month, quarter, year, custom, same_dow
        
        if not target_date:
            return jsonify({'error': 'target_date is required'})
        
        target_date = pd.to_datetime(target_date)
        target_day = global_df[global_df['WorkDate'] == target_date]
        
        if len(target_day) == 0:
            return jsonify({'error': 'No data found for target date'})
        
        target_record = target_day.iloc[0]
        
        # Get comparison data based on type
        if comparison_type == 'month':
            # Same month, different years
            comparison_data = global_df[
                (global_df['WorkDate'].dt.month == target_date.month) & 
                (global_df['WorkDate'] != target_date)
            ]
        elif comparison_type == 'quarter':
            # Same quarter, different years
            quarter = f"{target_date.year}Q{(target_date.month-1)//3 + 1}"
            comparison_data = global_df[
                (global_df['CY_Qtr'].str.contains(f"Q{(target_date.month-1)//3 + 1}")) & 
                (global_df['WorkDate'] != target_date)
            ]
        elif comparison_type == 'year':
            # Same year, different dates
            comparison_data = global_df[
                (global_df['WorkDate'].dt.year == target_date.year) & 
                (global_df['WorkDate'] != target_date)
            ]
        elif comparison_type == 'same_dow':
            # Same day of week
            comparison_data = global_df[
                (global_df['DayOfWeek'] == target_record['DayOfWeek']) & 
                (global_df['WorkDate'] != target_date)
            ]
        else:  # custom
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            if not start_date or not end_date:
                return jsonify({'error': 'start_date and end_date required for custom comparison'})
            try:
                start_dt = _parse_dashboard_iso_date_arg(start_date, field="start")
                end_day = _parse_dashboard_iso_date_arg(end_date, field="end")
                end_dt = end_day + pd.Timedelta(days=1)
            except ValueError as e:
                return jsonify({'error': str(e)})
            wd_all = pd.to_datetime(global_df["WorkDate"], errors="coerce")
            comparison_data = global_df[
                (wd_all >= start_dt) & (wd_all < end_dt) & (wd_all != target_date)
            ]
        
        # Calculate comparison statistics
        target_data = {
            'census': float(target_record['MDScensus']),
            'rn_hprd': float(target_record['RN_HPRD']),
            'lpn_hprd': float(target_record['LPN_HPRD']),
            'cna_hprd': float(target_record['CNA_HPRD']),
            'total_hprd': float(target_record['Total_Staff_HPRD']),
            'rn_hours': float(target_record['Hrs_RN']),
            'lpn_hours': float(target_record['Hrs_LPN']),
            'cna_hours': float(target_record['Hrs_CNA']),
            'rn_contract_hours': float(target_record['Hrs_RN_ctr']),
            'lpn_contract_hours': float(target_record['Hrs_LPN_ctr']),
            'cna_contract_hours': float(target_record['Hrs_CNA_ctr']),
            'rn_contract_pct': float(target_record['RN_Contract_Pct']),
            'lpn_contract_pct': float(target_record['LPN_Contract_Pct']),
            'cna_contract_pct': float(target_record['CNA_Contract_Pct'])
        }
        
        comparison_data = {
            'count': len(comparison_data),
            'avg_census': float(comparison_data['MDScensus'].mean()) if len(comparison_data) > 0 else 0,
            'avg_rn_hprd': float(comparison_data['RN_HPRD'].mean()) if len(comparison_data) > 0 else 0,
            'avg_lpn_hprd': float(comparison_data['LPN_HPRD'].mean()) if len(comparison_data) > 0 else 0,
            'avg_cna_hprd': float(comparison_data['CNA_HPRD'].mean()) if len(comparison_data) > 0 else 0,
            'avg_total_hprd': float(comparison_data['Total_Staff_HPRD'].mean()) if len(comparison_data) > 0 else 0,
            'avg_rn_hours': float(comparison_data['Hrs_RN'].mean()) if len(comparison_data) > 0 else 0,
            'avg_lpn_hours': float(comparison_data['Hrs_LPN'].mean()) if len(comparison_data) > 0 else 0,
            'avg_cna_hours': float(comparison_data['Hrs_CNA'].mean()) if len(comparison_data) > 0 else 0,
            'avg_rn_contract_hours': float(comparison_data['Hrs_RN_ctr'].mean()) if len(comparison_data) > 0 else 0,
            'avg_lpn_contract_hours': float(comparison_data['Hrs_LPN_ctr'].mean()) if len(comparison_data) > 0 else 0,
            'avg_cna_contract_hours': float(comparison_data['Hrs_CNA_ctr'].mean()) if len(comparison_data) > 0 else 0,
            'avg_rn_contract_pct': float(comparison_data['RN_Contract_Pct'].mean()) if len(comparison_data) > 0 else 0,
            'avg_lpn_contract_pct': float(comparison_data['LPN_Contract_Pct'].mean()) if len(comparison_data) > 0 else 0,
            'avg_cna_contract_pct': float(comparison_data['CNA_Contract_Pct'].mean()) if len(comparison_data) > 0 else 0,
            'std_rn_hprd': float(comparison_data['RN_HPRD'].std()) if len(comparison_data) > 0 else 0,
            'std_lpn_hprd': float(comparison_data['LPN_HPRD'].std()) if len(comparison_data) > 0 else 0,
            'std_cna_hprd': float(comparison_data['CNA_HPRD'].std()) if len(comparison_data) > 0 else 0,
            'std_total_hprd': float(comparison_data['Total_Staff_HPRD'].std()) if len(comparison_data) > 0 else 0,
            'std_census': float(comparison_data['MDScensus'].std()) if len(comparison_data) > 0 else 0
        }
        
        # Detect aberrations using statistical analysis
        aberrations = detect_aberrations(target_data, comparison_data)
        
        comparison_stats = {
            'target_date': target_date.strftime('%Y-%m-%d'),
            'target_day_of_week': target_record['DayOfWeek'],
            'target_data': target_data,
            'comparison_data': comparison_data,
            'aberrations': aberrations
        }
        
        return jsonify(comparison_stats)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/pbj_source_link')
def get_pbj_source_link():
    """Get PBJ source link for a specific date and quarter."""
    try:
        target_date = request.args.get('date')
        quarter = request.args.get('quarter')
        provnum = request.args.get('provnum', '225500')
        
        if not target_date or not quarter:
            return jsonify({'error': 'date and quarter parameters are required'})
        
        # Generate the source link
        source_link = format_pbj_source_link(quarter, target_date, provnum)
        
        if source_link:
            return jsonify({
                'source_link': source_link,
                'url': generate_pbj_source_link(quarter, target_date, provnum)
            })
        else:
            return jsonify({'error': 'Could not generate source link'})
            
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route("/api/report_builder/preview", methods=["POST"])
def api_report_builder_preview():
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400
        built = _report_builder_generate_html(cast(dict[str, Any], payload))
        return jsonify(
            {
                "success": True,
                "html": built["html"],
                "file_name": built["file_name"],
                "resolved_start_date": built["resolved_start_date"],
                "resolved_end_date": built["resolved_end_date"],
                "warnings": built["warnings"],
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Report preview failed: {exc}"}), 500


@app.route("/api/report_builder/download_html", methods=["POST"])
def api_report_builder_download_html():
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400
        built = _report_builder_generate_html(cast(dict[str, Any], payload))
        response = Response(str(built["html"]), mimetype="text/html")
        response.headers["Content-Disposition"] = f"attachment; filename={built['file_name']}"
        return response
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Report download failed: {exc}"}), 500


@app.route("/api/report_builder_v3/preview", methods=["POST"])
def api_report_builder_v3_preview():
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400
        built = _report_builder_v3_generate_html(cast(dict[str, Any], payload))
        return jsonify(
            {
                "success": True,
                "html": built["html"],
                "file_name": built["file_name"],
                "resolved_start_date": built["resolved_start_date"],
                "resolved_end_date": built["resolved_end_date"],
                "warnings": built["warnings"],
                "staffing_findings": built.get("staffing_findings") or [],
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except ImportError as exc:
        return jsonify({"error": f"Report builder v3 module missing: {exc}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Report preview failed: {exc}"}), 500


@app.route("/api/report_builder_v3/download_html", methods=["POST"])
def api_report_builder_v3_download_html():
    try:
        payload = request.get_json(silent=True) or {}
        if not isinstance(payload, dict):
            return jsonify({"error": "Invalid JSON payload"}), 400
        built = _report_builder_v3_generate_html(cast(dict[str, Any], payload))
        response = Response(str(built["html"]), mimetype="text/html")
        response.headers["Content-Disposition"] = f"attachment; filename={built['file_name']}"
        return response
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Report download failed: {exc}"}), 500


@app.route('/api/single_day_report')
def api_single_day_report():
    """Get comprehensive single day report with comparisons and aberrations"""
    try:
        target_date = request.args.get('date')
        if not target_date:
            return jsonify({'error': 'Date parameter required'})
        target_date = str(target_date).strip()[:10]
        if len(target_date) != 10 or target_date[4] != "-" or target_date[7] != "-":
            return jsonify({'error': 'Date must be YYYY-MM-DD'})

        # Convert date string to datetime (normalize WorkDate column — may be datetime64, date, or string)
        from datetime import datetime

        target_ts = pd.Timestamp(target_date).normalize()
        wt = pd.to_datetime(global_df["WorkDate"], errors="coerce").dt.normalize()
        target_data = global_df.loc[wt == target_ts]
        if target_data.empty:
            return jsonify({'error': f'No data found for {target_date}'})
        
        target_row = target_data.iloc[0]
        target_quarter = target_row["CY_Qtr"]
        target_year = int(target_ts.year)
        target_dow = target_row["DayOfWeek"]
        
        # Get comparison data
        quarter_data = global_df[global_df['CY_Qtr'] == target_quarter]
        year_data = global_df[global_df['WorkDate'].dt.year == target_year]
        # For day-of-week comparison, use all days of the same week day in the target year only
        dow_year_data = global_df[
            (global_df['DayOfWeek'] == target_dow) & 
            (global_df['WorkDate'].dt.year == target_year)
        ]
        
        # Calculate metrics for target day
        target_metrics = {
            'date': target_date,
            'day_of_week': target_dow,
            'quarter': target_quarter,
            'year': target_year,
            'census': round_financial(target_row['MDScensus']),
            'rn_hours': round_financial(target_row['Hrs_RN']),
            'rn_hprd': round_financial(target_row['RN_HPRD']),
            'lpn_hours': round_financial(target_row['Hrs_LPN']),
            'lpn_hprd': round_financial(target_row['LPN_HPRD']),
            'cna_hours': round_financial(target_row['Hrs_CNA']),
            'cna_hprd': round_financial(target_row['CNA_HPRD']),
            'indirect_staffing_hours': round_financial(target_row['Hrs_RNadmin'] + target_row['Hrs_RNDON'] + target_row['Hrs_LPNadmin']),
            'indirect_staffing_hprd': round_financial((target_row['Hrs_RNadmin'] + target_row['Hrs_RNDON'] + target_row['Hrs_LPNadmin']) / target_row['MDScensus'] if target_row['MDScensus'] > 0 else 0),
            'total_rn_hours': round_financial(target_row['Total_RN_Hours']),
            'total_rn_hprd': round_financial(target_row['Total_RN_HPRD']),
            'total_lpn_hours': round_financial(target_row['Total_LPN_Hours']),
            'total_lpn_hprd': round_financial(target_row['Total_LPN_HPRD']),
            'total_nurse_aide_hours': round_financial(target_row['Total_Nurse_Aide_Hours']),
            'total_nurse_aide_hprd': round_financial(target_row['Total_Nurse_Aide_HPRD']),
            'nurse_staff_hours_excl_admin': round_financial(target_row['Nurse_Staff_Hours_Excl_Admin']),
            'nurse_staff_hprd_excl_admin': round_financial(target_row['Nurse_Staff_HPRD_Excl_Admin']),
            'total_staff_hours': round_financial(target_row['Total_Staff_Hours']),
            'total_staff_hprd': round_financial(target_row['Total_Staff_HPRD']),
            'rn_contract_pct': round_financial(target_row['RN_Contract_Pct']),
            'lpn_contract_pct': round_financial(target_row['LPN_Contract_Pct']),
            'cna_contract_pct': round_financial(target_row['CNA_Contract_Pct']),
            'total_contract_pct': round_financial(target_row['Total_Contract_Pct']),
            'cna_only_contract_pct': round_financial(target_row['CNA_Only_Contract_Pct']),
            'nurse_aide_contract_pct': round_financial(target_row['Nurse_Aide_Contract_Pct']),
            'lpn_only_contract_pct': round_financial(target_row['LPN_Only_Contract_Pct']),
            'total_lpn_contract_pct': round_financial(target_row['Total_LPN_Contract_Pct']),
            'is_holiday': bool(target_row['IsHoliday'])
        }
        _merge_nonnurse_hours_into_target_metrics(target_metrics, target_date)

        # Calculate comparison averages with weighted HPRD calculations
        def calculate_comparison_metrics(data, label):
            if data.empty:
                return None
            
            # Calculate weighted HPRD (sum of hours / sum of census)
            total_census = data['MDScensus'].sum()
            total_rn_hours = data['Hrs_RN'].sum()
            total_lpn_hours = data['Hrs_LPN'].sum()
            total_cna_hours = data['Hrs_CNA'].sum()
            total_rn_all_hours = data['Total_RN_Hours'].sum()
            total_lpn_all_hours = data['Total_LPN_Hours'].sum()
            total_nurse_aide_hours = data['Total_Nurse_Aide_Hours'].sum()
            nurse_staff_hours_excl_admin = data['Nurse_Staff_Hours_Excl_Admin'].sum()
            total_staff_hours = data['Total_Staff_Hours'].sum()
            indirect_staffing_hours = (data['Hrs_RNadmin'].sum() + 
                                       data['Hrs_RNDON'].sum() + 
                                       data['Hrs_LPNadmin'].sum())
            
            # Calculate weighted HPRD values
            rn_hprd_weighted = (total_rn_hours / total_census) if total_census > 0 else 0
            lpn_hprd_weighted = (total_lpn_hours / total_census) if total_census > 0 else 0
            cna_hprd_weighted = (total_cna_hours / total_census) if total_census > 0 else 0
            total_rn_hprd_weighted = (total_rn_all_hours / total_census) if total_census > 0 else 0
            total_lpn_hprd_weighted = (total_lpn_all_hours / total_census) if total_census > 0 else 0
            total_nurse_aide_hprd_weighted = (total_nurse_aide_hours / total_census) if total_census > 0 else 0
            nurse_staff_hprd_excl_admin_weighted = (nurse_staff_hours_excl_admin / total_census) if total_census > 0 else 0
            total_staff_hprd_weighted = (total_staff_hours / total_census) if total_census > 0 else 0
            indirect_staffing_hprd_weighted = (indirect_staffing_hours / total_census) if total_census > 0 else 0
            
            return {
                'label': label,
                'count': len(data),
                'census': round_financial(data['MDScensus'].mean()),
                'rn_hours': round_financial(data['Hrs_RN'].mean()),
                'rn_hprd': round_financial(rn_hprd_weighted),
                'lpn_hours': round_financial(data['Hrs_LPN'].mean()),
                'lpn_hprd': round_financial(lpn_hprd_weighted),
                'cna_hours': round_financial(data['Hrs_CNA'].mean()),
                'cna_hprd': round_financial(cna_hprd_weighted),
                'indirect_staffing_hours': round_financial((data['Hrs_RNadmin'].mean() + data['Hrs_RNDON'].mean() + data['Hrs_LPNadmin'].mean())),
                'indirect_staffing_hprd': round_financial(indirect_staffing_hprd_weighted),
                'total_rn_hours': round_financial(data['Total_RN_Hours'].mean()),
                'total_rn_hprd': round_financial(total_rn_hprd_weighted),
                'total_lpn_hours': round_financial(data['Total_LPN_Hours'].mean()),
                'total_lpn_hprd': round_financial(total_lpn_hprd_weighted),
                'total_nurse_aide_hours': round_financial(data['Total_Nurse_Aide_Hours'].mean()),
                'total_nurse_aide_hprd': round_financial(total_nurse_aide_hprd_weighted),
                'nurse_staff_hours_excl_admin': round_financial(data['Nurse_Staff_Hours_Excl_Admin'].mean()),
                'nurse_staff_hprd_excl_admin': round_financial(nurse_staff_hprd_excl_admin_weighted),
                'total_staff_hours': round_financial(data['Total_Staff_Hours'].mean()),
                'total_staff_hprd': round_financial(total_staff_hprd_weighted),
                'rn_contract_pct': round_financial(data['RN_Contract_Pct'].mean()),
                'lpn_contract_pct': round_financial(data['LPN_Contract_Pct'].mean()),
                'cna_contract_pct': round_financial(data['CNA_Contract_Pct'].mean()),
                'total_contract_pct': round_financial(data['Total_Contract_Pct'].mean()),
                'cna_only_contract_pct': round_financial(data['CNA_Only_Contract_Pct'].mean()),
                'nurse_aide_contract_pct': round_financial(data['Nurse_Aide_Contract_Pct'].mean()),
                'lpn_only_contract_pct': round_financial(data['LPN_Only_Contract_Pct'].mean()),
                'total_lpn_contract_pct': round_financial(data['Total_LPN_Contract_Pct'].mean())
            }
        
        comparisons = {
            'quarter': calculate_comparison_metrics(quarter_data, f"Quarter {target_quarter}"),
            'year': calculate_comparison_metrics(year_data, f"Year {target_year}"),
            'dow': calculate_comparison_metrics(dow_year_data, f"{target_dow}s in {target_year}")
        }
        
        # Calculate aberrations (z-scores)
        def calculate_aberrations(target_val, comparison_data, metric_name, data_source):
            if comparison_data is None or comparison_data['count'] < 2:
                return None
            
            # Get the actual data for this metric
            if metric_name == 'census':
                values = data_source['MDScensus'].dropna()
            elif metric_name == 'rn_hours':
                values = data_source['Hrs_RN'].dropna()
            elif metric_name == 'rn_hprd':
                values = data_source['RN_HPRD'].dropna()
            elif metric_name == 'lpn_hours':
                values = data_source['Hrs_LPN'].dropna()
            elif metric_name == 'lpn_hprd':
                values = data_source['LPN_HPRD'].dropna()
            elif metric_name == 'cna_hours':
                values = data_source['Hrs_CNA'].dropna()
            elif metric_name == 'cna_hprd':
                values = data_source['CNA_HPRD'].dropna()
            elif metric_name == 'total_rn_hours':
                values = data_source['Total_RN_Hours'].dropna()
            elif metric_name == 'total_rn_hprd':
                values = data_source['Total_RN_HPRD'].dropna()
            elif metric_name == 'total_lpn_hours':
                values = data_source['Total_LPN_Hours'].dropna()
            elif metric_name == 'total_lpn_hprd':
                values = data_source['Total_LPN_HPRD'].dropna()
            elif metric_name == 'total_nurse_aide_hours':
                values = data_source['Total_Nurse_Aide_Hours'].dropna()
            elif metric_name == 'total_nurse_aide_hprd':
                values = data_source['Total_Nurse_Aide_HPRD'].dropna()
            elif metric_name == 'nurse_staff_hours_excl_admin':
                values = data_source['Nurse_Staff_Hours_Excl_Admin'].dropna()
            elif metric_name == 'nurse_staff_hprd_excl_admin':
                values = data_source['Nurse_Staff_HPRD_Excl_Admin'].dropna()
            elif metric_name == 'total_staff_hours':
                values = data_source['Total_Staff_Hours'].dropna()
            elif metric_name == 'total_staff_hprd':
                values = data_source['Total_Staff_HPRD'].dropna()
            elif metric_name == 'rn_contract_pct':
                values = data_source['RN_Contract_Pct'].dropna()
            elif metric_name == 'lpn_contract_pct':
                values = data_source['LPN_Contract_Pct'].dropna()
            elif metric_name == 'cna_contract_pct':
                values = data_source['CNA_Contract_Pct'].dropna()
            else:
                return None
            
            if len(values) < 2:
                return None
            
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val == 0:
                return None
            
            z_score = (target_val - mean_val) / std_val
            
            # Calculate actual percentile ranking
            sorted_values = values.sort_values()
            rank = (sorted_values < target_val).sum() + 1
            total_days = len(values)
            percentile = (rank / total_days) * 100
            
            # Determine if it's an outlier and get ranking info
            is_outlier = False
            ranking_info = ""
            
            if 'contract' in metric_name:
                # For contract percentages, both high and low are concerning
                if abs(z_score) > 1.5:
                    color = 'red'
                    is_outlier = True
                    if z_score < 0:
                        ranking_info = f"{rank} lowest of {total_days} days"
                    else:
                        ranking_info = f"{rank} highest of {total_days} days"
                else:
                    color = 'normal'
            else:
                # For other metrics, low values are red, high values are green
                if z_score < -1.5:
                    color = 'red'
                    is_outlier = True
                    ranking_info = f"{rank} lowest of {total_days} days"
                elif z_score > 1.5:
                    color = 'green'
                    is_outlier = True
                    ranking_info = f"{rank} highest of {total_days} days"
                else:
                    color = 'normal'
            
            return {
                'z_score': round_financial(z_score),
                'color': color,
                'percentile': round_financial(percentile),
                'is_outlier': is_outlier,
                'ranking_info': ranking_info,
                'rank': int(rank),
                'total_days': int(total_days)
            }
        
        # Calculate aberrations for all metrics (quarter, year, and day-of-week based)
        aberrations = {}
        for metric in ['census', 'rn_hours', 'rn_hprd', 'lpn_hours', 'lpn_hprd', 'cna_hours', 'cna_hprd',
                      'total_rn_hours', 'total_rn_hprd', 'total_lpn_hours', 'total_lpn_hprd',
                      'total_nurse_aide_hours', 'total_nurse_aide_hprd', 'nurse_staff_hours_excl_admin', 'nurse_staff_hprd_excl_admin',
                      'total_staff_hours', 'total_staff_hprd', 'rn_contract_pct', 'lpn_contract_pct', 'cna_contract_pct']:
            target_val = target_metrics[metric]
            
            # Calculate quarter-based aberration
            quarter_aberration = calculate_aberrations(target_val, comparisons['quarter'], metric, quarter_data)
            # Calculate year-based aberration
            year_aberration = calculate_aberrations(target_val, comparisons['year'], metric, year_data)
            # Calculate day-of-week-based aberration
            dow_aberration = calculate_aberrations(target_val, comparisons['dow'], metric, dow_year_data)
            
            aberrations[metric] = {
                'quarter': quarter_aberration,
                'year': year_aberration,
                'dow': dow_aberration
            }
        
        # Get the actual facility provider number
        facility_provnum = str(target_row['PROVNUM']).zfill(6) if 'PROVNUM' in target_row else "Unknown"
        
        # Generate PBJ source links with actual provider number
        nurse_source_link = format_pbj_source_link(target_quarter, target_date, facility_provnum, "nurse")
        nonnurse_source_link = format_pbj_source_link(target_quarter, target_date, facility_provnum, "nonnurse")

        # Illustrative shift split when PBJ has no shift columns (same shares as PBJ320 staffing memos).
        _RN_LPN_SHARES = (0.410, 0.316, 0.274)
        _NA_SHARES = (0.381, 0.341, 0.278)
        shift_projection = None
        try:
            cen_sp = float(target_row["MDScensus"]) if float(target_row["MDScensus"] or 0) > 0 else 0.0
            rn_lpn_pool = float(target_row["Total_RN_Hours"]) + float(target_row["Total_LPN_Hours"])
            na_pool = float(target_row["Total_Nurse_Aide_Hours"])
            direct_rn_lpn_pool = float(target_row["Hrs_RN"]) + float(target_row["Hrs_LPN"])
            direct_na_pool = (
                float(target_row["Hrs_CNA"])
                + float(target_row.get("Hrs_NAtrn", 0) or 0)
                + float(target_row.get("Hrs_MedAide", 0) or 0)
            )
            shift_projection = {
                "note": (
                    "PBJ reports calendar-day hours only—no shift lines. For context, published shift shares "
                    "(Ohio NH study; JAMDA 2024) are applied separately to the RN+LPN hour pool and the "
                    "nurse-aide hour pool. This is not CMS-reported, not facility-specific, and not for "
                    "regulatory or legal use."
                ),
                "citation_url": "https://www.sciencedirect.com/science/article/pii/S1525861024006765",
                "share_labels": {
                    "rn_lpn": "RN+LPN (total hours): 41.0% day · 31.6% evening · 27.4% overnight",
                    "nurse_aide": "Nurse aides (total aide hours): 38.1% day · 34.1% evening · 27.8% overnight",
                },
                "census": round_financial(cen_sp, 2) if cen_sp > 0 else None,
                "rn_lpn_hours_total": round_financial(rn_lpn_pool, 2),
                "nurse_aide_hours_total": round_financial(na_pool, 2),
                "direct_rn_lpn_hours_total": round_financial(direct_rn_lpn_pool, 2),
                "direct_nurse_aide_hours_total": round_financial(direct_na_pool, 2),
                "rn_lpn_hours": {
                    "Day": round_financial(rn_lpn_pool * _RN_LPN_SHARES[0]),
                    "Evening": round_financial(rn_lpn_pool * _RN_LPN_SHARES[1]),
                    "Overnight": round_financial(rn_lpn_pool * _RN_LPN_SHARES[2]),
                },
                "nurse_aide_hours": {
                    "Day": round_financial(na_pool * _NA_SHARES[0]),
                    "Evening": round_financial(na_pool * _NA_SHARES[1]),
                    "Overnight": round_financial(na_pool * _NA_SHARES[2]),
                },
                "rn_lpn_hprd": {
                    "Day": round_financial((rn_lpn_pool * _RN_LPN_SHARES[0]) / cen_sp, 3) if cen_sp > 0 else None,
                    "Evening": round_financial((rn_lpn_pool * _RN_LPN_SHARES[1]) / cen_sp, 3) if cen_sp > 0 else None,
                    "Overnight": round_financial((rn_lpn_pool * _RN_LPN_SHARES[2]) / cen_sp, 3) if cen_sp > 0 else None,
                },
                "nurse_aide_hprd": {
                    "Day": round_financial((na_pool * _NA_SHARES[0]) / cen_sp, 3) if cen_sp > 0 else None,
                    "Evening": round_financial((na_pool * _NA_SHARES[1]) / cen_sp, 3) if cen_sp > 0 else None,
                    "Overnight": round_financial((na_pool * _NA_SHARES[2]) / cen_sp, 3) if cen_sp > 0 else None,
                },
                # Per band: RN+LPN HPRD + aide HPRD using each pool’s own shift shares. Three bands sum to
                # Total_Staff_HPRD (total pools) or Nurse_Staff_HPRD_Excl_Admin (direct pools).
                "total_combined_hprd": {
                    "Day": round_financial(
                        (rn_lpn_pool * _RN_LPN_SHARES[0] + na_pool * _NA_SHARES[0]) / cen_sp, 3
                    )
                    if cen_sp > 0
                    else None,
                    "Evening": round_financial(
                        (rn_lpn_pool * _RN_LPN_SHARES[1] + na_pool * _NA_SHARES[1]) / cen_sp, 3
                    )
                    if cen_sp > 0
                    else None,
                    "Overnight": round_financial(
                        (rn_lpn_pool * _RN_LPN_SHARES[2] + na_pool * _NA_SHARES[2]) / cen_sp, 3
                    )
                    if cen_sp > 0
                    else None,
                },
                "direct_combined_hprd": {
                    "Day": round_financial(
                        (direct_rn_lpn_pool * _RN_LPN_SHARES[0] + direct_na_pool * _NA_SHARES[0]) / cen_sp, 3
                    )
                    if cen_sp > 0
                    else None,
                    "Evening": round_financial(
                        (direct_rn_lpn_pool * _RN_LPN_SHARES[1] + direct_na_pool * _NA_SHARES[1]) / cen_sp, 3
                    )
                    if cen_sp > 0
                    else None,
                    "Overnight": round_financial(
                        (direct_rn_lpn_pool * _RN_LPN_SHARES[2] + direct_na_pool * _NA_SHARES[2]) / cen_sp, 3
                    )
                    if cen_sp > 0
                    else None,
                },
                "reported_calendar_hprd": {
                    "total_staff": round_financial(float(target_row["Total_Staff_HPRD"]), 3)
                    if cen_sp > 0
                    else None,
                    "nurse_direct_excl_admin": round_financial(
                        float(target_row["Nurse_Staff_HPRD_Excl_Admin"]), 3
                    )
                    if cen_sp > 0
                    else None,
                },
            }
        except Exception:
            shift_projection = None
        
        return jsonify({
            'target_metrics': target_metrics,
            'comparisons': comparisons,
            'aberrations': aberrations,
            'nurse_source_link': nurse_source_link,
            'nonnurse_source_link': nonnurse_source_link,
            'shift_projection': shift_projection,
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def get_filter_description(start_date, end_date, quarter, day_of_week, holidays_only):
    """Generate a human-readable description of current filters for chart titles"""
    def format_date(date_str):
        """Convert YYYY-MM-DD to MM-DD-YYYY"""
        if not date_str:
            return ""
        try:
            from datetime import datetime
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.strftime('%m-%d-%Y')
        except:
            return date_str
    
    def format_quarter(q):
        """Convert 2023Q1 to Q1 2023"""
        if not q:
            return ""
        try:
            year = q[:4]
            q_num = q[5]
            return f"Q{q_num} {year}"
        except:
            return q

    def _quarter_sort_key(q):
        q = str(q).strip()
        if 'Q' not in q:
            return (0, 0)
        year, qn = q.split('Q', 1)
        try:
            return (int(year), int(qn))
        except ValueError:
            return (0, 0)
    
    # Build the filter description
    date_part = ""
    filter_part = ""
    
    if start_date and end_date:
        start_formatted = format_date(start_date)
        end_formatted = format_date(end_date)
        date_part = f"{start_formatted} to {end_formatted}"
    elif start_date:
        date_part = f"From {format_date(start_date)}"
    elif end_date:
        date_part = f"Until {format_date(end_date)}"
    
    filters = []
    
    def _is_full_calendar_year_quarters(quarter_keys):
        if len(quarter_keys) != 4:
            return False
        years = set()
        qnums = []
        for q in quarter_keys:
            q = str(q).strip()
            if 'Q' not in q:
                return False
            year, qn = q.split('Q', 1)
            if not year.isdigit() or not qn.isdigit():
                return False
            years.add(year)
            qnums.append(int(qn))
        return len(years) == 1 and sorted(qnums) == [1, 2, 3, 4]

    if quarter != 'all':
        # Format quarters nicely (always chronological for range labels)
        quarters = sorted(
            [q.strip() for q in quarter.split(',') if q.strip()],
            key=_quarter_sort_key,
        )
        if len(quarters) == 1:
            filters.append(f"{format_quarter(quarters[0])}")
        elif _is_full_calendar_year_quarters(quarters):
            filters.append(quarters[0][:4])
        else:
            # Create a range for multiple quarters
            formatted_quarters = [format_quarter(q) for q in quarters]
            if len(formatted_quarters) > 3:
                # Show range for many quarters
                first_quarter = formatted_quarters[0]
                last_quarter = formatted_quarters[-1]
                filters.append(f"{first_quarter} - {last_quarter}")
            else:
                # Show all quarters if 3 or fewer
                filters.append(f"{', '.join(formatted_quarters)}")
    
    if day_of_week != 'all':
        filters.append(f"Day: {day_of_week}")
    
    if holidays_only:
        filters.append("Holidays Only")
    
    if filters:
        filter_part = " | ".join(filters)
    
    # Create two-line title
    if date_part and filter_part:
        return f"{date_part}<br>{filter_part}"
    elif date_part:
        return date_part
    elif filter_part:
        return filter_part
    else:
        # Dynamic default label from loaded PBJ range.
        try:
            if global_df is not None and len(global_df) > 0 and "WorkDate" in global_df.columns:
                _wd = pd.to_datetime(global_df["WorkDate"], errors="coerce").dropna()
                if len(_wd) > 0:
                    y0 = int(_wd.min().year)
                    y1 = int(_wd.max().year)
                    return f"All Data ({y0}-{y1})"
        except Exception:
            pass
        return "All Data"

def detect_aberrations(target_data, comparison_data):
    """Detect statistical aberrations in the target day compared to historical data"""
    aberrations = []
    
    # Define metrics to analyze
    metrics = [
        ('census', 'Census'),
        ('rn_hprd', 'RN HPRD'),
        ('lpn_hprd', 'LPN HPRD'),
        ('cna_hprd', 'CNA HPRD'),
        ('total_hprd', 'Total HPRD'),
        ('rn_contract_pct', 'RN Contract %'),
        ('lpn_contract_pct', 'LPN Contract %'),
        ('cna_contract_pct', 'CNA Contract %')
    ]
    
    for metric_key, metric_name in metrics:
        target_value = target_data[metric_key]
        avg_value = comparison_data[f'avg_{metric_key}']
        std_value = comparison_data.get(f'std_{metric_key}', 0)
        
        if std_value > 0:  # Only analyze if we have standard deviation data
            # Calculate z-score (how many standard deviations from mean)
            z_score = (target_value - avg_value) / std_value
            
            # Determine aberration level
            if abs(z_score) >= 3.0:
                severity = 'EXTREME'
                color = 'danger'
            elif abs(z_score) >= 2.0:
                severity = 'HIGH'
                color = 'warning'
            elif abs(z_score) >= 1.5:
                severity = 'MODERATE'
                color = 'info'
            else:
                continue  # Not significant enough to report
            
            # Determine direction
            direction = 'HIGH' if z_score > 0 else 'LOW'
            
            # Calculate percentile
            if z_score > 0:
                percentile = min(99.9, 50 + (z_score * 34.1))  # Approximate percentile
            else:
                percentile = max(0.1, 50 - (abs(z_score) * 34.1))
            
            aberrations.append({
                'metric': metric_name,
                'target_value': target_value,
                'average_value': avg_value,
                'z_score': z_score,
                'severity': severity,
                'direction': direction,
                'percentile': percentile,
                'color': color,
                'description': f"{metric_name} was {direction} ({target_value:.3f} vs avg {avg_value:.3f}, {percentile:.1f}th percentile)"
            })
    
    # Sort by severity and z-score
    severity_order = {'EXTREME': 4, 'HIGH': 3, 'MODERATE': 2}
    aberrations.sort(key=lambda x: (severity_order.get(x['severity'], 1), abs(x['z_score'])), reverse=True)
    
    return aberrations

@app.route('/api/quarterly-stats')
def get_quarterly_stats():
    """Get comprehensive quarterly statistics for all positions"""
    try:
        # Get all filter parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        quarter = request.args.get('quarter', 'all')
        year = request.args.get('year', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        show_holidays_only = request.args.get('holidays_only', 'false') == 'true'
        
        # Filter data by all parameters (aligned with /api/data and /api/charts)
        global global_df
        if global_df is None or len(global_df) == 0:
            return jsonify({"error": "No data loaded", "quarterly_stats": {}})
        gdf_qstats = cast(pd.DataFrame, global_df)
        _ensure_pbj_dashboard_derived_columns_inplace(gdf_qstats)
        try:
            filtered_df = _filter_facility_daily_cached(
                gdf_qstats,
                start_date=start_date,
                end_date=end_date,
                day_of_week=day_of_week,
                quarter=quarter,
                year=year,
                holidays_only=show_holidays_only,
            )
        except ValueError as e:
            return jsonify({"error": str(e), "quarterly_stats": {}}), 400
        
        # Group data by quarter
        quarterly_data = filtered_df.groupby('CY_Qtr').agg({
            # Total Staff
            'Total_Staff_Hours': ['sum', 'mean', 'median', 'std'],
            'Total_Staff_HPRD': ['mean', 'median', 'std'],
            
            # Nurse Staff (excluding admin)
            'Nurse_Staff_Hours_Excl_Admin': ['sum', 'mean', 'median', 'std'],
            'Nurse_Staff_HPRD_Excl_Admin': ['mean', 'median', 'std'],
            
            # Total RN (including admin and DON)
            'Total_RN_Hours': ['sum', 'mean', 'median', 'std'],
            'Total_RN_HPRD': ['mean', 'median', 'std'],
            
            # RN Direct Care
            'Hrs_RN': ['sum', 'mean', 'median', 'std'],
            'RN_HPRD': ['mean', 'median', 'std'],
            
            # RN Admin
            'Hrs_RNadmin': ['sum', 'mean', 'median', 'std'],
            
            # RN DON
            'Hrs_RNDON': ['sum', 'mean', 'median', 'std'],
            
            # Total LPN (including admin)
            'Total_LPN_Hours': ['sum', 'mean', 'median', 'std'],
            'Total_LPN_HPRD': ['mean', 'median', 'std'],
            
            # LPN Direct Care
            'Hrs_LPN': ['sum', 'mean', 'median', 'std'],
            'LPN_HPRD': ['mean', 'median', 'std'],
            
            # LPN Admin
            'Hrs_LPNadmin': ['sum', 'mean', 'median', 'std'],
            
            # Total CNA (including trainees and med aides)
            'Total_Nurse_Aide_Hours': ['sum', 'mean', 'median', 'std'],
            'Total_Nurse_Aide_HPRD': ['mean', 'median', 'std'],
            
            # CNA Direct Care
            'Hrs_CNA': ['sum', 'mean', 'median', 'std'],
            'CNA_HPRD': ['mean', 'median', 'std'],
            
            # NA Trainee
            'Hrs_NAtrn': ['sum', 'mean', 'median', 'std'],
            
            # Med Aide
            'Hrs_MedAide': ['sum', 'mean', 'median', 'std'],
            
            # Contract Staff
            'Hrs_RN_ctr': ['sum', 'mean', 'median', 'std'],
            'Hrs_LPN_ctr': ['sum', 'mean', 'median', 'std'],
            'Hrs_CNA_ctr': ['sum', 'mean', 'median', 'std'],
            
            # Census for HPRD calculations
            'MDScensus': ['mean', 'median', 'std']
        }).round(3)
        
        # Flatten column names
        quarterly_data.columns = ['_'.join(col).strip() for col in quarterly_data.columns.values]
        
        # Calculate zero counts for each position
        zero_counts = {}
        position_columns = {
            'total_staff': 'Total_Staff_Hours',
            'total_rn': 'Total_RN_Hours', 
            'rn_direct': 'Hrs_RN',
            'rn_admin': 'Hrs_RNadmin',
            'rn_don': 'Hrs_RNDON',
            'total_lpn': 'Total_LPN_Hours',
            'lpn_direct': 'Hrs_LPN',
            'lpn_admin': 'Hrs_LPNadmin',
            'total_cna': 'Total_Nurse_Aide_Hours',
            'cna_direct': 'Hrs_CNA',
            'na_trainee': 'Hrs_NAtrn',
            'med_aide': 'Hrs_MedAide',
            'rn_contract': 'Hrs_RN_ctr',
            'lpn_contract': 'Hrs_LPN_ctr',
            'cna_contract': 'Hrs_CNA_ctr'
        }
        
        for pos_key, col_name in position_columns.items():
            if col_name in filtered_df.columns:
                zero_counts[pos_key] = len(filtered_df[filtered_df[col_name] == 0])
            else:
                zero_counts[pos_key] = 0
        
        # Structure the response
        quarterly_stats = {}
        
        # Total Nurse Staff
        quarterly_stats['total_nurse_staff'] = {
            'total_hours': {
                'mean': float(filtered_df['Total_Staff_Hours'].mean()),
                'median': float(filtered_df['Total_Staff_Hours'].median()),
                'std_dev': float(filtered_df['Total_Staff_Hours'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Total_Staff_HPRD'].mean()),
                'median': float(filtered_df['Total_Staff_HPRD'].median()),
                'std_dev': float(filtered_df['Total_Staff_HPRD'].std())
            },
            'zero_count': zero_counts['total_staff']
        }
        
        # Direct Staff (excl. Admin, DON)
        quarterly_stats['direct_staff_excl_admin'] = {
            'total_hours': {
                'mean': float(filtered_df['Nurse_Staff_Hours_Excl_Admin'].mean()),
                'median': float(filtered_df['Nurse_Staff_Hours_Excl_Admin'].median()),
                'std_dev': float(filtered_df['Nurse_Staff_Hours_Excl_Admin'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].mean()),
                'median': float(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].median()),
                'std_dev': float(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].std())
            },
            'zero_count': 0  # Calculate if needed
        }
        
        # Total RN
        quarterly_stats['total_rn'] = {
            'total_hours': {
                'mean': float(filtered_df['Total_RN_Hours'].mean()),
                'median': float(filtered_df['Total_RN_Hours'].median()),
                'std_dev': float(filtered_df['Total_RN_Hours'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Total_RN_HPRD'].mean()),
                'median': float(filtered_df['Total_RN_HPRD'].median()),
                'std_dev': float(filtered_df['Total_RN_HPRD'].std())
            },
            'zero_count': zero_counts['total_rn']
        }
        
        # Direct RN
        quarterly_stats['rn_direct'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RN'].mean()),
                'median': float(filtered_df['Hrs_RN'].median()),
                'std_dev': float(filtered_df['Hrs_RN'].std())
            },
            'hprd': {
                'mean': float(filtered_df['RN_HPRD'].mean()),
                'median': float(filtered_df['RN_HPRD'].median()),
                'std_dev': float(filtered_df['RN_HPRD'].std())
            },
            'zero_count': zero_counts['rn_direct']
        }
        
        # RN Admin
        quarterly_stats['rn_admin'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RNadmin'].mean()),
                'median': float(filtered_df['Hrs_RNadmin'].median()),
                'std_dev': float(filtered_df['Hrs_RNadmin'].std())
            },
            'hprd': {
                'mean': 0.0,  # Admin HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_admin']
        }
        
        # RN DON
        quarterly_stats['rn_don'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RNDON'].mean()),
                'median': float(filtered_df['Hrs_RNDON'].median()),
                'std_dev': float(filtered_df['Hrs_RNDON'].std())
            },
            'hprd': {
                'mean': 0.0,  # DON HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_don']
        }
        
        # Total LPN
        quarterly_stats['total_lpn'] = {
            'total_hours': {
                'mean': float(filtered_df['Total_LPN_Hours'].mean()),
                'median': float(filtered_df['Total_LPN_Hours'].median()),
                'std_dev': float(filtered_df['Total_LPN_Hours'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Total_LPN_HPRD'].mean()),
                'median': float(filtered_df['Total_LPN_HPRD'].median()),
                'std_dev': float(filtered_df['Total_LPN_HPRD'].std())
            },
            'zero_count': zero_counts['total_lpn']
        }
        
        # LPN
        quarterly_stats['lpn_direct'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_LPN'].mean()),
                'median': float(filtered_df['Hrs_LPN'].median()),
                'std_dev': float(filtered_df['Hrs_LPN'].std())
            },
            'hprd': {
                'mean': float(filtered_df['LPN_HPRD'].mean()),
                'median': float(filtered_df['LPN_HPRD'].median()),
                'std_dev': float(filtered_df['LPN_HPRD'].std())
            },
            'zero_count': zero_counts['lpn_direct']
        }
        
        # LPN Admin
        quarterly_stats['lpn_admin'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_LPNadmin'].mean()),
                'median': float(filtered_df['Hrs_LPNadmin'].median()),
                'std_dev': float(filtered_df['Hrs_LPNadmin'].std())
            },
            'hprd': {
                'mean': 0.0,  # Admin HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['lpn_admin']
        }
        
        # Total Nurse Aide
        quarterly_stats['total_cna'] = {
            'total_hours': {
                'mean': float(filtered_df['Total_Nurse_Aide_Hours'].mean()),
                'median': float(filtered_df['Total_Nurse_Aide_Hours'].median()),
                'std_dev': float(filtered_df['Total_Nurse_Aide_Hours'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Total_Nurse_Aide_HPRD'].mean()),
                'median': float(filtered_df['Total_Nurse_Aide_HPRD'].median()),
                'std_dev': float(filtered_df['Total_Nurse_Aide_HPRD'].std())
            },
            'zero_count': zero_counts['total_cna']
        }
        
        # CNA
        quarterly_stats['cna_direct'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_CNA'].mean()),
                'median': float(filtered_df['Hrs_CNA'].median()),
                'std_dev': float(filtered_df['Hrs_CNA'].std())
            },
            'hprd': {
                'mean': float(filtered_df['CNA_HPRD'].mean()),
                'median': float(filtered_df['CNA_HPRD'].median()),
                'std_dev': float(filtered_df['CNA_HPRD'].std())
            },
            'zero_count': zero_counts['cna_direct']
        }
        
        # Med Aide
        quarterly_stats['med_aide'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_MedAide'].mean()),
                'median': float(filtered_df['Hrs_MedAide'].median()),
                'std_dev': float(filtered_df['Hrs_MedAide'].std())
            },
            'hprd': {
                'mean': 0.0,  # Med Aide HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['med_aide']
        }
        
        # NA Trainee
        quarterly_stats['na_trainee'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_NAtrn'].mean()),
                'median': float(filtered_df['Hrs_NAtrn'].median()),
                'std_dev': float(filtered_df['Hrs_NAtrn'].std())
            },
            'hprd': {
                'mean': 0.0,  # Trainee HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['na_trainee']
        }
        
        # Total Contract
        total_contract_hours = filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']
        quarterly_stats['total_contract'] = {
            'total_hours': {
                'mean': float(total_contract_hours.mean()),
                'median': float(total_contract_hours.median()),
                'std_dev': float(total_contract_hours.std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_contract'] + zero_counts['lpn_contract'] + zero_counts['cna_contract']
        }
        
        # Direct Care Contract
        direct_contract_hours = filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']
        quarterly_stats['direct_care_contract'] = {
            'total_hours': {
                'mean': float(direct_contract_hours.mean()),
                'median': float(direct_contract_hours.median()),
                'std_dev': float(direct_contract_hours.std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_contract'] + zero_counts['lpn_contract'] + zero_counts['cna_contract']
        }
        
        # Total RN Contract
        quarterly_stats['total_rn_contract'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RN_ctr'].mean()),
                'median': float(filtered_df['Hrs_RN_ctr'].median()),
                'std_dev': float(filtered_df['Hrs_RN_ctr'].std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_contract']
        }
        
        # Direct RN Contract
        quarterly_stats['direct_rn_contract'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RN_ctr'].mean()),
                'median': float(filtered_df['Hrs_RN_ctr'].median()),
                'std_dev': float(filtered_df['Hrs_RN_ctr'].std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_contract']
        }
        
        # Nurse Aide Contract
        quarterly_stats['nurse_aide_contract'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_CNA_ctr'].mean()),
                'median': float(filtered_df['Hrs_CNA_ctr'].median()),
                'std_dev': float(filtered_df['Hrs_CNA_ctr'].std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['cna_contract']
        }
        
        # Handle NaN values by converting them to None
        def clean_nan_values(obj):
            if isinstance(obj, dict):
                return {k: clean_nan_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan_values(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        cleaned_stats = clean_nan_values(quarterly_stats)
        return jsonify({
            'quarterly_stats': cleaned_stats,
            'sample_size': len(filtered_df)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def _quarterly_data_pbj_groupby(gdf: pd.DataFrame):
    """Group PBJ daily rows by canonical quarter with numeric coercion and valid keys only."""
    if gdf is None or gdf.empty or "CY_Qtr" not in gdf.columns:
        return None
    pbj_canon_series = gdf["CY_Qtr"].map(_provider_quarter_to_canonical)
    valid = pbj_canon_series.notna() & pbj_canon_series.map(
        lambda x: _cy_quarter_sort_key_from_canonical(x) != (0, 0)
    )
    if not bool(valid.any()):
        return None
    work = gdf.loc[valid].copy()
    pbj_canon_series = pbj_canon_series.loc[valid]
    numeric_cols = (
        "MDScensus",
        "Total_Staff_Hours",
        "Total_RN_Hours",
        "Nurse_Staff_Hours_Excl_Admin",
        "Hrs_RN",
        "Hrs_RNadmin",
        "Hrs_RNDON",
        "Total_LPN_Hours",
        "Hrs_LPN",
        "Hrs_LPNadmin",
        "Total_Nurse_Aide_Hours",
        "Hrs_CNA",
        "Hrs_NAtrn",
        "Hrs_MedAide",
        "Total_Contract_Pct",
    )
    for col in numeric_cols:
        if col in work.columns:
            work[col] = pd.to_numeric(_as_1d_series(work[col]), errors="coerce")
    return work.groupby(pbj_canon_series, sort=False)


@app.route('/api/quarterly-data')
def get_quarterly_data():
    """Get quarterly data for all quarters with HPRD and hours"""
    try:
        global global_df
        if global_df is None or global_df.empty:
            return jsonify({'error': 'No data loaded'})
        _ensure_pbj_dashboard_derived_columns_inplace(global_df)

        prov_df_cm = _scoped_provider_info_df_for_facility()
        all_quarters = _all_canonical_quarters_pbj_and_provider()
        if not all_quarters:
            return jsonify({'quarterly_data': {}})

        all_quarters = [
            q for q in all_quarters
            if _cy_quarter_sort_key_from_canonical(q) != (0, 0)
        ]
        pbj_gb = _quarterly_data_pbj_groupby(global_df)

        quarterly_data: dict[str, Any] = {}
        for quarter in sorted(all_quarters, key=_cy_quarter_sort_key_from_canonical):
            prov_rows = _provider_rows_for_canonical_quarter(prov_df_cm, quarter)
            provider_footnote = _quarter_provider_footnote_text(prov_rows)
            quarter_df = pd.DataFrame()
            if pbj_gb is not None and quarter in pbj_gb.groups:
                quarter_df = pbj_gb.get_group(quarter)

            if len(quarter_df) == 0:
                quarterly_data[quarter] = {
                    'census': None,
                    'total_hprd': None,
                    'total_hours': None,
                    'direct_hprd': None,
                    'direct_hours': None,
                    'total_rn_hprd': None,
                    'total_rn_hours': None,
                    'rn_hprd': None,
                    'rn_hours': None,
                    'rn_admin_hprd': None,
                    'rn_admin_hours': None,
                    'rn_don_hprd': None,
                    'rn_don_hours': None,
                    'total_lpn_hprd': None,
                    'total_lpn_hours': None,
                    'lpn_hprd': None,
                    'lpn_hours': None,
                    'lpn_admin_hprd': None,
                    'lpn_admin_hours': None,
                    'total_nurse_aide_hprd': None,
                    'total_nurse_aide_hours': None,
                    'cna_hprd': None,
                    'cna_hours': None,
                    'na_train_hprd': None,
                    'na_train_hours': None,
                    'med_aide_hprd': None,
                    'med_aide_hours': None,
                    'contract_pct': None,
                    'pbj_has_data': False,
                    'pbj_day_count': 0,
                    'provider_footnote': provider_footnote,
                }
                continue

            mdc = pd.to_numeric(_as_1d_series(quarter_df['MDScensus']), errors='coerce') if 'MDScensus' in quarter_df.columns else pd.Series(dtype=float)
            total_census = float(mdc.sum()) if len(mdc) else 0.0
            total_staff_hours = quarter_df['Total_Staff_Hours'].sum() if 'Total_Staff_Hours' in quarter_df.columns else 0
            total_rn_hours = quarter_df['Total_RN_Hours'].sum() if 'Total_RN_Hours' in quarter_df.columns else 0
            nurse_staff_hours = quarter_df['Nurse_Staff_Hours_Excl_Admin'].sum() if 'Nurse_Staff_Hours_Excl_Admin' in quarter_df.columns else 0
            rn_hours = quarter_df['Hrs_RN'].sum() if 'Hrs_RN' in quarter_df.columns else 0
            rn_admin_hours = quarter_df['Hrs_RNadmin'].sum() if 'Hrs_RNadmin' in quarter_df.columns else 0
            rn_don_hours = quarter_df['Hrs_RNDON'].sum() if 'Hrs_RNDON' in quarter_df.columns else 0

            total_lpn_hours = quarter_df['Total_LPN_Hours'].sum() if 'Total_LPN_Hours' in quarter_df.columns else 0
            lpn_admin_hours = quarter_df['Hrs_LPNadmin'].sum() if 'Hrs_LPNadmin' in quarter_df.columns else 0
            lpn_direct_hours = quarter_df['Hrs_LPN'].sum() if 'Hrs_LPN' in quarter_df.columns else 0
            total_nurse_aide_hours = (
                quarter_df['Total_Nurse_Aide_Hours'].sum()
                if 'Total_Nurse_Aide_Hours' in quarter_df.columns
                else 0
            )
            cna_hours = quarter_df['Hrs_CNA'].sum() if 'Hrs_CNA' in quarter_df.columns else 0
            na_train_hours = quarter_df['Hrs_NAtrn'].sum() if 'Hrs_NAtrn' in quarter_df.columns else 0
            med_aide_hours = quarter_df['Hrs_MedAide'].sum() if 'Hrs_MedAide' in quarter_df.columns else 0
            if not total_nurse_aide_hours:
                total_nurse_aide_hours = cna_hours + na_train_hours + med_aide_hours

            total_hprd = (total_staff_hours / total_census) if total_census > 0 else 0
            total_rn_hprd = (total_rn_hours / total_census) if total_census > 0 else 0
            nurse_staff_hprd = (nurse_staff_hours / total_census) if total_census > 0 else 0
            rn_hprd = (rn_hours / total_census) if total_census > 0 else 0
            rn_admin_hprd = (rn_admin_hours / total_census) if total_census > 0 else 0
            rn_don_hprd = (rn_don_hours / total_census) if total_census > 0 else 0

            total_lpn_hprd = (total_lpn_hours / total_census) if total_census > 0 else 0
            lpn_hprd = (lpn_direct_hours / total_census) if total_census > 0 else 0
            lpn_admin_hprd = (lpn_admin_hours / total_census) if total_census > 0 else 0
            total_nurse_aide_hprd = (total_nurse_aide_hours / total_census) if total_census > 0 else 0
            cna_hprd = (cna_hours / total_census) if total_census > 0 else 0
            na_train_hprd = (na_train_hours / total_census) if total_census > 0 else 0
            med_aide_hprd = (med_aide_hours / total_census) if total_census > 0 else 0

            avg_census = quarter_df['MDScensus'].mean() if 'MDScensus' in quarter_df.columns else 0
            avg_staff_hours = quarter_df['Total_Staff_Hours'].mean() if 'Total_Staff_Hours' in quarter_df.columns else 0
            avg_rn_hours = quarter_df['Total_RN_Hours'].mean() if 'Total_RN_Hours' in quarter_df.columns else 0
            avg_nurse_staff_hours = quarter_df['Nurse_Staff_Hours_Excl_Admin'].mean() if 'Nurse_Staff_Hours_Excl_Admin' in quarter_df.columns else 0
            avg_rn_direct_hours = quarter_df['Hrs_RN'].mean() if 'Hrs_RN' in quarter_df.columns else 0
            avg_rn_admin_hours = quarter_df['Hrs_RNadmin'].mean() if 'Hrs_RNadmin' in quarter_df.columns else 0
            avg_rn_don_hours = quarter_df['Hrs_RNDON'].mean() if 'Hrs_RNDON' in quarter_df.columns else 0

            avg_total_lpn_hours = quarter_df['Total_LPN_Hours'].mean() if 'Total_LPN_Hours' in quarter_df.columns else 0
            avg_lpn_hours = quarter_df['Hrs_LPN'].mean() if 'Hrs_LPN' in quarter_df.columns else 0
            avg_lpn_admin_hours = quarter_df['Hrs_LPNadmin'].mean() if 'Hrs_LPNadmin' in quarter_df.columns else 0
            avg_total_nurse_aide_hours = (
                quarter_df['Total_Nurse_Aide_Hours'].mean()
                if 'Total_Nurse_Aide_Hours' in quarter_df.columns
                else 0
            )
            avg_cna_hours = quarter_df['Hrs_CNA'].mean() if 'Hrs_CNA' in quarter_df.columns else 0
            avg_na_train_hours = quarter_df['Hrs_NAtrn'].mean() if 'Hrs_NAtrn' in quarter_df.columns else 0
            avg_med_aide_hours = quarter_df['Hrs_MedAide'].mean() if 'Hrs_MedAide' in quarter_df.columns else 0
            if not avg_total_nurse_aide_hours:
                avg_total_nurse_aide_hours = avg_cna_hours + avg_na_train_hours + avg_med_aide_hours

            contract_hours_sum = 0.0
            if all(c in quarter_df.columns for c in ['Hrs_RN_ctr', 'Hrs_LPN_ctr', 'Hrs_CNA_ctr']):
                contract_hours_sum = float(
                    pd.to_numeric(quarter_df['Hrs_RN_ctr'], errors='coerce').fillna(0).sum()
                    + pd.to_numeric(quarter_df['Hrs_LPN_ctr'], errors='coerce').fillna(0).sum()
                    + pd.to_numeric(quarter_df['Hrs_CNA_ctr'], errors='coerce').fillna(0).sum()
                )
            direct_hours_sum = float(rn_hours + lpn_direct_hours + cna_hours)
            total_nurse_hours_sum = float(total_staff_hours)
            contract_pct_direct = (
                round_financial(contract_hours_sum / direct_hours_sum * 100.0, 1)
                if direct_hours_sum > 0
                else None
            )
            contract_pct_total = (
                round_financial(contract_hours_sum / total_nurse_hours_sum * 100.0, 1)
                if total_nurse_hours_sum > 0
                else None
            )

            quarterly_data[quarter] = {
                'census': round_financial(avg_census, 2),
                'total_hprd': round_financial(total_hprd, 2),
                'total_hours': round_financial(avg_staff_hours, 2),
                'direct_hprd': round_financial(nurse_staff_hprd, 2),
                'direct_hours': round_financial(avg_nurse_staff_hours, 2),
                'total_rn_hprd': round_financial(total_rn_hprd, 2),
                'total_rn_hours': round_financial(avg_rn_hours, 2),
                'rn_hprd': round_financial(rn_hprd, 2),
                'rn_hours': round_financial(avg_rn_direct_hours, 2),
                'rn_admin_hprd': round_financial(rn_admin_hprd, 2),
                'rn_admin_hours': round_financial(avg_rn_admin_hours, 2),
                'rn_don_hprd': round_financial(rn_don_hprd, 2),
                'rn_don_hours': round_financial(avg_rn_don_hours, 2),
                'total_lpn_hprd': round_financial(total_lpn_hprd, 2),
                'total_lpn_hours': round_financial(avg_total_lpn_hours, 2),
                'lpn_hprd': round_financial(lpn_hprd, 2),
                'lpn_hours': round_financial(avg_lpn_hours, 2),
                'lpn_admin_hprd': round_financial(lpn_admin_hprd, 2),
                'lpn_admin_hours': round_financial(avg_lpn_admin_hours, 2),
                'total_nurse_aide_hprd': round_financial(total_nurse_aide_hprd, 2),
                'total_nurse_aide_hours': round_financial(avg_total_nurse_aide_hours, 2),
                'cna_hprd': round_financial(cna_hprd, 2),
                'cna_hours': round_financial(avg_cna_hours, 2),
                'na_train_hprd': round_financial(na_train_hprd, 2),
                'na_train_hours': round_financial(avg_na_train_hours, 2),
                'med_aide_hprd': round_financial(med_aide_hprd, 2),
                'med_aide_hours': round_financial(avg_med_aide_hours, 2),
                'contract_pct': contract_pct_direct,
                'contract_pct_direct': contract_pct_direct,
                'contract_pct_total': contract_pct_total,
                'pbj_has_data': True,
                'pbj_day_count': int(len(quarter_df)),
                'provider_footnote': provider_footnote,
            }

        return jsonify({'quarterly_data': quarterly_data})

    except Exception as e:
        return jsonify({'error': str(e)})


def _compliance_day_failure_reason(
    *,
    met: bool,
    hprd_type: str,
    metric_unit: str,
    value_for_message: float,
    threshold: float,
    threshold_source: str,
    range_choice: str,
    state_value_type: Optional[str],
) -> Optional[str]:
    """Short, user-facing explanation when a day does not meet the criterion (None if met)."""
    if met:
        return None
    thr = float(threshold)
    val = float(value_for_message)
    if metric_unit == "hours_per_day":
        label = "Total RN hours" if hprd_type == "rn_total_8h" else "Direct RN hours"
        return f"{label} {val:.2f} < {thr:.1f} h/day"
    labels = {
        "total": "Total HPRD",
        "direct_care": "Direct HPRD",
        "cna": "CNA HPRD",
        "nurse_aide": "Nurse aide HPRD",
    }
    metric = labels.get(hprd_type, "HPRD")
    if threshold_source == "custom":
        return f"{metric} {val:.2f} < {thr:.2f} (custom min)"
    st = (state_value_type or "").strip().lower()
    rc = (range_choice or "min").strip().lower()
    if st == "range":
        band = "upper bound" if rc == "max" else "lower bound"
        return f"{metric} {val:.2f} < ~{thr:.2f} (state range, {band})"
    return f"{metric} {val:.2f} < ~{thr:.2f} (state min)"


def _parse_compliance_query_date(raw: Optional[str], field: str) -> Optional[pd.Timestamp]:
    """Parse strict YYYY-MM-DD for compliance filters; reject garbage that breaks pandas (e.g. year 0202)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        raise ValueError(f"Invalid {field}: use YYYY-MM-DD.")
    y, mo, d = int(s[0:4]), int(s[5:7]), int(s[8:10])
    if y < 1990 or y > 2040:
        raise ValueError(f"Invalid {field}: year must be between 1990 and 2040.")
    if mo < 1 or mo > 12 or d < 1 or d > 31:
        raise ValueError(f"Invalid {field}: not a valid calendar date.")
    try:
        return cast(pd.Timestamp, pd.Timestamp(year=y, month=mo, day=d))
    except Exception as exc:
        raise ValueError(f"Invalid {field}: not a valid calendar date.") from exc


@app.route('/api/state-standard-compliance')
def get_state_standard_compliance():
    """Get state standard compliance data for the facility"""
    try:
        global global_df, macpac_standards_df
        
        if global_df is None or len(global_df) == 0:
            return jsonify({'error': 'No data loaded'})

        # Get facility state
        facility_state = global_df['STATE'].iloc[0] if 'STATE' in global_df.columns else None
        if not facility_state:
            return jsonify({'error': 'Facility state not found'})
        
        # State abbreviation to full name mapping
        state_abbrev_to_name = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
            'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'DC': 'District of Columbia',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois',
            'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana',
            'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
            'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
            'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
            'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia',
            'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
        }
        
        # Convert state abbreviation to full name if needed
        state_name = facility_state
        if facility_state.upper() in state_abbrev_to_name:
            state_name = state_abbrev_to_name[facility_state.upper()]

        # Get parameters (needed before optional RN-hours branch)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        hprd_type = request.args.get('hprd_type', 'total')
        range_choice = request.args.get('range_choice', 'min')
        threshold_override = request.args.get('threshold_override')

        try:
            start_dt = _parse_compliance_query_date(start_date, "start_date")
            end_dt = _parse_compliance_query_date(end_date, "end_date")
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400
        if start_dt is not None and end_dt is not None and start_dt > end_dt:
            return jsonify({"error": "start_date must be on or before end_date."}), 400

        # Filter data by date range (same as attorney report)
        filtered_df = global_df.copy()
        if start_dt is not None:
            filtered_df = filtered_df[filtered_df["WorkDate"] >= start_dt]
        if end_dt is not None:
            end_upper = end_dt + pd.Timedelta(days=1)
            filtered_df = filtered_df[filtered_df["WorkDate"] < end_upper]

        filtered_df = filtered_df[filtered_df['MDScensus'] > 0].copy()

        # Fixed 8 h/day RN rules (total vs direct care RN), no MACPAC threshold
        if hprd_type in ('rn_total_8h', 'rn_direct_8h'):
            hours_col = 'Total_RN_Hours' if hprd_type == 'rn_total_8h' else 'Hrs_RN'
            if hours_col not in filtered_df.columns:
                return jsonify({'error': f'{hours_col} not found in facility data'}), 400
            threshold = 8.0
            threshold_source = 'rn_8h_hours'
            filtered_df['Met_Standard'] = filtered_df[hours_col].fillna(0) >= threshold
            filtered_df['Standard_Threshold'] = threshold
            total_days = len(filtered_df)
            days_met = int(filtered_df['Met_Standard'].sum())
            days_not_met = total_days - days_met
            pct_met = (days_met / total_days * 100) if total_days > 0 else 0
            daily_data = []
            for _, row in filtered_df.iterrows():
                work_date = pd.to_datetime(row['WorkDate'])
                met = bool(row['Met_Standard'])
                hrs_disp = round_financial(row[hours_col], 2) if pd.notna(row[hours_col]) else 0.0
                daily_data.append({
                    'date': row['WorkDate'].strftime('%Y-%m-%d'),
                    'hprd': hrs_disp,
                    'threshold': round_financial(threshold, 2),
                    'met_standard': met,
                    'failure_reason': _compliance_day_failure_reason(
                        met=met,
                        hprd_type=hprd_type,
                        metric_unit='hours_per_day',
                        value_for_message=float(hrs_disp),
                        threshold=float(threshold),
                        threshold_source=threshold_source,
                        range_choice='min',
                        state_value_type=None,
                    ),
                    'census': int(row['MDScensus']) if pd.notna(row['MDScensus']) else 0,
                    'day_of_week': work_date.strftime('%A'),
                    'day_of_week_num': work_date.dayofweek,
                    'month': work_date.strftime('%B'),
                    'year': int(work_date.year),
                    'quarter': f"Q{work_date.quarter} {work_date.year}"
                })
            not_met_df = filtered_df[~filtered_df['Met_Standard']].copy()
            day_of_week_counts = {}
            if len(not_met_df) > 0:
                not_met_df['DayOfWeek'] = pd.to_datetime(not_met_df['WorkDate']).dt.day_name()
                day_counts = not_met_df['DayOfWeek'].value_counts().to_dict()
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_of_week_counts = {day: day_counts.get(day, 0) for day in day_order}
            quarter_counts = {}
            if len(not_met_df) > 0:
                not_met_df['Quarter'] = pd.to_datetime(not_met_df['WorkDate']).dt.to_period('Q')
                not_met_df['Quarter'] = not_met_df['Quarter'].apply(lambda x: f"Q{x.quarter} {x.year}")
                quarter_counts = not_met_df['Quarter'].value_counts().to_dict()
            month_counts = {}
            if len(not_met_df) > 0:
                not_met_df['Month'] = pd.to_datetime(not_met_df['WorkDate']).dt.strftime('%B %Y')
                month_counts = not_met_df['Month'].value_counts().to_dict()
            most_common_day = None
            if day_of_week_counts and max(day_of_week_counts.values()) > 0:
                most_common_day = max(day_of_week_counts.items(), key=lambda x: x[1])
            most_common_quarter = None
            if quarter_counts and max(quarter_counts.values()) > 0:
                most_common_quarter = max(quarter_counts.items(), key=lambda x: x[1])
            rn_label = (
                'Total RN (incl. admin/DON) ≥ 8.0 hours per calendar day'
                if hprd_type == 'rn_total_8h'
                else 'Direct RN (excl. admin/DON) ≥ 8.0 hours per calendar day'
            )
            return jsonify({
                'state': state_name,
                'state_abbrev': facility_state if facility_state.upper() in state_abbrev_to_name else None,
                'standard': {
                    'display_text': rn_label,
                    'min_staffing': float(threshold),
                    'max_staffing': None,
                    'value_type': 'minimum_daily_hours',
                    'is_federal_minimum': False
                },
                'hprd_type': hprd_type,
                'range_choice': None,
                'threshold_source': threshold_source,
                'threshold_used': round_financial(threshold, 2),
                'metric_unit': 'hours_per_day',
                'summary': {
                    'total_days': total_days,
                    'days_met': days_met,
                    'days_not_met': days_not_met,
                    'pct_met': round_financial(pct_met, 1),
                    'pct_not_met': round_financial(100 - pct_met, 1)
                },
                'exclusion_note': 'Days with zero census are excluded from total days and from met/not met counts (rare).',
                'secondary_metrics': {
                    'day_of_week_breakdown': day_of_week_counts,
                    'quarter_breakdown': quarter_counts,
                    'month_breakdown': month_counts,
                    'most_common_day': most_common_day[0] if most_common_day else None,
                    'most_common_day_count': int(most_common_day[1]) if most_common_day else 0,
                    'most_common_quarter': most_common_quarter[0] if most_common_quarter else None,
                    'most_common_quarter_count': int(most_common_quarter[1]) if most_common_quarter else 0
                },
                'daily_data': daily_data
            })

        if macpac_standards_df is None or len(macpac_standards_df) == 0:
            return jsonify({'error': 'MACPAC standards not loaded'})
        
        # Get state standard (try both abbreviation and full name)
        state_standard = macpac_standards_df[macpac_standards_df['State'] == state_name]
        if len(state_standard) == 0:
            # Try case-insensitive match
            state_standard = macpac_standards_df[macpac_standards_df['State'].str.upper() == state_name.upper()]
        
        if len(state_standard) == 0:
            return jsonify({'error': f'State standard not found for {facility_state} (tried: {state_name})'})
        
        state_standard = state_standard.iloc[0]
        
        # filtered_df already date-filtered and census > 0 (same as RN branch preamble)
        
        # Compute unrounded HPRD for compliance comparison (match attorney report logic exactly)
        # Total = all staff; Direct = excluding admin/DON
        filtered_df['_Total_Nurse_Hours'] = (
            filtered_df['Hrs_RNDON'].fillna(0) + filtered_df['Hrs_RNadmin'].fillna(0) + filtered_df['Hrs_RN'].fillna(0) +
            filtered_df['Hrs_LPNadmin'].fillna(0) + filtered_df['Hrs_LPN'].fillna(0) + filtered_df['Hrs_CNA'].fillna(0) +
            filtered_df['Hrs_NAtrn'].fillna(0) + filtered_df['Hrs_MedAide'].fillna(0)
        )
        filtered_df['_Total_HPRD_raw'] = filtered_df['_Total_Nurse_Hours'] / filtered_df['MDScensus']
        filtered_df['_Direct_Care_Hours'] = (
            filtered_df['Hrs_RN'].fillna(0) + filtered_df['Hrs_LPN'].fillna(0) + filtered_df['Hrs_CNA'].fillna(0) +
            filtered_df['Hrs_NAtrn'].fillna(0) + filtered_df['Hrs_MedAide'].fillna(0)
        )
        filtered_df['_Direct_Care_HPRD_raw'] = filtered_df['_Direct_Care_Hours'] / filtered_df['MDScensus']

        filtered_df['_CNA_Hours'] = filtered_df['Hrs_CNA'].fillna(0)
        filtered_df['_CNA_HPRD_raw'] = filtered_df['_CNA_Hours'] / filtered_df['MDScensus']

        filtered_df['_Nurse_Aide_Hours'] = (
            filtered_df['Hrs_CNA'].fillna(0) +
            filtered_df['Hrs_NAtrn'].fillna(0) +
            filtered_df['Hrs_MedAide'].fillna(0)
        )
        filtered_df['_Nurse_Aide_HPRD_raw'] = filtered_df['_Nurse_Aide_Hours'] / filtered_df['MDScensus']
        
        # Which HPRD to use for compliance (same definitions as report)
        if hprd_type == 'direct_care':
            hprd_col_raw = '_Direct_Care_HPRD_raw'
            hprd_col_display = 'Direct_Care_HPRD'  # for rounded display
        elif hprd_type == 'cna':
            hprd_col_raw = '_CNA_HPRD_raw'
            hprd_col_display = 'CNA_HPRD'  # for rounded display
        elif hprd_type == 'nurse_aide':
            hprd_col_raw = '_Nurse_Aide_HPRD_raw'
            hprd_col_display = 'Total_Nurse_Aide_HPRD'  # for rounded display
        else:
            hprd_col_raw = '_Total_HPRD_raw'
            hprd_col_display = 'Total_Nurse_HPRD'
        
        if hprd_col_display not in filtered_df.columns:
            filtered_df[hprd_col_display] = filtered_df[hprd_col_raw]  # fallback
        
        # Get standard threshold
        threshold_source = 'state_default'
        threshold = None

        # CNA thresholds are not represented in the MACPAC summary file today; for now we
        # hard-code California's CNA minimum from the provided CA law excerpt.
        if hprd_type == 'cna' or hprd_type == 'nurse_aide':
            if facility_state and str(facility_state).upper() == 'CA':
                threshold = 2.40
        else:
            # CA law excerpt uses 3.50 direct care service hours per patient day.
            if hprd_type == 'direct_care' and facility_state and str(facility_state).upper() == 'CA':
                threshold = 3.50
            else:
                if state_standard['Value_Type'] == 'range':
                    if range_choice == 'max':
                        threshold = state_standard['Max_Staffing']
                    else:
                        threshold = state_standard['Min_Staffing']
                else:
                    threshold = state_standard['Min_Staffing']

        # Optional manual override from the dashboard controls.
        if threshold_override is not None and str(threshold_override).strip() != '':
            try:
                parsed_threshold = float(threshold_override)
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid threshold_override. Must be a number.'}), 400

            if parsed_threshold <= 0 or parsed_threshold > 20:
                return jsonify({'error': 'threshold_override must be greater than 0 and less than or equal to 20.'}), 400

            threshold = parsed_threshold
            threshold_source = 'custom'

        if threshold is None:
            return jsonify({
                'error': 'CNA minimum threshold not available for this state. Choose a custom CNA threshold.',
            }), 400

        # Georgia baseline override for state-default thresholding.
        if threshold_source == 'state_default' and facility_state and str(facility_state).upper() == 'GA' and hprd_type in ('total', 'direct_care'):
            threshold = 2.00

        # NJ direct-care minimum is slightly lower than the MACPAC summary value.
        # Override only for Direct Care compliance analysis.
        if threshold_source == 'state_default' and facility_state and str(facility_state).upper() == 'NJ' and hprd_type == 'direct_care':
            threshold = 2.50
        
        # Skip if federal minimum
        if state_standard.get('Is_Federal_Minimum', False) and hprd_type in ('total', 'direct_care'):
            return jsonify({
                'error': 'State uses federal minimum (0.30 HPRD) - compliance tracking not applicable',
                'is_federal_minimum': True
            })
        
        # Check compliance using unrounded HPRD (match attorney report: strict < threshold = below)
        filtered_df['Met_Standard'] = filtered_df[hprd_col_raw] >= threshold
        filtered_df['Standard_Threshold'] = threshold
        
        # Calculate summary
        total_days = len(filtered_df)
        days_met = filtered_df['Met_Standard'].sum()
        days_not_met = total_days - days_met
        pct_met = (days_met / total_days * 100) if total_days > 0 else 0
        
        # Prepare daily data (display rounded HPRD)
        daily_data = []
        state_vt = str(state_standard['Value_Type'])
        for _, row in filtered_df.iterrows():
            work_date = pd.to_datetime(row['WorkDate'])
            hprd_display = round_financial(row[hprd_col_display], 2) if hprd_col_display in row.index else round_financial(row[hprd_col_raw], 2)
            met = bool(row['Met_Standard'])
            daily_data.append({
                'date': row['WorkDate'].strftime('%Y-%m-%d'),
                'hprd': hprd_display,
                'threshold': round_financial(threshold, 2),
                'met_standard': met,
                'failure_reason': _compliance_day_failure_reason(
                    met=met,
                    hprd_type=hprd_type,
                    metric_unit='hprd',
                    value_for_message=float(hprd_display),
                    threshold=float(threshold),
                    threshold_source=str(threshold_source),
                    range_choice=str(range_choice),
                    state_value_type=state_vt,
                ),
                'census': int(row['MDScensus']) if pd.notna(row['MDScensus']) else 0,
                'day_of_week': work_date.strftime('%A'),  # Monday, Tuesday, etc.
                'day_of_week_num': work_date.dayofweek,  # 0=Monday, 6=Sunday
                'month': work_date.strftime('%B'),  # January, February, etc.
                'year': int(work_date.year),
                'quarter': f"Q{work_date.quarter} {work_date.year}"  # Q4 2022, etc.
            })
        
        # Calculate secondary metrics
        not_met_df = filtered_df[~filtered_df['Met_Standard']].copy()
        
        # Day of week analysis
        day_of_week_counts = {}
        if len(not_met_df) > 0:
            not_met_df['DayOfWeek'] = pd.to_datetime(not_met_df['WorkDate']).dt.day_name()
            day_counts = not_met_df['DayOfWeek'].value_counts().to_dict()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_of_week_counts = {day: day_counts.get(day, 0) for day in day_order}
        
        # Time period analysis (by quarter)
        quarter_counts = {}
        if len(not_met_df) > 0:
            # Create quarter in format "Q4 2022" instead of "2022Q4"
            not_met_df['Quarter'] = pd.to_datetime(not_met_df['WorkDate']).dt.to_period('Q')
            not_met_df['Quarter'] = not_met_df['Quarter'].apply(lambda x: f"Q{x.quarter} {x.year}")
            quarter_counts = not_met_df['Quarter'].value_counts().to_dict()
        
        # Month analysis
        month_counts = {}
        if len(not_met_df) > 0:
            not_met_df['Month'] = pd.to_datetime(not_met_df['WorkDate']).dt.strftime('%B %Y')
            month_counts = not_met_df['Month'].value_counts().to_dict()
        
        # Most common day of week for non-compliance
        most_common_day = None
        if day_of_week_counts and max(day_of_week_counts.values()) > 0:
            most_common_day = max(day_of_week_counts.items(), key=lambda x: x[1])
        
        # Most common quarter for non-compliance
        most_common_quarter = None
        if quarter_counts and max(quarter_counts.values()) > 0:
            most_common_quarter = max(quarter_counts.items(), key=lambda x: x[1])
        
        return jsonify({
            'state': state_name,
            'state_abbrev': facility_state if facility_state.upper() in state_abbrev_to_name else None,
            'standard': {
                # Compliance thresholds are derived from a MACPAC summary + local overrides.
                # Treat them as estimates and label accordingly in the UI.
                'display_text': f"Estimated state threshold: {round_financial(threshold, 2)} HPRD",
                'min_staffing': float(state_standard['Min_Staffing']),
                'max_staffing': float(state_standard['Max_Staffing']) if state_standard['Value_Type'] == 'range' else None,
                'value_type': state_standard['Value_Type'],
                'is_federal_minimum': bool(state_standard.get('Is_Federal_Minimum', False))
            },
            'hprd_type': hprd_type,
            'range_choice': range_choice if state_standard['Value_Type'] == 'range' else None,
            'threshold_source': threshold_source,
            'threshold_used': round_financial(threshold, 2),
            'metric_unit': 'hprd',
            'summary': {
                'total_days': total_days,
                'days_met': int(days_met),
                'days_not_met': int(days_not_met),
                'pct_met': round_financial(pct_met, 1),
                'pct_not_met': round_financial(100 - pct_met, 1)
            },
            'exclusion_note': 'Days with zero census are excluded from total days and from met/not met counts (rare).',
            'secondary_metrics': {
                'day_of_week_breakdown': day_of_week_counts,
                'quarter_breakdown': quarter_counts,
                'month_breakdown': month_counts,
                'most_common_day': most_common_day[0] if most_common_day else None,
                'most_common_day_count': int(most_common_day[1]) if most_common_day else 0,
                'most_common_quarter': most_common_quarter[0] if most_common_quarter else None,
                'most_common_quarter_count': int(most_common_quarter[1]) if most_common_quarter else 0
            },
            'daily_data': daily_data
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route("/api/geo-distribution")
def api_geo_distribution():
    """PBJ-derived facility-quarter distribution for a metric within a geography."""
    try:
        provnum = request.args.get("provnum") or request.args.get("ccn") or ""
        if not provnum and global_df is not None and len(global_df) and "PROVNUM" in global_df.columns:
            provnum = str(global_df["PROVNUM"].iloc[0])
        quarter = request.args.get("quarter") or request.args.get("cy_qtr") or ""
        metric = request.args.get("metric") or "total_nurse_hprd"
        geography_type = request.args.get("geography_type") or "state"
        geography_value = request.args.get("geography_value") or ""
        facility_name = request.args.get("facility_name") or ""
        threshold_override = request.args.get("threshold_override")
        thr_ov: Optional[float] = None
        if threshold_override is not None and str(threshold_override).strip() != "":
            try:
                thr_ov = float(threshold_override)
            except (TypeError, ValueError):
                return jsonify({"error": "Invalid threshold_override."}), 400
        peer_sort = request.args.get("peer_sort") or ""
        payload = build_geo_distribution_payload(
            provnum=str(provnum),
            quarter=str(quarter),
            metric=str(metric),
            geography_type=str(geography_type),
            geography_value=str(geography_value),
            facility_name=str(facility_name),
            threshold_override=thr_ov,
            peer_sort=str(peer_sort),
            macpac_getter=facility_report_lib.get_macpac_state_standards,
        )
        if payload.get("error"):
            return jsonify(payload), 400
        return jsonify(payload)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/geo-distribution/context")
def api_geo_distribution_context():
    """Geography facility counts and default comparison scope for a quarter."""
    try:
        provnum = request.args.get("provnum") or request.args.get("ccn") or ""
        if not provnum and global_df is not None and len(global_df) and "PROVNUM" in global_df.columns:
            provnum = str(global_df["PROVNUM"].iloc[0])
        quarter = request.args.get("quarter") or request.args.get("cy_qtr") or ""
        payload = build_geo_distribution_context(str(provnum), str(quarter))
        if payload.get("error"):
            return jsonify(payload), 400
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def _scoped_provider_info_df_for_facility() -> Optional[pd.DataFrame]:
    """Provider Information rows for the loaded facility CCN (never cross-facility)."""
    prov_df_cm: Optional[pd.DataFrame] = provider_info_df
    if provider_info_df is None or len(provider_info_df) == 0:
        return prov_df_cm
    if "ccn" not in provider_info_df.columns:
        return prov_df_cm
    ccn_fac: Optional[str] = None
    if global_df is not None and len(global_df) > 0:
        for col in ("PROVNUM", "provnum", "CCN", "ccn"):
            if col not in global_df.columns:
                continue
            raw_ccn = str(global_df[col].iloc[0]).strip()
            if raw_ccn and raw_ccn.lower() not in ("nan", "none", "unknown"):
                ccn_fac = raw_ccn.zfill(6)
                break
    if not ccn_fac:
        return prov_df_cm
    _mask = provider_info_df["ccn"].astype(str).str.strip().str.zfill(6) == ccn_fac
    _sub = provider_info_df.loc[_mask]
    if len(_sub) > 0:
        return cast(pd.DataFrame, _sub)
    return prov_df_cm


def _quarter_provider_footnote_text(rows: pd.DataFrame) -> Optional[str]:
    """Return concise provider footnote text for the matched quarter when present."""
    if rows is None or len(rows) == 0:
        return None
    cols = [c for c in rows.columns if isinstance(c, str)]
    ordered_cols: list[str] = []
    if "footnotes" in cols:
        ordered_cols.append("footnotes")
    ordered_cols.extend(
        [c for c in cols if c.endswith("_footnote") and c not in ordered_cols]
    )
    if not ordered_cols:
        return None
    seen: set[str] = set()
    notes: list[str] = []
    for col in ordered_cols:
        series = rows[col].dropna().astype(str)
        for raw in series.tolist():
            txt = str(raw).strip()
            if not txt:
                continue
            low = txt.lower()
            if low in {"nan", "none", "null", "n/a", "not available"}:
                continue
            if txt in seen:
                continue
            seen.add(txt)
            notes.append(txt)
        if notes:
            break
    if not notes:
        return None
    return " | ".join(notes[:2])


def _provider_rows_for_canonical_quarter(
    prov_df: Optional[pd.DataFrame], quarter: str
) -> pd.DataFrame:
    if prov_df is None or len(prov_df) == 0 or "quarter" not in prov_df.columns:
        return pd.DataFrame()
    prov_canon = prov_df["quarter"].map(_provider_quarter_to_canonical)
    return cast(pd.DataFrame, prov_df.loc[prov_canon == quarter])


def _all_canonical_quarters_pbj_and_provider() -> set[str]:
    """Union of PBJ CY_Qtr labels and Provider Information quarter labels."""
    all_quarters: set[str] = set()
    if global_df is not None and len(global_df) > 0 and "CY_Qtr" in global_df.columns:
        for q in global_df["CY_Qtr"].dropna().unique():
            cq = _provider_quarter_to_canonical(q)
            if cq:
                all_quarters.add(cq)
    prov_df_cm = _scoped_provider_info_df_for_facility()
    if prov_df_cm is not None and len(prov_df_cm) > 0 and "quarter" in prov_df_cm.columns:
        for q in prov_df_cm[prov_df_cm["quarter"].notna()]["quarter"].unique():
            cq = _provider_quarter_to_canonical(q)
            if cq:
                all_quarters.add(cq)
    return all_quarters


@app.route('/api/case-mix-data')
def get_case_mix_data():
    """Get case-mix acuity data by quarter from both Provider Info and PBJ calculations"""
    try:
        case_mix_data: Dict[str, Dict[str, Any]] = {}
        _ensure_pbj_dashboard_derived_columns_inplace(global_df)

        prov_df_cm = _scoped_provider_info_df_for_facility()

        # Single canonical key CYyyyyQn everywhere (matches PBJ CY_Qtr and _provider_quarter_to_canonical).
        all_quarters = _all_canonical_quarters_pbj_and_provider()
        if len(all_quarters) == 0:
            return jsonify({"error": "No data available", "case_mix_data": {}})

        sorted_quarters = sorted(all_quarters, key=_cy_quarter_sort_key_from_canonical)

        pbj_canon_series = None
        pbj_gb = None
        if global_df is not None and len(global_df) > 0 and "CY_Qtr" in global_df.columns:
            pbj_canon_series = global_df["CY_Qtr"].map(_provider_quarter_to_canonical)
            pbj_gb = global_df.groupby(pbj_canon_series, sort=False)

        prov_canon_series = None
        if prov_df_cm is not None and len(prov_df_cm) > 0 and "quarter" in prov_df_cm.columns:
            prov_canon_series = prov_df_cm["quarter"].map(_provider_quarter_to_canonical)

        for quarter in sorted_quarters:
            quarter_info: Dict[str, Any] = {"quarter": quarter}
            pbj_for_q = pd.DataFrame()
            if pbj_gb is not None and quarter in pbj_gb.groups:
                pbj_for_q = pbj_gb.get_group(quarter)
            quarter_info["pbj_day_count"] = int(len(pbj_for_q))
            quarter_info["pbj_has_data"] = bool(len(pbj_for_q) > 0)
            quarter_info["provider_footnote"] = None

            # === PROVIDER INFO DATA ===
            # IMPORTANT: Only use data from exact quarter matches or exact date range matches.
            # NEVER use fallback data from other quarters - if no match exists, leave fields as None.
            if prov_df_cm is not None and len(prov_df_cm) > 0:
                # Check if quarter column exists
                if prov_canon_series is not None:
                    prov_quarter_data = prov_df_cm.loc[prov_canon_series == quarter]

                    if len(prov_quarter_data) == 0:
                        # If no exact match, try matching null quarters by date (only within exact date range)
                        # NEVER use fallback data from other quarters - if no match, leave empty
                        if len(pbj_for_q) > 0:
                            # Get date range for this quarter from PBJ data
                            quarter_dates = pbj_for_q["WorkDate"]
                            if len(quarter_dates) > 0:
                                min_date = quarter_dates.min()
                                max_date = quarter_dates.max()
                                # Match provider info rows with null quarters that fall within this quarter's date range
                                null_quarter_rows = prov_df_cm[
                                    (prov_df_cm['quarter'].isna()) &
                                    (prov_df_cm['processing_date'] >= min_date) & 
                                    (prov_df_cm['processing_date'] <= max_date)
                                ]
                                if len(null_quarter_rows) > 0:
                                    prov_quarter_data = null_quarter_rows.sort_values(
                                        "processing_date", ascending=False
                                    )
                                # NO FALLBACK - if no exact match or date match, leave empty (prov_quarter_data stays empty)
                else:
                    # If no quarter column, try to match by exact date range only
                    # NEVER use all data - only match by exact date range
                    prov_quarter_data = pd.DataFrame()  # Start empty
                    # Try to match by date range if possible
                    if 'processing_date' in prov_df_cm.columns and global_df is not None:
                        # Get date range for this quarter from PBJ data
                        quarter_dates = pbj_for_q["WorkDate"] if len(pbj_for_q) > 0 else pd.Series(dtype="datetime64[ns]")
                        if len(quarter_dates) > 0:
                            min_date = quarter_dates.min()
                            max_date = quarter_dates.max()
                            # Only use data within exact date range - no fallback
                            prov_quarter_data = prov_df_cm[
                                (prov_df_cm['processing_date'] >= min_date) & 
                                (prov_df_cm['processing_date'] <= max_date)
                            ]
                            # NO FALLBACK - if no exact date match, leave empty (prov_quarter_data stays empty)
                
                if len(prov_quarter_data) > 0:
                    quarter_info["provider_footnote"] = _quarter_provider_footnote_text(
                        cast(pd.DataFrame, prov_quarter_data)
                    )
                    prov_data = coalesce_provider_quarter_snapshots(
                        cast(pd.DataFrame, prov_quarter_data)
                    )
                    provider_snapshot_for_quarter = prov_data
                    quarter_info["cmi_from_fallback_row"] = False
                    cmi_raw, cmi_src = extract_nursing_cmi_from_provider_series(prov_data)
                    # Nursing CMI (``nursing_case_mix_index``) can live on an older snapshot while the
                    # chronologically last row used by coalesce omits it; prefer rows with positive CMI.
                    if cmi_raw is None and len(prov_quarter_data) > 0:
                        narrow_df = narrow_provider_quarter_rows_for_case_mix(
                            cast(pd.DataFrame, prov_quarter_data)
                        )
                        fallback_snapshot: Optional[pd.Series] = None
                        if len(narrow_df) == 1:
                            fallback_snapshot = narrow_df.iloc[0]
                        elif len(narrow_df) > 1:
                            fallback_snapshot = coalesce_provider_quarter_snapshots(cast(pd.DataFrame, narrow_df))
                        if fallback_snapshot is not None:
                            cmi_fb, cmi_fb_src = extract_nursing_cmi_from_provider_series(fallback_snapshot)
                            if cmi_fb is not None:
                                cmi_raw, cmi_src = cmi_fb, cmi_fb_src
                                provider_snapshot_for_quarter = fallback_snapshot
                                quarter_info["cmi_from_fallback_row"] = True

                    # Keep a single provider snapshot per quarter so CMI and case-mix HPRD
                    # come from the same row whenever fallback is used.
                    quarter_info['prov_reported_total'] = _numeric_cell_to_optional_float(
                        provider_snapshot_for_quarter.get('reported_total_nurse_hrs_per_resident_per_day')
                    )
                    quarter_info['prov_reported_rn'] = _numeric_cell_to_optional_float(
                        provider_snapshot_for_quarter.get('reported_rn_hrs_per_resident_per_day')
                    )
                    quarter_info['prov_reported_lpn'] = _numeric_cell_to_optional_float(
                        provider_snapshot_for_quarter.get('reported_lpn_hrs_per_resident_per_day')
                    )
                    quarter_info['prov_reported_na'] = _numeric_cell_to_optional_float(
                        provider_snapshot_for_quarter.get('reported_na_hrs_per_resident_per_day')
                    )
                    quarter_info['case_mix_total'] = _numeric_cell_to_optional_float(
                        provider_snapshot_for_quarter.get('case_mix_total_nurse_hrs_per_resident_per_day')
                    )
                    quarter_info['case_mix_rn'] = _numeric_cell_to_optional_float(
                        provider_snapshot_for_quarter.get('case_mix_rn_hrs_per_resident_per_day')
                    )
                    quarter_info['case_mix_lpn'] = _numeric_cell_to_optional_float(
                        provider_snapshot_for_quarter.get('case_mix_lpn_hrs_per_resident_per_day')
                    )
                    quarter_info['case_mix_na'] = _numeric_cell_to_optional_float(
                        provider_snapshot_for_quarter.get('case_mix_na_hrs_per_resident_per_day')
                    )
                    if cmi_raw is not None:
                        quarter_info["cmi"] = cmi_raw
                        quarter_info["cmi_raw"] = cmi_raw
                        quarter_info["cmi_source"] = cmi_src
                    else:
                        quarter_info["cmi"] = None
                        quarter_info["cmi_raw"] = None
                        quarter_info["cmi_source"] = None
                else:
                    # No provider info data found for this quarter - explicitly set all fields to None
                    # This ensures we never use fallback data from other quarters
                    quarter_info['prov_reported_total'] = None
                    quarter_info['prov_reported_rn'] = None
                    quarter_info['prov_reported_lpn'] = None
                    quarter_info['prov_reported_na'] = None
                    quarter_info['case_mix_total'] = None
                    quarter_info['case_mix_rn'] = None
                    quarter_info['case_mix_lpn'] = None
                    quarter_info['case_mix_na'] = None
                    quarter_info['cmi'] = None
                    quarter_info['cmi_raw'] = None
                    quarter_info['cmi_source'] = None
                    quarter_info["cmi_from_fallback_row"] = False
            
            # === PBJ DATA (calculated from daily records) ===
            if global_df is not None and len(global_df) > 0:
                pbj_quarter_df = pbj_for_q
                if len(pbj_quarter_df) > 0:
                    total_census = pbj_quarter_df['MDScensus'].sum()
                    
                    # Total Staff (all nursing staff)
                    total_staff_hours = pbj_quarter_df['Total_Staff_Hours'].sum()
                    quarter_info['pbj_reported_total'] = (total_staff_hours / total_census) if total_census > 0 else None
                    
                    # Direct Staff (excludes RN admin, RN DON, LPN admin)
                    direct_staff_hours = pbj_quarter_df['Nurse_Staff_Hours_Excl_Admin'].sum()
                    quarter_info['pbj_reported_direct'] = (direct_staff_hours / total_census) if total_census > 0 else None
                    
                    # Total RN (includes RN + RN admin + RN DON)
                    total_rn_hours = pbj_quarter_df['Total_RN_Hours'].sum()
                    quarter_info['pbj_reported_total_rn'] = (total_rn_hours / total_census) if total_census > 0 else None
                    
                    # Direct RN (excludes RN admin and RN DON)
                    rn_hours = pbj_quarter_df['Hrs_RN'].sum()
                    quarter_info['pbj_reported_direct_rn'] = (rn_hours / total_census) if total_census > 0 else None
                    
                    # Total LPN (includes LPN + LPN admin)
                    total_lpn_hours = pbj_quarter_df['Total_LPN_Hours'].sum()
                    quarter_info['pbj_reported_total_lpn'] = (total_lpn_hours / total_census) if total_census > 0 else None
                    
                    # Direct LPN (excludes LPN admin)
                    lpn_hours = pbj_quarter_df['Hrs_LPN'].sum()
                    quarter_info['pbj_reported_direct_lpn'] = (lpn_hours / total_census) if total_census > 0 else None
                    
                    # Nurse Aide (CNA + Med Aide + NA Trainee)
                    na_hours = pbj_quarter_df['Total_Nurse_Aide_Hours'].sum()
                    quarter_info['pbj_reported_na'] = (na_hours / total_census) if total_census > 0 else None
            
            # === CALCULATE % CASE-MIX ===
            # Use raw (unrounded) direct hours / census for % so display matches full precision (e.g. 89.7% not 89.0%).
            raw_direct_hprd = None
            if len(pbj_for_q) > 0:
                pbj_q = pbj_for_q
                total_census = pbj_q['MDScensus'].sum()
                if total_census > 0:
                    hrs_rn = pbj_q['Hrs_RN'].fillna(0)
                    hrs_lpn = pbj_q['Hrs_LPN'].fillna(0)
                    hrs_cna = pbj_q['Hrs_CNA'].fillna(0)
                    hrs_na = pbj_q['Hrs_NAtrn'].fillna(0) if 'Hrs_NAtrn' in pbj_q.columns else 0
                    hrs_med = pbj_q['Hrs_MedAide'].fillna(0) if 'Hrs_MedAide' in pbj_q.columns else 0
                    raw_direct_hprd = (hrs_rn + hrs_lpn + hrs_cna + hrs_na + hrs_med).sum() / total_census
            # % of CMS case-mix total HPRD (not nursing_case_mix_index — that is separate in the CMI column).
            # Prefer Provider Info reported total; if CMS did not publish it for that row, use PBJ total nurse HPRD
            # so Total % stays aligned with Direct % (same denominator) and the Acuity pod does not show "—" / xx% split.
            cm_den = quarter_info.get("case_mix_total")
            pct_cmi_total_val = None
            if cm_den is not None and not pd.isna(cm_den) and float(cm_den) > 0:
                prov_tot = quarter_info.get("prov_reported_total")
                if prov_tot is not None and not pd.isna(prov_tot):
                    pct_cmi_total_val = (float(prov_tot) / float(cm_den)) * 100
                else:
                    pbj_tot = quarter_info.get("pbj_reported_total")
                    if pbj_tot is not None and not pd.isna(pbj_tot):
                        pct_cmi_total_val = (float(pbj_tot) / float(cm_den)) * 100
            quarter_info["pct_cmi_total"] = pct_cmi_total_val
            
            # Use provider-info case_mix_total as the single denominator for both Total and Direct
            # (do not use combined RN+LPN+NA so hover shows same Case-Mix value from provider info)
            quarter_info['case_mix_direct'] = quarter_info.get('case_mix_total')  # same as total for display/denominator
            
            # Direct row: PBJ direct HPRD ÷ same CMS case_mix_total (legacy JSON key pct_cmi_direct)
            if quarter_info.get('case_mix_total') and quarter_info['case_mix_total'] > 0:
                numer = raw_direct_hprd if raw_direct_hprd is not None else quarter_info.get('pbj_reported_direct')
                if numer is not None:
                    quarter_info['pct_cmi_direct'] = (numer / quarter_info['case_mix_total'] * 100)
                else:
                    quarter_info['pct_cmi_direct'] = None
            else:
                quarter_info['pct_cmi_direct'] = None
            
            # RN total %: Provider Info RN if present, else PBJ total RN (same case-mix RN denominator).
            cm_rn = quarter_info.get("case_mix_rn")
            pct_rn_tot = None
            if cm_rn is not None and not pd.isna(cm_rn) and float(cm_rn) > 0:
                prov_rn = quarter_info.get("prov_reported_rn")
                if prov_rn is not None and not pd.isna(prov_rn):
                    pct_rn_tot = (float(prov_rn) / float(cm_rn)) * 100
                else:
                    pbj_rn_tot = quarter_info.get("pbj_reported_total_rn")
                    if pbj_rn_tot is not None and not pd.isna(pbj_rn_tot):
                        pct_rn_tot = (float(pbj_rn_tot) / float(cm_rn)) * 100
            quarter_info["pct_cmi_total_rn"] = pct_rn_tot
            
            # RN direct
            if quarter_info.get('pbj_reported_direct_rn') and quarter_info.get('case_mix_rn') and quarter_info['case_mix_rn'] > 0:
                quarter_info['pct_cmi_direct_rn'] = (quarter_info['pbj_reported_direct_rn'] / quarter_info['case_mix_rn'] * 100)
            else:
                quarter_info['pct_cmi_direct_rn'] = None
            
            # Total LPN %: Provider Info LPN if present, else PBJ total LPN.
            cm_lpn = quarter_info.get("case_mix_lpn")
            pct_lpn_tot = None
            if cm_lpn is not None and not pd.isna(cm_lpn) and float(cm_lpn) > 0:
                prov_lpn = quarter_info.get("prov_reported_lpn")
                if prov_lpn is not None and not pd.isna(prov_lpn):
                    pct_lpn_tot = (float(prov_lpn) / float(cm_lpn)) * 100
                else:
                    pbj_lpn_tot = quarter_info.get("pbj_reported_total_lpn")
                    if pbj_lpn_tot is not None and not pd.isna(pbj_lpn_tot):
                        pct_lpn_tot = (float(pbj_lpn_tot) / float(cm_lpn)) * 100
            quarter_info["pct_cmi_total_lpn"] = pct_lpn_tot
            
            # Direct LPN CMI
            if quarter_info.get('pbj_reported_direct_lpn') and quarter_info.get('case_mix_lpn') and quarter_info['case_mix_lpn'] > 0:
                quarter_info['pct_cmi_direct_lpn'] = (quarter_info['pbj_reported_direct_lpn'] / quarter_info['case_mix_lpn'] * 100)
            else:
                quarter_info['pct_cmi_direct_lpn'] = None
            
            # Nurse aide %: Provider Info NA if present, else PBJ nurse aide HPRD.
            cm_na = quarter_info.get("case_mix_na")
            pct_na_val = None
            if cm_na is not None and not pd.isna(cm_na) and float(cm_na) > 0:
                prov_na = quarter_info.get("prov_reported_na")
                if prov_na is not None and not pd.isna(prov_na):
                    pct_na_val = (float(prov_na) / float(cm_na)) * 100
                else:
                    pbj_na = quarter_info.get("pbj_reported_na")
                    if pbj_na is not None and not pd.isna(pbj_na):
                        pct_na_val = (float(pbj_na) / float(cm_na)) * 100
            quarter_info["pct_cmi_na"] = pct_na_val
            
            # === CALCULATE HARRINGTON-ADJUSTED HPRD (single source: facility_report_lib) ===
            _cmi_src = quarter_info.get("cmi_raw")
            if _cmi_src is None:
                _cmi_src = quarter_info.get("cmi")
            cmi_h: Optional[float] = None
            if _cmi_src is not None and not pd.isna(_cmi_src):
                try:
                    cmi_h = float(_cmi_src)
                except (TypeError, ValueError):
                    cmi_h = None
            if cmi_h is not None and cmi_h > 0:
                quarter_info['harrington_total'] = calculate_harrington_adjusted_hprd(cmi_h, 'total')
                quarter_info['harrington_rn'] = calculate_harrington_adjusted_hprd(cmi_h, 'rn')
                quarter_info['harrington_cna'] = calculate_harrington_adjusted_hprd(cmi_h, 'cna')
                
                # Calculate Harrington-adjusted percentages
                # Total Harrington (use PBJ direct care)
                _pdir = quarter_info.get('pbj_reported_direct')
                _htot = quarter_info.get('harrington_total')
                if _pdir is not None and _htot is not None and float(_htot) > 0:
                    quarter_info['pct_harrington_total'] = (float(_pdir) / float(_htot)) * 100.0
                else:
                    quarter_info['pct_harrington_total'] = None
                
                # RN Harrington (use PBJ direct RN)
                _prn = quarter_info.get('pbj_reported_direct_rn')
                _hrn = quarter_info.get('harrington_rn')
                if _prn is not None and _hrn is not None and float(_hrn) > 0:
                    quarter_info['pct_harrington_rn'] = (float(_prn) / float(_hrn)) * 100.0
                else:
                    quarter_info['pct_harrington_rn'] = None
                
                # CNA Harrington (use PBJ reported NA)
                _pna = quarter_info.get('pbj_reported_na')
                _hcna = quarter_info.get('harrington_cna')
                if _pna is not None and _hcna is not None and float(_hcna) > 0:
                    quarter_info['pct_harrington_cna'] = (float(_pna) / float(_hcna)) * 100.0
                else:
                    quarter_info['pct_harrington_cna'] = None
            else:
                quarter_info['harrington_total'] = None
                quarter_info['harrington_rn'] = None
                quarter_info['harrington_cna'] = None
                quarter_info['pct_harrington_total'] = None
                quarter_info['pct_harrington_rn'] = None
                quarter_info['pct_harrington_cna'] = None
            
            # Round all numeric values with appropriate precision for display.
            # Use round_financial (half-up) so values like 1.535 display as 1.54, not 1.53
            # (JS .toFixed(2) on 1.535 can show 1.53 due to float representation).
            exclude_keys = {
                'quarter',
                'cmi_source',
                'cmi',
                'cmi_raw',
                'harrington_total',
                'harrington_rn',
                'harrington_cna',
                'pct_harrington_total',
                'pct_harrington_rn',
                'pct_harrington_cna',
            }
            hprd_keys = {
                'prov_reported_total', 'prov_reported_rn', 'prov_reported_lpn', 'prov_reported_na',
                'case_mix_total', 'case_mix_rn', 'case_mix_lpn', 'case_mix_na', 'case_mix_direct',
                'pbj_reported_total', 'pbj_reported_direct', 'pbj_reported_total_rn', 'pbj_reported_direct_rn',
                'pbj_reported_total_lpn', 'pbj_reported_direct_lpn', 'pbj_reported_na',
            }
            for key, value in quarter_info.items():
                if key not in exclude_keys and value is not None:
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and not pd.isna(value):
                        if key.startswith('pct_'):
                            quarter_info[key] = round_financial(value, 1)
                        elif key in hprd_keys:
                            quarter_info[key] = round_financial(value, 2)
                        else:
                            quarter_info[key] = round_financial(value, 3)
            
            # Include all quarters in the dashboard (CMI can be None/blank for some quarters)
            # The filtering for Harrington section is done in the report, not the dashboard
            case_mix_data[quarter] = quarter_info
        
        r = jsonify({'case_mix_data': case_mix_data})
        r.headers['Cache-Control'] = 'no-store'
        return r
        
    except Exception as e:
        return jsonify({'error': str(e), 'case_mix_data': {}})

@app.route('/api/harrington-cmi')
def get_harrington_cmi():
    """Get Harrington Expected HPRD calculations for a given quarter (or all quarters) and CMI."""
    import urllib.parse

    def _resolve_cmi_for_quarter(quarter_key: str, quarter_data: Optional[dict]) -> Optional[float]:
        if not quarter_data:
            return None
        cmi_raw = quarter_data.get('cmi_raw') if quarter_data.get('cmi_raw') is not None else quarter_data.get('cmi')
        cmi_from_payload = _numeric_cell_to_optional_float(cmi_raw)
        if cmi_from_payload is not None and cmi_from_payload > 0:
            return cmi_from_payload
        cmi = None
        if provider_info_df is not None and len(provider_info_df) > 0:
            facility_ccn = None
            if global_df is not None and len(global_df) > 0:
                for col in ['PROVNUM', 'provnum', 'ccn']:
                    if col in global_df.columns:
                        facility_ccn = str(global_df[col].iloc[0]).strip().zfill(6)
                        break
            if facility_ccn and 'ccn' in provider_info_df.columns:
                prov_sub = provider_info_df[provider_info_df['ccn'].astype(str).str.strip().str.zfill(6) == facility_ccn]
                if 'quarter' in prov_sub.columns:
                    target_c = _provider_quarter_to_canonical(quarter_key)
                    match = prov_sub[
                        prov_sub['quarter'].apply(
                            lambda x: _provider_quarter_to_canonical(x) == target_c
                        )
                    ]
                    if len(match) > 0:
                        for c in match.columns:
                            if c and isinstance(c, str) and 'nursing_case_mix_index' in c.lower() and 'ratio' not in c.lower():
                                num_m = pd.to_numeric(match[c], errors="coerce")
                                has_cmi = match[num_m.notna() & (num_m > 0)]
                                if len(has_cmi) > 0:
                                    row = has_cmi.sort_values('processing_date', ascending=False).iloc[0]
                                    try:
                                        cn = _numeric_cell_to_optional_float(row[c])
                                        if cn is not None and cn > 0:
                                            cmi = cn
                                            break
                                    except (ValueError, TypeError):
                                        pass
                        if cmi is None or pd.isna(cmi) or cmi <= 0:
                            row = match.sort_values('processing_date', ascending=False).iloc[0]
                            for c in match.columns:
                                if c and isinstance(c, str) and 'nursing_case_mix_index' in c.lower() and 'ratio' not in c.lower():
                                    try:
                                        v = _numeric_cell_to_optional_float(row.get(c))
                                        if v is not None and v > 0:
                                            cmi = v
                                            break
                                    except (ValueError, TypeError):
                                        pass
        if cmi is None or pd.isna(cmi) or cmi <= 0:
            return None
        out_cmi = _numeric_cell_to_optional_float(cmi)
        return out_cmi if out_cmi is not None and out_cmi > 0 else None

    def _harrington_row_for_quarter(
        quarter_key: str, quarter_data: dict, use_total: bool, *, require_cmi: bool
    ) -> Optional[Dict[str, Any]]:
        """Build Harrington table row; when ``require_cmi`` is False, include quarters without nursing CMI (expected/% N/A)."""
        cmi = _resolve_cmi_for_quarter(quarter_key, quarter_data)
        if use_total:
            reported_total_raw = quarter_data.get('pbj_reported_total')
            reported_rn_raw = quarter_data.get('pbj_reported_total_rn')
            reported_lpn_raw = quarter_data.get('pbj_reported_total_lpn')
        else:
            reported_total_raw = quarter_data.get('pbj_reported_direct')
            reported_rn_raw = quarter_data.get('pbj_reported_direct_rn')
            reported_lpn_raw = quarter_data.get('pbj_reported_direct_lpn')
        reported_na_raw = quarter_data.get('pbj_reported_na')
        pbj_total_staff = quarter_data.get('pbj_reported_total')
        pbj_direct_staff = quarter_data.get('pbj_reported_direct')
        pbj_rn_total_staff = quarter_data.get('pbj_reported_total_rn')
        pbj_rn_direct_staff = quarter_data.get('pbj_reported_direct_rn')
        reported_total = round_financial(reported_total_raw, 2) if reported_total_raw else None
        reported_rn = round_financial(reported_rn_raw, 2) if reported_rn_raw else None
        reported_lpn = round_financial(reported_lpn_raw, 2) if reported_lpn_raw else None
        reported_na = round_financial(reported_na_raw, 2) if reported_na_raw else None

        if cmi is None:
            if require_cmi:
                return None
            return {
                'quarter': quarter_key,
                'cmi': None,
                'harrington_total': None,
                'harrington_rn': None,
                'harrington_lpn': None,
                'harrington_na': None,
                'reported_total': reported_total,
                'reported_rn': reported_rn,
                'reported_lpn': reported_lpn,
                'reported_na': reported_na,
                'pct_total': None,
                'pct_rn': None,
                'pct_lpn': None,
                'pct_na': None,
                'pct_harrington_total_staff': None,
                'pct_harrington_direct_staff': None,
                'pct_harrington_rn_total': None,
                'pct_harrington_rn_direct': None,
            }

        harrington_total = calculate_harrington_adjusted_hprd(cmi, 'total')
        harrington_rn = calculate_harrington_adjusted_hprd(cmi, 'rn')
        harrington_lpn = calculate_harrington_residual_lpn_hprd(cmi)
        harrington_na = calculate_harrington_adjusted_hprd(cmi, 'cna')
        pct_harrington_total_staff = (
            round_financial((pbj_total_staff / harrington_total * 100) if (pbj_total_staff and harrington_total and harrington_total > 0) else None, 1)
            if pbj_total_staff and harrington_total and harrington_total > 0
            else None
        )
        pct_harrington_direct_staff = (
            round_financial((pbj_direct_staff / harrington_total * 100) if (pbj_direct_staff and harrington_total and harrington_total > 0) else None, 1)
            if pbj_direct_staff and harrington_total and harrington_total > 0
            else None
        )
        pct_harrington_rn_total = (
            round_financial((pbj_rn_total_staff / harrington_rn * 100) if (pbj_rn_total_staff and harrington_rn and harrington_rn > 0) else None, 1)
            if pbj_rn_total_staff and harrington_rn and harrington_rn > 0
            else None
        )
        pct_harrington_rn_direct = (
            round_financial((pbj_rn_direct_staff / harrington_rn * 100) if (pbj_rn_direct_staff and harrington_rn and harrington_rn > 0) else None, 1)
            if pbj_rn_direct_staff and harrington_rn and harrington_rn > 0
            else None
        )
        pct_total = (
            round_financial((reported_total_raw / harrington_total * 100) if (reported_total_raw and harrington_total and harrington_total > 0) else None, 1)
            if reported_total_raw and harrington_total and harrington_total > 0
            else None
        )
        pct_rn = (
            round_financial((reported_rn_raw / harrington_rn * 100) if (reported_rn_raw and harrington_rn and harrington_rn > 0) else None, 1)
            if reported_rn_raw and harrington_rn and harrington_rn > 0
            else None
        )
        pct_na = (
            round_financial((reported_na_raw / harrington_na * 100) if (reported_na_raw and harrington_na and harrington_na > 0) else None, 1)
            if reported_na_raw and harrington_na and harrington_na > 0
            else None
        )
        pct_lpn = (
            round_financial((reported_lpn_raw / harrington_lpn * 100) if (reported_lpn_raw and harrington_lpn and harrington_lpn > 0) else None, 1)
            if reported_lpn_raw and harrington_lpn and harrington_lpn > 0
            else None
        )
        return {
            'quarter': quarter_key,
            'cmi': round_financial(cmi, 5),
            'harrington_total': harrington_total,
            'harrington_rn': harrington_rn,
            'harrington_lpn': harrington_lpn,
            'harrington_na': harrington_na,
            'reported_total': reported_total,
            'reported_rn': reported_rn,
            'reported_lpn': reported_lpn,
            'reported_na': reported_na,
            'pct_total': pct_total,
            'pct_rn': pct_rn,
            'pct_lpn': pct_lpn,
            'pct_na': pct_na,
            'pct_harrington_total_staff': pct_harrington_total_staff,
            'pct_harrington_direct_staff': pct_harrington_direct_staff,
            'pct_harrington_rn_total': pct_harrington_rn_total,
            'pct_harrington_rn_direct': pct_harrington_rn_direct,
        }

    def _harrington_reference_links(quarter_key: str) -> Dict[str, str]:
        ccn = str(PROVNUM).strip().zfill(6) if PROVNUM else ''
        if not ccn and global_df is not None and len(global_df) > 0 and 'PROVNUM' in global_df.columns:
            ccn = str(global_df['PROVNUM'].iloc[0]).strip().zfill(6)
        out: Dict[str, str] = {
            'data_matching_path': _pbj_premium_facility_href(ccn or None, '/data-matching'),
            'provider_info_dataset_url': _CMS_PROVIDER_INFO_DATASET_PAGE,
        }
        qk = str(quarter_key).strip()
        murl = re.search(r"(?:CY)?(\d{4})Q([1-4])", qk.upper())
        if murl:
            y, qn = murl.group(1), murl.group(2)
            out['pbj_quarter_data_url'] = (
                f"https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data/q{qn.lower()}-{y}"
            )
        st = None
        if global_df is not None and len(global_df) > 0 and 'STATE' in global_df.columns:
            st = str(global_df['STATE'].iloc[0]).strip().upper()
        if ccn and st and len(st) == 2:
            out['medicare_compare_url'] = (
                f"https://www.medicare.gov/care-compare/details/nursing-home/{ccn}/view-all/?state={urllib.parse.quote(st)}"
            )
        return out

    try:
        quarter = request.args.get('quarter')
        use_total = request.args.get('use_total', 'false').lower() == 'true'

        if not quarter:
            return jsonify({'error': 'Quarter parameter required'})

        case_mix_response = get_case_mix_data()
        if isinstance(case_mix_response, tuple):
            case_mix_json = case_mix_response[0].get_json()
        else:
            case_mix_json = case_mix_response.get_json()

        cmd = case_mix_json.get('case_mix_data') or {}

        if quarter == '__ALL__':
            # Primary window from 2024 Q1 (nursing CMI era); pad with earlier quarters only if needed to reach 8 rows.
            _harrington_cmi_table_min_key = (2024, 1)
            _harrington_table_min_rows = 8
            sorted_keys = sorted(cmd.keys(), key=_cy_quarter_sort_key_from_canonical)
            in_window: list[tuple[str, dict]] = []
            before_window: list[tuple[str, dict]] = []
            for qk in sorted_keys:
                qsk = _cy_quarter_sort_key_from_canonical(qk)
                if qsk == (0, 0):
                    continue
                qd = cmd.get(qk)
                if not qd:
                    continue
                if qsk < _harrington_cmi_table_min_key:
                    before_window.append((qk, qd))
                else:
                    in_window.append((qk, qd))
            rows_out = [
                r
                for r in (
                    _harrington_row_for_quarter(qk, qd, use_total, require_cmi=False) for qk, qd in in_window
                )
                if r is not None
            ]
            if len(rows_out) < _harrington_table_min_rows and before_window:
                before_desc = sorted(
                    before_window,
                    key=lambda t: _cy_quarter_sort_key_from_canonical(t[0]),
                    reverse=True,
                )
                for qk, qd in before_desc:
                    if len(rows_out) >= _harrington_table_min_rows:
                        break
                    row = _harrington_row_for_quarter(qk, qd, use_total, require_cmi=False)
                    if row is not None:
                        rows_out.append(row)
            # Newest quarter first (Q3 2025 before Q1 2024).
            rows_out.sort(
                key=lambda r: _cy_quarter_sort_key_from_canonical(r.get("quarter")),
                reverse=True,
            )
            if not rows_out:
                return jsonify({'error': 'No case-mix quarters in range for Harrington table.'})
            return jsonify(
                {
                    'mode': 'all_quarters',
                    'use_total': use_total,
                    'rows': rows_out,
                    'meta': {
                        'cmi_era_from_sort_key': list(_harrington_cmi_table_min_key),
                        'cmi_era_label': '2024 Q1',
                    },
                    'reference_links': _harrington_reference_links(''),
                }
            )

        cq = _provider_quarter_to_canonical(quarter)
        quarter_data = cmd.get(quarter) if quarter in cmd else None
        resolved_key = quarter
        if quarter_data is None and cq:
            quarter_data = cmd.get(cq)
            resolved_key = cq
        if quarter_data is None and cq:
            for k, v in cmd.items():
                if _provider_quarter_to_canonical(k) == cq:
                    quarter_data = v
                    resolved_key = k
                    break
        if not quarter_data:
            return jsonify({'error': f'No data found for quarter {quarter}'})

        bundle = _harrington_row_for_quarter(resolved_key, quarter_data, use_total, require_cmi=True)
        if not bundle:
            return jsonify(
                {
                    'error': f'No CMI available for quarter {quarter}. Case Mix Index (CMI) comes from CMS Provider Info; it may be missing or not yet published for this quarter.'
                }
            )
        bundle['mode'] = 'single'
        bundle['use_total'] = use_total
        bundle['reference_links'] = _harrington_reference_links(resolved_key)
        return jsonify(bundle)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)})


@app.route("/api/entity-longitudinal-metrics")
def api_entity_longitudinal_metrics():
    """CMS chain longitudinal metrics for the affiliated entity (same data as the entity dashboard)."""
    try:
        try:
            from pbj_identifiers.validators import normalize_entity_id
        except ImportError:
            def normalize_entity_id(entity_id):
                if entity_id is None:
                    return None
                entity_id = str(entity_id).strip()
                if not entity_id:
                    return None
                if "." in entity_id:
                    entity_id = entity_id.split(".")[0]
                entity_id = "".join(c for c in entity_id if c.isdigit())
                return entity_id or None

        raw = (request.args.get("entity_id") or "").strip()
        if not raw:
            return jsonify({"error": "Missing entity_id parameter"}), 400
        entity_id = normalize_entity_id(raw)
        if not entity_id:
            return jsonify({"error": f"Invalid entity ID: {raw}"}), 400

        from entity_longitudinal_metrics import get_entity_key_metrics_over_time

        ccn = _ein_active_ccn()
        facility_ccn = ccn if ccn and str(ccn).strip() not in ("", "Unknown") else None

        metrics = get_entity_key_metrics_over_time(entity_id, facility_ccn=facility_ccn)
        if metrics is None:
            return jsonify(
                {
                    "entity_id": entity_id,
                    "available": False,
                    "message": "No longitudinal data available for this entity",
                }
            )
        metrics["available"] = True
        return jsonify(metrics)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/entity-longitudinal-facility-bundle")
def api_entity_longitudinal_facility_bundle():
    """
    Longitudinal chain metrics for every distinct affiliated entity (chain) in this
    facility's provider history — supports current vs prior ownership without shipping
    the full national longitudinal file when a per-CCN slice is deployed.
    """
    try:
        from entity_longitudinal_metrics import (
            affiliated_chain_ids_for_facility_provider_df,
            get_entity_key_metrics_over_time,
        )

        global provider_info_df

        ccn = _ein_active_ccn()
        if not ccn or str(ccn).strip() in ("", "Unknown"):
            return jsonify(
                {
                    "available": False,
                    "message": "Facility not loaded",
                    "entities": [],
                    "chain_ids": [],
                }
            )
        facility_ccn = str(ccn).strip().zfill(6)
        chain_ids: list[str] = []
        if provider_info_df is not None and not provider_info_df.empty and "ccn" in provider_info_df.columns:
            ccn_mask = provider_info_df["ccn"].astype(str).str.strip().str.zfill(6) == facility_ccn
            sub = cast(pd.DataFrame, provider_info_df.loc[ccn_mask])
            chain_ids = affiliated_chain_ids_for_facility_provider_df(sub)
        entities: list[dict] = []
        for cid in chain_ids:
            m = get_entity_key_metrics_over_time(cid, facility_ccn=facility_ccn)
            if m:
                entities.append(m)
        if not entities:
            return jsonify(
                {
                    "available": False,
                    "message": "No longitudinal data for this facility's affiliation history",
                    "entities": [],
                    "chain_ids": chain_ids,
                }
            )
        return jsonify({"available": True, "entities": entities, "chain_ids": chain_ids})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ein-position-summary")
def api_ein_position_summary():
    """PBJ Employee Detail aggregates (job/category by quarter)."""
    global ein_job_quarterly_df, ein_category_quarterly_df, ein_employee_detail_df
    try:
        if not _ein_mode_enabled():
            return jsonify({"available": False, "message": "Employee Detail mode is disabled for this dashboard."})
        if ein_job_quarterly_df is None or ein_category_quarterly_df is None:
            _load_ein_position_csvs()
        ccn = _ein_active_ccn()
        if ein_job_quarterly_df is None or ein_job_quarterly_df.empty:
            return jsonify(
                {
                    "available": False,
                    "message": (
                        f"No Employee Detail aggregates found. Run: "
                        f"python scripts/extract_facility_ein_from_zip.py {ccn} --detail-only-summarize. "
                        f"For per-employee nursing charts, omit --detail-only-summarize so "
                        f"facility_{ccn}_ein_employee_detail.parquet is written."
                    ),
                }
            )

        def _records(df: pd.DataFrame | None) -> list:
            if df is None or df.empty:
                return []
            out = df.replace({np.nan: None})
            return out.to_dict(orient="records")

        # Quarters: union job/category aggregates and row-level detail (detail may include more
        # CY quarters than filtered aggregate files, e.g. Vercel EIN_SELECTED_QUARTERS).
        q_from_job = {str(x) for x in ein_job_quarterly_df["CY_Qtr"].dropna().astype(str).unique()}
        q_union = set(q_from_job)
        _ensure_ein_employee_detail_loaded()
        dprep_all: pd.DataFrame | None = None
        if ein_employee_detail_df is not None and not ein_employee_detail_df.empty:
            dprep_all = prepare_ein_detail(ein_employee_detail_df)
            if dprep_all is not None and not dprep_all.empty and "CY_Qtr_norm" in dprep_all.columns:
                for qv in dprep_all["CY_Qtr_norm"].dropna().astype(str).unique():
                    q_union.add(str(qv))
        q_sorted = sorted(
            q_union,
            key=lambda x: parse_ein_quarter_bound(str(x)) or 0,
            reverse=True,
        )
        contract_pct = []
        for q in q_sorted:
            if str(q) in q_from_job:
                sub = ein_job_quarterly_df[ein_job_quarterly_df["CY_Qtr"].astype(str) == str(q)]
                te = float(sub["hours_employee"].sum()) if "hours_employee" in sub.columns else 0.0
                tc = float(sub["hours_contract"].sum()) if "hours_contract" in sub.columns else 0.0
                tot = te + tc
                contract_pct.append(round((tc / tot) * 100, 2) if tot > 0 else 0.0)
            else:
                if dprep_all is None or dprep_all.empty:
                    contract_pct.append(0.0)
                    continue
                sub = dprep_all[dprep_all["CY_Qtr_norm"].astype(str) == str(q)]
                if sub.empty:
                    contract_pct.append(0.0)
                else:
                    is_ctr = sub["EMP_CTR"].fillna(0).eq(2)
                    tc = float(sub.loc[is_ctr, "WORK_HRS_NUM"].sum())
                    te = float(sub.loc[~is_ctr, "WORK_HRS_NUM"].sum())
                    tot = te + tc
                    contract_pct.append(round((tc / tot) * 100, 2) if tot > 0 else 0.0)

        zip_ok = ein_employee_detail_sources_available(_app_root)
        q_chrono = sorted(
            q_sorted,
            key=lambda x: parse_ein_quarter_bound(str(x)) or 0,
        )
        span_note = (
            f"{q_chrono[0]} through {q_chrono[-1]} ({len(q_sorted)} quarters in this extract)"
            if len(q_sorted) > 1
            else (q_sorted[0] if q_sorted else "no quarters")
        )
        cms_quarter_urls: dict[str, str | None] = {}
        for qk in q_sorted:
            cms_quarter_urls[str(qk)] = build_cms_ein_detail_explorer_url(ccn, str(qk))
        return jsonify(
            {
                "available": True,
                "provnum": ccn,
                "quarters": q_sorted,
                "quarter_note": span_note,
                "contract_pct_of_hours": contract_pct,
                "by_job": _records(ein_job_quarterly_df),
                "by_category": _records(ein_category_quarterly_df),
                "data_dictionary_path": "EIN/data_dictionary.md",
                "ein_zip_present": zip_ok,
                "cms_ein_landing_url": CMS_EIN_DETAIL_LANDING_URL,
                "cms_ein_quarter_urls": cms_quarter_urls,
            }
        )
    except Exception as exc:
        return jsonify({"available": False, "error": str(exc)})


def _load_ein_employee_detail_only() -> None:
    """Load row-level employee detail only (for series, roster, bridge)."""
    global ein_employee_detail_df
    from file_path_utils import find_facility_ein_table_base

    prov = str(_ein_active_ccn()).strip().zfill(6)
    detail_base = find_facility_ein_table_base(prov, "employee_detail")
    if not detail_base:
        detail_base = os.path.join(_app_root, f"facility_{prov}_ein_employee_detail")
    ein_employee_detail_df = _filter_ein_by_selected_quarters(read_facility_ein_parquet_or_csv(detail_base))
    if ein_employee_detail_df is not None and not ein_employee_detail_df.empty:
        print(f"[EIN] Loaded row-level detail on demand ({len(ein_employee_detail_df):,} rows)")


def _ein_scalar_for_json(val):
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if hasattr(val, "item"):
        try:
            return val.item()
        except Exception:
            return val
    return val


def _enrich_nursing_api_rows(rows: list, ccn: str) -> None:
    """Add CMS explorer URLs to nursing summary rows (mutates rows in place)."""
    for row in rows:
        qn = row.get("quarter")
        jc = row.get("job_code")
        sid_raw = row.get("sys_employee_id")
        sid_i: int | None = None
        if sid_raw is not None:
            try:
                sid_i = int(sid_raw)
            except (TypeError, ValueError):
                sid_i = None

        def _cms_q_for_date(iso: str | None) -> str:
            return iso_date_to_cy_quarter(iso) or str(qn)

        if qn is None:
            row["cms_explorer_url"] = None
            row["first_work_date_cms_url"] = None
            row["last_work_date_cms_url"] = None
            row["max_hours_work_date_cms_url"] = None
            row["min_hours_work_date_cms_url"] = None
            row["tenure_days_career_in_file"] = None
            row["tenure_label"] = None
            continue

        fd = row.get("first_work_date")
        ld = row.get("last_work_date")
        mxd = row.get("max_hours_work_date")
        mnd = row.get("min_hours_work_date")

        if sid_i is not None:
            row["cms_explorer_url"] = build_cms_ein_detail_explorer_url(
                ccn, str(qn), sys_employee_id=sid_i
            )
            row["first_work_date_cms_url"] = (
                build_cms_ein_detail_explorer_url(
                    ccn, _cms_q_for_date(fd), work_date=fd, sys_employee_id=sid_i
                )
                if fd
                else None
            )
            row["last_work_date_cms_url"] = (
                build_cms_ein_detail_explorer_url(
                    ccn, _cms_q_for_date(ld), work_date=ld, sys_employee_id=sid_i
                )
                if ld
                else None
            )
            row["max_hours_work_date_cms_url"] = (
                build_cms_ein_detail_explorer_url(
                    ccn, _cms_q_for_date(mxd), work_date=mxd, sys_employee_id=sid_i
                )
                if mxd
                else None
            )
            row["min_hours_work_date_cms_url"] = (
                build_cms_ein_detail_explorer_url(
                    ccn, _cms_q_for_date(mnd), work_date=mnd, sys_employee_id=sid_i
                )
                if mnd
                else None
            )
        elif jc is not None:
            jci = int(jc)
            row["cms_explorer_url"] = build_cms_ein_detail_explorer_url(ccn, str(qn), job_code=jci)
            row["first_work_date_cms_url"] = (
                build_cms_ein_detail_explorer_url(ccn, _cms_q_for_date(fd), work_date=fd, job_code=jci)
                if fd
                else None
            )
            row["last_work_date_cms_url"] = (
                build_cms_ein_detail_explorer_url(ccn, _cms_q_for_date(ld), work_date=ld, job_code=jci)
                if ld
                else None
            )
            row["max_hours_work_date_cms_url"] = (
                build_cms_ein_detail_explorer_url(ccn, _cms_q_for_date(mxd), work_date=mxd, job_code=jci)
                if mxd
                else None
            )
            row["min_hours_work_date_cms_url"] = (
                build_cms_ein_detail_explorer_url(ccn, _cms_q_for_date(mnd), work_date=mnd, job_code=jci)
                if mnd
                else None
            )
        else:
            row["cms_explorer_url"] = None
            row["first_work_date_cms_url"] = None
            row["last_work_date_cms_url"] = None
            row["max_hours_work_date_cms_url"] = None
            row["min_hours_work_date_cms_url"] = None

        span_days = ein_span_days_from_workdate_raws(
            row.get("first_work_date_raw"), row.get("last_work_date_raw")
        )
        censored = bool(row.get("tenure_censored_at_file_start"))
        row["tenure_days_career_in_file"] = span_days
        if span_days is not None:
            row["tenure_label"] = format_ein_tenure_span_days(int(span_days), at_least=censored)
        else:
            row["tenure_label"] = None


def _ein_nursing_summaries_rows_all() -> list[dict[str, Any]]:
    """Process precomputed nursing summaries once per loaded dataframe."""
    global _EIN_NURSING_SUMMARIES_ROWS_CACHE, _EIN_NURSING_SUMMARIES_ROWS
    df = ein_nursing_summaries_df
    if df is None or df.empty:
        return []
    sig = (str(_ein_active_ccn()).strip().zfill(6), id(df), len(df))
    if _EIN_NURSING_SUMMARIES_ROWS_CACHE == sig and _EIN_NURSING_SUMMARIES_ROWS is not None:
        return _EIN_NURSING_SUMMARIES_ROWS

    def _json_rows_from_df(df_in: pd.DataFrame) -> list[dict[str, Any]]:
        rows_out: list[dict[str, Any]] = []
        for r in df_in.to_dict(orient="records"):
            rows_out.append({k: _ein_scalar_for_json(v) for k, v in r.items()})
        return rows_out

    rows_all = dedupe_nursing_roster_api_rows(_json_rows_from_df(cast(pd.DataFrame, df)))
    enrich_nursing_roster_display_fields(rows_all)
    enrich_nursing_rows_new_to_quarter_flags(rows_all)
    enrich_nursing_rows_multi_role_flags(rows_all)
    _EIN_NURSING_SUMMARIES_ROWS_CACHE = sig
    _EIN_NURSING_SUMMARIES_ROWS = rows_all
    return rows_all


def _warm_ein_sustained_work_cache(prov: str) -> None:
    """Precompute sustained-work flags once per process so roster day view stays fast."""
    if not _ein_mode_enabled():
        return
    try:
        _ensure_ein_employee_detail_loaded()
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return
        df = ein_nursing_summaries_df
        if df is None or df.empty or "quarter" not in df.columns:
            return
        quarters = sorted(
            {
                normalize_cy_qtr_ein(q)
                for q in df["quarter"].astype(str).tolist()
                if normalize_cy_qtr_ein(q)
            }
        )
        if not quarters:
            return
        latest_q = quarters[-1]
        dummy: list[dict[str, Any]] = [{"quarter": latest_q, "sys_employee_id": 0, "job_code": 1}]
        enrich_nursing_rows_sustained_work_flags(
            ein_employee_detail_df,
            dummy,
            facility_ccn=str(prov).strip().zfill(6),
            limit_quarters=[latest_q],
        )
        _ein_nursing_summaries_rows_all()
        print(f"[EIN] Pre-warmed roster caches for {latest_q}")
    except Exception as exc:
        print(f"[EIN] Roster pre-warm skipped: {exc}")


def _schedule_ein_roster_prewarm(prov: str) -> None:
    """Warm roster caches in a daemon thread so home-page init is not blocked."""
    global _EIN_ROSTER_PREWARM_STARTED
    if not _ein_mode_enabled():
        return
    if ein_nursing_summaries_df is None or ein_nursing_summaries_df.empty:
        return
    with _EIN_ROSTER_PREWARM_LOCK:
        if _EIN_ROSTER_PREWARM_STARTED:
            return
        _EIN_ROSTER_PREWARM_STARTED = True

    def _run() -> None:
        try:
            _warm_ein_sustained_work_cache(prov)
        except Exception as exc:
            print(f"[EIN] Background roster pre-warm failed: {exc}")

    threading.Thread(
        target=_run,
        name=f"ein-roster-prewarm-{str(prov).strip().zfill(6)}",
        daemon=True,
    ).start()

def _ensure_ein_employee_detail_loaded() -> None:
    global ein_employee_detail_df
    if ein_employee_detail_df is None:
        _load_ein_employee_detail_only()


_EIN_HEADCOUNT_API_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_EIN_HEADCOUNT_CACHE_TTL_SEC = 900.0


def _ein_headcount_api_cache_get(key: str) -> dict[str, Any] | None:
    row = _EIN_HEADCOUNT_API_CACHE.get(key)
    if not row:
        return None
    ts, payload = row
    if (time.time() - ts) > _EIN_HEADCOUNT_CACHE_TTL_SEC:
        _EIN_HEADCOUNT_API_CACHE.pop(key, None)
        return None
    return payload


def _ein_headcount_api_cache_set(key: str, payload: dict[str, Any]) -> None:
    _EIN_HEADCOUNT_API_CACHE[key] = (time.time(), payload)


def _pbj_day_census_value(date_iso: str) -> float | None:
    """Best-effort daily census lookup from PBJ complete-data row for one date."""
    global global_df
    try:
        row = get_pbj_complete_data_row_for_date(global_df, date_iso)
    except Exception:
        return None
    if row is None:
        return None
    for k in (
        "MDScensus",
        "mdscensus",
        "Census",
        "census",
        "MedRes",
        "medres",
        "resident_census",
        "Resident_Census",
    ):
        try:
            v = row.get(k) if hasattr(row, "get") else None
            if v is None:
                continue
            fv = float(v)
            if np.isfinite(fv):
                return fv
        except Exception:
            continue
    return None


def _ein_employee_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Counts for bridge/day-roster modal summary chips."""
    total = int(len(rows or []))
    rn = 0
    lpn = 0
    aide = 0
    admin = 0
    contract = 0
    for r in rows or []:
        jc_raw = r.get("job_code")
        try:
            jc = int(jc_raw) if jc_raw is not None else -1
        except (TypeError, ValueError):
            jc = -1
        if 5 <= jc <= 7:
            rn += 1
        elif 8 <= jc <= 9:
            lpn += 1
        elif 10 <= jc <= 12:
            aide += 1
        elif jc == 1:
            admin += 1
        try:
            pct = float(r.get("pct_contract_hours") or 0.0)
        except Exception:
            pct = 0.0
        if pct > 0.0:
            contract += 1
    return {
        "total_employees": total,
        "rn_count": rn,
        "lpn_count": lpn,
        "nurse_aide_count": aide,
        "admin_count": admin,
        "contract_count": contract,
    }


@app.route("/api/ein-cms-explorer-url")
def api_ein_cms_explorer_url():
    """Build CMS Employee Detail explorer URL."""
    try:
        q = request.args.get("quarter")
        if not q or not str(q).strip():
            return jsonify({"ok": False, "message": "quarter is required"})
        prov = request.args.get("provnum") or _ein_active_ccn()
        work_date = request.args.get("work_date")
        job_code = request.args.get("job_code", type=int)
        jlo = request.args.get("job_min", type=int)
        jhi = request.args.get("job_max", type=int)
        between = (jlo, jhi) if jlo is not None and jhi is not None else None
        url = build_cms_ein_detail_explorer_url(
            str(prov).strip(),
            str(q).strip(),
            work_date=work_date if work_date else None,
            job_code=job_code if between is None else None,
            job_code_between=between,
        )
        return jsonify({"ok": bool(url), "url": url, "cms_ein_landing_url": CMS_EIN_DETAIL_LANDING_URL})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)})


@app.route("/api/ein-nursing-employees")
def api_ein_nursing_employees():
    """Nursing job-code employee summaries (precomputed Parquet or row-level EIN)."""
    global ein_employee_detail_df, ein_nursing_summaries_df
    try:
        if not _ein_mode_enabled():
            return jsonify({"available": False, "employees": [], "message": "Employee Detail mode is disabled for this dashboard."})
        ccn = _ein_active_ccn()
        quarter = request.args.get("quarter") or "all"
        work_date_anchor = (request.args.get("work_date") or "").strip()
        offset = request.args.get("offset", type=int) or 0
        offset = max(0, int(offset))
        limit_raw = request.args.get("limit", type=int)
        q_filter_norm = (
            normalize_cy_qtr_ein(quarter)
            if quarter and str(quarter).strip().lower() not in ("", "all")
            else None
        )
        position_group = (request.args.get("position_group") or "all").strip().lower()

        def _json_rows_from_df(df_in: pd.DataFrame) -> list[dict[str, Any]]:
            rows_out: list[dict[str, Any]] = []
            for r in df_in.to_dict(orient="records"):
                rows_out.append({k: _ein_scalar_for_json(v) for k, v in r.items()})
            return rows_out

        work_anchor = (work_date_anchor or "").strip()
        if work_anchor:
            _ensure_ein_employee_detail_loaded()

        detail_ready = (
            ein_employee_detail_df is not None and not ein_employee_detail_df.empty
        )
        # Prefer row-level detail when a work-day anchor is set and the extract loaded; otherwise
        # still serve precomputed summaries so the roster is usable (rolling day fields skip).
        use_precomputed = (
            ein_nursing_summaries_df is not None
            and not ein_nursing_summaries_df.empty
            and (not work_anchor or not detail_ready)
        )
        if use_precomputed:
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
            )

        _ensure_ein_employee_detail_loaded()
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return jsonify(
                {
                    "available": False,
                    "employees": [],
                    "message": (
                        f"Employee-level roster detail is not loaded for this deployment yet. "
                        f"Include facility_{ccn}_ein_employee_detail.parquet or "
                        f"facility_{ccn}_ein_nursing_summaries.parquet to enable full row-level Employee Tracker views."
                    ),
                }
            )
        rows_all = dedupe_nursing_roster_api_rows(
            nursing_employee_summaries(ein_employee_detail_df, quarter="all")
        )
        enrich_nursing_roster_display_fields(rows_all)
        enrich_nursing_rows_new_to_quarter_flags(rows_all)
        enrich_nursing_rows_multi_role_flags(rows_all)
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
        if q_filter_norm:
            rows_filtered = [
                r for r in rows_all if normalize_cy_qtr_ein(r.get("quarter")) == q_filter_norm
            ]
        else:
            rows_filtered = rows_all
        if position_group and position_group != "all":
            rows_filtered = [
                r for r in rows_filtered if ein_job_code_matches_position_group(r.get("job_code"), position_group)
            ]
        apply_roster_tenure_quarter_span(rows_filtered, rows_for_global_bounds=rows_all)
        total = len(rows_filtered)
        rows_filtered.sort(
            key=lambda row: (
                -(parse_ein_quarter_bound(str(row.get("quarter") or "")) or 0),
                -float(row.get("total_hours") or 0),
                int(row.get("first_work_date_raw") or 0) or 10**9,
                int(row.get("sys_employee_id") or 0),
                int(row.get("job_code") or 0),
            )
        )
        if limit_raw is None:
            rows = rows_filtered[offset:]
        else:
            lim = max(1, int(limit_raw))
            rows = rows_filtered[offset : offset + lim]
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
                "source": "detail",
            }
        )
    except Exception as exc:
        return jsonify({"available": False, "employees": [], "error": str(exc)})


@app.route("/api/ein-nursing-employee-series")
def api_ein_nursing_employee_series():
    """Daily hours series for modal chart."""
    global ein_employee_detail_df
    try:
        if not _ein_mode_enabled():
            return jsonify({"available": False, "message": "Employee Detail mode is disabled for this dashboard."})
        _ensure_ein_employee_detail_loaded()
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return jsonify({"available": False, "message": "No row-level Employee Detail CSV loaded."})
        sid = request.args.get("sys_employee_id")
        jcid = request.args.get("job_code")
        q = request.args.get("quarter")
        if not sid or not jcid or not q:
            return jsonify({"available": False, "message": "sys_employee_id, job_code, and quarter required."})
        try:
            sid_i = int(str(sid).strip())
            jcid_i = int(str(jcid).strip())
        except ValueError:
            return jsonify({"available": False, "message": "Invalid sys_employee_id or job_code."})
        payload = nursing_employee_daily_series(ein_employee_detail_df, sid_i, jcid_i, q)
        if not payload:
            return jsonify({"available": False, "message": "No rows for that employee/job/quarter."})
        ccn = _ein_active_ccn()
        qn = payload.get("quarter") or q

        def _cms_q_for_date(iso_d: str | None) -> str:
            return iso_date_to_cy_quarter(iso_d) or str(qn)

        cms = {
            "landing_url": CMS_EIN_DETAIL_LANDING_URL,
            "facility_quarter": build_cms_ein_detail_explorer_url(ccn, str(qn)),
            "facility_quarter_job": build_cms_ein_detail_explorer_url(
                ccn, str(qn), sys_employee_id=sid_i
            ),
        }
        fd = payload.get("first_work_date")
        ld = payload.get("last_work_date")
        qfd = payload.get("quarter_first_work_date")
        qld = payload.get("quarter_last_work_date")
        mxd = payload.get("max_hours_work_date")
        mnd = payload.get("min_hours_work_date")
        payload["first_work_date_cms_url"] = (
            build_cms_ein_detail_explorer_url(
                ccn, _cms_q_for_date(fd), work_date=fd, sys_employee_id=sid_i
            )
            if fd
            else None
        )
        payload["last_work_date_cms_url"] = (
            build_cms_ein_detail_explorer_url(
                ccn, _cms_q_for_date(ld), work_date=ld, sys_employee_id=sid_i
            )
            if ld
            else None
        )
        payload["quarter_first_work_date_cms_url"] = (
            build_cms_ein_detail_explorer_url(ccn, str(qn), work_date=qfd, sys_employee_id=sid_i)
            if qfd
            else None
        )
        payload["quarter_last_work_date_cms_url"] = (
            build_cms_ein_detail_explorer_url(ccn, str(qn), work_date=qld, sys_employee_id=sid_i)
            if qld
            else None
        )
        payload["max_hours_work_date_cms_url"] = (
            build_cms_ein_detail_explorer_url(
                ccn, _cms_q_for_date(mxd), work_date=mxd, sys_employee_id=sid_i
            )
            if mxd
            else None
        )
        payload["min_hours_work_date_cms_url"] = (
            build_cms_ein_detail_explorer_url(
                ccn, _cms_q_for_date(mnd), work_date=mnd, sys_employee_id=sid_i
            )
            if mnd
            else None
        )
        return jsonify({"available": True, "series": payload, "cms": cms, "provnum": ccn})
    except Exception as exc:
        return jsonify({"available": False, "error": str(exc)})


def _nursing_quarters_for_employee_from_summaries(sid_i: int, jcid_i: int) -> list[str]:
    """Distinct CY quarters for one employee + job from precomputed nursing summaries."""
    global ein_nursing_summaries_df
    df = ein_nursing_summaries_df
    if df is None or df.empty:
        return []
    for col in ("sys_employee_id", "job_code", "quarter"):
        if col not in df.columns:
            return []
    m1 = pd.to_numeric(df["sys_employee_id"], errors="coerce").fillna(-1).astype(int) == sid_i
    m2 = pd.to_numeric(df["job_code"], errors="coerce").fillna(-1).astype(int) == jcid_i
    sub = df.loc[m1 & m2]
    if sub.empty:
        return []
    raw = {str(x).strip() for x in sub["quarter"].dropna().astype(str).unique() if str(x).strip()}
    return sorted(raw, key=lambda x: parse_ein_quarter_bound(str(x)) or 0, reverse=True)


@app.route("/api/ein-nursing-employee-quarters")
def api_ein_nursing_employee_quarters():
    """Quarters present in the loaded extract for one employee + job (for modal quarter picker)."""
    global ein_employee_detail_df
    try:
        if not _ein_mode_enabled():
            return jsonify({"available": False, "quarters": [], "message": "Employee Detail mode is disabled."})
        sid = request.args.get("sys_employee_id")
        jcid = request.args.get("job_code")
        if not sid or not jcid:
            return jsonify({"available": False, "quarters": [], "message": "sys_employee_id and job_code required."})
        try:
            sid_i = int(str(sid).strip())
            jcid_i = int(str(jcid).strip())
        except ValueError:
            return jsonify({"available": False, "quarters": [], "message": "Invalid sys_employee_id or job_code."})
        _ensure_ein_employee_detail_loaded()
        qs: list[str] = []
        if ein_employee_detail_df is not None and not ein_employee_detail_df.empty:
            qs = nursing_employee_quarters_for_job(ein_employee_detail_df, sid_i, jcid_i)
        if not qs:
            qs = _nursing_quarters_for_employee_from_summaries(sid_i, jcid_i)
        if not qs:
            return jsonify({"available": False, "quarters": [], "message": "No row-level Employee Detail loaded."})
        ccn = _ein_active_ccn()
        return jsonify({"available": True, "quarters": qs, "provnum": ccn})
    except Exception as exc:
        return jsonify({"available": False, "quarters": [], "error": str(exc)})


@app.route("/api/ein-pbj-bridge")
def api_ein_pbj_bridge():
    """Map a daily PBJ metric bucket to Employee Detail rows for one calendar date."""
    global ein_employee_detail_df, global_df
    try:
        if not _ein_mode_enabled():
            return jsonify({"ok": False, "available": False, "message": "Employee Detail mode is disabled for this dashboard."})
        _ensure_ein_employee_detail_loaded()
        date_s = (request.args.get("date") or "").strip()
        metric = (request.args.get("metric") or "").strip()
        if not date_s or not metric:
            return jsonify({"ok": False, "message": "date and metric are required"})
        if metric not in PBJ_METRIC_TO_EIN_JOB_CODES:
            return jsonify({"ok": False, "message": f"unknown metric: {metric}"})
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return jsonify(
                {
                    "ok": False,
                    "available": False,
                    "message": "Row-level Employee Detail file not loaded.",
                }
            )
        rows, total, qn, note = ein_pbj_bridge_for_metric(
            ein_employee_detail_df, date_s, metric
        )
        ccn = _ein_active_ccn()
        prov_cms = str(ccn).strip().zfill(6) if str(ccn).strip().isdigit() else str(ccn).strip()
        cms_day = (
            build_cms_ein_detail_explorer_url(prov_cms, str(qn), work_date=date_s)
            if qn
            else None
        )
        codes_t = PBJ_METRIC_TO_EIN_JOB_CODES[metric]
        p_row = get_pbj_complete_data_row_for_date(global_df, date_s)
        pbj_sum = sum_pbj_hours_for_bridge_metric(p_row, metric)
        if pbj_sum is None and metric in PBJ_NONNURSE_HRS_COLUMN_TO_EIN_JOB_CODES:
            nn_row = _nonnurse_row_for_work_date_iso(date_s)
            if nn_row is not None and metric in nn_row.index and pd.notna(nn_row[metric]):
                pbj_sum = round_financial(float(nn_row[metric]), 2)
        cross: dict = compare_pbj_vs_ein_hours(pbj_sum, float(total))
        cross["pbj_columns_used"] = list(PBJ_METRIC_TO_DAILY_COLUMNS.get(metric, ()))
        cav = PBJ_BRIDGE_COMPARE_CAVEATS.get(metric)
        if cav:
            cross["compare_caveat"] = cav
        day_census = _pbj_day_census_value(date_s)
        breakdown = _ein_employee_breakdown(rows)
        return jsonify(
            {
                "ok": True,
                "available": True,
                "date": date_s,
                "metric": metric,
                "metric_display_name": pbj_metric_display_name(metric),
                "ein_job_codes": list(codes_t),
                "ein_job_codes_label": format_ein_job_codes_for_ui(codes_t),
                "bridge_note": note,
                "employees": rows,
                "hours_total_ein": round(total, 2),
                "day_census": day_census,
                "employee_breakdown": breakdown,
                "hours_cross_check": cross,
                "quarter": qn,
                "cms_explorer_day_url": cms_day,
                "cms_landing_url": CMS_EIN_DETAIL_LANDING_URL,
                "provnum": ccn,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)})


@app.route("/api/ein-day-roster")
def api_ein_day_roster():
    """Employee Detail roster: job code 1 (Administrator) and nursing codes 5–12 for one work date."""
    global ein_employee_detail_df, global_df
    try:
        if not _ein_mode_enabled():
            return jsonify({"ok": False, "available": False, "message": "Employee Detail mode is disabled for this dashboard."})
        _ensure_ein_employee_detail_loaded()
        date_s = (request.args.get("date") or "").strip()
        if not date_s:
            return jsonify({"ok": False, "message": "date is required"})
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return jsonify(
                {
                    "ok": False,
                    "available": False,
                    "message": "Row-level Employee Detail file not loaded.",
                }
            )
        rows, total, qn = ein_nursing_roster_for_work_date(ein_employee_detail_df, date_s)
        headcount = ein_headcount_buckets_for_work_date(ein_employee_detail_df, date_s)
        day_census = _pbj_day_census_value(date_s)
        breakdown = _ein_employee_breakdown(rows)
        ccn = _ein_active_ccn()
        prov_cms = str(ccn).strip().zfill(6) if str(ccn).strip().isdigit() else str(ccn).strip()
        cms_day = (
            build_cms_ein_detail_explorer_url(prov_cms, str(qn), work_date=date_s)
            if qn
            else None
        )
        return jsonify(
            {
                "ok": True,
                "available": True,
                "date": date_s,
                "metric_display_name": "Nursing roster",
                "ein_job_codes_label": EIN_NURSING_ROSTER_CODES_LABEL,
                "bridge_note": EIN_NURSING_ROSTER_BRIDGE_NOTE,
                "employees": rows,
                "hours_total_ein": round(total, 2),
                "day_census": day_census,
                "employee_breakdown": breakdown,
                "quarter": qn,
                "cms_explorer_day_url": cms_day,
                "cms_landing_url": CMS_EIN_DETAIL_LANDING_URL,
                "provnum": ccn,
                "headcount_metrics": headcount,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)})


@app.route("/api/ein-daily-metric-headcounts")
def api_ein_daily_metric_headcounts():
    """Batch EIN distinct headcounts aligned to PBJ bridge metrics over a work-date range."""
    global ein_employee_detail_df
    try:
        if not _ein_mode_enabled():
            return jsonify(
                {
                    "ok": False,
                    "available": False,
                    "message": "Employee Detail mode is disabled for this dashboard.",
                }
            )
        _ensure_ein_employee_detail_loaded()
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return jsonify(
                {
                    "ok": False,
                    "available": False,
                    "message": "Row-level Employee Detail file not loaded.",
                }
            )
        fq = (request.args.get("from") or "").strip()[:10]
        tq = (request.args.get("to") or "").strip()[:10]
        if len(fq) < 10 or len(tq) < 10:
            return jsonify({"ok": False, "message": "from and to are required (YYYY-MM-DD)."})
        d0 = datetime.strptime(fq, "%Y-%m-%d")
        d1 = datetime.strptime(tq, "%Y-%m-%d")
        if d1 < d0:
            d0, d1 = d1, d0
        span_days = (d1 - d0).days + 1
        if span_days > 800:
            return jsonify(
                {
                    "ok": False,
                    "message": "Date range exceeds 800 days; narrow from/to.",
                }
            )
        metrics_raw = (request.args.get("metrics") or "").strip()
        mkeys: list[str] | None = None
        if metrics_raw:
            mkeys = [p.strip() for p in metrics_raw.split(",") if p.strip()]
        out = ein_daily_metric_headcounts_range(
            ein_employee_detail_df,
            fq,
            tq,
            metric_keys=mkeys,
        )
        out["available"] = True
        return jsonify(out)
    except ValueError as exc:
        return jsonify({"ok": False, "message": f"Invalid date: {exc}"})
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)})


@app.route("/api/ein-headcount-quarters")
def api_ein_headcount_quarters():
    """Longitudinal EIN headcount and contract staffing by CY quarter (row-level Employee Detail)."""
    global ein_employee_detail_df
    try:
        if not _ein_mode_enabled():
            return jsonify(
                {
                    "ok": False,
                    "available": False,
                    "message": "Employee Detail mode is disabled for this dashboard.",
                }
            )
        _ensure_ein_employee_detail_loaded()
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return jsonify(
                {
                    "ok": False,
                    "available": False,
                    "message": "Row-level Employee Detail file not loaded.",
                }
            )
        fq = (request.args.get("from_quarter") or "").strip()
        tq = (request.args.get("to_quarter") or "").strip()
        min_sk = parse_ein_quarter_bound(fq) if fq else None
        max_sk = parse_ein_quarter_bound(tq) if tq else None
        series = ein_headcount_quarter_series(
            ein_employee_detail_df,
            min_sort_key=min_sk,
            max_sort_key=max_sk,
        )
        ccn = _ein_active_ccn()
        return jsonify(
            {
                "ok": True,
                "available": True,
                "quarters": series,
                "provnum": ccn,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)})


@app.route("/api/ein-headcount-by-job")
def api_ein_headcount_by_job():
    """Distinct employees by CMS job code over time (EIN row-level Employee Detail)."""
    global ein_employee_detail_df, global_df
    try:
        if not _ein_mode_enabled():
            return jsonify(
                {
                    "ok": False,
                    "available": False,
                    "message": "Employee Detail mode is disabled for this dashboard.",
                }
            )
        _ensure_ein_employee_detail_loaded()
        if ein_employee_detail_df is None or ein_employee_detail_df.empty:
            return jsonify(
                {
                    "ok": False,
                    "available": False,
                    "message": "Row-level Employee Detail file not loaded.",
                }
            )
        grain = (request.args.get("grain") or "quarter").strip().lower()
        if grain not in ("day", "month", "quarter", "year"):
            grain = "quarter"
        slice_mode = (request.args.get("slice") or "nurse").strip().lower()
        if slice_mode not in ("nurse", "nonnurse", "all"):
            slice_mode = "nurse"
        start_date = (request.args.get("start_date") or "").strip()[:10] or None
        end_date = (request.args.get("end_date") or "").strip()[:10] or None
        context_raw = (request.args.get("context") or "1").strip().lower()
        context = context_raw not in ("0", "false", "no", "off")
        forensics_raw = (request.args.get("forensics") or "0").strip().lower()
        include_forensics = forensics_raw in ("1", "true", "yes", "on")
        cache_key = "|".join(
            [
                grain,
                slice_mode,
                start_date or "",
                end_date or "",
                "1" if context else "0",
                "1" if include_forensics else "0",
            ]
        )
        cached = _ein_headcount_api_cache_get(cache_key)
        if cached is not None:
            ccn = _ein_active_ccn()
            return jsonify({"ok": True, "available": True, "provnum": ccn, **cached})
        quarter_pbj_context = (
            ein_quarter_pbj_context_from_daily(global_df)
            if global_df is not None and not global_df.empty
            else {}
        )
        payload = ein_headcount_by_job_longitudinal_series(
            ein_employee_detail_df,
            grain=grain,
            slice_mode=slice_mode,
            start_date=start_date,
            end_date=end_date,
            context=context,
            quarter_pbj_context=quarter_pbj_context,
            facility_ccn=_ein_active_ccn(),
            include_forensics=include_forensics,
        )
        _ein_headcount_api_cache_set(cache_key, payload)
        ccn = _ein_active_ccn()
        return jsonify(
            {
                "ok": True,
                "available": True,
                "provnum": ccn,
                **payload,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "message": str(exc)})


@app.route("/api/pre-post-licensee-analysis")
def pre_post_licensee_analysis():
    """Pre vs post windows: daily PBJ metrics + quarter-level case-mix (Welch t-test when scipy is available)."""
    global global_df, provider_info_df
    try:
        import numpy as np

        try:
            from scipy import stats as scipy_stats
        except ImportError:
            return jsonify({"error": "SciPy is required for pre/post statistical analysis."}), 500

        before_start = (request.args.get("before_start") or "").strip()
        before_end = (request.args.get("before_end") or "").strip()
        after_start = (request.args.get("after_start") or "").strip()
        after_end = (request.args.get("after_end") or "").strip()
        if not all([before_start, before_end, after_start, after_end]):
            return jsonify({"error": "All four dates are required (pre start/end, post start/end)."}), 400

        if global_df is None or len(global_df) == 0:
            return jsonify({"error": "No data loaded"}), 400
        if "WorkDate" not in global_df.columns:
            return jsonify({"error": "WorkDate column missing"}), 400

        def _norm_col(s: str) -> str:
            return str(s).strip().lower().replace(" ", "").replace("-", "_")

        df = global_df
        workdate = pd.to_datetime(df["WorkDate"], errors="coerce")
        if workdate.isna().all():
            return jsonify({"error": "Could not parse WorkDate values"}), 400

        before_start_dt = pd.to_datetime(before_start, errors="coerce")
        before_end_dt = pd.to_datetime(before_end, errors="coerce")
        after_start_dt = pd.to_datetime(after_start, errors="coerce")
        after_end_dt = pd.to_datetime(after_end, errors="coerce")
        if any(pd.isna(x) for x in [before_start_dt, before_end_dt, after_start_dt, after_end_dt]):
            return jsonify({"error": "Invalid date range(s)"}), 400

        mask_pre = (workdate >= before_start_dt) & (workdate <= before_end_dt)
        mask_post = (workdate >= after_start_dt) & (workdate <= after_end_dt)
        df_pre = df[mask_pre]
        df_post = df[mask_post]

        if len(df_pre) == 0 or len(df_post) == 0:
            return (
                jsonify(
                    {
                        "error": "One of the windows has no rows. Adjust date ranges.",
                        "counts": {"pre_n_days": int(len(df_pre)), "post_n_days": int(len(df_post))},
                    }
                ),
                400,
            )

        col_map = {_norm_col(c): c for c in df.columns}

        def _find_col(candidates):
            for cand in candidates:
                key = _norm_col(cand)
                if key in col_map:
                    return col_map[key]
            return None

        def _to_numeric_series(series) -> np.ndarray:
            s = pd.to_numeric(series, errors="coerce")
            s = s.astype(float)
            s = s[np.isfinite(s)].dropna()
            return s.to_numpy()

        def _summary_and_pvalues(pre_arr: np.ndarray, post_arr: np.ndarray) -> dict:
            pre_n = int(len(pre_arr))
            post_n = int(len(post_arr))
            if pre_n == 0 or post_n == 0:
                return {
                    "pre_n": pre_n,
                    "post_n": post_n,
                    "pre_mean": None,
                    "post_mean": None,
                    "diff_mean": None,
                    "p_value": None,
                    "pre_sd": None,
                    "post_sd": None,
                    "effect_size_g": None,
                    "effect_size_label": None,
                }
            pre_mean = float(np.mean(pre_arr))
            post_mean = float(np.mean(post_arr))
            diff = post_mean - pre_mean
            pre_sd = float(np.std(pre_arr, ddof=1)) if pre_n >= 2 else None
            post_sd = float(np.std(post_arr, ddof=1)) if post_n >= 2 else None

            p_value = None
            t_stat = None
            df_welch = None
            ci_low = None
            ci_high = None
            se_diff = None
            if pre_n >= 2 and post_n >= 2 and pre_sd is not None and post_sd is not None:
                res = scipy_stats.ttest_ind(pre_arr, post_arr, equal_var=False, nan_policy="omit")
                if res and hasattr(res, "pvalue"):
                    try:
                        p_value = float(res.pvalue)
                    except Exception:
                        p_value = None
                if res and hasattr(res, "statistic"):
                    try:
                        t_stat = float(res.statistic)
                    except Exception:
                        t_stat = None
                var_pre = (pre_sd ** 2)
                var_post = (post_sd ** 2)
                se_term_pre = var_pre / pre_n
                se_term_post = var_post / post_n
                se_diff = float(np.sqrt(se_term_pre + se_term_post))
                den = ((se_term_pre ** 2) / (pre_n - 1)) + ((se_term_post ** 2) / (post_n - 1))
                num = (se_term_pre + se_term_post) ** 2
                if den > 0:
                    df_welch = float(num / den)
                if se_diff and se_diff > 0 and df_welch and df_welch > 0:
                    t_crit = float(scipy_stats.t.ppf(0.975, df_welch))
                    ci_low = float(diff - t_crit * se_diff)
                    ci_high = float(diff + t_crit * se_diff)

            effect_size_g = None
            effect_size_label = None
            if pre_n >= 2 and post_n >= 2 and pre_sd is not None and post_sd is not None:
                pooled_num = ((pre_n - 1) * (pre_sd ** 2)) + ((post_n - 1) * (post_sd ** 2))
                pooled_den = (pre_n + post_n - 2)
                if pooled_den > 0:
                    pooled_sd = float(np.sqrt(pooled_num / pooled_den))
                    if pooled_sd > 0:
                        d = (post_mean - pre_mean) / pooled_sd
                        # Small-sample corrected standardized mean difference (Hedges' g)
                        j = 1.0 - (3.0 / (4.0 * (pre_n + post_n) - 9.0)) if (pre_n + post_n) > 2 else 1.0
                        effect_size_g = float(d * j)
                        abs_g = abs(effect_size_g)
                        if abs_g < 0.2:
                            effect_size_label = "negligible"
                        elif abs_g < 0.5:
                            effect_size_label = "small"
                        elif abs_g < 0.8:
                            effect_size_label = "moderate"
                        else:
                            effect_size_label = "large"

            return {
                "pre_n": pre_n,
                "post_n": post_n,
                "pre_mean": round(pre_mean, 4),
                "post_mean": round(post_mean, 4),
                "diff_mean": round(diff, 4),
                "p_value": p_value,
                "pre_sd": (round(pre_sd, 4) if pre_sd is not None else None),
                "post_sd": (round(post_sd, 4) if post_sd is not None else None),
                "effect_size_g": (round(effect_size_g, 4) if effect_size_g is not None else None),
                "effect_size_label": effect_size_label,
                "t_stat": (round(t_stat, 4) if t_stat is not None else None),
                "df_welch": (round(df_welch, 3) if df_welch is not None else None),
                "se_diff": (round(se_diff, 5) if se_diff is not None else None),
                "ci95_low": (round(ci_low, 4) if ci_low is not None else None),
                "ci95_high": (round(ci_high, 4) if ci_high is not None else None),
                "q_value": None,
            }

        direct_care_col = _find_col(["Direct_Care_HPRD", "Nurse_Staff_HPRD_Excl_Admin"])
        direct_rn_col = _find_col(["RN_HPRD"])
        lpn_col = _find_col(["LPN_HPRD", "Total_LPN_HPRD"])
        nurse_aide_col = _find_col(["Total_Nurse_Aide_HPRD", "CNA_HPRD"])
        contract_pct_col = _find_col(["Total_Contract_Pct", "Contract_Pct", "Contract %"])
        census_col = _find_col(["MDScensus", "census"])
        certified_beds_col = _find_col(["certified_beds", "Certified Beds", "certified beds"])

        metrics: dict = {}

        if direct_care_col:
            pre_arr = _to_numeric_series(df_pre[direct_care_col])
            post_arr = _to_numeric_series(df_post[direct_care_col])
            metrics["direct_care_hprd"] = _summary_and_pvalues(pre_arr, post_arr)
        else:
            metrics["direct_care_hprd"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }

        if direct_rn_col:
            pre_arr = _to_numeric_series(df_pre[direct_rn_col])
            post_arr = _to_numeric_series(df_post[direct_rn_col])
            metrics["direct_rn_hprd"] = _summary_and_pvalues(pre_arr, post_arr)
        else:
            metrics["direct_rn_hprd"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }

        if lpn_col:
            pre_arr = _to_numeric_series(df_pre[lpn_col])
            post_arr = _to_numeric_series(df_post[lpn_col])
            metrics["lpn_hprd"] = _summary_and_pvalues(pre_arr, post_arr)
        else:
            metrics["lpn_hprd"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }

        if nurse_aide_col:
            pre_arr = _to_numeric_series(df_pre[nurse_aide_col])
            post_arr = _to_numeric_series(df_post[nurse_aide_col])
            metrics["nurse_aide_hprd"] = _summary_and_pvalues(pre_arr, post_arr)
        else:
            metrics["nurse_aide_hprd"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }

        if contract_pct_col:
            pre_arr = _to_numeric_series(df_pre[contract_pct_col])
            post_arr = _to_numeric_series(df_post[contract_pct_col])
            metrics["contract_pct"] = _summary_and_pvalues(pre_arr, post_arr)
        else:
            metrics["contract_pct"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }

        if census_col:
            pre_arr = _to_numeric_series(df_pre[census_col])
            post_arr = _to_numeric_series(df_post[census_col])
            metrics["census"] = _summary_and_pvalues(pre_arr, post_arr)
        else:
            metrics["census"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }

        facility_ccn = None
        for col in ["PROVNUM", "provnum", "ccn"]:
            if col in df.columns:
                facility_ccn = str(df[col].iloc[0]).strip().zfill(6)
                break

        if not certified_beds_col and provider_info_df is not None and len(provider_info_df) > 0:
            if facility_ccn and "ccn" in provider_info_df.columns:
                prov_sub = provider_info_df[
                    provider_info_df["ccn"].astype(str).str.strip().str.zfill(6) == facility_ccn
                ]
                if len(prov_sub) > 0:
                    pmap = {_norm_col(c): c for c in prov_sub.columns}
                    beds_col = None
                    for cand in ["certified_beds", "number_of_certified_beds", "num_certified_beds"]:
                        if _norm_col(cand) in pmap:
                            beds_col = pmap[_norm_col(cand)]
                            break
                    if beds_col:
                        prov_sub = prov_sub.copy()
                        prov_sub[beds_col] = pd.to_numeric(prov_sub[beds_col], errors="coerce")
                        prov_sub = prov_sub[prov_sub[beds_col] > 0]
                        if len(prov_sub) > 0:
                            latest_beds = float(
                                prov_sub.sort_values("processing_date", ascending=False)[beds_col].iloc[0]
                            )
                            if latest_beds > 0 and census_col:
                                df_pre_occ = df_pre[[census_col]].copy()
                                df_pre_occ["__occ"] = (
                                    pd.to_numeric(df_pre_occ[census_col], errors="coerce") / latest_beds * 100
                                )
                                df_pre_occ = df_pre_occ.replace([np.inf, -np.inf], np.nan).dropna(subset=["__occ"])

                                df_post_occ = df_post[[census_col]].copy()
                                df_post_occ["__occ"] = (
                                    pd.to_numeric(df_post_occ[census_col], errors="coerce") / latest_beds * 100
                                )
                                df_post_occ = df_post_occ.replace([np.inf, -np.inf], np.nan).dropna(subset=["__occ"])

                                pre_arr = _to_numeric_series(df_pre_occ["__occ"])
                                post_arr = _to_numeric_series(df_post_occ["__occ"])
                                metrics["occupancy_pct"] = _summary_and_pvalues(pre_arr, post_arr)

        if "occupancy_pct" not in metrics and certified_beds_col and census_col:
            df_pre_occ = df_pre[[census_col, certified_beds_col]].copy()
            df_pre_occ["__occ"] = (
                pd.to_numeric(df_pre_occ[census_col], errors="coerce")
                / pd.to_numeric(df_pre_occ[certified_beds_col], errors="coerce")
                * 100
            )
            df_pre_occ = df_pre_occ.replace([np.inf, -np.inf], np.nan).dropna(subset=["__occ"])

            df_post_occ = df_post[[census_col, certified_beds_col]].copy()
            df_post_occ["__occ"] = (
                pd.to_numeric(df_post_occ[census_col], errors="coerce")
                / pd.to_numeric(df_post_occ[certified_beds_col], errors="coerce")
                * 100
            )
            df_post_occ = df_post_occ.replace([np.inf, -np.inf], np.nan).dropna(subset=["__occ"])

            pre_arr = _to_numeric_series(df_pre_occ["__occ"])
            post_arr = _to_numeric_series(df_post_occ["__occ"])
            metrics["occupancy_pct"] = _summary_and_pvalues(pre_arr, post_arr)
        elif "occupancy_pct" not in metrics:
            metrics["occupancy_pct"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }

        pre_n_quarters = 0
        post_n_quarters = 0
        if "CY_Qtr" in df.columns:
            pre_quarters = set(df_pre["CY_Qtr"].dropna().astype(str).unique().tolist())
            post_quarters = set(df_post["CY_Qtr"].dropna().astype(str).unique().tolist())
            pre_n_quarters = int(len(pre_quarters))
            post_n_quarters = int(len(post_quarters))

            def _normalize_any_quarter_label(q: object) -> Optional[str]:
                nq = normalize_cy_qtr_ein(q)
                if nq:
                    return nq
                pq = _provider_quarter_to_canonical(q)
                if pq:
                    return pq
                return None

            def _quarter_metric_arrays(metric_key: str):
                try:
                    cm_resp = get_case_mix_data()
                    cm_json = cm_resp[0].get_json() if isinstance(cm_resp, tuple) else cm_resp.get_json()
                    cm_data = (cm_json or {}).get("case_mix_data") or {}
                except Exception:
                    cm_data = {}

                pre_qn = set()
                for q in pre_quarters:
                    nq = _normalize_any_quarter_label(q)
                    if nq:
                        pre_qn.add(nq)
                post_qn = set()
                for q in post_quarters:
                    nq = _normalize_any_quarter_label(q)
                    if nq:
                        post_qn.add(nq)

                pre_vals = []
                post_vals = []
                for q, row in cm_data.items():
                    if not isinstance(row, dict):
                        continue
                    val = row.get(metric_key)
                    if val is None:
                        continue
                    try:
                        f = float(val)
                    except (TypeError, ValueError):
                        continue
                    qn = _normalize_any_quarter_label(q)
                    if not qn:
                        continue
                    if qn in pre_qn:
                        pre_vals.append(f)
                    if qn in post_qn:
                        post_vals.append(f)
                return np.array(pre_vals, dtype=float), np.array(post_vals, dtype=float)

            pre_arr, post_arr = _quarter_metric_arrays("case_mix_total")
            metrics["case_mix_total_reported"] = _summary_and_pvalues(pre_arr, post_arr)

            pre_arr, post_arr = _quarter_metric_arrays("cmi")
            metrics["case_mix_index"] = _summary_and_pvalues(pre_arr, post_arr)

            def _provider_rating_by_quarter(col_name: str):
                if provider_info_df is None or len(provider_info_df) == 0:
                    return np.array([], dtype=float), np.array([], dtype=float)
                pinfo = provider_info_df
                if facility_ccn and "ccn" in pinfo.columns:
                    pinfo = pinfo[pinfo["ccn"].astype(str).str.strip().str.zfill(6) == facility_ccn]
                if pinfo is None or len(pinfo) == 0:
                    return np.array([], dtype=float), np.array([], dtype=float)
                pmap = {_norm_col(c): c for c in pinfo.columns}
                qcol = pmap.get("quarter")
                rcol = pmap.get(_norm_col(col_name))
                if not qcol or not rcol:
                    return np.array([], dtype=float), np.array([], dtype=float)
                pre_qn = set()
                for q in pre_quarters:
                    nq = normalize_cy_qtr_ein(q)
                    if nq:
                        pre_qn.add(nq)
                post_qn = set()
                for q in post_quarters:
                    nq = normalize_cy_qtr_ein(q)
                    if nq:
                        post_qn.add(nq)
                pre_v: list[float] = []
                post_v: list[float] = []
                for _, row in pinfo.iterrows():
                    q = str(row[qcol]).strip() if pd.notna(row.get(qcol)) else ""
                    if not q or q.upper() in ("N/A", "NAN", "NONE"):
                        continue
                    qn = _normalize_any_quarter_label(q)
                    if not qn:
                        continue
                    try:
                        v = float(row[rcol])
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(v) or v <= 0 or v > 5.5:
                        continue
                    if qn in pre_qn:
                        pre_v.append(v)
                    if qn in post_qn:
                        post_v.append(v)
                return np.array(pre_v, dtype=float), np.array(post_v, dtype=float)

            pre_r, post_r = _provider_rating_by_quarter("overall_rating")
            metrics["overall_rating"] = _summary_and_pvalues(pre_r, post_r)
            pre_r2, post_r2 = _provider_rating_by_quarter("staffing_rating")
            metrics["staffing_rating"] = _summary_and_pvalues(pre_r2, post_r2)
        else:
            metrics["case_mix_total_reported"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }
            metrics["case_mix_index"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }
            metrics["overall_rating"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }
            metrics["staffing_rating"] = {
                "pre_mean": None,
                "post_mean": None,
                "diff_mean": None,
                "p_value": None,
            }

        # Benjamini-Hochberg FDR correction across all metrics with valid p-values.
        p_items: list[tuple[str, float]] = []
        for mk, mv in metrics.items():
            if not isinstance(mv, dict):
                continue
            pv = mv.get("p_value")
            if pv is None:
                continue
            try:
                pvn = float(pv)
            except (TypeError, ValueError):
                continue
            if np.isfinite(pvn):
                p_items.append((mk, pvn))
        if p_items:
            p_items_sorted = sorted(p_items, key=lambda x: x[1])
            m_total = len(p_items_sorted)
            bh_raw: list[float] = [0.0] * m_total
            for i, (_, pv) in enumerate(p_items_sorted, start=1):
                bh_raw[i - 1] = min(1.0, (pv * m_total) / i)
            bh_adj: list[float] = [0.0] * m_total
            running = 1.0
            for i in range(m_total - 1, -1, -1):
                running = min(running, bh_raw[i])
                bh_adj[i] = running
            for (mk, _), qv in zip(p_items_sorted, bh_adj):
                if isinstance(metrics.get(mk), dict):
                    metrics[mk]["q_value"] = round(float(qv), 6)

        sample_flags: list[str] = []
        pre_n_days = int(len(df_pre))
        post_n_days = int(len(df_post))
        if pre_n_days < 180 or post_n_days < 180:
            sample_flags.append("short_window")
        day_ratio = (max(pre_n_days, post_n_days) / max(1, min(pre_n_days, post_n_days)))
        if day_ratio >= 1.5:
            sample_flags.append("uneven_windows")
        if pre_n_quarters > 0 and post_n_quarters > 0:
            q_ratio = max(pre_n_quarters, post_n_quarters) / max(1, min(pre_n_quarters, post_n_quarters))
            if q_ratio >= 1.5:
                sample_flags.append("uneven_quarters")

        def _json_finite_safe(obj):
            if isinstance(obj, dict):
                return {k: _json_finite_safe(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_json_finite_safe(v) for v in obj]
            if isinstance(obj, tuple):
                return [_json_finite_safe(v) for v in obj]
            if isinstance(obj, np.floating):
                return float(obj) if np.isfinite(obj) else None
            if isinstance(obj, float):
                return obj if np.isfinite(obj) else None
            return obj

        payload = {
                "params": {
                    "before_start": before_start_dt.strftime("%Y-%m-%d"),
                    "before_end": before_end_dt.strftime("%Y-%m-%d"),
                    "after_start": after_start_dt.strftime("%Y-%m-%d"),
                    "after_end": after_end_dt.strftime("%Y-%m-%d"),
                },
                "counts": {
                    "pre_n_days": pre_n_days,
                    "post_n_days": post_n_days,
                    "pre_n_quarters": int(pre_n_quarters),
                    "post_n_quarters": int(post_n_quarters),
                },
                "sample_quality": {
                    "flags": sample_flags,
                    "day_ratio": round(day_ratio, 3),
                },
                "metrics": metrics,
            }
        return jsonify(_json_finite_safe(payload))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Dynamic dashboard - no initialization needed

def run_dashboard(
    provnum: str,
    port: int = 5000,
    *,
    ein_mode: str = "all",
    ein_selected_quarters: Optional[Sequence[str]] = None,
) -> None:
    """Run the dynamic dashboard for a specific facility.

    For local/internal use, defaults to ``ein_mode='all'`` so Employee Detail
    loads every quarter present in the facility EIN files (not filtered).
    Set ``ein_mode='none'`` to disable EIN sections, or ``'selected'`` with
    ``ein_selected_quarters`` for a subset.
    """
    global PROVNUM, _data_initialized, EIN_DASHBOARD_MODE, EIN_SELECTED_QUARTERS
    PROVNUM = provnum
    raw = (ein_mode or "all").strip().lower() or "all"
    if raw not in {"all", "selected", "none"}:
        raw = "all"
    EIN_DASHBOARD_MODE = raw
    if ein_selected_quarters is None:
        EIN_SELECTED_QUARTERS = []
    else:
        EIN_SELECTED_QUARTERS = [str(q).strip() for q in ein_selected_quarters if str(q).strip()]

    print(f"Employee Detail (EIN) mode: {EIN_DASHBOARD_MODE}")
    if EIN_DASHBOARD_MODE == "selected" and EIN_SELECTED_QUARTERS:
        print(f"  Selected quarters: {', '.join(EIN_SELECTED_QUARTERS)}")

    print(f"Loading facility {provnum} data (CSV read can take a minute)...", flush=True)
    app_instance = create_dynamic_dashboard(provnum)
    if app_instance is None:
        print(f"ERROR: Failed to create dashboard for facility {provnum}")
        return

    _data_initialized = True

    print(f"Flask listening on http://127.0.0.1:{port}/  (all interfaces: http://0.0.0.0:{port}/)", flush=True)
    _use_reload = _pbj_env_str("FLASK_USE_RELOADER").lower() in ("1", "true", "yes")
    app_instance.run(
        debug=True, host="0.0.0.0", port=port, threaded=True, use_reloader=_use_reload
    )

if __name__ == "__main__":
    # For local testing: use the same controlled launcher path.
    # This avoids Flask's default broad debug-reloader file watching.
    _ccn = (PROVNUM or "").strip() or _default_provnum_for_bundle()
    if not _ccn:
        print(
            "ERROR: Cannot infer facility CCN. Set PBJ_FACILITY_CCN, PBJ_PROVNUM, or PROVNUM "
            "to a 6-digit CCN, or run this file from ``deployments/pbj320-<CCN>/`` (see run_dashboard).",
            file=sys.stderr,
        )
        raise SystemExit(2)
    _tpl = _superdynamic_dashboard_template_name()
    _port_raw = (os.environ.get("PORT") or "").strip()
    if _port_raw:
        _port = int(_port_raw)
    else:
        # v2 defaults to 5001 so a legacy server can keep 5000 during side-by-side QA.
        _port = 5001 if _tpl == "superdynamic_dashboard_v2.html" else 5000
    print(f"PBJ_SUPERDYNAMIC_TEMPLATE -> {_tpl}", flush=True)
    print(f"Listening on http://127.0.0.1:{_port}/", flush=True)
    run_dashboard(_ccn, port=_port)