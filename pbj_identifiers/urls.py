"""
URL generation functions for PBJ dashboards and external CMS links.

All URLs are generated deterministically from normalized identifiers.
No network calls, no scraping - pure functions only.
"""

import json
import re
from typing import Any, Optional
from urllib.parse import quote, urlencode

from .cms_datagov_legacy import (
    cms_datagov_canonical_quarter_key,
    cms_datagov_legacy_lowercase_prov_workdate_keys,
    cms_datagov_nurse_daily_uses_rolling_data_endpoint,
)
from .validators import normalize_ccn, normalize_state_code, normalize_entity_id, validate_ccn


# Base URLs (can be configured if needed)
# New public site structure uses path-based routes, e.g.:
# - Provider: https://www.pbj320.com/provider/056430
# - State:    https://www.pbj320.com/state/tn
# - Entity:   https://www.pbj320.com/entity/217
PBJ_DASHBOARD_BASE = "https://www.pbj320.com/"
# Interactive state + CMS region tables (filter by CMS Region in-app)
PBJ_RANKINGS_REPORT_URL = "https://www.pbj320.com/report"
CMS_CARE_COMPARE_BASE = "https://www.medicare.gov/care-compare/details/nursing-home/"
# Nursing home inspections (health) on Medicare Care Compare — deterministic templates only.
MEDICARE_NH_INSPECTIONS_BASE = "https://www.medicare.gov/care-compare/inspections/nursing-home"
MEDICARE_NH_INSPECTION_PDF_BASE = "https://www.medicare.gov/care-compare/inspections/pdf/nursing-home"


def generate_dashboard_url(
    facility: Optional[str] = None,
    state: Optional[str] = None,
    entity: Optional[str] = None
) -> str:
    """
    Generate dashboard URL with normalized identifiers.
    
    Args:
        facility: Facility CCN (will be normalized)
        state: State code (will be normalized)
        entity: Entity ID (will be normalized)
        
    Returns:
        Complete dashboard URL.
        - If only facility is provided: provider page (/provider/{ccn})
        - If only state is provided: state page (/state/{state})
        - If only entity is provided: entity page (/entity/{id})
        - If multiple are provided, falls back to query-string form on the base URL.
        
    Examples:
        >>> generate_dashboard_url(facility="335513")
        'https://www.pbj320.com/provider/335513'
        >>> generate_dashboard_url(state="NY")
        'https://www.pbj320.com/state/ny'
        >>> generate_dashboard_url(facility="335513", state="NY", entity="217")
        'https://www.pbj320.com/?facility=335513&state=NY&entity=217'
    """
    facility_norm = normalize_ccn(facility) if facility else None
    state_norm = normalize_state_code(state) if state else None
    entity_norm = normalize_entity_id(entity) if entity else None

    # Single-target URLs use the new path-based structure
    targets = [t for t in (facility_norm, state_norm, entity_norm) if t]
    if len(targets) == 1:
        if facility_norm:
            return f"{PBJ_DASHBOARD_BASE}provider/{facility_norm}"
        if state_norm:
            # Public state pages use lowercase state codes in the path (e.g. /state/tn)
            return f"{PBJ_DASHBOARD_BASE}state/{state_norm.lower()}"
        if entity_norm:
            return f"{PBJ_DASHBOARD_BASE}entity/{entity_norm}"

    # Fallback for unusual combinations: preserve legacy query-string semantics
    params = []
    if facility_norm:
        params.append(f"facility={facility_norm}")
    if state_norm:
        params.append(f"state={state_norm}")
    if entity_norm:
        params.append(f"entity={entity_norm}")

    if params:
        return PBJ_DASHBOARD_BASE + "?" + "&".join(params)

    return PBJ_DASHBOARD_BASE


def generate_cms_url(ccn: str, state: str) -> str:
    """
    Generate CMS Care Compare URL for a facility.
    
    Args:
        ccn: Facility CCN (will be normalized)
        state: State code (will be normalized)
        
    Returns:
        Complete CMS Care Compare URL
        
    Examples:
        >>> generate_cms_url("335513", "NY")
        'https://www.medicare.gov/care-compare/details/nursing-home/335513/view-all/?state=NY'
    """
    ccn_norm = normalize_ccn(ccn)
    state_norm = normalize_state_code(state)
    
    return f"{CMS_CARE_COMPARE_BASE}{ccn_norm}/view-all/?state={state_norm}"


def generate_facility_dashboard_url(ccn: str) -> str:
    """
    Generate facility-specific dashboard URL.
    
    Args:
        ccn: Facility CCN (will be normalized)
        
    Returns:
        Premium facility path on the public site (6-digit CCN, zero-padded), e.g.
        ``https://www.pbj320.com/premium/075182/``.
    """
    ccn_norm = normalize_ccn(ccn)
    return f"{PBJ_DASHBOARD_BASE}premium/{ccn_norm}/"


def generate_state_dashboard_url(state: str) -> str:
    """
    Generate state-specific dashboard URL.
    
    Args:
        state: State code (will be normalized)
        
    Returns:
        State dashboard URL
    """
    return generate_dashboard_url(state=state)


def medicare_nursing_home_inspections_landing_url(ccn: str) -> Optional[str]:
    """
    Care Compare inspections landing for a nursing home (health tab path).

    Returns None if CCN is not a valid 6-character facility identifier for URLs.
    """
    if not ccn or not validate_ccn(str(ccn).strip()):
        return None
    ccn_norm = normalize_ccn(str(ccn).strip())
    return f"{MEDICARE_NH_INSPECTIONS_BASE}/{ccn_norm}/health"


def medicare_nursing_home_health_inspection_pdf_url(ccn: str, survey_date_iso: str) -> Optional[str]:
    """
    PDF URL for a routine health inspection, e.g. .../health-inspection/?date=YYYY-MM-DD.

    survey_date_iso must already be a valid calendar date string (YYYY-MM-DD); otherwise returns None
    (callers must not guess dates).
    """
    if not survey_date_iso or not validate_ccn(str(ccn).strip()):
        return None
    d = str(survey_date_iso).strip()
    if len(d) != 10 or d[4] != "-" or d[7] != "-":
        return None
    y, m, day = d[:4], d[5:7], d[8:10]
    if not (y.isdigit() and m.isdigit() and day.isdigit()):
        return None
    ccn_norm = normalize_ccn(str(ccn).strip())
    q = urlencode({"date": d})
    return (
        f"{MEDICARE_NH_INSPECTION_PDF_BASE}/{ccn_norm}/health/health-inspection/?{q}"
    )


def medicare_nursing_home_complaint_inspection_pdf_url(ccn: str, survey_date_iso: str) -> Optional[str]:
    """
    PDF URL for a complaint inspection, e.g. .../complaint-inspection?date=YYYY-MM-DD.
    Same date validation as health inspection PDF.
    """
    if not survey_date_iso or not validate_ccn(str(ccn).strip()):
        return None
    d = str(survey_date_iso).strip()
    if len(d) != 10 or d[4] != "-" or d[7] != "-":
        return None
    y, m, day = d[:4], d[5:7], d[8:10]
    if not (y.isdigit() and m.isdigit() and day.isdigit()):
        return None
    ccn_norm = normalize_ccn(str(ccn).strip())
    q = urlencode({"date": d})
    return f"{MEDICARE_NH_INSPECTION_PDF_BASE}/{ccn_norm}/health/complaint-inspection?{q}"


def medicare_nursing_home_infection_control_inspection_pdf_url(ccn: str, survey_date_iso: str) -> Optional[str]:
    """
    PDF URL for an infection-control inspection, e.g. .../infection-control-inspection/?date=YYYY-MM-DD.
    """
    if not survey_date_iso or not validate_ccn(str(ccn).strip()):
        return None
    d = str(survey_date_iso).strip()
    if len(d) != 10 or d[4] != "-" or d[7] != "-":
        return None
    y, m, day = d[:4], d[5:7], d[8:10]
    if not (y.isdigit() and m.isdigit() and day.isdigit()):
        return None
    ccn_norm = normalize_ccn(str(ccn).strip())
    q = urlencode({"date": d})
    return (
        f"{MEDICARE_NH_INSPECTION_PDF_BASE}/{ccn_norm}/health/infection-control-inspection/?{q}"
    )


def medicare_nursing_home_inspection_pdf_url(
    ccn: str,
    survey_date_iso: str,
    inspection_kind: str,
) -> Optional[str]:
    """
    Care Compare inspection PDF for one survey date.

    ``inspection_kind``: ``complaint`` | ``health`` | ``infection_control``.
    """
    kind = str(inspection_kind or "").strip().lower()
    if kind == "complaint":
        return medicare_nursing_home_complaint_inspection_pdf_url(ccn, survey_date_iso)
    if kind == "infection_control":
        return medicare_nursing_home_infection_control_inspection_pdf_url(ccn, survey_date_iso)
    if kind == "health":
        return medicare_nursing_home_health_inspection_pdf_url(ccn, survey_date_iso)
    return None


def generate_entity_dashboard_url(entity_id: str) -> str:
    """
    Generate entity-specific dashboard URL.
    
    Args:
        entity_id: Entity ID (will be normalized)
        
    Returns:
        Entity dashboard URL, or base URL if entity_id is invalid
    """
    entity_norm = normalize_entity_id(entity_id)
    if entity_norm:
        return generate_dashboard_url(entity=entity_norm)
    return PBJ_DASHBOARD_BASE


# CMS open data (data.cms.gov) — explorer URLs for traceability (deterministic templates).
CMS_NONNURSE_PBJ_DATASET_BASE = (
    "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-non-nurse-staffing"
)
CMS_NURSE_DAILY_PBJ_DATASET_BASE = (
    "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing"
)
CMS_NH_HEALTH_CITATIONS_DATASET_ID = "r5ix-sfxw"


def _pbj_cms_daily_work_date_yyyymmdd(date: Any) -> Optional[str]:
    """Normalize a calendar day to ``YYYYMMDD`` for CMS explorer ``WorkDate`` / ``workdate`` filters."""
    if date is None:
        return None
    if isinstance(date, str):
        ds = str(date).strip()
        if len(ds) == 10 and "-" in ds:
            return ds.replace("-", "")
        if len(ds) == 8 and ds.isdigit():
            return ds
        return None
    if hasattr(date, "strftime"):
        return str(date.strftime("%Y%m%d"))
    return None


def cms_nonnurse_pbj_segment_path(cy_qtr: str) -> Optional[str]:
    """
    Explorer path segment for a PBJ quarter, e.g. CY2025Q1 -> ``q1-2025`` (matches data.cms.gov routes).
    """
    if not cy_qtr:
        return None
    s = str(cy_qtr).strip().upper().replace("CY", "")
    m = re.match(r"^(\d{4})Q([1-4])$", s)
    if not m:
        return None
    return f"q{int(m.group(2))}-{m.group(1)}"


def cms_pbj_daily_staffing_explorer_url(
    quarter: str,
    date: Any,
    provnum: str,
    *,
    data_type: str = "nurse",
) -> Optional[str]:
    """
    Build a data.cms.gov explorer URL for **daily** nurse or non-nurse PBJ (one CCN + one work date).

    ``quarter`` may be ``CY2019Q3``, ``2019Q3``, etc. Filter column names follow
    ``cms_datagov_legacy_lowercase_prov_workdate_keys`` (see ``docs/CMS_DATAGOV_PBJ_EXPLORER.md``).
    """
    ymd = _pbj_cms_daily_work_date_yyyymmdd(date)
    if not ymd:
        return None
    if not isinstance(quarter, str) or not str(quarter).strip():
        return None

    q_raw = str(quarter).strip()
    canon_q = cms_datagov_canonical_quarter_key(q_raw)
    if not canon_q:
        return None

    quarter_url = cms_nonnurse_pbj_segment_path(q_raw)
    if not quarter_url:
        return None

    legacy = cms_datagov_legacy_lowercase_prov_workdate_keys(q_raw)
    if legacy:
        provnum_col = "provnum"
        workdate_col = "workdate"
    else:
        provnum_col = "PROVNUM"
        workdate_col = "WorkDate"

    p = str(provnum or "").strip()
    if p.isdigit():
        p = p.zfill(6)

    query_params: dict[str, Any] = {
        "filters": {
            "list": [
                {
                    "conditions": [
                        {
                            "column": {"value": provnum_col},
                            "comparator": {"value": "="},
                            "filterValue": [p],
                        },
                        {
                            "column": {"value": workdate_col},
                            "comparator": {"value": "="},
                            "filterValue": [ymd],
                        },
                    ],
                }
            ],
            "rootConjunction": {"value": "AND"},
        },
        "keywords": "",
        "offset": 0,
        "limit": 10,
        "sort": {"sortBy": None, "sortOrder": None},
        "columns": [],
    }

    query_json = json.dumps(query_params)
    encoded_query = quote(query_json)

    if str(data_type).strip().lower() == "nonnurse":
        base_url = f"{CMS_NONNURSE_PBJ_DATASET_BASE}/data"
    else:
        base_url = f"{CMS_NURSE_DAILY_PBJ_DATASET_BASE}/data"

    y = int(canon_q[:4])
    qn = int(canon_q[-1])
    if str(data_type).strip().lower() != "nonnurse" and cms_datagov_nurse_daily_uses_rolling_data_endpoint(y, qn):
        return f"{base_url}?query={encoded_query}"
    return f"{base_url}/{quarter_url}?query={encoded_query}"


def _cms_datagov_provnum_filter_query_json(provnum: str, cy_qtr: Optional[str] = None) -> dict[str, Any]:
    p = str(provnum or "").strip()
    p = p.zfill(6) if p.isdigit() else p
    prov_col = "provnum" if cms_datagov_legacy_lowercase_prov_workdate_keys(str(cy_qtr or "")) else "PROVNUM"
    return {
        "filters": {
            "list": [
                {
                    "conditions": [
                        {
                            "column": {"value": prov_col},
                            "comparator": {"value": "="},
                            "filterValue": [p],
                        }
                    ],
                }
            ],
            "rootConjunction": {"value": "AND"},
        },
        "keywords": "",
        "offset": 0,
        "limit": 10,
        "sort": {"sortBy": None, "sortOrder": None},
        "columns": [],
    }


def cms_nonnurse_pbj_data_explorer_url(provnum: str, cy_qtr: Optional[str] = None) -> Optional[str]:
    """
    data.cms.gov Payroll-Based Journal Daily **Non-Nurse** Staffing explorer URL.
    When ``cy_qtr`` is set (canonical ``CYyyyyQn``), links into that quarter's data slice with a PROVNUM filter.

    Reference (example quarter + filter): `https://data.cms.gov/quality-of-care/payroll-based-journal-daily-non-nurse-staffing/data/q1-2025?query=...`
    """
    seg = cms_nonnurse_pbj_segment_path(cy_qtr) if cy_qtr else None
    base = f"{CMS_NONNURSE_PBJ_DATASET_BASE}/data/{seg}" if seg else f"{CMS_NONNURSE_PBJ_DATASET_BASE}/data"
    qjson = json.dumps(_cms_datagov_provnum_filter_query_json(provnum, cy_qtr), separators=(",", ":"))
    return f"{base}?query={quote(qjson, safe='')}"


def cms_nh_health_citations_dataset_explorer_url(ccn: str) -> Optional[str]:
    """
    CMS Provider Data Catalog — NH Health Citations dataset with CCN pre-filtered.

    Reference: `https://data.cms.gov/provider-data/dataset/r5ix-sfxw?conditions[0][property]=cms_certification_number_ccn&conditions[0][value]=335513&conditions[0][operator]=%3D`
    """
    if not ccn or not validate_ccn(str(ccn).strip()):
        return None
    ccn_norm = normalize_ccn(str(ccn).strip())
    params = {
        "conditions[0][property]": "cms_certification_number_ccn",
        "conditions[0][value]": ccn_norm,
        "conditions[0][operator]": "=",
    }
    return f"https://data.cms.gov/provider-data/dataset/{CMS_NH_HEALTH_CITATIONS_DATASET_ID}?{urlencode(params)}"
