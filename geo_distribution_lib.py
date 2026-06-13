"""
Facility-quarter geography distributions from PBJ-derived ``facility_quarterly_metrics.csv``.

Each facility contributes one value per metric per quarter. Distributions are not built from
published state/region/national rollups (those remain in the geo rollup table for quick comparison).
"""

from __future__ import annotations

import glob
import os
import re
from typing import Any, Callable, Optional, cast

import numpy as np
import pandas as pd

# Small-sample display rules (adjust here).
GEO_DIST_MIN_HISTOGRAM = 30
GEO_DIST_MIN_HISTOGRAM_COUNTY = 15
GEO_DIST_MIN_DOTPLOT = 10
GEO_DIST_MIN_PEER_TABLE = 5
# Share-below / rank language (aligns with summary ``pct_below_*`` when n >= this).
GEO_DIST_MIN_SHARE_BELOW = 5
# Minimum peer count for mean-rank percentile (dotplot / peer_table / histogram).
GEO_DIST_MIN_PERCENTILE = GEO_DIST_MIN_SHARE_BELOW

# Percentile: mean-rank — rank = (# strictly below) + (# equal + 1) / 2; pct = 100 * rank / n.
PERCENTILE_METHOD_NOTE = (
    "Percentile uses the mean-rank method on facility-quarter values in this geography "
    "(each facility counts once). Ties are split."
)

GEOGRAPHY_TYPES = frozenset({"national", "state", "region", "county", "city"})

# Metric registry: extend by adding entries; frontend uses the same keys.
GEO_DIST_METRICS: dict[str, dict[str, Any]] = {
    "total_nurse_hprd": {
        "label": "Total nurse HPRD",
        "column": "Total_Nurse_HPRD",
        "requires_positive_denominator": True,
        "threshold_hprd_type": "total",
        "peer_sort_default": "closest_total_nurse_hprd",
    },
    "nurse_care_hprd": {
        "label": "Direct nurse HPRD (PBJ nurse care)",
        "column": "Nurse_Care_HPRD",
        "requires_positive_denominator": True,
        "threshold_hprd_type": "direct_care",
        "peer_sort_default": "closest_nurse_care_hprd",
    },
    "rn_hprd": {
        "label": "RN total HPRD (incl. admin & DON)",
        "column": "RN_HPRD",
        "requires_positive_denominator": True,
        "threshold_hprd_type": None,
        "peer_sort_default": "closest_rn_hprd",
    },
    "rn_care_hprd": {
        "label": "RN direct HPRD",
        "column": "RN_Care_HPRD",
        "requires_positive_denominator": True,
        "threshold_hprd_type": None,
        "peer_sort_default": "closest_rn_care_hprd",
    },
    "lpn_hprd": {
        "label": "LPN total HPRD (incl. admin)",
        "column": "LPN_HPRD",
        "requires_positive_denominator": True,
        "threshold_hprd_type": None,
        "peer_sort_default": "closest_lpn_hprd",
        "optional_column": True,
    },
    "lpn_care_hprd": {
        "label": "LPN direct HPRD",
        "column": "LPN_Care_HPRD",
        "requires_positive_denominator": True,
        "threshold_hprd_type": None,
        "peer_sort_default": "closest_lpn_care_hprd",
        "optional_column": True,
    },
    "nurse_aide_hprd": {
        "label": "Nurse aide HPRD",
        "column": "Nurse_Assistant_HPRD",
        "requires_positive_denominator": True,
        "threshold_hprd_type": None,
        "peer_sort_default": "closest_nurse_aide_hprd",
    },
    "contract_pct": {
        "label": "Contract %",
        "column": "Contract_Percentage",
        "requires_positive_denominator": False,
        "threshold_hprd_type": None,
        "peer_sort_default": "highest_contract_pct",
    },
    "avg_census": {
        "label": "Avg census",
        "column": "avg_daily_census",
        "fallback_column": "MDScensus",
        "requires_positive_denominator": False,
        "threshold_hprd_type": None,
        "peer_sort_default": "closest_avg_census",
    },
}

_APP_ROOT = os.path.dirname(os.path.abspath(__file__))
_FACILITY_QUARTERLY_DF: Optional[pd.DataFrame] = None
_FACILITY_QUARTERLY_CACHE: Optional[tuple[str, float]] = None
_PROVNUM_CITY_LOOKUP: Optional[dict[str, dict[str, str]]] = None
_PROVNUM_CITY_CACHE: Optional[tuple[str, float]] = None
_CMS_REGION_BY_STATE: Optional[dict[str, int]] = None
_CMS_REGION_STATES: Optional[dict[int, list[str]]] = None


def _round3(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def _round2(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _resolve_metrics_csv(filename: str) -> Optional[str]:
    """Resolve facility-quarter metrics table (CSV or zstd parquet)."""
    roots = [_APP_ROOT, os.path.abspath(os.path.join(_APP_ROOT, ".."))]
    base = filename
    if base.endswith(".csv"):
        parquet_name = base[:-4] + ".parquet"
    else:
        parquet_name = base
    rels = (
        base,
        parquet_name,
        os.path.join("pbj_lite", base),
        os.path.join("pbj_lite", parquet_name),
        os.path.join("data", "geo", base),
        os.path.join("data", "geo", parquet_name),
    )
    for root in roots:
        for rel in rels:
            path = os.path.join(root, rel)
            if os.path.isfile(path):
                return path
    return None


def normalize_cy_qtr(q: object) -> Optional[str]:
    s = str(q or "").strip().upper().replace(" ", "")
    if not s:
        return None
    if s.startswith("CY"):
        s = s[2:]
    m = re.match(r"^(\d{4})Q([1-4])$", s)
    if m:
        return f"CY{m.group(1)}Q{m.group(2)}"
    m = re.match(r"^Q([1-4])(\d{4})$", s)
    if m:
        return f"CY{m.group(2)}Q{m.group(1)}"
    return None


def normalize_provnum(provnum: object) -> str:
    raw = str(provnum or "").strip()
    if raw.isdigit():
        return raw.zfill(6)
    return raw.upper()


def _normalize_county(name: object) -> str:
    s = re.sub(r"\s+", " ", str(name or "").strip()).lower()
    s = re.sub(r",?\s+[a-z]{2}$", "", s)
    if s.endswith(" county"):
        s = s[: -len(" county")].strip()
    return s


def _normalize_city(city: object, state: object) -> str:
    c = re.sub(r"\s+", " ", str(city or "").strip()).upper()
    st = str(state or "").strip().upper()
    if not c or not st:
        return ""
    return f"{c}|{st}"


def _load_facility_quarterly_df() -> Optional[pd.DataFrame]:
    global _FACILITY_QUARTERLY_DF, _FACILITY_QUARTERLY_CACHE
    path = _resolve_metrics_csv("facility_quarterly_metrics.csv")
    if not path:
        _FACILITY_QUARTERLY_DF = None
        _FACILITY_QUARTERLY_CACHE = None
        return None
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return None
    if (
        _FACILITY_QUARTERLY_DF is not None
        and _FACILITY_QUARTERLY_CACHE is not None
        and _FACILITY_QUARTERLY_CACHE[0] == path
        and _FACILITY_QUARTERLY_CACHE[1] == mtime
    ):
        return _FACILITY_QUARTERLY_DF
    try:
        if str(path).lower().endswith(".parquet"):
            raw = pd.read_parquet(path)
        else:
            raw = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    need = {"PROVNUM", "CY_Qtr", "STATE", "Total_Nurse_HPRD"}
    if raw.empty or not need.issubset(set(raw.columns)):
        return None
    df = cast(pd.DataFrame, raw.copy())
    df["PROVNUM"] = df["PROVNUM"].map(normalize_provnum)
    df["STATE"] = df["STATE"].astype(str).str.strip().str.upper()
    df["CY_Qtr"] = df["CY_Qtr"].astype(str).str.strip()
    if "COUNTY_NAME" in df.columns:
        df["_county_norm"] = df["COUNTY_NAME"].map(_normalize_county)
    else:
        df["_county_norm"] = ""
    _FACILITY_QUARTERLY_DF = df
    _FACILITY_QUARTERLY_CACHE = (path, mtime)
    return df


def supplement_facility_lite_peer_df(lite_df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Extend ``facility_lite_metrics`` peer rows with ``facility_quarterly_metrics`` for
    quarters missing from the lite extract (county rollups need every peer facility).
    """
    qdf = _load_facility_quarterly_df()
    if qdf is None or qdf.empty:
        return lite_df
    base = cast(pd.DataFrame, lite_df.copy()) if lite_df is not None and not lite_df.empty else pd.DataFrame()
    lite_quarters: set[str] = set()
    if not base.empty and "CY_Qtr" in base.columns:
        lite_quarters = {str(x).strip() for x in base["CY_Qtr"].dropna().unique()}
    q_quarters = {str(x).strip() for x in qdf["CY_Qtr"].dropna().unique()}
    missing_quarters = q_quarters - lite_quarters
    if not missing_quarters:
        return base if not base.empty else lite_df
    add = cast(pd.DataFrame, qdf[qdf["CY_Qtr"].astype(str).str.strip().isin(missing_quarters)].copy())
    if add.empty:
        return base if not base.empty else lite_df

    def _col(frame: pd.DataFrame, *names: str) -> pd.Series:
        for name in names:
            if name in frame.columns:
                return frame[name]
        return pd.Series([None] * len(frame), index=frame.index)

    mapped = pd.DataFrame(
        {
            "CY_Qtr": add["CY_Qtr"].astype(str).str.strip(),
            "PROVNUM": add["PROVNUM"].map(normalize_provnum),
            "STATE": add["STATE"].astype(str).str.strip().str.upper(),
            "COUNTY_NAME": _col(add, "COUNTY_NAME").astype(str).str.strip(),
            "Total_Nurse_HPRD": pd.to_numeric(_col(add, "Total_Nurse_HPRD"), errors="coerce"),
            "Nurse_Care_HPRD": pd.to_numeric(_col(add, "Nurse_Care_HPRD"), errors="coerce"),
            "Total_RN_HPRD": pd.to_numeric(_col(add, "Total_RN_HPRD", "RN_HPRD"), errors="coerce"),
            "Direct_Care_RN_HPRD": pd.to_numeric(
                _col(add, "Direct_Care_RN_HPRD", "RN_Care_HPRD"), errors="coerce"
            ),
            "Contract_Percentage": pd.to_numeric(_col(add, "Contract_Percentage"), errors="coerce"),
            "Census": pd.to_numeric(_col(add, "Census", "avg_daily_census", "MDScensus"), errors="coerce"),
        }
    )
    if "COUNTY_NAME" in mapped.columns:
        mapped["_county_norm"] = mapped["COUNTY_NAME"].map(_normalize_county)
    if base.empty:
        return mapped
    combined = pd.concat([base, mapped], ignore_index=True)
    return cast(pd.DataFrame, combined)


def _dedupe_facility_quarter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One row per PROVNUM per quarter (keep the row with the most resident-days)."""
    if df.empty or "PROVNUM" not in df.columns:
        return df
    work = df.copy()
    if "total_resident_days" in work.columns:
        work["_dedupe_key"] = pd.to_numeric(work["total_resident_days"], errors="coerce").fillna(0.0)
    elif "days_reported" in work.columns:
        work["_dedupe_key"] = pd.to_numeric(work["days_reported"], errors="coerce").fillna(0.0)
    else:
        return cast(pd.DataFrame, work.drop_duplicates(subset=["PROVNUM"], keep="last"))
    work = work.sort_values("_dedupe_key")
    out = cast(pd.DataFrame, work.drop_duplicates(subset=["PROVNUM"], keep="last"))
    return out.drop(columns=["_dedupe_key"], errors="ignore")


def _quarter_slice(df: pd.DataFrame, cy: str) -> pd.DataFrame:
    """All facility rows for canonical ``CYyyyyQn``, deduplicated."""
    if df.empty:
        return df
    norm = df["CY_Qtr"].map(lambda x: normalize_cy_qtr(x) or str(x).strip().upper())
    sub = cast(pd.DataFrame, df.loc[norm == cy.upper()].copy())
    return _dedupe_facility_quarter_rows(sub)


def _load_provnum_city_lookup() -> dict[str, dict[str, str]]:
    """CCN → {city, state} from latest bundled Provider Information CSV."""
    global _PROVNUM_CITY_LOOKUP, _PROVNUM_CITY_CACHE
    roots = [
        os.path.join(_APP_ROOT, "provider_info"),
        os.path.join(_APP_ROOT, "provider_info_extracted"),
    ]
    candidates: list[str] = []
    for root in roots:
        if os.path.isdir(root):
            candidates.extend(glob.glob(os.path.join(root, "NH_ProviderInfo_*.csv")))
    if not candidates:
        _PROVNUM_CITY_LOOKUP = {}
        return {}
    latest = max(candidates, key=os.path.getmtime)
    try:
        mtime = os.path.getmtime(latest)
    except OSError:
        return _PROVNUM_CITY_LOOKUP or {}
    if (
        _PROVNUM_CITY_LOOKUP is not None
        and _PROVNUM_CITY_CACHE is not None
        and _PROVNUM_CITY_CACHE[0] == latest
        and _PROVNUM_CITY_CACHE[1] == mtime
    ):
        return _PROVNUM_CITY_LOOKUP
    ccn_col = "CMS Certification Number (CCN)"
    usecols = [ccn_col, "City", "State"]
    try:
        pdf = pd.read_csv(latest, usecols=usecols, low_memory=False)
    except Exception:
        _PROVNUM_CITY_LOOKUP = {}
        return {}
    out: dict[str, dict[str, str]] = {}
    for _, row in pdf.iterrows():
        ccn = normalize_provnum(row.get(ccn_col))
        if not ccn:
            continue
        city = str(row.get("City") or "").strip()
        state = str(row.get("State") or "").strip().upper()
        if city and state:
            out[ccn] = {"city": city, "state": state}
    _PROVNUM_CITY_LOOKUP = out
    _PROVNUM_CITY_CACHE = (latest, mtime)
    return out


def _load_cms_region_maps() -> tuple[dict[str, int], dict[int, list[str]]]:
    global _CMS_REGION_BY_STATE, _CMS_REGION_STATES
    if _CMS_REGION_BY_STATE is not None and _CMS_REGION_STATES is not None:
        return _CMS_REGION_BY_STATE, _CMS_REGION_STATES
    by_state: dict[str, int] = {}
    by_region: dict[int, list[str]] = {}
    path = _resolve_metrics_csv("cms_region_state_mapping.csv")
    if path:
        try:
            mmap = pd.read_csv(path, low_memory=False)
            if "State_Code" in mmap.columns and "CMS_Region_Number" in mmap.columns:
                mmap = mmap.copy()
                mmap["State_Code"] = mmap["State_Code"].astype(str).str.strip().str.upper()
                mmap["CMS_Region_Number"] = pd.to_numeric(mmap["CMS_Region_Number"], errors="coerce")
                for _, row in mmap.iterrows():
                    st = row["State_Code"]
                    rn = row["CMS_Region_Number"]
                    if pd.isna(rn) or not st:
                        continue
                    rn_i = int(rn)
                    by_state[st] = rn_i
                    by_region.setdefault(rn_i, []).append(st)
                for rn in by_region:
                    by_region[rn] = sorted(set(by_region[rn]))
        except Exception:
            pass
    _CMS_REGION_BY_STATE = by_state
    _CMS_REGION_STATES = by_region
    return by_state, by_region


def metric_eligible_mask(df: pd.DataFrame, metric_key: str) -> tuple[pd.Series, dict[str, int]]:
    """
    Boolean mask for facilities included in a distribution for ``metric_key``.

    Matches ``tools/generation/generate_metrics.py`` grain: quarter HPRD uses
    sum(hours) / sum(MDScensus); contract % uses sum(contract hrs) / sum(total hrs).
    """
    spec = GEO_DIST_METRICS.get(metric_key)
    if not spec:
        raise ValueError(f"Unknown metric: {metric_key}")
    col = spec["column"]
    if col not in df.columns and spec.get("optional_column"):
        return pd.Series(False, index=df.index), {"ineligible_rows": len(df)}
    if col not in df.columns:
        fb = spec.get("fallback_column")
        if fb and fb in df.columns:
            col = fb
        else:
            raise ValueError(f"Metric column missing: {col}")

    reasons: dict[str, int] = {}
    mask = pd.Series(True, index=df.index)

    if spec.get("requires_positive_denominator"):
        if "total_resident_days" in df.columns:
            rd = pd.to_numeric(df["total_resident_days"], errors="coerce").fillna(0.0)
            m = rd > 0
            reasons["zero_resident_days"] = int((~m).sum())
            mask &= m
        elif "days_reported" in df.columns:
            dr = pd.to_numeric(df["days_reported"], errors="coerce").fillna(0.0)
            m = dr > 0
            reasons["zero_days_reported"] = int((~m).sum())
            mask &= m
        if metric_key != "avg_census" and "MDScensus" in df.columns:
            mc = pd.to_numeric(df["MDScensus"], errors="coerce").fillna(0.0)
            m2 = mc > 0
            reasons["zero_mdscensus"] = int((mask & ~m2).sum())
            mask &= m2

    if metric_key == "contract_pct" and "Total_Nurse_Hours" in df.columns:
        th = pd.to_numeric(df["Total_Nurse_Hours"], errors="coerce").fillna(0.0)
        m3 = th > 0
        reasons["zero_total_hours"] = int((mask & ~m3).sum())
        mask &= m3

    vals = pd.to_numeric(df[col], errors="coerce")
    m4 = vals.notna()
    reasons["missing_metric"] = int((mask & ~m4).sum())
    mask &= m4

    return mask, reasons


def metric_series_for_frame(df: pd.DataFrame, metric_key: str) -> tuple[pd.Series, int]:
    """Return numeric series for metric and total excluded row count."""
    mask, reasons = metric_eligible_mask(df, metric_key)
    spec = GEO_DIST_METRICS.get(metric_key) or {}
    col = spec.get("column", "")
    if col not in df.columns:
        fb = spec.get("fallback_column")
        col = fb if fb and fb in df.columns else col
    if col not in df.columns:
        return pd.Series(dtype=float), int(len(df))
    excluded = int((~mask).sum())
    vals = pd.to_numeric(df.loc[mask, col], errors="coerce")
    return cast(pd.Series, vals.astype(float)), excluded


def recommended_display_type(n: int, geography_type: str = "") -> str:
    geo = str(geography_type or "").strip().lower()
    hist_min = GEO_DIST_MIN_HISTOGRAM_COUNTY if geo == "county" else GEO_DIST_MIN_HISTOGRAM
    if n < GEO_DIST_MIN_PEER_TABLE:
        return "insufficient_sample"
    if n < GEO_DIST_MIN_DOTPLOT:
        return "peer_table"
    if n < hist_min:
        return "dotplot"
    return "histogram"


def _percentile_mean_rank(values: np.ndarray, facility_value: float) -> Optional[float]:
    """Mean-rank percentile in [0, 100]; requires caller to enforce minimum n."""
    if values.size == 0:
        return None
    below = int(np.sum(values < facility_value))
    equal = int(np.sum(values == facility_value))
    rank = below + (equal + 1) / 2.0
    return round(100.0 * rank / float(values.size), 1)


def _share_strictly_below(values: np.ndarray, facility_value: float) -> Optional[float]:
    """Share of peer facilities strictly below ``facility_value`` (0–100). Matches summary pct_below."""
    if values.size < GEO_DIST_MIN_SHARE_BELOW:
        return None
    return round(100.0 * float(np.sum(values < facility_value)) / float(values.size), 1)


def _facility_ranks(values: np.ndarray, facility_value: float) -> tuple[Optional[int], Optional[int]]:
    """
    Returns (rank_from_bottom, rank_from_top), 1-based.

    rank_from_bottom: 1 = lowest value; rank_from_top: 1 = highest value.
    """
    if values.size == 0:
        return None, None
    rank_bottom = int(np.sum(values < facility_value)) + 1
    rank_top = int(np.sum(values > facility_value)) + 1
    return rank_bottom, rank_top


def _histogram_bins(values: np.ndarray, n_bins: int = 18) -> list[dict[str, float]]:
    if values.size == 0:
        return []
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    if vmin == vmax:
        return [{"bin_start": vmin, "bin_end": vmax, "count": int(values.size)}]
    counts, edges = np.histogram(values, bins=n_bins)
    bins: list[dict[str, float]] = []
    for i, cnt in enumerate(counts):
        bins.append(
            {
                "bin_start": round(float(edges[i]), 4),
                "bin_end": round(float(edges[i + 1]), 4),
                "count": int(cnt),
            }
        )
    assert int(sum(b["count"] for b in bins)) == int(values.size)
    return bins


def _distribution_stats(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    return {
        "n": int(values.size),
        "mean": _round3(float(np.mean(values))),
        "median": _round3(float(np.median(values))),
        "min": _round3(float(np.min(values))),
        "max": _round3(float(np.max(values))),
        "p10": _round3(float(np.percentile(values, 10))),
        "p25": _round3(float(np.percentile(values, 25))),
        "p75": _round3(float(np.percentile(values, 75))),
        "p90": _round3(float(np.percentile(values, 90))),
    }


def filter_geography_frame(
    df: pd.DataFrame,
    geography_type: str,
    geography_value: str,
    provnum: str,
    city_lookup: dict[str, dict[str, str]],
) -> tuple[pd.DataFrame, str, Optional[str]]:
    gtype = (geography_type or "").strip().lower()
    if gtype not in GEOGRAPHY_TYPES:
        raise ValueError(f"Invalid geography_type: {geography_type}")
    fac_row = df[df["PROVNUM"] == provnum]
    fac_state = str(fac_row["STATE"].iloc[0]) if not fac_row.empty else ""
    label = geography_value or ""
    unavailable: Optional[str] = None

    if gtype == "national":
        out = df
        label = label or "United States"
        return out, label, unavailable

    if gtype == "state":
        st = (geography_value or fac_state).strip().upper()
        out = df[df["STATE"] == st]
        label = label or st
        return cast(pd.DataFrame, out), label, unavailable

    if gtype == "region":
        by_state, by_region = _load_cms_region_maps()
        rn: Optional[int] = None
        if geography_value and str(geography_value).strip().isdigit():
            rn = int(str(geography_value).strip())
        elif fac_state in by_state:
            rn = by_state[fac_state]
        if rn is None:
            unavailable = "CMS region mapping unavailable for this facility."
            return cast(pd.DataFrame, df.iloc[0:0]), label or "CMS region", unavailable
        states = by_region.get(rn) or []
        if not states and fac_state:
            states = [fac_state]
        out = df[df["STATE"].isin(states)]
        label = label or f"CMS Region {rn}"
        return cast(pd.DataFrame, out), label, unavailable

    if gtype == "county":
        county_norm = _normalize_county(geography_value)
        if not county_norm and not fac_row.empty and "_county_norm" in fac_row.columns:
            county_norm = str(fac_row["_county_norm"].iloc[0] or "")
        st = fac_state
        if not county_norm:
            unavailable = "County name not available for this facility."
            return cast(pd.DataFrame, df.iloc[0:0]), label or "County", unavailable
        out = df[(df["STATE"] == st) & (df["_county_norm"] == county_norm)]
        county_label = geography_value or (
            str(fac_row["COUNTY_NAME"].iloc[0]) if not fac_row.empty and "COUNTY_NAME" in fac_row.columns else ""
        )
        label = county_label.strip() or county_norm
        if st:
            label = f"{label}, {st}"
        return cast(pd.DataFrame, out), label, unavailable

    if gtype == "city":
        city_key = ""
        if geography_value and "|" in geography_value:
            city_key = geography_value.strip().upper()
        else:
            meta = city_lookup.get(provnum) or {}
            city_key = _normalize_city(meta.get("city"), meta.get("state") or fac_state)
        if not city_key:
            unavailable = (
                "City is not available in CMS Provider Information for this facility. "
                "Try county, state, region, or national comparison."
            )
            return cast(pd.DataFrame, df.iloc[0:0]), label or "City", unavailable
        city_name, st_part = city_key.split("|", 1)
        prov_cities = {normalize_provnum(k): v for k, v in city_lookup.items()}
        ccns_in_city = [
            ccn
            for ccn, meta in prov_cities.items()
            if _normalize_city(meta.get("city"), meta.get("state")) == city_key
        ]
        if not ccns_in_city:
            unavailable = "No facilities matched this city in Provider Information."
            return cast(pd.DataFrame, df.iloc[0:0]), label or city_name, unavailable
        out = df[(df["PROVNUM"].isin(ccns_in_city)) & (df["STATE"] == st_part)]
        label = f"{city_name.title()}, {st_part}"
        return cast(pd.DataFrame, out), label, unavailable

    raise ValueError(f"Unhandled geography_type: {geography_type}")


def resolve_hprd_threshold(
    state_abbr: str,
    metric_key: str,
    threshold_override: Optional[float],
    macpac_getter: Optional[Callable[[str], Optional[dict[str, Any]]]] = None,
) -> tuple[Optional[float], str]:
    if threshold_override is not None:
        try:
            t = float(threshold_override)
            if t > 0:
                return t, "custom"
        except (TypeError, ValueError):
            pass
    spec = GEO_DIST_METRICS.get(metric_key) or {}
    hprd_type = spec.get("threshold_hprd_type")
    if not hprd_type:
        return None, "none"
    st = (state_abbr or "").strip().upper()
    if hprd_type == "direct_care" and st == "CA":
        return 3.50, "ca_direct_care_statute"
    if hprd_type == "total" and st == "GA":
        return 2.00, "ga_total_statute"
    if hprd_type == "direct_care" and st == "NJ":
        return 2.50, "nj_direct_care_statute"
    if macpac_getter:
        std = macpac_getter(st) or macpac_getter(state_abbr)  # type: ignore[arg-type]
        if std:
            if std.get("Is_Federal_Minimum"):
                return None, "federal_minimum"
            vt = std.get("Value_Type") or std.get("value_type")
            if vt == "range":
                mn = std.get("Min_Staffing") or std.get("min_staffing")
                if mn is not None:
                    return float(mn), "macpac_min"
                mx = std.get("Max_Staffing") or std.get("max_staffing")
                if mx is not None:
                    return float(mx), "macpac_max"
            mn = std.get("Min_Staffing") or std.get("min_staffing")
            if mn is not None:
                return float(mn), "macpac_min"
    return None, "none"


def build_peer_rows(
    frame: pd.DataFrame,
    metric_key: str,
    focus_provnum: str,
    sort_mode: str,
    city_lookup: dict[str, dict[str, str]],
    limit: int = 12,
) -> list[dict[str, Any]]:
    spec = GEO_DIST_METRICS.get(metric_key) or {}
    col = spec.get("column", "")
    if col not in frame.columns and spec.get("fallback_column"):
        col = spec["fallback_column"]
    mask, _ = metric_eligible_mask(frame, metric_key)
    eligible = cast(pd.DataFrame, frame.loc[mask])
    peers: list[dict[str, Any]] = []
    focus_val: Optional[float] = None
    for _, row in eligible.iterrows():
        ccn = normalize_provnum(row.get("PROVNUM"))
        val = pd.to_numeric(row.get(col), errors="coerce")
        if pd.isna(val):
            continue
        v = float(val)
        if ccn == focus_provnum:
            focus_val = v
        city_meta = city_lookup.get(ccn) or {}
        peers.append(
            {
                "provnum": ccn,
                "provname": str(row.get("PROVNAME") or "").strip(),
                "state": str(row.get("STATE") or "").strip().upper(),
                "city": city_meta.get("city") or "",
                "county": str(row.get("COUNTY_NAME") or "").strip(),
                "total_nurse_hprd": _round3(row.get("Total_Nurse_HPRD")),
                "rn_care_hprd": _round3(row.get("RN_Care_HPRD")),
                "nurse_aide_hprd": _round3(row.get("Nurse_Assistant_HPRD")),
                "contract_pct": _round2(row.get("Contract_Percentage")),
                "avg_census": _round2(row.get("avg_daily_census") or row.get("MDScensus")),
                "metric_value": _round3(v),
                "delta_from_focus": None,
                "is_focus": ccn == focus_provnum,
            }
        )
    if focus_val is None:
        for p in peers:
            if p["is_focus"]:
                focus_val = p["metric_value"]
                break

    def sort_key(p: dict[str, Any]) -> float:
        mv = p.get("metric_value")
        if mv is None:
            return 1e9
        if sort_mode.startswith("closest"):
            return abs(float(mv) - float(focus_val or mv))
        if sort_mode == "highest_total_nurse_hprd":
            return -float(p.get("total_nurse_hprd") or mv)
        if sort_mode == "lowest_total_nurse_hprd":
            return float(p.get("total_nurse_hprd") or mv)
        if sort_mode == "highest_rn_care_hprd":
            return -float(p.get("rn_care_hprd") or mv)
        if sort_mode == "highest_contract_pct":
            return -float(p.get("contract_pct") or mv)
        return abs(float(mv) - float(focus_val or mv))

    peers.sort(key=sort_key)
    if focus_val is not None:
        for p in peers:
            if p["metric_value"] is not None:
                p["delta_from_focus"] = _round3(float(p["metric_value"]) - float(focus_val))

    # Always include focus facility when capping; limit=None returns full peer list (state modal pagination).
    if limit is None:
        return peers
    focus_rows = [p for p in peers if p["is_focus"]]
    others = [p for p in peers if not p["is_focus"]]
    if len(peers) <= max(limit, 25):
        return peers
    out = focus_rows[:]
    for p in others:
        if len(out) >= limit + len(focus_rows):
            break
        out.append(p)
    return out


def build_interpretation(
    *,
    facility_name: str,
    metric_label: str,
    quarter_label: str,
    geography_label: str,
    facility_value: float,
    median: Optional[float],
    percentile: Optional[float],
    share_below: Optional[float],
    n: int,
    display_type: str,
    small_sample_flag: bool,
    facility_in_sample: bool,
) -> str:
    parts: list[str] = []
    parts.append(
        f"{facility_name} reported {_round3(facility_value)} {metric_label} in {quarter_label} "
        f"(PBJ facility-quarter metric, same source as this distribution)."
    )
    if not facility_in_sample:
        parts.append(
            "This facility-quarter value could not be matched to the eligible peer sample "
            "(missing census, hours, or metric); chart statistics exclude it."
        )
        return " ".join(parts)
    if display_type == "insufficient_sample":
        parts.append(
            "Too few facilities in this geography for a meaningful distribution. "
            "Try county, state, region, or national comparison."
        )
        return " ".join(parts)
    if n > 0:
        parts.append(f"Among {n} peer facilities with valid {metric_label} in {geography_label},")
    if share_below is not None:
        parts.append(
            f"approximately {share_below:.0f}% had strictly lower values "
            f"(same definition as the summary “peers below” share)."
        )
    if median is not None:
        if facility_value < median:
            parts.append(f"The geography median was {_round3(median)} (this facility was below median).")
        elif facility_value > median:
            parts.append(f"The geography median was {_round3(median)} (this facility was above median).")
        else:
            parts.append(f"The geography median was {_round3(median)} (this facility matched median).")
    if percentile is not None:
        parts.append(f"Mean-rank percentile was approximately {int(round(percentile))}.")
    elif small_sample_flag:
        parts.append("Treat rankings as directional because the sample is small.")
    if metric_label.lower().find("contract") >= 0:
        parts.append("Contract staffing is often skewed; median is usually more informative than mean.")
    return " ".join(parts)


def geography_counts_for_quarter(
    df_q: pd.DataFrame,
    provnum: str,
    city_lookup: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Facility counts by geography for default selector logic."""
    fac = df_q[df_q["PROVNUM"] == provnum]
    st = str(fac["STATE"].iloc[0]) if not fac.empty else ""
    county_norm = str(fac["_county_norm"].iloc[0]) if not fac.empty and "_county_norm" in fac.columns else ""
    city_key = ""
    meta = city_lookup.get(provnum) or {}
    city_key = _normalize_city(meta.get("city"), meta.get("state") or st)
    by_state, by_region = _load_cms_region_maps()
    rn = by_state.get(st)
    region_n = int(len(df_q[df_q["STATE"].isin(by_region.get(rn, [st]))])) if rn else 0
    county_n = int(len(df_q[(df_q["STATE"] == st) & (df_q["_county_norm"] == county_norm)])) if county_norm else 0
    city_n = 0
    if city_key:
        city_name, st_part = city_key.split("|", 1)
        ccns = [
            ccn
            for ccn, m in city_lookup.items()
            if _normalize_city(m.get("city"), m.get("state")) == city_key
        ]
        city_n = int(len(df_q[(df_q["PROVNUM"].isin(ccns)) & (df_q["STATE"] == st_part)]))
    return {
        "national": int(len(df_q)),
        "state": int(len(df_q[df_q["STATE"] == st])) if st else 0,
        "region": region_n,
        "county": county_n,
        "city": city_n,
        "state_abbr": st,
        "county_label": str(fac["COUNTY_NAME"].iloc[0]) if not fac.empty and "COUNTY_NAME" in fac.columns else "",
        "city_label": meta.get("city") or "",
        "cms_region_number": rn,
    }


def default_geography_type(counts: dict[str, Any]) -> str:
    county_n = int(counts.get("county") or 0)
    state_n = int(counts.get("state") or 0)
    if county_n >= GEO_DIST_MIN_HISTOGRAM:
        return "county"
    if state_n >= GEO_DIST_MIN_PEER_TABLE:
        return "state"
    if county_n >= GEO_DIST_MIN_PEER_TABLE:
        return "county"
    return "national"


def build_geo_distribution_payload(
    *,
    provnum: str,
    quarter: str,
    metric: str,
    geography_type: str,
    geography_value: str = "",
    facility_name: str = "",
    threshold_override: Optional[float] = None,
    peer_sort: str = "",
    macpac_getter: Optional[Callable[[str], Optional[dict[str, Any]]]] = None,
) -> dict[str, Any]:
    metric_key = (metric or "").strip().lower()
    if metric_key not in GEO_DIST_METRICS:
        return {"error": f"Unknown metric: {metric}", "available_metrics": sorted(GEO_DIST_METRICS.keys())}

    cy = normalize_cy_qtr(quarter)
    if not cy:
        return {"error": "Invalid quarter. Use CYyyyyQn or yyyyQn."}

    df = _load_facility_quarterly_df()
    if df is None or df.empty:
        return {"error": "facility_quarterly_metrics.csv not available."}

    ccn = normalize_provnum(provnum)
    city_lookup = _load_provnum_city_lookup()
    df_q = _quarter_slice(df, cy)
    if df_q.empty:
        return {"error": f"No facility-quarter rows for {cy}."}

    fac_rows = df_q[df_q["PROVNUM"] == ccn]
    if fac_rows.empty:
        return {"error": f"Facility {ccn} not found for {cy}."}

    try:
        geo_df, geography_label, geo_unavailable = filter_geography_frame(
            df_q, geography_type, geography_value, ccn, city_lookup
        )
    except ValueError as exc:
        return {"error": str(exc)}

    if geo_unavailable:
        return {
            "error": geo_unavailable,
            "geography_type": geography_type,
            "quarter": cy,
            "metric": metric_key,
            "data_source": "facility_quarterly_metrics.csv",
            "distribution_note": "PBJ-derived facility-quarter distribution (not CMS published rollups).",
        }

    try:
        series, excluded = metric_series_for_frame(geo_df, metric_key)
    except ValueError as exc:
        return {"error": str(exc)}

    fac_series, fac_excluded = metric_series_for_frame(fac_rows, metric_key)
    facility_value: Optional[float] = None
    if not fac_series.empty:
        facility_value = float(fac_series.iloc[0])

    values = series.to_numpy(dtype=float)
    n = int(values.size)
    n_geography_rows = int(len(geo_df))
    geo_type_norm = str(geography_type or "state").lower()
    display_type = recommended_display_type(n, geo_type_norm)
    small_sample_flag = n < (
        GEO_DIST_MIN_HISTOGRAM_COUNTY if geo_type_norm == "county" else GEO_DIST_MIN_HISTOGRAM
    )
    stats = _distribution_stats(values)

    percentile: Optional[float] = None
    share_below: Optional[float] = None
    facility_rank_from_bottom: Optional[int] = None
    facility_rank_from_top: Optional[int] = None
    fac_mask, _ = metric_eligible_mask(fac_rows, metric_key)
    facility_in_sample = bool(fac_mask.any()) and facility_value is not None
    if facility_value is not None and n > 0 and facility_in_sample:
        rank_b, rank_t = _facility_ranks(values, float(facility_value))
        facility_rank_from_bottom = rank_b
        facility_rank_from_top = rank_t
        share_below = _share_strictly_below(values, float(facility_value))
        if n >= GEO_DIST_MIN_PERCENTILE:
            percentile = _percentile_mean_rank(values, float(facility_value))

    threshold, threshold_source = resolve_hprd_threshold(
        str(fac_rows["STATE"].iloc[0]),
        metric_key,
        threshold_override,
        macpac_getter=macpac_getter,
    )
    count_below_threshold = None
    share_below_threshold = None
    if threshold is not None and n > 0 and metric_key in ("total_nurse_hprd", "nurse_care_hprd"):
        count_below_threshold = int(np.sum(values < float(threshold)))
        share_below_threshold = round(100.0 * count_below_threshold / n, 1)

    q_label = cy[2:] if cy.upper().startswith("CY") else cy
    q_human = f"Q{q_label[-1]} {q_label[:4]}" if re.match(r"^\d{4}Q[1-4]$", q_label) else q_label

    sort_mode = peer_sort or GEO_DIST_METRICS[metric_key].get("peer_sort_default") or "closest"
    # State: return all eligible peers (client paginates). Region/county: cap closest peers.
    peer_limit: Optional[int] = 10
    if geo_type_norm == "state":
        peer_limit = None
    elif geo_type_norm == "county":
        peer_limit = 25
    eligible_geo = cast(pd.DataFrame, geo_df.loc[metric_eligible_mask(geo_df, metric_key)[0]])
    if geo_type_norm == "national":
        peers: list[dict[str, Any]] = []
    else:
        peers = build_peer_rows(eligible_geo, metric_key, ccn, sort_mode, city_lookup, limit=peer_limit)

    geo_counts = geography_counts_for_quarter(df_q, ccn, city_lookup)
    region_states: list[str] = []
    rn = int(geo_counts.get("cms_region_number") or 0)
    if geo_type_norm == "region" and rn:
        _, by_region = _load_cms_region_maps()
        region_states = list(by_region.get(rn) or [])

    interpretation = ""
    if facility_value is not None:
        interpretation = build_interpretation(
            facility_name=facility_name or str(fac_rows["PROVNAME"].iloc[0] or "This facility"),
            metric_label=GEO_DIST_METRICS[metric_key]["label"],
            quarter_label=q_human,
            geography_label=geography_label,
            facility_value=facility_value,
            median=stats.get("median"),
            percentile=percentile,
            share_below=share_below,
            n=n,
            display_type=display_type,
            small_sample_flag=small_sample_flag,
            facility_in_sample=facility_in_sample,
        )

    return {
        "facility_value": _round3(facility_value),
        "facility_value_source": "facility_quarterly_metrics.csv",
        "facility_in_sample": facility_in_sample,
        "geography_label": geography_label,
        "geography_type": geography_type,
        "geography_value": geography_value,
        "quarter": cy,
        "quarter_label": q_human,
        "metric": metric_key,
        "metric_label": GEO_DIST_METRICS[metric_key]["label"],
        "n": n,
        "n_geography_facilities": n_geography_rows,
        "n_excluded": excluded,
        "n_excluded_breakdown": metric_eligible_mask(geo_df, metric_key)[1],
        "mean": stats["mean"],
        "median": stats["median"],
        "min": stats["min"],
        "max": stats["max"],
        "p10": stats["p10"],
        "p25": stats["p25"],
        "p75": stats["p75"],
        "p90": stats["p90"],
        "percentile": percentile,
        "percentile_method": "mean_rank" if percentile is not None else None,
        "percentile_method_note": PERCENTILE_METHOD_NOTE if percentile is not None else None,
        "share_facilities_strictly_below": share_below,
        "facility_rank": facility_rank_from_top,
        "facility_rank_from_bottom": facility_rank_from_bottom,
        "facility_rank_from_top": facility_rank_from_top,
        "bins": _histogram_bins(values) if display_type == "histogram" else [],
        "values_ranked": [round(float(v), 4) for v in np.sort(values).tolist()] if n > 0 else [],
        "values_dotplot": [round(float(v), 4) for v in np.sort(values).tolist()] if display_type == "dotplot" else [],
        "peers": peers if display_type in ("dotplot", "peer_table", "histogram") else [],
        "peers_returned": len(peers),
        "peers_capped": peer_limit is not None and n > len(peers),
        "facility_state_abbr": str(geo_counts.get("state_abbr") or fac_rows["STATE"].iloc[0] or "").strip().upper(),
        "facility_county_label": str(geo_counts.get("county_label") or "").strip(),
        "cms_region_number": rn or geo_counts.get("cms_region_number"),
        "cms_region_states": region_states,
        "threshold": _round3(threshold),
        "threshold_source": threshold_source,
        "count_below_threshold": count_below_threshold,
        "share_below_threshold": share_below_threshold,
        "small_sample_flag": small_sample_flag,
        "recommended_display_type": display_type,
        "interpretation": interpretation,
        "data_source": "facility_quarterly_metrics.csv",
        "distribution_note": (
            "Facility-quarter values from PBJ-derived metrics (one value per facility per quarter). "
            "The comparison table may use CMS published rollups for state/region/national columns."
        ),
        "peer_sort": sort_mode,
        "available_metrics": sorted(GEO_DIST_METRICS.keys()),
    }


def build_geo_distribution_context(
    provnum: str,
    quarter: str,
) -> dict[str, Any]:
    cy = normalize_cy_qtr(quarter)
    if not cy:
        return {"error": "Invalid quarter."}
    df = _load_facility_quarterly_df()
    if df is None:
        return {"error": "facility_quarterly_metrics.csv not available."}
    ccn = normalize_provnum(provnum)
    city_lookup = _load_provnum_city_lookup()
    df_q = _quarter_slice(df, cy)
    counts = geography_counts_for_quarter(df_q, ccn, city_lookup)
    mask_t, _ = metric_eligible_mask(df_q, "total_nurse_hprd") if not df_q.empty else (pd.Series(dtype=bool), {})
    counts["n_eligible_total_nurse_hprd"] = int(mask_t.sum()) if len(mask_t) else 0
    default_geo = default_geography_type(counts)
    return {
        "quarter": cy,
        "quarter_label": cy[2:].replace("Q", " Q") if cy.startswith("CY") else cy,
        "geography_counts": counts,
        "default_geography_type": default_geo,
        "metrics": [
            {"key": k, "label": v["label"], "available": True}
            for k, v in GEO_DIST_METRICS.items()
        ],
        "thresholds": {
            "min_histogram": GEO_DIST_MIN_HISTOGRAM,
            "min_dotplot": GEO_DIST_MIN_DOTPLOT,
            "min_peer_table": GEO_DIST_MIN_PEER_TABLE,
        },
        "data_source": "facility_quarterly_metrics.csv",
    }
