"""
PBJ320 shared metric calculation helpers.

Architecture: clean at entry, calculate from raw, display at the edge.
Use these for all pooled HPRD, share, compliance, and safe-division math.
Display rounding belongs in export/render layers only.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

# Source attribution labels (compact; use in API metadata and exports).
SOURCE_CMS_PBJ = "cms_pbj"
SOURCE_PBJ320_DERIVED = "pbj320_derived_from_cms_pbj"
SOURCE_CMS_PROVIDER_INFO = "cms_provider_info"
SOURCE_CMS_OWNERSHIP = "cms_ownership"
SOURCE_CMS_SURVEY = "cms_survey"
SOURCE_APP_CONFIG = "app_config"
SOURCE_UNAVAILABLE = "unavailable"


def coerce_num(value: Any) -> Optional[float]:
    """Parse a scalar to float; None/NaN/empty → None (never 0)."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(n):
        return None
    return n


def safe_divide(numerator: Any, denominator: Any) -> Optional[float]:
    """Return numerator/denominator or None when denominator missing or ≤ 0."""
    num = coerce_num(numerator)
    den = coerce_num(denominator)
    if num is None or den is None or den <= 0:
        return None
    return num / den


def row_sum_hours(df: pd.DataFrame, hour_cols: Sequence[str], *, min_count: Optional[int] = None) -> pd.Series:
    """
    Row-wise sum of hour columns; NaN when any required column is missing.

    ``min_count`` defaults to len(hour_cols) so partial missing rows stay missing.
    """
    cols = [c for c in hour_cols if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    mc = len(hour_cols) if min_count is None else min_count
    M = df[cols].apply(lambda col: pd.to_numeric(col, errors="coerce"))
    return M.sum(axis=1, min_count=mc)


def pooled_hprd_from_df(
    df: pd.DataFrame,
    hour_cols: Sequence[str],
    census_col: str = "MDScensus",
) -> Optional[float]:
    """
    Pooled HPRD = sum(hours) / sum(census) on days with census > 0 and complete hours.
    """
    if df is None or len(df) == 0 or census_col not in df.columns:
        return None
    mdc = pd.to_numeric(df[census_col], errors="coerce")
    hrs = row_sum_hours(df, hour_cols)
    ok = mdc.notna() & (mdc > 0) & hrs.notna()
    den = float(mdc[ok].sum())
    if den <= 0:
        return None
    return float(hrs[ok].sum()) / den


def pooled_hprd_from_series(hours: pd.Series, census: pd.Series) -> Optional[float]:
    """Pooled HPRD from aligned hour/census series."""
    h = pd.to_numeric(hours, errors="coerce")
    c = pd.to_numeric(census, errors="coerce")
    ok = c.notna() & (c > 0) & h.notna()
    den = float(c[ok].sum())
    if den <= 0:
        return None
    return float(h[ok].sum()) / den


def contract_share_pct(contract_hours: Any, total_hours: Any) -> Optional[float]:
    """Contract share = 100 * sum(contract) / sum(total); None when total missing or ≤ 0."""
    c = coerce_num(contract_hours)
    t = coerce_num(total_hours)
    if c is None or t is None or t <= 0:
        return None
    return c / t * 100.0


def compliance_share(days_meeting: int, observed_days: int) -> Optional[float]:
    """Compliance share = days meeting / observed days; None when no observed days."""
    if observed_days <= 0:
        return None
    return float(days_meeting) / float(observed_days) * 100.0


def round_half_up_display(value: Any, decimals: int = 2) -> Optional[float]:
    """Financial rounding for display/export; preserves None/NaN (never maps to 0)."""
    n = coerce_num(value)
    if n is None:
        return None
    if decimals == 0:
        q = Decimal("1")
    elif decimals == 1:
        q = Decimal("0.1")
    elif decimals == 3:
        q = Decimal("0.001")
    else:
        q = Decimal("0.01")
    return float(Decimal(str(n)).quantize(q, rounding=ROUND_HALF_UP))


# Standard PBJ hour column groups (traceable to CMS PBJ daily nurse staffing).
TOTAL_NURSE_HOUR_COLS = (
    "Hrs_RNDON",
    "Hrs_RNadmin",
    "Hrs_RN",
    "Hrs_LPNadmin",
    "Hrs_LPN",
    "Hrs_CNA",
    "Hrs_NAtrn",
    "Hrs_MedAide",
)
TOTAL_RN_HOUR_COLS = ("Hrs_RNDON", "Hrs_RNadmin", "Hrs_RN")
DIRECT_CARE_HOUR_COLS = ("Hrs_RN", "Hrs_LPN", "Hrs_CNA", "Hrs_NAtrn", "Hrs_MedAide")
CONTRACT_HOUR_COLS = (
    "Hrs_RNDON_ctr",
    "Hrs_RNadmin_ctr",
    "Hrs_RN_ctr",
    "Hrs_LPNadmin_ctr",
    "Hrs_LPN_ctr",
    "Hrs_CNA_ctr",
    "Hrs_NAtrn_ctr",
    "Hrs_MedAide_ctr",
)


def sum_column_pool(df: pd.DataFrame, col: str, census_col: str = "MDScensus") -> Optional[float]:
    """Sum a single hour column on days with valid census > 0 and non-missing hours."""
    if df is None or len(df) == 0 or col not in df.columns or census_col not in df.columns:
        return None
    mdc = pd.to_numeric(df[census_col], errors="coerce")
    hrs = pd.to_numeric(df[col], errors="coerce")
    ok = mdc.notna() & (mdc > 0) & hrs.notna()
    return float(hrs[ok].sum()) if ok.any() else None


def unavailable_metrics_dict() -> dict[str, Any]:
    """Template for API payloads when scope has no computable metrics."""
    return {
        "available": False,
        "total_days": 0,
        "avg_total_hprd": None,
        "avg_census": None,
    }
