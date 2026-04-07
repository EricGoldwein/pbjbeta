"""
Chain-level longitudinal CMS performance metrics (entity / affiliated chain).

Loads normalized rows from ``ownership/chain_performance_longitudinal.parquet`` or
``.csv``. Deployed facility packages can ship a small per-CCN slice
(``ownership/facility_<CCN>_entity_longitudinal.csv``) with every distinct
affiliated chain the facility ever had, so the server does not need the full
national longitudinal file.
"""

from __future__ import annotations

import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent


def _read_global_longitudinal(ownership_dir: Path) -> Optional[pd.DataFrame]:
    """Load the full longitudinal table from parquet (preferred) or CSV."""
    parquet_path = ownership_dir / "chain_performance_longitudinal.parquet"
    csv_path = ownership_dir / "chain_performance_longitudinal.csv"
    if parquet_path.is_file():
        try:
            return pd.read_parquet(parquet_path)
        except ImportError:
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Error reading longitudinal parquet: {exc}")
    if csv_path.is_file():
        try:
            return pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Error reading longitudinal CSV: {exc}")
    return None


def resolve_entity_id_to_hash(entity_id: str, ownership_dir: Path) -> Optional[str]:
    """
    Map CMS chain id (digits) to hash ``entity_id`` in longitudinal files.

    Non-numeric values are returned stripped (assumed to already be a hash key).
    """
    eid = str(entity_id).strip()
    if not eid:
        return None
    if eid.isdigit():
        lookup_path = ownership_dir / "entity_lookup.csv"
        if not lookup_path.is_file():
            return None
        entity_lookup = pd.read_csv(lookup_path)
        chain_id = float(eid)
        entity_row = entity_lookup[entity_lookup["chain_id"] == chain_id]
        if entity_row.empty:
            return None
        return str(entity_row.iloc[0]["entity_id"])
    return eid


def _quarter_sort_key_series(series: pd.Series) -> Any:
    """Sort key for CMS quarters like 2023Q1 (same idea as facility dashboard)."""

    def one(q: Any) -> tuple:
        if pd.isna(q) or not str(q).strip():
            return (0, 0)
        s = str(q).strip().upper()
        m = re.search(r"(\d{4})Q(\d)", s)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (0, 0)

    return series.map(one)


def affiliated_chain_ids_for_facility_provider_df(facility_df: pd.DataFrame) -> List[str]:
    """
    All distinct CMS affiliated-entity chain IDs for this facility's provider rows.

    Order follows provider history (earliest quarter / processing date first) so
    \"current vs prior\" aligns with the dashboard's entity line when possible.
    """
    if facility_df.empty or "affiliated_entity_id" not in facility_df.columns:
        return []
    col = "affiliated_entity_id"
    work = facility_df
    if "quarter" in work.columns:
        work = work.dropna(subset=["quarter"]).copy()
        work = work.sort_values("quarter", key=_quarter_sort_key_series)
    elif "processing_date" in work.columns:
        work = work.sort_values("processing_date")
    out: List[str] = []
    for raw in work[col]:
        if pd.isna(raw) or str(raw).strip().upper() in ("", "N", "N/A", "NAN", "NONE"):
            continue
        try:
            eid = str(int(float(raw)))
        except (ValueError, TypeError):
            eid = str(raw).strip()
        if eid and eid not in out:
            out.append(eid)
    return out


def _chain_ids_to_entity_hashes(chain_ids: List[str], lookup_df: pd.DataFrame) -> List[str]:
    hashes: List[str] = []
    for cid in chain_ids:
        s = str(cid).strip()
        if not s:
            continue
        if s.isdigit():
            row = lookup_df[lookup_df["chain_id"] == float(s)]
            if not row.empty:
                hashes.append(str(row.iloc[0]["entity_id"]))
        else:
            hashes.append(s)
    # de-dupe, preserve order
    seen = set()
    uniq: List[str] = []
    for h in hashes:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq


def build_longitudinal_subset_for_chain_ids(repo_root: Path, chain_ids: List[str]) -> Optional[pd.DataFrame]:
    """
    Rows from the full longitudinal file for all given CMS chain ids (via lookup).
    """
    if not chain_ids:
        return None
    ownership_dir = repo_root / "ownership"
    lookup_path = ownership_dir / "entity_lookup.csv"
    if not lookup_path.is_file():
        return None
    lookup = pd.read_csv(lookup_path)
    hashes = _chain_ids_to_entity_hashes(chain_ids, lookup)
    if not hashes:
        return None
    full = _read_global_longitudinal(ownership_dir)
    if full is None or full.empty:
        return None
    sub = full[full["entity_id"].isin(hashes)].copy()
    return sub if not sub.empty else None


def write_facility_longitudinal_slice_for_deploy(
    repo_root: Path,
    provnum: str,
    provider_csv_path: Path,
    output_csv_path: Path,
) -> bool:
    """
    Build ``facility_<CCN>_entity_longitudinal.csv`` for a Vercel (or other) package.

    Includes longitudinal rows for every distinct affiliated chain ID appearing
    in that facility's provider-info extract (full affiliation history).

    Returns:
        True if the slice file was written and is non-empty.
    """
    provnum = str(provnum).strip().zfill(6)
    try:
        pdf = pd.read_csv(provider_csv_path, low_memory=False, dtype={"ccn": str})
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: could not read provider CSV for entity slice: {exc}")
        return False
    if "ccn" not in pdf.columns:
        return False
    sub = pdf[pdf["ccn"].astype(str).str.strip().str.zfill(6) == provnum]
    chain_ids = affiliated_chain_ids_for_facility_provider_df(sub)
    df = build_longitudinal_subset_for_chain_ids(repo_root, chain_ids)
    if df is None or df.empty:
        return False
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    return True


def load_entity_longitudinal_metrics(
    entity_id: str,
    facility_ccn: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Load longitudinal metric rows for one entity (chain).

    If ``facility_ccn`` is set and ``ownership/facility_<CCN>_entity_longitudinal.csv``
    exists (deployment slice), it is consulted first so the server does not need
    the full national longitudinal file.

    Args:
        entity_id: Hash key from ``entity_lookup`` or numeric CMS chain id string.
        facility_ccn: Optional 6-digit CCN for per-facility slice resolution.

    Returns:
        DataFrame for that entity, or None if not found.
    """
    try:
        ownership_dir = _REPO_ROOT / "ownership"
        target_hash = resolve_entity_id_to_hash(entity_id, ownership_dir)
        if target_hash is None:
            return None

        if facility_ccn:
            ccn = str(facility_ccn).strip().zfill(6)
            slice_path = ownership_dir / f"facility_{ccn}_entity_longitudinal.csv"
            if slice_path.is_file():
                try:
                    df = pd.read_csv(slice_path, low_memory=False)
                    entity_data = df[df["entity_id"] == target_hash].copy()
                    if not entity_data.empty:
                        return entity_data
                except Exception as exc:  # noqa: BLE001
                    print(f"Warning: Error loading facility entity slice for {ccn}: {exc}")

        full = _read_global_longitudinal(ownership_dir)
        if full is None or full.empty:
            return None
        entity_data = full[full["entity_id"] == target_hash].copy()
        return entity_data if not entity_data.empty else None

    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Error in load_entity_longitudinal_metrics for {entity_id}: {exc}")
        traceback.print_exc()
        return None


def get_entity_key_metrics_over_time(
    entity_id: str,
    facility_ccn: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build key chain metrics by report period for charts and tables.

    Args:
        entity_id: Hash or numeric chain id (same as ``load_entity_longitudinal_metrics``).
        facility_ccn: Optional CCN for per-facility deployment slice.

    Returns:
        Dict with ``entity_id``, ``entity_name``, ``metrics_by_period``, or None.
    """
    try:
        df = load_entity_longitudinal_metrics(entity_id, facility_ccn=facility_ccn)
        if df is None or df.empty:
            return None

        entity_name = df["entity_name"].iloc[0] if "entity_name" in df.columns else None

        key_metrics: Dict[str, str] = {
            "Number of facilities": "number_of_facilities",
            "Average overall 5-star rating": "avg_overall_rating",
            "Average health inspection rating": "avg_health_inspection_rating",
            "Average staffing rating": "avg_staffing_rating",
            "Average quality rating": "avg_quality_rating",
            "Number of Special Focus Facilities (SFF)": "total_sff",
            "Average total nurse hours per resident day": "avg_total_nurse_hprd",
            "Average total Registered Nurse hours per resident day": "avg_rn_hprd",
            "Average total weekend nurse hours per resident day": "avg_weekend_nurse_hprd",
            "Average total nursing staff turnover percentage": "avg_nursing_turnover",
            "Average Registered Nurse turnover percentage": "avg_rn_turnover",
            "Total number of fines": "total_fines",
            "Total amount of fines in dollars": "total_fine_amount",
            "Number of facilities with an abuse icon": "facilities_with_abuse_icon",
            "Percent of facilities classified as for-profit": "pct_for_profit",
        }

        metrics_by_period: List[Dict[str, Any]] = []
        for period in sorted(df["report_period"].unique()):
            period_data = df[df["report_period"] == period].copy()
            period_metrics: Dict[str, Any] = {"period": period}

            for metric_name, metric_key in key_metrics.items():
                metric_rows = period_data[period_data["metric_name"] == metric_name]
                if not metric_rows.empty:
                    value = metric_rows.iloc[0]["metric_value"]
                    try:
                        if pd.notna(value):
                            if isinstance(value, str):
                                value = float(value.replace(",", "").replace("$", ""))
                            else:
                                value = float(value)
                        else:
                            value = None
                    except (ValueError, TypeError):
                        value = None
                    period_metrics[metric_key] = value
                else:
                    period_metrics[metric_key] = None

            metrics_by_period.append(period_metrics)

        return {
            "entity_id": str(entity_id).strip(),
            "entity_name": entity_name,
            "metrics_by_period": metrics_by_period,
        }

    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Error extracting key metrics for entity {entity_id}: {exc}")
        return None
