"""
Shared normalization for PBJ daily staffing (nurse and non-nurse) and related quarter keys.

Used by facility CSV builders and APIs. Keeps CCN/PROVNUM, WorkDate, and CY_Qtr handling consistent.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional, cast

import json

import pandas as pd


def provnum_search_variants(provnum: str) -> list[str]:
    """Match facility extraction logic: upper, zfill(6), lstrip('0') for digit CCNs."""
    p = str(provnum).strip()
    variants = [p.upper()]
    if p.isdigit():
        variants.extend([p.zfill(6), p.lstrip("0") or "0"])
    return list(dict.fromkeys(variants))


def _normalize_provnum_like(value: Any) -> str:
    """Normalize CCN/PROVNUM values for chunk-index matching."""
    s = str(value or "").strip()
    if not s:
        return ""
    if s.isdigit():
        return s.zfill(6)
    return s.upper()


def _chunk_index_cache_path(csv_path: str) -> str:
    """v2 cache: built with PROVNUM/CCN alias resolution (invalidate v1 on upgrade)."""
    return csv_path + ".provnum_chunk_index.v2.json"


def normalize_header_key(col: object) -> str:
    """Uppercase slug with letters/digits only (for matching CMS header variants)."""
    s = str(col).strip().replace("\ufeff", "").upper()
    return re.sub(r"[^A-Z0-9]+", "", s)


# Lower = higher priority. Covers legacy non-nurse / PBJ headers before universal PROVNUM.
_PROVNUM_SLUG_PRIORITY: dict[str, int] = {
    "PROVNUM": 0,
    "CCN": 1,
    "CMSCERTIFICATIONNUMBERCCN": 2,
    "PRVDRNUM": 3,
    "PROVIDERNUM": 4,
    "PROVIDERNUMBER": 5,
    "FACILITYID": 6,
}


def pick_provnum_source_column(column_labels: list[str]) -> Optional[str]:
    """
    Pick the facility-id column label from already-stripped header names.

    Returns the matching label string (e.g. ``PROVNUM`` or ``CCN``) or None.
    """
    best_pr: Optional[int] = None
    best_lab: Optional[str] = None
    for raw in column_labels:
        lab = str(raw).strip().replace("\ufeff", "")
        if not lab:
            continue
        key = normalize_header_key(lab)
        pr = _PROVNUM_SLUG_PRIORITY.get(key)
        if pr is None:
            continue
        if best_pr is None or pr < best_pr:
            best_pr = pr
            best_lab = lab
    return best_lab


def coerce_provnum_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a ``PROVNUM`` column exists, renaming from CMS aliases (CCN, long CCN title, etc.).

    Operates on the dataframe's current column labels (call after stripping BOM/spaces).
    """
    if "PROVNUM" in df.columns:
        return df
    picked = pick_provnum_source_column([str(c) for c in df.columns])
    if not picked or picked not in df.columns:
        return df
    return df.rename(columns={picked: "PROVNUM"})


def provnum_original_column_name(csv_path: str) -> str:
    """
    Return the CSV header token pandas should use in ``usecols`` for the facility-id column.

    Prefers ``PROVNUM``; falls back to ``CCN`` / CMS long CCN label / other CMS variants.
    """
    last_err: Optional[Exception] = None
    for enc in (None, "utf-8-sig", "latin1", "cp1252"):
        try:
            peek = pd.read_csv(csv_path, nrows=0, low_memory=False, encoding=enc)
            best_raw: Optional[object] = None
            best_pr: Optional[int] = None
            for raw in peek.columns:
                lab = str(raw).strip().replace("\ufeff", "")
                key = normalize_header_key(lab)
                pr = _PROVNUM_SLUG_PRIORITY.get(key)
                if pr is None:
                    continue
                if best_pr is None or pr < best_pr:
                    best_pr = pr
                    best_raw = raw
            if best_raw is None:
                raise ValueError(f"No PROVNUM or CCN-like column in {csv_path}")
            return str(best_raw)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    if last_err is not None:
        raise last_err
    raise ValueError(f"No PROVNUM or CCN-like column in {csv_path}")


def invalidate_provnum_chunk_index_cache(csv_path: str) -> None:
    """Remove cached chunk index (e.g. before rebuilding with a different chunksize)."""
    for cache_path in (
        csv_path + ".provnum_chunk_index.v2.json",
        csv_path + ".provnum_chunk_index.json",
    ):
        try:
            if os.path.isfile(cache_path):
                os.remove(cache_path)
        except OSError:
            pass


def pandas_chunk_read_memory_error(exc: BaseException) -> bool:
    """True if failure is likely RAM pressure while parsing a wide CSV chunk."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cannot allocate" in msg


def _build_provnum_chunk_index(csv_path: str, chunksize: int = 250_000) -> dict[str, Any]:
    """Build a lightweight mapping of normalized PROVNUM -> 1-based chunk ids."""
    prov_col = provnum_original_column_name(csv_path)
    prov_key = str(prov_col).strip().replace("\ufeff", "")

    def _usecols_facility_id(c: object) -> bool:
        return str(c).strip().replace("\ufeff", "") == prov_key

    chunk_ids_by_provnum: dict[str, list[int]] = {}
    chunk_count = 0
    encoding_used: Optional[str] = None
    last_err: Optional[Exception] = None
    for enc in (None, "utf-8-sig", "latin1", "cp1252"):
        chunk_ids_by_provnum = {}
        chunk_count = 0
        try:
            for chunk_count, df_chunk in enumerate(
                pd.read_csv(
                    csv_path,
                    low_memory=False,
                    usecols=_usecols_facility_id,
                    dtype=str,
                    chunksize=max(1, int(chunksize)),
                    encoding=enc,
                ),
                start=1,
            ):
                df_chunk.columns = [str(c).strip().replace("\ufeff", "") for c in df_chunk.columns]
                df_chunk = coerce_provnum_column(df_chunk)
                if "PROVNUM" not in df_chunk.columns:
                    continue
                vals = (
                    df_chunk["PROVNUM"]
                    .astype(str)
                    .map(_normalize_provnum_like)
                    .dropna()
                    .tolist()
                )
                for pv in set(v for v in vals if v):
                    ids = chunk_ids_by_provnum.setdefault(pv, [])
                    ids.append(chunk_count)
            encoding_used = enc or "default"
            break
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    if encoding_used is None:
        if last_err is not None:
            raise last_err
        raise ValueError(f"Could not index PROVNUM chunks for {csv_path}")
    out = {
        "csv_path": os.path.abspath(csv_path),
        "chunksize": int(chunksize),
        "chunk_count": int(chunk_count),
        "chunk_ids_by_provnum": chunk_ids_by_provnum,
        "encoding": encoding_used,
    }
    return out


def _load_or_build_provnum_chunk_index(csv_path: str, chunksize: int = 250_000) -> dict[str, Any]:
    """Load cached chunk index when possible; rebuild on cache miss/corruption/stale source."""
    cache_path = _chunk_index_cache_path(csv_path)
    legacy_cache = csv_path + ".provnum_chunk_index.json"
    try:
        st = os.stat(csv_path)
        src_size = int(st.st_size)
        src_mtime = int(st.st_mtime)
    except OSError:
        src_size = None
        src_mtime = None

    def _source_matches(data: dict[str, Any]) -> bool:
        if src_size is None:
            return True
        if int(data.get("source_size") or -1) != src_size:
            return False
        if int(data.get("source_mtime") or -1) != src_mtime:
            return False
        return True

    for candidate_path in (cache_path, legacy_cache):
        try:
            if not os.path.isfile(candidate_path):
                continue
            with open(candidate_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "chunk_ids_by_provnum" not in data:
                continue
            cached_cs = data.get("chunksize")
            if cached_cs is not None and int(cached_cs) != int(chunksize):
                continue
            if _source_matches(data):
                if "source_size" not in data and src_size is not None:
                    data = dict(data)
                    data["source_size"] = src_size
                    data["source_mtime"] = src_mtime
                    try:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(data, f)
                    except Exception:
                        pass
                return data
        except Exception:
            continue

    built = _build_provnum_chunk_index(csv_path, chunksize=chunksize)
    if src_size is not None:
        built["source_size"] = src_size
    if src_mtime is not None:
        built["source_mtime"] = src_mtime
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(built, f)
    except Exception:
        # Cache write is best-effort; extraction can continue without it.
        pass
    return built


def select_targeted_chunk_ids(
    index_data: dict[str, Any],
    provnum: str,
    neighbor_margin: int = 1,
) -> list[int]:
    """
    Return 1-based chunk ids likely to contain the facility rows.

    Includes +/- neighbor chunks to guard against boundary-adjacent rows.
    """
    if not isinstance(index_data, dict):
        return []
    mapping = index_data.get("chunk_ids_by_provnum") or {}
    if not isinstance(mapping, dict):
        return []

    target: set[int] = set()
    for v in provnum_search_variants(provnum):
        k = _normalize_provnum_like(v)
        ids = mapping.get(k) or []
        for cid in ids:
            try:
                target.add(int(cid))
            except Exception:
                continue

    if not target:
        return []

    margin = max(0, int(neighbor_margin))
    if margin > 0:
        max_chunk = int(index_data.get("chunk_count") or 0)
        expanded: set[int] = set()
        for cid in target:
            for cand in range(cid - margin, cid + margin + 1):
                if cand < 1:
                    continue
                if max_chunk and cand > max_chunk:
                    continue
                expanded.add(cand)
        target = expanded

    return sorted(target)


def normalize_cy_qtr(val: Any) -> Optional[str]:
    """Normalize CY_Qtr to canonical 'CYyyyyQn' for comparison with PBJ filenames."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    s = str(val).strip().upper().replace("\ufeff", "")
    m = re.search(r"(?:CY)?(\d{4})Q([1-4])", s)
    if m:
        return f"CY{m.group(1)}Q{m.group(2)}"
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        i = int(val)
        if 20101 <= i <= 20304 and (i % 10) in (1, 2, 3, 4):
            y, q = str(i // 10), str(i % 10)
            return f"CY{y}Q{q}"
    return None


def quarter_from_pbj_staffing_filename(path: str) -> Optional[str]:
    """Parse quarter from PBJ daily staffing file name, e.g. PBJ_dailynonnursestaffing_CY2025Q3.csv -> CY2025Q3."""
    m = re.search(r"CY(\d{4})Q([1-4])", os.path.basename(path), re.IGNORECASE)
    return f"CY{m.group(1)}Q{m.group(2)}" if m else None


def parse_pbj_work_date_series(work_date: pd.Series) -> pd.Series:
    """
    Coerce PBJ WorkDate column to timezone-naive datetimes (invalid -> NaT).
    Accepts YYYYMMDD strings or integers, ISO dates (YYYY-MM-DD) from facility extracts,
    and other strings pandas can parse.
    """
    s = work_date
    if s.dtype == "object":
        s_str = s.astype(str).str.strip()
        parsed = pd.to_datetime(s_str, format="%Y%m%d", errors="coerce", utc=False)
        nat = parsed.isna() & s_str.ne("") & s_str.str.lower().ne("nan")
        if bool(nat.any()):
            alt = pd.to_datetime(s_str[nat], errors="coerce", utc=False)
            parsed = parsed.copy()
            parsed.loc[nat] = alt.values
        return parsed
    if str(s.dtype) in ("int64", "int32", "float64", "float32"):
        return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce", utc=False)
    return pd.to_datetime(s, errors="coerce", utc=False)


def normalize_cy_qtr_column(df: pd.DataFrame, col: str = "CY_Qtr") -> pd.Series:
    """Return a Series of canonical CYyyyyQn strings (NaN where unparseable)."""
    if col not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
    return cast(pd.Series, df[col].map(normalize_cy_qtr))


def iso_date_or_none(ts: Any) -> Optional[str]:
    """Format a Timestamp-like value as YYYY-MM-DD, or None if missing/invalid."""
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    t = pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts
    if pd.isna(t):
        return None
    try:
        return t.strftime("%Y-%m-%d")
    except (ValueError, OSError):
        return None


def parse_citation_date_series(col: pd.Series) -> pd.Series:
    """Parse CMS citation date columns (mixed formats, Excel serials); naive, no UTC shift."""
    return pd.to_datetime(col, errors="coerce", utc=False)
