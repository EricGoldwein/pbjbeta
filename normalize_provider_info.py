#!/usr/bin/env python3
"""Normalize CMS NH_ProviderInfo_*.csv snapshots to ProviderInfoNorm_YYYY_MM.csv.

Used by run_pipeline_update.py (providerinfo step). Output columns match the
ProviderInfoNorm schema consumed by pbj-root / provider dashboards.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

import cms_data_paths
from nh_provider_column_map import NH_TO_NORM
from prov_info_quarter_map import get_quarter_from_processing_month

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def is_git_lfs_pointer(path: Path) -> bool:
    """True if path is a Git LFS pointer stub, not real CSV bytes."""
    try:
        with path.open("rb") as f:
            head = f.read(128)
    except OSError:
        return False
    return head.startswith(LFS_POINTER_PREFIX)


class LfsPointerError(ValueError):
    """Raised when a Git LFS pointer is passed where real CMS data is required."""


def parse_month_year_from_filename(filename: str) -> tuple[int, int] | None:
    match = re.search(r"([A-Za-z]{3})(\d{4})", filename)
    if not match:
        return None
    month = MONTH_MAP.get(match.group(1).lower()[:3])
    if not month:
        return None
    return int(match.group(2)), month


def _template_columns(output_dir: Path) -> list[str]:
    existing = sorted(output_dir.glob("ProviderInfoNorm_*.csv"))
    if existing:
        return list(pd.read_csv(existing[-1], nrows=0).columns)
    return list(NH_TO_NORM.keys()) + ["quarter"]


def normalize_nh_file(nh_path: Path, output_path: Path, template_cols: list[str]) -> int:
    if is_git_lfs_pointer(nh_path):
        raise LfsPointerError(
            f"refusing to normalize Git LFS pointer (not real CMS data): {nh_path}"
        )
    if not nh_path.is_file() or nh_path.stat().st_size == 0:
        raise ValueError(f"raw Provider Info CSV missing or empty: {nh_path}")
    nh = pd.read_csv(nh_path, dtype=str, low_memory=False)
    nh.columns = [str(c).replace("\ufeff", "").strip() for c in nh.columns]
    out: dict[str, pd.Series] = {}
    for norm_col, nh_col in NH_TO_NORM.items():
        if nh_col in nh.columns:
            out[norm_col] = nh[nh_col]
        else:
            out[norm_col] = pd.Series([pd.NA] * len(nh), dtype="object")
    df = pd.DataFrame(out)
    if "ccn" in df.columns:
        df["ccn"] = (
            df["ccn"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
        )
    if "processing_date" in df.columns:
        proc = pd.to_datetime(df["processing_date"], errors="coerce")
        df["processing_date"] = proc.dt.strftime("%Y-%m-%d")
        df["quarter"] = proc.dt.strftime("%Y-%m").map(get_quarter_from_processing_month)
    else:
        df["quarter"] = pd.NA
    for col in template_cols:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[template_cols]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    cmi_ok = 0
    nh_cmi_ok = 0
    if "nursing_case_mix_index" in df.columns:
        s = pd.to_numeric(df["nursing_case_mix_index"], errors="coerce")
        cmi_ok = int(s.notna().sum())
    nh_cmi_col = NH_TO_NORM.get("nursing_case_mix_index")
    if nh_cmi_col and nh_cmi_col in nh.columns:
        nh_cmi_ok = int(pd.to_numeric(nh[nh_cmi_col], errors="coerce").notna().sum())
    print(f"Wrote {output_path} ({len(df):,} rows, {cmi_ok:,} with nursing_case_mix_index)")
    if nh_cmi_ok and cmi_ok < int(nh_cmi_ok * 0.9):
        print(
            f"ERROR: nursing_case_mix_index under-filled ({cmi_ok:,} vs {nh_cmi_ok:,} in NH); "
            "check NH_TO_NORM column map",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return len(df)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Normalize NH_ProviderInfo snapshots")
    parser.add_argument("--force", action="store_true", help="Rebuild even if output exists")
    parser.add_argument("--file", type=str, default="", help="Single NH_ProviderInfo_*.csv basename")
    args = parser.parse_args(argv)

    input_dir = cms_data_paths.provider_info_dir()
    output_dir = cms_data_paths.provider_info_normalized_dir()
    if not input_dir.is_dir():
        print(f"Missing {input_dir}", file=sys.stderr)
        return 1
    template_cols = _template_columns(output_dir)
    paths = sorted(input_dir.glob("NH_ProviderInfo_*.csv"))
    if args.file:
        needle = args.file.lower()
        paths = [p for p in paths if needle in p.name.lower()]
    if not paths:
        print("No NH_ProviderInfo_*.csv files found", file=sys.stderr)
        return 1
    wrote = 0
    for nh_path in paths:
        date_info = parse_month_year_from_filename(nh_path.name)
        if not date_info:
            print(f"Skip (unparsed filename): {nh_path.name}")
            continue
        if is_git_lfs_pointer(nh_path):
            print(
                f"ERROR: refusing Git LFS pointer (not real CMS data): {nh_path.name}",
                file=sys.stderr,
            )
            return 1
        year, month = date_info
        out_path = output_dir / f"ProviderInfoNorm_{year}_{month:02d}.csv"
        if out_path.exists() and not args.force:
            print(f"Skip existing: {out_path}")
            continue
        try:
            normalize_nh_file(nh_path, out_path, template_cols)
        except LfsPointerError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        wrote += 1
    if wrote == 0:
        print("No new provider info files normalized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
