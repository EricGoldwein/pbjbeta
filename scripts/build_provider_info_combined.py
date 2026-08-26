#!/usr/bin/env python3
"""Deterministically build provider_info_combined.csv from ProviderInfoNorm snapshots.

Semantics
---------
* One row per (processing_date, ccn) across all ``provider_info_normalized/ProviderInfoNorm_*.csv``.
* CCNs are six-character zero-padded strings.
* Sorted by processing_date then ccn; duplicates keep the last row (same rule as facility slice builders).
* This matches the historical combined artifact produced incrementally from monthly Norm files.

Usage
-----
  python scripts/build_provider_info_combined.py --verify-only
  python scripts/build_provider_info_combined.py --output provider_info_combined.csv
  python scripts/build_provider_info_combined.py --output provider_info_combined.csv --through 2026-05
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cms_data_paths


def _norm_key(path: Path) -> str | None:
    m = re.search(r"ProviderInfoNorm_(\d{4})_(\d{2})\.csv$", path.name)
    if not m:
        return None
    return f"{m.group(1)}-{m.group(2)}"


def _norm_paths(through: str | None = None) -> list[Path]:
    norm_dir = cms_data_paths.provider_info_normalized_dir()
    paths = sorted(norm_dir.glob("ProviderInfoNorm_*.csv"))
    if not through:
        return paths
    out: list[Path] = []
    for p in paths:
        key = _norm_key(p)
        if key and key <= through:
            out.append(p)
    return out


def validate_norm_inputs(paths: list[Path], *, require_month: str | None = None) -> None:
    """Fail fast on schema drift, within-file duplicate keys, or missing required month."""
    if not paths:
        raise FileNotFoundError("no ProviderInfoNorm_*.csv files found")
    if require_month:
        keys = {_norm_key(p) for p in paths}
        if require_month not in keys:
            raise FileNotFoundError(
                f"required normalized month missing: ProviderInfoNorm_{require_month.replace('-', '_')}.csv"
            )
    template_cols: list[str] | None = None
    for path in paths:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        cols = list(df.columns)
        if template_cols is None:
            template_cols = cols
        elif cols != template_cols:
            missing = set(template_cols) - set(cols)
            extra = set(cols) - set(template_cols)
            raise ValueError(
                f"column mismatch in {path.name}: missing={sorted(missing)[:5]} extra={sorted(extra)[:5]}"
            )
        if "ccn" in df.columns and "processing_date" in df.columns:
            sub = df[["processing_date", "ccn"]].copy()
            sub["ccn"] = sub["ccn"].astype(str).str.zfill(6)
            dup = int(sub.duplicated(subset=["processing_date", "ccn"]).sum())
            if dup:
                raise ValueError(
                    f"duplicate (processing_date, ccn) rows in {path.name}: {dup} — fix Norm before combine"
                )


def build_provider_info_combined(
    *,
    through: str | None = None,
    norm_paths: list[Path] | None = None,
    require_month: str | None = None,
) -> pd.DataFrame:
    paths = norm_paths if norm_paths is not None else _norm_paths(through)
    validate_norm_inputs(paths, require_month=require_month)
    template_cols: list[str] | None = None
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        if template_cols is None:
            template_cols = list(df.columns)
        frames.append(df[template_cols])
    combined = pd.concat(frames, ignore_index=True)
    if "ccn" in combined.columns:
        combined["ccn"] = (
            combined["ccn"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
        )
    if "processing_date" in combined.columns:
        combined["processing_date"] = pd.to_datetime(combined["processing_date"], errors="coerce").dt.strftime(
            "%Y-%m-%d"
        )
    combined = combined.sort_values([c for c in ("processing_date", "ccn") if c in combined.columns])
    dedup_cols = [c for c in ("processing_date", "ccn") if c in combined.columns]
    if dedup_cols:
        combined = combined.drop_duplicates(subset=dedup_cols, keep="last")
    return combined


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_against_existing(
    rebuilt: pd.DataFrame,
    existing_path: Path,
    *,
    through: str | None = None,
) -> dict[str, object]:
    if not existing_path.is_file():
        return {"status": "no_existing_file"}
    existing = pd.read_csv(
        existing_path,
        usecols=["ccn", "processing_date"],
        dtype=str,
        low_memory=False,
    )
    existing["ccn"] = existing["ccn"].astype(str).str.zfill(6)
    existing["processing_date"] = pd.to_datetime(existing["processing_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    if through:
        year_s, month_s = through.split("-", 1)
        cap = pd.Timestamp(int(year_s), int(month_s), 1) + pd.offsets.MonthEnd(0)
        cap_str = cap.strftime("%Y-%m-%d")
        existing = existing[existing["processing_date"] <= cap_str]
    rebuilt_keys = set(zip(rebuilt["processing_date"], rebuilt["ccn"]))
    existing_keys = set(zip(existing["processing_date"], existing["ccn"]))
    return {
        "status": "ok" if rebuilt_keys == existing_keys else "mismatch",
        "rebuilt_rows": len(rebuilt),
        "existing_rows": len(existing),
        "only_rebuilt": len(rebuilt_keys - existing_keys),
        "only_existing": len(existing_keys - rebuilt_keys),
        "rebuilt_max_processing_date": rebuilt["processing_date"].max(),
        "existing_max_processing_date": existing["processing_date"].max(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build provider_info_combined.csv from Norm snapshots")
    parser.add_argument("--output", type=str, default="", help="Write combined CSV (default: stdout stats only)")
    parser.add_argument("--through", type=str, default="", help="Include Norm files through YYYY-MM (e.g. 2026-05)")
    parser.add_argument(
        "--verify-existing",
        type=str,
        default="provider_info_combined.csv",
        help="Compare rebuild (through --through) to this file before writing",
    )
    parser.add_argument("--verify-only", action="store_true", help="Rebuild through --through and verify only")
    parser.add_argument(
        "--require-month",
        type=str,
        default="",
        help="Fail if ProviderInfoNorm for YYYY-MM is missing (e.g. 2026-06)",
    )
    args = parser.parse_args()

    through = args.through.strip() or None
    require_month = args.require_month.strip() or None
    rebuilt = build_provider_info_combined(through=through, require_month=require_month)
    print(f"Rebuilt {len(rebuilt):,} rows from {len(_norm_paths(through))} Norm file(s)")
    if "ccn" in rebuilt.columns:
        print(f"  unique CCNs: {rebuilt['ccn'].nunique():,}")
    if "processing_date" in rebuilt.columns:
        print(f"  processing_date range: {rebuilt['processing_date'].min()} .. {rebuilt['processing_date'].max()}")

    should_verify = args.verify_only or bool(through)
    if should_verify and args.verify_existing:
        report = verify_against_existing(rebuilt, Path(args.verify_existing), through=through)
        print(f"Verify vs {args.verify_existing}: {report}")
        if report.get("status") == "mismatch":
            print("ERROR: rebuild does not match existing combined file", file=sys.stderr)
            return 1

    if args.verify_only:
        return 0

    if not args.output:
        print("No --output given; nothing written.")
        return 0

    out = Path(args.output)
    rebuilt.to_csv(out, index=False)
    print(f"Wrote {out} ({out.stat().st_size:,} bytes, sha256={_sha256_file(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
