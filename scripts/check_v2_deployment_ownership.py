#!/usr/bin/env python3
"""
Preflight: facility Vercel bundles must ship NH ownership slice when the entrypoint
exposes /api/ownership-contacts.

Root cause (2026-06): 315128 deployed without facility_{CCN}_nh_ownership.csv while the
UI called the API — empty rows were shown as "no disclosures" instead of a bundle gap.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _entrypoint(deploy_dir: Path) -> Path | None:
    matches = sorted(deploy_dir.glob("facility_*_superdynamic_dashboard.py"))
    return matches[0] if matches else None


def run_check(deploy_dir: Path, provnum: str) -> tuple[bool, list[str]]:
    errors: list[str] = []
    deploy_dir = deploy_dir.resolve()
    ccn = str(provnum).strip().zfill(6)
    entry = _entrypoint(deploy_dir)
    if not entry or not entry.is_file():
        errors.append(f"No facility entrypoint in {deploy_dir}")
        return False, errors

    text = entry.read_text(encoding="utf-8", errors="replace")
    if "/api/ownership-contacts" not in text and "ownership-contacts" not in text:
        return True, []

    slice_path = deploy_dir / "ownership" / f"facility_{ccn}_nh_ownership.csv"
    if not slice_path.is_file():
        errors.append(
            f"Missing ownership slice: ownership/facility_{ccn}_nh_ownership.csv "
            "(required when entrypoint serves /api/ownership-contacts)"
        )
        return False, errors

    try:
        with slice_path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError as exc:
        errors.append(f"Cannot read ownership slice: {exc}")
        return False, errors

    if not rows:
        errors.append(f"Ownership slice is empty: {slice_path.name}")
        return False, errors

    return True, []


def main() -> int:
    parser = argparse.ArgumentParser(description="Check NH ownership slice in Vercel bundle.")
    parser.add_argument("provnum", help="Six-digit CCN")
    parser.add_argument(
        "--deploy-dir",
        type=Path,
        default=None,
        help="Deployment folder (default: deployments/pbj320-<CCN>)",
    )
    args = parser.parse_args()
    ccn = str(args.provnum).strip().zfill(6)
    deploy_dir = args.deploy_dir or (_ROOT / "deployments" / f"pbj320-{ccn}")
    ok, errors = run_check(deploy_dir, ccn)
    if ok:
        slice_path = deploy_dir / "ownership" / f"facility_{ccn}_nh_ownership.csv"
        row_count = 0
        if slice_path.is_file():
            with slice_path.open(newline="", encoding="utf-8") as f:
                row_count = sum(1 for _ in csv.DictReader(f))
        print(f"PASS: ownership slice OK ({slice_path.name}, {row_count} rows)")
        return 0
    for err in errors:
        print(f"FAIL: {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
