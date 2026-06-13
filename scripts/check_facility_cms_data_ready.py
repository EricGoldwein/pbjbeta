#!/usr/bin/env python3
"""
Pre-package check: national CMS quarters vs facility deploy slices.

Used before ``create_vercel_deployment.py`` / v2 deploy so superdynamic bundles
do not ship with stale nurse, non-nurse, or EIN data.

Exit 0 = slices match national sources (or only known-empty gaps).
Exit 1 = pending quarter(s) — run packaging to refresh slices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cms_data_paths import facility_deploy_dir  # noqa: E402
from file_path_utils import (  # noqa: E402
    find_facility_complete_data,
    find_facility_nonnurse_daily,
)
from packaging_refresh_gates import (  # noqa: E402
    gate_ein_slice,
    gate_nonnurse_pbj_slice,
    gate_nurse_pbj_slice,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check facility slices vs national CMS quarters")
    parser.add_argument("ccn", help="6-digit CCN")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if any slice needs refresh (default: warn only for EIN)",
    )
    args = parser.parse_args()
    ccn = str(args.ccn).strip().zfill(6)
    root = str(_ROOT)

    nurse_csv = find_facility_complete_data(ccn)
    nn_csv = find_facility_nonnurse_daily(ccn)
    deploy = facility_deploy_dir(ccn, _ROOT)

    issues: list[str] = []
    ok_lines: list[str] = []

    if nurse_csv:
        need, why = gate_nurse_pbj_slice(root, nurse_csv)
        if need:
            issues.append(f"nurse: {why}")
        else:
            ok_lines.append(f"nurse: {why}")
    else:
        issues.append("nurse: facility complete_data CSV missing")

    if nn_csv:
        need, why = gate_nonnurse_pbj_slice(root, nn_csv)
        if need:
            issues.append(f"non-nurse: {why}")
        else:
            ok_lines.append(f"non-nurse: {why}")
    else:
        issues.append("non-nurse: facility nonnurse_daily CSV missing")

    need_ein, ein_why = gate_ein_slice(root, ccn)
    if need_ein:
        issues.append(f"EIN: {ein_why}")
    else:
        ok_lines.append(f"EIN: {ein_why}")

    print(f"=== CMS data readiness: CCN {ccn} ===")
    print(f"Deploy dir: {deploy}")
    for line in ok_lines:
        print(f"  [OK] {line}")
    for line in issues:
        print(f"  [PENDING] {line}")

    if not issues:
        print("\nReady — national sources and deploy slices are aligned.")
        return 0
    print("\nRun packaging to refresh:")
    print(f"  python create_vercel_deployment.py {ccn}")
    if args.strict or any("nurse" in i or "non-nurse" in i for i in issues):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
