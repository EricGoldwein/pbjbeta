#!/usr/bin/env python3
"""Operator check for Provider Information → stars/charts.

One command covers the connected path:

  extract intervals → shipped quarter map → facility history slice → pytest proof

Examples (PowerShell, from pbj-data-ops):

  python scripts/check_provider_quarter_flow.py
  python scripts/check_provider_quarter_flow.py --ccn 365865
  python scripts/check_provider_quarter_flow.py --sync
  python scripts/check_provider_quarter_flow.py --prove --ccn 365865
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_OPS = Path(__file__).resolve().parents[1]
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))

from provider_quarter_mapping import (  # noqa: E402
    operator_quarter_flow,
    sync_interval_mapping_from_extract,
)

DATA_OPS_PROVE = (
    "tests/test_provider_quarter_mapping.py",
    "tests/test_cms_provider_release.py",
    "tests/test_data_ops_dashboard.py",
)
PBJAPP_PROVE = (
    "tests/test_provider_quarter_match_not_csv_label.py",
    "tests/test_provider_slice_history_rebuild.py",
)


def _pbjapp_root() -> Path | None:
    env = (os.environ.get("PBJ_REPO_ROOT") or "").strip().strip('"')
    if env:
        path = Path(env).expanduser().resolve()
        if (path / "create_vercel_deployment.py").is_file():
            return path
    sibling = _OPS.parent / "PBJapp"
    if (sibling / "create_vercel_deployment.py").is_file():
        return sibling
    return None


def _run_pytest(cwd: Path, files: tuple[str, ...]) -> int:
    cmd = [sys.executable, "-m", "pytest", *files, "--cache-clear", "-v", "--tb=short"]
    print(f"\n--- pytest in {cwd} ---")
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ccn", default="", help="Facility CCN to check history slice")
    parser.add_argument("--sync", action="store_true", help="Write extracted month into shipped quarter JSON")
    parser.add_argument(
        "--prove",
        action="store_true",
        help="Run the connected pytest files with a cleared cache (operator proof)",
    )
    args = parser.parse_args()
    ccn = str(args.ccn or "").strip() or None

    print("Provider Information → stars/charts")
    print("1. Extract intervals (Sources → Provider Information → Acquire / Process)")
    print("2. Apply quarter map (Needs attention, or --sync here)")
    print("3. Rebuild local dashboard (Dashboard Builder → Build)")
    print("4. Prove: python scripts/check_provider_quarter_flow.py --prove")
    print()

    if args.sync:
        result = sync_interval_mapping_from_extract()
        print("SYNC:", result.get("detail") or result)
        if not result.get("ok"):
            return 1

    flow = operator_quarter_flow(ccn=ccn)
    print(f"ACTIVE: {flow.get('release_id') or '—'} → {flow.get('resolved_quarter') or 'unmapped'}")
    print("MAP:", "OK" if not flow.get("needs_attention") else flow.get("detail"))
    if ccn:
        print(f"{ccn} HISTORY:", flow["history"].get("detail"))
    print("NEXT:", flow.get("next_step"))
    print("PROVE:", flow.get("prove_command"))

    status = 0 if flow.get("ready") or not ccn else 1
    if flow.get("needs_attention"):
        status = 1

    if args.prove:
        ops_rc = _run_pytest(_OPS, DATA_OPS_PROVE)
        app_root = _pbjapp_root()
        app_rc = 0
        if app_root is None:
            print("PBJapp root not found (set PBJ_REPO_ROOT); skipped PBJapp prove tests.")
            app_rc = 1
        else:
            app_rc = _run_pytest(app_root, PBJAPP_PROVE)
        if ops_rc or app_rc:
            return 1
        print("\nPROVE PASS")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
