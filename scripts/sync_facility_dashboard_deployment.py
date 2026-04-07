#!/usr/bin/env python3
"""
Sync a facility dashboard under deployments/pbj320-<CCN>/ from the canonical sources.

Copies:
  - dynamic_facility_dashboard.py -> facility_<CCN>_flask_app.py (sets PROVNUM)
  - templates/dynamic_facility_dashboard.html
  - pbj_favicon.png
  - Shared Python modules used by Vercel-style bundles (keeps folder self-consistent)

The Flask app prepends the repo root to sys.path when run from deployments/pbj320-<CCN>/,
so stale local copies of facility_ein_*.py cannot shadow the repo. Fresh copies are still
written here for offline zips and parity with create_vercel_deployment.py.

Usage:
  python scripts/sync_facility_dashboard_deployment.py 395052
  python scripts/sync_facility_dashboard_deployment.py 395052 --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync deployments/pbj320-<CCN>/ from canonical dashboard.")
    parser.add_argument(
        "provnum",
        help="Six-digit CCN (e.g. 395052)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    args = parser.parse_args()
    prov = str(args.provnum).strip().zfill(6)
    if not prov.isdigit() or len(prov) != 6:
        print("ERROR: provnum must be a 6-digit CCN.", file=sys.stderr)
        return 2

    root = _project_root()
    dest_dir = os.path.join(root, "deployments", f"pbj320-{prov}")
    template_src = os.path.join(root, "templates", "dynamic_facility_dashboard.html")
    dashboard_src = os.path.join(root, "dynamic_facility_dashboard.py")
    favicon_src = os.path.join(root, "pbj_favicon.png")
    out_py = os.path.join(dest_dir, f"facility_{prov}_flask_app.py")
    out_template_dir = os.path.join(dest_dir, "templates")
    out_template = os.path.join(out_template_dir, "dynamic_facility_dashboard.html")
    out_favicon = os.path.join(dest_dir, "pbj_favicon.png")

    extra_pairs: list[tuple[str, str]] = []
    for name in (
        "facility_ein_lib.py",
        "facility_ein_employee_analytics.py",
        "facility_report_lib.py",
        "file_path_utils.py",
    ):
        src = os.path.join(root, name)
        if os.path.isfile(src):
            extra_pairs.append((src, os.path.join(dest_dir, name)))

    plan: list[tuple[str, str]] = [
        (dashboard_src, out_py),
        (template_src, out_template),
        (favicon_src, out_favicon),
    ]
    plan.extend(extra_pairs)

    for src, dst in plan:
        if not os.path.isfile(src):
            print(f"ERROR: missing source: {src}", file=sys.stderr)
            return 1

    if args.dry_run:
        print(f"Would create: {dest_dir}")
        for src, dst in plan:
            print(f"  {src} -> {dst}")
        return 0

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(out_template_dir, exist_ok=True)

    for src, dst in plan:
        if src == dashboard_src:
            text = open(src, encoding="utf-8").read()
            text, n = re.subn(
                r'^PROVNUM\s*=\s*"\d{6}"\s*$',
                f'PROVNUM = "{prov}"',
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if n != 1:
                print("ERROR: could not patch PROVNUM in dynamic_facility_dashboard.py copy.", file=sys.stderr)
                return 1
            text, n2 = re.subn(
                r"^# Hardcoded for facility \d{6}\s*$",
                f"# Hardcoded for facility {prov}",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if n2 == 0:
                # tolerate missing comment line
                pass
            with open(dst, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            print(f"Wrote {dst} (PROVNUM={prov})")
        else:
            shutil.copy2(src, dst)
            print(f"Copied -> {dst}")

    print("\nDone. Run:")
    print(f"  python {out_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
