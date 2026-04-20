#!/usr/bin/env python3
"""
Sync a facility dashboard under deployments/pbj320-<CCN>/ from the canonical sources.

Copies:
  - dynamic_facility_dashboard.py -> facility_<CCN>_flask_app.py (sets PROVNUM)
  - templates/dynamic_facility_dashboard.html
  - pbj_lite/geo_nursing_cmi_quarterly.csv (when present; state/region/national nursing CMI rollups)
  - pbj_favicon.png
  - Shared Python modules used by Vercel-style bundles (keeps folder self-consistent)

Canonical dashboard + template include the geographic rollup flow:
  - lite_hprd_rollup_series_by_quarter() and geo_rollup_series passed into render_template
  - pbj320-export-page JSON: geoRollupSeries, geoStateDashboardUrl, geoStateAbbr, geoStateLong
  - Client pbjGeoRollupRefresh() after each /api/data filter (see template)

The Flask app prepends the repo root to sys.path when run from deployments/pbj320-<CCN>/,
so stale local copies of facility_ein_*.py cannot shadow the repo. Fresh copies are still
written here for offline zips and parity with create_vercel_deployment.py (which reads the
same dynamic_facility_dashboard.py).

Usage:
  python scripts/sync_facility_dashboard_deployment.py 395052
  python scripts/sync_facility_dashboard_deployment.py 395052 --dry-run
  python scripts/sync_facility_dashboard_deployment.py --all
  python scripts/sync_facility_dashboard_deployment.py --all --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _list_deployment_ccns(root: str) -> list[str]:
    """CCNs for deployments/pbj320-<CCN>/ that contain facility_<CCN>_flask_app.py."""
    dep_root = os.path.join(root, "deployments")
    if not os.path.isdir(dep_root):
        return []
    out: list[str] = []
    for name in os.listdir(dep_root):
        m = re.match(r"^pbj320-(\d{6})$", name, flags=re.I)
        if not m:
            continue
        ccn = m.group(1)
        flask_path = os.path.join(dep_root, name, f"facility_{ccn}_flask_app.py")
        if os.path.isfile(flask_path):
            out.append(ccn)
    return sorted(out)


def sync_facility_dashboard(root: str, prov: str, dry_run: bool) -> int:
    """
    Sync one facility. Returns 0 on success, 1 on I/O or patch error, 2 if prov is not a 6-digit CCN.
    """
    prov = str(prov).strip().zfill(6)
    if not prov.isdigit() or len(prov) != 6:
        print(f"ERROR: invalid CCN: {prov!r}", file=sys.stderr)
        return 2

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

    pbj_id = os.path.join(root, "pbj_identifiers")
    if os.path.isdir(pbj_id):
        for entry in sorted(os.listdir(pbj_id)):
            if entry == "__pycache__" or not entry.endswith(".py"):
                continue
            src = os.path.join(pbj_id, entry)
            if os.path.isfile(src):
                extra_pairs.append((src, os.path.join(dest_dir, "pbj_identifiers", entry)))

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

    if dry_run:
        print(f"Would sync: {dest_dir}")
        for src, dst in plan:
            print(f"  {src} -> {dst}")
        return 0

    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(out_template_dir, exist_ok=True)
    os.makedirs(os.path.join(dest_dir, "pbj_identifiers"), exist_ok=True)

    for src, dst in plan:
        if src == dashboard_src:
            text = open(src, encoding="utf-8").read()
            text, n = re.subn(
                r"^PROVNUM\s*=.*$",
                f'PROVNUM = "{prov}"',
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if n != 1:
                print(
                    "ERROR: could not patch PROVNUM in dynamic_facility_dashboard.py copy "
                    "(no top-level PROVNUM assignment found).",
                    file=sys.stderr,
                )
                return 1
            text, n2 = re.subn(
                r"^# Hardcoded for facility \d{6}\s*$",
                f"# Hardcoded for facility {prov}",
                text,
                count=1,
                flags=re.MULTILINE,
            )
            if n2 == 0:
                pass
            with open(dst, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            print(f"Wrote {dst} (PROVNUM={prov})")
        else:
            shutil.copy2(src, dst)
            print(f"Copied -> {dst}")

    geo_src = os.path.join(root, "pbj_lite", "geo_nursing_cmi_quarterly.csv")
    if os.path.isfile(geo_src):
        geo_dst = os.path.join(dest_dir, "geo_nursing_cmi_quarterly.csv")
        if dry_run:
            print(f"  Would copy {geo_src} -> {geo_dst}")
        else:
            shutil.copy2(geo_src, geo_dst)
            print(f"Copied -> {geo_dst}")

    print(f"Done {prov}. Run: python {out_py}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync deployments/pbj320-<CCN>/ from canonical dynamic_facility_dashboard.py + template."
    )
    parser.add_argument(
        "provnum",
        nargs="?",
        default=None,
        help="Six-digit CCN (e.g. 395052). Omit when using --all.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="all_deployments",
        help="Sync every deployments/pbj320-<CCN>/ that already has facility_<CCN>_flask_app.py",
    )
    args = parser.parse_args()
    root = _project_root()

    if args.all_deployments:
        if args.provnum:
            print("ERROR: do not pass provnum together with --all.", file=sys.stderr)
            return 2
        ccns = _list_deployment_ccns(root)
        if not ccns:
            print("No deployments/pbj320-<CCN>/ folders with facility_<CCN>_flask_app.py found.", file=sys.stderr)
            return 1
        print(f"--all: syncing {len(ccns)} facility package(s)\n")
        for i, ccn in enumerate(ccns, start=1):
            print(f"--- [{i}/{len(ccns)}] CCN {ccn} ---")
            rc = sync_facility_dashboard(root, ccn, args.dry_run)
            if rc != 0:
                return rc
        print("\nAll syncs finished.")
        return 0

    if not args.provnum:
        print("ERROR: provnum is required unless you pass --all.", file=sys.stderr)
        return 2

    return sync_facility_dashboard(root, args.provnum, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
