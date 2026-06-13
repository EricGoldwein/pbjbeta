#!/usr/bin/env python3
"""
Run a facility dashboard for any 6-digit CCN (internal Flask tool).

Defaults to Employee Detail (EIN) with all quarters present in the facility files.

Preferred local entry point (loads data via ``dynamic_facility_dashboard`` + repo template):

  python run_facility_dashboard.py <PROVNUM>

To run the frozen bundle under ``deployments/pbj320-<CCN>/`` (parity with Vercel cwd / copied app):

  python run_facility_dashboard.py <PROVNUM> --deployment-bundle

Do not run ``python facility_<CCN>_flask_app.py`` from the repo root — that file lives only under
``deployments/pbj320-<CCN>/`` unless you copy it.

Usage:
  python run_facility_dashboard.py <PROVNUM>
  python run_facility_dashboard.py 015009 --ein-mode none
  python run_facility_dashboard.py 015009 --ein-mode selected --ein-quarters CY2024Q1,CY2024Q2
  python run_facility_dashboard.py 335581 --deployment-bundle
  python run_facility_dashboard.py 315461 --v2
"""

import argparse
import os
import subprocess
import sys

from dynamic_facility_dashboard import run_dashboard
from file_path_utils import find_facility_complete_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local facility Flask dashboard.")
    parser.add_argument("provnum", help="6-digit CCN (provider number)")
    parser.add_argument("--port", type=int, default=5000, help="Flask port (default 5000)")
    parser.add_argument(
        "--ein-mode",
        choices=("all", "selected", "none"),
        default="all",
        help="Employee Detail: all quarters (default), selected subset, or off",
    )
    parser.add_argument(
        "--ein-quarters",
        default="",
        help="Comma-separated CY quarters for --ein-mode selected (e.g. CY2024Q1,CY2024Q2)",
    )
    parser.add_argument(
        "--deployment-bundle",
        action="store_true",
        help="Run deployments/pbj320-<CCN>/facility_*_flask_app.py with that folder as cwd (Vercel-style bundle).",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Render superdynamic_dashboard_v2.html (local V2 UI smoke; V1 APIs only).",
    )
    args = parser.parse_args()

    provnum = str(args.provnum).strip().zfill(6)
    
    if not provnum.isdigit() or len(provnum) != 6:
        print("❌ Please enter a valid 6-digit CCN (e.g., 015009)")
        sys.exit(1)
    
    print(f"🏥 Starting Dashboard for Facility {provnum}")
    print("=" * 50)

    if args.deployment_bundle:
        flask_path = find_facility_flask_app(provnum)
        if not flask_path or not os.path.isfile(flask_path):
            print(
                f"❌ No deployment Flask app found for {provnum}. "
                f"Expected: deployments/pbj320-{provnum}/facility_{provnum}_flask_app.py"
            )
            sys.exit(1)
        bundle_dir = os.path.dirname(os.path.abspath(flask_path))
        complete = find_facility_complete_data(provnum)
        if not complete or not os.path.isfile(complete):
            print(
                "⚠️  facility_*_complete_data.csv not found in standard locations; "
                "the bundle may fail to load until CSVs exist under the deployment folder."
            )
        else:
            print(f"✅ Found complete data: {complete}")
        print(f"🚀 Running deployment bundle: {flask_path}")
        print(f"   (working directory: {bundle_dir})")
        rc = subprocess.run(
            [sys.executable, os.path.basename(flask_path)],
            cwd=bundle_dir,
        ).returncode
        raise SystemExit(rc)
    
    # Check if CSV exists anywhere we normally store it; if not, create it
    existing_csv = find_facility_complete_data(provnum)
    if existing_csv and os.path.exists(existing_csv):
        print(f"✅ Found existing CSV file: {existing_csv}")
    else:
        csv_filename = f'facility_{provnum}_complete_data.csv'
        print(f"📊 CSV file not found. Creating {csv_filename}...")
        print("This may take a few minutes...")

        # Run the CSV generator (creates root CSV and/or facility_pbj/ copy)
        result = subprocess.run(
            [sys.executable, 'create_facility_csv.py', provnum],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ Error creating CSV: {result.stderr}")
            sys.exit(1)

        print("✅ CSV file created successfully!")
    
    # Create and run the dashboard
    print(f"\n🚀 Starting dashboard for facility {provnum}...")

    if args.v2:
        os.environ["PBJ_SUPERDYNAMIC_TEMPLATE"] = "v2"

    ein_quarters = [q.strip() for q in args.ein_quarters.split(",") if q.strip()]
    run_dashboard(
        provnum,
        port=args.port,
        ein_mode=args.ein_mode,
        ein_selected_quarters=ein_quarters if ein_quarters else None,
    )

if __name__ == '__main__':
    main()
