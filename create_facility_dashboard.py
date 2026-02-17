#!/usr/bin/env python3
"""
Create facility dashboard files (CSV and provider info)
Usage: python create_facility_dashboard.py <PROVNUM>
Example: python create_facility_dashboard.py 315461
"""

import sys
import os
import shutil
from pathlib import Path
from dynamic_facility_dashboard import create_facility_complete_csv, create_facility_provider_info_csv
from file_path_utils import find_facility_complete_data, find_facility_provider_info, get_facility_folder
import pandas as pd

def main():
    if len(sys.argv) != 2:
        print("Usage: python create_facility_dashboard.py <PROVNUM>")
        print("Example: python create_facility_dashboard.py 315461")
        sys.exit(1)
    
    provnum = sys.argv[1].strip().zfill(6)
    
    if not provnum.isdigit() or len(provnum) != 6:
        print(f"[ERROR] Please enter a valid 6-digit CCN (e.g., 315461)")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"Creating Dashboard Files for Facility {provnum}")
    print(f"{'='*60}\n")
    
    # Get facility folder
    facility_folder = get_facility_folder(provnum)
    
    # Step 1: Create facility CSV if it doesn't exist
    csv_file = find_facility_complete_data(provnum)
    if csv_file and os.path.exists(csv_file):
        print(f"[OK] CSV file already exists: {os.path.basename(csv_file)}")
    else:
        csv_filename = f'facility_{provnum}_complete_data.csv'
        csv_file = str(facility_folder / csv_filename)
        print(f"[INFO] Creating CSV file: {csv_filename}")
        print("This may take a few minutes...")
        try:
            create_facility_complete_csv(provnum)
            # Check if file was created in root, move it to facility folder
            root_csv = f'facility_{provnum}_complete_data.csv'
            if os.path.exists(root_csv) and not os.path.exists(csv_file):
                shutil.move(root_csv, csv_file)
            if os.path.exists(csv_file):
                print(f"[OK] CSV file created successfully: {csv_filename}")
            else:
                print(f"[ERROR] Failed to create CSV file")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Error creating CSV file: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    print()
    
    # Step 2: Create provider info CSV if it doesn't exist
    provider_csv_file = find_facility_provider_info(provnum)
    if provider_csv_file and os.path.exists(provider_csv_file):
        print(f"[OK] Provider info CSV already exists: {os.path.basename(provider_csv_file)}")
    else:
        provider_filename = f'facility_{provnum}_provider_info_data.csv'
        provider_csv_file = str(facility_folder / provider_filename)
        print(f"[INFO] Creating provider info CSV: {provider_filename}")
        try:
            provider_data = create_facility_provider_info_csv(provnum)
            if provider_data is not None and len(provider_data) > 0:
                provider_data.to_csv(provider_csv_file, index=False)
                # Check if file was created in root, move it to facility folder
                root_provider = f'facility_{provnum}_provider_info_data.csv'
                if os.path.exists(root_provider) and not os.path.exists(provider_csv_file):
                    shutil.move(root_provider, provider_csv_file)
                print(f"[OK] Provider info CSV created successfully: {provider_filename}")
                print(f"   Found {len(provider_data)} provider info records")
            else:
                print(f"[WARNING] No provider info data found for facility {provnum}")
                print(f"   (This is okay - the dashboard will still work)")
        except Exception as e:
            print(f"[WARNING] Error creating provider info CSV: {str(e)}")
            print(f"   (This is okay - the dashboard will still work)")
    
    print()
    print(f"{'='*60}")
    print(f"[SUCCESS] Dashboard files ready for facility {provnum}")
    print(f"{'='*60}")
    print(f"\nTo run the dashboard, use:")
    print(f"  run_facility_dashboard.bat {provnum}")
    print(f"  or")
    print(f"  python dynamic_facility_dashboard.py {provnum}")
    print()

if __name__ == '__main__':
    main()
