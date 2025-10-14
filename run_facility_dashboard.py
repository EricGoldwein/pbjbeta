#!/usr/bin/env python3
"""
Run a facility dashboard for any 6-digit CCN
Usage: python run_facility_dashboard.py <PROVNUM>
Example: python run_facility_dashboard.py 015009
"""

import sys
import os
import subprocess
from dynamic_facility_dashboard import run_dashboard

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_facility_dashboard.py <PROVNUM>")
        print("Example: python run_facility_dashboard.py 015009")
        sys.exit(1)
    
    provnum = sys.argv[1].strip().zfill(6)
    
    if not provnum.isdigit() or len(provnum) != 6:
        print("❌ Please enter a valid 6-digit CCN (e.g., 015009)")
        sys.exit(1)
    
    print(f"🏥 Starting Dashboard for Facility {provnum}")
    print("=" * 50)
    
    # Check if CSV exists, if not create it
    csv_filename = f'facility_{provnum}_complete_data.csv'
    
    if not os.path.exists(csv_filename):
        print(f"📊 CSV file not found. Creating {csv_filename}...")
        print("This may take a few minutes...")
        
        # Run the CSV generator
        result = subprocess.run([sys.executable, 'create_facility_csv.py', provnum], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Error creating CSV: {result.stderr}")
            sys.exit(1)
        
        print("✅ CSV file created successfully!")
    else:
        print(f"✅ Found existing CSV file: {csv_filename}")
    
    # Create and run the dashboard
    print(f"\n🚀 Starting dashboard for facility {provnum}...")
    
    success = run_dashboard(provnum, port=5000)
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()
