#!/usr/bin/env python3
"""
Simple launcher for the Dynamic Facility Dashboard
"""

import sys
import os

def main():
    print("🏥 PBJ Facility Dashboard Launcher")
    print("=" * 40)
    
    # Check if required files exist
    required_files = [
        'facility_quarterly_metrics.csv',
        'dynamic_facility_dashboard.py',
        'templates/dynamic_facility_dashboard.html'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present before running.")
        return
    
    print("✅ All required files found!")
    print("\nStarting Dynamic Facility Dashboard...")
    print("=" * 40)
    
    # Import and run the generator
    try:
        from dynamic_facility_dashboard import main as run_generator
        run_generator()
    except ImportError as e:
        print(f"❌ Error importing dashboard generator: {e}")
        print("Please ensure dynamic_facility_dashboard.py is in the current directory.")
    except Exception as e:
        print(f"❌ Error running dashboard generator: {e}")

if __name__ == '__main__':
    main()