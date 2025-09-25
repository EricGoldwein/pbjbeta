#!/usr/bin/env python3
"""
Create a complete CSV file for any facility, similar to the 225500 process
Usage: python create_facility_csv.py <PROVNUM>
"""

import os
import glob
import pandas as pd
import sys
from datetime import datetime

def create_facility_csv(provnum):
    """Extract all data for any facility into one CSV"""
    print(f"Creating comprehensive CSV for facility {provnum}...")
    
    # Load all standardized nurse files
    nurse_files = glob.glob('standardized_PBJ/PBJ_dailynursestaffing_*.csv')
    nurse_files.sort()
    
    all_data = []
    total_records = 0
    
    for file_path in nurse_files:
        print(f"Processing: {os.path.basename(file_path)}")
        
        try:
            # Read the CSV
            df = pd.read_csv(file_path, low_memory=False)
            
            # Format PROVNUM to ensure it's a 6-digit string
            df['PROVNUM'] = df['PROVNUM'].astype(str).str.zfill(6)
            
            # Filter for the specific facility (handle different PROVNUM formats)
            facility_data = df[(df['PROVNUM'] == provnum) | (df['PROVNUM'] == provnum.lstrip('0')) | (df['PROVNUM'] == provnum.zfill(6))].copy()
            
            if len(facility_data) == 0:
                print(f"  No data for facility {provnum}")
                continue
            
            print(f"  Found {len(facility_data)} records")
            all_data.append(facility_data)
            total_records += len(facility_data)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if len(all_data) == 0:
        print(f"No data found for facility {provnum}!")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    combined_data = combined_data.sort_values('WorkDate')
    
    # Save to CSV
    output_filename = f'facility_{provnum}_complete_data.csv'
    combined_data.to_csv(output_filename, index=False)
    
    print(f"\nExtraction Summary:")
    print(f"Total records: {total_records}")
    print(f"Date range: {combined_data['WorkDate'].min()} to {combined_data['WorkDate'].max()}")
    print(f"Quarters: {combined_data['CY_Qtr'].nunique()}")
    print(f"File saved as: {output_filename}")
    
    # Show sample data
    print(f"\nSample data:")
    print(combined_data[['WorkDate', 'PROVNUM', 'PROVNAME', 'MDScensus', 'Hrs_RN', 'Hrs_LPN', 'Hrs_CNA']].head())
    
    return combined_data

def main():
    if len(sys.argv) != 2:
        print("Usage: python create_facility_csv.py <PROVNUM>")
        print("Example: python create_facility_csv.py 015009")
        sys.exit(1)
    
    provnum = sys.argv[1].strip().zfill(6)
    
    if not provnum.isdigit() or len(provnum) != 6:
        print("❌ Please enter a valid 6-digit CCN (e.g., 015009)")
        sys.exit(1)
    
    print(f"🏥 Creating complete data file for facility {provnum}")
    print("=" * 50)
    
    result = create_facility_csv(provnum)
    
    if result is not None:
        print(f"\n✅ Successfully created facility_{provnum}_complete_data.csv")
    else:
        print(f"\n❌ Failed to create data file for facility {provnum}")
        sys.exit(1)

if __name__ == "__main__":
    main()
