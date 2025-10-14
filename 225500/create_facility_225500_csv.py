#!/usr/bin/env python3
"""
Create a single CSV with ALL data for facility 225500 from 2017-2025
"""

import os
import glob
import pandas as pd
from datetime import datetime

def create_facility_225500_csv():
    """Extract all data for facility 225500 into one CSV"""
    print("Creating comprehensive CSV for facility 225500...")
    
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
            
            # Filter for facility 225500
            facility_data = df[df['PROVNUM'] == '225500'].copy()
            
            if len(facility_data) == 0:
                print(f"  No data for facility 225500")
                continue
            
            print(f"  Found {len(facility_data)} records")
            all_data.append(facility_data)
            total_records += len(facility_data)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if len(all_data) == 0:
        print("No data found for facility 225500!")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    combined_data = combined_data.sort_values('WorkDate')
    
    # Save to CSV
    output_filename = 'facility_225500_complete_data.csv'
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

if __name__ == "__main__":
    create_facility_225500_csv()
