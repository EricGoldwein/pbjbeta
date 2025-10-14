#!/usr/bin/env python3
"""
Script to create a pivot table comparison with quarters as rows and specific metrics as columns.
"""

import pandas as pd
import os
from datetime import datetime

def create_pivot_comparison():
    """Create a pivot table comparison with quarters as rows and metrics as columns."""
    
    print("Loading comparison data...")
    
    # Load the comparison file
    comparison_file = 'facility_225500_vs_massachusetts_comparison_20250911_082044.csv'
    if not os.path.exists(comparison_file):
        print(f"Error: {comparison_file} not found!")
        return
    
    df = pd.read_csv(comparison_file)
    print(f"Loaded {len(df)} records")
    
    # Create pivot table with quarters as rows and metrics as columns
    pivot_data = df.pivot_table(
        index='CY_Qtr',
        columns='Data_Source',
        values=['Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Direct_Care_RN_HPRD'],
        aggfunc='mean'
    ).round(3)
    
    # Flatten the multi-level columns
    pivot_data.columns = [f"{col[1]}_{col[0]}" for col in pivot_data.columns]
    
    # Rename columns to match your requested format
    column_mapping = {
        'Facility_225500_Total_Nurse_HPRD': '225500 Total HPRD',
        'Massachusetts_State_Total_Nurse_HPRD': 'Mass Total HPRD',
        'Facility_225500_Nurse_Care_HPRD': '225500 Direct Care HPRD',
        'Massachusetts_State_Nurse_Care_HPRD': 'Mass Direct Care HPRD',
        'Facility_225500_Total_RN_HPRD': '225500 Total RN HPRD',
        'Massachusetts_State_Total_RN_HPRD': 'Mass Total RN HPRD',
        'Facility_225500_Direct_Care_RN_HPRD': '225500 Direct RN HPRD',
        'Massachusetts_State_Direct_Care_RN_HPRD': 'Mass Direct RN HPRD'
    }
    
    # Rename columns
    pivot_data = pivot_data.rename(columns=column_mapping)
    
    # Reorder columns to match your requested order
    column_order = [
        '225500 Direct Care HPRD',
        'Mass Direct Care HPRD', 
        '225500 Total HPRD',
        'Mass Total HPRD',
        '225500 Total RN HPRD',
        'Mass Total RN HPRD',
        '225500 Direct RN HPRD',
        'Mass Direct RN HPRD'
    ]
    
    # Only include columns that exist
    available_columns = [col for col in column_order if col in pivot_data.columns]
    pivot_data = pivot_data[available_columns]
    
    # Reset index to make CY_Qtr a regular column
    pivot_data = pivot_data.reset_index()
    
    # Format the quarter column to match your requested format (already in 2021Q2 format)
    # The quarters are already in the correct format, so no changes needed
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'facility_225500_mass_pivot_comparison_{timestamp}.csv'
    
    # Save to CSV
    pivot_data.to_csv(output_file, index=False)
    
    print(f"\nPivot comparison file created: {output_file}")
    print(f"Total quarters: {len(pivot_data)}")
    
    # Show the data
    print(f"\nPivot comparison data:")
    print(pivot_data.to_string(index=False))
    
    # Show recent quarters
    print(f"\nRecent quarters (last 5):")
    recent_data = pivot_data.tail(5)
    print(recent_data.to_string(index=False))
    
    return output_file

if __name__ == "__main__":
    create_pivot_comparison()


