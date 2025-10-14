#!/usr/bin/env python3
"""
Script to create a proper comparison file between facility 225500 and Massachusetts state average.
"""

import pandas as pd
import os
from datetime import datetime

def create_facility_state_comparison():
    """Create a comparison file between facility 225500 and Massachusetts state average."""
    
    print("Loading facility lite metrics for facility 225500...")
    
    # Load facility lite metrics
    facility_file = 'facility_lite_metrics.csv'
    if not os.path.exists(facility_file):
        print(f"Error: {facility_file} not found!")
        return
    
    facility_df = pd.read_csv(facility_file, dtype={'PROVNUM': str})
    print(f"Loaded {len(facility_df)} facility records")
    
    # Filter for facility 225500
    facility_225500 = facility_df[facility_df['PROVNUM'] == '225500'].copy()
    print(f"Found {len(facility_225500)} records for facility 225500")
    
    print("\nLoading state lite metrics for Massachusetts...")
    
    # Load state lite metrics
    state_file = 'state_lite_metrics.csv'
    if not os.path.exists(state_file):
        print(f"Error: {state_file} not found!")
        return
    
    state_df = pd.read_csv(state_file, dtype={'STATE': str})
    print(f"Loaded {len(state_df)} state records")
    
    # Filter for Massachusetts (MA)
    mass_data = state_df[state_df['STATE'] == 'MA'].copy()
    print(f"Found {len(mass_data)} records for Massachusetts")
    
    # Prepare facility data for comparison
    facility_comparison = facility_225500[['CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 
                                         'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 
                                         'Total_RN_HPRD', 'Direct_Care_RN_HPRD', 
                                         'Contract_Percentage', 'Census']].copy()
    facility_comparison['Data_Source'] = 'Facility_225500'
    facility_comparison['Source_Type'] = 'Facility'
    
    # Prepare state data for comparison
    state_comparison = mass_data[['CY_Qtr', 'STATE', 'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 
                                'Total_RN_HPRD', 'Direct_Care_RN_HPRD', 
                                'Contract_Percentage', 'Census']].copy()
    state_comparison['PROVNUM'] = 'MA_STATE'
    state_comparison['PROVNAME'] = 'Massachusetts (State Average)'
    state_comparison['Data_Source'] = 'Massachusetts_State'
    state_comparison['Source_Type'] = 'State'
    
    # Ensure both have the same column order
    column_order = ['CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 'Data_Source', 'Source_Type',
                   'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Direct_Care_RN_HPRD', 
                   'Contract_Percentage', 'Census']
    
    facility_comparison = facility_comparison[column_order]
    state_comparison = state_comparison[column_order]
    
    # Combine the data
    comparison_data = pd.concat([facility_comparison, state_comparison], ignore_index=True)
    
    # Sort by quarter for better organization
    comparison_data = comparison_data.sort_values('CY_Qtr')
    
    # Generate output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'facility_225500_vs_massachusetts_comparison_{timestamp}.csv'
    
    # Save to CSV
    comparison_data.to_csv(output_file, index=False)
    
    print(f"\nComparison file created: {output_file}")
    print(f"Total records: {len(comparison_data)}")
    print(f"  - Facility 225500: {len(facility_comparison)} records")
    print(f"  - Massachusetts State: {len(state_comparison)} records")
    
    # Show sample of the comparison data
    print(f"\nSample of comparison data:")
    print(comparison_data[['CY_Qtr', 'Data_Source', 'Total_Nurse_HPRD', 'Total_RN_HPRD', 
                          'Direct_Care_RN_HPRD', 'Contract_Percentage', 'Census']].head(10))
    
    # Create a side-by-side comparison for recent quarters
    print(f"\nSide-by-side comparison (last 5 quarters):")
    recent_quarters = ['2024Q1', '2024Q2', '2024Q3', '2024Q4', '2025Q1']
    recent_data = comparison_data[comparison_data['CY_Qtr'].isin(recent_quarters)]
    
    if not recent_data.empty:
        # Pivot to show facility vs state side by side
        pivot_comparison = recent_data.pivot_table(
            index='CY_Qtr', 
            columns='Data_Source', 
            values=['Total_Nurse_HPRD', 'Total_RN_HPRD', 'Direct_Care_RN_HPRD', 
                   'Contract_Percentage', 'Census'],
            aggfunc='mean'
        ).round(3)
        print(pivot_comparison)
        
        # Calculate differences
        print(f"\nPerformance differences (Facility 225500 - Massachusetts):")
        for quarter in recent_quarters:
            facility_row = recent_data[(recent_data['CY_Qtr'] == quarter) & 
                                     (recent_data['Data_Source'] == 'Facility_225500')]
            state_row = recent_data[(recent_data['CY_Qtr'] == quarter) & 
                                  (recent_data['Data_Source'] == 'Massachusetts_State')]
            
            if not facility_row.empty and not state_row.empty:
                print(f"\n{quarter}:")
                print(f"  Total Nurse HPRD: {facility_row['Total_Nurse_HPRD'].iloc[0]:.3f} vs {state_row['Total_Nurse_HPRD'].iloc[0]:.3f} (diff: {facility_row['Total_Nurse_HPRD'].iloc[0] - state_row['Total_Nurse_HPRD'].iloc[0]:+.3f})")
                print(f"  Total RN HPRD: {facility_row['Total_RN_HPRD'].iloc[0]:.3f} vs {state_row['Total_RN_HPRD'].iloc[0]:.3f} (diff: {facility_row['Total_RN_HPRD'].iloc[0] - state_row['Total_RN_HPRD'].iloc[0]:+.3f})")
                print(f"  Direct Care RN HPRD: {facility_row['Direct_Care_RN_HPRD'].iloc[0]:.3f} vs {state_row['Direct_Care_RN_HPRD'].iloc[0]:.3f} (diff: {facility_row['Direct_Care_RN_HPRD'].iloc[0] - state_row['Direct_Care_RN_HPRD'].iloc[0]:+.3f})")
                print(f"  Contract %: {facility_row['Contract_Percentage'].iloc[0]:.1f}% vs {state_row['Contract_Percentage'].iloc[0]:.1f}% (diff: {facility_row['Contract_Percentage'].iloc[0] - state_row['Contract_Percentage'].iloc[0]:+.1f}%)")
                print(f"  Census: {facility_row['Census'].iloc[0]:.0f} vs {state_row['Census'].iloc[0]:.0f} (diff: {facility_row['Census'].iloc[0] - state_row['Census'].iloc[0]:+.0f})")
    
    return output_file

if __name__ == "__main__":
    create_facility_state_comparison()
