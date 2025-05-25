import pandas as pd
import os
from pathlib import Path
import glob

# Expected column names in the correct order
EXPECTED_COLUMNS = [
    "PROVNUM", "PROVNAME", "CITY", "STATE", "COUNTY_NAME", "COUNTY_FIPS",
    "CY_Qtr", "WorkDate", "MDScensus",
    "Hrs_RNDON", "Hrs_RNDON_emp", "Hrs_RNDON_ctr",
    "Hrs_RNadmin", "Hrs_RNadmin_emp", "Hrs_RNadmin_ctr",
    "Hrs_RN", "Hrs_RN_emp", "Hrs_RN_ctr",
    "Hrs_LPNadmin", "Hrs_LPNadmin_emp", "Hrs_LPNadmin_ctr",
    "Hrs_LPN", "Hrs_LPN_emp", "Hrs_LPN_ctr",
    "Hrs_CNA", "Hrs_CNA_emp", "Hrs_CNA_ctr",
    "Hrs_NAtrn", "Hrs_NAtrn_emp", "Hrs_NAtrn_ctr",
    "Hrs_MedAide", "Hrs_MedAide_emp", "Hrs_MedAide_ctr"
]

def verify_pbj_files():
    # Get all PBJ CSV files in the standardized_PBJ directory
    pbj_files = glob.glob('standardized_PBJ/PBJ_dailynursestaffing_*.csv')
    
    # Sort files by date
    pbj_files.sort()
    
    print(f"\nFound {len(pbj_files)} files to verify\n")
    
    # Track statistics
    total_files = len(pbj_files)
    consistent_files = 0
    inconsistent_files = []
    
    for file_path in pbj_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path, nrows=0)  # Only read headers
            
            # Get actual columns
            actual_columns = list(df.columns)
            
            # Check if columns match expected format
            if actual_columns == EXPECTED_COLUMNS:
                print(f"✓ {file_path} - Columns are consistent")
                consistent_files += 1
            else:
                print(f"✗ {file_path} - Columns are inconsistent")
                print("  Expected columns:", EXPECTED_COLUMNS)
                print("  Actual columns:", actual_columns)
                inconsistent_files.append(file_path)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            inconsistent_files.append(file_path)
    
    # Print summary
    print(f"\nVerification Summary:")
    print(f"Total files checked: {total_files}")
    print(f"Consistent files: {consistent_files}")
    print(f"Inconsistent files: {len(inconsistent_files)}")
    
    if inconsistent_files:
        print("\nInconsistent files:")
        for file in inconsistent_files:
            print(f"  - {file}")

if __name__ == "__main__":
    verify_pbj_files() 