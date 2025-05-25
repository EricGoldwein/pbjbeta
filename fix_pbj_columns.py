import pandas as pd
import os
from pathlib import Path

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

def standardize_column_name(col):
    # Convert to lowercase for comparison
    col_lower = col.lower()
    
    # Map of known variations to standard names
    column_mapping = {
        'cy_qtr': 'CY_Qtr',
        'workdate': 'WorkDate',
        'mdscensus': 'MDScensus',
        'hrs_rndon': 'Hrs_RNDON',
        'hrs_rndon_emp': 'Hrs_RNDON_emp',
        'hrs_rndon_ctr': 'Hrs_RNDON_ctr',
        'hrs_rnadmin': 'Hrs_RNadmin',
        'hrs_rnadmin_emp': 'Hrs_RNadmin_emp',
        'hrs_rnadmin_ctr': 'Hrs_RNadmin_ctr',
        'hrs_rn': 'Hrs_RN',
        'hrs_rn_emp': 'Hrs_RN_emp',
        'hrs_rn_ctr': 'Hrs_RN_ctr',
        'hrs_lpn_admin': 'Hrs_LPNadmin',
        'hrs_lpnadmin_emp': 'Hrs_LPNadmin_emp',
        'hrs_lpnadmin_ctr': 'Hrs_LPNadmin_ctr',
        'hrs_lpn': 'Hrs_LPN',
        'hrs_lpn_emp': 'Hrs_LPN_emp',
        'hrs_lpn_ctr': 'Hrs_LPN_ctr',
        'hrs_cna': 'Hrs_CNA',
        'hrs_cna_emp': 'Hrs_CNA_emp',
        'hrs_cna_ctr': 'Hrs_CNA_ctr',
        'hrs_na_trn': 'Hrs_NAtrn',
        'hrs_natrn_emp': 'Hrs_NAtrn_emp',
        'hrs_natrn_ctr': 'Hrs_NAtrn_ctr',
        'hrs_medaide': 'Hrs_MedAide',
        'hrs_medaide_emp': 'Hrs_MedAide_emp',
        'hrs_medaide_ctr': 'Hrs_MedAide_ctr'
    }
    
    return column_mapping.get(col_lower, col)

def fix_pbj_file(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Standardize column names
        df.columns = [standardize_column_name(col) for col in df.columns]
        
        # Get actual columns after standardization
        actual_columns = list(df.columns)
        
        # Add missing columns with zero values
        for col in EXPECTED_COLUMNS:
            if col not in actual_columns:
                df[col] = 0
                print(f"Added missing column: {col}")
        
        # Reorder columns to match expected order
        df = df[EXPECTED_COLUMNS]
        
        # Save the fixed file
        df.to_csv(file_path, index=False)
        print(f"\nSuccessfully fixed {file_path}")
        
    except Exception as e:
        print(f"Error fixing {file_path}: {str(e)}")

if __name__ == "__main__":
    # Fix the problematic file
    fix_pbj_file('standardized_PBJ/PBJ_dailynursestaffing_CY2017Q2.csv') 