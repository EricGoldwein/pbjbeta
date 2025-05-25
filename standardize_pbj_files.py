import pandas as pd
import os
from pathlib import Path
import glob
from datetime import datetime
import shutil

# Define column name variations and their standard mappings
COLUMN_VARIATIONS = {
    # Facility information variations
    'provnum': 'PROVNUM',
    'PROVNUM': 'PROVNUM',
    'provname': 'PROVNAME',
    'PROVNAME': 'PROVNAME',
    'city': 'CITY',
    'CITY': 'CITY',
    'state': 'STATE',
    'STATE': 'STATE',
    'county_name': 'COUNTY_NAME',
    'COUNTY_NAME': 'COUNTY_NAME',
    'county_fips': 'COUNTY_FIPS',
    'COUNTY_FIPS': 'COUNTY_FIPS',
    
    # Time and census variations
    'cy_qtr': 'CY_Qtr',
    'CY_Qtr': 'CY_Qtr',
    'workdate': 'WorkDate',
    'WorkDate': 'WorkDate',
    'mdscensus': 'MDScensus',
    'MDScensus': 'MDScensus',
    
    # RN DON variations
    'hrs_rn_donadmin': 'Hrs_RNDON',
    'hrs_rndon': 'Hrs_RNDON',
    'Hrs_RNDON': 'Hrs_RNDON',
    'hrs_rndon_emp': 'Hrs_RNDON_emp',
    'Hrs_RNDON_emp': 'Hrs_RNDON_emp',
    'hrs_rndon_ctr': 'Hrs_RNDON_ctr',
    'Hrs_RNDON_ctr': 'Hrs_RNDON_ctr',
    
    # RN admin variations
    'hrs_rnadmin': 'Hrs_RNadmin',
    'Hrs_RNadmin': 'Hrs_RNadmin',
    'hrs_rnadmin_emp': 'Hrs_RNadmin_emp',
    'Hrs_RNadmin_emp': 'Hrs_RNadmin_emp',
    'hrs_rnadmin_ctr': 'Hrs_RNadmin_ctr',
    'Hrs_RNadmin_ctr': 'Hrs_RNadmin_ctr',
    
    # RN variations
    'hrs_rn': 'Hrs_RN',
    'Hrs_RN': 'Hrs_RN',
    'hrs_rn_emp': 'Hrs_RN_emp',
    'Hrs_RN_emp': 'Hrs_RN_emp',
    'hrs_rn_ctr': 'Hrs_RN_ctr',
    'Hrs_RN_ctr': 'Hrs_RN_ctr',
    
    # LPN admin variations
    'hrs_lpn_admin': 'Hrs_LPNadmin',
    'hrs_lpnadmin': 'Hrs_LPNadmin',
    'Hrs_LPNadmin': 'Hrs_LPNadmin',
    'hrs_lpnadmin_emp': 'Hrs_LPNadmin_emp',
    'Hrs_LPNadmin_emp': 'Hrs_LPNadmin_emp',
    'hrs_lpnadmin_ctr': 'Hrs_LPNadmin_ctr',
    'Hrs_LPNadmin_ctr': 'Hrs_LPNadmin_ctr',
    
    # LPN variations
    'hrs_lpn': 'Hrs_LPN',
    'Hrs_LPN': 'Hrs_LPN',
    'hrs_lpn_emp': 'Hrs_LPN_emp',
    'Hrs_LPN_emp': 'Hrs_LPN_emp',
    'hrs_lpn_ctr': 'Hrs_LPN_ctr',
    'Hrs_LPN_ctr': 'Hrs_LPN_ctr',
    
    # CNA variations
    'hrs_cna': 'Hrs_CNA',
    'Hrs_CNA': 'Hrs_CNA',
    'hrs_cna_emp': 'Hrs_CNA_emp',
    'Hrs_CNA_emp': 'Hrs_CNA_emp',
    'hrs_cna_ctr': 'Hrs_CNA_ctr',
    'Hrs_CNA_ctr': 'Hrs_CNA_ctr',
    
    # NA training variations
    'hrs_na_trn': 'Hrs_NAtrn',
    'hrs_natrn': 'Hrs_NAtrn',
    'Hrs_NAtrn': 'Hrs_NAtrn',
    'hrs_natrn_emp': 'Hrs_NAtrn_emp',
    'Hrs_NAtrn_emp': 'Hrs_NAtrn_emp',
    'hrs_natrn_ctr': 'Hrs_NAtrn_ctr',
    'Hrs_NAtrn_ctr': 'Hrs_NAtrn_ctr',
    
    # Med Aide variations
    'hrs_medaide': 'Hrs_MedAide',
    'Hrs_MedAide': 'Hrs_MedAide',
    'hrs_medaide_emp': 'Hrs_MedAide_emp',
    'Hrs_MedAide_emp': 'Hrs_MedAide_emp',
    'hrs_medaide_ctr': 'Hrs_MedAide_ctr',
    'Hrs_MedAide_ctr': 'Hrs_MedAide_ctr'
}

def format_provnum(df):
    """Preserve original PROVNUM values with minimal cleanup."""
    if 'PROVNUM' in df.columns:
        # First convert to string to handle any numeric values
        df['PROVNUM'] = df['PROVNUM'].astype(str)
        
        # Convert to uppercase to standardize case
        df['PROVNUM'] = df['PROVNUM'].str.upper()
        
        # Log any PROVNUMs that are not 1-6 characters
        invalid_provnums = df[~df['PROVNUM'].str.match(r'^[A-Z0-9]{1,6}$')]
        if not invalid_provnums.empty:
            print("\nWarning: Found PROVNUMs that are not 1-6 characters:")
            print(invalid_provnums[['PROVNUM', 'PROVNAME', 'STATE']].to_string())
        
        return True
    return False

def standardize_column_names(df, file_path):
    """Standardize column names to match Q4 2024 format."""
    # Create a mapping dictionary for column names
    column_mapping = {}
    changes_made = []
    
    # Special handling for 2021Q4 - remove incomplete column
    if 'PBJ_dailynursestaffing_CY2021Q4.csv' in file_path:
        if 'incomplete' in df.columns:
            df = df.drop(columns=['incomplete'])
            changes_made.append("Removed 'incomplete' column")
    
    for col in df.columns:
        # First check for exact matches in variations
        if col in COLUMN_VARIATIONS:
            if col != COLUMN_VARIATIONS[col]:
                column_mapping[col] = COLUMN_VARIATIONS[col]
                changes_made.append(f"{col} -> {COLUMN_VARIATIONS[col]}")
        else:
            # Try to match by converting to uppercase and removing underscores
            col_upper = col.upper().replace('_', '')
            for var, std in COLUMN_VARIATIONS.items():
                if var.upper().replace('_', '') == col_upper:
                    if col != std:
                        column_mapping[col] = std
                        changes_made.append(f"{col} -> {std}")
                    break
    
    # Only rename if there are changes
    if changes_made:
        df = df.rename(columns=column_mapping)
    
    # Format PROVNUM
    if format_provnum(df):
        changes_made.append("Formatted PROVNUM to uppercase")
    
    return df, changes_made

def process_pbj_files():
    # Create output directory if it doesn't exist
    output_dir = Path('standardized_PBJ')
    output_dir.mkdir(exist_ok=True)
    
    # Get all PBJ CSV files from PBJcsv directory
    pbj_files = glob.glob('PBJcsv/PBJ_dailynursestaffing_*.csv')
    
    # Sort files by date
    pbj_files.sort()
    
    # Track statistics
    total_files = len(pbj_files)
    processed_files = 0
    skipped_already_processed = 0
    skipped_no_changes = 0
    changed_files = 0
    errors = []
    
    print(f"\nStarting processing at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Found {total_files} files to process\n")
    
    for file_path in pbj_files:
        try:
            # Create output filename
            output_file = output_dir / Path(file_path).name
            
            # Try different encodings
            encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    # Read PROVNUM as string from the start
                    df = pd.read_csv(file_path, encoding=encoding, low_memory=False, dtype={'PROVNUM': str})
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Could not read file with any of the attempted encodings")
            
            # Standardize column names
            df_standardized, changes = standardize_column_names(df, file_path)
            
            if changes:
                # Save standardized file with UTF-8 encoding
                df_standardized.to_csv(output_file, index=False, encoding='utf-8')
                print(f"\nProcessed {file_path}")
                print("Changes made:")
                for change in changes:
                    print(f"  - {change}")
                changed_files += 1
            else:
                print(f"No changes needed for {file_path}")
                skipped_no_changes += 1
            
            processed_files += 1
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            print(error_msg)
            errors.append(error_msg)
    
    # Print summary
    print(f"\nProcessing completed at {datetime.now().strftime('%H:%M:%S')}")
    print(f"\nSummary:")
    print(f"Total files found: {total_files}")
    print(f"Files processed: {processed_files}")
    print(f"  - No changes needed: {skipped_no_changes}")
    print(f"  - Files changed: {changed_files}")
    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for error in errors:
            print(f"  - {error}")

if __name__ == "__main__":
    process_pbj_files() 