import pandas as pd
import os
from pathlib import Path
import re

# Standard columns from Q4 2024
STANDARD_COLUMNS = [
    "PROVNUM", "PROVNAME", "CITY", "STATE", "COUNTY_NAME", "COUNTY_FIPS",
    "CY_Qtr", "WorkDate", "MDScensus", "Hrs_RNDON", "Hrs_RNDON_emp",
    "Hrs_RNDON_ctr", "Hrs_RNadmin", "Hrs_RNadmin_emp", "Hrs_RNadmin_ctr",
    "Hrs_RN", "Hrs_RN_emp", "Hrs_RN_ctr", "Hrs_LPNadmin", "Hrs_LPNadmin_emp",
    "Hrs_LPNadmin_ctr", "Hrs_LPN", "Hrs_LPN_emp", "Hrs_LPN_ctr", "Hrs_CNA",
    "Hrs_CNA_emp", "Hrs_CNA_ctr", "Hrs_NAtrn", "Hrs_NAtrn_emp", "Hrs_NAtrn_ctr",
    "Hrs_MedAide", "Hrs_MedAide_emp", "Hrs_MedAide_ctr"
]

# Additional column mappings for special cases
SPECIAL_MAPPINGS = {
    'hrs_lpn_admin': 'Hrs_LPNadmin',
    'hrs_rn_donadmin': 'Hrs_RNDON',
    'hrs_na_trn': 'Hrs_NAtrn'
}

def normalize_column_name(col):
    """Normalize column names by removing special characters and converting to uppercase."""
    # Remove special characters and convert to uppercase
    col = re.sub(r'[^a-zA-Z0-9_]', '', col.upper())
    # Replace multiple underscores with single underscore
    col = re.sub(r'_+', '_', col)
    return col

def create_column_mapping(file_path):
    """Create a mapping of columns from a file to standard columns."""
    try:
        # Read just the header row
        df = pd.read_csv(file_path, nrows=0)
        file_columns = df.columns.tolist()
        
        # Create normalized versions of both standard and file columns
        normalized_standard = {normalize_column_name(col): col for col in STANDARD_COLUMNS}
        normalized_file = {normalize_column_name(col): col for col in file_columns}
        
        # Create mapping
        mapping = {}
        for orig_file_col in file_columns:
            # First check special mappings
            if orig_file_col in SPECIAL_MAPPINGS:
                mapping[orig_file_col] = SPECIAL_MAPPINGS[orig_file_col]
            else:
                # Then try normalized matching
                norm_file_col = normalize_column_name(orig_file_col)
                if norm_file_col in normalized_standard:
                    mapping[orig_file_col] = normalized_standard[norm_file_col]
                else:
                    mapping[orig_file_col] = None
        
        return mapping
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def fix_2017q2_file(file_path):
    """Fix the inconsistent columns in the 2017Q2 file."""
    try:
        # Read the file
        df = pd.read_csv(file_path)
        
        # Define the column mappings
        column_mappings = {
            'hrs_lpn_admin': 'Hrs_LPNadmin',
            'hrs_rn_donadmin': 'Hrs_RNDON',
            'hrs_na_trn': 'Hrs_NAtrn',
            'hrs_natrn_emp': 'Hrs_NAtrn_emp',
            'hrs_natrn_ctr': 'Hrs_NAtrn_ctr',
            'cy_qtr': 'CY_Qtr',
            'workdate': 'WorkDate',
            'mdscensus': 'MDScensus'
        }
        
        # Rename the columns
        df = df.rename(columns=column_mappings)
        
        # Add missing Hrs_MedAide_ctr column
        df['Hrs_MedAide_ctr'] = 0
        
        # Save the fixed file
        df.to_csv(file_path, index=False)
        print(f"Successfully fixed {file_path}")
        return True
    except Exception as e:
        print(f"Error fixing {file_path}: {str(e)}")
        return False

def main():
    # Directory containing CSV files
    csv_dir = Path("standardized_PBJ")
    
    # Create output directory for reports
    output_dir = Path("column_mapping_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Process each CSV file
    for csv_file in csv_dir.glob("PBJ_dailynursestaffing_*.csv"):
        print(f"\nProcessing {csv_file.name}")
        
        # Special handling for 2017Q2 file
        if "CY2017Q2" in csv_file.name:
            if fix_2017q2_file(csv_file):
                continue
        
        # Create mapping for this file
        mapping = create_column_mapping(csv_file)
        
        if mapping:
            # Create report
            report_path = output_dir / f"mapping_report_{csv_file.stem}.txt"
            with open(report_path, 'w') as f:
                f.write(f"Column Mapping Report for {csv_file.name}\n")
                f.write("=" * 50 + "\n\n")
                
                for orig_col, std_col in mapping.items():
                    if std_col:
                        f.write(f"Original: {orig_col} -> Standard: {std_col}\n")
                    else:
                        f.write(f"Original: {orig_col} -> No standard mapping found\n")
            
            print(f"Report created: {report_path}")

if __name__ == "__main__":
    main() 