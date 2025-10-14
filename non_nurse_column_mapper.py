import pandas as pd
import os
from pathlib import Path
import re

# Standard columns from Q4 2024
STANDARD_COLUMNS = [
    "PROVNUM", "PROVNAME", "CITY", "STATE", "COUNTY_NAME", "COUNTY_FIPS",
    "CY_Qtr", "WorkDate", "MDScensus", "Hrs_Admin", "Hrs_Admin_emp", "Hrs_Admin_ctr",
    "Hrs_Admin_fn", "Hrs_MedDir", "Hrs_MedDir_emp", "Hrs_MedDir_ctr", "Hrs_OthMD",
    "Hrs_OthMD_emp", "Hrs_OthMD_ctr", "Hrs_PA", "Hrs_PA_emp", "Hrs_PA_ctr",
    "Hrs_NP", "Hrs_NP_emp", "Hrs_NP_ctr", "Hrs_ClinNrsSpec", "Hrs_ClinNrsSpec_emp",
    "Hrs_ClinNrsSpec_ctr", "Hrs_Pharmacist", "Hrs_Pharmacist_emp", "Hrs_Pharmacist_ctr",
    "Hrs_Dietician", "Hrs_Dietician_emp", "Hrs_Dietician_ctr", "Hrs_FeedAsst",
    "Hrs_FeedAsst_emp", "Hrs_FeedAsst_ctr", "Hrs_OT", "Hrs_OT_emp", "Hrs_OT_ctr",
    "Hrs_OTasst", "Hrs_OTasst_emp", "Hrs_OTasst_ctr", "Hrs_OTaide", "Hrs_OTaide_emp",
    "Hrs_OTaide_ctr", "Hrs_PT", "Hrs_PT_emp", "Hrs_PT_ctr", "Hrs_PTasst",
    "Hrs_PTasst_emp", "Hrs_PTasst_ctr", "Hrs_PTaide", "Hrs_PTaide_emp", "Hrs_PTaide_ctr",
    "Hrs_RespTher", "Hrs_RespTher_emp", "Hrs_RespTher_ctr", "Hrs_RespTech",
    "Hrs_RespTech_emp", "Hrs_RespTech_ctr", "Hrs_SpcLangPath", "Hrs_SpcLangPath_emp",
    "Hrs_SpcLangPath_ctr", "Hrs_TherRecSpec", "Hrs_TherRecSpec_emp", "Hrs_TherRecSpec_ctr",
    "Hrs_QualActvProf", "Hrs_QualActvProf_emp", "Hrs_QualActvProf_ctr", "Hrs_OthActv",
    "Hrs_OthActv_emp", "Hrs_OthActv_ctr", "Hrs_QualSocWrk", "Hrs_QualSocWrk_emp",
    "Hrs_QualSocWrk_ctr", "Hrs_OthSocWrk", "Hrs_OthSocWrk_emp", "Hrs_OthSocWrk_ctr",
    "Hrs_MHSvc", "Hrs_MHSvc_emp", "Hrs_MHSvc_ctr"
]

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
            # Try normalized matching
            norm_file_col = normalize_column_name(orig_file_col)
            if norm_file_col in normalized_standard:
                mapping[orig_file_col] = normalized_standard[norm_file_col]
            else:
                mapping[orig_file_col] = None
        
        return mapping
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def main():
    # Directory containing CSV files
    csv_dir = Path("NonNursecsv")
    
    # Create output directory for reports
    output_dir = Path("column_mapping_reports")
    output_dir.mkdir(exist_ok=True)
    
    # Process each CSV file
    for csv_file in csv_dir.glob("PBJ_dailynonnursestaffing_*.csv"):
        print(f"\nProcessing {csv_file.name}")
        
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