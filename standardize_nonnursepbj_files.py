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
    
    # Admin variations
    'hrs_admin': 'Hrs_Admin',
    'Hrs_Admin': 'Hrs_Admin',
    'hrs_admin_emp': 'Hrs_Admin_emp',
    'Hrs_Admin_emp': 'Hrs_Admin_emp',
    'hrs_admin_ctr': 'Hrs_Admin_ctr',
    'Hrs_Admin_ctr': 'Hrs_Admin_ctr',
    'hrs_admin_fn': 'Hrs_Admin_fn',
    'Hrs_Admin_fn': 'Hrs_Admin_fn',
    
    # Medical Director variations
    'hrs_meddir': 'Hrs_MedDir',
    'Hrs_MedDir': 'Hrs_MedDir',
    'hrs_meddir_emp': 'Hrs_MedDir_emp',
    'Hrs_MedDir_emp': 'Hrs_MedDir_emp',
    'hrs_meddir_ctr': 'Hrs_MedDir_ctr',
    'Hrs_MedDir_ctr': 'Hrs_MedDir_ctr',
    
    # Other MD variations
    'hrs_othmd': 'Hrs_OthMD',
    'Hrs_OthMD': 'Hrs_OthMD',
    'hrs_othmd_emp': 'Hrs_OthMD_emp',
    'Hrs_OthMD_emp': 'Hrs_OthMD_emp',
    'hrs_othmd_ctr': 'Hrs_OthMD_ctr',
    'Hrs_OthMD_ctr': 'Hrs_OthMD_ctr',
    
    # PA variations
    'hrs_pa': 'Hrs_PA',
    'Hrs_PA': 'Hrs_PA',
    'hrs_pa_emp': 'Hrs_PA_emp',
    'Hrs_PA_emp': 'Hrs_PA_emp',
    'hrs_pa_ctr': 'Hrs_PA_ctr',
    'Hrs_PA_ctr': 'Hrs_PA_ctr',
    
    # NP variations
    'hrs_np': 'Hrs_NP',
    'Hrs_NP': 'Hrs_NP',
    'hrs_np_emp': 'Hrs_NP_emp',
    'Hrs_NP_emp': 'Hrs_NP_emp',
    'hrs_np_ctr': 'Hrs_NP_ctr',
    'Hrs_NP_ctr': 'Hrs_NP_ctr',
    
    # Clinical Nurse Specialist variations
    'hrs_clinnrsspec': 'Hrs_ClinNrsSpec',
    'Hrs_ClinNrsSpec': 'Hrs_ClinNrsSpec',
    'hrs_clinnrsspec_emp': 'Hrs_ClinNrsSpec_emp',
    'Hrs_ClinNrsSpec_emp': 'Hrs_ClinNrsSpec_emp',
    'hrs_clinnrsspec_ctr': 'Hrs_ClinNrsSpec_ctr',
    'Hrs_ClinNrsSpec_ctr': 'Hrs_ClinNrsSpec_ctr',
    
    # Pharmacist variations
    'hrs_pharmacist': 'Hrs_Pharmacist',
    'Hrs_Pharmacist': 'Hrs_Pharmacist',
    'hrs_pharmacist_emp': 'Hrs_Pharmacist_emp',
    'Hrs_Pharmacist_emp': 'Hrs_Pharmacist_emp',
    'hrs_pharmacist_ctr': 'Hrs_Pharmacist_ctr',
    'Hrs_Pharmacist_ctr': 'Hrs_Pharmacist_ctr',
    
    # Dietician variations
    'hrs_dietician': 'Hrs_Dietician',
    'Hrs_Dietician': 'Hrs_Dietician',
    'hrs_dietician_emp': 'Hrs_Dietician_emp',
    'Hrs_Dietician_emp': 'Hrs_Dietician_emp',
    'hrs_dietician_ctr': 'Hrs_Dietician_ctr',
    'Hrs_Dietician_ctr': 'Hrs_Dietician_ctr',
    
    # Feeding Assistant variations
    'hrs_feedasst': 'Hrs_FeedAsst',
    'Hrs_FeedAsst': 'Hrs_FeedAsst',
    'hrs_feedasst_emp': 'Hrs_FeedAsst_emp',
    'Hrs_FeedAsst_emp': 'Hrs_FeedAsst_emp',
    'hrs_feedasst_ctr': 'Hrs_FeedAsst_ctr',
    'Hrs_FeedAsst_ctr': 'Hrs_FeedAsst_ctr',
    
    # OT variations
    'hrs_ot': 'Hrs_OT',
    'Hrs_OT': 'Hrs_OT',
    'hrs_ot_emp': 'Hrs_OT_emp',
    'Hrs_OT_emp': 'Hrs_OT_emp',
    'hrs_ot_ctr': 'Hrs_OT_ctr',
    'Hrs_OT_ctr': 'Hrs_OT_ctr',
    
    # OT Assistant variations
    'hrs_otasst': 'Hrs_OTasst',
    'Hrs_OTasst': 'Hrs_OTasst',
    'hrs_otasst_emp': 'Hrs_OTasst_emp',
    'Hrs_OTasst_emp': 'Hrs_OTasst_emp',
    'hrs_otasst_ctr': 'Hrs_OTasst_ctr',
    'Hrs_OTasst_ctr': 'Hrs_OTasst_ctr',
    
    # OT Aide variations
    'hrs_otaide': 'Hrs_OTaide',
    'Hrs_OTaide': 'Hrs_OTaide',
    'hrs_otaide_emp': 'Hrs_OTaide_emp',
    'Hrs_OTaide_emp': 'Hrs_OTaide_emp',
    'hrs_otaide_ctr': 'Hrs_OTaide_ctr',
    'Hrs_OTaide_ctr': 'Hrs_OTaide_ctr',
    
    # PT variations
    'hrs_pt': 'Hrs_PT',
    'Hrs_PT': 'Hrs_PT',
    'hrs_pt_emp': 'Hrs_PT_emp',
    'Hrs_PT_emp': 'Hrs_PT_emp',
    'hrs_pt_ctr': 'Hrs_PT_ctr',
    'Hrs_PT_ctr': 'Hrs_PT_ctr',
    
    # PT Assistant variations
    'hrs_ptasst': 'Hrs_PTasst',
    'Hrs_PTasst': 'Hrs_PTasst',
    'hrs_ptasst_emp': 'Hrs_PTasst_emp',
    'Hrs_PTasst_emp': 'Hrs_PTasst_emp',
    'hrs_ptasst_ctr': 'Hrs_PTasst_ctr',
    'Hrs_PTasst_ctr': 'Hrs_PTasst_ctr',
    
    # PT Aide variations
    'hrs_ptaide': 'Hrs_PTaide',
    'Hrs_PTaide': 'Hrs_PTaide',
    'hrs_ptaide_emp': 'Hrs_PTaide_emp',
    'Hrs_PTaide_emp': 'Hrs_PTaide_emp',
    'hrs_ptaide_ctr': 'Hrs_PTaide_ctr',
    'Hrs_PTaide_ctr': 'Hrs_PTaide_ctr',
    
    # Respiratory Therapist variations
    'hrs_respther': 'Hrs_RespTher',
    'Hrs_RespTher': 'Hrs_RespTher',
    'hrs_respther_emp': 'Hrs_RespTher_emp',
    'Hrs_RespTher_emp': 'Hrs_RespTher_emp',
    'hrs_respther_ctr': 'Hrs_RespTher_ctr',
    'Hrs_RespTher_ctr': 'Hrs_RespTher_ctr',
    
    # Respiratory Technician variations
    'hrs_resptech': 'Hrs_RespTech',
    'Hrs_RespTech': 'Hrs_RespTech',
    'hrs_resptech_emp': 'Hrs_RespTech_emp',
    'Hrs_RespTech_emp': 'Hrs_RespTech_emp',
    'hrs_resptech_ctr': 'Hrs_RespTech_ctr',
    'Hrs_RespTech_ctr': 'Hrs_RespTech_ctr',
    
    # Speech Language Pathologist variations
    'hrs_spclangpath': 'Hrs_SpcLangPath',
    'Hrs_SpcLangPath': 'Hrs_SpcLangPath',
    'hrs_spclangpath_emp': 'Hrs_SpcLangPath_emp',
    'Hrs_SpcLangPath_emp': 'Hrs_SpcLangPath_emp',
    'hrs_spclangpath_ctr': 'Hrs_SpcLangPath_ctr',
    'Hrs_SpcLangPath_ctr': 'Hrs_SpcLangPath_ctr',
    
    # Therapeutic Recreation Specialist variations
    'hrs_therrecspec': 'Hrs_TherRecSpec',
    'Hrs_TherRecSpec': 'Hrs_TherRecSpec',
    'hrs_therrecspec_emp': 'Hrs_TherRecSpec_emp',
    'Hrs_TherRecSpec_emp': 'Hrs_TherRecSpec_emp',
    'hrs_therrecspec_ctr': 'Hrs_TherRecSpec_ctr',
    'Hrs_TherRecSpec_ctr': 'Hrs_TherRecSpec_ctr',
    
    # Qualified Activity Professional variations
    'hrs_qualactvprof': 'Hrs_QualActvProf',
    'Hrs_QualActvProf': 'Hrs_QualActvProf',
    'hrs_qualactvprof_emp': 'Hrs_QualActvProf_emp',
    'Hrs_QualActvProf_emp': 'Hrs_QualActvProf_emp',
    'hrs_qualactvprof_ctr': 'Hrs_QualActvProf_ctr',
    'Hrs_QualActvProf_ctr': 'Hrs_QualActvProf_ctr',
    
    # Other Activity variations
    'hrs_othactv': 'Hrs_OthActv',
    'Hrs_OthActv': 'Hrs_OthActv',
    'hrs_othactv_emp': 'Hrs_OthActv_emp',
    'Hrs_OthActv_emp': 'Hrs_OthActv_emp',
    'hrs_othactv_ctr': 'Hrs_OthActv_ctr',
    'Hrs_OthActv_ctr': 'Hrs_OthActv_ctr',
    
    # Qualified Social Worker variations
    'hrs_qualsocwrk': 'Hrs_QualSocWrk',
    'Hrs_QualSocWrk': 'Hrs_QualSocWrk',
    'hrs_qualsocwrk_emp': 'Hrs_QualSocWrk_emp',
    'Hrs_QualSocWrk_emp': 'Hrs_QualSocWrk_emp',
    'hrs_qualsocwrk_ctr': 'Hrs_QualSocWrk_ctr',
    'Hrs_QualSocWrk_ctr': 'Hrs_QualSocWrk_ctr',
    
    # Other Social Worker variations
    'hrs_othsocwrk': 'Hrs_OthSocWrk',
    'Hrs_OthSocWrk': 'Hrs_OthSocWrk',
    'hrs_othsocwrk_emp': 'Hrs_OthSocWrk_emp',
    'Hrs_OthSocWrk_emp': 'Hrs_OthSocWrk_emp',
    'hrs_othsocwrk_ctr': 'Hrs_OthSocWrk_ctr',
    'Hrs_OthSocWrk_ctr': 'Hrs_OthSocWrk_ctr',
    
    # Mental Health Service variations
    'hrs_mhsvc': 'Hrs_MHSvc',
    'Hrs_MHSvc': 'Hrs_MHSvc',
    'hrs_mhsvc_emp': 'Hrs_MHSvc_emp',
    'Hrs_MHSvc_emp': 'Hrs_MHSvc_emp',
    'hrs_mhsvc_ctr': 'Hrs_MHSvc_ctr',
    'Hrs_MHSvc_ctr': 'Hrs_MHSvc_ctr'
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
    output_dir = Path('standardized_NonNurse')
    output_dir.mkdir(exist_ok=True)
    
    # Get all PBJ CSV files from NonNursecsv directory
    pbj_files = glob.glob('NonNursecsv/PBJ_dailynonnursestaffing_*.csv')
    
    # Sort files by date and get only the latest one
    pbj_files.sort()
    latest_file = pbj_files[-1]  # Get the last (latest) file
    
    print(f"\nProcessing only the latest file: {latest_file}")
    
    # Track statistics
    processed_files = 0
    changed_files = 0
    errors = []
    
    print(f"\nStarting processing at {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        # Create output filename
        output_file = output_dir / Path(latest_file).name
        
        # Try different encodings
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                # Read PROVNUM as string from the start
                df = pd.read_csv(latest_file, encoding=encoding, low_memory=False, dtype={'PROVNUM': str})
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            raise Exception("Could not read file with any of the attempted encodings")
        
        # Standardize column names
        df_standardized, changes = standardize_column_names(df, latest_file)
        
        if changes:
            # Save standardized file with UTF-8 encoding
            df_standardized.to_csv(output_file, index=False, encoding='utf-8')
            print(f"\nProcessed {latest_file}")
            print("Changes made:")
            for change in changes:
                print(f"  - {change}")
            changed_files += 1
        else:
            print(f"No changes needed for {latest_file}")
        
        processed_files += 1
        
    except Exception as e:
        error_msg = f"Error processing {latest_file}: {str(e)}"
        print(error_msg)
        errors.append(error_msg)
    
    # Print summary
    print(f"\nProcessing completed at {datetime.now().strftime('%H:%M:%S')}")
    print(f"\nSummary:")
    print(f"Files processed: {processed_files}")
    print(f"  - Files changed: {changed_files}")
    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for error in errors:
            print(f"  - {error}")

if __name__ == "__main__":
    process_pbj_files() 