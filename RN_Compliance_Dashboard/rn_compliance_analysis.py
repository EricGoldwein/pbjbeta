import pandas as pd
import os
import glob
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_provnum(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure PROVNUM is a 6-digit string with leading zeros."""
    if 'PROVNUM' in df.columns:
        df['PROVNUM'] = df['PROVNUM'].astype(str)
        df['PROVNUM'] = df['PROVNUM'].apply(lambda x: x.split('e')[0] if 'e' in x.lower() else x)
        df['PROVNUM'] = df['PROVNUM'].apply(lambda x: x.split('.')[0])
        df['PROVNUM'] = df['PROVNUM'].str.zfill(6)
    return df

def calculate_daily_rn_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate total daily RN hours (RN + RN Admin + RN DON)."""
    # Sum all RN hours
    rn_columns = ['Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_RN']
    
    # Check which columns exist in the dataframe
    available_rn_columns = [col for col in rn_columns if col in df.columns]
    
    if not available_rn_columns:
        logger.warning("No RN columns found in data")
        df['Total_RN_Hours'] = 0
        return df
    
    # Fill NaN values with 0 for RN columns
    for col in available_rn_columns:
        df[col] = df[col].fillna(0)
    
    # Calculate total RN hours
    df['Total_RN_Hours'] = df[available_rn_columns].sum(axis=1)
    
    return df

def calculate_california_compliance(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate California-specific compliance metrics."""
    # Calculate Total Nurse HPRD (RN + LPN + CNA)
    nurse_columns = ['Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_RN', 'Hrs_LPN', 'Hrs_CNA']
    available_nurse_columns = [col for col in nurse_columns if col in df.columns]
    
    # Fill NaN values with 0
    for col in available_nurse_columns:
        df[col] = df[col].fillna(0)
    
    # Calculate Total Nurse HPRD
    df['Total_Nurse_HPRD'] = df[available_nurse_columns].sum(axis=1) / df['MDScensus']
    df['Total_Nurse_HPRD'] = df['Total_Nurse_HPRD'].fillna(0)
    
    # Calculate CNA HPRD
    cna_columns = ['Hrs_CNA']
    available_cna_columns = [col for col in cna_columns if col in df.columns]
    
    if available_cna_columns:
        df['CNA_HPRD'] = df[available_cna_columns].sum(axis=1) / df['MDScensus']
        df['CNA_HPRD'] = df['CNA_HPRD'].fillna(0)
    else:
        df['CNA_HPRD'] = 0
    
    # Calculate California compliance
    # CA compliance: Total Nurse HPRD >= 3.5 AND CNA HPRD >= 2.4
    df['CA_Compliant'] = ((df['Total_Nurse_HPRD'] >= 3.5) & (df['CNA_HPRD'] >= 2.4)).astype(int)
    
    # Count days out of compliance for each threshold
    df['Days_Below_Total_Nurse_Threshold'] = (df['Total_Nurse_HPRD'] < 3.5).astype(int)
    df['Days_Below_CNA_Threshold'] = (df['CNA_HPRD'] < 2.4).astype(int)
    
    return df

def analyze_rn_compliance_by_quarter(file_path: str) -> pd.DataFrame:
    """Analyze RN compliance for a single quarter file."""
    logger.info(f"Processing {file_path}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, low_memory=False)
        
        # Format PROVNUM
        df = format_provnum(df)
        
        # Calculate daily RN hours
        df = calculate_daily_rn_hours(df)
        
        # Calculate California compliance metrics
        df = calculate_california_compliance(df)
        
        # Filter out days with census = 0
        df = df[df['MDScensus'] > 0]
        
        # Create compliance indicator (1 if RN hours < 8, 0 otherwise)
        df['RN_Non_Compliant'] = (df['Total_RN_Hours'] < 8).astype(int)
        
        # Group by facility and calculate compliance metrics
        compliance_summary = df.groupby(['PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME', 'CITY']).agg({
            'MDScensus': 'mean',  # Average daily census
            'Total_RN_Hours': 'mean',  # Average daily RN hours
            'RN_Non_Compliant': 'sum',  # Count of non-compliant days
            'WorkDate': 'count',  # Total days reported
            'Total_Nurse_HPRD': 'mean',  # Average Total Nurse HPRD
            'CNA_HPRD': 'mean',  # Average CNA HPRD
            'CA_Compliant': 'sum',  # Count of CA compliant days
            'Days_Below_Total_Nurse_Threshold': 'sum',  # Days below 3.5 threshold
            'Days_Below_CNA_Threshold': 'sum'  # Days below 2.4 threshold
        }).reset_index()
        
        # Rename columns for clarity
        compliance_summary.columns = [
            'PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME', 'CITY',
            'Avg_Daily_Census', 'Avg_Daily_RN_Hours', 'Days_Non_Compliant', 'Total_Days_Reported',
            'Avg_Total_Nurse_HPRD', 'Avg_CNA_HPRD', 'Days_CA_Compliant', 
            'Days_Below_Total_Nurse_Threshold', 'Days_Below_CNA_Threshold'
        ]
        
        # Add quarter information
        quarter = os.path.basename(file_path).replace('PBJ_dailynursestaffing_', '').replace('.csv', '')
        compliance_summary['CY_Qtr'] = quarter
        
        return compliance_summary
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return pd.DataFrame()

def get_all_quarter_files() -> List[str]:
    """Get all standardized PBJ quarter files."""
    pattern = os.path.join('standardized_PBJ', 'PBJ_dailynursestaffing_CY*.csv')
    files = glob.glob(pattern)
    return sorted(files)

def main():
    """Main function to analyze RN compliance across all quarters."""
    logger.info("Starting RN compliance analysis")
    
    # Get all quarter files
    quarter_files = get_all_quarter_files()
    logger.info(f"Found {len(quarter_files)} quarter files to process")
    
    # Process each quarter
    all_compliance_data = []
    
    for file_path in quarter_files:
        quarter_data = analyze_rn_compliance_by_quarter(file_path)
        if not quarter_data.empty:
            all_compliance_data.append(quarter_data)
            logger.info(f"Completed processing {os.path.basename(file_path)}")
    
    if not all_compliance_data:
        logger.error("No data was processed successfully")
        return
    
    # Combine all quarters
    combined_data = pd.concat(all_compliance_data, ignore_index=True)
    
    # Sort by quarter and PROVNUM
    combined_data = combined_data.sort_values(['CY_Qtr', 'PROVNUM'])
    
    # Reorder columns for final output
    final_columns = [
        'CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME', 'CITY',
        'Avg_Daily_Census', 'Avg_Daily_RN_Hours', 'Days_Non_Compliant', 'Total_Days_Reported',
        'Avg_Total_Nurse_HPRD', 'Avg_CNA_HPRD', 'Days_CA_Compliant', 
        'Days_Below_Total_Nurse_Threshold', 'Days_Below_CNA_Threshold'
    ]
    
    # Ensure all columns exist
    for col in final_columns:
        if col not in combined_data.columns:
            combined_data[col] = ''
    
    final_output = combined_data[final_columns]
    
    # Save to CSV
    output_file = 'rn_compliance_analysis.csv'
    final_output.to_csv(output_file, index=False)
    
    logger.info(f"RN compliance analysis completed. Results saved to {output_file}")
    logger.info(f"Total records: {len(final_output)}")
    logger.info(f"Quarters processed: {final_output['CY_Qtr'].nunique()}")
    logger.info(f"Facilities processed: {final_output['PROVNUM'].nunique()}")
    
    # Print summary statistics
    print("\n=== RN Compliance Analysis Summary ===")
    print(f"Total facilities analyzed: {final_output['PROVNUM'].nunique()}")
    print(f"Total quarters analyzed: {final_output['CY_Qtr'].nunique()}")
    print(f"Total facility-quarter combinations: {len(final_output)}")
    
    # Calculate compliance statistics
    total_non_compliant_days = final_output['Days_Non_Compliant'].sum()
    total_days_reported = final_output['Total_Days_Reported'].sum()
    compliance_rate = ((total_days_reported - total_non_compliant_days) / total_days_reported * 100) if total_days_reported > 0 else 0
    
    print(f"Total days analyzed: {total_days_reported:,}")
    print(f"Total non-compliant days: {total_non_compliant_days:,}")
    print(f"Overall compliance rate: {compliance_rate:.2f}%")
    
    # Show facilities with most non-compliant days
    print("\n=== Top 10 Facilities by Non-Compliant Days ===")
    top_non_compliant = final_output.nlargest(10, 'Days_Non_Compliant')[['PROVNUM', 'PROVNAME', 'STATE', 'Days_Non_Compliant']]
    print(top_non_compliant.to_string(index=False))
    
    # Show California-specific statistics
    ca_data = final_output[final_output['STATE'] == 'CA']
    if not ca_data.empty:
        print("\n=== California Compliance Statistics ===")
        print(f"California facilities analyzed: {len(ca_data)}")
        
        # Calculate CA compliance rates
        total_ca_days = ca_data['Total_Days_Reported'].sum()
        total_ca_compliant_days = ca_data['Days_CA_Compliant'].sum()
        ca_compliance_rate = (total_ca_compliant_days / total_ca_days * 100) if total_ca_days > 0 else 0
        
        print(f"Total California days analyzed: {total_ca_days:,}")
        print(f"Total California compliant days: {total_ca_compliant_days:,}")
        print(f"California compliance rate: {ca_compliance_rate:.2f}%")
        
        # Show CA facilities with most non-compliant days
        print("\n=== Top 10 California Facilities by Non-Compliant Days ===")
        ca_non_compliant = ca_data.nlargest(10, 'Days_Below_Total_Nurse_Threshold')[['PROVNUM', 'PROVNAME', 'Days_Below_Total_Nurse_Threshold', 'Days_Below_CNA_Threshold']]
        print(ca_non_compliant.to_string(index=False))

if __name__ == "__main__":
    main()
