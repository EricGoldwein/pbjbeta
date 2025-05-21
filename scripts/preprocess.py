import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import re

# CMS Region mapping
CMS_REGION_MAP = {
    'Region 1': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
    'Region 2': ['NJ', 'NY', 'PR', 'VI'],
    'Region 3': ['DE', 'DC', 'MD', 'PA', 'VA', 'WV'],
    'Region 4': ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
    'Region 5': ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
    'Region 6': ['AR', 'LA', 'NM', 'OK', 'TX'],
    'Region 7': ['IA', 'KS', 'MO', 'NE'],
    'Region 8': ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
    'Region 9': ['AZ', 'CA', 'HI', 'NV'],
    'Region 10': ['AK', 'ID', 'OR', 'WA']
}

def validate_ccn(ccn: str) -> bool:
    """Validate CCN format (6 characters, alphanumeric)."""
    return bool(re.match(r'^[A-Z0-9]{6}$', str(ccn)))

def format_ccn(ccn: str) -> str:
    """Format CCN to ensure 6 characters with leading zeros."""
    return str(ccn).zfill(6)

def get_cms_region(state_abbr: str) -> str:
    """Get CMS region for a given state abbreviation."""
    for region, states in CMS_REGION_MAP.items():
        if state_abbr in states:
            return region
    return "Unknown"

def load_nurse_staffing(file_path: str) -> pd.DataFrame:
    """Load and preprocess nurse staffing data."""
    # Read CSV with PROVNUM as string to preserve leading zeros
    df = pd.read_csv(file_path, dtype={'PROVNUM': str})
    
    # Format CCNs
    df['PROVNUM'] = df['PROVNUM'].apply(format_ccn)
    
    # Validate CCN format
    invalid_ccns = df[~df['PROVNUM'].apply(validate_ccn)]['PROVNUM'].unique()
    if len(invalid_ccns) > 0:
        print(f"Warning: Found {len(invalid_ccns)} invalid CCNs: {invalid_ccns}")
    
    # Convert date column to datetime
    df['WorkDate'] = pd.to_datetime(df['WorkDate'])
    
    # Calculate total nurse staff hours
    df['Total_Nurse_Staff_Hours'] = df[[
        'Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_RN',
        'Hrs_LPNadmin', 'Hrs_LPN',
        'Hrs_CNA', 'Hrs_NAtrn', 'Hrs_MedAide'
    ]].sum(axis=1)
    
    # Calculate RN hours
    df['Total_RN_Hours'] = df[['Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_RN']].sum(axis=1)
    
    # Calculate LPN hours
    df['Total_LPN_Hours'] = df[['Hrs_LPNadmin', 'Hrs_LPN']].sum(axis=1)
    
    # Calculate CNA hours
    df['Total_CNA_Hours'] = df[['Hrs_CNA', 'Hrs_NAtrn', 'Hrs_MedAide']].sum(axis=1)
    
    # Handle zero or missing MDS census
    df['MDScensus'] = df['MDScensus'].replace(0, np.nan)
    
    # Calculate HPRD metrics
    df['Total_Nurse_HPRD'] = df['Total_Nurse_Staff_Hours'] / df['MDScensus']
    df['RN_HPRD'] = df['Total_RN_Hours'] / df['MDScensus']
    df['LPN_HPRD'] = df['Total_LPN_Hours'] / df['MDScensus']
    df['CNA_HPRD'] = df['Total_CNA_Hours'] / df['MDScensus']
    
    # Print calculation verification
    print("\nCalculation Verification for First Record:")
    first_record = df.iloc[0]
    print(f"Facility: {first_record['PROVNAME']} (CCN: {first_record['PROVNUM']})")
    print(f"Date: {first_record['WorkDate']}")
    print(f"MDS Census: {first_record['MDScensus']}")
    print("\nHours Breakdown:")
    print(f"RN (Direct + Admin + DON): {first_record['Hrs_RN'] + first_record['Hrs_RNadmin'] + first_record['Hrs_RNDON']}")
    print(f"LPN (Direct + Admin): {first_record['Hrs_LPN'] + first_record['Hrs_LPNadmin']}")
    print(f"CNA (Including Med Aide): {first_record['Hrs_CNA'] + first_record['Hrs_NAtrn'] + first_record['Hrs_MedAide']}")
    print(f"Total Hours: {first_record['Total_Nurse_Staff_Hours']}")
    print("\nHPRD Calculations:")
    print(f"Total Nurse HPRD: {first_record['Total_Nurse_HPRD']:.2f}")
    print(f"RN HPRD: {first_record['RN_HPRD']:.2f}")
    print(f"LPN HPRD: {first_record['LPN_HPRD']:.2f}")
    print(f"CNA HPRD: {first_record['CNA_HPRD']:.2f}")
    
    return df

def load_provider_info(file_path: str) -> pd.DataFrame:
    """Load and preprocess provider information."""
    # Read CSV with PROVNUM as string to preserve leading zeros
    df = pd.read_csv(file_path, dtype={'PROVNUM': str})
    
    # Format CCNs
    df['PROVNUM'] = df['PROVNUM'].apply(format_ccn)
    
    # Validate CCN format
    invalid_ccns = df[~df['PROVNUM'].apply(validate_ccn)]['PROVNUM'].unique()
    if len(invalid_ccns) > 0:
        print(f"Warning: Found {len(invalid_ccns)} invalid CCNs: {invalid_ccns}")
    
    # Add CMS region
    df['CMS_Region'] = df['STATE'].apply(get_cms_region)
    
    return df

def merge_datasets(nurse_df: pd.DataFrame, provider_df: pd.DataFrame) -> pd.DataFrame:
    """Merge nurse staffing data with provider information."""
    return pd.merge(
        nurse_df,
        provider_df,
        on='PROVNUM',
        how='left',
        suffixes=('', '_provider')
    )

def calculate_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily metrics for each facility."""
    metrics = df.groupby(['PROVNUM', 'PROVNAME', 'STATE', 'WorkDate']).agg({
        'Total_Nurse_HPRD': 'mean',
        'RN_HPRD': 'mean',
        'LPN_HPRD': 'mean',
        'CNA_HPRD': 'mean',
        'MDScensus': 'mean'
    }).reset_index()
    
    return metrics

def main():
    # Define file paths
    data_dir = Path('data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    # Load and process data
    print("Loading nurse staffing data...")
    nurse_df = load_nurse_staffing(raw_dir / 'PBJ_dailynursestaffing_CY2024Q3.csv')
    
    print("\nLoading provider information...")
    provider_df = load_provider_info(raw_dir / 'NH_ProviderInfo_Mar2025.csv')
    
    # Merge datasets
    print("\nMerging datasets...")
    merged_df = merge_datasets(nurse_df, provider_df)
    
    # Calculate daily metrics
    print("\nCalculating daily metrics...")
    daily_metrics = calculate_daily_metrics(merged_df)
    
    # Save processed data
    print("\nSaving processed data...")
    daily_metrics.to_csv(processed_dir / 'daily_metrics.csv', index=False)
    merged_df.to_csv(processed_dir / 'merged_data.csv', index=False)
    
    print("\nProcessing complete!")

if __name__ == '__main__':
    main() 