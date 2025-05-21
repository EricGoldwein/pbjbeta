import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

def calculate_hprd_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate percentage of providers meeting HPRD thresholds."""
    thresholds = {
        'Total_Nurse_HPRD': [4.1, 3.48],
        'RN_HPRD': [0.55, 0.45]  # Example thresholds for RN HPRD
    }
    
    results = []
    for metric, threshold_values in thresholds.items():
        for threshold in threshold_values:
            pct_meeting = (df[metric] >= threshold).mean() * 100
            results.append({
                'Metric': metric,
                'Threshold': threshold,
                'Percent_Meeting': pct_meeting
            })
    
    return pd.DataFrame(results)

def calculate_quarterly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate quarterly summary statistics."""
    quarterly = df.groupby(['PROVNUM', 'PROVNAME', 'STATE', 'CMS_Region']).agg({
        'Total_Nurse_HPRD': ['mean', 'std', 'min', 'max'],
        'RN_HPRD': ['mean', 'std', 'min', 'max'],
        'MDScensus': ['mean', 'min', 'max']
    }).reset_index()
    
    # Flatten column names
    quarterly.columns = ['_'.join(col).strip('_') for col in quarterly.columns.values]
    
    return quarterly

def generate_facility_report(df: pd.DataFrame, provnum: str, output_dir: Path) -> None:
    """Generate detailed report for a specific facility."""
    facility_data = df[df['PROVNUM'] == provnum].copy()
    
    if facility_data.empty:
        print(f"No data found for facility {provnum}")
        return
    
    # Calculate facility-specific metrics
    facility_summary = calculate_quarterly_summary(facility_data)
    facility_thresholds = calculate_hprd_thresholds(facility_data)
    
    # Save reports
    facility_summary.to_csv(output_dir / f'facility_{provnum}_summary.csv', index=False)
    facility_thresholds.to_csv(output_dir / f'facility_{provnum}_thresholds.csv', index=False)

def generate_state_report(df: pd.DataFrame, state: str, output_dir: Path) -> None:
    """Generate summary report for a specific state."""
    state_data = df[df['STATE'] == state].copy()
    
    if state_data.empty:
        print(f"No data found for state {state}")
        return
    
    # Calculate state-level metrics
    state_summary = calculate_quarterly_summary(state_data)
    state_thresholds = calculate_hprd_thresholds(state_data)
    
    # Save reports
    state_summary.to_csv(output_dir / f'state_{state}_summary.csv', index=False)
    state_thresholds.to_csv(output_dir / f'state_{state}_thresholds.csv', index=False)

def generate_national_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate national summary report."""
    # Calculate national metrics
    national_summary = calculate_quarterly_summary(df)
    national_thresholds = calculate_hprd_thresholds(df)
    
    # Save reports
    national_summary.to_csv(output_dir / 'national_summary.csv', index=False)
    national_thresholds.to_csv(output_dir / 'national_thresholds.csv', index=False)

def main():
    # Define file paths
    data_dir = Path('data')
    processed_dir = data_dir / 'processed'
    output_dir = Path('outputs')
    
    # Load processed data
    merged_data = pd.read_csv(processed_dir / 'merged_data.csv')
    merged_data['WorkDate'] = pd.to_datetime(merged_data['WorkDate'])
    
    # Generate reports for each level
    # Facility level (example for first few facilities)
    for provnum in merged_data['PROVNUM'].unique()[:5]:
        generate_facility_report(merged_data, provnum, output_dir / 'facility_reports')
    
    # State level
    for state in merged_data['STATE'].unique():
        generate_state_report(merged_data, state, output_dir / 'state_reports')
    
    # National level
    generate_national_report(merged_data, output_dir / 'national_reports')

if __name__ == '__main__':
    main() 