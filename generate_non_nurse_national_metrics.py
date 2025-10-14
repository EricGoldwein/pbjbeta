import pandas as pd
import numpy as np
from datetime import datetime
import calendar
import glob
import os

def get_workdays_in_quarter(year, quarter):
    """Calculate number of workdays in a given quarter."""
    # Define quarter start and end dates
    quarter_starts = {
        1: (1, 1),    # January 1
        2: (4, 1),    # April 1
        3: (7, 1),    # July 1
        4: (10, 1)    # October 1
    }
    quarter_ends = {
        1: (3, 31),   # March 31
        2: (6, 30),   # June 30
        3: (9, 30),   # September 30
        4: (12, 31)   # December 31
    }
    
    start_month, start_day = quarter_starts[quarter]
    end_month, end_day = quarter_ends[quarter]
    
    start_date = datetime(year, start_month, start_day)
    end_date = datetime(year, end_month, end_day)
    
    # Calculate business days (excluding weekends)
    business_days = pd.bdate_range(start=start_date, end=end_date)
    
    # Validate leap year handling
    if calendar.isleap(year):
        print(f"Leap year {year} detected for quarter {quarter}")
        # February should have 29 days in leap years
        if quarter == 1:  # Q1 includes February
            feb_days = calendar.monthrange(year, 2)[1]
            print(f"February has {feb_days} days in {year}")
    
    return len(business_days)

def process_file(file_path):
    """Process a single file and return aggregated metrics."""
    print(f"Processing {os.path.basename(file_path)}...")
    
    # Read file with low_memory=False to avoid dtype warnings
    df = pd.read_csv(file_path, low_memory=False)
    
    # Calculate workdays for the quarter
    year = int(df['CY_Qtr'].iloc[0][:4])
    quarter = int(df['CY_Qtr'].iloc[0][5:])
    workdays = get_workdays_in_quarter(year, quarter)
    
    # Calculate total resident days
    df['Total_Resident_Days'] = df['MDScensus']
    
    # Calculate total hours for each staff type
    df['Total_Admin_Hours'] = df['Hrs_Admin']
    df['Total_MedDir_Hours'] = df['Hrs_MedDir']
    df['Total_OthMD_Hours'] = df['Hrs_OthMD']
    df['Total_AdvPractice_Hours'] = df['Hrs_PA'] + df['Hrs_NP']
    df['Total_ClinNrsSpec_Hours'] = df['Hrs_ClinNrsSpec']
    df['Total_Pharmacy_Hours'] = df['Hrs_Pharmacist']
    df['Total_Dietary_Hours'] = df['Hrs_Dietician'] + df['Hrs_FeedAsst']
    df['Total_Therapy_Hours'] = (df['Hrs_OT'] + df['Hrs_OTasst'] + df['Hrs_OTaide'] + 
                                df['Hrs_PT'] + df['Hrs_PTasst'] + df['Hrs_PTaide'] + 
                                df['Hrs_RespTher'] + df['Hrs_RespTech'])
    df['Total_Speech_Hours'] = df['Hrs_SpcLangPath']
    df['Total_Activity_Hours'] = (df['Hrs_TherRecSpec'] + 
                                 df['Hrs_QualActvProf'] + 
                                 df['Hrs_OthActv'])
    df['Total_SocialWork_Hours'] = df['Hrs_QualSocWrk'] + df['Hrs_OthSocWrk']
    df['Total_MentalHealth_Hours'] = df['Hrs_MHSvc']
    
    # Calculate total contract hours
    contract_columns = [col for col in df.columns if col.endswith('_ctr')]
    df['Total_Contract_Hours'] = df[contract_columns].sum(axis=1)
    
    # Calculate total non-nurse hours
    df['Total_NonNurse_Hours'] = (df['Total_Admin_Hours'] + 
                                 df['Total_MedDir_Hours'] + 
                                 df['Total_OthMD_Hours'] + 
                                 df['Total_AdvPractice_Hours'] + 
                                 df['Total_ClinNrsSpec_Hours'] + 
                                 df['Total_Pharmacy_Hours'] + 
                                 df['Total_Dietary_Hours'] + 
                                 df['Total_Therapy_Hours'] + 
                                 df['Total_Speech_Hours'] + 
                                 df['Total_Activity_Hours'] + 
                                 df['Total_SocialWork_Hours'] + 
                                 df['Total_MentalHealth_Hours'])
    
    # Calculate HPRD metrics
    df['Total_NonNurse_HPRD'] = df['Total_NonNurse_Hours'] / df['Total_Resident_Days']
    df['Admin_HPRD'] = df['Total_Admin_Hours'] / df['Total_Resident_Days']
    df['MedDir_HPRD'] = df['Total_MedDir_Hours'] / df['Total_Resident_Days']
    df['AdvPractice_HPRD'] = df['Total_AdvPractice_Hours'] / df['Total_Resident_Days']
    df['Pharmacy_HPRD'] = df['Total_Pharmacy_Hours'] / df['Total_Resident_Days']
    df['Dietary_HPRD'] = df['Total_Dietary_Hours'] / df['Total_Resident_Days']
    df['Therapy_HPRD'] = df['Total_Therapy_Hours'] / df['Total_Resident_Days']
    df['Speech_HPRD'] = df['Total_Speech_Hours'] / df['Total_Resident_Days']
    df['Activity_HPRD'] = df['Total_Activity_Hours'] / df['Total_Resident_Days']
    df['SocialWork_HPRD'] = df['Total_SocialWork_Hours'] / df['Total_Resident_Days']
    df['MentalHealth_HPRD'] = df['Total_MentalHealth_Hours'] / df['Total_Resident_Days']
    
    # Calculate contract staff percentage
    df['Contract_Staff_Percentage'] = (df['Total_Contract_Hours'] / df['Total_NonNurse_Hours']) * 100
    
    # Aggregate by quarter
    metrics = df.groupby('CY_Qtr').agg({
        'Total_Resident_Days': 'sum',
        'Total_NonNurse_HPRD': 'mean',
        'Admin_HPRD': 'mean',
        'MedDir_HPRD': 'mean',
        'AdvPractice_HPRD': 'mean',
        'Pharmacy_HPRD': 'mean',
        'Dietary_HPRD': 'mean',
        'Therapy_HPRD': 'mean',
        'Speech_HPRD': 'mean',
        'Activity_HPRD': 'mean',
        'SocialWork_HPRD': 'mean',
        'MentalHealth_HPRD': 'mean',
        'Contract_Staff_Percentage': 'mean'
    }).reset_index()
    
    return metrics

def generate_non_nurse_national_metrics():
    """Generate national quarterly metrics for non-nurse staffing."""
    print("Loading standardized non-nurse PBJ files...")
    
    # Get list of standardized non-nurse files
    standardized_files = glob.glob('standardized_NonNurse/*.csv')
    if not standardized_files:
        raise FileNotFoundError("No standardized non-nurse files found in 'standardized_NonNurse' directory")
    
    # Process files one by one and combine results
    all_metrics = []
    for file in standardized_files:
        metrics = process_file(file)
        all_metrics.append(metrics)
    
    # Combine all metrics
    national_metrics = pd.concat(all_metrics, ignore_index=True)
    
    # Sort by quarter
    national_metrics = national_metrics.sort_values('CY_Qtr')
    
    # Save to CSV
    print("\nSaving national quarterly non-nurse metrics...")
    national_metrics.to_csv('national_quarterly_non_nurse_metrics.csv', index=False)
    print("Done! Generated national_quarterly_non_nurse_metrics.csv")
    
    # Print summary
    print("\nSummary of generated metrics:")
    print(f"Time period: {national_metrics['CY_Qtr'].min()} to {national_metrics['CY_Qtr'].max()}")
    print(f"Number of quarters: {len(national_metrics)}")
    print(f"Number of metrics: {len(national_metrics.columns) - 1}")  # Excluding CY_Qtr
    print("\nColumns included:")
    for col in national_metrics.columns:
        print(f"- {col}")

if __name__ == "__main__":
    generate_non_nurse_national_metrics() 