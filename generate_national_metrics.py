import pandas as pd
import numpy as np
from datetime import datetime
import calendar

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

def generate_national_metrics():
    """Generate national quarterly metrics from historical PBJ report data."""
    print("Loading historical PBJ report data...")
    
    # Read the historical PBJ report
    df = pd.read_csv('national_quarterly_metrics.csv')
    print("\nColumns in national_quarterly_metrics.csv:")
    for col in df.columns:
        print(f"- {col}")
    
    # Calculate workdays for each quarter
    df['Year'] = df['CY_Qtr'].str[:4].astype(int)
    df['Quarter'] = df['CY_Qtr'].str[5:].astype(int)
    
    # Print workdays calculation for each quarter
    print("\nCalculating workdays for each quarter:")
    workdays_by_quarter = {}
    for year in df['Year'].unique():
        for quarter in range(1, 5):
            workdays = get_workdays_in_quarter(year, quarter)
            workdays_by_quarter[f"{year}Q{quarter}"] = workdays
            print(f"{year}Q{quarter}: {workdays} workdays")
    
    # Calculate total resident days (using MDScensus directly)
    df['Total_Resident_Days'] = df['MDScensus']
    
    # Calculate total hours for each staff type
    # RN group (RN + RN Admin + RN DON)
    df['Total_RN_Hours'] = (df['Hrs_RN'] + 
                           df['Hrs_RNadmin'] + 
                           df['Hrs_RNDON'])
    
    # LPN group (LPN + LPN Admin)
    df['Total_LPN_Hours'] = (df['Hrs_LPN'] + 
                            df['Hrs_LPNadmin'])
    
    # Nurse Aide group (CNA + NAtr + MedAide)
    df['Total_Nurse_Aide_Hours'] = (df['Hrs_CNA'] + 
                                   df['Hrs_NAtrn'] + 
                                   df['Hrs_MedAide'])
    
    # Calculate total contract hours
    contract_columns = [col for col in df.columns if col.endswith('_ctr')]
    df['Total_Contract_Hours'] = df[contract_columns].sum(axis=1)
    
    # Calculate total nurse hours (all staff types)
    df['Total_Nurse_Hours'] = df['Total_RN_Hours'] + df['Total_LPN_Hours'] + df['Total_Nurse_Aide_Hours']
    
    # Calculate HPRD metrics
    df['Total_HPRD'] = df['Total_Nurse_Hours'] / df['Total_Resident_Days']
    df['RN_HPRD'] = df['Total_RN_Hours'] / df['Total_Resident_Days']
    df['LPN_HPRD'] = df['Total_LPN_Hours'] / df['Total_Resident_Days']
    df['Nurse_Aide_HPRD'] = df['Total_Nurse_Aide_Hours'] / df['Total_Resident_Days']
    
    # Calculate contract staff percentage
    df['Contract_Staff_Percentage'] = (df['Total_Contract_Hours'] / df['Total_Nurse_Hours']) * 100
    
    # Print debug info for Q4 2024
    q4_2024 = df[df['CY_Qtr'] == '2024Q4'].iloc[0]
    print("\nDebug info for Q4 2024:")
    print(f"Total RN Hours: {q4_2024['Total_RN_Hours']:,.2f}")
    print(f"Total LPN Hours: {q4_2024['Total_LPN_Hours']:,.2f}")
    print(f"Total Nurse Aide Hours: {q4_2024['Total_Nurse_Aide_Hours']:,.2f}")
    print(f"Total Nurse Hours: {q4_2024['Total_Nurse_Hours']:,.2f}")
    print(f"Total Resident Days: {q4_2024['Total_Resident_Days']:,.2f}")
    print(f"Total HPRD: {q4_2024['Total_HPRD']:.2f}")
    
    # Select and rename columns for output
    output_columns = {
        'CY_Qtr': 'CY_Qtr',
        'Total_Resident_Days': 'Total_Resident_Days',
        'Total_HPRD': 'Total_HPRD',
        'RN_HPRD': 'RN_HPRD',
        'LPN_HPRD': 'LPN_HPRD',
        'Nurse_Aide_HPRD': 'Nurse_Aide_HPRD',
        'Contract_Staff_Percentage': 'Contract_Staff_Percentage'
    }
    
    national_metrics = df[list(output_columns.keys())].rename(columns=output_columns)
    
    # Sort by quarter
    national_metrics = national_metrics.sort_values('CY_Qtr')
    
    # Save to CSV
    print("\nSaving national quarterly metrics...")
    national_metrics.to_csv('national_quarterly_metrics.csv', index=False)
    print("Done! Generated national_quarterly_metrics.csv")
    
    # Print summary
    print("\nSummary of generated metrics:")
    print(f"Time period: {national_metrics['CY_Qtr'].min()} to {national_metrics['CY_Qtr'].max()}")
    print(f"Number of quarters: {len(national_metrics)}")
    print(f"Number of metrics: {len(national_metrics.columns) - 1}")  # Excluding CY_Qtr
    print("\nColumns included:")
    for col in national_metrics.columns:
        print(f"- {col}")

if __name__ == "__main__":
    generate_national_metrics() 