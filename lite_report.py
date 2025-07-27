import pandas as pd
import os

def format_provnum(df):
    """Ensure PROVNUM is a 6-digit string with leading zeros."""
    if 'PROVNUM' in df.columns:
        # First convert to string
        df['PROVNUM'] = df['PROVNUM'].astype(str)
        # Remove any scientific notation
        df['PROVNUM'] = df['PROVNUM'].apply(lambda x: x.split('e')[0] if 'e' in x.lower() else x)
        # Remove any decimal points and everything after
        df['PROVNUM'] = df['PROVNUM'].apply(lambda x: x.split('.')[0])
        # Pad with leading zeros
        df['PROVNUM'] = df['PROVNUM'].str.zfill(6)
        return True
    return False

def generate_lite_metrics():
    # Read the existing metrics files
    facility_metrics = pd.read_csv('facility_quarterly_metrics.csv', low_memory=False)
    state_metrics = pd.read_csv('state_quarterly_metrics.csv', low_memory=False)
    
    # Format PROVNUMs
    format_provnum(facility_metrics)
    
    # Create facility lite metrics with new column order, including fields needed for calculations
    facility_lite = facility_metrics[[
        'CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME',
        'Total_Nurse_HPRD', 'Contract_Percentage', 'avg_daily_census',
        'total_resident_days', 'days_reported', 'MDScensus'
    ]].copy()
    
    # Sort by quarter and PROVNUM
    facility_lite = facility_lite.sort_values(['CY_Qtr', 'PROVNUM'])
    
    # Create state lite metrics with new column order
    state_lite = state_metrics[[
        'CY_Qtr', 'STATE', 'facility_count', 'avg_daily_census',
        'Total_Nurse_HPRD', 'Contract_Percentage'
    ]].copy()
    
    # Calculate statewide census (total census across all facilities in state)
    state_census_data = []
    for quarter in facility_lite['CY_Qtr'].unique():
        quarter_facilities = facility_lite[facility_lite['CY_Qtr'] == quarter]
        for state in quarter_facilities['STATE'].unique():
            state_facilities = quarter_facilities[quarter_facilities['STATE'] == state]
            # Sum all facility census values for the state
            total_state_census = state_facilities['avg_daily_census'].sum()
            state_census_data.append({
                'CY_Qtr': quarter,
                'STATE': state,
                'avg_state_census': total_state_census
            })
    
    state_census_df = pd.DataFrame(state_census_data)
    
    # Merge the statewide census data with state_lite
    state_lite = state_lite.merge(state_census_df, on=['CY_Qtr', 'STATE'], how='left')
    
    # Sort by state first, then quarter (AK 2017Q1, AK 2017Q2, etc.)
    state_lite = state_lite.sort_values(['STATE', 'CY_Qtr'])
    
    # Create national metrics by calculating proper ratios
    national_metrics = []
    for quarter in facility_lite['CY_Qtr'].unique():
        quarter_data = facility_lite[facility_lite['CY_Qtr'] == quarter]
        
        # Calculate national metrics using proper ratios
        total_census = quarter_data['avg_daily_census'].sum()
        total_nurse_hours = (quarter_data['Total_Nurse_HPRD'] * quarter_data['avg_daily_census']).sum()
        total_contract_hours = (quarter_data['Contract_Percentage'] * quarter_data['Total_Nurse_HPRD'] * quarter_data['avg_daily_census']).sum()
        
        # Calculate MDScensus (total resident days / avg days reported)
        total_resident_days = quarter_data['total_resident_days'].sum()
        avg_days_reported = quarter_data['days_reported'].mean()
        mdscensus = round(total_resident_days / avg_days_reported if avg_days_reported > 0 else 0, 1)
        
        national_metrics.append({
            'CY_Qtr': quarter,
            'Facility_Count': len(quarter_data),
            'Total_Nurse_HPRD': round(total_nurse_hours / total_census if total_census > 0 else 0, 2),
            'Contract_Percentage': round((total_contract_hours / total_nurse_hours) if total_nurse_hours > 0 else 0, 2),
            'MDS': mdscensus
        })
    
    national_lite = pd.DataFrame(national_metrics)
    national_lite = national_lite.sort_values('CY_Qtr')
    
    # Remove calculation fields from facility_lite before saving
    facility_lite_output = facility_lite.drop(['avg_daily_census', 'total_resident_days', 'days_reported'], axis=1)
    
    # Rename columns to be consistent across all files
    facility_lite_output.columns = [
        'CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME',
        'Total_Nurse_HPRD', 'Contract_Percentage', 'Census'
    ]
    
    state_lite.columns = [
        'CY_Qtr', 'STATE', 'Facility_Count', 'Census',
        'Total_Nurse_HPRD', 'Contract_Percentage', 'State_Census'
    ]
    
    national_lite.columns = [
        'CY_Qtr', 'Facility_Count', 'Total_Nurse_HPRD',
        'Contract_Percentage', 'MDS'
    ]
    
    # Save the lite metrics files
    facility_lite_output.to_csv('facility_lite_metrics.csv', index=False)
    state_lite.to_csv('state_lite_metrics.csv', index=False)
    national_lite.to_csv('national_lite_metrics.csv', index=False)
    
    # Print summary
    print("\nMetrics Generation Summary:")
    print(f"Total facility records: {len(facility_lite)}")
    print(f"Total state records: {len(state_lite)}")
    print(f"Total national records: {len(national_lite)}")
    print(f"Unique facilities: {facility_lite['PROVNUM'].nunique()}")
    
    # Print latest quarter's state summary
    latest_quarter = state_lite['CY_Qtr'].max()
    print(f"\nLatest quarter ({latest_quarter}) state summary:")
    latest_state = state_lite[state_lite['CY_Qtr'] == latest_quarter].sort_values('STATE')
    print(latest_state[['STATE', 'Facility_Count', 'Census', 'State_Census', 'Total_Nurse_HPRD', 'Contract_Percentage']].to_string())
    
    # Print latest quarter's national summary
    print(f"\nLatest quarter ({latest_quarter}) national summary:")
    latest_national = national_lite[national_lite['CY_Qtr'] == latest_quarter]
    print(latest_national[['Facility_Count', 'Total_Nurse_HPRD', 'Contract_Percentage', 'MDS']].to_string())

if __name__ == "__main__":
    generate_lite_metrics() 