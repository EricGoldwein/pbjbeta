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
    national_metrics_df = pd.read_csv('national_quarterly_metrics.csv', low_memory=False)
    
    # Format PROVNUMs
    format_provnum(facility_metrics)
    
    # Create facility lite metrics with existing columns and calculate missing ones
    facility_lite = facility_metrics[[
        'CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME',
        'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'RN_HPRD', 'RN_Care_HPRD', 'Contract_Percentage', 'Total_Contract_Hours', 'Total_Nurse_Hours', 'Total_Nurse_Care_Hours', 'Total_RN_Hours', 'Total_RN_Care_Hours', 'avg_daily_census',
        'total_resident_days', 'days_reported', 'MDScensus'
    ]].copy()
    
    # Use the existing RN_HPRD values for facilities - these are already calculated correctly
    facility_lite['Total_RN_HPRD'] = facility_lite['RN_HPRD']  # Total RN HPRD from existing RN_HPRD
    facility_lite['Direct_Care_RN_HPRD'] = facility_lite['RN_Care_HPRD']  # Direct Care RN HPRD from existing RN_Care_HPRD
    
    # Handle division by zero
    facility_lite['Total_RN_HPRD'] = facility_lite['Total_RN_HPRD'].fillna(0)
    facility_lite['Direct_Care_RN_HPRD'] = facility_lite['Direct_Care_RN_HPRD'].fillna(0)
    
    # Sort by quarter and PROVNUM
    facility_lite = facility_lite.sort_values(['CY_Qtr', 'PROVNUM'])
    
    # Calculate state lite metrics with proper weighted averages
    state_data = []
    for quarter in facility_lite['CY_Qtr'].unique():
        quarter_facilities = facility_lite[facility_lite['CY_Qtr'] == quarter]
        for state in quarter_facilities['STATE'].unique():
            state_facilities = quarter_facilities[quarter_facilities['STATE'] == state]
            
            # Calculate weighted averages
            total_census = state_facilities['avg_daily_census'].sum()
            total_nurse_hours = state_facilities['Total_Nurse_Hours'].sum()
            total_nurse_care_hours = state_facilities['Total_Nurse_Care_Hours'].sum()
            total_rn_hours = state_facilities['Total_RN_Hours'].sum()
            total_rn_care_hours = state_facilities['Total_RN_Care_Hours'].sum()
            total_contract_hours = state_facilities['Total_Contract_Hours'].sum()
            
            # Calculate HPRD values (weighted by census) 
            # Use total_resident_days as denominator since hours are quarterly totals
            total_resident_days = state_facilities['total_resident_days'].sum()
            total_nurse_hprd = total_nurse_hours / total_resident_days if total_resident_days > 0 else 0
            nurse_care_hprd = total_nurse_care_hours / total_resident_days if total_resident_days > 0 else 0
            total_rn_hprd = total_rn_hours / total_resident_days if total_resident_days > 0 else 0
            direct_care_rn_hprd = total_rn_care_hours / total_resident_days if total_resident_days > 0 else 0
            contract_percentage = (total_contract_hours / total_nurse_hours * 100) if total_nurse_hours > 0 else 0
            
            state_data.append({
                'CY_Qtr': quarter,
                'STATE': state,
                'facility_count': len(state_facilities),
                'avg_daily_census': state_facilities['avg_daily_census'].mean(),
                'Total_Nurse_HPRD': total_nurse_hprd,
                'Nurse_Care_HPRD': nurse_care_hprd,
                'Total_RN_HPRD': total_rn_hprd,
                'Direct_Care_RN_HPRD': direct_care_rn_hprd,
                'Contract_Percentage': contract_percentage,
                'avg_state_census': total_census
            })
    
    state_lite = pd.DataFrame(state_data)
    
    # Sort by state first, then quarter (AK 2017Q1, AK 2017Q2, etc.)
    state_lite = state_lite.sort_values(['STATE', 'CY_Qtr'])
    
    # Calculate national metrics with proper weighted averages
    national_metrics = []
    for quarter in facility_lite['CY_Qtr'].unique():
        quarter_facilities = facility_lite[facility_lite['CY_Qtr'] == quarter]
        
        # Calculate weighted averages nationally
        total_census = quarter_facilities['avg_daily_census'].sum()
        total_nurse_hours = quarter_facilities['Total_Nurse_Hours'].sum()
        total_nurse_care_hours = quarter_facilities['Total_Nurse_Care_Hours'].sum()
        total_rn_hours = quarter_facilities['Total_RN_Hours'].sum()
        total_rn_care_hours = quarter_facilities['Total_RN_Care_Hours'].sum()
        total_contract_hours = quarter_facilities['Total_Contract_Hours'].sum()
        
        # Calculate HPRD values (weighted by census)
        # Use total_resident_days as denominator since hours are quarterly totals
        total_resident_days = quarter_facilities['total_resident_days'].sum()
        total_nurse_hprd = total_nurse_hours / total_resident_days if total_resident_days > 0 else 0
        nurse_care_hprd = total_nurse_care_hours / total_resident_days if total_resident_days > 0 else 0
        total_rn_hprd = total_rn_hours / total_resident_days if total_resident_days > 0 else 0
        direct_care_rn_hprd = total_rn_care_hours / total_resident_days if total_resident_days > 0 else 0
        contract_percentage = (total_contract_hours / total_nurse_hours * 100) if total_nurse_hours > 0 else 0
        
        national_metrics.append({
            'CY_Qtr': quarter,
            'Facility_Count': len(quarter_facilities),
            'Total_Nurse_HPRD': total_nurse_hprd,
            'Nurse_Care_HPRD': nurse_care_hprd,
            'Total_RN_HPRD': total_rn_hprd,
            'Direct_Care_RN_HPRD': direct_care_rn_hprd,
            'Contract_Percentage': contract_percentage,
            'MDS': total_census
        })
    
    national_lite = pd.DataFrame(national_metrics)
    national_lite = national_lite.sort_values('CY_Qtr')
    
    # Create the output dataframe with correct column order
    facility_lite_output = facility_lite[[
        'CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME',
        'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Direct_Care_RN_HPRD', 'Contract_Percentage', 'MDScensus'
    ]].copy()
    
    # Rename the MDScensus column to Census
    facility_lite_output = facility_lite_output.rename(columns={'MDScensus': 'Census'})
    
    state_lite.columns = [
        'CY_Qtr', 'STATE', 'Facility_Count', 'Census',
        'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Direct_Care_RN_HPRD', 'Contract_Percentage', 'State_Census'
    ]
    
    national_lite.columns = [
        'CY_Qtr', 'Facility_Count', 'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Direct_Care_RN_HPRD',
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
    print(latest_state[['STATE', 'Facility_Count', 'Census', 'State_Census', 'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Contract_Percentage']].to_string())
    
    # Print latest quarter's national summary
    print(f"\nLatest quarter ({latest_quarter}) national summary:")
    latest_national = national_lite[national_lite['CY_Qtr'] == latest_quarter]
    print(latest_national[['Facility_Count', 'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Contract_Percentage', 'MDS']].to_string())

if __name__ == "__main__":
    generate_lite_metrics() 
