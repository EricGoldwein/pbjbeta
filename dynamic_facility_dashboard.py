#!/usr/bin/env python3
"""
Dynamic Facility Dashboard
Uses the complete CSV file for any facility for fast, detailed analysis
"""

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import json
from decimal import Decimal, ROUND_HALF_UP
import os
import glob
import sys

app = Flask(__name__)

# Global variables
df = None
global_df = None
provider_info_df = None

def create_facility_complete_csv(provnum):
    """Extract all data for any facility into one CSV"""
    print(f"Creating comprehensive CSV for facility {provnum}...")
    
    # Load all standardized nurse files
    nurse_files = glob.glob('standardized_PBJ/PBJ_dailynursestaffing_*.csv')
    nurse_files.sort()
    
    all_data = []
    total_records = 0
    
    for file_path in nurse_files:
        print(f"Processing: {os.path.basename(file_path)}")
        
        try:
            # Read the CSV
            df = pd.read_csv(file_path, low_memory=False)
            
            # Format PROVNUM to ensure it's a string and handle both numeric and alphanumeric formats
            df['PROVNUM'] = df['PROVNUM'].astype(str)
            # Only zero-pad if all digits, otherwise keep as-is
            df['PROVNUM'] = df['PROVNUM'].apply(lambda x: x.zfill(6) if x.isdigit() else x.upper())
            
            # Filter for the specific facility (handle different PROVNUM formats)
            # Create variations of the search provnum
            search_variants = [provnum.upper()]
            if provnum.isdigit():
                # For numeric provnums, also try with leading zeros
                search_variants.extend([provnum.zfill(6), provnum.lstrip('0')])
            
            facility_data = df[df['PROVNUM'].isin(search_variants)].copy()
            
            if len(facility_data) == 0:
                print(f"  No data for facility {provnum}")
                continue
            
            print(f"  Found {len(facility_data)} records")
            all_data.append(facility_data)
            total_records += len(facility_data)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if len(all_data) == 0:
        print(f"No data found for facility {provnum}!")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Sort by date
    combined_data = combined_data.sort_values('WorkDate')
    
    # Save to CSV
    output_filename = f'facility_{provnum}_complete_data.csv'
    combined_data.to_csv(output_filename, index=False)
    
    print(f"\nExtraction Summary:")
    print(f"Total records: {total_records}")
    print(f"Date range: {combined_data['WorkDate'].min()} to {combined_data['WorkDate'].max()}")
    print(f"Quarters: {combined_data['CY_Qtr'].nunique()}")
    print(f"File saved as: {output_filename}")
    
    return combined_data

def create_facility_provider_info_csv(provnum):
    """Extract provider info data for facility"""
    print(f"Creating provider info CSV for facility {provnum}...")
    
    # Load all provider info files
    provider_files = glob.glob('provider_info_normalized/ProviderInfoNorm_*.csv')
    provider_files.sort()
    
    all_provider_data = []
    total_provider_records = 0
    
    for file_path in provider_files:
        print(f"Processing provider info: {os.path.basename(file_path)}")
        
        try:
            # Read the CSV
            df = pd.read_csv(file_path, low_memory=False)
            
            # Format CCN to ensure it's a string and handle both numeric and alphanumeric formats
            df['ccn'] = df['ccn'].astype(str)
            # Only zero-pad if all digits, otherwise keep as-is
            df['ccn'] = df['ccn'].apply(lambda x: x.zfill(6) if x.isdigit() else x.upper())
            
            # Filter for the specific facility (handle different CCN formats)
            # Create variations of the search provnum
            search_variants = [provnum.upper()]
            if provnum.isdigit():
                # For numeric provnums, also try with leading zeros
                search_variants.extend([provnum.zfill(6), provnum.lstrip('0')])
            
            facility_data = df[df['ccn'].isin(search_variants)].copy()
            
            if len(facility_data) > 0:
                print(f"  Found {len(facility_data)} provider info records for {provnum}")
                all_provider_data.append(facility_data)
                total_provider_records += len(facility_data)
            else:
                print(f"  No provider info records found for {provnum}")
                
        except Exception as e:
            print(f"Error processing provider info {file_path}: {e}")
            continue
    
    if not all_provider_data:
        print(f"❌ No provider info data found for facility {provnum}")
        return None
    
    # Combine all provider info data
    combined_provider_df = pd.concat(all_provider_data, ignore_index=True)
    print(f"✅ Combined {total_provider_records} provider info records for facility {provnum}")
    
    # Sort by processing_date
    combined_provider_df['processing_date'] = pd.to_datetime(combined_provider_df['processing_date'])
    combined_provider_df = combined_provider_df.sort_values('processing_date')
    
    return combined_provider_df

def create_dynamic_dashboard(provnum):
    """Create and initialize the dynamic dashboard for a specific facility"""
    global global_df, provider_info_df
    
    # Create the facility CSV if it doesn't exist
    csv_file = f'facility_{provnum}_complete_data.csv'
    if not os.path.exists(csv_file):
        print(f"Creating CSV for facility {provnum}...")
        create_facility_complete_csv(provnum)
    
    # Create the provider info CSV if it doesn't exist
    provider_csv_file = f'facility_{provnum}_provider_info_data.csv'
    if not os.path.exists(provider_csv_file):
        print(f"Creating provider info CSV for facility {provnum}...")
        provider_data = create_facility_provider_info_csv(provnum)
        if provider_data is not None:
            provider_data.to_csv(provider_csv_file, index=False)
            print(f"Provider info CSV saved as: {provider_csv_file}")
    
    # Load the facility data
    global_df = load_facility_data(provnum)
    
    # Load the provider info data
    global provider_info_df
    if os.path.exists(provider_csv_file):
        try:
            provider_info_df = pd.read_csv(provider_csv_file, low_memory=False)
            provider_info_df['processing_date'] = pd.to_datetime(provider_info_df['processing_date'])
            print(f"✅ Loaded {len(provider_info_df)} provider info records")
        except Exception as e:
            print(f"Error loading provider info data: {e}")
            provider_info_df = None
    else:
        provider_info_df = None
    
    if global_df is None:
        print(f"Failed to load data for facility {provnum}")
        return None
    
    return app

def initialize_data():
    """Initialize the global data variable"""
    global df
    load_data()  # load_data() already sets the global df variable
    return global_df

def round_financial(value, decimals=2):
    """Round using financial rounding (ROUND_HALF_UP)"""
    if pd.isna(value) or value is None:
        return 0.0
    return float(Decimal(str(value)).quantize(Decimal('0.' + '0' * decimals), rounding=ROUND_HALF_UP))

def generate_pbj_source_link(quarter, date, provnum="225500", data_type="nurse"):
    """Generate PBJ source link for a specific quarter and date."""
    # Convert date to YYYYMMDD format if needed
    if isinstance(date, str):
        if len(date) == 10 and '-' in date:  # YYYY-MM-DD format
            date = date.replace('-', '')
        elif len(date) == 8:  # Already YYYYMMDD
            pass
        else:
            return None
    elif hasattr(date, 'strftime'):
        date = date.strftime('%Y%m%d')
    else:
        return None
    
    # Format quarter for URL (e.g., "2021Q3" -> "q3-2021")
    if isinstance(quarter, str):
        if 'Q' in quarter.upper():
            year = quarter[:4]
            q_num = quarter[-1]
            quarter_url = f"q{q_num}-{year}"
        else:
            return None
    else:
        return None
    
    # Determine column names based on quarter
    # Early quarters (2017-2019, Q2-Q3 2020) use lowercase
    early_quarters = [
        "2017Q1", "2017Q2", "2017Q3", "2017Q4",
        "2018Q4", 
        "2019Q1", "2019Q2", "2019Q3", "2019Q4",
        "2020Q2", "2020Q3"
    ]
    
    if quarter in early_quarters:
        provnum_col = "provnum"
        workdate_col = "workdate"
    else:
        provnum_col = "PROVNUM"
        workdate_col = "WorkDate"
    
    # Generate the query parameters
    query_params = {
        "filters": {
            "list": [{
                "conditions": [
                    {
                        "column": {"value": provnum_col},
                        "comparator": {"value": "="},
                        "filterValue": [provnum]
                    },
                    {
                        "column": {"value": workdate_col},
                        "comparator": {"value": "="},
                        "filterValue": [date]
                    }
                ]
            }],
            "rootConjunction": {"value": "AND"}
        },
        "keywords": "",
        "offset": 0,
        "limit": 10,
        "sort": {"sortBy": None, "sortOrder": None},
        "columns": []
    }
    
    # Convert to JSON and URL encode
    import json
    import urllib.parse
    query_json = json.dumps(query_params)
    encoded_query = urllib.parse.quote(query_json)
    
    # Generate the full URL based on data type
    if data_type == "nurse":
        base_url = "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data"
    else:  # nonnurse
        base_url = "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-non-nurse-staffing/data"
    
    url = f"{base_url}/{quarter_url}?query={encoded_query}"
    
    return url

def format_pbj_source_link(quarter, date, provnum="225500", data_type="nurse"):
    """Generate formatted PBJ source link with display text."""
    url = generate_pbj_source_link(quarter, date, provnum, data_type)
    if not url:
        return None
    
    # Format date for display (YYYYMMDD -> MM-DD-YYYY)
    if isinstance(date, str) and len(date) == 8:
        display_date = f"{date[4:6]}-{date[6:8]}-{date[:4]}"
    elif hasattr(date, 'strftime'):
        display_date = date.strftime('%m-%d-%Y')
    else:
        display_date = str(date)
    
    if data_type == "nurse":
        return f'<a href="{url}" target="_blank" style="font-size: 0.9em; color: #333; text-decoration: none; font-weight: 500;">CMS PBJ Source: {display_date}</a>'
    else:  # nonnurse
        return f'<a href="{url}" target="_blank" style="font-size: 0.9em; color: #333; text-decoration: none; font-weight: 500;">CMS NonNurse: {display_date}</a>'

def load_facility_data(provnum):
    """Load the facility data"""
    global global_df
    print(f"Loading facility {provnum} data...")
    try:
        global_df = pd.read_csv(f'facility_{provnum}_complete_data.csv')
        print(f"Loaded {len(global_df)} records")
        print(f"Columns: {list(global_df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Clean up the data - remove any columns with .1 suffix and handle NaN values
    global_df = global_df.drop(columns=[col for col in global_df.columns if '.1' in col])
    
    # Fill NaN values with 0 for numeric columns
    numeric_columns = global_df.select_dtypes(include=[np.number]).columns
    global_df[numeric_columns] = global_df[numeric_columns].fillna(0)
    
    # Apply financial rounding to hours columns
    hours_columns = ['Hrs_RN', 'Hrs_LPN', 'Hrs_CNA', 'Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_LPNadmin', 
                     'Hrs_RN_ctr', 'Hrs_LPN_ctr', 'Hrs_CNA_ctr', 'Hrs_NAtrn', 'Hrs_MedAide']
    for col in hours_columns:
        if col in global_df.columns:
            global_df[col] = global_df[col].apply(lambda x: round_financial(x, 2))
    
    # Convert WorkDate to datetime (naive, no timezone to avoid date shift issues)
    global_df['WorkDate'] = pd.to_datetime(global_df['WorkDate'], format='%Y%m%d', utc=False)
    
    # Add day of week
    global_df['DayOfWeek'] = global_df['WorkDate'].dt.day_name()
    global_df['DayOfWeekNum'] = global_df['WorkDate'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Add month and year
    global_df['Month'] = global_df['WorkDate'].dt.month
    global_df['Year'] = global_df['WorkDate'].dt.year
    
    # Calculate HPRD for each position with proper rounding
    # RN HPRD includes direct care only (not admin/DON)
    global_df['RN_HPRD'] = (global_df['Hrs_RN'] / global_df['MDScensus']).apply(lambda x: round_financial(x, 2))
    # LPN HPRD includes direct care only (not admin)
    global_df['LPN_HPRD'] = (global_df['Hrs_LPN'] / global_df['MDScensus']).apply(lambda x: round_financial(x, 2))
    # CNA HPRD includes direct care only (not medaide/natr)
    global_df['CNA_HPRD'] = (global_df['Hrs_CNA'] / global_df['MDScensus']).apply(lambda x: round_financial(x, 2))
    # Total HPRD includes ALL staff (RN + RNadmin + RNDON + LPN + LPNadmin + CNA + NAtrn + MedAide)
    global_df['Total_Nurse_HPRD'] = ((global_df['Hrs_RN'] + global_df['Hrs_RNadmin'] + global_df['Hrs_RNDON'] + global_df['Hrs_LPN'] + global_df['Hrs_LPNadmin'] + global_df['Hrs_CNA'] + global_df['Hrs_NAtrn'] + global_df['Hrs_MedAide']) / global_df['MDScensus']).apply(lambda x: round_financial(x, 2))
    
    # Calculate additional metrics for outlier detection and table display
    global_df['Total_RN_Hours'] = (global_df['Hrs_RN'] + global_df['Hrs_RNadmin'] + global_df['Hrs_RNDON']).apply(lambda x: round_financial(x, 2))
    global_df['Total_RN_HPRD'] = (global_df['Total_RN_Hours'] / global_df['MDScensus']).apply(lambda x: round_financial(x, 2))
    global_df['Total_LPN_Hours'] = (global_df['Hrs_LPN'] + global_df['Hrs_LPNadmin']).apply(lambda x: round_financial(x, 2))
    global_df['Total_LPN_HPRD'] = (global_df['Total_LPN_Hours'] / global_df['MDScensus']).apply(lambda x: round_financial(x, 2))
    global_df['Total_Nurse_Aide_Hours'] = (global_df['Hrs_CNA'] + global_df['Hrs_MedAide'] + global_df['Hrs_NAtrn']).apply(lambda x: round_financial(x, 2))
    global_df['Total_Nurse_Aide_HPRD'] = (global_df['Total_Nurse_Aide_Hours'] / global_df['MDScensus']).apply(lambda x: round_financial(x, 2))
    
    # Nurse Staff Hours (excluding Admin & DON) - includes all direct care staff
    global_df['Nurse_Staff_Hours_Excl_Admin'] = (global_df['Hrs_RN'] + global_df['Hrs_LPN'] + global_df['Hrs_CNA'] + global_df['Hrs_NAtrn'] + global_df['Hrs_MedAide']).apply(lambda x: round_financial(x, 2))
    global_df['Nurse_Staff_HPRD_Excl_Admin'] = (global_df['Nurse_Staff_Hours_Excl_Admin'] / global_df['MDScensus']).apply(lambda x: round_financial(x, 2))
    
    # Total Nurse Hours (All Staff including admin/DON)
    global_df['Total_Nurse_Hours'] = (global_df['Hrs_RN'] + global_df['Hrs_RNadmin'] + global_df['Hrs_RNDON'] + global_df['Hrs_LPN'] + global_df['Hrs_LPNadmin'] + global_df['Hrs_CNA'] + global_df['Hrs_NAtrn'] + global_df['Hrs_MedAide']).apply(lambda x: round_financial(x, 2))
    
    # Total Staff Hours and HPRD
    global_df['Total_Staff_Hours'] = (global_df['Total_RN_Hours'] + global_df['Total_LPN_Hours'] + global_df['Total_Nurse_Aide_Hours']).apply(lambda x: round_financial(x, 2))
    global_df['Total_Staff_HPRD'] = (global_df['Total_Staff_Hours'] / global_df['MDScensus']).apply(lambda x: round_financial(x, 2))
    
    # Calculate contract percentages with proper rounding
    global_df['RN_Contract_Pct'] = (global_df['Hrs_RN_ctr'] / global_df['Hrs_RN'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    global_df['LPN_Contract_Pct'] = (global_df['Hrs_LPN_ctr'] / global_df['Hrs_LPN'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    global_df['CNA_Contract_Pct'] = (global_df['Hrs_CNA_ctr'] / global_df['Hrs_CNA'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # Calculate more granular contract percentages
    # CNA contract percentage (CNA only)
    global_df['CNA_Only_Contract_Pct'] = (global_df['Hrs_CNA_ctr'] / global_df['Hrs_CNA'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # Nurse Aide contract percentage (CNA + MedAide + NAtrn)
    total_nurse_aide_hours = global_df['Hrs_CNA'] + global_df['Hrs_MedAide'] + global_df['Hrs_NAtrn']
    total_nurse_aide_contract_hours = global_df['Hrs_CNA_ctr'] + global_df.get('Hrs_MedAide_ctr', 0) + global_df.get('Hrs_NAtrn_ctr', 0)
    global_df['Nurse_Aide_Contract_Pct'] = (total_nurse_aide_contract_hours / total_nurse_aide_hours * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # LPN contract percentage (LPN only, excluding admin)
    global_df['LPN_Only_Contract_Pct'] = (global_df['Hrs_LPN_ctr'] / global_df['Hrs_LPN'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # Total LPN contract percentage (LPN + LPN admin)
    total_lpn_hours = global_df['Hrs_LPN'] + global_df['Hrs_LPNadmin']
    total_lpn_contract_hours = global_df['Hrs_LPN_ctr'] + global_df.get('Hrs_LPNadmin_ctr', 0)
    global_df['Total_LPN_Contract_Pct'] = (total_lpn_contract_hours / total_lpn_hours * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # Total Contract Percentage (Direct care contract hours / Direct care total hours)
    # Only use contract hours that actually exist in the data (direct care staff)
    total_contract_hours = (global_df['Hrs_RN_ctr'] + global_df['Hrs_LPN_ctr'] + global_df['Hrs_CNA_ctr'])
    total_direct_care_hours = (global_df['Hrs_RN'] + global_df['Hrs_LPN'] + global_df['Hrs_CNA'])
    global_df['Total_Contract_Pct'] = (total_contract_hours / total_direct_care_hours * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # Add holiday indicator (comprehensive federal holidays)
    def is_federal_holiday(date):
        """Check if a date is a US federal holiday"""
        year = date.year
        month = date.month
        day = date.day
        
        # Fixed holidays
        fixed_holidays = [
            (1, 1),   # New Year's Day
            (6, 19),  # Juneteenth (since 2021)
            (7, 4),   # Independence Day
            (11, 11), # Veterans Day
            (12, 25), # Christmas Day
        ]
        
        # Check fixed holidays
        if (month, day) in fixed_holidays:
            # Juneteenth only became a federal holiday in 2021
            if month == 6 and day == 19 and year < 2021:
                return False
            return True
        
        # Variable holidays (calculated for each year)
        # Martin Luther King Jr. Day (3rd Monday in January)
        mlk_day = get_third_monday(year, 1)
        if date == mlk_day:
            return True
        
        # Presidents Day (3rd Monday in February)
        presidents_day = get_third_monday(year, 2)
        if date == presidents_day:
            return True
        
        # Memorial Day (last Monday in May)
        memorial_day = get_last_monday(year, 5)
        if date == memorial_day:
            return True
        
        # Labor Day (1st Monday in September)
        labor_day = get_first_monday(year, 9)
        if date == labor_day:
            return True
        
        # Columbus Day (2nd Monday in October)
        columbus_day = get_second_monday(year, 10)
        if date == columbus_day:
            return True
        
        # Thanksgiving (4th Thursday in November)
        thanksgiving = get_fourth_thursday(year, 11)
        if date == thanksgiving:
            return True
        
        return False
    
    def get_first_monday(year, month):
        """Get the first Monday of a given month/year"""
        first_day = pd.Timestamp(year, month, 1)
        days_ahead = 0 - first_day.weekday()  # Monday is 0
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return first_day + pd.Timedelta(days=days_ahead)
    
    def get_second_monday(year, month):
        """Get the second Monday of a given month/year"""
        return get_first_monday(year, month) + pd.Timedelta(days=7)
    
    def get_third_monday(year, month):
        """Get the third Monday of a given month/year"""
        return get_first_monday(year, month) + pd.Timedelta(days=14)
    
    def get_fourth_thursday(year, month):
        """Get the fourth Thursday of a given month/year"""
        first_day = pd.Timestamp(year, month, 1)
        days_ahead = 3 - first_day.weekday()  # Thursday is 3
        if days_ahead <= 0:  # Target day already happened this week
            days_ahead += 7
        return first_day + pd.Timedelta(days=days_ahead + 21)  # 4th Thursday
    
    def get_last_monday(year, month):
        """Get the last Monday of a given month/year"""
        # Get the first day of next month, then go back to find last Monday
        if month == 12:
            next_month = pd.Timestamp(year + 1, 1, 1)
        else:
            next_month = pd.Timestamp(year, month + 1, 1)
        
        # Go back to the last day of current month
        last_day = next_month - pd.Timedelta(days=1)
        
        # Find the last Monday
        days_back = last_day.weekday()  # Monday is 0
        return last_day - pd.Timedelta(days=days_back)

    # Apply holiday detection to all dates
    global_df['IsHoliday'] = global_df['WorkDate'].apply(is_federal_holiday)
    
    print(f"Loaded {len(global_df)} records from {global_df['WorkDate'].min().date()} to {global_df['WorkDate'].max().date()}")
    print(f"Calculated columns: {[col for col in global_df.columns if 'Total' in col or 'HPRD' in col]}")
    
    # Verify critical columns exist
    critical_cols = ['Total_Staff_HPRD', 'Total_Staff_Hours', 'Total_RN_HPRD', 'Total_LPN_HPRD', 'Total_Nurse_Aide_HPRD']
    missing_cols = [col for col in critical_cols if col not in global_df.columns]
    if missing_cols:
        print(f"❌ MISSING CRITICAL COLUMNS: {missing_cols}")
    else:
        print(f"✓ All critical columns present")
    
    return global_df


@app.route('/')
def index():
    """Main dashboard page"""
    global global_df
    if global_df is not None and not global_df.empty:
        # Sort by WorkDate to get most recent data first
        if 'WorkDate' in global_df.columns:
            global_df_sorted = global_df.sort_values('WorkDate', ascending=False)
        else:
            global_df_sorted = global_df
            
        facility_name = global_df_sorted['PROVNAME'].iloc[0] if 'PROVNAME' in global_df_sorted.columns else "Unknown Facility"
        facility_provnum = str(global_df_sorted['PROVNUM'].iloc[0]).zfill(6) if 'PROVNUM' in global_df_sorted.columns else "Unknown"
        city = global_df_sorted['CITY'].iloc[0] if 'CITY' in global_df_sorted.columns else "Unknown"
        state = global_df_sorted['STATE'].iloc[0] if 'STATE' in global_df_sorted.columns else "Unknown"
        county_name = global_df_sorted['COUNTY_NAME'].iloc[0] if 'COUNTY_NAME' in global_df_sorted.columns else "Unknown"
    else:
        facility_name = "Unknown Facility"
        facility_provnum = "Unknown"
        city = "Unknown"
        state = "Unknown"
        county_name = "Unknown"
    
    return render_template('dynamic_facility_dashboard.html', 
                         facility_name=facility_name, 
                         provnum=facility_provnum,
                         city=city,
                         state=state,
                         county_name=county_name)

@app.route('/api/data')
def get_data():
    """Get filtered data"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        position = request.args.get('position', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        quarter = request.args.get('quarter', 'all')
        year = request.args.get('year', 'all')
        show_holidays_only = request.args.get('holidays_only', 'false') == 'true'
        
        # Filter data
        global global_df
        filtered_df = global_df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] <= end_date]
        if day_of_week != 'all':
            filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
        if quarter != 'all' and quarter.strip():
            # Handle multiple quarters (comma-separated)
            quarters = [q.strip() for q in quarter.split(',')]
            filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
        if show_holidays_only:
            filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
        
        # Convert to records and handle NaN values
        data = filtered_df.to_dict('records')
        
        # Replace NaN values with None for JSON serialization and format dates
        for record in data:
            for key, value in record.items():
                if pd.isna(value):
                    record[key] = None
                elif key == 'WorkDate' and value is not None and not pd.isna(value):
                    # Format date as YYYY-MM-DD without time
                    record[key] = value.strftime('%Y-%m-%d')
        
        # Handle date range formatting
        min_date = filtered_df['WorkDate'].min()
        max_date = filtered_df['WorkDate'].max()
        
        date_range = {}
        if not pd.isna(min_date):
            date_range['min'] = min_date.strftime('%Y-%m-%d')
        if not pd.isna(max_date):
            date_range['max'] = max_date.strftime('%Y-%m-%d')
        
        return jsonify({
            'data': data,
            'total_records': len(data),
            'date_range': date_range
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/summary')
def get_summary():
    """Get summary statistics"""
    try:
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        position = request.args.get('position', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        quarter = request.args.get('quarter', 'all')
        show_holidays_only = request.args.get('holidays_only', 'false') == 'true'
        
        # Filter data
        global global_df
        filtered_df = global_df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] <= end_date]
        if day_of_week != 'all':
            filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
        if quarter != 'all' and quarter.strip():
            # Handle multiple quarters (comma-separated)
            quarters = [q.strip() for q in quarter.split(',')]
            filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
        if show_holidays_only:
            filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
        
        # Calculate summary statistics
        summary = {
            'total_days': len(filtered_df),
            'avg_census': float(filtered_df['MDScensus'].mean()) if len(filtered_df) > 0 else 0,
            'avg_rn_hprd': float(filtered_df['RN_HPRD'].mean()) if len(filtered_df) > 0 else 0,
            'avg_total_rn_hprd': float(filtered_df['Total_RN_HPRD'].mean()) if len(filtered_df) > 0 else 0,
            'avg_lpn_hprd': float(filtered_df['LPN_HPRD'].mean()) if len(filtered_df) > 0 else 0,
            'avg_cna_hprd': float(filtered_df['CNA_HPRD'].mean()) if len(filtered_df) > 0 else 0,
            'avg_nurse_staff_hprd_excl_admin': float(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].mean()) if len(filtered_df) > 0 else 0,
            'avg_total_hprd': float(filtered_df['Total_Staff_HPRD'].mean()) if len(filtered_df) > 0 else 0,
            'avg_rn_contract_pct': float(filtered_df['RN_Contract_Pct'].mean()) if len(filtered_df) > 0 else 0,
            'avg_lpn_contract_pct': float(filtered_df['LPN_Contract_Pct'].mean()) if len(filtered_df) > 0 else 0,
            'avg_cna_contract_pct': float(filtered_df['CNA_Contract_Pct'].mean()) if len(filtered_df) > 0 else 0,
            'avg_total_contract_pct': float(filtered_df['Total_Contract_Pct'].mean()) if len(filtered_df) > 0 else 0,
            'total_rn_hours': float(filtered_df['Hrs_RN'].sum()) if len(filtered_df) > 0 else 0,
            # RN Sub 8 metrics - days with less than 8 hours of RN staffing
            'total_rn_sub8': int((filtered_df['Total_RN_Hours'] < 8).sum()) if len(filtered_df) > 0 else 0,
            'direct_rn_sub8': int((filtered_df['Hrs_RN'] < 8).sum()) if len(filtered_df) > 0 else 0,
            'total_lpn_hours': float(filtered_df['Hrs_LPN'].sum()) if len(filtered_df) > 0 else 0,
            'total_cna_hours': float(filtered_df['Hrs_CNA'].sum()) if len(filtered_df) > 0 else 0,
            'holiday_days': len(filtered_df[filtered_df['IsHoliday'] == True]),
            # Additional summary statistics
            'avg_rn_admin_hours': float(filtered_df['Hrs_RNadmin'].mean()) if len(filtered_df) > 0 else 0,
            'avg_rn_don_hours': float(filtered_df['Hrs_RNDON'].mean()) if len(filtered_df) > 0 else 0,
            'avg_lpn_admin_hours': float(filtered_df['Hrs_LPNadmin'].mean()) if len(filtered_df) > 0 else 0,
            'avg_na_trainee_hours': float(filtered_df['Hrs_NAtrn'].mean()) if len(filtered_df) > 0 else 0,
            'avg_med_aide_hours': float(filtered_df['Hrs_MedAide'].mean()) if len(filtered_df) > 0 else 0,
            'avg_rn_contract_hours': float(filtered_df['Hrs_RN_ctr'].mean()) if len(filtered_df) > 0 else 0,
            'avg_lpn_contract_hours': float(filtered_df['Hrs_LPN_ctr'].mean()) if len(filtered_df) > 0 else 0,
            'avg_cna_contract_hours': float(filtered_df['Hrs_CNA_ctr'].mean()) if len(filtered_df) > 0 else 0,
            'total_rn_admin_hours': float(filtered_df['Hrs_RNadmin'].sum()) if len(filtered_df) > 0 else 0,
            'total_rn_don_hours': float(filtered_df['Hrs_RNDON'].sum()) if len(filtered_df) > 0 else 0,
            'total_lpn_admin_hours': float(filtered_df['Hrs_LPNadmin'].sum()) if len(filtered_df) > 0 else 0,
            'total_na_trainee_hours': float(filtered_df['Hrs_NAtrn'].sum()) if len(filtered_df) > 0 else 0,
            'total_med_aide_hours': float(filtered_df['Hrs_MedAide'].sum()) if len(filtered_df) > 0 else 0,
            'total_rn_contract_hours': float(filtered_df['Hrs_RN_ctr'].sum()) if len(filtered_df) > 0 else 0,
            'total_lpn_contract_hours': float(filtered_df['Hrs_LPN_ctr'].sum()) if len(filtered_df) > 0 else 0,
            'total_cna_contract_hours': float(filtered_df['Hrs_CNA_ctr'].sum()) if len(filtered_df) > 0 else 0
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/provider_info')
def get_provider_info():
    """Get provider info data for the facility"""
    try:
        global provider_info_df
        
        if provider_info_df is None:
            return jsonify({'error': 'Provider info data not loaded'})
        
        # Convert to JSON-serializable format
        data = provider_info_df.copy()
        
        # Convert datetime to string
        data['processing_date'] = data['processing_date'].dt.strftime('%Y-%m-%d')
        
        # Round numeric values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].apply(lambda x: round_financial(x, 3) if pd.notna(x) else 0)
        
        return jsonify({
            'data': data.to_dict('records'),
            'facility_name': data['provider_name'].iloc[0] if len(data) > 0 else 'Unknown',
            'latest_rating': {
                'overall': int(data['overall_rating'].iloc[-1]) if len(data) > 0 and pd.notna(data['overall_rating'].iloc[-1]) else None,
                'staffing': int(data['staffing_rating'].iloc[-1]) if len(data) > 0 and pd.notna(data['staffing_rating'].iloc[-1]) else None,
                'health_inspection': int(data['health_inspection_rating'].iloc[-1]) if len(data) > 0 and pd.notna(data['health_inspection_rating'].iloc[-1]) else None
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/provider_info_summary')
def get_provider_info_summary():
    """Get provider info summary statistics"""
    try:
        global provider_info_df
        
        if provider_info_df is None:
            return jsonify({'error': 'Provider info data not loaded'})
        
        # Get latest record by processing date
        latest = provider_info_df.sort_values('processing_date').iloc[-1] if len(provider_info_df) > 0 else None
        
        if latest is None:
            return jsonify({'error': 'No provider info data available'})
        
        summary = {
            'facility_name': str(latest.get('provider_name', 'Unknown')),
            'city': str(latest.get('city', '')),
            'state': str(latest.get('state', '')),
            'county': str(latest.get('county', '')),
            'ownership_type': str(latest.get('ownership_type', '')),
            'latest_processing_date': latest['processing_date'].strftime('%Y-%m-%d') if pd.notna(latest.get('processing_date')) else '',
            'latest_quarter': str(latest.get('quarter', '')),
            'latest_census': float(latest.get('avg_residents_per_day', 0)) if pd.notna(latest.get('avg_residents_per_day')) else 0,
            'latest_overall_rating': int(latest.get('overall_rating', 0)) if pd.notna(latest.get('overall_rating')) else None,
            'latest_staffing_rating': int(latest.get('staffing_rating', 0)) if pd.notna(latest.get('staffing_rating')) else None,
            'latest_health_inspection_rating': int(latest.get('health_inspection_rating', 0)) if pd.notna(latest.get('health_inspection_rating')) else None,
            'latest_reported_total_hprd': float(latest.get('reported_total_nurse_hrs_per_resident_per_day', 0)) if pd.notna(latest.get('reported_total_nurse_hrs_per_resident_per_day')) else 0,
            'latest_case_mix_total_hprd': float(latest.get('case_mix_total_nurse_hrs_per_resident_per_day', 0)) if pd.notna(latest.get('case_mix_total_nurse_hrs_per_resident_per_day')) else 0,
            'latest_adjusted_total_hprd': float(latest.get('adjusted_total_nurse_hrs_per_resident_per_day', 0)) if pd.notna(latest.get('adjusted_total_nurse_hrs_per_resident_per_day')) else 0,
            'ownership_change_last_12_months': str(latest.get('provider_changed_ownership_in_last_12_months', 'Unknown')) if pd.notna(latest.get('provider_changed_ownership_in_last_12_months')) else 'Unknown',
            'sff_status': 'N/A',  # Not available in current data
            'total_records': len(provider_info_df),
            'quarters_covered': provider_info_df['quarter'].nunique() if 'quarter' in provider_info_df.columns else 0
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/provider_info_charts')
def get_provider_info_charts():
    """Get provider info chart data"""
    try:
        global provider_info_df
        
        if provider_info_df is None:
            return jsonify({'error': 'Provider info data not loaded'})
        
        # Group by quarter and take the latest processing date per quarter
        chart_data = provider_info_df.dropna(subset=['quarter']).copy()
        chart_data = chart_data.sort_values('processing_date').groupby('quarter').last().reset_index()
        
        # Format quarter labels for x-axis (Q1 2021 instead of 2021Q1)
        chart_data['quarter_label'] = chart_data['quarter'].apply(lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) and len(str(x)) == 6 else str(x) if pd.notna(x) else None)
        
        # Prepare data for charts
        charts = {
            'total_staffing': {
                'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
                'reported_total': chart_data['reported_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist(),
                'case_mix_total': chart_data['case_mix_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist(),
                'adjusted_total': chart_data['adjusted_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist()
            },
            'rn_staffing': {
                'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
                'reported_rn': chart_data['reported_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_rn_hrs_per_resident_per_day']), None).tolist(),
                'case_mix_rn': chart_data['case_mix_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_rn_hrs_per_resident_per_day']), None).tolist(),
                'adjusted_rn': chart_data['adjusted_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['adjusted_rn_hrs_per_resident_per_day']), None).tolist()
            },
            'cna_staffing': {
                'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
                'reported_cna': chart_data['reported_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_na_hrs_per_resident_per_day']), None).tolist(),
                'case_mix_cna': chart_data['case_mix_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_na_hrs_per_resident_per_day']), None).tolist(),
                'adjusted_cna': chart_data['adjusted_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['adjusted_na_hrs_per_resident_per_day']), None).tolist()
            },
            'census': {
                'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
                'census': chart_data['avg_residents_per_day'].fillna(0).tolist()
            },
            'ratings': {
                'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
                'overall': chart_data['overall_rating'].fillna(0).tolist(),
                'staffing': chart_data['staffing_rating'].fillna(0).tolist(),
                'health_inspection': chart_data['health_inspection_rating'].fillna(0).tolist()
            }
        }
        
        # Convert any remaining NaN values to None for JSON serialization
        def convert_nan_to_none(obj):
            if isinstance(obj, dict):
                return {key: convert_nan_to_none(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_nan_to_none(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        charts = convert_nan_to_none(charts)
        
        return jsonify(charts)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/charts')
def get_charts():
    """Get chart data"""
    global global_df
    try:
        # Check if data is loaded
        if global_df is None:
            return jsonify({
                'charts': {},
                'filter_info': 'Data not loaded',
                'error': 'Data not loaded. Please restart the application.'
            })
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        position = request.args.get('position', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        quarter = request.args.get('quarter', 'all')
        show_holidays_only = request.args.get('holidays_only', 'false') == 'true'
        
        # Get view mode parameters
        hprd_view = request.args.get('hprd_view', 'daily')
        hours_view = request.args.get('hours_view', 'daily')
        census_view = request.args.get('census_view', 'daily')
        contract_view = request.args.get('contract_view', 'daily')
        
        # Filter data
        filtered_df = global_df.copy()
        
        # Only apply filters if they are provided and not empty
        if start_date and start_date.strip():
            filtered_df = filtered_df[filtered_df['WorkDate'] >= start_date]
        if end_date and end_date.strip():
            filtered_df = filtered_df[filtered_df['WorkDate'] <= end_date]
        if day_of_week != 'all':
            filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
        if quarter != 'all' and quarter.strip():
            # Handle multiple quarters (comma-separated)
            quarters = [q.strip() for q in quarter.split(',')]
            filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
        if show_holidays_only:
            filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
        
        # Sort by date
        filtered_df = filtered_df.sort_values('WorkDate')
        
        # Debug: Print filtering results
        print(f"DEBUG: After filtering - filtered_df length: {len(filtered_df)}")
        if len(filtered_df) == 0:
            print("DEBUG: No data after filtering - returning error")
        
        # Helper function to aggregate data by view mode
        def aggregate_by_view_mode(df, view_mode, date_col='WorkDate'):
            if view_mode == 'daily':
                return df
            elif view_mode == 'month':
                # Aggregate by month
                df_copy = df.copy()
                df_copy['year_month'] = df_copy[date_col].dt.to_period('M')
                aggregated = df_copy.groupby('year_month').agg({
                    'Total_Nurse_HPRD': 'mean',
                    'Total_RN_HPRD': 'mean', 
                    'Total_LPN_HPRD': 'mean',
                    'Total_Nurse_Aide_HPRD': 'mean',
                    'Total_Staff_HPRD': 'mean',
                    'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                    'Total_RN_Hours': 'sum',
                    'Total_LPN_Hours': 'sum',
                    'Total_Nurse_Aide_Hours': 'sum',
                    'Total_Staff_Hours': 'sum',
                    'Nurse_Staff_Hours_Excl_Admin': 'sum',
                    'MDScensus': 'mean',
                    'RN_Contract_Pct': 'mean',
                    'LPN_Contract_Pct': 'mean',
                    'CNA_Contract_Pct': 'mean',
                    'Total_Contract_Pct': 'mean',
                    'IsHoliday': 'any'
                }).reset_index()
                aggregated[date_col] = aggregated['year_month'].dt.to_timestamp()
                return aggregated
            elif view_mode == 'quarter':
                # Aggregate by quarter
                df_copy = df.copy()
                df_copy['year_quarter'] = df_copy[date_col].dt.to_period('Q')
                aggregated = df_copy.groupby('year_quarter').agg({
                    'Total_Nurse_HPRD': 'mean',
                    'Total_RN_HPRD': 'mean',
                    'Total_LPN_HPRD': 'mean', 
                    'Total_Nurse_Aide_HPRD': 'mean',
                    'Total_Staff_HPRD': 'mean',
                    'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                    'Total_RN_Hours': 'sum',
                    'Total_LPN_Hours': 'sum',
                    'Total_Nurse_Aide_Hours': 'sum',
                    'Total_Staff_Hours': 'sum',
                    'Nurse_Staff_Hours_Excl_Admin': 'sum',
                    'MDScensus': 'mean',
                    'RN_Contract_Pct': 'mean',
                    'LPN_Contract_Pct': 'mean',
                    'CNA_Contract_Pct': 'mean',
                    'Total_Contract_Pct': 'mean',
                    'IsHoliday': 'any'
                }).reset_index()
                aggregated[date_col] = aggregated['year_quarter'].dt.to_timestamp()
                return aggregated
            elif view_mode == 'year':
                # Aggregate by year
                df_copy = df.copy()
                df_copy['year'] = df_copy[date_col].dt.year
                aggregated = df_copy.groupby('year').agg({
                    'Total_Nurse_HPRD': 'mean',
                    'Total_RN_HPRD': 'mean',
                    'Total_LPN_HPRD': 'mean',
                    'Total_Nurse_Aide_HPRD': 'mean', 
                    'Total_Staff_HPRD': 'mean',
                    'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                    'Total_RN_Hours': 'sum',
                    'Total_LPN_Hours': 'sum',
                    'Total_Nurse_Aide_Hours': 'sum',
                    'Total_Staff_Hours': 'sum',
                    'Nurse_Staff_Hours_Excl_Admin': 'sum',
                    'MDScensus': 'mean',
                    'RN_Contract_Pct': 'mean',
                    'LPN_Contract_Pct': 'mean',
                    'CNA_Contract_Pct': 'mean',
                    'Total_Contract_Pct': 'mean',
                    'IsHoliday': 'any'
                }).reset_index()
                aggregated[date_col] = pd.to_datetime(aggregated['year'], format='%Y')
                return aggregated
            else:
                return df
        
        # Check if we have any data after filtering
        if len(filtered_df) == 0:
            return jsonify({
                'charts': {},
                'filter_info': 'No data found for the selected filters',
                'error': 'No data available for the selected date range and filters'
            })
        
        # Check for required columns
        required_columns = ['WorkDate', 'Total_RN_HPRD', 'Total_LPN_HPRD', 'Total_Nurse_Aide_HPRD', 
                           'Total_Staff_HPRD', 'Nurse_Staff_HPRD_Excl_Admin', 'Total_RN_Hours', 'Total_LPN_Hours', 'Total_Nurse_Aide_Hours',
                           'Total_Staff_Hours', 'Nurse_Staff_Hours_Excl_Admin', 'MDScensus', 'RN_Contract_Pct', 'LPN_Contract_Pct', 'CNA_Contract_Pct', 'IsHoliday']
        
        missing_columns = [col for col in required_columns if col not in filtered_df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            print(f"Available columns: {list(filtered_df.columns)}")
            print(f"Data shape: {filtered_df.shape}")
            print(f"Columns with 'Total': {[col for col in filtered_df.columns if 'Total' in col]}")
            return jsonify({
                'charts': {},
                'filter_info': 'Data structure error',
                'error': f'Missing required columns: {", ".join(missing_columns)}'
            })
        
        charts = {}
        
        # Apply aggregation based on view modes
        hprd_df = aggregate_by_view_mode(filtered_df, hprd_view)
        hours_df = aggregate_by_view_mode(filtered_df, hours_view)
        census_df = aggregate_by_view_mode(filtered_df, census_view)
        contract_df = aggregate_by_view_mode(filtered_df, contract_view)
        
        # Helper function to format dates based on view mode
        def format_dates_for_view_mode(df, view_mode):
            if view_mode == 'daily':
                return df['WorkDate'].dt.strftime('%m-%d-%Y').tolist()
            elif view_mode == 'month':
                return df['WorkDate'].dt.strftime('%b %Y').tolist()
            elif view_mode == 'quarter':
                # Convert to quarter format like "Q3 2019"
                quarters = []
                for date in df['WorkDate']:
                    year = date.year
                    month = date.month
                    if month <= 3:
                        quarter = "Q1"
                    elif month <= 6:
                        quarter = "Q2"
                    elif month <= 9:
                        quarter = "Q3"
                    else:
                        quarter = "Q4"
                    quarters.append(f"{quarter} {year}")
                return quarters
            elif view_mode == 'year':
                return df['WorkDate'].dt.strftime('%Y').tolist()
            else:
                return df['WorkDate'].dt.strftime('%m-%d-%Y').tolist()
        
        # Debug: Check if Total_Staff_HPRD column exists
        print(f"DEBUG: Checking Total_Staff_HPRD column...")
        print(f"DEBUG: Total_Staff_HPRD in columns: {'Total_Staff_HPRD' in filtered_df.columns}")
        if 'Total_Staff_HPRD' in filtered_df.columns:
            print(f"DEBUG: Total_Staff_HPRD sample values: {filtered_df['Total_Staff_HPRD'].head().tolist()}")
        else:
            print(f"DEBUG: Available columns with 'Total': {[col for col in filtered_df.columns if 'Total' in col]}")
        
        # Daily HPRD trend
        # Add holiday indicators (only for daily view)
        hprd_holiday_markers = []
        if hprd_view == 'daily':
            holiday_data = filtered_df[filtered_df['IsHoliday'] == True]
            if len(holiday_data) > 0:
                hprd_holiday_markers = [{
                    'x': format_dates_for_view_mode(holiday_data, 'daily'),
                    'y': [0] * len(holiday_data),
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Holidays',
                    'marker': {'color': 'red', 'size': 8, 'symbol': 'star'},
                    'showlegend': True
                }]
        
        # Add holiday indicators for hours chart (only for daily view)
        hours_holiday_markers = []
        if hours_view == 'daily':
            holiday_data = filtered_df[filtered_df['IsHoliday'] == True]
            if len(holiday_data) > 0:
                hours_holiday_markers = [{
                    'x': format_dates_for_view_mode(holiday_data, 'daily'),
                    'y': [0] * len(holiday_data),
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Holidays',
                    'marker': {'color': 'red', 'size': 8, 'symbol': 'star'},
                    'showlegend': True
                }]
        
        # Add holiday indicators for census chart (only for daily view)
        census_holiday_markers = []
        if census_view == 'daily':
            holiday_data = filtered_df[filtered_df['IsHoliday'] == True]
            if len(holiday_data) > 0:
                census_holiday_markers = [{
                    'x': format_dates_for_view_mode(holiday_data, 'daily'),
                    'y': [0] * len(holiday_data),
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Holidays',
                    'marker': {'color': 'red', 'size': 8, 'symbol': 'star'},
                    'showlegend': True
                }]
        
        # Add holiday indicators for contract chart (only for daily view)
        contract_holiday_markers = []
        if contract_view == 'daily':
            holiday_data = filtered_df[filtered_df['IsHoliday'] == True]
            if len(holiday_data) > 0:
                contract_holiday_markers = [{
                    'x': format_dates_for_view_mode(holiday_data, 'daily'),
                    'y': [0] * len(holiday_data),
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Holidays',
                    'marker': {'color': 'red', 'size': 8, 'symbol': 'star'},
                    'showlegend': True
                }]
        
        charts['hprd_trend'] = {
            'data': [
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': hprd_df['Total_Nurse_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total HPRD (All Staff)',
                    'line': {'color': '#d62728', 'width': 3}
                },
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': hprd_df['Nurse_Staff_HPRD_Excl_Admin'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Staff HPRD (excl. Admin & DON)',
                    'line': {'color': '#9467bd'}
                },
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': hprd_df['Total_RN_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN HPRD (Total)',
                    'line': {'color': '#1f77b4'}
                },
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': hprd_df['Total_LPN_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN HPRD (Total)',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': hprd_df['Total_Nurse_Aide_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide HPRD (Total)',
                    'line': {'color': '#2ca02c'}
                }
            ] + hprd_holiday_markers,
            'layout': {
                'title': {
                    'text': 'Daily HPRD Trends',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'HPRD'},
                'height': 450,
                'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
            }
        }
        
        # Day of week comparison
        dow_summary = filtered_df.groupby('DayOfWeek').agg({
            'Total_RN_HPRD': 'mean',
            'Total_LPN_HPRD': 'mean',
            'Total_Nurse_Aide_HPRD': 'mean',
            'Nurse_Staff_HPRD_Excl_Admin': 'mean',
            'Total_Nurse_HPRD': 'mean'
        }).reset_index()
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_summary['DayOfWeek'] = pd.Categorical(dow_summary['DayOfWeek'], categories=day_order, ordered=True)
        dow_summary = dow_summary.sort_values('DayOfWeek')
        
        charts['dow_comparison'] = {
            'data': [
                {
                    'x': dow_summary['DayOfWeek'].tolist(),
                    'y': dow_summary['Total_Nurse_HPRD'].tolist(),
                    'type': 'bar',
                    'name': 'Total HPRD (All Staff)',
                    'marker': {'color': '#d62728'}
                },
                {
                    'x': dow_summary['DayOfWeek'].tolist(),
                    'y': dow_summary['Nurse_Staff_HPRD_Excl_Admin'].tolist(),
                    'type': 'bar',
                    'name': 'Nurse Staff HPRD (excl. Admin & DON)',
                    'marker': {'color': '#9467bd'}
                },
                {
                    'x': dow_summary['DayOfWeek'].tolist(),
                    'y': dow_summary['Total_RN_HPRD'].tolist(),
                    'type': 'bar',
                    'name': 'RN HPRD (Total)',
                    'marker': {'color': '#1f77b4'}
                },
                {
                    'x': dow_summary['DayOfWeek'].tolist(),
                    'y': dow_summary['Total_LPN_HPRD'].tolist(),
                    'type': 'bar',
                    'name': 'LPN HPRD (Total)',
                    'marker': {'color': '#ff7f0e'}
                },
                {
                    'x': dow_summary['DayOfWeek'].tolist(),
                    'y': dow_summary['Total_Nurse_Aide_HPRD'].tolist(),
                    'type': 'bar',
                    'name': 'Nurse Aide HPRD (Total)',
                    'marker': {'color': '#2ca02c'}
                }
            ],
            'layout': {
                'title': {
                    'text': 'Average HPRD by Day of Week',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {},
                'yaxis': {'title': 'HPRD'},
                'height': 450,
                'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
            }
        }
        
        # Hours trend
        charts['hours_trend'] = {
            'data': [
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': hours_df['Total_Staff_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total Nurse',
                    'line': {'color': '#d62728', 'width': 3}
                },
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': hours_df['Nurse_Staff_Hours_Excl_Admin'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Staff (excl. Admin/DON)',
                    'line': {'color': '#9467bd'}
                },
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': hours_df['Total_RN_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN (Total)',
                    'line': {'color': '#1f77b4'}
                },
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': hours_df['Total_LPN_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN (Total)',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': hours_df['Total_Nurse_Aide_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide (Total)',
                    'line': {'color': '#2ca02c'}
                }
            ] + hours_holiday_markers,
            'layout': {
                'title': {
                    'text': 'Daily Hours Trends',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'Hours'},
                'height': 450,
                'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
            }
        }
        
        # Census trend
        charts['census_trend'] = {
            'data': [
                {
                    'x': format_dates_for_view_mode(census_df, census_view),
                    'y': census_df['MDScensus'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Census',
                    'line': {'color': '#d62728'}
                }
            ],
            'layout': {
                'title': {
                    'text': 'Daily Census Trend',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'Census'},
                'height': 450,
                'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
            }
        }
        
        # Contract percentage trend
        charts['contract_trend'] = {
            'data': [
                {
                    'x': format_dates_for_view_mode(contract_df, contract_view),
                    'y': contract_df['Total_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total Contract %',
                    'line': {'color': '#d62728', 'width': 3}
                },
                {
                    'x': format_dates_for_view_mode(contract_df, contract_view),
                    'y': contract_df['RN_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN Contract %',
                    'line': {'color': '#1f77b4'}
                },
                {
                    'x': format_dates_for_view_mode(contract_df, contract_view),
                    'y': contract_df['LPN_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN Contract %',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': format_dates_for_view_mode(contract_df, contract_view),
                    'y': contract_df['CNA_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'CNA Contract %',
                    'line': {'color': '#2ca02c'}
                }
            ],
            'layout': {
                'title': {
                    'text': 'Contract Percentage Trends',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'Contract Percentage (%)', 'range': [0, None]},
                'height': 450,
                'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
            }
        }
        
        # Get filter information for chart titles
        filter_info = get_filter_description(start_date, end_date, quarter, day_of_week, show_holidays_only)
        
        # Update chart titles with filter information
        if filter_info and filter_info != "All Data (2017-2025)":
            charts['hprd_trend']['layout']['title']['text'] = f"Daily HPRD Trends<br>{filter_info}"
            charts['dow_comparison']['layout']['title']['text'] = f"Average HPRD by Day of Week<br>{filter_info}"
            charts['hours_trend']['layout']['title']['text'] = f"Daily Hours Trends<br>{filter_info}"
            charts['census_trend']['layout']['title']['text'] = f"Daily Census Trend<br>{filter_info}"
            charts['contract_trend']['layout']['title']['text'] = f"Contract Percentage Trends<br>{filter_info}"
        
        return jsonify({
            'charts': charts,
            'filter_info': filter_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/chart_aggregated')
def get_chart_aggregated():
    """Get chart data aggregated by quarter, month, or year for any chart type"""
    try:
        aggregation_type = request.args.get('type', 'quarter')  # 'quarter', 'month', or 'year'
        chart_type = request.args.get('chart_type', 'hprd')  # 'hprd', 'hours', 'census', 'contract'
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        quarter = request.args.get('quarter', 'all')
        year = request.args.get('year', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        show_holidays_only = request.args.get('show_holidays_only', 'false').lower() == 'true'
        
        # Apply filters (same logic as main charts)
        filtered_df = global_df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] <= end_date]
        if quarter != 'all':
            # Handle multiple quarters (comma-separated)
            if ',' in quarter:
                quarter_list = [q.strip() for q in quarter.split(',')]
                filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarter_list)]
            else:
                filtered_df = filtered_df[filtered_df['CY_Qtr'] == quarter]
        if year != 'all':
            # Handle multiple years (comma-separated)
            years = [int(y.strip()) for y in year.split(',')]
            filtered_df = filtered_df[filtered_df['WorkDate'].dt.year.isin(years)]
        if day_of_week != 'all':
            filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
        if show_holidays_only:
            filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
        
        if filtered_df.empty:
            return jsonify({'error': 'No data available for the selected filters'})
        
        # Sort by date
        filtered_df = filtered_df.sort_values('WorkDate')
        
        # Aggregate data based on type
        if aggregation_type == 'quarter':
            agg_data = filtered_df.groupby('CY_Qtr').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean',
                'Total_Staff_Hours': 'mean',
                'Nurse_Staff_Hours_Excl_Admin': 'mean',
                'Total_RN_Hours': 'mean',
                'Total_LPN_Hours': 'mean',
                'Total_Nurse_Aide_Hours': 'mean',
                'MDScensus': 'mean',
                'Total_Contract_Pct': 'mean',
                'RN_Contract_Pct': 'mean',
                'LPN_Contract_Pct': 'mean',
                'CNA_Contract_Pct': 'mean'
            }).round(2)
            # Convert 2017Q1 format to Q1 2017 format
            x_values = []
            for quarter in agg_data.index.tolist():
                if 'Q' in quarter:
                    year = quarter.split('Q')[0]
                    quarter_num = quarter.split('Q')[1]
                    x_values.append(f"Q{quarter_num} {year}")
                else:
                    x_values.append(quarter)
            title_suffix = "Quarterly"
        elif aggregation_type == 'year':
            filtered_df['Year'] = filtered_df['WorkDate'].dt.year.astype(str)
            agg_data = filtered_df.groupby('Year').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean',
                'Total_Staff_Hours': 'mean',
                'Nurse_Staff_Hours_Excl_Admin': 'mean',
                'Total_RN_Hours': 'mean',
                'Total_LPN_Hours': 'mean',
                'Total_Nurse_Aide_Hours': 'mean',
                'MDScensus': 'mean',
                'Total_Contract_Pct': 'mean',
                'RN_Contract_Pct': 'mean',
                'LPN_Contract_Pct': 'mean',
                'CNA_Contract_Pct': 'mean'
            }).round(2)
            x_values = agg_data.index.tolist()
            title_suffix = "Annually"
        else:  # month
            filtered_df['YearMonth'] = filtered_df['WorkDate'].dt.to_period('M').astype(str)
            agg_data = filtered_df.groupby('YearMonth').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean',
                'Total_Staff_Hours': 'mean',
                'Nurse_Staff_Hours_Excl_Admin': 'mean',
                'Total_RN_Hours': 'mean',
                'Total_LPN_Hours': 'mean',
                'Total_Nurse_Aide_Hours': 'mean',
                'MDScensus': 'mean',
                'Total_Contract_Pct': 'mean',
                'RN_Contract_Pct': 'mean',
                'LPN_Contract_Pct': 'mean',
                'CNA_Contract_Pct': 'mean'
            }).round(2)
            x_values = agg_data.index.tolist()
            title_suffix = "Monthly"
        
        # Create chart data based on chart type
        if chart_type == 'hprd':
            chart_data = [
                {
                    'x': x_values,
                    'y': agg_data['Total_Nurse_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total HPRD (All Staff)',
                    'line': {'color': '#d62728', 'width': 3}
                },
                {
                    'x': x_values,
                    'y': agg_data['Nurse_Staff_HPRD_Excl_Admin'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Staff HPRD (excl. Admin & DON)',
                    'line': {'color': '#9467bd'}
                },
                {
                    'x': x_values,
                    'y': agg_data['Total_RN_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN HPRD',
                    'line': {'color': '#2ca02c'}
                },
                {
                    'x': x_values,
                    'y': agg_data['Total_LPN_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN HPRD',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': x_values,
                    'y': agg_data['Total_Nurse_Aide_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide HPRD',
                    'line': {'color': '#1f77b4'}
                }
            ]
            y_title = 'HPRD'
        elif chart_type == 'hours':
            chart_data = [
                {
                    'x': x_values,
                    'y': agg_data['Total_Staff_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total Staff Hours',
                    'line': {'color': '#d62728', 'width': 3}
                },
                {
                    'x': x_values,
                    'y': agg_data['Nurse_Staff_Hours_Excl_Admin'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Staff Hours (excl. Admin & DON)',
                    'line': {'color': '#9467bd'}
                },
                {
                    'x': x_values,
                    'y': agg_data['Total_RN_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN Hours',
                    'line': {'color': '#2ca02c'}
                },
                {
                    'x': x_values,
                    'y': agg_data['Total_LPN_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN Hours',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': x_values,
                    'y': agg_data['Total_Nurse_Aide_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide Hours',
                    'line': {'color': '#1f77b4'}
                }
            ]
            y_title = 'Hours'
        elif chart_type == 'census':
            chart_data = [
                {
                    'x': x_values,
                    'y': agg_data['MDScensus'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Resident Census',
                    'line': {'color': '#2ca02c', 'width': 3}
                }
            ]
            y_title = 'Residents'
        elif chart_type == 'contract':
            chart_data = [
                {
                    'x': x_values,
                    'y': agg_data['Total_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total Contract %',
                    'line': {'color': '#d62728', 'width': 3}
                },
                {
                    'x': x_values,
                    'y': agg_data['RN_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN Contract %',
                    'line': {'color': '#2ca02c'}
                },
                {
                    'x': x_values,
                    'y': agg_data['LPN_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN Contract %',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': x_values,
                    'y': agg_data['CNA_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'CNA Contract %',
                    'line': {'color': '#1f77b4'}
                }
            ]
            y_title = 'Percentage (%)'
        else:
            return jsonify({'error': 'Invalid chart type'})
        
        layout = {
            'title': {
                'text': f'Average {chart_type.title()} {title_suffix}',
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'nticks': 10,
                'tickangle': 45
            },
            'yaxis': {'title': y_title},
            'height': 450,
            'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
        }
        
        return jsonify({
            'data': chart_data,
            'layout': layout,
            'aggregation_type': aggregation_type,
            'chart_type': chart_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/hprd_aggregated')
def get_hprd_aggregated():
    """Get HPRD data aggregated by quarter or month"""
    try:
        aggregation_type = request.args.get('type', 'quarter')  # 'quarter' or 'month'
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        quarter = request.args.get('quarter', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        show_holidays_only = request.args.get('show_holidays_only', 'false').lower() == 'true'
        
        # Apply filters (same logic as main charts)
        filtered_df = global_df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] <= end_date]
        if quarter != 'all':
            # Handle multiple quarters (comma-separated)
            if ',' in quarter:
                quarter_list = [q.strip() for q in quarter.split(',')]
                filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarter_list)]
            else:
                filtered_df = filtered_df[filtered_df['CY_Qtr'] == quarter]
        if day_of_week != 'all':
            filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
        if show_holidays_only:
            filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
        
        if filtered_df.empty:
            return jsonify({'error': 'No data available for the selected filters'})
        
        # Sort by date
        filtered_df = filtered_df.sort_values('WorkDate')
        
        # Aggregate data based on type
        if aggregation_type == 'quarter':
            agg_data = filtered_df.groupby('CY_Qtr').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean'
            }).round(2)
            # Convert 2017Q1 format to Q1 2017 format
            x_values = []
            for quarter in agg_data.index.tolist():
                if 'Q' in quarter:
                    year = quarter.split('Q')[0]
                    quarter_num = quarter.split('Q')[1]
                    x_values.append(f"Q{quarter_num} {year}")
                else:
                    x_values.append(quarter)
            title_suffix = "by Quarter"
        elif aggregation_type == 'year':
            filtered_df['Year'] = filtered_df['WorkDate'].dt.year.astype(str)
            agg_data = filtered_df.groupby('Year').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean'
            }).round(2)
            x_values = agg_data.index.tolist()
            title_suffix = "by Year"
        else:  # month
            filtered_df['YearMonth'] = filtered_df['WorkDate'].dt.to_period('M').astype(str)
            agg_data = filtered_df.groupby('YearMonth').agg({
                'Total_Nurse_HPRD': 'mean',
                'Nurse_Staff_HPRD_Excl_Admin': 'mean',
                'Total_RN_HPRD': 'mean',
                'Total_LPN_HPRD': 'mean',
                'Total_Nurse_Aide_HPRD': 'mean'
            }).round(2)
            x_values = agg_data.index.tolist()
            title_suffix = "by Month"
        
        # Create chart data
        chart_data = [
            {
                'x': x_values,
                'y': agg_data['Total_Nurse_HPRD'].fillna(0).tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Total HPRD (All Staff)',
                'line': {'color': '#d62728', 'width': 3}
            },
            {
                'x': x_values,
                'y': agg_data['Nurse_Staff_HPRD_Excl_Admin'].fillna(0).tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Nurse Staff HPRD (excl. Admin & DON)',
                'line': {'color': '#9467bd'}
            },
            {
                'x': x_values,
                'y': agg_data['Total_RN_HPRD'].fillna(0).tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'RN HPRD',
                'line': {'color': '#2ca02c'}
            },
            {
                'x': x_values,
                'y': agg_data['Total_LPN_HPRD'].fillna(0).tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'LPN HPRD',
                'line': {'color': '#ff7f0e'}
            },
            {
                'x': x_values,
                'y': agg_data['Total_Nurse_Aide_HPRD'].fillna(0).tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Nurse Aide HPRD',
                'line': {'color': '#1f77b4'}
            }
        ]
        
        layout = {
            'title': {
                'text': f'Average HPRD {title_suffix}',
                'x': 0.5,
                'xanchor': 'center'
            },
            'xaxis': {
                'nticks': 10,
                'tickangle': 45
            },
            'yaxis': {'title': 'HPRD'},
            'height': 450,
            'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
        }
        
        return jsonify({
            'data': chart_data,
            'layout': layout,
            'aggregation_type': aggregation_type
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/quarters')
def get_quarters():
    """Get available quarters"""
    quarters = sorted(global_df['CY_Qtr'].unique().tolist())
    return jsonify(quarters)

@app.route('/api/date_range')
def get_date_range():
    """Get available date range (filtered to valid PBJ data from 2017 onwards)"""
    # Filter out any data before 2017 (invalid/outlier data)
    # Use explicit date filtering to avoid timezone issues
    from datetime import datetime
    start_2017 = datetime(2017, 1, 1)
    valid_data = global_df[global_df['WorkDate'] >= start_2017]
    
    if len(valid_data) == 0:
        # Fallback if no valid data found
        return jsonify({
            'min_date': '2017-01-01',
            'max_date': '2025-12-31'
        })
    
    # Use .date() to ensure we get just the date part without time/timezone issues
    min_date = valid_data['WorkDate'].min().date()
    max_date = valid_data['WorkDate'].max().date()
    
    return jsonify({
        'min_date': min_date.strftime('%Y-%m-%d'),
        'max_date': max_date.strftime('%Y-%m-%d')
    })

@app.route('/api/data_completeness')
def get_data_completeness():
    """Analyze data completeness and identify missing quarters/days"""
    completeness_issues = []
    
    # Get all quarters that should exist (2017Q1 to 2025Q1)
    expected_quarters = []
    for year in range(2017, 2026):
        for quarter in range(1, 5):
            if year == 2025 and quarter > 1:  # Only Q1 2025 exists
                break
            expected_quarters.append(f"{year}Q{quarter}")
    
    # Check for missing quarters
    actual_quarters = set(global_df['CY_Qtr'].unique())
    missing_quarters = [q for q in expected_quarters if q not in actual_quarters]
    
    if missing_quarters:
        completeness_issues.append({
            'type': 'missing_quarter',
            'severity': 'high',
            'message': f"Missing {len(missing_quarters)} quarter(s): {', '.join(missing_quarters)}"
        })
    
    # Check for incomplete quarters (missing days)
    for quarter in actual_quarters:
        quarter_data = global_df[global_df['CY_Qtr'] == quarter]
        if len(quarter_data) > 0:
            # Calculate expected days in quarter
            year = int(quarter[:4])
            q_num = int(quarter[5])
            
            if q_num == 1:
                expected_days = 90  # Jan-Mar
            elif q_num == 2:
                expected_days = 91  # Apr-Jun
            elif q_num == 3:
                expected_days = 92  # Jul-Sep
            else:  # q_num == 4
                expected_days = 92  # Oct-Dec
            
            # Adjust for leap years in Q1
            if q_num == 1 and year % 4 == 0:
                expected_days = 91
            
            actual_days = len(quarter_data)
            missing_days = expected_days - actual_days
            
            if missing_days > 0:
                severity = 'high' if missing_days > 10 else 'medium' if missing_days > 5 else 'low'
                completeness_issues.append({
                    'type': 'incomplete_quarter',
                    'severity': severity,
                    'quarter': quarter,
                    'expected_days': expected_days,
                    'actual_days': actual_days,
                    'missing_days': missing_days,
                    'message': f"{quarter}: {missing_days} missing days ({actual_days}/{expected_days})"
                })
    
    # Check for data gaps (consecutive missing days)
    df_sorted = global_df.sort_values('WorkDate')
    date_gaps = []
    
    for i in range(1, len(df_sorted)):
        prev_date = df_sorted.iloc[i-1]['WorkDate']
        curr_date = df_sorted.iloc[i]['WorkDate']
        days_diff = (curr_date - prev_date).days
        
        if days_diff > 1:  # Gap of more than 1 day
            gap_start = prev_date + timedelta(days=1)
            gap_end = curr_date - timedelta(days=1)
            gap_days = days_diff - 1
            
            severity = 'high' if gap_days > 7 else 'medium' if gap_days > 3 else 'low'
            date_gaps.append({
                'start_date': gap_start.strftime('%Y-%m-%d'),
                'end_date': gap_end.strftime('%Y-%m-%d'),
                'gap_days': gap_days,
                'severity': severity
            })
    
    if date_gaps:
        # Group consecutive gaps
        gap_summary = {}
        for gap in date_gaps:
            quarter = global_df[global_df['WorkDate'] == gap['start_date']]['CY_Qtr'].iloc[0] if len(global_df[global_df['WorkDate'] == gap['start_date']]) > 0 else 'Unknown'
            if quarter not in gap_summary:
                gap_summary[quarter] = []
            gap_summary[quarter].append(gap)
        
        for quarter, gaps in gap_summary.items():
            total_gap_days = sum(gap['gap_days'] for gap in gaps)
            max_severity = max(gap['severity'] for gap in gaps)
            # Skip "Unknown" quarters - they're likely edge cases
            if quarter != 'Unknown':
                completeness_issues.append({
                    'type': 'data_gaps',
                    'severity': max_severity,
                    'quarter': quarter,
                    'gap_count': len(gaps),
                    'total_gap_days': total_gap_days,
                    'message': f"{quarter}: {len(gaps)} gap(s), {total_gap_days} missing days"
                })
    
    # Check for missing census data (days with 0 census)
    zero_census_data = global_df[global_df['MDScensus'] == 0]
    if len(zero_census_data) > 0:
        # Group by quarter for better reporting
        zero_census_by_quarter = zero_census_data.groupby('CY_Qtr').size()
        for quarter, count in zero_census_by_quarter.items():
            severity = 'high' if count > 10 else 'medium' if count > 5 else 'low'
            completeness_issues.append({
                'type': 'zero_census',
                'severity': severity,
                'quarter': quarter,
                'count': int(count),
                'message': f"{quarter}: {count} days with 0 census reported"
            })
    
    # Check for zero staffing hours when census > 0 (facility reported 0 staffing but had residents)
    zero_staffing_with_census = global_df[(global_df['Total_Staff_Hours'] == 0) & (global_df['MDScensus'] > 0)]
    if len(zero_staffing_with_census) > 0:
        # Group by quarter for better reporting
        zero_staffing_by_quarter = zero_staffing_with_census.groupby('CY_Qtr').size()
        for quarter, count in zero_staffing_by_quarter.items():
            severity = 'high' if count > 10 else 'medium' if count > 5 else 'low'
            completeness_issues.append({
                'type': 'zero_staffing_with_census',
                'severity': severity,
                'quarter': quarter,
                'count': int(count),
                'message': f"{quarter}: {count} days with 0 staffing hours but census > 0"
            })
    
    return jsonify({
        'issues': completeness_issues,
        'total_issues': len(completeness_issues),
        'has_issues': len(completeness_issues) > 0
    })

@app.route('/api/day_comparison')
def get_day_comparison():
    """Compare a specific day to other similar days"""
    try:
        target_date = request.args.get('target_date')
        comparison_type = request.args.get('comparison_type', 'month')  # month, quarter, year, custom, same_dow
        
        if not target_date:
            return jsonify({'error': 'target_date is required'})
        
        target_date = pd.to_datetime(target_date)
        target_day = global_df[global_df['WorkDate'] == target_date]
        
        if len(target_day) == 0:
            return jsonify({'error': 'No data found for target date'})
        
        target_record = target_day.iloc[0]
        
        # Get comparison data based on type
        if comparison_type == 'month':
            # Same month, different years
            comparison_data = global_df[
                (global_df['WorkDate'].dt.month == target_date.month) & 
                (global_df['WorkDate'] != target_date)
            ]
        elif comparison_type == 'quarter':
            # Same quarter, different years
            quarter = f"{target_date.year}Q{(target_date.month-1)//3 + 1}"
            comparison_data = global_df[
                (global_df['CY_Qtr'].str.contains(f"Q{(target_date.month-1)//3 + 1}")) & 
                (global_df['WorkDate'] != target_date)
            ]
        elif comparison_type == 'year':
            # Same year, different dates
            comparison_data = global_df[
                (global_df['WorkDate'].dt.year == target_date.year) & 
                (global_df['WorkDate'] != target_date)
            ]
        elif comparison_type == 'same_dow':
            # Same day of week
            comparison_data = global_df[
                (global_df['DayOfWeek'] == target_record['DayOfWeek']) & 
                (global_df['WorkDate'] != target_date)
            ]
        else:  # custom
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            if not start_date or not end_date:
                return jsonify({'error': 'start_date and end_date required for custom comparison'})
            
            comparison_data = global_df[
                (global_df['WorkDate'] >= start_date) & 
                (global_df['WorkDate'] <= end_date) & 
                (global_df['WorkDate'] != target_date)
            ]
        
        # Calculate comparison statistics
        target_data = {
            'census': float(target_record['MDScensus']),
            'rn_hprd': float(target_record['RN_HPRD']),
            'lpn_hprd': float(target_record['LPN_HPRD']),
            'cna_hprd': float(target_record['CNA_HPRD']),
            'total_hprd': float(target_record['Total_Staff_HPRD']),
            'rn_hours': float(target_record['Hrs_RN']),
            'lpn_hours': float(target_record['Hrs_LPN']),
            'cna_hours': float(target_record['Hrs_CNA']),
            'rn_contract_hours': float(target_record['Hrs_RN_ctr']),
            'lpn_contract_hours': float(target_record['Hrs_LPN_ctr']),
            'cna_contract_hours': float(target_record['Hrs_CNA_ctr']),
            'rn_contract_pct': float(target_record['RN_Contract_Pct']),
            'lpn_contract_pct': float(target_record['LPN_Contract_Pct']),
            'cna_contract_pct': float(target_record['CNA_Contract_Pct'])
        }
        
        comparison_data = {
            'count': len(comparison_data),
            'avg_census': float(comparison_data['MDScensus'].mean()) if len(comparison_data) > 0 else 0,
            'avg_rn_hprd': float(comparison_data['RN_HPRD'].mean()) if len(comparison_data) > 0 else 0,
            'avg_lpn_hprd': float(comparison_data['LPN_HPRD'].mean()) if len(comparison_data) > 0 else 0,
            'avg_cna_hprd': float(comparison_data['CNA_HPRD'].mean()) if len(comparison_data) > 0 else 0,
            'avg_total_hprd': float(comparison_data['Total_Staff_HPRD'].mean()) if len(comparison_data) > 0 else 0,
            'avg_rn_hours': float(comparison_data['Hrs_RN'].mean()) if len(comparison_data) > 0 else 0,
            'avg_lpn_hours': float(comparison_data['Hrs_LPN'].mean()) if len(comparison_data) > 0 else 0,
            'avg_cna_hours': float(comparison_data['Hrs_CNA'].mean()) if len(comparison_data) > 0 else 0,
            'avg_rn_contract_hours': float(comparison_data['Hrs_RN_ctr'].mean()) if len(comparison_data) > 0 else 0,
            'avg_lpn_contract_hours': float(comparison_data['Hrs_LPN_ctr'].mean()) if len(comparison_data) > 0 else 0,
            'avg_cna_contract_hours': float(comparison_data['Hrs_CNA_ctr'].mean()) if len(comparison_data) > 0 else 0,
            'avg_rn_contract_pct': float(comparison_data['RN_Contract_Pct'].mean()) if len(comparison_data) > 0 else 0,
            'avg_lpn_contract_pct': float(comparison_data['LPN_Contract_Pct'].mean()) if len(comparison_data) > 0 else 0,
            'avg_cna_contract_pct': float(comparison_data['CNA_Contract_Pct'].mean()) if len(comparison_data) > 0 else 0,
            'std_rn_hprd': float(comparison_data['RN_HPRD'].std()) if len(comparison_data) > 0 else 0,
            'std_lpn_hprd': float(comparison_data['LPN_HPRD'].std()) if len(comparison_data) > 0 else 0,
            'std_cna_hprd': float(comparison_data['CNA_HPRD'].std()) if len(comparison_data) > 0 else 0,
            'std_total_hprd': float(comparison_data['Total_Staff_HPRD'].std()) if len(comparison_data) > 0 else 0,
            'std_census': float(comparison_data['MDScensus'].std()) if len(comparison_data) > 0 else 0
        }
        
        # Detect aberrations using statistical analysis
        aberrations = detect_aberrations(target_data, comparison_data)
        
        comparison_stats = {
            'target_date': target_date.strftime('%Y-%m-%d'),
            'target_day_of_week': target_record['DayOfWeek'],
            'target_data': target_data,
            'comparison_data': comparison_data,
            'aberrations': aberrations
        }
        
        return jsonify(comparison_stats)
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/pbj_source_link')
def get_pbj_source_link():
    """Get PBJ source link for a specific date and quarter."""
    try:
        target_date = request.args.get('date')
        quarter = request.args.get('quarter')
        provnum = request.args.get('provnum', '225500')
        
        if not target_date or not quarter:
            return jsonify({'error': 'date and quarter parameters are required'})
        
        # Generate the source link
        source_link = format_pbj_source_link(quarter, target_date, provnum)
        
        if source_link:
            return jsonify({
                'source_link': source_link,
                'url': generate_pbj_source_link(quarter, target_date, provnum)
            })
        else:
            return jsonify({'error': 'Could not generate source link'})
            
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/single_day_report')
def api_single_day_report():
    """Get comprehensive single day report with comparisons and aberrations"""
    try:
        target_date = request.args.get('date')
        if not target_date:
            return jsonify({'error': 'Date parameter required'})
        
        # Convert date string to datetime
        from datetime import datetime
        target_dt = datetime.strptime(target_date, '%Y-%m-%d')
        
        # Get target day data
        target_data = global_df[global_df['WorkDate'] == target_dt]
        if target_data.empty:
            return jsonify({'error': f'No data found for {target_date}'})
        
        target_row = target_data.iloc[0]
        target_quarter = target_row['CY_Qtr']
        target_year = target_dt.year
        target_dow = target_row['DayOfWeek']
        
        # Get comparison data
        quarter_data = global_df[global_df['CY_Qtr'] == target_quarter]
        year_data = global_df[global_df['WorkDate'].dt.year == target_year]
        # For day-of-week comparison, use all days of the same week day in the target year only
        dow_year_data = global_df[
            (global_df['DayOfWeek'] == target_dow) & 
            (global_df['WorkDate'].dt.year == target_year)
        ]
        
        # Calculate metrics for target day
        target_metrics = {
            'date': target_date,
            'day_of_week': target_dow,
            'quarter': target_quarter,
            'year': target_year,
            'census': round_financial(target_row['MDScensus']),
            'rn_hours': round_financial(target_row['Hrs_RN']),
            'rn_hprd': round_financial(target_row['RN_HPRD']),
            'lpn_hours': round_financial(target_row['Hrs_LPN']),
            'lpn_hprd': round_financial(target_row['LPN_HPRD']),
            'cna_hours': round_financial(target_row['Hrs_CNA']),
            'cna_hprd': round_financial(target_row['CNA_HPRD']),
            'total_rn_hours': round_financial(target_row['Total_RN_Hours']),
            'total_rn_hprd': round_financial(target_row['Total_RN_HPRD']),
            'total_lpn_hours': round_financial(target_row['Total_LPN_Hours']),
            'total_lpn_hprd': round_financial(target_row['Total_LPN_HPRD']),
            'total_nurse_aide_hours': round_financial(target_row['Total_Nurse_Aide_Hours']),
            'total_nurse_aide_hprd': round_financial(target_row['Total_Nurse_Aide_HPRD']),
            'nurse_staff_hours_excl_admin': round_financial(target_row['Nurse_Staff_Hours_Excl_Admin']),
            'nurse_staff_hprd_excl_admin': round_financial(target_row['Nurse_Staff_HPRD_Excl_Admin']),
            'total_staff_hours': round_financial(target_row['Total_Staff_Hours']),
            'total_staff_hprd': round_financial(target_row['Total_Staff_HPRD']),
            'rn_contract_pct': round_financial(target_row['RN_Contract_Pct']),
            'lpn_contract_pct': round_financial(target_row['LPN_Contract_Pct']),
            'cna_contract_pct': round_financial(target_row['CNA_Contract_Pct']),
            'total_contract_pct': round_financial(target_row['Total_Contract_Pct']),
            'cna_only_contract_pct': round_financial(target_row['CNA_Only_Contract_Pct']),
            'nurse_aide_contract_pct': round_financial(target_row['Nurse_Aide_Contract_Pct']),
            'lpn_only_contract_pct': round_financial(target_row['LPN_Only_Contract_Pct']),
            'total_lpn_contract_pct': round_financial(target_row['Total_LPN_Contract_Pct']),
            'is_holiday': bool(target_row['IsHoliday'])
        }
        
        # Calculate comparison averages
        def calculate_comparison_metrics(data, label):
            if data.empty:
                return None
            
            return {
                'label': label,
                'count': len(data),
                'census': round_financial(data['MDScensus'].mean()),
                'rn_hours': round_financial(data['Hrs_RN'].mean()),
                'rn_hprd': round_financial(data['RN_HPRD'].mean()),
                'lpn_hours': round_financial(data['Hrs_LPN'].mean()),
                'lpn_hprd': round_financial(data['LPN_HPRD'].mean()),
                'cna_hours': round_financial(data['Hrs_CNA'].mean()),
                'cna_hprd': round_financial(data['CNA_HPRD'].mean()),
                'total_rn_hours': round_financial(data['Total_RN_Hours'].mean()),
                'total_rn_hprd': round_financial(data['Total_RN_HPRD'].mean()),
                'total_lpn_hours': round_financial(data['Total_LPN_Hours'].mean()),
                'total_lpn_hprd': round_financial(data['Total_LPN_HPRD'].mean()),
                'total_nurse_aide_hours': round_financial(data['Total_Nurse_Aide_Hours'].mean()),
                'total_nurse_aide_hprd': round_financial(data['Total_Nurse_Aide_HPRD'].mean()),
                'nurse_staff_hours_excl_admin': round_financial(data['Nurse_Staff_Hours_Excl_Admin'].mean()),
                'nurse_staff_hprd_excl_admin': round_financial(data['Nurse_Staff_HPRD_Excl_Admin'].mean()),
                'total_staff_hours': round_financial(data['Total_Staff_Hours'].mean()),
                'total_staff_hprd': round_financial(data['Total_Staff_HPRD'].mean()),
                'rn_contract_pct': round_financial(data['RN_Contract_Pct'].mean()),
                'lpn_contract_pct': round_financial(data['LPN_Contract_Pct'].mean()),
                'cna_contract_pct': round_financial(data['CNA_Contract_Pct'].mean()),
                'total_contract_pct': round_financial(data['Total_Contract_Pct'].mean()),
                'cna_only_contract_pct': round_financial(data['CNA_Only_Contract_Pct'].mean()),
                'nurse_aide_contract_pct': round_financial(data['Nurse_Aide_Contract_Pct'].mean()),
                'lpn_only_contract_pct': round_financial(data['LPN_Only_Contract_Pct'].mean()),
                'total_lpn_contract_pct': round_financial(data['Total_LPN_Contract_Pct'].mean())
            }
        
        comparisons = {
            'quarter': calculate_comparison_metrics(quarter_data, f"Quarter {target_quarter}"),
            'year': calculate_comparison_metrics(year_data, f"Year {target_year}"),
            'dow': calculate_comparison_metrics(dow_year_data, f"{target_dow}s in {target_year}")
        }
        
        # Calculate aberrations (z-scores)
        def calculate_aberrations(target_val, comparison_data, metric_name, data_source):
            if comparison_data is None or comparison_data['count'] < 2:
                return None
            
            # Get the actual data for this metric
            if metric_name == 'census':
                values = data_source['MDScensus'].dropna()
            elif metric_name == 'rn_hours':
                values = data_source['Hrs_RN'].dropna()
            elif metric_name == 'rn_hprd':
                values = data_source['RN_HPRD'].dropna()
            elif metric_name == 'lpn_hours':
                values = data_source['Hrs_LPN'].dropna()
            elif metric_name == 'lpn_hprd':
                values = data_source['LPN_HPRD'].dropna()
            elif metric_name == 'cna_hours':
                values = data_source['Hrs_CNA'].dropna()
            elif metric_name == 'cna_hprd':
                values = data_source['CNA_HPRD'].dropna()
            elif metric_name == 'total_rn_hours':
                values = data_source['Total_RN_Hours'].dropna()
            elif metric_name == 'total_rn_hprd':
                values = data_source['Total_RN_HPRD'].dropna()
            elif metric_name == 'total_lpn_hours':
                values = data_source['Total_LPN_Hours'].dropna()
            elif metric_name == 'total_lpn_hprd':
                values = data_source['Total_LPN_HPRD'].dropna()
            elif metric_name == 'total_nurse_aide_hours':
                values = data_source['Total_Nurse_Aide_Hours'].dropna()
            elif metric_name == 'total_nurse_aide_hprd':
                values = data_source['Total_Nurse_Aide_HPRD'].dropna()
            elif metric_name == 'nurse_staff_hours_excl_admin':
                values = data_source['Nurse_Staff_Hours_Excl_Admin'].dropna()
            elif metric_name == 'nurse_staff_hprd_excl_admin':
                values = data_source['Nurse_Staff_HPRD_Excl_Admin'].dropna()
            elif metric_name == 'total_staff_hours':
                values = data_source['Total_Staff_Hours'].dropna()
            elif metric_name == 'total_staff_hprd':
                values = data_source['Total_Staff_HPRD'].dropna()
            elif metric_name == 'rn_contract_pct':
                values = data_source['RN_Contract_Pct'].dropna()
            elif metric_name == 'lpn_contract_pct':
                values = data_source['LPN_Contract_Pct'].dropna()
            elif metric_name == 'cna_contract_pct':
                values = data_source['CNA_Contract_Pct'].dropna()
            else:
                return None
            
            if len(values) < 2:
                return None
            
            mean_val = values.mean()
            std_val = values.std()
            
            if std_val == 0:
                return None
            
            z_score = (target_val - mean_val) / std_val
            
            # Calculate actual percentile ranking
            sorted_values = values.sort_values()
            rank = (sorted_values < target_val).sum() + 1
            total_days = len(values)
            percentile = (rank / total_days) * 100
            
            # Determine if it's an outlier and get ranking info
            is_outlier = False
            ranking_info = ""
            
            if 'contract' in metric_name:
                # For contract percentages, both high and low are concerning
                if abs(z_score) > 1.5:
                    color = 'red'
                    is_outlier = True
                    if z_score < 0:
                        ranking_info = f"{rank} lowest of {total_days} days"
                    else:
                        ranking_info = f"{rank} highest of {total_days} days"
                else:
                    color = 'normal'
            else:
                # For other metrics, low values are red, high values are green
                if z_score < -1.5:
                    color = 'red'
                    is_outlier = True
                    ranking_info = f"{rank} lowest of {total_days} days"
                elif z_score > 1.5:
                    color = 'green'
                    is_outlier = True
                    ranking_info = f"{rank} highest of {total_days} days"
                else:
                    color = 'normal'
            
            return {
                'z_score': round_financial(z_score),
                'color': color,
                'percentile': round_financial(percentile),
                'is_outlier': is_outlier,
                'ranking_info': ranking_info,
                'rank': int(rank),
                'total_days': int(total_days)
            }
        
        # Calculate aberrations for all metrics (quarter, year, and day-of-week based)
        aberrations = {}
        for metric in ['census', 'rn_hours', 'rn_hprd', 'lpn_hours', 'lpn_hprd', 'cna_hours', 'cna_hprd',
                      'total_rn_hours', 'total_rn_hprd', 'total_lpn_hours', 'total_lpn_hprd',
                      'total_nurse_aide_hours', 'total_nurse_aide_hprd', 'nurse_staff_hours_excl_admin', 'nurse_staff_hprd_excl_admin',
                      'total_staff_hours', 'total_staff_hprd', 'rn_contract_pct', 'lpn_contract_pct', 'cna_contract_pct']:
            target_val = target_metrics[metric]
            
            # Calculate quarter-based aberration
            quarter_aberration = calculate_aberrations(target_val, comparisons['quarter'], metric, quarter_data)
            # Calculate year-based aberration
            year_aberration = calculate_aberrations(target_val, comparisons['year'], metric, year_data)
            # Calculate day-of-week-based aberration
            dow_aberration = calculate_aberrations(target_val, comparisons['dow'], metric, dow_year_data)
            
            aberrations[metric] = {
                'quarter': quarter_aberration,
                'year': year_aberration,
                'dow': dow_aberration
            }
        
        # Get the actual facility provider number
        facility_provnum = str(target_row['PROVNUM']).zfill(6) if 'PROVNUM' in target_row else "Unknown"
        
        # Generate PBJ source links with actual provider number
        nurse_source_link = format_pbj_source_link(target_quarter, target_date, facility_provnum, "nurse")
        nonnurse_source_link = format_pbj_source_link(target_quarter, target_date, facility_provnum, "nonnurse")
        
        return jsonify({
            'target_metrics': target_metrics,
            'comparisons': comparisons,
            'aberrations': aberrations,
            'nurse_source_link': nurse_source_link,
            'nonnurse_source_link': nonnurse_source_link
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

def get_filter_description(start_date, end_date, quarter, day_of_week, holidays_only):
    """Generate a human-readable description of current filters for chart titles"""
    def format_date(date_str):
        """Convert YYYY-MM-DD to MM-DD-YYYY"""
        if not date_str:
            return ""
        try:
            from datetime import datetime
            dt = datetime.strptime(date_str, '%Y-%m-%d')
            return dt.strftime('%m-%d-%Y')
        except:
            return date_str
    
    def format_quarter(q):
        """Convert 2023Q1 to Q1 2023"""
        if not q:
            return ""
        try:
            year = q[:4]
            q_num = q[5]
            return f"Q{q_num} {year}"
        except:
            return q
    
    # Build the filter description
    date_part = ""
    filter_part = ""
    
    if start_date and end_date:
        start_formatted = format_date(start_date)
        end_formatted = format_date(end_date)
        date_part = f"{start_formatted} to {end_formatted}"
    elif start_date:
        date_part = f"From {format_date(start_date)}"
    elif end_date:
        date_part = f"Until {format_date(end_date)}"
    
    filters = []
    
    if quarter != 'all':
        # Format quarters nicely
        quarters = [q.strip() for q in quarter.split(',')]
        if len(quarters) == 1:
            filters.append(f"{format_quarter(quarters[0])}")
        else:
            # Create a range for multiple quarters
            formatted_quarters = [format_quarter(q) for q in quarters]
            if len(formatted_quarters) > 3:
                # Show range for many quarters
                first_quarter = formatted_quarters[0]
                last_quarter = formatted_quarters[-1]
                filters.append(f"{first_quarter} - {last_quarter}")
            else:
                # Show all quarters if 3 or fewer
                filters.append(f"{', '.join(formatted_quarters)}")
    
    if day_of_week != 'all':
        filters.append(f"Day: {day_of_week}")
    
    if holidays_only:
        filters.append("Holidays Only")
    
    if filters:
        filter_part = " | ".join(filters)
    
    # Create two-line title
    if date_part and filter_part:
        return f"{date_part}<br>{filter_part}"
    elif date_part:
        return date_part
    elif filter_part:
        return filter_part
    else:
        return "All Data (2017-2025)"

def detect_aberrations(target_data, comparison_data):
    """Detect statistical aberrations in the target day compared to historical data"""
    aberrations = []
    
    # Define metrics to analyze
    metrics = [
        ('census', 'Census'),
        ('rn_hprd', 'RN HPRD'),
        ('lpn_hprd', 'LPN HPRD'),
        ('cna_hprd', 'CNA HPRD'),
        ('total_hprd', 'Total HPRD'),
        ('rn_contract_pct', 'RN Contract %'),
        ('lpn_contract_pct', 'LPN Contract %'),
        ('cna_contract_pct', 'CNA Contract %')
    ]
    
    for metric_key, metric_name in metrics:
        target_value = target_data[metric_key]
        avg_value = comparison_data[f'avg_{metric_key}']
        std_value = comparison_data.get(f'std_{metric_key}', 0)
        
        if std_value > 0:  # Only analyze if we have standard deviation data
            # Calculate z-score (how many standard deviations from mean)
            z_score = (target_value - avg_value) / std_value
            
            # Determine aberration level
            if abs(z_score) >= 3.0:
                severity = 'EXTREME'
                color = 'danger'
            elif abs(z_score) >= 2.0:
                severity = 'HIGH'
                color = 'warning'
            elif abs(z_score) >= 1.5:
                severity = 'MODERATE'
                color = 'info'
            else:
                continue  # Not significant enough to report
            
            # Determine direction
            direction = 'HIGH' if z_score > 0 else 'LOW'
            
            # Calculate percentile
            if z_score > 0:
                percentile = min(99.9, 50 + (z_score * 34.1))  # Approximate percentile
            else:
                percentile = max(0.1, 50 - (abs(z_score) * 34.1))
            
            aberrations.append({
                'metric': metric_name,
                'target_value': target_value,
                'average_value': avg_value,
                'z_score': z_score,
                'severity': severity,
                'direction': direction,
                'percentile': percentile,
                'color': color,
                'description': f"{metric_name} was {direction} ({target_value:.3f} vs avg {avg_value:.3f}, {percentile:.1f}th percentile)"
            })
    
    # Sort by severity and z-score
    severity_order = {'EXTREME': 4, 'HIGH': 3, 'MODERATE': 2}
    aberrations.sort(key=lambda x: (severity_order.get(x['severity'], 1), abs(x['z_score'])), reverse=True)
    
    return aberrations

@app.route('/api/quarterly-stats')
def get_quarterly_stats():
    """Get comprehensive quarterly statistics for all positions"""
    try:
        # Get all filter parameters
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        quarter = request.args.get('quarter', 'all')
        year = request.args.get('year', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        show_holidays_only = request.args.get('holidays_only', 'false') == 'true'
        
        # Filter data by all parameters
        global global_df
        filtered_df = global_df.copy()
        
        if start_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] >= start_date]
        if end_date:
            filtered_df = filtered_df[filtered_df['WorkDate'] <= end_date]
        if quarter != 'all':
            # Handle multiple quarters (comma-separated)
            quarters = [q.strip() for q in quarter.split(',')]
            filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
        if year != 'all':
            # Handle multiple years (comma-separated)
            years = [int(y.strip()) for y in year.split(',')]
            filtered_df = filtered_df[filtered_df['WorkDate'].dt.year.isin(years)]
        if day_of_week != 'all':
            filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
        if show_holidays_only:
            filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
        
        # Group data by quarter
        quarterly_data = filtered_df.groupby('CY_Qtr').agg({
            # Total Staff
            'Total_Staff_Hours': ['sum', 'mean', 'median', 'std'],
            'Total_Staff_HPRD': ['mean', 'median', 'std'],
            
            # Nurse Staff (excluding admin)
            'Nurse_Staff_Hours_Excl_Admin': ['sum', 'mean', 'median', 'std'],
            'Nurse_Staff_HPRD_Excl_Admin': ['mean', 'median', 'std'],
            
            # Total RN (including admin and DON)
            'Total_RN_Hours': ['sum', 'mean', 'median', 'std'],
            'Total_RN_HPRD': ['mean', 'median', 'std'],
            
            # RN Direct Care
            'Hrs_RN': ['sum', 'mean', 'median', 'std'],
            'RN_HPRD': ['mean', 'median', 'std'],
            
            # RN Admin
            'Hrs_RNadmin': ['sum', 'mean', 'median', 'std'],
            
            # RN DON
            'Hrs_RNDON': ['sum', 'mean', 'median', 'std'],
            
            # Total LPN (including admin)
            'Total_LPN_Hours': ['sum', 'mean', 'median', 'std'],
            'Total_LPN_HPRD': ['mean', 'median', 'std'],
            
            # LPN Direct Care
            'Hrs_LPN': ['sum', 'mean', 'median', 'std'],
            'LPN_HPRD': ['mean', 'median', 'std'],
            
            # LPN Admin
            'Hrs_LPNadmin': ['sum', 'mean', 'median', 'std'],
            
            # Total CNA (including trainees and med aides)
            'Total_Nurse_Aide_Hours': ['sum', 'mean', 'median', 'std'],
            'Total_Nurse_Aide_HPRD': ['mean', 'median', 'std'],
            
            # CNA Direct Care
            'Hrs_CNA': ['sum', 'mean', 'median', 'std'],
            'CNA_HPRD': ['mean', 'median', 'std'],
            
            # NA Trainee
            'Hrs_NAtrn': ['sum', 'mean', 'median', 'std'],
            
            # Med Aide
            'Hrs_MedAide': ['sum', 'mean', 'median', 'std'],
            
            # Contract Staff
            'Hrs_RN_ctr': ['sum', 'mean', 'median', 'std'],
            'Hrs_LPN_ctr': ['sum', 'mean', 'median', 'std'],
            'Hrs_CNA_ctr': ['sum', 'mean', 'median', 'std'],
            
            # Census for HPRD calculations
            'MDScensus': ['mean', 'median', 'std']
        }).round(3)
        
        # Flatten column names
        quarterly_data.columns = ['_'.join(col).strip() for col in quarterly_data.columns.values]
        
        # Calculate zero counts for each position
        zero_counts = {}
        position_columns = {
            'total_staff': 'Total_Staff_Hours',
            'total_rn': 'Total_RN_Hours', 
            'rn_direct': 'Hrs_RN',
            'rn_admin': 'Hrs_RNadmin',
            'rn_don': 'Hrs_RNDON',
            'total_lpn': 'Total_LPN_Hours',
            'lpn_direct': 'Hrs_LPN',
            'lpn_admin': 'Hrs_LPNadmin',
            'total_cna': 'Total_Nurse_Aide_Hours',
            'cna_direct': 'Hrs_CNA',
            'na_trainee': 'Hrs_NAtrn',
            'med_aide': 'Hrs_MedAide',
            'rn_contract': 'Hrs_RN_ctr',
            'lpn_contract': 'Hrs_LPN_ctr',
            'cna_contract': 'Hrs_CNA_ctr'
        }
        
        for pos_key, col_name in position_columns.items():
            if col_name in filtered_df.columns:
                zero_counts[pos_key] = len(filtered_df[filtered_df[col_name] == 0])
            else:
                zero_counts[pos_key] = 0
        
        # Structure the response
        quarterly_stats = {}
        
        # Total Nurse Staff
        quarterly_stats['total_nurse_staff'] = {
            'total_hours': {
                'mean': float(filtered_df['Total_Staff_Hours'].mean()),
                'median': float(filtered_df['Total_Staff_Hours'].median()),
                'std_dev': float(filtered_df['Total_Staff_Hours'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Total_Staff_HPRD'].mean()),
                'median': float(filtered_df['Total_Staff_HPRD'].median()),
                'std_dev': float(filtered_df['Total_Staff_HPRD'].std())
            },
            'zero_count': zero_counts['total_staff']
        }
        
        # Direct Staff (excl. Admin, DON)
        quarterly_stats['direct_staff_excl_admin'] = {
            'total_hours': {
                'mean': float(filtered_df['Nurse_Staff_Hours_Excl_Admin'].mean()),
                'median': float(filtered_df['Nurse_Staff_Hours_Excl_Admin'].median()),
                'std_dev': float(filtered_df['Nurse_Staff_Hours_Excl_Admin'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].mean()),
                'median': float(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].median()),
                'std_dev': float(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].std())
            },
            'zero_count': 0  # Calculate if needed
        }
        
        # Total RN
        quarterly_stats['total_rn'] = {
            'total_hours': {
                'mean': float(filtered_df['Total_RN_Hours'].mean()),
                'median': float(filtered_df['Total_RN_Hours'].median()),
                'std_dev': float(filtered_df['Total_RN_Hours'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Total_RN_HPRD'].mean()),
                'median': float(filtered_df['Total_RN_HPRD'].median()),
                'std_dev': float(filtered_df['Total_RN_HPRD'].std())
            },
            'zero_count': zero_counts['total_rn']
        }
        
        # Direct RN
        quarterly_stats['rn_direct'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RN'].mean()),
                'median': float(filtered_df['Hrs_RN'].median()),
                'std_dev': float(filtered_df['Hrs_RN'].std())
            },
            'hprd': {
                'mean': float(filtered_df['RN_HPRD'].mean()),
                'median': float(filtered_df['RN_HPRD'].median()),
                'std_dev': float(filtered_df['RN_HPRD'].std())
            },
            'zero_count': zero_counts['rn_direct']
        }
        
        # RN Admin
        quarterly_stats['rn_admin'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RNadmin'].mean()),
                'median': float(filtered_df['Hrs_RNadmin'].median()),
                'std_dev': float(filtered_df['Hrs_RNadmin'].std())
            },
            'hprd': {
                'mean': 0.0,  # Admin HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_admin']
        }
        
        # RN DON
        quarterly_stats['rn_don'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RNDON'].mean()),
                'median': float(filtered_df['Hrs_RNDON'].median()),
                'std_dev': float(filtered_df['Hrs_RNDON'].std())
            },
            'hprd': {
                'mean': 0.0,  # DON HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_don']
        }
        
        # Total LPN
        quarterly_stats['total_lpn'] = {
            'total_hours': {
                'mean': float(filtered_df['Total_LPN_Hours'].mean()),
                'median': float(filtered_df['Total_LPN_Hours'].median()),
                'std_dev': float(filtered_df['Total_LPN_Hours'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Total_LPN_HPRD'].mean()),
                'median': float(filtered_df['Total_LPN_HPRD'].median()),
                'std_dev': float(filtered_df['Total_LPN_HPRD'].std())
            },
            'zero_count': zero_counts['total_lpn']
        }
        
        # LPN
        quarterly_stats['lpn_direct'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_LPN'].mean()),
                'median': float(filtered_df['Hrs_LPN'].median()),
                'std_dev': float(filtered_df['Hrs_LPN'].std())
            },
            'hprd': {
                'mean': float(filtered_df['LPN_HPRD'].mean()),
                'median': float(filtered_df['LPN_HPRD'].median()),
                'std_dev': float(filtered_df['LPN_HPRD'].std())
            },
            'zero_count': zero_counts['lpn_direct']
        }
        
        # LPN Admin
        quarterly_stats['lpn_admin'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_LPNadmin'].mean()),
                'median': float(filtered_df['Hrs_LPNadmin'].median()),
                'std_dev': float(filtered_df['Hrs_LPNadmin'].std())
            },
            'hprd': {
                'mean': 0.0,  # Admin HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['lpn_admin']
        }
        
        # Total Nurse Aide
        quarterly_stats['total_cna'] = {
            'total_hours': {
                'mean': float(filtered_df['Total_Nurse_Aide_Hours'].mean()),
                'median': float(filtered_df['Total_Nurse_Aide_Hours'].median()),
                'std_dev': float(filtered_df['Total_Nurse_Aide_Hours'].std())
            },
            'hprd': {
                'mean': float(filtered_df['Total_Nurse_Aide_HPRD'].mean()),
                'median': float(filtered_df['Total_Nurse_Aide_HPRD'].median()),
                'std_dev': float(filtered_df['Total_Nurse_Aide_HPRD'].std())
            },
            'zero_count': zero_counts['total_cna']
        }
        
        # CNA
        quarterly_stats['cna_direct'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_CNA'].mean()),
                'median': float(filtered_df['Hrs_CNA'].median()),
                'std_dev': float(filtered_df['Hrs_CNA'].std())
            },
            'hprd': {
                'mean': float(filtered_df['CNA_HPRD'].mean()),
                'median': float(filtered_df['CNA_HPRD'].median()),
                'std_dev': float(filtered_df['CNA_HPRD'].std())
            },
            'zero_count': zero_counts['cna_direct']
        }
        
        # Med Aide
        quarterly_stats['med_aide'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_MedAide'].mean()),
                'median': float(filtered_df['Hrs_MedAide'].median()),
                'std_dev': float(filtered_df['Hrs_MedAide'].std())
            },
            'hprd': {
                'mean': 0.0,  # Med Aide HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['med_aide']
        }
        
        # NA Trainee
        quarterly_stats['na_trainee'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_NAtrn'].mean()),
                'median': float(filtered_df['Hrs_NAtrn'].median()),
                'std_dev': float(filtered_df['Hrs_NAtrn'].std())
            },
            'hprd': {
                'mean': 0.0,  # Trainee HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['na_trainee']
        }
        
        # Total Contract
        total_contract_hours = filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']
        quarterly_stats['total_contract'] = {
            'total_hours': {
                'mean': float(total_contract_hours.mean()),
                'median': float(total_contract_hours.median()),
                'std_dev': float(total_contract_hours.std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_contract'] + zero_counts['lpn_contract'] + zero_counts['cna_contract']
        }
        
        # Direct Care Contract
        direct_contract_hours = filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']
        quarterly_stats['direct_care_contract'] = {
            'total_hours': {
                'mean': float(direct_contract_hours.mean()),
                'median': float(direct_contract_hours.median()),
                'std_dev': float(direct_contract_hours.std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_contract'] + zero_counts['lpn_contract'] + zero_counts['cna_contract']
        }
        
        # Total RN Contract
        quarterly_stats['total_rn_contract'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RN_ctr'].mean()),
                'median': float(filtered_df['Hrs_RN_ctr'].median()),
                'std_dev': float(filtered_df['Hrs_RN_ctr'].std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_contract']
        }
        
        # Direct RN Contract
        quarterly_stats['direct_rn_contract'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_RN_ctr'].mean()),
                'median': float(filtered_df['Hrs_RN_ctr'].median()),
                'std_dev': float(filtered_df['Hrs_RN_ctr'].std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['rn_contract']
        }
        
        # Nurse Aide Contract
        quarterly_stats['nurse_aide_contract'] = {
            'total_hours': {
                'mean': float(filtered_df['Hrs_CNA_ctr'].mean()),
                'median': float(filtered_df['Hrs_CNA_ctr'].median()),
                'std_dev': float(filtered_df['Hrs_CNA_ctr'].std())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['cna_contract']
        }
        
        # Handle NaN values by converting them to None
        def clean_nan_values(obj):
            if isinstance(obj, dict):
                return {k: clean_nan_values(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [clean_nan_values(item) for item in obj]
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        cleaned_stats = clean_nan_values(quarterly_stats)
        return jsonify({
            'quarterly_stats': cleaned_stats,
            'sample_size': len(filtered_df)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/quarterly-data')
def get_quarterly_data():
    """Get quarterly data for all quarters with HPRD and hours"""
    try:
        # Group data by quarter and calculate quarterly averages (hours per day)
        quarterly_data = global_df.groupby('CY_Qtr').agg({
            'MDScensus': 'mean',
            'Total_Staff_Hours': 'mean',  # Average hours per day
            'Total_Staff_HPRD': 'mean',
            'Nurse_Staff_Hours_Excl_Admin': 'mean',  # Average hours per day
            'Nurse_Staff_HPRD_Excl_Admin': 'mean',
            'Total_RN_Hours': 'mean',  # Average hours per day
            'Total_RN_HPRD': 'mean',
            'Hrs_RN': 'mean',  # Average hours per day
            'RN_HPRD': 'mean',
            'Hrs_RNadmin': 'mean',  # Average hours per day
            'Hrs_RNDON': 'mean'  # Average hours per day
        }).round(2)
        
        # Calculate HPRD for admin and DON (they don't have pre-calculated HPRD)
        quarterly_data['RN_Admin_HPRD'] = (quarterly_data['Hrs_RNadmin'] / quarterly_data['MDScensus']).round(2)
        quarterly_data['RN_DON_HPRD'] = (quarterly_data['Hrs_RNDON'] / quarterly_data['MDScensus']).round(2)
        
        # Structure the response
        quarterly_data_dict = {}
        
        for quarter, row in quarterly_data.iterrows():
            quarterly_data_dict[quarter] = {
                'census': float(row['MDScensus']),
                'total_hprd': float(row['Total_Staff_HPRD']),
                'total_hours': float(row['Total_Staff_Hours']),
                'direct_hprd': float(row['Nurse_Staff_HPRD_Excl_Admin']),
                'direct_hours': float(row['Nurse_Staff_Hours_Excl_Admin']),
                'total_rn_hprd': float(row['Total_RN_HPRD']),
                'total_rn_hours': float(row['Total_RN_Hours']),
                'rn_hprd': float(row['RN_HPRD']),
                'rn_hours': float(row['Hrs_RN']),
                'rn_admin_hprd': float(row['RN_Admin_HPRD']),
                'rn_admin_hours': float(row['Hrs_RNadmin']),
                'rn_don_hprd': float(row['RN_DON_HPRD']),
                'rn_don_hours': float(row['Hrs_RNDON'])
            }
        
        return jsonify({'quarterly_data': quarterly_data_dict})
        
    except Exception as e:
        return jsonify({'error': str(e)})

# Dynamic dashboard - no initialization needed

def run_dashboard(provnum, port=5000):
    """Run the dynamic dashboard for a specific facility"""
    app_instance = create_dynamic_dashboard(provnum)
    if app_instance is None:
        print(f"ERROR: Failed to create dashboard for facility {provnum}")
        return
    
    print(f"Starting Dynamic Dashboard for facility {provnum}...")
    app_instance.run(debug=True, host='0.0.0.0', port=port)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python dynamic_facility_dashboard.py <PROVNUM>")
        print("Example: python dynamic_facility_dashboard.py 015009")
        sys.exit(1)
    
    provnum = sys.argv[1].strip().upper()
    
    # Validate provider number format - can contain letters and numbers, typically 6 characters
    if len(provnum) < 1 or len(provnum) > 6:
        print("❌ Please enter a valid provider number (1-6 characters, e.g., 015009 or 46A070)")
        sys.exit(1)
    
    # Pad with leading zeros if all digits, otherwise keep as-is
    if provnum.isdigit():
        provnum = provnum.zfill(6)
    
    run_dashboard(provnum)
