#!/usr/bin/env python3
"""
Comprehensive Facility 225500 Dashboard
Uses the complete CSV file for fast, detailed analysis
"""

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import json
from decimal import Decimal, ROUND_HALF_UP

app = Flask(__name__)

# Load the data once at startup
df = None

def initialize_data():
    """Initialize the global data variable"""
    global df
    load_data()  # load_data() already sets the global df variable
    return df

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

def load_data():
    """Load the facility 225500 data"""
    global df
    print("Loading facility 225500 data...")
    try:
        df = pd.read_csv('facility_225500_complete_data.csv')
        print(f"Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Clean up the data - remove any columns with .1 suffix and handle NaN values
    df = df.drop(columns=[col for col in df.columns if '.1' in col])
    
    # Fill NaN values with 0 for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Apply financial rounding to hours columns
    hours_columns = ['Hrs_RN', 'Hrs_LPN', 'Hrs_CNA', 'Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_LPNadmin', 
                     'Hrs_RN_ctr', 'Hrs_LPN_ctr', 'Hrs_CNA_ctr', 'Hrs_NAtrn', 'Hrs_MedAide']
    for col in hours_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: round_financial(x, 2))
    
    # Convert WorkDate to datetime
    df['WorkDate'] = pd.to_datetime(df['WorkDate'], format='%Y%m%d')
    
    # Add day of week
    df['DayOfWeek'] = df['WorkDate'].dt.day_name()
    df['DayOfWeekNum'] = df['WorkDate'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # Add month and year
    df['Month'] = df['WorkDate'].dt.month
    df['Year'] = df['WorkDate'].dt.year
    
    # Calculate HPRD for each position with proper rounding
    # RN HPRD includes direct care only (not admin/DON)
    df['RN_HPRD'] = (df['Hrs_RN'] / df['MDScensus']).apply(lambda x: round_financial(x, 2))
    # LPN HPRD includes direct care only (not admin)
    df['LPN_HPRD'] = (df['Hrs_LPN'] / df['MDScensus']).apply(lambda x: round_financial(x, 2))
    # CNA HPRD includes direct care only (not medaide/natr)
    df['CNA_HPRD'] = (df['Hrs_CNA'] / df['MDScensus']).apply(lambda x: round_financial(x, 2))
    # Total HPRD includes ALL staff (RN + RNadmin + RNDON + LPN + LPNadmin + CNA + NAtrn + MedAide)
    df['Total_Nurse_HPRD'] = ((df['Hrs_RN'] + df['Hrs_RNadmin'] + df['Hrs_RNDON'] + df['Hrs_LPN'] + df['Hrs_LPNadmin'] + df['Hrs_CNA'] + df['Hrs_NAtrn'] + df['Hrs_MedAide']) / df['MDScensus']).apply(lambda x: round_financial(x, 2))
    
    # Calculate additional metrics for outlier detection and table display
    df['Total_RN_Hours'] = (df['Hrs_RN'] + df['Hrs_RNadmin'] + df['Hrs_RNDON']).apply(lambda x: round_financial(x, 2))
    df['Total_RN_HPRD'] = (df['Total_RN_Hours'] / df['MDScensus']).apply(lambda x: round_financial(x, 2))
    df['Total_LPN_Hours'] = (df['Hrs_LPN'] + df['Hrs_LPNadmin']).apply(lambda x: round_financial(x, 2))
    df['Total_LPN_HPRD'] = (df['Total_LPN_Hours'] / df['MDScensus']).apply(lambda x: round_financial(x, 2))
    df['Total_Nurse_Aide_Hours'] = (df['Hrs_CNA'] + df['Hrs_MedAide'] + df['Hrs_NAtrn']).apply(lambda x: round_financial(x, 2))
    df['Total_Nurse_Aide_HPRD'] = (df['Total_Nurse_Aide_Hours'] / df['MDScensus']).apply(lambda x: round_financial(x, 2))
    
    # Nurse Staff Hours (excluding Admin & DON) - includes all direct care staff
    df['Nurse_Staff_Hours_Excl_Admin'] = (df['Hrs_RN'] + df['Hrs_LPN'] + df['Hrs_CNA'] + df['Hrs_NAtrn'] + df['Hrs_MedAide']).apply(lambda x: round_financial(x, 2))
    df['Nurse_Staff_HPRD_Excl_Admin'] = (df['Nurse_Staff_Hours_Excl_Admin'] / df['MDScensus']).apply(lambda x: round_financial(x, 2))
    
    # Total Nurse Hours (All Staff including admin/DON)
    df['Total_Nurse_Hours'] = (df['Hrs_RN'] + df['Hrs_RNadmin'] + df['Hrs_RNDON'] + df['Hrs_LPN'] + df['Hrs_LPNadmin'] + df['Hrs_CNA'] + df['Hrs_NAtrn'] + df['Hrs_MedAide']).apply(lambda x: round_financial(x, 2))
    
    # Total Staff Hours and HPRD
    df['Total_Staff_Hours'] = (df['Total_RN_Hours'] + df['Total_LPN_Hours'] + df['Total_Nurse_Aide_Hours']).apply(lambda x: round_financial(x, 2))
    df['Total_Staff_HPRD'] = (df['Total_Staff_Hours'] / df['MDScensus']).apply(lambda x: round_financial(x, 2))
    
    # Calculate contract percentages with proper rounding
    df['RN_Contract_Pct'] = (df['Hrs_RN_ctr'] / df['Hrs_RN'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    df['LPN_Contract_Pct'] = (df['Hrs_LPN_ctr'] / df['Hrs_LPN'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    df['CNA_Contract_Pct'] = (df['Hrs_CNA_ctr'] / df['Hrs_CNA'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # Calculate more granular contract percentages
    # CNA contract percentage (CNA only)
    df['CNA_Only_Contract_Pct'] = (df['Hrs_CNA_ctr'] / df['Hrs_CNA'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # Nurse Aide contract percentage (CNA + MedAide + NAtrn)
    total_nurse_aide_hours = df['Hrs_CNA'] + df['Hrs_MedAide'] + df['Hrs_NAtrn']
    total_nurse_aide_contract_hours = df['Hrs_CNA_ctr'] + df.get('Hrs_MedAide_ctr', 0) + df.get('Hrs_NAtrn_ctr', 0)
    df['Nurse_Aide_Contract_Pct'] = (total_nurse_aide_contract_hours / total_nurse_aide_hours * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # LPN contract percentage (LPN only, excluding admin)
    df['LPN_Only_Contract_Pct'] = (df['Hrs_LPN_ctr'] / df['Hrs_LPN'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # Total LPN contract percentage (LPN + LPN admin)
    total_lpn_hours = df['Hrs_LPN'] + df['Hrs_LPNadmin']
    total_lpn_contract_hours = df['Hrs_LPN_ctr'] + df.get('Hrs_LPNadmin_ctr', 0)
    df['Total_LPN_Contract_Pct'] = (total_lpn_contract_hours / total_lpn_hours * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
    # Total Contract Percentage (All contract hours / All total hours)
    total_contract_hours = (df['Hrs_RN_ctr'] + df['Hrs_LPN_ctr'] + df['Hrs_CNA_ctr'] + 
                           df.get('Hrs_RNadmin_ctr', 0) + df.get('Hrs_RNDON_ctr', 0) + 
                           df.get('Hrs_LPNadmin_ctr', 0) + df.get('Hrs_MedAide_ctr', 0) + 
                           df.get('Hrs_NAtrn_ctr', 0))
    df['Total_Contract_Pct'] = (total_contract_hours / df['Total_Nurse_Hours'] * 100).fillna(0).apply(lambda x: round_financial(x, 1))
    
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
    df['IsHoliday'] = df['WorkDate'].apply(is_federal_holiday)
    
    print(f"Loaded {len(df)} records from {df['WorkDate'].min().date()} to {df['WorkDate'].max().date()}")
    print(f"Calculated columns: {[col for col in df.columns if 'Total' in col or 'HPRD' in col]}")
    
    # Verify critical columns exist
    critical_cols = ['Total_Staff_HPRD', 'Total_Staff_Hours', 'Total_RN_HPRD', 'Total_LPN_HPRD', 'Total_Nurse_Aide_HPRD']
    missing_cols = [col for col in critical_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ MISSING CRITICAL COLUMNS: {missing_cols}")
    else:
        print(f"✓ All critical columns present")
    
    return df

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('facility_225500_dashboard.html')

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
        filtered_df = df.copy()
        
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
        filtered_df = df.copy()
        
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
            'avg_lpn_hprd': float(filtered_df['LPN_HPRD'].mean()) if len(filtered_df) > 0 else 0,
            'avg_cna_hprd': float(filtered_df['CNA_HPRD'].mean()) if len(filtered_df) > 0 else 0,
            'avg_nurse_staff_hprd_excl_admin': float(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].mean()) if len(filtered_df) > 0 else 0,
            'avg_total_hprd': float(filtered_df['Total_Staff_HPRD'].mean()) if len(filtered_df) > 0 else 0,
            'avg_rn_contract_pct': float(filtered_df['RN_Contract_Pct'].mean()) if len(filtered_df) > 0 else 0,
            'avg_lpn_contract_pct': float(filtered_df['LPN_Contract_Pct'].mean()) if len(filtered_df) > 0 else 0,
            'avg_cna_contract_pct': float(filtered_df['CNA_Contract_Pct'].mean()) if len(filtered_df) > 0 else 0,
            'avg_total_contract_pct': float(filtered_df['Total_Contract_Pct'].mean()) if len(filtered_df) > 0 else 0,
            'total_rn_hours': float(filtered_df['Hrs_RN'].sum()) if len(filtered_df) > 0 else 0,
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

@app.route('/api/charts')
def get_charts():
    """Get chart data"""
    try:
        # Check if data is loaded
        if df is None:
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
        
        # Filter data
        filtered_df = df.copy()
        
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
        
        # Check if we have any data after filtering
        if len(filtered_df) == 0:
            return jsonify({
                'charts': {},
                'filter_info': 'No data found for the selected filters',
                'error': 'No data available for the selected date range and filters'
            })
        
        # Check for required columns
        required_columns = ['WorkDate', 'Total_RN_HPRD', 'Total_LPN_HPRD', 'Total_Nurse_Aide_HPRD', 
                           'Total_Staff_HPRD', 'Total_RN_Hours', 'Total_LPN_Hours', 'Total_Nurse_Aide_Hours',
                           'MDScensus', 'RN_Contract_Pct', 'LPN_Contract_Pct', 'CNA_Contract_Pct', 'IsHoliday']
        
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
        
        # Debug: Check if Total_Staff_HPRD column exists
        print(f"DEBUG: Checking Total_Staff_HPRD column...")
        print(f"DEBUG: Total_Staff_HPRD in columns: {'Total_Staff_HPRD' in filtered_df.columns}")
        if 'Total_Staff_HPRD' in filtered_df.columns:
            print(f"DEBUG: Total_Staff_HPRD sample values: {filtered_df['Total_Staff_HPRD'].head().tolist()}")
        else:
            print(f"DEBUG: Available columns with 'Total': {[col for col in filtered_df.columns if 'Total' in col]}")
        
        # Daily HPRD trend
        # Add holiday indicators
        holiday_data = filtered_df[filtered_df['IsHoliday'] == True]
        holiday_markers = []
        if len(holiday_data) > 0:
            holiday_markers = [{
                'x': holiday_data['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
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
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Total_RN_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN HPRD (Total)',
                    'line': {'color': '#1f77b4'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Total_LPN_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN HPRD (Total)',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Total_Nurse_Aide_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide HPRD (Total)',
                    'line': {'color': '#2ca02c'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Nurse_Staff_HPRD_Excl_Admin'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Staff HPRD (excl. Admin & DON)',
                    'line': {'color': '#9467bd'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Total_Nurse_HPRD'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total HPRD (All Staff)',
                    'line': {'color': '#d62728', 'width': 3}
                }
            ] + holiday_markers,
            'layout': {
                'title': {
                    'text': 'Daily HPRD Trends',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'title': 'Date',
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'HPRD'},
                'height': 400
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
                    'y': dow_summary['Total_Nurse_HPRD'].tolist(),
                    'type': 'bar',
                    'name': 'Total HPRD (All Staff)',
                    'marker': {'color': '#d62728'}
                }
            ],
            'layout': {
                'title': {
                    'text': 'Average HPRD by Day of Week',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {'title': 'Day of Week'},
                'yaxis': {'title': 'HPRD'},
                'height': 400
            }
        }
        
        # Hours trend
        charts['hours_trend'] = {
            'data': [
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Total_RN_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN Hours (Total)',
                    'line': {'color': '#1f77b4'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Total_LPN_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN Hours (Total)',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Total_Nurse_Aide_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide Hours (Total)',
                    'line': {'color': '#2ca02c'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Nurse_Staff_Hours_Excl_Admin'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Staff Hours (excl. Admin & DON)',
                    'line': {'color': '#9467bd'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Total_Nurse_Hours'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total Nurse Hours (All Staff)',
                    'line': {'color': '#d62728', 'width': 3}
                }
            ] + holiday_markers,
            'layout': {
                'title': {
                    'text': 'Daily Hours Trends',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'title': 'Date',
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'Hours'},
                'height': 400
            }
        }
        
        # Census trend
        charts['census_trend'] = {
            'data': [
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['MDScensus'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Census',
                    'line': {'color': '#d62728'}
                }
            ] + holiday_markers,
            'layout': {
                'title': {
                    'text': 'Daily Census Trend',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'title': 'Date',
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'Census'},
                'height': 400
            }
        }
        
        # Contract percentage trend
        charts['contract_trend'] = {
            'data': [
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['RN_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN Contract %',
                    'line': {'color': '#1f77b4'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['LPN_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN Contract %',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['CNA_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'CNA Contract %',
                    'line': {'color': '#2ca02c'}
                },
                {
                    'x': filtered_df['WorkDate'].dt.strftime('%m-%d-%Y').tolist(),
                    'y': filtered_df['Total_Contract_Pct'].fillna(0).tolist(),
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total Contract %',
                    'line': {'color': '#d62728', 'width': 3}
                }
            ] + holiday_markers,
            'layout': {
                'title': {
                    'text': 'Contract Percentage Trends',
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'title': 'Date',
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'Contract Percentage (%)', 'range': [0, None]},
                'height': 400
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

@app.route('/api/quarters')
def get_quarters():
    """Get available quarters"""
    quarters = sorted(df['CY_Qtr'].unique().tolist())
    return jsonify(quarters)

@app.route('/api/date_range')
def get_date_range():
    """Get available date range"""
    return jsonify({
        'min_date': df['WorkDate'].min().strftime('%Y-%m-%d'),
        'max_date': df['WorkDate'].max().strftime('%Y-%m-%d')
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
    actual_quarters = set(df['CY_Qtr'].unique())
    missing_quarters = [q for q in expected_quarters if q not in actual_quarters]
    
    if missing_quarters:
        completeness_issues.append({
            'type': 'missing_quarter',
            'severity': 'high',
            'message': f"Missing {len(missing_quarters)} quarter(s): {', '.join(missing_quarters)}"
        })
    
    # Check for incomplete quarters (missing days)
    for quarter in actual_quarters:
        quarter_data = df[df['CY_Qtr'] == quarter]
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
    df_sorted = df.sort_values('WorkDate')
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
            quarter = df[df['WorkDate'] == gap['start_date']]['CY_Qtr'].iloc[0] if len(df[df['WorkDate'] == gap['start_date']]) > 0 else 'Unknown'
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
        target_day = df[df['WorkDate'] == target_date]
        
        if len(target_day) == 0:
            return jsonify({'error': 'No data found for target date'})
        
        target_record = target_day.iloc[0]
        
        # Get comparison data based on type
        if comparison_type == 'month':
            # Same month, different years
            comparison_data = df[
                (df['WorkDate'].dt.month == target_date.month) & 
                (df['WorkDate'] != target_date)
            ]
        elif comparison_type == 'quarter':
            # Same quarter, different years
            quarter = f"{target_date.year}Q{(target_date.month-1)//3 + 1}"
            comparison_data = df[
                (df['CY_Qtr'].str.contains(f"Q{(target_date.month-1)//3 + 1}")) & 
                (df['WorkDate'] != target_date)
            ]
        elif comparison_type == 'year':
            # Same year, different dates
            comparison_data = df[
                (df['WorkDate'].dt.year == target_date.year) & 
                (df['WorkDate'] != target_date)
            ]
        elif comparison_type == 'same_dow':
            # Same day of week
            comparison_data = df[
                (df['DayOfWeek'] == target_record['DayOfWeek']) & 
                (df['WorkDate'] != target_date)
            ]
        else:  # custom
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            if not start_date or not end_date:
                return jsonify({'error': 'start_date and end_date required for custom comparison'})
            
            comparison_data = df[
                (df['WorkDate'] >= start_date) & 
                (df['WorkDate'] <= end_date) & 
                (df['WorkDate'] != target_date)
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
        target_data = df[df['WorkDate'] == target_dt]
        if target_data.empty:
            return jsonify({'error': f'No data found for {target_date}'})
        
        target_row = target_data.iloc[0]
        target_quarter = target_row['CY_Qtr']
        target_year = target_dt.year
        target_dow = target_row['DayOfWeek']
        
        # Get comparison data
        quarter_data = df[df['CY_Qtr'] == target_quarter]
        year_data = df[df['WorkDate'].dt.year == target_year]
        dow_year_data = df[(df['WorkDate'].dt.year == target_year) & (df['DayOfWeek'] == target_dow)]
        
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
            'dow_year': calculate_comparison_metrics(dow_year_data, f"{target_dow}s in {target_year}")
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
            dow_aberration = calculate_aberrations(target_val, comparisons['dow_year'], metric, dow_year_data)
            
            aberrations[metric] = {
                'quarter': quarter_aberration,
                'year': year_aberration,
                'dow': dow_aberration
            }
        
        # Generate PBJ source links
        nurse_source_link = format_pbj_source_link(target_quarter, target_date, "225500", "nurse")
        nonnurse_source_link = format_pbj_source_link(target_quarter, target_date, "225500", "nonnurse")
        
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
            filters.append(f"Quarter: {format_quarter(quarters[0])}")
        else:
            # Create a range for multiple quarters
            formatted_quarters = [format_quarter(q) for q in quarters]
            if len(formatted_quarters) > 3:
                # Show range for many quarters
                first_quarter = formatted_quarters[0]
                last_quarter = formatted_quarters[-1]
                filters.append(f"Quarters: {first_quarter} - {last_quarter}")
            else:
                # Show all quarters if 3 or fewer
                filters.append(f"Quarters: {', '.join(formatted_quarters)}")
    
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
        # Get quarter parameter
        quarter = request.args.get('quarter', 'all')
        
        # Filter data by quarter if specified
        filtered_df = df.copy()
        if quarter != 'all':
            filtered_df = filtered_df[filtered_df['CY_Qtr'] == quarter]
        
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
            zero_counts[pos_key] = len(df[df[col_name] == 0].groupby('CY_Qtr'))
        
        # Structure the response
        quarterly_stats = {}
        
        # Total Nurse Staff
        quarterly_stats['total_nurse_staff'] = {
            'total_hours': {
                'mean': float(quarterly_data['Total_Staff_Hours_mean'].mean()),
                'median': float(quarterly_data['Total_Staff_Hours_median'].mean()),
                'std_dev': float(quarterly_data['Total_Staff_Hours_std'].mean())
            },
            'hprd': {
                'mean': float(quarterly_data['Total_Staff_HPRD_mean'].mean()),
                'median': float(quarterly_data['Total_Staff_HPRD_median'].mean()),
                'std_dev': float(quarterly_data['Total_Staff_HPRD_std'].mean())
            },
            'zero_count': zero_counts['total_staff']
        }
        
        # Direct Staff (excl. Admin, DON)
        quarterly_stats['direct_staff_excl_admin'] = {
            'total_hours': {
                'mean': float(quarterly_data['Nurse_Staff_Hours_Excl_Admin_mean'].mean()),
                'median': float(quarterly_data['Nurse_Staff_Hours_Excl_Admin_median'].mean()),
                'std_dev': float(quarterly_data['Nurse_Staff_Hours_Excl_Admin_std'].mean())
            },
            'hprd': {
                'mean': float(quarterly_data['Nurse_Staff_HPRD_Excl_Admin_mean'].mean()),
                'median': float(quarterly_data['Nurse_Staff_HPRD_Excl_Admin_median'].mean()),
                'std_dev': float(quarterly_data['Nurse_Staff_HPRD_Excl_Admin_std'].mean())
            },
            'zero_count': 0  # Calculate if needed
        }
        
        # Total RN
        quarterly_stats['total_rn'] = {
            'total_hours': {
                'mean': float(quarterly_data['Total_RN_Hours_mean'].mean()),
                'median': float(quarterly_data['Total_RN_Hours_median'].mean()),
                'std_dev': float(quarterly_data['Total_RN_Hours_std'].mean())
            },
            'hprd': {
                'mean': float(quarterly_data['Total_RN_HPRD_mean'].mean()),
                'median': float(quarterly_data['Total_RN_HPRD_median'].mean()),
                'std_dev': float(quarterly_data['Total_RN_HPRD_std'].mean())
            },
            'zero_count': zero_counts['total_rn']
        }
        
        # Direct RN
        quarterly_stats['rn_direct'] = {
            'total_hours': {
                'mean': float(quarterly_data['Hrs_RN_mean'].mean()),
                'median': float(quarterly_data['Hrs_RN_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_RN_std'].mean())
            },
            'hprd': {
                'mean': float(quarterly_data['RN_HPRD_mean'].mean()),
                'median': float(quarterly_data['RN_HPRD_median'].mean()),
                'std_dev': float(quarterly_data['RN_HPRD_std'].mean())
            },
            'zero_count': zero_counts['rn_direct']
        }
        
        # RN Admin
        quarterly_stats['rn_admin'] = {
            'total_hours': {
                'mean': float(quarterly_data['Hrs_RNadmin_mean'].mean()),
                'median': float(quarterly_data['Hrs_RNadmin_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_RNadmin_std'].mean())
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
                'mean': float(quarterly_data['Hrs_RNDON_mean'].mean()),
                'median': float(quarterly_data['Hrs_RNDON_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_RNDON_std'].mean())
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
                'mean': float(quarterly_data['Total_LPN_Hours_mean'].mean()),
                'median': float(quarterly_data['Total_LPN_Hours_median'].mean()),
                'std_dev': float(quarterly_data['Total_LPN_Hours_std'].mean())
            },
            'hprd': {
                'mean': float(quarterly_data['Total_LPN_HPRD_mean'].mean()),
                'median': float(quarterly_data['Total_LPN_HPRD_median'].mean()),
                'std_dev': float(quarterly_data['Total_LPN_HPRD_std'].mean())
            },
            'zero_count': zero_counts['total_lpn']
        }
        
        # LPN
        quarterly_stats['lpn_direct'] = {
            'total_hours': {
                'mean': float(quarterly_data['Hrs_LPN_mean'].mean()),
                'median': float(quarterly_data['Hrs_LPN_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_LPN_std'].mean())
            },
            'hprd': {
                'mean': float(quarterly_data['LPN_HPRD_mean'].mean()),
                'median': float(quarterly_data['LPN_HPRD_median'].mean()),
                'std_dev': float(quarterly_data['LPN_HPRD_std'].mean())
            },
            'zero_count': zero_counts['lpn_direct']
        }
        
        # LPN Admin
        quarterly_stats['lpn_admin'] = {
            'total_hours': {
                'mean': float(quarterly_data['Hrs_LPNadmin_mean'].mean()),
                'median': float(quarterly_data['Hrs_LPNadmin_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_LPNadmin_std'].mean())
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
                'mean': float(quarterly_data['Total_Nurse_Aide_Hours_mean'].mean()),
                'median': float(quarterly_data['Total_Nurse_Aide_Hours_median'].mean()),
                'std_dev': float(quarterly_data['Total_Nurse_Aide_Hours_std'].mean())
            },
            'hprd': {
                'mean': float(quarterly_data['Total_Nurse_Aide_HPRD_mean'].mean()),
                'median': float(quarterly_data['Total_Nurse_Aide_HPRD_median'].mean()),
                'std_dev': float(quarterly_data['Total_Nurse_Aide_HPRD_std'].mean())
            },
            'zero_count': zero_counts['total_cna']
        }
        
        # CNA
        quarterly_stats['cna_direct'] = {
            'total_hours': {
                'mean': float(quarterly_data['Hrs_CNA_mean'].mean()),
                'median': float(quarterly_data['Hrs_CNA_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_CNA_std'].mean())
            },
            'hprd': {
                'mean': float(quarterly_data['CNA_HPRD_mean'].mean()),
                'median': float(quarterly_data['CNA_HPRD_median'].mean()),
                'std_dev': float(quarterly_data['CNA_HPRD_std'].mean())
            },
            'zero_count': zero_counts['cna_direct']
        }
        
        # Med Aide
        quarterly_stats['med_aide'] = {
            'total_hours': {
                'mean': float(quarterly_data['Hrs_MedAide_mean'].mean()),
                'median': float(quarterly_data['Hrs_MedAide_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_MedAide_std'].mean())
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
                'mean': float(quarterly_data['Hrs_NAtrn_mean'].mean()),
                'median': float(quarterly_data['Hrs_NAtrn_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_NAtrn_std'].mean())
            },
            'hprd': {
                'mean': 0.0,  # Trainee HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['na_trainee']
        }
        
        # Total Contract
        total_contract_hours = quarterly_data['Hrs_RN_ctr_mean'] + quarterly_data['Hrs_LPN_ctr_mean'] + quarterly_data['Hrs_CNA_ctr_mean']
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
        direct_contract_hours = quarterly_data['Hrs_RN_ctr_mean'] + quarterly_data['Hrs_LPN_ctr_mean'] + quarterly_data['Hrs_CNA_ctr_mean']
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
                'mean': float(quarterly_data['Hrs_RN_ctr_mean'].mean()),
                'median': float(quarterly_data['Hrs_RN_ctr_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_RN_ctr_std'].mean())
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
                'mean': float(quarterly_data['Hrs_RN_ctr_mean'].mean()),
                'median': float(quarterly_data['Hrs_RN_ctr_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_RN_ctr_std'].mean())
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
                'mean': float(quarterly_data['Hrs_CNA_ctr_mean'].mean()),
                'median': float(quarterly_data['Hrs_CNA_ctr_median'].mean()),
                'std_dev': float(quarterly_data['Hrs_CNA_ctr_std'].mean())
            },
            'hprd': {
                'mean': 0.0,  # Contract HPRD not typically calculated
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': zero_counts['cna_contract']
        }
        
        return jsonify({'quarterly_stats': quarterly_stats})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/quarterly-data')
def get_quarterly_data():
    """Get quarterly data for all quarters with HPRD and hours"""
    try:
        # Group data by quarter and calculate quarterly averages (hours per day)
        quarterly_data = df.groupby('CY_Qtr').agg({
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

# Initialize data when module is imported
initialize_data()

if __name__ == '__main__':
    if df is None:
        print("ERROR: Failed to load data. Exiting.")
        exit(1)
    print("Starting Facility 225500 Dashboard...")
    app.run(debug=True, host='0.0.0.0', port=5003)
