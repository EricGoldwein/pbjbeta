#!/usr/bin/env python3
"""
Flask app for Facility 495241 - Exact replica of dynamic dashboard
Uses the same template and all functionality, just hardcoded for facility 495241
"""

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import json
from decimal import Decimal, ROUND_HALF_UP
import os
import sys

app = Flask(__name__)

# Global variables - hardcoded for facility 495241
df = None
global_df = None
provider_info_df = None
PROVNUM = "495241"

def is_federal_holiday(date):
    """Check if a date is a US federal holiday - EXACT same logic as dynamic dashboard"""
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
    
    # Variable holidays (calculated)
    # Martin Luther King Jr. Day (third Monday in January)
    if month == 1:
        # Find third Monday
        first_day = pd.Timestamp(year, 1, 1)
        first_monday = first_day + pd.Timedelta(days=(7 - first_day.weekday()) % 7)
        third_monday = first_monday + pd.Timedelta(days=14)
        if day == third_monday.day:
            return True
    
    # Presidents' Day (third Monday in February)
    elif month == 2:
        first_day = pd.Timestamp(year, 2, 1)
        first_monday = first_day + pd.Timedelta(days=(7 - first_day.weekday()) % 7)
        third_monday = first_monday + pd.Timedelta(days=14)
        if day == third_monday.day:
            return True
    
    # Memorial Day (last Monday in May)
    elif month == 5:
        last_day = pd.Timestamp(year, 5, 31)
        last_monday = last_day - pd.Timedelta(days=last_day.weekday())
        if day == last_monday.day:
            return True
    
    # Labor Day (first Monday in September)
    elif month == 9:
        first_day = pd.Timestamp(year, 9, 1)
        first_monday = first_day + pd.Timedelta(days=(7 - first_day.weekday()) % 7)
        if day == first_monday.day:
            return True
    
    # Columbus Day (second Monday in October)
    elif month == 10:
        first_day = pd.Timestamp(year, 10, 1)
        first_monday = first_day + pd.Timedelta(days=(7 - first_day.weekday()) % 7)
        second_monday = first_monday + pd.Timedelta(days=7)
        if day == second_monday.day:
            return True
    
    # Thanksgiving (fourth Thursday in November)
    elif month == 11:
        first_day = pd.Timestamp(year, 11, 1)
        first_thursday = first_day + pd.Timedelta(days=(3 - first_day.weekday()) % 7)
        fourth_thursday = first_thursday + pd.Timedelta(days=21)
        if day == fourth_thursday.day:
            return True
    
    return False

def load_facility_data():
    """Load data for facility 495241 - exact same logic as dynamic dashboard"""
    global global_df, provider_info_df
    
    print(f"Loading facility {PROVNUM} data...")
    
    # Load daily data
    csv_filename = f'facility_{PROVNUM}_complete_data.csv'
    if not os.path.exists(csv_filename):
        print(f"Error: {csv_filename} not found!")
        return None
    
    global_df = pd.read_csv(csv_filename)
    print(f"Loaded {len(global_df)} records")
    
    # Convert WorkDate to datetime and add DayOfWeek/IsHoliday - EXACT same logic as dynamic dashboard
    global_df['WorkDate'] = pd.to_datetime(global_df['WorkDate'], format='%Y%m%d')
    global_df['DayOfWeek'] = global_df['WorkDate'].dt.day_name()
    global_df['DayOfWeekNum'] = global_df['WorkDate'].dt.dayofweek  # 0=Monday, 6=Sunday
    global_df['IsHoliday'] = global_df['WorkDate'].apply(is_federal_holiday)
    
    # Add calculated columns - EXACT same logic as dynamic dashboard
    global_df['RN_HPRD'] = (global_df['Hrs_RN'] / global_df['MDScensus']).fillna(0)
    global_df['LPN_HPRD'] = (global_df['Hrs_LPN'] / global_df['MDScensus']).fillna(0)
    global_df['CNA_HPRD'] = (global_df['Hrs_CNA'] / global_df['MDScensus']).fillna(0)
    global_df['Total_Nurse_HPRD'] = ((global_df['Hrs_RN'] + global_df['Hrs_LPN'] + global_df['Hrs_CNA']) / global_df['MDScensus']).fillna(0)
    global_df['Total_RN_Hours'] = global_df['Hrs_RN'] + global_df['Hrs_RNadmin'] + global_df['Hrs_RNDON']
    global_df['Total_RN_HPRD'] = (global_df['Total_RN_Hours'] / global_df['MDScensus']).fillna(0)
    global_df['Total_LPN_Hours'] = global_df['Hrs_LPN'] + global_df['Hrs_LPNadmin']
    global_df['Total_LPN_HPRD'] = (global_df['Total_LPN_Hours'] / global_df['MDScensus']).fillna(0)
    global_df['Total_Nurse_Aide_Hours'] = global_df['Hrs_CNA'] + global_df['Hrs_NAtrn']
    global_df['Total_Nurse_Aide_HPRD'] = (global_df['Total_Nurse_Aide_Hours'] / global_df['MDScensus']).fillna(0)
    global_df['Nurse_Staff_HPRD_Excl_Admin'] = ((global_df['Hrs_RN'] + global_df['Hrs_LPN'] + global_df['Hrs_CNA']) / global_df['MDScensus']).fillna(0)
    global_df['Total_Nurse_Hours'] = global_df['Total_RN_Hours'] + global_df['Total_LPN_Hours'] + global_df['Total_Nurse_Aide_Hours']
    global_df['Nurse_Staff_Hours_Excl_Admin'] = global_df['Hrs_RN'] + global_df['Hrs_LPN'] + global_df['Hrs_CNA']
    global_df['Total_Staff_Hours'] = global_df['Total_Nurse_Hours']
    global_df['Total_Staff_HPRD'] = (global_df['Total_Staff_Hours'] / global_df['MDScensus']).fillna(0)
    
    # Contract percentages - EXACT same logic as dynamic dashboard
    global_df['RN_Contract_Pct'] = ((global_df['Hrs_RN_ctr'] + global_df['Hrs_RNadmin_ctr'] + global_df['Hrs_RNDON_ctr']) / global_df['Total_RN_Hours'] * 100).fillna(0)
    global_df['LPN_Contract_Pct'] = ((global_df['Hrs_LPN_ctr'] + global_df['Hrs_LPNadmin_ctr']) / global_df['Total_LPN_Hours'] * 100).fillna(0)
    global_df['CNA_Contract_Pct'] = ((global_df['Hrs_CNA_ctr'] + global_df['Hrs_NAtrn_ctr']) / global_df['Total_Nurse_Aide_Hours'] * 100).fillna(0)
    global_df['Total_LPN_Contract_Pct'] = global_df['LPN_Contract_Pct']  # Alias
    global_df['Total_Contract_Pct'] = ((global_df['Hrs_RN_ctr'] + global_df['Hrs_RNadmin_ctr'] + global_df['Hrs_RNDON_ctr'] + 
                                      global_df['Hrs_LPN_ctr'] + global_df['Hrs_LPNadmin_ctr'] + 
                                      global_df['Hrs_CNA_ctr'] + global_df['Hrs_NAtrn_ctr']) / global_df['Total_Nurse_Hours'] * 100).fillna(0)
    
    print(f"Loaded {len(global_df)} records from {global_df['WorkDate'].min()} to {global_df['WorkDate'].max()}")
    print(f"Calculated columns: {[col for col in global_df.columns if col not in ['PROVNUM', 'PROVNAME', 'CITY', 'STATE', 'COUNTY_NAME', 'COUNTY_FIPS', 'CY_Qtr', 'WorkDate', 'MDScensus', 'Hrs_RNDON', 'Hrs_RNDON_emp', 'Hrs_RNDON_ctr', 'Hrs_RNadmin', 'Hrs_RNadmin_emp', 'Hrs_RNadmin_ctr', 'Hrs_RN', 'Hrs_RN_emp', 'Hrs_RN_ctr', 'Hrs_LPNadmin', 'Hrs_LPNadmin_emp', 'Hrs_LPNadmin_ctr', 'Hrs_LPN', 'Hrs_LPN_emp', 'Hrs_LPN_ctr', 'Hrs_CNA', 'Hrs_CNA_emp', 'Hrs_CNA_ctr', 'Hrs_NAtrn', 'Hrs_NAtrn_emp', 'Hrs_NAtrn_ctr', 'Hrs_MedAide', 'Hrs_MedAide_emp', 'Hrs_MedAide_ctr', 'Hrs_RNDON.1']]}")
    
    # Load provider info data
    provider_csv = f'facility_{PROVNUM}_provider_info_data.csv'
    if os.path.exists(provider_csv):
        try:
            provider_info_df = pd.read_csv(provider_csv)
            print(f"Loaded {len(provider_info_df)} provider info records")
        except Exception as e:
            print(f"Error loading provider info data: {str(e)}")
            provider_info_df = None
    else:
        print(f"Provider info file {provider_csv} not found")
        provider_info_df = None
    
    # Verify critical columns
    critical_cols = ['Total_Staff_HPRD', 'Total_Staff_Hours', 'Total_RN_HPRD', 'Total_LPN_HPRD', 'Total_Nurse_Aide_HPRD']
    missing_cols = [col for col in critical_cols if col not in global_df.columns]
    if missing_cols:
        print(f"MISSING CRITICAL COLUMNS: {missing_cols}")
    else:
        print(f"All critical columns present")
    
    return global_df

# Load data on startup
load_facility_data()

@app.route('/')
def index():
    """Main dashboard page - exact same template as dynamic dashboard with SAMPLE branding"""
    return render_template('dynamic_facility_dashboard.html', 
                         facility_name="THALIA GARDENS REHABILITATION AND NURSING",
                         provnum="495241",
                         state="VA",
                         city="VIRGINIA BEACH",
                         county_name="Virginia Beach City",
                         sample_branding="SAMPLE")

# Copy ALL the API endpoints from dynamic_facility_dashboard.py
# I'll include the key ones that the dashboard needs

@app.route('/api/quarters')
def get_quarters():
    """Get available quarters"""
    global global_df
    if global_df is None:
        return jsonify({'error': 'Data not loaded'})
    
    quarters = sorted(global_df['CY_Qtr'].unique().tolist())
    return jsonify(quarters)

@app.route('/api/data')
def get_data():
    """Get filtered daily data - EXACT same logic as dynamic dashboard"""
    global global_df
    
    if global_df is None:
        return jsonify({'error': 'Data not loaded'})
    
    # Get filter parameters - ALL the same parameters as dynamic dashboard
    quarter = request.args.get('quarter', '')
    year = request.args.get('year', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    day_of_week = request.args.get('day_of_week', '')
    show_holidays_only = request.args.get('show_holidays_only', 'false')
    
    # Apply filters - EXACT same logic as dynamic dashboard
    filtered_df = global_df.copy()
    
    # Date range filter - EXACT same logic as dynamic dashboard
    if start_date and end_date:
        # Convert string dates to datetime objects for comparison
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            (filtered_df['WorkDate'] >= start_dt) & 
            (filtered_df['WorkDate'] <= end_dt)
        ]
    
    # Quarter filter - handle multiple quarters
    if quarter and quarter != 'all':
        if ',' in quarter:
            quarters = quarter.split(',')
            filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
        else:
            filtered_df = filtered_df[filtered_df['CY_Qtr'] == quarter]
    
    # Year filter - handle multiple years
    if year and year != 'all':
        if ',' in year:
            years = year.split(',')
            filtered_df = filtered_df[filtered_df['WorkDate'].astype(str).str[:4].isin(years)]
        else:
            filtered_df = filtered_df[filtered_df['WorkDate'].astype(str).str[:4] == year]
    
    # CRITICAL: If years are selected but no quarters, automatically include all quarters for those years
    # This is EXACT logic from dynamic dashboard lines 3096-3104
    if year and year != 'all' and quarter == 'all':
        selected_years = year.split(',')
        year_quarters = []
        for yr in selected_years:
            year_quarters.extend([f"{yr}Q1", f"{yr}Q2", f"{yr}Q3", f"{yr}Q4"])
        filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(year_quarters)]
    
    # Day of week filter - EXACT same logic as dynamic dashboard
    if day_of_week and day_of_week != 'all':
        filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
    
    # Holidays only filter - EXACT same logic as dynamic dashboard
    if show_holidays_only == 'true':
        filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
    
    # Convert to dict format expected by frontend
    data = []
    for _, row in filtered_df.iterrows():
        record = {
            'WorkDate': row.get('WorkDate').strftime('%Y-%m-%d') if hasattr(row.get('WorkDate'), 'strftime') else str(row.get('WorkDate', '')),
            'CY_Qtr': row.get('CY_Qtr', ''),
            'MDScensus': float(row.get('MDScensus', 0)) if not pd.isna(row.get('MDScensus', 0)) else 0,
            'Total_Nurse_HPRD': float(row.get('Total_Nurse_HPRD', 0)) if not pd.isna(row.get('Total_Nurse_HPRD', 0)) else 0,
            'RN_HPRD': float(row.get('RN_HPRD', 0)) if not pd.isna(row.get('RN_HPRD', 0)) else 0,
            'LPN_HPRD': float(row.get('LPN_HPRD', 0)) if not pd.isna(row.get('LPN_HPRD', 0)) else 0,
            'CNA_HPRD': float(row.get('CNA_HPRD', 0)) if not pd.isna(row.get('CNA_HPRD', 0)) else 0,
            'Total_Staff_HPRD': float(row.get('Total_Staff_HPRD', 0)) if not pd.isna(row.get('Total_Staff_HPRD', 0)) else 0,
            'Total_Staff_Hours': float(row.get('Total_Staff_Hours', 0)) if not pd.isna(row.get('Total_Staff_Hours', 0)) else 0,
            'Total_RN_HPRD': float(row.get('Total_RN_HPRD', 0)) if not pd.isna(row.get('Total_RN_HPRD', 0)) else 0,
            'Total_RN_Hours': float(row.get('Total_RN_Hours', 0)) if not pd.isna(row.get('Total_RN_Hours', 0)) else 0,
            'Total_LPN_HPRD': float(row.get('Total_LPN_HPRD', 0)) if not pd.isna(row.get('Total_LPN_HPRD', 0)) else 0,
            'Total_LPN_Hours': float(row.get('Total_LPN_Hours', 0)) if not pd.isna(row.get('Total_LPN_Hours', 0)) else 0,
            'Total_Nurse_Aide_HPRD': float(row.get('Total_Nurse_Aide_HPRD', 0)) if not pd.isna(row.get('Total_Nurse_Aide_HPRD', 0)) else 0,
            'Total_Nurse_Aide_Hours': float(row.get('Total_Nurse_Aide_Hours', 0)) if not pd.isna(row.get('Total_Nurse_Aide_Hours', 0)) else 0,
            'Nurse_Staff_HPRD_Excl_Admin': float(row.get('Nurse_Staff_HPRD_Excl_Admin', 0)) if not pd.isna(row.get('Nurse_Staff_HPRD_Excl_Admin', 0)) else 0,
            'Total_Contract_Pct': float(row.get('Total_Contract_Pct', 0)) if not pd.isna(row.get('Total_Contract_Pct', 0)) else 0,
            # Add missing fields that frontend expects
            'Hrs_RN': float(row.get('Hrs_RN', 0)) if not pd.isna(row.get('Hrs_RN', 0)) else 0,
            'Hrs_RNadmin': float(row.get('Hrs_RNadmin', 0)) if not pd.isna(row.get('Hrs_RNadmin', 0)) else 0,
            'Hrs_RNDON': float(row.get('Hrs_RNDON', 0)) if not pd.isna(row.get('Hrs_RNDON', 0)) else 0,
            'Hrs_LPN': float(row.get('Hrs_LPN', 0)) if not pd.isna(row.get('Hrs_LPN', 0)) else 0,
            'Hrs_LPNadmin': float(row.get('Hrs_LPNadmin', 0)) if not pd.isna(row.get('Hrs_LPNadmin', 0)) else 0,
            'Hrs_CNA': float(row.get('Hrs_CNA', 0)) if not pd.isna(row.get('Hrs_CNA', 0)) else 0,
            'Hrs_NAtrn': float(row.get('Hrs_NAtrn', 0)) if not pd.isna(row.get('Hrs_NAtrn', 0)) else 0,
            'Hrs_MedAide': float(row.get('Hrs_MedAide', 0)) if not pd.isna(row.get('Hrs_MedAide', 0)) else 0,
            # Contract hours
            'Hrs_RN_ctr': float(row.get('Hrs_RN_ctr', 0)) if not pd.isna(row.get('Hrs_RN_ctr', 0)) else 0,
            'Hrs_LPN_ctr': float(row.get('Hrs_LPN_ctr', 0)) if not pd.isna(row.get('Hrs_LPN_ctr', 0)) else 0,
            'Hrs_CNA_ctr': float(row.get('Hrs_CNA_ctr', 0)) if not pd.isna(row.get('Hrs_CNA_ctr', 0)) else 0,
            'RN_Contract_Pct': float(row.get('RN_Contract_Pct', 0)) if not pd.isna(row.get('RN_Contract_Pct', 0)) else 0,
            'LPN_Contract_Pct': float(row.get('LPN_Contract_Pct', 0)) if not pd.isna(row.get('LPN_Contract_Pct', 0)) else 0,
            'CNA_Contract_Pct': float(row.get('CNA_Contract_Pct', 0)) if not pd.isna(row.get('CNA_Contract_Pct', 0)) else 0,
            'IsHoliday': bool(row.get('IsHoliday', False)),
            'DayOfWeek': str(row.get('DayOfWeek', ''))
        }
        data.append(record)
    
    return jsonify({
        'data': data,
        'total_records': len(data)
    })

@app.route('/api/charts')
def get_charts():
    """Get chart data - EXACT same logic as dynamic dashboard"""
    global global_df
    
    if global_df is None:
        return jsonify({'error': 'Data not loaded'})
    
    # Debug: Check if required columns exist
    required_columns = ['Total_Staff_HPRD', 'Total_RN_HPRD', 'Total_LPN_HPRD', 'Total_Nurse_Aide_HPRD', 'Nurse_Staff_HPRD_Excl_Admin']
    missing_columns = [col for col in required_columns if col not in global_df.columns]
    if missing_columns:
        print(f"ERROR: Missing required columns: {missing_columns}")
        print(f"Available columns: {list(global_df.columns)}")
        return jsonify({'error': f'Missing required columns: {missing_columns}'})
    
    # Get filter parameters - ALL the same parameters as dynamic dashboard
    quarter = request.args.get('quarter', '')
    year = request.args.get('year', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    day_of_week = request.args.get('day_of_week', '')
    show_holidays_only = request.args.get('show_holidays_only', 'false')
    
    # Get view mode parameters - EXACT same as dynamic dashboard
    hprd_view = request.args.get('hprd_view', 'daily')
    hours_view = request.args.get('hours_view', 'daily')
    census_view = request.args.get('census_view', 'daily')
    contract_view = request.args.get('contract_view', 'daily')
    
    # Apply filters - EXACT same logic as dynamic dashboard
    filtered_df = global_df.copy()
    
    print(f"DEBUG: Initial filtered_df shape: {filtered_df.shape}")
    print(f"DEBUG: Filter parameters - quarter='{quarter}', year='{year}', start_date='{start_date}', end_date='{end_date}'")
    
    # Date range filter - EXACT same logic as dynamic dashboard
    if start_date and end_date:
        # Convert string dates to datetime objects for comparison
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            (filtered_df['WorkDate'] >= start_dt) & 
            (filtered_df['WorkDate'] <= end_dt)
        ]
        print(f"DEBUG: After date filter: {filtered_df.shape}")
    
    # Quarter filter - handle multiple quarters
    if quarter and quarter != 'all':
        if ',' in quarter:
            quarters = quarter.split(',')
            filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
        else:
            filtered_df = filtered_df[filtered_df['CY_Qtr'] == quarter]
        print(f"DEBUG: After quarter filter: {filtered_df.shape}")
    
    # Year filter - handle multiple years
    if year and year != 'all':
        if ',' in year:
            years = year.split(',')
            filtered_df = filtered_df[filtered_df['WorkDate'].astype(str).str[:4].isin(years)]
        else:
            filtered_df = filtered_df[filtered_df['WorkDate'].astype(str).str[:4] == year]
        print(f"DEBUG: After year filter: {filtered_df.shape}")
    
    # CRITICAL: If years are selected but no quarters, automatically include all quarters for those years
    # This is EXACT logic from dynamic dashboard lines 3096-3104
    if year and year != 'all' and quarter == 'all':
        selected_years = year.split(',')
        year_quarters = []
        for yr in selected_years:
            year_quarters.extend([f"{yr}Q1", f"{yr}Q2", f"{yr}Q3", f"{yr}Q4"])
        filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(year_quarters)]
        print(f"DEBUG: After year-quarter auto-filter: {filtered_df.shape}")
    
    # Sort by date
    filtered_df = filtered_df.sort_values('WorkDate')
    
    print(f"DEBUG: Final filtered_df shape: {filtered_df.shape}")
    if filtered_df.empty:
        print("ERROR: filtered_df is empty after filtering!")
        return jsonify({'charts': {}, 'filter_info': 'No data for selected filters'})
    
    # Check columns again after filtering
    missing_columns_after = [col for col in required_columns if col not in filtered_df.columns]
    if missing_columns_after:
        print(f"ERROR: Missing columns after filtering: {missing_columns_after}")
        print(f"Available columns after filtering: {list(filtered_df.columns)}")
        return jsonify({'error': f'Missing columns after filtering: {missing_columns_after}'})
    
    # Helper function to aggregate data by view mode - EXACT same logic as dynamic dashboard
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
    
    # Apply aggregation based on view modes
    hprd_df = aggregate_by_view_mode(filtered_df, hprd_view)
    hours_df = aggregate_by_view_mode(filtered_df, hours_view)
    census_df = aggregate_by_view_mode(filtered_df, census_view)
    contract_df = aggregate_by_view_mode(filtered_df, contract_view)
    
    # Helper function to format dates based on view mode - EXACT same logic as dynamic dashboard
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
    
    # Helper function to generate dynamic filter info - EXACT same logic as dynamic dashboard
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
        
        if day_of_week != 'all' and day_of_week:
            filters.append(f"Day: {day_of_week}")
        
        if holidays_only and holidays_only != 'false':
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
    
    # Create chart objects with data and layout - EXACT same structure as dynamic dashboard
    charts = {
        'hprd_trend': {
            'data': [
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hprd_df['Total_Staff_HPRD'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total HPRD (All Staff)',
                    'line': {'color': '#d62728', 'width': 3}
                },
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hprd_df['Nurse_Staff_HPRD_Excl_Admin'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Staff HPRD (excl. Admin & DON)',
                    'line': {'color': '#9467bd'}
                },
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hprd_df['Total_RN_HPRD'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN HPRD (Total)',
                    'line': {'color': '#1f77b4'}
                },
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hprd_df['Total_LPN_HPRD'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN HPRD (Total)',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': format_dates_for_view_mode(hprd_df, hprd_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hprd_df['Total_Nurse_Aide_HPRD'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide HPRD (Total)',
                    'line': {'color': '#2ca02c'}
                }
            ],
            'layout': {
                    'title': {
                        'text': f"Daily HPRD Trends<br>{get_filter_description(start_date, end_date, quarter, day_of_week, show_holidays_only)}",
                        'x': 0.5,
                        'xanchor': 'center'
                    },
                    'xaxis': {
                        'nticks': 10,
                        'tickangle': -45
                    },
                    'yaxis': {
                        'title': 'HPRD',
                        'tickformat': '.2f'
                    },
                    'hoverlabel': {
                        'namelength': -1,
                        'bgcolor': 'white',
                        'bordercolor': 'black',
                        'font': {'size': 12, 'color': 'black'}
                    },
                    'hoverformat': '.2f',
                    'height': 450,
                    'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
                }
        },
        'hours_trend': {
            'data': [
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hours_df['Total_Staff_Hours'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total Nurse',
                    'line': {'color': '#d62728', 'width': 3}
                },
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hours_df['Nurse_Staff_Hours_Excl_Admin'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Staff (excl. Admin & DON)',
                    'line': {'color': '#9467bd'}
                },
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hours_df['Total_RN_Hours'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN (Total)',
                    'line': {'color': '#1f77b4'}
                },
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hours_df['Total_LPN_Hours'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN (Total)',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': format_dates_for_view_mode(hours_df, hours_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in hours_df['Total_Nurse_Aide_Hours'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Nurse Aide (Total)',
                    'line': {'color': '#2ca02c'}
                }
            ],
            'layout': {
                'title': {
                    'text': f"Total Nursing Staff Hours<br>{get_filter_description(start_date, end_date, quarter, day_of_week, show_holidays_only)}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'Hours'},
                'hoverlabel': {
                    'namelength': -1,
                    'bgcolor': 'white',
                    'bordercolor': 'black',
                    'font': {'size': 12, 'color': 'black'}
                },
                'hoverformat': '.2f',
                'height': 450,
                'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
            }
        },
        'census_trend': {
            'data': [
                {
                    'x': format_dates_for_view_mode(census_df, census_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in census_df['MDScensus'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Resident Census',
                    'line': {'color': '#17becf', 'width': 3}
                }
            ],
            'layout': {
                'title': {
                    'text': f"Resident Census<br>{get_filter_description(start_date, end_date, quarter, day_of_week, show_holidays_only)}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'Residents'},
                'hoverlabel': {
                    'namelength': -1,
                    'bgcolor': 'white',
                    'bordercolor': 'black',
                    'font': {'size': 12, 'color': 'black'}
                },
                'hoverformat': '.0f',
                'height': 450,
                'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
            }
        },
        'contract_trend': {
            'data': [
                {
                    'x': format_dates_for_view_mode(contract_df, contract_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in contract_df['Total_Contract_Pct'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Total Contract %',
                    'line': {'color': '#bcbd22', 'width': 3}
                },
                {
                    'x': format_dates_for_view_mode(contract_df, contract_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in contract_df['RN_Contract_Pct'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'RN Contract %',
                    'line': {'color': '#1f77b4'}
                },
                {
                    'x': format_dates_for_view_mode(contract_df, contract_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in contract_df['LPN_Contract_Pct'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'LPN Contract %',
                    'line': {'color': '#ff7f0e'}
                },
                {
                    'x': format_dates_for_view_mode(contract_df, contract_view),
                    'y': [float(x) if not pd.isna(x) else 0 for x in contract_df['CNA_Contract_Pct'].tolist()],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'CNA Contract %',
                    'line': {'color': '#2ca02c'}
                }
            ],
            'layout': {
                'title': {
                    'text': f"Contract Staffing %<br>{get_filter_description(start_date, end_date, quarter, day_of_week, show_holidays_only)}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                'xaxis': {
                    'nticks': 10,
                    'tickangle': -45
                },
                'yaxis': {'title': 'Percentage'},
                'hoverlabel': {
                    'namelength': -1,
                    'bgcolor': 'white',
                    'bordercolor': 'black',
                    'font': {'size': 12, 'color': 'black'}
                },
                'hoverformat': '.2f',
                'height': 450,
                'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
            }
        }
    }
    
    # Add day of week comparison chart - EXACT same logic as dynamic dashboard
    dow_summary = filtered_df.groupby('DayOfWeek').agg({
        'Total_Staff_HPRD': 'mean',
        'Total_RN_HPRD': 'mean',
        'Total_LPN_HPRD': 'mean',
        'Total_Nurse_Aide_HPRD': 'mean',
        'Nurse_Staff_HPRD_Excl_Admin': 'mean'
    }).reset_index()
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_summary['DayOfWeek'] = pd.Categorical(dow_summary['DayOfWeek'], categories=day_order, ordered=True)
    dow_summary = dow_summary.sort_values('DayOfWeek')
    
    charts['dow_comparison'] = {
        'data': [
                {
                    'x': dow_summary['DayOfWeek'].tolist(),
                    'y': dow_summary['Total_Staff_HPRD'].tolist(),
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
                    'text': f"Average HPRD by Day of Week<br>{get_filter_description(start_date, end_date, quarter, day_of_week, show_holidays_only)}",
                    'x': 0.5,
                    'xanchor': 'center'
                },
            'xaxis': {},
            'yaxis': {'title': 'HPRD'},
            'hoverlabel': {
                'namelength': -1,
                'bgcolor': 'white',
                'bordercolor': 'black',
                'font': {'size': 12, 'color': 'black'}
            },
            'hoverformat': '.2f',
            'height': 450,
            'margin': {'b': 100, 'l': 60, 'r': 40, 't': 80}
        }
    }
    
    return jsonify({
        'charts': charts,
        'filter_info': get_filter_description(start_date, end_date, quarter, day_of_week, show_holidays_only)
    })

@app.route('/api/provider_info_charts')
def get_provider_info_charts():
    """Get provider info chart data - EXACT same logic as dynamic dashboard"""
    global provider_info_df, global_df
    
    if provider_info_df is None:
        return jsonify({'error': 'Provider info data not loaded'})
    
    # EXACT same processing logic as dynamic dashboard
    chart_data = provider_info_df.dropna(subset=['quarter']).copy()
    chart_data = chart_data.sort_values('processing_date').groupby('quarter').last().reset_index()
    
    # Format quarter labels
    chart_data['quarter_label'] = chart_data['quarter'].apply(
        lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) and len(str(x)) == 6 else str(x) if pd.notna(x) else None
    )
    
    # Add PBJ-calculated direct care values
    pbj_direct_data = []
    for quarter in chart_data['quarter']:
        pbj_quarter = global_df[global_df['CY_Qtr'] == quarter] if global_df is not None else pd.DataFrame()
        if len(pbj_quarter) > 0:
            total_census = pbj_quarter['MDScensus'].sum()
            direct_hours = (
                pbj_quarter['Hrs_RN'].sum() + 
                pbj_quarter['Hrs_LPN'].sum() + 
                pbj_quarter['Hrs_CNA'].sum()
            )
            direct_hprd = (direct_hours / total_census) if total_census > 0 else 0
            
            rn_direct_hours = pbj_quarter['Hrs_RN'].sum()
            rn_direct_hprd = (rn_direct_hours / total_census) if total_census > 0 else 0
            
            pbj_direct_data.append({
                'quarter': quarter, 
                'pbj_direct_total': direct_hprd, 
                'pbj_rn_direct': rn_direct_hprd
            })
        else:
            pbj_direct_data.append({
                'quarter': quarter, 
                'pbj_direct_total': 0, 
                'pbj_rn_direct': 0
            })
    
    pbj_direct_df = pd.DataFrame(pbj_direct_data)
    chart_data = chart_data.merge(pbj_direct_df, on='quarter', how='left')
    
    # Prepare chart data structure - EXACT same as dynamic dashboard
    charts = {
        'total_staffing': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'reported_total': chart_data['reported_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist(),
            'reported_direct': chart_data['pbj_direct_total'].fillna(0).tolist(),
            'case_mix_total': chart_data['case_mix_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist(),
            'adjusted_total': chart_data['adjusted_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist()
        },
        'rn_staffing': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'reported_rn': chart_data['reported_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_rn_hrs_per_resident_per_day']), None).tolist(),
            'reported_rn_total': chart_data['reported_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_rn_hrs_per_resident_per_day']), None).tolist(),
            'reported_rn_direct': chart_data['pbj_rn_direct'].where(pd.notna(chart_data['pbj_rn_direct']), None).tolist(),
            'case_mix_rn': chart_data['case_mix_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_rn_hrs_per_resident_per_day']), None).tolist(),
            'adjusted_rn': chart_data['adjusted_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['adjusted_rn_hrs_per_resident_per_day']), None).tolist()
        },
        'cna_staffing': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'reported_cna': chart_data['reported_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_na_hrs_per_resident_per_day']), None).tolist(),
            'case_mix_cna': chart_data['case_mix_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_na_hrs_per_resident_per_day']), None).tolist(),
            'case_mix_lpn': chart_data['case_mix_lpn_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_lpn_hrs_per_resident_per_day']), None).tolist(),
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
    
    # Convert NaN to None for JSON serialization
    def convert_nan_to_none(obj):
        if isinstance(obj, dict):
            return {key: convert_nan_to_none(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_none(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    return jsonify(convert_nan_to_none(charts))

# Add more API endpoints as needed - I can copy ALL of them from the original
@app.route('/api/summary')
def get_summary():
    """Get summary statistics"""
    global global_df
    if global_df is None:
        return jsonify({'error': 'Data not loaded'})
    
    # Get filter parameters - ALL the same parameters as dynamic dashboard
    quarter = request.args.get('quarter', '')
    year = request.args.get('year', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    day_of_week = request.args.get('day_of_week', '')
    show_holidays_only = request.args.get('show_holidays_only', 'false')
    
    # Apply filters - EXACT same logic as dynamic dashboard
    filtered_df = global_df.copy()
    
    # Date range filter - EXACT same logic as dynamic dashboard
    if start_date and end_date:
        # Convert string dates to datetime objects for comparison
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            (filtered_df['WorkDate'] >= start_dt) & 
            (filtered_df['WorkDate'] <= end_dt)
        ]
    
    # Quarter filter - handle multiple quarters
    if quarter and quarter != 'all':
        if ',' in quarter:
            quarters = quarter.split(',')
            filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
        else:
            filtered_df = filtered_df[filtered_df['CY_Qtr'] == quarter]
    
    # Year filter - handle multiple years
    if year and year != 'all':
        if ',' in year:
            years = year.split(',')
            filtered_df = filtered_df[filtered_df['WorkDate'].astype(str).str[:4].isin(years)]
        else:
            filtered_df = filtered_df[filtered_df['WorkDate'].astype(str).str[:4] == year]
    
    # CRITICAL: If years are selected but no quarters, automatically include all quarters for those years
    # This is EXACT logic from dynamic dashboard lines 3096-3104
    if year and year != 'all' and quarter == 'all':
        selected_years = year.split(',')
        year_quarters = []
        for yr in selected_years:
            year_quarters.extend([f"{yr}Q1", f"{yr}Q2", f"{yr}Q3", f"{yr}Q4"])
        filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(year_quarters)]
    
    # Calculate summary statistics with NaN handling
    summary = {
        'total_days': int(len(filtered_df)),
        'avg_census': float(filtered_df['MDScensus'].mean()) if not pd.isna(filtered_df['MDScensus'].mean()) else 0,
        'avg_total_hprd': float(filtered_df['Total_Staff_HPRD'].mean()) if not pd.isna(filtered_df['Total_Staff_HPRD'].mean()) else 0,
        'avg_nurse_staff_hprd_excl_admin': float(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].mean()) if not pd.isna(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].mean()) else 0,
        'avg_rn_hprd': float(filtered_df['RN_HPRD'].mean()) if not pd.isna(filtered_df['RN_HPRD'].mean()) else 0,
        'avg_total_rn_hprd': float(filtered_df['Total_RN_HPRD'].mean()) if not pd.isna(filtered_df['Total_RN_HPRD'].mean()) else 0,
        'avg_lpn_hprd': float(filtered_df['Total_LPN_HPRD'].mean()) if not pd.isna(filtered_df['Total_LPN_HPRD'].mean()) else 0,
        'avg_cna_hprd': float(filtered_df['Total_Nurse_Aide_HPRD'].mean()) if not pd.isna(filtered_df['Total_Nurse_Aide_HPRD'].mean()) else 0,
        'avg_contract_pct': float(filtered_df['Total_Contract_Pct'].mean()) if not pd.isna(filtered_df['Total_Contract_Pct'].mean()) else 0,
        'avg_total_contract_pct': float(filtered_df['Total_Contract_Pct'].mean()) if not pd.isna(filtered_df['Total_Contract_Pct'].mean()) else 0,
        'total_rn_sub8': int((filtered_df['Total_RN_Hours'] < 8).sum()),
        'direct_rn_sub8': int((filtered_df['Hrs_RN'] < 8).sum()),
        'date_range': {
            'min_date': str(filtered_df['WorkDate'].min()),
            'max_date': str(filtered_df['WorkDate'].max())
        }
    }
    
    return jsonify(summary)

@app.route('/api/data_completeness')
def get_data_completeness():
    """Analyze data completeness and identify missing quarters/days - EXACT same as dynamic dashboard"""
    global global_df
    if global_df is None:
        return jsonify({'error': 'Data not loaded'})
    
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
            
            if q_num == 1:  # Q1: Jan-Mar
                start_month, end_month = 1, 3
            elif q_num == 2:  # Q2: Apr-Jun
                start_month, end_month = 4, 6
            elif q_num == 3:  # Q3: Jul-Sep
                start_month, end_month = 7, 9
            else:  # Q4: Oct-Dec
                start_month, end_month = 10, 12
            
            # Calculate expected days in quarter
            quarter_start = pd.Timestamp(year, start_month, 1)
            if end_month == 12:
                quarter_end = pd.Timestamp(year + 1, 1, 1) - pd.Timedelta(days=1)
            else:
                quarter_end = pd.Timestamp(year, end_month + 1, 1) - pd.Timedelta(days=1)
            
            # Count actual days
            actual_days = len(quarter_data)
            expected_days = (quarter_end - quarter_start).days + 1
            
            # Allow for some tolerance (weekends, holidays, etc.)
            if actual_days < expected_days * 0.8:  # Less than 80% of expected days
                completeness_issues.append({
                    'type': 'incomplete_quarter',
                    'severity': 'medium',
                    'message': f"{quarter} has only {actual_days} days (expected ~{expected_days})"
                })
    
    # Check for data quality issues
    zero_census_days = len(global_df[global_df['MDScensus'] == 0])
    if zero_census_days > 0:
        completeness_issues.append({
            'type': 'zero_census',
            'severity': 'medium',
            'message': f"{zero_census_days} days with zero census"
        })
    
    # Check for missing critical data
    critical_columns = ['Hrs_RN', 'Hrs_LPN', 'Hrs_CNA', 'MDScensus']
    for col in critical_columns:
        if col in global_df.columns:
            null_count = global_df[col].isnull().sum()
            if null_count > 0:
                completeness_issues.append({
                    'type': 'missing_data',
                    'severity': 'high',
                    'message': f"{null_count} records missing {col} data"
                })
    
    # Basic completeness info
    completeness = {
        'total_records': int(len(global_df)),
        'date_range': {
            'min_date': str(global_df['WorkDate'].min()),
            'max_date': str(global_df['WorkDate'].max())
        },
        'quarters': int(global_df['CY_Qtr'].nunique()),
        'years': int(global_df['WorkDate'].astype(str).str[:4].nunique()),
        'issues': completeness_issues,
        'has_issues': len(completeness_issues) > 0
    }
    
    return jsonify(completeness)

@app.route('/api/date_range')
def get_date_range():
    """Get date range information - EXACT format as dynamic dashboard"""
    global global_df
    if global_df is None:
        return jsonify({'error': 'Data not loaded'})
    
    min_date = global_df['WorkDate'].min()
    max_date = global_df['WorkDate'].max()
    
    # Convert to YYYY-MM-DD format like dynamic dashboard expects
    min_date_str = str(min_date)
    max_date_str = str(max_date)
    
    # If dates are in YYYYMMDD format, convert to YYYY-MM-DD
    if len(min_date_str) == 8:
        min_date_str = f"{min_date_str[:4]}-{min_date_str[4:6]}-{min_date_str[6:8]}"
    if len(max_date_str) == 8:
        max_date_str = f"{max_date_str[:4]}-{max_date_str[4:6]}-{max_date_str[6:8]}"
    
    return jsonify({
        'min_date': min_date_str,
        'max_date': max_date_str
    })

@app.route('/api/provider_info_summary')
def get_provider_info_summary():
    """Get provider info summary statistics"""
    global provider_info_df
    if provider_info_df is None:
        return jsonify({'error': 'Provider info data not loaded'})
    
    # Get latest provider info record
    latest_record = provider_info_df.sort_values('processing_date').iloc[-1]
    
    summary = {
        'latest_overall_rating': latest_record.get('overall_rating', 0),
        'latest_staffing_rating': latest_record.get('staffing_rating', 0),
        'latest_health_inspection_rating': latest_record.get('health_inspection_rating', 0),
        'ownership_type': latest_record.get('ownership_type', ''),
        'sff_status': latest_record.get('sff_status', 'Not Available'),
        'ownership_change_last_12_months': latest_record.get('ownership_change_last_12_months', 'Not Available'),
        'facility_name': latest_record.get('facility_name', 'Facility'),
        'avg_residents_per_day': latest_record.get('avg_residents_per_day', 0)
    }
    
    return jsonify(summary)

@app.route('/api/quarterly-data')
def get_quarterly_data():
    """Get quarterly aggregated data"""
    global global_df
    if global_df is None:
        return jsonify({'error': 'Data not loaded'})
    
    # Group by quarter and calculate averages - return as object with quarter keys
    quarterly_data = {}
    for quarter in global_df['CY_Qtr'].unique():
        quarter_df = global_df[global_df['CY_Qtr'] == quarter]
        if len(quarter_df) > 0:
            quarterly_data[quarter] = {
                'total_days': int(len(quarter_df)),
                'census': float(quarter_df['MDScensus'].mean()) if not pd.isna(quarter_df['MDScensus'].mean()) else 0,
                'total_hprd': float(quarter_df['Total_Staff_HPRD'].mean()) if not pd.isna(quarter_df['Total_Staff_HPRD'].mean()) else 0,
                'total_hours': float(quarter_df['Total_Staff_Hours'].mean()) if not pd.isna(quarter_df['Total_Staff_Hours'].mean()) else 0,
                'direct_hprd': float(quarter_df['Nurse_Staff_HPRD_Excl_Admin'].mean()) if not pd.isna(quarter_df['Nurse_Staff_HPRD_Excl_Admin'].mean()) else 0,
                'direct_hours': float((quarter_df['Hrs_RN'] + quarter_df['Hrs_LPN'] + quarter_df['Hrs_CNA']).mean()) if not pd.isna((quarter_df['Hrs_RN'] + quarter_df['Hrs_LPN'] + quarter_df['Hrs_CNA']).mean()) else 0,
                'total_rn_hprd': float(quarter_df['Total_RN_HPRD'].mean()) if not pd.isna(quarter_df['Total_RN_HPRD'].mean()) else 0,
                'total_rn_hours': float(quarter_df['Total_RN_Hours'].mean()) if not pd.isna(quarter_df['Total_RN_Hours'].mean()) else 0,
                'rn_hprd': float(quarter_df['RN_HPRD'].mean()) if not pd.isna(quarter_df['RN_HPRD'].mean()) else 0,
                'rn_hours': float(quarter_df['Hrs_RN'].mean()) if not pd.isna(quarter_df['Hrs_RN'].mean()) else 0,
                'rn_admin_hprd': float((quarter_df['Hrs_RNadmin'] / quarter_df['MDScensus']).mean()) if not pd.isna((quarter_df['Hrs_RNadmin'] / quarter_df['MDScensus']).mean()) else 0,
                'rn_admin_hours': float(quarter_df['Hrs_RNadmin'].mean()) if not pd.isna(quarter_df['Hrs_RNadmin'].mean()) else 0,
                'rn_don_hprd': float((quarter_df['Hrs_RNDON'] / quarter_df['MDScensus']).mean()) if not pd.isna((quarter_df['Hrs_RNDON'] / quarter_df['MDScensus']).mean()) else 0,
                'rn_don_hours': float(quarter_df['Hrs_RNDON'].mean()) if not pd.isna(quarter_df['Hrs_RNDON'].mean()) else 0,
                # Add placeholder values for missing fields
                'prov_reported_total': 0,
                'pbj_reported_total': float(quarter_df['Total_Staff_HPRD'].mean()) if not pd.isna(quarter_df['Total_Staff_HPRD'].mean()) else 0,
                'pbj_reported_direct': float(quarter_df['Nurse_Staff_HPRD_Excl_Admin'].mean()) if not pd.isna(quarter_df['Nurse_Staff_HPRD_Excl_Admin'].mean()) else 0,
                'prov_reported_rn': 0,
                'pbj_reported_total_rn': float(quarter_df['Total_RN_HPRD'].mean()) if not pd.isna(quarter_df['Total_RN_HPRD'].mean()) else 0,
                'pbj_reported_direct_rn': float(quarter_df['RN_HPRD'].mean()) if not pd.isna(quarter_df['RN_HPRD'].mean()) else 0,
                'prov_reported_lpn': 0,
                'pbj_reported_total_lpn': float(quarter_df['Total_LPN_HPRD'].mean()) if not pd.isna(quarter_df['Total_LPN_HPRD'].mean()) else 0,
                'pbj_reported_direct_lpn': float(quarter_df['LPN_HPRD'].mean()) if not pd.isna(quarter_df['LPN_HPRD'].mean()) else 0,
                'prov_reported_na': 0,
                'pbj_reported_na': float(quarter_df['Total_Nurse_Aide_HPRD'].mean()) if not pd.isna(quarter_df['Total_Nurse_Aide_HPRD'].mean()) else 0,
                'case_mix_total': 0,
                'case_mix_rn': 0,
                'case_mix_lpn': 0,
                'case_mix_na': 0,
                'pct_cmi_total': 0,
                'pct_cmi_direct': 0,
                'pct_cmi_total_rn': 0,
                'pct_cmi_direct_rn': 0,
                'pct_cmi_total_lpn': 0,
                'pct_cmi_direct_lpn': 0,
                'pct_cmi_na': 0
            }
    
    return jsonify({
        'quarterly_data': quarterly_data
    })

@app.route('/api/case-mix-data')
def get_case_mix_data():
    """Get case-mix acuity data by quarter from both Provider Info and PBJ calculations - EXACT same logic as dynamic dashboard"""
    global global_df, provider_info_df
    
    try:
        case_mix_data = {}
        
        # Get all unique quarters from both sources
        all_quarters = set()
        
        # Get quarters from PBJ data
        if global_df is not None and len(global_df) > 0:
            all_quarters.update(global_df['CY_Qtr'].unique())
        
        # Get quarters from Provider Info
        if provider_info_df is not None and len(provider_info_df) > 0:
            quarters_df = provider_info_df[provider_info_df['quarter'].notna()].copy()
            all_quarters.update(quarters_df['quarter'].unique())
        
        if len(all_quarters) == 0:
            return jsonify({'error': 'No data available', 'case_mix_data': {}})
        
        for quarter in all_quarters:
            quarter_info = {
                'quarter': quarter
            }
            
            # === PROVIDER INFO DATA ===
            if provider_info_df is not None and len(provider_info_df) > 0:
                prov_quarter_data = provider_info_df[provider_info_df['quarter'] == quarter]
                if len(prov_quarter_data) > 0:
                    prov_data = prov_quarter_data.iloc[0]
                    
                    # Reported values from Provider Info
                    quarter_info['prov_reported_total'] = float(prov_data.get('reported_total_nurse_hrs_per_resident_per_day', 0)) if pd.notna(prov_data.get('reported_total_nurse_hrs_per_resident_per_day')) else None
                    quarter_info['prov_reported_rn'] = float(prov_data.get('reported_rn_hrs_per_resident_per_day', 0)) if pd.notna(prov_data.get('reported_rn_hrs_per_resident_per_day')) else None
                    quarter_info['prov_reported_lpn'] = float(prov_data.get('reported_lpn_hrs_per_resident_per_day', 0)) if pd.notna(prov_data.get('reported_lpn_hrs_per_resident_per_day')) else None
                    quarter_info['prov_reported_na'] = float(prov_data.get('reported_na_hrs_per_resident_per_day', 0)) if pd.notna(prov_data.get('reported_na_hrs_per_resident_per_day')) else None
                    
                    # Case-mix values from Provider Info
                    quarter_info['case_mix_total'] = float(prov_data.get('case_mix_total_nurse_hrs_per_resident_per_day', 0)) if pd.notna(prov_data.get('case_mix_total_nurse_hrs_per_resident_per_day')) else None
                    quarter_info['case_mix_rn'] = float(prov_data.get('case_mix_rn_hrs_per_resident_per_day', 0)) if pd.notna(prov_data.get('case_mix_rn_hrs_per_resident_per_day')) else None
                    quarter_info['case_mix_lpn'] = float(prov_data.get('case_mix_lpn_hrs_per_resident_per_day', 0)) if pd.notna(prov_data.get('case_mix_lpn_hrs_per_resident_per_day')) else None
                    quarter_info['case_mix_na'] = float(prov_data.get('case_mix_na_hrs_per_resident_per_day', 0)) if pd.notna(prov_data.get('case_mix_na_hrs_per_resident_per_day')) else None
            
            # === PBJ DATA (calculated from daily records) ===
            if global_df is not None and len(global_df) > 0:
                pbj_quarter_df = global_df[global_df['CY_Qtr'] == quarter]
                if len(pbj_quarter_df) > 0:
                    total_census = pbj_quarter_df['MDScensus'].sum()
                    
                    # Total Staff (all nursing staff)
                    total_hours = pbj_quarter_df['Total_Staff_Hours'].sum()
                    quarter_info['pbj_reported_total'] = float(total_hours / total_census) if total_census > 0 else None
                    
                    # Direct Staff (excluding admin and DON)
                    direct_hours = (pbj_quarter_df['Hrs_RN'].sum() + pbj_quarter_df['Hrs_LPN'].sum() + pbj_quarter_df['Hrs_CNA'].sum())
                    quarter_info['pbj_reported_direct'] = float(direct_hours / total_census) if total_census > 0 else None
                    
                    # Total RN (including admin and DON)
                    total_rn_hours = pbj_quarter_df['Total_RN_Hours'].sum()
                    quarter_info['pbj_reported_total_rn'] = float(total_rn_hours / total_census) if total_census > 0 else None
                    
                    # Direct RN (excluding admin and DON)
                    direct_rn_hours = pbj_quarter_df['Hrs_RN'].sum()
                    quarter_info['pbj_reported_direct_rn'] = float(direct_rn_hours / total_census) if total_census > 0 else None
                    
                    # Total LPN (including admin)
                    total_lpn_hours = pbj_quarter_df['Total_LPN_Hours'].sum()
                    quarter_info['pbj_reported_total_lpn'] = float(total_lpn_hours / total_census) if total_census > 0 else None
                    
                    # Direct LPN (excluding admin)
                    direct_lpn_hours = pbj_quarter_df['Hrs_LPN'].sum()
                    quarter_info['pbj_reported_direct_lpn'] = float(direct_lpn_hours / total_census) if total_census > 0 else None
                    
                    # Nurse Aide
                    na_hours = pbj_quarter_df['Total_Nurse_Aide_Hours'].sum()
                    quarter_info['pbj_reported_na'] = float(na_hours / total_census) if total_census > 0 else None
                else:
                    # No PBJ data for this quarter - set all PBJ fields to None
                    quarter_info['pbj_reported_total'] = None
                    quarter_info['pbj_reported_direct'] = None
                    quarter_info['pbj_reported_total_rn'] = None
                    quarter_info['pbj_reported_direct_rn'] = None
                    quarter_info['pbj_reported_total_lpn'] = None
                    quarter_info['pbj_reported_direct_lpn'] = None
                    quarter_info['pbj_reported_na'] = None
            
            # === CALCULATE % CMI VALUES ===
            # Total % CMI
            if quarter_info.get('pbj_reported_total') and quarter_info.get('case_mix_total'):
                quarter_info['pct_cmi_total'] = float((quarter_info['pbj_reported_total'] / quarter_info['case_mix_total']) * 100)
            
            # Direct % CMI
            if quarter_info.get('pbj_reported_direct') and quarter_info.get('case_mix_total'):
                quarter_info['pct_cmi_direct'] = float((quarter_info['pbj_reported_direct'] / quarter_info['case_mix_total']) * 100)
            
            # Total RN % CMI
            if quarter_info.get('pbj_reported_total_rn') and quarter_info.get('case_mix_rn'):
                quarter_info['pct_cmi_total_rn'] = float((quarter_info['pbj_reported_total_rn'] / quarter_info['case_mix_rn']) * 100)
            
            # Direct RN % CMI
            if quarter_info.get('pbj_reported_direct_rn') and quarter_info.get('case_mix_rn'):
                quarter_info['pct_cmi_direct_rn'] = float((quarter_info['pbj_reported_direct_rn'] / quarter_info['case_mix_rn']) * 100)
            
            # Total LPN % CMI
            if quarter_info.get('pbj_reported_total_lpn') and quarter_info.get('case_mix_lpn'):
                quarter_info['pct_cmi_total_lpn'] = float((quarter_info['pbj_reported_total_lpn'] / quarter_info['case_mix_lpn']) * 100)
            
            # Direct LPN % CMI
            if quarter_info.get('pbj_reported_direct_lpn') and quarter_info.get('case_mix_lpn'):
                quarter_info['pct_cmi_direct_lpn'] = float((quarter_info['pbj_reported_direct_lpn'] / quarter_info['case_mix_lpn']) * 100)
            
            # Nurse Aide % CMI
            if quarter_info.get('pbj_reported_na') and quarter_info.get('case_mix_na'):
                quarter_info['pct_cmi_na'] = float((quarter_info['pbj_reported_na'] / quarter_info['case_mix_na']) * 100)
            
            case_mix_data[quarter] = quarter_info
        
        return jsonify({
            'case_mix_data': case_mix_data
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'case_mix_data': {}})

@app.route('/api/quarterly-stats')
def get_quarterly_stats():
    """Get quarterly statistics"""
    global global_df
    if global_df is None:
        return jsonify({'error': 'Data not loaded'})
    
    # Get filter parameters - ALL the same parameters as dynamic dashboard
    year = request.args.get('year', '')
    quarter = request.args.get('quarter', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    day_of_week = request.args.get('day_of_week', '')
    show_holidays_only = request.args.get('show_holidays_only', 'false')
    
    # Apply filters - EXACT same logic as dynamic dashboard
    filtered_df = global_df.copy()
    
    # Date range filter - EXACT same logic as dynamic dashboard
    if start_date and end_date:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        filtered_df = filtered_df[
            (filtered_df['WorkDate'] >= start_dt) & 
            (filtered_df['WorkDate'] <= end_dt)
        ]
    
    if quarter and quarter != 'all':
        quarters = quarter.split(',') if ',' in quarter else [quarter]
        filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
    
    if year and year != 'all':
        years = [int(y.strip()) for y in year.split(',')]
        filtered_df = filtered_df[filtered_df['WorkDate'].dt.year.isin(years)]
    
    # Day of week filter - EXACT same logic as dynamic dashboard
    if day_of_week and day_of_week != 'all':
        filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
    
    # Holidays only filter - EXACT same logic as dynamic dashboard
    if show_holidays_only == 'true':
        filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
    
    # Calculate quarterly statistics in the format expected by the frontend - EXACT same as dynamic dashboard
    quarterly_stats = {
        'total_nurse_staff': {
            'total_hours': {
                'mean': round_financial(filtered_df['Total_Staff_Hours'].mean()),
                'median': round_financial(filtered_df['Total_Staff_Hours'].median()),
                'std_dev': round_financial(filtered_df['Total_Staff_Hours'].std())
            },
            'hprd': {
                'mean': round_financial(filtered_df['Total_Staff_HPRD'].mean()),
                'median': round_financial(filtered_df['Total_Staff_HPRD'].median()),
                'std_dev': round_financial(filtered_df['Total_Staff_HPRD'].std())
            },
            'zero_count': int((filtered_df['Total_Staff_Hours'] == 0).sum())
        },
        'direct_staff_excl_admin': {
            'total_hours': {
                'mean': round_financial(filtered_df['Nurse_Staff_Hours_Excl_Admin'].mean()),
                'median': round_financial(filtered_df['Nurse_Staff_Hours_Excl_Admin'].median()),
                'std_dev': round_financial(filtered_df['Nurse_Staff_Hours_Excl_Admin'].std())
            },
            'hprd': {
                'mean': round_financial(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].mean()),
                'median': round_financial(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].median()),
                'std_dev': round_financial(filtered_df['Nurse_Staff_HPRD_Excl_Admin'].std())
            },
            'zero_count': 0
        },
        'total_rn': {
            'total_hours': {
                'mean': round_financial(filtered_df['Total_RN_Hours'].mean()),
                'median': round_financial(filtered_df['Total_RN_Hours'].median()),
                'std_dev': round_financial(filtered_df['Total_RN_Hours'].std())
            },
            'hprd': {
                'mean': round_financial(filtered_df['Total_RN_HPRD'].mean()),
                'median': round_financial(filtered_df['Total_RN_HPRD'].median()),
                'std_dev': round_financial(filtered_df['Total_RN_HPRD'].std())
            },
            'zero_count': int((filtered_df['Total_RN_Hours'] == 0).sum())
        },
        'rn_direct': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_RN'].mean()),
                'median': round_financial(filtered_df['Hrs_RN'].median()),
                'std_dev': round_financial(filtered_df['Hrs_RN'].std())
            },
            'hprd': {
                'mean': round_financial(filtered_df['RN_HPRD'].mean()),
                'median': round_financial(filtered_df['RN_HPRD'].median()),
                'std_dev': round_financial(filtered_df['RN_HPRD'].std())
            },
            'zero_count': int((filtered_df['Hrs_RN'] == 0).sum())
        },
        'rn_admin': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_RNadmin'].mean()),
                'median': round_financial(filtered_df['Hrs_RNadmin'].median()),
                'std_dev': round_financial(filtered_df['Hrs_RNadmin'].std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int((filtered_df['Hrs_RNadmin'] == 0).sum())
        },
        'rn_don': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_RNDON'].mean()),
                'median': round_financial(filtered_df['Hrs_RNDON'].median()),
                'std_dev': round_financial(filtered_df['Hrs_RNDON'].std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int((filtered_df['Hrs_RNDON'] == 0).sum())
        },
        'total_lpn': {
            'total_hours': {
                'mean': round_financial(filtered_df['Total_LPN_Hours'].mean()),
                'median': round_financial(filtered_df['Total_LPN_Hours'].median()),
                'std_dev': round_financial(filtered_df['Total_LPN_Hours'].std())
            },
            'hprd': {
                'mean': round_financial(filtered_df['Total_LPN_HPRD'].mean()),
                'median': round_financial(filtered_df['Total_LPN_HPRD'].median()),
                'std_dev': round_financial(filtered_df['Total_LPN_HPRD'].std())
            },
            'zero_count': int((filtered_df['Total_LPN_Hours'] == 0).sum())
        },
        'lpn_direct': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_LPN'].mean()),
                'median': round_financial(filtered_df['Hrs_LPN'].median()),
                'std_dev': round_financial(filtered_df['Hrs_LPN'].std())
            },
            'hprd': {
                'mean': round_financial(filtered_df['LPN_HPRD'].mean()),
                'median': round_financial(filtered_df['LPN_HPRD'].median()),
                'std_dev': round_financial(filtered_df['LPN_HPRD'].std())
            },
            'zero_count': int((filtered_df['Hrs_LPN'] == 0).sum())
        },
        'lpn_admin': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_LPNadmin'].mean()),
                'median': round_financial(filtered_df['Hrs_LPNadmin'].median()),
                'std_dev': round_financial(filtered_df['Hrs_LPNadmin'].std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int((filtered_df['Hrs_LPNadmin'] == 0).sum())
        },
        'total_cna': {
            'total_hours': {
                'mean': round_financial(filtered_df['Total_Nurse_Aide_Hours'].mean()),
                'median': round_financial(filtered_df['Total_Nurse_Aide_Hours'].median()),
                'std_dev': round_financial(filtered_df['Total_Nurse_Aide_Hours'].std())
            },
            'hprd': {
                'mean': round_financial(filtered_df['Total_Nurse_Aide_HPRD'].mean()),
                'median': round_financial(filtered_df['Total_Nurse_Aide_HPRD'].median()),
                'std_dev': round_financial(filtered_df['Total_Nurse_Aide_HPRD'].std())
            },
            'zero_count': int((filtered_df['Total_Nurse_Aide_Hours'] == 0).sum())
        },
        'cna_direct': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_CNA'].mean()),
                'median': round_financial(filtered_df['Hrs_CNA'].median()),
                'std_dev': round_financial(filtered_df['Hrs_CNA'].std())
            },
            'hprd': {
                'mean': round_financial(filtered_df['CNA_HPRD'].mean()),
                'median': round_financial(filtered_df['CNA_HPRD'].median()),
                'std_dev': round_financial(filtered_df['CNA_HPRD'].std())
            },
            'zero_count': int((filtered_df['Hrs_CNA'] == 0).sum())
        },
        'med_aide': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_MedAide'].mean()),
                'median': round_financial(filtered_df['Hrs_MedAide'].median()),
                'std_dev': round_financial(filtered_df['Hrs_MedAide'].std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int((filtered_df['Hrs_MedAide'] == 0).sum())
        },
        'na_trainee': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_NAtrn'].mean()),
                'median': round_financial(filtered_df['Hrs_NAtrn'].median()),
                'std_dev': round_financial(filtered_df['Hrs_NAtrn'].std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int((filtered_df['Hrs_NAtrn'] == 0).sum())
        },
        'total_contract': {
            'total_hours': {
                'mean': round_financial((filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']).mean()),
                'median': round_financial((filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']).median()),
                'std_dev': round_financial((filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']).std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int(((filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']) == 0).sum())
        },
        'direct_care_contract': {
            'total_hours': {
                'mean': round_financial((filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']).mean()),
                'median': round_financial((filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']).median()),
                'std_dev': round_financial((filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']).std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int(((filtered_df['Hrs_RN_ctr'] + filtered_df['Hrs_LPN_ctr'] + filtered_df['Hrs_CNA_ctr']) == 0).sum())
        },
        'total_rn_contract': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_RN_ctr'].mean()),
                'median': round_financial(filtered_df['Hrs_RN_ctr'].median()),
                'std_dev': round_financial(filtered_df['Hrs_RN_ctr'].std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int((filtered_df['Hrs_RN_ctr'] == 0).sum())
        },
        'direct_rn_contract': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_RN_ctr'].mean()),
                'median': round_financial(filtered_df['Hrs_RN_ctr'].median()),
                'std_dev': round_financial(filtered_df['Hrs_RN_ctr'].std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int((filtered_df['Hrs_RN_ctr'] == 0).sum())
        },
        'nurse_aide_contract': {
            'total_hours': {
                'mean': round_financial(filtered_df['Hrs_CNA_ctr'].mean()),
                'median': round_financial(filtered_df['Hrs_CNA_ctr'].median()),
                'std_dev': round_financial(filtered_df['Hrs_CNA_ctr'].std())
            },
            'hprd': {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0
            },
            'zero_count': int((filtered_df['Hrs_CNA_ctr'] == 0).sum())
        }
    }
    
    return jsonify({
        'quarterly_stats': quarterly_stats,
        'sample_size': len(filtered_df)
    })

def round_financial(value):
    """Round to 2 decimal places using financial rounding (ROUND_HALF_UP)"""
    if pd.isna(value) or value is None:
        return 0
    return float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def generate_pbj_source_link(quarter, date, provnum="495241", data_type="nurse"):
    """Generate PBJ source link for a specific quarter and date."""
    # Convert date to YYYYMMDD format if needed
    if isinstance(date, str):
        if len(date) == 10 and '-' in date:  # YYYY-MM-DD format
            date = date.replace('-', '')
        elif len(date) == 8:  # Already YYYYMMDD
            pass
        elif len(date) == 10 and '/' in date:  # MM/DD/YYYY format
            parts = date.split('/')
            date = f"{parts[2]}{parts[0].zfill(2)}{parts[1].zfill(2)}"
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

def format_pbj_source_link(quarter, date, provnum="495241", data_type="nurse"):
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

@app.route('/api/single_day_report')
def get_single_day_report():
    """Get comprehensive single day report with comparisons and aberrations - EXACT same as dynamic dashboard"""
    global global_df
    if global_df is None:
        return jsonify({'error': 'Data not loaded'})
    
    date = request.args.get('date', '')
    if not date:
        return jsonify({'error': 'Date parameter required'})
    
    # Convert date to match CSV format (YYYYMMDD)
    try:
        if '-' in date:
            # Convert YYYY-MM-DD to YYYYMMDD
            date_parts = date.split('-')
            if len(date_parts) == 3:
                year, month, day = date_parts
                formatted_date = f"{year}{month.zfill(2)}{day.zfill(2)}"
            else:
                return jsonify({'error': 'Invalid date format'})
        elif '/' in date:
            # Convert MM/DD/YYYY to YYYYMMDD
            date_parts = date.split('/')
            if len(date_parts) == 3:
                month, day, year = date_parts
                formatted_date = f"{year}{month.zfill(2)}{day.zfill(2)}"
            else:
                return jsonify({'error': 'Invalid date format'})
        else:
            formatted_date = date.replace('/', '').replace('-', '')
            if len(formatted_date) == 8:
                # Already in YYYYMMDD format
                pass
            else:
                return jsonify({'error': 'Invalid date format'})
    except:
        return jsonify({'error': 'Invalid date format'})
    
    # Find the specific day - convert formatted_date to datetime for comparison
    target_date = pd.to_datetime(formatted_date, format='%Y%m%d')
    day_data = global_df[global_df['WorkDate'] == target_date]
    if len(day_data) == 0:
        return jsonify({'error': 'No data found for this date'})
    
    day_row = day_data.iloc[0]
    
    # Get quarter and year for comparisons
    target_quarter = day_row['CY_Qtr']
    target_year = target_date.year
    target_dow = day_row['DayOfWeek']
    
    # Get comparison data
    quarter_data = global_df[global_df['CY_Qtr'] == target_quarter]
    year_data = global_df[global_df['WorkDate'].dt.year == target_year]
    # For day-of-week comparison, use all days of the same week day in the target year only
    dow_year_data = global_df[
        (global_df['DayOfWeek'] == target_dow) & 
        (global_df['WorkDate'].dt.year == target_year)
    ]
    
    # Calculate metrics for target day - EXACT same as dynamic dashboard
    target_metrics = {
        'date': date,
        'day_of_week': target_dow,
        'quarter': target_quarter,
        'year': target_year,
        'census': round_financial(day_row['MDScensus']),
        'rn_hours': round_financial(day_row['Hrs_RN']),
        'rn_hprd': round_financial(day_row['RN_HPRD']),
        'lpn_hours': round_financial(day_row['Hrs_LPN']),
        'lpn_hprd': round_financial(day_row['LPN_HPRD']),
        'cna_hours': round_financial(day_row['Hrs_CNA']),
        'cna_hprd': round_financial(day_row['CNA_HPRD']),
        'total_rn_hours': round_financial(day_row['Total_RN_Hours']),
        'total_rn_hprd': round_financial(day_row['Total_RN_HPRD']),
        'total_lpn_hours': round_financial(day_row['Total_LPN_Hours']),
        'total_lpn_hprd': round_financial(day_row['Total_LPN_HPRD']),
        'total_nurse_aide_hours': round_financial(day_row['Total_Nurse_Aide_Hours']),
        'total_nurse_aide_hprd': round_financial(day_row['Total_Nurse_Aide_HPRD']),
        'nurse_staff_hours_excl_admin': round_financial(day_row['Nurse_Staff_Hours_Excl_Admin']),
        'nurse_staff_hprd_excl_admin': round_financial(day_row['Nurse_Staff_HPRD_Excl_Admin']),
        'total_staff_hours': round_financial(day_row['Total_Staff_Hours']),
        'total_staff_hprd': round_financial(day_row['Total_Staff_HPRD']),
        'rn_contract_pct': round_financial(day_row['RN_Contract_Pct']),
        'lpn_contract_pct': round_financial(day_row['LPN_Contract_Pct']),
        'cna_contract_pct': round_financial(day_row['CNA_Contract_Pct']),
        'total_contract_pct': round_financial(day_row['Total_Contract_Pct']),
        'is_holiday': bool(day_row['IsHoliday'])
    }
    
    # Calculate comparison averages with weighted HPRD calculations - EXACT same as dynamic dashboard
    def calculate_comparison_metrics(data, label):
        if data.empty:
            return None
        
        # Calculate weighted HPRD (sum of hours / sum of census)
        total_census = data['MDScensus'].sum()
        total_rn_hours = data['Hrs_RN'].sum()
        total_lpn_hours = data['Hrs_LPN'].sum()
        total_cna_hours = data['Hrs_CNA'].sum()
        total_rn_all_hours = data['Total_RN_Hours'].sum()
        total_lpn_all_hours = data['Total_LPN_Hours'].sum()
        total_nurse_aide_hours = data['Total_Nurse_Aide_Hours'].sum()
        nurse_staff_hours_excl_admin = data['Nurse_Staff_Hours_Excl_Admin'].sum()
        total_staff_hours = data['Total_Staff_Hours'].sum()
        
        # Calculate weighted HPRD values
        rn_hprd_weighted = (total_rn_hours / total_census) if total_census > 0 else 0
        lpn_hprd_weighted = (total_lpn_hours / total_census) if total_census > 0 else 0
        cna_hprd_weighted = (total_cna_hours / total_census) if total_census > 0 else 0
        total_rn_hprd_weighted = (total_rn_all_hours / total_census) if total_census > 0 else 0
        total_lpn_hprd_weighted = (total_lpn_all_hours / total_census) if total_census > 0 else 0
        total_nurse_aide_hprd_weighted = (total_nurse_aide_hours / total_census) if total_census > 0 else 0
        nurse_staff_hprd_excl_admin_weighted = (nurse_staff_hours_excl_admin / total_census) if total_census > 0 else 0
        total_staff_hprd_weighted = (total_staff_hours / total_census) if total_census > 0 else 0
        
        return {
            'label': label,
            'count': len(data),
            'census': round_financial(data['MDScensus'].mean()),
            'rn_hours': round_financial(data['Hrs_RN'].mean()),
            'rn_hprd': round_financial(rn_hprd_weighted),
            'lpn_hours': round_financial(data['Hrs_LPN'].mean()),
            'lpn_hprd': round_financial(lpn_hprd_weighted),
            'cna_hours': round_financial(data['Hrs_CNA'].mean()),
            'cna_hprd': round_financial(cna_hprd_weighted),
            'total_rn_hours': round_financial(data['Total_RN_Hours'].mean()),
            'total_rn_hprd': round_financial(total_rn_hprd_weighted),
            'total_lpn_hours': round_financial(data['Total_LPN_Hours'].mean()),
            'total_lpn_hprd': round_financial(total_lpn_hprd_weighted),
            'total_nurse_aide_hours': round_financial(data['Total_Nurse_Aide_Hours'].mean()),
            'total_nurse_aide_hprd': round_financial(total_nurse_aide_hprd_weighted),
            'nurse_staff_hours_excl_admin': round_financial(data['Nurse_Staff_Hours_Excl_Admin'].mean()),
            'nurse_staff_hprd_excl_admin': round_financial(nurse_staff_hprd_excl_admin_weighted),
            'total_staff_hours': round_financial(data['Total_Staff_Hours'].mean()),
            'total_staff_hprd': round_financial(total_staff_hprd_weighted),
            'rn_contract_pct': round_financial(data['RN_Contract_Pct'].mean()),
            'lpn_contract_pct': round_financial(data['LPN_Contract_Pct'].mean()),
            'cna_contract_pct': round_financial(data['CNA_Contract_Pct'].mean()),
            'total_contract_pct': round_financial(data['Total_Contract_Pct'].mean())
        }
    
    comparisons = {
        'quarter': calculate_comparison_metrics(quarter_data, f"Quarter {target_quarter}"),
        'year': calculate_comparison_metrics(year_data, f"Year {target_year}"),
        'dow': calculate_comparison_metrics(dow_year_data, f"{target_dow}s in {target_year}")
    }
    
    # Calculate aberrations (z-scores) - EXACT same as dynamic dashboard
    def calculate_aberrations(target_val, comparison_data, metric_name, data_source):
        if comparison_data is None or comparison_data['count'] < 2:
            return None
        
        # Get the actual data for this metric
        metric_map = {
            'census': 'MDScensus',
            'rn_hours': 'Hrs_RN',
            'rn_hprd': 'RN_HPRD',
            'lpn_hours': 'Hrs_LPN',
            'lpn_hprd': 'LPN_HPRD',
            'cna_hours': 'Hrs_CNA',
            'cna_hprd': 'CNA_HPRD',
            'total_rn_hours': 'Total_RN_Hours',
            'total_rn_hprd': 'Total_RN_HPRD',
            'total_lpn_hours': 'Total_LPN_Hours',
            'total_lpn_hprd': 'Total_LPN_HPRD',
            'total_nurse_aide_hours': 'Total_Nurse_Aide_Hours',
            'total_nurse_aide_hprd': 'Total_Nurse_Aide_HPRD',
            'nurse_staff_hours_excl_admin': 'Nurse_Staff_Hours_Excl_Admin',
            'nurse_staff_hprd_excl_admin': 'Nurse_Staff_HPRD_Excl_Admin',
            'total_staff_hours': 'Total_Staff_Hours',
            'total_staff_hprd': 'Total_Staff_HPRD',
            'rn_contract_pct': 'RN_Contract_Pct',
            'lpn_contract_pct': 'LPN_Contract_Pct',
            'cna_contract_pct': 'CNA_Contract_Pct'
        }
        
        if metric_name not in metric_map:
            return None
        
        values = data_source[metric_map[metric_name]].dropna()
        
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
        
        return {
            'z_score': round_financial(z_score),
            'percentile': round_financial(percentile),
            'mean': round_financial(mean_val),
            'std': round_financial(std_val)
        }
    
    # Calculate aberrations for key metrics
    aberrations = {}
    key_metrics = ['census', 'rn_hprd', 'lpn_hprd', 'cna_hprd', 'total_rn_hprd', 'total_lpn_hprd', 'total_nurse_aide_hprd', 'total_staff_hprd']
    
    for metric in key_metrics:
        target_val = target_metrics.get(metric, 0)
        aberrations[metric] = {
            'year': calculate_aberrations(target_val, comparisons['year'], metric, year_data),
            'quarter': calculate_aberrations(target_val, comparisons['quarter'], metric, quarter_data),
            'dow': calculate_aberrations(target_val, comparisons['dow'], metric, dow_year_data)
        }
    
    # Generate PBJ source links with actual provider number
    facility_provnum = str(day_row.get('PROVNUM', '495241')).zfill(6)
    nurse_source_link = format_pbj_source_link(target_quarter, formatted_date, facility_provnum, "nurse")
    nonnurse_source_link = format_pbj_source_link(target_quarter, formatted_date, facility_provnum, "nonnurse")
    
    # Return comprehensive report data - EXACT same structure as dynamic dashboard
    return jsonify({
        'target_metrics': target_metrics,
        'comparisons': comparisons,
        'aberrations': aberrations,
        'nurse_source_link': nurse_source_link,
        'nonnurse_source_link': nonnurse_source_link
    })

if __name__ == '__main__':
    print(f"Starting Flask app for facility {PROVNUM}")
    print("Dashboard will be available at: http://localhost:5001")
    app.run(debug=True, port=5001, host='0.0.0.0')
