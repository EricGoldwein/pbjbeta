#!/usr/bin/env python3
"""
Private Dashboard for Facility 335513
Deployable to Vercel with basic authentication
"""

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime, timedelta
import json
from decimal import Decimal, ROUND_HALF_UP
import os
import glob
import base64
from functools import wraps

app = Flask(__name__)

# Configuration
FACILITY_ID = "335513"
FACILITY_NAME = "Facility 335513"  # Update with actual facility name
REQUIRE_AUTH = False  # Set to False for no authentication

# Global variables
df = None
global_df = None

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        return f(*args, **kwargs)
    return decorated

def create_facility_complete_csv(provnum):
    """Load existing CSV for facility 335513"""
    print(f"Loading existing CSV for facility {provnum}...")
    
    # Try different possible file locations
    csv_paths = [
        'facility_335513_complete_data.csv',
        'facility_pbj/facility_335513_complete_data.csv',
        f'facility_{provnum}_complete_data.csv'
    ]
    
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            print(f"Found CSV at: {csv_path}")
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                df['WorkDate'] = pd.to_datetime(df['WorkDate'])
                df = df.sort_values('WorkDate')
                df = add_calculated_fields(df)
                print(f"✅ Loaded {len(df)} records for facility {provnum}")
                return df
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                continue
    
    print(f"❌ No CSV found for facility {provnum}")
    return None

def add_calculated_fields(df):
    """Add calculated HPRD and contract percentage fields"""
    
    # Ensure census is numeric and handle missing values
    df['MDScensus'] = pd.to_numeric(df['MDScensus'], errors='coerce').fillna(0)
    
    # HPRD Calculations (Hours Per Resident Day)
    def calculate_hprd(hours_col, census_col):
        return np.where(census_col > 0, hours_col / census_col, 0)
    
    # Total RN Hours (including admin and DON)
    df['Total_RN_Hours'] = (df.get('Hrs_RN', 0) + 
                           df.get('Hrs_RNadmin', 0) + 
                           df.get('Hrs_RNDON', 0)).fillna(0)
    
    # Total LPN Hours (including admin)
    df['Total_LPN_Hours'] = (df.get('Hrs_LPN', 0) + 
                            df.get('Hrs_LPNadmin', 0)).fillna(0)
    
    # Total Nurse Aide Hours
    df['Total_Nurse_Aide_Hours'] = (df.get('Hrs_CNA', 0) + 
                                   df.get('Hrs_NAtrn', 0) + 
                                   df.get('Hrs_MedAide', 0)).fillna(0)
    
    # Total Staff Hours
    df['Total_Staff_Hours'] = (df['Total_RN_Hours'] + 
                              df['Total_LPN_Hours'] + 
                              df['Total_Nurse_Aide_Hours'])
    
    # Nurse Staff Hours (excluding admin and DON)
    df['Nurse_Staff_Hours_Excl_Admin'] = (df.get('Hrs_RN', 0) + 
                                         df.get('Hrs_LPN', 0) + 
                                         df.get('Hrs_CNA', 0)).fillna(0)
    
    # HPRD Calculations
    df['Total_RN_HPRD'] = calculate_hprd(df['Total_RN_Hours'], df['MDScensus'])
    df['Total_LPN_HPRD'] = calculate_hprd(df['Total_LPN_Hours'], df['MDScensus'])
    df['Total_Nurse_Aide_HPRD'] = calculate_hprd(df['Total_Nurse_Aide_Hours'], df['MDScensus'])
    df['Total_Staff_HPRD'] = calculate_hprd(df['Total_Staff_Hours'], df['MDScensus'])
    df['Nurse_Staff_HPRD_Excl_Admin'] = calculate_hprd(df['Nurse_Staff_Hours_Excl_Admin'], df['MDScensus'])
    
    # Individual HPRD calculations
    df['RN_HPRD'] = calculate_hprd(df.get('Hrs_RN', 0), df['MDScensus'])
    df['LPN_HPRD'] = calculate_hprd(df.get('Hrs_LPN', 0), df['MDScensus'])
    df['CNA_HPRD'] = calculate_hprd(df.get('Hrs_CNA', 0), df['MDScensus'])
    df['RN_Admin_HPRD'] = calculate_hprd(df.get('Hrs_RNadmin', 0), df['MDScensus'])
    df['RN_DON_HPRD'] = calculate_hprd(df.get('Hrs_RNDON', 0), df['MDScensus'])
    
    # Contract Percentage Calculations
    def calculate_contract_percentage(contract_hours, total_hours):
        return np.where(total_hours > 0, (contract_hours / total_hours) * 100, 0)
    
    # RN Contract Percentage
    df['RN_Contract_Pct'] = calculate_contract_percentage(
        df.get('Hrs_RN_ctr', 0), 
        df.get('Hrs_RN', 0)
    )
    
    # LPN Contract Percentage
    df['LPN_Contract_Pct'] = calculate_contract_percentage(
        df.get('Hrs_LPN_ctr', 0), 
        df.get('Hrs_LPN', 0)
    )
    
    # CNA Contract Percentage
    df['CNA_Contract_Pct'] = calculate_contract_percentage(
        df.get('Hrs_CNA_ctr', 0), 
        df.get('Hrs_CNA', 0)
    )
    
    # Total Contract Percentage (based on direct care hours only)
    total_contract_hours = (df.get('Hrs_RN_ctr', 0) + 
                           df.get('Hrs_LPN_ctr', 0) + 
                           df.get('Hrs_CNA_ctr', 0))
    total_direct_care_hours = (df.get('Hrs_RN', 0) + 
                              df.get('Hrs_LPN', 0) + 
                              df.get('Hrs_CNA', 0))
    df['Total_Contract_Pct'] = calculate_contract_percentage(
        total_contract_hours, 
        total_direct_care_hours
    )
    
    # Add day of week and other date fields
    df['DayOfWeek'] = df['WorkDate'].dt.day_name()
    df['Year'] = df['WorkDate'].dt.year
    df['Month'] = df['WorkDate'].dt.month
    df['Quarter'] = df['WorkDate'].dt.quarter
    
    # Create quarter label
    df['CY_Qtr'] = df['Year'].astype(str) + 'Q' + df['Quarter'].astype(str)
    
    # Add holiday flag (you can expand this with actual holiday dates)
    df['IsHoliday'] = False
    
    return df

def round_financial(value, decimals=2):
    """Round using financial rounding (round half up)"""
    if pd.isna(value):
        return 0
    multiplier = 10 ** decimals
    return int(Decimal(str(value)) * Decimal(str(multiplier)) + Decimal('0.5')) / multiplier

@app.route('/')
@requires_auth
def index():
    """Main dashboard page"""
    return render_template('facility_335513_dashboard.html', 
                         provnum=FACILITY_ID, 
                         facility_name=FACILITY_NAME)

@app.route('/api/data')
@requires_auth
def get_data():
    """Get all data for the facility"""
    try:
        if global_df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Convert to JSON-serializable format
        data = global_df.copy()
        
        # Convert datetime to string
        data['WorkDate'] = data['WorkDate'].dt.strftime('%Y-%m-%d')
        
        # Round numeric values
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col] = data[col].apply(lambda x: round_financial(x, 3) if pd.notna(x) else 0)
        
        return jsonify({
            'data': data.to_dict('records'),
            'facility_id': FACILITY_ID,
            'facility_name': FACILITY_NAME
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/summary')
@requires_auth
def get_summary():
    """Get summary statistics"""
    try:
        if global_df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        # Apply any filters from query parameters
        filtered_df = global_df.copy()
        
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        quarter = request.args.get('quarter', 'all')
        day_of_week = request.args.get('day_of_week', 'all')
        show_holidays_only = request.args.get('show_holidays_only', 'false').lower() == 'true'
        
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
        }
        
        return jsonify(summary)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/single_day_report')
@requires_auth
def get_single_day_report():
    """Get single day report data"""
    try:
        if global_df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        target_date = request.args.get('date')
        if not target_date:
            return jsonify({'error': 'Date parameter required'}), 400
        
        # Find the target record
        target_record = global_df[global_df['WorkDate'].dt.strftime('%Y-%m-%d') == target_date]
        
        if len(target_record) == 0:
            return jsonify({'error': f'No data found for {target_date}'}), 404
        
        target_record = target_record.iloc[0]
        
        # Get comparison data (you can expand this with more sophisticated logic)
        target_quarter = target_record['CY_Qtr']
        target_year = target_record['Year']
        target_dow = target_record['DayOfWeek']
        
        # Quarter comparison
        quarter_data = global_df[global_df['CY_Qtr'] == target_quarter]
        # Year comparison
        year_data = global_df[global_df['Year'] == target_year]
        # Day of week comparison (same day of week in same year)
        dow_data = global_df[
            (global_df['DayOfWeek'] == target_dow) & 
            (global_df['Year'] == target_year)
        ]
        
        # Build response
        response = {
            'target_metrics': {
                'date': target_date,
                'day_of_week': target_dow,
                'quarter': target_quarter,
                'year': str(target_year),
                'census': float(target_record['MDScensus']),
                'total_staff_hprd': float(target_record['Total_Staff_HPRD']),
                'nurse_staff_hprd_excl_admin': float(target_record['Nurse_Staff_HPRD_Excl_Admin']),
                'total_rn_hprd': float(target_record['Total_RN_HPRD']),
                'rn_hprd': float(target_record['RN_HPRD']),
                'rn_admin_hprd': float(target_record['RN_Admin_HPRD']),
                'rn_don_hprd': float(target_record['RN_DON_HPRD']),
                'total_lpn_hprd': float(target_record['Total_LPN_HPRD']),
                'lpn_hprd': float(target_record['LPN_HPRD']),
                'total_nurse_aide_hprd': float(target_record['Total_Nurse_Aide_HPRD']),
                'cna_hprd': float(target_record['CNA_HPRD']),
                'total_staff_hours': float(target_record['Total_Staff_Hours']),
                'total_rn_hours': float(target_record['Total_RN_Hours']),
                'rn_hours': float(target_record['Hrs_RN']),
                'total_lpn_hours': float(target_record['Total_LPN_Hours']),
                'lpn_hours': float(target_record['Hrs_LPN']),
                'total_nurse_aide_hours': float(target_record['Total_Nurse_Aide_Hours']),
                'cna_hours': float(target_record['Hrs_CNA']),
                'total_contract_pct': float(target_record['Total_Contract_Pct']),
                'rn_contract_pct': float(target_record['RN_Contract_Pct']),
                'lpn_contract_pct': float(target_record['LPN_Contract_Pct']),
                'cna_contract_pct': float(target_record['CNA_Contract_Pct'])
            },
            'comparisons': {
                'quarter': {
                    'census': float(quarter_data['MDScensus'].mean()) if len(quarter_data) > 0 else 0,
                    'total_staff_hprd': float(quarter_data['Total_Staff_HPRD'].mean()) if len(quarter_data) > 0 else 0,
                    'nurse_staff_hprd_excl_admin': float(quarter_data['Nurse_Staff_HPRD_Excl_Admin'].mean()) if len(quarter_data) > 0 else 0,
                    'total_rn_hprd': float(quarter_data['Total_RN_HPRD'].mean()) if len(quarter_data) > 0 else 0,
                    'rn_hprd': float(quarter_data['RN_HPRD'].mean()) if len(quarter_data) > 0 else 0,
                    'rn_admin_hprd': float(quarter_data['RN_Admin_HPRD'].mean()) if len(quarter_data) > 0 else 0,
                    'rn_don_hprd': float(quarter_data['RN_DON_HPRD'].mean()) if len(quarter_data) > 0 else 0,
                    'total_lpn_hprd': float(quarter_data['Total_LPN_HPRD'].mean()) if len(quarter_data) > 0 else 0,
                    'lpn_hprd': float(quarter_data['LPN_HPRD'].mean()) if len(quarter_data) > 0 else 0,
                    'total_nurse_aide_hprd': float(quarter_data['Total_Nurse_Aide_HPRD'].mean()) if len(quarter_data) > 0 else 0,
                    'cna_hprd': float(quarter_data['CNA_HPRD'].mean()) if len(quarter_data) > 0 else 0,
                    'total_staff_hours': float(quarter_data['Total_Staff_Hours'].mean()) if len(quarter_data) > 0 else 0,
                    'total_rn_hours': float(quarter_data['Total_RN_Hours'].mean()) if len(quarter_data) > 0 else 0,
                    'rn_hours': float(quarter_data['Hrs_RN'].mean()) if len(quarter_data) > 0 else 0,
                    'total_lpn_hours': float(quarter_data['Total_LPN_Hours'].mean()) if len(quarter_data) > 0 else 0,
                    'lpn_hours': float(quarter_data['Hrs_LPN'].mean()) if len(quarter_data) > 0 else 0,
                    'total_nurse_aide_hours': float(quarter_data['Total_Nurse_Aide_Hours'].mean()) if len(quarter_data) > 0 else 0,
                    'cna_hours': float(quarter_data['Hrs_CNA'].mean()) if len(quarter_data) > 0 else 0,
                    'total_contract_pct': float(quarter_data['Total_Contract_Pct'].mean()) if len(quarter_data) > 0 else 0,
                    'rn_contract_pct': float(quarter_data['RN_Contract_Pct'].mean()) if len(quarter_data) > 0 else 0,
                    'lpn_contract_pct': float(quarter_data['LPN_Contract_Pct'].mean()) if len(quarter_data) > 0 else 0,
                    'cna_contract_pct': float(quarter_data['CNA_Contract_Pct'].mean()) if len(quarter_data) > 0 else 0
                },
                'year': {
                    'census': float(year_data['MDScensus'].mean()) if len(year_data) > 0 else 0,
                    'total_staff_hprd': float(year_data['Total_Staff_HPRD'].mean()) if len(year_data) > 0 else 0,
                    'nurse_staff_hprd_excl_admin': float(year_data['Nurse_Staff_HPRD_Excl_Admin'].mean()) if len(year_data) > 0 else 0,
                    'total_rn_hprd': float(year_data['Total_RN_HPRD'].mean()) if len(year_data) > 0 else 0,
                    'rn_hprd': float(year_data['RN_HPRD'].mean()) if len(year_data) > 0 else 0,
                    'rn_admin_hprd': float(year_data['RN_Admin_HPRD'].mean()) if len(year_data) > 0 else 0,
                    'rn_don_hprd': float(year_data['RN_DON_HPRD'].mean()) if len(year_data) > 0 else 0,
                    'total_lpn_hprd': float(year_data['Total_LPN_HPRD'].mean()) if len(year_data) > 0 else 0,
                    'lpn_hprd': float(year_data['LPN_HPRD'].mean()) if len(year_data) > 0 else 0,
                    'total_nurse_aide_hprd': float(year_data['Total_Nurse_Aide_HPRD'].mean()) if len(year_data) > 0 else 0,
                    'cna_hprd': float(year_data['CNA_HPRD'].mean()) if len(year_data) > 0 else 0,
                    'total_staff_hours': float(year_data['Total_Staff_Hours'].mean()) if len(year_data) > 0 else 0,
                    'total_rn_hours': float(year_data['Total_RN_Hours'].mean()) if len(year_data) > 0 else 0,
                    'rn_hours': float(year_data['Hrs_RN'].mean()) if len(year_data) > 0 else 0,
                    'total_lpn_hours': float(year_data['Total_LPN_Hours'].mean()) if len(year_data) > 0 else 0,
                    'lpn_hours': float(year_data['Hrs_LPN'].mean()) if len(year_data) > 0 else 0,
                    'total_nurse_aide_hours': float(year_data['Total_Nurse_Aide_Hours'].mean()) if len(year_data) > 0 else 0,
                    'cna_hours': float(year_data['Hrs_CNA'].mean()) if len(year_data) > 0 else 0,
                    'total_contract_pct': float(year_data['Total_Contract_Pct'].mean()) if len(year_data) > 0 else 0,
                    'rn_contract_pct': float(year_data['RN_Contract_Pct'].mean()) if len(year_data) > 0 else 0,
                    'lpn_contract_pct': float(year_data['LPN_Contract_Pct'].mean()) if len(year_data) > 0 else 0,
                    'cna_contract_pct': float(year_data['CNA_Contract_Pct'].mean()) if len(year_data) > 0 else 0
                },
                'dow': {
                    'census': float(dow_data['MDScensus'].mean()) if len(dow_data) > 0 else 0,
                    'total_staff_hprd': float(dow_data['Total_Staff_HPRD'].mean()) if len(dow_data) > 0 else 0,
                    'nurse_staff_hprd_excl_admin': float(dow_data['Nurse_Staff_HPRD_Excl_Admin'].mean()) if len(dow_data) > 0 else 0,
                    'total_rn_hprd': float(dow_data['Total_RN_HPRD'].mean()) if len(dow_data) > 0 else 0,
                    'rn_hprd': float(dow_data['RN_HPRD'].mean()) if len(dow_data) > 0 else 0,
                    'rn_admin_hprd': float(dow_data['RN_Admin_HPRD'].mean()) if len(dow_data) > 0 else 0,
                    'rn_don_hprd': float(dow_data['RN_DON_HPRD'].mean()) if len(dow_data) > 0 else 0,
                    'total_lpn_hprd': float(dow_data['Total_LPN_HPRD'].mean()) if len(dow_data) > 0 else 0,
                    'lpn_hprd': float(dow_data['LPN_HPRD'].mean()) if len(dow_data) > 0 else 0,
                    'total_nurse_aide_hprd': float(dow_data['Total_Nurse_Aide_HPRD'].mean()) if len(dow_data) > 0 else 0,
                    'cna_hprd': float(dow_data['CNA_HPRD'].mean()) if len(dow_data) > 0 else 0,
                    'total_staff_hours': float(dow_data['Total_Staff_Hours'].mean()) if len(dow_data) > 0 else 0,
                    'total_rn_hours': float(dow_data['Total_RN_Hours'].mean()) if len(year_data) > 0 else 0,
                    'rn_hours': float(dow_data['Hrs_RN'].mean()) if len(dow_data) > 0 else 0,
                    'total_lpn_hours': float(dow_data['Total_LPN_Hours'].mean()) if len(dow_data) > 0 else 0,
                    'lpn_hours': float(dow_data['Hrs_LPN'].mean()) if len(dow_data) > 0 else 0,
                    'total_nurse_aide_hours': float(dow_data['Total_Nurse_Aide_Hours'].mean()) if len(dow_data) > 0 else 0,
                    'cna_hours': float(dow_data['Hrs_CNA'].mean()) if len(dow_data) > 0 else 0,
                    'total_contract_pct': float(dow_data['Total_Contract_Pct'].mean()) if len(dow_data) > 0 else 0,
                    'rn_contract_pct': float(dow_data['RN_Contract_Pct'].mean()) if len(dow_data) > 0 else 0,
                    'lpn_contract_pct': float(dow_data['LPN_Contract_Pct'].mean()) if len(dow_data) > 0 else 0,
                    'cna_contract_pct': float(dow_data['CNA_Contract_Pct'].mean()) if len(dow_data) > 0 else 0
                }
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chart_aggregated')
@requires_auth
def get_chart_aggregated():
    """Get chart data aggregated by quarter, month, or year for any chart type"""
    try:
        if global_df is None:
            return jsonify({'error': 'Data not loaded'}), 500
        
        aggregation_type = request.args.get('type', 'quarter')  # 'quarter', 'month', or 'year'
        chart_type = request.args.get('chart_type', 'hprd')  # 'hprd', 'hours', 'census', 'contract'
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
            for quarter_val in agg_data.index.tolist():
                if 'Q' in quarter_val:
                    year = quarter_val.split('Q')[0]
                    quarter_num = quarter_val.split('Q')[1]
                    x_values.append(f"Q{quarter_num} {year}")
                else:
                    x_values.append(quarter_val)
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
        return jsonify({'error': str(e)}), 500

def run_dashboard(provnum):
    """Initialize and run the dashboard"""
    global global_df
    
    print(f"🚀 Starting private dashboard for facility {provnum}")
    print(f"📊 Facility Name: {FACILITY_NAME}")
    print(f"🔒 Authentication: Disabled")
    
    # Load facility data
    global_df = create_facility_complete_csv(provnum)
    
    if global_df is None:
        print(f"❌ Failed to load data for facility {provnum}")
        return
    
    print(f"✅ Loaded {len(global_df)} records for facility {provnum}")
    print(f"📅 Date range: {global_df['WorkDate'].min().strftime('%Y-%m-%d')} to {global_df['WorkDate'].max().strftime('%Y-%m-%d')}")
    
    # For Vercel deployment, we don't run the Flask app here
    # Instead, we just prepare the data
    print("✅ Data loaded and ready for deployment")

# For local testing
if __name__ == '__main__':
    run_dashboard(FACILITY_ID)
    
    # Run Flask app for local testing
    print(f"🌐 Dashboard available at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
