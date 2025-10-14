#!/usr/bin/env python3
"""
Dynamic Facility Dashboard Generator
Creates facility-specific dashboards for any 6-digit CCN
"""

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import json
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
import webbrowser
import threading
import time

def round_financial(value, decimals=2):
    """Round using financial rounding (ROUND_HALF_UP)"""
    if pd.isna(value) or value is None:
        return 0.0
    return float(Decimal(str(value)).quantize(Decimal('0.' + '0' * decimals), rounding=ROUND_HALF_UP))

def generate_pbj_source_link(quarter, date, provnum, data_type="nurse"):
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
    
    # Determine PROVNUM case based on quarter
    early_quarters = [
        "2017Q1", "2017Q2", "2017Q3", "2017Q4",
        "2018Q4", 
        "2019Q1", "2019Q2", "2019Q3", "2019Q4",
        "2020Q2", "2020Q3"
    ]
    
    provnum_col = "provnum" if quarter in early_quarters else "PROVNUM"
    
    # Determine URL format based on quarter
    year_int = int(year)
    quarter_int = int(q_num)
    
    if year_int > 2020 or (year_int == 2020 and quarter_int >= 4):
        # Recent quarters - no quarter in URL path
        base_url = "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data"
    else:
        # Older quarters - include quarter in URL path
        base_url = f"https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data/{quarter_url}"
    
    # Build query parameters
    query_params = {
        "filters": {
            "list": [{
                "conditions": [{
                    "column": {"value": provnum_col},
                    "comparator": {"value": "="},
                    "filterValue": [provnum]
                }]
            }],
            "rootConjunction": {"value": "AND"}
        },
        "keywords": "",
        "offset": 0,
        "limit": 10,
        "sort": {"sortBy": None, "sortOrder": None},
        "columns": []
    }
    
    # Add workdate filter for specific date
    if data_type == "nurse":
        workdate_col = "workdate" if quarter in early_quarters else "WorkDate"
    else:
        workdate_col = "WorkDate"
    
    query_params["filters"]["list"][0]["conditions"].append({
        "column": {"value": workdate_col},
        "comparator": {"value": "="},
        "filterValue": [date]
    })
    
    # Convert to URL-encoded JSON
    import urllib.parse
    query_string = urllib.parse.quote(json.dumps(query_params))
    
    return f"{base_url}?query={query_string}"

def format_pbj_source_link(quarter, date, provnum, data_type="nurse"):
    """Format PBJ source link for display."""
    url = generate_pbj_source_link(quarter, date, provnum, data_type)
    if not url:
        return ""
    
    # Format date for display (YYYYMMDD -> MM-DD-YYYY)
    if isinstance(date, str) and len(date) == 8:
        display_date = f"{date[4:6]}-{date[6:8]}-{date[:4]}"
    elif hasattr(date, 'strftime'):
        display_date = date.strftime('%m-%d-%Y')
    else:
        display_date = str(date)
    
    if data_type == "nurse":
        return f'<a href="{url}" target="_blank" style="font-size: 0.9em; color: #333; text-decoration: none; font-weight: 500;">CMS PBJ Source: {display_date}</a>'
    else:
        return f'<a href="{url}" target="_blank" style="font-size: 0.9em; color: #333; text-decoration: none; font-weight: 500;">CMS NonNurse: {display_date}</a>'

def load_facility_data(provnum):
    """Load data for a specific facility."""
    try:
        # Load the complete facility data
        df = pd.read_csv('facility_quarterly_metrics.csv', low_memory=False)
        
        # Format PROVNUM to ensure it's a 6-digit string
        df['PROVNUM'] = df['PROVNUM'].astype(str).str.zfill(6)
        
        # Filter for the specific facility
        facility_df = df[df['PROVNUM'] == provnum].copy()
        
        if facility_df.empty:
            print(f"No data found for facility {provnum}")
            return None
        
        # Convert WorkDate to datetime
        facility_df['WorkDate'] = pd.to_datetime(facility_df['WorkDate'], format='%Y%m%d')
        
        # Add day of week
        facility_df['DayOfWeek'] = facility_df['WorkDate'].dt.day_name()
        facility_df['DayOfWeekNum'] = facility_df['WorkDate'].dt.dayofweek  # 0=Monday, 6=Sunday
        
        # Add month and year
        facility_df['Month'] = facility_df['WorkDate'].dt.month
        facility_df['Year'] = facility_df['WorkDate'].dt.year
        
        # Calculate additional metrics
        facility_df['Total_RN_Hours'] = (facility_df['Hrs_RN'] + facility_df['Hrs_RNadmin'] + facility_df['Hrs_RNDON']).apply(lambda x: round_financial(x, 2))
        facility_df['Total_RN_HPRD'] = (facility_df['Total_RN_Hours'] / facility_df['MDScensus']).apply(lambda x: round_financial(x, 2))
        facility_df['Total_LPN_Hours'] = (facility_df['Hrs_LPN'] + facility_df['Hrs_LPNadmin']).apply(lambda x: round_financial(x, 2))
        facility_df['Total_LPN_HPRD'] = (facility_df['Total_LPN_Hours'] / facility_df['MDScensus']).apply(lambda x: round_financial(x, 2))
        facility_df['Total_Nurse_Aide_Hours'] = (facility_df['Hrs_CNA'] + facility_df['Hrs_NAtrn'] + facility_df['Hrs_MedAide']).apply(lambda x: round_financial(x, 2))
        facility_df['Total_Nurse_Aide_HPRD'] = (facility_df['Total_Nurse_Aide_Hours'] / facility_df['MDScensus']).apply(lambda x: round_financial(x, 2))
        
        # Calculate contract percentages
        facility_df['RN_Contract_Pct'] = (facility_df['Hrs_RN_ctr'] / facility_df['Total_RN_Hours'] * 100).apply(lambda x: round_financial(x, 1))
        facility_df['LPN_Contract_Pct'] = (facility_df['Hrs_LPN_ctr'] / facility_df['Total_LPN_Hours'] * 100).apply(lambda x: round_financial(x, 1))
        facility_df['CNA_Contract_Pct'] = (facility_df['Hrs_CNA_ctr'] / facility_df['Total_Nurse_Aide_Hours'] * 100).apply(lambda x: round_financial(x, 1))
        
        # Calculate total contract percentage
        total_nurse_hours = facility_df['Total_RN_Hours'] + facility_df['Total_LPN_Hours'] + facility_df['Total_Nurse_Aide_Hours']
        total_contract_hours = facility_df['Hrs_RN_ctr'] + facility_df['Hrs_LPN_ctr'] + facility_df['Hrs_CNA_ctr']
        facility_df['Total_Contract_Pct'] = (total_contract_hours / total_nurse_hours * 100).apply(lambda x: round_financial(x, 1))
        
        # Add holiday detection
        def is_federal_holiday(date):
            """Check if a date is a US federal holiday"""
            year = date.year
            month = date.month
            day = date.day
            
            # Fixed holidays
            fixed_holidays = [
                (1, 1),   # New Year's Day
                (7, 4),   # Independence Day
                (12, 25), # Christmas Day
            ]
            
            # Juneteenth (June 19) - federal holiday since 2021
            if month == 6 and day == 19 and year >= 2021:
                return True
            
            for holiday_month, holiday_day in fixed_holidays:
                if month == holiday_month and day == holiday_day:
                    return True
            
            # Variable holidays (calculated for each year)
            # MLK Day (3rd Monday in January)
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
            days_ahead = 0 - first_day.dayofweek()  # Monday is 0
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
            days_ahead = 3 - first_day.dayofweek()  # Thursday is 3
            if days_ahead <= 0:  # Target day already happened this week
                days_ahead += 7
            return first_day + pd.Timedelta(days=days_ahead) + pd.Timedelta(days=21)  # 4th Thursday
        
        def get_last_monday(year, month):
            """Get the last Monday of a given month/year"""
            if month == 12:
                next_month = pd.Timestamp(year + 1, 1, 1)
            else:
                next_month = pd.Timestamp(year, month + 1, 1)
            last_day = next_month - pd.Timedelta(days=1)
            days_back = last_day.dayofweek()  # Monday is 0
            return last_day - pd.Timedelta(days=days_back)
        
        # Apply holiday detection to all dates
        facility_df['IsHoliday'] = facility_df['WorkDate'].apply(is_federal_holiday)
        
        print(f"Loaded {len(facility_df)} records for facility {provnum} from {facility_df['WorkDate'].min().date()} to {facility_df['WorkDate'].max().date()}")
        
        return facility_df
        
    except Exception as e:
        print(f"Error loading data for facility {provnum}: {str(e)}")
        return None

def create_dynamic_dashboard(provnum):
    """Create a dynamic dashboard for any facility."""
    
    # Load facility data
    df = load_facility_data(provnum)
    if df is None:
        return None
    
    # Get facility info
    facility_info = df.iloc[0]
    facility_name = facility_info['PROVNAME']
    state = facility_info['STATE']
    
    # Create Flask app
    app = Flask(__name__)
    
    # Store data globally for the app
    app.df = df
    app.provnum = provnum
    app.facility_name = facility_name
    app.state = state
    
    # API Routes
    @app.route('/')
    def index():
        return render_template('dynamic_facility_dashboard.html', 
                             provnum=provnum, 
                             facility_name=facility_name, 
                             state=state)
    
    @app.route('/api/data')
    def get_data():
        """Get filtered data"""
        try:
            start_date = request.args.get('start_date')
            end_date = request.args.get('end_date')
            day_of_week = request.args.get('day_of_week', 'all')
            quarter = request.args.get('quarter', 'all')
            show_holidays_only = request.args.get('show_holidays_only', 'false') == 'true'
            
            # Filter data
            filtered_df = df.copy()
            
            if start_date:
                filtered_df = filtered_df[filtered_df['WorkDate'] >= start_date]
            if end_date:
                filtered_df = filtered_df[filtered_df['WorkDate'] <= end_date]
            if day_of_week != 'all':
                filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
            if quarter != 'all':
                quarters = quarter.split(',')
                filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
            if show_holidays_only:
                filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
            
            data = filtered_df.to_dict('records')
            
            # Format dates for JSON serialization
            for record in data:
                if 'WorkDate' in record and record['WorkDate'] is not None:
                    record['WorkDate'] = record['WorkDate'].strftime('%Y-%m-%d')
            
            # Get date range
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
            day_of_week = request.args.get('day_of_week', 'all')
            quarter = request.args.get('quarter', 'all')
            show_holidays_only = request.args.get('show_holidays_only', 'false') == 'true'
            
            # Filter data
            filtered_df = df.copy()
            
            if start_date:
                filtered_df = filtered_df[filtered_df['WorkDate'] >= start_date]
            if end_date:
                filtered_df = filtered_df[filtered_df['WorkDate'] <= end_date]
            if day_of_week != 'all':
                filtered_df = filtered_df[filtered_df['DayOfWeek'] == day_of_week]
            if quarter != 'all':
                quarters = quarter.split(',')
                filtered_df = filtered_df[filtered_df['CY_Qtr'].isin(quarters)]
            if show_holidays_only:
                filtered_df = filtered_df[filtered_df['IsHoliday'] == True]
            
            if len(filtered_df) == 0:
                return jsonify({
                    'filter_info': 'No data found for the selected filters',
                    'error': 'No data available for the selected date range and filters'
                })
            
            return jsonify({
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
                'total_rn_hours': float(filtered_df['Hrs_RN'].sum()) if len(filtered_df) > 0 else 0,
                'total_lpn_hours': float(filtered_df['Hrs_LPN'].sum()) if len(filtered_df) > 0 else 0,
                'total_cna_hours': float(filtered_df['Hrs_CNA'].sum()) if len(filtered_df) > 0 else 0,
                'holiday_days': len(filtered_df[filtered_df['IsHoliday'] == True]),
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
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/quarters')
    def get_quarters():
        """Get available quarters"""
        try:
            quarters = sorted(df['CY_Qtr'].unique(), reverse=True)
            return jsonify({'quarters': quarters})
        except Exception as e:
            return jsonify({'error': str(e)})
    
    @app.route('/api/single_day_report')
    def get_single_day_report():
        """Get single day report data"""
        try:
            target_date = request.args.get('date')
            if not target_date:
                return jsonify({'error': 'Date parameter is required'})
            
            target_date = pd.to_datetime(target_date)
            day_data = df[df['WorkDate'] == target_date]
            
            if day_data.empty:
                return jsonify({'error': 'No data found for the selected date'})
            
            day_info = day_data.iloc[0]
            
            # Generate PBJ source link
            quarter = day_info['CY_Qtr']
            date_str = target_date.strftime('%Y%m%d')
            pbj_link = format_pbj_source_link(quarter, date_str, provnum)
            
            return jsonify({
                'date': target_date.strftime('%Y-%m-%d'),
                'quarter': quarter,
                'pbj_link': pbj_link,
                'data': day_info.to_dict()
            })
            
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return app

def main():
    """Main function to run the facility dashboard generator."""
    print("🏥 Facility Dashboard Generator")
    print("=" * 40)
    
    while True:
        try:
            # Get facility CCN from user
            provnum = input("\nEnter 6-digit facility CCN (or 'quit' to exit): ").strip()
            
            if provnum.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            # Validate CCN
            if not provnum.isdigit() or len(provnum) != 6:
                print("❌ Please enter a valid 6-digit CCN (e.g., 225500)")
                continue
            
            # Pad with leading zeros if needed
            provnum = provnum.zfill(6)
            
            print(f"\n🔍 Loading data for facility {provnum}...")
            
            # Create the dashboard
            app = create_dynamic_dashboard(provnum)
            if app is None:
                print(f"❌ Could not create dashboard for facility {provnum}")
                continue
            
            # Get available port
            port = 5000
            while True:
                try:
                    print(f"\n🚀 Starting dashboard for facility {provnum} on port {port}...")
                    print(f"📊 Facility: {app.facility_name}")
                    print(f"🏛️  State: {app.state}")
                    print(f"🌐 Dashboard URL: http://localhost:{port}")
                    print("\nPress Ctrl+C to stop the server and return to facility selection")
                    
                    # Open browser after a short delay
                    def open_browser():
                        time.sleep(2)
                        webbrowser.open(f'http://localhost:{port}')
                    
                    browser_thread = threading.Thread(target=open_browser)
                    browser_thread.daemon = True
                    browser_thread.start()
                    
                    # Run the Flask app
                    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
                    break
                    
                except OSError as e:
                    if "Address already in use" in str(e):
                        port += 1
                        print(f"Port {port-1} is busy, trying port {port}...")
                        continue
                    else:
                        raise
                        
        except KeyboardInterrupt:
            print("\n\n🛑 Server stopped. Returning to facility selection...")
            continue
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            continue

if __name__ == '__main__':
    main()
