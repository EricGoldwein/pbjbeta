#!/usr/bin/env python3
"""
Generate an attorney-style HTML report for any facility (PBJ_Report_* format).
This is the generic version of generate_facility_415084_report.py
Usage: python generate_facility_report_attorney.py [provnum] [start_date] [end_date] [key_dates]
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import sys
import shutil
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Optional, List, Tuple

# Import all the functions from facility_report_lib.py
# This library contains all the core report generation functions

def parse_date(date_str):
    """Try parsing a date string in multiple formats. Prioritizes MM-DD-YYYY format."""
    # Try MM-DD-YYYY first (user-friendly format)
    for fmt in ("%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    raise ValueError(f"Date {date_str} is not in a recognized format. Use MM-DD-YYYY (e.g., 01-15-2024)")

# Import functions from the facility report library
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from facility_report_lib import (
        load_facility_data,
        get_facility_info,
        calculate_quarterly_metrics,
        get_state_averages,
        calculate_period_metrics,
        calculate_days_under_state_minimum,
        get_daily_staffing,
        load_provider_info_data,
        extract_red_flags_history,
        extract_case_mix_data,
        get_macpac_state_standards,
        generate_attorney_report
    )
except ImportError:
    print("Error: Could not import functions from facility_report_lib.py")
    print("Make sure facility_report_lib.py is in the same directory.")
    sys.exit(1)

# Import canonical identifier functions
try:
    from pbj_identifiers.validators import normalize_ccn, normalize_state_code
except ImportError:
    # Fallback if pbj_identifiers not available
    def normalize_ccn(ccn):
        ccn = str(ccn).strip().upper()
        ccn = ''.join(c for c in ccn if c.isalnum())
        return ccn.zfill(6)
    
    def normalize_state_code(state):
        return str(state).strip().upper()[:2]

def main():
    """Main function to generate the report."""
    # Get inputs from command line or prompt
    if len(sys.argv) > 1:
        provnum = sys.argv[1]
    else:
        provnum = input("Enter 6-digit Facility ID (CCN): ").strip()
    
    # Normalize CCN using canonical identifier layer
    try:
        provnum = normalize_ccn(provnum)
    except ValueError as e:
        print(f"Error: Invalid CCN format: {e}")
        return
    
    if len(sys.argv) > 2:
        start_date = parse_date(sys.argv[2])
    else:
        start_date_str = input("Enter Start Date (MM-DD-YYYY, e.g., 04-21-2024): ").strip()
        start_date = parse_date(start_date_str)
    
    if len(sys.argv) > 3:
        end_date = parse_date(sys.argv[3])
    else:
        end_date_str = input("Enter End Date (MM-DD-YYYY, e.g., 05-10-2024): ").strip()
        end_date = parse_date(end_date_str)
    
    if len(sys.argv) > 4:
        key_dates_str = sys.argv[4]
    else:
        key_dates_str = input("Enter Key Dates (comma-separated, MM-DD-YYYY, e.g., 04-23-2024, 04-24-2024, or press Enter for none): ").strip()
    
    # Parse key dates
    if key_dates_str:
        key_dates = [parse_date(date.strip()) for date in key_dates_str.split(",")]
    else:
        key_dates = []
    
    print(f"\nGenerating report for facility {provnum}...")
    print(f"Period: {start_date.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}")
    if key_dates:
        print(f"Key dates: {', '.join([d.strftime('%B %d, %Y') for d in key_dates])}")
    
    # Check if facility dashboard CSV exists, create if needed
    csv_file = f"facility_{provnum}_complete_data.csv"
    if not os.path.exists(csv_file):
        print(f"\nFacility dashboard CSV not found. Creating it now...")
        print("This may take a few minutes...")
        try:
            from dynamic_facility_dashboard import create_facility_complete_csv
            create_facility_complete_csv(provnum)
            if not os.path.exists(csv_file):
                print(f"Warning: Failed to create facility CSV, proceeding with existing data loading...")
        except Exception as e:
            print(f"Warning: Could not create facility CSV: {e}")
            print("Proceeding with existing data loading...")
    
    print(f"\nLoading data for facility {provnum}...")
    df = load_facility_data(provnum)
    
    if df.empty:
        print(f"Error: No data found for facility {provnum}")
        return
    
    # Get facility info using the resident stay period to get the correct facility name
    facility_info = get_facility_info(df, start_date, end_date)
    print(f"Found facility: {facility_info['name']}")
    print(f"Location: {facility_info['city']}, {facility_info['state']}")
    
    # Normalize state code using canonical identifier layer
    state_code = normalize_state_code(facility_info['state'])
    facility_info['state'] = state_code  # Update with normalized state
    
    # Get MACPAC state standards
    print(f"\nGetting MACPAC state standards for {state_code}...")
    macpac_standards = get_macpac_state_standards(state_code)
    if macpac_standards:
        print(f"  Found: {macpac_standards['display_text']}")
    else:
        print(f"  No MACPAC standards found")
    
    # Filter to quarters that overlap with the date range
    start_quarter = f"{start_date.year}Q{(start_date.month - 1) // 3 + 1}"
    end_quarter = f"{end_date.year}Q{(end_date.month - 1) // 3 + 1}"
    
    # Filter to quarters in the range
    all_quarters = sorted(df['CY_Qtr'].unique())
    quarters_in_range = [q for q in all_quarters if q >= start_quarter and q <= end_quarter]
    
    filtered_df = df[df['CY_Qtr'].isin(quarters_in_range)].copy()
    
    print(f"Quarters in date range: {quarters_in_range}")
    print(f"Total records: {len(filtered_df)}")
    
    if filtered_df.empty:
        print(f"Warning: No data found for the specified date range, using all available data")
        filtered_df = df.copy()
    
    # Ensure filtered_df is a DataFrame (not Series)
    if not isinstance(filtered_df, pd.DataFrame):
        filtered_df = pd.DataFrame(filtered_df)
    
    # Get unique quarters in the date range
    quarters = sorted(filtered_df['CY_Qtr'].unique())
    print(f"Quarters to analyze: {quarters}")
    
    # Calculate quarterly metrics
    quarterly_data = {}
    state_comparisons = {}
    
    for quarter in quarters:
        print(f"\nProcessing {quarter}...")
        
        # Facility metrics
        # Facility metrics - use full df for quarterly calculations (not filtered_df)
        q_metrics = calculate_quarterly_metrics(df, quarter)
        if q_metrics:
            quarterly_data[quarter] = q_metrics
            print(f"  Facility HPRD: {q_metrics['total_hprd']:.2f}, RN HPRD: {q_metrics['rn_hprd']:.2f}")
        
        # State averages
        print(f"  Getting state averages for {quarter}...")
        state_avg = get_state_averages(state_code, quarter)
        if state_avg:
            state_comparisons[quarter] = state_avg
            print(f"    State avg HPRD: {state_avg['total_hprd']:.2f}, RN HPRD: {state_avg['rn_hprd']:.2f}")
        else:
            print(f"    No state data found")
    
    # Calculate period-specific metrics (exact date range)
    print(f"\nCalculating period-specific metrics for {start_date.date()} to {end_date.date()}...")
    period_metrics = calculate_period_metrics(df, start_date, end_date)
    if period_metrics:
        print(f"  Period Total HPRD: {period_metrics['total_hprd']:.2f}")
        print(f"  Period Direct Care HPRD: {period_metrics.get('direct_care_hprd', 0):.2f}")
        print(f"  Period RN HPRD: {period_metrics['rn_hprd']:.2f}")
        print(f"  Period Direct Care RN HPRD: {period_metrics['direct_care_rn_hprd']:.2f}")
    else:
        print(f"  Could not calculate period metrics")
    
    # Calculate days under state minimum
    if macpac_standards and macpac_standards.get('min_staffing', 0) > 0:
        print(f"\nCalculating days under state minimum ({macpac_standards['min_staffing']:.2f} HPRD)...")
        days_under = calculate_days_under_state_minimum(df, start_date, end_date, macpac_standards['min_staffing'])
        if period_metrics:
            period_metrics.update(days_under)
        print(f"  Total days: {days_under['total_days']:,}")
        print(f"  Days under minimum (Total HPRD): {days_under['days_under_minimum_total']:,} ({days_under['percentage_under_total']:.1f}%)")
        print(f"  Days under minimum (Direct Care HPRD): {days_under['days_under_minimum_direct']:,} ({days_under['percentage_under_direct']:.1f}%)")
    
    # Get daily staffing for key dates
    print(f"\nGetting daily staffing for key dates...")
    daily_staffing = []
    for key_date in key_dates:
        print(f"  Getting data for {key_date.strftime('%B %d, %Y')}...")
        day_data = get_daily_staffing(df, key_date)
        if day_data:
            daily_staffing.append(day_data)
            print(f"    Census: {day_data['census']:.0f}, Total HPRD: {day_data['total_hprd']:.2f}, RN HPRD: {day_data['rn_hprd']:.2f}")
        else:
            print(f"    No data found for this date")
    
    # Load provider info data for historical context
    print(f"\nLoading provider info data for historical context...")
    provider_info_df = load_provider_info_data(provnum)
    red_flags_history = []
    case_mix_data = []
    
    if not provider_info_df.empty:
        print(f"  Found {len(provider_info_df)} provider info records")
        
        # Extract red flags history
        print(f"  Extracting red flags history...")
        red_flags_history = extract_red_flags_history(provider_info_df, start_date, end_date)
        print(f"    Found {len(red_flags_history)} periods with red flags")
        
        # Extract case-mix data
        print(f"  Extracting case-mix adjusted staffing data...")
        case_mix_data = extract_case_mix_data(provider_info_df, start_date, end_date, quarters_in_range)
        print(f"    Found {len(case_mix_data)} periods with case-mix data")
    else:
        print(f"  No provider info data found")
    
    # Generate both versions of the report
    print("\nGenerating HTML reports...")
    
    # Save reports to pbj320-reports folder
    pbj320_reports_dir = "pbj320-reports"
    os.makedirs(pbj320_reports_dir, exist_ok=True)
    facility_name_safe = facility_info['name'].replace(' ', '_').replace('&', 'and').replace(',', '').replace('.', '').replace('/', '_')
    print(f"\nSaving reports to: {pbj320_reports_dir}/")
    
    # Generate report with total staffing (original version)
    print("  Generating report with total staffing...")
    html_report_with_total = generate_attorney_report(
        provnum=provnum,
        facility_name=facility_info['name'],
        city=facility_info['city'],
        state=state_code,
        start_date=start_date,
        end_date=end_date,
        key_dates=key_dates,
        quarterly_data=quarterly_data,
        state_comparisons=state_comparisons,
        period_metrics=period_metrics,
        daily_staffing=daily_staffing,
        macpac_standards=macpac_standards,
        red_flags_history=red_flags_history,
        case_mix_data=case_mix_data,
        pbj_df=df,
        include_total_staffing=True
    )
    
    output_filename_with_total = f"PBJ_Report_{provnum}_{facility_name_safe}.html"
    output_path_with_total = os.path.join(pbj320_reports_dir, output_filename_with_total)
    with open(output_path_with_total, 'w', encoding='utf-8') as f:
        f.write(html_report_with_total)
    print(f"  Saved: {output_filename_with_total}")
    
    # Generate report without total staffing (direct care focus with appendix)
    print("  Generating report without total staffing (direct care focus)...")
    html_report_direct_only = generate_attorney_report(
        provnum=provnum,
        facility_name=facility_info['name'],
        city=facility_info['city'],
        state=state_code,
        start_date=start_date,
        end_date=end_date,
        key_dates=key_dates,
        quarterly_data=quarterly_data,
        state_comparisons=state_comparisons,
        period_metrics=period_metrics,
        daily_staffing=daily_staffing,
        macpac_standards=macpac_standards,
        red_flags_history=red_flags_history,
        case_mix_data=case_mix_data,
        pbj_df=df,
        include_total_staffing=False
    )
    
    output_filename_direct_only = f"PBJ_Report_{provnum}_{facility_name_safe}_DirectCare.html"
    output_path_direct_only = os.path.join(pbj320_reports_dir, output_filename_direct_only)
    with open(output_path_direct_only, 'w', encoding='utf-8') as f:
        f.write(html_report_direct_only)
    print(f"  Saved: {output_filename_direct_only}")
    
    # CSV files are now in pbj320-XXXXX folders, no need to copy or create README
    
    print(f"\nReport generated successfully!")
    print(f"Report folder: {report_folder}")
    print(f"Report file: {output_path}")
    print(f"Open {output_path} in your web browser to view the formatted report.")

if __name__ == "__main__":
    main()



