"""
Regenerate a report from an existing report folder.
Extracts dates from the HTML file and regenerates with updated styling.
"""
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from calendar import monthrange

# Add current directory to path
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

def parse_date_from_html(html_content: str, folder_path: str = None) -> dict:
    """Extract dates and facility info from HTML content."""
    info = {}
    
    # Try to extract facility number from folder name first (most reliable)
    if folder_path:
        folder_match = re.search(r'facility_(\d{6})_', os.path.basename(folder_path))
        if folder_match:
            info['provnum'] = folder_match.group(1)
    
    # Extract facility number from HTML content (multiple patterns)
    if 'provnum' not in info:
        # Try various patterns
        patterns = [
            r'Provider Number[:\s]+(\d{6})',
            r'<strong>Provider Number:</strong>\s*(\d{6})',
            r'pbj320-(\d{6})',
            r'facility=(\d{6})',
            r'nursing-home/(\d{6})'
        ]
        for pattern in patterns:
            provnum_match = re.search(pattern, html_content)
            if provnum_match:
                info['provnum'] = provnum_match.group(1)
                break
    
    # Extract resident stay period - handle HTML tags
    stay_period_patterns = [
        r'Resident Stay Period[:\s]+([A-Za-z]+ \d{1,2}, \d{4}) to ([A-Za-z]+ \d{1,2}, \d{4})',
        r'<strong>Resident Stay Period:</strong>[^<]*([A-Za-z]+ \d{1,2}, \d{4}) to ([A-Za-z]+ \d{1,2}, \d{4})',
        r'Resident Stay Period[^>]*>([A-Za-z]+ \d{1,2}, \d{4}) to ([A-Za-z]+ \d{1,2}, \d{4})',
        r'from ([A-Za-z]+ \d{1,2}, \d{4}) to ([A-Za-z]+ \d{1,2}, \d{4})',
        r'from ([A-Za-z]+ \d{1,2}, \d{4}) through ([A-Za-z]+ \d{1,2}, \d{4})'
    ]
    
    for pattern in stay_period_patterns:
        stay_period_match = re.search(pattern, html_content)
        if stay_period_match:
            try:
                info['start_date'] = datetime.strptime(stay_period_match.group(1), '%B %d, %Y')
                info['end_date'] = datetime.strptime(stay_period_match.group(2), '%B %d, %Y')
                break
            except ValueError:
                continue
    
    # Extract review period as fallback
    if 'start_date' not in info:
        review_period_patterns = [
            r'Review Period[:\s]+([A-Za-z]+ \d{4}) - ([A-Za-z]+ \d{4})',
            r'<strong>Review Period:</strong>[^<]*([A-Za-z]+ \d{4}) - ([A-Za-z]+ \d{4})'
        ]
        for pattern in review_period_patterns:
            review_period_match = re.search(pattern, html_content)
            if review_period_match:
                try:
                    # Use first day of start month and last day of end month
                    start_month = datetime.strptime(review_period_match.group(1), '%B %Y')
                    end_month = datetime.strptime(review_period_match.group(2), '%B %Y')
                    last_day = monthrange(end_month.year, end_month.month)[1]
                    info['start_date'] = start_month
                    info['end_date'] = datetime(end_month.year, end_month.month, last_day)
                    break
                except ValueError:
                    continue
    
    # Extract key dates from the key dates section
    key_dates = []
    key_dates_section = re.search(r'Key Dates of Interest.*?<ul>(.*?)</ul>', html_content, re.DOTALL)
    if key_dates_section:
        date_matches = re.findall(r'<strong>([A-Za-z]+ \d{1,2}, \d{4})', key_dates_section.group(1))
        for date_str in date_matches:
            try:
                key_dates.append(datetime.strptime(date_str, '%B %d, %Y'))
            except ValueError:
                pass
    info['key_dates'] = key_dates
    
    return info

def main():
    if len(sys.argv) < 2:
        print("Usage: regenerate_report_from_folder.py <report_folder_path>")
        sys.exit(1)
    
    report_folder = sys.argv[1]
    if not os.path.exists(report_folder):
        print(f"Error: Report folder not found: {report_folder}")
        sys.exit(1)
    
    # Find HTML file in the folder
    html_files = list(Path(report_folder).glob("PBJ_Report_*.html"))
    if not html_files:
        print(f"Error: No PBJ_Report_*.html file found in {report_folder}")
        sys.exit(1)
    
    html_file = html_files[0]
    print(f"Reading: {html_file.name}")
    
    # Read and parse HTML
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    info = parse_date_from_html(html_content, report_folder)
    
    if 'provnum' not in info:
        print("Error: Could not extract facility number from HTML")
        sys.exit(1)
    
    if 'start_date' not in info or 'end_date' not in info:
        print("Error: Could not extract dates from HTML")
        print("Please run generate_facility_report.bat to create a new report")
        sys.exit(1)
    
    provnum = info['provnum']
    start_date = info['start_date']
    end_date = info['end_date']
    key_dates = info.get('key_dates', [])
    
    print(f"\nExtracted information:")
    print(f"  Facility: {provnum}")
    print(f"  Start Date: {start_date.strftime('%B %d, %Y')}")
    print(f"  End Date: {end_date.strftime('%B %d, %Y')}")
    print(f"  Key Dates: {len(key_dates)} dates found")
    
    # Load data and generate report (same as main script)
    print(f"\nLoading data for facility {provnum}...")
    df = load_facility_data(provnum)
    
    if df.empty:
        print(f"Error: No data found for facility {provnum}")
        return
    
    facility_info = get_facility_info(df, start_date, end_date)
    print(f"Found facility: {facility_info['name']}")
    print(f"Location: {facility_info['city']}, {facility_info['state']}")
    
    # Filter to date range
    print(f"\nFiltering data to date range...")
    filtered_df = df[
        (df['WorkDate'] >= start_date) &
        (df['WorkDate'] <= end_date)
    ].copy()
    print(f"  Found {len(filtered_df)} records in date range")
    
    # Get unique quarters in the date range
    quarters = sorted(filtered_df['CY_Qtr'].unique())
    print(f"Quarters to analyze: {quarters}")
    
    # Get MACPAC state standards
    print(f"\nGetting MACPAC state standards for {facility_info['state']}...")
    macpac_standards = get_macpac_state_standards(facility_info['state'])
    
    # Calculate quarterly metrics
    print(f"\nCalculating quarterly metrics...")
    quarterly_data = {}
    state_comparisons = {}
    
    for quarter in quarters:
        print(f"\nProcessing {quarter}...")
        
        # Facility metrics - use full df for quarterly calculations (not filtered_df)
        q_metrics = calculate_quarterly_metrics(df, quarter)
        if q_metrics:
            quarterly_data[quarter] = q_metrics
            print(f"  Facility HPRD: {q_metrics['total_hprd']:.2f}, RN HPRD: {q_metrics['rn_hprd']:.2f}")
        
        # State averages
        print(f"  Getting state averages for {quarter}...")
        state_avg = get_state_averages(facility_info['state'], quarter)
        if state_avg:
            state_comparisons[quarter] = state_avg
            print(f"    State avg HPRD: {state_avg['total_hprd']:.2f}, RN HPRD: {state_avg['rn_hprd']:.2f}")
        else:
            print(f"    No state data found")
    
    # Calculate period metrics
    print(f"Calculating period metrics...")
    period_metrics = calculate_period_metrics(filtered_df, start_date, end_date)
    
    # Calculate days under state minimum
    if period_metrics and macpac_standards and macpac_standards.get('min_staffing', 0) > 0:
        print(f"Calculating days under state minimum...")
        state_minimum = macpac_standards.get('min_staffing', 0.0)
        days_under = calculate_days_under_state_minimum(filtered_df, start_date, end_date, state_minimum)
        period_metrics.update(days_under)
    
    # Get daily staffing for key dates
    print(f"\nGetting daily staffing for key dates...")
    daily_staffing = []
    for key_date in key_dates:
        day_data = get_daily_staffing(df, key_date)
        if day_data:
            daily_staffing.append(day_data)
    
    # Load provider info data
    print(f"\nLoading provider info data...")
    provider_info_df = load_provider_info_data(provnum)
    red_flags_history = []
    case_mix_data = []
    
    if not provider_info_df.empty:
        print(f"  Found {len(provider_info_df)} provider info records")
        red_flags_history = extract_red_flags_history(provider_info_df, start_date, end_date)
        print(f"    Found {len(red_flags_history)} periods with red flags")
        
        # Extract case-mix data
        quarters_in_range = sorted(quarterly_data.keys())
        case_mix_data = extract_case_mix_data(provider_info_df, start_date, end_date, quarters_in_range)
        print(f"    Found {len(case_mix_data)} case-mix entries")
    
    # Generate both versions of the report
    print("\nRegenerating HTML reports with updated styling...")
    
    # Generate report with total staffing
    print("  Generating report with total staffing...")
    html_report_with_total = generate_attorney_report(
        provnum=provnum,
        facility_name=facility_info['name'],
        city=facility_info['city'],
        state=facility_info['state'],
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
    
    output_filename_with_total = f"PBJ_Report_{provnum}_{facility_info['name'].replace(' ', '_').replace('&', 'and').replace(',', '').replace('.', '').replace('/', '_')}.html"
    output_path_with_total = os.path.join(report_folder, output_filename_with_total)
    with open(output_path_with_total, 'w', encoding='utf-8') as f:
        f.write(html_report_with_total)
    print(f"  Saved: {output_filename_with_total}")
    
    # Generate report without total staffing
    print("  Generating report without total staffing (direct care focus)...")
    html_report_direct_only = generate_attorney_report(
        provnum=provnum,
        facility_name=facility_info['name'],
        city=facility_info['city'],
        state=facility_info['state'],
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
    
    output_filename_direct_only = f"PBJ_Report_{provnum}_{facility_info['name'].replace(' ', '_').replace('&', 'and').replace(',', '').replace('.', '').replace('/', '_')}_DirectCare.html"
    output_path_direct_only = os.path.join(report_folder, output_filename_direct_only)
    with open(output_path_direct_only, 'w', encoding='utf-8') as f:
        f.write(html_report_direct_only)
    print(f"  Saved: {output_filename_direct_only}")
    
    print("\nReport regeneration complete!")

if __name__ == "__main__":
    main()
