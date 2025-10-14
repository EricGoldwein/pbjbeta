import datetime
import sys
import os
import pandas as pd
import re
import glob
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the dynamic facility dashboard functions to load data
from dynamic_facility_dashboard import create_facility_complete_csv, load_facility_data
from generate_report import generate_pbj_report

def fix_capitalization(text):
    """Fix capitalization to follow sentence case rules (first letter capitalized except for prepositions like 'in', 'of', 'at', etc.)"""
    if not text:
        return text
    
    # Words that should remain lowercase
    lowercase_words = {'in', 'of', 'at', 'and', 'the', 'a', 'an', 'for', 'to', 'with', 'by', 'on', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'shall'}
    
    # Split into words and fix capitalization
    words = text.split()
    fixed_words = []
    
    for i, word in enumerate(words):
        # Clean the word of punctuation for processing
        clean_word = re.sub(r'[^\w]', '', word)
        if clean_word.lower() in lowercase_words and i > 0:
            fixed_words.append(word.lower())
        else:
            # Capitalize first letter, keep rest as-is
            if word:
                fixed_words.append(word[0].upper() + word[1:].lower())
    
    return ' '.join(fixed_words)

def load_state_metrics(state_code):
    """Load state metrics for the given state code"""
    try:
        state_df = pd.read_csv('state_lite_metrics.csv')
        state_data = state_df[state_df['STATE'] == state_code].copy()
        return state_data
    except Exception as e:
        print(f"Error loading state metrics: {e}")
        return pd.DataFrame()

def get_state_data_for_quarters(state_data, quarters):
    """Get state data for specific quarters"""
    state_quarterly_data = {}
    for quarter in quarters:
        quarter_data = state_data[state_data['CY_Qtr'] == quarter]
        if not quarter_data.empty:
            row = quarter_data.iloc[0]
            state_quarterly_data[quarter] = {
                'total_nurse_hprd': row['Total_Nurse_HPRD'],
                'direct_care_rn_hprd': row['Direct_Care_RN_HPRD']
            }
        else:
            # Use federal minimum as fallback
            state_quarterly_data[quarter] = {
                'total_nurse_hprd': 0.30,
                'direct_care_rn_hprd': 0.30
            }
    return state_quarterly_data

def load_provider_info_data(provnum):
    """Load Provider Info data for a specific facility"""
    try:
        # Get all Provider Info files
        provider_files = glob.glob('provider_info_normalized/ProviderInfoNorm_*.csv')
        
        if not provider_files:
            print("No Provider Info files found")
            return pd.DataFrame()
        
        # Load and combine all Provider Info data for this facility
        all_provider_data = []
        
        for file_path in provider_files:
            try:
                df = pd.read_csv(file_path, low_memory=False)
                # Filter for this facility
                facility_data = df[df['ccn'] == provnum]
                if not facility_data.empty:
                    all_provider_data.append(facility_data)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        if all_provider_data:
            combined_df = pd.concat(all_provider_data, ignore_index=True)
            # Convert processing_date to datetime
            combined_df['processing_date'] = pd.to_datetime(combined_df['processing_date'])
            # Sort by processing_date to get most recent data first
            combined_df = combined_df.sort_values('processing_date', ascending=False)
            return combined_df
        else:
            print(f"No Provider Info data found for facility {provnum}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading Provider Info data: {e}")
        return pd.DataFrame()

def get_provider_info_for_quarters(provider_df, quarters):
    """Get Provider Info data for specific quarters"""
    provider_quarterly_data = {}
    
    if provider_df.empty:
        return provider_quarterly_data
    
    # Get the most recent Provider Info record as fallback
    latest_record = provider_df.iloc[0] if not provider_df.empty else None
    
    for quarter in quarters:
        # Find the most recent Provider Info record for this quarter
        quarter_data = provider_df[provider_df['quarter'] == quarter]
        
        if not quarter_data.empty:
            # Get the most recent record for this quarter
            record = quarter_data.iloc[0]
        elif latest_record is not None:
            # Use the most recent available record as fallback
            record = latest_record
            print(f"Using latest Provider Info data for quarter {quarter} (no specific quarter data found)")
        else:
            # No data available at all
            provider_quarterly_data[quarter] = {
                'case_mix_total_hprd': '-',
                'case_mix_rn_hprd': '-',
                'staffing_rating': '-',
                'overall_rating': '-',
                'health_inspection_rating': '-',
                'total_staff_turnover': '-',
                'rn_turnover': '-'
            }
            continue
        
        provider_quarterly_data[quarter] = {
            'case_mix_total_hprd': record.get('case_mix_total_nurse_hrs_per_resident_per_day', '-'),
            'case_mix_rn_hprd': record.get('case_mix_rn_hrs_per_resident_per_day', '-'),
            'staffing_rating': record.get('staffing_rating', '-'),
            'overall_rating': record.get('overall_rating', '-'),
            'health_inspection_rating': record.get('health_inspection_rating', '-'),
            'total_staff_turnover': record.get('total_nursing_staff_turnover', '-'),
            'rn_turnover': record.get('registered_nurse_turnover', '-')
        }
    
    return provider_quarterly_data

def parse_date(date_str):
    """Try parsing a date string in multiple formats."""
    for fmt in ("%m-%d-%Y", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date {date_str} is not in a recognized format.")

def generate_report_for_facility(facility_id, resident_stay_start, resident_stay_end, key_dates):
    print(f"Loading data for facility {facility_id}...")
    
    # Load the facility data using the dynamic facility dashboard
    try:
        # Create the facility CSV if it doesn't exist
        csv_file = f"facility_{facility_id}_complete_data.csv"
        if not os.path.exists(csv_file):
            print(f"Creating CSV for facility {facility_id}...")
            create_facility_complete_csv(facility_id)
        
        # Load the facility data
        print("Loading facility data...")
        df = load_facility_data(facility_id)
        
        if df is None or df.empty:
            print("No data available for this facility.")
            return
            
        # Get facility info from the data (same way as dynamic_facility_dashboard.py does it)
        facility_name = fix_capitalization(df['PROVNAME'].iloc[0]) if 'PROVNAME' in df.columns else "Unknown Facility"
        city = fix_capitalization(df['CITY'].iloc[0]) if 'CITY' in df.columns else "Unknown"
        state = df['STATE'].iloc[0] if 'STATE' in df.columns else "Unknown"
        county_name = fix_capitalization(df['COUNTY_NAME'].iloc[0]) if 'COUNTY_NAME' in df.columns else "Unknown"
        
        # Calculate metrics from the data (using the same column names as the dashboard)
        metrics = {
            'min_hppd': df['Total_Nurse_HPRD'].min() if 'Total_Nurse_HPRD' in df.columns else 0,
            'max_hppd': df['Total_Nurse_HPRD'].max() if 'Total_Nurse_HPRD' in df.columns else 0,
            'min_rn_hppd': df['Total_RN_HPRD'].min() if 'Total_RN_HPRD' in df.columns else 0,
            'max_rn_hppd': df['Total_RN_HPRD'].max() if 'Total_RN_HPRD' in df.columns else 0,
            'min_rating': 1,  # Default values since we don't have rating data
            'max_rating': 5
        }
        
        print(f"Found {len(df)} records for {facility_name}")
        print(f"Location: {city}, {state} - {county_name}")
        print(f"Metrics: Total HPRD range {metrics['min_hppd']:.2f} - {metrics['max_hppd']:.2f}")
        
        # Load state metrics and Provider Info data
        state_data = load_state_metrics(state)
        provider_info_df = load_provider_info_data(facility_id)
        
        # Generate quarterly data for the report
        quarterly_data = {}
        state_comparison_data = {}
        
        # Get quarterly aggregated data
        df['Year'] = pd.to_datetime(df['WorkDate']).dt.year
        df['Quarter'] = pd.to_datetime(df['WorkDate']).dt.quarter
        
        # Filter by date range
        date_filter = (pd.to_datetime(df['WorkDate']) >= resident_stay_start) & (pd.to_datetime(df['WorkDate']) <= resident_stay_end)
        filtered_df = df[date_filter].copy()
        
        if not filtered_df.empty:
            # Aggregate by quarter
            quarterly_agg = filtered_df.groupby(['Year', 'Quarter']).agg({
                'MDScensus': 'mean',
                'Total_Nurse_HPRD': 'mean',  # Direct Care HPRD
                'Total_Staff_HPRD': 'mean',  # Reported Total HPRD
                'RN_HPRD': 'mean',           # Direct RN HPPD
                'Total_RN_HPRD': 'mean'      # Reported Total RN HPRD
            }).round(2)
            
            # Get quarters for state data and Provider Info lookup
            quarters = [f"{year}Q{quarter}" for year, quarter in quarterly_agg.index]
            state_quarterly_data = get_state_data_for_quarters(state_data, quarters)
            provider_quarterly_data = get_provider_info_for_quarters(provider_info_df, quarters)
            
            # Create quarterly data
            for (year, quarter), row in quarterly_agg.iterrows():
                quarter_label = f"{year}Q{quarter}"
                
                # Get Provider Info data for this quarter
                provider_data = provider_quarterly_data.get(quarter_label, {})
                
                quarterly_data[quarter_label] = {
                    'census': f"{row['MDScensus']:.1f}",
                    'direct_care_hprd': f"{row['Total_Nurse_HPRD']:.3f}",
                    'reported_total_hprd': f"{row['Total_Staff_HPRD']:.3f}",
                    'case_mix_total_hprd': f"{provider_data.get('case_mix_total_hprd', '-'):.3f}" if provider_data.get('case_mix_total_hprd') != '-' and pd.notna(provider_data.get('case_mix_total_hprd')) else '-',
                    'direct_rn_hppd': f"{row['RN_HPRD']:.3f}",
                    'reported_total_rn_hprd': f"{row['Total_RN_HPRD']:.3f}",
                    'case_mix_rn_hprd': f"{provider_data.get('case_mix_rn_hprd', '-'):.3f}" if provider_data.get('case_mix_rn_hprd') != '-' and pd.notna(provider_data.get('case_mix_rn_hprd')) else '-',
                    'staffing_rating': provider_data.get('staffing_rating', '-'),
                    'overall_rating': provider_data.get('overall_rating', '-'),
                    'health_inspection_rating': provider_data.get('health_inspection_rating', '-'),
                    'total_staff_turnover': f"{provider_data.get('total_staff_turnover', '-'):.1f}" if provider_data.get('total_staff_turnover') != '-' and pd.notna(provider_data.get('total_staff_turnover')) else '-',
                    'rn_turnover': f"{provider_data.get('rn_turnover', '-'):.1f}" if provider_data.get('rn_turnover') != '-' and pd.notna(provider_data.get('rn_turnover')) else '-'
                }
                
                # State comparison data using actual state metrics
                if quarter_label in state_quarterly_data:
                    state_info = state_quarterly_data[quarter_label]
                    state_comparison_data[quarter_label] = {
                        'facility_direct_care_hppd': f"{row['Total_Nurse_HPRD']:.3f}",
                        'state_direct_care_hppd': f"{state_info['total_nurse_hprd']:.3f}",
                        'facility_direct_rn_hppd': f"{row['RN_HPRD']:.3f}",
                        'state_direct_rn_hppd': f"{state_info['direct_care_rn_hprd']:.3f}"
                    }
                else:
                    # Fallback to federal minimum
                    state_comparison_data[quarter_label] = {
                        'facility_direct_care_hppd': f"{row['Total_Nurse_HPRD']:.3f}",
                        'state_direct_care_hppd': '0.300',
                        'facility_direct_rn_hppd': f"{row['RN_HPRD']:.3f}",
                        'state_direct_rn_hppd': '0.300'
                    }
        
        # Get actual data for key dates
        key_dates_data = {}
        for date in key_dates:
            date_str = date.strftime('%Y-%m-%d')
            date_data = filtered_df[filtered_df['WorkDate'] == date_str]
            if not date_data.empty:
                row = date_data.iloc[0]
                key_dates_data[date_str] = {
                    'census': f"{row['MDScensus']:.0f}",
                    'direct_care_hppd': f"{row['Total_Nurse_HPRD']:.3f}",
                    'total_hppd': f"{row['Total_Staff_HPRD']:.3f}",
                    'rn_hppd': f"{row['RN_HPRD']:.3f}",
                    'total_rn_hppd': f"{row['Total_RN_HPRD']:.3f}"
                }
            else:
                key_dates_data[date_str] = {
                    'census': 'N/A',
                    'direct_care_hppd': 'N/A',
                    'total_hppd': 'N/A',
                    'rn_hppd': 'N/A',
                    'total_rn_hppd': 'N/A'
                }
        
        # Generate the report
        report = generate_pbj_report(
            facility_name=facility_name,
            location=f"{city}, {state}",
            ccn=facility_id,
            affiliate_entity="Elder Services",  # Default entity
            review_period=f"{resident_stay_start.strftime('%B %Y')} - {resident_stay_end.strftime('%B %Y')}",
            dates_of_interest=key_dates,
            metrics=metrics,
            quarterly_data=quarterly_data,
            state_comparison_data=state_comparison_data,
            key_dates_data=key_dates_data
        )
        
        # Save report to HTML file
        output_filename = f"PBJ_Brief_{facility_id}_{facility_name.replace(' ', '_')}.html"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReport generated successfully!")
        print(f"Saved as: {output_filename}")
        print(f"Open this file in your web browser to view the formatted report.")
        
    except Exception as e:
        print(f"Error loading facility data: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    # Ask for user input
    facility_id = input("Enter 6-digit Facility ID: ")
    resident_stay_start = parse_date(input("Enter Start Date of Resident Stay (YYYY-MM-DD): "))
    resident_stay_end = parse_date(input("Enter End Date of Resident Stay (YYYY-MM-DD): "))
    key_dates_input = input("Enter Key Dates (comma-separated, YYYY-MM-DD): ")

    # Process input
    key_dates = [parse_date(date.strip()) for date in key_dates_input.split(",")]

    # Generate the report
    generate_report_for_facility(facility_id, resident_stay_start, resident_stay_end, key_dates)
