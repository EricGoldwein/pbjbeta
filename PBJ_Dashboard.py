import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import re
from plotly.subplots import make_subplots
import duckdb
from typing import Dict, Optional, List, Tuple, Any

# Set sidebar collapsed on mobile
import streamlit as st
if st.session_state.get('is_mobile', False):
    st.set_page_config(page_title="Nursing Home Staffing Data by 320", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")
else:
    st.set_page_config(page_title="Nursing Home Staffing Data by 320", page_icon="📊", layout="wide", initial_sidebar_state="auto")

# Add subtle modern styling for metric containers only (not delta or value)
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background: #fafdff;
        border: 1px solid #e3eaf3;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(30,136,229,0.04);
        padding: 12px 18px 4px 18px;
        margin: 12px 4px 10px 4px;
    }
    </style>
""", unsafe_allow_html=True)

# Add CSS to hide the toggle tip on desktop
st.markdown("""
<style>
.toggle-tip-mobile {
    display: block;
}
@media (min-width: 900px) {
    .toggle-tip-mobile {
        display: none !important;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize DuckDB connection for facility data
facility_db = duckdb.connect(':memory:')

# Initialize provider info cache
provider_info_cache: Dict[str, Dict[str, str]] = {}

state_name_map = {
    'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona', 'CA': 'California', 'CO': 'Colorado',
    'CT': 'Connecticut', 'DC': 'District of Columbia', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland', 'ME': 'Maine',
    'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana',
    'NC': 'North Carolina', 'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
    'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon',
    'PA': 'Pennsylvania', 'PR': 'Puerto Rico', 'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota',
    'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VA': 'Virginia', 'VI': 'Virgin Islands', 'VT': 'Vermont',
    'WA': 'Washington', 'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming', 'USA': 'USA', 'US': 'USA'
}

def get_full_state_name(state_abbr: str) -> str:
    """Return the full state name for a given abbreviation."""
    return state_name_map.get(state_abbr, state_abbr)

@st.cache_data
def load_metrics_data():
    """Load and cache all metrics data."""
    try:
        # Load all metrics data at once
        national_metrics = pd.read_csv('national_lite_metrics.csv')
        state_metrics = pd.read_csv('state_lite_metrics.csv')
        facility_metrics = pd.read_csv('facility_lite_metrics.csv', dtype={'PROVNUM': str})

        # Standardize column names
        for df in [national_metrics, state_metrics, facility_metrics]:
            # Rename columns to match expected format
            column_mapping = {
                'CY_Qtr': 'CY_QTR',
                'Census': 'Census',
                'Total_Nurse_HPRD': 'Total_Nurse_HPRD',
                'Contract_Percentage': 'Contract_Percentage',
                'Facility_Count': 'Facility_Count',
                'MDS': 'Census'  # Map MDS to Census for national metrics
            }
            df.rename(columns=column_mapping, inplace=True)

        # Convert CY_QTR to datetime for all dataframes
        for df in [national_metrics, state_metrics, facility_metrics]:
            df['date'] = pd.to_datetime(df['CY_QTR'].str[:4] + '-' + 
                                      ((df['CY_QTR'].str[-1].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2) + 
                                      '-01')
        
        return national_metrics, state_metrics, facility_metrics
    except Exception as e:
        st.error(f"Error loading metrics data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_affiliated_entity_data():
    """Load and cache affiliated entity performance measures data."""
    try:
        df = pd.read_csv('Nursing_Home_Affiliated_Entity_Performance_Measures_Jun_2025.csv')
        
        # Clean and standardize the data
        # Convert percentage columns to numeric, handling empty strings
        percentage_columns = [
            'Percentage of facilities with an abuse icon',
            'Percent of facilities classified as for-profit',
            'Percent of facilities classified as non-profit',
            'Percent of facilities classified as government-owned',
            'Average total nursing staff turnover percentage',
            'Average Registered Nurse turnover percentage',
            'Average percentage of short-stay residents who were re-hospitalized after a nursing home admission',
            'Average percentage of short-stay residents who have had an outpatient emergency department visit',
            'Average percentage of short-stay residents who newly received an antipsychotic medication',
            'Average percentage of short-stay residents with pressure ulcers or pressure injuries that are new or worsened',
            'Average percentage of short-stay residents who are at or above an expected ability to care for themselves and move around at discharge',
            'Average percentage of short-stay residents who were assessed and appropriately given the seasonal influenza vaccine',
            'Average percentage of short-stay residents who were assessed and appropriately given the  pneumococcal vaccine',
            'Average percentage of long-stay residents who received an antipsychotic medication',
            'Average percentage of long-stay residents experiencing one or more falls with major injury',
            'Average percentage of long-stay residents with pressure ulcers',
            'Average percentage of long-stay residents with a urinary tract infection',
            'Average percentage of long-stay residents who have or had a catheter inserted and left in their bladder',
            'Average percentage of long-stay residents whose ability to move independently worsened',
            'Average percentage of long-stay residents whose need for help with activities of daily living has increased',
            'Average percentage of long-stay residents who were assessed and appropriately given the seasonal influenza vaccine',
            'Average percentage of long-stay residents who were assessed and appropriately given the  pneumococcal vaccine',
            'Average percentage of long-stay residents who were physically restrained',
            'Average percentage of long-stay residents with new or worsened bowel or bladder incontinence',
            'Average percentage of long-stay residents who lose too much weight',
            'Average percentage of long-stay residents who have symptoms of depression',
            'Average percentage of long-stay residents who used antianxiety or hypnotic medication',
            'Average rate of potentially preventable hospital readmissions 30 days after discharge from a SNF',
            'Average percentage of current residents up to date with COVID-19 vaccines',
            'Average percentage of healthcare personnel up to date with COVID-19 vaccines'
        ]
        
        for col in percentage_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert numeric columns
        numeric_columns = [
            'Number of facilities',
            'Number of states and territories with operations',
            'Number of Special Focus Facilities (SFF)',
            'Number of SFF candidates',
            'Number of facilities with an abuse icon',
            'Average overall 5-star rating',
            'Average health inspection rating',
            'Average staffing rating',
            'Average quality rating',
            'Average total nurse hours per resident day',
            'Average total weekend nurse hours per resident day',
            'Average total Registered Nurse hours per resident day',
            'Average number of administrators who have left the nursing home',
            'Total number of fines',
            'Average number of fines',
            'Total amount of fines in dollars',
            'Average amount of fines in dollars',
            'Total number of payment denials',
            'Average number of payment denials',
            'Average number of hospitalizations per 1,000 long-stay resident days',
            'Average number of outpatient emergency department visits per 1,000 long-stay resident days'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading affiliated entity data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_provider_info_data():
    """Load and cache provider information data."""
    try:
        df = pd.read_csv('NH_ProviderInfo_Jun2025.csv', dtype={'CMS Certification Number (CCN)': str})
        
        # Clean and standardize the data
        # Convert numeric columns
        numeric_columns = [
            'Number of Certified Beds',
            'Average Number of Residents per Day',
            'Overall Rating',
            'Health Inspection Rating',
            'Staffing Rating',
            'QM Rating'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading provider info data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_march_provider_info_data():
    """Load and cache March 2025 provider information data for comparison."""
    try:
        df = pd.read_csv('NH_ProviderInfo_Mar2025.csv', dtype={'CMS Certification Number (CCN)': str})
        
        # Clean and standardize the data
        # Convert numeric columns
        numeric_columns = [
            'Number of Certified Beds',
            'Average Number of Residents per Day',
            'Overall Rating',
            'Health Inspection Rating',
            'Staffing Rating',
            'QM Rating'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading March provider info data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def create_facility_db():
    """Create an optimized DuckDB database for facility data."""
    try:
        # Load facility metrics into DuckDB
        print("Loading facility metrics from CSV...")
        facility_metrics = pd.read_csv('facility_lite_metrics.csv', dtype={'PROVNUM': str})
        
        if facility_metrics.empty:
            print("Warning: facility_metrics DataFrame is empty")
            return
            
        print(f"Loaded {len(facility_metrics)} facility records")
        print("Sample of facility data:")
        print(facility_metrics.head())
        
        # Rename columns to match expected format
        column_mapping = {
            'CY_Qtr': 'CY_QTR',
            'Census': 'Census',
            'Total_Nurse_HPRD': 'Total_Nurse_HPRD',
            'Contract_Percentage': 'Contract_Percentage'
        }
        facility_metrics.rename(columns=column_mapping, inplace=True)
        
        # Add date column before creating table
        facility_metrics['date'] = pd.to_datetime(facility_metrics['CY_QTR'].str[:4] + '-' + 
                                                ((facility_metrics['CY_QTR'].str[-1].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2) + 
                                                '-01')
        
        # Drop the table if it exists
        facility_db.execute("DROP TABLE IF EXISTS facility_metrics")
        
        # Create the table with explicit schema
        facility_db.execute("""
            CREATE TABLE facility_metrics (
                PROVNUM VARCHAR,
                PROVNAME VARCHAR,
                STATE VARCHAR,
                COUNTY_NAME VARCHAR,
                CY_QTR VARCHAR,
                Census DOUBLE,
                Total_Nurse_HPRD DOUBLE,
                Contract_Percentage DOUBLE,
                date DATE
            )
        """)
        
        # Insert data using register method
        facility_db.register("temp_facility_metrics", facility_metrics)
        facility_db.execute("""
            INSERT INTO facility_metrics 
            SELECT PROVNUM, PROVNAME, STATE, COUNTY_NAME, CY_QTR, Census, Total_Nurse_HPRD, Contract_Percentage, date 
            FROM temp_facility_metrics
        """)
        
        # Create indexes for faster lookups
        facility_db.execute("CREATE INDEX IF NOT EXISTS idx_provnum ON facility_metrics(PROVNUM)")
        facility_db.execute("CREATE INDEX IF NOT EXISTS idx_date ON facility_metrics(date)")
        facility_db.execute("CREATE INDEX IF NOT EXISTS idx_quarter ON facility_metrics(CY_QTR)")
        
        # Verify data was loaded
        result = facility_db.execute("SELECT COUNT(*) FROM facility_metrics").fetchone()
        print(f"Total records in facility_metrics table: {result[0]}")
        
    except Exception as e:
        print(f"Error creating facility database: {str(e)}")
        st.error(f"Error creating facility database: {str(e)}")

# Initialize data at startup
try:
    national_metrics, state_metrics, facility_metrics = load_metrics_data()
    create_facility_db()
except Exception as e:
    st.error(f"Error during initialization: {str(e)}")
    national_metrics = pd.DataFrame()
    state_metrics = pd.DataFrame()
    facility_metrics = pd.DataFrame()

def proper_title_case(text: str) -> str:
    """
    Convert text to proper title case, keeping words like 'and', 'of', etc. lowercase.
    """
    if not text:
        return text
        
    # Words that should remain lowercase unless they're the first word
    lowercase_words = {'and', 'of', 'the', 'in', 'at', 'for', 'to', 'with', 'by'}
    
    # Split the text and capitalize first letter of each word
    words = text.lower().split()
    
    # Always capitalize the first word
    if words:
        words[0] = words[0].capitalize()
    
    # Process remaining words
    for i in range(1, len(words)):
        if words[i] not in lowercase_words:
            words[i] = words[i].capitalize()
            
    return ' '.join(words)

@st.cache_data
def get_provider_info(provnum: str, info_type: str) -> str:
    """Get provider information with optimized caching."""
    try:
        # Always treat provnum as string
        provnum = str(provnum).strip()
        if provnum in provider_info_cache:
            value = provider_info_cache[provnum].get(info_type, 'N/A')
            # Apply proper title case to name
            if info_type == 'name':
                value = proper_title_case(value)
            return value
        
        # If not in cache, try to get from facility_metrics
        try:
            query = f"""
                SELECT DISTINCT PROVNAME, STATE, COUNTY_NAME
                FROM facility_metrics 
                WHERE PROVNUM = '{provnum}'
                LIMIT 1
            """
            result = facility_db.execute(query).fetchdf()
            if not result.empty:
                value = result.iloc[0][info_type.upper() if info_type != 'name' else 'PROVNAME']
                # Apply proper title case to name
                if info_type == 'name':
                    value = proper_title_case(value)
                # Cache the result
                provider_info_cache[provnum] = {
                    'name': str(result.iloc[0]['PROVNAME']).strip(),
                    'state': str(result.iloc[0]['STATE']).strip(),
                    'county': str(result.iloc[0]['COUNTY_NAME']).strip()
                }
                return value
        except Exception as e:
            st.error(f"Error getting provider info from facility_metrics: {str(e)}")
        
        return 'N/A'
    except Exception as e:
        st.error(f"Error getting provider info: {str(e)}")
        return 'N/A'

def get_provider_name(provnum):
    """Get provider name from PBJ files."""
    return get_provider_info(provnum, 'name')

def get_provider_state(provnum):
    """Get provider state from PBJ files."""
    return get_provider_info(provnum, 'state')

def get_provider_county(provnum):
    """Get provider county from PBJ files."""
    return get_provider_info(provnum, 'county')

@st.cache_data
def get_facility_staffing_rating(provnum: str) -> float:
    """Get the staffing rating for a specific facility from provider info data."""
    try:
        provider_data = load_provider_info_data()
        if provider_data.empty:
            return None
            
        # Find the facility by CCN
        facility_data = provider_data[provider_data['CMS Certification Number (CCN)'] == provnum]
        
        if facility_data.empty:
            return None
            
        # Get the staffing rating
        staffing_rating = facility_data.iloc[0]['Staffing Rating']
        
        # Return None if it's NaN, otherwise return the rating
        return float(staffing_rating) if pd.notna(staffing_rating) else None
        
    except Exception as e:
        print(f"Error getting staffing rating for {provnum}: {str(e)}")
        return None

@st.cache_data
def get_facility_affiliated_entity(provnum: str) -> str:
    """Get the affiliated entity for a specific facility from provider info data."""
    try:
        provider_data = load_provider_info_data()
        if provider_data.empty:
            return None
            
        # Find the facility by CCN
        facility_data = provider_data[provider_data['CMS Certification Number (CCN)'] == provnum]
        
        if facility_data.empty:
            return None
            
        # Get the affiliated entity
        affiliated_entity = facility_data.iloc[0]['Affiliated Entity Name']
        
        # Return None if it's NaN, otherwise return the entity name with proper title case
        if pd.notna(affiliated_entity):
            return proper_title_case(str(affiliated_entity))
        return None
        
    except Exception as e:
        print(f"Error getting affiliated entity for {provnum}: {str(e)}")
        return None

@st.cache_data
def get_facility_overall_rating(provnum: str) -> float:
    """Get the overall rating for a specific facility from provider info data."""
    try:
        provider_data = load_provider_info_data()
        if provider_data.empty:
            return None
            
        # Find the facility by CCN
        facility_data = provider_data[provider_data['CMS Certification Number (CCN)'] == provnum]
        
        if facility_data.empty:
            return None
            
        # Get the overall rating
        overall_rating = facility_data.iloc[0]['Overall Rating']
        
        # Return None if it's NaN, otherwise return the rating
        if pd.notna(overall_rating):
            return int(overall_rating)
        else:
            return None
    except Exception as e:
        st.error(f"Error getting overall rating: {str(e)}")
        return None

@st.cache_data
def get_facility_staffing_rating_trend(provnum: str) -> str:
    """Get the staffing rating trend by comparing current vs March 2025 data."""
    try:
        current_data = load_provider_info_data()
        march_data = load_march_provider_info_data()
        
        if current_data.empty or march_data.empty:
            return None
        
        # Find the facility in current data
        current_facility = current_data[current_data['CMS Certification Number (CCN)'] == provnum]
        march_facility = march_data[march_data['CMS Certification Number (CCN)'] == provnum]
        
        if current_facility.empty or march_facility.empty:
            return None
        
        current_rating = current_facility['Staffing Rating'].iloc[0]
        march_rating = march_facility['Staffing Rating'].iloc[0]
        
        if pd.isna(current_rating) or pd.isna(march_rating):
            return None
        
        # Convert to integers for star ratings
        current_rating = int(current_rating) if pd.notna(current_rating) else None
        march_rating = int(march_rating) if pd.notna(march_rating) else None
        
        if current_rating is None or march_rating is None:
            return None
        
        if current_rating > march_rating:
            return current_rating - march_rating
        elif current_rating < march_rating:
            return -(march_rating - current_rating)
        else:
            return None
        
    except Exception as e:
        return None

@st.cache_data
def get_facility_overall_rating_trend(provnum: str) -> str:
    """Get the overall rating trend by comparing current vs March 2025 data."""
    try:
        current_data = load_provider_info_data()
        march_data = load_march_provider_info_data()
        
        if current_data.empty or march_data.empty:
            return None
        
        # Find the facility in current data
        current_facility = current_data[current_data['CMS Certification Number (CCN)'] == provnum]
        march_facility = march_data[march_data['CMS Certification Number (CCN)'] == provnum]
        
        if current_facility.empty or march_facility.empty:
            return None
        
        current_rating = current_facility['Overall Rating'].iloc[0]
        march_rating = march_facility['Overall Rating'].iloc[0]
        
        if pd.isna(current_rating) or pd.isna(march_rating):
            return None
        
        # Convert to integers for star ratings
        current_rating = int(current_rating) if pd.notna(current_rating) else None
        march_rating = int(march_rating) if pd.notna(march_rating) else None
        
        if current_rating is None or march_rating is None:
            return None
        
        if current_rating > march_rating:
            return current_rating - march_rating
        elif current_rating < march_rating:
            return -(march_rating - current_rating)
        else:
            return None
        
    except Exception as e:
        return None

@st.cache_data
def get_filtered_data(level: str, selected_value: str, start_quarter: str, end_quarter: str):
    """Get filtered data with optimized filtering."""
    try:
        if level == "Facility" and selected_value:
            # Use DuckDB for facility-level data
            query = f"""
                SELECT * FROM facility_metrics 
                WHERE PROVNUM = '{selected_value}'
                AND CY_QTR >= '{start_quarter}'
                AND CY_QTR <= '{end_quarter}'
                ORDER BY date
            """
            return facility_db.execute(query).fetchdf()
        
        # For other levels, use existing code
        national_metrics, state_metrics, facility_metrics = load_metrics_data()
        
        if level == "National":
            return national_metrics[
                (national_metrics['CY_QTR'] >= start_quarter) & 
                (national_metrics['CY_QTR'] <= end_quarter)
            ]
        elif level == "State":
            if selected_value == 'All States':
                return state_metrics[
                    (state_metrics['CY_QTR'] >= start_quarter) & 
                    (state_metrics['CY_QTR'] <= end_quarter)
                ]
            else:
                return state_metrics[
                    (state_metrics['STATE'] == selected_value) & 
                    (state_metrics['CY_QTR'] >= start_quarter) & 
                    (state_metrics['CY_QTR'] <= end_quarter)
                ]
        
        return pd.DataFrame()  # Return empty DataFrame if no conditions match
        
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return pd.DataFrame()  # Return empty DataFrame on error

@st.cache_data
def search_facilities(search_term: str) -> List[Dict[str, str]]:
    """Search facilities with lazy loading and caching."""
    try:
        # Sanitize search term to prevent SQL injection
        search_term = search_term.replace("'", "''")
        
        print(f"Searching for facilities matching: {search_term}")
        
        # Get matching facilities using DuckDB with parameterized query
        query = """
            SELECT DISTINCT PROVNUM, PROVNAME, STATE, COUNTY_NAME
            FROM facility_metrics 
            WHERE PROVNUM LIKE ? OR PROVNAME LIKE ?
            LIMIT 50
        """
        matching_facilities = facility_db.execute(
            query, 
            (f"%{search_term}%", f"%{search_term}%")
        ).fetchdf()
        
        print(f"Found {len(matching_facilities)} matching facilities")
        
        if not matching_facilities.empty:
            # Apply proper title case to PROVNAME and CITY
            matching_facilities['PROVNAME'] = matching_facilities['PROVNAME'].apply(proper_title_case)
            return matching_facilities.to_dict('records')
            
        print("No matching facilities found")
        return []
        
    except Exception as e:
        print(f"Error searching facilities: {str(e)}")
        st.error(f"Error searching facilities: {str(e)}")
        return []

@st.cache_data
def get_facility_info(provnum: str) -> dict:
    """Get facility information from the database."""
    try:
        conn = get_db_connection()
        if not conn:
            return None
            
        query = """
            SELECT DISTINCT
                PROVNUM,
                PROVNAME,
                STATE,
                COUNTY_NAME
            FROM staffing 
            WHERE PROVNUM = ?
            ORDER BY WORKDATE DESC
            LIMIT 1
        """
        
        result = conn.execute(query, (provnum,)).fetchone()
        conn.close()
        
        if result:
            return {
                'ccn': result[0],
                'provider_name': result[1],
                'state': result[2],
                'county': result[3]
            }
        return None
    except Exception as e:
        print(f"Error getting facility info: {str(e)}")
        return None

def get_quarterly_metrics(provnum: str, quarter: str) -> dict:
    """Get quarterly metrics for a facility."""
    try:
        conn = get_db_connection()
        if not conn:
            return None
            
        query = """
            WITH daily_metrics AS (
                SELECT 
                    PROVNUM,
                    WORKDATE,
                    MDSCENSUS,
                    (HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN + HRS_CNA + HRS_NATRN + HRS_MEDAIDE) as total_hours,
                    (HRS_RNDON + HRS_RNADMIN + HRS_RN) as rn_hours,
                    (HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN) as nurse_care_hours
                FROM staffing 
                WHERE CY_QTR = ? AND PROVNUM = ?
            )
            SELECT
                SUM(MDSCENSUS) as total_resident_days,
                SUM(total_hours) as total_hours,
                SUM(rn_hours) as rn_hours,
                SUM(nurse_care_hours) as nurse_care_hours
            FROM daily_metrics
        """
        
        result = conn.execute(query, (quarter, provnum)).fetchone()
        conn.close()
        
        if result and result[0] is not None:  # Check if we have resident days
            return {
                'total_hours': result[1] / result[0] if result[0] > 0 else 0,
                'rn_hours': result[2] / result[0] if result[0] > 0 else 0,
                'nurse_care_hours': result[3] / result[0] if result[0] > 0 else 0
            }
        return None
    except Exception as e:
        print(f"Error getting quarterly metrics: {str(e)}")
        return None

def generate_report(provnum: str, selected_quarter: str) -> str:
    """Generate a comprehensive HTML report for the facility."""
    try:
        # Get facility info
        facility_info = get_facility_info(provnum)
        if not facility_info:
            return "<p>Facility not found.</p>"
            
        # Get metrics for the selected quarter
        metrics = get_quarterly_metrics(provnum, selected_quarter)
        if not metrics:
            return "<p>No data available for the selected quarter.</p>"
            
        # Generate the report HTML
        report_html = f"""
            <html>
            <head>
                <title>PBJ Staffing Report - {facility_info['provider_name']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ margin-bottom: 20px; }}
                    .metrics {{ margin-bottom: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ width: 100%; border-collapse: collapse; }}
                    th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
                    th {{ background-color: #f8f9fa; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>{facility_info['provider_name']}</h1>
                    <p>CCN: {facility_info['ccn']}</p>
                    <p>Location: {facility_info['county']}, {facility_info['state']}</p>
                </div>
                
                <div class="metrics">
                    <h2>Key Metrics - {selected_quarter}</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Staffing Hours</td>
                            <td>{metrics['total_hours']:.2f}</td>
                        </tr>
                        <tr>
                            <td>RN Hours</td>
                            <td>{metrics['rn_hours']:.2f}</td>
                        </tr>
                        <tr>
                            <td>Nurse Care Hours</td>
                            <td>{metrics['nurse_care_hours']:.2f}</td>
                        </tr>
                    </table>
                </div>
            </body>
            </html>
        """
        
        return report_html
    except Exception as e:
        return f"<p>Error generating report: {str(e)}</p>"

def format_quarter_display(q):
    """Convert internal quarter format (2017Q1) to display format (Q1 2017)"""
    year = q[:4]
    quarter = q[-1]
    return f"Q{quarter} {year}"

def normalize_quarter(q):
    """Convert any quarter format to internal format (YYYYQN)"""
    if isinstance(q, str):
        # Handle YYYYQN format (e.g., "2017Q1")
        if re.match(r'^\d{4}Q[1-4]$', q):
            return q
        # Handle QN YYYY format (e.g., "Q1 2017")
        match = re.match(r'^Q([1-4])\s*(\d{4})$', q)
        if match:
            quarter, year = match.groups()
            return f"{year}Q{quarter}"
    return q

def sort_quarters(quarters, reverse=False):
    """Sort quarters in chronological order"""
    normalized = [normalize_quarter(q) for q in quarters]
    return sorted(normalized, reverse=reverse)

def display_facility_info(provnum: str):
    """Display facility information in a formatted box."""
    try:
        # Get basic facility info
        facility_info = get_facility_info(provnum)
        if not facility_info:
            return

        # Add CSS to the page
        st.markdown("""
            <style>
            div.facility-info-box {
                background-color: #f8f9fa;
                padding: 16px 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 1px solid #e0e0e0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.03);
            }
            div.facility-info-grid {
                display: flex;
                flex-wrap: wrap;
                gap: 12px 24px;
                align-items: center;
            }
            div.facility-info-item {
                color: #555;
                font-size: 0.95em;
                line-height: 1.4;
                display: flex;
                align-items: center;
                gap: 5px;
            }
            div.facility-info-item:not(:last-child):after {
                content: '|';
                color: #ddd;
                margin-left: 24px;
            }
            div.facility-info-item strong {
                color: #2c3338;
                font-weight: 500;
            }
            div.facility-info-item a {
                color: #1E88E5;
                text-decoration: none;
                font-weight: 500;
                margin-left: auto;
            }
            div.facility-info-item a:hover {
                text-decoration: underline;
            }
            
            /* Mobile-specific styles */
            @media (max-width: 768px) {
                div.facility-info-box {
                    padding: 12px 16px;
                }
                div.facility-info-grid {
                    flex-direction: column;
                    gap: 8px;
                    align-items: flex-start;
                }
                div.facility-info-item {
                    width: 100%;
                    padding: 4px 0;
                }
                div.facility-info-item:not(:last-child):after {
                    display: none;
                }
                div.facility-info-item a {
                    margin-left: 0;
                    margin-top: 8px;
                    display: block;
                }
                /* Hide labels on mobile */
                div.facility-info-item span.label {
                    display: none;
                }
                /* Adjust spacing for mobile */
                div.facility-info-item {
                    margin-bottom: 4px;
                }
                /* Make text slightly larger on mobile */
                div.facility-info-item strong {
                    font-size: 1.1em;
                }
            }
            </style>
        """, unsafe_allow_html=True)

        # Format provider name and city with proper title case
        def format_title_case(text):
            if pd.isna(text):
                return 'N/A'
            words = text.split()
            formatted_words = []
            for i, word in enumerate(words):
                if i == 0 or word.lower() not in ['and', 'at', 'of', 'the', 'in', 'on', 'for', 'to', 'with', 'by']:
                    formatted_words.append(word.capitalize())
                else:
                    formatted_words.append(word.lower())
            return ' '.join(formatted_words)

        formatted_provider_name = format_title_case(facility_info['provider_name'])
        formatted_county = format_title_case(facility_info['county'])

        # Add subtle modern styling for metric containers only (not delta or value)
        st.markdown("""
            <style>
            div[data-testid="stMetric"] {
                background: #f7fafd;
                border: 1px solid #e3eaf3;
                border-radius: 10px;
                box-shadow: 0 1px 4px rgba(30,136,229,0.04);
                padding: 18px 10px 4px 10px;
                margin: 0 4px 10px 4px;
                max-width: 240px;
            }
            </style>
        """, unsafe_allow_html=True)

        # Add the facility information HTML
        st.markdown(f"""
            <div class="facility-info-box">
                <div class="facility-info-grid">
                    <div class="facility-info-item">
                        <span class="label">Provider:</span> <strong>{formatted_provider_name} ({facility_info['ccn']})</strong>
                    </div>
                    <div class="facility-info-item">
                        <span class="label">Location:</span> <strong>{formatted_county}, {facility_info['state']}</strong>
                    </div>
                    <div class="facility-info-item">
                        <a href="https://www.medicare.gov/care-compare/details/nursing-home/{facility_info['ccn']}?state={facility_info['state']}" target="_blank">Care Compare</a>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying facility info: {str(e)}")

def on_mobile_change():
    """Callback function to handle mobile detection state changes."""
    if st.session_state.get('is_mobile', False):
        st.session_state['view_mode'] = "Mobile View"
    else:
        st.session_state['view_mode'] = "Desktop View"

def display_subscription_button(entity_type: str, entity_id: str, entity_name: str):
    """Display the premium services section with email link."""
    st.markdown("""
        <style>
        .premium-services {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px auto;
            max-width: 800px;
            text-align: center;
            border: 1px solid #e0e0e0;
        }
        .premium-services h3 {
            color: #2c3338;
            margin-bottom: 15px;
        }
        .premium-services p {
            color: #555;
            margin-bottom: 15px;
        }
        .premium-services a {
            color: #1E88E5;
            text-decoration: none;
            font-weight: 500;
        }
        .premium-services a:hover {
            text-decoration: underline;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
        <div class="premium-services">
            <h3>Premium Services</h3>
            <p>320 Consulting offers custom reports with full breakdowns of all nurse and non-nurse positions, staffing trends over time, ownership data, citation histories, and comparisons by geography or any category you need — built to support your case, investigation, or advocacy.</p>
            <p>To request a report:</p>
            <p><a href="mailto:eric@320insight.com">📧 eric@320insight.com</a></p>
        </div>
    """, unsafe_allow_html=True)

def display_metrics(metrics: pd.DataFrame, level: str):
    """Display metrics with optimized calculations."""
    try:
        if metrics.empty:
            st.warning(f"No data available for the selected {level}.")
            return

        # Get all available quarters for this dataset
        available_quarters = sort_quarters(metrics['CY_QTR'].unique(), reverse=True)

        # Always use the most recent quarter (first in sorted list)
        current_quarter = available_quarters[0]
        current_quarter = normalize_quarter(current_quarter)
        year = current_quarter[:4]
        quarter_num = current_quarter[-1]
        quarter_name = f"Q{quarter_num} {year}"

        # Filter metrics for the current quarter
        current_metrics = metrics[metrics['CY_QTR'] == current_quarter]

        if current_metrics.empty:
            st.warning(f"No data available for {quarter_name}.")
            return

        # Get the previous quarter's data
        current_idx = available_quarters.index(current_quarter)
        prev_quarter = available_quarters[current_idx + 1] if current_idx + 1 < len(available_quarters) else None
        prev_metrics = metrics[metrics['CY_QTR'] == prev_quarter] if prev_quarter else pd.DataFrame()

        # Display Key Metrics header with level-specific title
        if level == "State":
            state = metrics['STATE'].iloc[0]
            facility_count = current_metrics['Facility_Count'].iloc[0] if 'Facility_Count' in current_metrics else len(current_metrics['PROVNUM'].unique())
            prev_facility_count = prev_metrics['Facility_Count'].iloc[0] if not prev_metrics.empty and 'Facility_Count' in prev_metrics else None
            full_state_name = get_full_state_name(state)
            header_text = f"{full_state_name} Key Metrics ({quarter_name})"
        elif level == "National":
            facility_count = current_metrics['Facility_Count'].iloc[0] if 'Facility_Count' in current_metrics else len(current_metrics['PROVNUM'].unique())
            prev_facility_count = prev_metrics['Facility_Count'].iloc[0] if not prev_metrics.empty and 'Facility_Count' in prev_metrics else None
            header_text = f"USA Key Metrics ({quarter_name})"
        else:  # Facility level
            provnum = current_metrics['PROVNUM'].iloc[0]
            provname = proper_title_case(current_metrics['PROVNAME'].iloc[0])
            state = current_metrics['STATE'].iloc[0]
            county = proper_title_case(current_metrics['COUNTY_NAME'].iloc[0])
            care_compare_url = f"https://www.medicare.gov/care-compare/details/nursing-home/{provnum}/view-all?state={state}"
            
            # Get affiliated entity for header
            affiliated_entity = get_facility_affiliated_entity(provnum)
            if affiliated_entity:
                header_text = f"<div style='display: flex; justify-content: space-between; align-items: center;'><span style='color:#222; font-weight:400;'>{provname} ({county}, {state}) | {quarter_name} | {affiliated_entity}</span> <a href='{care_compare_url}' target='_blank' style='background:#e8f4fd; color:#1976d2; border-radius:6px; padding:2px 10px; font-size:0.97em; text-decoration:none; font-weight:500;'>View on Care Compare</a></div>"
            else:
                header_text = f"<div style='display: flex; justify-content: space-between; align-items: center;'><span style='color:#222; font-weight:400;'>{provname} ({county}, {state}) | {quarter_name}</span> <a href='{care_compare_url}' target='_blank' style='background:#e8f4fd; color:#1976d2; border-radius:6px; padding:2px 10px; font-size:0.97em; text-decoration:none; font-weight:500;'>Care Compare</a></div>"
        st.markdown(f'''
            <div class="section-header" style="margin-top: 8px; font-size: 1.35em; font-weight: 700; color: #1976d2; border-bottom: 2.5px solid #e3eaf3; padding-bottom: 4px; letter-spacing: 0.01em;">
                {header_text}
            </div>
        ''', unsafe_allow_html=True)
            
        # Add custom CSS for metrics containers
        # (Removed custom CSS for stMetric, stMetricDelta, stMetricContainer to restore Streamlit defaults)
        
        # Display metrics in columns - now use 5 columns for facility level, 4 for others
        if level == "Facility":
            col1, col2, col3, col4, col5 = st.columns(5)
        else:
            col1, col2, col3, col4 = st.columns(4)
        
        # For National and State, add facility count metric
        if level in ["National", "State"]:
            with col1:
                st.metric("Nursing Homes", 
                         format_metric(facility_count, decimal_places=0, thousands=True),
                         format_metric(facility_count - prev_facility_count, decimal_places=0, thousands=True) if prev_facility_count is not None else None)
        
        # Adjust column indices for other metrics
        if level == "Facility":
            metric_cols = [col1, col2, col3, col4, col5]
        else:
            metric_cols = [col2, col3, col4]
        
        with metric_cols[0]:
            st.metric("Census", 
                     format_metric(current_metrics['Census'].iloc[0], decimal_places=0, thousands=True),
                     format_metric(current_metrics['Census'].iloc[0] - prev_metrics['Census'].iloc[0], decimal_places=0, thousands=True) if not prev_metrics.empty else None)
        
        with metric_cols[1]:
            st.metric("Total Nurse HPRD", 
                     format_metric(current_metrics['Total_Nurse_HPRD'].iloc[0], decimal_places=2),
                     format_metric(current_metrics['Total_Nurse_HPRD'].iloc[0] - prev_metrics['Total_Nurse_HPRD'].iloc[0], decimal_places=2) if not prev_metrics.empty else None,
                     help="Hours Per Resident Day")
        
        with metric_cols[2]:
            st.metric(
                "Contract Staff %",
                format_metric(current_metrics['Contract_Percentage'].iloc[0], decimal_places=1, percentage=True),
                format_metric(current_metrics['Contract_Percentage'].iloc[0] - prev_metrics['Contract_Percentage'].iloc[0], decimal_places=1, percentage=True) if not prev_metrics.empty else None,
                help="Percent of nursing hours provided by contract staff"
            )
        
        # Add staffing rating and overall rating for facility level
        if level == "Facility":
            provnum = current_metrics['PROVNUM'].iloc[0]
            staffing_rating = get_facility_staffing_rating(provnum)
            overall_rating = get_facility_overall_rating(provnum)
            staffing_trend = get_facility_staffing_rating_trend(provnum)
            overall_trend = get_facility_overall_rating_trend(provnum)
            
            with metric_cols[3]:
                if staffing_rating is not None:
                    st.metric("CMS Staffing Rating", 
                             f"{int(staffing_rating)}",
                             staffing_trend,
                             help="CMS 5-star rating (June 2025 vs. March 2025).")
                else:
                    st.metric("CMS Staffing Rating", 
                             "N/A",
                             staffing_trend,
                             help="CMS 5-star rating (June 2025 vs. March 2025).")
            
            with metric_cols[4]:
                if overall_rating is not None:
                    st.metric("CMS Overall Rating", 
                             f"{overall_rating}",
                             overall_trend,
                             help="CMS 5-star rating (June 2025 vs. March 2025).")
                else:
                    st.metric("CMS Overall Rating", 
                             "N/A",
                             overall_trend,
                             help="CMS 5-star rating (June 2025 vs. March 2025).")
            
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

def format_metric(value, decimal_places=1, percentage=False, thousands=False):
    """Format a metric value with appropriate decimal places and formatting."""
    if pd.isna(value):
        return "N/A"
    if percentage:
        return f"{value:.{decimal_places}f}%"
    if thousands:
        return f"{value:,.{decimal_places}f}"
    return f"{value:.{decimal_places}f}"

def plot_quarterly_trends(df: pd.DataFrame, state: str = None, facility: str = None):
    """Plot quarterly trends with optimized data processing."""
    try:
        data = df.sort_values('date')
        # Restore title_prefix logic
        if state:
            full_state_name = get_full_state_name(state)
            title_prefix = f"{full_state_name} Staffing Trends (2017-2024)"
        elif facility:
            facility_name = get_provider_info(facility, 'name')
            facility_state = get_provider_info(facility, 'state')
            if facility_name and facility_state:
                title_prefix = f"{facility_name}, {facility_state} (2017-2024)"
            else:
                title_prefix = f"Facility {facility} (2017-2024)"
        else:
            title_prefix = "National Staffing Trends (2017-2024)"
        
        # Sort data by date
        data = data.sort_values('date')
        
        # Create year labels for x-axis ticks
        min_year = data['date'].dt.year.min()
        max_year = data['date'].dt.year.max()
        all_years = range(min_year, max_year + 1)
        tick_values = [pd.Timestamp(f"{year}-01-01") for year in all_years]
        tick_text = [str(year) for year in all_years]
        
        # Get the actual date range from the data
        date_range = [data['date'].min(), data['date'].max()]
        
        # Define custom hover templates
        hover_hprd = "<b>%{customdata}</b><br>%{y:.2f} HPRD<extra></extra>"
        hover_census = "<b>%{customdata}</b><br>%{y:,.0f}<extra></extra>"
        hover_contract = "<b>%{customdata}</b><br>%{y:.2f}%<extra></extra>"
        
        # Desktop figure (3 charts)
        if state:
            full_state_name = get_full_state_name(state)
            fig = make_subplots(rows=3, cols=1,
                  subplot_titles=(f'Total Nurse HPRD - {full_state_name}', f'Census - {full_state_name}', f'Contract Staff Percentage - {full_state_name}'),
                          vertical_spacing=0.15)
        elif facility:
            fig = make_subplots(rows=3, cols=1,
                  subplot_titles=('Total Nurse HPRD', 'Census', 'Contract Staff Percentage'),
                          vertical_spacing=0.15)
        else:
            fig = make_subplots(rows=3, cols=1,
                  subplot_titles=('Total Nurse HPRD - National', 'Census - National', 'Contract Staff Percentage - National'),
                          vertical_spacing=0.15)

        # Add all traces for desktop view
        fig.add_trace(go.Scatter(x=data['date'], y=data['Total_Nurse_HPRD'],
                       mode='lines+markers', name='Total HPRD',
                       customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                       hovertemplate=hover_hprd), row=1, col=1)

        fig.add_trace(go.Scatter(x=data['date'], y=data['Census'],
                       mode='lines+markers', name='Census',
                       customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                       hovertemplate=hover_census), row=2, col=1)

        fig.add_trace(go.Scatter(x=data['date'], y=data['Contract_Percentage'],
                       mode='lines+markers', name='Contract %',
                       customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                       hovertemplate=hover_contract), row=3, col=1)

        # Update desktop layout
        fig.update_layout(
            height=1400,
            width=1000,
            title_text=title_prefix,
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=100),
            hovermode='x unified'
        )
        
        # Add footer annotations for desktop view
        for row in range(1, 4):
            fig.add_annotation(
                text="320 Consulting | Source: CMS PBJ Data (2017-2024)",
                x=0.99,
                y=-0.25,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                font=dict(size=10, color="gray"),
                align="right",
                row=row,
                col=1
            )
        
        # Update desktop x-axes
        for row in range(1, 4):
            fig.update_xaxes(
                tickvals=tick_values,
                tickangle=45,
                row=row,
                col=1,
                showline=True,
                linewidth=1,
                linecolor="rgba(200, 200, 200, 0.1)",
                range=date_range,
                nticks=len(tick_values) // 2 if len(tick_values) > 4 else len(tick_values),
                tickmode='auto'
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting trends: {str(e)}")
        return None

def display_footer():
    """Display a consistent footer across all pages."""
    st.markdown("""
        <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
            <p>Source: <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" style="color: #1E88E5; text-decoration: none;">CMS Payroll-Based Journal Data, 2017-2024</a></p>
            <p>By <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">320 Consulting LLC</a></p>
        </div>
    """, unsafe_allow_html=True)

def main() -> None:
    """Main app layout and data flow."""
    try:
        # Initialize session state variables at the very start
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = "Desktop"

        # Get current page from URL
        current_page = st.query_params.get('page', 'dashboard')

        # Handle different pages
        if current_page == 'premium':
            st.switch_page("pages/1_Premium.py")
        elif current_page == 'facility_search':
            st.switch_page("pages/1_Facility_Search.py")
        elif current_page == 'affiliated_entities':
            st.switch_page("pages/2_Affiliated_Entities.py")

        # Get URL parameters using the new API
        initial_level = st.query_params.get('level', 'National')
        initial_facility = st.query_params.get('facility', None)

        # Title with custom styling
        st.markdown("""
            <div>
                <h1 class="main-header" style="margin-bottom: 0;">Nursing Home Staffing Dashboard</h1>
            </div>
        """, unsafe_allow_html=True)

        # Refined subhead: left-aligned, slightly wider
        st.markdown('''
            <div style="background: #f7fafd; border-radius: 6px; padding: 14px 14px 8px 14px; margin-bottom: 14px; border: 1px solid #e3eaf3; max-width: 850px; margin-left: 0;">
                <div style="font-size: 1.08em; color: #234; font-weight: 600; margin-bottom: 2px;">
                    A free public resource from <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 700;"><b>320 Consulting</b></a>, featuring quarterly staffing data (2017–2024) across every U.S. facility.                 <div class="toggle-tip-mobile" style="font-size:0.89em; color:#5a6473; font-style: italic; margin-bottom: 8px; font-weight: 400;">Toggle &gt;&gt; icon on top left to navigate dashboard.</div>
            </div>
        ''', unsafe_allow_html=True)
        # Sidebar
        st.sidebar.markdown("""
            <style>
            .sidebar-filters {
                margin-bottom: 20px;
            }
            .quarter-selectors {
                display: flex;
                gap: 10px;
                margin-bottom: 15px;
            }
            .quarter-selectors > div {
                flex: 1;
            }
            .sidebar .stSelectbox {
                margin-bottom: 0;
            }
            .sidebar h3 {
                margin-bottom: 0.5rem;
            }
            /* Add styles for quarter selectors */
            .sidebar .stSelectbox {
                width: 150px !important;
            }
            /* Match width of provider name box to CCN box */
            .sidebar div[data-testid="stSelectbox"] {
                width: 100% !important;
            }
            /* Remove top margin from first element in sidebar */
            .sidebar > div:first-child {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            /* Reduce spacing between elements */
            .sidebar .stRadio {
                margin-top: 0 !important;
                margin-bottom: 1rem !important;
            }
            .sidebar .stSelectbox {
                margin-top: 0 !important;
                margin-bottom: 1rem !important;
            }
            .sidebar-attribution {
                font-size: 0.8em;
                color: #666;
                margin-top: 0.5rem;
                margin-bottom: 0.5rem;
            }
            .sidebar-methodology {
                font-size: 0.8em;
                color: #666;
                font-style: italic;
                margin-top: 0.5rem;
                margin-bottom: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)

        # Sidebar: Only one radio button group for level selection, always present
        level = st.sidebar.radio(
            "Select Level",
            ["National", "State", "Facility", "Ownership"],
            index=["National", "State", "Facility", "Ownership"].index(initial_level) if initial_level in ["National", "State", "Facility", "Affiliated Entities"] else 0,
            key="level_selector"
        )

        # A & B: Fix radio button navigation
        if level == "Ownership":
            st.switch_page("pages/3_Ownership.py")
        elif level == "Facility Search":
            st.switch_page("pages/2_Facility_Search.py")

        # Get selected value based on level
        selected_value = None
        try:
            if level == "State":
                states = ["Select a state..."] + sorted(state_metrics['STATE'].unique().tolist())
                selected_state = st.sidebar.selectbox(
                    "Select State",
                    states,
                    index=0
                )
                selected_value = selected_state if selected_state != "Select a state..." else None
            elif level == "Facility":
                search_container = st.sidebar.container()
                
                # If we have a facility from URL, use it
                if initial_facility:
                    # Set the selected value directly from the CCN
                    selected_value = initial_facility
                    # Use the CCN as the search term
                    search_term = initial_facility
                else:
                    search_term = ""

                # Create the search input without using session state for the value
                search_term = search_container.text_input(
                    "Enter Provider CCN or Name",
                    value=search_term,
                    key="facility_search"
                )
                
                # Add help text with hyperlink
                st.sidebar.markdown(
                    '<div style="margin-top: -15px; margin-bottom: 15px;">'
                    '<a href="/Facility_Search" target="_self" style="color: #1E88E5; text-decoration: none; font-size: 0.9em;">'
                    'Help finding facility data</a></div>',
                    unsafe_allow_html=True
                )
                
                matching_facilities = []
                search_triggered = False

                if st.session_state.view_mode == "Mobile":
                    # On mobile, add a search button
                    if search_container.button("Search", key="facility_search_button"):
                        search_triggered = True
                    if search_triggered and search_term:
                        matching_facilities = search_facilities(search_term)
                else:
                    # On desktop, search as you type
                    if search_term:
                        matching_facilities = search_facilities(search_term)

                if search_term:
                    if matching_facilities:
                        facility_options = [
                            f"{fac['PROVNAME']}"
                            for fac in matching_facilities
                        ]
                        selected_facility_display = search_container.selectbox(
                            "Select Facility",
                            facility_options,
                            key="facility_selector",
                            label_visibility="collapsed"
                        )
                        if selected_facility_display:
                            # Find the matching facility to get the CCN
                            selected_facility = next(
                                (fac for fac in matching_facilities if fac['PROVNAME'] == selected_facility_display),
                                None
                            )
                            if selected_facility:
                                selected_value = selected_facility['PROVNUM']
                    else:
                        search_container.info("No matching facilities found")

            # Add back button if on facility page - now both level and selected_value are defined
            if level == "Facility" and selected_value:
                st.markdown("""
                    <div style="margin-bottom: 10px;">
                        <a href="/" target="_self" style="color: #1E88E5; text-decoration: none; font-weight: 500;">
                            ← Back to PBJ Dashboard
                        </a>
                    </div>
                """, unsafe_allow_html=True)

            # Get all available quarters
            try:
                all_quarters = sort_quarters(national_metrics['CY_QTR'].unique())  # Oldest to newest
                start_quarter = all_quarters[0]  # First quarter
                end_quarter = all_quarters[-1]   # Last quarter
            except Exception as e:
                st.error(f"Error loading quarters: {str(e)}")
                return
        except Exception as e:
            st.error(f"Error processing selection: {str(e)}")
            return

        # Get filtered data
        try:
            filtered_data = get_filtered_data(level, selected_value, start_quarter, end_quarter)
            
            # For facility level, add the info box and other elements in the new order
            if level == "Facility" and selected_value:
                # Get selected facility details
                selected_facility = next((fac for fac in matching_facilities if fac['PROVNUM'] == selected_value), None)
                if selected_facility:
                    # 1. Display facility info box
                    display_facility_info(selected_value)
                    
                    # 2. Display metrics
                    display_metrics(filtered_data, level)
                    
                    # 3. Display trends
                    fig = plot_quarterly_trends(filtered_data, 
                                          state=selected_value if level == "State" else None,
                                        facility=selected_value if level == "Facility" else None)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # 4. Add subscription button
                    display_subscription_button("facility", selected_value, selected_facility['PROVNAME'])

            # For other levels (National, State)
            else:
                if not filtered_data.empty:
                    display_metrics(filtered_data, level)
                    fig = plot_quarterly_trends(filtered_data, 
                                              state=selected_value if level == "State" else None,
                                        facility=selected_value if level == "Facility" else None)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add subscription button for all levels
                    if level == "National":
                        display_subscription_button("national", "national", "National Data")
                    elif level == "State":
                        display_subscription_button("state", selected_value, f"{selected_value} State Data")
                else:
                    st.warning("No data available for the selected filters.")
        except Exception as e:
            st.error(f"Error filtering data: {str(e)}")
            return
    except Exception as e:
        st.error(f"Error in main app: {str(e)}")

    # Add footer at the end of the page
    display_footer()

def get_db_connection():
    """Get a connection to the DuckDB database."""
    db_file = "nursing_home_staffing.db"
    if not os.path.exists(db_file):
        print(f"Database file {db_file} not found")
        return None
    try:
        return duckdb.connect(db_file)
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")
        return None

def query_nurse_staffing(provnum: str, start_date: str, end_date: str, staff_category: str) -> pd.DataFrame:
    """Query nurse staffing data for a specific facility and date range."""
    try:
        # Convert dates to datetime
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Get the quarters we need to check
        quarters = []
        current = start_dt
        while current <= end_dt:
            quarter = f"{current.year}Q{(current.month-1)//3 + 1}"
            if quarter not in quarters:
                quarters.append(quarter)
            current += pd.DateOffset(months=1)
        
        # Load data from each quarter file
        dfs = []
        for quarter in quarters:
            file_path = f'standardized_PBJ/PBJ_dailynursestaffing_CY{quarter}.csv'
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
            
        # Combine all quarters
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Filter for the specific facility and date range
        mask = (
            (combined_df['PROVNUM'] == provnum) &
            (pd.to_datetime(combined_df['WorkDate']) >= start_dt) &
            (pd.to_datetime(combined_df['WorkDate']) <= end_dt)
        )
        filtered_df = combined_df[mask].copy()
        
        if filtered_df.empty:
            return pd.DataFrame()
            
        # Convert WorkDate to datetime and add day of week
        filtered_df['WorkDate'] = pd.to_datetime(filtered_df['WorkDate'])
        filtered_df['DayOfWeek'] = filtered_df['WorkDate'].dt.day_name()
        
        # Select relevant columns based on staff category
        if staff_category == 'RN':
            hours_cols = ['Hrs_RN', 'Hrs_RN_emp', 'Hrs_RN_ctr']
        elif staff_category == 'LPN':
            hours_cols = ['Hrs_LPN', 'Hrs_LPN_emp', 'Hrs_LPN_ctr']
        elif staff_category == 'CNA':
            hours_cols = ['Hrs_CNA', 'Hrs_CNA_emp', 'Hrs_CNA_ctr']
        elif staff_category == 'Nurse Aide Trainee':
            hours_cols = ['Hrs_NAtrn', 'Hrs_NAtrn_emp', 'Hrs_NAtrn_ctr']
        elif staff_category == 'Medical Aide':
            hours_cols = ['Hrs_MedAide', 'Hrs_MedAide_emp', 'Hrs_MedAide_ctr']
        elif staff_category == 'RN Administrator':
            hours_cols = ['Hrs_RNadmin', 'Hrs_RNadmin_emp', 'Hrs_RNadmin_ctr']
        elif staff_category == 'LPN Administrator':
            hours_cols = ['Hrs_LPNadmin', 'Hrs_LPNadmin_emp', 'Hrs_LPNadmin_ctr']
        elif staff_category == 'RN Director of Nursing':
            hours_cols = ['Hrs_RNDON', 'Hrs_RNDON_emp', 'Hrs_RNDON_ctr']
        else:
            return pd.DataFrame()
            
        # Select only the columns we need
        result_df = filtered_df[['WorkDate', 'DayOfWeek', 'MDScensus'] + hours_cols].copy()
        
        # Rename columns for clarity
        result_df.rename(columns={
            'WorkDate': 'Date',
            'MDScensus': 'Census',
            hours_cols[0]: 'Total Hours',
            hours_cols[1]: 'Employee Hours',
            hours_cols[2]: 'Contract Hours'
        }, inplace=True)
        
        return result_df
        
    except Exception as e:
        print(f"Error querying nurse staffing data: {str(e)}")
        return pd.DataFrame()

if __name__ == "__main__":
    main()