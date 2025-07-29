import streamlit as st
import pandas as pd
import numpy as np

# Fix for NumPy compatibility with Plotly
# np.bool8 was deprecated and removed in NumPy 1.26+
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import duckdb
import os
import re
from plotly.subplots import make_subplots
from typing import Dict, Optional, List, Tuple, Any

# Set page config
st.set_page_config(
    page_title="PBJ Nursing Home Staffing Dashboard by 320", 
    page_icon="pbj_favicon.png", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

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

# Add CSS to hide the toggle tip on desktop and mobile-responsive title
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

/* Mobile-responsive title styling */
@media (max-width: 768px) {
    .dashboard-title {
        font-size: 2.1em !important;
        line-height: 1.2 !important;
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

        # Standardize column names - apply specific mappings to each dataframe
        # National metrics column mapping - only rename CY_Qtr to CY_QTR and MDS to Census
        national_column_mapping = {
            'CY_Qtr': 'CY_QTR',
            'MDS': 'Census'  # Map MDS to Census for national metrics
        }
        national_metrics.rename(columns=national_column_mapping, inplace=True)
        
        # State metrics column mapping - only rename CY_Qtr to CY_QTR
        state_column_mapping = {
            'CY_Qtr': 'CY_QTR'
        }
        state_metrics.rename(columns=state_column_mapping, inplace=True)
        
        # Ensure STATE column exists and is properly named
        if 'STATE' not in state_metrics.columns:
            # Try to find a similar column
            state_cols = [col for col in state_metrics.columns if 'state' in col.lower()]
            if state_cols:
                # Rename the first matching column to STATE
                state_metrics.rename(columns={state_cols[0]: 'STATE'}, inplace=True)
        
        # Facility metrics column mapping - only rename CY_Qtr to CY_QTR
        facility_column_mapping = {
            'CY_Qtr': 'CY_QTR'
        }
        facility_metrics.rename(columns=facility_column_mapping, inplace=True)

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
        
        # Convert Affiliated entity ID to numeric
        if 'Affiliated entity ID' in df.columns:
            df['Affiliated entity ID'] = pd.to_numeric(df['Affiliated entity ID'], errors='coerce')
        
        return df
    except Exception as e:
        st.error(f"Error loading affiliated entity data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def load_provider_info_data():
    """Load and cache provider information data."""
    try:
        df = pd.read_csv('NH_ProviderInfo_Jun2025.csv', dtype={'CMS Certification Number (CCN)': str})
        
        # Rename columns to match expected format
        column_mapping = {
            'CMS Certification Number (CCN)': 'PROVNUM',
            'Provider Name': 'PROVNAME',
            'State': 'STATE',
            'County/Parish': 'COUNTY_NAME'
        }
        df.rename(columns=column_mapping, inplace=True)
        
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
        
        # Rename columns to match expected format
        column_mapping = {
            'CMS Certification Number (CCN)': 'PROVNUM',
            'Provider Name': 'PROVNAME',
            'State': 'STATE',
            'County/Parish': 'COUNTY_NAME'
        }
        df.rename(columns=column_mapping, inplace=True)
        
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
def get_facility_affiliated_entity_id(provnum: str) -> str:
    """Get the affiliated entity ID for a specific facility from provider info data."""
    try:
        provider_data = load_provider_info_data()
        if provider_data.empty:
            return None
            
        # Find the facility by CCN
        facility_data = provider_data[provider_data['CMS Certification Number (CCN)'] == provnum]
        
        if facility_data.empty:
            return None
            
        # Get the affiliated entity ID
        affiliated_entity_id = facility_data.iloc[0]['Affiliated Entity ID']
        
        # Return None if it's NaN, otherwise return the entity ID
        if pd.notna(affiliated_entity_id):
            return str(int(affiliated_entity_id))
        return None
        
    except Exception as e:
        print(f"Error getting affiliated entity ID for {provnum}: {str(e)}")
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
                # Debug: Check available columns
                if 'STATE' not in state_metrics.columns:
                    st.error(f"STATE column not found. Available columns: {list(state_metrics.columns)}")
                    # Try to find and rename state column
                    state_cols = [col for col in state_metrics.columns if 'state' in col.lower()]
                    if state_cols:
                        state_metrics.rename(columns={state_cols[0]: 'STATE'}, inplace=True)
                        st.success(f"Renamed {state_cols[0]} to STATE")
                    else:
                        return pd.DataFrame()
                
                # Additional debug info
                print(f"Filtering state data for: {selected_value}")
                print(f"Available states: {sorted(state_metrics['STATE'].unique())}")
                
                return state_metrics[
                    (state_metrics['STATE'] == selected_value) & 
                    (state_metrics['CY_QTR'] >= start_quarter) & 
                    (state_metrics['CY_QTR'] <= end_quarter)
                ]
        elif level == "Entity":
            # For entities, return empty DataFrame since we'll handle entity display separately
            # Entity data is static, not time-series
            return pd.DataFrame()
        
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

def display_facility_info(provnum: str, quarter_name: str = None, affiliated_entity: str = None):
    """Display facility information in a formatted box. On mobile, remove ownership entity and show quarter below provider name."""
    try:
        # Add back to search button with styling
        st.markdown("""
        <style>
        .back-button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 16px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        .back-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }
        .back-button:active {
            transform: translateY(0);
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<button class="back-button" onclick="window.location.href=\'/PBJ_Dashboard\'">← Back to Search</button>', unsafe_allow_html=True)
        
        facility_info = get_facility_info(provnum)
        if not facility_info:
            return

        # Detect mobile
        is_mobile = st.session_state.get('is_mobile', False)

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
        ccn = facility_info['ccn']
        state = facility_info['state']
        care_compare_url = f"https://www.medicare.gov/care-compare/details/nursing-home/{ccn}?state={state}"

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
                div.facility-info-item span.label {
                    display: none;
                }
                div.facility-info-item {
                    margin-bottom: 4px;
                }
                div.facility-info-item strong {
                    font-size: 1.1em;
                }
                .facility-quarter-row {
                    font-size: 1.08em;
                    color: #1976d2;
                    font-weight: 600;
                    margin-top: 2px;
                    margin-bottom: 2px;
                }
            }
            </style>
        """, unsafe_allow_html=True)

        # Mobile: no ownership entity, quarter on its own row
        if is_mobile:
            st.markdown(f"""
                <div class="facility-info-box">
                    <div class="facility-info-grid">
                        <div class="facility-info-item">
                            <span class="label">Provider:</span> <strong>{formatted_provider_name} ({ccn})</strong>
                        </div>
                        <div class="facility-info-item">
                            <span class="label">Location:</span> <strong>{formatted_county}, {state}</strong>
                        </div>
                        <div class="facility-quarter-row">{quarter_name if quarter_name else ''}</div>
                        <div class="facility-info-item">
                            <a href="{care_compare_url}" target="_blank">CMS Care Compare</a> <span title="CMS Care Compare is a federal resource providing comprehensive information on U.S. nursing homes, including quality ratings, staffing data, and inspection results." style="cursor: help; color: #666; font-size: 0.8em; font-weight: bold;">?</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Desktop: show ownership entity if present, quarter inline
            st.markdown(f"""
                <div class="facility-info-box">
                    <div class="facility-info-grid">
                        <div class="facility-info-item">
                            <span class="label">Provider:</span> <strong>{formatted_provider_name} ({ccn})</strong>
                        </div>
                        <div class="facility-info-item">
                            <span class="label">Location:</span> <strong>{formatted_county}, {state}</strong>
                        </div>
                        <div class="facility-info-item">
                            <span class="label">Quarter:</span> <strong>{quarter_name if quarter_name else ''}</strong>
                        </div>
                        {f'<div class="facility-info-item"><span class="label">Ownership:</span> <strong>{affiliated_entity}</strong></div>' if affiliated_entity else ''}
                        <div class="facility-info-item">
                            <a href="{care_compare_url}" target="_blank">CMS Care Compare</a> <span title="CMS Care Compare is a federal resource providing comprehensive information on U.S. nursing homes, including quality ratings, staffing data, and inspection results." style="cursor: help; color: #666; font-size: 0.8em; font-weight: bold;">?</span>
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

def create_custom_metric(label, value, help_text=None, trend=None):
    """Create a custom metric display with consistent styling."""
    metric_html = f'''
    <div style="background: white; border: 1px solid #e0e0e0; border-radius: 8px; padding: 1rem; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <div style="font-size: 0.9em; color: #666; margin-bottom: 0.5rem; font-weight: 500;">{label}</div>
        <div style="font-size: 1.8em; font-weight: 700; color: #1a2233; margin-bottom: 0.2rem;">{value}</div>
    </div>
    '''
    return metric_html

def create_narrow_metric(label, value):
    """Create a narrower metric display for side-by-side layouts."""
    metric_html = f'''
    <div style="background: white; border: 1px solid #e0e0e0; border-radius: 6px; padding: 0.8rem; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.05); margin-bottom: 0.5rem;">
        <div style="font-size: 0.85em; color: #666; margin-bottom: 0.3rem; font-weight: 500;">{label}</div>
        <div style="font-size: 1.4em; font-weight: 700; color: #1a2233;">{value}</div>
    </div>
    '''
    return metric_html

def is_mobile():
    """Check if the current viewport is mobile-sized."""
    # This is a simplified check - in a real app you'd use JavaScript
    return False

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
        .nav-links {
            text-align: center;
            margin: 10px auto 5px auto;
            padding: 10px 0;
            border-top: 1px solid #e0e0e0;
        }
        .nav-links a {
            color: #1769aa;
            text-decoration: none;
            font-weight: 600;
            font-size: 0.95em;
            margin: 0 10px;
            transition: all 0.2s ease;
        }
        .nav-links a:hover {
            color: #0d47a1;
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
    
    # Add navigation links below premium services
    st.markdown("""
        <div class="nav-links">
            <a href="/About" target="_self">About the Dashboard</a> | 
            <a href="/Premium" target="_self">Premium</a>
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
            if 'STATE' in metrics.columns and not metrics.empty:
                state = metrics['STATE'].iloc[0]
            else:
                st.error("No data available for the selected state or 'STATE' column missing.")
                return
            facility_count = current_metrics['Facility_Count'].iloc[0] if 'Facility_Count' in current_metrics else len(current_metrics['PROVNUM'].unique())
            prev_facility_count = prev_metrics['Facility_Count'].iloc[0] if not prev_metrics.empty and 'Facility_Count' in prev_metrics else None
            full_state_name = get_full_state_name(state)
            header_text = f"{full_state_name} Key Metrics ({quarter_name})"
            # Display state header with back button
            st.markdown("""
            <style>
            .back-button-small {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 600;
                cursor: pointer;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
                transition: all 0.3s ease;
                margin-bottom: 10px;
            }
            .back-button-small:hover {
                transform: translateY(-1px);
                box-shadow: 0 4px 10px rgba(0,0,0,0.25);
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<button class="back-button-small" onclick="window.location.href=\'/PBJ_Dashboard\'">← Back to Search</button>', unsafe_allow_html=True)
            
            st.markdown(f'''
                <div class="section-header" style="margin-top: 8px; font-size: 1.35em; font-weight: 700; color: #1976d2; border-bottom: 2.5px solid #e3eaf3; padding-bottom: 4px; letter-spacing: 0.01em;">
                    <div style='font-size: 1.35em; font-weight: 700; color: #1976d2;'>{header_text}</div>
                </div>
            ''', unsafe_allow_html=True)
        elif level == "National":
            facility_count = current_metrics['Facility_Count'].iloc[0] if 'Facility_Count' in current_metrics else len(current_metrics['PROVNUM'].unique())
            prev_facility_count = prev_metrics['Facility_Count'].iloc[0] if not prev_metrics.empty and 'Facility_Count' in prev_metrics else None
            header_text = f"USA Key Metrics ({quarter_name})"
            # Display national header
            st.markdown(f'''
                <div class="section-header" style="margin-top: 8px; font-size: 1.35em; font-weight: 700; color: #1976d2; border-bottom: 2.5px solid #e3eaf3; padding-bottom: 4px; letter-spacing: 0.01em;">
                    <div style='font-size: 1.35em; font-weight: 700; color: #1976d2;'>{header_text}</div>
                </div>
            ''', unsafe_allow_html=True)
        else:  # Facility level
            provnum = current_metrics['PROVNUM'].iloc[0]
            provname = proper_title_case(current_metrics['PROVNAME'].iloc[0])
            state = current_metrics['STATE'].iloc[0]
            county = proper_title_case(current_metrics['COUNTY_NAME'].iloc[0])
            care_compare_url = f"https://www.medicare.gov/care-compare/details/nursing-home/{provnum}/view-all?state={state}"
            
            # Get affiliated entity and entity ID for header
            affiliated_entity = get_facility_affiliated_entity(provnum)
            affiliated_entity_id = get_facility_affiliated_entity_id(provnum)
            full_state_name = get_full_state_name(state)
            
            # Check if the affiliated entity exists in the current dataset
            entity_data = load_affiliated_entity_data()
            entity_exists = False
            if entity_data is not None and not entity_data.empty:
                entity_exists = entity_data[
                    (entity_data['Affiliated entity'] == affiliated_entity) | 
                    (entity_data['Affiliated entity ID'].astype(str) == str(affiliated_entity_id))
                ].shape[0] > 0
            
            if affiliated_entity and affiliated_entity_id and entity_exists:
                st.markdown(f'''
                    <div class="section-header" style="margin-top: 8px; font-size: 1.35em; font-weight: 700; color: #1976d2; border-bottom: 2.5px solid #e3eaf3; padding-bottom: 4px; letter-spacing: 0.01em;">
                        <div style='color:#222; font-weight:400;'>
                            <div style='font-size: 1.35em; font-weight: 700; color: #1976d2;'>{provname} ({quarter_name})</div>
                            <div style='font-size: 0.9em; color: #666; margin-top: 4px;'>
                                {county}, <a href='?level=State&state={state}' style='color: #1976d2; text-decoration: none;' target='_self'>{state}</a>. Ownership: <a href='?level=Entity&entity={affiliated_entity_id}' style='color: #1976d2; text-decoration: none;' target='_self'>{affiliated_entity}</a>
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            elif affiliated_entity and affiliated_entity_id:
                st.markdown(f'''
                    <div class="section-header" style="margin-top: 8px; font-size: 1.35em; font-weight: 700; color: #1976d2; border-bottom: 2.5px solid #e3eaf3; padding-bottom: 4px; letter-spacing: 0.01em;">
                        <div style='color:#222; font-weight:400;'>
                            <div style='font-size: 1.35em; font-weight: 700; color: #1976d2;'>{provname} ({quarter_name})</div>
                            <div style='font-size: 0.9em; color: #666; margin-top: 4px;'>
                                {county}, <a href='?level=State&state={state}' style='color: #1976d2; text-decoration: none;' target='_self'>{state}</a>. Ownership: {affiliated_entity} (not in current dataset)
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="section-header" style="margin-top: 8px; font-size: 1.35em; font-weight: 700; color: #1976d2; border-bottom: 2.5px solid #e3eaf3; padding-bottom: 4px; letter-spacing: 0.01em;">
                        <div style='color:#222; font-weight:400;'>
                            <div style='font-size: 1.35em; font-weight: 700; color: #1976d2;'>{provname} ({quarter_name})</div>
                            <div style='font-size: 0.9em; color: #666; margin-top: 4px;'>
                                {county}, <a href='?level=State&state={state}' style='color: #1976d2; text-decoration: none;' target='_self'>{state}</a>. Ownership: N/A
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            
        # Add custom CSS for metrics containers
        # (Removed custom CSS for stMetric, stMetricDelta, stMetricContainer to restore Streamlit defaults)
        
        # Display metrics in columns - use 4 columns for all levels
        col1, col2, col3, col4 = st.columns(4)
        
        # For National and State, add facility count metric
        if level in ["National", "State"]:
            with col1:
                st.metric("Nursing Homes", 
                         format_metric(facility_count, decimal_places=0, thousands=True),
                         format_metric(facility_count - prev_facility_count, decimal_places=0, thousands=True) if prev_facility_count is not None else None,
                        help="Total number of nursing homes during the reporting period. Arrow compares to previous quarter.")
            # Adjust column indices for other metrics
            metric_cols = [col2, col3, col4]
        else:
            # For facility level, use all 4 columns
            metric_cols = [col1, col2, col3, col4]
        
        with metric_cols[0]:
            # Use State_Census for state level, Census for facility level
            if level == "State":
                census_value = current_metrics['State_Census'].iloc[0]
                prev_census_value = prev_metrics['State_Census'].iloc[0] if not prev_metrics.empty else None
                help_text = "Total number of residents across all facilities in the state during the reporting period. Arrow compares to previous quarter."
            else:
                census_value = current_metrics['Census'].iloc[0]
                prev_census_value = prev_metrics['Census'].iloc[0] if not prev_metrics.empty else None
                help_text = "Average number of residents in facility during the reporting period. Arrow compares to previous quarter."
            
            st.metric("Resident Census", 
                     format_metric(census_value, decimal_places=0, thousands=True),
                     format_metric(census_value - prev_census_value, decimal_places=0, thousands=True) if prev_census_value is not None else None,
                     help=help_text)
        
        with metric_cols[1]:
            st.metric("Nurse Staffing (HPRD)", 
                     format_metric(current_metrics['Total_Nurse_HPRD'].iloc[0], decimal_places=2),
                     format_metric(current_metrics['Total_Nurse_HPRD'].iloc[0] - prev_metrics['Total_Nurse_HPRD'].iloc[0], decimal_places=2) if not prev_metrics.empty else None,
                     help="Total nurse staff hours per resident per day. Example: A nursing home with 100 residents providing 350 staffing hours per day has 3.5 nurse staff HPRD (350 ÷ 100). Arrow compares to previous quarter.")
        
        with metric_cols[2]:
            st.metric(
                "Contract Staff %",
                format_metric(current_metrics['Contract_Percentage'].iloc[0], decimal_places=1, percentage=True),
                format_metric(current_metrics['Contract_Percentage'].iloc[0] - prev_metrics['Contract_Percentage'].iloc[0], decimal_places=1, percentage=True) if not prev_metrics.empty else None,
                help="Percent of nursing hours provided by contract staff. Arrow compares to previous quarter."
            )
        
        # Add staffing rating for facility level
        if level == "Facility":
            provnum = current_metrics['PROVNUM'].iloc[0]
            staffing_rating = get_facility_staffing_rating(provnum)
            staffing_trend = get_facility_staffing_rating_trend(provnum)
            
            with metric_cols[3]:
                if staffing_rating is not None:
                    st.metric("CMS Staffing Rating", 
                             f"{int(staffing_rating)}",
                             staffing_trend,
                             help="5-star rating determined by federal CMS (June 2025 vs. March 2025).")
                else:
                    st.metric("CMS Staffing Rating", 
                             "N/A",
                             staffing_trend,
                             help="5-star rating determined by federal CMS (June 2025 vs. March 2025).")
            

            
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
        <div style="text-align: center; margin-top: 10px; color: #666; font-size: 0.9em;">
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
        
        # Theme detection using new Streamlit 1.46+ feature
        theme = st.context.theme
        is_dark_mode = theme == "dark"
        
        # Simple mobile detection for warning message only
        # Check if we're on a mobile device using screen width
        if 'is_mobile' not in st.session_state:
            # Default to desktop
            st.session_state.is_mobile = False
            
        # Add JavaScript to detect mobile for warning message
        st.markdown("""
        <script>
        (function() {
            const isMobile = window.innerWidth <= 768;
            if (isMobile && !window.location.search.includes('mobile=true')) {
                // Add mobile parameter to URL
                const url = new URL(window.location);
                url.searchParams.set('mobile', 'true');
                window.history.replaceState({}, '', url);
                window.location.reload();
            }
        })();
        </script>
        """, unsafe_allow_html=True)
        
        # Check URL parameter for mobile detection
        if st.query_params.get('mobile') == 'true':
            st.session_state.is_mobile = True

        # Get current page from URL
        current_page = st.query_params.get('page', 'dashboard')

        # Handle different pages - removed problematic navigation

        # Get URL parameters using the new API
        initial_level = st.query_params.get('level', 'National')
        initial_facility = st.query_params.get('facility', None)
        initial_state = st.query_params.get('state', None)
        initial_entity = st.query_params.get('entity', None)
        
        # Title with theme-aware styling
        title_color = "#1769aa" if not is_dark_mode else "#4fc3f7"
        
        st.markdown(f"""
            <div style='text-align: center; margin-top: -40px; margin-bottom: 1.2em;'>
                <span class="dashboard-title" style='font-size:2.8em; font-weight:800; color:{title_color}; letter-spacing:0.01em; line-height:1.1;'>PBJ Nursing Home Staffing Dashboard</span>
            </div>
        """, unsafe_allow_html=True)

        # Check if we should hide search based on URL parameters
        hide_search = False
        if initial_level and (initial_facility or initial_state or initial_entity):
            hide_search = True

        # Refined subhead with theme-aware styling and improved layout
        subhead_bg = "#f7fafd" if not is_dark_mode else "#1e1e1e"
        subhead_border = "#e3eaf3" if not is_dark_mode else "#404040"
        subhead_text = "#234" if not is_dark_mode else "#e0e0e0"
        link_color = "#1E88E5" if not is_dark_mode else "#4fc3f7"
        
        st.markdown(f'''
            <div style="background: {subhead_bg}; border-radius: 6px; padding: 14px 14px 8px 14px; margin-bottom: 10px; border: 1px solid {subhead_border}; max-width: 950px; margin-left: auto; margin-right: auto; text-align: center;">
                <div style="font-size: 1.08em; color: {subhead_text}; font-weight: 600; margin-bottom: 2px;">
                    A free public resource from <a href="https://www.320insight.com/" target="_blank" style="color: {link_color}; text-decoration: none; font-weight: 700;"><b>320 Consulting</b></a>, featuring quarterly staffing data (2017–2024) across every U.S. nursing home.
                </div>
                {"<div style=\"margin-top: 8px;\" class=\"mobile-about-link\"><a href=\"/About\" target=\"_self\" style=\"color: {link_color}; text-decoration: none; font-size: 0.9em; font-weight: 400;\">About the PBJ Dashboard</a></div>" if not hide_search else ""}
            </div>
        ''', unsafe_allow_html=True)
        
        # Add About link only on homepage and only on desktop
        if not hide_search:
            st.markdown('''
                <style>
                .mobile-about-link {
                    display: block;
                }
                @media (min-width: 768px) {
                    .mobile-about-link {
                        display: none;
                    }
                }
                .desktop-about-link {
                    display: none;
                }
                @media (min-width: 768px) {
                    .desktop-about-link {
                        display: block;
                    }
                }
                </style>
                <div class="desktop-about-link" style="text-align: center; margin-bottom: 2px;">
                    <a href="/About" target="_self" style="color: #1769aa; text-decoration: none; font-size: 0.9em; font-weight: 600; background: #e8f4fd; padding: 3px 12px; border-radius: 6px; border: 1px solid #1976d2; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; transition: all 0.2s ease;">
                        About the PBJ Dashboard
                    </a>
                </div>
            ''', unsafe_allow_html=True)
        
        # Load search data functions (always available)
        @st.cache_data
        def load_facility_data():
            """Load facility data for search."""
            try:
                return pd.read_csv('facility_lite_metrics.csv', dtype={'PROVNUM': str})
            except Exception as e:
                st.error(f"Error loading facility data: {str(e)}")
                return pd.DataFrame()
        
        @st.cache_data
        def load_provider_info_data():
            """Load provider info data."""
            try:
                df = pd.read_csv('NH_ProviderInfo_Jun2025.csv', dtype={'CMS Certification Number (CCN)': str})
                
                # Rename columns to match expected format
                column_mapping = {
                    'CMS Certification Number (CCN)': 'PROVNUM',
                    'Provider Name': 'PROVNAME',
                    'State': 'STATE',
                    'County/Parish': 'COUNTY_NAME'
                }
                df.rename(columns=column_mapping, inplace=True)
                
                return df
            except Exception as e:
                st.error(f"Error loading provider info: {str(e)}")
                return pd.DataFrame()
        
        @st.cache_data
        def load_ownership_data():
            """Load ownership data."""
            try:
                return pd.read_csv('Nursing_Home_Affiliated_Entity_Performance_Measures_Jun_2025.csv')
            except Exception as e:
                st.error(f"Error loading ownership data: {str(e)}")
                return pd.DataFrame()
        
        def proper_title_case(text):
            """Convert text to proper title case."""
            if pd.isna(text):
                return ""
            return text.title()
        
        def smart_title(name: str) -> str:
            """Convert facility name to smart title case."""
            if pd.isna(name):
                return ""
            name = str(name).strip()
            if not name:
                return ""
            
            # Convert to title case first
            name = name.title()
            
            # Handle common abbreviations and terms properly
            name = name.replace(" Llc", " LLC").replace(" Inc", " INC").replace(" Lp", " LP")
            name = name.replace(" Nh", " NH")
            name = name.replace(" Rehab", " Rehab").replace(" Rehabilitation", " Rehabilitation")
            name = name.replace(" Center", " Center").replace(" Facility", " Facility")
            name = name.replace(" Nursing Home", " Nursing Home")
            
            # Handle common words that should be lowercase
            name = name.replace(" Of ", " of ").replace(" At ", " at ").replace(" The ", " the ")
            name = name.replace(" And ", " and ").replace(" Or ", " or ").replace(" In ", " in ")
            name = name.replace(" On ", " on ").replace(" To ", " to ").replace(" For ", " for ")
            
            return name

        # Add search functionality
        if not hide_search:
            st.markdown("""
                <style>
                .search-header {
                    margin-bottom: 5px;
                    margin-top: -10px;
                }
                .stTabs [data-baseweb="tab-list"] {
                    margin-top: -5px;
                }
                @media (max-width: 768px) {
                    .search-header {
                        margin-top: -30px;
                        margin-bottom: 0px;
                    }
                    div[data-testid="stMarkdown"] > div:has(> div[style*="background: #f7fafd"]) {
                        margin-bottom: 4px !important;
                    }
                }
                </style>
            """, unsafe_allow_html=True)
            
            st.markdown('<h3 class="search-header">Search PBJ Data</h3>', unsafe_allow_html=True)
            
            # Load data for search
            facilities_df = load_facility_data()
            provider_info_df = load_provider_info_data()
            ownership_df = load_ownership_data()
            
            # Create search tabs with improved width control
            tab1, tab2, tab3 = st.tabs(["🔍 Facility", "🏢 Ownership", "🗺️ State"])
            
            # Set width for better mobile layout
            if st.session_state.get('is_mobile', False):
                st.markdown("""
                    <style>
                    .stTabs [data-baseweb="tab-list"] {
                        width: 100% !important;
                    }
                    .stTabs [data-baseweb="tab"] {
                        width: 33.33% !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
            
            with tab1:
                # Use responsive columns with width control for mobile-friendly layout
                if st.session_state.get('is_mobile', False):
                    col1, col2 = st.columns([1, 1], gap="small")
                else:
                    col1, col2 = st.columns(2)
                
                with col1:
                    # Use provider info data for state filtering since facilities_df doesn't have STATE column
                    state_filter = st.selectbox(
                        "Filter by State (Optional)",
                        [""] + sorted(provider_info_df['STATE'].unique().tolist()),
                        key="facility_state_filter",
                        help="Enter two letter state abbreviation"
                    )
                
                with col2:
                    # Create filtered search options based on selected state
                    if state_filter:
                        # Filter facilities by state using provider info
                        state_providers = provider_info_df[provider_info_df['STATE'] == state_filter]['PROVNUM'].tolist()
                        state_facilities = facilities_df[facilities_df['PROVNUM'].isin(state_providers)]
                        search_options = [f"{smart_title(row['PROVNAME'])} ({row['PROVNUM']})" for _, row in state_facilities[['PROVNAME', 'PROVNUM']].drop_duplicates().iterrows()]
                    else:
                        state_facilities = facilities_df  # Use full dataset when no state filter
                        search_options = [f"{smart_title(row['PROVNAME'])} ({row['PROVNUM']})" for _, row in facilities_df[['PROVNAME', 'PROVNUM']].drop_duplicates().iterrows()]

                    # Sort options alphabetically by facility name
                    search_options.sort()
                    
                    facility_search = st.selectbox(
                        "Enter Provider Name or CCN (6-digit ID).",
                        options=[""] + search_options,
                        key="facility_search_input",
                        help="Type to search facilities. Find CCN at https://data.cms.gov/provider-data/dataset/4pq5-n9py"
                    )
                
                # Display facility search results
                if facility_search:
                    # Use state_facilities which is now properly defined
                    search_data = state_facilities
                    
                    # Extract CCN from search term if it's in the format "Name (CCN)"
                    if '(' in facility_search:
                        ccn = facility_search.split('(')[-1].strip(')')
                        results = search_data[search_data['PROVNUM'] == ccn]
                    else:
                        facility_search = facility_search.lower()
                        results = search_data[
                            search_data['PROVNUM'].str.lower().str.contains(facility_search) |
                            search_data['PROVNAME'].str.lower().str.contains(facility_search)
                        ]
                    
                    if not results.empty:
                        # Get unique facilities with state info from provider_info_df
                        unique_results = results[['PROVNUM', 'PROVNAME']].drop_duplicates()
                        
                        # Add state information from provider_info_df
                        unique_results = unique_results.merge(
                            provider_info_df[['PROVNUM', 'STATE']], 
                            on='PROVNUM', 
                            how='left'
                        )
                        
                        # Create display DataFrame
                        display_df = pd.DataFrame()
                        display_df['State'] = unique_results['STATE']
                        display_df['Nursing Home (CCN)'] = unique_results['PROVNAME'].apply(smart_title) + ' (' + unique_results['PROVNUM'] + ')'
                        display_df['Dashboard'] = unique_results['PROVNUM'].apply(
                            lambda x: f'<a href="/?level=Facility&facility={x}" style="color: #1976d2; text-decoration: none; font-weight: bold;" target="_self">View</a>'
                        )
                        
                        # Sort alphabetically
                        display_df = display_df.sort_values('Nursing Home (CCN)')
                        
                        st.markdown("#### Search Results")
                        st.markdown("""
                            <style>
                            /* Target the specific table structure */
                            div[data-testid="stMarkdown"] table th:nth-child(2),
                            div[data-testid="stMarkdown"] table td:nth-child(2) {
                                text-align: left !important;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    else:
                        st.info("No facilities found matching your search criteria.")
                
                # Show all facilities for selected state (even without search)
                elif state_filter:
                    # Get all facilities for the selected state using provider info
                    state_providers = provider_info_df[provider_info_df['STATE'] == state_filter]['PROVNUM'].tolist()
                    state_facilities_all = facilities_df[facilities_df['PROVNUM'].isin(state_providers)]
                    
                    if not state_facilities_all.empty:
                        # Get unique facilities with state info from provider_info_df
                        unique_facilities = state_facilities_all[['PROVNUM', 'PROVNAME']].drop_duplicates()
                        
                        # Add state information from provider_info_df
                        unique_facilities = unique_facilities.merge(
                            provider_info_df[['PROVNUM', 'STATE']], 
                            on='PROVNUM', 
                            how='left'
                        )
                        
                        # Create display DataFrame
                        display_df = pd.DataFrame()
                        display_df['State'] = unique_facilities['STATE']
                        display_df['Nursing Home (CCN)'] = unique_facilities['PROVNAME'].apply(smart_title) + ' (' + unique_facilities['PROVNUM'] + ')'
                        display_df['Dashboard'] = unique_facilities['PROVNUM'].apply(
                            lambda x: f'<a href="/?level=Facility&facility={x}" style="color: #1976d2; text-decoration: none; font-weight: bold;" target="_self">View</a>'
                        )
                        
                        # Sort alphabetically
                        display_df = display_df.sort_values('Nursing Home (CCN)')
                        
                        # Map state abbreviation to full name
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
                        full_state_name = state_name_map.get(state_filter, state_filter)
                        
                        st.markdown(f"#### All Facilities in {full_state_name}")
                        st.markdown(f"*{len(display_df)} facilities found*")
                        st.markdown("""
                            <style>
                            /* Target the specific table structure */
                            div[data-testid="stMarkdown"] table th:nth-child(2),
                            div[data-testid="stMarkdown"] table td:nth-child(2) {
                                text-align: left !important;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                    else:
                        st.info("No facilities found for this state.")
            
            with tab2:
                if not ownership_df.empty:
                    # Filter to show only major ownership groups (you can customize this list)
                    major_ownership_groups = [
                        "Genesis Healthcare",
                        "Life Care Centers of America", 
                        "HCR ManorCare",
                        "Kindred Healthcare",
                        "Sun Healthcare Group",
                        "Golden Living",
                        "Extendicare",
                        "Brookdale Senior Living",
                        "Five Star Senior Living",
                        "Diversicare Healthcare Services"
                    ]
                    
                    # Get ownership entities that match major groups or have significant facility counts
                    ownership_entities = ownership_df[ownership_df['Affiliated entity'].notna()].copy()
                    
                    # Filter to major groups or those with 10+ facilities
                    major_entities = ownership_entities[
                        (ownership_entities['Affiliated entity'].isin(major_ownership_groups)) |
                        (ownership_entities['Number of facilities'] >= 10)
                    ].copy()
                    
                    # Remove National from the results
                    major_entities = major_entities[major_entities['Affiliated entity'] != 'National'].copy()
                    
                    # Create display names with entity ID stored in session state
                    ownership_options = [""]
                    for _, row in major_entities.iterrows():
                        display_name = smart_title(row['Affiliated entity']) + f" ({int(row['Number of facilities'])} NHs)"
                        ownership_options.append(display_name)
                        # Store entity ID in session state for later use, only if it's not NaN
                        if pd.notna(row['Affiliated entity ID']):
                            st.session_state[f"entity_{display_name}"] = int(row['Affiliated entity ID'])
                            st.session_state[f"name_{display_name}"] = row['Affiliated entity']
                    
                    ownership_search_display = st.selectbox(
                        "Select Ownership (Affiliated Entity)",
                        options=ownership_options,
                        key="ownership_search_input",
                        help="Select ownership (affiliated entity) to view their dashboard"
                    )
                    
                    if ownership_search_display:
                        # Get the stored entity ID and name
                        entity_id = st.session_state.get(f"entity_{ownership_search_display}")
                        ownership_name = st.session_state.get(f"name_{ownership_search_display}")
                        
                        if entity_id and ownership_name:
                            # Create styled button link - navigate to ownership page directly
                            st.markdown(f"""
                            <div style="text-align: center; margin: 20px 0;">
                                <a href="/?level=Entity&entity={entity_id}" 
                                   style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                          color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; 
                                          font-weight: 600; font-size: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
                                          transition: all 0.3s ease;">
                                    View {smart_title(ownership_name)} Dashboard
                                </a>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.info("Ownership group not found.")
                else:
                    st.info("Ownership data not available.")
            
            with tab3:
                # Load state data for dropdown
                state_metrics_df = pd.read_csv('state_lite_metrics.csv')
                state_search = st.selectbox(
                    "Select State",
                    ["", "USA"] + sorted(state_metrics_df['STATE'].unique().tolist()),
                    key="state_search_input",
                    help="Enter two letter state abbreviation"
                )
                
                if state_search:
                    # Map state abbreviation to full name
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
                    full_state_name = state_name_map.get(state_search, state_search)
                    
                    # Create styled button link - USA goes to main dashboard, others go to state dashboard
                    if state_search == "USA":
                        link_url = "/"
                        button_text = "View USA Dashboard"
                    else:
                        link_url = f"/?level=State&state={state_search}"
                        button_text = f"View {full_state_name} Dashboard"
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin: 20px 0;">
                        <a href="{link_url}" 
                           style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                  color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; 
                                  font-weight: 600; font-size: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
                                  transition: all 0.3s ease;">
                            {button_text}
                        </a>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Set default level and selected_value based on URL parameters
        level = initial_level if initial_level else "National"
        selected_value = None
        
        # Set selected_value based on URL parameters
        if initial_facility:
            selected_value = initial_facility
        elif initial_state:
            selected_value = initial_state
        elif initial_entity:
            selected_value = initial_entity

        # Get all available quarters
        try:
            all_quarters = sort_quarters(national_metrics['CY_QTR'].unique())  # Oldest to newest
            start_quarter = all_quarters[0]  # First quarter
            end_quarter = all_quarters[-1]   # Last quarter
        except Exception as e:
            st.error(f"Error loading quarters: {str(e)}")
            return

        # Get filtered data
        try:
            filtered_data = get_filtered_data(level, selected_value, start_quarter, end_quarter)
            
            # For facility level, add the info box and other elements in the new order
            if level == "Facility" and selected_value:
                # Get selected facility details from search results
                matching_facilities = search_facilities(selected_value) if selected_value else []
                selected_facility = next((fac for fac in matching_facilities if fac['PROVNUM'] == selected_value), None)
                if selected_facility:
                    # Compute current quarter label for facility info box
                    if not filtered_data.empty and 'CY_QTR' in filtered_data.columns:
                        available_quarters = sort_quarters(filtered_data['CY_QTR'].unique(), reverse=True)
                        current_quarter = available_quarters[0]
                        current_quarter = normalize_quarter(current_quarter)
                        year = current_quarter[:4]
                        quarter_num = current_quarter[-1]
                        quarter_label = f"Q{quarter_num} {year}"
                    else:
                        quarter_label = ""
                    
                    # 1. Display facility info box
                    display_facility_info(selected_value, quarter_name=quarter_label, affiliated_entity=get_facility_affiliated_entity(selected_value))
                    
                    # 2. Display metrics
                    display_metrics(filtered_data, level)
                    
                    # 3. Display trends
                    fig = plot_quarterly_trends(filtered_data, 
                                          state=selected_value if level == "State" else None,
                                        facility=selected_value if level == "Facility" else None)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add CMS Care Compare link below the chart for facility level
                        if level == "Facility":
                            care_compare_url = f"https://www.medicare.gov/care-compare/details/nursing-home/{selected_value}/view-all?state={selected_facility['STATE']}"
                            st.markdown(f"<div style='text-align: center; margin-top: 15px;'><a href='{care_compare_url}' target='_blank' style='background: #e8f4fd; color: #1976d2; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-weight: 500; border: 1px solid #1976d2; display: inline-block;'>View Details on CMS Care Compare</a></div>", unsafe_allow_html=True)

                    # 4. Add subscription button
                    display_subscription_button("facility", selected_value, selected_facility['PROVNAME'])

            # For entity level, display full entity content (verbatim from ownership page)
            elif level == "Entity" and selected_value:
                # Load data
                entity_data = load_affiliated_entity_data()
                provider_data = load_provider_info_data()
                
                if not entity_data.empty and not provider_data.empty:
                    # Find the selected entity by name or ID
                    # First try to find by name (more reliable since entity IDs may not match between datasets)
                    selected_entity_data = entity_data[
                        (entity_data['Affiliated entity'] == selected_value) | 
                        (entity_data['Affiliated entity ID'] == float(selected_value))
                    ]
                    
                    # Debug: Check if entity 237 exists
                    if selected_value == '237':
                        st.write(f"Debug: Looking for entity 237")
                        st.write(f"Debug: Entity 237 in data: {237.0 in entity_data['Affiliated entity ID'].values}")
                        st.write(f"Debug: Entity 237 as float: {float(237) in entity_data['Affiliated entity ID'].values}")
                        st.write(f"Debug: Sample entity IDs: {entity_data['Affiliated entity ID'].head().tolist()}")
                    
                    # If not found by ID, try to find by name from facility data
                    if selected_entity_data.empty and selected_value.isdigit():
                        # Get the entity name from facility data
                        provider_data = load_provider_info_data()
                        if not provider_data.empty:
                            # Find facilities with this entity ID
                            matching_facilities = provider_data[provider_data['Affiliated Entity ID'] == int(selected_value)]
                            if not matching_facilities.empty:
                                entity_name = matching_facilities.iloc[0]['Affiliated Entity Name']
                                if pd.notna(entity_name):
                                    # Search by name in the performance dataset
                                    selected_entity_data = entity_data[entity_data['Affiliated entity'] == entity_name]
                                    
                                    # If still not found, try partial name matching
                                    if selected_entity_data.empty:
                                        # Try to find by partial name match
                                        for idx, row in entity_data.iterrows():
                                            if entity_name.lower() in row['Affiliated entity'].lower() or row['Affiliated entity'].lower() in entity_name.lower():
                                                selected_entity_data = entity_data.iloc[[idx]]
                                                break
                    
                    if selected_entity_data.empty:
                        st.error(f"Entity '{selected_value}' not found in the data.")
                        
                        # Debug: Show what we were looking for
                        if selected_value.isdigit():
                            provider_data = load_provider_info_data()
                            if not provider_data.empty:
                                matching_facilities = provider_data[provider_data['Affiliated Entity ID'] == int(selected_value)]
                                if not matching_facilities.empty:
                                    entity_name = matching_facilities.iloc[0]['Affiliated Entity Name']
                                    st.info(f"Looking for entity ID {selected_value} which corresponds to '{entity_name}' in facility data")
                        
                        st.info("Available entities in performance dataset:")
                        
                        # Show first 20 available entities
                        available_entities = entity_data[['Affiliated entity', 'Affiliated entity ID']].dropna(subset=['Affiliated entity ID']).head(20)
                        for _, row in available_entities.iterrows():
                            st.write(f"- {row['Affiliated entity']} (ID: {row['Affiliated entity ID']})")
                        
                        st.button("← Back to Search", key="back_to_search_entity_not_found", on_click=lambda: st.switch_page("PBJ_Dashboard.py"))
                        return
                    if not selected_entity_data.empty:
                        entity_row = selected_entity_data.iloc[0]
                        entity_id = int(entity_row['Affiliated entity ID'])
                        
                        # Main entity dashboard with entity ID
                        entity_name_title_case = proper_title_case(selected_value)
                        st.markdown(f'''
                                <div class="entity-header" style="background: linear-gradient(90deg, #e3ecfa 80%, #dbeafe 100%); color: #1a2233; padding: 0.5rem 2rem; border-radius: 12px; margin-bottom: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #d3dbe8;">
                                    <h2 style="margin-bottom: 0.15em; font-size: 2.2em; font-weight: 700; letter-spacing: 0.01em; color: #1a2233;">{entity_name_title_case} <span style="font-size: 0.7em; font-weight: 400; color: #4b5563;">(ID: {entity_id})</span></h2>                </div>
                            ''', unsafe_allow_html=True)
                        # Key metrics overview using standard Streamlit metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Facilities", 
                                     format_metric(entity_row['Number of facilities'], decimal_places=0, thousands=True),
                                     help="Total number of nursing homes owned by this entity")
                        with col2:
                            st.metric("States of Operation", 
                                     format_metric(entity_row['Number of states and territories with operations'], decimal_places=0),
                                     help="Number of states where this entity operates nursing homes")
                        with col3:
                            st.metric("Overall Rating", 
                                     format_metric(entity_row['Average overall 5-star rating'], decimal_places=1),
                                     help="Average CMS 5-star overall rating across all facilities")
                        with col4:
                            st.metric("Total Fines", 
                                     f"${format_metric(entity_row['Total amount of fines in dollars'], decimal_places=0, thousands=True)}",
                                     help="Total amount of fines in dollars across all facilities")
                        # Calculate number of 1-star facilities for this entity
                        num_1star = 0
                        if entity_id and entity_id != "":
                            entity_facilities = provider_data[
                                provider_data['Affiliated Entity ID'] == entity_id
                            ].copy()
                            if not entity_facilities.empty:
                                num_1star = (entity_facilities['Overall Rating'] == 1).sum()

                        # High Risk Facilities
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em; margin-bottom:0.2em;">High-Risk Facilities - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                        with risk_col1:
                            st.metric("SFF", 
                                     format_metric(entity_row['Number of Special Focus Facilities (SFF)'], decimal_places=0),
                                     help="Special Focus Facilities with serious quality issues under CMS oversight")
                        with risk_col2:
                            st.metric("SFF Candidate", 
                                     format_metric(entity_row['Number of SFF candidates'], decimal_places=0),
                                     help="Facilities monitored for potential SFF designation")
                        with risk_col3:
                            st.metric("Abuse Icon", 
                                     format_metric(entity_row['Number of facilities with an abuse icon'], decimal_places=0),
                                     help="Facilities cited for abuse")
                        with risk_col4:
                            st.metric("1-Star Rating", 
                                     format_metric(num_1star, decimal_places=0),
                                     help="Facilities with the lowest CMS overall rating")
                        

                        
                        # Single column layout for detailed metrics
                        
                        # Ownership breakdown with pie chart
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Ownership Type - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        
                        own_col1, own_col2 = st.columns([1, 1])
                        
                        with own_col1:
                            # Pie chart for ownership
                            def safe_pct(val):
                                try:
                                    v = float(val)
                                    return v if pd.notna(v) else 0.0
                                except Exception:
                                    return 0.0
                            for_profit = safe_pct(entity_row.get('Percent of facilities classified as for-profit', 0))
                            non_profit = safe_pct(entity_row.get('Percent of facilities classified as non-profit', 0))
                            government = safe_pct(entity_row.get('Percent of facilities classified as government-owned', 0))
                            pie_labels = ['For-Profit', 'Non-Profit', 'Government']
                            pie_values = [for_profit, non_profit, government]
                            # Custom tooltip text for each slice
                            def format_hover_pct(value):
                                if pd.isna(value) or value == 0.0:
                                    return "N/A"
                                return f"{value:.1f}%"
                            
                            pie_hovertext = [
                                f'For-profit: {format_hover_pct(for_profit)}<br>Non-profit: {format_hover_pct(non_profit)}<br>Government: {format_hover_pct(government)}',
                                f'For-profit: {format_hover_pct(for_profit)}<br>Non-profit: {format_hover_pct(non_profit)}<br>Government: {format_hover_pct(government)}',
                                f'For-profit: {format_hover_pct(for_profit)}<br>Non-profit: {format_hover_pct(non_profit)}<br>Government: {format_hover_pct(government)}'
                            ]
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=pie_labels,
                                values=pie_values,
                                hole=0.3,
                                marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1'],
                                text=pie_hovertext,
                                hoverinfo='text',
                                textinfo='none',
                                showlegend=True
                            )])
                            fig_pie.update_layout(
                                height=260,
                                showlegend=True,
                                margin=dict(l=10, r=10, t=30, b=10)
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)
                        
                        with own_col2:
                            # Ownership metrics
                            def format_ownership_pct(value):
                                if pd.isna(value) or value is None:
                                    return "N/A"
                                return f"{value:.1f}%"
                            
                            st.metric("For-Profit", format_ownership_pct(entity_row['Percent of facilities classified as for-profit']))
                            st.metric("Non-Profit", format_ownership_pct(entity_row['Percent of facilities classified as non-profit']))
                            st.metric("Government", format_ownership_pct(entity_row['Percent of facilities classified as government-owned']))
                        
                        # CMS 5-Star Ratings
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">CMS 5-Star Ratings - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        # Quality metrics with decimals for entity averages
                        qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
                        with qual_col1:
                            st.metric("Overall", f"{entity_row['Average overall 5-star rating']:.1f}")
                        with qual_col2:
                            st.metric("Health Inspection", f"{entity_row['Average health inspection rating']:.1f}")
                        with qual_col3:
                            st.metric("Staffing", f"{entity_row['Average staffing rating']:.1f}")
                        with qual_col4:
                            st.metric("Quality", f"{entity_row['Average quality rating']:.1f}")
                        # Quality ratings chart and distribution chart side by side
                        chart_col1, chart_col2 = st.columns(2)
                        with chart_col1:
                            metrics = ['Overall', 'Staffing', 'Health Inspection', 'Quality']
                            values = [
                                entity_row['Average overall 5-star rating'],
                                entity_row['Average staffing rating'],
                                entity_row['Average health inspection rating'],
                                entity_row['Average quality rating']
                            ]
                            colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
                            # Each bar gets its own hovertemplate
                            bar_hovertemplates = [
                                'Overall: %{y:.1f}<extra></extra>',
                                'Staffing: %{y:.1f}<extra></extra>',
                                'Health Inspection: %{y:.1f}<extra></extra>',
                                'Quality: %{y:.1f}<extra></extra>'
                            ]
                            fig = go.Figure()
                            for i, (metric, value, color, hovertemplate) in enumerate(zip(metrics, values, colors, bar_hovertemplates)):
                                fig.add_trace(go.Bar(
                                    x=[metric],
                                    y=[value],
                                    marker_color=color,
                                    # Remove text labels from bars
                                    text=None,
                                    textposition=None,
                                    hovertemplate=hovertemplate,
                                    width=[0.5]
                                ))
                            fig.update_layout(
                                yaxis_title="CMS 5-Star Rating",
                                yaxis=dict(range=[0, 5], tickfont=dict(size=13)),
                                height=260,
                                showlegend=False,
                                margin=dict(l=10, r=10, t=30, b=10),
                                bargap=0.35
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        with chart_col2:
                            # Facility ratings breakdown pie chart
                            if entity_id and entity_id != "":
                                entity_facilities = provider_data[
                                    provider_data['Affiliated Entity ID'] == entity_id
                                ].copy()
                                if not entity_facilities.empty:
                                    # Count facilities by overall rating and ensure all ratings 1-5 are included
                                    rating_counts = entity_facilities['Overall Rating'].value_counts()
                                    # Create a complete series with all ratings 1-5, filling missing ones with 0
                                    complete_ratings = pd.Series(index=range(1, 6), data=0)
                                    for rating, count in rating_counts.items():
                                        if pd.notna(rating) and rating in range(1, 6):
                                            complete_ratings[rating] = count
                                    total_facilities = complete_ratings.sum()
                                    # Calculate percentages and filter out 0% slices
                                    rating_percents = [((count / total_facilities) * 100 if total_facilities > 0 else 0) for count in complete_ratings.values]
                                    # Only include slices where count > 0 (not just percent > 0)
                                    dist_labels = []
                                    dist_values = []
                                    dist_hovertext = []
                                    for rating, count, pct in zip(complete_ratings.index, complete_ratings.values, rating_percents):
                                        if count > 0:
                                            dist_labels.append(f"{rating}")
                                            dist_values.append(count)
                                            dist_hovertext.append(f'{rating} star: {pct:.1f}% ({count} NHs)')
                                    # Logical color scheme: 1=red, 2=orange, 3=yellow, 4=light green, 5=blue
                                    star_colors = ['#e74c3c', '#e67e22', '#f7dc6f', '#58d68d', '#3498db']
                                    # Only use as many colors as there are slices (always in 1-5 order)
                                    used_colors = [star_colors[int(rating)-1] for rating in dist_labels]
                                    fig_ratings = go.Figure(data=[go.Pie(
                                        labels=dist_labels,
                                        values=dist_values,
                                        hole=0.3,
                                        marker_colors=used_colors,
                                        sort=False,
                                        text=dist_hovertext,
                                        hoverinfo='text',
                                        textinfo='none',
                                        showlegend=True
                                    )])
                                    fig_ratings.update_layout(
                                        height=260,
                                        showlegend=True,
                                        margin=dict(l=10, r=10, t=30, b=10),
                                        legend=dict(
                                            orientation='v',
                                            x=1.05,
                                            y=0.5,
                                            xanchor='left',
                                            yanchor='middle',
                                            bgcolor='#f8f9fa',
                                            bordercolor='#e0e0e0',
                                            borderwidth=1,
                                            font=dict(size=13),
                                            itemclick='toggleothers',
                                            itemdoubleclick='toggle'
                                        )
                                    )
                                    title_col1, title_col2, title_col3 = st.columns([0.15, 0.7, 0.15])
                                    with title_col2:
                                        st.markdown('<div style="text-align:center; font-size:1em; font-weight:400; color:#444; margin-bottom:0.2em;">CMS 5-Star Rating Distribution</div>', unsafe_allow_html=True)
                                    st.plotly_chart(fig_ratings, use_container_width=True)
                        
                        # Staffing metrics
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Staffing Levels - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        
                        staff_col1, staff_col2, staff_col3, staff_col4 = st.columns(4)
                        with staff_col1:
                            st.metric("Total Nurse HPRD", f"{entity_row['Average total nurse hours per resident day']:.1f}")
                        with staff_col2:
                            st.metric("RN HPRD", f"{entity_row['Average total Registered Nurse hours per resident day']:.1f}")
                        with staff_col3:
                            st.metric("Weekend HPRD", f"{entity_row['Average total weekend nurse hours per resident day']:.1f}")
                        with staff_col4:
                            st.metric("Admin Turnover", f"{entity_row['Average number of administrators who have left the nursing home']:.1f}")
                        
                        # Turnover metrics
                        turn_col1, turn_col2 = st.columns(2)
                        with turn_col1:
                            def format_turnover_pct(value):
                                if pd.isna(value) or value is None:
                                    return "N/A"
                                return f"{value:.1f}%"
                            st.metric("Nursing Staff Turnover", format_turnover_pct(entity_row['Average total nursing staff turnover percentage']))
                        with turn_col2:
                            st.metric("RN Turnover", format_turnover_pct(entity_row['Average Registered Nurse turnover percentage']))
                        
                        # Compliance metrics
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Enforcement - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        
                        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                        with comp_col1:
                            st.metric("Total Fines", f"${entity_row['Total amount of fines in dollars']:,.0f}")
                        with comp_col2:
                            st.metric("Avg Fines per Facility", f"${entity_row['Average amount of fines in dollars']:,.0f}")
                        with comp_col3:
                            st.metric("Total Payment Denials", entity_row['Total number of payment denials'])
                        with comp_col4:
                            st.metric("Avg Payment Denials", f"{entity_row['Average number of payment denials']:.1f}")
                        
                        # Antipsychotic usage
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Antipsychotics - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        
                        anti_col1, anti_col2 = st.columns(2)
                        with anti_col1:
                            def format_antipsychotic_pct(value):
                                if pd.isna(value) or value is None:
                                    return "N/A"
                                return f"{value:.1f}%"
                            st.metric("Short-Stay Antipsychotic", format_antipsychotic_pct(entity_row['Average percentage of short-stay residents who newly received an antipsychotic medication']))
                        with anti_col2:
                            st.metric("Long-Stay Antipsychotic", format_antipsychotic_pct(entity_row['Average percentage of long-stay residents who received an antipsychotic medication']))
                        
                        # Facilities list
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Nursing homes affiliated with {selected_value}</h3></div>', unsafe_allow_html=True)
                        
                        # Get facilities for this entity
                        if entity_id and entity_id != "":
                            entity_facilities = provider_data[
                                provider_data['Affiliated Entity ID'] == entity_id
                            ].copy()
                            
                            if not entity_facilities.empty:
                                st.markdown(f"**{len(entity_facilities)} facilities found**")
                                
                                # Prepare facilities data for display with City instead of County
                                facilities_display = entity_facilities[[
                                    'State',
                                    'City/Town',
                                    'CMS Certification Number (CCN)',
                                    'Provider Name',
                                    'Overall Rating',
                                    'Staffing Rating',
                                    'Special Focus Status',
                                    'Abuse Icon'
                                ]].copy()
                                
                                # Rename City/Town to City
                                facilities_display = facilities_display.rename(columns={'City/Town': 'City'})
                                
                                # Special handling for Special Focus Status - replace NaN with "N"
                                facilities_display['Special Focus Status'] = facilities_display['Special Focus Status'].fillna('N')
                                
                                # Clean up the data and convert ratings to integers
                                facilities_display = facilities_display.fillna('N/A')
                                
                                # Convert numeric ratings to integers where possible
                                rating_columns = ['Overall Rating', 'Staffing Rating']
                                for col in rating_columns:
                                    facilities_display[col] = pd.to_numeric(facilities_display[col], errors='coerce')
                                    facilities_display[col] = facilities_display[col].apply(lambda x: int(x) if pd.notna(x) and x == int(x) else 'N/A')
                                
                                # Apply proper capitalization to provider names and city
                                def capitalize_name(name):
                                    if pd.isna(name):
                                        return name
                                    # Common words to keep lowercase
                                    lowercase_words = {'and', 'or', 'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
                                    words = name.lower().split()
                                    capitalized_words = []
                                    for i, word in enumerate(words):
                                        if i == 0 or word not in lowercase_words:
                                            capitalized_words.append(word.capitalize())
                                        else:
                                            capitalized_words.append(word)
                                    return ' '.join(capitalized_words)
                                
                                # Apply capitalization
                                facilities_display['Provider Name'] = facilities_display['Provider Name'].apply(capitalize_name)
                                facilities_display['City'] = facilities_display['City'].apply(capitalize_name)
                                
                                # Add filter for high-risk facilities
                                show_high_risk_only = st.checkbox(
                                    "High-risk facilities only", 
                                    value=False,
                                    help="Filter to show only facilities with Overall Rating '1', SFF status, SFF Candidate status, or Abuse Icon 'Y'"
                                )
                                
                                # Apply high-risk filter if selected
                                if show_high_risk_only:
                                    total_facilities = len(facilities_display)
                                    high_risk_mask = (
                                        (facilities_display['Overall Rating'] == 1) |
                                        (facilities_display['Special Focus Status'].str.contains('SFF', case=False, na=False)) |
                                        (facilities_display['Special Focus Status'].str.contains('Candidate', case=False, na=False)) |
                                        (facilities_display['Abuse Icon'] == 'Y')
                                    )
                                    facilities_display = facilities_display[high_risk_mask]
                                    
                                    if len(facilities_display) == 0:
                                        st.info("No high-risk facilities found for this entity.")
                                        st.markdown('</div>', unsafe_allow_html=True)
                                        return
                                    else:
                                        st.success(f"Showing {len(facilities_display)} high-risk facilities out of {total_facilities} total.")
                                
                                # Create provider names as HTML links
                                def format_provnum(provnum):
                                    provnum_str = str(provnum).strip().upper().zfill(6)
                                    if len(provnum_str) > 6:
                                        provnum_str = provnum_str[-6:]
                                    return provnum_str
                                facilities_display['Provider Name'] = facilities_display.apply(
                                    lambda row: f'<a href="https://nursinghomedashboard.streamlit.app/?level=Facility&facility={format_provnum(row["CMS Certification Number (CCN)"])}" target="_blank">{row["Provider Name"]}</a>',
                                    axis=1
                                )
                                
                                # Reorder columns to: State, Provider Name, CMS CCN, City, etc.
                                column_order = [
                                    'State',
                                    'Provider Name',
                                    'CMS Certification Number (CCN)',
                                    'City',
                                    'Overall Rating',
                                    'Staffing Rating',
                                    'Special Focus Status',
                                    'Abuse Icon'
                                ]
                                facilities_display = facilities_display[column_order]
                                
                                # Render as HTML table for clickable links
                                html_table = facilities_display.to_html(
                                    index=False,
                                    escape=False,
                                    classes=['dataframe', 'table', 'table-striped'],
                                    table_id='facilities-table'
                                )
                                
                                # Add CSS and JavaScript for table styling and sorting
                                st.markdown("""
                                <style>
                                .dataframe {
                                    width: 100%;
                                    border-collapse: collapse;
                                    margin: 0.3rem 0;
                                    font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                                    font-size: 0.8em;
                                    box-shadow: 0 1px 4px rgba(0,0,0,0.1);
                                    border-radius: 6px;
                                    overflow: hidden;
                                }
                                .dataframe th {
                                    background: #f8f9fa;
                                    padding: 6px 4px;
                                    text-align: left;
                                    font-weight: 600;
                                    border: none;
                                    border-bottom: 2px solid #e9ecef;
                                    color: #495057;
                                    font-size: 0.75em;
                                    text-transform: uppercase;
                                    letter-spacing: 0.3px;
                                }
                                .dataframe td {
                                    padding: 4px 6px;
                                    border-bottom: 1px solid #f0f0f0;
                                    vertical-align: middle;
                                    font-size: 0.8em;
                                }
                                .dataframe tr:hover {
                                    background-color: #f8f9fa;
                                    transform: translateY(-1px);
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                                }
                                .dataframe tr:nth-child(even) {
                                    background-color: #fafbfc;
                                }
                                .dataframe tr:nth-child(even):hover {
                                    background-color: #f0f2f5;
                                }
                                .dataframe a {
                                    color: #007bff;
                                    text-decoration: none;
                                    font-weight: 500;
                                    transition: color 0.2s ease;
                                }
                                .dataframe a:hover {
                                    color: #0056b3;
                                    text-decoration: underline;
                                }
                                </style>
                                """, unsafe_allow_html=True)
                                
                                st.markdown(html_table, unsafe_allow_html=True)
                                
                                st.markdown('</div>', unsafe_allow_html=True)
                                
                            else:
                                st.info("No facility data available for this entity.")
                        else:
                            st.info("Entity ID not available for facility lookup.")
                        
                        # Add subscription button for entity
                        display_subscription_button("entity", selected_value, f"{selected_value} Entity Data")
                    else:
                        st.error(f"Entity '{selected_value}' not found in the data.")
                else:
                    st.error("Unable to load entity data.")

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
                    # Show warning with CSS to hide on mobile using Streamlit's actual CSS classes
                    st.markdown("""
                    <style>
                    @media (max-width: 768px) {
                        /* Hide Streamlit warning boxes on mobile */
                        div[data-testid="stAlert"] {
                            display: none !important;
                        }
                        /* Alternative selectors for warning boxes */
                        .stAlert {
                            display: none !important;
                        }
                        [data-testid="stAlert"] {
                            display: none !important;
                        }
                    }
                    </style>
                    """, unsafe_allow_html=True)
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