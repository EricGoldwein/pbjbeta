import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import os
import time
import glob
from openai import OpenAI
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
import re
from plotly.subplots import make_subplots
from fpdf import FPDF
import tempfile
import duckdb
from functools import lru_cache
from streamlit_searchbox import st_searchbox
from typing import Dict, Optional, List, Tuple, Any
from fix_pdf import create_pdf_report

# Set page configuration with a more professional theme
st.set_page_config(
    page_title="Nursing Home Staffing Dashboard (Beta)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize DuckDB connection for facility data
facility_db = duckdb.connect(':memory:')

# Initialize provider info cache
provider_info_cache: Dict[str, Dict[str, str]] = {}

@st.cache_data
def load_metrics_data():
    """Load and cache all metrics data."""
    try:
        # Load all metrics data at once
        national_metrics = pd.read_csv('national_quarterly_metrics.csv')
        state_metrics = pd.read_csv('state_quarterly_metrics.csv')
        region_metrics = pd.read_csv('region_quarterly_metrics.csv')
        facility_metrics = pd.read_csv('facility_quarterly_metrics.csv', dtype={'PROVNUM': str})

        # Standardize column names
        for df in [national_metrics, state_metrics, region_metrics, facility_metrics]:
            if 'CY_Qtr' in df.columns:
                df.rename(columns={'CY_Qtr': 'CY_QTR'}, inplace=True)

        # Convert CY_QTR to datetime for all dataframes
        for df in [national_metrics, state_metrics, region_metrics, facility_metrics]:
            df['date'] = pd.to_datetime(df['CY_QTR'].str[:4] + '-' + 
                                      ((df['CY_QTR'].str[-1].astype(int) - 1) * 3 + 1).astype(str).str.zfill(2) + 
                               '-01')
        
        return national_metrics, state_metrics, region_metrics, facility_metrics
    except Exception as e:
        st.error(f"Error loading metrics data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

@st.cache_data
def create_facility_db():
    """Create an optimized DuckDB database for facility data."""
    try:
        # Load facility metrics into DuckDB
        facility_metrics = pd.read_csv('facility_quarterly_metrics.csv', dtype={'PROVNUM': str})
        
        if not facility_metrics.empty:
            facility_db.execute("""
                CREATE TABLE IF NOT EXISTS facility_metrics AS 
                SELECT * FROM facility_metrics
            """)
            
            # Create indexes for faster lookups
            facility_db.execute("CREATE INDEX IF NOT EXISTS idx_provnum ON facility_metrics(PROVNUM)")
            facility_db.execute("CREATE INDEX IF NOT EXISTS idx_date ON facility_metrics(date)")
            facility_db.execute("CREATE INDEX IF NOT EXISTS idx_quarter ON facility_metrics(CY_QTR)")
    except Exception as e:
        st.error(f"Error creating facility database: {str(e)}")

# Initialize data at startup
try:
    national_metrics, state_metrics, region_metrics, facility_metrics = load_metrics_data()
    create_facility_db()
except Exception as e:
    st.error(f"Error during initialization: {str(e)}")
    national_metrics = pd.DataFrame()
    state_metrics = pd.DataFrame()
    region_metrics = pd.DataFrame()
    facility_metrics = pd.DataFrame()

# Provider info cache with SQLite backend
class ProviderInfoCache:
    def __init__(self) -> None:
        self.conn = sqlite3.connect(':memory:')
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS provider_info (
                provnum TEXT PRIMARY KEY,
                name TEXT,
                state TEXT,
                county TEXT,
                city TEXT
            )
        ''')
        self.conn.commit()
    
    def get(self, provnum: str, info_type: str) -> Optional[str]:
        try:
            cursor = self.conn.execute(
                f'SELECT {info_type} FROM provider_info WHERE provnum = ?',
                (provnum,)
            )
            result = cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            st.error(f"Error getting provider info: {str(e)}")
            return None
    
    def set(self, provnum: str, info_type: str, value: str) -> None:
        try:
            self.conn.execute(
                f'''
                INSERT OR REPLACE INTO provider_info (provnum, {info_type})
                VALUES (?, ?)
                ''',
                (provnum, value)
            )
            self.conn.commit()
        except Exception as e:
            st.error(f"Error setting provider info: {str(e)}")
    
    def close(self) -> None:
        try:
            self.conn.close()
        except Exception as e:
            st.error(f"Error closing provider cache: {str(e)}")

# Initialize global cache
provider_cache = ProviderInfoCache()

@st.cache_data
def load_provider_info_cache() -> None:
    """Load all provider information into cache at startup."""
    global provider_info_cache
    
    # Map of info types to possible column names
    column_maps = {
        'name': ['PROVNAME', 'provname', 'PROVIDER_NAME', 'provider_name'],
        'state': ['STATE', 'state'],
        'county': ['COUNTY_NAME', 'county_name', 'COUNTY', 'county'],
        'city': ['CITY', 'city']
    }
    
    pbj_files = sorted(glob.glob('PBJ_Nurse/*.csv'), reverse=True)
    
    for pbj_file in pbj_files:
        try:
            df = pd.read_csv(pbj_file, dtype={'PROVNUM': str, 'provnum': str})
            
            # Standardize PROVNUM column
            prov_col = next(
                (col for col in df.columns 
                 if col.lower() in ['provnum', 'provider_number']),
                None
            )
            if not prov_col:
                continue
                
            df.rename(columns={prov_col: 'PROVNUM'}, inplace=True)
            
            # Process each info type
            for info_type, target_columns in column_maps.items():
                info_col = next(
                    (col for col in df.columns 
                     if col in target_columns),
                    None
                )
                if not info_col:
                    continue
                    
                # Update cache for each provider
                for _, row in df.iterrows():
                    provnum = row['PROVNUM']
                    if provnum not in provider_info_cache:
                        provider_info_cache[provnum] = {}
                    provider_info_cache[provnum][info_type] = row[info_col]
                    
        except Exception as e:
            st.error(f"Error processing {pbj_file}: {str(e)}")
            continue

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
        # Check cache first
        cache_key = f"{provnum}_{info_type}"
        if cache_key in provider_info_cache:
            value = provider_info_cache[cache_key]
            # Apply proper title case to name and city
            if info_type in ['name', 'city']:
                value = proper_title_case(value)
            return value
        
        # Map of info types to possible column names
        column_maps = {
            'name': ['PROVNAME', 'provname', 'PROVIDER_NAME', 'provider_name'],
            'state': ['STATE', 'state'],
            'county': ['COUNTY_NAME', 'county_name', 'COUNTY', 'county'],
            'city': ['CITY', 'city']
        }
        
        target_columns = column_maps.get(info_type.lower(), [])
        if not target_columns:
            return 'N/A'
        
        # Try different encodings
        encodings = ['latin1', 'cp1252', 'utf-8']
        pbj_files = sorted(glob.glob('PBJ_Nurse/*.csv'), reverse=True)
        
        for pbj_file in pbj_files:
            for encoding in encodings:
                try:
                    df = pd.read_csv(pbj_file, encoding=encoding, dtype={'PROVNUM': str, 'provnum': str})
                    
                    # Standardize PROVNUM column
                    prov_col = next(
                        (col for col in df.columns 
                         if col.lower() in ['provnum', 'provider_number']),
                        None
                    )
                    if not prov_col:
                        continue
                    
                    df.rename(columns={prov_col: 'PROVNUM'}, inplace=True)
                    
                    # Find matching info column
                    info_col = next(
                        (col for col in df.columns 
                         if col in target_columns),
                        None
                    )
                    if not info_col:
                        continue
                    
                    provider_data = df[df['PROVNUM'] == provnum]
                    if not provider_data.empty:
                        value = provider_data.iloc[0][info_col]
                        # Apply proper title case to name and city
                        if info_type in ['name', 'city']:
                            value = proper_title_case(value)
                        # Cache the result
                        provider_info_cache[cache_key] = value
                        return value
                    
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    st.error(f"Error processing {pbj_file}: {str(e)}")
                    continue
        
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

def get_provider_city(provnum):
    """Get provider city from PBJ files."""
    return get_provider_info(provnum, 'city')

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
        national_metrics, state_metrics, region_metrics, _ = load_metrics_data()
        
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
        elif level == "Region":
            if selected_value == 'All Regions':
                return region_metrics[
                    (region_metrics['CY_QTR'] >= start_quarter) & 
                    (region_metrics['CY_QTR'] <= end_quarter)
                ]
            else:
                return region_metrics[
                    (region_metrics['Region'] == selected_value) & 
                    (region_metrics['CY_QTR'] >= start_quarter) & 
                    (region_metrics['CY_QTR'] <= end_quarter)
                ]
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error filtering data: {str(e)}")
        return pd.DataFrame()

# Initialize the database at startup
create_facility_db()

# Load data at startup
national_metrics, state_metrics, region_metrics, facility_metrics = load_metrics_data()

def get_days_in_quarter(quarter):
    """Calculate the number of days in a given quarter.
    Args:
        quarter (str): Quarter in format 'YYYYQN' (e.g., '2024Q3')
    Returns:
        int: Number of days in the quarter
    """
    # Extract year and quarter number from the string
    year = int(quarter[:4])
    q = int(quarter[-1])
    
    if q == 1:  # Q1: January 1 to March 31
        # Check if it's a leap year
        if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
            return 91  # Leap year
        return 90  # Non-leap year
    elif q == 2:  # Q2: April 1 to June 30
        return 91
    elif q == 3:  # Q3: July 1 to September 30
        return 92
    else:  # Q4: October 1 to December 31
        return 92

# Function to format metrics
def format_metric(value, decimal_places=1, percentage=False, thousands=False):
    if pd.isna(value):
        return "N/A"
    if percentage:
        return f"{value:.{decimal_places}f}%"
    if thousands:
        return f"{value:,.{decimal_places}f}"
    return f"{value:.{decimal_places}f}"

@st.cache_data
def create_quarterly_table(filtered_data: pd.DataFrame, level: str) -> pd.DataFrame:
    """Create an optimized quarterly data table."""
    try:
        # Sort data by date
        filtered_data = filtered_data.sort_values('date', ascending=True)
        
        # Calculate MDS Census
        filtered_data['MDS_Census'] = filtered_data.apply(
            lambda row: row['Total_Resident_Days'] / get_days_in_quarter(row['CY_QTR']), 
            axis=1
        )
        
        # Format quarter display
        filtered_data['Quarter_Display'] = filtered_data['CY_QTR'].apply(
            lambda x: f"Q{x[-1]} {x[:4]}"
        )
        
        # Create table with specified columns
        if level.lower() == "facility":
            table_data = filtered_data[[
                'Quarter_Display',  # Use formatted quarter
                'MDS_Census',
                'Total_HPRD',
                'RN_HPRD',
                'Nurse_Assistant_HPRD',
                'Contract_Staff_Percentage'
            ]].copy()
            
            # Format the numbers for facility level
            table_data['MDS_Census'] = table_data['MDS_Census'].map('{:,.0f}'.format)
            table_data['Total_HPRD'] = table_data['Total_HPRD'].map('{:.2f}'.format)
            table_data['RN_HPRD'] = table_data['RN_HPRD'].map('{:.2f}'.format)
            table_data['Nurse_Assistant_HPRD'] = table_data['Nurse_Assistant_HPRD'].map('{:.2f}'.format)
            table_data['Contract_Staff_Percentage'] = table_data['Contract_Staff_Percentage'].map('{:.1f}%'.format)
            
            table_data.columns = [
                'Quarter',
                'MDS Census',
                'Total Nurse HPRD',
                'RN HPRD',
                'Nurse Assistant HPRD',
                'Contract Staff %'
            ]
        else:
            # First check which columns are available
            available_columns = ['Quarter_Display', 'Facility_Count', 'MDS_Census', 'Total_HPRD', 
                               'RN_HPRD', 'Nurse_Assistant_HPRD', 'Contract_Staff_Percentage']
            
            # Only include Median_Contract_Percentage if it exists
            if 'Median_Contract_Percentage' in filtered_data.columns:
                available_columns.append('Median_Contract_Percentage')
            
            # Filter columns that exist in the dataframe
            table_columns = [col for col in available_columns if col in filtered_data.columns]
            table_data = filtered_data[table_columns].copy()
            
            # Format the numbers
            if 'Facility_Count' in table_data.columns:
                table_data['Facility_Count'] = table_data['Facility_Count'].map('{:,}'.format)
            table_data['MDS_Census'] = table_data['MDS_Census'].map('{:,.0f}'.format)
            table_data['Total_HPRD'] = table_data['Total_HPRD'].map('{:.2f}'.format)
            table_data['RN_HPRD'] = table_data['RN_HPRD'].map('{:.2f}'.format)
            table_data['Nurse_Assistant_HPRD'] = table_data['Nurse_Assistant_HPRD'].map('{:.2f}'.format)
            table_data['Contract_Staff_Percentage'] = table_data['Contract_Staff_Percentage'].map('{:.1f}%'.format)
            if 'Median_Contract_Percentage' in table_data.columns:
                table_data['Median_Contract_Percentage'] = table_data['Median_Contract_Percentage'].map('{:.1f}%'.format)
            
            # Create column mapping
            column_mapping = {
                'Quarter_Display': 'Quarter',
                'Facility_Count': 'Facility Count',
                'MDS_Census': 'MDS Census',
                'Total_HPRD': 'Total Nurse HPRD',
                'RN_HPRD': 'RN HPRD',
                'Nurse_Assistant_HPRD': 'Nurse Assistant HPRD',
                'Contract_Staff_Percentage': 'Contract Staff %',
                'Median_Contract_Percentage': 'Median Contract %'
            }
            
            # Only rename columns that exist
            rename_cols = {k: v for k, v in column_mapping.items() if k in table_data.columns}
            table_data.columns = [rename_cols.get(col, col) for col in table_data.columns]
        
        return table_data
    except Exception as e:
        st.error(f"Error creating quarterly table: {str(e)}")
        return pd.DataFrame()

def display_metrics(metrics: pd.DataFrame, level: str):
    """Display metrics with optimized calculations."""
    try:
        if metrics.empty:
            st.warning(f"No data available for the selected {level}.")
            return

        # Get all available quarters for this dataset
        available_quarters = sort_quarters(metrics['CY_QTR'].unique(), reverse=True)  # Newest to oldest
        
        # Create a row with two columns for the header and quarter selector
        header_col1, header_col2 = st.columns([3, 1])
        
        # Get the selected quarter (default to most recent)
        with header_col2:
            current_quarter = st.selectbox(
                "Select Quarter",
                available_quarters,
                index=0,
                label_visibility="collapsed",
                format_func=format_quarter_display
            )
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

        # Calculate average daily census
        days_in_quarter = get_days_in_quarter(current_quarter)
        avg_daily_census = current_metrics['Total_Resident_Days'].iloc[0] / days_in_quarter
        
        # Calculate previous quarter's census if available
        if not prev_metrics.empty:
            prev_days_in_quarter = get_days_in_quarter(prev_quarter)
            prev_avg_daily_census = prev_metrics['Total_Resident_Days'].iloc[0] / prev_days_in_quarter
        else:
            prev_avg_daily_census = None
        
        # Display Key Metrics header with level-specific title
        if level == "State":
            header_text = f"{metrics['STATE'].iloc[0]} Key Metrics ({quarter_name})"
        elif level == "Region":
            header_text = f"{metrics['Region'].iloc[0]} Key Metrics ({quarter_name})"
        else:
            header_text = f"Key Metrics ({quarter_name})"
            
        with header_col1:
            st.markdown(f'<div class="section-header" style="margin-top: 8px;">{header_text}</div>', unsafe_allow_html=True)
            
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("MDS Census", 
                     format_metric(avg_daily_census, decimal_places=0, thousands=True),
                     format_metric(avg_daily_census - prev_avg_daily_census, decimal_places=0, thousands=True) if prev_avg_daily_census is not None else None)
        with col2:
            st.metric("Total HPRD", 
                     format_metric(current_metrics['Total_HPRD'].iloc[0], decimal_places=2),
                     format_metric(current_metrics['Total_HPRD'].iloc[0] - prev_metrics['Total_HPRD'].iloc[0], decimal_places=2) if not prev_metrics.empty else None)
        with col3:
            st.metric("RN HPRD", 
                     format_metric(current_metrics['RN_HPRD'].iloc[0], decimal_places=2),
                     format_metric(current_metrics['RN_HPRD'].iloc[0] - prev_metrics['RN_HPRD'].iloc[0], decimal_places=2) if not prev_metrics.empty else None)
        with col4:
            st.metric("Nurse Assistant HPRD", 
                     format_metric(current_metrics['Nurse_Assistant_HPRD'].iloc[0], decimal_places=2),
                     format_metric(current_metrics['Nurse_Assistant_HPRD'].iloc[0] - prev_metrics['Nurse_Assistant_HPRD'].iloc[0], decimal_places=2) if not prev_metrics.empty else None)
        with col5:
            st.metric("Contract Staff %", 
                     format_metric(current_metrics['Contract_Staff_Percentage'].iloc[0], decimal_places=1, percentage=True),
                     format_metric(current_metrics['Contract_Staff_Percentage'].iloc[0] - prev_metrics['Contract_Staff_Percentage'].iloc[0], decimal_places=1, percentage=True) if not prev_metrics.empty else None)
    except Exception as e:
        st.error(f"Error displaying metrics: {str(e)}")

@st.cache_data
def plot_quarterly_trends(df: pd.DataFrame, state: str = None, region: str = None, facility: str = None):
    """Plot quarterly trends with optimized data processing."""
    try:
        # State code to name mapping
        state_names = {
            'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DC': 'District of Columbia',
            'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii',
            'IA': 'Iowa', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana',
            'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts',
            'MD': 'Maryland', 'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota',
            'MO': 'Missouri', 'MS': 'Mississippi', 'MT': 'Montana', 'NC': 'North Carolina',
            'ND': 'North Dakota', 'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
            'NM': 'New Mexico', 'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio',
            'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island',
            'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
            'UT': 'Utah', 'VA': 'Virginia', 'VT': 'Vermont', 'WA': 'Washington',
            'WI': 'Wisconsin', 'WV': 'West Virginia', 'WY': 'Wyoming'
        }

        if state:
            # Get state name from mapping
            state_name = state_names.get(state, state)
            title_prefix = f"{state_name}"
            data = df[df['STATE'] == state].copy()
        elif region:
            title_prefix = f"{region}"
            data = df[df['Region'] == region].copy()
        elif facility:
            # Get facility name and state
            facility_name = get_provider_info(facility, 'name')
            facility_state = get_provider_info(facility, 'state')
            title_prefix = f"{facility_name} ({facility_state})"
            data = df[df['PROVNUM'] == facility].copy()
        else:
            title_prefix = "National"
            data = df.copy()
        
        # Sort data by date
        data = data.sort_values('date')
        
        # Calculate average daily census for each quarter
        data['Avg_Daily_Census'] = data.apply(
            lambda row: row['Total_Resident_Days'] / get_days_in_quarter(row['CY_QTR']), 
            axis=1
        )
        
        # Create year labels for x-axis ticks - ensure all years are included
        min_year = data['date'].dt.year.min()
        max_year = data['date'].dt.year.max()
        all_years = range(min_year, max_year + 1)
        tick_values = [pd.Timestamp(f"{year}-01-01") for year in all_years]
        tick_text = [str(year) for year in all_years]
        
        # Define hover template
        hover_template = "<b>%{customdata}</b><br>Value: %{y:.2f}<extra></extra>"
        if not facility:
            hover_template_count = "<b>%{customdata}</b><br>Count: %{y:,}<extra></extra>"
        else:
            hover_template_facility = "<b>%{customdata}</b><br>RN Care HPRD: %{y:.2f}<extra></extra>"
        
        # Create subplots
        fig = make_subplots(rows=3, cols=2,
                          subplot_titles=('Total Nurse HPRD', 'Contract Staff Percentage',
                                        'RN HPRD', 'Nurse Assistant HPRD',
                                        'Average Daily Census', 'Facility Count' if not facility else 'RN Care HPRD'))

        # Plot 1: Total Nurse HPRD
        fig.add_trace(go.Scatter(x=data['date'], y=data['Total_HPRD'],
                               mode='lines+markers', name='Total HPRD',
                               customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                               hovertemplate=hover_template), row=1, col=1)

        # Plot 2: Contract Staff Percentage
        fig.add_trace(go.Scatter(x=data['date'], y=data['Contract_Staff_Percentage'],
                               mode='lines+markers', name='Contract %',
                               customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                               hovertemplate=hover_template.replace(':.2f', ':.1f%')), row=1, col=2)

        # Plot 3: RN HPRD
        fig.add_trace(go.Scatter(x=data['date'], y=data['RN_HPRD'],
                               mode='lines+markers', name='RN HPRD',
                               customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                               hovertemplate=hover_template), row=2, col=1)

        # Plot 4: Nurse Assistant HPRD
        fig.add_trace(go.Scatter(x=data['date'], y=data['Nurse_Assistant_HPRD'],
                               mode='lines+markers', name='NA HPRD',
                               customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                               hovertemplate=hover_template), row=2, col=2)

        # Plot 5: Average Daily Census
        fig.add_trace(go.Scatter(x=data['date'], y=data['Avg_Daily_Census'],
                               mode='lines+markers', name='Avg Census',
                               customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                               hovertemplate=hover_template.replace(':.2f', ':,.0f')), row=3, col=1)

        # Plot 6: Facility Count or RN Care HPRD
        if not facility:
            fig.add_trace(go.Scatter(x=data['date'], y=data['Facility_Count'],
                                   mode='lines+markers', name='Facilities',
                                   customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                                   hovertemplate=hover_template_count), row=3, col=2)
        else:
            fig.add_trace(go.Scatter(x=data['date'], y=data['RN_Care_HPRD'],
                                   mode='lines+markers', name='RN Care HPRD',
                                   customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                                   hovertemplate=hover_template_facility), row=3, col=2)

        # Update layout
        fig.update_layout(
            height=1200,  # Increased height for better visibility
            title_text=f"{title_prefix} Staffing Trends",
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=200),  # Increased bottom margin
            hovermode='x unified',
            grid=dict(
                rows=3,
                columns=2,
                xgap=0.1,  # Horizontal spacing between subplots
                ygap=0.15  # Vertical spacing between subplots
            )
        )

        # Update x-axes to show only years and set line color
        for i in range(1, 4):
            for j in range(1, 3):
                fig.update_xaxes(
                    ticktext=tick_text,
                    tickvals=tick_values,
                    tickangle=45,  # Set diagonal angle
                    row=i,
                    col=j,
                    showline=True,
                    linewidth=1,
                    linecolor="rgba(200, 200, 200, 0.1)",  # Made even more faint
                    range=[tick_values[0], tick_values[-1]]  # Ensure consistent range across all subplots
                )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Hours per Resident Day", row=1, col=1)
        fig.update_yaxes(title_text="Percentage", row=1, col=2)
        fig.update_yaxes(title_text="Hours per Resident Day", row=2, col=1)
        fig.update_yaxes(title_text="Hours per Resident Day", row=2, col=2)
        fig.update_yaxes(title_text="Residents", row=3, col=1)
        if not facility:
            fig.update_yaxes(title_text="Number of Facilities", row=3, col=2)
        else:
            fig.update_yaxes(title_text="Hours per Resident Day", row=3, col=2)
            # Update subplot title for facility view plot 6
            fig.layout.annotations[5].update(text='RN Care HPRD')
        
        # Add individual footer annotations for each subplot using subplot-specific coordinates
        # Top row
        fig.add_annotation(
            text="320 Consulting | Source: CMS PBJ Data",
            x=0.95,  # Moved to right
            y=-0.33,  # Keep the lower position
            xref="x domain",
            yref="y domain",
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="right",  # Changed to right alignment
            row=1,
            col=1
        )
        
        fig.add_annotation(
            text="320 Consulting | Source: CMS PBJ Data",
            x=0.95,  # Moved to right
            y=-0.33,  # Keep the lower position
            xref="x domain",
            yref="y domain",
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="right",  # Changed to right alignment
            row=1,
            col=2
        )
        
        # Middle row
        fig.add_annotation(
            text="320 Consulting | Source: CMS PBJ Data",
            x=0.95,  # Moved to right
            y=-0.33,  # Keep the lower position
            xref="x domain",
            yref="y domain",
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="right",  # Changed to right alignment
            row=2,
            col=1
        )
        
        fig.add_annotation(
            text="320 Consulting | Source: CMS PBJ Data",
            x=0.95,  # Moved to right
            y=-0.33,  # Keep the lower position
            xref="x domain",
            yref="y domain",
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="right",  # Changed to right alignment
            row=2,
            col=2
        )
        
        # Bottom row
        fig.add_annotation(
            text="320 Consulting | Source: CMS PBJ Data",
            x=0.95,  # Moved to right
            y=-0.33,  # Keep the lower position
            xref="x domain",
            yref="y domain",
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="right",  # Changed to right alignment
            row=3,
            col=1
        )
        
        fig.add_annotation(
            text="320 Consulting | Source: CMS PBJ Data",
            x=0.95,  # Moved to right
            y=-0.33,  # Keep the lower position
            xref="x domain",
            yref="y domain",
            showarrow=False,
            font=dict(size=10, color="gray"),
            align="right",  # Changed to right alignment
            row=3,
            col=2
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Error plotting trends: {str(e)}")

# Function to create PDF report
def create_pdf_report(provnum: str, selected_quarter: str) -> bytes:
    """Create a PDF report for the facility."""
    try:
        # Get facility info
        conn = get_db_connection()
        if not conn:
            print("Could not connect to database")
            return None
            
        # Get facility info
        query = """
            SELECT DISTINCT
                PROVNUM,
                PROVNAME,
                STATE,
                COUNTY_NAME,
                CITY
            FROM staffing 
            WHERE PROVNUM = ?
            ORDER BY WORKDATE DESC
            LIMIT 1
        """
        
        result = conn.execute(query, (provnum,)).fetchone()
        if not result:
            print(f"No facility info found for {provnum}")
            return None
            
        facility_info = {
            'ccn': result[0],
            'provider_name': result[1],
            'state': result[2],
            'county': result[3],
            'city': result[4]
        }
        
        # Get metrics
        metrics_query = """
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
        
        metrics_result = conn.execute(metrics_query, (selected_quarter, provnum)).fetchone()
        if not metrics_result or metrics_result[0] is None:
            print(f"No metrics found for {provnum} in quarter {selected_quarter}")
            return None
            
        total_resident_days = float(metrics_result[0])
        metrics = {
            'total_hours': float(metrics_result[1]) / total_resident_days if total_resident_days > 0 else 0,
            'rn_hours': float(metrics_result[2]) / total_resident_days if total_resident_days > 0 else 0,
            'nurse_care_hours': float(metrics_result[3]) / total_resident_days if total_resident_days > 0 else 0
        }
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Set font
        pdf.set_font("Arial", "B", 16)
        
        # Add title
        pdf.cell(0, 10, f"PBJ Staffing Report - {facility_info['provider_name']}", ln=True, align="C")
        pdf.ln(10)
        
        # Add facility info
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"CCN: {facility_info['ccn']}", ln=True)
        pdf.cell(0, 10, f"Location: {facility_info['city']}, {facility_info['state']}", ln=True)
        pdf.ln(10)
        
        # Add metrics
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Key Metrics - {selected_quarter}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Total Staffing Hours: {metrics['total_hours']:.2f}", ln=True)
        pdf.cell(0, 10, f"RN Hours: {metrics['rn_hours']:.2f}", ln=True)
        pdf.cell(0, 10, f"Nurse Care Hours: {metrics['nurse_care_hours']:.2f}", ln=True)
        
        # Get citations
        citations = get_facility_citations(provnum)
        if not citations.empty:
            pdf.ln(10)
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 10, "Recent Citations", ln=True)
            pdf.ln(5)
            
            # Add citations table
            pdf.set_font("Arial", "", 10)
            col_widths = [40, 30, 30, 90]  # Adjust as needed
            pdf.cell(col_widths[0], 10, "Survey Date", 1)
            pdf.cell(col_widths[1], 10, "Tag Number", 1)
            pdf.cell(col_widths[2], 10, "Severity", 1)
            pdf.cell(col_widths[3], 10, "Deficiency Type", 1)
            pdf.ln()
            
            for _, row in citations.iterrows():
                pdf.cell(col_widths[0], 10, str(row.get('survey_date', 'N/A')), 1)
                pdf.cell(col_widths[1], 10, str(row.get('tag_number', 'N/A')), 1)
                pdf.cell(col_widths[2], 10, str(row.get('severity_code', 'N/A')), 1)
                pdf.cell(col_widths[3], 10, str(row.get('deficiency_type', 'N/A')), 1)
                pdf.ln()
        
        # Save to bytes
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        print(f"Error creating PDF: {str(e)}")
        return None

# Get the assistant
def get_assistant():
    assistant_id = os.getenv('OPENAI_ASSISTANT_ID')
    return client.beta.assistants.retrieve(assistant_id)

# Function to generate AI report
def generate_ai_report(state_data, report_type="brief", prompt_type=""):
    try:
        # Debug output
        st.write("Debug: state_data keys:", list(state_data.keys()))
        
        # Get the assistant
        assistant = get_assistant()
        
        # Create a thread
        thread = client.beta.threads.create()
        
        # Prepare the data context
        context = f"""
        State: {state_data['state']}
        Time Period: {state_data['quarter']}
        
        Key Metrics:
        - Total HPRD: {state_data['total_hprd']:.2f} (National: {state_data['national_total_hprd']:.2f})
        - RN HPRD: {state_data['rn_hprd']:.2f} (National: {state_data['national_rn_hprd']:.2f})
        - Number of Facilities: {state_data['facility_count']}
        - Average Daily Residents: {state_data['avg_daily_residents']:,.0f}
        
        Additional Metrics:
        - Nurse Care Staff HPRD: {state_data['nurse_care_hprd']:.2f}
        - RN Care Staff HPRD: {state_data['rn_care_hprd']:.2f}
        - Nurse Assistant HPRD: {state_data['nurse_assistant_hprd']:.2f}
        
        Contract Staff Metrics:
        - Contract Staff Percentage: {state_data['contract_staff_percentage']:.1f}% (National: {state_data['national_contract_percentage']:.1f}%)
        - Median Contract Staff Percentage: {state_data['median_contract_percentage']:.1f}%
        """
        
        # Create the message
        prompt = f"Please generate a {report_type} report analyzing the nursing home staffing trends in {state_data['state']} based on the following data. Focus on key insights, patterns, and recommendations. {prompt_type}"
        
        # Add the message to the thread
        message = client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=f"{prompt}\n\nData:\n{context}"
        )
        
        # Run the assistant
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        # Wait for completion
        while True:
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
            if run.status == 'completed':
                break
            time.sleep(1)
        
        # Get the response
        messages = client.beta.threads.messages.list(thread_id=thread.id)
        report = messages.data[0].content[0].text.value
        
        return report
        
    except Exception as e:
        st.error(f"Error generating report: {str(e)}")
        return None

# Add custom CSS for better styling
st.markdown("""
<style>
    .stMetric {
        background-color: white;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        transition: transform 0.2s;
        margin-bottom: 8px;
    }
    .stMetric:hover {
        transform: translateY(-1px);
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 20px;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 4px;
    }
    .stMetric [data-testid="stMetricLabel"] {
        font-size: 12px;
        color: #666;
    }
    .section-header {
        font-size: 16px;
        color: #1E88E5;
        margin-bottom: 8px;
        font-weight: 600;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        margin-bottom: 16px;
    }
    .main {
        padding: 2rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 1rem;
        color: #666;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        padding: 0.5rem;
        font-weight: bold;
    }
    .stSelectbox {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    .stSelectbox > div {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    .stSelectbox > div > div {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    /* Ensure consistent height for header row */
    [data-testid="column"] {
        min-height: 45px !important;
    }
    /* Adjust vertical alignment of header text */
    .section-header {
        line-height: 35px !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def search_facilities(search_term: str) -> List[Dict[str, str]]:
    """Search facilities with lazy loading and caching."""
    try:
        # Sanitize search term to prevent SQL injection
        search_term = search_term.replace("'", "''")
        
        # Get matching facilities using DuckDB with parameterized query
        query = """
            SELECT DISTINCT PROVNUM, PROVNAME, STATE, COUNTY_NAME, CITY
            FROM facility_metrics 
            WHERE PROVNUM LIKE ?
            LIMIT 50
        """
        matching_facilities = facility_db.execute(query, (f"%{search_term}%",)).fetchdf()
        
        if not matching_facilities.empty:
            # Apply proper title case to PROVNAME and CITY
            matching_facilities['PROVNAME'] = matching_facilities['PROVNAME'].apply(proper_title_case)
            matching_facilities['CITY'] = matching_facilities['CITY'].apply(proper_title_case)
            return matching_facilities.to_dict('records')
        return []
    except Exception as e:
        st.error(f"Error searching facilities: {str(e)}")
        return []

def init_citations_db():
    """Initialize citations database with DuckDB."""
    try:
        citations_db = duckdb.connect(':memory:')
        
        # Read citations CSV directly into DuckDB with all necessary columns
        citations_db.execute("""
            CREATE TABLE IF NOT EXISTS citations AS 
            SELECT 
                "CMS Certification Number (CCN)" as provnum,
                "Survey Date" as survey_date,
                "Deficiency Tag Number" as tag_number,
                "Scope Severity Code" as severity_code,
                "Standard Deficiency" as standard_deficiency,
                "Complaint Deficiency" as complaint_deficiency,
                "Infection Control Inspection Deficiency" as infection_deficiency,
                "Deficiency Category" as deficiency_category
            FROM read_csv('NH_HealthCitations_Mar2025.csv')
        """)
        
        # Create indexes for faster lookups
        citations_db.execute("CREATE INDEX IF NOT EXISTS idx_citations_provnum ON citations(provnum)")
        citations_db.execute("CREATE INDEX IF NOT EXISTS idx_citations_date ON citations(survey_date)")
        
        return citations_db
    except Exception as e:
        st.error(f"Error initializing citations database: {str(e)}")
        return None

@st.cache_data
def get_facility_citations(provnum: str, limit: int = 100) -> pd.DataFrame:
    """Get citations for a specific facility with caching."""
    try:
        citations_db = init_citations_db()
        if citations_db is None:
            return pd.DataFrame()
            
        query = f"""
            SELECT 
                survey_date,
                tag_number,
                severity_code,
                standard_deficiency,
                complaint_deficiency,
                infection_deficiency,
                deficiency_category
            FROM citations 
            WHERE provnum = '{provnum}'
            ORDER BY survey_date DESC
            LIMIT {limit}
        """
        
        result = citations_db.execute(query).fetchdf()
        citations_db.close()  # Close the connection after use
        return result
    except Exception as e:
        st.error(f"Error getting facility citations: {str(e)}")
        return pd.DataFrame()

def get_quarter_from_date(date_str):
    """Convert a date string to quarter format (e.g., Q1 2023)."""
    try:
        date = pd.to_datetime(date_str)
        quarter = (date.month - 1) // 3 + 1
        return f"Q{quarter} {date.year}"
    except:
        return "N/A"

def display_facility_citations(provnum: str):
    """Display citations for a facility in a clean table format."""
    try:
        citations_db = init_citations_db()
        if citations_db is None:
            return
            
        # Get all citations for the facility
        query = f"""
            SELECT 
                survey_date,
                tag_number,
                severity_code,
                standard_deficiency,
                complaint_deficiency,
                infection_deficiency,
                deficiency_category
            FROM citations 
            WHERE provnum = '{provnum}'
            ORDER BY survey_date DESC
        """
        
        citations = citations_db.execute(query).fetchdf()
        citations_db.close()
        
        if citations.empty:
            st.info("No citations found for this facility.")
            return
            
        # Format the data for display
        citations['survey_date'] = pd.to_datetime(citations['survey_date']).dt.strftime('%m-%d-%Y')
        
        # Create hyperlinks based on deficiency types
        def create_hyperlink(row):
            links = []
            # Convert display date back to datetime for hyperlink format
            link_date = pd.to_datetime(row['survey_date']).strftime('%Y-%m-%d')
            if row['standard_deficiency'] == 'Y':
                links.append(f'<a href="https://www.medicare.gov/care-compare/inspections/pdf/nursing-home/{provnum}/health/health-inspection?date={link_date}" target="_blank">View Standard Report</a>')
            if row['complaint_deficiency'] == 'Y':
                links.append(f'<a href="https://www.medicare.gov/care-compare/inspections/pdf/nursing-home/{provnum}/health/complaint-inspection?date={link_date}" target="_blank">View Complaint Report</a>')
            if row['infection_deficiency'] == 'Y':
                links.append(f'<a href="https://www.medicare.gov/care-compare/inspections/pdf/nursing-home/{provnum}/health/infection-control-inspection?date={link_date}" target="_blank">View Infection Report</a>')
            return ' | '.join(links) if links else 'N/A'
        
        citations['Links'] = citations.apply(create_hyperlink, axis=1)
        
        # Create a styled table with reduced column widths and sorting
        st.markdown("""
            <style>
            .citations-table {
                margin-top: 1rem;
                width: 100%;
                border-collapse: collapse;
                table-layout: fixed;
            }
            .citations-table th {
                background-color: #f8f9fa;
                font-weight: 600;
                padding: 8px;
                text-align: left;
                border: 1px solid #dee2e6;
                cursor: pointer;
                position: relative;
            }
            .citations-table th:hover {
                background-color: #e9ecef;
            }
            .citations-table th::after {
                content: '↕';
                position: absolute;
                right: 8px;
                opacity: 0.5;
            }
            .citations-table td {
                font-size: 0.9em;
                padding: 8px;
                border: 1px solid #dee2e6;
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            }
            .citations-table a {
                color: #1E88E5;
                text-decoration: none;
            }
            .citations-table a:hover {
                text-decoration: underline;
            }
            .citations-table th:nth-child(1) { width: 15%; }  /* Survey Date */
            .citations-table th:nth-child(2) { width: 10%; }  /* F-Tag */
            .citations-table th:nth-child(3) { width: 20%; }  /* Category */
            .citations-table th:nth-child(4) { width: 10%; }  /* Severity */
            .citations-table th:nth-child(5) { width: 45%; }  /* PDF */
            </style>
            
            <script>
            function sortTable(n) {
                var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                table = document.querySelector('.citations-table');
                switching = true;
                dir = "asc";
                
                while (switching) {
                    switching = false;
                    rows = table.rows;
                    
                    for (i = 1; i < (rows.length - 1); i++) {
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName("TD")[n];
                        y = rows[i + 1].getElementsByTagName("TD")[n];
                        
                        if (dir == "asc") {
                            if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
                                shouldSwitch = true;
                                break;
                            }
                        } else if (dir == "desc") {
                            if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {
                                shouldSwitch = true;
                                break;
                            }
                        }
                    }
                    
                    if (shouldSwitch) {
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                    } else {
                        if (switchcount == 0 && dir == "asc") {
                            dir = "desc";
                            switching = true;
                        }
                    }
                }
            }
            </script>
        """, unsafe_allow_html=True)
        
        # Prepare the display DataFrame with renamed columns
        display_df = citations[['survey_date', 'tag_number', 'deficiency_category', 'severity_code', 'Links']].copy()
        display_df.columns = ['Survey Date', 'F-Tag', 'Category', 'Severity', 'PDF']
        
        # Create the table HTML with sorting functionality
        table_html = display_df.to_html(
            escape=False,
            index=False,
            classes='citations-table'
        )
        
        # Add onclick handlers to the header cells
        table_html = table_html.replace('<th>', '<th onclick="sortTable(Array.from(this.parentNode.children).indexOf(this))">')
        
        # Display the table
        st.markdown("---")
        st.header("Recent Citations")
        st.markdown(table_html, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error displaying citations: {str(e)}")

def generate_citations_section(provnum: str) -> str:
    """Generate the citations section of the report."""
    try:
        citations_db = init_citations_db()
        if citations_db is None:
            return ""
            
        # Get all citations for the facility
        query = f"""
            SELECT 
                survey_date,
                tag_number,
                severity_code,
                standard_deficiency,
                complaint_deficiency,
                infection_deficiency,
                deficiency_category
            FROM citations 
            WHERE provnum = '{provnum}'
            ORDER BY survey_date DESC
        """
        
        citations = citations_db.execute(query).fetchdf()
        citations_db.close()
        
        if citations.empty:
            return "<h2>Recent Citations</h2><p>No citations found for this facility.</p>"
            
        # Format the data for display
        citations['survey_date'] = pd.to_datetime(citations['survey_date']).dt.strftime('%Y-%m-%d')
        citations['survey_quarter'] = citations['survey_date'].apply(get_quarter_from_date)
        
        # Create hyperlinks based on deficiency types
        def create_hyperlink(row):
            links = []
            if row['standard_deficiency'] == 'Y':
                links.append(f'<a href="https://www.medicare.gov/care-compare/inspections/pdf/nursing-home/{provnum}/health/health-inspection?date={row["survey_date"]}" target="_blank">View Standard Report</a>')
            if row['complaint_deficiency'] == 'Y':
                links.append(f'<a href="https://www.medicare.gov/care-compare/inspections/pdf/nursing-home/{provnum}/health/complaint-inspection?date={row["survey_date"]}" target="_blank">View Complaint Report</a>')
            if row['infection_deficiency'] == 'Y':
                links.append(f'<a href="https://www.medicare.gov/care-compare/inspections/pdf/nursing-home/{provnum}/health/infection-control-inspection?date={row["survey_date"]}" target="_blank">View Infection Report</a>')
            return ' | '.join(links) if links else 'N/A'
        
        citations['Links'] = citations.apply(create_hyperlink, axis=1)
        
        # Generate HTML table
        citations_html = citations[['survey_date', 'survey_quarter', 'tag_number', 'severity_code', 'Links']].to_html(
            escape=False,
            index=False,
            classes='citations-table',
            columns={
                'survey_date': 'Survey Date',
                'survey_quarter': 'Survey Quarter',
                'tag_number': 'Deficiency Tag',
                'severity_code': 'Severity',
                'Links': 'Inspection Reports'
            }
        )
        
        return f"""
            <h2>Recent Citations</h2>
            <style>
            .citations-table {{
                margin-top: 1rem;
                width: 100%;
                border-collapse: collapse;
            }}
            .citations-table th {{
                background-color: #f8f9fa;
                font-weight: 600;
                padding: 8px;
                text-align: left;
                border: 1px solid #dee2e6;
            }}
            .citations-table td {{
                font-size: 0.9em;
                padding: 8px;
                border: 1px solid #dee2e6;
            }}
            .citations-table a {{
                color: #1E88E5;
                text-decoration: none;
            }}
            .citations-table a:hover {{
                text-decoration: underline;
            }}
            </style>
            {citations_html}
        """
    except Exception as e:
        return f"<h2>Recent Citations</h2><p>Error generating citations section: {str(e)}</p>"

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
                COUNTY_NAME,
                CITY
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
                'county': result[3],
                'city': result[4]
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
            
        # Generate citations section
        citations_section = generate_citations_section(provnum)
        
        # Generate the report HTML
        report_html = f"""
            <html>
            <head>
                <title>PBJ Staffing Report - {facility_info['provider_name']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .header {{ margin-bottom: 20px; }}
                    .metrics {{ margin-bottom: 20px; }}
                    .citations {{ margin-top: 20px; }}
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
                    <p>Location: {facility_info['city']}, {facility_info['state']}</p>
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
                
                <div class="citations">
                    {citations_section}
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

@st.cache_data
def load_provider_info():
    """Load and process provider information data."""
    try:
        provider_info = pd.read_csv('NH_ProviderInfo_Mar2025.csv', dtype={'CMS Certification Number (CCN)': str})
        # Rename columns to be more database-friendly
        provider_info = provider_info.rename(columns={
            'CMS Certification Number (CCN)': 'PROVNUM',
            'Ownership Type': 'OWNERSHIP_TYPE',
            'Affiliated Entity Name': 'AFFILIATED_ENTITY_NAME',
            'Affiliated Entity ID': 'AFFILIATED_ENTITY_ID',
            'Special Focus Status': 'SPECIAL_FOCUS_STATUS',
            'Abuse Icon': 'ABUSE_ICON',
            'Most Recent Health Inspection More Than 2 Years Ago': 'INSPECTION_OVER_2_YEARS',
            'Provider Changed Ownership in Last 12 Months': 'OWNERSHIP_CHANGED',
            'Overall Rating': 'OVERALL_RATING',
            'Staffing Rating': 'STAFFING_RATING',
            'Total nursing staff turnover': 'NURSE_TURNOVER',
            'Number of administrators who have left the nursing home': 'ADMIN_TURNOVER',
            'Nursing Case-Mix Index': 'CASE_MIX_INDEX',
            'Latitude': 'LATITUDE',
            'Longitude': 'LONGITUDE',
            'Processing Date': 'PROCESSING_DATE'
        })
        return provider_info
    except Exception as e:
        st.error(f"Error loading provider info: {str(e)}")
        return pd.DataFrame()

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
        formatted_city = format_title_case(facility_info['city'])

        # Add the facility information HTML
        st.markdown(f"""
            <div class="facility-info-box">
                <div class="facility-info-grid">
                    <div class="facility-info-item">
                        Provider: <strong>{formatted_provider_name}</strong>
                    </div>
                    <div class="facility-info-item">
                        CCN: <strong>{facility_info['ccn']}</strong>
                    </div>
                    <div class="facility-info-item">
                        State: <strong>{facility_info['state']}</strong>
                    </div>
                    <div class="facility-info-item">
                        County: <strong>{facility_info['county']}</strong>
                    </div>
                    <div class="facility-info-item">
                        City: <strong>{formatted_city}</strong>
                    </div>
                    <div class="facility-info-item">
                        <a href="https://www.medicare.gov/care-compare/details/nursing-home/{facility_info['ccn']}?state={facility_info['state']}" target="_blank">View on Care Compare</a>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error displaying facility info: {str(e)}")

def main() -> None:
    """Main app layout and data flow."""
    try:
        # Title with custom styling
        st.markdown("""
            <div>
                <h1 class="main-header" style="margin-bottom: 0;">Nursing Home Staffing Analysis (Beta)</h1>
                <p style="color: #666; font-size: 0.9em; margin-top: 2px;">By 320 Consulting</p>
            </div>
        """, unsafe_allow_html=True)

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
            </style>
        """, unsafe_allow_html=True)
        
        st.sidebar.title("Filters")

        # Add level selection
        level = st.sidebar.radio(
            "Select Level",
            ["National", "State", "Region", "Facility"],
            key="level_selector"
        )

        # Date range selection with all available quarters
        try:
            all_quarters = sort_quarters(national_metrics['CY_QTR'].unique())  # Oldest to newest
            all_quarters_reversed = sort_quarters(all_quarters, reverse=True)  # Newest to oldest
            
            # Create two columns for quarter selection
            st.sidebar.markdown('<div class="quarter-selectors">', unsafe_allow_html=True)
            col1, col2 = st.sidebar.columns(2)
            
            with col1:
                start_quarter = st.selectbox(
                    "From",
                    all_quarters,  # Oldest to newest
                    index=0,
                    key="start_quarter",
                    format_func=format_quarter_display
                )
                start_quarter = normalize_quarter(start_quarter)
            
            with col2:
                end_quarter = st.selectbox(
                    "To",
                    all_quarters_reversed,  # Newest to oldest
                    index=0,
                    key="end_quarter",
                    format_func=format_quarter_display
                )
                end_quarter = normalize_quarter(end_quarter)
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error loading quarters: {str(e)}")
            return
        
        # Get selected value based on level
        selected_value = None
        try:
            if level == "State":
                # Get unique states and sort them
                states = sorted(state_metrics['STATE'].unique().tolist())
                # Set default to first state
                selected_state = st.sidebar.selectbox(
                    "Select State",
                    states,
                    index=0
                )
                selected_value = selected_state
            elif level == "Region":
                # Get unique regions in sorted order
                regions = sorted(region_metrics['Region'].unique(), 
                            key=lambda x: int(x.split()[-1]))
                # Default to first region
                selected_region = st.sidebar.selectbox(
                    "Select Region",
                    regions,
                    index=0
                )
                selected_value = selected_region
            elif level == "Facility":
                # Facility search with auto-complete
                search_container = st.sidebar.container()
                search_term = search_container.text_input("Enter Facility ID or Name", key="facility_search")
                
                if search_term:
                    matching_facilities = search_facilities(search_term)
                    
                    if matching_facilities:
                        # Create a formatted display string for each facility
                        facility_options = [
                            f"{fac['PROVNUM']} - {fac['PROVNAME']}"
                            for fac in matching_facilities
                        ]
                        
                        selected_facility_display = search_container.selectbox(
                            "Select Facility",
                            facility_options,
                            key="facility_selector",
                            label_visibility="collapsed"
                        )
                        if selected_facility_display:
                            selected_value = selected_facility_display.split(" - ")[0]
                    else:
                        search_container.info("No matching facilities found")
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
                    plot_quarterly_trends(filtered_data, 
                                        state=selected_value if level == "State" else None,
                                        region=selected_value if level == "Region" else None,
                                        facility=selected_value if level == "Facility" else None)

                    # 4. Display citations
                    st.markdown("---")
                    st.markdown("### Recent Citations (Sample)")
                    citations = get_dummy_citations(selected_value)
                    if not citations.empty:
                        # Format the citations data
                        citations_display = citations[[
                            'CITATION_DATE',
                            'CITATION_NUMBER',
                            'CITATION_DESCRIPTION',
                            'CITATION_SEVERITY',
                            'CITATION_STATUS',
                            'PDF_URL'
                        ]].copy()
                        
                        # Rename columns for display
                        citations_display.columns = [
                            'Date',
                            'Citation Number',
                            'Description',
                            'Severity',
                            'Status',
                            'CMS Report'
                        ]
                        
                        # Sort by date, most recent first
                        citations_display['Date'] = pd.to_datetime(citations_display['Date'])
                        citations_display = citations_display.sort_values('Date', ascending=False)
                        
                        # Add Report Summary column with dummy PDF links
                        citations_display['Report Summary'] = citations_display.apply(
                            lambda row: f'<a href="{row["CMS Report"]}" target="_blank">View Summary</a>', 
                            axis=1
                        )
                        
                        # Create hyperlinks for CMS Reports
                        citations_display['CMS Report'] = citations_display.apply(
                            lambda row: f'<a href="{row["CMS Report"]}" target="_blank">View Report</a>', 
                            axis=1
                        )
                        
                        # Display the table with HTML
                        st.markdown(citations_display.to_html(escape=False, index=False), unsafe_allow_html=True)
                    else:
                        st.warning("No citations found for this facility.")

                    # 5. Add custom reports button at the bottom
                    st.markdown("---")
                    facility_name = get_facility_info(selected_value)
                    if facility_name:
                        # Add custom CSS for the button
                        st.markdown("""
                            <style>
                            .custom-reports-section {
                                margin: 20px 0;
                            }
                            .custom-reports-button {
                                background-color: #1E88E5;
                                color: white;
                                padding: 12px 24px;
                                border: none;
                                border-radius: 6px;
                                cursor: pointer;
                                font-weight: 500;
                                font-size: 14px;
                                transition: background-color 0.3s;
                                width: fit-content;
                                text-align: left;
                                display: inline-block;
                            }
                            .custom-reports-button:hover {
                                background-color: #1565C0;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        
                        button_text = f"Subscribe to generate custom report on {facility_name['provider_name']}"
                        st.markdown(f"""
                            <div class="custom-reports-section">
                                <button class="custom-reports-button">{button_text}</button>
                            </div>
                        """, unsafe_allow_html=True)

            # For other levels (National, State, Region)
            else:
                if not filtered_data.empty:
                    display_metrics(filtered_data, level)
                    plot_quarterly_trends(filtered_data, 
                                        state=selected_value if level == "State" else None,
                                        region=selected_value if level == "Region" else None,
                                        facility=selected_value if level == "Facility" else None)
                else:
                    st.warning("No data available for the selected filters.")
        except Exception as e:
            st.error(f"Error filtering data: {str(e)}")
            return
    except Exception as e:
        st.error(f"Error in main app: {str(e)}")

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

def generate_test_data_for_facility(conn, facility_id="015009"):
    """Generate test data for a single facility without filtering by MDSCENSUS"""
    try:
        print(f"Generating test data for facility {facility_id}...")
        
        # Get all quarters in the database
        quarters = conn.execute("""
            SELECT DISTINCT CY_QTR
            FROM staffing 
            ORDER BY CY_QTR
        """).fetchall()
        quarters = [q[0] for q in quarters]
        
        # Get data for each quarter for this facility
        facility_metrics = []
        
        for quarter in quarters:
            query = f"""
            WITH daily_metrics AS (
                SELECT 
                    PROVNUM,
                    PROVNAME,
                    STATE,
                    WORKDATE,
                    MDSCENSUS,
                    (HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN + HRS_CNA + HRS_NATRN + HRS_MEDAIDE) as total_hours,
                    (HRS_RNDON + HRS_RNADMIN + HRS_RN) as rn_hours,
                    (HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN) as nurse_care_hours,
                    (HRS_RN) as rn_care_hours,
                    (HRS_CNA + HRS_NATRN + HRS_MEDAIDE) as nurse_assistant_hours,
                    (HRS_RNDON_CTR + HRS_RNADMIN_CTR + HRS_RN_CTR + HRS_LPNADMIN_CTR + HRS_LPN_CTR + HRS_CNA_CTR + HRS_NATRN_CTR + HRS_MEDAIDE_CTR) as contract_hours,
                    (HRS_RNADMIN) as rn_admin_hours,
                    (HRS_RNDON) as rn_don_hours,
                    (HRS_LPN) as lpn_hours,
                    (HRS_LPNADMIN) as lpn_admin_hours,
                    (HRS_CNA) as cna_hours,
                    (HRS_NATRN) as natr_hours,
                    (HRS_MEDAIDE) as medaide_hours
                FROM staffing 
                WHERE CY_QTR = '{quarter}' AND PROVNUM = '{facility_id}'
            )
            SELECT
                PROVNUM,
                MAX(PROVNAME) as PROVNAME,
                MAX(STATE) as STATE,
                SUM(MDSCENSUS) as Total_Resident_Days,
                ROUND(SUM(total_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Total_HPRD,
                ROUND(SUM(rn_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_HPRD,
                ROUND(SUM(nurse_care_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Nurse_Care_HPRD,
                ROUND(SUM(rn_care_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_Care_HPRD,
                ROUND(SUM(nurse_assistant_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Nurse_Assistant_HPRD,
                ROUND(SUM(contract_hours) / NULLIF(SUM(total_hours), 0) * 100, 3) as Contract_Staff_Percentage,
                ROUND(SUM(rn_admin_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_Admin_HPRD,
                ROUND(SUM(rn_don_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_DON_HPRD,
                ROUND(SUM(lpn_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as LPN_HPRD,
                ROUND(SUM(lpn_admin_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as LPN_Admin_HPRD,
                ROUND(SUM(cna_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as CNA_HPRD,
                ROUND(SUM(natr_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as NAtr_HPRD,
                ROUND(SUM(medaide_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as MedAide_HPRD,
                COUNT(*) as day_count
            FROM daily_metrics
            GROUP BY PROVNUM
            """
            
            result = conn.execute(query).fetchall()
            if result and len(result) > 0:
                row = result[0]
                facility_metrics.append({
                    'PROVNUM': row[0],
                    'PROVNAME': row[1],
                    'STATE': row[2],
                    'Total_Resident_Days': float(row[3]) if row[3] is not None else 0.0,
                    'Total_HPRD': float(row[4]) if row[4] is not None else 0.0,
                    'RN_HPRD': float(row[5]) if row[5] is not None else 0.0,
                    'Nurse_Care_HPRD': float(row[6]) if row[6] is not None else 0.0,
                    'RN_Care_HPRD': float(row[7]) if row[7] is not None else 0.0,
                    'Nurse_Assistant_HPRD': float(row[8]) if row[8] is not None else 0.0,
                    'Contract_Staff_Percentage': float(row[9]) if row[9] is not None else 0.0,
                    'RN_Admin_HPRD': float(row[10]) if row[10] is not None else 0.0,
                    'RN_DON_HPRD': float(row[11]) if row[11] is not None else 0.0,
                    'LPN_HPRD': float(row[12]) if row[12] is not None else 0.0,
                    'LPN_Admin_HPRD': float(row[13]) if row[13] is not None else 0.0,
                    'CNA_HPRD': float(row[14]) if row[14] is not None else 0.0,
                    'NAtr_HPRD': float(row[15]) if row[15] is not None else 0.0,
                    'MedAide_HPRD': float(row[16]) if row[16] is not None else 0.0,
                    'Day_Count': int(row[17]) if row[17] is not None else 0,
                    'CY_QTR': quarter
                })
                print(f"Found data for {quarter} with {row[17]} days")
            else:
                print(f"No data found for {quarter}")
                
        if facility_metrics:
            # Convert to DataFrame
            df = pd.DataFrame(facility_metrics)
            
            # Create date column
            df['date'] = df['CY_QTR'].apply(lambda x: pd.to_datetime(f"{x[:4]}-{int(x[-1])*3}-15"))
            
            # Get most recent name
            latest_name = None
            if not df.empty:
                latest_record = df.sort_values('date', ascending=False).iloc[0]
                latest_name = latest_record['PROVNAME']
                print(f"Most recent name: {latest_name}")
                
                # Use latest name for all records
                if latest_name:
                    df['PROVNAME'] = latest_name
            
            # Save to CSV
            output_file = f'facility_{facility_id}_test.csv'
            df.to_csv(output_file, index=False)
            print(f"\nTest data saved to {output_file}")
            print(f"Generated {len(df)} quarters of data")
            print("\nQuarters with data:")
            print(df['CY_QTR'].tolist())
            return df
        else:
            print("No data found for this facility")
            return None
            
    except Exception as e:
        print(f"Error generating test data: {str(e)}")
        return None

def get_dummy_provider_info(provnum: str) -> dict:
    """Get dummy provider information."""
    return {
        'PROVNUM': provnum,
        'PROVNAME': 'Sample Nursing Home',
        'OWNERSHIP_TYPE': 'For-Profit Corporation',
        'AFFILIATED_ENTITY_NAME': '••••••••••',
        'AFFILIATED_ENTITY_ID': '••••••••••',
        'SPECIAL_FOCUS_STATUS': 'None',
        'ABUSE_ICON': 'N',
        'INSPECTION_OVER_2_YEARS': 'N',
        'OWNERSHIP_CHANGED': 'N',
        'OVERALL_RATING': 4,
        'STAFFING_RATING': 3,
        'NURSE_TURNOVER': 25.5,
        'ADMIN_TURNOVER': 1,
        'CASE_MIX_INDEX': 1.25,
        'LATITUDE': 40.7128,
        'LONGITUDE': -74.0060,
        'PROCESSING_DATE': '2024-03-15'
    }

def get_dummy_citations(provnum: str) -> pd.DataFrame:
    """Get dummy citations data."""
    citations = [
        {
            'CITATION_DATE': '2024-02-15',
            'CITATION_NUMBER': 'F689',
            'CITATION_DESCRIPTION': '••••••••••',
            'CITATION_SEVERITY': 'D',
            'CITATION_STATUS': 'Active',
            'PDF_URL': f'https://example.com/citations/{provnum}/F689.pdf'
        },
        {
            'CITATION_DATE': '2023-11-30',
            'CITATION_NUMBER': 'F686',
            'CITATION_DESCRIPTION': '••••••••••',
            'CITATION_SEVERITY': 'E',
            'CITATION_STATUS': 'Active',
            'PDF_URL': f'https://example.com/citations/{provnum}/F686.pdf'
        },
        {
            'CITATION_DATE': '2023-08-15',
            'CITATION_NUMBER': 'F684',
            'CITATION_DESCRIPTION': '••••••••••',
            'CITATION_SEVERITY': 'D',
            'CITATION_STATUS': 'Active',
            'PDF_URL': f'https://example.com/citations/{provnum}/F684.pdf'
        }
    ]
    return pd.DataFrame(citations)

def display_facility_citations(provnum: str):
    """Display citations in a formatted table."""
    try:
        citations = get_dummy_citations(provnum)
        if not citations.empty:
            # Format the citations data
            citations_display = citations[[
                'CITATION_DATE',
                'CITATION_NUMBER',
                'CITATION_DESCRIPTION',
                'CITATION_SEVERITY',
                'CITATION_STATUS',
                'PDF_URL'
            ]].copy()
            
            # Rename columns for display
            citations_display.columns = [
                'Date',
                'Citation Number',
                'Description',
                'Severity',
                'Status',
                'PDF'
            ]
            
            # Sort by date, most recent first
            citations_display['Date'] = pd.to_datetime(citations_display['Date'])
            citations_display = citations_display.sort_values('Date', ascending=False)
            
            # Create hyperlinks for PDFs
            def create_hyperlink(row):
                return f'<a href="{row["PDF"]}" target="_blank">View PDF</a>'
            
            citations_display['PDF'] = citations_display.apply(create_hyperlink, axis=1)
            
            # Display the table with HTML
            st.markdown(citations_display.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.warning("No citations found for this facility.")
    except Exception as e:
        st.error(f"Error displaying citations: {str(e)}")

if __name__ == "__main__":
    main()