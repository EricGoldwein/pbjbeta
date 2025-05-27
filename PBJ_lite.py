import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import re
from plotly.subplots import make_subplots
import duckdb
from typing import Dict, Optional, List, Tuple, Any

# Set page configuration with a more professional theme
st.set_page_config(
    page_title="PBJ Dashboard (Beta)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Add custom CSS to reduce sidebar width and style the page name
st.markdown("""
    <style>
        [data-testid="stSidebar"][aria-expanded="true"]{
            width: 250px !important;
        }
        [data-testid="stSidebar"][aria-expanded="false"]{
            width: 250px !important;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize DuckDB connection for facility data
facility_db = duckdb.connect(':memory:')

# Initialize provider info cache
provider_info_cache: Dict[str, Dict[str, str]] = {}

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
                        <a href="https://www.medicare.gov/care-compare/details/nursing-home/{facility_info['ccn']}?state={facility_info['state']}" target="_blank">View on Care Compare</a>
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
            <h3>320 Premium Reports</h3>
            <p>320 Consulting offers custom reports with full breakdowns of all nurse and non-nurse positions, staffing trends over time, ownership data, citation histories, and comparisons by geography or any category you need — built to support your case, investigation, or advocacy work.</p>
            <p>To request a report or talk through what you need:</p>
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

        # Display Key Metrics header with level-specific title
        if level == "State":
            state = metrics['STATE'].iloc[0]
            facility_count = current_metrics['Facility_Count'].iloc[0] if 'Facility_Count' in current_metrics else len(current_metrics['PROVNUM'].unique())
            prev_facility_count = prev_metrics['Facility_Count'].iloc[0] if not prev_metrics.empty and 'Facility_Count' in prev_metrics else None
            header_text = f"{state} Key Metrics ({quarter_name})"
        elif level == "National":
            facility_count = current_metrics['Facility_Count'].iloc[0] if 'Facility_Count' in current_metrics else len(current_metrics['PROVNUM'].unique())
            prev_facility_count = prev_metrics['Facility_Count'].iloc[0] if not prev_metrics.empty and 'Facility_Count' in prev_metrics else None
            header_text = f"Key Metrics ({quarter_name})"
        else:  # Facility level
            provnum = current_metrics['PROVNUM'].iloc[0]
            provname = proper_title_case(current_metrics['PROVNAME'].iloc[0])
            state = current_metrics['STATE'].iloc[0]
            county = proper_title_case(current_metrics['COUNTY_NAME'].iloc[0])
            care_compare_url = f"https://www.medicare.gov/care-compare/details/nursing-home/{provnum}/view-all?state={state}"
            header_text = f"{provname} ({county}) | {quarter_name} | <a href='{care_compare_url}' target='_blank' style='color: #1E88E5; text-decoration: none; font-weight: 500;'>View on Care Compare</a>"
            
        with header_col1:
            st.markdown(f'<div class="section-header" style="margin-top: 8px;">{header_text}</div>', unsafe_allow_html=True)
            
        # Add custom CSS for metrics containers
        st.markdown("""
            <style>
            .metrics-container {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            }
            .stMetric {
                background-color: white !important;
                border: 1px solid #e0e0e0 !important;
                border-radius: 6px !important;
                padding: 15px !important;
                transition: all 0.2s ease-in-out !important;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
            }
            .stMetric:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1) !important;
            }
            .stMetric [data-testid="stMetricValue"] {
                font-size: 1.2em !important;
                font-weight: 500 !important;
            }
            .stMetric [data-testid="stMetricLabel"] {
                font-size: 0.9em !important;
                color: #666 !important;
            }
            .stMetric [data-testid="stMetricDelta"] {
                font-size: 0.9em !important;
            }
            /* Add styles for quarter selector */
            div[data-testid="stSelectbox"] {
                width: 120px !important;
            }
            @media (max-width: 768px) {
                .mobile-metrics {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 8px;
                    margin: 0 -8px;
                }
                .mobile-metrics .stMetric {
                    margin: 0;
                    padding: 8px !important;
                }
                .mobile-metrics .stMetric [data-testid="stMetricValue"] {
                    font-size: 16px !important;
                }
                .mobile-metrics .stMetric [data-testid="stMetricLabel"] {
                    font-size: 12px !important;
                }
                .mobile-metrics .stMetric [data-testid="stMetricDelta"] {
                    font-size: 12px !important;
                }
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Display metrics in columns
        if level in ["National", "State"]:
            col1, col2, col3, col4 = st.columns(4)
        else:
            col1, col2, col3 = st.columns(3)
        
        # For National and State, add facility count metric
        if level in ["National", "State"]:
            with col1:
                st.metric("Nursing Homes", 
                         format_metric(facility_count, decimal_places=0, thousands=True),
                         format_metric(facility_count - prev_facility_count, decimal_places=0, thousands=True) if prev_facility_count is not None else None)
        
        # Adjust column indices for other metrics
        metric_cols = [col2, col3, col4] if level in ["National", "State"] else [col1, col2, col3]
        
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
            st.metric("Contract Staff %", 
                     format_metric(current_metrics['Contract_Percentage'].iloc[0], decimal_places=1, percentage=True),
                     format_metric(current_metrics['Contract_Percentage'].iloc[0] - prev_metrics['Contract_Percentage'].iloc[0], decimal_places=1, percentage=True) if not prev_metrics.empty else None)
            
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
        if state:
            title_prefix = f"{state}"
            data = df[df['STATE'] == state].copy()
        elif facility:
            facility_name = get_provider_info(facility, 'name')
            facility_state = get_provider_info(facility, 'state')
            if facility_name and facility_state:
                title_prefix = f"{facility_name} ({facility_state})"
            else:
                title_prefix = f"Facility {facility}"
            data = df[df['PROVNUM'] == facility].copy()
        else:
            title_prefix = "National"
            data = df.copy()
        
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
        
        # Define hover template
        hover_template = "<b>%{customdata}</b><br>Value: %{y:.2f}<extra></extra>"
        
        # Desktop figure (3 charts)
        fig = make_subplots(rows=3, cols=1,
                  subplot_titles=('Total Nurse HPRD', 'Census', 'Contract Staff Percentage'),
                          vertical_spacing=0.15)

        # Add all traces for desktop view
        fig.add_trace(go.Scatter(x=data['date'], y=data['Total_Nurse_HPRD'],
                       mode='lines+markers', name='Total HPRD',
                       customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                       hovertemplate=hover_template), row=1, col=1)

        fig.add_trace(go.Scatter(x=data['date'], y=data['Census'],
                       mode='lines+markers', name='Census',
                       customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                       hovertemplate=hover_template.replace(':.2f', ':,.0f')), row=2, col=1)

        fig.add_trace(go.Scatter(x=data['date'], y=data['Contract_Percentage'],
                       mode='lines+markers', name='Contract %',
                       customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                       hovertemplate=hover_template.replace(':.2f', ':.1f%')), row=3, col=1)

        # Update desktop layout
        fig.update_layout(
            height=1400,
            width=1000,
            title_text=f"{title_prefix} Staffing Trends",
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=100),
            hovermode='x unified'
        )
        
        # Add footer annotations for desktop view
        for row in range(1, 4):
            fig.add_annotation(
                text="320 Consulting | Source: CMS PBJ Data",
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
            st.switch_page("pages/2_Facility_Search.py")

        # Get URL parameters using the new API
        initial_level = st.query_params.get('level', 'National')
        initial_facility = st.query_params.get('facility', None)

        # Title with custom styling
        st.markdown("""
            <div>
                <h1 class="main-header" style="margin-bottom: 0;">PBJ Dashboard (Beta)</h1>
                <p style="color: #666; font-size: 0.9em; margin-top: 2px;">
                    By 320 Consulting | 
                    <a href="?page=premium" target="_self" style="color: #1E88E5; text-decoration: none; font-weight: 500;">
                        ⭐ Premium
                    </a> |
                    <a href="?page=facility_search" target="_self" style="color: #1E88E5; text-decoration: none; font-weight: 500;">
                        🔍 Search Facility
                    </a>
                </p>
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
            </style>
        """, unsafe_allow_html=True)

        # Add level selection with initial value from URL
        level = st.sidebar.radio(
            "Select Level",
            ["National", "State", "Facility"],
            index=["National", "State", "Facility"].index(initial_level),
            key="level_selector"
        )

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
                    '<a href="?page=facility_search" target="_self" style="color: #1E88E5; text-decoration: none; font-size: 0.9em;">'
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


if __name__ == "__main__":
    main()