import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import os
import re
from plotly.subplots import make_subplots
import duckdb
from typing import Dict, Optional, List, Tuple, Any
import base64
import math
import io
import numpy as np

# Add this import at the top of your file, after the other imports
from pbj_icon_component import pbj_icon, pbj_icon_with_text  # Uncomment when you want to use the component

# Set sidebar collapsed on mobile
import streamlit as st
st.set_page_config(page_title="PBJ Nursing Home Staffing Dashboard by 320", page_icon="pbj_favicon.png", layout="wide", initial_sidebar_state="auto")

# Add subtle modern styling for metric containers only (not delta or value)
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background: #fafdff;
        border: 1px solid #e3eaf3;
        border-radius: 10px;
        box-shadow: 0 1px 4px rgba(30,136,229,0.04);
        padding: 12px 18px 4px 18px;
        margin: 12px 4px 6px 4px;
    }
    
    /* Override delta colors for neutral indicators */
    div[data-testid="stMetric"] div[data-testid="metric-container"] div[data-testid="metric-delta"] svg {
        color: #6c757d !important;
    }
    div[data-testid="stMetric"] div[data-testid="metric-container"] div[data-testid="metric-delta"] span {
        color: #6c757d !important;
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

@st.cache_data
def load_facility_data():
    """Load facility data for search."""
    try:
        return pd.read_csv('facility_lite_metrics.csv', dtype={'PROVNUM': str})
    except Exception as e:
        st.error(f"Error loading facility data: {str(e)}")
        return pd.DataFrame()

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

def _classify(v, ref, tol=0.03):
    if ref == 0 or v is None or ref is None:
        return "—"
    if v > ref * (1 + tol): return "above"
    if v < ref * (1 - tol): return "below"
    return "around"

def _fmt(x, nd=2):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (ValueError, TypeError):
        return str(x)

def pbj_takeaway_card(
    facility: str,
    reported_hprd: float,
    quarter_label: str,   # e.g., "Q1 2025"
    state_name: str,
    state_hprd: float,
    casemix_hprd: float,
    census: str = "—",
    contract: str = "—",
    trend_delta: float = None,  # Change from previous quarter
    census_trend: float = None,  # Change in census from previous quarter
    previous_quarter: str = "Q1 2024",  # Previous quarter for comparison
    aide_share: float = 0.60,  # or compute from PBJ if you have it
    floor_beds: int = 30
):
    # Force proper title case for facility name to ensure "At" is lowercase
    facility = proper_title_case(facility)
    # Core calcs
    res_per_staff = 24.0 / reported_hprd if reported_hprd else None
    floor_staff_total = (floor_beds * reported_hprd / 24.0) if reported_hprd else None
    floor_staff_aides = (floor_staff_total * aide_share) if floor_staff_total else None
    coverage = (reported_hprd / casemix_hprd) if (reported_hprd and casemix_hprd) else None

    vs_state = _classify(reported_hprd, state_hprd)
    vs_cmix  = _classify(reported_hprd, casemix_hprd)

    # Styles
    def chip(label, value, tone="neutral"):
        colors = {
            "good":   ("#065f46", "#ecfdf5"),
            "warn":   ("#7c2d12", "#fff7ed"),
            "neutral":("#334155", "#f1f5f9")
        }
        fg, bg = colors["neutral" if tone not in colors else tone]
        return f"""<span style="display:inline-block;padding:2px 8px;border-radius:999px;
                 background:{bg};color:{fg};font-weight:600;font-size:0.85rem;margin-right:6px;">
                 {label}: {value}</span>"""

    with st.container(border=True):
        # Add PBJ icon to the container - more compact
        try:
            with open('pbj_favicon.png', 'rb') as f:
                favicon_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <img src="data:image/png;base64,{favicon_data}" style="width: 24px; height: 24px; margin-right: 8px;">
                <strong style="font-size: 16px;">PBJ Takeaway: {facility}</strong>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            # Fallback without favicon if file can't be read
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <strong style="font-size: 16px;">PBJ Takeaway: {facility}</strong>
            </div>
            """, unsafe_allow_html=True)

        
        # Header chips
        tone_state = "neutral"
        tone_cmix = "neutral"
        trend_emoji = "📈" if trend_delta and trend_delta > 0 else ("📉" if trend_delta and trend_delta < 0 else "")
        st.markdown(
            chip("Reported HPRD", f"{_fmt(reported_hprd)} {trend_emoji}") +
            chip(f"{state_name} HPRD", f"{_fmt(state_hprd)}", tone_state) +
            chip("Census", census) +
            chip("Contract", contract),
            unsafe_allow_html=True
        )

        # Narrative
        # Pick "above/below/around" words
        word_state = {"above":"above", "below":"below", "around":"around", "—":"—"}[vs_state]
        word_cmix  = {"above":"above", "below":"below", "around":"around", "—":"—"}[vs_cmix]

        # Build trend sentences
        hprd_trend_text = ""
        if trend_delta is not None:
            trend_direction = "up" if trend_delta > 0 else "down"
            hprd_trend_text = f" HPRD is {trend_direction} {_fmt(abs(trend_delta))} since Q1 2024"
        
        census_trend_text = ""
        if census_trend is not None:
            census_direction = "up" if census_trend > 0 else "down"
            census_trend_text = f" Census is {census_direction} {_fmt(abs(census_trend), 1)} since Q1 2024"
        
        # Convert census to int if it's a string for calculations
        census_int = int(census) if isinstance(census, str) and census.isdigit() else (census if isinstance(census, (int, float)) else 120)
        
        para = (
            f"**{facility}**'s reported **{_fmt(reported_hprd)} hours per resident day** "
            f"(≈ {_fmt(res_per_staff,1)} residents per total staff) in {quarter_label}{hprd_trend_text}{census_trend_text}. "
            f"This level is {word_state} the {state_name} ratio of {_fmt(state_hprd)} HPRD "
            f"and {word_cmix} the case-mix (expected) {_fmt(casemix_hprd)} HPRD given resident acuity."
        )
        
        st.markdown(para)
        st.markdown(f"**Put another way...** On a typical **30-bed floor** at {facility} you'd see about **{_fmt(floor_staff_total,1)} staff members**, including ~{_fmt(floor_staff_aides,1)} nurse aides. For the entire {census}-resident facility, that's about {_fmt(census_int * reported_hprd / 24.0,1)} total staff, including ~{_fmt(census_int * reported_hprd / 24.0 * aide_share,1)} nurse aides.")

        # Note
        st.markdown("*Note: staffing varies by day and shift, with the lowest levels typically on nights and weekends.*")
        
        # Add 320 Consulting badge
        st.markdown("""
        <div style="position: absolute; bottom: -8px; right: 2px; background: linear-gradient(135deg, #333333 0%, #666666 100%); color: white; padding: 2px 5px; border-radius: 6px; font-size: 0.6em; font-weight: 600; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
            320 Consulting
        </div>
        <style>
        @media (max-width: 768px) {
            /* Force container to allow absolute positioning */
            div[data-testid="stContainer"] {
                padding: 0px !important;
                margin: 0px !important;
                position: relative !important;
                overflow: visible !important;
            }
            /* Override Streamlit's container styling */
            .stContainer {
                padding: 0px !important;
                margin: 0px !important;
            }
            /* Make badge truly stick to bottom */
            div[style*="320 Consulting"] {
                position: fixed !important;
                bottom: 10px !important;
                right: 5px !important;
                margin: 0 !important;
                padding: 3px 7px !important;
                font-size: 0.65em !important;
                z-index: 9999 !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        


def state_pbj_takeaway_card(
    state_name: str,
    reported_hprd: float,
    quarter_label: str,   # e.g., "Q1 2025"
    national_hprd: float,
    state_rank: int,
    total_states: int,
    trend_delta: float = None,  # Change from previous quarter
    previous_quarter: str = "Q1 2024",  # Previous quarter for comparison
    aide_share: float = 0.60,  # or compute from PBJ if you have it
    floor_beds: int = 30,
    avg_facility_size: float = 100  # Average facility size for the state
):
    # Core calcs
    res_per_staff = 24.0 / reported_hprd if reported_hprd else None
    floor_staff_total = (floor_beds * reported_hprd / 24.0) if reported_hprd else None
    floor_staff_aides = (floor_staff_total * aide_share) if floor_staff_total else None

    vs_national = _classify(reported_hprd, national_hprd)

    # Styles
    def chip(label, value, tone="neutral"):
        colors = {
            "good":   ("#065f46", "#ecfdf5"),
            "warn":   ("#7c2d12", "#fff7ed"),
            "neutral":("#334155", "#f1f5f9")
        }
        fg, bg = colors["neutral" if tone not in colors else tone]
        return f"""<span style="display:inline-block;padding:2px 8px;border-radius:999px;
                 background:{bg};color:{fg};font-weight:600;font-size:0.85rem;margin-right:6px;">
                 {label}: {value}</span>"""

    with st.container(border=True):
        # Add PBJ icon to the container - more compact
        try:
            with open('pbj_favicon.png', 'rb') as f:
                favicon_data = base64.b64encode(f.read()).decode()
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <img src="data:image/png;base64,{favicon_data}" style="width: 24px; height: 24px; margin-right: 8px;">
                <strong style="font-size: 16px;">PBJ Takeaway: {state_name}</strong>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            # Fallback without favicon if file can't be read
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <strong style="font-size: 16px;">PBJ Takeaway: {state_name}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Header chips
        tone_national = "neutral"  # Always neutral for National HPRD
        trend_emoji = "📈" if trend_delta and trend_delta > 0 else ("📉" if trend_delta and trend_delta < 0 else "")
        st.markdown(
            chip(f"{state_name} HPRD", f"{_fmt(reported_hprd)} {trend_emoji}") +
            chip("National HPRD", f"{_fmt(national_hprd)}", tone_national) +
            chip("State Rank", f"#{state_rank} of {total_states}"),
            unsafe_allow_html=True
        )

        # Narrative
        # Pick "above/below/around" words
        word_national = {"above":"above", "below":"below", "around":"around", "—":"—"}[vs_national]

        # Build trend sentences
        hprd_trend_text = ""
        if trend_delta is not None:
            trend_direction = "up" if trend_delta > 0 else "down"
            hprd_trend_text = f" HPRD is {trend_direction} {_fmt(abs(trend_delta))} since {previous_quarter}"

        para = (
            f"**{state_name}**'s reported **{_fmt(reported_hprd)} hours per resident day** "
            f"(≈ {_fmt(res_per_staff,1)} residents per total staff) in {quarter_label}{hprd_trend_text}. "
            f"This level is {word_national} the national ratio of {_fmt(national_hprd)} HPRD "
            f"and ranks **#{state_rank}** out of {total_states} states."
        )
        
        st.markdown(para)
        # Calculate staff for average facility size
        avg_facility_staff_total = (avg_facility_size * reported_hprd / 24.0) if reported_hprd else None
        avg_facility_staff_aides = (avg_facility_staff_total * aide_share) if avg_facility_staff_total else None
        
        st.markdown(f"**Put another way...** On a typical **30-bed floor** at a typical {state_name} nursing home you'd see about **{_fmt(floor_staff_total,1)} staff members**, including ~{_fmt(floor_staff_aides,1)} nurse aides. For the entire {int(avg_facility_size)}-resident facility, that's about {_fmt(avg_facility_staff_total,1)} total staff, including ~{_fmt(avg_facility_staff_aides,1)} nurse aides.")

        # Note
        st.markdown("*Note: staffing varies by day and shift, with the lowest levels typically on nights and weekends.*")
        
        # Add 320 Consulting badge
        st.markdown("""
        <div style="position: absolute; bottom: -8px; right: 2px; background: linear-gradient(135deg, #333333 0%, #666666 100%); color: white; padding: 2px 5px; border-radius: 6px; font-size: 0.6em; font-weight: 600; box-shadow: 0 1px 2px rgba(0,0,0,0.2);">
            320 Consulting
        </div>
        <style>
        @media (max-width: 768px) {
            /* Force container to allow absolute positioning */
            div[data-testid="stContainer"] {
                padding: 0px !important;
                margin: 0px !important;
                position: relative !important;
                overflow: visible !important;
            }
            /* Override Streamlit's container styling */
            .stContainer {
                padding: 0px !important;
                margin: 0px !important;
            }
            /* Make badge truly stick to bottom */
            div[style*="320 Consulting"] {
                position: fixed !important;
                bottom: 10px !important;
                right: 5px !important;
                margin: 0 !important;
                padding: 3px 7px !important;
                font-size: 0.65em !important;
                z-index: 9999 !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        


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
    """Load and cache chain performance measures data."""
    try:
        df = pd.read_csv('Nursing_Home_Chain_Performance_Measures_Jul_2025.csv')
        
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
        df = pd.read_csv('NH_ProviderInfo_Jul2025.csv', dtype={'CMS Certification Number (CCN)': str})
        
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
    """Load and cache July 2025 provider information data for comparison."""
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
        st.error(f"Error loading June provider info data: {str(e)}")
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
            query = """
                SELECT DISTINCT PROVNAME, STATE, COUNTY_NAME
                FROM facility_metrics 
                WHERE PROVNUM = ?
                LIMIT 1
            """
            result = facility_db.execute(query, (provnum,)).fetchdf()
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
    """Get the chain name for a specific facility from provider info data."""
    try:
        provider_data = load_provider_info_data()
        if provider_data.empty:
            return None
            
        # Find the facility by CCN
        facility_data = provider_data[provider_data['CMS Certification Number (CCN)'] == provnum]
        
        if facility_data.empty:
            return None
            
        # Get the chain name
        chain_name = facility_data.iloc[0]['Chain Name']
        
        # Return None if it's NaN, otherwise return the chain name with proper title case
        if pd.notna(chain_name):
            return proper_title_case(str(chain_name))
        return None
        
    except Exception as e:
        print(f"Error getting chain name for {provnum}: {str(e)}")
        return None

@st.cache_data
def get_facility_affiliated_entity_id(provnum: str) -> str:
    """Get the chain ID for a specific facility from provider info data."""
    try:
        provider_data = load_provider_info_data()
        if provider_data.empty:
            return None
            
        # Find the facility by CCN
        facility_data = provider_data[provider_data['CMS Certification Number (CCN)'] == provnum]
        
        if facility_data.empty:
            return None
            
        # Get the chain ID
        chain_id = facility_data.iloc[0]['Chain ID']
        
        # Return None if it's NaN, otherwise return the chain ID
        if pd.notna(chain_id):
            return str(int(chain_id))
        return None
        
    except Exception as e:
        print(f"Error getting chain ID for {provnum}: {str(e)}")
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
    """Get the staffing rating trend by comparing current vs June 2025 data."""
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
            return f"+{current_rating - march_rating}"
        elif current_rating < march_rating:
            return f"{current_rating - march_rating}"
        else:
            return "—"  # Neutral dash for no change
        
    except Exception as e:
        return None

@st.cache_data
def get_facility_overall_rating_trend(provnum: str) -> str:
    """Get the overall rating trend by comparing current vs June 2025 data."""
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
            return f"+{current_rating - march_rating}"
        elif current_rating < march_rating:
            return f"{current_rating - march_rating}"
        else:
            return "—"  # Neutral dash for no change
        
    except Exception as e:
        return None

@st.cache_data
def get_filtered_data(level: str, selected_value: str, start_quarter: str, end_quarter: str):
    """Get filtered data with optimized filtering."""
    try:
        if level == "Facility" and selected_value:
            # Use DuckDB for facility-level data with parameterized query
            query = """
                SELECT * FROM facility_metrics 
                WHERE PROVNUM = ?
                AND CY_QTR >= ?
                AND CY_QTR <= ?
                ORDER BY date
            """
            return facility_db.execute(query, (selected_value, start_quarter, end_quarter)).fetchdf()
        
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
            state = metrics['STATE'].iloc[0]
            facility_count = current_metrics['Facility_Count'].iloc[0] if 'Facility_Count' in current_metrics else len(current_metrics['PROVNUM'].unique())
            prev_facility_count = prev_metrics['Facility_Count'].iloc[0] if not prev_metrics.empty and 'Facility_Count' in prev_metrics else None
            full_state_name = get_full_state_name(state)
            header_text = f"{full_state_name} Key Metrics ({quarter_name})"
            # Handle District of Columbia abbreviation for mobile
            mobile_state_name = "D.C." if full_state_name == "District of Columbia" else full_state_name
            mobile_header_text = f"{mobile_state_name} ({quarter_name})"
            # Display state header
            st.markdown(f'''
                <div class="section-header" style="margin-top: 8px; font-size: 1.35em; font-weight: 700; color: #1976d2; border-bottom: 2.5px solid #e3eaf3; padding-bottom: 4px; letter-spacing: 0.01em;">
                    <div style='font-size: 1.35em; font-weight: 700; color: #1976d2;' class="desktop-header">{header_text}</div>
                    <div style='font-size: 1.35em; font-weight: 700; color: #1976d2;' class="mobile-header">{mobile_header_text}</div>
                </div>
                <style>
                .mobile-header {{
                    display: none;
                }}
                @media (max-width: 768px) {{
                    .desktop-header {{
                        display: none;
                    }}
                    .mobile-header {{
                        display: block;
                    }}
                }}
                /* Additional padding for state page on mobile */
                .state-page-mobile-padding {{
                    margin-top: 15px !important;
                }}
                /* Additional padding for state page on desktop */
                @media (min-width: 768px) {{
                    .state-page-mobile-padding {{
                        margin-top: 25px !important;
                    }}
                }}
                </style>
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
            
            if affiliated_entity and affiliated_entity_id:
                st.markdown(f'''
                    <div class="section-header" style="margin-top: 2px; font-size: 1.35em; font-weight: 700; color: #1976d2; border-bottom: 2.5px solid #e3eaf3; padding-bottom: 4px; letter-spacing: 0.01em;">
                        <div style='color:#222; font-weight:400;'>
                            <div style='font-size: 1.35em; font-weight: 700; color: #1976d2;'>{provname} ({quarter_name})</div>
                            <div style='font-size: 0.9em; color: #666; margin-top: 4px;'>
                                {county}, <a href='?state={state}' style='color: #1976d2; text-decoration: none;' target='_self'>{state}</a>. Ownership: <a href='?entity={affiliated_entity_id}' style='color: #1976d2; text-decoration: none;' target='_self'>{affiliated_entity}</a>
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                    <div class="section-header" style="margin-top: 2px; font-size: 1.35em; font-weight: 700; color: #1976d2; border-bottom: 2.5px solid #e3eaf3; padding-bottom: 4px; letter-spacing: 0.01em;">
                        <div style='color:#222; font-weight:400;'>
                            <div style='font-size: 1.35em; font-weight: 700; color: #1976d2;'>{provname} ({quarter_name})</div>
                            <div style='font-size: 0.9em; color: #666; margin-top: 4px;'>
                                {county}, <a href='?state={state}' style='color: #1976d2; text-decoration: none;' target='_self'>{state}</a>. Ownership: N/A
                            </div>
                        </div>
                    </div>
                ''', unsafe_allow_html=True)
            
        # Add custom CSS for metrics containers
        # (Removed custom CSS for stMetric, stMetricDelta, stMetricContainer to restore Streamlit defaults)
        
                                        # Add CSS to prevent metric container borders from being cut off
        st.markdown("""
            <style>
            /* Add padding to prevent metric container borders from being cut off */
            div[data-testid="stMetric"] {
                padding-bottom: 12px !important;
                margin-bottom: 16px !important;
                overflow: visible !important;
            }
            /* Additional padding for mobile */
            @media (max-width: 768px) {
                div[data-testid="stMetric"] {
                    padding-bottom: 14px !important;
                    margin-bottom: 18px !important;
                }
            }
            /* Ensure metric containers have proper spacing from charts below */
            div[data-testid="stMetric"] + div[data-testid="stPlotlyChart"] {
                margin-top: 25px !important;
                padding-top: 20px !important;
            }
            /* Additional spacing for mobile */
            @media (max-width: 768px) {
                div[data-testid="stMetric"] + div[data-testid="stPlotlyChart"] {
                    margin-top: 20px !important;
                    padding-top: 15px !important;
                }
            }
            </style>
        """, unsafe_allow_html=True)
        
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
            
            # Calculate delta for census
            census_delta = census_value - prev_census_value if prev_census_value is not None else None
            
            # Show neutral indicator for zero deltas
            if census_delta == 0:
                delta_display = "—"  # Neutral dash
            elif census_delta is not None:
                delta_display = format_metric(census_delta, decimal_places=0, thousands=True)
            else:
                delta_display = None
            
            st.metric("Resident Census", 
                     format_metric(census_value, decimal_places=0, thousands=True),
                     delta_display,
                     help=help_text)
        
        with metric_cols[1]:
            # Calculate delta for HPRD
            hprd_delta = current_metrics['Total_Nurse_HPRD'].iloc[0] - prev_metrics['Total_Nurse_HPRD'].iloc[0] if not prev_metrics.empty else None
            
            # Show neutral indicator for zero deltas
            if hprd_delta == 0:
                delta_display = "—"  # Neutral dash
            elif hprd_delta is not None:
                delta_display = format_metric(hprd_delta, decimal_places=2)
            else:
                delta_display = None
                
            st.metric("Nurse Staffing (HPRD)", 
                     format_metric(current_metrics['Total_Nurse_HPRD'].iloc[0], decimal_places=2),
                     delta_display,
                     help="Total nurse staff hours per resident per day. Example: A nursing home with 100 residents providing 350 staffing hours per day has 3.5 nurse staff HPRD (350 ÷ 100). Arrow compares to previous quarter.")
        
        with metric_cols[2]:
            # Calculate delta for contract percentage
            contract_delta = current_metrics['Contract_Percentage'].iloc[0] - prev_metrics['Contract_Percentage'].iloc[0] if not prev_metrics.empty else None
            
            # Show neutral indicator for zero deltas
            if contract_delta == 0:
                delta_display = "—"  # Neutral dash
            elif contract_delta is not None:
                delta_display = format_metric(contract_delta, decimal_places=1, percentage=True)
            else:
                delta_display = None
                
            st.metric(
                "Contract Staff %",
                format_metric(current_metrics['Contract_Percentage'].iloc[0], decimal_places=1, percentage=True),
                delta_display,
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
                             help="5-star rating determined by federal CMS (July 2025 vs. June 2025).")
                else:
                    st.metric("CMS Staffing Rating", 
                             "N/A",
                             staffing_trend,
                             help="5-star rating determined by federal CMS (July 2025 vs. June 2025).")
            

            
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


def create_case_mix_charts(provnum, quarter_label="", facility_name=""):
    """Create case-mix comparison charts for a facility."""
    try:
        # Load provider info data
        provider_df = pd.read_csv('NH_ProviderInfo_Jul2025.csv')
        
        # Find the facility by PROVNUM
        facility_data = provider_df[provider_df['CMS Certification Number (CCN)'] == provnum]
        
        if facility_data.empty:
            return None, None
            
        row = facility_data.iloc[0]
        
        # Define the order we want
        categories = ["Total Nursing", "RN", "Nurse Aide"]
        
        # Get the data directly from the row in the correct order
        deltas = []
        delta_percentages = []
        reported_values = []
        case_mix_values = []
        
        for category in categories:
            if category == "Total Nursing":
                reported_val = row["Reported Total Nurse Staffing Hours per Resident per Day"]
                case_mix_val = row["Case-Mix Total Nurse Staffing Hours per Resident per Day"]
            elif category == "RN":
                reported_val = row["Reported RN Staffing Hours per Resident per Day"]
                case_mix_val = row["Case-Mix RN Staffing Hours per Resident per Day"]
            elif category == "Nurse Aide":
                reported_val = row["Reported Nurse Aide Staffing Hours per Resident per Day"]
                case_mix_val = row["Case-Mix Nurse Aide Staffing Hours per Resident per Day"]
            
            # Calculate values
            delta = reported_val - case_mix_val if pd.notna(reported_val) and pd.notna(case_mix_val) else 0
            delta_pct = ((reported_val / case_mix_val - 1) * 100) if pd.notna(reported_val) and pd.notna(case_mix_val) and case_mix_val > 0 else 0
            
            deltas.append(delta)
            delta_percentages.append(delta_pct)
            reported_values.append(reported_val if pd.notna(reported_val) else 0)
            case_mix_values.append(case_mix_val if pd.notna(case_mix_val) else 0)
        
        if not any(reported_values) and not any(case_mix_values):
            return None
            
        # Create the chart using Plotly
        fig = go.Figure()
        
        # Create the grouped bar chart (Reported vs Case-Mix)
        fig = go.Figure()
        
        # Add Reported bars (Blue)
        fig.add_trace(go.Bar(
            x=categories,
            y=reported_values,
            name='Reported',
            marker_color='blue',
            hovertemplate='<b>%{x}</b><br>Reported: %{y:.2f} HPRD<extra></extra>'
        ))
        
        # Add Case-Mix bars (Red)
        fig.add_trace(go.Bar(
            x=categories,
            y=case_mix_values,
            name='Case-Mix',
            marker_color='red',
            hovertemplate='<b>%{x}</b><br>Case-Mix: %{y:.2f} HPRD<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text=f"<span style='color: blue;'>Reported</span> vs. <span style='color: red;'>Case-Mix (Expected)</span><br><span style='font-weight: normal;'>{proper_title_case(facility_name)}, July 2025</span>",
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            xaxis_title="",
            yaxis_title="Hours per Resident Day",
            height=280,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50),
            barmode='group',
            bargap=0.1,
            bargroupgap=0.05
        )
        
        # Add text boxes above each category showing delta and percentage difference
        for i, category in enumerate(categories):
            if i < len(deltas) and i < len(delta_percentages):
                delta = deltas[i]
                pct_diff = delta_percentages[i]
                
                # Position the annotation above the bars
                gap_color = "rgba(33, 150, 243, 0.9)" if delta >= 0 else "rgba(244, 67, 54, 0.9)"  # Soft blue/red
                fig.add_annotation(
                    text=f"Δ: {delta:.2f} HPRD ({pct_diff:.1f}%)",
                    x=i,
                    y=max(reported_values[i], case_mix_values[i]) + 0.4,  # Position higher above the bars
                    showarrow=False,
                    font=dict(size=8, color="white"),
                    align="center",
                    bgcolor=gap_color,
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                )
        
        # Add 320 Consulting badge
        fig.add_annotation(
            text="<b>320 Consulting</b> | Source: CMS Provider Info (July 2025)",
            x=0.99,
            y=-0.6,
            xref="x domain",
            yref="y domain",
            showarrow=False,
            font=dict(size=10, color="#666666"),
            align="right"
        )
        
        return fig, None
        
    except Exception as e:
        st.error(f"Error creating case-mix charts: {str(e)}")
        return None, None


def plot_quarterly_trends(df: pd.DataFrame, state: str = None, facility: str = None):
    """Plot quarterly trends with optimized data processing."""
    try:
        data = df.sort_values('date')
        # Restore title_prefix logic
        if state:
            full_state_name = get_full_state_name(state)
            title_prefix = f"{full_state_name} Staffing Trends (2017-2025)"
        elif facility:
            # Get the most recent facility name from provider info data
            try:
                provider_df = pd.read_csv('NH_ProviderInfo_Jul2025.csv')
                facility_info = provider_df[provider_df['CMS Certification Number (CCN)'] == facility]
                if not facility_info.empty:
                    facility_name = proper_title_case(facility_info.iloc[0]['Provider Name'])
                    facility_state = facility_info.iloc[0]['State']
                    title_prefix = f"{facility_name}, {facility_state} (2017-2025)"
                else:
                    facility_name = get_provider_info(facility, 'name')
                    facility_state = get_provider_info(facility, 'state')
                    if facility_name and facility_state:
                        title_prefix = f"{facility_name}, {facility_state} (2017-2025)"
                    else:
                        title_prefix = f"Facility {facility} (2017-2025)"
            except:
                facility_name = get_provider_info(facility, 'name')
                facility_state = get_provider_info(facility, 'state')
                if facility_name and facility_state:
                    title_prefix = f"{facility_name}, {facility_state} (2017-2025)"
                else:
                    title_prefix = f"Facility {facility} (2017-2025)"
        else:
            title_prefix = "National Staffing Trends (2017-2025)"
        
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
                          vertical_spacing=0.18)
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

        # Use State_Census for state-level charts, Census for facility and national charts
        census_column = 'State_Census' if state else 'Census'
        fig.add_trace(go.Scatter(x=data['date'], y=data[census_column],
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
            width=1200,
            title_text=title_prefix,
            showlegend=False,
            margin=dict(l=50, r=50, t=100, b=100),
            hovermode='x unified'
        )
        
        # Add footer annotations for desktop view with improved styling
        for row in range(1, 4):
            fig.add_annotation(
                text="<b>320 Consulting</b> | Source: CMS PBJ Data (2017-2025)",
                x=0.99,
                y=-0.3,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                font=dict(size=10, color="#666666"),
                align="right",
                row=row,
                col=1
            )
        
        # Update desktop x-axes with improved tick handling for 33 quarters
        for row in range(1, 4):
            # For 33 quarters (2017-2025), show more years on x-axis
            # Show every year instead of every other year
            nticks_to_show = len(tick_values) if len(tick_values) <= 9 else 9
            
            fig.update_xaxes(
                tickvals=tick_values,
                tickangle=45,
                row=row,
                col=1,
                showline=True,
                linewidth=1,
                linecolor="rgba(200, 200, 200, 0.1)",
                range=date_range,
                nticks=nticks_to_show,
                tickmode='auto'
            )
        
        # Add y-axis labels for each subplot using yaxis titles
        fig.update_yaxes(title_text="Hours Per Resident Day", row=1, col=1, title_font=dict(size=10, color="#999999"), title_standoff=10)
        fig.update_yaxes(title_text="Residents Per Day", row=2, col=1, title_font=dict(size=10, color="#999999"), title_standoff=10)
        fig.update_yaxes(title_text="% Contract Staff", row=3, col=1, title_font=dict(size=10, color="#999999"), title_standoff=10)
        
        # Create mobile chart if on mobile device
        if st.session_state.get('is_mobile', False):
            # Mobile chart (single chart with tabs)
            mobile_fig = go.Figure()
            
            # Add traces for mobile
            mobile_fig.add_trace(go.Scatter(
                x=data['date'], 
                y=data['Total_Nurse_HPRD'],
                mode='lines+markers', 
                name='Total HPRD',
                customdata=data['CY_QTR'].apply(lambda x: f"Q{x[-1]} {x[:4]}"), 
                hovertemplate=hover_hprd
            ))
            
            # Update mobile layout
            mobile_fig.update_layout(
                height=400,
                width=800,
                title_text=title_prefix,
                showlegend=True,
                margin=dict(l=50, r=50, t=80, b=80),
                hovermode='x unified',
                xaxis=dict(
                    tickvals=tick_values,
                    tickangle=45,
                    showline=True,
                    linewidth=1,
                    linecolor="rgba(200, 200, 200, 0.1)",
                    range=date_range,
                    nticks=nticks_to_show,
                    tickmode='auto'
                )
            )
            
            # Add footer annotation for mobile
            mobile_fig.add_annotation(
                text="<b>320 Consulting</b> | Source: CMS PBJ Data (2017-2025)",
                x=0.99,
                y=-0.6,
                xref="x domain",
                yref="y domain",
                showarrow=False,
                font=dict(size=10, color="#666666"),
                align="right"
            )
            

            
            return mobile_fig
        
        return fig
        
    except Exception as e:
        st.error(f"Error plotting trends: {str(e)}")
        return None

def display_footer():
    """Display a consistent footer across all pages."""
    st.markdown("""
        <div style="text-align: center; margin-top: 10px; color: #666; font-size: 0.9em;">
            <p>Source: <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" style="color: #1E88E5; text-decoration: none;">CMS Payroll-Based Journal Data, 2017-2025</a></p>
            <p>By <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">320 Consulting LLC</a></p>
        </div>
    """, unsafe_allow_html=True)

def _apply_navigation_and_stop():
    """If a pending navigation exists, apply it immediately and stop rendering."""
    nav = st.session_state.get("pending_navigation")
    if not nav:
        return

    # Preserve mobile flag from current query OR from nav payload
    mobile_flag = nav.get("preserve_mobile", st.query_params.get("mobile"))

    # Build target params
    new_params = {}
    t = nav.get("type")
    if t == "entity":
        new_params["entity"] = nav["id"]
    elif t == "state":
        new_params["state"] = nav["state"]
    # home => no params

    # Apply
    st.query_params.clear()
    if mobile_flag:
        st.query_params["mobile"] = mobile_flag
    for k, v in new_params.items():
        st.query_params[k] = v

    st.session_state.pop("pending_navigation", None)
    st.rerun()


def _go_home():
    """Navigate to home page while preserving mobile flag."""
    st.session_state.pending_navigation = {
        "type": "home",
        "preserve_mobile": st.query_params.get("mobile")
    }
    st.rerun()


def main() -> None:
    """Main app layout and data flow."""
    try:
        # Initialize session state variables at the very start
        if 'view_mode' not in st.session_state:
            st.session_state.view_mode = "Desktop"
        
        # CRITICAL: apply navigation before any rendering
        _apply_navigation_and_stop()
        
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

        # Load metrics data early for sidebar logic
        national_metrics, state_metrics, facility_metrics = load_metrics_data()

        # Stabilize search section selection across reruns
        if not (st.query_params.get('state') or st.query_params.get('entity') or st.query_params.get('facility')):
            st.session_state.pop("pending_navigation", None)  # nuke stale nav on home

        # Get current page from URL
        current_page = st.query_params.get('page', 'dashboard')

        # Handle different pages - removed problematic navigation

        # Get URL parameters using the new API
        initial_level = st.query_params.get('level', None)
        initial_facility = st.query_params.get('facility', None)
        initial_state = st.query_params.get('state', None)
        initial_entity = st.query_params.get('entity', None)
        initial_state_filter = st.query_params.get('state_filter', None)
        
        # Determine the actual level based on URL parameters
        if initial_state:
            actual_level = "State"
        elif initial_entity:
            actual_level = "Entity"
        elif initial_facility:
            actual_level = "Facility"
        elif initial_level:
            actual_level = initial_level
        else:
            actual_level = "National"
        
        # Handle direct URL navigation for state, entity, and facility levels
        if actual_level in ["State", "Entity", "Facility"]:
            level = actual_level
            if actual_level == "State":
                selected_value = initial_state
            elif actual_level == "Entity":
                selected_value = initial_entity
            elif actual_level == "Facility":
                selected_value = initial_facility
            # Hide the radio button since we're pre-setting the level
            st.session_state.level_pre_set = True
        else:
            level = None  # Will be set by radio button
            st.session_state.level_pre_set = False

        # Title with custom styling - matching About page dark blue color and mobile responsive
        if not initial_state_filter:
            # Add state page mobile padding class if we're on a state page
            state_page_class = "state-page-mobile-padding" if actual_level == "State" else ""
            st.markdown(f"""
                <div class="{state_page_class}" style='text-align: center; margin-top: -20px; margin-bottom: 1.2em;'>
                    <span class="dashboard-title" style='font-size:2.8em; font-weight:800; color:#1769aa; letter-spacing:0.01em; line-height:1.1;'>PBJ Nursing Home Staffing Dashboard</span>
                </div>
                <style>
                @media (max-width: 768px) {{
                    .dashboard-title {{
                        font-size: 2.1em !important;
                        line-height: 1.2 !important;
                    }}
                    div[data-testid="stMarkdown"] > div:has(> span.dashboard-title) {{
                        margin-top: -120px !important;
                        margin-bottom: 0.1em !important;
                    }}
                    /* Additional padding for state page on mobile */
                    .state-page-mobile-padding {{
                        margin-top: 15px !important;
                    }}
                    /* Additional padding for state page on desktop */
                    @media (min-width: 768px) {{
                        .state-page-mobile-padding {{
                            margin-top: 25px !important;
                        }}
                    }}
                }}
                </style>
            """, unsafe_allow_html=True)

        # Check if we should hide search based on URL parameters
        hide_search = False
        if initial_facility:
            hide_search = True

        # Refined subhead: centered text and box with search link
        if not initial_state_filter:
                        # Full text for all pages (mobile will be handled by CSS)
            st.markdown(f'''
                <div style="background: #f7fafd; border-radius: 6px; padding: 6px 14px 2px 14px; margin-bottom: 1px; border: 1px solid #e3eaf3; max-width: 950px; margin-left: auto; margin-right: auto; text-align: center;">
                    <div style="font-size: 1.08em; color: #234; font-weight: 600; margin-bottom: 1px;">
                        <span class="desktop-text">A free public resource from <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 700;"><b>320 Consulting</b></a>, featuring quarterly staffing data (2017–2025) across every U.S. nursing home.</span>
                        <span class="mobile-text">A free public resource from <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 700;"><b>320 Consulting</b></a>.</span>
                    </div>
                    {"<div style='margin-top: 0px;' class='mobile-about-link'><a href='/About' target='_self' style='color: #1769aa; text-decoration: none; font-size: 0.9em; font-weight: 400;'>About the PBJ Dashboard</a></div>" if not hide_search and level not in ["Entity", "State"] else ""}
                </div>
            ''', unsafe_allow_html=True)
        
        # Add mobile-specific CSS for reduced spacing
        st.markdown("""
            <style>
            @media (max-width: 768px) {
                div[data-testid="stMarkdown"] > div:has(> div[style*="background: #f7fafd"]) {
                    margin-bottom: 15px !important;
                }
                div[style*="background: #f7fafd"] {
                    margin-bottom: 15px !important;
                }
                /* Show mobile text, hide desktop text on mobile */
                .desktop-text {
                    display: none !important;
                }
                .mobile-text {
                    display: inline !important;
                }
                /* Move down Back to Search button on mobile state pages */
                a[href*="mobile=true"], a[href="/"] {
                    margin-top: 8px !important;
                }
            }
            @media (min-width: 769px) {
                div[data-testid="stMarkdown"] > div:has(> div[style*="background: #f7fafd"]) {
                    margin-bottom: 35px !important;
                }
                div[style*="background: #f7fafd"] {
                    margin-bottom: 35px !important;
                }
                /* Show desktop text, hide mobile text on desktop */
                .desktop-text {
                    display: inline !important;
                }
                .mobile-text {
                    display: none !important;
                }
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Add About link only on homepage and only on desktop (exclude state filter pages)
        if not hide_search and level not in ["Entity", "State"] and not initial_state_filter:
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
                <div class="desktop-about-link" style="text-align: center; margin-bottom: 45px; margin-top: 15px;">
                    <a href="/About" target="_self" style="color: #1769aa; text-decoration: none; font-size: 0.9em; font-weight: 600; background: #e8f4fd; padding: 3px 12px; border-radius: 6px; border: 1px solid #1976d2; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; transition: all 0.2s ease;">
                        About the PBJ Dashboard
                    </a>
                </div>
            ''', unsafe_allow_html=True)
        
        # Load search data functions (always available)
        
        @st.cache_data
        def load_provider_info_data():
            """Load provider info data."""
            try:
                return pd.read_csv('NH_ProviderInfo_Jul2025.csv', dtype={'PROVNUM': str})
            except Exception as e:
                st.error(f"Error loading provider info: {str(e)}")
                return pd.DataFrame()
        
        @st.cache_data
        def load_ownership_data():
            """Load ownership data."""
            try:
                return pd.read_csv('Nursing_Home_Chain_Performance_Measures_Jul_2025.csv')
            except Exception as e:
                st.error(f"Error loading ownership data: {str(e)}")
                return pd.DataFrame()
        
        @st.cache_data
        def load_previous_ownership_data():
            """Load March ownership data for comparison."""
            try:
                df = pd.read_csv('Nursing_Home_Affiliated_Entity_Performance_Measures_Mar_2025.csv')
                # Map March column names to July column names for comparison
                column_mapping = {
                    'Affiliated entity': 'Chain',
                    'Affiliated entity ID': 'Chain ID'
                }
                # Only rename columns that exist
                existing_columns = [col for col in column_mapping.keys() if col in df.columns]
                if existing_columns:
                    rename_dict = {col: column_mapping[col] for col in existing_columns}
                    df = df.rename(columns=rename_dict)
                return df
            except Exception as e:
                st.error(f"Error loading March ownership data: {str(e)}")
                return pd.DataFrame()
        
        @st.cache_data
        def load_march_provider_info_data():
            """Load March provider info data for comparison."""
            try:
                return pd.read_csv('NH_ProviderInfo_Mar2025.csv', dtype={'CMS Certification Number (CCN)': str})
            except Exception as e:
                st.error(f"Error loading March provider info data: {str(e)}")
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

        # Show "Back to Search" button for facility, entity, and state pages
        if level in ["Facility", "Entity", "State"]:
            # Preserve mobile parameter in home link
            home_href = "/?mobile=true" if st.query_params.get("mobile") else "/"
            
            if level == "Facility":
                # Get facility state for the "View other [State] nursing homes" button
                facility_state = None
                if initial_facility:
                    facilities_df = load_facility_data()
                    facility_data = facilities_df[facilities_df['PROVNUM'] == initial_facility]
                    if not facility_data.empty:
                        facility_state = facility_data['STATE'].iloc[0]
                
                if facility_state:
                    state_name = get_full_state_name(facility_state)
                    # Link to facility search with state filter pre-selected
                    state_href = f"/?mobile=true&state_filter={facility_state}" if st.query_params.get("mobile") else f"/?state_filter={facility_state}"
                    
                    st.markdown(f"""
                        <div style="margin-bottom: 0px; padding: 0px; display: flex; gap: 8px; align-items: center;">
                            <a href="{home_href}" target="_self" style="color: #1976d2; text-decoration: none; font-weight: 500; font-size: 0.8em; padding: 3px 8px; border-radius: 3px; background: #f8f9fa; border: 1px solid #e3eaf3; transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 3px;">
                                <span style="font-size: 0.9em;">←</span> Back to Search
                            </a>
                            <a href="{state_href}" target="_self" style="color: #1976d2; text-decoration: none; font-weight: 500; font-size: 0.8em; padding: 3px 8px; border-radius: 3px; background: #f8f9fa; border: 1px solid #e3eaf3; transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 3px;">
                                <span class="desktop-text">View {state_name} nursing homes</span>
                                <span class="mobile-text">View {facility_state} nursing homes</span>
                            </a>
                        </div>
                        <style>
                        /* Mobile/Desktop text switching for facility pages */
                        .mobile-text {{
                            display: none;
                        }}
                        @media (max-width: 768px) {{
                            .desktop-text {{
                                display: none;
                            }}
                            .mobile-text {{
                                display: inline;
                            }}
                            /* Lower the second button on mobile to align with Back to Search */
                            a[href*="state_filter"] {{
                                margin-top: 8px !important;
                            }}
                        }}
                        </style>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style="margin-bottom: 0px; padding: 0px;">
                            <a href="{home_href}" target="_self" style="color: #1976d2; text-decoration: none; font-weight: 500; font-size: 0.8em; padding: 3px 8px; border-radius: 3px; background: #f8f9fa; border: 1px solid #e3eaf3; transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 3px;">
                                <span style="font-size: 0.9em;">←</span> Back to Search
                            </a>
                        </div>
                    """, unsafe_allow_html=True)
            elif level == "State":
                # Get state name for the "View [State] facilities" button
                state_name = get_full_state_name(initial_state)
                # Link to facility search with state filter pre-selected
                state_href = f"/?mobile=true&state_filter={initial_state}" if st.query_params.get("mobile") else f"/?state_filter={initial_state}"
                
                if initial_state:
                    st.markdown(f"""
                        <div style="margin-bottom: 0px; padding: 0px; display: flex; gap: 8px; align-items: center;">
                            <a href="{home_href}" target="_self" style="color: #1976d2; text-decoration: none; font-weight: 500; font-size: 0.8em; padding: 3px 8px; border-radius: 3px; background: #f8f9fa; border: 1px solid #e3eaf3; transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 3px;">
                                <span style="font-size: 0.9em;">←</span> Back to Search
                            </a>
                            <a href="{state_href}" target="_self" style="color: #1976d2; text-decoration: none; font-weight: 500; font-size: 0.8em; padding: 3px 8px; border-radius: 3px; background: #f8f9fa; border: 1px solid #e3eaf3; transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 3px;">
                                <span class="desktop-text">View {state_name} facilities</span>
                                <span class="mobile-text">View {initial_state} facilities</span>
                            </a>


                        </div>
                        <style>
                        /* Lower both buttons on state pages to prevent bleeding into content above */
                        a[href*="state_filter"] {{
                            margin-top: 15px !important;
                        }}
                        a[href="/"] {{
                            margin-top: 15px !important;
                        }}
                        /* Move Select State dropdown down a bit */
                        div[data-testid="stSelectbox"] {{
                            margin-top: 10px !important;
                        }}
                        /* Mobile/Desktop text switching */
                        .mobile-text {{
                            display: none;
                        }}
                        @media (max-width: 768px) {{
                            .desktop-text {{
                                display: none;
                            }}
                            .mobile-text {{
                                display: inline;
                            }}
                        }}
                        </style>
                    """, unsafe_allow_html=True)
                    

                else:
                    st.markdown(f"""
                        <div style="margin-bottom: 0px; padding: 0px;">
                            <a href="{home_href}" target="_self" style="color: #1976d2; text-decoration: none; font-weight: 500; font-size: 0.8em; padding: 3px 8px; border-radius: 3px; background: #f8f9fa; border: 1px solid #e3eaf3; transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 3px;">
                                <span style="font-size: 0.9em;">←</span> Back to Search
                            </a>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style="margin-bottom: 0px; padding: 0px;">
                        <a href="{home_href}" target="_self" style="color: #1976d2; text-decoration: none; font-weight: 500; font-size: 0.8em; padding: 3px 8px; border-radius: 3px; background: #f8f9fa; border: 1px solid #e3eaf3; transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 3px;">
                            <span style="font-size: 0.9em;">←</span> Back to Search
                        </a>
                    </div>
                """, unsafe_allow_html=True)
            

            

            
            # Add specific CSS for entity pages to ensure proper spacing
            if level == "Entity":
                st.markdown("""
                    <style>
                    /* Specific styling for entity pages */
                    div[data-testid="stSelectbox"] {
                        margin-top: -20px !important;
                    }
                    /* Reduce spacing around the selectbox container */
                    div[data-testid="stElementContainer"] {
                        margin-top: -6px !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
        
        # Handle state filter parameter - show filtered facility list
        if initial_state_filter:
            hide_search = False
            level = "National"  # Set to national to show search interface
            st.session_state.level_pre_set = True
            
            # Load facility data and filter by state
            facilities_df = load_facility_data()
            state_facilities = facilities_df[facilities_df['STATE'] == initial_state_filter]
            
            if not state_facilities.empty:
                # Get unique facilities for the state
                unique_facilities = state_facilities[['PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME']].drop_duplicates()
                
                # Create display DataFrame
                display_df = pd.DataFrame()
                display_df['State'] = unique_facilities['STATE']
                display_df['Nursing Home (CCN)'] = unique_facilities['PROVNAME'].apply(smart_title) + ' (' + unique_facilities['PROVNUM'] + ')'
                display_df['County'] = unique_facilities['COUNTY_NAME']
                display_df['Dashboard'] = unique_facilities['PROVNUM'].apply(
                    lambda x: f'<a href="/?facility={x}" style="color: #1976d2; text-decoration: none; font-weight: bold;" target="_self">View</a>'
                )
                
                # Sort alphabetically by nursing home name
                display_df = display_df.sort_values('Nursing Home (CCN)')
                
                # Get state name for display
                state_name = get_full_state_name(initial_state_filter)
                
                # Add "Back to Search" button
                home_href = "/?mobile=true" if st.query_params.get("mobile") else "/"
                st.markdown(f"""
                    <div style="margin-bottom: 5px; padding: 0px;">
                        <a href="{home_href}" target="_self" style="color: #1976d2; text-decoration: none; font-weight: 500; font-size: 0.8em; padding: 3px 8px; border-radius: 3px; background: #f8f9fa; border: 1px solid #e3eaf3; transition: all 0.2s ease; display: inline-flex; align-items: center; gap: 3px;">
                            <span style="font-size: 0.9em;">←</span> Back to Search
                        </a>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"#### All Nursing Homes in {state_name}")
                st.markdown(f"*{len(display_df)} facilities found*")
                st.markdown("""
                    <style>
                    /* Target the specific table structure */
                    div[data-testid="stMarkdown"] table th:nth-child(2),
                    div[data-testid="stMarkdown"] table td:nth-child(2) {
                        text-align: left !important;
                    }
                    div[data-testid="stMarkdown"] table th:nth-child(3),
                    div[data-testid="stMarkdown"] table td:nth-child(3) {
                        text-align: left !important;
                    }
                    </style>
                """, unsafe_allow_html=True)
                st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.info(f"No facilities found for state {initial_state_filter}.")
        
        # Add search functionality (only when not hiding search)
        if not hide_search:
            st.markdown("""
                <style>
                /* Only apply search interface styling when search is not hidden */
                .search-header {
                    margin-bottom: 5px;
                    margin-top: -10px;
                }
                .stTabs [data-baseweb="tab-list"] {
                    margin-top: -5px;
                }
                @media (min-width: 769px) {
                    .search-header {
                        margin-top: -20px !important;
                        margin-bottom: 0px !important;
                    }
                }
                @media (max-width: 768px) {
                    .search-header {
                        margin-top: -150px;
                        margin-bottom: 0px;
                        transform: translateY(-30px);
                    }
                    h3.search-header {
                        margin-top: 8px !important;
                    }
                    div[data-testid="stMarkdown"] > div:has(> div[style*="background: #f7fafd"]) {
                        margin-bottom: 0px !important;
                    }
                    div[data-testid="stMarkdown"] > div:has(> h3.search-header) {
                        margin-top: -40px !important;
                        margin-bottom: 0px !important;
                    }
                    .stTabs [data-baseweb="tab-list"] {
                        margin-top: -25px !important;
                    }
                    div[data-testid="stTabs"] {
                        margin-top: -25px !important;
                    }
                }
                </style>
            """, unsafe_allow_html=True)
            
            # Show "Search PBJ Data" header for non-facility/entity/state pages
            if level not in ["Facility", "Entity", "State"]:
                st.markdown('''
                    <h3 class="search-header">
                        <span class="desktop-text">Search PBJ Data by Facility, State, or Ownership Group</span>
                        <span class="mobile-text">Search PBJ Data</span>
                    </h3>
                    <style>
                        @media (max-width: 768px) {
                            .desktop-text { display: none !important; }
                        }
                        @media (min-width: 769px) {
                            .mobile-text { display: none !important; }
                        }
                    </style>
                ''', unsafe_allow_html=True)
            
            # Load data for search
            facilities_df = load_facility_data()
            provider_info_df = load_provider_info_data()
            ownership_df = load_ownership_data()
            
            # Show different interface based on level
            if level == "Entity":
                # Show ownership dropdown for entity pages
                if not ownership_df.empty:
                    # Filter to show only major ownership groups
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
                    ownership_entities = ownership_df[ownership_df['Chain'].notna()].copy()
                    
                    # Filter to major groups or those with 10+ facilities
                    major_entities = ownership_entities[
                        (ownership_entities['Chain'].isin(major_ownership_groups)) |
                        (ownership_entities['Number of facilities'] >= 10)
                    ].copy()
                    
                    # Remove National from the results
                    major_entities = major_entities[major_entities['Chain'] != 'National'].copy()
                    
                    # Sort by number of facilities (descending)
                    major_entities = major_entities.sort_values('Number of facilities', ascending=False)
                    
                    # Create display names with entity ID stored in session state
                    ownership_options = [""]
                    for _, row in major_entities.iterrows():
                        display_name = smart_title(row['Chain']) + f" ({int(row['Number of facilities'])} NHs)"
                        ownership_options.append(display_name)
                        # Store entity ID in session state for later use, only if it's not NaN
                        if pd.notna(row['Chain ID']):
                            st.session_state[f"entity_{display_name}"] = int(row['Chain ID'])
                            st.session_state[f"name_{display_name}"] = row['Chain']
                    
                    # Add CSS to style the placeholder option
                    st.markdown("""
                        <style>
                        /* Style the placeholder option in the selectbox */
                        .stSelectbox option:first-child {
                            color: #666 !important;
                            font-style: italic !important;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    ownership_search_display = st.selectbox(
                        "Select Ownership Group",
                        options=ownership_options,
                        key="entity_ownership_dropdown",
                        help="Choose ownership group to view their dashboard"
                    )
                    
                    if ownership_search_display:
                        # Get the stored entity ID and name
                        entity_id = st.session_state.get(f"entity_{ownership_search_display}")
                        ownership_name = st.session_state.get(f"name_{ownership_search_display}")
                        
                        if entity_id and ownership_name:
                            # Store navigation info in session state and trigger rerun
                            if "pending_navigation" not in st.session_state:
                                st.session_state.pending_navigation = {
                                    "type": "entity", 
                                    "id": entity_id,
                                    "preserve_mobile": st.query_params.get("mobile")
                                }
                                st.rerun()
                        else:
                            st.info("Ownership group not found.")
                else:
                    st.info("Ownership data not available.")
                    
            elif level == "State":
                # Show state dropdown for state pages
                # Build options and map display -> code
                state_options = [""]
                for state_code in sorted(facilities_df['STATE'].unique().tolist()):
                    state_name = get_full_state_name(state_code)
                    display_name = f"{state_name} ({state_code})"
                    state_options.append(display_name)
                    st.session_state[f"state_{display_name}"] = state_code

                state_options.append("USA")
                st.session_state["state_USA"] = "USA"

                state_key = f"state_dropdown_{st.query_params.get('entity', 'main')}"

                def _go_state_from_dropdown():
                    display = st.session_state.get(state_key, "")
                    if not display:
                        return
                    code = st.session_state.get(f"state_{display}")
                    if not code:
                        return
                    if code == "USA":
                        _go_home()
                    else:
                        st.session_state.pending_navigation = {
                            "type": "state", 
                            "state": code,
                            "preserve_mobile": st.query_params.get("mobile")
                        }
                        st.rerun()

                st.selectbox(
                    "Select State",
                    options=state_options,
                    key=state_key,
                    help="Choose state to view their dashboard",
                    on_change=_go_state_from_dropdown
                )
                
            else:
                # Show full search interface for other pages (National, etc.)
                # Create search tabs
                tab1, tab2, tab3 = st.tabs(["🔍 Facility", "🏢 Ownership", "🗺️ State"])
                
                with tab1:
                    # Use responsive columns for mobile-friendly layout
                    col1, col2 = st.columns(2)
                
                    with col1:
                        state_filter = st.selectbox(
                            "Filter by State (Optional)",
                            [""] + sorted(facilities_df['STATE'].unique().tolist()),
                            key="facility_state_filter",
                            help="Enter two letter state abbreviation"
                        )
                    
                    with col2:
                        # Create filtered search options based on selected state
                        if state_filter:
                            state_facilities = facilities_df[facilities_df['STATE'] == state_filter]
                            search_options = [f"{smart_title(row['PROVNAME'])} ({row['PROVNUM']})" for _, row in state_facilities[['PROVNAME', 'PROVNUM']].drop_duplicates().iterrows()]
                        else:
                            state_facilities = facilities_df  # Use full dataset when no state filter
                            search_options = [f"{smart_title(row['PROVNAME'])} ({row['PROVNUM']})" for _, row in facilities_df[['PROVNAME', 'PROVNUM']].drop_duplicates().iterrows()]

                        # Sort options alphabetically by facility name
                        search_options.sort()
                        
                        facility_search = st.selectbox(
                            "Enter Provider Name or CCN (6-digit ID)",
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
                            # Get unique facilities
                            unique_results = results[['PROVNUM', 'PROVNAME', 'STATE']].drop_duplicates()
                            
                            # Create display DataFrame
                            display_df = pd.DataFrame()
                            display_df['State'] = unique_results['STATE']
                            display_df['Nursing Home (CCN)'] = unique_results['PROVNAME'].apply(smart_title) + ' (' + unique_results['PROVNUM'] + ')'
                            display_df['Dashboard'] = unique_results['PROVNUM'].apply(
                                lambda x: f'<a href="/?facility={x}" style="color: #1976d2; text-decoration: none; font-weight: bold;" target="_self">View</a>'
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
                        # Get all facilities for the selected state
                        state_facilities_all = facilities_df[facilities_df['STATE'] == state_filter]
                    
                        if not state_facilities_all.empty:
                            # Get unique facilities
                            unique_facilities = state_facilities_all[['PROVNUM', 'PROVNAME', 'STATE']].drop_duplicates()
                            
                            # Create display DataFrame
                            display_df = pd.DataFrame()
                            display_df['State'] = unique_facilities['STATE']
                            display_df['Nursing Home (CCN)'] = unique_facilities['PROVNAME'].apply(smart_title) + ' (' + unique_facilities['PROVNUM'] + ')'
                            display_df['Dashboard'] = unique_facilities['PROVNUM'].apply(
                                lambda x: f'<a href="/?facility={x}" style="color: #1976d2; text-decoration: none; font-weight: bold;" target="_self">View</a>'
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
                        ownership_entities = ownership_df[ownership_df['Chain'].notna()].copy()
                        
                        # Filter to major groups or those with 10+ facilities
                        major_entities = ownership_entities[
                            (ownership_entities['Chain'].isin(major_ownership_groups)) |
                            (ownership_entities['Number of facilities'] >= 10)
                        ].copy()
                        
                        # Remove National from the results
                        major_entities = major_entities[major_entities['Chain'] != 'National'].copy()
                        
                        # Sort by number of facilities (descending)
                        major_entities = major_entities.sort_values('Number of facilities', ascending=False)
                        
                        # Create display names with entity ID stored in session state
                        ownership_options = [""]
                        for _, row in major_entities.iterrows():
                            display_name = smart_title(row['Chain']) + f" ({int(row['Number of facilities'])} NHs)"
                            ownership_options.append(display_name)
                            # Store entity ID in session state for later use, only if it's not NaN
                            if pd.notna(row['Chain ID']):
                                st.session_state[f"entity_{display_name}"] = int(row['Chain ID'])
                                st.session_state[f"name_{display_name}"] = row['Chain']
                        
                        ownership_search_display = st.selectbox(
                            "Select Ownership Group",
                            options=ownership_options,
                            key="ownership_search_input",
                            help="Choose ownership group to view their dashboard"
                        )
                        
                        if ownership_search_display:
                            # Get the stored entity ID and name
                            entity_id = st.session_state.get(f"entity_{ownership_search_display}")
                            ownership_name = st.session_state.get(f"name_{ownership_search_display}")
                            
                            if entity_id and ownership_name:
                                # Store navigation info in session state and trigger rerun
                                if "pending_navigation" not in st.session_state:
                                    st.session_state.pending_navigation = {
                                        "type": "entity", 
                                        "id": entity_id,
                                        "preserve_mobile": st.query_params.get("mobile")
                                    }
                                    st.rerun()
                            else:
                                st.info("Ownership group not found.")
                    else:
                        st.info("Ownership data not available.")
                
                with tab3:
                    # --- inside the "State" tab (tab3) ---

                    # Build options and map display -> code
                    state_options = [""]
                    for state_code in sorted(facilities_df['STATE'].unique().tolist()):
                        state_name = get_full_state_name(state_code)
                        display_name = f"{state_name} ({state_code})"
                        state_options.append(display_name)
                        st.session_state[f"state_{display_name}"] = state_code

                    state_options.append("USA")
                    st.session_state["state_USA"] = "USA"

                    state_key = f"state_search_input_{st.query_params.get('entity', 'main')}"

                    def _go_state_from_tab():
                        display = st.session_state.get(state_key, "")
                        if not display:
                            return
                        code = st.session_state.get(f"state_{display}")
                        if not code:
                            return
                        if code == "USA":
                            _go_home()
                        else:
                            st.session_state.pending_navigation = {
                                "type": "state", 
                                "state": code,
                                "preserve_mobile": st.query_params.get("mobile")
                            }
                            st.rerun()

                    st.selectbox(
                        "Select State",
                        options=state_options,
                        key=state_key,
                        help="Choose state to view their dashboard",
                        on_change=_go_state_from_tab
                    )
        
        # st.markdown("""
        #     <hr style="margin: 8px 0; border: none; border-top: 1px solid #e0e0e0; height: 1px;">
        # """, unsafe_allow_html=True)
        
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

        # Sidebar: Only show radio button if level not pre-set by URL
        if not st.session_state.get('level_pre_set', False):  # Only show radio button if level not pre-set by URL
            # Initialize level in session state if not present
            if 'current_level' not in st.session_state:
                st.session_state.current_level = "National"
            
            # Use selectbox instead of radio to reduce reruns
            level = st.sidebar.selectbox(
                "Select Level",
                ["National", "State", "Facility"],
                index=["National", "State", "Facility", "Ownership"].index(initial_level) if initial_level in ["National", "State", "Facility", "Ownership"] else 0,
                key="level_selector"
            )
            
            # Update session state
            st.session_state.current_level = level
            
            # Track level changes for facility search persistence
            st.session_state.last_level = level
        # If level is pre-set by URL, don't show radio button
        if st.session_state.get('level_pre_set', False):
            # Ensure level is properly set when pre-set by URL
            if initial_state and not initial_level:
                level = "State"
            elif initial_entity:
                level = "Entity"
            elif initial_facility:
                level = "Facility"
            
            # Add a small button to allow switching levels
            # if st.sidebar.button("Switch Level", key="switch_level", help="Switch to different level"):
            #     st.session_state.level_pre_set = False
            #     st.query_params.clear()
            #     st.rerun()

        # Get selected value based on level
        selected_value = None
        try:
            if level == "State":
                states = ["---"] + sorted(state_metrics['STATE'].unique().tolist())
                
                # If we have a state from URL parameter, use it
                if initial_state and initial_state in state_metrics['STATE'].unique():
                    selected_state = initial_state
                    selected_value = initial_state
                    # Don't show selectbox if value is pre-set from URL
                else:
                    selected_state = st.sidebar.selectbox(
                        "Select State",
                        states,
                        index=0
                    )
                    selected_value = selected_state if selected_state != "---" else None
            elif level == "Entity":
                # Load chain data
                entity_data = load_affiliated_entity_data()
                if not entity_data.empty:
                    # Filter out the "National" row (aggregate data)
                    entity_data = entity_data[entity_data['Chain'] != 'National'].copy()
                    
                    # Sort by number of facilities (descending)
                    entity_data = entity_data.sort_values('Number of facilities', ascending=False)
                    
                    # Create entity options
                    entity_options = []
                    for idx, entity in entity_data.iterrows():
                        entity_name = entity['Chain']
                        facility_count = entity['Number of facilities']
                        entity_options.append(f"{entity_name} ({facility_count} facilities)")
                    
                    # If we have an entity from URL parameter, use it
                    if initial_entity:
                        # First try to find the entity by ID (convert to int for comparison)
                        try:
                            entity_id_int = int(initial_entity)
                            matching_entity = entity_data[entity_data['Chain ID'] == entity_id_int]
                            if not matching_entity.empty:
                                selected_entity = matching_entity.iloc[0]['Chain']
                                selected_value = selected_entity
                                # Don't show selectbox if value is pre-set from URL
                            else:
                                # If not found by ID, try by name
                                matching_entity = entity_data[entity_data['Chain'] == initial_entity]
                                if not matching_entity.empty:
                                    selected_entity = initial_entity
                                    selected_value = initial_entity
                                    # Don't show selectbox if value is pre-set from URL
                                else:
                                    selected_entity = st.sidebar.selectbox(
                                        "Select Chain",
                                        ["---"] + entity_options,
                                        index=0
                                    )
                                    selected_value = selected_entity.split(" (")[0] if selected_entity != "---" else None
                        except ValueError:
                            # If initial_entity is not a number, try by name
                            matching_entity = entity_data[entity_data['Chain'] == initial_entity]
                            if not matching_entity.empty:
                                selected_entity = initial_entity
                                selected_value = initial_entity
                                # Don't show selectbox if value is pre-set from URL
                            else:
                                selected_entity = st.sidebar.selectbox(
                                    "Select Chain",
                                    ["---"] + entity_options,
                                    index=0
                                )
                                selected_value = selected_entity.split(" (")[0] if selected_entity != "---" else None
                    else:
                        selected_entity = st.sidebar.selectbox(
                            "Select Chain",
                            ["---"] + entity_options,
                            index=0
                        )
                        selected_value = selected_entity.split(" (")[0] if selected_entity != "---" else None
                else:
                    st.error("Unable to load entity data.")
                    selected_value = None
            elif level == "Facility":
                # If we have a facility from URL, use it directly without showing search interface
                if initial_facility:
                    # Set the selected value directly from the CCN
                    selected_value = initial_facility
                    # Load facility data for display
                    facilities_df = load_facility_data()
                    matching_facilities = facilities_df[facilities_df['PROVNUM'] == initial_facility].to_dict('records')
                else:
                    # Show search interface only when not coming from URL
                    search_container = st.sidebar.container()
                    
                    # Initialize persistent session state variables that don't get cleared
                    if 'persistent_facility_search' not in st.session_state:
                        st.session_state.persistent_facility_search = ""
                    if 'persistent_facility_results' not in st.session_state:
                        st.session_state.persistent_facility_results = []
                    if 'facility_search_initialized' not in st.session_state:
                        st.session_state.facility_search_initialized = False
                    
                    # Use persistent search term
                    search_term = st.session_state.persistent_facility_search

                    # Create the search input
                    new_search_term = search_container.text_input(
                        "Enter Provider CCN or Name",
                        value=search_term,
                        key="facility_search_input_sidebar"
                    )
                    
                    # Add help text with hyperlink
                    st.sidebar.markdown(
                        '<div style="margin-top: -15px; margin-bottom: 15px;">'
                        '<a href="/Facility_Search" target="_self" style="color: #1E88E5; text-decoration: none; font-size: 0.9em;">'
                        'Help finding facility data</a></div>',
                        unsafe_allow_html=True
                    )
                    
                    # Handle search logic
                    search_triggered = False

                    # Check if search term changed
                    if new_search_term != st.session_state.persistent_facility_search:
                        st.session_state.persistent_facility_search = new_search_term
                        search_term = new_search_term
                        search_triggered = True
                    
                    # Handle mobile search button
                    if st.session_state.view_mode == "Mobile":
                        if search_container.button("Search", key="facility_search_button"):
                            search_triggered = True
                    
                    # Perform search if triggered
                    if search_triggered and search_term:
                        st.session_state.persistent_facility_results = search_facilities(search_term)
                        st.session_state.facility_search_initialized = True
                    
                    # Use persistent results - preserve results across reruns
                    matching_facilities = st.session_state.persistent_facility_results

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
        except Exception as e:
            st.error(f"Error processing selection: {str(e)}")
            return

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
                # Get selected facility details
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
                    
                    # Add flashy PBJ Takeaway button with responsive positioning
                    try:
                        with open('pbj_favicon.png', 'rb') as f:
                            favicon_data = base64.b64encode(f.read()).decode()
                        
                                                    # Create the button with favicon
                            state_button_class = "state-page-mobile-button" if level == "State" else ""
                            button_html = f"""
                            <div class="pbj-button-container {state_button_class}" id="pbj-takeaway-button" style="position: fixed; top: 80px; z-index: 1000;">
                                <a href="#pbj-takeaway" onclick="fadePBJButton()" style="background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%); color: white; padding: 10px 18px; border-radius: 25px; text-decoration: none; font-weight: 600; font-size: 13px; box-shadow: 0 4px 15px rgba(25, 118, 210, 0.4); border: none; display: inline-flex; align-items: center; gap: 8px; transition: all 0.3s ease;">
                                    <img src="data:image/png;base64,{favicon_data}" style="width: 20px; height: 20px; margin-right: 0px; display: inline-block; vertical-align: middle; object-fit: contain;">
                                    <span class="desktop-text">PBJ Takeaway</span>
                                    <span class="mobile-text">PBJ Brief</span>
                                    <span class="arrow" style="font-size: 10px; opacity: 0.8;">→</span>
                                </a>
                            </div>
                            <script>
                            function fadePBJButton() {{
                                setTimeout(function() {{
                                    var button = document.getElementById('pbj-takeaway-button');
                                    if (button) {{
                                        button.style.transition = 'opacity 0.5s ease';
                                        button.style.opacity = '0';
                                        setTimeout(function() {{
                                            button.style.display = 'none';
                                        }}, 500);
                                    }}
                                }}, 100);
                            }}
                            </script>
                            <style>
                            .pbj-button-container {{
                                left: 35px;
                            }}
                            .pbj-button-container a {{
                                gap: 4px !important;
                            }}
                            .pbj-button-container img {{
                                margin-right: -2px !important;
                            }}
                            .pbj-button-container img {{
                                display: inline-block !important;
                                vertical-align: middle !important;
                                object-fit: contain !important;
                            }}
                            /* Mobile text display */
                            .mobile-text {{
                                display: none;
                            }}
                            .desktop-text {{
                                display: inline;
                            }}
                            @media (max-width: 768px) {{
                                .mobile-text {{
                                    display: inline;
                                }}
                                .desktop-text {{
                                    display: none;
                                }}
                                /* Mobile button styling - tighter and transparent */
                                .pbj-button-container {{
                                    right: 18.5px !important;
                                    top: 90px !important;
                                }}
                                /* State page button - target state pages specifically */
                                .stSelectbox .pbj-button-container,
                                div:has(select) .pbj-button-container,
                                .main:has(.stSelectbox) .pbj-button-container {{
                                    top: 50px !important;
                                }}
                                .pbj-button-container a {{
                                    padding: 2px 8px !important;
                                    background: rgba(25, 118, 210, 0.15) !important;
                                    color: #333333 !important;
                                    border: none !important;
                                    box-shadow: none !important;
                                    font-size: 11px !important;
                                    gap: 4px !important;
                                }}
                                .pbj-button-container img {{
                                    margin-right: -2px !important;
                                    width: 18px !important;
                                    height: 18px !important;
                                }}
                                /* Additional rule to move button up on state pages */
                                .stSelectbox .state-page-mobile-button,
                                div:has(select) .state-page-mobile-button,
                                .main:has(.stSelectbox) .state-page-mobile-button {{
                                    top: 50px !important;
                                }}

                            }}
                            @media (min-width: 768px) {{
                                .pbj-button-container {{
                                    left: 84px;
                                }}
                            }}
                            a[href="#pbj-takeaway"]:hover {{
                                transform: translateY(-2px);
                                box-shadow: 0 6px 20px rgba(25, 118, 210, 0.5);
                                background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
                            }}
                            </style>
                            """
                        st.markdown(button_html, unsafe_allow_html=True)
                    except Exception as e:
                        # Fallback without favicon if file can't be read
                        st.markdown("""
                        <div class="pbj-button-container" style="position: fixed; top: 80px; z-index: 1000;">
                            <a href="#pbj-takeaway" style="background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%); color: white; padding: 10px 18px; border-radius: 25px; text-decoration: none; font-weight: 600; font-size: 13px; box-shadow: 0 4px 15px rgba(25, 118, 210, 0.4); border: none; display: inline-flex; align-items: center; gap: 8px; transition: all 0.3s ease;">
                                <span class="desktop-text">PBJ Takeaway</span>
                                <span class="mobile-text">PBJ Brief</span>
                                <span class="arrow" style="font-size: 10px; opacity: 0.8;">→</span>
                            </a>
                        </div>
                        <style>
                        .pbj-button-container {
                            right: 30px;
                        }
                        @media (min-width: 768px) {
                            .pbj-button-container {
                                left: 65px;
                            }
                        }
                        a[href="#pbj-takeaway"]:hover {
                            transform: translateY(-2px);
                            box-shadow: 0 6px 20px rgba(25, 118, 210, 0.5);
                            background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
                        }
                        </style>
                        """, unsafe_allow_html=True)
                    
                    # 2. Display metrics
                    display_metrics(filtered_data, level)
                    
                    # Add spacer to prevent chart bleeding into metrics
                    st.markdown("""
                        <div style="height: 20px; margin: 0; padding: 0;"></div>
                    """, unsafe_allow_html=True)
                    
                    # 3. Display trends
                    fig = plot_quarterly_trends(filtered_data, 
                                          state=selected_value if level == "State" else None,
                                        facility=selected_value if level == "Facility" else None)
                    if fig:
                        # Add CSS to prevent chart from bleeding into metrics above
                        st.markdown("""
                            <style>
                            /* Prevent charts from bleeding into metrics above */
                            div[data-testid="stPlotlyChart"] {
                                margin-top: 20px !important;
                                padding-top: 10px !important;
                            }
                            /* Additional spacing for mobile */
                            @media (max-width: 768px) {
                                div[data-testid="stPlotlyChart"] {
                                    margin-top: 15px !important;
                                    padding-top: 8px !important;
                                }
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add case-mix charts for facility level
                        if level == "Facility":
                            # Get the most recent facility name from provider info data
                            try:
                                provider_df = pd.read_csv('NH_ProviderInfo_Jul2025.csv')
                                facility_info = provider_df[provider_df['CMS Certification Number (CCN)'] == selected_value]
                                if not facility_info.empty:
                                    facility_name = proper_title_case(facility_info.iloc[0]['Provider Name'])
                                else:
                                    facility_name = proper_title_case(selected_facility['PROVNAME'])
                            except:
                                facility_name = proper_title_case(selected_facility['PROVNAME'])
                            
                            case_mix_fig, _ = create_case_mix_charts(selected_value, quarter_label, facility_name)
                            if case_mix_fig:
                                st.plotly_chart(case_mix_fig, use_container_width=True)
                                
                                # Add small caption for case-mix explanation
                                st.markdown("""
                                <div style="text-align: left; margin: 2px 0; font-size: 0.7em; color: #666;" class="case-mix-caption">
                                    <em>Case-mix: Government benchmark for expected staffing based on resident needs. <a href="https://www.cms.gov/medicare/provider-enrollment-and-certification/certificationandcomplianc/downloads/usersguide.pdf" target="_blank" style="color: #1976d2;">Learn more</a></em>
                                </div>
                                <style>
                                @media (min-width: 768px) {
                                    .case-mix-caption {
                                        margin-top: -15px !important;
                                        text-align: center !important;
                                    }
                                }
                                </style>
                                """, unsafe_allow_html=True)
                        
                        # Add HPRD Explanation for facility level
                        if level == "Facility":
                            # Get the current HPRD value from the filtered data (most recent quarter)
                            if not filtered_data.empty:
                                latest_data = filtered_data.sort_values('CY_QTR', ascending=False).iloc[0]
                                reported_hprd = latest_data['Total_Nurse_HPRD']
                                quarter = latest_data['CY_QTR'][-1]
                                year = latest_data['CY_QTR'][:4]
                            else:
                                reported_hprd = selected_facility['Total_Nurse_HPRD']
                                quarter = "1"
                                year = "2025"
                            
                            # Get state average HPRD
                            state_metrics = pd.read_csv('state_lite_metrics.csv')
                            state_avg = state_metrics[state_metrics['STATE'] == selected_facility['STATE']]['Total_Nurse_HPRD'].iloc[0] if not state_metrics[state_metrics['STATE'] == selected_facility['STATE']].empty else 3.5
                            
                            # Get case-mix expected HPRD
                            try:
                                provider_df = pd.read_csv('NH_ProviderInfo_Jul2025.csv')
                                facility_info = provider_df[provider_df['CMS Certification Number (CCN)'] == selected_value]
                                if not facility_info.empty:
                                    case_mix_hprd = facility_info.iloc[0]['Case-Mix Total Nurse Staffing Hours per Resident per Day']
                                else:
                                    case_mix_hprd = reported_hprd  # fallback
                            except:
                                case_mix_hprd = reported_hprd  # fallback
                            
                            # Get the most recent facility name from provider info data
                            try:
                                if not facility_info.empty:
                                    facility_name = proper_title_case(facility_info.iloc[0]['Provider Name'])
                                else:
                                    facility_name = proper_title_case(selected_facility['PROVNAME'])
                            except:
                                facility_name = proper_title_case(selected_facility['PROVNAME'])
                            
                            # Calculate values
                            res_per_staff = 24 / reported_hprd if reported_hprd > 0 else 0
                            floor_staff_total = 30 * reported_hprd / 24 if reported_hprd > 0 else 0
                            floor_staff_aides = floor_staff_total * 0.60
                            
                            # Determine comparisons
                            if abs(reported_hprd - state_avg) < 0.1:
                                comparison_to_state = "around"
                            elif reported_hprd < state_avg:
                                comparison_to_state = "below"
                            else:
                                comparison_to_state = "above"
                                
                            if abs(reported_hprd - case_mix_hprd) < 0.1:
                                comparison_to_casemix = "around"
                            elif reported_hprd < case_mix_hprd:
                                comparison_to_casemix = "below"
                            else:
                                comparison_to_casemix = "above"
                            
                            # Get full state name
                            state_full_name = get_full_state_name(selected_facility['STATE'])
                            
                            # Calculate trend from previous quarter
                            trend_delta = None
                            if not filtered_data.empty and len(filtered_data) > 1:
                                # Get the two most recent quarters
                                recent_data = filtered_data.sort_values('CY_QTR', ascending=False).head(2)
                                if len(recent_data) == 2:
                                    current_hprd = recent_data.iloc[0]['Total_Nurse_HPRD']
                                    previous_hprd = recent_data.iloc[1]['Total_Nurse_HPRD']
                                    trend_delta = current_hprd - previous_hprd
                            

                            
                            # Get census and contract from the most recent quarter data
                            if not filtered_data.empty:
                                latest_data = filtered_data.sort_values('CY_QTR', ascending=False).iloc[0]
                                census_value = str(int(latest_data['Census'])) if 'Census' in latest_data and pd.notna(latest_data['Census']) else "—"
                                contract_value = f"{latest_data['Contract_Percentage']:.1f}%" if 'Contract_Percentage' in latest_data and pd.notna(latest_data['Contract_Percentage']) else "—"
                            else:
                                census_value = str(int(selected_facility['Census'])) if 'Census' in selected_facility and pd.notna(selected_facility['Census']) else "—"
                                contract_value = f"{selected_facility['Contract_Percentage']:.1f}%" if 'Contract_Percentage' in selected_facility and pd.notna(selected_facility['Contract_Percentage']) else "—"
                            
                            # Add anchor for PBJ Takeaway section with higher positioning
                            st.markdown('<div id="pbj-takeaway" style="margin-top: -110px; padding-top: 60px;"></div>', unsafe_allow_html=True)
                            
                            # Use the new PBJ Takeaway card
                            pbj_takeaway_card(
                                facility=facility_name,
                                reported_hprd=reported_hprd,
                                quarter_label=f"Q{quarter} {year}",
                                state_name=state_full_name,
                                state_hprd=state_avg,
                                casemix_hprd=case_mix_hprd,
                                census=census_value,
                                contract=contract_value,
                                trend_delta=trend_delta,
                                census_trend=None,  # You can add actual census trend data here
                                previous_quarter="Q1 2024"  # Previous quarter for comparison
                            )
                        
                        # Add CMS Care Compare link below the chart for facility level
                        if level == "Facility":
                            care_compare_url = f"https://www.medicare.gov/care-compare/details/nursing-home/{selected_value}/view-all?state={selected_facility['STATE']}"
                            # Use the same facility_name that was already defined in the PBJ Takeaway section above
                            
                            st.markdown(f"""
                            <div style='text-align: center; margin-top: 15px;'>
                                <a href='{care_compare_url}' target='_blank' style='background: #e8f4fd; color: #1976d2; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-weight: 500; border: 1px solid #1976d2; display: inline-block;'>
                                    <span class="desktop-text">View {proper_title_case(facility_name)} Details on Federal CMS Care Compare Website</span>
                                    <span class="mobile-text">View Details on CMS Care Compare</span>
                                </a>
                            </div>
                            <style>
                                @media (max-width: 768px) {{
                                    .desktop-text {{ display: none !important; }}
                                }}
                                @media (min-width: 769px) {{
                                    .mobile-text {{ display: none !important; }}
                                }}
                            </style>
                            """, unsafe_allow_html=True)

                    # 4. Add subscription button
                    display_subscription_button("facility", selected_value, selected_facility['PROVNAME'])

            # For entity level, display full entity content (verbatim from ownership page)
            elif level == "Entity" and selected_value and "pending_navigation" not in st.session_state and not st.query_params.get('state') and not st.query_params.get('facility'):
                # Load current and previous data for comparison
                entity_data = load_affiliated_entity_data()
                previous_entity_data = load_previous_ownership_data()
                provider_data = load_provider_info_data()
                
                if not entity_data.empty and not provider_data.empty:
                    # Find the selected entity by name
                    selected_entity_data = entity_data[entity_data['Chain'] == selected_value]
                    if not selected_entity_data.empty:
                        entity_row = selected_entity_data.iloc[0]
                        entity_id = int(entity_row['Chain ID'])
                        
                        # Find previous month's data for comparison
                        # March data uses 'Affiliated entity' but we need to map it to 'Chain' for comparison
                        try:
                            # Check what columns are available in previous data
                            available_columns = list(previous_entity_data.columns)
                            
                            # Try with mapped column name first
                            if 'Chain' in available_columns:
                                previous_entity_data_filtered = previous_entity_data[previous_entity_data['Chain'] == selected_value]
                            elif 'Affiliated entity' in available_columns:
                                previous_entity_data_filtered = previous_entity_data[previous_entity_data['Affiliated entity'] == selected_value]
                            else:
                                # If neither column exists, skip comparison
                                previous_entity_data_filtered = pd.DataFrame()
                            
                            previous_row = previous_entity_data_filtered.iloc[0] if not previous_entity_data_filtered.empty else None
                        except Exception as e:
                            # If any error occurs, skip comparison
                            previous_row = None
                        
                        # Main entity dashboard with entity ID
                        entity_name_title_case = proper_title_case(selected_value)
                        
                        # Get the most recent data period (July 2025)
                        most_recent_period = "July 2025"  # This could be made dynamic based on data
                        
                        # Responsive header with mobile optimization
                        st.markdown(f'''
                                <div style="background: linear-gradient(90deg, #e3ecfa 80%, #dbeafe 100%); color: #1a2233; padding: 0.5rem 2rem; border-radius: 12px; margin-bottom: 0.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #d3dbe8;">
                                    <h2 style="margin-bottom: -0.2em; font-size: 2.2em; font-weight: 700; letter-spacing: 0.01em; color: #1a2233; line-height: 1.0;">{entity_name_title_case} <span class="desktop-id" style="font-size: 0.7em; font-weight: 400; color: #4b5563;">(ID: {entity_id})</span></h2>
                                    <div style="font-size: 0.8em; color: #666; margin-top: 0;">Source: CMS, {most_recent_period}</div>
                                </div>
                                <style>
                                    @media (max-width: 768px) {{
                                        .desktop-id {{ display: none !important; }}
                                    }}
                                </style>
                            ''', unsafe_allow_html=True)
                        # Key metrics overview with comparison to previous month
                        # Add spacing before the metrics row on mobile to prevent ownership chart overlap
                        if st.session_state.get('is_mobile', False):
                            st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
                        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1.5])
                        with col1:
                            current_facilities = entity_row['Number of facilities']
                            prev_facilities = previous_row['Number of facilities'] if previous_row is not None else None
                            delta_facilities = current_facilities - prev_facilities if prev_facilities is not None else None
                            
                            # For zero deltas, show neutral dash like the others
                            if delta_facilities == 0:
                                delta_display = "—"  # Neutral dash
                            elif delta_facilities is not None:
                                delta_display = format_metric(delta_facilities, decimal_places=0, thousands=True)
                            else:
                                delta_display = None
                                
                            st.metric("Total Facilities", 
                                     format_metric(current_facilities, decimal_places=0, thousands=True),
                                     delta_display,
                                     help="Total number of nursing homes owned by this entity (vs. March 2025)")
                        with col2:
                            current_states = entity_row['Number of states and territories with operations']
                            prev_states = previous_row['Number of states and territories with operations'] if previous_row is not None else None
                            delta_states = current_states - prev_states if prev_states is not None else None
                            
                            # Show neutral indicator for zero deltas
                            if delta_states == 0:
                                delta_display = "—"  # Neutral dash
                            elif delta_states is not None:
                                delta_display = format_metric(delta_states, decimal_places=0)
                            else:
                                delta_display = None
                                
                            st.metric("States of Operation", 
                                     format_metric(current_states, decimal_places=0),
                                     delta_display,
                                     help="Number of states where this entity operates nursing homes (vs. March 2025)")
                        with col3:
                            current_rating = entity_row['Average overall 5-star rating']
                            prev_rating = previous_row['Average overall 5-star rating'] if previous_row is not None else None
                            delta_rating = current_rating - prev_rating if prev_rating is not None else None
                            
                            # Show neutral indicator for zero deltas
                            if delta_rating == 0:
                                delta_display = "—"  # Neutral dash
                            elif delta_rating is not None:
                                delta_display = format_metric(delta_rating, decimal_places=1)
                            else:
                                delta_display = None
                                
                            st.metric("Overall Rating", 
                                     format_metric(current_rating, decimal_places=1),
                                     delta_display,
                                     help="Average CMS 5-star overall rating across all facilities (vs. March 2025)")
                        with col4:
                            current_fines = entity_row['Total amount of fines in dollars']
                            prev_fines = previous_row['Total amount of fines in dollars'] if previous_row is not None else None
                            delta_fines = current_fines - prev_fines if prev_fines is not None else None
                            
                            # For fines, show neutral indicator for zero deltas
                            if delta_fines == 0:
                                delta_display = "—"  # Neutral dash
                            elif delta_fines is not None:
                                # For fines, show absolute change (not relative to good/bad)
                                delta_display = f"${format_metric(abs(delta_fines), decimal_places=0, thousands=True)}"
                            else:
                                delta_display = None
                                
                            # Format Total Fines - show millions if over 1M
                            def format_fines_display(amount):
                                if amount >= 1000000:
                                    return f"${amount/1000000:.1f} million"
                                else:
                                    return f"${format_metric(amount, decimal_places=0, thousands=True)}"
                            
                            st.metric("Total Fines", 
                                     format_fines_display(current_fines),
                                     delta_display,
                                     help="Total amount of fines in dollars across all facilities (vs. March 2025)")
                        with col5:
                            # Add spacing before ownership chart on desktop only
                            if not st.session_state.get('is_mobile', False):
                                st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                            # Ownership breakdown pie chart (no heading on desktop)
                            if st.session_state.get('is_mobile', False):
                                st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Ownership Type</h3></div>', unsafe_allow_html=True)
                                # Add spacing before the chart on mobile to prevent overlap with metrics above
                                st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
                            
                            # Pie chart for ownership with embedded data
                            def safe_pct(val):
                                try:
                                    v = float(val)
                                    return v if pd.notna(v) else 0.0
                                except Exception:
                                    return 0.0
                            for_profit = safe_pct(entity_row.get('Percent of facilities classified as for-profit', 0))
                            non_profit = safe_pct(entity_row.get('Percent of facilities classified as non-profit', 0))
                            government = safe_pct(entity_row.get('Percent of facilities classified as government-owned', 0))
                            
                            # Custom tooltip text for each slice
                            def format_hover_pct(value):
                                if pd.isna(value) or value == 0.0:
                                    return "N/A"
                                return f"{value:.1f}%"
                            
                            def format_ownership_pct(value):
                                if pd.isna(value) or value is None:
                                    return "N/A"
                                return f"{value:.1f}%"
                            
                            pie_labels = [f'For-Profit ({format_ownership_pct(for_profit)})', f'Non-Profit ({format_ownership_pct(non_profit)})', f'Government ({format_ownership_pct(government)})']
                            pie_values = [for_profit, non_profit, government]
                            
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
                                height=140,
                                showlegend=True,
                                margin=dict(l=10, r=10, t=10, b=10)
                            )
                            
                            # Add 320 Consulting badge
                            fig_pie.add_annotation(
                                text="<b>320 Consulting</b> | Source: CMS Provider Info (July 2025)",
                                x=0.99,
                                y=-0.45,
                                xref="x domain",
                                yref="y domain",
                                showarrow=False,
                                font=dict(size=10, color="#666666"),
                                align="right"
                            )
                            
                            st.plotly_chart(fig_pie, use_container_width=True)
                        # Calculate number of 1-star facilities for this entity
                        num_1star = 0
                        if entity_id and entity_id != "":
                            entity_facilities = provider_data[
                                provider_data['Chain ID'] == entity_id
                            ].copy()
                            if not entity_facilities.empty:
                                num_1star = (entity_facilities['Overall Rating'] == 1).sum()

                        # High Risk Facilities with comparison
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em; margin-bottom:0.2em;">High-Risk Facilities - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
                        with risk_col1:
                            current_sff = entity_row['Number of Special Focus Facilities (SFF)']
                            prev_sff = previous_row['Number of Special Focus Facilities (SFF)'] if previous_row is not None else None
                            delta_sff = current_sff - prev_sff if prev_sff is not None else None
                            
                            # Show neutral indicator for zero deltas
                            if delta_sff == 0:
                                delta_display = "—"  # Neutral dash
                            elif delta_sff is not None:
                                delta_display = format_metric(delta_sff, decimal_places=0)
                            else:
                                delta_display = None
                                
                            st.metric("Special Focus Facilities (SFFs)", 
                                     format_metric(current_sff, decimal_places=0),
                                     delta_display,
                                     help="Special Focus Facilities are nursing homes with serious quality issues under CMS oversight (vs. March 2025)")
                        with risk_col2:
                            current_sff_candidate = entity_row['Number of SFF candidates']
                            prev_sff_candidate = previous_row['Number of SFF candidates'] if previous_row is not None else None
                            delta_sff_candidate = current_sff_candidate - prev_sff_candidate if prev_sff_candidate is not None else None
                            
                            # Show neutral indicator for zero deltas
                            if delta_sff_candidate == 0:
                                delta_display = "—"  # Neutral dash
                            elif delta_sff_candidate is not None:
                                delta_display = format_metric(delta_sff_candidate, decimal_places=0)
                            else:
                                delta_display = None
                                
                            st.metric("SFF Candidates", 
                                     format_metric(current_sff_candidate, decimal_places=0),
                                     delta_display,
                                     help="Facilities monitored for potential SFF designation (vs. March 2025)")
                        with risk_col3:
                            current_abuse = entity_row['Number of facilities with an abuse icon']
                            prev_abuse = previous_row['Number of facilities with an abuse icon'] if previous_row is not None else None
                            delta_abuse = current_abuse - prev_abuse if prev_abuse is not None else None
                            
                            # Show neutral indicator for zero deltas
                            if delta_abuse == 0:
                                delta_display = "—"  # Neutral dash
                            elif delta_abuse is not None:
                                delta_display = format_metric(delta_abuse, decimal_places=0)
                            else:
                                delta_display = None
                                
                            # Calculate percentage for abuse facilities
                            total_facilities = entity_row['Number of facilities']
                            abuse_percentage = (current_abuse / total_facilities * 100) if total_facilities > 0 else 0
                            abuse_display = f"{current_abuse} ({abuse_percentage:.1f}%)"
                            
                            st.metric("Facilited Cited for Abuse", 
                                     abuse_display,
                                     delta_display,
                                     help="Facilities cited for abuse (vs. March 2025)")
                        with risk_col4:
                            # Calculate 1-star comparison from previous data
                            prev_1star = 0
                            if previous_row is not None:
                                # Get previous 1-star count by looking up facilities with this chain ID in previous provider data
                                try:
                                    # Load previous provider data for comparison
                                    march_provider_data = load_march_provider_info_data()
                                    if not march_provider_data.empty and entity_id:
                                        march_chain_facilities = march_provider_data[
                                            march_provider_data['Affiliated Entity ID'] == entity_id
                                        ].copy()
                                        if not march_chain_facilities.empty:
                                            prev_1star = (march_chain_facilities['Overall Rating'] == 1).sum()
                                except:
                                    prev_1star = 0
                            
                            delta_1star = num_1star - prev_1star
                            
                            # Show neutral indicator for zero deltas
                            if delta_1star == 0:
                                delta_display = "—"  # Neutral dash
                            elif delta_1star is not None:
                                delta_display = format_metric(delta_1star, decimal_places=0)
                            else:
                                delta_display = None
                                
                            # Calculate percentage for 1-star facilities
                            total_facilities = entity_row['Number of facilities']
                            one_star_percentage = (num_1star / total_facilities * 100) if total_facilities > 0 else 0
                            one_star_display = f"{num_1star} ({one_star_percentage:.1f}%)"
                            
                            st.metric("1-Star Rating Facilities", 
                                     one_star_display,
                                     delta_display,
                                     help="Facilities with the lowest CMS overall rating (vs. March 2025)")
                        

                        
                        # Single column layout for detailed metrics
                        
                        # CMS 5-Star Rating section
                        def safe_pct(val):
                            try:
                                v = float(val)
                                return v if pd.notna(v) else 0.0
                            except Exception:
                                return 0.0
                        for_profit = safe_pct(entity_row.get('Percent of facilities classified as for-profit', 0))
                        non_profit = safe_pct(entity_row.get('Percent of facilities classified as non-profit', 0))
                        government = safe_pct(entity_row.get('Percent of facilities classified as government-owned', 0))
                        
                        # Custom tooltip text for each slice
                        def format_hover_pct(value):
                            if pd.isna(value) or value == 0.0:
                                return "N/A"
                            return f"{value:.1f}%"
                        
                        def format_ownership_pct(value):
                            if pd.isna(value) or value is None:
                                return "N/A"
                            return f"{value:.1f}%"
                        
                        pie_labels = [f'For-Profit ({format_ownership_pct(for_profit)})', f'Non-Profit ({format_ownership_pct(non_profit)})', f'Government ({format_ownership_pct(government)})']
                        pie_values = [for_profit, non_profit, government]
                        
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
                        

                        
                        st.markdown(f'<div class="section-header" style="font-size:1.05em; margin-top: 25px;"><h3 style="font-size:1.15em;">Avg. CMS 5-Star Rating - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
                        with qual_col1:
                            # Overall rating delta
                            current_overall = entity_row['Average overall 5-star rating']
                            prev_overall = previous_row['Average overall 5-star rating'] if previous_row is not None else None
                            overall_delta = current_overall - prev_overall if prev_overall is not None else None
                            
                            if overall_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif overall_delta is not None:
                                delta_display = f"{overall_delta:.1f}"
                            else:
                                delta_display = None
                                
                            st.metric("Overall", f"{current_overall:.1f}", delta_display, help="Average overall 5-star rating (vs. March 2025)")
                        with qual_col2:
                            # Health inspection rating delta
                            current_health = entity_row['Average health inspection rating']
                            prev_health = previous_row['Average health inspection rating'] if previous_row is not None else None
                            health_delta = current_health - prev_health if prev_health is not None else None
                            
                            if health_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif health_delta is not None:
                                delta_display = f"{health_delta:.1f}"
                            else:
                                delta_display = None
                                
                            st.metric("Health Inspection", f"{current_health:.1f}", delta_display, help="Average health inspection rating (vs. March 2025)")
                        with qual_col3:
                            # Staffing rating delta
                            current_staffing = entity_row['Average staffing rating']
                            prev_staffing = previous_row['Average staffing rating'] if previous_row is not None else None
                            staffing_delta = current_staffing - prev_staffing if prev_staffing is not None else None
                            
                            if staffing_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif staffing_delta is not None:
                                delta_display = f"{staffing_delta:.1f}"
                            else:
                                delta_display = None
                                
                            st.metric("Staffing", f"{current_staffing:.1f}", delta_display, help="Average staffing rating (vs. March 2025)")
                        with qual_col4:
                            # Quality rating delta
                            current_quality = entity_row['Average quality rating']
                            prev_quality = previous_row['Average quality rating'] if previous_row is not None else None
                            quality_delta = current_quality - prev_quality if prev_quality is not None else None
                            
                            if quality_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif quality_delta is not None:
                                delta_display = f"{quality_delta:.1f}"
                            else:
                                delta_display = None
                                
                            st.metric("Quality", f"{current_quality:.1f}", delta_display, help="Average quality rating (vs. March 2025)")
                        # Quality ratings chart and distribution chart side by side
                        # Add spacing before charts to prevent overlap with metrics above
                        st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
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
                                margin=dict(l=10, r=10, t=50, b=10),
                                bargap=0.35
                            )
                            
                            # Add 320 Consulting badge
                            fig.add_annotation(
                                text="<b>320 Consulting</b> | Source: CMS Provider Info (July 2025)",
                                x=0.99,
                                y=-0.45,
                                xref="x domain",
                                yref="y domain",
                                showarrow=False,
                                font=dict(size=10, color="#666666"),
                                align="right"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        with chart_col2:
                            # Facility ratings breakdown pie chart
                            if entity_id and entity_id != "":
                                entity_facilities = provider_data[
                                    provider_data['Chain ID'] == entity_id
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
                                        margin=dict(l=10, r=10, t=50, b=10),
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
                                    
                                    # Add 320 Consulting badge
                                    fig_ratings.add_annotation(
                                        text="<b>320 Consulting</b> | Source: CMS Provider Info (July 2025)",
                                        x=0.99,
                                        y=-0.45,
                                        xref="x domain",
                                        yref="y domain",
                                        showarrow=False,
                                        font=dict(size=10, color="#666666"),
                                        align="right"
                                    )
                                    title_col1, title_col2, title_col3 = st.columns([0.15, 0.7, 0.15])
                                    with title_col2:
                                        st.markdown('<div style="text-align:center; font-size:1em; font-weight:400; color:#444; margin-bottom:0.2em;">CMS 5-Star Rating Distribution</div>', unsafe_allow_html=True)
                                    # Add spacing before the chart to prevent cutting off the row above
                                    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
                                    st.plotly_chart(fig_ratings, use_container_width=True)
                                    

                        
                        # Staffing metrics
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Avg. Staffing Levels - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        
                        staff_col1, staff_col2, staff_col3, staff_col4 = st.columns(4)
                        with staff_col1:
                            # Total Nurse HPRD delta
                            current_total_hprd = entity_row['Average total nurse hours per resident day']
                            prev_total_hprd = previous_row['Average total nurse hours per resident day'] if previous_row is not None else None
                            total_hprd_delta = current_total_hprd - prev_total_hprd if prev_total_hprd is not None else None
                            
                            if total_hprd_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif total_hprd_delta is not None:
                                delta_display = f"{total_hprd_delta:.1f}"
                            else:
                                delta_display = None
                                
                            st.metric("Total Nurse HPRD", f"{current_total_hprd:.1f}", delta_display, help="Average total nurse hours per resident day (vs. March 2025)")
                        with staff_col2:
                            # RN HPRD delta
                            current_rn_hprd = entity_row['Average total Registered Nurse hours per resident day']
                            prev_rn_hprd = previous_row['Average total Registered Nurse hours per resident day'] if previous_row is not None else None
                            rn_hprd_delta = current_rn_hprd - prev_rn_hprd if prev_rn_hprd is not None else None
                            
                            if rn_hprd_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif rn_hprd_delta is not None:
                                delta_display = f"{rn_hprd_delta:.1f}"
                            else:
                                delta_display = None
                                
                            st.metric("RN HPRD", f"{current_rn_hprd:.1f}", delta_display, help="Average RN hours per resident day (vs. March 2025)")
                        with staff_col3:
                            # Weekend HPRD delta
                            current_weekend_hprd = entity_row['Average total weekend nurse hours per resident day']
                            prev_weekend_hprd = previous_row['Average total weekend nurse hours per resident day'] if previous_row is not None else None
                            weekend_hprd_delta = current_weekend_hprd - prev_weekend_hprd if prev_weekend_hprd is not None else None
                            
                            if weekend_hprd_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif weekend_hprd_delta is not None:
                                delta_display = f"{weekend_hprd_delta:.1f}"
                            else:
                                delta_display = None
                                
                            st.metric("Weekend HPRD", f"{current_weekend_hprd:.1f}", delta_display, help="The level of staffing on weekends provided by each nursing home over a quarter.")
                        with staff_col4:
                            # Admin Turnover delta
                            current_admin_turnover = entity_row['Average number of administrators who have left the nursing home']
                            prev_admin_turnover = previous_row['Average number of administrators who have left the nursing home'] if previous_row is not None else None
                            admin_turnover_delta = current_admin_turnover - prev_admin_turnover if prev_admin_turnover is not None else None
                            
                            if admin_turnover_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif admin_turnover_delta is not None:
                                delta_display = f"{admin_turnover_delta:.1f}"
                            else:
                                delta_display = None
                                
                            st.metric("Admin Turnover", f"{current_admin_turnover:.1f}", delta_display, help="Number of administrators that stopped working at the nursing home over a 12-month period (vs March 2025)")
                        
                        # Turnover metrics
                        turn_col1, turn_col2 = st.columns(2)
                        with turn_col1:
                            def format_turnover_pct(value):
                                if pd.isna(value) or value is None:
                                    return "N/A"
                                return f"{value:.1f}%"
                            
                            # Nursing Staff Turnover delta
                            current_nursing_turnover = entity_row['Average total nursing staff turnover percentage']
                            prev_nursing_turnover = previous_row['Average total nursing staff turnover percentage'] if previous_row is not None else None
                            nursing_turnover_delta = current_nursing_turnover - prev_nursing_turnover if prev_nursing_turnover is not None else None
                            
                            if nursing_turnover_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif nursing_turnover_delta is not None:
                                delta_display = f"{nursing_turnover_delta:.1f}%"
                            else:
                                delta_display = None
                                
                            st.metric("Nursing Staff Turnover", format_turnover_pct(current_nursing_turnover), delta_display, help="The percent of nursing staff that stopped working at the nursing home over a 12-month period (vs March 2025)")
                        with turn_col2:
                            # RN Turnover delta
                            current_rn_turnover = entity_row['Average Registered Nurse turnover percentage']
                            prev_rn_turnover = previous_row['Average Registered Nurse turnover percentage'] if previous_row is not None else None
                            rn_turnover_delta = current_rn_turnover - prev_rn_turnover if prev_rn_turnover is not None else None
                            
                            if rn_turnover_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif rn_turnover_delta is not None:
                                delta_display = f"{rn_turnover_delta:.1f}%"
                            else:
                                delta_display = None
                                
                            st.metric("RN Turnover", format_turnover_pct(current_rn_turnover), delta_display, help="The percent of RN staff that stopped working at the nursing home over a 12-month period (vs March 2025)")
                        
                        # Compliance metrics
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Enforcement - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        
                        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
                        with comp_col1:
                            # Format Total Fines - show millions if over 1M
                            def format_fines_display(amount):
                                if amount >= 1000000:
                                    return f"${amount/1000000:.1f} million"
                                else:
                                    return f"${amount:,.0f}"
                            
                            # Total Fines delta
                            current_total_fines = entity_row['Total amount of fines in dollars']
                            prev_total_fines = previous_row['Total amount of fines in dollars'] if previous_row is not None else None
                            total_fines_delta = current_total_fines - prev_total_fines if prev_total_fines is not None else None
                            
                            if total_fines_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif total_fines_delta is not None:
                                delta_display = format_fines_display(abs(total_fines_delta))
                            else:
                                delta_display = None
                                
                            st.metric("Total Fines", format_fines_display(current_total_fines), delta_display, help="Total amount of fines in dollars (vs. March 2025)")
                        with comp_col2:
                            # Avg Fines per Facility delta
                            current_avg_fines = entity_row['Average amount of fines in dollars']
                            prev_avg_fines = previous_row['Average amount of fines in dollars'] if previous_row is not None else None
                            avg_fines_delta = current_avg_fines - prev_avg_fines if prev_avg_fines is not None else None
                            
                            if avg_fines_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif avg_fines_delta is not None:
                                delta_display = f"${abs(avg_fines_delta):,.0f}"
                            else:
                                delta_display = None
                                
                            st.metric("Avg Fines per Facility", f"${current_avg_fines:,.0f}", delta_display, help="Average fines per facility (vs. March 2025)")
                        with comp_col3:
                            # Total Payment Denials delta
                            current_denials = entity_row['Total number of payment denials']
                            prev_denials = previous_row['Total number of payment denials'] if previous_row is not None else None
                            denials_delta = current_denials - prev_denials if prev_denials is not None else None
                            
                            if denials_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif denials_delta is not None:
                                delta_display = f"{abs(denials_delta):,.0f}"
                            else:
                                delta_display = None
                                
                            st.metric("Total Payment Denials", f"{current_denials:,.0f}", delta_display, help="Total number of payment denials (vs. March 2025)")
                        with comp_col4:
                            # Avg Payment Denials delta
                            current_avg_denials = entity_row['Average number of payment denials']
                            prev_avg_denials = previous_row['Average number of payment denials'] if previous_row is not None else None
                            avg_denials_delta = current_avg_denials - prev_avg_denials if prev_avg_denials is not None else None
                            
                            if avg_denials_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif avg_denials_delta is not None:
                                delta_display = f"{abs(avg_denials_delta):.1f}"
                            else:
                                delta_display = None
                                
                            st.metric("Avg Payment Denials", f"{current_avg_denials:.1f}", delta_display, help="Average number of payment denials (vs. March 2025)")
                        
                        # Antipsychotic usage
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Antipsychotics - {entity_name_title_case}</h3></div>', unsafe_allow_html=True)
                        
                        anti_col1, anti_col2 = st.columns(2)
                        with anti_col1:
                            def format_antipsychotic_pct(value):
                                if pd.isna(value) or value is None:
                                    return "N/A"
                                return f"{value:.1f}%"
                            
                            # Short-Stay Antipsychotic delta
                            current_short_stay = entity_row['Average percentage of short-stay residents who newly received an antipsychotic medication']
                            prev_short_stay = previous_row['Average percentage of short-stay residents who newly received an antipsychotic medication'] if previous_row is not None else None
                            short_stay_delta = current_short_stay - prev_short_stay if prev_short_stay is not None else None
                            
                            if short_stay_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif short_stay_delta is not None:
                                delta_display = f"{abs(short_stay_delta):.1f}%"
                            else:
                                delta_display = None
                                
                            st.metric("Short-Stay Antipsychotic", format_antipsychotic_pct(current_short_stay), delta_display, help="Short-stay residents receiving antipsychotics (vs. March 2025)")
                        with anti_col2:
                            # Long-Stay Antipsychotic delta
                            current_long_stay = entity_row['Average percentage of long-stay residents who received an antipsychotic medication']
                            prev_long_stay = previous_row['Average percentage of long-stay residents who received an antipsychotic medication'] if previous_row is not None else None
                            long_stay_delta = current_long_stay - prev_long_stay if prev_long_stay is not None else None
                            
                            if long_stay_delta == 0:
                                delta_display = "—"  # Neutral dash
                            elif long_stay_delta is not None:
                                delta_display = f"{abs(long_stay_delta):.1f}%"
                            else:
                                delta_display = None
                                
                            st.metric("Long-Stay Antipsychotic", format_antipsychotic_pct(current_long_stay), delta_display, help="Long-stay residents receiving antipsychotics (vs. March 2025)")
                        
                        # Facilities list
                        st.markdown(f'<div class="section-header" style="font-size:1.05em;"><h3 style="font-size:1.15em;">Nursing homes affiliated with {selected_value}</h3></div>', unsafe_allow_html=True)
                        
                        # Get facilities for this entity
                        if entity_id and entity_id != "":
                            entity_facilities = provider_data[
                                provider_data['Chain ID'] == entity_id
                            ].copy()
                            
                            if not entity_facilities.empty:
                                st.markdown(f"**{len(entity_facilities)} facilities found**")
                                
                                # Get most recent HPRD and census data for each facility from PBJ database
                                def get_facility_latest_metrics(provnum_list):
                                    try:
                                        # Query the facility database for the most recent data for each facility
                                        placeholders = ','.join(['?' for _ in provnum_list])
                                        query = f"""
                                        SELECT PROVNUM, Total_Nurse_HPRD, Census, CY_QTR
                                        FROM facility_metrics 
                                        WHERE PROVNUM IN ({placeholders})
                                        AND (PROVNUM, CY_QTR) IN (
                                            SELECT PROVNUM, MAX(CY_QTR) 
                                            FROM facility_metrics 
                                            WHERE PROVNUM IN ({placeholders})
                                            GROUP BY PROVNUM
                                        )
                                        """
                                        result = facility_db.execute(query, provnum_list + provnum_list).fetchdf()
                                        return result
                                    except Exception as e:
                                        st.error(f"Error querying facility metrics: {str(e)}")
                                        return pd.DataFrame()
                                
                                # Get facility metrics for all facilities in this entity
                                provnum_list = entity_facilities['CMS Certification Number (CCN)'].tolist()
                                facility_metrics = get_facility_latest_metrics(provnum_list)
                                
                                # Prepare facilities data for display with City instead of County
                                facilities_display = entity_facilities[[
                                    'State',
                                    'City/Town',
                                    'Provider Name',
                                    'Overall Rating',
                                    'Staffing Rating',
                                    'Special Focus Status',
                                    'Abuse Icon'
                                ]].copy()
                                
                                # Add HPRD and Census data from facility metrics
                                if not facility_metrics.empty:
                                    # Merge with facility metrics to get HPRD and Census
                                    facilities_display = facilities_display.merge(
                                        facility_metrics[['PROVNUM', 'Total_Nurse_HPRD', 'Census']],
                                        left_on=entity_facilities['CMS Certification Number (CCN)'],
                                        right_on='PROVNUM',
                                        how='left'
                                    )
                                    # Drop the duplicate PROVNUM column
                                    facilities_display = facilities_display.drop('PROVNUM', axis=1)
                                    
                                    # Format HPRD and Census columns
                                    facilities_display['Total Nurse HPRD'] = facilities_display['Total_Nurse_HPRD'].apply(
                                        lambda x: f"{x:.1f}" if pd.notna(x) else 'N/A'
                                    )
                                    facilities_display['Census'] = facilities_display['Census'].apply(
                                        lambda x: f"{x:,.0f}" if pd.notna(x) else 'N/A'
                                    )
                                    # Drop the original column name
                                    facilities_display = facilities_display.drop('Total_Nurse_HPRD', axis=1)
                                else:
                                    # Add empty columns if no facility metrics available
                                    facilities_display['Total Nurse HPRD'] = 'N/A'
                                    facilities_display['Census'] = 'N/A'
                                
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
                                    lambda row: f'<a href="/?facility={format_provnum(entity_facilities.iloc[row.name]["CMS Certification Number (CCN)"])}" target="_blank">{row["Provider Name"]}</a>',
                                    axis=1
                                )
                                
                                # Reorder columns to include HPRD and Census
                                column_order = [
                                    'State',
                                    'Provider Name',
                                    'City',
                                    'Census',
                                    'Total Nurse HPRD',
                                    'Overall Rating',
                                    'Staffing Rating',
                                    'Special Focus Status',
                                    'Abuse Icon'
                                ]
                                facilities_display = facilities_display[column_order]
                                
                                # Render as HTML table for clickable links
                                # Render with df.to_html for custom sorting + links
                                html_table = facilities_display.to_html(
                                    index=False,
                                    escape=False,
                                    classes=['dataframe', 'table', 'table-striped'],
                                    table_id='facilities-table'
                                )
                                
                                # Add CSS for table styling
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
                                     table-layout: fixed;
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
                                     cursor: pointer;
                                     user-select: none;
                                     position: relative;
                                 }
                                 /* Column width rules */
                                 .dataframe th:nth-child(1) { width: 8%; }  /* State */
                                 .dataframe th:nth-child(2) { width: 25%; } /* Provider Name */
                                 .dataframe th:nth-child(3) { width: 12%; } /* City */
                                 .dataframe th:nth-child(4) { width: 8%; }  /* Census */
                                 .dataframe th:nth-child(5) { width: 12%; } /* Total Nurse HPRD */
                                 .dataframe th:nth-child(6) { width: 8%; }  /* Overall Rating */
                                 .dataframe th:nth-child(7) { width: 8%; }  /* Staffing Rating */
                                 .dataframe th:nth-child(8) { width: 12%; } /* Special Focus Status */
                                 .dataframe th:nth-child(9) { width: 7%; }  /* Abuse Icon */
                                 
                                 .dataframe td:nth-child(1) { width: 8%; }  /* State */
                                 .dataframe td:nth-child(2) { width: 25%; } /* Provider Name */
                                 .dataframe td:nth-child(3) { width: 12%; } /* City */
                                 .dataframe td:nth-child(4) { width: 8%; }  /* Census */
                                 .dataframe td:nth-child(5) { width: 12%; } /* Total Nurse HPRD */
                                 .dataframe td:nth-child(6) { width: 8%; }  /* Overall Rating */
                                 .dataframe td:nth-child(7) { width: 8%; }  /* Staffing Rating */
                                 .dataframe td:nth-child(8) { width: 12%; } /* Special Focus Status */
                                 .dataframe td:nth-child(9) { width: 7%; }  /* Abuse Icon */
                                 
                                 /* Mobile responsive design */
                                 @media (max-width: 768px) {
                                     .dataframe th:nth-child(7),
                                     .dataframe td:nth-child(7) {
                                         display: none !important; /* Hide Staffing Rating on mobile */
                                     }
                                     /* Adjust widths for mobile */
                                     .dataframe th:nth-child(1) { width: 8%; }  /* State */
                                     .dataframe th:nth-child(2) { width: 30%; } /* Provider Name */
                                     .dataframe th:nth-child(3) { width: 15%; } /* City */
                                     .dataframe th:nth-child(4) { width: 10%; } /* Census */
                                     .dataframe th:nth-child(5) { width: 10%; padding-left: 2px !important; padding-right: 2px !important; } /* Total Nurse HPRD */
                                     .dataframe th:nth-child(6) { width: 10%; } /* Overall Rating */
                                     .dataframe th:nth-child(8) { width: 10%; padding-left: 2px !important; } /* Special Focus Status */
                                     .dataframe th:nth-child(9) { width: 0%; }  /* Abuse Icon */
                                     
                                     .dataframe td:nth-child(1) { width: 8%; }  /* State */
                                     .dataframe td:nth-child(2) { width: 30%; } /* Provider Name */
                                     .dataframe td:nth-child(3) { width: 15%; } /* City */
                                     .dataframe td:nth-child(4) { width: 10%; } /* Census */
                                     .dataframe td:nth-child(5) { width: 10%; padding-left: 2px !important; padding-right: 2px !important; } /* Total Nurse HPRD */
                                     .dataframe td:nth-child(6) { width: 10%; } /* Overall Rating */
                                     .dataframe td:nth-child(8) { width: 10%; padding-left: 2px !important; } /* Special Focus Status */
                                     .dataframe td:nth-child(9) { width: 0%; }  /* Abuse Icon */
                                     
                                     /* Reduce font size on mobile */
                                     .dataframe {
                                         font-size: 0.7em !important;
                                     }
                                     .dataframe th {
                                         font-size: 0.7em !important;
                                     }
                                     .dataframe td {
                                         font-size: 0.7em !important;
                                     }
                                     
                                     /* Change "Provider Name" to "Provider" on mobile */
                                     .dataframe th:nth-child(2) {
                                         position: relative;
                                         color: transparent !important;
                                     }
                                     .dataframe th:nth-child(2)::before {
                                         content: "Provider";
                                         position: absolute;
                                         top: 0;
                                         left: 0;
                                         background: transparent;
                                         padding: 6px 4px;
                                         width: 100%;
                                         height: 100%;
                                         display: flex;
                                         align-items: center;
                                         font-size: inherit !important;
                                         text-transform: inherit !important;
                                         color: #495057 !important;
                                         font-weight: inherit !important;
                                         letter-spacing: inherit !important;
                                         z-index: 1;
                                     }
                                     
                                     /* Reduce left padding on State column */

                                     
                                     /* Show sort arrows on mobile */
                                     .dataframe th::after {
                                         content: "↕";
                                         font-size: 0.6em;
                                         color: #6c757d;
                                         position: absolute;
                                         right: 2px;
                                         top: 50%;
                                         transform: translateY(-50%);
                                         z-index: 2;
                                     }
                                     
                                     /* Give State column more space for arrow */
                                     .dataframe th:nth-child(1) {
                                         padding-right: 8px !important;
                                     }
                                     .dataframe th.sort-asc::after {
                                         content: "↑";
                                         color: #495057;
                                     }
                                     .dataframe th.sort-desc::after {
                                         content: "↓";
                                         color: #495057;
                                     }
                                     
                                     /* Override custom header backgrounds when sorted */
                                     .dataframe th:nth-child(2).sort-asc::before,
                                     .dataframe th:nth-child(2).sort-desc::before,
                                     .dataframe th:nth-child(6).sort-asc::before,
                                     .dataframe th:nth-child(6).sort-desc::before {
                                         background: #e3f2fd !important;
                                     }
                                     

                                     
                                     /* Use default sort highlighting (gray) */
                                     .dataframe th.sort-asc,
                                     .dataframe th.sort-desc {
                                         background: #e9ecef !important;
                                     }
                                     
                                     /* Override custom header backgrounds when sorted */
                                     .dataframe th.sort-asc::before,
                                     .dataframe th.sort-desc::before {
                                         background: #e9ecef !important;
                                     }
                                     

                                 }
                                 
                                 /* Fix column header capitalization */
                                 .dataframe th {
                                     text-transform: capitalize !important;
                                 }
                                 .dataframe th:hover {
                                     background: #e9ecef;
                                 }
                                 .dataframe th::after {
                                     content: ' ↕';
                                     font-size: 0.7em;
                                     color: #6c757d;
                                     position: absolute;
                                     right: 4px;
                                     top: 50%;
                                     transform: translateY(-50%);
                                 }
                                 .dataframe th.sort-asc::after {
                                     content: ' ↑';
                                     color: #007bff;
                                 }
                                 .dataframe th.sort-desc::after {
                                     content: ' ↓';
                                     color: #007bff;
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
                                
                                # Add sorting functionality using Tablesort library - only when actually on entity page
                                if (level == "Entity" and selected_value and 
                                    not st.query_params.get('state') and 
                                    not st.query_params.get('facility') and 
                                    not st.session_state.get('pending_navigation') and 
                                    "pending_navigation" not in st.session_state):
                                    from streamlit.components.v1 import html
                                    html('''
                                    <script src='https://cdnjs.cloudflare.com/ajax/libs/tablesort/5.0.2/tablesort.min.js'></script>
                                    <script>
                                        var table = window.parent.document.getElementById("facilities-table");
                                        if (table) {
                                            new Tablesort(table);
                                            console.log("Table sorting initialized with Tablesort");
                                        }
                                    </script>
                                    ''')
                                
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
                    
                    # Add PBJ Takeaway button for state level
                    if level == "State":
                        try:
                            with open('pbj_favicon.png', 'rb') as f:
                                favicon_data = base64.b64encode(f.read()).decode()
                            
                            # Create the button with favicon
                            state_button_class = "state-page-mobile-button" if level == "State" else ""
                            button_html = f"""
                            <div class="pbj-button-container {state_button_class}" id="pbj-takeaway-button" style="position: fixed; top: 80px; z-index: 1000;">
                                <a href="#pbj-takeaway" onclick="fadePBJButton()" style="background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%); color: white; padding: 10px 18px; border-radius: 25px; text-decoration: none; font-weight: 600; font-size: 13px; box-shadow: 0 4px 15px rgba(25, 118, 210, 0.4); border: none; display: inline-flex; align-items: center; gap: 8px; transition: all 0.3s ease;">
                                    <img src="data:image/png;base64,{favicon_data}" style="width: 20px; height: 20px; margin-right: 0px; display: inline-block; vertical-align: middle; object-fit: contain;">
                                    <span class="desktop-text">PBJ Takeaway</span>
                                    <span class="mobile-text">PBJ Brief</span>
                                    <span style="font-size: 10px; opacity: 0.8;">→</span>
                                </a>
                            </div>
                            <script>
                            function fadePBJButton() {{
                                setTimeout(function() {{
                                    var button = document.getElementById('pbj-takeaway-button');
                                    if (button) {{
                                        button.style.transition = 'opacity 0.5s ease';
                                        button.style.opacity = '0';
                                        setTimeout(function() {{
                                            button.style.display = 'none';
                                        }}, 500);
                                    }}
                                }}, 100);
                            }}
                            </script>
                            <style>
                            .pbj-button-container {{
                                left: 65px;
                            }}
                            .pbj-button-container a {{
                                gap: 4px !important;
                            }}
                            .pbj-button-container img {{
                                margin-right: -2px !important;
                            }}
                            .pbj-button-container img {{
                                display: inline-block !important;
                                vertical-align: middle !important;
                                object-fit: contain !important;
                            }}
                            /* Mobile text display */
                            .mobile-text {{
                                display: none;
                            }}
                            .desktop-text {{
                                display: inline;
                            }}
                            @media (max-width: 768px) {{
                                .mobile-text {{
                                    display: inline;
                                }}
                                .desktop-text {{
                                    display: none;
                                }}
                                /* Mobile button styling - tighter and transparent */
                                .pbj-button-container {{
                                    right: 18.5px !important;
                                    top: 90px !important;
                                }}
                                /* State page button - target state pages specifically */
                                .stSelectbox .pbj-button-container,
                                div:has(select) .pbj-button-container,
                                .main:has(.stSelectbox) .pbj-button-container {{
                                    top: 50px !important;
                                }}
                                .pbj-button-container a {{
                                    padding: 2px 8px !important;
                                    background: rgba(25, 118, 210, 0.15) !important;
                                    color: #333333 !important;
                                    border: none !important;
                                    box-shadow: none !important;
                                    font-size: 11px !important;
                                    gap: 4px !important;
                                }}
                                .pbj-button-container img {{
                                    margin-right: -2px !important;
                                    width: 18px !important;
                                    height: 18px !important;
                                }}
                                /* Additional rule to move button up on state pages */
                                .stSelectbox .state-page-mobile-button,
                                div:has(select) .state-page-mobile-button,
                                .main:has(.stSelectbox) .state-page-mobile-button {{
                                    top: 50px !important;
                                }}

                            }}
                            @media (min-width: 768px) {{
                                .pbj-button-container {{
                                    left: 84px;
                                }}
                            }}
                            a[href="#pbj-takeaway"]:hover {{
                                transform: translateY(-2px);
                                box-shadow: 0 6px 20px rgba(25, 118, 210, 0.5);
                                background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
                            }}
                            </style>
                            """
                            st.markdown(button_html, unsafe_allow_html=True)
                        except Exception as e:
                            # Fallback without favicon if file can't be read
                            st.markdown("""
                            <div class="pbj-button-container" id="pbj-takeaway-button" style="position: fixed; top: 80px; z-index: 1000;">
                                <a href="#pbj-takeaway" onclick="fadePBJButton()" style="background: linear-gradient(135deg, #1976d2 0%, #42a5f5 100%); color: white; padding: 10px 18px; border-radius: 25px; text-decoration: none; font-weight: 600; font-size: 13px; box-shadow: 0 4px 15px rgba(25, 118, 210, 0.4); border: none; display: inline-flex; align-items: center; gap: 8px; transition: all 0.3s ease;">
                                    <span class="desktop-text">PBJ Takeaway</span>
                                    <span class="mobile-text">PBJ Brief</span>
                                    <span style="font-size: 10px; opacity: 0.8;">→</span>
                                </a>
                            </div>
                            <script>
                            function fadePBJButton() {
                                setTimeout(function() {
                                    var button = document.getElementById('pbj-takeaway-button');
                                    if (button) {
                                        button.style.transition = 'opacity 0.5s ease';
                                        button.style.opacity = '0';
                                        setTimeout(function() {
                                            button.style.display = 'none';
                                        }, 500);
                                    }
                                }, 100);
                            }
                            </script>
                            <style>
                            .pbj-button-container {
                                right: 35px;
                            }
                            @media (min-width: 768px) {
                                .pbj-button-container {
                                    left: 65px;
                                }
                            }
                            a[href="#pbj-takeaway"]:hover {
                                transform: translateY(-2px);
                                box-shadow: 0 6px 20px rgba(25, 118, 210, 0.5);
                                background: linear-gradient(135deg, #1565c0 0%, #1976d2 100%);
                            }
                            </style>
                            """, unsafe_allow_html=True)
                    
                    # Add spacer to prevent chart bleeding into metrics
                    st.markdown("""
                        <div style="height: 20px; margin: 0; padding: 0;"></div>
                    """, unsafe_allow_html=True)
                    
                    fig = plot_quarterly_trends(filtered_data, 
                                              state=selected_value if level == "State" else None,
                                        facility=selected_value if level == "Facility" else None)
                    if fig:
                        # Add CSS to reduce padding above chart containers
                        st.markdown("""
                            <style>
                            /* Reduce padding above chart containers */
                            div[data-testid="stElementContainer"] {
                                margin-top: -10px !important;
                                padding-top: 0px !important;
                            }
                            /* Prevent charts from bleeding into metrics above */
                            div[data-testid="stPlotlyChart"] {
                                margin-top: 20px !important;
                                padding-top: 10px !important;
                            }
                            /* Additional spacing for mobile */
                            @media (max-width: 768px) {
                                div[data-testid="stPlotlyChart"] {
                                    margin-top: 15px !important;
                                    padding-top: 8px !important;
                                }
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Add state PBJ Takeaway card for state level
                    if level == "State" and selected_value:
                        # Add anchor for PBJ Takeaway section
                        st.markdown('<div id="pbj-takeaway" style="margin-top: -110px; padding-top: 60px;"></div>', unsafe_allow_html=True)
                        # Get state data from filtered data
                        if not filtered_data.empty:
                            latest_data = filtered_data.sort_values('CY_QTR', ascending=False).iloc[0]
                            state_hprd = latest_data['Total_Nurse_HPRD']
                            quarter = latest_data['CY_QTR'][-1]
                            year = latest_data['CY_QTR'][:4]
                            
                            # Get national average HPRD
                            national_metrics = pd.read_csv('national_lite_metrics.csv')
                            national_hprd = national_metrics['Total_Nurse_HPRD'].iloc[0] if not national_metrics.empty else 3.5
                            
                            # Calculate state rank for Q1 2025 only
                            state_metrics = pd.read_csv('state_lite_metrics.csv')
                            # Apply column mapping to ensure consistency
                            state_metrics.rename(columns={'CY_Qtr': 'CY_QTR'}, inplace=True)
                            q1_2025_data = state_metrics[state_metrics['CY_QTR'] == '2025Q1']
                            if not q1_2025_data.empty:
                                state_metrics_sorted = q1_2025_data.sort_values('Total_Nurse_HPRD', ascending=False).reset_index(drop=True)
                                state_row = state_metrics_sorted[state_metrics_sorted['STATE'] == selected_value]
                                state_rank = state_row.index[0] + 1 if not state_row.empty else 0
                                total_states = len(state_metrics_sorted)
                            else:
                                # Fallback to all data if Q1 2025 not available
                                state_metrics_sorted = state_metrics.sort_values('Total_Nurse_HPRD', ascending=False).reset_index(drop=True)
                                state_row = state_metrics_sorted[state_metrics_sorted['STATE'] == selected_value]
                                state_rank = state_row.index[0] + 1 if not state_row.empty else 0
                                total_states = len(state_metrics_sorted)
                            
                            # Calculate trend from previous quarter
                            trend_delta = None
                            if len(filtered_data) > 1:
                                recent_data = filtered_data.sort_values('CY_QTR', ascending=False).head(2)
                                if len(recent_data) == 2:
                                    current_hprd = recent_data.iloc[0]['Total_Nurse_HPRD']
                                    previous_hprd = recent_data.iloc[1]['Total_Nurse_HPRD']
                                    trend_delta = current_hprd - previous_hprd
                            
                            # Get full state name
                            state_full_name = get_full_state_name(selected_value)
                            
                            # Get average facility size from state metrics
                            state_avg_census = latest_data['Census'] if 'Census' in latest_data and pd.notna(latest_data['Census']) else 100
                            
                            # Use the state PBJ Takeaway card
                            state_pbj_takeaway_card(
                                state_name=state_full_name,
                                reported_hprd=state_hprd,
                                quarter_label=f"Q{quarter} {year}",
                                national_hprd=national_hprd,
                                state_rank=state_rank,
                                total_states=total_states,
                                trend_delta=trend_delta,
                                previous_quarter="Q1 2024",
                                avg_facility_size=state_avg_census
                            )
                    
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