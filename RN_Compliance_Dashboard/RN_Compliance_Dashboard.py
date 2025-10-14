import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import duckdb
import os
from typing import Optional, Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="RN Compliance Dashboard by 320 Consulting",
    page_icon="pbj_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for consistent styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .facility-link {
        color: #1f77b4;
        text-decoration: none;
    }
    .facility-link:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)



@st.cache_data
def load_compliance_data():
    """Load RN compliance data from CSV."""
    try:
        df = pd.read_csv('rn_compliance_analysis.csv', low_memory=False)
        # Ensure PROVNUM is string with leading zeros
        df['PROVNUM'] = df['PROVNUM'].astype(str).str.zfill(6)
        
        # Format quarter labels for display (Q1 2018 format)
        df['CY_Qtr_Display'] = df['CY_Qtr'].str.replace('CY', '').str.replace('Q', ' Q').str.replace(r'(\d+) Q(\d+)', r'Q\2 \1', regex=True)
        # Create tooltip format (Q4 2022 instead of 2022 Q4)
        df['CY_Qtr_Tooltip'] = df['CY_Qtr'].str.replace('CY', '').str.replace('Q', ' Q').str.replace(r'(\d+) Q(\d+)', r'Q\2 \1', regex=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading compliance data: {str(e)}")
        return pd.DataFrame()

def get_db_connection():
    """Create in-memory DuckDB connection for compliance data."""
    try:
        df = load_compliance_data()
        if df.empty:
            return None
        
        conn = duckdb.connect(':memory:')
        conn.register('compliance_data', df)
        
        return conn
    except Exception as e:
        st.error(f"Error creating database connection: {str(e)}")
        return None

def get_state_list() -> List[str]:
    """Get list of all states in the data."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        result = conn.execute("SELECT DISTINCT STATE FROM compliance_data ORDER BY STATE").fetchall()
        return [row[0] for row in result]
    except Exception as e:
        st.error(f"Error getting state list: {str(e)}")
        return []
    finally:
        conn.close()

def get_facility_list(state: Optional[str] = None) -> List[Dict]:
    """Get list of facilities, optionally filtered by state."""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        if state:
            query = """
                SELECT DISTINCT PROVNUM, PROVNAME, STATE, COUNTY_NAME, CITY
                FROM compliance_data 
                WHERE STATE = ? 
                ORDER BY PROVNAME
            """
            result = conn.execute(query, (state,)).fetchall()
            # Add debugging for large states
            if len(result) > 500:
                st.info(f"Loading {len(result)} facilities for {state}...")
        else:
            query = """
                SELECT DISTINCT PROVNUM, PROVNAME, STATE, COUNTY_NAME, CITY
                FROM compliance_data 
                ORDER BY PROVNAME
            """
            result = conn.execute(query).fetchall()
        
        return [
            {
                'PROVNUM': row[0],
                'PROVNAME': row[1],
                'STATE': row[2],
                'COUNTY_NAME': row[3],
                'CITY': row[4]
            }
            for row in result
        ]
    except Exception as e:
        st.error(f"Error getting facility list: {str(e)}")
        return []
    finally:
        conn.close()

def get_national_metrics() -> Dict:
    """Get national compliance metrics."""
    conn = get_db_connection()
    if not conn:
        return {}
    
    try:
        # Get most recent quarter
        latest_quarter = conn.execute("SELECT MAX(CY_Qtr) FROM compliance_data").fetchone()[0]
        latest_quarter_display = conn.execute("SELECT MAX(CY_Qtr_Display) FROM compliance_data WHERE CY_Qtr = ?", (latest_quarter,)).fetchone()[0]
        
        # Calculate national metrics for most recent quarter
        query = """
            SELECT 
                SUM(Days_Non_Compliant) as total_non_compliant_days,
                SUM(Total_Days_Reported) as total_days_reported,
                COUNT(DISTINCT PROVNUM) as facility_count
            FROM compliance_data 
            WHERE CY_Qtr = ?
        """
        result = conn.execute(query, (latest_quarter,)).fetchone()
        
        if result and result[1] > 0:
            compliance_rate = min(((result[1] - result[0]) / result[1]) * 100, 100.0)
            return {
                'latest_quarter': latest_quarter_display,
                'compliance_rate': compliance_rate,
                'total_days': result[1],
                'non_compliant_days': result[0],
                'facility_count': result[2]
            }
        
        return {}
    except Exception as e:
        st.error(f"Error getting national metrics: {str(e)}")
        return {}
    finally:
        conn.close()

def get_overall_compliance_rate() -> float:
    """Get overall compliance rate across all quarters."""
    conn = get_db_connection()
    if not conn:
        return 0.0
    
    try:
        query = """
            SELECT 
                SUM(Days_Non_Compliant) as total_non_compliant_days,
                SUM(Total_Days_Reported) as total_days_reported
            FROM compliance_data
        """
        result = conn.execute(query).fetchone()
        
        if result and result[1] > 0:
            return min(((result[1] - result[0]) / result[1]) * 100, 100.0)
        
        return 0.0
    except Exception as e:
        st.error(f"Error calculating overall compliance rate: {str(e)}")
        return 0.0
    finally:
        conn.close()

def get_quarterly_compliance_trend() -> pd.DataFrame:
    """Get quarterly compliance trend data."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                CY_Qtr,
                CY_Qtr_Display,
                CY_Qtr_Tooltip,
                SUM(Days_Non_Compliant) as total_non_compliant_days,
                SUM(Total_Days_Reported) as total_days_reported,
                COUNT(DISTINCT PROVNUM) as facility_count
            FROM compliance_data 
            GROUP BY CY_Qtr, CY_Qtr_Display, CY_Qtr_Tooltip
            ORDER BY CY_Qtr
        """
        result = conn.execute(query).fetchall()
        
        if result:
            df = pd.DataFrame(result, columns=['CY_Qtr', 'CY_Qtr_Display', 'CY_Qtr_Tooltip', 'Non_Compliant_Days', 'Total_Days', 'Facility_Count'])
            df['Compliance_Rate'] = ((df['Total_Days'] - df['Non_Compliant_Days']) / df['Total_Days']) * 100
            df['Compliance_Rate'] = df['Compliance_Rate'].clip(upper=100.0)
            df['Non_Compliance_Rate'] = 100 - df['Compliance_Rate']
            return df
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting quarterly trend: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_state_metrics() -> pd.DataFrame:
    """Get state-level compliance metrics for most recent quarter."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        latest_quarter = conn.execute("SELECT MAX(CY_Qtr) FROM compliance_data").fetchone()[0]
        
        query = """
            SELECT 
                STATE,
                SUM(Days_Non_Compliant) as total_non_compliant_days,
                SUM(Total_Days_Reported) as total_days_reported,
                COUNT(DISTINCT PROVNUM) as facility_count
            FROM compliance_data 
            WHERE CY_Qtr = ?
            GROUP BY STATE 
            ORDER BY STATE
        """
        result = conn.execute(query, (latest_quarter,)).fetchall()
        
        if result:
            df = pd.DataFrame(result, columns=['STATE', 'Non_Compliant_Days', 'Total_Days', 'Facility_Count'])
            df['Compliance_Rate'] = ((df['Total_Days'] - df['Non_Compliant_Days']) / df['Total_Days']) * 100
            df['Compliance_Rate'] = df['Compliance_Rate'].clip(upper=100.0)
            return df
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting state metrics: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_state_quarterly_trend(state: str) -> pd.DataFrame:
    """Get quarterly compliance trend for a specific state."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                CY_Qtr,
                CY_Qtr_Display,
                CY_Qtr_Tooltip,
                SUM(Days_Non_Compliant) as total_non_compliant_days,
                SUM(Total_Days_Reported) as total_days_reported,
                COUNT(DISTINCT PROVNUM) as facility_count
            FROM compliance_data 
            WHERE STATE = ?
            GROUP BY CY_Qtr, CY_Qtr_Display, CY_Qtr_Tooltip
            ORDER BY CY_Qtr
        """
        result = conn.execute(query, (state,)).fetchall()
        
        if result:
            df = pd.DataFrame(result, columns=['CY_Qtr', 'CY_Qtr_Display', 'CY_Qtr_Tooltip', 'Non_Compliant_Days', 'Total_Days', 'Facility_Count'])
            df['Compliance_Rate'] = ((df['Total_Days'] - df['Non_Compliant_Days']) / df['Total_Days']) * 100
            df['Compliance_Rate'] = df['Compliance_Rate'].clip(upper=100.0)
            df['Non_Compliance_Rate'] = 100 - df['Compliance_Rate']
            return df
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting state quarterly trend: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_state_facilities_most_non_compliant(state: str, limit: int = 20) -> pd.DataFrame:
    """Get facilities in a state with the most non-compliant days in the latest quarter."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        latest_quarter = conn.execute("SELECT MAX(CY_Qtr) FROM compliance_data").fetchone()[0]
        
        query = """
            SELECT 
                PROVNUM,
                PROVNAME,
                COUNTY_NAME,
                CITY,
                Days_Non_Compliant,
                Total_Days_Reported,
                Avg_Daily_Census,
                Avg_Daily_RN_Hours
            FROM compliance_data 
            WHERE STATE = ? AND CY_Qtr = ?
            ORDER BY Days_Non_Compliant DESC
            LIMIT ?
        """
        result = conn.execute(query, (state, latest_quarter, limit)).fetchall()
        
        if result:
            df = pd.DataFrame(result, columns=[
                'PROVNUM', 'PROVNAME', 'COUNTY_NAME', 'CITY', 
                'Days_Non_Compliant', 'Total_Days_Reported', 'Avg_Daily_Census', 'Avg_Daily_RN_Hours'
            ])
            df['Compliance_Rate'] = ((df['Total_Days_Reported'] - df['Days_Non_Compliant']) / df['Total_Days_Reported']) * 100
            df['Compliance_Rate'] = df['Compliance_Rate'].clip(upper=100.0)
            return df
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting state facilities: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_facility_metrics(provnum: str) -> pd.DataFrame:
    """Get facility-level compliance metrics."""
    conn = get_db_connection()
    if not conn:
        return pd.DataFrame()
    
    try:
        query = """
            SELECT 
                CY_Qtr,
                CY_Qtr_Display,
                CY_Qtr_Tooltip,
                PROVNAME,
                STATE,
                COUNTY_NAME,
                CITY,
                Avg_Daily_Census,
                Avg_Daily_RN_Hours,
                Days_Non_Compliant,
                Total_Days_Reported
            FROM compliance_data 
            WHERE PROVNUM = ?
            ORDER BY CY_Qtr
        """
        result = conn.execute(query, (provnum,)).fetchall()
        
        if result:
            df = pd.DataFrame(result, columns=[
                'CY_Qtr', 'CY_Qtr_Display', 'CY_Qtr_Tooltip', 'PROVNAME', 'STATE', 'COUNTY_NAME', 'CITY',
                'Avg_Daily_Census', 'Avg_Daily_RN_Hours', 'Days_Non_Compliant', 'Total_Days_Reported'
            ])
            df['Compliance_Rate'] = ((df['Total_Days_Reported'] - df['Days_Non_Compliant']) / df['Total_Days_Reported']) * 100
            df['Compliance_Rate'] = df['Compliance_Rate'].clip(upper=100.0)
            return df
        
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting facility metrics: {str(e)}")
        return pd.DataFrame()

def create_care_compare_link(provnum: str) -> str:
    """Create Care Compare link for facility."""
    return f"https://www.medicare.gov/care-compare/details/nursing-home/{provnum}"

def display_national_overview():
    """Display national overview section with header."""
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
        <h1 style="font-size: 2.5rem; font-weight: 800; color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); letter-spacing: -1px;">
            PBJ Nursing Home RN Compliance Dashboard
        </h1>
        <p style="font-size: 1.1rem; color: rgba(255,255,255,0.9); margin: 0.3rem 0; font-weight: 300; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
            Analysis of RN 8-Hour Minimum
        </p>
        <p style="font-size: 0.9rem; color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-style: italic; font-weight: 300;">
            A free public resource by 320 Consulting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    display_national_overview_no_header()

def display_national_overview_no_header():
    """Display national overview section without header."""
    # Quarterly trend chart (moved after facility search)
    st.markdown("### National Quarterly Non-Compliance Rate")
    quarterly_data = get_quarterly_compliance_trend()
    
    if not quarterly_data.empty:
        # Create custom hover template with proper quarter format and rounded compliance rate
        hover_template = '<b>%{customdata}</b><br>' + \
                        'Non-Compliance Rate: %{y:.1f}%<br>' + \
                        'Compliance Rate: %{text:.1f}%<br>' + \
                        '<extra></extra>'
        
        fig = px.line(
            quarterly_data, 
            x='CY_Qtr_Display', 
            y='Non_Compliance_Rate',
            title='RN Non-Compliance Rate by Quarter',
            labels={'Non_Compliance_Rate': 'Non-Compliance Rate (%)', 'CY_Qtr_Display': 'Quarter'},
            markers=True
        )
        # Update hover template after creating the figure
        fig.update_traces(
            hovertemplate='<b>%{x}</b><br>' + \
                        'Non-Compliance Rate: %{y:.1f}%<br>' + \
                        '<extra></extra>'
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False,
            yaxis=dict(range=[0, min(10, max(quarterly_data['Non_Compliance_Rate']) * 1.2)]),  # Dynamic range based on max observed
            annotations=[
                dict(
                    text="<b>320 Consulting</b> | Source: CMS PBJ Data (2017-2025)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.99, y=-0.25,
                    xanchor='right', yanchor='top',
                    font=dict(size=10, color='gray')
                )
            ]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Also show non-compliant days trend
        fig2 = px.bar(
            quarterly_data,
            x='CY_Qtr_Display',
            y='Non_Compliant_Days',
            title='Total Non-Compliant Days by Quarter',
            labels={'Non_Compliant_Days': 'Non-Compliant Days', 'CY_Qtr_Display': 'Quarter'}
        )
        fig2.update_layout(
            xaxis_tickangle=-45,
            height=400,
            showlegend=False,
            annotations=[
                dict(
                    text="<b>320 Consulting</b> | Source: CMS PBJ Data (2017-2025)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.99, y=-0.25,
                    xanchor='right', yanchor='top',
                    font=dict(size=10, color='gray')
                )
            ]
        )
        st.plotly_chart(fig2, use_container_width=True)

def display_state_analysis():
    """Display state-level analysis."""
    st.markdown("## State-Level Analysis")
    
    # Get state metrics
    state_data = get_state_metrics()
    
    if not state_data.empty:
        # State compliance rate map/chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = px.bar(
                state_data.sort_values('Compliance_Rate'),
                x='STATE',
                y='Compliance_Rate',
                title='RN Compliance Rate by State (Latest Quarter)',
                labels={'Compliance_Rate': 'Compliance Rate (%)', 'STATE': 'State'}
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                yaxis=dict(range=[90, 100]),  # Set y-axis range for 90-100%
                annotations=[
                    dict(
                        text="<b>320 Consulting</b> | Source: CMS PBJ Data (2017-2025)",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.99, y=-0.25,
                        xanchor='right', yanchor='top',
                        font=dict(size=10, color='gray')
                    )
                ]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top and bottom performing states
            st.markdown("### Top Performing States")
            top_states = state_data.nlargest(5, 'Compliance_Rate')[['STATE', 'Compliance_Rate']]
            for _, row in top_states.iterrows():
                if st.button(f"{row['STATE']} ({row['Compliance_Rate']:.1f}%)", key=f"top_{row['STATE']}"):
                    st.session_state.selected_state = row['STATE']
                    st.rerun()
            
            st.markdown("### Bottom Performing States")
            bottom_states = state_data.nsmallest(5, 'Compliance_Rate')[['STATE', 'Compliance_Rate']]
            for _, row in bottom_states.iterrows():
                if st.button(f"{row['STATE']} ({row['Compliance_Rate']:.1f}%)", key=f"bottom_{row['STATE']}"):
                    st.session_state.selected_state = row['STATE']
                    st.rerun()
        
        # State selection for detailed view
        st.markdown("---")
        st.markdown("### Select State for Detailed Analysis")
        selected_state = st.selectbox("Choose a State", [""] + list(state_data['STATE'].unique()), index=0)
        
        if selected_state and selected_state != "":
            st.session_state.selected_state = selected_state
            st.rerun()

def display_facility_search():
    """Display facility search functionality."""
    states = get_state_list()
    
    # Initialize search-specific session state
    if 'search_state' not in st.session_state:
        st.session_state.search_state = None
    
    # Use a unique key for the state selection
    selected_state = st.selectbox(
        "Step 1: Select a state:", 
        ["Select a state..."] + states, 
        index=states.index(st.session_state.search_state) + 1 if st.session_state.search_state in states else 0,
        key="search_state_select"
    )
    
    # Update search state when state changes
    if selected_state != "Select a state..." and selected_state != st.session_state.search_state:
        st.session_state.search_state = selected_state
    
    # Only show Step 2 if a state is actually selected
    if selected_state and selected_state != "Select a state...":
        # Get facilities for selected state
        facilities = get_facility_list(selected_state)
        
        if facilities:
            # Step 2: Searchable dropdown for facilities
            def format_facility_name(name):
                """Apply proper capitalization rules to facility names."""
                # Convert to title case first
                name = name.title()
                # Fix common capitalization issues
                name = name.replace(" Of ", " of ")
                name = name.replace(" At ", " at ")
                name = name.replace(" Inc.", " Inc.")
                name = name.replace(" Llc", " LLC")
                name = name.replace(" Lp", " LP")
                name = name.replace(" L.L.C.", " L.L.C.")
                name = name.replace(" L.P.", " L.P.")
                # Fix ordinal numbers
                import re
                name = re.sub(r'(\d+)Th', r'\1th', name)
                name = re.sub(r'(\d+)Nd', r'\1nd', name)
                name = re.sub(r'(\d+)Rd', r'\1rd', name)
                name = re.sub(r'(\d+)St', r'\1st', name)
                return name
            
            # Create facility options with simple formatting for performance
            facility_options = {f"{f['PROVNAME'].title()} ({f['PROVNUM']}) - {f['CITY'].title()}": f['PROVNUM'] for f in facilities}
            
            # Add search option at the top
            search_options = ["Type to search..."] + list(facility_options.keys())
            selected_facility = st.selectbox(
                "Step 2: Select facility or type to search:",
                search_options,
                index=0,
                key="search_facility_select"
            )
            
            if selected_facility and selected_facility != "Type to search...":
                provnum = facility_options[selected_facility]
                st.session_state.selected_facility = provnum
                st.rerun()
        else:
            st.warning(f"No facilities found for {selected_state}.")
    


def display_facility_details(provnum: str):
    """Display detailed facility compliance information."""
    facility_data = get_facility_metrics(provnum)
    
    if facility_data.empty:
        st.error("No data found for this facility.")
        return
    
    # Add back button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Back to Search"):
            st.session_state.selected_facility = None
            st.rerun()
    with col2:
        st.markdown("")
    
    # Facility header
    facility_info = facility_data.iloc[0]
    
    def format_facility_name(name):
        """Apply proper capitalization rules to facility names."""
        # Convert to title case first
        name = name.title()
        # Fix common capitalization issues
        name = name.replace(" Of ", " of ")
        name = name.replace(" At ", " at ")
        name = name.replace(" Inc.", " Inc.")
        name = name.replace(" Llc", " LLC")
        name = name.replace(" Lp", " LP")
        name = name.replace(" L.L.C.", " L.L.C.")
        name = name.replace(" L.P.", " L.P.")
        # Fix ordinal numbers
        import re
        name = re.sub(r'(\d+)Th', r'\1th', name)
        name = re.sub(r'(\d+)Nd', r'\1nd', name)
        name = re.sub(r'(\d+)Rd', r'\1rd', name)
        name = re.sub(r'(\d+)St', r'\1st', name)
        return name
    
    facility_name = format_facility_name(facility_info['PROVNAME'])
    st.markdown(f"## {facility_name}")
    # Convert state abbreviation to full name for display
    state_names = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    state_full_name = state_names.get(facility_info['STATE'], facility_info['STATE'])
    
    # Display facility info
    st.markdown(f"**Provider Number:** {provnum} | **County:** {facility_info['COUNTY_NAME']} | **City:** {facility_info['CITY'].title()}")
    
    # Add state navigation button below the facility info
    if st.button(f"View {state_names.get(facility_info['STATE'], facility_info['STATE'])} RN Compliance", key=f"state_nav_{facility_info['STATE']}"):
        st.session_state.selected_state = facility_info['STATE']
        st.rerun()
    

    
    # Care Compare link will be moved to bottom as a button
    
    # Latest quarter metrics with custom styling
    latest_data = facility_data.iloc[-1]
    
    # Custom CSS for metric boxes
    st.markdown("""
    <style>
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .metric-title {
        font-size: 14px;
        font-weight: 600;
        color: #495057;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        non_compliance_rate = min(((latest_data['Days_Non_Compliant']) / latest_data['Total_Days_Reported']) * 100, 100.0)
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Latest Quarter Non-Compliance Rate</div>
            <div class="metric-value">{non_compliance_rate:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Non-Compliant Days (Latest Quarter)</div>
            <div class="metric-value">{latest_data['Days_Non_Compliant']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Average Daily Census</div>
            <div class="metric-value">{latest_data['Avg_Daily_Census']:.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Average Daily RN Hours</div>
            <div class="metric-value">{latest_data['Avg_Daily_RN_Hours']:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quarterly trend for this facility
    st.markdown("### Quarterly Non-Compliance Rate")
    facility_name = format_facility_name(facility_info["PROVNAME"])
    
    # Cap compliance rates at 100% and calculate non-compliance rate
    facility_data['Compliance_Rate_Capped'] = facility_data['Compliance_Rate'].clip(upper=100.0)
    facility_data['Non_Compliance_Rate'] = 100 - facility_data['Compliance_Rate_Capped']
    
    # Create custom hover template with proper quarter format and rounded compliance rate
    hover_template = '<b>%{customdata}</b><br>' + \
                    'Non-Compliance Rate: %{y:.1f}%<br>' + \
                    'Compliance Rate: %{text:.1f}%<br>' + \
                    '<extra></extra>'
    
    fig = px.line(
        facility_data,
        x='CY_Qtr_Display',
        y='Non_Compliance_Rate',
        title=f'RN Non-Compliance Rate by Quarter - {facility_name}',
        labels={'Non_Compliance_Rate': 'Non-Compliance Rate (%)', 'CY_Qtr_Display': 'Quarter'},
        markers=True
    )
    # Update hover template after creating the figure
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>' + \
                    'Non-Compliance Rate: %{y:.1f}%<br>' + \
                    '<extra></extra>'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, max(facility_data['Non_Compliance_Rate']) * 1.1 + 2]),  # Dynamic range with padding
        annotations=[
            dict(
                text="<b>320 Consulting</b> | Source: CMS PBJ Data (2017-2025)",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.99, y=-0.25,
                xanchor='right', yanchor='top',
                font=dict(size=10, color='gray')
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Non-compliant days trend
    fig2 = px.bar(
        facility_data,
        x='CY_Qtr_Display',
        y='Days_Non_Compliant',
        title=f'Non-Compliant Days by Quarter - {facility_name}',
        labels={'Days_Non_Compliant': 'Non-Compliant Days', 'CY_Qtr_Display': 'Quarter'}
    )
    fig2.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False,
        annotations=[
            dict(
                text="<b>320 Consulting</b> | Source: CMS PBJ Data (2017-2025)",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.99, y=-0.25,
                xanchor='right', yanchor='top',
                font=dict(size=10, color='gray')
            )
        ]
    )
    st.plotly_chart(fig2, use_container_width=True)
    
    # Detailed quarterly data table
    st.markdown("### Detailed Quarterly Data")
    display_data = facility_data[['CY_Qtr_Display', 'Compliance_Rate', 'Days_Non_Compliant', 'Total_Days_Reported', 'Avg_Daily_Census', 'Avg_Daily_RN_Hours']].copy()
    # Calculate non-compliance rate for display
    display_data['Non_Compliance_Rate'] = 100 - display_data['Compliance_Rate'].clip(upper=100.0)
    display_data['Non_Compliance_Rate'] = display_data['Non_Compliance_Rate'].round(1)
    display_data['Avg_Daily_Census'] = display_data['Avg_Daily_Census'].round(0)
    display_data['Avg_Daily_RN_Hours'] = display_data['Avg_Daily_RN_Hours'].round(1)
    
    # Sort by quarter to show most recent first (reverse the order)
    display_data = display_data.iloc[::-1].reset_index(drop=True)
    
    # Create the final display dataframe with correct columns
    final_display_data = pd.DataFrame({
        'Quarter': display_data['CY_Qtr_Display'],
        'Non-Compliance Rate (%)': display_data['Non_Compliance_Rate'],
        'Non-Compliant Days': display_data['Days_Non_Compliant'],
        'Total Days Reported': display_data['Total_Days_Reported'],
        'Avg Daily Census': display_data['Avg_Daily_Census'],
        'Avg Daily RN Hours': display_data['Avg_Daily_RN_Hours']
    })
    
    st.dataframe(final_display_data, use_container_width=True, hide_index=True)
    
    # Add methodology and PBJ Takeaway expanders
    st.markdown("---")
    
    # Methodology expander
    with st.expander("⚙️ Methodology", expanded=False):
        st.markdown("""
        This dashboard uses CMS Payroll-Based Journal (PBJ) data (2017–2025) to analyze RN compliance with the 8-hour minimum requirement.
        
        **Compliance Calculation**
        
        **RN Hours:** Total daily RN hours (RN + RN Admin + RN DON) must be ≥ 8 hours per day.
        
        **Compliance Rate:** Percentage of days where RN hours ≥ 8, excluding days with zero census.
        
        **Non-Compliance Rate:** Percentage of days where RN hours < 8, excluding days with zero census.
        
        **Data Sources:** CMS PBJ Daily Staffing Files (2017-2025), Provider Information, Affiliated Entity data.
        
        **Note:** Days with zero census are excluded from compliance calculations. The 8-hour minimum is a proposed federal requirement that was recently overturned (2025). Some states have their own RN staffing requirements.
        
        **Data Transparency**
        <div style="font-size: 0.9em; color: #666;">
        The RN Compliance Dashboard pulls directly from CMS data and is carefully vetted for accuracy. If you spot something that looks off, please let me know <a href="mailto:eric@320insight.com">eric@320insight.com</a> so I can set things right.
        </div>
        """, unsafe_allow_html=True)
    
    # PBJ Takeaway expander
    with st.expander("RN Compliance Takeaway", expanded=False):
        st.image("pbj_favicon.png", width=50)
        # Calculate compliance rate from latest data
        compliance_rate = ((latest_data['Total_Days_Reported'] - latest_data['Days_Non_Compliant']) / latest_data['Total_Days_Reported']) * 100
        st.markdown(f"""
        **RN Compliance Analysis: {facility_name}**
        
        **Latest Quarter Performance:** {compliance_rate:.1f}% compliance rate with {latest_data['Days_Non_Compliant']} non-compliant days out of {latest_data['Total_Days_Reported']} total days.
        
        **Facility Context:** Average daily census of {latest_data['Avg_Daily_Census']:.0f} residents with {latest_data['Avg_Daily_RN_Hours']:.1f} average RN hours per day.
        
        **Compliance Trend:** Review the quarterly trend chart above to see how this facility's RN compliance has changed over time.
        
                 **Key Insight:** Facilities with consistent RN compliance typically have better resident outcomes and regulatory standing. Non-compliance may indicate staffing challenges or operational issues that require attention.
         """, unsafe_allow_html=True)
    
    # Care Compare button at the bottom
    st.markdown("---")
    care_compare_url = create_care_compare_link(provnum)
    st.markdown(f"""
    <div style="text-align: center; margin: 2rem 0;">
        <a href="{care_compare_url}" target="_blank" style="
            display: inline-block;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        " onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 6px 16px rgba(0,0,0,0.2)'" 
           onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 4px 12px rgba(0,0,0,0.15)'">
            🔍 View on Care Compare
        </a>
    </div>
    """, unsafe_allow_html=True)

def display_state_details(state: str):
    """Display detailed state-level compliance information."""
    # Convert state abbreviations to full names
    state_names = {
        'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
        'CO': 'Colorado', 'CT': 'Connecticut', 'DC': 'District of Columbia', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
        'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
        'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
        'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri',
        'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey',
        'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
        'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'PR': 'Puerto Rico', 'RI': 'Rhode Island', 'SC': 'South Carolina',
        'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
        'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
    }
    
    # Add state search dropdown at the top
    st.markdown("### Search Different State")
    all_states = get_state_list()
    selected_new_state = st.selectbox(
        "Select a state to view:",
        all_states,
        index=all_states.index(state) if state in all_states else 0,
        key="state_switch_dropdown"
    )
    
    # If user selects a different state, switch to it
    if selected_new_state != state:
        st.session_state.selected_state = selected_new_state
        st.rerun()
    
    state_full_name = state_names.get(state, state)
    st.markdown(f"## {state_full_name} RN Compliance Analysis")
    
    # Get state data
    state_quarterly_data = get_state_quarterly_trend(state)
    state_facilities = get_state_facilities_most_non_compliant(state)
    
    if state_quarterly_data.empty:
        st.error(f"No data found for {state}.")
        return
    
    # State metrics for latest quarter
    latest_data = state_quarterly_data.iloc[-1]
    compliance_rate = latest_data['Compliance_Rate']
    non_compliant_days = latest_data['Non_Compliant_Days']
    total_days = latest_data['Total_Days']
    facility_count = latest_data['Facility_Count']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Latest Quarter Compliance Rate", f"{compliance_rate:.1f}%")
    
    with col2:
        st.metric("Non-Compliant Days (Latest Quarter)", f"{non_compliant_days:,}")
    
    with col3:
        st.metric("Total Days Reported", f"{total_days:,}")
    
    with col4:
        st.metric("Facilities in State", f"{facility_count}")
    
    # California-specific metrics
    if state == 'CA':
        st.markdown("---")
        st.markdown("### California-Specific Staffing Metrics")
        
        # Get California-specific data
        df = load_compliance_data()
        ca_data = df[df['STATE'] == 'CA']
        latest_qtr = ca_data['CY_Qtr'].max()
        latest_ca_data = ca_data[ca_data['CY_Qtr'] == latest_qtr]
        
        if not latest_ca_data.empty:
            # Calculate California-specific metrics
            total_facilities_ca = latest_ca_data['PROVNUM'].nunique()
            
            # Count facilities that failed to meet requirements
            facilities_below_total_nurse = len(latest_ca_data[latest_ca_data['Days_Below_Total_Nurse_Threshold'] > 0])
            facilities_below_cna = len(latest_ca_data[latest_ca_data['Days_Below_CNA_Threshold'] > 0])
            
            # Calculate compliance rates (facilities that met requirements)
            total_nurse_compliance_rate = ((total_facilities_ca - facilities_below_total_nurse) / total_facilities_ca * 100) if total_facilities_ca > 0 else 0
            cna_compliance_rate = ((total_facilities_ca - facilities_below_cna) / total_facilities_ca * 100) if total_facilities_ca > 0 else 0
            
            # Average HPRD metrics
            avg_total_nurse_hprd = latest_ca_data['Avg_Total_Nurse_HPRD'].mean()
            avg_cna_hprd = latest_ca_data['Avg_CNA_HPRD'].mean()
            
            # Total days below thresholds
            days_below_total_nurse = latest_ca_data['Days_Below_Total_Nurse_Threshold'].sum()
            days_below_cna = latest_ca_data['Days_Below_CNA_Threshold'].sum()
            
            # Display California metrics in columns
            ca_col1, ca_col2, ca_col3, ca_col4 = st.columns(4)
            
            with ca_col1:
                st.metric("Total Nurse Compliance", f"{total_nurse_compliance_rate:.1f}%", 
                         delta=f"{facilities_below_total_nurse} facilities failed")
            
            with ca_col2:
                st.metric("CNA Compliance", f"{cna_compliance_rate:.1f}%",
                         delta=f"{facilities_below_cna} facilities failed")
            
            with ca_col3:
                st.metric("Avg Total Nurse HPRD", f"{avg_total_nurse_hprd:.2f}")
            
            with ca_col4:
                st.metric("Avg CNA HPRD", f"{avg_cna_hprd:.2f}")
            
            # California-specific chart
            st.markdown("#### California Staffing Compliance Breakdown")
            
            # Create a bar chart showing different compliance metrics
            compliance_metrics = {
                'Federal RN 8-Hour': compliance_rate,
                'CA Total Nurse (3.5 HPRD)': total_nurse_compliance_rate,
                'CA CNA (2.4 HPRD)': cna_compliance_rate
            }
            
            fig_ca = px.bar(
                x=list(compliance_metrics.keys()),
                y=list(compliance_metrics.values()),
                title="California Compliance Rates by Standard",
                labels={'x': 'Compliance Standard', 'y': 'Compliance Rate (%)'},
                color=list(compliance_metrics.values()),
                color_continuous_scale='RdYlGn'
            )
            fig_ca.update_layout(
                height=400,
                showlegend=False,
                yaxis=dict(range=[0, 100]),
                annotations=[
                    dict(
                        text="<b>320 Consulting</b> | Source: CMS PBJ Data (2017-2025)",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.99, y=-0.25,
                        xanchor='right', yanchor='top',
                        font=dict(size=10, color='gray')
                    )
                ]
            )
            st.plotly_chart(fig_ca, use_container_width=True)
            
            # California compliance summary
            st.markdown("#### California Staffing Compliance Summary")
            ca_summary_col1, ca_summary_col2 = st.columns(2)
            
            with ca_summary_col1:
                st.markdown(f"""
                **Total Nurse Staffing (3.5 HPRD Target):**
                - **{total_facilities_ca - facilities_below_total_nurse}** facilities compliant
                - **{facilities_below_total_nurse}** facilities failed
                - **{days_below_total_nurse:,}** total days below threshold
                """)
            
            with ca_summary_col2:
                st.markdown(f"""
                **CNA Staffing (2.4 HPRD Target):**
                - **{total_facilities_ca - facilities_below_cna}** facilities compliant  
                - **{facilities_below_cna}** facilities failed
                - **{days_below_cna:,}** total days below threshold
                """)
            
            # California-specific facility table
            st.markdown("#### California Facilities - Detailed Staffing Metrics")
            
            # Get California facilities with detailed metrics
            ca_facilities_detailed = latest_ca_data[['PROVNUM', 'PROVNAME', 'COUNTY_NAME', 'CITY', 
                                                   'Days_CA_Compliant', 'Total_Days_Reported',
                                                   'Avg_Total_Nurse_HPRD', 'Avg_CNA_HPRD',
                                                   'Days_Below_Total_Nurse_Threshold', 'Days_Below_CNA_Threshold']].copy()
            
            # Calculate compliance rates
            ca_facilities_detailed['CA_Compliance_Rate'] = (ca_facilities_detailed['Days_CA_Compliant'] / 
                                                          ca_facilities_detailed['Total_Days_Reported'] * 100)
            ca_facilities_detailed['CNA_Compliance_Rate'] = ((ca_facilities_detailed['Total_Days_Reported'] - 
                                                            ca_facilities_detailed['Days_Below_CNA_Threshold']) / 
                                                           ca_facilities_detailed['Total_Days_Reported'] * 100)
            
            # Sort by CA compliance rate (worst first)
            ca_facilities_detailed = ca_facilities_detailed.sort_values('CA_Compliance_Rate').head(10)
            
            # Format for display
            ca_facilities_detailed['PROVNAME'] = ca_facilities_detailed['PROVNAME'].apply(lambda x: x.title())
            ca_facilities_detailed['CA_Compliance_Rate'] = ca_facilities_detailed['CA_Compliance_Rate'].round(1)
            ca_facilities_detailed['CNA_Compliance_Rate'] = ca_facilities_detailed['CNA_Compliance_Rate'].round(1)
            ca_facilities_detailed['Avg_Total_Nurse_HPRD'] = ca_facilities_detailed['Avg_Total_Nurse_HPRD'].round(2)
            ca_facilities_detailed['Avg_CNA_HPRD'] = ca_facilities_detailed['Avg_CNA_HPRD'].round(2)
            
            # Create display table
            ca_display_data = []
            for _, row in ca_facilities_detailed.iterrows():
                ca_display_data.append({
                    'Facility': row['PROVNAME'],
                    'Location': f"{row['COUNTY_NAME'].title()}, {row['CITY'].title()}",
                    'CA Compliance Rate': f"{row['CA_Compliance_Rate']:.1f}%",
                    'CNA Compliance Rate': f"{row['CNA_Compliance_Rate']:.1f}%",
                    'Total Nurse HPRD': f"{row['Avg_Total_Nurse_HPRD']:.2f}",
                    'CNA HPRD': f"{row['Avg_CNA_HPRD']:.2f}",
                    'Days Below CNA': row['Days_Below_CNA_Threshold'],
                    'PROVNUM': row['PROVNUM']
                })
            
            ca_table_df = pd.DataFrame(ca_display_data)
            
            # Display California-specific table
            st.dataframe(
                ca_table_df[['Facility', 'Location', 'CA Compliance Rate', 'CNA Compliance Rate', 
                            'Total Nurse HPRD', 'CNA HPRD', 'Days Below CNA']],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Facility": st.column_config.TextColumn("Facility Name", width="medium"),
                    "Location": st.column_config.TextColumn("Location", width="medium"),
                    "CA Compliance Rate": st.column_config.NumberColumn("CA Compliance Rate", width="small"),
                    "CNA Compliance Rate": st.column_config.NumberColumn("CNA Compliance Rate", width="small"),
                    "Total Nurse HPRD": st.column_config.NumberColumn("Total Nurse HPRD", width="small"),
                    "CNA HPRD": st.column_config.NumberColumn("CNA HPRD", width="small"),
                    "Days Below CNA": st.column_config.NumberColumn("Days Below CNA", width="small")
                }
            )
            
            # Add California-specific explanation
            st.markdown("""
            **California-Specific Metrics Explained:**
            
            - **Total Nurse Compliance:** Percentage of facilities meeting California's 3.5 total nurse HPRD requirement
            - **CNA Compliance:** Percentage of facilities meeting California's 2.4 CNA HPRD requirement
            - **Total Nurse HPRD:** Average total nurse hours per resident day (target: 3.5)
            - **CNA HPRD:** Average CNA hours per resident day (target: 2.4)
            - **Days Below CNA:** Total days across all facilities below California's CNA staffing threshold
            """)
    
    # Quarterly trend for this state
    st.markdown("### Quarterly Non-Compliance Rate Trend")
    
    fig = px.line(
        state_quarterly_data,
        x='CY_Qtr_Display',
        y='Non_Compliance_Rate',
        title=f'RN Non-Compliance Rate by Quarter - {state}',
        labels={'Non_Compliance_Rate': 'Non-Compliance Rate (%)', 'CY_Qtr_Display': 'Quarter'},
        markers=True
    )
    fig.update_traces(
        hovertemplate='<b>%{x}</b><br>' + \
                    'Non-Compliance Rate: %{y:.1f}%<br>' + \
                    '<extra></extra>'
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        height=400,
        showlegend=False,
        yaxis=dict(range=[0, min(10, max(state_quarterly_data['Non_Compliance_Rate']) * 1.2)]),  # Dynamic range based on max observed
        annotations=[
            dict(
                text="<b>320 Consulting</b> | Source: CMS PBJ Data (2017-2025)",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.99, y=-0.25,
                xanchor='right', yanchor='top',
                font=dict(size=10, color='gray')
            )
        ]
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Facilities most out of compliance
    st.markdown("### Facilities Most Out of Compliance (Latest Quarter)")
    
    if not state_facilities.empty:
        # Determine limit based on facility count
        limit = 10 if len(state_facilities) <= 50 else 20
        display_facilities = state_facilities.head(limit)
        
        # Format data for display - include PROVNUM for hyperlinking
        display_data = display_facilities[['PROVNUM', 'PROVNAME', 'COUNTY_NAME', 'CITY', 'Compliance_Rate', 'Days_Non_Compliant', 'Total_Days_Reported', 'Avg_Daily_Census', 'Avg_Daily_RN_Hours']].copy()
        
        def format_facility_name(name):
            """Apply proper capitalization rules to facility names."""
            # Convert to title case first
            name = name.title()
            # Fix common capitalization issues
            name = name.replace(" Of ", " of ")
            name = name.replace(" At ", " at ")
            name = name.replace(" Inc.", " Inc.")
            name = name.replace(" Llc", " LLC")
            name = name.replace(" Lp", " LP")
            name = name.replace(" L.L.C.", " L.L.C.")
            name = name.replace(" L.P.", " L.P.")
            # Fix ordinal numbers
            import re
            name = re.sub(r'(\d+)Th', r'\1th', name)
            name = re.sub(r'(\d+)Nd', r'\1nd', name)
            name = re.sub(r'(\d+)Rd', r'\1rd', name)
            name = re.sub(r'(\d+)St', r'\1st', name)
            return name
        
        display_data['PROVNAME'] = display_data['PROVNAME'].apply(format_facility_name)
        display_data['Compliance_Rate'] = display_data['Compliance_Rate'].round(1)
        display_data['Avg_Daily_Census'] = display_data['Avg_Daily_Census'].round(0)
        display_data['Avg_Daily_RN_Hours'] = display_data['Avg_Daily_RN_Hours'].round(1)
        
        # Create simple table data with clickable facility names
        table_data = []
        for _, row in display_data.iterrows():
            # Calculate non-compliance rate
            non_compliance_rate = 100 - row['Compliance_Rate']
            
            # Format location
            location = f"{row['COUNTY_NAME'].title()}, {row['CITY'].title()}"
            
            # Store facility name and PROVNUM for later use
            table_data.append({
                'Facility': row['PROVNAME'],
                'Location': location,
                'Non-Compliance Rate': f"{non_compliance_rate:.1f}%",
                'Days Non-Compliant': row['Days_Non_Compliant'],
                'Avg Census': f"{row['Avg_Daily_Census']:.0f}",
                'RN Hours per Day': f"{row['Avg_Daily_RN_Hours']:.1f}",
                'PROVNUM': row['PROVNUM']  # Store for navigation
            })
        
        # Convert to DataFrame for display
        table_df = pd.DataFrame(table_data)
        
        # Display the simple table (excluding PROVNUM column)
        st.dataframe(
            table_df[['Facility', 'Location', 'Non-Compliance Rate', 'Days Non-Compliant', 'Avg Census', 'RN Hours per Day']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Facility": st.column_config.TextColumn("Facility Name", width="medium"),
                "Location": st.column_config.TextColumn("Location", width="medium"),
                "Non-Compliance Rate": st.column_config.NumberColumn("Non-Compliance Rate", width="small"),
                "Days Non-Compliant": st.column_config.NumberColumn("Days Non-Compliant", width="small"),
                "Avg Census": st.column_config.NumberColumn("Avg Census", width="small"),
                "RN Hours per Day": st.column_config.NumberColumn("RN Hours per Day", width="small")
            }
        )
        
        # Add clickable facility navigation using buttons
        st.markdown("### Quick Facility Access")
        cols = st.columns(3)
        for idx, row in enumerate(table_data):
            col_idx = idx % 3
            with cols[col_idx]:
                if st.button(f"View {row['Facility'][:30]}{'...' if len(row['Facility']) > 30 else ''}", 
                           key=f"state_facility_btn_{idx}"):
                    st.session_state.selected_facility = row['PROVNUM']
                    st.rerun()
        
        # Add clickable facility navigation
        st.markdown("**Click on a facility row above to view detailed analysis**")
    else:
        st.info(f"No facility data available for {state} in the latest quarter.")
    
    # Add methodology and PBJ Takeaway expanders
    st.markdown("---")
    
    # Methodology expander
    with st.expander("⚙️ Methodology", expanded=False):
        st.markdown("""
        This dashboard uses CMS Payroll-Based Journal (PBJ) data (2017–2025) to analyze RN compliance with the 8-hour minimum requirement.
        
        **Compliance Calculation**
        
        **RN Hours:** Total daily RN hours (RN + RN Admin + RN DON) must be ≥ 8 hours per day.
        
        **Compliance Rate:** Percentage of days where RN hours ≥ 8, excluding days with zero census.
        
        **Non-Compliance Rate:** Percentage of days where RN hours < 8, excluding days with zero census.
        
        **Data Sources:** CMS PBJ Daily Staffing Files (2017-2025), Provider Information, Affiliated Entity data.
        
        **Note:** Days with zero census are excluded from compliance calculations. The 8-hour minimum is a proposed federal requirement that was recently overturned (2025). Some states have their own RN staffing requirements.
        
        **Data Transparency**
        <div style="font-size: 0.9em; color: #666;">
        The RN Compliance Dashboard pulls directly from CMS data and is carefully vetted for accuracy. If you spot something that looks off, please let me know <a href="mailto:eric@320insight.com">eric@320insight.com</a> so I can set things right.
        </div>
        """, unsafe_allow_html=True)
    
    # PBJ Takeaway expander
    with st.expander("RN Compliance Takeaway", expanded=False):
        st.image("pbj_favicon.png", width=50)
        st.markdown(f"""
        **RN Compliance Analysis: {state}**
        
        **Latest Quarter Performance:** {compliance_rate:.1f}% compliance rate with {non_compliant_days:,} non-compliant days out of {total_days:,} total days across {facility_count} facilities.
        
        **State Context:** {state} has {facility_count} nursing homes reporting data in the latest quarter.
        
        **Compliance Trend:** Review the quarterly trend chart above to see how {state}'s RN compliance has changed over time.
        
        **Key Insight:** States with higher RN compliance rates typically have better resident outcomes and fewer regulatory citations. The facilities listed above may require additional attention to improve RN staffing levels.
        """, unsafe_allow_html=True)

def main():
    """Main application function."""
    # Check if data is available
    if not os.path.exists('rn_compliance_analysis.csv'):
        st.error("RN compliance data not found. Please run the compliance analysis script first.")
        return
    
    # Initialize session state for state selection
    if 'selected_state' not in st.session_state:
        st.session_state.selected_state = None
    
    # Initialize session state for facility selection
    if 'selected_facility' not in st.session_state:
        st.session_state.selected_facility = None
    
    # Handle state selection first (takes precedence)
    if st.session_state.selected_state:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("← Back to States"):
                st.session_state.selected_state = None
                st.rerun()
        with col2:
            st.markdown("")
        
        # Display state details
        display_state_details(st.session_state.selected_state)
        return
    
    # Handle facility selection (only if no state is selected)
    if st.session_state.selected_facility:
        # Display facility details
        display_facility_details(st.session_state.selected_facility)
        return
    

    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Select Page",
        ["National Overview", "State Analysis"]
    )
    
    # Display header at the very top for all pages
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; margin-bottom: 1.5rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1);">
        <h1 style="font-size: 2.5rem; font-weight: 800; color: white; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); letter-spacing: -1px;">
            PBJ Nursing Home RN Compliance Dashboard
        </h1>
        <p style="font-size: 1.1rem; color: rgba(255,255,255,0.9); margin: 0.3rem 0; font-weight: 300; text-shadow: 1px 1px 2px rgba(0,0,0,0.2);">
            Analysis of RN 8-Hour Minimum
        </p>
        <p style="font-size: 0.9rem; color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-style: italic; font-weight: 300;">
            A free public resource by 320 Consulting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Facility search available on all pages
    st.markdown("## Facility Search")
    display_facility_search()
    
    # Display selected page
    if page == "National Overview":
        # Then display national overview with charts (without header)
        display_national_overview_no_header()
    elif page == "State Analysis":
        display_state_analysis()

if __name__ == "__main__":
    main()
