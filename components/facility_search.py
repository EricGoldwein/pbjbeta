import streamlit as st
import pandas as pd
import duckdb
from pathlib import Path

# Hide the page from the sidebar
st.set_page_config(
    page_title="Facility Search - PBJ Dashboard",
    page_icon="🔍",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# Hide the sidebar navigation
st.markdown("""
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Add back button
st.markdown("""
    <div style="margin-bottom: 20px;">
        <a href="/" target="_self" style="color: #1E88E5; text-decoration: none; font-weight: 500;">
            ← Back to PBJ Dashboard
        </a>
    </div>
""", unsafe_allow_html=True)

# Initialize DuckDB connection
facility_db = duckdb.connect(':memory:')

@st.cache_data
def load_facility_data():
    """Load facility data from CSV."""
    try:
        df = pd.read_csv('facility_quarterly_metrics.csv', dtype={'PROVNUM': str})
        facility_db.execute("""
            CREATE TABLE IF NOT EXISTS facility_metrics AS 
            SELECT * FROM df
        """)
        return df
    except Exception as e:
        st.error(f"Error loading facility data: {str(e)}")
        return pd.DataFrame()

# Load facility data
facility_data = load_facility_data()

# Title
st.title("Facility Search")

# Instructions
st.markdown("""
    Use this tool to find your facility's CCN and name. You can search by:
    - Facility name
    - CCN (Certification Control Number)
""")

# State filter
states = sorted(facility_data['STATE'].unique())
selected_state = st.selectbox("Filter by State (Optional)", ["All States"] + list(states))

# Search box
search_term = st.text_input("Search by Facility Name or CCN")

# Filter data based on search term and state
if search_term:
    # Search in both PROVNUM and PROVNAME
    query = f"""
        SELECT DISTINCT PROVNUM, PROVNAME, STATE, COUNTY_NAME, CITY
        FROM facility_metrics 
        WHERE (PROVNUM LIKE '%{search_term}%' OR PROVNAME LIKE '%{search_term}%')
        {f"AND STATE = '{selected_state}'" if selected_state != "All States" else ""}
        ORDER BY PROVNAME
        LIMIT 50
    """
    results = facility_db.execute(query).fetchdf()
    
    if not results.empty:
        st.markdown("### Results")
        # Display results in a table
        st.dataframe(
            results,
            column_config={
                "PROVNUM": "CCN",
                "PROVNAME": "Facility Name",
                "STATE": "State",
                "COUNTY_NAME": "County",
                "CITY": "City"
            },
            hide_index=True
        )
    else:
        st.info("No matching facilities found. Try a different search term or state.")
else:
    st.info("Enter a facility name or CCN to search.") 