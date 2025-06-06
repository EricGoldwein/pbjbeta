import streamlit as st
import pandas as pd
from typing import List, Dict

# Add custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #2c3338;
        font-size: 2.5em;
        margin-bottom: 0;
    }
    .back-link {
        color: #1E88E5;
        text-decoration: none;
        font-weight: 500;
        display: inline-block;
        margin-bottom: 20px;
    }
    .back-link:hover {
        text-decoration: underline;
    }
    .search-container {
        background-color: #f8f9fa;
        padding: 30px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .search-results {
        margin-top: 20px;
    }
    .facility-link {
        color: #1E88E5;
        text-decoration: none;
        font-weight: 500;
    }
    .facility-link:hover {
        text-decoration: underline;
    }
    .stSelectbox {
        margin-bottom: 0;
    }
    </style>
""", unsafe_allow_html=True)

# Main content
st.markdown("""
    <h1 class="main-header">Facility Search</h1>
    <a href="/" class="back-link">← Back to Dashboard</a>
""", unsafe_allow_html=True)

# Search interface
st.markdown('<div class="search-container">', unsafe_allow_html=True)
st.markdown("### Search for a Facility")

# Create columns for search inputs
col1, col2 = st.columns(2)

with col1:
    state = st.selectbox(
        "Select State",
        [""] + sorted(pd.read_csv('state_lite_metrics.csv')['STATE'].unique().tolist())
    )

# Load facility data once
@st.cache_data
def load_facility_data():
    return pd.read_csv('facility_lite_metrics.csv', dtype={'PROVNUM': str})

facilities_df = load_facility_data()

# Filter facilities by state if selected
if state:
    state_facilities = facilities_df[facilities_df['STATE'] == state]
else:
    state_facilities = facilities_df

# Create search options
search_options = [f"{row['PROVNAME']} ({row['PROVNUM']})" for _, row in state_facilities[['PROVNAME', 'PROVNUM']].drop_duplicates().iterrows()]

with col2:
    # Use selectbox for autocomplete functionality
    search_term = st.selectbox(
        "Enter Provider Name or CCN",
        options=[""] + search_options,
        key="search_input",
        help="Type to search facilities"
    )

# Function to search facilities
@st.cache_data
def search_facilities(state: str, search_term: str) -> List[Dict[str, str]]:
    try:
        # Filter by state if selected
        if state:
            facilities = facilities_df[facilities_df['STATE'] == state]
        else:
            facilities = facilities_df
        
        # Filter by search term if provided
        if search_term:
            # Extract CCN from search term if it's in the format "Name (CCN)"
            if '(' in search_term:
                ccn = search_term.split('(')[-1].strip(')')
                facilities = facilities[facilities['PROVNUM'] == ccn]
            else:
                search_term = search_term.lower()
                facilities = facilities[
                    facilities['PROVNUM'].str.lower().str.contains(search_term) |
                    facilities['PROVNAME'].str.lower().str.contains(search_term)
                ]
        
        # Get unique facilities
        unique_facilities = facilities[['PROVNUM', 'PROVNAME', 'STATE']].drop_duplicates()
        
        return unique_facilities.to_dict('records')
    except Exception as e:
        st.error(f"Error searching facilities: {str(e)}")
        return []

# Display search results
if search_term:
    results = search_facilities(state, search_term)
    
    if results:
        st.markdown('<div class="search-results">', unsafe_allow_html=True)
        st.markdown("### Search Results")
        
        # Create a DataFrame for better display
        df = pd.DataFrame(results)
        df['Link'] = df.apply(
            lambda row: f'<a href="/?level=Facility&facility={row["PROVNUM"]}" class="facility-link">View Details</a>',
            axis=1
        )
        
        # Display the results
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No facilities found matching your search criteria.")

st.markdown('</div>', unsafe_allow_html=True)

# Help section
st.markdown("""
    <div style="background-color: #f8f9fa; padding: 30px; border-radius: 10px;">
        <p style="color: #555; line-height: 1.6; text-align: center;">
            Contact <a href="mailto:eric@320insight.com" style="color: #1E88E5; text-decoration: none; font-weight: 500;">eric@320insight.com</a> to request a custom report or talk through what you need.
        </p>
    </div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
        <p>Source: <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" style="color: #1E88E5; text-decoration: none;">CMS Payroll-Based Journal Data, 2017-2024</a></p>
        <p>By <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">320 Consulting LLC</a></p>
    </div>
""", unsafe_allow_html=True) 