import streamlit as st
import pandas as pd
from typing import List, Dict

# Set page configuration
st.set_page_config(
    page_title="Facility Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    /* Add styles for the results table */
    .results-table {
        width: 100%;
        border-collapse: collapse;
    }
    .results-table th {
        background-color: #f8f9fa;
        padding: 12px;
        text-align: left;
        border-bottom: 2px solid #e0e0e0;
        color: #2c3338;
        font-weight: 500;
    }
    .results-table td {
        padding: 12px;
        border-bottom: 1px solid #e0e0e0;
        color: #555;
    }
    .results-table tr:hover {
        background-color: #f8f9fa;
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

def proper_title_case(text: str) -> str:
    """Convert text to proper title case, keeping words like 'and', 'of', etc. lowercase."""
    if not text:
        return text
        
    # Words that should remain lowercase unless they're the first word
    lowercase_words = {'and', 'at', 'of', 'the', 'in', 'on', 'for', 'to', 'with', 'by'}
    
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
        
        # Get unique facilities with county information
        unique_facilities = facilities[['PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME']].drop_duplicates()
        
        # Apply proper title case to names
        unique_facilities['PROVNAME'] = unique_facilities['PROVNAME'].apply(proper_title_case)
        unique_facilities['COUNTY_NAME'] = unique_facilities['COUNTY_NAME'].apply(proper_title_case)
        
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
        
        # Create a DataFrame for the results
        display_data = []
        for facility in results:
            care_compare_url = f"https://www.medicare.gov/care-compare/details/nursing-home/{facility['PROVNUM']}/view-all?state={facility['STATE']}"
            display_data.append({
                'State': facility['STATE'],
                'Prov Num': facility['PROVNUM'],
                'Prov Name (County)': f"{facility['PROVNAME']} ({facility['COUNTY_NAME']})",
                'View Details': f"[View Details](/?level=Facility&facility={facility['PROVNUM']})",
                'Care Compare': f"[Care Compare]({care_compare_url})"
            })
        
        # Convert to DataFrame and display
        if display_data:
            df = pd.DataFrame(display_data)
            st.dataframe(
                df,
                column_config={
                    "View Details": st.column_config.Column(
                        "View Details",
                        help="View detailed facility information",
                        display_text="View Details"
                    ),
                    "Care Compare": st.column_config.Column(
                        "Care Compare",
                        help="View facility on Medicare Care Compare",
                        display_text="Care Compare"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
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