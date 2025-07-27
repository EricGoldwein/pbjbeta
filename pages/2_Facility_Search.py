import streamlit as st
import pandas as pd
from typing import List, Dict
import re

st.set_page_config(page_title="Facility Search", page_icon="��", layout="wide")

st.title("Facility Search")

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
    /* Left-justify table headers */
    table th {
        text-align: left !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load facility data once
@st.cache_data
def load_facility_data():
    return pd.read_csv('facility_lite_metrics.csv', dtype={'PROVNUM': str})

@st.cache_data
def load_provider_info_data():
    """Load and cache provider info data for ownership entity lookup."""
    try:
        return pd.read_csv('NH_ProviderInfo_Jun2025.csv', dtype={'CMS Certification Number (CCN)': str, 'Affiliated Entity ID': str})
    except Exception as e:
        st.error(f"Error loading provider info data: {str(e)}")
        return pd.DataFrame()

def proper_title_case(text):
    """Convert text to proper title case (first letter capitalized, articles/prepositions lowercase)."""
    if pd.isna(text) or not isinstance(text, str):
        return '---'

    # Words to keep lowercase unless first word
    lowercase_words = {'and', 'of', 'at', 'the', 'in', 'on', 'for', 'to', 'with', 'by', 'a', 'an'}

    # True abbreviations to always uppercase
    uppercase_words = {'llc', 'ltc', 'lp', 'llp', 'pllc', 'pc', 'pa', 'plc', 'co', 'pl', 'corp', 'pllp', 'llc.', 'inc.', 'pllc.'}

    words = re.split(r'(\W+)', text)
    result = []
    for i, word in enumerate(words):
        w = word.lower()
        # Uppercase if in set
        if w in uppercase_words:
            result.append(w.upper())
        elif i != 0 and w in lowercase_words:
            result.append(w)
        else:
            result.append(w.capitalize())
    return ''.join(result)

def get_facility_affiliated_entity(provnum: str) -> tuple:
    """Get the affiliated entity name and ID for a specific facility."""
    try:
        provider_data = load_provider_info_data()
        if provider_data.empty:
            return None, None

        # Find the facility by CCN
        facility_data = provider_data[provider_data['CMS Certification Number (CCN)'] == provnum]

        if facility_data.empty:
            return None, None

        # Get the affiliated entity name and ID
        affiliated_entity_name = facility_data.iloc[0]['Affiliated Entity Name']
        affiliated_entity_id = facility_data.iloc[0]['Affiliated Entity ID']

        # Return None if it's NaN, otherwise return the entity name and ID
        if pd.notna(affiliated_entity_name) and pd.notna(affiliated_entity_id):
            return proper_title_case(str(affiliated_entity_name)), str(int(affiliated_entity_id))
        return None, None

    except Exception as e:
        print(f"Error getting affiliated entity for {provnum}: {str(e)}")
        return None, None

def get_ownership_entity_link(provnum: str) -> str:
    """Get the ownership entity link HTML for a facility."""
    entity_name, entity_id = get_facility_affiliated_entity(provnum)

    if entity_name and entity_id:
        return f'<a href="/?level=Entity&entity={entity_id}" style="color: #1976d2; text-decoration: none;" target="_self">{entity_name}</a>'
    else:
        return '---'

facilities_df = load_facility_data()

# Render the selectboxes in the right places, just below the container
col1, col2 = st.columns([1,2])
with col1:
    state = st.selectbox(
        "Select State",
        [""] + sorted(pd.read_csv('state_lite_metrics.csv')['STATE'].unique().tolist())
    )

def smart_title(name: str) -> str:
    if not isinstance(name, str):
        name = str(name) if name is not None else ""
    # Words to keep lowercase unless first word
    lowercase_words = {'and', 'of', 'at'}
    # True abbreviations to always uppercase
    uppercase_words = {'llc', 'ltc', 'lp', 'llp', 'pllc', 'pc', 'pa', 'plc', 'co', 'pl', 'corp', 'pllp', 'llc.', 'inc.', 'pllc.'}
    words = re.split(r'(\W+)', name)
    result = []
    for i, word in enumerate(words):
        w = word.lower()
        # Uppercase if in set
        if w in uppercase_words:
            result.append(w.upper())
        elif i != 0 and w in lowercase_words:
            result.append(w)
        else:
            result.append(w.capitalize())
    return ''.join(result)

# Create filtered search options based on selected state
if state:
    state_facilities = facilities_df[facilities_df['STATE'] == state]
    search_options = [f"{smart_title(row['PROVNAME'])} ({row['PROVNUM']})" for _, row in state_facilities[['PROVNAME', 'PROVNUM']].drop_duplicates().iterrows()]
else:
    search_options = [f"{smart_title(row['PROVNAME'])} ({row['PROVNUM']})" for _, row in facilities_df[['PROVNAME', 'PROVNUM']].drop_duplicates().iterrows()]

# Sort options alphabetically by facility name
search_options.sort()

with col2:
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
# Show selected state above the table if a state is selected (show immediately when state is selected)
if state:
    # Map state abbreviation to full name (copy from PBJ_Dashboard.py or define here)
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
    full_state_name = state_name_map.get(state, state)
    st.markdown(f"<div style='font-size: 1em; color: #1976d2; margin-bottom: 8px;'>Showing results for <a href='/?level=State&state={state}' style='color: #1976d2; text-decoration: underline;' target='_self'>{full_state_name}</a></div>", unsafe_allow_html=True)

if state or search_term:
    results = search_facilities(state, search_term)

    if results:
        st.markdown('<div class="search-results">', unsafe_allow_html=True)
        # Create a DataFrame for better display
        df = pd.DataFrame(results)
        # Apply smart capitalization to Nursing Home names
        df['Nursing Home'] = df['PROVNAME'].apply(smart_title) + ' (' + df['PROVNUM'] + ')'
        # Rename columns for display
        df = df.rename(columns={
            'STATE': 'State'
        })
        df['Dashboard'] = df.apply(
            lambda row: f'<a href="/?level=Facility&facility={row["PROVNUM"]}" class="facility-link">View Staffing</a>',
            axis=1
        )
        # Add ownership entity column
        df['Ownership Entity'] = df.apply(
            lambda row: get_ownership_entity_link(row["PROVNUM"]),
            axis=1
        )
        # Reorder columns to include Ownership Entity
        display_cols = ['State', 'Nursing Home', 'Ownership Entity', 'Dashboard']
        df = df[display_cols]
        # Sort alphabetically by Nursing Home name
        df = df.sort_values('Nursing Home')
        # Display the results as HTML for clickable links
        st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No facilities found matching your search criteria.")

st.markdown('</div>', unsafe_allow_html=True)

# Help section
st.markdown("""
    <div style="background-color: #f8f9fa; padding: 30px; border-radius: 10px;">
        <p style="color: #555; line-height: 1.6; text-align: center;">
            Need help finding a facility? <a href="https://nursinghomedashboard.streamlit.app/Facility_Search" style="color: #1E88E5; text-decoration: underline; font-weight: 500;">Try the Facility Search page</a>.
        </p>
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