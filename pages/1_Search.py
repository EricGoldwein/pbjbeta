import streamlit as st
import pandas as pd
from typing import List, Dict
import re

st.set_page_config(page_title="PBJ Nursing Home Staffing Dashboard", page_icon="🔍", layout="wide")

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
    .search-tabs {
        margin-bottom: 20px;
    }
    .search-section {
        background: white;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #e3eaf3;
        margin-bottom: 15px;
    }
    /* Dark blue title styling */
    .title-container {
        color: #1a2233;
        font-size: 2.5em;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    /* Info box styling */
    .info-box {
        background: #f7fafd;
        border-radius: 6px;
        padding: 14px 14px 8px 14px;
        margin-bottom: 14px;
        border: 1px solid #e3eaf3;
        width: 100%;
        margin-left: 0;
        margin-right: 0;
    }
    .info-text {
        font-size: 1.08em;
        color: #234;
        font-weight: 600;
        margin-bottom: 2px;
    }
    .mobile-info {
        font-size: 0.89em;
        color: #5a6473;
        font-style: italic;
        margin-bottom: 8px;
        font-weight: 400;
    }
    @media (max-width: 768px) {
        .title-container {
            font-size: 2em;
        }
        .info-box {
            padding: 12px;
            margin-bottom: 12px;
        }
        .info-text {
            font-size: 1em;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Styled title
st.markdown('<div class="title-container">PBJ Nursing Home Staffing Dashboard</div>', unsafe_allow_html=True)

# Info box
st.markdown('''
    <div class="info-box">
        <div class="info-text">
            A free public resource from <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 700;"><b>320 Consulting</b></a>, featuring quarterly staffing data (2017–2024) across every U.S. nursing home.
        </div>
    </div>
''', unsafe_allow_html=True)

# Load data once
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

@st.cache_data
def load_ownership_data():
    """Load and cache ownership entity data."""
    try:
        return pd.read_csv('Nursing_Home_Affiliated_Entity_Performance_Measures_Jun_2025.csv')
    except Exception as e:
        st.error(f"Error loading ownership data: {str(e)}")
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

def smart_title(name: str) -> str:
    if not isinstance(name, str):
        name = str(name) if name is not None else ""
    # Words to keep lowercase unless first word
    lowercase_words = {'and', 'of', 'at'}
    # True abbreviations to always uppercase
    uppercase_words = {'llc', 'ltc', 'lp', 'llp', 'pllc', 'pc', 'pa', 'plc', 'co', 'pl', 'corp', 'pllp', 'llc.', 'inc.', 'pllc.', 'abcm', 'snf', 'hcf', 'ltc', 'hca', 'cch', 'ajc', 'hmg'}
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

def get_facility_ownership_info(provnum: str) -> tuple:
    """Get the ownership entity name and ID for a specific facility."""
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
        print(f"Error getting ownership info for {provnum}: {str(e)}")
        return None, None

# Load data
facilities_df = load_facility_data()
provider_info_df = load_provider_info_data()
ownership_df = load_ownership_data()

# Create search tabs
tab1, tab2, tab3 = st.tabs(["🔍 Facility", "🏢 Ownership", "🗺️ State"])

with tab1:
    st.markdown("### Search by Facility Name or CCN")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        state_filter = st.selectbox(
            "Filter by State (Optional)",
            [""] + sorted(facilities_df['STATE'].unique().tolist()),
            key="facility_state_filter"
        )
    
    # Create filtered search options based on selected state
    if state_filter:
        state_facilities = facilities_df[facilities_df['STATE'] == state_filter]
        search_options = [f"{smart_title(row['PROVNAME'])} ({row['PROVNUM']})" for _, row in state_facilities[['PROVNAME', 'PROVNUM']].drop_duplicates().iterrows()]
    else:
        state_facilities = facilities_df  # Use full dataset when no state filter
        search_options = [f"{smart_title(row['PROVNAME'])} ({row['PROVNUM']})" for _, row in facilities_df[['PROVNAME', 'PROVNUM']].drop_duplicates().iterrows()]

    # Sort options alphabetically by facility name
    search_options.sort()
    
    with col2:
        facility_search = st.selectbox(
            "Enter Facility Name or CCN",
            options=[""] + search_options,
            key="facility_search_input",
            help="Type to search facilities"
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
            display_df['Nursing Home'] = unique_results['PROVNAME'].apply(smart_title) + ' (' + unique_results['PROVNUM'] + ')'
            display_df['State'] = unique_results['STATE']
            display_df['Dashboard'] = unique_results['PROVNUM'].apply(
                lambda x: f'<a href="/?level=Facility&facility={x}" style="color: #1976d2; text-decoration: none;" target="_self">View</a>'
            )
            
            # Sort alphabetically
            display_df = display_df.sort_values('Nursing Home')
            
            st.markdown("#### Search Results")
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
            display_df['Nursing Home'] = unique_facilities['PROVNAME'].apply(smart_title) + ' (' + unique_facilities['PROVNUM'] + ')'
            display_df['State'] = unique_facilities['STATE']
            display_df['Dashboard'] = unique_facilities['PROVNUM'].apply(
                lambda x: f'<a href="/?level=Facility&facility={x}" style="color: #1976d2; text-decoration: none;" target="_self">View</a>'
            )
            
            # Sort alphabetically
            display_df = display_df.sort_values('Nursing Home')
            
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
            st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.info("No facilities found for this state.")

with tab2:
    st.markdown("### Search by Ownership Group")
    
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
            "Select Ownership Group",
            options=ownership_options,
            key="ownership_search_input",
            help="Choose an ownership group to view their dashboard"
        )
        
        if ownership_search_display:
            # Get the stored entity ID and name
            entity_id = st.session_state.get(f"entity_{ownership_search_display}")
            ownership_name = st.session_state.get(f"name_{ownership_search_display}")
            
            if entity_id and ownership_name:
                # Create styled button link - navigate to ownership page directly
                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <a href="/Ownership?entity_id={entity_id}" 
                       style="display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                              color: white; padding: 12px 24px; text-decoration: none; border-radius: 8px; 
                              font-weight: 600; font-size: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
                              transition: all 0.3s ease;">
                        🏢 View Dashboard for {smart_title(ownership_name)}
                    </a>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Ownership group not found.")
    else:
        st.info("Ownership data not available.")

with tab3:
    st.markdown("### Search by State")
    
    state_search = st.selectbox(
        "Select State",
        ["", "USA"] + sorted(facilities_df['STATE'].unique().tolist()),
        key="state_search_input",
        help="Choose a state to view its dashboard"
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
            button_text = "🏠 View USA Dashboard"
        else:
            link_url = f"/?level=State&state={state_search}"
            button_text = f"🗺️ View Dashboard for {full_state_name}"
        
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

# Help section
st.markdown("""
    <div style="background-color: #f8f9fa; padding: 30px; border-radius: 10px;">
        <p style="color: #555; line-height: 1.6; text-align: center;">
            Search by facility (name or CCN), ownership group, or state.
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