import streamlit as st
import pandas as pd
import duckdb

st.set_page_config(
    page_title="Facility Data - PBJ Dashboard",
    page_icon="🔍",
    layout="wide"
)

# Add home button
st.markdown("""
    <div style="margin-bottom: 20px;">
        <a href="/" target="_self" style="color: #1E88E5; text-decoration: none; font-weight: 500;">
            ← Back to PBJ Dashboard
        </a>
    </div>
""", unsafe_allow_html=True)

st.title("Facility Data")
st.markdown("""
    Search for nursing home by name or CCN (Provider Number). 
    Filter by state to narrow down results.
""")

# Load facility data
@st.cache_data
def load_facility_data():
    try:
        # Load facility metrics into DuckDB
        facility_metrics = pd.read_csv('facility_quarterly_metrics.csv', dtype={'PROVNUM': str})
        
        # Get unique facilities with their latest names
        facility_db = duckdb.connect(':memory:')
        facility_db.execute("""
            CREATE TABLE facility_metrics AS 
            SELECT * FROM facility_metrics
        """)
        
        # Get unique facilities with their latest names
        query = """
            WITH latest_facilities AS (
                SELECT 
                    PROVNUM,
                    PROVNAME,
                    STATE,
                    COUNTY_NAME,
                    CITY,
                    ROW_NUMBER() OVER (PARTITION BY PROVNUM ORDER BY date DESC) as rn
                FROM facility_metrics
            )
            SELECT 
                PROVNUM,
                PROVNAME,
                STATE,
                COUNTY_NAME,
                CITY
            FROM latest_facilities
            WHERE rn = 1
            ORDER BY STATE, PROVNAME
        """
        
        facilities = facility_db.execute(query).fetchdf()
        return facilities
    except Exception as e:
        st.error(f"Error loading facility data: {str(e)}")
        return pd.DataFrame()

# Load the data
facilities = load_facility_data()

# Create filters
col1, col2 = st.columns(2)

with col1:
    # State filter
    states = ['All States'] + sorted(facilities['STATE'].unique().tolist())
    selected_state = st.selectbox('Filter by State', states)

with col2:
    # Search box
    search_term = st.text_input('Search by Facility Name or CCN', '')

# Filter the data
if selected_state != 'All States':
    facilities = facilities[facilities['STATE'] == selected_state]

if search_term:
    search_term = search_term.lower()
    facilities = facilities[
        facilities['PROVNUM'].str.lower().str.contains(search_term) |
        facilities['PROVNAME'].str.lower().str.contains(search_term)
    ]

# Display the results
if not facilities.empty:
    # Format the data for display
    display_data = facilities.copy()
    
    # Create clickable facility names with links
    def create_facility_link(row):
        return f'<a href="/?level=Facility&facility={row["PROVNUM"]}" target="_self">{row["PROVNAME"]} ({row["PROVNUM"]})</a>'
    
    display_data['Facility'] = display_data.apply(create_facility_link, axis=1)
    display_data['Location'] = display_data.apply(
        lambda row: f"{row['CITY']}, {row['STATE']}", 
        axis=1
    )
    
    # Select columns to display
    display_data = display_data[['Facility', 'Location', 'COUNTY_NAME']]
    display_data.columns = ['Facility', 'Location', 'County']
    
    # Display the table with HTML
    st.markdown(display_data.to_html(escape=False, index=False), unsafe_allow_html=True)
    
    st.markdown(f"Found {len(facilities)} facilities matching your search.")
else:
    st.info("No facilities found matching your search criteria.")

# Add instructions
st.markdown("""
    ### How to Use This Search
    1. Use the state filter to narrow down results by state
    2. Search by facility name or CCN (Provider Number)
    3. Click on a facility name to view its details in the main dashboard
    4. The CCN (Provider Number) is shown in parentheses after each facility name
""")

# Add custom CSS for better styling
st.markdown("""
    <style>
        /* Style the facility links */
        .facility-link {
            color: #1E88E5;
            text-decoration: none;
            font-weight: 500;
        }
        .facility-link:hover {
            text-decoration: underline;
        }
        /* Style the table */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1em 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 500;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
    </style>
""", unsafe_allow_html=True) 