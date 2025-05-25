import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Set page config
st.set_page_config(
    page_title="Nursing Home Citations Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Define staffing-related tags (just the numbers)
STAFFING_TAG_NUMBERS = [
    '725',  # Sufficient Nursing Staff
    '726',  # Competent Nursing Staff
    '727',  # RN 8 Hrs/7 Days/Wk, Full-Time DON
    '728',  # Facility Hiring and Use of Nurse
    '729',  # Nurse Aide Registry Verification, Retraining
    '730',  # Nurse Aide Performance Review
    '731',  # Waiver: Licensed Nurses 24 Hr/Day and RN Coverage
    '732',  # Posted Nurse Staffing Information
    '353',  # Have enough nurses to care for every resident in a way that maximizes the resident's well being
    '354',  # Use a registered nurse at least 8 hours a day, 7 days a week
    '355',  # Request a waiver if it can't meet the nurse staffing requirements
    '356',  # Post nurse staffing information/data on a daily basis
    '494',  # Ensure that all full-time nurse aides employed for more than 4 months are fully trained and competent
    '495',  # Ensure that all nurse aides who have worked less than 4 months are enrolled in appropriate training
    '496',  # Receive registry verification and ensure nurse aides receive required retraining
    '497',  # Review the work of each nurse aide every year and give regular in-service training
    '498'   # Make sure that nurse aides show they have the skills and techniques to care for residents' needs
]

# Load data
@st.cache_data
def load_data():
    citations_df = pd.read_csv('Citations/NH_HealthCitations_Apr2025.csv')
    descriptions_df = pd.read_csv('Citations/NH_CitationDescriptions_Apr2025.csv')
    
    # Print unique tag numbers to debug
    print("Unique tag numbers in citations:", citations_df['Deficiency Tag Number'].unique())
    print("Unique tag numbers in descriptions:", descriptions_df['Deficiency Tag Number'].unique())
    
    # Create a combined tag number and description for easier searching
    descriptions_df['Tag_Description'] = descriptions_df['Deficiency Prefix'] + descriptions_df['Deficiency Tag Number'].astype(str) + ': ' + descriptions_df['Deficiency Description']
    descriptions_df['Tag_Short'] = descriptions_df['Deficiency Prefix'] + descriptions_df['Deficiency Tag Number'].astype(str)
    
    # Print unique tag shorts to debug
    print("Unique tag shorts in descriptions:", descriptions_df['Tag_Short'].unique())
    
    # Add a flag for staffing-related citations
    descriptions_df['Is_Staffing'] = descriptions_df['Deficiency Tag Number'].astype(str).isin(STAFFING_TAG_NUMBERS)
    
    # Merge citations with descriptions
    merged_df = pd.merge(
        citations_df,
        descriptions_df,
        left_on=['Deficiency Prefix', 'Deficiency Tag Number'],
        right_on=['Deficiency Prefix', 'Deficiency Tag Number'],
        how='left'
    )
    
    # Combine Provider Name and CCN
    merged_df['Provider'] = merged_df['Provider Name'] + ' (' + merged_df['CMS Certification Number (CCN)'].astype(str) + ')'
    
    # Create a direct staffing flag in the merged dataframe
    merged_df['Is_Staffing'] = merged_df['Deficiency Tag Number'].astype(str).isin(STAFFING_TAG_NUMBERS)
    
    # Convert Survey Date to datetime and extract year
    merged_df['Survey Date'] = pd.to_datetime(merged_df['Survey Date'])
    merged_df['Year'] = merged_df['Survey Date'].dt.year
    
    # Print staffing-related tags found
    staffing_tags = merged_df[merged_df['Is_Staffing'] == True]['Tag_Short'].unique()
    print("Staffing tags found in merged data:", staffing_tags)
    print("Number of staffing citations:", len(merged_df[merged_df['Is_Staffing'] == True]))
    
    return merged_df, descriptions_df

# Load the data
df, descriptions_df = load_data()

# Create pages
page = st.sidebar.radio("Select Page", ["Citations Dashboard", "Citation Descriptions", "Premium Services"])

if page == "Citations Dashboard":
    # Title
    st.title("Nursing Home Citations Dashboard")

    # Sidebar filters
    st.sidebar.header("Search Filters")

    # Search by CCN
    ccn_search = st.sidebar.text_input("CMS Certification Number (CCN)")

    # Search by Provider Name
    provider_search = st.sidebar.text_input("Provider Name")

    # State filter
    states = sorted(df['State'].unique())
    selected_states = st.sidebar.multiselect("State", states)

    # Year filter
    years = sorted(df['Year'].unique(), reverse=True)  # Most recent years first
    selected_years = st.sidebar.multiselect("Year", years)

    # Deficiency Tag and Description filter
    tag_descriptions = sorted(df['Tag_Description'].unique())
    selected_tags = st.sidebar.multiselect("Deficiency Tag and Description", tag_descriptions)

    # Scope Severity Code filter
    severity_codes = sorted(df['Scope Severity Code'].unique())
    selected_severity = st.sidebar.multiselect("Scope Severity Code", severity_codes)

    # Main filters section
    st.markdown("### Filters")
    
    # Create two columns for the filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Staffing filter with tooltip
        show_staffing_only = st.checkbox(
            "Show Staffing-Related Citations Only",
            help="""Filter to show only citations related to nurse staffing and nurse aide training.
            Includes F-tags: 725-732 (newer tags) and 353-356, 494-498 (older tags)"""
        )
    
    with col2:
        # Severe citations filter with tooltip
        show_severe_only = st.checkbox(
            "Show Only Severe Citations",
            help="Filter to show only citations with scope severity of G or higher (G, H, I, J, K, L)"
        )

    # Apply filters
    filtered_df = df.copy()

    if ccn_search:
        filtered_df = filtered_df[filtered_df['CMS Certification Number (CCN)'].astype(str).str.contains(ccn_search, case=False)]

    if provider_search:
        filtered_df = filtered_df[filtered_df['Provider Name'].str.contains(provider_search, case=False)]

    if selected_states:
        filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

    if selected_years:
        filtered_df = filtered_df[filtered_df['Year'].isin(selected_years)]

    if show_staffing_only:
        # Filter for staffing-related citations
        filtered_df = filtered_df[filtered_df['Deficiency Tag Number'].astype(str).isin(STAFFING_TAG_NUMBERS)]

    if show_severe_only:
        # Filter for severe citations (G or higher)
        severe_codes = ['G', 'H', 'I', 'J', 'K', 'L']
        filtered_df = filtered_df[filtered_df['Scope Severity Code'].isin(severe_codes)]

    if selected_tags:
        filtered_df = filtered_df[filtered_df['Tag_Description'].isin(selected_tags)]

    if selected_severity:
        filtered_df = filtered_df[filtered_df['Scope Severity Code'].isin(selected_severity)]

    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Citations", f"{len(filtered_df):,}")
    with col2:
        st.metric("Unique Facilities", f"{filtered_df['CMS Certification Number (CCN)'].nunique():,}")
    with col3:
        st.metric("States Covered", f"{filtered_df['State'].nunique():,}")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Citations Data", "Deficiency Analysis", "Geographic Distribution", "Citation Details"])

    with tab1:
        # Display the filtered data
        display_df = filtered_df[[
            'State',
            'Provider',
            'Survey Date',
            'Scope Severity Code',
            'Tag_Description'
        ]].copy()
        
        # Sort by Survey Date in descending order (most recent first)
        display_df = display_df.sort_values('Survey Date', ascending=False)
        
        # Format the Survey Date
        display_df['Survey Date'] = display_df['Survey Date'].dt.strftime('%m-%d-%Y')
        
        # Remove index from display
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

    with tab2:
        # Deficiency Tag Distribution
        tag_counts = filtered_df['Tag_Short'].value_counts().reset_index()
        tag_counts.columns = ['Tag', 'Count']
        
        # Get the top 20 tags
        top_tags = tag_counts.head(20)
        
        # Merge with descriptions for tooltips
        top_tags = pd.merge(
            top_tags,
            descriptions_df[['Tag_Short', 'Deficiency Description']],
            left_on='Tag',
            right_on='Tag_Short',
            how='left'
        )
        
        # Create a more readable title based on filters
        if selected_states:
            title = f'Top 20 Citations by Tag in {", ".join(selected_states)}'
        else:
            title = 'Top 20 Citations by Tag (All States)'
            
        fig1 = px.bar(
            top_tags,
            x='Tag',
            y='Count',
            title=title,
            hover_data=['Deficiency Description']
        )
        fig1.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="F-Tag",
            yaxis_title="Number of Citations",
            hovermode='x unified'
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Scope Severity Distribution
        fig2 = px.pie(
            filtered_df,
            names='Scope Severity Code',
            title='Citations by Scope Severity'
        )
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        # State-wise distribution
        state_counts = filtered_df['State'].value_counts().reset_index()
        state_counts.columns = ['State', 'Count']
        fig3 = px.choropleth(
            state_counts,
            locations='State',
            locationmode="USA-states",
            color='Count',
            scope="usa",
            title='Citations by State',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with tab4:
        # Citation Details
        st.header("Citation Details")
        
        # Select a specific citation to view details
        citation_options = filtered_df['Tag_Description'].unique()
        selected_citation = st.selectbox("Select a Citation", citation_options)
        
        if selected_citation:
            citation_details = filtered_df[filtered_df['Tag_Description'] == selected_citation].iloc[0]
            
            st.subheader(f"Citation {selected_citation}")
            
            # Show facilities with this citation
            st.subheader("Facilities with this Citation")
            facilities_with_citation = filtered_df[filtered_df['Tag_Description'] == selected_citation]
            st.dataframe(
                facilities_with_citation[[
                    'State',
                    'Provider',
                    'Scope Severity Code'
                ]],
                use_container_width=True,
                hide_index=True
            )

elif page == "Citation Descriptions":
    st.title("Citation Descriptions Search")
    
    # Search functionality
    search_term = st.text_input("Search Citation Descriptions", "")
    
    if search_term:
        # Search in both tag number and description
        filtered_descriptions = descriptions_df[
            descriptions_df['Deficiency Tag Number'].astype(str).str.contains(search_term, case=False) |
            descriptions_df['Deficiency Description'].str.contains(search_term, case=False)
        ]
    else:
        filtered_descriptions = descriptions_df
    
    # Display the descriptions table
    st.dataframe(
        filtered_descriptions[[
            'Deficiency Prefix',
            'Deficiency Tag Number',
            'Deficiency Description'
        ]].rename(columns={
            'Deficiency Prefix': 'Prefix',
            'Deficiency Tag Number': 'Tag Number',
            'Deficiency Description': 'Description'
        }),
        use_container_width=True,
        hide_index=True
    )

else:  # Premium Services page
    st.title("320 Premium Services")
    
    # Premium Header
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%); color: white; padding: 2rem; border-radius: 10px; margin-bottom: 2rem;">
            <h1>320 Premium Services</h1>
            <p style="font-size: 1.2em; margin-bottom: 0;">Digging deeper into nursing home data</p>
        </div>
    """, unsafe_allow_html=True)

    # Introduction
    st.markdown("""
        <b>320 Consulting</b> offers custom reports with full breakdowns of all nurse and non-nurse positions, staffing trends over time, ownership data, citation histories, and comparisons by geography or any category you need — built to support your case, investigation, or advocacy work.<br><br>
        To request a report or talk through what you need: eric@320insight.com
    """, unsafe_allow_html=True)

    # Premium Features
    st.markdown("### Premium Services")

    # Feature 1: Daily Staffing Lookup
    st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #1E88E5;">
            <h3 style="color: #1E88E5; margin-top: 0;">📊 Daily Staffing Analysis</h3>
            <p>Access detailed daily staffing data for any facility since 2017, including:</p>
            <ul>
                <li>Daily staffing levels for all positions</li>
                <li>Anomaly detection for unusual staffing patterns</li>
                <li>Historical trend analysis</li>
                <li>Custom date range comparisons</li>
                <li>Shareable data visualizations and tables</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Feature 2: Custom Staffing Reports
    st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #1E88E5;">
            <h3 style="color: #1E88E5; margin-top: 0;">👥 Comprehensive Staffing Reports</h3>
            <p>Get detailed reports on every position, including:</p>
            <ul>
                <li>Administrators and DONs</li>
                <li>RNs, LPNs, and CNAs</li>
                <li>Physical, Occupational, and Speech Therapists</li>
                <li>Social Workers and Activities Staff</li>
                <li>Contract staff utilization</li>
                <li>Custom shareable data visualizations</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Feature 3: Ownership Analysis
    st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #1E88E5;">
            <h3 style="color: #1E88E5; margin-top: 0;">🏢 Ownership Group Analysis</h3>
            <p>Understand facility ownership patterns and trends:</p>
            <ul>
                <li>Affiliated entity identification</li>
                <li>Cross-facility staffing patterns</li>
                <li>Ownership group performance metrics</li>
                <li>Historical ownership changes</li>
                <li>Custom shareable ownership reports</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Feature 4: Citations Analysis
    st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #1E88E5;">
            <h3 style="color: #1E88E5; margin-top: 0;">📋 Citations Analysis</h3>
            <p>Comprehensive analysis of facility citations:</p>
            <ul>
                <li>Form CMS-2567 data integration</li>
                <li>Citation summaries and trends</li>
                <li>Staffing correlation analysis</li>
                <li>Historical citation patterns</li>
                <li>Custom shareable citation reports</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    # Contact Section
    st.markdown("""
        <div style="background-color: #e3f2fd; padding: 2rem; border-radius: 8px; margin-top: 2rem;">
            <h3>Ready to get started?</h3>
            <p>Contact Eric to request a custom report or talk through your project:</p>
            <p><a href="mailto:eric@320insight.com">📧 eric@320insight.com</a></p>
        </div>
    """, unsafe_allow_html=True)

    # Data Source
    st.markdown("""
        <div style="font-size: 0.9em; color: #666; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee;">
            <p><strong>Data Source:</strong> Our analysis is primarily based on CMS Payroll-Based Journal (PBJ) data, 
            supplemented with additional CMS datasets and proprietary analysis tools.</p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("320 Consulting | Source: CMS Citations Data") 