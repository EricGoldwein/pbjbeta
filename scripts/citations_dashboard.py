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

# Define staffing-related tags
STAFFING_TAGS = [
    'F-0725',  # Sufficient Nursing Staff
    'F-0726',  # Competent Nursing Staff
    'F-0727',  # RN 8 Hrs/7 Days/Wk, Full-Time DON
    'F-0728',  # Facility Hiring and Use of Nurse
    'F-0729',  # Nurse Aide Registry Verification, Retraining
    'F-0730',  # Nurse Aide Performance Review
    'F-0731',  # Waiver: Licensed Nurses 24 Hr/Day and RN Coverage
    'F-0732'   # Posted Nurse Staffing Information
]

# Load data
@st.cache_data
def load_data():
    citations_df = pd.read_csv('Citations/NH_HealthCitations_Apr2025.csv')
    descriptions_df = pd.read_csv('Citations/NH_CitationDescriptions_Apr2025.csv')
    
    # Create a combined tag number and description for easier searching
    descriptions_df['Tag_Description'] = descriptions_df['Deficiency Prefix'] + descriptions_df['Deficiency Tag Number'].astype(str) + ': ' + descriptions_df['Deficiency Description']
    descriptions_df['Tag_Short'] = descriptions_df['Deficiency Prefix'] + descriptions_df['Deficiency Tag Number'].astype(str)
    
    # Add a flag for staffing-related citations
    descriptions_df['Is_Staffing'] = descriptions_df['Tag_Short'].isin(STAFFING_TAGS)
    
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
    
    return merged_df, descriptions_df

# Load the data
df, descriptions_df = load_data()

# Create pages
page = st.sidebar.radio("Select Page", ["Citations Dashboard", "Citation Descriptions", "Premium Services"])

# Add premium services link in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### Premium Services")
st.sidebar.markdown("""
    Get deeper insights with our premium services:
    - Daily staffing analysis
    - Custom staffing reports
    - Ownership analysis
    - Citations analysis
""")
st.sidebar.markdown("[Learn More](premium_services)")

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

    # Staffing filter
    show_staffing_only = st.sidebar.checkbox("Show Staffing-Related Citations Only")

    # Deficiency Tag and Description filter
    tag_descriptions = sorted(df['Tag_Description'].unique())
    selected_tags = st.sidebar.multiselect("Deficiency Tag and Description", tag_descriptions)

    # Scope Severity Code filter
    severity_codes = sorted(df['Scope Severity Code'].unique())
    selected_severity = st.sidebar.multiselect("Scope Severity Code", severity_codes)

    # Apply filters
    filtered_df = df.copy()

    if ccn_search:
        filtered_df = filtered_df[filtered_df['CMS Certification Number (CCN)'].astype(str).str.contains(ccn_search, case=False)]

    if provider_search:
        filtered_df = filtered_df[filtered_df['Provider Name'].str.contains(provider_search, case=False)]

    if selected_states:
        filtered_df = filtered_df[filtered_df['State'].isin(selected_states)]

    if show_staffing_only:
        filtered_df = filtered_df[filtered_df['Is_Staffing'] == True]

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
            'Scope Severity Code',
            'Tag_Description'
        ]].copy()
        
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
        
        # Create a more readable title based on filters
        if selected_states:
            title = f'Top 20 Citations by Tag in {", ".join(selected_states)}'
        else:
            title = 'Top 20 Citations by Tag (All States)'
            
        fig1 = px.bar(
            top_tags,
            x='Tag',
            y='Count',
            title=title
        )
        fig1.update_layout(
            xaxis_tickangle=-45,
            xaxis_title="F-Tag",
            yaxis_title="Number of Citations"
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
    # Instead of using switch_page, we'll include the premium services content directly
    import premium_services

# Footer
st.markdown("---")
st.markdown("320 Consulting | Source: CMS Citations Data") 