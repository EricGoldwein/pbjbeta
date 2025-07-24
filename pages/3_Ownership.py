import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

st.set_page_config(page_title="Ownership", page_icon="🏢", layout="wide")

# Add the parent directory to the path to import from PBJ_Dashboard
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

def capitalize_entity_name(name):
    if pd.isna(name):
        return name
    lowercase_words = {'and', 'or', 'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    abbreviations = {'llc', 'inc', 'corp', 'ltd', 'lp', 'pllc', 'snf', 'pc', 'plc', 'llp', 'pa', 'pllp', 'pl', 'a&m', 'wlc', 'dba', 'p.c.', 'p.a.', 's.c.', 's.a.', 'nfp', 'cna', 'rn', 'lpn', 'md', 'do', 'msw', 'pt', 'ot', 'slp', 'hosp', 'med', 'svc', 'svcs'}
    words = name.split()
    capitalized_words = []
    for i, word in enumerate(words):
        word_clean = word.lower().strip('.,')
        # Capitalize abbreviations or 2-3 letter non-words
        if word_clean in abbreviations:
            capitalized_words.append(word.upper())
        elif (len(word) in (2, 3) and word_clean not in lowercase_words and not word.islower()):
            capitalized_words.append(word.upper())
        elif i == 0 or word_clean not in lowercase_words:
            capitalized_words.append(word.capitalize())
        else:
            capitalized_words.append(word.lower())
    return ' '.join(capitalized_words)

@st.cache_data
def load_affiliated_entity_data():
    """Load and cache affiliated entity performance measures data."""
    try:
        df = pd.read_csv('Nursing_Home_Affiliated_Entity_Performance_Measures_Jun_2025.csv')
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
        df = pd.read_csv('NH_ProviderInfo_Jun2025.csv', dtype={'CMS Certification Number (CCN)': str})
        # Clean and standardize the data
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

# Custom CSS for better styling
st.markdown("""
    <style>
        .entity-header {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            color: white;
            padding: 2rem;
            border-radius: 12px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .metric-card {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }
        .facility-link {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
        }
        .facility-link:hover {
            text-decoration: underline;
        }
        .search-container {
            background: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .section-header {
            background: linear-gradient(90deg, #f8f9fa, #e9ecef);
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin: 1.5rem 0 1rem 0;
            border-left: 4px solid #667eea;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        .chart-container {
            background: white;
            padding: 1.5rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin: 1rem 0;
        }
        .facility-table {
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .no-entity-selected {
            text-align: center;
            padding: 4rem 2rem;
            color: #666;
        }
        .no-entity-selected h3 {
            color: #333;
            margin-bottom: 1rem;
        }
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: help;
        }
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 320px;
            background-color: #1f2937;
            color: #f9fafb;
            text-align: left;
            border-radius: 8px;
            padding: 12px;
            position: absolute;
            z-index: 9999;
            bottom: 125%;
            left: 50%;
            margin-left: -160px;
            opacity: 0;
            transition: opacity 0.3s;
            font-size: 13px;
            line-height: 1.5;
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            border: 1px solid #374151;
        }
        .tooltip .tooltiptext::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #1f2937 transparent transparent transparent;
        }
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        /* Facility table styling */
        .dataframe {
            font-family: Arial, sans-serif;
        }
        .dataframe a {
            color: #667eea !important;
            text-decoration: none !important;
            font-weight: 500 !important;
        }
        .dataframe a:hover {
            text-decoration: underline !important;
        }
        /* Sleek custom metric styling */
        .custom-metric {
            background: white;
            border: 1px solid #e8eaed;
            border-radius: 10px;
            padding: 1.25rem;
            margin: 0.75rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            text-align: center;
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        .custom-metric::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        .custom-metric:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            transform: translateY(-2px);
            border-color: #d0d7de;
        }
        .custom-metric:hover::before {
            opacity: 1;
        }
        .custom-metric-label {
            font-size: 0.85em;
            color: #656d76;
            margin-bottom: 0.75rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        .custom-metric-value {
            font-size: 1.6em;
            font-weight: 600;
            color: #24292f;
            line-height: 1.2;
        }
        .help-icon {
            color: #6b7280;
            font-size: 0.8em;
            margin-left: 0.25rem;
            cursor: help;
            transition: color 0.2s ease;
            font-weight: normal;
        }
        .help-icon:hover {
            color: #6366f1;
        }
        /* Narrow ownership metrics */
        .narrow-metric {
            background: white;
            border: 1px solid #e8eaed;
            border-radius: 10px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.06);
            text-align: center;
            transition: all 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        .narrow-metric::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #6366f1, #8b5cf6);
            opacity: 0;
            transition: opacity 0.2s ease;
        }
        .narrow-metric:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.12);
            transform: translateY(-2px);
            border-color: #d0d7de;
        }
        .narrow-metric:hover::before {
            opacity: 1;
        }
        .narrow-metric-label {
            font-size: 0.8em;
            color: #656d76;
            margin-bottom: 0.5rem;
            font-weight: 500;
            letter-spacing: 0.02em;
        }
        .narrow-metric-value {
            font-size: 1.4em;
            font-weight: 600;
            color: #24292f;
            line-height: 1.2;
        }
    </style>
""", unsafe_allow_html=True)

def create_custom_metric(label, value, help_text=None, trend=None):
    """Create a custom metric with optional help tooltip and trend arrow (cross-platform safe)"""
    arrow = ''
    if trend == 'up':
        arrow = '<span style="color:green; font-size:1.2em;">↑</span>'
    elif trend == 'down':
        arrow = '<span style="color:red; font-size:1.2em;">↓</span>'
    metric_html = f'''
    <div class="custom-metric">
        <div class="custom-metric-label">{label}</div>
        <div class="custom-metric-value">{value} {arrow}</div>
    '''
    if help_text:
        metric_html += f'<div class="custom-metric-help">{help_text}</div>'
    metric_html += '</div>'
    return metric_html

def create_narrow_metric(label, value):
    """Create a narrower metric for ownership data"""
    return f"""
    <div class="narrow-metric">
        <div class="narrow-metric-label">{label}</div>
        <div class="narrow-metric-value">{value}</div>
    </div>
    """

# Add mobile detection helper
def is_mobile():
    return st.session_state.get('is_mobile', False)

def main():
    st.title("🏢 Affiliated Entities Dashboard")
    st.markdown("**Comprehensive performance metrics for nursing home ownership entities**")
    
    # Load data
    with st.spinner("Loading affiliated entity data..."):
        entity_data = load_affiliated_entity_data()
        provider_data = load_provider_info_data()
    
    if entity_data.empty or provider_data.empty:
        st.error("Unable to load data. Please check that the CSV files are available.")
        return
    
    # Filter out the "National" row (aggregate data)
    entity_data = entity_data[entity_data['Affiliated entity'] != 'National'].copy()
    
    # Search section
    entity_options = []
    for idx, entity in entity_data.iterrows():
        entity_name = capitalize_entity_name(entity['Affiliated entity'])
        facility_count = entity['Number of facilities']
        entity_options.append(f"{entity_name} ({facility_count} facilities)")

    # --- MOBILE: No info box, just dropdown ---
    st.subheader("🔍 Search for an Entity")
    selected_entity_display = st.selectbox(
        "Enter entity name to search:",
        options=[""] + entity_options,
        index=0,
        help="Start typing to search for an entity. Results will show entity name and facility count.",
        label_visibility="visible"
    )
    
    # If no entity is selected, show placeholder
    if not selected_entity_display:
        st.markdown("""
            <div class="no-entity-selected">
                <h3>👆 Search for an Entity</h3>
                <p>Use the search box above to find and analyze a specific nursing home ownership entity.</p>
                <p>You can search by entity name to see detailed performance metrics, facility listings, and quality indicators.</p>
            </div>
        """, unsafe_allow_html=True)
        return
    
    # Extract entity name from selection
    entity_name = capitalize_entity_name(selected_entity_display.split(" (")[0])
    # Robust match: strip, lower, and compare
    filtered = entity_data[entity_data['Affiliated entity'].str.strip().str.lower() == entity_name.strip().lower()]
    if not filtered.empty:
        entity_row = filtered.iloc[0]
        entity_id = int(entity_row['Affiliated entity ID'])
        # Main entity dashboard with entity ID
        st.markdown(f'''
                <div class="entity-header" style="background: linear-gradient(90deg, #e3ecfa 80%, #dbeafe 100%); color: #1a2233; padding: 1.7rem 2rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border: 1px solid #d3dbe8;">
                    <h2 style="margin-bottom: 0.15em; font-size: 2.2em; font-weight: 700; letter-spacing: 0.01em; color: #1a2233;">{entity_name} <span style="font-size: 0.7em; font-weight: 400; color: #4b5563;">(ID: {entity_id})</span></h2>
                    <div style="font-size: 1.13em; color: #234; font-weight: 600; margin-top: 0.1em; text-shadow: 0 1px 4px rgba(255,255,255,0.12);">Nursing Home Affiliated Entity Dashboard</div>
                </div>
            ''', unsafe_allow_html=True)
        # Key metrics overview using custom metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(create_custom_metric("Total Facilities", entity_row['Number of facilities']), unsafe_allow_html=True)
        with col2:
            st.markdown(create_custom_metric("States of Operation", entity_row['Number of states and territories with operations']), unsafe_allow_html=True)
        with col3:
            st.markdown(create_custom_metric("Overall Rating", f"{entity_row['Average overall 5-star rating']:.1f}"), unsafe_allow_html=True)
        with col4:
            st.markdown(create_custom_metric("Total Fines", f"${entity_row['Total amount of fines in dollars']:,.0f}"), unsafe_allow_html=True)
        # High Risk Facilities
        st.markdown(f'<div class="section-header"><h3>🚨 High Risk Facilities - {entity_name}</h3></div>', unsafe_allow_html=True)
        
        # Add native Streamlit tooltip
        st.caption("", help="SFF: Special Focus Facilities with serious problems over time.\n\nSFF Candidate: Facilities monitored for potential SFF designation.\n\nAbuse Icon: Cited for abuse with actual or potential harm.")
        
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        with risk_col1:
            if entity_row['Number of Special Focus Facilities (SFF)'] > 0:
                st.markdown(create_custom_metric("SFF Facilities", entity_row['Number of Special Focus Facilities (SFF)']), unsafe_allow_html=True)
            else:
                st.markdown(create_custom_metric("SFF Facilities", "0"), unsafe_allow_html=True)
        
        with risk_col2:
            if entity_row['Number of SFF candidates'] > 0:
                st.markdown(create_custom_metric("SFF Candidates", entity_row['Number of SFF candidates']), unsafe_allow_html=True)
            else:
                st.markdown(create_custom_metric("SFF Candidates", "0"), unsafe_allow_html=True)
        
        with risk_col3:
            if entity_row['Number of facilities with an abuse icon'] > 0:
                st.markdown(create_custom_metric("Abuse Icons", entity_row['Number of facilities with an abuse icon']), unsafe_allow_html=True)
            else:
                st.markdown(create_custom_metric("Abuse Icons", "0"), unsafe_allow_html=True)
        

        
        # Single column layout for detailed metrics
        
        # Ownership breakdown with pie chart
        st.markdown(f'<div class="section-header"><h3>📊 Ownership Structure - {entity_name}</h3></div>', unsafe_allow_html=True)
        
        own_col1, own_col2 = st.columns([1, 1])
        
        with own_col1:
            # Pie chart for ownership
            fig_pie = go.Figure(data=[go.Pie(
                labels=['For-Profit', 'Non-Profit', 'Government'],
                values=[
                    entity_row['Percent of facilities classified as for-profit'],
                    entity_row['Percent of facilities classified as non-profit'],
                    entity_row['Percent of facilities classified as government-owned']
                ],
                hole=0.3,
                marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1']
            )])
            
            fig_pie.update_layout(
                height=300,
                showlegend=True,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with own_col2:
            # Ownership metrics with narrower containers
            st.markdown(create_narrow_metric("For-Profit", f"{entity_row['Percent of facilities classified as for-profit']:.1f}%"), unsafe_allow_html=True)
            st.markdown(create_narrow_metric("Non-Profit", f"{entity_row['Percent of facilities classified as non-profit']:.1f}%"), unsafe_allow_html=True)
            st.markdown(create_narrow_metric("Government", f"{entity_row['Percent of facilities classified as government-owned']:.1f}%"), unsafe_allow_html=True)
        
        # CMS 5-Star Ratings
        st.markdown(f'<div class="section-header"><h3>⭐ CMS 5-Star Ratings - {entity_name}</h3></div>', unsafe_allow_html=True)
        
        # Quality metrics with decimals for entity averages
        qual_col1, qual_col2, qual_col3, qual_col4 = st.columns(4)
        with qual_col1:
            st.markdown(create_custom_metric("Overall Rating", f"{entity_row['Average overall 5-star rating']:.1f}"), unsafe_allow_html=True)
        with qual_col2:
            st.markdown(create_custom_metric("Health Inspection", f"{entity_row['Average health inspection rating']:.1f}"), unsafe_allow_html=True)
        with qual_col3:
            st.markdown(create_custom_metric("Staffing Rating", f"{entity_row['Average staffing rating']:.1f}"), unsafe_allow_html=True)
        with qual_col4:
            st.markdown(create_custom_metric("Quality Rating", f"{entity_row['Average quality rating']:.1f}"), unsafe_allow_html=True)
        
        # Quality ratings chart
        fig = go.Figure()
        
        metrics = ['Overall', 'Health Inspection', 'Staffing', 'Quality']
        values = [
            entity_row['Average overall 5-star rating'],
            entity_row['Average health inspection rating'],
            entity_row['Average staffing rating'],
            entity_row['Average quality rating']
        ]
        
        colors = ['#667eea', '#764ba2', '#f093fb', '#f5576c']
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f'{v:.1f}' for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            yaxis_title="Rating",
            yaxis=dict(range=[0, 5]),
            height=300,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Facility ratings breakdown pie chart
        if entity_id and entity_id != "":
            entity_facilities = provider_data[
                provider_data['Affiliated Entity ID'] == entity_id
            ].copy()
            
            if not entity_facilities.empty:
                # Count facilities by overall rating and ensure all ratings 1-5 are included
                rating_counts = entity_facilities['Overall Rating'].value_counts()
                
                # Create a complete series with all ratings 1-5, filling missing ones with 0
                complete_ratings = pd.Series(index=range(1, 6), data=0)
                for rating, count in rating_counts.items():
                    if pd.notna(rating) and rating in range(1, 6):
                        complete_ratings[rating] = count
                
                # Create pie chart for facility ratings with red to blue scale, sorted 1-5
                fig_ratings = go.Figure(data=[go.Pie(
                    labels=[f"{rating}" for rating in complete_ratings.index],
                    values=complete_ratings.values,
                    hole=0.3,
                    marker_colors=['#ff0000', '#ff6b6b', '#ffa726', '#4caf50', '#2196f3'],  # Red to blue scale
                    sort=False  # Keep the order as specified
                )])
                
                fig_ratings.update_layout(
                    height=400,
                    showlegend=True,
                    margin=dict(l=20, r=20, t=40, b=20),
                    title="Distribution of Facility Overall Ratings"
                )
                
                st.plotly_chart(fig_ratings, use_container_width=True)
        
        # Staffing metrics
        st.markdown(f'<div class="section-header"><h3>👥 Staffing Performance - {entity_name}</h3></div>', unsafe_allow_html=True)
        
        staff_col1, staff_col2, staff_col3, staff_col4 = st.columns(4)
        with staff_col1:
            st.markdown(create_custom_metric("Total Nurse HPRD", f"{entity_row['Average total nurse hours per resident day']:.1f}"), unsafe_allow_html=True)
        with staff_col2:
            st.markdown(create_custom_metric("RN HPRD", f"{entity_row['Average total Registered Nurse hours per resident day']:.1f}"), unsafe_allow_html=True)
        with staff_col3:
            st.markdown(create_custom_metric("Weekend HPRD", f"{entity_row['Average total weekend nurse hours per resident day']:.1f}"), unsafe_allow_html=True)
        with staff_col4:
            st.markdown(create_custom_metric("Admin Turnover", f"{entity_row['Average number of administrators who have left the nursing home']:.1f}"), unsafe_allow_html=True)
        
        # Turnover metrics
        turn_col1, turn_col2 = st.columns(2)
        with turn_col1:
            st.markdown(create_custom_metric("Nursing Staff Turnover", f"{entity_row['Average total nursing staff turnover percentage']:.1f}%"), unsafe_allow_html=True)
        with turn_col2:
            st.markdown(create_custom_metric("RN Turnover", f"{entity_row['Average Registered Nurse turnover percentage']:.1f}%"), unsafe_allow_html=True)
        
        # Compliance metrics
        st.markdown(f'<div class="section-header"><h3>⚠️ Compliance & Financial - {entity_name}</h3></div>', unsafe_allow_html=True)
        
        comp_col1, comp_col2, comp_col3, comp_col4 = st.columns(4)
        with comp_col1:
            st.markdown(create_custom_metric("Total Fines", f"${entity_row['Total amount of fines in dollars']:,.0f}"), unsafe_allow_html=True)
        with comp_col2:
            st.markdown(create_custom_metric("Avg Fines per Facility", f"${entity_row['Average amount of fines in dollars']:,.0f}"), unsafe_allow_html=True)
        with comp_col3:
            st.markdown(create_custom_metric("Total Payment Denials", entity_row['Total number of payment denials']), unsafe_allow_html=True)
        with comp_col4:
            st.markdown(create_custom_metric("Avg Payment Denials", f"{entity_row['Average number of payment denials']:.1f}"), unsafe_allow_html=True)
        
        # Antipsychotic usage
        st.markdown(f'<div class="section-header"><h3>💊 Antipsychotic Usage - {entity_name}</h3></div>', unsafe_allow_html=True)
        
        anti_col1, anti_col2 = st.columns(2)
        with anti_col1:
            st.markdown(create_custom_metric("Short-Stay Antipsychotic", f"{entity_row['Average percentage of short-stay residents who newly received an antipsychotic medication']:.1f}%"), unsafe_allow_html=True)
        with anti_col2:
            st.markdown(create_custom_metric("Long-Stay Antipsychotic", f"{entity_row['Average percentage of long-stay residents who received an antipsychotic medication']:.1f}%"), unsafe_allow_html=True)
        
        # Facilities list
        st.markdown(f'<div class="section-header"><h3>🏥 {entity_name} Facilities</h3></div>', unsafe_allow_html=True)
        
        # Get facilities for this entity
        if entity_id and entity_id != "":
            entity_facilities = provider_data[
                provider_data['Affiliated Entity ID'] == entity_id
            ].copy()
            
            if not entity_facilities.empty:
                st.markdown(f"**{len(entity_facilities)} facilities found for {entity_name}**")
                
                # Prepare facilities data for display with City instead of County
                facilities_display = entity_facilities[[
                    'State',
                    'City/Town',
                    'CMS Certification Number (CCN)',
                    'Provider Name',
                    'Overall Rating',
                    'Health Inspection Rating',
                    'Staffing Rating',
                    'QM Rating',
                    'Special Focus Status',
                    'Abuse Icon'
                ]].copy()
                
                # Rename City/Town to City
                facilities_display = facilities_display.rename(columns={'City/Town': 'City'})
                
                # Clean up the data and convert ratings to integers
                facilities_display = facilities_display.fillna('N/A')
                
                # Convert numeric ratings to integers where possible
                rating_columns = ['Overall Rating', 'Health Inspection Rating', 'Staffing Rating', 'QM Rating']
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
                    "Show High-Risk Facilities Only", 
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
                        st.success(f"Showing {len(facilities_display)} high-risk facilities out of {total_facilities} total facilities.")
                
                # Create provider names as HTML links
                facilities_display['Provider Name'] = facilities_display.apply(
                    lambda row: f'<a href="https://pbjlite.streamlit.app/?level=Facility&facility={row["CMS Certification Number (CCN)"]}" target="_blank">{row["Provider Name"]}</a>',
                    axis=1
                )
                
                # Reorder columns to: State, Provider Name, CMS CCN, City, etc.
                column_order = [
                    'State',
                    'Provider Name',
                    'CMS Certification Number (CCN)',
                    'City',
                    'Overall Rating',
                    'Health Inspection Rating',
                    'Staffing Rating',
                    'QM Rating',
                    'Special Focus Status',
                    'Abuse Icon'
                ]
                facilities_display = facilities_display[column_order]
                
                # Render as HTML table for clickable links
                html_table = facilities_display.to_html(
                    index=False,
                    escape=False,
                    classes=['dataframe', 'table', 'table-striped'],
                    table_id='facilities-table'
                )
                
                # Add CSS and JavaScript for table styling and sorting
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
                
                st.markdown('</div>', unsafe_allow_html=True)
                
            else:
                st.info("No facility data available for this entity.")
        else:
            st.info("Entity ID not available for facility lookup.")
        
        # Full data table (collapsible) - Fixed to show variable names in two columns
        with st.expander("📋 View Complete Entity Data", expanded=False):
            # Convert the series to a dataframe with proper column names
            entity_df = pd.DataFrame({
                'Variable': entity_row.index,
                'Value': entity_row.values
            })
            
            # Display in two columns
            col1, col2 = st.columns(2)
            
            # Split the data into two halves
            mid_point = len(entity_df) // 2
            left_data = entity_df.iloc[:mid_point]
            right_data = entity_df.iloc[mid_point:]
            
            with col1:
                st.dataframe(left_data, use_container_width=True, hide_index=True)
            
            with col2:
                st.dataframe(right_data, use_container_width=True, hide_index=True)
    else:
        st.warning(f"No data found for the selected entity: {entity_name}")
        return

    # Add source note at the bottom
    st.markdown(
        '<div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.95em;">'
        'Source: <a href="https://data.cms.gov/quality-of-care/nursing-home-affiliated-entity-performance-measures/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">CMS Nursing Home Affiliated Entity Performance Measures (June 2025)</a>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown('''
<div style="text-align:center; margin-top:2em; padding:1em; background:#f5f8fd; border-radius:8px; font-size:1.05em; color:#333;">
  A free public resource from <b>320 Consulting</b>.<br>
  <a href="/pages/2_About.py" style="color:#1E88E5; text-decoration:underline; font-weight:500;">About the PBJ Dashboard</a>
</div>
''', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 