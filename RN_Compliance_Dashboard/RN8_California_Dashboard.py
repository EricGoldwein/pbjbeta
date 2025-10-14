import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
from datetime import datetime, timedelta
import base64
import calendar

# Set page configuration
st.set_page_config(
    page_title="RN 8 California Compliance Dashboard | PBJ Nursing Home Staffing by 320",
    page_icon="pbj_favicon.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #f5f8fd 0%, #e3f2fd 100%);
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-card {
        background: white;
        padding: 1.2rem;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 3px solid #1976d2;
        margin-bottom: 0.8rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1976d2;
        margin-bottom: 0.3rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        font-weight: 500;
    }
    .facility-card {
        background: white;
        padding: 0.8rem;
        border-radius: 6px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        border: 1px solid #e0e0e0;
        margin-bottom: 0.4rem;
    }
    .facility-name {
        font-weight: 600;
        color: #1976d2;
        margin-bottom: 0.3rem;
    }
    .facility-stats {
        display: flex;
        gap: 0.8rem;
        font-size: 0.85rem;
        color: #666;
    }
    .high-risk {
        background: #ffebee;
        border-left-color: #f44336;
    }
    .medium-risk {
        background: #fff3e0;
        border-left-color: #ff9800;
    }
    .low-risk {
        background: #e8f5e8;
        border-left-color: #4caf50;
    }
    .compliance-alert {
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 6px;
        padding: 0.8rem;
        margin: 0.8rem 0;
    }
    .success-card {
        background: #e8f5e8;
        border-left-color: #4caf50;
    }
    .warning-card {
        background: #fff3e0;
        border-left-color: #ff9800;
    }
    .critical-card {
        background: #ffebee;
        border-left-color: #f44336;
    }
    .tooltip {
        position: relative;
        display: inline-block;
        color: #1976d2;
        cursor: help;
    }
    .export-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_california_metrics():
    """Load pre-calculated California RN 8 metrics."""
    try:
        metrics_file = "rn8_california_metrics.csv"
        quarterly_file = "rn8_california_quarterly_metrics.csv"
        
        if not os.path.exists(metrics_file):
            st.error(f"""
            **Pre-calculated metrics file not found: {metrics_file}**
            
            Please run the calculation script first:
            ```
            python calculate_rn8_california.py
            ```
            
            This will process all PBJ files and create the metrics file.
            """)
            return pd.DataFrame(), pd.DataFrame()
        
        facility_metrics = pd.read_csv(metrics_file, dtype={'PROVNUM': str})
        quarterly_metrics = pd.read_csv(quarterly_file, dtype={'PROVNUM': str}) if os.path.exists(quarterly_file) else pd.DataFrame()
        
        return facility_metrics, quarterly_metrics
        
    except Exception as e:
        st.error(f"Error loading California metrics: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def load_provider_info():
    """Load provider information for additional context."""
    try:
        provider_files = glob.glob("NH_ProviderInfo_*.csv")
        if provider_files:
            latest_file = max(provider_files)
            provider_data = pd.read_csv(latest_file, dtype={'CMS Certification Number (CCN)': str})
            return provider_data
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

def format_number(num):
    """Format numbers with thousands separators."""
    if pd.isna(num):
        return "—"
    return f"{num:,.0f}"

def format_percentage(num):
    """Format percentages with one decimal place."""
    if pd.isna(num):
        return "—"
    return f"{num:.1f}%"

def proper_title_case(text):
    """Convert text to proper title case with lowercase prepositions."""
    if pd.isna(text):
        return text
    
    lowercase_words = {'and', 'or', 'of', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by'}
    words = text.lower().split()
    formatted_words = []
    
    for i, word in enumerate(words):
        if i == 0 or word not in lowercase_words:
            formatted_words.append(word.capitalize())
        else:
            formatted_words.append(word)
    
    return ' '.join(formatted_words)

def format_quarter_display(quarter):
    """Convert quarter format from 'CY2017Q1' to 'Q1 2017'."""
    if pd.isna(quarter):
        return quarter
    
    quarter_str = str(quarter)
    
    # Handle CY2017Q1 format
    if quarter_str.startswith('CY') and 'Q' in quarter_str:
        year = quarter_str[2:6]  # Extract year from CY2017Q1
        q_num = quarter_str[-1]  # Extract quarter number
        return f"Q{q_num} {year}"
    # Handle 2017Q1 format
    elif 'Q' in quarter_str and not quarter_str.startswith('Q'):
        year = quarter_str[:4]
        q_num = quarter_str[-1]
        return f"Q{q_num} {year}"
    # Already in Q1 2017 format
    elif quarter_str.startswith('Q'):
        return quarter_str
    else:
        # Fallback
        return quarter_str

def calculate_compliance_metrics(facility_metrics):
    """Calculate high-level compliance metrics."""
    total_facilities = len(facility_metrics)
    facilities_with_violations = len(facility_metrics[facility_metrics['Days_RN_Less_8'] > 0])
    pct_with_violations = (facilities_with_violations / total_facilities * 100) if total_facilities > 0 else 0
    total_violation_days = facility_metrics['Days_RN_Less_8'].sum()
    
    # Perfect compliance facilities
    perfect_compliance = len(facility_metrics[facility_metrics['Days_RN_Less_8'] == 0])
    pct_perfect = (perfect_compliance / total_facilities * 100) if total_facilities > 0 else 0
    
    return {
        'total_facilities': total_facilities,
        'facilities_with_violations': facilities_with_violations,
        'pct_with_violations': pct_with_violations,
        'total_violation_days': total_violation_days,
        'perfect_compliance': perfect_compliance,
        'pct_perfect': pct_perfect
    }

def create_facility_table(facility_metrics, provider_data):
    """Create detailed facility table with additional context."""
    table_data = facility_metrics.copy()
    
    # Add provider info if available
    if not provider_data.empty:
        table_data = table_data.merge(
            provider_data[['CMS Certification Number (CCN)', 'Provider Name', 'Number of Certified Beds', 'Ownership Type']],
            left_on='PROVNUM',
            right_on='CMS Certification Number (CCN)',
            how='left'
        )
    
    # Calculate additional metrics
    table_data['Violation_Rate'] = (table_data['Days_RN_Less_8'] / table_data['Total_Days'] * 100).round(1)
    table_data['RN_HPRD_Avg'] = table_data['Avg_RN_HPRD'].round(2)
    
    # Sort by violation days (descending)
    table_data = table_data.sort_values('Days_RN_Less_8', ascending=False)
    
    return table_data

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="text-align: center; color: #1976d2; margin-bottom: 0.3rem; font-size: 1.5rem;">
            RN 8 California Compliance Dashboard
        </h1>
        <p style="text-align: center; color: #666; font-size: 0.8rem; margin-bottom: 0.2rem;">
            Federal RN Coverage Requirement Analysis for Legal & Media Use
        </p>
        <p style="text-align: center; color: #666; font-size: 0.7rem;">
            by 320 Consulting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading California RN 8 compliance data..."):
        facility_metrics, quarterly_metrics = load_california_metrics()
        provider_data = load_provider_info()
    
    if facility_metrics.empty:
        st.error("Unable to load compliance data. Please run the calculation script first.")
        return
    
    # Quarter selection and data filtering
    selected_quarter = "All Quarters"  # Default value
    
    if not quarterly_metrics.empty and 'CY_QTR' in quarterly_metrics.columns:
        available_quarters = sorted(quarterly_metrics['CY_QTR'].unique())
        quarter_options = ["All Quarters"] + [format_quarter_display(q) for q in available_quarters]
        
        selected_quarter = st.selectbox(
            "Select Quarter:",
            quarter_options,
            index=0,
            help="Choose a specific quarter or 'All Quarters' for complete analysis"
        )
        
        # Filter data based on selection
        if selected_quarter != "All Quarters":
            original_quarter = None
            for q in available_quarters:
                if format_quarter_display(q) == selected_quarter:
                    original_quarter = q
                    break
            facility_metrics_filtered = quarterly_metrics[quarterly_metrics['CY_QTR'] == original_quarter].copy()
            analysis_info = f"📊 {selected_quarter}"
        else:
            facility_metrics_filtered = facility_metrics.copy()
            analysis_info = "📊 All Available Quarters"
    else:
        facility_metrics_filtered = facility_metrics.copy()
        analysis_info = "📊 All Available Data (Aggregated)"
    
    # Calculate compliance metrics
    compliance_metrics = calculate_compliance_metrics(facility_metrics_filtered)
    
    # High-level overview with consolidated info
    st.subheader("📊 California RN 8 Compliance Overview")
    
    # Show analysis info and data coverage in one place
    if not quarterly_metrics.empty and 'CY_QTR' in quarterly_metrics.columns:
        available_quarters = sorted(quarterly_metrics['CY_QTR'].unique())
        if selected_quarter != "All Quarters":
            # Show specific quarter info
            st.info(f"{analysis_info} • Single quarter analysis")
        else:
            # Show full range info
            date_range = f"{format_quarter_display(available_quarters[0])} to {format_quarter_display(available_quarters[-1])}"
            st.info(f"{analysis_info} • {len(available_quarters)} quarters: {date_range}")
    else:
        st.info(f"{analysis_info} • Aggregated data (2017-2025)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{format_number(compliance_metrics['total_facilities'])}</div>
            <div class="metric-label">Total California Facilities</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card warning-card">
            <div class="metric-value">{format_percentage(compliance_metrics['pct_with_violations'])}</div>
            <div class="metric-label">Facilities with RN < 8 Hours</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card critical-card">
            <div class="metric-value">{format_number(compliance_metrics['total_violation_days'])}</div>
            <div class="metric-label">Total Sub-8 Days Statewide</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card success-card">
            <div class="metric-value">{format_percentage(compliance_metrics['pct_perfect'])}</div>
            <div class="metric-label">Perfect Compliance Rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Facility Search
    st.subheader("🔍 Search Individual Facility")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Facility search
        search_option = st.selectbox(
            "Search by:",
            ["Facility Name", "CCN Number", "City", "County"],
            index=0
        )
        
        if search_option == "Facility Name":
            facility_names = sorted(facility_metrics_filtered['PROVNAME'].unique())
            facility_names_formatted = [proper_title_case(name) for name in facility_names]
            selected_facility = st.selectbox("Select Facility:", [""] + facility_names_formatted)
            if selected_facility:
                # Find the original name
                for original_name in facility_names:
                    if proper_title_case(original_name) == selected_facility:
                        selected_facility = original_name
                        break
        elif search_option == "CCN Number":
            ccn_numbers = sorted(facility_metrics_filtered['PROVNUM'].unique())
            selected_facility = st.selectbox("Select CCN:", [""] + ccn_numbers)
        elif search_option == "City":
            cities = sorted(facility_metrics_filtered['CITY'].unique())
            cities_formatted = [proper_title_case(city) for city in cities]
            selected_city = st.selectbox("Select City:", [""] + cities_formatted)
            if selected_city:
                # Find the original city name
                for original_city in cities:
                    if proper_title_case(original_city) == selected_city:
                        selected_city = original_city
                        break
                city_facilities = facility_metrics_filtered[facility_metrics_filtered['CITY'] == selected_city]
                facility_names = sorted(city_facilities['PROVNAME'].unique())
                facility_names_formatted = [proper_title_case(name) for name in facility_names]
                selected_facility = st.selectbox("Select Facility:", [""] + facility_names_formatted)
                if selected_facility:
                    # Find the original facility name
                    for original_name in facility_names:
                        if proper_title_case(original_name) == selected_facility:
                            selected_facility = original_name
                            break
            else:
                selected_facility = ""
        else:  # County
            counties = sorted(facility_metrics_filtered['COUNTY_NAME'].unique())
            counties_formatted = [proper_title_case(county) for county in counties]
            selected_county = st.selectbox("Select County:", [""] + counties_formatted)
            if selected_county:
                # Find the original county name
                for original_county in counties:
                    if proper_title_case(original_county) == selected_county:
                        selected_county = original_county
                        break
                county_facilities = facility_metrics_filtered[facility_metrics_filtered['COUNTY_NAME'] == selected_county]
                facility_names = sorted(county_facilities['PROVNAME'].unique())
                facility_names_formatted = [proper_title_case(name) for name in facility_names]
                selected_facility = st.selectbox("Select Facility:", [""] + facility_names_formatted)
                if selected_facility:
                    # Find the original facility name
                    for original_name in facility_names:
                        if proper_title_case(original_name) == selected_facility:
                            selected_facility = original_name
                            break
            else:
                selected_facility = ""
    
    with col2:
        # Quick filters
        st.markdown("**Quick Filters:**")
        show_high_risk = st.checkbox("Show High Risk Only", value=False)
        show_violations = st.checkbox("Show Violations Only", value=False)
    
    # Display selected facility details
    if selected_facility:
        if search_option == "CCN Number":
            facility_data = facility_metrics_filtered[facility_metrics_filtered['PROVNUM'] == selected_facility]
        else:
            facility_data = facility_metrics_filtered[facility_metrics_filtered['PROVNAME'] == selected_facility]
        
        if not facility_data.empty:
            facility = facility_data.iloc[0]
            
            # Facility header
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #f5f8fd 0%, #e3f2fd 100%); padding: 1.5rem; border-radius: 8px; margin: 1rem 0;">
                <h2 style="color: #1976d2; margin-bottom: 0.5rem;">{proper_title_case(facility['PROVNAME'])}</h2>
                <p style="color: #666; margin: 0;">{facility['CITY']}, {facility['COUNTY_NAME']} • CCN: {facility['PROVNUM']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{format_number(facility['Total_Days'])}</div>
                    <div class="metric-label">Total Days Analyzed</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card {'critical-card' if facility['Days_RN_Less_8'] > 0 else 'success-card'}">
                    <div class="metric-value">{format_number(facility['Days_RN_Less_8'])}</div>
                    <div class="metric-label">RN < 8 Hours Days</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{format_percentage(facility['Pct_RN_Less_8'])}</div>
                    <div class="metric-label">Violation Rate</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{facility['Risk_Level']}</div>
                    <div class="metric-label">Risk Level</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Additional metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{format_number(facility.get('Census', 0))}</div>
                    <div class="metric-label">Average Census</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{facility['Avg_RN_HPRD']:.2f}</div>
                    <div class="metric-label">Avg RN HPRD</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{facility.get('Avg_Total_HPRD', 0):.2f}</div>
                    <div class="metric-label">Avg Total Nurse HPRD</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{format_number(facility.get('Days_Sub_350', 0))}</div>
                    <div class="metric-label">Days Below 3.50 HPRD</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Charts row
            col1, col2 = st.columns(2)
            
            with col1:
                # RN HPRD gauge with realistic range
                fig_rn = go.Figure()
                fig_rn.add_trace(go.Indicator(
                    mode="gauge+number",
                    value=facility['Avg_RN_HPRD'],
                    title={'text': "Average RN HPRD"},
                    gauge={'axis': {'range': [0, 2]},
                           'bar': {'color': "#1976d2"},
                           'steps': [{'range': [0, 0.5], 'color': "#ffebee"},
                                    {'range': [0.5, 1.0], 'color': "#fff3e0"},
                                    {'range': [1.0, 2.0], 'color': "#e8f5e8"}]}
                ))
                fig_rn.update_layout(height=300)
                st.plotly_chart(fig_rn, use_container_width=True)
            
            with col2:
                # Compliance breakdown
                compliance_data = pd.DataFrame({
                    'Category': ['Compliant Days', 'RN < 8 Hours'],
                    'Days': [facility['Total_Days'] - facility['Days_RN_Less_8'], facility['Days_RN_Less_8']]
                })
                
                fig_pie = px.pie(compliance_data, values='Days', names='Category',
                               title="RN Coverage Compliance Breakdown",
                               color_discrete_map={'Compliant Days': '#4caf50', 'RN < 8 Hours': '#f44336'})
                fig_pie.update_layout(height=300)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # RN 8 Takeaway Card
            st.markdown("""
            <div style="background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 4px solid #1976d2;">
                <h3 style="color: #1976d2; margin-bottom: 1rem; font-size: 1.2rem;">📊 RN 8 California Takeaway</h3>
            """, unsafe_allow_html=True)
            
            # Calculate state average for comparison
            state_avg_rn_less_8 = facility_metrics_filtered['Pct_RN_Less_8'].mean()
            state_avg_rn_hprd = facility_metrics_filtered['Avg_RN_HPRD'].mean()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📈 RN < 8 Hours Comparison:**")
                facility_pct = facility['Pct_RN_Less_8']
                state_pct = state_avg_rn_less_8
                
                if facility_pct > state_pct:
                    comparison_text = f"**{format_percentage(facility_pct)}** vs State Average: {format_percentage(state_pct)}"
                    st.markdown(f"<div style='color: #f44336; font-weight: bold; font-size: 1.1rem; margin: 0.5rem 0;'>{comparison_text}</div>", unsafe_allow_html=True)
                    st.markdown("⚠️ **Above state average** - More violations than typical California facility")
                else:
                    comparison_text = f"**{format_percentage(facility_pct)}** vs State Average: {format_percentage(state_pct)}"
                    st.markdown(f"<div style='color: #4caf50; font-weight: bold; font-size: 1.1rem; margin: 0.5rem 0;'>{comparison_text}</div>", unsafe_allow_html=True)
                    st.markdown("✅ **Below state average** - Fewer violations than typical California facility")
            
            with col2:
                st.markdown("**📊 RN HPRD Comparison:**")
                facility_hprd = facility['Avg_RN_HPRD']
                state_hprd = state_avg_rn_hprd
                
                if facility_hprd < state_hprd:
                    comparison_text = f"**{facility_hprd:.2f}** vs State Average: {state_hprd:.2f}"
                    st.markdown(f"<div style='color: #f44336; font-weight: bold; font-size: 1.1rem; margin: 0.5rem 0;'>{comparison_text}</div>", unsafe_allow_html=True)
                    st.markdown("⚠️ **Below state average** - Lower RN staffing than typical California facility")
                else:
                    comparison_text = f"**{facility_hprd:.2f}** vs State Average: {state_hprd:.2f}"
                    st.markdown(f"<div style='color: #4caf50; font-weight: bold; font-size: 1.1rem; margin: 0.5rem 0;'>{comparison_text}</div>", unsafe_allow_html=True)
                    st.markdown("✅ **Above state average** - Higher RN staffing than typical California facility")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Creative Analysis Section
            st.markdown("**🔍 Pattern Analysis:**")
            
            # Generate creative insights based on the data
            violation_rate = facility['Pct_RN_Less_8']
            rn_hprd = facility['Avg_RN_HPRD']
            total_days = facility['Total_Days']
            violation_days = facility['Days_RN_Less_8']
            
            insights = []
            
            if violation_rate > 50:
                insights.append(f"🚨 **Chronic Understaffing Pattern:** This facility violated federal RN coverage requirements on {format_percentage(violation_rate)} of analyzed days - a systematic failure that suggests ongoing resource allocation issues.")
            elif violation_rate > 25:
                insights.append(f"⚠️ **Intermittent Compliance Issues:** With {format_percentage(violation_rate)} violation rate, this facility shows periodic staffing challenges, possibly related to scheduling gaps or seasonal fluctuations.")
            elif violation_rate > 0:
                insights.append(f"📊 **Occasional Violations:** {format_percentage(violation_rate)} violation rate indicates isolated incidents, possibly due to unexpected absences or emergency situations.")
            else:
                insights.append(f"✅ **Perfect Compliance:** Zero violations across {format_number(total_days)} days analyzed - exemplary adherence to federal RN coverage standards.")
            
            if rn_hprd < 0.5:
                insights.append(f"📉 **Critically Low RN Staffing:** Average RN HPRD of {rn_hprd:.2f} is well below recommended levels, indicating severe understaffing that could compromise resident care quality.")
            elif rn_hprd < 1.0:
                insights.append(f"📊 **Below-Average RN Coverage:** RN HPRD of {rn_hprd:.2f} suggests staffing levels that may struggle to meet complex resident care needs.")
            else:
                insights.append(f"📈 **Strong RN Staffing:** RN HPRD of {rn_hprd:.2f} indicates adequate RN coverage for resident care requirements.")
            
            if violation_days > 30:
                insights.append(f"⏰ **Extended Non-Compliance:** {format_number(violation_days)} violation days represents over a month of federal requirement violations - a concerning pattern that may indicate systemic management issues.")
            elif violation_days > 10:
                insights.append(f"📅 **Significant Violation Period:** {format_number(violation_days)} days of violations suggests recurring staffing challenges that warrant management attention.")
            elif violation_days > 0:
                insights.append(f"📋 **Limited Violation Period:** {format_number(violation_days)} violation days indicates isolated incidents rather than systemic problems.")
            
            # Add insights about time patterns if we have quarter data
            if not quarterly_metrics.empty and 'CY_QTR' in quarterly_metrics.columns:
                facility_quarter_data = quarterly_metrics[quarterly_metrics['PROVNUM'] == facility['PROVNUM']]
                if not facility_quarter_data.empty:
                    # Add detailed quarterly breakdown
                    st.markdown("**📅 Quarterly Violation Breakdown:**")
                    quarter_breakdown = facility_quarter_data[['CY_QTR', 'Days_RN_Less_8', 'Pct_RN_Less_8']].copy()
                    quarter_breakdown = quarter_breakdown.sort_values('Days_RN_Less_8', ascending=False)
                    quarter_breakdown['Quarter_Display'] = quarter_breakdown['CY_QTR'].apply(format_quarter_display)
                    quarter_breakdown = quarter_breakdown[['Quarter_Display', 'Days_RN_Less_8', 'Pct_RN_Less_8']]
                    quarter_breakdown.columns = ['Quarter', 'Violation Days', 'Violation Rate (%)']
                    quarter_breakdown['Violation Rate (%)'] = quarter_breakdown['Violation Rate (%)'].apply(lambda x: f"{x:.1f}%")
                    st.dataframe(quarter_breakdown, use_container_width=True)
                    
                    # Facility-Specific Temporal Analysis
                    st.markdown("**📊 Facility-Specific Temporal Analysis: When This Facility Had RN Violations**")
                    
                    # Prepare facility quarterly data for charts
                    facility_quarterly_charts = facility_quarter_data.copy()
                    facility_quarterly_charts['Quarter_Display'] = facility_quarterly_charts['CY_QTR'].apply(format_quarter_display)
                    facility_quarterly_charts = facility_quarterly_charts.sort_values('CY_QTR')
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Facility violation days by quarter
                        fig_facility_quarterly = px.line(
                            facility_quarterly_charts,
                            x='Quarter_Display',
                            y='Days_RN_Less_8',
                            title=f"RN < 8 Hours Violations by Quarter - {proper_title_case(facility['PROVNAME'])}",
                            labels={'Days_RN_Less_8': 'Violation Days', 'Quarter_Display': 'Quarter'},
                            markers=True
                        )
                        fig_facility_quarterly.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_facility_quarterly, use_container_width=True)
                    
                    with col2:
                        # Facility violation rate by quarter
                        fig_facility_rate = px.bar(
                            facility_quarterly_charts,
                            x='Quarter_Display',
                            y='Pct_RN_Less_8',
                            title=f"Violation Rate by Quarter - {proper_title_case(facility['PROVNAME'])}",
                            labels={'Pct_RN_Less_8': 'Violation Rate (%)', 'Quarter_Display': 'Quarter'},
                            color='Pct_RN_Less_8',
                            color_continuous_scale='Reds'
                        )
                        fig_facility_rate.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig_facility_rate, use_container_width=True)
                    
                    # Facility quarterly summary metrics
                    facility_worst_quarter = facility_quarterly_charts.loc[facility_quarterly_charts['Days_RN_Less_8'].idxmax()]
                    facility_best_quarter = facility_quarterly_charts.loc[facility_quarterly_charts['Days_RN_Less_8'].idxmin()]
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card critical-card">
                            <div class="metric-value">{facility_worst_quarter['Quarter_Display']}</div>
                            <div class="metric-label">Worst Quarter: {facility_worst_quarter['Days_RN_Less_8']} violation days ({facility_worst_quarter['Pct_RN_Less_8']:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card success-card">
                            <div class="metric-value">{facility_best_quarter['Quarter_Display']}</div>
                            <div class="metric-label">Best Quarter: {facility_best_quarter['Days_RN_Less_8']} violation days ({facility_best_quarter['Pct_RN_Less_8']:.1f}%)</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Facility trend analysis
                    if len(facility_quarterly_charts) > 1:
                        st.markdown("**📈 Facility Trend Analysis:**")
                        recent_quarters = facility_quarterly_charts.tail(4)  # Last 4 quarters
                        if len(recent_quarters) > 1:
                            recent_trend = recent_quarters['Days_RN_Less_8'].tolist()
                            trend_direction = "📈 Increasing" if recent_trend[-1] > recent_trend[0] else "📉 Decreasing" if recent_trend[-1] < recent_trend[0] else "➡️ Stable"
                            st.markdown(f"• **Recent Trend:** {trend_direction}")
                            st.markdown(f"• **Latest quarter:** {recent_trend[-1]} violation days")
                            st.markdown(f"• **Previous quarter:** {recent_trend[-2]} violation days")
                            
                            # Calculate improvement/decline
                            if recent_trend[-1] != recent_trend[-2]:
                                change = recent_trend[-1] - recent_trend[-2]
                                change_text = f"{abs(change)} {'more' if change > 0 else 'fewer'} violation days"
                                st.markdown(f"• **Quarter-over-quarter change:** {change_text}")
                    
                    # Add seasonal pattern insights if multiple quarters
                    if len(facility_quarter_data) > 1:
                        quarter_violations = facility_quarter_data.groupby('CY_QTR')['Days_RN_Less_8'].sum()
                        if len(quarter_violations) > 1 and quarter_violations.max() > 0:
                            worst_quarter = quarter_violations.idxmax()
                            best_quarter = quarter_violations.idxmin()
                            insights.append(f"📊 **Seasonal Pattern:** Highest violations occurred in {format_quarter_display(worst_quarter)} ({quarter_violations.max()} days), while {format_quarter_display(best_quarter)} had the lowest violations ({quarter_violations.min()} days).")
            
            # Display insights
            for insight in insights:
                st.markdown(f"• {insight}")
            
            # Legal implications
            if facility['Days_RN_Less_8'] > 0:
                st.markdown("**⚖️ Legal Implications:**")
                st.markdown(f"""
                - **Federal Violations:** {format_number(facility['Days_RN_Less_8'])} days below 8-hour RN coverage requirement
                - **Violation Rate:** {format_percentage(facility['Pct_RN_Less_8'])} of analyzed days had insufficient RN coverage
                - **Risk Level:** {facility['Risk_Level']} risk facility - {'High risk may support punitive damages' if facility['Risk_Level'] == 'High' else 'Moderate risk may support negligence claims' if facility['Risk_Level'] == 'Medium' else 'Low risk - minimal legal exposure'}
                """)
    
    # Facility list with filters - moved to separate expandable section
    with st.expander("📋 All California Facilities Table", expanded=False):
        st.subheader("📋 All California Facilities")
        
        # Apply quick filters
        filtered_data = facility_metrics_filtered.copy()
        
        if show_high_risk:
            filtered_data = filtered_data[filtered_data['Risk_Level'] == 'High']
        
        if show_violations:
            filtered_data = filtered_data[filtered_data['Days_RN_Less_8'] > 0]
        
        # Sort by violations (descending)
        filtered_data = filtered_data.sort_values('Days_RN_Less_8', ascending=False)
         
        # Limit to top 20 facilities to save space
        filtered_data = filtered_data.head(20)
        
        # Create detailed table
        table_data = create_facility_table(filtered_data, provider_data)
        
        # Display key columns
        display_columns = ['PROVNAME', 'CITY', 'COUNTY_NAME', 'Days_RN_Less_8', 'Violation_Rate', 'RN_HPRD_Avg', 'Risk_Level']
        if not provider_data.empty:
            display_columns.extend(['Number of Certified Beds', 'Ownership Type'])
        
        # Format table for display
        display_table = table_data[display_columns].copy()
        
        # Apply proper title case formatting
        display_table['PROVNAME'] = display_table['PROVNAME'].apply(proper_title_case)
        display_table['CITY'] = display_table['CITY'].apply(proper_title_case)
        display_table['COUNTY_NAME'] = display_table['COUNTY_NAME'].apply(proper_title_case)
        
        # Format numbers with thousands separators
        display_table['Days_RN_Less_8'] = display_table['Days_RN_Less_8'].apply(lambda x: f"{x:,.0f}" if pd.notna(x) else "—")
        display_table['Violation_Rate'] = display_table['Violation_Rate'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        display_table['RN_HPRD_Avg'] = display_table['RN_HPRD_Avg'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        
        # Format beds with thousands separators if available
        if 'Number of Certified Beds' in display_table.columns:
            display_table['Number of Certified Beds'] = display_table['Number of Certified Beds'].apply(
                lambda x: f"{float(x):,.0f}" if pd.notna(x) and str(x).replace('.', '').replace('-', '').isdigit() else "—"
            )
        
        # Rename columns for display - handle different column counts
        if not provider_data.empty:
            display_table.columns = ['Facility Name', 'City', 'County', 'RN < 8 Days', 'Violation Rate', 'Avg RN HPRD', 'Risk Level', 'Beds', 'Ownership Type']
        else:
            display_table.columns = ['Facility Name', 'City', 'County', 'RN < 8 Days', 'Violation Rate', 'Avg RN HPRD', 'Risk Level']
        
        # Render as HTML table for better styling and sorting
        html_table = display_table.to_html(
            index=False,
            escape=False,
            classes=['dataframe', 'table', 'table-striped'],
            table_id='rn8-facilities-table'
        )
         
        # Add CSS for table styling (similar to PBJ Dashboard)
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
             table-layout: fixed;
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
             cursor: pointer;
             user-select: none;
             position: relative;
         }
         .dataframe th:hover {
             background: #e9ecef;
         }
         .dataframe th::after {
             content: ' ↕';
             font-size: 0.7em;
             color: #6c757d;
             position: absolute;
             right: 4px;
             top: 50%;
             transform: translateY(-50%);
         }
         .dataframe th.sort-asc::after {
             content: ' ↑';
             color: #007bff;
         }
         .dataframe th.sort-desc::after {
             content: ' ↓';
             color: #007bff;
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
         </style>
         """, unsafe_allow_html=True)

        st.markdown(html_table, unsafe_allow_html=True)

        # Add sorting functionality
        from streamlit.components.v1 import html
        html('''
        <script src='https://cdnjs.cloudflare.com/ajax/libs/tablesort/5.0.2/tablesort.min.js'></script>
        <script>
            var table = window.parent.document.getElementById("rn8-facilities-table");
            if (table) {
                new Tablesort(table);
                console.log("RN8 table sorting initialized");
            }
        </script>
        ''')
        
        # Export functionality
        with st.expander("📤 Export Data", expanded=False):
            st.markdown("""
            <div class="export-section">
                Export filtered data for legal filings or media reporting:
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = table_data.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"rn8_california_compliance_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Summary report
                summary_text = f"""
RN 8 California Compliance Report
Generated: {datetime.now().strftime('%B %d, %Y')}

Summary:
- Total California Facilities: {compliance_metrics['total_facilities']:,}
- Facilities with RN < 8 Hours: {compliance_metrics['facilities_with_violations']:,} ({compliance_metrics['pct_with_violations']:.1f}%)
- Total Sub-8 Days Statewide: {compliance_metrics['total_violation_days']:,}
- Perfect Compliance Rate: {compliance_metrics['pct_perfect']:.1f}%

Top Violators:
"""
                for i, row in table_data.head(10).iterrows():
                    summary_text += f"- {proper_title_case(row['PROVNAME'])} ({row['CITY']}): {row['Days_RN_Less_8']} sub-8 days\n"
                
                st.download_button(
                    label="📄 Download Summary Report",
                    data=summary_text,
                    file_name=f"rn8_california_summary_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
    
    # Temporal Analysis - When RN Violations Occurred
    st.subheader("📅 Temporal Analysis: When RN Violations Occurred")
     
    # Clear explanation of what this section shows
    if selected_quarter != "All Quarters":
        st.markdown(f"""
        **📊 What This Section Shows:**
        - **Quarter-Specific Analysis:** {selected_quarter} data for all California nursing homes
        - **Time Period:** {selected_quarter} only
        - **Data Scope:** All California facilities with daily RN staffing data in this quarter
        - **Violation Definition:** Days with less than 8 total RN hours (RN + RN Admin + RN DON)
        """)
    else:
        st.markdown("""
        **📊 What This Section Shows:**
        - **Statewide Analysis:** All California nursing homes across all quarters (2017-2025)
        - **Time Period:** 33 quarters from January 2017 through March 2025
        - **Data Scope:** 1,222 California facilities with daily RN staffing data
        - **Violation Definition:** Days with less than 8 total RN hours (RN + RN Admin + RN DON)
        """)
    
    if not quarterly_metrics.empty and 'CY_QTR' in quarterly_metrics.columns:
         # Quarterly violation trends
         quarterly_summary = quarterly_metrics.groupby('CY_QTR').agg({
             'Days_RN_Less_8': 'sum',
             'PROVNUM': 'count',
             'Pct_RN_Less_8': 'mean'
         }).reset_index()
         quarterly_summary.columns = ['Quarter', 'Total_Violation_Days', 'Facility_Count', 'Avg_Violation_Rate']
         
         # Format quarter display
         quarterly_summary['Quarter_Display'] = quarterly_summary['Quarter'].apply(format_quarter_display)
         
         col1, col2 = st.columns(2)
         
         with col1:
             # Violation days by quarter
             fig_quarterly = px.line(
                 quarterly_summary,
                 x='Quarter_Display',
                 y='Total_Violation_Days',
                 title="RN < 8 Hours Violations by Quarter",
                 labels={'Total_Violation_Days': 'Total Violation Days', 'Quarter_Display': 'Quarter'},
                 markers=True
             )
             fig_quarterly.update_layout(height=400, xaxis_tickangle=-45)
             st.plotly_chart(fig_quarterly, use_container_width=True)
         
         with col2:
             # Average violation rate by quarter
             fig_rate = px.bar(
                 quarterly_summary,
                 x='Quarter_Display',
                 y='Avg_Violation_Rate',
                 title="Average Violation Rate by Quarter",
                 labels={'Avg_Violation_Rate': 'Average Violation Rate (%)', 'Quarter_Display': 'Quarter'},
                 color='Avg_Violation_Rate',
                 color_continuous_scale='Reds'
             )
             fig_rate.update_layout(height=400, xaxis_tickangle=-45)
             st.plotly_chart(fig_rate, use_container_width=True)
         
         # Quarterly summary table
         st.markdown("**📊 Quarterly Violation Summary:**")
         quarterly_display = quarterly_summary.copy()
         quarterly_display['Total_Violation_Days'] = quarterly_display['Total_Violation_Days'].apply(lambda x: f"{x:,.0f}")
         quarterly_display['Avg_Violation_Rate'] = quarterly_display['Avg_Violation_Rate'].apply(lambda x: f"{x:.1f}%")
         quarterly_display = quarterly_display[['Quarter_Display', 'Total_Violation_Days', 'Facility_Count', 'Avg_Violation_Rate']]
         quarterly_display.columns = ['Quarter', 'Total Violation Days', 'Facilities', 'Avg Violation Rate']
         st.dataframe(quarterly_display, use_container_width=True)
         
         # Identify worst quarters
         worst_quarter = quarterly_summary.loc[quarterly_summary['Total_Violation_Days'].idxmax()]
         best_quarter = quarterly_summary.loc[quarterly_summary['Total_Violation_Days'].idxmin()]
         
         col1, col2 = st.columns(2)
         with col1:
             st.markdown(f"""
             <div class="metric-card critical-card">
                 <div class="metric-value">{worst_quarter['Quarter_Display']}</div>
                 <div class="metric-label">Worst Quarter: {worst_quarter['Total_Violation_Days']:,.0f} violation days</div>
             </div>
             """, unsafe_allow_html=True)
         
         with col2:
             st.markdown(f"""
             <div class="metric-card success-card">
                 <div class="metric-value">{best_quarter['Quarter_Display']}</div>
                 <div class="metric-label">Best Quarter: {best_quarter['Total_Violation_Days']:,.0f} violation days</div>
             </div>
             """, unsafe_allow_html=True)
     
    else:
        st.info("📅 **Quarterly data not available** - Run the calculation script to generate quarterly metrics for temporal analysis.")
    
    # Recent violations analysis
    if not quarterly_metrics.empty and 'CY_QTR' in quarterly_metrics.columns:
        st.subheader("🚨 Recent Violations Analysis")
        
        # Get the most recent quarters
        recent_quarters = sorted(quarterly_metrics['CY_QTR'].unique())[-4:]  # Last 4 quarters
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Most Recent Quarters:**")
            recent_summary = quarterly_metrics[quarterly_metrics['CY_QTR'].isin(recent_quarters)].groupby('CY_QTR').agg({
                'Days_RN_Less_8': 'sum',
                'PROVNUM': 'count'
            }).reset_index()
            recent_summary.columns = ['Quarter', 'Violation Days', 'Facilities']
            recent_summary = recent_summary.sort_values('Quarter')
            
            for _, row in recent_summary.iterrows():
                quarter_label = format_quarter_display(row['Quarter'])
                st.markdown(f"• **{quarter_label}:** {row['Violation Days']:,.0f} violation days across {row['Facilities']} facilities")
        
        with col2:
            # Recent trend
            recent_trend = quarterly_metrics[quarterly_metrics['CY_QTR'].isin(recent_quarters)].groupby('CY_QTR')['Days_RN_Less_8'].sum()
            if len(recent_trend) > 1:
                trend_direction = "📈 Increasing" if recent_trend.iloc[-1] > recent_trend.iloc[0] else "📉 Decreasing"
                st.markdown(f"**📈 Recent Trend:** {trend_direction}")
                st.markdown(f"• Latest quarter: {recent_trend.iloc[-1]:,.0f} violation days")
                st.markdown(f"• Previous quarter: {recent_trend.iloc[-2]:,.0f} violation days")
    
    # General patterns and insights
    st.subheader("📈 Compliance Patterns & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Violation distribution
        fig_dist = px.histogram(
            facility_metrics_filtered,
            x='Days_RN_Less_8',
            nbins=20,
            title="Distribution of RN < 8 Hours Violations",
            labels={'Days_RN_Less_8': 'Number of Sub-8 Days', 'count': 'Number of Facilities'}
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        # Risk level distribution
        risk_counts = facility_metrics_filtered['Risk_Level'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Facilities by Risk Level",
            color_discrete_map={'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4caf50'}
        )
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Compliance context
    st.subheader("🏆 Compliance Champions & Repeat Violators")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ Perfect Compliance Facilities**")
        perfect_facilities = facility_metrics[facility_metrics['Days_RN_Less_8'] == 0].head(10)
        
        if not perfect_facilities.empty:
            for _, facility in perfect_facilities.iterrows():
                st.markdown(f"""
                <div class="facility-card success-card">
                    <div class="facility-name">{proper_title_case(facility['PROVNAME'])}</div>
                    <div style="color: #666; font-size: 0.85rem;">
                        {facility['CITY']}, {facility['COUNTY_NAME']} • {format_number(facility['Total_Days'])} days analyzed
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No facilities with perfect compliance found.")
    
    with col2:
        st.markdown("**🚨 Top Violators**")
        top_violators = facility_metrics[facility_metrics['Days_RN_Less_8'] > 0].head(10)
        
        if not top_violators.empty:
            for _, facility in top_violators.iterrows():
                st.markdown(f"""
                <div class="facility-card critical-card">
                    <div class="facility-name">{proper_title_case(facility['PROVNAME'])}</div>
                    <div style="color: #666; font-size: 0.85rem;">
                        {facility['CITY']}, {facility['COUNTY_NAME']} • {format_number(facility['Days_RN_Less_8'])} sub-8 days
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No violation data found.")
    
    # Methodology and data notes
    with st.expander("📋 About This Dashboard", expanded=False):
        st.markdown("""
        **Methodology & Data Sources:**
        
        **Federal Requirement:** 42 CFR 483.35 requires "at least 8 consecutive hours of RN coverage every day"
        
        **Data Source:** CMS Payroll-Based Journal (PBJ) daily nurse staffing data, 2017-2025
        
        **Time Period:** This analysis covers all available PBJ data from 2017 through 2025, with facilities analyzed across multiple quarters to identify compliance patterns over time.
        
        **Calculation Method:**
        - RN hours = RN + RN Admin + RN DON positions
        - Sub-8 day = any day with < 8 total RN hours
        - Risk scoring: 60% weight for RN violations, 40% weight for HPRD violations
        
        **Risk Levels:**
        - **High Risk:** Combined score ≥ 50% - Chronic understaffing pattern
        - **Medium Risk:** Combined score 25-49% - Moderate violations  
        - **Low Risk:** Combined score < 25% - Minimal violations
        
        **Data Caveats:**
        - Based on self-reported staffing data
        - May not capture all RN positions in all facilities
        - Historical data may have reporting inconsistencies
        - Perfect compliance may indicate data quality issues
        
        **Legal Use:** This dashboard identifies potential negligence cases and regulatory violations for legal review.
        """)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
        <p>Source: <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" style="color: #1E88E5; text-decoration: none;">CMS Payroll-Based Journal Data, 2017-2025</a></p>
        <p>By <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">320 Consulting LLC</a></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
