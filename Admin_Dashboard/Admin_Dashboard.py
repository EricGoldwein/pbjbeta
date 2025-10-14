import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import os

# Page configuration
st.set_page_config(
    page_title="PBJ Admin Dashboard by 320 Consulting",
    page_icon="👨‍💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .chart-container {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_admin_data():
    """Load admin metrics data from CSV files."""
    try:
        # Load different levels of metrics
        facility_metrics = pd.read_csv("admin_data/admin_facility_metrics.csv")
        state_metrics = pd.read_csv("admin_data/admin_state_metrics.csv")
        national_metrics = pd.read_csv("admin_data/admin_national_metrics.csv")
        
        # Load new advanced analytics datasets
        day_of_week_metrics = pd.read_csv("admin_data/admin_day_of_week_metrics.csv")
        holiday_metrics = pd.read_csv("admin_data/admin_holiday_metrics.csv")
        holiday_proximity_metrics = pd.read_csv("admin_data/admin_holiday_proximity_metrics.csv")
        regional_metrics = pd.read_csv("admin_data/admin_regional_metrics.csv")
        ownership_metrics = pd.read_csv("admin_data/admin_ownership_metrics.csv")
        anomalies_metrics = pd.read_csv("admin_data/admin_anomalies_metrics.csv")
        
        # Filter out the problematic Q4 2021 data (has incorrect MPRD values and data quality issues)
        national_metrics = national_metrics[national_metrics['Quarter'] != 'CY2021Q4']
        state_metrics = state_metrics[state_metrics['Quarter'] != 'CY2021Q4']
        facility_metrics = facility_metrics[facility_metrics['Quarter'] != 'CY2021Q4']
        
        # Filter state metrics to only include actual states (2-letter codes)
        valid_states = ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
        state_metrics = state_metrics[state_metrics['STATE'].isin(valid_states)]
        
        # Convert Quarter to datetime for better sorting
        for df in [facility_metrics, state_metrics, national_metrics, day_of_week_metrics, 
                  holiday_metrics, holiday_proximity_metrics, regional_metrics, ownership_metrics]:
            # Extract year and quarter from CY2023Q1 format
            df['Year'] = df['Quarter'].str[2:6].astype(int)
            df['Quarter_Num'] = df['Quarter'].str[6:].str.replace('Q', '').astype(int)
            # Create a proper datetime for sorting (using first day of quarter)
            df['Quarter_Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + ((df['Quarter_Num'] - 1) * 3 + 1).astype(str).str.zfill(2) + '-01')
        
        return (facility_metrics, state_metrics, national_metrics, day_of_week_metrics, 
                holiday_metrics, holiday_proximity_metrics, regional_metrics, ownership_metrics, anomalies_metrics)
    except Exception as e:
        st.error(f"Error loading admin data: {str(e)}")
        return None, None, None, None, None, None, None, None, None

@st.cache_data
def load_facility_info():
    """Load facility information for search functionality."""
    try:
        # Load a sample of facility data to get facility names
        facility_metrics = pd.read_csv("admin_data/admin_facility_metrics.csv")
        facility_info = facility_metrics[['PROVNUM', 'PROVNAME', 'STATE']].drop_duplicates()
        return facility_info
    except Exception as e:
        st.error(f"Error loading facility info: {str(e)}")
        return None

def format_quarter_label(quarter_str):
    """Format quarter string for display."""
    year = quarter_str[2:6]
    quarter = quarter_str[6:]
    return f"{year} {quarter}"

def create_trend_chart(data, metric, title, y_axis_title, color='#1f77b4', is_percentage=False):
    """Create a trend chart for a specific metric with PBJ Dashboard styling."""
    # Sort data by Quarter_Date for proper chronological order
    data = data.sort_values('Quarter_Date')
    
    # Create year labels for x-axis ticks (matching PBJ Dashboard style)
    min_year = data['Year'].min()
    max_year = data['Year'].max()
    all_years = range(min_year, max_year + 1)
    tick_values = [pd.Timestamp(f"{year}-01-01") for year in all_years]
    tick_text = [str(year) for year in all_years]
    
    # Get the actual date range from the data
    date_range = [data['Quarter_Date'].min(), data['Quarter_Date'].max()]
    
    # Define hover template based on metric type
    if is_percentage:
        hover_template = "<b>%{customdata}</b><br>%{y:.1f}%<extra></extra>"
    else:
        hover_template = "<b>%{customdata}</b><br>%{y:.2f}<extra></extra>"
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data['Quarter_Date'], 
        y=data[metric],
        mode='lines+markers',
        name=metric,
        line=dict(color=color, width=2),
        marker=dict(size=6),
        customdata=data['Quarter'].apply(lambda x: f"Q{x[-1]} {x[2:6]}"),
        hovertemplate=hover_template
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title=y_axis_title,
        hovermode='x unified',
        showlegend=False,
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        template='plotly_white'
    )
    
    # Update x-axis to match PBJ Dashboard style
    fig.update_xaxes(
        tickvals=tick_values,
        ticktext=tick_text,
        tickangle=45,
        showline=True,
        linewidth=1,
        linecolor="rgba(200, 200, 200, 0.1)",
        range=date_range,
        nticks=len(tick_values) // 2 if len(tick_values) > 4 else len(tick_values),
        tickmode='auto'
    )
    
    # Add footer annotation
    fig.add_annotation(
        text="320 Consulting | Source: CMS PBJ Data",
        x=0.99,
        y=-0.15,
        xref="x domain",
        yref="y domain",
        showarrow=False,
        font=dict(size=10, color="gray"),
        align="right"
    )
    
    return fig

def create_histogram(data, metric, title, x_axis_title, bins=50, color='#1f77b4'):
    """Create a histogram for admin hours distribution."""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data[metric],
        nbinsx=bins,
        marker_color=color,
        opacity=0.7
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title="Frequency",
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        template='plotly_white'
    )
    
    return fig

def create_comparison_chart(data, metric, title, y_axis_title, group_col='STATE', top_n=10):
    """Create a comparison chart showing top states/facilities."""
    # Get the latest quarter
    latest_quarter = data['Quarter'].max()
    latest_data = data[data['Quarter'] == latest_quarter]
    
    # Sort by metric and get top N
    top_data = latest_data.nlargest(top_n, metric)
    
    fig = px.bar(
        top_data,
        x=group_col,
        y=metric,
        title=f"{title} - {format_quarter_label(latest_quarter)}",
        color=metric,
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=y_axis_title,
        height=400
    )
    
    fig.update_xaxes(tickangle=45)
    return fig

def create_days_comparison_chart(data, title):
    """Create a chart comparing days with vs without admin."""
    latest_quarter = data['Quarter'].max()
    latest_data = data[data['Quarter'] == latest_quarter]
    
    # Calculate totals
    total_with_admin = latest_data['Days_With_Admin'].sum()
    total_without_admin = latest_data['Days_Without_Admin'].sum()
    
    fig = go.Figure(data=[
        go.Pie(
            labels=['Days with Admin', 'Days without Admin'],
            values=[total_with_admin, total_without_admin],
            hole=0.4,
            marker_colors=['#1f77b4', '#ff7f0e']
        )
    ])
    
    fig.update_layout(
        title=f"{title} - {format_quarter_label(latest_quarter)}",
        height=400
    )
    
    return fig

def create_state_map(data, metric, title):
    """Create a choropleth map of US states."""
    # Get the latest quarter
    latest_quarter = data['Quarter'].max()
    latest_data = data[data['Quarter'] == latest_quarter]
    
    fig = px.choropleth(
        latest_data,
        locations='STATE',
        locationmode='USA-states',
        color=metric,
        scope='usa',
        title=f"{title} - {format_quarter_label(latest_quarter)}",
        color_continuous_scale='Blues',
        hover_data=['STATE', metric]
    )
    
    fig.update_layout(
        height=500,
        geo=dict(
            showlakes=True,
            lakecolor='rgb(255, 255, 255)'
        )
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">👨‍💼 PBJ Admin Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Administrative Staffing Analysis by 320 Consulting</p>', unsafe_allow_html=True)
    
    # About section focused on Admin data
    with st.expander("ℹ️ About this Dashboard"):
        st.markdown("""
        **PBJ Administrative Staffing Dashboard**
        
        This dashboard provides comprehensive analysis of administrative staffing in nursing homes using CMS PBJ (Provider Data for Nursing Home Compare) data.
        
        **Key Metrics:**
        - **Admin Hours per Day**: Average administrative staff hours worked per day
        - **Admin MPRD**: Minutes per Resident Day for administrative staff
        - **Days with Admin**: Percentage of days with administrative staff present
        - **Contract Admin**: Percentage of administrative hours provided by contract staff
        
        **Data Coverage:**
        - Quarterly data from 2017 to 2025
        - National, state, and facility-level analysis
        - Real-time calculations and trend analysis
        
        **Data Source:** CMS PBJ (Provider Data for Nursing Home Compare)
        
        **Note:** Q4 2021 data has been excluded due to data quality issues.
        """)
    
    # Load data
    (facility_metrics, state_metrics, national_metrics, day_of_week_metrics, 
     holiday_metrics, holiday_proximity_metrics, regional_metrics, ownership_metrics, 
     anomalies_metrics) = load_admin_data()
    facility_info = load_facility_info()
    
    if facility_metrics is None or state_metrics is None or national_metrics is None:
        st.error("Failed to load admin data. Please ensure the admin_data directory contains the required CSV files.")
        return
    
    # Sidebar for navigation and filters
    st.sidebar.markdown("## 📊 Dashboard Navigation")
    
    # Analysis level selection
    analysis_level = st.sidebar.selectbox(
        "Select Analysis Level",
        ["National Overview", "State Comparison", "Facility Search", "Advanced Analytics", "Regional Analysis", "Holiday Analysis", "Anomaly Detection"],
        index=0
    )
    
    # Date range filter
    st.sidebar.markdown("### 📅 Date Range")
    quarters = sorted(national_metrics['Quarter'].unique())
    start_quarter = st.sidebar.selectbox("Start Quarter", quarters, index=0)
    end_quarter = st.sidebar.selectbox("End Quarter", quarters, index=len(quarters)-1)
    
    # About section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About Admin Data")
    with st.sidebar.expander("Learn about Administrative Staffing"):
        st.markdown("""
        **Administrative Staffing Data** from CMS Payroll-Based Journal (PBJ) files tracks administrative personnel hours in nursing homes.
        
        **Key Metrics:**
        - **Admin Hours/Day**: Total administrative staff hours per day
        - **Admin MPRD**: Minutes per resident day for administrative staff
        - **Contract %**: Percentage of admin hours provided by contract staff
        - **Days with Admin**: Percentage of days with administrative staff present
        
        **Data Quality:** Facilities reporting >48 hours average per quarter are excluded as likely reporting errors.
        
        **Data Source:** CMS PBJ Non-Nurse Staffing files, 2017-2025
        """)
    
    # Filter data based on date range
    start_idx = quarters.index(start_quarter)
    end_idx = quarters.index(end_quarter)
    selected_quarters = quarters[start_idx:end_idx+1]
    
    filtered_national = national_metrics[national_metrics['Quarter'].isin(selected_quarters)]
    filtered_state = state_metrics[state_metrics['Quarter'].isin(selected_quarters)]
    filtered_facility = facility_metrics[facility_metrics['Quarter'].isin(selected_quarters)]
    filtered_day_of_week = day_of_week_metrics[day_of_week_metrics['Quarter'].isin(selected_quarters)]
    filtered_holiday = holiday_metrics[holiday_metrics['Quarter'].isin(selected_quarters)]
    filtered_holiday_proximity = holiday_proximity_metrics[holiday_proximity_metrics['Quarter'].isin(selected_quarters)]
    filtered_regional = regional_metrics[regional_metrics['Quarter'].isin(selected_quarters)]
    filtered_ownership = ownership_metrics[ownership_metrics['Quarter'].isin(selected_quarters)]
    filtered_anomalies = anomalies_metrics[anomalies_metrics['Quarter'].isin(selected_quarters)]
    
    # Main content based on analysis level
    if analysis_level == "National Overview":
        display_national_overview(filtered_national)
    elif analysis_level == "State Comparison":
        display_state_comparison(filtered_state)
    elif analysis_level == "Facility Search":
        display_facility_search(filtered_facility, facility_info)
    elif analysis_level == "Advanced Analytics":
        display_advanced_analytics(filtered_day_of_week, filtered_holiday, filtered_holiday_proximity)
    elif analysis_level == "Regional Analysis":
        display_regional_analysis(filtered_regional)
    elif analysis_level == "Holiday Analysis":
        display_holiday_analysis(filtered_holiday, filtered_holiday_proximity)
    elif analysis_level == "Anomaly Detection":
        display_anomaly_detection(filtered_anomalies)

def display_national_overview(national_data):
    """Display national overview dashboard."""
    st.markdown("## 🏛️ National Administrative Staffing Overview")
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    latest_data = national_data[national_data['Quarter'] == national_data['Quarter'].max()].iloc[0]
    
    with col1:
        st.metric(
            label="Mean Admin Hours/Day",
            value=f"{latest_data['Mean_Admin_Hours']:.1f}",
            delta=f"{latest_data['Mean_Admin_Hours'] - national_data['Mean_Admin_Hours'].mean():.1f}"
        )
    
    with col2:
        st.metric(
            label="Median Admin Hours/Day",
            value=f"{latest_data['Median_Admin_Hours']:.1f}",
            delta=f"{latest_data['Median_Admin_Hours'] - national_data['Median_Admin_Hours'].mean():.1f}"
        )
    
    with col3:
        st.metric(
            label="Mean Admin MPRD",
            value=f"{latest_data['Mean_Admin_MPRD']:.1f}",
            delta=f"{latest_data['Mean_Admin_MPRD'] - national_data['Mean_Admin_MPRD'].mean():.1f}"
        )
    
    with col4:
        st.metric(
            label="% Days with Admin",
            value=f"{latest_data['Pct_Days_With_Admin']:.1f}%",
            delta=f"{latest_data['Pct_Days_With_Admin'] - national_data['Pct_Days_With_Admin'].mean():.1f}%"
        )
    
    # Charts
    st.markdown("### 📈 National Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig1 = create_trend_chart(
            national_data, 
            'Mean_Admin_Hours', 
            'Mean Admin Hours per Day',
            'Hours'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        fig2 = create_trend_chart(
            national_data, 
            'Mean_Admin_MPRD', 
            'Mean Admin Minutes per Resident Day',
            'Minutes per Resident Day',
            color='#2ca02c'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        fig3 = create_trend_chart(
            national_data, 
            'Median_Admin_MPRD', 
            'Median Admin Minutes per Resident Day',
            'Minutes per Resident Day',
            color='#d62728'
        )
        st.plotly_chart(fig3, use_container_width=True)
    
    # Admin Hours Distribution Histogram
    st.markdown("### 📊 Admin Hours Distribution")
    fig_hist = create_histogram(
        national_data,
        'Mean_Admin_Hours',
        'Distribution of Mean Admin Hours per Day',
        'Admin Hours per Day'
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    
    # Contract Admin Percentage (de-prioritized)
    st.markdown("### 📋 Contract Admin Percentage")
    fig_contract = create_trend_chart(
        national_data, 
        'Pct_Contract_Admin', 
        'Percent Contract Admin Hours',
        'Percentage',
        color='#9467bd',
        is_percentage=True
    )
    st.plotly_chart(fig_contract, use_container_width=True)
    
    # Days comparison chart
    st.markdown("### 📊 Days with vs Without Admin Staff")
    fig_days = create_days_comparison_chart(national_data, "National Days Comparison")
    st.plotly_chart(fig_days, use_container_width=True)

def display_state_comparison(state_data):
    """Display state comparison dashboard."""
    st.markdown("## 🗺️ State Administrative Staffing Comparison")
    
    # State map
    st.markdown("### 🗺️ State Map - Median Admin MPRD")
    fig_map = create_state_map(state_data, 'Median_Admin_MPRD', 'Median Admin Minutes per Resident Day by State')
    st.plotly_chart(fig_map, use_container_width=True)
    
    # State selection
    states = sorted(state_data['STATE'].unique())
    selected_states = st.multiselect(
        "Select States to Compare (default: none selected)",
        states,
        default=[]  # No states selected by default
    )
    
    if not selected_states:
        st.info("Please select states above to view detailed comparisons.")
        return
    
    filtered_state_data = state_data[state_data['STATE'].isin(selected_states)]
    
    # State comparison charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Create multi-line chart for Admin Hours
        fig1 = go.Figure()
        for state in selected_states:
            state_data_filtered = filtered_state_data[filtered_state_data['STATE'] == state].sort_values('Quarter_Date')
            fig1.add_trace(go.Scatter(
                x=state_data_filtered['Quarter_Date'],
                y=state_data_filtered['Mean_Admin_Hours'],
                mode='lines+markers',
                name=state,
                line=dict(width=2),
                marker=dict(size=4),
                customdata=state_data_filtered['Quarter'].apply(lambda x: f"Q{x[-1]} {x[2:6]}"),
                hovertemplate="<b>%{customdata}</b><br>%{y:.1f} Hours<extra></extra>"
            ))
        
        fig1.update_layout(
            title='Mean Admin Hours per Day by State',
            xaxis_title="",
            yaxis_title="Hours",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white',
            hovermode='x unified'
        )
        
        # Update x-axis to match PBJ Dashboard style
        min_year = filtered_state_data['Year'].min()
        max_year = filtered_state_data['Year'].max()
        all_years = range(min_year, max_year + 1)
        tick_values = [pd.Timestamp(f"{year}-01-01") for year in all_years]
        tick_text = [str(year) for year in all_years]
        date_range = [filtered_state_data['Quarter_Date'].min(), filtered_state_data['Quarter_Date'].max()]
        
        fig1.update_xaxes(
            tickvals=tick_values,
            ticktext=tick_text,
            tickangle=45,
            showline=True,
            linewidth=1,
            linecolor="rgba(200, 200, 200, 0.1)",
            range=date_range,
            nticks=len(tick_values) // 2 if len(tick_values) > 4 else len(tick_values),
            tickmode='auto'
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Create multi-line chart for Admin MPRD
        fig2 = go.Figure()
        for state in selected_states:
            state_data_filtered = filtered_state_data[filtered_state_data['STATE'] == state].sort_values('Quarter_Date')
            fig2.add_trace(go.Scatter(
                x=state_data_filtered['Quarter_Date'],
                y=state_data_filtered['Mean_Admin_MPRD'],
                mode='lines+markers',
                name=state,
                line=dict(width=2),
                marker=dict(size=4),
                customdata=state_data_filtered['Quarter'].apply(lambda x: f"Q{x[-1]} {x[2:6]}"),
                hovertemplate="<b>%{customdata}</b><br>%{y:.1f} MPRD<extra></extra>"
            ))
        
        fig2.update_layout(
            title='Mean Admin Minutes per Resident Day by State',
            xaxis_title="",
            yaxis_title="Minutes per Resident Day",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig2.update_xaxes(
            tickvals=tick_values,
            ticktext=tick_text,
            tickangle=45,
            showline=True,
            linewidth=1,
            linecolor="rgba(200, 200, 200, 0.1)",
            range=date_range,
            nticks=len(tick_values) // 2 if len(tick_values) > 4 else len(tick_values),
            tickmode='auto'
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Create multi-line chart for Median Admin Hours
        fig3 = go.Figure()
        for state in selected_states:
            state_data_filtered = filtered_state_data[filtered_state_data['STATE'] == state].sort_values('Quarter_Date')
            fig3.add_trace(go.Scatter(
                x=state_data_filtered['Quarter_Date'],
                y=state_data_filtered['Median_Admin_Hours'],
                mode='lines+markers',
                name=state,
                line=dict(width=2),
                marker=dict(size=4),
                customdata=state_data_filtered['Quarter'].apply(lambda x: f"Q{x[-1]} {x[2:6]}"),
                hovertemplate="<b>%{customdata}</b><br>%{y:.1f} Hours<extra></extra>"
            ))
        
        fig3.update_layout(
            title='Median Admin Hours per Day by State',
            xaxis_title="",
            yaxis_title="Hours",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig3.update_xaxes(
            tickvals=tick_values,
            ticktext=tick_text,
            tickangle=45,
            showline=True,
            linewidth=1,
            linecolor="rgba(200, 200, 200, 0.1)",
            range=date_range,
            nticks=len(tick_values) // 2 if len(tick_values) > 4 else len(tick_values),
            tickmode='auto'
        )
        st.plotly_chart(fig3, use_container_width=True)
        
        # Create multi-line chart for Days with Admin
        fig4 = go.Figure()
        for state in selected_states:
            state_data_filtered = filtered_state_data[filtered_state_data['STATE'] == state].sort_values('Quarter_Date')
            fig4.add_trace(go.Scatter(
                x=state_data_filtered['Quarter_Date'],
                y=state_data_filtered['Pct_Days_With_Admin'],
                mode='lines+markers',
                name=state,
                line=dict(width=2),
                marker=dict(size=4),
                customdata=state_data_filtered['Quarter'].apply(lambda x: f"Q{x[-1]} {x[2:6]}"),
                hovertemplate="<b>%{customdata}</b><br>%{y:.1f}%<extra></extra>"
            ))
        
        fig4.update_layout(
            title='Percent Days with Admin by State',
            xaxis_title="",
            yaxis_title="Percentage",
            height=400,
            margin=dict(l=50, r=50, t=80, b=50),
            template='plotly_white',
            hovermode='x unified'
        )
        
        fig4.update_xaxes(
            tickvals=tick_values,
            ticktext=tick_text,
            tickangle=45,
            showline=True,
            linewidth=1,
            linecolor="rgba(200, 200, 200, 0.1)",
            range=date_range,
            nticks=len(tick_values) // 2 if len(tick_values) > 4 else len(tick_values),
            tickmode='auto'
        )
        st.plotly_chart(fig4, use_container_width=True)
    
    # Top states comparison
    st.markdown("### 🏆 Top States Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_top_hours = create_comparison_chart(
            state_data, 
            'Mean_Admin_Hours', 
            'Top States by Mean Admin Hours',
            'Hours per Day',
            'STATE'
        )
        st.plotly_chart(fig_top_hours, use_container_width=True)
    
    with col2:
        fig_top_mprd = create_comparison_chart(
            state_data, 
            'Mean_Admin_MPRD', 
            'Top States by Mean Admin MPRD',
            'Minutes per Resident Day',
            'STATE'
        )
        st.plotly_chart(fig_top_mprd, use_container_width=True)

def display_facility_search(facility_data, facility_info):
    """Display facility search and analysis."""
    st.markdown("## 🏥 Facility Administrative Staffing Search")
    
    # Search options
    search_option = st.radio(
        "Search by:",
        ["Facility Name", "Provider Number", "State"]
    )
    
    if search_option == "Facility Name":
        if facility_info is not None:
            facility_names = sorted(facility_info['PROVNAME'].unique())
            selected_facility = st.selectbox("Select Facility", facility_names)
            if selected_facility:
                provnum = facility_info[facility_info['PROVNAME'] == selected_facility]['PROVNUM'].iloc[0]
                facility_data_filtered = facility_data[facility_data['PROVNUM'] == provnum]
        else:
            st.error("Facility information not available.")
            return
    
    elif search_option == "Provider Number":
        provnum = st.text_input("Enter Provider Number (6 digits)")
        if provnum:
            provnum = provnum.zfill(6).upper()
            facility_data_filtered = facility_data[facility_data['PROVNUM'] == provnum]
        else:
            facility_data_filtered = pd.DataFrame()
    
    else:  # State
        states = sorted(facility_data['STATE'].unique())
        selected_state = st.selectbox("Select State", states)
        if selected_state:
            facility_data_filtered = facility_data[facility_data['STATE'] == selected_state]
        else:
            facility_data_filtered = pd.DataFrame()
    
    if facility_data_filtered.empty:
        st.warning("No data found for the selected criteria.")
        return
    
    # Display facility metrics
    if search_option in ["Facility Name", "Provider Number"] and len(facility_data_filtered) > 0:
        # Single facility analysis
        facility_name = facility_data_filtered['PROVNAME'].iloc[0]
        st.markdown(f"### 📊 {facility_name} Administrative Staffing Analysis")
        
        # Key metrics
        latest_data = facility_data_filtered[facility_data_filtered['Quarter'] == facility_data_filtered['Quarter'].max()].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Admin Hours/Day", f"{latest_data['Mean_Admin_Hours']:.1f}")
        with col2:
            st.metric("Median Admin Hours/Day", f"{latest_data['Median_Admin_Hours']:.1f}")
        with col3:
            st.metric("Mean Admin MPRD", f"{latest_data['Mean_Admin_MPRD']:.1f}")
        with col4:
            st.metric("% Days with Admin", f"{latest_data['Pct_Days_With_Admin']:.1f}%")
        
        # Trend charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = create_trend_chart(
                facility_data_filtered,
                'Mean_Admin_Hours',
                'Mean Admin Hours per Day Trend',
                'Hours'
            )
            st.plotly_chart(fig1, use_container_width=True)
            
            fig2 = create_trend_chart(
                facility_data_filtered,
                'Mean_Admin_MPRD',
                'Mean Admin MPRD Trend',
                'Minutes per Resident Day',
                color='#2ca02c'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        with col2:
            fig3 = create_trend_chart(
                facility_data_filtered,
                'Median_Admin_Hours',
                'Median Admin Hours per Day Trend',
                'Hours',
                color='#ff7f0e'
            )
            st.plotly_chart(fig3, use_container_width=True)
            
            fig4 = create_trend_chart(
                facility_data_filtered,
                'Pct_Days_With_Admin',
                'Days with Admin Trend',
                'Percentage',
                color='#d62728',
                is_percentage=True
            )
            st.plotly_chart(fig4, use_container_width=True)
    
    else:
        # Multiple facilities analysis (state view)
        st.markdown(f"### 📊 {selected_state} Facilities Analysis")
        
        # Top facilities in state
        latest_quarter = facility_data_filtered['Quarter'].max()
        latest_state_data = facility_data_filtered[facility_data_filtered['Quarter'] == latest_quarter]
        
        col1, col2 = st.columns(2)
        
        with col1:
            top_hours = latest_state_data.nlargest(10, 'Mean_Admin_Hours')
            fig1 = px.bar(
                top_hours,
                x='PROVNAME',
                y='Mean_Admin_Hours',
                title=f'Top 10 Facilities by Mean Admin Hours - {format_quarter_label(latest_quarter)}'
            )
            fig1.update_layout(height=400, xaxis_title="Facility", yaxis_title="Hours")
            fig1.update_xaxes(tickangle=45)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            top_mprd = latest_state_data.nlargest(10, 'Mean_Admin_MPRD')
            fig2 = px.bar(
                top_mprd,
                x='PROVNAME',
                y='Mean_Admin_MPRD',
                title=f'Top 10 Facilities by Mean Admin MPRD - {format_quarter_label(latest_quarter)}'
            )
            fig2.update_layout(height=400, xaxis_title="Facility", yaxis_title="Minutes per Resident Day")
            fig2.update_xaxes(tickangle=45)
            st.plotly_chart(fig2, use_container_width=True)

def display_advanced_analytics(day_of_week_data, holiday_data, holiday_proximity_data):
    """Display advanced analytics including day-of-week and holiday analysis."""
    st.markdown("## 🔬 Advanced Analytics")
    
    # Day of Week Analysis
    st.markdown("### 📅 Day of Week Analysis")
    st.markdown("Analyze administrative staffing patterns by day of the week to identify recurring gaps.")
    
    # Get latest quarter data for day of week
    latest_quarter = day_of_week_data['Quarter'].max()
    latest_dow_data = day_of_week_data[day_of_week_data['Quarter'] == latest_quarter]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Day of week admin hours
        fig1 = px.bar(
            latest_dow_data,
            x='DayOfWeek',
            y='Mean_Admin_Hours',
            title=f'Admin Hours by Day of Week - {format_quarter_label(latest_quarter)}',
            color='Mean_Admin_Hours',
            color_continuous_scale='viridis'
        )
        fig1.update_layout(height=400, xaxis_title="Day of Week", yaxis_title="Mean Admin Hours")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Day of week MPRD
        fig2 = px.bar(
            latest_dow_data,
            x='DayOfWeek',
            y='Mean_Admin_MPRD',
            title=f'Admin MPRD by Day of Week - {format_quarter_label(latest_quarter)}',
            color='Mean_Admin_MPRD',
            color_continuous_scale='plasma'
        )
        fig2.update_layout(height=400, xaxis_title="Day of Week", yaxis_title="Mean Admin MPRD")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Trend over time by day of week
    st.markdown("### 📈 Day of Week Trends Over Time")
    
    # Pivot data for trend analysis
    dow_trend = day_of_week_data.pivot_table(
        values='Mean_Admin_Hours', 
        index='Quarter_Date', 
        columns='DayOfWeek', 
        aggfunc='mean'
    ).reset_index()
    
    fig3 = go.Figure()
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        if day in dow_trend.columns:
            fig3.add_trace(go.Scatter(
                x=dow_trend['Quarter_Date'],
                y=dow_trend[day],
                mode='lines+markers',
                name=day
            ))
    
    fig3.update_layout(
        title='Admin Hours by Day of Week - Trend Over Time',
        xaxis_title='Quarter',
        yaxis_title='Mean Admin Hours',
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)

def display_regional_analysis(regional_data):
    """Display regional analysis of administrative staffing."""
    st.markdown("## 🗺️ Regional Analysis")
    st.markdown("Compare administrative staffing patterns across CMS regions.")
    
    # Latest quarter regional comparison
    latest_quarter = regional_data['Quarter'].max()
    latest_regional = regional_data[regional_data['Quarter'] == latest_quarter]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional admin hours
        fig1 = px.bar(
            latest_regional,
            x='Region',
            y='Mean_Admin_Hours',
            title=f'Mean Admin Hours by CMS Region - {format_quarter_label(latest_quarter)}',
            color='Mean_Admin_Hours',
            color_continuous_scale='viridis'
        )
        fig1.update_layout(height=400, xaxis_title="CMS Region", yaxis_title="Mean Admin Hours")
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Regional MPRD
        fig2 = px.bar(
            latest_regional,
            x='Region',
            y='Mean_Admin_MPRD',
            title=f'Mean Admin MPRD by CMS Region - {format_quarter_label(latest_quarter)}',
            color='Mean_Admin_MPRD',
            color_continuous_scale='plasma'
        )
        fig2.update_layout(height=400, xaxis_title="CMS Region", yaxis_title="Mean Admin MPRD")
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Regional trends over time
    st.markdown("### 📈 Regional Trends Over Time")
    
    fig3 = go.Figure()
    for region in regional_data['Region'].unique():
        region_data = regional_data[regional_data['Region'] == region]
        fig3.add_trace(go.Scatter(
            x=region_data['Quarter_Date'],
            y=region_data['Mean_Admin_Hours'],
            mode='lines+markers',
            name=region
        ))
    
    fig3.update_layout(
        title='Admin Hours by CMS Region - Trend Over Time',
        xaxis_title='Quarter',
        yaxis_title='Mean Admin Hours',
        height=500
    )
    st.plotly_chart(fig3, use_container_width=True)

def display_holiday_analysis(holiday_data, holiday_proximity_data):
    """Display holiday analysis of administrative staffing."""
    st.markdown("## 🎉 Holiday Analysis")
    st.markdown("Analyze administrative staffing patterns around holidays and special events.")
    
    # Holiday category analysis
    st.markdown("### 🎊 Holiday Categories")
    
    latest_quarter = holiday_data['Quarter'].max()
    latest_holiday = holiday_data[holiday_data['Quarter'] == latest_quarter]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Holiday categories admin hours
        fig1 = px.bar(
            latest_holiday,
            x='holiday_category',
            y='Mean_Admin_Hours',
            title=f'Admin Hours by Holiday Category - {format_quarter_label(latest_quarter)}',
            color='Mean_Admin_Hours',
            color_continuous_scale='viridis'
        )
        fig1.update_layout(height=400, xaxis_title="Holiday Category", yaxis_title="Mean Admin Hours")
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Holiday categories MPRD
        fig2 = px.bar(
            latest_holiday,
            x='holiday_category',
            y='Mean_Admin_MPRD',
            title=f'Admin MPRD by Holiday Category - {format_quarter_label(latest_quarter)}',
            color='Mean_Admin_MPRD',
            color_continuous_scale='plasma'
        )
        fig2.update_layout(height=400, xaxis_title="Holiday Category", yaxis_title="Mean Admin MPRD")
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Holiday proximity analysis
    st.markdown("### 📅 Holiday Proximity Analysis")
    
    latest_proximity = holiday_proximity_data[holiday_proximity_data['Quarter'] == latest_quarter]
    
    fig3 = px.bar(
        latest_proximity,
        x='HolidayProximity',
        y='Mean_Admin_Hours',
        title=f'Admin Hours by Holiday Proximity - {format_quarter_label(latest_quarter)}',
        color='Mean_Admin_Hours',
        color_continuous_scale='viridis'
    )
    fig3.update_layout(height=400, xaxis_title="Holiday Proximity", yaxis_title="Mean Admin Hours")
    fig3.update_xaxes(tickangle=45)
    st.plotly_chart(fig3, use_container_width=True)

def display_anomaly_detection(anomalies_data):
    """Display anomaly detection results."""
    st.markdown("## ⚠️ Anomaly Detection")
    st.markdown("Identify facilities with unusual administrative staffing patterns.")
    
    # Latest quarter anomalies
    latest_quarter = anomalies_data['Quarter'].max()
    latest_anomalies = anomalies_data[anomalies_data['Quarter'] == latest_quarter]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_facilities = len(latest_anomalies)
        st.metric("Total Facilities", total_facilities)
    
    with col2:
        consecutive_zero_facilities = len(latest_anomalies[latest_anomalies['Days_Consecutive_Zero'] > 0])
        st.metric("Facilities with Consecutive Zero Days", consecutive_zero_facilities)
    
    with col3:
        unusually_high_facilities = len(latest_anomalies[latest_anomalies['Days_Unusually_High'] > 0])
        st.metric("Facilities with Unusually High Hours", unusually_high_facilities)
    
    with col4:
        bottom_decile_facilities = len(latest_anomalies[latest_anomalies['Days_Bottom_Decile'] > 0])
        st.metric("Bottom Decile Facilities", bottom_decile_facilities)
    
    # Top facilities with anomalies
    st.markdown("### 🔍 Top Facilities with Anomalies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Facilities with most consecutive zero days
        top_consecutive_zero = latest_anomalies.nlargest(10, 'Days_Consecutive_Zero')
        fig1 = px.bar(
            top_consecutive_zero,
            x='PROVNAME',
            y='Days_Consecutive_Zero',
            title=f'Top 10 Facilities with Consecutive Zero Admin Days - {format_quarter_label(latest_quarter)}'
        )
        fig1.update_layout(height=400, xaxis_title="Facility", yaxis_title="Days with Consecutive Zeros")
        fig1.update_xaxes(tickangle=45)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Facilities with most unusually high days
        top_unusually_high = latest_anomalies.nlargest(10, 'Days_Unusually_High')
        fig2 = px.bar(
            top_unusually_high,
            x='PROVNAME',
            y='Days_Unusually_High',
            title=f'Top 10 Facilities with Unusually High Admin Hours - {format_quarter_label(latest_quarter)}'
        )
        fig2.update_layout(height=400, xaxis_title="Facility", yaxis_title="Days with Unusually High Hours")
        fig2.update_xaxes(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # State-level anomaly summary
    st.markdown("### 🗺️ State-Level Anomaly Summary")
    
    state_anomaly_summary = latest_anomalies.groupby('STATE').agg({
        'Days_Consecutive_Zero': 'sum',
        'Days_Unusually_High': 'sum',
        'Days_Significant_Drop': 'sum',
        'Days_Bottom_Decile': 'sum'
    }).reset_index()
    
    fig3 = px.scatter(
        state_anomaly_summary,
        x='Days_Consecutive_Zero',
        y='Days_Unusually_High',
        size='Days_Bottom_Decile',
        color='Days_Significant_Drop',
        hover_data=['STATE'],
        title=f'State-Level Anomaly Patterns - {format_quarter_label(latest_quarter)}'
    )
    fig3.update_layout(height=500, xaxis_title="Total Consecutive Zero Days", yaxis_title="Total Unusually High Days")
    st.plotly_chart(fig3, use_container_width=True)

if __name__ == "__main__":
    main()
