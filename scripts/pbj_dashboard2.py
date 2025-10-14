import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import duckdb
from typing import Optional, Tuple

# Set page config with minimal features
st.set_page_config(
    page_title="PBJ Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def proper_title_case(text: str) -> str:
    """
    Convert text to proper title case, keeping words like 'and', 'of', etc. lowercase.
    """
    if not text:
        return text
        
    # Words that should remain lowercase unless they're the first word
    lowercase_words = {'and', 'of', 'the', 'in', 'at', 'for', 'to', 'with', 'by'}
    
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

# Formatting utility
def format_metric(value, decimal_places=1, percentage=False, thousands=False):
    if pd.isna(value):
        return "N/A"
    if percentage:
        return f"{value:.{decimal_places}f}%"
    if thousands:
        return f"{value:,.{decimal_places}f}"
    return f"{value:.{decimal_places}f}"

def format_quarter_display(quarter: str) -> str:
    """Format quarter display as 'Q1 2024'."""
    return f"Q{quarter[-1]} {quarter[:4]}"

# Initialize DuckDB connection
@st.cache_resource
def init_db():
    """Initialize DuckDB connection and create tables with indexes."""
    conn = duckdb.connect(':memory:')
    
    # Create facility_quarter_metrics table with indexes
    conn.execute("""
        CREATE TABLE facility_quarter_metrics AS 
        SELECT * FROM read_csv_auto('facility_quarter_metrics.csv')
    """)
    
    # Create indexes for fast querying
    conn.execute("CREATE INDEX idx_provnum ON facility_quarter_metrics(PROVNUM)")
    conn.execute("CREATE INDEX idx_date ON facility_quarter_metrics(date)")
    conn.execute("CREATE INDEX idx_cy_qtr ON facility_quarter_metrics(CY_QTR)")
    
    return conn

# Get filtered data with optimized querying
@st.cache_data
def get_filtered_data(conn, start_date=None, end_date=None, provnum=None):
    """Get filtered data using DuckDB's optimized querying."""
    query = "SELECT * FROM facility_quarter_metrics WHERE 1=1"
    params = []
    
    if start_date:
        query += " AND date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND date <= ?"
        params.append(end_date)
    if provnum:
        query += " AND PROVNUM = ?"
        params.append(provnum)
    
    return conn.execute(query, params).df()

# Load data with minimal processing
@st.cache_data
def load_data():
    """Load and process national metrics data."""
    df = pd.read_csv('national_pbj_metrics.csv')
    
    # Convert Quarter to datetime
    df['Year'] = df['Quarter'].str[:4].astype(int)
    df['Quarter_Num'] = df['Quarter'].str[-1].astype(int)
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                               ((df['Quarter_Num'] * 3) - 2).astype(str) + '-01')
    
    # Filter for 2017-2024
    df = df[(df['Date'].dt.year >= 2017) & (df['Date'].dt.year <= 2024)]
    
    return df

# Cache the plot creation with minimal features
@st.cache_data(ttl=3600, show_spinner=False)
def create_plot(df: pd.DataFrame, view_mode: str = "Desktop") -> go.Figure:
    """Create and cache the plot figure."""
    try:
        if view_mode == "Mobile":
            fig = make_subplots(rows=2, cols=1,
                             subplot_titles=('Total Nurse HPRD', 'Average Daily Census'),
                             vertical_spacing=0.2)
            
            fig.add_trace(go.Scatter(x=df['date'], y=df['Total_Nurse_HPRD'],
                                  mode='lines', name='Total HPRD',
                                  hovertemplate="%{y:.2f}<extra></extra>"), 
                         row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['date'], y=df['Average_Daily_Census'],
                                  mode='lines', name='Avg Census',
                                  hovertemplate="%{y:,.0f}<extra></extra>"), 
                         row=2, col=1)
            
            fig.update_layout(
                height=600,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode='x unified'
            )
        else:
            fig = make_subplots(rows=3, cols=1,
                             subplot_titles=('MDS Census', 'Total Nurse HPRD', 'Contract %'),
                             vertical_spacing=0.1)
            
            fig.add_trace(go.Scatter(x=df['date'], y=df['Average_Daily_Census'],
                                  mode='lines', name='MDS Census',
                                  hovertemplate="%{y:,.0f}<extra></extra>"), 
                         row=1, col=1)
            
            fig.add_trace(go.Scatter(x=df['date'], y=df['Total_Nurse_HPRD'],
                                  mode='lines', name='Total HPRD',
                                  hovertemplate="%{y:.2f}<extra></extra>"), 
                         row=2, col=1)
            
            fig.add_trace(go.Scatter(x=df['date'], y=df['Contract_Percentage'],
                                  mode='lines', name='Contract %',
                                  hovertemplate="%{y:.1f}%<extra></extra>"), 
                         row=3, col=1)
            
            fig.update_layout(
                height=700,
                showlegend=False,
                margin=dict(l=20, r=20, t=40, b=20),
                hovermode='x unified'
            )
        
        # Minimal axis updates
        for i in range(1, 4 if view_mode == "Desktop" else 3):
            fig.update_xaxes(
                tickangle=45,
                tickformat="%Y Q%q",
                dtick="3M",
                row=i,
                col=1
            )
        
        return fig
    except Exception as e:
        st.error(f"Error creating plot: {str(e)}")
        return go.Figure()

def main():
    st.title("PBJ National Staffing Dashboard")

    try:
        # Initialize database
        conn = init_db()
        
        # Load the data
        df = load_data()
        
        # Display key metrics
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else None
        
        # Create metrics display
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Total Facilities",
                f"{latest['Average_Daily_Census']:,.0f}",
                f"{latest['Average_Daily_Census'] - prev['Average_Daily_Census']:,.0f}" if prev is not None else None
            )
        
        with col2:
            st.metric(
                "MDS Census",
                f"{latest['Average_Daily_Census']:,.0f}",
                f"{latest['Average_Daily_Census'] - prev['Average_Daily_Census']:,.0f}" if prev is not None else None
            )
        
        with col3:
            st.metric(
                "Total Nurse HPRD",
                f"{latest['Total_Nurse_HPRD']:.2f}",
                f"{latest['Total_Nurse_HPRD'] - prev['Total_Nurse_HPRD']:.2f}" if prev is not None else None
            )
        
        with col4:
            st.metric(
                "RN HPRD",
                f"{latest['RN_HPRD']:.2f}",
                f"{latest['RN_HPRD'] - prev['RN_HPRD']:.2f}" if prev is not None else None
            )
        
        with col5:
            st.metric(
                "Contract %",
                f"{latest['Contract_Percentage']:.1f}%",
                f"{latest['Contract_Percentage'] - prev['Contract_Percentage']:.1f}%" if prev is not None else None
            )
        
        # Add facility filter
        st.sidebar.header("Filters")
        facility_id = st.sidebar.text_input("Facility ID (PROVNUM)")
        
        # Date range filter
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Get filtered data
        filtered_data = get_filtered_data(
            conn,
            start_date=date_range[0] if len(date_range) > 0 else None,
            end_date=date_range[1] if len(date_range) > 1 else None,
            provnum=facility_id if facility_id else None
        )
        
        # Create two columns for the charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Create staffing chart
            fig_staffing = go.Figure()
            
            # Add Total HPRD line
            fig_staffing.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Total_Nurse_HPRD'],
                    name="Total Nurse HPRD",
                    line=dict(color='#ff7f0e', width=2),
                    mode='lines+markers',
                    marker=dict(size=8),
                    showlegend=False
                )
            )
            
            # Add horizontal reference line for 2001 study
            fig_staffing.add_hline(
                y=4.10,
                line_dash="dash",
                line_color="green",
                annotation_text="4.1 HPRD: Standard Recommended in 2001 Federal Study",
                annotation_position="top right",
                annotation=dict(font=dict(size=12))
            )
            
            # Update layout for staffing chart
            fig_staffing.update_layout(
                title=dict(
                    text="US Nursing Home Staffing Levels (2017-2024)",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="",
                yaxis_title="Hours Per Resident Day (HPRD)",
                hovermode="x unified",
                showlegend=False,
                yaxis=dict(
                    range=[3.6, 4.1],
                    showgrid=True,
                    gridcolor='rgba(211, 211, 211, 0.3)',
                    gridwidth=1,
                    zeroline=False,
                    tickformat=".1f",
                    title=dict(
                        text="Hours Per Resident Day (HPRD)",
                        standoff=20,
                        font=dict(size=12)
                    ),
                    ticklen=5,
                    tickwidth=1,
                    tickfont=dict(size=10)
                ),
                xaxis=dict(
                    range=[pd.Timestamp('2017-01-01'), pd.Timestamp('2024-12-31')],
                    tickangle=45,
                    showgrid=False,
                    zeroline=False
                ),
                margin=dict(b=5)
            )
            
            st.plotly_chart(fig_staffing, use_container_width=True)
        
        with col2:
            # Create census chart
            fig_census = go.Figure()
            
            # Add MDS Census line
            fig_census.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['Average_Daily_Census'] / 1000000,  # Convert to millions
                    name="MDS Census",
                    line=dict(color='#1f77b4', width=2),
                    mode='lines+markers',
                    marker=dict(size=8),
                    showlegend=False
                )
            )
            
            # Update layout for census chart
            fig_census.update_layout(
                title=dict(
                    text="MDS Census Over Time (2017-2024)",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title="",
                yaxis_title="MDS Census (Millions)",
                hovermode="x unified",
                showlegend=False,
                yaxis=dict(
                    tickformat=".2f",
                    showgrid=True,
                    gridcolor='rgba(211, 211, 211, 0.3)',
                    gridwidth=1,
                    zeroline=False
                ),
                xaxis=dict(
                    range=[pd.Timestamp('2017-01-01'), pd.Timestamp('2024-12-31')],
                    tickangle=45,
                    showgrid=False,
                    zeroline=False
                ),
                margin=dict(b=5)
            )
            
            st.plotly_chart(fig_census, use_container_width=True)
        
        # Add footer
        st.markdown("<div style='text-align: center; margin-top: -30px; font-size: 0.8em;'>320 Consulting | Source: CMS PBJ Data</div>", unsafe_allow_html=True)
        
        # Display filtered data table
        st.subheader("Facility Data")
        if not filtered_data.empty:
            st.dataframe(filtered_data)
        else:
            st.info("No data available for the selected filters.")

    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 