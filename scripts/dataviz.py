import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Nursing Home Data Visualization",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Nursing Home Staffing and Census Trends")

# Load data
@st.cache_data
def load_data():
    # Load the national PBJ metrics data
    df = pd.read_csv('national_pbj_metrics.csv')
    
    # Convert Quarter to datetime
    df['Year'] = df['Quarter'].str[:4].astype(int)
    df['Quarter_Num'] = df['Quarter'].str[-1].astype(int)
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + 
                               ((df['Quarter_Num'] * 3) - 2).astype(str) + '-01')
    
    # Filter for 2017-2024
    df = df[(df['Date'].dt.year >= 2017) & (df['Date'].dt.year <= 2024)]
    
    # Convert MDS Census to millions
    df['MDS Census'] = df['Total_MDScensus'] / 1000000
    
    return df

# Load the data
df = load_data()

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
            line=dict(color='#ff7f0e', width=2),  # Orange
            mode='lines+markers',  # Add markers
            marker=dict(size=8),  # Size of the dots
            showlegend=False  # Hide legend
        )
    )
    
    # Add horizontal reference line for 2001 study
    fig_staffing.add_hline(
        y=4.10,
        line_dash="dash",
        line_color="green",
        annotation_text="4.1 HPRD: Standard Recommended in 2001 Federal Study",
        annotation_position="top right",
        annotation=dict(
            font=dict(size=12)  # Adjust font size
        )
    )
    
    # Add x-axis line
    fig_staffing.add_hline(
        y=3.6,
        line=dict(color='lightgray', width=1),
        showlegend=False
    )
    
    # Update layout for staffing chart
    fig_staffing.update_layout(
        title=dict(
            text="US Nursing Home Staffing Levels (2017-2024)",
            x=0.5,  # Center title
            xanchor='center'
        ),
        xaxis_title="",
        yaxis_title="Hours Per Resident Day (HPRD)",
        hovermode="x unified",
        showlegend=False,
        yaxis=dict(
            range=[3.6, 4.1],  # Set max to 4.1
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)',  # More faded light gray
            gridwidth=1,
            zeroline=False,
            tickformat=".1f",  # Show one decimal place consistently
            title=dict(
                text="Hours Per Resident Day (HPRD)",
                standoff=20,  # Increased standoff
                font=dict(size=12)  # Slightly smaller font
            ),
            ticklen=5,  # Add some padding for ticks
            tickwidth=1,
            tickfont=dict(size=10)  # Slightly smaller tick font
        ),
        xaxis=dict(
            range=[pd.Timestamp('2017-01-01'), pd.Timestamp('2024-12-31')],  # Set x-axis range
            tickangle=45,  # Make year labels diagonal
            showgrid=False,
            zeroline=False
        ),
        margin=dict(b=5)  # Minimal bottom margin
    )
    
    st.plotly_chart(fig_staffing, use_container_width=True)
    st.markdown("<div style='text-align: center; margin-top: -30px; font-size: 0.8em;'>320 Consulting | Source: CMS PBJ Data</div>", unsafe_allow_html=True)

with col2:
    # Create census chart
    fig_census = go.Figure()
    
    # Add MDS Census line
    fig_census.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['MDS Census'] / 100,  # Divide by 100 for y-axis labels
            name="MDS Census",
            line=dict(color='#1f77b4', width=2),  # Blue
            mode='lines+markers',  # Add markers
            marker=dict(size=8),  # Size of the dots
            showlegend=False  # Hide legend
        )
    )
    
    # Update layout for census chart
    fig_census.update_layout(
        title=dict(
            text="MDS Census Over Time (2017-2024)",
            x=0.5,  # Center title
            xanchor='center'
        ),
        xaxis_title="",
        yaxis_title="MDS Census (Millions)",
        hovermode="x unified",
        showlegend=False,
        yaxis=dict(
            tickformat=".2f",  # Format y-axis ticks to show two decimal places
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)',  # More faded light gray
            gridwidth=1,
            zeroline=False
        ),
        xaxis=dict(
            range=[pd.Timestamp('2017-01-01'), pd.Timestamp('2024-12-31')],  # Set x-axis range
            tickangle=45,  # Make year labels diagonal
            showgrid=False,
            zeroline=False
        ),
        margin=dict(b=5)  # Minimal bottom margin
    )
    
    st.plotly_chart(fig_census, use_container_width=True)
    st.markdown("<div style='text-align: center; margin-top: -30px; font-size: 0.8em;'>320 Consulting | Source: CMS PBJ Data</div>", unsafe_allow_html=True)

# Add explanation
st.markdown("""
### About these Charts
- **Total Nurse HPRD**: Shows the average total nurse hours per resident day
- **MDS Census**: Shows the total census across all nursing homes (in millions)
- **Green Dashed Line**: 2001 Federal Study recommendation (4.1 HPRD)
- The charts allow for easy comparison of staffing levels and census trends over time
""") 