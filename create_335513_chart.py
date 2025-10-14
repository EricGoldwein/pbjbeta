#!/usr/bin/env python3
"""
Standalone script to create a chart for facility 335513 showing Total Nurse Staff HPRD
from April 1, 2022 to March 31, 2025 with NY minimum line at 3.50 HPRD.

For blog post use - creates a clean PNG chart.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP

def round_financial(value, decimals=2):
    """Apply financial rounding (ROUND_HALF_UP) to numeric values."""
    if pd.isna(value):
        return value
    return float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def main():
    print("Loading data for facility 335513...")
    
    # Load the data
    try:
        df = pd.read_csv('facility_335513_complete_data.csv')
        print(f"Loaded {len(df)} records")
    except FileNotFoundError:
        print("Error: facility_335513_complete_data.csv not found!")
        return
    
    # Convert WorkDate to datetime
    df['WorkDate'] = pd.to_datetime(df['WorkDate'], format='%Y%m%d')
    
    # Filter date range: April 1, 2022 to March 31, 2025
    start_date = datetime(2022, 4, 1)
    end_date = datetime(2025, 3, 31)
    
    df_filtered = df[(df['WorkDate'] >= start_date) & (df['WorkDate'] <= end_date)].copy()
    print(f"Filtered to {len(df_filtered)} records from {start_date.date()} to {end_date.date()}")
    
    if len(df_filtered) == 0:
        print("No data found in the specified date range!")
        return
    
    # Calculate Total Staff HPRD (same formula as dashboard)
    # Total HPRD includes ALL staff (RN + RNadmin + RNDON + LPN + LPNadmin + CNA + NAtrn + MedAide)
    df_filtered['Total_Staff_Hours'] = (
        df_filtered['Hrs_RN'] + 
        df_filtered['Hrs_RNadmin'] + 
        df_filtered['Hrs_RNDON'] + 
        df_filtered['Hrs_LPN'] + 
        df_filtered['Hrs_LPNadmin'] + 
        df_filtered['Hrs_CNA'] + 
        df_filtered['Hrs_NAtrn'] + 
        df_filtered['Hrs_MedAide']
    )
    
    df_filtered['Total_Staff_HPRD'] = (df_filtered['Total_Staff_Hours'] / df_filtered['MDScensus']).apply(lambda x: round_financial(x, 2))
    
    # Remove any invalid data points
    df_filtered = df_filtered.dropna(subset=['Total_Staff_HPRD', 'WorkDate'])
    df_filtered = df_filtered[df_filtered['Total_Staff_HPRD'] > 0]
    
    print(f"Final dataset: {len(df_filtered)} valid records")
    print(f"HPRD range: {df_filtered['Total_Staff_HPRD'].min():.2f} to {df_filtered['Total_Staff_HPRD'].max():.2f}")
    
    # Sort by date
    df_filtered = df_filtered.sort_values('WorkDate')
    
    # Debug: Check the datetime format
    print(f"Sample dates: {df_filtered['WorkDate'].head().tolist()}")
    print(f"Date type: {type(df_filtered['WorkDate'].iloc[0])}")
    print(f"Date range: {df_filtered['WorkDate'].min()} to {df_filtered['WorkDate'].max()}")
    
    # Get facility info
    facility_name = df_filtered['PROVNAME'].iloc[0]
    city_state = f"{df_filtered['CITY'].iloc[0]}, {df_filtered['STATE'].iloc[0]}"
    provnum = df_filtered['PROVNUM'].iloc[0]
    
    # Create the chart
    fig = go.Figure()
    
    # Convert dates to proper format for Plotly
    dates_for_plot = df_filtered['WorkDate'].dt.strftime('%Y-%m-%d')
    
    # Add the main HPRD line
    fig.add_trace(go.Scatter(
        x=dates_for_plot,
        y=df_filtered['Total_Staff_HPRD'],
        mode='lines',
        name='Total Nurse Staff HPRD',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{x|%B %d, %Y}</b><br>Total Staff HPRD: %{y:.2f}<extra></extra>'
    ))
    
    # Add NY minimum line at 3.50
    fig.add_hline(
        y=3.50,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text="New York Minimum (3.50 HPRD)",
        annotation_position="top right",
        annotation_font_size=12,
        annotation_font_color="black",
        annotation_bgcolor="rgba(255, 200, 200, 1.0)",
        annotation_borderpad=4
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f'<b>{facility_name} ({city_state})</b><br><sub>Staffing Levels by Day (April 2022 - March 2025)</sub><br><sub>Chain: Excelsior Care Group</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis={
            'type': 'date',
            'showgrid': True,
            'gridcolor': 'lightgray',
            'tickformat': '%b %Y',
            'dtick': 'M6',  # Show every 6 months
            'tickangle': 0,
            'title': None  # Remove x-axis title
        },
        yaxis={
            'title': 'Hours per Resident Day (HPRD)',
            'showgrid': True,
            'gridcolor': 'lightgray',
            'tickformat': '.1f'
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font={'family': 'Arial, sans-serif'},
        width=1200,
        height=600,
        margin=dict(l=80, r=80, t=120, b=80),
        showlegend=False,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    # Add source annotation
    fig.add_annotation(
        text="Source: CMS Payroll-Based Journal | Chart by 320 Consulting",
        xref="paper", yref="paper",
        x=1, y=-0.15,
        xanchor='right', yanchor='bottom',
        showarrow=False,
        font=dict(size=10, color="gray")
    )
    
    # Save as PNG
    filename = f"335513_total_staff_hprd_apr2022_mar2025.png"
    pio.write_image(fig, filename, format='png', scale=1, width=1200, height=600)  # Lower scale, explicit dimensions
    
    print(f"\nChart saved as: {filename}")
    print(f"Facility: {facility_name}")
    print(f"Location: {city_state}")
    print(f"Date range: {df_filtered['WorkDate'].min().strftime('%B %d, %Y')} to {df_filtered['WorkDate'].max().strftime('%B %d, %Y')}")
    print(f"Total days: {len(df_filtered):,}")
    print(f"Average HPRD: {df_filtered['Total_Staff_HPRD'].mean():.2f}")
    print(f"Min HPRD: {df_filtered['Total_Staff_HPRD'].min():.2f}")
    print(f"Max HPRD: {df_filtered['Total_Staff_HPRD'].max():.2f}")

if __name__ == "__main__":
    main()
