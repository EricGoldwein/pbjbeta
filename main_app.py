import streamlit as st
import base64
from utils.date_utils import get_latest_data_periods

def load_pbj_favicon():
    """Load PBJ favicon data for use in floating action button"""
    try:
        with open('pbj_images/pbj_favicon.png', 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

# Set page config early for Render deployment
st.set_page_config(
    page_title="PBJ Nursing Home Staffing Dashboard by 320", 
    page_icon="pbj_images/pbj_favicon.png", 
    layout="wide"
)

# Add SEO meta tags
st.markdown("""
    <!-- SEO Meta Tags -->
    <meta name="description" content="Explore staffing trends across 15,000+ U.S. nursing homes with CMS payroll-based journal data.">
    <meta name="keywords" content="PBJ, nursing home staffing, HPRD, healthcare staffing, nursing home compliance, healthcare analytics, nursing home data, staffing metrics">
    <meta name="author" content="320 Consulting">
    <meta name="robots" content="index, follow">
    <meta name="language" content="English">
    <meta name="revisit-after" content="7 days">
    
    <!-- Open Graph / Facebook -->
    <meta property="og:type" content="website">
    <meta property="og:url" content="https://pbjdashboard.com/">
    <meta property="og:title" content="PBJ Nursing Home Staffing Dashboard by 320 Consulting">
    <meta property="og:description" content="Explore staffing trends across 15,000+ U.S. nursing homes with CMS payroll-based journal data.">
    <meta property="og:image" content="https://pbjdashboard.com/pbj.seo.png">
    <meta property="og:image:width" content="1200">
    <meta property="og:image:height" content="630">
    <meta property="og:site_name" content="PBJ Nursing Home Staffing Dashboard">
    
    <!-- Twitter -->
    <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:url" content="https://pbjdashboard.com/">
    <meta name="twitter:title" content="PBJ Nursing Home Staffing Dashboard by 320 Consulting">
    <meta name="twitter:description" content="Explore staffing trends across 15,000+ U.S. nursing homes with CMS payroll-based journal data.">
    <meta name="twitter:image" content="https://pbjdashboard.com/pbj.seo.png">
    
    <!-- Additional Meta Tags -->
    <meta name="theme-color" content="#1565C0">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    
    <!-- Canonical URL -->
    <link rel="canonical" href="https://pbjdashboard.com/">
""", unsafe_allow_html=True)

def dashboard_page():
    """Main dashboard content"""
    # Import and run the main dashboard functionality from PBJ_Dashboard
    # This is a placeholder - you would need to refactor PBJ_Dashboard.py 
    # to extract the main dashboard logic into functions
    
    import pandas as pd
    import plotly.graph_objects as go
    from datetime import datetime
    import os
    import re
    from plotly.subplots import make_subplots
    import duckdb
    from typing import Dict, Optional, List, Tuple, Any
    import math
    import io
    import numpy as np
    from decimal import Decimal, ROUND_HALF_UP
    
    # This is where you would call the main dashboard functions
    # For now, let's include a simple version
    st.title("🏥 PBJ Nursing Home Staffing Dashboard")
    st.markdown("**Explore staffing trends across 15,000+ U.S. nursing homes with CMS payroll-based journal data.**")
    
    # Create search options
    search_type = st.selectbox(
        "Search by:",
        ["Facility", "State", "Ownership Group"],
        index=0
    )
    
    if search_type == "Facility":
        st.text_input("Enter facility name or provider number:")
        st.info("📍 **Tip:** Try searching for facility names like 'Genesis' or provider numbers like '015009'")
        
    elif search_type == "State":
        state_options = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
                        'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 
                        'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 
                        'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 
                        'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 
                        'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
                        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West Virginia', 
                        'Wisconsin', 'Wyoming']
        st.selectbox("Select a state:", state_options)
        
    elif search_type == "Ownership Group":
        st.text_input("Enter ownership group name:")
        st.info("📍 **Tip:** Try searching for chains like 'Genesis Healthcare' or 'Brookdale'")
    
    # Placeholder for charts and data
    st.subheader("Sample Dashboard Content")
    st.info("This is a demonstration of the top navigation structure. The full dashboard functionality would be integrated here.")
    
    # Sample chart
    import plotly.graph_objects as go
    
    # Create sample data
    quarters = ['2023Q1', '2023Q2', '2023Q3', '2023Q4', '2024Q1']
    national_hprd = [3.2, 3.3, 3.1, 3.4, 3.3]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=quarters,
        y=national_hprd,
        mode='lines+markers',
        name='National Average HPRD',
        line=dict(color='#1565C0', width=3)
    ))
    
    fig.update_layout(
        title='Sample: National Nurse Staffing Trends',
        xaxis_title='Quarter',
        yaxis_title='Hours Per Resident Day (HPRD)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

def about_page():
    """About page content"""
    st.markdown(f"""
    <div style="background: #f5f8fd; border-radius: 10px; padding: 2.2rem 2.5rem 1.5rem 2.5rem; margin-bottom: 2.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
        <div style='text-align: center; margin-bottom: 1.2em;'>
            <span style='font-size:2.3em; font-weight:800; color:#1769aa; letter-spacing:0.01em; line-height:1.1;'>PBJ Nursing Home Staffing Dashboard</span><br>
            <span style='font-size:1.15em; color:#7a869a; font-weight:400;'>by 320 Consulting</span>
        </div>

    ### Why this matters
    Staffing data is a key indicator of nursing home quality, revealing how much care residents receive and what resources facilities commit. Yet most public data shows only the latest quarter, offering a narrow and incomplete view. This dashboard stitches together **nine years of federal CMS staffing files—billions of data points from payroll-based journal (PBJ) submissions—into interactive visualizations**, so you can see how staffing has changed over time and bring data-driven context to what's happening inside the 15,000 nursing homes across the U.S.

    ### Who it helps  
    * **Attorneys** – identify staffing patterns and trends that may support negligence cases, regulatory violations, or quality of care claims. Access historical data to demonstrate chronic understaffing, seasonal variations, or ownership-related staffing deficiencies.
    * **Journalists** – plug numbers and data visualizations into a nursing home investigation or ownership-focused report without wrangling raw CSVs.
    * **Advocates & families** – see how a nursing home stacks up over time for residents and loved ones.
    * **Providers** – use historical staffing data to identify gaps, benchmark performance, and support quality improvement efforts.

    *Premium – custom reports with daily, position-level analysis and data visualizations tied to citations and inspections.*

    ### What you can explore
    | View | Data you get |
    |------|--------------|
    | **National / State** | Nurse staffing hours per resident day (HPRD), contract staff %, census — every quarter since 2017 |
    | **Facility** | A nursing home's quarterly staffing, contract, and census data; ratings and risk indicators |
    | **Ownership Group** | Essential data on any chain and its facilities (e.g., **Genesis** → 215 facilities in 19 states, 2.3-star average) |

    ### Under the hood  
    * **Payroll-Based Journal (PBJ) Staffing Data** – {get_latest_data_periods()['quarter_count']} quarters of daily data, aggregated for clarity  
    * **CMS Provider Info** – 5-star ratings, enforcement data, and other key indicators (July 2025 & June 2025)  
    * **CMS Affiliated Entity** – Selected quality and performance metrics for groups of nursing homes sharing common owners, officers, or entities (July 2025)
    * **CMS Citations (Premium)** - Citation data and inspection reports, categorized by date, type, severity, and more. 

    ### Quick tour  
    1. Start search by selecting **Facility**, **Ownership**, or **State**.
    2. Access state, facility, and ownership-level data and view data visualizations to spot trends over time.  
    3. Click **Export** for ready-to-use PNGs.

    ### Try Phoebe J
    Check out [Phoebe J, the PBJ nursing home staffing data assistant (in training!)](https://www.320insight.com/phoebe) for quick PBJ data searches by state or nursing home.

    ### Digging deeper?  
    Daily staffing data and analysis, role-specific hours (Nurse and Non-Nurse), weekend vs. weekday splits, and citation-linked timelines live in the premium layer.  
    Email **eric@320insight.com** for requests. Journalists: If you're working on a story, I'm happy to share data or walk you through it.

    *Built by 320 Consulting. Feedback welcome (tell me what's broken!).*
    </div>
    """, unsafe_allow_html=True)

    # Section divider
    st.markdown("<hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>", unsafe_allow_html=True)

    # Methodology section
    st.markdown("""
    <div style="background: #f5f8fd; border-radius: 10px; padding: 2.2rem 2.5rem 1.5rem 2.5rem; margin-bottom: 2.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
        <div>
            <h2 style='font-size:1.5em; font-weight:700; color:#1769aa; margin-bottom:0.7em;'>Methodology</h2>
            <div style='font-size:1.08em; color:#222; font-weight:400;'>
                This Nursing Home Staffing Dashboard uses <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank">CMS Payroll-Based Journal (PBJ) data</a> from 2017 to 2025, covering all nursing positions, including contract staff. CMS first published PBJ data in 2017. It also uses <a href="https://data.cms.gov/provider-data/dataset/4pq5-n9py" target="_blank">Provider Information</a> (July 2025, June 2025), <a href="https://data.cms.gov/quality-of-care/nursing-home-affiliated-entity-performance-measures/data" target="_blank">Affiliated Entity</a> (July 2025), and <a href="https://www.macpac.gov/publication/state-policies-related-to-nursing-facility-staffing/" target="_blank">MACPAC State Staffing Standards</a> (2022) datasets.
            </div>
            <hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>
            <h2 style='font-size:1.3em; font-weight:700; color:#1769aa; margin-bottom:0.5em;'>Staffing Categories</h2>
            <div style='font-size:1.08em; color:#222; font-weight:400;'>
                Total nurse staff includes:
                <ul style='margin-top:0.5em; margin-bottom:0.5em;'>
                    <li>Registered Nurse (RN)</li>
                    <li>RN Director of Nursing (DON)</li>
                    <li>RN Admin</li>
                    <li>Licensed Practical Nurse (LPN)</li>
                    <li>LPN Admin</li>
                    <li>Certified Nursing Assistant (CNA)</li>
                    <li>Nurse Aide in Training</li>
                    <li>Medication Aide/Technician</li>
                </ul>
            </div>
            <hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>
            <div style='font-size:1.13em; font-weight:700; color:#1769aa; margin-bottom:0.5em; margin-top:1.2em;'>Metrics Explained</div>
            <div style='font-size:1.08em; color:#222; font-weight:400; line-height:1.45;'>
                <b>Total Nurse Hours Per Resident Day (HPRD):</b> Total nurse staff hours per resident per day.*<br>
                <b>Direct Care (excl. Admin, DON):</b> Hours per resident day for direct care staff only (RN, LPN, CNA, NAtrn, MedAide), excluding administrative and supervisory roles.<br>
                <b>Contract Staff Percentage:</b> Percentage of nurse staff hours provided by contract staff.<br>
                <b>Census:</b> Average number of residents in facility or state during the reporting period.<br>
                <b>Ownership Change:</b> Indicates facility ownership changed in the last 12 months.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # About 320 Consulting Section
    st.markdown("""
    <div style="background: #f5f8fd; border-radius: 10px; padding: 2.2rem 2.5rem 1.5rem 2.5rem; margin-bottom: 2.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
        <div>
            <h2 style='font-size:1.5em; font-weight:700; color:#1769aa; margin-bottom:0.7em;'>About 320 Consulting</h2>
            <div style='font-size:1.08em; color:#222; font-weight:400;'>
                <b><a href="https://www.320insight.com/" target="_blank" style="color:#1769aa; text-decoration:none;">320 Consulting</a></b> is led by Eric Goldwein, MPH, a data consultant with expertise in nursing home staffing. His work on nursing home data has been published in the <i>Journal of the American Geriatrics Society</i>, and he has presented at national conferences hosted by the National Association of Medicaid Fraud Control Units, Consumer Voice, the American Society on Aging, and the NYS Long Term Care Ombudsman Program. He previously served as policy director at the Long Term Care Community Coalition.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def premium_page():
    """Premium page content"""
    # Premium Header
    st.markdown("""
        <div style="background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%); color: white; padding: 1rem 2rem; border-radius: 10px; margin-bottom: 0.5rem;">
            <h1>320 Premium</h1>
            <p style="font-size: 1.2em; margin-bottom: 0;">Dig deeper into nursing home data</p>
        </div>
    """, unsafe_allow_html=True)

    # Introduction
    st.markdown('''
    <div style="margin-top: 0.5rem;">
    <a href="https://www.320insight.com/" target="_blank" style="font-weight: bold; text-decoration: none; color: inherit;"><b>320 Consulting</b></a> offers custom dashboards and reports with full breakdowns of all nurse and non-nurse positions, staffing trends over time (including daily staffing data), ownership data, citation histories, and comparisons by geography or any category you need. These reports and analyses are designed to support your case, investigation, or advocacy work. Deliverables include tailored data visualizations, interactive tables, and in-depth analyses to help you uncover patterns and build evidence.

    <a href="mailto:eric@320insight.com" style="text-decoration: none; font-size: 1.08em;">📧 eric@320insight.com</a>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown(
        '<p><i>Journalists: Working on a story? Happy to help (no charge).</i></p>',
        unsafe_allow_html=True
    )

    # Divider
    st.markdown('''
    <div style="margin-bottom: 32px; padding-bottom: 8px; border-bottom: 1.5px solid #e3eaf3;"></div>
    ''', unsafe_allow_html=True)

    # Features sections
    st.markdown('''
    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #1E88E5;">
    <h3 style="color: #1E88E5; margin-top: 0;">Custom Dashboards</h3>
    Interactive dashboards tailored to your needs, built to spotlight the issues and geographies most relevant to your work. Examples include:
    <ul>
    <li><b>Mississippi Staffing Dashboard:</b> Track staffing levels across all nursing homes in the state.</li>
    <li><b>Region 5 Citations:</b> Explore survey and enforcement patterns across the Chicago CMS region.</li>
    <li><b>Tennessee Financials:</b> Analyze cost reports and financial data for facilities statewide.</li>
    <li><b>Medical Director Focus:</b> Drill into staffing and oversight trends tied to physician leadership.</li>
    <li><b>RN Compliance:</b> Monitor daily RN coverage to identify gaps in the federal 8-hour rule.</li>
    </ul>
    <b><i>Example: Build a dashboard showing weekend RN compliance in California while linking citation history and ownership data.</i></b>
    </div>

    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #1E88E5;">
    <h3 style="color: #1E88E5; margin-top: 0;">Daily Staffing Analysis</h3>
    Access detailed daily staffing data for any facility since 2017.
    <ul>
    <li>Daily staffing levels for all positions</li>
    <li>Anomaly detection for unusual patterns</li>
    <li>Historical trend analysis</li>
    <li>Custom date range comparisons</li>
    </ul>
    <b><i>Example: Spot weekend staffing dips or compare RN levels before and after a major inspection.</i></b>
    </div>

    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #1E88E5;">
    <h3 style="color: #1E88E5; margin-top: 0;">Comprehensive Staffing Reports</h3>
    Break down staffing by role, from RNs to social workers.
    <ul>
    <li>Administrators and DONs</li>
    <li>RNs, LPNs, CNAs</li>
    <li>Physical, Occupational, and Speech Therapists</li>
    <li>Social Workers and Activities Staff</li>
    <li>Contract staff utilization</li>
    </ul>
    <b><i>Example: Build a full staffing profile of a facility cited for resident neglect.</i></b>
    </div>

    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #1E88E5;">
    <h3 style="color: #1E88E5; margin-top: 0;">Ownership Group Analysis</h3>
    Trace staffing and performance across affiliated facilities.
    <ul>
    <li>Affiliated entity identification</li>
    <li>Cross-facility staffing patterns</li>
    <li>Ownership group performance metrics</li>
    <li>Historical ownership changes</li>
    </ul>
    <b><i>Example: Investigate how a chain's staffing changed in the months before bankruptcy.</i></b>
    </div>

    <div style="background-color: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin-bottom: 1rem; border-left: 4px solid #1E88E5;">
    <h3 style="color: #1E88E5; margin-top: 0;">Citations Analysis</h3>
    Analyze inspection reports and link citations to staffing.
    <ul>
    <li>Form CMS-2567 data integration</li>
    <li>Citation summaries and trends</li>
    <li>Staffing correlation analysis</li>
    <li>Historical citation patterns</li>
    </ul>
    <b><i>Example: Identify whether facilities with repeated infection-control citations had chronic CNA shortages.</i></b>
    </div>
    ''', unsafe_allow_html=True)

    # Contact Section
    st.markdown("""
        <div style="background-color: #e3f2fd; padding: 2rem; border-radius: 8px; margin-top: 2rem;">
            <p>Reach out to request a custom report or talk through your project: <a href="mailto:eric@320insight.com">eric@320insight.com</a></p>
        </div>
    """, unsafe_allow_html=True)

def display_footer():
    """Display a consistent footer across all pages."""
    st.markdown(f"""
        <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
            <p>Source: <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" style="color: #1E88E5; text-decoration: none;">CMS Payroll-Based Journal Data, 2017-2025</a></p>
            <p>By <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">320 Consulting LLC</a></p>
            <p><a href="https://www.320insight.com/phoebe" target="_blank" style="text-decoration: none;">Phoebe J</a></p>
        </div>
    """, unsafe_allow_html=True)

# Create navigation pages using Streamlit's native navigation
dashboard = st.Page(dashboard_page, title="Dashboard", icon="🏠")
about = st.Page(about_page, title="About", icon="ℹ️") 
premium = st.Page(premium_page, title="Premium", icon="💎")

# Set up navigation with the pages
pg = st.navigation([dashboard, about, premium])

# Run the navigation
pg.run()

# Display footer on all pages
display_footer()


