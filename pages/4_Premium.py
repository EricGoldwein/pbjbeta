import streamlit as st
import base64

# Set page config - this must be the first Streamlit command
st.set_page_config(page_title="Premium | PBJ Nursing Home Staffing Dashboard", page_icon="pbj_favicon.png", layout="wide")

# Load favicon data
def load_pbj_favicon():
    """Load PBJ favicon data for use in footer"""
    try:
        with open('pbj_favicon.png', 'rb') as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        print("Warning: pbj_favicon.png not found")
        return ""
    except Exception as e:
        print(f"Warning: Error loading favicon: {e}")
        return ""

favicon_data = load_pbj_favicon()

# Custom CSS for Home button styling
st.markdown("""
<style>
.home-button {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    color: #1565c0;
    border: 1px solid #90caf9;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    transition: all 0.2s ease;
    margin-top: 8px;
    margin-bottom: 4px;
}
.home-button:hover {
    background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%);
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}
.home-button:active {
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Add Home button to top left
col1, col2, col3 = st.columns([1, 8, 1])
with col1:
    st.markdown(f"""
        <div style="margin-top: 8px; margin-bottom: 8px;">
            <a href="PBJ_Dashboard.py" style="display: inline-block; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); color: #1565c0; border: 1px solid #90caf9; border-radius: 6px; padding: 6px 12px; font-size: 13px; font-weight: 500; text-decoration: none; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: all 0.2s ease;">
                <img src="data:image/png;base64,{favicon_data}" style="width: 16px; height: 16px; margin-right: 4px; vertical-align: middle;"> Home
            </a>
        </div>
    """, unsafe_allow_html=True)

# Custom CSS for premium styling
st.markdown("""
    <style>
    .premium-header {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }
    .premium-feature {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #1E88E5;
        max-width: 800px;
        margin-left: 0;
        margin-right: auto;
    }
    .premium-feature h3 {
        color: #1E88E5;
        margin-top: 0;
    }
    .contact-section {
        background-color: #e3f2fd;
        padding: 2rem;
        border-radius: 8px;
        margin-top: 2rem;
        max-width: 800px;
        margin-left: 0;
        margin-right: auto;
    }
    .data-source {
        font-size: 0.9em;
        color: #666;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #eee;
        max-width: 800px;
        margin-left: 0;
        margin-right: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Premium Header
st.markdown("""
    <div class="premium-header">
        <h1>320 Premium</h1>
        <p style="font-size: 1.2em; margin-bottom: 0;">Dig deeper into nursing home data</p>
    </div>
""", unsafe_allow_html=True)

# Revised Introduction
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

# Features
st.markdown('''
<div class="premium-feature">
<h3 style="margin-top: 0;">Custom Dashboards</h3>
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

<div class="premium-feature">
<h3 style="margin-top: 0;">Daily Staffing Analysis</h3>
Access detailed daily staffing data for any facility since 2017.
<ul>
<li>Daily staffing levels for all positions</li>
<li>Anomaly detection for unusual patterns</li>
<li>Historical trend analysis</li>
<li>Custom date range comparisons</li>
</ul>
<b><i>Example: Spot weekend staffing dips or compare RN levels before and after a major inspection.</i></b>
</div>

<div class="premium-feature">
<h3 style="margin-top: 0;">Comprehensive Staffing Reports</h3>
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

<div class="premium-feature">
<h3 style="margin-top: 0;">Ownership Group Analysis</h3>
Trace staffing and performance across affiliated facilities.
<ul>
<li>Affiliated entity identification</li>
<li>Cross-facility staffing patterns</li>
<li>Ownership group performance metrics</li>
<li>Historical ownership changes</li>
</ul>
<b><i>Example: Investigate how a chain's staffing changed in the months before bankruptcy.</i></b>
</div>

<div class="premium-feature">
<h3 style="margin-top: 0;">Citations Analysis</h3>
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
    <div class="contact-section" style="padding: 1rem;">
        <p>Reach out to request a custom report or talk through your project: <a href="mailto:eric@320insight.com" class="contact-link">eric@320insight.com</a></p>
    </div>
""", unsafe_allow_html=True)

def display_footer():
    """Display a consistent footer across all pages."""
    st.markdown(f"""
        <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
            <p>Source: <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" style="color: #1E88E5; text-decoration: none;">CMS Payroll-Based Journal Data, 2017-2025</a></p>
            <p>By <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">320 Consulting LLC</a></p>
            <p><a href="/About" target="_self" style="text-decoration: none;">About the Dashboard</a> | <a href="/Premium" target="_self" style="text-decoration: none;">Premium</a> | <a href="https://www.320insight.com/phoebe" target="_blank" style="text-decoration: none;">Phoebe J</a></p>
        </div>
    """, unsafe_allow_html=True)

# Add footer at the end of the page
display_footer() 