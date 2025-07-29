import streamlit as st

# Set page config - this must be the first Streamlit command
st.set_page_config(page_title="Premium | PBJ Nursing Home Staffing Dashboard", page_icon="pbj_favicon.png", layout="wide")

# Custom CSS for premium styling
st.markdown("""
    <style>
    .premium-header {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
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

# Replace the introduction paragraph at the top of the Premium page
st.markdown('''
<a href="https://www.320insight.com/" target="_blank" style="font-weight: bold; text-decoration: none; color: inherit;"><b>320 Consulting</b></a> offers custom reports with full breakdowns of all nurse and non-nurse positions, staffing trends over time (including daily staffing data), ownership data, citation histories, and comparisons by geography or any category you need. These reports and analyses are built to support your case, investigation, or advocacy work. Reports include tailored data visualizations, interactive tables, and in-depth analyses to help you uncover patterns and build evidence.

<a href="mailto:eric@320insight.com" style="text-decoration: none; font-size: 1.08em;">📧 eric@320insight.com</a>
''', unsafe_allow_html=True)

st.markdown(
    '<p><i>Journalists: Working on a story? Happy to help (no charge).</i></p>',
    unsafe_allow_html=True
)

# Premium Features
st.markdown('''

<div style="margin-bottom: 32px; padding-bottom: 8px; border-bottom: 1.5px solid #e3eaf3;"></div>

<div style="background: #fafdff; border: 1.5px solid #e3eaf3; border-radius: 12px; padding: 24px 24px 18px 24px; margin-bottom: 32px;">
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

<div style="background: #fafdff; border: 1.5px solid #e3eaf3; border-radius: 12px; padding: 24px 24px 18px 24px; margin-bottom: 32px;">
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

<div style="background: #fafdff; border: 1.5px solid #e3eaf3; border-radius: 12px; padding: 24px 24px 18px 24px; margin-bottom: 32px;">
<h3 style="margin-top: 0;">Ownership Group Analysis</h3>
Trace staffing and performance across affiliated facilities.
<ul>
<li>Affiliated entity identification</li>
<li>Cross-facility staffing patterns</li>
<li>Ownership group performance metrics</li>
<li>Historical ownership changes</li>
</ul>
<b><i>Example: Investigate how a chain’s staffing changed in the months before bankruptcy.</i></b>
</div>

<div style="background: #fafdff; border: 1.5px solid #e3eaf3; border-radius: 12px; padding: 24px 24px 18px 24px; margin-bottom: 16px;">
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
    st.markdown("""
        <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
            <p>Source: <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" style="color: #1E88E5; text-decoration: none;">CMS Payroll-Based Journal Data, 2017-2024</a></p>
            <p>By <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">320 Consulting LLC</a></p>
            <p><a href="/About" target="_self">About the Dashboard</a> | <a href="/Premium" target="_self">Premium</a></p>
        </div>
    """, unsafe_allow_html=True)

# Add back to dashboard button
if st.button("← Back to Dashboard", key="back_to_dashboard_premium"):
    st.switch_page("PBJ_Dashboard.py")

# Add footer at the end of the page
display_footer() 