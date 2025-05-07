import streamlit as st

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="320 Premium Services",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add to sidebar
with st.sidebar:
    st.title("Premium Services")
    st.markdown("""
        Access our premium features to get deeper insights into nursing home staffing data.
    """)

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
        <h1>320 Premium Services</h1>
        <p style="font-size: 1.2em; margin-bottom: 0;">Supporting resident advocacy through data transparency</p>
    </div>
""", unsafe_allow_html=True)

# Introduction
st.markdown("""
    ### Who We Serve
    320 Consulting provides specialized data analysis and reporting services for attorneys, advocates, journalists, 
    researchers, and consumers who are working to support nursing home residents. Our tools help you understand 
    staffing patterns, identify potential issues, and make data-driven decisions to bolster transparency and resident care.
""")

# Premium Features
st.markdown("### Premium Services")

# Feature 1: Daily Staffing Lookup
st.markdown("""
    <div class="premium-feature">
        <h3>📊 Daily Staffing Analysis</h3>
        <p>Access detailed daily staffing data for any facility since 2017, including:</p>
        <ul>
            <li>Daily staffing levels for all positions</li>
            <li>Anomaly detection for unusual staffing patterns</li>
            <li>Historical trend analysis</li>
            <li>Custom date range comparisons</li>
            <li>Shareable data visualizations and tables</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Feature 2: Custom Staffing Reports
st.markdown("""
    <div class="premium-feature">
        <h3>👥 Comprehensive Staffing Reports</h3>
        <p>Get detailed reports on every position, including:</p>
        <ul>
            <li>Administrators and DONs</li>
            <li>RNs, LPNs, and CNAs</li>
            <li>Physical, Occupational, and Speech Therapists</li>
            <li>Social Workers and Activities Staff</li>
            <li>Contract staff utilization</li>
            <li>Custom shareable data visualizations</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Feature 3: Ownership Analysis
st.markdown("""
    <div class="premium-feature">
        <h3>🏢 Ownership Group Analysis</h3>
        <p>Understand facility ownership patterns and trends:</p>
        <ul>
            <li>Affiliated entity identification</li>
            <li>Cross-facility staffing patterns</li>
            <li>Ownership group performance metrics</li>
            <li>Historical ownership changes</li>
            <li>Custom shareable ownership reports</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Feature 4: Citations Analysis
st.markdown("""
    <div class="premium-feature">
        <h3>📋 Citations Analysis</h3>
        <p>Comprehensive analysis of facility citations:</p>
        <ul>
            <li>Form CMS-2567 data integration</li>
            <li>Citation summaries and trends</li>
            <li>Staffing correlation analysis</li>
            <li>Historical citation patterns</li>
            <li>Custom shareable citation reports</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# Contact Section
st.markdown("""
    <div class="contact-section">
        <h3>Get Started with Premium Services</h3>
        <p>For custom reports and additional data analysis services, contact us at:</p>
        <p style="font-size: 1.2em; font-weight: bold;">📧 <a href="mailto:eric@320insight.com">eric@320insight.com</a></p>
    </div>
""", unsafe_allow_html=True)

# Data Source
st.markdown("""
    <div class="data-source">
        <p><strong>Data Source:</strong> Our analysis is primarily based on CMS Payroll-Based Journal (PBJ) data, 
        supplemented with additional CMS datasets and proprietary analysis tools.</p>
    </div>
""", unsafe_allow_html=True) 