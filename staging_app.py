import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title="PBJ Dashboard - Staging",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple staging version
st.title("🏥 PBJ Nursing Home Dashboard")
st.subheader("Staging Environment")

st.info("🚧 This is a staging environment for testing deployment.")

# Basic metrics display
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Facilities", "15,000+", "↗️ 2%")

with col2:
    st.metric("Active States", "50", "↗️ 0%")

with col3:
    st.metric("Data Points", "1M+", "↗️ 5%")

with col4:
    st.metric("Last Updated", "Today", "↗️ 0%")

# Simple chart
st.subheader("Sample Data")
chart_data = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['HPRD', 'Census', 'Staffing']
)

st.line_chart(chart_data)

st.success("✅ Staging deployment is working!")
st.info("This lightweight version is optimized for Render's starter plan.")
