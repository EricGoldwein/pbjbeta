import streamlit as st
import os

# Check if we're in staging environment
IS_STAGING = os.getenv('STAGING', 'false').lower() == 'true'

st.set_page_config(
    page_title="PBJ Dashboard - STAGING" if IS_STAGING else "PBJ Dashboard",
    layout="wide"
)

if IS_STAGING:
    st.warning("🚧 **STAGING ENVIRONMENT** - This is a test site")

st.title("PBJ Dashboard")
st.write("This is a minimal test version for staging.")

if IS_STAGING:
    st.success("✅ Staging environment detection is working!")
else:
    st.info("ℹ️ This is the production environment")
