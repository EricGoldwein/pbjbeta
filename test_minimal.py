import streamlit as st
import os

# Set page config
st.set_page_config(
    page_title="PBJ Test",
    layout="wide"
)

st.title("PBJ Test")
st.write("If you can see this, the basic app works!")

# Test staging environment
IS_STAGING = os.getenv('STAGING', 'false').lower() == 'true'
if IS_STAGING:
    st.warning("🚧 **STAGING ENVIRONMENT** - This is a test site")
else:
    st.info("ℹ️ Production environment")

st.success("✅ Basic test successful!")
