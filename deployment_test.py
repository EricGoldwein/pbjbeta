import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Deployment Test", layout="wide")
st.title("PBJ Dashboard - Deployment Test")

# Check if we're in staging
IS_STAGING = os.getenv('STAGING', 'false').lower() == 'true'
if IS_STAGING:
    st.warning("🚧 **STAGING ENVIRONMENT** - This is a test site")

st.header("File Check")
required_files = [
    'national_lite_metrics.csv',
    'state_lite_metrics.csv', 
    'facility_lite_metrics.csv'
]

for file in required_files:
    if os.path.exists(file):
        st.success(f"✅ {file} - Found")
        try:
            df = pd.read_csv(file)
            st.write(f"   - Rows: {len(df)}, Columns: {len(df.columns)}")
        except Exception as e:
            st.error(f"   - Error reading: {e}")
    else:
        st.error(f"❌ {file} - Missing")

st.header("Environment Check")
st.write(f"Current directory: {os.getcwd()}")
st.write(f"Files in directory: {os.listdir('.')[:10]}...")  # Show first 10 files

if IS_STAGING:
    st.success("✅ Staging environment detected!")
else:
    st.info("ℹ️ Production environment")
