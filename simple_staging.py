import streamlit as st
import os

# Minimal page config
st.set_page_config(
    page_title="PBJ Staging Test",
    layout="wide"
)

# Simple staging test
st.title("🚧 PBJ Staging Test")
st.write("If you can see this, the staging deployment is working!")

# Environment info
st.header("Environment Check")
st.write(f"**Python Version**: {os.sys.version}")
st.write(f"**Working Directory**: {os.getcwd()}")
st.write(f"**Files in directory**: {len(os.listdir('.'))} files")

# Simple test
st.header("Test Results")
st.success("✅ Staging deployment successful!")
st.info("This is a minimal test to verify the deployment works.")
