import streamlit as st
import os
import pandas as pd
import sys
from pathlib import Path
import traceback

# Set page config
st.set_page_config(
    page_title="PBJ Nursing Home Dashboard - Staging",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add staging indicator
st.warning("🚧 **STAGING ENVIRONMENT** - This is a test deployment")

def check_and_create_minimal_data():
    """Check for required CSV files and create minimal data if missing"""
    required_files = {
        'national_lite_metrics.csv': ['Quarter', 'Total_Nurse_HPRD', 'RN_HPRD', 'LPN_HPRD', 'CNA_HPRD'],
        'state_lite_metrics.csv': ['State', 'Quarter', 'Total_Nurse_HPRD', 'RN_HPRD', 'LPN_HPRD', 'CNA_HPRD'],
        'facility_lite_metrics.csv': ['PROVNUM', 'Facility_Name', 'State', 'Quarter', 'Total_Nurse_HPRD', 'RN_HPRD', 'LPN_HPRD', 'CNA_HPRD']
    }
    
    for filename, columns in required_files.items():
        if not os.path.exists(filename):
            st.info(f"Creating minimal {filename} for staging...")
            # Create minimal data
            if 'national' in filename:
                data = pd.DataFrame({
                    'Quarter': ['Q4 2024'],
                    'Total_Nurse_HPRD': [3.5],
                    'RN_HPRD': [1.2],
                    'LPN_HPRD': [0.8],
                    'CNA_HPRD': [1.5]
                })
            elif 'state' in filename:
                data = pd.DataFrame({
                    'State': ['CA', 'TX', 'FL'],
                    'Quarter': ['Q4 2024', 'Q4 2024', 'Q4 2024'],
                    'Total_Nurse_HPRD': [3.2, 3.8, 3.1],
                    'RN_HPRD': [1.1, 1.3, 1.0],
                    'LPN_HPRD': [0.7, 0.9, 0.6],
                    'CNA_HPRD': [1.4, 1.6, 1.5]
                })
            else:  # facility
                data = pd.DataFrame({
                    'PROVNUM': ['123456', '789012', '345678'],
                    'Facility_Name': ['Test Facility 1', 'Test Facility 2', 'Test Facility 3'],
                    'State': ['CA', 'TX', 'FL'],
                    'Quarter': ['Q4 2024', 'Q4 2024', 'Q4 2024'],
                    'Total_Nurse_HPRD': [3.2, 3.8, 3.1],
                    'RN_HPRD': [1.1, 1.3, 1.0],
                    'LPN_HPRD': [0.7, 0.9, 0.6],
                    'CNA_HPRD': [1.4, 1.6, 1.5]
                })
            
            data.to_csv(filename, index=False)
            st.success(f"✅ Created {filename}")

# Main application with error handling
try:
    # Initialize data
    check_and_create_minimal_data()
    
    # Main dashboard content
    st.title("📊 PBJ Nursing Home Dashboard")
    st.markdown("**Staging Environment - Test Data**")

    # File status check
    st.header("📁 File Status")
    required_files = [
        'national_lite_metrics.csv',
        'state_lite_metrics.csv', 
        'facility_lite_metrics.csv'
    ]

    for file in required_files:
        if os.path.exists(file):
            try:
                df = pd.read_csv(file)
                st.success(f"✅ {file} - {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                st.error(f"❌ {file} - Error: {e}")
        else:
            st.error(f"❌ {file} - Missing")

    # Environment info
    st.header("🔧 Environment Information")
    st.write(f"**Current Directory:** {os.getcwd()}")
    st.write(f"**Python Version:** {sys.version}")
    st.write(f"**Streamlit Version:** {st.__version__}")

    # Sample data display
    st.header("📈 Sample Data")
    try:
        if os.path.exists('national_lite_metrics.csv'):
            df = pd.read_csv('national_lite_metrics.csv')
            st.subheader("National Metrics")
            st.dataframe(df.head())
        
        if os.path.exists('state_lite_metrics.csv'):
            df = pd.read_csv('state_lite_metrics.csv')
            st.subheader("State Metrics")
            st.dataframe(df.head())
            
        if os.path.exists('facility_lite_metrics.csv'):
            df = pd.read_csv('facility_lite_metrics.csv')
            st.subheader("Facility Metrics")
            st.dataframe(df.head())
            
    except Exception as e:
        st.error(f"Error displaying data: {e}")

    # Deployment status
    st.header("🚀 Deployment Status")
    st.success("✅ Staging deployment is running successfully!")
    st.info("This is a test environment with sample data for deployment verification.")
    
    # Additional deployment info
    st.subheader("🔗 Deployment Links")
    st.markdown("""
    - **Staging Site**: https://pbj-dashboard-staging.onrender.com
    - **Production Site**: https://pbjdashboard.com
    - **GitHub Repository**: Check your repo for latest changes
    """)
    
    # Performance metrics
    st.subheader("📊 Performance Metrics")
    import psutil
    st.write(f"**Memory Usage**: {psutil.virtual_memory().percent:.1f}%")
    st.write(f"**CPU Usage**: {psutil.cpu_percent():.1f}%")
    st.write(f"**Disk Usage**: {psutil.disk_usage('/').percent:.1f}%")

except Exception as e:
    st.error("❌ **Error in staging dashboard**")
    st.error(f"Error details: {str(e)}")
    st.code(traceback.format_exc())
    st.info("This error has been logged for debugging.")