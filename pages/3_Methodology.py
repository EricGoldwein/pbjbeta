import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Methodology | PBJ Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add back link
st.markdown("""
    <a href="/" style="color: #1E88E5; text-decoration: none; font-weight: 500; display: inline-block; margin-bottom: 20px;">← Back to Dashboard</a>
""", unsafe_allow_html=True)

# Title
st.title("Methodology")

# Data Source Section
st.header("Data Source")
st.markdown("This dashboard uses [CMS Payroll-Based Journal (PBJ) data](https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing) from 2017 to 2024, covering all nursing positions, including contract staff. CMS first published PBJ data in 2017.")

# Staffing Categories Section
st.header("Staffing Categories")
st.write("Total nurse staff includes:")
st.markdown("""
- Registered Nurse (RN)
- Director of Nursing (DON)
- RN with administrative duties
- Licensed Practical Nurse (LPN) with administrative duties
- LPN
- Certified Nursing Assistant (CNA)
- Medication Aide/Technician
- Nurse Aide in Training
""")

# Metrics Section
st.header("Metrics Explained")
st.markdown("""
**Total Nurse HPRD:** Hours Per Resident Day - The total number of nursing hours provided per resident per day.

**Contract Staff Percentage:** The percentage of nursing hours provided by contract staff.

**Census:** The average number of residents in the facility during the reporting period.
""")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
        <p>Source: CMS Payroll-Based Journal Data, 2017-2024</p>
        <p>By <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">320 Consulting LLC</a></p>
    </div>
""", unsafe_allow_html=True) 