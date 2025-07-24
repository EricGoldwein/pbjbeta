import streamlit as st

# Set page configuration
st.set_page_config(page_title="About the Dashboard", page_icon="��", layout="wide")

# Main intro block with soft background and padding (combine into one call)
st.markdown("""
<div style="background: #f5f8fd; border-radius: 10px; padding: 2.2rem 2.5rem 1.5rem 2.5rem; margin-bottom: 2.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
    <div style='text-align: center; margin-bottom: 1.2em;'>
        <span style='font-size:2.3em; font-weight:800; color:#1769aa; letter-spacing:0.01em;'>PBJ Nursing Home Staffing Dashboard</span><br>
        <span style='font-size:1.15em; color:#7a869a; font-weight:400;'>by 320 Consulting</span>
    </div>

### Context Matters
Most publicly available nursing home data shows only the latest quarter. This dashboard stitches **eight years of staffing files—billions of data points from payroll-based journals (PBJ)—into a single view**, so you can see how staffing has changed over time—and bring data-driven context to what’s happening inside facilities and chains.

### Who it helps  
* **Journalists** – plug numbers and data visualizations into an investigation or ownership-focused report without wrangling raw CSVs.  
* **Advocates & families** – see how a home stacks up over time for loved ones.
* **Policymakers** – compare homes and trends to inform oversight or reform.
* **Attorneys** – spot staffing trends that may support a case.
    <span style='font-size:0.95em; color:#555;'>&nbsp;&nbsp;Premium option: daily, position-level hours tied to citations and inspection reports.</span>

### What you can explore (public edition)  
| View | Data you get |
|------|--------------|
| **National / State** | Nurse staffing hours per resident day (HPRD), contract staff %, census — every quarter since 2017 |
| **Facility** | A nursing home's quarterly staffing, contract, and census data; ratings and risk indicators |
| **Ownership Group** | Roll-ups for any chain (e.g., **Genesis** → 218 facilities in 19 states, 2.2-star average) |

### Under the hood  
* **PBJ Staffing** – 32 quarters of daily data, aggregated for clarity  
* **CMS Provider Info** – 5-star ratings, enforcement data, and other key indicators (June 2025 & March 2025)  
* **CMS Affiliated Entity** – Selected quality metrics for nursing homes with shared owners, officers, or operators.(June 2025)
* **CMS Citations (Premium)** - Citation data and inspection reports, categorized by date, type, severity, and more. 

### Quick tour  
1. Pick **National**, **State**, **Facility**, or **Affiliated Entity**.  
2. Hover charts for values; drag the date slider to focus on any span.  
3. Click **Export** for a ready-to-use PNG or CSV.

### Digging deeper?  
Daily shift logs, role-specific hours, weekend vs. weekday splits, and citation-linked timelines live in the premium layer.  
Email **eric@320insight.com** for a free demo. Journalists: If you're working on a story, I'm happy to share data or walk you through it.

*Built by 320 Consulting. Feedback welcome (tell me what's broken!).*
</div>
""", unsafe_allow_html=True)

# Section divider
st.markdown("<hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>", unsafe_allow_html=True)

# Data Source Section
st.header("Data Source")
st.markdown("""
This Nursing Home Staffing Dashboard uses [CMS Payroll-Based Journal (PBJ) data](https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing) from 2017 to 2024, covering all nursing positions, including contract staff. CMS first published PBJ data in 2017. It also uses [Provider Information](https://data.cms.gov/provider-data/dataset/4pq5-n9py) (June 2025, March 2025) and [Affiliated Entity](https://data.cms.gov/quality-of-care/nursing-home-affiliated-entity-performance-measures/data) (June 2025) datasets.
""")

# Section divider
st.markdown("<hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>", unsafe_allow_html=True)

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

# Section divider
st.markdown("<hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>", unsafe_allow_html=True)

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