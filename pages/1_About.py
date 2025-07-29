import streamlit as st

# Set page configuration
st.set_page_config(page_title="About | PBJ Nursing Home Staffing Dashboard by 320", page_icon="��", layout="wide")

# Main intro block with soft background and padding (combine into one call)
st.markdown("""
<div style="background: #f5f8fd; border-radius: 10px; padding: 2.2rem 2.5rem 1.5rem 2.5rem; margin-bottom: 2.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
    <div style='text-align: center; margin-bottom: 1.2em;'>
        <span style='font-size:2.3em; font-weight:800; color:#1769aa; letter-spacing:0.01em; line-height:1.1;'>PBJ Nursing Home Staffing Dashboard</span><br>
        <span style='font-size:1.15em; color:#7a869a; font-weight:400;'>by 320 Consulting</span>
    </div>

### Why this dashboard matters
Staffing data is a key indicator of nursing home quality, revealing how much care residents receive and what resources facilities commit. Yet most public data shows only the latest quarter, offering a narrow and incomplete view. This dashboard stitches together **eight years of CMS staffing files—billions of data points from payroll-based journal (PBJ) submissions—into interactive visualizations**, so you can see how staffing has changed over time and bring data-driven context to what’s happening inside the 15,000 nursing homes across the U.S.

### Who it helps  
* **Attorneys** – identify staffing patterns and trends that may support negligence cases, regulatory violations, or quality of care claims. Access historical data to demonstrate chronic understaffing, seasonal variations, or ownership-related staffing deficiencies.
* **Journalists** – plug numbers and data visualizations into a nursing home investigation or ownership-focused report without wrangling raw CSVs.
* **Advocates & families** – see how a nursing home stacks up over time for residents and loved ones.
* **Providers** – use historical staffing data to identify gaps, benchmark performance, and support quality improvement efforts.

*<a href="/Premium" style="color:#1769aa; text-decoration:underline;">Premium</a> – custom reports with daily, position-level analysis and data visualizations tied to citations and inspections.*

### What you can explore
| View | Data you get |
|------|--------------|
| **National / State** | Nurse staffing hours per resident day (HPRD), contract staff %, census — every quarter since 2017 |
| **Facility** | A nursing home's quarterly staffing, contract, and census data; ratings and risk indicators |
| **Ownership Group** | Essential data on any chain and its facilities (e.g., **Genesis** → 218 facilities in 19 states, 2.2-star average) |

### Under the hood  
* **Payroll-Based Journal (PBJ) Staffing Data** – 32 quarters of daily data, aggregated for clarity  
* **CMS Provider Info** – 5-star ratings, enforcement data, and other key indicators (June 2025 & March 2025)  
* **CMS Affiliated Entity** – Selected quality and performance metrics for groups of nursing homes sharing common owners, officers, or entities (June 2025)
* **CMS Citations (Premium)** - Citation data and inspection reports, categorized by date, type, severity, and more. 

### Quick tour  
1. Start search by selecting **Facility**, **Ownership**, or **State**.
2. Access state, facility, and ownership-level data and view data visualizations to spot trends over time.  
3. Click **Export** for ready-to-use PNGs.

### Digging deeper?  
Daily staffing data and analysis, role-specific hours (Nurse and Non-Nurse), weekend vs. weekday splits, and citation-linked timelines live in the premium layer.  
Email **eric@320insight.com** for requests. Journalists: If you're working on a story, I'm happy to share data or walk you through it.

*Built by 320 Consulting. Feedback welcome (tell me what's broken!).*
</div>
""", unsafe_allow_html=True)

# Section divider
st.markdown("<hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>", unsafe_allow_html=True)

# --- Styled container for Data Source through Census explanation ---
st.markdown("""
<div style="background: #f5f8fd; border-radius: 10px; padding: 2.2rem 2.5rem 1.5rem 2.5rem; margin-bottom: 2.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
    <div>
        <h2 style='font-size:1.5em; font-weight:700; color:#1769aa; margin-bottom:0.7em;'>Methodology</h2>
        <div style='font-size:1.08em; color:#222; font-weight:400;'>
            This Nursing Home Staffing Dashboard uses <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank">CMS Payroll-Based Journal (PBJ) data</a> from 2017 to 2024, covering all nursing positions, including contract staff. CMS first published PBJ data in 2017. It also uses <a href="https://data.cms.gov/provider-data/dataset/4pq5-n9py" target="_blank">Provider Information</a> (June 2025, March 2025) and <a href="https://data.cms.gov/quality-of-care/nursing-home-affiliated-entity-performance-measures/data" target="_blank">Affiliated Entity</a> (June 2025) datasets.
        </div>
        <hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>
        <h2 style='font-size:1.3em; font-weight:700; color:#1769aa; margin-bottom:0.5em;'>Staffing Categories</h2>
        <div style='font-size:1.08em; color:#222; font-weight:400;'>
            Total nurse staff includes:
            <ul style='margin-top:0.5em; margin-bottom:0.5em;'>
                <li>Registered Nurse (RN)</li>
                <li>RN Director of Nursing (DON)</li>
                <li>RN Admin</li>
                <li>Licensed Practical Nurse (LPN)</li>
                <li>LPN Admin</li>
                <li>Certified Nursing Assistant (CNA)</li>
                <li>Nurse Aide in Training</li>
                <li>Medication Aide/Technician</li>
            </ul>
        </div>
        <hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>
        <div style='font-size:1.13em; font-weight:700; color:#1769aa; margin-bottom:0.5em; margin-top:1.2em;'>Metrics Explained</div>
        <div style='font-size:1.08em; color:#222; font-weight:400; line-height:1.45;'>
            <b>Total Nurse Hours Per Resident Day (HPRD):</b> Total nurse staff hours per resident per day.*<br>
            <b>Contract Staff Percentage:</b> Percentage of nurse staff hours provided by contract staff.<br>
            <b>Census:</b> Average number of residents in facility or state during the reporting period.
        </div>
        <div style='font-size:0.95em; color:#666; font-weight:400; line-height:1.4; margin-top:1em; padding:1em; background:#f8f9fa; border-left:3px solid #1769aa; border-radius:3px;'>
            <b>* HPRD Explained:</b> This metric reflects the staffing ratio at a facility in terms of staff hours per resident. Example: A nursing home with 100 residents providing 350 staffing hours per day would have a 3.5 HPRD (350 ÷ 100).<br><br>
            A 2001 federal study identified 4.1 HPRD as the level linked to better outcomes for most residents. Facilities with higher-acuity residents—such as those with complex medical needs or limited mobility—generally require more staffing. Staffing levels can also vary significantly by day and shift.<br><br>
            Some states have their own standards (e.g., New Jersey, California, and New York each set a 3.5 HPRD minimum), though enforcement and definitions vary. Note: A federal 3.48 HPRD minimum was recently overturned by a court in 2025.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# About 320 Consulting Section
st.markdown("""
<div style="background: #f5f8fd; border-radius: 10px; padding: 2.2rem 2.5rem 1.5rem 2.5rem; margin-bottom: 2.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
    <div>
        <h2 style='font-size:1.5em; font-weight:700; color:#1769aa; margin-bottom:0.7em;'>About 320 Consulting</h2>
        <div style='font-size:1.08em; color:#222; font-weight:400;'>
            <b><a href="https://www.320insight.com/" target="_blank" style="color:#1769aa; text-decoration:none;">320 Consulting</a></b> is led by Eric Goldwein, MPH, a data consultant with expertise in nursing home staffing. His work on nursing home data has been published in the <i>Journal of the American Geriatrics Society</i>, and he has presented at national conferences hosted by the National Association of Medicaid Fraud Control Units, Consumer Voice, the American Society on Aging, and the NYS Long Term Care Ombudsman Program. He previously led data and policy work at the Long Term Care Community Coalition.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Section divider
st.markdown("<hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>", unsafe_allow_html=True)

# Add back to dashboard button
if st.button("← Back to Dashboard", key="back_to_dashboard_about"):
    st.switch_page("PBJ_Dashboard.py")

# Footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #666; font-size: 0.9em;">
        <p>Source: <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" style="color: #1E88E5; text-decoration: none;">CMS Payroll-Based Journal Data, 2017-2024</a></p>
        <p>By <a href="https://www.320insight.com/" target="_blank" style="color: #1E88E5; text-decoration: none; font-weight: 500;">320 Consulting LLC</a></p>
        <p><a href="/About" target="_self">About the Dashboard</a> | <a href="/Premium" target="_self">Premium</a></p>
    </div>
""", unsafe_allow_html=True) 