import streamlit as st
import base64

# Import dynamic date utilities
from utils.date_utils import get_latest_data_periods, apply_dynamic_replacements

# Set page configuration
st.set_page_config(page_title="About | PBJ Nursing Home Staffing Dashboard by 320", page_icon="pbj_images/pbj_favicon.png", layout="wide")

# Load favicon data
def load_pbj_favicon():
    """Load PBJ favicon data for use in footer"""
    try:
        import os
        # Try multiple possible paths for favicon
        possible_paths = [
            os.path.join(os.getcwd(), 'pbj_images', 'pbj_favicon.png'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pbj_images', 'pbj_favicon.png'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pbj_images', 'pbj_favicon.png'),
            os.path.join('pbj_images', 'pbj_favicon.png'),  # Try relative path
            'pbj_favicon.png',  # Try root directory
            os.path.join(os.getcwd(), 'pbj_favicon.png'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pbj_favicon.png'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pbj_favicon.png'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return base64.b64encode(f.read()).decode()
        
        # Fallback to hardcoded base64 if file not found
        return "iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAYAAAB/HSuDAADZUWNhQlgAANlRanVtYgAAAB5qdW1kYzJwYQARABCAAACqADibcQNjMnBhAAAANxNqdW1iAAAAR2p1bWRjMm1hABEAEIAAAKoAOJtxA3VybjpjMnBhOjAzNTc4YTlkLTlkOTctNGEyZi1iNzg2LTU4ZGRhMTU0MzUzNgAAAAHhanVtYgAAAClqdW1kYzJhcwA"
    except Exception as e:
        print(f"Warning: Error loading favicon: {e}")
        # Fallback to hardcoded base64 on any error
        return "iVBORw0KGgoAAAANSUhEUgAABAAAAAQACAYAAAB/HSuDAADZUWNhQlgAANlRanVtYgAAAB5qdW1kYzJwYQARABCAAACqADibcQNjMnBhAAAANxNqdW1iAAAAR2p1bWRjMm1hABEAEIAAAKoAOJtxA3VybjpjMnBhOjAzNTc4YTlkLTlkOTctNGEyZi1iNzg2LTU4ZGRhMTU0MzUzNgAAAAHhanVtYgAAAClqdW1kYzJhcwA"

favicon_data = load_pbj_favicon()

# Custom CSS for Home button styling
st.markdown("""
<style>
.home-button {
    background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
    color: #1565c0;
    border: 1px solid #90caf9;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    transition: all 0.2s ease;
    margin-top: 8px;
    margin-bottom: 8px;
}

.home-button:hover {
    background: linear-gradient(135deg, #bbdefb 0%, #90caf9 100%);
    transform: translateY(-1px);
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
}

.home-button:active {
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# Add Home button to top left - single row layout
col1, col2, col3 = st.columns([1, 8, 1])
with col1:
    st.markdown(f"""
        <div style="margin-top: 8px; margin-bottom: 8px; white-space: nowrap;">
            <a href="/" style="display: inline-flex; align-items: center; background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); color: #1565c0; border: 1px solid #90caf9; border-radius: 6px; padding: 6px 12px; font-size: 13px; font-weight: 500; text-decoration: none; box-shadow: 0 1px 3px rgba(0,0,0,0.08); transition: all 0.2s ease;">
                <img src="data:image/png;base64,{favicon_data}" style="width: 16px; height: 16px; margin-right: 4px; flex-shrink: 0;"> <span>Home</span>
            </a>
        </div>
    """, unsafe_allow_html=True)

# Main intro block with soft background and padding (combine into one call)
st.markdown(f"""
<div style="background: #f5f8fd; border-radius: 10px; padding: 2.2rem 2.5rem 1.5rem 2.5rem; margin-bottom: 2.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
    <div style='text-align: center; margin-bottom: 1.2em;'>
        <span style='font-size:2.3em; font-weight:800; color:#1769aa; letter-spacing:0.01em; line-height:1.1;'>PBJ Nursing Home Staffing Dashboard</span><br>
        <span style='font-size:1.15em; color:#7a869a; font-weight:400;'>by 320 Consulting</span>
    </div>

### Why this matters
Staffing data is a key indicator of nursing home quality, revealing how much care residents receive and what resources facilities commit. Yet most public data shows only the latest quarter, offering a narrow and incomplete view. This dashboard stitches together **nine years of federal CMS staffing files—billions of data points from payroll-based journal (PBJ) submissions—into interactive visualizations**, so you can see how staffing has changed over time and bring data-driven context to what's happening inside the 15,000 nursing homes across the U.S.

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
| **Ownership Group** | Essential data on any chain and its facilities (e.g., **Genesis** → 215 facilities in 19 states, 2.3-star average) |

### Under the hood  
* **Payroll-Based Journal (PBJ) Staffing Data** – {get_latest_data_periods()['quarter_count']} quarters of daily data, aggregated for clarity  
* **CMS Provider Info** – 5-star ratings, enforcement data, and other key indicators  
* **CMS Affiliated Entity** – Selected quality and performance metrics for groups of nursing homes sharing common owners, officers, or entities ({get_latest_data_periods()['affiliated_entity_latest']})  
* **CMS Citations (Premium)** - Citation data and inspection reports, categorized by date, type, severity, and more. 

### Quick tour  
1. Start search by selecting **Facility**, **Ownership**, or **State**.
2. Access state, facility, and ownership-level data and view data visualizations to spot trends over time.  
3. Click **Export** for ready-to-use PNGs.

### Try Phoebe J
Check out [Phoebe J, the PBJ nursing home staffing data assistant (in training!)](https://www.320insight.com/phoebe) for quick PBJ data searches by state or nursing home.

### Digging deeper?  
Daily staffing data and analysis, role-specific hours (Nurse and Non-Nurse), weekend vs. weekday splits, and citation-linked timelines live in the premium layer.  
Email **eric@320insight.com** for requests. Journalists: If you're working on a story, I'm happy to share data or walk you through it.

*Built by 320 Consulting. Feedback welcome (tell me what's broken!).*
</div>
""", unsafe_allow_html=True)

# Section divider
st.markdown("<hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>", unsafe_allow_html=True)

# --- Styled container for Data Source through Census explanation ---
st.markdown(f"""
<div style="background: #f5f8fd; border-radius: 10px; padding: 2.2rem 2.5rem 1.5rem 2.5rem; margin-bottom: 2.2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.03);">
    <div>
        <h2 style='font-size:1.5em; font-weight:700; color:#1769aa; margin-bottom:0.7em;'>Methodology</h2>
        <div style='font-size:1.08em; color:#222; font-weight:400;'>
            This Nursing Home Staffing Dashboard uses <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank">CMS Payroll-Based Journal (PBJ) data</a> from 2017 to {get_latest_data_periods()['current_year']}, covering all nursing positions, including contract staff. CMS first published PBJ data in 2017. It also uses <a href="https://data.cms.gov/provider-data/dataset/4pq5-n9py" target="_blank">Provider Information</a> ({get_latest_data_periods()['provider_info_latest']}, {get_latest_data_periods()['provider_info_previous']}), <a href="https://data.cms.gov/quality-of-care/nursing-home-affiliated-entity-performance-measures/data" target="_blank">Affiliated Entity</a> ({get_latest_data_periods()['affiliated_entity_latest']}), and <a href="https://www.macpac.gov/publication/state-policies-related-to-nursing-facility-staffing/" target="_blank">MACPAC State Staffing Standards</a> (2022) datasets.
        </div>
        <hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>
        <h2 style='font-size:1.3em; font-weight:700; color:#1769aa; margin-bottom:0.5em;'>Staff Categories</h2>
        <div style='font-size:1.08em; color:#222; font-weight:400;'>
            <ul style='margin-top:0.5em; margin-bottom:0.5em;'>
                <li><strong>Total Staff:</strong> All nursing staff including RNs, LPNs, CNAs, administrators, and DON</li>
                <li><strong>Direct Staff:</strong> Staff providing direct patient care (excludes administrators and DON)</li>
                <li><strong>RN (Registered Nurse):</strong> Includes direct care RNs + RN administrators + RN DON (Director of Nursing)</li>
                <li><strong>RN Direct (RN Only):</strong> Excludes RN Admin, RN DON</li>
                <li><strong>LPN (Licensed Practical Nurse):</strong> Includes direct care LPNs + LPN administrators</li>
                <li><strong>CNA (Certified Nursing Assistant):</strong> Includes CNAs + NA trainees + Medication Aides</li>
                <li><strong>Contract Staff:</strong> Temporary/agency staff (marked with "_ctr" suffix)</li>
                <li><strong>Case-mix:</strong> A CMS benchmark for staffing based on resident acuity.</li>
            </ul>
        </div>
        <hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>
        <div style='font-size:1.13em; font-weight:700; color:#1769aa; margin-bottom:0.5em; margin-top:1.2em;'>Metrics Explained</div>
        <div style='font-size:1.08em; color:#222; font-weight:400; line-height:1.45;'>
            <b>Total Nurse Hours Per Resident Day (HPRD):</b> Total nurse staff hours per resident per day.*<br>
            <b>Direct Care (excl. Admin, DON):</b> Hours per resident day for direct care staff only (RN, LPN, CNA, NAtrn, MedAide), excluding administrative and supervisory roles.<br>
            <b>Contract Staff Percentage:</b> Percentage of nurse staff hours provided by contract staff.<br>
            <b>Census:</b> Average number of residents in facility or state during the reporting period.<br>
            <b>Ownership Change:</b> Indicates facility ownership changed in the last 12 months.<br>
            <b>SER (Staffing-to-Expected Ratio) or % Case-Mix:</b> Reported ÷ CMS Case-Mix. Values below 1.0 indicate staffing deficit relative to CMS benchmark expectations; values above 1.0 indicate surplus staffing. Case-mix is a CMS benchmark for staffing based on resident acuity.
        </div>
        <div style='font-size:0.95em; color:#666; font-weight:400; line-height:1.4; margin-top:1em; padding:1em; background:#f8f9fa; border-left:3px solid #1769aa; border-radius:3px;'>
            <b>* HPRD Explained:</b> This metric reflects the staffing ratio at a facility in terms of staff hours per resident. Example: A nursing home with 100 residents providing 350 staffing hours per day would have a 3.5 HPRD (350 ÷ 100).<br><br>
            A 2001 federal study identified 4.1 HPRD as the level linked to better outcomes for most residents. Facilities with higher-acuity residents—such as those with complex medical needs or limited mobility—generally require more staffing. Staffing levels can also vary significantly by day and shift.<br><br>
            Some states have their own standards (e.g., New Jersey, California, and New York each set a 3.5 HPRD minimum), though enforcement and definitions vary. Note: A federal 3.48 HPRD minimum was recently overturned by a court in 2025.
        </div>
        <hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>
        <div style='font-size:1.13em; font-weight:700; color:#1769aa; margin-bottom:0.5em; margin-top:1.2em;'>Transparency Note</div>
        <div style='font-size:1.08em; color:#222; font-weight:400; line-height:1.45;'>
            The PBJ Dashboard pulls directly from CMS data and is carefully vetted for accuracy. Still, sometimes a fly sneaks into the jelly. 🪰 🥪
        </div>
        <div style='font-size:1.08em; color:#222; font-weight:400; line-height:1.45; margin-top:1em;'>
            That could mean:
            <ul style='margin-top:0.5em; margin-bottom:0.5em;'>
                <li>A facility reported bad data to CMS (more common than you'd think).</li>
                <li>Or I made a coding error (it happens).</li>
            </ul>
            Either way, I want to be the first to know. If you spot something that looks off, please let me know so I can squash the bug and set things right.
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
            <b><a href="https://www.320insight.com/" target="_blank" style="color:#1769aa; text-decoration:none;">320 Consulting</a></b> is led by Eric Goldwein, MPH, a data consultant with expertise in nursing home staffing. His work on nursing home data has been published in the <i>Journal of the American Geriatrics Society</i>, and he has presented at national conferences hosted by the National Association of Medicaid Fraud Control Units, Consumer Voice, the American Society on Aging, and the NYS Long Term Care Ombudsman Program. He previously served as policy director at the Long Term Care Community Coalition.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Section divider
st.markdown("<hr style='margin: 2.2em 0 1.5em 0; border: none; border-top: 1.5px solid #e3e8f0;'>", unsafe_allow_html=True)

# Footer - new format
st.markdown(f"""
    <div class="footer" style="margin-top: 40px;">
        <p style="color: #94a3b8; margin: 0 auto; font-style: italic; line-height: 1.6; text-align: center; max-width: 800px;">The <strong>PBJ Dashboard</strong> is a free public resource providing consumers with longitudinal data at 15,000 US nursing homes. It has been featured in <a href="https://www.publichealth.columbia.edu/news/alumni-make-data-shine-public-health-dashboards" target="_blank" style="color: #7c8fa3; text-decoration: none; font-weight: 450;">Columbia Public Health</a>, <a href="https://www.retirementlivingsourcebook.com/videos/why-nursing-home-staffing-data-matters-for-1-2-million-residents-and-beyond" target="_blank" style="color: #7c8fa3; text-decoration: none; font-weight: 450;">Positive Aging</a>, and <a href="https://aginginamerica.news/2025/09/16/crunching-the-nursing-home-data/" target="_blank" style="color: #7c8fa3; text-decoration: none; font-weight: 450;">Aging in America News</a>.</p>
        <div class="footer-divider" style="margin-top: 20px;">
            <p style="margin: 0; color: #6b7280; font-style: italic; font-size: 0.9rem; text-align: center;">Source: <a href="https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing" target="_blank" style="color: #6b7280; text-decoration: none;">CMS Payroll-Based Journal Data, {get_latest_data_periods()['data_range']}</a> | <a href="https://www.320insight.com/" style="color: #6b7280; text-decoration: none;">320 Consulting — Turning Spreadsheets into Stories</a></p>
        </div>
    </div>
""", unsafe_allow_html=True) 