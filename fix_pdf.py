from fpdf import FPDF
import pandas as pd
import duckdb
import os

def get_db_connection():
    if os.path.exists('nursing_home_staffing.db'):
        return duckdb.connect('nursing_home_staffing.db')
    else:
        print("Database file not found!")
        return None

def get_facility_info(provnum: str, conn) -> dict:
    """Get facility information from the database."""
    try:
        query = """
            SELECT DISTINCT
                PROVNUM,
                PROVNAME,
                STATE,
                COUNTY_NAME,
                CITY
            FROM staffing 
            WHERE PROVNUM = ?
            ORDER BY WORKDATE DESC
            LIMIT 1
        """
        
        result = conn.execute(query, (provnum,)).fetchone()
        if result:
            return {
                'ccn': result[0],
                'provider_name': result[1],
                'state': result[2],
                'county': result[3],
                'city': result[4]
            }
        return None
    except Exception as e:
        print(f"Error getting facility info: {str(e)}")
        return None

def get_quarterly_metrics(provnum: str, quarter: str, conn) -> dict:
    """Get quarterly metrics for a facility."""
    try:
        query = """
            WITH daily_metrics AS (
                SELECT 
                    PROVNUM,
                    WORKDATE,
                    MDSCENSUS,
                    (HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN + HRS_CNA + HRS_NATRN + HRS_MEDAIDE) as total_hours,
                    (HRS_RNDON + HRS_RNADMIN + HRS_RN) as rn_hours,
                    (HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN) as nurse_care_hours
                FROM staffing 
                WHERE CY_QTR = ? AND PROVNUM = ?
            )
            SELECT
                SUM(MDSCENSUS) as total_resident_days,
                SUM(total_hours) as total_hours,
                SUM(rn_hours) as rn_hours,
                SUM(nurse_care_hours) as nurse_care_hours
            FROM daily_metrics
        """
        
        result = conn.execute(query, (quarter, provnum)).fetchone()
        if result and result[0] is not None:  # Check if we have resident days
            return {
                'total_hours': result[1] / result[0] if result[0] > 0 else 0,
                'rn_hours': result[2] / result[0] if result[0] > 0 else 0,
                'nurse_care_hours': result[3] / result[0] if result[0] > 0 else 0
            }
        return None
    except Exception as e:
        print(f"Error getting quarterly metrics: {str(e)}")
        return None

def create_pdf_report(provnum: str, selected_quarter: str, conn) -> bytes:
    """Generate a PDF report for a facility."""
    try:
        # Get facility info
        facility_info = get_facility_info(provnum, conn)
        if not facility_info:
            return None

        # Get metrics for the selected quarter
        metrics = get_quarterly_metrics(provnum, selected_quarter, conn)
        if not metrics:
            return None

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, f"Facility Report: {facility_info['provider_name']}", ln=True, align="C")
        pdf.ln(10)
        
        # Add facility details
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Facility Details", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Provider Number: {provnum}", ln=True)
        pdf.cell(0, 10, f"Location: {facility_info['city']}, {facility_info['state']}", ln=True)
        pdf.cell(0, 10, f"County: {facility_info['county']}", ln=True)
        pdf.ln(10)
        
        # Add metrics
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Metrics for {selected_quarter}", ln=True)
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 10, f"Total Hours per Resident Day: {metrics['total_hours']:.2f}", ln=True)
        pdf.cell(0, 10, f"RN Hours per Resident Day: {metrics['rn_hours']:.2f}", ln=True)
        pdf.cell(0, 10, f"Nurse Care Hours per Resident Day: {metrics['nurse_care_hours']:.2f}", ln=True)
        pdf.ln(10)
        
        # Add citations
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Recent Citations", ln=True)
        pdf.set_font("Arial", "", 12)
        
        # Get citations
        citations = conn.execute("""
            SELECT tag_number, survey_date, deficiency, scope_severity, standard_deficiency, 
                   complaint_deficiency, infection_control_inspection_deficiency
            FROM citations
            WHERE provnum = ?
            ORDER BY survey_date DESC
            LIMIT 5
        """, [provnum]).fetchdf()
        
        if not citations.empty:
            # Create citations table
            pdf.set_font("Arial", "B", 10)
            pdf.cell(40, 10, "Tag Number", 1)
            pdf.cell(30, 10, "Date", 1)
            pdf.cell(80, 10, "Deficiency", 1)
            pdf.cell(40, 10, "Scope/Severity", 1, ln=True)
            
            pdf.set_font("Arial", "", 10)
            for _, row in citations.iterrows():
                pdf.cell(40, 10, str(row['tag_number']), 1)
                pdf.cell(30, 10, str(row['survey_date']), 1)
                pdf.cell(80, 10, str(row['deficiency']), 1)
                pdf.cell(40, 10, str(row['scope_severity']), 1, ln=True)
        else:
            pdf.cell(0, 10, "No citations found", ln=True)
        
        return pdf.output(dest='S').encode('latin1')
    except Exception as e:
        print(f"Error creating PDF report: {str(e)}")
        return None 