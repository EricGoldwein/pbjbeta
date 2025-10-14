from flask import Flask, render_template, request, jsonify
import duckdb
import pandas as pd
import os
from datetime import datetime, date
import plotly.express as px
import plotly.graph_objects as go
import json
import glob
import atexit
import tempfile

# Set the template folder to the current directory (scripts folder)
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
print(f"Template directory: {template_dir}")
print(f"Template directory exists: {os.path.exists(template_dir)}")
print(f"Index.html exists: {os.path.exists(os.path.join(template_dir, 'index.html'))}")
app = Flask(__name__, template_folder=template_dir)

# Create a temporary directory for DuckDB
temp_dir = tempfile.mkdtemp()
db_path = os.path.join(temp_dir, 'nurse_staffing.db')

# Initialize DuckDB connection
conn = duckdb.connect(db_path)
conn.execute("PRAGMA threads=8")
conn.execute("PRAGMA memory_limit='8GB'")

def cleanup():
    """Clean up database connection and temporary files"""
    try:
        if conn:
            conn.close()
        # Remove temporary directory and its contents
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                try:
                    os.remove(os.path.join(temp_dir, file))
                except:
                    pass
            try:
                os.rmdir(temp_dir)
            except:
                pass
    except Exception as e:
        print(f"Cleanup error: {str(e)}")

# Register cleanup function
atexit.register(cleanup)

def load_data():
    """Load data into DuckDB tables"""
    try:
        # Create tables first
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nurse_staffing (
                PROVNUM VARCHAR,
                PROVNAME VARCHAR,
                CITY VARCHAR,
                STATE VARCHAR,
                COUNTY_NAME VARCHAR,
                COUNTY_FIPS VARCHAR,
                CY_Qtr VARCHAR,
                WorkDate VARCHAR,
                MDScensus DOUBLE,
                Hrs_RNDON DOUBLE,
                Hrs_RNDON_emp DOUBLE,
                Hrs_RNDON_ctr DOUBLE,
                Hrs_RNadmin DOUBLE,
                Hrs_RNadmin_emp DOUBLE,
                Hrs_RNadmin_ctr DOUBLE,
                Hrs_RN DOUBLE,
                Hrs_RN_emp DOUBLE,
                Hrs_RN_ctr DOUBLE,
                Hrs_LPNadmin DOUBLE,
                Hrs_LPNadmin_emp DOUBLE,
                Hrs_LPNadmin_ctr DOUBLE,
                Hrs_LPN DOUBLE,
                Hrs_LPN_emp DOUBLE,
                Hrs_LPN_ctr DOUBLE,
                Hrs_CNA DOUBLE,
                Hrs_CNA_emp DOUBLE,
                Hrs_CNA_ctr DOUBLE,
                Hrs_NAtrn DOUBLE,
                Hrs_NAtrn_emp DOUBLE,
                Hrs_NAtrn_ctr DOUBLE,
                Hrs_MedAide DOUBLE,
                Hrs_MedAide_emp DOUBLE,
                Hrs_MedAide_ctr DOUBLE
            )
        """)
        
        # Create non-nurse staffing table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS non_nurse_staffing (
                PROVNUM VARCHAR,
                PROVNAME VARCHAR,
                CITY VARCHAR,
                STATE VARCHAR,
                COUNTY_NAME VARCHAR,
                COUNTY_FIPS VARCHAR,
                CY_Qtr VARCHAR,
                WorkDate VARCHAR,
                MDScensus DOUBLE,
                Hrs_Admin DOUBLE,
                Hrs_Admin_emp DOUBLE,
                Hrs_Admin_ctr DOUBLE,
                Hrs_Admin_fn DOUBLE,
                Hrs_MedDir DOUBLE,
                Hrs_MedDir_emp DOUBLE,
                Hrs_MedDir_ctr DOUBLE,
                Hrs_OthMD DOUBLE,
                Hrs_OthMD_emp DOUBLE,
                Hrs_OthMD_ctr DOUBLE,
                Hrs_PA DOUBLE,
                Hrs_PA_emp DOUBLE,
                Hrs_PA_ctr DOUBLE,
                Hrs_NP DOUBLE,
                Hrs_NP_emp DOUBLE,
                Hrs_NP_ctr DOUBLE,
                Hrs_ClinNrsSpec DOUBLE,
                Hrs_ClinNrsSpec_emp DOUBLE,
                Hrs_ClinNrsSpec_ctr DOUBLE,
                Hrs_Pharmacist DOUBLE,
                Hrs_Pharmacist_emp DOUBLE,
                Hrs_Pharmacist_ctr DOUBLE,
                Hrs_Dietician DOUBLE,
                Hrs_Dietician_emp DOUBLE,
                Hrs_Dietician_ctr DOUBLE,
                Hrs_FeedAsst DOUBLE,
                Hrs_FeedAsst_emp DOUBLE,
                Hrs_FeedAsst_ctr DOUBLE,
                Hrs_OT DOUBLE,
                Hrs_OT_emp DOUBLE,
                Hrs_OT_ctr DOUBLE,
                Hrs_PT DOUBLE,
                Hrs_PT_emp DOUBLE,
                Hrs_PT_ctr DOUBLE,
                Hrs_Speech DOUBLE,
                Hrs_Speech_emp DOUBLE,
                Hrs_Speech_ctr DOUBLE,
                Hrs_RespTherapy DOUBLE,
                Hrs_RespTherapy_emp DOUBLE,
                Hrs_RespTherapy_ctr DOUBLE,
                Hrs_MentalHealth DOUBLE,
                Hrs_MentalHealth_emp DOUBLE,
                Hrs_MentalHealth_ctr DOUBLE,
                Hrs_SocialServ DOUBLE,
                Hrs_SocialServ_emp DOUBLE,
                Hrs_SocialServ_ctr DOUBLE,
                Hrs_Activity DOUBLE,
                Hrs_Activity_emp DOUBLE,
                Hrs_Activity_ctr DOUBLE,
                Hrs_BeautyBarber DOUBLE,
                Hrs_BeautyBarber_emp DOUBLE,
                Hrs_BeautyBarber_ctr DOUBLE,
                Hrs_Transport DOUBLE,
                Hrs_Transport_emp DOUBLE,
                Hrs_Transport_ctr DOUBLE,
                Hrs_Laundry DOUBLE,
                Hrs_Laundry_emp DOUBLE,
                Hrs_Laundry_ctr DOUBLE,
                Hrs_Housekeeping DOUBLE,
                Hrs_Housekeeping_emp DOUBLE,
                Hrs_Housekeeping_ctr DOUBLE,
                Hrs_Maintenance DOUBLE,
                Hrs_Maintenance_emp DOUBLE,
                Hrs_Maintenance_ctr DOUBLE,
                Hrs_MedicalRecords DOUBLE,
                Hrs_MedicalRecords_emp DOUBLE,
                Hrs_MedicalRecords_ctr DOUBLE,
                Hrs_Other DOUBLE,
                Hrs_Other_emp DOUBLE,
                Hrs_Other_ctr DOUBLE,
                Hrs_Total DOUBLE,
                Hrs_Total_emp DOUBLE,
                Hrs_Total_ctr DOUBLE
            )
        """)
        
        # Load nurse staffing data
        nurse_files = glob.glob('standardized_PBJ/PBJ_dailynursestaffing_*.csv')
        for file in nurse_files:
            print(f"Loading nurse file: {file}")
            conn.execute(f"""
                INSERT INTO nurse_staffing 
                SELECT * FROM read_csv_auto('{file}',
                    types={{'WorkDate': 'VARCHAR'}},
                    dateformat='%Y%m%d'
                )
            """)
        
        # Load non-nurse staffing data
        non_nurse_files = glob.glob('standardized_NonNurse/PBJ_dailynonnursestaffing_*.csv')
        for file in non_nurse_files:
            print(f"Loading non-nurse file: {file}")
            conn.execute(f"""
                INSERT INTO non_nurse_staffing 
                SELECT * FROM read_csv_auto('{file}',
                    types={{'WorkDate': 'VARCHAR'}},
                    dateformat='%Y%m%d'
                )
            """)
        
        # Convert WorkDate to proper DATE format after loading
        conn.execute("""
            ALTER TABLE nurse_staffing 
            ALTER COLUMN WorkDate TYPE DATE 
            USING strptime(WorkDate, '%Y%m%d')
        """)
        
        conn.execute("""
            ALTER TABLE non_nurse_staffing 
            ALTER COLUMN WorkDate TYPE DATE 
            USING strptime(WorkDate, '%Y%m%d')
        """)
            
        print("Data loading completed successfully")
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Utility functions
def get_staff_categories():
    """Get list of available staff categories"""
    nurse_categories = [
        'RNDON', 'RNDON_emp', 'RNDON_ctr',
        'RNadmin', 'RNadmin_emp', 'RNadmin_ctr',
        'RN', 'RN_emp', 'RN_ctr',
        'LPNadmin', 'LPNadmin_emp', 'LPNadmin_ctr',
        'LPN', 'LPN_emp', 'LPN_ctr',
        'CNA', 'CNA_emp', 'CNA_ctr',
        'NAtrn', 'NAtrn_emp', 'NAtrn_ctr',
        'MedAide', 'MedAide_emp', 'MedAide_ctr',
        'Total_RN',
        'Total_Nurse_Assistant',
        'Total_Nurse_Hours'
    ]
    
    non_nurse_categories = [
        'Admin', 'Admin_emp', 'Admin_ctr', 'Admin_fn',
        'MedDir', 'MedDir_emp', 'MedDir_ctr',
        'OthMD', 'OthMD_emp', 'OthMD_ctr',
        'PA', 'PA_emp', 'PA_ctr',
        'NP', 'NP_emp', 'NP_ctr',
        'ClinNrsSpec', 'ClinNrsSpec_emp', 'ClinNrsSpec_ctr',
        'Pharmacist', 'Pharmacist_emp', 'Pharmacist_ctr',
        'Dietician', 'Dietician_emp', 'Dietician_ctr',
        'FeedAsst', 'FeedAsst_emp', 'FeedAsst_ctr',
        'OT', 'OT_emp', 'OT_ctr',
        'PT', 'PT_emp', 'PT_ctr',
        'Speech', 'Speech_emp', 'Speech_ctr',
        'RespTherapy', 'RespTherapy_emp', 'RespTherapy_ctr',
        'MentalHealth', 'MentalHealth_emp', 'MentalHealth_ctr',
        'SocialServ', 'SocialServ_emp', 'SocialServ_ctr',
        'Activity', 'Activity_emp', 'Activity_ctr',
        'BeautyBarber', 'BeautyBarber_emp', 'BeautyBarber_ctr',
        'Transport', 'Transport_emp', 'Transport_ctr',
        'Laundry', 'Laundry_emp', 'Laundry_ctr',
        'Housekeeping', 'Housekeeping_emp', 'Housekeeping_ctr',
        'Maintenance', 'Maintenance_emp', 'Maintenance_ctr',
        'MedicalRecords', 'MedicalRecords_emp', 'MedicalRecords_ctr',
        'Other', 'Other_emp', 'Other_ctr',
        'Total', 'Total_emp', 'Total_ctr'
    ]
    
    return {
        'nurse': nurse_categories,
        'non_nurse': non_nurse_categories
    }

def get_available_years():
    """Get available years from the database"""
    try:
        result = conn.execute("""
            SELECT DISTINCT EXTRACT(YEAR FROM WorkDate) as year 
            FROM nurse_staffing 
            ORDER BY year
        """).df()
        return [int(year) for year in result['year'].tolist()]
    except:
        return [2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017]

def format_date(date_str):
    """Convert date string to YYYY-MM-DD format"""
    # Accepts either YYYY-MM-DD or already a date object
    if isinstance(date_str, datetime):
        dt = date_str
    else:
        try:
            dt = datetime.strptime(date_str, '%Y-%m-%d')
        except Exception:
            try:
                dt = datetime.strptime(date_str, '%Y/%m/%d')
            except Exception:
                dt = pd.to_datetime(date_str)
    return dt.strftime('%Y-%m-%d')

def format_date_human(date_obj):
    """Format a date as 'Saturday, 8-3-2024'"""
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
        except Exception:
            date_obj = pd.to_datetime(date_obj)
    return date_obj.strftime('%A, %-m-%-d-%Y') if os.name != 'nt' else date_obj.strftime('%A, %#m-%#d-%Y')

def format_date_chart(date_obj):
    """Format a date as '8-3-2024' (no day of week) for charts"""
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
        except Exception:
            date_obj = pd.to_datetime(date_obj)
    return date_obj.strftime('%-m-%-d-%Y') if os.name != 'nt' else date_obj.strftime('%#m-%#d-%Y')

def get_day_name(day_num):
    """Convert numeric day to name (Monday-Sunday)"""
    days = {
        0: 'Monday',
        1: 'Tuesday',
        2: 'Wednesday',
        3: 'Thursday',
        4: 'Friday',
        5: 'Saturday',
        6: 'Sunday'
    }
    return days.get(day_num, 'Unknown')

# Routes
@app.route('/')
def index():
    """Render the main page"""
    staff_categories = get_staff_categories()
    available_years = get_available_years()
    return render_template('index.html', staff_categories=staff_categories, available_years=available_years)

@app.route('/query', methods=['POST'])
def query_data():
    """Handle data queries"""
    try:
        data = request.get_json()
        provnum = data.get('provnum')
        start_date = format_date(data.get('start_date'))
        end_date = format_date(data.get('end_date'))
        staff_category = data.get('staff_category')
        
        # Build the main query (excluding days with 0 MDScensus)
        if staff_category == 'Total_RN':
            hours_column = "COALESCE(Hrs_RN, 0) + COALESCE(Hrs_RNadmin, 0) + COALESCE(Hrs_RNDON, 0)"
            contract_column = "COALESCE(Hrs_RN_ctr, 0) + COALESCE(Hrs_RNadmin_ctr, 0) + COALESCE(Hrs_RNDON_ctr, 0)"
            select_clause = f"""
                {hours_column} as Hours,
                {contract_column} as Contract_Hours,
                {hours_column} as Total_Hours,
                {contract_column} as Total_Contract_Hours
            """
        elif staff_category == 'Total_Nurse_Assistant':
            hours_column = "COALESCE(Hrs_CNA, 0) + COALESCE(Hrs_NAtrn, 0) + COALESCE(Hrs_MedAide, 0)"
            contract_column = "COALESCE(Hrs_CNA_ctr, 0) + COALESCE(Hrs_NAtrn_ctr, 0) + COALESCE(Hrs_MedAide_ctr, 0)"
            select_clause = f"""
                {hours_column} as Hours,
                {contract_column} as Contract_Hours,
                {hours_column} as Total_Hours,
                {contract_column} as Total_Contract_Hours
            """
        elif staff_category == 'Total_Nurse_Hours':
            hours_column = "COALESCE(Hrs_RN, 0) + COALESCE(Hrs_RNadmin, 0) + COALESCE(Hrs_RNDON, 0) + COALESCE(Hrs_LPN, 0) + COALESCE(Hrs_LPNadmin, 0) + COALESCE(Hrs_CNA, 0) + COALESCE(Hrs_NAtrn, 0) + COALESCE(Hrs_MedAide, 0)"
            contract_column = "COALESCE(Hrs_RN_ctr, 0) + COALESCE(Hrs_RNadmin_ctr, 0) + COALESCE(Hrs_RNDON_ctr, 0) + COALESCE(Hrs_LPN_ctr, 0) + COALESCE(Hrs_LPNadmin_ctr, 0) + COALESCE(Hrs_CNA_ctr, 0) + COALESCE(Hrs_NAtrn_ctr, 0) + COALESCE(Hrs_MedAide_ctr, 0)"
            select_clause = f"""
                {hours_column} as Hours,
                {contract_column} as Contract_Hours,
                {hours_column} as Total_Hours,
                {contract_column} as Total_Contract_Hours
            """
        elif staff_category.endswith('_ctr'):
            base_category = staff_category.replace('_ctr', '')
            select_clause = f"""
                Hrs_{base_category} as Hours,
                Hrs_{staff_category} as Contract_Hours,
                Hrs_{base_category} as Total_Hours,
                Hrs_{staff_category} as Total_Contract_Hours
            """
        elif staff_category.endswith('_emp'):
            base_category = staff_category.replace('_emp', '')
            select_clause = f"""
                Hrs_{staff_category} as Hours,
                Hrs_{base_category}_ctr as Contract_Hours,
                Hrs_{staff_category} as Total_Hours,
                Hrs_{base_category}_ctr as Total_Contract_Hours
            """
        else:
            select_clause = f"""
                Hrs_{staff_category} as Hours,
                Hrs_{staff_category}_ctr as Contract_Hours,
                Hrs_{staff_category} as Total_Hours,
                Hrs_{staff_category}_ctr as Total_Contract_Hours
            """
        
        query = f"""
            SELECT 
                WorkDate,
                CASE DAYOFWEEK(WorkDate)
                    WHEN 0 THEN 'Monday'
                    WHEN 1 THEN 'Tuesday'
                    WHEN 2 THEN 'Wednesday'
                    WHEN 3 THEN 'Thursday'
                    WHEN 4 THEN 'Friday'
                    WHEN 5 THEN 'Saturday'
                    WHEN 6 THEN 'Sunday'
                END as DayName,
                MDScensus,
                {select_clause}
            FROM nurse_staffing
            WHERE PROVNUM = '{provnum}'
            AND WorkDate BETWEEN '{start_date}' AND '{end_date}'
            AND MDScensus > 0
            ORDER BY WorkDate
            """
        
        # Execute main query
        df = conn.execute(query).df()
        
        if df.empty:
            return jsonify({
                'error': f'No data found for provider {provnum} in the specified date range'
            }), 404
        
        # Get information about days with 0 MDScensus
        zero_census_query = f"""
            SELECT 
                WorkDate,
                MDScensus
            FROM nurse_staffing
            WHERE PROVNUM = '{provnum}'
            AND WorkDate BETWEEN '{start_date}' AND '{end_date}'
            AND MDScensus = 0
            ORDER BY WorkDate
            """
        
        zero_census_df = conn.execute(zero_census_query).df()
        
        # Calculate HPRD
        if staff_category in ['Total_RN', 'Total_Nurse_Assistant', 'Total_Nurse_Hours']:
            df['HPRD'] = df['Total_Hours'] / df['MDScensus']
        else:
            df['HPRD'] = df['Hours'] / df['MDScensus']
        
        # Create a copy for charts with dates without day of week
        df_chart = df.copy()
        df_chart['WorkDate'] = df_chart['WorkDate'].apply(format_date_chart)
        
        # Format dates for output (with day of week for display)
        df['WorkDate'] = df['WorkDate'].apply(format_date_human)
        
        # Create visualizations using chart-formatted dates
        if staff_category in ['Total_RN', 'Total_Nurse_Assistant', 'Total_Nurse_Hours']:
            daily_chart = create_daily_chart(df_chart, staff_category, use_total=True)
            contract_chart = create_contract_chart(df_chart, staff_category, use_total=True)
            hprd_chart = create_hprd_chart(df_chart, staff_category)
            hours_chart = create_hours_chart(df_chart, staff_category, use_total=True)
            contract_hours_chart = create_contract_hours_chart(df_chart, staff_category, use_total=True)
        else:
            daily_chart = create_daily_chart(df_chart, staff_category)
            contract_chart = create_contract_chart(df_chart, staff_category)
            hprd_chart = create_hprd_chart(df_chart, staff_category)
            hours_chart = create_hours_chart(df_chart, staff_category)
            contract_hours_chart = create_contract_hours_chart(df_chart, staff_category)
        
        # Prepare zero census information
        zero_census_info = {
            'total_days': len(zero_census_df),
            'dates': []
        }
        
        if not zero_census_df.empty:
            # Group consecutive dates
            zero_census_df['WorkDate'] = pd.to_datetime(zero_census_df['WorkDate'])
            zero_census_df = zero_census_df.sort_values('WorkDate')
            
            # Find consecutive date ranges
            current_start = zero_census_df['WorkDate'].iloc[0]
            current_end = current_start
            
            for i in range(1, len(zero_census_df)):
                current_date = zero_census_df['WorkDate'].iloc[i]
                if (current_date - current_end).days == 1:
                    current_end = current_date
                else:
                    # End of consecutive range
                    zero_census_info['dates'].append({
                        'start': format_date_human(current_start),
                        'end': format_date_human(current_end)
                    })
                    current_start = current_date
                    current_end = current_date
            
            # Add the last range
            zero_census_info['dates'].append({
                'start': format_date_human(current_start),
                'end': format_date_human(current_end)
            })
        
        return jsonify({
            'data': df.to_dict(orient='records'),
            'daily_chart': daily_chart,
            'contract_chart': contract_chart,
            'hprd_chart': hprd_chart,
            'hours_chart': hours_chart,
            'contract_hours_chart': contract_hours_chart,
            'zero_census_info': zero_census_info
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing query: {str(e)}'
        }), 500

@app.route('/query_under_threshold', methods=['POST'])
def query_under_threshold():
    """Handle queries for days under a specific threshold"""
    try:
        data = request.get_json()
        provnum = data.get('provnum')
        start_date = format_date(data.get('start_date'))
        end_date = format_date(data.get('end_date'))
        staff_category = data.get('staff_category')
        threshold = float(data.get('threshold', 0))
        day_filter = data.get('day_filter', 'all')
        
        # Handle different staff category naming patterns for threshold query
        if staff_category == 'Total_RN':
            # Special case for Total_RN - combine RN, RN_Admin_Emp, and RN_DON
            hours_column = "COALESCE(Hrs_RN, 0) + COALESCE(Hrs_RNadmin, 0) + COALESCE(Hrs_RNDON, 0)"
        elif staff_category == 'Total_Nurse_Assistant':
            # Special case for Total_Nurse_Assistant - combine CNA, NAtrn, and MedAide
            hours_column = "COALESCE(Hrs_CNA, 0) + COALESCE(Hrs_NAtrn, 0) + COALESCE(Hrs_MedAide, 0)"
        elif staff_category == 'Total_Nurse_Hours':
            # Special case for Total_Nurse_Hours - combine all nurse categories
            hours_column = "COALESCE(Hrs_RN, 0) + COALESCE(Hrs_RNadmin, 0) + COALESCE(Hrs_RNDON, 0) + COALESCE(Hrs_LPN, 0) + COALESCE(Hrs_LPNadmin, 0) + COALESCE(Hrs_CNA, 0) + COALESCE(Hrs_NAtrn, 0) + COALESCE(Hrs_MedAide, 0)"
        elif staff_category.endswith('_ctr'):
            # Already a contract category, don't add _ctr again
            base_category = staff_category.replace('_ctr', '')
            hours_column = f"Hrs_{base_category}"
        elif staff_category.endswith('_emp'):
            # Employee category
            hours_column = f"Hrs_{staff_category}"
        else:
            # Regular category
            hours_column = f"Hrs_{staff_category}"
        
        # Build day filter condition
        day_filter_condition = ""
        if day_filter == 'weekend':
            day_filter_condition = "AND CASE DAYOFWEEK(WorkDate) WHEN 5 THEN 'Saturday' WHEN 6 THEN 'Sunday' END IN ('Saturday', 'Sunday')"
        elif day_filter == 'weekday':
            day_filter_condition = "AND CASE DAYOFWEEK(WorkDate) WHEN 0 THEN 'Monday' WHEN 1 THEN 'Tuesday' WHEN 2 THEN 'Wednesday' WHEN 3 THEN 'Thursday' WHEN 4 THEN 'Friday' END IN ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')"
        elif day_filter != 'all':
            day_filter_condition = f"AND CASE DAYOFWEEK(WorkDate) WHEN 0 THEN 'Monday' WHEN 1 THEN 'Tuesday' WHEN 2 THEN 'Wednesday' WHEN 3 THEN 'Thursday' WHEN 4 THEN 'Friday' WHEN 5 THEN 'Saturday' WHEN 6 THEN 'Sunday' END = '{day_filter}'"
        
        # Build the query
        query = f"""
            WITH daily_hours AS (
                SELECT 
                    WorkDate,
                    CASE DAYOFWEEK(WorkDate)
                        WHEN 0 THEN 'Monday'
                        WHEN 1 THEN 'Tuesday'
                        WHEN 2 THEN 'Wednesday'
                        WHEN 3 THEN 'Thursday'
                        WHEN 4 THEN 'Friday'
                        WHEN 5 THEN 'Saturday'
                        WHEN 6 THEN 'Sunday'
                    END as DayName,
                    {hours_column} as Hours
                FROM nurse_staffing
                WHERE PROVNUM = '{provnum}'
                AND WorkDate BETWEEN '{start_date}' AND '{end_date}'
                AND MDScensus > 0
                {day_filter_condition}
            )
            SELECT 
                COUNT(*) as total_days,
                SUM(CASE WHEN Hours < {threshold} THEN 1 ELSE 0 END) as days_under_threshold,
                MIN(WorkDate) as start_date,
                MAX(WorkDate) as end_date
            FROM daily_hours
            """
        
        # Execute query
        result = conn.execute(query).df()
        
        if result.empty:
            return jsonify({
                'error': f'No data found for provider {provnum} in the specified date range'
            }), 404
        
        # Get the specific days under threshold
        days_query = f"""
            SELECT 
                WorkDate,
                CASE DAYOFWEEK(WorkDate)
                    WHEN 0 THEN 'Monday'
                    WHEN 1 THEN 'Tuesday'
                    WHEN 2 THEN 'Wednesday'
                    WHEN 3 THEN 'Thursday'
                    WHEN 4 THEN 'Friday'
                    WHEN 5 THEN 'Saturday'
                    WHEN 6 THEN 'Sunday'
                END as DayName,
                {hours_column} as Hours,
                MDScensus,
                CASE 
                    WHEN MONTH(WorkDate) IN (1,2,3) THEN 'Q1'
                    WHEN MONTH(WorkDate) IN (4,5,6) THEN 'Q2'
                    WHEN MONTH(WorkDate) IN (7,8,9) THEN 'Q3'
                    WHEN MONTH(WorkDate) IN (10,11,12) THEN 'Q4'
                END as Quarter,
                YEAR(WorkDate) as Year
            FROM nurse_staffing
            WHERE PROVNUM = '{provnum}'
            AND WorkDate BETWEEN '{start_date}' AND '{end_date}'
            AND {hours_column} < {threshold}
            AND MDScensus > 0
            {day_filter_condition}
            ORDER BY WorkDate
            """
        
        days_df = conn.execute(days_query).df()
        days_df['WorkDate'] = days_df['WorkDate'].apply(format_date_human)
        
        # Perform analysis on the days under threshold
        threshold_analysis = {}
        
        if not days_df.empty:
            # Quarter analysis
            quarter_counts = days_df.groupby(['Year', 'Quarter']).size().reset_index(name='count')
            threshold_analysis['quarter_breakdown'] = quarter_counts.to_dict(orient='records')
            
            # Day of week analysis
            dow_counts = days_df.groupby('DayName').size().reset_index(name='count')
            threshold_analysis['day_of_week_breakdown'] = dow_counts.to_dict(orient='records')
            
            # Weekend vs weekday analysis
            weekend_days = days_df[days_df['DayName'].isin(['Saturday', 'Sunday'])].shape[0]
            weekday_days = days_df[~days_df['DayName'].isin(['Saturday', 'Sunday'])].shape[0]
            threshold_analysis['weekend_weekday'] = {
                'weekend': weekend_days,
                'weekday': weekday_days
            }
            
            # Month analysis
            month_counts = days_df.groupby(days_df['WorkDate'].str[:7]).size().reset_index(name='count')
            month_counts.columns = ['month', 'count']
            threshold_analysis['month_breakdown'] = month_counts.to_dict(orient='records')
            
            # Hours range analysis
            hours_ranges = []
            for _, row in days_df.iterrows():
                hours = row['Hours']
                if hours < threshold * 0.5:
                    hours_ranges.append('Severe (< 50% of threshold)')
                elif hours < threshold * 0.75:
                    hours_ranges.append('Moderate (50-75% of threshold)')
                else:
                    hours_ranges.append('Minor (75-100% of threshold)')
            
            hours_range_counts = pd.Series(hours_ranges).value_counts().reset_index()
            hours_range_counts.columns = ['range', 'count']
            threshold_analysis['hours_range_breakdown'] = hours_range_counts.to_dict(orient='records')
        
        return jsonify({
            'total_days': int(result['total_days'].iloc[0]),
            'days_under_threshold': int(result['days_under_threshold'].iloc[0]),
            'start_date': format_date_human(result['start_date'].iloc[0]),
            'end_date': format_date_human(result['end_date'].iloc[0]),
            'threshold': threshold,
            'staff_category': staff_category,
            'days_under': days_df.to_dict(orient='records'),
            'threshold_analysis': threshold_analysis
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing query: {str(e)}'
        }), 500

def create_daily_chart(df, staff_category, use_total=False):
    """Create a line chart of daily regular hours with proper x-axis formatting"""
    fig = go.Figure()
    
    # Get date range for subtitle
    start_date = df['WorkDate'].iloc[0] if not df.empty else ''
    end_date = df['WorkDate'].iloc[-1] if not df.empty else ''
    date_range_text = f" ({start_date} to {end_date})" if start_date and end_date else ""
    
    if use_total and staff_category == 'Total_RN':
        hours_data = df['Total_Hours']
        title_text = f'Total RN Hours Over Time{date_range_text}'
        hover_suffix = 'Total RN Hours'
    elif use_total and staff_category == 'Total_Nurse_Assistant':
        hours_data = df['Total_Hours']
        title_text = f'Total Nurse Assistant Hours Over Time{date_range_text}'
        hover_suffix = 'Total Nurse Assistant Hours'
    elif use_total and staff_category == 'Total_Nurse_Hours':
        hours_data = df['Total_Hours']
        title_text = f'Total Nurse Hours Over Time{date_range_text}'
        hover_suffix = 'Total Nurse Hours'
    else:
        hours_data = df['Hours']
        title_text = f'{staff_category} Hours Over Time{date_range_text}'
        hover_suffix = 'Regular Hours'
    
    fig.add_trace(go.Scatter(
        x=df['WorkDate'],
        y=hours_data,
        name='Regular Hours',
        line=dict(color='#1f77b4', width=2),
        hovertemplate=f'<b>%{{x}}</b><br>{hover_suffix}: %{{y:.1f}}<extra></extra>'
    ))
    
    # Calculate proper x-axis tick configuration with better spacing
    num_dates = len(df)
    if num_dates <= 8:
        # Show all dates if 8 or fewer
        tick_vals = df['WorkDate'].tolist()
        tick_text = df['WorkDate'].tolist()
    else:
        # Show fewer dates for better spacing (max 6-8)
        max_ticks = min(8, max(6, num_dates // 4))
        step = max(1, num_dates // max_ticks)
        tick_vals = df['WorkDate'].iloc[::step].tolist()
        tick_text = df['WorkDate'].iloc[::step].tolist()
        # Always include the last date if not already included
        if tick_vals[-1] != df['WorkDate'].iloc[-1]:
            tick_vals.append(df['WorkDate'].iloc[-1])
            tick_text.append(df['WorkDate'].iloc[-1])
    
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Date',
        yaxis_title='Regular Hours',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=100, b=50),
        xaxis={
            'tickmode': 'array',
            'tickvals': tick_vals,
            'ticktext': tick_text,
            'tickangle': -45,
            'automargin': True
        }
    )
    
    # Add date range as subtitle with smaller font
    if date_range_text:
        fig.add_annotation(
            x=0.5,
            y=1.02,
            xref='paper',
            yref='paper',
            text=date_range_text.strip(' ()'),
            showarrow=False,
            font=dict(size=10, color='gray'),
            xanchor='center'
        )
    
    return json.loads(fig.to_json())

def create_contract_chart(df, staff_category, use_total=False):
    """Create a line chart of contract hours with proper x-axis formatting"""
    fig = go.Figure()
    
    # Get date range for subtitle
    start_date = df['WorkDate'].iloc[0] if not df.empty else ''
    end_date = df['WorkDate'].iloc[-1] if not df.empty else ''
    date_range_text = f" ({start_date} to {end_date})" if start_date and end_date else ""
    
    if use_total and staff_category == 'Total_RN':
        contract_data = df['Total_Contract_Hours']
        title_text = f'Total RN Contract Hours Over Time{date_range_text}'
        hover_suffix = 'Total RN Contract Hours'
    elif use_total and staff_category == 'Total_Nurse_Assistant':
        contract_data = df['Total_Contract_Hours']
        title_text = f'Total Nurse Assistant Contract Hours Over Time{date_range_text}'
        hover_suffix = 'Total Nurse Assistant Contract Hours'
    elif use_total and staff_category == 'Total_Nurse_Hours':
        contract_data = df['Total_Contract_Hours']
        title_text = f'Total Nurse Contract Hours Over Time{date_range_text}'
        hover_suffix = 'Total Nurse Contract Hours'
    else:
        contract_data = df['Contract_Hours']
        title_text = f'{staff_category} Contract Hours Over Time{date_range_text}'
        hover_suffix = 'Contract Hours'
    
    fig.add_trace(go.Scatter(
        x=df['WorkDate'],
        y=contract_data,
        name='Contract Hours',
        line=dict(color='#ff7f0e', width=2),
        hovertemplate=f'<b>%{{x}}</b><br>{hover_suffix}: %{{y:.1f}}<extra></extra>'
    ))
    
    # Calculate proper x-axis tick configuration with better spacing
    num_dates = len(df)
    if num_dates <= 8:
        # Show all dates if 8 or fewer
        tick_vals = df['WorkDate'].tolist()
        tick_text = df['WorkDate'].tolist()
    else:
        # Show fewer dates for better spacing (max 6-8)
        max_ticks = min(8, max(6, num_dates // 4))
        step = max(1, num_dates // max_ticks)
        tick_vals = df['WorkDate'].iloc[::step].tolist()
        tick_text = df['WorkDate'].iloc[::step].tolist()
        # Always include the last date if not already included
        if tick_vals[-1] != df['WorkDate'].iloc[-1]:
            tick_vals.append(df['WorkDate'].iloc[-1])
            tick_text.append(df['WorkDate'].iloc[-1])
    
    fig.update_layout(
        title={
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Date',
        yaxis_title='Contract Hours',
        yaxis_rangemode='tozero',
        hovermode='x unified',
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=100, b=50),
        xaxis={
            'tickmode': 'array',
            'tickvals': tick_vals,
            'ticktext': tick_text,
            'tickangle': -45,
            'automargin': True
        }
    )
    
    # Add date range as subtitle with smaller font
    if date_range_text:
        fig.add_annotation(
            x=0.5,
            y=1.02,
            xref='paper',
            yref='paper',
            text=date_range_text.strip(' ()'),
            showarrow=False,
            font=dict(size=10, color='gray'),
            xanchor='center'
        )
    
    return json.loads(fig.to_json())

def create_hprd_chart(df, staff_category):
    """Create a histogram of HPRD values with improved readability"""
    fig = go.Figure()
    
    # Get date range for subtitle
    start_date = df['WorkDate'].iloc[0] if not df.empty else ''
    end_date = df['WorkDate'].iloc[-1] if not df.empty else ''
    date_range_text = f" ({start_date} to {end_date})" if start_date and end_date else ""
    
    # Calculate better binning based on data range
    min_hprd = df['HPRD'].min()
    max_hprd = df['HPRD'].max()
    range_hprd = max_hprd - min_hprd
    
    # Use fewer bins for better distinction, but at least 10
    nbins = max(10, min(20, int(range_hprd * 10)))
    
    fig.add_trace(go.Histogram(
        x=df['HPRD'],
        name='HPRD Distribution',
        nbinsx=nbins,
        marker_color='#2ca02c',
        opacity=0.8,
        marker_line_color='#1a5f1a',
        marker_line_width=1,
        hovertemplate='<b>HPRD Range</b><br>Count: %{y}<br>Range: %{x}<extra></extra>'
    ))
    
    # Add mean line with better annotation
    mean_hprd = df['HPRD'].mean()
    fig.add_vline(x=mean_hprd, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"Mean: {mean_hprd:.3f}", 
                  annotation_position="top right",
                  annotation_font_size=10,
                  annotation_font_color="red",
                  annotation_bgcolor="rgba(255,255,255,0.8)",
                  annotation_bordercolor="red",
                  annotation_borderwidth=1)
    
    # Add median line
    median_hprd = df['HPRD'].median()
    fig.add_vline(x=median_hprd, line_dash="dot", line_color="orange", line_width=2,
                  annotation_text=f"Median: {median_hprd:.3f}",
                  annotation_position="bottom left",
                  annotation_font_size=10,
                  annotation_font_color="orange",
                  annotation_bgcolor="rgba(255,255,255,0.8)",
                  annotation_bordercolor="orange",
                  annotation_borderwidth=1)
    
    fig.update_layout(
        title=f'{staff_category} HPRD Distribution{date_range_text}',
        xaxis_title='HPRD (Hours Per Resident Day)',
        yaxis_title='Frequency',
        showlegend=False,
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return json.loads(fig.to_json())

def create_hours_chart(df, staff_category, use_total=False):
    """Create a histogram of Hours values with improved readability"""
    fig = go.Figure()
    
    # Get date range for subtitle
    start_date = df['WorkDate'].iloc[0] if not df.empty else ''
    end_date = df['WorkDate'].iloc[-1] if not df.empty else ''
    date_range_text = f" ({start_date} to {end_date})" if start_date and end_date else ""
    
    # Use combined metrics for Total_RN if requested
    if use_total and staff_category == 'Total_RN':
        hours_data = df['Total_Hours']
        title_suffix = 'Total RN'
    elif use_total and staff_category == 'Total_Nurse_Assistant':
        hours_data = df['Total_Hours']
        title_suffix = 'Total Nurse Assistant'
    elif use_total and staff_category == 'Total_Nurse_Hours':
        hours_data = df['Total_Hours']
        title_suffix = 'Total Nurse'
    else:
        hours_data = df['Hours']
        title_suffix = staff_category
    
    # Calculate better binning based on data range
    min_hours = hours_data.min()
    max_hours = hours_data.max()
    range_hours = max_hours - min_hours
    
    # Use fewer bins for better distinction, but at least 10
    nbins = max(10, min(20, int(range_hours / 2)))
    
    fig.add_trace(go.Histogram(
        x=hours_data,
        name='Hours Distribution',
        nbinsx=nbins,
        marker_color='#1f77b4',
        opacity=0.8,
        marker_line_color='#0d47a1',
        marker_line_width=1,
        hovertemplate='<b>Hours Range</b><br>Count: %{y}<br>Range: %{x}<extra></extra>'
    ))
    
    # Add mean line with better annotation
    mean_hours = hours_data.mean()
    fig.add_vline(x=mean_hours, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"Mean: {mean_hours:.1f}", 
                  annotation_position="top right",
                  annotation_font_size=10,
                  annotation_font_color="red",
                  annotation_bgcolor="rgba(255,255,255,0.8)",
                  annotation_bordercolor="red",
                  annotation_borderwidth=1)
    
    # Add median line
    median_hours = hours_data.median()
    fig.add_vline(x=median_hours, line_dash="dot", line_color="orange", line_width=2,
                  annotation_text=f"Median: {median_hours:.1f}",
                  annotation_position="bottom left",
                  annotation_font_size=10,
                  annotation_font_color="orange",
                  annotation_bgcolor="rgba(255,255,255,0.8)",
                  annotation_bordercolor="orange",
                  annotation_borderwidth=1)
    
    fig.update_layout(
        title=f'{title_suffix} Hours Distribution{date_range_text}',
        xaxis_title='Hours',
        yaxis_title='Frequency',
        showlegend=False,
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=80, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return json.loads(fig.to_json())

def create_contract_hours_chart(df, staff_category, use_total=False):
    """Create a lollipop chart (dot plot) of Contract Hours values"""
    fig = go.Figure()
    
    # Get date range for subtitle
    start_date = df['WorkDate'].iloc[0] if not df.empty else ''
    end_date = df['WorkDate'].iloc[-1] if not df.empty else ''
    date_range_text = f" ({start_date} to {end_date})" if start_date and end_date else ""
    
    # Use combined metrics for Total_RN if requested
    if use_total and staff_category == 'Total_RN':
        contract_data = df['Total_Contract_Hours']
        title_suffix = 'Total RN'
    elif use_total and staff_category == 'Total_Nurse_Assistant':
        contract_data = df['Total_Contract_Hours']
        title_suffix = 'Total Nurse Assistant'
    elif use_total and staff_category == 'Total_Nurse_Hours':
        contract_data = df['Total_Contract_Hours']
        title_suffix = 'Total Nurse'
    else:
        contract_data = df['Contract_Hours']
        title_suffix = staff_category
    
    # Filter out negative values and ensure all values are >= 0
    contract_data = contract_data.clip(lower=0)
    
    # Count occurrences of each contract hours value
    value_counts = contract_data.value_counts().sort_index()
    
    # Create bar chart instead of lollipop for better control
    fig.add_trace(go.Bar(
        x=value_counts.index,
        y=value_counts.values,
        name='Contract Hours Distribution',
        marker_color='#ff7f0e',
        marker_line_color='#cc6600',
        marker_line_width=1,
        hovertemplate='<b>Contract Hours: %{x}</b><br>Days: %{y}<extra></extra>'
    ))
    
    # Add mean line with better annotation
    mean_contract_hours = contract_data.mean()
    fig.add_vline(x=mean_contract_hours, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"Mean: {mean_contract_hours:.1f}", 
                  annotation_position="top right",
                  annotation_font_size=10,
                  annotation_font_color="red",
                  annotation_bgcolor="rgba(255,255,255,0.8)",
                  annotation_bordercolor="red",
                  annotation_borderwidth=1)
    
    # Add median line
    median_contract_hours = contract_data.median()
    fig.add_vline(x=median_contract_hours, line_dash="dot", line_color="orange", line_width=2,
                  annotation_text=f"Median: {median_contract_hours:.1f}",
                  annotation_position="bottom left",
                  annotation_font_size=10,
                  annotation_font_color="orange",
                  annotation_bgcolor="rgba(255,255,255,0.8)",
                  annotation_bordercolor="orange",
                  annotation_borderwidth=1)
    
    fig.update_layout(
        title={
            'text': f'{title_suffix} Contract Hours Distribution{date_range_text}',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16}
        },
        xaxis_title='Contract Hours',
        yaxis_title='Number of Days',
        yaxis_rangemode='tozero',
        xaxis=dict(
            range=[0, None],  # Force x-axis to start at 0
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=1
        ),
        showlegend=False,
        template='plotly_white',
        height=400,
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add date range as subtitle with smaller font
    if date_range_text:
        fig.add_annotation(
            x=0.5,
            y=1.02,
            xref='paper',
            yref='paper',
            text=date_range_text.strip(' ()'),
            showarrow=False,
            font=dict(size=10, color='gray'),
            xanchor='center'
        )
    
    return json.loads(fig.to_json())

if __name__ == '__main__':
    try:
        # Load data only once at startup
        print("Loading nurse staffing data...")
        load_data()
        print("Data loading completed successfully")
        
        # Run the app without auto-reloader to prevent double loading
        app.run(debug=True, use_reloader=False, port=5000, host='0.0.0.0')
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
    finally:
        cleanup() 