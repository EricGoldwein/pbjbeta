from flask import Flask, render_template, request, jsonify
import duckdb
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json
import glob
import atexit
import tempfile

app = Flask(__name__)

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
    """Load non-nurse staffing data into DuckDB tables"""
    try:
        # Create table for non-nurse staffing
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
                Hrs_OTasst DOUBLE,
                Hrs_OTasst_emp DOUBLE,
                Hrs_OTasst_ctr DOUBLE,
                Hrs_OTaide DOUBLE,
                Hrs_OTaide_emp DOUBLE,
                Hrs_OTaide_ctr DOUBLE,
                Hrs_PT DOUBLE,
                Hrs_PT_emp DOUBLE,
                Hrs_PT_ctr DOUBLE,
                Hrs_PTasst DOUBLE,
                Hrs_PTasst_emp DOUBLE,
                Hrs_PTasst_ctr DOUBLE,
                Hrs_PTaide DOUBLE,
                Hrs_PTaide_emp DOUBLE,
                Hrs_PTaide_ctr DOUBLE,
                Hrs_RespTher DOUBLE,
                Hrs_RespTher_emp DOUBLE,
                Hrs_RespTher_ctr DOUBLE,
                Hrs_RespTech DOUBLE,
                Hrs_RespTech_emp DOUBLE,
                Hrs_RespTech_ctr DOUBLE,
                Hrs_SpcLangPath DOUBLE,
                Hrs_SpcLangPath_emp DOUBLE,
                Hrs_SpcLangPath_ctr DOUBLE,
                Hrs_TherRecSpec DOUBLE,
                Hrs_TherRecSpec_emp DOUBLE,
                Hrs_TherRecSpec_ctr DOUBLE,
                Hrs_QualActvProf DOUBLE,
                Hrs_QualActvProf_emp DOUBLE,
                Hrs_QualActvProf_ctr DOUBLE,
                Hrs_OthActv DOUBLE,
                Hrs_OthActv_emp DOUBLE,
                Hrs_OthActv_ctr DOUBLE,
                Hrs_QualSocWrk DOUBLE,
                Hrs_QualSocWrk_emp DOUBLE,
                Hrs_QualSocWrk_ctr DOUBLE,
                Hrs_OthSocWrk DOUBLE,
                Hrs_OthSocWrk_emp DOUBLE,
                Hrs_OthSocWrk_ctr DOUBLE,
                Hrs_MHSvc DOUBLE,
                Hrs_MHSvc_emp DOUBLE,
                Hrs_MHSvc_ctr DOUBLE
            )
        """)
        
        # Load non-nurse staffing data
        non_nurse_files = glob.glob('standardized_NonNurse/PBJ_dailynonnursestaffing_*.csv')
        for file in non_nurse_files:
            print(f"Loading non-nurse file: {file}")
            # Specify data types for columns to avoid DtypeWarning and speed up loading
            dtype_spec = {
                'PROVNUM': 'str',
                'PROVNAME': 'str',
                'CITY': 'str',
                'STATE': 'str',
                'COUNTY_NAME': 'str',
                'COUNTY_FIPS': 'str',
                'CY_Qtr': 'str',
                'WorkDate': 'str',
                'MDScensus': 'float64',
                'Hrs_Admin': 'float64',
                'Hrs_Admin_emp': 'float64',
                'Hrs_Admin_ctr': 'float64',
                'Hrs_MedDir': 'float64',
                'Hrs_MedDir_emp': 'float64',
                'Hrs_MedDir_ctr': 'float64',
                'Hrs_OthMD': 'float64',
                'Hrs_OthMD_emp': 'float64',
                'Hrs_OthMD_ctr': 'float64',
                'Hrs_PA': 'float64',
                'Hrs_PA_emp': 'float64',
                'Hrs_PA_ctr': 'float64',
                'Hrs_NP': 'float64',
                'Hrs_NP_emp': 'float64',
                'Hrs_NP_ctr': 'float64',
                'Hrs_ClinNrsSpec': 'float64',
                'Hrs_ClinNrsSpec_emp': 'float64',
                'Hrs_ClinNrsSpec_ctr': 'float64',
                'Hrs_Pharmacist': 'float64',
                'Hrs_Pharmacist_emp': 'float64',
                'Hrs_Pharmacist_ctr': 'float64',
                'Hrs_Dietician': 'float64',
                'Hrs_Dietician_emp': 'float64',
                'Hrs_Dietician_ctr': 'float64',
                'Hrs_FeedAsst': 'float64',
                'Hrs_FeedAsst_emp': 'float64',
                'Hrs_FeedAsst_ctr': 'float64',
                'Hrs_OT': 'float64',
                'Hrs_OT_emp': 'float64',
                'Hrs_OT_ctr': 'float64',
                'Hrs_OTasst': 'float64',
                'Hrs_OTasst_emp': 'float64',
                'Hrs_OTasst_ctr': 'float64',
                'Hrs_OTaide': 'float64',
                'Hrs_OTaide_emp': 'float64',
                'Hrs_OTaide_ctr': 'float64',
                'Hrs_PT': 'float64',
                'Hrs_PT_emp': 'float64',
                'Hrs_PT_ctr': 'float64',
                'Hrs_PTasst': 'float64',
                'Hrs_PTasst_emp': 'float64',
                'Hrs_PTasst_ctr': 'float64',
                'Hrs_PTaide': 'float64',
                'Hrs_PTaide_emp': 'float64',
                'Hrs_PTaide_ctr': 'float64',
                'Hrs_RespTher': 'float64',
                'Hrs_RespTher_emp': 'float64',
                'Hrs_RespTher_ctr': 'float64',
                'Hrs_RespTech': 'float64',
                'Hrs_RespTech_emp': 'float64',
                'Hrs_RespTech_ctr': 'float64',
                'Hrs_SpcLangPath': 'float64',
                'Hrs_SpcLangPath_emp': 'float64',
                'Hrs_SpcLangPath_ctr': 'float64',
                'Hrs_TherRecSpec': 'float64',
                'Hrs_TherRecSpec_emp': 'float64',
                'Hrs_TherRecSpec_ctr': 'float64',
                'Hrs_QualActvProf': 'float64',
                'Hrs_QualActvProf_emp': 'float64',
                'Hrs_QualActvProf_ctr': 'float64',
                'Hrs_OthActv': 'float64',
                'Hrs_OthActv_emp': 'float64',
                'Hrs_OthActv_ctr': 'float64',
                'Hrs_QualSocWrk': 'float64',
                'Hrs_QualSocWrk_emp': 'float64',
                'Hrs_QualSocWrk_ctr': 'float64',
                'Hrs_OthSocWrk': 'float64',
                'Hrs_OthSocWrk_emp': 'float64',
                'Hrs_OthSocWrk_ctr': 'float64',
                'Hrs_MHSvc': 'float64',
                'Hrs_MHSvc_emp': 'float64',
                'Hrs_MHSvc_ctr': 'float64'
            }
            df = pd.read_csv(file, dtype=dtype_spec, low_memory=False)
            if 'Hrs_Admin_fn' in df.columns:
                conn.execute(f"""
                    INSERT INTO non_nurse_staffing 
                    SELECT * FROM read_csv_auto('{file}',
                        types={{'WorkDate': 'VARCHAR'}},
                        dateformat='%Y%m%d'
                    )
                """)
            else:
                # Exclude Hrs_Admin_fn from the insertion
                df.drop(columns=['Hrs_Admin_fn'], inplace=True, errors='ignore')
                df.to_csv(file, index=False)
                conn.execute(f"""
                    INSERT INTO non_nurse_staffing 
                    SELECT * FROM read_csv_auto('{file}',
                        types={{'WorkDate': 'VARCHAR'}},
                        dateformat='%Y%m%d'
                    )
                """)
        
        # Convert WorkDate to proper DATE format after loading
        conn.execute("""
            ALTER TABLE non_nurse_staffing 
            ALTER COLUMN WorkDate TYPE DATE 
            USING strptime(WorkDate, '%Y%m%d')
        """)
            
        print("Non-nurse data loading completed successfully")
        
    except Exception as e:
        print(f"Error loading non-nurse data: {str(e)}")
        raise

# Utility functions
def get_staff_categories():
    """Get list of available non-nurse staff categories"""
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
        'OTasst', 'OTasst_emp', 'OTasst_ctr',
        'OTaide', 'OTaide_emp', 'OTaide_ctr',
        'PT', 'PT_emp', 'PT_ctr',
        'PTasst', 'PTasst_emp', 'PTasst_ctr',
        'PTaide', 'PTaide_emp', 'PTaide_ctr',
        'RespTher', 'RespTher_emp', 'RespTher_ctr',
        'RespTech', 'RespTech_emp', 'RespTech_ctr',
        'SpcLangPath', 'SpcLangPath_emp', 'SpcLangPath_ctr',
        'TherRecSpec', 'TherRecSpec_emp', 'TherRecSpec_ctr',
        'QualActvProf', 'QualActvProf_emp', 'QualActvProf_ctr',
        'OthActv', 'OthActv_emp', 'OthActv_ctr',
        'QualSocWrk', 'QualSocWrk_emp', 'QualSocWrk_ctr',
        'OthSocWrk', 'OthSocWrk_emp', 'OthSocWrk_ctr',
        'MHSvc', 'MHSvc_emp', 'MHSvc_ctr'
    ]
    return {
        'non_nurse': non_nurse_categories
    }

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

def load_data_for_provnum(provnum):
    """Load non-nurse staffing data for a specific PROVNUM into DuckDB tables"""
    try:
        # Create table for non-nurse staffing if it doesn't exist
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
                Hrs_OTasst DOUBLE,
                Hrs_OTasst_emp DOUBLE,
                Hrs_OTasst_ctr DOUBLE,
                Hrs_OTaide DOUBLE,
                Hrs_OTaide_emp DOUBLE,
                Hrs_OTaide_ctr DOUBLE,
                Hrs_PT DOUBLE,
                Hrs_PT_emp DOUBLE,
                Hrs_PT_ctr DOUBLE,
                Hrs_PTasst DOUBLE,
                Hrs_PTasst_emp DOUBLE,
                Hrs_PTasst_ctr DOUBLE,
                Hrs_PTaide DOUBLE,
                Hrs_PTaide_emp DOUBLE,
                Hrs_PTaide_ctr DOUBLE,
                Hrs_RespTher DOUBLE,
                Hrs_RespTher_emp DOUBLE,
                Hrs_RespTher_ctr DOUBLE,
                Hrs_RespTech DOUBLE,
                Hrs_RespTech_emp DOUBLE,
                Hrs_RespTech_ctr DOUBLE,
                Hrs_SpcLangPath DOUBLE,
                Hrs_SpcLangPath_emp DOUBLE,
                Hrs_SpcLangPath_ctr DOUBLE,
                Hrs_TherRecSpec DOUBLE,
                Hrs_TherRecSpec_emp DOUBLE,
                Hrs_TherRecSpec_ctr DOUBLE,
                Hrs_QualActvProf DOUBLE,
                Hrs_QualActvProf_emp DOUBLE,
                Hrs_QualActvProf_ctr DOUBLE,
                Hrs_OthActv DOUBLE,
                Hrs_OthActv_emp DOUBLE,
                Hrs_OthActv_ctr DOUBLE,
                Hrs_QualSocWrk DOUBLE,
                Hrs_QualSocWrk_emp DOUBLE,
                Hrs_QualSocWrk_ctr DOUBLE,
                Hrs_OthSocWrk DOUBLE,
                Hrs_OthSocWrk_emp DOUBLE,
                Hrs_OthSocWrk_ctr DOUBLE,
                Hrs_MHSvc DOUBLE,
                Hrs_MHSvc_emp DOUBLE,
                Hrs_MHSvc_ctr DOUBLE
            )
        """)
        
        # Load non-nurse staffing data for the specified PROVNUM
        non_nurse_files = glob.glob('standardized_NonNurse/PBJ_dailynonnursestaffing_*.csv')
        for file in non_nurse_files:
            print(f"Loading non-nurse file: {file} for PROVNUM: {provnum}")
            # Specify data types for columns to avoid DtypeWarning and speed up loading
            dtype_spec = {
                'PROVNUM': 'str',
                'PROVNAME': 'str',
                'CITY': 'str',
                'STATE': 'str',
                'COUNTY_NAME': 'str',
                'COUNTY_FIPS': 'str',
                'CY_Qtr': 'str',
                'WorkDate': 'str',
                'MDScensus': 'float64',
                'Hrs_Admin': 'float64',
                'Hrs_Admin_emp': 'float64',
                'Hrs_Admin_ctr': 'float64',
                'Hrs_MedDir': 'float64',
                'Hrs_MedDir_emp': 'float64',
                'Hrs_MedDir_ctr': 'float64',
                'Hrs_OthMD': 'float64',
                'Hrs_OthMD_emp': 'float64',
                'Hrs_OthMD_ctr': 'float64',
                'Hrs_PA': 'float64',
                'Hrs_PA_emp': 'float64',
                'Hrs_PA_ctr': 'float64',
                'Hrs_NP': 'float64',
                'Hrs_NP_emp': 'float64',
                'Hrs_NP_ctr': 'float64',
                'Hrs_ClinNrsSpec': 'float64',
                'Hrs_ClinNrsSpec_emp': 'float64',
                'Hrs_ClinNrsSpec_ctr': 'float64',
                'Hrs_Pharmacist': 'float64',
                'Hrs_Pharmacist_emp': 'float64',
                'Hrs_Pharmacist_ctr': 'float64',
                'Hrs_Dietician': 'float64',
                'Hrs_Dietician_emp': 'float64',
                'Hrs_Dietician_ctr': 'float64',
                'Hrs_FeedAsst': 'float64',
                'Hrs_FeedAsst_emp': 'float64',
                'Hrs_FeedAsst_ctr': 'float64',
                'Hrs_OT': 'float64',
                'Hrs_OT_emp': 'float64',
                'Hrs_OT_ctr': 'float64',
                'Hrs_OTasst': 'float64',
                'Hrs_OTasst_emp': 'float64',
                'Hrs_OTasst_ctr': 'float64',
                'Hrs_OTaide': 'float64',
                'Hrs_OTaide_emp': 'float64',
                'Hrs_OTaide_ctr': 'float64',
                'Hrs_PT': 'float64',
                'Hrs_PT_emp': 'float64',
                'Hrs_PT_ctr': 'float64',
                'Hrs_PTasst': 'float64',
                'Hrs_PTasst_emp': 'float64',
                'Hrs_PTasst_ctr': 'float64',
                'Hrs_PTaide': 'float64',
                'Hrs_PTaide_emp': 'float64',
                'Hrs_PTaide_ctr': 'float64',
                'Hrs_RespTher': 'float64',
                'Hrs_RespTher_emp': 'float64',
                'Hrs_RespTher_ctr': 'float64',
                'Hrs_RespTech': 'float64',
                'Hrs_RespTech_emp': 'float64',
                'Hrs_RespTech_ctr': 'float64',
                'Hrs_SpcLangPath': 'float64',
                'Hrs_SpcLangPath_emp': 'float64',
                'Hrs_SpcLangPath_ctr': 'float64',
                'Hrs_TherRecSpec': 'float64',
                'Hrs_TherRecSpec_emp': 'float64',
                'Hrs_TherRecSpec_ctr': 'float64',
                'Hrs_QualActvProf': 'float64',
                'Hrs_QualActvProf_emp': 'float64',
                'Hrs_QualActvProf_ctr': 'float64',
                'Hrs_OthActv': 'float64',
                'Hrs_OthActv_emp': 'float64',
                'Hrs_OthActv_ctr': 'float64',
                'Hrs_QualSocWrk': 'float64',
                'Hrs_QualSocWrk_emp': 'float64',
                'Hrs_QualSocWrk_ctr': 'float64',
                'Hrs_OthSocWrk': 'float64',
                'Hrs_OthSocWrk_emp': 'float64',
                'Hrs_OthSocWrk_ctr': 'float64',
                'Hrs_MHSvc': 'float64',
                'Hrs_MHSvc_emp': 'float64',
                'Hrs_MHSvc_ctr': 'float64'
            }
            df = pd.read_csv(file, dtype=dtype_spec, low_memory=False)
            df_filtered = df[df['PROVNUM'] == provnum]
            if not df_filtered.empty:
                # Insert filtered data into the database
                df_filtered.to_csv(file, index=False)
                conn.execute(f"""
                    INSERT INTO non_nurse_staffing 
                    SELECT * FROM read_csv_auto('{file}',
                        types={{'WorkDate': 'VARCHAR'}},
                        dateformat='%Y%m%d'
                    )
                """)
        
        # Convert WorkDate to proper DATE format after loading
        conn.execute("""
            ALTER TABLE non_nurse_staffing 
            ALTER COLUMN WorkDate TYPE DATE 
            USING strptime(WorkDate, '%Y%m%d')
        """)
            
        print(f"Data loading for PROVNUM {provnum} completed successfully")
        
    except Exception as e:
        print(f"Error loading data for PROVNUM {provnum}: {str(e)}")
        raise

# Routes
@app.route('/')
def index():
    """Render the main page with a form to select PROVNUM"""
    return render_template('index.html')

@app.route('/load_data', methods=['POST'])
def load_data_for_provnum():
    """Load data for the specified PROVNUM"""
    try:
        provnum = request.form.get('provnum')
        if not provnum:
            return jsonify({'error': 'PROVNUM is required'}), 400
        
        # Load data for the specified PROVNUM
        load_data_for_provnum(provnum)
        
        return jsonify({'success': f'Data for PROVNUM {provnum} loaded successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/query', methods=['POST'])
def query_data():
    """Handle data queries"""
    try:
        data = request.get_json()
        provnum = data.get('provnum')
        start_date = format_date(data.get('start_date'))
        end_date = format_date(data.get('end_date'))
        staff_category = data.get('staff_category')
        
        # Build the query
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
                Hrs_{staff_category} as Hours,
                Hrs_{staff_category}_ctr as Contract_Hours
            FROM non_nurse_staffing
            WHERE PROVNUM = '{provnum}'
            AND WorkDate BETWEEN '{start_date}' AND '{end_date}'
        """
        df = conn.execute(query).fetchdf()
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/query_under_threshold', methods=['POST'])
def query_under_threshold():
    """Handle queries for data under a certain threshold"""
    try:
        data = request.get_json()
        threshold = data.get('threshold', 0)
        staff_category = data.get('staff_category')
        
        # Build the query
        query = f"""
            SELECT 
                PROVNUM, PROVNAME, CITY, STATE, 
                AVG(Hrs_{staff_category}) as Avg_Hours
            FROM non_nurse_staffing
            GROUP BY PROVNUM, PROVNAME, CITY, STATE
            HAVING AVG(Hrs_{staff_category}) < {threshold}
        """
        df = conn.execute(query).fetchdf()
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)})

# Chart creation functions
# These functions will be similar to those in nurse_staffing_app.py, adapted for non-nurse data

def create_daily_chart(df, staff_category):
    """Create a daily chart for the given staff category"""
    fig = px.line(df, x='WorkDate', y='Hours', title=f'Daily Hours for {staff_category}')
    return fig.to_html()


def create_hprd_chart(df, staff_category):
    """Create a Hours Per Resident Day (HPRD) chart for the given staff category"""
    df['HPRD'] = df['Hours'] / df['MDScensus']
    fig = px.bar(df, x='WorkDate', y='HPRD', title=f'HPRD for {staff_category}')
    return fig.to_html()

if __name__ == '__main__':
    try:
        load_data()
        app.run(debug=True)
    except Exception as e:
        print(f"Failed to start application: {str(e)}")
    finally:
        cleanup() 