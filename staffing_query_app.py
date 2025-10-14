from flask import Flask, render_template, request, jsonify
import duckdb
import pandas as pd
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import json
import glob

app = Flask(__name__)

# Initialize DuckDB connection - use persistent database
conn = None

def get_connection():
    """Get or create DuckDB connection"""
    global conn
    if conn is None:
        try:
            conn = duckdb.connect('staffing_data.db')
        except:
            # If database is locked, use a different name
            import time
            timestamp = int(time.time())
            conn = duckdb.connect(f'staffing_data_{timestamp}.db')
        conn.execute("PRAGMA threads=8")
        conn.execute("PRAGMA memory_limit='8GB'")
    return conn

# Load data into DuckDB
def load_data():
    """Load data into DuckDB tables"""
    try:
        conn = get_connection()
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
        
        # Load non-nurse staffing data - DISABLED due to column mismatch issues
        # non_nurse_files = glob.glob('standardized_NonNurse/PBJ_dailynonnursestaffing_*.csv')
        # for file in non_nurse_files:
        #     print(f"Loading non-nurse file: {file}")
        #     conn.execute(f"""
        #         INSERT INTO non_nurse_staffing 
        #         SELECT * FROM read_csv_auto('{file}',
        #             types={{'WorkDate': 'VARCHAR'}},
        #             dateformat='%Y%m%d'
        #         )
        #     """)
        print("Skipping non-nurse staffing data loading due to column mismatch issues")
            
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
        'RN', 'LPN', 'CNA', 'RN_ctr', 'LPN_ctr', 'CNA_ctr'
    ]
    non_nurse_categories = [
        'Admin', 'MedDir', 'OthMD', 'PA', 'NP', 'ClinNrsSpec',
        'Pharmacist', 'Dietician', 'FeedAsst', 'OT', 'PT', 'RespTher',
        'SpcLangPath', 'TherRecSpec', 'QualActvProf', 'QualSocWrk'
    ]
    return {
        'nurse': nurse_categories,
        'non_nurse': non_nurse_categories
    }

def format_date(date_str):
    """Convert date string to YYYY-MM-DD format"""
    return datetime.strptime(date_str, '%Y-%m-%d').strftime('%Y-%m-%d')

# Routes
@app.route('/')
def index():
    """Render the main page"""
    staff_categories = get_staff_categories()
    return render_template('index.html', staff_categories=staff_categories)

@app.route('/date_range')
def get_date_range():
    """Get available date range for a provider"""
    try:
        provnum = request.args.get('provnum')
        if not provnum:
            return jsonify({'error': 'Provider number required'}), 400
        
        conn = get_connection()
        query = """
        SELECT 
            MIN(WorkDate) as min_date,
            MAX(WorkDate) as max_date
        FROM nurse_staffing 
        WHERE PROVNUM = ?
        """
        
        result = conn.execute(query, [provnum]).fetchone()
        
        if result and result[0]:
            return jsonify({
                'min_date': result[0].strftime('%Y-%m-%d'),
                'max_date': result[1].strftime('%Y-%m-%d')
            })
        else:
            return jsonify({'error': 'No data found for provider'}), 404
            
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
        
        # Get connection
        conn = get_connection()
        
        # Determine if this is a nurse or non-nurse category
        staff_categories = get_staff_categories()
        table_name = 'nurse_staffing' if staff_category in staff_categories['nurse'] else 'non_nurse_staffing'
        
        # Build the query with all metrics
        query = f"""
        SELECT 
            WorkDate,
            DAYOFWEEK(WorkDate) as DayOfWeek,
            MDScensus as Census,
            Hrs_{staff_category} as Hours,
            Hrs_{staff_category}_ctr as Contract_Hours,
            (Hrs_{staff_category} + Hrs_{staff_category}_ctr) as Total_Hours,
            CASE 
                WHEN (Hrs_{staff_category} + Hrs_{staff_category}_ctr) > 0 
                THEN (Hrs_{staff_category}_ctr / (Hrs_{staff_category} + Hrs_{staff_category}_ctr)) * 100 
                ELSE 0 
            END as Contract_Percentage,
            CASE 
                WHEN MDScensus > 0 
                THEN (Hrs_{staff_category} + Hrs_{staff_category}_ctr) / MDScensus 
                ELSE 0 
            END as Total_HPRD
        FROM {table_name}
        WHERE PROVNUM = '{provnum}'
        AND WorkDate BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY WorkDate
        """
        
        # Execute query
        df = conn.execute(query).df()
        
        if df.empty:
            return jsonify({
                'error': f'No data found for provider {provnum} in the specified date range'
            }), 404
        
        # Calculate additional metrics
        df['HPRD'] = df['Hours'] / df['Census']
        df['Contract_HPRD'] = df['Contract_Hours'] / df['Census']
        
        # Create visualizations
        daily_chart = create_daily_chart(df, staff_category)
        hprd_chart = create_hprd_chart(df, staff_category)
        
        return jsonify({
            'data': df.to_dict(orient='records'),
            'daily_chart': daily_chart,
            'hprd_chart': hprd_chart
        })
        
    except Exception as e:
        return jsonify({
            'error': f'Error processing query: {str(e)}'
        }), 500

def create_daily_chart(df, staff_category):
    """Create a line chart of daily hours"""
    fig = go.Figure()
    
    # Add regular hours
    fig.add_trace(go.Scatter(
        x=df['WorkDate'],
        y=df['Hours'],
        name='Regular Hours',
        line=dict(color='blue')
    ))
    
    # Add contract hours
    fig.add_trace(go.Scatter(
        x=df['WorkDate'],
        y=df['Contract_Hours'],
        name='Contract Hours',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title=f'Daily {staff_category} Hours',
        xaxis_title='Date',
        yaxis_title='Hours',
        hovermode='x unified'
    )
    
    return json.loads(fig.to_json())

def create_hprd_chart(df, staff_category):
    """Create a histogram of HPRD values"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df['HPRD'],
        name='HPRD Distribution',
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f'{staff_category} HPRD Distribution',
        xaxis_title='HPRD',
        yaxis_title='Frequency',
        showlegend=False
    )
    
    return json.loads(fig.to_json())

if __name__ == '__main__':
    try:
        # Check if data already exists before loading
        conn = get_connection()
        try:
            # Check if nurse_staffing table exists and has data
            result = conn.execute("SELECT COUNT(*) FROM nurse_staffing LIMIT 1").fetchone()
            if result and result[0] > 0:
                print("Data already loaded, skipping data loading step")
            else:
                print("No data found, loading data...")
                load_data()
        except:
            print("Database doesn't exist or is empty, loading data...")
            load_data()
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Failed to start application: {str(e)}") 