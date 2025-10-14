#!/usr/bin/env python3
"""
Dedicated Flask app for facility CCN 225500
Loads all nurse staffing data for this specific facility for comprehensive analysis
"""

import os
import glob
import pandas as pd
import duckdb
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import json

app = Flask(__name__)

# Global connection
conn = None

def get_connection():
    """Get or create DuckDB connection"""
    global conn
    if conn is None:
        try:
            conn = duckdb.connect('facility_225500_data.db')
        except Exception as e:
            # If database is locked, create a new one with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_name = f'facility_225500_data_{timestamp}.db'
            print(f"Creating new database: {db_name}")
            conn = duckdb.connect(db_name)
    return conn

def load_facility_data():
    """Load all nurse staffing data for facility 225500"""
    print("Loading facility 225500 data...")
    
    db_conn = get_connection()
    
    # Create table for facility data
    db_conn.execute("""
        CREATE TABLE IF NOT EXISTS facility_staffing (
            WorkDate DATE,
            PROVNUM VARCHAR,
            Employee_Hours DECIMAL(10,2),
            Contract_Hours DECIMAL(10,2),
            Total_Hours DECIMAL(10,2),
            MDScensus DECIMAL(10,2),
            HPRD DECIMAL(10,2),
            Contract_Percentage DECIMAL(10,2),
            Position VARCHAR,
            Quarter VARCHAR
        )
    """)
    
    # Clear existing data
    db_conn.execute("DELETE FROM facility_staffing")
    
    # Load all nurse files
    nurse_files = glob.glob('standardized_PBJ/PBJ_dailynursestaffing_*.csv')
    nurse_files.sort()
    
    total_records = 0
    
    for file_path in nurse_files:
        print(f"Processing: {os.path.basename(file_path)}")
        
        try:
            # Read the CSV
            df = pd.read_csv(file_path)
            
            # Filter for facility 225500
            facility_data = df[df['PROVNUM'] == '225500'].copy()
            
            if len(facility_data) == 0:
                print(f"  No data for facility 225500 in {os.path.basename(file_path)}")
                continue
            
            # Extract quarter from filename
            filename = os.path.basename(file_path)
            quarter = filename.replace('PBJ_dailynursestaffing_', '').replace('.csv', '')
            
            # Process each position
            positions = ['RN', 'LPN', 'CNA', 'Total_RN', 'Total_LPN', 'Total_CNA']
            
            for position in positions:
                if position in facility_data.columns:
                    position_data = facility_data[['WorkDate', 'PROVNUM', f'{position}_Employee_Hours', 
                                                 f'{position}_Contract_Hours', 'MDScensus']].copy()
                    
                    # Rename columns for consistency
                    position_data.columns = ['WorkDate', 'PROVNUM', 'Employee_Hours', 'Contract_Hours', 'MDScensus']
                    
                    # Calculate derived fields
                    position_data['Total_Hours'] = position_data['Employee_Hours'] + position_data['Contract_Hours']
                    position_data['HPRD'] = position_data['Total_Hours'] / position_data['MDScensus']
                    position_data['Contract_Percentage'] = (position_data['Contract_Hours'] / position_data['Total_Hours'] * 100).fillna(0)
                    position_data['Position'] = position
                    position_data['Quarter'] = quarter
                    
                    # Filter out rows with 0 census
                    position_data = position_data[position_data['MDScensus'] > 0]
                    
                    if len(position_data) > 0:
                        # Insert into database
                        db_conn.execute("""
                            INSERT INTO facility_staffing 
                            (WorkDate, PROVNUM, Employee_Hours, Contract_Hours, Total_Hours, 
                             MDScensus, HPRD, Contract_Percentage, Position, Quarter)
                            SELECT * FROM position_data
                        """)
                        
                        total_records += len(position_data)
                        print(f"  Added {len(position_data)} records for {position}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    print(f"Total records loaded: {total_records}")
    
    # Create indexes for better performance
    db_conn.execute("CREATE INDEX IF NOT EXISTS idx_workdate ON facility_staffing(WorkDate)")
    db_conn.execute("CREATE INDEX IF NOT EXISTS idx_position ON facility_staffing(Position)")
    db_conn.execute("CREATE INDEX IF NOT EXISTS idx_quarter ON facility_staffing(Quarter)")
    
    return total_records

@app.route('/')
def index():
    """Main page"""
    return render_template('facility_225500.html')

@app.route('/query', methods=['POST'])
def query_data():
    """Query facility data with filters"""
    try:
        data = request.get_json()
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        position = data.get('position', 'RN')
        quarter = data.get('quarter', '')
        
        db_conn = get_connection()
        
        # Build query
        query = """
            SELECT 
                WorkDate,
                Position,
                Employee_Hours,
                Contract_Hours,
                Total_Hours,
                MDScensus as Census,
                HPRD,
                Contract_Percentage,
                Quarter
            FROM facility_staffing
            WHERE 1=1
        """
        
        params = []
        
        if start_date:
            query += " AND WorkDate >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND WorkDate <= ?"
            params.append(end_date)
        
        if position:
            query += " AND Position = ?"
            params.append(position)
        
        if quarter:
            query += " AND Quarter = ?"
            params.append(quarter)
        
        query += " ORDER BY WorkDate, Position"
        
        # Execute query
        result = db_conn.execute(query, params).fetchdf()
        
        if len(result) == 0:
            return jsonify({'error': 'No data found for the specified criteria'})
        
        # Calculate summary statistics
        summary = {
            'total_days': len(result),
            'avg_hours': result['Total_Hours'].mean(),
            'avg_hprd': result['HPRD'].mean(),
            'avg_contract_pct': result['Contract_Percentage'].mean(),
            'avg_census': result['Census'].mean(),
            'min_date': result['WorkDate'].min(),
            'max_date': result['WorkDate'].max(),
            'total_hours': result['Total_Hours'].sum(),
            'total_contract_hours': result['Contract_Hours'].sum()
        }
        
        # Create charts
        charts = create_charts(result)
        
        return jsonify({
            'data': result.to_dict('records'),
            'summary': summary,
            'charts': charts
        })
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/quarters')
def get_quarters():
    """Get available quarters"""
    try:
        db_conn = get_connection()
        result = db_conn.execute("SELECT DISTINCT Quarter FROM facility_staffing ORDER BY Quarter").fetchdf()
        return jsonify(result['Quarter'].tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/positions')
def get_positions():
    """Get available positions"""
    try:
        db_conn = get_connection()
        result = db_conn.execute("SELECT DISTINCT Position FROM facility_staffing ORDER BY Position").fetchdf()
        return jsonify(result['Position'].tolist())
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/date_range')
def get_date_range():
    """Get date range for facility"""
    try:
        db_conn = get_connection()
        result = db_conn.execute("SELECT MIN(WorkDate) as min_date, MAX(WorkDate) as max_date FROM facility_staffing").fetchdf()
        
        if len(result) > 0:
            return jsonify({
                'min_date': result['min_date'].iloc[0],
                'max_date': result['max_date'].iloc[0]
            })
        else:
            return jsonify({'error': 'No data found'})
    except Exception as e:
        return jsonify({'error': str(e)})

def create_charts(data):
    """Create chart data for visualization"""
    charts = {}
    
    # Daily HPRD chart
    daily_data = data.groupby('WorkDate').agg({
        'HPRD': 'mean',
        'Total_Hours': 'sum',
        'Census': 'mean',
        'Contract_Percentage': 'mean'
    }).reset_index()
    
    charts['daily_hprd'] = {
        'data': [{
            'x': daily_data['WorkDate'].tolist(),
            'y': daily_data['HPRD'].tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'HPRD',
            'line': {'color': '#1f77b4'}
        }],
        'layout': {
            'title': 'Daily HPRD Trend',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'HPRD'},
            'height': 400
        }
    }
    
    # Contract percentage chart
    charts['contract_pct'] = {
        'data': [{
            'x': daily_data['WorkDate'].tolist(),
            'y': daily_data['Contract_Percentage'].tolist(),
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Contract %',
            'line': {'color': '#ff7f0e'}
        }],
        'layout': {
            'title': 'Contract Percentage Trend',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Contract Percentage (%)'},
            'height': 400
        }
    }
    
    # Position comparison chart
    position_data = data.groupby('Position').agg({
        'HPRD': 'mean',
        'Total_Hours': 'mean',
        'Contract_Percentage': 'mean'
    }).reset_index()
    
    charts['position_comparison'] = {
        'data': [{
            'x': position_data['Position'].tolist(),
            'y': position_data['HPRD'].tolist(),
            'type': 'bar',
            'name': 'Average HPRD',
            'marker': {'color': '#2ca02c'}
        }],
        'layout': {
            'title': 'Average HPRD by Position',
            'xaxis': {'title': 'Position'},
            'yaxis': {'title': 'HPRD'},
            'height': 400
        }
    }
    
    return charts

if __name__ == '__main__':
    # Check if data is already loaded
    db_conn = get_connection()
    try:
        result = db_conn.execute("SELECT COUNT(*) as count FROM facility_staffing").fetchone()
        if result[0] == 0:
            print("No data found, loading facility data...")
            load_facility_data()
        else:
            print(f"Found {result[0]} existing records")
    except:
        print("Table doesn't exist, loading facility data...")
        load_facility_data()
    
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5001)
