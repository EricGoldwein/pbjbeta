#!/usr/bin/env python3
"""
Vercel-compatible dashboard for Facility 495241
"""

import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from datetime import datetime, timedelta
import json
import os

app = Flask(__name__)

# Global variables
df = None
provider_info_df = None

def load_data():
    """Load data for facility 495241"""
    global df, provider_info_df
    
    # Load daily data
    df = pd.read_csv('facility_495241_complete_data.csv')
    
    # Load provider info data
    provider_info_df = pd.read_csv('facility_495241_provider_info_data.csv')
    
    print(f"Loaded {len(df)} daily records and {len(provider_info_df)} provider info records")

# Load data on startup
load_data()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('facility_495241_dashboard.html', 
                         facility_name="KINDRED NURSING AND REHABILITATION-RIVER POINTE",
                         facility_ccn="495241",
                         facility_city="VIRGINIA BEACH",
                         facility_state="VA")

@app.route('/api/data')
def get_data():
    """Get filtered daily data"""
    global df
    
    if df is None:
        return jsonify({'error': 'Data not loaded'})
    
    # Apply filters
    filtered_df = df.copy()
    
    # Convert to dict format expected by frontend
    data = []
    for _, row in filtered_df.iterrows():
        record = {
            'WorkDate': row.get('WorkDate', ''),
            'CY_Qtr': row.get('CY_Qtr', ''),
            'MDScensus': row.get('MDScensus', 0),
            'Total_Nurse_HPRD': row.get('Total_Nurse_HPRD', 0),
            'RN_HPRD': row.get('RN_HPRD', 0),
            'LPN_HPRD': row.get('LPN_HPRD', 0),
            'CNA_HPRD': row.get('CNA_HPRD', 0),
            'Total_Staff_HPRD': row.get('Total_Staff_HPRD', 0),
            'Total_Staff_Hours': row.get('Total_Staff_Hours', 0),
            'Total_RN_HPRD': row.get('Total_RN_HPRD', 0),
            'Total_RN_Hours': row.get('Total_RN_Hours', 0),
            'Total_LPN_HPRD': row.get('Total_LPN_HPRD', 0),
            'Total_LPN_Hours': row.get('Total_LPN_Hours', 0),
            'Total_Nurse_Aide_HPRD': row.get('Total_Nurse_Aide_HPRD', 0),
            'Total_Nurse_Aide_Hours': row.get('Total_Nurse_Aide_Hours', 0),
            'Nurse_Staff_HPRD_Excl_Admin': row.get('Nurse_Staff_HPRD_Excl_Admin', 0),
            'Total_Contract_Pct': row.get('Total_Contract_Pct', 0),
            'IsHoliday': False,
            'DayOfWeek': ''
        }
        data.append(record)
    
    return jsonify(data)

@app.route('/api/provider_info_charts')
def get_provider_info_charts():
    """Get provider info chart data"""
    global provider_info_df
    
    if provider_info_df is None:
        return jsonify({'error': 'Provider info data not loaded'})
    
    # Process provider info data into chart format
    chart_data = provider_info_df.dropna(subset=['quarter']).copy()
    chart_data = chart_data.sort_values('processing_date').groupby('quarter').last().reset_index()
    
    # Format quarter labels
    chart_data['quarter_label'] = chart_data['quarter'].apply(
        lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) and len(str(x)) == 6 else str(x) if pd.notna(x) else None
    )
    
    # Prepare chart data structure
    charts = {
        'total_staffing': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'reported_total': chart_data['reported_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist(),
            'reported_direct': chart_data['reported_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist(),
            'case_mix_total': chart_data['case_mix_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist(),
            'adjusted_total': chart_data['adjusted_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist()
        },
        'rn_staffing': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'reported_rn': chart_data['reported_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_rn_hrs_per_resident_per_day']), None).tolist(),
            'reported_rn_total': chart_data['reported_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_rn_hrs_per_resident_per_day']), None).tolist(),
            'reported_rn_direct': chart_data['reported_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_rn_hrs_per_resident_per_day']), None).tolist(),
            'case_mix_rn': chart_data['case_mix_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_rn_hrs_per_resident_per_day']), None).tolist(),
            'adjusted_rn': chart_data['adjusted_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['adjusted_rn_hrs_per_resident_per_day']), None).tolist()
        },
        'cna_staffing': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'reported_cna': chart_data['reported_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_na_hrs_per_resident_per_day']), None).tolist(),
            'case_mix_cna': chart_data['case_mix_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_na_hrs_per_resident_per_day']), None).tolist(),
            'case_mix_lpn': chart_data['case_mix_lpn_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_lpn_hrs_per_resident_per_day']), None).tolist(),
            'adjusted_cna': chart_data['adjusted_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['adjusted_na_hrs_per_resident_per_day']), None).tolist()
        },
        'census': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'census': chart_data['avg_residents_per_day'].fillna(0).tolist()
        },
        'ratings': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'overall': chart_data['overall_rating'].fillna(0).tolist(),
            'staffing': chart_data['staffing_rating'].fillna(0).tolist(),
            'health_inspection': chart_data['health_inspection_rating'].fillna(0).tolist()
        }
    }
    
    # Convert NaN to None for JSON serialization
    def convert_nan_to_none(obj):
        if isinstance(obj, dict):
            return {key: convert_nan_to_none(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_none(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    return jsonify(convert_nan_to_none(charts))

if __name__ == '__main__':
    app.run(debug=True, port=5001)

