from flask import Flask, jsonify, send_from_directory
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

def load_and_process_data():
    """Load and process PBJ data from CSV files."""
    try:
        # Load national data
        national_df = pd.read_csv('pbj_lite/national_lite_metrics.csv')
        
        # Load facility data
        facility_df = pd.read_csv('pbj_lite/facility_lite_metrics.csv')
        
        # Get latest quarter
        latest_quarter = national_df['CY_Qtr'].max()
        latest_national = national_df[national_df['CY_Qtr'] == latest_quarter].iloc[0]
        latest_facility = facility_df[facility_df['CY_Qtr'] == latest_quarter]
        
        # Calculate statistics
        hprd_values = latest_facility['Total_Nurse_HPRD'].dropna()
        contract_values = latest_facility['Contract_Percentage'].dropna()
        
        # Process quarterly trends
        quarterly_trends = {
            'quarters': national_df['CY_Qtr'].tolist(),
            'hprd': national_df['Total_Nurse_HPRD'].tolist(),
            'nurse_care_hprd': national_df['Nurse_Care_HPRD'].tolist(),
            'total_rn_hprd': national_df['Total_RN_HPRD'].tolist(),
            'direct_care_rn_hprd': national_df['Direct_Care_RN_HPRD'].tolist(),
            'census': national_df['MDS'].tolist(),
            'facility_count': national_df['Facility_Count'].tolist(),
            'contract_mean': national_df['Contract_Percentage'].tolist()
        }
        
        # Calculate contract medians for each quarter
        contract_medians = []
        for quarter in national_df['CY_Qtr']:
            quarter_facility = facility_df[facility_df['CY_Qtr'] == quarter]
            if len(quarter_facility) > 0:
                median = quarter_facility['Contract_Percentage'].median()
                contract_medians.append(float(median))
            else:
                contract_medians.append(0.0)
        
        quarterly_trends['contract_median'] = contract_medians
        
        # Calculate distributions
        hprd_bins = np.arange(1.5, 7.5, 0.5)
        hprd_counts, _ = np.histogram(hprd_values, bins=hprd_bins)
        
        # Custom contract bins with catch-all for high percentages
        contract_bins = [0, 1, 5, 10, 15, 20, 25, 30, 100]
        contract_counts, _ = np.histogram(contract_values, bins=contract_bins)
        
        return {
            'latest_quarter': latest_quarter,
            'hprd': {
                'mean': float(latest_national['Total_Nurse_HPRD']),
                'median': float(hprd_values.median()),
                'std': float(hprd_values.std()),
                'min': float(hprd_values.min()),
                'max': float(hprd_values.max())
            },
            'contract': {
                'mean': float(latest_national['Contract_Percentage']),
                'median': float(contract_values.median()),
                'std': float(contract_values.std()),
                'min': float(contract_values.min()),
                'max': float(contract_values.max())
            },
            'quarterly_trends': quarterly_trends,
            'distributions': {
                'hprd': {
                    'bins': [f"{hprd_bins[i]:.1f}-{hprd_bins[i+1]:.1f}" for i in range(len(hprd_bins)-1)],
                    'counts': hprd_counts.tolist()
                },
                'contract': {
                    'bins': ['0%', '1-5%', '5-10%', '10-15%', '15-20%', '20-25%', '25-30%', '30%+'],
                    'counts': contract_counts.tolist()
                }
            }
        }
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

@app.route('/api/pbj-data')
def get_pbj_data():
    """API endpoint to get processed PBJ data."""
    data = load_and_process_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'Failed to load data'}), 500

@app.route('/pbj_playground.html')
def playground():
    """Serve the dynamic playground page."""
    return send_from_directory('.', 'pbj_playground_dynamic.html')

@app.route('/pbj_lite/<path:filename>')
def serve_csv(filename):
    """Serve CSV files from pbj_lite directory."""
    return send_from_directory('pbj_lite', filename)

@app.route('/favicon.ico')
def favicon():
    """Serve favicon."""
    return '📊', 200, {'Content-Type': 'text/plain; charset=utf-8'}

if __name__ == '__main__':
    print("Starting PBJ Playground server...")
    print("Open: http://localhost:5000/pbj_playground.html")
    app.run(debug=True, port=5000)
