#!/usr/bin/env python3
"""
PBJ320 Dashboard & Report Generator Web App
Provides a web interface for generating facility dashboards and reports
"""

try:
    from flask import Flask, render_template, request, jsonify, send_file, Response, stream_with_context
except ImportError:
    print("ERROR: Flask is not installed. Please run: pip install flask")
    sys.exit(1)

import os
import sys
import subprocess
from datetime import datetime
import threading
import time
import pandas as pd
try:
    from dateutil.relativedelta import relativedelta
except ImportError:
    # Fallback if dateutil not available
    relativedelta = None

app = Flask(__name__, template_folder='templates', static_folder='static')

# CORS is optional - only enable if available
try:
    from flask_cors import CORS
    CORS(app)
    print("✓ Flask-CORS enabled")
except ImportError:
    # CORS not installed - not needed for local development anyway
    print("Note: flask-cors not installed. CORS disabled (not needed for local use).")

# Import functions from existing modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from dynamic_facility_dashboard import create_facility_complete_csv, create_facility_provider_info_csv
except ImportError as e:
    print(f"Warning: Could not import dashboard functions: {e}")
    create_facility_complete_csv = None
    create_facility_provider_info_csv = None

try:
    from create_vercel_deployment import create_facility_vercel_package
except ImportError as e:
    print(f"Warning: Could not import Vercel deployment: {e}")
    create_facility_vercel_package = None

try:
    from facility_report_lib import load_facility_data, format_facility_name, load_provider_info_data
except ImportError as e:
    print(f"Warning: Could not import report library: {e}")
    load_facility_data = None
    format_facility_name = None
    load_provider_info_data = None

# Import canonical identifier functions
try:
    from pbj_identifiers.validators import normalize_ccn, normalize_state_code, normalize_entity_id, validate_state_code
    from pbj_identifiers.urls import generate_dashboard_url, generate_cms_url
    CANONICAL_IDENTIFIERS_AVAILABLE = True
except ImportError:
    # Fallback if pbj_identifiers not available
    CANONICAL_IDENTIFIERS_AVAILABLE = False
    def normalize_ccn(ccn):
        ccn = str(ccn).strip().upper()
        ccn = ''.join(c for c in ccn if c.isalnum())
        return ccn.zfill(6)
    
    def normalize_state_code(state):
        return str(state).strip().upper()[:2]
    
    def validate_state_code(state):
        US_STATE_CODES = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'}
        return normalize_state_code(state) in US_STATE_CODES
    
    def normalize_entity_id(entity_id):
        if not entity_id:
            return None
        entity_id = str(entity_id).strip()
        if '.' in entity_id:
            entity_id = entity_id.split('.')[0]
        return ''.join(c for c in entity_id if c.isdigit()) or None
    
    def generate_dashboard_url(facility=None, state=None, entity=None):
        base = "https://pbjdashboard.com/"
        params = []
        if facility:
            params.append(f"facility={normalize_ccn(facility)}")
        if state:
            params.append(f"state={normalize_state_code(state)}")
        if entity:
            eid = normalize_entity_id(entity)
            if eid:
                params.append(f"entity={eid}")
        result = base + ("?" + "&".join(params) if params else "")
        return str(result)  # Convert to str to avoid LiteralString type issue
    
    def generate_cms_url(ccn, state):
        ccn_norm = normalize_ccn(ccn)
        state_norm = normalize_state_code(state)
        return f"https://www.medicare.gov/care-compare/details/nursing-home/{ccn_norm}/view-all/?state={state_norm}"

def load_entity_longitudinal_metrics(entity_id: str):
    """
    Load longitudinal metrics for a specific entity.
    
    This function loads the normalized longitudinal chain performance data
    for a given entity_id. If entity_id is numeric (like "52"), it will
    look up the hash-based entity_id from entity_lookup.csv.
    
    Args:
        entity_id: Entity ID (hash-based identifier from entity_lookup.csv, or numeric chain_id)
        
    Returns:
        pd.DataFrame with columns: entity_id, entity_name, chain_id, report_period,
        period_type, metric_name, metric_value, source_file, ingested_at
        Returns None if file missing or entity not found.
    """
    try:
        # If entity_id is numeric, look up the hash-based entity_id
        if entity_id.isdigit():
            entity_lookup_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ownership', 'entity_lookup.csv')
            if os.path.exists(entity_lookup_file):
                entity_lookup = pd.read_csv(entity_lookup_file)
                chain_id = float(entity_id)
                entity_row = entity_lookup[entity_lookup['chain_id'] == chain_id]
                if not entity_row.empty:
                    entity_id = entity_row.iloc[0]['entity_id']
        
        ownership_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ownership')
        parquet_path = os.path.join(ownership_dir, 'chain_performance_longitudinal.parquet')
        csv_path = os.path.join(ownership_dir, 'chain_performance_longitudinal.csv')
        
        # Try parquet first
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                entity_data = df[df['entity_id'] == entity_id].copy()
                if not entity_data.empty:
                    return entity_data
            except ImportError:
                # PyArrow not available, try CSV
                pass
            except Exception as e:
                # Log error but don't raise
                print(f"Warning: Error loading parquet for entity {entity_id}: {e}")
        
        # Fallback to CSV
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path, low_memory=False)
                entity_data = df[df['entity_id'] == entity_id].copy()
                if not entity_data.empty:
                    return entity_data
            except Exception as e:
                print(f"Warning: Error loading CSV for entity {entity_id}: {e}")
        
        # No data found - return None (non-breaking)
        return None
        
    except Exception as e:
        # Graceful degradation - don't raise exceptions
        print(f"Warning: Error in load_entity_longitudinal_metrics for {entity_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_entity_key_metrics_over_time(entity_id: str):
    """
    Extract key metrics over time for an entity from longitudinal data.
    
    Returns key metrics like number of facilities, ratings, SFFs, staffing levels
    organized by report period.
    
    Args:
        entity_id: Entity ID (hash-based identifier)
        
    Returns:
        Dict with structure:
        {
            'entity_name': str,
            'metrics_by_period': [
                {
                    'period': 'YYYY-MM',
                    'number_of_facilities': int,
                    'avg_overall_rating': float,
                    'avg_staffing_rating': float,
                    'total_sff': int,
                    'avg_total_nurse_hprd': float,
                    'avg_rn_hprd': float,
                    ...
                },
                ...
            ]
        }
        Returns None if data not available.
    """
    try:
        df = load_entity_longitudinal_metrics(entity_id)
        if df is None or df.empty:
            return None
        
        # Get entity name
        entity_name = df['entity_name'].iloc[0] if 'entity_name' in df.columns else None
        
        # Key metrics to extract
        key_metrics = {
            'Number of facilities': 'number_of_facilities',
            'Average overall 5-star rating': 'avg_overall_rating',
            'Average health inspection rating': 'avg_health_inspection_rating',
            'Average staffing rating': 'avg_staffing_rating',
            'Average quality rating': 'avg_quality_rating',
            'Number of Special Focus Facilities (SFF)': 'total_sff',
            'Average total nurse hours per resident day': 'avg_total_nurse_hprd',
            'Average total Registered Nurse hours per resident day': 'avg_rn_hprd',
            'Average total weekend nurse hours per resident day': 'avg_weekend_nurse_hprd',
            'Average total nursing staff turnover percentage': 'avg_nursing_turnover',
            'Average Registered Nurse turnover percentage': 'avg_rn_turnover',
            'Total number of fines': 'total_fines',
            'Total amount of fines in dollars': 'total_fine_amount',
            'Number of facilities with an abuse icon': 'facilities_with_abuse_icon',
            'Percent of facilities classified as for-profit': 'pct_for_profit',
        }
        
        # Group by period
        metrics_by_period = []
        for period in sorted(df['report_period'].unique()):
            period_data = df[df['report_period'] == period].copy()
            
            period_metrics = {'period': period}
            
            # Extract each key metric
            for metric_name, metric_key in key_metrics.items():
                metric_rows = period_data[period_data['metric_name'] == metric_name]
                if not metric_rows.empty:
                    value = metric_rows.iloc[0]['metric_value']
                    # Convert to appropriate type
                    try:
                        if pd.notna(value):
                            if isinstance(value, str):
                                # Try to convert string to number
                                value = float(value.replace(',', '').replace('$', ''))
                            else:
                                value = float(value)
                        else:
                            value = None
                    except (ValueError, TypeError):
                        value = None
                    period_metrics[metric_key] = value
                else:
                    period_metrics[metric_key] = None
            
            metrics_by_period.append(period_metrics)
        
        return {
            'entity_id': entity_id,
            'entity_name': entity_name,
            'metrics_by_period': metrics_by_period
        }
        
    except Exception as e:
        print(f"Warning: Error extracting key metrics for entity {entity_id}: {e}")
        return None

def get_entity_chow_status(entity_id: str):
    """
    Get CHOW status for an entity.
    
    Args:
        entity_id: Entity ID (hash-based identifier)
        
    Returns:
        Dict with structure:
        {
            'has_chow': bool,
            'chow_count': int,
            'total_facilities': int,
            'chow_facilities': []  # Will be populated by caller
        }
        Returns None if data not available.
    """
    try:
        chow_facilities = get_entity_chow_facilities(entity_id)
        if chow_facilities is None:
            return None
        
        # Get total facilities count for this entity
        entity_lookup_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ownership', 'entity_lookup.csv')
        if not os.path.exists(entity_lookup_file):
            return None
        
        entity_lookup = pd.read_csv(entity_lookup_file)
        entity_row = entity_lookup[entity_lookup['entity_id'] == entity_id]
        if entity_row.empty:
            return None
        
        chain_id = entity_row.iloc[0]['chain_id']
        if pd.isna(chain_id):
            return None
        
        # Count total facilities in entity
        provider_info_file = 'provider_info_combined.csv'
        if not os.path.exists(provider_info_file):
            provider_info_file = os.path.join('provider_info', 'provider_info_combined.csv')
            if not os.path.exists(provider_info_file):
                return None
        
        provider_df = pd.read_csv(provider_info_file, low_memory=False, dtype={'ccn': str})
        
        chain_id_col = None
        for col in ['chain_id', 'Chain ID', 'Chain_ID', 'Entity ID', 'entity_id', 'affiliated_entity_id']:
            if col in provider_df.columns:
                chain_id_col = col
                break
        
        if not chain_id_col:
            return None
        
        entity_facilities = provider_df[
            pd.to_numeric(provider_df[chain_id_col], errors='coerce') == float(chain_id)
        ]
        total_facilities = len(entity_facilities['ccn'].unique()) if 'ccn' in entity_facilities.columns else 0
        
        return {
            'has_chow': len(chow_facilities) > 0,
            'chow_count': len(chow_facilities),
            'total_facilities': total_facilities,
            'chow_facilities': []  # Will be populated by caller
        }
    except Exception as e:
        print(f"Warning: Error getting CHOW status for entity {entity_id}: {e}")
        return None

def get_entity_chow_facilities(entity_id: str):
    """
    Get facilities in an entity that changed ownership in the last 12 months.
    Uses provider_changed_ownership_in_last_12_months column from provider_info_combined.csv.
    Includes provider info files and PBJ file associations.
    
    Args:
        entity_id: Entity ID (hash-based identifier)
        
    Returns:
        List of dicts with facility info:
        [
            {
                'ccn': str,
                'facility_name': str,
                'state': str,
                'chow_date': str (YYYY-MM-DD),
                'chow_quarter': str,
                'provider_info_files': [str, ...],  # List of provider info file names
                'pbj_files': [str, ...]  # List of associated PBJ file names
            },
            ...
        ]
        Returns empty list if no CHOW or data not available.
    """
    try:
        # Load entity lookup to get chain_id
        entity_lookup_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ownership', 'entity_lookup.csv')
        if not os.path.exists(entity_lookup_file):
            return []
        
        entity_lookup = pd.read_csv(entity_lookup_file)
        entity_row = entity_lookup[entity_lookup['entity_id'] == entity_id]
        if entity_row.empty:
            return []
        
        chain_id = entity_row.iloc[0]['chain_id']
        if pd.isna(chain_id):
            return []
        
        # Load provider info combined file
        provider_info_file = 'provider_info_combined.csv'
        if not os.path.exists(provider_info_file):
            # Try alternative locations
            provider_info_file = os.path.join('provider_info', 'provider_info_combined.csv')
            if not os.path.exists(provider_info_file):
                return []
        
        provider_df = pd.read_csv(provider_info_file, low_memory=False, dtype={'ccn': str})
        
        # Find chain_id column
        chain_id_col = None
        for col in ['chain_id', 'Chain ID', 'Chain_ID', 'Entity ID', 'entity_id', 'affiliated_entity_id']:
            if col in provider_df.columns:
                chain_id_col = col
                break
        
        if not chain_id_col:
            return []
        
        # Filter facilities by chain_id
        entity_facilities = provider_df[
            pd.to_numeric(provider_df[chain_id_col], errors='coerce') == float(chain_id)
        ]
        
        if entity_facilities.empty:
            return []
        
        # Get CCN column
        ccn_col = 'ccn'
        if ccn_col not in entity_facilities.columns:
            for col in ['CCN', 'CMS Certification Number (CCN)', 'PROVNUM']:
                if col in entity_facilities.columns:
                    ccn_col = col
                    break
        
        # Check for provider_changed_ownership_in_last_12_months column
        chow_col = 'provider_changed_ownership_in_last_12_months'
        if chow_col not in entity_facilities.columns:
            # Try alternative column names
            for col in ['Provider Changed Ownership in Last 12 Months', 'CHOW', 'chow']:
                if col in entity_facilities.columns:
                    chow_col = col
                    break
        
        if chow_col not in entity_facilities.columns:
            return []
        
        # Filter for facilities with CHOW = 'Y' or 'Yes'
        chow_facilities_raw = entity_facilities[
            entity_facilities[chow_col].astype(str).str.strip().str.upper().isin(['Y', 'YES', 'TRUE', '1'])
        ]
        
        if chow_facilities_raw.empty:
            return []
        
        chow_facilities = []
        
        # Group by CCN to get most recent CHOW record
        for ccn, facility_rows in chow_facilities_raw.groupby(ccn_col):
            # Sort by processing_date to get most recent
            if 'processing_date' in facility_rows.columns:
                facility_rows = facility_rows.sort_values('processing_date', ascending=False)
            
            latest = facility_rows.iloc[0]
            
            # Get facility info
            facility_name = str(latest.get('provider_name', latest.get('Provider Name', 'Unknown')))
            state = str(latest.get('state', latest.get('State', '')))
            
            # Get processing date and quarter
            proc_date = latest.get('processing_date')
            chow_date = None
            if pd.notna(proc_date):
                try:
                    if isinstance(proc_date, str):
                        proc_date = pd.to_datetime(proc_date, errors='coerce')
                    if pd.notna(proc_date):
                        chow_date = proc_date.strftime('%Y-%m-%d')
                except Exception:
                    pass
            
            quarter = str(latest.get('quarter', '')) if pd.notna(latest.get('quarter')) else None
            
            # Find provider info files that contain this CHOW flag
            provider_info_files = []
            pbj_files = []
            
            # Search for provider info files in provider_info directory
            provider_info_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'provider_info')
            if os.path.exists(provider_info_dir):
                import glob
                ccn_normalized = str(ccn).zfill(6)
                
                # Find provider info files
                for prov_file in glob.glob(os.path.join(provider_info_dir, 'NH_ProviderInfo_*.csv')):
                    try:
                        prov_df = pd.read_csv(prov_file, low_memory=False, dtype={'ccn': str})
                        prov_ccn_col = 'ccn'
                        if prov_ccn_col not in prov_df.columns:
                            for col in ['CCN', 'CMS Certification Number (CCN)', 'PROVNUM']:
                                if col in prov_df.columns:
                                    prov_ccn_col = col
                                    break
                        
                        if prov_ccn_col in prov_df.columns:
                            prov_df[prov_ccn_col] = prov_df[prov_ccn_col].astype(str).str.zfill(6)
                            if ccn_normalized in prov_df[prov_ccn_col].values:
                                # Check if this file has CHOW flag for this facility
                                facility_row = prov_df[prov_df[prov_ccn_col] == ccn_normalized]
                                if not facility_row.empty:
                                    prov_chow_col = 'provider_changed_ownership_in_last_12_months'
                                    if prov_chow_col not in facility_row.columns:
                                        for col in ['Provider Changed Ownership in Last 12 Months', 'CHOW']:
                                            if col in facility_row.columns:
                                                prov_chow_col = col
                                                break
                                    
                                    if prov_chow_col in facility_row.columns:
                                        chow_value = str(facility_row.iloc[0][prov_chow_col]).strip().upper()
                                        if chow_value in ['Y', 'YES', 'TRUE', '1']:
                                            provider_info_files.append(os.path.basename(prov_file))
                    except Exception:
                        pass
                
                # Find PBJ files associated with this facility/quarter
                pbj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PBJcsv')
                if os.path.exists(pbj_dir) and quarter:
                    # Try to find PBJ files for this quarter
                    quarter_normalized = quarter.replace(' ', '').upper()
                    for pbj_file in glob.glob(os.path.join(pbj_dir, f'*{quarter_normalized}*.csv')):
                        try:
                            pbj_df = pd.read_csv(pbj_file, low_memory=False, dtype={'PROVNUM': str})
                            if 'PROVNUM' in pbj_df.columns:
                                pbj_df['PROVNUM'] = pbj_df['PROVNUM'].astype(str).str.zfill(6)
                                if ccn_normalized in pbj_df['PROVNUM'].values:
                                    pbj_files.append(os.path.basename(pbj_file))
                        except Exception:
                            pass
            
            chow_facilities.append({
                'ccn': str(ccn).zfill(6),
                'facility_name': facility_name,
                'state': state,
                'chow_date': chow_date,
                'chow_quarter': quarter,
                'provider_info_files': provider_info_files,
                'pbj_files': pbj_files
            })
        
        return chow_facilities
        
    except Exception as e:
        print(f"Warning: Error getting CHOW facilities for entity {entity_id}: {e}")
        import traceback
        traceback.print_exc()
        return []

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('dashboard_generator.html')

@app.route('/veterans-homes')
def veterans_homes():
    """Serve the Veterans Homes batch deployment page"""
    return render_template('veterans_homes.html')

@app.route('/api/veterans-homes-stats', methods=['GET'])
def veterans_homes_stats():
    """
    Calculate basic statistics for Veterans nursing homes.
    Returns: facility count, total residents, average staffing metrics
    """
    try:
        # Load CSV file
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'veterans_homes_matches.csv')
        if not os.path.exists(csv_path):
            return jsonify({'error': 'veterans_homes_matches.csv not found'}), 404
        
        # Read CSV and extract CCNs
        df = pd.read_csv(csv_path)
        
        # Get CCN column
        ccn_col = None
        for col in ['CCN', 'ccn', 'Ccn']:
            if col in df.columns:
                ccn_col = col
                break
        
        if not ccn_col:
            return jsonify({'error': 'CCN column not found in CSV'}), 400
        
        # Extract unique CCNs
        ccn_list = []
        for _, row in df.iterrows():
            ccn = str(row[ccn_col]).strip()
            if ccn and ccn != 'nan' and ccn.isdigit():
                ccn_list.append(ccn.zfill(6))
        
        # De-duplicate
        unique_ccns = list(set(ccn_list))
        facility_count = len(unique_ccns)
        
        if facility_count == 0:
            return jsonify({
                'facility_count': 0,
                'total_residents': 0,
                'avg_total_hprd': 0,
                'avg_rn_hprd': 0,
                'avg_cna_hprd': 0
            })
        
        # Load provider info to get census data
        # Directly read provider_info_combined.csv (don't use load_provider_info_data which requires provnum)
        total_residents = 0
        facilities_with_data = 0
        
        try:
            provider_file = 'provider_info_combined.csv'
            if os.path.exists(provider_file):
                provider_df = pd.read_csv(provider_file, low_memory=False, dtype={'ccn': str})
                
                # Find census column
                census_col = None
                for col in ['avg_residents_per_day', 'census', 'Census', 'CENSUS', 'Average Census', 'average_census', 'Average Number of Residents per Day']:
                    if col in provider_df.columns:
                        census_col = col
                        break
                
                # Find CCN column in provider info
                provider_ccn_col = None
                for col in ['CCN', 'ccn', 'CMS Certification Number (CCN)', 'PROVNUM']:
                    if col in provider_df.columns:
                        provider_ccn_col = col
                        provider_df[provider_ccn_col] = provider_df[provider_ccn_col].astype(str).str.zfill(6)
                        break
                
                if census_col and provider_ccn_col:
                    # Filter for Veterans homes
                    veterans_provider_df = provider_df[provider_df[provider_ccn_col].isin(unique_ccns)]
                    
                    if not veterans_provider_df.empty:
                        # Get latest record for each facility (if multiple records exist)
                        if 'processing_date' in veterans_provider_df.columns:
                            veterans_provider_df['processing_date'] = pd.to_datetime(veterans_provider_df['processing_date'], errors='coerce')
                            veterans_provider_df = veterans_provider_df.sort_values('processing_date', ascending=False)
                            veterans_provider_df = veterans_provider_df.drop_duplicates(subset=[provider_ccn_col], keep='first')
                        
                        # Sum census values
                        census_data = veterans_provider_df[census_col].dropna()
                        census_data = census_data[census_data > 0]  # Filter out zeros
                        if len(census_data) > 0:
                            total_residents = census_data.sum()
                            facilities_with_data = len(census_data)
        except Exception as e:
            print(f"Error loading provider info: {e}")
        
        # Load quarterly metrics for staffing data (if available)
        avg_total_hprd = 0
        avg_rn_hprd = 0
        avg_cna_hprd = 0
        staffing_facilities = 0
        
        try:
            quarterly_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'facility_quarterly_metrics.csv')
            if os.path.exists(quarterly_file):
                quarterly_df = pd.read_csv(quarterly_file, low_memory=False)
                
                # Normalize PROVNUM column
                if 'PROVNUM' in quarterly_df.columns:
                    quarterly_df['PROVNUM'] = quarterly_df['PROVNUM'].astype(str).str.zfill(6)
                    
                    # Filter for Veterans homes and get latest quarter data
                    veterans_df = quarterly_df[quarterly_df['PROVNUM'].isin(unique_ccns)]
                    
                    if not veterans_df.empty:
                        # Get most recent quarter for each facility
                        if 'CY_Qtr' in veterans_df.columns:
                            # Sort by quarter and get latest
                            veterans_df = veterans_df.sort_values('CY_Qtr', ascending=False)
                            latest_quarters = veterans_df.drop_duplicates(subset=['PROVNUM'], keep='first')
                            
                            # Calculate averages
                            hprd_cols = {
                                'total': ['Total_Nurse_HPRD', 'Total_HPRD', 'Total Staffing HPRD'],
                                'rn': ['RN_HPRD', 'RN HPRD', 'Registered Nurse HPRD'],
                                'cna': ['CNA_HPRD', 'CNA HPRD', 'Nurse Aide HPRD']
                            }
                            
                            for metric_type, col_names in hprd_cols.items():
                                values = []
                                for col in col_names:
                                    if col in latest_quarters.columns:
                                        col_data = latest_quarters[col].dropna()
                                        col_data = col_data[col_data > 0]  # Filter out zeros
                                        values.extend(col_data.tolist())
                                        break
                                
                                if values:
                                    avg_val = sum(values) / len(values)
                                    if metric_type == 'total':
                                        avg_total_hprd = round(avg_val, 2)
                                    elif metric_type == 'rn':
                                        avg_rn_hprd = round(avg_val, 2)
                                    elif metric_type == 'cna':
                                        avg_cna_hprd = round(avg_val, 2)
                                    staffing_facilities = len(values)
        except Exception as e:
            print(f"Error loading quarterly metrics: {e}")
        
        return jsonify({
            'facility_count': facility_count,
            'total_residents': int(total_residents) if total_residents > 0 else None,
            'avg_total_hprd': avg_total_hprd if avg_total_hprd > 0 else None,
            'avg_rn_hprd': avg_rn_hprd if avg_rn_hprd > 0 else None,
            'avg_cna_hprd': avg_cna_hprd if avg_cna_hprd > 0 else None,
            'facilities_with_census': facilities_with_data,
            'facilities_with_staffing': staffing_facilities
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def _yield_progress_json(data):
    """Helper to format JSON progress updates for streaming"""
    import json
    return json.dumps(data) + '\n'

@app.route('/api/batch-deploy-veterans', methods=['POST'])
def batch_deploy_veterans():
    """
    Batch deploy Veterans nursing homes to Vercel.
    This is a wrapper around the existing single-CCN deploy mechanism.
    Reads CCNs from veterans_homes_matches.csv and processes them sequentially.
    """
    import json
    import subprocess
    import platform
    import shutil
    import re
    
    def deploy_single_facility(provnum, facility_name=None):
        """
        Deploy a single facility using the existing deploy mechanism.
        This replicates the logic from /api/deploy-vercel without HTTP response.
        Returns: (success: bool, vercel_url: str or None, error: str or None)
        """
        try:
            # Normalize CCN
            provnum = normalize_ccn(provnum)
            if not provnum.isdigit() or len(provnum) != 6:
                return False, None, f'Invalid CCN: {provnum}'
            
            if create_facility_vercel_package is None:
                return False, None, 'Vercel deployment functions not available'
            
            # Step 1: Create the Vercel deployment package (pass project root so paths work)
            _root = os.path.dirname(os.path.abspath(__file__))
            result = create_facility_vercel_package(provnum, project_root=_root)
            if not result:
                return False, None, 'Failed to create Vercel deployment package'
            
            deploy_dir = f"pbj320-{provnum}"
            deploy_path = os.path.join(os.getcwd(), deploy_dir)
            
            if not os.path.exists(deploy_path):
                return False, None, f'Deployment directory not found: {deploy_dir}'
            
            # Step 2: Deploy to Vercel (replicate deploy logic)
            vercel_cmd = None
            vercel_path = shutil.which('vercel')
            if vercel_path:
                vercel_cmd = vercel_path
            elif platform.system() == 'Windows':
                vercel_cmd_path = shutil.which('vercel.cmd')
                if vercel_cmd_path:
                    vercel_cmd = vercel_cmd_path
                else:
                    possible_paths = [
                        os.path.expanduser(r'~\AppData\Roaming\npm\vercel.cmd'),
                        os.path.expanduser(r'~\AppData\Local\Programs\nodejs\vercel.cmd'),
                        r'C:\Program Files\nodejs\vercel.cmd',
                        r'C:\Program Files (x86)\nodejs\vercel.cmd',
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            vercel_cmd = path
                            break
            
            if not vercel_cmd:
                # Try generic command
                try:
                    check_result = subprocess.run(
                        ['vercel', '--version'],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        shell=True
                    )
                    if check_result.returncode == 0:
                        vercel_cmd = 'vercel'
                except:
                    return False, None, 'Vercel CLI not found'
            
            # Deploy to Vercel
            if not vercel_cmd:
                return False, None, 'Vercel CLI not found'
            
            vercel_url = None
            os.chdir(deploy_path)
            try:
                if platform.system() == 'Windows':
                    deploy_command = f'{vercel_cmd} --prod --yes'
                    deploy_result = subprocess.run(
                        deploy_command,
                        capture_output=True,
                        text=True,
                        timeout=600,
                        shell=True
                    )
                else:
                    # Type check: vercel_cmd is guaranteed to be str here
                    deploy_result = subprocess.run(
                        [vercel_cmd, '--prod', '--yes'],
                        capture_output=True,
                        text=True,
                        timeout=600,
                        shell=False
                    )
                
                if deploy_result.returncode == 0:
                    output = deploy_result.stdout + deploy_result.stderr
                    url_match = re.search(r'https://pbj320-\d+\.vercel\.app', output)
                    if url_match:
                        vercel_url = url_match.group(0) + '/'
                    else:
                        vercel_url = f"https://pbj320-{provnum}.vercel.app/"
                    return True, vercel_url, None
                else:
                    error_output = deploy_result.stderr or deploy_result.stdout
                    return False, None, f'Vercel deploy failed: {error_output[:200]}'
            finally:
                os.chdir(os.path.dirname(os.path.abspath(__file__)))
                
        except Exception as e:
            return False, None, str(e)
    
    try:
        # Load CSV file
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'veterans_homes_matches.csv')
        if not os.path.exists(csv_path):
            return Response(
                _yield_progress_json({'type': 'error', 'message': 'veterans_homes_matches.csv not found'}),
                mimetype='application/json',
                status=404
            )
        
        # Read CSV and extract CCNs
        df = pd.read_csv(csv_path)
        
        # Get CCN column (handle case variations and "sccn" variant)
        ccn_col = None
        for col in ['CCN', 'ccn', 'Ccn', 'sccn', 'SCCN']:
            if col in df.columns:
                ccn_col = col
                break
        
        if not ccn_col:
            return Response(
                _yield_progress_json({'type': 'error', 'message': 'CCN column not found in CSV'}),
                mimetype='application/json',
                status=400
            )
        
        # Extract CCNs and facility names
        facilities = []
        for _, row in df.iterrows():
            ccn = str(row[ccn_col]).strip()
            if ccn and ccn != 'nan' and ccn.isdigit():
                # Get facility name from CSV if available
                name_col = None
                for col in ['Match Name', 'match_name', 'Match_Name', 'Facility Name']:
                    if col in df.columns:
                        name_col = col
                        break
                facility_name = str(row[name_col]).strip() if name_col and pd.notna(row.get(name_col)) else None
                facilities.append({'ccn': ccn.zfill(6), 'name': facility_name})
        
        # De-duplicate by CCN
        seen_ccns = set()
        unique_facilities = []
        for fac in facilities:
            if fac['ccn'] not in seen_ccns:
                seen_ccns.add(fac['ccn'])
                unique_facilities.append(fac)
        
        total = len(unique_facilities)
        
        if total == 0:
            return Response(
                _yield_progress_json({'type': 'error', 'message': 'No valid CCNs found in CSV'}),
                mimetype='application/json',
                status=400
            )
        
        # Process each facility sequentially
        results = []
        
        def generate():
            yield _yield_progress_json({
                'type': 'start',
                'total': total,
                'message': f'Starting batch deployment of {total} facilities...'
            })
            
            for idx, facility in enumerate(unique_facilities, 1):
                ccn = facility['ccn']
                facility_name = facility['name']
                
                # Send progress update
                progress_pct = ((idx - 1) / total) * 100
                yield _yield_progress_json({
                    'type': 'progress',
                    'current': idx,
                    'total': total,
                    'progress': progress_pct,
                    'ccn': ccn,
                    'facility_name': facility_name,
                    'status': 'Starting deployment...'
                })
                
                # Deploy facility (using existing mechanism)
                success, vercel_url, error = deploy_single_facility(ccn, facility_name)
                
                if success:
                    status = 'Published'
                    results.append({
                        'ccn': ccn,
                        'facility_name': facility_name,
                        'status': status,
                        'dashboard_url': vercel_url
                    })
                    yield _yield_progress_json({
                        'type': 'progress',
                        'current': idx,
                        'total': total,
                        'progress': (idx / total) * 100,
                        'ccn': ccn,
                        'facility_name': facility_name,
                        'status': f'Published - {vercel_url}'
                    })
                else:
                    # Check if it's a skip (already deployed) or failure
                    if 'already' in error.lower() or 'exists' in error.lower():
                        status = 'Skipped'
                    else:
                        status = 'Failed'
                    
                    results.append({
                        'ccn': ccn,
                        'facility_name': facility_name,
                        'status': status,
                        'dashboard_url': None,
                        'error': error
                    })
                    yield _yield_progress_json({
                        'type': 'progress',
                        'current': idx,
                        'total': total,
                        'progress': (idx / total) * 100,
                        'ccn': ccn,
                        'facility_name': facility_name,
                        'status': f'{status} - {error[:100] if error else ""}'
                    })
            
            # Send completion
            yield _yield_progress_json({
                'type': 'complete',
                'results': results,
                'total': total,
                'published': len([r for r in results if r['status'] == 'Published']),
                'skipped': len([r for r in results if r['status'] == 'Skipped']),
                'failed': len([r for r in results if r['status'] == 'Failed'])
            })
        
        return Response(stream_with_context(generate()), mimetype='application/json')
        
    except Exception as e:
        import traceback
        error_msg = f'Batch deployment error: {str(e)}'
        print(error_msg)
        traceback.print_exc()
        return Response(
            _yield_progress_json({'type': 'error', 'message': error_msg}),
            mimetype='application/json',
            status=500
        )

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'template_folder': app.template_folder})

@app.route('/api/generate-dashboard', methods=['POST'])
def generate_dashboard():
    """Generate facility dashboard CSV files"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid request data'}), 400
        
        # Normalize CCN using canonical identifier layer
        provnum_raw = str(data.get('provnum', '')).strip()
        try:
            provnum = normalize_ccn(provnum_raw)
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid facility code: {str(e)}'}), 400
        
        print(f"Starting dashboard generation for facility {provnum}...")
        
        if not provnum.isdigit() or len(provnum) != 6:
            return jsonify({'error': 'Invalid facility code. Must be 6 digits.'}), 400
        
        if create_facility_complete_csv is None:
            return jsonify({'error': 'Dashboard generation functions not available'}), 500
        
        # Import file path utilities
        from file_path_utils import find_facility_complete_data, find_facility_provider_info, get_facility_folder
        import shutil
        
        # Get facility folder
        facility_folder = get_facility_folder(provnum)
        
        # Create facility CSV
        csv_file = find_facility_complete_data(provnum)
        if not csv_file or not os.path.exists(csv_file):
            csv_filename = f'facility_{provnum}_complete_data.csv'
            csv_file = str(facility_folder / csv_filename)
            try:
                print(f"Creating facility CSV for {provnum}...")
                create_facility_complete_csv(provnum, output_path=csv_file)
                # Fallback: if created in root (no output_path support), move to facility folder
                root_csv = f'facility_{provnum}_complete_data.csv'
                if os.path.exists(root_csv) and not os.path.exists(csv_file):
                    shutil.move(root_csv, csv_file)
                print(f"Successfully created facility CSV for {provnum}")
            except (PermissionError, OSError) as pe:
                error_str = str(pe)
                if '32' in error_str or 'being used' in error_str.lower() or 'WinError 32' in error_str:
                    return jsonify({
                        'error': f'File is locked. Please close any programs that have {csv_filename} open (e.g., Excel) and try again.'
                    }), 500
                raise
            except Exception as e:
                error_str = str(e)
                if '32' in error_str or 'being used' in error_str.lower() or 'WinError 32' in error_str:
                    return jsonify({
                        'error': f'File is locked. Please close any programs that have {csv_filename} open (e.g., Excel) and try again.'
                    }), 500
                raise
        
        # Create or refresh provider info CSV (full build so all past quarters from provider_info_combined are included)
        provider_filename = f'facility_{provnum}_provider_info_data.csv'
        provider_csv_file = str(facility_folder / provider_filename)
        if create_facility_provider_info_csv is not None:
            try:
                provider_data = create_facility_provider_info_csv(provnum)  # No existing path = full build, all quarters
                if provider_data is not None and len(provider_data) > 0:
                    provider_data.to_csv(provider_csv_file, index=False)
                    # Check if file was created in root, move it to facility folder
                    root_provider = f'facility_{provnum}_provider_info_data.csv'
                    if os.path.exists(root_provider) and not os.path.exists(provider_csv_file):
                        shutil.move(root_provider, provider_csv_file)
            except (PermissionError, OSError) as pe:
                error_str = str(pe)
                if '32' in error_str or 'being used' in error_str.lower() or 'WinError 32' in error_str:
                    # Provider info is optional, so warn but don't fail
                    print(f"Warning: Could not create provider info CSV - file may be locked: {error_str}")
                else:
                    # Provider info is optional, so don't fail
                    pass
            except Exception as e:
                error_str = str(e)
                if '32' in error_str or 'being used' in error_str.lower() or 'WinError 32' in error_str:
                    # Provider info is optional, so warn but don't fail
                    print(f"Warning: Could not create provider info CSV - file may be locked: {error_str}")
                else:
                    # Provider info is optional, so don't fail
                    pass
        
        # Verify files were created
        files_created = []
        if csv_file and os.path.exists(csv_file):
            files_created.append(os.path.basename(csv_file))
        if provider_csv_file and os.path.exists(provider_csv_file):
            files_created.append(os.path.basename(provider_csv_file))
        
        # Get facility name for the link
        facility_name = None
        try:
            if load_facility_data is not None:
                df = load_facility_data(provnum)
                if df is not None and not df.empty and 'PROVNAME' in df.columns:
                    facility_name = df['PROVNAME'].iloc[0]
                    if format_facility_name is not None:
                        facility_name = format_facility_name(facility_name)
        except Exception:
            pass  # Ignore errors getting facility name
        
        return jsonify({
            'success': True,
            'provnum': provnum,
            'files': files_created,
            'facility_name': facility_name,
            'message': f'Dashboard files created for facility {provnum}'
        })
        
    except Exception as e:
        import traceback
        error_str = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in generate_dashboard: {error_str}")
        print(traceback_str)
        
        # Check for file locking errors
        if '32' in error_str or 'being used' in error_str.lower() or 'WinError 32' in error_str:
            return jsonify({
                'error': 'File is locked. This may be due to Flask auto-reloader or another program. Please close any programs that have the CSV files open (e.g., Excel) and try again.'
            }), 500
        
        return jsonify({'error': f'Error generating dashboard: {error_str}'}), 500

@app.route('/api/deploy-vercel', methods=['POST'])
def deploy_vercel():
    """Create Vercel deployment package and deploy to Vercel"""
    import subprocess
    import os
    
    try:
        data = request.json
        # Normalize CCN using canonical identifier layer
        provnum_raw = str(data.get('provnum', '')).strip()
        try:
            provnum = normalize_ccn(provnum_raw)
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid facility code: {str(e)}'}), 400
        
        if not provnum.isdigit() or len(provnum) != 6:
            return jsonify({'error': 'Invalid facility code. Must be 6 digits.'}), 400
        
        if create_facility_vercel_package is None:
            return jsonify({'error': 'Vercel deployment functions not available'}), 500
        
        # Step 1: Create the Vercel deployment package (pass our dir as project root so paths work regardless of cwd)
        print(f"Creating Vercel package for facility {provnum}...")
        _root = os.path.dirname(os.path.abspath(__file__))
        result = create_facility_vercel_package(provnum, project_root=_root)
        
        if not result:
            return jsonify({'error': 'Failed to create Vercel deployment package'}), 500
        
        deploy_dir = os.path.join("deployments", f"pbj320-{provnum}")
        deploy_path = os.path.join(os.getcwd(), deploy_dir)
        
        if not os.path.exists(deploy_path):
            return jsonify({'error': f'Deployment directory not found: {deploy_dir}'}), 500
        
        # Step 2: Deploy to Vercel
        print(f"Deploying to Vercel from {deploy_path}...")
        vercel_url = None
        
        # Check if Vercel CLI is installed and find its path
        vercel_installed = False
        vercel_version = None
        vercel_cmd = None
        
        # Try to find vercel command path first
        import platform
        import shutil
        
        # Try to find vercel in PATH
        vercel_path = shutil.which('vercel')
        if vercel_path:
            vercel_cmd = vercel_path
        elif platform.system() == 'Windows':
            # On Windows, try vercel.cmd
            vercel_cmd_path = shutil.which('vercel.cmd')
            if vercel_cmd_path:
                vercel_cmd = vercel_cmd_path
            else:
                # Try common npm global install locations
                possible_paths = [
                    os.path.expanduser(r'~\AppData\Roaming\npm\vercel.cmd'),
                    os.path.expanduser(r'~\AppData\Local\Programs\nodejs\vercel.cmd'),
                    r'C:\Program Files\nodejs\vercel.cmd',
                    r'C:\Program Files (x86)\nodejs\vercel.cmd',
                ]
                for path in possible_paths:
                    if os.path.exists(path):
                        vercel_cmd = path
                        break
        
        # If we found a path, verify it works
        if vercel_cmd:
            try:
                check_result = subprocess.run(
                    [vercel_cmd, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=True  # Use shell on Windows to find commands in PATH
                )
                if check_result.returncode == 0:
                    vercel_installed = True
                    vercel_version = check_result.stdout.strip()
                    print(f"Vercel CLI found: {vercel_version} at {vercel_cmd}")
            except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                vercel_cmd = None  # Reset if verification failed
        
        # If still not found, try generic 'vercel' command
        if not vercel_installed:
            try:
                check_result = subprocess.run(
                    ['vercel', '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=True  # Use shell on Windows to find commands in PATH
                )
                if check_result.returncode == 0:
                    vercel_installed = True
                    vercel_version = check_result.stdout.strip()
                    vercel_cmd = 'vercel'  # Use generic command
                    print(f"Vercel CLI found: {vercel_version}")
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        
        # If not installed, find npm and try to install Vercel CLI
        if not vercel_installed:
            print("Vercel CLI not found. Checking for npm...")
            npm_path = None
            
            # Try to find npm in common Windows locations
            possible_npm_paths = [
                'npm',  # In PATH
                r'C:\Program Files\nodejs\npm.cmd',
                r'C:\Program Files (x86)\nodejs\npm.cmd',
                os.path.expanduser(r'~\AppData\Roaming\npm\npm.cmd'),
                os.path.expanduser(r'~\AppData\Local\Programs\nodejs\npm.cmd'),
            ]
            
            # Also check environment variables
            node_path = os.environ.get('NODE_PATH', '')
            if node_path:
                possible_npm_paths.append(os.path.join(node_path, 'npm.cmd'))
            
            # Try each path
            for npm_candidate in possible_npm_paths:
                try:
                    npm_check = subprocess.run(
                        [npm_candidate, '--version'],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        shell=True
                    )
                    if npm_check.returncode == 0:
                        npm_path = npm_candidate
                        print(f"Found npm at: {npm_path} (version: {npm_check.stdout.strip()})")
                        break
                except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
                    continue
            
            if not npm_path:
                error_msg = (
                    "Vercel CLI not found and npm is not available. "
                    "To deploy to Vercel, you need to install Node.js (which includes npm) first.\n\n"
                    "Installation steps:\n"
                    "1. Download and install Node.js from https://nodejs.org/\n"
                    "2. Restart your terminal/command prompt\n"
                    "3. Run: npm i -g vercel\n"
                    "4. Run: vercel login\n"
                    "5. Then try deploying again\n\n"
                    "Alternatively, you can deploy manually by:\n"
                    f"1. Open a terminal in: {deploy_path}\n"
                    "2. Run: vercel --prod"
                )
                print(f"ERROR: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'deploy_dir': deploy_dir,
                    'provnum': provnum,
                    'needs_nodejs': True
                }), 500
            
            # Try to install Vercel CLI
            try:
                print(f"Installing Vercel CLI using npm at {npm_path} (this may take a minute)...")
                install_result = subprocess.run(
                    [npm_path, 'install', '-g', 'vercel'],
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3 minutes for installation
                    shell=True
                )
                
                if install_result.returncode == 0:
                    print("Vercel CLI installed successfully!")
                    vercel_installed = True
                    # Find the newly installed vercel command
                    vercel_path = shutil.which('vercel')
                    if vercel_path:
                        vercel_cmd = vercel_path
                    elif platform.system() == 'Windows':
                        vercel_cmd_path = shutil.which('vercel.cmd')
                        if vercel_cmd_path:
                            vercel_cmd = vercel_cmd_path
                        else:
                            # Check common locations
                            possible_paths = [
                                os.path.expanduser(r'~\AppData\Roaming\npm\vercel.cmd'),
                                os.path.expanduser(r'~\AppData\Local\Programs\nodejs\vercel.cmd'),
                            ]
                            for path in possible_paths:
                                if os.path.exists(path):
                                    vercel_cmd = path
                                    break
                    if not vercel_cmd:
                        vercel_cmd = 'vercel'  # Fallback
                    
                    # Verify installation
                    verify_result = subprocess.run(
                        [vercel_cmd, '--version'],
                        capture_output=True,
                        text=True,
                        timeout=5,
                        shell=True
                    )
                    if verify_result.returncode == 0:
                        vercel_version = verify_result.stdout.strip()
                        print(f"Verified Vercel CLI: {vercel_version} at {vercel_cmd}")
                else:
                    error_output = install_result.stderr or install_result.stdout
                    raise Exception(f"Installation failed: {error_output}")
                    
            except subprocess.TimeoutExpired:
                raise Exception("Vercel CLI installation timed out after 3 minutes")
            except Exception as install_error:
                error_msg = (
                    f"Vercel CLI installation failed: {str(install_error)}\n\n"
                    "Please install Vercel CLI manually:\n"
                    "1. Open a terminal/command prompt\n"
                    "2. Run: npm i -g vercel\n"
                    "3. Run: vercel login\n"
                    "4. Then try deploying again\n\n"
                    f"Or deploy manually from: {deploy_path}"
                )
                print(f"ERROR: {error_msg}")
                return jsonify({
                    'success': False,
                    'error': error_msg,
                    'deploy_dir': deploy_dir,
                    'provnum': provnum
                }), 500
        
        # Ensure we have a vercel command
        if not vercel_cmd:
            vercel_cmd = 'vercel'  # Fallback - will use shell=True to find it
        
        try:
            # Change to deployment directory
            original_cwd = os.getcwd()
            os.chdir(deploy_path)
            
            try:
                # On Windows with shell=True, pass command as string for proper PATH resolution
                # This ensures Windows can find the command even after os.chdir()
                if platform.system() == 'Windows':
                    # Link the project (non-interactive)
                    print(f"Linking Vercel project (using: {vercel_cmd})...")
                    # Use string command with quotes for paths with spaces
                    if ' ' in vercel_cmd or vercel_cmd != 'vercel':
                        link_command = f'"{vercel_cmd}" link --project pbj320-{provnum} --yes'
                    else:
                        link_command = f'{vercel_cmd} link --project pbj320-{provnum} --yes'
                    
                    link_result = subprocess.run(
                        link_command,
                        capture_output=True,
                        text=True,
                        timeout=60,
                        shell=True  # Required on Windows to find commands
                    )
                    
                    if link_result.returncode != 0:
                        print(f"Warning: vercel link failed: {link_result.stderr}")
                        # Continue anyway - project might already be linked
                    
                    # Deploy to production
                    print("Deploying to Vercel production...")
                    if ' ' in vercel_cmd or vercel_cmd != 'vercel':
                        deploy_command = f'"{vercel_cmd}" --prod --yes'
                    else:
                        deploy_command = f'{vercel_cmd} --prod --yes'
                    
                    deploy_result = subprocess.run(
                        deploy_command,
                        capture_output=True,
                        text=True,
                        timeout=600,  # 10 minutes timeout
                        shell=True  # Required on Windows to find commands
                    )
                else:
                    # Non-Windows: use list format
                    # Link the project (non-interactive)
                    print(f"Linking Vercel project (using: {vercel_cmd})...")
                    link_result = subprocess.run(
                        [vercel_cmd, 'link', '--project', f'pbj320-{provnum}', '--yes'],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        shell=False
                    )
                    
                    if link_result.returncode != 0:
                        print(f"Warning: vercel link failed: {link_result.stderr}")
                        # Continue anyway - project might already be linked
                    
                    # Deploy to production
                    print("Deploying to Vercel production...")
                    deploy_result = subprocess.run(
                        [vercel_cmd, '--prod', '--yes'],
                        capture_output=True,
                        text=True,
                        timeout=600,  # 10 minutes timeout
                        shell=False
                    )
                
                if deploy_result.returncode == 0:
                    # Extract URL from output
                    output = deploy_result.stdout + deploy_result.stderr
                    # Look for URL pattern: https://pbj320-{provnum}.vercel.app
                    import re
                    url_match = re.search(r'https://pbj320-\d+\.vercel\.app', output)
                    if url_match:
                        vercel_url = url_match.group(0) + '/'  # Add trailing slash
                    else:
                        # Fallback to expected URL format
                        vercel_url = f"https://pbj320-{provnum}.vercel.app/"
                    print(f"Deployment successful! URL: {vercel_url}")
                else:
                    error_output = deploy_result.stderr or deploy_result.stdout
                    print(f"ERROR: vercel deploy failed: {error_output}")
                    raise Exception(f"Vercel deployment failed: {error_output}")
                    
            finally:
                os.chdir(original_cwd)
                
        except subprocess.TimeoutExpired:
            error_msg = "Vercel deployment timed out after 10 minutes"
            print(f"ERROR: {error_msg}")
            return jsonify({
                'success': False,
                'error': error_msg,
                'deploy_dir': deploy_dir,
                'provnum': provnum
            }), 500
        except Exception as e:
            error_str = str(e)
            print(f"ERROR: Vercel deployment error: {error_str}")
            return jsonify({
                'success': False,
                'error': f"Vercel deployment failed: {error_str}",
                'deploy_dir': deploy_dir,
                'provnum': provnum
            }), 500
        
        # Get facility name for the link
        facility_name = None
        try:
            if load_facility_data is not None:
                df = load_facility_data(provnum)
                if df is not None and not df.empty and 'PROVNAME' in df.columns:
                    facility_name = df['PROVNAME'].iloc[0]
                    if format_facility_name is not None:
                        facility_name = format_facility_name(facility_name)
        except Exception:
            pass  # Ignore errors getting facility name
        
        response_data = {
            'success': True,
            'provnum': provnum,
            'deploy_dir': deploy_dir,
            'facility_name': facility_name,
        }
        
        if vercel_url:
            response_data['vercel_url'] = vercel_url
            response_data['message'] = f'Dashboard deployed to Vercel: {vercel_url}'
        else:
            # Should not happen if deployment succeeded, but handle gracefully
            vercel_url = f"https://pbj320-{provnum}.vercel.app/"
            response_data['vercel_url'] = vercel_url
            response_data['message'] = f'Vercel deployment package created. Expected URL: {vercel_url}'
        
        return jsonify(response_data)
            
    except Exception as e:
        import traceback
        error_str = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error in deploy_vercel: {error_str}")
        print(traceback_str)
        
        # Check for file locking errors
        if '32' in error_str or 'being used' in error_str.lower() or 'WinError 32' in error_str:
            return jsonify({
                'error': 'File is locked. The deployment script will retry automatically. If this persists, wait a few seconds and try again.'
            }), 500
        
        # Check for file not found errors (especially template)
        if 'not found' in error_str.lower() or 'FileNotFoundError' in error_str:
            if 'template' in error_str.lower() or 'dynamic_facility_dashboard.html' in error_str:
                return jsonify({
                    'error': f'Template file not found. Please ensure templates/dynamic_facility_dashboard.html exists in the project root. Error: {error_str}'
                }), 500
        
        return jsonify({'error': f'Error creating Vercel package: {error_str}'}), 500

@app.route('/api/generate-report', methods=['POST'])
def generate_report():
    """Generate facility report with custom parameters"""
    try:
        data = request.json
        # Normalize CCN using canonical identifier layer
        provnum_raw = str(data.get('provnum', '')).strip()
        try:
            provnum = normalize_ccn(provnum_raw)
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid facility code: {str(e)}'}), 400
        start_date_str = data.get('start_date', '')
        end_date_str = data.get('end_date', '')
        quarters = data.get('quarters', [])
        years = data.get('years', [])
        key_dates_str = data.get('key_dates', '')
        
        if not provnum.isdigit() or len(provnum) != 6:
            return jsonify({'error': 'Invalid facility code. Must be 6 digits.'}), 400
        
        # Determine date range from selected mode
        if quarters:
            # Convert quarters to date range
            # Quarters format: "2024Q1", "2024Q2", etc.
            quarter_dates = []
            for q in quarters:
                year, quarter = q.split('Q')
                if quarter == '1':
                    quarter_dates.append((f"{year}-01-01", f"{year}-03-31"))
                elif quarter == '2':
                    quarter_dates.append((f"{year}-04-01", f"{year}-06-30"))
                elif quarter == '3':
                    quarter_dates.append((f"{year}-07-01", f"{year}-09-30"))
                elif quarter == '4':
                    quarter_dates.append((f"{year}-10-01", f"{year}-12-31"))
            
            if quarter_dates:
                start_date_str = min(q[0] for q in quarter_dates)
                end_date_str = max(q[1] for q in quarter_dates)
        elif years:
            # Convert years to date range
            years_int = [int(y) for y in years if y.isdigit()]
            if years_int:
                start_date_str = f"{min(years_int)}-01-01"
                end_date_str = f"{max(years_int)}-12-31"
        
        if not start_date_str or not end_date_str:
            return jsonify({'error': 'Please select a date range, quarters, or years'}), 400
        
        # Parse dates
        def parse_date(date_str):
            for fmt in ("%Y-%m-%d", "%m-%d-%Y", "%m/%d/%Y", "%Y/%m/%d"):
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            raise ValueError(f"Invalid date format: {date_str}")
        
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)
        
        # Parse key dates
        key_dates = []
        if key_dates_str:
            for date_str in key_dates_str.split(','):
                date_str = date_str.strip()
                if date_str:  # Only process non-empty strings
                    try:
                        parsed_date = parse_date(date_str)
                        if parsed_date:
                            key_dates.append(parsed_date)
                            print(f"  Parsed key date: {date_str} -> {parsed_date}")
                    except (ValueError, AttributeError) as e:
                        print(f"  Warning: Could not parse key date '{date_str}': {e}")
                        continue
        
        print(f"  Total key dates parsed: {len(key_dates)}")
        
        # Generate report by calling the report generation functions directly
        from facility_report_lib import (
            load_facility_data,
            get_facility_info,
            calculate_quarterly_metrics,
            get_state_averages,
            calculate_period_metrics,
            calculate_days_under_state_minimum,
            get_daily_staffing,
            load_provider_info_data,
            extract_red_flags_history,
            extract_case_mix_data,
            get_macpac_state_standards,
            generate_attorney_report
        )
        
        if load_facility_data is None:
            return jsonify({'error': 'Report generation functions not available'}), 500
        
        # Load facility data
        df = load_facility_data(provnum)
        if df is None or df.empty:
            return jsonify({'error': f'No data found for facility {provnum}'}), 404
        
        # Get facility info
        facility_info = get_facility_info(df, start_date, end_date)
        
        # Filter to quarters in the date range
        start_quarter = f"{start_date.year}Q{(start_date.month - 1) // 3 + 1}"
        end_quarter = f"{end_date.year}Q{(end_date.month - 1) // 3 + 1}"
        all_quarters = sorted(df['CY_Qtr'].unique())
        quarters_in_range = [q for q in all_quarters if q >= start_quarter and q <= end_quarter]
        
        # Calculate quarterly metrics
        quarterly_data = {}
        state_comparisons = {}
        for quarter in quarters_in_range:
            q_metrics = calculate_quarterly_metrics(df, quarter)
            if q_metrics:
                quarterly_data[quarter] = q_metrics
            state_avg = get_state_averages(facility_info['state'], quarter)
            if state_avg:
                state_comparisons[quarter] = state_avg
        
        period_metrics = calculate_period_metrics(df, start_date, end_date)
        if period_metrics is None:
            period_metrics = {}
        
        macpac_standards = get_macpac_state_standards(facility_info['state'])
        if macpac_standards is None:
            macpac_standards = {}
        
        # Calculate days under state minimum and add to period_metrics
        if macpac_standards and macpac_standards.get('min_staffing', 0) > 0:
            days_under = calculate_days_under_state_minimum(df, start_date, end_date, macpac_standards['min_staffing'])
            if days_under:
                period_metrics.update(days_under)
                print(f"  Added compliance data: {days_under.get('total_days', 0)} total days, {days_under.get('days_under_minimum_total', 0)} days under minimum (Total), {days_under.get('days_under_minimum_direct', 0)} days under minimum (Direct)")
        
        # Get daily staffing for key dates (if any)
        daily_staffing = []
        if key_dates:
            for key_date in key_dates:
                day_data = get_daily_staffing(df, key_date)
                if day_data:
                    daily_staffing.append(day_data)
        
        # Load provider info
        provider_info_df = load_provider_info_data(provnum)
        red_flags_history = []
        case_mix_data = []
        
        if provider_info_df is not None and not provider_info_df.empty:
            try:
                red_flags_history = extract_red_flags_history(provider_info_df, start_date, end_date)
            except Exception as e:
                print(f"Error extracting red flags: {e}")
                red_flags_history = []
            
            try:
                case_mix_data = extract_case_mix_data(provider_info_df, start_date, end_date, list(quarterly_data.keys()))
            except Exception as e:
                print(f"Error extracting case mix: {e}")
                case_mix_data = []
        
        # Get watermark option
        watermark = data.get('watermark', False)
        
        # Generate report
        try:
            html_report = generate_attorney_report(
                provnum=provnum,
                facility_name=facility_info['name'],
                city=facility_info['city'],
                state=facility_info['state'],
                start_date=start_date,
                end_date=end_date,
                key_dates=key_dates,
                quarterly_data=quarterly_data,
                state_comparisons=state_comparisons,
                period_metrics=period_metrics,
                daily_staffing=daily_staffing,
                macpac_standards=macpac_standards,
                red_flags_history=red_flags_history,
                case_mix_data=case_mix_data,
                pbj_df=df,
                include_total_staffing=True,  # Include total nurse metrics (not just direct)
                watermark=watermark
            )
        except Exception as report_error:
            import traceback
            error_msg = str(report_error)
            traceback_str = traceback.format_exc()
            print(f"Error generating report: {error_msg}")
            print(traceback_str)
            return jsonify({
                'error': f'Error generating report: {error_msg}',
                'details': traceback_str if app.debug else None
            }), 500
        
        # Save report with lowercase filename
        facility_name_raw = facility_info['name']
        if format_facility_name is not None:
            facility_name_formatted = format_facility_name(facility_name_raw)
        else:
            facility_name_formatted = facility_name_raw
        facility_name_safe = facility_name_formatted.lower().replace(' ', '_').replace('&', 'and').replace(',', '').replace('.', '').replace('/', '_').replace("'", '').replace('-', '_')
        report_folder = f"reports/facility_{provnum}_{facility_name_safe}"
        os.makedirs(report_folder, exist_ok=True)
        
        output_filename = f"pbj320_report_{provnum}_{facility_name_safe}.html"
        report_file = os.path.join(report_folder, output_filename)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        return jsonify({
            'success': True,
            'provnum': provnum,
            'report_file': report_file,
            'report_folder': report_folder,
            'message': f'Report generated successfully'
        })
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/download-report/<path:filename>')
def download_report(filename):
    """Download generated report file"""
    try:
        # Decode the filename (it's URL encoded)
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        
        # Security: only allow files from reports directory or facility_red_flag_report/reports
        if not (filename.startswith('reports/') or filename.startswith('facility_red_flag_report/reports/')):
            return jsonify({'error': 'Invalid file path'}), 400
        
        # Normalize path separators for Windows
        filename = filename.replace('\\', '/')
        
        # Handle facility_red_flag_report paths
        actual_path = filename
        
        if os.path.exists(actual_path):
            # Get the original filename for download
            original_basename = os.path.basename(filename)
            
            # For entity/state reports, keep original filename
            # For facility reports, convert to lowercase pbj320 format
            if original_basename.startswith('PBJ_Report_'):
                # Extract parts and rebuild in lowercase format
                parts = original_basename.replace('.html', '').split('_')
                if len(parts) >= 3:
                    provnum_part = parts[2] if len(parts) > 2 else ''
                    facility_part = '_'.join(parts[3:]) if len(parts) > 3 else ''
                    download_name = f"pbj320_report_{provnum_part}_{facility_part}.html".lower()
                else:
                    download_name = original_basename.lower()
            elif original_basename.startswith('ENTITY_') or original_basename.startswith(('NY_', 'CA_', 'FL_')):
                # Entity or state reports - keep original name
                download_name = original_basename
            elif not original_basename.startswith('pbj320_report_'):
                # If it's already in pbj320 format, just ensure lowercase
                download_name = original_basename.lower()
            else:
                download_name = original_basename
            
            # Determine MIME type based on file extension
            if actual_path.endswith('.pdf'):
                mimetype = 'application/pdf'
            elif actual_path.endswith('.html'):
                mimetype = 'text/html; charset=utf-8'
            elif actual_path.endswith('.md'):
                mimetype = 'text/markdown; charset=utf-8'
            else:
                mimetype = 'application/octet-stream'
            
            # Use send_file for better browser compatibility and Chrome-friendly downloads
            # Flask's send_file handles headers properly and Chrome trusts it more
            return send_file(
                actual_path,
                mimetype=mimetype,
                as_attachment=True,
                download_name=download_name
            )
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/search-entities', methods=['GET'])
def search_entities():
    """Search entities for autocomplete"""
    try:
        query = request.args.get('q', '').strip().lower()
        limit = int(request.args.get('limit', '20'))
        
        # Load provider info to get entities
        if load_provider_info_data is None:
            return jsonify({'entities': []})
        
        df = load_provider_info_data()
        if df is None or df.empty:
            return jsonify({'entities': []})
        
        # Find chain ID column
        chain_id_col = None
        for col in ['Chain ID', 'chain_id', 'Chain_ID', 'Entity ID', 'entity_id', 'affiliated_entity_id']:
            if col in df.columns:
                chain_id_col = col
                break
        
        if not chain_id_col:
            return jsonify({'entities': []})
        
        # Get unique entities
        entity_df = df[[chain_id_col]].dropna()
        entity_df = entity_df[entity_df[chain_id_col].astype(str).str.strip() != '']
        entity_df = entity_df[entity_df[chain_id_col].astype(str).str.upper() != 'NAN']
        
        # Get entity names
        chain_name_col = None
        for col in ['Chain Name', 'chain_name', 'Chain_Name', 'Entity Name', 'entity_name', 'affiliated_entity_name']:
            if col in df.columns:
                chain_name_col = col
                break
        
        entities = []
        seen_ids = set()
        
        for _, row in entity_df.iterrows():
            entity_id_raw = str(row[chain_id_col]).strip()
            if not entity_id_raw or entity_id_raw.upper() in ['NAN', 'NONE', '']:
                continue
            
            try:
                # Normalize entity ID
                entity_id = normalize_entity_id(entity_id_raw)
                if not entity_id or entity_id in seen_ids:
                    continue
                seen_ids.add(entity_id)
                
                # Get entity name
                entity_name = None
                if chain_name_col:
                    entity_rows = df[df[chain_id_col].astype(str).str.strip() == entity_id_raw]
                    if not entity_rows.empty:
                        name_val = entity_rows[chain_name_col].iloc[0]
                        if pd.notna(name_val):
                            entity_name = str(name_val).strip()
                            if entity_name.upper() in ['NAN', 'NONE', '']:
                                entity_name = None
                
                # Count facilities
                facility_count = len(df[df[chain_id_col].astype(str).str.strip() == entity_id_raw])
                
                # Filter by query
                search_text = f"{entity_id} {entity_name or ''}".lower()
                if query and query not in search_text:
                    continue
                
                entities.append({
                    'id': entity_id,
                    'name': entity_name or f'Entity {entity_id}',
                    'facility_count': facility_count
                })
                
                if len(entities) >= limit:
                    break
                    
            except (ValueError, TypeError):
                continue
        
        # Sort by name or ID
        entities.sort(key=lambda x: (x['name'] or '').lower())
        
        return jsonify({'entities': entities})
        
    except Exception as e:
        return jsonify({'entities': [], 'error': str(e)})

@app.route('/api/search-states', methods=['GET'])
def search_states():
    """Get state list for autocomplete"""
    try:
        query = request.args.get('q', '').strip().lower()
        
        US_STATES = [
            {'code': 'AL', 'name': 'Alabama'}, {'code': 'AK', 'name': 'Alaska'},
            {'code': 'AZ', 'name': 'Arizona'}, {'code': 'AR', 'name': 'Arkansas'},
            {'code': 'CA', 'name': 'California'}, {'code': 'CO', 'name': 'Colorado'},
            {'code': 'CT', 'name': 'Connecticut'}, {'code': 'DE', 'name': 'Delaware'},
            {'code': 'DC', 'name': 'District of Columbia'}, {'code': 'FL', 'name': 'Florida'},
            {'code': 'GA', 'name': 'Georgia'}, {'code': 'HI', 'name': 'Hawaii'},
            {'code': 'ID', 'name': 'Idaho'}, {'code': 'IL', 'name': 'Illinois'},
            {'code': 'IN', 'name': 'Indiana'}, {'code': 'IA', 'name': 'Iowa'},
            {'code': 'KS', 'name': 'Kansas'}, {'code': 'KY', 'name': 'Kentucky'},
            {'code': 'LA', 'name': 'Louisiana'}, {'code': 'ME', 'name': 'Maine'},
            {'code': 'MD', 'name': 'Maryland'}, {'code': 'MA', 'name': 'Massachusetts'},
            {'code': 'MI', 'name': 'Michigan'}, {'code': 'MN', 'name': 'Minnesota'},
            {'code': 'MS', 'name': 'Mississippi'}, {'code': 'MO', 'name': 'Missouri'},
            {'code': 'MT', 'name': 'Montana'}, {'code': 'NE', 'name': 'Nebraska'},
            {'code': 'NV', 'name': 'Nevada'}, {'code': 'NH', 'name': 'New Hampshire'},
            {'code': 'NJ', 'name': 'New Jersey'}, {'code': 'NM', 'name': 'New Mexico'},
            {'code': 'NY', 'name': 'New York'}, {'code': 'NC', 'name': 'North Carolina'},
            {'code': 'ND', 'name': 'North Dakota'}, {'code': 'OH', 'name': 'Ohio'},
            {'code': 'OK', 'name': 'Oklahoma'}, {'code': 'OR', 'name': 'Oregon'},
            {'code': 'PA', 'name': 'Pennsylvania'}, {'code': 'RI', 'name': 'Rhode Island'},
            {'code': 'SC', 'name': 'South Carolina'}, {'code': 'SD', 'name': 'South Dakota'},
            {'code': 'TN', 'name': 'Tennessee'}, {'code': 'TX', 'name': 'Texas'},
            {'code': 'UT', 'name': 'Utah'}, {'code': 'VT', 'name': 'Vermont'},
            {'code': 'VA', 'name': 'Virginia'}, {'code': 'WA', 'name': 'Washington'},
            {'code': 'WV', 'name': 'West Virginia'}, {'code': 'WI', 'name': 'Wisconsin'},
            {'code': 'WY', 'name': 'Wyoming'}
        ]
        
        if query:
            filtered = [s for s in US_STATES if query in s['code'].lower() or query in s['name'].lower()]
        else:
            filtered = US_STATES
        
        return jsonify({'states': filtered})
        
    except Exception as e:
        return jsonify({'states': [], 'error': str(e)})

@app.route('/api/check-facility', methods=['POST'])
def check_facility():
    """Check if facility data exists and get date range - checks source data if CSV doesn't exist"""
    try:
        # Handle missing or invalid JSON
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.json
        if data is None:
            return jsonify({'error': 'Invalid JSON data'}), 400
        
        provnum_raw = str(data.get('provnum', '')).strip()
        if not provnum_raw:
            return jsonify({'error': 'Missing facility code'}), 400
        
        # Normalize CCN using canonical identifier layer
        try:
            provnum = normalize_ccn(provnum_raw)
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid facility code: {str(e)}'}), 400
        
        if not provnum.isdigit() or len(provnum) != 6:
            return jsonify({'error': f'Invalid facility code. Must be 6 digits. Got: {len(provnum)} digits from "{provnum_raw}"'}), 400
        
        csv_file = f'facility_{provnum}_complete_data.csv'
        provider_csv_file = f'facility_{provnum}_provider_info_data.csv'
        
        # Try to load facility data to get name and date range
        facility_name = None
        start_date = None
        end_date = None
        city = None
        state = None
        entity = None
        row_count = 0
        estimated_time_seconds = 60  # Default estimate
        df = None
        
        # First, try to load from existing CSV if it exists (fast path)
        if os.path.exists(csv_file) and load_facility_data is not None:
            try:
                df = load_facility_data(provnum)
                if df is not None and not df.empty:
                    row_count = len(df)
            except Exception as load_error:
                print(f"Could not load facility data from CSV: {load_error}")
                df = None
        
        # If CSV doesn't exist, do a quick check in source files (limit to prevent timeout)
        if (df is None or df.empty) and not os.path.exists(csv_file):
            try:
                import glob
                nurse_files = glob.glob('standardized_PBJ/PBJ_dailynursestaffing_*.csv')
                if not nurse_files:
                    nurse_files = glob.glob('PBJcsv/PBJ_dailynursestaffing_*.csv')
                
                nurse_files.sort()
                
                # Search variants for provnum matching
                # Use normalized CCN for search
                provnum_norm = normalize_ccn(provnum) if provnum else None
                search_variants = []
                if provnum_norm:
                    search_variants.append(provnum_norm)
                    if provnum_norm.lstrip('0'):
                        search_variants.append(provnum_norm.lstrip('0'))
                
                # Quick check: only check first 3 files to prevent timeout
                found = False
                for file_path in nurse_files[:3]:
                    try:
                        # Quick check: read first 1000 rows only
                        temp_df = pd.read_csv(file_path, low_memory=False, nrows=1000)
                        if 'PROVNUM' in temp_df.columns:
                            temp_df['PROVNUM'] = temp_df['PROVNUM'].astype(str).str.zfill(6)
                            if temp_df['PROVNUM'].isin(search_variants).any():
                                found = True
                                break
                    except Exception:
                        continue
                
                # If found, return exists=True but don't load full data (too slow)
                if found:
                    return jsonify({
                        'exists': True,
                        'has_provider_info': os.path.exists(provider_csv_file),
                        'facility_name': None,  # Will be populated when CSV is created
                        'city': None,
                        'state': None,
                        'entity': None,
                        'provnum': provnum,
                        'start_date': None,
                        'end_date': None,
                        'row_count': 0,
                        'estimated_time_seconds': 120  # Default estimate
                    })
            except Exception as source_error:
                print(f"Error checking source data: {source_error}")
                pass
        
        # If we still don't have data, return that facility doesn't exist
        if df is None or df.empty:
            return jsonify({
                'exists': False,
                'has_provider_info': os.path.exists(provider_csv_file),
                'facility_name': None,
                'city': None,
                'state': None,
                'entity': None,
                'provnum': provnum,
                'start_date': None,
                'end_date': None,
                'row_count': 0,
                'estimated_time_seconds': 60  # Default estimate for new facility
            })
        
        # We have data, extract info
        if df is not None and not df.empty:
            row_count = len(df)
            if 'PROVNAME' in df.columns:
                facility_name = df['PROVNAME'].iloc[0] if pd.notna(df['PROVNAME'].iloc[0]) else None
            
            # Get date range from WorkDate column
            if 'WorkDate' in df.columns:
                try:
                    # Convert to datetime if needed
                    if not pd.api.types.is_datetime64_any_dtype(df['WorkDate']):
                        df['WorkDate'] = pd.to_datetime(df['WorkDate'], errors='coerce')
                    
                    # Get min and max dates
                    valid_dates = df['WorkDate'].dropna()
                    if len(valid_dates) > 0:
                        start_date = valid_dates.min().strftime('%Y-%m-%d')
                        end_date = valid_dates.max().strftime('%Y-%m-%d')
                except Exception as date_error:
                    print(f"Error processing dates: {date_error}")
            
            # Estimate processing time based on row count
            # Rough estimate: ~1000 rows per second for CSV creation
            estimated_time_seconds = max(30, min(300, row_count / 1000))
            
            # Try to get city/state from facility data
            if 'CITY' in df.columns:
                city_val = df['CITY'].iloc[0] if pd.notna(df['CITY'].iloc[0]) else None
                if city_val and str(city_val).strip().upper() not in ['NAN', 'NONE', '']:
                    city = str(city_val).strip()
            if 'STATE' in df.columns:
                state_val = df['STATE'].iloc[0] if pd.notna(df['STATE'].iloc[0]) else None
                if state_val and str(state_val).strip().upper() not in ['NAN', 'NONE', '']:
                    state = str(state_val).strip()
        
        # Try to get additional info from provider info CSV
        if os.path.exists(provider_csv_file) and load_provider_info_data is not None:
            try:
                provider_df = load_provider_info_data(provnum)
                if provider_df is not None and not provider_df.empty:
                    # Get the most recent record
                    latest = provider_df.iloc[-1]
                    
                    # Try to get city if not already found
                    if not city:
                        city_cols = ['City', 'City/Town', 'City Town', 'Provider City', 'City Name', 'city']
                        for col in city_cols:
                            if col in latest.index and pd.notna(latest.get(col)):
                                city_val = str(latest.get(col)).strip()
                                if city_val and city_val.upper() not in ['NAN', 'NONE', '']:
                                    city = city_val
                                    break
                    
                    # Try to get state if not already found
                    if not state:
                        state_cols = ['State', 'State Code', 'Provider State', 'state']
                        for col in state_cols:
                            if col in latest.index and pd.notna(latest.get(col)):
                                state_val = str(latest.get(col)).strip()
                                if state_val and state_val.upper() not in ['NAN', 'NONE', '']:
                                    state = state_val
                                    break
                    
                    # Try to get entity/affiliated entity
                    entity_cols = ['affiliated_entity_name', 'Affiliated Entity Name', 'Chain Name', 'Entity Name', 'chain_name']
                    for col in entity_cols:
                        if col in latest.index and pd.notna(latest.get(col)):
                            entity_val = str(latest.get(col)).strip()
                            if entity_val and entity_val.upper() not in ['N', 'N/A', 'NAN', 'NONE', '']:
                                entity = entity_val
                                break
            except Exception as provider_error:
                print(f"Error loading provider info: {provider_error}")
                # Continue without provider info
        
        # Format facility name if available
        formatted_name = None
        if facility_name:
            if format_facility_name is not None:
                formatted_name = format_facility_name(facility_name)
            else:
                formatted_name = facility_name
        
        # Calculate estimated time if not already set (should be set above if df exists)
        if row_count > 0 and estimated_time_seconds == 60:
            estimated_time_seconds = max(30, min(300, row_count / 1000))
        
        return jsonify({
            'exists': os.path.exists(csv_file),
            'has_provider_info': os.path.exists(provider_csv_file),
            'facility_name': formatted_name or facility_name,
            'city': city,
            'state': state,
            'entity': entity,
            'provnum': provnum,
            'start_date': start_date,
            'end_date': end_date,
            'row_count': row_count,
            'estimated_time_seconds': estimated_time_seconds
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/entity-longitudinal-metrics', methods=['GET'])
def get_entity_longitudinal_metrics_api():
    """Get longitudinal metrics for an entity."""
    try:
        entity_id_raw = request.args.get('entity_id', '').strip()
        
        if not entity_id_raw:
            return jsonify({'error': 'Missing entity_id parameter'}), 400
        
        # Normalize entity ID
        try:
            entity_id = normalize_entity_id(entity_id_raw)
            if not entity_id:
                return jsonify({'error': f'Invalid entity ID format: {entity_id_raw}'}), 400
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid entity ID: {str(e)}'}), 400
        
        # Get key metrics over time
        metrics = get_entity_key_metrics_over_time(entity_id)
        
        if metrics is None:
            return jsonify({
                'entity_id': entity_id,
                'available': False,
                'message': 'No longitudinal data available for this entity'
            })
        
        # Get CHOW facilities with detailed information
        chow_facilities = get_entity_chow_facilities(entity_id)
        metrics['chow_facilities'] = chow_facilities
        metrics['chow_count'] = len(chow_facilities)
        
        # Update chow_status to include detailed facilities
        chow_status = get_entity_chow_status(entity_id)
        if chow_status:
            chow_status['chow_facilities'] = chow_facilities
        
        return jsonify(metrics)
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error getting longitudinal metrics: {str(e)}',
            'details': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/api/generate-entity-report', methods=['POST'])
def generate_entity_report_api():
    """Generate entity report via API"""
    try:
        data = request.json
        entity_id_raw = str(data.get('entity_id', '')).strip()
        
        if not entity_id_raw:
            return jsonify({'error': 'Missing entity ID'}), 400
        
        # Normalize entity ID using canonical identifier layer
        try:
            entity_id = normalize_entity_id(entity_id_raw)
            if not entity_id:
                return jsonify({'error': f'Invalid entity ID format: {entity_id_raw}'}), 400
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid entity ID: {str(e)}'}), 400
        
        # Get optional parameters
        skip_existing = data.get('skip_existing', False)
        min_census = data.get('min_census')
        drill_down_num = data.get('drill_down_num')
        drill_down_criteria = data.get('drill_down_criteria', [])
        selected_metrics = data.get('selected_metrics')  # List of metric keys to display
        
        # Get longitudinal metrics and CHOW data with detailed facility information
        longitudinal_metrics = load_entity_longitudinal_metrics(entity_id)
        
        # Get CHOW data safely
        chow_status = None
        try:
            chow_status = get_entity_chow_status(entity_id)
            if chow_status:
                try:
                    chow_facilities = get_entity_chow_facilities(entity_id)
                    chow_status['chow_facilities'] = chow_facilities
                except Exception as chow_error:
                    print(f"Warning: Error getting CHOW facilities: {chow_error}")
                    # Continue without CHOW facilities if there's an error
                    if chow_status:
                        chow_status['chow_facilities'] = []
        except Exception as chow_error:
            print(f"Warning: Error getting CHOW status: {chow_error}")
            chow_status = None
        
        # Import entity report generator
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'facility_red_flag_report'))
        try:
            from generate_entity_report import generate_entity_report  # type: ignore
        except ImportError:
            return jsonify({'error': 'Entity report generator module not found'}), 500
        
        # Generate report with all options including longitudinal and CHOW data
        success = generate_entity_report(
            entity_id=entity_id,
            skip_existing=skip_existing,
            min_census=min_census,
            drill_down_num=drill_down_num,
            drill_down_criteria=drill_down_criteria,
            longitudinal_data=longitudinal_metrics,
            chow_data=chow_status,
            selected_metrics=selected_metrics
        )
        
        if success:
            report_folder = os.path.join('facility_red_flag_report', 'reports', f'ENTITY_{entity_id}')
            
            # Find generated report files (exclude JSON - internal use only for interactive dashboard)
            report_files = []
            if os.path.exists(report_folder):
                import urllib.parse
                for filename in os.listdir(report_folder):
                    # Only include user-facing reports, exclude JSON data files (used by interactive dashboard)
                    if filename.endswith(('.html', '.pdf', '.md')) and not filename.endswith('_sff_data.json'):
                        file_path = os.path.join(report_folder, filename).replace('\\', '/')
                        # Create download URL (URL encode the path)
                        encoded_path = urllib.parse.quote(file_path, safe='')
                        download_url = f"/api/download-report/{encoded_path}"
                        report_files.append({
                            'filename': filename,
                            'path': file_path,
                            'download_url': download_url
                        })
            
            return jsonify({
                'success': True,
                'entity_id': entity_id,
                'report_folder': report_folder,
                'report_files': report_files,
                'message': f'Entity report generated for entity {entity_id}'
            })
        else:
            return jsonify({'error': 'Failed to generate entity report'}), 500
            
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error generating entity report: {str(e)}',
            'details': traceback.format_exc() if app.debug else None
        }), 500

@app.route('/api/generate-state-report', methods=['POST'])
def generate_state_report_api():
    """Generate state report via API"""
    try:
        data = request.json
        state_raw = str(data.get('state', '')).strip()
        
        if not state_raw:
            return jsonify({'error': 'Missing state code'}), 400
        
        # Normalize state code using canonical identifier layer
        try:
            state = normalize_state_code(state_raw)
            if not validate_state_code(state):
                return jsonify({'error': f'Invalid state code: {state_raw}'}), 400
        except (ValueError, TypeError) as e:
            return jsonify({'error': f'Invalid state code: {str(e)}'}), 400
        
        # Get optional parameters
        skip_existing = data.get('skip_existing', False)
        min_census = data.get('min_census')
        drill_down_num = data.get('drill_down_num')
        drill_down_criteria = data.get('drill_down_criteria', [])
        
        # Import state report generator
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'facility_red_flag_report'))
        try:
            from generate_state_report import generate_state_report  # type: ignore
        except ImportError:
            return jsonify({'error': 'State report generator module not found'}), 500
        
        # Generate report with all options
        success = generate_state_report(
            state=state,
            skip_existing=skip_existing,
            min_census=min_census,
            drill_down_num=drill_down_num,
            drill_down_criteria=drill_down_criteria
        )
        
        if success:
            report_folder = os.path.join('facility_red_flag_report', 'reports', state)
            
            # Find generated report files (exclude JSON - internal use only for interactive dashboard)
            report_files = []
            if os.path.exists(report_folder):
                import urllib.parse
                for filename in os.listdir(report_folder):
                    # Only include user-facing reports, exclude JSON data files (used by interactive dashboard)
                    if filename.endswith(('.html', '.pdf', '.md')) and not filename.endswith('_sff_data.json'):
                        file_path = os.path.join(report_folder, filename).replace('\\', '/')
                        # Create download URL (URL encode the path)
                        encoded_path = urllib.parse.quote(file_path, safe='')
                        download_url = f"/api/download-report/{encoded_path}"
                        report_files.append({
                            'filename': filename,
                            'path': file_path,
                            'download_url': download_url
                        })
            
            return jsonify({
                'success': True,
                'state': state,
                'report_folder': report_folder,
                'report_files': report_files,
                'message': f'State report generated for {state}'
            })
        else:
            return jsonify({'error': 'Failed to generate state report'}), 500
            
    except Exception as e:
        import traceback
        return jsonify({
            'error': f'Error generating state report: {str(e)}',
            'details': traceback.format_exc() if app.debug else None
        }), 500

if __name__ == '__main__':
    print("Starting PBJ320 Dashboard & Report Generator...")
    print("Open your browser to: http://127.0.0.1:5001")
    # Disable auto-reloader to prevent file locking issues during Vercel deployment
    app.run(debug=True, port=5001, host='127.0.0.1', use_reloader=False)
