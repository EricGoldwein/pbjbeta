import pandas as pd
import os
import duckdb
import glob
from pathlib import Path
import re

def get_db_connection():
    # Remove the existing database file if it exists
    if os.path.exists('nursing_home_staffing.db'):
        os.remove('nursing_home_staffing.db')
    return duckdb.connect('nursing_home_staffing.db')

def standardize_column_names(df):
    """Standardize column names to match 2024Q3 format."""
    column_mapping = {
        # Basic Information
        'PROVNUM': ['PROVNUM', 'provnum', 'ProvNum'],
        'PROVNAME': ['PROVNAME', 'provname', 'ProvName'],
        'CITY': ['CITY', 'city', 'City'],
        'STATE': ['STATE', 'state', 'State'],
        'COUNTY_NAME': ['COUNTY_NAME', 'county_name', 'County_Name', 'CountyName'],
        'COUNTY_FIPS': ['COUNTY_FIPS', 'county_fips', 'County_Fips', 'CountyFips'],
        'CY_QTR': ['CY_Qtr', 'CY_QTR', 'cy_qtr', 'Quarter', 'QUARTER'],
        'WORKDATE': ['WorkDate', 'workdate', 'WORKDATE', 'Work_Date'],
        'MDSCENSUS': ['MDScensus', 'mdscensus', 'MDSCENSUS', 'MDS_CENSUS'],
        
        # RN Hours
        'HRS_RN': ['Hrs_RN', 'hrs_rn', 'HRS_RN', 'RN_Hours'],
        'HRS_RN_EMP': ['Hrs_RN_emp', 'hrs_rn_emp', 'HRS_RN_EMP', 'RN_Hours_Emp'],
        'HRS_RN_CTR': ['Hrs_RN_ctr', 'hrs_rn_ctr', 'HRS_RN_CTR', 'RN_Hours_Ctr'],
        'HRS_RNADMIN': ['Hrs_RNadmin', 'hrs_rnadmin', 'HRS_RNADMIN', 'RN_Admin_Hours'],
        'HRS_RNADMIN_EMP': ['Hrs_RNadmin_emp', 'hrs_rnadmin_emp', 'HRS_RNADMIN_EMP', 'RN_Admin_Hours_Emp'],
        'HRS_RNADMIN_CTR': ['Hrs_RNadmin_ctr', 'hrs_rnadmin_ctr', 'HRS_RNADMIN_CTR', 'RN_Admin_Hours_Ctr'],
        'HRS_RNDON': ['Hrs_RNDON', 'hrs_rndon', 'hrs_rn_donadmin', 'HRS_RNDON', 'RN_DON_Hours'],
        'HRS_RNDON_EMP': ['Hrs_RNDON_emp', 'hrs_rndon_emp', 'HRS_RNDON_EMP', 'RN_DON_Hours_Emp'],
        'HRS_RNDON_CTR': ['Hrs_RNDON_ctr', 'hrs_rndon_ctr', 'HRS_RNDON_CTR', 'RN_DON_Hours_Ctr'],
        
        # LPN Hours
        'HRS_LPN': ['Hrs_LPN', 'hrs_lpn', 'HRS_LPN', 'LPN_Hours'],
        'HRS_LPN_EMP': ['Hrs_LPN_emp', 'hrs_lpn_emp', 'HRS_LPN_EMP', 'LPN_Hours_Emp'],
        'HRS_LPN_CTR': ['Hrs_LPN_ctr', 'hrs_lpn_ctr', 'HRS_LPN_CTR', 'LPN_Hours_Ctr'],
        'HRS_LPNADMIN': ['Hrs_LPNadmin', 'hrs_lpnadmin', 'hrs_lpn_admin', 'HRS_LPNADMIN', 'LPN_Admin_Hours'],
        'HRS_LPNADMIN_EMP': ['Hrs_LPNadmin_emp', 'hrs_lpnadmin_emp', 'HRS_LPNADMIN_EMP', 'LPN_Admin_Hours_Emp'],
        'HRS_LPNADMIN_CTR': ['Hrs_LPNadmin_ctr', 'hrs_lpnadmin_ctr', 'HRS_LPNADMIN_CTR', 'LPN_Admin_Hours_Ctr'],
        
        # CNA/NA Hours
        'HRS_CNA': ['Hrs_CNA', 'hrs_cna', 'HRS_CNA', 'CNA_Hours'],
        'HRS_CNA_EMP': ['Hrs_CNA_emp', 'hrs_cna_emp', 'HRS_CNA_EMP', 'CNA_Hours_Emp'],
        'HRS_CNA_CTR': ['Hrs_CNA_ctr', 'hrs_cna_ctr', 'HRS_CNA_CTR', 'CNA_Hours_Ctr'],
        'HRS_NATRN': ['Hrs_NAtrn', 'hrs_natrn', 'hrs_na_trn', 'HRS_NATRN', 'NA_Trn_Hours'],
        'HRS_NATRN_EMP': ['Hrs_NAtrn_emp', 'hrs_natrn_emp', 'HRS_NATRN_EMP', 'NA_Trn_Hours_Emp'],
        'HRS_NATRN_CTR': ['Hrs_NAtrn_ctr', 'hrs_natrn_ctr', 'HRS_NATRN_CTR', 'NA_Trn_Hours_Ctr'],
        
        # Med Aide Hours
        'HRS_MEDAIDE': ['Hrs_MedAide', 'hrs_medaide', 'HRS_MEDAIDE', 'Med_Aide_Hours'],
        'HRS_MEDAIDE_EMP': ['Hrs_MedAide_emp', 'hrs_medaide_emp', 'HRS_MEDAIDE_EMP', 'Med_Aide_Hours_Emp'],
        'HRS_MEDAIDE_CTR': ['Hrs_MedAide_ctr', 'hrs_medaide_ctr', 'HRS_MEDAIDE_CTR', 'Med_Aide_Hours_Ctr'],
        
        # Other
        'INCOMPLETE': ['incomplete', 'INCOMPLETE', 'Incomplete']
    }
    
    # Create reverse mapping for easier lookup
    reverse_mapping = {}
    for standard_name, variations in column_mapping.items():
        for variation in variations:
            reverse_mapping[variation.lower()] = standard_name
    
    # Rename columns
    new_columns = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in reverse_mapping:
            new_columns[col] = reverse_mapping[col_lower]
        else:
            # If column not in mapping, convert to uppercase and replace spaces with underscores
            new_columns[col] = col.upper().replace(' ', '_')
    
    return df.rename(columns=new_columns)

def load_data(conn):
    try:
        cursor = conn.cursor()
        
        # Find all PBJ data files in PBJ_Nurse directory
        data_files = list(Path("PBJ_Nurse").glob("*.csv"))
        if not data_files:
            print("No PBJ data files found in PBJ_Nurse directory!")
            return False
        
        # Sort files to ensure consistent loading order
        data_files = sorted(data_files)
        print(f"Found {len(data_files)} PBJ data files to process")
        
        # Load the first file to create the table structure
        first_file = data_files[0]
        print(f"Creating table structure from {first_file}...")
        
        # Load first file with pandas, ensuring PROVNUM is treated as string
        df = pd.read_csv(first_file, encoding='latin1', dtype={'PROVNUM': str, 'provnum': str, 'ProvNum': str})
        
        # Clean the data - standardize column names and ensure no NaN in critical fields
        df = standardize_column_names(df)
        
        # Make sure PROVNUM is clean and consistent
        if 'PROVNUM' in df.columns:
            # Preserve case but ensure it's a string
            df['PROVNUM'] = df['PROVNUM'].astype(str)
            # Remove any leading/trailing whitespace
            df['PROVNUM'] = df['PROVNUM'].str.strip()
            
            # Count records before filtering
            records_before = len(df)
            
            # Remove records with empty PROVNUM
            df = df[df['PROVNUM'].notna() & (df['PROVNUM'] != '')]
            
            records_after = len(df)
            if records_before > records_after:
                print(f"Removed {records_before - records_after} records with missing or empty PROVNUM")
        
        # Drop the INCOMPLETE column if it exists, as it's not needed for metrics calculation
        if 'INCOMPLETE' in df.columns:
            df = df.drop(columns=['INCOMPLETE'])
        
        # Register the dataframe with DuckDB
        cursor.execute("CREATE TABLE staffing AS SELECT * FROM df")
        conn.commit()
        print(f"Table structure created from {first_file}")
        
        # Load the remaining files
        for i, data_path in enumerate(data_files[1:], 1):
            print(f"Loading file {i+1} of {len(data_files)}: {data_path}...")
            try:
                # Load with pandas, ensuring PROVNUM is treated as string
                df = pd.read_csv(data_path, encoding='latin1', dtype={'PROVNUM': str, 'provnum': str, 'ProvNum': str})
                df = standardize_column_names(df)
                
                # Data cleaning for PROVNUM
                if 'PROVNUM' in df.columns:
                    df['PROVNUM'] = df['PROVNUM'].astype(str).str.strip()
                    df = df[df['PROVNUM'].notna() & (df['PROVNUM'] != '')]
                
                # Drop the INCOMPLETE column if it exists
                if 'INCOMPLETE' in df.columns:
                    df = df.drop(columns=['INCOMPLETE'])
                
                # Register and insert
                cursor.execute("INSERT INTO staffing SELECT * FROM df")
                conn.commit()
                print(f"Data loaded from {data_path}")
            except Exception as e:
                print(f"Error loading {data_path}: {str(e)}")
        
        # Create an index on PROVNUM and CY_QTR to speed up facility-level queries
        print("Creating indexes to improve query performance...")
        cursor.execute("CREATE INDEX idx_provnum_qtr ON staffing (PROVNUM, CY_QTR)")
        cursor.execute("CREATE INDEX idx_state_qtr ON staffing (STATE, CY_QTR)")
        conn.commit()
        
        return True
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        return False

def calculate_national_metrics(conn, quarter):
    """Calculate national metrics for a quarter"""
    query = f"""
    WITH daily_metrics AS (
        SELECT 
            PROVNUM,
            WORKDATE,
            MDSCENSUS,
            (HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN + HRS_CNA + HRS_NATRN + HRS_MEDAIDE) as total_hours,
            (HRS_RNDON + HRS_RNADMIN + HRS_RN) as rn_hours,
            (HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN) as nurse_care_hours,
            (HRS_RN) as rn_care_hours,
            (HRS_CNA + HRS_NATRN + HRS_MEDAIDE) as nurse_assistant_hours,
            (HRS_RNDON_CTR + HRS_RNADMIN_CTR + HRS_RN_CTR + HRS_LPNADMIN_CTR + HRS_LPN_CTR + HRS_CNA_CTR + HRS_NATRN_CTR + HRS_MEDAIDE_CTR) as contract_hours,
            (HRS_RNADMIN) as rn_admin_hours,
            (HRS_RNDON) as rn_don_hours,
            (HRS_LPN) as lpn_hours,
            (HRS_LPNADMIN) as lpn_admin_hours,
            (HRS_CNA) as cna_hours,
            (HRS_NATRN) as natr_hours,
            (HRS_MEDAIDE) as medaide_hours
        FROM staffing 
        WHERE CY_QTR = '{quarter}'
    )
    SELECT
        COUNT(DISTINCT PROVNUM) as Facility_Count,
        SUM(MDSCENSUS) as Total_Resident_Days,
        ROUND(SUM(total_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Total_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (total_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_Total_HPRD,
        ROUND(SUM(rn_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_RN_HPRD,
        ROUND(SUM(nurse_care_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Nurse_Care_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (nurse_care_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_Nurse_Care_HPRD,
        ROUND(SUM(rn_care_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_Care_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_care_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_RN_Care_HPRD,
        ROUND(SUM(nurse_assistant_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Nurse_Assistant_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (nurse_assistant_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_Nurse_Assistant_HPRD,
        ROUND(SUM(contract_hours) / NULLIF(SUM(total_hours), 0) * 100, 3) as Contract_Staff_Percentage,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (contract_hours / NULLIF(total_hours, 0) * 100)), 3) as Median_Contract_Staff_Percentage,
        ROUND(SUM(rn_admin_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_Admin_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_admin_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_RN_Admin_HPRD,
        ROUND(SUM(rn_don_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_DON_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_don_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_RN_DON_HPRD,
        ROUND(SUM(lpn_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as LPN_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (lpn_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_LPN_HPRD,
        ROUND(SUM(lpn_admin_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as LPN_Admin_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (lpn_admin_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_LPN_Admin_HPRD,
        ROUND(SUM(cna_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as CNA_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (cna_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_CNA_HPRD,
        ROUND(SUM(natr_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as NAtr_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (natr_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_NAtr_HPRD,
        ROUND(SUM(medaide_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as MedAide_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (medaide_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_MedAide_HPRD
    FROM daily_metrics
    """
    try:
        result = conn.execute(query).fetchone()
        return {
            'Facility_Count': int(result[0]),
            'Total_Resident_Days': float(result[1]),
            'Total_HPRD': float(result[2]),
            'Median_Total_HPRD': float(result[3]),
            'RN_HPRD': float(result[4]),
            'Median_RN_HPRD': float(result[5]),
            'Nurse_Care_HPRD': float(result[6]),
            'Median_Nurse_Care_HPRD': float(result[7]),
            'RN_Care_HPRD': float(result[8]),
            'Median_RN_Care_HPRD': float(result[9]),
            'Nurse_Assistant_HPRD': float(result[10]),
            'Median_Nurse_Assistant_HPRD': float(result[11]),
            'Contract_Staff_Percentage': float(result[12]),
            'Median_Contract_Staff_Percentage': float(result[13]),
            'RN_Admin_HPRD': float(result[14]),
            'Median_RN_Admin_HPRD': float(result[15]),
            'RN_DON_HPRD': float(result[16]),
            'Median_RN_DON_HPRD': float(result[17]),
            'LPN_HPRD': float(result[18]),
            'Median_LPN_HPRD': float(result[19]),
            'LPN_Admin_HPRD': float(result[20]),
            'Median_LPN_Admin_HPRD': float(result[21]),
            'CNA_HPRD': float(result[22]),
            'Median_CNA_HPRD': float(result[23]),
            'NAtr_HPRD': float(result[24]),
            'Median_NAtr_HPRD': float(result[25]),
            'MedAide_HPRD': float(result[26]),
            'Median_MedAide_HPRD': float(result[27])
        }
    except Exception as e:
        print(f"Error calculating metrics for quarter {quarter}: {str(e)}")
        return None

def calculate_state_metrics(conn, state, quarter):
    """Calculate state metrics for a quarter"""
    query = f"""
    WITH daily_metrics AS (
        SELECT 
            PROVNUM,
            WORKDATE,
            MDSCENSUS,
            COALESCE(HRS_RNDON, 0) + COALESCE(HRS_RNADMIN, 0) + COALESCE(HRS_RN, 0) + 
            COALESCE(HRS_LPNADMIN, 0) + COALESCE(HRS_LPN, 0) + COALESCE(HRS_CNA, 0) + 
            COALESCE(HRS_NATRN, 0) + COALESCE(HRS_MEDAIDE, 0) as total_hours,
            
            COALESCE(HRS_RNDON, 0) + COALESCE(HRS_RNADMIN, 0) + COALESCE(HRS_RN, 0) as rn_hours,
            
            COALESCE(HRS_RNDON, 0) + COALESCE(HRS_RNADMIN, 0) + COALESCE(HRS_RN, 0) + 
            COALESCE(HRS_LPNADMIN, 0) + COALESCE(HRS_LPN, 0) as nurse_care_hours,
            
            COALESCE(HRS_RN, 0) as rn_care_hours,
            
            COALESCE(HRS_CNA, 0) + COALESCE(HRS_NATRN, 0) + COALESCE(HRS_MEDAIDE, 0) as nurse_assistant_hours,
            
            COALESCE(HRS_RNDON_CTR, 0) + COALESCE(HRS_RNADMIN_CTR, 0) + COALESCE(HRS_RN_CTR, 0) + 
            COALESCE(HRS_LPNADMIN_CTR, 0) + COALESCE(HRS_LPN_CTR, 0) + COALESCE(HRS_CNA_CTR, 0) + 
            COALESCE(HRS_NATRN_CTR, 0) + COALESCE(HRS_MEDAIDE_CTR, 0) as contract_hours,
            
            COALESCE(HRS_RNADMIN, 0) as rn_admin_hours,
            COALESCE(HRS_RNDON, 0) as rn_don_hours,
            COALESCE(HRS_LPN, 0) as lpn_hours,
            COALESCE(HRS_LPNADMIN, 0) as lpn_admin_hours,
            COALESCE(HRS_CNA, 0) as cna_hours,
            COALESCE(HRS_NATRN, 0) as natr_hours,
            COALESCE(HRS_MEDAIDE, 0) as medaide_hours
        FROM staffing 
        WHERE STATE = '{state}' AND CY_QTR = '{quarter}'
        AND MDSCENSUS > 0  -- Ensure we only count days with residents
    )
    SELECT
        COUNT(DISTINCT PROVNUM) as Facility_Count,
        SUM(MDSCENSUS) as Total_Resident_Days,
        ROUND(SUM(total_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Total_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (total_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_Total_HPRD,
        ROUND(SUM(rn_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_RN_HPRD,
        ROUND(SUM(nurse_care_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Nurse_Care_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (nurse_care_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_Nurse_Care_HPRD,
        ROUND(SUM(rn_care_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_Care_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_care_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_RN_Care_HPRD,
        ROUND(SUM(nurse_assistant_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Nurse_Assistant_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (nurse_assistant_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_Nurse_Assistant_HPRD,
        ROUND(SUM(contract_hours) / NULLIF(SUM(total_hours), 0) * 100, 3) as Contract_Staff_Percentage,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (contract_hours / NULLIF(total_hours, 0) * 100)), 3) as Median_Contract_Staff_Percentage,
        ROUND(SUM(rn_admin_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_Admin_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_admin_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_RN_Admin_HPRD,
        ROUND(SUM(rn_don_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_DON_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_don_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_RN_DON_HPRD,
        ROUND(SUM(lpn_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as LPN_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (lpn_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_LPN_HPRD,
        ROUND(SUM(lpn_admin_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as LPN_Admin_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (lpn_admin_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_LPN_Admin_HPRD,
        ROUND(SUM(cna_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as CNA_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (cna_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_CNA_HPRD,
        ROUND(SUM(natr_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as NAtr_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (natr_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_NAtr_HPRD,
        ROUND(SUM(medaide_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as MedAide_HPRD,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (medaide_hours / NULLIF(MDSCENSUS, 0))), 3) as Median_MedAide_HPRD
    FROM daily_metrics
    """
    try:
        result = conn.execute(query).fetchone()
        return {
            'STATE': state,
            'Facility_Count': int(result[0]),
            'Total_Resident_Days': float(result[1]),
            'Total_HPRD': float(result[2]),
            'Median_Total_HPRD': float(result[3]),
            'RN_HPRD': float(result[4]),
            'Median_RN_HPRD': float(result[5]),
            'Nurse_Care_HPRD': float(result[6]),
            'Median_Nurse_Care_HPRD': float(result[7]),
            'RN_Care_HPRD': float(result[8]),
            'Median_RN_Care_HPRD': float(result[9]),
            'Nurse_Assistant_HPRD': float(result[10]),
            'Median_Nurse_Assistant_HPRD': float(result[11]),
            'Contract_Staff_Percentage': float(result[12]),
            'Median_Contract_Staff_Percentage': float(result[13]),
            'RN_Admin_HPRD': float(result[14]),
            'Median_RN_Admin_HPRD': float(result[15]),
            'RN_DON_HPRD': float(result[16]),
            'Median_RN_DON_HPRD': float(result[17]),
            'LPN_HPRD': float(result[18]),
            'Median_LPN_HPRD': float(result[19]),
            'LPN_Admin_HPRD': float(result[20]),
            'Median_LPN_Admin_HPRD': float(result[21]),
            'CNA_HPRD': float(result[22]),
            'Median_CNA_HPRD': float(result[23]),
            'NAtr_HPRD': float(result[24]),
            'Median_NAtr_HPRD': float(result[25]),
            'MedAide_HPRD': float(result[26]),
            'Median_MedAide_HPRD': float(result[27])
        }
    except Exception as e:
        print(f"Error calculating metrics for state {state} and quarter {quarter}: {str(e)}")
        return None

def calculate_median_contract_percentage(conn, quarter, state=None):
    """Calculate median contract staff percentage for a quarter (and optionally state)"""
    state_filter = f"AND STATE = '{state}'" if state else ""
    query = f"""
    WITH facility_contract_pct AS (
        SELECT 
            PROVNUM,
            SUM(HRS_RNDON_CTR + HRS_RNADMIN_CTR + HRS_RN_CTR + HRS_LPNADMIN_CTR + HRS_LPN_CTR + HRS_CNA_CTR + HRS_NATRN_CTR + HRS_MEDAIDE_CTR) * 100.0 /
            NULLIF(SUM(HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN + HRS_CNA + HRS_NATRN + HRS_MEDAIDE), 0) as contract_pct
        FROM staffing 
        WHERE CY_QTR = '{quarter}' {state_filter}
        GROUP BY PROVNUM
        HAVING SUM(HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN + HRS_CNA + HRS_NATRN + HRS_MEDAIDE) > 0
    )
    SELECT 
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY contract_pct) as median_contract_pct
    FROM facility_contract_pct
    """
    try:
        result = conn.execute(query).fetchone()
        return float(result[0]) if result and result[0] is not None else 0.0
    except Exception as e:
        print(f"Error calculating median contract percentage for {'state ' + state if state else 'national'} quarter {quarter}: {str(e)}")
        return 0.0

def calculate_regional_metrics(conn, region, quarter):
    """Calculate regional metrics for a quarter"""
    try:
        # Get state-region mapping
        state_region_df = pd.read_csv('state_region_mapping.csv')
        states_in_region = state_region_df[state_region_df['Region'] == region]['STATE'].tolist()
        
        if not states_in_region:
            print(f"No states found for {region}")
            return None
            
        # First check if we have any data for this region/quarter
        check_query = f"""
        SELECT COUNT(DISTINCT STATE) as state_count, COUNT(DISTINCT PROVNUM) as facility_count
        FROM staffing 
        WHERE STATE IN ({','.join([f"'{state}'" for state in states_in_region])}) AND CY_QTR = '{quarter}'
        """
        check_result = conn.execute(check_query).fetchone()
        if check_result[1] == 0:  # No facilities found
            print(f"No data found for {region} in {quarter}")
            return None
            
        query = f"""
        WITH daily_metrics AS (
            SELECT 
                PROVNUM,
                STATE,
                WORKDATE,
                MDSCENSUS,
                COALESCE(HRS_RNDON, 0) + COALESCE(HRS_RNADMIN, 0) + COALESCE(HRS_RN, 0) + 
                COALESCE(HRS_LPNADMIN, 0) + COALESCE(HRS_LPN, 0) + COALESCE(HRS_CNA, 0) + 
                COALESCE(HRS_NATRN, 0) + COALESCE(HRS_MEDAIDE, 0) as total_hours,
                
                COALESCE(HRS_RNDON, 0) + COALESCE(HRS_RNADMIN, 0) + COALESCE(HRS_RN, 0) as rn_hours,
                
                COALESCE(HRS_RNDON, 0) + COALESCE(HRS_RNADMIN, 0) + COALESCE(HRS_RN, 0) + 
                COALESCE(HRS_LPNADMIN, 0) + COALESCE(HRS_LPN, 0) as nurse_care_hours,
                
                COALESCE(HRS_RN, 0) as rn_care_hours,
                
                COALESCE(HRS_CNA, 0) + COALESCE(HRS_NATRN, 0) + COALESCE(HRS_MEDAIDE, 0) as nurse_assistant_hours,
                
                COALESCE(HRS_RNDON_CTR, 0) + COALESCE(HRS_RNADMIN_CTR, 0) + COALESCE(HRS_RN_CTR, 0) + 
                COALESCE(HRS_LPNADMIN_CTR, 0) + COALESCE(HRS_LPN_CTR, 0) + COALESCE(HRS_CNA_CTR, 0) + 
                COALESCE(HRS_NATRN_CTR, 0) + COALESCE(HRS_MEDAIDE_CTR, 0) as contract_hours
            FROM staffing 
            WHERE STATE IN ({','.join([f"'{state}'" for state in states_in_region])}) 
            AND CY_QTR = '{quarter}'
            AND MDSCENSUS > 0  -- Ensure we only count days with residents
        )
        SELECT
            '{region}' as Region,
            COUNT(DISTINCT PROVNUM) as Facility_Count,
            COUNT(DISTINCT STATE) as State_Count,
            SUM(MDSCENSUS) as Total_Resident_Days,
            ROUND(SUM(total_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Total_HPRD,
            ROUND(SUM(rn_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_HPRD,
            ROUND(SUM(nurse_care_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Nurse_Care_HPRD,
            ROUND(SUM(rn_care_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as RN_Care_HPRD,
            ROUND(SUM(nurse_assistant_hours) / NULLIF(SUM(MDSCENSUS), 0), 3) as Nurse_Assistant_HPRD,
            ROUND(SUM(contract_hours) / NULLIF(SUM(total_hours), 0) * 100, 3) as Contract_Staff_Percentage,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (contract_hours / NULLIF(total_hours, 0) * 100)), 3) as Median_Contract_Percentage
        FROM daily_metrics
        GROUP BY Region
        """
        
        result = conn.execute(query).fetchone()
        if result is None:
            print(f"No results returned for {region} in {quarter}")
            return None
            
        metrics = {
            'Region': result[0],
            'Facility_Count': int(result[1]),
            'State_Count': int(result[2]),
            'Total_Resident_Days': float(result[3]),
            'Total_HPRD': float(result[4]),
            'RN_HPRD': float(result[5]),
            'Nurse_Care_HPRD': float(result[6]),
            'RN_Care_HPRD': float(result[7]),
            'Nurse_Assistant_HPRD': float(result[8]),
            'Contract_Staff_Percentage': float(result[9]),
            'Median_Contract_Percentage': float(result[10])
        }
        
        # Add validation checks
        if metrics['State_Count'] == 0:
            print(f"Warning: No states with data found for {region} in {quarter}")
        if metrics['Facility_Count'] == 0:
            print(f"Warning: No facilities with data found for {region} in {quarter}")
        if metrics['Total_Resident_Days'] == 0:
            print(f"Warning: No resident days found for {region} in {quarter}")
            
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics for {region} in {quarter}: {str(e)}")
        return None

def calculate_facility_metrics_vectorized(conn, quarters):
    """Calculate facility metrics for all facilities and quarters in a single query"""
    query = f"""
    WITH facility_latest_names AS (
        SELECT 
            PROVNUM,
            PROVNAME,
            STATE,
            COUNTY_NAME,
            CITY,
            ROW_NUMBER() OVER (PARTITION BY PROVNUM ORDER BY CY_QTR DESC, WORKDATE DESC) as rn
        FROM staffing
        WHERE PROVNAME IS NOT NULL
    ),
    daily_metrics AS (
        SELECT 
            PROVNUM,
            CY_QTR,
            WORKDATE,
            MDSCENSUS,
            COALESCE(HRS_RNDON, 0) + COALESCE(HRS_RNADMIN, 0) + COALESCE(HRS_RN, 0) + 
            COALESCE(HRS_LPNADMIN, 0) + COALESCE(HRS_LPN, 0) + COALESCE(HRS_CNA, 0) + 
            COALESCE(HRS_NATRN, 0) + COALESCE(HRS_MEDAIDE, 0) as total_hours,
            
            COALESCE(HRS_RNDON, 0) + COALESCE(HRS_RNADMIN, 0) + COALESCE(HRS_RN, 0) as rn_hours,
            
            COALESCE(HRS_RNDON, 0) + COALESCE(HRS_RNADMIN, 0) + COALESCE(HRS_RN, 0) + 
            COALESCE(HRS_LPNADMIN, 0) + COALESCE(HRS_LPN, 0) as nurse_care_hours,
            
            COALESCE(HRS_RN, 0) as rn_care_hours,
            
            COALESCE(HRS_CNA, 0) + COALESCE(HRS_NATRN, 0) + COALESCE(HRS_MEDAIDE, 0) as nurse_assistant_hours,
            
            COALESCE(HRS_RNDON_CTR, 0) + COALESCE(HRS_RNADMIN_CTR, 0) + COALESCE(HRS_RN_CTR, 0) + 
            COALESCE(HRS_LPNADMIN_CTR, 0) + COALESCE(HRS_LPN_CTR, 0) + COALESCE(HRS_CNA_CTR, 0) + 
            COALESCE(HRS_NATRN_CTR, 0) + COALESCE(HRS_MEDAIDE_CTR, 0) as contract_hours,
            
            COALESCE(HRS_RNADMIN, 0) as rn_admin_hours,
            COALESCE(HRS_RNDON, 0) as rn_don_hours,
            COALESCE(HRS_LPN, 0) as lpn_hours,
            COALESCE(HRS_LPNADMIN, 0) as lpn_admin_hours,
            COALESCE(HRS_CNA, 0) as cna_hours,
            COALESCE(HRS_NATRN, 0) as natr_hours,
            COALESCE(HRS_MEDAIDE, 0) as medaide_hours
        FROM staffing 
        WHERE CY_QTR IN ({','.join([f"'{q}'" for q in quarters])})
        AND MDSCENSUS > 0  -- Ensure we only count days with residents
    ),
    facility_metrics AS (
        SELECT
            dm.PROVNUM,
            fln.PROVNAME,
            fln.STATE,
            fln.COUNTY_NAME,
            fln.CITY,
            dm.CY_QTR,
            COUNT(DISTINCT dm.WORKDATE) as Workdays_Count,
            SUM(dm.MDSCENSUS) as Total_Resident_Days,
            ROUND(SUM(dm.total_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as Total_HPRD,
            ROUND(SUM(dm.rn_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as RN_HPRD,
            ROUND(SUM(dm.nurse_care_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as Nurse_Care_HPRD,
            ROUND(SUM(dm.rn_care_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as RN_Care_HPRD,
            ROUND(SUM(dm.nurse_assistant_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as Nurse_Assistant_HPRD,
            ROUND(SUM(dm.contract_hours) / NULLIF(SUM(dm.total_hours), 0) * 100, 3) as Contract_Staff_Percentage,
            ROUND(SUM(dm.rn_admin_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as RN_Admin_HPRD,
            ROUND(SUM(dm.rn_don_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as RN_DON_HPRD,
            ROUND(SUM(dm.lpn_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as LPN_HPRD,
            ROUND(SUM(dm.lpn_admin_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as LPN_Admin_HPRD,
            ROUND(SUM(dm.cna_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as CNA_HPRD,
            ROUND(SUM(dm.natr_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as NAtr_HPRD,
            ROUND(SUM(dm.medaide_hours) / NULLIF(SUM(dm.MDSCENSUS), 0), 3) as MedAide_HPRD
        FROM daily_metrics dm
        LEFT JOIN facility_latest_names fln ON dm.PROVNUM = fln.PROVNUM AND fln.rn = 1
        GROUP BY dm.PROVNUM, fln.PROVNAME, fln.STATE, fln.COUNTY_NAME, fln.CITY, dm.CY_QTR
    )
    SELECT * FROM facility_metrics
    """
    try:
        result = conn.execute(query).fetchall()
        if not result:
            return []
            
        return [{
            'PROVNUM': row[0],
            'PROVNAME': row[1],
            'STATE': row[2],
            'COUNTY_NAME': row[3],
            'CITY': row[4],
            'CY_QTR': row[5],
            'Workdays_Count': int(row[6]),
            'Total_Resident_Days': float(row[7]),
            'Total_HPRD': float(row[8]),
            'RN_HPRD': float(row[9]),
            'Nurse_Care_HPRD': float(row[10]),
            'RN_Care_HPRD': float(row[11]),
            'Nurse_Assistant_HPRD': float(row[12]),
            'Contract_Staff_Percentage': float(row[13]),
            'RN_Admin_HPRD': float(row[14]),
            'RN_DON_HPRD': float(row[15]),
            'LPN_HPRD': float(row[16]),
            'LPN_Admin_HPRD': float(row[17]),
            'CNA_HPRD': float(row[18]),
            'NAtr_HPRD': float(row[19]),
            'MedAide_HPRD': float(row[20])
        } for row in result]
    except Exception as e:
        print(f"Error calculating facility metrics: {str(e)}")
        return []

def main(output_dir='.', verbose=False):
    """Main function to generate historical metrics"""
    print("Starting historical metrics generation...")
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a single database connection for the entire process
    conn = get_db_connection()
    try:
        # Load all quarterly data
        if not load_data(conn):
            print("Failed to load data. Exiting.")
            return

        cursor = conn.cursor()
        
        # Get unique quarters
        cursor.execute("""
            SELECT DISTINCT CY_QTR
            FROM staffing 
            ORDER BY CY_QTR
        """)
        quarters = [row[0] for row in cursor.fetchall()]

        if not quarters:
            print("No quarters found in the database. Please check if data was loaded correctly.")
            return

        print(f"\nFound {len(quarters)} quarters: {', '.join(quarters)}")
        
        # Calculate national metrics
        national_metrics = []
        for quarter in quarters:
            if verbose:
                print(f"\nProcessing national metrics for quarter: {quarter}")
            metrics = calculate_national_metrics(conn, quarter)
            if metrics:
                metrics['Median_Contract_Percentage'] = calculate_median_contract_percentage(conn, quarter)
                metrics['CY_Qtr'] = quarter
                national_metrics.append(metrics)
                if verbose:
                    print(f"Successfully calculated national metrics for {quarter}")
            else:
                print(f"No national metrics calculated for {quarter}")

        # Calculate state metrics
        cursor.execute("""
            SELECT DISTINCT STATE
            FROM staffing 
            ORDER BY STATE
        """)
        states = [row[0] for row in cursor.fetchall()]
        
        state_metrics = []
        for state in states:
            for quarter in quarters:
                if verbose:
                    print(f"\nProcessing state metrics for {state} quarter {quarter}")
                metrics = calculate_state_metrics(conn, state, quarter)
                if metrics:
                    metrics['CY_Qtr'] = quarter
                    state_metrics.append(metrics)
                    if verbose:
                        print(f"Successfully calculated metrics for {state} {quarter}")
                else:
                    print(f"No metrics calculated for {state} {quarter}")

        # Calculate regional metrics
        state_region_df = pd.read_csv('state_region_mapping.csv')
        regions = state_region_df['Region'].unique().tolist()
        
        regional_metrics = []
        for region in regions:
            for quarter in quarters:
                if verbose:
                    print(f"\nProcessing regional metrics for {region} quarter {quarter}")
                metrics = calculate_regional_metrics(conn, region, quarter)
                if metrics:
                    metrics['CY_Qtr'] = quarter
                    regional_metrics.append(metrics)
                    if verbose:
                        print(f"Successfully calculated metrics for {region} {quarter}")
                else:
                    print(f"No metrics calculated for {region} {quarter}")

        # Calculate facility metrics in batches
        batch_size = 500  # Process facilities in batches of 500
        total_quarters = len(quarters)
        facility_metrics = []
        
        print(f"\nCalculating facility metrics across {total_quarters} quarters...")
        print(f"Processing in batches of {batch_size} facilities")
        
        # Process quarters in batches
        for i in range(0, total_quarters, batch_size):
            batch_quarters = quarters[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"\nProcessing batch {batch_num} (quarters {i+1}-{min(i+batch_size, total_quarters)} of {total_quarters})")
            
            # Calculate metrics for this batch of quarters
            batch_results = calculate_facility_metrics_vectorized(conn, batch_quarters)
            
            if batch_results:
                # Save batch results
                batch_df = pd.DataFrame(batch_results)
                batch_file = output_path / f'facility_quarterly_metrics_batch_{batch_num}.csv'
                batch_df.to_csv(batch_file, index=False)
                print(f"Saved batch {batch_num} results to {batch_file}")
                
                # Add to master list
                facility_metrics.extend(batch_results)
            
            # Print progress
            progress = min(i+batch_size, total_quarters) / total_quarters * 100
            print(f"Progress: {progress:.1f}%")

        # Save results
        if national_metrics:
            df = pd.DataFrame(national_metrics)
            df.to_csv(output_path / 'national_quarterly_metrics.csv', index=False)
            print("\nnational_quarterly_metrics.csv generated successfully.")
            print(f"Generated national metrics for {len(quarters)} quarters.")
            if verbose:
                print("\nFirst few rows of the generated metrics:")
                print(df.head().to_string())

        if state_metrics:
            df = pd.DataFrame(state_metrics)
            df.to_csv(output_path / 'state_quarterly_metrics.csv', index=False)
            print("\nstate_quarterly_metrics.csv generated successfully.")
            print(f"Generated state metrics for {len(state_metrics)} state/quarter combinations.")
            if verbose:
                print("\nFirst few rows of the generated metrics:")
                print(df.head().to_string())

        if regional_metrics:
            df = pd.DataFrame(regional_metrics)
            df.to_csv(output_path / 'region_quarterly_metrics.csv', index=False)
            print("\nregion_quarterly_metrics.csv generated successfully.")
            print(f"Generated regional metrics for {len(regional_metrics)} region/quarter combinations.")
            if verbose:
                print("\nFirst few rows of the generated metrics:")
                print(df.head().to_string())

        if facility_metrics:
            df = pd.DataFrame(facility_metrics)
            # Sort by PROVNUM and CY_QTR for better performance
            df = df.sort_values(['PROVNUM', 'CY_QTR'])
            df.to_csv(output_path / 'facility_quarterly_metrics.csv', index=False)
            print("\nfacility_quarterly_metrics.csv generated successfully.")
            print(f"Generated facility metrics for {len(facility_metrics)} facility/quarter combinations.")
            if verbose:
                print("\nFirst few rows of the generated metrics:")
                print(df.head().to_string())

    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate historical metrics for nursing home staffing data')
    parser.add_argument('--output-dir', type=str, default='.', help='Output directory for generated files')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    main(output_dir=args.output_dir, verbose=args.verbose) 