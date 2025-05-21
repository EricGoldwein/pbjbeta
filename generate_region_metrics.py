import pandas as pd
import os
import duckdb
import glob
import traceback

def get_db_connection():
    # Remove the existing database file if it exists
    if os.path.exists('nursing_home_staffing.db'):
        os.remove('nursing_home_staffing.db')
    return duckdb.connect('nursing_home_staffing.db')

def standardize_column_names(df):
    """Standardize column names to match expected format"""
    rename_map = {
        'provnum': 'PROVNUM',
        'provider_number': 'PROVNUM',
        'provider number': 'PROVNUM',
        'federal_provider_number': 'PROVNUM',
        'federal provider number': 'PROVNUM',
        'state': 'STATE',
        'workdate': 'WorkDate',
        'work_date': 'WorkDate',
        'work date': 'WorkDate',
        'mdscensus': 'MDScensus',
        'mds_census': 'MDScensus',
        'mds census': 'MDScensus',
        'cy_qtr': 'CY_Qtr',
        'quarter': 'CY_Qtr'
    }
    
    # Convert column names to lowercase for case-insensitive matching
    df.columns = df.columns.str.lower()
    
    # Apply renaming
    df = df.rename(columns=rename_map)
    
    # Drop the 'incomplete' column if it exists
    if 'incomplete' in df.columns:
        df = df.drop(columns=['incomplete'])
    
    return df

def load_data(conn):
    """Load PBJ data files into DuckDB."""
    try:
        cursor = conn.cursor()
        
        # Find all PBJ data files
        data_files = glob.glob('PBJ_Nurse/*.csv')
        
        if not data_files:
            print("No PBJ data files found!")
            return False
            
        print(f"Found {len(data_files)} data files")
        print("Files found:")
        for f in sorted(data_files):
            print(f"  - {f}")
        
        # Load the first file to create the table structure
        first_file = data_files[0]
        print(f"\nCreating table structure from {first_file}...")
        
        # Try different encodings for first file
        encodings = ['utf-8', 'latin1', 'cp1252']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(first_file, encoding=encoding, dtype={'PROVNUM': str}, low_memory=False)
                print(f"Successfully loaded {first_file} with {encoding} encoding")
                print(f"Columns in file: {df.columns.tolist()}")
                print(f"Unique quarters in file: {df['CY_Qtr'].unique() if 'CY_Qtr' in df.columns else 'CY_Qtr column not found'}")
                print(f"Number of rows: {len(df)}")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print(f"Could not read {first_file} with any of the attempted encodings")
            return False
            
        df = standardize_column_names(df)
        print(f"Loaded first file with {len(df)} rows")
        
        # Register the dataframe with DuckDB
        cursor.execute("CREATE TABLE staffing AS SELECT * FROM df")
        conn.commit()
        print(f"Table structure created from {first_file}")
        
        # Load the remaining files
        for data_path in sorted(data_files[1:]):
            print(f"\nLoading {data_path}...")
            success = False
            
            # Try different encodings for each file
            for encoding in encodings:
                try:
                    df = pd.read_csv(data_path, encoding=encoding, dtype={'PROVNUM': str}, low_memory=False)
                    df = standardize_column_names(df)
                    print(f"Successfully loaded {data_path} with {encoding} encoding")
                    print(f"Columns in file: {df.columns.tolist()}")
                    print(f"Unique quarters in file: {df['CY_Qtr'].unique() if 'CY_Qtr' in df.columns else 'CY_Qtr column not found'}")
                    print(f"Number of rows: {len(df)}")
                    
                    # Register and insert
                    cursor.execute("INSERT INTO staffing SELECT * FROM df")
                    conn.commit()
                    print(f"Data loaded from {data_path}")
                    success = True
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error loading {data_path} with {encoding} encoding: {str(e)}")
                    print(traceback.format_exc())
                    break
            
            if not success:
                print(f"Failed to load {data_path} with any encoding")
                continue
        
        # Verify all quarters are loaded
        cursor.execute("""
            SELECT DISTINCT CY_Qtr, COUNT(*) as record_count
            FROM staffing 
            GROUP BY CY_Qtr
            ORDER BY CY_Qtr
        """)
        quarter_data = cursor.fetchall()
        print("\nQuarters in database after loading all files:")
        for quarter, count in quarter_data:
            print(f"  - {quarter}: {count:,} records")
        
        return True
        
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        print(traceback.format_exc())
        return False

def calculate_regional_metrics(conn, region, quarter):
    """Calculate regional metrics for a quarter"""
    try:
        # Get states in this region
        states_in_region = get_states_in_region(region)
        states_list = "', '".join(states_in_region)
        
        print(f"Calculating metrics for {region} ({states_list}) in {quarter}")
        
        query = f"""
        WITH daily_metrics AS (
            SELECT 
                PROVNUM,
                WorkDate,
                MDScensus,
                (Hrs_RNDON + Hrs_RNadmin + Hrs_RN + Hrs_LPNadmin + Hrs_LPN + Hrs_CNA + Hrs_NAtrn + Hrs_MedAide) as total_hours,
                (Hrs_RNDON + Hrs_RNadmin + Hrs_RN) as rn_hours,
                (Hrs_RNDON + Hrs_RNadmin + Hrs_RN + Hrs_LPNadmin + Hrs_LPN) as nurse_care_hours,
                (Hrs_RN) as rn_care_hours,
                (Hrs_CNA + Hrs_NAtrn + Hrs_MedAide) as nurse_assistant_hours,
                (Hrs_RNDON_ctr + Hrs_RNadmin_ctr + Hrs_RN_ctr + Hrs_LPNadmin_ctr + Hrs_LPN_ctr + Hrs_CNA_ctr + Hrs_NAtrn_ctr + Hrs_MedAide_ctr) as contract_hours,
                (Hrs_RNadmin) as rn_admin_hours,
                (Hrs_RNDON) as rn_don_hours,
                (Hrs_LPN) as lpn_hours,
                (Hrs_LPNadmin) as lpn_admin_hours,
                (Hrs_CNA) as cna_hours,
                (Hrs_NAtrn) as natr_hours,
                (Hrs_MedAide) as medaide_hours
            FROM staffing 
            WHERE STATE IN ('{states_list}') AND CY_Qtr = '{quarter}'
        )
        SELECT
            COUNT(DISTINCT PROVNUM) as Facility_Count,
            SUM(MDScensus) as Total_Resident_Days,
            ROUND(SUM(total_hours) / NULLIF(SUM(MDScensus), 0), 3) as Total_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (total_hours / NULLIF(MDScensus, 0))), 3) as Median_Total_HPRD,
            ROUND(SUM(rn_hours) / NULLIF(SUM(MDScensus), 0), 3) as RN_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_hours / NULLIF(MDScensus, 0))), 3) as Median_RN_HPRD,
            ROUND(SUM(nurse_care_hours) / NULLIF(SUM(MDScensus), 0), 3) as Nurse_Care_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (nurse_care_hours / NULLIF(MDScensus, 0))), 3) as Median_Nurse_Care_HPRD,
            ROUND(SUM(rn_care_hours) / NULLIF(SUM(MDScensus), 0), 3) as RN_Care_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_care_hours / NULLIF(MDScensus, 0))), 3) as Median_RN_Care_HPRD,
            ROUND(SUM(nurse_assistant_hours) / NULLIF(SUM(MDScensus), 0), 3) as Nurse_Assistant_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (nurse_assistant_hours / NULLIF(MDScensus, 0))), 3) as Median_Nurse_Assistant_HPRD,
            ROUND(SUM(contract_hours) / NULLIF(SUM(total_hours), 0) * 100, 3) as Contract_Staff_Percentage,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (contract_hours / NULLIF(total_hours, 0) * 100)), 3) as Median_Contract_Staff_Percentage,
            ROUND(SUM(rn_admin_hours) / NULLIF(SUM(MDScensus), 0), 3) as RN_Admin_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_admin_hours / NULLIF(MDScensus, 0))), 3) as Median_RN_Admin_HPRD,
            ROUND(SUM(rn_don_hours) / NULLIF(SUM(MDScensus), 0), 3) as RN_DON_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (rn_don_hours / NULLIF(MDScensus, 0))), 3) as Median_RN_DON_HPRD,
            ROUND(SUM(lpn_hours) / NULLIF(SUM(MDScensus), 0), 3) as LPN_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (lpn_hours / NULLIF(MDScensus, 0))), 3) as Median_LPN_HPRD,
            ROUND(SUM(lpn_admin_hours) / NULLIF(SUM(MDScensus), 0), 3) as LPN_Admin_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (lpn_admin_hours / NULLIF(MDScensus, 0))), 3) as Median_LPN_Admin_HPRD,
            ROUND(SUM(cna_hours) / NULLIF(SUM(MDScensus), 0), 3) as CNA_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (cna_hours / NULLIF(MDScensus, 0))), 3) as Median_CNA_HPRD,
            ROUND(SUM(natr_hours) / NULLIF(SUM(MDScensus), 0), 3) as NAtr_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (natr_hours / NULLIF(MDScensus, 0))), 3) as Median_NAtr_HPRD,
            ROUND(SUM(medaide_hours) / NULLIF(SUM(MDScensus), 0), 3) as MedAide_HPRD,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (medaide_hours / NULLIF(MDScensus, 0))), 3) as Median_MedAide_HPRD
        FROM daily_metrics
        """
        result = conn.execute(query).fetchone()
        
        if result is None:
            print(f"No data found for {region} in {quarter}")
            return None
            
        return {
            'Region': region,
            'CY_Qtr': quarter,
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
        print(f"Error calculating metrics for region {region} quarter {quarter}: {str(e)}")
        print(traceback.format_exc())
        return None

def get_states_in_region(region):
    """Get list of states in a CMS region"""
    region_mapping = {
        'Region 1': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
        'Region 2': ['NJ', 'NY', 'PR', 'VI'],
        'Region 3': ['DE', 'DC', 'MD', 'PA', 'VA', 'WV'],
        'Region 4': ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
        'Region 5': ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
        'Region 6': ['AR', 'LA', 'NM', 'OK', 'TX'],
        'Region 7': ['IA', 'KS', 'MO', 'NE'],
        'Region 8': ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
        'Region 9': ['AZ', 'CA', 'HI', 'NV', 'AS', 'GU', 'MP'],
        'Region 10': ['AK', 'ID', 'OR', 'WA']
    }
    return region_mapping.get(region, [])

def main():
    print("Starting regional metrics generation...")
    
    # Create a single database connection for the entire process
    conn = get_db_connection()
    try:
        # Load all quarterly data
        if not load_data(conn):
            print("Failed to load data. Exiting.")
            return

        cursor = conn.cursor()
        
        # Get unique quarters and print detailed information
        cursor.execute("""
            SELECT CY_Qtr, COUNT(*) as record_count
            FROM staffing 
            GROUP BY CY_Qtr
            ORDER BY CY_Qtr
        """)
        quarter_data = cursor.fetchall()
        
        if not quarter_data:
            print("No quarters found in the database. Please check if data was loaded correctly.")
            return

        print("\nQuarters found in database:")
        for quarter, count in quarter_data:
            print(f"  - {quarter}: {count:,} records")
        
        quarters = [row[0] for row in quarter_data]
        print(f"\nTotal quarters found: {len(quarters)}")
        print(f"Quarters: {', '.join(quarters)}")
        
        # Define CMS regions
        regions = [
            'Region 1', 'Region 2', 'Region 3', 'Region 4', 'Region 5',
            'Region 6', 'Region 7', 'Region 8', 'Region 9', 'Region 10'
        ]
        
        regional_metrics = []
        for region in regions:
            for quarter in quarters:
                print(f"\nProcessing region {region} quarter {quarter}")
                metrics = calculate_regional_metrics(conn, region, quarter)
                if metrics:
                    regional_metrics.append(metrics)
                    print(f"Successfully calculated metrics for {region} {quarter}")
                    print(f"  Facility Count: {metrics['Facility_Count']}")
                    print(f"  Total Resident Days: {metrics['Total_Resident_Days']:,.0f}")
                else:
                    print(f"No metrics calculated for {region} {quarter}")

        if not regional_metrics:
            print("No metrics were calculated. Please check the data and calculations.")
            return

        df = pd.DataFrame(regional_metrics)
        df.to_csv('region_quarterly_metrics.csv', index=False)
        print("\nregion_quarterly_metrics.csv generated successfully.")
        print(f"Generated metrics for {len(regional_metrics)} region/quarter combinations.")
        print("\nFirst few rows of the generated metrics:")
        print(df.head().to_string())

    except Exception as e:
        print(f"Error in main: {str(e)}")
        print(traceback.format_exc())
    finally:
        conn.close()

if __name__ == "__main__":
    main() 