import pandas as pd
import os
import duckdb
import glob

def get_db_connection():
    # Remove the existing database file if it exists
    if os.path.exists('nursing_home_staffing.db'):
        os.remove('nursing_home_staffing.db')
    return duckdb.connect('nursing_home_staffing.db')

def load_data(conn):
    try:
        cursor = conn.cursor()
        
        # Find all PBJ data files
        data_files = glob.glob('PBJ_Nurse/PBJ_dailynursestaffing_*.csv')
        if not data_files:
            print("No PBJ data files found!")
            return False
        
        # Load the first file to create the table structure
        first_file = sorted(data_files)[0]
        print(f"Creating table structure from {first_file}...")
        
        # Load first file with pandas
        df = pd.read_csv(first_file, encoding='latin1', dtype={'PROVNUM': str})
        
        # Register the dataframe with DuckDB
        cursor.execute("CREATE TABLE staffing AS SELECT * FROM df")
        conn.commit()
        print(f"Table structure created from {first_file}")
        
        # Load the remaining files
        for data_path in sorted(data_files)[1:]:
            print(f"Loading {data_path}...")
            try:
                # Load with pandas
                df = pd.read_csv(data_path, encoding='latin1', dtype={'PROVNUM': str})
                # Register and insert
                cursor.execute("INSERT INTO staffing SELECT * FROM df")
                conn.commit()
                print(f"Data loaded from {data_path}")
            except Exception as e:
                print(f"Error loading {data_path}: {str(e)}")
        return True
    except Exception as e:
        print(f"Error in load_data: {str(e)}")
        return False

def calculate_regional_metrics(conn, region, quarter):
    """Calculate regional metrics for a quarter"""
    # Get states in this region
    states_in_region = get_states_in_region(region)
    states_list = "', '".join(states_in_region)
    
    query = f"""
    WITH daily_metrics AS (
        SELECT 
            PROVNUM,
            WorkDate,
            MDScensus,
            (Hrs_RN + Hrs_LPN + Hrs_CNA) as total_hours,
            (Hrs_RN) as rn_hours,
            (Hrs_RN + Hrs_LPN) as nurse_care_hours,
            (Hrs_RN) as rn_care_hours,
            (Hrs_CNA) as nurse_assistant_hours,
            (Hrs_RN_ctr + Hrs_LPN_ctr + Hrs_CNA_ctr) as contract_hours
        FROM staffing 
        WHERE STATE IN ('{states_list}') AND CY_Qtr = '{quarter}'
    )
    SELECT
        COUNT(DISTINCT PROVNUM) as Facility_Count,
        SUM(MDScensus) as Total_Resident_Days,
        ROUND(SUM(total_hours) / NULLIF(SUM(MDScensus), 0), 3) as Total_HPRD,
        ROUND(SUM(rn_hours) / NULLIF(SUM(MDScensus), 0), 3) as RN_HPRD,
        ROUND(SUM(nurse_care_hours) / NULLIF(SUM(MDScensus), 0), 3) as Nurse_Care_HPRD,
        ROUND(SUM(rn_care_hours) / NULLIF(SUM(MDScensus), 0), 3) as RN_Care_HPRD,
        ROUND(SUM(nurse_assistant_hours) / NULLIF(SUM(MDScensus), 0), 3) as Nurse_Assistant_HPRD,
        ROUND(SUM(contract_hours) / NULLIF(SUM(total_hours), 0) * 100, 3) as Contract_Staff_Percentage
    FROM daily_metrics
    """
    try:
        result = conn.execute(query).fetchone()
        return {
            'Region': region,
            'Facility_Count': int(result[0]),
            'Total_Resident_Days': float(result[1]),
            'Total_HPRD': float(result[2]),
            'RN_HPRD': float(result[3]),
            'Nurse_Care_HPRD': float(result[4]),
            'RN_Care_HPRD': float(result[5]),
            'Nurse_Assistant_HPRD': float(result[6]),
            'Contract_Staff_Percentage': float(result[7])
        }
    except Exception as e:
        print(f"Error calculating metrics for region {region} quarter {quarter}: {str(e)}")
        return None

def calculate_regional_median_contract_percentage(conn, region, quarter):
    """Calculate median contract staff percentage for a region and quarter"""
    # Get states in this region
    states_in_region = get_states_in_region(region)
    states_list = "', '".join(states_in_region)
    
    query = f"""
    WITH facility_contract_pct AS (
        SELECT 
            PROVNUM,
            SUM(Hrs_RNDON_ctr + Hrs_RNadmin_ctr + Hrs_RN_ctr + Hrs_LPNadmin_ctr + Hrs_LPN_ctr + Hrs_CNA_ctr + Hrs_NAtrn_ctr + Hrs_MedAide_ctr) * 100.0 /
            NULLIF(SUM(Hrs_RNDON + Hrs_RNadmin + Hrs_RN + Hrs_LPNadmin + Hrs_LPN + Hrs_CNA + Hrs_NAtrn + Hrs_MedAide), 0) as contract_pct
        FROM staffing 
        WHERE STATE IN ('{states_list}') AND CY_Qtr = '{quarter}'
        GROUP BY PROVNUM
        HAVING SUM(Hrs_RNDON + Hrs_RNadmin + Hrs_RN + Hrs_LPNadmin + Hrs_LPN + Hrs_CNA + Hrs_NAtrn + Hrs_MedAide) > 0
    )
    SELECT 
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY contract_pct) as median_contract_pct
    FROM facility_contract_pct
    """
    try:
        result = conn.execute(query).fetchone()
        return float(result[0]) if result and result[0] is not None else 0.0
    except Exception as e:
        print(f"Error calculating median contract percentage for region {region} quarter {quarter}: {str(e)}")
        return 0.0

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
        
        # Get unique quarters
        cursor.execute("""
            SELECT DISTINCT CY_Qtr
            FROM staffing 
            ORDER BY CY_Qtr
        """)
        quarters = [row[0] for row in cursor.fetchall()]

        if not quarters:
            print("No quarters found in the database. Please check if data was loaded correctly.")
            return

        print(f"\nFound {len(quarters)} quarters")
        
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
                    metrics['Median_Contract_Percentage'] = calculate_regional_median_contract_percentage(conn, region, quarter)
                    metrics['CY_Qtr'] = quarter
                    regional_metrics.append(metrics)
                    print(f"Successfully calculated metrics for {region} {quarter}")
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
    finally:
        conn.close()

if __name__ == "__main__":
    main() 