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
        data_files = glob.glob('PBJ_dailynursestaffing_*.csv')
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

def calculate_state_metrics(conn, state, quarter):
    """Calculate state metrics for a quarter"""
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
            (Hrs_RNDON_ctr + Hrs_RNadmin_ctr + Hrs_RN_ctr + Hrs_LPNadmin_ctr + Hrs_LPN_ctr + Hrs_CNA_ctr + Hrs_NAtrn_ctr + Hrs_MedAide_ctr) as contract_hours
        FROM staffing 
        WHERE STATE = '{state}' AND CY_Qtr = '{quarter}'
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
            'STATE': state,
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
        print(f"Error calculating metrics for state {state} quarter {quarter}: {str(e)}")
        return None

def calculate_state_median_contract_percentage(conn, state, quarter):
    """Calculate median contract staff percentage for a state and quarter"""
    query = f"""
    WITH facility_contract_pct AS (
        SELECT 
            PROVNUM,
            SUM(Hrs_RNDON_ctr + Hrs_RNadmin_ctr + Hrs_RN_ctr + Hrs_LPNadmin_ctr + Hrs_LPN_ctr + Hrs_CNA_ctr + Hrs_NAtrn_ctr + Hrs_MedAide_ctr) * 100.0 /
            NULLIF(SUM(Hrs_RNDON + Hrs_RNadmin + Hrs_RN + Hrs_LPNadmin + Hrs_LPN + Hrs_CNA + Hrs_NAtrn + Hrs_MedAide), 0) as contract_pct
        FROM staffing 
        WHERE STATE = '{state}' AND CY_Qtr = '{quarter}'
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
        print(f"Error calculating median contract percentage for state {state} quarter {quarter}: {str(e)}")
        return 0.0

def main():
    print("Starting state metrics generation...")
    
    # Create a single database connection for the entire process
    conn = get_db_connection()
    try:
        # Load all quarterly data
        if not load_data(conn):
            print("Failed to load data. Exiting.")
            return

        cursor = conn.cursor()
        
        # Get unique states and quarters
        cursor.execute("""
            SELECT DISTINCT STATE, CY_Qtr
            FROM staffing 
            ORDER BY STATE, CY_Qtr
        """)
        state_quarters = cursor.fetchall()

        if not state_quarters:
            print("No state/quarter combinations found in the database. Please check if data was loaded correctly.")
            return

        print(f"\nFound {len(state_quarters)} state/quarter combinations")
        
        state_metrics = []
        for state, quarter in state_quarters:
            print(f"\nProcessing state {state} quarter {quarter}")
            metrics = calculate_state_metrics(conn, state, quarter)
            if metrics:
                metrics['Median_Contract_Percentage'] = calculate_state_median_contract_percentage(conn, state, quarter)
                metrics['CY_Qtr'] = quarter
                state_metrics.append(metrics)
                print(f"Successfully calculated metrics for {state} {quarter}")
            else:
                print(f"No metrics calculated for {state} {quarter}")

        if not state_metrics:
            print("No metrics were calculated. Please check the data and calculations.")
            return

        df = pd.DataFrame(state_metrics)
        df.to_csv('state_quarterly_metrics.csv', index=False)
        print("\nstate_quarterly_metrics.csv generated successfully.")
        print(f"Generated metrics for {len(state_metrics)} state/quarter combinations.")
        print("\nFirst few rows of the generated metrics:")
        print(df.head().to_string())

    except Exception as e:
        print(f"Error in main: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main() 