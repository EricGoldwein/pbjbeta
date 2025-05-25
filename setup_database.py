import pandas as pd
import sqlite3
import os
from pathlib import Path

def setup_database():
    """Initialize the SQLite database and load data if it doesn't exist."""
    db_path = 'pbj_data.db'
    
    # Check if database already exists
    if os.path.exists(db_path):
        print("Database already exists. Checking contents...")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM pbj_data")
        count = cursor.fetchone()[0]
        print(f"Found {count:,} records in database")
        
        # Check a sample of data
        cursor.execute("SELECT * FROM pbj_data LIMIT 1")
        sample = cursor.fetchone()
        if sample:
            print("Sample data found in database")
            conn.close()
            return
        else:
            print("WARNING: No data found in database despite record count")
    
    print("Setting up database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table with appropriate columns
    columns = [
        "PROVNUM TEXT",
        "PROVNAME TEXT",
        "CITY TEXT",
        "STATE TEXT",
        "COUNTY_NAME TEXT",
        "COUNTY_FIPS TEXT",
        "CY_Qtr TEXT",
        "WorkDate TEXT",
        "MDScensus REAL",
        "Hrs_RNDON REAL", "Hrs_RNDON_emp REAL", "Hrs_RNDON_ctr REAL",
        "Hrs_RNadmin REAL", "Hrs_RNadmin_emp REAL", "Hrs_RNadmin_ctr REAL",
        "Hrs_RN REAL", "Hrs_RN_emp REAL", "Hrs_RN_ctr REAL",
        "Hrs_LPNadmin REAL", "Hrs_LPNadmin_emp REAL", "Hrs_LPNadmin_ctr REAL",
        "Hrs_LPN REAL", "Hrs_LPN_emp REAL", "Hrs_LPN_ctr REAL",
        "Hrs_CNA REAL", "Hrs_CNA_emp REAL", "Hrs_CNA_ctr REAL",
        "Hrs_NAtrn REAL", "Hrs_NAtrn_emp REAL", "Hrs_NAtrn_ctr REAL",
        "Hrs_MedAide REAL", "Hrs_MedAide_emp REAL", "Hrs_MedAide_ctr REAL"
    ]
    
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS pbj_data (
        {', '.join(columns)}
    )
    """
    cursor.execute(create_table_sql)
    
    # Create indexes for faster querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cy_qtr ON pbj_data(CY_Qtr)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_provname ON pbj_data(PROVNAME)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_provnum ON pbj_data(PROVNUM)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_state ON pbj_data(STATE)")
    
    # Load historical data
    years = range(2017, 2025)
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    total_records = 0
    
    for year in years:
        for quarter in quarters:
            file_path = f'standardized_PBJ/PBJ_dailynursestaffing_CY{year}{quarter}.csv'
            try:
                print(f"\nAttempting to read {file_path}")
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    continue
                    
                df = pd.read_csv(file_path, encoding='latin1')
                print(f"Successfully read {len(df):,} records from {year}{quarter}")
                
                # Check if required columns exist
                required_columns = ['MDScensus', 'Hrs_RN', 'Hrs_RNadmin', 'Hrs_RNDON']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    print(f"WARNING: Missing required columns: {missing_columns}")
                    continue
                
                # Check for null values in key columns
                null_counts = df[required_columns].isnull().sum()
                if null_counts.any():
                    print(f"WARNING: Found null values in key columns:\n{null_counts}")
                
                df['CY_Qtr'] = f'{year}{quarter}'
                
                # Insert data into database
                df.to_sql('pbj_data', conn, if_exists='append', index=False)
                total_records += len(df)
                print(f"Loaded {year}{quarter} data ({len(df):,} records)")
                
            except FileNotFoundError:
                print(f"Warning: File not found for {year}{quarter}")
                continue
            except Exception as e:
                print(f"Warning: Error reading {year}{quarter}: {str(e)}")
                continue
    
    conn.commit()
    print(f"\nDatabase setup complete. Total records loaded: {total_records:,}")
    
    # Verify data was loaded
    cursor.execute("SELECT COUNT(*) FROM pbj_data")
    final_count = cursor.fetchone()[0]
    print(f"Final record count in database: {final_count:,}")
    
    conn.close()

if __name__ == "__main__":
    setup_database() 