import pandas as pd
import duckdb
import sqlite3
import os
import numpy as np
import glob

def fix_duckdb_import():
    """
    Fix the DuckDB import process to properly handle the missing facilities.
    """
    print("Fixing DuckDB import process...")
    
    # Remove the existing database file if it exists
    if os.path.exists('nursing_home_staffing.db'):
        os.remove('nursing_home_staffing.db')
    
    # Connect to DuckDB
    conn = duckdb.connect('nursing_home_staffing.db')
    cursor = conn.cursor()
    
    # Find all PBJ data files
    data_files = glob.glob('PBJ_dailynursestaffing_*.csv')
    if not data_files:
        print("No PBJ data files found!")
        return False
    
    # Create empty table first
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
    
    # Verify the import
    result = cursor.execute("""
        SELECT COUNT(DISTINCT PROVNUM) 
        FROM staffing 
        WHERE CY_Qtr = '2024Q3'
    """).fetchone()
    
    print(f"Total distinct PROVNUMs in database after fix: {result[0]}")
    conn.close()
    return True

def fix_sqlite_import():
    """
    Fix the SQLite import process to properly handle the missing facilities.
    """
    print("Fixing SQLite import process...")
    
    # Remove the existing database file if it exists
    if os.path.exists('pbj_data.db'):
        os.remove('pbj_data.db')
    
    # Connect to SQLite
    conn = sqlite3.connect('pbj_data.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Read first row of CSV to get column names
    file_path = 'PBJ_dailynursestaffing_CY2024Q3.csv'
    df_sample = pd.read_csv(file_path, nrows=1, encoding='latin1')
    
    # Create table with dynamic columns
    columns = []
    for col in df_sample.columns:
        if col in ['PROVNUM', 'PROVNAME', 'CITY', 'STATE', 'COUNTY_NAME', 'COUNTY_FIPS', 'CY_Qtr', 'WorkDate']:
            columns.append(f"{col} TEXT")
        else:
            columns.append(f"{col} REAL")
    
    # Create table without PRIMARY KEY constraint to avoid issues with duplicates
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS staffing (
        {', '.join(columns)}
    )
    """
    
    cursor.execute(create_table_sql)
    conn.commit()
    
    # Load data from CSV file
    print(f"Loading data from {file_path}...")
    
    # Process in chunks to handle large files
    chunksize = 100000
    for chunk in pd.read_csv(file_path, 
                           dtype={'PROVNUM': str},
                           encoding='latin1',
                           chunksize=chunksize):
        # Clean data before inserting
        # Replace NaN with None to avoid SQLite issues
        chunk = chunk.replace({np.nan: None})
        
        # Insert data
        num_columns = len(chunk.columns)
        safe_insert_chunksize = 999 // num_columns  # SQLite limit safeguard
        chunk.to_sql(
            'staffing',
            conn,
            if_exists='append',
            index=False,
            method='multi',
            chunksize=safe_insert_chunksize
        )
        conn.commit()
    
    # Create indexes
    print("Creating indexes...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_state ON staffing(STATE)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_provnum ON staffing(PROVNUM)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cy_qtr ON staffing(CY_Qtr)")
    conn.commit()
    
    # Verify the import
    cursor.execute("""
        SELECT COUNT(DISTINCT PROVNUM) 
        FROM staffing 
        WHERE CY_Qtr = '2024Q3'
    """)
    result = cursor.fetchone()
    
    print(f"Total distinct PROVNUMs in database after fix: {result[0]}")
    conn.close()
    return True

def check_missing_facilities():
    """
    Check if the missing facilities are now in the database.
    """
    print("\nChecking for missing facilities...")
    
    # Get PROVNUMs from CSV
    csv_file = 'PBJ_dailynursestaffing_CY2024Q3.csv'
    df = pd.read_csv(csv_file, encoding='latin1', low_memory=False)
    csv_provnums = set(df['PROVNUM'].unique())
    print(f"Total distinct PROVNUMs in CSV: {len(csv_provnums)}")
    
    # Get PROVNUMs from DuckDB
    conn_duckdb = duckdb.connect('nursing_home_staffing.db')
    result_duckdb = conn_duckdb.execute("""
        SELECT DISTINCT PROVNUM 
        FROM staffing 
        WHERE CY_Qtr = '2024Q3'
    """).fetchall()
    duckdb_provnums = set(row[0] for row in result_duckdb)
    print(f"Total distinct PROVNUMs in DuckDB: {len(duckdb_provnums)}")
    conn_duckdb.close()
    
    # Get PROVNUMs from SQLite
    conn_sqlite = sqlite3.connect('pbj_data.db', check_same_thread=False)
    cursor_sqlite = conn_sqlite.cursor()
    cursor_sqlite.execute("""
        SELECT DISTINCT PROVNUM 
        FROM staffing 
        WHERE CY_Qtr = '2024Q3'
    """)
    sqlite_provnums = set(row[0] for row in cursor_sqlite.fetchall())
    print(f"Total distinct PROVNUMs in SQLite: {len(sqlite_provnums)}")
    conn_sqlite.close()
    
    # Find differences
    only_in_csv_duckdb = csv_provnums - duckdb_provnums
    only_in_csv_sqlite = csv_provnums - sqlite_provnums
    
    print("\nAnalysis Results:")
    print(f"PROVNUMs only in CSV but missing from DuckDB ({len(only_in_csv_duckdb)}):", 
          sorted(only_in_csv_duckdb) if only_in_csv_duckdb else "None")
    print(f"PROVNUMs only in CSV but missing from SQLite ({len(only_in_csv_sqlite)}):", 
          sorted(only_in_csv_sqlite) if only_in_csv_sqlite else "None")
    
    if only_in_csv_duckdb or only_in_csv_sqlite:
        print("\nDetailed analysis of facilities still missing:")
        missing_provnums = sorted(only_in_csv_duckdb.union(only_in_csv_sqlite))
        for provnum in missing_provnums:
            facility_data = df[df['PROVNUM'] == provnum].copy()
            print(f"\n{'='*80}")
            print(f"PROVNUM: {provnum}")
            print(f"Name: {facility_data['PROVNAME'].iloc[0]}")
            print(f"Location: {facility_data['CITY'].iloc[0]}, {facility_data['STATE'].iloc[0]}")
            print(f"Number of records: {len(facility_data)}")
            
            # Check for duplicate WorkDate entries
            duplicates = facility_data[facility_data.duplicated(['WorkDate'], keep=False)]
            if not duplicates.empty:
                print(f"\nWARNING: Found {len(duplicates)} duplicate WorkDate entries")
                print("Duplicate dates:", sorted(duplicates['WorkDate'].unique()))
            
            # Check for data type issues
            print("\nData Types:")
            for col in facility_data.columns:
                print(f"  - {col}: {facility_data[col].dtype}")
    
    return len(only_in_csv_duckdb) == 0 and len(only_in_csv_sqlite) == 0

if __name__ == "__main__":
    print("Starting fix for missing facilities...")
    
    # Fix DuckDB import
    duckdb_fixed = fix_duckdb_import()
    
    # Fix SQLite import
    sqlite_fixed = fix_sqlite_import()
    
    # Check if the fixes worked
    if duckdb_fixed and sqlite_fixed:
        all_fixed = check_missing_facilities()
        if all_fixed:
            print("\nSUCCESS: All facilities are now properly imported into both databases!")
        else:
            print("\nWARNING: Some facilities are still missing. Manual intervention may be required.")
    else:
        print("\nERROR: Failed to fix the import processes.") 