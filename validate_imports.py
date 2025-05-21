import pandas as pd
import duckdb
import sqlite3
import os
import glob

def get_csv_provnums():
    """Get all PROVNUMs from CSV files"""
    csv_files = glob.glob('PBJ_dailynursestaffing_*.csv')
    all_provnums = set()
    
    for csv_file in sorted(csv_files):
        quarter = csv_file.split('_')[-1].split('.')[0]
        print(f"Reading {csv_file}...")
        df = pd.read_csv(csv_file, encoding='latin1', dtype={'PROVNUM': str})
        provnums = set(df['PROVNUM'].unique())
        all_provnums.update(provnums)
        print(f"  - {quarter}: {len(provnums)} facilities")
    
    return all_provnums

def get_duckdb_provnums():
    """Get all PROVNUMs from DuckDB"""
    if not os.path.exists('nursing_home_staffing.db'):
        print("DuckDB database not found!")
        return set()
    
    conn = duckdb.connect('nursing_home_staffing.db')
    cursor = conn.cursor()
    
    # Get all quarters
    cursor.execute("SELECT DISTINCT CY_Qtr FROM staffing ORDER BY CY_Qtr")
    quarters = [row[0] for row in cursor.fetchall()]
    
    all_provnums = set()
    for quarter in quarters:
        cursor.execute(f"""
            SELECT COUNT(DISTINCT PROVNUM) 
            FROM staffing 
            WHERE CY_Qtr = '{quarter}'
        """)
        count = cursor.fetchone()[0]
        print(f"  - {quarter}: {count} facilities")
        
        cursor.execute(f"""
            SELECT DISTINCT PROVNUM 
            FROM staffing 
            WHERE CY_Qtr = '{quarter}'
        """)
        provnums = set(row[0] for row in cursor.fetchall())
        all_provnums.update(provnums)
    
    conn.close()
    return all_provnums

def get_sqlite_provnums():
    """Get all PROVNUMs from SQLite"""
    if not os.path.exists('pbj_data.db'):
        print("SQLite database not found!")
        return set()
    
    conn = sqlite3.connect('pbj_data.db', check_same_thread=False)
    cursor = conn.cursor()
    
    # Get all quarters
    cursor.execute("SELECT DISTINCT CY_Qtr FROM staffing ORDER BY CY_Qtr")
    quarters = [row[0] for row in cursor.fetchall()]
    
    all_provnums = set()
    for quarter in quarters:
        cursor.execute(f"""
            SELECT COUNT(DISTINCT PROVNUM) 
            FROM staffing 
            WHERE CY_Qtr = '{quarter}'
        """)
        count = cursor.fetchone()[0]
        print(f"  - {quarter}: {count} facilities")
        
        cursor.execute(f"""
            SELECT DISTINCT PROVNUM 
            FROM staffing 
            WHERE CY_Qtr = '{quarter}'
        """)
        provnums = set(row[0] for row in cursor.fetchall())
        all_provnums.update(provnums)
    
    conn.close()
    return all_provnums

def validate_imports():
    """Validate that all facilities are properly imported in both databases"""
    print("Validating imports...")
    
    # Get PROVNUMs from CSV files
    print("\nCSV Files:")
    csv_provnums = get_csv_provnums()
    print(f"Total distinct PROVNUMs in CSV files: {len(csv_provnums)}")
    
    # Get PROVNUMs from DuckDB
    print("\nDuckDB Database:")
    duckdb_provnums = get_duckdb_provnums()
    print(f"Total distinct PROVNUMs in DuckDB: {len(duckdb_provnums)}")
    
    # Get PROVNUMs from SQLite
    print("\nSQLite Database:")
    sqlite_provnums = get_sqlite_provnums()
    print(f"Total distinct PROVNUMs in SQLite: {len(sqlite_provnums)}")
    
    # Find differences
    only_in_csv_duckdb = csv_provnums - duckdb_provnums
    only_in_csv_sqlite = csv_provnums - sqlite_provnums
    
    print("\nAnalysis Results:")
    print(f"PROVNUMs only in CSV but missing from DuckDB ({len(only_in_csv_duckdb)}):", 
          sorted(only_in_csv_duckdb) if only_in_csv_duckdb else "None")
    print(f"PROVNUMs only in CSV but missing from SQLite ({len(only_in_csv_sqlite)}):", 
          sorted(only_in_csv_sqlite) if only_in_csv_sqlite else "None")
    
    # Check if all databases have the same number of facilities
    if len(csv_provnums) == len(duckdb_provnums) == len(sqlite_provnums):
        print("\nSUCCESS: All databases have the same number of facilities!")
        return True
    else:
        print("\nWARNING: Databases have different numbers of facilities!")
        return False

if __name__ == "__main__":
    validate_imports() 