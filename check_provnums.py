import duckdb
import pandas as pd
import os
import numpy as np

def get_provnums_from_db():
    try:
        # Connect to DuckDB
        conn = duckdb.connect('nursing_home_staffing.db')
        
        # Get all distinct PROVNUMs from database
        result = conn.execute("""
            SELECT DISTINCT PROVNUM 
            FROM staffing 
            WHERE CY_Qtr = '2024Q3'
        """).fetchall()
        
        # Convert to set for easier comparison
        db_provnums = set(row[0] for row in result)
        print(f"Total distinct PROVNUMs in database for Q3 2024: {len(db_provnums)}")
        conn.close()
        return db_provnums
    except Exception as e:
        print(f"Error getting PROVNUMs from database: {str(e)}")
        return set()

def get_provnums_from_csv():
    csv_file = 'PBJ_dailynursestaffing_CY2024Q3.csv'
    encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    if not os.path.exists(csv_file):
        print(f"CSV file not found: {csv_file}")
        return set(), None
        
    for encoding in encodings:
        try:
            print(f"\nTrying to read CSV with {encoding} encoding...")
            df = pd.read_csv(csv_file, encoding=encoding, low_memory=False)
            csv_provnums = set(df['PROVNUM'].unique())
            print(f"Successfully read CSV with {encoding} encoding")
            print(f"Total distinct PROVNUMs in CSV: {len(csv_provnums)}")
            return csv_provnums, df
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding} encoding")
            continue
        except Exception as e:
            print(f"Error reading CSV with {encoding} encoding: {str(e)}")
            continue
    
    print("\nFailed to read CSV with any encoding")
    return set(), None

def analyze_facility_data(df, provnum):
    """Analyze data quality for a specific facility"""
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
    
    # Check data types and missing values
    print("\nData Types and Missing Values:")
    for col in facility_data.columns:
        missing = facility_data[col].isna().sum()
        if missing > 0:
            print(f"  - {col}: {missing} missing values")
    
    # Check for zero or negative values in numeric columns
    numeric_cols = facility_data.select_dtypes(include=[np.number]).columns
    print("\nZero and Negative Values:")
    for col in numeric_cols:
        zeros = (facility_data[col] == 0).sum()
        negatives = (facility_data[col] < 0).sum()
        if zeros > 0 or negatives > 0:
            print(f"  - {col}: {zeros} zero values, {negatives} negative values")
    
    # Check for potential outliers in numeric columns
    print("\nPotential Outliers (z-score > 3):")
    for col in numeric_cols:
        z_scores = np.abs((facility_data[col] - facility_data[col].mean()) / facility_data[col].std())
        outliers = (z_scores > 3).sum()
        if outliers > 0:
            print(f"  - {col}: {outliers} potential outliers")
    
    # Check for data type consistency
    print("\nData Type Consistency:")
    for col in facility_data.columns:
        if facility_data[col].dtype == 'object':
            unique_values = facility_data[col].nunique()
            if unique_values < len(facility_data) * 0.1:  # Less than 10% unique values
                print(f"  - {col}: Only {unique_values} unique values out of {len(facility_data)} records")
    
    return facility_data

def analyze_differences():
    print("\nAnalyzing PROVNUM differences...")
    
    db_provnums = get_provnums_from_db()
    csv_provnums, df = get_provnums_from_csv()
    
    if not db_provnums or not csv_provnums:
        print("Could not compare PROVNUMs due to data loading errors")
        return
    
    # Find differences
    only_in_csv = csv_provnums - db_provnums
    only_in_db = db_provnums - csv_provnums
    
    print("\nAnalysis Results:")
    print(f"PROVNUMs only in CSV ({len(only_in_csv)}):", sorted(only_in_csv) if only_in_csv else "None")
    print(f"PROVNUMs only in DB ({len(only_in_db)}):", sorted(only_in_db) if only_in_db else "None")
    
    if only_in_csv and df is not None:
        print("\nDetailed analysis of facilities missing from database:")
        for provnum in sorted(only_in_csv):
            analyze_facility_data(df, provnum)

if __name__ == "__main__":
    analyze_differences() 