import pandas as pd

def analyze_facility_metrics():
    """Analyze facility_quarterly_metrics.csv file."""
    try:
        # Read the CSV file with PROVNUM as string
        df = pd.read_csv('facility_quarterly_metrics.csv', dtype={'PROVNUM': str})
        
        # Get Q4 2020 data
        q4_2020_data = df[df['CY_Qtr'] == '2020Q4']
        
        # Count unique PROVNUMs
        q4_2020_count = q4_2020_data['PROVNUM'].nunique()
        print(f"\nUnique facilities in Q4 2020: {q4_2020_count:,}")
        
        # Check for duplicates
        duplicates = q4_2020_data[q4_2020_data.duplicated(['PROVNUM'], keep=False)]
        if not duplicates.empty:
            print("\nFound duplicate PROVNUMs in Q4 2020:")
            print(duplicates.sort_values('PROVNUM')[['PROVNUM', 'PROVNAME', 'STATE']].to_string())
        
        # Check for any unusual PROVNUMs
        print("\nChecking for unusual PROVNUMs (not 6 digits):")
        unusual_provnums = q4_2020_data[~q4_2020_data['PROVNUM'].str.len().isin([6])]
        if not unusual_provnums.empty:
            print(unusual_provnums[['PROVNUM', 'PROVNAME', 'STATE']].to_string())
        
        # Show first few rows for Q4 2020
        print("\nFirst 5 rows for Q4 2020 (sorted by PROVNUM):")
        print(q4_2020_data.sort_values('PROVNUM').head().to_string())
        
    except FileNotFoundError:
        print("Error: facility_quarterly_metrics.csv not found")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    analyze_facility_metrics() 