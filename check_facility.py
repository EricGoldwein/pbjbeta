import pandas as pd

def check_facility():
    # Read the raw CSV file
    csv_path = 'standardized_PBJ/PBJ_dailynursestaffing_CY2019Q1.csv'
    df = pd.read_csv(csv_path)
    
    # Filter for facility 015009
    facility_data = df[df['PROVNUM'] == '015009']
    
    if len(facility_data) > 0:
        print(f"\nFound {len(facility_data)} records for facility 015009 in 2019 Q1")
        
        # Print all columns to see what data we have
        print("\nAll columns in the data:")
        print(facility_data.columns.tolist())
        
        # Check for NULL values in all numeric columns
        numeric_columns = facility_data.select_dtypes(include=['float64', 'int64']).columns
        null_counts = facility_data[numeric_columns].isnull().sum()
        print("\nNULL value counts in numeric columns:")
        print(null_counts)
        
        # Check for zero values in all numeric columns
        zero_counts = (facility_data[numeric_columns] == 0).sum()
        print("\nZero value counts in numeric columns:")
        print(zero_counts)
        
        # Print summary statistics for all numeric columns
        print("\nSummary statistics for numeric columns:")
        print(facility_data[numeric_columns].describe())
        
        # Print first few rows with all columns
        print("\nFirst few rows of data (all columns):")
        print(facility_data.head().to_string())
    else:
        print("Facility 015009 NOT found in 2019 Q1 in the raw CSV")

if __name__ == '__main__':
    check_facility() 