import pandas as pd
import sqlite3
import os

def check_facility_data():
    """Check if facility 335386 exists in our data and generate its CSV file."""
    try:
        # Check facility quarterly metrics
        print("Checking facility quarterly metrics...")
        df = pd.read_csv('facility_quarterly_metrics.csv', dtype={'PROVNUM': str})
        facility_data = df[df['PROVNUM'] == '335386']
        
        if not facility_data.empty:
            print("\nFacility found in quarterly metrics:")
            print(facility_data[['PROVNUM', 'PROVNAME', 'STATE', 'CY_QTR']].to_string())
            
            # Save to CSV
            output_file = f'facility_335386_test.csv'
            facility_data.to_csv(output_file, index=False)
            print(f"\nFacility data saved to {output_file}")
        else:
            print("\nFacility not found in quarterly metrics")
            
        # Check provider info
        print("\nChecking provider info...")
        df = pd.read_csv('NH_ProviderInfo_Mar2025.csv', dtype={'PROVNUM': str})
        facility_info = df[df['PROVNUM'] == '335386']
        
        if not facility_info.empty:
            print("\nFacility found in provider info:")
            print(facility_info[['PROVNUM', 'PROVNAME', 'STATE', 'CITY']].to_string())
        else:
            print("\nFacility not found in provider info")
            
    except Exception as e:
        print(f"Error checking facility data: {str(e)}")

if __name__ == "__main__":
    check_facility_data() 