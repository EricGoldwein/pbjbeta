import sqlite3
from datetime import datetime

def find_providers():
    try:
        conn = sqlite3.connect('pbj.db')
        cursor = conn.cursor()
        
        # Get a sample of provider IDs and names with their date ranges
        cursor.execute("""
            SELECT 
                PROVNUM,
                PROVNAME,
                STATE,
                MIN(WORKDATE) as first_date,
                MAX(WORKDATE) as last_date,
                COUNT(DISTINCT WORKDATE) as days_of_data
            FROM pbj_data
            GROUP BY PROVNUM, PROVNAME, STATE
            ORDER BY days_of_data DESC
            LIMIT 5
        """)
        
        providers = cursor.fetchall()
        print("\nSample Providers (sorted by most data points):")
        for provider in providers:
            print(f"\nID: {provider[0]}")
            print(f"Name: {provider[1]}")
            print(f"State: {provider[2]}")
            print(f"Date Range: {provider[3]} to {provider[4]}")
            print(f"Days of Data: {provider[5]}")
            
        # Get a count of total providers
        cursor.execute("SELECT COUNT(DISTINCT PROVNUM) FROM pbj_data")
        total = cursor.fetchone()[0]
        print(f"\nTotal number of providers: {total}")
        
        # Get date range of the data
        cursor.execute("""
            SELECT MIN(WORKDATE), MAX(WORKDATE)
            FROM pbj_data
        """)
        min_date, max_date = cursor.fetchone()
        print(f"\nData Date Range: {min_date} to {max_date}")
        
        conn.close()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    find_providers() 