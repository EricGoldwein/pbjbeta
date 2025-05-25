import pandas as pd
import numpy as np
import sqlite3
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

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
        else:
            print("WARNING: No data found in database despite record count")
        
        conn.close()
        return
    
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

def generate_simplified_report():
    """Generate a simplified report with Quarter, Aggregate MDScensus, and Aggregate RN Hours."""
    # Ensure database exists
    setup_database()
    
    # Connect to database
    conn = sqlite3.connect('pbj_data.db')
    cursor = conn.cursor()
    
    # Check if we have data
    cursor.execute("SELECT COUNT(*) FROM pbj_data")
    count = cursor.fetchone()[0]
    print(f"\nFound {count:,} total records in database")
    
    if count == 0:
        print("ERROR: No data found in database!")
        return
    
    # Create SQL query for simplified report
    sql_query = """
    SELECT 
        CY_Qtr as Quarter,
        SUM(MDScensus) as Aggregate_MDScensus,
        SUM(Hrs_RN + Hrs_RNadmin + Hrs_RNDON) as Aggregate_RN_Hours
    FROM pbj_data
    GROUP BY CY_Qtr
    ORDER BY CY_Qtr
    """
    
    # Execute query and create report
    report_df = pd.read_sql_query(sql_query, conn)
    print(f"Generated report with {len(report_df)} quarters")
    
    if len(report_df) == 0:
        print("ERROR: No data returned from query!")
        return
    
    # Format numbers with thousands separators
    formatted_df = report_df.copy()
    formatted_df['Aggregate_MDScensus'] = formatted_df['Aggregate_MDScensus'].apply(lambda x: f"{x:,.0f}")
    formatted_df['Aggregate_RN_Hours'] = formatted_df['Aggregate_RN_Hours'].apply(lambda x: f"{x:,.0f}")
    
    # Print the report
    print("\nSimplified PBJ Report (2017-2024)")
    print("=" * 50)
    print(formatted_df.to_string(index=False))
    
    # Save to CSV (without thousands separators for data analysis)
    output_file = 'simplified_pbj_report.csv'
    report_df.to_csv(output_file, index=False)
    print(f"\nDetailed data saved to {output_file}")
    
    conn.close()

def generate_historical_report(pivot_by='quarter'):
    """
    Generate report using SQLite database.
    
    Args:
        pivot_by (str): 'quarter', 'state', 'provnum', or 'provname' to determine pivot grouping
    """
    # Ensure database exists
    setup_database()
    
    # Connect to database
    conn = sqlite3.connect('pbj_data.db')
    
    # Define columns to analyze
    columns = [
        "MDScensus",
        "Hrs_RNDON", "Hrs_RNDON_emp", "Hrs_RNDON_ctr",
        "Hrs_RNadmin", "Hrs_RNadmin_emp", "Hrs_RNadmin_ctr",
        "Hrs_RN", "Hrs_RN_emp", "Hrs_RN_ctr",
        "Hrs_LPNadmin", "Hrs_LPNadmin_emp", "Hrs_LPNadmin_ctr",
        "Hrs_LPN", "Hrs_LPN_emp", "Hrs_LPN_ctr",
        "Hrs_CNA", "Hrs_CNA_emp", "Hrs_CNA_ctr",
        "Hrs_NAtrn", "Hrs_NAtrn_emp", "Hrs_NAtrn_ctr",
        "Hrs_MedAide", "Hrs_MedAide_emp", "Hrs_MedAide_ctr"
    ]
    
    # Determine group by field based on pivot_by parameter
    group_by = {
        'quarter': 'CY_Qtr',
        'state': 'STATE',
        'provnum': 'PROVNUM',
        'provname': 'PROVNAME'
    }.get(pivot_by.lower(), 'CY_Qtr')
    
    # Create SQL query for pivot table
    select_columns = ', '.join([f'SUM({col}) as {col}' for col in columns])
    sql_query = f"""
    SELECT {group_by}, {select_columns}
    FROM pbj_data
    GROUP BY {group_by}
    ORDER BY {group_by}
    """
    
    # Execute query and create pivot table
    pivot_table = pd.read_sql_query(sql_query, conn)
    pivot_table.set_index(group_by, inplace=True)
    
    # Format numbers with thousands separators
    formatted_table = pivot_table.copy()
    for col in formatted_table.columns:
        formatted_table[col] = formatted_table[col].apply(lambda x: f"{x:,.0f}")
    
    # Print the report
    print(f"\nHistorical PBJ Report (2017-2024) - Pivoted by {pivot_by.upper()}")
    print("=" * 50)
    print(formatted_table)
    
    # Save to CSV (without thousands separators for data analysis)
    output_file = f'historical_pbj_report_{pivot_by}.csv'
    pivot_table.to_csv(output_file)
    print(f"\nDetailed data saved to {output_file}")
    
    conn.close()

def calculate_days_in_quarter(year, quarter):
    """Calculate the number of days in a given quarter, accounting for leap years."""
    quarter_months = {
        'Q1': (1, 3),
        'Q2': (4, 6),
        'Q3': (7, 9),
        'Q4': (10, 12)
    }
    start_month, end_month = quarter_months[quarter]
    start_date = datetime(year, start_month, 1)
    if end_month == 12:
        end_date = datetime(year + 1, 1, 1)
    else:
        end_date = datetime(year, end_month + 1, 1)
    return (end_date - start_date).days

def generate_national_metrics():
    """Generate national metrics report with HPRD calculations and other key metrics."""
    # Connect to existing database
    if not os.path.exists('pbj_data.db'):
        print("ERROR: Database not found. Please run setup_database.py first.")
        return
        
    conn = sqlite3.connect('pbj_data.db')
    
    # Create SQL query for national metrics
    sql_query = """
    WITH quarterly_totals AS (
        SELECT 
            CY_Qtr,
            SUM(MDScensus) as Total_MDScensus,
            SUM(Hrs_RNDON + Hrs_RNadmin + Hrs_RN + 
                Hrs_LPNadmin + Hrs_LPN + Hrs_CNA + 
                Hrs_NAtrn + Hrs_MedAide) as Total_Nurse_Hours,
            SUM(Hrs_RNDON_ctr + Hrs_RNadmin_ctr + Hrs_RN_ctr +
                Hrs_LPNadmin_ctr + Hrs_LPN_ctr +
                Hrs_CNA_ctr + Hrs_NAtrn_ctr + Hrs_MedAide_ctr) as Total_Contract_Hours,
            SUM(Hrs_RN + Hrs_RNadmin + Hrs_RNDON) as Total_RN_Hours,
            SUM(Hrs_CNA + Hrs_NAtrn + Hrs_MedAide) as Total_NA_Hours
        FROM pbj_data
        GROUP BY CY_Qtr
    )
    SELECT 
        CY_Qtr as Quarter,
        Total_MDScensus,
        Total_Nurse_Hours,
        Total_Contract_Hours,
        Total_RN_Hours,
        Total_NA_Hours,
        ROUND(Total_Nurse_Hours / Total_MDScensus, 2) as Total_Nurse_HPRD,
        ROUND(Total_RN_Hours / Total_MDScensus, 2) as RN_HPRD,
        ROUND(Total_NA_Hours / Total_MDScensus, 2) as NA_HPRD,
        ROUND((Total_Contract_Hours * 100.0 / Total_Nurse_Hours), 2) as Contract_Percentage
    FROM quarterly_totals
    ORDER BY CY_Qtr
    """
    
    # Execute query and create report
    report_df = pd.read_sql_query(sql_query, conn)
    
    # Calculate Average Daily Census
    report_df['Year'] = report_df['Quarter'].str[:4]
    report_df['Quarter_Code'] = report_df['Quarter'].str[4:]
    report_df['Days_in_Quarter'] = report_df.apply(
        lambda row: calculate_days_in_quarter(int(row['Year']), row['Quarter_Code']), 
        axis=1
    )
    report_df['Average_Daily_Census'] = report_df['Total_MDScensus'] / report_df['Days_in_Quarter']
    
    # Create a copy for formatted display
    formatted_df = report_df.copy()
    
    # Format numbers with thousands separators
    numeric_columns = [
        'Total_MDScensus', 'Total_Nurse_Hours', 'Total_Contract_Hours',
        'Total_RN_Hours', 'Total_NA_Hours'
    ]
    for col in numeric_columns:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{float(x):,.0f}")
    
    # Format HPRD and percentage values
    formatted_df['Total_Nurse_HPRD'] = formatted_df['Total_Nurse_HPRD'].apply(lambda x: f"{float(x):.2f}")
    formatted_df['RN_HPRD'] = formatted_df['RN_HPRD'].apply(lambda x: f"{float(x):.2f}")
    formatted_df['NA_HPRD'] = formatted_df['NA_HPRD'].apply(lambda x: f"{float(x):.2f}")
    formatted_df['Contract_Percentage'] = formatted_df['Contract_Percentage'].apply(lambda x: f"{float(x):.1f}%")
    formatted_df['Average_Daily_Census'] = formatted_df['Average_Daily_Census'].apply(lambda x: f"{float(x):,.1f}")
    
    # Select and reorder columns for final output
    final_columns = [
        'Quarter',
        'Total_MDScensus',
        'Average_Daily_Census',
        'Total_Nurse_Hours',
        'Total_Nurse_HPRD',
        'Total_RN_Hours',
        'RN_HPRD',
        'Total_NA_Hours',
        'NA_HPRD',
        'Contract_Percentage'
    ]
    formatted_df = formatted_df[final_columns]
    
    # Print the report
    print("\nNational PBJ Metrics Report (2017-2024)")
    print("=" * 80)
    print(formatted_df.to_string(index=False))
    
    # Save to CSV (without formatting for data analysis)
    output_file = 'national_pbj_metrics.csv'
    report_df[final_columns].to_csv(output_file, index=False)
    print(f"\nDetailed data saved to {output_file}")
    
    conn.close()

def generate_hprd_visualization():
    """Generate a line plot showing Total Nurse HPRD by quarter nationally."""
    # Connect to existing database
    if not os.path.exists('pbj_data.db'):
        print("ERROR: Database not found. Please run setup_database.py first.")
        return
        
    conn = sqlite3.connect('pbj_data.db')
    
    # Create SQL query for HPRD data
    sql_query = """
    WITH quarterly_totals AS (
        SELECT 
            CY_Qtr,
            SUM(MDScensus) as Total_MDScensus,
            SUM(Hrs_RNDON + Hrs_RNadmin + Hrs_RN + 
                Hrs_LPNadmin + Hrs_LPN + Hrs_CNA + 
                Hrs_NAtrn + Hrs_MedAide) as Total_Nurse_Hours
        FROM pbj_data
        GROUP BY CY_Qtr
    )
    SELECT 
        CY_Qtr as Quarter,
        ROUND(Total_Nurse_Hours / Total_MDScensus, 2) as Total_Nurse_HPRD
    FROM quarterly_totals
    ORDER BY CY_Qtr
    """
    
    # Execute query and create DataFrame
    hprd_df = pd.read_sql_query(sql_query, conn)
    
    # Set up the plot style
    plt.style.use('seaborn')
    plt.figure(figsize=(12, 6))
    
    # Create the line plot
    plt.plot(hprd_df['Quarter'], hprd_df['Total_Nurse_HPRD'], 
             marker='o', linewidth=2, markersize=8)
    
    # Customize the plot
    plt.title('National Total Nurse HPRD by Quarter (2017-2024)', 
              fontsize=14, pad=20)
    plt.xlabel('Quarter', fontsize=12)
    plt.ylabel('Total Nurse HPRD', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels on top of each point
    for x, y in zip(hprd_df['Quarter'], hprd_df['Total_Nurse_HPRD']):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('national_nurse_hprd_trend.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'national_nurse_hprd_trend.png'")
    
    # Show the plot
    plt.show()
    
    conn.close()

if __name__ == "__main__":
    # Generate national metrics report
    generate_national_metrics()
    
    # Generate HPRD visualization
    generate_hprd_visualization() 