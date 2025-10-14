import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_and_process_california_data():
    """Load and process California PBJ data from standardized files."""
    print("Starting California RN 8 calculation...")
    
    # Get all standardized PBJ files
    pbj_files = glob.glob("standardized_PBJ/PBJ_dailynursestaffing_CY*.csv")
    pbj_files.sort()  # Process chronologically
    
    if not pbj_files:
        print("No standardized PBJ files found in standardized_PBJ directory.")
        return pd.DataFrame()
    
    print(f"Found {len(pbj_files)} PBJ files to process")
    
    # Load California data from all files
    california_data = []
    
    for i, file in enumerate(pbj_files):
        try:
            print(f"Processing file {i+1}/{len(pbj_files)}: {os.path.basename(file)}")
            
            # Read only California data with low_memory=False to avoid dtype warnings
            df = pd.read_csv(file, dtype={'PROVNUM': str}, low_memory=False)
            ca_data = df[df['STATE'] == 'CA'].copy()
            
            if not ca_data.empty:
                # Extract quarter from filename
                quarter = file.split('_')[-1].replace('.csv', '')
                ca_data['Quarter'] = quarter
                california_data.append(ca_data)
                print(f"  - Found {len(ca_data)} California records")
            else:
                print(f"  - No California data found")
                
        except Exception as e:
            print(f"  - Error loading {file}: {str(e)}")
            continue
    
    if not california_data:
        print("No California data found in PBJ files.")
        return pd.DataFrame()
    
    print("Combining all data...")
    # Combine all data
    combined_data = pd.concat(california_data, ignore_index=True)
    
    print("Processing data...")
    # Convert WorkDate to datetime
    combined_data['WorkDate'] = pd.to_datetime(combined_data['WorkDate'])
    
    # Calculate RN hours (RN + RN Admin + RN DON)
    combined_data['Total_RN_Hours'] = (
        combined_data['Hrs_RN'].fillna(0) + 
        combined_data['Hrs_RNadmin'].fillna(0) + 
        combined_data['Hrs_RNDON'].fillna(0)
    )
    
    # Calculate total nurse hours
    combined_data['Total_Nurse_Hours'] = (
        combined_data['Hrs_RN'].fillna(0) + 
        combined_data['Hrs_RNadmin'].fillna(0) + 
        combined_data['Hrs_RNDON'].fillna(0) +
        combined_data['Hrs_LPN'].fillna(0) + 
        combined_data['Hrs_LPNadmin'].fillna(0) +
        combined_data['Hrs_CNA'].fillna(0) + 
        combined_data['Hrs_NAtrn'].fillna(0) +
        combined_data['Hrs_MedAide'].fillna(0)
    )
    
    # Calculate HPRD (Hours Per Resident Day)
    combined_data['RN_HPRD'] = combined_data['Total_RN_Hours'] / combined_data['MDScensus'].fillna(1)
    combined_data['Total_Nurse_HPRD'] = combined_data['Total_Nurse_Hours'] / combined_data['MDScensus'].fillna(1)
    
    # Flag days with less than 8 hours RN
    combined_data['RN_Less_Than_8'] = combined_data['Total_RN_Hours'] < 8
    
    # Flag days with sub 3.50 total nurse HPRD
    combined_data['Sub_3_50_HPRD'] = combined_data['Total_Nurse_HPRD'] < 3.50
    
    print(f"Processed {len(combined_data)} total records")
    return combined_data

def calculate_facility_metrics(data):
    """Calculate facility-level metrics with quarter-by-quarter analysis."""
    print("Calculating facility-level metrics...")
    
    if data.empty:
        return pd.DataFrame()
    
    facility_metrics = []
    total_facilities = data['PROVNUM'].unique()
    
    # Also create quarter-by-quarter analysis
    quarter_analysis = []
    
    for i, provnum in enumerate(total_facilities):
        if i % 50 == 0:  # Progress update every 50 facilities
            print(f"Processing facility {i+1}/{len(total_facilities)}")
            
        facility_data = data[data['PROVNUM'] == provnum]
        
        # Get facility info
        facility_info = facility_data.iloc[0]
        
        # Calculate metrics
        total_days = len(facility_data)
        days_rn_less_8 = facility_data['RN_Less_Than_8'].sum()
        days_sub_350 = facility_data['Sub_3_50_HPRD'].sum()
        
        # Calculate percentages
        pct_rn_less_8 = (days_rn_less_8 / total_days * 100) if total_days > 0 else 0
        pct_sub_350 = (days_sub_350 / total_days * 100) if total_days > 0 else 0
        
        # Calculate average HPRD
        avg_rn_hprd = facility_data['RN_HPRD'].mean()
        avg_total_hprd = facility_data['Total_Nurse_HPRD'].mean()
        
        # Risk assessment
        risk_score = (pct_rn_less_8 * 0.6) + (pct_sub_350 * 0.4)
        
        if risk_score >= 50:
            risk_level = "High"
        elif risk_score >= 25:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        # Get quarter information and create quarter-by-quarter analysis
        quarters = facility_data['Quarter'].unique()
        quarter_str = ', '.join(sorted(quarters)) if len(quarters) <= 3 else f"{len(quarters)} quarters"
        
        # Add facility-level summary
        facility_metrics.append({
            'PROVNUM': provnum,
            'PROVNAME': facility_info['PROVNAME'],
            'CITY': facility_info['CITY'],
            'COUNTY_NAME': facility_info['COUNTY_NAME'],
            'CY_QTR': quarter_str,  # Add quarter information
            'Total_Days': total_days,
            'Days_RN_Less_8': days_rn_less_8,
            'Days_Sub_350': days_sub_350,
            'Pct_RN_Less_8': pct_rn_less_8,
            'Pct_Sub_350': pct_sub_350,
            'Avg_RN_HPRD': avg_rn_hprd,
            'Avg_Total_HPRD': avg_total_hprd,
            'Risk_Score': risk_score,
            'Risk_Level': risk_level
        })
        
        # Create quarter-by-quarter analysis for this facility
        for quarter in quarters:
            quarter_data = facility_data[facility_data['Quarter'] == quarter]
            quarter_days = len(quarter_data)
            quarter_rn_less_8 = quarter_data['RN_Less_Than_8'].sum()
            quarter_sub_350 = quarter_data['Sub_3_50_HPRD'].sum()
            
            quarter_analysis.append({
                'PROVNUM': provnum,
                'PROVNAME': facility_info['PROVNAME'],
                'CITY': facility_info['CITY'],
                'COUNTY_NAME': facility_info['COUNTY_NAME'],
                'CY_QTR': quarter,
                'Total_Days': quarter_days,
                'Days_RN_Less_8': quarter_rn_less_8,
                'Days_Sub_350': quarter_sub_350,
                'Pct_RN_Less_8': (quarter_rn_less_8 / quarter_days * 100) if quarter_days > 0 else 0,
                'Pct_Sub_350': (quarter_sub_350 / quarter_days * 100) if quarter_days > 0 else 0,
                'Avg_RN_HPRD': quarter_data['RN_HPRD'].mean(),
                'Avg_Total_HPRD': quarter_data['Total_Nurse_HPRD'].mean(),
                'Risk_Score': ((quarter_rn_less_8 / quarter_days * 100) * 0.6 + (quarter_sub_350 / quarter_days * 100) * 0.4) if quarter_days > 0 else 0,
                'Risk_Level': "High" if ((quarter_rn_less_8 / quarter_days * 100) * 0.6 + (quarter_sub_350 / quarter_days * 100) * 0.4) >= 50 else "Medium" if ((quarter_rn_less_8 / quarter_days * 100) * 0.6 + (quarter_sub_350 / quarter_days * 100) * 0.4) >= 25 else "Low"
            })
    
    # Save both facility summary and quarter-by-quarter data
    facility_df = pd.DataFrame(facility_metrics)
    quarter_df = pd.DataFrame(quarter_analysis)
    
    return facility_df, quarter_df

def main():
    """Main function to calculate and save RN 8 California metrics."""
    print("=" * 60)
    print("RN 8 California Metrics Calculator")
    print("=" * 60)
    
    # Load and process data
    data = load_and_process_california_data()
    
    if data.empty:
        print("No data to process. Exiting.")
        return
    
    # Calculate facility metrics
    facility_metrics, quarter_metrics = calculate_facility_metrics(data)
    
    if facility_metrics.empty:
        print("No facility metrics calculated. Exiting.")
        return
    
    # Save results
    facility_output_file = "rn8_california_metrics.csv"
    quarter_output_file = "rn8_california_quarterly_metrics.csv"
    
    facility_metrics.to_csv(facility_output_file, index=False)
    quarter_metrics.to_csv(quarter_output_file, index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("CALCULATION COMPLETE")
    print("=" * 60)
    print(f"Total facilities processed: {len(facility_metrics)}")
    print(f"High risk facilities: {len(facility_metrics[facility_metrics['Risk_Level'] == 'High'])}")
    print(f"Medium risk facilities: {len(facility_metrics[facility_metrics['Risk_Level'] == 'Medium'])}")
    print(f"Low risk facilities: {len(facility_metrics[facility_metrics['Risk_Level'] == 'Low'])}")
    print(f"\nAverage days RN < 8 hours: {facility_metrics['Pct_RN_Less_8'].mean():.1f}%")
    print(f"Average days < 3.50 HPRD: {facility_metrics['Pct_Sub_350'].mean():.1f}%")
    print(f"\nResults saved to: {facility_output_file}")
    print(f"Quarterly analysis saved to: {quarter_output_file}")
    print(f"Total quarters analyzed: {len(quarter_metrics['CY_QTR'].unique())}")
    print(f"Quarters: {', '.join(sorted(quarter_metrics['CY_QTR'].unique()))}")
    print("=" * 60)

if __name__ == "__main__":
    main()
