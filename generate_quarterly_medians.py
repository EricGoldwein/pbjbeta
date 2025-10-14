import pandas as pd
import json

def calculate_medians():
    """Calculate facility-level medians for each quarter."""
    print("Loading facility quarterly metrics...")
    df = pd.read_csv('facility_quarterly_metrics.csv', dtype={'PROVNUM': str}, low_memory=False)
    
    print(f"Loaded {len(df)} facility-quarter records")
    
    # Get unique quarters
    quarters = sorted(df['CY_Qtr'].unique())
    print(f"Processing {len(quarters)} quarters...")
    
    medians_data = []
    
    for quarter in quarters:
        quarter_df = df[df['CY_Qtr'] == quarter].copy()
        
        # Calculate medians for key metrics
        hprd_median = quarter_df['Total_Nurse_HPRD'].median()
        contract_median = quarter_df['Contract_Percentage'].median()
        rn_hprd_median = quarter_df['RN_HPRD'].median()
        nurse_care_median = quarter_df['Nurse_Care_HPRD'].median()
        
        medians_data.append({
            'CY_Qtr': quarter,
            'Total_Nurse_HPRD_Median': round(hprd_median, 3),
            'Contract_Percentage_Median': round(contract_median, 3),
            'RN_HPRD_Median': round(rn_hprd_median, 3),
            'Nurse_Care_HPRD_Median': round(nurse_care_median, 3),
            'facility_count': len(quarter_df)
        })
        
        print(f"{quarter}: HPRD={hprd_median:.3f}, Contract={contract_median:.2f}%, Facilities={len(quarter_df)}")
    
    # Save to CSV
    medians_df = pd.DataFrame(medians_data)
    medians_df.to_csv('quarterly_medians.csv', index=False)
    print(f"\nSaved quarterly_medians.csv with {len(medians_df)} quarters")
    
    # Also save as JSON for easier loading
    with open('quarterly_medians.json', 'w') as f:
        json.dump(medians_data, f, indent=2)
    print("Saved quarterly_medians.json")
    
    return medians_df

if __name__ == '__main__':
    calculate_medians()




