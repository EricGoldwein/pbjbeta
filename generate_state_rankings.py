import pandas as pd
import os

def generate_state_rankings():
    """
    Generate state rankings based on Total_Nurse_HPRD for each quarter.
    Creates a new file similar to state_lite_metrics but with a rank column.
    """
    
    # Read the existing state_lite_metrics file
    try:
        state_metrics = pd.read_csv('state_lite_metrics.csv')
        print(f"Loaded {len(state_metrics)} state records from state_lite_metrics.csv")
    except FileNotFoundError:
        print("Error: state_lite_metrics.csv not found. Please run lite_report.py first to generate the base metrics.")
        return
    
    # Create a copy to avoid modifying the original
    state_rankings = state_metrics.copy()
    
    # Add rank column based on Total_Nurse_HPRD for each quarter
    # Higher HPRD gets lower rank (rank 1 = highest HPRD)
    state_rankings['HPRD_Rank'] = state_rankings.groupby('CY_Qtr')['Total_Nurse_HPRD'].rank(
        method='min', 
        ascending=False  # Higher HPRD gets rank 1
    ).astype(int)
    
    # Sort by quarter, then by rank
    state_rankings = state_rankings.sort_values(['CY_Qtr', 'HPRD_Rank'])
    
    # Reorder columns to put rank after state
    column_order = [
        'CY_Qtr', 'STATE', 'HPRD_Rank', 'Facility_Count', 'Census', 
        'Total_Nurse_HPRD', 'Contract_Percentage', 'State_Census'
    ]
    state_rankings = state_rankings[column_order]
    
    # Save the new file
    output_filename = 'state_lite_metrics_with_rankings.csv'
    state_rankings.to_csv(output_filename, index=False)
    
    # Print summary
    print(f"\nState Rankings Generation Summary:")
    print(f"Total state records: {len(state_rankings)}")
    print(f"Unique quarters: {state_rankings['CY_Qtr'].nunique()}")
    print(f"Unique states: {state_rankings['STATE'].nunique()}")
    
    # Show sample of latest quarter rankings
    latest_quarter = state_rankings['CY_Qtr'].max()
    print(f"\nLatest quarter ({latest_quarter}) rankings (top 10):")
    latest_rankings = state_rankings[state_rankings['CY_Qtr'] == latest_quarter].head(10)
    print(latest_rankings[['STATE', 'HPRD_Rank', 'Total_Nurse_HPRD', 'Facility_Count']].to_string(index=False))
    
    # Show sample of earliest quarter rankings
    earliest_quarter = state_rankings['CY_Qtr'].min()
    print(f"\nEarliest quarter ({earliest_quarter}) rankings (top 10):")
    earliest_rankings = state_rankings[state_rankings['CY_Qtr'] == earliest_quarter].head(10)
    print(earliest_rankings[['STATE', 'HPRD_Rank', 'Total_Nurse_HPRD', 'Facility_Count']].to_string(index=False))
    
    print(f"\nFile saved as: {output_filename}")
    
    return state_rankings

if __name__ == "__main__":
    generate_state_rankings()

