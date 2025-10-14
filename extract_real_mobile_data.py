import pandas as pd
import json

# Read the national data
national_df = pd.read_csv('national_quarterly_metrics.csv')
national_data = national_df[national_df['STATE'] == 'NATIONAL']['Total_Nurse_HPRD'].tolist()

# Read the state data
state_df = pd.read_csv('pbj_lite/state_lite_metrics.csv')

# Get all unique states
states = sorted(state_df['STATE'].unique())

# Create quarters list
quarters = []
for year in range(2017, 2026):
    for q in range(1, 5):
        if year == 2025 and q > 1:  # Only Q1 2025 available
            break
        quarters.append(f'Q{q} {year}')

# Extract data for each state
state_data = {}
for state in states:
    state_quarters = state_df[state_df['STATE'] == state].sort_values('CY_Qtr')
    hprd_values = state_quarters['Total_Nurse_HPRD'].tolist()
    
    # Pad with zeros if we don't have all quarters
    while len(hprd_values) < len(quarters):
        hprd_values.append(0.0)
    
    # Truncate if we have too many
    hprd_values = hprd_values[:len(quarters)]
    
    state_data[state] = hprd_values

# Add USA data
state_data['USA'] = national_data

print("// Real data extracted from CSV files")
print("const realStateData = {")
for state, data in state_data.items():
    formatted_data = [f"{x:.3f}" for x in data]
    print(f"  '{state}': [{','.join(formatted_data)}],")
print("};")

print(f"\n// Quarters: {len(quarters)}")
print("const quarters = [")
for q in quarters:
    print(f"  '{q}',")
print("];")
