import pandas as pd
import json

# Read the state data
df = pd.read_csv('pbj_lite/state_lite_metrics.csv')

# Extract data for each state
states = {}
for state in sorted(df['STATE'].unique()):
    state_data = df[df['STATE'] == state].sort_values('CY_Qtr')['Total_Nurse_HPRD'].round(3).tolist()
    states[state] = state_data

# Print JavaScript object format
print("const realStateData = {")
print("  'USA': [" + ", ".join(map(str, [3.707,3.737,3.724,3.74,3.74,3.771,3.766,3.757,3.74,3.754,3.754,3.75,3.775,3.894,3.869,3.926,3.925,3.747,3.618,3.613,3.616,3.625,3.614,3.606,3.63,3.663,3.662,3.679,3.677,3.714,3.73,3.749,3.73])) + "],")

# State name mapping
state_names = {
    'AK': 'Alaska', 'AL': 'Alabama', 'AR': 'Arkansas', 'AZ': 'Arizona',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'IA': 'Iowa',
    'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'MA': 'Massachusetts', 'MD': 'Maryland',
    'ME': 'Maine', 'MI': 'Michigan', 'MN': 'Minnesota', 'MO': 'Missouri',
    'MS': 'Mississippi', 'MT': 'Montana', 'NC': 'North Carolina', 'ND': 'North Dakota',
    'NE': 'Nebraska', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NV': 'Nevada', 'NY': 'New York', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
    'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WI': 'Wisconsin',
    'WV': 'West Virginia', 'WY': 'Wyoming'
}

for state_code in sorted(states.keys()):
    if state_code in state_names:
        full_name = state_names[state_code]
        values = ", ".join(map(str, states[state_code]))
        print(f"  '{full_name}': [{values}],")

print("};")
