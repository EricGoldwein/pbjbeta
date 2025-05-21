import pandas as pd

def create_state_region_mapping():
    """Create a mapping of states to CMS regions."""
    # CMS Region to State mapping
    region_mapping = {
        'Region 1': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
        'Region 2': ['NJ', 'NY', 'PR', 'VI'],
        'Region 3': ['DE', 'DC', 'MD', 'PA', 'VA', 'WV'],
        'Region 4': ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
        'Region 5': ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
        'Region 6': ['AR', 'LA', 'NM', 'OK', 'TX'],
        'Region 7': ['IA', 'KS', 'MO', 'NE'],
        'Region 8': ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
        'Region 9': ['AZ', 'CA', 'HI', 'NV', 'AS', 'GU', 'MP'],
        'Region 10': ['AK', 'ID', 'OR', 'WA']
    }
    
    # Create a DataFrame with the mapping
    mapping_data = []
    for region, states in region_mapping.items():
        for state in states:
            mapping_data.append({
                'State': state,
                'Region': region
            })
    
    # Convert to DataFrame and save
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv('state_region_mapping.csv', index=False)
    print("State to Region mapping saved to state_region_mapping.csv")
    
    return mapping_df

if __name__ == "__main__":
    create_state_region_mapping() 

def create_state_region_mapping():
    """Create a mapping of states to CMS regions."""
    # CMS Region to State mapping
    region_mapping = {
        'Region 1': ['CT', 'ME', 'MA', 'NH', 'RI', 'VT'],
        'Region 2': ['NJ', 'NY', 'PR', 'VI'],
        'Region 3': ['DE', 'DC', 'MD', 'PA', 'VA', 'WV'],
        'Region 4': ['AL', 'FL', 'GA', 'KY', 'MS', 'NC', 'SC', 'TN'],
        'Region 5': ['IL', 'IN', 'MI', 'MN', 'OH', 'WI'],
        'Region 6': ['AR', 'LA', 'NM', 'OK', 'TX'],
        'Region 7': ['IA', 'KS', 'MO', 'NE'],
        'Region 8': ['CO', 'MT', 'ND', 'SD', 'UT', 'WY'],
        'Region 9': ['AZ', 'CA', 'HI', 'NV', 'AS', 'GU', 'MP'],
        'Region 10': ['AK', 'ID', 'OR', 'WA']
    }
    
    # Create a DataFrame with the mapping
    mapping_data = []
    for region, states in region_mapping.items():
        for state in states:
            mapping_data.append({
                'State': state,
                'Region': region
            })
    
    # Convert to DataFrame and save
    mapping_df = pd.DataFrame(mapping_data)
    mapping_df.to_csv('state_region_mapping.csv', index=False)
    print("State to Region mapping saved to state_region_mapping.csv")
    
    return mapping_df

if __name__ == "__main__":
    create_state_region_mapping() 