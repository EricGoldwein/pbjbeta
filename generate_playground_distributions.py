import pandas as pd
import json

def calculate_distribution(values, bin_size, max_value=8.0):
    """Calculate distribution of values into bins."""
    if not values or len(values) == 0:
        return {}
    
    # Create bins up to max_value, then one final bin for outliers
    bins = {}
    current = 0
    while current < max_value:
        bin_label = f"{current:.2f}-{(current + bin_size):.2f}"
        bins[bin_label] = 0
        current += bin_size
    
    # Add final bin for outliers
    bins[f"{max_value:.2f}+"] = 0
    
    # Count values in each bin
    for value in values:
        if value >= max_value:
            bins[f"{max_value:.2f}+"] += 1
        else:
            bin_index = int(value / bin_size)
            bin_start = bin_index * bin_size
            bin_label = f"{bin_start:.2f}-{(bin_start + bin_size):.2f}"
            if bin_label in bins:
                bins[bin_label] += 1
    
    return bins

def calculate_contract_distribution(values):
    """Calculate contract staffing distribution with custom bins."""
    bins = {
        '0%': 0,
        '>0-5%': 0,
        '5-10%': 0,
        '10-15%': 0,
        '15-20%': 0,
        '20-25%': 0,
        '25-30%': 0,
        '30%+': 0
    }
    
    for value in values:
        if value == 0:
            bins['0%'] += 1
        elif value > 0 and value < 5:
            bins['>0-5%'] += 1
        elif value >= 5 and value < 10:
            bins['5-10%'] += 1
        elif value >= 10 and value < 15:
            bins['10-15%'] += 1
        elif value >= 15 and value < 20:
            bins['15-20%'] += 1
        elif value >= 20 and value < 25:
            bins['20-25%'] += 1
        elif value >= 25 and value < 30:
            bins['25-30%'] += 1
        elif value >= 30:
            bins['30%+'] += 1
    
    return bins

def generate_distributions():
    """Generate distribution data for the playground."""
    print("Loading facility quarterly metrics...")
    df = pd.read_csv('facility_quarterly_metrics.csv')
    
    # Get latest quarter
    latest_quarter = df['CY_Qtr'].max()
    print(f"Latest quarter: {latest_quarter}")
    
    # Filter to latest quarter
    latest_df = df[df['CY_Qtr'] == latest_quarter].copy()
    
    # Extract HPRD values
    hprd_values = latest_df['Total_Nurse_HPRD'].dropna().tolist()
    print(f"Found {len(hprd_values)} facilities with HPRD data")
    
    # Extract contract percentage values
    contract_values = latest_df['Contract_Percentage'].dropna().tolist()
    print(f"Found {len(contract_values)} facilities with contract data")
    
    # Calculate distributions
    print("Calculating HPRD distribution (bin size: 0.25, max: 8.0)...")
    hprd_distribution = calculate_distribution(hprd_values, 0.25, max_value=8.0)
    
    print("Calculating contract staffing distribution...")
    contract_distribution = calculate_contract_distribution(contract_values)
    
    # Calculate statistics
    hprd_mean = sum(hprd_values) / len(hprd_values) if hprd_values else 0
    hprd_median = sorted(hprd_values)[len(hprd_values) // 2] if hprd_values else 0
    
    contract_mean = sum(contract_values) / len(contract_values) if contract_values else 0
    contract_median = sorted(contract_values)[len(contract_values) // 2] if contract_values else 0
    
    # Create output data
    output = {
        'quarter': latest_quarter,
        'hprd': {
            'distribution': hprd_distribution,
            'mean': round(hprd_mean, 3),
            'median': round(hprd_median, 3),
            'count': len(hprd_values)
        },
        'contract': {
            'distribution': contract_distribution,
            'mean': round(contract_mean, 2),
            'median': round(contract_median, 2),
            'count': len(contract_values)
        }
    }
    
    # Save to JSON
    print(f"\nSaving distributions to playground_distributions.json...")
    with open('playground_distributions.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\nDistribution Summary:")
    print(f"Quarter: {latest_quarter}")
    print(f"\nHPRD Distribution:")
    print(f"  Mean: {hprd_mean:.3f}")
    print(f"  Median: {hprd_median:.3f}")
    print(f"  Facilities: {len(hprd_values):,}")
    print(f"  Bins: {len(hprd_distribution)}")
    
    print(f"\nContract Distribution:")
    print(f"  Mean: {contract_mean:.2f}%")
    print(f"  Median: {contract_median:.2f}%")
    print(f"  Facilities: {len(contract_values):,}")
    for bin_label, count in contract_distribution.items():
        print(f"  {bin_label}: {count:,} facilities ({count/len(contract_values)*100:.1f}%)")
    
    print("\nDone!")

if __name__ == '__main__':
    generate_distributions()

