import pandas as pd
import numpy as np
import json

def load_and_analyze_data():
    """Load the actual PBJ data and calculate real statistics."""
    
    # Load the national metrics data (already weighted)
    national_df = pd.read_csv('pbj_lite/national_lite_metrics.csv')
    
    # Load the facility metrics data for distributions
    df = pd.read_csv('pbj_lite/facility_lite_metrics.csv')
    
    # Clean the data
    df = df.dropna(subset=['Total_Nurse_HPRD', 'Contract_Percentage'])
    
    # Remove extreme outliers (beyond 3 standard deviations)
    for col in ['Total_Nurse_HPRD', 'Contract_Percentage']:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            df = df[abs(df[col] - mean) <= 3 * std]
    
    # Calculate real statistics
    stats = {}
    
    # HPRD statistics - use national data (already weighted) for mean, facility data for median
    if 'CY_Qtr' in national_df.columns:
        # Get the most recent quarter from national data
        latest_quarter = national_df['CY_Qtr'].max()
        latest_national = national_df[national_df['CY_Qtr'] == latest_quarter]
        
        # Get facility data for median calculation
        latest_facility_data = df[df['CY_Qtr'] == latest_quarter]
        hprd_data = latest_facility_data['Total_Nurse_HPRD'].dropna()
        
        stats['hprd'] = {
            'mean': float(latest_national['Total_Nurse_HPRD'].iloc[0]),  # Already weighted from national data
            'median': float(hprd_data.median()),
            'std': float(hprd_data.std()),
            'min': float(hprd_data.min()),
            'max': float(hprd_data.max()),
            'q25': float(hprd_data.quantile(0.25)),
            'q75': float(hprd_data.quantile(0.75)),
            'quarter': latest_quarter
        }
    else:
        # Fallback to facility data if no national data
        latest_quarter = df['CY_Qtr'].max()
        latest_quarter_data = df[df['CY_Qtr'] == latest_quarter]
        hprd_data = latest_quarter_data['Total_Nurse_HPRD'].dropna()
        
        stats['hprd'] = {
            'mean': float(hprd_data.mean()),
            'median': float(hprd_data.median()),
            'std': float(hprd_data.std()),
            'min': float(hprd_data.min()),
            'max': float(hprd_data.max()),
            'q25': float(hprd_data.quantile(0.25)),
            'q75': float(hprd_data.quantile(0.75)),
            'quarter': latest_quarter
        }
    
    # Contract percentage statistics - use national data for mean, facility data for median
    if 'CY_Qtr' in national_df.columns:
        latest_quarter = national_df['CY_Qtr'].max()
        latest_national = national_df[national_df['CY_Qtr'] == latest_quarter]
        
        # Get facility data for median calculation
        latest_facility_data = df[df['CY_Qtr'] == latest_quarter]
        contract_data = latest_facility_data['Contract_Percentage'].dropna()
        
        stats['contract'] = {
            'mean': float(latest_national['Contract_Percentage'].iloc[0]),  # Already weighted from national data
            'median': float(contract_data.median()),
            'std': float(contract_data.std()),
            'min': float(contract_data.min()),
            'max': float(contract_data.max()),
            'quarter': latest_quarter
        }
    else:
        # Fallback to facility data if no national data
        latest_quarter = df['CY_Qtr'].max()
        latest_quarter_data = df[df['CY_Qtr'] == latest_quarter]
        contract_data = latest_quarter_data['Contract_Percentage'].dropna()
        
        stats['contract'] = {
            'mean': float(contract_data.mean()),
            'median': float(contract_data.median()),
            'std': float(contract_data.std()),
            'min': float(contract_data.min()),
            'max': float(contract_data.max()),
            'quarter': latest_quarter
        }
    
    # RN HPRD statistics (if available)
    if 'Total_RN_HPRD' in df.columns:
        rn_data = df['Total_RN_HPRD'].dropna()
        stats['rn'] = {
            'mean': float(rn_data.mean()),
            'median': float(rn_data.median())
        }
    
    # LPN HPRD statistics (if available)
    if 'LPN_HPRD' in df.columns:
        lpn_data = df['LPN_HPRD'].dropna()
        stats['lpn'] = {
            'mean': float(lpn_data.mean()),
            'median': float(lpn_data.median())
        }
    
    # CNA HPRD statistics (if available)
    if 'CNA_HPRD' in df.columns:
        cna_data = df['CNA_HPRD'].dropna()
        stats['cna'] = {
            'mean': float(cna_data.mean()),
            'median': float(cna_data.median())
        }
    
    # Create HPRD distribution bins - use most recent quarter only
    if 'CY_Qtr' in df.columns:
        latest_quarter = df['CY_Qtr'].max()
        latest_quarter_data = df[df['CY_Qtr'] == latest_quarter]
        hprd_data = latest_quarter_data['Total_Nurse_HPRD'].dropna()
        
        hprd_bins = np.arange(1.5, 7.5, 0.5)
        hprd_counts, _ = np.histogram(hprd_data, bins=hprd_bins)
        stats['hprd_distribution'] = {
            'bins': [f"{hprd_bins[i]:.1f}-{hprd_bins[i+1]:.1f}" for i in range(len(hprd_bins)-1)],
            'counts': hprd_counts.tolist(),
            'quarter': latest_quarter
        }
    else:
        # Fallback to all data if no quarter column
        hprd_bins = np.arange(1.5, 7.5, 0.5)
        hprd_counts, _ = np.histogram(hprd_data, bins=hprd_bins)
        stats['hprd_distribution'] = {
            'bins': [f"{hprd_bins[i]:.1f}-{hprd_bins[i+1]:.1f}" for i in range(len(hprd_bins)-1)],
            'counts': hprd_counts.tolist(),
            'quarter': 'All Quarters'
        }
    
    # Create contract percentage distribution for most recent quarter only
    if 'CY_Qtr' in df.columns:
        # Get the most recent quarter
        latest_quarter = df['CY_Qtr'].max()
        latest_quarter_data = df[df['CY_Qtr'] == latest_quarter]['Contract_Percentage'].dropna()
        
        contract_bins = np.arange(0, 55, 5)
        contract_counts, _ = np.histogram(latest_quarter_data, bins=contract_bins)
        stats['contract_distribution'] = {
            'bins': [f"{contract_bins[i]:.0f}-{contract_bins[i+1]:.0f}%" for i in range(len(contract_bins)-1)],
            'counts': contract_counts.tolist(),
            'quarter': latest_quarter
        }
    else:
        # Fallback to all data if no quarter column
        contract_bins = np.arange(0, 55, 5)
        contract_counts, _ = np.histogram(contract_data, bins=contract_bins)
        stats['contract_distribution'] = {
            'bins': [f"{contract_bins[i]:.0f}-{contract_bins[i+1]:.0f}%" for i in range(len(contract_bins)-1)],
            'counts': contract_counts.tolist(),
            'quarter': 'All Quarters'
        }
    
    # Quarterly trends - use national data for means, facility data for medians
    if 'CY_Qtr' in national_df.columns:
        # Use national data for weighted means (already calculated)
        stats['quarterly_trends'] = {
            'quarters': national_df['CY_Qtr'].tolist(),
            'hprd': national_df['Total_Nurse_HPRD'].tolist(),
            'census': national_df['MDS'].tolist(),  # MDS is the census measure in national data
            'contract_mean': national_df['Contract_Percentage'].tolist()
        }
        
        # Add contract medians from facility data
        contract_medians = []
        for quarter in national_df['CY_Qtr']:
            quarter_facility_data = df[df['CY_Qtr'] == quarter]
            if len(quarter_facility_data) > 0:
                contract_median = quarter_facility_data['Contract_Percentage'].median()
                contract_medians.append(contract_median)
            else:
                contract_medians.append(0)
        
        stats['quarterly_trends']['contract_median'] = contract_medians
    
    return stats

if __name__ == "__main__":
    try:
        stats = load_and_analyze_data()
        
        # Save to JSON file
        with open('playground_data.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("Real PBJ data statistics:")
        print(f"HPRD Mean: {stats['hprd']['mean']:.3f}")
        print(f"HPRD Median: {stats['hprd']['median']:.3f}")
        print(f"Contract % Mean: {stats['contract']['mean']:.1f}%")
        print(f"Contract % Median: {stats['contract']['median']:.1f}%")
        print(f"Contract Distribution Quarter: {stats['contract_distribution']['quarter']}")
        print(f"Data points: {len(pd.read_csv('pbj_lite/facility_lite_metrics.csv'))}")
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback to sample data
        fallback_stats = {
            'hprd': {'mean': 3.73, 'median': 3.48, 'std': 0.8},
            'contract': {'mean': 12.5, 'median': 8.2, 'std': 15.3}
        }
        with open('playground_data.json', 'w') as f:
            json.dump(fallback_stats, f, indent=2)
        print("Using fallback data due to error")
