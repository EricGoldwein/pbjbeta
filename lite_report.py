import pandas as pd
import os
from pathlib import Path

def format_provnum(df):
    """Ensure PROVNUM is a 6-digit string with leading zeros."""
    if 'PROVNUM' in df.columns:
        # First convert to string
        df['PROVNUM'] = df['PROVNUM'].astype(str)
        # Remove any scientific notation
        df['PROVNUM'] = df['PROVNUM'].apply(lambda x: x.split('e')[0] if 'e' in x.lower() else x)
        # Remove any decimal points and everything after
        df['PROVNUM'] = df['PROVNUM'].apply(lambda x: x.split('.')[0])
        # Pad with leading zeros
        df['PROVNUM'] = df['PROVNUM'].str.zfill(6)
        return True
    return False

def validate_quarterly_metrics_structure(facility_metrics, state_metrics, national_metrics):
    """Validate that quarterly metrics files have expected structure."""
    warnings = []
    
    # Check facility metrics
    required_facility_cols = ['CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 'Total_Nurse_HPRD', 'RN_HPRD', 'Contract_Percentage', 'MDScensus']
    missing_facility = [col for col in required_facility_cols if col not in facility_metrics.columns]
    if missing_facility:
        warnings.append(f"⚠️  WARNING: Missing required columns in facility_quarterly_metrics.csv: {missing_facility}")
    
    # Check state metrics
    required_state_cols = ['CY_Qtr', 'STATE', 'Total_Nurse_Hours', 'Total_Nurse_HPRD', 'facility_count']
    missing_state = [col for col in required_state_cols if col not in state_metrics.columns]
    if missing_state:
        warnings.append(f"⚠️  WARNING: Missing required columns in state_quarterly_metrics.csv: {missing_state}")
    
    # Check national metrics
    required_national_cols = ['CY_Qtr', 'STATE', 'Total_Nurse_Hours', 'Total_Nurse_HPRD', 'facility_count']
    missing_national = [col for col in required_national_cols if col not in national_metrics.columns]
    if missing_national:
        warnings.append(f"⚠️  WARNING: Missing required columns in national_quarterly_metrics.csv: {missing_national}")
    
    return warnings

def generate_lite_metrics(*, input_dir=None, output_dir=None, control_root=None):
    """Generate a governed peer candidate without overwriting served files."""
    from derived_provenance import pending_build_directory, pending_upstream_provenance

    candidate_root = pending_build_directory(
        "cms.pbj_nurse_staffing", root=control_root
    )
    input_root = Path(input_dir).resolve() if input_dir is not None else candidate_root
    output_root = Path(output_dir).resolve() if output_dir is not None else candidate_root
    output_root.mkdir(parents=True, exist_ok=True)
    upstream_overrides = pending_upstream_provenance(
        "cms.pbj_nurse_staffing", root=control_root
    )
    print("="*70)
    print("Generating Lite Metrics from Quarterly Metrics")
    print("="*70)
    
    # Check if quarterly metrics files exist
    if not (input_root / 'facility_quarterly_metrics.csv').is_file():
        print("ERROR: facility_quarterly_metrics.csv not found!")
        print("Please run generate_metrics.py first.")
        return
    
    if not (input_root / 'state_quarterly_metrics.csv').is_file():
        print("ERROR: state_quarterly_metrics.csv not found!")
        print("Please run generate_metrics.py first.")
        return
    
    if not (input_root / 'national_quarterly_metrics.csv').is_file():
        print("ERROR: national_quarterly_metrics.csv not found!")
        print("Please run generate_metrics.py first.")
        return
    
    print("\nLoading quarterly metrics files...")
    # Read the existing metrics files
    facility_metrics = pd.read_csv(input_root / 'facility_quarterly_metrics.csv', low_memory=False)
    state_metrics = pd.read_csv(input_root / 'state_quarterly_metrics.csv', low_memory=False)
    national_metrics_df = pd.read_csv(input_root / 'national_quarterly_metrics.csv', low_memory=False)
    
    print(f"  Loaded {len(facility_metrics):,} facility records")
    print(f"  Loaded {len(state_metrics):,} state records")
    print(f"  Loaded {len(national_metrics_df):,} national records")
    
    # Validate structure
    warnings = validate_quarterly_metrics_structure(facility_metrics, state_metrics, national_metrics_df)
    if warnings:
        print("\n⚠️  WARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
        print("\n⚠️  Please review warnings above - file structure may have changed!")
    
    # Get quarter information
    quarters = sorted(facility_metrics['CY_Qtr'].unique())
    print(f"\nQuarters in dataset: {len(quarters)}")
    print(f"  Range: {quarters[0]} to {quarters[-1]}")
    
    # Format PROVNUMs
    format_provnum(facility_metrics)
    
    # Create facility lite metrics with existing columns and calculate missing ones
    facility_lite = facility_metrics[[
        'CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME',
        'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'RN_HPRD', 'RN_Care_HPRD', 'Contract_Percentage', 'Total_Contract_Hours', 'Total_Nurse_Hours', 'Total_Nurse_Care_Hours', 'Total_RN_Hours', 'Total_RN_Care_Hours', 'avg_daily_census',
        'total_resident_days', 'days_reported', 'MDScensus'
    ]].copy()
    
    # Use the existing RN_HPRD values for facilities - these are already calculated correctly
    facility_lite['Total_RN_HPRD'] = facility_lite['RN_HPRD']  # Total RN HPRD from existing RN_HPRD
    facility_lite['Direct_Care_RN_HPRD'] = facility_lite['RN_Care_HPRD']  # Direct Care RN HPRD from existing RN_Care_HPRD
    
    # Handle division by zero
    facility_lite['Total_RN_HPRD'] = facility_lite['Total_RN_HPRD'].fillna(0)
    facility_lite['Direct_Care_RN_HPRD'] = facility_lite['Direct_Care_RN_HPRD'].fillna(0)
    
    # Sort by quarter and PROVNUM
    facility_lite = facility_lite.sort_values(['CY_Qtr', 'PROVNUM'])
    
    # Calculate state lite metrics with proper weighted averages
    state_data = []
    for quarter in facility_lite['CY_Qtr'].unique():
        quarter_facilities = facility_lite[facility_lite['CY_Qtr'] == quarter]
        for state in quarter_facilities['STATE'].unique():
            state_facilities = quarter_facilities[quarter_facilities['STATE'] == state]
            
            # Calculate weighted averages
            total_census = state_facilities['avg_daily_census'].sum()
            total_nurse_hours = state_facilities['Total_Nurse_Hours'].sum()
            total_nurse_care_hours = state_facilities['Total_Nurse_Care_Hours'].sum()
            total_rn_hours = state_facilities['Total_RN_Hours'].sum()
            total_rn_care_hours = state_facilities['Total_RN_Care_Hours'].sum()
            total_contract_hours = state_facilities['Total_Contract_Hours'].sum()
            
            # Calculate HPRD values (weighted by census) 
            # Use total_resident_days as denominator since hours are quarterly totals
            total_resident_days = state_facilities['total_resident_days'].sum()
            total_nurse_hprd = total_nurse_hours / total_resident_days if total_resident_days > 0 else 0
            nurse_care_hprd = total_nurse_care_hours / total_resident_days if total_resident_days > 0 else 0
            total_rn_hprd = total_rn_hours / total_resident_days if total_resident_days > 0 else 0
            direct_care_rn_hprd = total_rn_care_hours / total_resident_days if total_resident_days > 0 else 0
            contract_percentage = (total_contract_hours / total_nurse_hours * 100) if total_nurse_hours > 0 else 0
            
            state_data.append({
                'CY_Qtr': quarter,
                'STATE': state,
                'facility_count': len(state_facilities),
                'avg_daily_census': state_facilities['avg_daily_census'].mean(),
                'Total_Nurse_HPRD': total_nurse_hprd,
                'Nurse_Care_HPRD': nurse_care_hprd,
                'Total_RN_HPRD': total_rn_hprd,
                'Direct_Care_RN_HPRD': direct_care_rn_hprd,
                'Contract_Percentage': contract_percentage,
                'avg_state_census': total_census
            })
    
    state_lite = pd.DataFrame(state_data)
    
    # Sort by state first, then quarter (AK 2017Q1, AK 2017Q2, etc.)
    state_lite = state_lite.sort_values(['STATE', 'CY_Qtr'])
    
    # Calculate national metrics with proper weighted averages
    national_metrics = []
    for quarter in facility_lite['CY_Qtr'].unique():
        quarter_facilities = facility_lite[facility_lite['CY_Qtr'] == quarter]
        
        # Calculate weighted averages nationally
        total_census = quarter_facilities['avg_daily_census'].sum()
        total_nurse_hours = quarter_facilities['Total_Nurse_Hours'].sum()
        total_nurse_care_hours = quarter_facilities['Total_Nurse_Care_Hours'].sum()
        total_rn_hours = quarter_facilities['Total_RN_Hours'].sum()
        total_rn_care_hours = quarter_facilities['Total_RN_Care_Hours'].sum()
        total_contract_hours = quarter_facilities['Total_Contract_Hours'].sum()
        
        # Calculate HPRD values (weighted by census)
        # Use total_resident_days as denominator since hours are quarterly totals
        total_resident_days = quarter_facilities['total_resident_days'].sum()
        total_nurse_hprd = total_nurse_hours / total_resident_days if total_resident_days > 0 else 0
        nurse_care_hprd = total_nurse_care_hours / total_resident_days if total_resident_days > 0 else 0
        total_rn_hprd = total_rn_hours / total_resident_days if total_resident_days > 0 else 0
        direct_care_rn_hprd = total_rn_care_hours / total_resident_days if total_resident_days > 0 else 0
        contract_percentage = (total_contract_hours / total_nurse_hours * 100) if total_nurse_hours > 0 else 0
        
        national_metrics.append({
            'CY_Qtr': quarter,
            'Facility_Count': len(quarter_facilities),
            'Total_Nurse_HPRD': total_nurse_hprd,
            'Nurse_Care_HPRD': nurse_care_hprd,
            'Total_RN_HPRD': total_rn_hprd,
            'Direct_Care_RN_HPRD': direct_care_rn_hprd,
            'Contract_Percentage': contract_percentage,
            'MDS': total_census
        })
    
    national_lite = pd.DataFrame(national_metrics)
    national_lite = national_lite.sort_values('CY_Qtr')
    
    # Create the output dataframe with correct column order
    facility_lite_output = facility_lite[[
        'CY_Qtr', 'PROVNUM', 'PROVNAME', 'STATE', 'COUNTY_NAME',
        'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Direct_Care_RN_HPRD', 'Contract_Percentage', 'MDScensus'
    ]].copy()
    
    # Rename the MDScensus column to Census
    facility_lite_output = facility_lite_output.rename(columns={'MDScensus': 'Census'})
    
    state_lite.columns = [
        'CY_Qtr', 'STATE', 'Facility_Count', 'Census',
        'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Direct_Care_RN_HPRD', 'Contract_Percentage', 'State_Census'
    ]
    
    national_lite.columns = [
        'CY_Qtr', 'Facility_Count', 'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Direct_Care_RN_HPRD',
        'Contract_Percentage', 'MDS'
    ]
    
    # Save the lite metrics files to root directory
    facility_lite_output.to_csv(output_root / 'facility_lite_metrics.csv', index=False)
    state_lite.to_csv(output_root / 'state_lite_metrics.csv', index=False)
    national_lite.to_csv(output_root / 'national_lite_metrics.csv', index=False)
    
    # Also save to pbj_lite directory (dashboard checks this first)
    pbj_lite_dir = output_root / 'pbj_lite'
    pbj_lite_dir.mkdir(parents=True, exist_ok=True)
    facility_lite_output.to_csv(pbj_lite_dir / 'facility_lite_metrics.csv', index=False)
    state_lite.to_csv(pbj_lite_dir / 'state_lite_metrics.csv', index=False)
    national_lite.to_csv(pbj_lite_dir / 'national_lite_metrics.csv', index=False)

    # Peer distribution is the governed facility-lite artifact. Capture exact
    # ACTIVE nurse provenance without changing the numerical output.
    from derived_provenance import record_validated_derived_candidate
    record_validated_derived_candidate(
        "pbj.peer_distribution",
        pbj_lite_dir / "facility_lite_metrics.csv",
        builder="lite_report.py",
        root=control_root,
        upstream_overrides=upstream_overrides,
    )
    
    # Print summary
    print(f"\n{'='*70}")
    print("Lite Metrics Generation Summary:")
    print(f"{'='*70}")
    print(f"Total facility records: {len(facility_lite):,}")
    print(f"Total state records: {len(state_lite):,}")
    print(f"Total national records: {len(national_lite):,}")
    print(f"Unique facilities: {facility_lite['PROVNUM'].nunique():,}")
    print(f"Quarters processed: {len(quarters)}")
    
    # Print latest quarter's state summary
    if not state_lite.empty:
        latest_quarter = state_lite['CY_Qtr'].max()
        print(f"\nLatest quarter ({latest_quarter}) state summary:")
        latest_state = state_lite[state_lite['CY_Qtr'] == latest_quarter].sort_values('STATE')
        print(latest_state[['STATE', 'Facility_Count', 'Census', 'State_Census', 'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Contract_Percentage']].to_string())
        
        # Print latest quarter's national summary
        print(f"\nLatest quarter ({latest_quarter}) national summary:")
        latest_national = national_lite[national_lite['CY_Qtr'] == latest_quarter]
        if not latest_national.empty:
            print(latest_national[['Facility_Count', 'Total_Nurse_HPRD', 'Nurse_Care_HPRD', 'Total_RN_HPRD', 'Contract_Percentage', 'MDS']].to_string())
    
    print(f"\n✓ Lite metrics files generated successfully!")
    print(f"  - facility_lite_metrics.csv (root and pbj_lite/)")
    print(f"  - state_lite_metrics.csv (root and pbj_lite/)")
    print(f"  - national_lite_metrics.csv (root and pbj_lite/)")
    print(f"  - governed candidate directory: {output_root}")
    return {
        "output_dir": output_root,
        "peer": pbj_lite_dir / "facility_lite_metrics.csv",
    }

if __name__ == "__main__":
    generate_lite_metrics() 
