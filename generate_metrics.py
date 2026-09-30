import pandas as pd
import duckdb
import os
import glob
import time
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
        
        # Validate PROVNUMs
        invalid_provnums = df[~df['PROVNUM'].str.match(r'^\d{6}$')]
        if not invalid_provnums.empty:
            print("\nWarning: Found invalid PROVNUMs:")
            print(invalid_provnums[['PROVNUM', 'PROVNAME', 'STATE']].to_string())
        
        return True
    return False

def generate_metrics(*, output_dir=None, control_root=None):
    """Build governed metrics in versioned staging, never over served files."""
    start_time = time.time()

    from derived_provenance import pending_build_directory, pending_upstream_provenance

    output_root = (
        Path(output_dir).resolve()
        if output_dir is not None
        else pending_build_directory("cms.pbj_nurse_staffing", root=control_root)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    upstream_overrides = pending_upstream_provenance(
        "cms.pbj_nurse_staffing", root=control_root
    )
    
    try:
        # Connect to in-memory DuckDB with optimized settings
        conn = duckdb.connect(':memory:')
        conn.execute("PRAGMA threads=8")  # Increase threads
        conn.execute("PRAGMA memory_limit='8GB'")  # Increase memory limit
        conn.execute("PRAGMA preserve_insertion_order=false")
        conn.execute("PRAGMA enable_progress_bar=false")  # Disable progress bar
        
        # Get list of PBJ CSV files
        pbj_files = glob.glob('standardized_PBJ/*.csv')
        if not pbj_files:
            print("ERROR: No PBJ CSV files found in standardized_PBJ directory")
            return
        
        print(f"Found {len(pbj_files)} PBJ files")
        
        # Get unique quarters
        print("Getting quarters...")
        quarters_query = """
        SELECT DISTINCT CY_Qtr 
        FROM read_csv_auto('standardized_PBJ/*.csv', 
            header=True,
            ignore_errors=True,
            union_by_name=True,
            types={'PROVNUM': 'VARCHAR'}
        )
        ORDER BY CY_Qtr
        """
        quarters = [q[0] for q in conn.execute(quarters_query).fetchall()]
        print(f"Found {len(quarters)} quarters")
        
        # Process each quarter
        all_facility_metrics = []
        all_state_metrics = []
        all_national_metrics = []
        
        for quarter in quarters:
            print(f"Processing {quarter}...")
            
            # First, get all unique facilities for this quarter
            unique_facilities_query = f"""
            SELECT DISTINCT 
                PROVNUM,
                PROVNAME,
                STATE,
                COUNTY_NAME
            FROM read_csv_auto('standardized_PBJ/*.csv', 
                header=True,
                ignore_errors=True,
                union_by_name=True,
                types={{'PROVNUM': 'VARCHAR'}}
            )
            WHERE CY_Qtr = '{quarter}'
            """
            unique_facilities = conn.execute(unique_facilities_query).df()
            
            # Get facility metrics for this quarter
            facility_query = f"""
            WITH daily_metrics AS (
                SELECT 
                    PROVNUM,
                    PROVNAME,
                    STATE,
                    COUNTY_NAME,
                    WorkDate,
                    MDScensus,
                    (Hrs_RNDON + Hrs_RNadmin + Hrs_RN + 
                     Hrs_LPNadmin + Hrs_LPN + Hrs_CNA + 
                     Hrs_NAtrn + Hrs_MedAide) as Total_Nurse_Hours,
                    (Hrs_RNDON + Hrs_RNadmin + Hrs_RN) as RN_Hours,
                    (Hrs_RN + Hrs_LPN + Hrs_CNA + 
                     Hrs_NAtrn + Hrs_MedAide) as Nurse_Care_Hours,
                    (Hrs_RN) as RN_Care_Hours,
                    (Hrs_CNA + Hrs_NAtrn + Hrs_MedAide) as Nurse_Assistant_Hours,
                    (COALESCE(Hrs_RNDON_ctr, 0) + COALESCE(Hrs_RNadmin_ctr, 0) + COALESCE(Hrs_RN_ctr, 0) +
                     COALESCE(Hrs_LPNadmin_ctr, 0) + COALESCE(Hrs_LPN_ctr, 0) +
                     COALESCE(Hrs_CNA_ctr, 0) + COALESCE(Hrs_NAtrn_ctr, 0) + COALESCE(Hrs_MedAide_ctr, 0)) as Contract_Hours
                FROM read_csv_auto('standardized_PBJ/*.csv', 
                    header=True,
                    ignore_errors=True,
                    union_by_name=True,
                    types={{'PROVNUM': 'VARCHAR'}}
                )
                WHERE CY_Qtr = '{quarter}'
            )
            SELECT
                u.PROVNUM,
                u.PROVNAME,
                u.STATE,
                u.COUNTY_NAME,
                '{quarter}' as CY_Qtr,
                COUNT(DISTINCT d.WorkDate) as days_reported,
                SUM(d.MDScensus) as total_resident_days,
                AVG(d.MDScensus) as avg_daily_census,
                SUM(d.MDScensus) * 1.0 / NULLIF(COUNT(DISTINCT d.WorkDate), 0) as MDScensus,
                SUM(d.Total_Nurse_Hours) as Total_Nurse_Hours,
                SUM(d.RN_Hours) as Total_RN_Hours,
                SUM(d.Nurse_Care_Hours) as Total_Nurse_Care_Hours,
                SUM(d.RN_Care_Hours) as Total_RN_Care_Hours,
                SUM(d.Nurse_Assistant_Hours) as Total_Nurse_Assistant_Hours,
                SUM(d.Contract_Hours) as Total_Contract_Hours,
                SUM(d.Total_Nurse_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0) as Total_Nurse_HPRD,
                SUM(d.RN_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0) as RN_HPRD,
                SUM(d.Nurse_Care_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0) as Nurse_Care_HPRD,
                SUM(d.RN_Care_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0) as RN_Care_HPRD,
                SUM(d.Nurse_Assistant_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0) as Nurse_Assistant_HPRD,
                SUM(d.Contract_Hours) * 100.0 / NULLIF(SUM(d.Total_Nurse_Hours), 0) as Contract_Percentage
            FROM unique_facilities u
            LEFT JOIN daily_metrics d ON u.PROVNUM = d.PROVNUM
            GROUP BY u.PROVNUM, u.PROVNAME, u.STATE, u.COUNTY_NAME
            """
            facility_df = conn.execute(facility_query).df()
            
            # Format PROVNUMs in the facility dataframe
            format_provnum(facility_df)
            
            # Get state metrics for this quarter
            state_query = f"""
            SELECT
                STATE,
                '{quarter}' as CY_Qtr,
                COUNT(DISTINCT PROVNUM) as facility_count,
                ROUND(AVG(days_reported), 1) as avg_days_reported,
                SUM(total_resident_days) as total_resident_days,
                ROUND(AVG(avg_daily_census), 1) as avg_daily_census,
                ROUND(SUM(total_resident_days) * 1.0 / NULLIF(AVG(days_reported), 0), 1) as MDScensus,
                SUM(Total_Nurse_Hours) as Total_Nurse_Hours,
                SUM(Total_RN_Hours) as Total_RN_Hours,
                SUM(Total_Nurse_Care_Hours) as Total_Nurse_Care_Hours,
                SUM(Total_RN_Care_Hours) as Total_RN_Care_Hours,
                SUM(Total_Nurse_Assistant_Hours) as Total_Nurse_Assistant_Hours,
                SUM(Total_Contract_Hours) as Total_Contract_Hours,
                SUM(Total_Nurse_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as Total_Nurse_HPRD,
                SUM(Total_RN_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as RN_HPRD,
                SUM(Total_Nurse_Care_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as Nurse_Care_HPRD,
                SUM(Total_RN_Care_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as RN_Care_HPRD,
                SUM(Total_Nurse_Assistant_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as Nurse_Assistant_HPRD,
                SUM(Total_Contract_Hours) * 100.0 / NULLIF(SUM(Total_Nurse_Hours), 0) as Contract_Percentage,
                SUM(Total_Nurse_Care_Hours) * 100.0 / NULLIF(SUM(Total_Nurse_Hours), 0) as Direct_Care_Percentage,
                SUM(Total_RN_Hours) * 100.0 / NULLIF(SUM(Total_Nurse_Hours), 0) as Total_RN_Percentage,
                SUM(Total_Nurse_Assistant_Hours) * 100.0 / NULLIF(SUM(Total_Nurse_Hours), 0) as Nurse_Aide_Percentage
            FROM facility_df
            GROUP BY STATE
            """
            state_df = conn.execute(state_query).df()
            
            # Get national metrics for this quarter
            national_query = f"""
            SELECT
                'NATIONAL' as STATE,
                '{quarter}' as CY_Qtr,
                COUNT(DISTINCT PROVNUM) as facility_count,
                ROUND(AVG(days_reported), 1) as avg_days_reported,
                SUM(total_resident_days) as total_resident_days,
                ROUND(AVG(avg_daily_census), 1) as avg_daily_census,
                ROUND(SUM(total_resident_days) * 1.0 / NULLIF(AVG(days_reported), 0), 1) as MDScensus,
                SUM(Total_Nurse_Hours) as Total_Nurse_Hours,
                SUM(Total_RN_Hours) as Total_RN_Hours,
                SUM(Total_Nurse_Care_Hours) as Total_Nurse_Care_Hours,
                SUM(Total_RN_Care_Hours) as Total_RN_Care_Hours,
                SUM(Total_Nurse_Assistant_Hours) as Total_Nurse_Assistant_Hours,
                SUM(Total_Contract_Hours) as Total_Contract_Hours,
                SUM(Total_Nurse_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as Total_Nurse_HPRD,
                SUM(Total_RN_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as RN_HPRD,
                SUM(Total_Nurse_Care_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as Nurse_Care_HPRD,
                SUM(Total_RN_Care_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as RN_Care_HPRD,
                SUM(Total_Nurse_Assistant_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0) as Nurse_Assistant_HPRD,
                SUM(Total_Contract_Hours) * 100.0 / NULLIF(SUM(Total_Nurse_Hours), 0) as Contract_Percentage
            FROM facility_df
            """
            national_df = conn.execute(national_query).df()
            
            all_facility_metrics.append(facility_df)
            all_state_metrics.append(state_df)
            all_national_metrics.append(national_df)
        
        # Combine all quarters
        facility_metrics = pd.concat(all_facility_metrics, ignore_index=True)
        state_metrics = pd.concat(all_state_metrics, ignore_index=True)
        national_metrics = pd.concat(all_national_metrics, ignore_index=True)
        
        # Sort metrics
        facility_metrics = facility_metrics.sort_values(['PROVNUM', 'CY_Qtr'])
        state_metrics = state_metrics.sort_values(['STATE', 'CY_Qtr'])
        national_metrics = national_metrics.sort_values('CY_Qtr')
        
        # Save to CSV files
        print("\nSaving data...")
        facility_path = output_root / 'facility_quarterly_metrics.csv'
        state_path = output_root / 'state_quarterly_metrics.csv'
        national_path = output_root / 'national_quarterly_metrics.csv'
        facility_metrics.to_csv(facility_path, index=False)
        state_metrics.to_csv(state_path, index=False)
        national_metrics.to_csv(national_path, index=False)

        # Record governed candidates only after all numerical outputs exist.
        # Promotion remains an explicit human action in Data Ops.
        from derived_provenance import record_validated_derived_candidate
        record_validated_derived_candidate(
            "pbj.benchmarks.state",
            state_path,
            builder="generate_metrics.py",
            root=control_root,
            upstream_overrides=upstream_overrides,
        )
        record_validated_derived_candidate(
            "pbj.benchmarks.national",
            national_path,
            builder="generate_metrics.py",
            root=control_root,
            upstream_overrides=upstream_overrides,
        )
        
        # Print summary
        print("\nData Summary:")
        print(f"Total facility records: {len(facility_metrics):,}")
        print(f"Total state records: {len(state_metrics):,}")
        print(f"Total national records: {len(national_metrics):,}")
        print(f"Unique facilities: {facility_metrics['PROVNUM'].nunique():,}")
        print(f"States with data: {len(state_metrics['STATE'].unique()):,}")
        print(f"Quarters processed: {len(quarters)}")
        
        # Print latest quarter summaries
        latest_quarter = quarters[-1]
        latest_states = state_metrics[state_metrics['CY_Qtr'] == latest_quarter]
        latest_national = national_metrics[national_metrics['CY_Qtr'] == latest_quarter]
        
        print(f"\nLatest Quarter ({latest_quarter}) Summary:")
        print(f"Facilities: {latest_states['facility_count'].sum():,}")
        print(f"National HPRD: {latest_national['Total_Nurse_HPRD'].iloc[0]:.3f}")
        print(f"National Contract %: {latest_national['Contract_Percentage'].iloc[0]:.1f}%")
        print(f"National MDScensus: {latest_national['MDScensus'].iloc[0]:.1f}")
        
        end_time = time.time()
        print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
        print(f"Governed candidate directory: {output_root}")
        return {
            "output_dir": output_root,
            "facility": facility_path,
            "state": state_path,
            "national": national_path,
        }
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        # Clean up
        if 'conn' in locals():
            conn.close()

if __name__ == '__main__':
    generate_metrics()
