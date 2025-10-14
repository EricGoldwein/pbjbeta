import pandas as pd
import duckdb
import os
import glob
import time

def format_provnum(df):
    """Ensure PROVNUM is a 6-character string with proper formatting."""
    if 'PROVNUM' in df.columns:
        # First convert to string
        df['PROVNUM'] = df['PROVNUM'].astype(str)
        
        # Remove any scientific notation
        df['PROVNUM'] = df['PROVNUM'].apply(lambda x: x.split('e')[0] if 'e' in x.lower() else x)
        
        # Remove any decimal points and everything after
        df['PROVNUM'] = df['PROVNUM'].apply(lambda x: x.split('.')[0])
        
        # Handle special cases
        def format_provnum_value(x):
            # Remove any whitespace
            x = x.strip()
            
            # If it's a number, pad with leading zeros to 6 digits
            if x.replace('-', '').isdigit():
                return x.zfill(6)
            
            # If it contains letters, ensure it's uppercase and pad with leading zeros if needed
            if any(c.isalpha() for c in x):
                # Split into numeric and alphabetic parts
                numeric_part = ''.join(c for c in x if c.isdigit())
                alpha_part = ''.join(c for c in x if c.isalpha()).upper()
                
                # Pad numeric part with leading zeros
                numeric_part = numeric_part.zfill(6 - len(alpha_part))
                
                # Combine parts
                return numeric_part + alpha_part
            
            # For any other case, pad to 6 characters
            return x.zfill(6)
        
        # Apply the formatting
        df['PROVNUM'] = df['PROVNUM'].apply(format_provnum_value)
        
        # Validate PROVNUMs (now accepts letters and numbers)
        invalid_provnums = df[~df['PROVNUM'].str.match(r'^[A-Z0-9]{6}$')]
        if not invalid_provnums.empty:
            print("\nWarning: Found invalid PROVNUMs:")
            print(invalid_provnums[['PROVNUM', 'PROVNAME', 'STATE']].to_string())
        
        return True
    return False

def generate_non_nurse_metrics():
    """Generate non-nurse metrics at state and facility levels using DuckDB."""
    start_time = time.time()
    
    try:
        # Connect to in-memory DuckDB with optimized settings
        conn = duckdb.connect(':memory:')
        conn.execute("PRAGMA threads=8")  # Increase threads
        conn.execute("PRAGMA memory_limit='8GB'")  # Increase memory limit
        conn.execute("PRAGMA preserve_insertion_order=false")
        conn.execute("PRAGMA enable_progress_bar=false")  # Disable progress bar
        
        # Get list of non-nurse PBJ CSV files
        pbj_files = glob.glob('standardized_NonNurse/*.csv')
        if not pbj_files:
            print("ERROR: No non-nurse PBJ CSV files found in standardized_NonNurse directory")
            return
        
        print(f"Found {len(pbj_files)} non-nurse PBJ files")
        
        # Get unique quarters
        print("Getting quarters...")
        quarters_query = """
        SELECT DISTINCT CY_Qtr 
        FROM read_csv_auto('standardized_NonNurse/*.csv', 
            header=True,
            ignore_errors=True,
            union_by_name=True
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
            FROM read_csv_auto('standardized_NonNurse/*.csv', 
                header=True,
                ignore_errors=True,
                union_by_name=True
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
                    -- Administrative group
                    Hrs_Admin as Admin_Hours,
                    -- Medical Director group
                    Hrs_MedDir as MedDir_Hours,
                    -- Other MD group
                    Hrs_OthMD as OthMD_Hours,
                    -- Advanced Practice group
                    (Hrs_PA + Hrs_NP) as AdvPractice_Hours,
                    -- Clinical Nurse Specialist group
                    Hrs_ClinNrsSpec as ClinNrsSpec_Hours,
                    -- Pharmacy group
                    Hrs_Pharmacist as Pharmacy_Hours,
                    -- Dietary group
                    (Hrs_Dietician + Hrs_FeedAsst) as Dietary_Hours,
                    -- Therapy group
                    (Hrs_OT + Hrs_OTasst + Hrs_OTaide + 
                     Hrs_PT + Hrs_PTasst + Hrs_PTaide + 
                     Hrs_RespTher + Hrs_RespTech) as Therapy_Hours,
                    -- Speech Language group
                    Hrs_SpcLangPath as Speech_Hours,
                    -- Activity group
                    (Hrs_TherRecSpec + Hrs_QualActvProf + Hrs_OthActv) as Activity_Hours,
                    -- Social Work group
                    (Hrs_QualSocWrk + Hrs_OthSocWrk) as SocialWork_Hours,
                    -- Mental Health group
                    Hrs_MHSvc as MentalHealth_Hours,
                    -- Contract hours
                    (Hrs_Admin_ctr + Hrs_MedDir_ctr + Hrs_OthMD_ctr +
                     Hrs_PA_ctr + Hrs_NP_ctr + Hrs_ClinNrsSpec_ctr +
                     Hrs_Pharmacist_ctr + Hrs_Dietician_ctr + Hrs_FeedAsst_ctr +
                     Hrs_OT_ctr + Hrs_OTasst_ctr + Hrs_OTaide_ctr +
                     Hrs_PT_ctr + Hrs_PTasst_ctr + Hrs_PTaide_ctr +
                     Hrs_RespTher_ctr + Hrs_RespTech_ctr + Hrs_SpcLangPath_ctr +
                     Hrs_TherRecSpec_ctr + Hrs_QualActvProf_ctr + Hrs_OthActv_ctr +
                     Hrs_QualSocWrk_ctr + Hrs_OthSocWrk_ctr + Hrs_MHSvc_ctr) as Contract_Hours
                FROM read_csv_auto('standardized_NonNurse/*.csv', 
                    header=True,
                    ignore_errors=True,
                    union_by_name=True
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
                ROUND(AVG(d.MDScensus), 1) as avg_daily_census,
                ROUND(SUM(d.MDScensus) * 1.0 / NULLIF(COUNT(DISTINCT d.WorkDate), 0), 1) as MDScensus,
                -- Total hours
                SUM(d.Admin_Hours + d.MedDir_Hours + d.OthMD_Hours + 
                    d.AdvPractice_Hours + d.ClinNrsSpec_Hours + d.Pharmacy_Hours + 
                    d.Dietary_Hours + d.Therapy_Hours + d.Speech_Hours + 
                    d.Activity_Hours + d.SocialWork_Hours + d.MentalHealth_Hours) as Total_NonNurse_Hours,
                -- Individual category hours
                SUM(d.Admin_Hours) as Total_Admin_Hours,
                SUM(d.MedDir_Hours) as Total_MedDir_Hours,
                SUM(d.OthMD_Hours) as Total_OthMD_Hours,
                SUM(d.AdvPractice_Hours) as Total_AdvPractice_Hours,
                SUM(d.ClinNrsSpec_Hours) as Total_ClinNrsSpec_Hours,
                SUM(d.Pharmacy_Hours) as Total_Pharmacy_Hours,
                SUM(d.Dietary_Hours) as Total_Dietary_Hours,
                SUM(d.Therapy_Hours) as Total_Therapy_Hours,
                SUM(d.Speech_Hours) as Total_Speech_Hours,
                SUM(d.Activity_Hours) as Total_Activity_Hours,
                SUM(d.SocialWork_Hours) as Total_SocialWork_Hours,
                SUM(d.MentalHealth_Hours) as Total_MentalHealth_Hours,
                SUM(d.Contract_Hours) as Total_Contract_Hours,
                -- HPRD metrics
                ROUND(SUM(d.Admin_Hours + d.MedDir_Hours + d.OthMD_Hours + 
                         d.AdvPractice_Hours + d.ClinNrsSpec_Hours + d.Pharmacy_Hours + 
                         d.Dietary_Hours + d.Therapy_Hours + d.Speech_Hours + 
                         d.Activity_Hours + d.SocialWork_Hours + d.MentalHealth_Hours) * 1.0 / 
                     NULLIF(SUM(d.MDScensus), 0), 3) as Total_NonNurse_HPRD,
                ROUND(SUM(d.Admin_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as Admin_HPRD,
                ROUND(SUM(d.MedDir_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as MedDir_HPRD,
                ROUND(SUM(d.AdvPractice_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as AdvPractice_HPRD,
                ROUND(SUM(d.Pharmacy_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as Pharmacy_HPRD,
                ROUND(SUM(d.Dietary_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as Dietary_HPRD,
                ROUND(SUM(d.Therapy_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as Therapy_HPRD,
                ROUND(SUM(d.Speech_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as Speech_HPRD,
                ROUND(SUM(d.Activity_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as Activity_HPRD,
                ROUND(SUM(d.SocialWork_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as SocialWork_HPRD,
                ROUND(SUM(d.MentalHealth_Hours) * 1.0 / NULLIF(SUM(d.MDScensus), 0), 3) as MentalHealth_HPRD,
                -- Contract percentage
                ROUND(SUM(d.Contract_Hours) * 100.0 / 
                     NULLIF(SUM(d.Admin_Hours + d.MedDir_Hours + d.OthMD_Hours + 
                               d.AdvPractice_Hours + d.ClinNrsSpec_Hours + d.Pharmacy_Hours + 
                               d.Dietary_Hours + d.Therapy_Hours + d.Speech_Hours + 
                               d.Activity_Hours + d.SocialWork_Hours + d.MentalHealth_Hours), 0), 3) as Contract_Percentage
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
                SUM(Total_NonNurse_Hours) as Total_NonNurse_Hours,
                SUM(Total_Admin_Hours) as Total_Admin_Hours,
                SUM(Total_MedDir_Hours) as Total_MedDir_Hours,
                SUM(Total_OthMD_Hours) as Total_OthMD_Hours,
                SUM(Total_AdvPractice_Hours) as Total_AdvPractice_Hours,
                SUM(Total_ClinNrsSpec_Hours) as Total_ClinNrsSpec_Hours,
                SUM(Total_Pharmacy_Hours) as Total_Pharmacy_Hours,
                SUM(Total_Dietary_Hours) as Total_Dietary_Hours,
                SUM(Total_Therapy_Hours) as Total_Therapy_Hours,
                SUM(Total_Speech_Hours) as Total_Speech_Hours,
                SUM(Total_Activity_Hours) as Total_Activity_Hours,
                SUM(Total_SocialWork_Hours) as Total_SocialWork_Hours,
                SUM(Total_MentalHealth_Hours) as Total_MentalHealth_Hours,
                SUM(Total_Contract_Hours) as Total_Contract_Hours,
                ROUND(SUM(Total_NonNurse_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Total_NonNurse_HPRD,
                ROUND(SUM(Total_Admin_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Admin_HPRD,
                ROUND(SUM(Total_MedDir_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as MedDir_HPRD,
                ROUND(SUM(Total_AdvPractice_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as AdvPractice_HPRD,
                ROUND(SUM(Total_Pharmacy_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Pharmacy_HPRD,
                ROUND(SUM(Total_Dietary_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Dietary_HPRD,
                ROUND(SUM(Total_Therapy_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Therapy_HPRD,
                ROUND(SUM(Total_Speech_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Speech_HPRD,
                ROUND(SUM(Total_Activity_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Activity_HPRD,
                ROUND(SUM(Total_SocialWork_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as SocialWork_HPRD,
                ROUND(SUM(Total_MentalHealth_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as MentalHealth_HPRD,
                ROUND(SUM(Total_Contract_Hours) * 100.0 / NULLIF(SUM(Total_NonNurse_Hours), 0), 3) as Contract_Percentage
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
                SUM(Total_NonNurse_Hours) as Total_NonNurse_Hours,
                SUM(Total_Admin_Hours) as Total_Admin_Hours,
                SUM(Total_MedDir_Hours) as Total_MedDir_Hours,
                SUM(Total_OthMD_Hours) as Total_OthMD_Hours,
                SUM(Total_AdvPractice_Hours) as Total_AdvPractice_Hours,
                SUM(Total_ClinNrsSpec_Hours) as Total_ClinNrsSpec_Hours,
                SUM(Total_Pharmacy_Hours) as Total_Pharmacy_Hours,
                SUM(Total_Dietary_Hours) as Total_Dietary_Hours,
                SUM(Total_Therapy_Hours) as Total_Therapy_Hours,
                SUM(Total_Speech_Hours) as Total_Speech_Hours,
                SUM(Total_Activity_Hours) as Total_Activity_Hours,
                SUM(Total_SocialWork_Hours) as Total_SocialWork_Hours,
                SUM(Total_MentalHealth_Hours) as Total_MentalHealth_Hours,
                SUM(Total_Contract_Hours) as Total_Contract_Hours,
                ROUND(SUM(Total_NonNurse_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Total_NonNurse_HPRD,
                ROUND(SUM(Total_Admin_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Admin_HPRD,
                ROUND(SUM(Total_MedDir_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as MedDir_HPRD,
                ROUND(SUM(Total_AdvPractice_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as AdvPractice_HPRD,
                ROUND(SUM(Total_Pharmacy_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Pharmacy_HPRD,
                ROUND(SUM(Total_Dietary_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Dietary_HPRD,
                ROUND(SUM(Total_Therapy_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Therapy_HPRD,
                ROUND(SUM(Total_Speech_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Speech_HPRD,
                ROUND(SUM(Total_Activity_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as Activity_HPRD,
                ROUND(SUM(Total_SocialWork_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as SocialWork_HPRD,
                ROUND(SUM(Total_MentalHealth_Hours) * 1.0 / NULLIF(SUM(total_resident_days), 0), 3) as MentalHealth_HPRD,
                ROUND(SUM(Total_Contract_Hours) * 100.0 / NULLIF(SUM(Total_NonNurse_Hours), 0), 3) as Contract_Percentage
            FROM facility_df
            """
            national_df = conn.execute(national_query).df()
            
            all_facility_metrics.append(facility_df)
            all_state_metrics.append(state_df)
            all_national_metrics.append(national_df)
        
        # Combine all metrics
        facility_metrics = pd.concat(all_facility_metrics, ignore_index=True)
        state_metrics = pd.concat(all_state_metrics, ignore_index=True)
        national_metrics = pd.concat(all_national_metrics, ignore_index=True)
        
        # Save metrics to CSV files
        print("\nSaving metrics to CSV files...")
        facility_metrics.to_csv('non_nurse_facility_metrics.csv', index=False)
        state_metrics.to_csv('non_nurse_state_metrics.csv', index=False)
        national_metrics.to_csv('non_nurse_national_metrics.csv', index=False)
        
        # Print summary
        print("\nSummary of generated metrics:")
        print(f"Time period: {quarters[0]} to {quarters[-1]}")
        print(f"Number of quarters: {len(quarters)}")
        print(f"Number of facilities: {len(facility_metrics['PROVNUM'].unique())}")
        print(f"Number of states: {len(state_metrics['STATE'].unique())}")
        print("\nFiles generated:")
        print("- non_nurse_facility_metrics.csv")
        print("- non_nurse_state_metrics.csv")
        print("- non_nurse_national_metrics.csv")
        
        # Print execution time
        execution_time = time.time() - start_time
        print(f"\nExecution time: {execution_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    generate_non_nurse_metrics() 