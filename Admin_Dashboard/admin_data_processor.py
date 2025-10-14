import pandas as pd
import os
import glob
from pathlib import Path
import logging
import numpy as np
from datetime import datetime, date
import holidays

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CMS Regional coding mapping
REGION_MAPPING = {
    # Region 1 - Boston
    'CT': 'Region 1', 'ME': 'Region 1', 'MA': 'Region 1', 'NH': 'Region 1', 'RI': 'Region 1', 'VT': 'Region 1',
    
    # Region 2 - New York
    'NJ': 'Region 2', 'NY': 'Region 2', 'PR': 'Region 2', 'VI': 'Region 2',
    
    # Region 3 - Philadelphia
    'DE': 'Region 3', 'DC': 'Region 3', 'MD': 'Region 3', 'PA': 'Region 3', 'VA': 'Region 3', 'WV': 'Region 3',
    
    # Region 4 - Atlanta
    'AL': 'Region 4', 'FL': 'Region 4', 'GA': 'Region 4', 'KY': 'Region 4', 'MS': 'Region 4', 'NC': 'Region 4', 'SC': 'Region 4', 'TN': 'Region 4',
    
    # Region 5 - Chicago
    'IL': 'Region 5', 'IN': 'Region 5', 'MI': 'Region 5', 'MN': 'Region 5', 'OH': 'Region 5', 'WI': 'Region 5',
    
    # Region 6 - Dallas
    'AR': 'Region 6', 'LA': 'Region 6', 'NM': 'Region 6', 'OK': 'Region 6', 'TX': 'Region 6',
    
    # Region 7 - Kansas City
    'IA': 'Region 7', 'KS': 'Region 7', 'MO': 'Region 7', 'NE': 'Region 7',
    
    # Region 8 - Denver
    'CO': 'Region 8', 'MT': 'Region 8', 'ND': 'Region 8', 'SD': 'Region 8', 'UT': 'Region 8', 'WY': 'Region 8',
    
    # Region 9 - San Francisco
    'AZ': 'Region 9', 'CA': 'Region 9', 'HI': 'Region 9', 'NV': 'Region 9',
    
    # Region 10 - Seattle
    'AK': 'Region 10', 'ID': 'Region 10', 'OR': 'Region 10', 'WA': 'Region 10'
}

def get_holiday_info(start_year=2017, end_year=2025):
    """
    Generate holiday information for the specified date range.
    
    Args:
        start_year (int): Start year for holiday data
        end_year (int): End year for holiday data
        
    Returns:
        pd.DataFrame: DataFrame with holiday dates and information
    """
    logger.info(f"Generating holiday data for {start_year}-{end_year}")
    
    holiday_data = []
    
    # Add major federal holidays manually for now
    federal_holidays = [
        # New Year's Day
        {'name': 'New Year\'s Day', 'month': 1, 'day': 1, 'year': 2017},
        {'name': 'New Year\'s Day', 'month': 1, 'day': 1, 'year': 2018},
        {'name': 'New Year\'s Day', 'month': 1, 'day': 1, 'year': 2019},
        {'name': 'New Year\'s Day', 'month': 1, 'day': 1, 'year': 2020},
        {'name': 'New Year\'s Day', 'month': 1, 'day': 1, 'year': 2021},
        {'name': 'New Year\'s Day', 'month': 1, 'day': 1, 'year': 2022},
        {'name': 'New Year\'s Day', 'month': 1, 'day': 1, 'year': 2023},
        {'name': 'New Year\'s Day', 'month': 1, 'day': 1, 'year': 2024},
        {'name': 'New Year\'s Day', 'month': 1, 'day': 1, 'year': 2025},
        
        # Independence Day
        {'name': 'Independence Day', 'month': 7, 'day': 4, 'year': 2017},
        {'name': 'Independence Day', 'month': 7, 'day': 4, 'year': 2018},
        {'name': 'Independence Day', 'month': 7, 'day': 4, 'year': 2019},
        {'name': 'Independence Day', 'month': 7, 'day': 4, 'year': 2020},
        {'name': 'Independence Day', 'month': 7, 'day': 4, 'year': 2021},
        {'name': 'Independence Day', 'month': 7, 'day': 4, 'year': 2022},
        {'name': 'Independence Day', 'month': 7, 'day': 4, 'year': 2023},
        {'name': 'Independence Day', 'month': 7, 'day': 4, 'year': 2024},
        {'name': 'Independence Day', 'month': 7, 'day': 4, 'year': 2025},
        
        # Christmas Day
        {'name': 'Christmas Day', 'month': 12, 'day': 25, 'year': 2017},
        {'name': 'Christmas Day', 'month': 12, 'day': 25, 'year': 2018},
        {'name': 'Christmas Day', 'month': 12, 'day': 25, 'year': 2019},
        {'name': 'Christmas Day', 'month': 12, 'day': 25, 'year': 2020},
        {'name': 'Christmas Day', 'month': 12, 'day': 25, 'year': 2021},
        {'name': 'Christmas Day', 'month': 12, 'day': 25, 'year': 2022},
        {'name': 'Christmas Day', 'month': 12, 'day': 25, 'year': 2023},
        {'name': 'Christmas Day', 'month': 12, 'day': 25, 'year': 2024},
        {'name': 'Christmas Day', 'month': 12, 'day': 25, 'year': 2025},
        
        # Thanksgiving Day (approximate dates)
        {'name': 'Thanksgiving Day', 'month': 11, 'day': 23, 'year': 2017},
        {'name': 'Thanksgiving Day', 'month': 11, 'day': 22, 'year': 2018},
        {'name': 'Thanksgiving Day', 'month': 11, 'day': 28, 'year': 2019},
        {'name': 'Thanksgiving Day', 'month': 11, 'day': 26, 'year': 2020},
        {'name': 'Thanksgiving Day', 'month': 11, 'day': 25, 'year': 2021},
        {'name': 'Thanksgiving Day', 'month': 11, 'day': 24, 'year': 2022},
        {'name': 'Thanksgiving Day', 'month': 11, 'day': 23, 'year': 2023},
        {'name': 'Thanksgiving Day', 'month': 11, 'day': 28, 'year': 2024},
        {'name': 'Thanksgiving Day', 'month': 11, 'day': 27, 'year': 2025},
    ]
    
    # Add federal holidays
    for holiday in federal_holidays:
        try:
            date_obj = date(holiday['year'], holiday['month'], holiday['day'])
            holiday_data.append({
                'date': date_obj,
                'holiday_name': holiday['name'],
                'year': holiday['year'],
                'month': holiday['month'],
                'day': holiday['day'],
                'day_of_week': date_obj.strftime('%A'),
                'is_federal_holiday': True
            })
        except ValueError:
            continue
    
    # Add additional important dates that might affect nursing home operations
    additional_holidays = [
        # Major religious holidays (approximate dates)
        {'name': 'Easter Sunday', 'month': 4, 'day': 16, 'year': 2017},
        {'name': 'Easter Sunday', 'month': 4, 'day': 1, 'year': 2018},
        {'name': 'Easter Sunday', 'month': 4, 'day': 21, 'year': 2019},
        {'name': 'Easter Sunday', 'month': 4, 'day': 12, 'year': 2020},
        {'name': 'Easter Sunday', 'month': 4, 'day': 4, 'year': 2021},
        {'name': 'Easter Sunday', 'month': 4, 'day': 17, 'year': 2022},
        {'name': 'Easter Sunday', 'month': 4, 'day': 9, 'year': 2023},
        {'name': 'Easter Sunday', 'month': 3, 'day': 31, 'year': 2024},
        {'name': 'Easter Sunday', 'month': 4, 'day': 20, 'year': 2025},
        
        # Super Bowl Sunday (first Sunday in February)
        {'name': 'Super Bowl Sunday', 'month': 2, 'day': 5, 'year': 2017},
        {'name': 'Super Bowl Sunday', 'month': 2, 'day': 4, 'year': 2018},
        {'name': 'Super Bowl Sunday', 'month': 2, 'day': 3, 'year': 2019},
        {'name': 'Super Bowl Sunday', 'month': 2, 'day': 2, 'year': 2020},
        {'name': 'Super Bowl Sunday', 'month': 2, 'day': 7, 'year': 2021},
        {'name': 'Super Bowl Sunday', 'month': 2, 'day': 13, 'year': 2022},
        {'name': 'Super Bowl Sunday', 'month': 2, 'day': 12, 'year': 2023},
        {'name': 'Super Bowl Sunday', 'month': 2, 'day': 11, 'year': 2024},
        {'name': 'Super Bowl Sunday', 'month': 2, 'day': 9, 'year': 2025},
        
        # Black Friday (day after Thanksgiving)
        {'name': 'Black Friday', 'month': 11, 'day': 24, 'year': 2017},
        {'name': 'Black Friday', 'month': 11, 'day': 23, 'year': 2018},
        {'name': 'Black Friday', 'month': 11, 'day': 29, 'year': 2019},
        {'name': 'Black Friday', 'month': 11, 'day': 27, 'year': 2020},
        {'name': 'Black Friday', 'month': 11, 'day': 26, 'year': 2021},
        {'name': 'Black Friday', 'month': 11, 'day': 25, 'year': 2022},
        {'name': 'Black Friday', 'month': 11, 'day': 24, 'year': 2023},
        {'name': 'Black Friday', 'month': 11, 'day': 29, 'year': 2024},
        {'name': 'Black Friday', 'month': 11, 'day': 28, 'year': 2025},
    ]
    
    for holiday in additional_holidays:
        try:
            date_obj = date(holiday['year'], holiday['month'], holiday['day'])
            holiday_data.append({
                'date': date_obj,
                'holiday_name': holiday['name'],
                'year': holiday['year'],
                'month': holiday['month'],
                'day': holiday['day'],
                'day_of_week': date_obj.strftime('%A'),
                'is_federal_holiday': False
            })
        except ValueError:
            # Skip invalid dates
            continue
    
    holiday_df = pd.DataFrame(holiday_data)
    holiday_df['date'] = pd.to_datetime(holiday_df['date'])
    
    # Add holiday categories
    holiday_df['holiday_category'] = holiday_df['holiday_name'].apply(categorize_holiday)
    
    logger.info(f"Generated {len(holiday_df)} holiday records")
    return holiday_df

def categorize_holiday(holiday_name):
    """
    Categorize holidays for analysis.
    
    Args:
        holiday_name (str): Name of the holiday
        
    Returns:
        str: Holiday category
    """
    major_holidays = ['Christmas Day', 'New Year\'s Day', 'Thanksgiving Day', 'Independence Day']
    minor_holidays = ['Memorial Day', 'Labor Day', 'Veterans Day', 'Presidents\' Day', 'Martin Luther King Jr. Day']
    religious = ['Easter Sunday']
    commercial = ['Black Friday', 'Super Bowl Sunday']
    
    if holiday_name in major_holidays:
        return 'Major Holiday'
    elif holiday_name in minor_holidays:
        return 'Minor Holiday'
    elif holiday_name in religious:
        return 'Religious Holiday'
    elif holiday_name in commercial:
        return 'Commercial Event'
    else:
        return 'Other'

def load_provider_info():
    """Load provider information for ownership type analysis."""
    try:
        # Try to load the most recent provider info file
        provider_files = ['NH_ProviderInfo_Jul2025.csv', 'NH_ProviderInfo_Jun2025.csv']
        provider_data = None
        
        for file in provider_files:
            if os.path.exists(file):
                provider_data = pd.read_csv(file)
                logger.info(f"Loaded provider info from {file}")
                break
        
        if provider_data is not None:
            # Extract relevant columns and clean data
            provider_clean = provider_data[['PROVNUM', 'OWNERSHIP_TYPE']].copy()
            provider_clean['PROVNUM'] = provider_clean['PROVNUM'].astype(str).str.zfill(6).str.upper()
            
            # Map ownership types to simplified categories
            ownership_mapping = {
                'For profit - Corporation': 'For-Profit',
                'For profit - Individual': 'For-Profit',
                'For profit - Partnership': 'For-Profit',
                'For profit - Limited Liability company': 'For-Profit',
                'Non profit - Corporation': 'Non-Profit',
                'Non profit - Church related': 'Non-Profit',
                'Government - City': 'Government',
                'Government - County': 'Government',
                'Government - Federal': 'Government',
                'Government - State': 'Government'
            }
            
            provider_clean['OWNERSHIP_TYPE_CLEAN'] = provider_clean['OWNERSHIP_TYPE'].map(ownership_mapping)
            provider_clean['OWNERSHIP_TYPE_CLEAN'] = provider_clean['OWNERSHIP_TYPE_CLEAN'].fillna('Other')
            
            return provider_clean
        else:
            logger.warning("No provider info files found")
            return None
            
    except Exception as e:
        logger.error(f"Error loading provider info: {str(e)}")
        return None

def load_admin_data_from_files(data_dir="standardized_NonNurse"):
    """
    Load and process admin data from all standardized NonNurse files.
    
    Returns:
        pd.DataFrame: Combined admin metrics dataset
    """
    logger.info("Starting admin data processing...")
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(data_dir, "PBJ_dailynonnursestaffing_*.csv"))
    csv_files.sort()  # Sort to ensure consistent order
    
    logger.info(f"Found {len(csv_files)} files to process")
    
    all_admin_data = []
    
    for file_path in csv_files:
        try:
            # Extract quarter from filename
            filename = os.path.basename(file_path)
            quarter = filename.replace("PBJ_dailynonnursestaffing_", "").replace(".csv", "")
            
            logger.info(f"Processing {filename}...")
            
            # Read only the columns we need - handle missing Hrs_Admin_fn column
            base_columns = [
                'PROVNUM', 'PROVNAME', 'CITY', 'STATE', 'COUNTY_NAME', 
                'CY_Qtr', 'WorkDate', 'MDScensus', 
                'Hrs_Admin', 'Hrs_Admin_ctr'
            ]
            
            # Check if Hrs_Admin_fn column exists in this file
            try:
                # Read just the header to check columns
                header = pd.read_csv(file_path, nrows=0)
                if 'Hrs_Admin_fn' in header.columns:
                    columns_needed = base_columns + ['Hrs_Admin_fn']
                else:
                    columns_needed = base_columns
            except:
                columns_needed = base_columns
            
            # Read the file in chunks to handle large files
            chunk_size = 500000  # Increased chunk size for better performance
            chunk_list = []
            
            for chunk in pd.read_csv(file_path, usecols=columns_needed, chunksize=chunk_size, low_memory=False):
                # Filter for admin data only - more efficient filtering
                # Include records that have any admin hours (regular or contract)
                admin_mask = (chunk['Hrs_Admin'].notna() & (chunk['Hrs_Admin'] > 0)) | \
                           (chunk['Hrs_Admin_ctr'].notna() & (chunk['Hrs_Admin_ctr'] > 0))
                admin_chunk = chunk[admin_mask]
                if not admin_chunk.empty:
                    chunk_list.append(admin_chunk)
            
            if chunk_list:
                file_data = pd.concat(chunk_list, ignore_index=True)
                file_data['Quarter'] = quarter
                all_admin_data.append(file_data)
                
                logger.info(f"Processed {len(file_data)} admin records from {filename}")
            else:
                logger.warning(f"No admin data found in {filename}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not all_admin_data:
        logger.error("No admin data found in any files")
        return None
    
    # Combine all data
    combined_data = pd.concat(all_admin_data, ignore_index=True)
    
    # Clean and standardize the data
    combined_data = clean_admin_data(combined_data)
    
    logger.info(f"Final dataset contains {len(combined_data)} records")
    return combined_data

def clean_admin_data(df):
    """
    Clean and standardize admin data.
    
    Args:
        df (pd.DataFrame): Raw admin data
        
    Returns:
        pd.DataFrame: Cleaned admin data
    """
    logger.info("Cleaning admin data...")
    
    # Convert PROVNUM to string and ensure proper formatting
    df['PROVNUM'] = df['PROVNUM'].astype(str).str.zfill(6).str.upper()
    
    # Convert WorkDate to datetime
    df['WorkDate'] = pd.to_datetime(df['WorkDate'], format='%Y%m%d', errors='coerce')
    
    # Add day-of-week analysis
    df['DayOfWeek'] = df['WorkDate'].dt.day_name()
    df['DayOfWeek_Num'] = df['WorkDate'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['IsWeekend'] = df['DayOfWeek_Num'].isin([5, 6])  # Saturday, Sunday
    
    # Add regional coding
    df['Region'] = df['STATE'].map(REGION_MAPPING)
    df['Region'] = df['Region'].fillna('Other')
    
    # Add holiday information
    holiday_df = get_holiday_info()
    logger.info(f"Holiday columns: {holiday_df.columns.tolist()}")
    logger.info(f"Sample WorkDate: {df['WorkDate'].head()}")
    logger.info(f"Sample holiday date: {holiday_df['date'].head()}")
    
    # Ensure both dates are in the same format
    holiday_df['date'] = pd.to_datetime(holiday_df['date'])
    df['WorkDate'] = pd.to_datetime(df['WorkDate'])
    
    df = df.merge(holiday_df[['date', 'holiday_name', 'is_federal_holiday']], 
                  left_on='WorkDate', right_on='date', how='left')
    
    # Fill missing holiday information
    df['holiday_name'] = df['holiday_name'].fillna('No Holiday')
    df['is_federal_holiday'] = df['is_federal_holiday'].fillna(False)
    df['IsHoliday'] = df['holiday_name'] != 'No Holiday'
    
    # Add holiday category based on holiday name
    df['holiday_category'] = df['holiday_name'].apply(categorize_holiday)
    
    # Add holiday proximity flags (day before and after holidays)
    df['DaysFromHoliday'] = 999  # Initialize with large number
    
    for holiday_date in holiday_df['date'].unique():
        mask = df['WorkDate'] == holiday_date
        df.loc[mask, 'DaysFromHoliday'] = 0
        
        # Day before holiday
        day_before = holiday_date - pd.Timedelta(days=1)
        mask_before = df['WorkDate'] == day_before
        df.loc[mask_before, 'DaysFromHoliday'] = -1
        
        # Day after holiday
        day_after = holiday_date + pd.Timedelta(days=1)
        mask_after = df['WorkDate'] == day_after
        df.loc[mask_after, 'DaysFromHoliday'] = 1
    
    # Create holiday proximity categories
    df['HolidayProximity'] = df['DaysFromHoliday'].apply(categorize_holiday_proximity)
    
    # Drop the temporary date column from merge
    df = df.drop('date', axis=1, errors='ignore')
    
    # Fill missing values
    df['Hrs_Admin'] = df['Hrs_Admin'].fillna(0)
    df['Hrs_Admin_ctr'] = df['Hrs_Admin_ctr'].fillna(0)
    
    # Handle Hrs_Admin_fn column if it exists
    if 'Hrs_Admin_fn' in df.columns:
        df['Hrs_Admin_fn'] = df['Hrs_Admin_fn'].fillna(0)
    else:
        df['Hrs_Admin_fn'] = 0
    
    df['MDScensus'] = df['MDScensus'].fillna(0)
    
    # Calculate total admin hours
    df['Total_Admin_Hours'] = df['Hrs_Admin'] + df['Hrs_Admin_ctr']
    
    # Calculate admin minutes per resident day (MPRD) - more efficient vectorized operations
    df['Admin_MPRD'] = (df['Total_Admin_Hours'] * 60) / df['MDScensus']
    df.loc[df['MDScensus'] <= 0, 'Admin_MPRD'] = 0
    
    # Calculate percent contract admin hours
    df['Pct_Contract_Admin'] = (df['Hrs_Admin_ctr'] / df['Total_Admin_Hours'] * 100)
    df.loc[df['Total_Admin_Hours'] <= 0, 'Pct_Contract_Admin'] = 0
    
    # Flag days with vs without admin
    df['Has_Admin'] = df['Total_Admin_Hours'] > 0
    
    # Add year and quarter columns for easier analysis
    df['Year'] = df['WorkDate'].dt.year
    df['Quarter_Year'] = df['Quarter']
    
    # Sort by date and facility
    df = df.sort_values(['PROVNUM', 'WorkDate'])
    
    logger.info("Admin data cleaning completed")
    return df

def categorize_holiday_proximity(days_from_holiday):
    """
    Categorize days based on proximity to holidays.
    
    Args:
        days_from_holiday (int): Number of days from holiday
        
    Returns:
        str: Proximity category
    """
    if days_from_holiday == 0:
        return 'Holiday'
    elif days_from_holiday == -1:
        return 'Day Before Holiday'
    elif days_from_holiday == 1:
        return 'Day After Holiday'
    elif days_from_holiday in [-2, -3]:
        return 'Week of Holiday'
    elif days_from_holiday in [2, 3]:
        return 'Week After Holiday'
    else:
        return 'Regular Day'

def detect_anomalies(df):
    """
    Detect anomalies in admin staffing data.
    
    Args:
        df (pd.DataFrame): Clean admin data
        
    Returns:
        pd.DataFrame: Data with anomaly flags
    """
    logger.info("Detecting anomalies...")
    
    # Create a copy to avoid modifying original
    df_anomalies = df.copy()
    
    # 1. Sudden drop to zero admin hours for multiple consecutive days
    df_anomalies['Prev_Admin_Hours'] = df_anomalies.groupby('PROVNUM')['Total_Admin_Hours'].shift(1)
    df_anomalies['Next_Admin_Hours'] = df_anomalies.groupby('PROVNUM')['Total_Admin_Hours'].shift(-1)
    
    # Flag consecutive zero days (current and next day both zero)
    df_anomalies['Consecutive_Zero'] = (
        (df_anomalies['Total_Admin_Hours'] == 0) & 
        (df_anomalies['Next_Admin_Hours'] == 0) &
        (df_anomalies['Prev_Admin_Hours'] > 0)  # Had admin hours before
    )
    
    # 2. Unusually high daily hours (possible data entry errors)
    # Calculate facility-specific thresholds (95th percentile + 2 std devs)
    facility_stats = df_anomalies.groupby('PROVNUM')['Total_Admin_Hours'].agg(['mean', 'std', 'quantile']).reset_index()
    facility_stats['quantile'] = facility_stats['quantile'].apply(lambda x: x[0.95] if hasattr(x, '__getitem__') else x)
    facility_stats['high_threshold'] = facility_stats['quantile'] + (2 * facility_stats['std'])
    
    # Merge back to main dataframe
    df_anomalies = df_anomalies.merge(
        facility_stats[['PROVNUM', 'high_threshold']], 
        on='PROVNUM', 
        how='left'
    )
    
    df_anomalies['Unusually_High'] = df_anomalies['Total_Admin_Hours'] > df_anomalies['high_threshold']
    
    # 3. Quarter-to-quarter fluctuations
    # Calculate quarter-over-quarter changes
    df_anomalies['Quarter_Change'] = df_anomalies.groupby('PROVNUM')['Total_Admin_Hours'].pct_change()
    
    # Flag significant drops (>50% decrease)
    df_anomalies['Significant_Drop'] = df_anomalies['Quarter_Change'] < -0.5
    
    # Clean up temporary columns
    df_anomalies = df_anomalies.drop(['Prev_Admin_Hours', 'Next_Admin_Hours', 'high_threshold'], axis=1)
    
    logger.info("Anomaly detection completed")
    return df_anomalies

def calculate_peer_comparisons(df, provider_info=None):
    """
    Calculate peer comparisons for each facility.
    
    Args:
        df (pd.DataFrame): Clean admin data
        provider_info (pd.DataFrame): Provider information with ownership types
        
    Returns:
        pd.DataFrame: Data with peer comparison metrics
    """
    logger.info("Calculating peer comparisons...")
    
    # Add ownership type if available
    if provider_info is not None:
        df = df.merge(provider_info[['PROVNUM', 'OWNERSHIP_TYPE_CLEAN']], on='PROVNUM', how='left')
        df['OWNERSHIP_TYPE_CLEAN'] = df['OWNERSHIP_TYPE_CLEAN'].fillna('Unknown')
    else:
        df['OWNERSHIP_TYPE_CLEAN'] = 'Unknown'
    
    # Calculate averages by different peer groups
    # State averages
    state_avg = df.groupby(['STATE', 'Quarter'])['Total_Admin_Hours'].mean().reset_index()
    state_avg = state_avg.rename(columns={'Total_Admin_Hours': 'State_Avg_Admin_Hours'})
    
    # Regional averages
    region_avg = df.groupby(['Region', 'Quarter'])['Total_Admin_Hours'].mean().reset_index()
    region_avg = region_avg.rename(columns={'Total_Admin_Hours': 'Region_Avg_Admin_Hours'})
    
    # Ownership type averages
    ownership_avg = df.groupby(['OWNERSHIP_TYPE_CLEAN', 'Quarter'])['Total_Admin_Hours'].mean().reset_index()
    ownership_avg = ownership_avg.rename(columns={'Total_Admin_Hours': 'Ownership_Avg_Admin_Hours'})
    
    # Merge peer averages back to main dataframe
    df = df.merge(state_avg, on=['STATE', 'Quarter'], how='left')
    df = df.merge(region_avg, on=['Region', 'Quarter'], how='left')
    df = df.merge(ownership_avg, on=['OWNERSHIP_TYPE_CLEAN', 'Quarter'], how='left')
    
    # Calculate peer comparison ratios
    df['State_Ratio'] = df['Total_Admin_Hours'] / df['State_Avg_Admin_Hours']
    df['Region_Ratio'] = df['Total_Admin_Hours'] / df['Region_Avg_Admin_Hours']
    df['Ownership_Ratio'] = df['Total_Admin_Hours'] / df['Ownership_Avg_Admin_Hours']
    
    # Flag bottom decile facilities
    # Calculate deciles by quarter and state
    df['State_Decile'] = df.groupby(['STATE', 'Quarter'])['Total_Admin_Hours'].rank(pct=True) * 10
    df['Bottom_Decile'] = df['State_Decile'] <= 1
    
    logger.info("Peer comparisons completed")
    return df

def generate_admin_metrics(df):
    """
    Generate aggregated admin metrics at facility, state, and national levels.
    
    Args:
        df (pd.DataFrame): Clean admin data
        
    Returns:
        dict: Dictionary containing different levels of metrics
    """
    logger.info("Generating admin metrics...")
    
    metrics = {}
    
    # Facility-level metrics (daily averages by quarter)
    facility_metrics = df.groupby(['PROVNUM', 'PROVNAME', 'STATE', 'Quarter']).agg({
        'Total_Admin_Hours': ['mean', 'median'],
        'Admin_MPRD': ['mean', 'median'],
        'Pct_Contract_Admin': 'mean',
        'Has_Admin': 'sum',
        'WorkDate': 'count'
    }).reset_index()
    
    # Flatten column names
    facility_metrics.columns = [
        'PROVNUM', 'PROVNAME', 'STATE', 'Quarter',
        'Mean_Admin_Hours', 'Median_Admin_Hours',
        'Mean_Admin_MPRD', 'Median_Admin_MPRD',
        'Pct_Contract_Admin', 'Days_With_Admin', 'Total_Days'
    ]
    
    facility_metrics['Days_Without_Admin'] = facility_metrics['Total_Days'] - facility_metrics['Days_With_Admin']
    facility_metrics['Pct_Days_With_Admin'] = (facility_metrics['Days_With_Admin'] / facility_metrics['Total_Days'] * 100)
    
    # Filter out facilities with >48 hours average per quarter (likely reported incorrectly)
    facility_metrics = facility_metrics[facility_metrics['Mean_Admin_Hours'] <= 48]
    logger.info(f"Filtered out facilities with >48 hours average. Remaining: {len(facility_metrics)} facility-quarters")
    
    metrics['facility'] = facility_metrics
    
    # State-level metrics (recalculate after filtering facilities)
    # Get list of facilities that passed the filter
    valid_facilities = facility_metrics['PROVNUM'].unique()
    filtered_df = df[df['PROVNUM'].isin(valid_facilities)]
    
    state_metrics = filtered_df.groupby(['STATE', 'Quarter']).agg({
        'Total_Admin_Hours': ['mean', 'median'],
        'Admin_MPRD': ['mean', 'median'],
        'Pct_Contract_Admin': 'mean',
        'Has_Admin': 'sum',
        'WorkDate': 'count'
    }).reset_index()
    
    # Flatten column names
    state_metrics.columns = [
        'STATE', 'Quarter',
        'Mean_Admin_Hours', 'Median_Admin_Hours',
        'Mean_Admin_MPRD', 'Median_Admin_MPRD',
        'Pct_Contract_Admin', 'Days_With_Admin', 'Total_Days'
    ]
    
    state_metrics['Days_Without_Admin'] = state_metrics['Total_Days'] - state_metrics['Days_With_Admin']
    state_metrics['Pct_Days_With_Admin'] = (state_metrics['Days_With_Admin'] / state_metrics['Total_Days'] * 100)
    
    metrics['state'] = state_metrics
    
    # National-level metrics (recalculate after filtering facilities)
    national_metrics = filtered_df.groupby('Quarter').agg({
        'Total_Admin_Hours': ['mean', 'median'],
        'Admin_MPRD': ['mean', 'median'],
        'Pct_Contract_Admin': 'mean',
        'Has_Admin': 'sum',
        'WorkDate': 'count'
    }).reset_index()
    
    # Flatten column names
    national_metrics.columns = [
        'Quarter',
        'Mean_Admin_Hours', 'Median_Admin_Hours',
        'Mean_Admin_MPRD', 'Median_Admin_MPRD',
        'Pct_Contract_Admin', 'Days_With_Admin', 'Total_Days'
    ]
    
    national_metrics['Days_Without_Admin'] = national_metrics['Total_Days'] - national_metrics['Days_With_Admin']
    national_metrics['Pct_Days_With_Admin'] = (national_metrics['Days_With_Admin'] / national_metrics['Total_Days'] * 100)
    
    metrics['national'] = national_metrics
    
    # Day-of-week analysis
    day_of_week_metrics = filtered_df.groupby(['DayOfWeek', 'Quarter']).agg({
        'Total_Admin_Hours': ['mean', 'median'],
        'Admin_MPRD': ['mean', 'median'],
        'Has_Admin': 'sum',
        'WorkDate': 'count'
    }).reset_index()
    
    # Flatten column names
    day_of_week_metrics.columns = [
        'DayOfWeek', 'Quarter',
        'Mean_Admin_Hours', 'Median_Admin_Hours',
        'Mean_Admin_MPRD', 'Median_Admin_MPRD',
        'Days_With_Admin', 'Total_Days'
    ]
    
    day_of_week_metrics['Pct_Days_With_Admin'] = (day_of_week_metrics['Days_With_Admin'] / day_of_week_metrics['Total_Days'] * 100)
    
    metrics['day_of_week'] = day_of_week_metrics
    
    # Holiday analysis
    holiday_metrics = filtered_df.groupby(['holiday_category', 'Quarter']).agg({
        'Total_Admin_Hours': ['mean', 'median'],
        'Admin_MPRD': ['mean', 'median'],
        'Has_Admin': 'sum',
        'WorkDate': 'count'
    }).reset_index()
    
    # Flatten column names
    holiday_metrics.columns = [
        'holiday_category', 'Quarter',
        'Mean_Admin_Hours', 'Median_Admin_Hours',
        'Mean_Admin_MPRD', 'Median_Admin_MPRD',
        'Days_With_Admin', 'Total_Days'
    ]
    
    holiday_metrics['Pct_Days_With_Admin'] = (holiday_metrics['Days_With_Admin'] / holiday_metrics['Total_Days'] * 100)
    
    metrics['holiday'] = holiday_metrics
    
    # Holiday proximity analysis
    holiday_proximity_metrics = filtered_df.groupby(['HolidayProximity', 'Quarter']).agg({
        'Total_Admin_Hours': ['mean', 'median'],
        'Admin_MPRD': ['mean', 'median'],
        'Has_Admin': 'sum',
        'WorkDate': 'count'
    }).reset_index()
    
    # Flatten column names
    holiday_proximity_metrics.columns = [
        'HolidayProximity', 'Quarter',
        'Mean_Admin_Hours', 'Median_Admin_Hours',
        'Mean_Admin_MPRD', 'Median_Admin_MPRD',
        'Days_With_Admin', 'Total_Days'
    ]
    
    holiday_proximity_metrics['Pct_Days_With_Admin'] = (holiday_proximity_metrics['Days_With_Admin'] / holiday_proximity_metrics['Total_Days'] * 100)
    
    metrics['holiday_proximity'] = holiday_proximity_metrics
    
    # Regional analysis
    regional_metrics = filtered_df.groupby(['Region', 'Quarter']).agg({
        'Total_Admin_Hours': ['mean', 'median'],
        'Admin_MPRD': ['mean', 'median'],
        'Pct_Contract_Admin': 'mean',
        'Has_Admin': 'sum',
        'WorkDate': 'count'
    }).reset_index()
    
    # Flatten column names
    regional_metrics.columns = [
        'Region', 'Quarter',
        'Mean_Admin_Hours', 'Median_Admin_Hours',
        'Mean_Admin_MPRD', 'Median_Admin_MPRD',
        'Pct_Contract_Admin', 'Days_With_Admin', 'Total_Days'
    ]
    
    regional_metrics['Days_Without_Admin'] = regional_metrics['Total_Days'] - regional_metrics['Days_With_Admin']
    regional_metrics['Pct_Days_With_Admin'] = (regional_metrics['Days_With_Admin'] / regional_metrics['Total_Days'] * 100)
    
    metrics['regional'] = regional_metrics
    
    # Ownership type analysis (if provider info is available)
    if 'OWNERSHIP_TYPE_CLEAN' in filtered_df.columns:
        ownership_metrics = filtered_df.groupby(['OWNERSHIP_TYPE_CLEAN', 'Quarter']).agg({
            'Total_Admin_Hours': ['mean', 'median'],
            'Admin_MPRD': ['mean', 'median'],
            'Pct_Contract_Admin': 'mean',
            'Has_Admin': 'sum',
            'WorkDate': 'count'
        }).reset_index()
        
        # Flatten column names
        ownership_metrics.columns = [
            'OWNERSHIP_TYPE_CLEAN', 'Quarter',
            'Mean_Admin_Hours', 'Median_Admin_Hours',
            'Mean_Admin_MPRD', 'Median_Admin_MPRD',
            'Pct_Contract_Admin', 'Days_With_Admin', 'Total_Days'
        ]
        
        ownership_metrics['Days_Without_Admin'] = ownership_metrics['Total_Days'] - ownership_metrics['Days_With_Admin']
        ownership_metrics['Pct_Days_With_Admin'] = (ownership_metrics['Days_With_Admin'] / ownership_metrics['Total_Days'] * 100)
        
        metrics['ownership'] = ownership_metrics
    
    # Anomaly summary
    anomaly_summary = filtered_df.groupby(['PROVNUM', 'PROVNAME', 'STATE', 'Quarter']).agg({
        'Consecutive_Zero': 'sum',
        'Unusually_High': 'sum',
        'Significant_Drop': 'sum',
        'Bottom_Decile': 'sum',
        'WorkDate': 'count'
    }).reset_index()
    
    anomaly_summary = anomaly_summary.rename(columns={
        'Consecutive_Zero': 'Days_Consecutive_Zero',
        'Unusually_High': 'Days_Unusually_High',
        'Significant_Drop': 'Days_Significant_Drop',
        'Bottom_Decile': 'Days_Bottom_Decile',
        'WorkDate': 'Total_Days'
    })
    
    metrics['anomalies'] = anomaly_summary
    
    logger.info("Admin metrics generation completed")
    return metrics

def save_admin_data(admin_data, metrics, output_dir="admin_data"):
    """
    Save the processed admin data and metrics.
    
    Args:
        admin_data (pd.DataFrame): Clean admin data
        metrics (dict): Generated metrics
        output_dir (str): Output directory
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save the full admin dataset
    admin_data.to_csv(os.path.join(output_dir, "admin_daily_data.csv"), index=False)
    logger.info(f"Saved admin daily data to {output_dir}/admin_daily_data.csv")
    
    # Save metrics
    for level, data in metrics.items():
        filename = f"admin_{level}_metrics.csv"
        data.to_csv(os.path.join(output_dir, filename), index=False)
        logger.info(f"Saved {level} metrics to {output_dir}/{filename}")
    
    # Save enhanced daily data with peer comparisons and anomalies
    enhanced_columns = [
        'PROVNUM', 'PROVNAME', 'CITY', 'STATE', 'Region', 'COUNTY_NAME',
        'CY_Qtr', 'Quarter', 'WorkDate', 'DayOfWeek', 'IsWeekend',
        'holiday_name', 'is_federal_holiday', 'holiday_category', 'IsHoliday',
        'HolidayProximity', 'DaysFromHoliday',
        'MDScensus', 'Hrs_Admin', 'Hrs_Admin_ctr', 'Hrs_Admin_fn',
        'Total_Admin_Hours', 'Admin_MPRD', 'Pct_Contract_Admin', 'Has_Admin',
        'Year', 'Quarter_Year', 'OWNERSHIP_TYPE_CLEAN',
        'State_Avg_Admin_Hours', 'Region_Avg_Admin_Hours', 'Ownership_Avg_Admin_Hours',
        'State_Ratio', 'Region_Ratio', 'Ownership_Ratio',
        'State_Decile', 'Bottom_Decile',
        'Consecutive_Zero', 'Unusually_High', 'Significant_Drop'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in enhanced_columns if col in admin_data.columns]
    enhanced_data = admin_data[available_columns].copy()
    
    enhanced_data.to_csv(os.path.join(output_dir, "admin_enhanced_daily_data.csv"), index=False)
    logger.info(f"Saved enhanced admin daily data to {output_dir}/admin_enhanced_daily_data.csv")

def main():
    """Main function to process admin data."""
    logger.info("Starting admin data processing pipeline...")
    
    # Load provider information for ownership analysis
    provider_info = load_provider_info()
    
    # Load and process admin data
    admin_data = load_admin_data_from_files()
    
    if admin_data is None:
        logger.error("Failed to load admin data")
        return
    
    # Clean and enhance data
    admin_data = clean_admin_data(admin_data)
    
    # Detect anomalies
    admin_data = detect_anomalies(admin_data)
    
    # Calculate peer comparisons
    admin_data = calculate_peer_comparisons(admin_data, provider_info)
    
    # Generate metrics
    metrics = generate_admin_metrics(admin_data)
    
    # Save data
    save_admin_data(admin_data, metrics)
    
    logger.info("Admin data processing pipeline completed successfully!")

if __name__ == "__main__":
    main()
