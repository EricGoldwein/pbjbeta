import pandas as pd
import os
import re
from pathlib import Path
import pytest
import numpy as np

def extract_quarter_from_filename(filename):
    """Extract year and quarter from various filename patterns."""
    filename = filename.lower()
    
    # Pattern 1: PBJ_dailynursestaffing_CY2023Q4.csv
    match1 = re.search(r'cy(\d{4})q(\d)', filename)
    if match1:
        return int(match1.group(1)), int(match1.group(2))
    
    # Pattern 2: pbj_daily_nurse_staffing_cy_2020q4.csv
    match2 = re.search(r'cy_(\d{4})q(\d)', filename)
    if match2:
        return int(match2.group(1)), int(match2.group(2))
    
    # Pattern 3: PBJ_Nurse_2019_Q3_rqet-pmzi.csv
    match3 = re.search(r'(\d{4})_q(\d)', filename)
    if match3:
        return int(match3.group(1)), int(match3.group(2))
    
    return None, None

def standardize_column_names(df):
    """Standardize column names to handle variations in capitalization and spacing."""
    column_mapping = {
        'cy_qtr': ['cy_qtr', 'CY_Qtr', 'CY_QTR'],
        'provnum': ['PROVNUM', 'provnum'],
        'provname': ['PROVNAME', 'provname'],
        'city': ['CITY', 'city'],
        'state': ['STATE', 'state'],
        'county_name': ['COUNTY_NAME', 'county_name'],
        'county_fips': ['COUNTY_FIPS', 'county_fips'],
        'workdate': ['WorkDate', 'workdate'],
        'mdscensus': ['MDScensus', 'mdscensus']
    }
    
    for standard_name, variations in column_mapping.items():
        for variation in variations:
            if variation in df.columns:
                if variation != standard_name:
                    df = df.rename(columns={variation: standard_name})
    
    return df

@pytest.mark.parametrize("file_path", [str(p) for p in Path("PBJ_Nurse").glob("*.csv")])
def test_pbj_file_headers(file_path):
    """Test PBJ nurse staffing file headers and quarter information."""
    print(f"\nTesting file: {file_path}")
    
    # Extract quarter info from filename
    year, quarter = extract_quarter_from_filename(Path(file_path).name)
    assert year is not None, f"Could not extract year from filename: {file_path}"
    assert quarter in [1, 2, 3, 4], f"Invalid quarter in filename: {file_path}"
    
    # Read just the headers
    df = pd.read_csv(file_path, nrows=0)
    
    # Standardize column names
    df = standardize_column_names(df)
    
    # Required basic columns
    required_columns = [
        'provnum', 'provname', 'city', 'state', 'county_name', 'county_fips',
        'cy_qtr', 'workdate', 'mdscensus'
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col} in {file_path}"
    
    # At least one hours column should exist
    hours_prefixes = ['hrs_rn', 'hrs_lpn', 'hrs_cna']
    has_hours = any(any(col.lower().startswith(prefix) for col in df.columns) for prefix in hours_prefixes)
    assert has_hours, f"No hours columns found in {file_path}"
    
    # Read first row to check cy_qtr
    df = pd.read_csv(file_path, nrows=1)
    df = standardize_column_names(df)
    
    # Test cy_qtr column matches filename
    qtr_str = str(df['cy_qtr'].iloc[0])
    qtr_year = int(qtr_str[:4])
    qtr_num = int(qtr_str[-1])
    
    assert qtr_year == year, f"Year mismatch in {file_path}: filename={year}, data={qtr_year}"
    assert qtr_num == quarter, f"Quarter mismatch in {file_path}: filename={quarter}, data={qtr_num}"
    
    print(f"✓ All header tests passed for {file_path}")

def test_required_columns():
    """Test that all files have the required columns."""
    required_columns = {
        'provnum', 'provname', 'city', 'state', 'county', 'fips', 'quarter', 'workdate',
        'mdscensus', 'rntotal', 'rnemployee', 'rncontractor', 'lpntotal', 'lpnemployee',
        'lpncontractor', 'aidetotal', 'aideemployee', 'aidecontractor'
    }

    for file_path in Path("PBJ_Nurse").glob("*.csv"):
        print(f"\nChecking required columns in: {file_path}")
        try:
            # Try different encodings
            encodings = ['utf-8', 'cp1252', 'latin1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not read file with any of the attempted encodings: {encodings}")

            # Convert all column names to lowercase for case-insensitive comparison
            df.columns = df.columns.str.lower()
            
            # Check that all required columns are present
            missing_cols = required_columns - set(df.columns)
            assert not missing_cols, f"Missing required columns in {file_path}: {missing_cols}"
            
            # Check that at least one hours column exists
            hours_cols = [col for col in df.columns if any(x in col.lower() for x in ['total', 'employee', 'contractor'])]
            assert hours_cols, f"No hours columns found in {file_path}"

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            raise

@pytest.mark.parametrize("file_path", [str(p) for p in Path("PBJ_Nurse").glob("*.csv")])
def test_pbj_data_format(file_path):
    """Test data format and value ranges."""
    print(f"\nTesting data format in: {file_path}")

    # Try different encodings
    encodings = ['utf-8', 'cp1252', 'latin1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not read file with any of the attempted encodings: {encodings}")

    # Convert all column names to lowercase for case-insensitive comparison
    df.columns = df.columns.str.lower()

    # Test provider number format (5 or 6 digits)
    # First ensure it's a string and remove any quotes
    df['provnum'] = df['provnum'].astype(str).str.strip('"')
    assert df['provnum'].str.match(r'^\d{5,6}$').all(), f"Invalid provider number format in {file_path}"

    # Test state format (2 uppercase letters)
    df['state'] = df['state'].str.upper()
    assert df['state'].str.match(r'^[A-Z]{2}$').all(), f"Invalid state format in {file_path}"

    # Test FIPS code format (3 digits, zero-padded)
    df['fips'] = df['fips'].astype(str).str.zfill(3)
    assert df['fips'].str.match(r'^\d{3}$').all(), f"Invalid FIPS code format in {file_path}"

    # Test workdate format (8 digits: YYYYMMDD)
    df['workdate'] = df['workdate'].astype(str)
    assert df['workdate'].str.match(r'^\d{8}$').all(), f"Invalid workdate format in {file_path}"

    # Test MDScensus values (numeric and non-negative)
    df['mdscensus'] = pd.to_numeric(df['mdscensus'], errors='coerce')
    assert df['mdscensus'].notna().all(), f"Invalid MDScensus values in {file_path}"
    assert (df['mdscensus'] >= 0).all(), f"Negative MDScensus values in {file_path}"

    # Test hours columns (numeric and non-negative)
    hours_cols = [col for col in df.columns if any(x in col.lower() for x in ['total', 'employee', 'contractor'])]
    for col in hours_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        assert df[col].notna().all(), f"Invalid hours values in column {col} in {file_path}"
        assert (df[col] >= 0).all(), f"Negative hours values in column {col} in {file_path}"

    # Test that total hours match sum of employee and contractor hours
    for role in ['rn', 'lpn', 'aide']:
        total_col = f"{role}total"
        emp_col = f"{role}employee"
        cont_col = f"{role}contractor"
        
        if all(col in df.columns for col in [total_col, emp_col, cont_col]):
            # Use pd.isclose for floating point comparison
            assert pd.isclose(
                df[total_col],
                df[emp_col] + df[cont_col],
                rtol=1e-05,  # relative tolerance
                atol=1e-08,  # absolute tolerance
                equal_nan=True  # treat NaN as equal
            ).all(), f"Total hours don't match sum of employee and contractor hours for {role} in {file_path}"

if __name__ == "__main__":
    pytest.main([__file__]) 
import os
import re
from pathlib import Path
import pytest
import numpy as np

def extract_quarter_from_filename(filename):
    """Extract year and quarter from various filename patterns."""
    filename = filename.lower()
    
    # Pattern 1: PBJ_dailynursestaffing_CY2023Q4.csv
    match1 = re.search(r'cy(\d{4})q(\d)', filename)
    if match1:
        return int(match1.group(1)), int(match1.group(2))
    
    # Pattern 2: pbj_daily_nurse_staffing_cy_2020q4.csv
    match2 = re.search(r'cy_(\d{4})q(\d)', filename)
    if match2:
        return int(match2.group(1)), int(match2.group(2))
    
    # Pattern 3: PBJ_Nurse_2019_Q3_rqet-pmzi.csv
    match3 = re.search(r'(\d{4})_q(\d)', filename)
    if match3:
        return int(match3.group(1)), int(match3.group(2))
    
    return None, None

def standardize_column_names(df):
    """Standardize column names to handle variations in capitalization and spacing."""
    column_mapping = {
        'cy_qtr': ['cy_qtr', 'CY_Qtr', 'CY_QTR'],
        'provnum': ['PROVNUM', 'provnum'],
        'provname': ['PROVNAME', 'provname'],
        'city': ['CITY', 'city'],
        'state': ['STATE', 'state'],
        'county_name': ['COUNTY_NAME', 'county_name'],
        'county_fips': ['COUNTY_FIPS', 'county_fips'],
        'workdate': ['WorkDate', 'workdate'],
        'mdscensus': ['MDScensus', 'mdscensus']
    }
    
    for standard_name, variations in column_mapping.items():
        for variation in variations:
            if variation in df.columns:
                if variation != standard_name:
                    df = df.rename(columns={variation: standard_name})
    
    return df

@pytest.mark.parametrize("file_path", [str(p) for p in Path("PBJ_Nurse").glob("*.csv")])
def test_pbj_file_headers(file_path):
    """Test PBJ nurse staffing file headers and quarter information."""
    print(f"\nTesting file: {file_path}")
    
    # Extract quarter info from filename
    year, quarter = extract_quarter_from_filename(Path(file_path).name)
    assert year is not None, f"Could not extract year from filename: {file_path}"
    assert quarter in [1, 2, 3, 4], f"Invalid quarter in filename: {file_path}"
    
    # Read just the headers
    df = pd.read_csv(file_path, nrows=0)
    
    # Standardize column names
    df = standardize_column_names(df)
    
    # Required basic columns
    required_columns = [
        'provnum', 'provname', 'city', 'state', 'county_name', 'county_fips',
        'cy_qtr', 'workdate', 'mdscensus'
    ]
    for col in required_columns:
        assert col in df.columns, f"Missing required column: {col} in {file_path}"
    
    # At least one hours column should exist
    hours_prefixes = ['hrs_rn', 'hrs_lpn', 'hrs_cna']
    has_hours = any(any(col.lower().startswith(prefix) for col in df.columns) for prefix in hours_prefixes)
    assert has_hours, f"No hours columns found in {file_path}"
    
    # Read first row to check cy_qtr
    df = pd.read_csv(file_path, nrows=1)
    df = standardize_column_names(df)
    
    # Test cy_qtr column matches filename
    qtr_str = str(df['cy_qtr'].iloc[0])
    qtr_year = int(qtr_str[:4])
    qtr_num = int(qtr_str[-1])
    
    assert qtr_year == year, f"Year mismatch in {file_path}: filename={year}, data={qtr_year}"
    assert qtr_num == quarter, f"Quarter mismatch in {file_path}: filename={quarter}, data={qtr_num}"
    
    print(f"✓ All header tests passed for {file_path}")

def test_required_columns():
    """Test that all files have the required columns."""
    required_columns = {
        'provnum', 'provname', 'city', 'state', 'county', 'fips', 'quarter', 'workdate',
        'mdscensus', 'rntotal', 'rnemployee', 'rncontractor', 'lpntotal', 'lpnemployee',
        'lpncontractor', 'aidetotal', 'aideemployee', 'aidecontractor'
    }

    for file_path in Path("PBJ_Nurse").glob("*.csv"):
        print(f"\nChecking required columns in: {file_path}")
        try:
            # Try different encodings
            encodings = ['utf-8', 'cp1252', 'latin1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not read file with any of the attempted encodings: {encodings}")

            # Convert all column names to lowercase for case-insensitive comparison
            df.columns = df.columns.str.lower()
            
            # Check that all required columns are present
            missing_cols = required_columns - set(df.columns)
            assert not missing_cols, f"Missing required columns in {file_path}: {missing_cols}"
            
            # Check that at least one hours column exists
            hours_cols = [col for col in df.columns if any(x in col.lower() for x in ['total', 'employee', 'contractor'])]
            assert hours_cols, f"No hours columns found in {file_path}"

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            raise

@pytest.mark.parametrize("file_path", [str(p) for p in Path("PBJ_Nurse").glob("*.csv")])
def test_pbj_data_format(file_path):
    """Test data format and value ranges."""
    print(f"\nTesting data format in: {file_path}")

    # Try different encodings
    encodings = ['utf-8', 'cp1252', 'latin1']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    
    if df is None:
        raise ValueError(f"Could not read file with any of the attempted encodings: {encodings}")

    # Convert all column names to lowercase for case-insensitive comparison
    df.columns = df.columns.str.lower()

    # Test provider number format (5 or 6 digits)
    # First ensure it's a string and remove any quotes
    df['provnum'] = df['provnum'].astype(str).str.strip('"')
    assert df['provnum'].str.match(r'^\d{5,6}$').all(), f"Invalid provider number format in {file_path}"

    # Test state format (2 uppercase letters)
    df['state'] = df['state'].str.upper()
    assert df['state'].str.match(r'^[A-Z]{2}$').all(), f"Invalid state format in {file_path}"

    # Test FIPS code format (3 digits, zero-padded)
    df['fips'] = df['fips'].astype(str).str.zfill(3)
    assert df['fips'].str.match(r'^\d{3}$').all(), f"Invalid FIPS code format in {file_path}"

    # Test workdate format (8 digits: YYYYMMDD)
    df['workdate'] = df['workdate'].astype(str)
    assert df['workdate'].str.match(r'^\d{8}$').all(), f"Invalid workdate format in {file_path}"

    # Test MDScensus values (numeric and non-negative)
    df['mdscensus'] = pd.to_numeric(df['mdscensus'], errors='coerce')
    assert df['mdscensus'].notna().all(), f"Invalid MDScensus values in {file_path}"
    assert (df['mdscensus'] >= 0).all(), f"Negative MDScensus values in {file_path}"

    # Test hours columns (numeric and non-negative)
    hours_cols = [col for col in df.columns if any(x in col.lower() for x in ['total', 'employee', 'contractor'])]
    for col in hours_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        assert df[col].notna().all(), f"Invalid hours values in column {col} in {file_path}"
        assert (df[col] >= 0).all(), f"Negative hours values in column {col} in {file_path}"

    # Test that total hours match sum of employee and contractor hours
    for role in ['rn', 'lpn', 'aide']:
        total_col = f"{role}total"
        emp_col = f"{role}employee"
        cont_col = f"{role}contractor"
        
        if all(col in df.columns for col in [total_col, emp_col, cont_col]):
            # Use pd.isclose for floating point comparison
            assert pd.isclose(
                df[total_col],
                df[emp_col] + df[cont_col],
                rtol=1e-05,  # relative tolerance
                atol=1e-08,  # absolute tolerance
                equal_nan=True  # treat NaN as equal
            ).all(), f"Total hours don't match sum of employee and contractor hours for {role} in {file_path}"

if __name__ == "__main__":
    pytest.main([__file__]) 