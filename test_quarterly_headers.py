import pandas as pd
from pathlib import Path
import pytest
import re
import csv
from collections import defaultdict

def extract_quarter_from_filename(filename):
    """Extract year and quarter from filename."""
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

def create_column_mapping():
    """Create a comprehensive mapping of column name variations to standard names."""
    return {
        # Basic Information
        'PROVNUM': ['PROVNUM', 'provnum'],
        'PROVNAME': ['PROVNAME', 'provname'],
        'CITY': ['CITY', 'city'],
        'STATE': ['STATE', 'state'],
        'COUNTY_NAME': ['COUNTY_NAME', 'county_name'],
        'COUNTY_FIPS': ['COUNTY_FIPS', 'county_fips'],
        'CY_QTR': ['CY_Qtr', 'CY_QTR', 'cy_qtr'],
        'WORKDATE': ['WorkDate', 'workdate'],
        'MDSCENSUS': ['MDScensus', 'mdscensus'],
        
        # RN Hours
        'HRS_RN': ['Hrs_RN', 'hrs_rn'],
        'HRS_RN_EMP': ['Hrs_RN_emp', 'hrs_rn_emp'],
        'HRS_RN_CTR': ['Hrs_RN_ctr', 'hrs_rn_ctr'],
        'HRS_RNADMIN': ['Hrs_RNadmin', 'hrs_rnadmin'],
        'HRS_RNADMIN_EMP': ['Hrs_RNadmin_emp', 'hrs_rnadmin_emp'],
        'HRS_RNADMIN_CTR': ['Hrs_RNadmin_ctr', 'hrs_rnadmin_ctr'],
        'HRS_RNDON': ['Hrs_RNDON', 'hrs_rndon', 'hrs_rn_donadmin'],
        'HRS_RNDON_EMP': ['Hrs_RNDON_emp', 'hrs_rndon_emp'],
        'HRS_RNDON_CTR': ['Hrs_RNDON_ctr', 'hrs_rndon_ctr'],
        
        # LPN Hours
        'HRS_LPN': ['Hrs_LPN', 'hrs_lpn'],
        'HRS_LPN_EMP': ['Hrs_LPN_emp', 'hrs_lpn_emp'],
        'HRS_LPN_CTR': ['Hrs_LPN_ctr', 'hrs_lpn_ctr'],
        'HRS_LPNADMIN': ['Hrs_LPNadmin', 'hrs_lpnadmin', 'hrs_lpn_admin'],
        'HRS_LPNADMIN_EMP': ['Hrs_LPNadmin_emp', 'hrs_lpnadmin_emp'],
        'HRS_LPNADMIN_CTR': ['Hrs_LPNadmin_ctr', 'hrs_lpnadmin_ctr'],
        
        # CNA/NA Hours
        'HRS_CNA': ['Hrs_CNA', 'hrs_cna'],
        'HRS_CNA_EMP': ['Hrs_CNA_emp', 'hrs_cna_emp'],
        'HRS_CNA_CTR': ['Hrs_CNA_ctr', 'hrs_cna_ctr'],
        'HRS_NATRN': ['Hrs_NAtrn', 'hrs_natrn', 'hrs_na_trn'],
        'HRS_NATRN_EMP': ['Hrs_NAtrn_emp', 'hrs_natrn_emp'],
        'HRS_NATRN_CTR': ['Hrs_NAtrn_ctr', 'hrs_natrn_ctr'],
        
        # Med Aide Hours
        'HRS_MEDAIDE': ['Hrs_MedAide', 'hrs_medaide'],
        'HRS_MEDAIDE_EMP': ['Hrs_MedAide_emp', 'hrs_medaide_emp'],
        'HRS_MEDAIDE_CTR': ['Hrs_MedAide_ctr', 'hrs_medaide_ctr'],
        
        # Other
        'INCOMPLETE': ['incomplete']
    }

def standardize_column_names(df, column_mapping):
    """Standardize column names using the provided mapping."""
    # Create reverse mapping for easier lookup
    reverse_mapping = {}
    for standard_name, variations in column_mapping.items():
        for variation in variations:
            reverse_mapping[variation] = standard_name
    
    # Rename columns
    new_columns = {}
    for col in df.columns:
        if col in reverse_mapping:
            new_columns[col] = reverse_mapping[col]
        else:
            # If column not in mapping, convert to uppercase and replace spaces with underscores
            new_columns[col] = col.upper().replace(' ', '_')
    
    return df.rename(columns=new_columns)

def analyze_column_patterns(headers_dict):
    """Analyze patterns in column names across quarters."""
    patterns = {
        'case_patterns': defaultdict(set),  # UPPER, lower, Mixed
        'separator_patterns': defaultdict(set),  # underscore, none, other
        'prefix_patterns': defaultdict(set),  # hrs_, Hrs_, etc.
        'suffix_patterns': defaultdict(set),  # _emp, _ctr, etc.
    }
    
    for quarter, headers in headers_dict.items():
        for header in headers:
            # Case pattern
            if header.isupper():
                patterns['case_patterns']['UPPER'].add(quarter)
            elif header.islower():
                patterns['case_patterns']['lower'].add(quarter)
            else:
                patterns['case_patterns']['Mixed'].add(quarter)
            
            # Separator pattern
            if '_' in header:
                patterns['separator_patterns']['underscore'].add(quarter)
            elif ' ' in header:
                patterns['separator_patterns']['space'].add(quarter)
            else:
                patterns['separator_patterns']['none'].add(quarter)
            
            # Prefix pattern
            if header.lower().startswith('hrs'):
                patterns['prefix_patterns']['hrs'].add(quarter)
            elif header.startswith('Hrs'):
                patterns['prefix_patterns']['Hrs'].add(quarter)
            
            # Suffix pattern
            if header.lower().endswith('_emp'):
                patterns['suffix_patterns']['_emp'].add(quarter)
            elif header.lower().endswith('_ctr'):
                patterns['suffix_patterns']['_ctr'].add(quarter)
    
    return patterns

def create_header_comparison_csv():
    """Create a CSV file showing header comparisons across all quarters."""
    # Get all CSV files in the PBJ_Nurse directory
    files = list(Path("PBJ_Nurse").glob("*.csv"))
    assert files, "No CSV files found in PBJ_Nurse directory"
    
    # Dictionary to store headers for each quarter
    headers_dict = {}
    
    # Read headers from each file
    for file_path in files:
        try:
            # Try different encodings
            encodings = ['utf-8', 'cp1252', 'latin1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, nrows=0, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not read file with any of the attempted encodings")
            
            # Extract quarter info from filename
            year, quarter = extract_quarter_from_filename(file_path.name)
            if year is None or quarter not in [1, 2, 3, 4]:
                continue
            
            # Store original headers
            headers_dict[f"{year}Q{quarter}"] = set(df.columns)
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
            continue
    
    # Get all unique headers across all files
    all_headers = set()
    for headers in headers_dict.values():
        all_headers.update(headers)
    
    # Sort headers and quarters for consistent output
    sorted_headers = sorted(all_headers)
    sorted_quarters = sorted(headers_dict.keys())
    
    # Create output CSV
    output_file = "header_comparison.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        writer.writerow(['Header'] + sorted_quarters)
        
        # Write data rows
        for header in sorted_headers:
            row = [header]
            for quarter in sorted_quarters:
                row.append('X' if header in headers_dict[quarter] else '')
            writer.writerow(row)
    
    # Create detailed analysis file
    analysis_file = "header_analysis.txt"
    with open(analysis_file, 'w') as f:
        f.write("DETAILED HEADER ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        
        # Write column mapping
        f.write("Column Name Standardization Mapping:\n")
        f.write("-" * 50 + "\n")
        column_mapping = create_column_mapping()
        for standard_name, variations in column_mapping.items():
            f.write(f"\n{standard_name}:\n")
            for var in variations:
                f.write(f"  - {var}\n")
        
        # Analyze patterns
        patterns = analyze_column_patterns(headers_dict)
        
        f.write("\n\nPattern Analysis:\n")
        f.write("-" * 50 + "\n")
        
        # Case patterns
        f.write("\nCase Patterns:\n")
        for pattern, quarters in patterns['case_patterns'].items():
            f.write(f"{pattern}: {', '.join(sorted(quarters))}\n")
        
        # Separator patterns
        f.write("\nSeparator Patterns:\n")
        for pattern, quarters in patterns['separator_patterns'].items():
            f.write(f"{pattern}: {', '.join(sorted(quarters))}\n")
        
        # Prefix patterns
        f.write("\nPrefix Patterns:\n")
        for pattern, quarters in patterns['prefix_patterns'].items():
            f.write(f"{pattern}: {', '.join(sorted(quarters))}\n")
        
        # Suffix patterns
        f.write("\nSuffix Patterns:\n")
        for pattern, quarters in patterns['suffix_patterns'].items():
            f.write(f"{pattern}: {', '.join(sorted(quarters))}\n")
        
        # Write standardization recommendations
        f.write("\n\nStandardization Recommendations:\n")
        f.write("-" * 50 + "\n")
        f.write("1. Use lowercase for all column names\n")
        f.write("2. Use underscores as separators\n")
        f.write("3. Use consistent prefixes (e.g., 'hrs_' for all hours columns)\n")
        f.write("4. Use consistent suffixes (e.g., '_emp' for employee hours, '_ctr' for contractor hours)\n")
        f.write("5. Standardize abbreviations (e.g., 'rn' for Registered Nurse, 'lpn' for Licensed Practical Nurse)\n")
    
    print(f"Header comparison CSV created: {output_file}")
    print(f"Detailed header analysis created: {analysis_file}")

def test_quarterly_headers():
    """Run the header comparison and create output files."""
    create_header_comparison_csv()

def test_quarterly_headers_consistency():
    """Test that all quarterly files have consistent headers."""
    # Get all CSV files in the PBJ_Nurse directory
    files = list(Path("PBJ_Nurse").glob("*.csv"))
    assert files, "No CSV files found in PBJ_Nurse directory"
    
    # Dictionary to store headers for each quarter
    headers_dict = {}
    
    # Get column mapping
    column_mapping = create_column_mapping()
    
    # Read headers from each file
    for file_path in files:
        try:
            # Try different encodings
            encodings = ['utf-8', 'cp1252', 'latin1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, nrows=0, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not read file with any of the attempted encodings")
            
            # Extract quarter info from filename
            year, quarter = extract_quarter_from_filename(file_path.name)
            assert year is not None, f"Could not extract year from filename: {file_path.name}"
            assert quarter in [1, 2, 3, 4], f"Invalid quarter in filename: {file_path.name}"
            
            # Standardize column names
            df = standardize_column_names(df, column_mapping)
            
            # Store headers in lowercase for case-insensitive comparison
            headers_dict[f"{year}Q{quarter}"] = set(col.lower() for col in df.columns)
            
        except Exception as e:
            pytest.fail(f"Error processing {file_path.name}: {str(e)}")
    
    # Get all unique headers across all files
    all_headers = set()
    for headers in headers_dict.values():
        all_headers.update(headers)
    
    # Sort headers and quarters for consistent output
    sorted_headers = sorted(all_headers)
    sorted_quarters = sorted(headers_dict.keys())
    
    # Required headers that should be present in all files
    required_headers = {
        'provnum', 'provname', 'city', 'state', 'county_name', 'county_fips', 'cy_qtr', 'workdate',
        'mdscensus'
    }
    
    # Check that all required headers are present in all files
    for quarter, headers in headers_dict.items():
        missing_headers = required_headers - headers
        assert not missing_headers, f"Missing required headers in {quarter}: {missing_headers}"
    
    # Compare headers across quarters
    quarters = sorted(headers_dict.keys())
    first_quarter = quarters[0]
    first_headers = headers_dict[first_quarter]
    
    for quarter in quarters[1:]:
        current_headers = headers_dict[quarter]
        missing_headers = first_headers - current_headers
        extra_headers = current_headers - first_headers
        
        # Allow for some variation in optional columns but ensure required ones are consistent
        assert not (missing_headers & required_headers), f"Missing required headers in {quarter}: {missing_headers & required_headers}"
        assert not (extra_headers & required_headers), f"Extra required headers in {quarter}: {extra_headers & required_headers}"

def test_quarterly_data_types():
    """Test that data types are consistent across quarters."""
    files = list(Path("PBJ_Nurse").glob("*.csv"))
    assert files, "No CSV files found in PBJ_Nurse directory"
    
    # Dictionary to store data types for each quarter
    dtypes_dict = {}
    
    # Get column mapping
    column_mapping = create_column_mapping()
    
    # Read first row from each file to check data types
    for file_path in files:
        try:
            # Try different encodings
            encodings = ['utf-8', 'cp1252', 'latin1']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, nrows=1, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError(f"Could not read file with any of the attempted encodings")
            
            # Extract quarter info from filename
            year, quarter = extract_quarter_from_filename(file_path.name)
            assert year is not None, f"Could not extract year from filename: {file_path.name}"
            assert quarter in [1, 2, 3, 4], f"Invalid quarter in filename: {file_path.name}"
            
            # Standardize column names
            df = standardize_column_names(df, column_mapping)
            
            # Store data types
            dtypes_dict[f"{year}Q{quarter}"] = df.dtypes.to_dict()
            
        except Exception as e:
            pytest.fail(f"Error processing {file_path.name}: {str(e)}")
    
    # Compare data types across quarters
    quarters = sorted(dtypes_dict.keys())
    first_quarter = quarters[0]
    first_dtypes = dtypes_dict[first_quarter]
    
    for quarter in quarters[1:]:
        current_dtypes = dtypes_dict[quarter]
        
        # Check that numeric columns remain numeric
        numeric_cols = ['mdscensus']
        
        for col in numeric_cols:
            if col in first_dtypes and col in current_dtypes:
                assert pd.api.types.is_numeric_dtype(first_dtypes[col]) == pd.api.types.is_numeric_dtype(current_dtypes[col]), \
                    f"Data type mismatch for {col} between {first_quarter} and {quarter}" 
 