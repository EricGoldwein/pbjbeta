"""
Dynamic file finder utility for PBJ Dashboard.
Automatically discovers and loads the most recent data files.
"""
import os
import glob
import re
from datetime import datetime
from typing import Optional, List, Tuple


# Month name to number mapping
MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}


def parse_date_from_filename(filename: str) -> Optional[datetime]:
    """
    Parse date from CMS filename formats.
    
    Handles formats like:
    - NH_ProviderInfo_Sep2025.csv
    - NH_ProviderInfo_Jun2025.csv
    - Nursing_Home_Affiliated_Entity_Performance_Measures_Mar_2025.csv
    
    Args:
        filename: The filename to parse
        
    Returns:
        datetime object if date found, None otherwise
    """
    # Pattern 1: Month name followed by year (e.g., Sep2025, Mar_2025)
    pattern = r'([A-Za-z]+)_?(\d{4})'
    match = re.search(pattern, filename)
    
    if match:
        month_str = match.group(1)
        year_str = match.group(2)
        
        # Find matching month
        for month_name, month_num in MONTH_MAP.items():
            if month_str.lower() == month_name.lower():
                try:
                    return datetime(int(year_str), month_num, 1)
                except ValueError:
                    continue
    
    return None


def find_files_with_pattern(base_dir: str, file_pattern: str, search_paths: Optional[List[str]] = None) -> List[Tuple[str, datetime]]:
    """
    Find all files matching a pattern in specified directories.
    
    Args:
        base_dir: Base directory to search (e.g., 'provider_info')
        file_pattern: Glob pattern for files (e.g., 'NH_ProviderInfo_*.csv')
        search_paths: Optional list of additional paths to search
        
    Returns:
        List of tuples (full_path, parsed_date) sorted by date (newest first)
    """
    files_with_dates = []
    
    # Build search paths
    if search_paths is None:
        search_paths = [
            os.path.join(os.getcwd(), base_dir),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', base_dir),
            base_dir,
            # Common deployment paths
            os.path.join('/app', base_dir),
            os.path.join('/workspace', base_dir)
        ]
    
    # Search in all paths
    for search_path in search_paths:
        pattern_path = os.path.join(search_path, file_pattern)
        matching_files = glob.glob(pattern_path)
        
        for file_path in matching_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                parsed_date = parse_date_from_filename(filename)
                
                if parsed_date:
                    files_with_dates.append((file_path, parsed_date))
                else:
                    # If can't parse date, use file modification time
                    mod_time = os.path.getmtime(file_path)
                    files_with_dates.append((file_path, datetime.fromtimestamp(mod_time)))
    
    # Sort by date (newest first)
    files_with_dates.sort(key=lambda x: x[1], reverse=True)
    
    return files_with_dates


def find_latest_file(base_dir: str, file_pattern: str, search_paths: Optional[List[str]] = None) -> Optional[str]:
    """
    Find the most recent file matching a pattern.
    
    Args:
        base_dir: Base directory to search (e.g., 'provider_info')
        file_pattern: Glob pattern for files (e.g., 'NH_ProviderInfo_*.csv')
        search_paths: Optional list of additional paths to search
        
    Returns:
        Full path to the most recent file, or None if no files found
        
    Example:
        >>> find_latest_file('provider_info', 'NH_ProviderInfo_*.csv')
        '/path/to/provider_info/NH_ProviderInfo_Oct2025.csv'
    """
    files = find_files_with_pattern(base_dir, file_pattern, search_paths)
    
    if files:
        latest_file, latest_date = files[0]
        print(f"Using latest file: {os.path.basename(latest_file)} (dated {latest_date.strftime('%B %Y')})")
        return latest_file
    
    return None


def find_nth_latest_file(base_dir: str, file_pattern: str, n: int = 2, search_paths: Optional[List[str]] = None) -> Optional[str]:
    """
    Find the Nth most recent file matching a pattern.
    
    Args:
        base_dir: Base directory to search
        file_pattern: Glob pattern for files
        n: Which file to return (1 = newest, 2 = second newest, etc.)
        search_paths: Optional list of additional paths to search
        
    Returns:
        Full path to the Nth most recent file, or None if not enough files found
        
    Example:
        >>> find_nth_latest_file('provider_info', 'NH_ProviderInfo_*.csv', n=2)
        '/path/to/provider_info/NH_ProviderInfo_Sep2025.csv'
    """
    files = find_files_with_pattern(base_dir, file_pattern, search_paths)
    
    if len(files) >= n:
        nth_file, nth_date = files[n - 1]
        print(f"Using {n}th latest file: {os.path.basename(nth_file)} (dated {nth_date.strftime('%B %Y')})")
        return nth_file
    
    return None


def find_all_matching_files(base_dir: str, file_pattern: str, search_paths: Optional[List[str]] = None) -> List[str]:
    """
    Find all files matching a pattern, sorted by date (newest first).
    
    Args:
        base_dir: Base directory to search
        file_pattern: Glob pattern for files
        search_paths: Optional list of additional paths to search
        
    Returns:
        List of full paths to matching files, sorted by date (newest first)
    """
    files = find_files_with_pattern(base_dir, file_pattern, search_paths)
    return [file_path for file_path, _ in files]


# Convenience functions for common file types
def find_latest_provider_info() -> Optional[str]:
    """Find the latest Provider Info file."""
    return find_latest_file('provider_info', 'NH_ProviderInfo_*.csv')


def find_previous_provider_info() -> Optional[str]:
    """Find the most recent Provider Info file from a different quarter (for comparisons)."""
    # Get all files and find the most recent file from a different quarter
    files = find_files_with_pattern('provider_info', 'NH_ProviderInfo_*.csv')
    
    if len(files) < 2:
        return None
    
    # Remove duplicates by filename
    unique_files = {}
    for file_path, date in files:
        filename = os.path.basename(file_path)
        if filename not in unique_files:
            unique_files[filename] = (file_path, date)
    
    # Sort by date (newest first)
    unique_file_list = list(unique_files.values())
    unique_file_list.sort(key=lambda x: x[1], reverse=True)
    
    if len(unique_file_list) < 2:
        return None
    
    # Get the latest file and its quarter
    latest_file, latest_date = unique_file_list[0]
    latest_quarter = (latest_date.year, (latest_date.month - 1) // 3 + 1)
    
    # Find the most recent file from a different quarter
    for file_path, file_date in unique_file_list[1:]:
        file_quarter = (file_date.year, (file_date.month - 1) // 3 + 1)
        if file_quarter != latest_quarter:
            print(f"Using previous quarter file: {os.path.basename(file_path)} (dated {file_date.strftime('%B %Y')}, Q{file_quarter[1]} {file_quarter[0]})")
            return file_path
    
    # If no different quarter found, fall back to second most recent
    second_latest_file, second_latest_date = unique_file_list[1]
    print(f"Using previous file (same quarter): {os.path.basename(second_latest_file)} (dated {second_latest_date.strftime('%B %Y')})")
    return second_latest_file


def find_latest_affiliated_entity() -> Optional[str]:
    """Find the latest Affiliated Entity file."""
    # Try ownership directory first with both naming patterns
    # Pattern 1: Nursing_Home_Chain_Performance_Measures_*.csv (newer format)
    file_path = find_latest_file('ownership', 'Nursing_Home_Chain_Performance_Measures_*.csv')
    if not file_path:
        # Pattern 2: Nursing_Home_Affiliated_Entity_Performance_Measures_*.csv (older format)
        file_path = find_latest_file('ownership', 'Nursing_Home_Affiliated_Entity_Performance_Measures_*.csv')
    if not file_path:
        # Try root directory
        file_path = find_latest_file('.', 'Nursing_Home_Chain_Performance_Measures_*.csv')
    if not file_path:
        file_path = find_latest_file('.', 'Nursing_Home_Affiliated_Entity_Performance_Measures_*.csv')
    return file_path


def find_previous_affiliated_entity() -> Optional[str]:
    """Find the most recent Affiliated Entity file from a different quarter (for comparisons)."""
    # Get all files matching both patterns
    files_chain = find_files_with_pattern('ownership', 'Nursing_Home_Chain_Performance_Measures_*.csv')
    files_affiliated = find_files_with_pattern('ownership', 'Nursing_Home_Affiliated_Entity_Performance_Measures_*.csv')
    
    # Also try root directory
    files_chain_root = find_files_with_pattern('.', 'Nursing_Home_Chain_Performance_Measures_*.csv')
    files_affiliated_root = find_files_with_pattern('.', 'Nursing_Home_Affiliated_Entity_Performance_Measures_*.csv')
    
    # Combine all files
    all_files = files_chain + files_affiliated + files_chain_root + files_affiliated_root
    
    if len(all_files) < 2:
        return None
    
    # Remove duplicates by filename
    unique_files = {}
    for file_path, date in all_files:
        filename = os.path.basename(file_path)
        if filename not in unique_files:
            unique_files[filename] = (file_path, date)
    
    # Sort by date (newest first)
    unique_file_list = list(unique_files.values())
    unique_file_list.sort(key=lambda x: x[1], reverse=True)
    
    if len(unique_file_list) < 2:
        return None
    
    # Get the latest file and its quarter
    latest_file, latest_date = unique_file_list[0]
    latest_quarter = (latest_date.year, (latest_date.month - 1) // 3 + 1)
    
    # Find the most recent file from a different quarter
    for file_path, file_date in unique_file_list[1:]:
        file_quarter = (file_date.year, (file_date.month - 1) // 3 + 1)
        if file_quarter != latest_quarter:
            print(f"Using previous quarter ownership file: {os.path.basename(file_path)} (dated {file_date.strftime('%B %Y')}, Q{file_quarter[1]} {file_quarter[0]})")
            return file_path
    
    # If no different quarter found, fall back to second most recent
    second_latest_file, second_latest_date = unique_file_list[1]
    print(f"Using previous ownership file (same quarter): {os.path.basename(second_latest_file)} (dated {second_latest_date.strftime('%B %Y')})")
    return second_latest_file


if __name__ == "__main__":
    # Test the file finder
    print("Testing file finder...")
    print("\n=== Provider Info Files ===")
    latest = find_latest_provider_info()
    if latest:
        print(f"Latest: {latest}")
    
    previous = find_previous_provider_info()
    if previous:
        print(f"Previous: {previous}")
    
    print("\n=== Affiliated Entity Files ===")
    entity = find_latest_affiliated_entity()
    if entity:
        print(f"Latest: {entity}")
    
    print("\n=== All Provider Info Files ===")
    all_files = find_all_matching_files('provider_info', 'NH_ProviderInfo_*.csv')
    for i, file in enumerate(all_files, 1):
        print(f"{i}. {os.path.basename(file)}")

