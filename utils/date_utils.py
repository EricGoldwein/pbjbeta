"""
Date utilities for dynamic date references in the PBJ Dashboard.
Automatically determines the latest data periods based on available files.
"""
import os
import sys
from datetime import datetime
from typing import Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils.file_finder import find_latest_provider_info, find_previous_provider_info, find_latest_affiliated_entity, find_previous_affiliated_entity
except ImportError:
    # Fallback for direct execution
    from file_finder import find_latest_provider_info, find_previous_provider_info, find_latest_affiliated_entity, find_previous_affiliated_entity


def get_latest_data_periods() -> dict:
    """
    Get the latest data periods for different data sources.
    
    Returns:
        dict with keys: 'provider_info_latest', 'provider_info_previous', 'affiliated_entity_latest', 'data_range', 'quarter_count'
    """
    # Get latest provider info file
    latest_provider_file = find_latest_provider_info()
    previous_provider_file = find_previous_provider_info()
    latest_entity_file = find_latest_affiliated_entity()
    previous_entity_file = find_previous_affiliated_entity()
    
    # Parse dates from filenames
    provider_latest_date = _parse_date_from_filename(latest_provider_file) if latest_provider_file else None
    provider_previous_date = _parse_date_from_filename(previous_provider_file) if previous_provider_file else None
    entity_latest_date = _parse_date_from_filename(latest_entity_file) if latest_entity_file else None
    entity_previous_date = _parse_date_from_filename(previous_entity_file) if previous_entity_file else None
    
    # Format dates for display
    provider_latest_str = _format_date_for_display(provider_latest_date) if provider_latest_date else "Latest Available"
    provider_previous_str = _format_date_for_display(provider_previous_date) if provider_previous_date else "Previous Available"
    entity_latest_str = _format_date_for_display(entity_latest_date) if entity_latest_date else "Latest Available"
    entity_previous_str = _format_date_for_display(entity_previous_date) if entity_previous_date else "Previous Available"
    
    # Calculate dynamic data range and quarter count
    data_range, quarter_count = _calculate_data_range_and_quarters()
    
    return {
        'provider_info_latest': provider_latest_str,
        'provider_info_previous': provider_previous_str,
        'affiliated_entity_latest': entity_latest_str,
        'affiliated_entity_previous': entity_previous_str,
        'data_range': data_range,
        'quarter_count': quarter_count,
        'current_year': datetime.now().year
    }


def _calculate_data_range_and_quarters() -> tuple:
    """Calculate the data range and quarter count from available PBJ files."""
    import glob
    import os
    
    # Look for PBJ files in standardized_PBJ directory
    pbj_files = glob.glob('standardized_PBJ/PBJ_dailynursestaffing_*.csv')
    
    if not pbj_files:
        # Fallback to PBJcsv directory
        pbj_files = glob.glob('PBJcsv/PBJ_dailynursestaffing_*.csv')
    
    if not pbj_files:
        # Default fallback
        return "2017-2025", 33
    
    # Extract years from filenames
    years = set()
    for file_path in pbj_files:
        filename = os.path.basename(file_path)
        # Extract year from filename like "PBJ_dailynursestaffing_CY2025Q1.csv"
        import re
        match = re.search(r'CY(\d{4})Q', filename)
        if match:
            years.add(int(match.group(1)))
    
    if not years:
        return "2017-2025", 33
    
    min_year = min(years)
    max_year = max(years)
    
    # Calculate quarter count (4 quarters per year)
    quarter_count = (max_year - min_year + 1) * 4
    
    return f"{min_year}-{max_year}", quarter_count


def _parse_date_from_filename(file_path: str) -> Optional[datetime]:
    """Parse date from CMS filename."""
    if not file_path:
        return None
    
    filename = os.path.basename(file_path)
    
    # Month name to number mapping
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12,
        'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
        'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    
    import re
    
    # Pattern 1: Month name followed by year (e.g., Sep2025, Mar_2025)
    pattern = r'([A-Za-z]+)_?(\d{4})'
    match = re.search(pattern, filename)
    
    if match:
        month_str = match.group(1)
        year_str = match.group(2)
        
        # Find matching month
        for month_name, month_num in month_map.items():
            if month_str.lower() == month_name.lower():
                try:
                    return datetime(int(year_str), month_num, 1)
                except ValueError:
                    continue
    
    return None


def _format_date_for_display(date_obj: datetime) -> str:
    """Format datetime object for display."""
    if not date_obj:
        return "Latest Available"
    
    return date_obj.strftime("%B %Y")


def get_dynamic_text_replacements() -> dict:
    """
    Get a dictionary of text replacements for dynamic content.
    
    Returns:
        dict with common text patterns and their dynamic replacements
    """
    periods = get_latest_data_periods()
    
    return {
        # Provider Info references
        'September 2025': periods['provider_info_latest'],
        'June 2025': periods['provider_info_previous'],
        'March 2025': periods['provider_info_previous'],
        
        # Affiliated Entity references  
        'Affiliated Entity (September 2025)': f"Affiliated Entity ({periods['affiliated_entity_latest']})",
        'Affiliated Entity (June 2025)': f"Affiliated Entity ({periods['affiliated_entity_latest']})",
        
        # Data range references
        '2017-2025': periods['data_range'],
        '2017 to 2025': f"2017 to {periods['current_year']}",
        
        # CMS references with dates
        'CMS Provider Info (September 2025, June 2025)': f"CMS Provider Info ({periods['provider_info_latest']}, {periods['provider_info_previous']})",
        'CMS Provider Info (September 2025)': f"CMS Provider Info ({periods['provider_info_latest']})",
        
        # Chart annotations
        'Source: CMS Provider Info (September 2025)': f"Source: CMS Provider Info ({periods['provider_info_latest']})",
        'Source: CMS PBJ Data (2017-2025)': f"Source: CMS PBJ Data ({periods['data_range']})",
        
        # Methodology references
        'PBJ data (2017–2025)': f"PBJ data ({periods['data_range']})",
        'PBJ data (2017 to 2025)': f"PBJ data (2017 to {periods['current_year']})",
        
        # Trend comparisons
        'vs. March 2025': f"vs. {periods['provider_info_previous']}",
        'vs. June 2025': f"vs. {periods['provider_info_previous']}",
        
        # Federal minimum reference
        'recently overturned by a court in 2025': f"recently overturned by a court in {periods['current_year']}"
    }


def apply_dynamic_replacements(text: str) -> str:
    """
    Apply dynamic text replacements to a string.
    
    Args:
        text: The text to process
        
    Returns:
        Text with dynamic replacements applied
    """
    replacements = get_dynamic_text_replacements()
    
    for old_text, new_text in replacements.items():
        text = text.replace(old_text, new_text)
    
    return text


if __name__ == "__main__":
    # Test the date utilities
    print("Testing date utilities...")
    periods = get_latest_data_periods()
    print(f"Latest provider info: {periods['provider_info_latest']}")
    print(f"Previous provider info: {periods['provider_info_previous']}")
    print(f"Latest affiliated entity: {periods['affiliated_entity_latest']}")
    print(f"Data range: {periods['data_range']}")
    print(f"Current year: {periods['current_year']}")
    
    print("\nTesting text replacements...")
    test_text = "This uses CMS Provider Info (September 2025, June 2025) and PBJ data (2017-2025)."
    dynamic_text = apply_dynamic_replacements(test_text)
    print(f"Original: {test_text}")
    print(f"Dynamic: {dynamic_text}")
