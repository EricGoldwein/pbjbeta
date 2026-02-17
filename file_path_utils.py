"""
Utility functions for finding files in the new organized project structure.
All scripts should use these functions to locate facility and donor files.
"""

import os
from pathlib import Path
from typing import Optional, List


def find_facility_file(provnum: str, filename: str) -> Optional[str]:
    """
    Find a facility-specific file in the organized structure.
    
    Checks locations in order:
    1. pbj320-XXXXX/filename (new organized location)
    2. filename (root directory, for backwards compatibility)
    
    Args:
        provnum: Facility provider number (6 digits, with or without leading zeros)
        filename: Name of the file to find (e.g., 'facility_XXXXX_complete_data.csv')
    
    Returns:
        Full path to the file if found, None otherwise
    """
    provnum = str(provnum).strip().zfill(6)
    
    # Replace XXXXX in filename with actual provnum if needed
    if 'XXXXX' in filename:
        filename = filename.replace('XXXXX', provnum)
    elif provnum not in filename:
        # If filename doesn't contain provnum, try to insert it
        # This handles cases like 'complete_data.csv' -> 'facility_XXXXX_complete_data.csv'
        if 'facility_' in filename:
            filename = filename.replace('facility_', f'facility_{provnum}_')
        else:
            filename = f'facility_{provnum}_{filename}'
    
    # Check organized location first (deployments/pbj320-XXXXX folder)
    organized_path = Path('deployments') / f'pbj320-{provnum}' / filename
    if organized_path.exists():
        return str(organized_path)
    
    # Check old root location for backwards compatibility
    old_path = Path(f'pbj320-{provnum}') / filename
    if old_path.exists():
        return str(old_path)
    
    # Check root directory for backwards compatibility
    root_path = Path(filename)
    if root_path.exists():
        return str(root_path)
    
    return None


def find_facility_complete_data(provnum: str) -> Optional[str]:
    """Find facility complete data CSV file."""
    return find_facility_file(provnum, f'facility_{provnum}_complete_data.csv')


def find_facility_provider_info(provnum: str) -> Optional[str]:
    """Find facility provider info CSV file."""
    return find_facility_file(provnum, f'facility_{provnum}_provider_info_data.csv')


def find_facility_flask_app(provnum: str) -> Optional[str]:
    """Find facility Flask app file."""
    return find_facility_file(provnum, f'facility_{provnum}_flask_app.py')


def get_facility_folder(provnum: str) -> Path:
    """
    Get the pbj320-XXXXX folder path for a facility in deployments/.
    Creates the folder if it doesn't exist.
    
    Args:
        provnum: Facility provider number
    
    Returns:
        Path object for the facility folder
    """
    provnum = str(provnum).strip().zfill(6)
    deployments_dir = Path('deployments')
    deployments_dir.mkdir(exist_ok=True)
    folder = deployments_dir / f'pbj320-{provnum}'
    folder.mkdir(exist_ok=True)
    return folder


def find_donor_file(filename: str) -> Optional[str]:
    """
    Find a donor-related file in the donor/ folder.
    
    Args:
        filename: Name of the file to find
    
    Returns:
        Full path to the file if found, None otherwise
    """
    donor_path = Path('donor') / filename
    if donor_path.exists():
        return str(donor_path)
    
    # Check root for backwards compatibility
    root_path = Path(filename)
    if root_path.exists():
        return str(root_path)
    
    return None


def get_all_facility_folders() -> List[str]:
    """Get list of all pbj320-XXXXX folder names."""
    folders = []
    deployments_dir = Path('deployments')
    if deployments_dir.exists():
        for item in os.listdir(deployments_dir):
            if os.path.isdir(deployments_dir / item) and item.startswith('pbj320-'):
                folders.append(item)
    # Also check root for backwards compatibility
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('pbj320-'):
            folders.append(item)
    return sorted(set(folders))


def check_file_exists_in_locations(provnum: str, filename: str) -> bool:
    """
    Check if a file exists in any of the expected locations.
    
    Returns:
        True if file exists, False otherwise
    """
    return find_facility_file(provnum, filename) is not None
