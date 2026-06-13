"""
Utility functions for finding files in the new organized project structure.
All scripts should use these functions to locate facility and donor files.

Canonical per-facility layout (preferred):
  deployments/pbj320-<CCN>/
    facility_<CCN>_complete_data.csv
    facility_<CCN>_nonnurse_daily.csv
    facility_<CCN>_citations.csv
    facility_<CCN>_provider_info_data.csv
    facility_<CCN>_flask_app.py
    facility_<CCN>_ein_{job_quarterly|category_quarterly|employee_detail|nursing_summaries}.parquet|.csv

CMS national sources are **not** stored under ``deployments/``. See ``cms_data_paths.py``:

- Nurse: ``standardized_PBJ/`` (from ``PBJcsv/``)
- Non-nurse: ``standardized_NonNurse/`` (from ``NonNursecsv/``)
- EIN: ``EIN/monolithic/`` (multi-quarter PUF) + ``EIN/quarters/`` (newer single-quarter zips)

Per-facility Parquet/CSV slices are written to ``deployments/pbj320-<CCN>/`` — where the
superdynamic v2 Vercel bundle loads them (same directory as ``facility_*_flask_app.py``).
``create_vercel_deployment.py`` uses ``packaging_refresh_gates`` to refresh slices when national
quarters move ahead of deploy artifacts.

The repo root still works as a legacy fallback for older workflows.

Archived CSV pairs may also live under ``Facility Reports/csv/`` (see ``find_facility_file``).
"""

import os
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent


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
    
    # Check organized location first (repo-root deployments/ preferred; cwd fallback)
    for organized_path in (
        _REPO_ROOT / "deployments" / f"pbj320-{provnum}" / filename,
        Path("deployments") / f"pbj320-{provnum}" / filename,
    ):
        if organized_path.exists():
            return str(organized_path.resolve())

    # Legacy: pbj320-XXXXX/ next to cwd
    old_path = Path(f"pbj320-{provnum}") / filename
    if old_path.exists():
        return str(old_path.resolve())

    # Legacy archive: Facility Reports/csv/ (flat CSV exports)
    fr_csv = _REPO_ROOT / "Facility Reports" / "csv" / filename
    if fr_csv.is_file():
        return str(fr_csv.resolve())

    # Repo root then cwd (legacy loose files at project root)
    for root_path in (_REPO_ROOT / filename, Path(filename)):
        if root_path.exists():
            return str(root_path.resolve())

    return None


def find_facility_complete_data(provnum: str) -> Optional[str]:
    """Find facility complete data CSV file."""
    return find_facility_file(provnum, f'facility_{provnum}_complete_data.csv')


def find_facility_nonnurse_daily(provnum: str) -> Optional[str]:
    """Find per-facility non-nurse daily PBJ CSV (``facility_<CCN>_nonnurse_daily.csv``)."""
    provnum = str(provnum).strip().zfill(6)
    return find_facility_file(provnum, f"facility_{provnum}_nonnurse_daily.csv")


def find_facility_citations(provnum: str) -> Optional[str]:
    """Find per-facility NH health citations slice (``facility_<CCN>_citations.csv``)."""
    provnum = str(provnum).strip().zfill(6)
    return find_facility_file(provnum, f"facility_{provnum}_citations.csv")


def find_facility_provider_info(provnum: str) -> Optional[str]:
    """Find facility provider info CSV file."""
    return find_facility_file(provnum, f'facility_{provnum}_provider_info_data.csv')


def find_facility_flask_app(provnum: str) -> Optional[str]:
    """Find facility Flask app file."""
    return find_facility_file(provnum, f'facility_{provnum}_flask_app.py')


def find_facility_ein_table_base(provnum: str, kind: str) -> Optional[str]:
    """
    Locate a facility EIN artifact (path without extension) for use with
    ``read_facility_ein_parquet_or_csv``.

    kind must be one of: ``job_quarterly``, ``category_quarterly``, ``employee_detail``.

    Search order (first path where either ``.parquet`` or ``.csv`` exists):
      1. deployments/pbj320-<CCN>/facility_<CCN>_ein_<kind>
      2. repo root (directory containing file_path_utils.py)
      3. current working directory

    Returns:
        Full path without extension, or None if no file exists in any location.
    """
    allowed = {"job_quarterly", "category_quarterly", "employee_detail", "nursing_summaries"}
    if kind not in allowed:
        raise ValueError(f"kind must be one of {sorted(allowed)}, got {kind!r}")
    prov = str(provnum).strip().zfill(6)
    stem = f"facility_{prov}_ein_{kind}"
    candidates = [
        _REPO_ROOT / "deployments" / f"pbj320-{prov}" / stem,
        _REPO_ROOT / stem,
        # Vercel / single-folder deploy: EIN files sit next to facility_*_flask_app.py (not under deployments/)
        Path.cwd() / stem,
    ]
    for base in candidates:
        try:
            p = base.resolve()
        except OSError:
            continue
        if p.with_suffix(".parquet").is_file() or p.with_suffix(".csv").is_file():
            return str(p)
    return None


def get_facility_folder(provnum: str) -> Path:
    """
    Return ``deployments/pbj320-XXXXX`` under the repo root (next to this module).
    Creates the folder if needed so paths are stable regardless of cwd.
    """
    provnum = str(provnum).strip().zfill(6)
    folder = _REPO_ROOT / "deployments" / f"pbj320-{provnum}"
    folder.mkdir(parents=True, exist_ok=True)
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
    """Get list of all pbj320-XXXXX folder names under repo ``deployments/`` and cwd."""
    folders = []
    for deployments_dir in (_REPO_ROOT / "deployments", Path("deployments")):
        if deployments_dir.exists():
            for item in os.listdir(deployments_dir):
                if os.path.isdir(deployments_dir / item) and item.startswith("pbj320-"):
                    folders.append(item)
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("pbj320-"):
            folders.append(item)
    return sorted(set(folders))


def check_file_exists_in_locations(provnum: str, filename: str) -> bool:
    """
    Check if a file exists in any of the expected locations.
    
    Returns:
        True if file exists, False otherwise
    """
    return find_facility_file(provnum, filename) is not None
