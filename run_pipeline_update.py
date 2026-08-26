"""
Pipeline Update Orchestrator

Automatically detects new quarterly/monthly data files and runs only necessary
processing steps incrementally.

Usage:
    python run_pipeline_update.py [--dry-run] [--verbose] [--force] [--only STEP]
    
    --dry-run: Print what would happen without making changes
    --verbose: Extra logging output
    --force: Re-run steps even if outputs exist
    --only STEP: Run only specific step (pbj, nonnurse, providerinfo, metrics, region, lite, ownership)
"""

import argparse
import logging
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import List, Set, Optional, Tuple
import re
import glob

# Setup logging
LOG_DIR = Path('logs')
LOG_DIR.mkdir(exist_ok=True)

# Configure logging
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Console handler with encoding error handling for Windows
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
# Set encoding to handle Unicode characters gracefully on Windows
if sys.platform == 'win32':
    try:
        import codecs
        # Try to set UTF-8 encoding for console output
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        # Fallback: handler will use errors='replace' behavior
        pass

# File handler (rotating)
from logging.handlers import RotatingFileHandler
file_handler = RotatingFileHandler(
    LOG_DIR / 'pipeline_update.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setFormatter(log_formatter)

# Root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(console_handler)
logger.addHandler(file_handler)


# Month name to number mapping
MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}


def parse_quarter_from_filename(filename: str) -> Optional[str]:
    """Extract quarter string (e.g., '2025Q3') from filename."""
    match = re.search(r'CY(\d{4})Q(\d)', filename)
    if match:
        year = match.group(1)
        quarter = match.group(2)
        return f"{year}Q{quarter}"
    return None


def parse_month_year_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    """Extract (year, month_num) from filename like NH_ProviderInfo_Oct2025.csv."""
    match = re.search(r'([A-Za-z]{3})(\d{4})', filename)
    if match:
        month_str = match.group(1)
        year = int(match.group(2))
        month_num = MONTH_MAP.get(month_str.capitalize())
        if month_num:
            return (year, month_num)
    return None


def normalize_nonnurse_filename(filename: str) -> str:
    """Normalize non-nurse filename to canonical lowercase format."""
    # Convert various casing patterns to canonical lowercase
    # Examples: PBJ_dailyNonnurseStaffing, PBJ_dailyNonNurseStaffing, PBJ_dailynonnurseStaffing
    # All should become: PBJ_dailynonnursestaffing
    # Use case-insensitive replacement
    normalized = re.sub(
        r'(PBJ_daily)(.*?)(nonnurse|non.nurse)(.*?)(staffing)',
        r'\1nonnursestaffing',
        filename,
        flags=re.IGNORECASE
    )
    return normalized.lower()


def detect_new_nurse_pbj_files(force: bool = False) -> List[Path]:
    """Detect new nurse PBJ files that need standardization."""
    input_dir = Path('PBJcsv')
    output_dir = Path('standardized_PBJ')
    
    if not input_dir.exists():
        logger.warning(f"Input directory {input_dir} does not exist")
        return []
    
    output_dir.mkdir(exist_ok=True)
    
    # Find all input files
    input_files = list(input_dir.glob('PBJ_dailynursestaffing_CY*.csv'))
    existing_outputs = {f.name for f in output_dir.glob('PBJ_dailynursestaffing_*.csv')}
    
    new_files = []
    for input_file in input_files:
        if force or input_file.name not in existing_outputs:
            new_files.append(input_file)
    
    logger.info(f"Nurse PBJ: Found {len(input_files)} input files, {len(existing_outputs)} existing outputs, {len(new_files)} new files")
    return new_files


def detect_new_nonnurse_pbj_files(force: bool = False) -> List[Path]:
    """Detect new non-nurse PBJ files that need standardization."""
    input_dir = Path('NonNursecsv')
    output_dir = Path('standardized_NonNurse')
    
    if not input_dir.exists():
        logger.warning(f"Input directory {input_dir} does not exist")
        return []
    
    output_dir.mkdir(exist_ok=True)
    
    # Find all input files (both casing patterns)
    input_files = []
    for pattern in ['PBJ_dailynonnursestaffing_*.csv', 'PBJ_dailyNonnurseStaffing_*.csv', 'PBJ_dailyNonnurseStaffing_*.csv']:
        input_files.extend(input_dir.glob(pattern))
    
    # Remove duplicates
    input_files = list(set(input_files))
    
    # Normalize to canonical names for comparison
    existing_outputs = {normalize_nonnurse_filename(f.name) for f in output_dir.glob('PBJ_dailynonnursestaffing_*.csv')}
    
    new_files = []
    for input_file in input_files:
        canonical_name = normalize_nonnurse_filename(input_file.name)
        if force or canonical_name not in existing_outputs:
            new_files.append(input_file)
    
    logger.info(f"Non-nurse PBJ: Found {len(input_files)} input files, {len(existing_outputs)} existing outputs, {len(new_files)} new files")
    return new_files


def detect_new_provider_info_files(force: bool = False) -> List[Path]:
    """Detect new provider info files that need normalization.

    Git LFS pointer stubs are never processable input — skip/reject them.
    """
    input_dir = Path('provider_info')
    output_dir = Path('provider_info_normalized')
    
    if not input_dir.exists():
        logger.warning(f"Input directory {input_dir} does not exist")
        return []
    
    output_dir.mkdir(exist_ok=True)
    
    # Find all input files
    input_files = list(input_dir.glob('NH_ProviderInfo_*.csv'))
    
    # Check which ones have normalized outputs
    new_files = []
    skipped_lfs = 0
    for input_file in input_files:
        if _is_git_lfs_pointer(input_file):
            skipped_lfs += 1
            logger.warning(
                f"Skipping Git LFS pointer (not real CMS data): {input_file.name}"
            )
            continue
        date_info = parse_month_year_from_filename(input_file.name)
        if date_info:
            year, month = date_info
            expected_output = output_dir / f"ProviderInfoNorm_{year}_{month:02d}.csv"
            if force or not expected_output.exists():
                new_files.append(input_file)
        else:
            logger.warning(f"Could not parse date from filename: {input_file.name}")
    
    logger.info(
        f"Provider Info: Found {len(input_files)} input files, "
        f"{skipped_lfs} LFS pointers skipped, {len(new_files)} new files"
    )
    return new_files


def _is_git_lfs_pointer(path: Path) -> bool:
    """True if path is a Git LFS pointer stub (defense in depth vs normalize)."""
    try:
        with path.open("rb") as f:
            head = f.read(128)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def detect_latest_ownership_file() -> Optional[Path]:
    """Detect latest ownership chain file."""
    ownership_dir = Path('ownership')
    
    if not ownership_dir.exists():
        logger.warning(f"Ownership directory {ownership_dir} does not exist")
        return None
    
    # Try to use file_finder utility
    try:
        from utils.file_finder import find_latest_affiliated_entity
        file_path = find_latest_affiliated_entity()
        if file_path:
            return Path(file_path)
    except ImportError:
        logger.warning("Could not import utils.file_finder, using fallback method")
    
    # Fallback: manual search
    patterns = [
        'Nursing_Home_Chain_Performance_Measures_*.csv',
        'Nursing_Home_Affiliated_Entity_Performance_Measures_*.csv'
    ]
    
    all_files = []
    for pattern in patterns:
        all_files.extend(ownership_dir.glob(pattern))
    
    if not all_files:
        return None
    
    # Sort by parsed date (use file_finder's parse_date_from_filename if available)
    try:
        from utils.file_finder import parse_date_from_filename
        files_with_dates = []
        for f in all_files:
            date = parse_date_from_filename(f.name)
            if date:
                files_with_dates.append((f, date))
        if files_with_dates:
            files_with_dates.sort(key=lambda x: x[1], reverse=True)
            return files_with_dates[0][0]
    except ImportError:
        pass
    
    # Last resort: use modification time
    return max(all_files, key=lambda f: f.stat().st_mtime)


def detect_new_quarters_in_metrics() -> Set[str]:
    """Detect new quarters in standardized files that aren't in metrics yet."""
    standardized_dir = Path('standardized_PBJ')
    metrics_file = Path('facility_quarterly_metrics.csv')
    
    if not standardized_dir.exists():
        return set()
    
    # Get all quarters from standardized files
    standardized_files = list(standardized_dir.glob('PBJ_dailynursestaffing_*.csv'))
    quarters_in_files = set()
    for f in standardized_files:
        quarter = parse_quarter_from_filename(f.name)
        if quarter:
            quarters_in_files.add(quarter)
    
    # Get quarters already in metrics
    quarters_in_metrics = set()
    if metrics_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(metrics_file, low_memory=False)
            if 'CY_Qtr' in df.columns:
                quarters_in_metrics = set(df['CY_Qtr'].unique())
        except Exception as e:
            logger.warning(f"Could not read existing metrics file: {e}")
    
    new_quarters = quarters_in_files - quarters_in_metrics
    logger.info(f"Metrics: Found {len(quarters_in_files)} quarters in files, {len(quarters_in_metrics)} in metrics, {len(new_quarters)} new")
    return new_quarters


def run_script(script_name: str, dry_run: bool = False, verbose: bool = False) -> bool:
    """Run a Python script using subprocess."""
    script_path = Path(script_name)
    if not script_path.exists():
        logger.error(f"Script not found: {script_name}")
        return False
    
    if dry_run:
        logger.info(f"[DRY RUN] Would run: {script_name}")
        return True
    
    logger.info(f"Running: {script_name}")
    try:
        # Set environment to use UTF-8 encoding for subprocess
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        if sys.platform == 'win32':
            env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=not verbose,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env,
            check=False
        )
        
        if result.returncode != 0:
            logger.error(f"Script {script_name} failed with exit code {result.returncode}")
            if not verbose and result.stderr:
                logger.error(f"Error output: {result.stderr[:500]}")
            return False
        
        if verbose and result.stdout:
            logger.info(f"Output from {script_name}:\n{result.stdout}")
        
        logger.info(f"[OK] Completed: {script_name}")
        return True
    except Exception as e:
        logger.error(f"Error running {script_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Pipeline Update Orchestrator - Detects and processes new data files'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Print what would happen without making changes')
    parser.add_argument('--verbose', action='store_true',
                       help='Extra logging output')
    parser.add_argument('--force', action='store_true',
                       help='Re-run steps even if outputs exist')
    parser.add_argument('--only', nargs='+', choices=['pbj', 'nonnurse', 'providerinfo', 'metrics', 'region', 'lite', 'ownership'],
                       help='Run only specific step(s)')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info("="*70)
    logger.info("Pipeline Update Orchestrator")
    logger.info("="*70)
    if args.dry_run:
        logger.info("DRY RUN MODE - No changes will be made")
    if args.force:
        logger.info("FORCE MODE - Will re-run steps even if outputs exist")
    if args.only:
        logger.info(f"ONLY MODE - Running only: {', '.join(args.only)}")
    logger.info("")
    
    # Track what was processed
    summary = {
        'nurse_quarters': [],
        'nonnurse_quarters': [],
        'provider_info_months': [],
        'metrics_quarters': [],
        'ownership_latest': None,
        'errors': []
    }
    
    # Determine which steps to run
    run_all = args.only is None
    run_pbj = run_all or 'pbj' in args.only
    run_nonnurse = run_all or 'nonnurse' in args.only
    run_providerinfo = run_all or 'providerinfo' in args.only
    run_metrics = run_all or 'metrics' in args.only
    run_region = run_all or 'region' in args.only
    run_lite = run_all or 'lite' in args.only
    run_ownership = run_all or 'ownership' in args.only
    
    # Step 1: Nurse PBJ Standardization
    if run_pbj:
        logger.info("\n" + "="*70)
        logger.info("Step 1: Nurse PBJ Standardization")
        logger.info("="*70)
        new_nurse_files = detect_new_nurse_pbj_files(force=args.force)
        if new_nurse_files:
            for f in new_nurse_files:
                quarter = parse_quarter_from_filename(f.name)
                if quarter:
                    summary['nurse_quarters'].append(quarter)
            if not args.dry_run:
                success = run_script('standardize_pbj_files.py', dry_run=args.dry_run, verbose=args.verbose)
                if not success:
                    summary['errors'].append('Nurse PBJ standardization failed')
        else:
            logger.info("No new nurse PBJ files to process")
    
    # Step 2: Non-Nurse PBJ Standardization
    if run_nonnurse:
        logger.info("\n" + "="*70)
        logger.info("Step 2: Non-Nurse PBJ Standardization")
        logger.info("="*70)
        new_nonnurse_files = detect_new_nonnurse_pbj_files(force=args.force)
        if new_nonnurse_files:
            for f in new_nonnurse_files:
                quarter = parse_quarter_from_filename(f.name)
                if quarter:
                    summary['nonnurse_quarters'].append(quarter)
            if not args.dry_run:
                success = run_script('standardize_nonnursepbj_files.py', dry_run=args.dry_run, verbose=args.verbose)
                if not success:
                    summary['errors'].append('Non-nurse PBJ standardization failed')
        else:
            logger.info("No new non-nurse PBJ files to process")
    
    # Step 3: Provider Info Normalization
    if run_providerinfo:
        logger.info("\n" + "="*70)
        logger.info("Step 3: Provider Info Normalization")
        logger.info("="*70)
        new_provider_files = detect_new_provider_info_files(force=args.force)
        if new_provider_files:
            for f in new_provider_files:
                date_info = parse_month_year_from_filename(f.name)
                if date_info:
                    year, month = date_info
                    summary['provider_info_months'].append(f"{year}-{month:02d}")
            if not args.dry_run:
                success = run_script('normalize_provider_info.py', dry_run=args.dry_run, verbose=args.verbose)
                if not success:
                    summary['errors'].append('Provider info normalization failed')
        else:
            logger.info("No new provider info files to process")
    
    # Step 4: Ownership File Validation
    if run_ownership:
        logger.info("\n" + "="*70)
        logger.info("Step 4: Ownership File Validation")
        logger.info("="*70)
        latest_ownership = detect_latest_ownership_file()
        if latest_ownership:
            summary['ownership_latest'] = latest_ownership.name
            logger.info(f"Latest ownership file: {latest_ownership.name}")
        else:
            logger.warning("No ownership chain files found in ownership/ directory")
            summary['errors'].append('No ownership files found')
    
    # Step 5: Metrics Generation (only if new nurse quarters or force)
    if run_metrics:
        logger.info("\n" + "="*70)
        logger.info("Step 5: Metrics Generation")
        logger.info("="*70)
        new_quarters = detect_new_quarters_in_metrics()
        if args.force or new_quarters:
            if not args.dry_run:
                success = run_script('generate_metrics.py', dry_run=args.dry_run, verbose=args.verbose)
                if not success:
                    summary['errors'].append('Metrics generation failed')
                else:
                    # Update summary with quarters that were processed
                    summary['metrics_quarters'] = sorted(new_quarters) if new_quarters else ['all (forced)']
            else:
                logger.info(f"[DRY RUN] Would generate metrics for quarters: {sorted(new_quarters)}")
        else:
            logger.info("No new quarters detected - skipping metrics generation")
    
    # Step 6: Region Metrics (regenerates all)
    if run_region:
        logger.info("\n" + "="*70)
        logger.info("Step 6: Regional Metrics Generation")
        logger.info("="*70)
        if not args.dry_run:
            success = run_script('generate_region_metrics.py', dry_run=args.dry_run, verbose=args.verbose)
            if not success:
                summary['errors'].append('Region metrics generation failed')
        else:
            logger.info("[DRY RUN] Would regenerate region metrics from state metrics")
    
    # Step 7: Lite Metrics (regenerates all)
    if run_lite:
        logger.info("\n" + "="*70)
        logger.info("Step 7: Lite Metrics Generation")
        logger.info("="*70)
        if not args.dry_run:
            success = run_script('lite_report.py', dry_run=args.dry_run, verbose=args.verbose)
            if not success:
                summary['errors'].append('Lite metrics generation failed')
        else:
            logger.info("[DRY RUN] Would regenerate lite metrics from quarterly metrics")
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("SUMMARY")
    logger.info("="*70)
    logger.info(f"New nurse quarters processed: {len(summary['nurse_quarters'])} {summary['nurse_quarters']}")
    logger.info(f"New non-nurse quarters processed: {len(summary['nonnurse_quarters'])} {summary['nonnurse_quarters']}")
    logger.info(f"Provider info months processed: {len(summary['provider_info_months'])} {summary['provider_info_months']}")
    logger.info(f"Metrics quarters generated: {len(summary['metrics_quarters'])} {summary['metrics_quarters']}")
    if summary['ownership_latest']:
        logger.info(f"Latest ownership file: {summary['ownership_latest']}")
    else:
        logger.warning("No ownership file found")
    
    if summary['errors']:
        logger.error(f"\nErrors encountered: {len(summary['errors'])}")
        for error in summary['errors']:
            logger.error(f"  - {error}")
        return 1
    
    logger.info("\n[SUCCESS] Pipeline update completed successfully")
    return 0


if __name__ == '__main__':
    sys.exit(main())
