#!/usr/bin/env python3
"""
Facility Report Library - Generic functions for generating attorney-style HTML reports.
This library contains all the core functions used by generate_facility_report_attorney.py
to generate PBJ facility reports for any facility.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import glob
import re
import zipfile
import json
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, List, Tuple, Union, Sequence, cast


def _xml_escape(s: str) -> str:
    """Escape for use inside XML text so the reference doc's formatting is preserved."""
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )

def round_half_up(value: float, decimals: int = 2) -> float:
    """Round using ROUND_HALF_UP (financial rounding)."""
    # Passive comparison: contract may specify default_decimals (observational only; local is authoritative).
    try:
        from contract_adapter import get_contract, get_value, compare_and_warn
        contract = get_contract()
        contract_decimals = get_value(contract, "formatting.rounding.default_decimals")
        if contract_decimals is not None and decimals == 2:
            compare_and_warn(
                "facility_report_lib.round_half_up",
                2,
                "formatting.rounding.default_decimals",
                authoritative="local",
                reason="Reports use 2 decimals for HPRD; templates may use different decimals.",
            )
    except Exception:
        pass
    if pd.isna(value) or value is None:
        return 0.0
    if decimals == 1:
        return float(Decimal(str(value)).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP))
    elif decimals == 0:
        return float(Decimal(str(value)).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    else:
        return float(Decimal(str(value)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))

def format_quarter_display(quarter: str) -> str:
    """Convert quarter from '2023Q1' format to 'Q1 2023' format for display."""
    # Passive comparison: contract quarter_rules.display.canonical_display documents expected pattern (observational only).
    try:
        from contract_adapter import get_contract, get_value, compare_and_warn
        contract = get_contract()
        contract_display = get_value(contract, "quarter_rules.display.canonical_display")
        if contract_display is not None:
            # Local implements "QN YYYY" (e.g. Q1 2023). Contract documents "QN YYYY".
            local_pattern = "QN YYYY"
            if contract_display != local_pattern:
                compare_and_warn(
                    "facility_report_lib.format_quarter_display",
                    local_pattern,
                    "quarter_rules.display.canonical_display",
                    authoritative="local",
                    reason="Quarter display format affects narratives and headers.",
                )
    except Exception:
        pass
    if not quarter or len(quarter) < 6:
        return quarter
    try:
        year = quarter[:4]
        q_num = quarter[5]
        return f"Q{q_num} {year}"
    except:
        return quarter

def _parse_quarter(q: str) -> Optional[tuple]:
    """Parse a quarter string into (year, quarter) tuple. Returns None if parsing fails.
    Handles formats like "2018Q1", "Q1 2018", "2018 Q1", "2023Q1", "Q1 2023", etc."""
    try:
        import re
        q_str = str(q).strip()
        
        # Handle format "Q1 2023" or "Q1 2024" (most common in output)
        match = re.match(r'Q(\d)\s+(\d{4})', q_str, re.IGNORECASE)
        if match:
            quarter = int(match.group(1))
            year = int(match.group(2))
            if 1 <= quarter <= 4 and 2000 <= year <= 2100:
                return (year, quarter)
        
        # Handle format "2018Q1" or "2023Q1" (year first, no space)
        match = re.match(r'(\d{4})Q(\d)', q_str, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            quarter = int(match.group(2))
            if 1 <= quarter <= 4 and 2000 <= year <= 2100:
                return (year, quarter)
        
        # Handle format "2018 Q1" (year first with space)
        match = re.match(r'(\d{4})\s+Q(\d)', q_str, re.IGNORECASE)
        if match:
            year = int(match.group(1))
            quarter = int(match.group(2))
            if 1 <= quarter <= 4 and 2000 <= year <= 2100:
                return (year, quarter)
        
        # Fallback: try splitting on 'Q'
        q_clean = q_str.upper().replace(' ', '')
        if 'Q' in q_clean:
            parts = q_clean.split('Q')
            if len(parts) == 2:
                # Format "2018Q1" (year first)
                if parts[0].isdigit() and parts[1].isdigit():
                    year = int(parts[0])
                    quarter = int(parts[1])
                    if 1 <= quarter <= 4 and 2000 <= year <= 2100:
                        return (year, quarter)
    except:
        pass
    return None


def _format_quarter_range(quarters: List[str]) -> str:
    """Format quarters as ranges where possible (e.g., 'Q1 2018 - Q4 2020' instead of listing all).
    Quarters are sorted by year first, then quarter (chronologically)."""
    if not quarters:
        return ''
    
    # Parse quarters into (year, quarter) tuples
    quarter_tuples = []
    unparsed = []
    for q in quarters:
        parsed = _parse_quarter(q)
        if parsed:
            quarter_tuples.append(parsed)
        else:
            unparsed.append(q)
    
    # If we couldn't parse any, try more aggressive parsing
    if not quarter_tuples and unparsed:
        import re
        for q in unparsed:
            q_str = str(q).strip()
            # Try "2023Q1" format more aggressively
            match = re.match(r'(\d{4})Q(\d)', q_str, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                quarter = int(match.group(2))
                if 1 <= quarter <= 4 and 2000 <= year <= 2100:
                    quarter_tuples.append((year, quarter))
                    continue
            # Try "Q1 2023" format
            match = re.match(r'Q(\d)\s+(\d{4})', q_str, re.IGNORECASE)
            if match:
                quarter = int(match.group(1))
                year = int(match.group(2))
                if 1 <= quarter <= 4 and 2000 <= year <= 2100:
                    quarter_tuples.append((year, quarter))
                    continue
    
    if not quarter_tuples:
        # If parsing fails completely, return as-is (but this shouldn't happen)
        return ', '.join(quarters)
    
    # Sort by year first, then quarter (year is more important for chronological ordering)
    quarter_tuples.sort(key=lambda x: (x[0], x[1]))
    
    # Try to create ranges
    if len(quarter_tuples) <= 3:
        # If 3 or fewer, just list them
        return ', '.join([f"Q{q} {y}" for y, q in quarter_tuples])
    
    # Group consecutive quarters
    ranges = []
    start = quarter_tuples[0]
    end = start
    
    for i in range(1, len(quarter_tuples)):
        current = quarter_tuples[i]
        # Check if consecutive (same year and next quarter, or next year and Q1)
        is_consecutive = (
            (current[0] == end[0] and current[1] == end[1] + 1) or
            (current[0] == end[0] + 1 and current[1] == 1 and end[1] == 4)
        )
        
        if is_consecutive:
            end = current
        else:
            # End current range, start new one
            if start == end:
                ranges.append(f"Q{start[1]} {start[0]}")
            else:
                ranges.append(f"Q{start[1]} {start[0]} - Q{end[1]} {end[0]}")
            start = current
            end = current
    
    # Add final range
    if start == end:
        ranges.append(f"Q{start[1]} {start[0]}")
    else:
        ranges.append(f"Q{start[1]} {start[0]} - Q{end[1]} {end[0]}")
    
    return ', '.join(ranges)


def format_facility_name(name: str) -> str:
    """
    Format facility name with proper title case.
    Converts ALL CAPS to title case and keeps words like 'at', 'of', etc. lowercase.
    """
    if not name or pd.isna(name):
        return str(name) if name else ""
    
    name = str(name).strip()
    if not name:
        return ""
    
    # Words that should remain lowercase (unless first word)
    lowercase_words = {'and', 'or', 'of', 'at', 'in', 'on', 'for', 'to', 'with', 'by', 'the', 'a', 'an'}
    
    # Handle special cases like "(the)" - move to proper position
    name = name.replace('(the)', 'the').replace('(The)', 'the').replace('(THE)', 'the')
    
    # Split the text into words
    words = name.split()
    result = []
    
    for i, word in enumerate(words):
        # Clean up the word
        word = word.strip()
        if not word:
            continue
        
        # Remove parentheses that might be left
        word = word.replace('(', '').replace(')', '')
        if not word:
            continue
            
        # Always capitalize the first word
        if i == 0:
            result.append(word.capitalize())
        elif word.lower() in lowercase_words:
            # Keep lowercase words lowercase
            result.append(word.lower())
        else:
            # Capitalize other words
            result.append(word.capitalize())
    
    formatted = ' '.join(result)
    
    # Fix common abbreviations that should be uppercase
    formatted = formatted.replace(' Llc', ' LLC').replace(' Inc', ' Inc').replace(' Lp', ' LP')
    formatted = formatted.replace(' Nh', ' NH')
    
    return formatted

def format_city_name(city: str) -> str:
    """Format city name with proper title case."""
    if not city or pd.isna(city):
        return str(city) if city else ""
    
    city = str(city).strip()
    if not city:
        return ""
    
    # Simple title case - capitalize first letter of each word
    return city.title()

def load_facility_data(provnum: str) -> pd.DataFrame:
    """Load facility data from existing complete data CSV file."""
    provnum = str(provnum).strip()
    provnum_6 = provnum.zfill(6)
    
    # Try multiple possible locations (must match file_path_utils so we find files in deployments/)
    possible_files = [
        f'deployments/pbj320-{provnum_6}/facility_{provnum_6}_complete_data.csv',
        f'facility_{provnum}_complete_data.csv',
        f'pbj320-{provnum}/facility_{provnum}_complete_data.csv',
        f'facility_{provnum_6}_complete_data.csv',
    ]
    
    for file_path in possible_files:
        if os.path.exists(file_path):
            print(f"Loading facility data from: {file_path}")
            df = pd.read_csv(file_path, low_memory=False)
            # WorkDate might be in YYYYMMDD format (integer) or string
            if df['WorkDate'].dtype in ['int64', 'float64']:
                df['WorkDate'] = pd.to_datetime(df['WorkDate'].astype(str), format='%Y%m%d', errors='coerce')
            else:
                df['WorkDate'] = pd.to_datetime(df['WorkDate'], errors='coerce')
            return df
    
    print(f"Error: Could not find facility data file for {provnum}")
    return pd.DataFrame()

def get_facility_info(df: pd.DataFrame, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
    """Extract facility information from the dataframe. If date range provided, uses facility name from that period."""
    if df.empty:
        return {}
    
    # If date range provided, filter to that period to get the facility name during the resident stay
    previous_names = []
    if start_date is not None and end_date is not None:
        # Ensure WorkDate is datetime
        if df['WorkDate'].dtype != 'datetime64[ns]':
            df['WorkDate'] = pd.to_datetime(df['WorkDate'], errors='coerce')
        
        # Filter to the date range
        period_df = df[
            (df['WorkDate'] >= start_date) &
            (df['WorkDate'] <= end_date)
        ].copy()
        
        if not period_df.empty:
            # Most recent name in period = name from latest WorkDate
            period_df = period_df.sort_values('WorkDate', ascending=False)
            period_row = period_df.iloc[0]
            facility_name = str(period_row.get('PROVNAME', 'Unknown')).strip()
            # Other distinct names in period (excluding current) = previously
            all_names = period_df['PROVNAME'].dropna().astype(str).str.strip()
            distinct = all_names[all_names.str.len() > 0].unique().tolist()
            current_lower = facility_name.lower()
            previous_names = [n for n in distinct if n.lower() != current_lower]
        else:
            return {}
    else:
        # No date range, use first row
        period_row = df.iloc[0]
        facility_name = period_row.get('PROVNAME', 'Unknown')
    
    return {
        'name': facility_name,
        'city': period_row.get('CITY', 'Unknown'),
        'state': period_row.get('STATE', 'Unknown'),
        'county': period_row.get('COUNTY_NAME', 'Unknown'),
        'provnum': period_row.get('PROVNUM', 'Unknown'),
        'previous_names': previous_names,
    }


def get_previous_names_with_years(provnum: str, provider_info_df: Optional[pd.DataFrame], current_name: str) -> List[Dict[str, str]]:
    """
    Get previous facility names with provider-info year ranges (from processing_date), not quarter.
    Returns list of {"name": str, "years": str} e.g. {"name": "Stonerise Beckley", "years": "2017–2019"}.
    """
    if provider_info_df is None or provider_info_df.empty:
        return []
    name_col = None
    for col in ['provider_name', 'Provider Name', 'PROVNAME', 'facility_name']:
        if col in provider_info_df.columns:
            name_col = col
            break
    if not name_col or 'processing_date' not in provider_info_df.columns:
        return []
    provnum_z = str(provnum).strip().zfill(6)
    ccn_col = None
    for col in ['ccn', 'CCN', 'CMS Certification Number (CCN)']:
        if col in provider_info_df.columns:
            ccn_col = col
            break
    if not ccn_col:
        return []
    df = provider_info_df.copy()
    df[ccn_col] = df[ccn_col].astype(str).str.zfill(6)
    facility_df = df[df[ccn_col] == provnum_z]
    if facility_df.empty:
        return []
    facility_df['processing_date'] = pd.to_datetime(facility_df['processing_date'], errors='coerce')
    facility_df = facility_df.dropna(subset=['processing_date'])
    facility_df['_year'] = facility_df['processing_date'].dt.year
    name_years = {}
    for _, row in facility_df.iterrows():
        name_val = row.get(name_col)
        if pd.isna(name_val) or not str(name_val).strip():
            continue
        name_val = str(name_val).strip()
        year = row.get('_year')
        if pd.isna(year) or year is None:
            continue
        year = int(year)  # type: ignore[arg-type]
        key = name_val.lower()
        if key not in name_years:
            name_years[key] = {'name': name_val, 'years': set()}
        name_years[key]['years'].add(year)
    current_lower = (current_name or '').strip().lower()
    result = []
    for key, data in name_years.items():
        if key == current_lower:
            continue
        years = sorted(data['years'])
        if not years:
            continue
        if years[0] == years[-1]:
            years_str = str(years[0])
        else:
            years_str = f"{years[0]}–{years[-1]}"
        result.append({'name': data['name'], 'years': years_str})
    return result


def get_daily_staffing(df: pd.DataFrame, date: datetime) -> Optional[Dict]:
    """Get staffing data for a specific date."""
    date_only = date.date()
    day_data = df[df['WorkDate'].dt.date == date_only].copy()
    
    if day_data.empty:
        return None
    
    row = day_data.iloc[0]
    census = row.get('MDScensus', 0)
    
    # Calculate hours
    hrs_rn = row.get('Hrs_RN', 0) or 0
    hrs_rnadmin = row.get('Hrs_RNadmin', 0) or 0
    hrs_rndon = row.get('Hrs_RNDON', 0) or 0
    hrs_lpn = row.get('Hrs_LPN', 0) or 0
    hrs_lpnadmin = row.get('Hrs_LPNadmin', 0) or 0
    hrs_cna = row.get('Hrs_CNA', 0) or 0
    hrs_natrn = row.get('Hrs_NAtrn', 0) or 0
    hrs_medaide = row.get('Hrs_MedAide', 0) or 0
    
    total_rn_hours = hrs_rn + hrs_rnadmin + hrs_rndon
    total_lpn_hours = hrs_lpn + hrs_lpnadmin  # Total LPN (includes admin) - for internal use only
    direct_lpn_hours = hrs_lpn  # Direct LPN (excludes admin) - this is what we display as "LPN Hours"
    total_nurse_aide_hours = hrs_cna + hrs_natrn + hrs_medaide
    total_nurse_hours = total_rn_hours + total_lpn_hours + total_nurse_aide_hours
    
    # Calculate HPRD
    rn_hprd = (total_rn_hours / census) if census > 0 else 0
    lpn_hprd = (direct_lpn_hours / census) if census > 0 else 0  # Direct LPN HPRD (excludes admin)
    cna_hprd = (total_nurse_aide_hours / census) if census > 0 else 0
    total_hprd = (total_nurse_hours / census) if census > 0 else 0
    direct_care_rn_hprd = (hrs_rn / census) if census > 0 else 0
    # Direct care hours (excluding admin/DON): RN + LPN + CNA + NAtrn + MedAide
    direct_care_hours = hrs_rn + hrs_lpn + hrs_cna + hrs_natrn + hrs_medaide
    direct_care_hprd = (direct_care_hours / census) if census > 0 else 0
    
    # Contract hours: include all positions (RN, RN Admin, RN DON, LPN, LPN Admin, CNA, Med Aide, NA Trn) to match quarterly/period
    hrs_rn_ctr = row.get('Hrs_RN_ctr', 0) or 0
    hrs_rnadmin_ctr = row.get('Hrs_RNadmin_ctr', 0) or 0
    hrs_rndon_ctr = row.get('Hrs_RNDON_ctr', 0) or 0
    hrs_lpn_ctr = row.get('Hrs_LPN_ctr', 0) or 0
    hrs_lpnadmin_ctr = row.get('Hrs_LPNadmin_ctr', 0) or 0
    hrs_cna_ctr = row.get('Hrs_CNA_ctr', 0) or 0
    hrs_natrn_ctr = row.get('Hrs_NAtrn_ctr', 0) or 0
    hrs_medaide_ctr = row.get('Hrs_MedAide_ctr', 0) or 0
    total_contract_hours = (hrs_rn_ctr + hrs_rnadmin_ctr + hrs_rndon_ctr + hrs_lpn_ctr + hrs_lpnadmin_ctr +
                           hrs_cna_ctr + hrs_natrn_ctr + hrs_medaide_ctr)
    # Contract % = contract hours as % of total nursing hours (all positions, incl. admin/DON)
    contract_pct = (total_contract_hours / total_nurse_hours * 100) if total_nurse_hours > 0 else 0
    
    return {
        'date': date,
        'census': round_half_up(census, 0),
        'hrs_rn': round_half_up(hrs_rn, 2),
        'hrs_rnadmin': round_half_up(hrs_rnadmin, 2),
        'hrs_rndon': round_half_up(hrs_rndon, 2),
        'hrs_lpn': round_half_up(hrs_lpn, 2),
        'direct_care_rn_hours': round_half_up(hrs_rn, 2),
        'hrs_lpnadmin': round_half_up(hrs_lpnadmin, 2),
        'hrs_cna': round_half_up(hrs_cna, 2),
        'hrs_natrn': round_half_up(hrs_natrn, 2),
        'hrs_medaide': round_half_up(hrs_medaide, 2),
        'total_rn_hours': round_half_up(total_rn_hours, 2),
        'total_lpn_hours': round_half_up(total_lpn_hours, 2),  # Total LPN (includes admin) - for internal use
        'direct_lpn_hours': round_half_up(direct_lpn_hours, 2),  # Direct LPN (excludes admin) - this is displayed as "LPN Hours" (rounded to 2 decimals)
        'total_nurse_aide_hours': round_half_up(total_nurse_aide_hours, 2),
        'total_nurse_hours': round_half_up(total_nurse_hours, 2),
        'rn_hprd': round_half_up(rn_hprd, 2),
        'lpn_hprd': round_half_up(lpn_hprd, 2),
        'cna_hprd': round_half_up(cna_hprd, 2),
        'total_hprd': round_half_up(total_hprd, 2),
        'direct_care_hprd': round_half_up(direct_care_hprd, 2),
        'direct_care_rn_hprd': round_half_up(direct_care_rn_hprd, 2),
        'contract_pct': round_half_up(contract_pct, 1),
        'day_of_week': date.strftime('%A')
    }

def calculate_quarterly_metrics(df: pd.DataFrame, quarter: str) -> Optional[Dict]:
    """Calculate quarterly metrics for a facility."""
    quarter_data = df[df['CY_Qtr'] == quarter].copy()
    
    if quarter_data.empty:
        return None
    
    # Calculate totals
    total_resident_days = quarter_data['MDScensus'].sum()
    
    if total_resident_days == 0:
        return None
    
    # Calculate hours
    total_nurse_hours = (
        quarter_data['Hrs_RNDON'].fillna(0) +
        quarter_data['Hrs_RNadmin'].fillna(0) +
        quarter_data['Hrs_RN'].fillna(0) +
        quarter_data['Hrs_LPNadmin'].fillna(0) +
        quarter_data['Hrs_LPN'].fillna(0) +
        quarter_data['Hrs_CNA'].fillna(0) +
        quarter_data['Hrs_NAtrn'].fillna(0) +
        quarter_data['Hrs_MedAide'].fillna(0)
    ).sum()
    
    total_rn_hours = (
        quarter_data['Hrs_RNDON'].fillna(0) +
        quarter_data['Hrs_RNadmin'].fillna(0) +
        quarter_data['Hrs_RN'].fillna(0)
    ).sum()
    
    direct_care_rn_hours = quarter_data['Hrs_RN'].fillna(0).sum()
    
    # Calculate direct care hours (excluding admin/DON): RN + LPN + CNA + NAtrn + MedAide
    direct_care_hours = (
        quarter_data['Hrs_RN'].fillna(0) +
        quarter_data['Hrs_LPN'].fillna(0) +
        quarter_data['Hrs_CNA'].fillna(0) +
        quarter_data['Hrs_NAtrn'].fillna(0) +
        quarter_data['Hrs_MedAide'].fillna(0)
    ).sum()
    
    # Calculate contract hours
    contract_hours = (
        quarter_data['Hrs_RNDON_ctr'].fillna(0) +
        quarter_data['Hrs_RNadmin_ctr'].fillna(0) +
        quarter_data['Hrs_RN_ctr'].fillna(0) +
        quarter_data['Hrs_LPNadmin_ctr'].fillna(0) +
        quarter_data['Hrs_LPN_ctr'].fillna(0) +
        quarter_data['Hrs_CNA_ctr'].fillna(0) +
        quarter_data['Hrs_NAtrn_ctr'].fillna(0) +
        quarter_data['Hrs_MedAide_ctr'].fillna(0)
    ).sum()
    
    # Calculate HPRD
    total_hprd = total_nurse_hours / total_resident_days if total_resident_days > 0 else 0
    rn_hprd = total_rn_hours / total_resident_days if total_resident_days > 0 else 0
    direct_care_rn_hprd = direct_care_rn_hours / total_resident_days if total_resident_days > 0 else 0
    direct_care_hprd = direct_care_hours / total_resident_days if total_resident_days > 0 else 0
    
    # Calculate contract percentage
    contract_pct = (contract_hours / total_nurse_hours * 100) if total_nurse_hours > 0 else 0
    
    # Average census
    avg_census = float(quarter_data['MDScensus'].mean())
    
    return {
        'quarter': quarter,
        'avg_census': round_half_up(avg_census, 1),
        'total_hprd': round_half_up(total_hprd, 2),
        'rn_hprd': round_half_up(rn_hprd, 2),
        'direct_care_rn_hprd': round_half_up(direct_care_rn_hprd, 2),
        'direct_care_hprd': round_half_up(direct_care_hprd, 2),
        'contract_pct': round_half_up(contract_pct, 1),
        'total_resident_days': total_resident_days,
        'total_rn_hours': round_half_up(total_rn_hours, 2)
    }

def calculate_days_under_state_minimum(df: pd.DataFrame, start_date: datetime, end_date: datetime, state_minimum: float) -> Dict:
    """Calculate number of days under state minimum staffing for both Total HPRD and Direct Care HPRD."""
    # Ensure WorkDate is datetime
    if df['WorkDate'].dtype != 'datetime64[ns]':
        df['WorkDate'] = pd.to_datetime(df['WorkDate'], errors='coerce')
    
    period_data = df[
        (df['WorkDate'] >= start_date) &
        (df['WorkDate'] <= end_date) &
        (df['MDScensus'] > 0)  # Only count days with census > 0
    ].copy()
    
    if period_data.empty:
        return {
            'total_days': 0,
            'days_under_minimum_total': 0,
            'percentage_under_total': 0.0,
            'days_under_minimum_direct': 0,
            'percentage_under_direct': 0.0
        }
    
    # Calculate total HPRD for each day (all staff including admin/DON)
    period_data['Total_Nurse_Hours'] = (
        period_data['Hrs_RNDON'].fillna(0) +
        period_data['Hrs_RNadmin'].fillna(0) +
        period_data['Hrs_RN'].fillna(0) +
        period_data['Hrs_LPNadmin'].fillna(0) +
        period_data['Hrs_LPN'].fillna(0) +
        period_data['Hrs_CNA'].fillna(0) +
        period_data['Hrs_NAtrn'].fillna(0) +
        period_data['Hrs_MedAide'].fillna(0)
    )
    period_data['Total_HPRD'] = period_data['Total_Nurse_Hours'] / period_data['MDScensus']
    
    # Calculate direct care HPRD for each day (excluding admin/DON)
    period_data['Direct_Care_Hours'] = (
        period_data['Hrs_RN'].fillna(0) +
        period_data['Hrs_LPN'].fillna(0) +
        period_data['Hrs_CNA'].fillna(0) +
        period_data['Hrs_NAtrn'].fillna(0) +
        period_data['Hrs_MedAide'].fillna(0)
    )
    period_data['Direct_Care_HPRD'] = period_data['Direct_Care_Hours'] / period_data['MDScensus']
    
    # Count days under minimum for both metrics
    days_under_total = (period_data['Total_HPRD'] < state_minimum).sum()
    days_under_direct = (period_data['Direct_Care_HPRD'] < state_minimum).sum()
    total_days = len(period_data)
    percentage_under_total = (days_under_total / total_days * 100) if total_days > 0 else 0.0
    percentage_under_direct = (days_under_direct / total_days * 100) if total_days > 0 else 0.0
    
    return {
        'total_days': total_days,
        'days_under_minimum_total': int(days_under_total),
        'percentage_under_total': round_half_up(percentage_under_total, 1),
        'days_under_minimum_direct': int(days_under_direct),
        'percentage_under_direct': round_half_up(percentage_under_direct, 1)
    }

def calculate_period_metrics(df: pd.DataFrame, start_date: datetime, end_date: datetime) -> Optional[Dict]:
    """Calculate metrics for a specific date range (not just quarters)."""
    # Ensure WorkDate is datetime
    if df['WorkDate'].dtype != 'datetime64[ns]':
        df['WorkDate'] = pd.to_datetime(df['WorkDate'], errors='coerce')
    
    period_data = df[
        (df['WorkDate'] >= start_date) &
        (df['WorkDate'] <= end_date)
    ].copy()
    
    if period_data.empty:
        print(f"    No data found for period {start_date.date()} to {end_date.date()}")
        print(f"    Available date range: {df['WorkDate'].min()} to {df['WorkDate'].max()}")
        return None
    
    # Calculate totals
    total_resident_days = period_data['MDScensus'].sum()
    
    if total_resident_days == 0:
        return None
    
    # Calculate hours
    total_nurse_hours = (
        period_data['Hrs_RNDON'].fillna(0) +
        period_data['Hrs_RNadmin'].fillna(0) +
        period_data['Hrs_RN'].fillna(0) +
        period_data['Hrs_LPNadmin'].fillna(0) +
        period_data['Hrs_LPN'].fillna(0) +
        period_data['Hrs_CNA'].fillna(0) +
        period_data['Hrs_NAtrn'].fillna(0) +
        period_data['Hrs_MedAide'].fillna(0)
    ).sum()
    
    total_rn_hours = (
        period_data['Hrs_RNDON'].fillna(0) +
        period_data['Hrs_RNadmin'].fillna(0) +
        period_data['Hrs_RN'].fillna(0)
    ).sum()
    
    direct_care_rn_hours = period_data['Hrs_RN'].fillna(0).sum()
    
    # Direct care hours (excluding admin/DON) - RN + LPN + CNA + NAtrn + MedAide
    direct_care_hours = (
        period_data['Hrs_RN'].fillna(0) +
        period_data['Hrs_LPN'].fillna(0) +
        period_data['Hrs_CNA'].fillna(0) +
        period_data['Hrs_NAtrn'].fillna(0) +
        period_data['Hrs_MedAide'].fillna(0)
    ).sum()
    
    # Calculate HPRD
    total_hprd = total_nurse_hours / total_resident_days if total_resident_days > 0 else 0
    direct_care_hprd = direct_care_hours / total_resident_days if total_resident_days > 0 else 0
    rn_hprd = total_rn_hours / total_resident_days if total_resident_days > 0 else 0
    direct_care_rn_hprd = direct_care_rn_hours / total_resident_days if total_resident_days > 0 else 0
    
    # Calculate contract hours
    contract_hours = (
        period_data['Hrs_RNDON_ctr'].fillna(0) +
        period_data['Hrs_RNadmin_ctr'].fillna(0) +
        period_data['Hrs_RN_ctr'].fillna(0) +
        period_data['Hrs_LPNadmin_ctr'].fillna(0) +
        period_data['Hrs_LPN_ctr'].fillna(0) +
        period_data['Hrs_CNA_ctr'].fillna(0) +
        period_data['Hrs_NAtrn_ctr'].fillna(0) +
        period_data['Hrs_MedAide_ctr'].fillna(0)
    ).sum()
    
    # Calculate contract percentage
    contract_pct = (contract_hours / total_nurse_hours * 100) if total_nurse_hours > 0 else 0
    
    # Average census
    avg_census = float(period_data['MDScensus'].mean())
    
    return {
        'total_hprd': round_half_up(total_hprd, 2),
        'direct_care_hprd': round_half_up(direct_care_hprd, 2),
        'rn_hprd': round_half_up(rn_hprd, 2),
        'direct_care_rn_hprd': round_half_up(direct_care_rn_hprd, 2),
        'avg_census': round_half_up(avg_census, 1),
        'total_resident_days': total_resident_days,
        'contract_pct': round_half_up(contract_pct, 1)
    }

# Primary source: MACPAC compendium (state policies / legislation summary)
MACPAC_SOURCE_URL = "https://www.macpac.gov/publication/state-policies-related-to-nursing-facility-staffing/"
# State legislation / code links (from MACPAC or official state sources)
# State abbreviation -> full name (used for MACPAC lookup and report headings)
STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
    'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
    'DC': 'District of Columbia', 'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii',
    'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine',
    'MD': 'Maryland', 'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota',
    'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska',
    'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio',
    'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island',
    'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas',
    'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}
# Fallback law links when state not in state_staffing_display.json (can be overridden by JSON)
STATE_LEGISLATION_URLS = {
    "NJ": "https://law.justia.com/codes/new-jersey/2023/title-8/chapter-39/section-8-39-25-1/",
    "NY": "https://www.nysenate.gov/legislation/laws/PBH/2803-D",
    "WV": "https://apps.sos.wv.gov/adlaw/csr/rule.aspx?rule=64-13",
}
STATE_LEGISLATION_LABELS = {
    "NJ": "N.J.A.C. 8:39-25.1",
    "NY": "N.Y. Public Health Law § 2803-d",
    "WV": "W. Va. Code R. § 64-13",
}

_state_staffing_display_cache: Optional[Dict[str, Dict]] = None

def _load_state_staffing_display() -> Dict[str, Dict]:
    """Load per-state display config from state_staffing_display.json (law URLs, labels, enforcement, template)."""
    global _state_staffing_display_cache
    if _state_staffing_display_cache is not None:
        return _state_staffing_display_cache
    for path in ('state_staffing_display.json', 'pbj_lite/state_staffing_display.json'):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    _state_staffing_display_cache = cast(Dict[str, Dict], json.load(f))
                return cast(Dict[str, Dict], _state_staffing_display_cache)
            except Exception:
                pass
    _state_staffing_display_cache = {}
    return cast(Dict[str, Dict], _state_staffing_display_cache)

# Report source URLs for the header Sources line (attorney report)
SOURCES_CMS_PROVIDER_INFO_URL = "https://data.cms.gov/provider-data/dataset/4pq5-n9py"
SOURCES_CMS_PBJ_URL = "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing"
# PBJ Dashboard URL is facility-specific: legacy was https://pbjdashboard.com/?facility={ccn}
# New public site uses https://www.pbj320.com/provider/{ccn}


def get_macpac_state_standards(state: str) -> Optional[Dict]:
    """Get state staffing requirements from MACPAC standards."""
    try:
        # Passive comparison: contract definitions.state_names may differ in keys/count (observational only).
        try:
            from contract_adapter import get_contract, get_value, compare_and_warn
            contract = get_contract()
            contract_state_names = get_value(contract, "definitions.state_names")
            if contract_state_names is not None and hasattr(contract_state_names, "__len__"):
                local_count = len(STATE_NAMES)
                contract_count = len(contract_state_names) if not isinstance(contract_state_names, (str, bytes)) else 0
                if contract_count and local_count != contract_count:
                    compare_and_warn(
                        "facility_report_lib.get_macpac_state_standards",
                        local_count,
                        "definitions.state_names (len)",
                        authoritative="local",
                        reason="State name list length affects MACPAC lookup and display.",
                    )
        except Exception:
            pass

        # Convert abbreviation to full name if needed
        state_to_search = STATE_NAMES.get(state.upper(), state)
        
        possible_paths = [
            'pbj_lite/macpac_state_standards_clean.csv',
            'macpac_state_standards_clean.csv',
        ]
        
        macpac_path = None
        for path in possible_paths:
            if os.path.exists(path):
                macpac_path = path
                break
        
        if not macpac_path:
            print(f"    Warning: MACPAC standards file not found")
            return None
        
        df = pd.read_csv(macpac_path, low_memory=False)
        
        # Match state name (handle variations)
        state_data = df[df['State'].str.upper() == state_to_search.upper()]
        
        if state_data.empty:
            # Try matching without "State" suffix
            state_data = df[df['State'].str.upper().str.replace(' STATE', '') == state_to_search.upper()]
        
        if state_data.empty:
            print(f"    No MACPAC data found for {state} (searched as '{state_to_search}')")
            return None
        
        row = state_data.iloc[0]
        
        return {
            'display_text': row.get('Display_Text', ''),
            'min_staffing': float(row.get('Min_Staffing', 0)),
            'max_staffing': float(row.get('Max_Staffing', 0)) if pd.notna(row.get('Max_Staffing')) else None,
            'value_type': row.get('Value_Type', 'single'),
            'is_federal_minimum': row.get('Is_Federal_Minimum', False)
        }
    except Exception as e:
        print(f"    Error loading MACPAC standards: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_provider_info_data(provnum: str) -> pd.DataFrame:
    """Load provider info data from CSV file. Prefers provider_info_combined.csv if available (has updated case mix index)."""
    provnum = str(provnum).strip()
    provnum_zfill = provnum.zfill(6)
    
    df = pd.DataFrame()
    
    # First, try provider_info_combined.csv (preferred - has updated case mix index)
    combined_file = 'provider_info_combined.csv'
    if os.path.exists(combined_file):
        print(f"Loading from {combined_file} for facility {provnum}...")
        try:
            combined_df = pd.read_csv(combined_file, low_memory=False, dtype={'ccn': str})
            # Format CCN to ensure consistency
            if 'ccn' in combined_df.columns:
                combined_df['ccn'] = combined_df['ccn'].astype(str).str.zfill(6)
            # Filter for this facility
            facility_df = combined_df[combined_df['ccn'] == provnum_zfill].copy()
            if not facility_df.empty:
                print(f"  Found {len(facility_df)} records in combined file")
                # Convert processing_date to datetime
                if 'processing_date' in facility_df.columns:
                    facility_df['processing_date'] = pd.to_datetime(facility_df['processing_date'], errors='coerce')
                df = facility_df
        except Exception as e:
            print(f"  Error loading from combined file: {e}")
    
    # If combined file doesn't have data, fall back to facility-specific CSV
    # Use same path order as load_facility_data: deployments/ first, then project root
    if df.empty or len(df) == 0:
        possible_files = [
            f'deployments/pbj320-{provnum_zfill}/facility_{provnum_zfill}_provider_info_data.csv',
            f'pbj320-{provnum}/facility_{provnum}_provider_info_data.csv',
            f'facility_{provnum}_provider_info_data.csv',
            f'facility_{provnum_zfill}_provider_info_data.csv',
        ]
        
        for file_path in possible_files:
            if os.path.exists(file_path):
                print(f"Loading provider info data from: {file_path}")
                df = pd.read_csv(file_path, low_memory=False, dtype={'ccn': str})
                # Format CCN to ensure consistency
                if 'ccn' in df.columns:
                    df['ccn'] = df['ccn'].astype(str).str.zfill(6)
                # Convert processing_date to datetime
                if 'processing_date' in df.columns:
                    df['processing_date'] = pd.to_datetime(df['processing_date'], errors='coerce')
                break
    
    if df.empty:
        print(f"Warning: Could not find provider info data file for {provnum}")
    else:
        print(f"  Loaded {len(df)} provider info records")
    
    # Ensure we return a DataFrame, not a Series
    if isinstance(df, pd.Series):
        df = df.to_frame().T
    return pd.DataFrame(df) if not isinstance(df, pd.DataFrame) else df

def extract_red_flags_history(provider_info_df: pd.DataFrame, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Extract red flags history from provider info data."""
    if provider_info_df.empty:
        return []
    
    # Filter to date range
    if 'processing_date' in provider_info_df.columns:
        mask = (provider_info_df['processing_date'] >= start_date) & (provider_info_df['processing_date'] <= end_date)
        filtered_df = provider_info_df[mask].copy()
    else:
        filtered_df = provider_info_df.copy()
    
    if filtered_df.empty:
        return []
    
    # Sort by processing date
    filtered_df = filtered_df.sort_values('processing_date')
    
    red_flags_history = []
    
    for _, row in filtered_df.iterrows():
        red_flags = []
        
        # Check SFF status
        sff_value = str(row.get('sff_status', '')).strip() if pd.notna(row.get('sff_status')) else ''
        if sff_value and sff_value.upper() not in ['N', 'N/A', 'NAN', 'NONE', '']:
            if 'SFF' in sff_value.upper():
                sff_formatted = 'SFF Candidate' if 'CANDIDATE' in sff_value.upper() else 'SFF'
                red_flags.append(sff_formatted)
        
        # Check 1-star overall rating
        overall_rating = row.get('overall_rating')
        if pd.notna(overall_rating) and overall_rating is not None:
            try:
                rating = float(overall_rating)
                if rating == 1.0:
                    red_flags.append("1-Star Overall Rating")
            except (ValueError, TypeError):
                pass
        
        # Check 1-star staffing rating
        staffing_rating = row.get('staffing_rating')
        if pd.notna(staffing_rating) and staffing_rating is not None:
            try:
                rating = float(staffing_rating)
                if rating == 1.0:
                    red_flags.append("1-Star Staffing Rating")
            except (ValueError, TypeError):
                pass
        
        # Check Abuse icon
        abuse_value = str(row.get('abuse_icon', '')).strip() if pd.notna(row.get('abuse_icon')) else ''
        abuse_upper = abuse_value.upper()
        if abuse_upper in ['Y', 'YES', 'TRUE', '1']:
            red_flags.append("Abuse Icon")
        
        # Check Administrator Turnover (number of admins who left NH in 12 months; display as integer)
        at_val = row.get('administrator_turnover')
        if pd.notna(at_val) and str(at_val).strip():
            try:
                at_float = float(at_val)  # type: ignore[arg-type]
                if at_float > 0:
                    red_flags.append(f"Admin TO: {int(at_float)}")
            except (ValueError, TypeError):
                if str(at_val).strip().upper() in ['Y', 'YES', 'TRUE', '1']:
                    red_flags.append("Admin TO")
        
        # Check Ownership Change
        ownership_value = str(row.get('provider_changed_ownership_in_last_12_months', '')).strip() if pd.notna(row.get('provider_changed_ownership_in_last_12_months')) else ''
        ownership_upper = ownership_value.upper()
        if ownership_upper in ['Y', 'YES', 'TRUE', '1']:
            red_flags.append("Ownership Change")
        
        # Only add if there are red flags
        if red_flags:
            proc_date = row.get('processing_date')
            if pd.notna(proc_date):
                if isinstance(proc_date, str):
                    proc_date = pd.to_datetime(proc_date, errors='coerce')
                date_str = proc_date.strftime('%Y-%m-%d') if pd.notna(proc_date) else 'Unknown'
            else:
                date_str = 'Unknown'
            
            # Get quarter
            quarter = row.get('quarter', '')
            if pd.notna(quarter) and str(quarter).strip():
                quarter_str = str(quarter).strip()
            elif pd.notna(proc_date) and isinstance(proc_date, pd.Timestamp):
                year = proc_date.year
                month = proc_date.month
                if month <= 3:
                    q = 1
                elif month <= 6:
                    q = 2
                elif month <= 9:
                    q = 3
                else:
                    q = 4
                quarter_str = f"{year}Q{q}"
            else:
                quarter_str = 'Unknown'
            
            red_flags_history.append({
                'date': date_str,
                'quarter': quarter_str,
                'red_flags': red_flags,
                'overall_rating': row.get('overall_rating'),
                'staffing_rating': row.get('staffing_rating'),
                'health_inspection_rating': row.get('health_inspection_rating'),
                'ownership_type': row.get('ownership_type'),
                'affiliated_entity_name': row.get('affiliated_entity_name'),
                'chain_name': row.get('chain_name')
            })
    
    return red_flags_history

def normalize_quarter_format(quarter_str: str) -> str:
    """Normalize quarter format from 'Q2 2024' or '2024Q2' to '2024Q2'."""
    if not quarter_str or pd.isna(quarter_str):
        return ''
    quarter_str = str(quarter_str).strip()
    if not quarter_str or quarter_str == 'nan':
        return ''
    
    # If already in "2024Q2" format, return as is
    if len(quarter_str) == 6 and quarter_str[4] == 'Q' and quarter_str[0:4].isdigit() and quarter_str[5].isdigit():
        return quarter_str
    
    # If in "Q2 2024" or "Q2 2 024" format, convert to "2024Q2"
    if quarter_str.startswith('Q') and ' ' in quarter_str:
        parts = quarter_str.replace('Q', '').split()
        if len(parts) >= 2:
            quarter_num = parts[0]
            year = ''.join(parts[1:])  # Join year parts in case of "2 024"
            if quarter_num.isdigit() and year.isdigit():
                return f"{year}Q{quarter_num}"
    
    return quarter_str

def _row_value(row: pd.Series, *candidates: str):
    """Get first non-null value from row for any of the candidate column names (exact or case-insensitive)."""
    def _ok(val):
        if val is None or pd.isna(val):
            return False
        s = str(val).strip()
        return s not in ('', 'nan')
    for col in candidates:
        val = row.get(col)
        if _ok(val):
            return val
    idx_lower = {str(c).lower(): c for c in row.index}
    for col in candidates:
        key = str(col).lower()
        if key in idx_lower:
            val = row.get(idx_lower[key])
            if _ok(val):
                return val
    return None


def extract_case_mix_data(provider_info_df: pd.DataFrame, start_date: datetime, end_date: datetime, quarters_in_range: Optional[List[str]] = None) -> List[Dict]:
    """Extract case-mix adjusted HPRD data from provider info.
    Quarter labels come from the provider file 'quarter' column (CMS/lagged naming may differ from calendar quarter)."""
    if provider_info_df.empty:
        print("  No provider info data available for case-mix extraction")
        return []
    
    # Normalize quarters_in_range to standard format
    normalized_quarters_in_range = []
    if quarters_in_range and len(quarters_in_range) > 0:
        print(f"  Normalizing quarters: {quarters_in_range}")
        for q in quarters_in_range:
            normalized = normalize_quarter_format(q)
            if normalized:
                normalized_quarters_in_range.append(normalized)
        print(f"  Normalized quarters: {normalized_quarters_in_range}")
    
    # Filter by quarters in range if provided, otherwise filter by date range
    if normalized_quarters_in_range and len(normalized_quarters_in_range) > 0:
        # Filter by quarter field instead of date (more reliable)
        # Convert quarter values to strings and normalize them
        provider_info_df_copy = provider_info_df.copy()
        provider_info_df_copy['quarter'] = provider_info_df_copy['quarter'].astype(str)
        # Normalize all quarters in the dataframe
        provider_info_df_copy['quarter_normalized'] = provider_info_df_copy['quarter'].apply(normalize_quarter_format)
        print(f"  Provider info quarters (normalized): {provider_info_df_copy['quarter_normalized'].dropna().unique()[:10]}")
        filtered_df = provider_info_df_copy[provider_info_df_copy['quarter_normalized'].isin(normalized_quarters_in_range)].copy()
        print(f"  Found {len(filtered_df)} records matching quarters {normalized_quarters_in_range}")
    elif 'processing_date' in provider_info_df.columns:
        # Include all provider info rows whose quarter overlaps the report date range,
        # so we don't miss case-mix/CMI data when the user picks a narrow date window.
        from datetime import date as date_type
        start_d, end_d = start_date.date() if hasattr(start_date, 'date') else start_date, end_date.date() if hasattr(end_date, 'date') else end_date
        overlapping_quarters = set()
        y, m = start_d.year, start_d.month
        end_y, end_m = end_d.year, end_d.month
        while (y, m) <= (end_y, end_m):
            q = (m - 1) // 3 + 1
            overlapping_quarters.add(f"{y}Q{q}")
            if m == 12:
                y, m = y + 1, 1
            else:
                m += 1
        # Also derive quarter from each row (column or processing_date)
        provider_info_df_copy = provider_info_df.copy()
        provider_info_df_copy['processing_date'] = pd.to_datetime(provider_info_df_copy['processing_date'], errors='coerce')
        def row_quarter(r):
            qcol = r.get('quarter', '')
            if pd.notna(qcol) and str(qcol).strip() and str(qcol).strip() != 'nan':
                return normalize_quarter_format(str(qcol).strip())
            pd_val = r.get('processing_date')
            if pd.notna(pd_val) and hasattr(pd_val, 'year'):
                return f"{pd_val.year}Q{(pd_val.month - 1) // 3 + 1}"
            return None
        provider_info_df_copy['_quarter_overlap'] = provider_info_df_copy.apply(
            lambda r: row_quarter(r) in overlapping_quarters if row_quarter(r) else False, axis=1
        )
        in_range = (provider_info_df_copy['processing_date'] >= start_date) & (provider_info_df_copy['processing_date'] <= end_date)
        mask = in_range | provider_info_df_copy['_quarter_overlap']
        filtered_df = provider_info_df_copy.loc[mask].copy()
        filtered_df = filtered_df.drop(columns=['_quarter_overlap'], errors='ignore')
        print(f"  Filtered by date range and overlapping quarters {sorted(overlapping_quarters)}: {len(filtered_df)} records")
    else:
        filtered_df = provider_info_df.copy()
        print(f"  Using all provider info data: {len(filtered_df)} records")
    
    if filtered_df.empty:
        print("  No provider info records found after filtering")
        return []
    
    # Sort by processing date (NaT last)
    if 'processing_date' in filtered_df.columns:
        filtered_df = filtered_df.sort_values('processing_date', na_position='last')
    
    case_mix_data = []
    
    for _, row in filtered_df.iterrows():
        proc_date = row.get('processing_date')
        if pd.notna(proc_date):
            if isinstance(proc_date, str):
                proc_date = pd.to_datetime(proc_date, errors='coerce')
            date_str = proc_date.strftime('%Y-%m-%d') if pd.notna(proc_date) else 'Unknown'
        else:
            date_str = 'Unknown'
        
        # Get quarter and normalize format
        quarter = row.get('quarter', '')
        if pd.notna(quarter) and str(quarter).strip() and str(quarter).strip() != 'nan':
            quarter_str = normalize_quarter_format(str(quarter).strip())
        elif pd.notna(proc_date) and isinstance(proc_date, pd.Timestamp):
            year = proc_date.year
            month = proc_date.month
            if month <= 3:
                q = 1
            elif month <= 6:
                q = 2
            elif month <= 9:
                q = 3
            else:
                q = 4
            quarter_str = f"{year}Q{q}"
        else:
            quarter_str = 'Unknown'
        
        # Note: CMI data is only available from 2024 onwards in provider info files
        # Earlier quarters will have CMI as None, which is correct
        # We process all quarters but CMI will be None for pre-2024 quarters
        
        # Extract case-mix values (case-insensitive column lookup for different CSV sources)
        case_mix_total = _row_value(row, 'case_mix_total_nurse_hrs_per_resident_per_day')
        case_mix_rn = _row_value(row, 'case_mix_rn_hrs_per_resident_per_day')
        case_mix_lpn = _row_value(row, 'case_mix_lpn_hrs_per_resident_per_day')
        case_mix_na = _row_value(row, 'case_mix_na_hrs_per_resident_per_day')
        
        # Calculate case-mix direct care (RN + LPN + NA, excluding admin/DON)
        case_mix_direct = None
        if pd.notna(case_mix_rn) and case_mix_rn is not None and pd.notna(case_mix_lpn) and case_mix_lpn is not None and pd.notna(case_mix_na) and case_mix_na is not None:
            case_mix_direct = float(case_mix_rn) + float(case_mix_lpn) + float(case_mix_na)
        
        # Extract reported values for comparison
        reported_total = _row_value(row, 'reported_total_nurse_hrs_per_resident_per_day')
        reported_rn = _row_value(row, 'reported_rn_hrs_per_resident_per_day')
        reported_lpn = _row_value(row, 'reported_lpn_hrs_per_resident_per_day')
        reported_na = _row_value(row, 'reported_na_hrs_per_resident_per_day')
        
        # Calculate reported direct care (RN + LPN + NA, excluding admin/DON)
        reported_direct = None
        if pd.notna(reported_rn) and reported_rn is not None and pd.notna(reported_lpn) and reported_lpn is not None and pd.notna(reported_na) and reported_na is not None:
            reported_direct = float(reported_rn) + float(reported_lpn) + float(reported_na)
        
        # Only add if we have case-mix data.
        # Store raw values here; round only when formatting for display. This keeps % and average
        # calculations consistent with the dashboard and avoids rounding drift (e.g. 89.0% vs 89.1%).
        if pd.notna(case_mix_total) or pd.notna(case_mix_rn):
            entry = {
                'date': date_str,
                'quarter': quarter_str,
                'case_mix_total': None,
                'case_mix_rn': None,
                'case_mix_lpn': None,
                'case_mix_na': None,
                'reported_total': None,
                'reported_rn': None,
                'reported_lpn': None,
                'reported_na': None,
                'avg_residents_per_day': None,
                'certified_beds': None,
                'occupancy_pct': None
            }
            
            # Store raw case_mix_total for % calculations (match dashboard); round only for display
            if pd.notna(case_mix_total) and case_mix_total is not None:
                entry['case_mix_total'] = float(case_mix_total)
            # Store raw case_mix values for % calculations (match dashboard); round only for display
            if pd.notna(case_mix_rn) and case_mix_rn is not None:
                entry['case_mix_rn'] = float(case_mix_rn)
            if pd.notna(case_mix_lpn) and case_mix_lpn is not None:
                entry['case_mix_lpn'] = float(case_mix_lpn)
            if pd.notna(case_mix_na) and case_mix_na is not None:
                entry['case_mix_na'] = float(case_mix_na)
            if case_mix_direct is not None:
                entry['case_mix_direct'] = float(case_mix_direct)
            # Store raw reported values for % and average calculations; round only for display
            if pd.notna(reported_total) and reported_total is not None:
                entry['reported_total'] = float(reported_total)
            if pd.notna(reported_rn) and reported_rn is not None:
                entry['reported_rn'] = float(reported_rn)
            if pd.notna(reported_lpn) and reported_lpn is not None:
                entry['reported_lpn'] = float(reported_lpn)
            if pd.notna(reported_na) and reported_na is not None:
                entry['reported_na'] = float(reported_na)
            if reported_direct is not None:
                entry['reported_direct'] = float(reported_direct)
            
            avg_residents = row.get('avg_residents_per_day')
            if pd.notna(avg_residents) and avg_residents is not None:
                entry['avg_residents_per_day'] = float(avg_residents)
            # Occupancy: census / certified beds (same quarter matching as PBJ)
            certified_beds = _row_value(row, 'certified_beds')
            if pd.notna(certified_beds) and certified_beds is not None:
                try:
                    b = float(certified_beds)
                    entry['certified_beds'] = int(b) if b == int(b) else int(round(b))
                    if b > 0 and entry.get('avg_residents_per_day') is not None:
                        c = float(entry['avg_residents_per_day'])
                        if c >= 0:
                            entry['occupancy_pct'] = round_half_up(c / b * 100, 2)
                        else:
                            entry['occupancy_pct'] = None
                    else:
                        entry['occupancy_pct'] = None
                except (TypeError, ValueError):
                    entry['certified_beds'] = None
                    entry['occupancy_pct'] = None
            else:
                entry['certified_beds'] = None
                entry['occupancy_pct'] = None
            
            # Extract CMI (Case Mix Index) - try multiple column name variations (case-insensitive)
            cmi = None
            cmi_columns = ['nursing_case_mix_index', 'nursing_case_mix_index_ratio', 'case_mix_index', 'CMI', 'Case Mix Index', 'case_mix', 'Case-Mix Index', 'Case Mix Index (CMI)']
            for col in cmi_columns:
                cmi_value = _row_value(row, col)
                if cmi_value is not None:
                    try:
                        cmi = float(cmi_value)
                        break
                    except (ValueError, TypeError):
                        continue
            if cmi is not None:
                entry['cmi'] = round_half_up(cmi, 3)
                entry['cmi_raw'] = float(cmi)  # full precision for Harrington formula (no pre-round)
            else:
                entry['cmi'] = None
                entry['cmi_raw'] = None
            
            case_mix_data.append(entry)
    
    print(f"  Extracted {len(case_mix_data)} case-mix entries")
    return case_mix_data

def get_state_averages(state: str, quarter: str) -> Optional[Dict]:
    """Get state-level averages from state_lite_metrics.csv and state_quarterly_metrics.csv."""
    batch = get_state_averages_batch(state, [quarter])
    return batch.get(quarter)


def get_state_averages_batch(state: str, quarters: List[str]) -> Dict[str, Dict]:
    """Load state metrics CSVs once and return averages for all requested quarters. Use instead of calling get_state_averages in a loop."""
    result: Dict[str, Dict] = {}
    if not quarters:
        return result
    try:
        quarterly_df = None
        if os.path.exists('state_quarterly_metrics.csv'):
            try:
                q_full = pd.read_csv('state_quarterly_metrics.csv', low_memory=False)
                quarterly_df = q_full[(q_full['STATE'] == state) & (q_full['CY_Qtr'].isin(quarters))]
            except Exception:
                quarterly_df = None
        state_metrics_path = None
        for path in ['state_lite_metrics.csv', 'pbj_lite/state_lite_metrics.csv']:
            if os.path.exists(path):
                state_metrics_path = path
                break
        if not state_metrics_path:
            return result
        df = pd.read_csv(state_metrics_path, low_memory=False)
        if 'CY_Qtr' in df.columns and 'CY_QTR' not in df.columns:
            df['CY_QTR'] = df['CY_Qtr']
        df = df[(df['STATE'] == state) & (df['CY_QTR'].isin(quarters))]
        for quarter in quarters:
            state_data = df[df['CY_QTR'] == quarter]
            q_quarter = quarterly_df[(quarterly_df['CY_Qtr'] == quarter)] if quarterly_df is not None and not quarterly_df.empty else None
            q_row = q_quarter.iloc[0] if q_quarter is not None and len(q_quarter) > 0 else None
            direct_care_hprd = None if q_row is None else q_row.get('Nurse_Care_HPRD')
            direct_care_rn_hprd = None if q_row is None else q_row.get('RN_Care_HPRD')
            if state_data.empty:
                continue
            row = state_data.iloc[0]
            rn_hprd = row.get('Total_RN_HPRD') if 'Total_RN_HPRD' in row else (row.get('RN_HPRD') if 'RN_HPRD' in row else row.get('Direct_Care_RN_HPRD'))
            if pd.isna(rn_hprd) or rn_hprd == 0:
                if 'Total_RN_Hours' in row and 'total_resident_days' in row and row['total_resident_days'] > 0:
                    rn_hprd = row['Total_RN_Hours'] / row['total_resident_days']
            rn_hprd = rn_hprd if rn_hprd is not None and not pd.isna(rn_hprd) else 0
            result[quarter] = {
                'total_hprd': round_half_up(row.get('Total_Nurse_HPRD', 0), 2),
                'rn_hprd': round_half_up(float(rn_hprd), 2),
                'direct_care_rn_hprd': round_half_up(direct_care_rn_hprd, 2) if direct_care_rn_hprd is not None and not pd.isna(direct_care_rn_hprd) else round_half_up(row.get('Direct_Care_RN_HPRD', 0), 2),
                'direct_care_hprd': round_half_up(direct_care_hprd, 2) if direct_care_hprd is not None and not pd.isna(direct_care_hprd) else None
            }
        return result
    except Exception as e:
        print(f"    Error loading state metrics batch: {e}")
        return result


def generate_daily_staffing_table(daily_data: List[Dict], state_minimum: float = 0.0, include_total_staffing: bool = True, direct_first: bool = False) -> str:
    """Generate concise, styled HTML table for daily staffing on key dates."""
    if not daily_data:
        return "<p><em>No daily staffing data available for key dates.</em></p>"
    
    rows = []
    for day in daily_data:
        date_str = day['date'].strftime('%b %d, %Y')
        day_name = day['day_of_week'][:3]  # Abbreviated day name
        
        # Check if direct care HPRD is below state minimum
        direct_care_hprd = day.get('direct_care_hprd', 0)
        direct_care_hprd_below = direct_care_hprd < state_minimum if state_minimum > 0 else False
        direct_care_hprd_class = 'class="below-state-min"' if direct_care_hprd_below else ''
        direct_care_compliance = '⚠️' if direct_care_hprd_below else ''
        
        # Check if total HPRD is below state minimum (for appendix only)
        total_hprd = day.get('total_hprd', 0)
        total_hprd_below = total_hprd < state_minimum if state_minimum > 0 else False
        total_hprd_class = 'class="below-state-min"' if total_hprd_below else ''
        total_compliance = '⚠️' if total_hprd_below else ''
        
        # Build compliance indicator for date cell (only show if below minimum)
        compliance_indicator = ''
        if state_minimum > 0 and direct_care_hprd_below:
            compliance_indicator = '<br><span style="font-size: 8pt; color: #e74c3c; font-weight: bold;">⚠ Below State Min</span>'
        
        # Build row cells conditionally
        row_cells = [
            f'<td><strong>{date_str}</strong><br><span style="font-size: 8pt; color: #666;">{day_name}</span>{compliance_indicator}</td>',
            f"<td>{day['census']:.0f}</td>"
        ]
        
        if include_total_staffing:
            total_hprd_display = f'<td {total_hprd_class}><strong>{day["total_hprd"]:.2f}</strong>'
            if state_minimum > 0:
                total_hprd_display += f'<br><span style="font-size: 7pt; color: {"#e74c3c" if total_hprd_below else "#27ae60"};">{total_compliance} {state_minimum:.2f}</span>'
            total_hprd_display += '</td>'
            row_cells.append(total_hprd_display)
        
        direct_care_display = f'<td {direct_care_hprd_class}><strong>{direct_care_hprd:.2f}</strong>'
        if state_minimum > 0:
            direct_care_display += f'<br><span style="font-size: 7pt; color: {"#e74c3c" if direct_care_hprd_below else "#27ae60"};">{direct_care_compliance} {state_minimum:.2f}</span>'
        direct_care_display += '</td>'
        row_cells.append(direct_care_display)
        
        if include_total_staffing:
            row_cells.append(f"<td>{day['rn_hprd']:.2f}</td>")  # Total RN HPRD
        
        row_cells.append(f"<td><strong>{day['direct_care_rn_hprd']:.2f}</strong></td>")  # RN HPRD (excl. Admin/DON)
        row_cells.append(f"<td>{day['lpn_hprd']:.2f}</td>")
        row_cells.append(f"<td>{day['cna_hprd']:.2f}</td>")
        
        # Hours columns - match the HPRD columns shown
        if include_total_staffing:
            row_cells.append(f"<td>{day['total_rn_hours']:.2f}</td>")  # Total RN Hours (matches Total RN HPRD)
        row_cells.append(f"<td>{day.get('hrs_rn', day.get('direct_care_rn_hours', 0)):.2f}</td>")  # RN Hours (matches RN HPRD)
        
        row_cells.append(f"<td>{day.get('direct_lpn_hours', day.get('hrs_lpn', 0)):.2f}</td>")  # Direct LPN Hours (excludes admin)
        row_cells.append(f"<td>{day['total_nurse_aide_hours']:.2f}</td>")
        row_cells.append(f"<td>{day['contract_pct']:.1f}%</td>")
        
        # Alternate row colors: white and light gray
        row_bg = '#ffffff' if len(rows) % 2 == 0 else '#f5f5f5'
        rows.append(f"""
        <tr class="key-date-row" style="background-color: {row_bg};">
            {''.join(row_cells)}
        </tr>
        """)
    
    # Build header conditionally (order by direct_first)
    header_cells = ["<th>Date</th>", "<th>Census</th>"]
    if direct_first:
        header_cells.append("<th>Direct Care<br>HPRD</th>")
        if include_total_staffing:
            header_cells.append("<th>Total<br>HPRD</th>")
        header_cells.append("<th>RN<br>HPRD</th>")
        if include_total_staffing:
            header_cells.append("<th>Total RN<br>HPRD</th>")
    else:
        if include_total_staffing:
            header_cells.append("<th>Total<br>HPRD</th>")
        header_cells.append("<th>Direct Care<br>HPRD</th>")
        if include_total_staffing:
            header_cells.append("<th>Total RN<br>HPRD</th>")
        header_cells.append("<th>RN<br>HPRD</th>")
    header_cells.extend([
        "<th>LPN<br>HPRD</th>",
        "<th>Nurse Aide<br>HPRD</th>"
    ])
    if direct_first:
        header_cells.append("<th>RN<br>Hours</th>")
        if include_total_staffing:
            header_cells.append("<th>Total RN<br>Hours</th>")
    else:
        if include_total_staffing:
            header_cells.append("<th>Total RN<br>Hours</th>")
        header_cells.append("<th>RN<br>Hours</th>")
    header_cells.extend([
        "<th>LPN<br>Hours</th>",
        "<th>Nurse Aide<br>Hours</th>",
        "<th>Contract<br>%</th>"
    ])
    
    return f"""
    <table>
        <thead>
            <tr style="background-color: #3498db; color: white;">
                {''.join(header_cells)}
            </tr>
        </thead>
        <tbody>
            {''.join(rows)}
        </tbody>
    </table>
    <p style="margin-top: 8px; font-size: 8pt; font-style: italic; color: #666;">Total RN = RN + RN Admin + RN DON. RN HPRD / RN Hours = direct care only (excl. Admin/DON). LPN = direct LPN only (excl. LPN Admin). Nurse Aide = CNA + Med Aide + NA Trn. Contract % = contract hours as % of total nursing hours (all positions).</p>
    """

def _render_nj_state_section(macpac_standards: Optional[Dict], cfg: Dict) -> str:
    """Render NJ-specific staffing section (ratio-based rules from N.J.A.C. 8:39-25.1). Data-driven via cfg."""
    law_url = cfg.get('law_url') or STATE_LEGISLATION_URLS.get('NJ', MACPAC_SOURCE_URL)
    law_label = cfg.get('law_label') or STATE_LEGISLATION_LABELS.get('NJ', 'N.J.A.C. 8:39-25.1')
    section_title = cfg.get('section_title') or 'New Jersey Administrative Code (N.J.A.C.) 8:39-25.1 - Minimum Staffing Requirements'
    intro_text = ""
    if macpac_standards and macpac_standards.get('min_staffing'):
        intro_text = f'<p style="margin-bottom: 15px; font-size: 10pt;"><strong>Estimated New Jersey Staffing Requirements:</strong> {macpac_standards["min_staffing"]:.2f} HPRD (Source: <a href="{MACPAC_SOURCE_URL}" target="_blank" rel="noopener">MACPAC</a>)</p>'
    return f"""
    <div style="page-break-inside: avoid; break-inside: avoid;">
    <h2>New Jersey State Staffing Requirements</h2>
    {intro_text}
    <div class="macpac-note" style="margin-top: 15px; margin-bottom: 15px;">
        <p><strong>{section_title}</strong></p>
        
        <table style="margin-top: 10px; font-size: 10pt;">
            <thead>
                <tr>
                    <th>Requirement</th>
                    <th>HPRD</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="font-weight: 600;">Total Direct Care Staff</td>
                    <td style="text-align: center; font-weight: 600;">2.50 HPRD</td>
                    <td>The facility shall provide nursing services by registered professional nurses, LPNs, and nurse aides (the hours of the director of nursing are not included in this computation, except for the direct care hours of the DON in facilities where the DON provides more than the minimum hours required at N.J.A.C. 8:39-25.1(a)) on the basis of: Total number of residents multiplied by 2.5 hours/day; plus Total number of residents receiving each service listed below, multiplied by the corresponding number of hours per day: Wound care 0.75 hour/day; Nasogastric tube feedings and/or gastrostomy 1.00 hour/day; Oxygen therapy 0.75 hour/day; Tracheostomy 1.25 hours/day; Intravenous therapy 1.50 hours/day; Use of respirator 1.25 hours/day; Head trauma stimulation/advanced 1.50 hours/day; neuromuscular/orthopedic care 1.50 hours per day.</td>
                </tr>
                <tr>
                    <td style="font-weight: 600;">Day Shift - CNA</td>
                    <td style="text-align: center;">1.04 HPRD (of total)</td>
                    <td>One CNA to every eight residents for the day shift.</td>
                </tr>
                <tr>
                    <td style="font-weight: 600;">Evening Shift - Direct Care</td>
                    <td style="text-align: center;">1:10 ratio</td>
                    <td>One direct care staff member to every 10 residents for the evening shift, provided that no fewer than half of all staff members shall be CNAs, and each staff member shall be signed in to work as a CNA and shall perform certified nurse aide duties.</td>
                </tr>
                <tr>
                    <td style="font-weight: 600;">Night Shift - Direct Care</td>
                    <td style="text-align: center;">1:14 ratio</td>
                    <td>One direct care staff member to every 14 residents for the night shift, provided that each direct care staff member shall sign in to work as a CNAs and perform CNA duties.</td>
                </tr>
                <tr>
                    <td style="font-weight: 600;">Director of Nursing</td>
                    <td style="text-align: center;">0.06 HPRD</td>
                    <td>There shall be a full-time DON or NHA who is a registered professional nurse licensed in the State of New Jersey, who has at least two years of supervisory experience in providing care to long-term care residents, and who supervises all nursing personnel.</td>
                </tr>
                <tr>
                    <td style="font-weight: 600;">Ventilator Dependent Patients</td>
                    <td style="text-align: center;">Not found</td>
                    <td>For facilities providing care to ventilator dependent patients/residents, the facility shall provide, on a twenty-four hour basis, an adequate number of RNs and respiratory therapist(s) who are trained and competent to take care of a ventilator dependent patient. Such training and the competency evaluation must be conducted by a respiratory therapist or a pulmonologist and records of said training maintained by the facility.</td>
                </tr>
            </tbody>
        </table>
        
        <p style="margin-top: 10px; font-size: 9pt;"><strong>Definition:</strong> "Long-term care facility direct care staff member" means any health care professional licensed or certified pursuant to Title 26 or Title 45 of the Revised Statutes who is employed by a long term care facility and who provides personal care, assistance, or treatment services directly to residents of the facility in the course of the professional's regular duties.</p>
        
        <p style="margin-top: 10px; font-size: 9pt;"><strong>Non-Licensed Staff:</strong> The non-licensed staff shall be added to the total licensed staff, to complete the required staffing requirements.</p>
        
        <p style="margin-top: 10px; font-size: 9pt; font-style: italic;">Source: <a href="{law_url}" target="_blank" rel="noopener">{law_label}</a> | <a href="{MACPAC_SOURCE_URL}" target="_blank" rel="noopener">MACPAC</a></p>
    </div>
    </div>
    """


def generate_state_staffing_section(state: str, macpac_standards: Optional[Dict] = None, period_metrics: Optional[Dict] = None) -> str:
    """Generate state staffing requirements section from MACPAC data + state_staffing_display.json. No state-specific branches."""
    if not macpac_standards or not macpac_standards.get('min_staffing') or macpac_standards.get('min_staffing', 0) < 2.00:
        return ""
    state_full_name = STATE_NAMES.get(state.upper(), state)
    cfg = _load_state_staffing_display().get(state.upper(), {})
    if cfg.get('section_template') == 'nj':
        return _render_nj_state_section(macpac_standards, cfg)
    min_staffing = macpac_standards['min_staffing']
    display_text = macpac_standards.get('display_text', f'{state_full_name} State Minimum Staffing Requirement')
    law_url = cfg.get('law_url') or STATE_LEGISLATION_URLS.get(state.upper())
    law_label = cfg.get('law_label') or STATE_LEGISLATION_LABELS.get(state.upper(), "State code")
    enforcement_agency = cfg.get('enforcement_agency') or "the state regulatory agency"
    is_estimated = cfg.get('is_estimated', False)
    title = f"Estimated {state_full_name} Staffing Requirements" if is_estimated else f"{state_full_name} State Staffing Requirements"
    if law_url:
        intro_text = f'<p style="margin-bottom: 15px; font-size: 10pt;"><strong>{title}:</strong> {min_staffing:.2f} HPRD (Source: <a href="{law_url}" target="_blank" rel="noopener">{law_label}</a> | <a href="{MACPAC_SOURCE_URL}" target="_blank" rel="noopener">MACPAC</a>)</p>'
    else:
        intro_text = f'<p style="margin-bottom: 15px; font-size: 10pt;"><strong>{title}:</strong> {min_staffing:.2f} HPRD (Source: <a href="{MACPAC_SOURCE_URL}" target="_blank" rel="noopener">MACPAC</a>)</p>'
    days_under_info = ""
    if period_metrics and min_staffing > 0:
        days_under_total = period_metrics.get('days_under_minimum_total', 0)
        days_under_direct = period_metrics.get('days_under_minimum_direct', 0)
        total_days = period_metrics.get('total_days', 0)
        pct_under_total = period_metrics.get('percentage_under_total', 0.0)
        pct_under_direct = period_metrics.get('percentage_under_direct', 0.0)
        if total_days > 0:
            days_under_info = f"""
        <div style="background-color: #ffffff; border: 2px solid #e9ecef; border-left: 4px solid #dc3545; padding: 15px; margin: 15px 0; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #2c3e50; font-size: 12pt; font-weight: 600; border-bottom: 2px solid #e9ecef; padding-bottom: 8px;">Compliance Analysis</h3>
            <table style="font-size: 10pt; width: 100%; margin-top: 8px; border-collapse: collapse;">
                <tr style="background-color: #f8f9fa;">
                    <td style="font-weight: 600; padding: 8px; color: #2c3e50; border-bottom: 1px solid #dee2e6;">Total Days in Period:</td>
                    <td style="padding: 8px; color: #2c3e50; border-bottom: 1px solid #dee2e6;"><strong>{total_days:,} days</strong></td>
                </tr>
                <tr style="background-color: {'#fff5f5' if days_under_total > 0 else '#f0f9ff'};">
                    <td style="font-weight: 600; padding: 8px; color: #2c3e50; border-bottom: 1px solid #dee2e6;">Days Below {state_full_name} Minimum - Total HPRD:</td>
                    <td style="padding: 8px; border-bottom: 1px solid #dee2e6; color: {'#dc3545' if days_under_total > 0 else '#28a745'};"><strong>{days_under_total:,} days ({pct_under_total:.1f}%)</strong></td>
                </tr>
                <tr style="background-color: {'#fff5f5' if days_under_direct > 0 else '#f0f9ff'};">
                    <td style="font-weight: 600; padding: 8px; color: #2c3e50; border-bottom: 1px solid #dee2e6;">Days Below {state_full_name} Minimum - Direct Care HPRD:</td>
                    <td style="padding: 8px; border-bottom: 1px solid #dee2e6; color: {'#dc3545' if days_under_direct > 0 else '#28a745'};"><strong>{days_under_direct:,} days ({pct_under_direct:.1f}%)</strong></td>
                </tr>
            </table>
        </div>"""
    source_line = f'<a href="{law_url}" target="_blank" rel="noopener">{law_label}</a> | ' if law_url else ''
    source_line += f'<a href="{MACPAC_SOURCE_URL}" target="_blank" rel="noopener">MACPAC</a>'
    return f"""
    <div style="page-break-inside: avoid; break-inside: avoid;">
    <h2>{title}</h2>
    {intro_text}
    <div class="macpac-note" style="margin-top: 15px; margin-bottom: 15px;">
        <p><strong>{display_text}</strong></p>
        
        <p style="font-size: 10pt; line-height: 1.6; margin-top: 10px;">{state_full_name} requires nursing homes to maintain minimum staffing levels to ensure adequate care for residents. The state has established a minimum of {min_staffing:.2f} hours per resident day (HPRD) of total nursing staff.</p>
        
        {days_under_info}
        
        <table style="margin-top: 15px; font-size: 10pt; width: 100%;">
            <thead>
                <tr style="background-color: #f8f9fa;">
                    <th style="padding: 8px; text-align: left; border-bottom: 2px solid #dee2e6;">Requirement</th>
                    <th style="padding: 8px; text-align: center; border-bottom: 2px solid #dee2e6;">HPRD</th>
                    <th style="padding: 8px; text-align: left; border-bottom: 2px solid #dee2e6;">Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 8px; font-weight: 600; border-bottom: 1px solid #dee2e6;">Minimum Total Staffing</td>
                    <td style="padding: 8px; text-align: center; font-weight: 600; border-bottom: 1px solid #dee2e6;">{min_staffing:.2f} HPRD</td>
                    <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">{state_full_name} requires nursing homes to maintain a minimum of {min_staffing:.2f} hours per resident day of total nursing staff (RN, LPN, and nurse aides—CNA, Med Aide, NA Trn—combined). This includes both direct care staff and administrative staff.</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: 600; border-bottom: 1px solid #dee2e6;">RN Staffing</td>
                    <td style="padding: 8px; text-align: center; border-bottom: 1px solid #dee2e6;">Minimum Required</td>
                    <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">Facilities must maintain adequate RN staffing to meet resident care needs. RN hours include both direct care RNs and RN administrators/DON.</td>
                </tr>
                <tr>
                    <td style="padding: 8px; font-weight: 600; border-bottom: 1px solid #dee2e6;">CNA Staffing</td>
                    <td style="padding: 8px; text-align: center; border-bottom: 1px solid #dee2e6;">Minimum Required</td>
                    <td style="padding: 8px; border-bottom: 1px solid #dee2e6;">Certified Nurse Aide staffing must be sufficient to provide direct care services to all residents.</td>
                </tr>
            </tbody>
        </table>
        
        <p style="margin-top: 15px; font-size: 9pt; line-height: 1.5;"><strong>Enforcement:</strong> Facilities that fail to meet {state_full_name} minimum staffing requirements may be subject to penalties, fines, and enforcement actions by {enforcement_agency}.</p>
        
        <p style="margin-top: 10px; font-size: 9pt; font-style: italic;">Source: {source_line}</p>
    </div>
    </div>
    """


def generate_red_flags_section(red_flags_history: Optional[List[Dict]]) -> str:
    """Generate HTML section for red flags history with quarter-based ranges and source disclaimer."""
    if not red_flags_history:
        return """
    <h2>Historical Red Flags</h2>
    <p><em>No red flags identified in the historical data for this period.</em></p>
    """
    
    # Group red flags by type; collect (entry_date, quarter_str) for range display
    flag_ranges = {}
    
    for entry in red_flags_history:
        date_str = entry.get('date', '')
        quarter_str = (entry.get('quarter') or '').strip() if pd.notna(entry.get('quarter')) else ''
        try:
            entry_date = pd.to_datetime(date_str)
        except Exception:
            continue
        
        red_flags = entry.get('red_flags', [])
        
        for flag in red_flags:
            flag_type = None
            flag_upper = str(flag).upper()
            if 'SFF' in flag_upper or 'SPECIAL FOCUS' in flag_upper:
                flag_type = 'SFF'
            elif '1-STAR OVERALL' in flag_upper or ('1' in flag_upper and 'OVERALL' in flag_upper and 'RATING' in flag_upper):
                flag_type = '1-Star Overall Rating'
            elif '1-STAR STAFFING' in flag_upper or ('1' in flag_upper and 'STAFFING' in flag_upper and 'RATING' in flag_upper):
                flag_type = '1-Star Staffing Rating'
            elif 'ABUSE' in flag_upper or 'ABUSE ICON' in flag_upper:
                flag_type = 'Abuse Icon'
            elif 'OWNERSHIP' in flag_upper or 'OWNERSHIP CHANGE' in flag_upper:
                flag_type = 'Ownership Change'
            elif 'ADMIN TO' in flag_upper:
                flag_type = 'Admin TO'
            
            if flag_type:
                if flag_type not in flag_ranges:
                    flag_ranges[flag_type] = []
                flag_ranges[flag_type].append((entry_date, quarter_str))
    
    summary_sentences = []
    for flag_type, date_quarters in flag_ranges.items():
        if not date_quarters:
            continue
        dates_sorted = sorted(date_quarters, key=lambda x: x[0])
        start_date, start_q = dates_sorted[0]
        end_date, end_q = dates_sorted[-1]
        # Prefer quarter display when available; normalize "Q2 2024" -> "2024Q2" so format_quarter_display works
        if start_q and end_q and start_q != 'Unknown' and end_q != 'Unknown':
            start_norm = normalize_quarter_format(str(start_q))
            end_norm = normalize_quarter_format(str(end_q))
            start_display = format_quarter_display(start_norm) if start_norm else start_date.strftime('%B %Y')
            end_display = format_quarter_display(end_norm) if end_norm else end_date.strftime('%B %Y')
        else:
            start_display = start_date.strftime('%B %Y')
            end_display = end_date.strftime('%B %Y')
        if start_display == end_display:
            summary_sentences.append(f"This facility had <strong>{flag_type}</strong> in provider info for <strong>{start_display}</strong>.<a href=\"#red-flags-matching-logic\">*</a>")
        else:
            summary_sentences.append(f"This facility had <strong>{flag_type}</strong> in provider info approximately <strong>{start_display} – {end_display}</strong>.<a href=\"#red-flags-matching-logic\">*</a>")
    
    if not summary_sentences:
        return """
    <h2>Historical Red Flags</h2>
    <p><em>No red flags identified in the historical data for this period.</em></p>
    """
    
    summary_html = '<ul>' + ''.join([f'<li>{s}</li>' for s in summary_sentences]) + '</ul>'
    
    return f"""
    <h2>Historical Red Flags</h2>
    <p>Red flags below are from <strong>CMS Provider Information</strong> (periodic snapshots), not exact incident dates. They include: SFF status, 1-star ratings (overall or staffing), abuse icon, Admin TO (admins who left NH in 12 months), and ownership changes.</p>
    {summary_html}
    <p id="red-flags-matching-logic" style="font-size: 9pt; color: #555; margin-top: 12px;"><strong>* Red flags</strong>: included when <code>processing_date</code> falls within the report period; quarter shown is from the file (estimate). <a href="#provider-info-matching">Provider information matching</a> (Appendix).</p>
    """

def calculate_quarterly_direct_care_hprd(df: pd.DataFrame, quarter: str) -> Optional[float]:
    """Calculate direct care HPRD from PBJ data for a quarter (excluding admin/DON). Returns value rounded to 2 decimals."""
    raw = _quarterly_direct_care_hprd_raw(df, quarter)
    return round_half_up(raw, 2) if raw is not None else None


def _quarterly_direct_care_hprd_raw(df: pd.DataFrame, quarter: str) -> Optional[float]:
    """Raw direct care HPRD for a quarter (no rounding). Use for % case-mix so dashboard and attorney report match."""
    quarter_data = df[df['CY_Qtr'] == quarter].copy()
    if quarter_data.empty:
        return None
    total_resident_days = quarter_data['MDScensus'].sum()
    if total_resident_days == 0:
        return None
    direct_care_hours = (
        quarter_data['Hrs_RN'].fillna(0) +
        quarter_data['Hrs_LPN'].fillna(0) +
        quarter_data['Hrs_CNA'].fillna(0) +
        quarter_data['Hrs_NAtrn'].fillna(0) +
        quarter_data['Hrs_MedAide'].fillna(0)
    ).sum()
    return float(direct_care_hours / total_resident_days)


def _quarterly_direct_rn_hprd_raw(df: pd.DataFrame, quarter: str) -> Optional[float]:
    """Raw direct RN HPRD for a quarter (Hrs_RN only, no rounding). Use for RN % case-mix so dashboard and attorney report match."""
    quarter_data = df[df['CY_Qtr'] == quarter].copy()
    if quarter_data.empty:
        return None
    total_resident_days = quarter_data['MDScensus'].sum()
    if total_resident_days == 0:
        return None
    rn_hours = quarter_data['Hrs_RN'].fillna(0).sum()
    return float(rn_hours / total_resident_days)


def _quarterly_direct_cna_hprd_raw(df: pd.DataFrame, quarter: str) -> Optional[float]:
    """Raw nurse aide HPRD for a quarter (CNA + MedAide + NAtrn, no rounding). Use for CNA % case-mix so dashboard and attorney report match."""
    quarter_data = df[df['CY_Qtr'] == quarter].copy()
    if quarter_data.empty:
        return None
    total_resident_days = quarter_data['MDScensus'].sum()
    if total_resident_days == 0:
        return None
    for c in ('Hrs_CNA', 'Hrs_MedAide', 'Hrs_NAtrn'):
        if c not in quarter_data.columns:
            quarter_data[c] = 0.0
    na_hours = (
        quarter_data['Hrs_CNA'].fillna(0) +
        quarter_data['Hrs_MedAide'].fillna(0) +
        quarter_data['Hrs_NAtrn'].fillna(0)
    ).sum()
    return float(na_hours / total_resident_days)


def _harrington_constants() -> dict:
    """Load Harrington formula constants from pbj-contract/definitions.yaml (single source of truth). Fallback to exact contract values if YAML missing."""
    try:
        from contract_adapter import get_contract, get_value
        contract = get_contract()
        if contract is None:
            raise ValueError("contract is None")
        base_cmi = float(get_value(contract, "definitions.harrington_case_mix.cmi_bounds.base_cmi") or 0.62)
        max_cmi = float(get_value(contract, "definitions.harrington_case_mix.cmi_bounds.max_cmi") or 3.84)
        total = {
            "intercept": float(get_value(contract, "definitions.harrington_case_mix.total.intercept") or 3.48),
            "high": float(get_value(contract, "definitions.harrington_case_mix.total.high") or 7.68),
            "exponent": float(get_value(contract, "definitions.harrington_case_mix.total.exponent") or 0.715361977219995),
        }
        rn = {
            "intercept": float(get_value(contract, "definitions.harrington_case_mix.rn.intercept") or 0.55),
            "high": float(get_value(contract, "definitions.harrington_case_mix.rn.high") or 2.39),
            "exponent": float(get_value(contract, "definitions.harrington_case_mix.rn.exponent") or 0.973947642000645),
        }
        cna = {
            "intercept": float(get_value(contract, "definitions.harrington_case_mix.cna.intercept") or 2.45),
            "high": float(get_value(contract, "definitions.harrington_case_mix.cna.high") or 3.6),
            "exponent": float(get_value(contract, "definitions.harrington_case_mix.cna.exponent") or 0.236050267902121),
        }
        return {"base_cmi": base_cmi, "max_cmi": max_cmi, "total": total, "rn": rn, "cna": cna}
    except Exception:
        # Fallback: exact values from pbj-contract/definitions.yaml (no truncation)
        return {
            "base_cmi": 0.62,
            "max_cmi": 3.84,
            "total": {"intercept": 3.48, "high": 7.68, "exponent": 0.715361977219995},
            "rn": {"intercept": 0.55, "high": 2.39, "exponent": 0.973947642000645},
            "cna": {"intercept": 2.45, "high": 3.6, "exponent": 0.236050267902121},
        }


def calculate_harrington_adjusted_hprd(cmi: float, hprd_type: str = 'total') -> Optional[float]:
    """
    Calculate Harrington-Adjusted Case-Mix HPRD. Constants (exponents, intercepts, CMI bounds) from
    pbj-contract/definitions.yaml harrington_case_mix so the formula is precise and single-sourced.
    """
    if cmi is None or pd.isna(cmi) or cmi <= 0:
        return None
    
    try:
        c = _harrington_constants()
        base_cmi = c["base_cmi"]
        max_cmi = c["max_cmi"]
        denominator = max_cmi - base_cmi
        if denominator <= 0:
            return None
        ratio = (cmi - base_cmi) / denominator

        if hprd_type == 'total':
            p = c["total"]
        elif hprd_type == 'rn':
            p = c["rn"]
        elif hprd_type == 'cna':
            p = c["cna"]
        else:
            return None

        power_factor = ratio ** p["exponent"]
        harrington_hprd = p["intercept"] + power_factor * (p["high"] - p["intercept"])
        return round_half_up(harrington_hprd, 2)
    except Exception as e:
        print(f"Error calculating Harrington-adjusted HPRD: {e}")
        return None

def generate_case_mix_section(case_mix_data: Optional[List[Dict]], quarterly_data: Optional[Dict] = None, pbj_df: Optional[pd.DataFrame] = None) -> str:
    """Generate HTML section for case-mix adjusted HPRD comparison with Harrington-adjusted calculations."""
    # Include all entries that have any case-mix data (total or RN); CMI is optional (needed for Harrington only)
    entries_with_case_mix = []
    if case_mix_data:
        for entry in case_mix_data:
            if (entry.get('case_mix_total') is not None and not pd.isna(entry.get('case_mix_total'))) or \
               (entry.get('case_mix_rn') is not None and not pd.isna(entry.get('case_mix_rn'))):
                entries_with_case_mix.append(entry)
    
    # If no case-mix data at all, return message
    if not entries_with_case_mix:
        return """
    <h2>Case-Mix Staffing Analysis</h2>
    <p><em>Case-mix adjusted staffing data (CMI) is not available for this period. CMI data is typically available from 2024 onwards.</em></p>
    """
    
    # Group by quarter (all quarters with case-mix data; CMI may be present or not per quarter)
    quarter_dict = {}
    quarters_with_case_mix = set()
    for entry in entries_with_case_mix:
        quarter = entry['quarter']
        if quarter not in quarter_dict:
            quarter_dict[quarter] = []
        quarter_dict[quarter].append(entry)
        quarters_with_case_mix.add(quarter)
    
    case_mix_rows = []
    harrington_rows = []
    occupancy_rows = []
    # Process all quarters that have case-mix data (Harrington columns show N/A when CMI missing)
    for quarter in sorted(quarters_with_case_mix):
        entries = quarter_dict.get(quarter, [])
        
        # Calculate averages for the quarter - Total, Direct Care, RN, and CNA from provider info
        case_mix_totals = [e['case_mix_total'] for e in entries if e['case_mix_total'] is not None]
        case_mix_directs = [e['case_mix_direct'] for e in entries if e.get('case_mix_direct') is not None]
        case_mix_rns = [e['case_mix_rn'] for e in entries if e['case_mix_rn'] is not None]
        case_mix_nas = [e['case_mix_na'] for e in entries if e.get('case_mix_na') is not None]
        reported_totals = [e['reported_total'] for e in entries if e['reported_total'] is not None]
        reported_rns = [e['reported_rn'] for e in entries if e['reported_rn'] is not None]
        reported_nas = [e['reported_na'] for e in entries if e.get('reported_na') is not None]
        
        # Get CMI for Harrington-adjusted calculations (use full-precision cmi_raw to avoid rounding drift)
        cmi_values = [e.get('cmi_raw') if e.get('cmi_raw') is not None else e['cmi'] for e in entries if e.get('cmi') is not None]
        avg_cmi = sum(cmi_values) / len(cmi_values) if cmi_values else None
        
        
        avg_case_mix_total = sum(case_mix_totals) / len(case_mix_totals) if case_mix_totals else None
        avg_case_mix_direct = sum(case_mix_directs) / len(case_mix_directs) if case_mix_directs else None
        avg_case_mix_rn = sum(case_mix_rns) / len(case_mix_rns) if case_mix_rns else None
        avg_case_mix_na = sum(case_mix_nas) / len(case_mix_nas) if case_mix_nas else None
        avg_reported_total = sum(reported_totals) / len(reported_totals) if reported_totals else None
        avg_reported_rn = sum(reported_rns) / len(reported_rns) if reported_rns else None
        
        # Raw direct care HPRD from PBJ (no rounding) for % case-mix - match dashboard and avoid 89.0 vs 89.7 discrepancy
        raw_direct_hprd = _quarterly_direct_care_hprd_raw(pbj_df, quarter) if pbj_df is not None else None
        
        # PRIORITIZE: Calculate reported total from PBJ data (including admin/DON) - this is the primary source
        avg_reported_total_pbj = None
        if pbj_df is not None:
            quarter_data = pbj_df[pbj_df['CY_Qtr'] == quarter].copy()
            if not quarter_data.empty:
                total_resident_days = quarter_data['MDScensus'].sum()
                if total_resident_days > 0:
                    # Total nurse hours = RN + RN Admin + RN DON + LPN + LPN Admin + CNA + MedAide + NAtrn
                    total_nurse_hours = (
                        quarter_data['Hrs_RN'].fillna(0) +
                        quarter_data['Hrs_RNadmin'].fillna(0) +
                        quarter_data['Hrs_RNDON'].fillna(0) +
                        quarter_data['Hrs_LPN'].fillna(0) +
                        quarter_data['Hrs_LPNadmin'].fillna(0) +
                        quarter_data['Hrs_CNA'].fillna(0) +
                        quarter_data['Hrs_MedAide'].fillna(0) +
                        quarter_data['Hrs_NAtrn'].fillna(0)
                    ).sum()
                    avg_reported_total_pbj = round_half_up(total_nurse_hours / total_resident_days, 2)
        
        # Use PBJ total if available, otherwise use provider info
        if avg_reported_total_pbj is not None:
            avg_reported_total = avg_reported_total_pbj
        
        # PRIORITIZE: Calculate reported direct care from PBJ data (excluding admin/DON) - this is the primary source
        avg_reported_direct = None
        if pbj_df is not None:
            avg_reported_direct = calculate_quarterly_direct_care_hprd(pbj_df, quarter)
        
        # If PBJ direct care not available, try provider info reported direct
        if avg_reported_direct is None:
            reported_directs = [e['reported_direct'] for e in entries if e.get('reported_direct') is not None]
            avg_reported_direct = sum(reported_directs) / len(reported_directs) if reported_directs else None
        
        # PRIORITIZE: Calculate reported total RN from PBJ data (including admin/DON)
        avg_reported_total_rn_pbj = None
        if pbj_df is not None:
            quarter_data = pbj_df[pbj_df['CY_Qtr'] == quarter].copy()
            if not quarter_data.empty:
                total_resident_days = quarter_data['MDScensus'].sum()
                if total_resident_days > 0:
                    # Total RN hours = RN + RN Admin + RN DON
                    total_rn_hours = (
                        quarter_data['Hrs_RN'].fillna(0) +
                        quarter_data['Hrs_RNadmin'].fillna(0) +
                        quarter_data['Hrs_RNDON'].fillna(0)
                    ).sum()
                    avg_reported_total_rn_pbj = round_half_up(total_rn_hours / total_resident_days, 2)
        
        # Use PBJ total RN if available, otherwise use provider info
        if avg_reported_total_rn_pbj is not None:
            avg_reported_rn = avg_reported_total_rn_pbj
        elif avg_reported_rn is None:
            avg_reported_rn = None
        
        # PRIORITIZE: Calculate reported direct care RN from PBJ data (Hrs_RN only, excluding admin/DON)
        # Use raw for % case-mix so it matches dashboard (48.2% not 48.0%); round only for display.
        raw_direct_rn_hprd = _quarterly_direct_rn_hprd_raw(pbj_df, quarter) if pbj_df is not None else None
        avg_reported_direct_rn = round_half_up(raw_direct_rn_hprd, 2) if raw_direct_rn_hprd is not None else None
        if avg_reported_direct_rn is None and pbj_df is not None:
            quarter_data = pbj_df[pbj_df['CY_Qtr'] == quarter].copy()
            if not quarter_data.empty:
                total_resident_days = quarter_data['MDScensus'].sum()
                if total_resident_days > 0:
                    direct_care_rn_hours = quarter_data['Hrs_RN'].fillna(0).sum()
                    avg_reported_direct_rn = round_half_up(direct_care_rn_hours / total_resident_days, 2)
        
        # PRIORITIZE: Calculate reported direct care CNA from PBJ data (CNA + MedAide + NAtrn)
        # Use raw for % case-mix so it matches dashboard; round only for display.
        raw_direct_cna_hprd = _quarterly_direct_cna_hprd_raw(pbj_df, quarter) if pbj_df is not None else None
        avg_reported_direct_cna = round_half_up(raw_direct_cna_hprd, 2) if raw_direct_cna_hprd is not None else None
        if avg_reported_direct_cna is None and pbj_df is not None:
            quarter_data = pbj_df[pbj_df['CY_Qtr'] == quarter].copy()
            if not quarter_data.empty:
                total_resident_days = quarter_data['MDScensus'].sum()
                if total_resident_days > 0:
                    total_nurse_aide_hours = (
                        quarter_data['Hrs_CNA'].fillna(0) +
                        quarter_data['Hrs_MedAide'].fillna(0) +
                        quarter_data['Hrs_NAtrn'].fillna(0)
                    ).sum()
                    avg_reported_direct_cna = round_half_up(total_nurse_aide_hours / total_resident_days, 2)
        if avg_reported_direct_cna is None:
            avg_reported_direct_cna = sum(reported_nas) / len(reported_nas) if reported_nas else None
        
        # Calculate Harrington-adjusted HPRD if CMI is available
        harrington_total = None
        harrington_rn = None
        harrington_cna = None
        if avg_cmi is not None:
            harrington_total = calculate_harrington_adjusted_hprd(avg_cmi, 'total')
            harrington_rn = calculate_harrington_adjusted_hprd(avg_cmi, 'rn')
            harrington_cna = calculate_harrington_adjusted_hprd(avg_cmi, 'cna')
        
        # Calculate proportion: (reported / case_mix) * 100 - PRIORITIZE DIRECT CARE
        # Direct %: use case_mix_total as denominator (same as dashboard) and raw direct HPRD for numerator
        pct_prop_direct = (raw_direct_hprd / avg_case_mix_total * 100) if (raw_direct_hprd is not None and avg_case_mix_total is not None and avg_case_mix_total > 0) else (avg_reported_direct / avg_case_mix_direct * 100) if (avg_reported_direct is not None and avg_case_mix_direct is not None and avg_case_mix_direct > 0) else None
        pct_prop_total = (avg_reported_total / avg_case_mix_total * 100) if (avg_reported_total is not None and avg_case_mix_total is not None and avg_case_mix_total > 0) else None
        # RN % case-mix: use raw RN HPRD when available so attorney report matches dashboard (e.g. 48.2% not 48.0%)
        pct_prop_direct_rn = (raw_direct_rn_hprd / avg_case_mix_rn * 100) if (raw_direct_rn_hprd is not None and avg_case_mix_rn is not None and avg_case_mix_rn > 0) else (avg_reported_direct_rn / avg_case_mix_rn * 100) if (avg_reported_direct_rn is not None and avg_case_mix_rn is not None and avg_case_mix_rn > 0) else None
        pct_prop_total_rn = (avg_reported_rn / avg_case_mix_rn * 100) if (avg_reported_rn is not None and avg_case_mix_rn is not None and avg_case_mix_rn > 0) else None
        
        # Nurse aide % case-mix: use raw CNA HPRD when available so attorney report matches dashboard
        pct_prop_cna = (raw_direct_cna_hprd / avg_case_mix_na * 100) if (raw_direct_cna_hprd is not None and avg_case_mix_na is not None and avg_case_mix_na > 0) else (avg_reported_direct_cna / avg_case_mix_na * 100) if (avg_reported_direct_cna is not None and avg_case_mix_na is not None and avg_case_mix_na > 0) else None
        
        # Calculate Harrington-adjusted proportions - both total and direct
        pct_prop_harrington_total_direct = (avg_reported_direct / harrington_total * 100) if (avg_reported_direct is not None and harrington_total is not None and harrington_total > 0) else None
        pct_prop_harrington_total_total = (avg_reported_total / harrington_total * 100) if (avg_reported_total is not None and harrington_total is not None and harrington_total > 0) else None
        pct_prop_harrington_rn_direct = (avg_reported_direct_rn / harrington_rn * 100) if (avg_reported_direct_rn is not None and harrington_rn is not None and harrington_rn > 0) else None
        pct_prop_harrington_rn_total = (avg_reported_rn / harrington_rn * 100) if (avg_reported_rn is not None and harrington_rn is not None and harrington_rn > 0) else None
        pct_prop_harrington_cna = (avg_reported_direct_cna / harrington_cna * 100) if (avg_reported_direct_cna is not None and harrington_cna is not None and harrington_cna > 0) else None
        
        quarter_display = format_quarter_display(quarter) if quarter != 'Unknown' else 'Unknown'
        cmi_display = f"{avg_cmi:.2f}" if avg_cmi is not None else 'N/A'
        
        # Format values with better formatting for % columns
        # Case-Mix: Quarter, Case-Mix Index, Total Nurse Case-Mix HPRD, % Case-Mix (Total), % Direct Case-Mix, RN Case-Mix HPRD, % RN Case-Mix (Total), % RN Only Case-Mix (Direct), Nurse Aide Case-Mix HPRD, % Nurse Aide Case-Mix
        case_mix_total_hprd = f"{avg_case_mix_total:.2f} HPRD" if avg_case_mix_total is not None else 'N/A'
        if pct_prop_total is not None and avg_reported_total is not None and avg_case_mix_total is not None:
            case_mix_total_pct = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_total:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_total:.2f} / {avg_case_mix_total:.2f}</div></div>'
        else:
            case_mix_total_pct = 'N/A'
        
        case_mix_direct_hprd = f"{avg_case_mix_total:.2f} HPRD" if avg_case_mix_total is not None else (f"{avg_case_mix_direct:.2f} HPRD" if avg_case_mix_direct is not None else 'N/A')
        # Show reported (rounded) / case_mix_total so denominator matches dashboard and Total column
        direct_denom = avg_case_mix_total if avg_case_mix_total is not None else avg_case_mix_direct
        if pct_prop_direct is not None and avg_reported_direct is not None and direct_denom is not None:
            case_mix_direct_pct = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_direct:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_direct:.2f} / {direct_denom:.2f}</div></div>'
        else:
            case_mix_direct_pct = 'N/A'
        
        case_mix_rn_hprd = f"{avg_case_mix_rn:.2f} HPRD" if avg_case_mix_rn is not None else 'N/A'
        if pct_prop_total_rn is not None and avg_reported_rn is not None and avg_case_mix_rn is not None:
            case_mix_rn_total_pct = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_total_rn:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_rn:.2f} / {avg_case_mix_rn:.2f}</div></div>'
        else:
            case_mix_rn_total_pct = 'N/A'
        
        if pct_prop_direct_rn is not None and avg_reported_direct_rn is not None and avg_case_mix_rn is not None:
            case_mix_rn_direct_pct = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_direct_rn:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_direct_rn:.2f} / {avg_case_mix_rn:.2f}</div></div>'
        else:
            case_mix_rn_direct_pct = 'N/A'
        
        case_mix_na_hprd = f"{avg_case_mix_na:.2f} HPRD" if avg_case_mix_na is not None else 'N/A'
        if pct_prop_cna is not None and avg_reported_direct_cna is not None and avg_case_mix_na is not None:
            case_mix_na_pct = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_cna:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_direct_cna:.2f} / {avg_case_mix_na:.2f}</div></div>'
        else:
            case_mix_na_pct = 'N/A'
        
        # Harrington: Quarter, Case-Mix Index, Total Expected, % Expected (Total), % Direct Expected, RN Expected, % RN Expected (Total), % RN Only Expected (Direct), Nurse Aide Expected, % Nurse Aide Expected
        harrington_total_expected = f"{harrington_total:.2f} HPRD" if harrington_total is not None else 'N/A'
        if pct_prop_harrington_total_total is not None and avg_reported_total is not None and harrington_total is not None:
            harrington_total_pct_total = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_harrington_total_total:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_total:.2f} / {harrington_total:.2f}</div></div>'
        else:
            harrington_total_pct_total = 'N/A'
        
        if pct_prop_harrington_total_direct is not None and avg_reported_direct is not None and harrington_total is not None:
            harrington_total_pct_direct = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_harrington_total_direct:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_direct:.2f} / {harrington_total:.2f}</div></div>'
        else:
            harrington_total_pct_direct = 'N/A'
        
        harrington_rn_expected = f"{harrington_rn:.2f} HPRD" if harrington_rn is not None else 'N/A'
        if pct_prop_harrington_rn_total is not None and avg_reported_rn is not None and harrington_rn is not None:
            harrington_rn_pct_total = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_harrington_rn_total:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_rn:.2f} / {harrington_rn:.2f}</div></div>'
        else:
            harrington_rn_pct_total = 'N/A'
        
        if pct_prop_harrington_rn_direct is not None and avg_reported_direct_rn is not None and harrington_rn is not None:
            harrington_rn_pct_direct = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_harrington_rn_direct:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_direct_rn:.2f} / {harrington_rn:.2f}</div></div>'
        else:
            harrington_rn_pct_direct = 'N/A'
        
        harrington_na_expected = f"{harrington_cna:.2f} HPRD" if harrington_cna is not None else 'N/A'
        if pct_prop_harrington_cna is not None and avg_reported_direct_cna is not None and harrington_cna is not None:
            harrington_na_pct = f'<div style="background-color: #ffe8e8; padding: 8px; text-align: center; width: 100%; height: 100%; box-sizing: border-box;"><div style="font-weight: bold; font-size: 11pt; color: #2c3e50;">{pct_prop_harrington_cna:.1f}%</div><div style="font-size: 8pt; color: #666; margin-top: 2px;">{avg_reported_direct_cna:.2f} / {harrington_cna:.2f}</div></div>'
        else:
            harrington_na_pct = 'N/A'
        
        # Case-Mix rows - new order with formatted % cells (using cell background instead of div)
        # Use the pre-formatted variables that handle None values
        case_mix_total_pct_display = case_mix_total_pct if case_mix_total_pct != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        case_mix_direct_pct_display = case_mix_direct_pct if case_mix_direct_pct != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        case_mix_rn_total_pct_display = case_mix_rn_total_pct if case_mix_rn_total_pct != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        case_mix_rn_direct_pct_display = case_mix_rn_direct_pct if case_mix_rn_direct_pct != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        case_mix_na_pct_display = case_mix_na_pct if case_mix_na_pct != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        
        case_mix_rows.append(f"""
        <tr>
            <td>{quarter_display}</td>
            <td>{cmi_display}</td>
            <td>{case_mix_total_hprd}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{case_mix_total_pct_display}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{case_mix_direct_pct_display}</td>
            <td>{case_mix_rn_hprd}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{case_mix_rn_total_pct_display}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{case_mix_rn_direct_pct_display}</td>
            <td>{case_mix_na_hprd}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{case_mix_na_pct_display}</td>
        </tr>
        """)
        
        # Harrington rows - new order with formatted % cells (using cell background instead of div)
        # Use the pre-formatted variables that handle None values
        harrington_total_pct_total_display = harrington_total_pct_total if harrington_total_pct_total != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        harrington_total_pct_direct_display = harrington_total_pct_direct if harrington_total_pct_direct != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        harrington_rn_pct_total_display = harrington_rn_pct_total if harrington_rn_pct_total != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        harrington_rn_pct_direct_display = harrington_rn_pct_direct if harrington_rn_pct_direct != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        harrington_na_pct_display = harrington_na_pct if harrington_na_pct != 'N/A' else '<div style="text-align: center; padding: 8px;">N/A</div>'
        
        harrington_rows.append(f"""
        <tr>
            <td>{quarter_display}</td>
            <td>{cmi_display}</td>
            <td>{harrington_total_expected}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{harrington_total_pct_total_display}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{harrington_total_pct_direct_display}</td>
            <td>{harrington_rn_expected}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{harrington_rn_pct_total_display}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{harrington_rn_pct_direct_display}</td>
            <td>{harrington_na_expected}</td>
            <td style="background-color: #ffe8e8; text-align: center; padding: 8px;">{harrington_na_pct_display}</td>
        </tr>
        """)
        # Occupancy: one row per quarter when we have occupancy_pct (from Provider Info, same quarter matching)
        occ_entry = next((e for e in entries if e.get('occupancy_pct') is not None and not pd.isna(e.get('occupancy_pct'))), None)
        if occ_entry is not None and occ_entry.get('certified_beds') and (occ_entry.get('certified_beds') or 0) > 0:
            occ_pct = occ_entry['occupancy_pct']
            occ_census = occ_entry.get('avg_residents_per_day')
            occ_beds = occ_entry.get('certified_beds')
            census_str = f"{occ_census:.1f}" if occ_census is not None and not pd.isna(occ_census) else "—"
            beds_str = str(int(occ_beds)) if occ_beds is not None else "—"
            occupancy_rows.append(f"""
        <tr>
            <td>{quarter_display}</td>
            <td>{occ_pct:.2f}%</td>
            <td>{census_str}</td>
            <td>{beds_str}</td>
        </tr>
        """)
    
    occupancy_table = ""
    if occupancy_rows:
        occupancy_table = f"""
    <h3 style="margin-top: 20px; margin-bottom: 10px; color: #2c3e50; font-size: 12pt; font-weight: 600;">Occupancy Rate</h3>
    <p style="margin-bottom: 10px; font-size: 9pt; color: #555;">Resident census ÷ certified beds from Provider Info (quarter-matched). Quarters with 0 beds or 0 census are omitted.</p>
    <table style="margin-bottom: 20px;">
        <thead>
            <tr>
                <th>Quarter</th>
                <th>Occupancy %</th>
                <th>Avg. census</th>
                <th>Certified beds</th>
            </tr>
        </thead>
        <tbody>
            {''.join(occupancy_rows)}
        </tbody>
    </table>
    """
    
    return f"""
    <h2>Case-Mix Staffing Analysis</h2>
    <p style="margin-bottom: 20px;">Case-mix staffing analysis accounts for resident acuity based on CMS Case-Mix. Values below 100% indicate staffing below case-mix. Direct Care metrics (excluding admin/DON) are prioritized and calculated from PBJ data.</p>
    
    <h3 style="margin-top: 20px; margin-bottom: 10px; color: #2c3e50; font-size: 12pt; font-weight: 600;">Case-Mix Analysis (Reported HPRD / Case-Mix)</h3>
    <table style="margin-bottom: 20px;">
        <thead>
            <tr>
                <th>Quarter</th>
                <th>Case-Mix Index</th>
                <th>Total Nurse<br>Case-Mix</th>
                <th>% Case-Mix</th>
                <th>% Direct Case-Mix</th>
                <th>RN Case-Mix</th>
                <th>% RN Case-Mix</th>
                <th>% RN Only Case-Mix</th>
                <th>Nurse Aide Case-Mix</th>
                <th>% Nurse Aide<br>Case-Mix</th>
            </tr>
        </thead>
        <tbody>
            {''.join(case_mix_rows)}
        </tbody>
    </table>
    
    <h3 style="margin-top: 20px; margin-bottom: 10px; color: #2c3e50; font-size: 12pt; font-weight: 600;">Harrington Expected Staffing</h3>
    <table style="margin-bottom: 15px;">
        <thead>
            <tr>
                <th>Quarter</th>
                <th>Case-Mix Index</th>
                <th>Total Expected</th>
                <th>% Expected</th>
                <th>% Direct Expected</th>
                <th>RN Expected</th>
                <th>% RN Expected</th>
                <th>% RN Only Expected</th>
                <th>Nurse Aide Expected</th>
                <th>% Nurse Aide<br>Expected</th>
            </tr>
        </thead>
        <tbody>
            {''.join(harrington_rows)}
        </tbody>
    </table>
    {occupancy_table}
    <div style="margin-top: 15px; padding: 12px; background-color: #f8f9fa; border-left: 4px solid #3498db; border-radius: 4px;">
        <p style="margin: 0 0 8px 0; font-weight: bold; color: #2c3e50; font-size: 10pt;">Harrington Expected Formulas:</p>
        <div style="font-family: 'Times New Roman', 'Georgia', serif; line-height: 1.6; color: #34495e; font-size: 8.5pt; background-color: white; padding: 10px; border-radius: 3px;">
            <p style="margin: 4px 0; font-size: 8.5pt;"><strong>Total Expected HPRD</strong> = 3.48 + <span style="display: inline-block; text-align: center; vertical-align: middle; margin: 0 2px; font-size: 8pt;">
                <span style="display: block; border-bottom: 1px solid #34495e; padding: 0 3px 1px 3px; font-size: 7.5pt;">CMI - 0.62</span>
                <span style="display: block; padding: 1px 3px 0 3px; font-size: 7.5pt;">3.84 - 0.62</span>
            </span><sup style="font-size: 6.5pt; line-height: 0; vertical-align: 0.5em; font-weight: normal;">0.715</sup> × (7.68 - 3.48)</p>
            <p style="margin: 4px 0; font-size: 8.5pt;"><strong>RN Expected HPRD</strong> = 0.55 + <span style="display: inline-block; text-align: center; vertical-align: middle; margin: 0 2px; font-size: 8pt;">
                <span style="display: block; border-bottom: 1px solid #34495e; padding: 0 3px 1px 3px; font-size: 7.5pt;">CMI - 0.62</span>
                <span style="display: block; padding: 1px 3px 0 3px; font-size: 7.5pt;">3.84 - 0.62</span>
            </span><sup style="font-size: 6.5pt; line-height: 0; vertical-align: 0.5em; font-weight: normal;">0.974</sup> × (2.39 - 0.55)</p>
            <p style="margin: 4px 0; font-size: 8.5pt;"><strong>Nurse Aide Expected HPRD</strong> = 2.45 + <span style="display: inline-block; text-align: center; vertical-align: middle; margin: 0 2px; font-size: 8pt;">
                <span style="display: block; border-bottom: 1px solid #34495e; padding: 0 3px 1px 3px; font-size: 7.5pt;">CMI - 0.62</span>
                <span style="display: block; padding: 1px 3px 0 3px; font-size: 7.5pt;">3.84 - 0.62</span>
            </span><sup style="font-size: 6.5pt; line-height: 0; vertical-align: 0.5em; font-weight: normal;">0.236</sup> × (3.6 - 2.45)</p>
        </div>
        <p style="margin: 8px 0 0 0; font-size: 9pt;"><strong>See:</strong> <a href="https://agsjournals.onlinelibrary.wiley.com/doi/10.1111/jgs.19501" target="_blank" style="color: #3498db;">Nursing Home Guide to Adjusting Nurse Staffing for Resident Case-Mix</a> (Harrington) | <strong>Note:</strong> Direct Care metrics are calculated from PBJ and exclude Admin/DON.</p>
    </div>
    """

def load_facility_citations(provnum: str, limit: int = 10) -> List[Dict]:
    """
    Load the most recent citations for a facility from NH_HealthCitations_Dec2025.csv.
    
    Args:
        provnum: Facility CCN (6-digit)
        limit: Maximum number of citations to return (default: 2 for most recent)
        
    Returns:
        List of citation dictionaries with key columns
    """
    provnum = str(provnum).strip().zfill(6)
    citations = []
    
    citations_file = 'Citations/NH_HealthCitations_Dec2025.csv'
    if not os.path.exists(citations_file):
        return citations
    
    try:
        df = pd.read_csv(citations_file, low_memory=False, dtype={'CMS Certification Number (CCN)': str})
        
        # Normalize CCN column
        ccn_col = 'CMS Certification Number (CCN)'
        if ccn_col in df.columns:
            df[ccn_col] = df[ccn_col].astype(str).str.strip().str.zfill(6)
            facility_citations = df[df[ccn_col] == provnum].copy()
            
            if not facility_citations.empty:
                # Sort by Survey Date (most recent first)
                if 'Survey Date' in facility_citations.columns:
                    facility_citations['Survey Date'] = pd.to_datetime(facility_citations['Survey Date'], errors='coerce')
                    facility_citations = facility_citations.sort_values('Survey Date', ascending=False)
                
                # Get most recent citations (limit)
                for _, row in facility_citations.head(limit).iterrows():
                    survey_date_raw = row.get('Survey Date', '')
                    survey_date_dt = None
                    
                    # Parse survey date
                    if isinstance(survey_date_raw, pd.Timestamp):
                        survey_date_dt = survey_date_raw
                    elif pd.notna(survey_date_raw):
                        try:
                            survey_date_dt = pd.to_datetime(str(survey_date_raw))
                        except:
                            pass
                    
                    # Format survey date for display (MM/DD/YYYY); URL stays ISO for links
                    survey_date_display = ''
                    survey_date_url = None
                    if survey_date_dt is not None:
                        survey_date_display = survey_date_dt.strftime('%m/%d/%Y')
                        survey_date_url = survey_date_dt.strftime('%Y-%m-%d')
                    elif pd.notna(survey_date_raw):
                        survey_date_display = str(survey_date_raw)
                    
                    citation = {
                        'survey_date': survey_date_display,
                        'survey_date_url': survey_date_url,
                        'deficiency_category': row.get('Deficiency Category', ''),
                        'deficiency_tag_number': row.get('Deficiency Tag Number', ''),
                        'deficiency_description': row.get('Deficiency Description', ''),
                        'scope_severity_code': row.get('Scope Severity Code', ''),
                        'complaint_deficiency': row.get('Complaint Deficiency', 'N'),
                        'infection_control': row.get('Infection Control Inspection Deficiency', 'N'),
                    }
                    citations.append(citation)
    except Exception as e:
        print(f"Error loading citations: {e}")
    
    return citations


def load_entity_info(provnum: str) -> Optional[Dict]:
    """
    Load entity information for a facility from the latest entity performance measures file.
    Shows entity affiliation history with quarters.
    
    Args:
        provnum: Facility CCN (6-digit)
        
    Returns:
        Dictionary with entity information and facilities list, or None if not found
    """
    provnum = str(provnum).strip().zfill(6)
    
    # Find latest entity file
    entity_files = [
        'Nursing_Home_Affiliated_Entity_Performance_Measures_Jun_2025.csv',
        'Nursing_Home_Affiliated_Entity_Performance_Measures_Mar_2025.csv',
    ]
    
    entity_file = None
    for f in entity_files:
        if os.path.exists(f):
            entity_file = f
            break
    
    if not entity_file:
        return None
    
    try:
        # Get all provider info records for this facility
        provider_info = load_provider_info_data(provnum)
        if provider_info.empty:
            return None
        
        # Find entity ID column
        entity_id_col = None
        for col in ['affiliated_entity_id', 'Entity ID', 'entity_id', 'Chain ID', 'chain_id']:
            if col in provider_info.columns:
                entity_id_col = col
                break
        
        if not entity_id_col:
            return None
        
        # Sort by processing_date to get chronological order
        if 'processing_date' in provider_info.columns:
            provider_info = provider_info.copy()
            provider_info['processing_date'] = pd.to_datetime(provider_info['processing_date'], errors='coerce')
            provider_info = provider_info.sort_values('processing_date', ascending=True)
        
        # Group by entity ID and collect quarters
        entity_history = {}
        for _, row in provider_info.iterrows():
            entity_id = row.get(entity_id_col)
            if pd.notna(entity_id) and entity_id:
                entity_id_str = str(entity_id).strip()
                quarter = row.get('quarter', '')
                
                if entity_id_str not in entity_history:
                    entity_history[entity_id_str] = {
                        'entity_id': entity_id_str,
                        'quarters': [],
                        'first_seen': row.get('processing_date'),
                        'last_seen': row.get('processing_date')
                    }
                
                if quarter and pd.notna(quarter) and quarter not in entity_history[entity_id_str]['quarters']:
                    entity_history[entity_id_str]['quarters'].append(str(quarter))
                
                # Update last seen date
                if pd.notna(row.get('processing_date')):
                    if pd.isna(entity_history[entity_id_str]['last_seen']) or row.get('processing_date') > entity_history[entity_id_str]['last_seen']:
                        entity_history[entity_id_str]['last_seen'] = row.get('processing_date')
        
        if not entity_history:
            # Facility is not part of an entity/chain
            return None
        
        # Get the most recent entity (by last_seen date)
        most_recent_entity_id = max(entity_history.keys(), key=lambda x: entity_history[x]['last_seen'] if pd.notna(entity_history[x]['last_seen']) else pd.Timestamp.min)
        entity_id = most_recent_entity_id
        
        entity_id = str(entity_id).strip()
        
        # Load entity performance measures
        entity_df = pd.read_csv(entity_file, low_memory=False)
        
        # Find entity ID column in entity file
        entity_id_col_file = None
        for col in ['Affiliated entity ID', 'Affiliated entity', 'Entity ID']:
            if col in entity_df.columns:
                entity_id_col_file = col
                break
        
        if not entity_id_col_file:
            return None
        
        # Match entity (try both string and numeric)
        entity_row = None
        try:
            entity_id_float = float(entity_id)
            entity_row = entity_df[pd.to_numeric(entity_df[entity_id_col_file], errors='coerce') == entity_id_float]
        except:
            entity_row = entity_df[entity_df[entity_id_col_file].astype(str).str.strip() == entity_id]
        
        if entity_row.empty:
            return None
        
        entity_row = entity_row.iloc[0]
        
        # Get entity name
        entity_name_col = None
        for col in ['Affiliated entity', 'Entity Name', 'entity_name']:
            if col in entity_df.columns:
                entity_name_col = col
                break
        
        entity_name = str(entity_row.get(entity_name_col, '')) if entity_name_col else ''
        
        # Get basic entity stats and ratings (for compact display)
        entity_info = {
            'entity_id': entity_id,
            'entity_name': entity_name,
            'num_facilities': entity_row.get('Number of facilities', ''),
            'num_states': entity_row.get('Number of states and territories with operations', ''),
            'num_sff': entity_row.get('Number of Special Focus Facilities (SFF)', ''),
            'avg_overall_rating': entity_row.get('Average overall 5-star rating', ''),
            'avg_health_rating': entity_row.get('Average health inspection rating', ''),
            'avg_staffing_rating': entity_row.get('Average staffing rating', ''),
            'avg_quality_rating': entity_row.get('Average quality rating', ''),
        }
        
        # Get facilities list from provider info with additional data
        facilities = []
        all_provider_info = pd.read_csv('provider_info_combined.csv', low_memory=False, dtype={'ccn': str}) if os.path.exists('provider_info_combined.csv') else pd.DataFrame()
        
        if not all_provider_info.empty and entity_id_col in all_provider_info.columns:
            # Normalize entity ID for comparison
            try:
                entity_id_float = float(entity_id)
                entity_facilities = all_provider_info[pd.to_numeric(all_provider_info[entity_id_col], errors='coerce') == entity_id_float].copy()
            except:
                entity_facilities = all_provider_info[all_provider_info[entity_id_col].astype(str).str.strip() == entity_id].copy()
            
            # Get latest record for each facility
            if 'processing_date' in entity_facilities.columns:
                entity_facilities = entity_facilities.copy()
                entity_facilities['processing_date'] = pd.to_datetime(entity_facilities['processing_date'], errors='coerce')
                entity_facilities = entity_facilities.sort_values('processing_date', ascending=False)
                entity_facilities = entity_facilities.drop_duplicates(subset=['ccn'], keep='first')
            
            # Build facilities list with additional data
            for _, fac_row in entity_facilities.iterrows():
                ccn = str(fac_row.get('ccn', '')).strip().zfill(6)
                name = str(fac_row.get('provider_name', fac_row.get('Provider Name', ''))).strip()
                state = str(fac_row.get('state', '')).strip().upper()[:2]
                
                if ccn and name:
                    # Get red flags
                    red_flags = []
                    sff_status = str(fac_row.get('sff_status', '')).strip().upper() if pd.notna(fac_row.get('sff_status')) else ''
                    if sff_status and sff_status not in ['N', 'N/A', 'NAN', 'NONE', '']:
                        if 'SFF' in sff_status:
                            red_flags.append('SFF')
                    
                    overall_rating = fac_row.get('overall_rating')
                    if pd.notna(overall_rating) and overall_rating == 1.0:
                        red_flags.append('1-Star')
                    
                    abuse_icon = str(fac_row.get('abuse_icon', '')).strip() if pd.notna(fac_row.get('abuse_icon')) else ''
                    if abuse_icon and abuse_icon.upper() not in ['N', 'N/A', 'NAN', 'NONE', '']:
                        red_flags.append('Abuse')
                    
                    # Get staffing HPRD
                    staffing_hprd = None
                    for col in ['reported_total_nurse_hrs_per_resident_per_day', 'Total_HPRD', 'total_hprd']:
                        if col in fac_row.index and pd.notna(fac_row.get(col)):
                            try:
                                val = fac_row.get(col)
                                if val is not None:
                                    staffing_hprd = float(val)
                                    break
                            except (ValueError, TypeError):
                                continue
                    
                    # Get census
                    census = fac_row.get('avg_residents_per_day', fac_row.get('census', None))
                    if pd.isna(census) or census is None:
                        census = None
                    else:
                        try:
                            census_val = float(census)
                            census = census_val if not pd.isna(census_val) else None
                        except (ValueError, TypeError):
                            census = None
                    
                    # Get rating
                    rating = overall_rating if pd.notna(overall_rating) else None
                    
                    facilities.append({
                        'ccn': ccn,
                        'name': name,
                        'state': state,
                        'red_flags': ', '.join(red_flags) if red_flags else 'None',
                        'staffing_hprd': staffing_hprd,
                        'census': census,
                        'rating': rating
                    })
        
        entity_info['facilities'] = sorted(facilities, key=lambda x: (x['state'], x['name']))
        
        # Add entity history information
        entity_info['entity_history'] = entity_history
        entity_info['current_entity_id'] = entity_id
        # Don't sort here - let _format_quarter_range handle proper chronological sorting
        entity_info['current_entity_quarters'] = entity_history[entity_id]['quarters'] if entity_id in entity_history else []
        
        return entity_info
    except Exception as e:
        print(f"Error loading entity info: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_time_series_data(
    df: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    include_total_staffing: bool = True,
    certified_beds: Optional[float] = None,
) -> Dict:
    """Generate time series data for charts aggregated by day, month, and quarter.

    When ``certified_beds`` is set and positive, each point includes ``occupancy_pct``
    (100 × mean daily census ÷ certified beds) for charting alongside raw census.
    """
    if df is None or df.empty:
        return {'daily': [], 'monthly': [], 'quarterly': []}
    
    # Ensure WorkDate is datetime
    if df['WorkDate'].dtype != 'datetime64[ns]':
        df['WorkDate'] = pd.to_datetime(df['WorkDate'], errors='coerce')
    
    # Filter to date range
    period_df = df[
        (df['WorkDate'] >= start_date) &
        (df['WorkDate'] <= end_date) &
        (df['MDScensus'] > 0)
    ].copy()
    
    if period_df.empty:
        return {'daily': [], 'monthly': [], 'quarterly': []}

    def _h(col: str) -> Any:
        if col not in period_df.columns:
            return pd.Series(0.0, index=period_df.index, dtype=float)
        return period_df[col].fillna(0).astype(float)

    if "CY_Qtr" not in period_df.columns:
        period_df["CY_Qtr"] = (
            period_df["WorkDate"].dt.year.astype(str) + "Q" + period_df["WorkDate"].dt.quarter.astype(str)
        )

    def _occ_pct(census_val: Any) -> Optional[float]:
        if certified_beds is None or certified_beds <= 0:
            return None
        if census_val is None or (isinstance(census_val, float) and pd.isna(census_val)):
            return None
        return float(100.0 * float(census_val) / float(certified_beds))
    
    # Calculate metrics for each day
    period_df['Total_Nurse_Hours'] = (
        period_df['Hrs_RNDON'].fillna(0) +
        period_df['Hrs_RNadmin'].fillna(0) +
        period_df['Hrs_RN'].fillna(0) +
        period_df['Hrs_LPNadmin'].fillna(0) +
        period_df['Hrs_LPN'].fillna(0) +
        period_df['Hrs_CNA'].fillna(0) +
        period_df['Hrs_NAtrn'].fillna(0) +
        period_df['Hrs_MedAide'].fillna(0)
    )
    period_df['Direct_Care_Hours'] = (
        period_df['Hrs_RN'].fillna(0) +
        period_df['Hrs_LPN'].fillna(0) +
        period_df['Hrs_CNA'].fillna(0) +
        period_df['Hrs_NAtrn'].fillna(0) +
        period_df['Hrs_MedAide'].fillna(0)
    )
    period_df['Total_RN_Hours'] = (
        period_df['Hrs_RNDON'].fillna(0) +
        period_df['Hrs_RNadmin'].fillna(0) +
        period_df['Hrs_RN'].fillna(0)
    )
    period_df['Direct_RN_Hours'] = period_df['Hrs_RN'].fillna(0)
    
    period_df['Total_HPRD'] = period_df['Total_Nurse_Hours'] / period_df['MDScensus']
    period_df['Direct_Care_HPRD'] = period_df['Direct_Care_Hours'] / period_df['MDScensus']
    period_df['Total_RN_HPRD'] = period_df['Total_RN_Hours'] / period_df['MDScensus']
    period_df['Direct_RN_HPRD'] = period_df['Direct_RN_Hours'] / period_df['MDScensus']
    period_df['Direct_LPN_HPRD'] = period_df['Hrs_LPN'].fillna(0) / period_df['MDScensus']

    _c = period_df["MDScensus"]
    period_df["stack_rn_hprd"] = (
        period_df["Hrs_RNDON"].fillna(0)
        + period_df["Hrs_RNadmin"].fillna(0)
        + period_df["Hrs_RN"].fillna(0)
    ) / _c
    period_df["stack_lpn_hprd"] = (
        period_df["Hrs_LPNadmin"].fillna(0) + period_df["Hrs_LPN"].fillna(0)
    ) / _c
    period_df["stack_aide_hprd"] = (
        period_df["Hrs_CNA"].fillna(0)
        + period_df["Hrs_NAtrn"].fillna(0)
        + period_df["Hrs_MedAide"].fillna(0)
    ) / _c
    period_df["stack_direct_rn_hprd"] = period_df["Hrs_RN"].fillna(0) / _c
    period_df["stack_direct_lpn_hprd"] = period_df["Hrs_LPN"].fillna(0) / _c
    period_df["stack_direct_aide_hprd"] = (
        period_df["Hrs_CNA"].fillna(0)
        + period_df["Hrs_NAtrn"].fillna(0)
        + period_df["Hrs_MedAide"].fillna(0)
    ) / _c

    # Position-level stacks (sum to total nurse HPRD; direct chart uses RN/LPN direct + aide pieces)
    period_df["stack_cna_hprd"] = _h("Hrs_CNA") / _c
    period_df["stack_medaide_hprd"] = _h("Hrs_MedAide") / _c
    period_df["stack_natrn_hprd"] = _h("Hrs_NAtrn") / _c
    period_df["stack_lpn_admin_hprd"] = _h("Hrs_LPNadmin") / _c
    period_df["stack_lpn_direct_hprd"] = _h("Hrs_LPN") / _c
    period_df["stack_rn_don_hprd"] = _h("Hrs_RNDON") / _c
    period_df["stack_rn_admin_hprd"] = _h("Hrs_RNadmin") / _c
    period_df["stack_rn_floor_hprd"] = _h("Hrs_RN") / _c

    # Daily data (iterrows for clarity; includes position-level stack fields for charts)
    _stack_detail_keys = (
        "stack_cna_hprd",
        "stack_medaide_hprd",
        "stack_natrn_hprd",
        "stack_lpn_admin_hprd",
        "stack_lpn_direct_hprd",
        "stack_rn_don_hprd",
        "stack_rn_admin_hprd",
        "stack_rn_floor_hprd",
    )

    def _fz(v: Any) -> float:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0.0
        return float(v)

    daily_data: List[Dict[str, Any]] = []
    for rec in period_df.to_dict("records"):
        wd = rec["WorkDate"]
        c = rec["MDScensus"]
        daily_data.append(
            {
                "date": wd.strftime("%Y-%m-%d"),
                "label": wd.strftime("%b %d, %Y"),
                "census": float(c),
                "occupancy_pct": _occ_pct(c),
                "total_hprd": float(rec["Total_HPRD"]) if include_total_staffing else None,
                "direct_care_hprd": float(rec["Direct_Care_HPRD"]),
                "total_rn_hprd": float(rec["Total_RN_HPRD"]) if include_total_staffing else None,
                "direct_rn_hprd": float(rec["Direct_RN_HPRD"]),
                "direct_lpn_hprd": float(rec["Direct_LPN_HPRD"]),
                "stack_rn_hprd": _fz(rec.get("stack_rn_hprd")),
                "stack_lpn_hprd": _fz(rec.get("stack_lpn_hprd")),
                "stack_aide_hprd": _fz(rec.get("stack_aide_hprd")),
                "stack_direct_rn_hprd": _fz(rec.get("stack_direct_rn_hprd")),
                "stack_direct_lpn_hprd": _fz(rec.get("stack_direct_lpn_hprd")),
                "stack_direct_aide_hprd": _fz(rec.get("stack_direct_aide_hprd")),
                **{k: _fz(rec.get(k)) for k in _stack_detail_keys},
            }
        )
    
    # Monthly data (vectorized)
    period_df["year_month"] = period_df["WorkDate"].dt.to_period("M")
    _month_agg: Dict[str, str] = {
        "MDScensus": "mean",
        "Total_HPRD": "mean",
        "Direct_Care_HPRD": "mean",
        "Total_RN_HPRD": "mean",
        "Direct_RN_HPRD": "mean",
        "Direct_LPN_HPRD": "mean",
        "stack_rn_hprd": "mean",
        "stack_lpn_hprd": "mean",
        "stack_aide_hprd": "mean",
        "stack_direct_rn_hprd": "mean",
        "stack_direct_lpn_hprd": "mean",
        "stack_direct_aide_hprd": "mean",
    }
    for _k in _stack_detail_keys:
        _month_agg[_k] = "mean"
    monthly_agg = period_df.groupby("year_month").agg(_month_agg).reset_index()
    monthly_data: List[Dict[str, Any]] = []
    for row in monthly_agg.to_dict("records"):
        m: Dict[str, Any] = {
            "date": row["year_month"].to_timestamp().strftime("%Y-%m-%d"),
            "label": row["year_month"].to_timestamp().strftime("%b %Y"),
            "census": float(row["MDScensus"]) if pd.notna(row["MDScensus"]) else 0.0,
            "occupancy_pct": _occ_pct(row["MDScensus"]),
            "total_hprd": float(row["Total_HPRD"])
            if include_total_staffing and pd.notna(row["Total_HPRD"])
            else None,
            "direct_care_hprd": float(row["Direct_Care_HPRD"]) if pd.notna(row["Direct_Care_HPRD"]) else 0.0,
            "total_rn_hprd": float(row["Total_RN_HPRD"])
            if include_total_staffing and pd.notna(row["Total_RN_HPRD"])
            else None,
            "direct_rn_hprd": float(row["Direct_RN_HPRD"]) if pd.notna(row["Direct_RN_HPRD"]) else 0.0,
            "direct_lpn_hprd": float(row["Direct_LPN_HPRD"]) if pd.notna(row["Direct_LPN_HPRD"]) else 0.0,
            "stack_rn_hprd": float(row["stack_rn_hprd"]) if pd.notna(row["stack_rn_hprd"]) else 0.0,
            "stack_lpn_hprd": float(row["stack_lpn_hprd"]) if pd.notna(row["stack_lpn_hprd"]) else 0.0,
            "stack_aide_hprd": float(row["stack_aide_hprd"]) if pd.notna(row["stack_aide_hprd"]) else 0.0,
            "stack_direct_rn_hprd": float(row["stack_direct_rn_hprd"])
            if pd.notna(row["stack_direct_rn_hprd"])
            else 0.0,
            "stack_direct_lpn_hprd": float(row["stack_direct_lpn_hprd"])
            if pd.notna(row["stack_direct_lpn_hprd"])
            else 0.0,
            "stack_direct_aide_hprd": float(row["stack_direct_aide_hprd"])
            if pd.notna(row["stack_direct_aide_hprd"])
            else 0.0,
        }
        for _k in _stack_detail_keys:
            m[_k] = float(row[_k]) if pd.notna(row.get(_k)) else 0.0
        monthly_data.append(m)
    
    # Quarterly data (vectorized)
    _q_agg: Dict[str, str] = {
        "MDScensus": "mean",
        "Total_HPRD": "mean",
        "Direct_Care_HPRD": "mean",
        "Total_RN_HPRD": "mean",
        "Direct_RN_HPRD": "mean",
        "Direct_LPN_HPRD": "mean",
        "stack_rn_hprd": "mean",
        "stack_lpn_hprd": "mean",
        "stack_aide_hprd": "mean",
        "stack_direct_rn_hprd": "mean",
        "stack_direct_lpn_hprd": "mean",
        "stack_direct_aide_hprd": "mean",
    }
    for _k in _stack_detail_keys:
        _q_agg[_k] = "mean"
    quarterly_agg = period_df.groupby("CY_Qtr").agg(_q_agg).reset_index()
    quarterly_data: List[Dict[str, Any]] = []
    for row in quarterly_agg.to_dict("records"):
        q = str(row["CY_Qtr"])
        label = f"Q{q[5]} {q[:4]}" if len(q) == 6 and q[4] == "Q" else q
        qd: Dict[str, Any] = {
            "date": q,
            "label": label,
            "census": float(row["MDScensus"]),
            "occupancy_pct": _occ_pct(row["MDScensus"]),
            "total_hprd": float(row["Total_HPRD"]) if include_total_staffing else None,
            "direct_care_hprd": float(row["Direct_Care_HPRD"]),
            "total_rn_hprd": float(row["Total_RN_HPRD"]) if include_total_staffing else None,
            "direct_rn_hprd": float(row["Direct_RN_HPRD"]),
            "direct_lpn_hprd": float(row["Direct_LPN_HPRD"]) if pd.notna(row["Direct_LPN_HPRD"]) else 0.0,
            "stack_rn_hprd": float(row["stack_rn_hprd"]) if pd.notna(row["stack_rn_hprd"]) else 0.0,
            "stack_lpn_hprd": float(row["stack_lpn_hprd"]) if pd.notna(row["stack_lpn_hprd"]) else 0.0,
            "stack_aide_hprd": float(row["stack_aide_hprd"]) if pd.notna(row["stack_aide_hprd"]) else 0.0,
            "stack_direct_rn_hprd": float(row["stack_direct_rn_hprd"])
            if pd.notna(row["stack_direct_rn_hprd"])
            else 0.0,
            "stack_direct_lpn_hprd": float(row["stack_direct_lpn_hprd"])
            if pd.notna(row["stack_direct_lpn_hprd"])
            else 0.0,
            "stack_direct_aide_hprd": float(row["stack_direct_aide_hprd"])
            if pd.notna(row["stack_direct_aide_hprd"])
            else 0.0,
        }
        for _k in _stack_detail_keys:
            qd[_k] = float(row[_k]) if pd.notna(row.get(_k)) else 0.0
        quarterly_data.append(qd)
    
    return {
        'daily': daily_data,
        'monthly': monthly_data,
        'quarterly': quarterly_data
    }

def generate_citations_section(provnum: str) -> str:
    """Generate citations section for appendix."""
    citations = load_facility_citations(provnum, limit=10)
    
    if not citations:
        return ''
    
    citations_html = []
    for citation in citations:
        # Determine inspection type for URL
        complaint_deficiency = citation.get('complaint_deficiency', 'N')
        infection_control = citation.get('infection_control', 'N')
        
        # Determine URL type (priority: Infection Control > Complaint > Health)
        # Default is health-inspection
        inspection_type = 'health-inspection'
        if pd.notna(infection_control) and str(infection_control).upper() == 'Y':
            inspection_type = 'infection-control-inspection'
        elif pd.notna(complaint_deficiency) and str(complaint_deficiency).upper() == 'Y':
            inspection_type = 'complaint-inspection'
        
        # Generate link if we have a survey date URL
        survey_date_display = citation.get('survey_date', 'N/A')
        survey_date_url = citation.get('survey_date_url', None)
        
        if survey_date_url:
            # URL format: medicare.gov/care-compare/inspections/pdf/nursing-home/[CCN]/health/[inspection type]-inspection/?date=YYYY-MM-DD
            citation_link = f'https://www.medicare.gov/care-compare/inspections/pdf/nursing-home/{provnum}/health/{inspection_type}/?date={survey_date_url}'
            survey_date_cell = f'<td><a href="{citation_link}" target="_blank" style="color: #3498db; text-decoration: underline;">{survey_date_display}</a></td>'
        else:
            survey_date_cell = f'<td>{survey_date_display}</td>'
        
        citations_html.append(f'''
            <tr>
                {survey_date_cell}
                <td>{citation.get('deficiency_category', 'N/A')}</td>
                <td>{citation.get('deficiency_tag_number', 'N/A')}</td>
                <td>{citation.get('deficiency_description', 'N/A')}</td>
                <td>{citation.get('scope_severity_code', 'N/A')}</td>
            </tr>''')
    
    return f'''
    <div class="page-break"></div>
    <h2 style="color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 8px; margin-bottom: 20px; margin-top: 30px;">Recent Health Citations</h2>
    <p style="margin-bottom: 15px;">The table below lists this facility's most recent health citations from CMS survey data. Click a date to view the full report, or <a href="https://www.medicare.gov/care-compare/inspections/nursing-home/{provnum}/health" target="_blank" style="color: #3498db;">see the complete inspection history on Medicare Care Compare</a>.</p>
    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
        <thead>
            <tr style="background-color: #34495e; color: white;">
                <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Survey Date</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Deficiency Category</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Tag Number</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Deficiency Description</th>
                <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Scope/Severity</th>
            </tr>
        </thead>
        <tbody>
            {''.join(citations_html)}
        </tbody>
    </table>
    <p style="font-size: 9pt; font-style: italic; color: #7f8c8d;">Source: CMS Health Citations Data (NH_HealthCitations_Dec2025.csv)</p>
    '''


def generate_entity_section(provnum: str) -> str:
    """Generate entity information section for appendix."""
    entity_info = load_entity_info(provnum)
    
    if not entity_info:
        return ''
    
    # Build entity basics (compact, space-efficient)
    entity_id = entity_info.get('entity_id', '')
    entity_id_int = int(float(entity_id)) if entity_id and str(entity_id).replace('.', '').isdigit() else entity_id
    entity_name = format_facility_name(entity_info['entity_name']) if entity_info.get('entity_name') else f'Entity {entity_id_int}'
    
    # Line 1: Entity name, ID, dashboard link
    line1_parts = [f"<strong>{entity_name}</strong>"]
    if entity_id:
        line1_parts.append(f"ID: {entity_id_int}")
    if entity_id_int:
        line1_parts.append(f'<a href="https://www.pbj320.com/entity/{entity_id_int}" target="_blank" rel="noopener">View Entity Dashboard</a>')
    entity_line1 = ' · '.join(line1_parts)
    
    # Line 2: Quarters (and previous affiliations if any)
    current_quarters = entity_info.get('current_entity_quarters', [])
    quarters_str = _format_quarter_range(current_quarters) if current_quarters else ''
    entity_history = entity_info.get('entity_history', {})
    previous_entities_str = ''
    if len(entity_history) > 1:
        other_entities = [eid for eid in entity_history.keys() if eid != entity_info.get('current_entity_id')]
        if other_entities:
            other_entities_info = []
            entity_file = None
            for f in ['Nursing_Home_Affiliated_Entity_Performance_Measures_Jun_2025.csv', 'Nursing_Home_Affiliated_Entity_Performance_Measures_Mar_2025.csv']:
                if os.path.exists(f):
                    entity_file = f
                    break
            entity_df = pd.read_csv(entity_file, low_memory=False) if entity_file else None
            try:
                for other_eid in other_entities:
                    other_quarters = entity_history[other_eid].get('quarters', [])
                    if not other_quarters:
                        continue
                    other_entity_name = None
                    if entity_df is not None:
                        for col in ['Affiliated entity ID', 'Affiliated entity', 'Entity ID']:
                            if col not in entity_df.columns:
                                continue
                            try:
                                other_eid_float = float(other_eid)
                                other_entity_row = entity_df[pd.to_numeric(entity_df[col], errors='coerce') == other_eid_float]
                            except Exception:
                                other_entity_row = entity_df[entity_df[col].astype(str).str.strip() == other_eid]
                            if not other_entity_row.empty:
                                for name_col in ['Affiliated entity', 'Entity Name', 'entity_name']:
                                    if name_col in entity_df.columns:
                                        other_entity_name = str(other_entity_row.iloc[0].get(name_col, '')).strip()
                                        break
                            break
                    if not other_entity_name and os.path.exists('provider_info_combined.csv'):
                        try:
                            all_pi = pd.read_csv('provider_info_combined.csv', low_memory=False, dtype={'ccn': str})
                            for eid_col in ['affiliated_entity_id', 'Entity ID', 'entity_id']:
                                if eid_col not in all_pi.columns:
                                    continue
                                try:
                                    other_eid_float = float(other_eid)
                                    match = all_pi[pd.to_numeric(all_pi[eid_col], errors='coerce') == other_eid_float]
                                except Exception:
                                    match = all_pi[all_pi[eid_col].astype(str).str.strip() == other_eid]
                                if not match.empty:
                                    for name_col in ['affiliated_entity_name', 'Entity Name', 'entity_name', 'Chain Name', 'chain_name']:
                                        if name_col in all_pi.columns:
                                            v = match.iloc[0].get(name_col)
                                            if pd.notna(v) and str(v).strip():
                                                other_entity_name = str(v).strip()
                                                break
                                break
                        except Exception:
                            pass
                    other_eid_int = int(float(other_eid)) if other_eid and str(other_eid).replace('.', '').isdigit() else other_eid
                    q_range = _format_quarter_range(other_quarters)
                    other_entities_info.append(f"{format_facility_name(other_entity_name)} (ID: {other_eid_int}, {q_range})" if other_entity_name else f"Entity {other_eid_int} ({q_range})")
                if other_entities_info:
                    previous_entities_str = ' <strong>Previous:</strong> ' + '; '.join(other_entities_info)
            except Exception:
                previous_entities_str = ''
    
    entity_line2 = ''
    if quarters_str:
        entity_line2 = f"<strong>Quarters:</strong> {quarters_str}{previous_entities_str}"
    elif previous_entities_str:
        entity_line2 = previous_entities_str.strip()
    
    # Line 3: Facilities, states, SFF, ratings (all ratings with 1 decimal)
    def _rating_str(val):
        if val is None or (isinstance(val, float) and pd.isna(val)) or val == '':
            return None
        try:
            return f"{float(val):.1f}"
        except (TypeError, ValueError):
            return None
    stats_parts = []
    if pd.notna(entity_info.get('num_facilities')) and entity_info.get('num_facilities'):
        stats_parts.append(f"<strong>Facilities:</strong> {entity_info['num_facilities']}")
    if pd.notna(entity_info.get('num_states')) and entity_info.get('num_states'):
        stats_parts.append(f"<strong>States:</strong> {entity_info['num_states']}")
    if pd.notna(entity_info.get('num_sff')) and entity_info.get('num_sff'):
        stats_parts.append(f"<strong>SFF:</strong> {entity_info['num_sff']}")
    rating_parts = []
    for label, key in [('Overall', 'avg_overall_rating'), ('Staffing', 'avg_staffing_rating'), ('Health', 'avg_health_rating'), ('Quality', 'avg_quality_rating')]:
        r = _rating_str(entity_info.get(key))
        if r is not None:
            rating_parts.append(f"{label}: {r}")
    if rating_parts:
        stats_parts.append(f"<strong>Avg. ratings:</strong> " + ", ".join(rating_parts) + " stars")
    entity_line3 = ' · '.join(stats_parts) if stats_parts else ''
    
    # Build facilities table with additional columns
    facilities_html = []
    for facility in entity_info.get('facilities', [])[:50]:  # Limit to 50 facilities
        ccn = facility.get('ccn', 'N/A')
        name = format_facility_name(facility.get('name', 'N/A'))
        state = facility.get('state', 'N/A')
        red_flags = facility.get('red_flags', 'None')
        staffing_hprd = facility.get('staffing_hprd')
        staffing_str = f"{staffing_hprd:.2f}" if staffing_hprd and pd.notna(staffing_hprd) else 'N/A'
        census = facility.get('census')
        census_str = f"{census:.0f}" if census and pd.notna(census) else 'N/A'
        rating = facility.get('rating')
        rating_str = f"{int(rating)}" if rating and pd.notna(rating) else 'N/A'
        
        facilities_html.append(f'''
            <tr>
                <td style="padding: 8px; border: 1px solid #bdc3c7;">{ccn}</td>
                <td style="padding: 8px; border: 1px solid #bdc3c7;"><a href="https://www.pbj320.com/provider/{ccn}" target="_blank" style="color: #3498db; text-decoration: none;">{name}</a></td>
                <td style="padding: 8px; border: 1px solid #bdc3c7;">{state}</td>
                <td style="padding: 8px; border: 1px solid #bdc3c7;">{red_flags}</td>
                <td style="padding: 8px; border: 1px solid #bdc3c7;">{staffing_str}</td>
                <td style="padding: 8px; border: 1px solid #bdc3c7;">{census_str}</td>
                <td style="padding: 8px; border: 1px solid #bdc3c7;">{rating_str}</td>
            </tr>''')
    
    if not facilities_html:
        facilities_table = '<p><em>No facilities found for this entity.</em></p>'
    else:
        facilities_table = f'''
        <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 9pt;">
            <thead>
                <tr style="background-color: #34495e; color: white;">
                    <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">CCN</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Facility Name</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">State</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Red Flags</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Staffing HPRD</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Census</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid #2c3e50;">Rating</th>
                </tr>
            </thead>
            <tbody>
                {''.join(facilities_html)}
            </tbody>
        </table>
        {f'<p style="font-size: 9pt; font-style: italic; color: #7f8c8d;">Showing first 50 facilities. Total: {len(entity_info.get("facilities", []))} facilities.</p>' if len(entity_info.get('facilities', [])) > 50 else ''}
        '''
    
    learn_more_html = ''
    if entity_id_int:
        learn_more_html = f'<p style="font-size: 10pt; margin-top: 12px;">Learn more about <strong>{entity_name}</strong> on <a href="https://www.pbj320.com/entity/{entity_id_int}" target="_blank" rel="noopener">PBJ320</a> or <a href="https://www.medicare.gov/care-compare/details/chains/{entity_id_int}/" target="_blank" rel="noopener">CMS Care Compare</a>.</p>'
    
    entity_block = '<p style="margin: 4px 0; font-size: 10pt; line-height: 1.5;">' + entity_line1 + '</p>'
    if entity_line2:
        entity_block += '<p style="margin: 4px 0; font-size: 10pt; line-height: 1.5;">' + entity_line2 + '</p>'
    if entity_line3:
        entity_block += '<p style="margin: 4px 0; font-size: 10pt; line-height: 1.5;">' + entity_line3 + '</p>'
    
    return f'''
    <div class="page-break"></div>
    <h2 style="color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 8px; margin-bottom: 20px; margin-top: 30px;">Entity Information</h2>
    <div style="margin-bottom: 16px;">
        {entity_block}
    </div>
    <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Facilities in Entity</h3>
    {facilities_table}
    {learn_more_html}
    <p style="font-size: 9pt; font-style: italic; color: #7f8c8d;">Source: CMS Affiliated Entity Performance Measures (latest available data)</p>
    '''


def _default_include_sections() -> Dict[str, bool]:
    return {
        'key_dates': True, 'date_ranges_of_interest': True, 'quarterly_staffing': True, 'case_mix': True, 'red_flags': True,
        'period_summary': True, 'state_compliance': True, 'daily_staffing_table': True, 'appendix': True
    }


def _format_previous_names_line(previous_names: Optional[Sequence]) -> str:
    """Format 'Previously: Name (years), ...' for list of strings or dicts with name/years."""
    if not previous_names or len(previous_names) == 0:
        return ''
    parts = []
    for n in previous_names:
        if isinstance(n, dict):
            name = format_facility_name(n.get('name', ''))
            years = n.get('years')
            parts.append(f"{name} ({years})" if years else name)
        else:
            parts.append(format_facility_name(n))
    return f'<p style="font-size: 9pt; color: #666; margin-top: 2px; margin-bottom: 4px;">Previously: ' + ', '.join(parts) + '</p>'


def generate_attorney_report(
    provnum: str,
    facility_name: str,
    city: str,
    state: str,
    start_date: datetime,
    end_date: datetime,
    key_dates: List[datetime],
    quarterly_data: Dict,
    state_comparisons: Dict,
    date_ranges_of_interest: Optional[List[Dict]] = None,
    period_metrics: Optional[Dict] = None,
    daily_staffing: Optional[List[Dict]] = None,
    macpac_standards: Optional[Dict] = None,
    red_flags_history: Optional[List[Dict]] = None,
    case_mix_data: Optional[List[Dict]] = None,
    pbj_df: Optional[pd.DataFrame] = None,
    include_total_staffing: bool = True,
    direct_first: bool = False,
    watermark: bool = False,
    include_sections: Optional[Dict[str, bool]] = None,
    previous_names: Optional[Sequence[Union[str, Dict[str, str]]]] = None,
) -> str:
    """Generate HTML report using attorney report styling with enhanced branding."""
    inc = include_sections if include_sections is not None else _default_include_sections()
    for k in _default_include_sections():
        if k not in inc:
            inc[k] = True
    
    # Generate quarters list
    quarters = sorted(quarterly_data.keys())
    
    # ------------------------------------------------------------------
    # Attach % Case-Mix metrics to period_metrics and quarterly_data
    # so the Resident Stay Period table can show non-N/A values.
    # ------------------------------------------------------------------
    pct_case_mix_by_quarter: Dict[str, Dict[str, Optional[float]]] = {}
    if case_mix_data:
        # Group case-mix entries by quarter
        quarter_dict: Dict[str, List[Dict]] = {}
        for entry in case_mix_data:
            q = entry.get('quarter')
            if not q:
                continue
            quarter_dict.setdefault(q, []).append(entry)

        for quarter, entries in quarter_dict.items():
            # Averages from provider info (use case_mix_total for Direct % denominator - same as dashboard)
            case_mix_totals_q: List[float] = [float(x) for e in entries for x in (e.get('case_mix_total'),) if x is not None and not pd.isna(x)]
            case_mix_rns: List[float] = [float(x) for e in entries for x in (e.get('case_mix_rn'),) if x is not None and not pd.isna(x)]
            avg_case_mix_total_q = sum(case_mix_totals_q) / len(case_mix_totals_q) if case_mix_totals_q else None
            avg_case_mix_rn = sum(case_mix_rns) / len(case_mix_rns) if case_mix_rns else None

            # Reported HPRD: use raw direct from PBJ for % (match dashboard); rounded for display fallback
            avg_reported_direct: Optional[float] = None
            raw_direct_hprd_q: Optional[float] = None
            raw_direct_rn_hprd_q: Optional[float] = None
            avg_reported_direct_rn: Optional[float] = None

            if pbj_df is not None and not pbj_df.empty:
                try:
                    raw_direct_hprd_q = _quarterly_direct_care_hprd_raw(pbj_df, quarter)
                    avg_reported_direct = calculate_quarterly_direct_care_hprd(pbj_df, quarter)
                except Exception:
                    pass
                try:
                    raw_direct_rn_hprd_q = _quarterly_direct_rn_hprd_raw(pbj_df, quarter)
                    avg_reported_direct_rn = round_half_up(raw_direct_rn_hprd_q, 2) if raw_direct_rn_hprd_q is not None else None
                except Exception:
                    pass
                if avg_reported_direct_rn is None:
                    try:
                        quarter_data = pbj_df[pbj_df['CY_Qtr'] == quarter].copy()
                        if not quarter_data.empty:
                            total_resident_days = quarter_data['MDScensus'].sum()
                            if total_resident_days > 0:
                                direct_care_rn_hours = quarter_data['Hrs_RN'].fillna(0).sum()
                                avg_reported_direct_rn = round_half_up(direct_care_rn_hours / total_resident_days, 2)
                    except Exception:
                        avg_reported_direct_rn = None

            if avg_reported_direct is None:
                reported_directs: List[float] = [float(x) for e in entries for x in (e.get('reported_direct'),) if x is not None and not pd.isna(x)]
                avg_reported_direct = sum(reported_directs) / len(reported_directs) if reported_directs else None
            if avg_reported_direct_rn is None:
                reported_rns: List[float] = [float(x) for e in entries for x in (e.get('reported_rn'),) if x is not None and not pd.isna(x)]
                avg_reported_direct_rn = sum(reported_rns) / len(reported_rns) if reported_rns else None

            # Direct %: numerator = raw direct HPRD (or rounded fallback), denominator = case_mix_total (same as dashboard)
            # RN %: use raw RN HPRD when available so attorney report matches dashboard (e.g. 48.2% not 48.0%)
            pct_direct: Optional[float] = None
            pct_rn: Optional[float] = None
            numer_direct = raw_direct_hprd_q if raw_direct_hprd_q is not None else avg_reported_direct
            if avg_case_mix_total_q is not None and numer_direct is not None and avg_case_mix_total_q > 0:
                pct_direct = round_half_up(numer_direct / avg_case_mix_total_q * 100, 1)
            numer_rn = raw_direct_rn_hprd_q if raw_direct_rn_hprd_q is not None else avg_reported_direct_rn
            if avg_case_mix_rn is not None and numer_rn is not None and avg_case_mix_rn > 0:
                pct_rn = round_half_up(numer_rn / avg_case_mix_rn * 100, 1)

            if pct_direct is not None or pct_rn is not None:
                pct_case_mix_by_quarter[quarter] = {
                    'direct': pct_direct,
                    'rn': pct_rn,
                }

        # Attach to quarterly_data for use in Resident Stay Period table
        for quarter, pct_vals in pct_case_mix_by_quarter.items():
            q_metrics = quarterly_data.get(quarter)
            if not q_metrics:
                continue
            if pct_vals.get('direct') is not None:
                q_metrics['pct_case_mix_direct'] = pct_vals['direct']
            if pct_vals.get('rn') is not None:
                q_metrics['pct_case_mix_rn'] = pct_vals['rn']

        # Attach period-level % Case-Mix (use simple average of quarters that have data)
        if period_metrics:
            directs = [v['direct'] for v in pct_case_mix_by_quarter.values() if v.get('direct') is not None]
            rns = [v['rn'] for v in pct_case_mix_by_quarter.values() if v.get('rn') is not None]
            directs = [float(x) for x in directs if x is not None]
            rns = [float(x) for x in rns if x is not None]
            if directs:
                period_metrics['pct_case_mix_direct'] = sum(directs) / len(directs)
            if rns:
                period_metrics['pct_case_mix_rn'] = sum(rns) / len(rns)

    # Format dates
    start_date_str = start_date.strftime('%B %d, %Y')
    end_date_str = end_date.strftime('%B %d, %Y')
    review_period = f"{start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')}"
    
    # Calculate summary metrics from daily data if available, otherwise from quarterly
    if daily_staffing and len(daily_staffing) > 0:
        total_hprd_values = [d['total_hprd'] for d in daily_staffing]
        rn_hprd_values = [d['rn_hprd'] for d in daily_staffing]
        min_total_hprd = min(total_hprd_values)
        max_total_hprd = max(total_hprd_values)
        min_rn_hprd = min(rn_hprd_values)
        max_rn_hprd = max(rn_hprd_values)
    else:
        total_hprd_values = [q['total_hprd'] for q in quarterly_data.values() if q]
        rn_hprd_values = [q['rn_hprd'] for q in quarterly_data.values() if q]
        min_total_hprd = min(total_hprd_values) if total_hprd_values else 0
        max_total_hprd = max(total_hprd_values) if total_hprd_values else 0
        min_rn_hprd = min(rn_hprd_values) if rn_hprd_values else 0
        max_rn_hprd = max(rn_hprd_values) if rn_hprd_values else 0
    
    # Format key dates with staffing info (prioritizing DIRECT)
    key_dates_str = []
    if key_dates:  # Only process if key_dates is provided
        if daily_staffing:
            # Create a lookup dict for daily staffing by date
            daily_staffing_dict = {}
            for day in daily_staffing:
                day_date = day['date'].date() if isinstance(day['date'], datetime) else day['date']
                daily_staffing_dict[day_date] = day
            
            for date in key_dates:
                date_only = date.date() if isinstance(date, datetime) else date
                date_str = date.strftime('%B %d, %Y')
                
                day_data = daily_staffing_dict.get(date_only)
                if day_data:
                    # Prioritize DIRECT care metrics
                    direct_care_hprd = day_data.get('direct_care_hprd', 0)
                    direct_care_rn_hprd = day_data.get('direct_care_rn_hprd', 0)
                    total_hprd = day_data.get('total_hprd', 0)
                    rn_hprd = day_data.get('rn_hprd', 0)
                    cna_hprd = day_data.get('cna_hprd', 0)
                    census = day_data.get('census', 0)
                    
                    # Get day of week
                    day_of_week = date.strftime('%A')
                    date_with_day = f"{date_str} ({day_of_week})"
                    
                    if include_total_staffing:
                        key_dates_str.append(f"<li><strong>{date_with_day}:</strong> Census: {census:.0f} | Direct Care HPRD: <strong>{direct_care_hprd:.2f}</strong> | Direct RN HPRD: <strong>{direct_care_rn_hprd:.2f}</strong> | Total HPRD: {total_hprd:.2f} | Total RN HPRD: {rn_hprd:.2f}</li>")
                    else:
                        key_dates_str.append(f"<li><strong>{date_with_day}:</strong> Census: {census:.0f} | Direct Care HPRD: <strong>{direct_care_hprd:.2f}</strong> | Direct RN HPRD: <strong>{direct_care_rn_hprd:.2f}</strong></li>")
                else:
                    day_of_week = date.strftime('%A')
                    date_with_day = f"{date_str} ({day_of_week})"
                    key_dates_str.append(f"<li>{date_with_day}: (no data available)</li>")
        else:
            # No daily staffing data, but we have key dates - just list them
            for date in key_dates:
                date_str = date.strftime('%B %d, %Y (%A)')
                key_dates_str.append(f"<li>{date_str}</li>")
    
    key_dates_html = "\n".join(key_dates_str) if key_dates_str else ""

    # Date ranges of interest (primary / secondary) with resident count per range
    date_ranges_of_interest = date_ranges_of_interest or []
    primary_ranges = [r for r in date_ranges_of_interest if (r.get('role') or 'primary').lower() == 'primary']
    secondary_ranges = [r for r in date_ranges_of_interest if (r.get('role') or '').lower() == 'secondary']

    def _metrics_for_range(r):
        if pbj_df is None or pbj_df.empty:
            return None
        start_d = r.get('start') or r.get('start_date')
        end_d = r.get('end') or r.get('end_date')
        if not hasattr(start_d, 'strftime') or not hasattr(end_d, 'strftime'):
            return None
        return calculate_period_metrics(pbj_df, start_d, end_d)

    def _fmt_range(r):
        s = r.get('start') or r.get('start_date')
        e = r.get('end') or r.get('end_date')
        if hasattr(s, 'strftime'):
            s = s.strftime('%B %d, %Y')
        if hasattr(e, 'strftime'):
            e = e.strftime('%B %d, %Y')
        text = f"{s} – {e}"
        pm = _metrics_for_range(r)
        if pm:
            # Consistent labels: Census; Direct care HPRD; RN direct HPRD (match report terminology)
            parts = [f"Census: {pm.get('avg_census', 0):.0f}"]
            if pm.get('direct_care_hprd') is not None:
                parts.append(f"Direct care HPRD: {pm['direct_care_hprd']:.2f}")
            if pm.get('direct_care_rn_hprd') is not None:
                parts.append(f"RN direct HPRD: {pm['direct_care_rn_hprd']:.2f}")
            text += " — " + "; ".join(parts)
        return text

    primary_ul = "<ul>" + "".join(f"<li>{_fmt_range(r)}</li>" for r in primary_ranges) + "</ul>" if primary_ranges else ""
    secondary_ul = "<ul>" + "".join(f"<li>{_fmt_range(r)}</li>" for r in secondary_ranges) + "</ul>" if secondary_ranges else ""
    date_ranges_html = ""
    if inc.get('date_ranges_of_interest', True) and (primary_ranges or secondary_ranges):
        date_ranges_html = '<h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Date Ranges of Interest</h3>'
        if primary_ranges:
            date_ranges_html += '<p style="margin-bottom: 6px;"><strong>Primary</strong></p>' + primary_ul
        if secondary_ranges:
            date_ranges_html += '<p style="margin-bottom: 6px; margin-top: 12px;"><strong>Secondary</strong></p>' + secondary_ul

    # Generate watermark section if requested
    watermark_section = ""
    if watermark:
        watermark_section = """
    <div style="background-color: #fff3cd; border: 2px solid #ffc107; padding: 15px; margin-bottom: 20px; border-radius: 4px;">
        <h4 style="color: #856404; margin-top: 0;">SAMPLE REPORT - FOR ILLUSTRATIVE PURPOSES ONLY</h4>
        <p style="color: #856404; font-size: 0.9em;">This report is a preliminary sample and should not be relied upon for any legal, financial, or operational decisions. It is provided for discussion and demonstration purposes only.</p>
    </div>
    """
    
    # Generate time series data for charts only when longitudinal section is included
    time_series_data = (
        generate_time_series_data(pbj_df, start_date, end_date, include_total_staffing)
        if (pbj_df is not None and inc.get('quarterly_staffing', True))
        else {'daily': [], 'monthly': [], 'quarterly': []}
    )
    import json
    # Convert NaN/None to null for JSON
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(item) for item in obj]
        elif isinstance(obj, float) and (pd.isna(obj) or obj != obj):
            return None
        return obj
    time_series_clean = clean_for_json(time_series_data)
    time_series_json = json.dumps(time_series_clean)
    state_minimum_value = macpac_standards.get('min_staffing', 0.0) if macpac_standards else 0.0
    
    # Generate quarterly table rows with visual cues
    quarterly_rows = []
    for quarter in quarters:
        q_data = quarterly_data.get(quarter)
        state_data = state_comparisons.get(quarter)
        
        if not q_data:
            continue
        
        # Format comparisons
        state_total = state_data['total_hprd'] if state_data and isinstance(state_data.get('total_hprd'), (int, float)) else None
        state_rn = state_data['rn_hprd'] if state_data and isinstance(state_data.get('rn_hprd'), (int, float)) else None
        state_direct_care = state_data.get('direct_care_hprd') if state_data and isinstance(state_data.get('direct_care_hprd'), (int, float)) else None
        state_direct_care_rn = state_data.get('direct_care_rn_hprd') if state_data and isinstance(state_data.get('direct_care_rn_hprd'), (int, float)) else None
        
        # Determine if below state average (subtle visual cue)
        total_hprd_class = ""
        rn_hprd_class = ""
        direct_care_class = ""
        direct_care_rn_class = ""
        if state_total is not None and q_data['total_hprd'] < state_total:
            total_hprd_class = 'class="below-state-avg"'
        if state_rn is not None and q_data['rn_hprd'] < state_rn:
            rn_hprd_class = 'class="below-state-avg"'
        if state_direct_care is not None and q_data.get('direct_care_hprd', 0) < state_direct_care:
            direct_care_class = 'class="below-state-avg"'
        if state_direct_care_rn is not None and q_data.get('direct_care_rn_hprd', 0) < state_direct_care_rn:
            direct_care_rn_class = 'class="below-state-avg"'
        
        # Format quarter for display
        quarter_display = format_quarter_display(quarter)
        state_total_display = f"{state_total:.2f}" if state_total is not None else 'N/A'
        state_rn_display = f"{state_rn:.2f}" if state_rn is not None else 'N/A'
        state_direct_care_display = f"{state_direct_care:.2f}" if state_direct_care is not None else 'N/A'
        state_direct_care_rn_display = f"{state_direct_care_rn:.2f}" if state_direct_care_rn is not None else 'N/A'
        facility_direct_care = q_data.get('direct_care_hprd', 0)
        facility_direct_care_rn = q_data.get('direct_care_rn_hprd', 0)
        
        # Build quarterly row conditionally - column order by direct_first
        row_cells = [f"<td><strong>{quarter_display}</strong></td>", f"<td>{q_data['avg_census']:.1f}</td>"]
        total_hprd_cells = [f"<td {total_hprd_class}>{q_data['total_hprd']:.2f}</td>", f"<td>{state_total_display}</td>"] if include_total_staffing else []
        direct_hprd_cells = [f"<td {direct_care_class}>{facility_direct_care:.2f}</td>", f"<td>{state_direct_care_display}</td>"]
        total_rn_cells = [f"<td {rn_hprd_class}>{q_data['rn_hprd']:.2f}</td>", f"<td>{state_rn_display}</td>"] if include_total_staffing else []
        direct_rn_cells = [f"<td {direct_care_rn_class}>{facility_direct_care_rn:.2f}</td>", f"<td>{state_direct_care_rn_display}</td>"]

        # % Case-Mix cells (Direct / RN) – use values from quarterly data if present, otherwise N/A
        pct_case_mix_direct = q_data.get('pct_case_mix_direct')
        pct_case_mix_rn = q_data.get('pct_case_mix_rn')
        direct_case_mix_cell = f"<td>{pct_case_mix_direct:.1f}%</td>" if isinstance(pct_case_mix_direct, (int, float)) else "<td>N/A</td>"
        rn_case_mix_cell = f"<td>{pct_case_mix_rn:.1f}%</td>" if isinstance(pct_case_mix_rn, (int, float)) else "<td>N/A</td>"
        pct_case_mix_cells = [direct_case_mix_cell, rn_case_mix_cell]

        if direct_first:
            row_cells.extend(direct_hprd_cells)
            row_cells.extend(total_hprd_cells)
            row_cells.extend(direct_rn_cells)
            row_cells.extend(total_rn_cells)
        else:
            row_cells.extend(total_hprd_cells)
            row_cells.extend(direct_hprd_cells)
            row_cells.extend(total_rn_cells)
            row_cells.extend(direct_rn_cells)
        # Add % Case-Mix (Direct / RN) before Contract %
        row_cells.extend(pct_case_mix_cells)
        row_cells.append(f"<td>{q_data['contract_pct']:.1f}%</td>")
        
        quarterly_rows.append(f"""
        <tr>
            {''.join(row_cells)}
        </tr>
        """)
    
    quarterly_table = "\n".join(quarterly_rows)
    
    # Appendix quarterly table: column order by direct_first
    _th_total = f'<th colspan="2">Total HPRD</th>' if include_total_staffing else ''
    _th_direct = '<th colspan="2">Direct Care HPRD</th>'
    _th_total_rn = f'<th colspan="2">Total RN HPRD</th>' if include_total_staffing else ''
    _th_direct_rn = '<th colspan="2">Direct RN HPRD</th>'
    # New third section: % Case-Mix with Direct / RN columns
    _th_pct_case_mix = '<th colspan="2">% Case-Mix</th>'

    _sub_total = f'<th>Facility</th><th>{state}</th>' if include_total_staffing else ''
    _sub_direct = f'<th>Facility</th><th>{state}</th>'
    _sub_total_rn = f'<th>Facility</th><th>{state}</th>' if include_total_staffing else ''
    _sub_direct_rn = f'<th>Facility</th><th>{state}</th>'
    _sub_pct_case_mix = '<th>Direct</th><th>RN</th>'

    if direct_first:
        appendix_thead_col1 = _th_direct + _th_total + _th_direct_rn + _th_total_rn + _th_pct_case_mix
        appendix_thead_col2 = _sub_direct + _sub_total + _sub_direct_rn + _sub_total_rn + _sub_pct_case_mix
    else:
        appendix_thead_col1 = _th_total + _th_direct + _th_total_rn + _th_direct_rn + _th_pct_case_mix
        appendix_thead_col2 = _sub_total + _sub_direct + _sub_total_rn + _sub_direct_rn + _sub_pct_case_mix

    _pd_total = f'<td>{period_metrics.get("total_hprd", 0):.2f}</td><td>N/A</td>' if include_total_staffing and period_metrics else ''
    _pd_direct = f'<td>{period_metrics.get("direct_care_hprd", 0):.2f}</td><td>N/A</td>' if period_metrics else ''
    _pd_total_rn = f'<td>{period_metrics.get("rn_hprd", 0):.2f}</td><td>N/A</td>' if include_total_staffing and period_metrics else ''
    _pd_direct_rn = f'<td>{period_metrics.get("direct_care_rn_hprd", 0):.2f}</td><td>N/A</td>' if period_metrics else ''

    # Period-level % Case-Mix cells, if metrics are available; otherwise N/A
    if period_metrics and 'pct_case_mix_direct' in period_metrics:
        _pd_pct_case_mix_direct = f'<td>{period_metrics.get("pct_case_mix_direct", 0):.1f}%</td>'
    else:
        _pd_pct_case_mix_direct = '<td>N/A</td>'
    if period_metrics and 'pct_case_mix_rn' in period_metrics:
        _pd_pct_case_mix_rn = f'<td>{period_metrics.get("pct_case_mix_rn", 0):.1f}%</td>'
    else:
        _pd_pct_case_mix_rn = '<td>N/A</td>'
    _pd_pct_case_mix = _pd_pct_case_mix_direct + _pd_pct_case_mix_rn

    if direct_first:
        appendix_period_cells = _pd_direct + _pd_total + _pd_direct_rn + _pd_total_rn + _pd_pct_case_mix
    else:
        appendix_period_cells = _pd_total + _pd_direct + _pd_total_rn + _pd_direct_rn + _pd_pct_case_mix
    
    # Period metrics section (if available)
    period_metrics_html = ""
    if period_metrics:
        # Get state requirement text and minimum value
        state_req_text = "Check state regulations"
        state_minimum = 0.0
        if macpac_standards:
            state_req_text = macpac_standards['display_text']
            state_minimum = macpac_standards.get('min_staffing', 0.0)
        
        # Calculate number of days in period
        days_in_period = (end_date - start_date).days + 1
        
        # Check if below state minimum for visual cue
        total_hprd_below = period_metrics['total_hprd'] < state_minimum if state_minimum > 0 else False
        direct_hprd_below = period_metrics.get('direct_care_hprd', 0) < state_minimum if state_minimum > 0 else False
        total_hprd_class = 'class="below-state-min"' if total_hprd_below else ''
        direct_hprd_class = 'class="below-state-min"' if direct_hprd_below else ''
        
        # Build period metrics table rows conditionally
        period_rows = []
        if include_total_staffing:
            period_rows.append(f"""
            <tr>
                <td><strong>Total HPRD</strong></td>
                <td {total_hprd_class}><strong>{period_metrics['total_hprd']:.2f}</strong></td>
                <td>{state_req_text}</td>
            </tr>""")
        period_rows.append(f"""
            <tr>
                <td><strong>Direct Care HPRD</strong></td>
                <td {direct_hprd_class}><strong>{period_metrics.get('direct_care_hprd', 0):.2f}</strong></td>
                <td>N/A</td>
            </tr>""")
        if include_total_staffing:
            period_rows.append(f"""
            <tr>
                <td>Total RN HPRD (incl. Admin/DON)</td>
                <td>{period_metrics['rn_hprd']:.2f}</td>
                <td>N/A</td>
            </tr>""")
        period_rows.append(f"""
            <tr>
                <td>Direct care RN HPRD (excl. Admin/DON)</td>
                <td>{period_metrics['direct_care_rn_hprd']:.2f}</td>
                <td>N/A</td>
            </tr>""")
        
        # Build simplified period metrics for executive summary
        # Get case-mix findings if available
        case_mix_summary = ""
        harrington_summary = ""
        if case_mix_data:
            # Get the most recent quarter's case-mix data
            latest_quarter = max([entry.get('quarter', '') for entry in case_mix_data if entry.get('quarter')], default='')
            if latest_quarter:
                # Get all entries for this quarter
                quarter_entries = [e for e in case_mix_data if e.get('quarter') == latest_quarter]
                if quarter_entries:
                    # Use case_mix_total for Direct % denominator (same as dashboard), not combined RN+LPN+NA
                    case_mix_totals_latest = [e['case_mix_total'] for e in quarter_entries if e.get('case_mix_total') is not None]
                    case_mix_rns = [e['case_mix_rn'] for e in quarter_entries if e.get('case_mix_rn') is not None]
                    cmi_values = [e['cmi'] for e in quarter_entries if e.get('cmi') is not None]
                    avg_cmi = sum(cmi_values) / len(cmi_values) if cmi_values else None
                    avg_case_mix_total_latest = sum(case_mix_totals_latest) / len(case_mix_totals_latest) if case_mix_totals_latest else None
                    avg_case_mix_rn = sum(case_mix_rns) / len(case_mix_rns) if case_mix_rns else None
                    
                    raw_direct_latest = _quarterly_direct_care_hprd_raw(pbj_df, latest_quarter) if pbj_df is not None else None
                    avg_reported_direct = calculate_quarterly_direct_care_hprd(pbj_df, latest_quarter) if pbj_df is not None else None
                    avg_reported_direct_rn = None
                    if pbj_df is not None:
                        quarter_data = pbj_df[pbj_df['CY_Qtr'] == latest_quarter].copy()
                        if not quarter_data.empty:
                            total_resident_days = quarter_data['MDScensus'].sum()
                            if total_resident_days > 0:
                                direct_care_rn_hours = quarter_data['Hrs_RN'].fillna(0).sum()
                                avg_reported_direct_rn = round_half_up(direct_care_rn_hours / total_resident_days, 2)
                    if avg_reported_direct is None:
                        reported_directs = [e['reported_direct'] for e in quarter_entries if e.get('reported_direct') is not None]
                        avg_reported_direct = sum(reported_directs) / len(reported_directs) if reported_directs else None
                    if avg_reported_direct_rn is None:
                        reported_rns = [e['reported_rn'] for e in quarter_entries if e.get('reported_rn') is not None]
                        avg_reported_direct_rn = sum(reported_rns) / len(reported_rns) if reported_rns else None
                    
                    pct_prop_direct = None
                    pct_prop_direct_rn = None
                    numer_direct = raw_direct_latest if raw_direct_latest is not None else avg_reported_direct
                    if avg_case_mix_total_latest is not None and numer_direct is not None and avg_case_mix_total_latest > 0:
                        pct_prop_direct = (numer_direct / avg_case_mix_total_latest * 100)
                    if avg_case_mix_rn is not None and avg_reported_direct_rn is not None and avg_case_mix_rn > 0:
                        pct_prop_direct_rn = (avg_reported_direct_rn / avg_case_mix_rn * 100)
                    
                    if pct_prop_direct is not None and pct_prop_direct_rn is not None:
                        case_mix_summary = f"<li><strong>{pct_prop_direct:.1f}%</strong> Case-Mix; <strong>{pct_prop_direct_rn:.1f}%</strong> RN Case-Mix</li>"
                    elif pct_prop_direct is not None:
                        case_mix_summary = f"<li><strong>{pct_prop_direct:.1f}%</strong> Case-Mix</li>"
                    
                    # Calculate Harrington-adjusted percentages
                    if avg_cmi is not None:
                        harrington_total = calculate_harrington_adjusted_hprd(avg_cmi, 'total')
                        harrington_rn = calculate_harrington_adjusted_hprd(avg_cmi, 'rn')
                        
                        pct_prop_harrington_total = None
                        pct_prop_harrington_rn = None
                        if harrington_total is not None and avg_reported_direct is not None and harrington_total > 0:
                            pct_prop_harrington_total = (avg_reported_direct / harrington_total * 100)
                        if harrington_rn is not None and avg_reported_direct_rn is not None and harrington_rn > 0:
                            pct_prop_harrington_rn = (avg_reported_direct_rn / harrington_rn * 100)
                        
                        if pct_prop_harrington_total is not None and pct_prop_harrington_rn is not None:
                            harrington_summary = f"<li><strong>{pct_prop_harrington_total:.1f}%</strong> Harrington Expected; <strong>{pct_prop_harrington_rn:.1f}%</strong> Harrington RN Expected</li>"
                        elif pct_prop_harrington_total is not None:
                            harrington_summary = f"<li><strong>{pct_prop_harrington_total:.1f}%</strong> Harrington Expected</li>"
        
        # Build staffing metrics summary with total nurse included; include full-period resident count
        staffing_metrics_html = ""
        if period_metrics:
            total_hprd = period_metrics.get('total_hprd', 0)
            direct_hprd = period_metrics.get('direct_care_hprd', 0)
            total_rn_hprd = period_metrics.get('rn_hprd', 0) if include_total_staffing else None
            direct_rn_hprd = period_metrics.get('direct_care_rn_hprd', 0)
            avg_census = period_metrics.get('avg_census')
            period_note = f' (avg. daily census {avg_census:.0f})' if avg_census is not None else ''
            li_census = f'<li><strong>Period:</strong> {start_date_str} – {end_date_str}{period_note}</li>'
            # Concise labels: define "direct" once; avoid repeating excludes/includes
            li_total_hprd = f'<li><strong>Total HPRD:</strong> {total_hprd:.2f} (all staff, incl. admin/DON)</li>' if include_total_staffing else ''
            li_direct_hprd = f'<li><strong>Direct Care HPRD:</strong> {direct_hprd:.2f} (direct care only)</li>'
            li_total_rn = f'<li><strong>Total RN HPRD:</strong> {total_rn_hprd:.2f} (incl. Admin/DON)</li>' if include_total_staffing and total_rn_hprd else ''
            li_direct_rn = f'<li><strong>Direct RN HPRD:</strong> {direct_rn_hprd:.2f} (direct care RN only)</li>'
            if direct_first:
                bullets = [li_census, li_direct_hprd, li_direct_rn, li_total_hprd, li_total_rn]
            else:
                bullets = [li_census, li_total_hprd, li_direct_hprd, li_total_rn, li_direct_rn]
            staffing_metrics_html = f"""
            <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Resident Stay Period Staffing Metrics</h3>
            <ul style="font-size: 11pt; line-height: 1.8;">
                {''.join(b for b in bullets if b)}
            </ul>
            """
        
        # Build days below minimum paragraph
        days_below_minimum_html = ""
        if period_metrics and macpac_standards and macpac_standards.get('min_staffing', 0) > 0:
            days_under_total = period_metrics.get('days_under_minimum_total', 0)
            days_under_direct = period_metrics.get('days_under_minimum_direct', 0)
            total_days = period_metrics.get('total_days', 0)
            min_staffing = macpac_standards.get('min_staffing', 0)
            state_name = STATE_NAMES.get(state.upper(), state)
            
            # Determine if this is New Jersey (estimated) or other state
            is_nj = state.upper() == 'NJ'
            if is_nj:
                requirement_text = f"estimated New Jersey state minimum requirement"
                requirement_note = " (Note: New Jersey state minimum is estimated based on MACPAC data)"
            else:
                requirement_text = f"{state_name} state minimum requirement"
                requirement_note = ""
            
            if days_under_total > 0 or days_under_direct > 0:
                part_total = f'{days_under_total:,} days ({period_metrics.get("percentage_under_total", 0.0):.1f}%) for total staffing' if include_total_staffing and days_under_total > 0 else ''
                part_direct = f'{days_under_direct:,} days ({period_metrics.get("percentage_under_direct", 0.0):.1f}%) for direct care staffing' if days_under_direct > 0 else ''
                connector = ' and ' if part_total and part_direct else ''
                ordered = (part_direct + connector + part_total) if direct_first else (part_total + connector + part_direct)
                days_below_minimum_html = f"""
                <p style="font-size: 11pt; line-height: 1.6; margin-top: 15px; padding: 12px; background-color: #fff5f5; border-left: 4px solid #dc3545; border-radius: 4px;">
                    <strong>State Minimum Compliance:</strong> During the {total_days:,}-day review period, the facility fell below the {requirement_text} of {min_staffing:.2f} HPRD on {ordered}.{requirement_note}
                </p>
                """
            elif total_days > 0:
                days_below_minimum_html = f"""
                <p style="font-size: 11pt; line-height: 1.6; margin-top: 15px; padding: 12px; background-color: #f0f9ff; border-left: 4px solid #28a745; border-radius: 4px;">
                    <strong>State Minimum Compliance:</strong> The facility maintained staffing levels at or above the {requirement_text} of {min_staffing:.2f} HPRD throughout the {total_days:,}-day review period.{requirement_note}
                </p>
                """
        
        period_metrics_summary = f"""
        {staffing_metrics_html}
        {days_below_minimum_html}
        """
        
        period_metrics_html = ""
    else:
        period_metrics_html = """
    <h2 style="color: #2c3e50; margin-bottom: 20px;">Resident Stay Period Staffing (Exact Dates)</h2>
    <p><em>Period-specific staffing metrics will be calculated and added separately.</em></p>
    """
        period_metrics_summary = ""
    
    # Daily staffing section
    daily_staffing_html = ""
    if daily_staffing:
        state_minimum = macpac_standards.get('min_staffing', 0.0) if macpac_standards else 0.0
        daily_staffing_html = f"""
    <div class="daily-staffing-section">
    <h2 style="color: #2c3e50; margin-bottom: 20px;">Daily Staffing on Key Dates</h2>
    <p>The following table shows detailed staffing levels on the specific dates when resident incidents occurred:</p>
    {generate_daily_staffing_table(daily_staffing, state_minimum, include_total_staffing, direct_first)}
    </div>
    """
    
    # State Compliance Analysis Tool
    days_under_html = ""
    if period_metrics and macpac_standards and macpac_standards.get('min_staffing', 0) > 0:
        total_days = period_metrics.get('total_days', 0)
        min_staffing = macpac_standards['min_staffing']
        
        # Calculate compliance metrics
        days_under_total = period_metrics.get('days_under_minimum_total', 0)
        days_under_direct = period_metrics.get('days_under_minimum_direct', 0)
        days_in_compliance_total = total_days - days_under_total
        days_in_compliance_direct = total_days - days_under_direct
        pct_under_total = period_metrics.get('percentage_under_total', 0.0)
        pct_under_direct = period_metrics.get('percentage_under_direct', 0.0)
        pct_in_compliance_total = 100.0 - pct_under_total
        pct_in_compliance_direct = 100.0 - pct_under_direct
        
        # Build comprehensive compliance table
        compliance_rows = []
        
        # Header row
        compliance_rows.append("""
            <tr style="background-color: #f8f9fa; font-weight: 600;">
                <th style="padding: 10px; text-align: left; border-bottom: 2px solid #dee2e6;">Metric</th>
                <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">Days</th>
                <th style="padding: 10px; text-align: center; border-bottom: 2px solid #dee2e6;">Percentage</th>
            </tr>""")
        
        # Total days row
        compliance_rows.append(f"""
            <tr>
                <td style="padding: 8px; font-weight: 600;">Total Days in Period</td>
                <td style="padding: 8px; text-align: center;">{total_days:,}</td>
                <td style="padding: 8px; text-align: center;">100.0%</td>
            </tr>""")
        
        if include_total_staffing:
            # Total HPRD compliance
            compliance_rows.append(f"""
            <tr>
                <td style="padding: 8px; font-weight: 600; border-top: 1px solid #dee2e6;">Total HPRD Compliance ({min_staffing:.2f} HPRD minimum)</td>
                <td style="padding: 8px; text-align: center; border-top: 1px solid #dee2e6;">—</td>
                <td style="padding: 8px; text-align: center; border-top: 1px solid #dee2e6;">—</td>
            </tr>
            <tr>
                <td style="padding: 8px; padding-left: 20px;">Days IN Compliance (Total HPRD)</td>
                <td style="padding: 8px; text-align: center;">{days_in_compliance_total:,}</td>
                <td style="padding: 8px; text-align: center;">{pct_in_compliance_total:.1f}%</td>
            </tr>
            <tr>
                <td style="padding: 8px; padding-left: 20px;">✗ Days OUT of Compliance (Total HPRD)</td>
                <td style="padding: 8px; text-align: center; color: #dc3545; font-weight: 600;">{days_under_total:,}</td>
                <td style="padding: 8px; text-align: center; color: #dc3545; font-weight: 600;">{pct_under_total:.1f}%</td>
            </tr>""")
        
        # Direct Care HPRD compliance
        compliance_rows.append(f"""
            <tr>
                <td style="padding: 8px; font-weight: 600; border-top: 1px solid #dee2e6;">Direct Care HPRD Compliance ({min_staffing:.2f} HPRD minimum)</td>
                <td style="padding: 8px; text-align: center; border-top: 1px solid #dee2e6;">—</td>
                <td style="padding: 8px; text-align: center; border-top: 1px solid #dee2e6;">—</td>
            </tr>
            <tr>
                <td style="padding: 8px; padding-left: 20px;">Days IN Compliance (Direct Care HPRD)</td>
                <td style="padding: 8px; text-align: center;">{days_in_compliance_direct:,}</td>
                <td style="padding: 8px; text-align: center;">{pct_in_compliance_direct:.1f}%</td>
            </tr>
            <tr>
                <td style="padding: 8px; padding-left: 20px;">✗ Days OUT of Compliance (Direct Care HPRD)</td>
                <td style="padding: 8px; text-align: center; color: #dc3545; font-weight: 600;">{days_under_direct:,}</td>
                <td style="padding: 8px; text-align: center; color: #dc3545; font-weight: 600;">{pct_under_direct:.1f}%</td>
            </tr>""")
        
        # Use full state name for display (consistent with STATE_NAMES)
        state_full_display = STATE_NAMES.get(state.upper(), state)
        is_nj_compliance = state.upper() == 'NJ'
        if is_nj_compliance:
            requirement_text_compliance = "estimated New Jersey state minimum requirement"
            compliance_note = " <em>(Note: New Jersey state minimum is estimated based on MACPAC data)</em>"
        else:
            requirement_text_compliance = f"{state_full_display} state minimum requirement"
            compliance_note = ""
        
        days_under_html = f"""
    <div style="page-break-inside: avoid; break-inside: avoid; margin-top: 30px;">
    <h2 style="color: #2c3e50; margin-bottom: 20px;">State Minimum Staffing Compliance Analysis</h2>
    <p style="font-size: 10pt; margin-bottom: 15px;">This compliance tool analyzes staffing levels against the {requirement_text_compliance} of <strong>{min_staffing:.2f} HPRD</strong> during the review period ({start_date_str} to {end_date_str}).{compliance_note} The analysis shows both total staffing (including administrative and DON hours) and direct care staffing (excluding administrative and DON hours).</p>
    
    <table style="width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 10pt; border: 1px solid #dee2e6;">
        <thead>
            {compliance_rows[0]}
        </thead>
        <tbody>
            {''.join(compliance_rows[1:])}
        </tbody>
    </table>
    
    <div style="margin-top: 20px; padding: 12px; background-color: #e7f3ff; border-left: 4px solid #2196F3; border-radius: 4px;">
        <p style="margin: 0; font-size: 9pt; line-height: 1.6;">
            <strong>Definitions:</strong><br>
            • <strong>Total HPRD:</strong> Includes all nursing staff (RN, LPN, nurse aides [CNA, Med Aide, NA Trn]) plus RN Admin, RN DON, and LPN Admin hours.<br>
            • <strong>Direct Care HPRD:</strong> Excludes administrative and DON hours, representing only direct patient care staff.<br>
            • <strong>State Minimum:</strong> {state_full_display} requires a minimum of {min_staffing:.2f} HPRD of total nursing staff{' (estimated based on MACPAC data)' if is_nj_compliance else ''}.<br>
            • <strong>Compliance:</strong> Days when staffing met or exceeded the {requirement_text_compliance}.
        </p>
    </div>
    
    <p style="font-size: 9pt; font-style: italic; margin-top: 15px;">Source: <a href="{MACPAC_SOURCE_URL}" target="_blank" rel="noopener">MACPAC</a> | CMS PBJ data</p>
    </div>
    """
    
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PBJ Report {provnum} | {start_date.strftime('%b %Y')}–{end_date.strftime('%b %Y')}</title>
    <style>
        /* Page formatting */
        @page {{
            size: letter;
            margin: 0.75in;
            margin-bottom: 1in;
        }}
        .page-break {{
            page-break-before: always;
            break-before: page;
        }}
        /* Page numbering footer */
        .page-footer {{
            position: fixed;
            bottom: 0.5in;
            right: 0.75in;
            font-size: 9pt;
            color: #666;
            font-family: 'Calibri', 'Arial', sans-serif;
        }}
        @media print {{
            .page-footer {{
                position: fixed;
                bottom: 0.5in;
                right: 0.75in;
            }}
        }}
        /* Word-compatible styles - preserve padding for PDF */
        @media print {{
            body {{
                margin: 0;
                padding: 0.75in !important;
            }}
        }}
        table {{
            border-collapse: collapse;
        }}
        table, td, th {{
            border-collapse: collapse;
        }}
        /* Prevent table breaks across pages */
        table {{
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        thead {{
            display: table-header-group;
        }}
        tbody {{
            display: table-row-group;
        }}
        tr {{
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        /* Keep daily staffing table together */
        .daily-staffing-section {{
            page-break-inside: avoid;
            break-inside: avoid;
        }}
        body {{
            font-family: 'Calibri', 'Arial', sans-serif;
            font-size: 10.5pt;
            line-height: 1.5;
            color: #2c2c2c;
            max-width: 8.5in;
            margin: 0 auto;
            padding: 0.75in;
            background-color: #ffffff;
        }}
        .header-branding {{
            text-align: right;
            margin-bottom: 25px;
            padding-bottom: 12px;
            border-bottom: 3px solid #2980b9;
        }}
        .header-branding .company-name {{
            font-size: 16pt;
            font-weight: 600;
            color: #2c3e50;
            margin-bottom: 0;
        }}
        .header-branding .company-tagline {{
            font-size: 9pt;
            color: #7f8c8d;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #2980b9;
            padding-bottom: 8px;
            margin-top: 15px;
            margin-bottom: 12px;
            font-size: 17pt;
            font-weight: 600;
            page-break-after: avoid;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 6px;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 13pt;
            font-weight: 600;
            page-break-after: avoid;
        }}
        h3 {{
            color: #34495e;
            margin-top: 15px;
            margin-bottom: 8px;
            font-size: 11.5pt;
            font-weight: 600;
            page-break-after: avoid;
        }}
        p {{
            margin-top: 6px;
            margin-bottom: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            page-break-inside: avoid;
            font-size: 9.5pt;
            text-align: left !important;
            border: 1px solid #bdc3c7;
        }}
        th, td {{
            border: 1px solid #ecf0f1;
            padding: 6px 8px;
            text-align: left !important;
        }}
        th {{
            background-color: #e8f4f8;
            font-weight: 600;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
        }}
        tr:nth-child(even) {{
            background-color: #f5f9fb;
        }}
        ul, ol {{
            margin: 6px 0;
            padding-left: 22px;
        }}
        li {{
            margin: 3px 0;
            line-height: 1.4;
        }}
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        .summary-box {{
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            padding: 12px;
            margin: 15px 0;
        }}
        .sources {{
            font-size: 9pt;
            color: #6c757d;
            margin-top: 8px;
            margin-bottom: 12px;
            font-style: italic;
        }}
        .macpac-note {{
            background-color: #ffffff;
            border-left: 3px solid #667eea;
            padding: 10px;
            margin: 12px 0;
            font-size: 9.5pt;
        }}
        .highlight-important {{
            background-color: #f0f4ff;
            border-left: 2px solid #667eea;
            padding: 3px 5px;
        }}
        .below-state-avg {{
            background-color: #fff5f5;
            color: #c0392b;
            font-weight: 500;
        }}
        .below-state-min {{
            background-color: #ffe8e8;
            color: #c0392b;
            font-weight: 600;
        }}
        .key-date-row {{
            background-color: #f0f8ff;
        }}
        .footer-branding {{
            text-align: center;
            margin-top: 35px;
            padding-top: 18px;
            border-top: 2px solid #bdc3c7;
            font-size: 9pt;
            color: #6c757d;
        }}
        .footer-branding .contact {{
            margin-top: 8px;
            font-weight: 500;
        }}
        .sources-section {{
            margin-top: 25px;
            font-size: 9.5pt;
        }}
        .sources-section a {{
            color: #2980b9;
            text-decoration: none;
        }}
        .sources-section a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    {watermark_section}
    
    <div class="header-branding">
        <div class="company-name">320 Consulting</div>
        <div class="company-tagline">Nursing Home Staffing Analysis</div>
    </div>
    
    <h1 style="color: #2c3e50; margin-top: 10px; margin-bottom: 15px;">PBJ Analysis: {format_facility_name(facility_name)}</h1>
    <p><strong>Location:</strong> {format_city_name(city)}, {state} | <strong>Provider Number:</strong> {provnum} | <strong>Resident Stay Period:</strong> {start_date_str} to {end_date_str}</p>
    {_format_previous_names_line(previous_names)}
    <p class="sources"><strong>Sources:</strong> <a href="https://www.pbj320.com/provider/{provnum}" target="_blank" rel="noopener">PBJ320</a> | <a href="{SOURCES_CMS_PBJ_URL}" target="_blank" rel="noopener">CMS Payroll-Based Journal</a> | <a href="{SOURCES_CMS_PROVIDER_INFO_URL}" target="_blank" rel="noopener">CMS Provider Information</a></p>
    
    <div class="summary-box">
        <h2 style="color: #2c3e50; margin-top: 0; margin-bottom: 15px; font-size: 16pt;">Executive Summary</h2>
        <p style="font-size: 11pt; line-height: 1.6;">Staffing levels at {format_facility_name(facility_name)} from {start_date_str} through {end_date_str}, with focus on potential understaffing relevant to resident incidents.</p>
        {f'<h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Key Dates of Interest (Resident Incidents)</h3><ul>{key_dates_html}</ul>' if (inc.get('key_dates', True) and key_dates_html) else ''}
        {date_ranges_html}
        {period_metrics_summary if (period_metrics and inc.get('period_summary', True)) else ''}
    </div>
    
    {daily_staffing_html if inc.get('daily_staffing_table', True) else ''}
    
    {f'''
    <div style="page-break-inside: avoid; break-inside: avoid; margin-top: 30px; margin-bottom: 30px;">
    <h2 style="color: #2c3e50; margin-bottom: 20px;">Days Below State Minimum During Resident Stay Period</h2>
    {f'''
    <div style="background-color: #ffffff; border: 2px solid #e9ecef; border-left: 4px solid #dc3545; padding: 20px; margin: 15px 0; border-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <p style="margin: 0 0 15px 0; font-size: 12pt; font-weight: 700; color: #2c3e50; border-bottom: 2px solid #e9ecef; padding-bottom: 10px;">State Minimum Requirement{' (Estimated)' if state.upper() == 'NJ' else ''}: <span style="color: #dc3545;">{macpac_standards.get('min_staffing', 0):.2f} HPRD</span></p>
        <table style="width: 100%; font-size: 10pt; margin-top: 10px; border-collapse: collapse;">
            <tr style="background-color: #f8f9fa;">
                <td style="padding: 12px; font-weight: 600; width: 50%; color: #2c3e50; border-bottom: 1px solid #dee2e6;">Total Days in Period:</td>
                <td style="padding: 12px; color: #2c3e50; border-bottom: 1px solid #dee2e6;"><strong>{period_metrics.get('total_days', 0):,} days</strong></td>
            </tr>
            {f'''
            <tr style="background-color: {'#fff5f5' if period_metrics.get('days_under_minimum_total', 0) > 0 else '#f0f9ff'};">
                <td style="padding: 12px; font-weight: 600; color: #2c3e50; border-bottom: 1px solid #dee2e6;">Days Below Minimum - Total HPRD:</td>
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6; color: {'#dc3545' if period_metrics.get('days_under_minimum_total', 0) > 0 else '#28a745'};">
                    <strong>{period_metrics.get('days_under_minimum_total', 0):,} days ({period_metrics.get('percentage_under_total', 0.0):.1f}%)</strong>
                </td>
            </tr>
            ''' if include_total_staffing else ''}
            <tr style="background-color: {'#fff5f5' if period_metrics.get('days_under_minimum_direct', 0) > 0 else '#f0f9ff'};">
                <td style="padding: 12px; font-weight: 600; color: #2c3e50; border-bottom: 1px solid #dee2e6;">Days Below Minimum - Direct Care HPRD:</td>
                <td style="padding: 12px; border-bottom: 1px solid #dee2e6; color: {'#dc3545' if period_metrics.get('days_under_minimum_direct', 0) > 0 else '#28a745'};">
                    <strong>{period_metrics.get('days_under_minimum_direct', 0):,} days ({period_metrics.get('percentage_under_direct', 0.0):.1f}%)</strong>
                </td>
            </tr>
        </table>
    </div>
    ''' if period_metrics and macpac_standards and macpac_standards.get('min_staffing', 0) > 0 else '<p style="font-size: 10pt; color: #666;"><em>State minimum staffing data not available for this period.</em></p>'}
    </div>
    ''' if (inc.get('state_compliance', True) and period_metrics and macpac_standards and macpac_standards.get('min_staffing', 0) > 0) else ''}
    
    {f'''
    <div style="page-break-inside: avoid; break-inside: avoid; margin-top: 40px; margin-bottom: 30px;">
    <h2 style="color: #2c3e50; margin-bottom: 20px;">Longitudinal Staffing Analysis</h2>
    <p style="font-size: 10pt; margin-bottom: 20px;">Interactive charts showing staffing trends over time. Use the view toggle to switch between daily, monthly, and quarterly views.</p>
    
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    
    <div style="margin-bottom: 40px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin: 0;">Total and Direct Care HPRD Over Time</h3>
            <div style="display: flex; gap: 5px;">
                <button onclick="changeView('hprd-chart', 'daily')" id="hprd-daily-btn" style="padding: 6px 12px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 9pt;">Day</button>
                <button onclick="changeView('hprd-chart', 'monthly')" id="hprd-monthly-btn" style="padding: 6px 12px; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 9pt;">Month</button>
                <button onclick="changeView('hprd-chart', 'quarterly')" id="hprd-quarterly-btn" style="padding: 6px 12px; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 9pt;">Quarter</button>
            </div>
        </div>
        <canvas id="hprd-chart" style="max-height: 400px;"></canvas>
    </div>
    
    <div style="margin-bottom: 40px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin: 0;">RN Total and Direct Care HPRD Over Time</h3>
            <div style="display: flex; gap: 5px;">
                <button onclick="changeView('rn-chart', 'daily')" id="rn-daily-btn" style="padding: 6px 12px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 9pt;">Day</button>
                <button onclick="changeView('rn-chart', 'monthly')" id="rn-monthly-btn" style="padding: 6px 12px; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 9pt;">Month</button>
                <button onclick="changeView('rn-chart', 'quarterly')" id="rn-quarterly-btn" style="padding: 6px 12px; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 9pt;">Quarter</button>
            </div>
        </div>
        <canvas id="rn-chart" style="max-height: 400px;"></canvas>
    </div>
    
    <div style="margin-bottom: 40px;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin: 0;">Census Over Time</h3>
            <div style="display: flex; gap: 5px;">
                <button onclick="changeView('census-chart', 'daily')" id="census-daily-btn" style="padding: 6px 12px; background: #667eea; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 9pt;">Day</button>
                <button onclick="changeView('census-chart', 'monthly')" id="census-monthly-btn" style="padding: 6px 12px; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 9pt;">Month</button>
                <button onclick="changeView('census-chart', 'quarterly')" id="census-quarterly-btn" style="padding: 6px 12px; background: #e0e0e0; color: #333; border: none; border-radius: 4px; cursor: pointer; font-size: 9pt;">Quarter</button>
            </div>
        </div>
        <canvas id="census-chart" style="max-height: 400px;"></canvas>
    </div>
    
    <script>
        const timeSeriesData = {time_series_json};
        const stateMinimum = {state_minimum_value};
        const keyDates = {json.dumps([d.strftime('%Y-%m-%d') for d in key_dates] if key_dates else [])};
        let hprdChart, rnChart, censusChart;
        // Calculate default view based on number of days
        const daysDiff = Math.ceil((new Date('{end_date_str}') - new Date('{start_date_str}')) / (1000 * 60 * 60 * 24));
        let defaultView = 'daily';
        if (daysDiff > 732) {{
            defaultView = 'quarterly';  // > 2 years: default to quarter
        }} else if (daysDiff >= 32) {{
            defaultView = 'monthly';    // 32-731 days: default to month
        }} else {{
            defaultView = 'daily';      // ≤31 days: default to day
        }}
        
        let currentView = {{'hprd-chart': defaultView, 'rn-chart': defaultView, 'census-chart': defaultView}};
        
        // Initialize charts with default view
        setTimeout(() => {{
            changeView('hprd-chart', defaultView);
            changeView('rn-chart', defaultView);
            changeView('census-chart', defaultView);
        }}, 100);
        
        function changeView(chartId, view) {{
            currentView[chartId] = view;
            updateButtons(chartId, view);
            updateChart(chartId, view);
        }}
        
        function updateButtons(chartId, view) {{
            const prefixes = {{'hprd-chart': 'hprd', 'rn-chart': 'rn', 'census-chart': 'census'}};
            const prefix = prefixes[chartId];
            ['daily', 'monthly', 'quarterly'].forEach(v => {{
                const btnId = v === 'monthly' ? `${{prefix}}-monthly-btn` : (v === 'quarterly' ? `${{prefix}}-quarterly-btn` : `${{prefix}}-daily-btn`);
                const btn = document.getElementById(btnId);
                if (btn) {{
                    btn.style.background = v === view ? '#667eea' : '#e0e0e0';
                    btn.style.color = v === view ? 'white' : '#333';
                }}
            }});
        }}
        
        function calculateYAxisMin(dataValues, includeZero = false) {{
            const validValues = dataValues.filter(v => v !== null && v !== undefined && !isNaN(v));
            if (validValues.length === 0) return 0;
            const min = Math.min(...validValues);
            const max = Math.max(...validValues);
            const range = max - min;
            // Set min to 10% below data min, but not below 0 if includeZero is true
            const calculatedMin = Math.max(0, min - (range * 0.1));
            return includeZero ? 0 : calculatedMin;
        }}
        
        function findKeyDateIndices(labels, dates, view) {{
            // Red circles only on daily view (exact key dates). No red circles on monthly or quarterly.
            const indices = [];
            if (!keyDates || keyDates.length === 0 || view !== 'daily') return indices;
            keyDates.forEach(keyDate => {{
                const dateStr = new Date(keyDate).toISOString().split('T')[0];  // YYYY-MM-DD
                dates.forEach((dataDate, idx) => {{
                    if (dataDate === dateStr && !indices.includes(idx)) indices.push(idx);
                }});
            }});
            return indices;
        }}
        
        function updateChart(chartId, view) {{
            const data = timeSeriesData[view] || [];
            if (!data || data.length === 0) return;
            
            const labels = data.map(d => d.label);
            const dates = data.map(d => d.date);
            
            if (chartId === 'hprd-chart') {{
                const datasets = [
                    {{
                        label: 'Direct Care HPRD',
                        data: data.map(d => d.direct_care_hprd),
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.1
                    }}
                ];
                if (data[0].total_hprd !== null && data[0].total_hprd !== undefined) {{
                    datasets.push({{
                        label: 'Total HPRD',
                        data: data.map(d => d.total_hprd),
                        borderColor: '#9b59b6',
                        backgroundColor: 'rgba(155, 89, 182, 0.1)',
                        tension: 0.1
                    }});
                }}
                if (stateMinimum > 0) {{
                    datasets.push({{
                        label: 'State Minimum',
                        data: Array(labels.length).fill(stateMinimum),
                        borderColor: '#e74c3c',
                        borderDash: [5, 5],
                        borderWidth: 2,
                        pointRadius: 0
                    }});
                }}
                
                // Calculate dynamic y-axis min
                const allHprdValues = [...data.map(d => d.direct_care_hprd), ...data.map(d => d.total_hprd || []), stateMinimum > 0 ? [stateMinimum] : []].flat().filter(v => v !== null && v !== undefined);
                const yMin = calculateYAxisMin(allHprdValues, false);
                const keyDateIndices = findKeyDateIndices(labels, dates, view);
                
                // Add large point markers at key dates on actual data lines (not as separate datasets)
                datasets.forEach((dataset, datasetIdx) => {{
                    if (dataset.label !== 'State Minimum') {{
                        // Create point radius array - larger at key dates
                        dataset.pointRadius = data.map((_, idx) => keyDateIndices.includes(idx) ? 8 : 3);
                        dataset.pointHoverRadius = data.map((_, idx) => keyDateIndices.includes(idx) ? 10 : 5);
                        dataset.pointBackgroundColor = data.map((_, idx) => keyDateIndices.includes(idx) ? '#e74c3c' : dataset.borderColor);
                        dataset.pointBorderColor = data.map((_, idx) => keyDateIndices.includes(idx) ? '#ffffff' : dataset.borderColor);
                        dataset.pointBorderWidth = data.map((_, idx) => keyDateIndices.includes(idx) ? 2 : 1);
                    }}
                }});
                
                if (hprdChart) hprdChart.destroy();
                hprdChart = new Chart(document.getElementById('hprd-chart'), {{
                    type: 'line',
                    data: {{ labels, datasets }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {{
                            legend: {{ display: true, position: 'top' }},
                            title: {{ display: false }}
                        }},
                        scales: {{
                            y: {{
                                min: yMin,
                                title: {{ display: true, text: 'HPRD' }}
                            }},
                            x: {{
                                ticks: {{
                                    maxRotation: view === 'daily' ? 45 : 0,
                                    autoSkip: true,
                                    maxTicksLimit: view === 'daily' ? 20 : (view === 'monthly' ? 12 : 8),
                                    callback: function(value, index, ticks) {{
                                        if (view === 'monthly' || view === 'quarterly') {{
                                            const label = this.getLabelForValue(value);
                                            if (view === 'monthly') {{
                                                // Format: "Jan 2024" instead of "Jan 15, 2024"
                                                const match = label.match(/(\\w+)\\s+\\d+,\\s+(\\d+)/);
                                                if (match) return match[1] + ' ' + match[2];
                                                return label.split(',')[0] + (label.includes(',') ? ' ' + label.split(',')[1].trim() : '');
                                            }} else {{
                                                // Format: "Q1 2024" - already simplified
                                                return label;
                                            }}
                                        }}
                                        // For daily view, return the default label
                                        return this.getLabelForValue(value);
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            }} else if (chartId === 'rn-chart') {{
                const datasets = [
                    {{
                        label: 'Direct RN HPRD',
                        data: data.map(d => d.direct_rn_hprd),
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        tension: 0.1,
                        borderWidth: 2
                    }}
                ];
                if (data[0].total_rn_hprd !== null && data[0].total_rn_hprd !== undefined) {{
                    datasets.push({{
                        label: 'Total RN HPRD',
                        data: data.map(d => d.total_rn_hprd),
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.1,
                        borderWidth: 2,
                        borderDash: [5, 5]
                    }});
                }}
                
                // Calculate dynamic y-axis min
                const allRnValues = [...data.map(d => d.direct_rn_hprd), ...data.map(d => d.total_rn_hprd || [])].flat().filter(v => v !== null && v !== undefined);
                const rnYMin = calculateYAxisMin(allRnValues, false);
                const rnYMax = Math.max(...allRnValues) * 1.1;
                const rnKeyDateIndices = findKeyDateIndices(labels, dates, view);
                
                // Add large point markers at key dates on actual data lines (not as separate datasets)
                datasets.forEach((dataset) => {{
                    // Create point radius array - larger at key dates
                    dataset.pointRadius = data.map((_, idx) => rnKeyDateIndices.includes(idx) ? 8 : 3);
                    dataset.pointHoverRadius = data.map((_, idx) => rnKeyDateIndices.includes(idx) ? 10 : 5);
                    dataset.pointBackgroundColor = data.map((_, idx) => rnKeyDateIndices.includes(idx) ? '#e74c3c' : dataset.borderColor);
                    dataset.pointBorderColor = data.map((_, idx) => rnKeyDateIndices.includes(idx) ? '#ffffff' : dataset.borderColor);
                    dataset.pointBorderWidth = data.map((_, idx) => rnKeyDateIndices.includes(idx) ? 2 : 1);
                }});
                
                if (rnChart) rnChart.destroy();
                rnChart = new Chart(document.getElementById('rn-chart'), {{
                    type: 'line',
                    data: {{ labels, datasets }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {{
                            legend: {{ display: true, position: 'top' }},
                            title: {{ display: false }}
                        }},
                        scales: {{
                            y: {{
                                min: rnYMin,
                                title: {{ display: true, text: 'RN HPRD' }}
                            }},
                            x: {{
                                ticks: {{
                                    maxRotation: view === 'daily' ? 45 : 0,
                                    autoSkip: true,
                                    maxTicksLimit: view === 'daily' ? 20 : (view === 'monthly' ? 12 : 8),
                                    callback: function(value, index, ticks) {{
                                        if (view === 'monthly' || view === 'quarterly') {{
                                            const label = this.getLabelForValue(value);
                                            if (view === 'monthly') {{
                                                // Format: "Jan 2024" instead of "Jan 15, 2024"
                                                const match = label.match(/(\\w+)\\s+\\d+,\\s+(\\d+)/);
                                                if (match) return match[1] + ' ' + match[2];
                                                return label.split(',')[0] + (label.includes(',') ? ' ' + label.split(',')[1].trim() : '');
                                            }} else {{
                                                // Format: "Q1 2024" - already simplified
                                                return label;
                                            }}
                                        }}
                                        // For daily view, return the default label
                                        return this.getLabelForValue(value);
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            }} else if (chartId === 'census-chart') {{
                // Calculate dynamic y-axis min and max with better rounding
                const censusValues = data.map(d => d.census).filter(v => v !== null && v !== undefined);
                const censusMin = Math.min(...censusValues);
                const censusMax = Math.max(...censusValues);
                const censusRange = censusMax - censusMin;
                
                // Round min down to nearest 10 (or 5 for smaller facilities)
                const roundTo = censusRange < 50 ? 5 : 10;
                const censusYMin = Math.floor(censusMin / roundTo) * roundTo;
                // Round max up to nearest 10 (or 5 for smaller facilities) with some padding
                const censusYMax = Math.ceil(censusMax / roundTo) * roundTo + roundTo;
                
                const censusKeyDateIndices = findKeyDateIndices(labels, dates, view);
                
                const censusDatasets = [{{
                    label: 'Census',
                    data: data.map(d => d.census),
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    tension: 0.1,
                    // Add large point markers at key dates
                    pointRadius: data.map((_, idx) => censusKeyDateIndices.includes(idx) ? 8 : 3),
                    pointHoverRadius: data.map((_, idx) => censusKeyDateIndices.includes(idx) ? 10 : 5),
                    pointBackgroundColor: data.map((_, idx) => censusKeyDateIndices.includes(idx) ? '#e74c3c' : '#f39c12'),
                    pointBorderColor: data.map((_, idx) => censusKeyDateIndices.includes(idx) ? '#ffffff' : '#f39c12'),
                    pointBorderWidth: data.map((_, idx) => censusKeyDateIndices.includes(idx) ? 2 : 1)
                }}];
                
                if (censusChart) censusChart.destroy();
                censusChart = new Chart(document.getElementById('census-chart'), {{
                    type: 'line',
                    data: {{
                        labels,
                        datasets: censusDatasets
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: true,
                        plugins: {{
                            legend: {{ display: true, position: 'top' }},
                            title: {{ display: false }}
                        }},
                        scales: {{
                            y: {{
                                min: censusYMin,
                                max: censusYMax,
                                title: {{ display: true, text: 'Residents' }}
                            }},
                            x: {{
                                ticks: {{
                                    maxRotation: view === 'daily' ? 45 : 0,
                                    autoSkip: true,
                                    maxTicksLimit: view === 'daily' ? 20 : (view === 'monthly' ? 12 : 8),
                                    callback: function(value, index, ticks) {{
                                        if (view === 'monthly' || view === 'quarterly') {{
                                            const label = this.getLabelForValue(value);
                                            if (view === 'monthly') {{
                                                // Format: "Jan 2024" instead of "Jan 15, 2024"
                                                const match = label.match(/(\\w+)\\s+\\d+,\\s+(\\d+)/);
                                                if (match) return match[1] + ' ' + match[2];
                                                return label.split(',')[0] + (label.includes(',') ? ' ' + label.split(',')[1].trim() : '');
                                            }} else {{
                                                // Format: "Q1 2024" - already simplified
                                                return label;
                                            }}
                                        }}
                                        // For daily view, return the default label
                                        return this.getLabelForValue(value);
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
            }}
        }}
        
        // Initialize all charts with default view based on date range
        document.addEventListener('DOMContentLoaded', function() {{
            ['hprd-chart', 'rn-chart', 'census-chart'].forEach(chartId => {{
                updateButtons(chartId, defaultView);
                updateChart(chartId, defaultView);
            }});
        }});
    </script>
    </div>
    ''' if (inc.get('quarterly_staffing', True) and pbj_df is not None and not time_series_data.get('daily') == []) else ''}
    
    {f'''
    <h2>Resident Stay Period Analysis</h2>
    <p>Staffing metrics for {format_facility_name(facility_name)} during the resident stay period and by quarter, compared to {state} state averages.</p>
    
    <table>
        <thead>
            <tr>
                <th rowspan="2">Period</th>
                <th rowspan="2">Census</th>
                {appendix_thead_col1}
                <th rowspan="2">Contract %</th>
            </tr>
            <tr>
                {appendix_thead_col2}
            </tr>
        </thead>
        <tbody>
            {f'''
            <tr>
                <td><strong>Resident Stay Period</strong><br><span style="font-size: 8pt; color: #666;">{start_date_str} to {end_date_str}</span></td>
                <td>{period_metrics.get("avg_census", 0):.1f}</td>
                {appendix_period_cells}
                <td>{period_metrics.get("contract_pct", 0):.1f}%</td>
            </tr>
            ''' if period_metrics else ''}
            {quarterly_table}
        </tbody>
    </table>
    ''' if inc.get('quarterly_staffing', True) else ''}
    
    <div class="page-break"></div>
    {generate_case_mix_section(case_mix_data, quarterly_data, pbj_df) if (inc.get('case_mix', True) and (case_mix_data or quarterly_data)) else ''}
    
    <div class="page-break"></div>
    {generate_red_flags_section(red_flags_history) if (inc.get('red_flags', True) and red_flags_history) else ''}
    
    {f'''
    <div class="page-break"></div>
    <h1 style="color: #2c3e50; border-bottom: 3px solid #2c3e50; padding-bottom: 10px; margin-bottom: 20px;">Appendix</h1>
    
    <h2>Data Sources and Methodology</h2>
    <p>This report is based on data from the Centers for Medicare & Medicaid Services (CMS) Payroll-Based Journal (PBJ) system. The PBJ system requires nursing homes to submit daily staffing data, including:</p>
    <ul>
        <li>Total nursing staff hours (RN, LPN, CNA, and other nursing staff)</li>
        <li>Contract staff hours</li>
        <li>Daily resident census</li>
    </ul>
    <p>Hours Per Resident Day (HPRD) is calculated by dividing total nursing hours by total resident days for each quarter. This metric provides a standardized measure of staffing levels that accounts for variations in census.</p>
    <p>State averages are calculated using weighted averages (total hours divided by total resident days) for all facilities in the state of {state}.</p>
    
    <h3 id="provider-info-matching" style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Provider information matching</h3>
    <p style="font-size: 10pt;"><strong>PBJ data</strong> is quarterly; hours and census are rolled from daily submissions, so quarter-level totals are accurate.</p>
    <p style="font-size: 10pt;"><strong>Provider info</strong> (CMS Provider Information files) is published on a rolling basis and has both a <code>quarter</code> and a <code>processing_date</code>. How we use it in this report:</p>
    <ul style="font-size: 10pt; margin-top: 6px;">
        <li><strong>Red flags</strong> (SFF, 1-star, abuse icon, Admin TO, ownership): we include rows whose <code>processing_date</code> falls within the report’s date range. The quarter we display is from the file or derived from that date—it is an <strong>estimate</strong> of when the flag applied, not an exact incident date.</li>
        <li><strong>Case-mix</strong>: when the report has specific quarters, we match provider info rows by <code>quarter</code> (row’s quarter must be in the report’s quarter list). Otherwise we filter by <code>processing_date</code> within the report range.</li>
    </ul>
    
    <div class="sources-section">
        <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Resources</h3>
        <p style="font-size: 10pt; margin: 0;"><a href="https://www.pbj320.com/provider/{provnum}" target="_blank" rel="noopener">PBJ320</a> · <a href="https://www.medicare.gov/care-compare/details/nursing-home/{provnum}/view-all?state={state}" target="_blank" rel="noopener">CMS Care Compare</a> · <a href="{MACPAC_SOURCE_URL}" target="_blank" rel="noopener">State staffing (MACPAC)</a></p>
    </div>
    <p style="font-size: 10pt; margin-bottom: 25px; color: #34495e;">The following tables provide comprehensive total nurse staffing metrics (including administrative and DON hours) for reference. These metrics include all staff hours, not just direct care hours.</p>
    
    <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Resident Stay Period Staffing - Total HPRD</h3>
    {f'''
    <p>The following metrics reflect total staffing levels during the exact resident stay period from {start_date_str} to {end_date_str} ({((end_date - start_date).days + 1):,} days):</p>
    <table>
        <thead>
            <tr>
                <th>Metric</th>
                <th>Facility Value</th>
                <th>{STATE_NAMES.get(state.upper(), state)} State Requirement*</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Total HPRD</strong></td>
                <td {('class="below-state-min"' if period_metrics['total_hprd'] < (macpac_standards.get('min_staffing', 0) if macpac_standards else 0) else '')}><strong>{period_metrics['total_hprd']:.2f}</strong></td>
                <td>{macpac_standards['display_text'] if macpac_standards else 'Check state regulations'}</td>
            </tr>
            <tr>
                <td>RN HPRD (Total)</td>
                <td>{period_metrics['rn_hprd']:.2f}</td>
                <td>N/A</td>
            </tr>
        </tbody>
    </table>
    <p style="font-size: 9pt; font-style: italic;">*<a href="{MACPAC_SOURCE_URL}" target="_blank" rel="noopener">MACPAC</a> state staffing standards.</p>
    ''' if period_metrics else '<p><em>Period-specific staffing metrics not available.</em></p>'}
    
    <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Days Under State Minimum Staffing - Total HPRD</h3>
    {f'''
    <p>The following analysis shows the number of days during the review period when total staffing fell below the {'estimated New Jersey state minimum requirement (based on MACPAC data)' if state.upper() == 'NJ' else f'{STATE_NAMES.get(state.upper(), state)} state minimum requirement'} of {macpac_standards['min_staffing']:.2f} HPRD:</p>
    <table>
        <thead>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Total Days in Period</td>
                <td>{period_metrics.get('total_days', 0):,}</td>
            </tr>
            <tr>
                <td><strong>Days Under State Minimum - Total HPRD</strong> ({macpac_standards['min_staffing']:.2f} HPRD)</td>
                <td class="below-state-min"><strong>{period_metrics.get('days_under_minimum_total', 0):,} days ({period_metrics.get('percentage_under_total', 0.0):.1f}%)</strong></td>
            </tr>
        </tbody>
    </table>
    <p style="font-size: 9pt; font-style: italic;">Note: Total HPRD includes all staff (RN, LPN, nurse aides [CNA, Med Aide, NA Trn], plus admin and DON hours).</p>
    ''' if period_metrics and macpac_standards and macpac_standards.get('min_staffing', 0) > 0 else '<p><em>Days under minimum data not available.</em></p>'}
    
    <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Daily Staffing on Key Dates - <strong>Total</strong> HPRD Complete Details</h3>
    {f'''
    <div style="page-break-inside: avoid; break-inside: avoid;">
    <p>The following comprehensive table shows all staffing metrics on the specific dates when resident incidents occurred:</p>
    <table style="font-size: 7.5pt; width: 100%; table-layout: fixed;">
        <thead>
            <tr>
                <th style="width: 8%;">Date</th>
                <th style="width: 4%;">Census</th>
                <th style="width: 4%;">Total<br>HPRD</th>
                <th style="width: 5%;">Direct<br>Care<br>HPRD</th>
                <th style="width: 4%;">Total<br>RN<br>HPRD</th>
                <th style="width: 3%;">RN<br>HPRD</th>
                <th style="width: 3%;">LPN<br>HPRD</th>
                <th style="width: 4%;">Nurse<br>Aide<br>HPRD</th>
                <th style="width: 4%;">Total<br>RN<br>Hrs</th>
                <th style="width: 3%;">RN<br>Hrs</th>
                <th style="width: 4%;">RN<br>Admin<br>Hrs</th>
                <th style="width: 3%;">RN<br>DON<br>Hrs</th>
                <th style="width: 4%;">LPN<br>Hrs</th>
                <th style="width: 4%;">LPN<br>Admin<br>Hrs</th>
                <th style="width: 5%;">Nurse<br>Aide<br>Hrs</th>
                <th style="width: 4%;">CNA<br>Hrs</th>
                <th style="width: 4%;">Med<br>Aide<br>Hrs</th>
                <th style="width: 3%;">NA<br>Trn<br>Hrs</th>
                <th style="width: 3%;">Contract<br>%</th>
            </tr>
        </thead>
        <tbody>
            {''.join([f'''
            <tr>
                <td><strong>{day['date'].strftime('%b %d, %Y') if isinstance(day['date'], datetime) else day['date']}</strong></td>
                <td>{day.get('census', 0):.0f}</td>
                <td {('class="below-state-min"' if day.get('total_hprd', 0) < (macpac_standards.get('min_staffing', 0) if macpac_standards else 0) else '')}>{day.get('total_hprd', 0):.2f}</td>
                <td>{day.get('direct_care_hprd', 0):.2f}</td>
                <td>{day.get('rn_hprd', 0):.2f}</td>
                <td>{day.get('direct_care_rn_hprd', 0):.2f}</td>
                <td>{day.get('lpn_hprd', 0):.2f}</td>
                <td>{day.get('cna_hprd', 0):.2f}</td>  <!-- Nurse Aide HPRD (CNA + MedAide + NAtrn) -->
                <td>{day.get('total_rn_hours', 0):.2f}</td>
                <td>{day.get('hrs_rn', 0):.2f}</td>
                <td>{day.get('hrs_rnadmin', 0):.2f}</td>
                <td>{day.get('hrs_rndon', 0):.2f}</td>
                <td>{day.get('direct_lpn_hours', day.get('hrs_lpn', 0)):.2f}</td>  <!-- Direct LPN Hours (excludes admin) -->
                <td>{day.get('hrs_lpnadmin', 0):.2f}</td>
                <td>{day.get('total_nurse_aide_hours', 0):.2f}</td>
                <td>{day.get('hrs_cna', 0):.2f}</td>
                <td>{day.get('hrs_medaide', 0):.2f}</td>
                <td>{day.get('hrs_natrn', 0):.2f}</td>
                <td>{day.get('contract_pct', 0):.1f}%</td>
            </tr>''' for day in daily_staffing]) if daily_staffing else '<tr><td colspan="19"><em>No daily staffing data available for key dates.</em></td></tr>'}
        </tbody>
    </table>
    <p style="font-size: 9pt; font-style: italic; margin-top: 10px;"><strong>Note:</strong> Total RN HPRD/Total RN Hrs = RN + RN Admin + RN DON. RN HPRD/RN Hrs = direct care RN only (excl. Admin/DON). LPN HPRD/LPN Hrs = direct LPN only (excl. LPN Admin). Nurse Aide = CNA + Med Aide + NA Trn combined. Contract % = contract hours as % of total nursing hours (all positions).</p>
    </div>
    ''' if daily_staffing else '<p><em>Daily staffing data not available for key dates.</em></p>'}
    
    <h3 style="color: #2c3e50; font-size: 12pt; font-weight: 600; margin-top: 20px; margin-bottom: 10px;">Quarterly Staffing Analysis - Total HPRD</h3>
    <p>Quarterly total staffing metrics for {format_facility_name(facility_name)} compared to {state} state averages:</p>
    <table>
        <thead>
            <tr>
                <th>Quarter</th>
                <th>Census</th>
                <th>Facility<br>Total HPRD</th>
                <th>{state}<br>Total HPRD</th>
                <th>Facility<br>Total RN HPRD</th>
                <th>{state}<br>Total RN HPRD</th>
                <th>Contract %</th>
            </tr>
        </thead>
        <tbody>
            {''.join([f'''
            <tr>
                <td>{format_quarter_display(q)}</td>
                <td>{quarterly_data[q]['avg_census']:.1f}</td>
                <td>{quarterly_data[q]['total_hprd']:.2f}</td>
                <td>{state_comparisons.get(q, {}).get("total_hprd", "N/A") if isinstance(state_comparisons.get(q, {}).get("total_hprd"), (int, float)) else "N/A"}</td>
                <td>{quarterly_data[q]['rn_hprd']:.2f}</td>
                <td>{state_comparisons.get(q, {}).get("rn_hprd", "N/A") if isinstance(state_comparisons.get(q, {}).get("rn_hprd"), (int, float)) else "N/A"}</td>
                <td>{quarterly_data[q]['contract_pct']:.1f}%</td>
            </tr>''' for q in quarters if quarterly_data.get(q)])}
        </tbody>
    </table>
    
    <p style="font-size: 9pt; font-style: italic; margin-top: 20px;"><strong>Note:</strong> <strong>Total</strong> HPRD includes all staff (RN, LPN, nurse aides [CNA, Med Aide, NA Trn], plus RN/LPN admin and DON hours). Total RN HPRD = RN + RN Admin + RN DON. Contract % = contract hours as % of total nursing hours. This appendix is provided for reference only.{' It supplements the direct care staffing analysis in the main report.' if not include_total_staffing else ''}</p>
    
    {days_under_html if days_under_html else ''}
    
    {generate_state_staffing_section(state, macpac_standards, period_metrics)}
    </div>
    ''' if (inc.get('appendix', True) and (quarterly_data or period_metrics or daily_staffing)) else ''}
    
    {generate_citations_section(provnum)}
    {generate_entity_section(provnum)}
    
    <div class="footer-branding">
        <p><strong>320 Consulting</strong> | Nursing Home Staffing Analysis</p>
        <p>Report generated on {datetime.now().strftime('%B %d, %Y')} | Data source: CMS Payroll-Based Journal</p>
        <p class="contact">Contact: eric@320insight.com</p>
    </div>
    
    <!-- Page numbering for Word compatibility -->
    <style>
        @media print {{
            @page {{
                @bottom-right {{
                    content: "Page " counter(page) " of " counter(pages);
                    font-size: 9pt;
                    color: #666;
                    font-family: 'Calibri', 'Arial', sans-serif;
                }}
            }}
        }}
    </style>
    <!-- Note: For Microsoft Word, use Insert > Page Number > Bottom of Page > Plain Number 3 (right-aligned) after opening this file -->
</body>
</html>"""
    
    return html_content


def _replace_placeholders_in_docx(
    reference_docx_path: str,
    output_docx_path: str,
    replacements: Dict[str, str],
) -> None:
    """
    Copy reference .docx to output; replace placeholders only in text-bearing XML.
    Body, headers, footers: wording, font, capitalization, spacing, colors, page numbers
    stay exactly as in the reference. Only {{...}} tokens are replaced.
    """
    os.makedirs(os.path.dirname(output_docx_path) or ".", exist_ok=True)
    # XML parts that can contain visible text (body, headers, footers)
    text_parts = {"word/document.xml"}
    with zipfile.ZipFile(reference_docx_path, "r") as z_in:
        for info in z_in.infolist():
            if info.filename.startswith("word/header") and info.filename.endswith(".xml"):
                text_parts.add(info.filename)
            if info.filename.startswith("word/footer") and info.filename.endswith(".xml"):
                text_parts.add(info.filename)
    with zipfile.ZipFile(reference_docx_path, "r") as z_in:
        with zipfile.ZipFile(output_docx_path, "w", zipfile.ZIP_DEFLATED) as z_out:
            for info in z_in.infolist():
                if info.filename in text_parts:
                    xml = z_in.read(info.filename).decode("utf-8")
                    for placeholder, value in replacements.items():
                        xml = xml.replace(placeholder, _xml_escape(value))
                    z_out.writestr(info.filename, xml.encode("utf-8"))
                else:
                    z_out.writestr(info.filename, z_in.read(info.filename))


def generate_attorney_report_docx(
    output_docx_path: str,
    provnum: str,
    facility_name: str,
    city: str,
    state: str,
    start_date: datetime,
    end_date: datetime,
    key_dates: List[datetime],
    quarterly_data: Dict,
    state_comparisons: Dict,
    period_metrics: Optional[Dict] = None,
    daily_staffing: Optional[List[Dict]] = None,
    macpac_standards: Optional[Dict] = None,
    red_flags_history: Optional[List[Dict]] = None,
    include_sections: Optional[Dict[str, bool]] = None,
    reference_docx_path: Optional[str] = None,
    **kwargs
) -> str:
    """
    Generate Word (.docx) report that mirrors the HTML report: same sections (Executive Summary,
    Key Dates of Interest, Period Summary, Days Below State Minimum, Quarterly Staffing, Red Flags),
    same data. If reference_docx_path is provided, that doc is used for styles only (fonts, spacing,
    Table Grid) so the output looks like the brief; content is still built from report data.
    """
    start_date_str = start_date.strftime("%B %d, %Y")
    end_date_str = end_date.strftime("%B %d, %Y")
    review_period = f"{start_date.strftime('%B %Y')} - {end_date.strftime('%B %Y')}"
    report_date_str = datetime.now().strftime("%B %d, %Y")

    # Build Word doc that mirrors the HTML report (same sections, same data). If reference .docx provided, use it for styles only so output looks like the brief.
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        from docx.opc.constants import RELATIONSHIP_TYPE
    except ImportError as e:
        raise ImportError("python-docx is required for .docx output. Install with: pip install python-docx") from e

    def _set_cell_shading(cell, hex_fill: str) -> None:
        """Set table cell background (shading) to exact hex color. hex_fill e.g. 'D9E2F3'."""
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        for shd in tcPr.findall(qn("w:shd"), None):
            tcPr.remove(shd)
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), hex_fill)
        tcPr.append(shd)

    def _apply_table_shading(table, header_hex: str = "D9E2F3", even_hex: str = "F2F2F2", odd_hex: str = "FFFFFF", header_rows: int = 1) -> None:
        """Apply alternate row shading: header row(s) header_hex, then even/odd data rows."""
        for ri, row in enumerate(table.rows):
            fill = header_hex if ri < header_rows else (even_hex if (ri - header_rows) % 2 == 0 else odd_hex)
            for cell in row.cells:
                _set_cell_shading(cell, fill)

    def _add_hyperlink_paragraph(doc, text: str, url: str, style_name: str) -> None:
        """Add a paragraph containing a single hyperlink (exact color/underline from style)."""
        p = doc.add_paragraph(style=style_name)
        part = doc.part
        r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)
        hyperlink = OxmlElement("w:hyperlink")
        hyperlink.set(qn("r:id"), r_id)
        run_el = OxmlElement("w:r")
        rPr = OxmlElement("w:rPr")
        run_el.append(rPr)
        t_el = OxmlElement("w:t")
        t_el.text = text
        run_el.append(t_el)
        hyperlink.append(run_el)
        p._p.append(hyperlink)

    inc = include_sections if include_sections is not None else _default_include_sections()
    for k in _default_include_sections():
        if k not in inc:
            inc[k] = True
    case_mix_data = kwargs.get("case_mix_data") or []
    state_comparisons = kwargs.get("state_comparisons") or {}

    # Use reference doc for styles only (fonts, spacing, table shading) so output looks like the brief
    header_hex, even_hex, odd_hex = "D9E2F3", "F2F2F2", "FFFFFF"
    if reference_docx_path and os.path.isfile(reference_docx_path):
        doc = Document(reference_docx_path)
        # Read table shading from reference before clearing body (exact colors)
        try:
            for tbl in doc.tables:
                if tbl.rows:
                    tc0 = tbl.rows[0].cells[0]._tc
                    tcPr0 = tc0.tcPr
                    if tcPr0 is not None:
                        for shd in tcPr0.findall(qn("w:shd"), None):
                            fill = shd.get(qn("w:fill"))
                            if fill:
                                header_hex = fill.upper().replace("#", "")
                                break
                    if len(tbl.rows) > 1:
                        tc1 = tbl.rows[1].cells[0]._tc
                        tcPr1 = tc1.tcPr if tc1 is not None else None
                        if tcPr1 is not None:
                            for shd in tcPr1.findall(qn("w:shd"), None):
                                fill = shd.get(qn("w:fill"))
                                if fill:
                                    even_hex = fill.upper().replace("#", "")
                                    break
                    if len(tbl.rows) > 2:
                        tc2 = tbl.rows[2].cells[0]._tc
                        tcPr2 = tc2.tcPr if tc2 is not None else None
                        if tcPr2 is not None:
                            for shd in tcPr2.findall(qn("w:shd"), None):
                                fill = shd.get(qn("w:fill"))
                                if fill:
                                    odd_hex = fill.upper().replace("#", "")
                                    break
                break
        except Exception:
            pass
        body = doc.element.body
        for child in list(body):
            body.remove(child)
    else:
        doc = Document()
        # When no reference: set exact font and spacing so output is consistent (Calibri 11pt, spacing like reference)
        try:
            normal = doc.styles["Normal"]
            if hasattr(normal, "font") and normal.font:
                normal.font.name = "Calibri"
                normal.font.size = Pt(11)
            if hasattr(normal, "paragraph_format") and normal.paragraph_format:
                normal.paragraph_format.space_after = Pt(6)
                normal.paragraph_format.space_before = Pt(0)
            for hname, before_pt, after_pt in [("Heading 1", 12, 8), ("Heading 2", 10, 6), ("Heading 3", 8, 4)]:
                try:
                    h = doc.styles[hname]
                    if hasattr(h, "paragraph_format") and h.paragraph_format:
                        h.paragraph_format.space_before = Pt(before_pt)
                        h.paragraph_format.space_after = Pt(after_pt)
                except Exception:
                    pass
        except Exception:
            pass

    def _style(name: str) -> str:
        try:
            doc.styles[name]
            return name
        except Exception:
            return "Normal"

    # Match reference-berlin-315461-pbj320.docx structure and wording exactly
    title_style = _style("Title")
    h1_style = _style("Heading 1")
    h2_style = _style("Heading 2")
    h3_style = _style("Heading 3")
    normal_style = _style("Normal")
    list_style = _style("List Paragraph")
    table_style = "Table Grid" if any(s.name == "Table Grid" for s in doc.styles) else None

    # 1. Location line (reference: "Location: Berlin, NJ | Provider Number: 315461 | Resident Stay Period: ...")
    doc.add_paragraph(
        f"Location: {city}, {state} | Provider Number: {provnum} | Resident Stay Period: {start_date_str} to {end_date_str}",
        style=normal_style,
    )
    # 2. PBJ Summary (reference H1)
    doc.add_paragraph("PBJ Summary", style=h1_style)
    # 3. Intro (reference: "This report analyzes staffing levels at Berlin Rehabilitation...")
    doc.add_paragraph(
        f"This report analyzes staffing levels at {format_facility_name(facility_name)} during the period from {start_date_str} through {end_date_str}. The analysis focuses on potential understaffing issues that may have contributed to resident incidents during this period.",
        style=normal_style,
    )
    # 4. Key Dates of Interest (Resident Falls) - reference uses "Resident Falls"
    if inc.get("key_dates", True) and key_dates:
        doc.add_paragraph("Key Dates of Interest (Resident Falls)", style=normal_style)
        daily_lookup = {d["date"].date() if hasattr(d["date"], "date") else d["date"]: d for d in (daily_staffing or [])}
        for date in key_dates:
            date_only = date.date() if hasattr(date, "date") else date
            day_data = daily_lookup.get(date_only, {}) if daily_lookup else {}
            day_str = date.strftime("%A")
            line = f"{date.strftime('%B %d, %Y')} ({day_str}): Census: {round_half_up(day_data.get('census', 0), 0):.0f} | Direct Care HPRD: {round_half_up(day_data.get('direct_care_hprd', 0), 2):.2f} | RN HPRD: {round_half_up(day_data.get('rn_hprd', 0), 2):.2f}"
            doc.add_paragraph(line, style=list_style)
        # 5. Case-Mix Analysis bullets (reference)
        if inc.get("case_mix", True) and case_mix_data:
            doc.add_paragraph("Case-Mix Analysis", style=normal_style)
            latest = case_mix_data[-1] if case_mix_data else {}
            total_pct = latest.get("case_mix_total_pct") or latest.get("total_pct")
            rn_pct = latest.get("case_mix_rn_pct") or latest.get("rn_pct")
            harrington_total_pct = latest.get("harrington_total_pct")
            harrington_rn_pct = latest.get("harrington_rn_pct")
            if total_pct is not None or rn_pct is not None:
                doc.add_paragraph(f"{round_half_up(total_pct or 0, 1):.1f}% Case-Mix; {round_half_up(rn_pct or 0, 1):.1f}% RN Case-Mix", style=list_style)
            if harrington_total_pct is not None or harrington_rn_pct is not None:
                doc.add_paragraph(f"{round_half_up(harrington_total_pct or 0, 1):.1f}% Harrington Expected; {round_half_up(harrington_rn_pct or 0, 1):.1f}% Harrington RN Expected", style=list_style)
            doc.add_paragraph("Note: Excludes Admin/DON", style=normal_style)
        # 6. Key Dates: PBJ Data (H2) + table with 10 columns (reference)
        doc.add_paragraph("Key Dates: PBJ Data", style=h2_style)
        if daily_staffing:
            tbl = doc.add_table(rows=1 + len(key_dates), cols=10)
            if table_style:
                try:
                    tbl.style = table_style
                except Exception:
                    pass
            hdr = tbl.rows[0].cells
            hdr[0].text = "Date"
            hdr[1].text = "Census"
            hdr[2].text = "Direct Care\nHPRD"
            hdr[3].text = "RN\nHPRD"
            hdr[4].text = "LPN\nHPRD"
            hdr[5].text = "Nurse Aide\nHPRD"
            hdr[6].text = "RN\nHrs"
            hdr[7].text = "LPN\nHrs"
            hdr[8].text = "Nurse Aide\nHours"
            hdr[9].text = "Contract\n%"
            _apply_table_shading(tbl, header_hex=header_hex, even_hex=even_hex, odd_hex=odd_hex, header_rows=1)
            for i, date in enumerate(key_dates):
                date_only = date.date() if hasattr(date, "date") else date
                day_data = daily_lookup.get(date_only, {})
                row = tbl.rows[i + 1].cells
                row[0].text = f"{date.strftime('%b %d, %Y')}\n{date.strftime('%a')}"
                row[1].text = f"{round_half_up(day_data.get('census', 0), 0):.0f}"
                row[2].text = f"{round_half_up(day_data.get('direct_care_hprd', 0), 2):.2f}"
                row[3].text = f"{round_half_up(day_data.get('rn_hprd', 0), 2):.2f}"
                row[4].text = f"{round_half_up(day_data.get('lpn_hprd', 0), 2):.2f}"
                row[5].text = f"{round_half_up(day_data.get('cna_hprd', 0), 2):.2f}"
                rn_hrs = day_data.get("total_rn_hours") or (day_data.get("hrs_rn", 0) + day_data.get("hrs_rnadmin", 0) + day_data.get("hrs_rndon", 0))
                lpn_hrs = day_data.get("hrs_lpn") or (day_data.get("lpn_hprd", 0) * day_data.get("census", 1) or 0)
                na_hrs = day_data.get("hrs_nurse_aide") or day_data.get("hrs_cna") or (day_data.get("cna_hprd", 0) * day_data.get("census", 1) or 0)
                row[6].text = f"{round_half_up(rn_hrs, 1):.1f}"
                row[7].text = f"{round_half_up(lpn_hrs, 1):.1f}"
                row[8].text = f"{round_half_up(na_hrs, 1):.1f}"
                row[9].text = f"{round_half_up(day_data.get('contract_pct', 0), 1):.1f}%"
        doc.add_paragraph("", style=normal_style)
    # 7. Resident Stay Period (MM/DD/YYYY - MM/DD/YYYY) (H2) + table (reference)
    if inc.get("quarterly_staffing", True) or inc.get("period_summary", True):
        start_mdy = start_date.strftime("%m/%d/%Y")
        end_mdy = end_date.strftime("%m/%d/%Y")
        doc.add_paragraph(f"Resident Stay Period ({start_mdy} - {end_mdy})", style=h2_style)
        quarters = sorted(quarterly_data.keys()) if quarterly_data else []
        # Table: 2 header rows + data rows (reference)
        n_data = (1 if period_metrics else 0) + len(quarters)
        n_rows = 2 + n_data
        tbl = doc.add_table(rows=n_rows, cols=7)
        if table_style:
            try:
                tbl.style = table_style
            except Exception:
                pass
        hdr0 = tbl.rows[0].cells
        hdr0[0].text = "Period"
        hdr0[1].text = "Census"
        hdr0[2].text = "Direct Care HPRD"
        hdr0[3].text = "Direct Care HPRD"
        hdr0[4].text = "Direct RN HPRD"
        hdr0[5].text = "Direct RN HPRD"
        hdr0[6].text = "Contract %"
        hdr1 = tbl.rows[1].cells
        hdr1[0].text = "Period"
        hdr1[1].text = "Census"
        hdr1[2].text = "Facility"
        hdr1[3].text = state
        hdr1[4].text = "Facility"
        hdr1[5].text = state
        hdr1[6].text = "Contract %"
        _apply_table_shading(tbl, header_hex="D9E2F3", even_hex="F2F2F2", odd_hex="FFFFFF", header_rows=2)
        r = 2
        if period_metrics and n_data > 0:
            row = tbl.rows[r].cells
            row[0].text = f"Resident Stay Period\n{start_date.strftime('%b')} {start_date.day}, {start_date.year} - {end_date.strftime('%b')} {end_date.day}, {end_date.year}"
            row[1].text = f"{round_half_up(period_metrics.get('avg_census', 0), 1):.1f}"
            row[2].text = f"{round_half_up(period_metrics.get('direct_care_hprd', 0), 2):.2f}"
            row[3].text = "N/A"
            row[4].text = f"{round_half_up(period_metrics.get('direct_care_rn_hprd', 0), 2):.2f}"
            row[5].text = "N/A"
            row[6].text = f"{round_half_up(period_metrics.get('contract_pct', 0), 1):.1f}%"
            r += 1
        for q in quarters:
            if r >= n_rows:
                break
            d = quarterly_data.get(q) or {}
            st = state_comparisons.get(q) or {}
            row = tbl.rows[r].cells
            row[0].text = format_quarter_display(q)
            row[1].text = f"{round_half_up(d.get('avg_census', 0), 1):.1f}"
            row[2].text = f"{round_half_up(d.get('direct_care_hprd', 0), 2):.2f}"
            sc_hprd = st.get("direct_care_hprd")
            sr_hprd = st.get("direct_care_rn_hprd")
            row[3].text = f"{round_half_up(sc_hprd, 2):.2f}" if isinstance(sc_hprd, (int, float)) else "N/A"
            row[4].text = f"{round_half_up(d.get('direct_care_rn_hprd', 0), 2):.2f}"
            row[5].text = f"{round_half_up(sr_hprd, 2):.2f}" if isinstance(sr_hprd, (int, float)) else "N/A"
            row[6].text = f"{round_half_up(d.get('contract_pct', 0), 1):.1f}%"
            r += 1
        doc.add_paragraph("", style=normal_style)
    # 8. Case-Mix / Expected Staffing Analysis (H2) + intro + H3s + table (reference)
    if inc.get("case_mix", True) and case_mix_data:
        doc.add_paragraph("Case-Mix / Expected Staffing Analysis", style=h2_style)
        doc.add_paragraph(
            "Case-mix staffing analysis accounts for resident acuity based on CMS Case-Mix. Values below 100% indicate staffing below expected for the facility's acuity.",
            style=normal_style,
        )
        doc.add_paragraph("Case-Mix Analysis (Reported HPRD / Case-Mix)", style=h3_style)
        doc.add_paragraph("Harrington Expected Staffing", style=h3_style)
        doc.add_paragraph(
            "Harrington Total Expected HPRD = 3.48 + ((CMI - 0.62) / (3.84 - 0.62)) × (7.68 - 3.48). Harrington RN Expected HPRD = 0.55 + ((CMI - 0.62) / (3.84 - 0.62)) × (2.39 - 0.55). See: Nursing Home Guide to Adjusting Nurse Staffing for Resident Case-Mix. Note: Direct care excludes Admin/DON.",
            style=normal_style,
        )
        # Case-Mix table (reference: Quarter, Case-Mix Index, Total Nurse Case-Mix, % Case-Mix, RN Case-Mix, % RN Case-Mix, Nurse Aide Case-Mix, % Nurse Aide Case-Mix)
        tbl = doc.add_table(rows=1 + len(case_mix_data), cols=8)
        if table_style:
            try:
                tbl.style = table_style
            except Exception:
                pass
        hdr = tbl.rows[0].cells
        hdr[0].text = "Quarter"
        hdr[1].text = "Case-Mix Index"
        hdr[2].text = "Total Nurse\nCase-Mix"
        hdr[3].text = "% Case-Mix"
        hdr[4].text = "RN Case-Mix"
        hdr[5].text = "% RN Case-Mix"
        hdr[6].text = "Nurse Aide Case-Mix"
        hdr[7].text = "% Nurse Aide\nCase-Mix"
        _apply_table_shading(tbl, header_hex=header_hex, even_hex=even_hex, odd_hex=odd_hex, header_rows=1)
        for i, cm in enumerate(case_mix_data):
            row = tbl.rows[i + 1].cells
            row[0].text = format_quarter_display(cm.get("quarter", ""))
            row[1].text = f"{round_half_up(cm.get('cmi', 0), 2):.2f}"
            row[2].text = f"{round_half_up(cm.get('case_mix_total_hprd', 0), 2):.2f} HPRD"
            tc_hprd = cm.get("case_mix_total_hprd") or 0.01
            rn_hprd_cm = cm.get("case_mix_rn_hprd") or 0.01
            na_hprd_cm = cm.get("case_mix_na_hprd") or 0.01
            row[3].text = f"{round_half_up(cm.get('case_mix_total_pct', 0), 1):.1f}%\n{round_half_up(cm.get('direct_care_hprd', 0), 2):.2f} / {tc_hprd:.2f}" if cm.get("case_mix_total_hprd") else ""
            row[4].text = f"{round_half_up(cm.get('case_mix_rn_hprd', 0), 2):.2f} HPRD"
            row[5].text = f"{round_half_up(cm.get('case_mix_rn_pct', 0), 1):.1f}%\n{round_half_up(cm.get('direct_care_rn_hprd', 0), 2):.2f} / {rn_hprd_cm:.2f}" if cm.get("case_mix_rn_hprd") else ""
            row[6].text = f"{round_half_up(cm.get('case_mix_na_hprd', 0), 2):.2f} HPRD"
            row[7].text = f"{round_half_up(cm.get('case_mix_na_pct', 0), 1):.1f}%\n{round_half_up(cm.get('cna_hprd', 0), 2):.2f} / {na_hprd_cm:.2f}" if cm.get("case_mix_na_hprd") else ""
        doc.add_paragraph("", style=normal_style)
    # 9. Appendix (reference H2)
    if inc.get("appendix", True):
        doc.add_paragraph("Appendix", style=h2_style)
        doc.add_paragraph("Data Sources and Methodology", style=normal_style)
        doc.add_paragraph(
            "This report is based on data from the Centers for Medicare & Medicaid Services (CMS) Payroll-Based Journal (PBJ) system. Total nursing staff hours (RN, LPN, CNA, and other nursing staff), contract staff hours, and daily resident census. Hours Per Resident Day (HPRD) is calculated by dividing total nursing hours by total resident days for each quarter. State averages are calculated using weighted averages for all facilities in the state.",
            style=normal_style,
        )
        doc.add_paragraph("Resources", style=normal_style)
        _add_hyperlink_paragraph(doc, "CMS Provider Information", "https://data.cms.gov/provider-data/dataset/4pq5-n9py", normal_style)
        _add_hyperlink_paragraph(doc, "CMS Payroll-Based Journal", "https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing", normal_style)
        _add_hyperlink_paragraph(doc, "PBJ320", f"https://www.pbj320.com/provider/{provnum}", normal_style)
        _add_hyperlink_paragraph(doc, "CMS Care Compare", "https://www.medicare.gov/care-compare/", normal_style)
    # State requirements (data-driven from state_staffing_display.json + MACPAC)
    if macpac_standards and macpac_standards.get("min_staffing", 0) >= 2.0 and inc.get("appendix", True):
        cfg = _load_state_staffing_display().get(state.upper(), {})
        state_full_name = STATE_NAMES.get(state.upper(), state)
        is_estimated = cfg.get("is_estimated", False)
        title = f"Estimated {state_full_name} Staffing Requirements" if is_estimated else f"{state_full_name} State Staffing Requirements"
        doc.add_paragraph(title, style=h2_style)
        min_hprd = macpac_standards.get("min_staffing", 0)
        doc.add_paragraph(f"{title}: {round_half_up(min_hprd, 2):.2f} HPRD (Source: MACPAC - Medicaid and CHIP Payment and Access Commission)", style=normal_style)
        section_title = cfg.get("section_title")
        if section_title:
            doc.add_paragraph(section_title, style=normal_style)
    doc.add_paragraph(f"Report generated {report_date_str}", style=normal_style)

    os.makedirs(os.path.dirname(output_docx_path) or ".", exist_ok=True)
    doc.save(output_docx_path)
    return output_docx_path


def main():
    """Main function to generate the report. This is a library - use generate_facility_report_attorney.py instead."""
    print("This is a library file. Please use generate_facility_report_attorney.py to generate reports.")
    print("Or run: python generate_facility_report_attorney.py [provnum] [start_date] [end_date] [key_dates]")
    return
    
    # Get MACPAC state standards (do this early so we can use it for days under minimum calculation)
    print(f"\nGetting MACPAC state standards for {facility_info['state']}...")
    macpac_standards = get_macpac_state_standards(facility_info['state'])
    if macpac_standards:
        print(f"  Found: {macpac_standards['display_text']}")
    else:
        print(f"  No MACPAC standards found")
    
    # Filter to quarters that overlap with the date range
    start_quarter = f"{start_date.year}Q{(start_date.month - 1) // 3 + 1}"
    end_quarter = f"{end_date.year}Q{(end_date.month - 1) // 3 + 1}"
    
    # Filter to quarters in the range
    all_quarters = sorted(df['CY_Qtr'].unique())
    quarters_in_range = [q for q in all_quarters if q >= start_quarter and q <= end_quarter]
    
    filtered_df = df[df['CY_Qtr'].isin(quarters_in_range)].copy()
    
    print(f"Quarters in date range: {quarters_in_range}")
    print(f"Total records: {len(filtered_df)}")
    
    if filtered_df.empty:
        print(f"Warning: No data found for the specified date range, using all available data")
        filtered_df = df.copy()
    
    # Ensure filtered_df is a DataFrame (not Series)
    if not isinstance(filtered_df, pd.DataFrame):
        filtered_df = pd.DataFrame(filtered_df)
    
    # Get unique quarters in the date range
    quarters = sorted(filtered_df['CY_Qtr'].unique())
    print(f"Quarters to analyze: {quarters}")
    
    # Calculate quarterly metrics
    quarterly_data = {}
    state_comparisons = {}
    
    for quarter in quarters:
        print(f"\nProcessing {quarter}...")
        
        # Facility metrics - use full df for quarterly calculations (not filtered_df)
        q_metrics = calculate_quarterly_metrics(df, quarter)
        if q_metrics:
            quarterly_data[quarter] = q_metrics
            print(f"  Facility HPRD: {q_metrics['total_hprd']:.2f}, RN HPRD: {q_metrics['rn_hprd']:.2f}")
        
        # State averages
        print(f"  Getting state averages for {quarter}...")
        state_avg = get_state_averages(facility_info['state'], quarter)
        if state_avg:
            state_comparisons[quarter] = state_avg
            print(f"    State avg HPRD: {state_avg['total_hprd']:.2f}, RN HPRD: {state_avg['rn_hprd']:.2f}")
        else:
            print(f"    No state data found")
    
    # Calculate period-specific metrics (exact date range)
    print(f"\nCalculating period-specific metrics for {start_date.date()} to {end_date.date()}...")
    period_metrics = calculate_period_metrics(df, start_date, end_date)
    if period_metrics:
        print(f"  Period Total HPRD: {period_metrics['total_hprd']:.2f}")
        print(f"  Period Direct Care HPRD: {period_metrics.get('direct_care_hprd', 0):.2f}")
        print(f"  Period RN HPRD: {period_metrics['rn_hprd']:.2f}")
        print(f"  Period Direct Care RN HPRD: {period_metrics['direct_care_rn_hprd']:.2f}")
    else:
        print(f"  Could not calculate period metrics")
    
    # Calculate days under state minimum
    if macpac_standards and macpac_standards.get('min_staffing', 0) > 0:
        print(f"\nCalculating days under state minimum ({macpac_standards['min_staffing']:.2f} HPRD)...")
        days_under = calculate_days_under_state_minimum(df, start_date, end_date, macpac_standards['min_staffing'])
        if period_metrics:
            period_metrics.update(days_under)
        print(f"  Total days: {days_under['total_days']:,}")
        print(f"  Days under minimum (Total HPRD): {days_under['days_under_minimum_total']:,} ({days_under['percentage_under_total']:.1f}%)")
        print(f"  Days under minimum (Direct Care HPRD): {days_under['days_under_minimum_direct']:,} ({days_under['percentage_under_direct']:.1f}%)")
    
    # Get daily staffing for key dates
    print(f"\nGetting daily staffing for key dates...")
    daily_staffing = []
    for key_date in key_dates:
        print(f"  Getting data for {key_date.strftime('%B %d, %Y')}...")
        day_data = get_daily_staffing(df, key_date)
        if day_data:
            daily_staffing.append(day_data)
            print(f"    Census: {day_data['census']:.0f}, Total HPRD: {day_data['total_hprd']:.2f}, RN HPRD: {day_data['rn_hprd']:.2f}")
        else:
            print(f"    No data found for this date")
    
    # Load provider info data for historical context
    print(f"\nLoading provider info data for historical context...")
    provider_info_df = load_provider_info_data(provnum)
    red_flags_history = []
    case_mix_data = []
    
    if not provider_info_df.empty:
        print(f"  Found {len(provider_info_df)} provider info records")
        
        # Extract red flags history
        print(f"  Extracting red flags history...")
        red_flags_history = extract_red_flags_history(provider_info_df, start_date, end_date)
        print(f"    Found {len(red_flags_history)} periods with red flags")
        
        # Extract case-mix data (use date range only so all quarters with case-mix in provider info are included;
        # do not restrict to PBJ quarters, since Harrington being Q1/Q2 should not limit case-mix analysis)
        print(f"  Extracting case-mix adjusted staffing data...")
        case_mix_data = extract_case_mix_data(provider_info_df, start_date, end_date, quarters_in_range=None)
        print(f"    Found {len(case_mix_data)} periods with case-mix data")
    else:
        print(f"  No provider info data found")
    
    # Generate report
    print("\nGenerating HTML report...")
    html_report = generate_attorney_report(
        provnum=provnum,
        facility_name=facility_info['name'],
        city=facility_info['city'],
        state=facility_info['state'],
        start_date=start_date,
        end_date=end_date,
        key_dates=key_dates,
        quarterly_data=quarterly_data,
        state_comparisons=state_comparisons,
        period_metrics=period_metrics,
        daily_staffing=daily_staffing,
        macpac_standards=macpac_standards,
        red_flags_history=red_flags_history,
        case_mix_data=case_mix_data,
        pbj_df=df
    )
    
    # Save report
    facility_name_safe = facility_info['name'].replace(' ', '_').replace('&', 'and').replace(',', '').replace('.', '')
    output_filename = f"PBJ_Report_{provnum}_{facility_name_safe}.html"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"\nReport generated successfully!")
    print(f"Saved as: {output_filename}")
    print(f"Open this file in your web browser to view the formatted report.")

if __name__ == "__main__":
    main()

