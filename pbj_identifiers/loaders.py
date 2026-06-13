"""
Data loaders for facilities, states, and entities.

These functions load data from existing CSV files and return
Python dicts that conform to the JSON schemas.

All loaders are read-only and normalize identifiers.
"""

import os
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime

from .validators import (
    normalize_ccn, 
    normalize_state_code, 
    normalize_entity_id, 
    validate_state_code,
    validate_json_against_schema
)
from .urls import generate_dashboard_url, generate_cms_url

# Schema validation flag (can be disabled if jsonschema not available)
SCHEMA_VALIDATION_ENABLED = True
try:
    import jsonschema
except ImportError:
    SCHEMA_VALIDATION_ENABLED = False


def _find_provider_info_file() -> Optional[str]:
    """Find the latest provider info file."""
    # Try provider_info_combined.csv first (preferred)
    if os.path.exists('provider_info_combined.csv'):
        return 'provider_info_combined.csv'
    
    # Try provider_info directory
    from utils.file_finder import find_latest_provider_info
    return find_latest_provider_info()


def _find_macpac_file() -> Optional[str]:
    """Find the MACPAC state standards file."""
    possible_paths = [
        'macpac/macpac_state_standards.csv',
        'macpac_state_standards.csv',
        'macpac_state_standards_clean.csv',
        'pbj_lite/macpac_state_standards_clean.csv',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def load_facility(ccn: str) -> Dict:
    """
    Load facility data and return as canonical JSON structure.
    
    Args:
        ccn: Facility CCN (will be normalized)
        
    Returns:
        Dict conforming to facility.schema.json
        
    Raises:
        ValueError: If required fields are missing
        FileNotFoundError: If data files not found
    """
    ccn_norm = normalize_ccn(ccn)
    
    # Load provider info
    provider_file = _find_provider_info_file()
    if not provider_file:
        raise FileNotFoundError("Provider info file not found. Expected: provider_info_combined.csv or files in provider_info/")
    
    df = pd.read_csv(provider_file, low_memory=False, dtype={'ccn': str})
    
    # Normalize CCN column (handle various column names)
    ccn_col = None
    for col in ['ccn', 'CCN', 'CMS Certification Number (CCN)', 'PROVNUM', 'Provider_CCN']:
        if col in df.columns:
            ccn_col = col
            df[ccn_col] = df[ccn_col].astype(str).str.zfill(6)
            break
    
    if not ccn_col:
        raise ValueError("Could not find CCN column in provider info file")
    
    # Find facility
    facility_row = df[df[ccn_col] == ccn_norm]
    if facility_row.empty:
        raise ValueError(f"Facility {ccn_norm} not found in provider info")
    
    row = facility_row.iloc[0]
    
    # Extract required fields
    name_cols = ['Provider Name', 'Provider_Name', 'provider_name', 'PROVNAME', 'Facility Name']
    name = None
    for col in name_cols:
        if col in row.index and pd.notna(row.get(col)):
            name = str(row.get(col)).strip()
            break
    
    if not name:
        raise ValueError(f"Facility name not found for CCN {ccn_norm}")
    
    # Extract state
    state_cols = ['State', 'STATE', 'state', 'State Code', 'Provider State']
    state = None
    for col in state_cols:
        if col in row.index and pd.notna(row.get(col)):
            state_raw = str(row.get(col)).strip()
            state = normalize_state_code(state_raw)
            break
    
    if not state:
        raise ValueError(f"State not found for facility {ccn_norm}")
    
    # Extract optional fields
    city_cols = ['City', 'CITY', 'city', 'City/Town', 'Provider City']
    city = None
    for col in city_cols:
        if col in row.index and pd.notna(row.get(col)):
            city = str(row.get(col)).strip()
            break
    
    county_cols = ['County', 'COUNTY', 'county', 'County Name', 'COUNTY_NAME']
    county = None
    for col in county_cols:
        if col in row.index and pd.notna(row.get(col)):
            county = str(row.get(col)).strip()
            break
    
    # Extract entity info
    entity_id = None
    entity_name = None
    
    chain_id_cols = ['Chain ID', 'chain_id', 'Chain_ID', 'Entity ID', 'entity_id', 'affiliated_entity_id']
    for col in chain_id_cols:
        if col in row.index and pd.notna(row.get(col)):
            entity_id = normalize_entity_id(row.get(col))
            break
    
    if entity_id:
        chain_name_cols = ['Chain Name', 'chain_name', 'Chain_Name', 'Entity Name', 'entity_name', 'affiliated_entity_name']
        for col in chain_name_cols:
            if col in row.index and pd.notna(row.get(col)):
                entity_name = str(row.get(col)).strip()
                if entity_name.upper() in ['NAN', 'NONE', '']:
                    entity_name = None
                break
    
    # Build entity object
    entity_obj = None
    if entity_id:
        entity_obj = {
            "id": entity_id,
            "name": entity_name
        }
    
    # Generate URLs
    links: Dict[str, Optional[str]] = {
        "dashboard": generate_dashboard_url(facility=ccn_norm),
        "state_dashboard": generate_dashboard_url(state=state),
        "cms_care_compare": generate_cms_url(ccn_norm, state)
    }
    
    if entity_id:
        links["entity_dashboard"] = generate_dashboard_url(entity=entity_id)
    else:
        links["entity_dashboard"] = None
    
    # Build result
    result = {
        "facility": {
            "ccn": ccn_norm,
            "name": name,
            "state": state,
            "links": links
        }
    }
    
    # Add optional fields
    if city:
        result["facility"]["city"] = city
    if county:
        result["facility"]["county"] = county
    if entity_obj:
        result["facility"]["entity"] = entity_obj
    else:
        result["facility"]["entity"] = None
    
    # Add metadata
    result["facility"]["metadata"] = {
        "last_updated": datetime.utcnow().isoformat() + "Z",
        "data_source": "PBJ + Provider Info"
    }
    
    # Validate against schema if enabled
    if SCHEMA_VALIDATION_ENABLED:
        is_valid, error = validate_json_against_schema(result, "facility")
        if not is_valid:
            raise ValueError(f"Facility JSON does not conform to schema: {error}")
    
    return result


def load_state(state_code: str) -> Dict:
    """
    Load state data and return as canonical JSON structure.
    
    Args:
        state_code: State code (will be normalized)
        
    Returns:
        Dict conforming to state.schema.json
        
    Raises:
        ValueError: If state code is invalid or required fields missing
        FileNotFoundError: If MACPAC file not found
    """
    state_norm = normalize_state_code(state_code)
    
    if not validate_state_code(state_norm):
        raise ValueError(f"Invalid state code: {state_code}")
    
    # State name mapping
    state_names = {
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
    
    state_name = state_names.get(state_norm, state_norm)
    
    # Load MACPAC standards
    macpac_file = _find_macpac_file()
    macpac_standard = None
    
    if macpac_file:
        try:
            df = pd.read_csv(macpac_file, low_memory=False)
            
            # Find state in MACPAC file
            # File has "State" column with full names
            state_row = df[df['State'].str.upper() == state_name.upper()]
            
            if state_row.empty:
                # Try without "State" suffix
                state_row = df[df['State'].str.upper().str.replace(' STATE', '') == state_name.upper()]
            
            if not state_row.empty:
                row = state_row.iloc[0]
                requirements = str(row.get('Total_Estimated_Staffing_Requirements', ''))
                
                # Parse HPRD from text like "3.56 HPRD" or "2.56—3.86 HPRD"
                import re
                hprd_match = re.search(r'([\d.]+)', requirements)
                min_hprd = None
                max_hprd = None
                
                if hprd_match:
                    # Check for range (— or -)
                    if '—' in requirements or '-' in requirements:
                        range_match = re.search(r'([\d.]+)[—\-]+([\d.]+)', requirements)
                        if range_match:
                            min_hprd = float(range_match.group(1))
                            max_hprd = float(range_match.group(2))
                    else:
                        min_hprd = float(hprd_match.group(1))
                
                if min_hprd is not None:
                    macpac_standard = {
                        "min_hprd": min_hprd,
                        "max_hprd": max_hprd,
                        "display_text": requirements,
                        "source": "MACPAC State Standards"
                    }
        except Exception as e:
            # Don't fail if MACPAC loading fails - it's optional
            pass
    
    # Build result
    result = {
        "state": {
            "code": state_norm,
            "name": state_name,
            "links": {
                "dashboard": generate_dashboard_url(state=state_norm)
            }
        }
    }
    
    if macpac_standard:
        result["state"]["macpac_standard"] = macpac_standard
    else:
        result["state"]["macpac_standard"] = None
    
    # Validate against schema if enabled
    if SCHEMA_VALIDATION_ENABLED:
        is_valid, error = validate_json_against_schema(result, "state")
        if not is_valid:
            raise ValueError(f"State JSON does not conform to schema: {error}")
    
    return result


def load_entity(entity_id: str) -> Dict:
    """
    Load entity data and return as canonical JSON structure.
    
    NOTE: This is a partial implementation. Entity data is scattered
    across provider info files. This function loads what it can find.
    
    Args:
        entity_id: Entity ID (will be normalized)
        
    Returns:
        Dict conforming to entity.schema.json
        
    Raises:
        ValueError: If entity ID is invalid
        FileNotFoundError: If data files not found
    """
    entity_norm = normalize_entity_id(entity_id)
    
    if not entity_norm:
        raise ValueError(f"Invalid entity ID: {entity_id}")
    
    # Load provider info to find facilities in this entity
    provider_file = _find_provider_info_file()
    if not provider_file:
        raise FileNotFoundError("Provider info file not found")
    
    df = pd.read_csv(provider_file, low_memory=False, dtype={'ccn': str})
    
    # Find chain ID column
    chain_id_col = None
    for col in ['Chain ID', 'chain_id', 'Chain_ID', 'Entity ID', 'entity_id', 'affiliated_entity_id']:
        if col in df.columns:
            chain_id_col = col
            break
    
    if not chain_id_col:
        raise ValueError("Chain ID column not found in provider info file")
    
    # Convert entity ID to float for comparison (handles "453.0" format)
    try:
        entity_id_float = float(entity_norm)
    except ValueError:
        raise ValueError(f"Entity ID must be numeric: {entity_id}")
    
    # Filter facilities by entity
    entity_facilities = df[
        pd.to_numeric(df[chain_id_col], errors='coerce') == entity_id_float
    ]
    
    if entity_facilities.empty:
        raise ValueError(f"No facilities found for entity ID: {entity_norm}")
    
    # Get entity name
    entity_name = None
    chain_name_cols = ['Chain Name', 'chain_name', 'Chain_Name', 'Entity Name', 'entity_name', 'affiliated_entity_name']
    for col in chain_name_cols:
        if col in entity_facilities.columns and not entity_facilities[col].isna().all():
            entity_name = str(entity_facilities[col].iloc[0])
            if entity_name.upper() not in ['NAN', 'NONE', '']:
                break
    
    # Build facilities list
    facilities = []
    facility_ccns = []
    
    ccn_col = None
    for col in ['ccn', 'CCN', 'CMS Certification Number (CCN)', 'PROVNUM']:
        if col in entity_facilities.columns:
            ccn_col = col
            break
    
    name_cols = ['Provider Name', 'Provider_Name', 'provider_name', 'PROVNAME']
    state_cols = ['State', 'STATE', 'state', 'State Code']
    
    for _, row in entity_facilities.iterrows():
        if ccn_col and pd.notna(row.get(ccn_col)):
            ccn = normalize_ccn(str(row.get(ccn_col)))
            facility_ccns.append(ccn)
            
            name = None
            for col in name_cols:
                if col in row.index and pd.notna(row.get(col)):
                    name = str(row.get(col)).strip()
                    break
            
            state = None
            for col in state_cols:
                if col in row.index and pd.notna(row.get(col)):
                    state_raw = str(row.get(col)).strip()
                    state = normalize_state_code(state_raw)
                    break
            
            if name and state:
                facilities.append({
                    "ccn": ccn,
                    "name": name,
                    "state": state
                })
    
    # Build result
    result = {
        "entity": {
            "id": entity_norm,
            "links": {
                "dashboard": generate_dashboard_url(entity=entity_norm)
            }
        }
    }
    
    if entity_name:
        result["entity"]["name"] = entity_name
    
    result["entity"]["facilities"] = facilities
    result["entity"]["facility_ccns"] = facility_ccns
    
    # Add metadata
    states = list(set([f["state"] for f in facilities]))
    result["entity"]["metadata"] = {
        "facility_count": len(facilities),
        "states": sorted(states),
        "last_updated": datetime.utcnow().isoformat() + "Z"
    }
    
    # Validate against schema if enabled
    if SCHEMA_VALIDATION_ENABLED:
        is_valid, error = validate_json_against_schema(result, "entity")
        if not is_valid:
            raise ValueError(f"Entity JSON does not conform to schema: {error}")
    
    return result
