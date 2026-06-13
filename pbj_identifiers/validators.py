"""
Identifier normalization and validation functions.

These functions ensure consistent identifier formats across the system.
All identifiers are normalized before use to prevent lookup failures.
"""

from typing import Optional, Dict, Any
import pandas as pd
import json
import os


def _load_schema(schema_name: str) -> Dict[str, Any]:
    """Load a JSON schema file."""
    schema_path = os.path.join(os.path.dirname(__file__), 'schemas', f'{schema_name}.schema.json')
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    with open(schema_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_json_against_schema(data: Dict[str, Any], schema_name: str) -> tuple:
    """
    Validate JSON data against a schema.
    
    Args:
        data: JSON data to validate
        schema_name: Name of schema file (without .schema.json extension)
        
    Returns:
        Tuple of (is_valid, error_message)
        
    Examples:
        >>> facility_data = {"facility": {"ccn": "335513", "name": "Test", "state": "NY", "links": {...}}}
        >>> is_valid, error = validate_json_against_schema(facility_data, "facility")
        >>> is_valid
        True
    """
    try:
        import jsonschema
    except ImportError:
        return False, "jsonschema library not installed. Install with: pip install jsonschema"
    
    try:
        schema = _load_schema(schema_name)
        jsonschema.validate(instance=data, schema=schema)
        return True, None
    except jsonschema.ValidationError as e:
        error_msg = f"Validation error at {'.'.join(str(p) for p in e.path)}: {e.message}"
        return False, error_msg
    except Exception as e:
        return False, f"Schema validation error: {str(e)}"


def normalize_ccn(ccn: str) -> str:
    """
    Normalize CCN to 6-character zero-padded uppercase string.
    
    Args:
        ccn: Facility identifier (can be any format)
        
    Returns:
        6-character zero-padded uppercase string
        
    Examples:
        >>> normalize_ccn("335513")
        '335513'
        >>> normalize_ccn("15009")
        '015009'
        >>> normalize_ccn(" 335513 ")
        '335513'
        >>> normalize_ccn("abc123")
        'ABC123'
    """
    if not ccn:
        raise ValueError("CCN cannot be empty")
    
    ccn = str(ccn).strip().upper()
    # Remove any non-alphanumeric characters
    ccn = ''.join(c for c in ccn if c.isalnum())
    
    if not ccn:
        raise ValueError("CCN must contain at least one alphanumeric character")
    
    # Zero-pad to 6 characters
    return ccn.zfill(6)


def validate_ccn(ccn: str) -> bool:
    """
    Validate CCN format.
    
    Args:
        ccn: Facility identifier to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> validate_ccn("335513")
        True
        >>> validate_ccn("015009")
        True
        >>> validate_ccn("12345")  # Too short
        False
        >>> validate_ccn("1234567")  # Too long
        False
    """
    try:
        normalized = normalize_ccn(ccn)
        return len(normalized) == 6 and normalized.isalnum()
    except (ValueError, TypeError):
        return False


def normalize_state_code(state: str) -> str:
    """
    Normalize state code to 2-letter uppercase.
    
    Args:
        state: State identifier (can be full name or code)
        
    Returns:
        2-letter uppercase state code
        
    Examples:
        >>> normalize_state_code("NY")
        'NY'
        >>> normalize_state_code("new york")
        'NE'
        >>> normalize_state_code("  CA  ")
        'CA'
    """
    if not state:
        raise ValueError("State code cannot be empty")
    
    state = str(state).strip().upper()
    # Remove any whitespace and non-alphabetic characters
    state = ''.join(c for c in state if c.isalpha())
    
    if not state:
        raise ValueError("State code must contain at least one alphabetic character")
    
    # Take first 2 letters (handles both "NY" and "New York" -> "NE")
    return state[:2]


def validate_state_code(state: str) -> bool:
    """
    Validate state code against standard US state codes.
    
    Args:
        state: State code to validate
        
    Returns:
        True if valid US state code, False otherwise
        
    Examples:
        >>> validate_state_code("NY")
        True
        >>> validate_state_code("CA")
        True
        >>> validate_state_code("XX")
        False
    """
    US_STATE_CODES = {
        'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
        'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
        'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
        'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
        'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
    }
    
    try:
        normalized = normalize_state_code(state)
        return normalized in US_STATE_CODES
    except (ValueError, TypeError):
        return False


def normalize_entity_id(entity_id) -> Optional[str]:
    """
    Normalize entity ID to numeric string (no decimals).
    
    Args:
        entity_id: Entity/chain identifier (can be string, number, or None)
        
    Returns:
        Numeric string without decimals, or None if invalid/empty
        
    Examples:
        >>> normalize_entity_id("217")
        '217'
        >>> normalize_entity_id("453.0")
        '453'
        >>> normalize_entity_id(608)
        '608'
        >>> normalize_entity_id(None)
        None
        >>> normalize_entity_id("")
        None
    """
    if entity_id is None:
        return None
    
    # Handle pandas NaN
    if pd.isna(entity_id):
        return None
    
    entity_id = str(entity_id).strip()
    
    # Remove decimal point and trailing zeros (e.g., "453.0" -> "453")
    if '.' in entity_id:
        entity_id = entity_id.split('.')[0]
    
    # Remove any non-numeric characters
    entity_id = ''.join(c for c in entity_id if c.isdigit())
    
    return entity_id if entity_id else None


def validate_entity_id(entity_id) -> bool:
    """
    Validate entity ID format.
    
    Args:
        entity_id: Entity identifier to validate
        
    Returns:
        True if valid, False otherwise
        
    Examples:
        >>> validate_entity_id("217")
        True
        >>> validate_entity_id("453.0")
        True
        >>> validate_entity_id(None)
        False
        >>> validate_entity_id("abc")
        False
    """
    normalized = normalize_entity_id(entity_id)
    return normalized is not None and normalized.isdigit()
