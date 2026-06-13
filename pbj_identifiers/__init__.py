"""
PBJ Canonical Identifier Layer

This module provides a read-only canonical identifier layer for PBJ facilities,
states, and entities. It serves as the single source of truth for:
- Identifier normalization and validation
- JSON schema definitions
- URL generation (internal PBJ and external CMS)
- Data loading with schema validation

This is Phase 1 implementation - read-only, additive only.
No existing functionality is modified.
"""

__version__ = "1.0.0"

from .validators import (
    normalize_ccn,
    validate_ccn,
    normalize_state_code,
    validate_state_code,
    normalize_entity_id,
    validate_entity_id,
    validate_json_against_schema,
)

from .loaders import (
    load_facility,
    load_state,
    load_entity,
)

from .urls import (
    generate_dashboard_url,
    generate_cms_url,
)

__all__ = [
    "normalize_ccn",
    "validate_ccn",
    "normalize_state_code",
    "validate_state_code",
    "normalize_entity_id",
    "validate_entity_id",
    "validate_json_against_schema",
    "load_facility",
    "load_state",
    "load_entity",
    "generate_dashboard_url",
    "generate_cms_url",
]
