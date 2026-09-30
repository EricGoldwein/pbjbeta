"""Read-only source inventory. Inventory coverage is distinct from ACTIVE lifecycle."""
from cms_source_registry import get_registry


def source_family_inventory(control, snapshots=(), catalog=None):
    datasets = {r['dataset_id']: r for r in control.get('datasets', [])}
    probes = {r.get('source_id'): r for r in snapshots}
    rows = []
    for source in get_registry():
        state = datasets.get(source.source_id, {})
        active = state.get('active') or {}
        probe = probes.get(source.source_id, {})
        rows.append({
            'source': 'Health Deficiencies (Health Citations)' if source.source_id == 'cms.health_citations' else source.human_name, 'source_id': source.source_id,
            'governance_status': 'GOVERNED' if state else 'REGISTERED_ONLY',
            'implementation_status': source.automation_maturity.value.upper(),
            'evidence': active.get('active_release_id') or probe.get('local_release') or 'No governed ACTIVE release',
            'health': state.get('health') or probe.get('status') or 'UNKNOWN',
            'next_action': 'Review source workflow' if state else 'Review registry evidence',
            'workflow_source_id': source.source_id,
        })
    penalties = next((r for r in (catalog or {}).get('datasets', []) if 'penalties' in str(r.get('title', '')).lower()), {})
    rows.extend([
        {'source': 'Penalties', 'source_id': 'penalties', 'governance_status': 'PARTIAL',
         'implementation_status': 'MERGED', 'evidence': ('CMS released ' + str(penalties['released'])) if penalties.get('released') else 'Partial merged implementation; no governed ACTIVE lifecycle',
         'health': 'NOT_GOVERNED', 'next_action': 'Complete governed validation and lifecycle', 'workflow_source_id': None},
        {'source': 'HCRIS', 'source_id': 'hcris', 'governance_status': 'AUDITED_ONLY',
         'implementation_status': 'UNMERGED', 'evidence': 'Audited pilot only',
         'health': 'PILOT_ONLY', 'next_action': 'Review pilot before merge', 'workflow_source_id': None},
        {'source': 'NPI / NPPES', 'source_id': 'npi_nppes', 'governance_status': 'UNAVAILABLE',
         'implementation_status': 'NO_GOVERNED_PIPELINE', 'evidence': 'No governed release evidence',
         'health': 'UNAVAILABLE', 'next_action': 'Establish a governed pipeline', 'workflow_source_id': None},
    ])
    return rows
