"""Read-only projection of canonical identities, control state and observations.

No Git merge state or lifecycle is inferred from automation labels/filenames.
"""
from cms_source_registry import ADJACENT_SOURCE_IDENTITIES, get_registry
from source_evidence import adjacent_source_observations


def source_family_inventory(control, snapshots=(), catalog=None, observations=None):
    observations = adjacent_source_observations() if observations is None else observations
    datasets = {r['dataset_id']: r for r in control.get('datasets', [])}
    probes = {r.get('source_id'): r for r in snapshots}
    catalog_rows = {r.get('stable_id'): r for r in (catalog or {}).get('datasets', [])}
    descriptors = [{"source_id": r.source_id, "human_name": r.human_name,
                    "cms_dataset_id": r.cms_dataset_id, "adapter": r.acquisition_implementation}
                   for r in get_registry()]
    descriptors.extend(ADJACENT_SOURCE_IDENTITIES)
    rows = []
    for source in descriptors:
        source_id = source['source_id']
        state = datasets.get(source_id, {})
        active, pending = state.get('active') or {}, state.get('pending') or {}
        probe = probes.get(source_id, {})
        observation = (observations or {}).get(source_id, {})
        published = catalog_rows.get(source.get('cms_dataset_id'), {})
        validation = pending.get('validation') or {}
        lifecycle = pending.get('state')
        governed = bool(active or pending)
        evidence = active.get('active_release_id') or pending.get('release_id') or probe.get('local_release') or observation.get('evidence')
        if not evidence and published.get('released'):
            evidence = 'CMS released ' + str(published['released']) + '; catalog observation only'
        if lifecycle == 'VALIDATED' and validation.get('status') == 'PASS':
            next_action = 'Review validated candidate; explicit approval required'
        elif lifecycle == 'FAILED' or validation.get('status') == 'FAIL':
            next_action = 'Inspect failed validation receipt'
        elif lifecycle:
            next_action = 'Validate acquired source' if lifecycle == 'ACQUIRED' else 'Acquire detected source'
        elif active:
            next_action = 'Review ACTIVE release health'
        elif observation:
            next_action = observation.get('next_action') or 'Review retained evidence'
        else:
            next_action = 'Verify source evidence before acquisition'
        rows.append({
            'source': 'Health Deficiencies (Health Citations)' if source_id == 'cms.health_citations' else source['human_name'],
            'source_id': source_id,
            'governance_status': 'GOVERNED' if governed else ('REGISTERED_ONLY' if source.get('adapter') else 'NO_GOVERNED_RELEASE'),
            'implementation_status': lifecycle or ('ACTIVE' if active else ('ADAPTER_REGISTERED' if source.get('adapter') else 'NO_REGISTERED_ADAPTER')),
            'evidence': evidence or 'No governed release evidence',
            'health': ('VALIDATION_' + validation['status']) if validation.get('status') else state.get('health') or probe.get('status') or observation.get('health') or 'UNKNOWN',
            'next_action': next_action,
            'workflow_source_id': source_id if source.get('adapter') or governed else None,
        })
    return rows
