"""Read-only review context shared with Sources readiness cards."""
from website_release_readiness import FAMILIES, family_readiness


def website_review_context(family, release_id, control, *, root=None):
    context = family_readiness(family, control, release_id=release_id, root=root)
    if context['manifest']:
        import copy
        context['original_manifest'] = context['manifest']
        context['manifest'] = copy.deepcopy(context['manifest'])
        names = {'ownership_release_policy': 'Ownership release policy', 'ownership_bridge_lookup': 'Facility ownership bridge',
                 'enrollment_release_artifact': 'Enrollment source for website builds', 'sff_facilities_json': 'SFF facility dataset',
                 'sff_public_json': 'Public SFF and candidate list', 'search_index': 'Website search index',
                 'owner_profile_index': 'Owner profile index'}
        for artifact in context['manifest'].get('artifacts', []):
            artifact['display_name'] = names.get(artifact.get('destination_id'), str(artifact.get('destination_id') or 'Website artifact').replace('_', ' ').title())
            artifact['display_category'] = {'commit_destination': 'Git committed', 'shared_derived': 'Git committed',
                                           'validation_only': 'Validation only', 'deploy_generated': 'Deploy generated'}.get(artifact.get('publication_class'), 'Category not recorded')
            artifact['display_change'] = 'generated' if artifact.get('publication_class') == 'deploy_generated' else artifact.get('publication_action') or 'No change recorded'
        required = {source['source_id'] for source in context['source_evidence']}
        context['other_inputs'] = [item for item in context['inputs'] if item.get('source_id') not in required]
    return context if context['manifest'] else None
