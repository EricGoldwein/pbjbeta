"""Read-only Sources UI guidance from local releases and publication evidence."""

def public_update_guidance(control, *, root=None):
    from website_release_readiness import FAMILIES, family_readiness
    return [family_readiness(family, control, root=root) for family in FAMILIES]


def survey_review_guidance(candidate):
    state = candidate.get('state', '')
    passed = state == 'VALIDATED' and (candidate.get('validation') or {}).get('status') == 'PASS'
    if passed:
        return {'label': 'VALIDATED / REVIEW-ONLY — saved for reference',
                'detail': 'CMS data was downloaded, saved with provenance, and passed automated checks. Review the results below; no activation required. It is not ACTIVE and is not used by dashboards or the public website. No activation or publication action is available in this Survey Summary workflow.'}
    if state == 'FAILED' or (candidate.get('validation') or {}).get('status') == 'FAIL':
        return {'label': 'Checks failed — inspect the validation errors',
                'detail': 'The saved candidate cannot be used yet. Review the errors below before preparing it again.'}
    return {'label': 'Prepare a file for review',
            'detail': 'Check the official CMS release, download and retain the source file, and run automated checks. This does not activate or publish it.'}
