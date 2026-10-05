"""Read-only Sources UI guidance from local releases and publication evidence."""

import hashlib
import json

from release_control_plane import control_plane_root
from pbj320_stage_common import load_stage_manifest
from pbj320_publication_contract import merge_destination_layers_for_display


def public_update_guidance(control, *, root=None):
    rows = {r['dataset_id']: r for r in control.get('datasets', [])}
    result = []
    # Presentation groups only; release/hash/lifecycle always come from evidence.
    for label, family, members in [
        ('PECOS ownership (Owners + Enrollments)', 'cms.snf_ownership_pair',
         ('cms.snf_all_owners', 'cms.snf_enrollments')),
        ('SFF / Candidate posting', 'cms.sff_pdf_list', ('cms.sff_pdf_list',)),
    ]:
        active = {key: rows.get(key, {}).get('active') or {} for key in members}
        releases = {a.get('active_release_id') for a in active.values()}
        item = {'label': label, 'source_id': members[0], 'local_release': 'Not selected',
                'website_status': 'Not verified', 'next_step': 'Open source details to prepare and review local data.'}
        if None in releases or len(releases) != 1:
            result.append(item)
            continue
        release = next(iter(releases))
        item['local_release'] = release
        if family == 'cms.snf_ownership_pair':
            from release_check import load_check_state
            checks = {row['dataset_id']: row for row in load_check_state(control_plane_root(root)).get('datasets', [])}
            observations = [checks.get(key, {}) for key in members]
            item['cms_release_vintage'] = observations[0].get('cms_release_vintage')
            item['snapshot_date'] = observations[0].get('snapshot_date') or release
            if len({check.get('cms_release_vintage') for check in observations}) != 1 or not all(
                       check.get('cms_release_vintage') and check.get('snapshot_date') and
                       check.get('publisher_url') and check.get('cms_dataset_version_id') and check.get('publisher_file_uuid') and
                       check.get('status') == 'CURRENT' and check.get('publisher_checked_at') and
                       check.get('publisher_sha256') == active[key].get('hash')
                       for key, check in zip(members, observations)):
                item.update(website_status='CMS bytes require verification',
                            next_step='Check CMS for both Owners and Enrollments before website release review.')
                result.append(item)
                continue
        stage = load_stage_manifest(family, release, root=root) or {}
        inputs = [i for a in stage.get('artifacts', []) for i in a.get('inputs', [])]
        matches = all(a.get('hash') and any(i.get('source_id') == key and
                      i.get('sha256') == a['hash'] for i in inputs) for key, a in active.items())
        if not matches:
            item['website_status'] = 'Website update needs preparation'
            item['next_step'] = ('Prepare a website candidate from the current local release. '
                                 'Existing website staging is missing or uses different source bytes. '
                                 'Do not download or activate the same local release again.')
        elif family == 'cms.snf_ownership_pair' and not all(any(
                i.get('source_id') == key and i.get('cms_release_vintage') == check.get('cms_release_vintage')
                and i.get('snapshot_date') == check.get('snapshot_date')
                and i.get('cms_dataset_version_id') == check.get('cms_dataset_version_id')
                and i.get('cms_file_uuid') == check.get('publisher_file_uuid')
                and i.get('cms_publisher_url') == check.get('publisher_url')
                for i in inputs) for key, check in zip(members, observations)):
            item.update(website_status='Website candidate needs provenance review',
                        next_step='Refresh website candidate provenance with the CMS release vintage and snapshot date. Do not reacquire or reactivate matching ACTIVE data.')
        elif stage.get('status') != 'STAGED' or not stage.get('validation_gates') or not all(
                gate.get('passed') is True for gate in stage['validation_gates']):
            item.update(website_status='Website candidate needs checks',
                        next_step='Open source details and resolve the website staging checks before publication.')
        else:
            path = control_plane_root(root) / 'state' / 'pbj320_publications' / family / f'{release}.json'
            try:
                publication = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                publication = {}
            # An older receipt for the same release date cannot prove this stage was shipped.
            # Use actual file bytes, matching the publication adapter's manifest fingerprint.
            from pbj320_stage_common import stage_manifest_path
            stage_sha = hashlib.sha256(stage_manifest_path(family, release, root=root).read_bytes()).hexdigest()
            if publication.get('stage_manifest_sha256') != stage_sha:
                publication = {}
            layers = merge_destination_layers_for_display(stage, publication)
            if layers.get('production_verified') == 'YES':
                item.update(website_status='Production verification recorded', next_step='No website update needed for this recorded release. Check CMS for a newer release when due.')
            elif layers.get('pushed') == 'YES':
                item.update(website_status='Sent for publication; live status unverified', next_step='Verify the website deployment and source provenance before calling this live.')
            else:
                item.update(website_status='Website candidate prepared', next_step='Review the staged website changes, publish the reviewed release, then verify the live website.')
        result.append(item)
    return result


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
