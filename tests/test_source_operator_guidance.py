import hashlib
import json

from source_operator_guidance import public_update_guidance, survey_review_guidance


def setup_stage(tmp_path, *, stage_hash='current', gates=True):
    control = {'datasets': [{'dataset_id': 'cms.sff_pdf_list',
                            'active': {'active_release_id': '2026-09', 'hash': 'current'}}]}
    path = tmp_path / 'state/pbj320_stages/cms.sff_pdf_list/2026-09.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'status': 'STAGED', 'validation_gates': [{'passed': gates}],
                               'artifacts': [{'inputs': [{'source_id': 'cms.sff_pdf_list',
                                                        'sha256': stage_hash}]}]}))
    return control, path


def test_same_release_revision_requires_new_website_preparation(tmp_path):
    control, _ = setup_stage(tmp_path, stage_hash='old')
    row = public_update_guidance(control, root=tmp_path)[1]
    assert row['local_release'] == '2026-09'
    assert row['website_status'] == 'Website update needs preparation'


def test_failed_stage_checks_are_not_a_prepared_website_candidate(tmp_path):
    control, _ = setup_stage(tmp_path, gates=False)
    assert public_update_guidance(control, root=tmp_path)[1]['website_status'] == 'Website candidate needs checks'


def test_only_receipt_for_exact_stage_proves_publication(tmp_path):
    control, stage = setup_stage(tmp_path)
    receipt = tmp_path / 'state/pbj320_publications/cms.sff_pdf_list/2026-09.json'
    receipt.parent.mkdir(parents=True)
    record = {'stage_manifest_sha256': 'old', 'production_verified': True}
    receipt.write_text(json.dumps(record))
    assert public_update_guidance(control, root=tmp_path)[1]['website_status'] == 'Website candidate prepared'
    record['stage_manifest_sha256'] = hashlib.sha256(stage.read_bytes()).hexdigest()
    record.pop('production_verified')
    record['destination_layers'] = {'pushed': 'YES', 'production_verified': 'NO'}
    receipt.write_text(json.dumps(record))
    assert 'live status unverified' in public_update_guidance(control, root=tmp_path)[1]['website_status']
    record['production_verified'] = True
    receipt.write_text(json.dumps(record))
    assert public_update_guidance(control, root=tmp_path)[1]['website_status'] == 'Production verification recorded'


def test_validation_is_review_readiness_not_activation():
    guidance = survey_review_guidance({'state': 'VALIDATED', 'validation': {'status': 'PASS'}})
    assert guidance['label'] == 'VALIDATED / REVIEW-ONLY — saved for reference'
    assert 'not ACTIVE' in guidance['detail']
    assert 'Checks passed' not in survey_review_guidance({'state': 'VALIDATED'})['label']
    assert 'Checks failed' in survey_review_guidance({'state': 'FAILED'})['label']
