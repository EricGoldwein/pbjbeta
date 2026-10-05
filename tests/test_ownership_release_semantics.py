import hashlib
import json

import pytest

from active_release_registry import promote_release
from generic_cms_csv import assess_feed, detect, run_feed
from release_check import ownership_csv_feeds


def fixture(tmp_path, monkeypatch, source='cms.snf_all_owners', vintage='2026-08-01', raw=b'ENROLLMENT ID\n1\n'):
    feed = ownership_csv_feeds(tmp_path)[source]
    filename = 'SNF_All_Owners_2026.07.31.csv' if source.endswith('all_owners') else 'SNF_Enrollments_2026.07.31.csv'
    url = 'https://data.cms.gov/current.csv'
    registry = tmp_path / 'state/active_releases.json'
    monkeypatch.setenv('PBJ_ACTIVE_RELEASE_REGISTRY', str(registry))
    feed.destination.mkdir(parents=True)
    local = feed.destination / filename
    local.write_bytes(raw)
    promote_release(source, '2026-07-31', local, validated_at='fixture', path=registry,
                    metadata={'cms_release_vintage': '2026-08', 'cms_publisher_url': url,
                              'cms_file_uuid': 'file', 'cms_dataset_version_id': 'version'})

    def fetch(url_requested):
        if '/slug?' in url_requested:
            return {'data': {'uuid': feed.cms_dataset_id, 'current_dataset': {'uuid': 'version'}}}
        if '/jsonapi/' in url_requested:
            return {'data': [{'id': 'version', 'attributes': {'field_dataset_version': vintage,
                             'field_last_updated_date': '2026-08-17'}}]}
        return {'data': [{'type': 'Primary', 'file_name': filename, 'file_url': url, 'file_uuid': 'file'}]}
    return feed, fetch, local, registry


@pytest.mark.parametrize('source', ['cms.snf_all_owners', 'cms.snf_enrollments'])
def test_august_cms_vintage_july_snapshot_and_matching_bytes_are_current(tmp_path, monkeypatch, source):
    feed, fetch, local, registry = fixture(tmp_path, monkeypatch, source)
    before = registry.read_bytes()
    found = detect(feed, fetch_json=fetch)
    assert found['cms_release_vintage'] == '2026-08'
    assert found['snapshot_date'] == '2026-07-31'
    # Existing aligned-pair key remains the snapshot period; no rename/promotion.
    assert found['release_id'] == '2026-07-31'
    result = run_feed(feed, True, root=tmp_path, fetch_json=fetch, fetch_bytes=lambda _: local.read_bytes())
    assert result['status'] == 'CURRENT'
    assert result['publisher_sha256'] == hashlib.sha256(local.read_bytes()).hexdigest()
    assert result['cms_release_vintage'] == '2026-08'
    assert result['snapshot_date'] == '2026-07-31'
    assert registry.read_bytes() == before
    assert json.loads((tmp_path / 'state/release_candidates.json').read_text())['datasets'] == {}


def test_same_filename_url_and_uuids_but_different_bytes_are_revised(tmp_path, monkeypatch):
    feed, fetch, local, registry = fixture(tmp_path, monkeypatch)
    before = registry.read_bytes()
    revised = b'ENROLLMENT ID\n2\n'
    result = run_feed(feed, False, root=tmp_path, fetch_json=fetch, fetch_bytes=lambda _: revised)
    assert result['status'] == 'REVISED'
    assert result['new_release_available'] is True
    assert result['revision_identity_changes'] == ['source_sha256']
    assert registry.read_bytes() == before
    assert local.read_bytes() == b'ENROLLMENT ID\n1\n'
    candidate = json.loads((tmp_path / 'state/release_candidates.json').read_text())['datasets'][feed.dataset_id]
    assert candidate['state'] == 'DETECTED'
    assert candidate['metadata']['cms_source_sha256'] == hashlib.sha256(revised).hexdigest()
    acquired = run_feed(feed, True, root=tmp_path, fetch_json=fetch, fetch_bytes=lambda _: revised)
    assert acquired['status'] == 'ACQUIRED'
    candidate = json.loads((tmp_path / 'state/release_candidates.json').read_text())['datasets'][feed.dataset_id]
    assert candidate['source_uri'] != local.as_uri()
    assert local.read_bytes() == b'ENROLLMENT ID\n1\n' and registry.read_bytes() == before


def test_newer_cms_vintage_is_newer_even_with_same_filename_snapshot(tmp_path, monkeypatch):
    feed, fetch, local, registry = fixture(tmp_path, monkeypatch, vintage='2026-09-01')
    result = assess_feed(feed, root=tmp_path, fetch_json=fetch, fetch_bytes=lambda _: b'ENROLLMENT ID\n2\n')
    assert result['status'] == 'NEWER'
    assert result['cms_release_vintage'] == '2026-09'
    assert result['snapshot_date'] == '2026-07-31'


def test_resource_identity_change_with_identical_bytes_never_reacquires(tmp_path, monkeypatch):
    feed, fetch, local, registry = fixture(tmp_path, monkeypatch)
    original = fetch
    def changed(url):
        payload = original(url)
        if '/resources' in url:
            payload['data'][0].update(file_uuid='changed', file_url='https://data.cms.gov/relocated.csv')
        return payload
    result = run_feed(feed, True, root=tmp_path, fetch_json=changed, fetch_bytes=lambda _: local.read_bytes())
    assert result['status'] == 'CURRENT'
    assert json.loads((tmp_path / 'state/release_candidates.json').read_text())['datasets'] == {}


def test_failed_byte_check_never_reports_current(tmp_path, monkeypatch):
    feed, fetch, _, _ = fixture(tmp_path, monkeypatch)
    def failed(_):
        raise TimeoutError('CMS unavailable')
    result = assess_feed(feed, root=tmp_path, fetch_json=fetch, fetch_bytes=failed)
    assert result['status'] == 'ERROR' and result['new_release_available'] is None


def test_same_date_byte_revision_passes_governed_acquisition_gate(tmp_path, monkeypatch):
    from release_check import acquire_detected_source
    feed, fetch, local, registry = fixture(tmp_path, monkeypatch)
    run_feed(feed, False, root=tmp_path, fetch_json=fetch, fetch_bytes=lambda _: b'ENROLLMENT ID\n2\n')
    calls = []
    monkeypatch.setattr('release_check.production_handlers', lambda: {
        feed.dataset_id: lambda acquire: calls.append(acquire) or {'status': 'ACQUIRED'}})
    assert acquire_detected_source(feed.dataset_id, root=tmp_path)['status'] == 'ACQUIRED'
    assert calls == [True]


def test_resource_change_during_acquisition_preserves_active(tmp_path, monkeypatch):
    feed, fetch, local, registry = fixture(tmp_path, monkeypatch)
    before = registry.read_bytes()
    responses = iter([b'ENROLLMENT ID\n2\n', b'ENROLLMENT ID\n3\n'])
    with pytest.raises(RuntimeError, match='changed between detection and acquisition'):
        run_feed(feed, True, root=tmp_path, fetch_json=fetch, fetch_bytes=lambda _: next(responses))
    assert registry.read_bytes() == before and local.read_bytes() == b'ENROLLMENT ID\n1\n'
    candidate = json.loads((tmp_path / 'state/release_candidates.json').read_text())['datasets'][feed.dataset_id]
    assert candidate['state'] == 'DETECTED'


def test_ui_does_not_hide_same_snapshot_revision(tmp_path, monkeypatch):
    from cms_data_ops import build_release_availability_context
    feed, fetch, local, registry = fixture(tmp_path, monkeypatch)
    result = assess_feed(feed, root=tmp_path, fetch_json=fetch, fetch_bytes=lambda _: b'ENROLLMENT ID\n2\n')
    active = json.loads(registry.read_text())['datasets'][feed.dataset_id]
    view = build_release_availability_context(feed.dataset_id, control_row={'active': active},
                                             check_row=result, root=tmp_path)
    assert view['new_release_available'] is True
    assert view['cms_byte_verified_current'] is False
    assert view['publisher_latest_label'] == 'Aug 2026'
    assert view['snapshot_date'] == '2026-07-31'


def test_website_input_provenance_binds_both_dates_and_sha(tmp_path, monkeypatch):
    from pbj320_stage_ownership import ownership_input_provenance
    from pbj320_stage_common import StageError
    feed, fetch, local, registry = fixture(tmp_path, monkeypatch)
    active = json.loads(registry.read_text())['datasets'][feed.dataset_id]
    with pytest.raises(StageError, match='verify current raw bytes'):
        ownership_input_provenance(feed.dataset_id, active, root=tmp_path)
    check = assess_feed(feed, root=tmp_path, fetch_json=fetch, fetch_bytes=lambda _: local.read_bytes())
    (tmp_path / 'state/release_checks.json').write_text(json.dumps({'datasets': [{'dataset_id': feed.dataset_id, **check}]}))
    provenance = ownership_input_provenance(feed.dataset_id, active, root=tmp_path)
    assert provenance['cms_release_vintage'] == '2026-08'
    assert provenance['snapshot_date'] == '2026-07-31'
    assert provenance['cms_source_sha256'] == active['hash']
    assert provenance['cms_file_uuid'] == 'file'
    assert provenance['acquired_at'] is None  # Do not invent an acquisition date.


def test_ownership_website_card_requires_both_members_byte_verified_and_vintage_provenance(tmp_path, monkeypatch):
    from source_operator_guidance import public_update_guidance
    owners = 'cms.snf_all_owners'
    enrollments = 'cms.snf_enrollments'
    control = {'datasets': [{'dataset_id': key, 'active': {'active_release_id': '2026-07-31', 'hash': key}}
                            for key in (owners, enrollments)]}
    state = tmp_path / 'state'
    state.mkdir(exist_ok=True)
    stages = state / 'pbj320_stages/cms.snf_ownership_pair'
    stages.mkdir(parents=True)
    inputs = [{'source_id': key, 'sha256': key} for key in (owners, enrollments)]
    path = stages / '2026-07-31.json'
    path.write_text(json.dumps({'status': 'STAGED', 'validation_gates': [{'passed': True}],
                                'artifacts': [{'inputs': inputs}]}))
    row = public_update_guidance(control, root=tmp_path)[0]
    assert row['website_status'] == 'CMS bytes require verification'
    checks = [{'dataset_id': key, 'status': 'CURRENT', 'publisher_checked_at': 'fixture',
               'publisher_sha256': key, 'cms_release_vintage': '2026-08', 'snapshot_date': '2026-07-31',
               'cms_dataset_version_id': key+'-version', 'publisher_file_uuid': key+'-file',
               'publisher_url': 'https://data.cms.gov/'+key} for key in (owners, enrollments)]
    (state / 'release_checks.json').write_text(json.dumps({'datasets': checks}))
    row = public_update_guidance(control, root=tmp_path)[0]
    assert row['website_status'] == 'Website candidate needs provenance review'
    for item, check in zip(inputs, checks):
        item.update(cms_release_vintage=check['cms_release_vintage'], snapshot_date=check['snapshot_date'],
                    cms_dataset_version_id=check['cms_dataset_version_id'], cms_file_uuid=check['publisher_file_uuid'],
                    cms_publisher_url=check['publisher_url'])
    path.write_text(json.dumps({'status': 'STAGED', 'validation_gates': [{'passed': True}],
                                'artifacts': [{'inputs': inputs}]}))
    assert public_update_guidance(control, root=tmp_path)[0]['website_status'] == 'Website candidate prepared'
    checks[1]['publisher_sha256'] = 'different'
    (state / 'release_checks.json').write_text(json.dumps({'datasets': checks}))
    assert public_update_guidance(control, root=tmp_path)[0]['website_status'] == 'CMS bytes require verification'
