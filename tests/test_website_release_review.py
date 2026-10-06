import hashlib
import json
import re

import pytest

import data_ops_app
from pbj320_source_adapters import stage_publish_spec
from website_release_review import FAMILIES, website_review_context


def staged_candidate(root, family, *, passed=True, provenance=True):
    release = '2026-07-31' if family == 'cms.snf_ownership_pair' else '2026-09'
    members = ('cms.snf_all_owners', 'cms.snf_enrollments') if family == 'cms.snf_ownership_pair' else (family,)
    inputs = []
    checks = []
    for member in members:
        fingerprint = dict(source_id=member, sha256=member + '-sha', path=member + '.csv')
        metadata = dict(cms_release_vintage='2026-08', snapshot_date='2026-07-31',
                        cms_dataset_version_id=member + '-version', cms_file_uuid=member + '-file',
                        cms_publisher_url='https://data.cms.gov/' + member)
        if provenance:
            fingerprint.update(metadata)
        inputs.append(fingerprint)
        checks.append(dict(dataset_id=member, status='CURRENT', publisher_checked_at='2026-10-05',
                           identity_check_version=1, active_hash=member + '-sha', active_raw_sha256=member + '-sha',
                           publisher_identity_basis='official_page_link',
                           publisher_sha256=fingerprint['sha256'], publisher_url=metadata['cms_publisher_url'],
                           publisher_file_uuid=metadata['cms_file_uuid'], **{k: v for k, v in metadata.items() if k not in ('cms_file_uuid', 'cms_publisher_url')}))
    control = dict(datasets=[dict(dataset_id=member, active=dict(active_release_id=release, hash=member + '-sha',
                        metadata=dict(source_pdf_hash=member + '-sha'))) for member in members], facilities=dict(facilities={}))
    artifacts = []
    for role in stage_publish_spec(family).required_destination_roles:
        path = role + '.json'
        # Use the same cache location as the existing publication contract.
        from pbj320_publication import stage_artifact_cache_path
        cache = stage_artifact_cache_path(family, release, root=root)
        cache.mkdir(parents=True, exist_ok=True)
        (cache / path).write_text('{}')
        artifacts.append(dict(destination_id=role, path=path, proposed_sha256=hashlib.sha256(b'{}').hexdigest(),
                              publication_class='commit_destination', inputs=inputs))
    manifest = dict(source_id=family, active_release_id=release, status='STAGED', schema_version=2,
                    publication_contract_version=2, publication_base_sha='baseline-sha',
                    validation_gates=[dict(command='validate website package', passed=passed, summary='fixture gate')],
                    artifacts=artifacts, next_human_step='Review then publish through the governed workflow.')
    path = root / 'state/pbj320_stages' / family / (release + '.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    if not provenance:
        for check in checks:
            check.pop('identity_check_version')
    (root / 'state/release_checks.json').write_text(json.dumps(dict(datasets=checks)))
    return release, control, path


@pytest.mark.parametrize('family', FAMILIES)
def test_exact_candidate_review_is_read_only_and_shows_evidence(tmp_path, monkeypatch, family):
    release, control, path = staged_candidate(tmp_path, family)
    before = {p: p.read_bytes() for p in (tmp_path / 'state').rglob('*') if p.is_file()}
    context = website_review_context(family, release, control, root=tmp_path)
    assert context['publication_ready']
    assert context['stage_sha'] == hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setenv('PBJ_DATA_OPS_PASSWORD', 'test-only')
    monkeypatch.setattr(data_ops_app, 'control_panel_payload', lambda: control)
    monkeypatch.setattr(data_ops_app, 'snapshots_with_control_plane', lambda **kwargs: [])
    monkeypatch.setattr(data_ops_app, 'build_needs_attention_queue', lambda **kwargs: [])
    app = data_ops_app.create_app()
    browser = app.test_client()
    with browser.session_transaction() as session:
        session['data_ops_authenticated'] = True
    html = browser.get('/sources?check_cms=0').get_data(as_text=True)
    source_id = FAMILIES[family].required_source_evidence[0].source_id
    card = re.search(r'<article[^>]*data-do-website-source="' + source_id + r'".*?</article>', html, re.S).group()
    target = '/website-releases/' + family + '/' + release + '/panel'
    assert 'data-do-source-panel="' + target + '"' in card
    assert '>Review website release</button>' in card
    assert 'href="/sources/' + source_id + '?check_cms=0">Source details' in card
    response = browser.get(target)
    assert response.status_code == 200
    panel = response.get_data(as_text=True)
    for text in ('Review website release', 'Source evidence', 'What would be published',
                 'Validation gates', 'Next publication action', 'baseline-sha', hashlib.sha256(b'{}').hexdigest()):
        assert text in panel
    assert '<form' not in panel and 'Check CMS' not in panel
    assert browser.post(target).status_code == 405
    assert before == {p: p.read_bytes() for p in (tmp_path / 'state').rglob('*') if p.is_file()}


@pytest.mark.parametrize('family', FAMILIES)
def test_failed_gates_block_publication_handoff(tmp_path, family):
    release, control, _ = staged_candidate(tmp_path, family, passed=False)
    context = website_review_context(family, release, control, root=tmp_path)
    assert not context['publication_ready']
    assert 'validation gates not all PASS' in context['blocked_reasons']


def test_ownership_missing_provenance_blocks_handoff(tmp_path):
    release, control, _ = staged_candidate(tmp_path, 'cms.snf_ownership_pair', provenance=False)
    context = website_review_context('cms.snf_ownership_pair', release, control, root=tmp_path)
    assert not context['publication_ready']
    assert any('bytes not verified' in reason for reason in context['blocked_reasons'])


@pytest.mark.parametrize('family', FAMILIES)
def test_changed_active_and_old_receipt_do_not_authorize_publication(tmp_path, family):
    release, control, _ = staged_candidate(tmp_path, family)
    receipt = tmp_path / 'state/pbj320_publications' / family / (release + '.json')
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(dict(stage_manifest_sha256='old-stage', production_verified=True)))
    context = website_review_context(family, release, control, root=tmp_path)
    assert not context['publication']['receipt_matches'] and not context['publication']['production_verified']
    control['datasets'][0]['active']['hash'] = 'changed-active'
    assert not website_review_context(family, release, control, root=tmp_path)['publication_ready']


def test_blocked_review_route_and_missing_stage_do_not_offer_publication(tmp_path, monkeypatch):
    release, control, _ = staged_candidate(tmp_path, 'cms.snf_ownership_pair', provenance=False)
    monkeypatch.setenv('PBJ_DATA_OPS_PASSWORD', 'test-only')
    monkeypatch.setattr(data_ops_app, 'control_panel_payload', lambda: control)
    app = data_ops_app.create_app()
    browser = app.test_client()
    target = '/website-releases/cms.snf_ownership_pair/' + release + '/panel'
    assert browser.get(target).status_code in (302, 503)
    with browser.session_transaction() as session:
        session['data_ops_authenticated'] = True
    panel = browser.get(target).get_data(as_text=True)
    assert 'Publication blockers' in panel and 'Next publication action' not in panel
    assert 'Metadata checked; bytes not verified' in panel
    assert 'Website candidate unavailable' in browser.get('/website-releases/cms.sff_pdf_list/2025-01/panel').get_data(as_text=True)
    assert browser.get('/website-releases/cms.provider_info/2026-08/panel').status_code == 404


@pytest.mark.parametrize('family', FAMILIES)
def test_matching_publication_receipt_prevents_duplicate_handoff(tmp_path, family):
    release, control, path = staged_candidate(tmp_path, family)
    receipt = tmp_path / 'state/pbj320_publications' / family / (release + '.json')
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps(dict(stage_manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                      commit_sha='commit', committed_at='now', push_succeeded=True, push_timestamp='now',
                                      destination_layers=dict(pushed='YES', production_verified='NO'))))
    context = website_review_context(family, release, control, root=tmp_path)
    assert context['publication']['receipt_matches'] and not context['publication_ready']
