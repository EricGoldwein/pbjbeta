import re

import pytest

import data_ops_app


def client(monkeypatch):
    monkeypatch.setenv('PBJ_DATA_OPS_PASSWORD', 'test-only')
    app = data_ops_app.create_app()
    result = app.test_client()
    with result.session_transaction() as session:
        session['data_ops_authenticated'] = True
    return result


def test_review_page_and_modal_present_candidate_without_activation(monkeypatch):
    candidate = {'state': 'VALIDATED', 'release_id': 'fixture', 'hash': 'abc',
                 'validation': {'status': 'PASS', 'row_count': 2, 'provider_count': 1, 'duplicate_key_rows': 0}}
    monkeypatch.setattr(data_ops_app, 'control_panel_payload', lambda: {'datasets': [{'dataset_id': 'cms.survey_summary', 'pending': candidate}]})
    browser = client(monkeypatch)
    for endpoint in ('/sources/cms.survey_summary', '/sources/cms.survey_summary/panel'):
        response = browser.get(endpoint)
        html = response.get_data(as_text=True)
        assert response.status_code == 200
        assert 'VALIDATED' in html and 'No ACTIVE release' in html
        assert 'CCN + Inspection Cycle' in html
        assert 'VALIDATED / REVIEW-ONLY — saved for reference' in html
        assert 'Check CMS again for a newer file' in html
        assert 'not ACTIVE' in html
        assert 'explicit review required' not in html
        assert '/actions/survey-summary/prepare' in html
        actions = re.findall(r'<form[^>]+action="([^"]+)"', html)
        assert '/actions/survey-summary/prepare' in actions
        assert not any('activate' in action or 'promote' in action for action in actions)


def test_preparation_route_requires_auth_and_uses_canonical_adapter(monkeypatch):
    app = data_ops_app.create_app()
    assert app.test_client().post('/actions/survey-summary/prepare').status_code in (302, 503)
    called = []
    monkeypatch.setattr('survey_summary.prepare_candidate', lambda: called.append(True) or {'state': 'VALIDATED'})
    response = client(monkeypatch).post('/actions/survey-summary/prepare')
    assert response.status_code == 302
    assert called == [True]
    assert response.headers['Location'].endswith('/sources/cms.survey_summary')


def test_sources_page_explains_local_and_website_steps(monkeypatch):
    candidate = {'state': 'VALIDATED', 'validation': {'status': 'PASS'}}
    monkeypatch.setattr(data_ops_app, 'control_panel_payload', lambda: {
        'datasets': [{'dataset_id': 'cms.survey_summary', 'pending': candidate,
                      'impact': {'would_mark_stale': []}}],
        'facilities': {'facilities': {}}})
    monkeypatch.setattr(data_ops_app, 'snapshots_with_control_plane', lambda **kwargs: [])
    monkeypatch.setattr(data_ops_app, 'build_needs_attention_queue', lambda **kwargs: [{
        'source_id': 'cms.survey_summary', 'human_name': 'Survey Summary',
        'release_line': 'Saved candidate', 'concise_state': 'ready to promote'}])
    response = client(monkeypatch).get('/sources?check_cms=0')
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    for copy in ('Next actions', 'Local source updates and website publishing are separate steps',
                 'Website release readiness', 'Selected local releases (ACTIVE)',
                 'VALIDATED / REVIEW-ONLY', 'Review saved file'):
        assert copy in html
    assert 'ready to promote' not in html

    assert 'data-do-reference-source="cms.survey_summary"' in html
    row = re.search(r'<tr data-do-release-row data-dataset-id="cms.survey_summary"[^>]*>', html)
    assert row and 'data-needs-attention="0"' in row.group()


@pytest.mark.parametrize('status,chip,action', [
    ('Website update needs preparation', 'NEEDS PREPARATION', 'Prepare website release'),
    ('Website candidate needs checks', 'CHECKS REQUIRED', 'Review staging checks'),
    ('Website candidate prepared', 'STAGED', 'Review website release'),
    ('Sent for publication; live status unverified', 'AWAITING VERIFICATION', 'Verify live website'),
    ('Production verification recorded', 'VERIFIED', 'Review verification'),
    ('Not verified', 'LOCAL REVIEW REQUIRED', 'Review local source'),
])
def test_sources_website_action_follows_existing_guidance(monkeypatch, status, chip, action):
    monkeypatch.setattr(data_ops_app, 'control_panel_payload', lambda: {
        'datasets': [], 'facilities': {'facilities': {}}})
    monkeypatch.setattr(data_ops_app, 'snapshots_with_control_plane', lambda **kwargs: [])
    monkeypatch.setattr(data_ops_app, 'build_needs_attention_queue', lambda **kwargs: [])
    monkeypatch.setattr('source_operator_guidance.public_update_guidance', lambda control: [{
        'source_id': 'cms.snf_all_owners', 'local_release': '2026-07-31',
        'website_status': status, 'next_step': 'Recorded guidance'}])
    html = client(monkeypatch).get('/sources?check_cms=0').get_data(as_text=True)
    card = re.search(r'<article[^>]*data-do-website-source=.*?</article>', html, re.S).group()
    assert chip in card
    assert f'>{action}</button>' in card
    assert 'Jul 31, 2026' in card
    assert '<form' not in card  # Opening details must never execute a lifecycle action.


def test_saved_survey_card_keeps_hash_secondary_and_only_offers_review(monkeypatch):
    candidate_id = '2026-08-9b534d95f43a-76ad33cab36e'
    monkeypatch.setattr(data_ops_app, 'control_panel_payload', lambda: {
        'datasets': [{'dataset_id': 'cms.survey_summary', 'pending': {
            'release_id': candidate_id, 'state': 'VALIDATED', 'validation': {'status': 'PASS'}},
            'impact': {'would_mark_stale': []}}], 'facilities': {'facilities': {}}})
    monkeypatch.setattr(data_ops_app, 'snapshots_with_control_plane', lambda **kwargs: [])
    monkeypatch.setattr(data_ops_app, 'build_needs_attention_queue', lambda **kwargs: [])
    html = client(monkeypatch).get('/sources?check_cms=0').get_data(as_text=True)
    card = re.search(r'<article[^>]*data-do-reference-source=.*?</article>', html, re.S).group()
    visible, evidence = card.split('<details>', 1)
    assert 'Aug 2026' in visible and 'Checks passed' in visible
    assert candidate_id not in visible and candidate_id in evidence
    assert 'Review saved file' in visible
    assert '<form' not in card and '/actions/approve' not in card and '/actions/promote' not in card
