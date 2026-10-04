import re

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
    for copy in ('Do these next', 'Local data and the live website are separate steps',
                 'Ownership and SFF', 'Selected local releases (ACTIVE)',
                 'VALIDATED / REVIEW-ONLY', 'Review saved Survey Summary'):
        assert copy in html
    assert 'ready to promote' not in html

    assert 'data-do-reference-source="cms.survey_summary"' in html
    row = re.search(r'<tr data-do-release-row data-dataset-id="cms.survey_summary"[^>]*>', html)
    assert row and 'data-needs-attention="0"' in row.group()
