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
