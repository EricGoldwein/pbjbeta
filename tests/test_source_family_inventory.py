from source_family_inventory import source_family_inventory


def test_unintegrated_families_do_not_claim_active_lifecycle():
    rows = {r['source_id']: r for r in source_family_inventory({})}
    assert (rows['penalties']['governance_status'], rows['penalties']['implementation_status']) == ('PARTIAL', 'MERGED')
    assert (rows['hcris']['governance_status'], rows['hcris']['implementation_status']) == ('AUDITED_ONLY', 'UNMERGED')
    assert rows['npi_nppes']['implementation_status'] == 'NO_GOVERNED_PIPELINE'
    for key in ('penalties', 'hcris', 'npi_nppes'):
        assert rows[key]['workflow_source_id'] is None
        assert not any(k in rows[key] for k in ('active', 'active_release_id'))


def test_inventory_covers_required_sources_and_uses_real_release_health():
    rows = {r['source_id']: r for r in source_family_inventory({'datasets': [
        {'dataset_id': 'cms.snf_all_owners', 'active': {'active_release_id': '2026-07-31'}, 'health': 'CURRENT'}]})}
    assert rows['cms.snf_all_owners']['evidence'] == '2026-07-31'
    assert rows['cms.snf_all_owners']['health'] == 'CURRENT'
    assert {'cms.provider_info', 'cms.health_citations', 'cms.sff_pdf_list', 'cms.snf_enrollments',
            'cms.pbj_nurse_staffing', 'cms.pbj_non_nurse_staffing', 'cms.pbj_employee_ein_detail'} <= rows.keys()
    assert rows['cms.provider_info']['evidence'] == 'No governed ACTIVE release'
