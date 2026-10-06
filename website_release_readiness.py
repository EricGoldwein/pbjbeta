"""Evidence-based website readiness; lifecycle writes are deliberately excluded."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request
from urllib.parse import urlparse
from urllib.parse import urljoin
from html.parser import HTMLParser

from pbj320_publication import stage_artifact_cache_path
from pbj320_publication_contract import evaluate_stage_publish_eligibility
from pbj320_source_adapters import stage_publish_spec
from pbj320_stage_common import load_stage_manifest, stage_manifest_path
from release_control_plane import control_plane_root


@dataclass(frozen=True)
class SourceRequirement:
    source_id: str
    name: str
    role: str
    observer: str
    raw_primary: bool = True


@dataclass(frozen=True)
class WebsiteFamily:
    label: str
    required_source_evidence: tuple[SourceRequirement, ...]
    paired_vintage: bool = False
    description: str = ''


FAMILIES = {
    'cms.snf_ownership_pair': WebsiteFamily('PECOS ownership', (
        SourceRequirement('cms.snf_all_owners', 'SNF All Owners', 'Owners', 'cms_csv'),
        SourceRequirement('cms.snf_enrollments', 'SNF Enrollments', 'Enrollments', 'cms_csv'),
    ), paired_vintage=True, description='Owners + Enrollments'),
    'cms.sff_pdf_list': WebsiteFamily('SFF', (
        SourceRequirement('cms.sff_pdf_list', 'SFF / Candidate posting', 'CMS posting PDF', 'sff_pdf', False),
    ), description='Special Focus Facilities + Candidates'),
}


def _json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def evidence_path(family, release_id, *, root=None):
    return control_plane_root(root) / 'state/website_source_evidence' / family / f'{release_id}.json'


def publication_evidence(manifest, stage_sha, receipt):
    """Only a receipt for this exact candidate may establish publication state."""
    matches = bool(receipt) and receipt.get('stage_manifest_sha256') == stage_sha
    candidate_inputs = {i.get('source_id'): i.get('sha256') for a in manifest.get('artifacts', [])
                        for i in a.get('inputs', []) if i.get('source_id')}
    if receipt.get('source_hashes') and any(candidate_inputs.get(k) != v for k, v in receipt['source_hashes'].items()):
        matches = False
    record = receipt if matches else {}
    layers = record.get('destination_layers') or {}
    committed = bool(record.get('commit_sha')) and (layers.get('committed') == 'YES' or record.get('committed_at'))
    pushed = bool(committed and record.get('push_succeeded') and record.get('push_timestamp'))
    checks = record.get('production_verification_checks') or []
    proposed = {a.get('proposed_sha256') for a in manifest.get('artifacts', []) if a.get('publication_class') != 'validation_only'} - {None, ''}
    production_origin = (os.environ.get('PBJ320_PRODUCTION_ORIGIN') or 'https://www.pbj320.com').rstrip('/')
    verified_hashes = {c.get('expected') for c in checks if c.get('expected') == c.get('actual')
                       and str(c.get('target') or '').startswith(production_origin + '/') and c.get('verification_timestamp')}
    provenance_verified = bool(pushed and record.get('production_verified') is True
                               and record.get('production_verified_at') and checks
                               and proposed and proposed.issubset(verified_hashes)
                               and all(c.get('passed') is True or c.get('result') == 'PASS' for c in checks))
    deployment = record.get('deployment_observation') or {}
    deployment_observed = bool(provenance_verified or (pushed and deployment.get('observed_at') and deployment.get('commit_sha') == record.get('commit_sha')
                               and deployment.get('status') == 'SUCCESS'))
    if provenance_verified:
        label = 'Live deployment verified'
    elif deployment_observed:
        label = 'Deployment observed; provenance not verified'
    elif pushed:
        label = 'Pushed; deployment not verified'
    elif committed:
        label = 'Committed locally, not pushed'
    else:
        label = 'Not published'
    return dict(label=label, committed=bool(committed), pushed=pushed, deployment_observed=deployment_observed,
                production_verified=provenance_verified, receipt_matches=matches, receipt_present=bool(receipt),
                receipt_notice='Existing publication receipt belongs to a different candidate' if receipt and not matches else None,
                deployment_detail='Live artifact provenance matches this candidate' if provenance_verified else
                'Deployment status not observable from Data Ops' if not deployment_observed else 'Matching commit deployment observed',
                receipt=receipt)


def family_readiness(family, control, *, root=None, release_id=None):
    config = FAMILIES[family]
    rows = {r['dataset_id']: r for r in control.get('datasets', [])}
    active = {req.source_id: rows.get(req.source_id, {}).get('active') or {} for req in config.required_source_evidence}
    selected = {a.get('active_release_id') for a in active.values()}
    aligned = None not in selected and len(selected) == 1
    release = release_id or (next(iter(selected)) if aligned else None)
    manifest = load_stage_manifest(family, release, root=root) if release else None
    stage_sha = hashlib.sha256(stage_manifest_path(family, release, root=root).read_bytes()).hexdigest() if manifest else None
    artifacts = (manifest or {}).get('artifacts') or []
    inputs = []
    for artifact in artifacts:
        for item in artifact.get('inputs', []):
            if item not in inputs:
                inputs.append(item)
    recorded = _json(evidence_path(family, release, root=root)) if release else {}
    observations = recorded.get('sources', {}) if recorded.get('stage_manifest_sha256') == stage_sha else {}
    from release_check import load_check_state
    checks = {c['dataset_id']: c for c in load_check_state(control_plane_root(root)).get('datasets', [])}
    source_evidence = []
    blockers = []
    candidate_matches = bool(aligned and release in selected and manifest)
    for req in config.required_source_evidence:
        selected_active = active[req.source_id]
        fingerprints = [i for i in inputs if i.get('source_id') == req.source_id]
        hashes = {i.get('sha256') for i in fingerprints}
        staged_sha = next(iter(hashes)) if len(hashes) == 1 else None
        active_sha = selected_active.get('hash')
        raw_sha = active_sha if req.raw_primary else (selected_active.get('metadata') or {}).get('source_pdf_hash')
        bound = bool(staged_sha and staged_sha == active_sha and raw_sha)
        candidate_matches = candidate_matches and bound
        observation = observations.get(req.source_id) or checks.get(req.source_id, {})
        publisher = urlparse(str(observation.get('publisher_url') or ''))
        authoritative = bool(observation.get('identity_check_version') == 1 and observation.get('publisher_checked_at')
                             and publisher.scheme == 'https' and publisher.hostname in {'data.cms.gov', 'www.cms.gov', 'cms.gov'}
                             and observation.get('cms_release_vintage'))
        if req.observer == 'cms_csv':
            authoritative = authoritative and bool(observation.get('cms_dataset_version_id') and observation.get('publisher_file_uuid'))
        elif req.observer == 'sff_pdf':
            authoritative = authoritative and observation.get('publisher_identity_basis') == 'official_page_link'
        verified = bool(bound and authoritative and observation.get('status') == 'CURRENT'
                        and observation.get('publisher_sha256') == raw_sha
                        and observation.get('active_hash') == active_sha
                        and observation.get('active_raw_sha256') == raw_sha)
        if verified:
            status = 'Byte verified current'
        elif observation.get('status') in {'REVISED', 'NEWER'} and observation.get('publisher_sha256'):
            status = 'Publisher bytes differ'
        elif observation.get('status') in {'ERROR', 'UNKNOWN'}:
            status = 'Verification failed/unavailable'
        elif observation.get('publisher_checked_at') or observation.get('checked_at'):
            status = 'Metadata checked; bytes not verified'
        else:
            status = 'Verification required'
        if not verified:
            blockers.append(f'{req.name}: {status.lower()}.')
        source_evidence.append(dict(source_id=req.source_id, name=req.name, role=req.role, verified=verified,
                                    status=status, staged_sha256=staged_sha, active_sha256=active_sha,
                                    raw_sha256=raw_sha, observation=observation, fingerprints=fingerprints,
                                    active_source_uri=selected_active.get('source_uri'),
                                    raw_source_uri=selected_active.get('source_uri') if req.raw_primary else (selected_active.get('metadata') or {}).get('source_pdf_uri'),
                                    acquired_at=selected_active.get('downloaded_at') or (selected_active.get('metadata') or {}).get('acquired_at'),
                                    vintage=observation.get('cms_release_vintage'), snapshot=observation.get('snapshot_date')))
    if config.paired_vintage and len({s['vintage'] for s in source_evidence}) != 1:
        blockers.append('Owners and Enrollments CMS release vintages do not match.')
    if manifest and (manifest.get('source_id') != family or manifest.get('active_release_id') != release):
        candidate_matches = False
        blockers.append('Stage manifest family or release identity does not match this review.')
    if not candidate_matches:
        blockers.append('Website candidate is missing or does not match the selected ACTIVE source hashes.')
    eligibility = evaluate_stage_publish_eligibility(manifest, stage_publish_spec(family), root=root, release_id=release)
    blockers.extend(eligibility['reasons'])
    cache = stage_artifact_cache_path(family, release, root=root) if release else None
    for artifact in artifacts:
        if artifact.get('publication_class') not in {'commit_destination', 'shared_derived'}:
            continue
        path = (cache / str(artifact.get('path') or '')).resolve()
        if not path.is_relative_to(cache.resolve()) or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != artifact.get('proposed_sha256'):
            blockers.append(f"Saved destination bytes do not match candidate: {artifact.get('destination_id') or artifact.get('path')}")
    receipt = _json(control_plane_root(root) / 'state/pbj320_publications' / family / f'{release}.json') if release else {}
    publication = publication_evidence(manifest or {}, stage_sha, receipt)
    validated = bool((manifest or {}).get('validation_gates')) and all(g.get('passed') is True for g in manifest['validation_gates'])
    verified_count = sum(s['verified'] for s in source_evidence)
    source_status = 'Byte verified current' if verified_count == len(source_evidence) else next(s['status'] for s in source_evidence if not s['verified'])
    if not manifest:
        action, label = 'source_details', 'Prepare website release'
    elif verified_count != len(source_evidence):
        action, label = 'verify_sources', 'Verify CMS source bytes'
    elif publication['production_verified'] and not blockers:
        action, label = 'none', None
    elif publication['pushed']:
        action, label = 'review', 'Verify publication'
    elif not manifest:
        action, label = 'source_details', 'Prepare website release'
    else:
        action, label = 'review', 'Review website release'
    publication_next_step = ('Verify the live artifact provenance against this candidate.' if publication['pushed'] else
                            'Review the recorded publication commit, then explicitly push it through the governed workflow.' if publication['committed'] else
                            'Review the staged destinations, then explicitly publish the approved release through the governed workflow.')
    return dict(family=family, label=config.label, family_description=config.description, source_id=config.required_source_evidence[0].source_id,
                release_id=release, local_release=next(iter(selected)) if aligned else 'Not selected',
                local_selected=aligned, manifest=manifest, stage_sha=stage_sha, inputs=inputs,
                source_evidence=source_evidence, verified_count=verified_count, required_count=len(source_evidence),
                source_status=source_status, candidate_valid=bool(candidate_matches and eligibility['publishable'] and not any('Saved destination' in b for b in blockers)),
                validation_passed=validated, publication=publication, blocked_reasons=list(dict.fromkeys(blockers)),
                publication_ready=not blockers and not publication['pushed'], publication_next_step=publication_next_step, next_action=action, action_label=label,
                cms_release_vintage=source_evidence[0]['vintage'], snapshot_date=source_evidence[0]['snapshot'],
                website_review_family=family if manifest else None, website_review_release=release if manifest else None,
                website_status=source_status if blockers else 'Website candidate prepared',
                next_step=blockers[0] if blockers else label or 'No urgent action; this release is verified live.')


def observe_required_source(requirement, *, root, fetch_json=None, fetch_bytes=None):
    """Read-only adapters; never call run_feed (which can record candidates)."""
    started = time.monotonic()
    def bounded_bytes(url, *, limit=512 * 1024 * 1024):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname not in {'data.cms.gov', 'www.cms.gov', 'cms.gov'}:
            raise ValueError('Publisher resource is not an official HTTPS CMS resource')
        if time.monotonic() - started > 180:
            raise TimeoutError('CMS observation time budget exceeded')
        if fetch_bytes:
            payload = fetch_bytes(url)
            if len(payload) > limit:
                raise ValueError('CMS observation exceeds byte limit')
            return payload
        request = urllib.request.Request(url, headers={'User-Agent': 'PBJ-website-source-verification/1.0'})
        payload = bytearray()
        with urllib.request.urlopen(request, timeout=30) as response:
            final = urlparse(response.geturl())
            if final.scheme != 'https' or final.hostname not in {'data.cms.gov', 'www.cms.gov', 'cms.gov'}:
                raise ValueError('CMS resource redirected outside the official publisher')
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                if len(payload) + len(chunk) > limit or time.monotonic() - started > 180:
                    raise ValueError('CMS observation byte/time limit exceeded')
                payload.extend(chunk)
        return bytes(payload)
    if requirement.observer == 'cms_csv':
        from release_check import ownership_csv_feeds
        from generic_cms_csv import assess_feed
        feed = ownership_csv_feeds(root)[requirement.source_id]
        def metadata(url):
            payload = fetch_json(url) if fetch_json else json.loads(bounded_bytes(url, limit=8 * 1024 * 1024))
            if url.endswith('/resources'):
                rows = payload.get('data') if isinstance(payload, dict) else payload
                matches = [r for r in (rows or []) if re.search(feed.filename_pattern, str(r.get('file_name') or ''), re.I)]
                primaries = [r for r in matches if r.get('type') == 'Primary' or r.get('media_bundle') == 'primary_dataset_file']
                if len(primaries or matches) != 1:
                    raise ValueError('CMS current distribution is ambiguous; refusing to guess')
            return payload
        return assess_feed(feed, root=root, fetch_json=metadata, fetch_bytes=bounded_bytes)
    if requirement.observer == 'sff_pdf':
        from cms_source_registry import get_source
        from sff_release import parse_sff_posting_updated_label
        from active_release_registry import get_active_release, registry_path
        from cms_release_identity import assess_raw_identity
        landing = get_source(requirement.source_id).landing_url
        class PostingLinks(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = set()
            def handle_starttag(self, tag, attrs):
                href = dict(attrs).get('href', '')
                if tag == 'a' and re.search(r'/files/document/sff-posting-candidate-list[^/?]*\.pdf(?:\?|$)', href, re.I):
                    self.links.add(urljoin(landing, href))
        parser = PostingLinks()
        parser.feed(bounded_bytes(landing, limit=8 * 1024 * 1024).decode('utf-8', errors='replace'))
        if len(parser.links) != 1:
            raise ValueError('Official CMS SFF page has no unique current posting link; refusing to infer currentness from monthly URL probes')
        url = next(iter(parser.links))
        payload = bounded_bytes(url, limit=32 * 1024 * 1024)
        if not payload.startswith(b'%PDF-'):
            raise ValueError('CMS current SFF resource is not a PDF')
        parsed = parse_sff_posting_updated_label(payload)
        if not parsed:
            import io
            import pdfplumber
            with pdfplumber.open(io.BytesIO(payload)) as pdf:
                text = pdf.pages[0].extract_text() if pdf.pages else ''
            labels = set(re.findall(r'Updated\s+([A-Za-z]+)(?:\s+\d{1,2},?)?\s+(20\d{2})', text or '', re.I))
            if len(labels) == 1:
                month, year = next(iter(labels))
                parsed = parse_sff_posting_updated_label(f'Updated {month} {year}'.encode())
        if not parsed:
            raise ValueError('Current CMS SFF PDF has no deterministic Updated release label')
        active = get_active_release(requirement.source_id, registry_path(root)) or {}
        return assess_raw_identity(requirement.source_id, active=active, remote_sha256=hashlib.sha256(payload).hexdigest(),
            current=dict(release_id=parsed[0], cms_release_vintage=parsed[0], snapshot_date=None,
                         publisher_url=url, publisher_landing_url=landing, publisher_identity_basis='official_page_link',
                         publisher_period_basis='CMS PDF Updated label', publisher_release_label=parsed[1]))
    raise ValueError('No deterministic publisher observation adapter is configured')


def verify_family_sources(family, release_id, control, *, root=None, observer=None):
    """Persist a complete attempt bound to exact candidate and ACTIVE hashes."""
    root = control_plane_root(root)
    before = family_readiness(family, control, root=root, release_id=release_id)
    if not before['manifest']:
        raise ValueError('No staged website release exists to verify')
    result = dict(schema_version=1, family=family, release_id=release_id,
                  stage_manifest_sha256=before['stage_sha'], observed_at=datetime.now(timezone.utc).isoformat(), sources={})
    observe = observer or observe_required_source
    for requirement, source in zip(FAMILIES[family].required_source_evidence, before['source_evidence']):
        try:
            evidence = observe(requirement, root=root)
        except Exception as exc:
            evidence = dict(status='ERROR', detail=str(exc), publisher_checked_at=result['observed_at'])
        result['sources'][requirement.source_id] = dict(evidence, source_id=requirement.source_id,
            staged_sha256=source['staged_sha256'], staged_raw_sha256=source['raw_sha256'],
            staged_active_sha256=source['active_sha256'],
            comparison_result='MATCH' if evidence.get('publisher_sha256') == source['raw_sha256'] and source['staged_sha256'] == source['active_sha256'] and source['raw_sha256'] else
            'DIFFERENT' if evidence.get('publisher_sha256') and source['raw_sha256'] else 'UNVERIFIED')
    if hashlib.sha256(stage_manifest_path(family, release_id, root=root).read_bytes()).hexdigest() != before['stage_sha']:
        raise ValueError('Candidate changed during observation; evidence not recorded')
    path = evidence_path(family, release_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    os.replace(temporary, path)
    return family_readiness(family, control, root=root, release_id=release_id)
