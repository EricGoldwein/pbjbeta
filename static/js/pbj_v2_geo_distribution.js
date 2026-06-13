/**
 * v2 — Regional analysis: geography distributions via peer column clicks + modal.
 * Uses /api/geo-distribution (PBJ facility-quarter distributions).
 */
(function (global) {
    'use strict';

    var LS_CENTRAL_TREND = 'pbjV2GeoCentralTrend';
    var _lastDistributionPayload = null;

    var METRIC_ROW_MAP = [
        { key: 'total_nurse_hprd', label: 'Total nursing HPRD', rowMatch: /Total nursing HPRD/i },
        { key: 'nurse_care_hprd', label: 'Direct-care nursing HPRD', rowMatch: /Direct-care nursing HPRD/i },
        { key: 'rn_hprd', label: 'RN HPRD', rowMatch: /^RN HPRD$/i },
        { key: 'rn_care_hprd', label: 'Direct RN HPRD', rowMatch: /Direct RN HPRD/i },
        { key: 'lpn_hprd', label: 'LPN HPRD', rowMatch: /^LPN HPRD$/i },
        { key: 'lpn_care_hprd', label: 'Direct LPN HPRD', rowMatch: /Direct LPN HPRD/i },
        { key: 'nurse_aide_hprd', label: 'Nurse aide HPRD', rowMatch: /Nurse aide HPRD/i },
        { key: 'avg_census', label: 'Average census', rowMatch: /Average census/i },
    ];

    function fmtGeoCount(n) {
        if (n == null || n === '' || isNaN(Number(n))) {
            return '?';
        }
        return Number(n).toLocaleString('en-US');
    }

    function selectedFacilityMeta(context) {
        var cfg = pageConfig();
        var full = String(cfg.exportFacilityFallback || cfg.exportFacilityDisplay || '').trim();
        if (!full) {
            return {
                label: 'This facility',
                fullName: 'This facility',
                title: '',
                tooltipTitle: 'This facility',
                ariaLabel: 'This facility',
                wasShortened: false,
            };
        }
        if (typeof global.getFacilityNameForContext === 'function') {
            return global.getFacilityNameForContext(full, context || 'legend');
        }
        if (typeof global.pbjFormatProviderCompactName === 'function') {
            var pack = global.pbjFormatProviderCompactName(full, 'short');
            return {
                label: pack.label,
                fullName: pack.fullName,
                title: pack.wasShortened ? pack.tooltipTitle : '',
                tooltipTitle: pack.tooltipTitle || pack.fullName,
                ariaLabel: pack.wasShortened ? pack.label + '. ' + pack.tooltipTitle : pack.label,
                wasShortened: !!pack.wasShortened,
            };
        }
        return {
            label: formatProviderDisplayName(full),
            fullName: full,
            title: '',
            tooltipTitle: full,
            ariaLabel: formatProviderDisplayName(full),
            wasShortened: false,
        };
    }

    function geoLabelForType(geo) {
        var labels = geoLabelsForScope();
        return labels[geo] || geo;
    }

    function geoLabelsForScope() {
        var facMeta = selectedFacilityMeta('legend');
        return {
            facility: facMeta.label,
            county: 'County',
            state: 'State',
            region: 'CMS region',
            national: 'National',
        };
    }

    var VALID_GEOGRAPHY_TYPES = {
        national: true,
        state: true,
        region: true,
        county: true,
        city: true,
    };

    function normalizeGeographyType(raw) {
        if (raw == null) {
            return 'state';
        }
        if (Array.isArray(raw)) {
            raw = raw.length > 4 ? raw[4] : null;
        }
        var geo = String(raw).trim().toLowerCase();
        if (!geo) {
            return 'state';
        }
        if (geo === 'urban') {
            return 'state';
        }
        if (geo === 'facility') {
            return null;
        }
        if (VALID_GEOGRAPHY_TYPES[geo]) {
            return geo;
        }
        return null;
    }

    /** Regional peer cells: globe icon to the right of the value (temporary uniform cue). */
    function geoDistIconClass() {
        return 'fa-globe';
    }

    var PEER_LOC_BY_GEO = {
        national: { field: 'state', header: 'State' },
        region: { field: 'state', header: 'State' },
        state: { field: 'county', header: 'County' },
        county: { field: 'city', header: 'City/town' },
        city: { field: 'city', header: 'City' },
    };

    function geoMetricKind(metricKey) {
        var k = String(metricKey || '').toLowerCase();
        if (k === 'contract_pct' || k.indexOf('contract') >= 0) {
            return 'pct';
        }
        if (k === 'avg_census' || k.indexOf('census') >= 0) {
            return 'census';
        }
        return 'hprd';
    }

    function formatGeoMetric(v, kind) {
        if (v == null || v === '' || isNaN(Number(v))) {
            return '—';
        }
        var n = Number(v);
        kind = kind || 'hprd';
        if (kind === 'pct') {
            return (
                n.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + '%'
            );
        }
        if (kind === 'census') {
            if (n >= 1000) {
                return n.toLocaleString('en-US', { maximumFractionDigits: 0 });
            }
            return n.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
        }
        if (kind === 'count') {
            return Math.round(n).toLocaleString('en-US');
        }
        var abs = Math.abs(n);
        var minD = abs < 1 ? 3 : 2;
        var maxD = abs < 1 ? 3 : abs < 10 ? 3 : 2;
        return n.toLocaleString('en-US', { minimumFractionDigits: minD, maximumFractionDigits: maxD });
    }

    function formatProviderDisplayName(name, opts) {
        if (typeof global.pbjFormatProviderDisplayName === 'function') {
            return global.pbjFormatProviderDisplayName(name, opts);
        }
        var raw = String(name || '').trim();
        if (!raw) {
            return '';
        }
        if (typeof global.capitalizeProviderName === 'function') {
            return global.capitalizeProviderName(raw);
        }
        var smallWords = ['and', 'of', 'at', 'by', 'the', 'a', 'an', 'in', 'on', 'for', 'to', 'with'];
        return raw
            .toLowerCase()
            .split(/\s+/)
            .map(function (word, index) {
                if (index === 0 || smallWords.indexOf(word) < 0) {
                    return word.charAt(0).toUpperCase() + word.slice(1);
                }
                return word;
            })
            .join(' ');
    }

    function formatProviderCompactLabel(name, level) {
        if (typeof global.pbjFormatProviderCompactName === 'function') {
            return global.pbjFormatProviderCompactName(name, level || 'short');
        }
        var label = formatProviderDisplayName(name);
        return {
            label: label,
            fullName: String(name || '').trim(),
            tooltipTitle: String(name || '').trim(),
            wasShortened: false,
        };
    }

    function formatPlaceLabel(raw) {
        var s = String(raw || '').trim();
        if (!s) {
            return '';
        }
        if (typeof global.pbjTitleCasePlaceName === 'function') {
            return global.pbjTitleCasePlaceName(s);
        }
        return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
    }

    function peerLocationSpec(geographyType) {
        return PEER_LOC_BY_GEO[String(geographyType || '').toLowerCase()] || PEER_LOC_BY_GEO.state;
    }

    function peerLocationCell(p, geographyType) {
        var spec = peerLocationSpec(geographyType);
        var v = p[spec.field];
        if (spec.field === 'state') {
            return v ? String(v).toUpperCase() : '—';
        }
        if (spec.field === 'county') {
            return v ? formatPlaceLabel(v) : '—';
        }
        if (spec.field === 'city') {
            var city = p.city ? formatPlaceLabel(p.city) : '';
            var county = p.county ? formatPlaceLabel(p.county) : '';
            if (city && county && city !== county) {
                return city + ', ' + county;
            }
            return city || county || '—';
        }
        return v ? String(v) : '—';
    }

    function parseGeoCellNumber(s) {
        var t = String(s || '')
            .replace(/,/g, '')
            .replace(/%$/, '')
            .trim();
        if (!t || t === '—') {
            return null;
        }
        var n = Number(t);
        return isNaN(n) ? null : n;
    }

    function wrapGeoDistCellHtml(geo, cellVal, metricKey) {
        var n = parseGeoCellNumber(cellVal);
        var disp = cellVal;
        if (n != null && metricKey) {
            disp = formatGeoMetric(n, geoMetricKind(metricKey));
        }
        return (
            '<span class="geo-rollup-dist-cell-inner">' +
            '<span class="geo-rollup-dist-val">' +
            esc(disp) +
            '</span>' +
            '<i class="fas ' +
            geoDistIconClass(geo) +
            ' geo-rollup-dist-icon" aria-hidden="true"></i>' +
            '</span>'
        );
    }

    function pageConfig() {
        var el = document.getElementById('pbj320-export-page');
        if (!el) {
            return {};
        }
        try {
            return JSON.parse(el.textContent || '{}');
        } catch (e) {
            return {};
        }
    }

    /** Free PBJ320 provider page (not premium dashboard). */
    function publicProviderPageHref(provnum) {
        var cfg = pageConfig();
        var base = String(cfg.pbjPublicProviderBaseUrl || 'https://www.pbj320.com/provider').replace(
            /\/$/,
            ''
        );
        return base + '/' + encodeURIComponent(String(provnum || '').trim());
    }

    /** Free PBJ320 state dashboard (e.g. https://www.pbj320.com/state/nj). */
    function statePageHref(abbr) {
        var a = String(abbr || '').trim().toUpperCase();
        if (!a) {
            return '';
        }
        var cfg = pageConfig();
        if (
            cfg.geoStateAbbr &&
            String(cfg.geoStateAbbr).toUpperCase() === a &&
            cfg.geoStateDashboardUrl
        ) {
            return String(cfg.geoStateDashboardUrl);
        }
        return 'https://www.pbj320.com/state/' + encodeURIComponent(a.toLowerCase());
    }

    function renderGeoContextStrip(data) {
        if (!data || data.error) {
            return '';
        }
        var geo = String(data.geography_type || '').toLowerCase();
        var parts = [];
        if (geo === 'state') {
            var st = data.facility_state_abbr || '';
            var stLong = pageConfig().geoStateLong || st;
            var href = statePageHref(st);
            if (href) {
                parts.push(
                    '<a href="' +
                        esc(href) +
                        '" target="_blank" rel="noopener" class="link-secondary link-offset-1">' +
                        esc(stLong) +
                        ' statewide</a>'
                );
            }
        } else if (geo === 'county') {
            var county = data.facility_county_label || data.geography_label || '';
            var st2 = data.facility_state_abbr || '';
            if (county) {
                parts.push(esc(formatPlaceLabel(county)) + (st2 ? ', ' + esc(st2) : ''));
            }
            var href2 = statePageHref(st2);
            if (href2) {
                parts.push(
                    '<a href="' +
                        esc(href2) +
                        '" target="_blank" rel="noopener" class="link-secondary link-offset-1">' +
                        esc(st2 || 'State') +
                        ' PBJ320</a>'
                );
            }
        } else if (geo === 'region') {
            var rn = data.cms_region_number;
            var states = data.cms_region_states || [];
            if (rn) {
                parts.push('Region ' + esc(String(rn)));
            }
            if (states.length) {
                parts.push(
                    states
                        .map(function (stAb) {
                            var h = statePageHref(stAb);
                            return h
                                ? '<a href="' +
                                      esc(h) +
                                      '" target="_blank" rel="noopener" class="link-secondary link-offset-1">' +
                                      esc(stAb) +
                                      '</a>'
                                : esc(stAb);
                        })
                        .join(' · ')
                );
            }
        }
        return parts.filter(Boolean).join(' · ');
    }

    function apiUrl(path, params) {
        var q = [];
        Object.keys(params || {}).forEach(function (k) {
            if (params[k] != null && params[k] !== '') {
                q.push(encodeURIComponent(k) + '=' + encodeURIComponent(String(params[k])));
            }
        });
        var qs = q.length ? '?' + q.join('&') : '';
        var full = path + qs;
        if (typeof global.pbjApiUrl === 'function') {
            return global.pbjApiUrl(full);
        }
        return full;
    }

    function fetchJson(path, params) {
        return fetch(apiUrl(path, params)).then(function (r) {
            return r.text().then(function (text) {
                var trimmed = String(text || '').trim();
                if (!r.ok) {
                    if (trimmed.indexOf('<') === 0) {
                        throw new Error(
                            'Regional distribution API returned HTML (HTTP ' +
                                r.status +
                                '). Restart the dashboard server or redeploy with geo_distribution_lib.py.'
                        );
                    }
                    try {
                        var errBody = JSON.parse(trimmed);
                        throw new Error(errBody.error || 'HTTP ' + r.status);
                    } catch (parseErr) {
                        if (parseErr && parseErr.message && parseErr.message.indexOf('HTTP') >= 0) {
                            throw parseErr;
                        }
                        throw new Error(trimmed.slice(0, 200) || 'HTTP ' + r.status);
                    }
                }
                try {
                    return JSON.parse(trimmed);
                } catch (eJson) {
                    throw new Error(
                        'Regional distribution API returned non-JSON. Ensure facility_quarterly_metrics.csv is bundled and /api/geo-distribution is available.'
                    );
                }
            });
        });
    }

    function esc(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function readGeoState() {
        if (!global.__pbjV2GeoDist) {
            global.__pbjV2GeoDist = {
                quarter: null,
                quarterLabel: '',
                context: null,
            };
        }
        return global.__pbjV2GeoDist;
    }

    function humanQuarterLabel(cy) {
        var s = String(cy || '')
            .trim()
            .toUpperCase()
            .replace(/^CY/i, '');
        var m = s.match(/^(\d{4})Q([1-4])$/);
        if (!m) {
            return String(cy || '').trim() || '—';
        }
        return 'Q' + m[2] + ' ' + m[1];
    }

    function currentQuarterFromRollup() {
        var mount = document.getElementById('geoRollupDynamicMount');
        if (!mount) {
            return { cy: null, label: '' };
        }
        if (mount.getAttribute('data-geo-scope-mode') === 'year') {
            var scopeYear = mount.getAttribute('data-geo-scope-year') || '';
            if (scopeYear) {
                return { cy: null, label: 'Calendar year ' + scopeYear, yearMode: true };
            }
        }
        var dataCy = mount.getAttribute('data-geo-cy-quarter');
        var dataLbl = mount.getAttribute('data-geo-quarter-label');
        if (dataCy) {
            return { cy: dataCy, label: dataLbl || humanQuarterLabel(dataCy) };
        }
        var sel = mount.querySelector('#geoRollupScopeSelect');
        if (!sel) {
            return { cy: null, label: '' };
        }
        var v = sel.value || 'latest';
        if (v.indexOf('year:') === 0) {
            var yy = v.slice(5);
            return { cy: null, label: 'Calendar year ' + yy, yearMode: true };
        }
        if (v.indexOf('q:') === 0) {
            var qk = v.slice(2);
            var mm = String(qk).match(/^(\d{4})Q([1-4])$/i);
            return {
                cy: mm ? 'CY' + mm[1] + 'Q' + mm[2] : qk,
                label: humanQuarterLabel(qk),
            };
        }
        return { cy: null, label: '' };
    }

    function injectPeriodBanner(mount) {
        if (!mount) {
            return;
        }
        if (mount.querySelector('#geoPeerComparisonPanel')) {
            return;
        }
        var q = currentQuarterFromRollup();
        var st = readGeoState();
        if (q.cy) {
            st.quarter = q.cy;
            st.quarterLabel = q.label || q.cy;
        }
        var existing = mount.querySelector('.geo-rollup-period-banner');
        var label = st.quarterLabel || (st.quarter ? st.quarter.replace(/^CY/i, '') : '');
        if (!label && q.label) {
            label = q.label;
        }
        var human = label;
        if (q.yearMode && q.label) {
            human = q.label;
        } else if (st.quarter) {
            human = humanQuarterLabel(st.quarter) || human;
        }
        var html =
            '<p class="geo-rollup-period-banner small mb-2 mb-md-1">' +
            '<strong class="text-body">Comparison period: ' +
            esc(human || '—') +
            '</strong>' +
            '<span class="text-muted"> · Click any globe value to see where this facility falls in the peer distribution.</span>' +
            '</p>';
        if (existing) {
            existing.outerHTML = html;
            return;
        }
        var panel = mount.querySelector('.geo-rollup-panel');
        if (panel) {
            panel.insertAdjacentHTML('beforebegin', html);
        } else {
            mount.insertAdjacentHTML('afterbegin', html);
        }
    }

    function metricKeyForRow(tr) {
        var fromAttr = tr.getAttribute('data-geo-metric');
        if (fromAttr) {
            return String(fromAttr).trim();
        }
        var th = tr.querySelector('th.geo-rollup-th');
        if (!th) {
            return null;
        }
        var txt = String(th.textContent || '');
        for (var i = 0; i < METRIC_ROW_MAP.length; i++) {
            if (METRIC_ROW_MAP[i].rowMatch.test(txt)) {
                return METRIC_ROW_MAP[i].key;
            }
        }
        return null;
    }

    function geoColumnKeys(mount) {
        var raw = mount.getAttribute('data-geo-column-keys') || 'facility,state,region,national';
        return raw.split(',').map(function (s) {
            return String(s || '').trim();
        }).filter(Boolean);
    }

    function bindDistCellDelegation(mount) {
        if (!mount || mount.getAttribute('data-geo-dist-mount-bound')) {
            return;
        }
        mount.setAttribute('data-geo-dist-mount-bound', '1');
        function openFromDistCell(cell) {
            if (!cell) {
                return;
            }
            var metric = cell.getAttribute('data-geo-metric');
            var geo = cell.getAttribute('data-geo-geography');
            if (metric && geo) {
                openDistributionModal(metric, geo);
            }
        }
        mount.addEventListener('click', function (ev) {
            var cell = ev.target && ev.target.closest ? ev.target.closest('td.geo-rollup-dist-cell') : null;
            openFromDistCell(cell);
        });
        mount.addEventListener('keydown', function (ev) {
            if (ev.key !== 'Enter' && ev.key !== ' ') {
                return;
            }
            var cell = ev.target && ev.target.closest ? ev.target.closest('td.geo-rollup-dist-cell') : null;
            if (!cell) {
                return;
            }
            ev.preventDefault();
            openFromDistCell(cell);
        });
    }

    function attachGeoDistributionCells(mount) {
        if (!mount) {
            return;
        }
        bindDistCellDelegation(mount);
        mount.querySelectorAll('.geo-dist-col-head').forEach(function (el) {
            el.remove();
        });
        mount.querySelectorAll('.geo-dist-action-cell').forEach(function (el) {
            el.remove();
        });
        var tbodies = mount.querySelectorAll('#geoPeerFullRollupWrap .geo-rollup-tbody, .geo-rollup-panel > .table-responsive .geo-rollup-tbody');
        if (!tbodies.length) {
            tbodies = mount.querySelectorAll('.geo-rollup-tbody');
        }
        tbodies.forEach(function (tbody) {
            decorateGeoRollupTbody(tbody, mount);
        });
    }

    function decorateGeoRollupTbody(tbody, mount) {
        if (!tbody) {
            return;
        }
        var colKeys = geoColumnKeys(mount);
        var rows = tbody.querySelectorAll('tr');
        rows.forEach(function (tr) {
            var metricKey = metricKeyForRow(tr);
            if (!metricKey) {
                return;
            }
            tr.setAttribute('data-geo-metric', metricKey);
            var tds = tr.querySelectorAll('td');
            tds.forEach(function (td, idx) {
                if (td.classList.contains('geo-rollup-dist-cell') && td.querySelector('.geo-rollup-dist-icon')) {
                    return;
                }
                td.classList.remove('geo-rollup-dist-cell', 'geo-rollup-dist-cell--active');
                td.removeAttribute('data-geo-geography');
                td.removeAttribute('data-geo-metric');
                td.removeAttribute('title');
                td.removeAttribute('role');
                td.removeAttribute('tabindex');
                if (idx >= colKeys.length) {
                    return;
                }
                var geo = colKeys[idx];
                if (geo === 'facility') {
                    return;
                }
                var cellVal = String(td.textContent || '').trim();
                if (!cellVal || cellVal === '—') {
                    return;
                }
                td.classList.add('geo-rollup-dist-cell');
                td.setAttribute('data-geo-geography', geo);
                td.setAttribute('data-geo-metric', metricKey);
                td.setAttribute('role', 'button');
                td.setAttribute('tabindex', '0');
                td.setAttribute(
                    'title',
                    'View ' +
                        (geoLabelForType(geo) || geo) +
                        ' distribution (histogram) for ' +
                        metricKey.replace(/_/g, ' ')
                );
                td.innerHTML = wrapGeoDistCellHtml(geo, cellVal, metricKey);
            });
        });
    }

    function fetchContext(quarter) {
        var cfg = pageConfig();
        return fetchJson('/api/geo-distribution/context', {
            provnum: cfg.exportCcn,
            quarter: quarter,
        });
    }

    function fetchDistribution(params) {
        var next = Object.assign({}, params || {});
        var geo = normalizeGeographyType(next.geography_type);
        if (!geo) {
            return Promise.resolve({
                error:
                    'Invalid geography_type. Expected national, state, region, county, or city.',
            });
        }
        next.geography_type = geo;
        return fetchJson('/api/geo-distribution', next);
    }

    function centralTrendMode() {
        try {
            var v = localStorage.getItem(LS_CENTRAL_TREND);
            if (v === 'mean' || v === 'both') {
                return v;
            }
        } catch (e) {}
        return 'median';
    }

    function fmtGeoNum(v, metricKey) {
        return formatGeoMetric(v, geoMetricKind(metricKey));
    }

    function geoPlotlyTickformat(metricKey, axisRole) {
        if (axisRole === 'yCount') {
            return ',d';
        }
        var kind = geoMetricKind(metricKey);
        if (kind === 'pct') {
            return ',.1f';
        }
        if (kind === 'census') {
            return ',.0f';
        }
        return ',.2f';
    }

    function geoDistValueAxisSpec(data, disp) {
        var metricKey = data.metric || '';
        var vals = [];
        if (disp === 'dotplot' && data.values_dotplot && data.values_dotplot.length) {
            vals = data.values_dotplot.slice();
        }
        if (disp === 'histogram' && data.bins && data.bins.length) {
            data.bins.forEach(function (b) {
                vals.push(Number(b.bin_start), Number(b.bin_end));
            });
        }
        ['facility_value', 'median', 'mean', 'min', 'max', 'threshold'].forEach(function (k) {
            if (data[k] != null && !isNaN(Number(data[k]))) {
                vals.push(Number(data[k]));
            }
        });
        vals = vals.filter(function (x) {
            return isFinite(x);
        });
        var spec = {
            title: data.metric_label || 'Value',
            automargin: true,
            tickformat: geoPlotlyTickformat(metricKey, 'x'),
            separatethousands: true,
        };
        if (!vals.length) {
            return spec;
        }
        var lo = Math.min.apply(null, vals);
        var hi = Math.max.apply(null, vals);
        var span = hi - lo;
        var pad = span > 0 ? Math.max(span * 0.06, 0.05) : 0.15;
        spec.range = [lo - pad, hi + pad];
        return spec;
    }

    function geoDistCountAxisSpec(data, disp) {
        if (disp !== 'histogram' || !data.bins || !data.bins.length) {
            return { showticklabels: false, automargin: true };
        }
        var maxC = 0;
        data.bins.forEach(function (b) {
            maxC = Math.max(maxC, Number(b.count) || 0);
        });
        return {
            title: 'Facilities',
            showticklabels: true,
            automargin: true,
            tickformat: geoPlotlyTickformat(null, 'yCount'),
            separatethousands: true,
            rangemode: 'tozero',
            range: maxC > 0 ? [0, maxC * 1.12] : undefined,
        };
    }

    function formatGeoStatsLine(data) {
        var n = data.n;
        var bits = [];
        if (n != null) {
            bits.push('n=' + fmtGeoCount(n) + ' peer' + (n === 1 ? '' : 's'));
        }
        if (data.percentile != null) {
            bits.push('~' + Math.round(data.percentile) + 'th percentile');
        }
        var mode = centralTrendMode();
        var mk = data.metric || '';
        if ((mode === 'mean' || mode === 'both') && data.mean != null) {
            bits.push('mean ' + fmtGeoNum(data.mean, mk));
        }
        if (data.median != null && (mode === 'median' || mode === 'both')) {
            bits.push('median ' + fmtGeoNum(data.median, mk));
        }
        if (data.share_facilities_strictly_below != null && data.facility_in_sample !== false) {
            bits.push('~' + data.share_facilities_strictly_below + '% of peers lower');
        }
        if (data.share_below_threshold != null && data.threshold != null) {
            bits.push(data.share_below_threshold + '% below staffing threshold');
        }
        if (data.n_excluded) {
            bits.push(fmtGeoCount(data.n_excluded) + ' excluded (missing data)');
        }
        if (data.small_sample_flag) {
            bits.push('small sample — interpret with caution');
        }
        return bits.join(' · ');
    }

    function formatGeoInterpLine(data) {
        if (data.error) {
            return '';
        }
        var cfg = pageConfig();
        var facMeta = selectedFacilityMeta('chart');
        var name = facMeta.label;
        if (data.facility_in_sample === false) {
            return (
                name +
                ' is not in the eligible peer sample for this chart (missing census, hours, or metric).'
            );
        }
        if (data.recommended_display_type === 'insufficient_sample') {
            return data.interpretation || 'Too few peers in this geography for a distribution.';
        }
        var fv = data.facility_value;
        var ml = data.metric_label || 'value';
        if (fv == null) {
            return '';
        }
        var mk = data.metric || '';
        var line = name + ': ' + fmtGeoNum(fv, mk) + ' ' + ml;
        if (data.median != null) {
            if (Number(fv) < Number(data.median)) {
                line += ' — below peer median (' + fmtGeoNum(data.median, mk) + ')';
            } else if (Number(fv) > Number(data.median)) {
                line += ' — above peer median (' + fmtGeoNum(data.median, mk) + ')';
            } else {
                line += ' — at peer median';
            }
        }
        if (data.small_sample_flag) {
            line += '. Small sample (n=' + fmtGeoCount(data.n) + ') — treat rank as directional only.';
        }
        return line;
    }

    function thresholdLegendLabel(data) {
        var src = String(data.threshold_source || '').toLowerCase();
        if (src.indexOf('macpac') >= 0) {
            return 'MACPAC minimum';
        }
        if (src.indexOf('auto') >= 0 || src.indexOf('default') >= 0) {
            return 'Staffing threshold';
        }
        if (data.threshold_source) {
            return String(data.threshold_source).replace(/_/g, ' ');
        }
        return 'Staffing threshold';
    }

    function renderGeoDistLegend(legEl, data, mode) {
        if (!legEl) {
            return;
        }
        var items = [];
        var mk = data.metric || '';
        var fv = data.facility_value;
        if (fv != null && !isNaN(Number(fv))) {
            var facMeta = selectedFacilityMeta('legend');
            items.push({
                kind: 'line',
                color: '#0d6efd',
                dash: '',
                label: facMeta.label,
                value: fmtGeoNum(fv, mk),
                title: facMeta.title || '',
            });
        }
        if ((mode === 'median' || mode === 'both') && data.median != null) {
            items.push({
                kind: 'line',
                color: '#198754',
                dash: '',
                label: 'Median',
                value: fmtGeoNum(data.median, mk),
            });
        }
        if ((mode === 'mean' || mode === 'both') && data.mean != null) {
            items.push({
                kind: 'line',
                color: '#fd7e14',
                dash: 'dash',
                label: 'Mean',
                value: fmtGeoNum(data.mean, mk),
            });
        }
        if (data.threshold != null && !isNaN(Number(data.threshold))) {
            items.push({
                kind: 'line',
                color: '#dc3545',
                dash: 'dot',
                label: thresholdLegendLabel(data),
                value: fmtGeoNum(data.threshold, mk),
            });
        }
        items.push({ kind: 'peer', label: 'Peers', value: 'one dot each' });
        legEl.innerHTML = items
            .map(function (it) {
                var swatch =
                    it.kind === 'peer'
                        ? '<span class="geo-dist-legend-swatch geo-dist-legend-swatch--peer" aria-hidden="true"></span>'
                        : '<span class="geo-dist-legend-swatch' +
                          (it.dash === 'dash'
                              ? ' geo-dist-legend-swatch--dash'
                              : it.dash === 'dot'
                                ? ' geo-dist-legend-swatch--dot'
                                : '') +
                          '" style="border-top-color:' +
                          esc(it.color) +
                          ';" aria-hidden="true"></span>';
                return (
                    '<li class="geo-dist-legend-item">' +
                    swatch +
                    '<span' +
                    (it.title ? ' title="' + esc(it.title) + '"' : '') +
                    '>' +
                    esc(it.label) +
                    (it.value ? ' <span class="font-monospace">' + esc(it.value) + '</span>' : '') +
                    '</span></li>'
                );
            })
            .join('');
        legEl.hidden = !items.length;
    }

    function clearGeoDistLegend() {
        var legEl = document.getElementById('geoDistributionLegend');
        if (legEl) {
            legEl.innerHTML = '';
            legEl.hidden = true;
        }
    }

    function renderChart(host, data) {
        host.innerHTML = '';
        if (!global.Plotly) {
            host.innerHTML = '<p class="small text-muted">Chart library unavailable.</p>';
            return;
        }
        var disp = data.recommended_display_type;
        var fv = data.facility_value;
        var mode = centralTrendMode();
        clearGeoDistLegend();
        if (disp === 'insufficient_sample') {
            host.innerHTML =
                '<p class="small text-warning mb-0">' + esc(data.interpretation || data.error || 'Insufficient sample.') + '</p>';
            return;
        }
        if (disp === 'peer_table' || (disp === 'dotplot' && (!data.values_dotplot || !data.values_dotplot.length))) {
            host.innerHTML =
                '<p class="small text-muted mb-2">Small sample (n=' +
                esc(fmtGeoCount(data.n)) +
                '). Peer table below.</p>';
            renderGeoDistLegend(document.getElementById('geoDistributionLegend'), data, mode);
            return;
        }
        var traces = [];
        if (disp === 'histogram' && data.bins && data.bins.length) {
            traces.push({
                type: 'bar',
                x: data.bins.map(function (b) {
                    return (b.bin_start + b.bin_end) / 2;
                }),
                y: data.bins.map(function (b) {
                    return b.count;
                }),
                name: 'Facilities',
                marker: { color: 'rgba(13, 110, 253, 0.55)' },
                hovertemplate:
                    '%{customdata[0]:.2f}–%{customdata[1]:.2f}<br>%{y} facilities<extra></extra>',
                customdata: data.bins.map(function (b) {
                    return [b.bin_start, b.bin_end];
                }),
            });
        } else if (disp === 'dotplot' && data.values_dotplot) {
            traces.push({
                type: 'scatter',
                mode: 'markers',
                x: data.values_dotplot,
                y: data.values_dotplot.map(function () {
                    return 1;
                }),
                name: 'Facilities',
                marker: { color: 'rgba(100, 116, 139, 0.65)', size: 7 },
                hovertemplate: '%{x}<extra></extra>',
            });
        }
        var shapes = [];
        if (fv != null && !isNaN(fv)) {
            shapes.push({
                type: 'line',
                x0: fv,
                x1: fv,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: { color: '#0d6efd', width: 2, dash: 'solid' },
            });
        }
        if ((mode === 'median' || mode === 'both') && data.median != null) {
            shapes.push({
                type: 'line',
                x0: data.median,
                x1: data.median,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: { color: '#198754', width: 2 },
            });
        }
        if ((mode === 'mean' || mode === 'both') && data.mean != null) {
            shapes.push({
                type: 'line',
                x0: data.mean,
                x1: data.mean,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: { color: '#fd7e14', width: 1.5, dash: 'dash' },
            });
        }
        if (data.threshold != null) {
            shapes.push({
                type: 'line',
                x0: data.threshold,
                x1: data.threshold,
                y0: 0,
                y1: 1,
                yref: 'paper',
                line: { color: '#dc3545', width: 1.5, dash: 'dot' },
            });
        }
        global.Plotly.newPlot(
            host,
            traces,
            {
                margin: { t: 8, r: 12, b: 48, l: 48 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(248,250,252,0.9)',
                xaxis: geoDistValueAxisSpec(data, disp),
                yaxis: geoDistCountAxisSpec(data, disp),
                shapes: shapes,
                showlegend: false,
            },
            { responsive: true, displayModeBar: false }
        );
        renderGeoDistLegend(document.getElementById('geoDistributionLegend'), data, mode);
    }

    var _geoPeerTableSort = { key: 'total_nurse_hprd', dir: 'desc' };
    var _geoPeerTablePage = 0;
    var PEER_TABLE_PAGE_SIZE = 10;

    function geoPeerSortValue(p, key, geographyType) {
        if (key === 'facility') {
            return String(formatProviderCompactLabel(p.provname || p.provnum, 'short').label || '').toLowerCase();
        }
        if (key === 'state') {
            return String(p.state || '').toLowerCase();
        }
        if (key === 'county' || key === 'city' || key === 'location') {
            return String(peerLocationCell(p, geographyType) || '').toLowerCase();
        }
        if (key === 'metric_value') {
            return Number(p.metric_value);
        }
        if (key === 'total_nurse_hprd') {
            return Number(p.total_nurse_hprd);
        }
        if (key === 'contract_pct') {
            return Number(p.contract_pct);
        }
        if (key === 'avg_census') {
            return Number(p.avg_census);
        }
        return Number(p[key]);
    }

    function sortGeoPeerRows(peers, sortKey, dir, geographyType) {
        var out = (peers || []).slice();
        var d = dir === 'asc' ? 1 : -1;
        out.sort(function (a, b) {
            if (a.is_focus && !b.is_focus) {
                return -1;
            }
            if (!a.is_focus && b.is_focus) {
                return 1;
            }
            var av = geoPeerSortValue(a, sortKey, geographyType);
            var bv = geoPeerSortValue(b, sortKey, geographyType);
            if (
                sortKey === 'facility' ||
                sortKey === 'state' ||
                sortKey === 'county' ||
                sortKey === 'city' ||
                sortKey === 'location'
            ) {
                av = String(av || '');
                bv = String(bv || '');
                if (av < bv) {
                    return -1 * d;
                }
                if (av > bv) {
                    return 1 * d;
                }
                return 0;
            }
            var an = isNaN(av) ? null : av;
            var bn = isNaN(bv) ? null : bv;
            if (an == null && bn == null) {
                return 0;
            }
            if (an == null) {
                return 1;
            }
            if (bn == null) {
                return -1;
            }
            return (an - bn) * d;
        });
        return out;
    }

    function renderPeerTableBodyRows(peers, data, geoType) {
        var metricKey = String(data.metric || '').toLowerCase();
        var showMetricCol = metricKey !== 'total_nurse_hprd' && metricKey !== 'contract_pct';
        return peers
            .map(function (p) {
                var href = publicProviderPageHref(p.provnum);
                var namePack = formatProviderCompactLabel(p.provname || p.provnum, 'short');
                var name = esc(namePack.label);
                var nameTitle = namePack.wasShortened
                    ? ' title="' + esc(namePack.tooltipTitle) + '" aria-label="' + esc(namePack.label + '. ' + namePack.tooltipTitle) + '"'
                    : '';
                var cells =
                    '<tr' +
                    (p.is_focus ? ' class="table-primary"' : '') +
                    '><td><a href="' +
                    esc(href) +
                    '" target="_blank" rel="noopener"' +
                    nameTitle +
                    '>' +
                    name +
                    '</a></td>' +
                    '<td class="text-muted small">' +
                    esc(peerLocationCell(p, geoType)) +
                    '</td>';
                if (showMetricCol) {
                    cells +=
                        '<td class="text-end font-monospace">' +
                        esc(fmtGeoNum(p.metric_value, data.metric)) +
                        '</td>';
                }
                cells +=
                    '<td class="text-end font-monospace">' +
                    esc(fmtGeoNum(p.total_nurse_hprd, 'total_nurse_hprd')) +
                    '</td>' +
                    '<td class="text-end font-monospace">' +
                    esc(fmtGeoNum(p.contract_pct, 'contract_pct')) +
                    '</td>' +
                    '<td class="text-end font-monospace">' +
                    esc(fmtGeoNum(p.avg_census, 'avg_census')) +
                    '</td></tr>';
                return cells;
            })
            .join('');
    }

    function renderPeerTable(host, data) {
        var geoType = String(data.geography_type || 'state').toLowerCase();
        if (geoType === 'national') {
            host.innerHTML =
                '<p class="small text-muted mb-0">National scope shows the distribution chart only — a facility peer list is not applicable.</p>';
            return;
        }
        var peers = data.peers || [];
        if (!peers.length) {
            host.innerHTML = '';
            return;
        }
        var geoLabel = data.geography_label || geoLabelForType(geoType) || geoType;
        var metricKey = String(data.metric || '').toLowerCase();
        var showMetricCol = metricKey !== 'total_nurse_hprd' && metricKey !== 'contract_pct';
        var locSpec = peerLocationSpec(geoType);
        var locSortKey = locSpec.field === 'state' ? 'state' : locSpec.field === 'county' ? 'county' : 'location';
        var sorted = sortGeoPeerRows(peers, _geoPeerTableSort.key, _geoPeerTableSort.dir, geoType);
        var nTotal = data.n != null ? Number(data.n) : sorted.length;
        var usePagination = geoType === 'state' && sorted.length > PEER_TABLE_PAGE_SIZE;
        var pageSize = usePagination ? PEER_TABLE_PAGE_SIZE : sorted.length;
        var pageCount = usePagination ? Math.max(1, Math.ceil(sorted.length / pageSize)) : 1;
        if (_geoPeerTablePage >= pageCount) {
            _geoPeerTablePage = pageCount - 1;
        }
        if (_geoPeerTablePage < 0) {
            _geoPeerTablePage = 0;
        }
        var pageStart = usePagination ? _geoPeerTablePage * pageSize : 0;
        var pageRows = sorted.slice(pageStart, pageStart + pageSize);
        var capNote = usePagination
            ? 'Page ' +
              (_geoPeerTablePage + 1) +
              ' of ' +
              pageCount +
              ' · ' +
              sorted.length +
              ' peers · click headers to sort'
            : 'Showing ' +
              sorted.length +
              ' peer' +
              (sorted.length === 1 ? '' : 's') +
              (nTotal && nTotal > sorted.length && data.peers_capped
                  ? ' (closest to this facility, of ' + nTotal + ' in sample)'
                  : '') +
              ' · click headers to sort';

        function sortBtn(key, label, extraClass) {
            var active = _geoPeerTableSort.key === key;
            var arrow = active ? (_geoPeerTableSort.dir === 'asc' ? ' ▲' : ' ▼') : '';
            return (
                '<button type="button" class="btn btn-link btn-sm p-0 text-decoration-none geo-dist-peer-sort' +
                (active ? ' fw-semibold' : '') +
                (extraClass ? ' ' + extraClass : '') +
                '" data-geo-peer-sort="' +
                esc(key) +
                '">' +
                esc(label) +
                arrow +
                '</button>'
            );
        }

        var head =
            '<th scope="col">' +
            sortBtn('facility', 'Facility') +
            '</th><th scope="col">' +
            sortBtn(locSortKey, locSpec.header) +
            '</th>';
        if (showMetricCol) {
            head +=
                '<th scope="col" class="text-end">' +
                sortBtn('metric_value', data.metric_label || 'Value', 'text-end') +
                '</th>';
        }
        head +=
            '<th scope="col" class="text-end">' +
            sortBtn('total_nurse_hprd', 'Total nurse HPRD', 'text-end') +
            '</th><th scope="col" class="text-end">' +
            sortBtn('contract_pct', 'Contract %', 'text-end') +
            '</th><th scope="col" class="text-end">' +
            sortBtn('avg_census', 'Census', 'text-end') +
            '</th>';

        var pagerHtml = '';
        if (usePagination) {
            pagerHtml =
                '<div class="d-flex flex-wrap align-items-center gap-2 mt-1 geo-dist-peer-pager">' +
                '<button type="button" class="btn btn-sm btn-outline-secondary geo-dist-peer-page" data-dir="prev"' +
                (_geoPeerTablePage <= 0 ? ' disabled' : '') +
                '>Prev</button>' +
                '<span class="small text-muted">Page ' +
                (_geoPeerTablePage + 1) +
                ' of ' +
                pageCount +
                '</span>' +
                '<button type="button" class="btn btn-sm btn-outline-secondary geo-dist-peer-page" data-dir="next"' +
                (_geoPeerTablePage >= pageCount - 1 ? ' disabled' : '') +
                '>Next</button>' +
                '</div>';
        }

        host.innerHTML =
            '<p class="small text-muted mb-1"><strong>' +
            esc(geoLabel) +
            '</strong> · ' +
            esc(capNote) +
            '</p>' +
            '<div class="table-responsive geo-dist-peer-scroll"><table class="table table-sm table-striped mb-0 geo-dist-peer-table">' +
            '<thead><tr>' +
            head +
            '</tr></thead><tbody id="geoDistPeerTableBody">' +
            renderPeerTableBodyRows(pageRows, data, geoType) +
            '</tbody></table></div>' +
            pagerHtml;

        host.querySelectorAll('[data-geo-peer-sort]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var key = btn.getAttribute('data-geo-peer-sort');
                if (!key) {
                    return;
                }
                if (_geoPeerTableSort.key === key) {
                    _geoPeerTableSort.dir = _geoPeerTableSort.dir === 'asc' ? 'desc' : 'asc';
                } else {
                    _geoPeerTableSort.key = key;
                    _geoPeerTableSort.dir =
                        key === 'facility' || key === 'state' || key === 'county' || key === 'location'
                            ? 'asc'
                            : 'desc';
                }
                _geoPeerTablePage = 0;
                renderPeerTable(host, data);
            });
        });
        host.querySelectorAll('.geo-dist-peer-page').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var dir = btn.getAttribute('data-dir');
                if (dir === 'prev' && _geoPeerTablePage > 0) {
                    _geoPeerTablePage -= 1;
                    renderPeerTable(host, data);
                } else if (dir === 'next' && _geoPeerTablePage < pageCount - 1) {
                    _geoPeerTablePage += 1;
                    renderPeerTable(host, data);
                }
            });
        });
    }

    function openDistributionModal(metricKey, geographyType) {
        var modalEl = document.getElementById('geoDistributionModal');
        if (!modalEl || typeof global.bootstrap === 'undefined') {
            return;
        }
        var st = readGeoState();
        var cfg = pageConfig();
        var q = currentQuarterFromRollup();
        var quarter = st.quarter || q.cy;
        if (!quarter) {
            window.alert('Select a calendar quarter in the comparison period dropdown first.');
            return;
        }
        var geo = normalizeGeographyType(geographyType);
        if (!geo) {
            window.alert(
                'Could not open distribution: invalid geography. Click a peer bar (National, State, County, or Region).'
            );
            return;
        }
        var geoValue = '';
        if (geo === 'county') {
            var rollupPage = global.PBJ320_GEO_ROLLUP_PAGE || {};
            geoValue = String(
                rollupPage.facility_county || rollupPage.county_label || ''
            ).trim();
        }
        var title = document.getElementById('geoDistributionModalLabel');
        var body = document.getElementById('geoDistributionModalBody');
        var interp = document.getElementById('geoDistributionInterp');
        var stats = document.getElementById('geoDistributionStats');
        var chartHost = document.getElementById('geoDistributionChart');
        var peerHost = document.getElementById('geoDistributionPeers');
        var contextEl = document.getElementById('geoDistributionContext');
        if (title) {
            title.textContent = 'Distribution — loading…';
        }
        if (contextEl) {
            contextEl.textContent = '';
            contextEl.classList.add('d-none');
        }
        if (body) {
            body.setAttribute('aria-busy', 'true');
        }
        var threshInput = document.getElementById('geoDistThresholdInput');
        var threshOverride = threshInput && threshInput.value ? threshInput.value : '';
        _geoPeerTablePage = 0;
        fetchDistribution({
            provnum: cfg.exportCcn,
            quarter: quarter,
            metric: metricKey,
            geography_type: geo,
            geography_value: geoValue,
            facility_name: cfg.exportFacilityDisplay || cfg.exportFacilityFallback,
            threshold_override: threshOverride,
        })
            .then(function (data) {
                _lastDistributionPayload = data.error ? null : data;
                if (data.error) {
                    if (interp) {
                        interp.textContent = data.error;
                    }
                    if (stats) {
                        stats.textContent = '';
                    }
                    if (chartHost) {
                        chartHost.innerHTML = '';
                    }
                    clearGeoDistLegend();
                    if (peerHost) {
                        peerHost.innerHTML = '';
                    }
                    return;
                }
                if (title) {
                    title.textContent =
                        (data.metric_label || metricKey) +
                        ' · ' +
                        (data.geography_label || geo) +
                        ' · ' +
                        (data.quarter_label || quarter);
                }
                if (stats) {
                    stats.textContent = formatGeoStatsLine(data);
                }
                if (contextEl) {
                    var ctxHtml = renderGeoContextStrip(data);
                    if (ctxHtml) {
                        contextEl.innerHTML = ctxHtml;
                        contextEl.classList.remove('d-none');
                    } else {
                        contextEl.textContent = '';
                        contextEl.classList.add('d-none');
                    }
                }
                if (interp) {
                    interp.textContent = formatGeoInterpLine(data);
                }
                if (chartHost) {
                    renderChart(chartHost, data);
                }
                if (peerHost) {
                    renderPeerTable(peerHost, data);
                }
            })
            .catch(function (err) {
                if (interp) {
                    interp.textContent = String(err && err.message ? err.message : err);
                }
                if (chartHost) {
                    chartHost.innerHTML = '';
                }
                clearGeoDistLegend();
                if (peerHost) {
                    peerHost.innerHTML = '';
                }
            })
            .finally(function () {
                if (body) {
                    body.removeAttribute('aria-busy');
                }
                var m = global.bootstrap.Modal.getOrCreateInstance(modalEl);
                m.show();
            });
    }

    function enhanceRollupMount() {
        var mount = document.getElementById('geoRollupDynamicMount');
        if (!mount || !mount.querySelector('.geo-rollup-panel')) {
            return;
        }
        injectPeriodBanner(mount);
        attachGeoDistributionCells(mount);
        var st = readGeoState();
        var q = currentQuarterFromRollup();
        if (!q.cy) {
            return;
        }
        fetchContext(q.cy)
            .then(function (ctx) {
                if (ctx.error) {
                    return;
                }
                st.context = ctx;
                st.quarter = ctx.quarter || q.cy;
            })
            .catch(function () {
                /* context optional */
            });
    }

    function patchGeoRollupRefresh() {
        if (global.__pbjV2GeoDistPatched) {
            return true;
        }
        var orig = global.pbjGeoRollupRefresh;
        if (typeof orig !== 'function') {
            return false;
        }
        global.__pbjV2GeoDistPatched = true;
        global.pbjGeoRollupRefresh = function (rows) {
            orig(rows);
            enhanceRollupMount();
        };
        return true;
    }

    function waitForGeoRollupPatch(attemptsLeft) {
        if (patchGeoRollupRefresh()) {
            if (global.__pbjGeoRollupLastRows) {
                enhanceRollupMount();
            }
            return;
        }
        if (attemptsLeft <= 0) {
            return;
        }
        global.setTimeout(function () {
            waitForGeoRollupPatch(attemptsLeft - 1);
        }, 50);
    }

    function bindCentralTrendToggles() {
        document.querySelectorAll('input[name="geoDistCentralTrend"]').forEach(function (inp) {
            inp.addEventListener('change', function () {
                try {
                    localStorage.setItem(LS_CENTRAL_TREND, inp.value);
                } catch (e) {}
                var chart = document.getElementById('geoDistributionChart');
                if (chart && _lastDistributionPayload) {
                    renderChart(chart, _lastDistributionPayload);
                }
            });
        });
        var mode = centralTrendMode();
        var pick = document.querySelector('input[name="geoDistCentralTrend"][value="' + mode + '"]');
        if (pick) {
            pick.checked = true;
        }
    }

    function init() {
        waitForGeoRollupPatch(80);
        bindCentralTrendToggles();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    global.PbjV2GeoDistribution = {
        openDistributionModal: openDistributionModal,
        enhanceRollupMount: enhanceRollupMount,
        wrapDistCellHtml: wrapGeoDistCellHtml,
        formatMetric: formatGeoMetric,
        formatProviderDisplayName: formatProviderDisplayName,
        formatProviderCompactLabel: formatProviderCompactLabel,
        geoMetricKind: geoMetricKind,
        peerLocationCell: peerLocationCell,
        peerLocationSpec: peerLocationSpec,
        normalizeGeographyType: normalizeGeographyType,
    };
})(typeof window !== 'undefined' ? window : this);
