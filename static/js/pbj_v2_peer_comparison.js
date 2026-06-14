/**
 * v2 — Peer & Regional Comparison: benchmark chart, view tabs, peer-scope logic, enhanced table.
 */
(function (global) {
    'use strict';

    var LS_VIEW = 'pbjV2PeerCompView_v2';
    var LS_METRIC = 'pbjV2PeerCompMetric';
    var LS_PEER_SCOPE = 'pbjV2PeerCompScope';

    var THRESHOLDS = { county: 10, urban: 10, state: 20 };

    var METRIC_GROUPS = [
        { id: 'core', label: 'Core staffing' },
        { id: 'rn', label: 'RN coverage' },
        { id: 'acuity', label: 'Acuity-adjusted' },
        { id: 'workforce', label: 'Workforce' },
        { id: 'census', label: 'Census' },
    ];

    var METRIC_DEFS = [
        {
            key: 'total_nurse_hprd',
            label: 'Total nursing HPRD',
            group: 'core',
            kind: 'hprd',
            distKey: 'total_nurse_hprd',
            fsKey: 'total_nurse_hprd',
            liteSuffix: 'total_nurse_hprd',
            title: 'Total nursing HPRD from PBJ (RN + LPN + aide, incl. admin/DON).',
        },
        {
            key: 'nurse_care_hprd',
            label: 'Direct-care nursing HPRD',
            group: 'core',
            kind: 'hprd',
            distKey: 'nurse_care_hprd',
            fsKey: 'direct_care_hprd',
            liteSuffix: 'nurse_care_hprd',
            title: 'Facility: PBJ direct-care HPRD. Peers: CMS Nurse care HPRD rollup.',
        },
        {
            key: 'rn_hprd',
            label: 'RN HPRD',
            group: 'rn',
            kind: 'hprd',
            distKey: 'rn_hprd',
            fsKey: 'rn_hprd',
            liteSuffix: 'rn_hprd',
        },
        {
            key: 'rn_care_hprd',
            label: 'Direct RN HPRD',
            group: 'rn',
            kind: 'hprd',
            distKey: 'rn_care_hprd',
            fsKey: 'rn_direct_care_hprd',
            liteSuffix: 'rn_care_hprd',
        },
        {
            key: 'lpn_hprd',
            label: 'LPN HPRD',
            group: 'core',
            kind: 'hprd',
            distKey: 'lpn_hprd',
            fsKey: 'lpn_hprd',
            liteSuffix: 'lpn_hprd',
        },
        {
            key: 'lpn_care_hprd',
            label: 'Direct LPN HPRD',
            group: 'core',
            kind: 'hprd',
            distKey: 'lpn_care_hprd',
            fsKey: 'lpn_direct_care_hprd',
            liteSuffix: 'lpn_care_hprd',
        },
        {
            key: 'nurse_aide_hprd',
            label: 'Nurse aide HPRD',
            group: 'core',
            kind: 'hprd',
            distKey: 'nurse_aide_hprd',
            fsKey: 'nurse_aide_hprd',
            liteSuffix: 'nurse_aide_hprd',
        },
        {
            key: 'contract_pct',
            label: 'Contract share',
            group: 'workforce',
            kind: 'pct',
            distKey: 'contract_pct',
            fsKey: 'contract_pct',
            liteSuffix: 'contract_pct',
            title: 'Contract hours as % of total nurse hours (PBJ).',
        },
        {
            key: 'avg_census',
            label: 'Average census',
            group: 'census',
            kind: 'census',
            distKey: 'avg_census',
            fsKey: 'avg_census',
            liteSuffix: 'avg_census',
        },
        {
            key: 'cms_case_mix_pct',
            label: '% CMS case-mix expected',
            group: 'acuity',
            kind: 'pct',
            distKey: null,
            acuityKey: 'case_mix_direct_pct',
        },
        {
            key: 'harrington_pct',
            label: '% Harrington expected',
            group: 'acuity',
            kind: 'pct',
            distKey: null,
            acuityKey: 'harrington_total_pct',
        },
        {
            key: 'acuity_staffing_gap',
            label: 'Acuity-adjusted staffing gap',
            group: 'acuity',
            kind: 'hprd',
            distKey: null,
            acuityKey: 'harrington_gap_hprd',
        },
        {
            key: 'rn_share_pct',
            label: 'RN share of nursing hours',
            group: 'rn',
            kind: 'pct',
            distKey: null,
            computedShare: 'rn',
        },
        {
            key: 'aide_share_pct',
            label: 'Aide share of nursing hours',
            group: 'core',
            kind: 'pct',
            distKey: null,
            computedShare: 'aide',
        },
        {
            key: 'headcount_per_100',
            label: 'Reported employee headcount per 100 residents',
            group: 'workforce',
            kind: 'count',
            distKey: null,
            facilityOnly: true,
        },
    ];

    var PEER_COMP_METRIC_ORDER = [
        'total_nurse_hprd',
        'nurse_care_hprd',
        'rn_hprd',
        'rn_care_hprd',
        'lpn_hprd',
        'lpn_care_hprd',
        'nurse_aide_hprd',
        'contract_pct',
        'avg_census',
    ];

    function peerCompMetricSortIndex(key) {
        var i = PEER_COMP_METRIC_ORDER.indexOf(key);
        return i >= 0 ? i : 100 + PEER_COMP_METRIC_ORDER.length;
    }

    function sortPeerCompMetrics(metrics) {
        return (metrics || []).slice().sort(function (a, b) {
            var da = peerCompMetricSortIndex(a.key);
            var db = peerCompMetricSortIndex(b.key);
            if (da !== db) {
                return da - db;
            }
            return String(a.label || '').localeCompare(String(b.label || ''));
        });
    }

    function esc(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function fmtVal(v, kind) {
        if (global.PbjV2GeoDistribution && typeof global.PbjV2GeoDistribution.formatMetric === 'function') {
            return global.PbjV2GeoDistribution.formatMetric(v, kind || 'hprd');
        }
        if (v == null || v === '' || isNaN(Number(v))) {
            return '—';
        }
        return String(v);
    }

    function fmtCount(n) {
        if (n == null || n === '' || isNaN(Number(n))) {
            return '—';
        }
        return Number(n).toLocaleString('en-US');
    }

    function shortenFacilityName(full, context) {
        var name = full || 'This facility';
        if (typeof global.getFacilityNameForContext === 'function') {
            return global.getFacilityNameForContext(name, context || 'comparison').label || name;
        }
        if (typeof global.getFacilityDisplayName === 'function') {
            var pack = global.getFacilityDisplayName(name);
            return pack.shortName || pack.displayName || name;
        }
        if (typeof global.pbjFormatProviderCompactName === 'function') {
            return global.pbjFormatProviderCompactName(name, 'short').label || name;
        }
        if (typeof global.smartFacilityShortName === 'function') {
            return global.smartFacilityShortName(name, { level: 'short' }) || name;
        }
        return name;
    }

    function facilityMetaFromFull(full, context) {
        if (typeof global.getFacilityNameForContext === 'function') {
            return global.getFacilityNameForContext(full || 'This facility', context || 'table');
        }
        var label = shortenFacilityName(full, context);
        return {
            label: label,
            title: full && full !== label ? full : '',
            tooltipTitle: full && full !== label ? 'CMS name: ' + full : full || '',
            ariaLabel: label,
        };
    }

    function facilityInsightName(modelOrName) {
        if (modelOrName && modelOrName.geoLabels) {
            if (modelOrName.geoLabels.facilityShort) {
                return modelOrName.geoLabels.facilityShort;
            }
            var glShort = modelOrName.geoLabels.facility;
            var glFull = modelOrName.geoLabels.facilityFull;
            if (glShort && glFull && glShort !== glFull) {
                return glShort;
            }
            return shortenFacilityName(glShort || glFull, 'comparison');
        }
        return shortenFacilityName(modelOrName || 'This facility', 'comparison');
    }

    function facilityTableHeaderMeta(model) {
        var full =
            (model && model.geoLabels && model.geoLabels.facilityFull) ||
            (model && model.geoLabels && model.geoLabels.facility) ||
            'This facility';
        var meta = facilityMetaFromFull(full, 'table');
        return {
            label: meta.label,
            title: meta.title || meta.tooltipTitle || '',
            ariaLabel: meta.ariaLabel || meta.label,
        };
    }

    function benchmarkValueColumnLabel(peerLabel) {
        return peerLabel + ' average';
    }

    function benchmarkTableColumnLabel(peerLabel, compact) {
        var label = String(peerLabel || 'Peer').trim();
        if (!compact) {
            return benchmarkValueColumnLabel(label);
        }
        if (!label) {
            return 'Peer';
        }
        if (label.length <= 14) {
            return label;
        }
        return label
            .replace(/\s+County$/i, ' Co.')
            .replace(/\s+Parish$/i, ' Par.')
            .replace(/^State of\s+/i, '')
            .slice(0, 14);
    }

    function metricLabelLower(metric) {
        var label = metric && metric.label ? metric.label : 'this metric';
        return String(label).toLowerCase();
    }

    function diffCellHtml(diff, kind) {
        if (diff == null) {
            return '—';
        }
        var formatted = fmtVal(diff, kind);
        if (kind !== 'hprd') {
            return diff > 0 ? '+' + formatted : formatted;
        }
        if (diff > 0) {
            return '<span class="text-success">+' + formatted + '</span>';
        }
        if (diff < 0) {
            return '<span class="text-danger">' + formatted + '</span>';
        }
        return formatted;
    }

    function updateGeoPeerChartLegend(panel, model, primaryScope) {
        if (!panel) {
            return;
        }
        var leg = panel.querySelector('#geoPeerChartLegend');
        if (!leg) {
            return;
        }
        var fac = facilityInsightName(model);
        var facMeta = facilityMetaFromFull(
            (model.geoLabels && model.geoLabels.facilityFull) || fac,
            'legend'
        );
        var bench = (model.geoLabels && model.geoLabels[primaryScope]) || primaryScope;
        leg.innerHTML =
            '<span class="geo-peer-chart-legend-item"' +
            (facMeta.title ? ' title="' + esc(facMeta.title) + '"' : '') +
            ' aria-label="' +
            esc(facMeta.ariaLabel || facMeta.label) +
            '"><span class="geo-peer-chart-swatch geo-peer-chart-swatch--facility" aria-hidden="true"></span>' +
            esc(facMeta.label) +
            '</span>' +
            '<span class="geo-peer-chart-legend-item"><span class="geo-peer-chart-swatch geo-peer-chart-swatch--benchmark" aria-hidden="true"></span>' +
            esc(bench) +
            '</span>' +
            '<span class="geo-peer-chart-legend-item"><span class="geo-peer-chart-swatch geo-peer-chart-swatch--other" aria-hidden="true"></span>Other geographies</span>';
    }

    function renderAllGeographiesDisclosure(model, ctx) {
        return (
            '<div class="geo-peer-all-geo-disclosure pbj-chart-utility-disclosure mt-2">' +
            '<button class="pbj-chart-utility-disclosure-btn collapsed w-100" type="button" data-bs-toggle="collapse" data-bs-target="#geoPeerAllGeographiesCollapse" aria-expanded="false" aria-controls="geoPeerAllGeographiesCollapse" id="geoPeerAllGeographiesBtn">' +
            '<span class="pbj-chart-utility-disclosure-main">' +
            '<i class="fas fa-table pbj-chart-utility-disclosure-icon" aria-hidden="true"></i>' +
            '<span class="pbj-chart-utility-disclosure-copy">' +
            '<span class="pbj-chart-utility-disclosure-title">All geography comparisons</span>' +
            '<span class="pbj-chart-utility-disclosure-helper">County, state, region, and national averages. Click 🌐 values to view distributions where available.</span>' +
            '</span></span>' +
            '<span class="pbj-chart-utility-disclosure-action" aria-hidden="true">' +
            '<span class="pbj-chart-utility-action-expand">Expand <span class="pbj-disclosure-affordance">›</span></span>' +
            '<span class="pbj-chart-utility-action-collapse">Collapse <span class="pbj-disclosure-affordance">↑</span></span>' +
            '</span></button>' +
            '<div class="collapse" id="geoPeerAllGeographiesCollapse" aria-labelledby="geoPeerAllGeographiesBtn">' +
            '<div class="pbj-chart-utility-disclosure-body p-0">' +
            renderGeoRollupTable(model, ctx) +
            '</div></div></div>'
        );
    }

    function numOrNull(v) {
        if (v == null || v === '' || isNaN(Number(v))) {
            return null;
        }
        return Number(v);
    }

    function lsGet(key, fallback) {
        try {
            var v = localStorage.getItem(key);
            return v != null && v !== '' ? v : fallback;
        } catch (e) {
            return fallback;
        }
    }

    function lsSet(key, val) {
        try {
            localStorage.setItem(key, val);
        } catch (e) {}
    }

    /** Chart is default; table only when explicitly chosen. Rankings tab removed. */
    function normalizePeerCompView(raw) {
        return String(raw || '').toLowerCase() === 'table' ? 'table' : 'chart';
    }

    function geoLiteMetric(lite, prefix, suffix) {
        if (!lite || !prefix) {
            return null;
        }
        return lite[prefix + '_' + suffix];
    }

    function geoPeerMetric(lite, prefix, suffix, fallbackPrefix) {
        var primary = geoLiteMetric(lite, prefix, suffix);
        if (primary != null && primary !== '' && !isNaN(primary)) {
            return Number(primary);
        }
        if (fallbackPrefix && fallbackPrefix !== prefix) {
            var fb = geoLiteMetric(lite, fallbackPrefix, suffix);
            if (fb != null && fb !== '' && !isNaN(fb)) {
                return Number(fb);
            }
        }
        return null;
    }

    function computeWeekendMetrics(subRows) {
        var wkSum = 0,
            wkN = 0,
            weSum = 0,
            weN = 0,
            wkRn = 0,
            wkRnN = 0,
            weRn = 0,
            weRnN = 0;
        (subRows || []).forEach(function (r) {
            if (!r) {
                return;
            }
            var cen = parseFloat(r.MDScensus);
            if (!(cen > 0)) {
                return;
            }
            var dow = String(r.DayOfWeek || '').toLowerCase();
            var isWe = dow === 'saturday' || dow === 'sunday';
            var hrsRN =
                (parseFloat(r.Hrs_RN) || 0) +
                (parseFloat(r.Hrs_RNadmin) || 0) +
                (parseFloat(r.Hrs_RNDON) || 0);
            var hrsLPN = (parseFloat(r.Hrs_LPN) || 0) + (parseFloat(r.Hrs_LPNadmin) || 0);
            var hrsAide =
                (parseFloat(r.Hrs_CNA) || 0) +
                (parseFloat(r.Hrs_NAtrn) || 0) +
                (parseFloat(r.Hrs_MedAide) || 0);
            var totalHrs = hrsRN + hrsLPN + hrsAide;
            var totalHprd = totalHrs / cen;
            var rnHprd = hrsRN / cen;
            if (isWe) {
                weSum += totalHprd;
                weN += 1;
                weRn += rnHprd;
                weRnN += 1;
            } else {
                wkSum += totalHprd;
                wkN += 1;
                wkRn += rnHprd;
                wkRnN += 1;
            }
        });
        if (!wkN || !weN) {
            return null;
        }
        var wkAvg = wkSum / wkN;
        var weAvg = weSum / weN;
        var gapPct = wkAvg !== 0 ? ((weAvg - wkAvg) / wkAvg) * 100 : null;
        return {
            weekend_total_hprd: weAvg,
            weekend_rn_hprd: weRnN ? weRn / weRnN : null,
            weekday_weekend_gap_pct: gapPct,
        };
    }

    function extendFacilityStats(fs, subRows) {
        if (!fs) {
            return fs;
        }
        var out = Object.assign({}, fs);
        var we = computeWeekendMetrics(subRows);
        if (we) {
            out.weekend_total_hprd = we.weekend_total_hprd;
            out.weekend_rn_hprd = we.weekend_rn_hprd;
            out.weekday_weekend_gap_pct = we.weekday_weekend_gap_pct;
        }
        if (out.total_nurse_hprd > 0) {
            if (out.rn_hprd != null) {
                out.rn_share_pct = (out.rn_hprd / out.total_nurse_hprd) * 100;
            }
            if (out.nurse_aide_hprd != null) {
                out.aide_share_pct = (out.nurse_aide_hprd / out.total_nurse_hprd) * 100;
            }
        }
        if (out.contract_hprd == null && subRows && subRows.length) {
            var rd = 0,
                ch = 0;
            subRows.forEach(function (r) {
                var cen = parseFloat(r.MDScensus);
                if (!(cen > 0)) {
                    return;
                }
                rd += cen;
                ch +=
                    (parseFloat(r.Hrs_RNDON_ctr) || 0) +
                    (parseFloat(r.Hrs_RNadmin_ctr) || 0) +
                    (parseFloat(r.Hrs_RN_ctr) || 0) +
                    (parseFloat(r.Hrs_LPNadmin_ctr) || 0) +
                    (parseFloat(r.Hrs_LPN_ctr) || 0) +
                    (parseFloat(r.Hrs_CNA_ctr) || 0) +
                    (parseFloat(r.Hrs_NAtrn_ctr) || 0) +
                    (parseFloat(r.Hrs_MedAide_ctr) || 0);
            });
            if (rd > 0) {
                out.contract_hprd = ch / rd;
            }
        }
        return out;
    }

    function acuityValues(cmg, key, fs) {
        if (!cmg || !cmg.available) {
            return { facility: null, state: null, region: null, national: null, county: null };
        }
        if (key === 'case_mix_direct_pct') {
            var hp = cmg.hprd || {};
            function pctBlk(blk, rep) {
                if (!blk || !blk.available || !blk.case_mix_direct_hprd || rep == null) {
                    return null;
                }
                return (Number(rep) / Number(blk.case_mix_direct_hprd)) * 100;
            }
            return {
                facility: pctBlk(hp.facility, fs ? fs.direct_care_hprd : null),
                county: pctBlk(hp.county, cmg._countyReported),
                state: pctBlk(hp.state, cmg._stateReported),
                region: pctBlk(hp.cms_region, cmg._regionReported),
                national: pctBlk(hp.national, cmg._nationalReported),
            };
        }
        if (key === 'harrington_total_pct') {
            var har = global.__pbjHarringtonGeoPct;
            if (har) {
                return har;
            }
            return { facility: null, state: null, region: null, national: null, county: null };
        }
        if (key === 'harrington_gap_hprd') {
            var gap = global.__pbjHarringtonGeoGap;
            if (gap) {
                return gap;
            }
            return { facility: null, state: null, region: null, national: null, county: null };
        }
        return { facility: null, state: null, region: null, national: null, county: null };
    }

    function peerCount(lite, prefix) {
        var v = geoLiteMetric(lite, prefix, 'facility_count');
        if (v == null || v === '' || isNaN(v)) {
            return null;
        }
        return Math.round(Number(v));
    }

    function ordinal(n) {
        n = Math.round(Number(n));
        if (!isFinite(n)) {
            return '';
        }
        var mod100 = n % 100;
        if (mod100 >= 11 && mod100 <= 13) {
            return n + 'th';
        }
        var mod10 = n % 10;
        if (mod10 === 1) {
            return n + 'st';
        }
        if (mod10 === 2) {
            return n + 'nd';
        }
        if (mod10 === 3) {
            return n + 'rd';
        }
        return n + 'th';
    }

    function populateGeoPeriodValueSelect(mount, selectedScope, ctx) {
        var valEl = mount.querySelector('#geoRollupPeriodValue');
        if (!valEl) {
            return;
        }
        ctx = ctx || {};
        var qOpts = ctx.geoPeriodQuarterOpts || [];
        var yOpts = ctx.geoPeriodYearOpts || [];
        var html = '';
        qOpts.forEach(function (o) {
            html +=
                '<option value="' +
                esc(o.value) +
                '"' +
                (selectedScope === o.value ? ' selected' : '') +
                '>' +
                esc(o.label) +
                '</option>';
        });
        if (yOpts.length) {
            html += '<optgroup label="Calendar year">';
            yOpts.forEach(function (o) {
                html +=
                    '<option value="' +
                    esc(o.value) +
                    '"' +
                    (selectedScope === o.value ? ' selected' : '') +
                    '>' +
                    esc(o.label) +
                    '</option>';
            });
            html += '</optgroup>';
        }
        valEl.innerHTML = html;
        if (!valEl.value && qOpts.length) {
            valEl.value = qOpts[0].value;
        }
    }

    function readGeoPeriodScope(mount) {
        var valEl = mount.querySelector('#geoRollupPeriodValue');
        var val = valEl ? valEl.value : 'latest';
        if (val.indexOf('year:') === 0 || val.indexOf('q:') === 0 || val === 'latest') {
            return val;
        }
        return 'latest';
    }

    function bindPeriodSelects(mount, cfg) {
        if (!mount) {
            return;
        }
        cfg = cfg || {};
        var valEl = mount.querySelector('#geoRollupPeriodValue');
        if (!valEl) {
            return;
        }
        populateGeoPeriodValueSelect(mount, cfg.scopeVal || 'latest', cfg);
        function persistAndRefresh() {
            var scope = readGeoPeriodScope(mount);
            try {
                localStorage.setItem(cfg.GEO_SCOPE_LS, scope);
            } catch (e) {}
            if (typeof cfg.refresh === 'function') {
                cfg.refresh();
            }
        }
        valEl.onchange = persistAndRefresh;
    }

    function distGeoKey(scope) {
        return scope === 'urban' ? 'state' : scope;
    }

    function peerDistGeoForScope(primaryScope) {
        var geo = distGeoKey(primaryScope);
        if (
            global.PbjV2GeoDistribution &&
            typeof global.PbjV2GeoDistribution.normalizeGeographyType === 'function'
        ) {
            geo = global.PbjV2GeoDistribution.normalizeGeographyType(geo);
        } else if (!geo || !/^(national|state|region|county|city)$/i.test(String(geo))) {
            geo = null;
        }
        return geo;
    }

    function setPeerPercentileCellsLoading(mount, loading) {
        if (!mount) {
            return;
        }
        mount.querySelectorAll('.geo-peer-pct-cell').forEach(function (td) {
            td.classList.remove('geo-peer-pct-cell--ready', 'geo-peer-pct-cell--empty');
            if (loading) {
                td.classList.add('geo-peer-pct-cell--loading');
                td.textContent = '…';
                td.removeAttribute('title');
            } else {
                td.classList.remove('geo-peer-pct-cell--loading');
            }
        });
    }

    function geoDistPeerCellInner(geo, val, metricKey, kind) {
        if (val == null || isNaN(val) || !metricKey) {
            return fmtVal(val, kind);
        }
        var g = distGeoKey(geo);
        if (
            global.PbjV2GeoDistribution &&
            typeof global.PbjV2GeoDistribution.wrapDistCellHtml === 'function'
        ) {
            return global.PbjV2GeoDistribution.wrapDistCellHtml(g, val, metricKey);
        }
        return fmtVal(val, kind);
    }

    function geoDistPeerTd(geo, val, metricKey, kind, extraClass) {
        if (val == null || isNaN(val) || !metricKey) {
            return (
                '<td class="text-end font-monospace pe-2 py-1' +
                (extraClass ? ' ' + extraClass : '') +
                '">' +
                (val != null && !isNaN(val) ? fmtVal(val, kind) : '—') +
                '</td>'
            );
        }
        var g = distGeoKey(geo);
        return (
            '<td class="text-end pe-2 py-1 geo-rollup-dist-cell' +
            (extraClass ? ' ' + extraClass : '') +
            '" data-geo-geography="' +
            esc(g) +
            '" data-geo-metric="' +
            esc(metricKey) +
            '" role="button" tabindex="0" title="View ' +
            esc(g) +
            ' peer distribution">' +
            geoDistPeerCellInner(g, val, metricKey, kind) +
            '</td>'
        );
    }

    function peerBarAggNote(geoKey, model) {
        var yearMode = model.ctx && model.ctx.yearForAgg;
        if (yearMode) {
            return 'Aggregation: mean of bundled quarterly CMS rollups';
        }
        return 'Aggregation: CMS published geography mean for this quarter';
    }

    function resolveDefaultPeerScope(counts, facilityLocale, useLocale) {
        var countyN = counts.countyRaw != null ? counts.countyRaw : counts.county;
        var localeTag =
            useLocale && facilityLocale !== 'unknown'
                ? ' with ' + (facilityLocale === 'rural' ? 'rural' : 'urban') + ' peer filter'
                : '';
        if (countyN != null && countyN >= THRESHOLDS.county) {
            return { scope: 'county', reason: '' };
        }
        var urbanN = counts.urban;
        if (
            (facilityLocale === 'urban' || facilityLocale === 'rural') &&
            urbanN != null &&
            urbanN >= THRESHOLDS.urban
        ) {
            var locLabel = facilityLocale === 'rural' ? 'rural' : 'urban';
            return {
                scope: 'urban',
                reason:
                    'Using ' +
                    locLabel +
                    ' state peers — county n' +
                    (countyN != null ? '=' + fmtCount(countyN) : '') +
                    localeTag +
                    ' is below ' +
                    THRESHOLDS.county +
                    '.',
            };
        }
        var stateN = counts.state;
        if (stateN != null && stateN >= THRESHOLDS.state) {
            return {
                scope: 'state',
                reason:
                    countyN != null && countyN < THRESHOLDS.county
                        ? 'Using state peers — county n=' +
                          fmtCount(countyN) +
                          localeTag +
                          ' is below ' +
                          THRESHOLDS.county +
                          '.'
                        : '',
            };
        }
        return {
            scope: 'region',
            reason:
                'Using CMS region — county and state peer counts' +
                localeTag +
                ' are below thresholds.',
        };
    }

    function buildModel(ctx) {
        var lite = ctx.lite;
        var fs = extendFacilityStats(ctx.fs, ctx.subRows);
        var peerPx = ctx.peerPx;
        var useLocale = ctx.useLocalePeers;
        var locale = ctx.facilityLocale || 'unknown';

        function gv(suffix) {
            return geoPeerMetric(lite, peerPx.state, suffix, 'state');
        }
        function gvc(suffix) {
            return geoPeerMetric(lite, peerPx.county, suffix, 'county');
        }
        function gvr(suffix) {
            if (ctx.gvr) {
                return ctx.gvr(suffix);
            }
            return geoPeerMetric(lite, peerPx.region, suffix, 'region');
        }
        function gvu(suffix) {
            if (locale === 'rural') {
                return geoPeerMetric(lite, 'state_rural', suffix, 'state');
            }
            if (locale === 'urban') {
                return geoPeerMetric(lite, 'state_urban', suffix, 'state');
            }
            return null;
        }

        var countyNRaw = peerCount(lite, 'county');
        var countyN =
            useLocale && peerPx.county === 'county_locale'
                ? peerCount(lite, 'county_locale')
                : peerCount(lite, peerPx.county) || countyNRaw;
        var stateN = peerCount(lite, peerPx.state);
        var regionN = peerCount(lite, peerPx.region);
        var nationalN = numOrNull(lite.national_facility_count);
        var urbanN =
            locale === 'rural'
                ? peerCount(lite, 'state_rural')
                : locale === 'urban'
                  ? peerCount(lite, 'state_urban')
                  : null;

        var countyRawTotal = geoPeerMetric(lite, 'county', 'total_nurse_hprd', null);
        var countyHasData =
            countyRawTotal != null || gvc('total_nurse_hprd') != null || ctx.showCounty;
        var counts = {
            county: countyN,
            countyRaw: countyNRaw,
            state: stateN,
            region: regionN,
            national: nationalN,
            urban: urbanN,
        };
        var def = resolveDefaultPeerScope(counts, locale, useLocale);
        var storedScope = lsGet(LS_PEER_SCOPE, '');
        var validScopes = ['county', 'urban', 'state', 'region', 'national'];
        var preferLocalScopes = ['county', 'state'];
        if (useLocale && (locale === 'urban' || locale === 'rural')) {
            preferLocalScopes.push('urban');
        }
        var usingAutoScope = !storedScope || validScopes.indexOf(storedScope) < 0;
        if (
            !usingAutoScope &&
            (storedScope === 'region' || storedScope === 'national') &&
            preferLocalScopes.indexOf(def.scope) >= 0
        ) {
            usingAutoScope = true;
        }
        var primaryScope = usingAutoScope ? def.scope : storedScope;
        if (
            primaryScope === 'county' &&
            (countyNRaw == null || countyNRaw < 1) &&
            countyRawTotal == null &&
            !countyHasData
        ) {
            primaryScope = def.scope;
            usingAutoScope = true;
        }

        var facilityFull = ctx.facNameFull || ctx.facColTitle || ctx.facName || 'This facility';
        var facilityShort = ctx.facNameShort || ctx.facName || shortenFacilityName(facilityFull);
        var geoLabels = {
            facility: facilityShort,
            facilityShort: facilityShort,
            facilityFull: facilityFull,
            county: ctx.countyLabel || 'County',
            urban:
                locale === 'rural'
                    ? 'Rural peers (state)'
                    : locale === 'urban'
                      ? 'Urban peers (state)'
                      : 'Locale peers',
            state: ctx.stLong || 'State',
            region: ctx.regLabel || 'CMS region',
            national: 'National',
        };

        var metrics = [];
        METRIC_DEFS.forEach(function (def) {
            var m = {
                key: def.key,
                label: def.label,
                group: def.group,
                kind: def.kind || 'hprd',
                distKey: def.distKey,
                liteSuffix: def.liteSuffix || null,
                title: def.title || '',
                facilityOnly: !!def.facilityOnly,
                values: {
                    facility: null,
                    county: null,
                    urban: null,
                    state: null,
                    region: null,
                    national: null,
                },
            };
            if (def.acuityKey) {
                if (def.acuityKey === 'case_mix_direct_pct' && ctx.cmg) {
                    ctx.cmg._countyReported = geoPeerMetric(lite, 'county', 'nurse_care_hprd', null);
                    ctx.cmg._stateReported = gv('nurse_care_hprd');
                    ctx.cmg._regionReported = ctx.showReg ? gvr('nurse_care_hprd') : null;
                    ctx.cmg._nationalReported = numOrNull(lite.national_nurse_care_hprd);
                }
                var av = acuityValues(ctx.cmg, def.acuityKey, fs);
                m.values = Object.assign(m.values, av);
            } else if (def.computedShare === 'rn') {
                m.values.facility = fs ? numOrNull(fs.rn_share_pct) : null;
            } else if (def.computedShare === 'aide') {
                m.values.facility = fs ? numOrNull(fs.aide_share_pct) : null;
            } else if (def.fsKey && fs) {
                m.values.facility = numOrNull(fs[def.fsKey]);
            } else if (def.key === 'headcount_per_100') {
                m.values.facility = ctx.headcountPer100 != null ? ctx.headcountPer100 : null;
                if (m.values.facility == null) {
                    return;
                }
            }
            if (def.liteSuffix && !def.facilityOnly) {
                m.values.county = geoPeerMetric(lite, 'county', def.liteSuffix, null);
                m.values.state = gv(def.liteSuffix);
                m.values.region = ctx.showReg ? gvr(def.liteSuffix) : null;
                m.values.national = numOrNull(lite['national_' + def.liteSuffix]);
                if (useLocale && (locale === 'urban' || locale === 'rural')) {
                    m.values.urban = gvu(def.liteSuffix);
                }
            }
            if (m.facilityOnly) {
                if (m.values.facility != null) {
                    metrics.push(m);
                }
            } else if (
                m.values.facility != null ||
                m.values.state != null ||
                m.values.county != null ||
                m.values.region != null ||
                m.values.national != null
            ) {
                if (metricHasRegionalPeerValue(m, null)) {
                    metrics.push(m);
                }
            }
        });

        return {
            ctx: ctx,
            fs: fs,
            lite: lite,
            counts: counts,
            defaultScope: def.scope,
            fallbackReason: usingAutoScope ? def.reason : '',
            usingAutoScope: usingAutoScope,
            primaryScope: primaryScope,
            geoLabels: geoLabels,
            metrics: metrics,
            useLocalePeers: useLocale,
            facilityLocale: locale,
            showCounty: ctx.showCounty,
            showReg: ctx.showReg,
            countyAvailable:
                ctx.showCounty !== false &&
                (countyRawTotal != null || (countyNRaw != null && countyNRaw > 0)),
            countySuppressed:
                ctx.showCounty && countyNRaw != null && countyNRaw > 0 && countyNRaw < THRESHOLDS.county,
        };
    }

    function countyUnavailableNote(n, show) {
        if (!show) {
            return '';
        }
        if (n == null || n === 0) {
            return 'County comparison unavailable — insufficient comparable facilities in this county.';
        }
        if (n < THRESHOLDS.county) {
            return (
                'County comparison suppressed — peer count (' +
                fmtCount(n) +
                ') is below minimum threshold (' +
                THRESHOLDS.county +
                ').'
            );
        }
        return '';
    }

    function formatPeerPeriodLabel(scopeVal, ctx) {
        ctx = ctx || {};
        var v = String(scopeVal || 'latest').trim();
        if (v === 'latest' || !v) {
            var refCy = ctx.defaultCy || ctx.cy || '';
            if (refCy) {
                if (typeof global.pbjV2FormatCyQuarter === 'function') {
                    return global.pbjV2FormatCyQuarter(refCy);
                }
                var mLatest = String(refCy).match(/^(?:CY)?(\d{4})Q([1-4])$/i);
                if (mLatest) {
                    return 'Q' + mLatest[2] + ' ' + mLatest[1];
                }
                return String(refCy);
            }
            return 'selected quarter';
        }
        if (v.indexOf('year:') === 0) {
            return v.slice(5);
        }
        if (v.indexOf('q:') === 0) {
            var raw = v.slice(2);
            if (typeof global.pbjV2FormatCyQuarter === 'function') {
                return global.pbjV2FormatCyQuarter(raw);
            }
            var m = raw.match(/^(?:CY)?(\d{4})Q([1-4])$/i);
            if (m) {
                return 'Q' + m[2] + ' ' + m[1];
            }
            return raw;
        }
        return v;
    }

    function metricHasRegionalPeerValue(m, primaryScope) {
        if (!m || m.facilityOnly) {
            return true;
        }
        var scopes = ['county', 'urban', 'state', 'region', 'national'];
        if (primaryScope && scopes.indexOf(primaryScope) >= 0) {
            var pv = m.values[primaryScope];
            return pv != null && !isNaN(pv);
        }
        return scopes.some(function (gk) {
            var val = m.values[gk];
            return val != null && !isNaN(val);
        });
    }

    function insightLine(facVal, peerVal, peerLabel, kind, facName, pctData, metric, periodLabel) {
        var metricLabel = metricLabelLower(metric);
        var periodBit = periodLabel ? ' in <strong>' + esc(periodLabel) + '</strong>' : '';
        var facDisplay = esc(facName || 'This facility');
        if (facVal == null || isNaN(facVal)) {
            return '<span class="text-muted">No facility value for this metric in the selected period.</span>';
        }
        var base =
            '<strong>' +
            facDisplay +
            '</strong> had <strong>' +
            fmtVal(facVal, kind) +
            '</strong> ' +
            esc(metricLabel) +
            periodBit;
        if (
            pctData &&
            pctData.percentile != null &&
            metric &&
            metric.distKey &&
            pctData.n != null
        ) {
            var pctTip =
                'Percentile rank among ' +
                fmtCount(pctData.n) +
                ' ' +
                peerLabel +
                ' facilities with eligible PBJ days. Distribution uses individual facility values; peer bars show CMS published geography averages.';
            var pctPart =
                ' — <strong title="' +
                esc(pctTip) +
                '">' +
                ordinal(pctData.percentile) +
                ' percentile</strong> among ' +
                fmtCount(pctData.n) +
                ' ' +
                esc(peerLabel) +
                ' facilities.';
            var benchPart =
                peerVal != null && !isNaN(peerVal)
                    ? ' ' + esc(peerLabel) + ' comparison value: <strong>' + fmtVal(peerVal, kind) + '</strong>.'
                    : '';
            return base + pctPart + benchPart;
        }
        if (peerVal == null || isNaN(peerVal)) {
            return base + '. ' + esc(peerLabel) + ' comparison value unavailable.';
        }
        return base + '. ' + esc(peerLabel) + ' comparison value: <strong>' + fmtVal(peerVal, kind) + '</strong>.';
    }

    function chartTickText(v, kind) {
        if (v == null || isNaN(v)) {
            return '';
        }
        var n = Number(v);
        if (kind === 'pct') {
            return n.toFixed(1);
        }
        if (kind === 'census') {
            return n >= 1000 ? Math.round(n).toLocaleString('en-US') : n.toFixed(1);
        }
        return n.toFixed(2);
    }

    function benchmarkXaxis(values, kind, label) {
        var nums = (values || []).filter(function (x) {
            return x != null && isFinite(x);
        });
        if (!nums.length) {
            return { automargin: true };
        }
        var lo = Math.min.apply(null, nums);
        var hi = Math.max.apply(null, nums);
        var span = Math.max(hi - lo, kind === 'pct' ? 1 : 0.08);
        var pad = Math.max(span * 0.14, kind === 'pct' ? 1.5 : 0.12);
        return {
            title: { text: label || '', font: { size: 11 } },
            automargin: true,
            range: [Math.max(0, lo - pad), hi + pad * 1.25],
            tickformat: kind === 'pct' ? '.1f' : kind === 'census' && hi >= 500 ? '~s' : '.2f',
            separatethousands: false,
            nticks: 5,
            fixedrange: true,
        };
    }

    function peerSampleN(model, geoKey) {
        if (geoKey === 'county') {
            return model.counts.countyRaw != null ? model.counts.countyRaw : model.counts.county;
        }
        return model.counts[geoKey];
    }

    function renderBenchmarkChart(host, model, metricKey, primaryScope, pctData) {
        if (!host || !global.Plotly) {
            if (host) {
                host.innerHTML = '<p class="small text-muted mb-0">Chart library unavailable.</p>';
            }
            return;
        }
        var metric = model.metrics.filter(function (m) {
            return m.key === metricKey;
        })[0];
        if (!metric) {
            host.innerHTML = '<p class="small text-muted mb-0">Select a metric.</p>';
            return;
        }
        var peerOrder = [];
        if (model.showCounty && metric.values.county != null && !isNaN(metric.values.county)) {
            peerOrder.push('county');
        }
        if (model.useLocalePeers && (model.facilityLocale === 'urban' || model.facilityLocale === 'rural')) {
            peerOrder.push('urban');
        }
        peerOrder.push('state');
        if (model.showReg) {
            peerOrder.push('region');
        }
        peerOrder.push('national');

        var peerLabels = [];
        var peerBars = [];
        var peerColors = [];
        var peerLineColors = [];
        var peerLineWidths = [];
        var peerOpacities = [];
        var peerTextColors = [];
        var peerGeo = [];
        var peerHoverData = [];
        peerOrder.forEach(function (gk) {
            var v = metric.values[gk];
            if (v == null || isNaN(v)) {
                return;
            }
            var lbl = model.geoLabels[gk] || gk;
            var n = peerSampleN(model, gk);
            var isBench = gk === primaryScope;
            peerLabels.push(lbl);
            peerBars.push(Number(v));
            peerColors.push(isBench ? '#64748b' : '#e8edf2');
            peerLineColors.push(isBench ? '#475569' : '#cbd5e1');
            peerLineWidths.push(isBench ? 2 : 1);
            peerOpacities.push(isBench ? 0.92 : 0.78);
            peerTextColors.push(isBench ? '#334155' : '#94a3b8');
            peerGeo.push(gk);
            peerHoverData.push([
                lbl,
                fmtCount(n),
                peerBarAggNote(gk, model),
                isBench ? 'Selected benchmark group' : 'Click bar for distribution',
            ]);
        });

        var facVal = metric.values.facility;
        var facLabel = facilityInsightName(model);
        var facFull =
            (model.geoLabels && model.geoLabels.facilityFull) ||
            (model.geoLabels && model.geoLabels.facility) ||
            facLabel;
        if (facVal == null || isNaN(facVal)) {
            host.innerHTML =
                '<p class="small text-muted mb-0">No facility value for this metric in the selected period.</p>';
            return;
        }
        if (!peerLabels.length) {
            host.innerHTML =
                '<p class="small text-muted mb-0">No peer benchmark values for this metric in the selected period.</p>';
            return;
        }

        /* Top-to-bottom: facility, county, state, region, national — Plotly categoryarray is bottom-first. */
        var peerKeysTopToBottom = ['facility'];
        peerOrder.forEach(function (gk) {
            var v = metric.values[gk];
            if (v != null && !isNaN(v)) {
                peerKeysTopToBottom.push(gk);
            }
        });
        var yCatsBottomToTop = peerKeysTopToBottom
            .slice()
            .reverse()
            .map(function (gk) {
                if (gk === 'facility') {
                    return facLabel;
                }
                return model.geoLabels[gk] || gk;
            });
        var peerLabelsBottomToTop = peerLabels.slice().reverse();
        var peerBarsBottomToTop = peerBars.slice().reverse();
        var peerColorsBottomToTop = peerColors.slice().reverse();
        var peerLineColorsBottomToTop = peerLineColors.slice().reverse();
        var peerLineWidthsBottomToTop = peerLineWidths.slice().reverse();
        var peerOpacitiesBottomToTop = peerOpacities.slice().reverse();
        var peerTextColorsBottomToTop = peerTextColors.slice().reverse();
        var peerHoverBottomToTop = peerHoverData.slice().reverse();
        var peerGeoBottomToTop = peerGeo.slice().reverse();

        var allVals = peerBars.concat([Number(facVal)]);

        var facHover =
            '<b>' +
            facLabel +
            '</b><br>%{x:.3f}' +
            (pctData && pctData.percentile != null
                ? '<br>' + ordinal(pctData.percentile) + ' percentile (n=' + fmtCount(pctData.n) + ')'
                : '') +
            (facFull && facFull !== facLabel ? '<br><i>' + esc(facFull) + '</i>' : '') +
            '<extra>' +
            esc(facFull) +
            '</extra>';
        var peerHoverTemplate =
            '<b>%{customdata[0]}</b><br>%{x:.3f}<br>' +
            (metric.kind === 'pct' ? '' : '') +
            '%{customdata[2]}<br>' +
            'Sample: n=%{customdata[1]} facilities<br>' +
            '<i>%{customdata[3]}</i><extra></extra>';
        var peerBarLabels = peerBarsBottomToTop.map(function (v) {
            return chartTickText(v, metric.kind);
        });

        var barCount = peerLabelsBottomToTop.length + 1;
        var chartHeight = Math.max(96, Math.min(210, barCount * 32 + 20));

        global.Plotly.newPlot(
            host,
            [
                {
                    type: 'bar',
                    orientation: 'h',
                    y: peerLabelsBottomToTop,
                    x: peerBarsBottomToTop,
                    marker: {
                        color: peerColorsBottomToTop,
                        opacity: peerOpacitiesBottomToTop,
                        line: { color: peerLineColorsBottomToTop, width: peerLineWidthsBottomToTop },
                    },
                    text: peerBarLabels,
                    textposition: 'outside',
                    textfont: {
                        size: 11,
                        color: peerTextColorsBottomToTop,
                        family: 'system-ui, sans-serif',
                    },
                    hovertemplate: peerHoverTemplate,
                    customdata: peerHoverBottomToTop,
                    cliponaxis: false,
                },
                {
                    type: 'bar',
                    orientation: 'h',
                    y: [facLabel],
                    x: [Number(facVal)],
                    marker: {
                        color: '#0d6efd',
                        line: { color: '#084298', width: 2.5 },
                    },
                    text: [chartTickText(facVal, metric.kind)],
                    textposition: 'outside',
                    textfont: { size: 12, color: '#084298', family: 'system-ui, sans-serif' },
                    hovertemplate: facHover,
                    customdata: ['facility'],
                    cliponaxis: false,
                },
            ],
            {
                height: chartHeight,
                margin: { t: 2, r: 52, b: 24, l: 4 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(248,250,252,0.9)',
                barmode: 'overlay',
                xaxis: benchmarkXaxis(allVals, metric.kind, metric.label),
                yaxis: {
                    automargin: true,
                    categoryorder: 'array',
                    categoryarray: yCatsBottomToTop,
                    tickfont: { size: 10 },
                },
                showlegend: false,
            },
            { responsive: true, displayModeBar: false }
        );
        host.style.height = chartHeight + 'px';
        host.style.minHeight = '0';
        updateGeoPeerChartLegend(host.closest('#geoPeerComparisonPanel'), model, primaryScope);
        host.on('plotly_click', function (ev) {
            if (!ev || !ev.points || !ev.points[0]) {
                return;
            }
            var pt = ev.points[0];
            var geo =
                pt.curveNumber === 0
                    ? peerGeoBottomToTop[pt.pointNumber]
                    : pt.curveNumber === 1
                      ? 'facility'
                      : null;
            if (geo && geo !== 'facility' && metric.distKey && global.PbjV2GeoDistribution) {
                global.PbjV2GeoDistribution.openDistributionModal(
                    metric.distKey,
                    geo === 'urban' ? 'state' : geo
                );
            }
        });
    }

    function csvEscapeCell(text) {
        var t = String(text == null ? '' : text).replace(/\s+/g, ' ').trim();
        if (/[",\n]/.test(t)) {
            return '"' + t.replace(/"/g, '""') + '"';
        }
        return t;
    }

    function tableElementToCsv(table) {
        if (!table) {
            return '';
        }
        var rows = [];
        table.querySelectorAll('tr').forEach(function (tr) {
            var cells = [];
            tr.querySelectorAll('th, td').forEach(function (cell) {
                cells.push(csvEscapeCell(cell.textContent));
            });
            if (cells.length) {
                rows.push(cells.join(','));
            }
        });
        return rows.join('\n');
    }

    function downloadPeerComparisonCsv(mount, model) {
        if (!mount) {
            return;
        }
        var parts = [];
        var sideTable = mount.querySelector('#geoPeerBenchmarkTableMount table');
        if (sideTable) {
            parts.push('Benchmark comparison');
            parts.push(tableElementToCsv(sideTable));
        }
        var fullTable = mount.querySelector('#geoPeerAllGeographiesCollapse table');
        if (fullTable) {
            if (parts.length) {
                parts.push('');
            }
            parts.push('Regional geography comparison');
            parts.push(tableElementToCsv(fullTable));
        }
        if (!parts.length) {
            return;
        }
        var ccn =
            (global.PBJ320_EXPORT_CCN && String(global.PBJ320_EXPORT_CCN).replace(/\D/g, '')) ||
            (global.PROVNUM && String(global.PROVNUM).replace(/\D/g, '')) ||
            'facility';
        var blob = new Blob([parts.join('\n')], { type: 'text/csv;charset=utf-8' });
        var url = URL.createObjectURL(blob);
        var a = document.createElement('a');
        a.href = url;
        a.download = 'pbj_peer_regional_' + ccn + '.csv';
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    }

    function openDistributionForSelection(model, metricKey, primaryScope) {
        var m = model.metrics.filter(function (x) {
            return x.key === metricKey;
        })[0];
        if (!m || !m.distKey || !global.PbjV2GeoDistribution) {
            return;
        }
        var geo = primaryScope === 'urban' ? 'state' : primaryScope;
        if (geo === 'facility') {
            geo = 'county';
        }
        global.PbjV2GeoDistribution.openDistributionModal(m.distKey, geo);
    }

    function renderTable(model, primaryScope, compact, periodLabel) {
        var peerLabel = model.geoLabels[primaryScope] || primaryScope;
        var facHead = facilityTableHeaderMeta(model);
        var countyNote = countyUnavailableNote(model.counts.county, model.showCounty && !model.countyAvailable);
        var peerCountKey = primaryScope === 'urban' ? 'urban' : primaryScope;
        var peerN = model.counts[peerCountKey];
        var periodNote =
            periodLabel && String(periodLabel).trim()
                ? ' · ' + esc(String(periodLabel).trim())
                : '';
        var caption = compact
            ? '<caption class="small text-muted">Facility vs <strong>' +
              esc(peerLabel) +
              '</strong>' +
              (peerN != null ? ' (n=' + fmtCount(peerN) + ')' : '') +
              periodNote +
              '. Diff = facility − CMS geography mean; percentile within peer group.</caption>'
            : '';
        var head =
            '<thead class="geo-rollup-thead"><tr>' +
            '<th scope="col" class="ps-2 py-1 geo-rollup-metric-col">Metric</th>' +
            '<th scope="col" class="text-end pe-2 py-1"' +
            (facHead.title ? ' title="' + esc(facHead.title) + '"' : '') +
            '>' +
            esc(facHead.label) +
            '</th>' +
            '<th scope="col" class="text-end pe-2 py-1">' +
            esc(benchmarkTableColumnLabel(peerLabel, compact)) +
            (!compact && peerN != null
                ? ' <span class="text-muted fw-normal">(n=' + fmtCount(peerN) + ')</span>'
                : '') +
            '</th>' +
            '<th scope="col" class="text-end pe-2 py-1">' +
            (compact ? 'Diff' : 'Difference') +
            '</th>' +
            '<th scope="col" class="text-end pe-2 py-1' +
            (compact ? '' : ' d-none d-md-table-cell') +
            '">Pct</th>' +
            '</tr></thead>';
        var body = sortPeerCompMetrics(
            model.metrics.filter(function (m) {
                return metricHasRegionalPeerValue(m, primaryScope);
            })
        )
            .map(function (m) {
                var fv = m.values.facility;
                var pv = m.values[primaryScope];
                var diff = fv != null && pv != null && !isNaN(fv) && !isNaN(pv) ? Number(fv) - Number(pv) : null;
                var diffHtml = diffCellHtml(diff, m.kind);
                var pctGeo = peerDistGeoForScope(primaryScope) || distGeoKey(primaryScope);
                var pctCell =
                    '<td class="text-end font-monospace pe-2 py-1' +
                    (compact ? '' : ' d-none d-md-table-cell') +
                    ' geo-peer-pct-cell geo-peer-pct-cell--loading" data-metric="' +
                    esc(m.key) +
                    '" data-geo="' +
                    esc(pctGeo) +
                    '" data-geo-scope="' +
                    esc(primaryScope) +
                    '">…</td>';
                var peerTd =
                    pv != null && m.distKey
                        ? geoDistPeerTd(primaryScope, pv, m.distKey, m.kind)
                        : '<td class="text-end font-monospace pe-2 py-1">' +
                          (pv != null ? fmtVal(pv, m.kind) : '—') +
                          '</td>';
                return (
                    '<tr data-geo-metric="' +
                    esc(m.key) +
                    '"><th scope="row" class="geo-rollup-th geo-rollup-metric-label ps-2 py-1"' +
                    (m.title ? ' title="' + esc(m.title) + '"' : '') +
                    '>' +
                    esc(m.label) +
                    '</th><td class="text-end font-monospace pe-2 py-1 geo-rollup-num-cell">' +
                    fmtVal(fv, m.kind) +
                    '</td>' +
                    peerTd +
                    '<td class="text-end font-monospace pe-2 py-1 geo-rollup-num-cell">' +
                    diffHtml +
                    '</td>' +
                    pctCell +
                    '</tr>'
                );
            })
            .join('');
        var fullTable =
            '<div class="table-responsive mb-0" style="-webkit-overflow-scrolling:touch">' +
            '<table class="table table-sm table-bordered table-hover align-middle mb-0 small geo-rollup-table">' +
            caption +
            head +
            '<tbody class="geo-rollup-tbody">' +
            body +
            '</tbody></table></div>';
        if (countyNote) {
            fullTable =
                '<p class="small text-muted mb-1"><i class="fas fa-info-circle me-1" aria-hidden="true"></i>' +
                esc(countyNote) +
                '</p>' +
                fullTable;
        }
        return fullTable;
    }

    function renderGeoRollupTable(model, ctx) {
        var lite = model.lite;
        var fs = model.fs;
        var peerPx = ctx.peerPx;
        function gv(s) {
            return geoPeerMetric(lite, peerPx.state, s, 'state');
        }
        function gvc(s) {
            return geoPeerMetric(lite, peerPx.county, s, 'county');
        }
        function gvr(s) {
            return ctx.gvr(s);
        }
        function cell(inner) {
            return '<td class="text-end font-monospace pe-2 py-1 geo-rollup-num-cell">' + inner + '</td>';
        }
        function countyCell(val, n, metricKey, kind) {
            if (!model.showCounty) {
                return '';
            }
            if (val == null || isNaN(val)) {
                var note = countyUnavailableNote(n, true);
                return (
                    '<td class="text-end pe-2 py-1 text-muted" title="' +
                    esc(note) +
                    '"><span class="geo-county-unavail" tabindex="0"><i class="fas fa-info-circle fa-xs me-1" aria-hidden="true"></i>—</span></td>'
                );
            }
            return geoDistPeerTd('county', val, metricKey, kind || 'hprd');
        }
        var rows = sortPeerCompMetrics(
            model.metrics.filter(function (m) {
                if (!m.distKey || !m.liteSuffix) {
                    return false;
                }
                return metricHasRegionalPeerValue(m, null);
            })
        )
            .map(function (m) {
                var suffix = m.liteSuffix;
                var st = gv(suffix);
                var co = gvc(suffix);
                var rg = model.showReg ? gvr(suffix) : null;
                var nat = numOrNull(lite['national_' + suffix]);
                var distKey = m.distKey || m.key;
                return (
                    '<tr data-geo-metric="' +
                    esc(distKey) +
                    '"><th scope="row" class="geo-rollup-th geo-rollup-metric-label ps-2 py-1">' +
                    esc(m.label) +
                    '</th>' +
                    cell(fmtVal(m.values.facility, m.kind)) +
                    countyCell(co, model.counts.county, distKey, m.kind) +
                    geoDistPeerTd('state', st, distKey, m.kind) +
                    (model.showReg ? geoDistPeerTd('region', rg, distKey, m.kind) : '') +
                    geoDistPeerTd('national', nat, distKey, m.kind) +
                    '</tr>'
                );
            })
            .join('');
        var countyHead = model.showCounty
            ? '<th scope="col" class="text-end pe-2 py-1">' +
              esc(model.geoLabels.county) +
              (model.counts.county != null
                  ? ' <span class="text-muted fw-normal">(n=' + fmtCount(model.counts.county) + ')</span>'
                  : '') +
              '</th>'
            : '';
        var facHeadFull = facilityTableHeaderMeta(model);
        return (
            '<div class="table-responsive mb-0"><table class="table table-sm table-bordered table-hover align-middle mb-0 small geo-rollup-table">' +
            '<thead class="geo-rollup-thead"><tr><th scope="col" class="ps-2 py-1 geo-rollup-metric-col">Metric</th><th scope="col" class="text-end pe-2 py-1"' +
            (facHeadFull.title ? ' title="' + esc(facHeadFull.title) + '"' : '') +
            '>' +
            esc(facHeadFull.label) +
            '</th>' +
            countyHead +
            '<th scope="col" class="text-end pe-2 py-1">' +
            esc(model.geoLabels.state) +
            '</th>' +
            (model.showReg ? '<th scope="col" class="text-end pe-2 py-1">' + esc(model.geoLabels.region) + '</th>' : '') +
            '<th scope="col" class="text-end pe-2 py-1">National</th></tr></thead><tbody class="geo-rollup-tbody">' +
            rows +
            '</tbody></table></div>'
        );
    }

    function renderShell(model, ctx) {
        var metricKey = lsGet(LS_METRIC, 'total_nurse_hprd');
        if (
            !model.metrics.some(function (m) {
                return m.key === metricKey;
            })
        ) {
            metricKey = model.metrics.length ? model.metrics[0].key : 'total_nurse_hprd';
        }
        var primaryScope = model.primaryScope;
        var metricOpts = '';
        sortPeerCompMetrics(
            model.metrics.filter(function (m) {
                return PEER_COMP_METRIC_ORDER.indexOf(m.key) >= 0 && metricHasRegionalPeerValue(m, primaryScope);
            })
        ).forEach(function (m) {
            metricOpts +=
                '<option value="' +
                esc(m.key) +
                '"' +
                (m.key === metricKey ? ' selected' : '') +
                '>' +
                esc(m.label) +
                '</option>';
        });
        var advancedMetrics = model.metrics.filter(function (m) {
            return PEER_COMP_METRIC_ORDER.indexOf(m.key) < 0 && metricHasRegionalPeerValue(m, primaryScope);
        });
        if (advancedMetrics.length) {
            metricOpts += '<optgroup label="Advanced">';
            sortPeerCompMetrics(advancedMetrics).forEach(function (m) {
                metricOpts +=
                    '<option value="' +
                    esc(m.key) +
                    '"' +
                    (m.key === metricKey ? ' selected' : '') +
                    '>' +
                    esc(m.label) +
                    '</option>';
            });
            metricOpts += '</optgroup>';
        }
        var scopeOpts = '';
        [
            ['county', model.geoLabels.county, model.counts.county],
            ['urban', model.geoLabels.urban, model.counts.urban],
            ['state', model.geoLabels.state, model.counts.state],
            ['region', model.geoLabels.region, model.counts.region],
            ['national', model.geoLabels.national, model.counts.national],
        ].forEach(function (pair) {
            var k = pair[0],
                lbl = pair[1],
                n = pair[2];
            if (k === 'county' && (!model.showCounty || !model.countyAvailable)) {
                return;
            }
            if (k === 'urban' && (!model.useLocalePeers || model.facilityLocale === 'unknown')) {
                return;
            }
            if (k === 'region' && !model.showReg) {
                return;
            }
            var warn = n != null && n < THRESHOLDS[k === 'urban' ? 'urban' : k === 'county' ? 'county' : k === 'state' ? 'state' : 999];
            var nSuffix = n != null ? ' (n=' + fmtCount(n) + ')' : '';
            if (k === 'county' && model.counts.countyRaw != null && model.counts.countyRaw !== n) {
                nSuffix =
                    ' (n=' + fmtCount(n) + ' urban/rural, ' + fmtCount(model.counts.countyRaw) + ' all)';
            }
            scopeOpts +=
                '<option value="' +
                k +
                '"' +
                (k === primaryScope ? ' selected' : '') +
                '>' +
                esc(lbl) +
                nSuffix +
                (warn ? ' ⚠' : '') +
                '</option>';
        });
        var fallbackBanner = model.fallbackReason
            ? '<p class="small text-muted mb-2 geo-peer-fallback-note py-1 px-2 rounded bg-light border-start border-3 border-warning">' +
              esc(model.fallbackReason) +
              '</p>'
            : '';
        var selectedMetric = model.metrics.filter(function (m) {
            return m.key === metricKey;
        })[0] || model.metrics[0];
        var insight = selectedMetric
            ? insightLine(
                  selectedMetric.values.facility,
                  selectedMetric.values[primaryScope],
                  model.geoLabels[primaryScope],
                  selectedMetric.kind,
                  facilityInsightName(model),
                  null,
                  selectedMetric,
                  formatPeerPeriodLabel(ctx.scopeVal, ctx)
              )
            : '';
        var localeToggle =
            ctx.localeToggleHtml && model.facilityLocale !== 'unknown'
                ? '<div class="geo-peer-locale-wrap align-self-end">' + ctx.localeToggleHtml + '</div>'
                : '';
        return (
            '<div class="geo-rollup-panel mb-0" id="geoPeerComparisonPanel">' +
            '<div class="d-flex flex-wrap align-items-end gap-2 mb-2 geo-peer-toolbar">' +
            (ctx.scopeSelectHtml || '') +
            '<div class="geo-peer-metric-wrap" style="min-width:9rem;flex:1 1 9rem">' +
            '<label for="geoPeerMetricSelect" class="form-label small text-muted mb-0">Metric</label>' +
            '<select id="geoPeerMetricSelect" class="form-select form-select-sm" aria-label="Comparison metric">' +
            metricOpts +
            '</select></div>' +
            '<div class="geo-peer-scope-wrap" style="min-width:9rem;flex:1 1 9rem">' +
            '<label for="geoPeerScopeSelect" class="form-label small text-muted mb-0">Benchmark group</label>' +
            '<select id="geoPeerScopeSelect" class="form-select form-select-sm" aria-label="Primary peer group">' +
            scopeOpts +
            '</select></div>' +
            localeToggle +
            '<div class="d-flex flex-wrap gap-1 ms-md-auto align-self-end geo-peer-toolbar-actions">' +
            '<button type="button" class="btn btn-sm btn-outline-secondary geo-peer-dist-btn" id="geoPeerOpenDistBtn" title="Histogram for selected metric and peer group" aria-label="View distribution"><i class="fas fa-chart-bar" aria-hidden="true"></i><span>View distribution</span></button>' +
            '<button type="button" class="btn btn-sm btn-outline-secondary pbj-export-btn pbj-export-btn--csv" id="geoPeerExportCsvBtn" title="Download regional table as CSV" aria-label="Download regional table as CSV"><i class="fas fa-file-csv" aria-hidden="true"></i><span class="pbj-export-btn-label">Export</span></button>' +
            '<button type="button" class="btn btn-link btn-sm p-0 pbj-summary-info-btn text-muted" id="geoPeerHowToBtn" data-bs-toggle="modal" data-bs-target="#geoRollupNotesModal" title="Peer comparison definitions" aria-label="Peer comparison definitions"><i class="fas fa-circle-info" style="font-size:0.8rem;" aria-hidden="true"></i></button>' +
            '</div></div>' +
            fallbackBanner +
            '<p class="small mb-2 geo-peer-insight" aria-live="polite">' +
            insight +
            '</p>' +
            '<div class="geo-peer-chart-table-row">' +
            '<div class="geo-peer-chart-table-row__chart">' +
            '<div id="geoPeerBenchmarkChart" class="geo-peer-benchmark-chart-host" role="img" aria-label="Benchmark comparison chart"></div>' +
            '<div id="geoPeerChartLegend" class="geo-peer-chart-legend small text-muted d-flex flex-wrap gap-2 gap-md-3 mt-1 mb-0" aria-hidden="false"></div>' +
            '</div>' +
            '<div class="geo-peer-chart-table-row__table">' +
            '<div class="geo-peer-side-table-wrap">' +
            '<div id="geoPeerBenchmarkTableMount">' +
            renderTable(model, primaryScope, true) +
            '</div></div></div></div>' +
            '<div class="geo-peer-full-table-wrap mt-2" id="geoPeerFullRollupWrap">' +
            renderAllGeographiesDisclosure(model, ctx) +
            '</div></div>'
        );
    }

    function peerPercentileRankValues(data) {
        if (!data) {
            return [];
        }
        if (Array.isArray(data.values_ranked) && data.values_ranked.length) {
            return data.values_ranked;
        }
        if (Array.isArray(data.values_dotplot) && data.values_dotplot.length) {
            return data.values_dotplot;
        }
        if (Array.isArray(data.peers) && data.peers.length) {
            return data.peers
                .map(function (p) {
                    return p && (p.metric_value != null ? p.metric_value : p.value);
                })
                .filter(function (v) {
                    return v != null && v !== '' && !isNaN(Number(v));
                })
                .map(function (v) {
                    return Number(v);
                });
        }
        return [];
    }

    function peerPercentileFromPayload(data) {
        if (!data) {
            return null;
        }
        if (data.percentile != null && data.percentile !== '' && !isNaN(Number(data.percentile))) {
            return Number(data.percentile);
        }
        if (data.facility_value == null || data.facility_value === '') {
            return null;
        }
        var fv = Number(data.facility_value);
        if (!Number.isFinite(fv)) {
            return null;
        }
        var vals = peerPercentileRankValues(data);
        if (!vals.length) {
            return null;
        }
        var below = 0;
        var equal = 0;
        vals.forEach(function (v) {
            var n = Number(v);
            if (!Number.isFinite(n)) {
                return;
            }
            if (n < fv) {
                below += 1;
            } else if (n === fv) {
                equal += 1;
            }
        });
        var rank = below + (equal + 1) / 2;
        return Math.round((100 * rank) / vals.length * 10) / 10;
    }

    function applyPercentileToCells(mount, metricKey, geo, primaryScope, data) {
        var selector =
            '.geo-peer-pct-cell[data-metric="' + metricKey + '"]';
        mount.querySelectorAll(selector).forEach(function (td) {
            var cellGeo = td.getAttribute('data-geo') || '';
            var cellScope = td.getAttribute('data-geo-scope') || '';
            if (geo && cellGeo && cellGeo !== geo && cellScope !== primaryScope) {
                return;
            }
            if (!geo && primaryScope && cellScope && cellScope !== primaryScope) {
                return;
            }
            var pct = peerPercentileFromPayload(data);
            td.classList.remove('geo-peer-pct-cell--loading');
            if (pct != null) {
                td.textContent = ordinal(pct);
                td.classList.add('geo-peer-pct-cell--ready');
                td.classList.remove('geo-peer-pct-cell--empty');
                td.title =
                    'Percentile among ' +
                    (data && data.n != null ? fmtCount(data.n) : '?') +
                    ' facilities. Distribution uses individual facility values; peer bars show CMS published means.' +
                    (data && data.small_sample_flag ? ' Small sample — treat as directional only.' : '');
            } else {
                td.textContent = '—';
                td.classList.add('geo-peer-pct-cell--empty');
                td.classList.remove('geo-peer-pct-cell--ready');
                td.removeAttribute('title');
            }
        });
    }

    function peerPercentileFetchUrl(cfg, cy, distKey, geo) {
        return (
            (typeof global.pbjApiUrl === 'function' ? global.pbjApiUrl('/api/geo-distribution') : '/api/geo-distribution') +
            '?provnum=' +
            encodeURIComponent(cfg.exportCcn || '') +
            '&quarter=' +
            encodeURIComponent(cy) +
            '&metric=' +
            encodeURIComponent(distKey) +
            '&geography_type=' +
            encodeURIComponent(geo)
        );
    }

    function fetchPeerPercentilePayload(url) {
        return fetch(url).then(function (r) {
            return r.json();
        });
    }

    function fetchPercentiles(mount, model, metricKey, primaryScope, onDone, requestId) {
        var m = model.metrics.filter(function (x) {
            return x.key === metricKey;
        })[0];
        if (!m || !m.distKey || primaryScope === 'facility') {
            if (typeof onDone === 'function') {
                onDone(null, requestId);
            }
            return;
        }
        var geo = peerDistGeoForScope(primaryScope);
        if (!geo) {
            setPeerPercentileCellsLoading(mount, false);
            if (typeof onDone === 'function') {
                onDone(null, requestId);
            }
            return;
        }
        var cfgEl = document.getElementById('pbj320-export-page');
        var cfg = {};
        try {
            cfg = JSON.parse((cfgEl && cfgEl.textContent) || '{}');
        } catch (e) {}
        var cy = mount.getAttribute('data-geo-cy-quarter');
        if (!cy) {
            setPeerPercentileCellsLoading(mount, false);
            if (typeof onDone === 'function') {
                onDone(null, requestId);
            }
            return;
        }
        setPeerPercentileCellsLoading(mount, true);
        fetchPeerPercentilePayload(peerPercentileFetchUrl(cfg, cy, m.distKey, geo))
            .then(function (data) {
                if (requestId != null && requestId !== global.__pbjPeerPctReqId) {
                    return;
                }
                if (!data || data.error) {
                    if (typeof onDone === 'function') {
                        onDone(null, requestId);
                    }
                    return;
                }
                applyPercentileToCells(mount, metricKey, geo, primaryScope, data);
                if (typeof onDone === 'function') {
                    onDone(data, requestId);
                }
            })
            .catch(function () {
                if (requestId != null && requestId !== global.__pbjPeerPctReqId) {
                    return;
                }
                if (typeof onDone === 'function') {
                    onDone(null, requestId);
                }
            });
    }

    function fetchAllTablePercentiles(mount, model, primaryScope, selectedMetricKey, onDone, requestId) {
        var geo = peerDistGeoForScope(primaryScope);
        var metrics = sortPeerCompMetrics(
            model.metrics.filter(function (m) {
                return m.distKey && metricHasRegionalPeerValue(m, primaryScope);
            })
        );
        if (!geo || !metrics.length) {
            setPeerPercentileCellsLoading(mount, false);
            if (typeof onDone === 'function') {
                onDone(null, requestId);
            }
            return;
        }
        var cfgEl = document.getElementById('pbj320-export-page');
        var cfg = {};
        try {
            cfg = JSON.parse((cfgEl && cfgEl.textContent) || '{}');
        } catch (e) {}
        var cy = mount.getAttribute('data-geo-cy-quarter');
        if (!cy) {
            setPeerPercentileCellsLoading(mount, false);
            if (typeof onDone === 'function') {
                onDone(null, requestId);
            }
            return;
        }
        setPeerPercentileCellsLoading(mount, true);
        Promise.all(
            metrics.map(function (m) {
                return fetchPeerPercentilePayload(peerPercentileFetchUrl(cfg, cy, m.distKey, geo))
                    .then(function (data) {
                        return { metric: m, data: data };
                    })
                    .catch(function () {
                        return { metric: m, data: null };
                    });
            })
        ).then(function (rows) {
            if (requestId != null && requestId !== global.__pbjPeerPctReqId) {
                return;
            }
            var selectedPct = null;
            rows.forEach(function (row) {
                if (!row || !row.metric || !row.data || row.data.error) {
                    return;
                }
                applyPercentileToCells(mount, row.metric.key, geo, primaryScope, row.data);
                if (row.metric.key === selectedMetricKey) {
                    selectedPct = row.data;
                }
            });
            mount.querySelectorAll('.geo-peer-pct-cell.geo-peer-pct-cell--loading').forEach(function (td) {
                td.classList.remove('geo-peer-pct-cell--loading');
                if (!td.classList.contains('geo-peer-pct-cell--ready')) {
                    td.textContent = '—';
                    td.classList.add('geo-peer-pct-cell--empty');
                }
            });
            if (typeof onDone === 'function') {
                onDone(selectedPct, requestId);
            }
        });
    }

    function bind(mount, model, ctx) {
        if (!mount) {
            return;
        }
        global.__pbjPeerComparisonModel = model;
        var metricKey = lsGet(LS_METRIC, 'total_nurse_hprd');
        var primaryScope = model.primaryScope;
        var pctFetchTimer = null;

        function refreshView() {
            var msel = mount.querySelector('#geoPeerMetricSelect');
            var psel = mount.querySelector('#geoPeerScopeSelect');
            if (msel) {
                metricKey = msel.value;
                if (
                    !model.metrics.some(function (m) {
                        return m.key === metricKey;
                    })
                ) {
                    metricKey = model.metrics.length ? model.metrics[0].key : 'total_nurse_hprd';
                    msel.value = metricKey;
                }
                lsSet(LS_METRIC, metricKey);
            }
            if (psel) {
                primaryScope = psel.value;
                lsSet(LS_PEER_SCOPE, primaryScope);
            }
            var metric = model.metrics.filter(function (m) {
                return m.key === metricKey;
            })[0];
            var insightEl = mount.querySelector('.geo-peer-insight');
            var periodLabel = formatPeerPeriodLabel(readGeoPeriodScope(mount) || ctx.scopeVal || 'latest', ctx);
            function applyInsight(pctData, reqId) {
                if (reqId != null && reqId !== global.__pbjPeerPctReqId) {
                    return;
                }
                if (insightEl && metric) {
                    var pctForInsight =
                        metric.distKey && pctData && peerPercentileFromPayload(pctData) != null
                            ? Object.assign({}, pctData, {
                                  percentile: peerPercentileFromPayload(pctData),
                              })
                            : null;
                    insightEl.innerHTML = insightLine(
                        metric.values.facility,
                        metric.values[primaryScope],
                        model.geoLabels[primaryScope],
                        metric.kind,
                        facilityInsightName(model),
                        pctForInsight,
                        metric,
                        periodLabel
                    );
                    insightEl.title =
                        pctForInsight
                            ? 'Percentile uses individual facility values in the selected peer group. Peer bars show CMS published geography means.'
                            : '';
                }
            }
            applyInsight(null, null);
            renderBenchmarkChart(
                mount.querySelector('#geoPeerBenchmarkChart'),
                model,
                metricKey,
                primaryScope,
                null
            );
            var tm = mount.querySelector('#geoPeerBenchmarkTableMount');
            if (tm) {
                tm.innerHTML = renderTable(model, primaryScope, true, periodLabel);
            }
            if (
                global.PbjV2GeoDistribution &&
                typeof global.PbjV2GeoDistribution.enhanceRollupMount === 'function'
            ) {
                global.PbjV2GeoDistribution.enhanceRollupMount();
            }
            if (pctFetchTimer) {
                clearTimeout(pctFetchTimer);
                pctFetchTimer = null;
            }
            global.__pbjPeerPctReqId = (global.__pbjPeerPctReqId || 0) + 1;
            var pctReqId = global.__pbjPeerPctReqId;
            var fetchPctDone = function (pctData, reqId) {
                applyInsight(pctData, reqId);
                if (reqId != null && reqId !== global.__pbjPeerPctReqId) {
                    return;
                }
                renderBenchmarkChart(
                    mount.querySelector('#geoPeerBenchmarkChart'),
                    model,
                    metricKey,
                    primaryScope,
                    metric.distKey && pctData ? pctData : null
                );
            };
            var schedulePctFetch = function () {
                fetchAllTablePercentiles(mount, model, primaryScope, metricKey, fetchPctDone, pctReqId);
            };
            setPeerPercentileCellsLoading(mount, true);
            pctFetchTimer = setTimeout(schedulePctFetch, 80);
        }
        var msel = mount.querySelector('#geoPeerMetricSelect');
        if (msel) {
            msel.addEventListener('change', refreshView);
        }
        var psel = mount.querySelector('#geoPeerScopeSelect');
        if (psel) {
            psel.addEventListener('change', refreshView);
        }
        var distBtn = mount.querySelector('#geoPeerOpenDistBtn');
        if (distBtn) {
            distBtn.addEventListener('click', function () {
                openDistributionForSelection(model, metricKey, primaryScope);
            });
        }
        var exportBtn = mount.querySelector('#geoPeerExportCsvBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', function () {
                downloadPeerComparisonCsv(mount, model);
            });
        }
        mount.__pbjPeerCompModel = model;
        bindPeriodSelects(mount, {
            scopeVal: ctx.scopeVal || 'latest',
            geoPeriodQuarterOpts: ctx.geoPeriodQuarterOpts || [],
            geoPeriodYearOpts: ctx.geoPeriodYearOpts || [],
            GEO_SCOPE_LS: ctx.GEO_SCOPE_LS,
            refresh: ctx.refresh,
        });
        var geoLoc = mount.querySelector('#geoRollupLocalePeers');
        if (geoLoc) {
            geoLoc.onchange = function () {
                try {
                    localStorage.setItem(ctx.GEO_LOCALE_LS, this.checked ? '1' : '0');
                } catch (e) {}
                ctx.refresh();
            };
        }
        refreshView();
        if (
            global.PbjV2GeoDistribution &&
            typeof global.PbjV2GeoDistribution.enhanceRollupMount === 'function'
        ) {
            global.PbjV2GeoDistribution.enhanceRollupMount();
        }
    }

    function renderMount(mount, ctx) {
        var model = buildModel(ctx);
        mount.innerHTML = renderShell(model, ctx);
        var geoDistCy = ctx.geoDistCy;
        if (geoDistCy) {
            mount.setAttribute('data-geo-cy-quarter', geoDistCy.cy);
            mount.setAttribute('data-geo-quarter-label', geoDistCy.label || '');
        }
        var geoColKeys = ['facility'];
        if (model.showCounty) {
            geoColKeys.push('county');
        }
        geoColKeys.push('state');
        if (model.showReg) {
            geoColKeys.push('region');
        }
        geoColKeys.push('national');
        mount.setAttribute('data-geo-column-keys', geoColKeys.join(','));
        bind(mount, model, ctx);
        var geoNotes = document.getElementById('geoRollupNotesCountsLine');
        if (geoNotes) {
            var note =
                'Peer counts — ' +
                (model.showCounty
                    ? model.geoLabels.county +
                      ' ' +
                      (model.counts.county != null ? fmtCount(model.counts.county) : '—') +
                      '; '
                    : '') +
                model.geoLabels.state +
                ' ' +
                (model.counts.state != null ? fmtCount(model.counts.state) : '—') +
                (model.showReg
                    ? '; ' +
                      model.geoLabels.region +
                      ' ' +
                      (model.counts.region != null ? fmtCount(model.counts.region) : '—')
                    : '') +
                '; national ' +
                (model.counts.national != null ? fmtCount(model.counts.national) : '—') +
                '.';
            if (model.useLocalePeers && model.facilityLocale !== 'unknown') {
                note += ' Locale peers: ' + (model.facilityLocale === 'rural' ? 'rural' : 'urban') + ' only.';
            }
            geoNotes.textContent = note;
        }
    }

    global.PbjV2PeerComparison = {
        buildModel: buildModel,
        renderMount: renderMount,
        bindPeriodSelects: bindPeriodSelects,
        populateGeoPeriodValueSelect: populateGeoPeriodValueSelect,
        readGeoPeriodScope: readGeoPeriodScope,
        THRESHOLDS: THRESHOLDS,
        METRIC_DEFS: METRIC_DEFS,
    };
})(typeof window !== 'undefined' ? window : this);
