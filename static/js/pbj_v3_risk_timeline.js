/**
 * PBJ320 V3 — unified Risk Timeline (normalized events from existing client data).
 */
(function (global) {
    'use strict';

    var FILTER_MAP = {
        staffing_flag: ['staffing_flag'],
        inspection: ['inspection', 'citation'],
        enforcement: ['enforcement', 'sff'],
        ownership: ['ownership', 'chow'],
        other: ['other'],
        all: null,
    };

    var TYPE_LABELS = {
        staffing_flag: 'Staffing flag',
        inspection: 'Inspection',
        citation: 'Citation',
        enforcement: 'Enforcement',
        sff: 'SFF',
        ownership: 'Ownership',
        chow: 'CHOW',
        other: 'Other',
    };

    /** User-facing type labels for facility-registry rows mapped to filter bucket "other". */
    var FACILITY_TYPE_DISPLAY = {
        name_change: 'Name change',
        admin_turnover: 'Admin TO',
        manual_incident: 'Facility event',
    };

    var TABLE_CLUSTER_TYPES = { citation: true, inspection: true };

    var CLUSTER_TYPE_PLURAL = {
        citation: 'citations',
        inspection: 'inspections',
        staffing_flag: 'staffing flags',
        enforcement: 'enforcement events',
        sff: 'SFF events',
        ownership: 'ownership events',
        chow: 'CHOW events',
        other: 'events',
    };

    var SEV_RANK = { crit: 4, warn: 3, info: 2, low: 1, '': 0 };

    function v3Active() {
        return !!global.__pbjV3PanesActive;
    }

    function esc(t) {
        return typeof global.escapeHtml === 'function' ? global.escapeHtml(t) : String(t || '');
    }

    function escAttr(t) {
        return typeof global.pbjEscapeAttr === 'function' ? global.pbjEscapeAttr(t) : esc(t);
    }

    function isoOk(iso) {
        return /^\d{4}-\d{2}-\d{2}$/.test(String(iso || '').slice(0, 10));
    }

    function normDate(raw) {
        var s = String(raw || '').trim();
        if (isoOk(s)) {
            return s.slice(0, 10);
        }
        var m = s.match(/^(\d{2})-(\d{2})-(\d{4})$/);
        if (m) {
            return m[3] + '-' + m[1] + '-' + m[2];
        }
        m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
        if (m) {
            return m[0];
        }
        return '';
    }

    function formatDateDisplay(iso) {
        if (!isoOk(iso)) {
            return '—';
        }
        if (typeof global.formatDate === 'function') {
            return global.formatDate(iso);
        }
        var p = iso.split('-');
        return p[1] + '/' + p[2] + '/' + p[0];
    }

    function quarterMidIso(q) {
        if (typeof global.pbjQuarterToMidIso === 'function') {
            return global.pbjQuarterToMidIso(q);
        }
        var cy =
            typeof global.pbjNormalizeQuarterToCy === 'function'
                ? global.pbjNormalizeQuarterToCy(q)
                : String(q || '');
        var m = cy.match(/^(\d{4})Q([1-4])$/);
        if (!m) {
            return '';
        }
        var month = (parseInt(m[2], 10) - 1) * 3 + 2;
        return m[1] + '-' + String(month).padStart(2, '0') + '-15';
    }

    function mapFacilityEventType(evType) {
        switch (evType) {
            case 'citation_g_plus':
                return 'citation';
            case 'chow':
                return 'chow';
            case 'ownership_provider':
                return 'ownership';
            case 'sff_status':
                return 'sff';
            case 'abuse_event':
                return 'enforcement';
            case 'admin_turnover':
            case 'manual_incident':
            case 'name_change':
                return 'other';
            default:
                return 'other';
        }
    }

    function severityFromScope(scope) {
        var s = String(scope || '').trim().toUpperCase();
        if (!s) {
            return { label: '—', tone: '' };
        }
        if (/^J/.test(s)) {
            return { label: s, tone: 'crit' };
        }
        if (/^[HI]/.test(s)) {
            return { label: s, tone: 'warn' };
        }
        if (/^G/.test(s)) {
            return { label: s, tone: 'warn' };
        }
        return { label: s, tone: 'info' };
    }

    function shortLabelText(text, maxLen) {
        var s = String(text || '').trim();
        if (!s) {
            return '';
        }
        maxLen = maxLen || 24;
        if (s.length <= maxLen) {
            return s;
        }
        return s.slice(0, maxLen - 1) + '…';
    }

    function displayTypeLabel(ev) {
        if (!ev) {
            return 'Event';
        }
        if (
            ev.rawFacilityType === 'ownership_provider' ||
            (ev.type === 'ownership' && String(ev.label || '').trim())
        ) {
            var ownLbl = shortLabelText(ev.label, 24);
            if (ownLbl) {
                return ownLbl;
            }
            return 'Ownership';
        }
        if (ev.rawFacilityType && FACILITY_TYPE_DISPLAY[ev.rawFacilityType]) {
            return FACILITY_TYPE_DISPLAY[ev.rawFacilityType];
        }
        if (ev.type === 'other') {
            return 'Facility event';
        }
        return TYPE_LABELS[ev.type] || ev.type || 'Event';
    }

    function eventHasDetail(ev) {
        return !!(String(ev.summary || '').trim() || String(ev.periodScope || '').trim());
    }

    function typePillHtml(ev, detailId, options) {
        options = options || {};
        var typeKey = options.typeKey || (ev && ev.type) || 'other';
        var label = options.label != null ? options.label : displayTypeLabel(ev);
        var clusterClass = options.cluster ? ' pbj-v3-risk-type-pill--cluster' : '';
        var cls = 'pbj-v3-risk-type-pill pbj-v3-risk-type-pill--' + esc(typeKey) + clusterClass;
        var expandable =
            detailId &&
            (options.forceExpandable || options.cluster || (ev && eventHasDetail(ev)));
        if (expandable) {
            return (
                '<button type="button" class="' +
                cls +
                ' pbj-v3-risk-type-pill-btn" data-bs-toggle="collapse" data-bs-target="#' +
                detailId +
                '" aria-expanded="false" aria-controls="' +
                detailId +
                '" title="Toggle details">' +
                esc(label) +
                '<span class="pbj-v3-risk-type-chevron" aria-hidden="true">▾</span></button>'
            );
        }
        return '<span class="' + cls + '">' + esc(label) + '</span>';
    }

    function renderDetailRow(detailId, innerHtml, extraClass) {
        return (
            '<tr class="collapse pbj-v3-risk-detail-row' +
            (extraClass ? ' ' + extraClass : '') +
            '" id="' +
            detailId +
            '"><td colspan="5" class="pbj-v3-risk-detail-cell"><div class="pbj-v3-risk-detail-box">' +
            innerHtml +
            '</div></td></tr>'
        );
    }

    function clusterTypeLabel(type, count) {
        var plural = CLUSTER_TYPE_PLURAL[type];
        if (plural) {
            return count + ' ' + plural;
        }
        var base = displayTypeLabel({ type: type });
        return count + ' ' + String(base).toLowerCase();
    }

    function clusterTrackEvents(events) {
        var groups = {};
        var order = [];
        events.forEach(function (ev) {
            var key = String(ev.date) + '|' + String(ev.type);
            if (!groups[key]) {
                groups[key] = [];
                order.push(key);
            }
            groups[key].push(ev);
        });
        return order.map(function (key) {
            var items = groups[key];
            if (items.length === 1) {
                return { clustered: false, ev: items[0] };
            }
            return {
                clustered: true,
                date: items[0].date,
                type: items[0].type,
                count: items.length,
                firstId: items[0].id,
            };
        });
    }

    function severityIsEmpty(tone, label) {
        var lbl = String(label || '').trim();
        return !tone && (!lbl || lbl === '—' || lbl === '-');
    }

    function severityBadge(tone, label) {
        if (severityIsEmpty(tone, label)) {
            return '';
        }
        var lbl = String(label || '').trim();
        if (!lbl || lbl === '—' || lbl === '-') {
            return '<span class="pbj-v3-risk-sev-empty" aria-hidden="true">—</span>';
        }
        var cls = 'text-bg-light border text-muted';
        if (tone === 'crit') {
            cls = 'pbj-v3-risk-sev--crit';
        } else if (tone === 'warn') {
            cls = 'pbj-v3-risk-sev--warn';
        } else if (tone === 'info') {
            cls = 'pbj-v3-risk-sev--info';
        }
        return '<span class="badge rounded-pill ' + cls + ' fw-normal">' + esc(lbl) + '</span>';
    }

    function clusterTableTypeLabel(type) {
        if (type === 'citation') {
            return 'Citation cluster';
        }
        if (type === 'inspection') {
            return 'Inspection cluster';
        }
        return displayTypeLabel({ type: type }) + ' cluster';
    }

    function clusterTableSignalLabel(type, count) {
        if (type === 'citation') {
            return count + ' inspection citation' + (count === 1 ? '' : 's');
        }
        if (type === 'inspection') {
            return count + ' inspection' + (count === 1 ? '' : 's');
        }
        return clusterTypeLabel(type, count);
    }

    function clusterTableSeverity(items) {
        var best = items[0];
        items.forEach(function (ev) {
            if ((SEV_RANK[ev.severityTone] || 0) > (SEV_RANK[best.severityTone] || 0)) {
                best = ev;
            }
        });
        return { tone: best.severityTone, label: best.severity };
    }

    function clusterTableSource(items) {
        var src = items[0] && items[0].source ? String(items[0].source) : '';
        items.forEach(function (ev) {
            if (ev.source && ev.source !== src) {
                src = 'CMS inspection';
            }
        });
        return src || 'CMS inspection';
    }

    function clusterTableEvents(events) {
        var groups = {};
        events.forEach(function (ev) {
            if (!TABLE_CLUSTER_TYPES[ev.type]) {
                return;
            }
            var key = String(ev.date) + '|' + String(ev.type);
            if (!groups[key]) {
                groups[key] = [];
            }
            groups[key].push(ev);
        });
        var emitted = {};
        var result = [];
        events.forEach(function (ev) {
            if (!TABLE_CLUSTER_TYPES[ev.type]) {
                result.push({ clustered: false, ev: ev });
                return;
            }
            var key = String(ev.date) + '|' + String(ev.type);
            if (emitted[key]) {
                return;
            }
            emitted[key] = true;
            var items = groups[key];
            if (items.length === 1) {
                result.push({ clustered: false, ev: items[0] });
            } else {
                result.push({
                    clustered: true,
                    date: items[0].date,
                    type: items[0].type,
                    count: items.length,
                    items: items,
                    firstId: items[0].id,
                });
            }
        });
        return result;
    }

    function renderEventDetailHtml(ev) {
        return (
            '<div class="small text-muted">' +
            esc(ev.summary || '—') +
            (ev.periodScope ? '<div class="mt-1">Period: ' + esc(ev.periodScope) + '</div>' : '') +
            '</div>'
        );
    }

    function renderNestedEventRow(ev, parentDetailId) {
        var detailId = 'pbj-v3-risk-detail-' + ev.id.replace(/[^a-zA-Z0-9_-]/g, '_');
        var sevHtml = severityBadge(ev.severityTone, ev.severity);
        var detailRow = eventHasDetail(ev)
            ? renderDetailRow(detailId, renderEventDetailHtml(ev), 'pbj-v3-risk-detail-row--nested')
            : '';
        return (
            '<tr class="pbj-v3-risk-row pbj-v3-risk-row--nested" data-pbj-v3-risk-ev="' +
            escAttr(ev.id) +
            '" data-pbj-v3-risk-nested-parent="' +
            escAttr(parentDetailId) +
            '">' +
            '<td class="pbj-v3-risk-col-date"></td>' +
            '<td class="text-nowrap pbj-v3-risk-col-type">' +
            typePillHtml(ev, detailId) +
            '</td>' +
            '<td class="pbj-v3-risk-col-signal">' +
            '<div class="pbj-v3-risk-signal-line">' +
            '<span class="pbj-v3-risk-signal-text">' +
            esc(ev.label) +
            '</span>' +
            '<span class="d-md-none">' +
            sevHtml +
            '</span>' +
            '</div>' +
            '</td>' +
            '<td class="pbj-v3-risk-col-severity d-none d-md-table-cell">' +
            sevHtml +
            '</td>' +
            '<td class="small text-muted pbj-v3-risk-col-source d-none d-lg-table-cell">' +
            esc(ev.source) +
            '</td>' +
            '</tr>' +
            detailRow
        );
    }

    function renderTableRow(ev, stripeClass) {
        var detailId = 'pbj-v3-risk-detail-' + ev.id.replace(/[^a-zA-Z0-9_-]/g, '_');
        var sevHtml = severityBadge(ev.severityTone, ev.severity);
        var detailRow = eventHasDetail(ev) ? renderDetailRow(detailId, renderEventDetailHtml(ev)) : '';
        var stripe = stripeClass ? ' ' + stripeClass : '';
        return (
            '<tr class="pbj-v3-risk-row' + stripe + '" data-pbj-v3-risk-ev="' +
            escAttr(ev.id) +
            '">' +
            '<td class="text-nowrap pbj-v3-risk-col-date">' +
            esc(formatDateDisplay(ev.date)) +
            '</td>' +
            '<td class="text-nowrap pbj-v3-risk-col-type">' +
            typePillHtml(ev, detailId) +
            '</td>' +
            '<td class="pbj-v3-risk-col-signal">' +
            '<div class="pbj-v3-risk-signal-line">' +
            '<span class="pbj-v3-risk-signal-text">' +
            esc(ev.label) +
            '</span>' +
            '<span class="d-md-none">' +
            sevHtml +
            '</span>' +
            '</div>' +
            '</td>' +
            '<td class="pbj-v3-risk-col-severity d-none d-md-table-cell">' +
            sevHtml +
            '</td>' +
            '<td class="small text-muted pbj-v3-risk-col-source d-none d-lg-table-cell">' +
            esc(ev.source) +
            '</td>' +
            '</tr>' +
            detailRow
        );
    }

    function renderClusterTableRow(item, stripeClass) {
        var clusterId = 'pbj-v3-risk-cluster-' + item.firstId.replace(/[^a-zA-Z0-9_-]/g, '_');
        var sev = clusterTableSeverity(item.items);
        var sevHtml = severityBadge(sev.tone, sev.label);
        var nested =
            '<table class="table table-sm mb-0 pbj-v3-risk-cluster-nested"><tbody>' +
            item.items
                .map(function (ev) {
                    return renderNestedEventRow(ev, clusterId);
                })
                .join('') +
            '</tbody></table>';
        var stripe = stripeClass ? ' ' + stripeClass : '';
        return (
            '<tr class="pbj-v3-risk-row pbj-v3-risk-row--cluster' + stripe + '" data-pbj-v3-risk-ev="' +
            escAttr(item.firstId) +
            '" data-pbj-v3-risk-cluster="' +
            escAttr(String(item.count)) +
            '">' +
            '<td class="text-nowrap pbj-v3-risk-col-date">' +
            esc(formatDateDisplay(item.date)) +
            '</td>' +
            '<td class="text-nowrap pbj-v3-risk-col-type">' +
            typePillHtml(null, clusterId, {
                typeKey: item.type,
                label: clusterTableTypeLabel(item.type),
                cluster: true,
                forceExpandable: true,
            }) +
            '</td>' +
            '<td class="pbj-v3-risk-col-signal">' +
            '<div class="pbj-v3-risk-signal-line">' +
            '<span class="pbj-v3-risk-signal-text">' +
            esc(clusterTableSignalLabel(item.type, item.count)) +
            '</span>' +
            '<span class="d-md-none">' +
            sevHtml +
            '</span>' +
            '</div>' +
            '</td>' +
            '<td class="pbj-v3-risk-col-severity d-none d-md-table-cell">' +
            sevHtml +
            '</td>' +
            '<td class="small text-muted pbj-v3-risk-col-source d-none d-lg-table-cell">' +
            esc(clusterTableSource(item.items)) +
            '</td>' +
            '</tr>' +
            '<tr class="collapse pbj-v3-risk-detail-row pbj-v3-risk-cluster-detail" id="' +
            clusterId +
            '"><td colspan="5" class="p-0 pbj-v3-risk-cluster-detail-cell">' +
            nested +
            '</td></tr>'
        );
    }

    function eventKey(ev) {
        return (
            String(ev.type) +
            '|' +
            String(ev.date) +
            '|' +
            String(ev.label || '')
                .toLowerCase()
                .slice(0, 48)
        );
    }

    function pushUnique(list, seen, ev) {
        if (!ev || !ev.date) {
            return;
        }
        var k = eventKey(ev);
        if (seen[k]) {
            return;
        }
        seen[k] = true;
        ev.sortKey = ev.date + '|' + String(SEV_RANK[ev.severityTone] || 0) + '|' + k;
        list.push(ev);
    }

    function collectFacilityEvents(seen, out) {
        var reg =
            typeof global.pbjFacilityEventsDisplayRegistry === 'function'
                ? global.pbjFacilityEventsDisplayRegistry()
                : global.__pbjFacilityEventsRegistry || [];
        reg.forEach(function (ev) {
            if (!ev || !ev.date_iso) {
                return;
            }
            var mapped = mapFacilityEventType(ev.type);
            var sev = mapped === 'citation' || mapped === 'sff' || mapped === 'enforcement' ? 'warn' : 'info';
            pushUnique(out, seen, {
                id: ev.id || eventKey({ type: mapped, date: ev.date_iso, label: ev.label }),
                date: ev.date_iso,
                type: mapped,
                rawFacilityType: ev.type,
                label: ev.label || ev.type,
                summary: ev.detail || '',
                severity: sev === 'warn' ? 'Elevated' : 'Info',
                severityTone: sev,
                source: ev.source || 'CMS',
                sourceUrl: '',
                details: ev,
                periodScope: '',
            });
        });
    }

    function collectCitationRows(seen, out) {
        var rows = Array.isArray(global.__pbjCitationsRows) ? global.__pbjCitationsRows : [];
        rows.forEach(function (row, idx) {
            var d = normDate(row.survey_date || row.SurveyDate || row.survey_dt);
            if (!d) {
                d = normDate(row.processing_date);
            }
            if (!d) {
                return;
            }
            var scope = row.scope_severity || row.ScopeSeverity || row.severity || '';
            var sev = severityFromScope(scope);
            var desc = String(row.description || row.Description || row.deficiency || '').trim();
            pushUnique(out, seen, {
                id: 'cit-' + idx + '-' + d,
                date: d,
                type: 'citation',
                label: desc.slice(0, 120) || 'Citation',
                summary: String(row.category || row.Category || '').trim(),
                severity: sev.label,
                severityTone: sev.tone,
                source: 'CMS inspection',
                sourceUrl: '',
                details: row,
                periodScope: '',
            });
        });
    }

    function isRegistryCoveredFlag(flag) {
        return /ownership|special focus|\bSFF\b|abuse|G\+|G-or-higher|Admin TO/i.test(String(flag || ''));
    }

    function collectStaffingFlags(seen, out) {
        var hist = Array.isArray(global.__lastRedFlagHistory) ? global.__lastRedFlagHistory : [];
        hist.forEach(function (item, hi) {
            var quarter = item && item.quarter ? String(item.quarter) : '';
            var d =
                normDate(item.citation_survey_min_iso) ||
                normDate(item.processing_date) ||
                quarterMidIso(quarter);
            if (!d) {
                return;
            }
            (item.red_flags || []).forEach(function (raw, fi) {
                var flag = String(raw || '').trim();
                if (!flag || isRegistryCoveredFlag(flag)) {
                    return;
                }
                var tone = /severe|1-star|below average|low staffing/i.test(flag) ? 'warn' : 'info';
                pushUnique(out, seen, {
                    id: 'rf-' + hi + '-' + fi,
                    date: d,
                    type: 'staffing_flag',
                    label: flag,
                    summary: quarter ? 'Quarter: ' + quarter : '',
                    severity: tone === 'warn' ? 'Watch' : 'Info',
                    severityTone: tone,
                    source: item.source_label || 'CMS Provider Information',
                    sourceUrl: '',
                    details: { quarter: quarter, raw: flag, item: item },
                    periodScope: quarter,
                });
            });
        });
    }

    function normalizeEvents() {
        var seen = {};
        var out = [];
        collectFacilityEvents(seen, out);
        collectCitationRows(seen, out);
        collectStaffingFlags(seen, out);
        out.sort(function (a, b) {
            return String(b.sortKey).localeCompare(String(a.sortKey));
        });
        return out;
    }

    function filterEvents(events, filterId) {
        var allowed = FILTER_MAP[filterId];
        if (!allowed) {
            return events;
        }
        return events.filter(function (ev) {
            return allowed.indexOf(ev.type) >= 0;
        });
    }

    function countByBucket(events) {
        var c = { inspection: 0, ownership: 0, staffing_flag: 0, enforcement: 0, other: 0 };
        events.forEach(function (ev) {
            if (ev.type === 'citation' || ev.type === 'inspection') {
                c.inspection += 1;
            } else if (ev.type === 'chow' || ev.type === 'ownership') {
                c.ownership += 1;
            } else if (ev.type === 'staffing_flag') {
                c.staffing_flag += 1;
            } else if (ev.type === 'enforcement' || ev.type === 'sff') {
                c.enforcement += 1;
            } else {
                c.other += 1;
            }
        });
        return c;
    }

    function periodLineText() {
        var el = document.getElementById('pbjScopeLabel') || document.getElementById('pbjV2ScopeLabel');
        if (el && el.textContent && el.textContent.trim()) {
            return el.textContent.trim();
        }
        if (typeof global.pbjControlCenterIsoDateRange === 'function') {
            var r = global.pbjControlCenterIsoDateRange();
            if (r && r.startDate && r.endDate) {
                return r.startDate + ' to ' + r.endDate;
            }
        }
        return 'All available history';
    }

    function renderCounts(events) {
        var host = document.getElementById('pbjV3RiskCountChips');
        if (!host) {
            return;
        }
        var c = countByBucket(events);
        var chips = [
            ['Inspections', c.inspection],
            ['Ownership/CHOW', c.ownership],
            ['Staffing flags', c.staffing_flag],
            ['Enforcement/SFF', c.enforcement],
            ['Other', c.other],
        ];
        host.innerHTML = chips
            .filter(function (pair) {
                return pair[1] > 0;
            })
            .map(function (pair) {
                return (
                    '<span class="badge rounded-pill pbj-v3-risk-count-chip">' +
                    esc(pair[0]) +
                    ' ' +
                    pair[1] +
                    '</span>'
                );
            })
            .join('');
    }

    function renderTrack(events) {
        var host = document.getElementById('pbjV3RiskTimelineTrack');
        if (!host) {
            return;
        }
        if (!events.length) {
            host.innerHTML = '<p class="small text-muted mb-0">No risk signals loaded yet for this facility.</p>';
            return;
        }
        var clusters = clusterTrackEvents(events);
        var max = 20;
        var slice = clusters.slice(0, max);
        var html = slice
            .map(function (item) {
                if (item.clustered) {
                    var title = clusterTypeLabel(item.type, item.count) + ' on ' + formatDateDisplay(item.date);
                    return (
                        '<button type="button" class="pbj-v3-risk-chip pbj-v3-risk-chip--cluster pbj-v3-risk-chip--' +
                        esc(item.type) +
                        '" data-pbj-v3-risk-ev="' +
                        escAttr(item.firstId) +
                        '" data-pbj-v3-risk-cluster="' +
                        escAttr(String(item.count)) +
                        '" title="' +
                        escAttr(title) +
                        '">' +
                        '<span class="pbj-v3-risk-chip-type">' +
                        esc(clusterTypeLabel(item.type, item.count)) +
                        '</span>' +
                        '<span class="pbj-v3-risk-chip-date">' +
                        esc(formatDateDisplay(item.date)) +
                        '</span>' +
                        '</button>'
                    );
                }
                var ev = item.ev;
                return (
                    '<button type="button" class="pbj-v3-risk-chip pbj-v3-risk-chip--' +
                    esc(ev.type) +
                    '" data-pbj-v3-risk-ev="' +
                    escAttr(ev.id) +
                    '" title="' +
                    escAttr(ev.label) +
                    '">' +
                    '<span class="pbj-v3-risk-chip-type">' +
                    esc(displayTypeLabel(ev)) +
                    '</span>' +
                    '<span class="pbj-v3-risk-chip-date">' +
                    esc(formatDateDisplay(ev.date)) +
                    '</span>' +
                    '</button>'
                );
            })
            .join('');
        var shownEventCount = slice.reduce(function (n, item) {
            return n + (item.clustered ? item.count : 1);
        }, 0);
        var more =
            events.length > shownEventCount
                ? '<span class="small text-muted align-self-center pbj-v3-risk-track-more">+' +
                  (events.length - shownEventCount) +
                  ' more in table</span>'
                : '';
        host.innerHTML = '<div class="pbj-v3-risk-track-inner">' + html + more + '</div>';
    }

    function buildInspectionRollup(events) {
        var citations = events.filter(function (ev) {
            return ev.type === 'citation';
        });
        var card = document.getElementById('pbjV3RiskInspectionRollup');
        var stats = document.getElementById('pbjV3RiskInspectionStats');
        var cms = document.getElementById('pbjV3RiskInspectionCmsLink');
        if (!card || !stats) {
            return;
        }
        var meta = global.__pbjCitationsSummaryMeta || {};
        var total = meta.total != null ? meta.total : citations.length;
        if (!total && !citations.length) {
            card.hidden = true;
            return;
        }
        card.hidden = false;
        var lastDate = citations.length ? citations[0].date : '';
        var severe = citations.filter(function (c) {
            return c.severityTone === 'crit' || c.severityTone === 'warn';
        }).length;
        var topLabel = '—';
        if (citations.length) {
            var topEv = citations.reduce(function (best, c) {
                return (SEV_RANK[c.severityTone] || 0) > (SEV_RANK[best.severityTone] || 0) ? c : best;
            }, citations[0]);
            topLabel = topEv.severity || '—';
        }
        var summaryParts = [
            'Last survey ' + (lastDate ? formatDateDisplay(lastDate) : '—'),
            String(total) + ' citation' + (total === 1 ? '' : 's'),
            String(severe) + ' serious',
            'Highest: ' + topLabel,
        ];
        stats.innerHTML =
            '<span class="pbj-v3-risk-inspection-summary-label">Inspection summary:</span> ' +
            esc(summaryParts.join(' · '));
        if (cms) {
            var linkEl = document.getElementById('pbjCitationsCmsLink');
            cms.innerHTML = linkEl ? linkEl.innerHTML : '';
        }
    }

    function renderTable(events) {
        var tbody = document.getElementById('pbjV3RiskEventsTbody');
        var empty = document.getElementById('pbjV3RiskEventsEmpty');
        if (!tbody) {
            return;
        }
        global.__pbjV3RiskTimelineEvents = events;
        if (!events.length) {
            tbody.innerHTML = '';
            if (empty) {
                empty.classList.remove('d-none');
            }
            return;
        }
        if (empty) {
            empty.classList.add('d-none');
        }
        tbody.innerHTML = clusterTableEvents(events)
            .map(function (item, idx) {
                var stripeClass = idx % 2 === 0 ? 'pbj-v3-risk-row--stripe-odd' : 'pbj-v3-risk-row--stripe-even';
                if (item.clustered) {
                    return renderClusterTableRow(item, stripeClass);
                }
                return renderTableRow(item.ev, stripeClass);
            })
            .join('');
    }

    function currentFilter() {
        var active = document.querySelector('#pbjV3RiskFilterGroup .active[data-pbj-v3-risk-filter]');
        return active ? active.getAttribute('data-pbj-v3-risk-filter') || 'all' : 'all';
    }

    function refresh() {
        if (!v3Active()) {
            return;
        }
        var shell = document.getElementById('pbjV3RiskTimelineShell');
        if (!shell) {
            return;
        }
        var periodEl = document.getElementById('pbjV3RiskPeriodLine');
        if (periodEl) {
            periodEl.textContent = 'Scope: ' + periodLineText();
        }
        var all = normalizeEvents();
        global.__pbjV3RiskTimelineAllEvents = all;
        var filtered = filterEvents(all, currentFilter());
        renderCounts(all);
        renderTrack(filtered);
        buildInspectionRollup(all);
        renderTable(filtered);
        if (typeof global.pbjV3UpdateRiskHandoff === 'function') {
            global.pbjV3UpdateRiskHandoff();
        }
    }

    function wireFiltersOnce() {
        if (global.__pbjV3RiskTimelineWired) {
            return;
        }
        global.__pbjV3RiskTimelineWired = true;
        var group = document.getElementById('pbjV3RiskFilterGroup');
        if (group) {
            group.addEventListener('click', function (e) {
                var btn = e.target.closest('[data-pbj-v3-risk-filter]');
                if (!btn) {
                    return;
                }
                group.querySelectorAll('[data-pbj-v3-risk-filter]').forEach(function (b) {
                    b.classList.toggle('active', b === btn);
                });
                refresh();
            });
        }
        var exportBtn = document.getElementById('pbjV3RiskExportCsvBtn');
        if (exportBtn) {
            exportBtn.addEventListener('click', function () {
                var rows = global.__pbjV3RiskTimelineAllEvents || [];
                if (!rows.length) {
                    alert('No timeline events to export.');
                    return;
                }
                var lines = ['Date,Type,Signal,Severity,Source,Summary'];
                rows.forEach(function (ev) {
                    lines.push(
                        [
                            ev.date,
                            displayTypeLabel(ev),
                            '"' + String(ev.label || '').replace(/"/g, '""') + '"',
                            ev.severity,
                            '"' + String(ev.source || '').replace(/"/g, '""') + '"',
                            '"' + String(ev.summary || '').replace(/"/g, '""') + '"',
                        ].join(',')
                    );
                });
                var blob = new Blob([lines.join('\n') + '\n'], { type: 'text/csv;charset=utf-8' });
                var a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download =
                    'pbj320_risk_timeline_' +
                    String(global.PROVNUM || global.PBJ320_EXPORT_CCN || 'facility') +
                    '.csv';
                document.body.appendChild(a);
                a.click();
                a.remove();
            });
        }
        document.addEventListener('click', function (e) {
            var chip = e.target.closest('[data-pbj-v3-risk-ev]');
            if (!chip) {
                return;
            }
            var id = chip.getAttribute('data-pbj-v3-risk-ev');
            var row = document.querySelector('.pbj-v3-risk-row[data-pbj-v3-risk-ev="' + id + '"]');
            if (row) {
                row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                row.classList.add('pbj-v3-risk-row--flash');
                setTimeout(function () {
                    row.classList.remove('pbj-v3-risk-row--flash');
                }, 1200);
            }
        });
    }

    function mountLegacySlots() {
        if (!v3Active() || global.__pbjV3RiskLegacyMounted) {
            return;
        }
        var citBlock = document.getElementById('pbjCitationsTableBlock');
        var citSlot = document.querySelector('[data-pbj-v3-legacy-slot="citations"]');
        if (citBlock && citSlot && !citSlot.contains(citBlock)) {
            citSlot.appendChild(citBlock);
        }
        var flagSec = document.getElementById('redFlagHistorySection');
        var flagSlot = document.querySelector('[data-pbj-v3-legacy-slot="flags"]');
        if (flagSec && flagSlot && !flagSlot.contains(flagSec)) {
            flagSlot.appendChild(flagSec);
        }
        global.__pbjV3RiskLegacyMounted = true;
    }

    function init() {
        if (!v3Active()) {
            return;
        }
        mountLegacySlots();
        wireFiltersOnce();
        refresh();
    }

    global.pbjV3RiskTimelineRefresh = refresh;
    global.pbjV3RiskTimelineInit = init;
})(typeof window !== 'undefined' ? window : globalThis);
