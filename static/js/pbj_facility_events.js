/**
 * Superdynamic v2 — facility timeline events (master toggle + hierarchical modal).
 */
(function (global) {
    'use strict';

    var TYPE_ORDER = [
        'manual_incident',
        'chow',
        'citation_g_plus',
        'ownership_provider',
        'name_change',
        'sff_status',
        'admin_turnover',
        'abuse_event'
    ];

    var EVENT_TYPES = {
        manual_incident: {
            label: 'User event',
            shortLabel: 'Event',
            desc: 'Notes you add for this facility',
            help: 'Manual markers for dates you want on charts (falls, policy changes, etc.). They appear first on charts and in this list.',
            color: 'rgba(217, 119, 6, 0.52)',
            dash: 'dash',
            css: 'pbj-event-type-badge--manual_incident'
        },
        chow: {
            label: 'CHOW',
            shortLabel: 'CHOW',
            desc: 'Ownership transfer effective dates',
            help: 'When CMS records a change of ownership for this nursing home.',
            color: 'rgba(111, 66, 193, 0.45)',
            dash: 'dot',
            css: 'pbj-event-type-badge--chow'
        },
        citation_g_plus: {
            label: 'G+ citation',
            shortLabel: 'G+',
            desc: 'Severe survey citations (G and above)',
            help: 'High-harm deficiency citations from CMS inspection surveys, grouped by survey quarter.',
            color: 'rgba(220, 53, 69, 0.36)',
            dash: 'dot',
            css: 'pbj-event-type-badge--citation_g_plus'
        },
        ownership_provider: {
            label: 'Ownership',
            shortLabel: 'Ownership',
            desc: 'Quarterly ownership change signal (hidden near a CHOW date)',
            help: 'CMS flags when ownership changed in the prior 12 months. Omitted when a CHOW covers the same window.',
            color: 'rgba(108, 117, 125, 0.42)',
            dash: 'dot',
            css: 'pbj-event-type-badge--ownership_provider'
        },
        name_change: {
            label: 'Name change',
            shortLabel: 'Name Change',
            desc: 'Significant facility name change',
            help: 'When the CMS-registered name changed materially from a prior quarter.',
            color: 'rgba(13, 110, 253, 0.38)',
            dash: 'dot',
            css: 'pbj-event-type-badge--name_change'
        },
        sff_status: {
            label: 'SFF status',
            shortLabel: 'SFF',
            desc: 'Special Focus Facility or candidate status',
            help: 'When CMS designated this home as an SFF or SFF candidate.',
            color: 'rgba(220, 53, 69, 0.4)',
            dash: 'dot',
            css: 'pbj-event-type-badge--sff_status'
        },
        admin_turnover: {
            label: 'Admin turnover',
            shortLabel: 'Admin TO',
            desc: 'Administrator turnover in the prior year',
            help: 'How many administrators left the nursing home in the prior 12 months.',
            color: 'rgba(255, 193, 7, 0.45)',
            dash: 'dot',
            css: 'pbj-event-type-badge--admin_turnover'
        },
        abuse_event: {
            label: 'Abuse icon',
            shortLabel: 'Abuse',
            desc: 'Abuse icon on Care Compare',
            help: 'When CMS displayed an abuse icon for this facility on Care Compare.',
            color: 'rgba(111, 66, 193, 0.42)',
            dash: 'dot',
            css: 'pbj-event-type-badge--abuse_event'
        }
    };

    function pbjEventsCcn() {
        var p = String(
            global.PROVNUM ||
                (typeof global.PBJ320_EXPORT_CCN !== 'undefined' ? global.PBJ320_EXPORT_CCN : '') ||
                ''
        ).replace(/\D/g, '');
        return p ? p.padStart(6, '0').slice(-6) : '';
    }

    function pbjFacilityEventsMigrateLegacyStorage() {
        var ccn = pbjEventsCcn();
        if (!ccn) {
            return;
        }
        try {
            var legacyManual = localStorage.getItem('pbj_facility_events_manual');
            if (legacyManual !== null && localStorage.getItem(pbjEventsManualKey()) === null) {
                var parsed = legacyManual ? JSON.parse(legacyManual) : [];
                var scoped = Array.isArray(parsed)
                    ? parsed.filter(function (row) {
                          if (!row || typeof row !== 'object') {
                              return false;
                          }
                          if (!row.ccn) {
                              return false;
                          }
                          return String(row.ccn).replace(/\D/g, '').slice(-6) === ccn;
                      })
                    : [];
                if (scoped.length) {
                    localStorage.setItem(pbjEventsManualKey(), JSON.stringify(scoped));
                }
                localStorage.removeItem('pbj_facility_events_manual');
            }
            [
                ['pbj_facility_events_master', pbjEventsMasterKey()],
                ['pbj_facility_events_types', pbjEventsTypesKey()]
            ].forEach(function (pair) {
                var legacy = localStorage.getItem(pair[0]);
                if (legacy !== null && localStorage.getItem(pair[1]) === null) {
                    localStorage.setItem(pair[1], legacy);
                    localStorage.removeItem(pair[0]);
                }
            });
        } catch (eMigrate) { /* ignore */ }
    }

    function pbjEventsMasterKey() {
        return 'pbj_facility_events_master_' + pbjEventsCcn();
    }

    function pbjEventsTypesKey() {
        return 'pbj_facility_events_types_' + pbjEventsCcn();
    }

    function pbjEventsManualKey() {
        return 'pbj_facility_events_manual_' + pbjEventsCcn();
    }

    function pbjEventsDefaultTypeState(allOn) {
        var o = {};
        TYPE_ORDER.forEach(function (t) {
            o[t] = !!allOn;
        });
        return o;
    }

    function pbjFacilityEventsLoadTypeState() {
        try {
            var raw = localStorage.getItem(pbjEventsTypesKey());
            if (raw === null) {
                return pbjEventsDefaultTypeState(true);
            }
            var parsed = raw ? JSON.parse(raw) : null;
            var base = pbjEventsDefaultTypeState(false);
            if (parsed && typeof parsed === 'object') {
                TYPE_ORDER.forEach(function (t) {
                    if (typeof parsed[t] === 'boolean') {
                        base[t] = parsed[t];
                    }
                });
            }
            return base;
        } catch (e) {
            return pbjEventsDefaultTypeState(true);
        }
    }

    function pbjFacilityEventsSaveTypeState(state) {
        try {
            localStorage.setItem(pbjEventsTypesKey(), JSON.stringify(state || {}));
        } catch (e2) { /* ignore */ }
    }

    function pbjFacilityEventsMasterOn() {
        return global.__pbjFacilityEventsMaster === true;
    }

    function pbjFacilityEventsTypeEnabled(type) {
        if (!pbjFacilityEventsMasterOn()) {
            return false;
        }
        var st = global.__pbjFacilityEventsTypes || pbjEventsDefaultTypeState(false);
        return !!st[type];
    }

    function pbjFacilityEventsMarkersActive() {
        if (!pbjFacilityEventsMasterOn()) {
            return false;
        }
        var reg = global.__pbjFacilityEventsRegistry || [];
        return reg.some(function (ev) {
            return pbjFacilityEventsTypeEnabled(ev.type);
        });
    }

    function pbjFacilityEventsVisible() {
        return pbjFacilityEventsMarkersActive();
    }

    function pbjFacilityEventsSetMaster(on) {
        global.__pbjFacilityEventsMaster = !!on;
        try {
            localStorage.setItem(pbjEventsMasterKey(), on ? '1' : '0');
        } catch (e3) { /* ignore */ }
        pbjFacilityEventsSyncUi();
    }

    function pbjFacilityEventsSetType(type, on) {
        global.__pbjFacilityEventsTypes = global.__pbjFacilityEventsTypes || pbjEventsDefaultTypeState(false);
        if (EVENT_TYPES[type]) {
            global.__pbjFacilityEventsTypes[type] = !!on;
            pbjFacilityEventsSaveTypeState(global.__pbjFacilityEventsTypes);
        }
        pbjFacilityEventsSyncUi();
    }

    function pbjFacilityEventsLoadManual() {
        pbjFacilityEventsMigrateLegacyStorage();
        var ccn = pbjEventsCcn();
        if (!ccn) {
            return [];
        }
        try {
            var raw = localStorage.getItem(pbjEventsManualKey());
            var arr = raw ? JSON.parse(raw) : [];
            if (!Array.isArray(arr)) {
                return [];
            }
            return arr.filter(function (row) {
                if (!row || typeof row !== 'object') {
                    return false;
                }
                if (row.ccn && String(row.ccn).replace(/\D/g, '').slice(-6) !== ccn) {
                    return false;
                }
                return true;
            });
        } catch (e4) {
            return [];
        }
    }

    function pbjFacilityEventsSaveManual(rows) {
        var ccn = pbjEventsCcn();
        if (!ccn) {
            return;
        }
        try {
            var stamped = (rows || []).map(function (row) {
                var out = Object.assign({}, row || {});
                out.ccn = ccn;
                return out;
            });
            localStorage.setItem(pbjEventsManualKey(), JSON.stringify(stamped));
        } catch (e5) { /* ignore */ }
    }

    function pbjQuarterToMidIso(qRaw) {
        var cy = null;
        if (typeof global.pbjNormalizeQuarterToCy === 'function') {
            cy = global.pbjNormalizeQuarterToCy(qRaw);
        }
        if (!cy) {
            var s = String(qRaw || '').trim();
            var m = s.match(/^Q([1-4])\s+(\d{4})$/i);
            if (m) {
                cy = m[2] + 'Q' + m[1];
            } else if (/^\d{4}Q[1-4]$/i.test(s)) {
                cy = s.toUpperCase();
            }
        }
        if (!cy || cy.length !== 6) {
            return null;
        }
        var y = parseInt(cy.slice(0, 4), 10);
        var qn = parseInt(cy.slice(5), 10);
        var month = (qn - 1) * 3 + 1;
        return y + '-' + String(month).padStart(2, '0') + '-15';
    }

    function pbjIsoOk(iso) {
        return /^\d{4}-\d{2}-\d{2}$/.test(String(iso || '').trim());
    }

    var PBJ_TITLE_SUFFIX = { llc: 'LLC', inc: 'Inc.', lp: 'LP', llp: 'LLP', pc: 'PC', pllc: 'PLLC', corp: 'Corp.', dba: 'DBA', snf: 'SNF' };
    var PBJ_TITLE_SMALL = { at: true, of: true, the: true, and: true, for: true, in: true };

    function pbjSmartDisplayCase(str) {
        var s = String(str || '').trim();
        if (!s || s === '—') {
            return s;
        }
        var letters = (s.match(/[A-Za-z]/g) || []).length;
        var uppers = (s.match(/[A-Z]/g) || []).length;
        if (letters > 3 && uppers / letters > 0.85) {
            return s.toLowerCase().split(/\s+/).map(function (word, i) {
                var bare = word.replace(/[^a-z0-9]/gi, '').toLowerCase();
                if (PBJ_TITLE_SUFFIX[bare]) {
                    return PBJ_TITLE_SUFFIX[bare];
                }
                if (i > 0 && PBJ_TITLE_SMALL[bare]) {
                    return bare;
                }
                if (!word.length) {
                    return word;
                }
                return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
            }).join(' ');
        }
        return s;
    }

    function pbjHumanizeChowTag(tag) {
        var t = String(tag || '').trim();
        if (!t) {
            return '';
        }
        if (/^entity changed$/i.test(t)) {
            return 'Entity changed';
        }
        if (/^identifiers? changed$/i.test(t)) {
            return 'ID fields updated';
        }
        return t.charAt(0).toUpperCase() + t.slice(1).toLowerCase();
    }

    function pbjFormatChowEventDetail(t) {
        var buyer = pbjSmartDisplayCase(t.buyer_org_name || t.buyer_dba_name || '');
        var seller = pbjSmartDisplayCase(t.seller_org_name || '');
        if (buyer && seller) {
            return buyer + ' ← ' + seller;
        }
        return buyer || seller || '';
    }

    function pbjEscapeHtml(s) {
        if (typeof global.escapeHtml === 'function') {
            return global.escapeHtml(s);
        }
        return String(s || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    var PBJ_PROVIDER_INFO_DATASET_URL = 'https://data.cms.gov/provider-data/dataset/4pq5-n9py';

    function pbjNormalizeEventSourceLabel(label) {
        var s = String(label || '').trim();
        if (!s) {
            return '';
        }
        s = s.replace(/CMS Nursing Home Provider Information/gi, 'CMS Provider Information');
        s = s.replace(/\s*\([^)]+\.csv\)\s*$/i, '').trim();
        return s;
    }

    function pbjFormatEventQuarterLabel(qRaw) {
        var s = String(qRaw || '').trim();
        if (!s) {
            return '';
        }
        var cy = null;
        if (typeof global.pbjNormalizeQuarterToCy === 'function') {
            cy = global.pbjNormalizeQuarterToCy(s);
        }
        if (!cy) {
            var m = s.match(/^Q([1-4])\s+(\d{4})$/i);
            if (m) {
                cy = m[2] + 'Q' + m[1];
            } else if (/^\d{4}Q[1-4]$/i.test(s)) {
                cy = s.toUpperCase();
            } else if (/^CY\d{4}Q[1-4]$/i.test(s)) {
                cy = s.slice(2).toUpperCase();
            }
        }
        if (cy && typeof global.formatCyQuarterLabel === 'function') {
            var formatted = global.formatCyQuarterLabel('CY' + cy);
            if (formatted) {
                return formatted;
            }
        }
        return s;
    }

    function pbjEventSourceMeta(ev) {
        if (!ev) {
            return { label: '', url: '' };
        }
        if (ev.type === 'manual_incident') {
            return { label: 'User', url: '' };
        }
        if (ev.type === 'chow') {
            return {
                label: 'CMS CHOW',
                url: String(ev.source_url || '').trim()
            };
        }
        var label = pbjNormalizeEventSourceLabel(ev.source_label || ev.source || '');
        var url = String(ev.source_url || '').trim();
        if (ev.type === 'name_change') {
            label = pbjNormalizeEventSourceLabel(
                global.__pbjProviderPreviousNameSourceLabel || label || 'CMS Provider Information'
            );
            url = url || String(global.__pbjProviderPreviousNameCmsUrl || '').trim();
        }
        if (!label && ev.type === 'citation_g_plus') {
            label = 'CMS Deficiencies';
        }
        if (!label) {
            label = 'CMS Provider Information';
        }
        if (!url && /CMS Provider Information/i.test(label)) {
            url = PBJ_PROVIDER_INFO_DATASET_URL;
        }
        return { label: label, url: url };
    }

    function pbjFormatEventSourceText(ev) {
        return pbjEventSourceMeta(ev).label;
    }

    function pbjFormatEventSourceHtml(ev) {
        var meta = pbjEventSourceMeta(ev);
        if (!meta.label) {
            return '';
        }
        var html = '<span class="text-muted">Source:</span> ';
        if (meta.url) {
            html +=
                '<a href="' +
                pbjEscapeHtml(meta.url) +
                '" target="_blank" rel="noopener">' +
                pbjEscapeHtml(meta.label) +
                '</a>';
        } else {
            html += pbjEscapeHtml(meta.label);
        }
        return html;
    }

    function pbjNameChangeEventBodyHtml() {
        var priorNm = global.__pbjProviderPreviousNameSignificant
            ? String(global.__pbjProviderPreviousNameSignificant).trim()
            : '';
        var priorQ = pbjFormatEventQuarterLabel(global.__pbjProviderPreviousNameQuarter);
        var eventQ = pbjFormatEventQuarterLabel(global.__pbjProviderPreviousNameEventQuarter);
        var currentNm =
            (global.PBJ320_EXPORT_FACILITY_DISPLAY && String(global.PBJ320_EXPORT_FACILITY_DISPLAY).trim()) ||
            (document.getElementById('pbjProfileFacilityName') &&
                String(document.getElementById('pbjProfileFacilityName').textContent || '').trim()) ||
            '';
        if (!priorNm && !currentNm) {
            return '<p class="mb-0 text-muted small">Significant facility name change in CMS Provider Information.</p>';
        }
        var priorDisplay = pbjSmartDisplayCase(priorNm || '—');
        var currentDisplay = pbjSmartDisplayCase(currentNm || '—');
        var timeline = '';
        if (priorQ && eventQ) {
            timeline = priorQ + ' → ' + eventQ;
        } else if (eventQ) {
            timeline = eventQ;
        } else if (priorQ) {
            timeline = 'Prior name last in ' + priorQ;
        }
        return (
            '<p class="mb-1">' +
            pbjEscapeHtml(priorDisplay) +
            ' → ' +
            pbjEscapeHtml(currentDisplay) +
            '</p>' +
            (timeline ? '<p class="mb-0 text-muted small">' + pbjEscapeHtml(timeline) + '</p>' : '')
        );
    }

    function pbjEventId(type, dateIso, suffix) {
        return [type, dateIso, suffix || ''].join('|').replace(/\s+/g, '_');
    }

    function pbjTrendTraceYMax(traces) {
        var ymax = 0;
        (traces || []).forEach(function (t) {
            if (
                !t ||
                t.visible === 'legendonly' ||
                t.name === '_pbjEventsHover' ||
                t.name === '_pbjEventsHoverLabels' ||
                t.name === '_pbjEventsHoverAnnotations'
            ) {
                return;
            }
            (t.y || []).forEach(function (v) {
                var n = typeof v === 'number' ? v : parseFloat(String(v));
                if (Number.isFinite(n) && n > ymax) {
                    ymax = n;
                }
            });
        });
        return ymax;
    }

    function pbjEventHoverText(ev) {
        var meta = EVENT_TYPES[ev.type] || EVENT_TYPES.manual_incident;
        var dateLabel = ev.date_iso;
        if (typeof global.pbjV2FormatIsoShort === 'function') {
            dateLabel = global.pbjV2FormatIsoShort(ev.date_iso);
        } else if (typeof global.pbjFormatIsoWorkDateUsDashed === 'function') {
            dateLabel = global.pbjFormatIsoWorkDateUsDashed(ev.date_iso);
        }
        if (ev.type === 'manual_incident') {
            var manualLabel = String(ev.label || ev.title || '').trim();
            var manualNote = String(ev.detail || '').trim();
            var manualLines = [
                '<span class="pbj-event-hover-title">' + pbjEscapeHtml(dateLabel) + '</span>',
                '<span class="pbj-event-hover-body">' + pbjEscapeHtml('Event') + '</span>'
            ];
            if (manualLabel) {
                manualLines.push('<span class="pbj-event-hover-detail">' + pbjEscapeHtml(manualLabel) + '</span>');
            } else if (manualNote) {
                manualLines.push('<span class="pbj-event-hover-detail">' + pbjEscapeHtml(manualNote) + '</span>');
            }
            manualLines.push('<span class="pbj-event-hover-source">Source: User</span>');
            return manualLines.join('<br>');
        }
        var parts = [meta.label];
        if (ev.type === 'manual_incident' && ev.label) {
            parts.push(ev.label);
        } else if (ev.type === 'chow' && ev.detail) {
            parts.push(ev.detail);
        } else if (ev.type === 'citation_g_plus') {
            if (ev.label && ev.label !== meta.label && String(ev.label).indexOf('G+') < 0) {
                parts.push(String(ev.label).replace(/\(Provider Info\)/i, '').trim());
            }
        } else if (ev.type !== 'chow' && ev.label && ev.label !== meta.label) {
            parts.push(ev.label.replace(/\(Provider Info\)/i, '').trim());
        }
        if (ev.change_tag) {
            parts.push(ev.change_tag);
        } else if (ev.detail && ev.type !== 'chow' && ev.type !== 'manual_incident' && ev.type !== 'citation_g_plus') {
            var d = String(ev.detail).replace(/^Quarter:\s*/i, '');
            if (d) {
                parts.push(d);
            }
        }
        var lines = [
            '<span class="pbj-event-hover-title">' + pbjEscapeHtml(dateLabel) + '</span>',
            '<span class="pbj-event-hover-body">' + pbjEscapeHtml(parts.filter(Boolean).join(' · ')) + '</span>'
        ];
        if (ev.detail && (ev.type === 'citation_g_plus' || ev.type === 'manual_incident')) {
            var detailLine = String(ev.detail).replace(/^Quarter:\s*/i, '').trim();
            var bodyText = parts.filter(Boolean).join(' · ');
            var repeatsDate =
                !detailLine ||
                detailLine === dateLabel ||
                detailLine === String(ev.date_iso || '').trim() ||
                bodyText.indexOf(detailLine) >= 0 ||
                (dateLabel && detailLine.indexOf(dateLabel) >= 0);
            if (!repeatsDate) {
                lines.push('<span class="pbj-event-hover-detail">' + pbjEscapeHtml(detailLine) + '</span>');
            }
        }
        var src = pbjFormatEventSourceText(ev);
        if (src) {
            lines.push('<span class="pbj-event-hover-source">Source: ' + pbjEscapeHtml(src) + '</span>');
        }
        return lines.join('<br>');
    }

    function pbjEventShortChartLabel(ev) {
        var meta = EVENT_TYPES[ev.type] || EVENT_TYPES.manual_incident;
        if (ev.type === 'manual_incident') {
            var t = String(ev.label || ev.title || 'Event').trim();
            if (!t) {
                return 'Event';
            }
            return t.length > 16 ? t.slice(0, 15) + '…' : t;
        }
        return meta.shortLabel || meta.label;
    }

    function pbjFacilityEventsVisibleOnChart(traces) {
        var reg = global.__pbjFacilityEventsRegistry || [];
        var out = [];
        reg.forEach(function (ev) {
            if (!pbjFacilityEventsTypeEnabled(ev.type)) {
                return;
            }
            var xv = pbjEventXValForTraces(ev, traces);
            if (xv == null || xv === '') {
                return;
            }
            out.push({ ev: ev, x: xv });
        });
        return out;
    }

    function pbjCurrentChartScopeLabel() {
        if (typeof global.getCurrentAnalysisRangeLabel === 'function') {
            return String(global.getCurrentAnalysisRangeLabel() || '').trim();
        }
        if (global.lastPbjFilterCleanSubtitle) {
            return String(global.lastPbjFilterCleanSubtitle || '').trim();
        }
        return '';
    }

    function pbjFacilityEventsCountByType() {
        var counts = {};
        TYPE_ORDER.forEach(function (t) {
            counts[t] = 0;
        });
        (global.__pbjFacilityEventsRegistry || []).forEach(function (ev) {
            if (counts[ev.type] != null) {
                counts[ev.type] += 1;
            }
        });
        return counts;
    }

    function pbjDaysApart(isoA, isoB) {
        var a = new Date(String(isoA) + 'T12:00:00').getTime();
        var b = new Date(String(isoB) + 'T12:00:00').getTime();
        if (!isFinite(a) || !isFinite(b)) {
            return Infinity;
        }
        return Math.abs(a - b) / 86400000;
    }

    function pbjFacilityEventsDedupeChowSameDate(list) {
        var seen = {};
        return list.filter(function (ev) {
            if (ev.type !== 'chow') {
                return true;
            }
            var k = String(ev.date_iso || '');
            if (!k || seen[k]) {
                return false;
            }
            seen[k] = true;
            return true;
        });
    }

    function pbjFacilityEventsDisplayRegistry() {
        var reg = global.__pbjFacilityEventsRegistry || [];
        var out = [];
        var chowDates = {};
        var adminKept = 0;
        reg.forEach(function (ev) {
            if (!ev) {
                return;
            }
            if (ev.type === 'chow') {
                var cKey = String(ev.date_iso || '');
                if (!cKey || chowDates[cKey]) {
                    return;
                }
                chowDates[cKey] = true;
            }
            if (ev.type === 'admin_turnover') {
                if (adminKept >= 5) {
                    return;
                }
                adminKept += 1;
            }
            out.push(ev);
        });
        return out;
    }

    function pbjFacilityEventsDedupeOwnershipNearChow(list) {
        var chowDates = list.filter(function (ev) {
            return ev.type === 'chow';
        }).map(function (ev) {
            return ev.date_iso;
        });
        if (!chowDates.length) {
            return list;
        }
        return list.filter(function (ev) {
            if (ev.type !== 'ownership_provider') {
                return true;
            }
            for (var i = 0; i < chowDates.length; i++) {
                if (pbjDaysApart(ev.date_iso, chowDates[i]) <= 120) {
                    return false;
                }
            }
            return true;
        });
    }

    function pbjFacilityEventsRebuildRegistry() {
        var list = [];
        var seen = {};

        function push(ev) {
            if (!ev || !ev.date_iso || !ev.type) {
                return;
            }
            var key = ev.id || pbjEventId(ev.type, ev.date_iso, ev.label);
            if (seen[key]) {
                return;
            }
            seen[key] = true;
            list.push(ev);
        }

        pbjFacilityEventsLoadManual().forEach(function (row) {
            var iso = String(row.date_iso || row.date || '').trim().slice(0, 10);
            if (!pbjIsoOk(iso)) {
                return;
            }
            push({
                id: row.id || pbjEventId('manual_incident', iso, row.title),
                type: 'manual_incident',
                ccn: pbjEventsCcn(),
                date_iso: iso,
                label: String(row.title || 'Resident incident').trim() || 'Resident incident',
                detail: String(row.note || '').trim(),
                source: 'User',
                link_action: null
            });
        });

        var chow = global.__pbjLastChowPayload;
        var txs = chow && Array.isArray(chow.transactions) ? chow.transactions : [];
        txs.forEach(function (t, i) {
            var iso = String(t.effective_date || '').trim().slice(0, 10);
            if (!pbjIsoOk(iso)) {
                return;
            }
            push({
                id: pbjEventId('chow', iso, String(i)),
                type: 'chow',
                date_iso: iso,
                label: 'CHOW',
                detail: pbjFormatChowEventDetail(t),
                change_tag: pbjHumanizeChowTag(t.change_summary),
                source: 'CMS CHOW index',
                link_action: 'open_chow'
            });
        });

        var hist = Array.isArray(global.__lastRedFlagHistory) ? global.__lastRedFlagHistory : [];
        var histSorted = hist.slice().sort(function (a, b) {
            var qa = String((a && a.quarter) || '');
            var qb = String((b && b.quarter) || '');
            if (qa && qb) {
                return qa.localeCompare(qb);
            }
            return String((a && a.processing_date) || '').localeCompare(String((b && b.processing_date) || ''));
        });
        var prevAdminToCount = null;
        var adminToQuartersFromFlat = {};
        histSorted.forEach(function (item) {
            var flags = Array.isArray(item.red_flags) ? item.red_flags : [];
            var quarter = item.quarter || '';
            var qMid = pbjQuarterToMidIso(quarter);
            flags.forEach(function (flag) {
                var f = String(flag || '');
                if (/ownership/i.test(f)) {
                    var oIso = pbjIsoOk(item.processing_date) ? String(item.processing_date).slice(0, 10) : qMid;
                    if (oIso) {
                        push({
                            id: pbjEventId('ownership_provider', oIso, quarter),
                            type: 'ownership_provider',
                            date_iso: oIso,
                            label: 'Ownership change',
                            detail: quarter ? 'Quarter: ' + quarter : '',
                            source: item.source_label || item.source_file || 'CMS nursing home data',
                            link_action: 'scroll_red_flags'
                        });
                    }
                }
                if (/G\+\s*citation|G-or-higher deficiency/i.test(f)) {
                    var cIso = pbjIsoOk(item.citation_survey_min_iso)
                        ? String(item.citation_survey_min_iso).slice(0, 10)
                        : qMid;
                    if (cIso) {
                        push({
                            id: pbjEventId('citation_g_plus', cIso, quarter),
                            type: 'citation_g_plus',
                            date_iso: cIso,
                            label: f.replace(/G-or-higher deficiency citations/gi, 'G+ citations'),
                            detail: (item.citation_survey_dates_tooltip || '').trim(),
                            source: item.citation_source_label || item.citation_source_file || 'CMS Deficiencies',
                            link_action: 'scroll_red_flags'
                        });
                    }
                }
                if (/\bSFF\b/i.test(f) || /Special Focus/i.test(f)) {
                    var sIso = pbjIsoOk(item.processing_date)
                        ? String(item.processing_date).slice(0, 10)
                        : qMid;
                    if (sIso) {
                        push({
                            id: pbjEventId('sff_status', sIso, quarter + f),
                            type: 'sff_status',
                            date_iso: sIso,
                            label: f.indexOf('Candidate') >= 0 ? 'SFF Candidate' : 'SFF',
                            detail: quarter ? 'Quarter: ' + quarter : '',
                            source: item.source_label || item.source_file || 'CMS nursing home data',
                            link_action: 'scroll_red_flags'
                        });
                    }
                }
                if (/Abuse/i.test(f)) {
                    var aIso = pbjIsoOk(item.processing_date)
                        ? String(item.processing_date).slice(0, 10)
                        : qMid;
                    if (aIso) {
                        push({
                            id: pbjEventId('abuse_event', aIso, quarter),
                            type: 'abuse_event',
                            date_iso: aIso,
                            label: 'Abuse icon',
                            detail: quarter ? 'Quarter: ' + quarter : '',
                            source: item.source_label || item.source_file || 'CMS nursing home data',
                            link_action: 'scroll_red_flags'
                        });
                    }
                }
                if (/Admin TO/i.test(f)) {
                    var tIso = pbjIsoOk(item.processing_date)
                        ? String(item.processing_date).slice(0, 10)
                        : qMid;
                    var adminMatch = f.match(/Admin TO:\s*(\d+)/i);
                    var adminCount = adminMatch ? parseInt(adminMatch[1], 10) : null;
                    if (tIso && adminCount != null && adminCount !== prevAdminToCount) {
                        push({
                            id: pbjEventId('admin_turnover', tIso, quarter + f),
                            type: 'admin_turnover',
                            date_iso: tIso,
                            label: 'Admin turnover',
                            detail: f + (quarter ? ' · ' + quarter : ''),
                            source: item.source_label || item.source_file || 'CMS nursing home data',
                            link_action: 'scroll_red_flags'
                        });
                        prevAdminToCount = adminCount;
                        if (quarter) {
                            adminToQuartersFromFlat[String(quarter)] = true;
                        }
                    }
                }
            });
        });

        var reviews = Array.isArray(global.__pbjAdminTurnoverReviews) ? global.__pbjAdminTurnoverReviews : [];
        reviews.forEach(function (ep, idx) {
            var iso = String(ep.first_observed_date || ep.last_observed_date || '').trim().slice(0, 10);
            if (!pbjIsoOk(iso)) {
                return;
            }
            var quarters = ep.affected_quarters || [];
            var missingFromFlat = quarters.some(function (q) {
                return q && !adminToQuartersFromFlat[String(q)];
            });
            if (!missingFromFlat) {
                return;
            }
            var count = ep.turnover_count;
            var flag = 'Admin TO';
            if (count != null && !isNaN(parseInt(count, 10)) && parseInt(count, 10) > 0) {
                flag = 'Admin TO: ' + parseInt(count, 10);
            }
            push({
                id: pbjEventId('admin_turnover_review', iso, String(ep.episode_id || idx)),
                type: 'admin_turnover',
                date_iso: iso,
                label: 'Admin turnover',
                detail: flag + (quarters.length ? ' · ' + quarters.join(', ') : ''),
                source: 'CMS Provider Information',
                link_action: 'scroll_red_flags'
            });
        });

        var prevNm = global.__pbjProviderPreviousNameSignificant
            ? String(global.__pbjProviderPreviousNameSignificant).trim()
            : '';
        var prevNmLastQ = global.__pbjProviderPreviousNameQuarter
            ? String(global.__pbjProviderPreviousNameQuarter).trim()
            : '';
        if (prevNm) {
            var eventQ = global.__pbjProviderPreviousNameEventQuarter
                ? String(global.__pbjProviderPreviousNameEventQuarter).trim()
                : '';
            var nIso = global.__pbjProviderPreviousNameEventDateIso
                ? String(global.__pbjProviderPreviousNameEventDateIso).trim()
                : '';
            if (!nIso && eventQ) {
                nIso = pbjQuarterToMidIso(eventQ);
            }
            if (!nIso && prevNmLastQ && /^\d{4}-\d{2}-\d{2}$/.test(prevNmLastQ.slice(0, 10))) {
                nIso = prevNmLastQ.slice(0, 10);
            }
            if (nIso) {
                var srcLabel = global.__pbjProviderPreviousNameSourceLabel
                    ? String(global.__pbjProviderPreviousNameSourceLabel).trim()
                    : 'CMS Provider Information';
                var srcUrl = global.__pbjProviderPreviousNameCmsUrl
                    ? String(global.__pbjProviderPreviousNameCmsUrl).trim()
                    : '';
                var priorQDisp = pbjFormatEventQuarterLabel(prevNmLastQ);
                var eventQDisp = pbjFormatEventQuarterLabel(eventQ);
                var detailParts = [pbjSmartDisplayCase(prevNm)];
                if (priorQDisp && eventQDisp) {
                    detailParts.push(priorQDisp + ' → ' + eventQDisp);
                } else if (eventQDisp) {
                    detailParts.push(eventQDisp);
                }
                push({
                    id: pbjEventId('name_change', nIso, prevNm),
                    type: 'name_change',
                    date_iso: nIso,
                    label: 'Name change',
                    detail: detailParts.filter(Boolean).join(' · '),
                    source: pbjNormalizeEventSourceLabel(srcLabel),
                    source_url: srcUrl,
                    link_action: 'scroll_red_flags'
                });
            }
        }

        list.sort(function (a, b) {
            if (a.type === 'manual_incident' && b.type !== 'manual_incident') {
                return -1;
            }
            if (b.type === 'manual_incident' && a.type !== 'manual_incident') {
                return 1;
            }
            return String(b.date_iso).localeCompare(String(a.date_iso));
        });
        list = pbjFacilityEventsDedupeOwnershipNearChow(list);
        list = pbjFacilityEventsDedupeChowSameDate(list);
        global.__pbjFacilityEventsRegistry = list;
        return list;
    }

    function pbjCyQuarterFromIso(iso) {
        var d = new Date(String(iso) + 'T12:00:00');
        if (!isFinite(d.getTime())) {
            return null;
        }
        var y = d.getFullYear();
        var q = Math.floor(d.getMonth() / 3) + 1;
        return { cy: y + 'Q' + q, label: 'Q' + q + ' ' + y };
    }

    function pbjTrendTraceXs(traces) {
        for (var ti = 0; ti < (traces || []).length; ti++) {
            var tr = traces[ti];
            if (!tr || !tr.x || !tr.x.length) {
                continue;
            }
            if (tr.name === '_pbjEventsHover' || tr.name === '_pbjEventsHoverLabels' || tr.name === '_pbjEventsHoverAnnotations') {
                continue;
            }
            return tr.x;
        }
        return traces && traces[0] && traces[0].x;
    }

    function pbjTrendXOnChart(traces, xv) {
        var xs = pbjTrendTraceXs(traces);
        if (!xs || xv == null || xv === '') {
            return false;
        }
        for (var i = 0; i < xs.length; i++) {
            if (xs[i] === xv || String(xs[i]) === String(xv)) {
                return true;
            }
        }
        return false;
    }

    function pbjParseTrendXLabelToMs(x) {
        if (x == null || x === '') {
            return NaN;
        }
        if (typeof x === 'number' && isFinite(x)) {
            return x;
        }
        var s = String(x).trim();
        if (/^\d{4}-\d{2}-\d{2}/.test(s)) {
            var d0 = new Date(s.slice(0, 10) + 'T12:00:00').getTime();
            return isFinite(d0) ? d0 : NaN;
        }
        var qm = s.match(/^Q([1-4])\s+(\d{4})$/i);
        if (qm) {
            var qMid = pbjQuarterToMidIso('Q' + qm[1] + ' ' + qm[2]);
            if (qMid) {
                return new Date(qMid + 'T12:00:00').getTime();
            }
        }
        if (/^\d{4}$/.test(s)) {
            return new Date(s + '-07-01T12:00:00').getTime();
        }
        var p = Date.parse(s);
        return isFinite(p) ? p : NaN;
    }

    function pbjFirstTraceXSample(traces) {
        var xs = pbjTrendTraceXs(traces);
        if (!xs || !xs.length) {
            return null;
        }
        for (var i = 0; i < xs.length; i++) {
            if (xs[i] != null && xs[i] !== '') {
                return xs[i];
            }
        }
        return xs[0];
    }

    /** Map event ISO date to the chart trace x value (daily / monthly / quarterly / annual). */
    function pbjDateIsoToTrendXVal(iso, traces) {
        var xs = pbjTrendTraceXs(traces);
        if (!xs || !xs.length || !iso) {
            return null;
        }
        var sample = pbjFirstTraceXSample(traces);
        if (sample == null) {
            return null;
        }
        var sampleStr = String(sample).trim();
        var d = new Date(String(iso).slice(0, 10) + 'T12:00:00');
        if (!isFinite(d.getTime())) {
            return null;
        }

        if (/^Q[1-4]\s+\d{4}$/.test(sampleStr)) {
            var qm = pbjCyQuarterFromIso(iso);
            if (!qm || xs.indexOf(qm.label) < 0) {
                return null;
            }
            return qm.label;
        }
        if (/^\d{4}$/.test(sampleStr)) {
            var y = String(d.getFullYear());
            if (xs.indexOf(y) >= 0) {
                return y;
            }
            var yn = Number(y);
            return xs.indexOf(yn) >= 0 ? yn : null;
        }
        if (/^[A-Za-z]{3}\s+\d{4}$/.test(sampleStr)) {
            var mon = d.toLocaleString('en-US', { month: 'short', year: 'numeric' });
            return xs.indexOf(mon) >= 0 ? mon : null;
        }
        if (/^\d{2}-\d{2}-\d{4}$/.test(sampleStr)) {
            var mm = String(d.getMonth() + 1).padStart(2, '0');
            var dd = String(d.getDate()).padStart(2, '0');
            var fmt = mm + '-' + dd + '-' + d.getFullYear();
            return xs.indexOf(fmt) >= 0 ? fmt : null;
        }
        if (/^\d{4}-\d{2}-\d{2}/.test(sampleStr)) {
            var isoOut = String(iso).slice(0, 10);
            return xs.indexOf(isoOut) >= 0 ? isoOut : null;
        }
        if (typeof sample === 'number') {
            var ms = d.getTime();
            if (xs.indexOf(ms) >= 0) {
                return ms;
            }
            for (var ni = 0; ni < xs.length; ni++) {
                if (Number(xs[ni]) === ms) {
                    return xs[ni];
                }
            }
            return null;
        }
        return null;
    }

    /** Snap to nearest chart bucket when exact label is missing (events / citations). */
    function pbjDateIsoToTrendXValNearest(iso, traces) {
        var exact = pbjDateIsoToTrendXVal(iso, traces);
        if (exact != null && pbjTrendXOnChart(traces, exact)) {
            return exact;
        }
        var xs = pbjTrendTraceXs(traces);
        if (!xs || !iso) {
            return null;
        }
        var target = new Date(String(iso).slice(0, 10) + 'T12:00:00').getTime();
        if (!isFinite(target)) {
            return null;
        }
        var best = null;
        var bestDist = Infinity;
        xs.forEach(function (xv) {
            var t = pbjParseTrendXLabelToMs(xv);
            if (!isFinite(t)) {
                return;
            }
            var dist = Math.abs(t - target);
            if (dist < bestDist) {
                bestDist = dist;
                best = xv;
            }
        });
        return best;
    }

    function pbjEventDateWithinTraceSpan(iso, traces) {
        var xs = pbjTrendTraceXs(traces);
        if (!xs || !xs.length || !iso) {
            return false;
        }
        var target = new Date(String(iso).slice(0, 10) + 'T12:00:00').getTime();
        if (!isFinite(target)) {
            return false;
        }
        var times = [];
        xs.forEach(function (xv) {
            var t = pbjParseTrendXLabelToMs(xv);
            if (isFinite(t)) {
                times.push(t);
            }
        });
        if (!times.length) {
            return false;
        }
        var minT = Math.min.apply(null, times);
        var maxT = Math.max.apply(null, times);
        var slackMs = 46 * 86400000;
        return target >= minT - slackMs && target <= maxT + slackMs;
    }

    function pbjTrendXGrain(traces) {
        var sample = pbjFirstTraceXSample(traces);
        if (sample == null) {
            return 'unknown';
        }
        var s = String(sample).trim();
        if (/^Q[1-4]\s+\d{4}$/i.test(s)) {
            return 'quarter';
        }
        if (/^\d{4}$/.test(s)) {
            return 'year';
        }
        if (/^[A-Za-z]{3}\s+\d{4}$/.test(s)) {
            return 'month';
        }
        return 'daily';
    }

    function pbjEventNearestXVal(ev, traces) {
        var nearest = pbjDateIsoToTrendXValNearest(ev.date_iso, traces);
        if (nearest == null || nearest === '' || !pbjTrendXOnChart(traces, nearest)) {
            return null;
        }
        var grain = pbjTrendXGrain(traces);
        if (grain === 'quarter' || grain === 'month' || grain === 'year') {
            return nearest;
        }
        var target = new Date(String(ev.date_iso).slice(0, 10) + 'T12:00:00').getTime();
        var nearMs = pbjParseTrendXLabelToMs(nearest);
        if (isFinite(target) && isFinite(nearMs) && Math.abs(nearMs - target) <= 21 * 86400000) {
            return nearest;
        }
        return null;
    }

    function pbjEventXValForTraces(ev, traces) {
        var xs = pbjTrendTraceXs(traces);
        if (!xs || !xs.length) {
            return null;
        }
        if (!pbjEventDateWithinTraceSpan(ev.date_iso, traces)) {
            return null;
        }
        var exact = pbjDateIsoToTrendXVal(ev.date_iso, traces);
        if (exact != null && exact !== '' && pbjTrendXOnChart(traces, exact)) {
            return exact;
        }
        return pbjEventNearestXVal(ev, traces);
    }

    function pbjFacilityEventsShapeForEvent(ev, traces, opts) {
        opts = opts || {};
        if (!pbjFacilityEventsTypeEnabled(ev.type)) {
            return null;
        }
        var xv = pbjEventXValForTraces(ev, traces);
        if (xv == null || xv === '' || !pbjTrendXOnChart(traces, xv)) {
            return null;
        }
        var st = EVENT_TYPES[ev.type] || EVENT_TYPES.manual_incident;
        var subtle = !!opts.subtleMarkers;
        return {
            type: 'line',
            x0: xv,
            x1: xv,
            y0: 0,
            y1: 1,
            xref: 'x',
            yref: 'paper',
            layer: subtle ? 'below' : 'above',
            line: {
                color: subtle ? 'rgba(100, 116, 139, 0.45)' : st.color,
                width: subtle ? 1 : 1.5,
                dash: st.dash
            }
        };
    }

    function pbjAppendFacilityEventShapes(layout, traces, opts) {
        opts = opts || {};
        if (!layout || !pbjFacilityEventsMarkersActive()) {
            return;
        }
        var skipHoverTraces = !!opts.skipHoverTraces;
        var subtleMarkers = !!opts.subtleMarkers;
        var reg = global.__pbjFacilityEventsRegistry;
        if (!reg || !reg.length) {
            pbjFacilityEventsRebuildRegistry();
            reg = global.__pbjFacilityEventsRegistry || [];
        }
        var shapes = [];
        var visibleOnChart = pbjFacilityEventsVisibleOnChart(traces);
        visibleOnChart.forEach(function (item) {
            var sh = pbjFacilityEventsShapeForEvent(item.ev, traces, opts);
            if (sh) {
                shapes.push(sh);
            }
        });
        if (shapes.length) {
            layout.shapes = ([]).concat(layout.shapes || [], shapes);
        }

        if (subtleMarkers) {
            skipHoverTraces = false;
        }

        var annotations = [];
        var labelColors = {
            manual_incident: '#9a3412',
            chow: '#5b21b6',
            citation_g_plus: '#991b1b',
            ownership_provider: '#475569'
        };
        var byX = {};
        visibleOnChart.forEach(function (item) {
            var xKey = String(item.x);
            if (!byX[xKey]) {
                byX[xKey] = [];
            }
            byX[xKey].push(item);
        });
        var maxStack = 1;
        if (!subtleMarkers) {
            Object.keys(byX).forEach(function (xKey) {
                var group = byX[xKey];
                if (group.length > maxStack) {
                    maxStack = group.length;
                }
                group.forEach(function (item, idx) {
                    var ev = item.ev;
                    var st = EVENT_TYPES[ev.type] || EVENT_TYPES.manual_incident;
                    annotations.push({
                        x: item.x,
                        y: 1.01 + idx * 0.065,
                        xref: 'x',
                        yref: 'paper',
                        text: pbjEventShortChartLabel(ev),
                        showarrow: false,
                        font: { size: 9, color: labelColors[ev.type] || '#334155' },
                        xanchor: 'center',
                        yanchor: 'bottom',
                        bgcolor: 'rgba(255,255,255,0.92)',
                        bordercolor: st.color,
                        borderwidth: 1,
                        borderpad: 2,
                        pbjFacilityEvent: true
                    });
                });
            });
        }
        if (annotations.length) {
            layout.annotations = ([]).concat(layout.annotations || [], annotations);
            layout.margin = layout.margin || {};
            var minTop = 52 + Math.max(1, maxStack) * 14;
            layout.margin.t = Math.max(Number(layout.margin.t || 0), minTop);
        }

        if (Array.isArray(traces) && visibleOnChart.length && !skipHoverTraces) {
            for (var ti = traces.length - 1; ti >= 0; ti--) {
                if (
                    traces[ti] &&
                    (traces[ti].name === '_pbjEventsHover' ||
                        traces[ti].name === '_pbjEventsHoverLabels' ||
                        traces[ti].name === '_pbjEventsHoverAnnotations')
                ) {
                    traces.splice(ti, 1);
                }
            }
            var ymax = pbjTrendTraceYMax(traces);
            if (!ymax || ymax <= 0) {
                ymax = 1;
            }
            var lineHoverX = [];
            var lineHoverY = [];
            var lineHoverText = [];
            var labelHoverX = [];
            var labelHoverY = [];
            var labelHoverText = [];
            var steps = 14;
            visibleOnChart.forEach(function (item) {
                var ht = pbjEventHoverText(item.ev);
                var si;
                for (si = 0; si <= steps; si++) {
                    lineHoverX.push(item.x);
                    lineHoverY.push((ymax * si) / steps);
                    lineHoverText.push(ht);
                }
                labelHoverX.push(item.x);
                labelHoverY.push(1.02);
                labelHoverText.push(ht);
            });
            // showlegend:false hover helpers — do not set legendgroup (Plotly stacks legend vertically).
            traces.push(
                {
                    type: 'scatter',
                    name: '_pbjEventsHoverLabels',
                    x: labelHoverX,
                    y: labelHoverY,
                    xref: 'x',
                    yref: 'paper',
                    mode: 'markers',
                    marker: {
                        size: 44,
                        color: 'rgba(255,255,255,0)',
                        line: { width: 0 }
                    },
                    hovertemplate: '%{text}<extra></extra>',
                    text: labelHoverText,
                    hoverlabel: {
                        align: 'left',
                        bgcolor: '#ffffff',
                        bordercolor: '#cbd5e1',
                        font: { size: 12, color: '#0f172a' }
                    },
                    showlegend: false
                },
                {
                    type: 'scatter',
                    name: '_pbjEventsHoverAnnotations',
                    x: labelHoverX,
                    y: labelHoverY.map(function () {
                        return 1.008;
                    }),
                    xref: 'x',
                    yref: 'paper',
                    mode: 'markers',
                    marker: {
                        size: 52,
                        color: 'rgba(255,255,255,0)',
                        line: { width: 0 }
                    },
                    hovertemplate: '%{text}<extra></extra>',
                    text: labelHoverText,
                    hoverlabel: {
                        align: 'left',
                        bgcolor: '#ffffff',
                        bordercolor: '#cbd5e1',
                        font: { size: 12, color: '#0f172a' }
                    },
                    showlegend: false
                },
                {
                    type: 'scatter',
                    name: '_pbjEventsHover',
                    x: lineHoverX,
                    y: lineHoverY,
                    mode: 'markers',
                    marker: {
                        size: 28,
                        color: 'rgba(255,255,255,0)',
                        opacity: 0.01,
                        symbol: 'circle',
                        line: { width: 0 }
                    },
                    hovertemplate: '%{text}<extra></extra>',
                    text: lineHoverText,
                    hoverlabel: {
                        align: 'left',
                        bgcolor: '#ffffff',
                        bordercolor: '#cbd5e1',
                        font: { size: 12, color: '#0f172a' }
                    },
                    showlegend: false
                }
            );
        }
        layout.hovermode = 'closest';
        layout.hoverdistance = Math.max(Number(layout.hoverdistance || 0), 28);
    }

    var PBJ_EVENT_HOVER_TRACE_NAMES = [
        '_pbjEventsHover',
        '_pbjEventsHoverLabels',
        '_pbjEventsHoverAnnotations'
    ];

    /** Legacy no-op — event hover traces are included in newPlot data (no deferred addTraces). */
    function pbjFacilityEventsApplyDeferredHoverTraces() {
        return Promise.resolve();
    }

    function pbjFacilityEventsRefreshCharts() {
        if (typeof global.updateCharts === 'function' && global.lastChartsResponse) {
            try {
                global.updateCharts(global.lastChartsResponse, global.lastTotalDays);
            } catch (e6) { /* ignore */ }
        }
        if (typeof global.pbjFacilityEventsRepaintEinHeadcount === 'function') {
            try {
                global.pbjFacilityEventsRepaintEinHeadcount();
            } catch (e7) { /* ignore */ }
        }
        if (typeof global.pbjFacilityEventsRepaintProviderCaseMix === 'function') {
            try {
                global.pbjFacilityEventsRepaintProviderCaseMix();
            } catch (e8) { /* ignore */ }
        }
    }

    function pbjFacilityEventsScheduleRefreshCharts() {
        if (typeof global.updateCharts !== 'function' || !global.lastChartsResponse) {
            return;
        }
        clearTimeout(global.__pbjFacilityEventsRefreshTimer);
        global.__pbjFacilityEventsRefreshTimer = setTimeout(function () {
            global.__pbjFacilityEventsRefreshTimer = null;
            pbjFacilityEventsRefreshCharts();
        }, 120);
    }

    function pbjFacilityEventsCompactChipMeta(ev) {
        var meta = EVENT_TYPES[ev.type];
        var typeShort = meta ? (meta.shortLabel || meta.label) : ev.type;
        var dateLabel = ev.date_iso;
        if (typeof global.pbjV2FormatIsoShort === 'function') {
            dateLabel = global.pbjV2FormatIsoShort(ev.date_iso);
        }
        var label = String(ev.label || ev.title || '').replace(/\(Provider Info\)/i, '').trim();
        if (ev.type === 'manual_incident') {
            if (label) {
                var words = label.split(/\s+/);
                typeShort = words.length > 1 && label.length > 16 ? label.slice(0, 15) + '…' : label;
            } else {
                typeShort = 'Event';
            }
        } else if (ev.type === 'chow') {
            label = 'CHOW';
        }
        return {
            typeShort: typeShort,
            dateLabel: dateLabel,
            label: label,
            css: meta ? meta.css : ''
        };
    }

    function pbjFacilityEventsOpenTimelineDetail(evId) {
        var reg = global.__pbjFacilityEventsRegistry || [];
        var ev = null;
        for (var i = 0; i < reg.length; i++) {
            if (reg[i].id === evId) {
                ev = reg[i];
                break;
            }
        }
        if (!ev) {
            return;
        }
        if (ev.link_action === 'open_chow') {
            pbjFacilityEventsShowChowModal();
            return;
        }
        var chip = pbjFacilityEventsCompactChipMeta(ev);
        var titleEl = document.getElementById('pbjEventsTimelineDetailModalLabel');
        var dateEl = document.getElementById('pbjEventsTimelineDetailDate');
        var headEl = document.getElementById('pbjEventsTimelineDetailTitle');
        var bodyEl = document.getElementById('pbjEventsTimelineDetailBody');
        var sourceEl = document.getElementById('pbjEventsTimelineDetailSource');
        var actionsEl = document.getElementById('pbjEventsTimelineDetailActions');
        if (titleEl) {
            titleEl.textContent = chip.label || chip.typeShort;
        }
        if (dateEl) {
            dateEl.textContent = chip.dateLabel;
        }
        if (headEl) {
            headEl.textContent = chip.typeShort;
        }
        if (bodyEl) {
            if (ev.type === 'name_change') {
                bodyEl.innerHTML = pbjNameChangeEventBodyHtml();
                if (sourceEl) {
                    sourceEl.innerHTML = pbjFormatEventSourceHtml(ev);
                }
            } else {
                var detail = ev.detail ? String(ev.detail).replace(/^Quarter:\s*/i, '') : '';
                if (ev.type === 'citation_g_plus' && detail.indexOf('Survey date:') === 0) {
                    detail = detail.replace(/^Survey date:\s*/i, '');
                }
                bodyEl.textContent = detail || (EVENT_TYPES[ev.type] ? EVENT_TYPES[ev.type].desc : '');
                if (sourceEl) {
                    sourceEl.innerHTML = pbjFormatEventSourceHtml(ev);
                }
            }
        }
        if (actionsEl) {
            actionsEl.innerHTML = '';
            if (ev.link_action === 'scroll_red_flags') {
                actionsEl.innerHTML =
                    '<button type="button" class="btn btn-sm btn-outline-secondary" data-pbj-timeline-jump="red_flags">Flag history</button>';
            }
        }
        var modalEl = document.getElementById('pbjEventsTimelineDetailModal');
        if (modalEl && typeof bootstrap !== 'undefined') {
            bootstrap.Modal.getOrCreateInstance(modalEl).show();
        }
    }

    function pbjFacilityEventsRenderCompactTimeline() {
        var host = document.getElementById('pbjEventsCompactTimeline');
        if (!host) {
            return;
        }
        var reg = pbjFacilityEventsDisplayRegistry();
        if (!reg.length) {
            host.innerHTML = '<p class="small text-muted mb-0">No suggested events yet.</p>';
            return;
        }
        var maxShow = 14;
        var slice = reg.slice().sort(function (a, b) {
            return String(a.date_iso).localeCompare(String(b.date_iso));
        }).slice(0, maxShow);
        var chips = slice.map(function (ev) {
            var c = pbjFacilityEventsCompactChipMeta(ev);
            var evIdEsc = encodeURIComponent(ev.id);
            return (
                '<button type="button" class="pbj-events-timeline-chip ' + c.css + '" data-pbj-timeline-ev="' + evIdEsc + '" ' +
                'title="' + pbjEscapeHtml(c.label) + '">' +
                '<span class="pbj-events-timeline-chip-type">' + pbjEscapeHtml(c.typeShort) + '</span>' +
                '<span class="pbj-events-timeline-chip-sep" aria-hidden="true">·</span>' +
                '<span class="pbj-events-timeline-chip-date">' + pbjEscapeHtml(c.dateLabel) + '</span>' +
                '</button>'
            );
        }).join('');
        var fullReg = global.__pbjFacilityEventsRegistry || [];
        var more =
            fullReg.length > maxShow
                ? '<button type="button" class="btn btn-sm btn-link p-0 text-decoration-none flex-shrink-0" data-bs-toggle="modal" data-bs-target="#pbjFacilityEventsModal">+' +
                  (fullReg.length - maxShow) +
                  ' more</button>'
                : '';
        host.innerHTML = '<div class="pbj-events-timeline-track">' + chips + more + '</div>';
    }

    function pbjFacilityEventsOnSourcesUpdated() {
        pbjFacilityEventsRebuildRegistry();
        pbjFacilityEventsRenderHierarchy();
        pbjFacilityEventsRenderCompactTimeline();
        pbjFacilityEventsSyncUi();
        var eventsSec = document.getElementById('pbjCitationsSection');
        if (eventsSec && (global.__pbjFacilityEventsRegistry || []).length) {
            if (!global.__pbjV3PanesActive) {
                eventsSec.classList.remove('d-none');
            } else if (typeof global.pbjV3ReapplyPaneVisibility === 'function') {
                global.pbjV3ReapplyPaneVisibility();
            }
        }
        if (global.__pbjV3PanesActive && typeof global.pbjV3RiskTimelineRefresh === 'function') {
            global.pbjV3RiskTimelineRefresh();
        }
        if (pbjFacilityEventsMasterOn() && global.lastChartsResponse) {
            pbjFacilityEventsScheduleRefreshCharts();
        }
        if (typeof global.prePostRefreshSuggestedEventChips === 'function') {
            global.prePostRefreshSuggestedEventChips();
        }
    }

    function pbjFacilityEventsMasterLabelText() {
        if (!pbjFacilityEventsMasterOn()) {
            return 'Events off';
        }
        var traces = global.__pbjLastTrendChartTraces;
        var scope = pbjCurrentChartScopeLabel();
        var enabled = (global.__pbjFacilityEventsRegistry || []).filter(function (ev) {
            return pbjFacilityEventsTypeEnabled(ev.type);
        }).length;
        if (traces) {
            var visible = pbjFacilityEventsVisibleOnChart(traces).length;
            var scopeBit = scope ? (' · chart scope: ' + scope) : ' · current chart view';
            if (visible === 0 && enabled > 0) {
                return 'Events on · none in this chart view' + scopeBit;
            }
            if (visible === 0) {
                return 'Events on · none loaded yet';
            }
            return visible + ' on chart' + (enabled > visible ? (' · ' + enabled + ' total enabled') : '') + scopeBit;
        }
        var counts = pbjFacilityEventsCountByType();
        var names = TYPE_ORDER.filter(function (t) {
            return (counts[t] || 0) > 0;
        }).map(function (t) {
            return EVENT_TYPES[t].shortLabel || EVENT_TYPES[t].label;
        });
        if (!names.length) {
            return 'Events on · none loaded yet';
        }
        return 'Events on · types: ' + names.join(', ');
    }

    function pbjEventTypeBadgeHtml(type) {
        var meta = EVENT_TYPES[type] || EVENT_TYPES.manual_incident;
        return '<span class="badge pbj-event-type-badge pbj-readable-badge ' + meta.css + '">' + meta.label + '</span>';
    }

    function pbjFacilityEventsExtractQuarter(ev) {
        var d = String(ev.detail || '');
        var m = d.match(/Quarter:\s*(Q[1-4]\s+\d{4})/i);
        if (m) {
            return m[1];
        }
        if (ev.date_iso && ev.date_iso.length >= 4) {
            return ev.date_iso.slice(0, 4);
        }
        return '';
    }

    function pbjFacilityEventsRenderEventRow(ev) {
        var act = '';
        if (ev.link_action === 'scroll_red_flags') {
            act = ' <button type="button" class="btn btn-sm p-0 align-baseline pbj-event-jump-btn" data-pbj-event-jump="red_flags">Flags</button>';
        } else if (ev.link_action === 'open_chow') {
            act = ' <button type="button" class="btn btn-sm p-0 align-baseline pbj-event-jump-btn" data-pbj-event-jump="chow">Details</button>';
        }
        var del = '';
        if (ev.type === 'manual_incident') {
            del = ' <button type="button" class="btn btn-link btn-sm p-0 text-danger align-baseline" data-pbj-event-delete="' +
                encodeURIComponent(ev.id) + '">Remove</button>';
        }
        var dateLabel = ev.date_iso;
        if (typeof global.pbjV2FormatIsoShort === 'function') {
            dateLabel = global.pbjV2FormatIsoShort(ev.date_iso);
        }
        var detail = ev.detail ? String(ev.detail).replace(/^Quarter:\s*/i, '') : '';
        if (ev.type === 'citation_g_plus' && detail.indexOf('Survey date:') === 0) {
            detail = detail.replace(/^Survey date:\s*/i, '');
        }
        var tagHtml = '';
        if (ev.type === 'chow' && ev.change_tag) {
            tagHtml = ' <span class="pbj-event-chow-tag">' + pbjEscapeHtml(ev.change_tag) + '</span>';
        }
        var labelText = ev.label.replace(/\(Provider Info\)/i, '').trim();
        if (ev.type === 'chow') {
            var buyerSeller = (detail || '').trim();
            var sellerMatch = buyerSeller.match(/^(.+?)\s*←\s*(.+)$/);
            var buyerLine = sellerMatch ? sellerMatch[1].trim() : buyerSeller;
            var sellerLine = sellerMatch ? sellerMatch[2].trim() : '';
            return (
                '<li class="pbj-facility-events-row pbj-facility-events-row--chow">' +
                '<span class="pbj-chow-events-head">' +
                '<span class="text-body fw-semibold">' +
                pbjEscapeHtml(dateLabel) +
                '</span>' +
                tagHtml +
                '</span>' +
                '<span class="pbj-chow-events-entities">' +
                '<span class="text-body fw-semibold">' +
                pbjEscapeHtml(buyerLine || buyerSeller) +
                '</span>' +
                (sellerLine
                    ? '<span class="pbj-event-detail-muted">← ' + pbjEscapeHtml(sellerLine) + '</span>'
                    : '') +
                '</span>' +
                act +
                del +
                '</li>'
            );
        }
        if (ev.type === 'manual_incident') {
            return '<li class="pbj-facility-events-row">' +
                '<span class="text-body fw-semibold">' + pbjEscapeHtml(dateLabel) + '</span>' +
                '<span class="text-body">' + pbjEscapeHtml(labelText) + '</span>' +
                (detail ? '<span class="pbj-event-detail-muted">' + pbjEscapeHtml(detail) + '</span>' : '') +
                del + '</li>';
        }
        return '<li class="pbj-facility-events-row">' + pbjEventTypeBadgeHtml(ev.type) +
            '<span class="text-body fw-semibold">' + pbjEscapeHtml(dateLabel) + '</span>' +
            '<span class="text-body">' + pbjEscapeHtml(labelText) + '</span>' +
            (detail ? '<span class="pbj-event-detail-muted">' + pbjEscapeHtml(detail) + '</span>' : '') +
            act + del + '</li>';
    }

    function pbjFacilityEventsOwnershipGroupedHtml(events) {
        if (!events.length) {
            return '';
        }
        if (events.length === 1) {
            return pbjFacilityEventsRenderEventRow(events[0]);
        }
        var quarters = events.map(pbjFacilityEventsExtractQuarter).filter(Boolean);
        var years = events.map(function (ev) {
            return String(ev.date_iso || '').slice(0, 4);
        }).filter(Boolean);
        years.sort();
        var yearSpan = years.length === 1 ? years[0] : (years[0] + '–' + years[years.length - 1]);
        var label = yearSpan + ' · ' + events.length + ' ownership quarter' + (events.length === 1 ? '' : 's');
        var detail = quarters.slice(0, 4).join(', ') + (quarters.length > 4 ? '…' : '');
        return '<li class="pbj-facility-events-row">' + pbjEventTypeBadgeHtml('ownership_provider') +
            '<span class="text-body fw-semibold">' + pbjEscapeHtml(label) + '</span>' +
            (detail ? '<span class="pbj-event-detail-muted">' + pbjEscapeHtml(detail) + '</span>' : '') +
            ' <button type="button" class="btn btn-sm p-0 align-baseline pbj-event-jump-btn" data-pbj-event-jump="red_flags">Flags</button></li>';
    }

    function pbjFacilityEventsRenderHierarchy() {
        var host = document.getElementById('pbjFacilityEventsHierarchy');
        if (!host) {
            return;
        }
        var reg = global.__pbjFacilityEventsRegistry || [];
        var counts = pbjFacilityEventsCountByType();
        var masterOn = pbjFacilityEventsMasterOn();
        var html = '';

        TYPE_ORDER.forEach(function (type) {
            var meta = EVENT_TYPES[type];
            var n = counts[type] || 0;
            var id = 'pbjFacilityEventType_' + type;
            var checked = pbjFacilityEventsTypeEnabled(type) ? ' checked' : '';
            var disabled = !masterOn ? ' disabled' : '';
            var dim = !pbjFacilityEventsTypeEnabled(type) ? ' is-dim' : '';
            var typeEvents = reg.filter(function (ev) {
                return ev.type === type;
            });
            var itemsHtml = '';

            if (type === 'ownership_provider' && typeEvents.length) {
                itemsHtml = '<ul class="list-unstyled mb-0">' + pbjFacilityEventsOwnershipGroupedHtml(typeEvents) + '</ul>';
            } else if (typeEvents.length) {
                itemsHtml = '<ul class="list-unstyled mb-0">';
                typeEvents.forEach(function (ev) {
                    itemsHtml += pbjFacilityEventsRenderEventRow(ev);
                });
                itemsHtml += '</ul>';
            } else {
                itemsHtml = '<p class="pbj-event-type-empty mb-0">No ' + pbjEscapeHtml(meta.shortLabel || meta.label) + ' loaded.</p>';
            }

            html += '<div class="pbj-event-type-section' + dim + '" data-pbj-event-section="' + type + '">' +
                '<div class="pbj-event-type-section-head">' +
                '<div class="form-check form-switch mb-0">' +
                '<input class="form-check-input pbj-event-type-switch" type="checkbox" role="switch" id="' + id + '" data-pbj-event-type="' + type + '"' + checked + disabled + '>' +
                '<label class="form-check-label d-inline-flex align-items-center gap-1" for="' + id + '">' +
                '<span class="badge pbj-event-type-badge pbj-readable-badge ' + meta.css + '">' +
                (typeof global.pbjV2EventTypeDisplayLabel === 'function'
                    ? global.pbjV2EventTypeDisplayLabel(meta)
                    : meta.label) +
                '</span>' +
                '<span class="text-muted">(' + n + ')</span></label></div>' +
                '<button type="button" class="btn btn-link btn-sm p-0 pbj-event-type-help-btn" data-pbj-event-type-help="' + type + '" title="About ' + pbjEscapeHtml(meta.label) + '" aria-label="About ' + pbjEscapeHtml(meta.label) + '">' +
                '<i class="fas fa-circle-info" aria-hidden="true"></i></button></div>' +
                '<p class="pbj-event-type-section-desc mb-0">' + pbjEscapeHtml(meta.desc) + '</p>' +
                '<div class="pbj-event-type-items">' + itemsHtml + '</div></div>';
        });

        host.innerHTML = html;
        host.querySelectorAll('.pbj-event-type-switch').forEach(function (inp) {
            inp.addEventListener('change', function () {
                var t = inp.getAttribute('data-pbj-event-type');
                pbjFacilityEventsSetType(t, inp.checked);
                pbjFacilityEventsRefreshCharts();
            });
        });
    }

    function pbjFacilityEventsEnableAllTypes() {
        global.__pbjFacilityEventsTypes = global.__pbjFacilityEventsTypes || pbjEventsDefaultTypeState(false);
        TYPE_ORDER.forEach(function (t) {
            global.__pbjFacilityEventsTypes[t] = true;
        });
        pbjFacilityEventsSaveTypeState(global.__pbjFacilityEventsTypes);
    }

    function pbjFacilityEventsSyncUi() {
        var masterOn = pbjFacilityEventsMasterOn();
        var masterSw = document.getElementById('pbjFacilityEventsMasterSwitch');
        if (masterSw) {
            masterSw.checked = masterOn;
        }
        var legacy = document.getElementById('pbjFacilityEventsShowSwitch');
        if (legacy) {
            legacy.checked = masterOn;
        }
        var labelEl = document.getElementById('pbjFacilityEventsMasterLabel');
        if (labelEl) {
            labelEl.textContent = pbjFacilityEventsMasterLabelText();
        }
        document.querySelectorAll('.pbj-trend-events-switch').forEach(function (el) {
            el.checked = masterOn;
            el.setAttribute('aria-checked', masterOn ? 'true' : 'false');
        });
        document.querySelectorAll('.pbj-trend-events-panel-btn').forEach(function (btn) {
            var label = btn.querySelector('.pbj-trend-events-btn-label');
            if (label) {
                label.textContent = 'Events';
            } else if (btn.classList.contains('dropdown-item')) {
                btn.textContent = 'Events';
            }
            btn.classList.toggle('pbj-trend-events-panel-btn--active', masterOn);
            btn.setAttribute('aria-pressed', masterOn ? 'true' : 'false');
        });
        var countEl = document.getElementById('pbjFacilityEventsCount');
        var reg = global.__pbjFacilityEventsRegistry || [];
        var traces = global.__pbjLastTrendChartTraces;
        var activeN = reg.filter(function (ev) {
            return pbjFacilityEventsTypeEnabled(ev.type);
        }).length;
        var visibleN = traces ? pbjFacilityEventsVisibleOnChart(traces).length : null;
        var scope = pbjCurrentChartScopeLabel();
        if (countEl) {
            if (!reg.length) {
                countEl.textContent = 'No events loaded for this facility yet.';
            } else if (!masterOn) {
                countEl.textContent = reg.length + ' event' + (reg.length === 1 ? '' : 's') + ' available · hidden on charts';
            } else if (visibleN != null) {
                if (visibleN === 0 && activeN > 0) {
                    countEl.textContent =
                        '0 visible in current chart view' +
                        (scope ? ' (' + scope + ')' : '') +
                        ' · ' +
                        activeN +
                        ' enabled overall';
                } else {
                    countEl.textContent =
                        visibleN +
                        ' visible in chart view' +
                        (scope ? ' (' + scope + ')' : '') +
                        (activeN > visibleN ? ' · ' + activeN + ' enabled overall' : '');
                }
            } else {
                countEl.textContent = activeN + ' of ' + reg.length + ' shown on trend charts';
            }
        }
        pbjFacilityEventsRenderHierarchy();
    }

    function pbjFacilityEventsShowTypeHelp(type) {
        var meta = EVENT_TYPES[type];
        if (!meta) {
            return;
        }
        var titleEl = document.getElementById('pbjFacilityEventTypeHelpModalLabel');
        var bodyEl = document.getElementById('pbjFacilityEventTypeHelpBody');
        if (titleEl) {
            titleEl.textContent = meta.label;
        }
        if (bodyEl) {
            bodyEl.textContent = meta.help || meta.desc;
        }
        var m = document.getElementById('pbjFacilityEventTypeHelpModal');
        if (m && typeof bootstrap !== 'undefined') {
            bootstrap.Modal.getOrCreateInstance(m).show();
        }
    }

    function pbjFacilityEventsShowChowModal() {
        var m = document.getElementById('pbjChowOwnershipModal');
        if (!m || typeof bootstrap === 'undefined') {
            return;
        }
        var eventsModal = document.getElementById('pbjFacilityEventsModal');
        if (eventsModal) {
            var evInst = bootstrap.Modal.getInstance(eventsModal);
            if (evInst) {
                evInst.hide();
            }
        }
        var show = function () {
            if (typeof global.pbjV2RenderChowPanel === 'function' && global.__pbjLastChowPayload) {
                global.pbjV2RenderChowPanel(global.__pbjLastChowPayload);
            }
            bootstrap.Modal.getOrCreateInstance(m).show();
        };
        if (!global.__pbjLastChowPayload && typeof global.pbjV2LoadChowPanel === 'function') {
            var loadP = global.pbjV2LoadChowPanel(
                typeof global.pbjDashboardFacilityCcn === 'function'
                    ? global.pbjDashboardFacilityCcn()
                    : global.PROVNUM || global.PBJ320_EXPORT_CCN
            );
            (loadP && typeof loadP.then === 'function' ? loadP : Promise.resolve()).then(show);
            return;
        }
        show();
    }

    function pbjFacilityEventsJump(action) {
        if (action === 'chow') {
            pbjFacilityEventsShowChowModal();
            return;
        }
        if (action === 'red_flags') {
            if (typeof global.pbjV2RevealInspectionsSection === 'function') {
                global.pbjV2RevealInspectionsSection({ openFlags: true, scrollTarget: 'riskScreeningSection' });
            } else {
                var sec = document.getElementById('riskScreeningSection');
                if (sec) {
                    sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
        }
    }

    function pbjFacilityEventsAddManualRow(iso, title, note) {
        iso = String(iso || '').trim();
        title = String(title || '').trim();
        note = String(note || '').trim();
        if (!pbjIsoOk(iso) || !title) {
            return false;
        }
        var rows = pbjFacilityEventsLoadManual();
        rows.unshift({
            id: 'manual_' + Date.now(),
            ccn: pbjEventsCcn(),
            date_iso: iso,
            title: title,
            note: note
        });
        pbjFacilityEventsSaveManual(rows);
        if (!pbjFacilityEventsMasterOn()) {
            pbjFacilityEventsEnableAllTypes();
            pbjFacilityEventsSetMaster(true);
        } else {
            pbjFacilityEventsSetType('manual_incident', true);
        }
        pbjFacilityEventsOnSourcesUpdated();
        pbjFacilityEventsRefreshCharts();
        return true;
    }

    function pbjFacilityEventsAddManual() {
        var dateEl = document.getElementById('pbjFacilityEventDate');
        var titleEl = document.getElementById('pbjFacilityEventTitle');
        var noteEl = document.getElementById('pbjFacilityEventNote');
        var iso = dateEl ? String(dateEl.value || '').trim() : '';
        var title = titleEl ? String(titleEl.value || '').trim() : '';
        var note = noteEl ? String(noteEl.value || '').trim() : '';
        if (!pbjFacilityEventsAddManualRow(iso, title, note)) {
            return;
        }
        if (titleEl) {
            titleEl.value = '';
        }
        if (noteEl) {
            noteEl.value = '';
        }
    }

    function pbjFacilityEventsAddManualFromFields(iso, title, note) {
        return pbjFacilityEventsAddManualRow(iso, title, note);
    }

    function pbjFacilityEventsDeleteManual(encodedId) {
        var id = decodeURIComponent(encodedId || '');
        var rows = pbjFacilityEventsLoadManual().filter(function (r) {
            return String(r.id) !== id && pbjEventId('manual_incident', r.date_iso, r.title) !== id;
        });
        pbjFacilityEventsSaveManual(rows);
        pbjFacilityEventsOnSourcesUpdated();
        pbjFacilityEventsRefreshCharts();
    }

    function pbjFacilityEventsEditManualInScope(eventId) {
        var id = String(eventId || '').trim();
        if (!id) {
            return false;
        }
        var rows = pbjFacilityEventsLoadManual();
        var row = rows.find(function (r) {
            return String(r.id) === id || pbjEventId('manual_incident', r.date_iso, r.title) === id;
        });
        if (!row) {
            return false;
        }
        var dateEl = document.getElementById('pbjChartScopeEventDate');
        var noteEl = document.getElementById('pbjChartScopeEventNote');
        if (dateEl) {
            dateEl.value = String(row.date_iso || '').slice(0, 10);
        }
        if (noteEl) {
            noteEl.value = String(row.title || row.note || '').trim();
            noteEl.focus();
        }
        pbjFacilityEventsDeleteManual(encodeURIComponent(id));
        return true;
    }

    function pbjFacilityEventsOpenModal() {
        var m = document.getElementById('pbjFacilityEventsModal');
        if (m && typeof bootstrap !== 'undefined') {
            bootstrap.Modal.getOrCreateInstance(m).show();
        }
    }

    function pbjFacilityEventsToggleFromChip() {
        var now = pbjFacilityEventsMasterOn();
        if (!now) {
            pbjFacilityEventsEnableAllTypes();
        }
        pbjFacilityEventsSetMaster(!now);
        pbjFacilityEventsRefreshCharts();
    }

    function pbjFacilityEventsInit() {
        pbjFacilityEventsMigrateLegacyStorage();
        try {
            var hasMaster = localStorage.getItem(pbjEventsMasterKey()) !== null;
            global.__pbjFacilityEventsMaster = hasMaster
                ? localStorage.getItem(pbjEventsMasterKey()) === '1'
                : true;
        } catch (e7) {
            global.__pbjFacilityEventsMaster = true;
        }
        global.__pbjFacilityEventsTypes = pbjFacilityEventsLoadTypeState();
        if (global.__pbjFacilityEventsMaster) {
            var anyOn = TYPE_ORDER.some(function (t) {
                return global.__pbjFacilityEventsTypes[t];
            });
            if (!anyOn) {
                pbjFacilityEventsEnableAllTypes();
            }
        }
        pbjFacilityEventsRebuildRegistry();
        pbjFacilityEventsRenderHierarchy();
        pbjFacilityEventsRenderCompactTimeline();
        pbjFacilityEventsSyncUi();

        document.getElementById('pbjFacilityEventsMasterSwitch')?.addEventListener('change', function (e) {
            if (e.target.checked) {
                pbjFacilityEventsEnableAllTypes();
            }
            pbjFacilityEventsSetMaster(!!e.target.checked);
            pbjFacilityEventsRefreshCharts();
        });
        document.addEventListener('change', function (ev) {
            var sw = ev.target && ev.target.closest ? ev.target.closest('.pbj-trend-events-switch') : null;
            if (sw) {
                if (sw.checked && !pbjFacilityEventsMasterOn()) {
                    pbjFacilityEventsEnableAllTypes();
                }
                pbjFacilityEventsSetMaster(!!sw.checked);
                pbjFacilityEventsRefreshCharts();
            }
        });
        document.addEventListener('click', function (ev) {
            var helpBtn = ev.target.closest('[data-pbj-event-type-help]');
            if (helpBtn) {
                ev.preventDefault();
                pbjFacilityEventsShowTypeHelp(helpBtn.getAttribute('data-pbj-event-type-help'));
                return;
            }
        });
        document.getElementById('pbjFacilityEventAddBtn')?.addEventListener('click', pbjFacilityEventsAddManual);
        document.getElementById('pbjFacilityEventsHierarchy')?.addEventListener('click', function (e) {
            var t = e.target.closest('[data-pbj-event-jump], [data-pbj-event-delete]');
            if (!t) {
                return;
            }
            var jump = t.getAttribute('data-pbj-event-jump');
            if (jump) {
                pbjFacilityEventsJump(jump);
                return;
            }
            var del = t.getAttribute('data-pbj-event-delete');
            if (del) {
                pbjFacilityEventsDeleteManual(del);
            }
        });
        document.getElementById('pbjEventsCompactTimeline')?.addEventListener('click', function (e) {
            var chip = e.target.closest('[data-pbj-timeline-ev]');
            if (!chip) {
                return;
            }
            e.preventDefault();
            pbjFacilityEventsOpenTimelineDetail(decodeURIComponent(chip.getAttribute('data-pbj-timeline-ev') || ''));
        });
        document.getElementById('pbjEventsTimelineDetailActions')?.addEventListener('click', function (e) {
            var jumpBtn = e.target.closest('[data-pbj-timeline-jump]');
            if (!jumpBtn) {
                return;
            }
            var jump = jumpBtn.getAttribute('data-pbj-timeline-jump');
            var detailModal = document.getElementById('pbjEventsTimelineDetailModal');
            if (detailModal && typeof bootstrap !== 'undefined') {
                bootstrap.Modal.getOrCreateInstance(detailModal).hide();
            }
            if (jump) {
                pbjFacilityEventsJump(jump);
            }
        });
        var eventsModal = document.getElementById('pbjFacilityEventsModal');
        if (eventsModal) {
            eventsModal.addEventListener('show.bs.modal', function () {
                pbjFacilityEventsOnSourcesUpdated();
            });
        }
        if (global.__pbjFacilityEventsMaster) {
            pbjFacilityEventsRefreshCharts();
        }
    }

    global.pbjEventsCcn = pbjEventsCcn;
    global.pbjFacilityEventsVisible = pbjFacilityEventsVisible;
    global.pbjFacilityEventsMarkersActive = pbjFacilityEventsMarkersActive;
    global.pbjFacilityEventsSetMaster = pbjFacilityEventsSetMaster;
    global.pbjFacilityEventsSetVisible = pbjFacilityEventsSetMaster;
    global.pbjFacilityEventsRebuildRegistry = pbjFacilityEventsRebuildRegistry;
    global.pbjFacilityEventsOnSourcesUpdated = pbjFacilityEventsOnSourcesUpdated;
    global.pbjAppendFacilityEventShapes = pbjAppendFacilityEventShapes;
    global.pbjFacilityEventsApplyDeferredHoverTraces = pbjFacilityEventsApplyDeferredHoverTraces;
    global.pbjDateIsoToTrendXVal = pbjDateIsoToTrendXVal;
    global.pbjTrendXOnChart = pbjTrendXOnChart;
    global.pbjFacilityEventsRefreshCharts = pbjFacilityEventsRefreshCharts;
    global.pbjFacilityEventsMaybeRefreshCharts = pbjFacilityEventsRefreshCharts;
    global.pbjFacilityEventsOpenModal = pbjFacilityEventsOpenModal;
    global.pbjFacilityEventsAddManualFromFields = pbjFacilityEventsAddManualFromFields;
    global.pbjFacilityEventsDeleteManual = pbjFacilityEventsDeleteManual;
    global.pbjFacilityEventsEditManualInScope = pbjFacilityEventsEditManualInScope;
    global.pbjFacilityEventsSyncUi = pbjFacilityEventsSyncUi;
    global.pbjFacilityEventsTypeEnabled = pbjFacilityEventsTypeEnabled;
    global.pbjFacilityEventsSetType = pbjFacilityEventsSetType;
    global.pbjFacilityEventsRenderCompactTimeline = pbjFacilityEventsRenderCompactTimeline;
    global.pbjFacilityEventsCompactChipMeta = pbjFacilityEventsCompactChipMeta;
    global.pbjFacilityEventsDisplayRegistry = pbjFacilityEventsDisplayRegistry;
    global.pbjSmartDisplayCase = pbjSmartDisplayCase;
    global.pbjHumanizeChowTag = pbjHumanizeChowTag;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', pbjFacilityEventsInit);
    } else {
        pbjFacilityEventsInit();
    }
})(typeof window !== 'undefined' ? window : globalThis);
