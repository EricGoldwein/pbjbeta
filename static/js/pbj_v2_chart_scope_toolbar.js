/**
 * v2 — Per-chart toolbar: date range panel + events panel (synced to Control Center).
 */
(function (global) {
    'use strict';

    var SCOPE_MENU_RANGE = 'range';
    var SCOPE_MENU_EVENTS = 'events';

    var EVENT_LIST_MAX = 24;
    var EVENT_LIST_SCROLL_AT = 8;
    var SCOPE_MENU_MIN_W = 272;
    var SCOPE_MENU_MAX_W_RATIO = 0.92;
    var SCOPE_MENU_MIN_H = 220;
    var SCOPE_MENU_MAX_H_RATIO = 0.72;
    var EVENT_TYPE_PRIORITY = {
        manual_incident: 0,
        chow: 1,
        citation_g_plus: 2,
        sff_status: 3,
        ownership_provider: 4,
        admin_turnover: 5,
        name_change: 6,
        abuse_event: 7
    };

    function $(id) {
        return document.getElementById(id);
    }

    function qs(sel, root) {
        return (root || document).querySelector(sel);
    }

    function qsa(sel, root) {
        return Array.prototype.slice.call((root || document).querySelectorAll(sel));
    }

    function isVisible(el) {
        return !!(el && (el.offsetParent !== null || (el.getClientRects && el.getClientRects().length)));
    }

    function readMultiSelectValues(sel) {
        if (!sel) {
            return 'all';
        }
        var vals = Array.from(sel.selectedOptions || []).map(function (o) { return o.value; });
        if (!vals.length || vals.indexOf('all') >= 0) {
            return 'all';
        }
        return vals.join(',');
    }

    function syncMultiSelectValues(sel, csv) {
        if (!sel) {
            return;
        }
        var want = (!csv || csv === 'all') ? ['all'] : String(csv).split(',');
        Array.from(sel.options || []).forEach(function (opt) {
            opt.selected = want.indexOf(opt.value) >= 0;
        });
    }

    function readControlCenterState() {
        var filterType = (qs('input[name="filterType"]:checked') || {}).value || 'quarters';
        var state = {
            mode: 'custom',
            filterType: filterType,
            quarter: 'all',
            year: 'all',
            startDate: '',
            endDate: '',
            startMonth: '',
            endMonth: '',
            dayDate: ''
        };
        if (filterType === 'daterange') {
            state.startDate = ($('startDate') || {}).value || '';
            state.endDate = ($('endDate') || {}).value || '';
        } else if (filterType === 'months') {
            state.startMonth = ($('startMonth') || {}).value || '';
            state.endMonth = ($('endMonth') || {}).value || '';
        } else if (filterType === 'quarters') {
            var qSel = $('quarterRange');
            if (isVisible(qSel)) {
                state.quarter = readMultiSelectValues(qSel);
            } else {
                qSel = $('quarterRangeMobileCompact');
                state.quarter = qSel && qSel.value ? qSel.value : 'all';
            }
        } else if (filterType === 'years') {
            var ySel = $('years');
            if (isVisible(ySel)) {
                state.year = readMultiSelectValues(ySel);
            } else {
                ySel = $('yearsMobileCompact');
                state.year = ySel && ySel.value ? ySel.value : 'all';
            }
        } else if (filterType === 'day') {
            state.dayDate = ($('filterDayDate') || {}).value || '';
        }
        return state;
    }

    function cloneSelectOptions(fromSel, toSel) {
        if (!fromSel || !toSel) {
            return;
        }
        toSel.innerHTML = '';
        Array.from(fromSel.options || []).forEach(function (opt) {
            var o = document.createElement('option');
            o.value = opt.value;
            o.textContent = opt.textContent;
            toSel.appendChild(o);
        });
    }

    function populateScopeSelects() {
        cloneSelectOptions($('quarterRange') || $('quarterRangeMobileCompact'), $('pbjChartScopeQuarterSelect'));
        cloneSelectOptions($('years') || $('yearsMobileCompact'), $('pbjChartScopeYearSelect'));
    }

    function scopeStateFromControlCenter() {
        return readControlCenterState();
    }

    function scopeMenuEl(kind) {
        if (kind === SCOPE_MENU_EVENTS) {
            return $('pbjChartEventsMenu');
        }
        return $('pbjChartRangeMenu');
    }

    function applyProfileConstraints(profile) {
        var menu = scopeMenuEl(SCOPE_MENU_RANGE);
        if (!menu) {
            return;
        }
        var hideDay = profile === 'headcount' || profile === 'casemix';
        var casemixProfile = profile === 'casemix';
        var dayRadio = $('pbjChartScopeGrainDay');
        var dayLabel = qs('.pbj-chart-scope-grain-day-label', menu);
        if (dayRadio) {
            dayRadio.disabled = hideDay;
        }
        if (dayLabel) {
            dayLabel.classList.toggle('d-none', hideDay);
        }
        var restrictedGrainIds = [
            ['pbjChartScopeGrainMonths', 'label[for="pbjChartScopeGrainMonths"]'],
            ['pbjChartScopeGrainRange', 'label[for="pbjChartScopeGrainRange"]']
        ];
        restrictedGrainIds.forEach(function (pair) {
            var radio = $(pair[0]);
            var label = qs(pair[1], menu);
            if (radio) {
                radio.disabled = casemixProfile;
            }
            if (label) {
                label.classList.toggle('d-none', casemixProfile);
            }
        });
        var yearsRadio = $('pbjChartScopeGrainYears');
        var yearsLabel = qs('label[for="pbjChartScopeGrainYears"]', menu);
        if (yearsRadio) {
            yearsRadio.disabled = false;
        }
        if (yearsLabel) {
            yearsLabel.classList.remove('d-none');
        }
        if (hideDay && dayRadio && dayRadio.checked) {
            var qRadio = $('pbjChartScopeGrainQuarters');
            if (qRadio) {
                qRadio.checked = true;
            }
            showScopePicker('quarters');
        }
        if (casemixProfile) {
            var activeGrain = (qs('input[name="pbjChartScopeGrain"]:checked') || {}).value || 'quarters';
            if (activeGrain === 'months' || activeGrain === 'daterange' || activeGrain === 'day') {
                var qOnly = $('pbjChartScopeGrainQuarters');
                if (qOnly) {
                    qOnly.checked = true;
                }
                showScopePicker('quarters');
            }
        }
    }

    function ensureScopeMonthDefaults() {
        var start = $('pbjChartScopeStartMonth');
        var end = $('pbjChartScopeEndMonth');
        if (!start || !end) {
            return;
        }
        if (start.value && end.value) {
            return;
        }
        var cc = readControlCenterState();
        if (cc.filterType === 'months' && cc.startMonth && cc.endMonth) {
            if (!start.value) {
                start.value = cc.startMonth;
            }
            if (!end.value) {
                end.value = cc.endMonth;
            }
            return;
        }
        var bounds = global.__pbjLastDataDateRange || global.__pbjFacilityDateBounds || {};
        var maxIso = bounds.max || bounds.max_date || '';
        var minIso = bounds.min || bounds.min_date || '';
        var maxM = maxIso ? String(maxIso).slice(0, 7) : '';
        var minM = minIso ? String(minIso).slice(0, 7) : '';
        if (maxM && !end.value) {
            end.value = maxM;
        }
        if (!start.value) {
            if (minM && maxM && minM <= maxM) {
                start.value = minM;
            } else if (maxM) {
                var parts = maxM.split('-');
                var y = parseInt(parts[0], 10);
                var mo = parseInt(parts[1], 10);
                mo -= 11;
                while (mo <= 0) {
                    mo += 12;
                    y -= 1;
                }
                start.value = y + '-' + String(mo).padStart(2, '0');
            }
        }
    }

    function ensureChartEventsVisible() {
        if (typeof global.pbjFacilityEventsSetMaster === 'function') {
            global.pbjFacilityEventsSetMaster(true);
        }
        qsa('.pbj-trend-events-switch').forEach(function (sw) {
            sw.checked = true;
            sw.setAttribute('aria-checked', 'true');
        });
    }

    function showScopePicker(grain) {
        var menu = scopeMenuEl(SCOPE_MENU_RANGE);
        if (!menu) {
            return;
        }
        qsa('.pbj-chart-scope-picker', menu).forEach(function (el) {
            el.classList.add('d-none');
        });
        var key = grain === 'daterange' ? 'daterange' : grain;
        var panel = qs('.pbj-chart-scope-picker--' + key, menu);
        if (panel) {
            panel.classList.remove('d-none');
        }
        if (grain === 'months') {
            ensureScopeMonthDefaults();
        }
    }

    function syncMenuFromState(state) {
        var st = state || global.__pbjChartScopeState || scopeStateFromControlCenter();
        var ft = st.filterType || 'quarters';
        var grainRadio = qs('input[name="pbjChartScopeGrain"][value="' + ft + '"]');
        if (grainRadio) {
            grainRadio.checked = true;
        }
        showScopePicker(ft);
        syncMultiSelectValues($('pbjChartScopeQuarterSelect'), st.quarter || 'all');
        syncMultiSelectValues($('pbjChartScopeYearSelect'), st.year || 'all');
        if ($('pbjChartScopeStartMonth')) {
            $('pbjChartScopeStartMonth').value = st.startMonth || '';
        }
        if ($('pbjChartScopeEndMonth')) {
            $('pbjChartScopeEndMonth').value = st.endMonth || '';
        }
        if ($('pbjChartScopeStartDate')) {
            $('pbjChartScopeStartDate').value = st.startDate || '';
        }
        if ($('pbjChartScopeEndDate')) {
            $('pbjChartScopeEndDate').value = st.endDate || '';
        }
        if ($('pbjChartScopeDayDate')) {
            $('pbjChartScopeDayDate').value = st.dayDate || '';
        }
        if (ft === 'months') {
            ensureScopeMonthDefaults();
        }
    }

    function readMenuCustomState() {
        var ft = (qs('input[name="pbjChartScopeGrain"]:checked') || {}).value || 'quarters';
        var state = {
            mode: 'custom',
            filterType: ft,
            quarter: 'all',
            year: 'all',
            startDate: '',
            endDate: '',
            startMonth: '',
            endMonth: '',
            dayDate: ''
        };
        if (ft === 'quarters') {
            state.quarter = readMultiSelectValues($('pbjChartScopeQuarterSelect'));
        } else if (ft === 'years') {
            state.year = readMultiSelectValues($('pbjChartScopeYearSelect'));
        } else if (ft === 'months') {
            state.startMonth = ($('pbjChartScopeStartMonth') || {}).value || '';
            state.endMonth = ($('pbjChartScopeEndMonth') || {}).value || '';
        } else if (ft === 'daterange') {
            state.startDate = ($('pbjChartScopeStartDate') || {}).value || '';
            state.endDate = ($('pbjChartScopeEndDate') || {}).value || '';
        } else if (ft === 'day') {
            state.dayDate = ($('pbjChartScopeDayDate') || {}).value || '';
        }
        return state;
    }

    function quarterToDateRange(qKey) {
        var m = String(qKey || '').match(/^(\d{4})Q([1-4])$/i);
        if (!m) {
            return null;
        }
        var y = parseInt(m[1], 10);
        var q = parseInt(m[2], 10);
        var startMonth = (q - 1) * 3 + 1;
        var endMonth = startMonth + 2;
        var endDay = new Date(y, endMonth, 0).getDate();
        return {
            start: y + '-' + String(startMonth).padStart(2, '0') + '-01',
            end: y + '-' + String(endMonth).padStart(2, '0') + '-' + String(endDay).padStart(2, '0')
        };
    }

    function scopePreviewHasDefinedRange(state) {
        if (!state || state.mode === 'all') {
            return false;
        }
        var ft = state.filterType || 'quarters';
        if (ft === 'quarters') {
            return !!(state.quarter && state.quarter !== 'all');
        }
        if (ft === 'years') {
            return !!(state.year && state.year !== 'all');
        }
        if (ft === 'months') {
            return !!(state.startMonth && state.endMonth);
        }
        if (ft === 'daterange') {
            return !!(state.startDate && state.endDate);
        }
        if (ft === 'day') {
            return !!state.dayDate;
        }
        return false;
    }

    function resolveChartTitleFromAnchor(anchor) {
        if (!anchor) {
            return 'Chart';
        }
        var card = anchor.closest('.card, .pbj-trend-chart-card, section');
        var parts = [];
        if (card) {
            var titleEl = card.querySelector(
                '.pbj-chart-toolbar-title, #pbjHprdChartHeading, .card-header h5, .card-header .pbj-chart-toolbar-title'
            );
            if (titleEl) {
                parts.push(String(titleEl.textContent || '').trim());
            }
            var activeTab = card.querySelector(
                '.pbj-segment-tabs .nav-link.active, .pbj-ui-tabs .nav-link.active, .pbj-composition-segment-tabs .nav-link.active'
            );
            if (activeTab) {
                var tabLabel = String(activeTab.textContent || '').trim();
                var joined = parts.join(' ').toLowerCase();
                if (tabLabel && joined.indexOf(tabLabel.toLowerCase()) < 0) {
                    parts.push(tabLabel);
                }
            }
        }
        if (!parts.length) {
            var sec = anchor.closest('section[id]');
            var heading = sec && sec.querySelector('.dashboard-section-title, h2');
            if (heading) {
                parts.push(String(heading.textContent || '').replace(/^\d+\.\s*/, '').trim());
            }
        }
        return parts.filter(Boolean).join(' · ') || 'Chart';
    }

    function updateScopeMenuSubtitle(anchor) {
        var title = resolveChartTitleFromAnchor(anchor || global.__pbjChartScopeAnchor);
        var rangeEl = $('pbjChartRangeMenuSubtitle');
        var eventsEl = $('pbjChartEventsMenuSubtitle');
        if (rangeEl) {
            rangeEl.textContent = title;
        }
        if (eventsEl) {
            eventsEl.textContent = title;
        }
    }

    function showScopeAddFeedback(iso, label) {
        showScopeMenuMessage('Added: ' + formatEventDateUsShort(iso) + ' · ' + String(label || 'Event').trim(), 'success');
    }

    function showScopeMenuMessage(text, tone) {
        var host = $('pbjChartScopeEventsList');
        if (!host || !host.parentNode) {
            return;
        }
        var fb = $('pbjChartScopeAddFeedback');
        if (!fb) {
            fb = document.createElement('p');
            fb.id = 'pbjChartScopeAddFeedback';
            fb.className = 'small mb-1';
            host.parentNode.insertBefore(fb, host);
        }
        fb.className = 'small mb-1 ' + (tone === 'error' ? 'text-danger' : 'text-success');
        fb.textContent = String(text || '');
        window.clearTimeout(fb._pbjHideTimer);
        if (tone !== 'error') {
            fb._pbjHideTimer = window.setTimeout(function () {
                if (fb) {
                    fb.textContent = '';
                }
            }, 4500);
        }
    }

    function scopeMenuBasePosition(anchor, kind) {
        var menu = scopeMenuEl(kind || global.__pbjChartScopeMenuKind || SCOPE_MENU_RANGE);
        if (!menu || !anchor) {
            return { left: 0, top: 0 };
        }
        applyScopeMenuSize(menu);
        var rect = anchor.getBoundingClientRect();
        var menuW = menu.offsetWidth || 336;
        var baseLeft = Math.min(rect.left, window.innerWidth - menuW - 8);
        var baseTop = rect.bottom + 6;
        if (baseTop + menu.offsetHeight > window.innerHeight - 8 && rect.top - menu.offsetHeight - 6 > 8) {
            baseTop = rect.top - menu.offsetHeight - 6;
        }
        return { left: baseLeft, top: baseTop };
    }

    function scopeStateToIsoRange(state) {
        if (!state || state.mode === 'all') {
            var bounds = global.__pbjFacilityDateBounds || global.__pbjLastDataDateRange || {};
            return {
                start: bounds.min_date || bounds.min || '2017-01-01',
                end: bounds.max_date || bounds.max || '2099-12-31'
            };
        }
        var st = state.mode === 'custom' ? state : readMenuCustomState();
        var ft = st.filterType || 'quarters';
        if (ft === 'daterange' && st.startDate && st.endDate) {
            return { start: st.startDate, end: st.endDate };
        }
        if (ft === 'day' && st.dayDate) {
            return { start: st.dayDate, end: st.dayDate };
        }
        if (ft === 'months') {
            var start = st.startMonth ? st.startMonth + '-01' : '';
            var end = '';
            if (st.endMonth) {
                var p = st.endMonth.split('-');
                var ld = new Date(parseInt(p[0], 10), parseInt(p[1], 10), 0).getDate();
                end = st.endMonth + '-' + String(ld).padStart(2, '0');
            }
            return { start: start || '2017-01-01', end: end || '2099-12-31' };
        }
        if (ft === 'years') {
            if (!st.year || st.year === 'all') {
                return scopeStateToIsoRange({ mode: 'all' });
            }
            var yrs = String(st.year).split(',').map(function (y) { return parseInt(y, 10); }).filter(Boolean);
            if (!yrs.length) {
                return scopeStateToIsoRange({ mode: 'all' });
            }
            yrs.sort(function (a, b) { return a - b; });
            return { start: yrs[0] + '-01-01', end: yrs[yrs.length - 1] + '-12-31' };
        }
        if (ft === 'quarters') {
            if (!st.quarter || st.quarter === 'all') {
                return scopeStateToIsoRange({ mode: 'all' });
            }
            var qs_ = String(st.quarter).split(',');
            var minStart = null;
            var maxEnd = null;
            qs_.forEach(function (q) {
                var r = quarterToDateRange(q);
                if (!r) {
                    return;
                }
                if (!minStart || r.start < minStart) {
                    minStart = r.start;
                }
                if (!maxEnd || r.end > maxEnd) {
                    maxEnd = r.end;
                }
            });
            if (minStart && maxEnd) {
                return { start: minStart, end: maxEnd };
            }
        }
        var dr = global.__pbjLastDataDateRange;
        if (dr && dr.min && dr.max) {
            return { start: dr.min, end: dr.max };
        }
        return { start: '2017-01-01', end: '2099-12-31' };
    }

    function formatEventDateUsShort(iso) {
        var p = String(iso || '').trim().split('-');
        if (p.length !== 3) {
            return iso || '';
        }
        return p[1] + '/' + p[2] + '/' + p[0].slice(-2);
    }

    function eventTypeShort(ev) {
        if (typeof global.pbjFacilityEventsCompactChipMeta === 'function') {
            return global.pbjFacilityEventsCompactChipMeta(ev).typeShort;
        }
        if (ev.type === 'chow') {
            return 'CHOW';
        }
        if (ev.type === 'citation_g_plus') {
            return 'G+';
        }
        if (ev.type === 'manual_incident') {
            return String(ev.label || ev.title || 'Incident').trim().slice(0, 18) || 'Incident';
        }
        return ev.type || 'Event';
    }

    function eventsInScopeRange(state) {
        var range = scopeStateToIsoRange(state);
        var reg = global.__pbjFacilityEventsRegistry || [];
        var ccn = '';
        if (typeof global.pbjEventsCcn === 'function') {
            ccn = global.pbjEventsCcn();
        } else if (global.PROVNUM) {
            ccn = String(global.PROVNUM).replace(/\D/g, '').padStart(6, '0').slice(-6);
        }
        var inRange = reg.filter(function (ev) {
            if (!ev || !ev.date_iso) {
                return false;
            }
            if (ccn && ev.ccn) {
                var evCcn = String(ev.ccn).replace(/\D/g, '').padStart(6, '0').slice(-6);
                if (evCcn && evCcn !== ccn) {
                    return false;
                }
            }
            if (typeof global.pbjFacilityEventsTypeEnabled === 'function' && !global.pbjFacilityEventsTypeEnabled(ev.type)) {
                return false;
            }
            var d = String(ev.date_iso).slice(0, 10);
            return d >= range.start && d <= range.end;
        });
        inRange.sort(function (a, b) {
            var pa = EVENT_TYPE_PRIORITY[a.type];
            var pb = EVENT_TYPE_PRIORITY[b.type];
            if (pa !== pb) {
                return (pa == null ? 99 : pa) - (pb == null ? 99 : pb);
            }
            return String(b.date_iso).localeCompare(String(a.date_iso));
        });
        return { items: inRange, range: range };
    }

    function scopeMenuInner(kind) {
        var menu = scopeMenuEl(kind || global.__pbjChartScopeMenuKind || SCOPE_MENU_RANGE);
        return menu ? qs('.pbj-chart-scope-menu-inner', menu) : null;
    }

    function scopeEventsPreviewState() {
        var rangeMenu = scopeMenuEl(SCOPE_MENU_RANGE);
        if (rangeMenu && !rangeMenu.classList.contains('d-none')) {
            return readMenuCustomState();
        }
        return global.__pbjChartScopeState;
    }

    function bindScopePreviewRefresh(menu) {
        if (!menu || menu.dataset.pbjPreviewBound === '1') {
            return;
        }
        menu.dataset.pbjPreviewBound = '1';
        function refreshPreview() {
            renderScopeEventsList(readMenuCustomState());
        }
        qsa('input[name="pbjChartScopeGrain"]', menu).forEach(function (inp) {
            inp.addEventListener('change', refreshPreview);
        });
        [
            'pbjChartScopeQuarterSelect',
            'pbjChartScopeYearSelect',
            'pbjChartScopeStartMonth',
            'pbjChartScopeEndMonth',
            'pbjChartScopeStartDate',
            'pbjChartScopeEndDate',
            'pbjChartScopeDayDate'
        ].forEach(function (id) {
            var el = $(id);
            if (!el) {
                return;
            }
            el.addEventListener('change', refreshPreview);
            el.addEventListener('input', refreshPreview);
        });
    }

    function resetScopeMenuLayout(kind) {
        var menu = scopeMenuEl(kind || global.__pbjChartScopeMenuKind || SCOPE_MENU_RANGE);
        var inner = scopeMenuInner(kind);
        global.__pbjChartScopeDragOffset = { dx: 0, dy: 0 };
        global.__pbjChartScopeSize = null;
        if (menu) {
            menu.classList.remove('pbj-chart-scope-menu--sized');
            menu.style.width = '';
            menu.style.minWidth = '';
            menu.style.maxWidth = '';
        }
        if (inner) {
            inner.style.maxHeight = '';
            inner.style.overflowY = '';
        }
    }

    function applyScopeMenuSize(menu, kind) {
        var inner = scopeMenuInner(kind || (menu && menu.id === 'pbjChartEventsMenu' ? SCOPE_MENU_EVENTS : SCOPE_MENU_RANGE));
        var sz = global.__pbjChartScopeSize;
        if (!menu) {
            return;
        }
        if (sz && sz.w) {
            menu.classList.add('pbj-chart-scope-menu--sized');
            var w = Math.round(sz.w);
            menu.style.width = w + 'px';
            menu.style.minWidth = w + 'px';
            menu.style.maxWidth = w + 'px';
        } else {
            menu.classList.remove('pbj-chart-scope-menu--sized');
            menu.style.width = '';
            menu.style.minWidth = '';
            menu.style.maxWidth = '';
        }
        if (inner) {
            if (sz && sz.h) {
                inner.style.maxHeight = Math.round(sz.h) + 'px';
                inner.style.overflowY = 'auto';
            } else {
                inner.style.maxHeight = '';
                inner.style.overflowY = '';
            }
        }
    }

    function clampScopeMenuPosition(menu, left, top) {
        var menuW = menu.offsetWidth || 280;
        var menuH = menu.offsetHeight || 320;
        var maxLeft = Math.max(8, window.innerWidth - menuW - 8);
        var maxTop = Math.max(8, window.innerHeight - menuH - 8);
        return {
            left: Math.max(8, Math.min(left, maxLeft)),
            top: Math.max(8, Math.min(top, maxTop))
        };
    }

    function escapeHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function scopeEventChipMeta(ev) {
        if (typeof global.pbjFacilityEventsCompactChipMeta === 'function') {
            return global.pbjFacilityEventsCompactChipMeta(ev);
        }
        return {
            typeShort: eventTypeShort(ev),
            dateLabel: formatEventDateUsShort(ev.date_iso),
            label: ev.label || ev.title || '',
            css: ''
        };
    }

    function renderScopeEventsList(state) {
        var host = $('pbjChartScopeEventsList');
        if (!host) {
            return;
        }
        var previewState = state != null ? state : scopeEventsPreviewState();
        host.classList.remove('pbj-chart-scope-events-list--scroll');
        if (!scopePreviewHasDefinedRange(previewState)) {
            host.innerHTML = '<span class="small text-muted">Pick a range to preview events here.</span>';
            host.removeAttribute('title');
            return;
        }
        var pack = eventsInScopeRange(previewState);
        var items = pack.items;
        if (!items.length) {
            host.innerHTML = '<span class="small text-muted pbj-chart-scope-events-empty-msg">No saved events in this range.</span>';
            host.removeAttribute('title');
            return;
        }
        var show = items.slice(0, EVENT_LIST_MAX);
        var chips = show.map(function (ev) {
            var c = scopeEventChipMeta(ev);
            var typeOn = typeof global.pbjFacilityEventsTypeEnabled === 'function'
                ? global.pbjFacilityEventsTypeEnabled(ev.type)
                : true;
            var onClass = typeOn ? ' pbj-chart-scope-event-chip--on' : '';
            var manual = ev.type === 'manual_incident';
            var actions = '';
            if (manual && ev.id) {
                actions =
                    '<span class="pbj-chart-scope-event-actions">' +
                    '<button type="button" class="btn btn-link btn-sm p-0 pbj-chart-scope-event-edit" data-pbj-scope-event-edit="' +
                    escapeHtml(ev.id) +
                    '" title="Edit event" aria-label="Edit event"><i class="fas fa-pen fa-xs" aria-hidden="true"></i></button>' +
                    '<button type="button" class="btn btn-link btn-sm p-0 text-danger pbj-chart-scope-event-delete" data-pbj-scope-event-delete="' +
                    escapeHtml(encodeURIComponent(ev.id)) +
                    '" title="Delete event" aria-label="Delete event"><i class="fas fa-xmark fa-xs" aria-hidden="true"></i></button>' +
                    '</span>';
            }
            return (
                '<span class="pbj-chart-scope-event-row d-inline-flex align-items-center gap-1">' +
                '<button type="button" class="pbj-events-timeline-chip pbj-chart-scope-event-chip' + onClass + '" ' +
                'data-pbj-scope-event-type="' + escapeHtml(ev.type) + '" ' +
                'title="' + escapeHtml(c.label || c.typeShort) + '">' +
                '<span class="pbj-events-timeline-chip-date">' + escapeHtml(c.dateLabel) + '</span>' +
                '<span class="pbj-events-timeline-chip-sep" aria-hidden="true">·</span>' +
                '<span class="pbj-events-timeline-chip-type">' + escapeHtml(c.typeShort) + '</span>' +
                '</button>' +
                actions +
                '</span>'
            );
        }).join('');
        var more =
            items.length > EVENT_LIST_MAX
                ? '<span class="small text-muted">+' + (items.length - EVENT_LIST_MAX) + ' more</span>'
                : '';
        if (show.length > EVENT_LIST_SCROLL_AT) {
            host.classList.add('pbj-chart-scope-events-list--scroll');
        }
        host.innerHTML = '<div class="pbj-events-timeline-track">' + chips + more + '</div>';
        host.title = items.length > EVENT_LIST_MAX
            ? 'Showing ' + EVENT_LIST_MAX + ' of ' + items.length + ' events'
            : '';
    }

    function buildScopeApiPayload(state) {
        if (state.mode === 'all') {
            return { mode: 'all' };
        }
        var ft = state.filterType;
        var payload = { mode: 'custom', filterType: ft };
        if (ft === 'quarters') {
            payload.quarter = state.quarter || 'all';
        } else if (ft === 'years') {
            payload.year = state.year || 'all';
        } else if (ft === 'months') {
            if (state.startMonth) {
                payload.start_date = state.startMonth + '-01';
            }
            if (state.endMonth) {
                var parts = state.endMonth.split('-');
                var lastDay = new Date(parseInt(parts[0], 10), parseInt(parts[1], 10), 0).getDate();
                payload.end_date = state.endMonth + '-' + String(lastDay).padStart(2, '0');
            }
        } else if (ft === 'daterange') {
            payload.start_date = state.startDate;
            payload.end_date = state.endDate;
        } else if (ft === 'day') {
            var d = state.dayDate;
            if (d && typeof global.pbjIsoAddDays === 'function') {
                payload.start_date = global.pbjIsoAddDays(d, -15);
                payload.end_date = global.pbjIsoAddDays(d, 15);
            } else if (d) {
                payload.start_date = d;
                payload.end_date = d;
            }
        }
        return payload;
    }

    function activeScopeProfile() {
        var anchor = global.__pbjChartScopeAnchor;
        if (!anchor) {
            return global.__pbjChartScopeActiveProfile || 'default';
        }
        return (
            (anchor.closest('.pbj-trend-events-toolbar') || {}).getAttribute('data-pbj-chart-scope-profile') ||
            global.__pbjChartScopeActiveProfile ||
            'default'
        );
    }

    async function applyChartScope(state, opts) {
        opts = opts || {};
        var profile = opts.profile || activeScopeProfile();
        global.__pbjChartScopeState = state;

        if (profile === 'casemix') {
            global.__pbjCaseMixChartScopeState = state;
            global.__pbjCaseMixChartScopeUserEdited = state.mode !== 'all';
            if (typeof global.pbjFacilityEventsRefreshCharts === 'function') {
                global.pbjFacilityEventsRefreshCharts();
            }
            if (typeof global.pbjFacilityEventsRepaintProviderCaseMix === 'function') {
                global.pbjFacilityEventsRepaintProviderCaseMix();
            }
            renderScopeEventsList(state);
            return;
        }

        if (state.mode === 'all') {
            if (typeof global.pbjFetchChartsIndependentOfPageFilter === 'function') {
                await global.pbjFetchChartsIndependentOfPageFilter(null);
            }
        } else if (typeof global.pbjFetchChartsWithScope === 'function') {
            var payload = buildScopeApiPayload(state);
            var sameAsCc = JSON.stringify(state) === JSON.stringify(scopeStateFromControlCenter());
            if (sameAsCc && typeof global.pbjApplyFilteredChartsFromSnapshot === 'function') {
                global.pbjApplyFilteredChartsFromSnapshot();
            } else {
                await global.pbjFetchChartsWithScope(payload);
            }
        }
        if (typeof global.pbjFacilityEventsRefreshCharts === 'function') {
            global.pbjFacilityEventsRefreshCharts();
        }
        if (typeof global.pbjRefreshEinHeadcountAfterAnalysis === 'function') {
            global.pbjRefreshEinHeadcountAfterAnalysis();
        } else if (typeof global.pbjLoadEinHeadcountByJobChart === 'function') {
            global.pbjLoadEinHeadcountByJobChart({ force: true });
        }
        renderScopeEventsList(state);
    }

    function closeScopeMenu(kind) {
        var kinds = kind ? [kind] : [SCOPE_MENU_RANGE, SCOPE_MENU_EVENTS];
        kinds.forEach(function (k) {
            var menu = scopeMenuEl(k);
            if (!menu) {
                return;
            }
            menu.classList.add('d-none');
            menu.classList.remove('pbj-chart-scope-menu--dragging', 'pbj-chart-scope-menu--resizing');
            menu.style.visibility = '';
            menu.style.pointerEvents = '';
        });
        if (!kind) {
            global.__pbjChartScopeAnchor = null;
            global.__pbjChartScopeMenuKind = null;
        } else if (global.__pbjChartScopeMenuKind === kind) {
            global.__pbjChartScopeAnchor = null;
            global.__pbjChartScopeMenuKind = null;
        }
    }

    function closeAllScopeMenus() {
        closeScopeMenu();
    }

    function positionScopeMenu(anchor, opts) {
        opts = opts || {};
        var kind = opts.kind || global.__pbjChartScopeMenuKind || SCOPE_MENU_RANGE;
        var menu = scopeMenuEl(kind);
        if (!menu || !anchor) {
            return;
        }
        if (!opts.repositionOnly) {
            menu.classList.remove('d-none');
        }
        applyScopeMenuSize(menu, kind);
        menu.style.position = 'fixed';
        menu.style.zIndex = '1065';
        menu.style.visibility = 'hidden';
        menu.style.pointerEvents = 'none';

        function place() {
            var rect = anchor.getBoundingClientRect();
            var off = global.__pbjChartScopeDragOffset || { dx: 0, dy: 0 };
            var menuW = menu.offsetWidth || 336;
            var baseLeft = Math.min(rect.left, window.innerWidth - menuW - 8);
            var baseTop = rect.bottom + 6;
            if (baseTop + menu.offsetHeight > window.innerHeight - 8 && rect.top - menu.offsetHeight - 6 > 8) {
                baseTop = rect.top - menu.offsetHeight - 6;
            }
            var pos = clampScopeMenuPosition(menu, baseLeft + off.dx, baseTop + off.dy);
            menu.style.left = pos.left + 'px';
            menu.style.top = pos.top + 'px';
        }

        place();
        requestAnimationFrame(function () {
            place();
            requestAnimationFrame(function () {
                place();
                menu.style.visibility = '';
                menu.style.pointerEvents = '';
            });
        });
    }

    function openScopeMenu(anchor, kind) {
        kind = kind || SCOPE_MENU_RANGE;
        if (global.__pbjChartScopeAnchor !== anchor || global.__pbjChartScopeMenuKind !== kind) {
            resetScopeMenuLayout(kind);
        }
        closeScopeMenu(kind === SCOPE_MENU_RANGE ? SCOPE_MENU_EVENTS : SCOPE_MENU_RANGE);
        if (typeof global.pbjV2CloseControlCenter === 'function') {
            global.pbjV2CloseControlCenter();
        }
        populateScopeSelects();
        var profile = (anchor.closest('.pbj-trend-events-toolbar') || {}).getAttribute('data-pbj-chart-scope-profile') || 'default';
        global.__pbjChartScopeActiveProfile = profile;
        applyProfileConstraints(profile);
        if (!global.__pbjChartScopeUserEdited) {
            global.__pbjChartScopeState = scopeStateFromControlCenter();
        }
        syncMenuFromState(global.__pbjChartScopeState);
        global.__pbjChartScopeAnchor = anchor;
        global.__pbjChartScopeMenuKind = kind;
        updateScopeMenuSubtitle(anchor);
        positionScopeMenu(anchor, { kind: kind });
        if (kind === SCOPE_MENU_EVENTS) {
            renderScopeEventsList(readMenuCustomState());
        }
    }

    function wireScopeMenuDragResize(menu) {
        var dragHandle = qs('.pbj-chart-scope-drag-handle', menu);
        var resizeHandle = qs('.pbj-chart-scope-resize-handle', menu);
        if (!dragHandle && !resizeHandle) {
            return;
        }

        var dragState = null;

        function finishPointer() {
            if (!dragState) {
                return;
            }
            menu.classList.remove('pbj-chart-scope-menu--dragging', 'pbj-chart-scope-menu--resizing');
            dragState = null;
        }

        function onPointerMove(ev) {
            if (!dragState) {
                return;
            }
            if (dragState.mode === 'drag') {
                var dx = ev.clientX - dragState.startX;
                var dy = ev.clientY - dragState.startY;
                global.__pbjChartScopeDragOffset = {
                    dx: dragState.startDx + dx,
                    dy: dragState.startDy + dy
                };
                if (global.__pbjChartScopeAnchor) {
                    positionScopeMenu(global.__pbjChartScopeAnchor, { kind: global.__pbjChartScopeMenuKind });
                }
            } else if (dragState.mode === 'resize') {
                var maxW = Math.floor(window.innerWidth * SCOPE_MENU_MAX_W_RATIO);
                var maxH = Math.floor(window.innerHeight * SCOPE_MENU_MAX_H_RATIO);
                var nextW = Math.max(SCOPE_MENU_MIN_W, Math.min(maxW, dragState.startW + (ev.clientX - dragState.startX)));
                var nextH = Math.max(SCOPE_MENU_MIN_H, Math.min(maxH, dragState.startH + (ev.clientY - dragState.startY)));
                global.__pbjChartScopeSize = { w: nextW, h: nextH };
                applyScopeMenuSize(menu);
                var pos = clampScopeMenuPosition(
                    menu,
                    dragState.anchorRight - nextW,
                    dragState.anchorBottom - menu.offsetHeight
                );
                menu.style.position = 'fixed';
                menu.style.left = pos.left + 'px';
                menu.style.top = pos.top + 'px';
                menu.style.zIndex = '1065';
                if (global.__pbjChartScopeAnchor) {
                    var base = scopeMenuBasePosition(global.__pbjChartScopeAnchor, global.__pbjChartScopeMenuKind);
                    global.__pbjChartScopeDragOffset = {
                        dx: pos.left - base.left,
                        dy: pos.top - base.top
                    };
                }
            }
            ev.preventDefault();
        }

        document.addEventListener('pointermove', onPointerMove);
        document.addEventListener('pointerup', finishPointer);
        document.addEventListener('pointercancel', finishPointer);

        if (dragHandle) {
            dragHandle.addEventListener('pointerdown', function (ev) {
                if (ev.button !== 0) {
                    return;
                }
                dragState = {
                    mode: 'drag',
                    startX: ev.clientX,
                    startY: ev.clientY,
                    startDx: (global.__pbjChartScopeDragOffset || {}).dx || 0,
                    startDy: (global.__pbjChartScopeDragOffset || {}).dy || 0
                };
                menu.classList.add('pbj-chart-scope-menu--dragging');
                dragHandle.setPointerCapture(ev.pointerId);
                ev.preventDefault();
            });
        }

        if (resizeHandle) {
            resizeHandle.addEventListener('pointerdown', function (ev) {
                if (ev.button !== 0) {
                    return;
                }
                var inner = scopeMenuInner();
                var rect = menu.getBoundingClientRect();
                dragState = {
                    mode: 'resize',
                    startX: ev.clientX,
                    startY: ev.clientY,
                    startW: menu.offsetWidth || SCOPE_MENU_MIN_W,
                    startH: (inner && inner.offsetHeight) || menu.offsetHeight || SCOPE_MENU_MIN_H,
                    anchorRight: rect.right,
                    anchorBottom: rect.bottom
                };
                menu.classList.add('pbj-chart-scope-menu--resizing');
                resizeHandle.setPointerCapture(ev.pointerId);
                ev.preventDefault();
                ev.stopPropagation();
            });
        }
    }

    function wireScopeMenus() {
        if (document.body.dataset.pbjScopeMenusWired === '1') {
            return;
        }
        document.body.dataset.pbjScopeMenusWired = '1';

        var rangeMenu = scopeMenuEl(SCOPE_MENU_RANGE);
        var eventsMenu = scopeMenuEl(SCOPE_MENU_EVENTS);
        if (rangeMenu) {
            wireScopeMenuDragResize(rangeMenu);
            bindScopePreviewRefresh(rangeMenu);
            qsa('input[name="pbjChartScopeGrain"]', rangeMenu).forEach(function (inp) {
                inp.addEventListener('change', function () {
                    showScopePicker(inp.value);
                });
            });
        }
        if (eventsMenu) {
            wireScopeMenuDragResize(eventsMenu);
        }

        $('pbjChartRangeApplyBtn')?.addEventListener('click', function () {
            var state = readMenuCustomState();
            var profile = activeScopeProfile();
            if (profile !== 'casemix') {
                global.__pbjChartScopeUserEdited = true;
            }
            applyChartScope(state, { profile: profile });
            closeScopeMenu(SCOPE_MENU_RANGE);
        });

        $('pbjChartRangeResetBtn')?.addEventListener('click', function () {
            var profile = activeScopeProfile();
            if (profile === 'casemix') {
                global.__pbjCaseMixChartScopeUserEdited = false;
                global.__pbjCaseMixChartScopeState = { mode: 'all' };
                syncMenuFromState({ mode: 'all' });
                applyChartScope({ mode: 'all' }, { profile: 'casemix' });
                return;
            }
            global.__pbjChartScopeUserEdited = false;
            var state = scopeStateFromControlCenter();
            global.__pbjChartScopeState = state;
            syncMenuFromState(state);
            applyChartScope(state, { profile: profile });
        });

        $('pbjChartRangeCloseBtn')?.addEventListener('click', function () {
            closeScopeMenu(SCOPE_MENU_RANGE);
        });

        $('pbjChartEventsCloseBtn')?.addEventListener('click', function () {
            closeScopeMenu(SCOPE_MENU_EVENTS);
        });

        $('pbjChartEventsApplyBtn')?.addEventListener('click', function () {
            if (typeof global.pbjFacilityEventsRefreshCharts === 'function') {
                global.pbjFacilityEventsRefreshCharts();
            }
            renderScopeEventsList(global.__pbjChartScopeState);
            closeScopeMenu(SCOPE_MENU_EVENTS);
        });

        function bindScopeEventsListActions(menu) {
            if (!menu || menu.dataset.pbjScopeEventActionsBound === '1') {
                return;
            }
            menu.dataset.pbjScopeEventActionsBound = '1';
            menu.addEventListener('click', function (ev) {
                var editBtn = ev.target.closest('[data-pbj-scope-event-edit]');
                if (editBtn) {
                    ev.preventDefault();
                    ev.stopPropagation();
                    var editId = editBtn.getAttribute('data-pbj-scope-event-edit');
                    if (editId && typeof global.pbjFacilityEventsEditManualInScope === 'function') {
                        global.pbjFacilityEventsEditManualInScope(editId);
                        renderScopeEventsList(readMenuCustomState());
                    }
                    return;
                }
                var delBtn = ev.target.closest('[data-pbj-scope-event-delete]');
                if (delBtn) {
                    ev.preventDefault();
                    ev.stopPropagation();
                    var delId = delBtn.getAttribute('data-pbj-scope-event-delete');
                    if (delId && typeof global.pbjFacilityEventsDeleteManual === 'function') {
                        global.pbjFacilityEventsDeleteManual(delId);
                        renderScopeEventsList(readMenuCustomState());
                        if (typeof global.pbjFacilityEventsRefreshCharts === 'function') {
                            global.pbjFacilityEventsRefreshCharts();
                        }
                    }
                    return;
                }
                var chip = ev.target.closest('[data-pbj-scope-event-type]');
                if (!chip) {
                    return;
                }
                var type = chip.getAttribute('data-pbj-scope-event-type');
                if (!type || typeof global.pbjFacilityEventsSetType !== 'function') {
                    return;
                }
                var on = global.pbjFacilityEventsTypeEnabled(type);
                global.pbjFacilityEventsSetType(type, !on);
                if (!on) {
                    ensureChartEventsVisible();
                }
                renderScopeEventsList(readMenuCustomState());
                if (typeof global.pbjFacilityEventsRefreshCharts === 'function') {
                    global.pbjFacilityEventsRefreshCharts();
                }
            });
        }

        bindScopeEventsListActions(eventsMenu);
        bindScopeEventsListActions(rangeMenu);

        $('pbjChartScopeEventsManageBtn')?.addEventListener('click', function () {
            closeScopeMenu(SCOPE_MENU_EVENTS);
            if (typeof global.pbjFacilityEventsOpenModal === 'function') {
                global.pbjFacilityEventsOpenModal();
            }
        });

        $('pbjChartScopeEventAddBtn')?.addEventListener('click', function () {
            var iso = ($('pbjChartScopeEventDate') || {}).value || '';
            var noteEl = $('pbjChartScopeEventNote');
            var note = noteEl ? String(noteEl.value || '').trim() : '';
            if (!iso) {
                showScopeMenuMessage('Pick a date for this event.', 'error');
                return;
            }
            if (!note) {
                showScopeMenuMessage('Add a short description (e.g. resident fall).', 'error');
                if (noteEl) {
                    noteEl.focus();
                }
                return;
            }
            if (typeof global.pbjFacilityEventsAddManualFromFields !== 'function') {
                showScopeMenuMessage('Events are not available on this page.', 'error');
                return;
            }
            var ok = global.pbjFacilityEventsAddManualFromFields(iso, note, note);
            if (ok) {
                if (noteEl) {
                    noteEl.value = '';
                    noteEl.focus();
                }
                ensureChartEventsVisible();
                showScopeAddFeedback(iso, note);
                renderScopeEventsList(readMenuCustomState());
                if (typeof global.pbjFacilityEventsRefreshCharts === 'function') {
                    global.pbjFacilityEventsRefreshCharts();
                }
            }
        });

        document.addEventListener('click', function (ev) {
            var openRange = rangeMenu && !rangeMenu.classList.contains('d-none');
            var openEvents = eventsMenu && !eventsMenu.classList.contains('d-none');
            if (!openRange && !openEvents) {
                return;
            }
            if ((rangeMenu && rangeMenu.contains(ev.target)) || (eventsMenu && eventsMenu.contains(ev.target))) {
                return;
            }
            if (ev.target.closest('.pbj-trend-range-menu-btn, .pbj-trend-events-panel-btn')) {
                return;
            }
            closeAllScopeMenus();
        });

        window.addEventListener('resize', function () {
            if (global.__pbjChartScopeAnchor && global.__pbjChartScopeMenuKind) {
                var activeMenu = scopeMenuEl(global.__pbjChartScopeMenuKind);
                if (activeMenu && !activeMenu.classList.contains('d-none')) {
                    positionScopeMenu(global.__pbjChartScopeAnchor, { kind: global.__pbjChartScopeMenuKind });
                }
            }
        });

        window.addEventListener('scroll', function () {
            if (global.__pbjChartScopeAnchor && global.__pbjChartScopeMenuKind) {
                var activeMenu = scopeMenuEl(global.__pbjChartScopeMenuKind);
                if (activeMenu && !activeMenu.classList.contains('d-none')) {
                    positionScopeMenu(global.__pbjChartScopeAnchor, { kind: global.__pbjChartScopeMenuKind });
                }
            }
        }, true);

        document.addEventListener('click', function (ev) {
            var bench = ev.target.closest('.pbj-chart-scope-mobile-benchmark-btn');
            if (!bench) {
                return;
            }
            var id = bench.getAttribute('data-pbj-benchmark-trigger');
            var target = id && document.getElementById(id);
            if (!target) {
                return;
            }
            if (target.tagName === 'INPUT') {
                target.focus();
                target.select();
                return;
            }
            if (typeof bootstrap !== 'undefined') {
                bootstrap.Dropdown.getOrCreateInstance(target).show();
            }
        });
    }

    function resolveScopeMenuAnchor(btn) {
        var toolbar = btn.closest('.pbj-trend-events-toolbar');
        if (!toolbar) {
            return btn;
        }
        if (!btn.classList.contains('dropdown-item')) {
            return btn;
        }
        if (btn.classList.contains('pbj-trend-range-menu-btn')) {
            return toolbar.querySelector('.pbj-chart-scope-header-btns .pbj-trend-range-menu-btn') ||
                toolbar.querySelector('.pbj-chart-scope-mobile-options .dropdown-toggle') ||
                btn;
        }
        if (btn.classList.contains('pbj-trend-events-panel-btn')) {
            return toolbar.querySelector('.pbj-chart-scope-header-btns .pbj-trend-events-panel-btn') ||
                toolbar.querySelector('.pbj-chart-scope-mobile-options .dropdown-toggle') ||
                btn;
        }
        return btn;
    }

    function handleScopeMenuButtonClick(btn, kind) {
        var anchor = resolveScopeMenuAnchor(btn);
        var menuEl = scopeMenuEl(kind);
        if (
            menuEl &&
            !menuEl.classList.contains('d-none') &&
            global.__pbjChartScopeAnchor === anchor &&
            global.__pbjChartScopeMenuKind === kind
        ) {
            closeScopeMenu(kind);
            return;
        }
        openScopeMenu(anchor, kind);
    }

    function init() {
        global.__pbjChartScopeState = scopeStateFromControlCenter();
        global.__pbjChartScopeUserEdited = false;
        wireScopeMenus();
        populateScopeSelects();

        global.pbjChartScopeSyncFromControlCenter = function () {
            populateScopeSelects();
            global.__pbjChartScopeState = scopeStateFromControlCenter();
            global.__pbjChartScopeUserEdited = false;
            syncMenuFromState(global.__pbjChartScopeState);
            renderScopeEventsList(global.__pbjChartScopeState);
            if (global.__pbjChartScopeAnchor && global.__pbjChartScopeMenuKind) {
                var activeMenu = scopeMenuEl(global.__pbjChartScopeMenuKind);
                if (activeMenu && !activeMenu.classList.contains('d-none')) {
                    positionScopeMenu(global.__pbjChartScopeAnchor, {
                        kind: global.__pbjChartScopeMenuKind,
                        repositionOnly: true
                    });
                }
            }
        };
        global.pbjChartScopePopulateSelects = populateScopeSelects;
        global.pbjChartScopeRenderEventsList = renderScopeEventsList;
        global.pbjChartScopeStateToIsoRange = scopeStateToIsoRange;
        global.pbjChartScopeQuarterToDateRange = quarterToDateRange;
        global.pbjBuildScopeApiPayload = buildScopeApiPayload;

        document.addEventListener('click', function (ev) {
            var rangeBtn = ev.target.closest('.pbj-trend-range-menu-btn');
            if (rangeBtn) {
                ev.preventDefault();
                ev.stopPropagation();
                handleScopeMenuButtonClick(rangeBtn, SCOPE_MENU_RANGE);
                return;
            }
            var eventsBtn = ev.target.closest('.pbj-trend-events-panel-btn');
            if (eventsBtn) {
                ev.preventDefault();
                ev.stopPropagation();
                handleScopeMenuButtonClick(eventsBtn, SCOPE_MENU_EVENTS);
            }
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(typeof window !== 'undefined' ? window : globalThis);
