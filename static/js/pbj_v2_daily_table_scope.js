/**
 * v2 — Daily PBJ table range control (shared scope config + chart-style range menu).
 */
(function (global) {
    'use strict';

    var MENU_MIN_W = 272;
    var MENU_MAX_W_RATIO = 0.92;

    function $(id) {
        return document.getElementById(id);
    }

    function qs(sel, root) {
        return (root || document).querySelector(sel);
    }

    function qsa(sel, root) {
        return Array.prototype.slice.call((root || document).querySelectorAll(sel));
    }

    function defaultState() {
        return { mode: 'all', filterType: 'quarters', quarter: 'all', year: 'all' };
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

    function populateMenuSelects() {
        cloneSelectOptions($('quarterFilterTable'), $('pbjDailyTableScopeQuarterSelect'));
        cloneSelectOptions($('yearFilterTable'), $('pbjDailyTableScopeYearSelect'));
    }

    function showPicker(grain) {
        var menu = $('pbjDailyTableRangeMenu');
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
    }

    function syncMenuFromState(state) {
        var st = state || global.__pbjDailyTableScopeState || defaultState();
        var ft = st.mode === 'all' ? 'quarters' : (st.filterType || 'quarters');
        var grainRadio = qs('input[name="pbjDailyTableScopeGrain"][value="' + ft + '"]');
        if (grainRadio) {
            grainRadio.checked = true;
        }
        showPicker(ft);
        var qSel = $('pbjDailyTableScopeQuarterSelect');
        var ySel = $('pbjDailyTableScopeYearSelect');
        if (qSel) {
            var qWant = st.mode === 'all' ? 'all' : String(st.quarter || 'all');
            Array.from(qSel.options || []).forEach(function (opt) {
                opt.selected = opt.value === qWant;
            });
        }
        if (ySel) {
            var yWant = st.mode === 'all' ? 'all' : String(st.year || 'all');
            Array.from(ySel.options || []).forEach(function (opt) {
                opt.selected = opt.value === yWant;
            });
        }
        if ($('pbjDailyTableScopeStartDate')) {
            $('pbjDailyTableScopeStartDate').value = st.startDate || '';
        }
        if ($('pbjDailyTableScopeEndDate')) {
            $('pbjDailyTableScopeEndDate').value = st.endDate || '';
        }
        if ($('pbjDailyTableScopeDayDate')) {
            $('pbjDailyTableScopeDayDate').value = st.dayDate || '';
        }
    }

    function readMenuState() {
        var ft = (qs('input[name="pbjDailyTableScopeGrain"]:checked') || {}).value || 'quarters';
        var state = {
            mode: 'custom',
            filterType: ft,
            quarter: 'all',
            year: 'all',
            startDate: '',
            endDate: '',
            dayDate: ''
        };
        var qSel = $('pbjDailyTableScopeQuarterSelect');
        var ySel = $('pbjDailyTableScopeYearSelect');
        if (ft === 'quarters' && qSel) {
            state.quarter = qSel.value || 'all';
        } else if (ft === 'years' && ySel) {
            state.year = ySel.value || 'all';
        } else if (ft === 'daterange') {
            state.startDate = ($('pbjDailyTableScopeStartDate') || {}).value || '';
            state.endDate = ($('pbjDailyTableScopeEndDate') || {}).value || '';
        } else if (ft === 'day') {
            state.dayDate = ($('pbjDailyTableScopeDayDate') || {}).value || '';
        }
        return state;
    }

    function loadedRowsForLabel() {
        if (Array.isArray(global.__pbjDailyRowsFull) && global.__pbjDailyRowsFull.length) {
            return global.__pbjDailyRowsFull;
        }
        if (Array.isArray(global.__pbjDailyScopedRows) && global.__pbjDailyScopedRows.length) {
            return global.__pbjDailyScopedRows;
        }
        if (typeof global.currentData !== 'undefined' && Array.isArray(global.currentData)) {
            return global.currentData;
        }
        return [];
    }

    function auditPeriodLabel(state) {
        var st = state || global.__pbjDailyTableScopeState || defaultState();
        if (st.mode === 'all') {
            if (typeof global.pbjScopeLabelFromLoadedDailyRows === 'function') {
                return global.pbjScopeLabelFromLoadedDailyRows(loadedRowsForLabel());
            }
            return 'All loaded days';
        }
        if (typeof global.pbjScopeStateDisplayLabel === 'function') {
            return global.pbjScopeStateDisplayLabel(st, {
                allLabel: typeof global.pbjScopeLabelFromLoadedDailyRows === 'function'
                    ? global.pbjScopeLabelFromLoadedDailyRows(loadedRowsForLabel())
                    : 'All loaded days'
            });
        }
        return 'Applied range';
    }

    function flaggedDaysAuditSuffix() {
        if (global.__pbjDailyFlaggedOnlyActive !== true) {
            return '';
        }
        if (typeof global.pbjDailyFlaggedDaysCriteriaLabel === 'function') {
            var crit = global.pbjDailyFlaggedDaysCriteriaLabel();
            if (crit) {
                return ' · ' + crit;
            }
        }
        return ' · flagged days only';
    }

    function refreshAuditPeriodLabel() {
        var el = $('dailyTableAuditPeriodLabel');
        if (!el) {
            return;
        }
        var label = auditPeriodLabel();
        var suffix = flaggedDaysAuditSuffix();
        var display = (label || '—') + suffix;
        el.textContent = display;
        el.title = display;
        var hint = $('dailyDataScopeHint');
        if (hint) {
            hint.textContent = display;
        }
    }

    function syncHiddenFilterFields(state) {
        var st = state || defaultState();
        var dayInp = $('specificDaySearch');
        var rs = $('dateRangeStart');
        var re = $('dateRangeEnd');
        var qf = $('quarterFilterTable');
        var yf = $('yearFilterTable');
        if (st.mode === 'all') {
            if (dayInp) {
                dayInp.value = '';
            }
            if (rs) {
                rs.value = '';
            }
            if (re) {
                re.value = '';
            }
            if (qf) {
                qf.value = 'all';
            }
            if (yf) {
                yf.value = 'all';
            }
            return;
        }
        var ft = st.filterType || 'quarters';
        if (dayInp) {
            dayInp.value = ft === 'day' ? (st.dayDate || '') : '';
        }
        if (rs) {
            rs.value = ft === 'daterange' ? (st.startDate || '') : '';
        }
        if (re) {
            re.value = ft === 'daterange' ? (st.endDate || '') : '';
        }
        if (qf) {
            qf.value = ft === 'quarters' ? (st.quarter || 'all') : 'all';
        }
        if (yf) {
            yf.value = ft === 'years' ? (st.year || 'all') : 'all';
        }
    }

    async function applyScopeState(state) {
        var st = state || defaultState();
        global.__pbjDailyTableScopeState = st;
        syncHiddenFilterFields(st);
        refreshAuditPeriodLabel();

        if (st.mode === 'all') {
            if (typeof global.restoreFullData === 'function' && global.originalData) {
                global.restoreFullData();
            } else if (
                Array.isArray(global.__pbjDailyRowsFull) &&
                global.__pbjDailyRowsFull.length &&
                typeof global.pbjCommitDailyTableScopedRows === 'function'
            ) {
                global.pbjCommitDailyTableScopedRows(global.__pbjDailyRowsFull, { keepPage: false, clearOriginal: true });
                if (typeof global.clearFilterMessages === 'function') {
                    global.clearFilterMessages();
                }
            }
            try {
                if (typeof global.pbjSyncNonnurseFromDailyQuarter === 'function') {
                    global.pbjSyncNonnurseFromDailyQuarter();
                }
            } catch (eSync) { /* noop */ }
            if (typeof global.pbjRefreshDailyNonnurseInlineIfActive === 'function') {
                global.pbjRefreshDailyNonnurseInlineIfActive();
            }
            return;
        }

        var ft = st.filterType || 'quarters';
        try {
            if (ft === 'day' && typeof global.searchSpecificDay === 'function') {
                await global.searchSpecificDay();
            } else if (ft === 'daterange' && typeof global.searchDateRange === 'function') {
                await global.searchDateRange();
            } else if (ft === 'years' && typeof global.filterByYear === 'function') {
                await global.filterByYear();
            } else if (ft === 'quarters' && typeof global.filterByQuarter === 'function') {
                await global.filterByQuarter();
            }
        } catch (err) {
            console.warn('Daily table scope apply failed:', err);
        }
        refreshAuditPeriodLabel();
        if (typeof global.pbjRefreshDailyNonnurseInlineIfActive === 'function') {
            global.pbjRefreshDailyNonnurseInlineIfActive();
        }
    }

    function positionMenu(anchor) {
        var menu = $('pbjDailyTableRangeMenu');
        if (!menu || !anchor) {
            return;
        }
        menu.classList.remove('d-none');
        menu.style.position = 'fixed';
        menu.style.zIndex = '1065';
        var rect = anchor.getBoundingClientRect();
        var menuW = Math.min(Math.max(menu.offsetWidth || MENU_MIN_W, MENU_MIN_W), window.innerWidth * MENU_MAX_W_RATIO);
        menu.style.width = menuW + 'px';
        var left = Math.min(rect.left, window.innerWidth - menuW - 8);
        var top = rect.bottom + 6;
        if (top + menu.offsetHeight > window.innerHeight - 8 && rect.top - menu.offsetHeight - 6 > 8) {
            top = rect.top - menu.offsetHeight - 6;
        }
        menu.style.left = Math.max(8, left) + 'px';
        menu.style.top = Math.max(8, top) + 'px';
        menu.style.visibility = 'visible';
        menu.style.pointerEvents = '';
    }

    function closeMenu() {
        var menu = $('pbjDailyTableRangeMenu');
        if (!menu) {
            return;
        }
        menu.classList.add('d-none');
        global.__pbjDailyTableRangeAnchor = null;
    }

    function openMenu(anchor) {
        populateMenuSelects();
        syncMenuFromState(global.__pbjDailyTableScopeState);
        global.__pbjDailyTableRangeAnchor = anchor || $('pbjDailyTableRangeBtn');
        positionMenu(global.__pbjDailyTableRangeAnchor);
    }

    function toggleMenu(anchor) {
        var menu = $('pbjDailyTableRangeMenu');
        if (menu && !menu.classList.contains('d-none') && global.__pbjDailyTableRangeAnchor === anchor) {
            closeMenu();
            return;
        }
        openMenu(anchor);
    }

    function pbjDailyTableScopeToIsoRange(state) {
        if (typeof global.pbjChartScopeStateToIsoRange === 'function') {
            return global.pbjChartScopeStateToIsoRange(state || global.__pbjDailyTableScopeState || defaultState());
        }
        var st = state || defaultState();
        if (st.mode === 'all') {
            var rows = loadedRowsForLabel();
            var minD = '';
            var maxD = '';
            rows.forEach(function (r) {
                var wd = r && r.WorkDate ? String(r.WorkDate).trim().slice(0, 10) : '';
                if (!/^\d{4}-\d{2}-\d{2}$/.test(wd)) {
                    return;
                }
                if (!minD || wd < minD) {
                    minD = wd;
                }
                if (!maxD || wd > maxD) {
                    maxD = wd;
                }
            });
            return { start: minD || '', end: maxD || '' };
        }
        if (st.filterType === 'daterange' && st.startDate && st.endDate) {
            return { start: st.startDate, end: st.endDate };
        }
        if (st.filterType === 'day' && st.dayDate) {
            return { start: st.dayDate, end: st.dayDate };
        }
        return { start: '', end: '' };
    }

    function resetScope() {
        global.__pbjDailyTableScopeState = defaultState();
        syncMenuFromState(global.__pbjDailyTableScopeState);
        syncHiddenFilterFields(global.__pbjDailyTableScopeState);
        applyScopeState(global.__pbjDailyTableScopeState);
        closeMenu();
    }

    function wireMenu() {
        if (document.body.dataset.pbjDailyTableScopeWired === '1') {
            return;
        }
        document.body.dataset.pbjDailyTableScopeWired = '1';

        $('pbjDailyTableRangeApplyBtn')?.addEventListener('click', function () {
            var state = readMenuState();
            if (state.filterType === 'quarters' && (!state.quarter || state.quarter === 'all')) {
                state.mode = 'all';
            } else if (state.filterType === 'years' && (!state.year || state.year === 'all')) {
                state.mode = 'all';
            }
            applyScopeState(state);
            closeMenu();
        });

        $('pbjDailyTableRangeResetBtn')?.addEventListener('click', resetScope);
        $('pbjDailyTableRangeCloseBtn')?.addEventListener('click', closeMenu);

        qsa('input[name="pbjDailyTableScopeGrain"]', $('pbjDailyTableRangeMenu')).forEach(function (inp) {
            inp.addEventListener('change', function () {
                showPicker(inp.value);
            });
        });

        document.addEventListener('click', function (ev) {
            var btn = ev.target.closest('.pbj-daily-table-range-btn');
            if (btn) {
                ev.preventDefault();
                ev.stopPropagation();
                toggleMenu(btn);
                return;
            }
            var menu = $('pbjDailyTableRangeMenu');
            if (!menu || menu.classList.contains('d-none')) {
                return;
            }
            if (menu.contains(ev.target)) {
                return;
            }
            closeMenu();
        });

        window.addEventListener('resize', function () {
            if (global.__pbjDailyTableRangeAnchor) {
                positionMenu(global.__pbjDailyTableRangeAnchor);
            }
        });
    }

    function init() {
        global.__pbjDailyTableScopeState = global.__pbjDailyTableScopeState || defaultState();
        wireMenu();
        refreshAuditPeriodLabel();
    }

    global.pbjRefreshDailyTableAuditPeriodLabel = refreshAuditPeriodLabel;
    global.pbjDailyTableScopeToIsoRange = pbjDailyTableScopeToIsoRange;
    global.pbjApplyDailyTableScopeState = applyScopeState;
    global.pbjResetDailyTableScope = resetScope;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(typeof window !== 'undefined' ? window : globalThis);
