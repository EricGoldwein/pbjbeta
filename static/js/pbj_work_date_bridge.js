/**
 * PBJ320 superdynamic v2 — single work-date control synced across dashboard surfaces.
 * Requires superdynamic_utils.js (pbjNormalizeIsoWorkDate, pbjClampIsoToPbjBounds) when available.
 */
(function (global) {
    'use strict';

    function norm(iso) {
        if (typeof global.pbjNormalizeIsoWorkDate === 'function') {
            return global.pbjNormalizeIsoWorkDate(iso);
        }
        var s = String(iso == null ? '' : iso).trim();
        var m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
        return m ? m[1] + '-' + m[2] + '-' + m[3] : '';
    }

    function clamp(iso) {
        if (typeof global.pbjClampIsoToPbjBounds === 'function') {
            return global.pbjClampIsoToPbjBounds(iso);
        }
        return iso;
    }

    function workDateInputs() {
        return {
            primary: document.getElementById('pbjV2WorkDatePrimary'),
            filterDay: document.getElementById('filterDayDate'),
            einDay: document.getElementById('einNursingWorkDayFilter'),
            dailySearch: document.getElementById('specificDaySearch'),
            singleDay: document.getElementById('singleDayDate'),
        };
    }

    var _syncLock = false;

    /**
     * @param {string} iso YYYY-MM-DD
     * @param {object} [options]
     * @param {boolean} [options.setDashboardDayFilter] check Day grain + onFilterTypeChange
     * @param {boolean} [options.scrollDaily] scroll/highlight daily table row (default true unless openReport)
     * @param {boolean} [options.openReport] open single-day report modal
     * @param {boolean} [options.loadRoster] fetch EIN day roster
     * @param {string} [options.scrollTo] 'daily' | 'roster' | 'dayLevel'
     * @returns {string|null}
     */
    function pbjSetActiveWorkDate(iso, options) {
        options = options || {};
        var d = clamp(norm(iso));
        if (!d) {
            return null;
        }

        _syncLock = true;
        try {
            var inp = workDateInputs();
            if (inp.primary) {
                inp.primary.value = d;
            }
            if (inp.filterDay) {
                inp.filterDay.value = d;
            }
            if (inp.einDay) {
                inp.einDay.value = d;
            }
            if (inp.dailySearch) {
                inp.dailySearch.value = d;
            }
            if (inp.singleDay) {
                inp.singleDay.value = d;
            }
        } finally {
            _syncLock = false;
        }

        if (options.setDashboardDayFilter) {
            var dayRadio = document.getElementById('filterTypeDay');
            if (dayRadio && !dayRadio.checked) {
                dayRadio.checked = true;
                if (typeof global.onFilterTypeChange === 'function') {
                    global.onFilterTypeChange();
                }
            }
        }

        if (options.loadRoster && typeof global.fetchEinDayRoster === 'function') {
            global.fetchEinDayRoster(d);
        }

        if (options.openReport && typeof global.openSingleDayReport === 'function') {
            global.openSingleDayReport(d);
        } else if (options.scrollDaily !== false && typeof global.pbjScrollDailyTableToIso === 'function') {
            global.pbjScrollDailyTableToIso(d);
        }

        var scrollTarget = options.scrollTo;
        if (scrollTarget === 'roster' || scrollTarget === 'dayLevel') {
            var sec =
                document.getElementById('dayLevelEvidenceSection') ||
                document.getElementById('dailyStaffingEinSection') ||
                document.getElementById('einNursingEmployeeSection');
            if (sec && sec.scrollIntoView) {
                sec.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        } else if (scrollTarget === 'daily') {
            var daily = document.getElementById('dailyDataSection');
            if (daily && daily.scrollIntoView) {
                daily.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }

        try {
            global.dispatchEvent(
                new CustomEvent('pbj:active-work-date', { detail: { iso: d, options: options } })
            );
        } catch (eEv) {
            /* IE / old WebView */
        }
        return d;
    }

    function pbjBindWorkDateSync() {
        var inp = workDateInputs();
        function onChange(el, source) {
            if (!el || _syncLock) {
                return;
            }
            var v = norm(el.value);
            if (!v) {
                return;
            }
            var scrollTo = null;
            if (source === 'dailySearch' || source === 'primaryDaily') {
                scrollTo = 'daily';
            }
            pbjSetActiveWorkDate(v, {
                scrollDaily: false,
                scrollTo: scrollTo,
            });
        }
        if (inp.primary) {
            inp.primary.addEventListener('change', function () {
                onChange(inp.primary, 'primary');
            });
        }
        if (inp.filterDay) {
            inp.filterDay.addEventListener('change', function () {
                onChange(inp.filterDay, 'filterDay');
            });
        }
        if (inp.einDay) {
            inp.einDay.addEventListener('change', function () {
                onChange(inp.einDay, 'einDay');
            });
        }
        if (inp.dailySearch) {
            inp.dailySearch.addEventListener('change', function () {
                onChange(inp.dailySearch, 'dailySearch');
            });
        }
        var btnDaily = document.getElementById('pbjV2WorkDateGoDaily');
        var btnRoster = document.getElementById('pbjV2WorkDateGoRoster');
        var btnReport = document.getElementById('pbjV2WorkDateDayReport');
        if (btnDaily) {
            btnDaily.addEventListener('click', function () {
                var v = inp.primary && inp.primary.value ? inp.primary.value : '';
                pbjSetActiveWorkDate(v, { scrollDaily: true, scrollTo: 'daily', setDashboardDayFilter: true });
            });
        }
        if (btnRoster) {
            btnRoster.addEventListener('click', function () {
                var v = inp.primary && inp.primary.value ? inp.primary.value : '';
                pbjSetActiveWorkDate(v, { loadRoster: true, scrollTo: 'roster' });
            });
        }
        if (btnReport) {
            btnReport.addEventListener('click', function () {
                var v = inp.primary && inp.primary.value ? inp.primary.value : '';
                pbjSetActiveWorkDate(v, { openReport: true });
            });
        }
    }

    global.pbjSetActiveWorkDate = pbjSetActiveWorkDate;
    global.pbjBindWorkDateSync = pbjBindWorkDateSync;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', pbjBindWorkDateSync);
    } else {
        pbjBindWorkDateSync();
    }
})(typeof window !== 'undefined' ? window : this);
