/**

 * Case Builder standalone — minimal PBJ320 AI bridge when dashboard inline export is absent.

 */

(function () {

    'use strict';



    function metaCcn() {

        try {

            var el = document.getElementById('pbj-report-builder-v3-meta');

            if (el && el.textContent) {

                var o = JSON.parse(el.textContent);

                return String(o.ccn || '').replace(/\D/g, '');

            }

        } catch (e) { /* ignore */ }

        return String(window.PBJ320_EXPORT_CCN || '').replace(/\D/g, '');

    }



    function rb3IsoRange() {

        var s = (document.getElementById('rb3StartDate') || {}).value || '';

        var e = (document.getElementById('rb3EndDate') || {}).value || '';

        return { start: s, end: e };

    }



    function apiUrl(path) {

        if (typeof pbjApiUrl === 'function') return pbjApiUrl(path);

        return path;

    }



    function csvCell(v) {

        if (v == null) return '';

        var s = String(v);

        if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';

        return s;

    }



    function pushRow(out, section, recordType, period, metric, value, unit, definition, source) {

        out.push([

            section, recordType, period, metric, value, unit, definition, source

        ].map(csvCell).join(','));

    }



    function collectRb3Context() {

        if (typeof window.pbjRb3CollectAiPackContext === 'function') {

            try {

                return window.pbjRb3CollectAiPackContext();

            } catch (e0) { /* ignore */ }

        }

        return { key_dates: [], user_items: [] };

    }



    function hourFields(row) {

        return [

            'Total_Nurse_Hours', 'Total_Staff_Hours',

            'Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_RN', 'Hrs_OthRN',

            'Hrs_LPNadmin', 'Hrs_LPN', 'Hrs_CNA', 'Hrs_NAtrn', 'Hrs_MedAide',

            'Hrs_RN_ctr', 'Hrs_LPN_ctr', 'Hrs_CNA_ctr',

            'Hrs_Admin', 'Hrs_MedDir', 'Hrs_PT', 'Hrs_OT', 'Hrs_NP'

        ].map(function (key) {

            return [key, row[key], 'hours'];

        });

    }



    function buildSimplePackCsv(rows, range) {

        var hdr = 'section,record_type,period,metric,value,unit,definition,source';

        var out = [hdr];

        var label = (range.start && range.end) ? (range.start + ' to ' + range.end) : 'selected period';

        pushRow(out, 'META', 'filter', label, 'date_range', range.start + '..' + range.end, 'iso', 'Case Builder review window', 'PBJ320');

        var focus = (document.getElementById('pbjAiPromptFocusDates') || {}).value || '';

        if (focus.trim()) {

            pushRow(out, 'CONTEXT', 'focus', '', 'focus_dates', focus.trim(), '', 'User focus dates for AI review', 'PBJ320');

        }

        var ctx = collectRb3Context();

        (ctx.key_dates || []).forEach(function (row, idx) {

            var period = row.date || String(idx + 1);

            if (row.type) pushRow(out, 'KEY_DATE', 'row', period, 'type', row.type, '', '', 'Case Builder');

            if (row.note) pushRow(out, 'KEY_DATE', 'row', period, 'note', row.note, '', '', 'Case Builder');

            if (row.date) pushRow(out, 'KEY_DATE', 'row', period, 'date', row.date, 'iso', '', 'Case Builder');

        });

        (ctx.user_items || []).forEach(function (item, idx) {

            var period = String(idx + 1);

            if (item.title) pushRow(out, 'INCIDENT', 'item', period, 'title', item.title, '', '', item.source || 'user');

            if (item.subtitle) pushRow(out, 'INCIDENT', 'item', period, 'detail', item.subtitle, '', '', item.source || 'user');

        });

        (rows || []).forEach(function (row) {

            var wd = row.WorkDate || row.workdate || row.date || '';

            var period = String(wd).slice(0, 10);

            hourFields(row).forEach(function (triple) {

                if (triple[1] == null || triple[1] === '') return;

                pushRow(out, 'DAILY', 'hours', period, triple[0], triple[1], triple[2], 'CMS PBJ job-level hours', 'CMS PBJ');

            });

            [

                ['hprd', 'Total_Staff_HPRD', row.Total_Staff_HPRD, 'HPRD'],

                ['hprd', 'Nurse_Staff_HPRD_Excl_Admin', row.Nurse_Staff_HPRD_Excl_Admin, 'HPRD'],

                ['hprd', 'Total_RN_HPRD', row.Total_RN_HPRD, 'HPRD'],

                ['hprd', 'Total_LPN_HPRD', row.Total_LPN_HPRD, 'HPRD'],

                ['hprd', 'Total_Nurse_Aide_HPRD', row.Total_Nurse_Aide_HPRD, 'HPRD'],

                ['census', 'MDScensus', row.MDScensus, 'residents']

            ].forEach(function (pair) {

                if (pair[2] == null || pair[2] === '') return;

                pushRow(out, 'DAILY', pair[0], period, pair[1], pair[2], pair[3], 'CMS PBJ derived', 'CMS PBJ');

            });

        });

        return out.join('\n') + '\n';

    }



    function fetchDailyRows(range) {

        var q = '?start_date=' + encodeURIComponent(range.start) + '&end_date=' + encodeURIComponent(range.end);

        return fetch(apiUrl('/api/data' + q), { credentials: 'include' })

            .then(function (r) { return r.json(); })

            .then(function (payload) {

                if (payload && Array.isArray(payload.data)) return payload.data;

                if (payload && Array.isArray(payload.records)) return payload.records;

                return [];

            });

    }



    window.exportPbj320AiContextPack = function exportPbj320AiContextPackRb3() {
        var range = rb3IsoRange();
        if (!range.start || !range.end) return;
        var previewOnly = !!window.__pbjAiPackPreviewOnly;
        fetchDailyRows(range).then(function (rows) {
            window.currentData = rows;
            var ccn = metaCcn() || 'facility';
            var csv = buildSimplePackCsv(rows, range);
            window.__pbjLastAiContextPackCsv = csv;
            if (previewOnly) {
                if (typeof window.pbjV2RenderAiPackPreview === 'function') {
                    window.pbjV2RenderAiPackPreview({
                        sectionCounts: { DAILY: rows.length, META: 1 },
                        sampleLines: csv.split('\n').slice(1, 20),
                        filterLabel: range.start + ' to ' + range.end,
                        dailyRowCount: rows.length,
                        dailyMetricRows: rows.length,
                        quarters: [],
                        facilityName: '',
                        ccn: ccn,
                    });
                }
                return;
            }
            if (window.__pbjAiPackWantClipboard && navigator.clipboard) {
                navigator.clipboard.writeText(csv).catch(function () {});
                return;
            }
            triggerCsvDownload('pbj320_ai_context_' + ccn + '.csv', csv);
        }).catch(function () { /* ignore */ });
    };



    function triggerCsvDownload(name, csv) {

        var blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });

        var url = URL.createObjectURL(blob);

        var a = document.createElement('a');

        a.href = url;

        a.download = name;

        document.body.appendChild(a);

        a.click();

        a.remove();

        URL.revokeObjectURL(url);

    }



    window.pbjRb3DownloadAiContextPackCsv = function () {

        window.__pbjAiPackPreviewOnly = false;

        window.exportPbj320AiContextPack();

    };



    window.pbjRb3CopyAiStarterPrompt = function () {

        if (typeof window.pbjV2CopyAiStarterPrompt === 'function') {

            window.pbjV2CopyAiStarterPrompt();

            return;

        }

        var range = rb3IsoRange();

        var ccn = metaCcn();

        var focus = (document.getElementById('pbjAiPromptFocusDates') || {}).value || '';

        var text = [

            'You are reviewing CMS Payroll-Based Journal (PBJ) staffing for a single nursing home.',

            'Facility CCN: ' + (ccn || '(unknown)'),

            'Period: ' + (range.start || '?') + ' to ' + (range.end || '?')

        ];

        if (focus.trim()) {

            text.push('Dates of interest: ' + focus.trim() + '.');

        }

        text.push(

            '',

            'Use only the attached export. Separate what the data shows from what it may suggest.',

            'Flag RN coverage, state minimum shortfalls, acuity gaps, and unusual daily patterns when supported.',

            'State what the file cannot prove and what records would help verify.'

        );

        var joined = text.join('\n');

        if (navigator.clipboard && navigator.clipboard.writeText) {

            navigator.clipboard.writeText(joined);

        }

    };



    var ccn = metaCcn();

    if (ccn) window.PBJ320_EXPORT_CCN = ccn;

})();

