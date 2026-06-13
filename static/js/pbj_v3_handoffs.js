/**
 * PBJ320 V3 — data-aware bottom-of-pane "Continue analysis" handoff cards.
 * Progressive enhancement only; never triggers heavy API loads for personalization.
 */
(function (global) {
    'use strict';

    var LOADED_PREFIX = '__pbjV3PaneLoaded_';

    function v3Enabled() {
        return !!global.__pbjV3PanesActive;
    }

    function el(id) {
        return document.getElementById(id);
    }

    function textOf(node) {
        if (!node) {
            return '';
        }
        return String(node.textContent || '').replace(/\s+/g, ' ').trim();
    }

    function isDash(val) {
        return !val || val === '—' || val === '-';
    }

    function paneScopeKey() {
        if (typeof global.pbjEinHeadcountScopeSignature === 'function') {
            try {
                return global.pbjEinHeadcountScopeSignature();
            } catch (e) {
                /* ignore */
            }
        }
        return String(global.lastFilters || '');
    }

    function paneLoadedForScope(pane) {
        return global[LOADED_PREFIX + pane] === paneScopeKey();
    }

    function setSummary(paneKey, sentence) {
        var cap = paneKey.charAt(0).toUpperCase() + paneKey.slice(1);
        var node = el('pbjV3Handoff' + cap + 'Summary');
        if (!node) {
            return;
        }
        var s = String(sentence || '').trim();
        node.textContent = s || node.getAttribute('data-pbj-fallback') || '';
    }

    function storeFallbacks() {
        document.querySelectorAll('[data-pbj-v3-handoff]').forEach(function (card) {
            var sum = card.querySelector('.pbj-v3-handoff-summary');
            if (sum && !sum.getAttribute('data-pbj-fallback')) {
                sum.setAttribute('data-pbj-fallback', sum.textContent.trim());
            }
        });
    }

    function parseBenchGap(gapEl) {
        if (!gapEl || isDash(textOf(gapEl))) {
            return null;
        }
        var numEl = gapEl.querySelector('.pbj-bench-gap-num');
        var dirEl = gapEl.querySelector('.pbj-bench-gap-dir');
        var num = numEl ? textOf(numEl) : '';
        var dir = dirEl ? textOf(dirEl).toLowerCase() : '';
        if (!num) {
            return null;
        }
        if (!dir) {
            var n = parseFloat(String(num).replace(/[^\d.+-]/g, ''));
            if (isNaN(n)) {
                return null;
            }
            if (Math.abs(n) <= 0.02) {
                dir = 'at';
            } else {
                dir = n > 0 ? 'above' : 'below';
            }
            num = String(Math.abs(n));
        }
        return { num: num, dir: dir };
    }

    function gapSentence(gap, benchmarkLabel) {
        if (!gap) {
            return '';
        }
        if (gap.dir === 'at') {
            return 'Reported staffing matched the ' + benchmarkLabel + ' for the selected period.';
        }
        return (
            'Reported staffing was ' +
            gap.num +
            ' HPRD ' +
            gap.dir +
            ' the ' +
            benchmarkLabel +
            '.'
        );
    }

    function periodLabelShort() {
        var pill = textOf(el('pbjSummaryScopePill'));
        if (!pill) {
            return '';
        }
        var m = pill.match(/Selected period:\s*(.+?)(?:\s*·|$)/i);
        if (m) {
            return m[1].trim();
        }
        return pill.replace(/^Selected period:\s*/i, '').split('·')[0].trim();
    }

    function readDirectCareHprd() {
        var totalDisp = el('pbjSummaryTotalHprdDisplay');
        if (totalDisp) {
            var tip = totalDisp.getAttribute('title') || '';
            var tm = tip.match(/Direct care HPRD:\s*([\d.]+)/i);
            if (tm) {
                return tm[1];
            }
        }
        var td = el('totalDirectHPRD');
        var t = textOf(td);
        return isDash(t) ? '' : t;
    }

    function readOverviewFacts() {
        var total = textOf(el('pbjSummaryTotalHprdDisplay'));
        var direct = readDirectCareHprd();
        var contract = textOf(el('contractPct'));
        var cmGap = parseBenchGap(el('forensicCaseMixDelta'));
        var period = periodLabelShort();
        return { total: total, direct: direct, contract: contract, cmGap: cmGap, period: period };
    }

    function pickOverviewSummary() {
        var f = readOverviewFacts();
        if (f.period && !isDash(f.total) && f.direct) {
            return (
                'For ' +
                f.period +
                ', this facility averaged ' +
                f.total +
                ' total nurse HPRD and ' +
                f.direct +
                ' direct-care HPRD.'
            );
        }
        var cm = gapSentence(f.cmGap, 'case-mix benchmark');
        if (cm) {
            return cm;
        }
        if (!isDash(f.contract)) {
            return 'Contract share was ' + f.contract + ' of direct nurse hours.';
        }
        return '';
    }

    function geoPeerBlockVisible() {
        var block = el('forensicGeoPeerSummaryBlock');
        return block && !block.classList.contains('d-none');
    }

    function harringtonRowVisible() {
        var row = el('forensicHarringtonDeltaRow');
        return row && !row.classList.contains('d-none');
    }

    function pickBenchmarksSummary() {
        var cm = gapSentence(parseBenchGap(el('forensicCaseMixDelta')), 'case-mix benchmark');
        if (cm) {
            return cm;
        }
        if (harringtonRowVisible()) {
            var har = gapSentence(parseBenchGap(el('forensicHarringtonDelta')), 'Harrington expected staffing');
            if (har) {
                return har;
            }
        }
        if (geoPeerBlockVisible()) {
            var county = parseBenchGap(el('forensicGeoPeerCountyDelta'));
            var state = parseBenchGap(el('forensicGeoPeerStateDelta'));
            if (county && state) {
                return (
                    'Reported staffing was ' +
                    county.num +
                    ' HPRD ' +
                    county.dir +
                    ' the county median and ' +
                    state.num +
                    ' HPRD ' +
                    state.dir +
                    ' the state median.'
                );
            }
            if (county) {
                return gapSentence(county, 'county median');
            }
            if (state) {
                return gapSentence(state, 'state median');
            }
        }
        return '';
    }

    function workforcePaneReady() {
        return paneLoadedForScope('workforce');
    }

    function dailyRowCount() {
        if (!workforcePaneReady()) {
            return null;
        }
        var tbody = el('dataTableBody');
        if (!tbody) {
            return null;
        }
        var rows = tbody.querySelectorAll('tr[data-work-date]');
        if (rows.length) {
            return rows.length;
        }
        var msg = textOf(tbody.querySelector('tr td'));
        if (/no daily rows/i.test(msg)) {
            return 0;
        }
        return null;
    }

    function lowestDowDay() {
        if (!workforcePaneReady()) {
            return '';
        }
        var body = el('dowRollupTableBody');
        if (!body) {
            return '';
        }
        var trs = body.querySelectorAll('tr');
        var best = null;
        var bestVal = Infinity;
        trs.forEach(function (tr) {
            var cells = tr.querySelectorAll('td');
            if (cells.length < 2) {
                return;
            }
            var day = textOf(cells[0]);
            var val = parseFloat(textOf(cells[1]).replace(/,/g, ''));
            if (!day || isNaN(val)) {
                return;
            }
            if (val < bestVal) {
                bestVal = val;
                best = day;
            }
        });
        return best || '';
    }

    function rosterLoaded() {
        return workforcePaneReady() && !!global.__pbjLazyLoaded_einHeadcountChart;
    }

    function pickWorkforceSummary() {
        if (!workforcePaneReady()) {
            return '';
        }
        var n = dailyRowCount();
        if (n != null && n > 0) {
            return 'Daily PBJ table contains ' + n.toLocaleString('en-US') + ' rows for the selected period.';
        }
        if (rosterLoaded()) {
            return 'Roster/headcount data is loaded for the selected scope.';
        }
        var dow = lowestDowDay();
        if (dow) {
            return dow + ' was the lowest-staffed day of week.';
        }
        return '';
    }

    function pickRiskSummary() {
        if (!paneLoadedForScope('risk')) {
            return '';
        }
        var events = global.__pbjV3RiskTimelineAllEvents;
        if (Array.isArray(events) && events.length) {
            return 'Risk Timeline shows ' + events.length + ' facility-history events.';
        }
        var meta = global.__pbjCitationsSummaryMeta || {};
        if (meta.total != null && meta.total > 0) {
            return 'The citations panel includes ' + meta.total + ' citation rows.';
        }
        var citBody = el('pbjCitationsTbody');
        if (citBody) {
            var citRows = citBody.querySelectorAll('tr').length;
            if (citRows > 0) {
                return 'The citations panel includes ' + citRows + ' citation rows.';
            }
        }
        var stats = el('pbjV3RiskInspectionStats');
        if (stats) {
            var firstVal = stats.querySelector('.fw-semibold');
            var label = stats.querySelector('.text-muted');
            if (firstVal && label && /last survey/i.test(textOf(label))) {
                var d = textOf(firstVal);
                if (!isDash(d)) {
                    return 'Most recent survey: ' + d + '.';
                }
            }
        }
        if (Array.isArray(events)) {
            var own = events.filter(function (ev) {
                return ev.type === 'ownership' || ev.type === 'chow';
            });
            if (own.length) {
                return 'Ownership/CHOW events are present for this facility.';
            }
        }
        return '';
    }

    function pbjV3UpdateOverviewHandoff() {
        if (!v3Enabled()) {
            return;
        }
        setSummary('overview', pickOverviewSummary());
    }

    function pbjV3UpdateBenchmarksHandoff() {
        if (!v3Enabled()) {
            return;
        }
        setSummary('benchmarks', pickBenchmarksSummary());
    }

    function pbjV3UpdateWorkforceHandoff() {
        if (!v3Enabled()) {
            return;
        }
        setSummary('workforce', pickWorkforceSummary());
    }

    function pbjV3UpdateRiskHandoff() {
        if (!v3Enabled()) {
            return;
        }
        setSummary('risk', pickRiskSummary());
    }

    function pbjV3RenderHandoffCards() {
        if (!v3Enabled()) {
            return;
        }
        pbjV3UpdateOverviewHandoff();
        pbjV3UpdateBenchmarksHandoff();
        pbjV3UpdateWorkforceHandoff();
        pbjV3UpdateRiskHandoff();
    }

    function openCaseBuilder() {
        if (global.__pbjReportBuilderV3Href) {
            global.location.href = String(global.__pbjReportBuilderV3Href);
            return;
        }
        if (typeof global.pbjSwitchTopTab === 'function') {
            global.pbjSwitchTopTab('reportBuilder');
        }
    }

    function handleAction(action) {
        switch (action) {
            case 'overview':
            case 'benchmarks':
            case 'workforce':
            case 'risk':
                if (typeof global.pbjV3SwitchPane === 'function') {
                    global.pbjV3SwitchPane(action);
                }
                break;
            case 'caseBuilder':
                openCaseBuilder();
                break;
            default:
                break;
        }
    }

    function wireButtonsOnce() {
        if (global.__pbjV3HandoffsWired) {
            return;
        }
        global.__pbjV3HandoffsWired = true;
        document.addEventListener('click', function (ev) {
            var btn = ev.target.closest('[data-pbj-v3-handoff-action]');
            if (!btn || !btn.closest('[data-pbj-v3-handoff]')) {
                return;
            }
            ev.preventDefault();
            handleAction(btn.getAttribute('data-pbj-v3-handoff-action'));
        });
    }

    function init() {
        if (!v3Enabled()) {
            return;
        }
        storeFallbacks();
        wireButtonsOnce();
        pbjV3RenderHandoffCards();
    }

    global.pbjV3RenderHandoffCards = pbjV3RenderHandoffCards;
    global.pbjV3UpdateOverviewHandoff = pbjV3UpdateOverviewHandoff;
    global.pbjV3UpdateBenchmarksHandoff = pbjV3UpdateBenchmarksHandoff;
    global.pbjV3UpdateWorkforceHandoff = pbjV3UpdateWorkforceHandoff;
    global.pbjV3UpdateRiskHandoff = pbjV3UpdateRiskHandoff;
    global.pbjV3InitHandoffs = init;

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})(typeof window !== 'undefined' ? window : globalThis);
