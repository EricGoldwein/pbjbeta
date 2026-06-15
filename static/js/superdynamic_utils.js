/** Escape text for safe insertion into HTML (attribute or text). Used by superdynamic_dashboard inline script. */
function escapeHtml(t) {
    if (t === null || t === undefined) {
        return '';
    }
    return String(t)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

/** Normalize PBJ WorkDate (string or datetime) to YYYY-MM-DD for APIs and CMS links. */
function pbjNormalizeIsoWorkDate(workDate) {
    var s = String(workDate == null ? '' : workDate).trim();
    var m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
    return m ? m[1] + '-' + m[2] + '-' + m[3] : '';
}

/** Display ISO work date as MM-DD-YYYY for table labels and CMS-facing links. */
function pbjFormatIsoWorkDateUsDashed(isoYmd) {
    var d = pbjNormalizeIsoWorkDate(isoYmd);
    var m = d.match(/^(\d{4})-(\d{2})-(\d{2})$/);
    return m ? m[2] + '-' + m[3] + '-' + m[1] : String(isoYmd == null ? '' : isoYmd).trim();
}

/** CMS data.cms.gov PBJ daily nurse staffing explorer for this CCN + calendar day (WorkDate YYYYMMDD). */
function pbjCmsPbjDailyStaffingExplorerUrl(isoYmd, ccnOverride) {
    var ccnRaw = ccnOverride != null ? String(ccnOverride) : String(PBJ320_EXPORT_CCN || '');
    var ccn = ccnRaw.replace(/\D/g, '').padStart(6, '0');
    var d = pbjNormalizeIsoWorkDate(isoYmd);
    if (!ccn || !d) {
        return null;
    }
    var y = parseInt(d.slice(0, 4), 10);
    var mo = parseInt(d.slice(5, 7), 10);
    if (!y || !mo) {
        return null;
    }
    var q = Math.ceil(mo / 3);
    var slug = 'q' + q + '-' + y;
    var ymd = d.replace(/-/g, '');
    var canonQ = y + 'Q' + q;
    var legacy = pbjCmsDatagovLegacyLowercaseProvWorkdate(canonQ);
    var provCol = legacy ? 'provnum' : 'PROVNUM';
    var wdCol = legacy ? 'workdate' : 'WorkDate';
    var queryObj = {
        filters: {
            list: [
                {
                    conditions: [
                        {
                            column: { value: provCol },
                            comparator: { value: '=' },
                            filterValue: [ccn],
                        },
                        {
                            column: { value: wdCol },
                            comparator: { value: '=' },
                            filterValue: [ymd],
                        },
                    ],
                },
            ],
            rootConjunction: { value: 'AND' },
        },
        keywords: '',
        offset: 0,
        limit: 10,
        sort: { sortBy: null, sortOrder: null },
        columns: [],
    };
    if (y === 2025 && q === 1) {
        return (
            'https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data?query=' +
            encodeURIComponent(JSON.stringify(queryObj))
        );
    }
    return (
        'https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing/data/' +
        slug +
        '?query=' +
        encodeURIComponent(JSON.stringify(queryObj))
    );
}

function pbj320CsvEscapeCell(val) {
    const s = val === null || val === undefined ? '' : String(val);
    if (/[",\n\r]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
}

function pbj320ExportFilename(slug) {
    const d = new Date();
    const y = d.getUTCFullYear();
    const m = String(d.getUTCMonth() + 1).padStart(2, '0');
    const day = String(d.getUTCDate()).padStart(2, '0');
    return `pbj320_${PBJ320_EXPORT_CCN}_${slug}_${y}${m}${day}.csv`;
}

function pbjCyQtrStripCy(raw) {
    let s = String(raw || '').trim().toUpperCase();
    if (s.startsWith('CY')) {
        s = s.slice(2);
    }
    return s;
}

/** Match acuity / filter quarter labels: ``2017Q1`` → ``Q1 2017``. */
function pbjFormatQuarterLikeAcuity(raw) {
    const s = pbjCyQtrStripCy(raw);
    const m = s.match(/^(\d{4})Q([1-4])$/);
    if (m && typeof formatQuarter === 'function') {
        return formatQuarter(m[1] + 'Q' + m[2]);
    }
    return raw || '—';
}

function pbjCitationsAttrEsc(t) {
    return String(t || '')
        .replace(/&/g, '&amp;')
        .replace(/"/g, '&quot;')
        .replace(/</g, '&lt;');
}

/** Display survey dates as MM-DD-YYYY when input is YYYY-MM-DD; sorting still uses raw ISO in row data. */
function pbjFormatSurveyDateForDisplay(raw) {
    const s = String(raw || '').trim();
    if (!s) {
        return '—';
    }
    const m = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
    if (m) {
        return m[2] + '-' + m[3] + '-' + m[1];
    }
    const d = new Date(s);
    if (!isNaN(d.getTime())) {
        const mm = String(d.getMonth() + 1).padStart(2, '0');
        const dd = String(d.getDate()).padStart(2, '0');
        const yyyy = String(d.getFullYear());
        if (yyyy && mm && dd) {
            return mm + '-' + dd + '-' + yyyy;
        }
    }
    return s;
}

/**
 * US federal public holiday name for a calendar date (local), or '' if none.
 * @param {string} iso YYYY-MM-DD
 * @returns {string}
 */
function pbjFederalHolidayNameUs(iso) {
    const m = String(iso || '').match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (!m) {
        return '';
    }
    const y = parseInt(m[1], 10);
    const mo = parseInt(m[2], 10);
    const da = parseInt(m[3], 10);
    const nthWeekday = function (year, monthIndex, weekday, nth) {
        const firstDow = new Date(year, monthIndex, 1).getDay();
        const delta = (7 + weekday - firstDow) % 7 + (nth - 1) * 7;
        return new Date(year, monthIndex, 1 + delta);
    };
    const lastWeekday = function (year, monthIndex, weekday) {
        const last = new Date(year, monthIndex + 1, 0).getDate();
        for (let d = last; d >= 1; d--) {
            const t = new Date(year, monthIndex, d);
            if (t.getDay() === weekday) {
                return t;
            }
        }
        return null;
    };
    const same = function (dt) {
        return dt && dt.getFullYear() === y && dt.getMonth() + 1 === mo && dt.getDate() === da;
    };
    if (mo === 1 && da === 1) {
        return "New Year's Day";
    }
    if (same(nthWeekday(y, 0, 1, 3))) {
        return 'Martin Luther King Jr. Day';
    }
    if (same(nthWeekday(y, 1, 1, 3))) {
        return "Presidents' Day";
    }
    if (same(lastWeekday(y, 4, 1))) {
        return 'Memorial Day';
    }
    if (mo === 6 && da === 19) {
        return 'Juneteenth';
    }
    if (mo === 7 && da === 4) {
        return 'Independence Day';
    }
    if (same(nthWeekday(y, 8, 1, 1))) {
        return 'Labor Day';
    }
    if (same(nthWeekday(y, 9, 1, 2))) {
        return 'Columbus Day / Indigenous Peoples Day';
    }
    if (mo === 11 && da === 11) {
        return 'Veterans Day';
    }
    if (same(nthWeekday(y, 10, 4, 4))) {
        return 'Thanksgiving';
    }
    if (mo === 12 && da === 25) {
        return 'Christmas Day';
    }
    return '';
}

// Half-up rounding for display strings only (returns fixed-decimal string — do not parse back for math).
function roundHalfUpDisplay(val, decimals) {
    if (val === null || val === undefined) return '';
    const n = parseFloat(val);
    if (isNaN(n)) return '';
    const factor = Math.pow(10, decimals);
    return (Math.round(n * factor + 1e-10) / factor).toFixed(decimals);
}

/** Summary / charts: show em dash when API returns null (undefined HPRD), else half-up number. */
function pbjDashOrNumber(val, decimals) {
    if (val === null || val === undefined || val === '') {
        return '—';
    }
    const n = parseFloat(val);
    if (isNaN(n)) {
        return '—';
    }
    return roundHalfUpDisplay(n, decimals);
}

/** Parse numeric scalar; null/undefined/NaN → null (never 0). Use for calculations, not display. */
function parseNumOrNull(val) {
    if (val === null || val === undefined || val === '') {
        return null;
    }
    var n = parseFloat(val);
    return isFinite(n) ? n : null;
}

/**
 * Sum hour fields on one daily row; null when any field is missing (do not treat missing as 0).
 * @param {object} row
 * @param {string[]} fieldNames
 * @returns {number|null}
 */
function pbjSumHourFieldsOrNull(row, fieldNames) {
    if (!row || !fieldNames || !fieldNames.length) {
        return null;
    }
    var sum = 0;
    for (var i = 0; i < fieldNames.length; i++) {
        var v = parseNumOrNull(row[fieldNames[i]]);
        if (v === null) {
            return null;
        }
        sum += v;
    }
    return sum;
}

/** CMS PBJ total nurse hour columns (all positions incl. admin/DON). */
var PBJ_TOTAL_NURSE_HOUR_FIELDS = [
    'Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_RN', 'Hrs_LPNadmin', 'Hrs_LPN',
    'Hrs_CNA', 'Hrs_NAtrn', 'Hrs_MedAide'
];
var PBJ_TOTAL_RN_HOUR_FIELDS = ['Hrs_RNDON', 'Hrs_RNadmin', 'Hrs_RN'];
var PBJ_DIRECT_CARE_HOUR_FIELDS = ['Hrs_RN', 'Hrs_LPN', 'Hrs_CNA', 'Hrs_NAtrn', 'Hrs_MedAide'];
var PBJ_CONTRACT_HOUR_FIELDS = [
    'Hrs_RNDON_ctr', 'Hrs_RNadmin_ctr', 'Hrs_RN_ctr', 'Hrs_LPNadmin_ctr', 'Hrs_LPN_ctr',
    'Hrs_CNA_ctr', 'Hrs_NAtrn_ctr', 'Hrs_MedAide_ctr'
];

/**
 * Pooled HPRD = sum(hours) / sum(census) on days with census > 0 and complete hours.
 * @param {object[]} rows
 * @param {string[]} hourFields
 * @returns {number|null}
 */
function pbjPooledHprdFromRows(rows, hourFields) {
    if (!rows || !rows.length || !hourFields || !hourFields.length) {
        return null;
    }
    var totalHours = 0;
    var totalCensus = 0;
    rows.forEach(function (r) {
        if (!r) {
            return;
        }
        var cen = parseNumOrNull(r.MDScensus);
        if (cen === null || cen <= 0) {
            return;
        }
        var hrs = pbjSumHourFieldsOrNull(r, hourFields);
        if (hrs === null) {
            return;
        }
        totalHours += hrs;
        totalCensus += cen;
    });
    if (totalCensus <= 0) {
        return null;
    }
    return totalHours / totalCensus;
}

/**
 * Contract share = 100 * sum(contract hours) / sum(total nurse hours); null when denominator missing.
 */
function pbjContractSharePctFromRows(rows) {
    if (!rows || !rows.length) {
        return null;
    }
    var contractHrs = 0;
    var totalHrs = 0;
    var hasData = false;
    rows.forEach(function (r) {
        if (!r) {
            return;
        }
        var cen = parseNumOrNull(r.MDScensus);
        if (cen === null || cen <= 0) {
            return;
        }
        var th = pbjSumHourFieldsOrNull(r, PBJ_TOTAL_NURSE_HOUR_FIELDS);
        var ch = pbjSumHourFieldsOrNull(r, PBJ_CONTRACT_HOUR_FIELDS);
        if (th === null || ch === null) {
            return;
        }
        totalHrs += th;
        contractHrs += ch;
        hasData = true;
    });
    if (!hasData || totalHrs <= 0) {
        return null;
    }
    return (contractHrs / totalHrs) * 100;
}

/** Compliance share = days meeting / observed days; null when no observed days. */
function pbjComplianceSharePct(daysMeeting, observedDays) {
    var n = parseNumOrNull(observedDays);
    if (n === null || n <= 0) {
        return null;
    }
    var met = parseNumOrNull(daysMeeting);
    if (met === null) {
        return null;
    }
    return (met / n) * 100;
}

/** Census / hours: half-up rounding plus thousands separators in the integer part. */
function formatDailyCountDisplay(val, decimals) {
    if (val === null || val === undefined) return '';
    const n = parseFloat(val);
    if (isNaN(n)) return '';
    const factor = Math.pow(10, decimals);
    const rounded = Math.round(n * factor + 1e-10) / factor;
    return rounded.toLocaleString('en-US', {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

/** Employee Detail / EIN hours: always two decimal places (e.g. 0.50). */
function einFmtHours2(v, useDash) {
    if (v === null || v === undefined || v === '') {
        return useDash ? '—' : '';
    }
    const n = parseFloat(v);
    if (isNaN(n)) {
        return useDash ? '—' : '';
    }
    return roundHalfUpDisplay(n, 2);
}

/** EIN roster Tot column: hours with grouped thousands, one decimal (e.g. 1,234.5). */
function einFmtHours2Comma(v) {
    if (v === null || v === undefined || v === '') {
        return '—';
    }
    const n = parseFloat(v);
    if (isNaN(n)) {
        return '—';
    }
    const r = Math.round(n * 10 + 1e-10) / 10;
    return r.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
}

function pbjNormalizeCyQuarterToCanon(qVal) {
    const m = String(qVal || '').trim().match(/^(?:CY)?(\d{4})Q([1-4])$/i);
    return m ? (m[1] + 'Q' + m[2]) : null;
}

function pbjEinApiQuarterFromFilterValue(qVal) {
    const c = pbjNormalizeCyQuarterToCanon(qVal);
    return c ? ('CY' + c) : '';
}

function pbjFirstDayOfCyQuarterCanon(canon) {
    const m = String(canon || '').match(/^(\d{4})Q([1-4])$/i);
    if (!m) {
        return null;
    }
    const y = parseInt(m[1], 10);
    const qi = parseInt(m[2], 10);
    const month = (qi - 1) * 3 + 1;
    return y + '-' + String(month).padStart(2, '0') + '-01';
}

function einIsoWeekdayAbbrev(iso) {
    if (!iso || !/^\d{4}-\d{2}-\d{2}$/.test(String(iso))) {
        return '';
    }
    const d = new Date(iso + 'T12:00:00');
    return ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.getDay()];
}

function formatCyQuarterLabel(q) {
    if (q === null || q === undefined || q === '') { return ''; }
    const m = String(q).match(/(?:CY)?(\d{4})Q([1-4])/i);
    if (m) { return 'Q' + m[2] + ' ' + m[1]; }
    return String(q);
}

function einDateLabelShort(iso, withYy) {
    if (!iso) { return ''; }
    return formatEinTableDateShort(iso, !!withYy);
}

/** US-style full date for career span rows (MM-DD-YYYY). */
function einDateMMDDYYYY(iso) {
    if (!iso) { return ''; }
    const m = String(iso).match(/^(\d{4})-(\d{2})-(\d{2})$/);
    if (m) { return m[2] + '-' + m[3] + '-' + m[1]; }
    return String(iso);
}

function formatEinInt(n) {
    if (n === null || n === undefined || n === '') {
        return '';
    }
    var x = typeof n === 'number' ? n : parseInt(String(n), 10);
    if (isNaN(x)) {
        return String(n);
    }
    return x.toLocaleString('en-US');
}

function einMmYyyyFromIso(iso) {
    if (!iso || typeof iso !== 'string') {
        return '';
    }
    var p = String(iso).trim().split(/[-T]/);
    if (p.length < 2) {
        return '';
    }
    var y = p[0];
    var mo = p[1];
    if (!/^\d{4}$/.test(y) || !/^\d{2}$/.test(mo)) {
        return '';
    }
    return mo + '/' + y;
}

/** Lazy-load Plotly once before chart rendering (removed from synchronous head). */
(function (global) {
    'use strict';
    var DEFAULT_PLOTLY_URL = 'https://cdn.plot.ly/plotly-2.27.0.min.js';
    var plotlyLoadPromise = null;

    function pbjPlotlyLazyUrl() {
        try {
            var el = document.getElementById('pbj-plotly-lazy-script');
            if (el && el.textContent) {
                var cfg = JSON.parse(el.textContent);
                if (cfg && cfg.url) {
                    return String(cfg.url);
                }
            }
        } catch (ePlotlyCfg) {
            /* ignore */
        }
        return DEFAULT_PLOTLY_URL;
    }

    global.pbjEnsurePlotly = function pbjEnsurePlotly() {
        if (global.Plotly && typeof global.Plotly.newPlot === 'function') {
            return Promise.resolve(global.Plotly);
        }
        if (plotlyLoadPromise) {
            return plotlyLoadPromise;
        }
        plotlyLoadPromise = new Promise(function (resolve, reject) {
            var existing = document.querySelector('script[data-pbj-plotly="1"]');
            if (existing) {
                if (global.Plotly && typeof global.Plotly.newPlot === 'function') {
                    resolve(global.Plotly);
                    return;
                }
                existing.addEventListener('load', function () {
                    if (global.Plotly && typeof global.Plotly.newPlot === 'function') {
                        resolve(global.Plotly);
                    } else {
                        plotlyLoadPromise = null;
                        reject(new Error('Plotly unavailable after script load'));
                    }
                });
                existing.addEventListener('error', function () {
                    plotlyLoadPromise = null;
                    reject(new Error('Plotly script failed to load'));
                });
                return;
            }
            var script = document.createElement('script');
            script.src = pbjPlotlyLazyUrl();
            script.async = true;
            script.setAttribute('data-pbj-plotly', '1');
            script.onload = function () {
                if (global.Plotly && typeof global.Plotly.newPlot === 'function') {
                    resolve(global.Plotly);
                } else {
                    plotlyLoadPromise = null;
                    reject(new Error('Plotly unavailable after script load'));
                }
            };
            script.onerror = function () {
                plotlyLoadPromise = null;
                reject(new Error('Plotly script failed to load'));
            };
            document.head.appendChild(script);
        });
        return plotlyLoadPromise;
    };
})(typeof window !== 'undefined' ? window : this);
