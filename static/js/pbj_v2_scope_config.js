/**
 * v2 — shared period scope state labels (Control Center, charts, daily table).
 */
(function (global) {
    'use strict';

    function formatIsoShort(iso) {
        var p = String(iso || '').trim().split('-');
        if (p.length !== 3) {
            return iso || '';
        }
        return p[1] + '/' + p[2] + '/' + p[0];
    }

    function formatIsoCompact(iso) {
        var p = String(iso || '').trim().split('-');
        if (p.length !== 3) {
            return iso || '';
        }
        return p[1] + '/' + p[2] + '/' + String(p[0]).slice(-2);
    }

    function formatCyQuarter(q) {
        var s = String(q || '').trim();
        var m = s.match(/^(\d{4})Q([1-4])$/i);
        if (m) {
            return 'Q' + m[2] + ' ' + m[1];
        }
        return s;
    }

    function cyQuarterSortKey(q) {
        var parts = String(q || '').trim().split('Q');
        if (parts.length !== 2 || !/^\d{4}$/.test(parts[0]) || !/^[1-4]$/.test(parts[1])) {
            return 0;
        }
        return parseInt(parts[0], 10) * 4 + parseInt(parts[1], 10);
    }

    function compactQuarterScopeLabel(text) {
        if (typeof global.pbjCompactQuarterScopeLabel === 'function') {
            return global.pbjCompactQuarterScopeLabel(text);
        }
        return String(text || '').trim();
    }

    function isFullCalendarYearQuarters(quarterKeys) {
        if (typeof global.pbjIsFullCalendarYearQuarters === 'function') {
            return global.pbjIsFullCalendarYearQuarters(quarterKeys);
        }
        if (!quarterKeys || quarterKeys.length !== 4) {
            return false;
        }
        var years = {};
        var qnums = [];
        for (var i = 0; i < quarterKeys.length; i++) {
            var q = String(quarterKeys[i] || '').trim();
            var parts = q.split('Q');
            if (parts.length !== 2) {
                return false;
            }
            years[parts[0]] = 1;
            qnums.push(parseInt(parts[1], 10));
        }
        if (Object.keys(years).length !== 1) {
            return false;
        }
        qnums.sort(function (a, b) { return a - b; });
        return qnums[0] === 1 && qnums[1] === 2 && qnums[2] === 3 && qnums[3] === 4;
    }

    function formatYearScopeLabel(yearsSorted) {
        if (!yearsSorted || !yearsSorted.length) {
            return '';
        }
        if (yearsSorted.length === 1) {
            return String(yearsSorted[0]);
        }
        return yearsSorted[0] + ' – ' + yearsSorted[yearsSorted.length - 1];
    }

    /**
     * @param {object} state scope state ({ mode, filterType, quarter, year, ... })
     * @param {object} [opts]
     * @param {string} [opts.allLabel] label when mode === 'all'
     */
    function pbjScopeStateDisplayLabel(state, opts) {
        opts = opts || {};
        if (!state || state.mode === 'all') {
            return opts.allLabel || 'All loaded days';
        }
        var ft = state.filterType || 'quarters';
        if (ft === 'daterange') {
            var s = String(state.startDate || '').trim();
            var e = String(state.endDate || '').trim();
            if (s && e) {
                if (s === e) {
                    return formatIsoCompact(s);
                }
                return formatIsoCompact(s) + ' – ' + formatIsoCompact(e);
            }
            if (s) {
                return 'From ' + formatIsoCompact(s);
            }
            if (e) {
                return 'Through ' + formatIsoCompact(e);
            }
            return 'Custom range';
        }
        if (ft === 'day') {
            var dv = String(state.dayDate || '').trim();
            return dv ? formatIsoCompact(dv) : 'Single day';
        }
        if (ft === 'months') {
            var ms = String(state.startMonth || '').trim();
            var me = String(state.endMonth || '').trim();
            if (ms && me) {
                return ms + ' – ' + me;
            }
            return ms || 'Months';
        }
        if (ft === 'years') {
            var yrCsv = String(state.year || 'all');
            if (!yrCsv || yrCsv === 'all') {
                return opts.allLabel || 'All years';
            }
            var yrs = yrCsv.split(',').map(function (y) {
                return parseInt(String(y).trim(), 10);
            }).filter(function (y) {
                return isFinite(y);
            }).sort(function (a, b) {
                return a - b;
            });
            return formatYearScopeLabel(yrs) || 'Year';
        }
        var qCsv = String(state.quarter || 'all');
        if (!qCsv || qCsv === 'all') {
            return opts.allLabel || 'All quarters';
        }
        var qKeys = qCsv.split(',').map(function (q) { return q.trim(); }).filter(Boolean);
        qKeys.sort(function (a, b) {
            return cyQuarterSortKey(a) - cyQuarterSortKey(b);
        });
        if (isFullCalendarYearQuarters(qKeys)) {
            return qKeys[0].slice(0, 4);
        }
        var qs = qKeys.map(formatCyQuarter).filter(Boolean);
        if (qs.length === 1) {
            return qs[0];
        }
        if (qs.length > 3) {
            return compactQuarterScopeLabel(qs[0] + ' – ' + qs[qs.length - 1]);
        }
        return qs.join(', ');
    }

    function rowCyQuarter(row) {
        if (typeof global.pbj320RowCyQuarter === 'function') {
            return global.pbj320RowCyQuarter(row);
        }
        if (typeof global.pbjNormalizeQuarterToCy === 'function' && row && row.CY_Qtr) {
            return global.pbjNormalizeQuarterToCy(row.CY_Qtr);
        }
        var wd = row && row.WorkDate ? String(row.WorkDate).trim().slice(0, 10) : '';
        var m = wd.match(/^(\d{4})-(\d{2})-/);
        if (!m) {
            return '';
        }
        var month = parseInt(m[2], 10);
        var q = Math.floor((month - 1) / 3) + 1;
        return m[1] + 'Q' + q;
    }

  /** Default audit-period label from loaded daily rows (mode === 'all'). */
    function pbjScopeLabelFromLoadedDailyRows(rows) {
        var src = Array.isArray(rows) ? rows : [];
        if (!src.length) {
            return 'All loaded days';
        }
        var seen = {};
        var qKeys = [];
        var minD = '';
        var maxD = '';
        src.forEach(function (r) {
            var cy = rowCyQuarter(r);
            if (cy && !seen[cy]) {
                seen[cy] = 1;
                qKeys.push(cy);
            }
            var wd = r && r.WorkDate ? String(r.WorkDate).trim().slice(0, 10) : '';
            if (/^\d{4}-\d{2}-\d{2}$/.test(wd)) {
                if (!minD || wd < minD) {
                    minD = wd;
                }
                if (!maxD || wd > maxD) {
                    maxD = wd;
                }
            }
        });
        if (qKeys.length) {
            qKeys.sort(function (a, b) {
                return cyQuarterSortKey(a) - cyQuarterSortKey(b);
            });
            return pbjScopeStateDisplayLabel(
                { mode: 'custom', filterType: 'quarters', quarter: qKeys.join(',') },
                { allLabel: 'All loaded days' }
            );
        }
        if (minD && maxD) {
            if (minD === maxD) {
                return formatIsoCompact(minD);
            }
            return formatIsoCompact(minD) + ' – ' + formatIsoCompact(maxD);
        }
        return 'All loaded days';
    }

    global.pbjScopeStateDisplayLabel = pbjScopeStateDisplayLabel;
    global.pbjScopeLabelFromLoadedDailyRows = pbjScopeLabelFromLoadedDailyRows;
    global.pbjScopeFormatIsoCompact = formatIsoCompact;
})(typeof window !== 'undefined' ? window : globalThis);
