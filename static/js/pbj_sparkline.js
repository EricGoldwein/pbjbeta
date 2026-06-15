/**
 * Premium v2 — 8-quarter trend mini charts (SVG). UI copy: "8-quarter trend" / "trend" only.
 */
(function (root, factory) {
    'use strict';
    var api = factory();
    if (typeof module === 'object' && module.exports) {
        module.exports = api;
    }
    root.PBJSparkline = api;
})(typeof globalThis !== 'undefined' ? globalThis : globalThis, function () {
    'use strict';

    var DEFAULT_COUNT = 8;
    var MIN_VALID_POINTS = 3;
    var DEFAULT_WIDTH = 100;
    var DEFAULT_HEIGHT = 28;
    var PAD_X = 3;
    var PAD_Y = 3;

    function isValidNumber(v) {
        if (v === null || v === undefined || v === '') {
            return false;
        }
        var n = Number(v);
        return !isNaN(n) && isFinite(n);
    }

    function normCy(q) {
        var s = String(q || '')
            .trim()
            .toUpperCase()
            .replace(/^CY/i, '');
        var m = s.match(/^(\d{4})Q([1-4])$/);
        return m ? m[1] + 'Q' + m[2] : '';
    }

    function quarterSortKey(cy) {
        var m = String(cy || '').match(/^(\d{4})Q([1-4])$/);
        if (!m) {
            return null;
        }
        return parseInt(m[1], 10) * 10 + parseInt(m[2], 10);
    }

    function sortQuartersChrono(keys) {
        return (keys || [])
            .map(function (k) {
                return normCy(k) || String(k || '').trim();
            })
            .filter(Boolean)
            .sort(function (a, b) {
                var ka = quarterSortKey(a);
                var kb = quarterSortKey(b);
                if (ka == null && kb == null) {
                    return String(a).localeCompare(String(b));
                }
                if (ka == null) {
                    return 1;
                }
                if (kb == null) {
                    return -1;
                }
                return ka - kb;
            });
    }

    /**
     * @param {string[]} allQuarterKeys
     * @param {string} endQuarter
     * @param {number} count
     * @returns {string[]}
     */
    function lastNQuarterKeys(allQuarterKeys, endQuarter, count) {
        var n = count == null ? DEFAULT_COUNT : count;
        var end = normCy(endQuarter);
        var sorted = sortQuartersChrono(allQuarterKeys);
        if (!sorted.length) {
            return [];
        }
        if (!end) {
            return sorted.slice(-n);
        }
        var idx = -1;
        for (var i = 0; i < sorted.length; i++) {
            if (sorted[i] === end) {
                idx = i;
            }
        }
        if (idx < 0) {
            return sorted.slice(-n);
        }
        var start = Math.max(0, idx - n + 1);
        return sorted.slice(start, idx + 1);
    }

    /**
     * @param {Object} dataByQuarter
     * @param {function(Object):*} pickValue
     * @param {string} endQuarter
     * @param {number} count
     */
    function buildSeriesFromQuarters(dataByQuarter, pickValue, endQuarter, count) {
        dataByQuarter = dataByQuarter || {};
        var keys = Object.keys(dataByQuarter);
        var quarters = lastNQuarterKeys(keys, endQuarter, count);
        var values = quarters.map(function (q) {
            var row = dataByQuarter[q];
            if (!row && dataByQuarter['CY' + q]) {
                row = dataByQuarter['CY' + q];
            }
            if (!row) {
                return null;
            }
            var v = pickValue(row, q);
            return isValidNumber(v) ? Number(v) : null;
        });
        return { quarters: quarters, values: values };
    }

    function countValid(values) {
        var n = 0;
        for (var i = 0; i < (values || []).length; i++) {
            if (isValidNumber(values[i])) {
                n++;
            }
        }
        return n;
    }

    function domainFromValues(values, sharedDomain) {
        if (sharedDomain && sharedDomain.length === 2) {
            var lo = Number(sharedDomain[0]);
            var hi = Number(sharedDomain[1]);
            if (!isNaN(lo) && !isNaN(hi) && hi > lo) {
                return [lo, hi];
            }
        }
        var min = Infinity;
        var max = -Infinity;
        (values || []).forEach(function (v) {
            if (!isValidNumber(v)) {
                return;
            }
            var n = Number(v);
            if (n < min) {
                min = n;
            }
            if (n > max) {
                max = n;
            }
        });
        if (!isFinite(min) || !isFinite(max)) {
            return [0, 1];
        }
        if (min === max) {
            var pad = Math.abs(min) * 0.05 || 0.1;
            return [min - pad, max + pad];
        }
        var margin = (max - min) * 0.08;
        return [min - margin, max + margin];
    }

    function formatValue(v, decimals) {
        var d = decimals == null ? 2 : decimals;
        var n = Number(v);
        if (isNaN(n)) {
            return '—';
        }
        return n.toFixed(d);
    }

    function formatQuarterLabel(q, formatter) {
        if (typeof formatter === 'function') {
            return formatter(q);
        }
        var m = String(q || '').match(/^(\d{4})Q([1-4])$/);
        return m ? 'Q' + m[2] + ' ' + m[1] : String(q || '');
    }

    function sparklineGrainLabel(grain) {
        if (grain === 'year') {
            return 'year';
        }
        if (grain === 'month') {
            return 'month';
        }
        if (grain === 'day') {
            return 'day';
        }
        if (grain === 'quarter_bucket') {
            return 'period';
        }
        return 'quarter';
    }

    function buildA11ySummary(periods, values, decimals, grain, trendCount) {
        var parts = [];
        for (var i = 0; i < values.length; i++) {
            parts.push(isValidNumber(values[i]) ? formatValue(values[i], decimals) : '—');
        }
        var g = sparklineGrainLabel(grain || 'quarter');
        var n =
            trendCount != null && !isNaN(parseInt(trendCount, 10))
                ? parseInt(trendCount, 10)
                : (periods || []).length;
        return n + '-' + g + ' trend: ' + parts.join(', ') + '.';
    }

    function trendLineClass(values) {
        var first = null;
        var last = null;
        (values || []).forEach(function (v) {
            if (!isValidNumber(v)) {
                return;
            }
            var n = Number(v);
            if (first == null) {
                first = n;
            }
            last = n;
        });
        if (first == null || last == null) {
            return 'pbj-spark-trend-line';
        }
        if (last > first + 1e-9) {
            return 'pbj-spark-trend-line pbj-spark-trend-line--up';
        }
        if (last < first - 1e-9) {
            return 'pbj-spark-trend-line pbj-spark-trend-line--down';
        }
        return 'pbj-spark-trend-line pbj-spark-trend-line--flat';
    }

    function coordsForSeries(values, width, height, domain, threshold) {
        var w = width - PAD_X * 2;
        var h = height - PAD_Y * 2;
        var yMin = domain[0];
        var yMax = domain[1];
        var ySpan = yMax - yMin || 1;
        var n = values.length;
        var segments = [];
        var points = [];
        var thresholdY = null;
        if (threshold != null && isValidNumber(threshold)) {
            thresholdY = PAD_Y + h - ((Number(threshold) - yMin) / ySpan) * h;
        }
        for (var i = 0; i < n; i++) {
            var v = values[i];
            var x = n <= 1 ? PAD_X + w / 2 : PAD_X + (i / (n - 1)) * w;
            if (!isValidNumber(v)) {
                points.push({ x: x, y: null, index: i, value: null });
                continue;
            }
            var y = PAD_Y + h - ((Number(v) - yMin) / ySpan) * h;
            points.push({ x: x, y: y, index: i, value: Number(v) });
            var seg = segments[segments.length - 1];
            if (!seg || seg.closed) {
                segments.push({ d: 'M' + x.toFixed(2) + ' ' + y.toFixed(2), closed: false });
            } else {
                seg.d += ' L' + x.toFixed(2) + ' ' + y.toFixed(2);
            }
        }
        for (var j = 0; j < points.length; j++) {
            if (!isValidNumber(values[j])) {
                if (segments.length && !segments[segments.length - 1].closed) {
                    segments[segments.length - 1].closed = true;
                }
            }
        }
        return { segments: segments, points: points, thresholdY: thresholdY };
    }

    function escAttr(s) {
        return String(s || '')
            .replace(/&/g, '&amp;')
            .replace(/"/g, '&quot;')
            .replace(/</g, '&lt;');
    }

    /**
     * @param {HTMLElement} mount
     * @param {Object} opts
     */
    function render(mount, opts) {
        opts = opts || {};
        if (!mount) {
            return;
        }
        var values = opts.values || [];
        var quarters = opts.quarters || [];
        var selectedQuarter = normCy(opts.selectedQuarter);
        var metricId = opts.metricId || 'metric';
        var width = opts.width == null ? DEFAULT_WIDTH : opts.width;
        var height = opts.height == null ? DEFAULT_HEIGHT : opts.height;
        var decimals = opts.decimals == null ? 2 : opts.decimals;
        var threshold = opts.threshold;
        var sharedDomain = opts.sharedDomain;
        var quarterFormatter = opts.quarterFormatter;

        mount.innerHTML = '';
        mount.classList.add('pbj-spark-trend-host');

        var trendCount = opts.trendCount != null ? opts.trendCount : (quarters || []).length;
        var minValid =
            opts.minValidPoints != null
                ? opts.minValidPoints
                : trendCount != null && trendCount < MIN_VALID_POINTS
                  ? trendCount
                  : MIN_VALID_POINTS;

        if (countValid(values) < minValid) {
            mount.classList.add('pbj-spark-trend-host--limited');
            mount.innerHTML =
                '<span class="pbj-spark-trend-limited small text-muted">limited history</span>';
            mount.setAttribute('role', 'img');
            mount.setAttribute(
                'aria-label',
                trendCount +
                    '-' +
                    sparklineGrainLabel(opts.grain) +
                    ' trend: limited history'
            );
            return;
        }

        mount.classList.remove('pbj-spark-trend-host--limited');
        var domain = domainFromValues(values, sharedDomain);
        var geom = coordsForSeries(values, width, height, domain, threshold);
        var selectedIdx = -1;
        var priorIdx = -1;
        var grain = opts.grain || 'quarter';
        for (var i = 0; i < quarters.length; i++) {
            var periodMatch =
                grain === 'quarter'
                    ? normCy(quarters[i]) === selectedQuarter
                    : String(quarters[i]) === String(selectedQuarter);
            if (periodMatch) {
                selectedIdx = i;
            }
        }
        if (selectedIdx > 0) {
            for (var p = selectedIdx - 1; p >= 0; p--) {
                if (isValidNumber(values[p])) {
                    priorIdx = p;
                    break;
                }
            }
        }

        var svg =
            '<svg class="pbj-spark-trend-svg" width="' +
            width +
            '" height="' +
            height +
            '" viewBox="0 0 ' +
            width +
            ' ' +
            height +
            '" focusable="true" role="img" aria-label="' +
            escAttr(buildA11ySummary(quarters, values, decimals, opts.grain, trendCount)) +
            '">';
        if (geom.thresholdY != null) {
            svg +=
                '<line class="pbj-spark-trend-threshold" x1="' +
                PAD_X +
                '" y1="' +
                geom.thresholdY.toFixed(2) +
                '" x2="' +
                (width - PAD_X) +
                '" y2="' +
                geom.thresholdY.toFixed(2) +
                '" />';
        }
        var lineCls = trendLineClass(values);
        geom.segments.forEach(function (seg) {
            svg += '<path class="' + lineCls + '" fill="none" d="' + seg.d + '" />';
        });
        geom.points.forEach(function (pt) {
            if (pt.y == null) {
                return;
            }
            var cls = 'pbj-spark-trend-dot';
            var dotR = '1.5';
            if (pt.index === selectedIdx) {
                cls += ' pbj-spark-trend-dot--selected';
                dotR = '3';
            } else if (pt.index === priorIdx) {
                cls += ' pbj-spark-trend-dot--prior';
                dotR = '2';
            }
            var qLabel = formatQuarterLabel(quarters[pt.index], quarterFormatter);
            var tip = qLabel + ' · ' + formatValue(pt.value, decimals);
            var fillPart =
                pt.index === selectedIdx || pt.index === priorIdx ? '' : ' fill="#64748b"';
            svg +=
                '<circle class="' +
                cls +
                '" cx="' +
                pt.x.toFixed(2) +
                '" cy="' +
                pt.y.toFixed(2) +
                '" r="' +
                dotR +
                '"' +
                fillPart +
                ' data-pbj-spark-tip="' +
                escAttr(tip) +
                '"><title>' +
                escAttr(tip) +
                '</title></circle>';
        });
        svg += '</svg>';

        mount.innerHTML = svg;
        mount.setAttribute(
            'title',
            trendCount + '-' + sparklineGrainLabel(opts.grain) + ' trend'
        );
        var svgEl = mount.querySelector('svg');
        if (svgEl) {
            svgEl.setAttribute('data-metric-id', metricId);
        }
    }

    function pickQuarterlyField(key) {
        return function (row) {
            if (!row) {
                return null;
            }
            if (row[key] != null) {
                return row[key];
            }
            var alt = {
                total_hprd: row.Total_HPRD,
                direct_hprd: row.Direct_Care_HPRD,
                total_rn_hprd: row.Total_RN_HPRD,
                total_nurse_aide_hprd: row.Total_Nurse_Aide_HPRD,
                census: row.Census,
            };
            return alt[key] != null ? alt[key] : null;
        };
    }

    function filterDataByAllowlist(data, allowlist) {
        if (!allowlist || !allowlist.length) {
            return data;
        }
        var allowed = {};
        allowlist.forEach(function (q) {
            var n = normCy(q);
            if (n) {
                allowed[n] = true;
            }
        });
        var out = {};
        Object.keys(data || {}).forEach(function (k) {
            var n = normCy(k);
            if (n && allowed[n]) {
                out[k] = data[k];
            }
        });
        return out;
    }

    function selectedQuarterGlobal() {
        if (typeof window !== 'undefined') {
            if (window.__pbjSparklineEndQuarter && String(window.__pbjSparklineEndQuarter).trim()) {
                return normCy(window.__pbjSparklineEndQuarter);
            }
            var q = window.__pbjProviderInfoMatchQuarter;
            if (q && String(q).trim()) {
                return normCy(q);
            }
            var keys = Object.keys(window.__pbjQuarterlyDataByQuarter || {});
            var sorted = sortQuartersChrono(keys);
            if (sorted.length) {
                return sorted[sorted.length - 1];
            }
        }
        return '';
    }

    function quarterFormatterGlobal() {
        if (typeof window !== 'undefined' && typeof window.pbjV2FormatCyQuarter === 'function') {
            return window.pbjV2FormatCyQuarter;
        }
        return null;
    }

    function summarySparklineGrain() {
        if (typeof window !== 'undefined' && window.__pbjSparklineGrain) {
            return window.__pbjSparklineGrain;
        }
        return 'quarter';
    }

    function summarySparklineCount() {
        if (typeof window !== 'undefined' && window.__pbjSparklineDisabled) {
            return 0;
        }
        if (typeof window !== 'undefined' && window.__pbjSparklineCount != null) {
            var n = parseInt(window.__pbjSparklineCount, 10);
            if (!isNaN(n) && n >= 0) {
                return n;
            }
        }
        return 4;
    }

    function clearSparklineMount(mount) {
        if (!mount) {
            return;
        }
        mount.innerHTML = '';
        mount.classList.remove('pbj-spark-trend-host', 'pbj-spark-trend-host--limited');
        mount.removeAttribute('aria-label');
        mount.removeAttribute('title');
    }

    function summarySparklineEndPeriod(fallbackQuarter) {
        if (typeof window !== 'undefined' && window.__pbjSparklineEndPeriod) {
            return String(window.__pbjSparklineEndPeriod).trim();
        }
        return fallbackQuarter || '';
    }

    function yearSortKey(y) {
        var n = parseInt(String(y || ''), 10);
        return isNaN(n) ? null : n;
    }

    function monthSortKey(m) {
        var s = String(m || '').trim();
        if (!/^\d{4}-\d{2}$/.test(s)) {
            return null;
        }
        return parseInt(s.replace('-', ''), 10);
    }

    function daySortKey(d) {
        var s = String(d || '').trim().slice(0, 10);
        if (!/^\d{4}-\d{2}-\d{2}$/.test(s)) {
            return null;
        }
        return parseInt(s.replace(/-/g, ''), 10);
    }

    function sortKeysChrono(keys, sortKeyFn) {
        return (keys || [])
            .map(function (k) {
                return String(k || '').trim();
            })
            .filter(Boolean)
            .sort(function (a, b) {
                var ka = sortKeyFn(a);
                var kb = sortKeyFn(b);
                if (ka == null && kb == null) {
                    return String(a).localeCompare(String(b));
                }
                if (ka == null) {
                    return 1;
                }
                if (kb == null) {
                    return -1;
                }
                return ka - kb;
            });
    }

    function lastNPeriodKeys(allKeys, endKey, count, sortKeyFn) {
        var n = count == null ? DEFAULT_COUNT : count;
        var end = String(endKey || '').trim();
        var sorted = sortKeysChrono(allKeys, sortKeyFn);
        if (!sorted.length) {
            return [];
        }
        if (!end) {
            return sorted.slice(-n);
        }
        var idx = -1;
        for (var i = 0; i < sorted.length; i++) {
            if (sorted[i] === end) {
                idx = i;
            }
        }
        if (idx < 0) {
            return sorted.slice(-n);
        }
        var start = Math.max(0, idx - n + 1);
        return sorted.slice(start, idx + 1);
    }

    function evenBuckets(sortedKeys, bucketCount) {
        var keys = sortedKeys || [];
        var n = keys.length;
        var bc = Math.max(1, Math.min(bucketCount == null ? DEFAULT_COUNT : bucketCount, n));
        if (!n) {
            return [];
        }
        if (n <= bc) {
            return keys.map(function (k) {
                return { keys: [k] };
            });
        }
        var buckets = [];
        var base = Math.floor(n / bc);
        var extra = n % bc;
        var idx = 0;
        for (var b = 0; b < bc; b++) {
            var size = base + (b < extra ? 1 : 0);
            buckets.push({ keys: keys.slice(idx, idx + size) });
            idx += size;
        }
        return buckets;
    }

    function bucketQuarterLabel(keys, quarterFormatter) {
        if (!keys || !keys.length) {
            return '';
        }
        if (keys.length === 1) {
            return formatQuarterLabel(keys[0], quarterFormatter);
        }
        var first = normCy(keys[0]);
        var last = normCy(keys[keys.length - 1]);
        var fm = first.match(/^(\d{4})Q([1-4])$/);
        var lm = last.match(/^(\d{4})Q([1-4])$/);
        if (fm && lm) {
            if (fm[1] === lm[1]) {
                return 'Q' + fm[2] + '–Q' + lm[2] + ' ' + fm[1];
            }
            return (
                formatQuarterLabel(first, quarterFormatter) +
                '–' +
                formatQuarterLabel(last, quarterFormatter)
            );
        }
        return formatQuarterLabel(first, quarterFormatter) + '–' + formatQuarterLabel(last, quarterFormatter);
    }

    function resolveQuarterRow(dataByQuarter, q) {
        var row = dataByQuarter[q];
        if (!row && dataByQuarter['CY' + q]) {
            row = dataByQuarter['CY' + q];
        }
        return row;
    }

    function quartersInAllowlistSpan(dataByQuarter, allowlist) {
        if (allowlist && allowlist.length) {
            var seen = {};
            var spanKeys = sortQuartersChrono(
                allowlist
                    .map(function (q) {
                        return normCy(q) || String(q || '').trim();
                    })
                    .filter(Boolean)
            ).filter(function (k) {
                if (seen[k]) {
                    return false;
                }
                seen[k] = true;
                return true;
            });
            if (spanKeys.length) {
                return spanKeys;
            }
        }
        return sortQuartersChrono(Object.keys(dataByQuarter || {}));
    }

    function avgPickInBucket(dataByQuarter, pickValue, qKeys) {
        var vals = [];
        (qKeys || []).forEach(function (q) {
            var row = resolveQuarterRow(dataByQuarter, q);
            if (!row) {
                return;
            }
            var v = pickValue(row, q);
            if (isValidNumber(v)) {
                vals.push(Number(v));
            }
        });
        if (!vals.length) {
            return null;
        }
        return vals.reduce(function (a, b) {
            return a + b;
        }, 0) / vals.length;
    }

    function summarySparklineUseFullSpan() {
        return !(
            typeof window !== 'undefined' &&
            window.__pbjSparklineUseFullSpan === false
        );
    }

    function summarySparklineBucketed() {
        return !!(typeof window !== 'undefined' && window.__pbjSparklineBucketed);
    }

    /**
     * Build quarter sparkline points across the active filter span (not trailing-only),
     * bucketing into `count` groups when the span is wider than the target point count.
     */
    function buildSeriesFromQuartersSpan(
        dataByQuarter,
        pickValue,
        endQuarter,
        count,
        allowlist,
        useFullSpan,
        bucketed,
        quarterFormatter
    ) {
        dataByQuarter = dataByQuarter || {};
        var spanQs = quartersInAllowlistSpan(dataByQuarter, allowlist);
        if (!spanQs.length) {
            return { periods: [], values: [], grain: 'quarter' };
        }
        var n = count == null ? DEFAULT_COUNT : count;
        if (!useFullSpan) {
            var legacy = buildSeriesFromQuarters(dataByQuarter, pickValue, endQuarter, n);
            return { periods: legacy.quarters, values: legacy.values, grain: 'quarter' };
        }
        var qsWithData = spanQs.filter(function (q) {
            var row = resolveQuarterRow(dataByQuarter, q);
            if (!row) {
                return false;
            }
            return isValidNumber(pickValue(row, q));
        });
        // Trailing-only fallback: only when no explicit filter span (avoid misleading 2025-only on multi-year filters).
        if ((!allowlist || !allowlist.length) && qsWithData.length && qsWithData.length < spanQs.length) {
            var firstDataK = quarterSortKey(qsWithData[0]);
            var firstSpanK = quarterSortKey(spanQs[0]);
            if (firstDataK != null && firstSpanK != null && firstDataK > firstSpanK) {
                var emptyLeading = spanQs.filter(function (q) {
                    var k = quarterSortKey(q);
                    return k != null && firstDataK != null && k < firstDataK;
                }).length;
                if (emptyLeading >= Math.ceil(spanQs.length / 2) && qsWithData.length <= n) {
                    var trail = buildSeriesFromQuarters(
                        dataByQuarter,
                        pickValue,
                        qsWithData[qsWithData.length - 1],
                        Math.min(n, qsWithData.length)
                    );
                    return { periods: trail.quarters, values: trail.values, grain: 'quarter' };
                }
            }
        }
        if (spanQs.length <= n && !bucketed) {
            var periods = spanQs;
            var values = periods.map(function (q) {
                var row = resolveQuarterRow(dataByQuarter, q);
                if (!row) {
                    return null;
                }
                var v = pickValue(row, q);
                return isValidNumber(v) ? Number(v) : null;
            });
            return { periods: periods, values: values, grain: 'quarter' };
        }
        var buckets = evenBuckets(spanQs, n);
        var bPeriods = [];
        var bValues = [];
        buckets.forEach(function (b) {
            bPeriods.push(bucketQuarterLabel(b.keys, quarterFormatter));
            bValues.push(avgPickInBucket(dataByQuarter, pickValue, b.keys));
        });
        return { periods: bPeriods, values: bValues, grain: 'quarter_bucket' };
    }

    function buildSeriesFromMapSpan(dataByKey, pickValue, endKey, count, sortKeyFn, allowlistKeys, useFullSpan, bucketed, labelFn) {
        dataByKey = dataByKey || {};
        var keys = sortKeysChrono(Object.keys(dataByKey), sortKeyFn);
        if (allowlistKeys && allowlistKeys.length) {
            var allowed = {};
            allowlistKeys.forEach(function (k) {
                allowed[String(k).trim()] = true;
            });
            keys = keys.filter(function (k) {
                return allowed[k];
            });
        }
        if (!keys.length) {
            return { periods: [], values: [] };
        }
        var n = count == null ? DEFAULT_COUNT : count;
        if (!useFullSpan || (!bucketed && keys.length <= n)) {
            var periods = !useFullSpan || keys.length > n ? lastNPeriodKeys(keys, endKey, n, sortKeyFn) : keys;
            var values = periods.map(function (k) {
                var row = dataByKey[k];
                if (!row) {
                    return null;
                }
                var v = pickValue(row, k);
                return isValidNumber(v) ? Number(v) : null;
            });
            return { periods: periods, values: values };
        }
        var buckets = evenBuckets(keys, n);
        var bPeriods = [];
        var bValues = [];
        buckets.forEach(function (b) {
            bPeriods.push(
                typeof labelFn === 'function' ? labelFn(b.keys) : bucketQuarterLabel(b.keys, null)
            );
            var vals = [];
            b.keys.forEach(function (k) {
                var row = dataByKey[k];
                if (!row) {
                    return;
                }
                var v = pickValue(row, k);
                if (isValidNumber(v)) {
                    vals.push(Number(v));
                }
            });
            bValues.push(
                vals.length
                    ? vals.reduce(function (a, c) {
                          return a + c;
                      }, 0) / vals.length
                    : null
            );
        });
        return { periods: bPeriods, values: bValues };
    }

    function buildSeriesFromMap(dataByKey, pickValue, endKey, count, sortKeyFn) {
        dataByKey = dataByKey || {};
        var keys = Object.keys(dataByKey);
        var periods = lastNPeriodKeys(keys, endKey, count, sortKeyFn);
        var values = periods.map(function (k) {
            var row = dataByKey[k];
            if (!row) {
                return null;
            }
            var v = pickValue(row, k);
            return isValidNumber(v) ? Number(v) : null;
        });
        return { periods: periods, values: values };
    }

    function aggregateQuarterlyByYear(dataByQuarter, fieldNames) {
        var sums = {};
        var counts = {};
        Object.keys(dataByQuarter || {}).forEach(function (k) {
            var cy = normCy(k);
            var m = cy.match(/^(\d{4})Q[1-4]$/);
            if (!m) {
                return;
            }
            var y = m[1];
            var row = dataByQuarter[k];
            if (!row) {
                return;
            }
            var v = null;
            for (var fi = 0; fi < fieldNames.length; fi += 1) {
                var raw = row[fieldNames[fi]];
                if (raw != null && raw !== '' && !isNaN(parseFloat(raw))) {
                    v = parseFloat(raw);
                    break;
                }
            }
            if (!isValidNumber(v)) {
                return;
            }
            sums[y] = (sums[y] || 0) + Number(v);
            counts[y] = (counts[y] || 0) + 1;
        });
        var out = {};
        Object.keys(sums).forEach(function (y) {
            if (counts[y]) {
                out[y] = sums[y] / counts[y];
            }
        });
        return out;
    }

    var _dailyMonthCacheKey = null;
    var _dailyMonthCache = null;

    function aggregateDailyByDay(rows) {
        var out = {};
        (rows || []).forEach(function (r) {
            if (!r) {
                return;
            }
            var iso = String(r.WorkDate || '').trim().slice(0, 10);
            if (!/^\d{4}-\d{2}-\d{2}$/.test(iso)) {
                return;
            }
            var cen = parseFloat(r.MDScensus);
            if (!(cen > 0)) {
                return;
            }
            var totalNurseHrs = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, PBJ_TOTAL_NURSE_HOUR_FIELDS)
                : null;
            var rnHrs = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, PBJ_TOTAL_RN_HOUR_FIELDS)
                : null;
            var hrsAide = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, ['Hrs_CNA', 'Hrs_NAtrn', 'Hrs_MedAide'])
                : null;
            if (totalNurseHrs === null || rnHrs === null || hrsAide === null) {
                return;
            }
            var contractHrs = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, PBJ_CONTRACT_HOUR_FIELDS)
                : null;
            out[iso] = {
                total_hprd: totalNurseHrs / cen,
                total_rn_hprd: rnHrs / cen,
                total_nurse_aide_hprd: hrsAide / cen,
                contract_pct: contractHrs !== null && totalNurseHrs > 0 ? (contractHrs / totalNurseHrs) * 100 : null,
            };
        });
        return out;
    }

    function workDateToCyQuarter(iso) {
        var s = String(iso || '').trim().slice(0, 10);
        var m = s.match(/^(\d{4})-(\d{2})/);
        if (!m) {
            return '';
        }
        var y = parseInt(m[1], 10);
        var mo = parseInt(m[2], 10);
        if (isNaN(y) || isNaN(mo) || mo < 1 || mo > 12) {
            return '';
        }
        return y + 'Q' + Math.ceil(mo / 3);
    }

    function aggregateDailyByQuarter(rows) {
        var buckets = {};
        (rows || []).forEach(function (r) {
            if (!r) {
                return;
            }
            var cy = workDateToCyQuarter(r.WorkDate);
            if (!cy) {
                return;
            }
            var cen = parseFloat(r.MDScensus);
            if (!(cen > 0)) {
                return;
            }
            if (!buckets[cy]) {
                buckets[cy] = {
                    residentDays: 0,
                    totalNurseHrs: 0,
                    rnHrs: 0,
                    aideHrs: 0,
                    contractHrs: 0,
                };
            }
            var b = buckets[cy];
            b.residentDays += cen;
            var totalNurseHrs = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, PBJ_TOTAL_NURSE_HOUR_FIELDS)
                : null;
            var rnHrs = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, PBJ_TOTAL_RN_HOUR_FIELDS)
                : null;
            var hrsAide = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, ['Hrs_CNA', 'Hrs_NAtrn', 'Hrs_MedAide'])
                : null;
            var contractHrs = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, PBJ_CONTRACT_HOUR_FIELDS)
                : null;
            if (totalNurseHrs === null || rnHrs === null || hrsAide === null) {
                return;
            }
            b.totalNurseHrs += totalNurseHrs;
            b.rnHrs += rnHrs;
            b.aideHrs += hrsAide;
            if (contractHrs !== null) {
                b.contractHrs += contractHrs;
            }
        });
        var out = {};
        Object.keys(buckets).forEach(function (cy) {
            var b = buckets[cy];
            if (!(b.residentDays > 0)) {
                return;
            }
            out[cy] = {
                total_hprd: b.totalNurseHrs / b.residentDays,
                total_rn_hprd: b.rnHrs / b.residentDays,
                total_nurse_aide_hprd: b.aideHrs / b.residentDays,
                contract_pct: b.totalNurseHrs > 0 ? (b.contractHrs / b.totalNurseHrs) * 100 : null,
            };
        });
        return out;
    }

    function aggregateDailyByMonth(rows) {
        var rowRef = rows || [];
        var cacheKey =
            typeof window !== 'undefined' && window.__pbjSparklineDailyRows === rowRef
                ? String(window.__pbjSparklineEndPeriod || '') +
                  '|' +
                  String(window.__pbjSparklineGrain || '') +
                  '|' +
                  rowRef.length
                : null;
        if (cacheKey && cacheKey === _dailyMonthCacheKey && _dailyMonthCache) {
            return _dailyMonthCache;
        }
        var buckets = {};
        rowRef.forEach(function (r) {
            if (!r) {
                return;
            }
            var iso = String(r.WorkDate || '').trim().slice(0, 10);
            if (!/^\d{4}-\d{2}-\d{2}$/.test(iso)) {
                return;
            }
            var mk = iso.slice(0, 7);
            var cen = parseFloat(r.MDScensus);
            if (!(cen > 0)) {
                return;
            }
            if (!buckets[mk]) {
                buckets[mk] = {
                    residentDays: 0,
                    totalNurseHrs: 0,
                    rnHrs: 0,
                    contractHrs: 0,
                };
            }
            var b = buckets[mk];
            b.residentDays += cen;
            var totalNurseHrs = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, PBJ_TOTAL_NURSE_HOUR_FIELDS)
                : null;
            var rnHrs = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, PBJ_TOTAL_RN_HOUR_FIELDS)
                : null;
            var contractHrs = typeof pbjSumHourFieldsOrNull === 'function'
                ? pbjSumHourFieldsOrNull(r, PBJ_CONTRACT_HOUR_FIELDS)
                : null;
            if (totalNurseHrs === null || rnHrs === null) {
                return;
            }
            b.totalNurseHrs += totalNurseHrs;
            b.rnHrs += rnHrs;
            if (contractHrs !== null) {
                b.contractHrs += contractHrs;
            }
        });
        var out = {};
        Object.keys(buckets).forEach(function (mk) {
            var b = buckets[mk];
            if (!(b.residentDays > 0)) {
                return;
            }
            out[mk] = {
                total_hprd: b.totalNurseHrs / b.residentDays,
                total_rn_hprd: b.rnHrs / b.residentDays,
                contract_pct: b.totalNurseHrs > 0 ? (b.contractHrs / b.totalNurseHrs) * 100 : null,
            };
        });
        if (cacheKey) {
            _dailyMonthCacheKey = cacheKey;
            _dailyMonthCache = out;
        }
        return out;
    }

    function formatPeriodLabel(period, grain, quarterFormatter) {
        if (grain === 'year') {
            return String(period || '');
        }
        if (grain === 'day') {
            var parts = String(period || '').split('-');
            if (parts.length === 3) {
                var monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                var mi = parseInt(parts[1], 10) - 1;
                var di = parseInt(parts[2], 10);
                if (mi >= 0 && mi < 12 && !isNaN(di)) {
                    return monthNames[mi] + ' ' + di;
                }
            }
            return String(period || '');
        }
        if (grain === 'month') {
            var parts = String(period || '').split('-');
            if (parts.length === 2) {
                var monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
                var mi = parseInt(parts[1], 10) - 1;
                if (mi >= 0 && mi < 12) {
                    return monthNames[mi] + ' ' + parts[0];
                }
            }
            return String(period || '');
        }
        return formatQuarterLabel(period, quarterFormatter);
    }

    function buildSummaryPodSeries(data, spec, grain, endPeriod, dailyRows, count, allowlist) {
        var n = count == null ? DEFAULT_COUNT : count;
        var useFullSpan = summarySparklineUseFullSpan();
        var bucketed = summarySparklineBucketed();
        var qFmt = quarterFormatterGlobal();
        if (grain === 'day') {
            var dData = aggregateDailyByDay(dailyRows);
            if (useFullSpan && bucketed) {
                var dayKeys = sortKeysChrono(Object.keys(dData), daySortKey);
                var dayBuilt = buildSeriesFromMapSpan(
                    dData,
                    pickQuarterlyField(spec.key),
                    endPeriod || '',
                    n,
                    daySortKey,
                    dayKeys,
                    useFullSpan,
                    bucketed,
                    function (keys) {
                        if (!keys || !keys.length) {
                            return '';
                        }
                        if (keys.length === 1) {
                            return formatPeriodLabel(keys[0], 'day', qFmt);
                        }
                        return (
                            formatPeriodLabel(keys[0], 'day', qFmt) +
                            '–' +
                            formatPeriodLabel(keys[keys.length - 1], 'day', qFmt)
                        );
                    }
                );
                return { periods: dayBuilt.periods, values: dayBuilt.values, grain: 'quarter_bucket' };
            }
            return buildSeriesFromMap(
                dData,
                pickQuarterlyField(spec.key),
                endPeriod || '',
                n,
                daySortKey
            );
        }
        if (grain === 'year') {
            var yearFields = {
                total_hprd: ['total_hprd', 'Total_HPRD'],
                total_rn_hprd: ['total_rn_hprd', 'Total_RN_HPRD'],
                contract_pct: ['contract_pct', 'total_contract_pct', 'Total_Contract_Pct'],
                total_nurse_aide_hprd: ['total_nurse_aide_hprd', 'Total_Nurse_Aide_HPRD'],
            };
            var yData = aggregateQuarterlyByYear(data, yearFields[spec.key] || [spec.key]);
            return buildSeriesFromMap(
                yData,
                function (row) {
                    return row;
                },
                endPeriod || '',
                n,
                yearSortKey
            );
        }
        if (grain === 'month') {
            var mData = aggregateDailyByMonth(dailyRows);
            if (useFullSpan && bucketed) {
                var monthKeys = sortKeysChrono(Object.keys(mData), monthSortKey);
                var monthBuilt = buildSeriesFromMapSpan(
                    mData,
                    pickQuarterlyField(spec.key),
                    endPeriod || '',
                    n,
                    monthSortKey,
                    monthKeys,
                    useFullSpan,
                    bucketed,
                    function (keys) {
                        if (!keys || !keys.length) {
                            return '';
                        }
                        if (keys.length === 1) {
                            return formatPeriodLabel(keys[0], 'month', qFmt);
                        }
                        return (
                            formatPeriodLabel(keys[0], 'month', qFmt) +
                            '–' +
                            formatPeriodLabel(keys[keys.length - 1], 'month', qFmt)
                        );
                    }
                );
                return { periods: monthBuilt.periods, values: monthBuilt.values, grain: 'quarter_bucket' };
            }
            return buildSeriesFromMap(
                mData,
                pickQuarterlyField(spec.key),
                endPeriod || '',
                n,
                monthSortKey
            );
        }
        var endQ = endPeriod || selectedQuarterGlobal();
        var quarterData = data;
        if (dailyRows && dailyRows.length) {
            var fromDaily = aggregateDailyByQuarter(dailyRows);
            if (fromDaily && Object.keys(fromDaily).length) {
                quarterData = fromDaily;
            }
        }
        return buildSeriesFromQuartersSpan(
            quarterData,
            pickQuarterlyField(spec.key),
            endQ,
            n,
            allowlist,
            useFullSpan,
            bucketed,
            qFmt
        );
    }

    function summaryPodSparklineWidth(mount) {
        if (!mount || typeof mount.getBoundingClientRect !== 'function') {
            return 120;
        }
        var w = mount.getBoundingClientRect().width;
        if (!w || w < 48) {
            return 120;
        }
        return Math.round(Math.min(w, 168));
    }

    function refreshSummaryPods() {
        if (typeof document === 'undefined') {
            return;
        }
        var data = (typeof window !== 'undefined' && window.__pbjQuarterlyDataByQuarter) || {};
        var allowlist =
            typeof window !== 'undefined' && window.__pbjSparklineQuarterAllowlist
                ? window.__pbjSparklineQuarterAllowlist
                : null;
        var endQ = selectedQuarterGlobal();
        var grain = summarySparklineGrain();
        var count = summarySparklineCount();
        var endPeriod = summarySparklineEndPeriod(endQ);
        var dailyRows =
            (typeof window !== 'undefined' && window.__pbjSparklineDailyRows) || [];
        var qFmt = quarterFormatterGlobal();
        var cmData = filterDataByAllowlist(
            (typeof window !== 'undefined' && window.__pbjCaseMixDataByQuarter) || {},
            allowlist
        );
        var selectedPeriod =
            grain === 'year' || grain === 'day' || grain === 'month' ? endPeriod : endQ;
        var minValid =
            count > 0 && count < MIN_VALID_POINTS ? count : count > 0 ? MIN_VALID_POINTS : 0;

        var specs = [
            { id: 'pbjSparkTrendTotalHprd', key: 'total_hprd', decimals: 2 },
            { id: 'pbjSparkTrendRnHprd', key: 'total_rn_hprd', decimals: 2 },
            { id: 'pbjSparkTrendContract', key: 'contract_pct', decimals: 1 },
            { id: 'pbjSparkTrendNaHprd', key: 'total_nurse_aide_hprd', decimals: 2 },
        ];
        var charts =
            typeof window !== 'undefined' && window.lastChartsResponse && window.lastChartsResponse.charts
                ? window.lastChartsResponse.charts
                : null;
        var usedChartTrends =
            charts &&
            typeof window !== 'undefined' &&
            typeof window.pbjV2RefreshPbjTrendsContextStrips === 'function';
        if (usedChartTrends) {
            window.pbjV2RefreshPbjTrendsContextStrips(charts);
        }
        specs.forEach(function (spec) {
            var el = document.getElementById(spec.id);
            if (!el) {
                return;
            }
            if (usedChartTrends) {
                return;
            }
            if (!count || (typeof window !== 'undefined' && window.__pbjSparklineDisabled)) {
                clearSparklineMount(el);
                return;
            }
            if (el.closest('.d-none')) {
                return;
            }
            var built = buildSummaryPodSeries(data, spec, grain, endPeriod, dailyRows, count, allowlist);
            var renderGrain = built.grain || grain;
            if (
                typeof window !== 'undefined' &&
                typeof window.pbjV2RenderCensusContextSparkline === 'function'
            ) {
                var censusOpts =
                    spec.key === 'contract_pct'
                        ? { allowZero: true, minSpan: 0.05, flatEpsilon: 0.02 }
                        : {};
                window.pbjV2RenderCensusContextSparkline(el, built.values, censusOpts);
                return;
            }
            render(el, {
                values: built.values,
                quarters: built.periods,
                selectedQuarter: selectedPeriod,
                metricId: spec.key,
                width: summaryPodSparklineWidth(el),
                height: 22,
                decimals: spec.decimals,
                quarterFormatter: qFmt,
                grain: renderGrain,
                trendCount: count,
                minValidPoints: minValid,
            });
        });

        var cmEl = document.getElementById('pbjSparkTrendCaseMix');
        if (cmEl) {
            var cmBuilt = buildSeriesFromQuartersSpan(
                cmData,
                function (row) {
                    return row && row.pct_cmi_total != null ? row.pct_cmi_total : null;
                },
                endQ,
                count,
                allowlist,
                summarySparklineUseFullSpan(),
                summarySparklineBucketed(),
                qFmt
            );
            render(cmEl, {
                values: cmBuilt.values,
                quarters: cmBuilt.periods,
                selectedQuarter: endQ,
                metricId: 'pct_cmi_total',
                width: 100,
                height: 28,
                decimals: 1,
                threshold: 100,
                quarterFormatter: qFmt,
                grain: cmBuilt.grain || grain,
                trendCount: count,
            });
        }

        var harEl = document.getElementById('pbjSparkTrendHarrington');
        if (harEl && typeof window !== 'undefined') {
            var har = window.__lastHarringtonCmiExport;
            var useTotal =
                document.querySelector('input[name="harringtonUseTotalMode"]:checked') &&
                document.querySelector('input[name="harringtonUseTotalMode"]:checked').value === 'true';
            var harRows = har && Array.isArray(har.rows) ? har.rows : [];
            var harByQ = {};
            harRows.forEach(function (r) {
                var k = normCy(r.quarter);
                if (k) {
                    harByQ[k] = r;
                }
            });
            var pctKey = useTotal ? 'pct_harrington_total_staff' : 'pct_harrington_direct_staff';
            var harBuilt = buildSeriesFromQuartersSpan(
                harByQ,
                function (row) {
                    if (!row) {
                        return null;
                    }
                    var v = row[pctKey];
                    if (v == null && useTotal) {
                        v = row.pct_total;
                    }
                    if (v == null && !useTotal) {
                        v = row.pct_harrington_direct_staff;
                    }
                    return v;
                },
                endQ,
                count,
                allowlist,
                summarySparklineUseFullSpan(),
                summarySparklineBucketed(),
                qFmt
            );
            render(harEl, {
                values: harBuilt.values,
                quarters: harBuilt.periods,
                selectedQuarter: endQ,
                metricId: 'harrington_pct',
                width: 100,
                height: 28,
                decimals: 1,
                threshold: 100,
                quarterFormatter: qFmt,
                grain: harBuilt.grain || grain,
                trendCount: count,
            });
        }
    }

    var QUARTERLY_TREND_METRICS = [
        { key: 'total_hprd', decimals: 2 },
        { key: 'direct_hprd', decimals: 2 },
        { key: 'total_rn_hprd', decimals: 2 },
        { key: 'total_nurse_aide_hprd', decimals: 2 },
    ];

    function refreshQuarterlyStaffingTable() {
        if (typeof document === 'undefined') {
            return;
        }
        var table = document.getElementById('quarterlyDataTable');
        if (!table) {
            return;
        }
        var data = (typeof window !== 'undefined' && window.__pbjQuarterlyDataByQuarter) || {};
        var qFmt = quarterFormatterGlobal();
        var globalEnd = selectedQuarterGlobal();
        var domains = {};
        QUARTERLY_TREND_METRICS.forEach(function (m) {
            var pool = [];
            var rows = table.querySelectorAll('tbody tr');
            rows.forEach(function (tr) {
                var qCell = tr.querySelector('td.quarter-cell');
                var q = qCell ? qCell.getAttribute('data-quarter') || '' : '';
                var series = buildSeriesFromQuarters(data, pickQuarterlyField(m.key), normCy(q) || globalEnd, DEFAULT_COUNT);
                series.values.forEach(function (v) {
                    if (isValidNumber(v)) {
                        pool.push(Number(v));
                    }
                });
            });
            domains[m.key] = domainFromValues(pool, null);
        });

        var rows = table.querySelectorAll('tbody tr');
        rows.forEach(function (tr) {
            var qTd = tr.querySelector('td.quarter-cell');
            var inlineTrend = tr.querySelector('.pbj-qtr-inline-trend[data-quarter]');
            var rowQ =
                (qTd && qTd.getAttribute('data-quarter')) ||
                (inlineTrend && inlineTrend.getAttribute('data-quarter')) ||
                '';
            var endQ = normCy(rowQ) || globalEnd;
            QUARTERLY_TREND_METRICS.forEach(function (m) {
                var cell = tr.querySelector(
                    '.pbj-qtr-inline-trend[data-pbj-trend-metric="' + m.key + '"]'
                );
                if (!cell) {
                    return;
                }
                var series = buildSeriesFromQuarters(data, pickQuarterlyField(m.key), endQ, DEFAULT_COUNT);
                render(cell, {
                    values: series.values,
                    quarters: series.quarters,
                    selectedQuarter: endQ,
                    metricId: m.key,
                    width: 54,
                    height: 16,
                    decimals: m.decimals,
                    sharedDomain: domains[m.key],
                    quarterFormatter: qFmt,
                });
            });
        });
    }

    function refreshAll() {
        refreshSummaryPods();
        refreshQuarterlyStaffingTable();
    }

    return {
        MIN_VALID_POINTS: MIN_VALID_POINTS,
        DEFAULT_COUNT: DEFAULT_COUNT,
        normCy: normCy,
        quarterSortKey: quarterSortKey,
        sortQuartersChrono: sortQuartersChrono,
        lastNQuarterKeys: lastNQuarterKeys,
        evenBuckets: evenBuckets,
        buildSeriesFromQuartersSpan: buildSeriesFromQuartersSpan,
        buildSeriesFromQuarters: buildSeriesFromQuarters,
        countValid: countValid,
        domainFromValues: domainFromValues,
        buildA11ySummary: buildA11ySummary,
        coordsForSeries: coordsForSeries,
        render: render,
        refreshSummaryPods: refreshSummaryPods,
        refreshQuarterlyStaffingTable: refreshQuarterlyStaffingTable,
        refreshAll: refreshAll,
    };
});
