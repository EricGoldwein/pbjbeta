/**
 * Superdynamic v2 — staffing hub layout, scope labels, chart chrome, CHOW/AI helpers.
 */
(function (global) {
    'use strict';

    global.PBJ_V2_MINIMAL_CHART_CHROME = true;

    function pbjV2ClonePlotlyLayout(layout) {
        var lay = layout ? Object.assign({}, layout) : {};
        if (layout && layout.legend) {
            lay.legend = Object.assign({}, layout.legend);
        }
        if (layout && layout.margin) {
            lay.margin = Object.assign({}, layout.margin);
        }
        if (layout && layout.xaxis) {
            lay.xaxis = Object.assign({}, layout.xaxis);
        }
        if (layout && layout._pbjKeepChartAnnotations === true) {
            lay._pbjKeepChartAnnotations = true;
        }
        return lay;
    }

    function pbjPlotlyIsMobileChart() {
        return pbjV2IsMobileViewport();
    }

    function pbjV2IsMobileViewport() {
        try {
            return global.matchMedia('(max-width: 767.98px)').matches;
        } catch (eMob) {
            return typeof global.innerWidth === 'number' && global.innerWidth < 768;
        }
    }

    /** Pick short copy on mobile; full copy on desktop (767.98px breakpoint). */
    function pbjV2MobilePick(desktop, mobile) {
        return pbjV2IsMobileViewport() ? mobile : desktop;
    }

    var PBJ_V2_CITATION_CHIP_LABELS_MOBILE = {
        staffing_sufficiency: 'Staffing',
        licensed_nurse_coverage: 'LPN/RN',
        rn_coverage: 'RN cov.',
        rn_leadership: 'RN lead',
        nurse_aide_staffing: 'Aides',
        staffing_competency: 'Competency',
        staffing_posting: 'Posted',
        abuse_neglect: 'Abuse',
        care_planning: 'Care plan',
        complaint_inspection: 'Complaint',
        falls_accidents: 'Falls',
        infection_control: 'Infection',
        severe_g_plus: 'G+'
    };

    function pbjV2CitationChipLabel(topicId, desktopLabel) {
        if (!pbjV2IsMobileViewport()) {
            return desktopLabel;
        }
        return PBJ_V2_CITATION_CHIP_LABELS_MOBILE[topicId] || desktopLabel;
    }

    function pbjV2EventTypeDisplayLabel(meta) {
        if (!meta) {
            return '';
        }
        return pbjV2MobilePick(meta.label, meta.shortLabel || meta.label);
    }

    /** Extra legend y / bottom margin from x-axis tick rotation (paper y is below plot). */
    function pbjPlotlyTrendLegendClearance(isMobile, tickangle) {
        var mob = !!isMobile;
        var ang = Math.abs(Number(tickangle) || 0);
        if (mob) {
            return { y: -0.38, marginExtra: ang >= 35 ? 12 : 0 };
        }
        if (ang >= 40) {
            return { y: -0.36, marginExtra: 32 };
        }
        if (ang >= 25) {
            return { y: -0.3, marginExtra: 20 };
        }
        if (ang >= 12) {
            return { y: -0.26, marginExtra: 10 };
        }
        return { y: -0.22, marginExtra: 0 };
    }

    /** DOW-style horizontal legend below chart (desktop: no entrywidth — Plotly sizes items naturally). */
    function pbjPlotlyHorizontalLegendBelow(isMobile, overrides) {
        var mob = !!isMobile;
        var base = {
            orientation: 'h',
            yanchor: 'top',
            y: mob ? -0.38 : -0.22,
            xanchor: 'center',
            x: 0.5,
            xref: 'paper',
            yref: 'paper',
            font: { size: mob ? 8 : 10 },
            tracegroupgap: 10,
            itemsizing: 'constant',
            itemwidth: mob ? 32 : 24,
            itemclick: 'toggle',
            itemdoubleclick: 'toggleothers'
        };
        if (mob) {
            base.entrywidth = 0.22;
        }
        var leg = Object.assign(base, overrides || {});
        delete leg.manyItems;
        if (!mob && leg.entrywidth != null && leg.entrywidth < 0.14) {
            delete leg.entrywidth;
        }
        return leg;
    }

    function pbjPlotlyLegendBottomMargin(isMobile, extra) {
        var mob = !!isMobile;
        var bump = Number(extra) || 0;
        return mob ? 248 + bump : 158 + bump;
    }

    function pbjPlotlyApplyHorizontalLegendLayout(layout, isMobile, opts) {
        opts = opts || {};
        if (!layout) {
            return layout;
        }
        layout.showlegend = layout.showlegend !== false;
        layout.legend = pbjPlotlyHorizontalLegendBelow(isMobile, opts.legend || {});
        layout.margin = layout.margin || {};
        layout.margin.b = pbjPlotlyLegendBottomMargin(isMobile, opts.marginExtra || 0);
        if (layout.xaxis) {
            layout.xaxis = Object.assign({}, layout.xaxis, { automargin: true });
        }
        return layout;
    }

    /** After event shapes/annotations, re-apply horizontal legend + bottom clearance (no entrywidth squeeze). */
    function pbjPlotlyReapplyTrendLegendAfterEventShapes(layout, isMobile, opts) {
        opts = opts || {};
        if (!layout || layout.showlegend === false) {
            return layout;
        }
        var shapeCount = (layout.shapes && layout.shapes.length) || 0;
        var annCount = (layout.annotations && layout.annotations.length) || 0;
        if (shapeCount === 0 && annCount === 0 && !opts.force) {
            return layout;
        }
        var tickangle =
            layout.xaxis && layout.xaxis.tickangle != null ? layout.xaxis.tickangle : 0;
        var clearance = pbjPlotlyTrendLegendClearance(isMobile, tickangle);
        var marginExtra =
            (Number(opts.marginExtra) || 0) +
            clearance.marginExtra +
            (annCount > 0 ? Math.min(12, annCount * 4) : 0);
        return pbjPlotlyApplyHorizontalLegendLayout(layout, isMobile, {
            marginExtra: marginExtra,
            legend: Object.assign(
                {
                    y: clearance.y,
                },
                opts.legend || {}
            ),
        });
    }

    var PBJ_PLOTLY_MONTHS = [
        'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
    ];

    function pbjPlotlyParseChartXToIso(x) {
        if (x == null || x === '') {
            return '';
        }
        if (typeof global.pbjChartXToIsoDate === 'function') {
            return global.pbjChartXToIsoDate(x);
        }
        var s = String(x).trim();
        var m = s.match(/^(\d{2})-(\d{2})-(\d{4})$/);
        if (m) {
            return m[3] + '-' + m[1] + '-' + m[2];
        }
        var m2 = s.match(/^(\d{4})-(\d{2})-(\d{2})/);
        if (m2) {
            return m2[1] + '-' + m2[2] + '-' + m2[3];
        }
        return '';
    }

    function pbjPlotlyFormatIsoMmDdYy(iso) {
        var m = String(iso || '').match(/^(\d{4})-(\d{2})-(\d{2})/);
        if (!m) {
            return String(iso || '');
        }
        return m[2] + '-' + m[3] + '-' + m[1].slice(-2);
    }

    function pbjPlotlyMainTraceXValues(traces) {
        var skip = /^(Holidays|_pbjEvents)/;
        for (var i = 0; i < (traces || []).length; i++) {
            var t = traces[i];
            if (!t || !Array.isArray(t.x) || !t.x.length) {
                continue;
            }
            if (t.name && skip.test(String(t.name))) {
                continue;
            }
            return t.x.slice();
        }
        return [];
    }

    /** Smart daily x-axis: month labels for long spans, day labels for short spans. */
    function pbjPlotlySmartXaxisLayout(traces, opts) {
        opts = opts || {};
        var grain = String(opts.grain || opts.view || 'daily').toLowerCase();
        var xVals = opts.xVals || pbjPlotlyMainTraceXValues(traces);
        var nPoints = opts.nPoints != null ? Number(opts.nPoints) : xVals.length;
        if (!nPoints && xVals.length) {
            nPoints = xVals.length;
        }

        if (grain === 'month' || grain === 'monthly') {
            return {
                tickangle: -30,
                nticks: Math.min(12, Math.max(4, nPoints)),
                automargin: true
            };
        }
        if (grain === 'quarter' || grain === 'quarterly' || grain === 'year' || grain === 'annual') {
            return {
                tickangle: grain === 'year' || grain === 'annual' ? 0 : -20,
                nticks: Math.min(12, Math.max(4, nPoints)),
                automargin: true
            };
        }

        if (grain !== 'daily' && grain !== 'day') {
            return { automargin: true };
        }

        var categoryIso = !!opts.categoryIso;
        if (!xVals.length) {
            return { automargin: true };
        }

        if (nPoints <= 45) {
            if (categoryIso) {
                var step = nPoints <= 14 ? 1 : (nPoints <= 30 ? 2 : Math.ceil(nPoints / 12));
                var tickvalsDay = [];
                var ticktextDay = [];
                for (var j = 0; j < xVals.length; j += step) {
                    tickvalsDay.push(xVals[j]);
                    ticktextDay.push(pbjPlotlyFormatIsoMmDdYy(pbjPlotlyParseChartXToIso(xVals[j]) || xVals[j]));
                }
                if (xVals.length && tickvalsDay[tickvalsDay.length - 1] !== xVals[xVals.length - 1]) {
                    tickvalsDay.push(xVals[xVals.length - 1]);
                    ticktextDay.push(
                        pbjPlotlyFormatIsoMmDdYy(
                            pbjPlotlyParseChartXToIso(xVals[xVals.length - 1]) || xVals[xVals.length - 1]
                        )
                    );
                }
                return {
                    tickmode: 'array',
                    tickvals: tickvalsDay,
                    ticktext: ticktextDay,
                    tickangle: nPoints > 20 ? -35 : -25,
                    automargin: true
                };
            }
            return {
                tickmode: 'auto',
                nticks: Math.min(12, Math.max(5, Math.ceil(nPoints / 3))),
                tickangle: nPoints > 24 ? -35 : -25,
                automargin: true
            };
        }

        if (nPoints <= 120 && !categoryIso) {
            return {
                tickmode: 'auto',
                nticks: 8,
                tickangle: -30,
                automargin: true
            };
        }

        var seen = {};
        var tickvals = [];
        var ticktext = [];
        xVals.forEach(function (xv) {
            var iso = pbjPlotlyParseChartXToIso(xv);
            if (!iso && /^\d{4}-\d{2}-\d{2}$/.test(String(xv))) {
                iso = String(xv);
            }
            if (!iso) {
                return;
            }
            var ym = iso.slice(0, 7);
            if (seen[ym]) {
                return;
            }
            seen[ym] = true;
            tickvals.push(xv);
            var mi = parseInt(iso.slice(5, 7), 10) - 1;
            ticktext.push(PBJ_PLOTLY_MONTHS[mi] + ' ' + iso.slice(0, 4));
        });
        return {
            tickmode: 'array',
            tickvals: tickvals,
            ticktext: ticktext,
            tickangle: 0,
            automargin: true
        };
    }

    function pbjPlotlyApplySmartXaxis(layout, traces, opts) {
        if (!layout) {
            return layout;
        }
        var patch = pbjPlotlySmartXaxisLayout(traces, opts);
        layout.xaxis = Object.assign({}, layout.xaxis || {}, patch);
        return layout;
    }

    /** Grain-aware smart x-axis opts for any trend chart (pass censusView, hprdView, etc.). */
    function pbjPlotlySmartXOptsForGrain(grain, traces, totalDays) {
        var g = grain || 'daily';
        var opts = { grain: g };
        if (g === 'daily' && totalDays > 0) {
            opts.nPoints = totalDays;
        } else {
            var xVals = pbjPlotlyMainTraceXValues(traces);
            if (xVals.length) {
                opts.nPoints = xVals.length;
            }
        }
        return opts;
    }

    function pbjPlotlyIsReferenceTraceName(name) {
        return /min\.|MACPAC|state avg|national avg/i.test(String(name || ''));
    }

    /** Y-axis range from visible data; omit zero floor when values sit well above 0. */
    function pbjPlotlyDynamicYRangeFromTraces(traces, opts) {
        opts = opts || {};
        var dataYmin = Infinity;
        var dataYmax = -Infinity;
        var refYmin = Infinity;
        var refYmax = -Infinity;
        var hasData = false;
        var hasRef = false;
        (traces || []).forEach(function (t) {
            var n = String(t.name || '');
            var mode = String(t.mode || '');
            if (mode.indexOf('lines') < 0 && mode.indexOf('markers') < 0 && t.type !== 'bar') {
                return;
            }
            if (t.visible === false || t.visible === 'legendonly') {
                return;
            }
            if (/^(_pbjEvents|Holidays)/.test(n)) {
                return;
            }
            var isRef = pbjPlotlyIsReferenceTraceName(n);
            if (isRef && opts.excludeReference) {
                return;
            }
            (t.y || []).forEach(function (v) {
                var vn = typeof v === 'number' ? v : parseFloat(String(v));
                if (!Number.isFinite(vn)) {
                    return;
                }
                if (opts.positiveOnly && vn <= 0) {
                    return;
                }
                if (isRef) {
                    hasRef = true;
                    if (vn < refYmin) {
                        refYmin = vn;
                    }
                    if (vn > refYmax) {
                        refYmax = vn;
                    }
                } else {
                    hasData = true;
                    if (vn < dataYmin) {
                        dataYmin = vn;
                    }
                    if (vn > dataYmax) {
                        dataYmax = vn;
                    }
                }
            });
        });
        var ymin;
        var ymax;
        if (hasData) {
            ymin = dataYmin;
            ymax = dataYmax;
            if (opts.includeReference !== false && hasRef && Number.isFinite(refYmin)) {
                var dataSpan = Math.max(dataYmax - dataYmin, opts.minSpan || 0.05);
                var refSlack = Math.max(dataSpan * 0.15, 0.12);
                if (refYmax <= dataYmin + refSlack * 2 || refYmax <= dataYmax + refSlack) {
                    ymin = Math.min(ymin, refYmin);
                }
            }
        } else if (hasRef) {
            ymin = refYmin;
            ymax = refYmax;
        } else {
            return null;
        }
        if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) {
            return null;
        }
        if (opts.stacked) {
            var sums = [];
            (traces || []).forEach(function (t) {
                if (!t || !Array.isArray(t.y)) {
                    return;
                }
                if (/^(_pbjEvents|Holidays)/.test(String(t.name || ''))) {
                    return;
                }
                (t.y || []).forEach(function (v, i) {
                    var vn = typeof v === 'number' ? v : parseFloat(String(v));
                    if (!Number.isFinite(vn)) {
                        return;
                    }
                    sums[i] = (sums[i] || 0) + vn;
                });
            });
            if (sums.length) {
                ymin = Math.min.apply(null, sums.filter(function (v) {
                    return Number.isFinite(v);
                }));
                ymax = Math.max.apply(null, sums);
            }
        }
        if (!Number.isFinite(ymin) || !Number.isFinite(ymax)) {
            return null;
        }
        var span = Math.max(ymax - ymin, opts.minSpan || 0.05);
        var padLo = Math.max(0.06, span * (opts.padLoRatio || 0.12));
        var padHi = Math.max(0.06, span * (opts.padHiRatio || 0.18));
        var lo = ymin - padLo;
        if (ymin <= 0.2) {
            lo = Math.max(0, lo);
        }
        var hi = Math.max(ymax + padHi, lo + (opts.minSpan || 0.25));
        if (opts.minCap != null && Number.isFinite(opts.minCap)) {
            lo = Math.max(opts.minCap, lo);
        }
        if (opts.maxCap != null && Number.isFinite(opts.maxCap)) {
            hi = Math.min(opts.maxCap, hi);
        }
        if (hi - lo < (opts.minSpan || 0.05)) {
            hi = Math.min(
                opts.maxCap != null && Number.isFinite(opts.maxCap) ? opts.maxCap : hi,
                lo + (opts.minSpan || 0.05)
            );
        }
        return [lo, hi];
    }

    function pbjPlotlyApplyDynamicYaxis(layout, traces, opts) {
        if (!layout) {
            return layout;
        }
        var yR = pbjPlotlyDynamicYRangeFromTraces(traces, opts);
        if (!yR) {
            return layout;
        }
        layout.yaxis = Object.assign({}, layout.yaxis || {}, {
            range: yR,
            autorange: false
        });
        if (Object.prototype.hasOwnProperty.call(layout.yaxis, 'rangemode')) {
            delete layout.yaxis.rangemode;
        }
        if (layout.yaxis2 && opts.applyY2 !== false) {
            layout.yaxis2 = Object.assign({}, layout.yaxis2, {
                range: yR,
                autorange: false
            });
            if (Object.prototype.hasOwnProperty.call(layout.yaxis2, 'rangemode')) {
                delete layout.yaxis2.rangemode;
            }
        }
        return layout;
    }

    /** Recompute y-axis from current gd.data (respects legend visibility) and relayout. */
    function pbjPlotlyRelayoutDynamicYaxisFromChart(gd, opts) {
        if (!gd || !gd.data || typeof global.Plotly === 'undefined' || !global.Plotly.relayout) {
            return;
        }
        var yR = pbjPlotlyDynamicYRangeFromTraces(gd.data, opts || {});
        if (!yR) {
            try {
                global.Plotly.relayout(gd, { 'yaxis.autorange': true });
            } catch (eAuto) { /* ignore */ }
            return;
        }
        var patch = {
            'yaxis.autorange': false,
            'yaxis.range': yR
        };
        if ((opts || {}).applyY2 !== false && gd.layout && gd.layout.yaxis2) {
            patch['yaxis2.autorange'] = false;
            patch['yaxis2.range'] = yR;
        }
        try {
            global.Plotly.relayout(gd, patch);
        } catch (eRel) { /* ignore */ }
    }

    /** After legend toggles, rescale y-axis to visible series (Staffing trends, etc.). */
    function pbjPlotlyWireDynamicYaxisOnLegend(chartElOrId, opts) {
        var el = typeof chartElOrId === 'string' ? document.getElementById(chartElOrId) : chartElOrId;
        if (!el || typeof el.on !== 'function') {
            return;
        }
        try {
            if (typeof el.removeAllListeners === 'function') {
                el.removeAllListeners('plotly_restyle');
            }
        } catch (eRm) { /* ignore */ }
        var wireOpts = opts || {};
        el.on('plotly_restyle', function () {
            pbjPlotlyRelayoutDynamicYaxisFromChart(el, wireOpts);
        });
    }

    /** Tab/visibility resize only — re-apply legend patch after Plotly.Plots.resize. */
    function pbjPlotlyResizeChartPreserveLegend(chartElOrId, layout) {
        var el = typeof chartElOrId === 'string' ? document.getElementById(chartElOrId) : chartElOrId;
        if (!el || typeof global.Plotly === 'undefined' || !global.Plotly.Plots || !global.Plotly.Plots.resize) {
            return;
        }
        try {
            global.Plotly.Plots.resize(el);
        } catch (eResize) { /* ignore */ }
        if (!layout || !layout.legend || !global.Plotly.relayout) {
            return;
        }
        var leg = layout.legend;
        var patch = {
            'legend.orientation': leg.orientation || 'h',
            'legend.x': leg.x != null ? leg.x : 0.5,
            'legend.xanchor': leg.xanchor || 'center',
            'legend.y': leg.y != null ? leg.y : -0.22,
            'legend.yanchor': leg.yanchor || 'top',
            'legend.xref': leg.xref || 'paper',
            'legend.yref': leg.yref || 'paper',
            'legend.tracegroupgap': leg.tracegroupgap != null ? leg.tracegroupgap : 8,
            'legend.itemsizing': leg.itemsizing || 'constant',
            'legend.itemwidth': leg.itemwidth != null ? leg.itemwidth : 20
        };
        if (leg.font && leg.font.size != null) {
            patch['legend.font.size'] = leg.font.size;
        }
        if (leg.entrywidth != null) {
            patch['legend.entrywidth'] = leg.entrywidth;
        }
        if (layout.margin && layout.margin.b != null) {
            patch['margin.b'] = layout.margin.b;
        }
        try {
            global.Plotly.relayout(el, patch);
        } catch (eRel) { /* ignore */ }
    }

    global.pbjV2IsMobileViewport = pbjV2IsMobileViewport;
    global.pbjV2MobilePick = pbjV2MobilePick;
    global.pbjV2CitationChipLabel = pbjV2CitationChipLabel;
    global.pbjV2EventTypeDisplayLabel = pbjV2EventTypeDisplayLabel;

    global.pbjPlotlyIsMobileChart = pbjPlotlyIsMobileChart;
    global.pbjPlotlyTrendLegendClearance = pbjPlotlyTrendLegendClearance;
    global.pbjPlotlySmartXaxisLayout = pbjPlotlySmartXaxisLayout;
    global.pbjPlotlyApplySmartXaxis = pbjPlotlyApplySmartXaxis;
    global.pbjPlotlySmartXOptsForGrain = pbjPlotlySmartXOptsForGrain;
    global.pbjPlotlyDynamicYRangeFromTraces = pbjPlotlyDynamicYRangeFromTraces;
    global.pbjPlotlyApplyDynamicYaxis = pbjPlotlyApplyDynamicYaxis;
    global.pbjPlotlyRelayoutDynamicYaxisFromChart = pbjPlotlyRelayoutDynamicYaxisFromChart;
    global.pbjPlotlyWireDynamicYaxisOnLegend = pbjPlotlyWireDynamicYaxisOnLegend;
    global.pbjPlotlyHorizontalLegendBelow = pbjPlotlyHorizontalLegendBelow;
    global.pbjPlotlyLegendBottomMargin = pbjPlotlyLegendBottomMargin;
    global.pbjPlotlyApplyHorizontalLegendLayout = pbjPlotlyApplyHorizontalLegendLayout;
    global.pbjPlotlyReapplyTrendLegendAfterEventShapes = pbjPlotlyReapplyTrendLegendAfterEventShapes;
    global.pbjPlotlyResizeChartPreserveLegend = pbjPlotlyResizeChartPreserveLegend;
    global.pbjPlotlyLegendRelayoutPatch = pbjPlotlyLegendRelayoutPatch;

    var PBJ_PLOTLY_TRACE_IDS = [
        'hprdTrendChart',
        'compositionTotalTrendChart',
        'compositionDirectTrendChart',
        'einHeadcountByJobChart',
        'contractTrendChart'
    ];

    function pbjPlotlyResolveChartId(graphDiv) {
        if (typeof graphDiv === 'string') {
            return graphDiv;
        }
        if (graphDiv && graphDiv.id) {
            return graphDiv.id;
        }
        if (graphDiv && graphDiv.node && graphDiv.node().id) {
            return graphDiv.node().id;
        }
        return '';
    }

    function pbjPlotlyResolveChartEl(graphDiv) {
        if (typeof graphDiv === 'string') {
            return document.getElementById(graphDiv);
        }
        return graphDiv;
    }

    function pbjPlotlyIsTrackedChart(chartId) {
        return PBJ_PLOTLY_TRACE_IDS.indexOf(chartId) >= 0;
    }

    function pbjPlotlySnapshotLegendLayout(layout) {
        if (!layout) {
            return null;
        }
        return {
            legend: layout.legend ? Object.assign({}, layout.legend) : null,
            margin: layout.margin ? Object.assign({}, layout.margin) : null,
            showlegend: layout.showlegend
        };
    }

    function pbjPlotlyStoreTrendChartLayout(chartId, layout) {
        if (!pbjPlotlyIsTrackedChart(chartId) || !layout) {
            return;
        }
        global.__pbjTrendChartLayouts = global.__pbjTrendChartLayouts || {};
        global.__pbjTrendChartLayouts[chartId] = pbjPlotlySnapshotLegendLayout(layout);
    }

    function pbjPlotlyTraceLog(method, chartId, phase, detail) {
        if (!global.__PBJ_DEBUG__ || !pbjPlotlyIsTrackedChart(chartId)) {
            return;
        }
        console.log('[PBJ Plotly trace]', {
            method: method,
            chartId: chartId,
            phase: phase,
            t: Date.now(),
            detail: detail || {}
        });
    }

    function pbjPlotlyReadGdLayout(gd) {
        if (!gd || !gd._fullLayout) {
            return {};
        }
        var fl = gd._fullLayout;
        return {
            orientation: fl.legend && fl.legend.orientation,
            x: fl.legend && fl.legend.x,
            y: fl.legend && fl.legend.y,
            marginB: fl.margin && fl.margin.b
        };
    }

    function pbjPlotlyLegendRelayoutPatch(layoutSnap) {
        if (!layoutSnap || !layoutSnap.legend) {
            return null;
        }
        var leg = layoutSnap.legend;
        var patch = {
            'legend.orientation': leg.orientation || 'h',
            'legend.x': leg.x != null ? leg.x : 0.5,
            'legend.xanchor': leg.xanchor || 'center',
            'legend.y': leg.y != null ? leg.y : -0.22,
            'legend.yanchor': leg.yanchor || 'top',
            'legend.xref': leg.xref || 'paper',
            'legend.yref': leg.yref || 'paper',
            'legend.tracegroupgap': leg.tracegroupgap != null ? leg.tracegroupgap : 8,
            'legend.itemsizing': leg.itemsizing || 'constant',
            'legend.itemwidth': leg.itemwidth != null ? leg.itemwidth : 20
        };
        if (leg.font && leg.font.size != null) {
            patch['legend.font.size'] = leg.font.size;
        }
        if (leg.entrywidth != null) {
            patch['legend.entrywidth'] = leg.entrywidth;
        }
        if (layoutSnap.margin && layoutSnap.margin.b != null) {
            patch['margin.b'] = layoutSnap.margin.b;
        }
        return patch;
    }

    function pbjPlotlyAfterOp(method, chartId, gd) {
        pbjPlotlyTraceLog(method, chartId, 'after', pbjPlotlyReadGdLayout(gd));
    }

    function pbjPlotlyInstallTraceAndResizeGuard() {
        if (typeof global.Plotly === 'undefined' || global.__pbjPlotlyTraceInstalled) {
            return;
        }
        global.__pbjPlotlyTraceInstalled = true;

        function wrapPromise(method, chartId, p) {
            if (!p || typeof p.then !== 'function') {
                return p;
            }
            return p.then(function (gd) {
                pbjPlotlyAfterOp(method, chartId, gd || pbjPlotlyResolveChartEl(chartId));
                return gd;
            });
        }

        var origRelayout = global.Plotly.relayout;
        if (origRelayout) {
            global.Plotly.relayout = function (graphDiv, update) {
                var chartId = pbjPlotlyResolveChartId(graphDiv);
                pbjPlotlyTraceLog('relayout', chartId, 'before', {
                    update: update,
                    gdBefore: pbjPlotlyReadGdLayout(pbjPlotlyResolveChartEl(graphDiv))
                });
                return wrapPromise('relayout', chartId, origRelayout.apply(this, arguments));
            };
        }

        var origUpdate = global.Plotly.update;
        if (origUpdate) {
            global.Plotly.update = function (graphDiv, traceUpdate, layoutUpdate, traces) {
                var chartId = pbjPlotlyResolveChartId(graphDiv);
                pbjPlotlyTraceLog('update', chartId, 'before', {
                    layoutUpdate: layoutUpdate,
                    gdBefore: pbjPlotlyReadGdLayout(pbjPlotlyResolveChartEl(graphDiv))
                });
                return wrapPromise('update', chartId, origUpdate.apply(this, arguments));
            };
        }

        if (global.Plotly.Plots && global.Plotly.Plots.resize) {
            var origResize = global.Plotly.Plots.resize;
            global.Plotly.Plots.resize = function (graphDiv) {
                var chartId = pbjPlotlyResolveChartId(graphDiv);
                var el = pbjPlotlyResolveChartEl(graphDiv);
                pbjPlotlyTraceLog('Plots.resize', chartId, 'before', {
                    gdBefore: pbjPlotlyReadGdLayout(el)
                });
                var out = origResize.apply(this, arguments);
                var snap = global.__pbjTrendChartLayouts && global.__pbjTrendChartLayouts[chartId];
                if (snap && snap.legend && origRelayout) {
                    try {
                        origRelayout.call(global.Plotly, el, pbjPlotlyLegendRelayoutPatch(snap));
                    } catch (ePatch) { /* ignore */ }
                }
                pbjPlotlyAfterOp('Plots.resize', chartId, el);
                return out;
            };
        }
    }

    global.pbjPlotlyStoreTrendChartLayout = pbjPlotlyStoreTrendChartLayout;
    global.pbjPlotlyVerifyTrendLegends = function () {
        return PBJ_PLOTLY_TRACE_IDS.map(function (id) {
            var gd = document.getElementById(id);
            return {
                id: id,
                exists: !!gd,
                orientation: gd && gd._fullLayout && gd._fullLayout.legend ? gd._fullLayout.legend.orientation : null,
                x: gd && gd._fullLayout && gd._fullLayout.legend ? gd._fullLayout.legend.x : null,
                y: gd && gd._fullLayout && gd._fullLayout.legend ? gd._fullLayout.legend.y : null,
                marginB: gd && gd._fullLayout && gd._fullLayout.margin ? gd._fullLayout.margin.b : null
            };
        });
    };

    function pbjV2ApplyMinimalChartChrome(lay) {
        if (!global.PBJ_V2_MINIMAL_CHART_CHROME || !lay) {
            return lay;
        }
        if (lay._pbjKeepChartAnnotations !== true) {
            lay.annotations = [];
        }
        var legHoriz = lay.legend && String(lay.legend.orientation || '').toLowerCase() === 'h';
        var showLeg = lay.showlegend !== false;
        if (showLeg && legHoriz) {
            if (lay.xaxis) {
                lay.xaxis = Object.assign({}, lay.xaxis, { automargin: true });
            }
            return lay;
        }
        var legendRoom = showLeg && lay.margin && lay.margin.b >= 130;
        if (legendRoom) {
            if (lay.xaxis) {
                lay.xaxis = Object.assign({}, lay.xaxis, { automargin: true });
            }
            return lay;
        }
        if (lay.margin && lay.margin.b > 110) {
            var ang = 0;
            if (lay.xaxis && lay.xaxis.tickangle != null) {
                ang = Math.abs(Number(lay.xaxis.tickangle)) || 0;
            }
            var minB = ang >= 45 ? 92 : (ang >= 25 ? 80 : 68);
            var maxB = lay.showlegend === true ? 120 : 100;
            lay.margin = Object.assign({}, lay.margin, {
                b: Math.max(minB, Math.min(lay.margin.b, maxB))
            });
            if (lay.xaxis) {
                lay.xaxis = Object.assign({}, lay.xaxis, { automargin: true });
            }
        }
        return lay;
    }

    function pbjV2InstallPlotlyMinimalChrome() {
        if (typeof global.Plotly === 'undefined' || !global.Plotly.newPlot) {
            return false;
        }
        if (global.__pbjV2PlotlyPatched) {
            return true;
        }
        global.__pbjV2PlotlyPatched = true;
        pbjPlotlyInstallTraceAndResizeGuard();
        var origNewPlot = global.Plotly.newPlot;

        function wrapPromise(method, chartId, p) {
            if (!p || typeof p.then !== 'function') {
                pbjPlotlyAfterOp(method, chartId, p);
                return p;
            }
            return p.then(function (gd) {
                pbjPlotlyAfterOp(method, chartId, gd || pbjPlotlyResolveChartEl(chartId));
                return gd;
            });
        }

        global.Plotly.newPlot = function (graphDiv, data, layout, config) {
            var chartId = pbjPlotlyResolveChartId(graphDiv);
            var lay = pbjV2ApplyMinimalChartChrome(pbjV2ClonePlotlyLayout(layout));
            pbjPlotlyStoreTrendChartLayout(chartId, lay);
            pbjPlotlyTraceLog('newPlot', chartId, 'before', {
                inLegend: lay && lay.legend,
                inMarginB: lay && lay.margin && lay.margin.b
            });
            return wrapPromise(
                'newPlot',
                chartId,
                origNewPlot.call(this, graphDiv, data, lay, config)
            );
        };
        if (global.Plotly.react) {
            var origReact = global.Plotly.react;
            global.Plotly.react = function (graphDiv, data, layout, config) {
                var chartId = pbjPlotlyResolveChartId(graphDiv);
                var lay = pbjV2ApplyMinimalChartChrome(pbjV2ClonePlotlyLayout(layout));
                pbjPlotlyStoreTrendChartLayout(chartId, lay);
                pbjPlotlyTraceLog('react', chartId, 'before', {
                    inLegend: lay && lay.legend,
                    inMarginB: lay && lay.margin && lay.margin.b
                });
                return wrapPromise(
                    'react',
                    chartId,
                    origReact.call(this, graphDiv, data, lay, config)
                );
            };
        }
        return true;
    }
    global.pbjV2InstallPlotlyMinimalChrome = pbjV2InstallPlotlyMinimalChrome;

    pbjV2InstallPlotlyMinimalChrome();

    function pbjV2UpdateScopeLabels(label) {
        var text = label || '—';
        var sub = document.getElementById('pbjScopeModalSubtitle');
        if (sub) {
            sub.textContent = text;
        }
        var floating = document.getElementById('pbjFloatingScopeLine');
        if (floating) {
            floating.textContent = text;
            floating.title = 'Active scope: ' + text;
        }
        var staffingPeriod = document.getElementById('pbjStaffingScopePeriod');
        if (staffingPeriod) {
            staffingPeriod.textContent = text;
            staffingPeriod.title = 'Active period: ' + text;
        }
        if (typeof global.reportBuilderRefreshUsePeriodBtnLabel === 'function') {
            global.reportBuilderRefreshUsePeriodBtnLabel();
        }
    }

    function pbjV2RefreshFloatingReportContext() {
        /* Report block no longer repeats applied period (shown in Summary kicker + Control Center). */
    }

    function pbjV2ScrollToSection(sectionId) {
        if (typeof global.__pbjUpdateGuidedScrollMargin === 'function') {
            global.__pbjUpdateGuidedScrollMargin();
        }
        if (typeof global.__pbjScrollToGuidedSection === 'function') {
            global.__pbjScrollToGuidedSection(sectionId);
            return;
        }
        var el = document.getElementById(sectionId);
        if (el && typeof el.scrollIntoView === 'function') {
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function pbjV2RevealInspectionsSection(opts) {
        opts = opts || {};
        if (global.__pbjV3PanesActive && typeof global.pbjV3SwitchPane === 'function' && global.__pbjV3ActivePane !== 'risk') {
            global.pbjV3SwitchPane('risk');
        }
        var citSec = document.getElementById('pbjCitationsSection');
        if (citSec) {
            citSec.classList.remove('d-none');
        }
        if (global.__pbjV3PanesActive && typeof global.pbjV3ReapplyPaneVisibility === 'function') {
            global.pbjV3ReapplyPaneVisibility();
        }
        if (typeof global.pbjLoadCitationsPanelOnDemand === 'function') {
            global.pbjLoadCitationsPanelOnDemand();
        }
        if (opts.skipScroll === true) {
            return;
        }
        var target = opts.scrollTarget || (opts.openFlags ? 'riskScreeningSection' : 'pbjCitationsSection');
        if (global.__pbjV3PanesActive && target === 'pbjCitationsSection') {
            target = 'pbjV3PaneRisk';
        }
        var delay = opts.openFlags ? 80 : 0;
        setTimeout(function () {
            pbjV2ScrollToSection(target);
        }, delay);
    }
    global.pbjV2RevealInspectionsSection = pbjV2RevealInspectionsSection;
    global.pbjV2RevealEventsSection = pbjV2RevealInspectionsSection;
    global.pbjV2ScrollToSection = pbjV2ScrollToSection;

    function pbjV2ApplyChowCompareFromModal() {
        if (typeof global.prePostApplyPreset === 'function') {
            global.prePostApplyPreset('chow_effective_date');
        }
        var modalEl = document.getElementById('pbjChowOwnershipModal');
        var scrollAfter = function () {
            setTimeout(function () {
                pbjV2ScrollToSection('prePostAnalysisCard');
            }, 100);
        };
        if (modalEl && typeof bootstrap !== 'undefined' && bootstrap.Modal) {
            modalEl.addEventListener('hidden.bs.modal', scrollAfter, { once: true });
            bootstrap.Modal.getOrCreateInstance(modalEl).hide();
            return;
        }
        scrollAfter();
    }
    global.pbjV2ApplyChowCompareFromModal = pbjV2ApplyChowCompareFromModal;

    function pbjV2OpenControlCenter(opts) {
        opts = opts || {};
        var wrap = document.getElementById('pbjFloatingControls');
        var fab = document.getElementById('pbjFloatingControlsFab');
        var panel = document.getElementById('pbjFloatingControlsPanel');
        var hint = document.getElementById('pbjFloatingControlsHintDot');
        if (!fab || !panel) {
            return;
        }
        fab.setAttribute('aria-expanded', 'true');
        panel.hidden = false;
        if (wrap) {
            wrap.classList.add('is-open');
            wrap.style.pointerEvents = 'auto';
        }
        panel.style.pointerEvents = 'auto';
        pbjV2SyncControlsDockBackdrop(true);
        pbjV2AdjustControlsDockPanelPlacement();
        requestAnimationFrame(function () {
            pbjV2SyncFloatingGrain();
            pbjV2RefreshFloatingPickerOptions();
            pbjV2PullFloatingPeriodFromSummary();
            if (typeof global.pbjV2RefreshScopeLabel === 'function') {
                global.pbjV2RefreshScopeLabel();
            }
            pbjV2RefreshFloatingReportContext();
            pbjV2AdjustControlsDockPanelPlacement();
        });
        try {
            localStorage.setItem('pbj_v2_controls_hint_seen', '1');
        } catch (eHint) { /* ignore */ }
        if (hint) {
            hint.classList.add('d-none');
        }
        var focusEl = null;
        if (opts.focusReport) {
            focusEl = document.getElementById('pbjFloatingReportBlock');
        } else if (opts.focusPeriod) {
            focusEl = document.querySelector('.pbj-floating-period-block');
        }
        if (focusEl && typeof focusEl.scrollIntoView === 'function') {
            setTimeout(function () {
                focusEl.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }, 80);
        }
    }

    /** Chrome/Edge: clicking the window scrollbar fires pointerdown on document outside the panel. */
    function pbjV2PointerOnWindowScrollbar(ev) {
        if (!ev || !isFinite(ev.clientX) || !isFinite(ev.clientY)) {
            return false;
        }
        var de = document.documentElement;
        var vw = de.clientWidth;
        var vh = de.clientHeight;
        var sw = window.innerWidth - vw;
        var sh = window.innerHeight - vh;
        if (sw > 0 && ev.clientX >= vw - 0.5) {
            return true;
        }
        if (sh > 0 && ev.clientY >= vh - 0.5) {
            return true;
        }
        return false;
    }

    /** Native scrollbar hits often miss wrap.contains(target); use path + panel rect. */
    function pbjV2FloatingControlsPointerInside(wrap, panel, ev) {
        if (!wrap || !panel || !ev) {
            return false;
        }
        if (ev.target && wrap.contains(ev.target)) {
            return true;
        }
        var path = typeof ev.composedPath === 'function' ? ev.composedPath() : [];
        for (var i = 0; i < path.length; i++) {
            if (path[i] === wrap || path[i] === panel) {
                return true;
            }
        }
        var x = ev.clientX;
        var y = ev.clientY;
        if (!isFinite(x) || !isFinite(y)) {
            return false;
        }
        var r = panel.getBoundingClientRect();
        return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
    }

    function pbjV2CloseControlCenter() {
        var wrap = document.getElementById('pbjFloatingControls');
        var fab = document.getElementById('pbjFloatingControlsFab');
        var panel = document.getElementById('pbjFloatingControlsPanel');
        var dock = document.getElementById('pbjControlsDock');
        if (!fab || !panel) {
            return;
        }
        fab.setAttribute('aria-expanded', 'false');
        panel.hidden = true;
        if (wrap) {
            wrap.classList.remove('is-open');
            wrap.classList.remove('pbj-floating-controls--panel-below');
            wrap.classList.remove('pbj-floating-controls--panel-align-left');
            wrap.classList.remove('pbj-floating-controls--panel-force-right');
        }
        if (dock) {
            pbjV2ResetControlsDockPosition(dock);
        }
        pbjV2SyncControlsDockBackdrop(false);
    }

    function pbjV2SyncControlsDockBackdrop(show) {
        var backdrop = document.getElementById('pbjControlsDockBackdrop');
        if (!backdrop) {
            return;
        }
        var mobile = false;
        try {
            mobile = window.matchMedia('(max-width: 767.98px)').matches;
        } catch (eMq) { /* ignore */ }
        if (show && mobile) {
            backdrop.hidden = false;
            backdrop.setAttribute('aria-hidden', 'false');
            backdrop.classList.add('is-visible');
            return;
        }
        backdrop.classList.remove('is-visible');
        backdrop.setAttribute('aria-hidden', 'true');
        backdrop.hidden = true;
    }

    function pbjV2NudgeDockForPanelInViewport(dock, panel, pad, minTop) {
        if (!dock || !panel) {
            return;
        }
        var pr = panel.getBoundingClientRect();
        var dr = dock.getBoundingClientRect();
        var shiftL = 0;
        var shiftT = 0;
        if (pr.right > window.innerWidth - pad) {
            shiftL -= pr.right - (window.innerWidth - pad);
        }
        if (pr.left < pad) {
            shiftL += pad - pr.left;
        }
        if (pr.top < minTop) {
            shiftT += minTop - pr.top;
        }
        if (pr.bottom > window.innerHeight - pad) {
            shiftT -= pr.bottom - (window.innerHeight - pad);
        }
        if (!shiftL && !shiftT) {
            return;
        }
        var clamped = pbjV2ClampControlsDockPosition(dr.left + shiftL, dr.top + shiftT, dock);
        pbjV2ApplyControlsDockPosition(dock, clamped.left, clamped.top, true);
    }

    function pbjV2AdjustControlsDockPanelPlacement() {
        var wrap = document.getElementById('pbjFloatingControls');
        var panel = document.getElementById('pbjFloatingControlsPanel');
        var dock = document.getElementById('pbjControlsDock');
        if (!wrap || !panel || panel.hidden) {
            return;
        }
        panel.style.maxHeight = '';
        panel.style.maxWidth = '';
        wrap.classList.remove(
            'pbj-floating-controls--panel-below',
            'pbj-floating-controls--panel-align-left',
            'pbj-floating-controls--panel-force-right'
        );
        if (window.innerWidth < 768) {
            pbjV2SyncControlsDockBackdrop(true);
            return;
        }
        pbjV2SyncControlsDockBackdrop(false);
        var pad = 12;
        var minTop = pbjV2ControlsDockMinTop();
        panel.style.maxWidth = Math.min(336, window.innerWidth - pad * 2) + 'px';
        var cluster = document.getElementById('pbjFloatingControlsFabCluster');
        var anchorRect = wrap.getBoundingClientRect();
        if (cluster) {
            try {
                if (getComputedStyle(cluster).display !== 'none') {
                    anchorRect = cluster.getBoundingClientRect();
                }
            } catch (eCluster) { /* ignore */ }
        }
        var spaceAbove = anchorRect.top - minTop - pad;
        var spaceBelow = window.innerHeight - anchorRect.bottom - pad;
        if (spaceAbove < 200 && spaceBelow > spaceAbove) {
            wrap.classList.add('pbj-floating-controls--panel-below');
        }
        var openBelow = wrap.classList.contains('pbj-floating-controls--panel-below');
        var available = openBelow ? spaceBelow : spaceAbove;
        if (available > 140) {
            panel.style.maxHeight = Math.min(window.innerHeight * 0.72, available) + 'px';
        }
        var panelRect = panel.getBoundingClientRect();
        if (panelRect.top < minTop) {
            wrap.classList.add('pbj-floating-controls--panel-below');
            openBelow = true;
            panelRect = panel.getBoundingClientRect();
        }
        if (panelRect.right > window.innerWidth - pad) {
            wrap.classList.add('pbj-floating-controls--panel-align-left');
            panelRect = panel.getBoundingClientRect();
        }
        if (panelRect.left < pad) {
            wrap.classList.remove('pbj-floating-controls--panel-align-left');
            wrap.classList.add('pbj-floating-controls--panel-force-right');
            panelRect = panel.getBoundingClientRect();
        }
        if (panelRect.bottom > window.innerHeight - pad) {
            var mh = Math.max(140, window.innerHeight - pad - Math.max(minTop, panelRect.top));
            panel.style.maxHeight = mh + 'px';
        }
        if (dock && dock.classList.contains('pbj-controls-dock--custom')) {
            pbjV2NudgeDockForPanelInViewport(dock, panel, pad, minTop);
        }
    }

    function pbjNavigateToReportBuilder() {
        var siteBase = '';
        try {
            var cfgEl = document.getElementById('pbj-client-config');
            if (cfgEl) {
                var cfg = JSON.parse(cfgEl.textContent || '{}');
                siteBase = String((cfg && cfg.siteBasePath) || '').replace(/\/+$/, '');
            }
        } catch (cfgErr) { /* ignore */ }
        var localPath = (siteBase || '') + '/case-builder';
        if (window.__pbjReportBuilderV3Href) {
            var h = String(window.__pbjReportBuilderV3Href);
            global.location.href = h;
            return true;
        }
        var desk = document.getElementById('guidedNavTabReportBuilder');
        var href = desk && desk.getAttribute('href');
        if (href && href !== '#reportBuilder' && href.charAt(0) !== '#') {
            global.location.href = href;
            return true;
        }
        if (localPath) {
            global.location.href = localPath;
            return true;
        }
        return false;
    }

    function pbjOpenReportBuilderFromControlCenter() {
        if (typeof global.pbjV2FloatingApplyPeriod === 'function') {
            global.pbjV2FloatingApplyPeriod();
        }
        pbjV2CloseControlCenter();
        if (pbjNavigateToReportBuilder()) {
            return;
        }
        if (typeof global.pbjSwitchTopTab === 'function') {
            global.pbjSwitchTopTab('reportBuilder');
            return;
        }
        var basePath = String(global.location.pathname || '/').replace(/\/+$/, '');
        global.location.href = (basePath || '') + '/?view=reportBuilder';
    }

    function pbjScrollToCompareWindows() {
        pbjV2ScrollToSection('prePostAnalysisCard');
    }

    function pbjOpenCompareForQuarter(quarterRaw, opts) {
        opts = opts || {};
        var q = String(quarterRaw || '').trim();
        if (!q || typeof global.prePostApplyQuarterAnchor !== 'function') {
            return false;
        }
        if (!global.prePostApplyQuarterAnchor(q)) {
            return false;
        }
        var noteEl = document.getElementById('prePostDefaultNote');
        if (noteEl) {
            var label = typeof global.pbjV2FormatCyQuarter === 'function'
                ? global.pbjV2FormatCyQuarter(q)
                : q;
            noteEl.textContent = 'Anchor applied: ' + label + ' (bounded by loaded PBJ dates).';
        }
        if (opts.scroll !== false) {
            pbjScrollToCompareWindows();
        }
        return true;
    }

    function pbjOpenCompareWindows(opts) {
        opts = opts || {};
        if (opts.preset && typeof global.prePostApplyPreset === 'function') {
            global.prePostApplyPreset(opts.preset);
        } else if (opts.quarter) {
            pbjOpenCompareForQuarter(opts.quarter, { scroll: false });
        }
        if (opts.scroll !== false) {
            pbjScrollToCompareWindows();
        }
    }

    function pbjNormalizeWorkDateIso(workDateRaw) {
        if (typeof global.pbjNormalizeIsoWorkDate === 'function') {
            return global.pbjNormalizeIsoWorkDate(workDateRaw);
        }
        return String(workDateRaw || '').trim().slice(0, 10);
    }

    function pbjSetActiveWorkDate(workDateRaw, opts) {
        opts = opts || {};
        var iso = pbjNormalizeWorkDateIso(workDateRaw);
        if (!iso) {
            return;
        }
        if (typeof global.pbjClampIsoToPbjBounds === 'function') {
            iso = global.pbjClampIsoToPbjBounds(iso);
        }
        if (!iso) {
            return;
        }

        var syncing = !!global.__pbjSetActiveWorkDateSyncing;
        if (!syncing) {
            global.__pbjSetActiveWorkDateSyncing = true;
        }

        var syncRosterDay = !!(opts.syncRosterDay || opts.loadRoster || opts.scrollTo === 'roster');
        [
            'pbjV2WorkDatePrimary',
            'filterDayDate',
            'specificDaySearch',
            'pbjFloatingFilterDay'
        ].forEach(function (id) {
            var el = document.getElementById(id);
            if (el && el.value !== iso) {
                el.value = iso;
            }
        });
        if (syncRosterDay) {
            var rosterDay = document.getElementById('einNursingWorkDayFilter');
            if (rosterDay && rosterDay.value !== iso) {
                rosterDay.value = iso;
                rosterDay.dataset.einUserSetDay = '1';
            }
            if (typeof global.einRosterSyncDayPickerEmptyState === 'function') {
                global.einRosterSyncDayPickerEmptyState();
            }
        }

        if (typeof global.pbjAssignSingleDayDate === 'function') {
            global.pbjAssignSingleDayDate(iso);
        } else {
            var sdi = document.getElementById('singleDayDate');
            if (sdi) {
                sdi.value = iso;
            }
        }

        if (typeof global.syncFilterDayGenerateReportLabel === 'function') {
            global.syncFilterDayGenerateReportLabel();
        }
        if (typeof global.pbjV2PullFloatingPeriodFromSummary === 'function') {
            global.pbjV2PullFloatingPeriodFromSummary();
        }

        global.__pbjActiveWorkDate = iso;

        if (!syncing) {
            global.__pbjSetActiveWorkDateSyncing = false;
        }

        if (opts.syncOnly) {
            return;
        }

        if (opts.setDashboardDayFilter) {
            var dayRadio = document.getElementById('filterTypeDay');
            if (dayRadio && !dayRadio.checked) {
                dayRadio.checked = true;
                if (typeof global.onFilterTypeChange === 'function') {
                    global.onFilterTypeChange();
                }
                if (typeof pbjV2SyncFloatingGrain === 'function') {
                    pbjV2SyncFloatingGrain();
                }
            }
            if (opts.applyDayFilter && typeof global.applyFilters === 'function') {
                global.applyFilters();
            }
        }

        if (opts.scrollDaily || opts.scrollTo === 'daily') {
            if (global.__pbjV3PanesActive && typeof global.pbjV3SwitchPane === 'function') {
                global.pbjV3SwitchPane('workforce');
            }
            if (typeof global.pbjScrollDailyTableToIso === 'function') {
                global.pbjScrollDailyTableToIso(iso);
            }
        } else if (opts.scrollTo === 'roster') {
            if (global.__pbjV3PanesActive && typeof global.pbjV3SwitchPane === 'function') {
                global.pbjV3SwitchPane('workforce');
            }
            var rosterSec =
                document.getElementById('einNursingEmployeeExplorerSection') ||
                document.getElementById('einNursingSection') ||
                document.getElementById('dayLevelEvidenceSection');
            if (rosterSec) {
                pbjV2ScrollToSection(rosterSec.id || 'dayLevelEvidenceSection');
            }
        }

        if (opts.loadRoster && typeof global.fetchEinDayRoster === 'function') {
            if (global.__pbjV3PanesActive && typeof global.pbjV3SwitchPane === 'function') {
                global.pbjV3SwitchPane('workforce');
            }
            global.fetchEinDayRoster(iso);
        }

        if (opts.openReport) {
            if (typeof global.openPbj320SnapshotModalForDay === 'function') {
                global.openPbj320SnapshotModalForDay(iso);
            } else if (typeof global.openSingleDayReport === 'function') {
                global.openSingleDayReport(iso);
            }
        }
    }

    function pbjInitV2WorkDateBar() {
        var primary = document.getElementById('pbjV2WorkDatePrimary');
        if (!primary || primary.dataset.pbjBound === '1') {
            return;
        }
        primary.dataset.pbjBound = '1';

        primary.addEventListener('change', function () {
            if (primary.value) {
                pbjSetActiveWorkDate(primary.value, { syncOnly: true });
            }
        });

        var goDaily = document.getElementById('pbjV2WorkDateGoDaily');
        if (goDaily) {
            goDaily.addEventListener('click', function () {
                var v = primary.value || global.__pbjActiveWorkDate || '';
                if (!v) {
                    return;
                }
                pbjSetActiveWorkDate(v, {
                    scrollDaily: true,
                    scrollTo: 'daily',
                    setDashboardDayFilter: true
                });
            });
        }

        var goRoster = document.getElementById('pbjV2WorkDateGoRoster');
        if (goRoster) {
            goRoster.addEventListener('click', function () {
                var v = primary.value || global.__pbjActiveWorkDate || '';
                if (!v) {
                    return;
                }
                pbjSetActiveWorkDate(v, {
                    loadRoster: true,
                    scrollTo: 'roster',
                    setDashboardDayFilter: true
                });
            });
        }

        var dayReport = document.getElementById('pbjV2WorkDateDayReport');
        if (dayReport) {
            dayReport.addEventListener('click', function () {
                var v = primary.value || global.__pbjActiveWorkDate || '';
                if (!v) {
                    alert('Select a work date first.');
                    return;
                }
                pbjSetActiveWorkDate(v, { openReport: true, setDashboardDayFilter: true });
            });
        }

        if (!primary.value && global.__pbjMaxWorkDate) {
            primary.value = global.__pbjMaxWorkDate;
            global.__pbjActiveWorkDate = global.__pbjMaxWorkDate;
            ['filterDayDate', 'specificDaySearch', 'pbjFloatingFilterDay'].forEach(function (id) {
                var el = document.getElementById(id);
                if (el && !el.value) {
                    el.value = global.__pbjMaxWorkDate;
                }
            });
            if (typeof global.syncFilterDayGenerateReportLabel === 'function') {
                global.syncFilterDayGenerateReportLabel();
            }
        }

        var rosterDayEl = document.getElementById('einNursingWorkDayFilter');
        if (rosterDayEl && rosterDayEl.dataset.einUserSetDay !== '1' && rosterDayEl.value) {
            rosterDayEl.value = '';
            if (typeof global.einRosterSyncDayPickerEmptyState === 'function') {
                global.einRosterSyncDayPickerEmptyState();
            }
        }

        ['einNursingWorkDayFilter', 'specificDaySearch'].forEach(function (id) {
            var el = document.getElementById(id);
            if (!el || el.dataset.pbjWorkDateSyncBound === '1') {
                return;
            }
            el.dataset.pbjWorkDateSyncBound = '1';
            el.addEventListener('change', function () {
                if (id === 'einNursingWorkDayFilter') {
                    el.dataset.einUserSetDay = el.value ? '1' : '';
                    if (typeof global.einRosterSyncDayPickerEmptyState === 'function') {
                        global.einRosterSyncDayPickerEmptyState();
                    }
                }
                if (el.value) {
                    pbjSetActiveWorkDate(el.value, {
                        syncOnly: true,
                        syncRosterDay: id === 'einNursingWorkDayFilter'
                    });
                }
            });
        });
    }

    /** Normalize quarter labels to canonical yyyyQn (matches v2 inline pbjNormalizeQuarterToCy). */
    function pbjV2NormalizeQuarterKey(qRaw) {
        var s = String(qRaw || '').trim();
        if (!s) {
            return '';
        }
        var m = s.match(/^(?:CY)?(\d{4})Q([1-4])$/i);
        if (m) {
            return m[1] + 'Q' + m[2];
        }
        m = s.match(/^Q([1-4])\s+(\d{4})$/i);
        if (m) {
            return m[2] + 'Q' + m[1];
        }
        m = s.match(/^(\d{4})\s+Q([1-4])$/i);
        if (m) {
            return m[1] + 'Q' + m[2];
        }
        return '';
    }

    function pbjApplyDashboardQuarter(quarterKey, opts) {
        opts = opts || {};
        var q = pbjV2NormalizeQuarterKey(quarterKey) || String(quarterKey || '').trim();
        if (!q) {
            return;
        }
        var quarterSelect = document.getElementById('quarterRange');
        if (!quarterSelect) {
            return;
        }

        var qRadio = document.getElementById('filterTypeQuarters');
        if (qRadio && !qRadio.checked) {
            qRadio.checked = true;
            if (typeof global.onFilterTypeChange === 'function') {
                global.onFilterTypeChange();
            }
            pbjV2SyncFloatingGrain();
        }

        Array.prototype.forEach.call(quarterSelect.options, function (opt) {
            opt.selected = opt.value === q;
        });
        if (typeof global.onQuarterChange === 'function') {
            global.onQuarterChange();
        }
        if (typeof global.applyFilters === 'function') {
            global.applyFilters();
        }

        if (opts.drillDays) {
            setTimeout(function () {
                pbjV2ScrollToSection('dayLevelEvidenceSection');
            }, opts.scrollDelay != null ? opts.scrollDelay : 350);
        }
    }

    function pbjBindQuarterDrillClicks() {
        /* Quarter cells use PBJ320 snapshot favicon buttons (openPbj320SnapshotModalForQuarter). */
    }

    function pbjBuildDayEvidenceActionCellHtml(isoRaw) {
        var iso = pbjNormalizeWorkDateIso(isoRaw);
        if (!iso) {
            return '';
        }
        var esc = iso.replace(/'/g, '');
        var parts = [];
        parts.push(
            '<button type="button" class="btn btn-link btn-sm py-0 px-0 text-nowrap fw-semibold text-primary" ' +
            'onclick="pbjSetActiveWorkDate(\'' + esc + '\',{scrollDaily:true,scrollTo:\'daily\',setDashboardDayFilter:true})" ' +
            'title="Scroll to daily PBJ row">Daily</button>'
        );
        parts.push('<span class="text-muted px-1">·</span>');
        parts.push(
            '<button type="button" class="btn btn-link btn-sm py-0 px-0 text-nowrap" ' +
            'onclick="pbjV2OpenOutlierDay(\'' + esc + '\')" title="Open PBJ320 day report">Report</button>'
        );
        if (typeof global.pbjEinRosterAllowedForIso === 'function' && global.pbjEinRosterAllowedForIso(iso)) {
            parts.push('<span class="text-muted px-1">·</span>');
            parts.push(
                '<button type="button" class="btn btn-link btn-sm py-0 px-0 text-nowrap fw-semibold text-primary" ' +
                'onclick="pbjSetActiveWorkDate(\'' + esc + '\',{loadRoster:true,scrollTo:\'roster\',setDashboardDayFilter:true})" ' +
                'title="Open CMS Employee Detail roster">Roster</button>'
            );
        }
        return parts.join('');
    }

    function pbjInitScopeChipClick() {
        var chip = document.getElementById('pbjV2ScopeChip');
        if (!chip || chip.dataset.pbjBound === '1') {
            return;
        }
        chip.dataset.pbjBound = '1';
        chip.addEventListener('click', function () {
            pbjV2OpenControlCenter({ focusPeriod: true });
        });
    }

    function pbjV2SyncFloatingGrain() {
        var checked = document.querySelector('input[name="filterType"]:checked');
        var v = checked ? checked.value : 'quarters';
        document.querySelectorAll('[data-pbj-floating-grain]').forEach(function (btn) {
            var on = btn.getAttribute('data-pbj-floating-grain') === v;
            btn.classList.toggle('active', on);
            btn.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
        pbjV2SyncFloatingPeriodFieldsVisibility();
    }

    function pbjV2SyncFloatingPeriodFieldsVisibility() {
        var ft = document.querySelector('input[name="filterType"]:checked');
        var filterType = ft ? ft.value : 'quarters';
        var quarters = document.getElementById('pbjFloatingPeriodQuarters');
        var years = document.getElementById('pbjFloatingPeriodYears');
        var months = document.getElementById('pbjFloatingPeriodMonths');
        var custom = document.getElementById('pbjFloatingPeriodCustom');
        var day = document.getElementById('pbjFloatingPeriodDay');
        if (quarters) {
            quarters.classList.toggle('d-none', filterType !== 'quarters');
        }
        if (years) {
            years.classList.toggle('d-none', filterType !== 'years');
        }
        if (months) {
            months.classList.toggle('d-none', filterType !== 'months');
        }
        if (custom) {
            custom.classList.toggle('d-none', filterType !== 'daterange');
        }
        if (day) {
            day.classList.toggle('d-none', filterType !== 'day');
        }
        var wrap = document.getElementById('pbjFloatingControls');
        if (wrap) {
            wrap.classList.add('pbj-floating-ready');
        }
    }

    function pbjV2SelectHasRealOptions(srcId, selectEl) {
        if (!selectEl || !selectEl.options || !selectEl.options.length) {
            return false;
        }
        if (srcId === 'quarterRange' || srcId === 'years') {
            return selectEl.options.length > 1;
        }
        return true;
    }

    function pbjV2BestOptionsSource(srcId) {
        var primary = document.getElementById(srcId);
        if (pbjV2SelectHasRealOptions(srcId, primary)) {
            return primary;
        }
        if (srcId === 'quarterRange') {
            var mobileQ = document.getElementById('quarterRangeMobileCompact');
            if (pbjV2SelectHasRealOptions(srcId, mobileQ)) {
                return mobileQ;
            }
        }
        if (srcId === 'years') {
            var mobileY = document.getElementById('yearsMobileCompact');
            if (pbjV2SelectHasRealOptions(srcId, mobileY)) {
                return mobileY;
            }
        }
        return null;
    }

    function pbjV2BuildFloatingQuarterOptions(dst, keys) {
        if (!dst || !Array.isArray(keys) || !keys.length) {
            return false;
        }
        var formatQ = typeof global.formatQuarter === 'function'
            ? global.formatQuarter
            : function (q) { return String(q || ''); };
        dst.innerHTML = '';
        var phQ = document.createElement('option');
        phQ.value = '';
        phQ.disabled = true;
        phQ.textContent = 'Select quarters…';
        dst.appendChild(phQ);
        var allQ = document.createElement('option');
        allQ.value = 'all';
        allQ.textContent = 'All Quarters';
        dst.appendChild(allQ);
        keys.forEach(function (q) {
            var o = document.createElement('option');
            o.value = q;
            o.textContent = pbjV2FormatQuarterOptionLabel(q, formatQ(q));
            dst.appendChild(o);
        });
        return true;
    }

    function pbjV2BuildFloatingYearOptions(dst, quarterKeys) {
        if (!dst || !Array.isArray(quarterKeys) || !quarterKeys.length) {
            return false;
        }
        var minY = 2017;
        var maxY = 2025;
        quarterKeys.forEach(function (q) {
            var m = String(q).match(/(\d{4})/);
            if (m) {
                var y = parseInt(m[1], 10);
                if (!isNaN(y)) {
                    minY = Math.min(minY, y);
                    maxY = Math.max(maxY, y);
                }
            }
        });
        dst.innerHTML = '';
        var ph = document.createElement('option');
        ph.value = '';
        ph.disabled = true;
        ph.textContent = 'Select years…';
        dst.appendChild(ph);
        var allOpt = document.createElement('option');
        allOpt.value = 'all';
        allOpt.textContent = 'All Years';
        dst.appendChild(allOpt);
        for (var year = maxY; year >= minY; year--) {
            var o = document.createElement('option');
            o.value = String(year);
            o.textContent = String(year);
            dst.appendChild(o);
        }
        return true;
    }

    function pbjV2FloatingSelectNeedsPopulate(dstId, dst) {
        if (!dst || !dst.options || !dst.options.length) {
            return true;
        }
        if (dst.options.length === 1 && (dst.options[0].value === '' || dst.options[0].disabled)) {
            return true;
        }
        if (
            (dstId === 'pbjFloatingQuarterSelect' ||
                dstId === 'pbjFloatingYearSelect' ||
                dstId === 'pbjAiQuarterSelect' ||
                dstId === 'pbjAiYearSelect') &&
            dst.options.length <= 1
        ) {
            return true;
        }
        return false;
    }

    function pbjV2FacilityCcnKey() {
        if (typeof global.pbjDashboardFacilityCcn === 'function') {
            return global.pbjDashboardFacilityCcn();
        }
        return String(typeof global.PBJ320_EXPORT_CCN !== 'undefined' ? global.PBJ320_EXPORT_CCN : '')
            .replace(/\D/g, '')
            .padStart(6, '0')
            .slice(-6);
    }

    function pbjV2GetCachedFacilityQuarters() {
        var ccn = pbjV2FacilityCcnKey();
        var keys = global.__pbjFacilityQuarterKeys;
        if (Array.isArray(keys) && keys.length && global.__pbjFacilityQuarterKeysCcn === ccn) {
            return keys.slice();
        }
        return null;
    }

    function pbjV2ApplyFloatingQuarterOptionsFromList(dst, quarters) {
        if (!dst || !Array.isArray(quarters) || !quarters.length) {
            return false;
        }
        if (!pbjV2BuildFloatingQuarterOptions(dst, quarters)) {
            return false;
        }
        var qr = document.getElementById('quarterRange');
        if (qr && qr.options && qr.options.length > 1) {
            pbjV2MirrorSelectByValue(qr, dst);
        }
        return true;
    }

    function pbjV2FetchFloatingQuarterOptionsIfNeeded(dst) {
        if (!dst || !pbjV2FloatingSelectNeedsPopulate('pbjFloatingQuarterSelect', dst)) {
            return;
        }
        var cached = pbjV2GetCachedFacilityQuarters();
        if (cached && pbjV2ApplyFloatingQuarterOptionsFromList(dst, cached)) {
            if (typeof global.pbjPerfLog === 'function') {
                global.pbjPerfLog('floating quarters cache', ['hit: __pbjFacilityQuarterKeys']);
            }
            return;
        }
        if (global.__pbjFloatingQuarterFetchPending) {
            return;
        }
        if (global.__pbjQuartersBootstrapActive) {
            var attempts = 0;
            var waitForBootstrapCache = function () {
                var bootCached = pbjV2GetCachedFacilityQuarters();
                if (bootCached && pbjV2ApplyFloatingQuarterOptionsFromList(dst, bootCached)) {
                    if (typeof global.pbjPerfLog === 'function') {
                        global.pbjPerfLog('floating quarters cache', ['hit: bootstrap wait']);
                    }
                    return;
                }
                if (++attempts < 60) {
                    setTimeout(waitForBootstrapCache, 50);
                    return;
                }
                pbjV2FetchFloatingQuarterOptionsIfNeededFetch(dst);
            };
            waitForBootstrapCache();
            return;
        }
        pbjV2FetchFloatingQuarterOptionsIfNeededFetch(dst);
    }

    function pbjV2FetchFloatingQuarterOptionsIfNeededFetch(dst) {
        if (!dst || global.__pbjFloatingQuarterFetchPending) {
            return;
        }
        var cached = pbjV2GetCachedFacilityQuarters();
        if (cached && pbjV2ApplyFloatingQuarterOptionsFromList(dst, cached)) {
            if (typeof global.pbjPerfLog === 'function') {
                global.pbjPerfLog('floating quarters cache', ['hit: __pbjFacilityQuarterKeys']);
            }
            return;
        }
        if (typeof global.pbjPerfLog === 'function') {
            global.pbjPerfLog('floating quarters cache', ['miss: fetching /api/quarters']);
        }
        global.__pbjFloatingQuarterFetchPending = true;
        var api = typeof global.pbjApiUrl === 'function' ? global.pbjApiUrl('/api/quarters') : '/api/quarters';
        fetch(api)
            .then(function (r) { return r.ok ? r.json() : []; })
            .then(function (quarters) {
                if (!Array.isArray(quarters) || !quarters.length) {
                    return;
                }
                global.__pbjFacilityQuarterKeys = quarters.slice();
                global.__pbjFacilityQuarterKeysCcn = pbjV2FacilityCcnKey();
                pbjV2ApplyFloatingQuarterOptionsFromList(dst, quarters);
            })
            .catch(function () { /* ignore */ })
            .finally(function () {
                global.__pbjFloatingQuarterFetchPending = false;
            });
    }

    function pbjV2FormatQuarterOptionLabel(value, fallback) {
        var v = String(value || '').trim();
        if (!v || v === 'all' || v === '__auto__') {
            return String(fallback || v);
        }
        if (typeof global.formatQuarter === 'function') {
            var formatted = global.formatQuarter(v);
            if (formatted && formatted !== v) {
                return formatted;
            }
        }
        return pbjV2FormatCyQuarter(v) || String(fallback || v);
    }

    function pbjV2CopySelectOptions(srcId, dstId) {
        var dst = document.getElementById(dstId);
        if (!dst) {
            return;
        }
        var src = pbjV2BestOptionsSource(srcId);
        if (src) {
            dst.innerHTML = '';
            Array.prototype.forEach.call(src.options, function (opt) {
                var o = document.createElement('option');
                o.value = opt.value;
                o.textContent = pbjV2FormatQuarterOptionLabel(opt.value, opt.textContent);
                o.selected = opt.selected;
                dst.appendChild(o);
            });
            if (!pbjV2FloatingSelectNeedsPopulate(dstId, dst)) {
                return;
            }
        }
        var cachedQuarters = pbjV2GetCachedFacilityQuarters();
        if (dstId === 'pbjFloatingQuarterSelect' && cachedQuarters) {
            if (pbjV2BuildFloatingQuarterOptions(dst, cachedQuarters)) {
                return;
            }
        }
        if (dstId === 'pbjFloatingYearSelect' && cachedQuarters) {
            if (pbjV2BuildFloatingYearOptions(dst, cachedQuarters)) {
                return;
            }
        }
        if (dstId === 'pbjAiQuarterSelect' && cachedQuarters) {
            if (pbjV2BuildFloatingQuarterOptions(dst, cachedQuarters)) {
                var qrAi = pbjV2BestOptionsSource('quarterRange');
                if (qrAi) {
                    pbjV2MirrorSelectByValue(qrAi, dst);
                }
                return;
            }
        }
        if (dstId === 'pbjAiYearSelect' && cachedQuarters) {
            if (pbjV2BuildFloatingYearOptions(dst, cachedQuarters)) {
                var yrAi = pbjV2BestOptionsSource('years');
                if (yrAi) {
                    pbjV2MirrorSelectByValue(yrAi, dst);
                }
                return;
            }
        }
        if (dstId === 'pbjFloatingQuarterSelect' || dstId === 'pbjAiQuarterSelect') {
            pbjV2FetchFloatingQuarterOptionsIfNeeded(dst);
        }
    }

    /** Shift-click range select on native multi-selects (no hint text needed). */
    function pbjWireMultiSelectShiftRange(selectEl) {
        if (!selectEl || selectEl.dataset.pbjShiftRangeWired === '1') {
            return;
        }
        selectEl.dataset.pbjShiftRangeWired = '1';
        var anchorIdx = -1;
        selectEl.addEventListener('mousedown', function (ev) {
            var opt = ev.target;
            if (!opt || opt.tagName !== 'OPTION') {
                return;
            }
            var idx = opt.index;
            if (ev.shiftKey && anchorIdx >= 0 && anchorIdx !== idx) {
                ev.preventDefault();
                var lo = Math.min(anchorIdx, idx);
                var hi = Math.max(anchorIdx, idx);
                for (var i = lo; i <= hi; i++) {
                    selectEl.options[i].selected = true;
                }
                var allOpt = Array.prototype.find.call(selectEl.options, function (o) {
                    return o.value === 'all';
                });
                if (allOpt) {
                    allOpt.selected = false;
                }
                selectEl.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
        selectEl.addEventListener('mouseup', function (ev) {
            var opt = ev.target;
            if (opt && opt.tagName === 'OPTION') {
                anchorIdx = opt.index;
            }
        });
    }

    global.pbjWireMultiSelectShiftRange = pbjWireMultiSelectShiftRange;

    function pbjV2MirrorSelectByValue(src, dst) {
        if (!src || !dst) {
            return;
        }
        Array.prototype.forEach.call(dst.options, function (o) {
            var mirror = Array.prototype.find.call(src.options, function (m) {
                return m.value === o.value;
            });
            o.selected = mirror ? mirror.selected : false;
        });
    }

    function pbjV2RefreshFloatingPickerOptions() {
        pbjV2CopySelectOptions('quarterRange', 'pbjFloatingQuarterSelect');
        pbjV2CopySelectOptions('years', 'pbjFloatingYearSelect');
    }

    function pbjV2PullFloatingPeriodFromSummary() {
        var sd = document.getElementById('startDate');
        var ed = document.getElementById('endDate');
        var fd = document.getElementById('filterDayDate');
        var sm = document.getElementById('startMonth');
        var em = document.getElementById('endMonth');
        var fsd = document.getElementById('pbjFloatingStartDate');
        var fed = document.getElementById('pbjFloatingEndDate');
        var ffd = document.getElementById('pbjFloatingFilterDay');
        var fsm = document.getElementById('pbjFloatingStartMonth');
        var fem = document.getElementById('pbjFloatingEndMonth');
        var fq = document.getElementById('pbjFloatingQuarterSelect');
        var fy = document.getElementById('pbjFloatingYearSelect');
        var qr = document.getElementById('quarterRange');
        var yr = document.getElementById('years');
        pbjV2RefreshFloatingPickerOptions();
        if (fsd && sd) {
            fsd.value = sd.value || '';
        }
        if (fed && ed) {
            fed.value = ed.value || '';
        }
        if (ffd && fd) {
            ffd.value = fd.value || '';
        }
        if (fsm && sm) {
            fsm.value = sm.value || '';
        }
        if (fem && em) {
            fem.value = em.value || '';
        }
        if (fq && qr && qr.options && qr.options.length) {
            pbjV2MirrorSelectByValue(qr, fq);
        } else if (fq) {
            pbjV2FetchFloatingQuarterOptionsIfNeeded(fq);
        }
        if (fy && yr && yr.options && yr.options.length) {
            pbjV2MirrorSelectByValue(yr, fy);
        }
        pbjV2SyncFloatingPeriodFieldsVisibility();
    }

    function pbjV2PushFloatingPeriodToSummary() {
        var ft = document.querySelector('input[name="filterType"]:checked');
        var filterType = ft ? ft.value : 'quarters';
        var sd = document.getElementById('startDate');
        var ed = document.getElementById('endDate');
        var fd = document.getElementById('filterDayDate');
        var sm = document.getElementById('startMonth');
        var em = document.getElementById('endMonth');
        var fsd = document.getElementById('pbjFloatingStartDate');
        var fed = document.getElementById('pbjFloatingEndDate');
        var ffd = document.getElementById('pbjFloatingFilterDay');
        var fsm = document.getElementById('pbjFloatingStartMonth');
        var fem = document.getElementById('pbjFloatingEndMonth');
        var fq = document.getElementById('pbjFloatingQuarterSelect');
        var fy = document.getElementById('pbjFloatingYearSelect');
        var qr = document.getElementById('quarterRange');
        var yr = document.getElementById('years');
        if (filterType === 'quarters' && fq && qr) {
            pbjV2MirrorSelectByValue(fq, qr);
            if (typeof global.onQuarterChange === 'function') {
                global.onQuarterChange();
            }
        }
        if (filterType === 'years' && fy && yr) {
            pbjV2MirrorSelectByValue(fy, yr);
            if (typeof global.onYearChange === 'function') {
                global.onYearChange();
            }
        }
        if (filterType === 'months') {
            if (sm && fsm) {
                sm.value = fsm.value || '';
            }
            if (em && fem) {
                em.value = fem.value || '';
            }
            if (typeof global.onMonthChange === 'function') {
                global.onMonthChange();
            }
        }
        if (filterType === 'daterange') {
            if (sd && fsd) {
                sd.value = fsd.value || '';
            }
            if (ed && fed) {
                ed.value = fed.value || '';
            }
        }
        if (filterType === 'day' && fd && ffd) {
            fd.value = ffd.value || '';
        }
    }

    function pbjV2FloatingApplyPeriod() {
        pbjV2PushFloatingPeriodToSummary();
        if (typeof global.pbjV2RefreshScopeLabel === 'function') {
            global.pbjV2RefreshScopeLabel();
        }
        if (typeof global.applyFilters === 'function') {
            global.applyFilters();
        }
    }

    global.pbjV2FloatingApplyPeriod = pbjV2FloatingApplyPeriod;
    global.pbjV2PullFloatingPeriodFromSummary = pbjV2PullFloatingPeriodFromSummary;
    global.pbjV2RefreshFloatingPickerOptions = pbjV2RefreshFloatingPickerOptions;

    function pbjV2WireFloatingPeriod() {
        document.querySelectorAll('[data-pbj-floating-grain]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var grain = btn.getAttribute('data-pbj-floating-grain');
                var radio = document.querySelector('input[name="filterType"][value="' + grain + '"]');
                if (!radio) {
                    return;
                }
                radio.checked = true;
                if (typeof global.onFilterTypeChange === 'function') {
                    global.onFilterTypeChange();
                }
                pbjV2SyncFloatingGrain();
                pbjV2PullFloatingPeriodFromSummary();
            });
        });
        document.querySelectorAll('input[name="filterType"]').forEach(function (radio) {
            radio.addEventListener('change', function () {
                pbjV2SyncFloatingGrain();
                pbjV2PullFloatingPeriodFromSummary();
            });
        });
        ['quarterRange', 'years', 'pbjFloatingQuarterSelect', 'pbjFloatingYearSelect'].forEach(function (id) {
            var el = document.getElementById(id);
            if (!el) {
                return;
            }
            pbjWireMultiSelectShiftRange(el);
            if (id === 'quarterRange' || id === 'years') {
                el.addEventListener('change', function () {
                    pbjV2RefreshFloatingPickerOptions();
                });
            }
        });
        ['pbjFloatingStartDate', 'pbjFloatingEndDate', 'pbjFloatingFilterDay',
            'pbjFloatingStartMonth', 'pbjFloatingEndMonth'].forEach(function (id) {
            var el = document.getElementById(id);
            if (el) {
                el.addEventListener('change', function () {
                    pbjV2PushFloatingPeriodToSummary();
                });
            }
        });
    }

    function pbjV2RefreshAiToolkitScopeLine() {
        var el = document.getElementById('pbjAiToolkitScopeLine');
        if (!el) {
            return;
        }
        var modal = document.getElementById('pbjAiToolkitModal');
        if (modal && modal.classList.contains('show')) {
            el.textContent = pbjV2AiToolkitScopeLabel() || '—';
            return;
        }
        var label = typeof pbjV2ScopeLabelFromDom === 'function' ? pbjV2ScopeLabelFromDom() : '';
        if (!label || label.indexOf('All ') === 0) {
            label = typeof pbjV2ScopeLabelFromFilterInfo === 'function'
                ? pbjV2ScopeLabelFromFilterInfo()
                : label;
        }
        el.textContent = label || 'Full loaded history';
    }

    function pbjV2AiScopeIsBroad(label) {
        var t = String(label || '').trim();
        if (!t || t === 'Full loaded history') {
            return true;
        }
        return /^(All quarters|All years|Custom date range|Single day)$/i.test(t);
    }

    function pbjV2SyncAiScopeCollapse(label) {
        /* scope controls are always inline in the AI modal */
    }

    function pbjV2SyncAiScopeGrainButtons(grain) {
        document.querySelectorAll('[data-pbj-ai-grain]').forEach(function (btn) {
            var on = btn.getAttribute('data-pbj-ai-grain') === grain;
            btn.classList.toggle('active', on);
            btn.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
    }

    function pbjV2SyncAiScopePanels(grain) {
        var q = document.getElementById('pbjAiScopeQuarters');
        var y = document.getElementById('pbjAiScopeYears');
        var r = document.getElementById('pbjAiScopeRange');
        var d = document.getElementById('pbjAiScopeDay');
        if (q) {
            q.classList.toggle('d-none', grain !== 'quarters');
        }
        if (y) {
            y.classList.toggle('d-none', grain !== 'years');
        }
        if (r) {
            r.classList.toggle('d-none', grain !== 'daterange');
        }
        if (d) {
            d.classList.toggle('d-none', grain !== 'day');
        }
    }

    function pbjV2EnsureAiSelectPlaceholder(selectId, label) {
        var sel = document.getElementById(selectId);
        if (!sel) {
            return;
        }
        var hasSelection = Array.prototype.some.call(sel.options, function (o) {
            return o.selected && String(o.value || '').trim();
        });
        var first = sel.options[0];
        if (first && first.value === '' && first.disabled) {
            first.textContent = label;
            first.selected = !hasSelection;
            return;
        }
        var ph = document.createElement('option');
        ph.value = '';
        ph.disabled = true;
        ph.selected = !hasSelection;
        ph.textContent = label;
        sel.insertBefore(ph, sel.firstChild);
    }

    function pbjV2RefreshAiQuarterSelectOptions() {
        pbjV2CopySelectOptions('quarterRange', 'pbjAiQuarterSelect');
        var dst = document.getElementById('pbjAiQuarterSelect');
        if (dst && pbjV2FloatingSelectNeedsPopulate('pbjAiQuarterSelect', dst)) {
            pbjV2FetchFloatingQuarterOptionsIfNeeded(dst);
        }
        pbjV2EnsureAiSelectPlaceholder('pbjAiQuarterSelect', 'Select quarters…');
    }

    function pbjV2RefreshAiYearSelectOptions() {
        pbjV2CopySelectOptions('years', 'pbjAiYearSelect');
        var dst = document.getElementById('pbjAiYearSelect');
        if (dst && pbjV2FloatingSelectNeedsPopulate('pbjAiYearSelect', dst)) {
            var cached = pbjV2GetCachedFacilityQuarters();
            if (cached) {
                pbjV2BuildFloatingYearOptions(dst, cached);
                var yr = pbjV2BestOptionsSource('years');
                if (yr) {
                    pbjV2MirrorSelectByValue(yr, dst);
                }
            }
        }
        pbjV2EnsureAiSelectPlaceholder('pbjAiYearSelect', 'Select years…');
    }

    function pbjV2GetActiveAiGrain() {
        var grainBtn = document.querySelector('[data-pbj-ai-grain].active');
        return grainBtn ? grainBtn.getAttribute('data-pbj-ai-grain') : 'daterange';
    }

    function pbjV2SyncAiFocusDatesUiForGrain(grain) {
        var focusBlock = document.getElementById('pbjAiFocusDatesBlock');
        if (focusBlock) {
            focusBlock.classList.toggle('d-none', grain === 'day');
        }
    }

    function pbjV2AiModalIsoBounds() {
        var grain = pbjV2GetActiveAiGrain();
        if (grain === 'daterange') {
            var sd = document.getElementById('pbjAiStartDate');
            var ed = document.getElementById('pbjAiEndDate');
            var s = sd ? String(sd.value || '').trim() : '';
            var e = ed ? String(ed.value || '').trim() : '';
            if (s && e) {
                return { start: s, end: e, grain: grain };
            }
            return null;
        }
        if (grain === 'day') {
            var fd = document.getElementById('pbjAiFilterDay');
            var d = fd ? String(fd.value || '').trim() : '';
            if (d) {
                return { start: d, end: d, grain: grain };
            }
            return null;
        }
        return null;
    }

    function pbjV2FetchDailyRowsByIsoRange(start, end) {
        var api =
            typeof global.pbjApiUrl === 'function'
                ? global.pbjApiUrl('/api/data')
                : '/api/data';
        var q =
            '?start_date=' +
            encodeURIComponent(start) +
            '&end_date=' +
            encodeURIComponent(end);
        return fetch(api + q, { credentials: 'include' })
            .then(function (r) {
                return r.json();
            })
            .then(function (payload) {
                if (payload && Array.isArray(payload.data)) {
                    return payload.data;
                }
                if (payload && Array.isArray(payload.records)) {
                    return payload.records;
                }
                return [];
            })
            .catch(function () {
                return [];
            });
    }

    function pbjV2PullAiScopeFromDashboard() {
        var ft = document.querySelector('input[name="filterType"]:checked');
        var filterType = ft ? ft.value : 'quarters';
        pbjV2SyncAiScopeGrainButtons(filterType);
        pbjV2SyncAiScopePanels(filterType);
        pbjV2SyncAiFocusDatesUiForGrain(filterType);
        pbjV2RefreshAiQuarterSelectOptions();
        pbjV2RefreshAiYearSelectOptions();
        var sd = document.getElementById('startDate');
        var ed = document.getElementById('endDate');
        var fd = document.getElementById('filterDayDate');
        var aiSd = document.getElementById('pbjAiStartDate');
        var aiEd = document.getElementById('pbjAiEndDate');
        var aiFd = document.getElementById('pbjAiFilterDay');
        var aiQ = document.getElementById('pbjAiQuarterSelect');
        var aiY = document.getElementById('pbjAiYearSelect');
        var qr = document.getElementById('quarterRange');
        var yr = document.getElementById('years');
        if (aiSd && sd) {
            aiSd.value = sd.value || '';
        }
        if (aiEd && ed) {
            aiEd.value = ed.value || '';
        }
        if (aiFd && fd) {
            aiFd.value = fd.value || '';
        }
        if (aiQ && qr) {
            pbjV2MirrorSelectByValue(qr, aiQ);
            pbjV2EnsureAiSelectPlaceholder('pbjAiQuarterSelect', 'Select quarters…');
        }
        if (aiY && yr && yr.options && yr.options.length) {
            pbjV2MirrorSelectByValue(yr, aiY);
            pbjV2EnsureAiSelectPlaceholder('pbjAiYearSelect', 'Select years…');
        }
        pbjV2RefreshAiToolkitScopeLine();
    }

    function pbjV2PushAiScopeToDashboard() {
        var grainBtn = document.querySelector('[data-pbj-ai-grain].active');
        var grain = grainBtn ? grainBtn.getAttribute('data-pbj-ai-grain') : 'quarters';
        var radio = document.querySelector('input[name="filterType"][value="' + grain + '"]');
        if (radio) {
            radio.checked = true;
            if (typeof global.onFilterTypeChange === 'function') {
                global.onFilterTypeChange();
            }
        }
        var sd = document.getElementById('startDate');
        var ed = document.getElementById('endDate');
        var fd = document.getElementById('filterDayDate');
        var aiSd = document.getElementById('pbjAiStartDate');
        var aiEd = document.getElementById('pbjAiEndDate');
        var aiFd = document.getElementById('pbjAiFilterDay');
        var aiQ = document.getElementById('pbjAiQuarterSelect');
        var aiY = document.getElementById('pbjAiYearSelect');
        var qr = document.getElementById('quarterRange');
        var yr = document.getElementById('years');
        var singleDay = document.getElementById('singleDayDate');
        if (grain === 'quarters' && aiQ && qr) {
            pbjV2MirrorSelectByValue(aiQ, qr);
            if (typeof global.onQuarterChange === 'function') {
                global.onQuarterChange();
            }
        }
        if (grain === 'years' && aiY && yr) {
            pbjV2MirrorSelectByValue(aiY, yr);
            if (typeof global.onYearChange === 'function') {
                global.onYearChange();
            }
        }
        if (grain === 'daterange') {
            if (sd && aiSd) {
                sd.value = aiSd.value || '';
            }
            if (ed && aiEd) {
                ed.value = aiEd.value || '';
            }
        }
        if (grain === 'day') {
            if (fd && aiFd) {
                fd.value = aiFd.value || '';
            }
            if (singleDay && aiFd && aiFd.value) {
                singleDay.value = aiFd.value;
            }
        }
    }

    function pbjV2AiToolkitApplyScope() {
        pbjV2PushAiScopeToDashboard();
        if (typeof global.applyFilters === 'function') {
            global.applyFilters();
        }
        pbjV2InvalidateAiPackCache();
        pbjV2RefreshAiToolkitScopeLine();
        var grainBtn = document.querySelector('[data-pbj-ai-grain].active');
        var grain = grainBtn ? grainBtn.getAttribute('data-pbj-ai-grain') : '';
        if (grain === 'day' && typeof global.generateSingleDayReport === 'function') {
            global.generateSingleDayReport().finally(function () {
                pbjV2RefreshAiPackPreview();
            });
            return;
        }
        setTimeout(function () {
            pbjV2RefreshAiPackPreview();
        }, 500);
    }

    function pbjV2WireAiToolkitScope() {
        document.querySelectorAll('[data-pbj-ai-grain]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var grain = btn.getAttribute('data-pbj-ai-grain');
                pbjV2SyncAiScopeGrainButtons(grain);
                pbjV2SyncAiScopePanels(grain);
                pbjV2SyncAiFocusDatesUiForGrain(grain);
                if (grain === 'quarters') {
                    pbjV2RefreshAiQuarterSelectOptions();
                } else if (grain === 'years') {
                    pbjV2RefreshAiYearSelectOptions();
                }
                global.__pbjAiPackScopeStale = true;
            });
        });
        var applyBtn = document.getElementById('pbjAiScopeApplyBtn');
        if (applyBtn) {
            applyBtn.addEventListener('click', function () {
                pbjV2AiToolkitApplyScope();
            });
        }
        var aiQ = document.getElementById('pbjAiQuarterSelect');
        var aiY = document.getElementById('pbjAiYearSelect');
        if (aiQ) {
            pbjWireMultiSelectShiftRange(aiQ);
        }
        if (aiY) {
            pbjWireMultiSelectShiftRange(aiY);
        }
    }

    function pbjV2EnsureAiPackDataThen(fn) {
        function finish() {
            var ft = pbjV2CurrentFilterType();
            if (ft === 'day' && typeof global.generateSingleDayReport === 'function') {
                var fd = document.getElementById('filterDayDate');
                var sd = document.getElementById('singleDayDate');
                var iso = fd ? String(fd.value || '').trim() : '';
                if (sd && iso) {
                    sd.value = iso;
                }
                var pack = global.__singleDayReportPayload;
                if (!pack || !pack.date || (iso && pack.date !== iso)) {
                    global.generateSingleDayReport().finally(function () {
                        pbjV2PrefetchAiPackBenchmarks(fn);
                    });
                    return;
                }
            }
            pbjV2PrefetchAiPackBenchmarks(fn);
        }
        function afterRows(rows) {
            if (rows && rows.length) {
                global.currentData = rows;
            }
            finish();
        }
        var modal = document.getElementById('pbjAiToolkitModal');
        var modalOpen = !!(modal && modal.classList.contains('show'));
        var bounds = modalOpen ? pbjV2AiModalIsoBounds() : null;
        if (bounds) {
            pbjV2FetchDailyRowsByIsoRange(bounds.start, bounds.end).then(afterRows);
            return;
        }
        if (modalOpen) {
            pbjV2PushAiScopeToDashboard();
            if (typeof global.applyFilters === 'function') {
                global.applyFilters();
                setTimeout(function () {
                    afterRows(global.currentData || []);
                }, 900);
                return;
            }
        }
        var rows =
            typeof global.currentData !== 'undefined' && global.currentData && global.currentData.length
                ? global.currentData
                : [];
        if (rows.length) {
            finish();
            return;
        }
        if (typeof global.applyFilters === 'function') {
            global.applyFilters();
            setTimeout(finish, 900);
            return;
        }
        finish();
    }

    function pbjV2ReadAiToolkitConfig() {
        var el = document.getElementById('pbjAiToolkitConfig');
        if (!el) {
            return {};
        }
        try {
            return JSON.parse(el.textContent || '{}');
        } catch (_e) {
            return {};
        }
    }

    function pbjV2DownloadClaudeSkillZip(ev) {
        if (ev && typeof ev.preventDefault === 'function') {
            ev.preventDefault();
        }
        var cfg = pbjV2ReadAiToolkitConfig();
        var url = cfg.skillZipUrl || '';
        var statusEl = document.getElementById('pbjAiSkillZipStatus');
        function showStatus(msg, ok) {
            if (!statusEl) {
                return;
            }
            statusEl.textContent = msg;
            statusEl.classList.remove('d-none', 'text-success', 'text-danger');
            statusEl.classList.add(ok ? 'text-success' : 'text-danger');
            global.setTimeout(function () {
                statusEl.classList.add('d-none');
            }, 4000);
        }
        if (!cfg.skillZipEnabled || !url) {
            showStatus('Claude add-on not available on this server.', false);
            return false;
        }
        if (typeof global.fetch !== 'function') {
            var a = document.createElement('a');
            a.href = url;
            a.download = 'pbj320-staffing-review.zip';
            a.rel = 'noopener';
            document.body.appendChild(a);
            a.click();
            a.remove();
            showStatus('Download started.', true);
            return false;
        }
        showStatus('Preparing download…', true);
        global.fetch(url, { credentials: 'same-origin' })
            .then(function (res) {
                if (!res.ok) {
                    throw new Error('HTTP ' + res.status);
                }
                return res.blob();
            })
            .then(function (blob) {
                var objUrl = global.URL.createObjectURL(blob);
                var link = document.createElement('a');
                link.href = objUrl;
                link.download = 'pbj320-staffing-review.zip';
                document.body.appendChild(link);
                link.click();
                link.remove();
                global.setTimeout(function () {
                    global.URL.revokeObjectURL(objUrl);
                }, 1000);
                showStatus('Download started.', true);
            })
            .catch(function () {
                showStatus('Could not download. Try again from the dashboard menu.', false);
            });
        return false;
    }

    function pbjV2CopyTextToClipboard(text, statusId) {
        var msg = String(text || '');
        if (!msg) {
            return Promise.resolve(false);
        }
        function showOk() {
            if (!statusId) {
                return;
            }
            var st = document.getElementById(statusId);
            if (!st) {
                return;
            }
            st.textContent = 'Copied';
            st.classList.remove('d-none');
            setTimeout(function () {
                st.classList.add('d-none');
            }, 2800);
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(msg).then(function () {
                showOk();
                return true;
            }).catch(function () {
                return false;
            });
        }
        var ta = document.createElement('textarea');
        ta.value = msg;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        try {
            document.execCommand('copy');
            showOk();
            document.body.removeChild(ta);
            return Promise.resolve(true);
        } catch (_e2) {
            document.body.removeChild(ta);
            return Promise.resolve(false);
        }
    }

    function pbjV2AiPackMetaFromPage() {
        var page = global._pbj320Page || {};
        var ccn = String(global.PBJ320_EXPORT_CCN || page.exportCcn || '').trim();
        var name = String(global.PBJ320_EXPORT_FACILITY_DISPLAY || page.exportFacilityDisplay || '').trim();
        if (!name && typeof global.pbj320ExportFacilityName === 'function') {
            name = String(global.pbj320ExportFacilityName() || '').trim();
        }
        return { ccn: ccn, facilityName: name };
    }

    function pbjV2GetAiToolkitTool() {
        var picked = document.querySelector('input[name="pbjAiToolkitTool"]:checked');
        return picked ? picked.value : 'claude';
    }

    function pbjV2GetAiToolkitAudience() {
        var picked = document.querySelector('input[name="pbjAiToolkitAudience"]:checked');
        return picked ? picked.value : 'attorney';
    }

    function pbjV2UpdateAiToolkitToolHelp() {
        /* Step 3 tool pills open external chats; help text lives in step lede. */
    }

    function pbjV2FormatIsoUsDash(iso) {
        var p = String(iso || '').trim().split('-');
        if (p.length !== 3 || !/^\d{4}$/.test(p[0])) {
            return '';
        }
        return p[1] + '-' + p[2] + '-' + p[0];
    }

    var pbjAiFocusDatesEntries = [];

    function pbjV2AiFocusDatesBlockEl() {
        return document.getElementById('pbjAiFocusDatesBlock');
    }

    function pbjV2AiFocusDatesHiddenEl() {
        return document.getElementById('pbjAiPromptFocusDates');
    }

    function pbjV2AiFocusDatesFormatEntry(entry) {
        var display = pbjV2FormatIsoUsDash(entry.iso);
        if (!display) {
            return '';
        }
        if (entry.note) {
            return display + ' \u2014 ' + entry.note;
        }
        return display;
    }

    function pbjV2AiFocusDatesSyncHidden() {
        var hidden = pbjV2AiFocusDatesHiddenEl();
        if (!hidden) {
            return;
        }
        hidden.value = pbjAiFocusDatesEntries.map(pbjV2AiFocusDatesFormatEntry).filter(Boolean).join('; ');
    }

    function pbjV2AiFocusDatesMarkUserEdited() {
        var block = pbjV2AiFocusDatesBlockEl();
        if (block) {
            block.dataset.pbjUserEdited = pbjAiFocusDatesEntries.length ? '1' : '';
        }
    }

    function pbjV2AiFocusDatesNotifyChanged() {
        pbjV2AiFocusDatesSyncHidden();
        pbjV2AiFocusDatesMarkUserEdited();
        pbjV2RefreshAiStarterPromptPreview(global.__pbjLastAiPackMeta || {});
        global.__pbjAiPackScopeStale = true;
    }

    function pbjV2AiFocusDatesRender() {
        var list = document.getElementById('pbjAiFocusDatesList');
        if (!list) {
            pbjV2AiFocusDatesSyncHidden();
            return;
        }
        list.innerHTML = '';
        pbjAiFocusDatesEntries.forEach(function (entry, idx) {
            var chip = document.createElement('div');
            chip.className = 'pbj-ai-focus-date-chip';
            chip.setAttribute('role', 'listitem');

            var dateSpan = document.createElement('span');
            dateSpan.className = 'pbj-ai-focus-date-chip__date';
            dateSpan.textContent = pbjV2FormatIsoUsDash(entry.iso) || entry.iso;

            chip.appendChild(dateSpan);

            if (entry.note) {
                var noteSpan = document.createElement('span');
                noteSpan.className = 'pbj-ai-focus-date-chip__note';
                noteSpan.textContent = entry.note;
                noteSpan.title = entry.note;
                chip.appendChild(noteSpan);
            }

            var removeBtn = document.createElement('button');
            removeBtn.type = 'button';
            removeBtn.className = 'pbj-ai-focus-date-chip__remove';
            removeBtn.setAttribute('aria-label', 'Remove focus date ' + pbjV2FormatIsoUsDash(entry.iso));
            removeBtn.innerHTML = '&times;';
            removeBtn.addEventListener('click', function () {
                pbjAiFocusDatesEntries.splice(idx, 1);
                pbjV2AiFocusDatesRender();
                pbjV2AiFocusDatesNotifyChanged();
            });
            chip.appendChild(removeBtn);

            list.appendChild(chip);
        });
        pbjV2AiFocusDatesSyncHidden();
    }

    function pbjV2AiFocusDatesAddFromPicker() {
        var picker = document.getElementById('pbjAiFocusDatePicker');
        var noteEl = document.getElementById('pbjAiFocusDateNote');
        var iso = picker ? String(picker.value || '').trim() : '';
        if (!iso || !/^\d{4}-\d{2}-\d{2}$/.test(iso)) {
            if (picker) {
                picker.focus();
            }
            return false;
        }
        var note = noteEl ? String(noteEl.value || '').trim() : '';
        var exists = pbjAiFocusDatesEntries.some(function (entry) {
            return entry.iso === iso && (entry.note || '') === note;
        });
        if (!exists) {
            pbjAiFocusDatesEntries.push({ iso: iso, note: note });
            pbjAiFocusDatesEntries.sort(function (a, b) {
                return a.iso < b.iso ? -1 : a.iso > b.iso ? 1 : 0;
            });
        }
        if (noteEl) {
            noteEl.value = '';
        }
        if (picker) {
            picker.value = '';
        }
        pbjV2AiFocusDatesRender();
        pbjV2AiFocusDatesNotifyChanged();
        return true;
    }

    function pbjV2AiFocusDatesSetFromRows(rows, force) {
        var block = pbjV2AiFocusDatesBlockEl();
        if (!block) {
            return;
        }
        if (!force && block.dataset.pbjUserEdited === '1') {
            return;
        }
        var next = [];
        (rows || []).forEach(function (row) {
            var iso = String((row && row.date) || '').trim();
            var note = String((row && row.note) || '').trim();
            if (!iso && !note) {
                return;
            }
            if (iso && !/^\d{4}-\d{2}-\d{2}$/.test(iso)) {
                return;
            }
            if (!iso) {
                return;
            }
            var type = String((row && row.type) || '').trim();
            if (type && type !== 'other' && type !== 'incident' && !note) {
                note = type.replace(/_/g, ' ');
            }
            var dup = next.some(function (entry) {
                return entry.iso === iso && (entry.note || '') === note;
            });
            if (!dup) {
                next.push({ iso: iso, note: note });
            }
        });
        pbjAiFocusDatesEntries = next;
        pbjV2AiFocusDatesRender();
        pbjV2AiFocusDatesSyncHidden();
        if (!force && !next.length) {
            block.dataset.pbjUserEdited = '';
        }
    }

    function pbjV2WireAiFocusDates() {
        var addBtn = document.getElementById('pbjAiFocusDateAddBtn');
        var picker = document.getElementById('pbjAiFocusDatePicker');
        var noteEl = document.getElementById('pbjAiFocusDateNote');
        if (addBtn && addBtn.dataset.pbjBound !== '1') {
            addBtn.dataset.pbjBound = '1';
            addBtn.addEventListener('click', function () {
                pbjV2AiFocusDatesAddFromPicker();
            });
        }
        if (noteEl && noteEl.dataset.pbjBound !== '1') {
            noteEl.dataset.pbjBound = '1';
            noteEl.addEventListener('keydown', function (ev) {
                if (ev.key === 'Enter') {
                    ev.preventDefault();
                    pbjV2AiFocusDatesAddFromPicker();
                }
            });
        }
        var dayNoteEl = document.getElementById('pbjAiDayNote');
        if (dayNoteEl && dayNoteEl.dataset.pbjBound !== '1') {
            dayNoteEl.dataset.pbjBound = '1';
            dayNoteEl.addEventListener('input', function () {
                pbjV2RefreshAiStarterPromptPreview(global.__pbjLastAiPackMeta || {});
                global.__pbjAiPackScopeStale = true;
            });
        }
        ['pbjAiStartDate', 'pbjAiEndDate', 'pbjAiFilterDay'].forEach(function (id) {
            var el = document.getElementById(id);
            if (!el || el.dataset.pbjAiScopeBound === '1') {
                return;
            }
            el.dataset.pbjAiScopeBound = '1';
            el.addEventListener('change', function () {
                global.__pbjAiPackScopeStale = true;
                pbjV2RefreshAiToolkitScopeLine();
                clearTimeout(global.__pbjAiPackRefreshTimer);
                global.__pbjAiPackRefreshTimer = setTimeout(function () {
                    pbjV2RefreshAiPackPreview();
                }, 450);
            });
        });
        if (picker && picker.dataset.pbjBound !== '1') {
            picker.dataset.pbjBound = '1';
            picker.addEventListener('keydown', function (ev) {
                if (ev.key === 'Enter') {
                    ev.preventDefault();
                    pbjV2AiFocusDatesAddFromPicker();
                }
            });
        }
    }

    function pbjV2AiPromptFocusDatesText() {
        if (pbjV2GetActiveAiGrain() === 'day') {
            var dayNoteEl = document.getElementById('pbjAiDayNote');
            var dayNote = dayNoteEl ? String(dayNoteEl.value || '').trim() : '';
            var fd = document.getElementById('pbjAiFilterDay');
            var iso = fd ? String(fd.value || '').trim() : '';
            if (!iso && !dayNote) {
                return '';
            }
            var display = iso ? pbjV2FormatIsoUsDash(iso) || iso : '';
            if (display && dayNote) {
                return display + ' \u2014 ' + dayNote;
            }
            return display || dayNote;
        }
        pbjV2AiFocusDatesSyncHidden();
        var el = pbjV2AiFocusDatesHiddenEl();
        return el ? String(el.value || '').trim() : '';
    }

    function pbjV2AiToolkitScopeLabel() {
        var grainBtn = document.querySelector('[data-pbj-ai-grain].active');
        var grain = grainBtn ? grainBtn.getAttribute('data-pbj-ai-grain') : '';
        if (grain === 'daterange') {
            var sd = document.getElementById('pbjAiStartDate');
            var ed = document.getElementById('pbjAiEndDate');
            var s = sd ? String(sd.value || '').trim() : '';
            var e = ed ? String(ed.value || '').trim() : '';
            if (s && e) {
                return pbjV2FormatIsoShort(s) + ' – ' + pbjV2FormatIsoShort(e);
            }
            if (s) {
                return 'from ' + pbjV2FormatIsoShort(s);
            }
            if (e) {
                return 'through ' + pbjV2FormatIsoShort(e);
            }
            return 'custom date range';
        }
        if (grain === 'day') {
            var fd = document.getElementById('pbjAiFilterDay');
            var dv = fd ? String(fd.value || '').trim() : '';
            return dv ? ('work day ' + pbjV2FormatIsoShort(dv)) : 'single work day';
        }
        if (grain === 'years') {
            var yrs = pbjV2GetMultiSelectValue('pbjAiYearSelect', 'all');
            if (!yrs || yrs === 'all' || pbjV2AllYearsSelected('pbjAiYearSelect')) {
                return pbjV2YearSpanLabelFromSelect('pbjAiYearSelect');
            }
            var parts = yrs.split(',').map(function (y) {
                return parseInt(String(y).trim(), 10);
            }).filter(function (y) {
                return isFinite(y);
            }).sort(function (a, b) {
                return a - b;
            });
            return pbjV2FormatYearScopeLabel(parts);
        }
        var qv = pbjV2GetMultiSelectValue('pbjAiQuarterSelect', 'all');
        if (!qv || qv === 'all') {
            var line = document.getElementById('pbjAiToolkitScopeLine');
            var fromLine = line ? String(line.textContent || '').trim() : '';
            if (fromLine && fromLine !== '—' && fromLine !== 'Full loaded history') {
                return fromLine;
            }
            return typeof pbjV2ScopeLabelFromDom === 'function' ? pbjV2ScopeLabelFromDom() : 'selected quarters';
        }
        var qKeys = qv.split(',').map(function (q) { return q.trim(); }).filter(Boolean);
        qKeys.sort(function (a, b) {
            return pbjCyQuarterSortKey(a) - pbjCyQuarterSortKey(b);
        });
        if (pbjIsFullCalendarYearQuarters(qKeys)) {
            return qKeys[0].slice(0, 4);
        }
        var qs = qKeys.map(function (q) {
            return pbjV2FormatCyQuarter(q);
        }).filter(Boolean);
        if (qs.length === 1) {
            return qs[0];
        }
        if (qs.length > 3) {
            return qs[0] + ' – ' + qs[qs.length - 1];
        }
        return qs.join(', ');
    }

    function pbjV2DashDisplayText(id) {
        var node = document.getElementById(id);
        if (!node) {
            return '';
        }
        var t = String(node.textContent || '').replace(/\s+/g, ' ').trim();
        return t && t !== '—' && t !== '-' ? t : '';
    }

    function pbjV2ParseBenchGapEl(gapEl) {
        if (!gapEl) {
            return null;
        }
        var raw = String(gapEl.textContent || '').replace(/\s+/g, ' ').trim();
        if (!raw || raw === '—' || raw === '-') {
            return null;
        }
        var numEl = gapEl.querySelector('.pbj-bench-gap-num');
        var dirEl = gapEl.querySelector('.pbj-bench-gap-dir');
        var num = numEl ? String(numEl.textContent || '').replace(/\s+/g, ' ').trim() : '';
        var dir = dirEl ? String(dirEl.textContent || '').trim().toLowerCase() : '';
        if (!num) {
            return null;
        }
        if (!dir) {
            var n = parseFloat(String(num).replace(/[^\d.+-]/g, ''));
            if (isNaN(n)) {
                return null;
            }
            dir = Math.abs(n) <= 0.02 ? 'at' : n > 0 ? 'above' : 'below';
            num = String(Math.abs(n));
        }
        return { num: num, dir: dir };
    }

    function pbjV2BenchGapPhrase(gap, label) {
        if (!gap) {
            return '';
        }
        if (gap.dir === 'at') {
            return 'at the ' + label;
        }
        return gap.num + ' HPRD ' + gap.dir + ' the ' + label;
    }

    function pbjV2ComputePeriodStaffingFromRows(rows) {
        if (!rows || !rows.length) {
            return null;
        }
        var totalHours = 0;
        var directHours = 0;
        var rnHours = 0;
        var lpnHours = 0;
        var naHours = 0;
        var totalCensus = 0;
        var contractSum = 0;
        var contractN = 0;
        rows.forEach(function (r) {
            var census = parseFloat(r.MDScensus);
            if (!isFinite(census) || census <= 0) {
                return;
            }
            totalCensus += census;
            var th = parseFloat(r.Total_Staff_Hours || r.Total_Nurse_Hours || 0);
            if (isFinite(th)) {
                totalHours += th;
            }
            var dh = parseFloat(r.Nurse_Staff_Hours_Excl_Admin || 0);
            if (isFinite(dh)) {
                directHours += dh;
            }
            var rh = parseFloat(r.Total_RN_Hours || 0);
            if (isFinite(rh)) {
                rnHours += rh;
            }
            var lh = parseFloat(r.Total_LPN_Hours || 0);
            if (isFinite(lh)) {
                lpnHours += lh;
            }
            var ah = parseFloat(r.Total_Nurse_Aide_Hours || 0);
            if (isFinite(ah)) {
                naHours += ah;
            }
            var cp = parseFloat(r.RN_Contract_Pct || r.Contract_Percentage || '');
            if (isFinite(cp)) {
                contractSum += cp;
                contractN += 1;
            }
        });
        if (totalCensus <= 0) {
            return null;
        }
        function hprd(hours) {
            return (hours / totalCensus).toFixed(2);
        }
        return {
            workDays: rows.length,
            avgTotalHprd: hprd(totalHours),
            avgDirectHprd: directHours > 0 ? hprd(directHours) : '',
            avgRnHprd: hprd(rnHours),
            avgLpnHprd: hprd(lpnHours),
            avgNaHprd: hprd(naHours),
            avgContractPct: contractN ? (contractSum / contractN).toFixed(1) : ''
        };
    }

    function pbjV2BuildAiPeriodStaffingSnippet(packMeta) {
        packMeta = packMeta || {};
        var bits = [];
        var scope =
            packMeta.filterLabel ||
            pbjV2AiToolkitScopeLabel() ||
            (typeof pbjV2ScopeLabelFromDom === 'function' ? pbjV2ScopeLabelFromDom() : '') ||
            'selected period';
        var rows =
            typeof global.currentData !== 'undefined' && global.currentData && global.currentData.length
                ? global.currentData
                : [];
        var computed = pbjV2ComputePeriodStaffingFromRows(rows);
        var totalH = pbjV2DashDisplayText('pbjSummaryTotalHprdDisplay') || (computed && computed.avgTotalHprd) || '';
        var rnH = pbjV2DashDisplayText('pbjSummaryRnHprdDisplay') || (computed && computed.avgRnHprd) || '';
        var directH = '';
        var totalDisp = document.getElementById('pbjSummaryTotalHprdDisplay');
        if (totalDisp) {
            var tip = totalDisp.getAttribute('title') || '';
            var tm = tip.match(/Direct care HPRD:\s*([\d.]+)/i);
            if (tm) {
                directH = tm[1];
            }
        }
        if (!directH && computed && computed.avgDirectHprd) {
            directH = computed.avgDirectHprd;
        }
        var contract = pbjV2DashDisplayText('contractPct') || (computed && computed.avgContractPct) || '';
        var workDays =
            packMeta.dailyRowCount != null
                ? packMeta.dailyRowCount
                : computed
                    ? computed.workDays
                    : rows.length;
        if (workDays != null && Number.isFinite(Number(workDays)) && Number(workDays) > 0) {
            bits.push(
                workDays +
                    ' work day' +
                    (Number(workDays) === 1 ? '' : 's') +
                    ' in scope (' +
                    scope +
                    ')'
            );
        }
        if (totalH) {
            bits.push('weighted avg total nurse HPRD ' + totalH);
        }
        if (directH) {
            bits.push('direct-care HPRD ' + directH);
        }
        if (rnH) {
            bits.push('RN HPRD ' + rnH);
        }
        if (contract) {
            bits.push('contract share ' + contract + (String(contract).indexOf('%') >= 0 ? '' : '%'));
        }
        var cmGap = pbjV2ParseBenchGapEl(document.getElementById('forensicCaseMixDelta'));
        if (cmGap) {
            bits.push('vs CMS case-mix expected: ' + pbjV2BenchGapPhrase(cmGap, 'case-mix benchmark'));
        }
        var harGap = pbjV2ParseBenchGapEl(document.getElementById('forensicHarringtonDelta'));
        if (harGap) {
            bits.push('vs Harrington expected: ' + pbjV2BenchGapPhrase(harGap, 'Harrington benchmark'));
        }
        var countyGap = pbjV2ParseBenchGapEl(document.getElementById('forensicGeoPeerCountyDelta'));
        var stateGap = pbjV2ParseBenchGapEl(document.getElementById('forensicGeoPeerStateDelta'));
        if (countyGap) {
            bits.push('vs county peer median: ' + pbjV2BenchGapPhrase(countyGap, 'county median'));
        }
        if (stateGap) {
            bits.push('vs state peer median: ' + pbjV2BenchGapPhrase(stateGap, 'state median'));
        }
        return bits;
    }

    function pbjV2PrefetchAiPackBenchmarks(fn) {
        var waits = [];
        if (typeof global.loadCaseMixData === 'function') {
            var cm = global.__pbjCaseMixDataByQuarter || {};
            if (!global.__pbjCaseMixLoadSuccess || !Object.keys(cm).length) {
                waits.push(Promise.resolve(global.loadCaseMixData()).catch(function () {}));
            }
        }
        if (typeof global.pbjPrefetchHarringtonCmiExport === 'function') {
            var har = global.__lastHarringtonCmiExport;
            if (!har || !har.rows || !har.rows.length) {
                waits.push(Promise.resolve(global.pbjPrefetchHarringtonCmiExport()).catch(function () {}));
            }
        }
        if (!waits.length) {
            if (typeof fn === 'function') {
                fn();
            }
            return;
        }
        Promise.all(waits).finally(function () {
            if (typeof fn === 'function') {
                fn();
            }
        });
    }

    function pbjV2BuildAiStarterPrompt(audienceKey, packMeta) {
        packMeta = packMeta || {};
        var aud = String(audienceKey || 'attorney').trim() || 'attorney';
        var meta = pbjV2AiPackMetaFromPage();
        var fn = packMeta.facilityName || meta.facilityName || 'this nursing home';
        var ccn = packMeta.ccn || meta.ccn || '';
        var scope =
            packMeta.filterLabel ||
            pbjV2AiToolkitScopeLabel() ||
            (typeof pbjV2ScopeLabelFromDom === 'function' ? pbjV2ScopeLabelFromDom() : '') ||
            'selected period';
        var workDays = packMeta.dailyRowCount;
        if (workDays == null && global.currentData && global.currentData.length) {
            workDays = global.currentData.length;
        }
        var workDaysText = workDays != null && Number.isFinite(Number(workDays)) ? String(workDays) : 'pending export';
        var qtrs = Array.isArray(packMeta.quarters) ? packMeta.quarters.join(', ') : (packMeta.quarters || '');
        var focusDates = packMeta.focusDates || pbjV2AiPromptFocusDatesText();
        var staffingBits = pbjV2BuildAiPeriodStaffingSnippet(packMeta);
        var audienceLine =
            aud === 'attorney'
                ? 'Written for an attorney: include limitations, timing context, and records worth requesting. Do not state legal violations or causation.'
                : aud === 'journalist'
                    ? 'Written for a journalist: lead with what is supportable and what still needs verification before publication.'
                    : aud === 'researcher'
                        ? 'Written for a researcher: emphasize methods, denominators, and caveats (census, case-mix).'
                        : 'Written for an analyst: clear summary of reported staffing vs benchmarks.';
        var lines = [
            'Review CMS Payroll-Based Journal staffing for ' + fn + (ccn ? ' (CCN ' + ccn + ').' : '.'),
            audienceLine,
            'Period: ' + scope + '.',
            'Attached: PBJ320 export CSV (' + workDaysText + ' work day' + (workDaysText === '1' ? '' : 's') + (qtrs ? '; quarters ' + qtrs : '') + ').',
            'CSV sections include DAILY staffing for this filter plus CASE_MIX, HARRINGTON, and GEO_PEER regional rollups when loaded.'
        ];
        if (staffingBits.length) {
            lines.push('Selected-period staffing (active dashboard filter): ' + staffingBits.join('; ') + '.');
        }
        if (focusDates) {
            lines.push('Dates of interest: ' + focusDates + '.');
        }
        lines.push(
            'Read the scope and column-definition rows at the top of the file before analyzing daily staffing.',
            '',
            'Guidelines:',
            '- PBJ is facility-reported payroll data only.',
            '- Say what the data shows, what it may suggest, and what it cannot establish.',
            '- Treat red flags as screening only until confirmed.',
            '- Prefer charts or small multiples over tables for the main visual.',
            '',
            'Please:',
            '1. Summarize total and RN staffing for this period vs case-mix, Harrington, and regional peer rows in the CSV.',
            '2. Note unusual days, census effects, contract staff share, and county/state/region comparisons when present.',
            '3. Propose one primary data visualization (line chart, bar chart, timeline, or small multiples — not a table) that best communicates the main staffing story for this period. Choose the chart type from the data (e.g., daily HPRD trend, RN vs aide mix, case-mix or Harrington gap by quarter, facility vs regional peer). Describe axes, series, highlights, and why that visual fits.',
            '4. List what this dataset cannot establish.',
            '5. End with up to five practical follow-up questions or records to request.'
        );
        return lines.join('\n');
    }

    function pbjV2RefreshAiStarterPromptPreview(packMeta) {
        var text = pbjV2BuildAiStarterPrompt(
            pbjV2GetAiToolkitAudience(),
            packMeta || global.__pbjLastAiPackMeta || {}
        );
        ['pbjAiToolkitPromptLive', 'pbjAiToolkitPromptPreview'].forEach(function (id) {
            var pre = document.getElementById(id);
            if (pre) {
                pre.textContent = text;
            }
        });
    }

    function pbjV2CollectAiPackUserContext() {
        var ctx = { key_dates: [], user_items: [], focus_dates: pbjV2AiPromptFocusDatesText() };
        if (typeof global.pbjRb3CollectAiPackContext === 'function') {
            try {
                var rb3 = global.pbjRb3CollectAiPackContext() || {};
                ctx.key_dates = rb3.key_dates || [];
                ctx.user_items = rb3.user_items || [];
            } catch (_eRb3) { /* ignore */ }
        }
        if (!ctx.key_dates.length) {
            var root = document.getElementById('rb3KeyDatesList');
            if (root) {
                root.querySelectorAll('.rb3-keydate-row').forEach(function (row) {
                    var date = ((row.querySelector('.rb3-keydate-date') || {}).value || '').trim();
                    var note = ((row.querySelector('.rb3-keydate-label') || {}).value || '').trim();
                    var type = ((row.querySelector('.rb3-keydate-type') || {}).value || 'other').trim();
                    if (date || note) {
                        ctx.key_dates.push({ date: date, note: note, type: type });
                    }
                });
            }
        }
        return ctx;
    }

    function pbjV2AppendAiPackContextRows(pushRow, out) {
        var ctx = pbjV2CollectAiPackUserContext();
        if (ctx.focus_dates) {
            pushRow(out, 'CONTEXT', 'focus', '', 'focus_dates', ctx.focus_dates, '', 'User focus dates for AI review', 'PBJ320');
        }
        (ctx.key_dates || []).forEach(function (row, idx) {
            var period = row.date || String(idx + 1);
            if (row.type) {
                pushRow(out, 'KEY_DATE', 'row', period, 'type', row.type, '', '', 'Case Builder');
            }
            if (row.note) {
                pushRow(out, 'KEY_DATE', 'row', period, 'note', row.note, '', '', 'Case Builder');
            }
            if (row.date) {
                pushRow(out, 'KEY_DATE', 'row', period, 'date', row.date, 'iso', '', 'Case Builder');
            }
        });
        (ctx.user_items || []).forEach(function (item, idx) {
            var period = String(idx + 1);
            if (item.title) {
                pushRow(out, 'INCIDENT', 'item', period, 'title', item.title, '', '', item.source || 'user');
            }
            if (item.subtitle) {
                pushRow(out, 'INCIDENT', 'item', period, 'detail', item.subtitle, '', '', item.source || 'user');
            }
            if (item.category) {
                pushRow(out, 'INCIDENT', 'item', period, 'category', item.category, '', '', item.source || 'user');
            }
        });
    }

    function pbjV2AiPackScopeFingerprint() {
        var ft = document.querySelector('input[name="filterType"]:checked');
        var filterType = ft ? ft.value : '';
        var day = document.getElementById('filterDayDate');
        var sd = document.getElementById('startDate');
        var ed = document.getElementById('endDate');
        var qv = pbjV2GetMultiSelectValue('quarterRange', 'all');
        var yv = pbjV2GetMultiSelectValue('years', 'all');
        return JSON.stringify({
            filterType: filterType,
            day: day ? day.value : '',
            start: sd ? sd.value : '',
            end: ed ? ed.value : '',
            quarters: qv,
            years: yv,
            lastFilters: String(global.lastFilters || ''),
            rowCount: (global.currentData && global.currentData.length) || 0
        });
    }

    function pbjV2InvalidateAiPackCache() {
        global.__pbjLastAiContextPackCsv = null;
        global.__pbjAiPackCacheFingerprint = null;
        global.__pbjAiPackScopeStale = true;
    }

    function pbjV2OnDashboardScopeChanged() {
        var fp = pbjV2AiPackScopeFingerprint();
        if (
            global.__pbjAiPackCacheFingerprint &&
            global.__pbjAiPackCacheFingerprint !== fp
        ) {
            pbjV2InvalidateAiPackCache();
        }
        pbjV2RefreshAiToolkitScopeLine();
        var modal = document.getElementById('pbjAiToolkitModal');
        if (modal && modal.classList.contains('show')) {
            clearTimeout(global.__pbjAiPackRefreshTimer);
            global.__pbjAiPackRefreshTimer = setTimeout(function () {
                pbjV2RefreshAiPackPreview();
            }, 400);
        }
    }

    function pbjV2CurrentFilterType() {
        var ft = document.querySelector('input[name="filterType"]:checked');
        return ft ? ft.value : 'quarters';
    }

    function pbjV2RenderAiPackAdvisories(packMeta) {
        var ul = document.getElementById('pbjAiToolkitAdvisories');
        if (!ul) {
            return;
        }
        packMeta = packMeta || {};
        var counts = packMeta.sectionCounts || {};
        var workDays = packMeta.dailyRowCount != null ? packMeta.dailyRowCount : 0;
        var filterType = pbjV2CurrentFilterType();
        var audience = pbjV2GetAiToolkitAudience();
        var items = [];

        if (global.__pbjAiPackScopeStale) {
            items.push('The period changed — download or copy the file again.');
        }
        if (!workDays) {
            var aiBounds = pbjV2AiModalIsoBounds();
            if (aiBounds) {
                items.push('No work days in this period for the dates above. Try a different range or day.');
            } else {
                items.push('Pick quarters, years, a date range, or a day above, then click Apply.');
            }
        }
        if (filterType === 'day' && workDays === 1 && !(counts.SINGLE_DAY > 0)) {
            items.push('Day benchmarks load after the single-day report — click Apply period or reopen.');
        }
        if (filterType === 'day' && !(counts.CASE_MIX > 0) && workDays > 0) {
            items.push('Quarterly case-mix for this day’s quarter is not in the file yet — try Apply period again.');
        }
        if (workDays > 0 && !(counts.META > 0)) {
            items.push('Export looks incomplete — try downloading again.');
        }
        if (workDays > 0 && !(counts.CASE_MIX > 0)) {
            items.push('Case-mix benchmarks are not in the file yet — wait a moment and export again.');
        }
        if (workDays > 0 && !(counts.HARRINGTON > 0)) {
            items.push('Harrington expected staffing is not in the file yet — wait a moment and export again.');
        }
        if (workDays > 0 && !(counts.GEO_PEER > 0)) {
            items.push('Regional peer rollups are not bundled for this facility — daily staffing still exports.');
        }

        if (!items.length) {
            ul.classList.add('d-none');
            ul.innerHTML = '';
            return;
        }
        ul.innerHTML = items.map(function (text) {
            return '<li class="pbj-ai-toolkit-advisory">' + String(text).replace(/</g, '&lt;') + '</li>';
        }).join('');
        ul.classList.remove('d-none');
    }

    function pbjV2RenderAiPackPreview(packMeta) {
        packMeta = packMeta || {};
        global.__pbjLastAiPackMeta = packMeta;
        var summaryEl = document.getElementById('pbjAiToolkitSummary');
        var pre = document.getElementById('pbjAiPackPreview');
        var counts = packMeta.sectionCounts || {};
        var totalRows = 0;
        Object.keys(counts).forEach(function (k) {
            totalRows += counts[k] || 0;
        });
        var workDays = packMeta.dailyRowCount != null ? packMeta.dailyRowCount : 0;
        var dailyLines = counts.DAILY || 0;
        var filterType = pbjV2CurrentFilterType();

        pbjV2RenderAiPackAdvisories(packMeta);
        if (summaryEl) {
            if (!totalRows) {
                summaryEl.textContent = 'No export for this scope — pick a period above and Apply.';
                summaryEl.className = 'small text-warning fw-semibold text-body mb-2';
            } else {
                var parts = [];
                if (workDays) {
                    parts.push(workDays + ' work day' + (workDays === 1 ? '' : 's'));
                }
                if (dailyLines) {
                    parts.push(dailyLines.toLocaleString() + ' daily metric rows');
                }
                if (filterType === 'day' && (counts.SINGLE_DAY || 0) > 0) {
                    parts.push('day benchmarks included');
                } else {
                    if ((counts.CASE_MIX || 0) > 0) {
                        parts.push('case-mix included');
                    }
                    if ((counts.HARRINGTON || 0) > 0) {
                        parts.push('Harrington included');
                    }
                    if ((counts.GEO_PEER || 0) > 0) {
                        parts.push('regional peers included');
                    }
                }
                parts.push(totalRows.toLocaleString() + ' CSV lines');
                summaryEl.textContent = parts.join(' · ');
                summaryEl.className = 'small fw-semibold text-body mb-2';
            }
        }
        if (pre) {
            var samples = packMeta.sampleLines || [];
            if (!samples.length) {
                pre.textContent = 'No sample rows for this scope.';
            } else {
                pre.textContent = samples.join('\n');
            }
        }
        pbjV2RefreshAiStarterPromptPreview(packMeta);
        global.__pbjAiPackCacheFingerprint = pbjV2AiPackScopeFingerprint();
        global.__pbjAiPackScopeStale = false;
    }

    function pbjV2UpdateAiToolkitSummaryIdle() {
        var summaryEl = document.getElementById('pbjAiToolkitSummary');
        if (!summaryEl) {
            return;
        }
        if (global.__pbjLastAiPackMeta && global.__pbjLastAiPackMeta.dailyRowCount && !global.__pbjAiPackScopeStale) {
            pbjV2RenderAiPackAdvisories(global.__pbjLastAiPackMeta || {});
            return;
        }
        summaryEl.textContent = 'Use the buttons below when you are ready to export.';
        summaryEl.className = 'small text-body-secondary mb-2';
        var ul = document.getElementById('pbjAiToolkitAdvisories');
        if (ul) {
            ul.classList.add('d-none');
            ul.innerHTML = '';
        }
    }

    function pbjV2RefreshAiPackPreview() {
        var pre = document.getElementById('pbjAiPackPreview');
        if (!pre || typeof global.exportPbj320AiContextPack !== 'function') {
            return;
        }
        pbjV2EnsureAiPackDataThen(function () {
            global.__pbjAiPackPreviewOnly = true;
            try {
                global.exportPbj320AiContextPack();
            } finally {
                global.__pbjAiPackPreviewOnly = false;
            }
        });
    }

    function pbjV2CopyAiStarterPrompt() {
        var text = pbjV2BuildAiStarterPrompt(
            pbjV2GetAiToolkitAudience(),
            global.__pbjLastAiPackMeta || pbjV2AiPackMetaFromPage() || {}
        );
        var btns = document.querySelectorAll('#pbjAiPackCopyPromptBtn, #pbjAiPackCopyPromptBtnInline');
        pbjV2CopyTextToClipboard(text, 'pbjAiPackPromptStatus').then(function (ok) {
            if (!ok || !btns.length) {
                return;
            }
            Array.prototype.forEach.call(btns, function (btn) {
                var prev = btn.innerHTML;
                btn.innerHTML = '<i class="fas fa-check me-1" aria-hidden="true"></i>Copied';
                btn.setAttribute('aria-label', 'Copied');
                setTimeout(function () {
                    btn.innerHTML = prev;
                    btn.setAttribute('aria-label', 'Copy prompt');
                }, 2200);
            });
        });
    }

    function pbjV2CopyAiContextPackCsv() {
        if (typeof global.exportPbj320AiContextPack !== 'function') {
            return;
        }
        var fp = pbjV2AiPackScopeFingerprint();
        if (
            global.__pbjLastAiContextPackCsv &&
            global.__pbjAiPackCacheFingerprint === fp &&
            !global.__pbjAiPackScopeStale
        ) {
            pbjV2CopyTextToClipboard(global.__pbjLastAiContextPackCsv, 'pbjAiPackCopyStatus');
            return;
        }
        pbjV2EnsureAiPackDataThen(function () {
            global.__pbjAiPackWantClipboard = true;
            global.exportPbj320AiContextPack();
            global.__pbjAiPackWantClipboard = false;
        });
    }

    function pbjV2FormatIsoShort(iso) {
        var p = String(iso || '').trim().split('-');
        if (p.length !== 3) {
            return iso || '';
        }
        return p[1] + '/' + p[2] + '/' + p[0];
    }

    function pbjV2FormatCyQuarter(q) {
        var s = String(q || '').trim();
        if (!s) {
            return s;
        }
        var cy = s.match(/^CY(\d{4})Q([1-4])$/i);
        if (cy) {
            return 'Q' + cy[2] + ' ' + cy[1];
        }
        var plain = s.match(/^(\d{4})Q([1-4])$/i);
        if (plain) {
            return 'Q' + plain[2] + ' ' + plain[1];
        }
        var spaced = s.match(/^Q([1-4])\s+(\d{4})$/i);
        if (spaced) {
            return 'Q' + spaced[1] + ' ' + spaced[2];
        }
        return s;
    }

    function pbjV2GetMultiSelectValue(id, fallback) {
        var el = document.getElementById(id);
        if (!el) {
            return fallback || 'all';
        }
        if (el.tagName === 'SELECT' && el.multiple) {
            var picked = Array.prototype.filter.call(el.options, function (o) {
                return o.selected && o.value && o.value !== 'all';
            }).map(function (o) { return o.value; });
            return picked.length ? picked.join(',') : (fallback || 'all');
        }
        return el.value || fallback || 'all';
    }

    function pbjV2NumericYearsFromSelect(id) {
        var el = document.getElementById(id);
        if (!el || !el.options) {
            return [];
        }
        return Array.prototype.map
            .call(el.options, function (o) {
                return parseInt(String(o.value || '').trim(), 10);
            })
            .filter(function (y) {
                return isFinite(y) && y >= 1900;
            })
            .sort(function (a, b) {
                return a - b;
            });
    }

    function pbjV2AllYearsSelected(id) {
        var el = document.getElementById(id);
        if (!el || !el.options) {
            return false;
        }
        var allOpts = Array.prototype.filter.call(el.options, function (o) {
            return o.value && o.value !== 'all';
        });
        if (!allOpts.length) {
            return false;
        }
        var picked = Array.prototype.filter.call(el.options, function (o) {
            return o.selected && o.value && o.value !== 'all';
        });
        return picked.length === allOpts.length;
    }

    function pbjV2YearSpanLabelFromSelect(id) {
        var years = pbjV2NumericYearsFromSelect(id);
        if (years.length) {
            var lo = years[0];
            var hi = years[years.length - 1];
            return lo === hi ? String(lo) : lo + ' – ' + hi;
        }
        var minIso = String(global.__pbjMinWorkDate || '').trim();
        var maxIso = String(global.__pbjMaxWorkDate || '').trim();
        var minY = minIso.length >= 4 ? parseInt(minIso.slice(0, 4), 10) : NaN;
        var maxY = maxIso.length >= 4 ? parseInt(maxIso.slice(0, 4), 10) : NaN;
        if (isFinite(minY) && isFinite(maxY)) {
            return minY === maxY ? String(minY) : minY + ' – ' + maxY;
        }
        return '2017 – 2025';
    }

    function pbjV2FormatYearScopeLabel(yearsSorted) {
        if (!yearsSorted || !yearsSorted.length) {
            return pbjV2YearSpanLabelFromSelect('years');
        }
        if (yearsSorted.length === 1) {
            return String(yearsSorted[0]);
        }
        return yearsSorted[0] + ' – ' + yearsSorted[yearsSorted.length - 1];
    }

    function pbjV2SelectedQuarterKeysFromDom() {
        var fq = document.getElementById('pbjFloatingQuarterSelect');
        if (fq && fq.options && fq.options.length) {
            var fromFloating = Array.from(fq.selectedOptions || [])
                .map(function (o) { return String(o.value || '').trim(); })
                .filter(function (v) { return v && v !== 'all'; });
            if (fromFloating.length) {
                return fromFloating;
            }
        }
        var qv = pbjV2GetMultiSelectValue('quarterRange', 'all');
        if (!qv || qv === 'all') {
            return [];
        }
        return qv.split(',').map(function (q) { return q.trim(); }).filter(Boolean);
    }

    function pbjV2ScopeLabelFromDom() {
        var ft = document.querySelector('input[name="filterType"]:checked');
        var filterType = ft ? ft.value : 'quarters';
        if (filterType === 'daterange') {
            var sd = document.getElementById('startDate');
            var ed = document.getElementById('endDate');
            var s = sd ? String(sd.value || '').trim() : '';
            var e = ed ? String(ed.value || '').trim() : '';
            if (s && e) {
                return 'Custom: ' + pbjV2FormatIsoShort(s) + ' – ' + pbjV2FormatIsoShort(e);
            }
            if (s) {
                return 'Custom from ' + pbjV2FormatIsoShort(s);
            }
            if (e) {
                return 'Custom through ' + pbjV2FormatIsoShort(e);
            }
            return 'Custom date range';
        }
        if (filterType === 'day') {
            var fd = document.getElementById('filterDayDate');
            var dv = fd ? String(fd.value || '').trim() : '';
            return dv ? 'Day: ' + pbjV2FormatIsoShort(dv) : 'Single day';
        }
        if (filterType === 'months') {
            var sm = document.getElementById('startMonth');
            var em = document.getElementById('endMonth');
            var ms = sm ? String(sm.value || '').trim() : '';
            var me = em ? String(em.value || '').trim() : '';
            if (ms && me) {
                return 'Months: ' + ms + ' – ' + me;
            }
            return ms ? 'Month: ' + ms : 'Months';
        }
        if (filterType === 'years') {
            var yrs = pbjV2GetMultiSelectValue('years', 'all');
            if (!yrs || yrs === 'all' || pbjV2AllYearsSelected('years')) {
                return pbjV2YearSpanLabelFromSelect('years');
            }
            var parts = yrs.split(',').map(function (y) {
                return parseInt(String(y).trim(), 10);
            }).filter(function (y) {
                return isFinite(y);
            }).sort(function (a, b) {
                return a - b;
            });
            return pbjV2FormatYearScopeLabel(parts);
        }
        var qKeys = pbjV2SelectedQuarterKeysFromDom();
        if (!qKeys.length) {
            return 'All quarters';
        }
        qKeys.sort(function (a, b) {
            return pbjCyQuarterSortKey(a) - pbjCyQuarterSortKey(b);
        });
        if (pbjIsFullCalendarYearQuarters(qKeys)) {
            return qKeys[0].slice(0, 4);
        }
        var qs = qKeys.map(function (q) {
            return pbjV2FormatCyQuarter(q);
        }).filter(Boolean);
        if (qs.length === 1) {
            return qs[0];
        }
        if (qs.length > 3) {
            return pbjCompactQuarterScopeLabel(qs[0] + ' – ' + qs[qs.length - 1]);
        }
        return qs.join(', ');
    }

    function pbjCyQuarterSortKey(q) {
        var parts = String(q || '').trim().split('Q');
        if (parts.length !== 2 || !/^\d{4}$/.test(parts[0]) || !/^[1-4]$/.test(parts[1])) {
            return 0;
        }
        return parseInt(parts[0], 10) * 4 + parseInt(parts[1], 10);
    }

    function pbjQuarterDisplaySortKey(label) {
        var m = String(label || '').trim().match(/^Q([1-4])\s+(\d{4})$/i);
        if (!m) {
            return 0;
        }
        return parseInt(m[2], 10) * 4 + parseInt(m[1], 10);
    }

    function pbjCompactQuarterScopeLabel(text) {
        var s = String(text || '').replace(/<br\s*\/?>/gi, ' · ').trim();
        if (!s) {
            return s;
        }
        var allDataM = s.match(/^All Data\s*\(([^)]+)\)\s*$/i);
        if (allDataM) {
            return allDataM[1].trim();
        }
        if (/^All Data$/i.test(s)) {
            return '';
        }
        if (/Q[1-4]\s+\d{4}/i.test(s) && s.indexOf(',') >= 0) {
            var parts = s.split(/\s*,\s*/).map(function (p) { return p.trim(); }).filter(Boolean);
            if (parts.length >= 2 && parts.every(function (p) { return /^Q[1-4]\s+\d{4}$/i.test(p); })) {
                parts.sort(function (a, b) {
                    return pbjQuarterDisplaySortKey(a) - pbjQuarterDisplaySortKey(b);
                });
                if (parts.length === 4) {
                    var yrM = parts[0].match(/\d{4}/);
                    var yr = yrM ? yrM[0] : '';
                    if (yr && /^Q1\s+/i.test(parts[0]) && /^Q4\s+/i.test(parts[3]) && parts.every(function (p) { return p.indexOf(yr) >= 0; })) {
                        return yr;
                    }
                }
                return parts[0] + ' – ' + parts[parts.length - 1];
            }
        }
        var rangeM = s.match(/^(Q[1-4]\s+\d{4})\s*(?:[-–—]|to)\s*(Q[1-4]\s+\d{4})$/i);
        if (rangeM) {
            var left = rangeM[1];
            var right = rangeM[2];
            var kl = pbjQuarterDisplaySortKey(left);
            var kr = pbjQuarterDisplaySortKey(right);
            if (kl && kr && kl > kr) {
                s = right + ' – ' + left;
            } else {
                s = left + ' – ' + right;
            }
        }
        var fullYear = s.match(/^Q1\s+(\d{4})\s*[-–—]\s*Q4\s+\1$/i);
        if (fullYear) {
            return fullYear[1];
        }
        var multiYearM = s.match(/^Q[1-4]\s+(\d{4})\s*[-–—]\s*Q[1-4]\s+(\d{4})$/i);
        if (multiYearM) {
            var startYear = multiYearM[1];
            var endYear = multiYearM[2];
            if (startYear !== endYear) {
                return startYear + ' – ' + endYear;
            }
        }
        return s;
    }

    function pbjIsFullCalendarYearQuarters(quarterKeys) {
        if (!quarterKeys || quarterKeys.length !== 4) {
            return false;
        }
        var years = {};
        var qnums = [];
        for (var i = 0; i < quarterKeys.length; i++) {
            var q = String(quarterKeys[i] || '').trim();
            var parts = q.split('Q');
            if (parts.length !== 2 || !/^\d{4}$/.test(parts[0]) || !/^[1-4]$/.test(parts[1])) {
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

    function pbjV2ScopeLabelFromFilterInfo(clean) {
        var raw = String(clean || '').replace(/<br\s*\/?>/gi, ' · ').trim();
        if (!raw) {
            return 'Applied scope';
        }
        raw = raw.replace(/^\((.*)\)$/, '$1').trim();
        var dom = pbjV2ScopeLabelFromDom();
        if (dom && dom.indexOf('Custom:') === 0) {
            return dom;
        }
        if (/\d{2}-\d{2}-\d{4}\s+to\s+\d{2}-\d{2}-\d{4}/i.test(raw)) {
            return 'Custom: ' + raw.replace(/\s+to\s+/i, ' – ');
        }
        raw = raw.replace(/^Quarters:\s*/i, '').trim();
        return pbjCompactQuarterScopeLabel(raw) || dom || 'Applied scope';
    }

    function pbjV2RefreshScopeLabel(filterInfoFallback) {
        var label = pbjV2ScopeLabelFromDom();
        if (!label || label.indexOf('All ') === 0) {
            label = pbjV2ScopeLabelFromFilterInfo(filterInfoFallback) || label;
        }
        if (typeof global.pbjUpdateSummaryPodTitles === 'function') {
            global.pbjUpdateSummaryPodTitles(label);
        }
        pbjV2UpdateScopeLabels(label || 'Applied scope');
    }

    function pbjV2OpenPbj320SummaryFromControlCenter() {
        pbjV2CloseControlCenter();
        if (typeof global.openPbj320SnapshotModal === 'function') {
            global.openPbj320SnapshotModal();
        }
    }

    function pbjV2MountNavControls() {
        var ctrl = document.getElementById('pbjFloatingControls');
        var dock = document.getElementById('pbjControlsDock');
        if (dock && dock.parentElement !== document.body) {
            document.body.appendChild(dock);
        }
        if (ctrl && dock && ctrl.parentElement !== dock) {
            dock.appendChild(ctrl);
            dock.removeAttribute('aria-hidden');
            ctrl.setAttribute('data-pbj-dock-mounted', '1');
        }
    }

    function pbjV2ControlsDockPositionLooksInvalid(left, top, dock) {
        if (!isFinite(left) || !isFinite(top)) {
            return true;
        }
        if (left < -8 || top < pbjV2ControlsDockMinTop() - 8) {
            return true;
        }
        var target = dock || document.getElementById('pbjControlsDock');
        if (!target) {
            return false;
        }
        target.classList.add('pbj-controls-dock--custom');
        target.style.left = Math.round(left) + 'px';
        target.style.top = Math.round(top) + 'px';
        target.style.right = 'auto';
        target.style.bottom = 'auto';
        var u = pbjV2ControlsDockUnionRect(target);
        var pad = 8;
        if (u.left < pad - 1 || u.right > window.innerWidth - pad + 1) {
            return true;
        }
        if (u.top < pbjV2ControlsDockMinTop() - 1 || u.bottom > window.innerHeight - pad + 1) {
            return true;
        }
        return false;
    }

    function pbjV2ControlsDockMostlyVisible(dock) {
        if (!dock) {
            return false;
        }
        var u = pbjV2ControlsDockUnionRect(dock);
        if (!u.width && !u.height) {
            return false;
        }
        var minT = pbjV2ControlsDockMinTop();
        var pad = 8;
        var visW = Math.min(u.right, window.innerWidth - pad) - Math.max(u.left, pad);
        var visH = Math.min(u.bottom, window.innerHeight - pad) - Math.max(u.top, minT);
        return visW >= 40 && visH >= 28;
    }

    function pbjV2FinalizeControlsDockReady(dock) {
        if (!dock) {
            return;
        }
        if (dock.classList.contains('pbj-controls-dock--custom') && !pbjV2ControlsDockMostlyVisible(dock)) {
            pbjV2ResetControlsDockPosition(dock);
            pbjV2DefaultControlsDockPosition(dock);
        }
        dock.classList.add('pbj-controls-dock--ready');
    }

    function pbjV2ControlsDockMinTop() {
        var gap = 10;
        var nav = document.getElementById('guidedNavWrap');
        if (nav) {
            try {
                var navRect = nav.getBoundingClientRect();
                if (navRect.height > 0 && navRect.bottom > 0 && navRect.bottom < window.innerHeight * 0.5) {
                    return Math.ceil(navRect.bottom + gap);
                }
            } catch (eNav) { /* ignore */ }
        }
        var banner = 0;
        try {
            banner = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--pbj-premium-banner-h')) || 0;
        } catch (eBanner) { /* ignore */ }
        /* Banner + typical guided-nav row when nav not measured yet */
        return Math.max(Math.ceil(banner + 52 + gap), 12);
    }

    function pbjV2ControlsDockPanelIsOpen() {
        var panel = document.getElementById('pbjFloatingControlsPanel');
        if (!panel || panel.hidden) {
            return false;
        }
        var wrap = document.getElementById('pbjFloatingControls');
        if (wrap && wrap.classList.contains('is-open')) {
            return true;
        }
        return !panel.hidden;
    }

    function pbjV2ControlsDockRectFromElements(elems, dock) {
        var minL = Infinity;
        var minT = Infinity;
        var maxR = -Infinity;
        var maxB = -Infinity;
        (elems || []).forEach(function (el) {
            if (!el) {
                return;
            }
            var r = el.getBoundingClientRect();
            if (!r.width && !r.height) {
                return;
            }
            minL = Math.min(minL, r.left);
            minT = Math.min(minT, r.top);
            maxR = Math.max(maxR, r.right);
            maxB = Math.max(maxB, r.bottom);
        });
        if (!isFinite(minL) && dock) {
            var dr = dock.getBoundingClientRect();
            return {
                left: dr.left,
                top: dr.top,
                right: dr.right,
                bottom: dr.bottom,
                width: Math.max(dr.width || 0, 48),
                height: Math.max(dr.height || 0, 36)
            };
        }
        return {
            left: minL,
            top: minT,
            right: maxR,
            bottom: maxB,
            width: maxR - minL,
            height: maxB - minT
        };
    }

    function pbjV2ControlsDockUnionRect(dock) {
        if (!dock) {
            return { left: 0, top: 0, right: 0, bottom: 0, width: 120, height: 40 };
        }
        var wrap = document.getElementById('pbjFloatingControls');
        var panel = document.getElementById('pbjFloatingControlsPanel');
        var fab = document.getElementById('pbjFloatingControlsFab');
        var elems = [dock, wrap, panel, fab];
        var filtered = [];
        elems.forEach(function (el) {
            if (!el) {
                return;
            }
            if (el === panel && panel.hidden) {
                return;
            }
            if (el === fab) {
                try {
                    if (getComputedStyle(fab).display === 'none') {
                        return;
                    }
                } catch (eFab) { /* ignore */ }
            }
            filtered.push(el);
        });
        return pbjV2ControlsDockRectFromElements(filtered, dock);
    }

    /** Drag/clamp footprint: FAB + dock only while panel is open (panel height must not shrink vertical range). */
    function pbjV2ControlsDockDragRect(dock) {
        if (!dock) {
            return { left: 0, top: 0, right: 0, bottom: 0, width: 120, height: 40 };
        }
        var panelOpen = pbjV2ControlsDockPanelIsOpen();
        var fab = document.getElementById('pbjFloatingControlsFab');
        var elems = [dock];
        if (fab) {
            try {
                if (getComputedStyle(fab).display !== 'none') {
                    elems.push(fab);
                }
            } catch (eFab) {
                elems.push(fab);
            }
        }
        if (!panelOpen) {
            var wrap = document.getElementById('pbjFloatingControls');
            var panel = document.getElementById('pbjFloatingControlsPanel');
            if (wrap) {
                elems.push(wrap);
            }
            if (panel && !panel.hidden) {
                elems.push(panel);
            }
        }
        return pbjV2ControlsDockRectFromElements(elems, dock);
    }

    function pbjV2ControlsDockSize(dock) {
        var u = pbjV2ControlsDockDragRect(dock);
        return {
            w: Math.max(u.width || 48, 48),
            h: Math.max(u.height || 36, 36)
        };
    }

    function pbjV2ControlsDockBounds(dock) {
        var size = pbjV2ControlsDockSize(dock);
        var minT = pbjV2ControlsDockMinTop();
        var pad = 8;
        return {
            minLeft: pad,
            maxLeft: Math.max(pad, window.innerWidth - size.w - pad),
            minTop: minT,
            maxTop: Math.max(minT, window.innerHeight - size.h - pad)
        };
    }

    function pbjV2ControlsDockViewportRect(dock) {
        if (!dock) {
            return { left: 0, top: 0, width: 120, height: 40 };
        }
        var u = pbjV2ControlsDockDragRect(dock);
        return {
            left: u.left,
            top: u.top,
            width: u.width,
            height: u.height
        };
    }

    function pbjV2FreezeControlsDockPosition(dock) {
        if (!dock) {
            return { left: 0, top: 0, width: 120, height: 40 };
        }
        var rect = pbjV2ControlsDockViewportRect(dock);
        dock.classList.add('pbj-controls-dock--custom');
        dock.style.left = Math.round(rect.left) + 'px';
        dock.style.top = Math.round(rect.top) + 'px';
        dock.style.right = 'auto';
        dock.style.bottom = 'auto';
        return pbjV2ControlsDockViewportRect(dock);
    }

    function pbjV2ClampControlsDockPosition(left, top, dock) {
        if (!dock) {
            return { left: left, top: top };
        }
        var pad = 8;
        var minT = pbjV2ControlsDockMinTop();
        dock.classList.add('pbj-controls-dock--custom');
        dock.style.left = Math.round(left) + 'px';
        dock.style.top = Math.round(top) + 'px';
        dock.style.right = 'auto';
        dock.style.bottom = 'auto';
        var u = pbjV2ControlsDockDragRect(dock);
        var shiftL = 0;
        var shiftT = 0;
        if (u.left < pad) {
            shiftL += pad - u.left;
        }
        if (u.right > window.innerWidth - pad) {
            shiftL -= u.right - (window.innerWidth - pad);
        }
        if (u.top < minT) {
            shiftT += minT - u.top;
        }
        if (u.bottom > window.innerHeight - pad) {
            shiftT -= u.bottom - (window.innerHeight - pad);
        }
        return {
            left: Math.round(left + shiftL),
            top: Math.round(top + shiftT)
        };
    }

    function pbjV2DefaultControlsDockPosition(dock) {
        if (!dock) {
            return;
        }
        pbjV2ResetControlsDockPosition(dock);
    }

    function pbjV2ControlsDockSessionStore() {
        try {
            return window.sessionStorage;
        } catch (eStore) {
            return null;
        }
    }

    function pbjV2EnsureControlsDockPosition(dock) {
        if (!dock) {
            return;
        }
        var wrap = document.getElementById('pbjFloatingControls');
        var panel = document.getElementById('pbjFloatingControlsPanel');
        var panelOpen = !!(wrap && wrap.classList.contains('is-open') && panel && !panel.hidden);
        if (!panelOpen) {
            pbjV2DefaultControlsDockPosition(dock);
        }
    }

    function pbjV2ApplyControlsDockPosition(dock, left, top, skipClamp) {
        if (!dock) {
            return;
        }
        var nextLeft = left;
        var nextTop = top;
        if (!skipClamp) {
            var clamped = pbjV2ClampControlsDockPosition(left, top, dock);
            nextLeft = clamped.left;
            nextTop = clamped.top;
        }
        dock.classList.add('pbj-controls-dock--custom');
        dock.style.left = Math.round(nextLeft) + 'px';
        dock.style.top = Math.round(nextTop) + 'px';
        dock.style.right = 'auto';
        dock.style.bottom = 'auto';
    }

    function pbjV2ResetControlsDockPosition(dock) {
        if (!dock) {
            return;
        }
        dock.classList.remove('pbj-controls-dock--custom');
        dock.style.left = '';
        dock.style.top = '';
        dock.style.right = '';
        dock.style.bottom = '';
        var store = pbjV2ControlsDockSessionStore();
        try {
            if (store) {
                store.removeItem('pbj_v2_controls_dock_pos');
            }
            localStorage.removeItem('pbj_v2_controls_dock_pos');
        } catch (eReset) { /* ignore */ }
    }

    function pbjV2RestoreControlsDockPosition(dock) {
        if (!dock) {
            return;
        }
        var store = pbjV2ControlsDockSessionStore();
        if (!store) {
            return;
        }
        try {
            var raw = store.getItem('pbj_v2_controls_dock_pos');
            if (!raw) {
                return;
            }
            var pos = JSON.parse(raw);
            if (!pos || typeof pos.left !== 'number' || typeof pos.top !== 'number') {
                store.removeItem('pbj_v2_controls_dock_pos');
                return;
            }
            var clamped = pbjV2ClampControlsDockPosition(pos.left, pos.top, dock);
            if (
                !isFinite(clamped.left) ||
                !isFinite(clamped.top) ||
                clamped.left < 0 ||
                clamped.top < 0 ||
                pbjV2ControlsDockPositionLooksInvalid(clamped.left, clamped.top, dock)
            ) {
                pbjV2ResetControlsDockPosition(dock);
                return;
            }
            pbjV2ApplyControlsDockPosition(dock, clamped.left, clamped.top, true);
            if (!pbjV2ControlsDockMostlyVisible(dock)) {
                pbjV2ResetControlsDockPosition(dock);
            }
        } catch (eLoad) {
            pbjV2ResetControlsDockPosition(dock);
        }
    }

    function pbjV2InitControlsDockDrag() {
        var dock = document.getElementById('pbjControlsDock');
        if (!dock || dock.dataset.pbjDockDragBound === '1') {
            return;
        }
        dock.dataset.pbjDockDragBound = '1';
        try {
            localStorage.removeItem('pbj_v2_controls_dock_pos');
            if (sessionStorage.getItem('pbj_v2_controls_dock_pos_v') !== '9') {
                sessionStorage.removeItem('pbj_v2_controls_dock_pos');
                sessionStorage.setItem('pbj_v2_controls_dock_pos_v', '9');
            }
        } catch (eDockVer) { /* ignore */ }
        pbjV2DefaultControlsDockPosition(dock);
        pbjV2FinalizeControlsDockReady(dock);

        var dragState = null;
        var pendingDrag = null;
        var suppressClickUntil = 0;
        var DRAG_THRESHOLD_PX = 5;

        function pbjV2ControlsDockClickSuppressed() {
            return Date.now() < suppressClickUntil;
        }
        global.pbjV2ControlsDockClickSuppressed = pbjV2ControlsDockClickSuppressed;

        function clampDockPosition(left, top) {
            return pbjV2ClampControlsDockPosition(left, top, dock);
        }

        function saveDockPosition() {
            var wrap = document.getElementById('pbjFloatingControls');
            var panel = document.getElementById('pbjFloatingControlsPanel');
            if (!wrap || !wrap.classList.contains('is-open') || !panel || panel.hidden) {
                return;
            }
            if (!pbjV2ControlsDockMostlyVisible(dock)) {
                return;
            }
            var rect = pbjV2ControlsDockViewportRect(dock);
            var clamped = pbjV2ClampControlsDockPosition(rect.left, rect.top, dock);
            if (pbjV2ControlsDockPositionLooksInvalid(clamped.left, clamped.top, dock)) {
                return;
            }
            var store = pbjV2ControlsDockSessionStore();
            try {
                if (store) {
                    store.setItem('pbj_v2_controls_dock_pos', JSON.stringify({
                        left: Math.round(clamped.left),
                        top: Math.round(clamped.top)
                    }));
                }
            } catch (eSave) { /* ignore */ }
        }

        function moveDrag(clientX, clientY) {
            if (!dragState) {
                return;
            }
            var next = clampDockPosition(clientX - dragState.offsetX, clientY - dragState.offsetY);
            if (!dragState.moved) {
                dragState.moved = true;
            }
            pbjV2ApplyControlsDockPosition(dock, next.left, next.top);
        }

        function endDrag(didMove) {
            if (!dragState && !pendingDrag) {
                return;
            }
            if (dragState && (dragState.moved || didMove)) {
                suppressClickUntil = Date.now() + 280;
                saveDockPosition();
            }
            dock.classList.remove('is-dragging');
            dragState = null;
            pendingDrag = null;
            pbjV2AdjustControlsDockPanelPlacement();
        }

        function resolveDockDragHandle(ev, panelOpen) {
            if (panelOpen) {
                if (ev.target.closest('.btn-close, button, a, input, select, textarea, label')) {
                    return null;
                }
                return ev.target.closest('[data-pbj-dock-drag-panel]');
            }
            return ev.target.closest(
                '.pbj-floating-controls-fab-cluster [data-pbj-dock-drag], ' +
                '.pbj-floating-controls-fab-cluster .pbj-floating-controls-drag-zone'
            );
        }

        dock.addEventListener('pointerdown', function (ev) {
            /* Closed: FAB ⋮⋮ only. Open: panel head left of close (not the X). */
            var wrap = document.getElementById('pbjFloatingControls');
            var panel = document.getElementById('pbjFloatingControlsPanel');
            var panelOpen = !!(wrap && wrap.classList.contains('is-open') && panel && !panel.hidden);
            var handle = resolveDockDragHandle(ev, panelOpen);
            if (!handle) {
                return;
            }
            if (ev.button !== 0 && ev.pointerType === 'mouse') {
                return;
            }
            pendingDrag = {
                pointerId: ev.pointerId,
                startX: ev.clientX,
                startY: ev.clientY,
                handle: handle
            };
        });

        dock.addEventListener('pointermove', function (ev) {
            if (pendingDrag && pendingDrag.pointerId === ev.pointerId && !dragState) {
                var dx = ev.clientX - pendingDrag.startX;
                var dy = ev.clientY - pendingDrag.startY;
                if (Math.hypot(dx, dy) < DRAG_THRESHOLD_PX) {
                    return;
                }
                pbjV2FreezeControlsDockPosition(dock);
                var dockR = dock.getBoundingClientRect();
                dragState = {
                    pointerId: ev.pointerId,
                    offsetX: pendingDrag.startX - dockR.left,
                    offsetY: pendingDrag.startY - dockR.top,
                    moved: true
                };
                dock.classList.add('is-dragging');
                try {
                    dock.setPointerCapture(ev.pointerId);
                } catch (eCapStart) { /* ignore */ }
                moveDrag(ev.clientX, ev.clientY);
                return;
            }
            if (!dragState || dragState.pointerId !== ev.pointerId) {
                return;
            }
            moveDrag(ev.clientX, ev.clientY);
        });

        dock.addEventListener('pointerup', function (ev) {
            if (pendingDrag && pendingDrag.pointerId === ev.pointerId && !dragState) {
                pendingDrag = null;
                return;
            }
            if (!dragState || dragState.pointerId !== ev.pointerId) {
                return;
            }
            try {
                dock.releasePointerCapture(ev.pointerId);
            } catch (eRel) { /* ignore */ }
            endDrag(true);
        });

        dock.addEventListener('pointercancel', function () {
            endDrag(false);
        });
        dock.addEventListener('lostpointercapture', function () {
            endDrag(false);
        });

        dock.addEventListener('dblclick', function (ev) {
            var wrap = document.getElementById('pbjFloatingControls');
            var panel = document.getElementById('pbjFloatingControlsPanel');
            var panelOpen = !!(wrap && wrap.classList.contains('is-open') && panel && !panel.hidden);
            var onHandle = panelOpen
                ? ev.target.closest('[data-pbj-dock-drag-panel]')
                : ev.target.closest(
                    '.pbj-floating-controls-fab-cluster [data-pbj-dock-drag], ' +
                    '.pbj-floating-controls-fab-cluster .pbj-floating-controls-drag-zone'
                );
            if (!onHandle) {
                return;
            }
            ev.preventDefault();
            pbjV2ResetControlsDockPosition(dock);
            pbjV2DefaultControlsDockPosition(dock);
        });

        dock.addEventListener('click', function (ev) {
            if (!pbjV2ControlsDockClickSuppressed()) {
                return;
            }
            if (
                ev.target.closest('#pbjFloatingControlsFab') ||
                ev.target.closest('.pbj-floating-controls-drag-zone')
            ) {
                ev.preventDefault();
                ev.stopPropagation();
            }
        }, true);

        global.addEventListener('resize', function () {
            if (dock.classList.contains('pbj-controls-dock--custom')) {
                var rect = pbjV2ControlsDockViewportRect(dock);
                var clamped = pbjV2ClampControlsDockPosition(rect.left, rect.top, dock);
                pbjV2ApplyControlsDockPosition(dock, clamped.left, clamped.top, true);
                if (!pbjV2ControlsDockMostlyVisible(dock)) {
                    pbjV2ResetControlsDockPosition(dock);
                    pbjV2DefaultControlsDockPosition(dock);
                }
            } else {
                pbjV2DefaultControlsDockPosition(dock);
            }
            pbjV2AdjustControlsDockPanelPlacement();
        });
        global.addEventListener('scroll', function () {
            if (pbjV2ControlsDockPanelIsOpen()) {
                pbjV2AdjustControlsDockPanelPlacement();
            }
        }, { passive: true });
        var backdrop = document.getElementById('pbjControlsDockBackdrop');
        if (backdrop) {
            backdrop.addEventListener('click', function () {
                pbjV2CloseControlCenter();
            });
        }
    }

    global.pbjV2ResetControlsDockPosition = pbjV2ResetControlsDockPosition;
    global.pbjCompactQuarterScopeLabel = pbjCompactQuarterScopeLabel;
    global.pbjV2OpenPbj320SummaryFromControlCenter = pbjV2OpenPbj320SummaryFromControlCenter;

    function pbjV2CensusTableGrain() {
        var r = document.querySelector('input[name="censusView"]:checked');
        return r && r.value ? r.value : 'month';
    }

    function pbjV2ParseCensusChartPeriodDate(period) {
        var s = String(period == null ? '' : period).trim();
        var m = s.match(/^(\d{2})-(\d{2})-(\d{4})$/);
        if (m) {
            return new Date(parseInt(m[3], 10), parseInt(m[1], 10) - 1, parseInt(m[2], 10));
        }
        return null;
    }

    function pbjV2PeriodToCyQuarterKey(period, grain) {
        var s = String(period == null ? '' : period).trim();
        if (!s) {
            return '';
        }
        if (grain === 'daily') {
            var d = pbjV2ParseCensusChartPeriodDate(s);
            if (d && !isNaN(d.getTime())) {
                return d.getFullYear() + 'Q' + Math.ceil((d.getMonth() + 1) / 3);
            }
            return '';
        }
        if (grain === 'month') {
            var mo = s.match(/^([A-Za-z]{3})\s+(\d{4})$/);
            if (mo) {
                var months = {
                    Jan: 1, Feb: 2, Mar: 3, Apr: 4, May: 5, Jun: 6,
                    Jul: 7, Aug: 8, Sep: 9, Oct: 10, Nov: 11, Dec: 12
                };
                var mn = months[mo[1]];
                if (mn) {
                    return mo[2] + 'Q' + Math.ceil(mn / 3);
                }
            }
            return '';
        }
        var qm = s.match(/^Q([1-4])\s+(\d{4})$/i);
        if (qm) {
            return qm[2] + 'Q' + qm[1];
        }
        qm = s.match(/^(?:CY)?(\d{4})Q([1-4])$/i);
        if (qm) {
            return qm[1] + 'Q' + qm[2];
        }
        return '';
    }

    function pbjV2FormatCensusTablePeriod(period, grain) {
        var s = String(period == null ? '' : period).trim();
        if (!s) {
            return '—';
        }
        if (grain === 'daily') {
            var d = pbjV2ParseCensusChartPeriodDate(s);
            if (d && !isNaN(d.getTime())) {
                var dow = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.getDay()];
                var mm = String(d.getMonth() + 1).padStart(2, '0');
                var dd = String(d.getDate()).padStart(2, '0');
                var yy = String(d.getFullYear()).slice(2);
                return dow + ' ' + mm + '-' + dd + '-' + yy;
            }
        }
        return s;
    }

    function pbjV2FormatCensusTableNumber(val, decimals) {
        var n = Number(val);
        if (!isFinite(n)) {
            return '—';
        }
        return n.toFixed(decimals == null ? 1 : decimals);
    }

    function pbjV2FormatCensusTableBeds(val) {
        var n = Number(val);
        if (!isFinite(n) || n <= 0) {
            return '—';
        }
        return String(Math.round(n));
    }

    function pbjV2FormatCensusTableOccupancy(val) {
        var n = Number(val);
        if (!isFinite(n)) {
            return '—';
        }
        return n.toFixed(1) + '%';
    }

    function pbjV2BuildOccupancyLookup() {
        var map = {};
        var occ = global.providerChartData && global.providerChartData.occupancy;
        if (!occ || !Array.isArray(occ.quarters)) {
            return map;
        }
        (occ.quarters || []).forEach(function (q, i) {
            var key = pbjV2PeriodToCyQuarterKey(q, 'quarter');
            if (!key) {
                return;
            }
            map[key] = {
                occupancy_pct: occ.occupancy_pct ? occ.occupancy_pct[i] : null,
                beds: occ.certified_beds ? occ.certified_beds[i] : null
            };
        });
        return map;
    }

    function pbjV2AnnualOccupancyFromQuarters(period, occMap) {
        var s = String(period == null ? '' : period).trim();
        var ym = s.match(/^(\d{4})$/);
        if (!ym) {
            return { beds: null, occupancy_pct: null };
        }
        var yr = parseInt(ym[1], 10);
        if (!yr || isNaN(yr)) {
            return { beds: null, occupancy_pct: null };
        }
        var bedVals = [];
        var occVals = [];
        var q;
        for (q = 1; q <= 4; q += 1) {
            var key = yr + 'Q' + q;
            var hit = occMap && occMap[key] ? occMap[key] : null;
            if (!hit) {
                continue;
            }
            if (hit.beds != null && isFinite(Number(hit.beds)) && Number(hit.beds) > 0) {
                bedVals.push(Number(hit.beds));
            }
            if (hit.occupancy_pct != null && isFinite(Number(hit.occupancy_pct))) {
                occVals.push(Number(hit.occupancy_pct));
            }
        }
        var beds = bedVals.length
            ? bedVals.reduce(function (a, b) {
                  return a + b;
              }, 0) / bedVals.length
            : null;
        var occupancy_pct = occVals.length
            ? occVals.reduce(function (a, b) {
                  return a + b;
              }, 0) / occVals.length
            : null;
        return { beds: beds, occupancy_pct: occupancy_pct };
    }

    function pbjV2LookupOccupancyForPeriod(period, grain, occMap) {
        if (grain === 'year') {
            var annual = pbjV2AnnualOccupancyFromQuarters(period, occMap);
            if (annual.beds != null || annual.occupancy_pct != null) {
                return annual;
            }
        }
        var key = pbjV2PeriodToCyQuarterKey(period, grain);
        var hit = key && occMap[key] ? occMap[key] : null;
        return {
            beds: hit && hit.beds != null ? hit.beds : null,
            occupancy_pct: hit && hit.occupancy_pct != null ? hit.occupancy_pct : null
        };
    }

    function pbjV2RefreshCensusOccupancyTable(charts) {
        var tbody = document.getElementById('censusOccupancyTableBody');
        if (!tbody) {
            return;
        }
        var grain = pbjV2CensusTableGrain();
        var capEl = document.getElementById('pbjCensusTableCaption');
        if (capEl) {
            var grainLabels = { daily: 'Daily', month: 'Monthly', quarter: 'Quarterly', year: 'Annual' };
            capEl.textContent = (grainLabels[grain] || 'Period') + ' census and occupancy';
        }
        var occMap = pbjV2BuildOccupancyLookup();
        var rows = [];
        if (charts && charts.census_trend && charts.census_trend.data && charts.census_trend.data.length) {
            var trace = charts.census_trend.data.find(function (t) {
                return t && t.x && t.y && !(t.name && String(t.name).indexOf('Holiday') >= 0);
            }) || charts.census_trend.data[0];
            (trace.x || []).forEach(function (x, i) {
                var census = trace.y ? trace.y[i] : null;
                if (census == null || !(Number(census) > 0)) {
                    return;
                }
                var occHit = pbjV2LookupOccupancyForPeriod(x, grain, occMap);
                var beds = occHit.beds;
                var occPct = occHit.occupancy_pct;
                if ((occPct == null || !isFinite(Number(occPct))) && beds != null && isFinite(Number(beds)) && Number(beds) > 0) {
                    occPct = Number(census) / Number(beds) * 100;
                }
                rows.push({
                    period: x,
                    census: census,
                    beds: beds,
                    occupancy_pct: occPct
                });
            });
        }
        if (!rows.length) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-muted small">No rows for this view yet.</td></tr>';
            return;
        }
        rows.reverse();
        if (rows.length > 120) {
            rows = rows.slice(0, 120);
        }
        var esc = typeof pbjV2EinHcEscHtml === 'function' ? pbjV2EinHcEscHtml : function (t) { return String(t == null ? '' : t); };
        tbody.innerHTML = rows.map(function (r) {
            var periodLabel = pbjV2FormatCensusTablePeriod(r.period, grain);
            var periodKey = pbjV2RollupPeriodKeyFromRaw(r.period);
            var periodCell;
            if (grain === 'daily' && pbjV2RollupPeriodIsActionable(r.period)) {
                periodCell = '<td class="pbj-census-col-period" data-sort="' + esc(periodKey) + '">' +
                    pbjV2RollupPeriodBtnHtml(r.period, periodLabel, esc) + '</td>';
            } else if (pbjV2RollupPeriodIsActionable(r.period)) {
                periodCell = '<td class="pbj-census-col-period" data-sort="' + esc(periodKey) + '">' +
                    pbjV2RollupPeriodBtnHtml(r.period, periodLabel, esc) + '</td>';
            } else {
                periodCell = '<td class="pbj-census-col-period" data-sort="' + esc(periodLabel) + '">' + esc(periodLabel) + '</td>';
            }
            return '<tr>' +
                periodCell +
                '<td class="pbj-census-col-num" data-sort="' + esc(String(r.census)) + '">' + pbjV2FormatCensusTableNumber(r.census, 1) + '</td>' +
                '<td class="pbj-census-col-num" data-sort="' + esc(r.beds != null ? String(r.beds) : '') + '">' + pbjV2FormatCensusTableBeds(r.beds) + '</td>' +
                '<td class="pbj-census-col-num" data-sort="' + esc(r.occupancy_pct != null ? String(r.occupancy_pct) : '') + '">' + pbjV2FormatCensusTableOccupancy(r.occupancy_pct) + '</td>' +
                '</tr>';
        }).join('');
    }

    function pbjV2FormatRollupCell(val, digits) {
        var vn = typeof val === 'number' ? val : parseFloat(String(val));
        if (!Number.isFinite(vn)) {
            return '—';
        }
        return vn.toFixed(digits != null ? digits : 2);
    }

    function pbjV2RollupSortTh(text, colIndex, extraClass) {
        var cls = 'sortable pbj-rollup-sort-th' + (extraClass ? ' ' + extraClass : '');
        return '<th scope="col" class="' + cls + '" data-pbj-rollup-col="' + colIndex + '">' + text + '</th>';
    }

    function pbjV2RollupPeriodIsActionable(periodRaw) {
        var s = String(periodRaw == null ? '' : periodRaw).trim();
        if (!s) {
            return false;
        }
        if (/^\d{4}-\d{2}-\d{2}/.test(s)) {
            return true;
        }
        if (pbjV2PeriodToCyQuarterKey(s, 'quarter')) {
            return true;
        }
        return /^Q[1-4]\s+\d{4}$/i.test(s);
    }

    function pbjV2RollupPeriodKeyFromRaw(periodRaw) {
        var s = String(periodRaw == null ? '' : periodRaw).trim();
        if (/^\d{4}-\d{2}-\d{2}/.test(s)) {
            return s.slice(0, 10);
        }
        var qk = pbjV2PeriodToCyQuarterKey(s, 'quarter');
        if (qk) {
            return qk;
        }
        return s;
    }

    function pbjV2RollupPeriodBtnHtml(periodRaw, periodLabel, escFn) {
        var esc = escFn || function (t) { return String(t == null ? '' : t); };
        var key = pbjV2RollupPeriodKeyFromRaw(periodRaw);
        var label = periodLabel != null ? String(periodLabel) : String(periodRaw);
        var title = /^\d{4}-\d{2}-\d{2}$/.test(key)
            ? 'Open day staffing report'
            : 'Open PBJ320 quarter snapshot';
        return '<button type="button" class="btn btn-link btn-sm p-0 text-start pbj-rollup-period-btn" ' +
            'data-pbj-rollup-period-key="' + esc(key) + '" ' +
            'data-pbj-rollup-period-raw="' + esc(String(periodRaw)) + '" ' +
            'title="' + esc(title) + '">' + esc(label) + '</button>';
    }

    function pbjV2HandleRollupPeriodClick(periodKey, periodRaw) {
        var key = String(periodKey || periodRaw || '').trim();
        if (/^\d{4}-\d{2}-\d{2}$/.test(key)) {
            if (typeof global.openSingleDayReport === 'function') {
                global.openSingleDayReport(key);
            } else if (typeof global.openPbj320SnapshotModalForDay === 'function') {
                global.openPbj320SnapshotModalForDay(key);
            }
            return;
        }
        var qk = /^(\d{4})Q([1-4])$/i.test(key) ? key : pbjV2PeriodToCyQuarterKey(key, 'quarter');
        if (qk && typeof global.openPbj320SnapshotModalForQuarter === 'function') {
            global.openPbj320SnapshotModalForQuarter(qk);
        }
    }

    function pbjV2ExportRollupTableCsv(tableId, slug) {
        var table = document.getElementById(tableId);
        if (!table || typeof global.pbj320TriggerCsvDownload !== 'function') {
            return;
        }
        var rows = [];
        var theadRows = table.querySelectorAll('thead tr');
        var headerRow = theadRows.length ? theadRows[theadRows.length - 1] : null;
        if (headerRow) {
            var headers = Array.from(headerRow.querySelectorAll('th')).map(function (th) {
                return '"' + String(th.textContent).replace(/\s+/g, ' ').trim().replace(/"/g, '""') + '"';
            });
            if (headers.length) {
                rows.push(headers.join(','));
            }
        }
        table.querySelectorAll('tbody tr').forEach(function (tr) {
            if (tr.querySelector('td.text-muted')) {
                return;
            }
            var cells = Array.from(tr.querySelectorAll('td')).map(function (td) {
                var sort = td.getAttribute('data-sort');
                var txt = sort != null ? sort : td.textContent.replace(/\s+/g, ' ').trim();
                return '"' + String(txt).replace(/"/g, '""') + '"';
            });
            if (cells.length) {
                rows.push(cells.join(','));
            }
        });
        if (!rows.length) {
            return;
        }
        var fnameFn = typeof global.pbj320ExportFilename === 'function'
            ? global.pbj320ExportFilename
            : function (s) { return String(s || 'rollup_table') + '.csv'; };
        var preambleFn = typeof global.pbj320CsvPreamble === 'function'
            ? global.pbj320CsvPreamble
            : function () { return ''; };
        var detail = slug ? ('Rollup table: ' + slug.replace(/_/g, ' ')) : 'Rollup table export';
        global.pbj320TriggerCsvDownload(fnameFn(slug || 'rollup_table'), preambleFn(detail) + rows.join('\r\n'));
    }

    function pbjV2SortRollupTable(tableId, colIndex) {
        var table = document.getElementById(tableId);
        if (!table) {
            return;
        }
        var tbody = table.querySelector('tbody');
        var th = table.querySelector('th[data-pbj-rollup-col="' + colIndex + '"]');
        if (!tbody || !th) {
            return;
        }
        var asc = !th.classList.contains('sort-asc');
        table.querySelectorAll('th.pbj-rollup-sort-th').forEach(function (h) {
            h.classList.remove('sort-asc', 'sort-desc');
        });
        th.classList.add(asc ? 'sort-asc' : 'sort-desc');
        var rows = Array.from(tbody.querySelectorAll('tr')).filter(function (r) {
            return !r.querySelector('td.text-muted');
        });
        rows.sort(function (a, b) {
            var ac = a.cells[colIndex];
            var bc = b.cells[colIndex];
            if (!ac || !bc) {
                return 0;
            }
            var aVal = ac.getAttribute('data-sort') != null ? ac.getAttribute('data-sort') : ac.textContent.trim();
            var bVal = bc.getAttribute('data-sort') != null ? bc.getAttribute('data-sort') : bc.textContent.trim();
            var aq = pbjV2PeriodToCyQuarterKey(aVal, 'quarter') || (String(aVal).match(/^(\d{4})Q([1-4])$/i) ? aVal : '');
            var bq = pbjV2PeriodToCyQuarterKey(bVal, 'quarter') || (String(bVal).match(/^(\d{4})Q([1-4])$/i) ? bVal : '');
            if (aq && bq) {
                var am = String(aq).match(/^(\d{4})Q([1-4])$/i);
                var bm = String(bq).match(/^(\d{4})Q([1-4])$/i);
                if (am && bm) {
                    var cmp = am[1] !== bm[1] ? Number(am[1]) - Number(bm[1]) : Number(am[2]) - Number(bm[2]);
                    return asc ? cmp : -cmp;
                }
            }
            if (/^\d{4}-\d{2}-\d{2}$/.test(aVal) && /^\d{4}-\d{2}-\d{2}$/.test(bVal)) {
                return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
            }
            var an = parseFloat(String(aVal).replace(/%/g, ''));
            var bn = parseFloat(String(bVal).replace(/%/g, ''));
            if (Number.isFinite(an) && Number.isFinite(bn)) {
                return asc ? an - bn : bn - an;
            }
            return asc ? String(aVal).localeCompare(String(bVal)) : String(bVal).localeCompare(String(aVal));
        });
        rows.forEach(function (r) {
            tbody.appendChild(r);
        });
    }

    function pbjV2WireRollupTableInteractions(tableId) {
        var table = document.getElementById(tableId);
        if (!table || table.dataset.pbjRollupChromeWired === '1') {
            return;
        }
        table.dataset.pbjRollupChromeWired = '1';
        table.addEventListener('click', function (ev) {
            var sortTh = ev.target.closest('th.pbj-rollup-sort-th');
            if (sortTh && sortTh.hasAttribute('data-pbj-rollup-col')) {
                ev.preventDefault();
                pbjV2SortRollupTable(tableId, parseInt(sortTh.getAttribute('data-pbj-rollup-col'), 10));
                return;
            }
            var periodBtn = ev.target.closest('.pbj-rollup-period-btn');
            if (periodBtn) {
                ev.preventDefault();
                pbjV2HandleRollupPeriodClick(
                    periodBtn.getAttribute('data-pbj-rollup-period-key'),
                    periodBtn.getAttribute('data-pbj-rollup-period-raw')
                );
                return;
            }
            var ratingBtn = ev.target.closest('.pbj-rollup-rating-btn');
            if (ratingBtn) {
                ev.preventDefault();
                var q = ratingBtn.getAttribute('data-pbj-rating-quarter');
                if (q && typeof global.openPbj320SnapshotModalForQuarter === 'function') {
                    global.openPbj320SnapshotModalForQuarter(q);
                }
                return;
            }
        });
    }

    function pbjV2InitRollupExportButtons() {
        if (document.documentElement.dataset.pbjRollupExportBound === '1') {
            return;
        }
        document.documentElement.dataset.pbjRollupExportBound = '1';
        document.addEventListener('click', function (ev) {
            var exportBtn = ev.target.closest('[data-pbj-rollup-export]');
            if (!exportBtn) {
                return;
            }
            ev.preventDefault();
            pbjV2ExportRollupTableCsv(
                exportBtn.getAttribute('data-pbj-rollup-export'),
                exportBtn.getAttribute('data-pbj-rollup-slug') || exportBtn.getAttribute('data-pbj-rollup-export')
            );
        });
    }

    function pbjV2InitRollupTableChrome() {
        [
            'compositionRollupTable',
            'einHeadcountRollupTable',
            'dowRollupTable',
            'contractRollupTable',
            'ratingsOverTimeRollupTable',
            'censusOccupancyTable'
        ].forEach(function (id) {
            pbjV2WireRollupTableInteractions(id);
        });
        pbjV2InitRollupExportButtons();
    }

    function pbjV2RefreshPlotlyRollupTable(chartId, tableId, tbodyId, options) {
        options = options || {};
        var gd = document.getElementById(chartId);
        var tbody = document.getElementById(tbodyId);
        var theadRow = document.querySelector('#' + tableId + 'Head tr');
        if (!tbody) {
            return;
        }
        if (!gd || !gd.data) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-muted small">No rows for this view yet.</td></tr>';
            return;
        }
        var skip = /^(Holidays|_pbjEvents)/;
        var traces = (gd.data || []).filter(function (t) {
            return t && t.x && t.y && !(t.name && skip.test(String(t.name)));
        });
        if (!traces.length) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-muted small">No rows for this view yet.</td></tr>';
            return;
        }
        var x = traces[0].x || [];
        var headers = ['Period'].concat(traces.map(function (t) {
            return String(t.name || 'Series');
        }));
        if (theadRow) {
            theadRow.innerHTML = headers.map(function (h, i) {
                return pbjV2RollupSortTh(h, i, i > 0 ? 'text-end' : '');
            }).join('');
        }
        var esc = typeof pbjV2EinHcEscHtml === 'function' ? pbjV2EinHcEscHtml : function (t) { return String(t == null ? '' : t); };
        var rows = [];
        for (var i = x.length - 1; i >= 0; i--) {
            if (rows.length >= (options.maxRows || 120)) {
                break;
            }
            var periodRaw = x[i];
            var periodLabel = String(periodRaw);
            var periodKey = pbjV2RollupPeriodKeyFromRaw(periodRaw);
            var periodTd = pbjV2RollupPeriodIsActionable(periodRaw)
                ? '<td data-sort="' + esc(periodKey) + '">' + pbjV2RollupPeriodBtnHtml(periodRaw, periodLabel, esc) + '</td>'
                : '<td data-sort="' + esc(periodLabel) + '">' + esc(periodLabel) + '</td>';
            var cells = [periodTd];
            traces.forEach(function (t) {
                var v = t.y && t.y[i];
                var vn = typeof v === 'number' ? v : parseFloat(String(v));
                var display = pbjV2FormatRollupCell(v, options.digits);
                var sortVal = Number.isFinite(vn) ? String(vn) : '';
                cells.push('<td class="text-end" data-sort="' + esc(sortVal) + '">' + display + '</td>');
            });
            rows.push('<tr>' + cells.join('') + '</tr>');
        }
        tbody.innerHTML = rows.join('') ||
            '<tr><td colspan="' + headers.length + '" class="text-muted small">No rows for this view yet.</td></tr>';
    }

    function pbjV2RefreshCompositionRollupTable() {
        var activePane = document.querySelector('#compositionTrendTabs .nav-link.active');
        var chartId = activePane && activePane.id === 'compositionTrendTabDirect'
            ? 'compositionDirectTrendChart'
            : 'compositionTotalTrendChart';
        pbjV2RefreshPlotlyRollupTable(chartId, 'compositionRollupTable', 'compositionRollupTableBody', { digits: 3 });
    }

    function pbjV2RefreshWorkforceRollupTables() {
        pbjV2RefreshCompositionRollupTable();
        pbjV2RefreshEinHeadcountRollupTable();
        pbjV2RefreshPlotlyRollupTable('dowComparisonChart', 'dowRollupTable', 'dowRollupTableBody', { digits: 3 });
        pbjV2RefreshPlotlyRollupTable('contractTrendChart', 'contractRollupTable', 'contractRollupTableBody', { digits: 1 });
    }

    function pbjV2WireChartRollupCollapse(collapseId, refreshFn) {
        var el = document.getElementById(collapseId);
        if (!el || el.dataset.pbjRollupWired === '1' || typeof refreshFn !== 'function') {
            return;
        }
        el.dataset.pbjRollupWired = '1';
        el.addEventListener('show.bs.collapse', function () {
            refreshFn();
        });
    }

    function pbjV2InitWorkforceRollupDisclosures() {
        pbjV2WireEinHeadcountRollupClicks();
        pbjV2WireChartRollupCollapse('compositionRollupCollapse', pbjV2RefreshCompositionRollupTable);
        pbjV2WireChartRollupCollapse('einHeadcountRollupCollapse', pbjV2RefreshEinHeadcountRollupTable);
        pbjV2WireChartRollupCollapse('ratingsOverTimeDetailCollapse', function () {
            pbjV2ResizeCensusDetailCharts();
            pbjV2RefreshRatingsRollupTable();
        });
        pbjV2WireChartRollupCollapse('dowRollupCollapse', function () {
            pbjV2RefreshPlotlyRollupTable('dowComparisonChart', 'dowRollupTable', 'dowRollupTableBody', { digits: 3 });
        });
        pbjV2WireChartRollupCollapse('contractRollupCollapse', function () {
            pbjV2RefreshPlotlyRollupTable('contractTrendChart', 'contractRollupTable', 'contractRollupTableBody', { digits: 1 });
        });
    }

    function pbjV2HprdViewGrainValue() {
        var r = document.querySelector('input[name="hprdView"]:checked');
        return r && r.value ? r.value : 'daily';
    }

    function pbjV2CensusViewIdForGrain(grain) {
        var map = {
            daily: 'censusDaily',
            day: 'censusDaily',
            month: 'censusMonthly',
            monthly: 'censusMonthly',
            quarter: 'censusQuarterly',
            quarterly: 'censusQuarterly',
            year: 'censusYearly',
            annual: 'censusYearly'
        };
        return map[String(grain || '').toLowerCase()] || 'censusMonthly';
    }

    function pbjV2SyncCensusViewFromHprd() {
        if (global.__pbjCensusViewUserOverride) {
            return false;
        }
        var grain = pbjV2HprdViewGrainValue();
        var id = pbjV2CensusViewIdForGrain(grain);
        var el = document.getElementById(id);
        if (!el) {
            return false;
        }
        var changed = !el.checked;
        if (changed) {
            el.checked = true;
        }
        return changed;
    }

    function pbjV2CensusMainTraceFromCharts(charts) {
        if (!charts || !charts.census_trend || !charts.census_trend.data) {
            return null;
        }
        return charts.census_trend.data.find(function (t) {
            return t && t.x && t.y && !(t.name && String(t.name).indexOf('Holiday') >= 0);
        }) || charts.census_trend.data[0] || null;
    }

    function pbjV2FormatCensusContextNumber(val, digits) {
        var n = typeof val === 'number' ? val : parseFloat(String(val));
        if (!Number.isFinite(n)) {
            return '—';
        }
        return n.toFixed(digits != null ? digits : 1);
    }

    function pbjV2RenderCensusContextSparkline(mount, values, opts) {
        opts = opts || {};
        var allowZero = !!opts.allowZero;
        var minSpan = opts.minSpan != null ? opts.minSpan : 0.5;
        var flatEpsilon = opts.flatEpsilon != null ? opts.flatEpsilon : 0.05;
        if (!mount) {
            return;
        }
        var nums = (values || []).filter(function (v) {
            if (v == null || !Number.isFinite(Number(v))) {
                return false;
            }
            var n = Number(v);
            return allowZero ? n >= 0 : n > 0;
        }).map(function (v) {
            return Number(v);
        });
        mount.innerHTML = '';
        if (nums.length < 2) {
            mount.innerHTML = '<span class="small text-muted">—</span>';
            return;
        }
        var w = 168;
        if (mount.closest && mount.closest('.pbj-census-context-strip--rail')) {
            w = global.innerWidth < 768 ? 128 : 132;
        } else if (
            mount.closest &&
            (mount.closest('.pbj-census-context-strip--profile') ||
                mount.closest('.pbj-census-context-strip--metric-trend'))
        ) {
            w = 112;
        } else if (global.innerWidth < 768) {
            w = 128;
        }
        var h = 32;
        var pad = 3;
        var min = Math.min.apply(null, nums);
        var max = Math.max.apply(null, nums);
        var span = Math.max(max - min, minSpan);
        var pts = [];
        for (var i = 0; i < nums.length; i++) {
            var x = nums.length <= 1 ? w / 2 : pad + (i / (nums.length - 1)) * (w - pad * 2);
            var y = pad + (h - pad * 2) - ((nums[i] - min) / span) * (h - pad * 2);
            pts.push(x.toFixed(1) + ',' + y.toFixed(1));
        }
        var stroke = nums[nums.length - 1] >= nums[0] ? '#15803d' : '#b91c1c';
        if (Math.abs(nums[nums.length - 1] - nums[0]) < flatEpsilon) {
            stroke = '#64748b';
        }
        mount.innerHTML =
            '<svg class="pbj-spark-trend-svg" width="' + w + '" height="' + h + '" viewBox="0 0 ' + w + ' ' + h + '" aria-hidden="true">' +
            '<polyline fill="none" stroke="' + stroke + '" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" points="' + pts.join(' ') + '" />' +
            '</svg>';
    }

    function pbjV2ChartTrendTraceY(charts, chartKey, pickTrace) {
        var pack = charts && charts[chartKey];
        if (!pack || !pack.data || !pack.data.length) {
            return null;
        }
        var trace = pack.data.find(pickTrace);
        if (!trace || !trace.y || !trace.y.length) {
            return null;
        }
        return trace.y.slice();
    }

    function pbjV2RefreshPbjTrendsContextStrips(charts) {
        if (typeof document === 'undefined') {
            return;
        }
        function paintMount(mountId, values, renderOpts) {
            var el = document.getElementById(mountId);
            if (!el || el.closest('.d-none')) {
                return;
            }
            if (!values || !values.length) {
                el.innerHTML = '<span class="small text-muted">—</span>';
                return;
            }
            pbjV2RenderCensusContextSparkline(el, values, renderOpts || {});
        }
        if (!charts) {
            return;
        }
        paintMount(
            'pbjSparkTrendTotalHprd',
            pbjV2ChartTrendTraceY(charts, 'hprd_trend', function (t) {
                return t && String(t.name) === 'Total';
            })
        );
        paintMount(
            'pbjSparkTrendRnHprd',
            pbjV2ChartTrendTraceY(charts, 'hprd_trend', function (t) {
                return t && String(t.name) === 'RN (Total)';
            })
        );
        var naStrip = document.getElementById('pbjSummaryNaHprdMetricStrip');
        var showNa = naStrip && !naStrip.classList.contains('d-none');
        if (showNa) {
            paintMount(
                'pbjSparkTrendNaHprd',
                pbjV2ChartTrendTraceY(charts, 'hprd_trend', function (t) {
                    return t && String(t.name) === 'Nurse Aide';
                })
            );
        } else {
            paintMount(
                'pbjSparkTrendContract',
                pbjV2ChartTrendTraceY(charts, 'contract_trend', function (t) {
                    return t && String(t.name) === 'Total %';
                }),
                { allowZero: true, minSpan: 0.05, flatEpsilon: 0.02 }
            );
        }
    }

    function pbjV2LatestCertifiedBedsFromProvider() {
        var occ = global.providerChartData && global.providerChartData.occupancy;
        if (!occ || !Array.isArray(occ.certified_beds)) {
            return null;
        }
        for (var i = occ.certified_beds.length - 1; i >= 0; i--) {
            var v = occ.certified_beds[i];
            if (v != null && v !== '' && Number.isFinite(Number(v)) && Number(v) > 0) {
                return Number(v);
            }
        }
        return null;
    }

    function pbjV2CertifiedBedsForCensusContext(period, grain) {
        if (typeof global.__pbjProviderMatchedCertifiedBeds === 'number' && global.__pbjProviderMatchedCertifiedBeds > 0) {
            return global.__pbjProviderMatchedCertifiedBeds;
        }
        if (period != null && typeof pbjV2BuildOccupancyLookup === 'function' && typeof pbjV2LookupOccupancyForPeriod === 'function') {
            var occMap = pbjV2BuildOccupancyLookup();
            if (grain === 'year') {
                var annual = pbjV2AnnualOccupancyFromQuarters(period, occMap);
                if (annual && annual.beds != null && Number.isFinite(Number(annual.beds)) && Number(annual.beds) > 0) {
                    return Number(annual.beds);
                }
            }
            var hit = pbjV2LookupOccupancyForPeriod(period, grain, occMap);
            if (hit && hit.beds != null && Number.isFinite(Number(hit.beds)) && Number(hit.beds) > 0) {
                return Number(hit.beds);
            }
        }
        return pbjV2LatestCertifiedBedsFromProvider();
    }

    function pbjV2LatestOccupancyPctForPeriod(period, grain, census) {
        if (typeof pbjV2BuildOccupancyLookup !== 'function' || typeof pbjV2LookupOccupancyForPeriod !== 'function') {
            return null;
        }
        var hit = pbjV2LookupOccupancyForPeriod(period, grain, pbjV2BuildOccupancyLookup());
        if (!hit) {
            return null;
        }
        if (hit.occupancy_pct != null && Number.isFinite(Number(hit.occupancy_pct))) {
            return Number(hit.occupancy_pct);
        }
        var beds = hit.beds;
        if (census != null && beds != null && Number.isFinite(Number(beds)) && Number(beds) > 0 && Number.isFinite(Number(census)) && Number(census) > 0) {
            return Number(census) / Number(beds) * 100;
        }
        return null;
    }

    function pbjV2FormatCensusContextDateShort(isoYmd) {
        var s = String(isoYmd || '').trim();
        var m = s.match(/^(\d{4})-(\d{2})-(\d{2})$/);
        if (!m) {
            return s;
        }
        return parseInt(m[2], 10) + '-' + parseInt(m[3], 10) + '-' + m[1].slice(-2);
    }

    function pbjV2FormatCensusContextPeriodLabel() {
        var ft = (document.querySelector('input[name="filterType"]:checked') || {}).value || 'quarters';
        var fmtQ = typeof global.formatQuarter === 'function'
            ? global.formatQuarter
            : function (q) {
                var mm = String(q || '').match(/(\d{4})Q(\d)/i);
                return mm ? ('Q' + mm[2] + ' ' + mm[1]) : String(q || '');
            };
        if (ft === 'day') {
            var dayIso = (document.getElementById('filterDayDate') || {}).value || '';
            if (dayIso) {
                return pbjV2FormatCensusContextDateShort(dayIso);
            }
        }
        if (ft === 'daterange') {
            var sd = (document.getElementById('startDate') || {}).value || '';
            var ed = (document.getElementById('endDate') || {}).value || '';
            if (sd && ed) {
                var a = pbjV2FormatCensusContextDateShort(sd);
                var b = pbjV2FormatCensusContextDateShort(ed);
                return a === b ? a : (a + ' to ' + b);
            }
        }
        if (ft === 'years') {
            var ySel = document.getElementById('years');
            var yrs = ySel
                ? Array.from(ySel.selectedOptions || []).map(function (o) { return String(o.value || '').trim(); }).filter(function (v) { return v && v !== 'all' && /^\d{4}$/.test(v); })
                : [];
            if (!yrs.length && document.getElementById('yearsMobileCompact')) {
                var mv = String(document.getElementById('yearsMobileCompact').value || '').trim();
                if (mv && mv !== 'all') {
                    yrs = [mv];
                }
            }
            yrs.sort();
            if (yrs.length === 1) {
                return yrs[0];
            }
            if (yrs.length > 1) {
                return yrs[0] + '–' + yrs[yrs.length - 1];
            }
        }
        if (ft === 'quarters') {
            var qKeys = typeof pbjV2SelectedQuarterKeysFromDom === 'function'
                ? pbjV2SelectedQuarterKeysFromDom()
                : [];
            if (!qKeys.length) {
                var qSel = document.getElementById('quarterRange');
                qKeys = qSel
                    ? Array.from(qSel.selectedOptions || []).map(function (o) { return String(o.value || '').trim(); }).filter(function (v) { return v && v !== 'all'; })
                    : [];
            }
            if (qKeys.length === 1) {
                return fmtQ(qKeys[0]);
            }
            if (qKeys.length > 1) {
                var sortedQ = qKeys.slice().sort(function (a, b) {
                    return pbjCyQuarterSortKey(a) - pbjCyQuarterSortKey(b);
                });
                return fmtQ(sortedQ[0]) + ' – ' + fmtQ(sortedQ[sortedQ.length - 1]);
            }
        }
        return 'All data';
    }

    function pbjV2EinHcTableJobOrder(codes) {
        if (typeof global.pbjEinHcTableJobOrder === 'function') {
            return global.pbjEinHcTableJobOrder(codes);
        }
        var tableLeftFirst = [7, 5, 6, 8, 9, 10, 12, 11];
        var seen = {};
        var ordered = [];
        (codes || []).forEach(function (c) {
            var n = parseInt(c, 10);
            if (!isNaN(n)) {
                seen[n] = 1;
            }
        });
        tableLeftFirst.forEach(function (c) {
            if (seen[c]) {
                ordered.push(String(c));
                delete seen[c];
            }
        });
        Object.keys(seen).map(function (k) { return parseInt(k, 10); }).sort(function (a, b) { return a - b; })
            .forEach(function (c) { ordered.push(String(c)); });
        return ordered;
    }

    function pbjV2EinHcStackJobOrder(codes) {
        if (typeof global.pbjEinHcStackJobOrder === 'function') {
            return global.pbjEinHcStackJobOrder(codes);
        }
        var stackBottomFirst = [11, 12, 10, 8, 9, 5, 6, 7];
        var seen = {};
        var ordered = [];
        (codes || []).forEach(function (c) {
            var n = parseInt(c, 10);
            if (!isNaN(n)) {
                seen[n] = 1;
            }
        });
        stackBottomFirst.forEach(function (c) {
            if (seen[c]) {
                ordered.push(String(c));
                delete seen[c];
            }
        });
        Object.keys(seen).map(function (k) { return parseInt(k, 10); }).sort(function (a, b) { return a - b; })
            .forEach(function (c) { ordered.push(String(c)); });
        return ordered;
    }

    var PBJ_EIN_HC_LEGEND_JOB_CODES = {
        'RN DON': '5',
        'RN Admin': '6',
        RN: '7',
        'LPN Admin': '8',
        LPN: '9',
        CNA: '10',
        'Aide Trainee': '11',
        'Med Aide': '12',
    };

    function pbjV2EinHcJobCodeFromTrace(t) {
        if (t && t.meta && t.meta.jobCode != null && String(t.meta.jobCode).trim() !== '') {
            return String(t.meta.jobCode).trim();
        }
        var name = String((t && t.name) || '').trim();
        var paren = name.match(/\((\d{1,2})\)\s*$/);
        if (paren) {
            return paren[1];
        }
        return PBJ_EIN_HC_LEGEND_JOB_CODES[name] || '';
    }

    function pbjV2EinHcParseJobCodeFromTraceName(name) {
        return pbjV2EinHcJobCodeFromTrace({ name: name });
    }

    function pbjV2SetEinHeadcountRollupTablePlaceholder(message) {
        var tbody = document.getElementById('einHeadcountRollupTableBody');
        var theadRow = document.querySelector('#einHeadcountRollupTableHead tr');
        if (!tbody) {
            return;
        }
        if (theadRow) {
            theadRow.innerHTML = pbjV2RollupSortTh('Period', 0, '');
        }
        var msg = String(message || 'Loading…');
        tbody.innerHTML =
            '<tr><td colspan="8" class="text-muted small">' + pbjV2EinHcEscHtml(msg) + '</td></tr>';
    }

    function pbjV2EinHcChartHasPlot() {
        var gd = document.getElementById('einHeadcountByJobChart');
        return !!(gd && (gd.data && gd.data.length || gd.querySelector && gd.querySelector('.js-plotly-plot')));
    }

    function pbjV2EinHcPeriodKeyFromLabel(label, pack) {
        if (!pack || !pack.data || !pack.data.periods) {
            return String(label || '');
        }
        var hit = (pack.data.periods || []).find(function (p) {
            return String(p.label || '') === String(label) || String(p.key || '') === String(label);
        });
        return hit ? String(hit.key) : String(label || '');
    }

    function pbjV2EinHcPeriodToQuarter(periodKey) {
        var k = String(periodKey || '').trim();
        var cy = k.match(/^CY(\d{4})Q([1-4])$/i);
        if (cy) {
            return 'CY' + cy[1] + 'Q' + cy[2];
        }
        var q = k.match(/^Q([1-4])\s+(\d{4})$/i);
        if (q) {
            return 'CY' + q[2] + 'Q' + q[1];
        }
        var iso = k.match(/^(\d{4})-(\d{2})-(\d{2})$/);
        if (iso) {
            var month = parseInt(iso[2], 10);
            return 'CY' + iso[1] + 'Q' + Math.ceil(month / 3);
        }
        var ym = k.match(/^(\d{4})-(\d{2})$/);
        if (ym) {
            return 'CY' + ym[1] + 'Q' + Math.ceil(parseInt(ym[2], 10) / 3);
        }
        if (/^\d{4}$/.test(k)) {
            return 'CY' + k + 'Q4';
        }
        return '';
    }

    function pbjV2EinHcEscHtml(t) {
        if (typeof global.escapeHtml === 'function') {
            return global.escapeHtml(t);
        }
        return String(t == null ? '' : t)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function pbjV2EinHcRollupHasRenderableData(tracesArg) {
        if (global.__pbjEinHeadcountByJobExportData && global.__pbjEinHeadcountByJobExportData.periods) {
            return true;
        }
        if (Array.isArray(tracesArg) && tracesArg.length) {
            return true;
        }
        var gd = document.getElementById('einHeadcountByJobChart');
        if (gd && gd.data && gd.data.length) {
            return gd.data.some(function (t) {
                return t && t.x && t.y && t.type === 'bar';
            });
        }
        return false;
    }

    function pbjV2RefreshEinHeadcountRollupTable(tracesArg) {
        var tbody = document.getElementById('einHeadcountRollupTableBody');
        var theadRow = document.querySelector('#einHeadcountRollupTableHead tr');
        if (!tbody) {
            return;
        }
        if (
            global.__pbjEinHeadcountByJobLoading &&
            !pbjV2EinHcRollupHasRenderableData(tracesArg) &&
            !pbjV2EinHcChartHasPlot()
        ) {
            pbjV2SetEinHeadcountRollupTablePlaceholder('Loading…');
            return;
        }
        if (global.__pbjEinHeadcountByJobLoading && pbjV2EinHcChartHasPlot()) {
            global.__pbjEinHeadcountByJobLoading = false;
        }
        var pack = global.__pbjEinHeadcountByJobExportData || null;
        var traces = Array.isArray(tracesArg) ? tracesArg : null;
        if (!traces) {
            var gd = document.getElementById('einHeadcountByJobChart');
            if (!gd || !gd.data) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-muted small">No rows for this view yet.</td></tr>';
                return;
            }
            traces = (gd.data || []).filter(function (t) {
                return t && t.x && t.y && t.type === 'bar' && !(t.name && /^Contract$/i.test(String(t.name)));
            });
        } else {
            traces = traces.filter(function (t) {
                return t && t.x && t.y && t.type === 'bar' && !(t.name && /^Contract$/i.test(String(t.name)));
            });
        }
        if (!traces.length) {
            tbody.innerHTML = '<tr><td colspan="4" class="text-muted small">No rows for this view yet.</td></tr>';
            return;
        }
        var jobMeta = traces.map(function (t) {
            var jc = pbjV2EinHcJobCodeFromTrace(t);
            return { trace: t, jobCode: jc, name: String(t.name || 'Series') };
        });
        var jobOrder = pbjV2EinHcTableJobOrder(jobMeta.map(function (j) { return j.jobCode; }).filter(Boolean));
        jobMeta.sort(function (a, b) {
            var ai = jobOrder.indexOf(String(a.jobCode));
            var bi = jobOrder.indexOf(String(b.jobCode));
            if (ai < 0) ai = 999;
            if (bi < 0) bi = 999;
            return ai - bi;
        });
        var priorityCodes = { '7': 1, '9': 1, '10': 1, '12': 1, '5': 1, '6': 1, '8': 1, '11': 1 };
        if (theadRow) {
            var headCells = [pbjV2RollupSortTh('Period', 0, '')];
            jobMeta.forEach(function (j, idx) {
                var pri = priorityCodes[j.jobCode] ? ' pbj-ein-hc-rollup-col-priority' : '';
                headCells.push(pbjV2RollupSortTh(pbjV2EinHcEscHtml(j.name), idx + 1, 'text-end' + pri));
            });
            theadRow.innerHTML = headCells.join('');
        }
        var x = traces[0].x || [];
        var rows = [];
        for (var i = x.length - 1; i >= 0; i--) {
            if (rows.length >= 120) {
                break;
            }
            var periodLabel = String(x[i]);
            var periodKey = pbjV2EinHcPeriodKeyFromLabel(periodLabel, pack);
            var periodTd = pbjV2RollupPeriodIsActionable(periodKey || periodLabel)
                ? '<td data-sort="' + pbjV2EinHcEscHtml(periodKey || periodLabel) + '">' +
                    pbjV2RollupPeriodBtnHtml(periodKey || periodLabel, periodLabel, pbjV2EinHcEscHtml) + '</td>'
                : '<td data-sort="' + pbjV2EinHcEscHtml(periodLabel) + '">' + pbjV2EinHcEscHtml(periodLabel) + '</td>';
            var cells = [periodTd];
            jobMeta.forEach(function (j) {
                var v = j.trace.y && j.trace.y[i];
                var n = typeof v === 'number' ? v : parseFloat(String(v));
                var display = Number.isFinite(n) ? String(Math.round(n)) : '—';
                if (Number.isFinite(n) && j.jobCode) {
                    var priCls = priorityCodes[j.jobCode] ? ' pbj-ein-hc-rollup-col-priority' : '';
                    if (n > 0) {
                        var title = j.name.replace(/"/g, '&quot;');
                        cells.push(
                            '<td class="text-end' + priCls + '" data-sort="' + pbjV2EinHcEscHtml(String(Math.round(n))) + '">' +
                            '<button type="button" class="pbj-ein-hc-rollup-cell-btn" ' +
                            'data-period-key="' + pbjV2EinHcEscHtml(periodKey) + '" ' +
                            'data-period-label="' + pbjV2EinHcEscHtml(periodLabel) + '" ' +
                            'data-job-code="' + pbjV2EinHcEscHtml(j.jobCode) + '" ' +
                            'data-job-title="' + title + '">' + display + '</button></td>'
                        );
                    } else {
                        cells.push(
                            '<td class="text-end' + priCls + '" data-sort="0">0</td>'
                        );
                    }
                } else {
                    cells.push(
                        '<td class="text-end" data-sort="' + (Number.isFinite(n) ? pbjV2EinHcEscHtml(String(Math.round(n))) : '') + '">' +
                        (Number.isFinite(n) ? display : '—') + '</td>'
                    );
                }
            });
            rows.push('<tr>' + cells.join('') + '</tr>');
        }
        tbody.innerHTML = rows.join('') ||
            '<tr><td colspan="' + (jobMeta.length + 1) + '" class="text-muted small">No rows for this view yet.</td></tr>';
    }

    function pbjV2WireEinHeadcountRollupClicks() {
        var tbody = document.getElementById('einHeadcountRollupTableBody');
        if (!tbody || tbody.dataset.pbjEinHcRollupWired === '1') {
            return;
        }
        tbody.dataset.pbjEinHcRollupWired = '1';
        tbody.addEventListener('click', function (ev) {
            var btn = ev.target.closest('.pbj-ein-hc-rollup-cell-btn');
            if (!btn) {
                return;
            }
            pbjV2OpenEinHeadcountJobModal(
                btn.getAttribute('data-period-key'),
                btn.getAttribute('data-period-label'),
                btn.getAttribute('data-job-code'),
                btn.getAttribute('data-job-title'),
                btn.textContent
            );
        });
    }

    function pbjV2EinHcEmployeeFlagsHtml(e, inline) {
        var parts = [];
        if (e && e.new_to_quarter_known && e.is_new_to_quarter) {
            parts.push(
                '<span class="badge rounded-pill ein-new-quarter-badge' +
                    (inline ? ' ms-1' : ' me-1') +
                    '" role="status" title="New to this job code in quarter">New</span>'
            );
        }
        var ctrPct = parseFloat(e && e.pct_contract_hours);
        if (e && (e.day_contract_flag || (!isNaN(ctrPct) && ctrPct > 0))) {
            parts.push(
                '<span class="badge rounded-pill ein-contract-pill-badge' +
                    (inline ? ' ms-1' : ' me-1') +
                    '" role="status" title="Contract (agency) hours in period">Contract</span>'
            );
        }
        if (e && e.is_multi_role_employee && e.multi_role_job_titles && e.multi_role_job_titles.length > 1) {
            var tip = 'Dual role — also held: ' + e.multi_role_job_titles.join(', ');
            parts.push(
                '<span class="badge rounded-pill ein-dual-role-badge' +
                    (inline ? ' ms-1' : '') +
                    '" role="status" title="' +
                    pbjV2EinHcEscHtml(tip) +
                    '">Dual</span>'
            );
        }
        if (!parts.length) {
            return inline ? '' : '<span class="text-muted">—</span>';
        }
        return parts.join('');
    }

    function pbjV2EinHcModalSummaryLine(rows, roleLabel, shownCount) {
        var dual = 0;
        var contract = 0;
        var neu = 0;
        (rows || []).forEach(function (e) {
            if (e && e.is_multi_role_employee && e.multi_role_job_titles && e.multi_role_job_titles.length > 1) {
                dual += 1;
            }
            var ctrPct = parseFloat(e && e.pct_contract_hours);
            if (e && (e.day_contract_flag || (!isNaN(ctrPct) && ctrPct > 0))) {
                contract += 1;
            }
            if (e && e.new_to_quarter_known && e.is_new_to_quarter) {
                neu += 1;
            }
        });
        var roleWord = String(roleLabel || 'employee').trim() || 'employee';
        var line =
            rows.length +
            ' unique ' +
            roleWord +
            '; ' +
            dual +
            ' dual · ' +
            contract +
            ' contract · ' +
            neu +
            ' new';
        if (shownCount && shownCount < rows.length) {
            line += ' · showing top ' + shownCount + ' by hours';
        }
        line += '. Tap a row for employee detail.';
        return line;
    }

    function pbjV2OpenEinHeadcountJobModal(periodKey, periodLabel, jobCode, jobTitle, countText) {
        var modalEl = document.getElementById('einHeadcountJobEmployeesModal');
        var body = document.getElementById('einHeadcountJobEmployeesModalBody');
        var titleEl = document.getElementById('einHeadcountJobEmployeesModalLabel');
        if (!body) {
            return;
        }
        var jc = String(jobCode || '').trim();
        var label = String(jobTitle || ('Job ' + jc)).trim();
        if (titleEl) {
            titleEl.textContent = label + ' · ' + String(periodLabel || periodKey || 'Period');
        }
        body.innerHTML = '<p class="text-muted mb-0"><span class="spinner-border spinner-border-sm me-2" role="status" aria-hidden="true"></span>Loading employees…</p>';
        if (modalEl && typeof global.bootstrap !== 'undefined' && global.bootstrap.Modal) {
            global.bootstrap.Modal.getOrCreateInstance(modalEl).show();
        }
        var slice = String(global.__pbjEinHcSlice || 'nurse').toLowerCase();
        var apiBase = typeof global.pbjApiUrl === 'function' ? global.pbjApiUrl : function (p) { return p; };
        var apiPath = slice === 'nonnurse' ? '/api/ein-nonnurse-employees' : '/api/ein-nursing-employees';
        var quarter = pbjV2EinHcPeriodToQuarter(periodKey);
        var qs = '?limit=80&job_code=' + encodeURIComponent(jc);
        if (quarter) {
            qs += '&quarter=' + encodeURIComponent(quarter);
        }
        if (String(periodKey).match(/^\d{4}-\d{2}-\d{2}$/)) {
            qs += '&work_date=' + encodeURIComponent(periodKey);
        }
        fetch(apiBase(apiPath) + qs)
            .then(function (r) { return r.json(); })
            .then(function (j) {
                var rows = (j && j.employees) ? j.employees : [];
                if (!rows.length) {
                    body.innerHTML = '<p class="text-muted mb-0">No employees found for this position in the selected period. Try the full <a href="#einNursingEmployeeSection">roster</a> for broader filters.</p>';
                    return;
                }
                rows.sort(function (a, b) {
                    return -(parseFloat(a.total_hours) || 0) + (parseFloat(b.total_hours) || 0);
                });
                var openFn = typeof global.openEinNursingEmployeeModal === 'function'
                    ? 'openEinNursingEmployeeModal'
                    : '';
                var shown = rows.slice(0, 80);
                var tableRows = shown.map(function (e) {
                    var sid = e.sys_employee_id;
                    var qtr = e.quarter || quarter || '';
                    var hrs = e.total_hours != null ? Number(e.total_hours).toFixed(1) : '—';
                    var days = e.work_days != null && !isNaN(parseInt(e.work_days, 10))
                        ? String(parseInt(e.work_days, 10))
                        : '—';
                    var role = e.job_title_short || e.job_title || label;
                    var flags = pbjV2EinHcEmployeeFlagsHtml(e, true);
                    var rowClick = openFn
                        ? ' onclick="' + openFn + '(\'' + sid + '\',\'' + jc + '\',\'' + String(qtr).replace(/'/g, '') + '\')"'
                        : '';
                    return (
                        '<tr class="ein-hc-job-row"' + rowClick + ' title="Open employee detail">' +
                        '<td class="text-nowrap fw-semibold">' + pbjV2EinHcEscHtml(String(role)) + '</td>' +
                        '<td class="font-monospace ein-hc-emp-id-cell">' +
                        pbjV2EinHcEscHtml(String(sid)) +
                        flags +
                        '</td>' +
                        '<td class="text-end font-monospace text-nowrap">' + pbjV2EinHcEscHtml(days) + '</td>' +
                        '<td class="text-end font-monospace text-nowrap">' + pbjV2EinHcEscHtml(hrs) + '</td>' +
                        '</tr>'
                    );
                }).join('');
                var summaryLine = pbjV2EinHcModalSummaryLine(rows, label, shown.length);
                body.innerHTML =
                    '<p class="text-muted mb-2" style="font-size:0.78rem;">' + pbjV2EinHcEscHtml(summaryLine) + '</p>' +
                    '<div class="table-responsive border rounded">' +
                    '<table class="table table-sm table-hover mb-0 align-middle ein-hc-job-table">' +
                    '<thead><tr>' +
                    '<th scope="col">Role</th>' +
                    '<th scope="col">Employee ID</th>' +
                    '<th scope="col" class="text-end">Days</th>' +
                    '<th scope="col" class="text-end">Hrs</th>' +
                    '</tr></thead>' +
                    '<tbody>' + tableRows + '</tbody></table></div>';
            })
            .catch(function () {
                body.innerHTML = '<p class="text-danger mb-0">Could not load employees for this position.</p>';
            });
    }

    function pbjV2RefreshRatingsRollupTable() {
        var tbody = document.getElementById('ratingsOverTimeRollupTableBody');
        if (!tbody) {
            return;
        }
        var data = global.__pbjRatingsRollupData;
        if (!data || !data.quarters || !data.quarters.length) {
            tbody.innerHTML = '<tr><td colspan="7" class="text-muted small">No ratings rows for this view yet.</td></tr>';
            return;
        }
        var series = [
            { key: 'overall', label: 'Overall' },
            { key: 'staffing', label: 'Staffing' },
            { key: 'health_inspection', label: 'Inspection' },
            { key: 'quality', label: 'Quality' },
            { key: 'long_stay_qm', label: 'Long-stay QM' },
            { key: 'short_stay_qm', label: 'Short-stay QM' }
        ];
        var fmtQ = typeof global.formatQuarter === 'function'
            ? global.formatQuarter
            : function (q) { return String(q || ''); };
        var maxRows = 10;
        var startIdx = Math.max(0, data.quarters.length - maxRows);
        var rows = [];
        for (var i = data.quarters.length - 1; i >= startIdx; i--) {
            var q = data.quarters[i];
            var qKey = pbjV2PeriodToCyQuarterKey(q, 'quarter') || String(q || '');
            var qLabel = fmtQ(q);
            var qCell = qKey
                ? '<td data-sort="' + pbjV2EinHcEscHtml(qKey) + '">' +
                    pbjV2RollupPeriodBtnHtml(qKey, qLabel, pbjV2EinHcEscHtml) + '</td>'
                : '<td data-sort="' + pbjV2EinHcEscHtml(qLabel) + '">' + pbjV2EinHcEscHtml(qLabel) + '</td>';
            var cells = [qCell];
            series.forEach(function (s) {
                var arr = data[s.key] || [];
                var v = arr[i];
                var num = v != null && Number.isFinite(Number(v)) ? Number(v) : null;
                var display = num != null ? num.toFixed(0) : '—';
                cells.push(
                    '<td class="text-end" data-sort="' +
                    (num != null ? pbjV2EinHcEscHtml(String(num)) : '') +
                    '">' +
                    pbjV2EinHcEscHtml(display) +
                    '</td>'
                );
            });
            rows.push('<tr>' + cells.join('') + '</tr>');
        }
        tbody.innerHTML = rows.join('');
    }

    function pbjV2RefreshCensusContextStripMount(idPrefix, charts, resizeChart) {
        var strip = document.getElementById('pbj' + idPrefix + 'CensusContextStrip');
        if (!strip) {
            return;
        }
        var avgEl = document.getElementById('pbj' + idPrefix + 'CensusCtxAvg');
        var avgLabelEl = document.getElementById('pbj' + idPrefix + 'CensusCtxAvgLabel');
        var occEl = document.getElementById('pbj' + idPrefix + 'CensusCtxOccupancy');
        var sparkEl = document.getElementById('pbj' + idPrefix + 'CensusContextSparkline');
        var periodLabel = pbjV2FormatCensusContextPeriodLabel();
        if (avgLabelEl) {
            avgLabelEl.textContent = 'Avg census';
            avgLabelEl.removeAttribute('title');
        }
        var trace = pbjV2CensusMainTraceFromCharts(charts);
        if (!trace || !trace.y || !trace.y.length) {
            if (avgEl) avgEl.textContent = '—';
            if (occEl) occEl.textContent = '—';
            if (sparkEl) sparkEl.innerHTML = '<span class="small text-muted">—</span>';
            return;
        }
        var vals = [];
        var latestX = null;
        (trace.y || []).forEach(function (y, i) {
            var n = typeof y === 'number' ? y : parseFloat(String(y));
            if (!Number.isFinite(n) || n <= 0) {
                return;
            }
            vals.push(n);
            latestX = trace.x ? trace.x[i] : null;
        });
        if (!vals.length) {
            if (avgEl) avgEl.textContent = '—';
            if (occEl) occEl.textContent = '—';
            if (sparkEl) sparkEl.innerHTML = '<span class="small text-muted">—</span>';
            return;
        }
        var sum = vals.reduce(function (a, b) { return a + b; }, 0);
        var avg = sum / vals.length;
        if (avgEl) avgEl.textContent = pbjV2FormatCensusContextNumber(avg, 1);
        var tableGrain = pbjV2CensusTableGrain();
        var beds = pbjV2CertifiedBedsForCensusContext(latestX, tableGrain);
        var occPct = null;
        if (beds != null && beds > 0 && avg > 0) {
            occPct = avg / beds * 100;
        } else if (latestX != null) {
            occPct = pbjV2LatestOccupancyPctForPeriod(latestX, tableGrain, avg);
        }
        if (occEl) {
            if (occPct != null && Number.isFinite(occPct)) {
                occEl.textContent = pbjV2FormatCensusContextNumber(occPct, 1) + '%';
                if (beds != null && beds > 0) {
                    occEl.setAttribute(
                        'title',
                        'Avg census ÷ ' + Math.round(beds).toLocaleString('en-US') + ' certified beds (Provider Information)'
                    );
                } else {
                    occEl.removeAttribute('title');
                }
            } else {
                occEl.textContent = 'N/A';
                occEl.setAttribute('title', 'Certified bed count unavailable for this period');
            }
        }
        pbjV2RenderCensusContextSparkline(sparkEl, trace.y || []);
        if (resizeChart && typeof global.Plotly !== 'undefined' && global.Plotly.Plots && typeof global.Plotly.Plots.resize === 'function') {
            var chartEl = document.getElementById('hprdTrendChart');
            if (chartEl) {
                try {
                    global.Plotly.Plots.resize(chartEl);
                } catch (resizeErr) {
                    /* ignore */
                }
            }
        }
    }

    function pbjV2RefreshCensusContextStrip(charts) {
        pbjV2RefreshCensusContextStripMount('', charts, true);
        pbjV2RefreshCensusContextStripMount('Profile', charts, false);
        pbjV2RefreshPbjTrendsContextStrips(charts);
    }

    function pbjV2ResizeCensusDetailCharts() {
        ['censusTrendChart', 'providerInfoOccupancyChart', 'providerInfoRatingsChart'].forEach(function (chartId) {
            var el = document.getElementById(chartId);
            if (!el || !global.Plotly || !global.Plotly.Plots || !global.Plotly.Plots.resize) {
                return;
            }
            try {
                global.Plotly.Plots.resize(el);
            } catch (eResize) { /* ignore */ }
        });
    }

    function pbjV2InitCensusDetailDisclosure() {
        var detailCollapse = document.getElementById('censusOccupancyDetailCollapse');
        if (!detailCollapse || detailCollapse.dataset.pbjCensusDetailWired === '1') {
            return;
        }
        detailCollapse.dataset.pbjCensusDetailWired = '1';
        detailCollapse.addEventListener('show.bs.collapse', function () {
            pbjV2ResizeCensusDetailCharts();
            if (global.lastChartsResponse && global.lastChartsResponse.charts) {
                pbjV2RefreshCensusOccupancyTable(global.lastChartsResponse.charts);
            }
        });
        document.querySelectorAll('input[name="censusView"]').forEach(function (radio) {
            if (radio.dataset.pbjCensusDetailGrainWired === '1') {
                return;
            }
            radio.dataset.pbjCensusDetailGrainWired = '1';
            radio.addEventListener('change', function () {
                global.__pbjCensusViewUserOverride = true;
            });
        });
    }

    function pbjV2InitCensusRollupDisclosure() {
        pbjV2InitCensusDetailDisclosure();
    }

    function pbjV2EnsureControlsDockVisible() {
        var dock = document.getElementById('pbjControlsDock');
        var wrap = document.getElementById('pbjFloatingControls');
        if (!dock || !wrap) {
            return;
        }
        if (!dock.classList.contains('pbj-controls-dock--ready')) {
            pbjV2MountNavControls();
            pbjV2FinalizeControlsDockReady(dock);
        }
        dock.removeAttribute('aria-hidden');
        wrap.setAttribute('data-pbj-dock-mounted', '1');
    }

    function pbjV2InitFloatingControls() {
        var wrap = document.getElementById('pbjFloatingControls');
        var fab = document.getElementById('pbjFloatingControlsFab');
        var panel = document.getElementById('pbjFloatingControlsPanel');
        var closeBtn = document.getElementById('pbjFloatingControlsClose');
        var hint = document.getElementById('pbjFloatingControlsHintDot');
        if (!fab || !panel) {
            return;
        }
        if (wrap && wrap.dataset.pbjFloatingBound === '1') {
            return;
        }
        if (wrap) {
            wrap.dataset.pbjFloatingBound = '1';
        }
        pbjV2MountNavControls();
        function setOpen(open) {
            if (open) {
                pbjV2OpenControlCenter();
            } else {
                pbjV2CloseControlCenter();
            }
        }
        function toggleOpen() {
            setOpen(panel.hidden);
        }
        fab.addEventListener('click', function (ev) {
            if (typeof global.pbjV2ControlsDockClickSuppressed === 'function' && global.pbjV2ControlsDockClickSuppressed()) {
                ev.preventDefault();
                ev.stopPropagation();
                return;
            }
            ev.stopPropagation();
            toggleOpen();
        });
        pbjV2InitControlsDockDrag();
        pbjV2WireFloatingPeriod();
        pbjV2SyncFloatingGrain();
        /* Quarter options: wait for pbjQuartersLoaded — pull here races loadInitialData /api/quarters */
        document.addEventListener('pointerdown', function (ev) {
            if (panel.hidden || !wrap) {
                return;
            }
            if (pbjV2PointerOnWindowScrollbar(ev)) {
                return;
            }
            if (!pbjV2FloatingControlsPointerInside(wrap, panel, ev)) {
                setOpen(false);
            }
        });
        if (closeBtn) {
            closeBtn.addEventListener('click', function () {
                setOpen(false);
            });
        }
        var resetPosBtn = document.getElementById('pbjFloatingControlsResetPos');
        if (resetPosBtn) {
            resetPosBtn.addEventListener('click', function (ev) {
                ev.preventDefault();
                ev.stopPropagation();
                var dock = document.getElementById('pbjControlsDock');
                if (dock) {
                    pbjV2ResetControlsDockPosition(dock);
                    pbjV2DefaultControlsDockPosition(dock);
                }
                pbjV2AdjustControlsDockPanelPlacement();
            });
        }
        document.addEventListener('keydown', function (ev) {
            if (ev.key === 'Escape' && !panel.hidden) {
                setOpen(false);
            }
        });
        try {
            if (!localStorage.getItem('pbj_v2_controls_hint_seen') && hint) {
                hint.classList.remove('d-none');
            }
        } catch (eStore) {
            /* ignore */
        }
        panel.querySelectorAll('.pbj-floating-investigate-btn, #pbjFloatingPeriodFiltersLink').forEach(function (link) {
            link.addEventListener('click', function () {
                setOpen(false);
            });
        });
        panel.querySelectorAll('[data-pbj-floating-pane]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var pane = String(btn.getAttribute('data-pbj-floating-pane') || '').trim();
                if (!pane) {
                    return;
                }
                if (global.__pbjV3PanesActive && typeof global.pbjV3SwitchPane === 'function') {
                    global.pbjV3SwitchPane(pane);
                } else {
                    var legacyTarget = {
                        benchmarks: 'staffingBenchmarkingSection',
                        workforce: 'staffingPatternsWorkforceSection',
                        risk: 'pbjCitationsSection',
                        overview: 'pbjStaffingCoreSection',
                    }[pane];
                    if (legacyTarget) {
                        pbjV2ScrollToSection(legacyTarget);
                    }
                }
                setOpen(false);
            });
        });
        panel.querySelectorAll('[data-bs-toggle="modal"]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                setOpen(false);
            });
        });
        var compareBtn = document.getElementById('pbjFloatingCompareWindowsBtn');
        if (compareBtn) {
            compareBtn.addEventListener('click', function () {
                pbjOpenCompareWindows({ scroll: true });
                setOpen(false);
            });
        }
        var investigateBenchmarksBtn = document.getElementById('pbjFloatingInvestigateBenchmarksBtn');
        if (investigateBenchmarksBtn) {
            investigateBenchmarksBtn.addEventListener('click', function () {
                if (typeof global.pbjScreenCurrentPeriod === 'function') {
                    global.pbjScreenCurrentPeriod({ scroll: true, expand: true });
                }
                setOpen(false);
            });
        }
        var flagsBtn = document.getElementById('pbjFloatingFlagsBtn');
        if (flagsBtn) {
            flagsBtn.addEventListener('click', function () {
                pbjV2RevealInspectionsSection({ openFlags: true, scrollTarget: 'riskScreeningSection' });
                setOpen(false);
            });
        }
        var reportBtn = document.getElementById('pbjFloatingOpenReportBuilderBtn');
        if (reportBtn) {
            reportBtn.addEventListener('click', function () {
                pbjOpenReportBuilderFromControlCenter();
            });
        }
        var methodsBtn = document.getElementById('pbjFloatingMethodsBtn');
        if (methodsBtn) {
            methodsBtn.addEventListener('click', function () {
                setOpen(false);
            });
        }
        pbjV2RefreshFloatingReportContext();
    }

    /** Move HPRD trend chart into section 2 (filters stay in Summary). */
    function pbjV2AssembleStaffingCoreHub() {
        /* HPRD trend is server-rendered in partials/v2/pbj_staffing_core_shell.html (legacy mount optional). */
        var trendMount = document.getElementById('pbjCoreHubTrendMount');
        var hprd = document.getElementById('hprdTrendSection');
        var stack = document.getElementById('pbjStaffingCoreStack');
        if (!hprd || !stack) {
            return;
        }
        if (stack.contains(hprd)) {
            return;
        }
        if (trendMount && !trendMount.contains(hprd)) {
            trendMount.appendChild(hprd);
            return;
        }
        var compliance = document.getElementById('complianceStandardsSection');
        if (compliance && compliance.parentElement === stack) {
            stack.insertBefore(hprd, compliance);
        }
    }

    function pbjV2FormatChowDate(iso) {
        var p = String(iso || '').split('-');
        if (p.length !== 3) {
            return iso || '—';
        }
        return p[1] + '/' + p[2] + '/' + p[0];
    }

    function pbjV2RenderChowInto(hostId, payload) {
        var body = document.getElementById(hostId);
        if (!body) {
            return;
        }
        payload = payload || {};
        var txs = Array.isArray(payload.transactions) ? payload.transactions : [];
        global.__pbjChowEffectiveDates = txs.map(function (t) { return t.effective_date; }).filter(Boolean);
        global.__pbjChowLatestEffectiveDate = payload.latest_effective_date || global.__pbjChowEffectiveDates[0] || null;
        var anchorBtn = document.getElementById('pbjChowModalAnchorBtn');
        if (anchorBtn) {
            anchorBtn.classList.toggle('d-none', !global.__pbjChowLatestEffectiveDate);
        }
        global.__pbjLastChowPayload = payload;
        if (!txs.length) {
            body.innerHTML = '<p class="text-muted mb-0">No CMS CHOW transactions for this CCN in the current index. Provider Information may still flag ownership by quarter.</p>';
            if (typeof global.pbjFacilityEventsOnSourcesUpdated === 'function') {
                global.pbjFacilityEventsOnSourcesUpdated();
            }
            return;
        }
        var smartCase = typeof global.pbjSmartDisplayCase === 'function'
            ? global.pbjSmartDisplayCase
            : function (s) { return s; };
        var humanTag = typeof global.pbjHumanizeChowTag === 'function'
            ? global.pbjHumanizeChowTag
            : function (s) { return s; };
        var html = '<ul class="list-unstyled mb-2">';
        txs.forEach(function (t) {
            var buyer = smartCase(t.buyer_org_name || t.buyer_dba_name || '—');
            var seller = smartCase(t.seller_org_name || '—');
            var tag = humanTag(t.change_summary);
            var tagHtml = tag
                ? '<span class="pbj-chow-change-tag">' + tag.replace(/</g, '&lt;') + '</span>'
                : '';
            html +=
                '<li class="pbj-chow-tx-row">' +
                '<div class="pbj-chow-tx-head">' +
                '<span class="pbj-chow-tx-date">' +
                pbjV2FormatChowDate(t.effective_date) +
                '</span>' +
                tagHtml +
                '</div>' +
                '<div class="pbj-chow-tx-entities">' +
                '<span class="pbj-chow-buyer">' +
                String(buyer).replace(/</g, '&lt;') +
                '</span>' +
                '<span class="pbj-chow-seller">← ' +
                String(seller).replace(/</g, '&lt;') +
                '</span>' +
                '</div>' +
                '</li>';
        });
        html += '</ul>';
        html += '<p class="text-muted mb-0 small">Buyer and seller are CMS enrollment names from the ownership transfer record.</p>';
        body.innerHTML = html;
        if (typeof global.prePostSyncEventPresetButtons === 'function') {
            global.prePostSyncEventPresetButtons();
        }
        if (typeof global.pbjFacilityEventsOnSourcesUpdated === 'function') {
            global.pbjFacilityEventsOnSourcesUpdated();
        }
    }

    function pbjV2ResolveFacilityCcn(ccn) {
        var raw =
            ccn ||
            global.PROVNUM ||
            (typeof global.PBJ320_EXPORT_CCN !== 'undefined' ? global.PBJ320_EXPORT_CCN : '') ||
            (typeof global.pbjDashboardFacilityCcn === 'function' ? global.pbjDashboardFacilityCcn() : '');
        var prov = String(raw || '').replace(/\D/g, '');
        if (!prov) {
            return '';
        }
        if (prov.length > 6) {
            prov = prov.slice(-6);
        }
        prov = prov.padStart(6, '0');
        return prov === '000000' ? '' : prov;
    }

    function pbjV2LoadChowPanels(ccn) {
        var prov = pbjV2ResolveFacilityCcn(ccn);
        if (!prov) {
            return Promise.resolve();
        }
        var apiBase = typeof global.pbjApiUrl === 'function' ? global.pbjApiUrl : function (p) { return p; };
        return fetch(apiBase('/api/facility-chow/' + prov))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                if (data && data.error) {
                    return;
                }
                global.__pbjLastChowPayload = data;
                pbjV2RenderChowInto('pbjChowOwnershipModalBody', data);
            })
            .catch(function () {
                var body = document.getElementById('pbjChowOwnershipModalBody');
                if (body) {
                    body.innerHTML = '<p class="text-muted mb-0">CHOW index unavailable.</p>';
                }
            });
    }

    function pbjV2ChowTrendShapes() {
        var dates = Array.isArray(global.__pbjChowEffectiveDates) ? global.__pbjChowEffectiveDates : [];
        return dates.map(function (iso) {
            return {
                type: 'line',
                x0: iso,
                x1: iso,
                y0: 0,
                y1: 1,
                xref: 'x',
                yref: 'paper',
                layer: 'below',
                line: { color: 'rgba(111, 66, 193, 0.45)', width: 1, dash: 'dot' }
            };
        });
    }

    function pbjV2AppendChowShapes(layout) {
        if (!layout) {
            return;
        }
        var chow = pbjV2ChowTrendShapes();
        if (!chow.length) {
            return;
        }
        layout.shapes = ([]).concat(layout.shapes || [], chow);
    }

    function pbjV2ResolveAiPackHarringtonRows() {
        if (global.__lastHarringtonRows && global.__lastHarringtonRows.length) {
            return global.__lastHarringtonRows;
        }
        var pack = global.__lastHarringtonCmiExport;
        if (pack && Array.isArray(pack.rows) && pack.rows.length) {
            return pack.rows;
        }
        return [];
    }

    function pbjV2AppendAiPackPeriodStaffingRows(pushRow, out, rows, quarters) {
        rows = rows || [];
        quarters = quarters || [];
        var computed = pbjV2ComputePeriodStaffingFromRows(rows);
        if (computed) {
            [
                ['work_days', computed.workDays, 'days'],
                ['avg_total_nurse_hprd', computed.avgTotalHprd, 'HPRD'],
                ['avg_direct_care_hprd', computed.avgDirectHprd, 'HPRD'],
                ['avg_rn_hprd', computed.avgRnHprd, 'HPRD'],
                ['avg_lpn_hprd', computed.avgLpnHprd, 'HPRD'],
                ['avg_nurse_aide_hprd', computed.avgNaHprd, 'HPRD'],
                ['avg_contract_pct', computed.avgContractPct, 'percent']
            ].forEach(function (pair) {
                if (pair[1] != null && pair[1] !== '') {
                    pushRow(
                        out,
                        'PERIOD_STAFFING',
                        'rollup',
                        '',
                        pair[0],
                        pair[1],
                        pair[2],
                        'Weighted average for active dashboard filter',
                        'PBJ320'
                    );
                }
            });
        }
        var byQuarter = {};
        rows.forEach(function (r) {
            var q =
                (typeof global.pbj320RowCyQuarter === 'function' ? global.pbj320RowCyQuarter(r) : '') ||
                r.CY_Qtr ||
                '';
            var nq =
                typeof global.pbjNormalizeQuarterToCy === 'function'
                    ? global.pbjNormalizeQuarterToCy(q) || q
                    : q;
            if (!nq) {
                return;
            }
            if (!byQuarter[nq]) {
                byQuarter[nq] = { hours: 0, rn: 0, lpn: 0, na: 0, census: 0, days: 0 };
            }
            var bucket = byQuarter[nq];
            bucket.days += 1;
            var census = parseFloat(r.MDScensus);
            if (!isFinite(census) || census <= 0) {
                return;
            }
            bucket.census += census;
            var th = parseFloat(r.Total_Staff_Hours || r.Total_Nurse_Hours || 0);
            if (isFinite(th)) {
                bucket.hours += th;
            }
            var rh = parseFloat(r.Total_RN_Hours || 0);
            if (isFinite(rh)) {
                bucket.rn += rh;
            }
            var lh = parseFloat(r.Total_LPN_Hours || 0);
            if (isFinite(lh)) {
                bucket.lpn += lh;
            }
            var ah = parseFloat(r.Total_Nurse_Aide_Hours || 0);
            if (isFinite(ah)) {
                bucket.na += ah;
            }
        });
        quarters.forEach(function (q) {
            var bucket = byQuarter[q];
            if (!bucket || bucket.census <= 0) {
                return;
            }
            [
                ['facility_total_nurse_hprd', bucket.hours / bucket.census, 'HPRD'],
                ['facility_rn_hprd', bucket.rn / bucket.census, 'HPRD'],
                ['facility_lpn_hprd', bucket.lpn / bucket.census, 'HPRD'],
                ['facility_nurse_aide_hprd', bucket.na / bucket.census, 'HPRD'],
                ['facility_avg_census', bucket.census / bucket.days, 'residents'],
                ['facility_work_days', bucket.days, 'days']
            ].forEach(function (pair) {
                var val = pair[1];
                if (val == null || val === '' || isNaN(Number(val))) {
                    return;
                }
                pushRow(
                    out,
                    'GEO_PEER',
                    'quarter',
                    q,
                    pair[0],
                    typeof val === 'number' ? Number(val).toFixed(3) : val,
                    pair[2],
                    'Facility weighted rollup for peer comparison',
                    'PBJ320'
                );
            });
        });
    }

    function pbjV2AppendAiPackSingleDayRows(pushRow, out, iso) {
        var pack = global.__singleDayReportPayload;
        if (!pack || !pack.payload || !iso) {
            return;
        }
        if (pack.date && pack.date !== iso) {
            return;
        }
        var data = pack.payload;
        var target = data.target_metrics || {};
        var comparisons = data.comparisons || {};
        var aberrations = data.aberrations || {};
        var metricKeys = [
            'census',
            'total_staff_hours',
            'total_staff_hprd',
            'nurse_staff_hours_excl_admin',
            'nurse_staff_hprd_excl_admin',
            'total_rn_hours',
            'total_rn_hprd',
            'rn_hours',
            'rn_hprd',
            'total_lpn_hours',
            'total_lpn_hprd',
            'lpn_hours',
            'lpn_hprd',
            'total_nurse_aide_hours',
            'total_nurse_aide_hprd',
            'cna_hours',
            'cna_hprd',
            'indirect_staffing_hours',
            'indirect_staffing_hprd',
            'rn_contract_pct',
            'lpn_contract_pct',
            'cna_contract_pct'
        ];
        pushRow(
            out,
            'SINGLE_DAY',
            'meta',
            iso,
            'report_quarter',
            target.quarter || '',
            '',
            'PBJ320 single-day report baselines',
            'PBJ320'
        );
        pushRow(out, 'SINGLE_DAY', 'meta', iso, 'report_year', target.year || '', '', '', 'PBJ320');
        pushRow(out, 'SINGLE_DAY', 'meta', iso, 'day_of_week', target.day_of_week || '', '', '', 'PBJ320');
        metricKeys.forEach(function (key) {
            if (target[key] != null && target[key] !== '') {
                pushRow(out, 'SINGLE_DAY', 'day_value', iso, key, target[key], '', 'Work date value', 'PBJ320');
            }
            var cmp = comparisons || {};
            ['quarter', 'year', 'dow'].forEach(function (basis) {
                var bucket = cmp[basis];
                if (bucket && bucket[key] != null && bucket[key] !== '') {
                    pushRow(
                        out,
                        'SINGLE_DAY',
                        'baseline_' + basis,
                        iso,
                        key,
                        bucket[key],
                        '',
                        basis + ' baseline for ' + key,
                        'PBJ320'
                    );
                }
            });
            var ab = aberrations[key];
            if (ab && typeof ab === 'object') {
                ['year', 'quarter', 'dow'].forEach(function (basis) {
                    var z = ab[basis] && ab[basis].z_score;
                    if (z != null && z !== '') {
                        pushRow(
                            out,
                            'SINGLE_DAY',
                            'zscore_' + basis,
                            iso,
                            key,
                            z,
                            'z',
                            basis + ' z-score',
                            'PBJ320'
                        );
                    }
                });
            }
        });
    }

    function pbjV2AppendAiPackOwnershipRows(pushRow, line, out) {
        [
            ['pbjSummaryTotalHprdDisplay', 'summary_total_hprd', 'HPRD'],
            ['pbjSummaryRnHprdDisplay', 'summary_rn_hprd', 'HPRD'],
            ['pbjSummaryNurseAideHprdDisplay', 'summary_nurse_aide_hprd', 'HPRD'],
            ['contractPct', 'summary_contract_pct', 'percent']
        ].forEach(function (pair) {
            var el = document.getElementById(pair[0]);
            var v = el ? String(el.textContent || '').trim() : '';
            if (v && v !== '—' && v !== '-') {
                pushRow(out, 'SUMMARY', 'rollup', '', pair[1], v, pair[2], 'Active filter banner', 'PBJ320');
            }
        });
        var chow = global.__pbjLastChowPayload;
        if (chow && Array.isArray(chow.transactions)) {
            chow.transactions.forEach(function (t, i) {
                pushRow(out, 'CHOW', 'transaction', t.effective_date || '', 'buyer_org', t.buyer_org_name || '', '', '', 'CMS CHOW');
                pushRow(out, 'CHOW', 'transaction', t.effective_date || '', 'seller_org', t.seller_org_name || '', '', '', 'CMS CHOW');
                if (i >= 4) {
                    return;
                }
            });
        }
        var rfRows = typeof global.pbjResolvedRedFlagExportRows === 'function'
            ? global.pbjResolvedRedFlagExportRows().slice(0, 12)
            : [];
        if (!rfRows.length) {
            var flags = Array.isArray(global.__lastRedFlagHistory) ? global.__lastRedFlagHistory : [];
            flags.slice(0, 12).forEach(function (r) {
                var q = r.quarter || r.provider_info_quarter || '';
                var list = Array.isArray(r.red_flags) ? r.red_flags.join('; ') : '';
                if (list) {
                    pushRow(out, 'RED_FLAG', 'quarter', q, 'flags', list, '', '', 'CMS Provider Information');
                }
            });
        } else {
            rfRows.forEach(function (row) {
                var q = row.quarter_cy || row.quarter || '';
                var list = Array.isArray(row.red_flags) ? row.red_flags.join('; ') : '';
                if (list) {
                    pushRow(out, 'RED_FLAG', 'quarter', q, 'flags', list, '', '', 'CMS Provider Information');
                }
            });
        }
        var evs = Array.isArray(global.__pbjFacilityEventsRegistry) ? global.__pbjFacilityEventsRegistry : [];
        evs.slice(0, 20).forEach(function (ev) {
            pushRow(out, 'EVENT', ev.type || 'event', ev.date_iso || '', 'label', ev.label || '', '', ev.detail || '', ev.source || '');
        });
    }

    function pbjV2DownloadAiContextPackJson() {
        if (typeof global.exportPbj320AiContextPack !== 'function') {
            return;
        }
        pbjV2EnsureAiPackDataThen(function () {
            global.__pbjAiPackWantJson = true;
            global.exportPbj320AiContextPack();
            global.__pbjAiPackWantJson = false;
        });
    }

    function pbjV2DownloadAiContextPackCsv() {
        if (typeof global.exportPbj320AiContextPack !== 'function') {
            return;
        }
        pbjV2EnsureAiPackDataThen(function () {
            global.exportPbj320AiContextPack();
        });
    }

    global.pbjV2FormatCyQuarter = pbjV2FormatCyQuarter;
    global.pbjV2DownloadAiContextPackCsv = pbjV2DownloadAiContextPackCsv;
    global.pbjSetActiveWorkDate = pbjSetActiveWorkDate;
    global.pbjV2NormalizeQuarterKey = pbjV2NormalizeQuarterKey;
    global.pbjApplyDashboardQuarter = pbjApplyDashboardQuarter;
    global.pbjBindQuarterDrillClicks = pbjBindQuarterDrillClicks;
    global.pbjInitV2WorkDateBar = pbjInitV2WorkDateBar;
    global.pbjInitScopeChipClick = pbjInitScopeChipClick;
    global.pbjOpenCompareForQuarter = pbjOpenCompareForQuarter;
    global.pbjOpenCompareWindows = pbjOpenCompareWindows;
    global.pbjScrollToCompareWindows = pbjScrollToCompareWindows;
    global.pbjBuildDayEvidenceActionCellHtml = pbjBuildDayEvidenceActionCellHtml;

    global.pbjV2SelectedQuarterKeysFromDom = pbjV2SelectedQuarterKeysFromDom;
    global.pbjV2ScopeLabelFromFilterInfo = pbjV2ScopeLabelFromFilterInfo;
    global.pbjV2ScopeLabelFromDom = pbjV2ScopeLabelFromDom;
    global.pbjV2RefreshScopeLabel = pbjV2RefreshScopeLabel;
    global.pbjV2UpdateScopeLabels = pbjV2UpdateScopeLabels;
    global.pbjV2OpenControlCenter = pbjV2OpenControlCenter;
    global.pbjV2CloseControlCenter = pbjV2CloseControlCenter;
    global.pbjV2RefreshFloatingReportContext = pbjV2RefreshFloatingReportContext;
    global.pbjOpenReportBuilderFromControlCenter = pbjOpenReportBuilderFromControlCenter;

    function pbjV2DismissHowToModal() {
        var modal = document.getElementById('pbjDashboardHowToModal');
        if (!modal || typeof bootstrap === 'undefined' || !bootstrap.Modal) {
            return;
        }
        var inst = bootstrap.Modal.getInstance(modal);
        if (inst) {
            inst.hide();
        }
    }

    function pbjV2DismissGuidedNavOffcanvas() {
        var oc = document.getElementById('guidedNavOffcanvas');
        if (!oc || typeof bootstrap === 'undefined' || !bootstrap.Offcanvas) {
            return;
        }
        var inst = bootstrap.Offcanvas.getInstance(oc);
        if (inst) {
            inst.hide();
        }
    }

    function pbjV2WireHowToModalJumps() {
        var modal = document.getElementById('pbjDashboardHowToModal');
        if (!modal || modal.dataset.pbjHowToBound === '1') {
            return;
        }
        modal.dataset.pbjHowToBound = '1';
        modal.querySelectorAll('.pbj-howto-jump').forEach(function (el) {
            el.addEventListener('click', function (ev) {
                var action = el.getAttribute('data-howto-action');
                var target = el.getAttribute('data-guided-nav-target');
                if (action || target) {
                    ev.preventDefault();
                }
                pbjV2DismissHowToModal();
                if (action === 'reportBuilder') {
                    if (!pbjNavigateToReportBuilder() && typeof global.pbjSwitchTopTab === 'function') {
                        global.pbjSwitchTopTab('reportBuilder');
                    }
                    return;
                }
                if (action === 'openControlCenter') {
                    pbjV2OpenControlCenter({ focusPeriod: true });
                    return;
                }
                if (!target) {
                    return;
                }
                if (global.__pbjReportBuilderViewActive && typeof global.pbjSwitchTopTab === 'function') {
                    global.pbjSwitchTopTab('dashboard');
                }
                if (target === 'pbjCitationsSection' && typeof global.pbjV2RevealInspectionsSection === 'function') {
                    global.pbjV2RevealInspectionsSection({ openFlags: false, scrollTarget: 'pbjCitationsSection' });
                } else if (typeof global.__pbjScrollToGuidedSection === 'function') {
                    global.__pbjScrollToGuidedSection(target);
                }
            });
        });
    }

    function pbjV2WireGuidedNavToolButtons() {
        var ccBtn = document.getElementById('guidedNavOpenControlCenterMobile');
        if (ccBtn && ccBtn.dataset.pbjBound !== '1') {
            ccBtn.dataset.pbjBound = '1';
            ccBtn.addEventListener('click', function () {
                pbjV2DismissGuidedNavOffcanvas();
                pbjV2OpenControlCenter({ focusPeriod: true });
            });
        }
    }
    global.pbjV2RefreshCensusOccupancyTable = pbjV2RefreshCensusOccupancyTable;
    global.pbjV2SyncCensusViewFromHprd = pbjV2SyncCensusViewFromHprd;
    global.pbjV2RefreshCensusContextStrip = pbjV2RefreshCensusContextStrip;
    global.pbjV2RefreshPbjTrendsContextStrips = pbjV2RefreshPbjTrendsContextStrips;
    global.pbjV2RenderCensusContextSparkline = pbjV2RenderCensusContextSparkline;
    global.pbjV2RefreshPlotlyRollupTable = pbjV2RefreshPlotlyRollupTable;
    global.pbjV2RefreshCompositionRollupTable = pbjV2RefreshCompositionRollupTable;
    global.pbjV2RefreshEinHeadcountRollupTable = pbjV2RefreshEinHeadcountRollupTable;
    global.pbjV2SetEinHeadcountRollupTablePlaceholder = pbjV2SetEinHeadcountRollupTablePlaceholder;
    global.pbjV2RefreshRatingsRollupTable = pbjV2RefreshRatingsRollupTable;
    global.pbjV2ExportRollupTableCsv = pbjV2ExportRollupTableCsv;
    global.pbjV2InitRollupTableChrome = pbjV2InitRollupTableChrome;
    global.pbjV2OpenEinHeadcountJobModal = pbjV2OpenEinHeadcountJobModal;
    global.pbjV2RefreshWorkforceRollupTables = pbjV2RefreshWorkforceRollupTables;
    global.pbjV2AssembleStaffingCoreHub = pbjV2AssembleStaffingCoreHub;
    global.pbjV2LoadChowPanel = pbjV2LoadChowPanels;
    global.pbjV2RenderChowPanel = function (p) {
        global.__pbjLastChowPayload = p;
        pbjV2RenderChowInto('pbjChowOwnershipModalBody', p);
    };
    global.pbjV2AppendChowShapes = pbjV2AppendChowShapes;
    global.pbjV2AppendAiPackSingleDayRows = pbjV2AppendAiPackSingleDayRows;
    global.pbjV2AppendAiPackPeriodStaffingRows = pbjV2AppendAiPackPeriodStaffingRows;
    global.pbjV2AppendAiPackContextRows = pbjV2AppendAiPackContextRows;
    global.pbjV2AppendAiPackOwnershipRows = pbjV2AppendAiPackOwnershipRows;
    global.pbjV2AiToolkitApplyScope = pbjV2AiToolkitApplyScope;
    global.pbjV2PullAiScopeFromDashboard = pbjV2PullAiScopeFromDashboard;
    global.pbjV2DownloadAiContextPackJson = pbjV2DownloadAiContextPackJson;
    global.pbjV2FormatIsoShort = pbjV2FormatIsoShort;
    global.pbjV2FormatIsoUsDash = pbjV2FormatIsoUsDash;
    global.pbjV2AiFocusDatesSetFromRows = pbjV2AiFocusDatesSetFromRows;
    global.pbjV2FormatCensusContextPeriodLabel = pbjV2FormatCensusContextPeriodLabel;
    global.pbjV2RenderAiPackPreview = pbjV2RenderAiPackPreview;
    global.pbjV2CopyAiStarterPrompt = pbjV2CopyAiStarterPrompt;
    global.pbjV2DownloadClaudeSkillZip = pbjV2DownloadClaudeSkillZip;
    global.pbjV2CopyAiContextPackCsv = pbjV2CopyAiContextPackCsv;
    global.pbjV2CopyTextToClipboard = pbjV2CopyTextToClipboard;
    global.pbjV2BuildAiStarterPrompt = pbjV2BuildAiStarterPrompt;
    global.pbjV2BuildAiPeriodStaffingSnippet = pbjV2BuildAiPeriodStaffingSnippet;
    global.pbjV2PrefetchAiPackBenchmarks = pbjV2PrefetchAiPackBenchmarks;
    global.pbjV2OnDashboardScopeChanged = pbjV2OnDashboardScopeChanged;
    global.pbjV2UpdateAiToolkitToolHelp = pbjV2UpdateAiToolkitToolHelp;

    /**
     * Build a v2 export button (matches partials/v2/export_macros.html).
     * @param {string} kind csv|pdf|jpg|json
     * @param {{id?, onclick?, title?, ariaLabel?, label?, hint?, primary?, extraClass?}} opts
     */
    function pbjExportBtnHtml(kind, opts) {
        opts = opts || {};
        var k = String(kind || 'csv').toLowerCase();
        var icon = 'fa-download';
        var label = opts.label;
        if (!label) {
            if (k === 'csv') {
                label = '<span class="d-md-none">CSV</span><span class="d-none d-md-inline">Export CSV</span>';
            } else if (k === 'pdf') {
                label = '<span class="d-md-none">PDF</span><span class="d-none d-md-inline">Download PDF</span>';
            } else {
                label = k === 'jpg' ? 'JPG' : k.toUpperCase();
            }
        }
        var title = opts.title || ('Download ' + (opts.label || k.toUpperCase()));
        var aria = opts.ariaLabel || title;
        var hint = opts.hint
            ? ('<span class="pbj-export-btn-hint">' + String(opts.hint).replace(/</g, '&lt;') + '</span>')
            : '';
        var primary = opts.primary
            ? ' btn-primary pbj-export-btn--primary'
            : ' btn-outline-secondary';
        var extra = opts.extraClass ? (' ' + opts.extraClass) : '';
        var idAttr = opts.id ? (' id="' + String(opts.id).replace(/"/g, '&quot;') + '"') : '';
        var clickAttr = opts.onclick ? (' onclick="' + String(opts.onclick).replace(/"/g, '&quot;') + '"') : '';
        return '<button type="button" class="btn btn-sm pbj-export-btn pbj-export-btn--' + k + primary + extra + '"' +
            idAttr + clickAttr + ' title="' + String(title).replace(/"/g, '&quot;') + '" aria-label="' + String(aria).replace(/"/g, '&quot;') + '">' +
            '<i class="fas ' + icon + '" aria-hidden="true"></i>' +
            '<span class="pbj-export-btn-label">' + (opts.label ? String(label).replace(/</g, '&lt;') : label) + '</span>' + hint + '</button>';
    }
    global.pbjExportBtnHtml = pbjExportBtnHtml;

    document.addEventListener('DOMContentLoaded', function () {
        try {
            var rbView = new URLSearchParams(global.location.search).get('view') === 'reportBuilder';
            if (rbView) {
                if (!pbjNavigateToReportBuilder() && typeof global.pbjSwitchTopTab === 'function') {
                    global.pbjSwitchTopTab('reportBuilder');
                }
            }
        } catch (eRbExtrasInit) { /* ignore */ }
        function pbjV2SchedulePlotlyChromeInstall() {
            pbjV2InstallPlotlyMinimalChrome();
            var plotlyPatchAttempts = 0;
            var plotlyPatchTimer = setInterval(function () {
                plotlyPatchAttempts += 1;
                if (pbjV2InstallPlotlyMinimalChrome() || plotlyPatchAttempts > 40) {
                    clearInterval(plotlyPatchTimer);
                }
            }, 100);
        }
        if (typeof global.pbjEnsurePlotly === 'function') {
            global.pbjEnsurePlotly().then(pbjV2SchedulePlotlyChromeInstall).catch(function () {
                pbjV2SchedulePlotlyChromeInstall();
            });
        } else {
            pbjV2SchedulePlotlyChromeInstall();
        }
        pbjV2AssembleStaffingCoreHub();
        pbjV2InitFloatingControls();
        pbjV2CloseControlCenter();
        pbjV2WireHowToModalJumps();
        pbjV2WireGuidedNavToolButtons();
        if (typeof global.pbjWireProfileRatingPopovers === 'function') {
            global.pbjWireProfileRatingPopovers();
        }
        setTimeout(pbjV2EnsureControlsDockVisible, 0);
        setTimeout(pbjV2EnsureControlsDockVisible, 2500);
        pbjV2InitCensusRollupDisclosure();
        pbjV2InitWorkforceRollupDisclosures();
        pbjV2InitRollupTableChrome();
        pbjInitV2WorkDateBar();
        document.addEventListener('pbjQuartersLoaded', function () {
            pbjV2RefreshFloatingPickerOptions();
            pbjV2PullFloatingPeriodFromSummary();
            pbjV2RefreshScopeLabel();
        });
        setTimeout(function () {
            var qSel = document.getElementById('quarterRange');
            if (qSel && qSel.options && qSel.options.length > 1) {
                pbjV2RefreshFloatingPickerOptions();
                pbjV2PullFloatingPeriodFromSummary();
            }
        }, 1200);
        pbjBindQuarterDrillClicks();
        pbjInitScopeChipClick();
        pbjV2RefreshScopeLabel();
        pbjV2LoadChowPanels(
            typeof global.pbjDashboardFacilityCcn === 'function'
                ? global.pbjDashboardFacilityCcn()
                : global.PROVNUM || global.PBJ320_EXPORT_CCN
        );
        pbjV2WireAiToolkitScope();
        pbjV2WireAiFocusDates();
        var aiModal = document.getElementById('pbjAiToolkitModal');
        if (aiModal) {
            aiModal.addEventListener('show.bs.modal', function () {
                pbjV2UpdateAiToolkitToolHelp();
                pbjV2RefreshAiQuarterSelectOptions();
                pbjV2RefreshAiYearSelectOptions();
                pbjV2PullAiScopeFromDashboard();
                var grain = pbjV2GetActiveAiGrain();
                pbjV2SyncAiFocusDatesUiForGrain(grain);
                if (typeof global.pbjRb3SyncFocusDatesToAiToolkit === 'function') {
                    global.pbjRb3SyncFocusDatesToAiToolkit();
                } else {
                    pbjV2AiFocusDatesRender();
                }
                pbjV2RefreshAiToolkitScopeLine();
                pbjV2RefreshAiStarterPromptPreview(global.__pbjLastAiPackMeta || {});
                pbjV2UpdateAiToolkitSummaryIdle();
            });
        }
        var aiRefPanel = document.getElementById('pbjAiToolkitReference');
        if (aiRefPanel) {
            aiRefPanel.addEventListener('shown.bs.collapse', function () {
                pbjV2RefreshAiPackPreview();
            });
        }
        var aiScopeBody = document.getElementById('pbjAiScopeBody');
        if (aiScopeBody) {
            /* panels toggled by grain buttons only */
        }
        document.addEventListener('pbjQuartersLoaded', function () {
            pbjV2RefreshAiQuarterSelectOptions();
            pbjV2RefreshAiYearSelectOptions();
        });
        document.querySelectorAll('input[name="pbjAiToolkitAudience"]').forEach(function (el) {
            el.addEventListener('change', function () {
                pbjV2RefreshAiStarterPromptPreview(global.__pbjLastAiPackMeta || {});
                pbjV2RenderAiPackAdvisories(global.__pbjLastAiPackMeta || {});
            });
        });
        pbjV2UpdateAiToolkitToolHelp();
        var refBtn = document.getElementById('pbjAiToolkitReferenceBtn');
        var refPanel = document.getElementById('pbjAiToolkitReference');
        if (refBtn && refPanel) {
            refPanel.addEventListener('shown.bs.collapse', function () {
                refBtn.setAttribute('aria-expanded', 'true');
            });
            refPanel.addEventListener('hidden.bs.collapse', function () {
                refBtn.setAttribute('aria-expanded', 'false');
            });
        }
        var chowModal = document.getElementById('pbjChowOwnershipModal');
        if (chowModal) {
            chowModal.addEventListener('show.bs.modal', function () {
                if (global.__pbjLastChowPayload) {
                    pbjV2RenderChowInto('pbjChowOwnershipModalBody', global.__pbjLastChowPayload);
                } else {
                    pbjV2LoadChowPanels(
                        typeof global.pbjDashboardFacilityCcn === 'function'
                            ? global.pbjDashboardFacilityCcn()
                            : global.PROVNUM || global.PBJ320_EXPORT_CCN
                    );
                }
            });
        }
    });
})(typeof window !== 'undefined' ? window : globalThis);
