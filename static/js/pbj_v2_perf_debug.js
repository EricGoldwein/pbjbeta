/**
 * PBJ320 v2 dashboard performance instrumentation (no-op unless enabled).
 * Enable: ?debugPerf=1 or window.PBJ320_DEBUG_PERF = true before this script loads.
 */
(function (global) {
    'use strict';

    var PREFIX = 'pbj:';
    var _enabled = null;
    var _fetchPatched = false;

    function parseDebugPerfFromUrl() {
        try {
            var sp = new URLSearchParams(global.location.search);
            var v = sp.get('debugPerf');
            return v === '1' || v === 'true';
        } catch (e) {
            return false;
        }
    }

    function enabled() {
        if (_enabled === null) {
            _enabled = !!(global.PBJ320_DEBUG_PERF || parseDebugPerfFromUrl());
        }
        return _enabled;
    }

    function mark(name) {
        if (!enabled() || !global.performance || !global.performance.mark) {
            return;
        }
        try {
            global.performance.mark(PREFIX + name);
        } catch (e) { /* ignore */ }
    }

    function measure(name, startMark, endMark) {
        if (!enabled() || !global.performance || !global.performance.measure) {
            return null;
        }
        try {
            global.performance.measure(PREFIX + name, PREFIX + startMark, PREFIX + endMark);
            var entries = global.performance.getEntriesByName(PREFIX + name);
            return entries.length ? entries[entries.length - 1].duration : null;
        } catch (e) {
            return null;
        }
    }

    function start(label) {
        mark(label + ':start');
        return label;
    }

    function end(label) {
        mark(label + ':end');
        return measure(label, label + ':start', label + ':end');
    }

    function logGroup(title, rows) {
        if (!enabled() || !global.console) {
            return;
        }
        var fn = global.console.groupCollapsed || global.console.group;
        if (!fn) {
            return;
        }
        fn.call(global.console, '[PBJ perf] ' + title);
        (rows || []).forEach(function (row) {
            if (global.console.log) {
                global.console.log(row);
            }
        });
        if (global.console.groupEnd) {
            global.console.groupEnd();
        }
    }

    function installFetchLogger() {
        if (_fetchPatched || !enabled() || typeof global.fetch !== 'function') {
            return;
        }
        _fetchPatched = true;
        var orig = global.fetch.bind(global);
        global.fetch = function (input, init) {
            var url = typeof input === 'string' ? input : (input && input.url) || '';
            var apiPath = '';
            try {
                var u = new URL(url, global.location.href);
                if (u.pathname.indexOf('/api/') !== -1) {
                    apiPath = u.pathname + (u.search || '');
                }
            } catch (e2) {
                apiPath = String(url).slice(0, 160);
            }
            if (!apiPath || apiPath.indexOf('/api/') === -1) {
                return orig(input, init);
            }
            var t0 = global.performance && global.performance.now ? global.performance.now() : Date.now();
            return orig(input, init).then(function (resp) {
                var t1 = global.performance && global.performance.now ? global.performance.now() : Date.now();
                var size = resp.headers && resp.headers.get ? resp.headers.get('content-length') : null;
                logGroup('fetch ' + apiPath.split('?')[0], [
                    (t1 - t0).toFixed(1) + ' ms',
                    size ? size + ' bytes (content-length)' : 'payload size unknown'
                ]);
                return resp;
            });
        };
    }

    function wrap(label, fn) {
        if (!enabled()) {
            return fn();
        }
        start(label);
        try {
            var result = fn();
            if (result && typeof result.then === 'function') {
                return result
                    .then(function (value) {
                        var ms = end(label);
                        logGroup(label, [ms != null ? ms.toFixed(1) + ' ms' : 'done']);
                        return value;
                    })
                    .catch(function (err) {
                        end(label);
                        throw err;
                    });
            }
            var syncMs = end(label);
            logGroup(label, [syncMs != null ? syncMs.toFixed(1) + ' ms' : 'done']);
            return result;
        } catch (err) {
            end(label);
            throw err;
        }
    }

    function chartRender(chartId, plotFn) {
        if (!enabled()) {
            return plotFn();
        }
        var label = 'chart:' + chartId;
        start(label);
        try {
            var result = plotFn();
            if (result && typeof result.then === 'function') {
                return result.finally(function () {
                    var ms = end(label);
                    logGroup(label, [ms != null ? ms.toFixed(1) + ' ms' : 'done']);
                });
            }
            var ms = end(label);
            logGroup(label, [ms != null ? ms.toFixed(1) + ' ms' : 'done']);
            return result;
        } catch (err) {
            end(label);
            throw err;
        }
    }

    global.pbjPerfEnabled = enabled;
    global.pbjPerfMark = mark;
    global.pbjPerfMeasure = measure;
    global.pbjPerfStart = start;
    global.pbjPerfEnd = end;
    global.pbjPerfLog = logGroup;
    global.pbjPerfWrap = wrap;
    global.pbjPerfChartRender = chartRender;

    if (enabled()) {
        mark('boot:perf-script-loaded');
        installFetchLogger();
    }
})(typeof window !== 'undefined' ? window : global);
