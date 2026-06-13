/**

 * v2 — Group quarterly benchmark charts with their tables (Chart / Table / Both).

 */

(function (global) {

    'use strict';



    var TOPICS = {

        casemix: {

            bodyId: 'v2BenchBodyCaseMix',

            holderId: 'pbjBenchViewHolder_casemix',

            mountId: 'pbjBenchViewMount_casemix',

            chartPanel: 'v2BenchCaseMixChartPanel',

            tablePanel: null,

            chartAnchor: 'pbjBenchViewAnchor_casemix',

            tableAnchor: 'caseMixAcuityRollupSection',

            chartSources: ['providerStaffingBenchmarkChartsSection'],

            tableSources: [],

            rollupCollapseId: 'caseMixAcuityRollupCollapse',

            rollupChartsHostId: 'providerCaseMixModeTabContent',

            defaultView: 'chart',

            plotlyIds: [

                'providerInfoTotalChart', 'providerInfoRNChart', 'providerInfoLPNChart', 'providerInfoCNAChart',

                'providerInfoCombinedDeltaChart', 'providerInfoRNDeltaChart', 'providerInfoLPNDeltaChart', 'providerInfoNADeltaChart'

            ]

        },

        harrington: {

            bodyId: 'v2BenchBodyHarrington',

            holderId: 'pbjBenchViewHolder_harrington',

            mountId: 'pbjBenchViewMount_harrington',

            chartPanel: 'v2BenchHarringtonChartPanel',

            tablePanel: 'v2BenchHarringtonTablePanel',

            chartAnchor: '',

            tableAnchor: 'harringtonExpectedRollupSection',

            chartSources: [],

            tableSources: ['v2BenchHarringtonTableHost'],

            rollupCollapseId: 'harringtonExpectedRollupCollapse',

            defaultView: 'table',

            plotlyIds: []

        }

    };



    function pbjBenchGetBootstrapCollapse(el) {

        if (!el || typeof global.bootstrap === 'undefined' || !global.bootstrap.Collapse) {

            return null;

        }

        return global.bootstrap.Collapse.getOrCreateInstance(el, { toggle: false });

    }



    function pbjBenchHarringtonRollupOpen() {

        var collapseEl = document.getElementById('harringtonExpectedRollupCollapse');

        return !!(collapseEl && collapseEl.classList.contains('show'));

    }



    function pbjBenchRequestHarringtonRefresh(delayMs) {

        var baseWait = typeof delayMs === 'number' ? delayMs : 0;

        function attempt(n) {

            global.setTimeout(function () {

                if (typeof global.pbjPrefetchHarringtonCmiExport === 'function') {

                    global.pbjPrefetchHarringtonCmiExport();

                }

                var body = document.getElementById('harringtonCMITableBody');

                var head = document.getElementById('harringtonCMITableHead');

                if (typeof global.updateHarringtonCMI === 'function' && body && head) {

                    if (!pbjBenchHarringtonRollupOpen()) {

                        return;

                    }

                    global.updateHarringtonCMI();

                    return;

                }

                if (n < 10) {

                    attempt(n + 1);

                }

            }, baseWait + (n ? 100 : 0));

        }

        attempt(0);

    }



    function pbjBenchSyncHarringtonRollup() {
        /* Harrington rollup stays collapsed until the user expands it. */
    }



    function pbjBenchWireHarringtonRollup() {

        var cfg = TOPICS.harrington;

        var collapseEl = cfg.rollupCollapseId ? document.getElementById(cfg.rollupCollapseId) : null;

        if (!collapseEl || collapseEl.dataset.pbjBenchHarringtonBound === '1') {

            return;

        }

        collapseEl.dataset.pbjBenchHarringtonBound = '1';

        collapseEl.addEventListener('shown.bs.collapse', function () {

            pbjBenchRequestHarringtonRefresh(40);

        });

    }



    function pbjBenchSyncCaseMixRollup(view) {

        var cfg = TOPICS.casemix;

        var collapseEl = cfg.rollupCollapseId ? document.getElementById(cfg.rollupCollapseId) : null;

        var chartsHost = cfg.rollupChartsHostId ? document.getElementById(cfg.rollupChartsHostId) : null;

        var rollupBtn = document.getElementById('caseMixAcuityRollupBtn');

        if (!collapseEl) {

            return;

        }

        var bsCollapse = pbjBenchGetBootstrapCollapse(collapseEl);

        if (chartsHost) {

            chartsHost.classList.remove('d-none');

            chartsHost.classList.toggle('pbj-case-mix-charts-host--hidden', view === 'table');

        }

        if (view === 'table') {

            if (bsCollapse && !collapseEl.classList.contains('show')) {

                bsCollapse.show();

            } else if (!bsCollapse) {

                collapseEl.classList.add('show');

                if (rollupBtn) {

                    rollupBtn.classList.remove('collapsed');

                    rollupBtn.setAttribute('aria-expanded', 'true');

                }

            }

            global.requestAnimationFrame(function () {

                var section = document.getElementById('caseMixSection');

                if (section && typeof section.scrollIntoView === 'function') {

                    section.scrollIntoView({ block: 'nearest', behavior: 'smooth' });

                }

            });

            return;

        }

        if (view === 'chart') {

            if (bsCollapse && collapseEl.classList.contains('show')) {

                bsCollapse.hide();

            } else if (!bsCollapse) {

                collapseEl.classList.remove('show');

                if (rollupBtn) {

                    rollupBtn.classList.add('collapsed');

                    rollupBtn.setAttribute('aria-expanded', 'false');

                }

            }

        }

    }



    function pbjBenchResizePlotly(ids) {

        (ids || []).forEach(function (id) {

            var el = document.getElementById(id);

            if (!el) {

                return;

            }

            var compLayouts = global.__pbjCompositionChartLayouts || {};

            var layoutSnap =

                (id === 'compositionTotalTrendChart' && compLayouts.total) ||

                (id === 'compositionDirectTrendChart' && compLayouts.direct) ||

                (global.__pbjTrendChartLayouts && global.__pbjTrendChartLayouts[id]) ||

                null;

            if (layoutSnap && typeof global.pbjPlotlyResizeChartPreserveLegend === 'function') {

                try {

                    global.pbjPlotlyResizeChartPreserveLegend(el, layoutSnap);

                    return;

                } catch (ePreserve) { /* fall through */ }

            }

            if (typeof global.Plotly === 'undefined' || !global.Plotly.Plots || !global.Plotly.Plots.resize) {

                return;

            }

            try {

                global.Plotly.Plots.resize(el);

            } catch (e) { /* ignore */ }

        });

    }



    function pbjBenchSyncViewRadio(topic, view) {

        var chrome = pbjBenchGetChrome(topic);

        var scope = chrome || document;

        scope.querySelectorAll('input[name="pbjBenchView_' + topic + '"]').forEach(function (radio) {

            radio.checked = radio.value === view;

        });

    }



    function pbjBenchGetView(topic) {

        var cfg = TOPICS[topic];

        if (!cfg) {

            return 'table';

        }

        if (!cfg.chartSources.length) {

            return 'table';

        }

        var hasTable = cfg.tableSources.length > 0 || !!cfg.rollupCollapseId || !!cfg.tablePanel;

        if (!hasTable) {

            return 'chart';

        }

        var chrome = pbjBenchGetChrome(topic);

        var picked = chrome

            ? chrome.querySelector('input[name="pbjBenchView_' + topic + '"]:checked')

            : null;

        if (!picked) {

            picked = document.querySelector('input[name="pbjBenchView_' + topic + '"]:checked');

        }

        var v = picked ? picked.value : cfg.defaultView;

        if (v !== 'chart' && v !== 'table' && v !== 'both') {

            return cfg.defaultView;

        }

        return v;

    }



    function pbjBenchActiveTopic() {

        var active = document.querySelector('#v2BenchmarkTopicTabs .nav-link.active');

        if (!active) {

            return 'casemix';

        }

        var target = active.getAttribute('data-bs-target') || '';

        if (target.indexOf('Harrington') >= 0) {

            return 'harrington';

        }

        return 'casemix';

    }



    function pbjBenchSyncMountVisibility(activeTopic) {

        Object.keys(TOPICS).forEach(function (topic) {

            var mount = document.getElementById(TOPICS[topic].mountId);

            if (mount) {

                mount.classList.toggle('d-none', topic !== activeTopic);

            }

        });

    }



    function pbjBenchGetChrome(topic) {

        var cfg = TOPICS[topic];

        if (!cfg) {

            return null;

        }

        var scoped = document.querySelector('.pbj-bench-view-chrome[data-bench-toolbar="' + topic + '"]');

        if (scoped) {

            return scoped;

        }

        var holder = cfg.holderId ? document.getElementById(cfg.holderId) : null;

        if (!holder) {

            return null;

        }

        return holder.querySelector('.pbj-bench-view-chrome');

    }



    function pbjBenchPlaceViewChrome(topic) {

        var cfg = TOPICS[topic];

        var chrome = pbjBenchGetChrome(topic);

        if (!cfg || !chrome) {

            return;

        }

        var mount = cfg.mountId ? document.getElementById(cfg.mountId) : null;

        if (mount) {

            mount.appendChild(chrome);

            return;

        }

        var view = pbjBenchGetView(topic);

        var anchorId = view === 'table' ? cfg.tableAnchor : cfg.chartAnchor;

        if (!anchorId && cfg.tableAnchor) {

            anchorId = cfg.tableAnchor;

        }

        var anchor = anchorId ? document.getElementById(anchorId) : null;

        if (!anchor) {

            var panel = document.getElementById(view === 'table' ? cfg.tablePanel : cfg.chartPanel);

            if (panel) {

                anchor = panel;

            }

        }

        if (anchor) {

            anchor.appendChild(chrome);

        }

    }



    function pbjBenchApplyView(topic) {

        var cfg = TOPICS[topic];

        if (!cfg) {

            return;

        }

        var body = document.getElementById(cfg.bodyId);

        var chartPanel = document.getElementById(cfg.chartPanel);

        var tablePanel = cfg.tablePanel ? document.getElementById(cfg.tablePanel) : null;

        if (!body || (!chartPanel && !tablePanel)) {

            return;

        }

        var view = pbjBenchGetView(topic);

        var hasChart = cfg.chartSources.length > 0;

        var hasTable = cfg.tableSources.length > 0 || !!cfg.rollupCollapseId || !!cfg.tablePanel;

        if (!hasChart) {

            view = 'table';

        } else if (!hasTable) {

            view = 'chart';

        } else if (!chartPanel && tablePanel) {

            view = 'table';

        }



        pbjBenchSyncViewRadio(topic, view);



        body.classList.remove('pbj-bench-view--chart', 'pbj-bench-view--table', 'pbj-bench-view--both');

        body.classList.add('pbj-bench-view--' + view);



        pbjBenchSyncMountVisibility(topic);

        pbjBenchPlaceViewChrome(topic);



        if (topic === 'casemix') {

            chartPanel.classList.remove('d-none');

            pbjBenchSyncCaseMixRollup(view);

            if (view !== 'table') {

                pbjV2RefreshCaseMixChartLayout();

            }

        } else if (topic === 'harrington') {

            if (tablePanel) {

                tablePanel.classList.toggle('d-none', view === 'chart');

            }

            if (chartPanel) {

                chartPanel.classList.toggle('d-none', view === 'table');

            }

            if (view !== 'chart') {

                pbjBenchSyncHarringtonRollup();

                pbjBenchRequestHarringtonRefresh(60);

            }

        } else {

            if (tablePanel) {

                tablePanel.classList.toggle('d-none', view === 'chart');

            }

            if (chartPanel) {

                chartPanel.classList.toggle('d-none', view === 'table');

            }

        }



        if (view !== 'table' && hasChart) {

            global.requestAnimationFrame(function () {

                setTimeout(function () {

                    pbjBenchResizePlotly(cfg.plotlyIds);

                }, 60);

            });

            setTimeout(function () {

                pbjBenchResizePlotly(cfg.plotlyIds);

            }, 280);

        }

    }



    function pbjBenchMountSources(panelId, sourceIds) {

        if (!panelId) {

            return;

        }

        var panel = document.getElementById(panelId);

        if (!panel) {

            return;

        }

        sourceIds.forEach(function (sid) {

            var node = document.getElementById(sid);

            if (node && !panel.contains(node)) {

                node.removeAttribute('hidden');

                node.classList.remove('d-none');

                panel.appendChild(node);

            }

        });

    }



    function pbjBenchWireViewRadios(topic) {

        var chrome = pbjBenchGetChrome(topic);

        if (!chrome) {

            return;

        }

        chrome.querySelectorAll('.pbj-bench-view-radio').forEach(function (radio) {

            if (radio.dataset.pbjBenchWired === '1') {

                return;

            }

            radio.dataset.pbjBenchWired = '1';

            radio.addEventListener('change', function () {

                pbjBenchApplyView(topic);

            });

        });

    }



    function pbjBenchInitTopic(topic) {

        var cfg = TOPICS[topic];

        if (!cfg) {

            return;

        }

        pbjBenchMountSources(cfg.chartPanel, cfg.chartSources);

        if (cfg.tablePanel) {

            pbjBenchMountSources(cfg.tablePanel, cfg.tableSources);

        }

        pbjBenchSyncViewRadio(topic, cfg.defaultView);

        pbjBenchWireViewRadios(topic);

        pbjBenchApplyView(topic);

        if (topic === 'harrington') {

            pbjBenchWireHarringtonRollup();

            pbjBenchSyncHarringtonRollup();

            pbjBenchRequestHarringtonRefresh(80);

        }

    }



    function pbjV2RefreshCaseMixChartLayout() {

        var cfg = TOPICS.casemix;

        var body = document.getElementById(cfg.bodyId);

        var chartsHost = cfg.rollupChartsHostId ? document.getElementById(cfg.rollupChartsHostId) : null;

        if (!body || !chartsHost) {

            return;

        }

        var view = pbjBenchGetView('casemix');

        chartsHost.classList.remove('d-none');

        chartsHost.classList.toggle('pbj-case-mix-charts-host--hidden', view === 'table');

        if (view === 'table') {

            return;

        }

        global.requestAnimationFrame(function () {

            setTimeout(function () {

                pbjBenchResizePlotly(cfg.plotlyIds);

            }, 40);

        });

        setTimeout(function () {

            pbjBenchResizePlotly(cfg.plotlyIds);

        }, 220);

    }



    function pbjBenchWireTopicTabs() {

        var tabList = document.getElementById('v2BenchmarkTopicTabs');

        if (!tabList) {

            return;

        }

        tabList.querySelectorAll('[data-bs-toggle="tab"]').forEach(function (tab) {

            tab.addEventListener('shown.bs.tab', function (ev) {

                var target = ev.target && ev.target.getAttribute('data-bs-target');

                if (target === '#v2BenchPaneCaseMix') {

                    pbjBenchApplyView('casemix');

                } else if (target === '#v2BenchPaneHarrington') {

                    pbjBenchApplyView('harrington');

                    pbjBenchSyncHarringtonRollup();

                    pbjBenchRequestHarringtonRefresh(60);

                }

            });

        });

        pbjBenchSyncMountVisibility(pbjBenchActiveTopic());

    }



    function pbjV2AssembleBenchmarkWorkspace() {

        Object.keys(TOPICS).forEach(pbjBenchInitTopic);

        pbjBenchWireTopicTabs();

    }



    function pbjBenchRunWhenReady() {

        if (document.getElementById('v2BenchmarkTopicTabs')) {

            pbjV2AssembleBenchmarkWorkspace();

        }

    }



    global.pbjV2AssembleBenchmarkWorkspace = pbjV2AssembleBenchmarkWorkspace;

    global.pbjBenchApplyView = pbjBenchApplyView;

    global.pbjBenchResizePlotly = pbjBenchResizePlotly;

    global.pbjV2RefreshCaseMixChartLayout = pbjV2RefreshCaseMixChartLayout;



    if (document.readyState === 'loading') {

        document.addEventListener('DOMContentLoaded', pbjBenchRunWhenReady);

    } else {

        pbjBenchRunWhenReady();

    }

    global.addEventListener('load', function () {

        pbjBenchRunWhenReady();

        ['casemix', 'harrington'].forEach(pbjBenchApplyView);

        pbjBenchWireHarringtonRollup();

        pbjBenchSyncHarringtonRollup();

    });

    global.pbjScheduleUpdateHarringtonCMI = pbjBenchRequestHarringtonRefresh;

})(typeof window !== 'undefined' ? window : globalThis);


