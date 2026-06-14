/**
 * PBJ320 Premium Dashboard V3 — multi-pane shell (Overview, Benchmarks, Workforce & Daily, Risk Timeline).
 * Virtual panes: sections stay in DOM order; visibility toggled by data-pbj-v3-pane.
 */
(function (global) {
    'use strict';

    var PANE_ORDER = ['overview', 'benchmarks', 'workforce', 'risk'];

    /** Always visible under the pane anchor (not toggled per pane). */
    var PANE_ALWAYS_TOP_IDS = ['facility-header'];

    var PANE_SECTIONS = {
        overview: ['pbjStaffingCoreSection'],
        benchmarks: ['staffingBenchmarkingSection'],
        workforce: ['staffingPatternsWorkforceSection', 'dayLevelEvidenceSection'],
        risk: ['pbjCitationsSection'],
    };

    /** Legacy guided-nav scroll targets → V3 pane id */
    var LEGACY_NAV_TO_PANE = {
        pbjStaffingCoreSection: 'overview',
        complianceReviewSection: 'overview',
        hprdTrendSection: 'overview',
        staffingBenchmarkingSection: 'benchmarks',
        staffingPatternsWorkforceSection: 'workforce',
        dayLevelEvidenceSection: 'workforce',
        dailyStaffingEinSection: 'workforce',
        einNursingEmployeeSection: 'workforce',
        dailyDateReviewSection: 'workforce',
        pbjCitationsSection: 'risk',
        riskScreeningSection: 'risk',
    };

    var LOADED_KEY_PREFIX = '__pbjV3PaneLoaded_';

    function v3Enabled() {
        return !!global.__pbjV3PanesActive;
    }

    function currentPane() {
        return global.__pbjV3ActivePane || 'overview';
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

    function markPaneLoaded(pane) {
        global[LOADED_KEY_PREFIX + pane] = paneScopeKey();
    }

    function paneLoadedForCurrentScope(pane) {
        return global[LOADED_KEY_PREFIX + pane] === paneScopeKey();
    }

    function invalidatePaneLoads() {
        PANE_ORDER.forEach(function (p) {
            delete global[LOADED_KEY_PREFIX + p];
        });
        delete global.__pbjV3WorkforcePaneOpened;
        delete global.__pbjV3WorkforceLazyWired;
        delete global.__pbjDeferredDailyTableBundle;
        if (global.__pbjLazyLoaded_einHeadcountChart && typeof global.pbjInvalidateEinHeadcountCacheIfScopeChanged === 'function') {
            global.pbjInvalidateEinHeadcountCacheIfScopeChanged();
        }
    }

    function sectionsForPane(pane) {
        return (PANE_SECTIONS[pane] || []).slice();
    }

    function allManagedSectionIds() {
        var seen = {};
        var out = [];
        PANE_ORDER.forEach(function (p) {
            sectionsForPane(p).forEach(function (id) {
                if (!seen[id]) {
                    seen[id] = true;
                    out.push(id);
                }
            });
        });
        return out;
    }

    function setSectionVisible(sectionId, visible) {
        var el = document.getElementById(sectionId);
        if (!el) {
            return;
        }
        el.classList.toggle('d-none', !visible);
        el.classList.toggle('pbj-v3-pane-hidden', !visible);
        if (visible) {
            el.removeAttribute('aria-hidden');
        } else {
            el.setAttribute('aria-hidden', 'true');
        }
    }

    function ensureAlwaysTopSections(root) {
        if (!root) {
            return null;
        }
        var anchor = document.getElementById('pbjV3PaneOverview');
        var insertRef = anchor || root.firstChild;
        PANE_ALWAYS_TOP_IDS.forEach(function (id) {
            var el = document.getElementById(id);
            if (!el || el.parentNode !== root) {
                return;
            }
            el.classList.remove('d-none', 'pbj-v3-pane-hidden');
            el.removeAttribute('aria-hidden');
            if (insertRef.nextSibling !== el) {
                root.insertBefore(el, insertRef.nextSibling);
            }
            insertRef = el;
        });
        return insertRef;
    }

    function reorderActivePaneSections(active) {
        var root = document.getElementById('pbjDashboardPane');
        if (!root || !v3Enabled()) {
            return;
        }
        var insertRef = ensureAlwaysTopSections(root) || root.firstChild;
        var activeIds = sectionsForPane(active);
        var hiddenIds = allManagedSectionIds().filter(function (id) {
            return activeIds.indexOf(id) < 0;
        });
        activeIds.forEach(function (id) {
            var el = document.getElementById(id);
            if (!el || el.parentNode !== root) {
                return;
            }
            if (insertRef.nextSibling !== el) {
                root.insertBefore(el, insertRef.nextSibling);
            }
            insertRef = el;
        });
        hiddenIds.forEach(function (id) {
            var el = document.getElementById(id);
            if (el && el.parentNode === root) {
                root.appendChild(el);
            }
        });
    }

    function applyPaneVisibility(pane) {
        if (!v3Enabled()) {
            return;
        }
        var active = pane || currentPane();
        var show = {};
        sectionsForPane(active).forEach(function (id) {
            show[id] = true;
        });
        allManagedSectionIds().forEach(function (id) {
            setSectionVisible(id, !!show[id]);
        });
        reorderActivePaneSections(active);
        document.body.classList.toggle('pbj-v3-pane--overview', active === 'overview');
        document.body.classList.toggle('pbj-v3-pane--benchmarks', active === 'benchmarks');
        document.body.classList.toggle('pbj-v3-pane--workforce', active === 'workforce');
        document.body.classList.toggle('pbj-v3-pane--risk', active === 'risk');
    }

    function resizePaneCharts() {
        if (typeof global.Plotly === 'undefined' || !Plotly.Plots || !Plotly.Plots.resize) {
            return;
        }
        requestAnimationFrame(function () {
            document.querySelectorAll('.js-plotly-plot').forEach(function (plotEl) {
                try {
                    Plotly.Plots.resize(plotEl);
                } catch (e) {
                    /* ignore */
                }
            });
        });
    }

    function loadOverviewPane() {
        if (typeof global.pbjScheduleProviderInfoHeavyLoads === 'function') {
            global.pbjScheduleProviderInfoHeavyLoads();
        }
        if (!global.__pbjLazyLoaded_sffHistory && typeof global.loadSFFHistory === 'function') {
            global.loadSFFHistory();
            global.__pbjLazyLoaded_sffHistory = true;
        }
        markPaneLoaded('overview');
        if (typeof global.pbjV3UpdateOverviewHandoff === 'function') {
            global.pbjV3UpdateOverviewHandoff();
        }
    }

    function loadBenchmarksPane() {
        if (paneLoadedForCurrentScope('benchmarks')) {
            return;
        }
        if (typeof global.loadCaseMixData === 'function') {
            global.loadCaseMixData();
        }
        if (typeof global.pbjScheduleProviderInfoHeavyLoads === 'function') {
            global.pbjScheduleProviderInfoHeavyLoads();
        }
        if (typeof global.pbjMaybeRefreshGeoRollupWhenVisible === 'function') {
            global.pbjMaybeRefreshGeoRollupWhenVisible();
        }
        if (typeof global.pbjV2AssembleBenchmarkWorkspace === 'function') {
            global.pbjV2AssembleBenchmarkWorkspace();
        }
        if (typeof global.pbjBenchApplyView === 'function') {
            global.pbjBenchApplyView('casemix');
        }
        if (typeof global.pbjV2RefreshCaseMixChartLayout === 'function') {
            setTimeout(function () {
                global.pbjV2RefreshCaseMixChartLayout();
            }, 80);
        }
        markPaneLoaded('benchmarks');
        if (typeof global.pbjV3UpdateBenchmarksHandoff === 'function') {
            global.pbjV3UpdateBenchmarksHandoff();
        }
    }

    function loadWorkforcePane() {
        global.__pbjV3WorkforcePaneOpened = true;
        if (global.__pbjDeferredDailyTableBundle && typeof global.updateTables === 'function') {
            global.updateTables(global.__pbjDeferredDailyTableBundle);
            delete global.__pbjDeferredDailyTableBundle;
        }
        if (typeof global.pbjV3WireWorkforceLazyLoads === 'function') {
            global.pbjV3WireWorkforceLazyLoads(true);
        }
        if (typeof global.initEinPbjBridgeUI === 'function') {
            global.initEinPbjBridgeUI();
        }
        if (typeof global.pbjLoadSupplementalStaffingPanels === 'function') {
            global.pbjLoadSupplementalStaffingPanels();
        }
        if (paneLoadedForCurrentScope('workforce')) {
            resizePaneCharts();
            return;
        }
        /* EIN headcount: IO may have wired while hidden — paint after pane is visible. */
        requestAnimationFrame(function () {
            requestAnimationFrame(function () {
                var hcHost = document.getElementById('einHeadcountByJobChart');
                if (!hcHost || hcHost.querySelector('.js-plotly-plot')) {
                    resizePaneCharts();
                    return;
                }
                if (typeof global.pbjInitEinHeadcountByJobChart === 'function') {
                    global.pbjInitEinHeadcountByJobChart();
                }
                if (typeof global.pbjLoadEinHeadcountByJobChart === 'function') {
                    global.pbjLoadEinHeadcountByJobChart({});
                }
                resizePaneCharts();
            });
        });
        markPaneLoaded('workforce');
        if (typeof global.pbjV3UpdateWorkforceHandoff === 'function') {
            global.pbjV3UpdateWorkforceHandoff();
        }
    }

    function loadRiskPane() {
        if (typeof global.pbjLoadCitationsPanelOnDemand === 'function') {
            global.pbjLoadCitationsPanelOnDemand();
        }
        if (typeof global.pbjV2RevealInspectionsSection === 'function') {
            global.pbjV2RevealInspectionsSection({
                openFlags: false,
                scrollTarget: 'pbjV3PaneRisk',
                skipScroll: true,
            });
        }
        if (!global.__pbjLazyLoaded_sffHistory && typeof global.loadSFFHistory === 'function') {
            global.loadSFFHistory();
            global.__pbjLazyLoaded_sffHistory = true;
        }
        if (typeof global.pbjV2RefreshFacilityEventsTimeline === 'function') {
            global.pbjV2RefreshFacilityEventsTimeline();
        }
        if (typeof global.pbjV3RiskTimelineInit === 'function') {
            global.pbjV3RiskTimelineInit();
        } else if (typeof global.pbjV3RiskTimelineRefresh === 'function') {
            global.pbjV3RiskTimelineRefresh();
        }
        markPaneLoaded('risk');
        if (typeof global.pbjV3UpdateRiskHandoff === 'function') {
            global.pbjV3UpdateRiskHandoff();
        }
    }

    function lazyLoadPane(pane) {
        switch (pane) {
            case 'overview':
                loadOverviewPane();
                break;
            case 'benchmarks':
                loadBenchmarksPane();
                break;
            case 'workforce':
                loadWorkforcePane();
                break;
            case 'risk':
                loadRiskPane();
                break;
            default:
                break;
        }
    }

    function setActiveNav(pane) {
        var navTargets = {
            overview: 'pbjV3PaneOverview',
            benchmarks: 'pbjV3PaneBenchmarks',
            workforce: 'pbjV3PaneWorkforce',
            risk: 'pbjV3PaneRisk',
        };
        var target = navTargets[pane] || 'pbjV3PaneOverview';
        document.querySelectorAll('[data-guided-nav-target]').forEach(function (a) {
            if (a.id === 'guidedNavBrand') {
                return;
            }
            var on = a.getAttribute('data-guided-nav-target') === target;
            a.classList.toggle('active', on);
            a.setAttribute('aria-current', on ? 'page' : 'false');
        });
        if (typeof global.pbjSyncReportBuilderNavPills === 'function') {
            global.pbjSyncReportBuilderNavPills();
        }
    }

    function switchPane(pane, opts) {
        opts = opts || {};
        if (!v3Enabled()) {
            return;
        }
        if (PANE_ORDER.indexOf(pane) < 0) {
            pane = 'overview';
        }
        if (global.__pbjReportBuilderViewActive && typeof global.pbjSwitchTopTab === 'function') {
            global.pbjSwitchTopTab('dashboard');
        }
        var applySwitch = function () {
            document.body.classList.remove('pbj-v3-pane-switching');
            global.__pbjV3ActivePane = pane;
            applyPaneVisibility(pane);
            setActiveNav(pane);
            if (!opts.skipLazy) {
                lazyLoadPane(pane);
            }
            window.scrollTo(0, 0);
            if (typeof global.__pbjUpdateGuidedScrollMargin === 'function') {
                global.__pbjUpdateGuidedScrollMargin();
            }
            requestAnimationFrame(function () {
                window.scrollTo(0, 0);
                document.body.classList.remove('pbj-v3-pane-switching');
            });
        };
        var canAnimate =
            !opts.skipTransition &&
            global.__pbjV3PaneTransitionReady &&
            pane !== currentPane();
        if (canAnimate) {
            document.body.classList.add('pbj-v3-pane-switching');
            setTimeout(applySwitch, 160);
            return;
        }
        document.body.classList.remove('pbj-v3-pane-switching');
        applySwitch();
    }

    function resolvePaneFromLegacyTarget(targetId) {
        if (LEGACY_NAV_TO_PANE[targetId]) {
            return LEGACY_NAV_TO_PANE[targetId];
        }
        return null;
    }

    function dismissGuidedNavOffcanvas() {
        if (global.matchMedia && global.matchMedia('(max-width: 767.98px)').matches) {
            var oc = document.getElementById('guidedNavOffcanvas');
            if (oc && typeof bootstrap !== 'undefined' && bootstrap.Offcanvas) {
                var inst = bootstrap.Offcanvas.getInstance(oc);
                if (inst) {
                    inst.hide();
                }
            }
        }
    }

    function scrollToGuidedSection(sectionId) {
        if (typeof global.__pbjScrollToGuidedSection === 'function') {
            global.__pbjScrollToGuidedSection(sectionId);
            return;
        }
        var el = document.getElementById(sectionId);
        if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }

    function wireGuidedNavV3() {
        document.querySelectorAll('[data-guided-nav-target^="pbjV3Pane"]').forEach(function (a) {
            a.addEventListener('click', function (ev) {
                ev.preventDefault();
                var pane = (a.getAttribute('data-pbj-v3-pane') || '').trim();
                if (!pane) {
                    var tid = a.getAttribute('data-guided-nav-target') || '';
                    if (tid === 'pbjV3PaneOverview') pane = 'overview';
                    else if (tid === 'pbjV3PaneBenchmarks') pane = 'benchmarks';
                    else if (tid === 'pbjV3PaneWorkforce') pane = 'workforce';
                    else if (tid === 'pbjV3PaneRisk') pane = 'risk';
                }
                switchPane(pane);
                dismissGuidedNavOffcanvas();
            });
        });
        document.querySelectorAll('[data-guided-nav-target="complianceReviewSection"]').forEach(function (a) {
            a.addEventListener('click', function (ev) {
                ev.preventDefault();
                switchPane('overview', { skipLazy: true });
                requestAnimationFrame(function () {
                    scrollToGuidedSection('complianceReviewSection');
                });
                dismissGuidedNavOffcanvas();
            });
        });
        var brand = document.getElementById('guidedNavBrand');
        if (brand) {
            brand.addEventListener('click', function (ev) {
                if (!v3Enabled()) {
                    return;
                }
                ev.preventDefault();
                switchPane('overview');
            });
        }
    }

    function init() {
        if (!v3Enabled()) {
            return;
        }
        document.documentElement.classList.remove('pbj-v3-panes-pending');
        document.body.classList.add('pbj-v3-panes-active');
        global.__pbjV3ActivePane = 'overview';
        applyPaneVisibility('overview');
        setActiveNav('overview');
        wireGuidedNavV3();
        loadOverviewPane();
        var dashPane = document.getElementById('pbjDashboardPane');
        if (dashPane) {
            dashPane.classList.add('pbj-v3-ready');
        }
        global.__pbjV3PaneTransitionReady = true;
    }

    function onPeriodChanged() {
        if (!v3Enabled()) {
            return;
        }
        invalidatePaneLoads();
        var pane = currentPane();
        if (pane !== 'overview') {
            lazyLoadPane(pane);
        }
        if (pane === 'risk' && typeof global.pbjV3RiskTimelineRefresh === 'function') {
            global.pbjV3RiskTimelineRefresh();
        }
        if (pane !== 'workforce' && global.__pbjDeferredDailyTableBundle && typeof global.updateTables === 'function') {
            /* keep deferred until workforce opens */
        }
        if (typeof global.pbjV3RenderHandoffCards === 'function') {
            global.pbjV3RenderHandoffCards();
        }
    }

    function shouldDeferDailyTable() {
        return v3Enabled() && !global.__pbjV3WorkforcePaneOpened;
    }

    /** Re-apply V3 pane visibility after legacy code removes d-none on managed sections. */
    function reapplyPaneVisibility() {
        if (!v3Enabled()) {
            return;
        }
        applyPaneVisibility(currentPane());
    }

    global.pbjV3SwitchPane = switchPane;
    global.pbjV3ResolvePaneFromLegacyTarget = resolvePaneFromLegacyTarget;
    global.pbjV3OnPeriodChanged = onPeriodChanged;
    global.pbjV3ShouldDeferDailyTable = shouldDeferDailyTable;
    global.pbjV3InvalidatePaneLoads = invalidatePaneLoads;
    global.pbjV3ReapplyPaneVisibility = reapplyPaneVisibility;
    global.pbjV3InitPanes = init;
})(typeof window !== 'undefined' ? window : globalThis);
