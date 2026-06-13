/**
 * EIN nursing roster — day-key vs employee-row render sequencing (v2).
 * Pure helpers + coordinator for unit tests; dashboard wires these from inline init.
 */
(function (global) {
    'use strict';

    /**
     * Day-grain render must wait while work-day keys are in-flight (null).
     * undefined = no intersection filter; Set = filter ready.
     */
    function pbjEinRosterShouldDeferRender(grain, dayIso, workDayKeys) {
        return String(grain || '') === 'day' && !!dayIso && workDayKeys === null;
    }

    function pbjEinRosterWorkDayKeysFromDayRosterResponse(d) {
        var set = new Set();
        if (d && d.ok && Array.isArray(d.employees) && d.employees.length) {
            d.employees.forEach(function (row) {
                if (!row) {
                    return;
                }
                set.add(String(row.sys_employee_id) + '|' + String(row.job_code));
            });
        }
        return set.size ? set : undefined;
    }

    function pbjEinRosterShouldClearRequest(requestId, activeId) {
        return requestId === activeId;
    }

    /**
     * After employee fetch succeeds — decide render vs defer without painting a stuck loading row.
     */
    function pbjEinRosterPlanEmployeeFetchSuccess(opts) {
        opts = opts || {};
        var loadId = opts.loadId;
        var activeLoadId = opts.activeLoadId;
        if (loadId !== activeLoadId) {
            return { superseded: true, deferRender: false, renderNow: false, clearLoading: false };
        }
        var defer = pbjEinRosterShouldDeferRender(opts.grain, opts.dayIso, opts.workDayKeys);
        return {
            superseded: false,
            deferRender: defer,
            renderNow: !defer,
            clearLoading: !defer,
            setPendingRender: defer
        };
    }

    /**
     * After day-key fetch completes — flush cached employees if ready.
     */
    function pbjEinRosterPlanDayKeyFetchComplete(opts) {
        opts = opts || {};
        if (opts.requestId !== opts.activeRequestId) {
            return { stale: true, flushRender: false, startEmployeeLoad: false };
        }
        var hasCache = !!(opts.employeeCache && opts.employeeCache.length);
        var pending = !!opts.pendingRender;
        return {
            stale: false,
            flushRender: pending || hasCache,
            startEmployeeLoad: true,
            clearPending: pending || hasCache
        };
    }

    /** In-memory coordinator for unit tests (mirrors dashboard sequencing). */
    function PbjEinRosterCoordinator() {
        this.dayReqSeq = 0;
        this.loadSeq = 0;
        this.workDayKeys = undefined;
        this.employeeCache = [];
        this.pendingRender = false;
        this.loading = false;
        this.loadingMessage = '';
        this.renderCount = 0;
        this.grain = 'day';
        this.dayIso = '';
    }

    PbjEinRosterCoordinator.prototype.beginDayKeyFetch = function (dayIso) {
        this.dayReqSeq += 1;
        this.dayIso = String(dayIso || '');
        this.workDayKeys = null;
        this.pendingRender = false;
        this.loading = true;
        this.loadingMessage = 'Matching work day…';
        return this.dayReqSeq;
    };

    PbjEinRosterCoordinator.prototype.failDayKeyFetch = function (requestId) {
        if (requestId !== this.dayReqSeq) {
            return { stale: true };
        }
        this.workDayKeys = undefined;
        var plan = pbjEinRosterPlanDayKeyFetchComplete({
            requestId: requestId,
            activeRequestId: this.dayReqSeq,
            employeeCache: this.employeeCache,
            pendingRender: this.pendingRender
        });
        if (plan.flushRender) {
            this.pendingRender = false;
            this.renderCount += 1;
        }
        this.beginEmployeeLoad();
        return { stale: false, rendered: plan.flushRender };
    };

    PbjEinRosterCoordinator.prototype.completeDayKeyFetch = function (requestId, dayRosterResponse) {
        if (requestId !== this.dayReqSeq) {
            return { stale: true };
        }
        this.workDayKeys = pbjEinRosterWorkDayKeysFromDayRosterResponse(dayRosterResponse);
        var plan = pbjEinRosterPlanDayKeyFetchComplete({
            requestId: requestId,
            activeRequestId: this.dayReqSeq,
            employeeCache: this.employeeCache,
            pendingRender: this.pendingRender
        });
        if (plan.flushRender) {
            this.pendingRender = false;
            this.renderCount += 1;
        }
        this.beginEmployeeLoad();
        return { stale: false, rendered: plan.flushRender };
    };

    PbjEinRosterCoordinator.prototype.beginEmployeeLoad = function () {
        this.loadSeq += 1;
        this.loading = true;
        this.loadingMessage = 'Loading roster…';
        return this.loadSeq;
    };

    PbjEinRosterCoordinator.prototype.completeEmployeeLoad = function (loadId, employees) {
        var plan = pbjEinRosterPlanEmployeeFetchSuccess({
            loadId: loadId,
            activeLoadId: this.loadSeq,
            grain: this.grain,
            dayIso: this.dayIso,
            workDayKeys: this.workDayKeys
        });
        if (plan.superseded) {
            return { superseded: true, rendered: false, loading: this.loading };
        }
        this.employeeCache = employees || [];
        if (plan.deferRender) {
            this.pendingRender = true;
        } else if (plan.renderNow) {
            this.pendingRender = false;
            this.renderCount += 1;
        }
        if (plan.clearLoading) {
            this.loading = false;
            this.loadingMessage = '';
        }
        return {
            superseded: false,
            rendered: plan.renderNow,
            deferred: plan.deferRender,
            loading: this.loading
        };
    };

    global.pbjEinRosterShouldDeferRender = pbjEinRosterShouldDeferRender;
    global.pbjEinRosterWorkDayKeysFromDayRosterResponse = pbjEinRosterWorkDayKeysFromDayRosterResponse;
    global.pbjEinRosterShouldClearRequest = pbjEinRosterShouldClearRequest;
    global.pbjEinRosterPlanEmployeeFetchSuccess = pbjEinRosterPlanEmployeeFetchSuccess;
    global.pbjEinRosterPlanDayKeyFetchComplete = pbjEinRosterPlanDayKeyFetchComplete;
    global.PbjEinRosterCoordinator = PbjEinRosterCoordinator;
})(typeof window !== 'undefined' ? window : globalThis);
