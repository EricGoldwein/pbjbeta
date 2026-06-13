/**
 * Daily PBJ staffing proxy flags — shared logic for the daily table and summaries.
 * Federal proxies (RN < 8 h, licensed nurse < 24 h) are not definitive noncompliance.
 */
(function (global) {
    'use strict';

    function parseNum(v) {
        if (v == null || v === '') {
            return null;
        }
        var n = parseFloat(String(v).replace(/,/g, '').trim());
        return Number.isFinite(n) ? n : null;
    }

    function dailyCensus(row) {
        if (!row) {
            return null;
        }
        var n = parseNum(row.MDScensus != null ? row.MDScensus : row.mdscensus);
        return n != null && n > 0 ? n : null;
    }

    function dailyRnHours(row) {
        if (!row) {
            return null;
        }
        return parseNum(row.Total_RN_Hours != null ? row.Total_RN_Hours : row.total_rn_hours);
    }

    function dailyLpnHours(row) {
        if (!row) {
            return null;
        }
        return parseNum(row.Total_LPN_Hours != null ? row.Total_LPN_Hours : row.total_lpn_hours);
    }

    function dailyLicensedNurseHours(row) {
        var rnH = dailyRnHours(row);
        var lpn = dailyLpnHours(row);
        if (rnH == null && lpn == null) {
            return null;
        }
        return (rnH != null ? rnH : 0) + (lpn != null ? lpn : 0);
    }

    function dailyTotalHprd(row) {
        if (!row) {
            return null;
        }
        var v =
            row.Total_Staff_HPRD != null
                ? row.Total_Staff_HPRD
                : row.total_staff_hprd != null
                  ? row.total_staff_hprd
                  : row.Total_Nurse_HPRD;
        return parseNum(v);
    }

    function dailyDirectHprd(row) {
        if (!row) {
            return null;
        }
        var v =
            row.Nurse_Staff_HPRD_Excl_Admin != null
                ? row.Nurse_Staff_HPRD_Excl_Admin
                : row.nurse_staff_hprd_excl_admin != null
                  ? row.nurse_staff_hprd_excl_admin
                  : row.Direct_Care_HPRD != null
                    ? row.Direct_Care_HPRD
                    : row.direct_care_hprd;
        return parseNum(v);
    }

    /**
     * @param {Object} row PBJ daily row
     * @param {Object} [ctx]
     * @param {string} [ctx.stateLabel]
     * @param {'total'|'direct_care'|'none'|''} [ctx.hprdBasis]
     * @param {number|null} [ctx.stateThreshold]
     * @returns {string[]}
     */
    function getReasons(row, ctx) {
        ctx = ctx || {};
        if (dailyCensus(row) == null) {
            return [];
        }
        var reasons = [];
        var basis = String(ctx.hprdBasis || '').trim();
        var thr = parseNum(ctx.stateThreshold);
        if ((basis === 'total' || basis === 'direct_care') && thr != null && thr > 0) {
            var hprd = basis === 'direct_care' ? dailyDirectHprd(row) : dailyTotalHprd(row);
            if (hprd != null && hprd < thr) {
                var stateLabel = String(ctx.stateLabel || 'State').trim() || 'State';
                reasons.push(stateLabel + ' staffing benchmark');
            }
        }
        var rnH = dailyRnHours(row);
        if (rnH != null && rnH < 8) {
            reasons.push('Total RN < 8 hours');
        }
        var lnH = dailyLicensedNurseHours(row);
        if (lnH != null && lnH < 24) {
            reasons.push('Licensed nurse < 24 hours');
        }
        return reasons;
    }

    function isFlagged(row, ctx) {
        return getReasons(row, ctx).length > 0;
    }

    function tooltipText(reasons) {
        var list = Array.isArray(reasons) ? reasons : [];
        var base =
            'Possible staffing flag: reported below one or more selected state/federal proxy thresholds.';
        if (!list.length) {
            return base;
        }
        return base + ' ' + list.join(' · ');
    }

    global.pbjDailyStaffingFlags = {
        dailyCensus: dailyCensus,
        dailyRnHours: dailyRnHours,
        dailyLpnHours: dailyLpnHours,
        dailyLicensedNurseHours: dailyLicensedNurseHours,
        dailyTotalHprd: dailyTotalHprd,
        dailyDirectHprd: dailyDirectHprd,
        getReasons: getReasons,
        isFlagged: isFlagged,
        tooltipText: tooltipText,
    };
})(typeof window !== 'undefined' ? window : globalThis);
