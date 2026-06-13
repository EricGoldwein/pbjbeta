/**
 * Canonical Provider Information red-flag resolver by CMS reporting quarter.
 * Re-hydrates Admin TO from __pbjAdminTurnoverReviews when flat history was stripped.
 */
(function (global) {
    'use strict';

    var STATUS = {
        FLAGS: 'provider_info_flags',
        NO_FLAGS: 'provider_info_no_flags',
        UNAVAILABLE: 'provider_info_unavailable',
        INVALID: 'invalid_quarter',
    };

    function normalizeQuarterToCy(qRaw) {
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

    function quarterSortKey(cy) {
        var m = String(cy || '').match(/^(\d{4})Q([1-4])$/);
        if (!m) {
            return null;
        }
        return parseInt(m[1], 10) * 10 + parseInt(m[2], 10);
    }

    function quarterDisplayLabel(cy) {
        if (typeof global.harringtonCyQuarterDisplay === 'function') {
            return global.harringtonCyQuarterDisplay(cy);
        }
        var m = String(cy || '').match(/^(\d{4})Q([1-4])$/);
        return m ? 'Q' + m[2] + ' ' + m[1] : String(cy || '');
    }

    function resolveCaseMixForQuarter(cy) {
        if (typeof global.pbj320ResolveCaseMixDataForQuarter === 'function') {
            return global.pbj320ResolveCaseMixDataForQuarter(cy);
        }
        var map =
            typeof global.__pbjCaseMixDataByQuarter === 'object' && global.__pbjCaseMixDataByQuarter
                ? global.__pbjCaseMixDataByQuarter
                : null;
        if (!map) {
            return null;
        }
        var target = normalizeQuarterToCy(cy);
        if (!target) {
            return null;
        }
        var keys = Object.keys(map);
        for (var i = 0; i < keys.length; i++) {
            if (normalizeQuarterToCy(keys[i]) === target) {
                return { key: keys[i], data: map[keys[i]] };
            }
        }
        return null;
    }

    function findInFlatHistory(target) {
        var hist = global.__lastRedFlagHistory;
        if (!Array.isArray(hist) || !hist.length) {
            return null;
        }
        for (var i = 0; i < hist.length; i++) {
            var rec = hist[i];
            if (normalizeQuarterToCy(rec.quarter || rec.qtr || '') === target) {
                return rec;
            }
        }
        return null;
    }

    function rehydrateFromAdminReviews(target) {
        var reviews = global.__pbjAdminTurnoverReviews;
        if (!Array.isArray(reviews) || !reviews.length) {
            return null;
        }
        for (var j = 0; j < reviews.length; j++) {
            var ep = reviews[j];
            var quarters = ep.affected_quarters || [];
            for (var k = 0; k < quarters.length; k++) {
                if (normalizeQuarterToCy(quarters[k]) !== target) {
                    continue;
                }
                var count = ep.turnover_count;
                var flag = 'Admin TO';
                if (count != null && !isNaN(parseInt(count, 10)) && parseInt(count, 10) > 0) {
                    flag = 'Admin TO: ' + parseInt(count, 10);
                }
                return {
                    quarter: quarters[k],
                    red_flags: [flag],
                    processing_date: ep.last_observed_date || ep.first_observed_date || '',
                    source_file: 'Provider Information',
                    _fromAdminTurnoverReview: true,
                    _episode_id: ep.episode_id || '',
                };
            }
        }
        return null;
    }

    function buildResult(status, target, record, source) {
        var flags = record && Array.isArray(record.red_flags) ? record.red_flags.slice() : [];
        return {
            status: status,
            quarter: target,
            quarter_display: (record && record.quarter) || quarterDisplayLabel(target),
            red_flags: flags,
            record: record,
            source: source || null,
            has_provider_info: status === STATUS.FLAGS || status === STATUS.NO_FLAGS,
        };
    }

    /**
     * @param {string} cyQuarter - 2025Q4 or Q4 2025
     * @returns {{status:string, quarter:string, quarter_display:string, red_flags:string[], record:object|null, source:string|null, has_provider_info:boolean}}
     */
    function resolveRedFlagsForQuarter(cyQuarter) {
        var target = normalizeQuarterToCy(cyQuarter) || String(cyQuarter || '').trim();
        if (!target || quarterSortKey(target) == null) {
            return buildResult(STATUS.INVALID, '', null, null);
        }
        var flat = findInFlatHistory(target);
        if (flat) {
            var flatFlags = Array.isArray(flat.red_flags) ? flat.red_flags : [];
            return buildResult(
                flatFlags.length ? STATUS.FLAGS : STATUS.NO_FLAGS,
                target,
                flat,
                'flat_history'
            );
        }
        var reviewRec = rehydrateFromAdminReviews(target);
        if (reviewRec) {
            return buildResult(STATUS.FLAGS, target, reviewRec, 'admin_turnover_review');
        }
        var cm = resolveCaseMixForQuarter(target);
        if (cm && cm.data) {
            return buildResult(STATUS.NO_FLAGS, target, {
                quarter: quarterDisplayLabel(target),
                red_flags: [],
            }, 'case_mix');
        }
        return buildResult(STATUS.UNAVAILABLE, target, null, null);
    }

    /** True when flag-history UI has rows (flat history and/or grouped admin-turnover reviews). */
    function pbjRedFlagSectionHasContent() {
        var hist = global.__lastRedFlagHistory;
        if (Array.isArray(hist) && hist.length) {
            return true;
        }
        var reviews = global.__pbjAdminTurnoverReviews;
        return Array.isArray(reviews) && reviews.length > 0;
    }

    /** True when any red-flag signal exists (flat history, admin reviews, or case-mix PI). */
    function pbjRedFlagHistoryHasAnySignal() {
        if (pbjRedFlagSectionHasContent()) {
            return true;
        }
        var cm = global.__pbjCaseMixDataByQuarter;
        return !!(cm && typeof cm === 'object' && Object.keys(cm).length);
    }

    /** Unique normalized quarters from flat history + admin review episodes + optional case-mix. */
    function pbjCollectKnownRedFlagQuarters(includeCaseMix) {
        var seen = {};
        var out = [];
        function add(qRaw) {
            var cy = normalizeQuarterToCy(qRaw);
            if (!cy || seen[cy]) {
                return;
            }
            seen[cy] = true;
            out.push(cy);
        }
        (global.__lastRedFlagHistory || []).forEach(function (r) {
            add(r && (r.quarter || r.qtr));
        });
        (global.__pbjAdminTurnoverReviews || []).forEach(function (ep) {
            (ep.affected_quarters || []).forEach(add);
        });
        if (includeCaseMix) {
            var cm = global.__pbjCaseMixDataByQuarter || {};
            Object.keys(cm).forEach(add);
        }
        out.sort(function (a, b) {
            var ka = quarterSortKey(a);
            var kb = quarterSortKey(b);
            if (ka == null && kb == null) {
                return 0;
            }
            if (ka == null) {
                return 1;
            }
            if (kb == null) {
                return -1;
            }
            return ka - kb;
        });
        return out;
    }

    /**
     * Export rows with red flags for AI pack / CSV (flat + rehydrated admin TO).
     * @param {string[]} [filterQuarters] - optional CY quarters; empty = all known quarters
     */
    function pbjResolvedRedFlagExportRows(filterQuarters) {
        var quarters = Array.isArray(filterQuarters) && filterQuarters.length
            ? filterQuarters.map(normalizeQuarterToCy).filter(Boolean)
            : pbjCollectKnownRedFlagQuarters(false);
        var rows = [];
        var seen = {};
        quarters.forEach(function (cy) {
            if (!cy || seen[cy]) {
                return;
            }
            var res = resolveRedFlagsForQuarter(cy);
            if (res.status !== STATUS.FLAGS || !res.red_flags.length) {
                return;
            }
            seen[cy] = true;
            rows.push({
                quarter: res.quarter_display || cy,
                quarter_cy: cy,
                red_flags: res.red_flags.slice(),
                record: res.record,
            });
        });
        return rows;
    }

    /**
     * Earliest calendar quarter (Date) with a flag matching matchRe (flat + rehydrated).
     */
    function pbjRegexFromMatch(matchRe) {
        if (matchRe && typeof matchRe === 'object' && typeof matchRe.source === 'string' && typeof matchRe.test === 'function') {
            var flags = String(matchRe.flags || 'i').replace(/g/g, '');
            return new RegExp(matchRe.source, flags || 'i');
        }
        return new RegExp(String(matchRe || ''), 'i');
    }

    function pbjFindEarliestResolvedRedFlagQuarter(matchRe) {
        var re = pbjRegexFromMatch(matchRe);
        var best = null;
        var flagStatus = STATUS.FLAGS;
        var quarters = pbjCollectKnownRedFlagQuarters(false);
        for (var qi = 0; qi < quarters.length; qi++) {
            var cy = quarters[qi];
            var res = resolveRedFlagsForQuarter(cy);
            if (!res || res.status !== flagStatus) {
                continue;
            }
            var hit = false;
            var flags = res.red_flags || [];
            for (var fi = 0; fi < flags.length; fi++) {
                if (re.test(String(flags[fi] || ''))) {
                    hit = true;
                    break;
                }
            }
            if (!hit) {
                continue;
            }
            var m = String(cy).match(/^(\d{4})Q([1-4])$/);
            if (!m) {
                continue;
            }
            var y = parseInt(m[1], 10);
            var qn = parseInt(m[2], 10);
            var d = new Date(y, (qn - 1) * 3, 1);
            if (isNaN(d.getTime())) {
                continue;
            }
            if (!best || d.getTime() < best.date.getTime()) {
                best = { quarter: cy, date: d, record: res.record };
            }
        }
        return best;
    }

    global.PBJ_RED_FLAG_STATUS = STATUS;
    global.resolveRedFlagsForQuarter = resolveRedFlagsForQuarter;
    global.pbjRedFlagSectionHasContent = pbjRedFlagSectionHasContent;
    global.pbjRedFlagHistoryHasAnySignal = pbjRedFlagHistoryHasAnySignal;
    global.pbjCollectKnownRedFlagQuarters = pbjCollectKnownRedFlagQuarters;
    global.pbjResolvedRedFlagExportRows = pbjResolvedRedFlagExportRows;
    global.pbjFindEarliestResolvedRedFlagQuarter = pbjFindEarliestResolvedRedFlagQuarter;
})(typeof window !== 'undefined' ? window : this);
