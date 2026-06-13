/**
 * Sustained reported-work pattern flags — shared narrative, labels, and display helpers.
 * Used by EIN roster badges, employee modal alerts, and forensics panels.
 */
(function (global) {
    'use strict';

    var LEGACY_NARRATIVE_MARKERS = [
        'One or more reported employee IDs',
        'concentrated among CNA records'
    ];

    var SUSTAINED_WORK_DISCLAIMER_PREFIX = 'PBJ data alone cannot determine';

    function pbjSustainedWorkDisclaimerText() {
        return (
            'PBJ data alone cannot determine whether this reflects actual scheduling practices, reporting conventions, ' +
            'identifier assignment methods, or data quality issues. Review source rows before interpreting this as an ' +
            "individual worker's schedule."
        );
    }

    function pbjStripSustainedWorkDisclaimer(text) {
        var t = String(text || '').trim();
        if (!t) {
            return '';
        }
        var idx = t.indexOf(SUSTAINED_WORK_DISCLAIMER_PREFIX);
        if (idx < 0) {
            return t;
        }
        return t.slice(0, idx).replace(/\s+$/, '').trim();
    }

    function escHtml(s) {
        if (typeof global.einEscHtml === 'function') {
            return global.einEscHtml(s);
        }
        if (typeof global.escapeHtml === 'function') {
            return global.escapeHtml(s);
        }
        return String(s || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;');
    }

    function continuityChip(label, value) {
        if (typeof global.pbjEinContinuityChip === 'function') {
            return global.pbjEinContinuityChip(label, value);
        }
        return (
            '<span class="pbj-ein-continuity-chip">' +
            '<span class="pbj-ein-continuity-chip-label">' + escHtml(label) + '</span> ' +
            value +
            '</span>'
        );
    }

    function formatContinuityNum(n) {
        if (typeof global.pbjFormatEinContinuityNum === 'function') {
            return global.pbjFormatEinContinuityNum(n);
        }
        return n == null || n === '' ? '—' : String(n);
    }

    function formatQuarterLabel(q) {
        if (typeof global.formatCyQuarterLabel === 'function') {
            return global.formatCyQuarterLabel(q);
        }
        return String(q || '—');
    }

    function pbjSustainedWorkReviewPriorityLabel(severity) {
        var key = String(severity || '').toLowerCase();
        var labels = { advisory: 'Advisory', review: 'Review', major: 'Major', extreme: 'Extreme' };
        return labels[key] || (severity ? String(severity) : '—');
    }

    function pbjFormatSustainedWorkMetric(value, decimals) {
        decimals = decimals == null ? 0 : Number(decimals);
        var n = Number(value);
        if (!isFinite(n)) {
            return '0';
        }
        if (decimals <= 0) {
            return String(Math.round(n));
        }
        var text = n.toFixed(decimals);
        if (text.indexOf('.') >= 0) {
            text = text.replace(/\.?0+$/, '');
        }
        return text;
    }

    function pbjSustainedWorkFlagLooksLegacy(text) {
        var t = String(text || '');
        for (var i = 0; i < LEGACY_NARRATIVE_MARKERS.length; i++) {
            if (t.indexOf(LEGACY_NARRATIVE_MARKERS[i]) >= 0) {
                return true;
            }
        }
        return false;
    }

    function pbjCrossRoleNarrativeSentence(groups) {
        var labels = [];
        if (Array.isArray(groups)) {
            groups.forEach(function (g) {
                var s = String(g || '').trim();
                if (s) {
                    labels.push(s);
                }
            });
        }
        labels.sort();
        if (labels.length) {
            return (
                'The same reported ID also appears in multiple role groups (' +
                labels.join(', ') +
                ') on the same day in this quarter.'
            );
        }
        return 'The same reported ID also appears in multiple role groups on the same day in this quarter.';
    }

    function pbjBuildSustainedWorkFlagNarrative(flag, options) {
        options = options || {};
        var includeDisclaimer = options.includeDisclaimer !== false;
        if (!flag) {
            return '';
        }
        if (String(flag.narrative || '').trim() && !pbjSustainedWorkFlagLooksLegacy(flag.narrative)) {
            var prebuilt = String(flag.narrative).trim();
            if (includeDisclaimer) {
                return prebuilt;
            }
            var strippedNarr = pbjStripSustainedWorkDisclaimer(prebuilt);
            if (strippedNarr && !/[.!?]$/.test(strippedNarr)) {
                strippedNarr += '.';
            }
            return strippedNarr;
        }
        if (String(flag.interpretation || '').trim() && !pbjSustainedWorkFlagLooksLegacy(flag.interpretation)) {
            var interp = String(flag.interpretation).trim();
            if (includeDisclaimer) {
                return interp;
            }
            var strippedInterp = pbjStripSustainedWorkDisclaimer(interp);
            if (strippedInterp && !/[.!?]$/.test(strippedInterp)) {
                strippedInterp += '.';
            }
            return strippedInterp;
        }
        var rg = String(flag.role_group || 'role group').trim() || 'role group';
        var windowDays = Number(flag.window_days) || 90;
        var hours = Number(flag.rolling_90_hours) || 0;
        var workedDays = Number(flag.rolling_90_worked_days) || 0;
        var parts = [
            'This reported employee ID accounts for an unusually high volume of ' + rg + ' hours: ' +
            pbjFormatSustainedWorkMetric(hours) + ' hours across ' + workedDays + ' worked days in a ' +
            windowDays + '-day window.'
        ];
        var maxDaily = Number(flag.max_daily_hours) || 0;
        if (maxDaily >= 14) {
            parts.push('The maximum reported day was ' + pbjFormatSustainedWorkMetric(maxDaily, 1) + ' hours.');
        }
        var streak4 = Number(flag.max_consecutive_shift_equivalent_days) || 0;
        var streak7 = Number(flag.max_consecutive_full_shift_days) || 0;
        var has4 = streak4 >= 21;
        var has7 = streak7 >= 14;
        if (has4 && has7) {
            parts.push(
                'It also shows sustained multi-day work activity, including ' + streak4 +
                ' consecutive days with at least 4 reported hours and ' + streak7 +
                ' consecutive days with at least 7 reported hours.'
            );
        } else if (has4) {
            parts.push(
                'It also shows sustained multi-day work activity, including ' + streak4 +
                ' consecutive days with at least 4 reported hours.'
            );
        } else if (has7) {
            parts.push(
                'It also shows sustained multi-day work activity, including ' + streak7 +
                ' consecutive days with at least 7 reported hours.'
            );
        }
        var share = Number(flag.share_of_role_hours);
        if (isFinite(share) && share >= 0.02) {
            var shareLine =
                'This ID accounts for ' + pbjFormatSustainedWorkMetric(share * 100, 1) +
                '% of ' + rg + ' hours in the quarter';
            var distinctRg = Number(flag.distinct_role_employees_in_period);
            if (isFinite(distinctRg) && distinctRg > 0) {
                shareLine += ' (among ' + distinctRg + ' ' + (distinctRg === 1 ? rg : rg + 's') + ' with reported hours)';
            }
            parts.push(shareLine + '.');
        }
        var crossRole = flag.appears_across_role_groups === true;
        var crossEmpCtr = flag.appears_across_employee_contract === true;
        if (crossRole) {
            parts.push(pbjCrossRoleNarrativeSentence(flag.cross_role_groups));
        }
        if (crossEmpCtr) {
            parts.push('The same reported ID also appears under both employee and contract reporting.');
        }
        if (includeDisclaimer) {
            parts.push(pbjSustainedWorkDisclaimerText());
        }
        return parts.join(' ');
    }

    function pbjNormalizeSustainedWorkFlag(flag) {
        if (!flag || typeof flag !== 'object') {
            return flag;
        }
        var narrative = pbjBuildSustainedWorkFlagNarrative(flag);
        flag.narrative = narrative;
        flag.interpretation = narrative;
        return flag;
    }

    function pbjFormatSustainedWorkPeriodRange(startIso, endIso) {
        var fmt =
            typeof global.pbjFormatIsoWorkDateUsDashed === 'function'
                ? global.pbjFormatIsoWorkDateUsDashed
                : function (iso) {
                      return String(iso || '—');
                  };
        var start = String(startIso || '').trim();
        var end = String(endIso || '').trim();
        if (start && end) {
            return fmt(start) + ' – ' + fmt(end);
        }
        return fmt(start || end || '—');
    }

    function pbjSustainedWorkLimitationsForDisplay(limitations) {
        return (Array.isArray(limitations) ? limitations : []).filter(function (lim) {
            var s = String(lim || '').trim();
            if (!s) {
                return false;
            }
            if (/WORK_HRS_FN/i.test(s)) {
                return false;
            }
            if (/Same reported ID appears in multiple nursing role groups/i.test(s)) {
                return false;
            }
            if (/fractional-hour flags omitted/i.test(s)) {
                return false;
            }
            if (/fractional-hour checks are skipped/i.test(s)) {
                return false;
            }
            return true;
        });
    }

    function pbjSwMetricCell(label, valueHtml) {
        return (
            '<div class="pbj-sw-metric">' +
            '<span class="pbj-sw-metric-label">' + escHtml(label) + '</span>' +
            '<span class="pbj-sw-metric-value">' + valueHtml + '</span>' +
            '</div>'
        );
    }

    function pbjSustainedWorkEmployeeId(flag) {
        if (!flag) {
            return '';
        }
        var raw = flag.employee_id != null ? flag.employee_id : flag.sys_employee_id;
        if (raw != null && String(raw).trim()) {
            return String(raw).trim();
        }
        return '';
    }

    function pbjSustainedWorkEmployeeIdLinkHtml(flag) {
        var eid = pbjSustainedWorkEmployeeId(flag);
        if (!eid) {
            return escHtml(String(flag.masked_employee_id || '—'));
        }
        var canOpen = typeof global.openEinNursingEmployeeModal === 'function';
        if (!canOpen) {
            return '<span class="font-monospace">' + escHtml(eid) + '</span>';
        }
        return (
            '<a href="#" class="pbj-sw-employee-link font-monospace" data-pbj-sw-open-employee="1" ' +
            'data-pbj-sw-employee-id="' + escHtml(eid) + '" ' +
            'data-pbj-sw-job-code="' + escHtml(String(flag.job_code != null ? flag.job_code : '')) + '" ' +
            'data-pbj-sw-quarter="' + escHtml(String(flag.quarter || '')) + '">' +
            escHtml(eid) +
            '</a>'
        );
    }

    function pbjSustainedWorkModalLeadHtml(flag) {
        if (!flag) {
            return '';
        }
        var rg = String(flag.role_group || 'role group').trim() || 'role group';
        var windowDays = Number(flag.window_days) || 90;
        var hours = Number(flag.rolling_90_hours) || 0;
        var workedDays = Number(flag.rolling_90_worked_days) || 0;
        var parts = [
            'Employee ID ' + pbjSustainedWorkEmployeeIdLinkHtml(flag) +
                ' accounts for an unusually high volume of ' + escHtml(rg) + ' hours: ' +
                escHtml(pbjFormatSustainedWorkMetric(hours)) + ' hours across ' + workedDays +
                ' worked days in a ' + windowDays + '-day window.'
        ];
        var maxDaily = Number(flag.max_daily_hours) || 0;
        if (maxDaily >= 14) {
            parts.push(
                'The maximum reported day was ' + escHtml(pbjFormatSustainedWorkMetric(maxDaily, 1)) + ' hours.'
            );
        }
        var streak4 = Number(flag.max_consecutive_shift_equivalent_days) || 0;
        var streak7 = Number(flag.max_consecutive_full_shift_days) || 0;
        var has4 = streak4 >= 21;
        var has7 = streak7 >= 14;
        if (has4 && has7) {
            parts.push(
                'It also shows sustained multi-day work activity, including ' + streak4 +
                ' consecutive days with at least 4 reported hours and ' + streak7 +
                ' consecutive days with at least 7 reported hours.'
            );
        } else if (has4) {
            parts.push(
                'It also shows sustained multi-day work activity, including ' + streak4 +
                ' consecutive days with at least 4 reported hours.'
            );
        } else if (has7) {
            parts.push(
                'It also shows sustained multi-day work activity, including ' + streak7 +
                ' consecutive days with at least 7 reported hours.'
            );
        }
        var share = Number(flag.share_of_role_hours);
        if (isFinite(share) && share >= 0.02) {
            var shareLine =
                'This ID accounts for ' + escHtml(pbjFormatSustainedWorkMetric(share * 100, 1)) +
                '% of ' + escHtml(rg) + ' hours in the quarter';
            var distinctRg = Number(flag.distinct_role_employees_in_period);
            if (isFinite(distinctRg) && distinctRg > 0) {
                shareLine += ' (among ' + distinctRg + ' ' + (distinctRg === 1 ? escHtml(rg) : escHtml(rg) + 's') + ' with reported hours)';
            }
            parts.push(shareLine + '.');
        }
        if (flag.appears_across_role_groups === true) {
            parts.push(escHtml(pbjCrossRoleNarrativeSentence(flag.cross_role_groups)));
        }
        if (flag.appears_across_employee_contract === true) {
            parts.push('The same reported ID also appears under both employee and contract reporting.');
        }
        return '<p class="pbj-sw-modal-lead">' + parts.join(' ') + '</p>';
    }

    function pbjWireSustainedWorkModalEmployeeLinks(root) {
        (root || document).querySelectorAll('[data-pbj-sw-open-employee]').forEach(function (link) {
            if (link.dataset.pbjSwEmployeeBound === '1') {
                return;
            }
            link.dataset.pbjSwEmployeeBound = '1';
            link.addEventListener('click', function (ev) {
                ev.preventDefault();
                ev.stopPropagation();
                var sid = link.getAttribute('data-pbj-sw-employee-id') || '';
                var jcid = link.getAttribute('data-pbj-sw-job-code') || '';
                var qtr = link.getAttribute('data-pbj-sw-quarter') || '';
                var swModal = document.getElementById('einSustainedWorkFlagModal');
                if (swModal && typeof bootstrap !== 'undefined' && bootstrap.Modal) {
                    var swInst = bootstrap.Modal.getInstance(swModal);
                    if (swInst) {
                        swInst.hide();
                    }
                }
                if (typeof global.openEinNursingEmployeeModal === 'function') {
                    global.openEinNursingEmployeeModal(sid, jcid, qtr);
                }
            });
        });
    }

    function pbjSwSeverityBadgeClass(severity) {
        var key = String(severity || '').toLowerCase();
        if (key === 'major' || key === 'extreme') {
            return 'pbj-sw-priority--high';
        }
        if (key === 'advisory') {
            return 'pbj-sw-priority--low';
        }
        return 'pbj-sw-priority--mid';
    }

    /**
     * Polished sustained-work detail layout for the dedicated review modal.
     */
    function pbjSustainedWorkFlagModalBodyHtml(flag, options) {
        options = options || {};
        if (!flag) {
            return '';
        }
        var sharePct =
            flag.share_of_role_hours != null && !isNaN(Number(flag.share_of_role_hours))
                ? pbjFormatSustainedWorkMetric(Number(flag.share_of_role_hours) * 100, 1) + '%'
                : '—';
        var priorityLabel = pbjSustainedWorkReviewPriorityLabel(flag.severity);
        var priorityHtml =
            '<span class="pbj-sw-priority-badge ' + pbjSwSeverityBadgeClass(flag.severity) + '">' +
            escHtml(priorityLabel) +
            '</span>';
        var timingLine = '';
        if (flag.near_employee_id_continuity_anomaly) {
            timingLine =
                '<div class="pbj-sw-modal-note">' +
                '<span class="pbj-sw-modal-note-label">Headcount context</span>' +
                '<p class="mb-0">' +
                escHtml('Timing vs employee-ID continuity anomaly: ' + String(flag.timing_vs_id_anomaly || 'unrelated')) +
                '</p></div>';
        }
        var limitations = pbjSustainedWorkLimitationsForDisplay(flag.limitations);
        var limitationsHtml = '';
        if (limitations.length) {
            limitationsHtml =
                '<div class="pbj-sw-modal-note pbj-sw-modal-note--muted">' +
                '<span class="pbj-sw-modal-note-label">Limitations</span>' +
                '<p class="mb-0">' + escHtml(limitations.join(' ')) + '</p></div>';
        }
        var html = '<div class="pbj-sw-modal">';
        html += pbjSustainedWorkModalLeadHtml(flag);
        html += '<div class="pbj-sw-metrics-grid">';
        html += pbjSwMetricCell('Quarter', escHtml(formatQuarterLabel(flag.quarter || '')));
        html += pbjSwMetricCell('Review priority', priorityHtml);
        html += pbjSwMetricCell('Role group', escHtml(String(flag.role_group || '—')));
        html += pbjSwMetricCell(
            'Period',
            escHtml(pbjFormatSustainedWorkPeriodRange(flag.period_start, flag.period_end))
        );
        html += pbjSwMetricCell('Worked days', escHtml(formatContinuityNum(flag.rolling_90_worked_days)));
        html += pbjSwMetricCell(
            'Hours in window',
            escHtml(flag.rolling_90_hours != null ? pbjFormatSustainedWorkMetric(flag.rolling_90_hours) : '—')
        );
        html += pbjSwMetricCell(
            'Max daily hrs',
            escHtml(flag.max_daily_hours != null ? pbjFormatSustainedWorkMetric(flag.max_daily_hours, 1) : '—')
        );
        html += pbjSwMetricCell('≥4h streak', escHtml(formatContinuityNum(flag.max_consecutive_shift_equivalent_days)));
        html += pbjSwMetricCell('≥7h streak', escHtml(formatContinuityNum(flag.max_consecutive_full_shift_days)));
        html += pbjSwMetricCell('Share of role hours', escHtml(sharePct));
        html += '</div>';
        html += timingLine + limitationsHtml;
        html +=
            '<p class="pbj-sw-modal-disclaimer">' + escHtml(pbjSustainedWorkDisclaimerText()) + '</p>';
        html += '</div>';
        return html;
    }

    function pbjSustainedWorkFlagChipsHtml(flag, options) {
        options = options || {};
        var sharePct =
            flag.share_of_role_hours != null && !isNaN(Number(flag.share_of_role_hours))
                ? pbjFormatSustainedWorkMetric(Number(flag.share_of_role_hours) * 100, 1) + '%'
                : '—';
        var qLabel = formatQuarterLabel(flag.quarter || '');
        var chipClass = options.chipClass || 'pbj-ein-continuity-chips mb-2';
        var html = '<div class="' + chipClass + '">';
        if (options.showQuarter) {
            html += continuityChip('Quarter', escHtml(qLabel));
        }
        if (options.showReviewPriority !== false) {
            html += continuityChip('Review priority', escHtml(pbjSustainedWorkReviewPriorityLabel(flag.severity)));
        }
        html += continuityChip('Role group', escHtml(String(flag.role_group || '—')));
        html += continuityChip('Reported employee ID', escHtml(String(flag.masked_employee_id || '—')));
        html += continuityChip(
            'Period',
            escHtml(pbjFormatSustainedWorkPeriodRange(flag.period_start, flag.period_end))
        );
        html += continuityChip('≥4h streak', formatContinuityNum(flag.max_consecutive_shift_equivalent_days));
        html += continuityChip('≥7h streak', formatContinuityNum(flag.max_consecutive_full_shift_days));
        html += continuityChip('Worked days in window', formatContinuityNum(flag.rolling_90_worked_days));
        html += continuityChip(
            'Hours in window',
            flag.rolling_90_hours != null ? pbjFormatSustainedWorkMetric(flag.rolling_90_hours) : '—'
        );
        html += continuityChip(
            'Max daily hrs',
            flag.max_daily_hours != null ? pbjFormatSustainedWorkMetric(flag.max_daily_hours, 1) : '—'
        );
        html += continuityChip('Share of role hours', sharePct);
        html += '</div>';
        return html;
    }

    global.pbjSustainedWorkReviewPriorityLabel = pbjSustainedWorkReviewPriorityLabel;
    global.pbjFormatSustainedWorkMetric = pbjFormatSustainedWorkMetric;
    global.pbjSustainedWorkFlagLooksLegacy = pbjSustainedWorkFlagLooksLegacy;
    global.pbjBuildSustainedWorkFlagNarrative = pbjBuildSustainedWorkFlagNarrative;
    global.pbjNormalizeSustainedWorkFlag = pbjNormalizeSustainedWorkFlag;
    global.pbjFormatSustainedWorkPeriodRange = pbjFormatSustainedWorkPeriodRange;
    global.pbjSustainedWorkLimitationsForDisplay = pbjSustainedWorkLimitationsForDisplay;
    global.pbjSustainedWorkFlagChipsHtml = pbjSustainedWorkFlagChipsHtml;
    global.pbjSustainedWorkFlagModalBodyHtml = pbjSustainedWorkFlagModalBodyHtml;
    global.pbjWireSustainedWorkModalEmployeeLinks = pbjWireSustainedWorkModalEmployeeLinks;
})(typeof window !== 'undefined' ? window : this);
