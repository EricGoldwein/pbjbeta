/**
 * Report Builder v3 — memo assembly surface (isolated from V2 dashboard).
 * Uses PBJReportBuilderItems registry; preview/export share the same ordered item model.
 */
(function () {
    'use strict';

    var Items = window.PBJReportBuilderItems || {};
    var defaultBuiltinItems = Items.defaultBuiltinItems || function () { return []; };
    var categoryLabel = Items.categoryLabel || function () { return ''; };
    var serializeItemsForApi = Items.serializeItemsForApi || function (x) { return x; };
    var mergeItems = Items.mergeItems || function (a) { return a; };
    var createReportScope = Items.createReportScope || function () { return {}; };

    var EVENT_WINDOW_PRESETS = [
        { id: 'before_30', label: '30 days before event' },
        { id: 'before_60', label: '60 days before event' },
        { id: 'before_90', label: '90 days before event' },
        { id: 'after_14', label: '14 days after event' },
        { id: 'after_30', label: '30 days after event' },
        { id: 'event_quarter', label: 'Quarter containing event' },
        { id: 'custom', label: 'Custom window' }
    ];

    var KEY_DATE_TYPES = [
        { value: 'incident', label: 'Incident' },
        { value: 'admission', label: 'Admission' },
        { value: 'discharge', label: 'Discharge' },
        { value: 'survey', label: 'Survey' },
        { value: 'complaint', label: 'Complaint' },
        { value: 'other', label: 'Other' }
    ];

    var FOCUS_CHIP_CATEGORIES = {
        compliance_rn: ['compliance', 'rn_coverage', 'headcount'],
        acuity: ['acuity_gap'],
        patterns: ['weekend_pattern', 'employee_anomaly', 'staffing_volatility'],
        case_dates: ['event_window'],
        data_quality: ['data_quality']
    };

    var state = {
        items: [],
        dragId: null,
        previewReady: false
    };

    function $(id) { return document.getElementById(id); }

    function readMeta() {
        var ccn = '';
        var facility = '';
        var quarters = [];
        var minWorkdate = '';
        var maxWorkdate = '';
        try {
            var el = $('pbj-report-builder-v3-meta');
            if (el && el.textContent) {
                var o = JSON.parse(el.textContent);
                ccn = String(o.ccn || '').replace(/\D/g, '');
                facility = String(o.facility || '').trim();
                if (Array.isArray(o.quarters)) {
                    quarters = o.quarters.map(function (q) { return String(q || '').trim(); }).filter(Boolean);
                }
                minWorkdate = String(o.min_workdate || '').trim();
                maxWorkdate = String(o.max_workdate || '').trim();
            }
        } catch (e) { /* ignore */ }
        if (!ccn && typeof PBJ320_EXPORT_CCN !== 'undefined' && PBJ320_EXPORT_CCN) {
            ccn = String(PBJ320_EXPORT_CCN).replace(/\D/g, '');
        }
        if (!facility) {
            var hero = $('facilityHeroName');
            if (hero) facility = String(hero.textContent || '').trim();
        }
        return {
            ccn: ccn || '',
            facility: facility || 'This facility',
            quarters: quarters,
            minWorkdate: minWorkdate,
            maxWorkdate: maxWorkdate
        };
    }

    function formatDisplayDate(iso) {
        if (!iso || !/^\d{4}-\d{2}-\d{2}$/.test(iso)) return '';
        var d = new Date(iso + 'T00:00:00');
        if (isNaN(d.getTime())) return iso;
        return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
    }

    var PREVIEW_GATED_TITLE = 'Generate a memo preview first.';

    function resizePreviewFrame() {
        var frame = $('rb3PreviewFrame');
        if (!frame || frame.classList.contains('d-none')) {
            return;
        }
        try {
            var doc = frame.contentDocument || (frame.contentWindow && frame.contentWindow.document);
            if (!doc || !doc.body) {
                return;
            }
            var h = Math.max(
                doc.body.scrollHeight || 0,
                doc.documentElement ? doc.documentElement.scrollHeight : 0,
                320
            );
            frame.style.height = h + 'px';
            frame.style.minHeight = h + 'px';
            var shell = $('rb3PaperShell');
            if (shell) {
                shell.style.minHeight = '0';
            }
        } catch (e) { /* cross-origin guard */ }
    }

    function updateSelectedPeriodLabel() {
        var el = $('rb3SelectedPeriodLabel');
        if (!el) return;
        var s = $('rb3StartDate') && $('rb3StartDate').value;
        var e = $('rb3EndDate') && $('rb3EndDate').value;
        el.textContent = (s && e) ? (formatDisplayDate(s) + ' \u2013 ' + formatDisplayDate(e)) : '\u2014';
        syncRb3AiScope();
        updateAdvancedSummary();
    }

    function updateCaseDatesSummary() {
        var summary = $('rb3CaseDatesSummary');
        var panel = $('rb3KeyDatesBlock');
        var rows = collectKeyDates().filter(function (r) { return r.date || r.note; });
        var n = rows.length;
        if (summary) {
            if (n === 0) {
                summary.textContent = 'e.g. Apr 12 \u2014 incident';
                summary.classList.add('rb3-keydates-summary-hint');
            } else if (n === 1 && rows[0].note) {
                var d = rows[0].date ? formatDisplayDate(rows[0].date) + ' \u2014 ' : '';
                summary.textContent = d + rows[0].note;
                summary.classList.remove('rb3-keydates-summary-hint');
            } else {
                summary.textContent = n + ' key date' + (n === 1 ? '' : 's');
                summary.classList.remove('rb3-keydates-summary-hint');
            }
        }
        if (panel) panel.classList.toggle('rb3-keydates-has-rows', n > 0);
        var winBlock = $('rb3EventWindowsBlock');
        if (winBlock) winBlock.classList.toggle('d-none', n === 0);
        updateAiPanelSummary();
        syncRb3FocusDatesToAiToolkit();
    }

    function syncRb3FocusDatesToAiToolkit() {
        var block = $('pbjAiFocusDatesBlock');
        if (block && block.dataset.pbjUserEdited === '1') {
            return;
        }
        var rows = collectKeyDates().filter(function (r) { return r.date || r.note; });
        if (!rows.length) {
            return;
        }
        if (typeof global.pbjV2AiFocusDatesSetFromRows === 'function') {
            global.pbjV2AiFocusDatesSetFromRows(rows);
            return;
        }
        var target = $('pbjAiPromptFocusDates');
        if (!target) {
            return;
        }
        target.value = rows.map(function (r) {
            var line = r.date || '';
            if (r.note) {
                line = line ? (line + ' \u2014 ' + r.note) : r.note;
            }
            if (r.type && r.type !== 'other' && r.type !== 'incident') {
                line = line ? (line + ' (' + r.type + ')') : r.type;
            }
            return line;
        }).join('; ');
    }

    function updateAiPanelSummary() {
        var el = $('rb3AiPanelSummary');
        if (!el) return;
        var keyCount = collectKeyDates().filter(function (r) { return r.date || r.note; }).length;
        if (keyCount) {
            el.textContent = keyCount + ' key date' + (keyCount === 1 ? '' : 's') + ' \u00b7 export + prompt for your AI chat';
        } else {
            el.textContent = 'Send scoped PBJ data and a starter prompt to Claude, ChatGPT, or Gemini';
        }
    }

    function updateAdvancedSummary() {
        updateAiPanelSummary();
    }

    function syncRb3AiScope() {
        var s = ($('rb3StartDate') || {}).value || '';
        var e = ($('rb3EndDate') || {}).value || '';
        var label = (s && e) ? (formatDisplayDate(s) + ' \u2013 ' + formatDisplayDate(e)) : '\u2014';
        var rb3Line = $('rb3AiScopeLine');
        var toolkitLine = $('pbjAiToolkitScopeLine');
        if (rb3Line) rb3Line.textContent = label;
        if (toolkitLine) toolkitLine.textContent = label;
        var aiStart = $('pbjAiStartDate');
        var aiEnd = $('pbjAiEndDate');
        if (aiStart && s) aiStart.value = s;
        if (aiEnd && e) aiEnd.value = e;
        document.querySelectorAll('[data-pbj-ai-grain]').forEach(function (btn) {
            var grain = btn.getAttribute('data-pbj-ai-grain');
            var on = grain === 'daterange';
            btn.classList.toggle('active', on);
            btn.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
        ['pbjAiScopeQuarters', 'pbjAiScopeYears', 'pbjAiScopeDay'].forEach(function (id) {
            var panel = $(id);
            if (panel) panel.classList.add('d-none');
        });
        var rangePanel = $('pbjAiScopeRange');
        if (rangePanel) rangePanel.classList.remove('d-none');
        if (typeof window !== 'undefined') {
            window.__pbjAiPackScopeStale = true;
        }
    }

    function setPreviewGatedControls(on) {
        var title = on ? '' : PREVIEW_GATED_TITLE;
        ['rb3DownloadBtn', 'rb3FullscreenBtn', 'rb3PrintBtn'].forEach(function (id) {
            var el = $(id);
            if (!el) return;
            el.disabled = !on;
            el.title = title;
            el.setAttribute('aria-disabled', on ? 'false' : 'true');
            el.classList.toggle('d-none', !on);
        });
        var dl = $('rb3DownloadBtn');
        if (dl) {
            dl.classList.toggle('btn-outline-primary', on);
            dl.classList.toggle('btn-outline-secondary', !on);
        }
        var hint = $('rb3DownloadHint');
        if (hint) {
            hint.classList.toggle('d-none', !!on);
        }
    }

    function setStatus(msg, isError) {
        var el = $('rb3Status');
        if (!el) return;
        el.textContent = msg || '';
        el.classList.toggle('text-danger', !!isError);
        el.classList.toggle('text-muted', !isError);
    }

    function setBusy(on) {
        var overlay = $('rb3PreviewOverlay');
        if (overlay) {
            overlay.classList.toggle('show', !!on);
            overlay.setAttribute('aria-hidden', on ? 'false' : 'true');
        }
        var p = $('rb3PreviewBtn');
        var d = $('rb3DownloadBtn');
        if (p) p.disabled = !!on;
        if (d && !on) d.disabled = !state.previewReady;
    }

    function setPreviewReady(on) {
        state.previewReady = !!on;
        setPreviewGatedControls(!!on);
        var shell = $('rb3PaperShell');
        if (shell) shell.classList.toggle('rb3-paper-empty', !on);
        var empty = $('rb3PreviewEmptyState');
        if (empty) {
            empty.classList.toggle('d-none', !!on);
            empty.setAttribute('aria-hidden', on ? 'true' : 'false');
        }
        var frame = $('rb3PreviewFrame');
        if (frame) {
            frame.classList.toggle('d-none', !on);
            if (on) {
                frame.addEventListener('load', resizePreviewFrame);
                setTimeout(resizePreviewFrame, 60);
                setTimeout(resizePreviewFrame, 350);
            }
        }
    }

    function itemRowFromStored(row) {
        return {
            id: row.id,
            type: row.type || 'finding',
            category: row.category || 'finding',
            title: row.title,
            subtitle: row.subtitle || '',
            source: row.source || 'report_builder',
            enabled: row.enabled !== false,
            order: row.order,
            include_key: row.include_key || row.id,
            section_marker_id: row.section_marker_id || row.id,
            removable: !!row.removable
        };
    }

    function initSections() {
        var raw = null;
        try {
            raw = sessionStorage.getItem('pbj_rb3_report_items_v1') || sessionStorage.getItem('pbj_rb3_sections_v1');
        } catch (e) { /* ignore */ }
        if (raw) {
            try {
                var parsed = JSON.parse(raw);
                if (Array.isArray(parsed) && parsed.length) {
                    state.items = parsed.map(itemRowFromStored);
                    renderSectionsList();
                    renderUserItemsPanel();
                    updateAdvancedSummary();
                    return;
                }
            } catch (e2) { /* ignore */ }
        }
        state.items = defaultBuiltinItems();
        renderSectionsList();
        renderUserItemsPanel();
        updateAdvancedSummary();
    }

    function persistSections() {
        try {
            sessionStorage.setItem('pbj_rb3_report_items_v1', JSON.stringify(state.items));
        } catch (e) { /* ignore */ }
    }

    function resetSections() {
        state.items = defaultBuiltinItems();
        if (window.PBJReportBuilder && typeof window.PBJReportBuilder.clearQueuedItems === 'function') {
            window.PBJReportBuilder.clearQueuedItems();
        }
        renderSectionsList();
        renderUserItemsPanel();
        persistSections();
    }

    function syncQueuedUserItems() {
        if (!window.PBJReportBuilder || typeof window.PBJReportBuilder.peekQueuedItems !== 'function') return;
        var queued = window.PBJReportBuilder.peekQueuedItems();
        if (!queued.length) return;
        state.items = mergeItems(state.items.filter(function (i) {
            return i.source === 'report_builder' || !i.removable;
        }), queued);
        window.PBJReportBuilder.clearQueuedItems();
        state.items.forEach(function (row, idx) { row.order = idx; });
        renderSectionsList();
        renderUserItemsPanel();
        persistSections();
    }

    function renderUserItemsPanel() {
        var panel = $('rb3UserItemsPanel');
        var list = $('rb3UserItemsList');
        if (!panel || !list) return;
        var userItems = state.items.filter(function (i) { return i.category === 'user_added' || (i.source && i.source !== 'report_builder'); });
        if (!userItems.length) {
            panel.classList.add('d-none');
            list.innerHTML = '';
            return;
        }
        panel.classList.remove('d-none');
        list.innerHTML = userItems.map(function (i) {
            return '<li class="small"><span class="fw-semibold">' + escapeHtml(i.title) + '</span>' +
                (i.subtitle ? ' <span class="text-secondary">— ' + escapeHtml(i.subtitle) + '</span>' : '') +
                ' <span class="badge text-bg-light border">' + escapeHtml(i.source || 'user') + '</span></li>';
        }).join('');
    }

    function renderSectionsList() {
        var root = $('rb3SectionsList');
        if (!root) return;
        root.innerHTML = '';
        var builtinRows = state.items.filter(function (i) { return i.source === 'report_builder'; });
        builtinRows.forEach(function (sec, idx) {
            var row = document.createElement('div');
            row.className = 'rb3-section-row' + (sec.enabled ? '' : ' is-disabled');
            row.setAttribute('role', 'listitem');
            row.setAttribute('data-section-id', sec.id);
            row.setAttribute('draggable', 'true');
            row.innerHTML =
                '<span class="rb3-section-handle" aria-hidden="true" title="Drag to reorder">\u22ee\u22ee</span>' +
                '<div class="rb3-section-copy flex-grow-1 min-w-0">' +
                '<div class="rb3-section-title-row d-flex align-items-center gap-1">' +
                '<span class="rb3-section-title">' + escapeHtml(sec.title) + '</span>' +
                (sec.subtitle ? '<button type="button" class="btn btn-link btn-sm p-0 rb3-section-info" aria-expanded="false" aria-label="Show details for ' + escapeHtml(sec.title) + '" title="Details">\u2139</button>' : '') +
                '</div>' +
                (sec.subtitle ? '<div class="rb3-section-subtitle small text-secondary d-none">' + escapeHtml(sec.subtitle) + '</div>' : '') +
                '</div>' +
                '<div class="rb3-section-actions d-flex align-items-center gap-1 flex-shrink-0">' +
                '<button type="button" class="btn btn-outline-secondary btn-sm rb3-section-move d-md-none" data-dir="-1" aria-label="Move up" title="Move up">\u2191</button>' +
                '<button type="button" class="btn btn-outline-secondary btn-sm rb3-section-move d-md-none" data-dir="1" aria-label="Move down" title="Move down">\u2193</button>' +
                '<div class="form-check form-switch mb-0">' +
                '<input class="form-check-input rb3-section-enabled" type="checkbox" id="rb3Sec_' + sec.id + '"' + (sec.enabled ? ' checked' : '') + '>' +
                '<label class="form-check-label visually-hidden" for="rb3Sec_' + sec.id + '">Include ' + escapeHtml(sec.title) + '</label>' +
                '</div></div>';
            row.addEventListener('dragstart', function (ev) {
                state.dragId = sec.id;
                row.classList.add('is-dragging');
                if (ev.dataTransfer) {
                    ev.dataTransfer.effectAllowed = 'move';
                    ev.dataTransfer.setData('text/plain', sec.id);
                }
            });
            row.addEventListener('dragend', function () {
                state.dragId = null;
                row.classList.remove('is-dragging');
                root.querySelectorAll('.rb3-section-row').forEach(function (r) { r.classList.remove('is-drop-target'); });
            });
            row.addEventListener('dragover', function (ev) {
                ev.preventDefault();
                if (state.dragId && state.dragId !== sec.id) row.classList.add('is-drop-target');
            });
            row.addEventListener('dragleave', function () { row.classList.remove('is-drop-target'); });
            row.addEventListener('drop', function (ev) {
                ev.preventDefault();
                row.classList.remove('is-drop-target');
                var fromId = state.dragId || (ev.dataTransfer && ev.dataTransfer.getData('text/plain'));
                if (!fromId || fromId === sec.id) return;
                moveSection(fromId, sec.id);
            });
            var toggle = row.querySelector('.rb3-section-enabled');
            if (toggle) {
                toggle.addEventListener('change', function () {
                    sec.enabled = !!toggle.checked;
                    row.classList.toggle('is-disabled', !sec.enabled);
                    persistSections();
                    updateAdvancedSummary();
                });
            }
            var infoBtn = row.querySelector('.rb3-section-info');
            if (infoBtn) {
                infoBtn.addEventListener('click', function (ev) {
                    ev.preventDefault();
                    ev.stopPropagation();
                    var sub = row.querySelector('.rb3-section-subtitle');
                    if (!sub) return;
                    var wasHidden = sub.classList.contains('d-none');
                    sub.classList.toggle('d-none');
                    infoBtn.setAttribute('aria-expanded', wasHidden ? 'true' : 'false');
                    row.classList.toggle('is-expanded', wasHidden);
                });
            }
            row.querySelectorAll('.rb3-section-move').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    var dir = parseInt(btn.getAttribute('data-dir') || '0', 10);
                    var ids = builtinRows.map(function (s) { return s.id; });
                    var pos = ids.indexOf(sec.id);
                    var newIdx = pos + dir;
                    if (newIdx < 0 || newIdx >= ids.length) return;
                    var fromGlobal = state.items.findIndex(function (s) { return s.id === ids[pos]; });
                    var toGlobal = state.items.findIndex(function (s) { return s.id === ids[newIdx]; });
                    var tmp = state.items[fromGlobal];
                    state.items[fromGlobal] = state.items[toGlobal];
                    state.items[toGlobal] = tmp;
                    state.items.forEach(function (r, i) { r.order = i; });
                    renderSectionsList();
                    persistSections();
                });
            });
            root.appendChild(row);
        });
    }

    function moveSection(fromId, toId) {
        var fromIdx = state.items.findIndex(function (s) { return s.id === fromId; });
        var toIdx = state.items.findIndex(function (s) { return s.id === toId; });
        if (fromIdx < 0 || toIdx < 0) return;
        var item = state.items.splice(fromIdx, 1)[0];
        state.items.splice(toIdx, 0, item);
        state.items.forEach(function (r, i) { r.order = i; });
        renderSectionsList();
        persistSections();
    }

    function escapeHtml(s) {
        return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    function addKeyDateRow(dateVal, labelVal, typeVal) {
        var root = $('rb3KeyDatesList');
        if (!root) return;
        var row = document.createElement('div');
        row.className = 'rb3-keydate-row rb-keydate-row';
        var typeOpts = KEY_DATE_TYPES.map(function (t) {
            var sel = (typeVal || 'other') === t.value ? ' selected' : '';
            return '<option value="' + t.value + '"' + sel + '>' + t.label + '</option>';
        }).join('');
        row.innerHTML =
            '<input type="date" class="form-control form-control-sm rb3-keydate-date" autocomplete="off" aria-label="Key date">' +
            '<select class="form-select form-select-sm rb3-keydate-type" aria-label="Key date type">' + typeOpts + '</select>' +
            '<input type="text" class="form-control form-control-sm rb3-keydate-label" maxlength="240" placeholder="Note" aria-label="Key date note">' +
            '<button type="button" class="btn btn-outline-secondary btn-sm rb-keydate-remove" title="Remove" aria-label="Remove date">&times;</button>';
        if (dateVal) row.querySelector('.rb3-keydate-date').value = String(dateVal).slice(0, 10);
        if (labelVal) row.querySelector('.rb3-keydate-label').value = String(labelVal);
        row.querySelector('.rb-keydate-remove').addEventListener('click', function () {
            var root = $('rb3KeyDatesList');
            row.remove();
            if (root && !root.querySelector('.rb3-keydate-row')) {
                addKeyDateRow();
            } else {
                updateCaseDatesSummary();
            }
        });
        row.querySelectorAll('input, select').forEach(function (inp) {
            inp.addEventListener('change', updateCaseDatesSummary);
            inp.addEventListener('input', updateCaseDatesSummary);
        });
        root.appendChild(row);
        updateCaseDatesSummary();
    }

    function ensureDefaultKeyDateRows() {
        var root = $('rb3KeyDatesList');
        if (!root || root.querySelector('.rb3-keydate-row')) return;
        addKeyDateRow();
    }

    function collectKeyDates() {
        var root = $('rb3KeyDatesList');
        if (!root) return [];
        var out = [];
        root.querySelectorAll('.rb3-keydate-row').forEach(function (row) {
            var date = (row.querySelector('.rb3-keydate-date') || {}).value || '';
            var note = (row.querySelector('.rb3-keydate-label') || {}).value || '';
            var type = (row.querySelector('.rb3-keydate-type') || {}).value || 'other';
            if (date || note) out.push({ date: date, note: note, type: type });
        });
        return out;
    }

    function renderEventWindows() {
        var root = $('rb3EventWindowsList');
        if (!root) return;
        root.innerHTML = '';
        EVENT_WINDOW_PRESETS.forEach(function (preset) {
            var id = 'rb3Win_' + preset.id;
            var wrap = document.createElement('div');
            wrap.className = 'form-check';
            wrap.innerHTML =
                '<input class="form-check-input rb3-event-window" type="checkbox" id="' + id + '" data-preset="' + preset.id + '">' +
                '<label class="form-check-label small" for="' + id + '">' + escapeHtml(preset.label) + '</label>';
            root.appendChild(wrap);
        });
        root.querySelectorAll('.rb3-event-window').forEach(function (cb) {
            cb.addEventListener('change', function () {
                var customWrap = $('rb3CustomWindowWrap');
                if (!customWrap) return;
                var customOn = root.querySelector('[data-preset="custom"]') && root.querySelector('[data-preset="custom"]').checked;
                customWrap.classList.toggle('d-none', !customOn);
                updateAdvancedSummary();
            });
        });
    }

    function collectEventWindows() {
        var root = $('rb3EventWindowsList');
        if (!root) return [];
        var out = [];
        root.querySelectorAll('.rb3-event-window:checked').forEach(function (cb) {
            var preset = cb.getAttribute('data-preset') || '';
            var row = { preset: preset };
            if (preset === 'custom') {
                row.start_date = ($('rb3CustomWindowStart') || {}).value || '';
                row.end_date = ($('rb3CustomWindowEnd') || {}).value || '';
            }
            out.push(row);
        });
        return out;
    }

    function collectIncludeSections() {
        if (Items.includeSectionsFromItems) {
            return Items.includeSectionsFromItems(state.items);
        }
        var inc = { key_dates: true, date_ranges_of_interest: true, period_summary: true };
        state.items.forEach(function (s) {
            var key = s.include_key || s.id;
            if (key) inc[key] = !!s.enabled;
        });
        return inc;
    }

    function parseDateRanges(raw) {
        if (typeof reportBuilderParseDateRanges === 'function') {
            return reportBuilderParseDateRanges(raw);
        }
        var lines = String(raw || '').split(/\r?\n/).map(function (s) { return s.trim(); }).filter(Boolean);
        var ranges = [];
        var re = /\d{4}-\d{2}-\d{2}/g;
        lines.forEach(function (line) {
            var m = line.match(re);
            if (m && m.length >= 2) ranges.push({ start_date: m[0], end_date: m[1], role: 'primary' });
        });
        return ranges;
    }

    function collectFindingCategories() {
        var picked = [];
        document.querySelectorAll('.rb3-finding-cat:checked').forEach(function (el) {
            if (el.value) picked.push(el.value);
        });
        return picked;
    }

    function applyFocusChipsToCategoryCheckboxes() {
        var enabled = new Set();
        document.querySelectorAll('.rb3-focus-chip.active').forEach(function (btn) {
            var key = btn.getAttribute('data-rb3-focus') || '';
            (FOCUS_CHIP_CATEGORIES[key] || []).forEach(function (c) { enabled.add(c); });
        });
        document.querySelectorAll('.rb3-finding-cat').forEach(function (el) {
            el.checked = enabled.has(el.value);
        });
    }

    function syncFocusChipsFromCategoryCheckboxes() {
        document.querySelectorAll('.rb3-focus-chip').forEach(function (btn) {
            var key = btn.getAttribute('data-rb3-focus') || '';
            var cats = FOCUS_CHIP_CATEGORIES[key] || [];
            var on = cats.length && cats.every(function (c) {
                var el = document.querySelector('.rb3-finding-cat[value="' + c + '"]');
                return el && el.checked;
            });
            btn.classList.toggle('active', on);
            btn.setAttribute('aria-pressed', on ? 'true' : 'false');
        });
    }

    function updateFocusChipSummary() {
        var hint = $('rb3FocusHint');
        var on = !$('rb3AutoDetectFindings') || $('rb3AutoDetectFindings').checked;
        if (!hint) {
            return;
        }
        if (!on) {
            hint.textContent = 'Auto-detect off — no staffing flags will be scanned.';
            hint.classList.remove('d-none');
            hint.setAttribute('aria-hidden', 'false');
            return;
        }
        var chips = document.querySelectorAll('.rb3-focus-chip');
        var active = Array.prototype.filter.call(chips, function (btn) {
            return btn.classList.contains('active');
        });
        if (!active.length) {
            hint.textContent = 'Select at least one issue type to scan.';
            hint.classList.remove('d-none');
            hint.setAttribute('aria-hidden', 'false');
            return;
        }
        if (active.length === chips.length) {
            hint.textContent = 'All issue types selected — full scan for the memo period.';
        } else {
            var labels = active.map(function (btn) {
                return String(btn.textContent || '').trim();
            }).filter(Boolean);
            hint.textContent = 'Scanning: ' + labels.join(', ') + '.';
        }
        hint.classList.remove('d-none');
        hint.setAttribute('aria-hidden', 'false');
    }

    function syncFindingCategoriesEnabled() {
        var wrap = $('rb3FindingCategoriesWrap');
        var chips = $('rb3FocusChips');
        var on = !$('rb3AutoDetectFindings') || $('rb3AutoDetectFindings').checked;
        if (wrap) wrap.classList.toggle('opacity-50', !on);
        if (chips) chips.classList.toggle('opacity-50', !on);
        document.querySelectorAll('.rb3-finding-cat').forEach(function (el) {
            el.disabled = !on;
        });
        document.querySelectorAll('.rb3-focus-chip').forEach(function (btn) {
            btn.disabled = !on;
        });
        updateFocusChipSummary();
    }

    function getLatestQuarterKey() {
        var qSel = $('rb3PeriodQuarter');
        if (!qSel || !qSel.options.length) return '';
        return String(qSel.options[0].value || '');
    }

    function getLatestYearKey() {
        var yearSel = $('rb3PeriodYear');
        if (!yearSel || !yearSel.options.length) return '';
        return String(yearSel.options[0].value || '');
    }

    function setPeriodQuickPickActive(quick) {
        document.querySelectorAll('[data-rb3-period-quick]').forEach(function (btn) {
            btn.classList.toggle('active', btn.getAttribute('data-rb3-period-quick') === quick);
        });
    }

    function syncPeriodQuickPickFromSelection() {
        var mode = ($('rb3PeriodMode') || {}).value || 'quarter';
        var quick = 'custom';
        if (mode === 'quarter') {
            var qs = Array.from(($('rb3PeriodQuarter') || {}).selectedOptions || [])
                .map(function (o) { return String(o.value || ''); })
                .filter(Boolean);
            if (qs.length === 1 && qs[0] === getLatestQuarterKey()) {
                quick = 'latest_quarter';
            }
        } else if (mode === 'year') {
            var years = Array.from(($('rb3PeriodYear') || {}).selectedOptions || [])
                .map(function (o) { return String(o.value || ''); })
                .filter(Boolean);
            if (years.length === 1 && years[0] === getLatestYearKey()) {
                quick = 'latest_year';
            }
        }
        setPeriodQuickPickActive(quick);
    }

    function clearYearSelections() {
        var yearSel = $('rb3PeriodYear');
        if (!yearSel) return;
        Array.from(yearSel.options).forEach(function (o) { o.selected = false; });
    }

    function clearQuarterSelections() {
        var qSel = $('rb3PeriodQuarter');
        if (!qSel) return;
        Array.from(qSel.options).forEach(function (o) { o.selected = false; });
    }

    function applyPeriodQuickPick(quick) {
        var qSel = $('rb3PeriodQuarter');
        var yearSel = $('rb3PeriodYear');
        var modeInput = $('rb3PeriodMode');
        if (!modeInput) return;
        setPeriodQuickPickActive(quick);
        if (quick === 'latest_quarter') {
            modeInput.value = 'quarter';
            clearYearSelections();
            if (qSel && qSel.options.length) {
                Array.from(qSel.options).forEach(function (o, i) { o.selected = i === 0; });
            }
            syncPeriodModeUI();
            var editorQ = $('rb3PeriodEditor');
            if (editorQ) editorQ.open = false;
            return;
        }
        if (quick === 'latest_year') {
            modeInput.value = 'year';
            clearQuarterSelections();
            if (yearSel && yearSel.options.length) {
                Array.from(yearSel.options).forEach(function (o, i) { o.selected = i === 0; });
            }
            syncPeriodModeUI();
            var editorYear = $('rb3PeriodEditor');
            if (editorYear) editorYear.open = false;
            return;
        }
        if (quick === 'custom') {
            modeInput.value = 'custom';
            syncPeriodModeUI();
            var editorCustom = $('rb3PeriodEditor');
            if (editorCustom) editorCustom.open = true;
            var startInput = $('rb3StartDate');
            if (startInput) startInput.focus();
            return;
        }
    }

    function bindPeriodQuickPicks() {
        document.querySelectorAll('[data-rb3-period-quick]').forEach(function (btn) {
            btn.addEventListener('click', function () {
                applyPeriodQuickPick(btn.getAttribute('data-rb3-period-quick') || 'latest_quarter');
            });
        });
    }

    function bindFocusChips() {
        document.querySelectorAll('.rb3-focus-chip').forEach(function (btn) {
            btn.addEventListener('click', function () {
                if (btn.disabled) return;
                var on = !btn.classList.contains('active');
                if (!on) {
                    var activeCount = document.querySelectorAll('.rb3-focus-chip.active').length;
                    if (activeCount <= 1) {
                        return;
                    }
                }
                btn.classList.toggle('active', on);
                btn.setAttribute('aria-pressed', on ? 'true' : 'false');
                applyFocusChipsToCategoryCheckboxes();
                updateFocusChipSummary();
            });
        });
        document.querySelectorAll('.rb3-finding-cat').forEach(function (el) {
            el.addEventListener('change', syncFocusChipsFromCategoryCheckboxes);
        });
    }

    function collectPayload() {
        syncQueuedUserItems();
        applyFocusChipsToCategoryCheckboxes();
        var keyDates = collectKeyDates();
        var scope = createReportScope({
            start_date: ($('rb3StartDate') || {}).value || '',
            end_date: ($('rb3EndDate') || {}).value || '',
            case_events: keyDates
        });
        var userItems = state.items.filter(function (i) { return i.category === 'user_added' || (i.source && i.source !== 'report_builder'); });
        var autoFindingsEl = $('rb3AutoDetectFindings');
        return {
            start_date: scope.report_period.start,
            end_date: scope.report_period.end,
            key_dates: keyDates,
            staffing_emphasis: ($('rb3StaffingEmphasis') || {}).value || 'total_first',
            date_ranges_of_interest: parseDateRanges(($('rb3DateRangesText') || {}).value || ''),
            include_sections: collectIncludeSections(),
            section_order: state.items.map(function (s) { return s.id; }),
            report_items: serializeItemsForApi(state.items),
            user_report_items: serializeItemsForApi(userItems),
            report_scope: scope,
            event_windows: collectEventWindows(),
            auto_detect_findings: !autoFindingsEl || autoFindingsEl.checked,
            finding_categories: collectFindingCategories(),
            smart_section_flow: true
        };
    }

    function apiUrl(path) {
        if (typeof pbjApiUrl === 'function') return pbjApiUrl(path);
        return path;
    }

    function placeholderSrcdoc() {
        return '<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Memo preview</title><style>' +
            'body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;display:flex;align-items:flex-start;justify-content:center;min-height:auto;color:#64748b;padding:1.5rem 1.25rem 1.75rem;line-height:1.5;margin:0;}' +
            'ul{text-align:left;padding-left:1.15rem;margin:0.5rem 0 0;}li{margin:0.2rem 0;font-size:0.86rem;color:#475569;}' +
            '</style></head><body><div style="max-width:24rem;width:100%;">' +
            '<h1 style="font-size:1rem;color:#334155;font-weight:600;margin:0 0 0.35rem;">No preview generated</h1>' +
            '<p style="font-size:0.84rem;margin:0;color:#64748b;">Select a period and generate a preview to build the memo.</p>' +
            '</div></body></html>';
    }

    function ensurePreviewSurface() {
        var frame = $('rb3PreviewFrame');
        if (!frame) return;
        frame.srcdoc = placeholderSrcdoc();
        setPreviewReady(false);
    }

    async function generatePreview() {
        var frame = $('rb3PreviewFrame');
        var warnEl = $('rb3Warnings');
        var startIso = ($('rb3StartDate') || {}).value || '';
        var endIso = ($('rb3EndDate') || {}).value || '';
        if (!startIso || !endIso) {
            setStatus('Select a review period first (quarter, year, or custom dates).', true);
            return;
        }
        setStatus('Generating preview...', false);
        setBusy(true);
        if (warnEl) { warnEl.textContent = ''; warnEl.classList.add('d-none'); }
        try {
            var resp = await fetch(apiUrl('/api/report_builder_v3/preview'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(collectPayload()),
                credentials: 'include'
            });
            var data = {};
            try {
                data = await resp.json();
            } catch (parseErr) {
                throw new Error('Preview failed (server returned ' + resp.status + '). Try again in a moment.');
            }
            if (!resp.ok || data.error) {
                var msg = data.error || ('Preview request failed (' + resp.status + ').');
                if (resp.status === 500 && /module|import/i.test(msg)) {
                    msg += ' The deployment may be missing v3 Python modules — redeploy the 315461 bundle.';
                }
                throw new Error(msg);
            }
            if (frame) {
                frame.srcdoc = String(data.html || '');
                frame.addEventListener('load', resizePreviewFrame, { once: true });
            }
            setPreviewReady(true);
            resizePreviewFrame();
            if (warnEl && Array.isArray(data.warnings) && data.warnings.length) {
                warnEl.textContent = data.warnings.join(' ');
                warnEl.classList.remove('d-none');
            }
            var fc = Array.isArray(data.staffing_findings) ? data.staffing_findings.length : 0;
            var keyDates = collectKeyDates().filter(function (r) { return r.date; });
            var statusMsg = 'Memo ready: ' + (data.resolved_start_date || '?') + ' to ' + (data.resolved_end_date || '?') + '.';
            if (keyDates.length) {
                statusMsg += ' ' + keyDates.length + ' key date' + (keyDates.length === 1 ? '' : 's') + ' included.';
            }
            if (fc) statusMsg += ' ' + fc + ' suggested issue' + (fc === 1 ? '' : 's') + ' flagged.';
            setStatus(statusMsg, false);
            var hint = $('rb3FocusHint');
            if (hint && fc) {
                hint.classList.remove('d-none');
                hint.setAttribute('aria-hidden', 'false');
                hint.textContent = fc + ' flag' + (fc === 1 ? '' : 's') + ' detected — customize sections to adjust what is included.';
            }
        } catch (err) {
            setStatus('Preview failed: ' + (err && err.message ? err.message : String(err)), true);
        } finally {
            setBusy(false);
        }
    }

    async function downloadHtml() {
        setStatus('Preparing download...', false);
        setBusy(true);
        try {
            var resp = await fetch(apiUrl('/api/report_builder_v3/download_html'), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(collectPayload()),
                credentials: 'include'
            });
            if (!resp.ok) {
                var msg = 'Download request failed.';
                try {
                    var errJson = await resp.json();
                    if (errJson && errJson.error) msg = errJson.error;
                } catch (e) { /* ignore */ }
                throw new Error(msg);
            }
            var blob = await resp.blob();
            var cd = resp.headers.get('Content-Disposition') || '';
            var m = cd.match(/filename=([^;]+)/i);
            var fileName = m ? m[1].replace(/['"]/g, '').trim() : 'staffing_analysis_memo.html';
            var url = window.URL.createObjectURL(blob);
            var a = document.createElement('a');
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            setStatus('Download started: ' + fileName, false);
        } catch (err) {
            setStatus('Download failed: ' + (err && err.message ? err.message : String(err)), true);
        } finally {
            setBusy(false);
        }
    }

    function rb3QuarterKey(raw) {
        if (typeof rbQuarterKey === 'function') return rbQuarterKey(raw);
        var s = String(raw || '').trim().toUpperCase();
        if (!s) return '';
        var m1 = s.match(/^CY(\d{4})Q([1-4])$/);
        if (m1) return m1[1] + 'Q' + m1[2];
        var m2 = s.match(/^(\d{4})Q([1-4])$/);
        if (m2) return m2[1] + 'Q' + m2[2];
        return '';
    }

    function rb3QuarterBounds(qKey) {
        if (typeof rbQuarterBounds === 'function') return rbQuarterBounds(qKey);
        var m = String(qKey || '').match(/^(\d{4})Q([1-4])$/);
        if (!m) return null;
        var year = parseInt(m[1], 10);
        var q = parseInt(m[2], 10);
        if (!year || !q) return null;
        var startMonth = (q - 1) * 3 + 1;
        var endMonth = startMonth + 2;
        var start = year + '-' + String(startMonth).padStart(2, '0') + '-01';
        var endDate = new Date(year, endMonth, 0);
        var end = year + '-' + String(endMonth).padStart(2, '0') + '-' + String(endDate.getDate()).padStart(2, '0');
        return { start: start, end: end };
    }

    function rb3QuarterSortNum(qKey) {
        if (typeof rbQuarterSortNum === 'function') return rbQuarterSortNum(qKey);
        var m = String(qKey || '').match(/^(\d{4})Q([1-4])$/);
        if (!m) return 0;
        return parseInt(m[1], 10) * 10 + parseInt(m[2], 10);
    }

    function rb3FormatQuarter(qKey) {
        if (typeof formatQuarter === 'function') return formatQuarter(qKey);
        var m = String(qKey || '').match(/^(\d{4})Q([1-4])$/);
        return m ? ('Q' + m[2] + ' ' + m[1]) : String(qKey || '');
    }

    function rb3QuartersFromIsoBounds(minIso, maxIso) {
        function parseIso(s) {
            var t = String(s || '').trim();
            if (!/^\d{4}-\d{2}-\d{2}$/.test(t)) return null;
            var d = new Date(t + 'T00:00:00');
            return isNaN(d.getTime()) ? null : d;
        }
        function yq(d) {
            return { y: d.getFullYear(), q: Math.floor(d.getMonth() / 3) + 1 };
        }
        var lo = parseIso(minIso);
        var hi = parseIso(maxIso);
        if (!lo || !hi) return [];
        if (hi < lo) { var tmp = lo; lo = hi; hi = tmp; }
        var cur = yq(lo);
        var end = yq(hi);
        var out = [];
        while (cur.y < end.y || (cur.y === end.y && cur.q <= end.q)) {
            out.push(cur.y + 'Q' + cur.q);
            cur.q += 1;
            if (cur.q > 4) { cur.q = 1; cur.y += 1; }
        }
        return out;
    }

    function rb3CollectQuarterKeys() {
        var meta = readMeta();
        var quarterSet = new Set();
        if (meta.quarters.length) {
            meta.quarters.forEach(function (raw) {
                var q = rb3QuarterKey(raw);
                if (q) quarterSet.add(q);
            });
        }
        if (!quarterSet.size && meta.minWorkdate && meta.maxWorkdate) {
            rb3QuartersFromIsoBounds(meta.minWorkdate, meta.maxWorkdate).forEach(function (q) {
                quarterSet.add(q);
            });
        }
        if (!quarterSet.size) {
            document.querySelectorAll('#quarterRange option').forEach(function (opt) {
                var q = rb3QuarterKey(opt.value);
                if (q) quarterSet.add(q);
            });
        }
        return Array.from(quarterSet);
    }

    function populatePeriodSelectors() {
        var yearSel = $('rb3PeriodYear');
        var qSel = $('rb3PeriodQuarter');
        if (!yearSel || !qSel) return;
        var quarterList = rb3CollectQuarterKeys().sort(function (a, b) {
            return rb3QuarterSortNum(b) - rb3QuarterSortNum(a);
        });
        if (!quarterList.length) {
            populatePeriodSelectors._attempts = (populatePeriodSelectors._attempts || 0) + 1;
            yearSel.innerHTML = '<option value="" disabled>Loading PBJ years…</option>';
            qSel.innerHTML = '<option value="" disabled>Loading PBJ quarters…</option>';
            if (populatePeriodSelectors._attempts < 12) {
                setTimeout(populatePeriodSelectors, 300);
            } else {
                yearSel.innerHTML = '<option value="" disabled>No PBJ years available</option>';
                qSel.innerHTML = '<option value="" disabled>No PBJ quarters available</option>';
                setStatus('Could not load PBJ quarters for this facility. Try again or use Custom range.', true);
            }
            return;
        }
        populatePeriodSelectors._attempts = 0;
        var yearMap = new Map();
        quarterList.forEach(function (qk) {
            var m = String(qk).match(/^(\d{4})Q([1-4])$/);
            if (!m) return;
            var y = m[1];
            var q = parseInt(m[2], 10);
            if (!yearMap.has(y)) yearMap.set(y, []);
            yearMap.get(y).push(q);
        });
        yearSel.innerHTML = '';
        Array.from(yearMap.keys()).sort(function (a, b) { return parseInt(b, 10) - parseInt(a, 10); }).forEach(function (y) {
            var opt = document.createElement('option');
            opt.value = y;
            var qs = (yearMap.get(y) || []).slice().sort(function (a, b) { return a - b; });
            var qMin = qs.length ? qs[0] : null;
            var qMax = qs.length ? qs[qs.length - 1] : null;
            opt.textContent = (qMin && qMax) ? (String(y) + ' (Q' + qMin + '–Q' + qMax + ')') : String(y);
            yearSel.appendChild(opt);
        });
        qSel.innerHTML = '';
        quarterList.forEach(function (qk) {
            var opt = document.createElement('option');
            opt.value = qk;
            opt.textContent = rb3FormatQuarter(qk);
            qSel.appendChild(opt);
        });
        if (qSel.options.length && !Array.from(qSel.selectedOptions).length) {
            qSel.options[0].selected = true;
        }
        var meta = readMeta();
        var startEl = $('rb3StartDate');
        var endEl = $('rb3EndDate');
        if (startEl && meta.minWorkdate) startEl.min = meta.minWorkdate;
        if (endEl && meta.maxWorkdate) endEl.max = meta.maxWorkdate;
        applyPeriodQuickPick('latest_quarter');
        syncPeriodQuickPickFromSelection();
    }

    function onPeriodSelectionChange() {
        applyPeriodPreset();
        syncPeriodQuickPickFromSelection();
    }

    function applyPeriodPreset() {
        var mode = ($('rb3PeriodMode') || {}).value || 'quarter';
        var startEl = $('rb3StartDate');
        var endEl = $('rb3EndDate');
        if (!startEl || !endEl) return;
        if (mode === 'custom') {
            updateSelectedPeriodLabel();
            return;
        }
        if (mode === 'year') {
            var years = Array.from(($('rb3PeriodYear') || {}).selectedOptions || []).map(function (o) { return parseInt(o.value, 10); }).filter(function (v) { return !isNaN(v); });
            if (years.length) {
                var yMin = Math.min.apply(null, years);
                var yMax = Math.max.apply(null, years);
                startEl.value = yMin + '-01-01';
                endEl.value = yMax + '-12-31';
            }
        } else if (mode === 'quarter') {
            var qs = Array.from(($('rb3PeriodQuarter') || {}).selectedOptions || []).map(function (o) { return o.value; }).filter(Boolean);
            if (qs.length) {
                var sorted = qs.slice().sort(function (a, b) { return rb3QuarterSortNum(a) - rb3QuarterSortNum(b); });
                var bLo = rb3QuarterBounds(sorted[0]);
                var bHi = rb3QuarterBounds(sorted[sorted.length - 1]);
                if (bLo && bHi) {
                    startEl.value = bLo.start;
                    endEl.value = bHi.end;
                }
            }
        }
        updateSelectedPeriodLabel();
    }

    function syncPeriodModeUI() {
        var mode = ($('rb3PeriodMode') || {}).value || 'quarter';
        var show = function (id, on) {
            var el = $(id);
            if (el) el.classList.toggle('d-none', !on);
        };
        var dateWrap = $('rb3PeriodDatesWrap');
        if (dateWrap) dateWrap.classList.toggle('rb-period-dates-visible', mode === 'custom');
        show('rb3PeriodYearWrap', mode === 'year');
        show('rb3PeriodQuarterWrap', mode === 'quarter');
        document.querySelectorAll('.rb3-period-tab').forEach(function (btn) {
            btn.classList.toggle('active', btn.getAttribute('data-rb3-period-mode') === mode);
        });
        applyPeriodPreset();
        syncPeriodQuickPickFromSelection();
        if (mode === 'custom') {
            var editor = $('rb3PeriodEditor');
            if (editor) editor.open = true;
        }
    }

    function seedFromControlCenter() {
        if (typeof reportBuilderSeedFromControlCenter !== 'function') return;
        reportBuilderSeedFromControlCenter();
        var sd = ($('rbStartDate') || {}).value;
        var ed = ($('rbEndDate') || {}).value;
        var mode = ($('rbPeriodMode') || {}).value;
        if ($('rb3StartDate') && sd) $('rb3StartDate').value = sd;
        if ($('rb3EndDate') && ed) $('rb3EndDate').value = ed;
        if ($('rb3PeriodMode') && mode) $('rb3PeriodMode').value = mode;
        syncPeriodModeUI();
    }

    function bindRb3AiCard() {
        var openBtn = $('rb3AiOpenToolkitBtn');
        if (openBtn) {
            openBtn.addEventListener('click', function () {
                syncRb3AiScope();
            });
        }
        var copyBtn = $('rb3AiCopyPromptBtn');
        if (copyBtn) {
            copyBtn.addEventListener('click', function () {
                syncRb3AiScope();
                if (typeof pbjV2CopyAiStarterPrompt === 'function') {
                    pbjV2CopyAiStarterPrompt();
                } else if (typeof window.pbjRb3CopyAiStarterPrompt === 'function') {
                    window.pbjRb3CopyAiStarterPrompt();
                }
            });
        }
        var dlBtn = $('rb3AiDownloadCsvBtn');
        if (dlBtn) {
            dlBtn.addEventListener('click', function () {
                syncRb3AiScope();
                if (typeof pbjV2DownloadAiContextPackCsv === 'function') {
                    pbjV2DownloadAiContextPackCsv();
                } else if (typeof window.pbjRb3DownloadAiContextPackCsv === 'function') {
                    window.pbjRb3DownloadAiContextPackCsv();
                }
            });
        }
    }

    function initCoverageBadge() {
        var badge = $('rb3DataFreshnessBadge');
        if (!badge || typeof reportBuilderCoverageLabelFromIso !== 'function') return;
        var span = reportBuilderCoverageLabelFromIso(
            badge.getAttribute('data-min-workdate') || '',
            badge.getAttribute('data-max-workdate') || ''
        );
        badge.textContent = span ? span : 'Latest available';
        badge.classList.remove('is-loading');
    }

    function onPaneShow() {
        syncQueuedUserItems();
        seedFromControlCenter();
        populatePeriodSelectors();
        ensureDefaultKeyDateRows();
        updateCaseDatesSummary();
        applyFocusChipsToCategoryCheckboxes();
        syncFindingCategoriesEnabled();
        ensurePreviewSurface();
        initCoverageBadge();
        syncRb3AiScope();
        updateAdvancedSummary();
    }

    function bindUi() {
        bindPeriodQuickPicks();
        bindFocusChips();
        bindRb3AiCard();
        ensureDefaultKeyDateRows();
        var addBtn = $('rb3KeyDatesAddBtn');
        if (addBtn) addBtn.addEventListener('click', function () { addKeyDateRow(); });
        var resetBtn = $('rb3SectionsResetBtn');
        if (resetBtn) resetBtn.addEventListener('click', resetSections);
        var previewBtn = $('rb3PreviewBtn');
        if (previewBtn) previewBtn.addEventListener('click', generatePreview);
        var downloadBtn = $('rb3DownloadBtn');
        if (downloadBtn) downloadBtn.addEventListener('click', function () {
            if (!state.previewReady) return;
            downloadHtml();
        });
        var fsBtn = $('rb3FullscreenBtn');
        if (fsBtn) {
            fsBtn.addEventListener('click', function () {
                if (!state.previewReady) return;
                var shell = $('rb3PaperShell');
                if (!shell) return;
                if (document.fullscreenElement) document.exitFullscreen().catch(function () {});
                else if (shell.requestFullscreen) shell.requestFullscreen().catch(function () {});
            });
        }
        var printBtn = $('rb3PrintBtn');
        if (printBtn) {
            printBtn.addEventListener('click', function () {
                if (!state.previewReady) return;
                var frame = $('rb3PreviewFrame');
                try {
                    if (frame && frame.contentWindow) {
                        frame.contentWindow.focus();
                        frame.contentWindow.print();
                    }
                } catch (e) { window.print(); }
            });
        }
        ['rb3PeriodYear', 'rb3PeriodQuarter', 'rb3StartDate', 'rb3EndDate'].forEach(function (id) {
            var el = $(id);
            if (el) el.addEventListener('change', onPeriodSelectionChange);
        });
        document.querySelectorAll('.rb3-period-tab').forEach(function (btn) {
            btn.addEventListener('click', function () {
                var mode = btn.getAttribute('data-rb3-period-mode') || 'quarter';
                if ($('rb3PeriodMode')) $('rb3PeriodMode').value = mode;
                syncPeriodModeUI();
                syncPeriodQuickPickFromSelection();
            });
        });
        var autoFindings = $('rb3AutoDetectFindings');
        if (autoFindings) {
            autoFindings.addEventListener('change', syncFindingCategoriesEnabled);
            syncFindingCategoriesEnabled();
        }
        var emphasis = $('rb3StaffingEmphasis');
        if (emphasis) emphasis.addEventListener('change', updateAdvancedSummary);
        updateAiPanelSummary();
    }

    window.pbjRb3SyncFocusDatesToAiToolkit = syncRb3FocusDatesToAiToolkit;

    window.pbjRb3CollectAiPackContext = function pbjRb3CollectAiPackContext() {
        syncQueuedUserItems();
        return {
            key_dates: collectKeyDates().filter(function (r) { return r.date || r.note; }),
            user_items: state.items
                .filter(function (i) {
                    return i.category === 'user_added' || (i.source && i.source !== 'report_builder');
                })
                .map(function (i) {
                    return {
                        title: i.title || '',
                        subtitle: i.subtitle || '',
                        source: i.source || '',
                        category: i.category || ''
                    };
                }),
            event_windows: collectEventWindows()
        };
    };

    window.pbjReportBuilderV3OnShow = onPaneShow;

    function isStandalonePage() {
        try {
            var el = document.getElementById('pbj-report-builder-v3-meta');
            if (el && el.textContent) {
                var o = JSON.parse(el.textContent);
                return !!(o && o.standalone);
            }
        } catch (e0) { /* ignore */ }
        return false;
    }

    document.addEventListener('DOMContentLoaded', function () {
        initSections();
        renderEventWindows();
        bindUi();
        if (isStandalonePage()) {
            window.__pbjReportBuilderViewActive = true;
            onPaneShow();
        }
        document.addEventListener('pbj:report-builder:item-queued', function () {
            if (window.__pbjReportBuilderViewActive || isStandalonePage()) {
                syncQueuedUserItems();
            }
        });
    });
})();
