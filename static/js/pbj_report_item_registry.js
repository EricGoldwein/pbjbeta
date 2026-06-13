/**
 * Report Builder item registry (client).
 *
 * V2 dashboard must NOT import this module directly. Future integration uses:
 *   window.PBJReportBuilder.ingestItem(item)
 * which queues structured items without coupling dashboard DOM to builder internals.
 */
(function (global) {
    'use strict';

    var SOURCE_REPORT_BUILDER = 'report_builder';
    var SOURCE_DASHBOARD = 'dashboard';
    var SOURCE_CONTROL_CENTER = 'pbj320_control_center';
    var SOURCE_AI = 'ai';
    var SOURCE_USER = 'user';

    var BUILTIN_REPORT_ITEMS = [
        { id: 'key_staffing_findings', type: 'finding', category: 'finding', title: 'Key Staffing Findings', subtitle: 'Auto-detected staffing patterns for the report period and event windows', source: SOURCE_REPORT_BUILDER, enabled: true, include_key: 'key_staffing_findings', section_marker_id: 'key_staffing_findings', removable: false },
        { id: 'daily_staffing_table', type: 'table', category: 'finding', title: 'Daily Key-Date Staffing', subtitle: 'Staffing table for each case event date', source: SOURCE_REPORT_BUILDER, enabled: true, include_key: 'daily_staffing_table', section_marker_id: 'daily_staffing_table', removable: false },
        { id: 'state_compliance', type: 'finding', category: 'finding', title: 'Days Below State Minimum', subtitle: 'Compliance summary for the report period', source: SOURCE_REPORT_BUILDER, enabled: true, include_key: 'state_compliance', section_marker_id: 'state_compliance', removable: false },
        { id: 'quarterly_staffing', type: 'chart', category: 'finding', title: 'Longitudinal Staffing Analysis', subtitle: 'Charts and period/quarter tables', source: SOURCE_REPORT_BUILDER, enabled: true, include_key: 'quarterly_staffing', section_marker_id: 'quarterly_staffing', removable: false },
        { id: 'case_mix', type: 'table', category: 'finding', title: 'Case-Mix Analysis', subtitle: 'Case-mix and Harrington-adjusted metrics', source: SOURCE_REPORT_BUILDER, enabled: true, include_key: 'case_mix', section_marker_id: 'case_mix', removable: false },
        { id: 'red_flags', type: 'finding', category: 'finding', title: 'CMS Red Flags', subtitle: 'Provider info flags during the report period', source: SOURCE_REPORT_BUILDER, enabled: true, include_key: 'red_flags', section_marker_id: 'red_flags', removable: false },
        { id: 'event_windows', type: 'event_window', category: 'finding', title: 'Staffing Around Key Events', subtitle: 'Supplemental windows around case events', source: SOURCE_REPORT_BUILDER, enabled: true, include_key: 'event_windows', section_marker_id: 'event_windows', removable: false },
        { id: 'supporting_context', type: 'note', category: 'supporting_context', title: 'Relevant Context Outside Selected Period', subtitle: 'Limited context when patterns extend beyond the report period', source: SOURCE_REPORT_BUILDER, enabled: true, include_key: 'supporting_context', section_marker_id: 'supporting_context', removable: false },
        { id: 'appendix', type: 'methodology', category: 'finding', title: 'Appendix', subtitle: 'Methods, sources, and reference tables', source: SOURCE_REPORT_BUILDER, enabled: true, include_key: 'appendix', section_marker_id: 'appendix', removable: false }
    ];

    var DEFAULT_ITEM_ORDER = BUILTIN_REPORT_ITEMS.map(function (i) { return i.id; });
    var BUILTIN_BY_ID = {};
    BUILTIN_REPORT_ITEMS.forEach(function (i) { BUILTIN_BY_ID[i.id] = i; });

    var EXECUTIVE_INCLUDE_KEYS = ['key_dates', 'date_ranges_of_interest', 'period_summary'];

    function cloneItem(item, order) {
        var row = {};
        Object.keys(item).forEach(function (k) { row[k] = item[k]; });
        row.order = typeof order === 'number' ? order : (item.order || 0);
        return row;
    }

    function defaultBuiltinItems() {
        return BUILTIN_REPORT_ITEMS.map(function (item, idx) { return cloneItem(item, idx); });
    }

    function categoryLabel(category) {
        var map = {
            finding: 'Finding',
            user_added: 'Added',
            supporting_context: 'Context',
            scope: 'Scope'
        };
        return map[category] || 'Item';
    }

    function normalizeItemOrder(items) {
        var sorted = items.slice().sort(function (a, b) {
            return (a.order || 0) - (b.order || 0);
        });
        var out = [];
        var seen = {};
        sorted.forEach(function (row) {
            var sid = row.section_marker_id || row.id;
            if (sid && BUILTIN_BY_ID[sid] && !seen[sid]) {
                out.push(sid);
                seen[sid] = true;
            }
        });
        DEFAULT_ITEM_ORDER.forEach(function (id) {
            if (!seen[id]) out.push(id);
        });
        return out;
    }

    function includeSectionsFromItems(items) {
        var inc = { key_dates: true, date_ranges_of_interest: true, period_summary: true };
        BUILTIN_REPORT_ITEMS.forEach(function (b) {
            inc[b.include_key || b.id] = !!b.enabled;
        });
        items.forEach(function (row) {
            var key = row.include_key || row.id;
            if (key && inc.hasOwnProperty(key)) inc[key] = !!row.enabled;
        });
        return inc;
    }

    function coerceExternalItem(raw, order) {
        if (!raw || typeof raw !== 'object') return null;
        var id = String(raw.id || '').trim();
        if (!id) return null;
        var builtin = BUILTIN_BY_ID[id];
        var row = builtin ? cloneItem(builtin, order) : {
            id: id,
            type: raw.type || 'note',
            category: raw.category || 'user_added',
            title: raw.title || id,
            subtitle: raw.subtitle || '',
            source: raw.source || SOURCE_USER,
            enabled: true,
            include_key: id,
            section_marker_id: id,
            removable: true
        };
        if (raw.title) row.title = String(raw.title);
        if (raw.subtitle != null) row.subtitle = String(raw.subtitle || '');
        if (raw.type) row.type = String(raw.type);
        if (raw.category) row.category = String(raw.category);
        if (raw.source) row.source = String(raw.source);
        if ('enabled' in raw) row.enabled = !!raw.enabled;
        if (raw.dataRef || raw.data_ref) row.data_ref = String(raw.dataRef || raw.data_ref);
        if (raw.dateRange || raw.date_range) {
            var dr = raw.dateRange || raw.date_range;
            row.date_range = { start: String(dr.start || ''), end: String(dr.end || '') };
        }
        if (Array.isArray(raw.relatedEvents || raw.related_events)) {
            row.related_events = (raw.relatedEvents || raw.related_events).map(String);
        }
        row.order = typeof raw.order === 'number' ? raw.order : order;
        return row;
    }

    function mergeItems(builtinItems, externalItems) {
        var merged = builtinItems.map(function (i) { return cloneItem(i, i.order); });
        var byId = {};
        merged.forEach(function (i) { byId[i.id] = i; });
        var nextOrder = merged.reduce(function (m, i) { return Math.max(m, i.order || 0); }, -1) + 1;
        (externalItems || []).forEach(function (raw) {
            var row = coerceExternalItem(raw, nextOrder);
            if (!row) return;
            if (byId[row.id] && row.source === SOURCE_REPORT_BUILDER) return;
            if (byId[row.id]) {
                Object.keys(row).forEach(function (k) { byId[row.id][k] = row[k]; });
            } else {
                merged.push(row);
                byId[row.id] = row;
                nextOrder += 1;
            }
        });
        return merged;
    }

    function serializeItemsForApi(items) {
        return items.map(function (row, idx) {
            return {
                id: row.id,
                type: row.type,
                category: row.category,
                title: row.title,
                subtitle: row.subtitle || '',
                source: row.source,
                enabled: !!row.enabled,
                order: typeof row.order === 'number' ? row.order : idx,
                include_key: row.include_key || row.id,
                section_marker_id: row.section_marker_id || row.id,
                data_ref: row.data_ref || null,
                date_range: row.date_range || null,
                related_events: row.related_events || []
            };
        });
    }

    function createReportScope(opts) {
        opts = opts || {};
        return {
            report_period: {
                start: String(opts.start_date || ''),
                end: String(opts.end_date || '')
            },
            resident_stay_period: opts.resident_stay_start && opts.resident_stay_end ? {
                start: String(opts.resident_stay_start),
                end: String(opts.resident_stay_end)
            } : null,
            case_events: Array.isArray(opts.case_events) ? opts.case_events : []
        };
    }

    var userQueue = [];

    function ingestItem(item) {
        var row = coerceExternalItem(item, userQueue.length + DEFAULT_ITEM_ORDER.length);
        if (!row) return false;
        if (row.source === SOURCE_REPORT_BUILDER && BUILTIN_BY_ID[row.id]) return false;
        userQueue.push(row);
        try {
            global.dispatchEvent(new CustomEvent('pbj:report-builder:item-queued', { detail: { item: row } }));
        } catch (e) { /* IE fallback N/A */ }
        return true;
    }

    function drainUserQueue() {
        var q = userQueue.slice();
        userQueue = [];
        return q;
    }

    function peekUserQueue() {
        return userQueue.slice();
    }

    function clearUserQueue() {
        userQueue = [];
    }

    var Items = {
        BUILTIN_REPORT_ITEMS: BUILTIN_REPORT_ITEMS,
        DEFAULT_ITEM_ORDER: DEFAULT_ITEM_ORDER,
        BUILTIN_BY_ID: BUILTIN_BY_ID,
        SOURCES: {
            report_builder: SOURCE_REPORT_BUILDER,
            dashboard: SOURCE_DASHBOARD,
            pbj320_control_center: SOURCE_CONTROL_CENTER,
            ai: SOURCE_AI,
            user: SOURCE_USER
        },
        defaultBuiltinItems: defaultBuiltinItems,
        categoryLabel: categoryLabel,
        normalizeItemOrder: normalizeItemOrder,
        includeSectionsFromItems: includeSectionsFromItems,
        mergeItems: mergeItems,
        serializeItemsForApi: serializeItemsForApi,
        createReportScope: createReportScope,
        ingestItem: ingestItem,
        drainUserQueue: drainUserQueue,
        peekUserQueue: peekUserQueue,
        clearUserQueue: clearUserQueue
    };

    global.PBJReportBuilderItems = Items;
    global.PBJReportBuilder = global.PBJReportBuilder || {};
    global.PBJReportBuilder.ingestItem = ingestItem;
    global.PBJReportBuilder.peekQueuedItems = peekUserQueue;
    global.PBJReportBuilder.clearQueuedItems = clearUserQueue;
})(typeof window !== 'undefined' ? window : globalThis);
