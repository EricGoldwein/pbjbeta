/**
 * CMS ownership & control disclosure grouping for facility modals.
 * Disclosure layout only — not beneficial-owner reconstruction.
 */
(function (global) {
    'use strict';

    var SECTION_DEFS = [
        { id: 'ownership_interests', title: 'Ownership interests', pctColumn: 'own' },
        { id: 'security_interests', title: 'Security interests', pctColumn: 'security' },
        { id: 'operational_control', title: 'Operational / managerial control', pctColumn: null },
        { id: 'governing_officers', title: 'Governing body & officers', pctColumn: null },
        { id: 'other_adp', title: 'Other disclosures (ADP-only)', pctColumn: null },
    ];

    var SECTION_PRIORITY = [
        'ownership_interests',
        'security_interests',
        'operational_control',
        'governing_officers',
        'other_adp',
    ];

    function ownershipPrimarySectionForKeys(sectionKeys) {
        for (var i = 0; i < SECTION_PRIORITY.length; i++) {
            if (sectionKeys.indexOf(SECTION_PRIORITY[i]) >= 0) {
                return SECTION_PRIORITY[i];
            }
        }
        return 'other_adp';
    }

    function ownershipPctNumber(raw) {
        if (raw == null || raw === '') {
            return null;
        }
        var pct = Number(raw);
        return Number.isFinite(pct) ? pct : null;
    }

    function ownershipRoleDisplay(raw) {
        var txt = String(raw || '').trim();
        if (!txt) {
            return '—';
        }
        var lower = txt.toLowerCase();
        var smallWords = { or: 1, and: 1, of: 1, by: 1, in: 1, for: 1, the: 1, to: 1, a: 1, an: 1 };
        return lower.split(/\s+/).map(function (word, idx) {
            if (!word) {
                return word;
            }
            if (/^\d+%$/.test(word)) {
                return word;
            }
            return word.split('-').map(function (part, jdx) {
                if (!part) {
                    return part;
                }
                if ((idx > 0 || jdx > 0) && smallWords[part]) {
                    return part;
                }
                return part.charAt(0).toUpperCase() + part.slice(1);
            }).join('-');
        }).join(' ');
    }

    function ownershipAssociationDateClean(raw) {
        return String(raw || '').replace(/^since\s+/i, '').trim();
    }

    function ownershipRoleIsAdp(roleText) {
        var rl = String(roleText || '').trim().toLowerCase();
        return rl.indexOf('adp') >= 0 && rl.indexOf('snf') >= 0;
    }

    function ownershipClassifySection(roleText) {
        var rl = String(roleText || '').trim().toLowerCase();
        if (ownershipRoleIsAdp(rl)) {
            return 'adp';
        }
        if (rl.indexOf('security interest') >= 0) {
            return 'security_interests';
        }
        if (
            rl.indexOf('direct ownership') >= 0 ||
            rl.indexOf('indirect ownership') >= 0 ||
            (rl.indexOf('ownership interest') >= 0 && rl.indexOf('security') < 0)
        ) {
            return 'ownership_interests';
        }
        if (rl.indexOf('operational') >= 0 || rl.indexOf('managerial control') >= 0) {
            return 'operational_control';
        }
        if (
            rl.indexOf('governing') >= 0 ||
            rl.indexOf('officer') >= 0 ||
            rl.indexOf('director') >= 0
        ) {
            return 'governing_officers';
        }
        return 'other_adp';
    }

    function ownershipPctForRole(roleText, sectionId) {
        if (sectionId === 'ownership_interests') {
            var rlOwn = String(roleText || '').toLowerCase();
            if (
                rlOwn.indexOf('ownership interest') >= 0 &&
                rlOwn.indexOf('security') < 0 &&
                rlOwn.indexOf('mortgage') < 0
            ) {
                return true;
            }
            return false;
        }
        if (sectionId === 'security_interests') {
            return String(roleText || '').toLowerCase().indexOf('security interest') >= 0;
        }
        return false;
    }

    function ownershipAssociateKey(row) {
        var aid = String(row.owner_associate_id || '').trim();
        if (/^\d+$/.test(aid)) {
            return 'aid:' + aid;
        }
        return 'name:' + String(row.owner_name || row.owner_name_display || '').trim().toUpperCase();
    }

    function normalizeContactRow(row) {
        return {
            owner_name: row.owner_name,
            owner_name_display: row.owner_name_display || row.owner_name,
            owner_associate_id: row.owner_associate_id,
            owner_dashboard_href: row.owner_dashboard_href,
            role_text: row.role_text || '',
            ownership_pct: row.ownership_pct,
            association_date: row.association_date || '',
            role_kind: row.role_kind || '',
        };
    }

    function ownershipBuildDisclosureSections(rawRows) {
        var associates = new Map();
        (rawRows || []).forEach(function (row) {
            var norm = normalizeContactRow(row);
            var key = ownershipAssociateKey(norm);
            if (!associates.has(key)) {
                associates.set(key, {
                    key: key,
                    owner_name: norm.owner_name,
                    owner_name_display: norm.owner_name_display,
                    owner_associate_id: norm.owner_associate_id,
                    owner_dashboard_href: norm.owner_dashboard_href,
                    roles: [],
                });
            }
            var bucket = associates.get(key);
            if (!bucket.owner_associate_id && norm.owner_associate_id) {
                bucket.owner_associate_id = norm.owner_associate_id;
            }
            if (!bucket.owner_dashboard_href && norm.owner_dashboard_href) {
                bucket.owner_dashboard_href = norm.owner_dashboard_href;
            }
            bucket.roles.push(norm);
        });

        var sectionMap = {};
        SECTION_DEFS.forEach(function (def) {
            sectionMap[def.id] = { def: def, rows: [] };
        });

        associates.forEach(function (assoc) {
            var adpRoles = assoc.roles.filter(function (r) {
                return ownershipRoleIsAdp(r.role_text);
            });
            var nonAdp = assoc.roles.filter(function (r) {
                return !ownershipRoleIsAdp(r.role_text);
            });

            if (!nonAdp.length) {
                sectionMap.other_adp.rows.push({
                    associate: assoc,
                    primaryRoles: [],
                    adpRoles: adpRoles.slice(),
                });
                return;
            }

            var rolesBySection = {};
            nonAdp.forEach(function (roleRow) {
                var secId = ownershipClassifySection(roleRow.role_text);
                if (secId === 'adp' || secId === 'other_adp') {
                    secId = 'other_adp';
                }
                if (!rolesBySection[secId]) {
                    rolesBySection[secId] = [];
                }
                rolesBySection[secId].push(roleRow);
            });

            var sectionKeys = Object.keys(rolesBySection);
            var adpHostSection = ownershipPrimarySectionForKeys(sectionKeys);

            sectionKeys.forEach(function (secId) {
                if (!sectionMap[secId]) {
                    return;
                }
                sectionMap[secId].rows.push({
                    associate: assoc,
                    primaryRoles: rolesBySection[secId],
                    adpRoles: secId === adpHostSection ? adpRoles.slice() : [],
                });
            });
        });

        var sections = SECTION_DEFS.map(function (def) {
            var entry = sectionMap[def.id];
            var rows = entry.rows.slice().sort(function (a, b) {
                var nameA = String(a.associate.owner_name_display || a.associate.owner_name || '');
                var nameB = String(b.associate.owner_name_display || b.associate.owner_name || '');
                if (def.pctColumn) {
                    var pctA = ownershipSectionRowPct(a, def.id) || -1;
                    var pctB = ownershipSectionRowPct(b, def.id) || -1;
                    if (pctB !== pctA) {
                        return pctB - pctA;
                    }
                }
                return nameA.localeCompare(nameB);
            });
            return { def: def, rows: rows };
        });

        return {
            sections: sections,
            rawRowCount: (rawRows || []).length,
        };
    }

    function ownershipSectionRowPct(sectionRow, sectionId) {
        var best = null;
        (sectionRow.primaryRoles || []).forEach(function (roleRow) {
            if (!ownershipPctForRole(roleRow.role_text, sectionId)) {
                return;
            }
            var pct = ownershipPctNumber(roleRow.ownership_pct);
            if (pct != null && (best == null || pct > best)) {
                best = pct;
            }
        });
        return best;
    }

    function ownershipRoleChipLabel(roleRow, sectionId) {
        var label = ownershipRoleDisplay(roleRow.role_text || '');
        if (ownershipPctForRole(roleRow.role_text, sectionId)) {
            var pct = ownershipPctNumber(roleRow.ownership_pct);
            if (pct != null) {
                label += ' (' + pct.toFixed(1) + '%)';
            }
        }
        var dateClean = ownershipAssociationDateClean(roleRow.association_date || '');
        if (dateClean) {
            label += ' · ' + dateClean;
        }
        return label;
    }

    function ownershipDisclosureLines(operatorName) {
        var op = String(operatorName || '').trim();
        var assocLine = op
            ? 'CMS association: Disclosures are tied to the facility\u2019s enrolled operator: ' + op + '.'
            : 'CMS association: Disclosures are tied to the facility\u2019s enrolled operator.';
        var pctLine =
            'Percentages: Ownership and security-interest percentages may reflect different layers and are not expected to sum to 100%.';
        return { assocLine: assocLine, pctLine: pctLine };
    }

    /** @deprecated use ownershipDisclosureLines */
    function ownershipDisclosureDisclaimer() {
        var lines = ownershipDisclosureLines('');
        return lines.assocLine + ' ' + lines.pctLine;
    }

    global.PbjOwnershipDisclosures = {
        SECTION_DEFS: SECTION_DEFS,
        ownershipPctNumber: ownershipPctNumber,
        ownershipRoleDisplay: ownershipRoleDisplay,
        ownershipAssociationDateClean: ownershipAssociationDateClean,
        ownershipRoleIsAdp: ownershipRoleIsAdp,
        ownershipClassifySection: ownershipClassifySection,
        ownershipPctForRole: ownershipPctForRole,
        ownershipAssociateKey: ownershipAssociateKey,
        ownershipBuildDisclosureSections: ownershipBuildDisclosureSections,
        ownershipSectionRowPct: ownershipSectionRowPct,
        ownershipRoleChipLabel: ownershipRoleChipLabel,
        ownershipDisclosureLines: ownershipDisclosureLines,
        ownershipDisclosureDisclaimer: ownershipDisclosureDisclaimer,
    };
})(typeof window !== 'undefined' ? window : global);
