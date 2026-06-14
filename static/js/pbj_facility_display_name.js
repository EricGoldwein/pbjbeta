/**
 * PBJ320 — smart CMS provider (facility) display names.
 * Display-layer only: never mutates the official CMS provider name.
 */
(function (global) {
    'use strict';

    var CMS_PROVIDER_NAME_SOURCE = 'provider_info_combined.provider_name';

    var THRESHOLDS = {
        keepFullMax: 28,
        lightMax: 44,
        displayTarget: 28,
        shortTarget: 22,
        microTarget: 16,
        minCompactSavings: 5,
        compactLabelMax: 36,
    };

    var INSTITUTIONAL_ANCHOR_RES = [
        /\bmedical\s+center\b/i,
        /\bcontinuing\s+care\s+hospital\b/i,
        /\bcontinuing\s+care\b/i,
        /\bmethodist\s+hospital\b/i,
        /\bbaylor\s+scott\s*&\s*white\b/i,
        /\bveterans\b/i,
        /\bcommunity\s+living\b/i,
        /\bhospital\b/i,
    ];

    var REMOVABLE_SUFFIXES = [
        'Rehabilitation and Healthcare Center',
        'Rehabilitation & Healthcare Center',
        'Rehab and Healthcare Center',
        'Rehab & Healthcare Center',
        'Rehabilitation and Health Care Center',
        'Rehabilitation & Health Care Center',
        'Rehabilitation and Nursing Center',
        'Rehabilitation & Nursing Center',
        'Rehabilitation and Nursing',
        'Rehabilitation & Nursing',
        'Rehab and Nursing Center',
        'Rehab & Nursing Center',
        'Nursing and Rehabilitation Center',
        'Nursing & Rehabilitation Center',
        'Skilled Nursing and Rehabilitation Center',
        'Skilled Nursing & Rehabilitation Center',
        'Skilled Nursing and Rehabilitation',
        'Skilled Nursing & Rehabilitation',
        'Healthcare and Rehabilitation Center',
        'Health Care and Rehabilitation Center',
        'Health and Rehabilitation Center',
        'Center for Nursing and Rehabilitation',
        'Center for Nursing & Rehabilitation',
        'Center for Rehab and Nursing',
        'Post-Acute and Rehabilitation Center',
        'Post Acute and Rehabilitation Center',
        'and Healthcare Center',
        '& Healthcare Center',
        'and Health Care Center',
        '& Health Care Center',
        'Healthcare Center',
        'Health Care Center',
        'Rehabilitation Center',
        'Rehab Center',
        'Nursing Center',
        'Care Center',
        'Nursing Home',
        'Skilled Nursing Facility',
        'Skilled Nursing Facili',
        'Skilled Nursing Unit',
        'Post-Acute Care',
        'Post Acute Care',
        'Post-Acute',
        'Post Acute',
        'Convalescent Center',
        'Living Center',
        'Health Center',
        'Medical Center',
        'Wellness Center',
        'Senior Living',
        'Senior Care',
        'Long-Term Care',
        'Long Term Care',
        'Care and Rehabilitation Center LLC',
        'Health and Rehabilitation Center LLC',
        'Nursing and Rehabilitation LLC',
        'SNF',
        'LTC',
    ];

    var LIGHT_REMOVABLE_SUFFIXES = [
        'Rehabilitation and Healthcare Center',
        'Rehabilitation & Healthcare Center',
        'Rehab and Healthcare Center',
        'Rehab & Healthcare Center',
        'Rehabilitation and Nursing Center',
        'Rehabilitation & Nursing Center',
        'Rehabilitation and Nursing',
        'Rehabilitation & Nursing',
        'Rehab and Nursing Center',
        'Rehab & Nursing Center',
        'Nursing and Rehabilitation Center',
        'Nursing & Rehabilitation Center',
        'Center for Nursing and Rehabilitation',
        'Center for Nursing & Rehabilitation',
        'Healthcare and Rehabilitation Center',
        'Health Care and Rehabilitation Center',
        'Health and Rehabilitation Center',
        'and Healthcare Center',
        '& Healthcare Center',
        'and Health Care Center',
        '& Health Care Center',
        'Healthcare Center',
        'Health Care Center',
        'Rehabilitation Center',
        'Rehab Center',
        'Nursing Center',
        'Care Center',
        'Post-Acute Care',
        'Post Acute Care',
        'Post-Acute',
        'Post Acute',
        'Convalescent Center',
        'Living Center',
        'Health Center',
        'Medical Center',
        'Wellness Center',
        'Senior Living',
        'Senior Care',
        'Long-Term Care',
        'Long Term Care',
    ];

    var GENERIC_SOLO_WORDS = {
        the: true,
        at: true,
        and: true,
        of: true,
        center: true,
        centre: true,
        club: true,
        healthcare: true,
        health: true,
        care: true,
        rehab: true,
        rehabilitation: true,
        nursing: true,
        home: true,
        snf: true,
        ltc: true,
        living: true,
        medical: true,
        wellness: true,
        convalescent: true,
        acute: true,
        post: true,
        term: true,
        long: true,
        skilled: true,
        facility: true,
        senior: true,
    };

    var ARTIFACT_TOKEN_RE = /\b(cente|cent|cen|ce|rehabilita|rehabilitati|nursin|facilit|childr|skil)\b/i;
    var MIXED_UGLY_RE = /\b(?:Rehab|Health|Healthcare|Nursing|Health Care)\s+AND\s+[A-Z]/;
    var DANGLING_END_RE = /\b(and|&|at|of|for|the|with|skilled|llc)\s*$/i;
    var INCOMPLETE_FRAGMENT_RES = [
        /\s+at\s+fle\s*$/i,
        /\s+at\s+wel\s*$/i,
        /\s+at\s+the\s+v\s*$/i,
        /\s+of\s+del\s*$/i,
        /\s+of\s+day\s*$/i,
        /\s+of\s+win\s*$/i,
        /\s+skilled\s*$/i,
    ];

    var PRESERVE_AT_PLACE = /\bat\s+[A-Za-z0-9]/i;
    var TITLE_MINOR = {
        and: true,
        of: true,
        at: true,
        by: true,
        the: true,
        a: true,
        an: true,
        in: true,
        on: true,
        for: true,
        to: true,
        with: true,
    };
    var TITLE_PRESERVE = {
        cms: true,
        ccn: true,
        snf: true,
        llc: true,
        lpn: true,
        rn: true,
        'd/p': true,
    };

    function escapeRegex(s) {
        return String(s).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    function collapseWhitespace(s) {
        return String(s || '')
            .replace(/\u00a0/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
    }

    function trimRawCmsName(name) {
        return String(name || '')
            .replace(/\u00a0/g, ' ')
            .trim();
    }

    function normalizeFacilityNameSpacing(name) {
        return collapseWhitespace(
            String(name || '')
                .replace(/\s*&\s*/g, ' & ')
                .replace(/\s*-\s*/g, ' - ')
                .replace(/\s*,\s*/g, ', ')
        );
    }

    function processingName(name) {
        return normalizeFacilityNameSpacing(trimRawCmsName(name));
    }

    function normalizeForComparison(name) {
        return collapseWhitespace(String(name || ''))
            .toLowerCase()
            .replace(/\s*&\s*/g, ' and ')
            .replace(/\s*-\s*/g, ' - ')
            .replace(/\brehabilitation\b/g, 'rehab')
            .replace(/\bhealthcare\b/g, 'health')
            .replace(/\bhealth care\b/g, 'health');
    }

    function wasSubstantivelyShortened(before, after) {
        var beforeNorm = normalizeForComparison(before);
        var afterNorm = normalizeForComparison(after);
        if (!beforeNorm || !afterNorm || beforeNorm === afterNorm) {
            return false;
        }
        return afterNorm.length < beforeNorm.length;
    }

    function wasLabelNormalized(raw, label) {
        if (!raw || !label) {
            return false;
        }
        return (
            normalizeForComparison(raw) === normalizeForComparison(label) &&
            collapseWhitespace(raw) !== collapseWhitespace(label)
        );
    }

    function institutionalAnchorPhrases(name) {
        var phrases = [];
        INSTITUTIONAL_ANCHOR_RES.forEach(function (re) {
            var match = String(name || '').match(re);
            if (match && match[0]) {
                phrases.push(match[0].toLowerCase());
            }
        });
        return phrases;
    }

    function preservesInstitutionalAnchor(original, candidate) {
        var anchors = institutionalAnchorPhrases(original);
        if (!anchors.length) {
            return true;
        }
        var cand = collapseWhitespace(candidate).toLowerCase();
        for (var i = 0; i < anchors.length; i++) {
            if (cand.indexOf(anchors[i]) < 0) {
                return false;
            }
        }
        return true;
    }

    function shouldStripSeparatorFragments(rawCms, hadArtifact) {
        return trimRawCmsName(rawCms).length === 50 || !!hadArtifact;
    }

    function hasDanglingSeparatorFragment(text) {
        var out = collapseWhitespace(text);
        if (!out) {
            return false;
        }
        if (/,\s*[A-Za-z]\s*$/.test(out)) {
            return true;
        }
        if (/\/\s*[A-Za-z]\s*$/.test(out)) {
            return true;
        }
        if (/\s+at\s+[A-Za-z]\s*$/i.test(out)) {
            return true;
        }
        if (/\s+of\s+[A-Za-z]\s*$/i.test(out)) {
            return true;
        }
        var dashMatch = out.match(/\s[-–]\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)\s*$/);
        if (dashMatch && dashMatch[1]) {
            var dashTokens = dashMatch[1].split(/\s+/).filter(Boolean);
            if (dashTokens.some(function (token) { return token.length <= 2; })) {
                return true;
            }
        }
        var last = out.split(/\s+/).pop() || '';
        return last.length === 1 && /[A-Za-z]/.test(last);
    }

    function stripSeparatorTruncationFragments(name, rawCms, hadArtifact) {
        if (!shouldStripSeparatorFragments(rawCms, hadArtifact)) {
            return collapseWhitespace(name);
        }
        var out = collapseWhitespace(name);
        var prev;
        do {
            prev = out;
            out = out.replace(/,\s*[A-Za-z]\s*$/g, '');
            out = out.replace(/\/\s*[A-Za-z]\s*$/g, '');
            out = out.replace(/\s+at\s+[A-Za-z]\s*$/gi, '');
            out = out.replace(/\s+of\s+[A-Za-z]\s*$/gi, '');
            out = out.replace(/\s[-–]\s+([A-Za-z0-9]+(?:\s+[A-Za-z0-9]+)*)\s*$/g, function (whole, frag) {
                var tokens = frag.split(/\s+/).filter(Boolean);
                if (!tokens.length) {
                    return '';
                }
                if (tokens.some(function (token) { return token.length <= 2; })) {
                    return '';
                }
                return whole;
            });
            var tokens = out.split(/\s+/);
            if (tokens.length > 2) {
                var lastTok = tokens[tokens.length - 1];
                if (lastTok.length === 1 && /[A-Za-z]/.test(lastTok)) {
                    tokens.pop();
                    out = tokens.join(' ');
                }
            }
            out = collapseWhitespace(out);
        } while (out !== prev);
        return out;
    }

    function suffixTokenPattern(token) {
        var t = String(token || '').toLowerCase().replace(/\./g, '');
        if (t === '&') {
            return '(?:&|and)';
        }
        if (t === 'center' || t === 'centre') {
            return '(?:Center|Centre|Cente|Cent|Cen|Ce)(?:er|re)?';
        }
        if (t === 'rehabilitation') {
            return '(?:Rehabilitation|Rehabilita(?:tion)?)';
        }
        if (t === 'nursing') {
            return '(?:Nursing|Nursin(?:g)?)';
        }
        if (t === 'facility') {
            return '(?:Facility|Facilit(?:y)?)';
        }
        if (t === 'children') {
            return '(?:Children|Childr(?:en)?)';
        }
        if (t === 'healthcare') {
            return '(?:Healthcare|Health\\s*Care)';
        }
        if (t === 'care' && token === 'Care') {
            return 'Care';
        }
        if (t === 'health' && token === 'Health') {
            return 'Health';
        }
        return escapeRegex(token);
    }

    function suffixPatternRegex(suffix) {
        var parts = normalizeFacilityNameSpacing(suffix).split(/\s+/).map(suffixTokenPattern);
        return new RegExp('(?:[\\s,\\-]+|^)' + parts.join('\\s+') + '\\s*$', 'i');
    }

    function detectCmsTruncationArtifacts(rawName) {
        var raw = collapseWhitespace(rawName);
        var warnings = [];
        if (!raw) {
            return { hadCmsTruncationArtifact: false, artifactWarnings: warnings };
        }
        var tokens = raw.split(/\s+/);
        var last = tokens[tokens.length - 1] || '';
        if (/\b(cente|cent|cen|rehabilita|rehabilitati|nursin|facilit|childr|skil)$/i.test(last)) {
            warnings.push('truncated_final_token');
        }
        if (/\bce$/i.test(last) && last.length <= 3) {
            warnings.push('truncated_final_token_ce');
        }
        if (raw.length === 50) {
            warnings.push('cms_field_50_chars');
            if (!/[.!?]$/.test(raw) && /\w$/.test(raw) && /\b(cente|cent|cen|rehabilita|nursin|facilit|childr|fle|wel|del|day|win|v)$/i.test(last)) {
                warnings.push('likely_mid_word_cutoff');
            }
        }
        INCOMPLETE_FRAGMENT_RES.forEach(function (re) {
            if (re.test(raw)) {
                warnings.push('incomplete_location_fragment');
            }
        });
        return {
            hadCmsTruncationArtifact: warnings.length > 0,
            artifactWarnings: warnings,
        };
    }

    function stripIncompleteLocationFragments(name) {
        var out = collapseWhitespace(name);
        var prev;
        do {
            prev = out;
            INCOMPLETE_FRAGMENT_RES.forEach(function (re) {
                out = out.replace(re, '');
            });
            out = collapseWhitespace(out);
        } while (out !== prev);
        return out;
    }

    function stripTrailingJunk(name) {
        var out = collapseWhitespace(name);
        var prev;
        do {
            prev = out;
            out = out
                .replace(/[\s,\-&]+(?:and|&)\s*$/i, '')
                .replace(/[\s,\-&]+(?:at|the|of|for|with|skilled|llc)\s*$/i, '')
                .replace(/[\s,\-&]+$/g, '')
                .trim();
        } while (out !== prev);
        return out;
    }

    function applyAbbreviations(name) {
        var out = String(name || '');
        out = out.replace(/\bRehabilitation\b/gi, 'Rehab');
        out = out.replace(/\bRehabilita\b/gi, 'Rehab');
        out = out.replace(/\bHealth Care\b/gi, 'Health');
        out = out.replace(/\bHealthcare\b/gi, 'Health');
        out = out.replace(/\bPost-Acute\b/gi, 'Post Acute');
        return collapseWhitespace(out);
    }

    function wordCount(s) {
        return collapseWhitespace(s).split(/\s+/).filter(Boolean).length;
    }

    function soloWordKey(word) {
        return String(word || '')
            .toLowerCase()
            .replace(/\./g, '')
            .trim();
    }

    function lettersMostlyUpper(s) {
        var letters = String(s || '').replace(/[^A-Za-z]/g, '');
        if (!letters) {
            return false;
        }
        var upper = letters.replace(/[^A-Z]/g, '').length;
        return upper / letters.length >= 0.65;
    }

    function titleCaseToken(word, index) {
        var raw = String(word || '');
        if (!raw) {
            return raw;
        }
        if (raw === '&') {
            return '&';
        }
        if (raw === '-') {
            return '-';
        }
        var key = raw.toLowerCase().replace(/\./g, '');
        if (TITLE_PRESERVE[key]) {
            return raw.toUpperCase();
        }
        if (/^[A-Z]{2,}\/[A-Z]{2,}$/.test(raw)) {
            return raw;
        }
        if (/^[A-Z]\.[A-Z]\.?$/i.test(raw) || /^[A-Z]\.[A-Z]\.?[A-Za-z]*$/i.test(raw)) {
            return raw.replace(/([a-z]+)/g, function (m) {
                return m.charAt(0).toUpperCase() + m.slice(1).toLowerCase();
            }).replace(/^([A-Z])\.([A-Z])\./i, function (_, a, b) {
                return a.toUpperCase() + '.' + b.toUpperCase() + '.';
            });
        }
        if (/^[A-Z]\.$/.test(raw)) {
            return raw;
        }
        if (/^St\.?$/i.test(raw)) {
            return 'St.';
        }
        if (/^Mt\.?$/i.test(raw)) {
            return 'Mt.';
        }
        if (index > 0 && TITLE_MINOR[key]) {
            return key;
        }
        if (raw.length === 1) {
            return raw.toUpperCase();
        }
        return raw.charAt(0).toUpperCase() + raw.slice(1).toLowerCase();
    }

    function normalizeDisplayCasing(label, rawFull) {
        var text = collapseWhitespace(label);
        if (!text) {
            return text;
        }
        if (!lettersMostlyUpper(rawFull || text) && !MIXED_UGLY_RE.test(text)) {
            return text;
        }
        return text
            .split(/\s+/)
            .map(function (word, i) {
                return titleCaseToken(word, i);
            })
            .join(' ');
    }

    function compactLabelWarnings(label, rawFull, level) {
        var warnings = [];
        var text = collapseWhitespace(label);
        var raw = collapseWhitespace(rawFull);
        if (!text) {
            warnings.push('blank_label');
            return warnings;
        }
        if (ARTIFACT_TOKEN_RE.test(text)) {
            warnings.push('truncation_artifact_visible');
        }
        if (/\bce\b/i.test(text.split(/\s+/).pop() || '')) {
            warnings.push('truncation_artifact_ce');
        }
        if (DANGLING_END_RE.test(text)) {
            warnings.push('dangling_fragment_end');
        }
        INCOMPLETE_FRAGMENT_RES.forEach(function (re) {
            if (re.test(text)) {
                warnings.push('incomplete_location_fragment');
            }
        });
        if (hasDanglingSeparatorFragment(text)) {
            warnings.push('dangling_separator_fragment');
        }
        if (MIXED_UGLY_RE.test(text)) {
            warnings.push('mixed_case_artifact');
        }
        if (/^(rehab|nursing|healthcare|health care|health|center|centre|care|facility|snf|ltc)$/i.test(text)) {
            warnings.push('generic_solo_label');
        }
        if (text.length > raw.length) {
            warnings.push('longer_than_raw');
        }
        if (
            level !== 'display' &&
            raw.length - text.length > 0 &&
            raw.length - text.length < THRESHOLDS.minCompactSavings
        ) {
            warnings.push('minimal_savings');
        }
        if ((level === 'short' || level === 'micro') && text.length > THRESHOLDS.compactLabelMax) {
            warnings.push('long_compact_label');
        }
        return warnings;
    }

    function isBadShortName(candidate, fullName, level, removedSuffix) {
        var short = collapseWhitespace(candidate);
        var full = collapseWhitespace(fullName);
        if (!short) {
            return true;
        }
        if (ARTIFACT_TOKEN_RE.test(short)) {
            return true;
        }
        if (DANGLING_END_RE.test(short)) {
            return true;
        }
        for (var fragIdx = 0; fragIdx < INCOMPLETE_FRAGMENT_RES.length; fragIdx++) {
            if (INCOMPLETE_FRAGMENT_RES[fragIdx].test(short)) {
                return true;
            }
        }
        if (hasDanglingSeparatorFragment(short)) {
            return true;
        }
        if (MIXED_UGLY_RE.test(short)) {
            return true;
        }
        if (/^the$/i.test(short)) {
            return true;
        }
        if (/^at\s+\S+$/i.test(short) && wordCount(short) <= 2) {
            return true;
        }
        if (/^st\.?$/i.test(short)) {
            return true;
        }
        var words = short.split(/\s+/);
        if (words.length === 1) {
            var key = soloWordKey(words[0]);
            if (GENERIC_SOLO_WORDS[key] && wordCount(full) > 1) {
                return true;
            }
            if (/\brehabilitation\b/i.test(full) && !/\brehab\b/i.test(short)) {
                var solo = words[0];
                var rehabAfterSolo = new RegExp('^' + escapeRegex(solo) + '\\s+Rehabilitation\\b', 'i');
                if (rehabAfterSolo.test(full)) {
                    return true;
                }
            }
            var healthPlaceOnly = full.match(/^(\S+)\s+(Health Care|Healthcare)\s+(Center|Centre|Cente|Cent|Facilit(?:y)?)\s*$/i);
            if (healthPlaceOnly && soloWordKey(healthPlaceOnly[1]) === soloWordKey(words[0])) {
                return true;
            }
        }
        if (PRESERVE_AT_PLACE.test(full) && !/\bat\s+/i.test(short) && wordCount(full) <= 5) {
            var atMatch = full.match(/\bat\s+(.+)$/i);
            if (atMatch && atMatch[1] && short.toLowerCase() === atMatch[1].trim().toLowerCase()) {
                return true;
            }
        }
        if (
            words.length === 1 &&
            removedSuffix &&
            wordCount(full) > 2 &&
            GENERIC_SOLO_WORDS[soloWordKey(words[0])]
        ) {
            return true;
        }
        if (
            level === 'micro' &&
            words.length === 1 &&
            wordCount(full) > 2 &&
            !PRESERVE_AT_PLACE.test(full) &&
            GENERIC_SOLO_WORDS[soloWordKey(words[0])]
        ) {
            return true;
        }
        return false;
    }

    function removeTrailingSuffix(name, suffixList) {
        var working = normalizeFacilityNameSpacing(name);
        var removedSuffix = null;
        var list = suffixList || REMOVABLE_SUFFIXES;
        var changed = true;
        while (changed) {
            changed = false;
            for (var i = 0; i < list.length; i++) {
                var suffix = list[i];
                var re = suffixPatternRegex(suffix);
                if (!re.test(working)) {
                    continue;
                }
                if (/^(SNF|LTC)$/i.test(String(suffix).trim()) && /Hospital\s+(SNF|LTC)\s*$/i.test(working)) {
                    continue;
                }
                var candidate = stripTrailingJunk(working.replace(re, ''));
                candidate = stripIncompleteLocationFragments(candidate);
                if (!candidate || candidate.length < 2) {
                    continue;
                }
                if (isBadShortName(candidate, name, 'short', suffix)) {
                    continue;
                }
                if (!preservesInstitutionalAnchor(working, candidate)) {
                    continue;
                }
                var prefixWords = wordCount(candidate);
                var eatsRehabIdentity = /^(rehabilitation|rehab)\s+(and|&)\s+/i.test(
                    normalizeFacilityNameSpacing(suffix)
                );
                if (prefixWords === 1 && eatsRehabIdentity) {
                    continue;
                }
                working = candidate;
                removedSuffix = removedSuffix || suffix;
                changed = true;
                break;
            }
        }
        return { name: working, removedSuffix: removedSuffix };
    }

    function facilityNameParts(fullName, options) {
        var opts = options || {};
        var full = collapseWhitespace(fullName);
        var normalized = stripIncompleteLocationFragments(full);
        var list =
            opts.suffixList || (opts.lightOnly ? LIGHT_REMOVABLE_SUFFIXES : REMOVABLE_SUFFIXES);
        var stripped = removeTrailingSuffix(normalized, list);
        return {
            fullName: full,
            prefix: stripped.name,
            suffix: stripped.removedSuffix || null,
        };
    }

    function finalizeLabel(text, rawFull, level, removedSuffix) {
        var working = normalizeDisplayCasing(stripTrailingJunk(stripIncompleteLocationFragments(text)), rawFull);
        if (isBadShortName(working, rawFull, level, removedSuffix)) {
            return null;
        }
        return working;
    }

    function isTruncatedPartialToken(token) {
        var t = String(token || '').toLowerCase().replace(/\./g, '');
        if (!t) {
            return false;
        }
        return /^(cente|cent|cen|ce|rehabilita|rehabilitati|nursin|facilit|childr|skil|un|hea|vill|ridg)$/.test(t);
    }

    function stripDanglingTruncatedTokens(name, rawFull) {
        var out = collapseWhitespace(name);
        var raw = collapseWhitespace(rawFull);
        if (raw.length !== 50 || wordCount(out) < 2) {
            return out;
        }
        var tokens = out.split(/\s+/);
        while (tokens.length > 2) {
            var last = tokens[tokens.length - 1];
            if (!isTruncatedPartialToken(last)) {
                break;
            }
            tokens.pop();
            out = tokens.join(' ');
        }
        return collapseWhitespace(out);
    }

    function stripTruncatedFacilityTail(name) {
        var out = collapseWhitespace(name);
        var tailRes = [
            /\s+(?:skilled\s+)?nursing\s+(?:and|&)\s+rehabilita(?:tion)?\s*$/i,
            /\s+(?:and|&)\s+rehabilita(?:tion)?\s*$/i,
            /\s+(?:and|&)\s+nursin(?:g)?\s*$/i,
            /\s+(?:rehabilitation|rehabilita)\s+(?:and|&)\s+nursing(?:\s+(?:ce|cente|cent|cen))?\s*$/i,
            /\s+(?:health\s*care|healthcare)\s+(?:and|&)\s+rehabilitation(?:\s+(?:ce|cente|cent|cen))?\s*$/i,
            /\s+for\s+medically\s+fragile\s+childr(?:en)?\s*$/i,
            /\s+(?:rehabilitation|rehabilita|rehabilitati)\s+(?:ce|cente|cent|cen)\s*$/i,
            /\s+(?:and|&)\s+nursing\s+(?:ce|cente|cent|cen|un(?:it)?)\s*$/i,
            /\s+(?:skilled\s+)?nursing\s+un(?:it)?\s*$/i,
            /\s+hospital\s+skil(?:led)?\s*$/i,
            /\s+(?:and\s+)?rehabilitati(?:on)?\s*$/i,
            /\s+(?:and\s+)?rehabilita(?:tion)?\s*$/i,
        ];
        var prev;
        do {
            prev = out;
            tailRes.forEach(function (re) {
                out = out.replace(re, '');
            });
            out = collapseWhitespace(out);
        } while (out !== prev);
        return out;
    }

    function buildShortName(fullName, level, rawForThreshold) {
        var rawCms = trimRawCmsName(rawForThreshold != null ? rawForThreshold : fullName);
        var full = processingName(fullName);
        if (!full) {
            return { text: '', removedSuffix: null, wasShortened: false, warnings: [] };
        }

        var len = rawCms.length;
        if (len <= THRESHOLDS.keepFullMax && level === 'display') {
            return { text: rawCms, removedSuffix: null, wasShortened: false, warnings: [] };
        }

        var aggressive = level === 'short' || level === 'micro' || len > THRESHOLDS.lightMax;
        var lightOnly = len <= THRESHOLDS.lightMax && len > THRESHOLDS.keepFullMax && level === 'display';

        var parts = facilityNameParts(full, {
            lightOnly: lightOnly,
            suffixList: lightOnly ? LIGHT_REMOVABLE_SUFFIXES : REMOVABLE_SUFFIXES,
        });
        var working = parts.prefix;
        var removedSuffix = parts.suffix;

        working = applyAbbreviations(working);
        working = stripTrailingJunk(stripIncompleteLocationFragments(working));
        var hadArtifact = detectCmsTruncationArtifacts(full).hadCmsTruncationArtifact;
        if (hadArtifact || rawCms.length === 50) {
            working = stripSeparatorTruncationFragments(working, rawCms, hadArtifact);
        }
        if (hadArtifact) {
            var tailStripped = stripTruncatedFacilityTail(working);
            tailStripped = stripDanglingTruncatedTokens(tailStripped, rawCms);
            tailStripped = stripSeparatorTruncationFragments(tailStripped, rawCms, hadArtifact);
            if (tailStripped && tailStripped.length >= 2 && !isBadShortName(tailStripped, full, level, removedSuffix)) {
                working = tailStripped;
            }
        }

        var finalized = finalizeLabel(working, full, level, removedSuffix);
        if (!finalized) {
            if (removedSuffix) {
                var lighter = finalizeLabel(applyAbbreviations(parts.prefix), full, level, removedSuffix);
                if (lighter) {
                    working = lighter;
                } else {
                    working = full;
                    removedSuffix = null;
                }
            } else if (lightOnly) {
                var abbrOnly = finalizeLabel(applyAbbreviations(full), full, level, null);
                working = abbrOnly || full;
            } else {
                working = full;
                removedSuffix = null;
            }
        } else {
            working = finalized;
        }

        if (working.length > THRESHOLDS.lightMax && level === 'display' && !removedSuffix) {
            var retry = removeTrailingSuffix(full, REMOVABLE_SUFFIXES);
            if (retry.removedSuffix) {
                var retryText = finalizeLabel(applyAbbreviations(retry.name), full, level, retry.removedSuffix);
                if (retryText) {
                    working = retryText;
                    removedSuffix = retry.removedSuffix;
                }
            }
        }

        var warnings = compactLabelWarnings(working, full, level);
        var cleanedTail = stripIncompleteLocationFragments(stripTrailingJunk(working));
        cleanedTail = stripSeparatorTruncationFragments(cleanedTail, rawCms, hadArtifact);
        if (cleanedTail !== working) {
            var cleanedFinal = finalizeLabel(cleanedTail, full, level, removedSuffix);
            if (cleanedFinal) {
                working = cleanedFinal;
                warnings = compactLabelWarnings(working, full, level);
            }
        }
        return {
            text: working,
            removedSuffix: removedSuffix,
            wasShortened: !!removedSuffix || wasSubstantivelyShortened(rawCms, working),
            warnings: warnings,
        };
    }

    function smartFacilityShortName(fullName, options) {
        var opts = options || {};
        var level = opts.level || 'display';
        return buildShortName(fullName, level).text;
    }

    function mergeWarnings() {
        var out = [];
        for (var i = 0; i < arguments.length; i++) {
            var arr = arguments[i] || [];
            arr.forEach(function (w) {
                if (w && out.indexOf(w) < 0) {
                    out.push(w);
                }
            });
        }
        return out;
    }

    function structuralCompactWarnings(label, rawFull, level) {
        return compactLabelWarnings(label, rawFull, level).filter(function (warning) {
            return warning !== 'minimal_savings' && warning !== 'long_compact_label';
        });
    }

    function getFacilityDisplayName(fullName, options) {
        var rawCms = trimRawCmsName(fullName);
        var full = processingName(fullName);
        var artifact = detectCmsTruncationArtifacts(full);

        var displayBuilt = buildShortName(full, 'display', rawCms);
        var shortBuilt = buildShortName(full, 'short', rawCms);
        var microBuilt = buildShortName(full, 'micro', rawCms);

        var displayName = displayBuilt.text || full;
        var shortName = shortBuilt.text || displayName;
        var microName = microBuilt.text || shortName;

        if (structuralCompactWarnings(shortName, full, 'short').length) {
            if (!structuralCompactWarnings(displayName, full, 'display').length) {
                shortName = displayName;
            } else {
                shortName = rawCms;
            }
        }
        if (structuralCompactWarnings(microName, full, 'micro').length) {
            microName = structuralCompactWarnings(shortName, full, 'short').length ? rawCms : shortName;
        }

        displayName = normalizeDisplayCasing(displayName, full);
        shortName = normalizeDisplayCasing(shortName, full);
        microName = normalizeDisplayCasing(microName, full);

        var removedSuffix =
            displayBuilt.removedSuffix || shortBuilt.removedSuffix || microBuilt.removedSuffix || null;
        var wasShortened =
            !!removedSuffix ||
            wasSubstantivelyShortened(rawCms, displayName) ||
            wasSubstantivelyShortened(rawCms, shortName) ||
            wasSubstantivelyShortened(rawCms, microName);
        var wasNormalized =
            wasLabelNormalized(rawCms, displayName) ||
            wasLabelNormalized(rawCms, shortName) ||
            wasLabelNormalized(rawCms, microName);

        var allWarnings = mergeWarnings(
            displayBuilt.warnings,
            shortBuilt.warnings,
            microBuilt.warnings,
            structuralCompactWarnings(shortName, full, 'short'),
            structuralCompactWarnings(microName, full, 'micro'),
            compactLabelWarnings(shortName, full, 'short').filter(function (w) {
                return w === 'long_compact_label';
            }),
            compactLabelWarnings(microName, full, 'micro').filter(function (w) {
                return w === 'long_compact_label';
            })
        );

        var displayWarning = allWarnings.length ? allWarnings.join('; ') : null;
        var tooltipTitle = wasShortened || artifact.hadCmsTruncationArtifact ? 'CMS name: ' + rawCms : rawCms;

        return {
            rawCmsProviderName: rawCms,
            cmsProviderNameSource: CMS_PROVIDER_NAME_SOURCE,
            fullName: rawCms,
            displayName: displayName,
            shortName: shortName,
            microName: microName,
            wasShortened: wasShortened,
            wasNormalized: wasNormalized,
            removedSuffix: removedSuffix,
            hadCmsTruncationArtifact: artifact.hadCmsTruncationArtifact,
            displayWarning: displayWarning,
            tooltipTitle: tooltipTitle,
        };
    }

    function formatProviderDisplayName(name, options) {
        var pack = getFacilityDisplayName(name, options || { level: 'display' });
        if (typeof global.capitalizeProviderName === 'function' && !lettersMostlyUpper(pack.rawCmsProviderName)) {
            return global.capitalizeProviderName(pack.displayName);
        }
        return pack.displayName;
    }

    function formatProviderCompactName(name, level) {
        return getFacilityNameForContext(name, level === 'display' ? 'compact' : level || 'short');
    }

    var FULL_NAME_CONTEXTS = {
        hero: true,
        profile: true,
        export: true,
        methodology: true,
        source: true,
        legal: true,
        citation: true,
        audit: true,
        identity: true,
    };

    function contextLevel(context) {
        var ctx = String(context || 'display').toLowerCase();
        if (ctx === 'micro') {
            return 'micro';
        }
        if (FULL_NAME_CONTEXTS[ctx]) {
            return 'display';
        }
        if (ctx === 'display') {
            return 'display';
        }
        return 'short';
    }

    function pickRawLabelForContext(pack, context) {
        var ctx = String(context || 'display').toLowerCase();
        if (FULL_NAME_CONTEXTS[ctx]) {
            return pack.rawCmsProviderName;
        }
        if (ctx === 'micro') {
            return pack.microName || pack.shortName || pack.displayName || pack.rawCmsProviderName;
        }
        if (
            ctx === 'chart' ||
            ctx === 'legend' ||
            ctx === 'table' ||
            ctx === 'comparison' ||
            ctx === 'peer' ||
            ctx === 'mobile' ||
            ctx === 'breadcrumb' ||
            ctx === 'chip' ||
            ctx === 'compact'
        ) {
            return pack.shortName || pack.displayName || pack.rawCmsProviderName;
        }
        if (ctx === 'display') {
            return pack.displayName || pack.rawCmsProviderName;
        }
        return pack.shortName || pack.displayName || pack.rawCmsProviderName;
    }

    function applyNameCasing(label, rawFull) {
        return normalizeDisplayCasing(label, rawFull);
    }

    function getFacilityNameForContext(fullName, context) {
        var pack = getFacilityDisplayName(fullName, { level: contextLevel(context) });
        var ctx = String(context || 'display').toLowerCase();
        var raw = pickRawLabelForContext(pack, context);

        if (!FULL_NAME_CONTEXTS[ctx]) {
            var compactWarnings = structuralCompactWarnings(raw, pack.rawCmsProviderName, contextLevel(context));
            if (compactWarnings.length) {
                if (!structuralCompactWarnings(pack.displayName, pack.rawCmsProviderName, 'display').length) {
                    raw = pack.displayName;
                } else {
                    raw = pack.rawCmsProviderName;
                }
            }
        }

        if (!raw || isBadShortName(raw, pack.rawCmsProviderName, context === 'micro' ? 'micro' : 'short', null)) {
            raw = pack.rawCmsProviderName;
        }

        var label = applyNameCasing(raw, pack.rawCmsProviderName);
        var wasShortened = pack.wasShortened;
        var tooltipTitle = pack.tooltipTitle;

        return {
            label: label,
            rawCmsProviderName: pack.rawCmsProviderName,
            cmsProviderNameSource: pack.cmsProviderNameSource,
            fullName: pack.rawCmsProviderName,
            displayName: pack.displayName,
            shortName: pack.shortName,
            microName: pack.microName,
            wasShortened: wasShortened,
            wasNormalized: pack.wasNormalized,
            removedSuffix: pack.removedSuffix,
            hadCmsTruncationArtifact: pack.hadCmsTruncationArtifact,
            displayWarning: pack.displayWarning,
            title: wasShortened || pack.hadCmsTruncationArtifact ? tooltipTitle : '',
            tooltipTitle: tooltipTitle,
            ariaLabel: wasShortened || pack.hadCmsTruncationArtifact ? label + '. ' + tooltipTitle : label,
        };
    }

    global.getFacilityDisplayName = getFacilityDisplayName;
    global.getFacilityNameForContext = getFacilityNameForContext;
    global.smartFacilityShortName = smartFacilityShortName;
    global.facilityNameParts = facilityNameParts;
    global.pbjFormatProviderDisplayName = formatProviderDisplayName;
    global.pbjFormatProviderCompactName = formatProviderCompactName;
    global.detectCmsTruncationArtifacts = detectCmsTruncationArtifacts;
})(typeof window !== 'undefined' ? window : this);
