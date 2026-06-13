/**
 * PBJ320 Summary modal — formatting, export bundle, structured PDF & XLSX.
 * Loaded by superdynamic_dashboard_v2.html; expects dashboard globals (Plotly, roundHalfUpDisplay, etc.).
 */
(function (global) {
    'use strict';

    function num(v) {
        if (v === null || v === undefined || v === '') return null;
        const n = parseFloat(v);
        return Number.isFinite(n) ? n : null;
    }

    function pbj320FormatHprd(v) {
        const n = num(v);
        if (n === null) return '—';
        const r = Math.round(n * 100 + 1e-10) / 100;
        return r.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    function pbj320FormatPercent(v, opts) {
        const n = num(v);
        if (n === null) return '—';
        const o = opts || {};
        const d = o.decimals != null ? o.decimals : 1;
        const r = Math.round(n * Math.pow(10, d) + 1e-10) / Math.pow(10, d);
        if (o.allowInteger && Math.abs(r - Math.round(r)) < 0.05) {
            return String(Math.round(r)) + '%';
        }
        return r.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }) + '%';
    }

    function pbj320FormatCensus(v) {
        const n = num(v);
        if (n === null) return '—';
        const r = Math.round(n * 10 + 1e-10) / 10;
        return r.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
    }

    function pbj320FormatBeds(v) {
        const n = num(v);
        if (n === null) return '—';
        return Math.round(n).toLocaleString('en-US');
    }

    function pbj320FormatHours(v) {
        const n = num(v);
        if (n === null) return '—';
        if (Math.abs(n - Math.round(n)) < 0.005) {
            return Math.round(n).toLocaleString('en-US') + ' h';
        }
        const r = Math.round(n * 10 + 1e-10) / 10;
        return r.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }) + ' h';
    }

    function pbj320FormatContractPct(v) {
        const n = num(v);
        if (n === null) return '—';
        if (Math.abs(n - Math.round(n)) < 0.05) {
            return String(Math.round(n)) + '%';
        }
        return pbj320FormatPercent(n, { decimals: 1 });
    }

    function pbj320SnapshotMetricRow(label, valueHtml, opts) {
        const o = opts || {};
        const cls =
            'pbj320-snapshot-metric-row' +
            (o.primary ? ' pbj320-snapshot-metric-row--primary' : '') +
            (o.secondary ? ' pbj320-snapshot-metric-row--secondary' : '');
        const hint = o.hint
            ? '<span class="pbj320-snapshot-metric-hint" title="' +
              String(o.hint).replace(/"/g, '&quot;') +
              '"><i class="fas fa-circle-info" aria-hidden="true"></i></span>'
            : '';
        const spark =
            o.sparkId
                ? '<div id="' +
                  String(o.sparkId).replace(/"/g, '') +
                  '" class="pbj-census-context-sparkline pbj320-snapshot-sparkline" role="img" aria-hidden="true"></div>'
                : '';
        return (
            '<div class="' +
            cls +
            '"><span class="pbj320-snapshot-metric-label">' +
            label +
            hint +
            '</span><span class="pbj320-snapshot-metric-value-cell">' +
            '<span class="pbj320-snapshot-metric-value pbj320-snapshot-mono">' +
            valueHtml +
            '</span>' +
            spark +
            '</span></div>'
        );
    }

    function pbj320SnapshotBadge(text, tone) {
        const t = tone || 'neutral';
        return '<span class="pbj320-snapshot-badge pbj320-snapshot-badge--' + t + '">' + text + '</span>';
    }

    function pbj320SnapshotTrendModeLabel(mode) {
        const m = {
            daily: 'Daily',
            monthly: 'Monthly average',
            quarterly: 'Quarterly average',
            annual: 'Annual average',
        };
        return m[mode] || 'Trend';
    }

    function pbj320SnapshotGroupedPieSlices(row) {
        const g = function (k) {
            const v = num(row && row[k]);
            return v === null ? 0 : Math.max(0, v);
        };
        const rnAdmin = g('Hrs_RNadmin') + g('Hrs_RNDON');
        const lpnAdmin = g('Hrs_LPNadmin');
        const other = g('Hrs_NAtrn') + g('Hrs_MedAide');
        const slices = [
            { label: 'RN direct', h: g('Hrs_RN'), color: '#1d4ed8' },
            { label: 'RN admin/DON', h: rnAdmin, color: '#3b82f6' },
            { label: 'LPN direct', h: g('Hrs_LPN'), color: '#6d28d9' },
            { label: 'LPN admin', h: lpnAdmin, color: '#a78bfa' },
            { label: 'CNA', h: g('Hrs_CNA'), color: '#047857' },
        ];
        if (other > 0.01) {
            slices.push({ label: 'Other aide', h: other, color: '#10b981' });
        }
        return slices.filter(function (s) {
            return s.h > 0.01;
        });
    }

    function pbj320SnapshotBuildMonthlyTrends(rows) {
        const buckets = {};
        (rows || []).forEach(function (r) {
            const iso = String((r && r.WorkDate) || '').trim().slice(0, 10);
            const m = iso.match(/^(\d{4})-(\d{2})-/);
            if (!m) return;
            const key = m[1] + '-' + m[2];
            if (!buckets[key]) {
                buckets[key] = { sumTH: 0, sumDH: 0, sumRH: 0, sumC: 0, sumCt: 0, n: 0, below: 0 };
            }
            const b = buckets[key];
            const th = num(r.Total_Staff_HPRD);
            const dh = num(r.Direct_Care_HPRD != null ? r.Direct_Care_HPRD : r.direct_care_hprd);
            const rh = num(r.Total_RN_HPRD);
            const c = num(r.MDScensus != null ? r.MDScensus : r.mdscensus);
            b.n += 1;
            if (th !== null) b.sumTH += th;
            if (dh !== null) b.sumDH += dh;
            if (rh !== null) b.sumRH += rh;
            if (c !== null) b.sumC += c;
        });
        return Object.keys(buckets)
            .sort()
            .map(function (key) {
                const b = buckets[key];
                return {
                    month: key,
                    avg_total_hprd: b.n ? b.sumTH / b.n : null,
                    avg_direct_hprd: b.n ? b.sumDH / b.n : null,
                    avg_rn_hprd: b.n ? b.sumRH / b.n : null,
                    avg_census: b.n ? b.sumC / b.n : null,
                    n_days: b.n,
                };
            });
    }

    function pbj320SnapshotPlotlyChartDataUrl(chartEl) {
        if (!chartEl || typeof Plotly === 'undefined' || !Plotly.toImage) {
            return Promise.resolve(null);
        }
        return Plotly.toImage(chartEl, { format: 'png', width: 900, height: 320, scale: 2 }).catch(function () {
            return null;
        });
    }

    let _xlsxLibsPromise = null;

    function pbjEnsureXlsxExportLibs() {
        if (global.XLSX && global.XLSX.utils) {
            return Promise.resolve(true);
        }
        if (_xlsxLibsPromise) {
            return _xlsxLibsPromise;
        }
        _xlsxLibsPromise = new Promise(function (resolve, reject) {
            const s = document.createElement('script');
            s.src = 'https://cdn.jsdelivr.net/npm/xlsx@0.18.5/dist/xlsx.full.min.js';
            s.async = true;
            s.onload = function () {
                if (global.XLSX && global.XLSX.utils) {
                    resolve(true);
                } else {
                    reject(new Error('XLSX library did not initialize'));
                }
            };
            s.onerror = function () {
                _xlsxLibsPromise = null;
                reject(new Error('Failed to load XLSX library'));
            };
            document.head.appendChild(s);
        });
        return _xlsxLibsPromise;
    }

    function pbj320SnapshotExportFilenameStem(ccn) {
        const c = String(ccn || 'facility').replace(/\D/g, '').padStart(6, '0');
        const d = new Date().toISOString().slice(0, 10);
        return { ccn: c, date: d };
    }

    function pbj320SnapshotWorkbookFromBundle(bundle) {
        const XLSX = global.XLSX;
        if (!XLSX || !bundle) {
            throw new Error('Workbook data unavailable');
        }
        const wb = XLSX.utils.book_new();
        const b = bundle;

        const readme = [
            ['PBJ320 Summary data workbook'],
            [''],
            ['Facility', b.facilityName || ''],
            ['CCN', b.ccn || ''],
            ['Audit period', b.auditPeriod || ''],
            ['Generated', b.generatedAt || ''],
            [''],
            ['This workbook mirrors the PBJ320 Summary modal for audit and client packages.'],
            ['Daily Staffing rows are limited to the selected audit range.'],
            ['Quarterly Context may include quarters outside the audit window for benchmark context.'],
            ['Sources: CMS PBJ daily nurse staffing · CMS Provider Information.'],
            ['Contact', 'eric@320insight.com · 320 Consulting'],
        ];
        XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(readme), 'README');

        const summaryRows = [['metric', 'value']];
        Object.keys(b.summaryMetrics || {}).forEach(function (k) {
            summaryRows.push([k, b.summaryMetrics[k]]);
        });
        XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(summaryRows), 'Summary');

        const dailyHeader = [
            'date',
            'census',
            'total_hprd',
            'direct_hprd',
            'rn_hprd',
            'lpn_hprd',
            'cna_hprd',
            'contract_pct',
            'rn_hours',
            'lpn_hours',
            'aide_hours',
            'total_nursing_hours',
            'benchmark_hprd',
            'below_benchmark',
            'notes',
        ];
        const dailyRows = [dailyHeader];
        (b.dailyRows || []).forEach(function (r) {
            dailyRows.push([
                r.date,
                r.census,
                r.total_hprd,
                r.direct_hprd,
                r.rn_hprd,
                r.lpn_hprd,
                r.cna_hprd,
                r.contract_pct,
                r.rn_hours,
                r.lpn_hours,
                r.aide_hours,
                r.total_nursing_hours,
                r.benchmark_hprd,
                r.below_benchmark,
                r.notes,
            ]);
        });
        XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(dailyRows), 'Daily Staffing');

        const monthHeader = [
            'month',
            'avg_total_hprd',
            'avg_direct_hprd',
            'avg_rn_hprd',
            'avg_census',
            'n_facility_days',
            'benchmark_hprd',
            'below_benchmark_days',
            'pct_below_benchmark',
        ];
        const monthRows = [monthHeader];
        (b.monthlyTrends || []).forEach(function (r) {
            monthRows.push([
                r.month,
                r.avg_total_hprd,
                r.avg_direct_hprd,
                r.avg_rn_hprd,
                r.avg_census,
                r.n_days,
                r.benchmark_hprd,
                r.below_benchmark_days,
                r.pct_below_benchmark,
            ]);
        });
        XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(monthRows), 'Monthly Trends');

        const qHeader = [
            'quarter',
            'total_hprd',
            'direct_hprd',
            'rn_hprd',
            'case_mix_expected',
            'rn_case_mix_expected',
            'harrington_expected',
            'state_benchmark',
            'delta_vs_benchmark',
        ];
        const qRows = [qHeader];
        (b.quarterlyContext || []).forEach(function (r) {
            qRows.push([
                r.quarter,
                r.total_hprd,
                r.direct_hprd,
                r.rn_hprd,
                r.case_mix_expected,
                r.rn_case_mix_expected,
                r.harrington_expected,
                r.state_benchmark,
                r.delta_vs_benchmark,
            ]);
        });
        XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(qRows), 'Quarterly Context');

        const benchHeader = ['benchmark_name', 'type', 'value', 'unit', 'geography', 'applicability', 'notes'];
        const benchRows = [benchHeader];
        (b.benchmarks || []).forEach(function (r) {
            benchRows.push([
                r.name,
                r.type,
                r.value,
                r.unit,
                r.geography,
                r.applicability,
                r.notes,
            ]);
        });
        XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(benchRows), 'Benchmarks');

        const mixHeader = ['snapshot_date', 'category', 'hours', 'share_pct', 'hprd'];
        const mixRows = [mixHeader];
        (b.staffingMix || []).forEach(function (r) {
            mixRows.push([r.snapshot_date, r.category, r.hours, r.share_pct, r.hprd]);
        });
        XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(mixRows), 'Staffing Mix');

        const metaRows = [['field', 'value']];
        Object.keys(b.metadata || {}).forEach(function (k) {
            metaRows.push([k, b.metadata[k]]);
        });
        XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(metaRows), 'Metadata');

        return wb;
    }

    async function exportPbj320SnapshotModalXlsx() {
        const bundle = global.__pbjSnapshotExportBundle;
        if (!bundle) {
            throw new Error('Open PBJ Summary and wait for data to load before exporting.');
        }
        await pbjEnsureXlsxExportLibs();
        const wb = pbj320SnapshotWorkbookFromBundle(bundle);
        const stem = pbj320SnapshotExportFilenameStem(bundle.ccn);
        global.XLSX.writeFile(wb, 'PBJ320_Summary_Data_' + stem.ccn + '_' + stem.date + '.xlsx');
    }

    function pbj320PdfAddSectionTable(doc, startY, title, rows) {
        const pageW = doc.internal.pageSize.getWidth();
        const pageH = doc.internal.pageSize.getHeight();
        let y = startY;
        if (y > pageH - 80) {
            doc.addPage();
            y = 48;
        }
        doc.setFontSize(10);
        doc.setTextColor(31, 78, 121);
        doc.text(title, 14, y);
        y += 6;
        doc.autoTable({
            startY: y,
            head: [['Metric', 'Value']],
            body: rows,
            theme: 'plain',
            styles: { fontSize: 8.5, cellPadding: 2.5 },
            headStyles: { fillColor: [241, 245, 249], textColor: [51, 65, 85], fontStyle: 'bold' },
            columnStyles: { 0: { cellWidth: 130 }, 1: { cellWidth: pageW - 170 } },
            margin: { left: 14, right: 14 },
        });
        return doc.lastAutoTable.finalY + 10;
    }

    async function exportPbj320SnapshotModalPdfStructured() {
        const bundle = global.__pbjSnapshotExportBundle;
        if (!bundle) {
            throw new Error('Open PBJ Summary and wait for data to load before exporting.');
        }
        if (typeof global.pbjEnsurePdfExportLibs === 'function') {
            await global.pbjEnsurePdfExportLibs();
        } else if (!global.jspdf || !global.jspdf.jsPDF) {
            throw new Error('PDF export libraries are not available.');
        }
        const { jsPDF } = global.jspdf;
        const doc = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'letter' });
        const pageW = doc.internal.pageSize.getWidth();
        const pageH = doc.internal.pageSize.getHeight();
        const M = global.PBJ320_EXPORT_META || { org: '320 Consulting', email: 'eric@320insight.com' };

        let y = 44;
        doc.setFillColor(41, 128, 185);
        doc.rect(0, 0, pageW, 32, 'F');
        doc.setFontSize(15);
        doc.setTextColor(255, 255, 255);
        doc.text('PBJ320 Summary', 14, 20);
        doc.setFontSize(8);
        doc.text(M.org, pageW - 14, 20, { align: 'right' });

        doc.setTextColor(44, 62, 80);
        doc.setFontSize(11);
        const facLine = doc.splitTextToSize(
            String(bundle.facilityName || '—') + (bundle.ccn ? ' · CCN ' + bundle.ccn : ''),
            pageW - 28
        );
        doc.text(facLine, 14, y);
        y += facLine.length * 13 + 4;
        doc.setFontSize(9);
        doc.setTextColor(71, 85, 105);
        if (bundle.auditPeriod) {
            doc.text('Audit period: ' + bundle.auditPeriod, 14, y);
            y += 12;
        }
        doc.setDrawColor(222, 226, 230);
        doc.line(14, y, pageW - 14, y);
        y += 12;

        y = pbj320PdfAddSectionTable(doc, y, 'Profile', bundle.pdfProfileRows || []);
        y = pbj320PdfAddSectionTable(doc, y, 'Staffing snapshot', bundle.pdfStaffingRows || []);
        y = pbj320PdfAddSectionTable(doc, y, 'Acuity & benchmarks', bundle.pdfAcuityRows || []);

        const chartEl = document.getElementById('pbj320SnapshotHprdChart');
        const chartImg = await pbj320SnapshotPlotlyChartDataUrl(chartEl);
        if (chartImg) {
            if (y > pageH - 200) {
                doc.addPage();
                y = 48;
            }
            doc.setFontSize(10);
            doc.setTextColor(31, 78, 121);
            const chartTitle = (bundle.chartMeta && bundle.chartMeta.title) || 'PBJ trends';
            const chartSub = (bundle.chartMeta && bundle.chartMeta.subtitle) || '';
            doc.text(chartTitle, 14, y);
            y += 10;
            if (chartSub) {
                doc.setFontSize(8);
                doc.setTextColor(100, 116, 139);
                doc.text(chartSub, 14, y);
                y += 10;
            }
            const imgW = pageW - 28;
            const imgH = 155;
            doc.addImage(chartImg, 'PNG', 14, y, imgW, imgH);
            y += imgH + 12;
        }

        if (bundle.pdfDayStaffingRows && bundle.pdfDayStaffingRows.length) {
            y = pbj320PdfAddSectionTable(doc, y, bundle.pdfDayStaffingTitle || 'Staffing composition', bundle.pdfDayStaffingRows);
        }

        if (bundle.pdfFindingsRows && bundle.pdfFindingsRows.length) {
            y = pbj320PdfAddSectionTable(doc, y, 'Key stats', bundle.pdfFindingsRows);
        }

        if (y > pageH - 100) {
            doc.addPage();
            y = 48;
        }
        doc.setFontSize(8);
        doc.setTextColor(71, 85, 105);
        const notes = doc.splitTextToSize(
            'Sources: CMS PBJ daily nurse staffing · CMS Provider Information. Derived for analysis—not legal or clinical advice. Case-mix and Harrington values are CMS/acuity benchmarks, not staffing mandates.',
            pageW - 28
        );
        doc.text(notes, 14, y);
        y += notes.length * 10 + 8;

        const gen = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
        doc.setFontSize(8);
        doc.setTextColor(60, 60, 75);
        doc.text('CMS public data · Generated PBJ320 Premium · ' + gen, 14, pageH - 18);
        doc.setFontSize(7);
        doc.setTextColor(120, 120, 130);
        doc.text(M.org + ' · ' + M.email, 14, pageH - 8);

        const stem = pbj320SnapshotExportFilenameStem(bundle.ccn);
        doc.save('PBJ320_Summary_' + stem.ccn + '_' + stem.date + '.pdf');
    }

    global.pbj320FormatHprd = pbj320FormatHprd;
    global.pbj320FormatPercent = pbj320FormatPercent;
    global.pbj320FormatCensus = pbj320FormatCensus;
    global.pbj320FormatBeds = pbj320FormatBeds;
    global.pbj320FormatHours = pbj320FormatHours;
    global.pbj320FormatContractPct = pbj320FormatContractPct;
    global.pbj320SnapshotMetricRow = pbj320SnapshotMetricRow;
    global.pbj320SnapshotBadge = pbj320SnapshotBadge;
    global.pbj320SnapshotTrendModeLabel = pbj320SnapshotTrendModeLabel;
    global.pbj320SnapshotGroupedPieSlices = pbj320SnapshotGroupedPieSlices;
    global.pbj320SnapshotBuildMonthlyTrends = pbj320SnapshotBuildMonthlyTrends;
    global.pbjEnsureXlsxExportLibs = pbjEnsureXlsxExportLibs;
    global.exportPbj320SnapshotModalXlsx = exportPbj320SnapshotModalXlsx;
    global.exportPbj320SnapshotModalPdfStructured = exportPbj320SnapshotModalPdfStructured;
})(typeof window !== 'undefined' ? window : globalThis);
