#!/usr/bin/env python3
"""
Generate Static HTML Dashboard for Facility 495241
Converts CSV data to embedded JavaScript for a standalone dashboard
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
import os

def process_facility_data():
    """Process the CSV files and create embedded data for the dashboard"""
    # Read the CSV files
    print("Reading CSV files...")
    daily_df = pd.read_csv('facility_495241_complete_data.csv')
    provider_df = pd.read_csv('facility_495241_provider_info_data.csv')
    # Get facility info from the data
    facility_name = daily_df['PROVNAME'].iloc[0] if 'PROVNAME' in daily_df.columns else "KINDRED NURSING AND REHABILITATION-RIVER POINTE"
    facility_city = daily_df['CITY'].iloc[0] if 'CITY' in daily_df.columns else "VIRGINIA BEACH"
    facility_state = daily_df['STATE'].iloc[0] if 'STATE' in daily_df.columns else "VA"
    facility_ccn = "495241"
    print(f"Processing data for {facility_name}")
    # Process provider info data for charts
    provider_chart_data = process_provider_charts(provider_df, daily_df)
    # Process daily data
    daily_data = process_daily_data(daily_df)
    return {
        'facility_name': facility_name,
        'facility_city': facility_city,
        'facility_state': facility_state,
        'facility_ccn': facility_ccn,
        'provider_charts': provider_chart_data,
        'daily_data': daily_data
    }

def process_provider_charts(provider_df, daily_df):
    """Process provider info data into chart format"""
    # Group by quarter and take the latest processing date per quarter
    chart_data = provider_df.dropna(subset=['quarter']).copy()
    chart_data = chart_data.sort_values('processing_date').groupby('quarter').last().reset_index()
    # Format quarter labels for x-axis (Q1 2021 instead of 2021Q1)
    chart_data['quarter_label'] = chart_data['quarter'].apply(
        lambda x: f"Q{x[-1]} {x[:4]}" if pd.notna(x) and len(str(x)) == 6 else str(x) if pd.notna(x) else None
    )
    # Add PBJ-calculated direct care values
    pbj_direct_data = []
    for quarter in chart_data['quarter']:
        pbj_quarter = daily_df[daily_df['CY_Qtr'] == quarter]
        if len(pbj_quarter) > 0:
            total_census = pbj_quarter['MDScensus'].sum()
            direct_hours = (
                pbj_quarter['Hrs_RN'].sum() + 
                pbj_quarter['Hrs_LPN'].sum() + 
                pbj_quarter['Hrs_CNA'].sum()
            )
            direct_hprd = (direct_hours / total_census) if total_census > 0 else 0
            rn_direct_hours = pbj_quarter['Hrs_RN'].sum()
            rn_direct_hprd = (rn_direct_hours / total_census) if total_census > 0 else 0
            pbj_direct_data.append({'quarter': quarter, 'pbj_direct_total': direct_hprd, 'pbj_rn_direct': rn_direct_hprd})
        else:
            pbj_direct_data.append({'quarter': quarter, 'pbj_direct_total': 0, 'pbj_rn_direct': 0})
    pbj_direct_df = pd.DataFrame(pbj_direct_data)
    chart_data = chart_data.merge(pbj_direct_df, on='quarter', how='left')
    charts = {
        'total_staffing': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'reported_total': chart_data['reported_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist(),
            'reported_direct': chart_data['pbj_direct_total'].fillna(0).tolist(),
            'case_mix_total': chart_data['case_mix_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist(),
            'adjusted_total': chart_data['adjusted_total_nurse_hrs_per_resident_per_day'].fillna(0).tolist()
        },
        'rn_staffing': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'reported_rn': chart_data['reported_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_rn_hrs_per_resident_per_day']), None).tolist(),
            'reported_rn_total': chart_data['reported_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_rn_hrs_per_resident_per_day']), None).tolist(),
            'reported_rn_direct': chart_data['pbj_rn_direct'].where(pd.notna(chart_data['pbj_rn_direct']), None).tolist(),
            'case_mix_rn': chart_data['case_mix_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_rn_hrs_per_resident_per_day']), None).tolist(),
            'adjusted_rn': chart_data['adjusted_rn_hrs_per_resident_per_day'].where(pd.notna(chart_data['adjusted_rn_hrs_per_resident_per_day']), None).tolist()
        },
        'cna_staffing': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'reported_cna': chart_data['reported_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['reported_na_hrs_per_resident_per_day']), None).tolist(),
            'case_mix_cna': chart_data['case_mix_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_na_hrs_per_resident_per_day']), None).tolist(),
            'case_mix_lpn': chart_data['case_mix_lpn_hrs_per_resident_per_day'].where(pd.notna(chart_data['case_mix_lpn_hrs_per_resident_per_day']), None).tolist(),
            'adjusted_cna': chart_data['adjusted_na_hrs_per_resident_per_day'].where(pd.notna(chart_data['adjusted_na_hrs_per_resident_per_day']), None).tolist()
        },
        'census': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'census': chart_data['avg_residents_per_day'].fillna(0).tolist()
        },
        'ratings': {
            'quarters': chart_data['quarter_label'].where(pd.notna(chart_data['quarter_label']), None).tolist(),
            'overall': chart_data['overall_rating'].fillna(0).tolist(),
            'staffing': chart_data['staffing_rating'].fillna(0).tolist(),
            'health_inspection': chart_data['health_inspection_rating'].fillna(0).tolist()
        }
    }
    def convert_nan_to_none(obj):
        if isinstance(obj, dict):
            return {key: convert_nan_to_none(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_nan_to_none(item) for item in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    return convert_nan_to_none(charts)

def process_daily_data(daily_df):
    """Process daily data for the dashboard"""
    daily_data = []
    for _, row in daily_df.iterrows():
        census = row.get('MDScensus', 0)
        total_rn_hours = row.get('Hrs_RN', 0) + row.get('Hrs_RNadmin', 0) + row.get('Hrs_RNDON', 0)
        total_lpn_hours = row.get('Hrs_LPN', 0) + row.get('Hrs_LPNadmin', 0)
        total_cna_hours = row.get('Hrs_CNA', 0) + row.get('Hrs_NAtrn', 0)
        total_nurse_hours = total_rn_hours + total_lpn_hours + total_cna_hours
        total_hprd = (total_nurse_hours / census) if census > 0 else 0
        rn_hprd = (total_rn_hours / census) if census > 0 else 0
        lpn_hprd = (total_lpn_hours / census) if census > 0 else 0
        cna_hprd = (total_cna_hours / census) if census > 0 else 0
        total_contract_hours = (
            row.get('Hrs_RN_ctr', 0) + 
            row.get('Hrs_RNadmin_ctr', 0) + 
            row.get('Hrs_RNDON_ctr', 0) +
            row.get('Hrs_LPN_ctr', 0) + 
            row.get('Hrs_LPNadmin_ctr', 0) +
            row.get('Hrs_CNA_ctr', 0) + 
            row.get('Hrs_NAtrn_ctr', 0)
        )
        contract_percentage = (total_contract_hours / total_nurse_hours * 100) if total_nurse_hours > 0 else 0
        daily_record = {
            'WorkDate': row.get('WorkDate', ''),
            'CY_Qtr': row.get('CY_Qtr', ''),
            'MDScensus': census,
            'Total_Nurse_HPRD': round(total_hprd, 2),
            'RN_HPRD': round(rn_hprd, 2),
            'LPN_HPRD': round(lpn_hprd, 2),
            'CNA_HPRD': round(cna_hprd, 2),
            'Contract_Percentage': round(contract_percentage, 1),
            'Hrs_RN': row.get('Hrs_RN', 0),
            'Hrs_LPN': row.get('Hrs_LPN', 0),
            'Hrs_CNA': row.get('Hrs_CNA', 0),
            'Hrs_RNadmin': row.get('Hrs_RNadmin', 0),
            'Hrs_LPNadmin': row.get('Hrs_LPNadmin', 0),
            'Hrs_RNDON': row.get('Hrs_RNDON', 0),
            'IsHoliday': False,
            'DayOfWeek': ''
        }
        daily_data.append(daily_record)
    return daily_data

def create_static_dashboard():
    """Create the static HTML dashboard"""
    data = process_facility_data()
    with open('templates/dynamic_facility_dashboard.html', 'r', encoding='utf-8') as f:
        template_content = f.read()
    template_content = template_content.replace('{{ facility_name }}', data['facility_name'])
    template_content = template_content.replace('{{ facility_name | title }}', data['facility_name'].title())
    script_start = template_content.find('<script>')
    if script_start != -1:
        embedded_data = f"""
        // Embedded data for facility {data['facility_ccn']}
        const FACILITY_CCN = '{data['facility_ccn']}';
        const FACILITY_NAME = '{data['facility_name']}';
        const FACILITY_CITY = '{data['facility_city']}';
        const FACILITY_STATE = '{data['facility_state']}';
        const EMBEDDED_CHART_DATA = {json.dumps(data['provider_charts'], indent=2)};
        const EMBEDDED_DAILY_DATA = {json.dumps(data['daily_data'], indent=2)};
        function loadProviderInfoData() {{
            console.log('Using embedded provider info data');
            renderProviderInfoCharts(EMBEDDED_CHART_DATA);
        }}
        function loadInitialData() {{
            console.log('Using embedded data - skipping API calls');
            const quarters = EMBEDDED_CHART_DATA.total_staffing.quarters.filter(q => q !== null);
            const quarterSelect = document.getElementById('quarterRange');
            if (quarterSelect) {{
                quarterSelect.innerHTML = '<option value="all">All Quarters</option>' + 
                    quarters.map(q => `<option value="${{q}}">${{q}}</option>`).join('');
            }}
            const dates = EMBEDDED_DAILY_DATA.map(d => d.WorkDate).filter(d => d);
            if (dates.length > 0) {{
                document.getElementById('startDate').value = dates[0];
                document.getElementById('endDate').value = dates[dates.length - 1];
            }}
            loadProviderInfoData();
        }}
        function updateAnalysis() {{
            console.log('Using embedded daily data for analysis');
            currentData = EMBEDDED_DAILY_DATA;
            renderTable();
            updateDataRangeDisplay();
            calculateAndDisplayStats();
        }}
        function loadQuarterlyData() {{
            console.log('Using embedded data for quarterly table');
            if (!EMBEDDED_DAILY_DATA || EMBEDDED_DAILY_DATA.length === 0) {{
                console.log('No embedded data available');
                return;
            }}
            const quarterlyMap = {{}};
            EMBEDDED_DAILY_DATA.forEach(record => {{
                const quarter = record.CY_Qtr;
                if (!quarterlyMap[quarter]) {{
                    quarterlyMap[quarter] = {{
                        quarter: quarter,
                        totalDays: 0,
                        totalCensus: 0,
                        totalRNHours: 0,
                        totalLPNHours: 0,
                        totalCNAHours: 0,
                        totalNurseHours: 0,
                        avgCensus: 0,
                        avgRNHPRD: 0,
                        avgLPNHPRD: 0,
                        avgCNAHPRD: 0,
                        avgTotalHPRD: 0
                    }};
                }}
                const q = quarterlyMap[quarter];
                q.totalDays++;
                q.totalCensus += record.MDScensus;
                q.totalRNHours += record.Hrs_RN;
                q.totalLPNHours += record.Hrs_LPN;
                q.totalCNAHours += record.Hrs_CNA;
                q.totalNurseHours += record.Hrs_RN + record.Hrs_LPN + record.Hrs_CNA;
            }});
            Object.values(quarterlyMap).forEach(q => {{
                q.avgCensus = q.totalCensus / q.totalDays;
                q.avgRNHPRD = (q.totalRNHours / q.totalCensus);
                q.avgLPNHPRD = (q.totalLPNHours / q.totalCensus);
                q.avgCNAHPRD = (q.totalCNAHours / q.totalCensus);
                q.avgTotalHPRD = (q.totalNurseHours / q.totalCensus);
            }});
            renderQuarterlyTable(Object.values(quarterlyMap));
        }}
        """
        template_content = template_content[:script_start + 8] + embedded_data + template_content[script_start + 8:]
    output_filename = f'facility_{data["facility_ccn"]}_static_dashboard.html'
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(template_content)
    print(f"Static dashboard created: {output_filename}")
    print(f"Facility: {data['facility_name']}")
    print(f"Location: {data['facility_city']}, {data['facility_state']}")
    print(f"CCN: {data['facility_ccn']}")
    return output_filename

if __name__ == "__main__":
    create_static_dashboard()
