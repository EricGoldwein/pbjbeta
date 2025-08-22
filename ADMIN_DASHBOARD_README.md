# PBJ Admin Dashboard

## Overview

The PBJ Admin Dashboard is a specialized analytics tool focused on administrative staffing data from the PBJ (Provider Data for Nursing Home Compare) system. This dashboard provides comprehensive analysis of administrative hours, staffing patterns, and trends at facility, state, and national levels.

## Features

### 📊 National Overview
- **Key Metrics**: Average admin hours per day, admin hours per resident day (HPRD), percent contract admin hours, and percent days with admin staff
- **Trend Analysis**: Longitudinal charts showing changes over time across all metrics
- **Days Comparison**: Visual breakdown of days with vs. without administrative staff

### 🗺️ State Comparison
- **Multi-State Analysis**: Compare administrative staffing patterns across selected states
- **Trend Comparison**: Side-by-side trend analysis for multiple states
- **Top States Ranking**: Identify states with highest admin hours and HPRD

### 🏥 Facility Search
- **Search Options**: Find facilities by name, provider number, or state
- **Individual Facility Analysis**: Detailed metrics and trends for specific facilities
- **State-Level Facility Comparison**: Top facilities within a selected state

## Data Sources

The dashboard uses processed administrative data from the standardized NonNurse PBJ files, including:

- **Hrs_Admin**: Direct administrative hours
- **Hrs_Admin_ctr**: Contract administrative hours  
- **Hrs_Admin_fn**: Administrative hours with no data flag (available in later datasets)
- **MDScensus**: Resident census data for HPRD calculations

## Key Metrics

### Total Admin Hours
- Combined direct and contract administrative hours per day
- Calculated as: `Hrs_Admin + Hrs_Admin_ctr`

### Admin Hours per Resident Day (HPRD)
- Administrative hours normalized by resident census
- Calculated as: `Total_Admin_Hours / MDScensus`

### Percent Contract Admin Hours
- Percentage of administrative hours provided by contract staff
- Calculated as: `(Hrs_Admin_ctr / Total_Admin_Hours) * 100`

### Days with vs. Without Admin
- Count and percentage of days with administrative staff present
- Helps identify staffing consistency patterns

## Data Processing

The admin data is processed using the `admin_data_processor.py` script, which:

1. **Extracts Admin Data**: Filters NonNurse files for administrative staffing records
2. **Calculates Metrics**: Computes HPRD, contract percentages, and day counts
3. **Aggregates by Level**: Creates facility, state, and national summaries
4. **Handles Data Variations**: Manages the `Hrs_Admin_fn` column availability in later datasets

## Usage

### Running the Dashboard

```bash
streamlit run Admin_Dashboard.py
```

### Data Requirements

Ensure the following files are present in the `admin_data/` directory:
- `admin_facility_metrics.csv`
- `admin_state_metrics.csv` 
- `admin_national_metrics.csv`

### Generating Data

To regenerate the admin dataset:

```bash
python admin_data_processor.py
```

## Technical Details

### Data Coverage
- **Time Period**: 2023 Q1 - 2025 Q1 (latest available data)
- **Geographic Coverage**: All 50 states plus DC
- **Facility Coverage**: All nursing homes reporting administrative staffing data

### Performance Considerations
- Data is cached using Streamlit's `@st.cache_data` decorator
- Large datasets are processed in chunks to manage memory usage
- Interactive charts use Plotly for responsive visualization

### Data Quality Notes
- `Hrs_Admin_fn` column is only available in datasets from 2023 onwards
- Missing values are handled gracefully with appropriate defaults
- Census values of 0 are handled to prevent division by zero errors

## Dashboard Navigation

### Sidebar Controls
- **Analysis Level**: Switch between National, State, and Facility views
- **Date Range**: Filter data by quarter range
- **State Selection**: Choose states for comparison (State view)
- **Facility Search**: Search by name, provider number, or state (Facility view)

### Interactive Features
- **Hover Information**: Detailed tooltips on all charts
- **Zoom and Pan**: Interactive chart controls
- **Data Filtering**: Real-time filtering based on selections
- **Export Options**: Charts can be downloaded as images

## Comparison with Main PBJ Dashboard

While the main PBJ dashboard focuses on nursing staff metrics, the Admin Dashboard provides:

- **Specialized Focus**: Administrative staffing only
- **Contract Analysis**: Detailed breakdown of contract vs. direct admin hours
- **Staffing Consistency**: Analysis of days with/without admin staff
- **Administrative HPRD**: Hours per resident day specifically for admin functions

## Future Enhancements

Potential improvements for the admin dashboard:

1. **Benchmarking**: Compare facilities against peer groups
2. **Predictive Analytics**: Forecast staffing needs based on trends
3. **Regulatory Compliance**: Track against administrative staffing requirements
4. **Cost Analysis**: Integrate with cost data for efficiency analysis
5. **Quality Correlation**: Link admin staffing to quality measures

## Support

For technical support or questions about the admin dashboard, contact 320 Consulting.

---

*Last Updated: August 2025*
