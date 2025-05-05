# PBJ Staffing Dashboard Overview

## 1. Project Purpose
The app analyzes U.S. nursing home staffing data from the CMS Payroll-Based Journal (PBJ) system and enables facility-, state-, and national-level insights, using Streamlit + Plotly. It allows users to view trends, download reports, and generate AI-assisted summaries.

## 2. Core Components
- `streamlit_app.py`: Main entry point.
- Uses `DuckDB` for fast local querying of preprocessed CSV files.
- Integrated with OpenAI Assistant via `openai.Client` for report generation.
- Outputs include tables, interactive charts, PDF summaries, and GPT-generated insights.

## 3. File Inputs
- `facility_quarterly_metrics.csv`
- `state_quarterly_metrics.csv`
- `region_quarterly_metrics.csv`
- `national_quarterly_metrics.csv`
- `NH_HealthCitations_Mar2025.csv` (citations)
- `PBJ_dailyNurseStaffing_CY2024Q3.csv`
- `PBJ_dailyNonnurseStaffing_CY2024Q3.csv`
- `NH_ProviderInfo_Mar2025.csv` (name, state, county, city)
- `Regional Coding.csv` (region mapping)

## 4. Data Coverage
- Total unique providers across all 31 PBJ files: ~15,000
- Not all providers appear in all 31 quarters
- Provider participation varies by quarter
- Data completeness should be considered when analyzing trends

## 4. Key Concepts
- **HPRD (Hours Per Resident Day)**: Central staffing metric.
- **Contract Staff %**: Indicator of reliance on external labor.
- **MDS Census**: Derived from MDS data; estimates daily residents.
- **CY_QTR**: Format = `2024Q1`; used throughout for temporal grouping.

## 5. Data Transformations
- Quarters are mapped to first-of-month datetime (`YYYY-MM-01`) for plotting.
- Derived metrics like Avg Daily Census, RN HPRD, Contract %.
- Facility filtering powered by DuckDB for performance.

## 6. AI Reporting
- GPT-based summaries (OpenAI API) for state-level analysis.
- Context passed includes staffing ratios, contract %, national benchmarks.

## 7. Known Constraints
- Smaller facilities (and states with smaller facilities) may have higher staffing levels.
- States with fewer nursing homes are more likely to have outlier data
- Staffing ratio increases in roughly 2020-22 are likely a result of a decline in resident census coinciding with covid-19 pendemic.
- Cursor should avoid compliance judgments particularly those based on staffing ratios - higher staffing ratio facilities, states, etc. are not necessarily meeting staffing standards.

## 8. Future Enhancements
- Add SFF/abuse flags to facility summaries.
- Incorporate real-time data fetching.
- Enable cross-quarter comparison views.

# PBJ Staffing Dashboard Documentation

## Data Sources

### Primary Data Sources
1. **PBJ Staffing Data**
   - Source: CMS Payroll-Based Journal (PBJ) system
   - Format: CSV files in `PBJ_Nurse/` directory
   - Key Fields: PROVNUM, WORKDATE, MDSCENSUS, various staffing hours fields
   - Update Frequency: Quarterly

2. **Facility Information**
   - Source: CMS Nursing Home Compare
   - Format: CSV files in `NH_ProviderInfo_Mar2025.csv`
   - Key Fields: PROVNUM, PROVNAME, STATE, COUNTY_NAME, CITY
   - Update Frequency: Monthly

3. **Affiliated Entity Performance Data**
   - Source: CMS Nursing Home Compare
   - Format: CSV files in `Nursing_Home_Affiliated_Entity_Performance_Measures_Mar_2025.csv`
   - Key Fields: Affiliated entity ID, various performance metrics
   - Update Frequency: Monthly
   - Relationship: Can be joined with NH_ProviderInfo_Mar2025.csv using Affiliated entity ID

### Data Relationships
- PBJ Staffing Data (PROVNUM) ↔ Facility Information (PROVNUM)
- Facility Information (Affiliated Entity ID) ↔ Affiliated Entity Performance Data (Affiliated entity ID)

## Database Schema Details

### PBJ Staffing Data Schema
```sql
-- Daily staffing records
CREATE TABLE staffing (
    PROVNUM TEXT PRIMARY KEY,        -- CMS Certification Number
    WORKDATE DATE,                   -- Date of staffing record
    CY_QTR TEXT,                     -- Calendar Year Quarter (YYYYQN)
    MDSCENSUS INTEGER,               -- Minimum Data Set Census
    -- Direct Care Hours
    HRS_RNDON FLOAT,                 -- RN Director of Nursing
    HRS_RNADMIN FLOAT,               -- RN Administrative
    HRS_RN FLOAT,                    -- RN Direct Care
    HRS_LPNADMIN FLOAT,              -- LPN Administrative
    HRS_LPN FLOAT,                   -- LPN Direct Care
    HRS_CNA FLOAT,                   -- Certified Nursing Assistant
    HRS_NATRN FLOAT,                 -- Nurse Aide in Training
    HRS_MEDAIDE FLOAT,               -- Medication Aide
    -- Contract Staff Hours
    HRS_RNDON_CTR FLOAT,             -- Contract RN Director of Nursing
    HRS_RNADMIN_CTR FLOAT,           -- Contract RN Administrative
    HRS_RN_CTR FLOAT,                -- Contract RN Direct Care
    HRS_LPNADMIN_CTR FLOAT,          -- Contract LPN Administrative
    HRS_LPN_CTR FLOAT,               -- Contract LPN Direct Care
    HRS_CNA_CTR FLOAT,               -- Contract CNA
    HRS_NATRN_CTR FLOAT,             -- Contract Nurse Aide in Training
    HRS_MEDAIDE_CTR FLOAT            -- Contract Medication Aide
);

-- Indexes for performance
CREATE INDEX idx_staffing_provnum ON staffing(PROVNUM);
CREATE INDEX idx_staffing_date ON staffing(WORKDATE);
CREATE INDEX idx_staffing_quarter ON staffing(CY_QTR);
```

### Facility Information Schema
```sql
CREATE TABLE facility_info (
    PROVNUM TEXT PRIMARY KEY,         -- CMS Certification Number
    PROVNAME TEXT,                    -- Provider Name
    STATE TEXT,                       -- State Code
    COUNTY_NAME TEXT,                 -- County Name
    CITY TEXT,                        -- City
    AFFILIATED_ENTITY_ID TEXT,        -- Parent Organization ID
    AFFILIATED_ENTITY_NAME TEXT,      -- Parent Organization Name
    OWNERSHIP_TYPE TEXT,              -- For-profit/Non-profit/Government
    BED_COUNT INTEGER,                -- Number of Beds
    OVERALL_RATING INTEGER,           -- CMS 5-star Rating
    HEALTH_INSPECTION_RATING INTEGER, -- Health Inspection Rating
    STAFFING_RATING INTEGER,          -- Staffing Rating
    QUALITY_RATING INTEGER            -- Quality Rating
);

-- Indexes
CREATE INDEX idx_facility_state ON facility_info(STATE);
CREATE INDEX idx_facility_entity ON facility_info(AFFILIATED_ENTITY_ID);
```

### Affiliated Entity Schema
```sql
CREATE TABLE affiliated_entity (
    AFFILIATED_ENTITY_ID TEXT PRIMARY KEY,
    AFFILIATED_ENTITY_NAME TEXT,
    FACILITY_COUNT INTEGER,
    STATE_COUNT INTEGER,
    SFF_COUNT INTEGER,                -- Special Focus Facility Count
    SFF_CANDIDATE_COUNT INTEGER,
    ABUSE_ICON_COUNT INTEGER,
    ABUSE_ICON_PERCENTAGE FLOAT,
    FOR_PROFIT_PERCENTAGE FLOAT,
    NON_PROFIT_PERCENTAGE FLOAT,
    GOVERNMENT_PERCENTAGE FLOAT,
    AVG_OVERALL_RATING FLOAT,
    AVG_HEALTH_INSPECTION_RATING FLOAT,
    AVG_STAFFING_RATING FLOAT,
    AVG_QUALITY_RATING FLOAT,
    AVG_TOTAL_NURSE_HPRD FLOAT,
    AVG_WEEKEND_NURSE_HPRD FLOAT,
    AVG_RN_HPRD FLOAT,
    AVG_NURSE_TURNOVER FLOAT,
    AVG_RN_TURNOVER FLOAT,
    ADMIN_TURNOVER_COUNT INTEGER,
    TOTAL_FINES INTEGER,
    AVG_FINES FLOAT,
    TOTAL_FINE_AMOUNT FLOAT,
    AVG_FINE_AMOUNT FLOAT,
    TOTAL_PAYMENT_DENIALS INTEGER,
    AVG_PAYMENT_DENIALS FLOAT
);

-- Indexes
CREATE INDEX idx_entity_name ON affiliated_entity(AFFILIATED_ENTITY_NAME);
```

## Metric Calculations

### Hours Per Resident Day (HPRD) Calculations

1. **Total Nurse HPRD**
   ```python
   # Direct care hours
   direct_care_hours = (
       HRS_RNDON + HRS_RNADMIN + HRS_RN +
       HRS_LPNADMIN + HRS_LPN +
       HRS_CNA + HRS_NATRN + HRS_MEDAIDE
   )
   
   # Contract hours
   contract_hours = (
       HRS_RNDON_CTR + HRS_RNADMIN_CTR + HRS_RN_CTR +
       HRS_LPNADMIN_CTR + HRS_LPN_CTR +
       HRS_CNA_CTR + HRS_NATRN_CTR + HRS_MEDAIDE_CTR
   )
   
   # Total hours
   total_hours = direct_care_hours + contract_hours
   
   # HPRD calculation
   total_hprd = total_hours / total_resident_days
   ```

2. **RN HPRD**
   ```python
   # RN hours (direct + contract)
   rn_hours = (
       (HRS_RNDON + HRS_RNADMIN + HRS_RN) +
       (HRS_RNDON_CTR + HRS_RNADMIN_CTR + HRS_RN_CTR)
   )
   
   rn_hprd = rn_hours / total_resident_days
   ```

3. **Contract Staff Percentage**
   ```python
   contract_percentage = (contract_hours / total_hours) * 100
   ```

4. **Average Daily Census**
   ```python
   days_in_quarter = get_days_in_quarter(quarter)
   avg_daily_census = total_resident_days / days_in_quarter
   ```

## Data Processing Pipeline

### 1. Data Ingestion
- New PBJ files placed in `PBJ_Nurse/` directory
- Facility info and affiliated entity data updated monthly
- Citations data updated monthly

### 2. Data Validation
```python
def validate_staffing_data(df):
    """Validate staffing data integrity"""
    checks = {
        'PROVNUM': lambda x: x.str.match(r'^\d{6}$'),
        'WORKDATE': lambda x: pd.to_datetime(x, errors='coerce').notna(),
        'MDSCENSUS': lambda x: x >= 0,
        'HRS_*': lambda x: x >= 0
    }
    return all(df[col].apply(check) for col, check in checks.items())
```

### 3. Data Processing Steps
1. Load raw data into DuckDB
2. Apply data validation checks
3. Calculate daily metrics
4. Aggregate to quarterly metrics
5. Generate national/state/region summaries
6. Update cache

### 4. Cache Management
```python
@st.cache_data
def load_metrics_data():
    """Load and cache metrics data"""
    return {
        'national': pd.read_csv('national_quarterly_metrics.csv'),
        'state': pd.read_csv('state_quarterly_metrics.csv'),
        'region': pd.read_csv('region_quarterly_metrics.csv'),
        'facility': pd.read_csv('facility_quarterly_metrics.csv')
    }
```

### 5. Performance Optimization
- DuckDB indexes on frequently queried columns
- Streamlit caching for expensive computations
- Lazy loading of facility details
- Batch processing for large datasets

### 6. Error Handling
```python
try:
    # Data processing
    df = process_data()
except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    log_error(e)
    return None
```

## API Integration

### OpenAI API Configuration
```python
# Environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ASSISTANT_ID = os.getenv('OPENAI_ASSISTANT_ID')

# Rate limiting
MAX_REQUESTS_PER_MINUTE = 3
REQUEST_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE
```

### Report Generation Template
```python
REPORT_TEMPLATE = """
State: {state}
Time Period: {quarter}

Key Metrics:
- Total HPRD: {total_hprd:.2f} (National: {national_total_hprd:.2f})
- RN HPRD: {rn_hprd:.2f} (National: {national_rn_hprd:.2f})
- Contract Staff %: {contract_percentage:.1f}% (National: {national_contract_percentage:.1f}%)
"""
```

## Troubleshooting Guide

### Common Issues

1. **Missing Provider Information**
   - Check provider_info_cache
   - Verify facility database connection
   - Validate PROVNUM format

2. **PDF Generation Errors**
   - Verify template formatting
   - Check data availability
   - Validate PDF library installation

3. **Performance Issues**
   - Check DuckDB indexes
   - Verify cache invalidation
   - Monitor memory usage

4. **Data Quality Issues**
   - Validate input data
   - Check metric calculations
   - Verify aggregation methods

### Data Quality Checks
```python
def validate_metrics(df):
    """Validate metric calculations"""
    checks = {
        'HPRD': lambda x: x >= 0 and x <= 24,
        'Contract %': lambda x: x >= 0 and x <= 100,
        'Census': lambda x: x >= 0
    }
    return all(df[metric].apply(check) for metric, check in checks.items())
```

## Data Update Procedures

### File Naming Conventions
- PBJ Files: `PBJ_dailyNurseStaffing_CY{YYYY}Q{N}.csv`
- Facility Info: `NH_ProviderInfo_{MMM}{YYYY}.csv`
- Citations: `NH_HealthCitations_{MMM}{YYYY}.csv`
- Affiliated Entity: `Nursing_Home_Affiliated_Entity_Performance_Measures_{MMM}{YYYY}.csv`

### Update Schedule
- PBJ Data: Quarterly (Q1, Q2, Q3, Q4)
- Facility Info: Monthly
- Citations: Monthly
- Affiliated Entity: Monthly

### Validation Steps
1. Verify file format and encoding
2. Check required columns
3. Validate data types
4. Run data quality checks
5. Update indexes and caches

### Backup Procedures
1. Create backup of existing data
2. Store backup with timestamp
3. Verify backup integrity
4. Document backup location

## Utility Functions

### format_title_case

```python
def format_title_case(text):
    if pd.isna(text):
        return 'N/A'
    words = text.split()
    formatted_words = []
    for i, word in enumerate(words):
        if i == 0 or word.lower() not in ['and', 'at', 'of', 'the', 'in', 'on', 'for', 'to', 'with', 'by']:
            formatted_words.append(word.capitalize())
        else:
            formatted_words.append(word.lower())
    return ' '.join(formatted_words)
```

- **Purpose:** Ensures provider names and city names are displayed in proper title case, with common words (e.g., 'and', 'of', 'the') in lowercase unless they are the first word.
- **Usage:** Use this function when displaying provider names and city names in the UI to maintain consistent and professional formatting.

## Coding Rules

- Always use `format_title_case` for displaying provider names and city names.
