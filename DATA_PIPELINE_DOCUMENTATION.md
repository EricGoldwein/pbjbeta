# PBJ Dashboard Data Pipeline Documentation

## Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Raw Data Files](#raw-data-files)
4. [Data Standardization Process](#data-standardization-process)
5. [Metrics Generation](#metrics-generation)
6. [Provider Info Processing](#provider-info-processing)
7. [Ownership & Chain Data](#ownership--chain-data)
8. [Application Integration](#application-integration)
9. [Complete Workflow](#complete-workflow)
10. [File Naming Conventions](#file-naming-conventions)

---

## Overview

The PBJ Dashboard processes Payroll-Based Journal (PBJ) staffing data from CMS to create comprehensive analytics at facility, state, regional, and national levels. The pipeline handles:

- **Daily staffing data** (nurse and non-nurse)
- **Provider information** (facility details, ratings, compliance)
- **Ownership data** (chains, affiliated entities, performance measures)

The system is designed to be **incremental** - it only processes new data, avoiding redundant work and enabling efficient updates.

---

## Directory Structure

```
PBJapp/
├── PBJcsv/                          # Raw nurse staffing CSV files
│   ├── PBJ_dailynursestaffing_CY2017Q1.csv
│   ├── PBJ_dailynursestaffing_CY2017Q2.csv
│   └── ...
│
├── NonNursecsv/                     # Raw non-nurse staffing CSV files
│   ├── PBJ_dailynonnursestaffing_CY2017Q1.csv
│   ├── PBJ_dailynonnursestaffing_CY2017Q2.csv
│   └── ...
│
├── standardized_PBJ/                # Standardized nurse staffing files
│   ├── PBJ_dailynursestaffing_CY2017Q1.csv
│   └── ...
│
├── standardized_PBJ_NonNurse/           # Standardized non-nurse staffing files
│   ├── PBJ_dailynonnursestaffing_CY2017Q1.csv
│   └── ...
│
├── provider_info/                   # Raw provider info files from CMS
│   ├── NH_ProviderInfo_Oct2025.csv
│   ├── NH_ProviderInfo_Sep2025.csv
│   └── ...
│
├── provider_info_normalized/        # Normalized provider info files
│   ├── ProviderInfoNorm_2025_10.csv
│   ├── ProviderInfoNorm_2025_09.csv
│   └── ...
│
├── ownership/                       # Ownership and chain performance files
│   ├── Nursing_Home_Chain_Performance_Measures_Nov_2025.csv
│   ├── Nursing_Home_Chain_Performance_Measures_Jul_2025.csv
│   ├── NH_Ownership_Oct2025.csv
│   └── ...
│
├── pbj_lite/                        # Lite metrics for dashboard (optional)
│   ├── facility_lite_metrics.csv
│   ├── state_lite_metrics.csv
│   └── national_lite_metrics.csv
│
├── Facility Reports/                # Individual facility complete data files
│   ├── facility_015009_complete_data.csv
│   ├── facility_495241_complete_data.csv
│   └── ...
│
└── [Root metrics files]             # Generated quarterly metrics
    ├── facility_quarterly_metrics.csv
    ├── state_quarterly_metrics.csv
    ├── national_quarterly_metrics.csv
    ├── cms_region_quarterly_metrics.csv
    ├── facility_lite_metrics.csv
    ├── state_lite_metrics.csv
    └── national_lite_metrics.csv
```

---

## Raw Data Files

### 1. Nurse Staffing Files (`PBJcsv/`)

**File Pattern:** `PBJ_dailynursestaffing_CY{YYYY}Q{Q}.csv`

**Examples:**
- `PBJ_dailynursestaffing_CY2017Q1.csv` (Q1 2017)
- `PBJ_dailynursestaffing_CY2025Q3.csv` (Q3 2025)

**Structure:**
- **Size:** ~1.3 million rows per quarter (daily data for all facilities)
- **Key Columns:**
  - `PROVNUM` - Facility identifier (6-character CCN)
  - `PROVNAME` - Facility name
  - `STATE` - State code
  - `CITY` - City name
  - `COUNTY_NAME` - County name
  - `WorkDate` - Date (YYYYMMDD format)
  - `CY_Qtr` - Calendar year quarter (e.g., "2017Q1")
  - `MDScensus` - Resident census count
  - **Staffing Hours (Employee):**
    - `Hrs_RNDON` - RN Director of Nursing
    - `Hrs_RNadmin` - RN Admin hours
    - `Hrs_RN` - RN direct care hours
    - `Hrs_LPNadmin` - LPN Admin hours
    - `Hrs_LPN` - LPN hours
    - `Hrs_CNA` - CNA hours
    - `Hrs_NAtrn` - Nurse Aide in training
    - `Hrs_MedAide` - Medication Aide hours
  - **Staffing Hours (Contract):** Same columns with `_ctr` suffix
  - **Staffing Hours (Full-time/Part-time):** Same columns with `_emp` suffix

**Notes:**
- Column names may vary by quarter (handled in standardization)
- Files may use different encodings (utf-8, latin1, cp1252, iso-8859-1)
- Special handling for 2021Q4 (removes 'incomplete' column)

### 2. Non-Nurse Staffing Files (`NonNursecsv/`)

**File Pattern:** `PBJ_dailynonnursestaffing_CY{YYYY}Q{Q}.csv`

**Examples:**
- `PBJ_dailynonnursestaffing_CY2017Q1.csv`
- `PBJ_dailynonnursestaffing_CY2025Q3.csv`

**Structure:**
- Similar structure to nurse files but with different staffing categories:
  - `Hrs_Admin` - Administrative staff
  - `Hrs_MedDir` - Medical Director
  - `Hrs_PA` - Physician Assistant
  - `Hrs_NP` - Nurse Practitioner
  - `Hrs_Pharmacist` - Pharmacist
  - `Hrs_Dietician` - Dietician
  - `Hrs_OT`, `Hrs_PT` - Occupational/Physical Therapy
  - `Hrs_QualSocWrk` - Qualified Social Worker
  - And many more...

### 3. Provider Info Files (`provider_info/`)

**File Pattern:** `NH_ProviderInfo_{Month}{Year}.csv`

**Examples:**
- `NH_ProviderInfo_Oct2025.csv`
- `NH_ProviderInfo_Sep2025.csv`

**Structure:**
- **Size:** ~15,000 rows (one per facility)
- **Key Columns:**
  - `CMS Certification Number (CCN)` - Facility identifier
  - `Provider Name` - Facility name
  - `State`, `City/Town`, `County/Parish` - Location
  - `Ownership Type` - For-Profit, Non-Profit, Government
  - `Overall Rating` - 1-5 star rating
  - `Staffing Rating` - 1-5 star rating
  - `Health Inspection Rating` - 1-5 star rating
  - `Reported Total Nurse Staffing Hours per Resident per Day`
  - `Adjusted Total Nurse Staffing Hours per Resident per Day`
  - `Chain Name`, `Chain ID` - Ownership chain information
  - `Special Focus Status` - SFF designation
  - `Number of Fines`, `Total Number of Penalties`
  - And many more quality/compliance metrics

### 4. Ownership/Chain Files (`ownership/`)

**File Pattern:** `Nursing_Home_Chain_Performance_Measures_{Month}_{Year}.csv`

**Examples:**
- `Nursing_Home_Chain_Performance_Measures_Nov_2025.csv`
- `Nursing_Home_Chain_Performance_Measures_Jul_2025.csv`

**Alternative Pattern:** `Nursing_Home_Affiliated_Entity_Performance_Measures_{Month}_{Year}.csv`

**Structure:**
- **Size:** ~600-700 rows (one per chain/entity)
- **Key Columns:**
  - `Entity_Name` - Chain/entity name
  - `Entity_ID` - Unique identifier
  - `Total_Facilities` - Number of facilities in chain
  - `Total_Beds` - Total certified beds
  - `Avg_HPRD` - Average hours per resident day
  - `Entity_Type` - For-Profit, Non-Profit, Government
  - Various performance metrics aggregated at chain level

---

## Data Standardization Process

### Step 1: Standardize Nurse Staffing Files

**Script:** `standardize_pbj_files.py`

**Process:**
1. **Scan Input Directory:** Finds all files matching `PBJcsv/PBJ_dailynursestaffing_*.csv`
2. **Check Existing Outputs:** Compares against `standardized_PBJ/` directory
3. **Process Only New Files:** Only standardizes files that don't have corresponding outputs
4. **Column Standardization:**
   - Maps column name variations to standard format (handles case, underscores, etc.)
   - Standardizes `PROVNUM` to uppercase
   - Handles special cases (e.g., removes 'incomplete' column from 2021Q4)
5. **Encoding Handling:** Tries multiple encodings (utf-8, latin1, cp1252, iso-8859-1)
6. **Validation:** Checks for expected columns and warns about structure changes
7. **Output:** Saves to `standardized_PBJ/` with same filename

**Key Features:**
- **Incremental:** Only processes new files
- **Structure Validation:** Warns about missing/unexpected columns
- **Error Handling:** Continues processing even if individual files fail

**Example Output:**
```
Found 34 input files in PBJcsv/
Found 32 already standardized files in standardized_PBJ/
Files to process: 2

Files to standardize:
  - PBJ_dailynursestaffing_CY2025Q3.csv
  - PBJ_dailynursestaffing_CY2025Q4.csv
```

### Step 2: Standardize Non-Nurse Staffing Files

**Script:** `standardize_nonnursepbj_files.py`

**Process:**
- Similar to nurse staffing but processes all unprocessed files (not just latest)
- Uses different column mapping for non-nurse categories
- Outputs to `standardized_NonNurse/` directory

---

## Metrics Generation

### Step 3: Generate Quarterly Metrics

**Script:** `generate_metrics.py`

**Process:**
1. **Load Standardized Files:** Reads all files from `standardized_PBJ/*.csv`
2. **Check Existing Metrics:** Loads `facility_quarterly_metrics.csv`, `state_quarterly_metrics.csv`, `national_quarterly_metrics.csv` if they exist
3. **Identify New Quarters:** Compares quarters in standardized files vs. existing metrics
4. **Process Only New Quarters:** Only generates metrics for quarters not already processed
5. **Calculate Metrics Using DuckDB:**
   - **Facility Level:** Aggregates daily data to quarterly metrics per facility
   - **State Level:** Aggregates facility metrics to state level
   - **National Level:** Aggregates state metrics to national level
6. **Merge with Existing:** Appends new quarters to existing metrics files
7. **Remove Duplicates:** Ensures no duplicate quarters (keeps most recent)
8. **Save:** Updates all three metrics files

**Key Calculations:**

**Total Nurse Hours:**
```
Total_Nurse_Hours = Hrs_RNDON + Hrs_RNadmin + Hrs_RN + 
                    Hrs_LPNadmin + Hrs_LPN + Hrs_CNA + 
                    Hrs_NAtrn + Hrs_MedAide
```

**Total RN Hours:**
```
Total_RN_Hours = Hrs_RNDON + Hrs_RNadmin + Hrs_RN
```

**Contract Hours:**
```
Total_Contract_Hours = Sum of all *_ctr columns
```

**HPRD (Hours Per Resident Day):**
```
Total_Nurse_HPRD = Total_Nurse_Hours / Total_Resident_Days
RN_HPRD = Total_RN_Hours / Total_Resident_Days
```

**Contract Percentage:**
```
Contract_Percentage = (Total_Contract_Hours / Total_Nurse_Hours) × 100
```

**Output Files:**
- `facility_quarterly_metrics.csv` - One row per facility per quarter
- `state_quarterly_metrics.csv` - One row per state per quarter
- `national_quarterly_metrics.csv` - One row per quarter (national totals)

**Example Output:**
```
Found 34 standardized PBJ files
Found 32 already processed quarters: ['2017Q1', '2017Q2', ..., '2025Q2']
Processing 2 new quarter(s): ['2025Q3', '2025Q4']

Processing 2025Q3...
Processing 2025Q4...

Merging new facility metrics with existing data...
Merging new state metrics with existing data...
Merging new national metrics with existing data...
```

### Step 4: Generate Regional Metrics

**Script:** `generate_region_metrics.py`

**Process:**
1. **Load State Metrics:** Reads `state_quarterly_metrics.csv`
2. **Load Region Mapping:** Reads `cms_region_state_mapping.csv` (maps states to CMS regions)
3. **Aggregate by Region:** Groups state metrics by CMS region and quarter
4. **Calculate Regional Metrics:** Weighted averages and sums
5. **Output:**
   - `cms_region_quarterly_metrics.csv` - Full regional metrics
   - `cms_region_lite_metrics.csv` - Simplified regional metrics

**Note:** This is not incremental - it regenerates all quarters from state metrics.

### Step 5: Generate Lite Metrics

**Script:** `lite_report.py`

**Process:**
1. **Load Quarterly Metrics:** Reads the three quarterly metrics files
2. **Create Simplified Versions:** Selects key columns for dashboard performance
3. **Recalculate Aggregations:** Ensures state/national metrics are properly weighted
4. **Output:**
   - `facility_lite_metrics.csv` - Simplified facility metrics
   - `state_lite_metrics.csv` - Simplified state metrics
   - `national_lite_metrics.csv` - Simplified national metrics

**Lite Metrics Columns:**
- **Facility:** `CY_Qtr`, `PROVNUM`, `PROVNAME`, `STATE`, `COUNTY_NAME`, `Total_Nurse_HPRD`, `Nurse_Care_HPRD`, `Total_RN_HPRD`, `Direct_Care_RN_HPRD`, `Contract_Percentage`, `Census`
- **State:** `CY_Qtr`, `STATE`, `Facility_Count`, `Census`, `Total_Nurse_HPRD`, `Nurse_Care_HPRD`, `Total_RN_HPRD`, `Direct_Care_RN_HPRD`, `Contract_Percentage`, `State_Census`
- **National:** `CY_Qtr`, `Facility_Count`, `Total_Nurse_HPRD`, `Nurse_Care_HPRD`, `Total_RN_HPRD`, `Direct_Care_RN_HPRD`, `Contract_Percentage`, `MDS`

**Note:** This regenerates all quarters from quarterly metrics (fast operation).

---

## Provider Info Processing

### Step 6: Normalize Provider Info Files

**Script:** `normalize_provider_info.py`

**Process:**
1. **Scan Input Directory:** Finds all files matching `provider_info/NH_ProviderInfo_*.csv`
2. **Extract Date from Filename:** Parses month/year from filename (e.g., "Oct2025")
3. **Check Existing Outputs:** Skips files already normalized
4. **Column Mapping:** Maps CMS column names to normalized format:
   - `CMS Certification Number (CCN)` → `ccn`
   - `Provider Name` → `provider_name`
   - `State` → `state`
   - `Overall Rating` → `overall_rating`
   - And 40+ more mappings...
5. **CCN Formatting:** Ensures CCN is properly formatted (zero-padded, uppercase)
6. **Output:** Saves to `provider_info_normalized/ProviderInfoNorm_{YYYY}_{MM}.csv`

**Normalized File Structure:**
- Standardized column names (lowercase with underscores)
- Consistent data types
- All expected columns present (filled with None if missing)
- Processing date included

**Usage in App:**
- Facility dashboards load provider info for facility details
- Used for facility search and filtering
- Provides ratings, compliance data, ownership information

---

## Ownership & Chain Data

### Ownership Files

**Location:** `ownership/` directory

**File Patterns:**
- `Nursing_Home_Chain_Performance_Measures_{Month}_{Year}.csv` (newer format)
- `Nursing_Home_Affiliated_Entity_Performance_Measures_{Month}_{Year}.csv` (older format)
- `NH_Ownership_{Month}{Year}.csv` (ownership details)

**Processing:**
- **No normalization required** - files are used directly by the app
- **Dynamic file discovery** - app uses `utils/file_finder.py` to find latest files
- **Quarter-based comparisons** - app can load previous quarter files for comparisons

**Usage in App:**
- **Affiliated Entities Dashboard:** Shows chain-level performance
- **Facility Details:** Links facilities to their chains
- **Ownership Analysis:** For-Profit vs. Non-Profit vs. Government comparisons
- **Chain Performance:** Aggregated metrics across all facilities in a chain

**File Finder Logic:**
- Automatically discovers latest file by parsing date from filename
- Handles both naming patterns
- Can find previous quarter files for comparisons
- Prefers March files for Q1 comparisons (as referenced in dashboard help text)

---

## Application Integration

### How Files Are Used in PBJ_Dashboard.py

#### 1. Lite Metrics Loading

**Function:** `load_metrics_data()`

**Files Used:**
- `pbj_lite/facility_lite_metrics.csv` (or root `facility_lite_metrics.csv`)
- `pbj_lite/state_lite_metrics.csv` (or root `state_lite_metrics.csv`)
- `pbj_lite/national_lite_metrics.csv` (or root `national_lite_metrics.csv`)

**Usage:**
- Main dashboard displays
- Facility search
- State comparisons
- National trends
- Charts and visualizations

**Caching:** Streamlit `@st.cache_data` with TTL for performance

#### 2. Provider Info Loading

**Function:** `load_provider_info_data()`

**Files Used:**
- Latest file from `provider_info/` or `provider_info_normalized/`
- Discovered via `utils/file_finder.py`

**Usage:**
- Facility details page
- Search functionality
- Ratings display
- Compliance information

#### 3. Ownership/Chain Data Loading

**Function:** `load_affiliated_entity_data()`

**Files Used:**
- Latest file from `ownership/` directory
- Pattern: `Nursing_Home_Chain_Performance_Measures_*.csv`
- Discovered via `utils/file_finder.py`

**Usage:**
- Affiliated Entities page
- Chain performance metrics
- Ownership type analysis
- Facility-to-chain linking

#### 4. Facility-Specific Data

**Function:** `create_facility_complete_csv(provnum)`

**Files Used:**
- All files from `standardized_PBJ/`
- Filters for specific PROVNUM
- Combines all quarters

**Output:**
- `facility_{PROVNUM}_complete_data.csv`

**Usage:**
- Individual facility dashboards
- Detailed facility analysis
- Historical trends for specific facilities

### File Discovery System

**Utility:** `utils/file_finder.py`

**Features:**
- **Automatic Date Parsing:** Extracts dates from filenames (handles various formats)
- **Latest File Discovery:** Finds most recent file by date
- **Previous File Discovery:** Finds previous quarter files for comparisons
- **Multiple Path Support:** Searches multiple directory locations
- **Pattern Matching:** Handles different naming conventions

**Functions:**
- `find_latest_provider_info()` - Latest provider info file
- `find_previous_provider_info()` - Previous quarter provider info
- `find_latest_affiliated_entity()` - Latest chain performance file
- `find_previous_affiliated_entity()` - Previous quarter chain file

---

## Complete Workflow

### Initial Setup (First Time)

1. **Add Raw Data Files:**
   - Place PBJ CSV files in `PBJcsv/`
   - Place non-nurse files in `NonNursecsv/`
   - Place provider info files in `provider_info/`
   - Place ownership files in `ownership/`

2. **Standardize Staffing Files:**
   ```bash
   python standardize_pbj_files.py
   python standardize_nonnursepbj_files.py
   ```

3. **Generate Metrics:**
   ```bash
   python generate_metrics.py
   ```

4. **Generate Regional Metrics:**
   ```bash
   python generate_region_metrics.py
   ```

5. **Generate Lite Metrics:**
   ```bash
   python lite_report.py
   ```

6. **Normalize Provider Info:**
   ```bash
   python normalize_provider_info.py
   ```

### Incremental Updates (Adding New Quarters)

1. **Add New Quarterly Files:**
   - Add new `PBJ_dailynursestaffing_CY{YYYY}Q{Q}.csv` to `PBJcsv/`
   - Add new `PBJ_dailynonnursestaffing_CY{YYYY}Q{Q}.csv` to `NonNursecsv/` (if available)

2. **Standardize (Only New Files):**
   ```bash
   python standardize_pbj_files.py
   # Only processes files that don't have outputs yet
   ```

3. **Generate Metrics (Only New Quarters):**
   ```bash
   python generate_metrics.py
   # Only processes quarters not already in metrics files
   ```

4. **Regenerate Regional Metrics:**
   ```bash
   python generate_region_metrics.py
   # Regenerates all quarters from state metrics
   ```

5. **Regenerate Lite Metrics:**
   ```bash
   python lite_report.py
   # Regenerates all quarters from quarterly metrics
   ```

6. **Update Provider Info (If New File Available):**
   ```bash
   python normalize_provider_info.py
   # Only processes new files
   ```

7. **Update Ownership Files:**
   - Simply add new file to `ownership/` directory
   - App will automatically discover it via file finder

### Creating Facility-Specific Files

**For Individual Facility Dashboards:**
```bash
python create_facility_csv.py <PROVNUM>
# Example: python create_facility_csv.py 495241
```

**Output:** `facility_{PROVNUM}_complete_data.csv`

---

## File Naming Conventions

### Input Files (Raw Data)

| File Type | Pattern | Example |
|-----------|---------|---------|
| Nurse Staffing | `PBJ_dailynursestaffing_CY{YYYY}Q{Q}.csv` | `PBJ_dailynursestaffing_CY2025Q3.csv` |
| Non-Nurse Staffing | `PBJ_dailynonnursestaffing_CY{YYYY}Q{Q}.csv` | `PBJ_dailynonnursestaffing_CY2025Q3.csv` |
| Provider Info | `NH_ProviderInfo_{Month}{Year}.csv` | `NH_ProviderInfo_Oct2025.csv` |
| Chain Performance | `Nursing_Home_Chain_Performance_Measures_{Month}_{Year}.csv` | `Nursing_Home_Chain_Performance_Measures_Nov_2025.csv` |
| Ownership | `NH_Ownership_{Month}{Year}.csv` | `NH_Ownership_Oct2025.csv` |

### Standardized Files

| File Type | Pattern | Example |
|-----------|---------|---------|
| Standardized Nurse | Same as input (in `standardized_PBJ/`) | `PBJ_dailynursestaffing_CY2025Q3.csv` |
| Standardized Non-Nurse | Same as input (in `standardized_NonNurse/`) | `PBJ_dailynonnursestaffing_CY2025Q3.csv` |
| Normalized Provider Info | `ProviderInfoNorm_{YYYY}_{MM}.csv` | `ProviderInfoNorm_2025_10.csv` |

### Generated Metrics Files

| File Type | Filename | Description |
|-----------|----------|-------------|
| Facility Metrics | `facility_quarterly_metrics.csv` | Full facility-level metrics |
| State Metrics | `state_quarterly_metrics.csv` | Full state-level metrics |
| National Metrics | `national_quarterly_metrics.csv` | Full national-level metrics |
| Regional Metrics | `cms_region_quarterly_metrics.csv` | Full CMS region metrics |
| Regional Lite | `cms_region_lite_metrics.csv` | Simplified regional metrics |
| Facility Lite | `facility_lite_metrics.csv` | Simplified facility metrics |
| State Lite | `state_lite_metrics.csv` | Simplified state metrics |
| National Lite | `national_lite_metrics.csv` | Simplified national metrics |
| Facility Complete | `facility_{PROVNUM}_complete_data.csv` | All quarters for one facility |

### Quarter Format

- **Format:** `{YYYY}Q{Q}`
- **Examples:** `2017Q1`, `2025Q3`, `2025Q4`
- **Used in:** CY_Qtr column, filenames, quarter identification

### Date Formats in Filenames

- **Provider Info:** `{Month}{Year}` (e.g., `Oct2025`, `Sep2025`)
- **Chain Performance:** `{Month}_{Year}` (e.g., `Nov_2025`, `Jul_2025`)
- **Normalized Provider Info:** `{YYYY}_{MM}` (e.g., `2025_10`, `2025_09`)

---

## Key Features

### Incremental Processing

- **Standardization:** Only processes files without corresponding outputs
- **Metrics Generation:** Only processes quarters not already in metrics files
- **Efficient Updates:** Avoids redundant processing of existing data

### Structure Validation

- **Column Checking:** Validates expected columns are present
- **Warning System:** Alerts about missing or unexpected columns
- **File Structure Changes:** Detects and warns about schema changes

### Error Handling

- **Graceful Failures:** Continues processing even if individual files fail
- **Encoding Fallbacks:** Tries multiple encodings automatically
- **Detailed Logging:** Reports what was processed, skipped, or failed

### Data Integrity

- **Duplicate Prevention:** Removes duplicate quarters when re-processing
- **Data Merging:** Properly merges new data with existing data
- **Validation:** Checks data structure before processing

---

## Performance Considerations

### File Sizes

- **Raw PBJ Files:** ~1.3 million rows per quarter (~100-200 MB)
- **Standardized Files:** Similar size (UTF-8 encoding)
- **Quarterly Metrics:** ~15,000 facilities × quarters (~10-20 MB)
- **Lite Metrics:** Smaller subset (~5-10 MB)

### Processing Time

- **Standardization:** ~30-60 seconds per file
- **Metrics Generation:** ~2-5 minutes per quarter (using DuckDB)
- **Lite Metrics:** ~10-30 seconds (regenerates all quarters)
- **Regional Metrics:** ~30-60 seconds (regenerates all quarters)

### Optimization

- **DuckDB:** Fast columnar processing for metrics generation
- **Incremental Processing:** Only processes new data
- **Caching:** Streamlit caches loaded data
- **Lite Metrics:** Reduced columns for faster dashboard loading

---

## Troubleshooting

### Common Issues

1. **Missing Columns Warning:**
   - **Cause:** CMS changed file structure
   - **Solution:** Review warnings, update column mappings if needed

2. **Encoding Errors:**
   - **Cause:** File uses unexpected encoding
   - **Solution:** Script tries multiple encodings automatically

3. **Duplicate Quarters:**
   - **Cause:** Re-processing same quarter
   - **Solution:** Script automatically removes duplicates (keeps most recent)

4. **File Not Found:**
   - **Cause:** File not in expected location
   - **Solution:** Check directory structure, verify file naming

5. **Metrics Not Updating:**
   - **Cause:** Quarter already processed
   - **Solution:** Check existing metrics files, delete quarter if re-processing needed

---

## Maintenance

### Regular Tasks

1. **Monthly:** Add new quarterly PBJ files when released by CMS
2. **Quarterly:** Update provider info and ownership files
3. **As Needed:** Review warnings about file structure changes
4. **Periodically:** Verify data integrity and check for anomalies

### Backup Recommendations

- Backup standardized files before major updates
- Keep raw files as source of truth
- Version control metrics files for rollback capability

---

## Summary

The PBJ Dashboard data pipeline is a comprehensive, incremental system that:

1. **Standardizes** raw CMS data files
2. **Generates** aggregated metrics at multiple levels
3. **Normalizes** provider information
4. **Integrates** ownership and chain data
5. **Serves** data to the dashboard application

The system is designed for efficiency, reliability, and maintainability, with built-in validation, error handling, and incremental processing capabilities.
