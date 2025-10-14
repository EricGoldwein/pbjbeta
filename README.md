# PBJ Dashboard

A comprehensive dashboard for analyzing Payroll-Based Journal (PBJ) nursing home staffing data from CMS.

## Overview

The PBJ Dashboard provides insights into nursing home staffing metrics at national, state, and facility levels. It processes quarterly PBJ data from CMS and presents key metrics including:

- Total Nurse Hours Per Resident Day (HPRD)
- Contract Staff Percentage
- Census Data
- Facility Counts

## Data Processing Pipeline

### 1. Data Collection
- Download quarterly PBJ CSV files from CMS website
- Each quarter contains ~1.3 million rows (daily data)
- Key columns include:
  - PROVNUM (6-character facility identifier)
  - Geography data (STATE, COUNTY_NAME)
  - Staffing hours (Employee and Contract)
  - MDSCENSUS

### 2. Data Standardization
- Standardize all data to Q4 2024 format
- Handle variations in:
  - Column capitalization
  - Syntax differences
  - Extra empty columns
  - Facility name changes (PROVNAME)

### 3. Metric Generation
Generate three main CSV files:

1. **National Metrics** (`national_lite_metrics.csv`)
   - Pivoted by CY_QTR
   - Weighted HPRD calculations (weights for MDSCENSUS)

2. **State Metrics** (`state_lite_metrics.csv`)
   - Pivoted by STATE
   - Weighted HPRD calculations (weights for MDSCENSUS)

3. **Facility Metrics** (`facility_lite_metrics.csv`)
   - Pivoted by PROVNUM
   - Includes all providers per quarter
   - Handles missing MDSCENSUS data

### 4. Key Calculations

#### Staffing Categories
- **Total Nurse Hours** = HRS_RNDON + HRS_RNADMIN + HRS_RN + HRS_LPNADMIN + HRS_LPN + HRS_CNA + HRS_NATRN + HRS_MEDAIDE
- **Total RN Hours** = HRS_RNDON + HRS_RNADMIN + HRS_RN
- **Total LPN Hours** = HRS_LPNADMIN + HRS_LPN
- **Nurse Aide Hours** = HRS_CNA + HRS_NATRN + HRS_MEDAIDE

#### Key Metrics
- **HPRD** = Hours Per Resident Day (staff position / MDSCENSUS)
- **Total Nurse HPRD** = Total Nurse Hours / MDSCENSUS
- **Contract Percentage** = Contract Hours / Total Nurse Hours

## PBJ Playground

For interactive data visualization and exploration, use the PBJ Playground:

**Quick Start:**
1. Copy files from `pbj-root/` to `PBJapp/` directory
2. Run: `python -m http.server 8080` 
3. Open: `http://localhost:8080/pbj_playground.html`

**Features:**
- Interactive US state maps with HPRD data
- Contract staffing distribution charts
- RN staffing breakdowns by state
- Animated time-lapse across quarters
- Mobile-responsive design

See `PLAYGROUND_DEPLOYMENT_GUIDE.md` for detailed setup instructions.

## Application Structure

### Main Components

1. **PBJ_lite.py**
   - Main dashboard application
   - Streamlit-based interface
   - Mobile-responsive design

2. **Pages**
   - Facility Search
   - Premium Reports
   - Affiliated Entities Dashboard
   - Help Documentation

### Local Development Components

**streamlit-extras/** - Custom Streamlit components
- Contains locally developed Streamlit components
- Not included in this repository (separate git repository)
- Used for enhanced UI elements and functionality
- See streamlit-extras/README.md for component documentation

### Features

- **National View**
  - Overall staffing trends
  - Facility count tracking
  - Weighted HPRD calculations

- **State View**
  - State-specific metrics
  - Comparison with national averages
  - Facility distribution

- **Facility View**
  - Individual facility metrics
  - Historical trends
  - Care Compare integration

- **Affiliated Entities View**
  - Entity-level performance metrics
  - Ownership analysis (For-Profit, Non-Profit, Government)
  - Quality ratings and compliance data
  - Facility listings with links to PBJ data
  - Risk indicators (SFF, abuse icons, fines)

### Mobile Optimization
- Responsive design
- Optimized layout for small screens
- Touch-friendly interface

## Setup and Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Place PBJ CSV files in the data directory

3. Run the data processing script:
```bash
python generate_metrics.py
```

4. Place additional data files (optional):
   - `Nursing_Home_Affiliated_Entity_Performance_Measures_<Month>_<Year>.csv` - Latest affiliated entity file (automatically detected)
   - `NH_ProviderInfo_<Month><Year>.csv` - Latest provider info file (automatically detected)

5. Launch the dashboard:
```bash
streamlit run PBJ_lite.py
```

## Notes and Considerations

1. **Data Quality**
   - Handle missing MDSCENSUS data appropriately
   - Account for facility name changes (PROVNAME)
   - Validate PROVNUM consistency

2. **Performance**
   - Optimize for large datasets
   - Implement efficient data loading
   - Cache frequently accessed data

## Contact

For custom reports or project inquiries:
- Email: eric@320insight.com 