# MACPAC State Standards Integration Summary

## Overview
Successfully integrated MACPAC (Medicaid and CHIP Payment and Access Commission) state staffing requirements data into the PBJ Nursing Home Dashboard.

## Data Source
- **Source**: MACPAC Compendium: State Policies Related to Nursing Facility Staffing
- **Published**: March 2022
- **Link**: https://www.macpac.gov/publication/state-policies-related-to-nursing-facility-staffing/

## Data Processing Steps

### 1. Excel to CSV Conversion
- Converted `macpac_state_standards.xlsx` to `macpac_state_standards.csv`
- Extracted primary columns: "State" and "Total estimated staffing requirements"

### 2. Data Cleaning and Standardization
- Created `macpac_state_standards_clean.csv` with enhanced data structure
- Handled ranges (e.g., "3.56—4.16 HPRD") by extracting minimum values
- Identified federal minimum states (0.30 HPRD)
- Created display-friendly text for dashboard integration

### 3. Data Structure
Final cleaned data includes:
- `State`: State name
- `Min_Staffing`: Minimum staffing requirement (numeric)
- `Max_Staffing`: Maximum staffing requirement for ranges
- `Value_Type`: 'single', 'range', or 'invalid'
- `Is_Federal_Minimum`: Boolean flag for federal minimum states
- `Display_Text`: Formatted text for dashboard display

## Key Findings

### State Distribution
- **Total States**: 51 (including DC)
- **Federal Minimum States**: 12 states use 0.30 HPRD
- **States with Ranges**: 6 states have range requirements
- **States with Single Values**: 45 states

### Federal Minimum States (0.30 HPRD)
Alabama, Hawaii, Kentucky, Missouri, Nebraska, Nevada, New Hampshire, North Carolina, North Dakota, South Dakota, Utah, Virginia

### States with Range Requirements
- District of Columbia: 3.56—4.16 HPRD
- Illinois: 2.56—3.86 HPRD
- Iowa: 1.76—2.06 HPRD
- Kansas: 1.91—2.06 HPRD
- Wisconsin: 2.06—3.31 HPRD
- Wyoming: 1.56—2.31 HPRD

## Dashboard Integration

### Implementation
1. **Added MACPAC data loading function** (`load_macpac_standards()`) with caching
2. **Modified state takeaway cards** to include state staffing requirements
3. **Enhanced display logic** to show federal minimum vs. state standards
4. **Updated methodology sections** across all dashboard levels
5. **Updated About page** to include MACPAC as a data source

### User Experience
- **Federal Minimum States**: Display "Federal Minimum (0.30 HPRD)"
- **State Standards**: Display "State Standard: X.XX HPRD"
- **Range Standards**: Display "State Standard: X.XX—Y.YY HPRD"
- **Integration**: Added as a new line in each state's PBJ Takeaway card

### Technical Details
- Uses `@st.cache_data` for performance optimization
- Handles missing data gracefully with fallback messages
- Case-insensitive state name matching
- Mobile-responsive display
- Fixed undefined function issues (`get_db_connection`)

## Methodology Updates

### Updated Sections
- **Facility Methodology**: Added MACPAC reference and changed icon to ⚙️
- **Entity Methodology**: Added MACPAC reference and changed icon to ⚙️
- **National Methodology**: Added MACPAC reference and changed icon to ⚙️
- **State Methodology**: Added MACPAC reference and changed icon to ⚙️

### Content Addition
All methodology sections now include:
"This dashboard uses CMS Payroll-Based Journal (PBJ) data (2017–2025), along with other public datasets (Provider Information, Affiliated Entity). **State staffing standards via MACPAC (2022).**"

### PBJ Takeaway Chip Implementation
- **Removed**: Simple line display of state standards
- **Added**: State standard chips in header section
- **Chip Formatting**:
  - Federal minimum: "State Standard: 0.30 HPRD (federal min)"
  - Range requirements: "State Standard: 3.56-4.16 HPRD"
  - Single values: "State Standard: 3.56 HPRD"
  - Missing data: "Data Not Available"

### State Rankings Table Enhancement
- **Added**: "State Min. HPRD" column as the final column
- **Column Formatting**:
  - Federal minimum: "0.30 (fed. min)"
  - Range requirements: "3.56-4.16"
  - Single values: "3.56"
  - Missing data: "N/A"
- **Styling**: Purple background (#f3e5f5) with blue text (#1976d2) for emphasis
- **Integration**: Seamlessly merged with existing rankings table

## About Page Updates

### Data Sources Section
Updated to include MACPAC as a formal data source:
- Added link to MACPAC Compendium
- Included attribution with publication year (2022)
- Integrated with existing CMS data sources

## Files Created/Modified

### New Files
- `macpac_state_standards.csv`: Initial CSV conversion
- `macpac_state_standards_clean.csv`: Final cleaned data
- `MACPAC_INTEGRATION_SUMMARY.md`: This summary document

### Modified Files
- `PBJ_Dashboard.py`: Integrated MACPAC data loading, display, and methodology updates
- `requirements.txt`: Added openpyxl dependency
- `pages/1_About.py`: Added MACPAC as data source

### Technical Fixes
- Added missing `get_db_connection()` function to resolve Pylance errors
- Updated methodology expander icons from 📊 to ⚙️
- Ensured consistent MACPAC attribution across all sections
- Implemented state standard chips in PBJ Takeaway cards
- Enhanced state rankings table with MACPAC minimum requirements

## Usage
The MACPAC state standards are now automatically displayed in each state's PBJ Takeaway card, providing users with:
- Clear indication of federal vs. state requirements
- Context for understanding staffing levels relative to standards
- Transparent source attribution to MACPAC
- Updated methodology documentation across all dashboard levels

## Data Quality Notes
- All states and DC included
- Ranges properly handled with minimum value extraction
- Federal minimum correctly identified
- Display text formatted for readability
- Source attribution maintained throughout
- Consistent methodology documentation across all dashboard sections
