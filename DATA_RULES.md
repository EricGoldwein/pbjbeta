# PBJ Data Rules

## Data Availability Rules

### NEVER use fallback data
- If data is not available, show "N/A" or "Data Unavailable"
- Do not create fake/sample data
- Do not use placeholder values
- Always be transparent about data limitations

### Data Quality Standards
- Use real PBJ data from `pbj_lite/facility_lite_metrics.csv`
- Validate data exists before displaying
- Show loading states while data is being processed
- Handle missing data gracefully with clear messaging

### Error Handling
- If data file is missing: Show "Data file not found"
- If data is corrupted: Show "Data processing error"
- If no data points: Show "No data available for this analysis"
- Never silently fail or show misleading information

### Examples of Good vs Bad
❌ **BAD**: Using fake data when real data unavailable
❌ **BAD**: Showing "3.48 HPRD" without data source
❌ **BAD**: Silent fallback to sample data

✅ **GOOD**: "HPRD Mean: N/A (Data unavailable)"
✅ **GOOD**: "Data file not found - please check pbj_lite/facility_lite_metrics.csv"
✅ **GOOD**: Clear error messages and loading states
