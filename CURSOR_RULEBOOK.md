# PBJ Project Cursor Rulebook

## Code Organization

### File Structure
```
NewPBJ/
├── PBJ_lite.py              # Main dashboard application
├── generate_metrics.py      # Data processing script
├── pages/                   # Streamlit pages
│   ├── 1_Premium.py        # Premium reports page
│   └── 2_Facility_Search.py # Facility search page
├── data/                    # Data directory
│   ├── raw/                # Raw PBJ CSV files
│   └── processed/          # Processed metrics files
└── requirements.txt         # Project dependencies
```

### Naming Conventions

1. **Files**
   - Use lowercase with underscores for Python files
   - Use descriptive names that indicate purpose
   - Example: `generate_metrics.py`, `facility_search.py`

2. **Functions**
   - Use lowercase with underscores
   - Start with verb for action functions
   - Use descriptive names
   - Example: `load_metrics_data()`, `calculate_hprd()`

3. **Variables**
   - Use lowercase with underscores
   - Use descriptive names
   - Example: `total_nurse_hours`, `facility_count`

4. **Constants**
   - Use uppercase with underscores
   - Example: `MAX_FACILITIES`, `DEFAULT_QUARTER`

## Data Processing Rules

### PROVNUM Handling
- Always treat as string
- Preserve leading zeros
- Validate length (1-6 characters)
- Convert to uppercase for consistency
- Example: `"015009"` not `15009`

### Column Names
- Standardize to Q4 2024 format
- Use consistent capitalization
- Handle variations in naming
- Example: `"Total_Nurse_HPRD"` not `"total nurse hprd"`

### Calculations

1. **HPRD Calculations**
   ```python
   def calculate_hprd(hours, census):
       return hours / census if census > 0 else 0
   ```

2. **Contract Percentage**
   ```python
   def calculate_contract_percentage(contract_hours, total_hours):
       return (contract_hours / total_hours * 100) if total_hours > 0 else 0
   ```

### Data Validation
- Check for missing values
- Validate numeric ranges
- Handle division by zero
- Log warnings for data issues

## UI/UX Guidelines

### Dashboard Layout
1. **Header**
   - Clear title
   - Navigation links
   - Consistent styling

2. **Metrics Display**
   - Use metric boxes
   - Include deltas
   - Show tooltips
   - Mobile-responsive

3. **Charts**
   - Consistent color scheme
   - Clear labels
   - Interactive tooltips
   - Mobile optimization

### Mobile Optimization
1. **Layout**
   - Stack elements vertically
   - Adjust font sizes
   - Optimize touch targets

2. **Performance**
   - Lazy loading
   - Data caching
   - Efficient queries

## Error Handling

### Data Processing
```python
try:
    # Process data
except Exception as e:
    st.error(f"Error processing data: {str(e)}")
    return None
```

### UI Components
```python
try:
    # Display component
except Exception as e:
    st.error(f"Error displaying component: {str(e)}")
    return None
```

## Performance Optimization

### Data Loading
- Use caching decorators
- Implement lazy loading
- Optimize database queries

### UI Updates
- Minimize re-renders
- Cache expensive calculations
- Use efficient data structures

## Documentation

### Code Comments
- Use docstrings for functions
- Explain complex logic
- Document assumptions
- Include examples

### User Documentation
- Clear instructions
- Screenshots
- Examples
- Troubleshooting guide

## Testing

### Data Validation
- Test calculations
- Verify data integrity
- Check edge cases

### UI Testing
- Test responsive design
- Verify interactions
- Check accessibility

## Version Control

### Git Workflow
1. Create feature branch
2. Make changes
3. Test thoroughly
4. Create pull request
5. Review and merge

### Commit Messages
- Use present tense
- Be descriptive
- Reference issues

## Security

### Data Protection
- Validate user input
- Sanitize queries
- Handle sensitive data

### Access Control
- Implement authentication
- Control data access
- Log user actions

## Maintenance

### Code Review
- Check style guide
- Verify functionality
- Test edge cases
- Review performance

### Updates
- Regular dependency updates
- Security patches
- Feature enhancements
- Bug fixes 