# Streamlit Cloud Deployment Rules

## Package Management

### Requirements.txt Best Practices
1. Use exact versions (`==`) instead of version ranges (`>=`, `<`) for critical packages
2. Match package versions with a known working deployment
3. Keep requirements.txt minimal - only include packages actually used in the app
4. For development dependencies, use a separate dev-requirements.txt

### Python Version
1. Always specify Python version in runtime.txt
2. Use Python 3.12.x for best compatibility
3. Avoid Python 3.13.x until it's more stable
4. Match Python version with package compatibility

### Common Issues and Solutions
1. If deployment fails:
   - Check package versions against a working deployment
   - Ensure Python version compatibility
   - Remove any unused dependencies
   - Use exact versions instead of version ranges

2. If packages conflict:
   - Use a known working combination of versions
   - Check package compatibility with Python version
   - Consider using older, more stable versions

### Example Working Configuration
```txt
# runtime.txt
python-3.12.10

# requirements.txt
streamlit==1.45.1
pandas==2.2.3
plotly==6.1.1
numpy==2.2.6
duckdb==1.3.0
```

### Development vs Production
1. Use dev-requirements.txt for local development with additional tools
2. Keep requirements.txt minimal for production deployment
3. Test deployment with exact versions before using version ranges
4. Document any special deployment requirements

## Best Practices
1. Always test deployment with a minimal set of dependencies first
2. Add dependencies one at a time, testing after each addition
3. Keep track of working package combinations
4. Document any special deployment steps or requirements
5. Use version control to track changes to deployment configuration

## Streamlit UI Component Rules

### LinkColumn Limitations
1. **LinkColumn shows URLs, not display text** - The column content must be URLs, not the text you want to display
2. **No custom display text** - LinkColumn cannot show custom text while linking to different URLs
3. **Limited styling options** - LinkColumn has minimal customization for appearance
4. **Version compatibility** - Some LinkColumn parameters may not work in older Streamlit versions

### HTML Table Rendering for Clickable Links
1. **Use HTML tables for custom link text** - When you need to show custom text as clickable links
2. **Escape HTML properly** - Use `escape=False` in `to_html()` to render HTML links
3. **Add CSS styling** - Include custom CSS for professional table appearance
4. **Target="_blank"** - Use `target="_blank"` for external links to open in new tabs

### Table Implementation Patterns
```python
# ❌ WRONG: LinkColumn shows URLs, not provider names
st.dataframe(df, column_config={
    "Provider Name": st.column_config.LinkColumn("Provider Name")
})

# ✅ CORRECT: HTML table with custom link text
df['Provider Name'] = df.apply(
    lambda row: f'<a href="{url}" target="_blank">{row["Provider Name"]}</a>',
    axis=1
)
html_table = df.to_html(escape=False, classes=['dataframe'])
st.markdown(html_table, unsafe_allow_html=True)
```

### Data Type Handling
1. **Arrow serialization errors** - Convert numpy types to native Python types before display
2. **Mixed data types** - Handle columns with mixed string/numeric data carefully
3. **NaN values** - Replace NaN with appropriate display values before rendering
4. **Integer conversion** - Use `pd.to_numeric()` with `errors='coerce'` for safe conversion

### Performance Considerations
1. **Cache expensive operations** - Use `@st.cache_data` for data loading and processing
2. **Lazy rendering** - Only render tables when data is available
3. **Efficient HTML generation** - Avoid complex HTML generation in loops
4. **Memory management** - Clean up large dataframes after use 