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