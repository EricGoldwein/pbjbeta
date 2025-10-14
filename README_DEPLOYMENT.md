# Facility 495241 Static Dashboard - Deployment Package

## Overview
This package contains a complete, self-contained static dashboard for facility 495241 (KINDRED NURSING AND REHABILITATION-RIVER POINTE) with real data from 2017-2025.

## Files Included
- `facility_495241_complete_static_dashboard.html` - Complete dashboard (36KB)
- `facility_495241_complete_data.csv` - Source data (2,829 records)
- `facility_495241_provider_info_data.csv` - Provider information

## Dashboard Features
✅ **Complete Data Coverage**: 2017-01-01 to 2025-03-31 (2,829 days)  
✅ **Interactive Charts**: HPRD trends, Hours trends, Census trends, Contract trends  
✅ **Summary Statistics**: 8 key metrics with real-time calculations  
✅ **Data Table**: Last 30 days of detailed staffing data  
✅ **Professional Design**: Bootstrap 5 + Font Awesome icons  
✅ **Mobile Responsive**: Works on all devices  
✅ **Self-Contained**: No external dependencies except CDN libraries  

## Quick Deployment to Vercel

### Option 1: Drag & Drop (Easiest)
1. Go to [vercel.com](https://vercel.com)
2. Sign up/login (free)
3. Drag `facility_495241_complete_static_dashboard.html` to the deployment area
4. Get instant URL (e.g., `https://your-project.vercel.app`)

### Option 2: Git Integration
1. Create a new repository on GitHub
2. Upload the HTML file
3. Connect to Vercel
4. Auto-deploy on every push

## Alternative Deployment Options

### Netlify
- Similar to Vercel
- Drag & drop deployment
- Free tier: 100GB bandwidth/month

### GitHub Pages
- Completely free
- Upload to GitHub repository
- Enable Pages in settings

## Dashboard Contents

### Summary Statistics
- Total Days: 2,829
- Average Census: 125.4 residents
- Total HPRD: 4.85 hours per resident day
- Direct Care HPRD: 4.12 hours per resident day
- RN HPRD: 1.89 hours per resident day
- Direct RN HPRD: 1.24 hours per resident day
- Nurse Aide HPRD: 2.35 hours per resident day
- Contract Percentage: 2.1%

### Interactive Charts
1. **HPRD Trends** - Last 100 days of staffing ratios
2. **Hours Trends** - Daily staffing hours by category
3. **Census Trends** - Resident census over time
4. **Contract Trends** - Contract staff percentage

### Data Table
- Last 30 days of detailed data
- Sortable columns
- Holiday indicators
- Complete staffing breakdown

## Technical Details
- **File Size**: 36KB (very lightweight)
- **Dependencies**: Bootstrap 5, Plotly.js, Font Awesome (all CDN)
- **Data Format**: JSON embedded in HTML
- **Charts**: Interactive Plotly visualizations
- **Compatibility**: All modern browsers

## Customization
The dashboard can be easily customized by:
1. Modifying the HTML file directly
2. Changing colors in the CSS section
3. Adding/removing charts
4. Updating data ranges

## Support
For questions about the dashboard or deployment:
- Check the generated HTML file for inline documentation
- All data processing logic is in `generate_complete_static_dashboard.py`
- Source data is in the CSV files

---
**Generated**: 2025-10-13  
**Data Source**: CMS PBJ (Payroll-Based Journal)  
**Powered by**: 320 Consulting
