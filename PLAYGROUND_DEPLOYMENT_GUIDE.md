# PBJ Playground Deployment Guide

## Quick Start

The PBJ Playground is an interactive HTML dashboard that visualizes nursing home staffing data. Here's how to get it running:

### Prerequisites
- Python 3.x installed
- All data files present in pbj-root directory

### Step 1: Copy Required Files
```bash
# Copy playground HTML and data files to PBJapp directory
Copy-Item "C:\Users\egold\PycharmProjects\pbj-root\pbj_playground.html" "C:\Users\egold\PycharmProjects\PBJapp\"
Copy-Item "C:\Users\egold\PycharmProjects\pbj-root\national_quarterly_metrics.csv" "C:\Users\egold\PycharmProjects\PBJapp\"
Copy-Item "C:\Users\egold\PycharmProjects\pbj-root\state_quarterly_metrics.csv" "C:\Users\egold\PycharmProjects\PBJapp\"
Copy-Item "C:\Users\egold\PycharmProjects\pbj-root\playground_distributions.json" "C:\Users\egold\PycharmProjects\PBJapp\"
Copy-Item "C:\Users\egold\PycharmProjects\pbj-root\quarterly_medians.json" "C:\Users\egold\PycharmProjects\PBJapp\"
```

### Step 2: Start HTTP Server
```bash
# Navigate to PBJapp directory
cd "C:\Users\egold\PycharmProjects\PBJapp"

# Start HTTP server on port 8080
python -m http.server 8080
```

### Step 3: Access Playground
Open your browser and go to:
**🌐 http://localhost:8080/pbj_playground.html**

## Important Notes

### ❌ DON'T DO THIS
- **Never open the HTML file directly** from file explorer (shows `file://` in address bar)
- This causes CORS errors because browsers block JavaScript from fetching local files

### ✅ DO THIS
- **Always access via HTTP server** (shows `http://` in address bar)
- This allows JavaScript to fetch CSV and JSON data files properly

## Troubleshooting

### CORS Errors
If you see errors like:
```
Access to fetch at 'file:///...' from origin 'null' has been blocked by CORS policy
```

**Solution:** Make sure you're accessing via HTTP URL, not opening the file directly.

### 404 File Not Found
If the playground doesn't load:

1. Verify server is running: `netstat -an | findstr :8080`
2. Check files are in PBJapp directory: `dir pbj_playground.html`
3. Restart server: `python -m http.server 8080`

### Server Won't Start
- Kill existing Python processes: `Get-Process python | Stop-Process -Force`
- Try different port: `python -m http.server 8081`

## Playground Features

The HTML playground includes:
- 📊 Interactive US state map with HPRD data
- 📈 Contract staffing distribution charts
- 🏥 RN staffing breakdowns by state
- ⏱️ Animated time-lapse across quarters
- 📱 Mobile-responsive design

## Alternative: Streamlit Version

If you prefer the Streamlit version:
```bash
cd "C:\Users\egold\PycharmProjects\pbj-root"
streamlit run PBJ_Playground.py
```

But the HTML version is recommended for better performance and no Python dependencies.

## Data Files Required

- `pbj_playground.html` - Main playground interface
- `national_quarterly_metrics.csv` - National-level data
- `state_quarterly_metrics.csv` - State-level data  
- `playground_distributions.json` - Pre-calculated distributions
- `quarterly_medians.json` - Quarterly median values

All files must be in the same directory as the HTTP server.

