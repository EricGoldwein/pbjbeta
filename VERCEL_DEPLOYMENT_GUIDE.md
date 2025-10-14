# Vercel Deployment Guide for Facility 495241 Dashboard

## Files Structure for Vercel

```
your-project/
├── api/
│   ├── index.py              # Vercel entry point
│   └── requirements.txt      # Python dependencies
├── facility_495241_flask_app.py  # Main Flask app
├── templates/
│   └── dynamic_facility_dashboard.html  # HTML template
├── facility_495241_complete_data.csv    # Data files
├── facility_495241_provider_info_data.csv
├── vercel.json              # Vercel configuration
└── requirements.txt         # Root requirements
```

## Deployment Steps

### 1. Prepare Your Files

Make sure you have all these files in your project directory:

- `facility_495241_flask_app.py` (your main Flask app)
- `templates/dynamic_facility_dashboard.html` (HTML template)
- `facility_495241_complete_data.csv` (data file)
- `facility_495241_provider_info_data.csv` (data file)
- `vercel.json` (Vercel config)
- `requirements.txt` (Python dependencies)
- `api/index.py` (Vercel entry point)
- `api/requirements.txt` (API dependencies)

### 2. Deploy to Vercel

#### Option A: Using Vercel CLI

1. Install Vercel CLI:
   ```bash
   npm i -g vercel
   ```

2. Login to Vercel:
   ```bash
   vercel login
   ```

3. Deploy:
   ```bash
   vercel
   ```

#### Option B: Using GitHub Integration

1. Push your code to a GitHub repository
2. Connect the repository to Vercel at https://vercel.com
3. Vercel will automatically deploy on every push

### 3. Environment Variables (if needed)

If you need any environment variables, add them in the Vercel dashboard:
- Go to your project settings
- Add environment variables in the "Environment Variables" section

### 4. Custom Domain (Optional)

- In Vercel dashboard, go to your project
- Go to "Domains" tab
- Add your custom domain

## Important Notes

1. **File Size Limits**: Vercel has file size limits. Your CSV files should be under 50MB total.

2. **Cold Starts**: The first request might be slow due to cold start. Subsequent requests will be faster.

3. **Data Loading**: The CSV files are loaded into memory when the app starts. This happens on each cold start.

4. **Memory Limits**: Vercel has memory limits. If your data is too large, consider:
   - Compressing the CSV files
   - Using a database instead of CSV files
   - Implementing data pagination

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all file paths are correct
2. **Memory Issues**: Reduce data size or optimize data loading
3. **Timeout Issues**: Optimize your data processing functions

### Debugging:

1. Check Vercel function logs in the dashboard
2. Use `print()` statements for debugging (they appear in logs)
3. Test locally with `vercel dev` before deploying

## Local Testing

Test your Vercel deployment locally:

```bash
# Install Vercel CLI
npm i -g vercel

# Run locally
vercel dev
```

This will simulate the Vercel environment locally.

## Performance Optimization

1. **Data Caching**: Consider implementing data caching
2. **Lazy Loading**: Load data only when needed
3. **Data Compression**: Compress CSV files if possible
4. **Database**: For large datasets, consider using a database

## Security Notes

- Never commit API keys or secrets to your repository
- Use environment variables for sensitive data
- The CSV files will be publicly accessible, so ensure they don't contain sensitive information
