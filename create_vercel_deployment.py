#!/usr/bin/env python3
"""
Script to create a Vercel deployment package for a specific facility
This replicates the process used for facility 495241 but uses the updated dynamic dashboard
"""

import os
import shutil
import sys
from pathlib import Path

def create_facility_vercel_package(provnum):
    """Create a complete Vercel deployment package for a facility"""
    
    provnum = str(provnum).strip().zfill(6)
    print(f"\n{'='*60}")
    print(f"Creating Vercel Deployment Package for Facility {provnum}")
    print(f"{'='*60}\n")
    
    # Step 1: Create CSV files if they don't exist
    print("Step 1: Creating facility data files...")
    csv_file = f"facility_{provnum}_complete_data.csv"
    provider_csv_file = f"facility_{provnum}_provider_info_data.csv"
    
    if not os.path.exists(csv_file):
        print(f"  Creating {csv_file}...")
        try:
            from dynamic_facility_dashboard import create_facility_complete_csv
            create_facility_complete_csv(provnum)
            if not os.path.exists(csv_file):
                print(f"  ERROR: Failed to create {csv_file}")
                return False
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            return False
    else:
        print(f"  ✓ {csv_file} already exists")
    
    if not os.path.exists(provider_csv_file):
        print(f"  Creating {provider_csv_file}...")
        try:
            from dynamic_facility_dashboard import create_facility_provider_info_csv
            data = create_facility_provider_info_csv(provnum)
            if data is not None:
                data.to_csv(provider_csv_file, index=False)
            if not os.path.exists(provider_csv_file):
                print(f"  WARNING: {provider_csv_file} not created (may not have provider info data)")
        except Exception as e:
            print(f"  WARNING: {str(e)}")
    else:
        print(f"  ✓ {provider_csv_file} already exists")
    
    # Step 2: Read the dynamic dashboard template
    print("\nStep 2: Reading dynamic dashboard source...")
    dynamic_dashboard_file = "dynamic_facility_dashboard.py"
    if not os.path.exists(dynamic_dashboard_file):
        print(f"  ERROR: {dynamic_dashboard_file} not found!")
        return False
    
    with open(dynamic_dashboard_file, 'r', encoding='utf-8') as f:
        dashboard_code = f.read()
    
    print(f"  ✓ Read {dynamic_dashboard_file}")
    
    # Step 3: Create facility-specific Flask app
    print(f"\nStep 3: Creating facility-specific Flask app...")
    flask_app_file = f"facility_{provnum}_flask_app.py"
    
    # Modify the code to be facility-specific
    lines = dashboard_code.split('\n')
    
    # Find the if __name__ == "__main__" block
    main_block_start = -1
    for i, line in enumerate(lines):
        if 'if __name__' in line and '__main__' in line:
            main_block_start = i
            break
    
    # Replace everything from the main block to the end
    if main_block_start >= 0:
        # Keep everything up to (but not including) the main block
        new_lines = lines[:main_block_start]
        
        # Find the first @app.route to insert before_request hook before it
        first_route_idx = -1
        for i, line in enumerate(new_lines):
            if line.strip().startswith('@app.route'):
                first_route_idx = i
                break
        
        # Add lazy initialization code (prevents Vercel deployment hangs)
        init_code = [
            '',
            '# Initialize data lazily (for Vercel deployment)',
            f'# Hardcoded for facility {provnum}',
            f'PROVNUM = "{provnum}"',
            '_data_initialized = False',
            '',
            'def ensure_data_loaded():',
            '    """Lazy initialization - only load data on first request"""',
            '    global _data_initialized',
            '    if not _data_initialized:',
            '        try:',
            '            print(f"Initializing facility {PROVNUM} dashboard (lazy load)...")',
            '            create_dynamic_dashboard(PROVNUM)',
            '            print(f"✅ Successfully initialized facility {PROVNUM} dashboard")',
            '            _data_initialized = True',
            '        except Exception as e:',
            '            print(f"⚠️ Error initializing facility {PROVNUM} dashboard: {e}")',
            '            import traceback',
            '            traceback.print_exc()',
            '            # Will retry on next request',
            '',
            '# Ensure data is loaded before any request',
            '@app.before_request',
            'def before_request():',
            '    ensure_data_loaded()',
            ''
        ]
        
        # Insert initialization code before first route, or at end if no route found
        if first_route_idx >= 0:
            new_lines = new_lines[:first_route_idx] + init_code + new_lines[first_route_idx:]
        else:
            new_lines.extend(init_code)
        
        # Add main block
        new_lines.extend([
            'if __name__ == "__main__":',
            '    # For local testing',
            '    ensure_data_loaded()  # Load immediately for local dev',
            '    app.run(debug=True, port=5000)'
        ])
        
        modified_code = '\n'.join(new_lines)
    else:
        # If no main block found, find first route and insert before_request hook
        lines = dashboard_code.split('\n')
        first_route_idx = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('@app.route'):
                first_route_idx = i
                break
        
        if first_route_idx >= 0:
            # Insert before_request hook before first route
            init_code = f'''
# Initialize data lazily (for Vercel deployment)
# Hardcoded for facility {provnum}
PROVNUM = "{provnum}"
_data_initialized = False

def ensure_data_loaded():
    """Lazy initialization - only load data on first request"""
    global _data_initialized
    if not _data_initialized:
        try:
            print(f"Initializing facility {{PROVNUM}} dashboard (lazy load)...")
            create_dynamic_dashboard(PROVNUM)
            print(f"✅ Successfully initialized facility {{PROVNUM}} dashboard")
            _data_initialized = True
        except Exception as e:
            print(f"⚠️ Error initializing facility {{PROVNUM}} dashboard: {{e}}")
            import traceback
            traceback.print_exc()
            # Will retry on next request

# Ensure data is loaded before any request
@app.before_request
def before_request():
    ensure_data_loaded()

'''
            new_lines = lines[:first_route_idx] + init_code.split('\n') + lines[first_route_idx:]
            new_lines.extend([
                '',
                'if __name__ == "__main__":',
                '    # For local testing',
                '    ensure_data_loaded()  # Load immediately for local dev',
                '    app.run(debug=True, port=5000)'
            ])
            modified_code = '\n'.join(new_lines)
        else:
            # Fallback: just append
            modified_code = dashboard_code + f'''

# Initialize data lazily (for Vercel deployment)
# Hardcoded for facility {provnum}
PROVNUM = "{provnum}"
_data_initialized = False

def ensure_data_loaded():
    """Lazy initialization - only load data on first request"""
    global _data_initialized
    if not _data_initialized:
        try:
            print(f"Initializing facility {{PROVNUM}} dashboard (lazy load)...")
            create_dynamic_dashboard(PROVNUM)
            print(f"✅ Successfully initialized facility {{PROVNUM}} dashboard")
            _data_initialized = True
        except Exception as e:
            print(f"⚠️ Error initializing facility {{PROVNUM}} dashboard: {{e}}")
            import traceback
            traceback.print_exc()
            # Will retry on next request

# Ensure data is loaded before any request
@app.before_request
def before_request():
    ensure_data_loaded()

if __name__ == "__main__":
    # For local testing
    ensure_data_loaded()  # Load immediately for local dev
    app.run(debug=True, port=5000)
'''
    
    # Write the facility-specific Flask app
    with open(flask_app_file, 'w', encoding='utf-8') as f:
        f.write(modified_code)
    
    print(f"  ✓ Created {flask_app_file}")
    
    # Step 4: Prepare vercel.json configuration
    print(f"\nStep 4: Preparing Vercel configuration...")
    import json
    vercel_config = {
        "version": 2,
        "builds": [
            {
                "src": flask_app_file,
                "use": "@vercel/python"
            }
        ],
        "routes": [
            {
                "src": "/(.*)",
                "dest": flask_app_file
            }
        ],
        "env": {
            "PYTHONPATH": "."
        }
    }
    
    # Step 5: Create deployment directory structure
    print(f"\nStep 5: Creating deployment directory structure...")
    deploy_dir = f"pbj320-{provnum}"
    
    # Create deployment directory if it doesn't exist
    if os.path.exists(deploy_dir):
        print(f"  ⚠ {deploy_dir} already exists, will update files...")
    else:
        os.makedirs(deploy_dir, exist_ok=True)
        print(f"  ✓ Created directory: {deploy_dir}")
    
    # Create templates subdirectory in deployment directory
    deploy_templates_dir = os.path.join(deploy_dir, "templates")
    os.makedirs(deploy_templates_dir, exist_ok=True)
    
    # List all files needed for deployment
    deployment_files = {
        flask_app_file: flask_app_file,  # source -> destination (same name)
        csv_file: csv_file,
        "templates/dynamic_facility_dashboard.html": "templates/dynamic_facility_dashboard.html"
    }
    
    if os.path.exists(provider_csv_file):
        deployment_files[provider_csv_file] = provider_csv_file
    
    # Add MACPAC standards file (try clean version first, fall back to original)
    macpac_clean = "pbj_lite/macpac_state_standards_clean.csv"
    macpac_original = "macpac/macpac_state_standards.csv"
    if os.path.exists(macpac_clean):
        deployment_files[macpac_clean] = "macpac_state_standards_clean.csv"
    elif os.path.exists(macpac_original):
        deployment_files[macpac_original] = "macpac_state_standards.csv"
    
    # Copy files to deployment directory
    print("\n  Copying files to deployment directory:")
    for src_file, dest_file in deployment_files.items():
        if os.path.exists(src_file):
            dest_path = os.path.join(deploy_dir, dest_file)
            dest_dir = os.path.dirname(dest_path)
            if dest_dir and not os.path.exists(dest_dir):
                os.makedirs(dest_dir, exist_ok=True)
            
            shutil.copy2(src_file, dest_path)
            size = os.path.getsize(src_file) / (1024 * 1024)  # Size in MB
            print(f"    ✓ {dest_file} ({size:.2f} MB)")
        else:
            print(f"    ✗ {src_file} (MISSING)")
    
    # Create vercel.json in deployment directory
    deploy_vercel_json = os.path.join(deploy_dir, "vercel.json")
    with open(deploy_vercel_json, 'w', encoding='utf-8') as f:
        json.dump(vercel_config, f, indent=2)
    print(f"    ✓ vercel.json created in {deploy_dir}")
    
    # Create requirements.txt for Flask deployment (not Streamlit)
    deploy_requirements = os.path.join(deploy_dir, "requirements.txt")
    flask_requirements = """Flask==3.0.0
pandas==2.2.3
numpy==1.26.4
python-dateutil==2.9.0
pytz==2023.3
"""
    with open(deploy_requirements, 'w', encoding='utf-8') as f:
        f.write(flask_requirements)
    print(f"    ✓ requirements.txt created in {deploy_dir}")
    
    # Step 6: Create deployment instructions
    print(f"\nStep 6: Creating deployment guide...")
    guide_file = f"DEPLOY_{provnum}_TO_VERCEL.md"
    
    guide_content = f"""# Deploy Facility {provnum} Dashboard to Vercel

## Files Created

This script has created the following files for deployment:

- `{flask_app_file}` - Facility-specific Flask app
- `{csv_file}` - Facility daily data
- `{provider_csv_file if os.path.exists(provider_csv_file) else '(optional)'}` - Provider info data
- `vercel.json` - Vercel configuration

## Deployment Steps

### Option 1: Using Vercel CLI (Recommended)

1. **Install Vercel CLI** (if not already installed):
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   vercel
   ```
   
   Follow the prompts:
   - Set up and deploy? **Yes**
   - Which scope? (select your account)
   - Link to existing project? **No**
   - Project name? **pbj320-{provnum}** (use this exact format)
   - Directory? **./** (current directory)
   - Override settings? **No**
   
   **Note:** The project will be deployed as `pbj320-{provnum}` and accessible at `https://pbj320-{provnum}.vercel.app`

4. **Production Deployment**:
   ```bash
   vercel --prod
   ```

### Option 2: Using GitHub Integration

1. **Create a new GitHub repository** (or use existing)

2. **Add files to repository**:
   ```bash
   git init
   git add {flask_app_file} {csv_file} vercel.json requirements.txt templates/
   """
    
    if os.path.exists(provider_csv_file):
        guide_content += f"   git add {provider_csv_file}\n"
    
    guide_content += f"""   git commit -m "Add facility {provnum} dashboard"
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

3. **Connect to Vercel**:
   - Go to https://vercel.com
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect settings from `vercel.json`
   - Click "Deploy"

### Option 3: Drag & Drop (Simple HTML only)

If you want a static version instead:

1. Create a static HTML file (see `create_static_dashboard_495241.py` for reference)
2. Go to https://vercel.com
3. Drag and drop the HTML file
4. Get instant URL

## File Structure for Vercel

```
your-project/
├── {flask_app_file}          # Main Flask app
├── {csv_file}                # Facility data
"""
    
    if os.path.exists(provider_csv_file):
        guide_content += f"├── {provider_csv_file}         # Provider info\n"
    
    guide_content += """├── vercel.json                  # Vercel config
├── requirements.txt            # Python dependencies
└── templates/
    └── dynamic_facility_dashboard.html  # HTML template
```

## Important Notes

1. **File Size Limits**: Vercel has file size limits. Your CSV files should be under 50MB total.

2. **Cold Starts**: The first request might be slow due to cold start. Subsequent requests will be faster.

3. **Data Loading**: The CSV files are loaded into memory when the app starts. This happens on each cold start.

4. **Memory Limits**: Vercel has memory limits. If your data is too large, consider:
   - Compressing the CSV files
   - Using a database instead of CSV files
   - Implementing data pagination

## Testing Locally

Before deploying, test locally:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Flask app
python {flask_app_file}

# Or use Vercel CLI to simulate Vercel environment
vercel dev
```

## Troubleshooting

### Common Issues:

1. **Import Errors**: Make sure all file paths are correct
2. **Memory Issues**: Reduce data size or optimize data loading
3. **Timeout Issues**: Optimize your data processing functions

### Debugging:

1. Check Vercel function logs in the dashboard
2. Use `print()` statements for debugging (they appear in logs)
3. Test locally with `vercel dev` before deploying

## Next Steps

After deployment:

1. Visit your Vercel URL: `https://pbj320-{provnum}.vercel.app`
2. Test all functionality
3. Set up custom domain (optional) in Vercel dashboard
4. Configure environment variables if needed

## Updating the Dashboard

To update with new data:

1. Regenerate CSV files:
   ```bash
   python -c "from dynamic_facility_dashboard import create_facility_complete_csv; create_facility_complete_csv('{provnum}')"
   ```

2. Redeploy to Vercel:
   ```bash
   vercel --prod
   ```

---

Generated by `create_vercel_deployment.py`
"""
    
    with open(guide_file, 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print(f"  ✓ Created {guide_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ Deployment Package Created Successfully!")
    print(f"{'='*60}\n")
    print(f"Deployment directory: {deploy_dir}/")
    print(f"All files are ready in: {deploy_dir}/")
    print(f"\nNext Steps:")
    print(f"1. Test locally: cd {deploy_dir} && python {flask_app_file}")
    print(f"2. Deploy to Vercel:")
    print(f"   - Run: deploy_to_vercel.bat (enter {provnum} when prompted)")
    print(f"   - Or manually: cd {deploy_dir} && vercel link --project=pbj320-{provnum} && vercel --prod")
    print(f"\nProject will be deployed as: pbj320-{provnum}")
    print(f"URL will be: https://pbj320-{provnum}.vercel.app\n")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python create_vercel_deployment.py <PROVNUM>")
        print("Example: python create_vercel_deployment.py 495241")
        sys.exit(1)
    
    provnum = sys.argv[1]
    success = create_facility_vercel_package(provnum)
    
    if not success:
        print("\n❌ Failed to create deployment package. Please check errors above.")
        sys.exit(1)

