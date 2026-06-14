#!/usr/bin/env python3
"""
Script to create a Vercel deployment package for a specific facility
This replicates the process used for facility 495241 but uses the updated dynamic dashboard
"""

import os
import shutil
import sys
import time
from pathlib import Path
from file_path_utils import (
    find_facility_complete_data,
    find_facility_citations,
    find_facility_nonnurse_daily,
    find_facility_provider_info,
    find_facility_flask_app,
    get_facility_folder,
)

def _project_root():
    """Project root = directory containing this script. Use for resolving paths when cwd may differ (e.g. Flask)."""
    return os.path.dirname(os.path.abspath(__file__))

def create_facility_vercel_package(
    provnum,
    project_root=None,
    ein_mode="all",
    ein_selected_quarters=None,
    *,
    include_entity_longitudinal: bool = True,
):
    """Create a complete Vercel deployment package for a facility.
    CCN (provnum) is the only facility-specific input; all paths, filenames, and generated
    app content are derived from it—do not hardcode a 6-digit CCN.
    project_root: if provided (e.g. by dashboard), use this instead of __file__ so paths work when cwd differs.
    include_entity_longitudinal: when False, omit chain longitudinal slice/lookup files (smaller deploy; no entity trends section data).

    The facility Flask module is derived from ``dynamic_facility_dashboard.py`` (Step 2–3), so each
    generated package automatically picks up dashboard features such as ``geo_rollup_series`` /
    ``lite_hprd_rollup_series_by_quarter`` and the matching ``dynamic_facility_dashboard.html`` /
    ``pbj320-export-page`` JSON used for the geographic rollup panel (``pbjGeoRollupRefresh``)."""
    
    provnum = str(provnum).strip().zfill(6)
    print(f"\n{'='*60}")
    print(f"Creating Vercel Deployment Package for Facility {provnum}")
    print(f"{'='*60}\n")
    
    root = os.path.abspath(project_root) if project_root else _project_root()
    use_central_csv_host = bool(os.getenv("PBJ_CSV_HOST_BASE_URL", "").strip())
    
    ein_mode = str(ein_mode or "all").strip().lower()
    if ein_mode not in {"all", "selected", "none"}:
        ein_mode = "all"
    quarters = [str(q).strip() for q in (ein_selected_quarters or []) if str(q).strip()]
    quarters_csv = ",".join(quarters)

    # Get facility folder (will be created if needed)
    facility_folder = get_facility_folder(provnum)
    
    # Step 1: Create or incrementally update CSV files
    print("Step 1: Creating/updating facility data files...")
    csv_file = find_facility_complete_data(provnum)
    provider_csv_file = find_facility_provider_info(provnum)
    csv_filename = f"facility_{provnum}_complete_data.csv"
    provider_filename = f"facility_{provnum}_provider_info_data.csv"
    if not csv_file:
        csv_file = str(facility_folder / csv_filename)
    if not provider_csv_file:
        provider_csv_file = str(facility_folder / provider_filename)
    # Resolve paths relative to project root so they work when run from dashboard (cwd may not be project root)
    csv_abs = os.path.normpath(os.path.join(root, csv_file)) if not os.path.isabs(csv_file) else csv_file
    provider_abs = os.path.normpath(os.path.join(root, provider_csv_file)) if not os.path.isabs(provider_csv_file) else provider_csv_file

    from dynamic_facility_dashboard import create_facility_complete_csv, create_facility_provider_info_csv

    # Complete data: full build only if file missing; if file exists, incremental update only (never full rebuild)
    if not os.path.exists(csv_abs):
        print(f"  Creating {csv_filename} (full build)...")
        try:
            create_facility_complete_csv(provnum, output_path=csv_abs)
            if not os.path.exists(csv_abs):
                root_csv = os.path.join(root, f"facility_{provnum}_complete_data.csv")
                if os.path.exists(root_csv):
                    shutil.move(root_csv, csv_abs)
            if not os.path.exists(csv_abs):
                print(f"  ERROR: Failed to create {csv_abs}")
                return False
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            return False
    else:
        print(f"  Updating {csv_filename} (incremental: new quarters only)...")
        try:
            create_facility_complete_csv(provnum, existing_csv_path=csv_abs, output_path=csv_abs)
        except Exception as e:
            print(f"  WARNING: Incremental update failed: {e}")
        # If file still exists with data, we're done (create_facility_complete_csv skips full rebuild when file has data)

    # Non-nurse + citations before provider info (provider incremental can OOM on huge combined CSVs).
    nonnurse_filename = f"facility_{provnum}_nonnurse_daily.csv"
    citations_filename = f"facility_{provnum}_citations.csv"
    nonnurse_file = find_facility_nonnurse_daily(provnum) or str(facility_folder / nonnurse_filename)
    citations_file = find_facility_citations(provnum) or str(facility_folder / citations_filename)
    nonnurse_abs = (
        os.path.normpath(nonnurse_file)
        if os.path.isabs(nonnurse_file)
        else os.path.normpath(os.path.join(root, nonnurse_file))
    )
    citations_abs = (
        os.path.normpath(citations_file)
        if os.path.isabs(citations_file)
        else os.path.normpath(os.path.join(root, citations_file))
    )
    from nonnurse_staffing_lib import create_facility_nonnurse_csv
    from citation_lib import build_facility_citations_csv

    if not os.path.exists(nonnurse_abs):
        print(f"  Creating {nonnurse_filename} (full build)...")
        try:
            create_facility_nonnurse_csv(provnum, output_path=nonnurse_abs, root=root)
            if not os.path.exists(nonnurse_abs):
                root_nn = os.path.join(root, f"facility_{provnum}_nonnurse_daily.csv")
                if os.path.exists(root_nn):
                    shutil.move(root_nn, nonnurse_abs)
            if not os.path.exists(nonnurse_abs):
                print(f"  WARNING: {nonnurse_abs} not created (no non-nurse PBJ rows for this CCN?)")
        except Exception as e:
            print(f"  WARNING: Non-nurse CSV build failed: {e}")
    else:
        print(f"  Updating {nonnurse_filename} (incremental: new quarters only)...")
        try:
            create_facility_nonnurse_csv(
                provnum, existing_csv_path=nonnurse_abs, output_path=nonnurse_abs, root=root
            )
        except Exception as e:
            print(f"  WARNING: Non-nurse incremental update failed: {e}")

    print(f"  Updating {citations_filename} (CCN slice from latest Citations/NH_HealthCitations_*.csv)...")
    try:
        n_cit = build_facility_citations_csv(provnum, citations_abs, root=root)
        if n_cit is None:
            print("  WARNING: Citations slice not written (missing national file or invalid CCN).")
        else:
            print(f"    [OK] {n_cit} citation row(s)")
    except Exception as e:
        print(f"  WARNING: Citations slice failed: {e}")

    # Provider info: full build if missing, else incremental (only newer records)
    if not os.path.exists(provider_abs):
        print(f"  Creating {provider_filename}...")
        try:
            data = create_facility_provider_info_csv(provnum)
            if data is not None:
                data.to_csv(provider_abs, index=False)
            if not os.path.exists(provider_abs):
                print(f"  WARNING: {provider_abs} not created (may not have provider info data)")
        except Exception as e:
            print(f"  WARNING: {str(e)}")
    else:
        print(f"  Updating {provider_filename} (incremental: newer records only)...")
        try:
            create_facility_provider_info_csv(provnum, existing_csv_path=provider_abs, output_path=provider_abs)
        except Exception as e:
            print(f"  WARNING: Incremental update failed: {e}")
    
    # Step 2: Read the dynamic dashboard template
    print("\nStep 2: Reading dynamic dashboard source...")
    dynamic_dashboard_file = "dynamic_facility_dashboard.py"
    if not os.path.exists(dynamic_dashboard_file):
        print(f"  ERROR: {dynamic_dashboard_file} not found!")
        return False
    
    with open(dynamic_dashboard_file, 'r', encoding='utf-8') as f:
        dashboard_code = f.read()
    
    print(f"  [OK] Read {dynamic_dashboard_file} (includes geo rollup + template JSON for facility packages)")
    
    # Step 3: Create facility-specific Flask app under deployments/pbj320-<CCN>/
    print(f"\nStep 3: Creating facility-specific Flask app...")
    flask_app_filename = f"facility_{provnum}_flask_app.py"
    # Canonical location: deployments/pbj320-<CCN>/ (same folder as CSVs and EIN outputs)
    facility_folder_abs = os.path.normpath(os.path.join(root, str(facility_folder)))
    os.makedirs(facility_folder_abs, exist_ok=True)
    flask_app_file = os.path.join(facility_folder_abs, flask_app_filename)
    
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
        
        # Remove duplicate lazy-init from the dashboard template without deleting EIN helpers.
        # dynamic_facility_dashboard.py order: comment + PROVNUM/EIN env vars, then
        # _ein_active_ccn / _ein_mode_enabled / _filter_ein_by_selected_quarters / _load_ein_position_csvs,
        # then ensure_data_loaded + before_request, then @app.route. Older packagers stripped through
        # the first @app.route and dropped _load_ein_position_csvs (NameError in deployed apps).
        lazy_start = None
        for i, line in enumerate(new_lines):
            if 'Initialize data lazily (for Vercel deployment)' in line or (
                line.strip().startswith('# Initialize data lazily') and 'Vercel' in line
            ):
                lazy_start = i
                break
        ein_helpers_start = None
        old_ensure_start = None
        lazy_end = None
        if lazy_start is not None:
            for i in range(lazy_start + 1, len(new_lines)):
                if 'def _ein_active_ccn' in new_lines[i]:
                    ein_helpers_start = i
                    break
            if ein_helpers_start is not None:
                for i in range(ein_helpers_start, len(new_lines)):
                    stripped = new_lines[i].strip()
                    if stripped.startswith('def ensure_data_loaded'):
                        old_ensure_start = i
                        break
            if old_ensure_start is not None:
                for i in range(old_ensure_start + 1, len(new_lines)):
                    if new_lines[i].strip().startswith('@app.route'):
                        lazy_end = i
                        break
        if (
            lazy_start is not None
            and ein_helpers_start is not None
            and old_ensure_start is not None
            and lazy_end is not None
        ):
            new_lines = (
                new_lines[:lazy_start]
                + new_lines[ein_helpers_start:old_ensure_start]
                + new_lines[lazy_end:]
            )
        elif lazy_start is not None:
            print(
                "  [WARN] Precise lazy-init strip skipped (expected _ein_active_ccn / "
                "ensure_data_loaded markers not found). Check dynamic_facility_dashboard.py."
            )

        # Find the first @app.route to insert before_request hook before it
        first_route_idx = -1
        for i, line in enumerate(new_lines):
            if line.strip().startswith('@app.route'):
                first_route_idx = i
                break
        
        # Add lazy initialization code (prevents Vercel deployment hangs)
        # DEPLOYED_DATE stays at module top (datetime at import); do not duplicate here.
        init_code = [
            '',
            '# Initialize data lazily (for Vercel deployment)',
            f'# Hardcoded for facility {provnum}',
            f'PROVNUM = "{provnum}"',
            f'EIN_DASHBOARD_MODE = "{ein_mode}"',
            f'EIN_SELECTED_QUARTERS = {[q for q in quarters]}',
            '_data_initialized = False',
            '',
            'def ensure_data_loaded():',
            '    """Lazy initialization - only load data on first request"""',
            '    global _data_initialized',
            '    if not _data_initialized:',
            '        try:',
            '            print(f"Initializing facility {PROVNUM} dashboard (lazy load)...")',
            '            create_dynamic_dashboard(PROVNUM)',
            '            print(f"[OK] Successfully initialized facility {PROVNUM} dashboard")',
            '            _data_initialized = True',
            '        except Exception as e:',
            '            print(f"[WARNING] Error initializing facility {PROVNUM} dashboard: {e}")',
            '            import traceback',
            '            traceback.print_exc()',
            '',
            '@app.before_request',
            'def before_request():',
            '    auth_resp = _dashboard_basic_auth_challenge()',
            '    if auth_resp is not None:',
            '        return auth_resp',
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
            '    app.run(debug=True, port=5000, threaded=True)'
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
EIN_DASHBOARD_MODE = "{ein_mode}"
EIN_SELECTED_QUARTERS = {[q for q in quarters]}
_data_initialized = False

def ensure_data_loaded():
    """Lazy initialization - only load data on first request"""
    global _data_initialized
    if not _data_initialized:
        try:
            print(f"Initializing facility {{PROVNUM}} dashboard (lazy load)...")
            create_dynamic_dashboard(PROVNUM)
            print(f"[OK] Successfully initialized facility {{PROVNUM}} dashboard")
            _data_initialized = True
        except Exception as e:
            print(f"[WARNING] Error initializing facility {{PROVNUM}} dashboard: {{e}}")
            import traceback
            traceback.print_exc()

@app.before_request
def before_request():
    auth_resp = _dashboard_basic_auth_challenge()
    if auth_resp is not None:
        return auth_resp
    ensure_data_loaded()

'''
            new_lines = lines[:first_route_idx] + init_code.split('\n') + lines[first_route_idx:]
            new_lines.extend([
                '',
                'if __name__ == "__main__":',
                '    # For local testing',
                '    ensure_data_loaded()  # Load immediately for local dev',
                '    app.run(debug=True, port=5000, threaded=True)'
            ])
            modified_code = '\n'.join(new_lines)
        else:
            # Fallback: just append
            modified_code = dashboard_code + f'''

# Initialize data lazily (for Vercel deployment)
# Hardcoded for facility {provnum}
PROVNUM = "{provnum}"
EIN_DASHBOARD_MODE = "{ein_mode}"
EIN_SELECTED_QUARTERS = {[q for q in quarters]}
_data_initialized = False

def ensure_data_loaded():
    """Lazy initialization - only load data on first request"""
    global _data_initialized
    if not _data_initialized:
        try:
            print(f"Initializing facility {{PROVNUM}} dashboard (lazy load)...")
            create_dynamic_dashboard(PROVNUM)
            print(f"[OK] Successfully initialized facility {{PROVNUM}} dashboard")
            _data_initialized = True
        except Exception as e:
            print(f"[WARNING] Error initializing facility {{PROVNUM}} dashboard: {{e}}")
            import traceback
            traceback.print_exc()

@app.before_request
def before_request():
    auth_resp = _dashboard_basic_auth_challenge()
    if auth_resp is not None:
        return auth_resp
    ensure_data_loaded()

if __name__ == "__main__":
    # For local testing
    ensure_data_loaded()  # Load immediately for local dev
    app.run(debug=True, port=5000, threaded=True)
'''
    
    # Write the facility-specific Flask app
    with open(flask_app_file, 'w', encoding='utf-8') as f:
        f.write(modified_code)
    
    print(f"  [OK] Created {os.path.basename(flask_app_file)}")
    
    # Step 4: Prepare vercel.json configuration
    print(f"\nStep 4: Preparing Vercel configuration...")
    import json
    # Explicitly include CSVs and templates so Vercel bundles them (Python may omit non-.py files otherwise)
    include_files = "templates/**,macpac_state_standards_clean.csv,static/**"
    if not use_central_csv_host:
        # Parquet: facility_*_ein_*.parquet and similar must ship with the function bundle
        include_files = "*.csv,*.parquet,templates/**,static/**,config/**"
    vercel_config = {
        "version": 2,
        "builds": [
            {
                "src": flask_app_filename,
                "use": "@vercel/python",
                "config": {
                    "includeFiles": include_files
                }
            }
        ],
        "routes": [
            {
                "src": "/(.*)",
                "dest": flask_app_filename
            }
        ],
        "env": {
            "PYTHONPATH": ".",
            "EIN_DASHBOARD_MODE": ein_mode,
            "EIN_SELECTED_QUARTERS": quarters_csv
        }
    }
    
    # Step 5: Create deployment directory structure (absolute path so it works when cwd differs, e.g. Flask)
    print(f"\nStep 5: Creating deployment directory structure...")
    deploy_dir = os.path.normpath(os.path.join(root, str(facility_folder)))
    
    # Deployment directory already exists (we created it earlier)
    print(f"  [OK] Using directory: {deploy_dir}")
    
    # Create templates subdirectory in deployment directory
    deploy_templates_dir = os.path.join(deploy_dir, "templates")
    os.makedirs(deploy_templates_dir, exist_ok=True)
    
    # List all files needed for deployment (paths from project root so they work when cwd differs)
    template_source = os.path.normpath(os.path.join(root, "templates", "dynamic_facility_dashboard.html"))
    if not os.path.exists(template_source):
        raise FileNotFoundError(f"Template file not found: {template_source}")
    
    # Get just the filenames for deployment (files should already be in deploy_dir)
    flask_app_filename = os.path.basename(flask_app_file)
    csv_filename = os.path.basename(csv_file)
    provider_csv_filename = os.path.basename(provider_csv_file) if provider_csv_file and os.path.exists(provider_csv_file) else None
    
    deployment_files = {
        flask_app_file: flask_app_filename,  # source -> destination (just filename)
        template_source: "templates/dynamic_facility_dashboard.html"
    }
    favicon_src = os.path.normpath(os.path.join(root, "pbj_favicon.png"))
    if os.path.isfile(favicon_src):
        deployment_files[favicon_src] = "pbj_favicon.png"

    # Shared snapshot for /data-matching + interval API fallback (same JSON for every facility; small file)
    interval_mapping_src = os.path.normpath(os.path.join(root, "static", "data", "interval_quarter_mapping.json"))
    if os.path.isfile(interval_mapping_src):
        deployment_files[interval_mapping_src] = "static/data/interval_quarter_mapping.json"

    # If using centralized CSV hosting, don't bundle facility CSVs into each deployment
    if not use_central_csv_host:
        deployment_files[csv_abs] = csv_filename
        if os.path.exists(provider_abs):
            deployment_files[provider_abs] = os.path.basename(provider_csv_file)
        if os.path.exists(nonnurse_abs):
            deployment_files[nonnurse_abs] = os.path.basename(nonnurse_abs)
        if os.path.exists(citations_abs):
            deployment_files[citations_abs] = os.path.basename(citations_abs)
    
    # Add prov_info.py if it exists (needed for quarter mapping)
    prov_info_file = "prov_info.py"
    if os.path.exists(prov_info_file):
        deployment_files[prov_info_file] = prov_info_file
    quarter_map_src = os.path.normpath(os.path.join(root, "prov_info_quarter_map.py"))
    if os.path.isfile(quarter_map_src):
        deployment_files[quarter_map_src] = "prov_info_quarter_map.py"
    
    # Add file_path_utils.py so deployed app can resolve paths (import in create_dynamic_dashboard)
    fp_utils = os.path.normpath(os.path.join(root, "file_path_utils.py"))
    if os.path.exists(fp_utils):
        deployment_files[fp_utils] = "file_path_utils.py"
    
    # Add facility_report_lib.py so deployed app can use calculate_harrington_adjusted_hprd (single source for Harrington formula)
    facility_report_lib_src = os.path.normpath(os.path.join(root, "facility_report_lib.py"))
    if os.path.exists(facility_report_lib_src):
        deployment_files[facility_report_lib_src] = "facility_report_lib.py"

    # EIN / employee-detail bridge (dynamic_facility_dashboard imports these at module load)
    facility_ein_lib_src = os.path.normpath(os.path.join(root, "facility_ein_lib.py"))
    if os.path.isfile(facility_ein_lib_src):
        deployment_files[facility_ein_lib_src] = "facility_ein_lib.py"
    facility_ein_analytics_src = os.path.normpath(os.path.join(root, "facility_ein_employee_analytics.py"))
    if os.path.isfile(facility_ein_analytics_src):
        deployment_files[facility_ein_analytics_src] = "facility_ein_employee_analytics.py"

    pbj_facility_display_name_src = os.path.normpath(os.path.join(root, "pbj_facility_display_name.py"))
    if os.path.isfile(pbj_facility_display_name_src):
        deployment_files[pbj_facility_display_name_src] = "pbj_facility_display_name.py"
    
    # Add MACPAC standards file (try clean version first, fall back to original)
    macpac_clean = "pbj_lite/macpac_state_standards_clean.csv"
    macpac_original = "macpac/macpac_state_standards.csv"
    if os.path.exists(macpac_clean):
        deployment_files[macpac_clean] = "macpac_state_standards_clean.csv"
    elif os.path.exists(macpac_original):
        deployment_files[macpac_original] = "macpac_state_standards.csv"

    # Quarterly + CMS region HPRD rollups for Summary benchmarks and geographic rollup UI.
    # On Vercel, _resolve_pbj_lite_csv cannot reach repo ../../ — these must live in the bundle.
    added_pbj_lite_subdir_csvs = False
    for geo_name in (
        "state_quarterly_metrics.csv",
        "national_quarterly_metrics.csv",
        "cms_region_quarterly_metrics.csv",
        "cms_region_state_mapping.csv",
    ):
        src_geo = os.path.normpath(os.path.join(root, geo_name))
        if os.path.isfile(src_geo):
            deployment_files[src_geo] = geo_name
    for lite_name in ("state_lite_metrics.csv", "national_lite_metrics.csv"):
        src_lite = os.path.normpath(os.path.join(root, "pbj_lite", lite_name))
        if os.path.isfile(src_lite):
            deployment_files[src_lite] = f"pbj_lite/{lite_name}"
            added_pbj_lite_subdir_csvs = True

    entity_longitudinal_needs_pyarrow = False
    added_ownership_bundle = False
    if include_entity_longitudinal:
        elib = os.path.normpath(os.path.join(root, "entity_longitudinal_metrics.py"))
        if os.path.isfile(elib):
            deployment_files[elib] = "entity_longitudinal_metrics.py"
        own_lookup = os.path.normpath(os.path.join(root, "ownership", "entity_lookup.csv"))
        long_csv = os.path.normpath(os.path.join(root, "ownership", "chain_performance_longitudinal.csv"))
        long_pq = os.path.normpath(os.path.join(root, "ownership", "chain_performance_longitudinal.parquet"))
        if os.path.isfile(own_lookup):
            deployment_files[own_lookup] = "ownership/entity_lookup.csv"
            added_ownership_bundle = True

        # Per-facility slice: all affiliated chain IDs from provider history → small CSV on the server
        deploy_ownership_dir = os.path.join(deploy_dir, "ownership")
        os.makedirs(deploy_ownership_dir, exist_ok=True)
        slice_filename = f"facility_{provnum}_entity_longitudinal.csv"
        slice_deploy_path = os.path.join(deploy_ownership_dir, slice_filename)
        slice_written = False
        if os.path.isfile(provider_abs):
            try:
                from entity_longitudinal_metrics import write_facility_longitudinal_slice_for_deploy

                slice_written = write_facility_longitudinal_slice_for_deploy(
                    Path(root),
                    provnum,
                    Path(provider_abs),
                    Path(slice_deploy_path),
                )
            except Exception as e:
                print(f"  [WARN] Entity longitudinal slice not built: {e}")
        if slice_written:
            added_ownership_bundle = True
            try:
                sz_kb = os.path.getsize(slice_deploy_path) / 1024
                print(f"    [OK] {slice_filename} ({sz_kb:.1f} KB) — affiliation-history slice; omitting full chain_performance_longitudinal from bundle")
            except OSError:
                print(f"    [OK] {slice_filename} — affiliation-history slice; omitting full chain_performance_longitudinal from bundle")

        if not slice_written:
            if os.path.isfile(long_csv):
                deployment_files[long_csv] = "ownership/chain_performance_longitudinal.csv"
                added_ownership_bundle = True
            elif os.path.isfile(long_pq):
                deployment_files[long_pq] = "ownership/chain_performance_longitudinal.parquet"
                added_ownership_bundle = True
                entity_longitudinal_needs_pyarrow = True
    else:
        print("  [OK] Skipping entity/chain longitudinal bundle (include_entity_longitudinal=False)")

    for lib_name in ("nonnurse_staffing_lib.py", "citation_lib.py", "pbj_staffing_normalize.py"):
        lib_src = os.path.normpath(os.path.join(root, lib_name))
        if os.path.isfile(lib_src):
            deployment_files[lib_src] = lib_name

    for cfg_name in ("nonnurse_staff_groups.json", "citation_severity_rank.json"):
        cfg_src = os.path.normpath(os.path.join(root, "config", cfg_name))
        if os.path.isfile(cfg_src):
            deployment_files[cfg_src] = f"config/{cfg_name}"

    pbj_identifiers_root = os.path.normpath(os.path.join(root, "pbj_identifiers"))
    added_pbj_identifiers = False
    if os.path.isdir(pbj_identifiers_root):
        for entry in sorted(os.listdir(pbj_identifiers_root)):
            if entry == "__pycache__" or not entry.endswith(".py"):
                continue
            src_p = os.path.join(pbj_identifiers_root, entry)
            if os.path.isfile(src_p):
                deployment_files[src_p] = f"pbj_identifiers/{entry}"
                added_pbj_identifiers = True

    if added_ownership_bundle and ",ownership/**" not in include_files:
        include_files = include_files + ",ownership/**"
    if added_pbj_identifiers and ",pbj_identifiers/**" not in include_files:
        include_files = include_files + ",pbj_identifiers/**"
    if added_pbj_lite_subdir_csvs and ",pbj_lite/**" not in include_files:
        include_files = include_files + ",pbj_lite/**"
    vercel_config["builds"][0]["config"]["includeFiles"] = include_files

    ein_prefix = f"facility_{provnum}_ein_"
    ein_bundle_names = sorted(
        p.name
        for p in Path(deploy_dir).iterdir()
        if p.is_file()
        and p.suffix.lower() in (".csv", ".parquet")
        and p.name.startswith(ein_prefix)
    )
    print(
        "EIN bundle (already in deploy_dir): "
        + (", ".join(ein_bundle_names) if ein_bundle_names else "(none)")
    )
    # EIN detail/summary tables are often parquet-only in deploy bundles.
    # Ensure the runtime can read parquet even when entity longitudinal CSV path
    # does not trigger the older pyarrow flag.
    ein_bundle_has_parquet = any(name.lower().endswith(".parquet") for name in ein_bundle_names)

    # Copy files to deployment directory
    print("\n  Copying files to deployment directory:")
    for src_file, dest_file in deployment_files.items():
        # Convert to absolute path for reliable checking
        if not os.path.isabs(src_file):
            src_file_abs = os.path.abspath(src_file)
        else:
            src_file_abs = src_file
        
        # Check if source file exists (with absolute path check)
        if not os.path.exists(src_file_abs):
            print(f"    ✗ {dest_file} (source file missing: {src_file_abs})")
            # For template file, provide helpful error
            if 'dynamic_facility_dashboard.html' in src_file_abs:
                raise FileNotFoundError(f"Template file not found: {src_file_abs}. Expected location: {os.path.join(os.getcwd(), 'templates', 'dynamic_facility_dashboard.html')}")
            continue
        
        # Verify file still exists right before copying
        if not os.path.isfile(src_file_abs):
            print(f"    ✗ {dest_file} (source is not a file: {src_file_abs})")
            continue
        
        dest_path = os.path.join(deploy_dir, dest_file)
        dest_dir = os.path.dirname(dest_path)
        if dest_dir and not os.path.exists(dest_dir):
            os.makedirs(dest_dir, exist_ok=True)
        
        # Skip copy when source is already in deploy_dir (same path); copy2(same, same) fails on Windows
        try:
            if os.path.normpath(os.path.abspath(src_file_abs)) == os.path.normpath(os.path.abspath(dest_path)):
                try:
                    size = os.path.getsize(src_file_abs) / (1024 * 1024)
                    print(f"    [OK] {dest_file} (already in place, {size:.2f} MB)")
                except OSError:
                    print(f"    [OK] {dest_file} (already in place)")
                continue
        except (ValueError, OSError):
            pass  # Paths not on same drive etc.; fall through to copy
        
        # Retry logic for file copying (handles Flask auto-reloader file locks)
        max_retries = 8  # Increased retries
        retry_delay = 0.5  # Start with 0.5 seconds
        copied = False
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Verify source file still exists before each attempt
                if not os.path.exists(src_file_abs):
                    raise FileNotFoundError(f"Source file disappeared: {src_file_abs}")
                
                # If destination exists and is locked, try to remove it first
                if os.path.exists(dest_path) and attempt > 0:
                    try:
                        os.remove(dest_path)
                        time.sleep(0.1)  # Brief pause after removal
                    except (PermissionError, OSError):
                        pass  # Ignore if can't remove
                
                shutil.copy2(src_file_abs, dest_path)
                copied = True
                break
            except FileNotFoundError as e:
                last_error = e
                # Don't retry file not found errors
                raise Exception(f"Source file not found: {src_file_abs}. Error: {str(e)}")
            except (PermissionError, OSError) as e:
                last_error = e
                error_str = str(e)
                if '32' in error_str or 'being used' in error_str.lower() or 'WinError 32' in error_str:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (1.5 ** attempt)  # Slower exponential backoff
                        print(f"    [WARNING] File locked, retrying in {wait_time:.1f}s... ({attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        raise Exception(f"File {src_file_abs} is locked after {max_retries} attempts. Please close any programs using it (including Flask auto-reloader) and try again.")
                elif '2' in error_str and ('cannot find' in error_str.lower() or 'not found' in error_str.lower()):
                    raise Exception(f"File not found: {src_file_abs}. The file may have been moved or deleted.")
                else:
                    raise
        
        if copied:
            try:
                size = os.path.getsize(src_file_abs) / (1024 * 1024)  # Size in MB
                print(f"    [OK] {dest_file} ({size:.2f} MB)")
            except OSError:
                print(f"    [OK] {dest_file}")
        else:
            raise Exception(f"Failed to copy {src_file_abs} after {max_retries} attempts. Last error: {last_error}")
    
    # Create vercel.json in deployment directory
    deploy_vercel_json = os.path.join(deploy_dir, "vercel.json")
    with open(deploy_vercel_json, 'w', encoding='utf-8') as f:
        json.dump(vercel_config, f, indent=2)
    print(f"    [OK] vercel.json created in {deploy_dir}")
    
    # Create requirements.txt for Flask deployment (not Streamlit)
    deploy_requirements = os.path.join(deploy_dir, "requirements.txt")
    flask_requirements = """Flask==3.0.0
pandas==2.2.3
numpy==1.26.4
scipy==1.14.1
python-dateutil==2.9.0
pytz==2023.3
"""
    if entity_longitudinal_needs_pyarrow or ein_bundle_has_parquet:
        flask_requirements += "pyarrow>=14.0.0\n"
    with open(deploy_requirements, 'w', encoding='utf-8') as f:
        f.write(flask_requirements)
    print(f"    [OK] requirements.txt created in {deploy_dir}")
    
    # Step 6: Create deployment instructions
    print(f"\nStep 6: Creating deployment guide...")
    guide_file = str(facility_folder / f"DEPLOY_{provnum}_TO_VERCEL.md")
    
    guide_content = f"""# Deploy Facility {provnum} Dashboard to Vercel

## Files Created

This script has created the following files for deployment:

- `{flask_app_filename}` - Facility-specific Flask app
- `{csv_filename}` - Facility daily data{" (NOT included when PBJ_CSV_HOST_BASE_URL is set)" if use_central_csv_host else ""}
- `{provider_csv_filename if provider_csv_filename else '(optional)'}` - Provider info data{" (NOT included when PBJ_CSV_HOST_BASE_URL is set)" if use_central_csv_host else ""}
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
   git add {flask_app_filename} {csv_filename} vercel.json requirements.txt templates/
   """
    
    if provider_csv_filename:
        guide_content += f"   git add {provider_csv_filename}\n"
    
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
├── {flask_app_filename}          # Main Flask app
├── {csv_filename}                # Facility data{" (not included when using centralized CSV hosting)" if use_central_csv_host else ""}
"""
    
    if provider_csv_filename:
        guide_content += f"├── {provider_csv_filename}         # Provider info{' (not included when using centralized CSV hosting)' if use_central_csv_host else ''}\n"
    
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
python {flask_app_filename}

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
    
    print(f"  [OK] Created {guide_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("[SUCCESS] Deployment Package Created Successfully!")
    print(f"{'='*60}\n")
    print(f"Deployment directory: {deploy_dir}/")
    print(f"All files are ready in: {deploy_dir}/")
    print(f"\nNext Steps:")
    print(f"1. Test locally: cd {deploy_dir} && python {flask_app_filename}")
    print(f"2. Deploy to Vercel:")
    print(f"   - Run: deploy_to_vercel.bat (enter {provnum} when prompted)")
    print(f"   - Or manually: cd {deploy_dir} && vercel link --project=pbj320-{provnum} && vercel --prod")
    print(f"\nProject will be deployed as: pbj320-{provnum}")
    print(f"URL will be: https://pbj320-{provnum}.vercel.app\n")
    
    return True

if __name__ == "__main__":
    try:
        if len(sys.argv) < 2:
            print("Usage: python create_vercel_deployment.py <PROVNUM> [ein_mode] [ein_selected_quarters_csv]")
            print("Example: python create_vercel_deployment.py 495241")
            sys.exit(1)

        provnum = str(sys.argv[1]).strip().zfill(6)
        if not provnum.isdigit() or len(provnum) != 6:
            print("Error: PROVNUM must be a 6-digit CMS certification number (CCN), e.g. 395052.")
            sys.exit(1)

        ein_mode = sys.argv[2] if len(sys.argv) > 2 else "all"
        q_csv = sys.argv[3] if len(sys.argv) > 3 else ""
        ein_selected_quarters = [q.strip() for q in q_csv.split(",") if q.strip()]

        success = create_facility_vercel_package(
            provnum,
            ein_mode=ein_mode,
            ein_selected_quarters=ein_selected_quarters,
        )

        if not success:
            print("\nFailed to create deployment package. Review the messages above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

