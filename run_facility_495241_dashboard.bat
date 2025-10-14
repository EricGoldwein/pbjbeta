@echo off
cd /d "%~dp0"
echo Starting Dashboard for Facility 495241...
echo.

if exist "facility_495241_complete_data.csv" (
    echo CSV file already exists for facility 495241
    echo.
) else (
    echo Creating CSV for facility 495241...
    python -c "from dynamic_facility_dashboard import create_facility_complete_csv; create_facility_complete_csv('495241')"
    echo.
)

if exist "facility_495241_provider_info_data.csv" (
    echo Provider info CSV already exists for facility 495241
    echo.
) else (
    echo Creating provider info CSV for facility 495241...
    python -c "from dynamic_facility_dashboard import create_facility_provider_info_csv; create_facility_provider_info_csv('495241')"
    echo.
)

echo Starting dashboard for facility 495241...
echo Dashboard will be available at: http://localhost:5001
echo.
python dynamic_facility_dashboard.py 495241
pause
