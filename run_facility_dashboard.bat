@echo off
cd /d "%~dp0"
echo Enter 6-digit facility code:
set /p provnum=
echo.

if exist "facility_%provnum%_complete_data.csv" (
    echo CSV file already exists for facility %provnum%
    echo.
) else (
    echo Creating CSV for facility %provnum%...
    python -c "from dynamic_facility_dashboard import create_facility_complete_csv; create_facility_complete_csv('%provnum%')"
    echo.
)

if exist "facility_%provnum%_provider_info_data.csv" (
    echo Provider info CSV already exists for facility %provnum%
    echo.
) else (
    echo Creating provider info CSV for facility %provnum%...
    python -c "from dynamic_facility_dashboard import create_facility_provider_info_csv; import pandas as pd; data = create_facility_provider_info_csv('%provnum%'); data.to_csv('facility_%provnum%_provider_info_data.csv', index=False) if data is not None else None"
    echo.
)

echo Starting dashboard for facility %provnum%...
echo Dashboard will be available at: http://localhost:5000
echo.
python dynamic_facility_dashboard.py %provnum%
pause
