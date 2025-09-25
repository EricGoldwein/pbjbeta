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

echo Starting dashboard for facility %provnum%...
echo Dashboard will be available at: http://localhost:5000
echo.
python dynamic_facility_dashboard.py %provnum%
pause
