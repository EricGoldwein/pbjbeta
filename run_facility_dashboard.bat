@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

REM Internal facility Flask dashboard — Employee Detail defaults to all quarters.
REM App file is under deployments\pbj320-<CCN>\ ; do not run facility_*_flask_app.py from repo root.
REM Examples:
REM   run_facility_dashboard.bat 335581
REM   run_facility_dashboard.bat 335581 --deployment-bundle
REM   run_facility_dashboard.bat 335581 --ein-mode none

if "%~1"=="" (
    echo Enter 6-digit facility code:
    set /p provnum=
    echo.
    echo Starting dashboard (EIN: all quarters by default^) ...
    python run_facility_dashboard.py !provnum!
) else (
    echo Using args: %*
    echo.
    echo Starting dashboard (EIN: all quarters by default^) ...
    python run_facility_dashboard.py %*
)

echo.
echo Open http://localhost:5000 when the server is ready.
pause
