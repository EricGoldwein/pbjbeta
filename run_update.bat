@echo off
REM Pipeline Update Batch File
REM Activates virtual environment and runs the pipeline orchestrator

cd /d "%~dp0"

REM Try to activate virtual environment
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
) else if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
)

REM Run the orchestrator with all passed arguments
python run_pipeline_update.py %*

REM Preserve exit code
exit /b %ERRORLEVEL%
