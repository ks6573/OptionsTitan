@echo off
REM Fetch all 45 tickers - Windows batch script
REM Usage: scripts\fetch_all_tickers.bat

echo ======================================================================
echo OptionsTitan - Multi-Ticker Data Collection
echo ======================================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.7+
    pause
    exit /b 1
)

REM Check ThetaData connection
echo [1/4] Checking ThetaData connection...
python -c "from src.data_collection.thetadata_client import ThetaDataClient; ThetaDataClient()" 2>&1

if errorlevel 1 (
    echo.
    echo Cannot connect to ThetaData Terminal
    echo.
    echo Please ensure:
    echo 1. ThetaData Terminal is installed
    echo 2. Terminal is running
    echo 3. You're logged in
    echo.
    pause
    exit /b 1
)

echo OK - Connected to ThetaData Terminal
echo.

REM Create log directory
if not exist logs mkdir logs

REM Set date range
set START_DATE=2019-01-01
set END_DATE=2024-12-31
set LOG_FILE=logs\fetch_all_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.log

echo [2/4] Configuration:
echo   Start date: %START_DATE%
echo   End date:   %END_DATE%
echo   Log file:   %LOG_FILE%
echo.

REM Ask for confirmation
set /p CONFIRM="Start fetching all 45 tickers? This may take 6-15 hours. (y/n): "
if /i not "%CONFIRM%"=="y" (
    echo Aborted.
    pause
    exit /b 0
)

echo.
echo [3/4] Starting data fetch...
echo   Log file: %LOG_FILE%
echo.

REM Start fetch
start /B python -m src.data_collection.data_fetcher --start %START_DATE% --end %END_DATE% --verbose > %LOG_FILE% 2>&1

echo [4/4] Fetch started in background
echo.
echo ======================================================================
echo Data fetch is running
echo ======================================================================
echo.
echo Monitor progress:
echo   type %LOG_FILE%
echo.
echo Check completion:
echo   find /c "Completed" %LOG_FILE%
echo.
echo Check for errors:
echo   find "ERROR" %LOG_FILE%
echo.
echo Resume (if interrupted):
echo   python -m src.data_collection.data_fetcher --start %START_DATE% --end %END_DATE%
echo.
echo ======================================================================
echo.
pause
