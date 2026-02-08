@echo off
REM OptionsTitan GUI Launcher for Windows

echo.
echo ======================================
echo  OptionsTitan Strategy Analyzer
echo ======================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.7+ from python.org
    pause
    exit /b 1
)

echo [INFO] Checking dependencies...
python -c "import yfinance, pandas, numpy, tkinter" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Some dependencies are missing
    echo [INFO] Installing required packages...
    pip install -r requirements.txt
)

echo.
echo [INFO] Launching GUI...
echo.
python options_gui.py

echo.
echo [INFO] GUI closed. Thanks for using OptionsTitan!
pause
