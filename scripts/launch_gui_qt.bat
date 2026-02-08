@echo off
REM OptionsTitan Qt GUI Launcher for Windows (PySide6 version)

echo.
echo ======================================
echo  OptionsTitan Qt Strategy Analyzer
echo  PySide6 Version
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
python -c "import PySide6, yfinance, pandas, numpy" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Some dependencies are missing
    echo.
    set /p install="Install dependencies now? (y/n): "
    
    if /i "%install%"=="y" (
        echo [INFO] Installing dependencies...
        pip install -r requirements.txt
        
        if errorlevel 1 (
            echo [ERROR] Failed to install dependencies
            pause
            exit /b 1
        )
        
        echo [INFO] Dependencies installed successfully
    ) else (
        echo [WARN] GUI may not function properly without dependencies
        timeout /t 2 >nul
    )
) else (
    echo [INFO] All dependencies installed
)

echo.
echo [INFO] Launching OptionsTitan Qt GUI...
echo.

REM Launch the Qt GUI
python options_gui_qt.py

if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with an error
    pause
    exit /b 1
)

echo.
echo [INFO] GUI closed. Thanks for using OptionsTitan!
pause
