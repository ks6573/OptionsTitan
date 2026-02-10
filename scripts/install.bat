@echo off
REM OptionsTitan Installation Script with UV (Windows)
REM Modern, fast installation using UV package manager

echo ======================================================================
echo          OptionsTitan - Installation with UV
echo ======================================================================
echo.

REM Check if uv is installed
where uv >nul 2>&1
if errorlevel 1 (
    echo UV not found. Installing UV...
    echo.
    
    REM Install UV using PowerShell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    
    if errorlevel 1 (
        echo Failed to install UV
        echo Falling back to pip installation...
        pip install -r requirements.txt
        exit /b %errorlevel%
    )
    
    echo UV installed successfully
) else (
    echo UV is already installed
)

echo.
echo Installing OptionsTitan dependencies...
echo.

REM Sync dependencies using UV
uv sync

if errorlevel 1 (
    echo.
    echo Installation failed with UV
    echo.
    echo Try manual installation:
    echo   pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo              Installation Complete!
echo ======================================================================
echo.
echo You can now run:
echo   - Modern GUI:  uv run python options_gui_qt.py
echo   - Classic GUI: uv run python options_gui.py
echo   - Train AI:    uv run python main.py
echo.
echo Next steps:
echo   1. Read GETTING_STARTED.md for setup guide
echo   2. Optional: Enable LLAMA AI (scripts\setup_llama.sh)
echo   3. Run: uv run python verify_installation.py
echo.
pause
