#!/bin/bash
# OptionsTitan Qt GUI Launcher (PySide6 version)

echo "╔══════════════════════════════════════════════════════════╗"
echo "║      OptionsTitan Qt Strategy Analyzer (PySide6)        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check Python version
python_version=$(python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 1-2)
required_version="3.7"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then 
    echo "❌ Python 3.7 or higher is required. You have Python $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"
echo ""

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python3 -c "import PySide6, yfinance, pandas, numpy" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing."
    echo ""
    read -p "Install dependencies now? (y/n): " response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Installing..."
        pip3 install -r requirements.txt
        
        if [ $? -ne 0 ]; then
            echo "❌ Failed to install dependencies"
            exit 1
        fi
        
        echo "✅ Dependencies installed successfully"
    else
        echo "⚠️  GUI will launch, but may not function properly without dependencies."
        sleep 2
    fi
else
    echo "✅ All dependencies installed"
fi

echo ""
echo "🚀 Launching OptionsTitan Qt GUI..."
echo ""

# Launch the Qt GUI
python3 options_gui_qt.py

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "👋 GUI closed. Thanks for using OptionsTitan!"
else
    echo ""
    echo "❌ GUI exited with an error. Check the output above."
    exit 1
fi
