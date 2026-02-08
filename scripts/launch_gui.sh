#!/bin/bash
# OptionsTitan GUI Launcher

echo "🚀 Launching OptionsTitan Strategy Analyzer GUI..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python3 -c "import yfinance, pandas, numpy, tkinter" 2>/dev/null

if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing."
    echo "Installing required packages..."
    pip3 install -r requirements.txt
fi

# Launch the GUI
echo "✅ Starting GUI..."
python3 options_gui.py

echo ""
echo "👋 GUI closed. Thanks for using OptionsTitan!"
