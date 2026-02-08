#!/bin/bash
# Fetch all 45 tickers with progress monitoring
# Usage: ./scripts/fetch_all_tickers.sh

set -e  # Exit on error

echo "======================================================================"
echo "OptionsTitan - Multi-Ticker Data Collection"
echo "======================================================================"
echo ""

# Check if ThetaData Terminal is accessible
echo "[1/4] Checking ThetaData connection..."
python3 -c "from src.data_collection.thetadata_client import ThetaDataClient; ThetaDataClient()" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ Connected to ThetaData Terminal"
else
    echo "❌ Cannot connect to ThetaData Terminal"
    echo ""
    echo "Please ensure:"
    echo "1. ThetaData Terminal is installed"
    echo "2. Terminal is running"
    echo "3. You're logged in"
    echo ""
    exit 1
fi

# Create log directory
mkdir -p logs

# Set date range
START_DATE="2019-01-01"
END_DATE="2024-12-31"

echo ""
echo "[2/4] Configuration:"
echo "  Start date: $START_DATE"
echo "  End date:   $END_DATE"
echo "  Log file:   logs/fetch_all_$(date +%Y%m%d_%H%M%S).log"
echo ""

# Ask for confirmation
read -p "Start fetching all 45 tickers? This may take 6-15 hours. (y/n): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Start fetch in background
LOG_FILE="logs/fetch_all_$(date +%Y%m%d_%H%M%S).log"

echo "[3/4] Starting data fetch..."
echo "  Log file: $LOG_FILE"
echo "  You can close this terminal. Fetch will continue in background."
echo ""

# Run fetch with nohup
nohup python3 -m src.data_collection.data_fetcher \
    --start "$START_DATE" \
    --end "$END_DATE" \
    --verbose > "$LOG_FILE" 2>&1 &

FETCH_PID=$!

echo "  Process ID: $FETCH_PID"
echo ""
echo "[4/4] Monitoring (press Ctrl+C to stop monitoring, fetch continues):"
echo ""

# Monitor for first minute, then show how to check progress
sleep 5

echo "Initial progress:"
tail -20 "$LOG_FILE"

echo ""
echo "======================================================================"
echo "Data fetch is running in background (PID: $FETCH_PID)"
echo "======================================================================"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check completion:"
echo "  grep 'Completed' $LOG_FILE | wc -l"
echo ""
echo "Check for errors:"
echo "  grep 'ERROR' $LOG_FILE"
echo ""
echo "Stop fetch:"
echo "  kill $FETCH_PID"
echo ""
echo "Resume (if interrupted):"
echo "  python3 -m src.data_collection.data_fetcher --start $START_DATE --end $END_DATE"
echo ""
echo "======================================================================"
