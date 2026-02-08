#!/usr/bin/env python3
"""
OptionsTitan - AI-Powered Options Trading System
Main entry point for the trading system

Usage:
    python main.py

This script runs the complete OptionsTitan pipeline including:
- Data preprocessing and feature engineering
- 5-model ensemble training
- Risk management analysis
- Model explainability
- Production artifact generation
"""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Import and run the main training pipeline
if __name__ == "__main__":
    print("🚀 Starting OptionsTitan AI Options Trading System...")
    print("=" * 60)
    
    try:
        # Import the main training module as part of src package
        import src.Training
        
        print("\n✅ OptionsTitan execution completed successfully!")
        print("Check the models/ folder for trained models and artifacts.")
        print("Check the logs/ folder for detailed execution logs.")
        
    except Exception as e:
        print(f"❌ Error running OptionsTitan: {e}")
        print("Please check the logs/ folder for detailed error information.")
        sys.exit(1)