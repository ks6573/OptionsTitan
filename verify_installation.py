#!/usr/bin/env python3
"""
OptionsTitan Installation Verification Script

Checks if all required dependencies are properly installed.
Run this after: pip install -r requirements.txt

Usage:
    python verify_installation.py
"""

import sys

def check_dependencies():
    """Check all required and optional dependencies."""
    
    print("=" * 60)
    print("🔍 OptionsTitan Installation Verification")
    print("=" * 60)
    print()
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'sklearn': 'scikit-learn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'yfinance': 'yfinance',
        'ta': 'ta',
        'optuna': 'optuna',
    }
    
    gui_packages = {
        'PySide6': 'PySide6',
        'tkinter': 'tkinter (built-in)',
    }
    
    optional_packages = {
        'dotenv': 'python-dotenv',
        'shap': 'shap',
    }
    
    ai_packages = {
        'llama_api_client': 'llama-api-client',
    }
    
    # Check required packages
    print("📦 CORE DEPENDENCIES (Required)")
    print("-" * 60)
    all_required_ok = True
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"✅ {package:30} - Installed")
        except ImportError:
            print(f"❌ {package:30} - MISSING")
            all_required_ok = False
    print()
    
    # Check GUI packages
    print("🎨 GUI DEPENDENCIES")
    print("-" * 60)
    for module, package in gui_packages.items():
        try:
            __import__(module)
            print(f"✅ {package:30} - Installed")
        except ImportError:
            if module == 'tkinter':
                print(f"⚠️  {package:30} - Not available (install python3-tk)")
            else:
                print(f"⚠️  {package:30} - Not installed (Qt GUI unavailable)")
    print()
    
    # Check optional packages
    print("⭐ OPTIONAL ENHANCEMENTS")
    print("-" * 60)
    for module, package in optional_packages.items():
        try:
            __import__(module)
            print(f"✅ {package:30} - Installed")
        except ImportError:
            print(f"ℹ️  {package:30} - Not installed (optional)")
    print()
    
    # Check AI packages
    print("🤖 AI ENHANCEMENTS (Optional)")
    print("-" * 60)
    for module, package in ai_packages.items():
        try:
            __import__(module)
            print(f"✅ {package:30} - Installed")
        except ImportError:
            print(f"ℹ️  {package:30} - Not installed (optional)")
    print()
    
    # Check Python version
    print("🐍 PYTHON VERSION")
    print("-" * 60)
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.7+")
        all_required_ok = False
    print()
    
    # Final verdict
    print("=" * 60)
    if all_required_ok:
        print("🎉 SUCCESS! All core dependencies are installed.")
        print()
        print("✅ You can now run:")
        print("   - python options_gui_qt.py    (Modern Qt GUI)")
        print("   - python options_gui.py       (Classic Tkinter GUI)")
        print("   - python main.py              (Train AI models)")
        print()
        print("📖 Next steps:")
        print("   1. Read GETTING_STARTED.md for setup guide")
        print("   2. Optional: Enable LLAMA AI (docs/llama/LLAMA_QUICKSTART.md)")
        print("   3. Launch the GUI and analyze your first strategy!")
    else:
        print("❌ MISSING REQUIRED DEPENDENCIES")
        print()
        print("🔧 To fix, run:")
        print("   pip install -r requirements.txt")
        print()
        print("📖 For help, see:")
        print("   - GETTING_STARTED.md")
        print("   - docs/TROUBLESHOOTING.md")
    print("=" * 60)
    print()
    
    return all_required_ok

if __name__ == "__main__":
    try:
        success = check_dependencies()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        sys.exit(1)
