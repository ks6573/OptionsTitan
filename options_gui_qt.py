#!/usr/bin/env python3
"""
OptionsTitan - PySide6 GUI
AI-powered options strategy analysis dashboard
"""

import sys
import os

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    env_file = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

from ui.main_window import OptionsTitanMainWindow
from ui.analyzer import OptionsStrategyAnalyzer, DEPENDENCIES_AVAILABLE


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("OptionsTitan")
    app.setOrganizationName("OptionsTitan")
    app.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    app.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    analyzer = OptionsStrategyAnalyzer() if DEPENDENCIES_AVAILABLE else None
    llama_enabled = analyzer.llama_enhancer.enabled if analyzer else False

    window = OptionsTitanMainWindow(analyzer=analyzer, llama_enabled=llama_enabled)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
