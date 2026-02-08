"""
OptionsTitan UI Module
PySide6 GUI components for the Options Strategy Analyzer
"""

__version__ = "2.0.0"
__author__ = "OptionsTitan"

# Make key classes available at package level
from .main_window import OptionsTitanMainWindow
from .input_panel import InputPanel
from .results_widget import ResultsWidget
from .workers import AnalysisWorker

__all__ = [
    'OptionsTitanMainWindow',
    'InputPanel',
    'ResultsWidget',
    'AnalysisWorker',
]
