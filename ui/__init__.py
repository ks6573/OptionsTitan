"""
OptionsTitan UI Module
PySide6 GUI components for the Options Strategy Analyzer
"""

__version__ = "2.0.0"
__author__ = "OptionsTitan"

__all__ = [
    "OptionsTitanMainWindow",
    "InputPanel",
    "ResultsWidget",
    "AnalysisWorker",
]


def __getattr__(name):
    """Lazy-load Qt-dependent modules so tests can import ui.analyzer without PySide6."""
    if name == "OptionsTitanMainWindow":
        from .main_window import OptionsTitanMainWindow
        return OptionsTitanMainWindow
    if name == "InputPanel":
        from .input_panel import InputPanel
        return InputPanel
    if name == "ResultsWidget":
        from .results_widget import ResultsWidget
        return ResultsWidget
    if name == "AnalysisWorker":
        from .workers import AnalysisWorker
        return AnalysisWorker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
