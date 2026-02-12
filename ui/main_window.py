"""
OptionsTitan Main Window
Main application window with menu bar, status bar, and UI components
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QProgressBar, QLabel,
    QMessageBox, QFileDialog, QSplitter
)
from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QAction, QIcon
import sys
import os

from .input_panel import InputPanel
from .results_widget import ResultsWidget
from .kpi_cards import KPIStrip
from .workers import AnalysisWorker
from .styles import get_dark_theme_stylesheet, get_status_colors
from .ui_utils import (
    get_screen_size,
    get_sidebar_width,
    get_responsive_spacing,
)


class OptionsTitanMainWindow(QMainWindow):
    """
    Main application window for OptionsTitan Strategy Analyzer.
    Integrates all UI components and manages application state.
    """
    
    def __init__(self, analyzer=None, llama_enabled=False):
        super().__init__()
        
        self.analyzer = analyzer
        self.llama_enabled = llama_enabled
        self.current_worker = None
        
        self.setup_ui()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.apply_theme()
        
        # Show initial status
        self.update_status("Ready to analyze options strategies")
    
    def setup_ui(self):
        """Create the main dashboard layout"""
        title_suffix = " (LLAMA AI Enhanced)" if self.llama_enabled else ""
        self.setWindowTitle(f"OptionsTitan - AI Options Strategy Dashboard{title_suffix}")

        # Responsive window sizing
        width, height = get_screen_size()
        win_w = max(1100, min(1800, int(width * 0.92)))
        win_h = max(650, min(1200, int(height * 0.9)))
        self.resize(win_w, win_h)
        self.setMinimumSize(1100, 650)
        self.move(max(0, (width - win_w) // 2), max(0, (height - win_h) // 2))
        self.showMaximized()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left sidebar - responsive width
        sidebar_w = get_sidebar_width()
        margins = max(16, min(28, int(width * 0.012)))
        spacing = get_responsive_spacing()

        left_sidebar = QWidget()
        left_sidebar.setProperty("sidebar", True)
        left_sidebar.setFixedWidth(sidebar_w)
        left_sidebar_layout = QVBoxLayout(left_sidebar)
        left_sidebar_layout.setContentsMargins(margins, margins, margins, margins)
        left_sidebar_layout.setSpacing(spacing)
        
        # Dashboard title
        title_label = QLabel("🚀 OptionsTitan")
        title_label.setProperty("dashboard_title", True)
        title_label.setAlignment(Qt.AlignLeft)
        left_sidebar_layout.addWidget(title_label)
        
        # Subtitle
        subtitle_label = QLabel("Strategy Analyzer Dashboard")
        subtitle_label.setProperty("dashboard_subtitle", True)
        subtitle_label.setAlignment(Qt.AlignLeft)
        left_sidebar_layout.addWidget(subtitle_label)
        
        left_sidebar_layout.addSpacing(max(6, spacing // 2))
        
        # Input panel (now compact for sidebar)
        self.input_panel = InputPanel()
        self.input_panel.analyze_requested.connect(self.on_analyze_requested)
        self.input_panel.clear_requested.connect(self.on_clear_requested)
        self.input_panel.validation_failed.connect(self.on_validation_failed)
        left_sidebar_layout.addWidget(self.input_panel)
        
        # Spacer to push everything to top
        left_sidebar_layout.addStretch()
        
        main_layout.addWidget(left_sidebar)
        
        # Right panel - Results Dashboard
        results_container = QWidget()
        results_container_layout = QVBoxLayout(results_container)
        results_container_layout.setContentsMargins(margins, margins, margins, margins)
        results_container_layout.setSpacing(spacing)
        
        # Results header
        header_layout = QHBoxLayout()
        results_header = QLabel("Analysis Results")
        results_header.setProperty("section_header", True)
        header_layout.addWidget(results_header)
        header_layout.addStretch()

        from PySide6.QtWidgets import QPushButton
        export_h = max(40, min(52, int(height * 0.048)))
        for fmt, label in [('txt', 'Export TXT'), ('html', 'Export HTML')]:
            btn = QPushButton(label)
            btn.setProperty("secondary", True)
            btn.setMinimumHeight(export_h)
            btn.clicked.connect(lambda checked, f=fmt: self.on_export_requested(f))
            header_layout.addWidget(btn)

        results_container_layout.addLayout(header_layout)

        self.kpi_strip = KPIStrip()
        results_container_layout.addWidget(self.kpi_strip)

        self.results_widget = ResultsWidget()
        results_container_layout.addWidget(self.results_widget)
        
        main_layout.addWidget(results_container, 1)
    
    def setup_menu_bar(self):
        """Create the menu bar"""
        menu_bar = self.menuBar()
        
        # File menu
        file_menu = menu_bar.addMenu("&File")
        
        export_action = QAction("&Export Results...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.setStatusTip("Export analysis results")
        export_action.triggered.connect(lambda: self.on_export_requested('txt'))
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Settings menu
        settings_menu = menu_bar.addMenu("&Settings")
        
        api_key_action = QAction("Configure &LLAMA API Key...", self)
        api_key_action.setStatusTip("Set or update LLAMA API key")
        api_key_action.triggered.connect(self.show_api_key_dialog)
        settings_menu.addAction(api_key_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        docs_action = QAction("&Documentation", self)
        docs_action.setShortcut("F1")
        docs_action.setStatusTip("View documentation")
        docs_action.triggered.connect(self.show_documentation)
        help_menu.addAction(docs_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("&About OptionsTitan", self)
        about_action.setStatusTip("About this application")
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
    
    def setup_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, 1)
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # LLAMA status indicator
        colors = get_status_colors()
        llama_status = "✨ LLAMA AI Active" if self.llama_enabled else "ℹ️ LLAMA AI Disabled"
        llama_color = colors['success'] if self.llama_enabled else colors['text_secondary']
        
        self.llama_label = QLabel(llama_status)
        self.llama_label.setStyleSheet(f"color: {llama_color}; padding: 0 10px;")
        self.status_bar.addPermanentWidget(self.llama_label)
    
    def apply_theme(self):
        """Apply DPI-aware dark theme stylesheet"""
        from .ui_utils import get_dpi_scale_factor
        scale = get_dpi_scale_factor()
        stylesheet = get_dark_theme_stylesheet(scale)
        self.setStyleSheet(stylesheet)
    
    @Slot(str, str)
    def on_validation_failed(self, symbol, error_msg):
        """Handle ticker validation failure (invalid symbol, no data)."""
        QMessageBox.warning(
            self,
            "Invalid Symbol",
            f"Cannot analyze {symbol or 'this symbol'}:\n\n{error_msg}\n\n"
            "Please enter a valid stock symbol with available market data.",
        )
        self.update_status("Validation failed")

    @Slot(str, float, float, float, float)
    def on_analyze_requested(self, symbol, liquidity, max_risk, target_profit, max_loss):
        """Handle analysis request from input panel"""
        
        if not self.analyzer:
            QMessageBox.warning(
                self,
                "Dependencies Missing",
                "Required packages not installed. Please run:\npip install -r requirements.txt"
            )
            return
        
        # Disable input during analysis
        self.input_panel.set_enabled(False)
        
        # Show progress
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.show()
        self.update_status("Starting analysis...")
        
        # Store user params for results display
        self.results_widget.set_user_params(liquidity, max_risk, target_profit, max_loss)
        
        # Create and start worker thread
        self.current_worker = AnalysisWorker(
            self.analyzer,
            symbol,
            liquidity,
            max_risk,
            target_profit,
            max_loss
        )
        
        # Connect worker signals
        self.current_worker.progress_update.connect(self.update_status)
        self.current_worker.analysis_complete.connect(self.on_analysis_complete)
        self.current_worker.analysis_error.connect(self.on_analysis_error)
        self.current_worker.finished.connect(self.on_analysis_finished)
        
        # Start analysis
        self.current_worker.start()
    
    @Slot()
    def on_clear_requested(self):
        """Handle clear results request"""
        self.kpi_strip.clear_data()
        self.results_widget.clear_results()
        self.update_status("Results cleared - ready for new analysis")
    
    @Slot(dict, list)
    def on_analysis_complete(self, stock_data, strategies):
        """Handle successful analysis completion"""
        self.kpi_strip.set_data(stock_data, strategies)
        self.results_widget.display_results(stock_data, strategies)
        has_coverage = self.input_panel.has_dataset_coverage()
        if has_coverage:
            self.update_status(f"Analysis complete - {len(strategies)} strategies found (options history available)")
        else:
            self.update_status(f"Analysis complete - {len(strategies)} strategies found (underlying-only mode)")
    
    @Slot(str)
    def on_analysis_error(self, error_msg):
        """Handle analysis error"""
        QMessageBox.critical(
            self,
            "Analysis Error",
            f"An error occurred during analysis:\n\n{error_msg}\n\n"
            "Please check your inputs and try again."
        )
        self.update_status("Analysis failed")
    
    @Slot()
    def on_analysis_finished(self):
        """Clean up after analysis (success or failure)"""
        # Hide progress bar
        self.progress_bar.hide()
        
        # Re-enable input
        self.input_panel.set_enabled(True)
        
        # Clean up worker
        if self.current_worker:
            self.current_worker.deleteLater()
            self.current_worker = None
    
    @Slot(str)
    def on_export_requested(self, format_type):
        """Handle export request"""
        if not self.results_widget.stock_data or not self.results_widget.strategies:
            QMessageBox.information(
                self,
                "No Results",
                "No analysis results to export. Please run an analysis first."
            )
            return
        
        # Get save file name
        filters = {
            'txt': "Text Files (*.txt)",
            'html': "HTML Files (*.html)",
            'pdf': "PDF Files (*.pdf)"
        }
        
        file_filter = filters.get(format_type, "Text Files (*.txt)")
        
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Results",
            f"optionstitan_analysis.{format_type}",
            file_filter
        )
        
        if file_name:
            try:
                self.export_results(file_name, format_type)
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Results exported to:\n{file_name}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export results:\n{str(e)}"
                )
    
    def export_results(self, file_name, format_type):
        """Export results to file"""
        if format_type == 'txt':
            # Get text from overview tab
            text = self.results_widget.overview_text.toPlainText()
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(text)
        
        elif format_type == 'html':
            # Get HTML from overview tab
            html = self.results_widget.overview_text.toHtml()
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(html)
        
        # PDF export would require additional library (reportlab or similar)
    
    def show_api_key_dialog(self):
        """Show dialog for configuring LLAMA API key"""
        QMessageBox.information(
            self,
            "Configure API Key",
            "To configure your LLAMA API key:\n\n"
            "1. Create a .env file in the project root\n"
            "2. Add the line: LLAMA_API_KEY=your_key_here\n"
            "3. Restart the application\n\n"
            "See LLAMA_AI_SETUP.md for detailed instructions."
        )
    
    def show_documentation(self):
        """Show documentation"""
        QMessageBox.information(
            self,
            "Documentation",
            "Documentation is available in the docs/ folder:\n\n"
            "• docs/gui/GUI_GUIDE.md - Complete GUI guide\n"
            "• docs/llama/LLAMA_AI_SETUP.md - AI setup guide\n"
            "• readme.md - Project overview\n\n"
            "Visit the project repository for online documentation."
        )
    
    def show_about_dialog(self):
        """Show about dialog"""
        about_text = """
<h2>OptionsTitan Strategy Analyzer</h2>
<p><b>Version:</b> 2.0.0 (PySide6)</p>
<p><b>Description:</b> AI-powered options strategy analysis tool</p>

<p>Features:</p>
<ul>
  <li>5 ML models for strategy ranking</li>
  <li>Institutional-grade risk management</li>
  <li>Real-time market data analysis</li>
  <li>Meta LLAMA AI integration (optional)</li>
  <li>Beautiful, modern UI</li>
</ul>

<p><b>⚠️ Disclaimer:</b> Educational purposes only. Options trading involves 
substantial risk. Always consult a financial advisor.</p>

<p><b>Technology:</b> Python, PySide6, yfinance, pandas, numpy</p>
"""
        
        QMessageBox.about(self, "About OptionsTitan", about_text)
    
    def update_status(self, message):
        """Update status bar message"""
        self.status_label.setText(message)
        # Status message will automatically clear after 5 seconds
        if not message.startswith("Analysis") and not message == "Ready":
            QTimer.singleShot(5000, lambda: self.status_label.setText("Ready"))
    
    def closeEvent(self, event):
        """Handle window close event"""
        # Stop any running analysis
        if self.current_worker and self.current_worker.isRunning():
            reply = QMessageBox.question(
                self,
                "Analysis in Progress",
                "Analysis is currently running. Are you sure you want to quit?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.current_worker.terminate()
                self.current_worker.wait()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
