"""
OptionsTitan Qt Stylesheets
Modern dark theme styling for PySide6 GUI
"""

def get_dark_theme_stylesheet():
    """
    Return the complete dark theme stylesheet for the application.
    
    Colors:
    - Background: #1e1e1e
    - Accent: #0078d4 (blue)
    - Success: #00d47c (green)
    - Warning: #ffaa00 (orange)
    - Error: #ff4444 (red)
    - Text: #ffffff
    - Secondary text: #b0b0b0
    - Border: #3d3d3d
    """
    
    return """
        /* Main Application */
        QMainWindow {
            background-color: #1e1e1e;
        }
        
        /* Widget Defaults */
        QWidget {
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            font-size: 13pt;
        }
        
        /* Labels */
        QLabel {
            color: #ffffff;
            background-color: transparent;
            font-size: 13pt;
        }
        
        QLabel[dashboard_title="true"] {
            font-size: 32pt;
            font-weight: 700;
            color: #0078d4;
            padding: 0px;
            margin-bottom: 5px;
        }
        
        QLabel[dashboard_subtitle="true"] {
            font-size: 16pt;
            font-weight: 400;
            color: #b0b0b0;
            padding: 0px;
            margin-bottom: 0px;
        }
        
        QLabel[section_header="true"] {
            font-size: 20pt;
            font-weight: 600;
            color: #ffffff;
            padding: 0px 0px 10px 0px;
        }
        
        QLabel[input_label="true"] {
            font-size: 13pt;
            font-weight: 500;
            color: #b0b0b0;
            padding: 0px;
            margin-bottom: 6px;
        }
        
        QLabel[heading="true"] {
            font-size: 22pt;
            font-weight: bold;
            color: #0078d4;
            padding: 10px 0px;
        }
        
        QLabel[subheading="true"] {
            font-size: 16pt;
            font-weight: bold;
            color: #ffffff;
            padding: 5px 0px;
        }
        
        /* Input Fields */
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: 8px;
            padding: 12px 16px;
            color: #ffffff;
            font-size: 14pt;
            min-height: 35px;
        }
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 2px solid #0078d4;
        }
        
        QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
            border: 2px solid #505050;
        }
        
        QLineEdit[valid="true"] {
            border: 2px solid #00d47c;
        }
        
        QLineEdit[valid="false"] {
            border: 2px solid #ff4444;
        }
        
        /* Combo Box */
        QComboBox::drop-down {
            border: none;
            width: 30px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #ffffff;
            margin-right: 10px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #2d2d2d;
            border: 2px solid #0078d4;
            selection-background-color: #0078d4;
            selection-color: #ffffff;
            outline: none;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #0078d4;
            color: #ffffff;
            border: none;
            border-radius: 6px;
            padding: 10px 20px;
            font-size: 12pt;
            font-weight: bold;
            min-height: 35px;
        }
        
        QPushButton:hover {
            background-color: #005a9e;
        }
        
        QPushButton:pressed {
            background-color: #004578;
        }
        
        QPushButton:disabled {
            background-color: #3d3d3d;
            color: #666666;
        }
        
        QPushButton[secondary="true"] {
            background-color: #3d3d3d;
            color: #ffffff;
        }
        
        QPushButton[secondary="true"]:hover {
            background-color: #505050;
        }
        
        QPushButton[success="true"] {
            background-color: #00d47c;
            color: #000000;
        }
        
        QPushButton[success="true"]:hover {
            background-color: #00b069;
        }
        
        QPushButton[danger="true"] {
            background-color: #ff4444;
            color: #ffffff;
        }
        
        QPushButton[danger="true"]:hover {
            background-color: #dd3333;
        }
        
        QPushButton[chip="true"] {
            background-color: #2d2d2d;
            color: #b0b0b0;
            font-size: 11pt;
        }
        
        QPushButton[chip="true"]:hover {
            background-color: #3d3d3d;
            color: #ffffff;
            border: 1px solid #0078d4;
        }
        
        /* Group Box */
        QGroupBox {
            border: 2px solid #3d3d3d;
            border-radius: 8px;
            margin-top: 15px;
            padding-top: 15px;
            font-weight: bold;
            color: #ffffff;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 15px;
            padding: 0 10px;
            background-color: #1e1e1e;
            color: #0078d4;
        }
        
        /* Tab Widget */
        QTabWidget::pane {
            border: 2px solid #3d3d3d;
            border-radius: 8px;
            background-color: #1e1e1e;
            padding: 5px;
        }
        
        QTabBar::tab {
            background-color: #2d2d2d;
            color: #b0b0b0;
            border: 2px solid #3d3d3d;
            border-bottom: none;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            padding: 12px 24px;
            margin-right: 3px;
            min-width: 120px;
            font-size: 14pt;
            font-weight: 500;
        }
        
        QTabBar::tab:selected {
            background-color: #1e1e1e;
            color: #0078d4;
            border-bottom: 2px solid #1e1e1e;
            font-weight: 600;
        }
        
        QTabBar::tab:hover:!selected {
            background-color: #3d3d3d;
            color: #ffffff;
        }
        
        /* Text Edit */
        QTextEdit, QPlainTextEdit {
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: 8px;
            color: #ffffff;
            padding: 15px;
            font-size: 14pt;
            line-height: 1.8;
            selection-background-color: #0078d4;
        }
        
        /* Scrollbar */
        QScrollBar:vertical {
            background-color: #2d2d2d;
            width: 14px;
            border-radius: 7px;
        }
        
        QScrollBar::handle:vertical {
            background-color: #505050;
            border-radius: 7px;
            min-height: 30px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #606060;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QScrollBar:horizontal {
            background-color: #2d2d2d;
            height: 14px;
            border-radius: 7px;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #505050;
            border-radius: 7px;
            min-width: 30px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #606060;
        }
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        /* Progress Bar */
        QProgressBar {
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: 6px;
            text-align: center;
            color: #ffffff;
            height: 25px;
        }
        
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 4px;
        }
        
        /* Status Bar */
        QStatusBar {
            background-color: #2d2d2d;
            color: #b0b0b0;
            border-top: 1px solid #3d3d3d;
            font-size: 12pt;
        }
        
        QStatusBar::item {
            border: none;
        }
        
        QStatusBar QLabel {
            font-size: 12pt;
        }
        
        /* Menu Bar */
        QMenuBar {
            background-color: #1e1e1e;
            color: #ffffff;
            border-bottom: 1px solid #3d3d3d;
            padding: 5px;
            font-size: 13pt;
        }
        
        QMenuBar::item {
            background-color: transparent;
            padding: 8px 18px;
            border-radius: 4px;
        }
        
        QMenuBar::item:selected {
            background-color: #0078d4;
        }
        
        QMenuBar::item:pressed {
            background-color: #005a9e;
        }
        
        /* Menu */
        QMenu {
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: 8px;
            padding: 5px;
            font-size: 13pt;
        }
        
        QMenu::item {
            padding: 8px 30px 8px 15px;
            border-radius: 4px;
        }
        
        QMenu::item:selected {
            background-color: #0078d4;
        }
        
        QMenu::separator {
            height: 1px;
            background-color: #3d3d3d;
            margin: 5px 10px;
        }
        
        /* Tool Tip */
        QToolTip {
            background-color: #2d2d2d;
            color: #ffffff;
            border: 2px solid #0078d4;
            border-radius: 6px;
            padding: 8px;
        }
        
        /* Tree Widget */
        QTreeWidget {
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: 6px;
            outline: none;
        }
        
        QTreeWidget::item {
            padding: 5px;
            border-radius: 4px;
        }
        
        QTreeWidget::item:selected {
            background-color: #0078d4;
        }
        
        QTreeWidget::item:hover:!selected {
            background-color: #3d3d3d;
        }
        
        QTreeWidget::branch {
            background-color: transparent;
        }
        
        /* Splitter */
        QSplitter::handle {
            background-color: #3d3d3d;
        }
        
        QSplitter::handle:hover {
            background-color: #0078d4;
        }
        
        /* Message Box */
        QMessageBox {
            background-color: #1e1e1e;
        }
        
        QMessageBox QLabel {
            color: #ffffff;
        }
    """


def get_strategy_card_style():
    """Return stylesheet for strategy cards"""
    return """
        QFrame#strategyCard {
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: 8px;
            padding: 15px;
        }
        
        QFrame#strategyCard:hover {
            border: 2px solid #0078d4;
        }
        
        QLabel#strategyName {
            font-size: 16pt;
            font-weight: bold;
            color: #0078d4;
        }
        
        QLabel#fitScore {
            font-size: 14pt;
            font-weight: bold;
            color: #00d47c;
        }
        
        QLabel#riskLevel {
            font-size: 11pt;
            padding: 4px 8px;
            border-radius: 4px;
        }
        
        QLabel#riskLow {
            background-color: #00d47c;
            color: #000000;
        }
        
        QLabel#riskMedium {
            background-color: #ffaa00;
            color: #000000;
        }
        
        QLabel#riskHigh {
            background-color: #ff4444;
            color: #ffffff;
        }
    """


def get_status_colors():
    """Return color codes for status indicators"""
    return {
        'success': '#00d47c',
        'warning': '#ffaa00',
        'error': '#ff4444',
        'info': '#0078d4',
        'text': '#ffffff',
        'text_secondary': '#b0b0b0',
    }
