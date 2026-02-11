"""
OptionsTitan Qt Stylesheets
DPI-aware dark theme for responsive displays
"""

from .ui_utils import (
    get_dpi_scale_factor,
    get_responsive_font_size,
    get_responsive_spacing,
    get_screen_size,
)


def _scale(px: int, dpi_scale: float) -> int:
    """Scale pixel value by DPI."""
    return max(1, int(px * dpi_scale))


def get_dark_theme_stylesheet(dpi_scale: float = None) -> str:
    """
    Return DPI-scaled dark theme stylesheet.
    If dpi_scale is None, uses get_dpi_scale_factor().
    """
    scale = dpi_scale if dpi_scale is not None else get_dpi_scale_factor()
    width, height = get_screen_size()

    # Font sizes (px) - scale by DPI and screen
    fs_body = get_responsive_font_size('body')
    fs_title = get_responsive_font_size('title')
    fs_subtitle = get_responsive_font_size('subheading')
    fs_heading = get_responsive_font_size('heading')
    fs_input = get_responsive_font_size('body') + 1
    fs_small = get_responsive_font_size('small')
    fs_button = max(11, fs_body)

    # Spacing
    base_spacing = get_responsive_spacing()
    pad_sm = _scale(8, scale)
    pad_md = _scale(12, scale)
    pad_lg = _scale(16, scale)
    radius = _scale(6, scale)
    radius_lg = _scale(8, scale)

    # Compact mode adjustments
    if width < 1440:
        fs_title = int(fs_title * 0.85)
        pad_lg = _scale(12, scale)

    return f"""
        QMainWindow {{ background-color: #1e1e1e; }}

        QWidget {{
            background-color: #1e1e1e;
            color: #ffffff;
            font-family: "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
            font-size: {fs_body}px;
        }}

        QLabel {{
            color: #ffffff;
            background-color: transparent;
            font-size: {fs_body}px;
        }}

        QLabel[dashboard_title="true"] {{
            font-size: {fs_title}px;
            font-weight: 700;
            color: #0078d4;
            padding: 0;
            margin-bottom: {pad_sm}px;
        }}

        QLabel[dashboard_subtitle="true"] {{
            font-size: {fs_subtitle}px;
            font-weight: 400;
            color: #b0b0b0;
            padding: 0;
        }}

        QLabel[section_header="true"] {{
            font-size: {fs_heading}px;
            font-weight: 600;
            color: #ffffff;
            padding: 0 0 {pad_sm}px 0;
        }}

        QLabel[input_label="true"] {{
            font-size: {fs_body}px;
            font-weight: 500;
            color: #b0b0b0;
            padding: 0;
            margin-bottom: {pad_sm}px;
        }}

        QLabel[heading="true"] {{
            font-size: {int(fs_heading * 1.2)}px;
            font-weight: bold;
            color: #0078d4;
            padding: {pad_sm}px 0;
        }}

        QLabel[subheading="true"] {{
            font-size: {fs_subtitle}px;
            font-weight: bold;
            color: #ffffff;
            padding: {pad_sm}px 0;
        }}

        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: {radius_lg}px;
            padding: {pad_md}px {pad_lg}px;
            color: #ffffff;
            font-size: {fs_input}px;
            min-height: {_scale(32, scale)}px;
        }}

        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border: 2px solid #0078d4;
        }}

        QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
            border: 2px solid #505050;
        }}

        QLineEdit[valid="true"] {{ border: 2px solid #00d47c; }}
        QLineEdit[valid="false"] {{ border: 2px solid #ff4444; }}

        QComboBox::drop-down {{ border: none; width: {_scale(28, scale)}px; }}

        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 6px solid #ffffff;
            margin-right: {pad_sm}px;
        }}

        QComboBox QAbstractItemView {{
            background-color: #2d2d2d;
            border: 2px solid #0078d4;
            selection-background-color: #0078d4;
            selection-color: #ffffff;
            outline: none;
        }}

        QPushButton {{
            background-color: #0078d4;
            color: #ffffff;
            border: none;
            border-radius: {radius}px;
            padding: {pad_sm}px {pad_lg}px;
            font-size: {fs_button}px;
            font-weight: bold;
            min-height: {_scale(32, scale)}px;
        }}

        QPushButton:hover {{ background-color: #005a9e; }}
        QPushButton:pressed {{ background-color: #004578; }}
        QPushButton:disabled {{ background-color: #3d3d3d; color: #666666; }}

        QPushButton[secondary="true"] {{
            background-color: #3d3d3d;
            color: #ffffff;
        }}
        QPushButton[secondary="true"]:hover {{ background-color: #505050; }}

        QPushButton[success="true"] {{ background-color: #00d47c; color: #000000; }}
        QPushButton[success="true"]:hover {{ background-color: #00b069; }}

        QPushButton[danger="true"] {{ background-color: #ff4444; color: #ffffff; }}
        QPushButton[danger="true"]:hover {{ background-color: #dd3333; }}

        QPushButton[chip="true"] {{
            background-color: #2d2d2d;
            color: #b0b0b0;
            font-size: {fs_small}px;
        }}
        QPushButton[chip="true"]:hover {{
            background-color: #3d3d3d;
            color: #ffffff;
            border: 1px solid #0078d4;
        }}

        QGroupBox {{
            border: 2px solid #3d3d3d;
            border-radius: {radius_lg}px;
            margin-top: {base_spacing}px;
            padding-top: {base_spacing}px;
            font-weight: bold;
            color: #ffffff;
        }}

        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: {pad_lg}px;
            padding: 0 {pad_sm}px;
            background-color: #1e1e1e;
            color: #0078d4;
        }}

        QTabWidget::pane {{
            border: 2px solid #3d3d3d;
            border-radius: {radius_lg}px;
            background-color: #1e1e1e;
            padding: {pad_sm}px;
        }}

        QTabBar::tab {{
            background-color: #2d2d2d;
            color: #b0b0b0;
            border: 2px solid #3d3d3d;
            border-bottom: none;
            border-top-left-radius: {radius_lg}px;
            border-top-right-radius: {radius_lg}px;
            padding: {pad_md}px {_scale(20, scale)}px;
            margin-right: 3px;
            min-width: {_scale(90, scale)}px;
            font-size: {fs_input}px;
            font-weight: 500;
        }}

        QTabBar::tab:selected {{
            background-color: #1e1e1e;
            color: #0078d4;
            border-bottom: 2px solid #1e1e1e;
            font-weight: 600;
        }}

        QTabBar::tab:hover:!selected {{
            background-color: #3d3d3d;
            color: #ffffff;
        }}

        QTextEdit, QPlainTextEdit {{
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: {radius_lg}px;
            color: #ffffff;
            padding: {pad_lg}px;
            font-size: {fs_input}px;
            line-height: 1.6;
            selection-background-color: #0078d4;
        }}

        QScrollBar:vertical {{
            background-color: #2d2d2d;
            width: {_scale(12, scale)}px;
            border-radius: 6px;
        }}
        QScrollBar::handle:vertical {{
            background-color: #505050;
            border-radius: 6px;
            min-height: {_scale(24, scale)}px;
        }}
        QScrollBar::handle:vertical:hover {{ background-color: #606060; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

        QScrollBar:horizontal {{
            background-color: #2d2d2d;
            height: {_scale(12, scale)}px;
            border-radius: 6px;
        }}
        QScrollBar::handle:horizontal {{
            background-color: #505050;
            border-radius: 6px;
            min-width: {_scale(24, scale)}px;
        }}
        QScrollBar::handle:horizontal:hover {{ background-color: #606060; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

        QProgressBar {{
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: {radius}px;
            text-align: center;
            color: #ffffff;
            height: {_scale(22, scale)}px;
        }}
        QProgressBar::chunk {{
            background-color: #0078d4;
            border-radius: 4px;
        }}

        QStatusBar {{
            background-color: #2d2d2d;
            color: #b0b0b0;
            border-top: 1px solid #3d3d3d;
            font-size: {fs_small}px;
        }}
        QStatusBar::item {{ border: none; }}
        QStatusBar QLabel {{ font-size: {fs_small}px; }}

        QMenuBar {{
            background-color: #1e1e1e;
            color: #ffffff;
            border-bottom: 1px solid #3d3d3d;
            padding: {pad_sm}px;
            font-size: {fs_body}px;
        }}
        QMenuBar::item {{ background-color: transparent; padding: {pad_sm}px {pad_lg}px; border-radius: 4px; }}
        QMenuBar::item:selected {{ background-color: #0078d4; }}
        QMenuBar::item:pressed {{ background-color: #005a9e; }}

        QMenu {{
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: {radius_lg}px;
            padding: {pad_sm}px;
            font-size: {fs_body}px;
        }}
        QMenu::item {{ padding: {pad_sm}px {_scale(24, scale)}px {pad_sm}px {pad_lg}px; border-radius: 4px; }}
        QMenu::item:selected {{ background-color: #0078d4; }}
        QMenu::separator {{ height: 1px; background-color: #3d3d3d; margin: {pad_sm}px {pad_sm}px; }}

        QToolTip {{
            background-color: #2d2d2d;
            color: #ffffff;
            border: 2px solid #0078d4;
            border-radius: {radius}px;
            padding: {pad_sm}px;
        }}

        QTreeWidget {{
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: {radius}px;
            outline: none;
        }}
        QTreeWidget::item {{ padding: {pad_sm}px; border-radius: 4px; }}
        QTreeWidget::item:selected {{ background-color: #0078d4; }}
        QTreeWidget::item:hover:!selected {{ background-color: #3d3d3d; }}
        QTreeWidget::branch {{ background-color: transparent; }}

        QSplitter::handle {{ background-color: #3d3d3d; }}
        QSplitter::handle:hover {{ background-color: #0078d4; }}

        QMessageBox {{ background-color: #1e1e1e; }}
        QMessageBox QLabel {{ color: #ffffff; }}
    """


def get_strategy_card_style():
    """Return stylesheet for strategy cards (uses responsive fonts via parent)."""
    fs_name = get_responsive_font_size('heading')
    fs_score = get_responsive_font_size('subheading')
    fs_small = get_responsive_font_size('small')
    pad = get_responsive_spacing()
    return f"""
        QFrame#strategyCard {{
            background-color: #2d2d2d;
            border: 2px solid #3d3d3d;
            border-radius: 8px;
            padding: {pad}px;
        }}
        QFrame#strategyCard:hover {{ border: 2px solid #0078d4; }}
        QLabel#strategyName {{ font-size: {fs_name}px; font-weight: bold; color: #0078d4; }}
        QLabel#fitScore {{ font-size: {fs_score}px; font-weight: bold; color: #00d47c; }}
        QLabel#riskLevel {{ font-size: {fs_small}px; padding: 4px 8px; border-radius: 4px; }}
        QLabel#riskLow {{ background-color: #00d47c; color: #000000; }}
        QLabel#riskMedium {{ background-color: #ffaa00; color: #000000; }}
        QLabel#riskHigh {{ background-color: #ff4444; color: #ffffff; }}
    """


def get_status_colors():
    """Return color codes for status indicators."""
    return {
        'success': '#00d47c',
        'warning': '#ffaa00',
        'error': '#ff4444',
        'info': '#0078d4',
        'text': '#ffffff',
        'text_secondary': '#b0b0b0',
    }
